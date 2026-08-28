#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-sample Single-Layer Spatial-State Rescue Scan (Direction-only, No Gradients)
===============================================================================

Goal
----
For every sample, scan EVERY layer independently and ask:

    "If the spatial relation STATE at this layer is abnormally weak compared
     with generation-correct TRAIN trajectories, does a minimal Direction-only
     correction at THIS ONE layer rescue the final model.generate() answer?"

This is designed to test sample-specific causal rescue points/windows.

No gradients are used.

State diagnostic
----------------
At a chosen stage (default: pre_attn), define the image-grounded relation state

    q_l = (h_sub - h_ref)^img_l - (h_sub - h_ref)^noimg_l

TRAIN codebook at layer l:
    center_l
    p_left, p_right, p_above, p_below

where p_r is the normalized centered mean Direction vector for relation r.

For a sample with GT=g:

    score_r = (q_l - center_l) dot p_r
    foil_l  = argmax_{r != g} score_r
    M_l     = score_g - score_foil_l

Generation-correct TRAIN samples with the same GT define a normal trajectory:

    Q10_l,g, Q25_l,g, median_l,g

Default abnormality rule:
    M_l < Q10_l,g

Single-layer minimal repair
---------------------------
For an abnormal layer, keep all other layers untouched.

Let:
    v = p_GT - p_foil

Optionally project v into the learned 2-D spatial subspace:
    d = unit(P_spatial v)

A hidden-state edit delta = alpha d changes the GT-vs-foil margin by:

    delta_margin = alpha * (v dot d)

Therefore the exact minimum one-axis correction to target T is:

    alpha = (T - M_l) / (v dot d)

with T = Q10 by default.

Then apply pair-preserving editing:

    subject tokens   += delta / 2
    reference tokens -= delta / 2

so q_l changes by exactly delta.

Each layer is tested INDEPENDENTLY from the original baseline trajectory:
    baseline -> repair only L0 -> generate
    baseline -> repair only L1 -> generate
    ...
    baseline -> repair only Llast -> generate

This is NOT the previous dynamic multi-layer repair.

Stages
------
pre_attn:
    edit decoder block input before Attention.

post_attn:
    edit self_attn output, which is algebraically equivalent to editing the
    post-Attention residual before MLP.

Default is pre_attn. You can later run post_attn separately.

Primary outputs
---------------
per_sample_layer_rescue.csv
    One row per sample x layer:
      baseline generation
      GT / local foil
      spatial margin M_l
      correct Q10/Q25/median
      abnormal?
      edit norm
      repaired generation
      W2C / C2W / rescue

sample_rescue_summary.csv
    Per sample:
      first abnormal layer
      abnormal layers
      rescue layers
      first rescue layer
      consecutive rescue windows
      whether Direction state repair can rescue this sample at all

layer_rescue_summary.csv
    Per layer:
      abnormal rate
      W2C count
      rescue rate among fresh-baseline-wrong abnormal samples

generation_overall_summary.csv
    How many fresh baseline wrong samples are rescuable at >=1 layer.

Recommended command
-------------------
CUDA_VISIBLE_DEVICES=0 python eval_single_layer_direction_state_rescue_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --train-cache output/qwen7b_per_sample_direction_repair_all_layers_v2/train_all_layer_direction_states.npz \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers all \
  --stage pre_attn \
  --target q10 \
  --sample-group wrong \
  --output-dir output/qwen7b_single_layer_direction_state_rescue_v1 \
  --overwrite

Smoke
-----
CUDA_VISIBLE_DEVICES=0 python eval_single_layer_direction_state_rescue_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --train-cache output/qwen7b_per_sample_direction_repair_all_layers_v2/train_all_layer_direction_states.npz \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 10-20 \
  --stage pre_attn \
  --target q10 \
  --sample-group wrong \
  --max-samples 8 \
  --output-dir output/qwen7b_single_layer_direction_state_rescue_smoke \
  --overwrite

Interpretation
--------------
Strong evidence for sample-specific spatial rescue would look like:

    sample A: rescue L12-L14
    sample B: rescue L18
    sample C: no Direction-state rescue
    sample D: rescue L9-L10

and the rescue layers should preferentially occur where M_l is outside the
generation-correct spatial trajectory.

This remains an oracle mechanistic experiment because GT is known.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as direction


RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-10


# =============================================================================
# CLI / I/O
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
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
        default="all",
        help="'all', comma-separated layer ids, or ranges such as 10-20.",
    )
    p.add_argument(
        "--stage",
        default="pre_attn",
        choices=["pre_attn", "post_attn"],
    )
    p.add_argument(
        "--target",
        default="q10",
        choices=["zero", "q10", "q25", "median"],
        help=(
            "Normal-boundary target. For the intended minimal rescue experiment "
            "use q10."
        ),
    )
    p.add_argument(
        "--sample-group",
        default="wrong",
        choices=["wrong", "correct", "all"],
        help="Selection uses cached generation group from direction-dir.",
    )
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--project-spatial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project GT-vs-foil correction into TRAIN 2-D spatial subspace.",
    )
    p.add_argument(
        "--repair-normal",
        action="store_true",
        help=(
            "Normally only M < target is edited. With this flag every layer is "
            "forced toward the target when M != target. Not recommended."
        ),
    )
    p.add_argument(
        "--max-edit-norm",
        type=float,
        default=0.0,
        help="Optional per-layer edit norm cap; <=0 means no cap.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(vals.mean()) if len(vals) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_words(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_layers(text: str, n_layers: int) -> List[int]:
    if str(text).strip().lower() == "all":
        return list(range(n_layers))

    vals = []
    for piece in parse_words(text):
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            vals.extend(range(a, b + step, step))
        else:
            vals.append(int(piece))

    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No layers selected.")

    bad = [x for x in vals if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(
            f"Invalid layers {bad}; valid range is 0..{n_layers - 1}"
        )
    return vals


# =============================================================================
# Metadata
# =============================================================================

def norm_relation(x: Any) -> str:
    return direction.norm_relation(x)


def load_metadata(direction_dir: Path):
    vec_path = direction_dir / "vectors.npz"
    gen_path = direction_dir / "sample_split_and_generation.csv"

    if not vec_path.exists():
        raise FileNotFoundError(vec_path)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    with np.load(vec_path, allow_pickle=True) as z:
        sids = z["sample_index"].astype(np.int64)
        labels = [norm_relation(x) for x in z["relation"]]

    gt = {
        int(sid): str(labels[i])
        for i, sid in enumerate(sids.tolist())
    }

    split = {}
    generation = {}

    for r in read_csv(gen_path):
        sid = int(r["sample_index"])
        split[sid] = str(r.get("split", "")).strip()

        pred = norm_relation(r.get("generation_pred", ""))
        group = str(r.get("generation_group", "")).strip().lower()

        g = gt.get(sid, "")
        if group not in ("correct", "wrong"):
            if g in REL2ID and pred in REL2ID:
                group = "correct" if pred == g else "wrong"

        generation[sid] = {
            "generation_group": group,
            "generation_pred": pred,
            "generation_text": str(r.get("generation_text", "")),
        }

    return {
        "sids": [int(x) for x in sids.tolist()],
        "gt": gt,
        "split": split,
        "generation": generation,
    }


# =============================================================================
# Model helpers
# =============================================================================

def get_attr_path(obj: Any, path: str):
    cur = obj
    for piece in path.split("."):
        cur = getattr(cur, piece)
    return cur


def resolve_decoder_layers(model):
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
    ]
    for path in candidates:
        try:
            layers = get_attr_path(model, path)
            if len(layers) > 0 and hasattr(layers[0], "self_attn"):
                return layers, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers.")


def first_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            if torch.is_tensor(y):
                return y
    raise RuntimeError(f"No tensor found in {type(x)}")


def replace_first_tensor(output: Any, new_tensor: torch.Tensor):
    if torch.is_tensor(output):
        return new_tensor

    if isinstance(output, tuple):
        vals = list(output)
        for i, x in enumerate(vals):
            if torch.is_tensor(x):
                vals[i] = new_tensor
                return tuple(vals)

    if isinstance(output, list):
        vals = list(output)
        for i, x in enumerate(vals):
            if torch.is_tensor(x):
                vals[i] = new_tensor
                return vals

    raise RuntimeError(f"Cannot replace tensor in {type(output)}")


def pool_positions(x: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    valid = [
        int(p)
        for p in positions
        if 0 <= int(p) < int(x.shape[0])
    ]
    if not valid:
        raise RuntimeError("No valid object token positions.")

    idx = torch.as_tensor(
        valid,
        device=x.device,
        dtype=torch.long,
    )
    return x.index_select(0, idx).mean(dim=0)


def pair_diff(x: torch.Tensor, subj_pos, ref_pos) -> torch.Tensor:
    return (
        pool_positions(x, subj_pos)
        - pool_positions(x, ref_pos)
    )


def build_batch(processor, rec, question, image, device):
    rendered = direction.build_chat_prompt(
        processor,
        question,
        image is not None,
    )
    batch = direction.process_inputs(
        processor,
        rendered,
        image,
        device,
    )

    ids = [
        int(x)
        for x in batch["input_ids"][0].detach().cpu().tolist()
    ]
    subj_pos = direction.locate_phrase_positions(
        processor.tokenizer,
        ids,
        str(rec.subject),
    )
    ref_pos = direction.locate_phrase_positions(
        processor.tokenizer,
        ids,
        str(rec.reference),
    )
    return batch, subj_pos, ref_pos


# =============================================================================
# Generation
# =============================================================================

def parse_generated_relation(text: str) -> Optional[str]:
    s = str(text).strip().lower()

    patterns = [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"),
        ("below", r"\bunder(?:neath)?\b"),
    ]

    hits = []
    for rel, pat in patterns:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))

    if not hits:
        return None

    hits.sort()
    return hits[0][1]


def generate_answer(model, processor, batch, max_new_tokens):
    input_len = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    suffix = generated[0, input_len:]
    text = processor.tokenizer.decode(
        suffix,
        skip_special_tokens=True,
    ).strip()

    pred = parse_generated_relation(text)
    del generated
    return text, pred


# =============================================================================
# Baseline state capture
# =============================================================================

class StagePairCapture:
    def __init__(
        self,
        decoder_layers,
        selected_layers,
        subj_pos,
        ref_pos,
        stage,
    ):
        self.selected_layers = list(map(int, selected_layers))
        self.subj_pos = list(map(int, subj_pos))
        self.ref_pos = list(map(int, ref_pos))
        self.stage = stage
        self.states = {}
        self.handles = []

        for li in self.selected_layers:
            block = decoder_layers[li]

            if stage == "pre_attn":
                self.handles.append(
                    block.register_forward_pre_hook(
                        self._make_pre_hook(li)
                    )
                )
            elif stage == "post_attn":
                if not hasattr(block, "post_attention_layernorm"):
                    raise RuntimeError(
                        f"L{li} has no post_attention_layernorm."
                    )
                self.handles.append(
                    block.post_attention_layernorm.register_forward_pre_hook(
                        self._make_pre_hook(li)
                    )
                )
            else:
                raise ValueError(stage)

    def _make_pre_hook(self, li):
        def hook(_module, args):
            if not args:
                return None

            x = first_tensor(args)
            if x.ndim != 3:
                return None

            seq = x[0]
            if int(seq.shape[0]) <= max(
                self.subj_pos + self.ref_pos
            ):
                return None

            self.states[li] = (
                pair_diff(
                    seq,
                    self.subj_pos,
                    self.ref_pos,
                )
                .detach().float().cpu().numpy()
                .astype(np.float32)
            )
            return None

        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def capture_stage_states(
    *,
    model,
    processor,
    decoder_layers,
    selected_layers,
    rec,
    image,
    device,
    prompt_template,
    stage,
):
    question = prompt_template.format(
        subject=rec.subject,
        reference=rec.reference,
    )

    batch, sp, rp = build_batch(
        processor,
        rec,
        question,
        image,
        device,
    )

    with StagePairCapture(
        decoder_layers,
        selected_layers,
        sp,
        rp,
        stage,
    ) as cap:
        with torch.inference_mode():
            _ = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

    missing = [
        li for li in selected_layers
        if li not in cap.states
    ]
    if missing:
        raise RuntimeError(
            f"Missing {stage} captures for layers {missing}"
        )

    states = dict(cap.states)
    del batch
    return states


# =============================================================================
# TRAIN codebooks and normal margin trajectory
# =============================================================================

def unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if n <= EPS:
        return np.zeros_like(x, dtype=np.float32)
    return (x / n).astype(np.float32)


def orthonormal_basis(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """
    Return an orthonormal basis for the learned 2-D spatial subspace.

    Some very early layers can be genuinely degenerate: TRAIN relation means
    may not yet separate left/right or above/below.  That is a valid diagnostic
    result, not a reason to abort an all-layer scan.  In that case return an
    empty basis with shape [hidden_dim, 0].  Downstream repair code will fall
    back to the raw GT-vs-foil Direction axis (or skip the edit if even that
    axis is degenerate).
    """
    vecs = [np.asarray(v, dtype=np.float64) for v in vectors]
    A = np.stack(vecs, axis=1)
    u, s, _ = np.linalg.svd(A, full_matrices=False)

    scale = max(float(s.max()) if len(s) else 0.0, 1.0)
    keep = s > 1e-8 * scale

    if not np.any(keep):
        return np.zeros((A.shape[0], 0), dtype=np.float32)

    return u[:, keep].astype(np.float32)


def project_spatial(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    v64 = np.asarray(v, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)

    # Degenerate layer: no reliable TRAIN-derived 2-D spatial subspace exists.
    # Return a zero projection and let the caller explicitly decide whether to
    # fall back to the raw semantic Direction axis.
    if B64.ndim != 2 or B64.shape[1] == 0:
        return np.zeros_like(v64, dtype=np.float32)

    return (B64 @ (B64.T @ v64)).astype(np.float32)


def fit_codebook(X: np.ndarray, labels: np.ndarray):
    center = X.mean(axis=0).astype(np.float32)
    Xc = X - center

    means = {}
    protos = {}

    for rel in RELATIONS:
        mask = labels == rel
        if not np.any(mask):
            raise RuntimeError(
                f"No TRAIN examples for relation={rel}"
            )

        mu = Xc[mask].mean(axis=0).astype(np.float32)
        means[rel] = mu
        protos[rel] = unit(mu)

    basis = orthonormal_basis([
        means["right"] - means["left"],
        means["above"] - means["below"],
    ])

    return {
        "center": center,
        "means": means,
        "protos": protos,
        "basis": basis,
        "basis_rank": int(basis.shape[1]),
    }


def scores_and_margin(q: np.ndarray, cb, gt: str):
    qc = (
        np.asarray(q, dtype=np.float32)
        - np.asarray(cb["center"], dtype=np.float32)
    )

    scores = {
        rel: float(qc @ cb["protos"][rel])
        for rel in RELATIONS
    }

    foil = max(
        [r for r in RELATIONS if r != gt],
        key=lambda r: scores[r],
    )
    margin = float(scores[gt] - scores[foil])

    return scores, foil, margin


def load_codebooks_and_trajectory(
    *,
    cache_path: Path,
    selected_layers: Sequence[int],
    stage: str,
):
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    stage_key = (
        "pre_attn"
        if stage == "pre_attn"
        else "post_attn"
    )

    codebooks = {}
    targets = {}
    target_rows = []

    with np.load(cache_path, allow_pickle=True) as z:
        available = set(
            int(x)
            for x in z["selected_layers"].tolist()
        )
        missing = [
            li for li in selected_layers
            if li not in available
            or f"L{li}_{stage_key}" not in z.files
        ]
        if missing:
            raise RuntimeError(
                f"TRAIN cache is missing {stage} layers {missing}."
            )

        labels = np.asarray(
            [norm_relation(x) for x in z["relation"]],
            dtype=object,
        )
        groups = np.asarray(
            [str(x).strip().lower() for x in z["generation_group"]],
            dtype=object,
        )
        correct_mask = groups == "correct"

        if not np.any(correct_mask):
            raise RuntimeError(
                "TRAIN cache contains no generation-correct samples."
            )

        for li in selected_layers:
            X = np.asarray(
                z[f"L{li}_{stage_key}"],
                dtype=np.float32,
            )

            cb = fit_codebook(X, labels)
            codebooks[li] = cb

            for gt in RELATIONS:
                mask = (
                    correct_mask
                    & (labels == gt)
                )

                margins = []
                for i in np.where(mask)[0]:
                    _, _, m = scores_and_margin(
                        X[i],
                        cb,
                        gt,
                    )
                    margins.append(m)

                vals = np.asarray(margins, dtype=np.float64)
                if len(vals) == 0:
                    raise RuntimeError(
                        f"No generation-correct TRAIN controls "
                        f"for L{li}, GT={gt}."
                    )

                q10 = float(np.quantile(vals, 0.10))
                q25 = float(np.quantile(vals, 0.25))
                med = float(np.median(vals))

                targets[(li, gt)] = {
                    "zero": 0.0,
                    "q10": q10,
                    "q25": q25,
                    "median": med,
                    "n": int(len(vals)),
                }

                target_rows.append({
                    "layer": li,
                    "stage": stage,
                    "gt": gt,
                    "spatial_basis_rank": int(cb.get("basis_rank", cb["basis"].shape[1])),
                    "n_generation_correct_train": int(len(vals)),
                    "q10_margin": q10,
                    "q25_margin": q25,
                    "median_margin": med,
                })

    return codebooks, targets, target_rows


# =============================================================================
# Exact single-layer Direction correction
# =============================================================================

def make_margin_correction(
    *,
    cb,
    gt: str,
    foil: str,
    current_margin: float,
    target_margin: float,
    project_to_spatial: bool,
    max_edit_norm: float,
):
    """
    score_GT - score_foil = qc dot (p_GT - p_foil).

    Choose delta along a spatial semantic axis d. Then:
        new_margin = current_margin + delta dot (p_GT-p_foil).

    alpha is solved exactly, so there is no hidden overshoot from prototype
    non-orthogonality.
    """
    v = (
        np.asarray(cb["protos"][gt], dtype=np.float32)
        - np.asarray(cb["protos"][foil], dtype=np.float32)
    )

    used_spatial_projection = False
    used_raw_fallback = False

    if project_to_spatial and int(cb.get("basis_rank", cb["basis"].shape[1])) > 0:
        axis_raw = project_spatial(
            v,
            cb["basis"],
        )
        # A particular GT-vs-foil contrast can still be nearly orthogonal to
        # the learned spatial basis even when the basis itself has rank > 0.
        if float(np.linalg.norm(axis_raw)) > 1e-8:
            used_spatial_projection = True
        else:
            axis_raw = v
            used_raw_fallback = True
    else:
        axis_raw = v
        if project_to_spatial:
            used_raw_fallback = True

    d = unit(axis_raw)

    denom = float(
        np.asarray(v, dtype=np.float64)
        @ np.asarray(d, dtype=np.float64)
    )

    if abs(denom) <= 1e-8:
        return (
            np.zeros_like(v, dtype=np.float32),
            0.0,
            0.0,
            denom,
            used_spatial_projection,
            used_raw_fallback,
        )

    deficit = float(target_margin - current_margin)

    # Intended experiment only boosts weak margins. If repair-normal is used
    # the caller may pass a negative deficit.
    alpha = deficit / denom

    delta = (
        float(alpha) * d
    ).astype(np.float32)

    raw_norm = float(np.linalg.norm(delta))

    if max_edit_norm > 0 and raw_norm > max_edit_norm:
        delta = (
            delta
            * (float(max_edit_norm) / raw_norm)
        ).astype(np.float32)

    achieved_margin_change = float(
        np.asarray(delta, dtype=np.float64)
        @ np.asarray(v, dtype=np.float64)
    )

    return (
        delta,
        achieved_margin_change,
        raw_norm,
        denom,
        used_spatial_projection,
        used_raw_fallback,
    )


class SingleLayerStateRepair:
    def __init__(
        self,
        *,
        block,
        stage,
        subj_pos,
        ref_pos,
        delta,
    ):
        self.block = block
        self.stage = stage
        self.subj_pos = list(map(int, subj_pos))
        self.ref_pos = list(map(int, ref_pos))
        self.delta = np.asarray(delta, dtype=np.float32)
        self.applied = False
        self.handle = None

    def _edit_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.applied:
            return x

        if x.ndim != 3:
            return x

        if int(x.shape[1]) <= max(
            self.subj_pos + self.ref_pos
        ):
            return x

        y = x.clone()

        if float(np.linalg.norm(self.delta)) > EPS:
            half = 0.5 * torch.from_numpy(
                self.delta
            ).to(
                device=y.device,
                dtype=y.dtype,
            )

            y[0, self.subj_pos, :] = (
                y[0, self.subj_pos, :]
                + half
            )
            y[0, self.ref_pos, :] = (
                y[0, self.ref_pos, :]
                - half
            )

        self.applied = True
        return y

    def _pre_attn_hook(self, _module, args):
        if self.applied or not args:
            return None

        vals = list(args)
        x = first_tensor(vals)

        # Decoder blocks have hidden_states as their first positional tensor.
        idx = None
        for i, item in enumerate(vals):
            if torch.is_tensor(item):
                idx = i
                break

        if idx is None:
            return None

        y = self._edit_tensor(x)

        if y is x:
            return None

        vals[idx] = y
        return tuple(vals)

    def _post_attn_hook(self, _module, _args, output):
        if self.applied:
            return output

        x = first_tensor(output)
        y = self._edit_tensor(x)

        if y is x:
            return output

        return replace_first_tensor(
            output,
            y,
        )

    def __enter__(self):
        if self.stage == "pre_attn":
            self.handle = self.block.register_forward_pre_hook(
                self._pre_attn_hook
            )
        elif self.stage == "post_attn":
            self.handle = self.block.self_attn.register_forward_hook(
                self._post_attn_hook
            )
        else:
            raise ValueError(self.stage)

        return self

    def __exit__(self, *_args):
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# =============================================================================
# Eval sample selection
# =============================================================================

def select_eval_sids(
    metadata,
    records,
    split,
    sample_group,
    max_samples,
    seed,
):
    sids = []

    for sid in metadata["sids"]:
        if (
            split != "all"
            and metadata["split"].get(sid, "") != split
        ):
            continue

        if sid not in records:
            continue

        if metadata["gt"].get(sid, "") not in REL2ID:
            continue

        group = metadata["generation"].get(
            sid, {}
        ).get("generation_group", "")

        if (
            sample_group != "all"
            and group != sample_group
        ):
            continue

        sids.append(sid)

    if max_samples is not None and len(sids) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(sids)
        sids = sids[:max_samples]

    return sorted(sids)


# =============================================================================
# Rescue windows
# =============================================================================

def consecutive_windows(layers: Sequence[int]) -> str:
    xs = sorted(set(int(x) for x in layers))
    if not xs:
        return ""

    groups = []
    start = prev = xs[0]

    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue

        groups.append((start, prev))
        start = prev = x

    groups.append((start, prev))

    parts = []
    for a, b in groups:
        if a == b:
            parts.append(f"L{a}")
        else:
            parts.append(f"L{a}-L{b}")

    return ",".join(parts)


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    *,
    model,
    processor,
    decoder_layers,
    selected_layers,
    stage,
    target_name,
    codebooks,
    targets,
    records,
    metadata,
    eval_sids,
    device,
    prompt_template,
    max_new_tokens,
    project_spatial,
    repair_normal,
    max_edit_norm,
    out_dir,
    save_every,
):
    layer_rows = []
    sample_rows = []
    errors = []

    for sample_i, sid in enumerate(
        tqdm(
            eval_sids,
            desc="single-layer Direction rescue scan",
        ),
        1,
    ):
        rec = records[sid]
        image = None

        try:
            gt = metadata["gt"][sid]

            image = Image.open(
                rec.image_path
            ).convert("RGB")

            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )

            real_batch, sp, rp = build_batch(
                processor,
                rec,
                question,
                image,
                device,
            )

            # Baseline generation.
            baseline_text, baseline_pred = generate_answer(
                model,
                processor,
                real_batch,
                max_new_tokens,
            )
            baseline_correct = int(
                baseline_pred == gt
            )

            # Baseline spatial trajectory.
            real_states = capture_stage_states(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                rec=rec,
                image=image,
                device=device,
                prompt_template=prompt_template,
                stage=stage,
            )
            noimg_states = capture_stage_states(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                rec=rec,
                image=None,
                device=device,
                prompt_template=prompt_template,
                stage=stage,
            )

            abnormal_layers = []
            repaired_layers = []
            rescue_layers = []
            harm_layers = []

            for li in selected_layers:
                q = (
                    real_states[li]
                    - noimg_states[li]
                ).astype(np.float32)

                cb = codebooks[li]
                scores, foil, margin = scores_and_margin(
                    q,
                    cb,
                    gt,
                )

                t = targets[(li, gt)]
                target_margin = float(
                    t[target_name]
                )

                abnormal = int(
                    margin < target_margin
                )

                should_repair = (
                    bool(abnormal)
                    or bool(repair_normal)
                )

                if abnormal:
                    abnormal_layers.append(li)

                # Default: no edit / identical generation.
                edited_text = baseline_text
                edited_pred = baseline_pred
                edited_correct = baseline_correct
                edit_applied = 0
                edit_norm = 0.0
                raw_edit_norm = 0.0
                achieved_change = 0.0
                expected_margin_after = margin
                denom = float("nan")
                used_spatial_projection = False
                used_raw_fallback = False

                if should_repair:
                    (
                        delta,
                        achieved_change,
                        raw_edit_norm,
                        denom,
                        used_spatial_projection,
                        used_raw_fallback,
                    ) = make_margin_correction(
                        cb=cb,
                        gt=gt,
                        foil=foil,
                        current_margin=margin,
                        target_margin=target_margin,
                        project_to_spatial=project_spatial,
                        max_edit_norm=max_edit_norm,
                    )

                    edit_norm = float(
                        np.linalg.norm(delta)
                    )

                    if edit_norm > EPS:
                        edit_applied = 1
                        repaired_layers.append(li)

                        with SingleLayerStateRepair(
                            block=decoder_layers[li],
                            stage=stage,
                            subj_pos=sp,
                            ref_pos=rp,
                            delta=delta,
                        ):
                            edited_text, edited_pred = generate_answer(
                                model,
                                processor,
                                real_batch,
                                max_new_tokens,
                            )

                        edited_correct = int(
                            edited_pred == gt
                        )
                        expected_margin_after = (
                            margin + achieved_change
                        )

                W2C = int(
                    baseline_correct == 0
                    and edited_correct == 1
                )
                C2W = int(
                    baseline_correct == 1
                    and edited_correct == 0
                )

                if W2C:
                    rescue_layers.append(li)

                if C2W:
                    harm_layers.append(li)

                layer_rows.append({
                    "sid": sid,
                    "layer": li,
                    "stage": stage,
                    "target": target_name,

                    "gt": gt,
                    "local_foil": foil,

                    "score_left": scores["left"],
                    "score_right": scores["right"],
                    "score_above": scores["above"],
                    "score_below": scores["below"],

                    "margin_GT_minus_maxNonGT":
                        margin,

                    "correct_q10":
                        float(t["q10"]),
                    "correct_q25":
                        float(t["q25"]),
                    "correct_median":
                        float(t["median"]),
                    "target_margin":
                        target_margin,
                    "n_correct_train_controls":
                        int(t["n"]),

                    "abnormal":
                        abnormal,
                    "edit_applied":
                        edit_applied,

                    "raw_edit_norm":
                        raw_edit_norm,
                    "edit_norm":
                        edit_norm,
                    "margin_axis_denominator":
                        denom,
                    "spatial_basis_rank":
                        int(cb.get("basis_rank", cb["basis"].shape[1])),
                    "used_spatial_projection":
                        int(bool(used_spatial_projection)),
                    "used_raw_direction_fallback":
                        int(bool(used_raw_fallback)),
                    "expected_margin_change":
                        achieved_change,
                    "expected_margin_after":
                        expected_margin_after,

                    "cached_group":
                        metadata["generation"].get(
                            sid, {}
                        ).get("generation_group", ""),
                    "cached_pred":
                        metadata["generation"].get(
                            sid, {}
                        ).get("generation_pred", ""),

                    "baseline_text":
                        baseline_text,
                    "baseline_pred":
                        baseline_pred or "",
                    "baseline_correct":
                        baseline_correct,

                    "edited_text":
                        edited_text,
                    "edited_pred":
                        edited_pred or "",
                    "edited_correct":
                        edited_correct,

                    "W2C": W2C,
                    "C2W": C2W,
                    "rescue":
                        W2C,
                })

            sample_rows.append({
                "sid": sid,
                "gt": gt,

                "cached_group":
                    metadata["generation"].get(
                        sid, {}
                    ).get("generation_group", ""),
                "cached_pred":
                    metadata["generation"].get(
                        sid, {}
                    ).get("generation_pred", ""),

                "baseline_text":
                    baseline_text,
                "baseline_pred":
                    baseline_pred or "",
                "baseline_correct":
                    baseline_correct,

                "n_layers_scanned":
                    len(selected_layers),

                "n_abnormal_layers":
                    len(abnormal_layers),
                "abnormal_layers":
                    ",".join(
                        str(x) for x in abnormal_layers
                    ),
                "first_abnormal_layer":
                    (
                        abnormal_layers[0]
                        if abnormal_layers
                        else ""
                    ),

                "n_repaired_layers":
                    len(repaired_layers),
                "repaired_layers":
                    ",".join(
                        str(x) for x in repaired_layers
                    ),

                "n_rescue_layers":
                    len(rescue_layers),
                "rescue_layers":
                    ",".join(
                        str(x) for x in rescue_layers
                    ),
                "first_rescue_layer":
                    (
                        rescue_layers[0]
                        if rescue_layers
                        else ""
                    ),
                "rescue_windows":
                    consecutive_windows(rescue_layers),
                "rescuable_by_direction_state":
                    int(len(rescue_layers) > 0),

                "n_harm_layers":
                    len(harm_layers),
                "harm_layers":
                    ",".join(
                        str(x) for x in harm_layers
                    ),
                "harm_windows":
                    consecutive_windows(harm_layers),
            })

            del real_batch, real_states, noimg_states

            if (
                save_every > 0
                and sample_i % save_every == 0
            ):
                write_csv(
                    out_dir / "per_sample_layer_rescue.csv",
                    layer_rows,
                )
                write_csv(
                    out_dir / "sample_rescue_summary.csv",
                    sample_rows,
                )

        except Exception as e:
            errors.append({
                "sid": sid,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[ERROR sid={sid}] "
                f"{type(e).__name__}: {e}"
            )

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(
        out_dir / "per_sample_layer_rescue.csv",
        layer_rows,
    )
    write_csv(
        out_dir / "sample_rescue_summary.csv",
        sample_rows,
    )
    write_csv(
        out_dir / "errors.csv",
        errors,
    )

    return layer_rows, sample_rows, errors


# =============================================================================
# Summaries
# =============================================================================

def summarize_layers(layer_rows):
    buckets = defaultdict(list)

    for r in layer_rows:
        buckets[int(r["layer"])].append(r)

    out = []

    for li, rr in sorted(buckets.items()):
        abnormal = [
            r for r in rr
            if int(r["abnormal"]) == 1
        ]

        base_wrong = [
            r for r in rr
            if int(r["baseline_correct"]) == 0
        ]
        base_wrong_abnormal = [
            r for r in abnormal
            if int(r["baseline_correct"]) == 0
        ]

        out.append({
            "layer": li,
            "stage":
                rr[0]["stage"],
            "target":
                rr[0]["target"],
            "n": len(rr),

            "abnormal_rate": safe_frac(
                int(r["abnormal"]) == 1
                for r in rr
            ),

            "mean_margin": safe_mean(
                r["margin_GT_minus_maxNonGT"]
                for r in rr
            ),
            "mean_target_margin": safe_mean(
                r["target_margin"]
                for r in rr
            ),
            "mean_edit_norm_abnormal":
                safe_mean(
                    r["edit_norm"]
                    for r in abnormal
                ),

            "n_fresh_baseline_wrong":
                len(base_wrong),
            "n_fresh_baseline_wrong_abnormal":
                len(base_wrong_abnormal),

            "W2C": int(
                sum(int(r["W2C"]) for r in rr)
            ),
            "rescue_rate_over_baseline_wrong":
                (
                    sum(int(r["W2C"]) for r in rr)
                    / len(base_wrong)
                    if base_wrong else float("nan")
                ),
            "rescue_rate_over_wrong_abnormal":
                (
                    sum(int(r["W2C"]) for r in rr)
                    / len(base_wrong_abnormal)
                    if base_wrong_abnormal
                    else float("nan")
                ),

            "C2W": int(
                sum(int(r["C2W"]) for r in rr)
            ),
        })

    return out


def summarize_overall(sample_rows):
    n = len(sample_rows)

    base_wrong = [
        r for r in sample_rows
        if int(r["baseline_correct"]) == 0
    ]
    base_correct = [
        r for r in sample_rows
        if int(r["baseline_correct"]) == 1
    ]

    rescuable_wrong = [
        r for r in base_wrong
        if int(r["rescuable_by_direction_state"]) == 1
    ]
    harmed_correct = [
        r for r in base_correct
        if int(r["n_harm_layers"]) > 0
    ]

    rows = [{
        "n_samples": n,
        "n_fresh_baseline_wrong":
            len(base_wrong),
        "n_fresh_baseline_correct":
            len(base_correct),

        "n_wrong_rescuable_at_any_layer":
            len(rescuable_wrong),
        "wrong_rescuable_fraction":
            (
                len(rescuable_wrong) / len(base_wrong)
                if base_wrong else float("nan")
            ),

        "mean_rescue_layers_among_rescuable":
            safe_mean(
                r["n_rescue_layers"]
                for r in rescuable_wrong
            ),

        "n_correct_harmed_at_any_layer":
            len(harmed_correct),
        "correct_harmed_fraction":
            (
                len(harmed_correct) / len(base_correct)
                if base_correct else float("nan")
            ),

        "mean_abnormal_layers":
            safe_mean(
                r["n_abnormal_layers"]
                for r in sample_rows
            ),
    }]

    return rows


def print_layer_summary(rows):
    print("\n" + "=" * 150)
    print("SINGLE-LAYER DIRECTION STATE RESCUE")
    print("=" * 150)
    print(
        "layer | abnormal | margin/target | editNorm | "
        "freshWrong abnormalWrong | W2C / wrong / wrongAbnormal | C2W"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} | "
            f"{float(r['abnormal_rate']):.3f} | "
            f"{float(r['mean_margin']):+7.3f}/"
            f"{float(r['mean_target_margin']):+7.3f} | "
            f"{float(r['mean_edit_norm_abnormal']):6.3f} | "
            f"{int(r['n_fresh_baseline_wrong']):3d} "
            f"{int(r['n_fresh_baseline_wrong_abnormal']):3d} | "
            f"{int(r['W2C']):3d} / "
            f"{float(r['rescue_rate_over_baseline_wrong']):.3f} / "
            f"{float(r['rescue_rate_over_wrong_abnormal']):.3f} | "
            f"{int(r['C2W']):3d}"
        )


def print_sample_summary(sample_rows):
    print("\n" + "=" * 160)
    print("PER-SAMPLE RESCUE WINDOWS")
    print("=" * 160)
    print(
        "sid GT base | abnormal(first) | rescue(first) windows | harm windows"
    )

    for r in sample_rows:
        print(
            f"{int(r['sid']):4d} "
            f"{str(r['gt']):5s} "
            f"{str(r['baseline_pred']):5s} | "
            f"{int(r['n_abnormal_layers']):2d}"
            f"({str(r['first_abnormal_layer']):>2s}) | "
            f"{int(r['n_rescue_layers']):2d}"
            f"({str(r['first_rescue_layer']):>2s}) "
            f"{str(r['rescue_windows']):20s} | "
            f"{str(r['harm_windows'])}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA requested but unavailable.")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(
        Path(args.direction_dir)
    )

    records_list, _audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    records = {
        int(r.sid): r
        for r in records_list
    }

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    dtype = base.resolve_dtype(spec.dtype_name)

    kw: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(
        f"[model] loading {spec.repo_id} on {args.device}"
    )

    try:
        model = cls.from_pretrained(
            spec.repo_id,
            dtype=dtype,
            **kw,
        )
    except TypeError:
        model = cls.from_pretrained(
            spec.repo_id,
            torch_dtype=dtype,
            **kw,
        )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(
        model,
        processor,
    )

    device = torch.device(args.device)

    decoder_layers, layer_path = resolve_decoder_layers(
        model
    )
    n_layers = len(decoder_layers)

    selected_layers = parse_layers(
        args.layers,
        n_layers,
    )

    print(
        f"[decoder] {layer_path}; "
        f"n_layers={n_layers}; "
        f"selected={selected_layers}"
    )

    codebooks, targets, target_rows = load_codebooks_and_trajectory(
        cache_path=Path(args.train_cache),
        selected_layers=selected_layers,
        stage=args.stage,
    )

    write_csv(
        out_dir / "correct_margin_trajectory.csv",
        target_rows,
    )

    eval_sids = select_eval_sids(
        metadata,
        records,
        args.eval_split,
        args.sample_group,
        args.max_samples,
        args.seed,
    )

    print(
        f"[eval] cached group={args.sample_group}, "
        f"N={len(eval_sids)}, "
        f"stage={args.stage}, "
        f"target={args.target}, "
        f"project_spatial={args.project_spatial}"
    )

    layer_rows, sample_rows, errors = run_experiment(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        selected_layers=selected_layers,
        stage=args.stage,
        target_name=args.target,
        codebooks=codebooks,
        targets=targets,
        records=records,
        metadata=metadata,
        eval_sids=eval_sids,
        device=device,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        project_spatial=args.project_spatial,
        repair_normal=args.repair_normal,
        max_edit_norm=args.max_edit_norm,
        out_dir=out_dir,
        save_every=args.save_every,
    )

    layer_summary = summarize_layers(
        layer_rows
    )
    overall_summary = summarize_overall(
        sample_rows
    )

    write_csv(
        out_dir / "layer_rescue_summary.csv",
        layer_summary,
    )
    write_csv(
        out_dir / "generation_overall_summary.csv",
        overall_summary,
    )

    print_layer_summary(
        layer_summary
    )
    print_sample_summary(
        sample_rows
    )

    print("\n" + "=" * 110)
    print("OVERALL")
    print("=" * 110)
    if overall_summary:
        r = overall_summary[0]
        print(
            f"fresh baseline wrong: "
            f"{int(r['n_fresh_baseline_wrong'])}"
        )
        print(
            f"wrong rescuable at >=1 layer: "
            f"{int(r['n_wrong_rescuable_at_any_layer'])} "
            f"({float(r['wrong_rescuable_fraction']):.3f})"
        )
        print(
            f"mean abnormal layers/sample: "
            f"{float(r['mean_abnormal_layers']):.2f}"
        )
        print(
            f"fresh baseline correct harmed at >=1 layer: "
            f"{int(r['n_correct_harmed_at_any_layer'])}"
        )

    summary = {
        "experiment":
            "single-layer sample-specific Direction state rescue scan",
        "gradient_used": False,
        "gt_oracle": True,
        "stage": args.stage,
        "target": args.target,
        "project_spatial": bool(args.project_spatial),
        "repair_normal": bool(args.repair_normal),
        "selected_layers": selected_layers,
        "sample_group": args.sample_group,
        "n_eval": len(eval_sids),
        "n_errors": len(errors),
        "important_definition":
            "M_l = score_GT(q_l) - max_nonGT score(q_l); "
            "abnormal if M_l is below generation-correct TRAIN target.",
        "intervention":
            "Each layer tested independently. Minimal pair-preserving edit "
            "along TRAIN spatial GT-vs-current-foil Direction axis.",
        "primary_result":
            "Per-sample rescue layers/windows measured with fresh model.generate().",
    }

    (out_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "correct_margin_trajectory.csv",
        "per_sample_layer_rescue.csv",
        "sample_rescue_summary.csv",
        "layer_rescue_summary.csv",
        "generation_overall_summary.csv",
        "errors.csv",
        "summary.json",
    ]:
        print(" ", out_dir / name)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
