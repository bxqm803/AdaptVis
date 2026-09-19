#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure B: Synthetic-400 dual-basis spatial steering -> COCO belief shifts.

This version follows the paper-style 2-D spatial subspace more closely than the
raw centroid-difference Figure B.

Source / basis construction
===========================
Use ALL Synthetic-400 samples to fit per-layer spatial centroids from the same
Real-Gray object-pair residual used by the previous Figure-B script:

    q_L = (h_real_sub - h_real_ref) - (h_gray_sub - h_gray_ref)

For the selected layer L, let the four synthetic centroids be
mu_left, mu_right, mu_above, mu_below and their common center be c.
Build paper-style primal axes from centered class directions:

    u_r = unit(mu_r - c)

    d_H = unit(u_right - u_left)
    d_V = unit(u_above - u_below)

The two primal axes need not be mutually orthogonal. Define

    B = [d_H, d_V]
    D = B (B^T B)^(-1)

so D is the dual basis. Its first column is orthogonal to d_V and its second
column is orthogonal to d_H. We then scale each dual direction by half of the
corresponding synthetic centroid gap:

    halfH = 0.5 * |(mu_right - mu_left)^T d_H|
    halfV = 0.5 * |(mu_above - mu_below)^T d_V|

    M = D diag(halfH, halfV)

A horizontal coordinate edit alpha therefore uses

    delta_H = M [alpha, 0]^T

and a vertical coordinate edit uses

    delta_V = M [0, alpha]^T.

This makes the requested coordinate change pure with respect to the other
primal axis:

    B^T delta_H = [alpha*halfH, 0]
    B^T delta_V = [0, alpha*halfV]

Intervention
============
At the SAME selected layer L on COCO:

    h_sub += 0.5 * delta
    h_ref -= 0.5 * delta

COCO is not used to construct the spatial basis. All COCO samples are used for
evaluation (unless capped or filtered).

Outputs
=======
Horizontal (Linear-style split):
  - Original = left
  - Original = right
  curves: mapped left/right answer log-probability changes

Vertical (same style):
  - Original = under
  - Original = on
  curves: mapped under/on answer log-probability changes

All curves show mean +/- SEM.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u figureB_qwen3b_synthetic400_dualbasis_bothaxes_v1.py \
  --model qwen-3b \
  --layer 22 \
  --synthetic-root synthetic_shapes_4dir_400 \
  --alphas -1.5 -1.0 -0.5 0 0.5 1.0 1.5 \
  --synthetic-max-samples 0 \
  --coco-max-samples 0 \
  --test-filter all \
  --output-dir output/figureB_synth400_dualbasis_L22_v1 \
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
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj



def canonical_relation(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    table = {
        "left": "left",
        "left_of": "left",
        "right": "right",
        "right_of": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
        "bottom": "below",
    }
    if s not in table:
        raise ValueError(f"Unknown relation label: {x!r}")
    return table[s]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--synthetic-root",
        default="synthetic_shapes_4dir_400",
        help="Root directory of the original 400 synthetic shapes dataset.",
    )
    p.add_argument(
        "--synthetic-labels",
        default="labels.jsonl",
        help="labels.jsonl path relative to --synthetic-root (or absolute path).",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--layer", type=int, default=22)
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        help="Space-separated steering strengths. Negative=leftward, positive=rightward.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument(
        "--synthetic-max-samples",
        type=int,
        default=0,
        help="0 = all synthetic samples. Intended paper run should use all 400.",
    )
    p.add_argument(
        "--coco-max-samples",
        type=int,
        default=0,
        help="0 = all COCO-two samples before keeping only left/right.",
    )
    p.add_argument(
        "--test-filter",
        default="all",
        choices=["all", "correct", "wrong"],
        help="Optional filter based on ORIGINAL mapped-option prediction. Paper default: all.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# =============================================================================
# Random option mapping / prompt / option scoring
# =============================================================================

def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Mapping[str, str]) -> str:
    letter_to_rel = {a: r for r, a in rel_to_letter.items()}
    return "\n".join([
        f"Determine the spatial relation of the {subject} to the {reference} in the image.",
        *(f"{a}. {disp_rel(letter_to_rel[a])}" for a in LETTERS),
        "Answer with only A, B, C, or D.",
    ])


def assign_relation_balanced_mappings(
    items: Sequence[Mapping[str, Any]],
    seed: int,
) -> Dict[int, Dict[str, str]]:
    import itertools

    by_rel: Dict[str, List[Mapping[str, Any]]] = {r: [] for r in REL}
    for m in items:
        by_rel[str(m["gt"])].append(m)

    out: Dict[int, Dict[str, str]] = {}
    rng = random.Random(seed)
    perms = list(itertools.permutations(LETTERS, 4))

    for r in REL:
        arr = list(by_rel[r])
        rng.shuffle(arr)
        for i, m in enumerate(arr):
            perm = perms[i % len(perms)]
            rels = list(REL)
            rels.remove(r)
            rng_local = random.Random(seed * 1000003 + int(m["sid"]) * 9176 + i)
            rng_local.shuffle(rels)
            order = [r] + rels
            out[int(m["sid"])] = {rel: letter for rel, letter in zip(order, perm)}
    return out


def mapping_string(mp: Mapping[str, str]) -> str:
    return ",".join(f"{r}->{mp[r]}" for r in REL)


def build_option_token_map(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids: List[int] = []
        for text in [letter, " " + letter, "\n" + letter, "(" + letter + ")"]:
            enc = tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == 1 and int(enc[0]) not in ids:
                ids.append(int(enc[0]))
        if not ids:
            enc = tokenizer.encode(letter, add_special_tokens=False)
            if not enc:
                raise RuntimeError(f"Tokenizer cannot encode option {letter}")
            ids = [int(enc[0])]
        out[letter] = ids
    return out


def extract_logits(outputs: Any) -> torch.Tensor:
    for value in [
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ]:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("Could not locate logits tensor")


def option_scores_from_vector(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, float]]:
    """Return both raw option logits and full-vocab log probabilities."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    raw_out: Dict[str, float] = {}
    lp_out: Dict[str, float] = {}
    for letter in LETTERS:
        idx = torch.tensor(
            [int(x) for x in token_map[letter]],
            device=logits.device,
            dtype=torch.long,
        )
        raw_out[letter] = float(logits.index_select(0, idx).max().detach().cpu())
        lp_out[letter] = float(logp.index_select(0, idx).max().detach().cpu())
    return {"letter_logits": raw_out, "letter_logprobs": lp_out}


@torch.inference_mode()
def first_step_scores(
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    outputs = model(**batch, use_cache=False, return_dict=True)
    logits = extract_logits(outputs)[0, -1]
    scores = option_scores_from_vector(logits, token_map)
    pred = max(LETTERS, key=lambda a: scores["letter_logits"][a])
    del outputs
    return {**scores, "prediction": pred}


# =============================================================================
# Synthetic-400 loading
# =============================================================================

def resolve_synthetic_labels(root: Path, spec: str) -> Path:
    p = Path(spec)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"Synthetic labels not found: {p}")
    return p


def resolve_synthetic_image(root: Path, image_field: str) -> Path:
    p = Path(str(image_field))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            root / p,
            root / "images" / p,
            root / "images" / p.name,
        ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not resolve synthetic image {image_field!r}. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def load_synthetic400(root: Path, labels_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            relation_raw = obj.get("relation", obj.get("answer", obj.get("label")))
            if relation_raw is None:
                raise KeyError(f"No relation/answer/label in {labels_path}:{line_no}")
            gt = canonical_relation(relation_raw)
            subject = obj.get("subject")
            reference = obj.get("reference")
            image_field = obj.get("image", obj.get("image_path", obj.get("file")))
            if subject is None or reference is None or image_field is None:
                raise KeyError(
                    f"Synthetic row {line_no} needs image, subject, reference; got keys={sorted(obj.keys())}"
                )
            sid_raw = obj.get("id", obj.get("sid", line_no - 1))
            try:
                sid = int(sid_raw)
            except Exception:
                sid = line_no - 1
            image_path = resolve_synthetic_image(root, str(image_field))
            rows.append({
                "sid": sid,
                "gt": gt,
                "subject": str(subject),
                "reference": str(reference),
                "image_path": str(image_path),
            })

    # Ensure unique integer ids for mapping assignment even if the JSON ids collide.
    seen = set()
    for i, r in enumerate(rows):
        if r["sid"] in seen:
            r["sid"] = 1_000_000 + i
        seen.add(r["sid"])
    return rows


def stratified_cap(
    items: Sequence[Mapping[str, Any]],
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_samples <= 0 or len(items) <= max_samples:
        return [dict(x) for x in items]
    return traj.stratified_cap([dict(x) for x in items], max_samples, seed)


# =============================================================================
# Image batches / state extraction
# =============================================================================

def make_gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, color=(v, v, v))


def make_batch(
    processor: Any,
    device: torch.device,
    image: Image.Image,
    prompt: str,
) -> Dict[str, Any]:
    return base.make_question_batch(
        processor=processor,
        image=image,
        question_text=prompt,
        device=device,
    )


def collect_realgray_pair_states_from_image(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    image: Image.Image,
    prompt: str,
    subject: str,
    reference: str,
    layers: Sequence[int],
    object_state: str,
    relation_token_map: Mapping[str, Sequence[int]],
    gray_value: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Tuple[int, ...], Tuple[int, ...]]:
    real = image.convert("RGB") if hasattr(image, "convert") else image
    gray = make_gray_image(real, gray_value)
    rb = gb = None
    try:
        rb = make_batch(processor, device, real, prompt)
        gb = make_batch(processor, device, gray, prompt)

        cr = traj.clean_forward(
            base, model, processor, decoder_layers, rb,
            subject, reference, layers, object_state, relation_token_map,
        )
        cg = traj.clean_forward(
            base, model, processor, decoder_layers, gb,
            subject, reference, layers, object_state, relation_token_map,
        )

        q = {
            int(L): (
                np.asarray(cr["states"][L], np.float32)
                - np.asarray(cg["states"][L], np.float32)
            ).astype(np.float32)
            for L in layers
        }
        return q, tuple(cr["subject_positions"]), tuple(cr["reference_positions"])
    finally:
        with contextlib.suppress(Exception):
            gray.close()
        del rb, gb


def collect_coco_realgray_pair_states(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    rec: Any,
    prompt: str,
    subject: str,
    reference: str,
    layers: Sequence[int],
    object_state: str,
    relation_token_map: Mapping[str, Sequence[int]],
    gray_value: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Tuple[int, ...], Tuple[int, ...]]:
    image = None
    try:
        image = base.record_image(rec)
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return collect_realgray_pair_states_from_image(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            image=image,
            prompt=prompt,
            subject=subject,
            reference=reference,
            layers=layers,
            object_state=object_state,
            relation_token_map=relation_token_map,
            gray_value=gray_value,
            device=device,
        )
    finally:
        if image is not None:
            with contextlib.suppress(Exception):
                image.close()


# =============================================================================
# Intervention
# =============================================================================

class PairDeltaPatch:
    def __init__(
        self,
        layer_module: Any,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        delta_pair: np.ndarray,
    ):
        self.subject_positions = tuple(map(int, subject_positions))
        self.reference_positions = tuple(map(int, reference_positions))
        self.delta_pair = np.asarray(delta_pair, np.float32)
        self.applied = 0
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        h = traj.first_tensor(output)
        max_pos = max(self.subject_positions + self.reference_positions)
        if h.ndim != 3 or int(h.shape[1]) <= max_pos:
            return output
        y = h.float().clone()
        d = torch.as_tensor(self.delta_pair, device=y.device, dtype=torch.float32)
        for p in self.subject_positions:
            y[:, p, :] += 0.5 * d
        for p in self.reference_positions:
            y[:, p, :] -= 0.5 * d
        self.applied += 1
        return traj.replace_first_tensor(output, y.to(h.dtype))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


@torch.inference_mode()
def patched_first_step_scores(
    *,
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    layer: int,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    delta_pair: np.ndarray,
) -> Dict[str, Any]:
    patch = PairDeltaPatch(
        decoder_layers[layer],
        subject_positions,
        reference_positions,
        delta_pair,
    )
    try:
        outputs = model(**batch, use_cache=False, return_dict=True)
        if patch.applied < 1:
            raise RuntimeError(f"L{layer} pair-state patch did not fire")
        logits = extract_logits(outputs)[0, -1]
        scores = option_scores_from_vector(logits, token_map)
        pred = max(LETTERS, key=lambda a: scores["letter_logits"][a])
        del outputs
        return {**scores, "prediction": pred}
    finally:
        patch.close()


REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
DISPLAY_REL = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
SCRIPT_VERSION = "figureB-qwen3b-synthetic400-dualbasis-bothaxes-v2-standalone"
EPS = 1e-8


def disp_rel(r: str) -> str:
    return DISPLAY_REL.get(str(r), str(r))


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = float(np.linalg.norm(x))
    if n < EPS:
        raise RuntimeError("Near-zero vector in spatial-basis construction")
    return (x / n).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--synthetic-root",
        default="synthetic_shapes_4dir_400",
    )
    p.add_argument(
        "--synthetic-labels",
        default="labels.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--layer", type=int, default=22)
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        help=(
            "Coordinate edit strengths. Horizontal: negative=left, positive=right. "
            "Vertical: negative=under, positive=on."
        ),
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--synthetic-max-samples", type=int, default=0)
    p.add_argument("--coco-max-samples", type=int, default=0)
    p.add_argument(
        "--test-filter",
        default="all",
        choices=["all", "correct", "wrong"],
        help="Filter by ORIGINAL mapped-option prediction. Paper default: all.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sem(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_dual_basis(centroids: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    mu = {r: np.asarray(centroids[r], np.float32) for r in REL}
    c = np.mean(np.stack([mu[r] for r in REL], axis=0), axis=0).astype(np.float32)

    u = {r: unit(mu[r] - c) for r in REL}
    d_h = unit(u["right"] - u["left"])
    d_v = unit(u["above"] - u["below"])

    B = np.stack([d_h, d_v], axis=1).astype(np.float64)  # hidden_dim x 2
    gram = B.T @ B
    cond = float(np.linalg.cond(gram))
    if not np.isfinite(cond) or cond > 1e8:
        raise RuntimeError(f"Ill-conditioned spatial basis: cond(B^T B)={cond:.3e}")

    D = (B @ np.linalg.inv(gram)).astype(np.float32)

    gap_h = float(abs(np.dot(mu["right"] - mu["left"], d_h)))
    gap_v = float(abs(np.dot(mu["above"] - mu["below"], d_v)))
    half_h = 0.5 * gap_h
    half_v = 0.5 * gap_v

    M = (D @ np.diag([half_h, half_v]).astype(np.float32)).astype(np.float32)
    steer_h = M[:, 0].astype(np.float32)
    steer_v = M[:, 1].astype(np.float32)

    cos_hv = float(np.dot(d_h, d_v))  # both unit
    check_h = np.asarray(B.T @ steer_h, np.float64)
    check_v = np.asarray(B.T @ steer_v, np.float64)

    return {
        "center": c,
        "u": u,
        "d_h": d_h,
        "d_v": d_v,
        "B": B.astype(np.float32),
        "D": D,
        "M": M,
        "steer_h": steer_h,
        "steer_v": steer_v,
        "gap_h": gap_h,
        "gap_v": gap_v,
        "half_h": half_h,
        "half_v": half_v,
        "cos_hv": cos_hv,
        "gram_cond": cond,
        "check_h": check_h,
        "check_v": check_v,
        "cross_h_to_v": float(np.dot(steer_h, d_v)),
        "cross_v_to_h": float(np.dot(steer_v, d_h)),
    }


def summarize_axis_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    original_relation: str,
) -> List[Dict[str, Any]]:
    by_alpha: Dict[float, List[Mapping[str, Any]]] = {}
    for r in rows:
        if str(r["axis"]) != axis:
            continue
        if str(r["original_relation"]) != original_relation:
            continue
        by_alpha.setdefault(float(r["alpha"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for alpha in sorted(by_alpha):
        chunk = by_alpha[alpha]
        neg_lp = np.asarray([float(x["delta_neg_logprob"]) for x in chunk], np.float64)
        pos_lp = np.asarray([float(x["delta_pos_logprob"]) for x in chunk], np.float64)
        neg_logit = np.asarray([float(x["delta_neg_logit"]) for x in chunk], np.float64)
        pos_logit = np.asarray([float(x["delta_pos_logit"]) for x in chunk], np.float64)
        out.append({
            "axis": axis,
            "original_relation": original_relation,
            "alpha": float(alpha),
            "n": len(chunk),
            "neg_logprob_mean": float(neg_lp.mean()),
            "neg_logprob_sem": sem(neg_lp),
            "pos_logprob_mean": float(pos_lp.mean()),
            "pos_logprob_sem": sem(pos_lp),
            "neg_logit_mean": float(neg_logit.mean()),
            "neg_logit_sem": sem(neg_logit),
            "pos_logit_mean": float(pos_logit.mean()),
            "pos_logit_sem": sem(pos_logit),
        })
    return out


def plot_panel(
    ax: Any,
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    neg_name: str,
    pos_name: str,
    title: str,
) -> None:
    x = np.asarray([float(r["alpha"]) for r in summary_rows], np.float64)
    y_pos = np.asarray([float(r["pos_logprob_mean"]) for r in summary_rows], np.float64)
    e_pos = np.asarray([float(r["pos_logprob_sem"]) for r in summary_rows], np.float64)
    y_neg = np.asarray([float(r["neg_logprob_mean"]) for r in summary_rows], np.float64)
    e_neg = np.asarray([float(r["neg_logprob_sem"]) for r in summary_rows], np.float64)

    ax.axhline(0.0, linestyle="--", linewidth=1.3, color="0.35", zorder=0)

    lp, = ax.plot(
        x, y_pos,
        marker="o", markersize=5.5, linewidth=2.4,
        label=pos_name.capitalize(),
    )
    ax.fill_between(
        x, y_pos - e_pos, y_pos + e_pos,
        color=lp.get_color(), alpha=0.18, linewidth=0,
    )

    ln, = ax.plot(
        x, y_neg,
        marker="o", markersize=5.5, linewidth=2.4,
        label=neg_name.capitalize(),
    )
    ax.fill_between(
        x, y_neg - e_neg, y_neg + e_neg,
        color=ln.get_color(), alpha=0.18, linewidth=0,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r"Spatial coordinate edit $\alpha$", fontsize=14)
    ax.set_ylabel("Change in log probability", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.20)
    ax.legend(frameon=False, fontsize=11, loc="best")


def save_axis_split(
    path: Path,
    summary_neg: Sequence[Mapping[str, Any]],
    summary_pos: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    neg_name: str,
    pos_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), dpi=240)
    plot_panel(
        axes[0], summary_neg,
        neg_name=neg_name, pos_name=pos_name,
        title=f"Layer {layer} — Original: '{neg_name}'",
    )
    plot_panel(
        axes[1], summary_pos,
        neg_name=neg_name, pos_name=pos_name,
        title=f"Layer {layer} — Original: '{pos_name}'",
    )
    fig.tight_layout(pad=0.7, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_all_2x2(
    path: Path,
    h_left: Sequence[Mapping[str, Any]],
    h_right: Sequence[Mapping[str, Any]],
    v_under: Sequence[Mapping[str, Any]],
    v_on: Sequence[Mapping[str, Any]],
    *,
    layer: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), dpi=240)
    plot_panel(
        axes[0, 0], h_left,
        neg_name="left", pos_name="right",
        title=f"Layer {layer} — Original: 'left'",
    )
    plot_panel(
        axes[0, 1], h_right,
        neg_name="left", pos_name="right",
        title=f"Layer {layer} — Original: 'right'",
    )
    plot_panel(
        axes[1, 0], v_under,
        neg_name="under", pos_name="on",
        title=f"Layer {layer} — Original: 'under'",
    )
    plot_panel(
        axes[1, 1], v_on,
        neg_name="under", pos_name="on",
        title=f"Layer {layer} — Original: 'on'",
    )
    fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    if not 0 <= a.gray_value <= 255:
        raise ValueError("--gray-value must be in [0,255]")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    err_path = out / "errors.jsonl"

    alphas = sorted(float(x) for x in a.alphas)
    if not alphas:
        raise ValueError("No --alphas supplied")

    # ------------------------------------------------------------------
    # Synthetic-400 source
    # ------------------------------------------------------------------
    synth_root = Path(a.synthetic_root)
    synth_labels = resolve_synthetic_labels(synth_root, a.synthetic_labels)
    synthetic = load_synthetic400(synth_root, synth_labels)
    synthetic = stratified_cap(synthetic, a.synthetic_max_samples, a.seed)
    synth_counts = {r: sum(1 for m in synthetic if m["gt"] == r) for r in REL}
    synth_map = assign_relation_balanced_mappings(synthetic, a.seed + 1001)

    # ------------------------------------------------------------------
    # COCO target: ALL four relations, no train/test split
    # ------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    coco_all: List[Dict[str, Any]] = []
    for r in records:
        sid = int(r.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        coco_all.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
        })

    coco_all = stratified_cap(coco_all, a.coco_max_samples, a.seed + 53)
    coco_counts = {r: sum(1 for m in coco_all if m["gt"] == r) for r in REL}
    coco_map = assign_relation_balanced_mappings(coco_all, a.seed + 2003)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)
    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    model = processor = None
    try:
        print(f"[LOAD] {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        layer = int(a.layer)
        if not 0 <= layer < len(decoder_layers):
            raise ValueError(f"Invalid --layer {layer}; decoder has {len(decoder_layers)} blocks")
        one_layer = [layer]

        relation_token_map = base.relation_token_variants(processor.tokenizer)
        option_token_map = build_option_token_map(processor.tokenizer)

        print("=" * 112)
        print("FIGURE B — SYNTHETIC-400 DUAL-BASIS STEERING -> COCO BOTH SPATIAL AXES")
        print("=" * 112)
        print(f"decoder={decoder_path} | layer=L{layer} | alphas={alphas}")
        print(f"synthetic n={len(synthetic)} counts={synth_counts}")
        print(f"COCO n={len(coco_all)} counts={coco_counts} | filter={a.test_filter}")
        print("COCO is evaluation only; Synthetic-400 defines the basis.")
        print()

        # --------------------------------------------------------------
        # 1) Synthetic per-layer centroids
        # --------------------------------------------------------------
        synth_q: Dict[int, Dict[int, np.ndarray]] = {}
        synth_valid: List[Dict[str, Any]] = []

        for m in tqdm(synthetic, desc=f"Synthetic-400 Real-Gray states @ L{layer}"):
            sid = int(m["sid"])
            prompt = build_randmap_prompt(m["subject"], m["reference"], synth_map[sid])
            image = None
            try:
                image = Image.open(m["image_path"]).convert("RGB")
                q, _sp, _rp = collect_realgray_pair_states_from_image(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    image=image,
                    prompt=prompt,
                    subject=m["subject"],
                    reference=m["reference"],
                    layers=one_layer,
                    object_state=a.object_state,
                    relation_token_map=relation_token_map,
                    gray_value=a.gray_value,
                    device=device,
                )
                synth_q[sid] = q
                synth_valid.append(m)
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "synthetic_basis_fit",
                    "sid": sid,
                    "image_path": m["image_path"],
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                gc.collect()

        cent = traj.fit_centroids(synth_valid, synth_q, one_layer)
        cent_L = {r: np.asarray(cent[layer][r], np.float32) for r in REL}
        basis = build_dual_basis(cent_L)

        np.savez_compressed(
            out / f"synthetic400_dual_basis_L{layer}.npz",
            **{f"mu_{r}": cent_L[r] for r in REL},
            center=basis["center"],
            d_H=basis["d_h"],
            d_V=basis["d_v"],
            B=basis["B"],
            D=basis["D"],
            M=basis["M"],
            steer_H=basis["steer_h"],
            steer_V=basis["steer_v"],
            halfH=np.asarray([basis["half_h"]], np.float32),
            halfV=np.asarray([basis["half_v"]], np.float32),
        )

        print("[BASIS]")
        print(f"  cos(d_H,d_V)            = {basis['cos_hv']:+.6f}")
        print(f"  cond(B^T B)             = {basis['gram_cond']:.6f}")
        print(f"  gap_H / halfH           = {basis['gap_h']:.6f} / {basis['half_h']:.6f}")
        print(f"  gap_V / halfV           = {basis['gap_v']:.6f} / {basis['half_v']:.6f}")
        print(f"  dot(steer_H,d_V)        = {basis['cross_h_to_v']:+.6e}")
        print(f"  dot(steer_V,d_H)        = {basis['cross_v_to_h']:+.6e}")
        print(f"  B^T steer_H             = {basis['check_h'].tolist()}")
        print(f"  B^T steer_V             = {basis['check_v'].tolist()}")
        print()

        # --------------------------------------------------------------
        # 2) COCO evaluation on both axes
        # --------------------------------------------------------------
        rows: List[Dict[str, Any]] = []
        kept_counts = {r: 0 for r in REL}

        axis_cfg = {
            "horizontal": {
                "neg_rel": "left",
                "pos_rel": "right",
                "steer": basis["steer_h"],
            },
            "vertical": {
                "neg_rel": "below",   # display = under
                "pos_rel": "above",   # display = on
                "steer": basis["steer_v"],
            },
        }

        for m in tqdm(coco_all, desc=f"COCO dual-basis steering @ L{layer}"):
            sid = int(m["sid"])
            mapping = coco_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mapping)
            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = make_batch(processor, device, image, prompt)
                base_sc = first_step_scores(model, batch, option_token_map)

                gt_letter = mapping[m["gt"]]
                base_correct = (base_sc["prediction"] == gt_letter)
                if a.test_filter == "correct" and not base_correct:
                    continue
                if a.test_filter == "wrong" and base_correct:
                    continue

                _q, subject_positions, reference_positions = collect_coco_realgray_pair_states(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    rec=rec_by_sid[sid],
                    prompt=prompt,
                    subject=m["subject"],
                    reference=m["reference"],
                    layers=one_layer,
                    object_state=a.object_state,
                    relation_token_map=relation_token_map,
                    gray_value=a.gray_value,
                    device=device,
                )
                kept_counts[m["gt"]] += 1

                # Evaluate the axis corresponding to the sample's original GT.
                if m["gt"] in ("left", "right"):
                    axis_names = ["horizontal"]
                elif m["gt"] in ("below", "above"):
                    axis_names = ["vertical"]
                else:
                    axis_names = []

                for axis_name in axis_names:
                    cfg = axis_cfg[axis_name]
                    neg_rel = cfg["neg_rel"]
                    pos_rel = cfg["pos_rel"]
                    steer = np.asarray(cfg["steer"], np.float32)

                    neg_letter = mapping[neg_rel]
                    pos_letter = mapping[pos_rel]
                    base_neg_logit = float(base_sc["letter_logits"][neg_letter])
                    base_pos_logit = float(base_sc["letter_logits"][pos_letter])
                    base_neg_logp = float(base_sc["letter_logprobs"][neg_letter])
                    base_pos_logp = float(base_sc["letter_logprobs"][pos_letter])

                    for alpha in alphas:
                        patched = patched_first_step_scores(
                            model=model,
                            decoder_layers=decoder_layers,
                            batch=batch,
                            token_map=option_token_map,
                            layer=layer,
                            subject_positions=subject_positions,
                            reference_positions=reference_positions,
                            delta_pair=(float(alpha) * steer).astype(np.float32),
                        )

                        neg_logit = float(patched["letter_logits"][neg_letter])
                        pos_logit = float(patched["letter_logits"][pos_letter])
                        neg_logp = float(patched["letter_logprobs"][neg_letter])
                        pos_logp = float(patched["letter_logprobs"][pos_letter])

                        rows.append({
                            "sid": sid,
                            "axis": axis_name,
                            "original_relation": m["gt"],
                            "original_relation_display": disp_rel(m["gt"]),
                            "mapping": mapping_string(mapping),
                            "neg_relation": neg_rel,
                            "neg_relation_display": disp_rel(neg_rel),
                            "pos_relation": pos_rel,
                            "pos_relation_display": disp_rel(pos_rel),
                            "neg_letter": neg_letter,
                            "pos_letter": pos_letter,
                            "base_prediction": base_sc["prediction"],
                            "base_correct": int(base_correct),
                            "layer": layer,
                            "alpha": float(alpha),
                            "delta_neg_logprob": neg_logp - base_neg_logp,
                            "delta_pos_logprob": pos_logp - base_pos_logp,
                            "delta_neg_logit": neg_logit - base_neg_logit,
                            "delta_pos_logit": pos_logit - base_pos_logit,
                            "patched_prediction": patched["prediction"],
                        })

            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "coco_dualbasis_steering",
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

        if not rows:
            raise RuntimeError("No COCO evaluation rows were produced")

        write_csv(out / "figureB_dualbasis_rows.csv", rows)

        h_left = summarize_axis_rows(rows, axis="horizontal", original_relation="left")
        h_right = summarize_axis_rows(rows, axis="horizontal", original_relation="right")
        v_under = summarize_axis_rows(rows, axis="vertical", original_relation="below")
        v_on = summarize_axis_rows(rows, axis="vertical", original_relation="above")

        summaries = h_left + h_right + v_under + v_on
        write_csv(out / "figureB_dualbasis_summary.csv", summaries)

        save_axis_split(
            out / "figureB_dualbasis_horizontal_left_right.png",
            h_left, h_right,
            layer=layer,
            neg_name="left",
            pos_name="right",
        )
        save_axis_split(
            out / "figureB_dualbasis_vertical_under_on.png",
            v_under, v_on,
            layer=layer,
            neg_name="under",
            pos_name="on",
        )
        save_all_2x2(
            out / "figureB_dualbasis_both_axes_2x2.png",
            h_left, h_right, v_under, v_on,
            layer=layer,
        )

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "layer": layer,
            "alphas": alphas,
            "seed": int(a.seed),
            "object_state": a.object_state,
            "gray_value": int(a.gray_value),
            "synthetic_root": str(synth_root),
            "synthetic_labels": str(synth_labels),
            "synthetic_n": len(synthetic),
            "synthetic_counts": synth_counts,
            "synthetic_valid_n": len(synth_valid),
            "coco_n": len(coco_all),
            "coco_counts": coco_counts,
            "coco_kept_counts": kept_counts,
            "test_filter": a.test_filter,
            "basis_definition": (
                "paper-style centered class directions; B=[d_H,d_V]; "
                "dual D=B(B^T B)^-1; M=D diag(halfH,halfV)"
            ),
            "d_H_definition": "unit(unit(mu_right-c)-unit(mu_left-c))",
            "d_V_definition": "unit(unit(mu_above-c)-unit(mu_below-c))",
            "halfH": basis["half_h"],
            "halfV": basis["half_v"],
            "cos_dH_dV": basis["cos_hv"],
            "gram_condition_number": basis["gram_cond"],
            "dot_steerH_dV": basis["cross_h_to_v"],
            "dot_steerV_dH": basis["cross_v_to_h"],
            "B_T_steerH": basis["check_h"].tolist(),
            "B_T_steerV": basis["check_v"].tolist(),
            "evaluation_split": "all COCO; horizontal samples split by original left/right, vertical by original under/on",
            "plot_metric": "change in full-vocabulary log probability of each relation's mapped option",
            "error_band": "mean +/- SEM",
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + "=" * 112)
        print("FIGURE B COMPLETE")
        print("=" * 112)
        print(f"COCO kept counts: {kept_counts}")
        print(f"[SAVED] {out / 'figureB_dualbasis_horizontal_left_right.png'}")
        print(f"[SAVED] {out / 'figureB_dualbasis_vertical_under_on.png'}")
        print(f"[SAVED] {out / 'figureB_dualbasis_both_axes_2x2.png'}")
        print(f"[SAVED] {out / 'figureB_dualbasis_rows.csv'}")
        print(f"[SAVED] {out / 'figureB_dualbasis_summary.csv'}")

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
