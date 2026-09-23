#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial-margin matched diagnosis for the claim:

    a correct intermediate spatial state can still fail to control
    the downstream answer decision.

Design
======
1) Spatial reader (NO COCO leakage):
   - Fit per-layer relation centroids on ALL Synthetic-400 samples.
   - Carrier:
       q_obj,L = (h_real_sub - h_real_ref) - (h_gray_sub - h_gray_ref)
   - Spatial score is centered cosine to the four synthetic relation centroids.

2) Decision reader (held-out):
   - Assign each COCO example an independent balanced relation -> A/B/C/D mapping.
   - Split COCO 30%/70% after mapping, stratified by (GT relation, GT mapped option).
   - Fit per-layer A/B/C/D centroids on TRAIN-30 only using prompt-last residual:
       q_last,L = h_real_last - h_gray_last
   - Evaluate all diagnosis on TEST-70 only.

3) For each TEST-70 sample and layer:
   spatial_margin(L) = score_spatial(GT) - max(other relation scores)
   decision_margin(L)= score_option(mapped-GT) - max(other option scores)

4) At --match-layer (default L22), keep samples with positive spatial margin.
   Pair each final-WRONG example with a final-CORRECT example that has:
     - the same GT relation,
     - the same mapped correct option A/B/C/D,
     - the closest spatial margin at the match layer,
   without replacement.

   This controls upstream spatial strength. If the matched groups have nearly
   identical intermediate spatial margins but diverge later in option/decision
   margin, the final error is difficult to explain as merely weaker spatial
   evidence.

5) Also report a high-margin subset: among spatial-correct TEST examples,
   take the top --high-margin-quantile fraction by match-layer spatial margin
   and count how many are still final-wrong.

Outputs
=======
  per_sample_layer.csv
  matched_pairs.csv
  matched_pair_layer.csv
  matched_summary.csv
  high_margin_summary.json
  split_audit.csv
  figure_spatial_margin_distribution.png
  figure_matched_trajectories.png
  figure_spatial_vs_decision_scatter.png
  metadata.json

Expected repo modules (existing AdaptVis utilities):
  analyze_coco_centroid_generation_step1_v4.py
  eval_coco_multilayer_relation_trajectory_repair_v1.py

This script does NOT import any other Figure-A/Figure-B helper script.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u eval_qwen3b_coco_spatial_margin_matched_v1.py \
  --model qwen-3b \
  --layers 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --match-layer 22 \
  --decision-layer 30 \
  --synthetic-root synthetic_shapes_4dir_400 \
  --train-frac 0.30 \
  --high-margin-quantile 0.50 \
  --output-dir output/qwen3b_coco_spatial_margin_matched_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import itertools
import json
import math
import random
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

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


REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
DISPLAY_REL = {"left": "left", "right": "right", "above": "on", "below": "under"}
SCRIPT_VERSION = "qwen3b-coco-spatial-margin-matched-v1"
EPS = 1e-8


def canonical_relation(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    table = {
        "left": "left", "left_of": "left",
        "right": "right", "right_of": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    if s not in table:
        raise ValueError(f"Unknown relation label: {x!r}")
    return table[s]


def disp_rel(r: str) -> str:
    return DISPLAY_REL.get(str(r), str(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--synthetic-root", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default="labels.jsonl")
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--layers", nargs="+", type=int,
                   default=list(range(18, 33)))
    p.add_argument("--match-layer", type=int, default=22)
    p.add_argument("--decision-layer", type=int, default=30,
                   help="Late layer used for the 2-D spatial-vs-decision scatter.")
    p.add_argument("--train-frac", type=float, default=0.30,
                   help="COCO fraction used ONLY to fit option/decision centroids.")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--synthetic-max-samples", type=int, default=0)
    p.add_argument("--coco-max-samples", type=int, default=0)
    p.add_argument("--max-margin-gap", type=float, default=-1.0,
                   help="Maximum allowed |wrong margin - correct margin| for pairing. <0 means unlimited.")
    p.add_argument("--high-margin-quantile", type=float, default=0.50,
                   help="Top fraction threshold among spatial-correct TEST samples. 0.50 means top half.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Basic I/O helpers
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sem(x: Sequence[float]) -> float:
    a = np.asarray(list(x), np.float64)
    return float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0


def mean_or_nan(x: Sequence[float]) -> float:
    a = np.asarray(list(x), np.float64)
    return float(a.mean()) if len(a) else float("nan")


def stratified_cap(items: Sequence[Mapping[str, Any]], max_samples: int, seed: int) -> List[Dict[str, Any]]:
    if max_samples <= 0 or len(items) <= max_samples:
        return [dict(x) for x in items]
    return traj.stratified_cap([dict(x) for x in items], max_samples, seed)


# -----------------------------------------------------------------------------
# Random per-example relation -> A/B/C/D mapping
# -----------------------------------------------------------------------------

def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Mapping[str, str]) -> str:
    letter_to_rel = {a: r for r, a in rel_to_letter.items()}
    return "\n".join([
        f"Determine the spatial relation of the {subject} to the {reference} in the image.",
        *(f"{a}. {disp_rel(letter_to_rel[a])}" for a in LETTERS),
        "Answer with only A, B, C, or D.",
    ])


def assign_relation_balanced_mappings(items: Sequence[Mapping[str, Any]], seed: int) -> Dict[int, Dict[str, str]]:
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


def joint_stratified_split(
    items: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, Mapping[str, str]],
    train_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train-frac must be in (0,1)")

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for x in items:
        sid = int(x["sid"])
        key = (str(x["gt"]), str(mapping[sid][str(x["gt"])]))
        groups.setdefault(key, []).append(dict(x))

    rng = random.Random(seed)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for key in sorted(groups):
        arr = groups[key]
        rng.shuffle(arr)
        n_train = int(round(len(arr) * train_frac))
        if len(arr) >= 2:
            n_train = max(1, min(len(arr) - 1, n_train))
        train.extend(arr[:n_train])
        test.extend(arr[n_train:])

    train.sort(key=lambda x: int(x["sid"]))
    test.sort(key=lambda x: int(x["sid"]))
    return train, test


# -----------------------------------------------------------------------------
# Synthetic-400 loading
# -----------------------------------------------------------------------------

def resolve_synthetic_labels(root: Path, spec: str) -> Path:
    p = Path(spec)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"Synthetic labels not found: {p}")
    return p


def resolve_synthetic_image(root: Path, image_field: str) -> Path:
    p = Path(str(image_field))
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([root / p, root / "images" / p, root / "images" / p.name])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not resolve synthetic image {image_field!r}; tried: "
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
            rel_raw = obj.get("relation", obj.get("answer", obj.get("label")))
            if rel_raw is None:
                raise KeyError(f"No relation/answer/label in {labels_path}:{line_no}")
            gt = canonical_relation(rel_raw)
            subject = obj.get("subject")
            reference = obj.get("reference")
            image_field = obj.get("image", obj.get("image_path", obj.get("file")))
            if subject is None or reference is None or image_field is None:
                raise KeyError(
                    f"Synthetic row {line_no} needs image, subject, reference; keys={sorted(obj.keys())}"
                )
            sid_raw = obj.get("id", obj.get("sid", line_no - 1))
            try:
                sid = int(sid_raw)
            except Exception:
                sid = line_no - 1
            rows.append({
                "sid": sid,
                "gt": gt,
                "subject": str(subject),
                "reference": str(reference),
                "image_path": str(resolve_synthetic_image(root, str(image_field))),
            })

    seen = set()
    for i, r in enumerate(rows):
        if r["sid"] in seen:
            r["sid"] = 1_000_000 + i
        seen.add(r["sid"])
    return rows


# -----------------------------------------------------------------------------
# Option scoring
# -----------------------------------------------------------------------------

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


def option_scores_from_vector(logits: torch.Tensor, token_map: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    logp = torch.log_softmax(logits.float(), dim=-1)
    raw: Dict[str, float] = {}
    lp: Dict[str, float] = {}
    for letter in LETTERS:
        idx = torch.tensor([int(x) for x in token_map[letter]], device=logits.device, dtype=torch.long)
        raw[letter] = float(logits.index_select(0, idx).max().detach().cpu())
        lp[letter] = float(logp.index_select(0, idx).max().detach().cpu())
    pred = max(LETTERS, key=lambda a: raw[a])
    return {"letter_logits": raw, "letter_logprobs": lp, "prediction": pred}


@torch.inference_mode()
def first_step_scores(model: Any, batch: Mapping[str, Any], token_map: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    outputs = model(**batch, use_cache=False, return_dict=True)
    logits = extract_logits(outputs)[0, -1]
    ans = option_scores_from_vector(logits, token_map)
    del outputs
    return ans


# -----------------------------------------------------------------------------
# Real/gray object-pair + prompt-last residual extraction
# -----------------------------------------------------------------------------

def make_gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, color=(v, v, v))


def make_batch(processor: Any, device: torch.device, image: Image.Image, prompt: str) -> Dict[str, Any]:
    return base.make_question_batch(
        processor=processor,
        image=image,
        question_text=prompt,
        device=device,
    )


class LastTokenCapture:
    """Capture decoder-layer output at the final prompt position."""
    def __init__(self, decoder_layers: Sequence[Any], layers: Sequence[int]):
        self.states: Dict[int, np.ndarray] = {}
        self.handles = []
        for L in layers:
            self.handles.append(decoder_layers[int(L)].register_forward_hook(self._make_hook(int(L))))

    def _make_hook(self, L: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            h = traj.first_tensor(output)
            if h.ndim != 3:
                raise RuntimeError(f"Expected layer tensor [B,T,D] at L{L}, got {tuple(h.shape)}")
            self.states[L] = h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            return output
        return hook

    def close(self) -> None:
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def clean_forward_with_last_capture(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    subject: str,
    reference: str,
    layers: Sequence[int],
    object_state: str,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
    cap = LastTokenCapture(decoder_layers, layers)
    try:
        clean = traj.clean_forward(
            base, model, processor, decoder_layers, batch,
            subject, reference, layers, object_state, relation_token_map,
        )
        missing = [L for L in layers if int(L) not in cap.states]
        if missing:
            raise RuntimeError(f"Last-token hooks did not capture layers: {missing}")
        return clean, dict(cap.states)
    finally:
        cap.close()


def collect_realgray_states_from_image(
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
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    real = image.convert("RGB") if hasattr(image, "convert") else image
    gray = make_gray_image(real, gray_value)
    rb = gb = None
    try:
        rb = make_batch(processor, device, real, prompt)
        gb = make_batch(processor, device, gray, prompt)

        cr, last_r = clean_forward_with_last_capture(
            model=model, processor=processor, decoder_layers=decoder_layers,
            batch=rb, subject=subject, reference=reference, layers=layers,
            object_state=object_state, relation_token_map=relation_token_map,
        )
        cg, last_g = clean_forward_with_last_capture(
            model=model, processor=processor, decoder_layers=decoder_layers,
            batch=gb, subject=subject, reference=reference, layers=layers,
            object_state=object_state, relation_token_map=relation_token_map,
        )

        q_obj = {
            int(L): (
                np.asarray(cr["states"][int(L)], np.float32)
                - np.asarray(cg["states"][int(L)], np.float32)
            ).astype(np.float32)
            for L in layers
        }
        q_last = {
            int(L): (
                np.asarray(last_r[int(L)], np.float32)
                - np.asarray(last_g[int(L)], np.float32)
            ).astype(np.float32)
            for L in layers
        }
        return q_obj, q_last
    finally:
        with contextlib.suppress(Exception):
            gray.close()
        del rb, gb


# -----------------------------------------------------------------------------
# Centroid readers
# -----------------------------------------------------------------------------

def fit_centered_centroid_reader(
    items: Sequence[Mapping[str, Any]],
    states: Mapping[int, Mapping[int, np.ndarray]],
    layers: Sequence[int],
    label_fn,
    classes: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for L in layers:
        per_class: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
        for m in items:
            sid = int(m["sid"])
            if sid not in states or int(L) not in states[sid]:
                continue
            c = str(label_fn(m))
            if c not in per_class:
                continue
            per_class[c].append(np.asarray(states[sid][int(L)], np.float32))

        mu: Dict[str, np.ndarray] = {}
        for c in classes:
            if not per_class[c]:
                raise RuntimeError(f"No states for class={c} at L{L}")
            mu[c] = np.mean(np.stack(per_class[c], axis=0), axis=0).astype(np.float32)

        center = np.mean(np.stack([mu[c] for c in classes], axis=0), axis=0).astype(np.float32)
        dirs: Dict[str, np.ndarray] = {}
        for c in classes:
            v = (mu[c] - center).astype(np.float32)
            n = float(np.linalg.norm(v))
            if n < EPS:
                raise RuntimeError(f"Near-zero centroid direction for class={c} at L{L}")
            dirs[c] = (v / n).astype(np.float32)

        out[int(L)] = {"center": center, "mu": mu, "dirs": dirs}
    return out


def reader_scores(state: np.ndarray, reader_L: Mapping[str, Any], classes: Sequence[str]) -> Dict[str, float]:
    x = np.asarray(state, np.float32) - np.asarray(reader_L["center"], np.float32)
    n = float(np.linalg.norm(x))
    if n < EPS:
        return {c: 0.0 for c in classes}
    x = x / n
    return {c: float(np.dot(x, np.asarray(reader_L["dirs"][c], np.float32))) for c in classes}


def margin_from_scores(scores: Mapping[str, float], target: str, classes: Sequence[str]) -> float:
    return float(scores[target] - max(scores[c] for c in classes if c != target))


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------

def greedy_margin_match(
    test_items: Sequence[Mapping[str, Any]],
    sample_at_match: Mapping[int, Mapping[str, Any]],
    mapping: Mapping[int, Mapping[str, str]],
    max_gap: float,
) -> List[Dict[str, Any]]:
    """Match each final-wrong S+ example to one final-correct S+ example.

    Exact strata: (GT relation, mapped correct option).
    Distance: absolute difference in match-layer spatial margin.
    Matching is without replacement within each stratum.
    """
    by_key_correct: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    wrong: List[Dict[str, Any]] = []

    item_by_sid = {int(x["sid"]): dict(x) for x in test_items}

    for sid, row in sample_at_match.items():
        if float(row["spatial_margin"]) <= 0:
            continue
        gt = str(row["gt"])
        gt_option = str(row["gt_option"])
        key = (gt, gt_option)
        rec = {
            "sid": int(sid),
            "gt": gt,
            "gt_option": gt_option,
            "spatial_margin": float(row["spatial_margin"]),
            "final_correct": int(row["final_correct"]),
        }
        if int(row["final_correct"]) == 1:
            by_key_correct.setdefault(key, []).append(rec)
        else:
            wrong.append(rec)

    for key in by_key_correct:
        by_key_correct[key].sort(key=lambda z: (z["spatial_margin"], z["sid"]))

    # Match hard cases first: wrong examples with fewer/less close candidates naturally
    # get handled deterministically by sorting within strata and then nearest neighbor.
    wrong.sort(key=lambda z: (z["gt"], z["gt_option"], z["spatial_margin"], z["sid"]))

    used_correct = set()
    pairs: List[Dict[str, Any]] = []
    pair_id = 0
    for w in wrong:
        key = (w["gt"], w["gt_option"])
        candidates = [c for c in by_key_correct.get(key, []) if int(c["sid"]) not in used_correct]
        if not candidates:
            continue
        c = min(candidates, key=lambda z: (abs(float(z["spatial_margin"]) - float(w["spatial_margin"])), z["sid"]))
        gap = abs(float(c["spatial_margin"]) - float(w["spatial_margin"]))
        if max_gap >= 0 and gap > max_gap:
            continue
        used_correct.add(int(c["sid"]))
        pairs.append({
            "pair_id": pair_id,
            "gt": w["gt"],
            "gt_display": disp_rel(w["gt"]),
            "gt_option": w["gt_option"],
            "wrong_sid": int(w["sid"]),
            "wrong_spatial_margin": float(w["spatial_margin"]),
            "correct_sid": int(c["sid"]),
            "correct_spatial_margin": float(c["spatial_margin"]),
            "abs_margin_gap": float(gap),
        })
        pair_id += 1

    return pairs


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_margin_distribution(path: Path, rows_match: Sequence[Mapping[str, Any]], match_layer: int) -> None:
    corr = np.asarray([float(r["spatial_margin"]) for r in rows_match if int(r["final_correct"]) == 1], np.float64)
    wrong = np.asarray([float(r["spatial_margin"]) for r in rows_match if int(r["final_correct"]) == 0], np.float64)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=240)
    bins = 24
    if len(corr):
        ax.hist(corr, bins=bins, density=True, alpha=0.45, label="Final correct")
    if len(wrong):
        ax.hist(wrong, bins=bins, density=True, alpha=0.45, label="Final wrong")
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="0.35")
    ax.set_xlabel(f"Spatial margin at L{match_layer}", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(frameon=False, fontsize=16)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _trajectory_stats(rows: Sequence[Mapping[str, Any]], metric: str, group: str, layers: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    means: List[float] = []
    sems: List[float] = []
    for L in layers:
        vals = [float(r[metric]) for r in rows if int(r["layer"]) == int(L) and str(r["group"]) == group]
        means.append(mean_or_nan(vals))
        sems.append(sem(vals))
    return np.asarray(means, np.float64), np.asarray(sems, np.float64)


def plot_matched_trajectories(
    path: Path,
    pair_layer_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    match_layer: int,
) -> None:
    x = np.asarray(list(layers), np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=240)

    for ax, metric, ylabel in [
        (axes[0], "spatial_margin", "Spatial margin"),
        (axes[1], "decision_margin", "Decision margin"),
    ]:
        for group, label in [("correct", "Final correct"), ("wrong", "Final wrong")]:
            y, e = _trajectory_stats(pair_layer_rows, metric, group, layers)
            line, = ax.plot(x, y, marker="o", markersize=4.5, linewidth=2.1, label=label)
            ax.fill_between(x, y - e, y + e, color=line.get_color(), alpha=0.16, linewidth=0)
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35")
        ax.axvline(match_layer, linestyle=":", linewidth=1.2, color="0.45")
        ax.set_xlabel("Decoder layer", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, alpha=0.18)
        ax.legend(frameon=False, fontsize=16)

    fig.tight_layout(pad=0.6, w_pad=1.1)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    path: Path,
    sample_by_layer: Mapping[Tuple[int, int], Mapping[str, Any]],
    test_items: Sequence[Mapping[str, Any]],
    match_layer: int,
    decision_layer: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.3), dpi=240)
    for final_correct, label, marker in [(1, "Final correct", "o"), (0, "Final wrong", "x")]:
        xs: List[float] = []
        ys: List[float] = []
        for m in test_items:
            sid = int(m["sid"])
            a = sample_by_layer.get((sid, match_layer))
            b = sample_by_layer.get((sid, decision_layer))
            if a is None or b is None:
                continue
            if int(a["final_correct"]) != final_correct:
                continue
            if float(a["spatial_margin"]) <= 0:
                continue
            xs.append(float(a["spatial_margin"]))
            ys.append(float(b["decision_margin"]))
        ax.scatter(xs, ys, s=24, alpha=0.60, marker=marker, label=label)
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35")
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="0.35")
    ax.set_xlabel(f"Spatial margin at L{match_layer}", fontsize=16)
    ax.set_ylabel(f"Decision margin at L{decision_layer}", fontsize=16)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    a = parse_args()
    if not 0 <= a.gray_value <= 255:
        raise ValueError("--gray-value must be in [0,255]")
    if not 0.0 < a.high_margin_quantile <= 1.0:
        raise ValueError("--high-margin-quantile must be in (0,1]")

    layers = sorted(set(int(x) for x in a.layers))
    if a.match_layer not in layers:
        raise ValueError("--match-layer must be included in --layers")
    if a.decision_layer not in layers:
        raise ValueError("--decision-layer must be included in --layers")

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

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    synth_root = Path(a.synthetic_root)
    synth_labels = resolve_synthetic_labels(synth_root, a.synthetic_labels)
    synthetic = load_synthetic400(synth_root, synth_labels)
    synthetic = stratified_cap(synthetic, a.synthetic_max_samples, a.seed + 101)
    synth_map = assign_relation_balanced_mappings(synthetic, a.seed + 1001)
    synth_counts = {r: sum(1 for x in synthetic if x["gt"] == r) for r in REL}

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

    coco_all = stratified_cap(coco_all, a.coco_max_samples, a.seed + 211)
    coco_map = assign_relation_balanced_mappings(coco_all, a.seed + 2003)
    coco_train, coco_test = joint_stratified_split(coco_all, coco_map, a.train_frac, a.seed + 3007)

    split_rows: List[Dict[str, Any]] = []
    for split_name, arr in [("train", coco_train), ("test", coco_test)]:
        for rel in REL:
            for letter in LETTERS:
                n = sum(1 for x in arr if x["gt"] == rel and coco_map[int(x["sid"])][rel] == letter)
                split_rows.append({"split": split_name, "gt": rel, "gt_option": letter, "n": n})
    write_csv(out / "split_audit.csv", split_rows)

    print(f"[Synthetic] n={len(synthetic)} counts={synth_counts}")
    print(f"[COCO] all={len(coco_all)} train={len(coco_train)} test={len(coco_test)} train_frac={a.train_frac:.2f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
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
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor)

        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        if min(layers) < 0 or max(layers) >= len(decoder_layers):
            raise ValueError(f"Requested layers={layers}; decoder has {len(decoder_layers)} layers")

        relation_token_map = base.relation_token_variants(processor.tokenizer)
        option_token_map = build_option_token_map(processor.tokenizer)

        print("=" * 112)
        print("SPATIAL-MARGIN MATCHED DIAGNOSIS")
        print("=" * 112)
        print(f"decoder={decoder_path}")
        print(f"layers={layers} | match=L{a.match_layer} | decision=L{a.decision_layer}")
        print("Spatial reader: Synthetic-400 only. Decision reader: COCO train-30 only. Diagnosis: COCO test-70 only.")
        print()

        # --------------------------------------------------------------
        # 1) Synthetic q_obj -> relation reader
        # --------------------------------------------------------------
        synth_obj: Dict[int, Dict[int, np.ndarray]] = {}
        synth_valid: List[Dict[str, Any]] = []
        for m in tqdm(synthetic, desc="Synthetic-400 spatial states"):
            sid = int(m["sid"])
            prompt = build_randmap_prompt(m["subject"], m["reference"], synth_map[sid])
            image = None
            try:
                image = Image.open(m["image_path"]).convert("RGB")
                q_obj, _q_last = collect_realgray_states_from_image(
                    model=model, processor=processor, decoder_layers=decoder_layers,
                    image=image, prompt=prompt, subject=m["subject"], reference=m["reference"],
                    layers=layers, object_state=a.object_state,
                    relation_token_map=relation_token_map, gray_value=a.gray_value, device=device,
                )
                synth_obj[sid] = q_obj
                synth_valid.append(m)
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "synthetic_spatial_reader", "sid": sid,
                    "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                gc.collect()

        spatial_reader = fit_centered_centroid_reader(
            synth_valid, synth_obj, layers,
            label_fn=lambda m: m["gt"], classes=REL,
        )

        # --------------------------------------------------------------
        # 2) Extract COCO train+test q_obj/q_last and final prediction
        # --------------------------------------------------------------
        coco_obj: Dict[int, Dict[int, np.ndarray]] = {}
        coco_last: Dict[int, Dict[int, np.ndarray]] = {}
        final_meta: Dict[int, Dict[str, Any]] = {}

        for split_name, arr in [("train", coco_train), ("test", coco_test)]:
            for m in tqdm(arr, desc=f"COCO {split_name} states"):
                sid = int(m["sid"])
                mp = coco_map[sid]
                prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
                image = batch = None
                try:
                    image = base.record_image(rec_by_sid[sid])
                    if hasattr(image, "convert"):
                        image = image.convert("RGB")
                    batch = make_batch(processor, device, image, prompt)
                    final_sc = first_step_scores(model, batch, option_token_map)

                    q_obj, q_last = collect_realgray_states_from_image(
                        model=model, processor=processor, decoder_layers=decoder_layers,
                        image=image, prompt=prompt, subject=m["subject"], reference=m["reference"],
                        layers=layers, object_state=a.object_state,
                        relation_token_map=relation_token_map, gray_value=a.gray_value, device=device,
                    )
                    coco_obj[sid] = q_obj
                    coco_last[sid] = q_last
                    gt_option = mp[m["gt"]]
                    final_meta[sid] = {
                        "split": split_name,
                        "gt": m["gt"],
                        "gt_option": gt_option,
                        "mapping": mapping_string(mp),
                        "final_prediction": final_sc["prediction"],
                        "final_correct": int(final_sc["prediction"] == gt_option),
                        "final_gt_logprob": float(final_sc["letter_logprobs"][gt_option]),
                        "final_gt_logit": float(final_sc["letter_logits"][gt_option]),
                    }
                except Exception as e:
                    traj.append_jsonl(err_path, {
                        "phase": f"coco_{split_name}_states", "sid": sid,
                        "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                    })
                    raise
                finally:
                    if image is not None:
                        with contextlib.suppress(Exception):
                            image.close()
                    del batch
                    gc.collect()

        # --------------------------------------------------------------
        # 3) Fit held-out decision/option reader from COCO TRAIN-30 q_last
        # --------------------------------------------------------------
        decision_reader = fit_centered_centroid_reader(
            coco_train, coco_last, layers,
            label_fn=lambda m: coco_map[int(m["sid"])][str(m["gt"])],
            classes=LETTERS,
        )

        # --------------------------------------------------------------
        # 4) TEST-70 per-sample/per-layer margins
        # --------------------------------------------------------------
        per_rows: List[Dict[str, Any]] = []
        sample_by_layer: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for m in coco_test:
            sid = int(m["sid"])
            meta = final_meta[sid]
            for L in layers:
                s_sp = reader_scores(coco_obj[sid][L], spatial_reader[L], REL)
                s_dec = reader_scores(coco_last[sid][L], decision_reader[L], LETTERS)
                gt = str(m["gt"])
                gt_option = str(meta["gt_option"])
                row: Dict[str, Any] = {
                    "sid": sid,
                    "layer": int(L),
                    "gt": gt,
                    "gt_display": disp_rel(gt),
                    "gt_option": gt_option,
                    "mapping": meta["mapping"],
                    "final_prediction": meta["final_prediction"],
                    "final_correct": int(meta["final_correct"]),
                    "spatial_margin": margin_from_scores(s_sp, gt, REL),
                    "decision_margin": margin_from_scores(s_dec, gt_option, LETTERS),
                    "spatial_pred": max(REL, key=lambda r: s_sp[r]),
                    "decision_pred": max(LETTERS, key=lambda z: s_dec[z]),
                }
                for r in REL:
                    row[f"spatial_score_{r}"] = float(s_sp[r])
                for z in LETTERS:
                    row[f"decision_score_{z}"] = float(s_dec[z])
                per_rows.append(row)
                sample_by_layer[(sid, int(L))] = row

        write_csv(out / "per_sample_layer.csv", per_rows)

        rows_match = [r for r in per_rows if int(r["layer"]) == a.match_layer]
        sample_at_match = {int(r["sid"]): r for r in rows_match}

        # --------------------------------------------------------------
        # 5) Matching
        # --------------------------------------------------------------
        pairs = greedy_margin_match(coco_test, sample_at_match, coco_map, a.max_margin_gap)
        if not pairs:
            raise RuntimeError(
                "No matched pairs. Check whether there are spatial-correct final-wrong samples, "
                "or loosen --max-margin-gap."
            )
        write_csv(out / "matched_pairs.csv", pairs)

        pair_layer_rows: List[Dict[str, Any]] = []
        for p in pairs:
            for group, sid_key in [("wrong", "wrong_sid"), ("correct", "correct_sid")]:
                sid = int(p[sid_key])
                for L in layers:
                    r = sample_by_layer[(sid, int(L))]
                    pair_layer_rows.append({
                        "pair_id": int(p["pair_id"]),
                        "group": group,
                        "sid": sid,
                        "gt": p["gt"],
                        "gt_option": p["gt_option"],
                        "layer": int(L),
                        "spatial_margin": float(r["spatial_margin"]),
                        "decision_margin": float(r["decision_margin"]),
                    })
        write_csv(out / "matched_pair_layer.csv", pair_layer_rows)

        matched_summary: List[Dict[str, Any]] = []
        for L in layers:
            row: Dict[str, Any] = {"layer": int(L), "n_pairs": len(pairs)}
            for metric in ["spatial_margin", "decision_margin"]:
                c = [float(r[metric]) for r in pair_layer_rows if int(r["layer"]) == L and r["group"] == "correct"]
                w = [float(r[metric]) for r in pair_layer_rows if int(r["layer"]) == L and r["group"] == "wrong"]
                row[f"correct_{metric}_mean"] = mean_or_nan(c)
                row[f"correct_{metric}_sem"] = sem(c)
                row[f"wrong_{metric}_mean"] = mean_or_nan(w)
                row[f"wrong_{metric}_sem"] = sem(w)
                row[f"paired_{metric}_diff_correct_minus_wrong"] = mean_or_nan(np.asarray(c) - np.asarray(w))
            matched_summary.append(row)
        write_csv(out / "matched_summary.csv", matched_summary)

        # --------------------------------------------------------------
        # 6) High-margin subset
        # --------------------------------------------------------------
        spatial_correct = [r for r in rows_match if float(r["spatial_margin"]) > 0]
        margins = np.asarray([float(r["spatial_margin"]) for r in spatial_correct], np.float64)
        # Top q fraction => threshold at (1-q) quantile.
        threshold = float(np.quantile(margins, 1.0 - a.high_margin_quantile)) if len(margins) else float("nan")
        high = [r for r in spatial_correct if float(r["spatial_margin"]) >= threshold]
        high_wrong = [r for r in high if int(r["final_correct"]) == 0]
        high_correct = [r for r in high if int(r["final_correct"]) == 1]
        high_summary = {
            "match_layer": int(a.match_layer),
            "high_margin_top_fraction": float(a.high_margin_quantile),
            "threshold": threshold,
            "n_spatial_correct_test": len(spatial_correct),
            "n_high_margin": len(high),
            "n_high_margin_final_correct": len(high_correct),
            "n_high_margin_final_wrong": len(high_wrong),
            "high_margin_final_wrong_rate": float(len(high_wrong) / len(high)) if high else float("nan"),
            "mean_margin_high_correct": mean_or_nan([float(r["spatial_margin"]) for r in high_correct]),
            "mean_margin_high_wrong": mean_or_nan([float(r["spatial_margin"]) for r in high_wrong]),
        }
        (out / "high_margin_summary.json").write_text(
            json.dumps(high_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # --------------------------------------------------------------
        # 7) Figures
        # --------------------------------------------------------------
        plot_margin_distribution(out / "figure_spatial_margin_distribution.png", rows_match, a.match_layer)
        plot_matched_trajectories(
            out / "figure_matched_trajectories.png", pair_layer_rows, layers, a.match_layer
        )
        plot_scatter(
            out / "figure_spatial_vs_decision_scatter.png",
            sample_by_layer, coco_test, a.match_layer, a.decision_layer,
        )

        # --------------------------------------------------------------
        # 8) Console diagnostics
        # --------------------------------------------------------------
        c_pre = [float(r["spatial_margin"]) for r in rows_match if int(r["final_correct"]) == 1 and float(r["spatial_margin"]) > 0]
        w_pre = [float(r["spatial_margin"]) for r in rows_match if int(r["final_correct"]) == 0 and float(r["spatial_margin"]) > 0]
        matched_c = [float(p["correct_spatial_margin"]) for p in pairs]
        matched_w = [float(p["wrong_spatial_margin"]) for p in pairs]
        gaps = [float(p["abs_margin_gap"]) for p in pairs]

        def layer_summary(L: int) -> Mapping[str, Any]:
            return next(r for r in matched_summary if int(r["layer"]) == int(L))

        ms = layer_summary(a.match_layer)
        ds = layer_summary(a.decision_layer)

        print("\n" + "=" * 112)
        print("RESULT SUMMARY")
        print("=" * 112)
        print(f"TEST-70 n={len(coco_test)}")
        print(f"L{a.match_layer} spatial-correct: final-correct={len(c_pre)} final-wrong={len(w_pre)}")
        print(f"Pre-match spatial margin: correct={mean_or_nan(c_pre):+.4f} wrong={mean_or_nan(w_pre):+.4f}")
        print(f"Matched pairs: n={len(pairs)} | mean |margin gap|={mean_or_nan(gaps):.6f} | max={max(gaps):.6f}")
        print(
            f"Matched L{a.match_layer} spatial margin: "
            f"correct={float(ms['correct_spatial_margin_mean']):+.4f} "
            f"wrong={float(ms['wrong_spatial_margin_mean']):+.4f}"
        )
        print(
            f"Matched L{a.decision_layer} decision margin: "
            f"correct={float(ds['correct_decision_margin_mean']):+.4f} "
            f"wrong={float(ds['wrong_decision_margin_mean']):+.4f} "
            f"diff={float(ds['paired_decision_margin_diff_correct_minus_wrong']):+.4f}"
        )
        print(
            f"High-margin top {100*a.high_margin_quantile:.0f}% among spatial-correct: "
            f"n={len(high)}, final-wrong={len(high_wrong)} "
            f"({100*high_summary['high_margin_final_wrong_rate']:.1f}%)"
        )
        print(f"[SAVED] {out / 'figure_matched_trajectories.png'}")
        print(f"[SAVED] {out / 'figure_spatial_vs_decision_scatter.png'}")
        print(f"[SAVED] {out / 'matched_pairs.csv'}")

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "layers": layers,
            "match_layer": int(a.match_layer),
            "decision_layer": int(a.decision_layer),
            "train_frac": float(a.train_frac),
            "seed": int(a.seed),
            "gray_value": int(a.gray_value),
            "object_state": a.object_state,
            "synthetic_n": len(synthetic),
            "synthetic_counts": synth_counts,
            "coco_all_n": len(coco_all),
            "coco_train_n": len(coco_train),
            "coco_test_n": len(coco_test),
            "spatial_reader": "Synthetic-400 centered cosine centroid reader on Real-Gray object-pair residual",
            "decision_reader": "COCO train-30 centered cosine centroid reader on Real-Gray prompt-last residual, grouped by mapped GT A/B/C/D",
            "test_diagnosis": "COCO held-out test-70 only",
            "matching": "exact (GT relation, mapped GT option), nearest match-layer spatial margin, no replacement",
            "n_matched_pairs": len(pairs),
            "mean_abs_match_gap": mean_or_nan(gaps),
            "high_margin": high_summary,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

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
