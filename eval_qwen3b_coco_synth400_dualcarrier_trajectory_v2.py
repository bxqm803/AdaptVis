#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic-400 dual-carrier trajectory diagnosis for COCO.

Goal
====
Make the representation/decision trajectory analysis consistent with the
paper definition that considers TWO carriers at every decoder layer:

  object-pair carrier:
      q_obj,L = (h_real_sub-h_real_ref) - (h_gray_sub-h_gray_ref)

  prompt-final carrier:
      q_last,L = h_real_last - h_gray_last

All readers are learned ONLY on Synthetic-400. COCO is evaluation only.
Synthetic examples receive independently randomized relation->A/B/C/D maps,
so we can learn both:

  (1) semantic spatial readers: left/right/above/below
  (2) mapped-option readers:    A/B/C/D

For each layer we evaluate both carriers. The recommended aggregation is
`global_best`: choose the stronger carrier at that layer on Synthetic-400
and use that same carrier for every COCO example. This mirrors the paper's
"max across carriers" idea without choosing a carrier after seeing a COCO
sample's GT label. `sample_max` is provided only as an exploratory option.

Outputs
=======
  per_sample_layer.csv
  synthetic_carrier_selection.csv
  figure_dualcarrier_trajectories.png
  figure_dualcarrier_left_right_gap.png
  metadata.json

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u eval_qwen3b_coco_synth400_dualcarrier_trajectory_v2.py \
  --model qwen-3b \
  --layers 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --synthetic-root synthetic_shapes_4dir_400 \
  --carrier-mode global_best \
  --output-dir output/qwen3b_coco_synth400_dualcarrier_v2 \
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
SCRIPT_VERSION = "qwen3b-coco-synth400-dualcarrier-trajectory-v2"
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
    p.add_argument("--carrier-mode", default="global_best", choices=["global_best", "sample_max"],
                   help="global_best: choose carrier per layer using Synthetic-400 readout accuracy; sample_max: per-sample max target margin (exploratory).")
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
# Synthetic carrier selection + plotting
# -----------------------------------------------------------------------------

def reader_accuracy(
    items: Sequence[Mapping[str, Any]],
    states: Mapping[int, Mapping[int, np.ndarray]],
    reader: Mapping[int, Mapping[str, Any]],
    layers: Sequence[int],
    label_fn,
    classes: Sequence[str],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for L in layers:
        ok = n = 0
        for m in items:
            sid = int(m["sid"])
            if sid not in states or int(L) not in states[sid]:
                continue
            scores = reader_scores(states[sid][int(L)], reader[int(L)], classes)
            pred = max(classes, key=lambda c: scores[c])
            ok += int(pred == str(label_fn(m)))
            n += 1
        out[int(L)] = float(ok / n) if n else float("nan")
    return out


def sem_arr(vals: Sequence[float]) -> float:
    a = np.asarray(list(vals), np.float64)
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def summarize_groups(per_rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[int, Dict[str, Tuple[float, float, int]]]:
    layers = sorted({int(r["layer"]) for r in per_rows})
    out: Dict[int, Dict[str, Tuple[float, float, int]]] = {}
    for L in layers:
        out[L] = {}
        for name, fc in [("correct", 1), ("wrong", 0)]:
            vals = [float(r[metric]) for r in per_rows if int(r["layer"]) == L and int(r["final_correct"]) == fc]
            out[L][name] = (float(np.mean(vals)) if vals else float("nan"), sem_arr(vals), len(vals))
    return out


def plot_dualcarrier_trajectories(path: Path, per_rows: Sequence[Mapping[str, Any]]) -> None:
    sp = summarize_groups(per_rows, "spatial_margin")
    dc = summarize_groups(per_rows, "decision_margin")
    layers = sorted(sp)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=240)
    for ax, summ, ylabel, title in [
        (axes[0], sp, "Spatial margin", "Intermediate spatial evidence"),
        (axes[1], dc, "Decision margin", "Downstream decision evidence"),
    ]:
        for name, label in [("correct", "Final correct"), ("wrong", "Final wrong")]:
            y = np.asarray([summ[L][name][0] for L in layers], np.float64)
            e = np.asarray([summ[L][name][1] for L in layers], np.float64)
            line, = ax.plot(layers, y, marker="o", linewidth=2.3, markersize=4.8, label=label)
            ax.fill_between(layers, y-e, y+e, color=line.get_color(), alpha=0.16, linewidth=0)
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35")
        ax.set_xlabel("Decoder layer", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.18)
        ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(pad=0.7, w_pad=1.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_left_right_gap(path: Path, per_rows: Sequence[Mapping[str, Any]]) -> None:
    # Restrict to GT=left so positive means evidence for the GT left over its semantic opposite right.
    rows = [r for r in per_rows if str(r["gt"]) == "left"]
    sp = summarize_groups(rows, "spatial_lr_gap")
    dc = summarize_groups(rows, "decision_lr_gap")
    layers = sorted(sp)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=240)
    for ax, summ, ylabel, title in [
        (axes[0], sp, "Left - Right spatial gap", "GT = left: spatial evidence"),
        (axes[1], dc, "Left - Right decision gap", "GT = left: decision evidence"),
    ]:
        for name, label in [("correct", "Final correct"), ("wrong", "Final wrong")]:
            y = np.asarray([summ[L][name][0] for L in layers], np.float64)
            e = np.asarray([summ[L][name][1] for L in layers], np.float64)
            line, = ax.plot(layers, y, marker="o", linewidth=2.3, markersize=4.8, label=label)
            ax.fill_between(layers, y-e, y+e, color=line.get_color(), alpha=0.16, linewidth=0)
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35")
        ax.set_xlabel("Decoder layer", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.18)
        ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(pad=0.7, w_pad=1.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    layers = sorted(set(int(x) for x in a.layers))
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

    # ------------------------------ data ------------------------------
    synth_root = Path(a.synthetic_root)
    synth_labels = resolve_synthetic_labels(synth_root, a.synthetic_labels)
    synthetic = load_synthetic400(synth_root, synth_labels)
    synthetic = stratified_cap(synthetic, a.synthetic_max_samples, a.seed + 101)
    synth_map = assign_relation_balanced_mappings(synthetic, a.seed + 1001)

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
        coco_all.append({"sid": sid, "gt": gt, "subject": str(p["subject"]), "reference": str(p["reference"])})
    coco_all = stratified_cap(coco_all, a.coco_max_samples, a.seed + 211)
    coco_map = assign_relation_balanced_mappings(coco_all, a.seed + 2003)

    print(f"[Synthetic] n={len(synthetic)}")
    print(f"[COCO eval only] n={len(coco_all)}")

    # ----------------------------- model ------------------------------
    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)
    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name), low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code, device_map={"": a.device},
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

        # ---------------- Synthetic states: BOTH carriers ----------------
        synth_obj: Dict[int, Dict[int, np.ndarray]] = {}
        synth_last: Dict[int, Dict[int, np.ndarray]] = {}
        synth_valid: List[Dict[str, Any]] = []
        for m in tqdm(synthetic, desc="Synthetic-400 obj+last states"):
            sid = int(m["sid"])
            prompt = build_randmap_prompt(m["subject"], m["reference"], synth_map[sid])
            image = None
            try:
                image = Image.open(m["image_path"]).convert("RGB")
                q_obj, q_last = collect_realgray_states_from_image(
                    model=model, processor=processor, decoder_layers=decoder_layers,
                    image=image, prompt=prompt, subject=m["subject"], reference=m["reference"],
                    layers=layers, object_state=a.object_state, relation_token_map=relation_token_map,
                    gray_value=a.gray_value, device=device,
                )
                synth_obj[sid] = q_obj
                synth_last[sid] = q_last
                synth_valid.append(m)
            except Exception as e:
                traj.append_jsonl(err_path, {"phase": "synthetic", "sid": sid, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception): image.close()
                gc.collect()

        rel_label = lambda m: str(m["gt"])
        opt_label = lambda m: str(synth_map[int(m["sid"])][str(m["gt"])])
        sp_obj_reader = fit_centered_centroid_reader(synth_valid, synth_obj, layers, rel_label, REL)
        sp_last_reader = fit_centered_centroid_reader(synth_valid, synth_last, layers, rel_label, REL)
        op_obj_reader = fit_centered_centroid_reader(synth_valid, synth_obj, layers, opt_label, LETTERS)
        op_last_reader = fit_centered_centroid_reader(synth_valid, synth_last, layers, opt_label, LETTERS)

        sp_obj_acc = reader_accuracy(synth_valid, synth_obj, sp_obj_reader, layers, rel_label, REL)
        sp_last_acc = reader_accuracy(synth_valid, synth_last, sp_last_reader, layers, rel_label, REL)
        op_obj_acc = reader_accuracy(synth_valid, synth_obj, op_obj_reader, layers, opt_label, LETTERS)
        op_last_acc = reader_accuracy(synth_valid, synth_last, op_last_reader, layers, opt_label, LETTERS)

        sp_choice = {L: ("obj" if sp_obj_acc[L] >= sp_last_acc[L] else "last") for L in layers}
        op_choice = {L: ("obj" if op_obj_acc[L] >= op_last_acc[L] else "last") for L in layers}
        selection_rows = []
        for L in layers:
            selection_rows.append({
                "layer": L,
                "spatial_obj_acc": sp_obj_acc[L], "spatial_last_acc": sp_last_acc[L], "spatial_choice": sp_choice[L],
                "option_obj_acc": op_obj_acc[L], "option_last_acc": op_last_acc[L], "option_choice": op_choice[L],
            })
        write_csv(out / "synthetic_carrier_selection.csv", selection_rows)

        print("[Synthetic carrier choice]")
        for L in layers:
            print(f"  L{L:02d}: spatial={sp_choice[L]:>4s} ({sp_obj_acc[L]:.3f}/{sp_last_acc[L]:.3f}) | option={op_choice[L]:>4s} ({op_obj_acc[L]:.3f}/{op_last_acc[L]:.3f})")

        # -------------------------- COCO eval ---------------------------
        per_rows: List[Dict[str, Any]] = []
        for m in tqdm(coco_all, desc="COCO dual-carrier evaluation"):
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
                    layers=layers, object_state=a.object_state, relation_token_map=relation_token_map,
                    gray_value=a.gray_value, device=device,
                )
                gt = str(m["gt"])
                gt_option = str(mp[gt])
                final_pred = str(final_sc["prediction"])
                for L in layers:
                    sp_obj = reader_scores(q_obj[L], sp_obj_reader[L], REL)
                    sp_last = reader_scores(q_last[L], sp_last_reader[L], REL)
                    op_obj = reader_scores(q_obj[L], op_obj_reader[L], LETTERS)
                    op_last = reader_scores(q_last[L], op_last_reader[L], LETTERS)

                    sp_obj_margin = margin_from_scores(sp_obj, gt, REL)
                    sp_last_margin = margin_from_scores(sp_last, gt, REL)
                    op_obj_margin = margin_from_scores(op_obj, gt_option, LETTERS)
                    op_last_margin = margin_from_scores(op_last, gt_option, LETTERS)

                    if a.carrier_mode == "sample_max":
                        sp_carrier = "obj" if sp_obj_margin >= sp_last_margin else "last"
                        op_carrier = "obj" if op_obj_margin >= op_last_margin else "last"
                    else:
                        sp_carrier = sp_choice[L]
                        op_carrier = op_choice[L]
                    sp = sp_obj if sp_carrier == "obj" else sp_last
                    op = op_obj if op_carrier == "obj" else op_last

                    mapped_left = mp["left"]
                    mapped_right = mp["right"]
                    row: Dict[str, Any] = {
                        "sid": sid, "layer": L, "gt": gt, "gt_display": disp_rel(gt),
                        "gt_option": gt_option, "mapping": mapping_string(mp),
                        "final_prediction": final_pred, "final_correct": int(final_pred == gt_option),
                        "spatial_carrier": sp_carrier, "decision_carrier": op_carrier,
                        "spatial_margin": margin_from_scores(sp, gt, REL),
                        "decision_margin": margin_from_scores(op, gt_option, LETTERS),
                        "spatial_obj_margin": sp_obj_margin, "spatial_last_margin": sp_last_margin,
                        "decision_obj_margin": op_obj_margin, "decision_last_margin": op_last_margin,
                        "spatial_pred": max(REL, key=lambda r: sp[r]),
                        "decision_pred": max(LETTERS, key=lambda z: op[z]),
                        "spatial_lr_gap": float(sp["left"] - sp["right"]),
                        "decision_lr_gap": float(op[mapped_left] - op[mapped_right]),
                    }
                    for r in REL:
                        row[f"spatial_score_{r}"] = float(sp[r])
                        row[f"spatial_obj_score_{r}"] = float(sp_obj[r])
                        row[f"spatial_last_score_{r}"] = float(sp_last[r])
                    for z in LETTERS:
                        row[f"decision_score_{z}"] = float(op[z])
                        row[f"decision_obj_score_{z}"] = float(op_obj[z])
                        row[f"decision_last_score_{z}"] = float(op_last[z])
                    per_rows.append(row)
            except Exception as e:
                traj.append_jsonl(err_path, {"phase": "coco", "sid": sid, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception): image.close()
                del batch
                gc.collect()

        write_csv(out / "per_sample_layer.csv", per_rows)
        plot_dualcarrier_trajectories(out / "figure_dualcarrier_trajectories.png", per_rows)
        plot_left_right_gap(out / "figure_dualcarrier_left_right_gap.png", per_rows)

        meta = {
            "script_version": SCRIPT_VERSION,
            "model": a.model, "repo_id": spec.repo_id, "decoder_path": decoder_path,
            "layers": layers, "carrier_mode": a.carrier_mode,
            "synthetic_n": len(synth_valid), "coco_eval_n": len(coco_all),
            "reader_training": "Synthetic-400 only for BOTH spatial and mapped-option readers",
            "spatial_labels": list(REL), "option_labels": list(LETTERS),
            "spatial_carrier_choice": {str(L): sp_choice[L] for L in layers},
            "option_carrier_choice": {str(L): op_choice[L] for L in layers},
            "note": "global_best chooses one carrier per layer from Synthetic-400 accuracy and applies it to every COCO example; sample_max is exploratory and GT-aware at evaluation time.",
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] {out / 'figure_dualcarrier_trajectories.png'}")
        print(f"[SAVED] {out / 'figure_dualcarrier_left_right_gap.png'}")
        print(f"[SAVED] {out / 'per_sample_layer.csv'}")
        print(f"[SAVED] {out / 'synthetic_carrier_selection.csv'}")
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
