#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introduction diagnostic: similar intermediate spatial evidence -> divergent downstream decisions.

This script tries TWO visualizations on the SAME automatically selected pair.

VIEW 1 (recommended / scientifically clean)
--------------------------------------------
Two panels with different, explicitly named observables:

  A) Intermediate spatial evidence
     Synthetic-400 learns a four-way spatial reader at every layer for TWO carriers:

       q_obj,L  = (h_real_sub-h_real_ref) - (h_gray_sub-h_gray_ref)
       q_last,L =  h_real_last - h_gray_last

     Synthetic-400 also chooses one carrier per layer (obj or last), globally for all
     COCO examples.  The plotted score is the four-way GT spatial margin:

       S_i,L = score(GT) - max_{r != GT} score(r)

     Therefore S>0 means the GT relation is the dominant spatial relation.

  B) Downstream decision evidence
     No learned decision reader is used.  We apply the model's own final norm + LM
     head to the REAL prompt-last hidden state at every decoder layer (logit lens).
     For target relation t and its semantic opposite o, under sample-specific
     relation->A/B/C/D mapping pi_i:

       D_i,L = logit_L(pi_i(t)) - logit_L(pi_i(o))

     D>0 favors the mapped target option; D<0 favors the mapped opposite option.

VIEW 2 (exploratory single-axis story plot)
-------------------------------------------
A stage-specific preference curve with TWO sample lines:

  - before/at --transition-layer: spatial preference from a softmax over the four
    Synthetic-400 spatial-reader cosine scores;
  - after  --transition-layer: decision preference from a softmax over the four
    A/B/C/D logit-lens scores.

Both are bounded GT-vs-opposite preference gaps in [-1,1], but the metric changes
at the marked transition.  The figure labels this explicitly; use View 1 for the
main quantitative claim.

PAIR SEARCH
-----------
Default target = left, opposite = right.  The script searches all COCO examples
for one final-correct target sample and one final-wrong->opposite sample such that:

  * both have positive / similar mid-layer spatial trajectories;
  * optionally their pre-decision logit-lens trajectories are similar;
  * their late decision trajectories strongly diverge.

Pair selection never uses image identity/case-study metadata.  Criteria are saved
for auditing in top_pairs.csv.

All spatial readers and carrier choices are learned ONLY from Synthetic-400.
COCO is evaluation only.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u eval_qwen3b_intro_spatial_to_decision_two_views_v1.py \
  --model qwen-3b \
  --layers 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --synthetic-root synthetic_shapes_4dir_400 \
  --target-relation left \
  --opposite-relation right \
  --mid-layers 19 20 21 22 23 24 25 \
  --predecision-layers 22 23 24 25 26 \
  --late-layers 27 28 29 30 31 32 \
  --transition-layer 26 \
  --output-dir output/qwen3b_intro_spatial_to_decision_two_views_v1 \
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


REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
DISPLAY_REL = {"left": "left", "right": "right", "above": "on", "below": "under"}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
EPS = 1e-8
SCRIPT_VERSION = "qwen3b-intro-spatial-to-decision-two-views-v1"


def canonical_relation(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    table = {
        "left": "left", "left_of": "left",
        "right": "right", "right_of": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    if s not in table:
        raise ValueError(f"Unknown relation: {x!r}")
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
    p.add_argument("--layers", nargs="+", type=int, default=list(range(18, 33)))
    p.add_argument("--target-relation", default="left")
    p.add_argument("--opposite-relation", default="right")
    p.add_argument("--mid-layers", nargs="+", type=int, default=[19,20,21,22,23,24,25])
    p.add_argument("--predecision-layers", nargs="+", type=int, default=[22,23,24,25,26])
    p.add_argument("--late-layers", nargs="+", type=int, default=[27,28,29,30,31,32])
    p.add_argument("--transition-layer", type=int, default=26)
    p.add_argument("--min-mid-positive-frac", type=float, default=0.80)
    p.add_argument("--min-mid-mean-margin", type=float, default=0.02)
    p.add_argument("--max-mid-rmse", type=float, default=-1.0,
                   help="Hard cap on spatial trajectory RMSE; <0 disables.")
    p.add_argument("--max-predecision-rmse", type=float, default=-1.0,
                   help="Hard cap on pre-decision logit-lens RMSE; <0 disables.")
    p.add_argument("--min-late-separation", type=float, default=0.20,
                   help="Require mean late decision(correct)-mean late decision(wrong) >= this.")
    p.add_argument("--require-late-sign-split", action="store_true", default=True,
                   help="Require correct late mean >0 and wrong late mean <0.")
    p.add_argument("--no-require-late-sign-split", dest="require_late_sign_split", action="store_false")
    p.add_argument("--allow-different-option-pair", action="store_true",
                   help="By default require target/opposite to map to the same two letters in both samples.")
    p.add_argument("--predecision-weight", type=float, default=0.50)
    p.add_argument("--late-separation-weight", type=float, default=0.15)
    p.add_argument("--spatial-softmax-temp", type=float, default=0.10,
                   help="Only for exploratory piecewise preference plot.")
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--synthetic-max-samples", type=int, default=0)
    p.add_argument("--coco-max-samples", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def resolve_synthetic_labels(root: Path, spec: str) -> Path:
    p = Path(spec)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"Synthetic labels not found: {p}")
    return p


def resolve_synthetic_image(root: Path, field: str) -> Path:
    p = Path(str(field))
    candidates = [p] if p.is_absolute() else [root / p, root / "images" / p, root / "images" / p.name]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve synthetic image {field!r}; tried {candidates}")


def load_synthetic400(root: Path, labels_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rel = canonical_relation(obj.get("relation", obj.get("answer", obj.get("label"))))
            subject = obj.get("subject")
            reference = obj.get("reference")
            image_field = obj.get("image", obj.get("image_path", obj.get("file")))
            if subject is None or reference is None or image_field is None:
                raise KeyError(f"Synthetic row {line_no}: need image, subject, reference")
            sid_raw = obj.get("id", obj.get("sid", line_no - 1))
            try:
                sid = int(sid_raw)
            except Exception:
                sid = line_no - 1
            rows.append({
                "sid": sid, "gt": rel, "subject": str(subject), "reference": str(reference),
                "image_path": str(resolve_synthetic_image(root, str(image_field))),
            })
    seen = set()
    for i, r in enumerate(rows):
        if r["sid"] in seen:
            r["sid"] = 1_000_000 + i
        seen.add(r["sid"])
    return rows


def stratified_cap(items: Sequence[Mapping[str, Any]], max_samples: int, seed: int) -> List[Dict[str, Any]]:
    if max_samples <= 0 or len(items) <= max_samples:
        return [dict(x) for x in items]
    return traj.stratified_cap([dict(x) for x in items], max_samples, seed)


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
            rr = random.Random(seed * 1000003 + int(m["sid"]) * 9176 + i)
            rr.shuffle(rels)
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
                raise RuntimeError(f"Tokenizer cannot encode {letter}")
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


@torch.inference_mode()
def first_step_scores(model: Any, batch: Mapping[str, Any], token_map: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    outputs = model(**batch, use_cache=False, return_dict=True)
    logits = extract_logits(outputs)[0, -1].float()
    raw: Dict[str, float] = {}
    for a in LETTERS:
        idx = torch.tensor(token_map[a], device=logits.device, dtype=torch.long)
        raw[a] = float(logits.index_select(0, idx).max().detach().cpu())
    pred = max(LETTERS, key=lambda z: raw[z])
    del outputs
    return {"letter_logits": raw, "prediction": pred}


def make_gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, color=(v, v, v))


def make_batch(processor: Any, device: torch.device, image: Image.Image, prompt: str) -> Dict[str, Any]:
    return base.make_question_batch(processor=processor, image=image, question_text=prompt, device=device)


class LastTokenCapture:
    def __init__(self, decoder_layers: Sequence[Any], layers: Sequence[int]):
        self.states: Dict[int, np.ndarray] = {}
        self.handles = [decoder_layers[int(L)].register_forward_hook(self._hook(int(L))) for L in layers]

    def _hook(self, L: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            h = traj.first_tensor(output)
            if h.ndim != 3:
                raise RuntimeError(f"Expected [B,T,D] at L{L}, got {tuple(h.shape)}")
            self.states[L] = h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            return output
        return hook

    def close(self) -> None:
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def clean_forward_with_last_capture(
    *, model: Any, processor: Any, decoder_layers: Sequence[Any], batch: Mapping[str, Any],
    subject: str, reference: str, layers: Sequence[int], object_state: str,
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
            raise RuntimeError(f"Last-token capture missed layers {missing}")
        return clean, dict(cap.states)
    finally:
        cap.close()


def collect_states_from_image(
    *, model: Any, processor: Any, decoder_layers: Sequence[Any], image: Image.Image,
    prompt: str, subject: str, reference: str, layers: Sequence[int], object_state: str,
    relation_token_map: Mapping[str, Sequence[int]], gray_value: int, device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return (real-gray object-pair, real-gray last, REAL last)."""
    real = image.convert("RGB") if hasattr(image, "convert") else image
    gray = make_gray_image(real, gray_value)
    rb = gb = None
    try:
        rb = make_batch(processor, device, real, prompt)
        gb = make_batch(processor, device, gray, prompt)
        cr, last_r = clean_forward_with_last_capture(
            model=model, processor=processor, decoder_layers=decoder_layers, batch=rb,
            subject=subject, reference=reference, layers=layers, object_state=object_state,
            relation_token_map=relation_token_map,
        )
        cg, last_g = clean_forward_with_last_capture(
            model=model, processor=processor, decoder_layers=decoder_layers, batch=gb,
            subject=subject, reference=reference, layers=layers, object_state=object_state,
            relation_token_map=relation_token_map,
        )
        q_obj = {int(L): (np.asarray(cr["states"][int(L)], np.float32) - np.asarray(cg["states"][int(L)], np.float32)).astype(np.float32) for L in layers}
        q_last = {int(L): (np.asarray(last_r[int(L)], np.float32) - np.asarray(last_g[int(L)], np.float32)).astype(np.float32) for L in layers}
        real_last = {int(L): np.asarray(last_r[int(L)], np.float32).astype(np.float32) for L in layers}
        return q_obj, q_last, real_last
    finally:
        with contextlib.suppress(Exception):
            gray.close()
        del rb, gb


def fit_centered_centroid_reader(
    items: Sequence[Mapping[str, Any]], states: Mapping[int, Mapping[int, np.ndarray]],
    layers: Sequence[int], classes: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for L in layers:
        per: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
        for m in items:
            sid = int(m["sid"])
            if sid in states and int(L) in states[sid]:
                per[str(m["gt"])].append(np.asarray(states[sid][int(L)], np.float32))
        mu = {c: np.mean(np.stack(per[c], axis=0), axis=0).astype(np.float32) for c in classes}
        center = np.mean(np.stack([mu[c] for c in classes], axis=0), axis=0).astype(np.float32)
        dirs: Dict[str, np.ndarray] = {}
        for c in classes:
            v = (mu[c] - center).astype(np.float32)
            n = float(np.linalg.norm(v))
            if n < EPS:
                raise RuntimeError(f"Near-zero direction {c} at L{L}")
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


def margin(scores: Mapping[str, float], target: str, classes: Sequence[str]) -> float:
    return float(scores[target] - max(scores[c] for c in classes if c != target))


def reader_accuracy(items, states, reader, layers, classes) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for L in layers:
        ok = n = 0
        for m in items:
            sid = int(m["sid"])
            if sid not in states:
                continue
            sc = reader_scores(states[sid][L], reader[L], classes)
            ok += int(max(classes, key=lambda c: sc[c]) == str(m["gt"]))
            n += 1
        out[L] = ok / n if n else float("nan")
    return out


def _get_nested(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_norm_and_head(model: Any) -> Tuple[Any, Any, str, str]:
    norm_paths = [
        "language_model.norm",
        "model.language_model.norm",
        "model.norm",
        "transformer.norm",
    ]
    head_paths = [
        "lm_head",
        "language_model.lm_head",
        "model.lm_head",
        "model.language_model.lm_head",
    ]
    norm = norm_path = None
    head = head_path = None
    for p in norm_paths:
        x = _get_nested(model, p)
        if x is not None and callable(x):
            norm, norm_path = x, p
            break
    for p in head_paths:
        x = _get_nested(model, p)
        if x is not None and callable(x):
            head, head_path = x, p
            break
    if norm is None or head is None:
        raise RuntimeError(
            f"Could not resolve final norm/lm_head. norm={norm_path}, head={head_path}. "
            "Add your architecture path in resolve_norm_and_head()."
        )
    return norm, head, str(norm_path), str(head_path)


def module_dtype_device(module: Any, fallback_device: torch.device) -> Tuple[torch.dtype, torch.device]:
    for p in module.parameters(recurse=True):
        return p.dtype, p.device
    return torch.float32, fallback_device


@torch.inference_mode()
def logit_lens_option_logits(
    state: np.ndarray, norm: Any, lm_head: Any, token_map: Mapping[str, Sequence[int]],
    fallback_device: torch.device,
) -> Dict[str, float]:
    dtype, dev = module_dtype_device(lm_head, fallback_device)
    x = torch.as_tensor(np.asarray(state, np.float32), device=dev, dtype=dtype).view(1, 1, -1)
    y = norm(x)
    logits = lm_head(y)[0, 0].float()
    out: Dict[str, float] = {}
    for a in LETTERS:
        idx = torch.tensor([int(v) for v in token_map[a]], device=logits.device, dtype=torch.long)
        out[a] = float(logits.index_select(0, idx).max().detach().cpu())
    return out


def softmax_dict(scores: Mapping[str, float], classes: Sequence[str], temp: float = 1.0) -> Dict[str, float]:
    t = max(float(temp), 1e-6)
    v = np.asarray([float(scores[c]) / t for c in classes], np.float64)
    v -= np.max(v)
    e = np.exp(v)
    p = e / np.sum(e)
    return {c: float(p[i]) for i, c in enumerate(classes)}


def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def rows_by_sid(per_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for r in per_rows:
        out.setdefault(int(r["sid"]), {})[int(r["layer"])] = dict(r)
    return out


def trajectory(sample: Mapping[int, Mapping[str, Any]], layers: Sequence[int], key: str) -> List[float]:
    return [float(sample[int(L)][key]) for L in layers]


def find_pairs(
    per_rows: Sequence[Mapping[str, Any]], target: str, opposite: str,
    mid_layers: Sequence[int], pre_layers: Sequence[int], late_layers: Sequence[int],
    min_mid_positive_frac: float, min_mid_mean_margin: float,
    max_mid_rmse: float, max_pre_rmse: float, min_late_sep: float,
    require_late_sign_split: bool, require_same_option_pair: bool,
    pre_weight: float, late_weight: float,
) -> List[Dict[str, Any]]:
    by = rows_by_sid(per_rows)
    correct: List[Tuple[int, Dict[int, Dict[str, Any]]]] = []
    wrong: List[Tuple[int, Dict[int, Dict[str, Any]]]] = []

    needed = set(map(int, list(mid_layers) + list(pre_layers) + list(late_layers)))
    for sid, s in by.items():
        if not needed.issubset(set(s.keys())):
            continue
        anyrow = s[min(s.keys())]
        if str(anyrow["gt"]) != target:
            continue
        mids = trajectory(s, mid_layers, "spatial_margin")
        pos_frac = float(np.mean(np.asarray(mids) > 0.0))
        if pos_frac < min_mid_positive_frac or float(np.mean(mids)) < min_mid_mean_margin:
            continue
        pred_rel = str(anyrow["final_prediction_relation"])
        if pred_rel == target:
            correct.append((sid, s))
        elif pred_rel == opposite:
            wrong.append((sid, s))

    pairs: List[Dict[str, Any]] = []
    for sid_c, c in correct:
        rowc = c[min(c.keys())]
        for sid_w, w in wrong:
            roww = w[min(w.keys())]
            if require_same_option_pair:
                if (rowc["target_option"] != roww["target_option"] or
                    rowc["opposite_option"] != roww["opposite_option"]):
                    continue
            mid_c = trajectory(c, mid_layers, "spatial_margin")
            mid_w = trajectory(w, mid_layers, "spatial_margin")
            mid_dist = rmse(mid_c, mid_w)
            if max_mid_rmse >= 0 and mid_dist > max_mid_rmse:
                continue
            pre_c = trajectory(c, pre_layers, "decision_pair_margin")
            pre_w = trajectory(w, pre_layers, "decision_pair_margin")
            pre_dist = rmse(pre_c, pre_w)
            if max_pre_rmse >= 0 and pre_dist > max_pre_rmse:
                continue
            late_c = float(np.mean(trajectory(c, late_layers, "decision_pair_margin")))
            late_w = float(np.mean(trajectory(w, late_layers, "decision_pair_margin")))
            late_sep = late_c - late_w
            if late_sep < min_late_sep:
                continue
            if require_late_sign_split and not (late_c > 0.0 and late_w < 0.0):
                continue
            score = mid_dist + pre_weight * pre_dist - late_weight * late_sep
            pairs.append({
                "score": float(score), "sid_correct": sid_c, "sid_wrong": sid_w,
                "mid_spatial_rmse": mid_dist, "predecision_rmse": pre_dist,
                "late_correct_mean": late_c, "late_wrong_mean": late_w,
                "late_separation": late_sep,
                "target_option": rowc["target_option"], "opposite_option": rowc["opposite_option"],
                "same_option_pair": int(rowc["target_option"] == roww["target_option"] and rowc["opposite_option"] == roww["opposite_option"]),
            })
    pairs.sort(key=lambda x: x["score"])
    return pairs


def plot_two_stage(path: Path, best: Mapping[str, Any], by: Mapping[int, Mapping[int, Mapping[str, Any]]],
                   layers: Sequence[int], mid_layers: Sequence[int], late_layers: Sequence[int], target: str, opposite: str) -> None:
    c = by[int(best["sid_correct"])]
    w = by[int(best["sid_wrong"])]
    x = np.asarray(layers, np.int32)
    s_c = np.asarray(trajectory(c, layers, "spatial_margin"), np.float64)
    s_w = np.asarray(trajectory(w, layers, "spatial_margin"), np.float64)
    d_c = np.asarray(trajectory(c, layers, "decision_pair_margin"), np.float64)
    d_w = np.asarray(trajectory(w, layers, "decision_pair_margin"), np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), dpi=240)
    for ax in axes:
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35", zorder=0)
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=11)
    axes[0].axvspan(min(mid_layers)-0.45, max(mid_layers)+0.45, alpha=0.06, color="0.4")
    axes[1].axvspan(min(late_layers)-0.45, max(late_layers)+0.45, alpha=0.06, color="0.4")

    axes[0].plot(x, s_c, marker="o", linewidth=2.4, markersize=4.8, label=f"Final correct (sid={best['sid_correct']})")
    axes[0].plot(x, s_w, marker="o", linewidth=2.4, markersize=4.8, linestyle="--", label=f"Final wrong→{disp_rel(opposite)} (sid={best['sid_wrong']})")
    axes[0].set_title("Intermediate spatial evidence", fontsize=15)
    axes[0].set_xlabel("Decoder layer", fontsize=13)
    axes[0].set_ylabel("GT spatial margin", fontsize=13)
    axes[0].legend(frameon=False, fontsize=10, loc="best")

    axes[1].plot(x, d_c, marker="o", linewidth=2.4, markersize=4.8, label="Final correct")
    axes[1].plot(x, d_w, marker="o", linewidth=2.4, markersize=4.8, linestyle="--", label=f"Final wrong→{disp_rel(opposite)}")
    axes[1].set_title("Downstream decision evidence", fontsize=15)
    axes[1].set_xlabel("Decoder layer", fontsize=13)
    axes[1].set_ylabel(f"Mapped-{disp_rel(target)} − mapped-{disp_rel(opposite)} logit", fontsize=13)
    axes[1].legend(frameon=False, fontsize=10, loc="best")

    fig.text(0.5, 0.005,
             f"mid spatial RMSE={best['mid_spatial_rmse']:.3f}   |   pre-decision RMSE={best['predecision_rmse']:.3f}   |   late separation={best['late_separation']:.3f}",
             ha="center", va="bottom", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.045, 1, 1], w_pad=1.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_piecewise(path: Path, best: Mapping[str, Any], by: Mapping[int, Mapping[int, Mapping[str, Any]]],
                   layers: Sequence[int], transition_layer: int, target: str, opposite: str) -> None:
    c = by[int(best["sid_correct"])]
    w = by[int(best["sid_wrong"])]
    x = np.asarray(layers, np.int32)
    y_c, y_w = [], []
    for L in layers:
        key = "spatial_pair_preference" if int(L) <= int(transition_layer) else "decision_pair_preference"
        y_c.append(float(c[int(L)][key]))
        y_w.append(float(w[int(L)][key]))
    y_c = np.asarray(y_c, np.float64)
    y_w = np.asarray(y_w, np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=240)
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35", zorder=0)
    ax.axvline(float(transition_layer) + 0.5, linestyle=":", linewidth=1.4, color="0.35")
    ax.axvspan(min(layers)-0.5, transition_layer+0.5, alpha=0.05, color="0.4")
    ax.axvspan(transition_layer+0.5, max(layers)+0.5, alpha=0.025, color="0.4")
    ax.plot(x, y_c, marker="o", linewidth=2.5, markersize=5, label=f"Final correct (sid={best['sid_correct']})")
    ax.plot(x, y_w, marker="o", linewidth=2.5, markersize=5, linestyle="--", label=f"Final wrong→{disp_rel(opposite)} (sid={best['sid_wrong']})")
    ax.text(np.mean([min(layers), transition_layer]), 0.96, "spatial relation", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
    ax.text(np.mean([transition_layer+1, max(layers)]), 0.96, "decision", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
    ax.set_title("Stage-specific GT preference", fontsize=15)
    ax.set_xlabel("Decoder layer", fontsize=13)
    ax.set_ylabel(f"GT-vs-{disp_rel(opposite)} preference", fontsize=13)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    target = canonical_relation(a.target_relation)
    opposite = canonical_relation(a.opposite_relation)
    if opposite == target:
        raise ValueError("target and opposite must differ")
    if OPPOSITE.get(target) != opposite:
        print(f"[WARN] {target}/{opposite} are not canonical semantic opposites; continuing by request.")
    layers = sorted(set(map(int, a.layers)))
    for name, vals in [("mid", a.mid_layers), ("predecision", a.predecision_layers), ("late", a.late_layers)]:
        missing = sorted(set(map(int, vals)) - set(layers))
        if missing:
            raise ValueError(f"{name} layers missing from --layers: {missing}")
    if a.transition_layer not in layers:
        raise ValueError("--transition-layer must be included in --layers")
    if not 0 <= a.gray_value <= 255:
        raise ValueError("--gray-value must be in [0,255]")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    err_path = out / "errors.jsonl"

    synth_root = Path(a.synthetic_root)
    synth_labels = resolve_synthetic_labels(synth_root, a.synthetic_labels)
    synthetic = load_synthetic400(synth_root, synth_labels)
    synthetic = stratified_cap(synthetic, a.synthetic_max_samples, a.seed + 101)
    synth_map = assign_relation_balanced_mappings(synthetic, a.seed + 1001)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}
    coco: List[Dict[str, Any]] = []
    for r in records:
        sid = int(r.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        coco.append({"sid": sid, "gt": gt, "subject": str(p["subject"]), "reference": str(p["reference"])})
    coco = stratified_cap(coco, a.coco_max_samples, a.seed + 211)
    coco_map = assign_relation_balanced_mappings(coco, a.seed + 2003)

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
            raise ValueError(f"Requested {layers}; decoder has {len(decoder_layers)} layers")
        relation_token_map = base.relation_token_variants(processor.tokenizer)
        option_token_map = build_option_token_map(processor.tokenizer)
        final_norm, lm_head, norm_path, head_path = resolve_norm_and_head(model)
        print(f"[LOGIT LENS] norm={norm_path} | head={head_path}")

        # 1) Synthetic-400: fit ONLY spatial readers and choose carrier per layer.
        synth_obj: Dict[int, Dict[int, np.ndarray]] = {}
        synth_last: Dict[int, Dict[int, np.ndarray]] = {}
        synth_valid: List[Dict[str, Any]] = []
        for m in tqdm(synthetic, desc="Synthetic-400 spatial readers"):
            sid = int(m["sid"])
            prompt = build_randmap_prompt(m["subject"], m["reference"], synth_map[sid])
            image = None
            try:
                image = Image.open(m["image_path"]).convert("RGB")
                q_obj, q_last, _real_last = collect_states_from_image(
                    model=model, processor=processor, decoder_layers=decoder_layers, image=image,
                    prompt=prompt, subject=m["subject"], reference=m["reference"], layers=layers,
                    object_state=a.object_state, relation_token_map=relation_token_map,
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

        sp_obj_reader = fit_centered_centroid_reader(synth_valid, synth_obj, layers, REL)
        sp_last_reader = fit_centered_centroid_reader(synth_valid, synth_last, layers, REL)
        sp_obj_acc = reader_accuracy(synth_valid, synth_obj, sp_obj_reader, layers, REL)
        sp_last_acc = reader_accuracy(synth_valid, synth_last, sp_last_reader, layers, REL)
        sp_choice = {L: ("obj" if sp_obj_acc[L] >= sp_last_acc[L] else "last") for L in layers}
        selection_rows = [{"layer": L, "spatial_obj_acc": sp_obj_acc[L], "spatial_last_acc": sp_last_acc[L], "spatial_choice": sp_choice[L]} for L in layers]
        write_csv(out / "synthetic_spatial_carrier_selection.csv", selection_rows)
        print("[Synthetic spatial carrier choice]")
        for L in layers:
            print(f"  L{L:02d}: {sp_choice[L]:>4s} | obj={sp_obj_acc[L]:.3f} last={sp_last_acc[L]:.3f}")

        # 2) COCO: spatial reader + direct LM-head logit lens.
        per_rows: List[Dict[str, Any]] = []
        for m in tqdm(coco, desc="COCO spatial + decision trajectories"):
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
                q_obj, q_last, real_last = collect_states_from_image(
                    model=model, processor=processor, decoder_layers=decoder_layers, image=image,
                    prompt=prompt, subject=m["subject"], reference=m["reference"], layers=layers,
                    object_state=a.object_state, relation_token_map=relation_token_map,
                    gray_value=a.gray_value, device=device,
                )
                inv = {letter: rel for rel, letter in mp.items()}
                final_pred_letter = str(final_sc["prediction"])
                final_pred_relation = str(inv.get(final_pred_letter, "unknown"))
                gt = str(m["gt"])
                target_option = str(mp[target])
                opposite_option = str(mp[opposite])
                for L in layers:
                    sp_obj = reader_scores(q_obj[L], sp_obj_reader[L], REL)
                    sp_last = reader_scores(q_last[L], sp_last_reader[L], REL)
                    sp_carrier = sp_choice[L]
                    sp = sp_obj if sp_carrier == "obj" else sp_last
                    ll = logit_lens_option_logits(real_last[L], final_norm, lm_head, option_token_map, device)
                    sp_prob = softmax_dict(sp, REL, a.spatial_softmax_temp)
                    dec_prob = softmax_dict(ll, LETTERS, 1.0)
                    row: Dict[str, Any] = {
                        "sid": sid, "layer": L, "gt": gt,
                        "mapping": mapping_string(mp),
                        "target_relation": target, "opposite_relation": opposite,
                        "target_option": target_option, "opposite_option": opposite_option,
                        "final_prediction": final_pred_letter,
                        "final_prediction_relation": final_pred_relation,
                        "final_correct": int(final_pred_relation == gt),
                        "spatial_carrier": sp_carrier,
                        "spatial_margin": margin(sp, gt, REL),
                        "spatial_pair_margin": float(sp[target] - sp[opposite]),
                        "spatial_pair_preference": float(sp_prob[target] - sp_prob[opposite]),
                        "decision_pair_margin": float(ll[target_option] - ll[opposite_option]),
                        "decision_pair_preference": float(dec_prob[target_option] - dec_prob[opposite_option]),
                    }
                    for r in REL:
                        row[f"spatial_score_{r}"] = float(sp[r])
                    for z in LETTERS:
                        row[f"logitlens_{z}"] = float(ll[z])
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
        require_same = not bool(a.allow_different_option_pair)
        pairs = find_pairs(
            per_rows, target, opposite, a.mid_layers, a.predecision_layers, a.late_layers,
            a.min_mid_positive_frac, a.min_mid_mean_margin, a.max_mid_rmse,
            a.max_predecision_rmse, a.min_late_separation, a.require_late_sign_split,
            require_same, a.predecision_weight, a.late_separation_weight,
        )
        if not pairs:
            msg = (
                "No matched pair found. Try --allow-different-option-pair, lower --min-late-separation, "
                "or use --no-require-late-sign-split."
            )
            raise RuntimeError(msg)
        top = pairs[:max(1, int(a.topk))]
        write_csv(out / "top_pairs.csv", top)
        best = top[0]
        by = rows_by_sid(per_rows)
        plot_two_stage(out / "figure_view1_two_stage.png", best, by, layers, a.mid_layers, a.late_layers, target, opposite)
        plot_piecewise(out / "figure_view2_piecewise_preference.png", best, by, layers, a.transition_layer, target, opposite)

        best_rows: List[Dict[str, Any]] = []
        for role, sid in [("correct", int(best["sid_correct"])), ("wrong", int(best["sid_wrong"]))]:
            for L in layers:
                rr = dict(by[sid][L])
                rr["role"] = role
                best_rows.append(rr)
        write_csv(out / "best_pair_layer.csv", best_rows)

        meta = {
            "script_version": SCRIPT_VERSION,
            "model": a.model, "repo_id": spec.repo_id, "decoder_path": decoder_path,
            "norm_path": norm_path, "lm_head_path": head_path,
            "layers": layers, "target_relation": target, "opposite_relation": opposite,
            "mid_layers": list(map(int, a.mid_layers)),
            "predecision_layers": list(map(int, a.predecision_layers)),
            "late_layers": list(map(int, a.late_layers)),
            "transition_layer": int(a.transition_layer),
            "spatial_reader_source": "Synthetic-400 only; real-gray residual; per-layer global carrier choice",
            "decision_metric": "final-norm + LM-head logit lens on REAL prompt-last state",
            "view1": "separate spatial and decision observables",
            "view2": "stage-specific bounded GT-vs-opposite preference; metric changes at transition",
            "best_pair": best,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n[BEST PAIR]")
        for k, v in best.items():
            print(f"  {k}: {v}")
        print(f"[SAVED] {out / 'figure_view1_two_stage.png'}")
        print(f"[SAVED] {out / 'figure_view2_piecewise_preference.png'}")
        print(f"[SAVED] {out / 'top_pairs.csv'}")
        print(f"[SAVED] {out / 'best_pair_layer.csv'}")
        print(f"[SAVED] {out / 'per_sample_layer.csv'}")

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
