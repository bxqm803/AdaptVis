#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direction-head logit-gradient decomposition for Qwen-3B / COCO_two.

Purpose
-------
Previous experiments found that for high-ACC Direction heads, the signed
spatial contribution can become small or negative on samples where the head
itself decodes the relation correctly but final generation is wrong.

This script decomposes that negative spatial-margin effect into the FOUR final
relation logits.  For each selected Direction head h and test sample i, it asks:

    If the subject/reference activations of this head are moved a tiny amount
    along the GT-oriented spatial axis, what is the first-order change in
    each final relation logit: left/right/above/below?

For a GT=left sample, for example:

    effect_left   = d z_left   / d epsilon
    effect_right  = d z_right  / d epsilon
    effect_above  = d z_above  / d epsilon
    effect_below  = d z_below  / d epsilon

where epsilon moves the two object-token head activations oppositely along the
head's train-derived left-vs-right axis:

    z_sub <- z_sub + epsilon/2 * d_left
    z_ref <- z_ref - epsilon/2 * d_left

Therefore:

    margin effect (GT vs competitor)
      = effect_GT - effect_competitor.

If this is negative, the script lets us distinguish:
  * GT suppression:       effect_GT < 0
  * competitor boosting:  effect_competitor > 0
  * both
  * relative competition (e.g. both increase, but competitor increases more)

Two intervention directions are reported:
  1) prototype-axis (recommended): a clean GT-oriented LR/UD axis learned only
     from the TRAIN split of existing Image-NoImage residual Direction vectors.
  2) sample-residual: the current sample's normalized Image-NoImage residual
     vector, for continuity with the previous g.r analysis.

The script also reconstructs each head's TEST prediction using TRAIN-only
prototypes, then splits results into:
    G+ H+ : generation correct, head correct
    G+ H- : generation correct, head wrong
    G- H+ : generation wrong,   head correct   <-- most diagnostic
    G- H- : generation wrong,   head wrong

Inputs expected from previous experiments
-----------------------------------------
  --direction-results:
      output/qwen3b_coco_head_direction_residual/head_results.csv
  --direction-vectors-npz:
      output/qwen3b_coco_head_direction_residual/relation_vectors.npz
  --feasibility-dir:
      output/qwen3b_coco_grounded_consensus_v1/
        split.csv
        test_samples.csv

This script RE-RUNS model forward/backward because the previous scalar-margin
backward cannot be decomposed into four individual relation-logit gradients.
It does NOT train or fine-tune the model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_coco_head_object_residual_direction_probe_v1 as direction_base

RELATIONS = ("left", "right", "above", "below")
REL_TO_IDX = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
        help="Keep identical to the Direction-vector / feasibility experiment.",
    )
    p.add_argument(
        "--direction-results",
        default="output/qwen3b_coco_head_direction_residual/head_results.csv",
    )
    p.add_argument(
        "--direction-vectors-npz",
        default="output/qwen3b_coco_head_direction_residual/relation_vectors.npz",
    )
    p.add_argument(
        "--feasibility-dir",
        default="output/qwen3b_coco_grounded_consensus_v1",
    )
    p.add_argument("--direction-top-k", type=int, default=10)
    p.add_argument(
        "--heads",
        default="auto",
        help="Comma-separated LxHy list, or auto = top-K residual Direction heads.",
    )
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--print-every", type=int, default=20)
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
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm_relation(x: Any) -> str:
    return direction_base.norm_relation(x)


def parse_head_name(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*[Ll](\d+)[Hh](\d+)\s*", str(text))
    if not m:
        raise ValueError(f"Bad head name {text!r}")
    return int(m.group(1)), int(m.group(2))


def head_name(l: int, h: int) -> str:
    return f"L{int(l)}H{int(h):02d}"


def load_top_heads(path: Path, k: int) -> Tuple[List[str], Dict[str, float]]:
    rows = read_csv(path)
    scored: List[Tuple[float, str]] = []
    accs: Dict[str, float] = {}
    for r in rows:
        try:
            acc = float(r.get("residual_accuracy_mean", "nan"))
            name = r.get("head_name") or head_name(int(r["layer"]), int(r["head"]))
            l, h = parse_head_name(name)
            name = head_name(l, h)
            if math.isfinite(acc):
                scored.append((acc, name))
                accs[name] = acc
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:k]], accs


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    try:
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    except Exception:
        obj = tokenizer(text, add_special_tokens=False)
        ids = obj["input_ids"] if isinstance(obj, dict) else getattr(obj, "input_ids", [])
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in ids]


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    unk = getattr(tokenizer, "unk_token_id", None)
    for rel in RELATIONS:
        ids: List[int] = []
        for s in [rel, " " + rel, "\n" + rel, rel.capitalize(), " " + rel.capitalize()]:
            xx = tokenizer_ids(tokenizer, s)
            if len(xx) != 1:
                continue
            tid = int(xx[0])
            if unk is not None and tid == int(unk):
                continue
            ids.append(tid)
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise RuntimeError(f"No one-token variants for {rel}")
        out[rel] = ids
    return out


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ]
    for x in candidates:
        if torch.is_tensor(x):
            return x
    if isinstance(outputs, (tuple, list)):
        for x in outputs:
            if torch.is_tensor(x) and x.ndim == 3:
                return x
    raise RuntimeError("Could not locate logits")


def relation_scores(score_vector: torch.Tensor, token_map: Mapping[str, Sequence[int]]) -> torch.Tensor:
    vals = []
    for rel in RELATIONS:
        ids = torch.as_tensor(token_map[rel], device=score_vector.device, dtype=torch.long)
        vals.append(score_vector.index_select(0, ids).max())
    return torch.stack(vals)


def build_prompt_and_batch(processor: Any, rec: Any, question: str, image: Image.Image, device: torch.device):
    rendered = direction_base.build_chat_prompt(processor, question, True)
    batch = direction_base.process_inputs(processor, rendered, image, device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    apos = direction_base.locate_phrase_positions(processor.tokenizer, ids, str(rec.subject))
    bpos = direction_base.locate_phrase_positions(processor.tokenizer, ids, str(rec.reference))
    return batch, apos, bpos


def valid_positions(pos: Sequence[int], seq_len: int) -> List[int]:
    return [int(x) for x in pos if 0 <= int(x) < seq_len]


def pool_tensor(x: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    pos = valid_positions(positions, int(x.shape[0]))
    if not pos:
        raise RuntimeError("No valid object token positions")
    if mode == "last":
        return x[pos[-1]]
    idx = torch.as_tensor(pos, device=x.device, dtype=torch.long)
    return x.index_select(0, idx).mean(dim=0)


class PreWOCapture:
    def __init__(self, layers: Sequence[Any], layer_ids: Sequence[int]):
        self.tensors: Dict[int, torch.Tensor] = {}
        self.handles = []
        self.layer_ids = sorted(set(map(int, layer_ids)))
        for li in self.layer_ids:
            op = direction_base.resolve_o_proj(direction_base.resolve_self_attention(layers[li]))

            def make_hook(layer_id: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{layer_id} o_proj input unavailable")
                    self.tensors[layer_id] = inputs[0]
                return hook

            self.handles.append(op.register_forward_pre_hook(make_hook(li)))

    def validate(self):
        miss = [l for l in self.layer_ids if l not in self.tensors]
        if miss:
            raise RuntimeError(f"Capture missing layers {miss}")

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def head_slice(x: torch.Tensor, h: int, head_dim: int) -> torch.Tensor:
    a = int(h) * int(head_dim)
    b = a + int(head_dim)
    return x[0, :, a:b]


def normalize_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / max(n, EPS)).astype(np.float32)


def fit_head_train_geometry(
    residual: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    l: int,
    h: int,
) -> Dict[str, Any]:
    X = residual[train_rows, l, h, :].astype(np.float32)
    y = labels[train_rows].astype(str)
    center = X.mean(axis=0).astype(np.float32)
    means: Dict[str, np.ndarray] = {}
    proto: Dict[str, np.ndarray] = {}
    for rel in RELATIONS:
        m = y == rel
        if not np.any(m):
            raise RuntimeError(f"No train samples for {rel}")
        mu = X[m].mean(axis=0).astype(np.float32)
        means[rel] = mu
        proto[rel] = normalize_np(mu - center)

    lr = normalize_np(means["left"] - means["right"])
    ud = normalize_np(means["above"] - means["below"])
    gt_axis = {
        "left": lr,
        "right": -lr,
        "above": ud,
        "below": -ud,
    }
    return {"center": center, "proto": proto, "gt_axis": gt_axis}


def predict_from_geometry(x: np.ndarray, geom: Mapping[str, Any]) -> Tuple[str, float]:
    v = x.astype(np.float32) - np.asarray(geom["center"], dtype=np.float32)
    v = normalize_np(v)
    scores = np.asarray([float(v @ np.asarray(geom["proto"][r])) for r in RELATIONS])
    order = np.sort(scores)
    return RELATIONS[int(np.argmax(scores))], float(order[-1] - order[-2])


def mean(xs: Iterable[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    return float(vals.mean()) if vals.size else float("nan")


def median(xs: Iterable[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    return float(np.median(vals)) if vals.size else float("nan")


def fraction(rows: Sequence[Mapping[str, Any]], key: str, pred) -> float:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return float(np.mean([pred(v) for v in vals])) if vals else float("nan")


def cause_label(gt_eff: float, comp_eff: float) -> str:
    margin = gt_eff - comp_eff
    if margin >= 0:
        return "constructive_or_neutral"
    gt_down = gt_eff < 0
    comp_up = comp_eff > 0
    if gt_down and comp_up:
        return "both_gt_down_and_wrong_up"
    if gt_down and not comp_up:
        return "gt_suppression"
    if (not gt_down) and comp_up:
        return "wrong_boost"
    return "relative_competitor_dominance"


def summarize_group(rows: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    keys = [
        f"{prefix}_gt_effect",
        f"{prefix}_bestwrong_effect",
        f"{prefix}_opposite_effect",
        f"{prefix}_margin_bestwrong",
        f"{prefix}_margin_opposite",
    ]
    out: Dict[str, Any] = {"n": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[f"mean_{k}"] = mean(vals)
        out[f"median_{k}"] = median(vals)
        out[f"mean_abs_{k}"] = mean(abs(v) for v in vals)
    out[f"frac_{prefix}_gt_suppressed"] = fraction(rows, f"{prefix}_gt_effect", lambda v: v < 0)
    out[f"frac_{prefix}_bestwrong_boosted"] = fraction(rows, f"{prefix}_bestwrong_effect", lambda v: v > 0)
    out[f"frac_{prefix}_margin_negative"] = fraction(rows, f"{prefix}_margin_bestwrong", lambda v: v < 0)
    out[f"frac_{prefix}_both"] = float(np.mean([
        (float(r[f"{prefix}_gt_effect"]) < 0 and float(r[f"{prefix}_bestwrong_effect"]) > 0)
        for r in rows
    ])) if rows else float("nan")
    return out


GROUPS = [
    (True, True, "G+ H+"),
    (True, False, "G+ H-"),
    (False, True, "G- H+"),
    (False, False, "G- H-"),
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.direction_results)
    vec_path = Path(args.direction_vectors_npz)
    feas_dir = Path(args.feasibility_dir)
    split_path = feas_dir / "split.csv"
    test_path = feas_dir / "test_samples.csv"
    for p in [results_path, vec_path, split_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    top_heads, probe_accs = load_top_heads(results_path, args.direction_top_k)
    if args.heads.strip().lower() == "auto":
        heads = top_heads
    else:
        heads = [head_name(*parse_head_name(x.strip())) for x in args.heads.split(",") if x.strip()]
    print("Direction heads:", ", ".join(heads))

    split_rows = read_csv(split_path)
    sid_to_split = {int(r["sid"]): r["split"] for r in split_rows}
    train_sids = {sid for sid, sp in sid_to_split.items() if sp == "train"}
    test_sids = {sid for sid, sp in sid_to_split.items() if sp == "test"}

    test_eval_rows = read_csv(test_path)
    test_eval = {int(r["sid"]): r for r in test_eval_rows if int(r["sid"]) in test_sids}
    test_sid_order = [int(r["sid"]) for r in test_eval_rows if int(r["sid"]) in test_sids]
    if args.max_test_samples is not None:
        test_sid_order = test_sid_order[: int(args.max_test_samples)]

    z = np.load(vec_path, allow_pickle=False)
    residual = np.asarray(z["residual"], dtype=np.float32)
    sids = np.asarray(z["sample_index"], dtype=np.int64)
    labels = np.asarray(z["relation"]).astype(str)
    sid_to_vec = {int(s): i for i, s in enumerate(sids.tolist())}
    train_rows = np.asarray([sid_to_vec[s] for s in sorted(train_sids) if s in sid_to_vec], dtype=np.int64)

    geometry: Dict[str, Dict[str, Any]] = {}
    for hn in heads:
        l, h = parse_head_name(hn)
        geometry[hn] = fit_head_train_geometry(residual, labels, train_rows, l, h)

    # Dataset records, restricted to untouched TEST SIDs.
    records, _audit = base.load_records(args.dataset, Path(args.data_root), None)
    by_sid = {int(r.sid): r for r in records}
    missing = [s for s in test_sid_order if s not in by_sid]
    if missing:
        raise RuntimeError(f"Missing dataset records for test sids: {missing[:10]}")

    # Model.
    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    kw: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    # Some transformers versions use torch_dtype rather than dtype.
    try:
        model = cls.from_pretrained(spec.repo_id, **({**kw, **({"attn_implementation": args.attn_impl} if args.attn_impl != "none" else {})}))
    except TypeError:
        kw.pop("dtype", None)
        kw["torch_dtype"] = base.resolve_dtype(spec.dtype_name)
        if args.attn_impl != "none":
            kw["attn_implementation"] = args.attn_impl
        model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layer_path = direction_base.resolve_decoder_layers(model)
    n_heads, head_dim = direction_base.scan_shape(model, layers)
    selected_layers = sorted(set(parse_head_name(h)[0] for h in heads))
    print(f"decoder={layer_path} layers={len(layers)} heads={n_heads} head_dim={head_dim}")
    print("captured layers:", selected_layers)
    for hn in heads:
        l, h = parse_head_name(hn)
        if not (0 <= l < len(layers) and 0 <= h < n_heads):
            raise ValueError(f"{hn} outside model shape")

    token_map = relation_token_variants(processor.tokenizer)

    per_rows: List[Dict[str, Any]] = []

    for si, sid in enumerate(tqdm(test_sid_order, desc="four-logit-gradient"), 1):
        rec = by_sid[sid]
        image = None
        try:
            gt = norm_relation(rec.relation)
            if gt not in REL_TO_IDX:
                continue
            q = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
            image = Image.open(rec.image_path).convert("RGB")
            batch, apos, bpos = build_prompt_and_batch(processor, rec, q, image, device)

            with PreWOCapture(layers, selected_layers) as cap:
                outputs = model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                cap.validate()
                logits = extract_logits(outputs)
                rel_scores = relation_scores(logits[0, -1], token_map)
                score_np = rel_scores.detach().float().cpu().numpy()
                wrongs = [r for r in RELATIONS if r != gt]
                best_wrong = max(wrongs, key=lambda r: float(score_np[REL_TO_IDX[r]]))
                opposite = OPPOSITE[gt]

                grad_inputs = [cap.tensors[l] for l in selected_layers]

                # Pre-create output rows for each head and head correctness.
                current: Dict[str, Dict[str, Any]] = {}
                vec_row = sid_to_vec.get(sid)
                if vec_row is None:
                    raise RuntimeError(f"sid={sid} absent from relation_vectors.npz")
                for hn in heads:
                    l, h = parse_head_name(hn)
                    r_res = np.asarray(residual[vec_row, l, h], dtype=np.float32)
                    pred, pred_margin = predict_from_geometry(r_res, geometry[hn])
                    head_correct = pred == gt
                    gen_correct = as_bool(test_eval[sid].get("generation_correct"))
                    base_row: Dict[str, Any] = {
                        "sid": sid,
                        "gt": gt,
                        "generation_correct": int(gen_correct),
                        "generation_prediction": test_eval[sid].get("generation_prediction", ""),
                        "head_name": hn,
                        "layer": l,
                        "head": h,
                        "probe_accuracy_original": probe_accs.get(hn, float("nan")),
                        "head_prediction": pred,
                        "head_correct": int(head_correct),
                        "head_probe_margin": pred_margin,
                        "group": (
                            "G+ H+" if gen_correct and head_correct else
                            "G+ H-" if gen_correct and not head_correct else
                            "G- H+" if (not gen_correct) and head_correct else
                            "G- H-"
                        ),
                        "best_wrong": best_wrong,
                        "opposite": opposite,
                        "logit_left": float(score_np[0]),
                        "logit_right": float(score_np[1]),
                        "logit_above": float(score_np[2]),
                        "logit_below": float(score_np[3]),
                    }
                    current[hn] = base_row

                # Four backward traversals: one per spatial logit.
                for ri, rel in enumerate(RELATIONS):
                    grads = torch.autograd.grad(
                        rel_scores[ri],
                        grad_inputs,
                        retain_graph=(ri < len(RELATIONS) - 1),
                        create_graph=False,
                        allow_unused=True,
                    )
                    grad_by_layer = {l: g for l, g in zip(selected_layers, grads)}
                    for hn in heads:
                        l, h = parse_head_name(hn)
                        g_full = grad_by_layer.get(l)
                        if g_full is None:
                            current[hn][f"proto_effect_{rel}"] = float("nan")
                            current[hn][f"resunit_effect_{rel}"] = float("nan")
                            current[hn][f"resraw_effect_{rel}"] = float("nan")
                            continue
                        g = head_slice(g_full, h, head_dim).float()
                        g_sub = pool_tensor(g, apos, args.pool)
                        g_ref = pool_tensor(g, bpos, args.pool)
                        # 1/2 because epsilon perturbs sub by +eps/2*d and ref by -eps/2*d.
                        g_rel = 0.5 * (g_sub - g_ref)

                        d_proto = torch.from_numpy(
                            np.asarray(geometry[hn]["gt_axis"][gt], dtype=np.float32)
                        ).to(device=g_rel.device)
                        r_res_np = np.asarray(residual[sid_to_vec[sid], l, h], dtype=np.float32)
                        r_res_raw = torch.from_numpy(r_res_np).to(device=g_rel.device)
                        r_res_unit_np = normalize_np(r_res_np)
                        r_res_unit = torch.from_numpy(r_res_unit_np).to(device=g_rel.device)

                        current[hn][f"proto_effect_{rel}"] = float(torch.dot(g_rel, d_proto).item())
                        current[hn][f"resunit_effect_{rel}"] = float(torch.dot(g_rel, r_res_unit).item())
                        current[hn][f"resraw_effect_{rel}"] = float(torch.dot(g_rel, r_res_raw).item())

                    del grads, grad_by_layer

                # Derived GT-vs-wrong effects and cause labels.
                for hn, row in current.items():
                    for prefix in ["proto", "resunit", "resraw"]:
                        gt_e = float(row[f"{prefix}_effect_{gt}"])
                        bw_e = float(row[f"{prefix}_effect_{best_wrong}"])
                        op_e = float(row[f"{prefix}_effect_{opposite}"])
                        row[f"{prefix}_gt_effect"] = gt_e
                        row[f"{prefix}_bestwrong_effect"] = bw_e
                        row[f"{prefix}_opposite_effect"] = op_e
                        row[f"{prefix}_margin_bestwrong"] = gt_e - bw_e
                        row[f"{prefix}_margin_opposite"] = gt_e - op_e
                        row[f"{prefix}_cause_bestwrong"] = cause_label(gt_e, bw_e)
                    per_rows.append(row)

            if args.print_every > 0 and si % int(args.print_every) == 0:
                tqdm.write(f"processed {si}/{len(test_sid_order)}")

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(out_dir / "per_sample_logit_gradient_effects.csv", per_rows)

    # Head x correctness-group summaries.
    head_summary: List[Dict[str, Any]] = []
    by_head: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_rows:
        by_head[str(r["head_name"])].append(r)

    for hn in heads:
        rs = by_head[hn]
        test_acc = mean(float(r["head_correct"]) for r in rs)
        for gen_c, head_c, gname in GROUPS:
            grp = [
                r for r in rs
                if bool(int(r["generation_correct"])) == gen_c
                and bool(int(r["head_correct"])) == head_c
            ]
            item: Dict[str, Any] = {
                "head_name": hn,
                "probe_accuracy_original": probe_accs.get(hn, float("nan")),
                "head_test_accuracy": test_acc,
                "group": gname,
                "n": len(grp),
            }
            for prefix in ["proto", "resunit", "resraw"]:
                item.update(summarize_group(grp, prefix))
                causes = [str(r[f"{prefix}_cause_bestwrong"]) for r in grp]
                for cause in [
                    "constructive_or_neutral",
                    "gt_suppression",
                    "wrong_boost",
                    "both_gt_down_and_wrong_up",
                    "relative_competitor_dominance",
                ]:
                    item[f"frac_{prefix}_cause_{cause}"] = (
                        float(np.mean([c == cause for c in causes])) if causes else float("nan")
                    )
            head_summary.append(item)

    write_csv(out_dir / "head_correctness_group_summary.csv", head_summary)

    # Descriptive family aggregate (head-sample rows are NOT independent samples).
    family_summary: List[Dict[str, Any]] = []
    for gen_c, head_c, gname in GROUPS:
        grp = [
            r for r in per_rows
            if bool(int(r["generation_correct"])) == gen_c
            and bool(int(r["head_correct"])) == head_c
        ]
        item: Dict[str, Any] = {"group": gname, "nrows": len(grp)}
        for prefix in ["proto", "resunit", "resraw"]:
            item.update(summarize_group(grp, prefix))
        family_summary.append(item)
    write_csv(out_dir / "family_correctness_group_summary.csv", family_summary)

    # Print compact diagnostic table, prototype axis first because signs are comparable across heads.
    print("\n" + "=" * 150)
    print("DIRECTION FOUR-LOGIT GRADIENT DECOMPOSITION")
    print("=" * 150)
    print(f"model / dataset : {args.model} / {args.dataset}")
    print(f"TEST N          : {len(test_sid_order)}")
    print(f"heads           : {', '.join(heads)}")
    print("Effects below use the GT-oriented TRAIN prototype axis. Positive GT effect raises the correct relation logit; positive wrong effect raises the competitor logit.")
    print("-")
    print(
        f"{'head':9s} {'testACC':>8s} {'group':>7s} {'n':>5s} "
        f"{'GT eff':>11s} {'bestWrong eff':>14s} {'margin eff':>12s} "
        f"{'GT<0':>8s} {'wrong>0':>9s} {'margin<0':>10s} {'both':>8s}"
    )
    for hn in heads:
        rows_h = [r for r in head_summary if r["head_name"] == hn]
        first = True
        for r in rows_h:
            print(
                f"{hn if first else '':9s} "
                f"{float(r['head_test_accuracy']):8.4f} "
                f"{str(r['group']):>7s} {int(r['n']):5d} "
                f"{float(r['mean_proto_gt_effect']):11.6f} "
                f"{float(r['mean_proto_bestwrong_effect']):14.6f} "
                f"{float(r['mean_proto_margin_bestwrong']):12.6f} "
                f"{float(r['frac_proto_gt_suppressed']):8.3f} "
                f"{float(r['frac_proto_bestwrong_boosted']):9.3f} "
                f"{float(r['frac_proto_margin_negative']):10.3f} "
                f"{float(r['frac_proto_both']):8.3f}"
            )
            first = False
        print("-")

    print("\nFamily-level descriptive aggregate (head-sample rows; not independent):")
    print(
        f"{'group':8s} {'nrows':>7s} {'GT eff':>11s} {'bestWrong':>11s} {'margin':>11s} "
        f"{'GT<0':>8s} {'wrong>0':>9s} {'margin<0':>10s} {'both':>8s}"
    )
    for r in family_summary:
        print(
            f"{str(r['group']):8s} {int(r['nrows']):7d} "
            f"{float(r['mean_proto_gt_effect']):11.6f} "
            f"{float(r['mean_proto_bestwrong_effect']):11.6f} "
            f"{float(r['mean_proto_margin_bestwrong']):11.6f} "
            f"{float(r['frac_proto_gt_suppressed']):8.3f} "
            f"{float(r['frac_proto_bestwrong_boosted']):9.3f} "
            f"{float(r['frac_proto_margin_negative']):10.3f} "
            f"{float(r['frac_proto_both']):8.3f}"
        )

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "test_n": len(test_sid_order),
        "heads": heads,
        "note": (
            "prototype-axis effects are unit-direction directional derivatives; "
            "family aggregates treat head-sample rows descriptively, not as independent observations"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved:")
    for fn in [
        "per_sample_logit_gradient_effects.csv",
        "head_correctness_group_summary.csv",
        "family_correctness_group_summary.csv",
        "summary.json",
    ]:
        print(" ", out_dir / fn)


if __name__ == "__main__":
    main()
