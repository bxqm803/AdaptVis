#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_synthetic3d_spatialmqa_recovery_v1.py

Run SpatialMQA test with the same recovery-style controller used in the
Controlled-B standalone script, but with:

- target dataset: SpatialMQA test jsonl
- source dataset: Blender synthetic_shapes_6dir_600_3d
- relation set: left / right / above / below / in front / behind
- models: qwen2-2b, qwen-3b, qwen-7b, llava-7b, llava-13b

This file reuses the standalone helper code from
`eval_synthetic3d_controlledB_spatial_control_genflip_trust_standalone_v1.py`
that should live in the same directory.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REL = ("left", "right", "above", "below", "in_front", "behind")
MODEL_ORDER = ("qwen2-2b", "qwen-3b", "qwen-7b", "llava-7b", "llava-13b")
DATASET_ORDER = ("spatialmqa_test",)
SCRIPT_VERSION = "synthetic3d-to-spatialmqa-standalone-v1"


# -----------------------------------------------------------------------------
# load base standalone module
# -----------------------------------------------------------------------------
BASE_FILE = Path(__file__).with_name("eval_synthetic3d_controlledB_spatial_control_genflip_trust_standalone_v1.py")
if not BASE_FILE.exists():
    raise FileNotFoundError(f"Missing base file: {BASE_FILE}")
_spec = importlib.util.spec_from_file_location("adaptvis_base_ctrlb", str(BASE_FILE))
base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(base)

base.REL = REL
base.MODEL_ORDER = MODEL_ORDER
base.DATASET_ORDER = DATASET_ORDER
base.SCRIPT_VERSION = SCRIPT_VERSION


# -----------------------------------------------------------------------------
# relation helpers
# -----------------------------------------------------------------------------
def norm_rel(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = re.sub(r"\s+", " ", str(x).strip().lower().replace("_", " ").replace("-", " "))
    table = {
        "left": "left", "left of": "left", "to the left of": "left",
        "right": "right", "right of": "right", "to the right of": "right",
        "above": "above", "on": "above", "on top of": "above", "over": "above", "on above": "above",
        "below": "below", "under": "below", "underneath": "below",
        "front": "in_front", "in front": "in_front", "in front of": "in_front", "ahead of": "in_front",
        "behind": "behind", "in back of": "behind", "back": "behind",
    }
    key = s.replace(" ", "_")
    return table.get(s, key if key in REL else None)


def parse_generation(text: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", str(text).lower().replace("_", " ").replace("-", " "))
    pats = [
        ("in_front", r"\bin front of\b"),
        ("in_front", r"\bin front\b"),
        ("behind", r"\bbehind\b"),
        ("above", r"\bon top of\b"),
        ("above", r"\babove\b"),
        ("above", r"\bon\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bund(?:er|erneath)?\b"),
        ("left", r"\bto the left of\b"),
        ("right", r"\bto the right of\b"),
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("in_front", r"\bfront\b"),
    ]
    hits = []
    for lab, pat in pats:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), lab))
    return min(hits, key=lambda z: z[0])[1] if hits else None


def surface_for(rel: str, dataset: str) -> str:
    del dataset
    mp = {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
        "in_front": "in front",
        "behind": "behind",
    }
    return mp[str(rel)]


base.norm_rel = norm_rel
base.parse_generation = parse_generation
base.surface_for = surface_for


# -----------------------------------------------------------------------------
# parsing SpatialMQA
# -----------------------------------------------------------------------------
QUESTION_PATTERNS = [
    re.compile(r"where is the (?P<subj>.+?) located relative to the (?P<ref>.+?)\?", re.I),
    re.compile(r"where is the (?P<subj>.+?) relative to the (?P<ref>.+?)\?", re.I),
    re.compile(r"from your perspective, where is the (?P<subj>.+?) located relative to the (?P<ref>.+?)\?", re.I),
    re.compile(r"from your perspective, where is the (?P<subj>.+?) relative to the (?P<ref>.+?)\?", re.I),
]


def _clean_obj(s: str) -> str:
    s = re.sub(r"[_-]+", " ", str(s)).strip()
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)
    return s.strip(" .,'\"")


def parse_spatialmqa_subject_reference(question: str) -> Optional[Tuple[str, str]]:
    q = re.sub(r"\s+", " ", str(question).strip())
    ql = q.lower()
    if "relative to you" in ql:
        return None
    for pat in QUESTION_PATTERNS:
        m = pat.search(q)
        if m:
            subj = _clean_obj(m.group("subj"))
            ref = _clean_obj(m.group("ref"))
            if subj and ref and subj.lower() != "you" and ref.lower() != "you":
                return subj, ref
    return None


def options_to_prompt_suffix(options: Sequence[str]) -> str:
    mapped = []
    for x in options:
        r = norm_rel(x)
        if r is not None:
            mapped.append(surface_for(r, "spatialmqa_test"))
    seen = []
    for x in mapped:
        if x not in seen:
            seen.append(x)
    if not seen:
        seen = [surface_for(r, "spatialmqa_test") for r in REL]
    return " Answer with " + ", ".join(seen[:-1]) + ", or " + seen[-1] + "." if len(seen) > 1 else f" Answer with {seen[0]}."


def load_spatialmqa(args: argparse.Namespace, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(args.spatialmqa_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)
    image_root = Path(args.spatialmqa_image_root)

    rows: List[Dict[str, Any]] = []
    skip_parse = 0
    skip_ref_you = 0
    skip_missing = 0
    skip_bad_rel = 0
    skip_bad_img = 0

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            question = str(x.get("question", "")).strip()
            parsed = parse_spatialmqa_subject_reference(question)
            if parsed is None:
                if "relative to you" in question.lower():
                    skip_ref_you += 1
                else:
                    skip_parse += 1
                continue
            subject, reference = parsed
            rel = norm_rel(x.get("answer"))
            if rel not in REL:
                skip_bad_rel += 1
                continue
            img_name = str(x.get("image", "")).strip()
            if not img_name:
                skip_missing += 1
                continue
            p = Path(img_name)
            candidates = [p, image_root / p.name, image_root / img_name]
            found = None
            for cand in candidates:
                if cand.exists():
                    found = cand
                    break
            if found is None:
                skip_bad_img += 1
                continue
            options = x.get("options") or []
            qtext = question.rstrip()
            if not qtext.endswith("?"):
                qtext = qtext + "?"
            qtext = qtext + options_to_prompt_suffix(options)
            rows.append({
                "sid": int(i),
                "dataset": "spatialmqa_test",
                "image_path": str(found),
                "subject": subject,
                "reference": reference,
                "relation": rel,
                "question_text": qtext,
                "raw_question": question,
                "raw_answer": x.get("answer"),
                "raw_options": options,
                "image_name": img_name,
            })
    rows.sort(key=lambda r: int(r["sid"]))
    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]

    counts = {r: sum(row["relation"] == r for row in rows) for r in REL}
    print(
        f"[SpatialMQA test] n={len(rows)} counts={counts} "
        f"skip_parse={skip_parse} skip_relative_to_you={skip_ref_you} "
        f"skip_bad_rel={skip_bad_rel} skip_missing={skip_missing} skip_bad_img={skip_bad_img}"
    )
    if not rows:
        raise RuntimeError("No usable SpatialMQA rows after filtering")
    return rows


# -----------------------------------------------------------------------------
# source synthetic-3d, all 6 directions
# -----------------------------------------------------------------------------
def load_synthetic_3d(args: argparse.Namespace, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    root = Path(args.synthetic_3d_dir)
    labels_path = Path(args.synthetic_3d_labels) if args.synthetic_3d_labels else root / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows: List[Dict[str, Any]] = []
    raw_counts: Dict[str, int] = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            raw_rel = str(x.get("relation", "")).strip().lower()
            raw_counts[raw_rel] = raw_counts.get(raw_rel, 0) + 1
            rel = norm_rel(raw_rel)
            if rel not in REL:
                continue
            subject = _clean_obj(str(x.get("subject", "")))
            reference = _clean_obj(str(x.get("reference", "")))
            rel_img = str(x.get("image", "")).strip()
            if not subject or not reference or not rel_img:
                continue
            image_path = Path(rel_img)
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            qtext = (
                f"Where is the {subject} in relation to the {reference}? "
                "Answer with left, right, above, below, in front, or behind."
            )
            rows.append({
                "sid": int(i),
                "dataset": "synthetic3d_6dir_source",
                "image_path": str(image_path),
                "subject": subject,
                "reference": reference,
                "relation": rel,
                "question_text": qtext,
                "source_relation_raw": raw_rel,
            })
    rows.sort(key=lambda r: int(r["sid"]))
    if limit is not None and int(limit) > 0 and int(limit) < len(rows):
        n = int(limit)
        per = max(1, n // len(REL))
        chosen: List[Dict[str, Any]] = []
        leftovers: List[Dict[str, Any]] = []
        for rel in REL:
            rr = [r for r in rows if r["relation"] == rel]
            chosen.extend(rr[:per])
            leftovers.extend(rr[per:])
        if len(chosen) < n:
            chosen.extend(leftovers[: n - len(chosen)])
        rows = sorted(chosen[:n], key=lambda r: int(r["sid"]))
    counts = {r: sum(x["relation"] == r for x in rows) for r in REL}
    print(f"[Synthetic-3D 6dir source] n={len(rows)} counts={counts} raw_counts={raw_counts}")
    if not rows or any(counts[r] == 0 for r in REL):
        raise RuntimeError(f"Synthetic-3D source invalid counts={counts}")
    return rows


# -----------------------------------------------------------------------------
# generic 3-axis geometry
# -----------------------------------------------------------------------------
def fit_geometry(X: np.ndarray, y: np.ndarray, layers: Sequence[int]) -> Tuple[dict, pd.DataFrame]:
    geom: Dict[int, dict] = {}
    rows: List[dict] = []
    if X.shape[1] != len(layers):
        raise RuntimeError(f"Geometry X shape {X.shape} incompatible with layers={layers}")
    for i, L in enumerate(layers):
        Xf = X[:, i].astype(np.float64)
        center = Xf.mean(0)
        means: Dict[str, np.ndarray] = {}
        for r in REL:
            m = y == r
            if not np.any(m):
                raise RuntimeError(f"Synthetic-3D geometry missing class={r}")
            means[r] = Xf[m].mean(0)
        class_dirs = {r: base.unit(means[r] - center) for r in REL}
        dH = base.unit(class_dirs["right"] - class_dirs["left"])
        dV = base.unit(class_dirs["above"] - class_dirs["below"])
        dD = base.unit(class_dirs["in_front"] - class_dirs["behind"])

        qH = dH.copy()
        v2 = dV - float(np.dot(dV, qH)) * qH
        qV = base.unit(v2)
        if float(np.dot(qV, dV)) < 0:
            qV = -qV
        v3 = dD - float(np.dot(dD, qH)) * qH - float(np.dot(dD, qV)) * qV
        qD = base.unit(v3)
        if float(np.dot(qD, dD)) < 0:
            qD = -qD

        Q = np.stack([qH, qV, qD], axis=1)
        B = np.stack([dH, dV, dD], axis=1)
        geom[int(L)] = {
            "control_basis": Q.astype(np.float32),
            "B": B.astype(np.float32),
            "center": center.astype(np.float32),
            "class_means": {r: means[r].astype(np.float32) for r in REL},
        }
        rows.append({
            "layer": int(L),
            "fit_N": int(len(Xf)),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "axis_H_dot_axis_D": float(np.dot(dH, dD)),
            "axis_V_dot_axis_D": float(np.dot(dV, dD)),
            "orthonormal_check_max_abs": float(np.max(np.abs(Q.T @ Q - np.eye(3)))),
        })
    return geom, pd.DataFrame(rows)


def control_dim(geom: Mapping[int, Any]) -> int:
    first = next(iter(geom.values()))
    return int(np.asarray(first["control_basis"]).shape[1])


# -----------------------------------------------------------------------------
# 6-way head routing
# -----------------------------------------------------------------------------
def build_standalone_routing(
    *, model_alias: str, model: Any, processor: Any, decoder_layers: Sequence[Any],
    n_heads: int, head_dim: int, source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]], root_out: Path, args: argparse.Namespace,
) -> Tuple[dict, Dict[int, dict]]:
    reader_dir = root_out / "reader" / model_alias / "spatialmqa_test"
    reader_dir.mkdir(parents=True, exist_ok=True)
    source_cache = reader_dir / "synthetic3d_6dir_head_vectors.npz"
    target_cache = reader_dir / "spatialmqa_test_head_vectors.npz"

    src = base.extract_head_vectors(
        records=source_rows, model=model, processor=processor,
        decoder_layers=decoder_layers, n_heads=n_heads, head_dim=head_dim,
        model_alias=model_alias, dataset_name="synthetic3d_6dir",
        cache_path=source_cache, args=args,
    )
    tgt = base.extract_head_vectors(
        records=target_rows, model=model, processor=processor,
        decoder_layers=decoder_layers, n_heads=n_heads, head_dim=head_dim,
        model_alias=model_alias, dataset_name="spatialmqa_test",
        cache_path=target_cache, args=args,
    )

    Xs = np.asarray(src["vectors"])
    ys = np.asarray(src["relation"], dtype=object)
    Xt = np.asarray(tgt["vectors"])
    yt = np.asarray(tgt["relation"], dtype=object)
    center, dirs = base.fit_source_head_codebooks(Xs, ys)
    np.savez_compressed(
        reader_dir / "synthetic3d_frozen_head_codebooks.npz",
        center=center.astype(np.float32),
        directions=dirs.astype(np.float32),
        relations=np.asarray(REL, dtype=object),
        vector_definition=np.asarray(
            "pre-W_O per-head [(subject-reference)_real-(subject-reference)_gray]",
            dtype=object,
        ),
    )

    ranking: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    all_idx = np.arange(len(yt), dtype=np.int64)
    for l in tqdm(range(Xt.shape[1]), desc=f"{model_alias}:rank synthetic3d heads", leave=False):
        for h in range(Xt.shape[2]):
            pred_t, margin_t = base.predict_one_head(Xt, center, dirs, l, h)
            pred_s, _ = base.predict_one_head(Xs, center, dirs, l, h)
            predictions[(l, h)] = (pred_t, margin_t)
            row = {
                "layer": int(l),
                "head": int(h),
                "head_name": f"L{l}H{h:02d}",
                "syn_self_acc": base.safe_acc(pred_s, ys),
                "selection_acc": base.safe_acc(pred_t[all_idx], yt[all_idx]),
                "test_acc": base.safe_acc(pred_t[all_idx], yt[all_idx]),
                "all_target_acc": base.safe_acc(pred_t, yt),
                "mean_margin": float(np.mean(margin_t)),
            }
            for rel in REL:
                m = yt == rel
                row[f"{rel}_acc"] = base.safe_acc(pred_t[m], yt[m]) if np.any(m) else float("nan")
            ranking.append(row)
    ranking.sort(key=lambda r: (-float(r["selection_acc"]), -float(r["syn_self_acc"]), -float(r["mean_margin"]), int(r["layer"]), int(r["head"])))
    for rank, r in enumerate(ranking, 1):
        r["rank"] = int(rank)
    pd.DataFrame(ranking).to_csv(reader_dir / "head_ranking.csv", index=False)

    best = ranking[0]
    key = (int(best["layer"]), int(best["head"]))
    best_pred, best_margin = predictions[key]
    target_sid = np.asarray(tgt["sid"], dtype=np.int64)
    sample_rows: List[Dict[str, Any]] = []
    routing: Dict[int, dict] = {}
    for i in range(len(yt)):
        row = {
            "row_index": int(i),
            "sid": int(target_sid[i]),
            "split": "eval",
            "gt": str(yt[i]),
            "best_head": str(best["head_name"]),
            "head_pred": str(best_pred[i]),
            "head_correct": int(best_pred[i] == yt[i]),
            "head_margin": float(best_margin[i]),
        }
        sample_rows.append(row)
        routing[int(target_sid[i])] = row
    pd.DataFrame(sample_rows).to_csv(reader_dir / "samples_best_head.csv", index=False)

    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model_alias,
        "dataset": "spatialmqa_test",
        "control": "gray",
        "source": "synthetic_shapes_6dir_600_3d",
        "direction_fit_uses_target_gt": False,
        "head_selection_uses_target_gt": True,
        "head_selection_frac": 1.0,
        "target_n": int(len(yt)),
        "best_head": str(best["head_name"]),
        "best_layer": int(best["layer"]),
        "best_head_index": int(best["head"]),
        "best_syn_self_acc": float(best["syn_self_acc"]),
        "best_selection_acc": float(best["selection_acc"]),
        "best_test_acc": float(best["test_acc"]),
        "best_all_target_acc": float(best["all_target_acc"]),
        "top_heads": ranking[: int(args.top_k_heads)],
    }
    base.write_json(reader_dir / "summary.json", summary)
    print("\n" + "=" * 138)
    print("SYNTHETIC-3D FROZEN HEAD READER -> SPATIALMQA")
    print("=" * 138)
    print(f"best={summary['best_head']} | syn={summary['best_syn_self_acc']:.4f} | target={summary['best_all_target_acc']:.4f} | N={summary['target_n']}")
    cols = ["rank", "head_name", "syn_self_acc", "all_target_acc"] + [f"{r}_acc" for r in REL] + ["mean_margin"]
    print(pd.DataFrame(ranking[: min(int(args.top_k_heads), 10)])[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary, routing


# -----------------------------------------------------------------------------
# generic-dim score / optimize helpers
# -----------------------------------------------------------------------------
def fast_relation_scores_and_grads(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Any,
    relation_token_ids: Mapping[str, Sequence[int]], layers: Sequence[int], geom: Mapping[int, Any],
    base_coords: np.ndarray, target: str, need_grads: bool = True,
) -> Tuple[Dict[str, float], Optional[np.ndarray], List[str]]:
    dev = prepared.batch["input_ids"].device
    d = control_dim(geom)
    delta = torch.zeros((len(layers), d), device=dev, dtype=torch.float32, requires_grad=need_grads)
    base_coords_t = torch.as_tensor(base_coords, device=dev, dtype=torch.float32)
    coords = base_coords_t + delta
    with base.SpatialPatch(
        decoder_layers=decoder_layers, layers=layers, geom=geom,
        sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=coords,
    ):
        if need_grads:
            out = model(**prepared.batch, use_cache=False, return_dict=True)
        else:
            with torch.inference_mode():
                out = model(**prepared.batch, use_cache=False, return_dict=True)
        last = out.logits[0, -1].float()
        score_tensors = {r: torch.stack([last[int(tok)] for tok in relation_token_ids[r]]).max() for r in REL}
        scores = {r: float(score_tensors[r].detach().item()) for r in REL}
        competitors = [r for r in REL if r != target]
        if not need_grads:
            return scores, None, competitors
        grads: List[np.ndarray] = []
        for j, r in enumerate(competitors):
            margin = score_tensors[target] - score_tensors[r]
            g = torch.autograd.grad(margin, delta, retain_graph=(j + 1 < len(competitors)), create_graph=False)[0]
            grads.append(g.detach().float().cpu().numpy().astype(np.float64).reshape(-1))
        A = np.stack(grads, axis=0)
    return scores, A, competitors


def sequence_relation_scores_and_grads(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Any,
    candidate_ids: Mapping[str, Sequence[int]], layers: Sequence[int], geom: Mapping[int, Any],
    base_coords: np.ndarray, target: str, reduction: str,
) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    sg: Dict[str, Tuple[float, np.ndarray]] = {}
    for r in REL:
        sg[r] = score_and_grad(
            model=model, decoder_layers=decoder_layers, prepared=prepared,
            answer_ids=candidate_ids[r], layers=layers, geom=geom,
            base_coords=base_coords, reduction=reduction,
        )
    scores = {r: float(sg[r][0]) for r in REL}
    competitors = [r for r in REL if r != target]
    A = np.stack([(sg[target][1] - sg[r][1]).reshape(-1) for r in competitors], axis=0)
    return scores, A, competitors


def score_and_grad(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Any,
    answer_ids: Sequence[int], layers: Sequence[int], geom: Mapping[int, Any],
    base_coords: np.ndarray, reduction: str,
) -> Tuple[float, np.ndarray]:
    ext = base.extend_batch(prepared.batch, answer_ids)
    dev = prepared.batch["input_ids"].device
    d = control_dim(geom)
    delta = torch.zeros((len(layers), d), device=dev, dtype=torch.float32, requires_grad=True)
    base_coords_t = torch.as_tensor(base_coords, device=dev, dtype=torch.float32)
    coords = base_coords_t + delta
    with base.SpatialPatch(
        decoder_layers=decoder_layers, layers=layers, geom=geom,
        sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=coords,
    ):
        out = model(**ext, use_cache=False, return_dict=True)
        score = base.sequence_score_from_logits(out.logits, answer_ids, reduction)
        grad = torch.autograd.grad(score, delta, retain_graph=False, create_graph=False)[0]
    return float(score.detach().item()), grad.detach().float().cpu().numpy().astype(np.float64)


def generate_with_coords(*, model_alias: str, model: Any, decoder_layers: Sequence[Any], prepared: Any, layers: Sequence[int], geom: Mapping[int, Any], coords: np.ndarray, args: argparse.Namespace):
    return base.generate_with_coords(model_alias=model_alias, model=model, decoder_layers=decoder_layers, prepared=prepared, layers=layers, geom=geom, coords=coords, args=args)


def refine_first_generation_flip(*, model_alias: str, model: Any, decoder_layers: Sequence[Any], prepared: Any, layers: Sequence[int], geom: Mapping[int, Any], target: str, low_coords: np.ndarray, high_coords: np.ndarray, high_text: str, args: argparse.Namespace):
    return base.refine_first_generation_flip(model_alias=model_alias, model=model, decoder_layers=decoder_layers, prepared=prepared, layers=layers, geom=geom, target=target, low_coords=low_coords, high_coords=high_coords, high_text=high_text, args=args)


def optimize_to_target(
    *, model_alias: str, model: Any, decoder_layers: Sequence[Any], prepared: Any,
    candidate_ids: Mapping[str, Sequence[int]], fast_token_ids: Optional[Mapping[str, Sequence[int]]],
    layers: Sequence[int], geom: Mapping[int, Any], target: str,
    base_gen_pred: Optional[str], base_gen_text: str, args: argparse.Namespace,
) -> Dict[str, Any]:
    d = control_dim(geom)
    coords = np.zeros((len(layers), d), dtype=np.float64)
    gen_pred = base_gen_pred
    gen_text = base_gen_text
    steps = 0
    qp_failed = False
    capped_steps = 0
    stop_reason = "max_steps_without_generation_target"
    last_active_names: List[str] = []
    last_step_norm = 0.0
    last_requested_step_norm = 0.0
    max_requested_step_norm = 0.0
    last_step_cap_ratio = 1.0
    last_pre_step_margin = float("nan")
    last_predicted_post_step_margin = float("nan")
    last_scores: Dict[str, float] = {}
    score_mode = "single_token_shared_forward" if fast_token_ids is not None else "sequence_fallback"
    n_gradient_evals = 0
    n_generation_checks = 0
    n_margin_deepenings = 0
    n_binary_checks = 0
    binary_alpha = float("nan")
    first_flip_total_residual_norm = float("nan")
    required_margin = float(args.decision_margin_eps)
    max_required_margin = required_margin

    if gen_pred == target:
        stop_reason = "already_target"
    else:
        for step_idx in range(1, int(args.max_steps) + 1):
            if fast_token_ids is not None:
                dev = prepared.batch["input_ids"].device
                delta = torch.zeros((len(layers), d), device=dev, dtype=torch.float32, requires_grad=True)
                base_coords_t = torch.as_tensor(coords, device=dev, dtype=torch.float32)
                current = base_coords_t + delta
                with base.SpatialPatch(decoder_layers=decoder_layers, layers=layers, geom=geom, sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=current):
                    out = model(**prepared.batch, use_cache=False, return_dict=True)
                    last = out.logits[0, -1].float()
                    score_tensors = {r: torch.stack([last[int(tok)] for tok in fast_token_ids[r]]).max() for r in REL}
                    scores = {r: float(score_tensors[r].detach().item()) for r in REL}
                    competitors = [r for r in REL if r != target]
                    margins = np.asarray([scores[target] - scores[r] for r in competitors], dtype=np.float64)
                    last_scores = dict(scores)
                    last_pre_step_margin = float(np.min(margins))
                    if last_pre_step_margin >= required_margin - float(args.qp_feas_tol):
                        inc = max(float(args.generation_margin_increment), 1e-8)
                        required_margin = max(required_margin + inc, last_pre_step_margin + inc)
                        max_required_margin = max(max_required_margin, required_margin)
                        n_margin_deepenings += 1
                    b = required_margin - margins
                    grad_rows: List[np.ndarray] = []
                    for j, r in enumerate(competitors):
                        margin_t = score_tensors[target] - score_tensors[r]
                        g = torch.autograd.grad(margin_t, delta, retain_graph=(j + 1 < len(competitors)), create_graph=False)[0]
                        grad_rows.append(g.detach().float().cpu().numpy().astype(np.float64).reshape(-1))
                    A = np.stack(grad_rows, axis=0)
                    n_gradient_evals += 1
                del out, last, score_tensors, delta, current, base_coords_t
            else:
                scores, A, competitors = sequence_relation_scores_and_grads(
                    model=model, decoder_layers=decoder_layers, prepared=prepared,
                    candidate_ids=candidate_ids, layers=layers, geom=geom,
                    base_coords=coords, target=target, reduction=args.sequence_score_reduction,
                )
                margins = np.asarray([scores[target] - scores[r] for r in competitors], dtype=np.float64)
                last_scores = dict(scores)
                last_pre_step_margin = float(np.min(margins))
                if last_pre_step_margin >= required_margin - float(args.qp_feas_tol):
                    inc = max(float(args.generation_margin_increment), 1e-8)
                    required_margin = max(required_margin + inc, last_pre_step_margin + inc)
                    max_required_margin = max(max_required_margin, required_margin)
                    n_margin_deepenings += 1
                b = required_margin - margins
                n_gradient_evals += 1

            flat_step, qp_info = base.solve_min_norm_halfspaces(A, b, feas_tol=float(args.qp_feas_tol), lambda_tol=float(args.qp_lambda_tol))
            if flat_step is None:
                qp_failed = True
                stop_reason = str(qp_info.get("solver_status", "local_boundary_qp_infeasible"))
                break
            raw_step_norm = float(np.linalg.norm(flat_step))
            if not math.isfinite(raw_step_norm):
                qp_failed = True
                stop_reason = "nonfinite_boundary_step"
                break
            if raw_step_norm < float(args.min_residual_step):
                stop_reason = "minimum_residual_step_too_small_before_generation_flip"
                break

            requested_step_norm = raw_step_norm
            last_requested_step_norm = requested_step_norm
            max_requested_step_norm = max(max_requested_step_norm, requested_step_norm)
            max_step = float(args.max_residual_step)
            last_step_cap_ratio = 1.0
            if max_step > 0.0 and requested_step_norm > max_step:
                scale = max_step / requested_step_norm
                flat_step = flat_step * scale
                raw_step_norm = max_step
                last_step_cap_ratio = float(scale)
                capped_steps += 1

            prev_coords = coords.copy()
            step = flat_step.reshape(len(layers), d)
            proposed_coords = coords + step
            steps = step_idx
            last_step_norm = raw_step_norm
            active_idx = qp_info.get("active_constraints") or []
            last_active_names = [competitors[int(i)] for i in active_idx]
            linearized_post = margins + A @ flat_step
            last_predicted_post_step_margin = float(np.min(linearized_post))

            proposed_pred, proposed_text = generate_with_coords(
                model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                prepared=prepared, layers=layers, geom=geom, coords=proposed_coords, args=args,
            )
            n_generation_checks += 1
            if proposed_pred == target:
                first_flip_total_residual_norm = float(np.linalg.norm(proposed_coords.reshape(-1)))
                coords, gen_pred, gen_text, extra_checks, binary_alpha = refine_first_generation_flip(
                    model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                    prepared=prepared, layers=layers, geom=geom, target=target,
                    low_coords=prev_coords, high_coords=proposed_coords,
                    high_text=proposed_text, args=args,
                )
                n_generation_checks += extra_checks
                n_binary_checks += extra_checks
                stop_reason = "generation_target_refined" if extra_checks > 0 else "generation_target"
                break
            coords = proposed_coords
            gen_pred = proposed_pred
            gen_text = proposed_text

    layer_norms = np.linalg.norm(coords, axis=1)
    return {
        "patched_prediction": gen_pred,
        "patched_generation_text": gen_text,
        "steps_taken": int(steps),
        "final_total_residual_norm": float(np.linalg.norm(coords.reshape(-1))),
        "final_layer_residual_norms": [float(x) for x in layer_norms.tolist()],
        "final_control_coords": coords.tolist(),
        "last_boundary_step_norm": float(last_step_norm),
        "last_requested_boundary_step_norm": float(last_requested_step_norm),
        "max_requested_boundary_step_norm": float(max_requested_step_norm),
        "last_step_cap_ratio": float(last_step_cap_ratio),
        "last_pre_step_target_margin": float(last_pre_step_margin),
        "last_predicted_post_step_margin": float(last_predicted_post_step_margin),
        "last_relation_scores": {k: float(v) for k, v in last_scores.items()},
        "last_active_competitors": list(last_active_names),
        "final_required_relation_margin": float(required_margin),
        "max_required_relation_margin": float(max_required_margin),
        "margin_deepenings": int(n_margin_deepenings),
        "binary_search_checks": int(n_binary_checks),
        "binary_search_alpha_last_step": float(binary_alpha),
        "first_flip_total_residual_norm": float(first_flip_total_residual_norm),
        "target_reached": bool(gen_pred == target),
        "qp_failed": bool(qp_failed),
        "capped_steps": int(capped_steps),
        "stop_reason": str(stop_reason),
        "score_mode": score_mode,
        "gradient_evaluations": int(n_gradient_evals),
        "generation_checks": int(n_generation_checks),
    }


def summarize_pair(df: pd.DataFrame) -> pd.DataFrame:
    return base.summarize_pair(df)


def run_pair(
    *, model_alias: str, dataset: str, model: Any, processor_or_backend: Any,
    decoder_layers: Sequence[Any], layers: Sequence[int], geom: Mapping[int, Any],
    hs_summary: Mapping[str, Any], routing: Mapping[int, Mapping[str, Any]],
    target_rows_all: Sequence[Mapping[str, Any]], args: argparse.Namespace, root_out: Path,
) -> dict:
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    target_rows = list(target_rows_all)
    if "head" in modes:
        target_rows = [r for r in target_rows if int(r["sid"]) in routing]
    else:
        hs_summary = {"best_head": None, "best_all_target_acc": float("nan")}
        routing = {int(r["sid"]): {"sid": int(r["sid"]), "gt": r["relation"], "head_pred": r["relation"]} for r in target_rows}
    target_rows = base.stratified_cap(target_rows, int(args.eval_max_samples), int(args.eval_seed))
    if not target_rows:
        raise RuntimeError(f"No target rows after routing alignment for {model_alias}/{dataset}")

    outdir = root_out / model_alias / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "per_sample.jsonl"
    error_path = outdir / "errors.jsonl"
    if args.overwrite:
        for p in (result_path, error_path):
            if p.exists():
                p.unlink()

    existing = base.read_jsonl(result_path)
    done = {(int(r["sid"]), str(r["mode"])) for r in existing}
    tokenizer = processor_or_backend.tokenizer
    candidate_ids = base.encode_candidates(tokenizer, dataset)
    fast_token_ids, fast_token_details = base.encode_fast_relation_tokens(tokenizer, dataset)

    print("\n" + "=" * 180)
    print(f"CONTROL {model_alias} / {dataset} | selected_head={hs_summary.get('best_head')} | head_acc(all)={float(hs_summary.get('best_all_target_acc', float('nan'))):.4f}")
    print(f"Synthetic-3D 6dir geometry layers={list(layers)} | eval N={len(target_rows)}")
    if fast_token_ids is not None:
        print(f"FAST SCORE PATH: one prompt forward for six relation scores | token_variants={fast_token_ids}")
    else:
        print("FALLBACK SCORE PATH: at least one relation is multi-token; using exact teacher-forced sequence scores")
        print(json.dumps(fast_token_details, ensure_ascii=False))
    print("Synthetic-3D 6dir geometry -> relation target -> minimum residual 3-axis step -> deepen score margin until actual generation flips -> trust-cap each local step -> binary-refine final step")
    print("=" * 180, flush=True)

    pbar = tqdm(target_rows, desc=f"CONTROL {model_alias}:{dataset}")
    processed_since_cleanup = 0
    d = control_dim(geom)
    for row in pbar:
        sid = int(row["sid"])
        rr = routing[sid]
        gt = norm_rel(row["relation"])
        head_pred = norm_rel(rr["head_pred"])
        if gt not in REL or head_pred not in REL:
            continue
        pending_modes = [mode for mode in modes if (sid, mode) not in done]
        if not pending_modes:
            continue

        sample_t0 = time.perf_counter()
        image = prepared = None
        try:
            image = Image.open(str(row["image_path"])).convert("RGB")
            prepared = base.prepare_standard(model=model, processor=processor_or_backend, decoder_layers=decoder_layers, row=row, image=image, args=args)
            zero = np.zeros((len(layers), d), dtype=np.float64)
            base_pred, base_text = generate_with_coords(model_alias=model_alias, model=model, decoder_layers=decoder_layers, prepared=prepared, layers=layers, geom=geom, coords=zero, args=args)
            result_by_target: Dict[str, Dict[str, Any]] = {}
            for mode in pending_modes:
                control_target = head_pred if mode == "head" else gt
                reused = control_target in result_by_target
                if not reused:
                    target_t0 = time.perf_counter()
                    res = optimize_to_target(
                        model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                        prepared=prepared, candidate_ids=candidate_ids,
                        fast_token_ids=fast_token_ids, layers=layers,
                        geom=geom, target=control_target,
                        base_gen_pred=base_pred, base_gen_text=base_text, args=args,
                    )
                    res = dict(res)
                    res["runtime_seconds"] = float(time.perf_counter() - target_t0)
                    result_by_target[control_target] = res
                else:
                    res = dict(result_by_target[control_target])
                    res["reused_source_runtime_seconds"] = float(res.get("runtime_seconds", 0.0))
                    res["runtime_seconds"] = 0.0
                    res["reused_same_target_result"] = True
                patched = norm_rel(res["patched_prediction"])
                outrow = {
                    "model": model_alias,
                    "dataset": dataset,
                    "sid": sid,
                    "mode": mode,
                    "selected_head": str(hs_summary.get("best_head")),
                    "selected_head_target_acc": float(hs_summary.get("best_all_target_acc", float("nan"))),
                    "gt": gt,
                    "head_prediction": head_pred,
                    "head_correct": bool(head_pred == gt),
                    "control_target": control_target,
                    "baseline_prediction": base_pred,
                    "baseline_correct": bool(base_pred == gt),
                    "baseline_generation_text": base_text,
                    "patched_prediction": patched,
                    "patched_correct": bool(patched == gt),
                    "patched_matches_control_target": bool(patched == control_target),
                    "reused_same_target_result": bool(reused),
                    **res,
                }
                base.append_jsonl(result_path, outrow)
                done.add((sid, mode))
            sample_sec = time.perf_counter() - sample_t0
            latest = next(reversed(result_by_target.values())) if result_by_target else {}
            pbar.set_postfix(sec=f"{sample_sec:.1f}", steps=int(latest.get("steps_taken", 0)), stop=str(latest.get("stop_reason", "-"))[:18], refresh=True)
        except Exception as exc:
            base.append_jsonl(error_path, {
                "sid": sid, "model": model_alias, "dataset": dataset,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-60:],
            })
            tqdm.write(f"[ERROR] {model_alias}/{dataset} sid={sid}: {type(exc).__name__}: {exc}")
            base.cleanup_cuda()
            processed_since_cleanup = 0
            if args.fail_fast:
                raise
        finally:
            if prepared is not None:
                prepared.close()
            if image is not None:
                image.close()

        processed_since_cleanup += 1
        every = int(args.cuda_cleanup_every)
        if every > 0 and processed_since_cleanup >= every:
            base.cleanup_cuda()
            processed_since_cleanup = 0

    base.cleanup_cuda()
    all_rows = base.read_jsonl(result_path)
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError(f"No successful control rows for {model_alias}/{dataset}")
    df.to_csv(outdir / "per_sample.csv", index=False)
    summary_df = summarize_pair(df)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    meta = {
        "script_version": SCRIPT_VERSION,
        "model": model_alias,
        "dataset": dataset,
        "reader_dir": str(root_out / "reader" / model_alias / "spatialmqa_test"),
        "selected_head": hs_summary.get("best_head"),
        "selected_head_target_acc": hs_summary.get("best_all_target_acc"),
        "direction_source": "Blender Synthetic-3D six directions",
        "head_id_selection": "full target GT",
        "per_sample_head_routing_uses_target_gt": False,
        "residual_geometry_source": "Blender Synthetic-3D six directions",
        "geometry_control": "gray",
        "controller_layers": list(map(int, layers)),
        "controller_dim": int(d * len(layers)),
        "single_layer_control_dim": int(d),
        "spatial_basis": "orthonormal basis spanning horizontal / vertical / depth directions",
        "single_token_shared_forward": bool(fast_token_ids is not None),
        "relation_token_ids": fast_token_ids,
        "relation_token_details": fast_token_details,
    }
    base.write_json(outdir / "metadata.json", meta)
    print("\nPAIR SUMMARY")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    result = {
        "model": model_alias,
        "dataset": dataset,
        "N": int(df["sid"].nunique()),
        "head": str(hs_summary.get("best_head")),
        "head_readout_accuracy": float(df[df["mode"] == modes[0]]["head_correct"].mean()) if modes else float("nan"),
    }
    for mode in ("head", "oracle"):
        q = summary_df[summary_df["mode"] == mode]
        if len(q):
            r = q.iloc[0]
            result[f"{mode}_generation_accuracy"] = float(r["patched_generation_accuracy"])
            result[f"{mode}_gain"] = float(r["gain_vs_baseline"])
            result[f"{mode}_compliance"] = float(r["target_compliance"])
            result[f"{mode}_W2C"] = int(r["wrong_to_correct"])
            result[f"{mode}_C2W"] = int(r["correct_to_wrong"])
            result["baseline_accuracy"] = float(r["baseline_accuracy"])
    return result


# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", default="all")
    p.add_argument("--datasets", default="spatialmqa_test")
    p.add_argument("--pairs", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--synthetic-3d-dir", default="synthetic_shapes_6dir_600_3d")
    p.add_argument("--synthetic-3d-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)

    p.add_argument("--spatialmqa-jsonl", default="data/SpatialMQA/test.jsonl")
    p.add_argument("--spatialmqa-image-root", default="data/SpatialMQA/COCO2017")
    p.add_argument("--target-max-samples", type=int, default=None)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--revision", default="main")
    p.add_argument("--control", default="gray", choices=["gray"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--head-selection-frac", type=float, default=1.0)
    p.add_argument("--top-k-heads", type=int, default=30)
    p.add_argument("--cache-vectors", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--controller-layers", default="")
    p.add_argument("--geometry-cache-dir", default=None)
    p.add_argument("--modes", default="head,oracle")
    p.add_argument("--eval-max-samples", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=17)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--decision-margin-eps", type=float, default=1e-4)
    p.add_argument("--generation-margin-increment", type=float, default=4.0)
    p.add_argument("--binary-search-steps", type=int, default=4)
    p.add_argument("--qp-feas-tol", type=float, default=1e-7)
    p.add_argument("--qp-lambda-tol", type=float, default=1e-9)
    p.add_argument("--min-residual-step", type=float, default=1e-8)
    p.add_argument("--max-residual-step", type=float, default=5000.0)
    p.add_argument("--generation-check-every", type=int, default=1)
    p.add_argument("--cuda-cleanup-every", type=int, default=25)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--sequence-score-reduction", default="mean", choices=["mean", "sum"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if str(args.control) != "gray":
        raise ValueError("This protocol requires Real-Gray source residuals")

    base.seed_all(int(args.seed))
    models = base.parse_name_list(args.models, MODEL_ORDER)
    datasets = base.parse_name_list(args.datasets, DATASET_ORDER)
    pairs = base.parse_pairs(args.pairs, models, datasets)
    overrides = base.parse_layer_overrides(args.controller_layers)
    root_out = Path(args.output_dir)
    if args.overwrite and root_out.exists():
        shutil.rmtree(root_out)
    root_out.mkdir(parents=True, exist_ok=True)
    geom_root = Path(args.geometry_cache_dir) if args.geometry_cache_dir else root_out / "geometry_cache"

    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    bad = [x for x in modes if x not in {"head", "oracle"}]
    if bad:
        raise ValueError(f"Unknown modes {bad}")

    source_rows = load_synthetic_3d(args, limit=args.source_max_samples)
    target_rows_all = load_spatialmqa(args, limit=args.target_max_samples)
    print(f"[PROTOCOL] source=Blender Synthetic-3D 6dir | source N={len(source_rows)} | SpatialMQA N={len(target_rows_all)} | modes={modes}")

    summaries: List[dict] = []
    failures: List[dict] = []
    for model_alias in MODEL_ORDER:
        model_pairs = [(m, d) for m, d in pairs if m == model_alias]
        if not model_pairs:
            continue
        model = processor = None
        try:
            model, processor, decoder_layers, n_heads, head_dim, spec = base.load_model_bundle(model_alias, args)
            for p in model.parameters():
                p.requires_grad_(False)
            n_layers = len(decoder_layers)
            layers = overrides.get(model_alias, base.relative7_layers(n_layers))
            if len(layers) != 7:
                raise RuntimeError(f"{model_alias}: controller must use exactly 7 layers; got {layers}")
            print(f"\n[MODEL CONTROLLER] {model_alias}: n_layers={n_layers}, layers={layers}")

            if "head" in modes:
                hs_summary, routing = build_standalone_routing(
                    model_alias=model_alias, model=model, processor=processor,
                    decoder_layers=decoder_layers, n_heads=n_heads, head_dim=head_dim,
                    source_rows=source_rows, target_rows=target_rows_all,
                    root_out=root_out, args=args,
                )
            else:
                hs_summary = {"best_head": None, "best_all_target_acc": float("nan")}
                routing = {int(r["sid"]): {"sid": int(r["sid"]), "gt": str(r["relation"]), "head_pred": str(r["relation"])} for r in target_rows_all}

            geom_cache = geom_root / model_alias / "synthetic3d_6dir_residual_real_minus_gray.npz"
            Xg, yg = base.build_synthetic_geometry_cache(
                model_alias=model_alias, model=model, processor_or_backend=processor,
                decoder_layers=decoder_layers, layers=layers, source_rows=source_rows,
                cache_path=geom_cache, args=args,
            )
            geom, axis_df = fit_geometry(Xg, yg, layers)
            model_geom_dir = geom_root / model_alias
            model_geom_dir.mkdir(parents=True, exist_ok=True)
            axis_df.to_csv(model_geom_dir / "synthetic3d_3axis_geometry.csv", index=False)

            for _m, dataset in model_pairs:
                try:
                    result = run_pair(
                        model_alias=model_alias, dataset=dataset, model=model,
                        processor_or_backend=processor, decoder_layers=decoder_layers,
                        layers=layers, geom=geom, hs_summary=hs_summary, routing=routing,
                        target_rows_all=target_rows_all, args=args, root_out=root_out,
                    )
                    summaries.append(result)
                except Exception as exc:
                    failures.append({"model": model_alias, "dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[PAIR FAILED] {model_alias}/{dataset}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise
        finally:
            with contextlib.suppress(Exception):
                del model
            with contextlib.suppress(Exception):
                del processor
            base.cleanup_cuda()

    if summaries:
        df = pd.DataFrame(summaries)
        df = df[[c for c in [
            "model", "dataset", "N", "head", "head_readout_accuracy",
            "head_generation_accuracy", "head_gain", "head_compliance", "head_W2C", "head_C2W",
            "baseline_accuracy",
            "oracle_generation_accuracy", "oracle_gain", "oracle_compliance", "oracle_W2C", "oracle_C2W",
        ] if c in df.columns]]
        df.to_csv(root_out / "all_results.csv", index=False)
        print("\n" + "=" * 180)
        print("FINAL SYNTHETIC-3D -> SPATIALMQA CONTROL SUMMARY")
        print("=" * 180)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved to: {root_out}")
    if failures:
        fdf = pd.DataFrame(failures)
        fdf.to_csv(root_out / "failures.csv", index=False)
        print("\nFAILURES")
        print(fdf.to_string(index=False))


if __name__ == "__main__":
    main()
