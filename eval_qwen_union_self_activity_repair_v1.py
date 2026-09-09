#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation-free self-activity repair from a union of oracle-discovered K24
canonical middle-token candidates.

Offline discovery only
----------------------
Use selected_tokens.csv from the existing oracle writer-guided K24 run.
For each relation separately, canonicalize the repeatedly selected positions,
take the top-N candidates for that relation, then form one UNION pool:

    U = U_left union U_right union U_on union U_under

At evaluation time there is NO relation selector and NO GT-dependent routing.

Evaluation-time signal
----------------------
For every candidate c=(layer, canonical slot) in U, resolve its position in the
current prompt and compute only its own image-conditioned displacement:

    delta_c = h_real[c] - h_gray[c]

Two relation-free methods are tested:

(2) SELF-WEIGHTED UNION
    All candidates participate, but their amplification is proportional to
    their endogenous current-sample activity:

        e_c = ||delta_c||
        w_c = clip(e_c / mean(e), 0, max_weight)
        h'_c = h_c + alpha * w_c * delta_c

(3) SELF TOP-K
    Rank ALL candidates in the unified pool directly by e_c, without first
    choosing left/right/on/under.  Amplify the top-K:

        C* = TopK_c e_c
        h'_c = h_c + alpha * delta_c, c in C*

Optional score modes:
    delta_norm:
        ||h_real - h_gray||

    relative_delta:
        ||h_real - h_gray|| / (||h_real|| + eps)

The intervention NEVER reads GT relation, late writer, gradient, answer logits,
or a learned relation classifier.  GT is used only after generation to score
accuracy.

Important experimental hygiene
------------------------------
Default --eval-scope unseen excludes every sample ID used in selected_tokens.csv
from evaluation.  Thus the oracle writer is used only to discover a reusable
candidate pool offline; it is absent on unseen evaluation samples.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_union_self_activity_repair_v1.py \
  --model qwen-3b \
  --selected-csv output/qwen3b_coco_minimal_token_search/selected_tokens.csv \
  --discovery-bundle "L22+L24+L26[global_unique]" \
  --discovery-k 24 \
  --per-relation-candidates 12 \
  --score-modes delta_norm \
  --methods weighted,topk \
  --topks 4,8,12,16,24 \
  --alphas 0.5,0.75,1.0 \
  --eval-scope unseen \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_coco_union_self_activity_unseen80 \
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL_INTERNAL = ("left", "right", "above", "below")
REL_DISPLAY = ("left", "right", "on", "under")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
UNDISPLAY = {"left": "left", "right": "right", "on": "above", "under": "below"}


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])

    p.add_argument("--selected-csv", required=True)
    p.add_argument(
        "--discovery-bundle",
        default="L22+L24+L26[global_unique]",
        help="Bundle in selected_tokens.csv used only for offline candidate discovery.",
    )
    p.add_argument("--discovery-k", type=int, default=24)
    p.add_argument("--discovery-condition", default="positive")
    p.add_argument(
        "--per-relation-candidates",
        type=int,
        default=12,
        help=(
            "Take this many highest-frequency canonical candidates from EACH "
            "relation, then union them. 0 means keep every discovered candidate "
            "that passes --candidate-min-rate."
        ),
    )
    p.add_argument(
        "--candidate-min-rate",
        type=float,
        default=0.0,
        help="Minimum within-relation discovery selection frequency.",
    )
    p.add_argument("--visual-bins", type=int, default=16)

    p.add_argument(
        "--score-modes",
        default="delta_norm",
        help="Comma-separated subset of delta_norm,relative_delta.",
    )
    p.add_argument(
        "--methods",
        default="weighted,topk",
        help="Comma-separated subset of weighted,topk.",
    )
    p.add_argument("--topks", default="4,8,12,16,24")
    p.add_argument("--alphas", default="0.5,0.75,1.0")
    p.add_argument(
        "--max-weight",
        type=float,
        default=2.0,
        help="Clip upper bound for self-weighted union weights.",
    )
    p.add_argument(
        "--weight-power",
        type=float,
        default=1.0,
        help="Apply score**power before mean normalization in weighted method.",
    )

    p.add_argument(
        "--eval-scope",
        default="unseen",
        choices=["unseen", "selected", "all_test"],
        help=(
            "unseen excludes all discovery sample IDs; selected evaluates only "
            "discovery IDs; all_test uses the full held-out test split."
        ),
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def parse_csv_strings(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def canon_rel(x):
    x = str(x).strip().lower()
    if x == "above":
        return "on"
    if x == "below":
        return "under"
    return x


def safe_mean(xs):
    vals = np.asarray([float(x) for x in xs], np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def stratified_take(rows, n, seed):
    if not n or n <= 0 or n >= len(rows):
        return list(rows)
    return traj.stratified_cap(rows, n, seed)


# =============================================================================
# Canonical prompt mapping
# =============================================================================

def visual_positions_from_tokens(tokens):
    return {
        i
        for i, tok in enumerate(tokens)
        if "image_pad" in str(tok) or "video_pad" in str(tok)
    }


def get_object_spans(tokenizer, ids, subject, reference):
    sspan, rspan = base.locate_object_spans(
        tokenizer, ids, subject, reference
    )
    spos = set(range(int(sspan[0]), int(sspan[1]) + 1))
    rpos = set(range(int(rspan[0]), int(rspan[1]) + 1))
    return spos, rpos


def build_canonical_mapping(
    tokens,
    subject_pos,
    reference_pos,
    visual_pos,
    visual_bins,
):
    """
    Text:
        exact canonical slots after collapsing <SUBJ>/<REF>/<VISUAL_BLOCK>.
    Visual:
        normalized VISBIN_xx, because absolute image token positions are not
        semantically aligned across images.
    """
    n = len(tokens)
    mapping = {}
    canonical_index = 0

    visual_sorted = sorted(visual_pos)
    visual_ord = {p: j for j, p in enumerate(visual_sorted)}
    nvis = len(visual_sorted)

    i = 0
    while i < n:
        if i in visual_pos:
            block = []
            while i < n and i in visual_pos:
                block.append(i)
                i += 1
            canonical_index += 1

            for p in block:
                j = visual_ord[p]
                if nvis <= 1:
                    b = 0
                else:
                    frac = j / (nvis - 1)
                    b = min(visual_bins - 1, int(frac * visual_bins))
                mapping[p] = {
                    "kind": "visual",
                    "canonical_slot": None,
                    "visual_bin": f"VISBIN_{b:02d}",
                }
            continue

        if i in subject_pos:
            block = []
            while i < n and i in subject_pos:
                block.append(i)
                i += 1
            label = f"C{canonical_index:03d}:<SUBJ>"
            canonical_index += 1
            for p in block:
                mapping[p] = {
                    "kind": "text",
                    "canonical_slot": label,
                    "visual_bin": None,
                }
            continue

        if i in reference_pos:
            block = []
            while i < n and i in reference_pos:
                block.append(i)
                i += 1
            label = f"C{canonical_index:03d}:<REF>"
            canonical_index += 1
            for p in block:
                mapping[p] = {
                    "kind": "text",
                    "canonical_slot": label,
                    "visual_bin": None,
                }
            continue

        tok = str(tokens[i]).replace("\n", "\\n")
        label = f"C{canonical_index:03d}:{tok}"
        mapping[i] = {
            "kind": "text",
            "canonical_slot": label,
            "visual_bin": None,
        }
        canonical_index += 1
        i += 1

    return mapping


def candidate_id(layer, kind, key):
    if kind == "visual":
        return f"L{int(layer)}::<VISUAL>::{key}"
    return f"L{int(layer)}::{key}"


# =============================================================================
# Offline candidate-pool discovery from selected_tokens.csv
# =============================================================================

def canonicalize_discovery_rows(
    sel,
    prompts,
    rec_by_sid,
    processor,
    visual_bins,
):
    tokenizer = processor.tokenizer
    wanted_sids = sorted(set(sel["sid"].astype(int)))
    maps = {}

    for sid in tqdm(wanted_sids, desc="Canonicalize discovery K24"):
        if sid not in prompts or sid not in rec_by_sid:
            raise RuntimeError(f"Discovery sid={sid} missing from prompts/records")

        p = prompts[sid]
        real = batch = None
        try:
            real = base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")

            batch = base.make_question_batch(
                processor=processor,
                image=real,
                question_text=str(p["question_text"]),
                device=torch.device("cpu"),
            )
            ids = batch["input_ids"][0].detach().cpu().tolist()
            toks = [
                str(x)
                for x in processor.tokenizer.convert_ids_to_tokens(ids)
            ]
            spos, rpos = get_object_spans(
                tokenizer,
                ids,
                str(p["subject"]),
                str(p["reference"]),
            )
            vpos = visual_positions_from_tokens(toks)
            maps[sid] = build_canonical_mapping(
                toks, spos, rpos, vpos, visual_bins
            )
        finally:
            if real is not None:
                with contextlib.suppress(Exception):
                    real.close()
            del batch

    rows = []
    for r in sel.to_dict("records"):
        sid = int(r["sid"])
        pos = int(r["position"])
        info = maps.get(sid, {}).get(pos)
        if info is None:
            continue

        kind = info["kind"]
        key = (
            info["visual_bin"]
            if kind == "visual"
            else info["canonical_slot"]
        )
        rows.append({
            **r,
            "candidate_kind": kind,
            "candidate_key": key,
            "candidate": candidate_id(
                int(r["source_layer"]), kind, key
            ),
        })

    return pd.DataFrame(rows)


def build_union_pool(
    ann,
    per_relation_candidates,
    min_rate,
):
    """
    Rank candidate units within each relation by:
      1) sample-level selection rate
      2) mean positive mediation when selected

    Then union the top candidates from all four relations.
    """
    relation_tables = []
    picked_membership = defaultdict(list)

    for rel in REL_DISPLAY:
        rr = ann[ann["relation"] == rel].copy()
        denom = int(rr["sid"].nunique())
        if denom == 0:
            continue

        rr = rr.sort_values("mediation", ascending=False).drop_duplicates(
            ["sid", "candidate"], keep="first"
        )

        agg = (
            rr.groupby(
                [
                    "candidate",
                    "source_layer",
                    "candidate_kind",
                    "candidate_key",
                ],
                as_index=False,
            )
            .agg(
                n_samples_selected=("sid", "nunique"),
                mean_mediation=("mediation", "mean"),
                median_mediation=("mediation", "median"),
                max_mediation=("mediation", "max"),
            )
        )
        agg["relation"] = rel
        agg["discovery_N_relation"] = denom
        agg["selection_rate"] = agg["n_samples_selected"] / denom

        agg = agg[agg["selection_rate"] >= float(min_rate)]
        agg = agg.sort_values(
            ["selection_rate", "mean_mediation", "max_mediation"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        agg["relation_rank"] = np.arange(1, len(agg) + 1)

        if per_relation_candidates and per_relation_candidates > 0:
            chosen = agg.head(int(per_relation_candidates)).copy()
        else:
            chosen = agg.copy()

        chosen["chosen_for_union"] = True
        relation_tables.append(agg)

        for row in chosen.to_dict("records"):
            picked_membership[row["candidate"]].append(rel)

    all_rel = (
        pd.concat(relation_tables, ignore_index=True)
        if relation_tables
        else pd.DataFrame()
    )

    if not picked_membership:
        raise RuntimeError("Union candidate pool is empty")

    # Pull canonical metadata from the relation tables.
    meta_lookup = {}
    for r in all_rel.to_dict("records"):
        meta_lookup[r["candidate"]] = {
            "source_layer": int(r["source_layer"]),
            "candidate_kind": str(r["candidate_kind"]),
            "candidate_key": str(r["candidate_key"]),
        }

    pool_rows = []
    for cand, rels in picked_membership.items():
        sub = all_rel[all_rel["candidate"] == cand]
        meta = meta_lookup[cand]

        pool_rows.append({
            "candidate": cand,
            **meta,
            "n_relations_selected_for_union": len(rels),
            "relations_selected_for_union": ",".join(
                r for r in REL_DISPLAY if r in rels
            ),
            "max_selection_rate_across_relations": float(
                sub["selection_rate"].max()
            ),
            "mean_selection_rate_across_present_relations": float(
                sub["selection_rate"].mean()
            ),
            "max_mean_mediation_across_relations": float(
                sub["mean_mediation"].max()
            ),
        })

    pool = pd.DataFrame(pool_rows).sort_values(
        [
            "n_relations_selected_for_union",
            "max_selection_rate_across_relations",
            "max_mean_mediation_across_relations",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return pool, all_rel


# =============================================================================
# Hidden-state capture and editing
# =============================================================================

class Capture:
    def __init__(self, decoder_layers, layers):
        self.states = {}
        self.handles = []
        for L in layers:
            self.handles.append(
                decoder_layers[L].register_forward_hook(self._hook(L))
            )

    def _hook(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            self.states[L] = h.detach().float().cpu()
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_states(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing captured source layers: {missing}")
        return {
            L: cap.states[L][0].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


class WeightedDeltaEditor:
    def __init__(
        self,
        decoder_layers,
        specs_by_layer,
        alpha,
        prompt_len,
    ):
        """
        specs_by_layer[L] = list of (position, delta_np, local_weight)
        """
        self.handles = []
        self.applied = defaultdict(int)
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)

        for L, entries in specs_by_layer.items():
            if entries:
                self.handles.append(
                    decoder_layers[L].register_forward_hook(
                        self._hook(L, entries)
                    )
                )

    def _hook(self, L, entries):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)

            # Prefill only; do not repeatedly edit cached one-token decode.
            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()
            for pos, delta_np, local_weight in entries:
                pos = int(pos)
                if 0 <= pos < y.shape[1]:
                    d = torch.as_tensor(
                        delta_np,
                        device=y.device,
                        dtype=torch.float32,
                    )
                    y[:, pos, :] = (
                        y[:, pos, :]
                        + self.alpha * float(local_weight) * d
                    )
                    self.applied[L] += 1

            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def generate_with_specs(
    model,
    processor,
    decoder_layers,
    batch,
    specs_by_layer,
    alpha,
    max_new_tokens,
):
    prompt_len = int(batch["input_ids"].shape[1])
    editor = WeightedDeltaEditor(
        decoder_layers,
        specs_by_layer,
        alpha,
        prompt_len,
    )
    try:
        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )
        pred = traj.normalize_relation(base, text)
        return text, pred, dict(editor.applied)
    finally:
        editor.close()


# =============================================================================
# Relation-free endogenous candidate scoring
# =============================================================================

def local_score(mode, hreal, delta):
    dn = float(np.linalg.norm(delta))
    if mode == "delta_norm":
        return dn
    if mode == "relative_delta":
        denom = float(np.linalg.norm(hreal)) + 1e-8
        return dn / denom
    raise ValueError(mode)


def resolve_current_candidates(
    pool,
    mapping,
    hreal_by_layer,
    hgray_by_layer,
    score_mode,
):
    """
    Each canonical candidate resolves to ONE actual current-sample token.

    If a canonical unit contains multiple actual positions (e.g. multi-subtoken
    <REF>, <SUBJ>, or a visual bin), choose the position with the largest
    endogenous score.  This is relation-free local competition.
    """
    resolved = []

    for row in pool.to_dict("records"):
        L = int(row["source_layer"])
        kind = str(row["candidate_kind"])
        key = str(row["candidate_key"])

        positions = []
        for pos, info in mapping.items():
            if kind == "visual":
                if (
                    info["kind"] == "visual"
                    and str(info["visual_bin"]) == key
                ):
                    positions.append(int(pos))
            else:
                if (
                    info["kind"] == "text"
                    and str(info["canonical_slot"]) == key
                ):
                    positions.append(int(pos))

        if not positions:
            continue

        best = None
        for pos in positions:
            if (
                pos >= hreal_by_layer[L].shape[0]
                or pos >= hgray_by_layer[L].shape[0]
            ):
                continue

            hr = hreal_by_layer[L][pos]
            hg = hgray_by_layer[L][pos]
            delta = (hr - hg).astype(np.float32)
            score = local_score(score_mode, hr, delta)

            cand = {
                **row,
                "position": int(pos),
                "score": float(score),
                "delta_norm": float(np.linalg.norm(delta)),
                "real_norm": float(np.linalg.norm(hr)),
                "delta_h": delta,
                "n_positions_in_canonical_unit": len(positions),
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

        if best is not None:
            resolved.append(best)

    return resolved


def make_weighted_specs(
    resolved,
    max_weight,
    weight_power,
):
    if not resolved:
        return {}, []

    scores = np.asarray(
        [max(0.0, float(r["score"])) for r in resolved],
        np.float64,
    )
    scores = np.power(scores, float(weight_power))

    mean_score = float(scores.mean())
    if mean_score < 1e-12:
        weights = np.ones_like(scores)
    else:
        weights = scores / mean_score

    weights = np.clip(weights, 0.0, float(max_weight))

    specs = defaultdict(list)
    details = []

    for r, w in zip(resolved, weights):
        specs[int(r["source_layer"])].append(
            (
                int(r["position"]),
                r["delta_h"],
                float(w),
            )
        )
        details.append({
            **r,
            "local_weight": float(w),
        })

    return dict(specs), details


def make_topk_specs(resolved, k):
    chosen = sorted(
        resolved,
        key=lambda r: float(r["score"]),
        reverse=True,
    )[:int(k)]

    specs = defaultdict(list)
    for r in chosen:
        specs[int(r["source_layer"])].append(
            (
                int(r["position"]),
                r["delta_h"],
                1.0,
            )
        )

    return dict(specs), chosen


# =============================================================================
# Summary
# =============================================================================

def summarize(baseline_rows, condition_rows):
    base = {int(r["sid"]): r for r in baseline_rows}
    base_acc = safe_mean(
        r["baseline_correct"] for r in baseline_rows
    )

    groups = defaultdict(list)
    for r in condition_rows:
        groups[
            (
                r["method"],
                r["score_mode"],
                int(r["k"]),
                float(r["alpha"]),
            )
        ].append(r)

    rows = []
    for (method, score_mode, k, alpha), rr in groups.items():
        w2c = c2w = changed = 0
        corr = []

        for r in rr:
            sid = int(r["sid"])
            b = base[sid]
            c = bool(r["correct"])
            corr.append(c)
            w2c += int((not bool(b["baseline_correct"])) and c)
            c2w += int(bool(b["baseline_correct"]) and (not c))
            changed += int(
                str(r["prediction"]) != str(b["baseline_prediction"])
            )

        acc = safe_mean(corr)
        rows.append({
            "method": method,
            "score_mode": score_mode,
            "k": k,
            "alpha": alpha,
            "N": len(rr),
            "baseline_acc": base_acc,
            "edited_acc": acc,
            "gain": acc - base_acc,
            "W2C": w2c,
            "C2W": c2w,
            "net": w2c - c2w,
            "changed": changed,
            "mean_n_edited": safe_mean(
                r["n_edited"] for r in rr
            ),
        })

    return sorted(
        rows,
        key=lambda r: (r["edited_acc"], r["net"]),
        reverse=True,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    score_modes = parse_csv_strings(a.score_modes)
    methods = parse_csv_strings(a.methods)
    topks = parse_ints(a.topks)
    alphas = parse_floats(a.alphas)

    for x in score_modes:
        if x not in {"delta_norm", "relative_delta"}:
            raise ValueError(f"Unknown score mode: {x}")
    for x in methods:
        if x not in {"weighted", "topk"}:
            raise ValueError(f"Unknown method: {x}")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data / prompt metadata.
    # -------------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue

        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL_INTERNAL:
            continue

        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    _train, test = traj.stratified_split(
        meta, a.train_ratio, a.seed
    )

    # -------------------------------------------------------------------------
    # Load processor first; build reusable union pool from discovery samples.
    # -------------------------------------------------------------------------
    specs = base.merged_model_specs(two)
    spec = specs[a.model]

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )

    sel = pd.read_csv(a.selected_csv)
    sel["relation"] = sel["relation"].map(canon_rel)
    sel["k"] = pd.to_numeric(
        sel["k"], errors="raise"
    ).astype(int)
    sel["source_layer"] = pd.to_numeric(
        sel["source_layer"], errors="raise"
    ).astype(int)
    sel["position"] = pd.to_numeric(
        sel["position"], errors="raise"
    ).astype(int)
    sel["mediation"] = pd.to_numeric(
        sel["mediation"], errors="coerce"
    )

    sel = sel[
        (sel["condition"].astype(str) == str(a.discovery_condition))
        & (
            sel["source_bundle"].astype(str)
            == str(a.discovery_bundle)
        )
        & (sel["k"] == int(a.discovery_k))
        & (sel["relation"].isin(REL_DISPLAY))
    ].copy()

    if not len(sel):
        raise RuntimeError(
            "No discovery rows after filtering selected_tokens.csv for "
            f"condition={a.discovery_condition}, "
            f"bundle={a.discovery_bundle}, K={a.discovery_k}"
        )

    # Defensive dedupe.
    sel = sel.sort_values(
        "mediation", ascending=False
    ).drop_duplicates(
        ["sid", "relation", "source_layer", "position"],
        keep="first",
    )

    discovery_sids = sorted(set(sel["sid"].astype(int)))

    ann = canonicalize_discovery_rows(
        sel,
        prompts,
        rec_by_sid,
        processor,
        a.visual_bins,
    )
    ann.to_csv(
        outdir / "discovery_k24_canonical_rows.csv",
        index=False,
    )

    pool, rel_table = build_union_pool(
        ann,
        a.per_relation_candidates,
        a.candidate_min_rate,
    )
    pool.to_csv(
        outdir / "union_candidate_pool.csv",
        index=False,
    )
    rel_table.to_csv(
        outdir / "per_relation_candidate_ranking.csv",
        index=False,
    )

    source_layers = sorted(
        set(pool["source_layer"].astype(int))
    )

    # -------------------------------------------------------------------------
    # Evaluation split.  GT is used only for final scoring.
    # -------------------------------------------------------------------------
    discovery_set = set(discovery_sids)

    if a.eval_scope == "unseen":
        eval_pool = [
            r for r in test
            if int(r["sid"]) not in discovery_set
        ]
    elif a.eval_scope == "selected":
        eval_pool = [
            r for r in test
            if int(r["sid"]) in discovery_set
        ]
    else:
        eval_pool = list(test)

    eval_set = stratified_take(
        eval_pool,
        a.eval_max_samples,
        a.seed + 101,
    )

    if not eval_set:
        raise RuntimeError("Evaluation set is empty")

    # -------------------------------------------------------------------------
    # Model.
    # -------------------------------------------------------------------------
    model_cls = getattr(transformers, spec.model_class)
    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    print("=" * 150)
    print("RELATION-FREE UNION SELF-ACTIVITY REPAIR")
    print("=" * 150)
    print(f"model={a.model} | repo={spec.repo_id}")
    print(
        f"discovery: bundle={a.discovery_bundle} "
        f"K={a.discovery_k} | N_sids={len(discovery_sids)}"
    )
    print(
        f"union pool={len(pool)} canonical candidates "
        f"from top {a.per_relation_candidates or 'ALL'} per relation"
    )
    print(f"source layers={source_layers}")
    print(
        f"eval_scope={a.eval_scope} | eval N={len(eval_set)} | "
        f"discovery/eval overlap="
        f"{len(discovery_set & set(int(r['sid']) for r in eval_set))}"
    )
    print(f"methods={methods}")
    print(f"score_modes={score_modes}")
    print(f"topKs={topks} | alphas={alphas}")
    print()
    print(
        "INTERVENTION GUARANTEE: no GT relation, no relation selector, "
        "no late writer, no gradient, no answer logits at evaluation time."
    )
    print()

    print("Union candidate pool:")
    for i, r in pool.head(60).iterrows():
        print(
            f"  {r['candidate']:<48s} | "
            f"rels={r['relations_selected_for_union']:<20s} | "
            f"maxRate={float(r['max_selection_rate_across_relations']):.3f}"
        )
    if len(pool) > 60:
        print(f"  ... {len(pool)-60} more")
    print()

    model = None

    try:
        model = model_cls.from_pretrained(
            spec.repo_id,
            **load_kw,
        )
        model.eval()
        base.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        for L in source_layers:
            if not (0 <= L < n_layers):
                raise ValueError(
                    f"Candidate source L{L} invalid for "
                    f"{a.model} with {n_layers} layers"
                )

        device = torch.device(a.device)

        baseline_rows = []
        condition_rows = []
        resolved_score_rows = []

        for m in tqdm(eval_set, desc="Self-activity repair"):
            sid = int(m["sid"])
            gt = m["gt"]

            real = gray = rb = gb = None

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                toks = [
                    str(x)
                    for x in processor.tokenizer.convert_ids_to_tokens(ids)
                ]
                spos, rpos = get_object_spans(
                    processor.tokenizer,
                    ids,
                    m["subject"],
                    m["reference"],
                )
                vpos = visual_positions_from_tokens(toks)
                mapping = build_canonical_mapping(
                    toks,
                    spos,
                    rpos,
                    vpos,
                    a.visual_bins,
                )

                hreal = capture_states(
                    model,
                    decoder_layers,
                    rb,
                    source_layers,
                )
                hgray = capture_states(
                    model,
                    decoder_layers,
                    gb,
                    source_layers,
                )

                # Baseline generation.
                baseline_text = base.generate_text(
                    model,
                    processor,
                    rb,
                    max_new_tokens=a.max_new_tokens,
                )
                baseline_pred = traj.normalize_relation(
                    base, baseline_text
                )

                baseline_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "baseline_prediction": DISPLAY.get(
                        baseline_pred, baseline_pred
                    ),
                    "baseline_correct": baseline_pred == gt,
                    "baseline_text": baseline_text,
                })

                for score_mode in score_modes:
                    resolved = resolve_current_candidates(
                        pool,
                        mapping,
                        hreal,
                        hgray,
                        score_mode,
                    )

                    for rank, r in enumerate(
                        sorted(
                            resolved,
                            key=lambda x: x["score"],
                            reverse=True,
                        ),
                        1,
                    ):
                        resolved_score_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "score_mode": score_mode,
                            "rank": rank,
                            "candidate": r["candidate"],
                            "source_layer": r["source_layer"],
                            "candidate_kind": r["candidate_kind"],
                            "candidate_key": r["candidate_key"],
                            "position": r["position"],
                            "score": r["score"],
                            "delta_norm": r["delta_norm"],
                            "real_norm": r["real_norm"],
                            "n_positions_in_canonical_unit":
                                r["n_positions_in_canonical_unit"],
                            "offline_relations":
                                r["relations_selected_for_union"],
                            "offline_max_selection_rate":
                                r["max_selection_rate_across_relations"],
                        })

                    if "weighted" in methods:
                        specs_weighted, weighted_details = make_weighted_specs(
                            resolved,
                            a.max_weight,
                            a.weight_power,
                        )

                        for alpha in alphas:
                            text, pred, applied = generate_with_specs(
                                model,
                                processor,
                                decoder_layers,
                                rb,
                                specs_weighted,
                                alpha,
                                a.max_new_tokens,
                            )

                            condition_rows.append({
                                "sid": sid,
                                "gt": DISPLAY[gt],
                                "method": "self_weighted_union",
                                "score_mode": score_mode,
                                "k": 0,
                                "alpha": alpha,
                                "prediction": DISPLAY.get(pred, pred),
                                "correct": pred == gt,
                                "text": text,
                                "n_edited": sum(applied.values()),
                                "n_resolved_candidates": len(resolved),
                                "mean_local_weight": safe_mean(
                                    x["local_weight"]
                                    for x in weighted_details
                                ),
                                "max_local_weight": (
                                    max(
                                        [
                                            float(x["local_weight"])
                                            for x in weighted_details
                                        ]
                                        or [float("nan")]
                                    )
                                ),
                            })

                    if "topk" in methods:
                        for k in topks:
                            specs_topk, chosen = make_topk_specs(
                                resolved, k
                            )
                            if not chosen:
                                continue

                            for alpha in alphas:
                                text, pred, applied = generate_with_specs(
                                    model,
                                    processor,
                                    decoder_layers,
                                    rb,
                                    specs_topk,
                                    alpha,
                                    a.max_new_tokens,
                                )

                                condition_rows.append({
                                    "sid": sid,
                                    "gt": DISPLAY[gt],
                                    "method": "self_topk",
                                    "score_mode": score_mode,
                                    "k": int(k),
                                    "alpha": alpha,
                                    "prediction": DISPLAY.get(pred, pred),
                                    "correct": pred == gt,
                                    "text": text,
                                    "n_edited": sum(applied.values()),
                                    "n_resolved_candidates": len(resolved),
                                    "top_candidates": " | ".join(
                                        x["candidate"]
                                        for x in chosen
                                    ),
                                })

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        summary = summarize(
            baseline_rows,
            condition_rows,
        )

        # Per-relation summary; relation is used ONLY here for evaluation.
        base_lookup = {
            int(r["sid"]): r for r in baseline_rows
        }

        per_rel_rows = []
        for s in summary:
            rr = [
                r for r in condition_rows
                if r["method"] == s["method"]
                and r["score_mode"] == s["score_mode"]
                and int(r["k"]) == int(s["k"])
                and float(r["alpha"]) == float(s["alpha"])
            ]

            for rel in REL_DISPLAY:
                rel_rows = [
                    r for r in rr if r["gt"] == rel
                ]
                if not rel_rows:
                    continue

                sids = [int(r["sid"]) for r in rel_rows]
                bacc = safe_mean(
                    base_lookup[sid]["baseline_correct"]
                    for sid in sids
                )
                eacc = safe_mean(
                    r["correct"] for r in rel_rows
                )
                w2c = sum(
                    (not base_lookup[int(r["sid"])]["baseline_correct"])
                    and bool(r["correct"])
                    for r in rel_rows
                )
                c2w = sum(
                    base_lookup[int(r["sid"])]["baseline_correct"]
                    and (not bool(r["correct"]))
                    for r in rel_rows
                )

                per_rel_rows.append({
                    "method": s["method"],
                    "score_mode": s["score_mode"],
                    "k": s["k"],
                    "alpha": s["alpha"],
                    "relation": rel,
                    "N": len(rel_rows),
                    "baseline_acc": bacc,
                    "edited_acc": eacc,
                    "gain": eacc - bacc,
                    "W2C": w2c,
                    "C2W": c2w,
                    "net": w2c - c2w,
                })

        write_csv(
            outdir / "baseline.csv",
            baseline_rows,
        )
        write_csv(
            outdir / "generation_conditions.csv",
            condition_rows,
        )
        write_csv(
            outdir / "resolved_candidate_scores.csv",
            resolved_score_rows,
        )
        write_csv(
            outdir / "summary.csv",
            summary,
        )
        write_csv(
            outdir / "per_relation_summary.csv",
            per_rel_rows,
        )

        print("\n" + "=" * 150)
        print("RESULTS")
        print("=" * 150)

        base_acc = safe_mean(
            r["baseline_correct"]
            for r in baseline_rows
        )
        print(
            f"Baseline: N={len(baseline_rows)} "
            f"acc={base_acc:.4f}"
        )
        print()
        print(
            f"{'method':<22s} {'score':<15s} {'K':>4s} "
            f"{'alpha':>7s} {'acc':>8s} {'gain':>8s} "
            f"{'W2C':>5s} {'C2W':>5s} {'net':>5s} "
            f"{'nEdit':>7s}"
        )
        print("-" * 150)

        for r in summary[:60]:
            print(
                f"{r['method']:<22s} "
                f"{r['score_mode']:<15s} "
                f"{int(r['k']):>4d} "
                f"{float(r['alpha']):>7.3f} "
                f"{float(r['edited_acc']):>8.4f} "
                f"{float(r['gain']):>+8.4f} "
                f"{int(r['W2C']):>5d} "
                f"{int(r['C2W']):>5d} "
                f"{int(r['net']):>5d} "
                f"{float(r['mean_n_edited']):>7.2f}"
            )

        print("\nBest per relation:")
        pr = pd.DataFrame(per_rel_rows)
        if len(pr):
            for rel in REL_DISPLAY:
                q = pr[pr["relation"] == rel].sort_values(
                    ["edited_acc", "net"],
                    ascending=[False, False],
                )
                if len(q):
                    r = q.iloc[0]
                    print(
                        f"  {rel:<5s} | "
                        f"{r['method']} / {r['score_mode']} "
                        f"K={int(r['k'])} a={float(r['alpha']):.3f} | "
                        f"{float(r['baseline_acc']):.4f}"
                        f"->{float(r['edited_acc']):.4f} "
                        f"({float(r['gain']):+.4f}) | "
                        f"W2C/C2W={int(r['W2C'])}/{int(r['C2W'])}"
                    )

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "eval_scope": a.eval_scope,
            "discovery_selected_csv": str(a.selected_csv),
            "discovery_bundle": a.discovery_bundle,
            "discovery_k": a.discovery_k,
            "discovery_condition": a.discovery_condition,
            "discovery_N_sids": len(discovery_sids),
            "per_relation_candidates": a.per_relation_candidates,
            "candidate_min_rate": a.candidate_min_rate,
            "union_pool_N": len(pool),
            "source_layers": source_layers,
            "eval_N": len(eval_set),
            "discovery_eval_overlap": len(
                discovery_set
                & set(int(r["sid"]) for r in eval_set)
            ),
            "score_modes": score_modes,
            "methods": methods,
            "topks": topks,
            "alphas": alphas,
            "max_weight": a.max_weight,
            "weight_power": a.weight_power,
            "gt_relation_used_for_intervention": False,
            "relation_selector_used": False,
            "late_writer_used_at_evaluation": False,
            "gradient_used_at_evaluation": False,
            "answer_logits_used_for_intervention": False,
            "offline_oracle_used_only_for_candidate_pool_discovery": True,
            "edit": (
                "h_real[S,p] += alpha * local_weight * "
                "(h_real[S,p]-h_gray[S,p])"
            ),
        }

        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("\nSaved:", outdir)

    finally:
        if model is not None:
            del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
