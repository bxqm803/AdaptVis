#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype-matched four-template repair for Qwen VL on COCO_two.

Goal
----
Constrain the adaptive intervention to FOUR FIXED relation templates:

    T_left, T_right, T_on, T_under

At evaluation time, the model does NOT select individual tokens and does NOT
use GT relation.  It only matches its current middle-layer Real-Gray state
against four fixed relation prototypes, then either:

  HARD:
      choose the single best matching template and amplify the whole template

  SOFT:
      softmax over the four template scores and mix the four fixed templates

Controls:
  ORACLE:
      choose the GT template (ceiling only)

  RANDOM:
      choose one of the four templates randomly

Offline discovery / calibration
-------------------------------
1) Build each fixed K24 template from an existing selected_tokens.csv produced
   by the oracle writer-guided middle-token experiment.
2) On those discovery samples, estimate a prototype for every
   (relation, canonical candidate):

       mu[r,c] = mean_i delta_h[i,c],   delta_h = h_real - h_gray

3) Center each prototype across relations:

       v[r,c] = mu[r,c] - mean_r' mu[r',c]

Evaluation-time template score
------------------------------
For an unseen sample x:

    delta_x[c] = h_real[x,c] - h_gray[x,c]

    score_r(x) =
        weighted_mean_{c in T_r} cosine(delta_x[c], v[r,c])

By default, answer relation-word positions (left/right/above/below) are
EXCLUDED from template MATCHING, but they can remain inside the template
INTERVENTION after a template has been chosen.

Thus the routing signal comes from grounded/object/visual/structural middle
states rather than directly looking at answer relation tokens.

Hard:
    r* = argmax_r score_r
    amplify every candidate in T_r*

Soft:
    w_r = softmax(score_r / tau)

    For candidate c:
        local_weight[c] = sum_{r: c in T_r} w_r

    h'[c] = h_real[c] + alpha * local_weight[c] * delta_x[c]

Shared candidates naturally receive the summed template mass.  If a candidate
appears in all four templates, its local weight is exactly 1.

IMPORTANT
---------
Evaluation-time HARD/SOFT uses:
  - no GT relation
  - no late writer
  - no gradient
  - no answer logits
  - no learned probe/classifier
  - no per-token Top-K selection

It does use a fixed offline template/prototype bank learned from discovery
samples.  Default --eval-scope unseen excludes those discovery sample IDs.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_prototype_matched_four_templates_v1.py \
  --model qwen-3b \
  --selected-csv output/qwen3b_coco_minimal_token_search/selected_tokens.csv \
  --discovery-bundle "L22+L24+L26[global_unique]" \
  --discovery-k 24 \
  --template-k 24 \
  --score-weight frequency \
  --exclude-relation-words-from-score \
  --methods hard,soft,oracle,random \
  --soft-temperatures 0.25,0.5,1.0 \
  --alphas 0.5,0.75,1.0 \
  --eval-scope unseen \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_coco_prototype_four_templates_unseen80 \
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
REL = ("left", "right", "on", "under")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}


# =============================================================================
# CLI / generic helpers
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
    p.add_argument("--discovery-bundle", default="L22+L24+L26[global_unique]")
    p.add_argument("--discovery-k", type=int, default=24)
    p.add_argument("--discovery-condition", default="positive")
    p.add_argument("--template-k", type=int, default=24)
    p.add_argument("--candidate-min-rate", type=float, default=0.0)
    p.add_argument("--visual-bins", type=int, default=16)

    p.add_argument(
        "--score-weight",
        default="frequency",
        choices=["uniform", "frequency"],
        help="Fixed weighting of candidates when computing each template score.",
    )
    p.add_argument(
        "--exclude-relation-words-from-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude canonical positions containing left/right/above/below "
            "from template MATCHING. They can still be edited after routing."
        ),
    )
    p.add_argument(
        "--exclude-answer-structural-from-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also exclude canonical tokens Answer/assistant from template matching."
        ),
    )

    p.add_argument(
        "--methods",
        default="hard,soft,oracle,random",
        help="Comma-separated subset of hard,soft,oracle,random.",
    )
    p.add_argument("--soft-temperatures", default="0.25,0.5,1.0")
    p.add_argument("--alphas", default="0.5,0.75,1.0")

    p.add_argument(
        "--eval-scope",
        default="unseen",
        choices=["unseen", "selected", "all_test"],
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


def parse_strings(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


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


def write_csv(path, rows):
    path = Path(path)
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


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def softmax_np(xs, temperature):
    x = np.asarray(xs, np.float64)
    temp = max(float(temperature), 1e-6)
    x = x / temp
    x = x - np.nanmax(x)
    ex = np.exp(x)
    ex[~np.isfinite(ex)] = 0.0
    s = float(ex.sum())
    if s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s


# =============================================================================
# Canonical prompt positions
# =============================================================================

def visual_positions(tokens):
    return {
        i
        for i, tok in enumerate(tokens)
        if "image_pad" in str(tok) or "video_pad" in str(tok)
    }


def object_positions(tokenizer, ids, subject, reference):
    sspan, rspan = base.locate_object_spans(tokenizer, ids, subject, reference)
    spos = set(range(int(sspan[0]), int(sspan[1]) + 1))
    rpos = set(range(int(rspan[0]), int(rspan[1]) + 1))
    return spos, rpos


def build_canonical_mapping(tokens, spos, rpos, vpos, visual_bins):
    """
    Mapping:
        original token position ->
            kind = text / visual
            key  = exact canonical text slot or normalized visual bin
    """
    mapping = {}
    n = len(tokens)
    c = 0

    vis_sorted = sorted(vpos)
    vis_ord = {p: j for j, p in enumerate(vis_sorted)}
    nvis = len(vis_sorted)

    i = 0
    while i < n:
        if i in vpos:
            block = []
            while i < n and i in vpos:
                block.append(i)
                i += 1

            for p in block:
                j = vis_ord[p]
                if nvis <= 1:
                    b = 0
                else:
                    frac = j / (nvis - 1)
                    b = min(visual_bins - 1, int(frac * visual_bins))

                mapping[p] = {
                    "kind": "visual",
                    "key": f"VISBIN_{b:02d}",
                    "canonical_slot": f"C{c:03d}:<VISUAL_BLOCK>",
                }

            c += 1
            continue

        if i in spos:
            block = []
            while i < n and i in spos:
                block.append(i)
                i += 1
            label = f"C{c:03d}:<SUBJ>"
            for p in block:
                mapping[p] = {
                    "kind": "text",
                    "key": label,
                    "canonical_slot": label,
                }
            c += 1
            continue

        if i in rpos:
            block = []
            while i < n and i in rpos:
                block.append(i)
                i += 1
            label = f"C{c:03d}:<REF>"
            for p in block:
                mapping[p] = {
                    "kind": "text",
                    "key": label,
                    "canonical_slot": label,
                }
            c += 1
            continue

        tok = str(tokens[i]).replace("\n", "\\n")
        label = f"C{c:03d}:{tok}"
        mapping[i] = {
            "kind": "text",
            "key": label,
            "canonical_slot": label,
        }
        c += 1
        i += 1

    return mapping


def candidate_id(layer, kind, key):
    if kind == "visual":
        return f"L{int(layer)}::<VISUAL>::{key}"
    return f"L{int(layer)}::{key}"


def is_relation_word_candidate(candidate):
    x = str(candidate).lower()
    return any(
        word in x
        for word in ("ġleft", "ġright", "ġabove", "ġbelow", "::left", "::right", "::above", "::below")
    )


def is_answer_structural_candidate(candidate):
    x = str(candidate).lower()
    return ("answer" in x) or ("assistant" in x)


def resolve_candidate_position(candidate_row, mapping):
    """
    Strongly constrained resolution:
    each canonical candidate resolves to exactly ONE current token position.
    For multi-token units / visual bins, use the center position, not dynamic
    max-norm selection.
    """
    kind = str(candidate_row["candidate_kind"])
    key = str(candidate_row["candidate_key"])

    positions = sorted(
        p
        for p, info in mapping.items()
        if info["kind"] == kind and str(info["key"]) == key
    )

    if not positions:
        return None

    return positions[len(positions) // 2]


# =============================================================================
# Template discovery
# =============================================================================

def canonicalize_discovery(sel, prompts, rec_by_sid, processor, visual_bins):
    tokenizer = processor.tokenizer
    maps = {}

    for sid in tqdm(
        sorted(set(sel["sid"].astype(int))),
        desc="Canonicalize discovery",
    ):
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
            spos, rpos = object_positions(
                tokenizer,
                ids,
                str(p["subject"]),
                str(p["reference"]),
            )
            vpos = visual_positions(toks)

            maps[sid] = build_canonical_mapping(
                toks,
                spos,
                rpos,
                vpos,
                visual_bins,
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

        rows.append({
            **r,
            "candidate_kind": info["kind"],
            "candidate_key": info["key"],
            "candidate": candidate_id(
                int(r["source_layer"]),
                info["kind"],
                info["key"],
            ),
        })

    return pd.DataFrame(rows)


def build_templates(ann, template_k, min_rate):
    templates = {}
    ranking_rows = []

    for rel in REL:
        rr = ann[ann["relation"] == rel].copy()
        denom = int(rr["sid"].nunique())
        if denom == 0:
            raise RuntimeError(f"No discovery samples for relation={rel}")

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
        agg["discovery_N"] = denom
        agg["selection_rate"] = agg["n_samples_selected"] / denom

        agg = agg[
            agg["selection_rate"] >= float(min_rate)
        ].copy()

        agg = agg.sort_values(
            ["selection_rate", "mean_mediation", "max_mediation"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        agg["rank"] = np.arange(1, len(agg) + 1)
        agg["in_template"] = agg["rank"] <= int(template_k)

        templates[rel] = agg.head(int(template_k)).copy()
        ranking_rows.append(agg)

    return templates, pd.concat(ranking_rows, ignore_index=True)


def template_overlap_rows(templates):
    rows = []

    for i, ra in enumerate(REL):
        A = set(templates[ra]["candidate"].astype(str))
        for rb in REL[i + 1:]:
            B = set(templates[rb]["candidate"].astype(str))
            inter = A & B
            union = A | B

            rows.append({
                "template_a": ra,
                "template_b": rb,
                "n_a": len(A),
                "n_b": len(B),
                "intersection": len(inter),
                "union": len(union),
                "jaccard": len(inter) / len(union) if union else float("nan"),
                "shared_candidates": " | ".join(sorted(inter)),
            })

    return rows


# =============================================================================
# Hidden-state capture
# =============================================================================

class Capture:
    def __init__(self, decoder_layers, layers):
        self.states = {}
        self.handles = []

        for L in layers:
            self.handles.append(
                decoder_layers[L].register_forward_hook(
                    self._hook(L)
                )
            )

    def _hook(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            self.states[L] = h.detach().float().cpu()
            return out
        return hook

    def close(self):
        for handle in self.handles:
            with contextlib.suppress(Exception):
                handle.remove()


@torch.inference_mode()
def capture_states(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers)

    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)

        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing captured layers: {missing}")

        return {
            L: cap.states[L][0].numpy().astype(np.float32)
            for L in layers
        }

    finally:
        cap.close()


# =============================================================================
# Prototype bank
# =============================================================================

def union_candidate_meta(templates):
    meta = {}

    for rel in REL:
        for r in templates[rel].to_dict("records"):
            cand = str(r["candidate"])
            if cand not in meta:
                meta[cand] = {
                    "candidate": cand,
                    "source_layer": int(r["source_layer"]),
                    "candidate_kind": str(r["candidate_kind"]),
                    "candidate_key": str(r["candidate_key"]),
                }

    return meta


def estimate_prototypes(
    discovery_meta,
    candidate_meta,
    prompts,
    rec_by_sid,
    processor,
    model,
    decoder_layers,
    device,
    visual_bins,
    gray_value,
):
    source_layers = sorted({
        int(x["source_layer"])
        for x in candidate_meta.values()
    })

    sums = {
        rel: {
            cand: None
            for cand in candidate_meta
        }
        for rel in REL
    }
    counts = {
        rel: {
            cand: 0
            for cand in candidate_meta
        }
        for rel in REL
    }

    for m in tqdm(discovery_meta, desc="Estimate relation prototypes"):
        sid = int(m["sid"])
        rel = DISPLAY[m["gt"]]

        real = gray = rb = gb = None

        try:
            real = base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")

            gray = make_gray_image(real, gray_value)

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
            spos, rpos = object_positions(
                processor.tokenizer,
                ids,
                m["subject"],
                m["reference"],
            )
            vpos = visual_positions(toks)
            mapping = build_canonical_mapping(
                toks,
                spos,
                rpos,
                vpos,
                visual_bins,
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

            for cand, meta in candidate_meta.items():
                p = resolve_candidate_position(meta, mapping)
                if p is None:
                    continue

                L = int(meta["source_layer"])
                if (
                    p >= hreal[L].shape[0]
                    or p >= hgray[L].shape[0]
                ):
                    continue

                delta = (
                    hreal[L][p] - hgray[L][p]
                ).astype(np.float32)

                if sums[rel][cand] is None:
                    sums[rel][cand] = delta.copy()
                else:
                    sums[rel][cand] += delta
                counts[rel][cand] += 1

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

    means = {
        rel: {}
        for rel in REL
    }

    for rel in REL:
        for cand in candidate_meta:
            n = counts[rel][cand]
            if n > 0:
                means[rel][cand] = (
                    sums[rel][cand] / n
                ).astype(np.float32)
            else:
                means[rel][cand] = None

    centered = {
        rel: {}
        for rel in REL
    }

    proto_rows = []

    for cand, meta in candidate_meta.items():
        available = [
            means[rel][cand]
            for rel in REL
            if means[rel][cand] is not None
        ]

        if not available:
            common = None
        else:
            common = np.mean(
                np.stack(available),
                axis=0,
            ).astype(np.float32)

        for rel in REL:
            mu = means[rel][cand]

            if mu is None or common is None:
                v = None
                mu_norm = float("nan")
                v_norm = float("nan")
            else:
                v = (mu - common).astype(np.float32)
                mu_norm = float(np.linalg.norm(mu))
                v_norm = float(np.linalg.norm(v))

            centered[rel][cand] = v

            proto_rows.append({
                "relation": rel,
                "candidate": cand,
                "source_layer": meta["source_layer"],
                "candidate_kind": meta["candidate_kind"],
                "candidate_key": meta["candidate_key"],
                "N": counts[rel][cand],
                "mu_norm": mu_norm,
                "centered_proto_norm": v_norm,
            })

    return centered, proto_rows


# =============================================================================
# Template matching
# =============================================================================

def template_membership(templates):
    return {
        rel: set(templates[rel]["candidate"].astype(str))
        for rel in REL
    }


def fixed_score_weight_lookup(templates, mode):
    out = {
        rel: {}
        for rel in REL
    }

    for rel in REL:
        for r in templates[rel].to_dict("records"):
            cand = str(r["candidate"])

            if mode == "uniform":
                w = 1.0
            else:
                w = float(r["selection_rate"])

            out[rel][cand] = w

    return out


def resolve_current_deltas(
    candidate_meta,
    mapping,
    hreal,
    hgray,
):
    current = {}

    for cand, meta in candidate_meta.items():
        p = resolve_candidate_position(meta, mapping)
        if p is None:
            continue

        L = int(meta["source_layer"])
        if (
            p >= hreal[L].shape[0]
            or p >= hgray[L].shape[0]
        ):
            continue

        delta = (
            hreal[L][p] - hgray[L][p]
        ).astype(np.float32)

        current[cand] = {
            **meta,
            "position": int(p),
            "delta_h": delta,
            "delta_norm": float(np.linalg.norm(delta)),
        }

    return current


def compute_template_scores(
    current,
    templates,
    prototypes,
    score_weights,
    exclude_relation_words,
    exclude_answer_structural,
):
    scores = {}
    detail_rows = []

    for rel in REL:
        vals = []
        weights = []

        for r in templates[rel].to_dict("records"):
            cand = str(r["candidate"])

            if exclude_relation_words and is_relation_word_candidate(cand):
                continue

            if exclude_answer_structural and is_answer_structural_candidate(cand):
                continue

            if cand not in current:
                continue

            proto = prototypes[rel].get(cand)
            if proto is None:
                continue

            cs = cosine_np(
                current[cand]["delta_h"],
                proto,
            )

            if not np.isfinite(cs):
                continue

            w = float(score_weights[rel].get(cand, 1.0))

            vals.append(cs)
            weights.append(w)

            detail_rows.append({
                "template": rel,
                "candidate": cand,
                "cosine": cs,
                "fixed_score_weight": w,
                "weighted_contribution": w * cs,
            })

        if not vals:
            scores[rel] = float("-inf")
        else:
            vals = np.asarray(vals, np.float64)
            weights = np.asarray(weights, np.float64)
            denom = float(weights.sum())

            if denom <= 0:
                scores[rel] = float(vals.mean())
            else:
                scores[rel] = float(
                    np.dot(vals, weights) / denom
                )

    return scores, detail_rows


# =============================================================================
# Editing
# =============================================================================

def specs_for_template(
    template_rel,
    templates,
    current,
):
    specs = defaultdict(list)

    for cand in templates[template_rel]["candidate"].astype(str):
        if cand not in current:
            continue

        r = current[cand]
        specs[int(r["source_layer"])].append(
            (
                int(r["position"]),
                r["delta_h"],
                1.0,
            )
        )

    return dict(specs)


def specs_for_soft_mix(
    rel_weights,
    membership,
    current,
):
    """
    local_weight[c] = sum of template weights among templates containing c.
    Shared candidates therefore behave as a common scaffold.
    """
    specs = defaultdict(list)

    for cand, cur in current.items():
        local_w = sum(
            float(rel_weights[rel])
            for rel in REL
            if cand in membership[rel]
        )

        if local_w <= 1e-12:
            continue

        specs[int(cur["source_layer"])].append(
            (
                int(cur["position"]),
                cur["delta_h"],
                float(local_w),
            )
        )

    return dict(specs)


class DeltaEditor:
    def __init__(
        self,
        decoder_layers,
        specs,
        alpha,
        prompt_len,
    ):
        self.handles = []
        self.applied = defaultdict(int)
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)

        for L, entries in specs.items():
            if entries:
                self.handles.append(
                    decoder_layers[L].register_forward_hook(
                        self._hook(L, entries)
                    )
                )

    def _hook(self, L, entries):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)

            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()

            for p, delta_np, local_weight in entries:
                if 0 <= p < y.shape[1]:
                    d = torch.as_tensor(
                        delta_np,
                        device=y.device,
                        dtype=torch.float32,
                    )

                    y[:, p, :] = (
                        y[:, p, :]
                        + self.alpha
                        * float(local_weight)
                        * d
                    )

                    self.applied[L] += 1

            return traj.replace_first_tensor(
                out,
                y.to(h.dtype),
            )

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
    specs,
    alpha,
    max_new_tokens,
):
    editor = DeltaEditor(
        decoder_layers,
        specs,
        alpha,
        int(batch["input_ids"].shape[1]),
    )

    try:
        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )

        pred = traj.normalize_relation(
            base,
            text,
        )

        return text, pred, dict(editor.applied)

    finally:
        editor.close()


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(baseline_rows, condition_rows):
    base = {
        int(r["sid"]): r
        for r in baseline_rows
    }

    base_acc = safe_mean(
        r["baseline_correct"]
        for r in baseline_rows
    )

    groups = defaultdict(list)

    for r in condition_rows:
        groups[
            (
                r["method"],
                float(r["temperature"]),
                float(r["alpha"]),
            )
        ].append(r)

    rows = []

    for (method, temp, alpha), rr in groups.items():
        w2c = c2w = changed = 0
        corr = []

        for r in rr:
            sid = int(r["sid"])
            b = base[sid]
            c = bool(r["correct"])

            corr.append(c)

            w2c += int(
                (not bool(b["baseline_correct"]))
                and c
            )

            c2w += int(
                bool(b["baseline_correct"])
                and (not c)
            )

            changed += int(
                str(r["prediction"])
                != str(b["baseline_prediction"])
            )

        acc = safe_mean(corr)

        rows.append({
            "method": method,
            "temperature": temp,
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
                r["n_edited"]
                for r in rr
            ),
        })

    return sorted(
        rows,
        key=lambda r: (
            r["edited_acc"],
            r["net"],
        ),
        reverse=True,
    )


def per_relation_summary(
    baseline_rows,
    condition_rows,
):
    base = {
        int(r["sid"]): r
        for r in baseline_rows
    }

    rows = []

    keys = sorted({
        (
            r["method"],
            float(r["temperature"]),
            float(r["alpha"]),
        )
        for r in condition_rows
    })

    for method, temp, alpha in keys:
        cond = [
            r
            for r in condition_rows
            if (
                r["method"] == method
                and float(r["temperature"]) == temp
                and float(r["alpha"]) == alpha
            )
        ]

        for rel in REL:
            rr = [
                r
                for r in cond
                if r["gt"] == rel
            ]

            if not rr:
                continue

            sids = [
                int(r["sid"])
                for r in rr
            ]

            bacc = safe_mean(
                base[sid]["baseline_correct"]
                for sid in sids
            )

            eacc = safe_mean(
                r["correct"]
                for r in rr
            )

            w2c = sum(
                (not base[int(r["sid"])]["baseline_correct"])
                and bool(r["correct"])
                for r in rr
            )

            c2w = sum(
                base[int(r["sid"])]["baseline_correct"]
                and (not bool(r["correct"]))
                for r in rr
            )

            rows.append({
                "method": method,
                "temperature": temp,
                "alpha": alpha,
                "relation": rel,
                "N": len(rr),
                "baseline_acc": bacc,
                "edited_acc": eacc,
                "gain": eacc - bacc,
                "W2C": w2c,
                "C2W": c2w,
                "net": w2c - c2w,
            })

    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    methods = parse_strings(a.methods)
    temperatures = parse_floats(a.soft_temperatures)
    alphas = parse_floats(a.alphas)

    for method in methods:
        if method not in {
            "hard",
            "soft",
            "oracle",
            "random",
        }:
            raise ValueError(
                f"Unknown method={method}"
            )

    outdir = Path(a.output_dir)

    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(
            f"Non-empty output dir: {outdir}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    two = base.import_two_object_module()

    prompts = base.load_standard_prompts(
        Path(a.prompt_jsonl)
    )

    records, audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )

    rec_by_sid = {
        int(r.sid): r
        for r in records
    }

    meta = []

    for rec in records:
        sid = int(rec.sid)

        if sid not in prompts:
            continue

        p = prompts[sid]

        gt = traj.normalize_relation(
            base,
            p["answer_raw"],
        )

        if gt not in REL_INTERNAL:
            continue

        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(
        meta,
        a.max_samples,
        a.seed,
    )

    _train, test = traj.stratified_split(
        meta,
        a.train_ratio,
        a.seed,
    )

    # -------------------------------------------------------------------------
    # Model spec + processor
    # -------------------------------------------------------------------------
    specs = base.merged_model_specs(two)
    spec = specs[a.model]

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )

    # -------------------------------------------------------------------------
    # Fixed four-template discovery from selected_tokens.csv
    # -------------------------------------------------------------------------
    sel = pd.read_csv(a.selected_csv)

    sel["relation"] = sel["relation"].map(
        canon_rel
    )

    sel["k"] = pd.to_numeric(
        sel["k"],
        errors="raise",
    ).astype(int)

    sel["source_layer"] = pd.to_numeric(
        sel["source_layer"],
        errors="raise",
    ).astype(int)

    sel["position"] = pd.to_numeric(
        sel["position"],
        errors="raise",
    ).astype(int)

    sel["mediation"] = pd.to_numeric(
        sel["mediation"],
        errors="coerce",
    )

    sel = sel[
        (
            sel["condition"].astype(str)
            == str(a.discovery_condition)
        )
        & (
            sel["source_bundle"].astype(str)
            == str(a.discovery_bundle)
        )
        & (
            sel["k"]
            == int(a.discovery_k)
        )
        & sel["relation"].isin(REL)
    ].copy()

    if not len(sel):
        raise RuntimeError(
            "No selected-token discovery rows after filtering."
        )

    sel = (
        sel.sort_values(
            "mediation",
            ascending=False,
        )
        .drop_duplicates(
            [
                "sid",
                "relation",
                "source_layer",
                "position",
            ],
            keep="first",
        )
    )

    discovery_sids = set(
        sel["sid"].astype(int)
    )

    ann = canonicalize_discovery(
        sel,
        prompts,
        rec_by_sid,
        processor,
        a.visual_bins,
    )

    ann.to_csv(
        outdir / "discovery_canonical_rows.csv",
        index=False,
    )

    templates, rankings = build_templates(
        ann,
        a.template_k,
        a.candidate_min_rate,
    )

    rankings.to_csv(
        outdir / "relation_template_rankings.csv",
        index=False,
    )

    template_rows = []

    for rel in REL:
        for r in templates[rel].to_dict("records"):
            template_rows.append({
                "template": rel,
                **r,
            })

    pd.DataFrame(template_rows).to_csv(
        outdir / "relation_templates.csv",
        index=False,
    )

    overlap_rows = template_overlap_rows(
        templates
    )

    write_csv(
        outdir / "template_overlap.csv",
        overlap_rows,
    )

    candidate_meta = union_candidate_meta(
        templates
    )

    source_layers = sorted({
        int(x["source_layer"])
        for x in candidate_meta.values()
    })

    membership = template_membership(
        templates
    )

    score_weights = fixed_score_weight_lookup(
        templates,
        a.score_weight,
    )

    # Discovery metadata for prototype estimation.
    discovery_meta = [
        r
        for r in test
        if int(r["sid"]) in discovery_sids
    ]

    if not discovery_meta:
        raise RuntimeError(
            "No discovery selected-token IDs found inside held-out split."
        )

    # -------------------------------------------------------------------------
    # Evaluation set
    # -------------------------------------------------------------------------
    if a.eval_scope == "unseen":
        eval_pool = [
            r
            for r in test
            if int(r["sid"]) not in discovery_sids
        ]

    elif a.eval_scope == "selected":
        eval_pool = [
            r
            for r in test
            if int(r["sid"]) in discovery_sids
        ]

    else:
        eval_pool = list(test)

    eval_set = stratified_take(
        eval_pool,
        a.eval_max_samples,
        a.seed + 101,
    )

    if not eval_set:
        raise RuntimeError(
            "Evaluation set is empty."
        )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model_cls = getattr(
        transformers,
        spec.model_class,
    )

    load_kw = dict(
        dtype=base.resolve_dtype(
            spec.dtype_name
        ),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )

    if a.attn_impl != "none":
        load_kw["attn_implementation"] = (
            a.attn_impl
        )

    print("=" * 150)
    print("PROTOTYPE-MATCHED FOUR FIXED RELATION TEMPLATES")
    print("=" * 150)
    print(
        f"model={a.model} | repo={spec.repo_id}"
    )
    print(
        f"discovery bundle={a.discovery_bundle} "
        f"K={a.discovery_k}"
    )
    print(
        f"template K={a.template_k} per relation"
    )
    print(
        f"union candidates={len(candidate_meta)} "
        f"| source layers={source_layers}"
    )
    print(
        f"prototype discovery N={len(discovery_meta)}"
    )
    print(
        f"eval_scope={a.eval_scope} "
        f"| eval N={len(eval_set)}"
    )
    print(
        "discovery/eval overlap="
        + str(
            len(
                discovery_sids
                & set(
                    int(r["sid"])
                    for r in eval_set
                )
            )
        )
    )
    print(
        f"score_weight={a.score_weight}"
    )
    print(
        "exclude relation words from score="
        f"{a.exclude_relation_words_from_score}"
    )
    print(
        "exclude Answer/assistant from score="
        f"{a.exclude_answer_structural_from_score}"
    )
    print(
        f"methods={methods}"
    )
    print(
        f"soft temperatures={temperatures}"
    )
    print(
        f"alphas={alphas}"
    )
    print()

    print("Template overlap:")
    for r in overlap_rows:
        print(
            f"  {r['template_a']:>5s} vs {r['template_b']:<5s} "
            f"| shared={r['intersection']:2d} "
            f"| Jaccard={r['jaccard']:.3f}"
        )
    print()

    model = None

    try:
        model = model_cls.from_pretrained(
            spec.repo_id,
            **load_kw,
        )

        model.eval()

        base.configure_processor(
            model,
            processor,
        )

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = (
            base.resolve_decoder_layers(model)
        )

        n_layers = len(decoder_layers)

        for L in source_layers:
            if not (
                0 <= L < n_layers
            ):
                raise ValueError(
                    f"L{L} invalid for {a.model}; "
                    f"n_layers={n_layers}"
                )

        device = torch.device(
            a.device
        )

        # ---------------------------------------------------------------------
        # Prototype estimation from discovery samples.
        # ---------------------------------------------------------------------
        prototypes, prototype_rows = estimate_prototypes(
            discovery_meta=discovery_meta,
            candidate_meta=candidate_meta,
            prompts=prompts,
            rec_by_sid=rec_by_sid,
            processor=processor,
            model=model,
            decoder_layers=decoder_layers,
            device=device,
            visual_bins=a.visual_bins,
            gray_value=a.gray_value,
        )

        write_csv(
            outdir / "prototype_geometry.csv",
            prototype_rows,
        )

        # ---------------------------------------------------------------------
        # Evaluation.
        # ---------------------------------------------------------------------
        baseline_rows = []
        score_rows = []
        condition_rows = []
        score_detail_rows = []

        rng = random.Random(
            a.seed + 98731
        )

        for m in tqdm(
            eval_set,
            desc="Prototype-template eval",
        ):
            sid = int(m["sid"])
            gt = m["gt"]
            gt_disp = DISPLAY[gt]

            real = gray = rb = gb = None

            try:
                real = base.record_image(
                    rec_by_sid[sid]
                )

                if hasattr(
                    real,
                    "convert",
                ):
                    real = real.convert(
                        "RGB"
                    )

                gray = make_gray_image(
                    real,
                    a.gray_value,
                )

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

                ids = (
                    rb["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                toks = [
                    str(x)
                    for x in processor.tokenizer.convert_ids_to_tokens(ids)
                ]

                spos, rpos = object_positions(
                    processor.tokenizer,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                vpos = visual_positions(
                    toks
                )

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

                current = resolve_current_deltas(
                    candidate_meta,
                    mapping,
                    hreal,
                    hgray,
                )

                scores, details = compute_template_scores(
                    current=current,
                    templates=templates,
                    prototypes=prototypes,
                    score_weights=score_weights,
                    exclude_relation_words=a.exclude_relation_words_from_score,
                    exclude_answer_structural=a.exclude_answer_structural_from_score,
                )

                hard_choice = max(
                    REL,
                    key=lambda r: scores[r],
                )

                score_rows.append({
                    "sid": sid,
                    "gt": gt_disp,
                    **{
                        f"score_{rel}":
                        scores[rel]
                        for rel in REL
                    },
                    "hard_choice": hard_choice,
                    "hard_choice_correct":
                        hard_choice == gt_disp,
                    "score_margin": (
                        sorted(
                            scores.values(),
                            reverse=True,
                        )[0]
                        - sorted(
                            scores.values(),
                            reverse=True,
                        )[1]
                    ),
                })

                for d in details:
                    score_detail_rows.append({
                        "sid": sid,
                        "gt": gt_disp,
                        **d,
                    })

                baseline_text = (
                    base.generate_text(
                        model,
                        processor,
                        rb,
                        max_new_tokens=
                            a.max_new_tokens,
                    )
                )

                baseline_pred_raw = (
                    traj.normalize_relation(
                        base,
                        baseline_text,
                    )
                )

                baseline_pred = DISPLAY.get(
                    baseline_pred_raw,
                    baseline_pred_raw,
                )

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt_disp,
                    "baseline_prediction":
                        baseline_pred,
                    "baseline_correct":
                        baseline_pred_raw == gt,
                    "baseline_text":
                        baseline_text,
                    "hard_template_choice":
                        hard_choice,
                    "hard_template_choice_correct":
                        hard_choice == gt_disp,
                })

                # Fixed random template per sample, shared across alpha.
                random_choice = rng.choice(
                    list(REL)
                )

                # HARD
                if "hard" in methods:
                    hard_specs = specs_for_template(
                        hard_choice,
                        templates,
                        current,
                    )

                    for alpha in alphas:
                        text, pred_raw, applied = (
                            generate_with_specs(
                                model,
                                processor,
                                decoder_layers,
                                rb,
                                hard_specs,
                                alpha,
                                a.max_new_tokens,
                            )
                        )

                        pred = DISPLAY.get(
                            pred_raw,
                            pred_raw,
                        )

                        condition_rows.append({
                            "sid": sid,
                            "gt": gt_disp,
                            "method":
                                "prototype_hard",
                            "temperature": 0.0,
                            "alpha": alpha,
                            "template_choice":
                                hard_choice,
                            "template_choice_correct":
                                hard_choice == gt_disp,
                            "prediction": pred,
                            "correct":
                                pred_raw == gt,
                            "text": text,
                            "n_edited":
                                sum(applied.values()),
                        })

                # SOFT
                if "soft" in methods:
                    score_vec = [
                        scores[rel]
                        for rel in REL
                    ]

                    for temp in temperatures:
                        w = softmax_np(
                            score_vec,
                            temp,
                        )

                        rel_weights = {
                            rel: float(w[i])
                            for i, rel in enumerate(REL)
                        }

                        soft_specs = specs_for_soft_mix(
                            rel_weights,
                            membership,
                            current,
                        )

                        for alpha in alphas:
                            text, pred_raw, applied = (
                                generate_with_specs(
                                    model,
                                    processor,
                                    decoder_layers,
                                    rb,
                                    soft_specs,
                                    alpha,
                                    a.max_new_tokens,
                                )
                            )

                            pred = DISPLAY.get(
                                pred_raw,
                                pred_raw,
                            )

                            condition_rows.append({
                                "sid": sid,
                                "gt": gt_disp,
                                "method":
                                    "prototype_soft",
                                "temperature":
                                    temp,
                                "alpha": alpha,
                                "template_choice":
                                    hard_choice,
                                "template_choice_correct":
                                    hard_choice == gt_disp,
                                "prediction": pred,
                                "correct":
                                    pred_raw == gt,
                                "text": text,
                                "n_edited":
                                    sum(applied.values()),
                                **{
                                    f"weight_{rel}":
                                        rel_weights[rel]
                                    for rel in REL
                                },
                            })

                # ORACLE ceiling
                if "oracle" in methods:
                    oracle_specs = specs_for_template(
                        gt_disp,
                        templates,
                        current,
                    )

                    for alpha in alphas:
                        text, pred_raw, applied = (
                            generate_with_specs(
                                model,
                                processor,
                                decoder_layers,
                                rb,
                                oracle_specs,
                                alpha,
                                a.max_new_tokens,
                            )
                        )

                        pred = DISPLAY.get(
                            pred_raw,
                            pred_raw,
                        )

                        condition_rows.append({
                            "sid": sid,
                            "gt": gt_disp,
                            "method":
                                "oracle_template",
                            "temperature": 0.0,
                            "alpha": alpha,
                            "template_choice":
                                gt_disp,
                            "template_choice_correct":
                                True,
                            "prediction": pred,
                            "correct":
                                pred_raw == gt,
                            "text": text,
                            "n_edited":
                                sum(applied.values()),
                        })

                # RANDOM control
                if "random" in methods:
                    random_specs = specs_for_template(
                        random_choice,
                        templates,
                        current,
                    )

                    for alpha in alphas:
                        text, pred_raw, applied = (
                            generate_with_specs(
                                model,
                                processor,
                                decoder_layers,
                                rb,
                                random_specs,
                                alpha,
                                a.max_new_tokens,
                            )
                        )

                        pred = DISPLAY.get(
                            pred_raw,
                            pred_raw,
                        )

                        condition_rows.append({
                            "sid": sid,
                            "gt": gt_disp,
                            "method":
                                "random_template",
                            "temperature": 0.0,
                            "alpha": alpha,
                            "template_choice":
                                random_choice,
                            "template_choice_correct":
                                random_choice == gt_disp,
                            "prediction": pred,
                            "correct":
                                pred_raw == gt,
                            "text": text,
                            "n_edited":
                                sum(applied.values()),
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

        # ---------------------------------------------------------------------
        # Save / summarize
        # ---------------------------------------------------------------------
        write_csv(
            outdir / "baseline.csv",
            baseline_rows,
        )

        write_csv(
            outdir / "template_scores.csv",
            score_rows,
        )

        write_csv(
            outdir / "template_score_details.csv",
            score_detail_rows,
        )

        write_csv(
            outdir / "generation_conditions.csv",
            condition_rows,
        )

        summary = summarize_generation(
            baseline_rows,
            condition_rows,
        )

        write_csv(
            outdir / "summary.csv",
            summary,
        )

        per_rel = per_relation_summary(
            baseline_rows,
            condition_rows,
        )

        write_csv(
            outdir / "per_relation_summary.csv",
            per_rel,
        )

        score_df = pd.DataFrame(
            score_rows
        )

        template_acc = float(
            score_df[
                "hard_choice_correct"
            ].mean()
        )

        # Choice confusion matrix.
        confusion_rows = []

        for gt_rel in REL:
            g = score_df[
                score_df["gt"] == gt_rel
            ]

            for pred_rel in REL:
                n = int(
                    (
                        g["hard_choice"]
                        == pred_rel
                    ).sum()
                )

                confusion_rows.append({
                    "gt": gt_rel,
                    "chosen_template":
                        pred_rel,
                    "N_gt": len(g),
                    "count": n,
                    "rate":
                        n / len(g)
                        if len(g)
                        else float("nan"),
                })

        write_csv(
            outdir / "template_choice_confusion.csv",
            confusion_rows,
        )

        print("\n" + "=" * 150)
        print("RESULTS")
        print("=" * 150)

        baseline_acc = safe_mean(
            r["baseline_correct"]
            for r in baseline_rows
        )

        print(
            f"Baseline: N={len(baseline_rows)} "
            f"acc={baseline_acc:.4f}"
        )

        print(
            f"Prototype hard-template matching accuracy: "
            f"{template_acc:.4f}"
        )

        print()

        print(
            f"{'method':<22s} "
            f"{'temp':>7s} "
            f"{'alpha':>7s} "
            f"{'acc':>8s} "
            f"{'gain':>8s} "
            f"{'W2C':>5s} "
            f"{'C2W':>5s} "
            f"{'net':>5s} "
            f"{'nEdit':>7s}"
        )

        print("-" * 150)

        for r in summary:
            print(
                f"{r['method']:<22s} "
                f"{float(r['temperature']):>7.3f} "
                f"{float(r['alpha']):>7.3f} "
                f"{float(r['edited_acc']):>8.4f} "
                f"{float(r['gain']):>+8.4f} "
                f"{int(r['W2C']):>5d} "
                f"{int(r['C2W']):>5d} "
                f"{int(r['net']):>5d} "
                f"{float(r['mean_n_edited']):>7.2f}"
            )

        print("\nTemplate-choice confusion:")
        for gt_rel in REL:
            row = {
                r["chosen_template"]: r["rate"]
                for r in confusion_rows
                if r["gt"] == gt_rel
            }

            print(
                f"  GT={gt_rel:<5s} | "
                + " ".join(
                    f"{pred}={row.get(pred, float('nan')):.3f}"
                    for pred in REL
                )
            )

        print("\nBest per relation:")
        pr = pd.DataFrame(
            per_rel
        )

        if len(pr):
            for rel in REL:
                q = (
                    pr[
                        pr["relation"]
                        == rel
                    ]
                    .sort_values(
                        [
                            "edited_acc",
                            "net",
                        ],
                        ascending=[
                            False,
                            False,
                        ],
                    )
                )

                if len(q):
                    r = q.iloc[0]

                    print(
                        f"  {rel:<5s} | "
                        f"{r['method']} "
                        f"T={float(r['temperature']):.3f} "
                        f"a={float(r['alpha']):.3f} | "
                        f"{float(r['baseline_acc']):.4f}"
                        f"->{float(r['edited_acc']):.4f} "
                        f"({float(r['gain']):+.4f}) | "
                        f"W2C/C2W="
                        f"{int(r['W2C'])}/"
                        f"{int(r['C2W'])}"
                    )

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "selected_csv": a.selected_csv,
            "discovery_bundle":
                a.discovery_bundle,
            "discovery_k":
                a.discovery_k,
            "template_k":
                a.template_k,
            "candidate_min_rate":
                a.candidate_min_rate,
            "visual_bins":
                a.visual_bins,
            "score_weight":
                a.score_weight,
            "exclude_relation_words_from_score":
                a.exclude_relation_words_from_score,
            "exclude_answer_structural_from_score":
                a.exclude_answer_structural_from_score,
            "prototype_discovery_N":
                len(discovery_meta),
            "eval_scope":
                a.eval_scope,
            "eval_N":
                len(eval_set),
            "discovery_eval_overlap":
                len(
                    discovery_sids
                    & set(
                        int(r["sid"])
                        for r in eval_set
                    )
                ),
            "hard_template_match_acc":
                template_acc,
            "methods":
                methods,
            "soft_temperatures":
                temperatures,
            "alphas":
                alphas,
            "hard_soft_gt_used_for_template_choice":
                False,
            "hard_soft_late_writer_used_at_eval":
                False,
            "hard_soft_gradient_used_at_eval":
                False,
            "hard_soft_answer_logits_used_for_choice":
                False,
            "hard_soft_probe_classifier_used":
                False,
            "hard_soft_per_token_topk_used":
                False,
            "oracle_control_uses_gt":
                "oracle" in methods,
            "random_control":
                "random" in methods,
            "dataset_audit":
                audit,
        }

        (
            outdir
            / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print(
            "\nSaved:",
            outdir,
        )

    finally:
        if model is not None:
            del model

        del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
