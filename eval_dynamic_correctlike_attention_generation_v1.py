#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Correct-Like Attention Direction Repair
================================================

Purpose
-------
Test the hypothesis suggested by the Direction decomposition:

    generation-correct samples:
        Attention strongly increases GT spatial evidence
        and only modestly increases competing evidence.

    generation-wrong samples:
        Attention increases GT less
        and/or increases the final competing relation more.

Instead of directly steering the final hidden state, this script modifies the
ATTENTION UPDATE itself so that its Direction-coordinate update becomes more
"correct-like".

The primary evaluation is fresh full model.generate().

Core idea
---------
For decoder layer l, use a common POST_ATTN Direction codebook.  Define the
image-grounded Attention update:

    Delta r_attn,l
      = (r_post,l - r_pre,l)

where r is the Image-NoImage subject-reference residual.

Because the same codebook is used for both sides of the subtraction, define:

    Delta s_GT   = Delta r_attn dot mu_GT
    Delta s_foil = Delta r_attn dot mu_foil

TRAIN generation-correct controls provide, for every layer/GT/foil:

    target_GT   = median correct Delta s_GT
    target_foil = median correct Delta s_foil

At runtime the hook observes the CURRENT Attention output.  Therefore when
multiple earlier layers have already been edited, later-layer corrections are
recomputed from the modified trajectory.  This is a genuinely dynamic
multi-layer intervention.

Modes
-----
1) correct_like_match
   Match both current Attention update coordinates to correct-control targets:

       Delta s_GT   -> target_GT
       Delta s_foil -> target_foil

   The minimum-L2 hidden-space edit satisfying the two constraints is used.

2) correct_like_safe
   Only repair a deficit/excess:
       - boost GT only if current_GT < target_GT
       - reduce foil only if current_foil > target_foil

   If a sample is already better than the target along one coordinate, that
   coordinate is held fixed.

3) gt_only
   Boost GT toward target while holding foil fixed.

4) foil_only
   Reduce foil toward target while holding GT fixed.

5) random_correct_like
   Compute the same correction norm as correct_like_safe on the CURRENT
   trajectory, but replace it with a random direction orthogonal to the
   TRAIN-derived 2-D spatial subspace.

This is still an ORACLE diagnostic experiment because GT is known.

Layer variants
--------------
For selected layers, the script automatically tests:

    single_L14
    single_L16
    single_L17
    single_L19

and cumulative dynamic variants:

    cumulative_L14_L16
    cumulative_L14_L16_L17
    cumulative_L14_L16_L17_L19

Optionally all layer pairs can also be tested.

Why POST_ATTN / Attention-output intervention?
----------------------------------------------
The Direction failure was observed in the Attention transition itself.
Adding delta to self_attn output changes the post-attention residual by exactly
the same delta, before MLP and before downstream layers.  For a pair-preserving
edit:

    subject tokens  += delta/2
    reference tokens -= delta/2

the pooled subject-reference pair difference changes by exactly delta.

Dependencies
------------
Only project-native modules are required:

    extract_two_object_relation_states.py
    analyze_layerwise_direction_failure_scan_v1.py

No previously generated .py helper is required.

Required existing experiment output
-----------------------------------
--prepost-dir must contain:

    train_stage_vectors.npz

from the previous pre/post-Attention Direction experiment.  This file is used
to build POST_ATTN Direction codebooks and TRAIN generation-correct Attention
update targets.

Recommended smoke test
----------------------
CUDA_VISIBLE_DEVICES=0 python eval_dynamic_correctlike_attention_generation_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 14,16,17,19 \
  --modes correct_like_safe,gt_only,random_correct_like \
  --max-samples 20 \
  --variant-strategy singles,cumulative \
  --output-dir output/qwen7b_dynamic_correctlike_attention_smoke \
  --overwrite

Recommended full generation run
-------------------------------
CUDA_VISIBLE_DEVICES=0 python eval_dynamic_correctlike_attention_generation_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 14,16,17,19 \
  --modes correct_like_match,correct_like_safe,gt_only,random_correct_like \
  --variant-strategy singles,cumulative \
  --output-dir output/qwen7b_dynamic_correctlike_attention_generation_v1 \
  --overwrite

Main outputs
------------
correct_attention_targets.csv
    TRAIN generation-correct target Delta s_GT / Delta s_foil.

generation_per_sample.csv
    Fresh baseline generation and every intervention variant.

generation_summary.csv
    Main result:
        generation accuracy
        gain vs fresh baseline
        W->C
        C->W
        net

dynamic_layer_updates.csv
    What each dynamic hook actually saw and changed at each layer:
        current Delta s_GT / foil
        target
        requested correction
        edit norm

dynamic_layer_update_summary.csv
    Correct vs wrong aggregate layer behavior under each variant/mode.

Important interpretation
------------------------
A positive result for a multi-layer correct-like intervention means that
Direction vectors identify a spatial Attention-update pattern that is not only
correlated with errors but can be used to alter actual generation.

A negative result does NOT prove that Attention Direction composition is
irrelevant; it can mean that two Direction coordinates are only a partial
description of the computation.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    p.add_argument("--prepost-dir", required=True)
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
        default="14,16,17,19",
        help="Decoder layers whose Attention updates will be tested.",
    )
    p.add_argument(
        "--modes",
        default=(
            "correct_like_match,correct_like_safe,"
            "gt_only,random_correct_like"
        ),
    )
    p.add_argument(
        "--variant-strategy",
        default="singles,cumulative",
        help="Comma-separated: singles,cumulative,pairs,all.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional stratified cap for smoke tests.",
    )
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--target-stat",
        default="median",
        choices=["median", "mean"],
    )
    p.add_argument(
        "--max-edit-norm",
        type=float,
        default=0.0,
        help="Optional per-layer delta norm cap; <=0 means no cap.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
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
    out = []
    for piece in parse_words(text):
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(piece))

    out = sorted(set(out))
    if not out:
        raise ValueError("No layers selected.")

    bad = [x for x in out if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(
            f"Invalid layers {bad}; valid range is 0..{n_layers - 1}"
        )
    return out


def target_stat(vals: Sequence[float], which: str) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        return float("nan")
    return float(np.median(x) if which == "median" else np.mean(x))


# =============================================================================
# Cached metadata
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
        labels = [
            norm_relation(x)
            for x in z["relation"]
        ]

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
        group = str(
            r.get("generation_group", "")
        ).strip().lower()

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
# Model / decoder / prompt helpers
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
            if len(layers) > 0:
                b = layers[0]
                if hasattr(b, "self_attn"):
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


def build_batch(
    processor,
    rec,
    question,
    image,
    device,
):
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
# Relation-token diagnostics
# =============================================================================

def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    try:
        return [
            int(x)
            for x in tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        ]
    except Exception:
        obj = tokenizer(text, add_special_tokens=False)
        ids = (
            obj["input_ids"]
            if isinstance(obj, dict)
            else getattr(obj, "input_ids", [])
        )
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in ids]


def relation_token_map(tokenizer):
    out = {}
    unk = getattr(tokenizer, "unk_token_id", None)

    for rel in RELATIONS:
        ids = []
        for text in (
            rel,
            " " + rel,
            "\n" + rel,
            rel.capitalize(),
            " " + rel.capitalize(),
        ):
            xx = tokenizer_ids(tokenizer, text)
            if len(xx) != 1:
                continue
            tid = int(xx[0])
            if unk is not None and tid == int(unk):
                continue
            ids.append(tid)

        ids = list(dict.fromkeys(ids))
        if not ids:
            raise RuntimeError(
                f"No one-token variant for {rel}"
            )
        out[rel] = ids

    return out


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "logits",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "logits",
            None,
        ),
    ]
    for x in candidates:
        if torch.is_tensor(x):
            return x

    if isinstance(outputs, (tuple, list)):
        for x in outputs:
            if torch.is_tensor(x) and x.ndim == 3:
                return x

    raise RuntimeError("Could not locate logits.")


def firststep_scores(model, batch, token_map):
    with torch.inference_mode():
        out = model(
            **batch,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        logits = extract_logits(out)[0, -1]

    scores = {}
    for rel in RELATIONS:
        ids = torch.as_tensor(
            token_map[rel],
            device=logits.device,
            dtype=torch.long,
        )
        scores[rel] = float(
            logits.index_select(0, ids).max().item()
        )
    return scores


# =============================================================================
# Generation parsing
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


def generate_answer(
    model,
    processor,
    batch,
    max_new_tokens,
):
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
# TRAIN stage vectors / Direction codebooks / correct Attention targets
# =============================================================================

def unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if n <= EPS:
        return np.zeros_like(x, dtype=np.float32)
    return (x / n).astype(np.float32)


def orthonormal_basis(vectors: Sequence[np.ndarray]) -> np.ndarray:
    A = np.stack(
        [np.asarray(v, dtype=np.float64) for v in vectors],
        axis=1,
    )
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > 1e-8 * max(float(s.max()), 1.0)
    if not np.any(keep):
        raise RuntimeError("Degenerate spatial basis.")
    return u[:, keep].astype(np.float32)


def fit_post_codebook(
    X_post: np.ndarray,
    labels: np.ndarray,
):
    center = X_post.mean(axis=0).astype(np.float32)
    Xc = X_post - center

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
    }


def load_train_targets(
    *,
    prepost_dir: Path,
    metadata,
    selected_layers,
    target_stat_name,
):
    path = prepost_dir / "train_stage_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        train_sids = np.asarray(z["sid"], dtype=np.int64)
        labels = np.asarray(
            [norm_relation(x) for x in z["relation"]],
            dtype=object,
        )

        codebooks = {}
        targets = {}
        target_rows = []

        sid_to_i = {
            int(sid): i
            for i, sid in enumerate(train_sids.tolist())
        }

        correct_train_mask = np.asarray(
            [
                metadata["generation"].get(
                    int(sid),
                    {},
                ).get("generation_group", "")
                == "correct"
                for sid in train_sids.tolist()
            ],
            dtype=bool,
        )

        if not np.any(correct_train_mask):
            raise RuntimeError(
                "No generation-correct TRAIN controls found in "
                "sample_split_and_generation.csv."
            )

        for li in selected_layers:
            pre_key = f"L{li}_pre_attn"
            post_key = f"L{li}_post_attn"

            if pre_key not in z.files or post_key not in z.files:
                raise KeyError(
                    f"{path} does not contain {pre_key}/{post_key}. "
                    f"Rerun the pre/post experiment including L{li}."
                )

            Xpre = np.asarray(z[pre_key], dtype=np.float32)
            Xpost = np.asarray(z[post_key], dtype=np.float32)

            cb = fit_post_codebook(
                Xpost,
                labels,
            )
            codebooks[li] = cb

            # Common POST_ATTN coordinate system.
            delta_attn = Xpost - Xpre

            for gt in RELATIONS:
                gt_mask = (
                    correct_train_mask
                    & (labels == gt)
                )

                if not np.any(gt_mask):
                    raise RuntimeError(
                        f"No generation-correct TRAIN controls for GT={gt}"
                    )

                p_gt = cb["protos"][gt]
                gt_vals = delta_attn[gt_mask] @ p_gt

                for foil in RELATIONS:
                    if foil == gt:
                        continue

                    p_foil = cb["protos"][foil]
                    foil_vals = (
                        delta_attn[gt_mask] @ p_foil
                    )

                    t_gt = target_stat(
                        gt_vals,
                        target_stat_name,
                    )
                    t_foil = target_stat(
                        foil_vals,
                        target_stat_name,
                    )

                    key = (li, gt, foil)
                    targets[key] = {
                        "target_gt": t_gt,
                        "target_foil": t_foil,
                        "n_controls": int(np.sum(gt_mask)),
                        "gt_std": float(np.std(gt_vals)),
                        "foil_std": float(np.std(foil_vals)),
                    }

                    target_rows.append({
                        "layer": li,
                        "gt": gt,
                        "foil": foil,
                        "target_stat": target_stat_name,
                        "n_generation_correct_train":
                            int(np.sum(gt_mask)),
                        "target_delta_s_GT": t_gt,
                        "target_delta_s_foil": t_foil,
                        "std_delta_s_GT":
                            float(np.std(gt_vals)),
                        "std_delta_s_foil":
                            float(np.std(foil_vals)),
                    })

    return codebooks, targets, target_rows


# =============================================================================
# NO-IMAGE pair-difference capture
# =============================================================================

class PrePostPairCapture:
    def __init__(
        self,
        decoder_layers,
        selected_layers,
        subj_pos,
        ref_pos,
    ):
        self.layers = list(map(int, selected_layers))
        self.subj_pos = list(map(int, subj_pos))
        self.ref_pos = list(map(int, ref_pos))
        self.pre = {}
        self.post = {}
        self.handles = []

        for li in self.layers:
            block = decoder_layers[li]

            def make_block_pre(layer_id):
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
                    self.pre[layer_id] = (
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

            def make_postnorm_pre(layer_id):
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
                    self.post[layer_id] = (
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

            self.handles.append(
                block.register_forward_pre_hook(
                    make_block_pre(li)
                )
            )

            if hasattr(block, "post_attention_layernorm"):
                self.handles.append(
                    block.post_attention_layernorm.register_forward_pre_hook(
                        make_postnorm_pre(li)
                    )
                )
            else:
                raise RuntimeError(
                    f"Layer {li} has no post_attention_layernorm."
                )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def capture_noimage_pairs(
    *,
    model,
    processor,
    decoder_layers,
    selected_layers,
    rec,
    device,
    prompt_template,
):
    question = prompt_template.format(
        subject=rec.subject,
        reference=rec.reference,
    )

    batch, sp, rp = build_batch(
        processor,
        rec,
        question,
        None,
        device,
    )

    with PrePostPairCapture(
        decoder_layers,
        selected_layers,
        sp,
        rp,
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
        li
        for li in selected_layers
        if li not in cap.pre or li not in cap.post
    ]
    if missing:
        raise RuntimeError(
            f"Missing no-image pre/post captures for layers {missing}"
        )

    out = {
        li: {
            "pre": cap.pre[li],
            "post": cap.post[li],
            "attn_delta":
                cap.post[li] - cap.pre[li],
        }
        for li in selected_layers
    }

    del batch
    return out


# =============================================================================
# Dynamic correct-like Attention hook
# =============================================================================

def min_l2_two_constraint(
    v1: np.ndarray,
    v2: np.ndarray,
    c1: float,
    c2: float,
) -> np.ndarray:
    A = np.stack(
        [
            np.asarray(v1, dtype=np.float64),
            np.asarray(v2, dtype=np.float64),
        ],
        axis=1,
    )
    c = np.asarray([c1, c2], dtype=np.float64)
    gram = A.T @ A
    delta = A @ (
        np.linalg.pinv(gram, rcond=1e-10) @ c
    )
    return delta.astype(np.float32)


def clip_delta(
    delta: np.ndarray,
    max_norm: float,
) -> np.ndarray:
    if max_norm <= 0:
        return np.asarray(delta, dtype=np.float32)

    d = np.asarray(delta, dtype=np.float32)
    n = float(np.linalg.norm(d))
    if n <= max_norm or n <= EPS:
        return d

    return (
        d * (float(max_norm) / n)
    ).astype(np.float32)


def random_orthogonal_delta(
    norm: float,
    basis: np.ndarray,
    dim: int,
    seed: int,
) -> np.ndarray:
    if norm <= EPS:
        return np.zeros(dim, dtype=np.float32)

    rng = np.random.default_rng(seed)
    B = np.asarray(basis, dtype=np.float64)

    for _ in range(100):
        v = rng.standard_normal(dim)
        v = v - B @ (B.T @ v)
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            return (
                v / n * float(norm)
            ).astype(np.float32)

    raise RuntimeError("Could not sample orthogonal random direction.")


class DynamicAttentionRepair:
    """
    Register dynamic hooks on selected Attention modules.

    Every layer recomputes its correction from the CURRENT trajectory, so
    upstream edits automatically affect downstream repair decisions.
    """

    def __init__(
        self,
        *,
        decoder_layers,
        active_layers,
        subj_pos,
        ref_pos,
        noimg_pairs,
        codebooks,
        targets,
        gt,
        foil,
        mode,
        max_edit_norm,
        sid,
        seed,
    ):
        self.decoder_layers = decoder_layers
        self.active_layers = list(map(int, active_layers))
        self.subj_pos = list(map(int, subj_pos))
        self.ref_pos = list(map(int, ref_pos))
        self.noimg_pairs = noimg_pairs
        self.codebooks = codebooks
        self.targets = targets
        self.gt = gt
        self.foil = foil
        self.mode = mode
        self.max_edit_norm = float(max_edit_norm)
        self.sid = int(sid)
        self.seed = int(seed)

        self.applied = set()
        self.logs = []
        self.handles = []

        for li in self.active_layers:
            self.handles.append(
                decoder_layers[li].self_attn.register_forward_hook(
                    self._make_attn_hook(li)
                )
            )

    def _make_attn_hook(self, li):
        def hook(_module, _args, output):
            # Apply once on prompt processing, never on cached 1-token decoding.
            if li in self.applied:
                return output

            x = first_tensor(output)
            if x.ndim != 3:
                return output

            seq = x[0]
            max_pos = max(self.subj_pos + self.ref_pos)

            if int(seq.shape[0]) <= max_pos:
                return output

            attn_pair_real = (
                pair_diff(
                    seq,
                    self.subj_pos,
                    self.ref_pos,
                )
                .detach().float().cpu().numpy()
                .astype(np.float32)
            )

            # Image-grounded Attention update:
            # (post-pre)_img - (post-pre)_noimg.
            current_delta_r = (
                attn_pair_real
                - self.noimg_pairs[li]["attn_delta"]
            )

            cb = self.codebooks[li]
            p_gt = np.asarray(
                cb["protos"][self.gt],
                dtype=np.float32,
            )
            p_foil = np.asarray(
                cb["protos"][self.foil],
                dtype=np.float32,
            )

            current_gt = float(
                current_delta_r @ p_gt
            )
            current_foil = float(
                current_delta_r @ p_foil
            )

            target = self.targets[
                (li, self.gt, self.foil)
            ]
            target_gt = float(
                target["target_gt"]
            )
            target_foil = float(
                target["target_foil"]
            )

            if self.mode == "correct_like_match":
                req_gt = target_gt - current_gt
                req_foil = target_foil - current_foil

            elif self.mode == "correct_like_safe":
                req_gt = max(
                    0.0,
                    target_gt - current_gt,
                )
                req_foil = min(
                    0.0,
                    target_foil - current_foil,
                )

            elif self.mode == "gt_only":
                req_gt = max(
                    0.0,
                    target_gt - current_gt,
                )
                req_foil = 0.0

            elif self.mode == "foil_only":
                req_gt = 0.0
                req_foil = min(
                    0.0,
                    target_foil - current_foil,
                )

            elif self.mode == "random_correct_like":
                # First compute the SAFE targeted correction norm on the
                # current random trajectory, then redirect that norm outside
                # the 2-D spatial subspace.
                safe_gt = max(
                    0.0,
                    target_gt - current_gt,
                )
                safe_foil = min(
                    0.0,
                    target_foil - current_foil,
                )
                target_delta = min_l2_two_constraint(
                    p_gt,
                    p_foil,
                    safe_gt,
                    safe_foil,
                )
                target_delta = clip_delta(
                    target_delta,
                    self.max_edit_norm,
                )

                delta = random_orthogonal_delta(
                    float(np.linalg.norm(target_delta)),
                    cb["basis"],
                    len(target_delta),
                    seed=(
                        self.seed
                        + self.sid * 100003
                        + li * 1009
                    ),
                )
                req_gt = safe_gt
                req_foil = safe_foil

            else:
                raise ValueError(
                    f"Unknown repair mode: {self.mode}"
                )

            if self.mode != "random_correct_like":
                delta = min_l2_two_constraint(
                    p_gt,
                    p_foil,
                    req_gt,
                    req_foil,
                )
                delta = clip_delta(
                    delta,
                    self.max_edit_norm,
                )

            delta_norm = float(
                np.linalg.norm(delta)
            )

            y = x.clone()

            if delta_norm > EPS:
                half = 0.5 * torch.from_numpy(
                    delta
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

            achieved_gt = float(
                delta @ p_gt
            )
            achieved_foil = float(
                delta @ p_foil
            )

            self.logs.append({
                "sid": self.sid,
                "layer": li,
                "mode": self.mode,
                "gt": self.gt,
                "foil": self.foil,

                "current_delta_s_GT":
                    current_gt,
                "current_delta_s_foil":
                    current_foil,

                "target_delta_s_GT":
                    target_gt,
                "target_delta_s_foil":
                    target_foil,

                "requested_GT_correction":
                    req_gt,
                "requested_foil_correction":
                    req_foil,

                "achieved_GT_correction":
                    achieved_gt,
                "achieved_foil_correction":
                    achieved_foil,

                "post_edit_delta_s_GT":
                    current_gt + achieved_gt,
                "post_edit_delta_s_foil":
                    current_foil + achieved_foil,

                "edit_norm": delta_norm,
                "triggered": int(delta_norm > EPS),
            })

            self.applied.add(li)
            return replace_first_tensor(
                output,
                first_tensor(y),
            )

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


# =============================================================================
# Layer variants
# =============================================================================

def build_variants(
    selected_layers: Sequence[int],
    strategy_text: str,
):
    layers = list(sorted(set(map(int, selected_layers))))
    strategies = set(parse_words(strategy_text))

    allowed = {"singles", "cumulative", "pairs", "all"}
    bad = strategies - allowed
    if bad:
        raise ValueError(
            f"Unknown variant strategies: {sorted(bad)}"
        )

    variants = []
    seen = set()

    def add(name, vals):
        key = tuple(vals)
        if not key or key in seen:
            return
        seen.add(key)
        variants.append({
            "variant": name,
            "layers": list(key),
        })

    if "singles" in strategies:
        for li in layers:
            add(
                f"single_L{li}",
                [li],
            )

    if "cumulative" in strategies:
        for k in range(2, len(layers) + 1):
            vals = layers[:k]
            add(
                "cumulative_"
                + "_".join(f"L{x}" for x in vals),
                vals,
            )

    if "pairs" in strategies:
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                vals = [layers[i], layers[j]]
                add(
                    f"pair_L{layers[i]}_L{layers[j]}",
                    vals,
                )

    if "all" in strategies:
        add(
            "all_selected",
            layers,
        )

    return variants


# =============================================================================
# Eval sample selection
# =============================================================================

def select_eval_sids(
    metadata,
    records,
    split,
    max_samples,
    seed,
):
    sids = []

    for sid in metadata["sids"]:
        if (
            split != "all"
            and metadata["split"].get(sid, "")
            != split
        ):
            continue

        if sid not in records:
            continue

        gt = metadata["gt"].get(sid, "")
        if gt not in REL2ID:
            continue

        sids.append(sid)

    if max_samples is None or len(sids) <= max_samples:
        return sorted(sids)

    rng = random.Random(seed)

    correct = [
        sid for sid in sids
        if metadata["generation"].get(
            sid, {}
        ).get("generation_group", "")
        == "correct"
    ]
    wrong = [
        sid for sid in sids
        if metadata["generation"].get(
            sid, {}
        ).get("generation_group", "")
        == "wrong"
    ]
    other = [
        sid for sid in sids
        if sid not in set(correct + wrong)
    ]

    rng.shuffle(correct)
    rng.shuffle(wrong)
    rng.shuffle(other)

    nw = min(len(wrong), max_samples // 2)
    nc = min(len(correct), max_samples - nw)

    chosen = wrong[:nw] + correct[:nc]

    if len(chosen) < max_samples:
        remain = [
            sid for sid in sids
            if sid not in set(chosen)
        ]
        rng.shuffle(remain)
        chosen += remain[
            : max_samples - len(chosen)
        ]

    return sorted(set(chosen))


# =============================================================================
# Main evaluation
# =============================================================================

def run_experiment(
    *,
    model,
    processor,
    token_map,
    decoder_layers,
    selected_layers,
    variants,
    modes,
    codebooks,
    targets,
    records,
    metadata,
    eval_sids,
    device,
    prompt_template,
    max_new_tokens,
    max_edit_norm,
    seed,
    out_dir,
    save_every,
):
    result_rows = []
    update_rows = []
    errors = []

    for i, sid in enumerate(
        tqdm(
            eval_sids,
            desc="dynamic correct-like generation",
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

            # No-image reference is fixed for all intervention variants.
            noimg_pairs = capture_noimage_pairs(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                rec=rec,
                device=device,
                prompt_template=prompt_template,
            )

            # First-step diagnostics only for selecting a foil on baseline-
            # correct samples.
            base_scores = firststep_scores(
                model,
                real_batch,
                token_map,
            )
            best_non_gt = max(
                [r for r in RELATIONS if r != gt],
                key=lambda r: base_scores[r],
            )

            baseline_text, baseline_pred = generate_answer(
                model,
                processor,
                real_batch,
                max_new_tokens,
            )

            baseline_parsed = (
                baseline_pred in REL2ID
            )
            baseline_correct = int(
                baseline_pred == gt
            )

            if (
                baseline_pred in REL2ID
                and baseline_pred != gt
            ):
                foil = baseline_pred
                foil_kind = "fresh_generation_wrong"
            else:
                foil = best_non_gt
                foil_kind = "firststep_best_nonGT"

            # Fresh baseline row.
            result_rows.append({
                "sid": sid,
                "variant": "baseline",
                "active_layers": "",
                "mode": "baseline",
                "gt": gt,
                "foil": foil,
                "foil_kind": foil_kind,

                "cached_generation_group":
                    metadata["generation"].get(
                        sid, {}
                    ).get("generation_group", ""),
                "cached_generation_pred":
                    metadata["generation"].get(
                        sid, {}
                    ).get("generation_pred", ""),

                "baseline_text": baseline_text,
                "baseline_pred": baseline_pred or "",
                "baseline_parsed":
                    int(baseline_parsed),
                "baseline_correct":
                    baseline_correct,

                "edited_text": baseline_text,
                "edited_pred": baseline_pred or "",
                "edited_parsed":
                    int(baseline_parsed),
                "edited_correct":
                    baseline_correct,

                "W2C": 0,
                "C2W": 0,
            })

            for variant in variants:
                active_layers = variant["layers"]
                variant_name = variant["variant"]

                for mode in modes:
                    with DynamicAttentionRepair(
                        decoder_layers=decoder_layers,
                        active_layers=active_layers,
                        subj_pos=sp,
                        ref_pos=rp,
                        noimg_pairs=noimg_pairs,
                        codebooks=codebooks,
                        targets=targets,
                        gt=gt,
                        foil=foil,
                        mode=mode,
                        max_edit_norm=max_edit_norm,
                        sid=sid,
                        seed=seed,
                    ) as repair:
                        edited_text, edited_pred = generate_answer(
                            model,
                            processor,
                            real_batch,
                            max_new_tokens,
                        )

                    edited_parsed = (
                        edited_pred in REL2ID
                    )
                    edited_correct = int(
                        edited_pred == gt
                    )

                    result_rows.append({
                        "sid": sid,
                        "variant": variant_name,
                        "active_layers":
                            ",".join(
                                str(x) for x in active_layers
                            ),
                        "mode": mode,
                        "gt": gt,
                        "foil": foil,
                        "foil_kind": foil_kind,

                        "cached_generation_group":
                            metadata["generation"].get(
                                sid, {}
                            ).get("generation_group", ""),
                        "cached_generation_pred":
                            metadata["generation"].get(
                                sid, {}
                            ).get("generation_pred", ""),

                        "baseline_text": baseline_text,
                        "baseline_pred":
                            baseline_pred or "",
                        "baseline_parsed":
                            int(baseline_parsed),
                        "baseline_correct":
                            baseline_correct,

                        "edited_text": edited_text,
                        "edited_pred":
                            edited_pred or "",
                        "edited_parsed":
                            int(edited_parsed),
                        "edited_correct":
                            edited_correct,

                        "W2C": int(
                            baseline_correct == 0
                            and edited_correct == 1
                        ),
                        "C2W": int(
                            baseline_correct == 1
                            and edited_correct == 0
                        ),
                    })

                    for log in repair.logs:
                        rr = dict(log)
                        rr.update({
                            "variant": variant_name,
                            "active_layers":
                                ",".join(
                                    str(x)
                                    for x in active_layers
                                ),
                            "baseline_correct":
                                baseline_correct,
                            "baseline_pred":
                                baseline_pred or "",
                        })
                        update_rows.append(rr)

            del real_batch

            if (
                save_every > 0
                and i % save_every == 0
            ):
                write_csv(
                    out_dir / "generation_per_sample.csv",
                    result_rows,
                )
                write_csv(
                    out_dir / "dynamic_layer_updates.csv",
                    update_rows,
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
        out_dir / "generation_per_sample.csv",
        result_rows,
    )
    write_csv(
        out_dir / "dynamic_layer_updates.csv",
        update_rows,
    )
    write_csv(
        out_dir / "errors.csv",
        errors,
    )

    return result_rows, update_rows, errors


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(rows):
    baseline_by_sid = {}

    for r in rows:
        if r["mode"] == "baseline":
            baseline_by_sid[int(r["sid"])] = r

    out = []
    buckets = defaultdict(list)

    for r in rows:
        if r["mode"] == "baseline":
            continue
        key = (
            str(r["variant"]),
            str(r["mode"]),
        )
        buckets[key].append(r)

    for (variant, mode), rr in sorted(buckets.items()):
        n = len(rr)

        baseline_acc = safe_mean(
            r["baseline_correct"] for r in rr
        )
        edited_acc = safe_mean(
            r["edited_correct"] for r in rr
        )

        n_base_wrong = int(
            sum(
                int(r["baseline_correct"]) == 0
                for r in rr
            )
        )
        n_base_correct = int(
            sum(
                int(r["baseline_correct"]) == 1
                for r in rr
            )
        )

        W2C = int(sum(int(r["W2C"]) for r in rr))
        C2W = int(sum(int(r["C2W"]) for r in rr))

        out.append({
            "variant": variant,
            "mode": mode,
            "n": n,

            "baseline_parsed_rate": safe_frac(
                int(r["baseline_parsed"]) == 1
                for r in rr
            ),
            "edited_parsed_rate": safe_frac(
                int(r["edited_parsed"]) == 1
                for r in rr
            ),

            "baseline_acc": baseline_acc,
            "edited_acc": edited_acc,
            "acc_gain":
                edited_acc - baseline_acc,

            "n_baseline_wrong": n_base_wrong,
            "n_baseline_correct": n_base_correct,

            "W2C": W2C,
            "W2C_over_wrong": (
                W2C / n_base_wrong
                if n_base_wrong else float("nan")
            ),
            "C2W": C2W,
            "C2W_over_correct": (
                C2W / n_base_correct
                if n_base_correct else float("nan")
            ),
            "net": W2C - C2W,
        })

    out.sort(
        key=lambda r: (
            float(r["acc_gain"]),
            int(r["net"]),
        ),
        reverse=True,
    )
    return out


def summarize_updates(rows):
    buckets = defaultdict(list)

    for r in rows:
        group = (
            "baseline_correct"
            if int(r["baseline_correct"]) == 1
            else "baseline_wrong"
        )
        key = (
            str(r["variant"]),
            str(r["mode"]),
            int(r["layer"]),
            group,
        )
        buckets[key].append(r)

    out = []

    for key, rr in sorted(buckets.items()):
        variant, mode, li, group = key

        out.append({
            "variant": variant,
            "mode": mode,
            "layer": li,
            "baseline_group": group,
            "n": len(rr),

            "trigger_rate": safe_frac(
                int(r["triggered"]) == 1
                for r in rr
            ),
            "mean_edit_norm": safe_mean(
                r["edit_norm"] for r in rr
            ),

            "current_delta_s_GT": safe_mean(
                r["current_delta_s_GT"] for r in rr
            ),
            "target_delta_s_GT": safe_mean(
                r["target_delta_s_GT"] for r in rr
            ),
            "post_edit_delta_s_GT": safe_mean(
                r["post_edit_delta_s_GT"] for r in rr
            ),

            "current_delta_s_foil": safe_mean(
                r["current_delta_s_foil"] for r in rr
            ),
            "target_delta_s_foil": safe_mean(
                r["target_delta_s_foil"] for r in rr
            ),
            "post_edit_delta_s_foil": safe_mean(
                r["post_edit_delta_s_foil"] for r in rr
            ),

            "mean_requested_GT_correction": safe_mean(
                r["requested_GT_correction"] for r in rr
            ),
            "mean_requested_foil_correction": safe_mean(
                r["requested_foil_correction"] for r in rr
            ),
        })

    return out


def print_generation_summary(rows):
    print("\n" + "=" * 150)
    print("ACTUAL model.generate() — DYNAMIC CORRECT-LIKE ATTENTION REPAIR")
    print("=" * 150)
    print(
        "variant mode N | acc base->edit gain | "
        "W2C / wrong | C2W / correct | net | parsed"
    )

    for r in rows:
        print(
            f"{str(r['variant']):34s} "
            f"{str(r['mode']):22s} "
            f"{int(r['n']):3d} | "
            f"{float(r['baseline_acc']):.4f}->"
            f"{float(r['edited_acc']):.4f} "
            f"{float(r['acc_gain']):+7.4f} | "
            f"{int(r['W2C']):3d}/"
            f"{float(r['W2C_over_wrong']):.3f} | "
            f"{int(r['C2W']):3d}/"
            f"{float(r['C2W_over_correct']):.3f} | "
            f"{int(r['net']):+4d} | "
            f"{float(r['edited_parsed_rate']):.3f}"
        )


def print_update_summary(rows):
    print("\n" + "=" * 176)
    print("DYNAMIC LAYER UPDATE CHECK")
    print("=" * 176)
    print(
        "variant mode layer group | trig norm | "
        "GT current->target->post | foil current->target->post"
    )

    # Keep console compact: print only cumulative/all variants and singles.
    for r in rows:
        print(
            f"{str(r['variant']):30s} "
            f"{str(r['mode']):20s} "
            f"L{int(r['layer']):02d} "
            f"{str(r['baseline_group']):16s} | "
            f"{float(r['trigger_rate']):.3f} "
            f"{float(r['mean_edit_norm']):6.3f} | "
            f"{float(r['current_delta_s_GT']):+7.3f}->"
            f"{float(r['target_delta_s_GT']):+7.3f}->"
            f"{float(r['post_edit_delta_s_GT']):+7.3f} | "
            f"{float(r['current_delta_s_foil']):+7.3f}->"
            f"{float(r['target_delta_s_foil']):+7.3f}->"
            f"{float(r['post_edit_delta_s_foil']):+7.3f}"
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

    # Model.
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
        f"selected layers={selected_layers}"
    )

    modes = parse_words(args.modes)
    valid_modes = {
        "correct_like_match",
        "correct_like_safe",
        "gt_only",
        "foil_only",
        "random_correct_like",
    }
    bad = [m for m in modes if m not in valid_modes]
    if bad:
        raise ValueError(f"Unknown modes: {bad}")

    variants = build_variants(
        selected_layers,
        args.variant_strategy,
    )

    print("[variants]")
    for v in variants:
        print(
            f"  {v['variant']}: {v['layers']}"
        )
    print("[modes]", modes)

    codebooks, targets, target_rows = load_train_targets(
        prepost_dir=Path(args.prepost_dir),
        metadata=metadata,
        selected_layers=selected_layers,
        target_stat_name=args.target_stat,
    )

    write_csv(
        out_dir / "correct_attention_targets.csv",
        target_rows,
    )

    eval_sids = select_eval_sids(
        metadata,
        records,
        args.eval_split,
        args.max_samples,
        args.seed,
    )

    cached_groups = Counter(
        metadata["generation"].get(
            sid, {}
        ).get("generation_group", "unknown")
        for sid in eval_sids
    )

    print(
        f"[eval] N={len(eval_sids)}, "
        f"cached groups={dict(cached_groups)}"
    )

    token_map = relation_token_map(
        processor.tokenizer
    )

    result_rows, update_rows, errors = run_experiment(
        model=model,
        processor=processor,
        token_map=token_map,
        decoder_layers=decoder_layers,
        selected_layers=selected_layers,
        variants=variants,
        modes=modes,
        codebooks=codebooks,
        targets=targets,
        records=records,
        metadata=metadata,
        eval_sids=eval_sids,
        device=device,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        max_edit_norm=args.max_edit_norm,
        seed=args.seed,
        out_dir=out_dir,
        save_every=args.save_every,
    )

    generation_summary = summarize_generation(
        result_rows
    )
    update_summary = summarize_updates(
        update_rows
    )

    write_csv(
        out_dir / "generation_summary.csv",
        generation_summary,
    )
    write_csv(
        out_dir / "dynamic_layer_update_summary.csv",
        update_summary,
    )

    print_generation_summary(
        generation_summary
    )
    print_update_summary(
        update_summary
    )

    meta_out = {
        "experiment":
            "dynamic correct-like Attention Direction repair",

        "selected_layers": selected_layers,
        "variants": variants,
        "modes": modes,

        "target_source":
            "generation-correct TRAIN samples",

        "target_coordinate_system":
            "layer-specific POST_ATTN Direction prototypes",

        "dynamic_update_definition":
            "(post-pre)_img - (post-pre)_noimg, recomputed at each "
            "active layer after all upstream interventions",

        "primary_metric":
            "fresh full model.generate() accuracy / W2C / C2W",

        "oracle_status":
            "GT is known; wrong baseline generation relation or first-step "
            "best non-GT is used as foil",

        "important_comparison":
            "correct_like_safe vs random_correct_like, especially for "
            "cumulative multi-layer variants",

        "n_eval": len(eval_sids),
        "cached_generation_groups":
            dict(cached_groups),
        "n_errors": len(errors),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(
            meta_out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "correct_attention_targets.csv",
        "generation_per_sample.csv",
        "generation_summary.csv",
        "dynamic_layer_updates.csv",
        "dynamic_layer_update_summary.csv",
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
