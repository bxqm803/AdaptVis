#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matched within-block causal-source diagnostic for spatial update failures.

This script is designed for the current Qwen-7B Direction analysis.  It does
NOT repair the model.  It asks WHY the problematic layer updates differ between
actual generation-correct and generation-wrong samples.

Key idea
========
A bad layer update can have at least three explanations:

1) INHERITED / UPSTREAM
   The wrong sample already enters the block with a different spatial state.

2) ATTENTION READING / ROUTING FAILURE
   After matching wrong and correct samples on the block INPUT spatial state,
   the Attention update still shows:
       - missing GT update, and/or
       - excess foil/other-relation update.

3) MLP TRANSFORMATION FAILURE
   After matching wrong and correct samples on the POST-ATTENTION / PRE-MLP
   spatial state, the MLP update still shows:
       - missing GT update, and/or
       - excess foil/other-relation update.

The script performs BOTH matching tests, for ALL automatically selected
problematic layers rather than only L16.

Correct / wrong is based on cached ACTUAL model.generate() results.

Required existing outputs
=========================
Direction cache:
    <direction-dir>/vectors.npz
    <direction-dir>/sample_split_and_generation.csv

Four-way failure scan:
    <failure-dir>/fourway_role_summary.csv
    <failure-dir>/top_candidate_layers.csv

Required repo helper scripts
============================
extract_two_object_relation_states.py
analyze_layerwise_direction_failure_scan_v1.py

Automatic problem-layer selection
=================================
With --layers auto, a layer is selected when:
    pairwise_deficit_ci_positive == 1
AND
    wrong_minus_correct_gt_minus_maxnonGT_gap <= -min-role-gap

The default min-role-gap is 0.5.  You can override this with explicit layers:
    --layers 14,15,16,17,18,19,26,27

Within-block stages captured
============================
For every selected decoder block:
    block_input
    attn_update
    post_attn_residual
    pre_mlp_norm
    mlp_update
    block_output

Each stage is stored as the same image-minus-noimage subject-reference vector.

Four-way module update coordinates
==================================
For Attention and MLP updates:
    u_left  = delta_module dot mu_left,l
    u_right = delta_module dot mu_right,l
    u_above = delta_module dot mu_above,l
    u_below = delta_module dot mu_below,l

For each wrong sample:
    GT deficit   = correct_match(u_GT)   - wrong(u_GT)
    foil excess  = wrong(u_finalWrong)   - correct_match(u_finalWrong)
    other excess = max over the remaining two relation coordinates
    pair deficit = correct_match(u_GT-u_foil) - wrong(u_GT-u_foil)

Positive deficit/excess means the wrong computation is worse in that sense.

Two matching analyses
=====================
A) BLOCK-INPUT MATCHING
   Match each generation-wrong sample to a generation-correct test sample with
   the same GT, using the 4-way Direction scores + norm at the block input.

   If Attention still has a positive pair deficit after this matching, that is
   evidence that the divergence is generated inside Attention rather than only
   inherited from earlier layers.

B) PRE-MLP MATCHING
   After collecting real activations, rematch each wrong sample to a correct
   sample with the same GT using the 4-way scores + norm of the actual
   image-minus-noimage pre-MLP normalized state.

   If MLP still has a positive pair deficit after this matching, that is
   evidence for an MLP transformation difference beyond the post-Attention
   input difference.

Important limitation
====================
This remains a representation-space mechanism diagnostic, not full causal
proof.  A positive matched module deficit identifies where the computation
diverges after controlling for the measured spatial state.  The next causal
test would patch the identified Attention source contribution or MLP input/
output computation, not steer the whole Direction state.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python analyze_problem_layers_within_block_cause_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers auto \
  --eval-split test \
  --output-dir output/qwen7b_problem_layers_within_block_cause_v1 \
  --overwrite

Smoke test
==========
CUDA_VISIBLE_DEVICES=0 python analyze_problem_layers_within_block_cause_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 14,16,19 \
  --eval-split test \
  --max-wrong 10 \
  --output-dir output/qwen7b_problem_layers_within_block_cause_smoke \
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
STAGES = (
    "block_input",
    "attn_update",
    "post_attn_residual",
    "pre_mlp_norm",
    "mlp_update",
    "block_output",
)
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
    p.add_argument("--failure-dir", required=True)
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
        default="auto",
        help="auto, all, 14,16,19, or 14-19",
    )
    p.add_argument(
        "--min-role-gap",
        type=float,
        default=0.5,
        help=(
            "For auto selection, require "
            "wrong-correct GT-minus-maxNonGT gap <= -this value."
        ),
    )
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--max-wrong",
        type=int,
        default=None,
        help="Optional smoke-test cap on generation-wrong samples.",
    )
    p.add_argument(
        "--match-with-replacement",
        action="store_true",
        help=(
            "Allow the same correct control to match multiple wrong samples "
            "inside a GT/foil/layer group."
        ),
    )
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=20)
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


def norm_relation(x: Any) -> str:
    return direction.norm_relation(x)


def safe_mean(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(vals.mean()) if len(vals) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


# =============================================================================
# Direction assets / problem layers
# =============================================================================

def fit_codebook(X_train: np.ndarray, y_train: np.ndarray):
    center = X_train.mean(axis=0)
    Xc = X_train - center
    protos = []
    for rel in RELATIONS:
        mask = y_train == rel
        if not np.any(mask):
            raise RuntimeError(f"No train examples for {rel}")
        p = Xc[mask].mean(axis=0)
        p = p / max(float(np.linalg.norm(p)), EPS)
        protos.append(p)
    return center.astype(np.float32), np.stack(protos).astype(np.float32)


def load_direction_assets(direction_dir: Path):
    vec_path = direction_dir / "vectors.npz"
    gen_path = direction_dir / "sample_split_and_generation.csv"
    if not vec_path.exists():
        raise FileNotFoundError(vec_path)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    with np.load(vec_path, allow_pickle=True) as z:
        arr = {k: z[k] for k in z.files}

    gen_rows = read_csv(gen_path)

    sids = arr["sample_index"].astype(np.int64)
    labels = np.asarray([norm_relation(x) for x in arr["relation"]])
    residual = np.asarray(arr["residual"], dtype=np.float32)
    idx_by_sid = {int(s): i for i, s in enumerate(sids.tolist())}

    split_by_sid = {}
    gen_by_sid = {}
    for r in gen_rows:
        sid = int(r["sample_index"])
        split_by_sid[sid] = str(r.get("split", "")).strip()
        group = str(r.get("generation_group", "")).strip().lower()
        pred = norm_relation(r.get("generation_pred", ""))
        idx = idx_by_sid.get(sid)
        gt = labels[idx] if idx is not None else ""
        if group not in ("correct", "wrong"):
            if pred in REL2ID and gt in REL2ID:
                group = "correct" if pred == gt else "wrong"
        gen_by_sid[sid] = {
            "generation_group": group,
            "generation_pred": pred,
            "generation_text": str(r.get("generation_text", "")),
        }

    train_idx = np.asarray(
        [
            idx_by_sid[sid]
            for sid, sp in split_by_sid.items()
            if sp == "train" and sid in idx_by_sid
        ],
        dtype=np.int64,
    )
    if len(train_idx) == 0:
        raise RuntimeError("No train split for Direction codebook.")

    codebooks = {}
    for li in range(residual.shape[1]):
        center, protos = fit_codebook(
            residual[train_idx, li, :],
            labels[train_idx],
        )
        codebooks[li] = {"center": center, "protos": protos}

    return {
        "labels": labels,
        "residual": residual,
        "idx_by_sid": idx_by_sid,
        "split": split_by_sid,
        "generation": gen_by_sid,
        "codebooks": codebooks,
        "n_layers": residual.shape[1],
    }


def parse_explicit_layers(text: str, n_layers: int) -> List[int]:
    if text.strip().lower() == "all":
        return list(range(1, n_layers))

    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(part))

    out = sorted(set(out))
    bad = [x for x in out if not 1 <= x < n_layers]
    if bad:
        raise ValueError(f"Invalid layers: {bad}")
    return out


def select_problem_layers(
    failure_dir: Path,
    layer_arg: str,
    n_layers: int,
    min_role_gap: float,
):
    if layer_arg.strip().lower() != "auto":
        layers = parse_explicit_layers(layer_arg, n_layers)
        audit = [
            {
                "layer": li,
                "selected": 1,
                "reason": "explicit",
            }
            for li in layers
        ]
        return layers, audit

    top_path = failure_dir / "top_candidate_layers.csv"
    if not top_path.exists():
        raise FileNotFoundError(top_path)

    rows = read_csv(top_path)
    layers = []
    audit = []

    for r in rows:
        li = int(r["layer"])
        pair_ci = int(float(r.get("pairwise_deficit_ci_positive", "0")))
        role_gap = float(
            r["wrong_minus_correct_gt_minus_maxnonGT_gap"]
        )
        selected = (
            pair_ci == 1
            and role_gap <= -float(min_role_gap)
        )
        audit.append({
            "layer": li,
            "pairwise_deficit_ci_positive": pair_ci,
            "wrong_minus_correct_gt_minus_maxnonGT_gap": role_gap,
            "min_role_gap_threshold": min_role_gap,
            "selected": int(selected),
            "reason": (
                "pairDefCI>0_and_sameTestRoleGapNegative"
                if selected else "not_selected"
            ),
        })
        if selected:
            layers.append(li)

    layers = sorted(set(layers))
    if not layers:
        raise RuntimeError(
            "Auto selection found zero problem layers. "
            "Lower --min-role-gap or use --layers explicitly."
        )
    return layers, audit


# =============================================================================
# Model structure / stage capture
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
                if (
                    hasattr(b, "self_attn")
                    and hasattr(b, "mlp")
                    and hasattr(b, "input_layernorm")
                    and hasattr(b, "post_attention_layernorm")
                ):
                    return layers, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers.")


def first_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            if torch.is_tensor(item):
                return item
    raise RuntimeError(f"No tensor in output type={type(x)}")


def pool_positions(
    x: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    valid = [
        int(p) for p in positions
        if 0 <= int(p) < int(x.shape[1])
    ]
    if not valid:
        raise RuntimeError("No valid object-token positions.")
    idx = torch.as_tensor(
        valid,
        device=x.device,
        dtype=torch.long,
    )
    return x[0].index_select(0, idx).mean(dim=0)


class WithinBlockCollector:
    """
    Capture pooled subject-reference differences at multiple internal points.

    We hook:
      input_layernorm PRE       -> block_input
      self_attn OUTPUT          -> attn_update
      post_attention_layernorm PRE -> post_attn_residual
      post_attention_layernorm OUTPUT -> pre_mlp_norm
      mlp OUTPUT                -> mlp_update
      block OUTPUT              -> block_output
    """

    def __init__(
        self,
        decoder_layers,
        layers: Sequence[int],
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
    ):
        self.decoder_layers = decoder_layers
        self.layers = list(layers)
        self.subj = list(map(int, subject_positions))
        self.ref = list(map(int, reference_positions))
        self.data: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.handles = []

    def _pooldiff(self, x: torch.Tensor):
        return (
            pool_positions(x, self.subj)
            - pool_positions(x, self.ref)
        ).detach().float().cpu()

    def _pre_hook(self, li: int, stage: str):
        def hook(_module, args):
            if not args:
                return
            x = first_tensor(args)
            self.data[li][stage] = self._pooldiff(x)
        return hook

    def _fwd_hook(self, li: int, stage: str):
        def hook(_module, _args, output):
            x = first_tensor(output)
            self.data[li][stage] = self._pooldiff(x)
            return output
        return hook

    def __enter__(self):
        for li in self.layers:
            block = self.decoder_layers[li]

            self.handles.append(
                block.input_layernorm.register_forward_pre_hook(
                    self._pre_hook(li, "block_input")
                )
            )
            self.handles.append(
                block.self_attn.register_forward_hook(
                    self._fwd_hook(li, "attn_update")
                )
            )
            self.handles.append(
                block.post_attention_layernorm.register_forward_pre_hook(
                    self._pre_hook(li, "post_attn_residual")
                )
            )
            self.handles.append(
                block.post_attention_layernorm.register_forward_hook(
                    self._fwd_hook(li, "pre_mlp_norm")
                )
            )
            self.handles.append(
                block.mlp.register_forward_hook(
                    self._fwd_hook(li, "mlp_update")
                )
            )
            self.handles.append(
                block.register_forward_hook(
                    self._fwd_hook(li, "block_output")
                )
            )
        return self

    def validate(self):
        for li in self.layers:
            missing = [
                s for s in STAGES
                if s not in self.data.get(li, {})
            ]
            if missing:
                raise RuntimeError(
                    f"L{li} missing captured stages: {missing}"
                )

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __exit__(self, *_args):
        self.close()


def build_batch_and_positions(
    *,
    processor,
    device,
    question,
    subject,
    reference,
    image: Optional[Image.Image],
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
        subject,
    )
    ref_pos = direction.locate_phrase_positions(
        processor.tokenizer,
        ids,
        reference,
    )
    return batch, subj_pos, ref_pos


def capture_one_condition(
    *,
    model,
    processor,
    device,
    decoder_layers,
    selected_layers,
    question,
    subject,
    reference,
    image,
):
    batch, subj_pos, ref_pos = build_batch_and_positions(
        processor=processor,
        device=device,
        question=question,
        subject=subject,
        reference=reference,
        image=image,
    )

    with WithinBlockCollector(
        decoder_layers,
        selected_layers,
        subj_pos,
        ref_pos,
    ) as col:
        with torch.inference_mode():
            _ = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        col.validate()

    out = {}
    for li in selected_layers:
        out[li] = {
            stage: col.data[li][stage].numpy().astype(np.float32)
            for stage in STAGES
        }

    del batch
    return out


# =============================================================================
# Stage geometry
# =============================================================================

def fourway(v: np.ndarray, protos: np.ndarray) -> np.ndarray:
    return (v @ protos.T).astype(np.float64)


def q_state_scores(
    v: np.ndarray,
    center: np.ndarray,
    protos: np.ndarray,
) -> np.ndarray:
    return ((v - center) @ protos.T).astype(np.float64)


def stage_feature(
    vec: np.ndarray,
    *,
    center: Optional[np.ndarray],
    protos: np.ndarray,
    include_norm: bool = True,
):
    if center is None:
        scores = fourway(vec, protos)
    else:
        scores = q_state_scores(vec, center, protos)
    if include_norm:
        return np.concatenate(
            [scores, [float(np.linalg.norm(vec))]]
        ).astype(np.float64)
    return scores.astype(np.float64)


def module_metrics(
    vec: np.ndarray,
    *,
    protos: np.ndarray,
    gt: str,
    foil: str,
):
    u = fourway(vec, protos)
    gi = REL2ID[gt]
    fi = REL2ID[foil]
    others = [
        r for r in RELATIONS
        if r not in (gt, foil)
    ]

    return {
        "u": u,
        "u_gt": float(u[gi]),
        "u_foil": float(u[fi]),
        "pair": float(u[gi] - u[fi]),
        "other_relations": others,
    }


# =============================================================================
# Match construction from cached block-input state
# =============================================================================

def cached_input_feature(
    assets,
    sid: int,
    layer: int,
):
    idx = assets["idx_by_sid"][sid]
    prev = layer - 1
    r = assets["residual"][idx, prev]
    cb = assets["codebooks"][prev]
    return stage_feature(
        r,
        center=cb["center"],
        protos=cb["protos"],
        include_norm=True,
    )


def standardize_features(A: np.ndarray, B: np.ndarray):
    X = np.concatenate([A, B], axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (A - mean) / std, (B - mean) / std


def nearest_matches(
    *,
    wrong_sids: Sequence[int],
    correct_sids: Sequence[int],
    feature_fn,
    with_replacement: bool,
):
    """
    One-to-one greedy nearest matching by default.
    Returns list of (wrong_sid, correct_sid, distance).
    """
    if not wrong_sids or not correct_sids:
        return []

    W = np.stack([feature_fn(s) for s in wrong_sids])
    C = np.stack([feature_fn(s) for s in correct_sids])
    Wz, Cz = standardize_features(W, C)

    dist = np.sqrt(
        ((Wz[:, None, :] - Cz[None, :, :]) ** 2).sum(axis=2)
    )

    matches = []

    if with_replacement:
        for i, wsid in enumerate(wrong_sids):
            j = int(np.argmin(dist[i]))
            matches.append(
                (int(wsid), int(correct_sids[j]), float(dist[i, j]))
            )
        return matches

    # Greedy global nearest pair, no replacement within this GT/foil/layer
    # matching group.
    available_w = set(range(len(wrong_sids)))
    available_c = set(range(len(correct_sids)))

    while available_w and available_c:
        best = None
        for i in available_w:
            js = list(available_c)
            j_local = int(np.argmin(dist[i, js]))
            j = js[j_local]
            d = float(dist[i, j])
            if best is None or d < best[0]:
                best = (d, i, j)
        d, i, j = best
        matches.append(
            (int(wrong_sids[i]), int(correct_sids[j]), d)
        )
        available_w.remove(i)
        available_c.remove(j)

    return matches


def build_block_input_matches(
    *,
    assets,
    selected_layers,
    eval_split,
    max_wrong,
    seed,
    with_replacement,
):
    rng = random.Random(seed)

    eval_sids = [
        sid
        for sid in assets["idx_by_sid"]
        if eval_split == "all"
        or assets["split"].get(sid, "") == eval_split
    ]

    correct_sids = [
        sid for sid in eval_sids
        if assets["generation"].get(sid, {}).get(
            "generation_group", ""
        ) == "correct"
    ]
    wrong_sids = [
        sid for sid in eval_sids
        if assets["generation"].get(sid, {}).get(
            "generation_group", ""
        ) == "wrong"
        and assets["generation"].get(sid, {}).get(
            "generation_pred", ""
        ) in REL2ID
    ]

    if max_wrong is not None and len(wrong_sids) > max_wrong:
        rng.shuffle(wrong_sids)
        wrong_sids = wrong_sids[:max_wrong]

    rows = []

    for li in selected_layers:
        # Match separately for each GT x final-foil group.
        groups = defaultdict(list)
        for sid in wrong_sids:
            idx = assets["idx_by_sid"][sid]
            gt = assets["labels"][idx]
            foil = assets["generation"][sid]["generation_pred"]
            if gt in REL2ID and foil in REL2ID and foil != gt:
                groups[(gt, foil)].append(sid)

        for (gt, foil), wsids in sorted(groups.items()):
            csids = [
                sid for sid in correct_sids
                if assets["labels"][assets["idx_by_sid"][sid]] == gt
            ]
            if not csids:
                continue

            matches = nearest_matches(
                wrong_sids=wsids,
                correct_sids=csids,
                feature_fn=lambda s, li=li: cached_input_feature(
                    assets, s, li
                ),
                with_replacement=with_replacement,
            )

            for wsid, csid, dist in matches:
                wf = cached_input_feature(assets, wsid, li)
                cf = cached_input_feature(assets, csid, li)
                rows.append({
                    "layer": li,
                    "gt": gt,
                    "foil": foil,
                    "wrong_sid": wsid,
                    "correct_sid": csid,
                    "match_distance_z": dist,
                    "input_feature_l2_raw":
                        float(np.linalg.norm(wf - cf)),
                })

    return rows


# =============================================================================
# Activation capture for all matched samples
# =============================================================================

def capture_needed_samples(
    *,
    match_rows,
    records_by_sid,
    model,
    processor,
    device,
    decoder_layers,
    selected_layers,
    prompt_template,
    out_dir,
    save_every,
):
    needed = sorted(
        {
            int(r["wrong_sid"])
            for r in match_rows
        }
        |
        {
            int(r["correct_sid"])
            for r in match_rows
        }
    )

    stage_cache: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    errors = []

    for n, sid in enumerate(
        tqdm(needed, desc="capture matched sample stages"),
        1,
    ):
        rec = records_by_sid.get(sid)
        if rec is None:
            errors.append({
                "sid": sid,
                "error": "record_not_found",
            })
            continue

        image = None
        try:
            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            image = Image.open(rec.image_path).convert("RGB")

            real = capture_one_condition(
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
                image=image,
            )
            noimg = capture_one_condition(
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
                image=None,
            )

            stage_cache[sid] = {}
            for li in selected_layers:
                stage_cache[sid][li] = {}
                for stage in STAGES:
                    stage_cache[sid][li][stage] = (
                        real[li][stage] - noimg[li][stage]
                    ).astype(np.float32)

            if n % save_every == 0:
                # Save compact numeric cache to survive long runs.
                save_stage_cache_npz(
                    out_dir / "stage_cache_partial.npz",
                    stage_cache,
                    selected_layers,
                )

        except Exception as e:
            errors.append({
                "sid": sid,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[ERROR sid={sid}] {type(e).__name__}: {e}"
            )
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_stage_cache_npz(
        out_dir / "stage_cache.npz",
        stage_cache,
        selected_layers,
    )
    (out_dir / "capture_errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return stage_cache, errors


def save_stage_cache_npz(
    path: Path,
    stage_cache,
    selected_layers,
):
    sids = sorted(stage_cache)
    if not sids:
        return

    arrays = {
        "sid": np.asarray(sids, dtype=np.int64),
        "layers": np.asarray(selected_layers, dtype=np.int64),
        "stages": np.asarray(STAGES, dtype=object),
    }

    for li in selected_layers:
        for stage in STAGES:
            vals = [
                stage_cache[sid][li][stage]
                for sid in sids
                if li in stage_cache[sid]
            ]
            if len(vals) == len(sids):
                arrays[f"L{li}_{stage}"] = np.stack(vals)

    np.savez_compressed(path, **arrays)


# =============================================================================
# Block-input matched module decomposition
# =============================================================================

def build_input_matched_pair_rows(
    *,
    matches,
    stage_cache,
    assets,
):
    rows = []

    for m in matches:
        li = int(m["layer"])
        wsid = int(m["wrong_sid"])
        csid = int(m["correct_sid"])
        gt = str(m["gt"])
        foil = str(m["foil"])

        if wsid not in stage_cache or csid not in stage_cache:
            continue

        protos = assets["codebooks"][li]["protos"]
        prev_cb = assets["codebooks"][li - 1]

        wstage = stage_cache[wsid][li]
        cstage = stage_cache[csid][li]

        # Check actual captured block-input match in previous-layer basis.
        w_in_scores = q_state_scores(
            wstage["block_input"],
            prev_cb["center"],
            prev_cb["protos"],
        )
        c_in_scores = q_state_scores(
            cstage["block_input"],
            prev_cb["center"],
            prev_cb["protos"],
        )
        gi = REL2ID[gt]
        fi = REL2ID[foil]

        row = dict(m)
        row.update({
            "wrong_input_gt_score": float(w_in_scores[gi]),
            "correct_input_gt_score": float(c_in_scores[gi]),
            "input_gt_gap_wrong_minus_correct":
                float(w_in_scores[gi] - c_in_scores[gi]),
            "wrong_input_foil_score": float(w_in_scores[fi]),
            "correct_input_foil_score": float(c_in_scores[fi]),
            "input_foil_gap_wrong_minus_correct":
                float(w_in_scores[fi] - c_in_scores[fi]),
            "wrong_input_pair_margin":
                float(w_in_scores[gi] - w_in_scores[fi]),
            "correct_input_pair_margin":
                float(c_in_scores[gi] - c_in_scores[fi]),
            "input_pair_gap_wrong_minus_correct":
                float(
                    (w_in_scores[gi] - w_in_scores[fi])
                    - (c_in_scores[gi] - c_in_scores[fi])
                ),
        })

        for module, stage_name in (
            ("attn", "attn_update"),
            ("mlp", "mlp_update"),
        ):
            wm = module_metrics(
                wstage[stage_name],
                protos=protos,
                gt=gt,
                foil=foil,
            )
            cm = module_metrics(
                cstage[stage_name],
                protos=protos,
                gt=gt,
                foil=foil,
            )

            row[f"{module}_wrong_u_gt"] = wm["u_gt"]
            row[f"{module}_correct_u_gt"] = cm["u_gt"]
            row[f"{module}_gt_deficit_correct_minus_wrong"] = (
                cm["u_gt"] - wm["u_gt"]
            )

            row[f"{module}_wrong_u_foil"] = wm["u_foil"]
            row[f"{module}_correct_u_foil"] = cm["u_foil"]
            row[f"{module}_foil_excess_wrong_minus_correct"] = (
                wm["u_foil"] - cm["u_foil"]
            )

            row[f"{module}_wrong_pair"] = wm["pair"]
            row[f"{module}_correct_pair"] = cm["pair"]
            row[f"{module}_pair_deficit_correct_minus_wrong"] = (
                cm["pair"] - wm["pair"]
            )

            other_excess = {}
            for rel in wm["other_relations"]:
                ri = REL2ID[rel]
                excess = float(wm["u"][ri] - cm["u"][ri])
                other_excess[rel] = excess
                row[f"{module}_other_excess_{rel}"] = excess

            best_other = max(
                other_excess,
                key=other_excess.get,
            )
            row[f"{module}_max_other_excess_relation"] = best_other
            row[f"{module}_max_other_excess"] = other_excess[best_other]

            # Save all four coordinates.
            for rel in RELATIONS:
                ri = REL2ID[rel]
                row[f"{module}_wrong_u_{rel}"] = float(wm["u"][ri])
                row[f"{module}_correct_u_{rel}"] = float(cm["u"][ri])
                row[f"{module}_wrong_minus_correct_u_{rel}"] = float(
                    wm["u"][ri] - cm["u"][ri]
                )

        # Stage state margins in current-layer basis.
        for stage in (
            "post_attn_residual",
            "pre_mlp_norm",
            "block_output",
        ):
            wv = wstage[stage]
            cv = cstage[stage]

            # post-attn and block-output are residual states; center them.
            # pre_mlp_norm is normalized, so use raw projection only.
            if stage == "pre_mlp_norm":
                ws = fourway(wv, protos)
                cs = fourway(cv, protos)
            else:
                center = assets["codebooks"][li]["center"]
                ws = q_state_scores(wv, center, protos)
                cs = q_state_scores(cv, center, protos)

            row[f"{stage}_wrong_pair_margin"] = float(
                ws[gi] - ws[fi]
            )
            row[f"{stage}_correct_pair_margin"] = float(
                cs[gi] - cs[fi]
            )
            row[f"{stage}_pair_gap_wrong_minus_correct"] = float(
                (ws[gi] - ws[fi]) - (cs[gi] - cs[fi])
            )

        # Reconstruction sanity.
        for sid_name, st in (("wrong", wstage), ("correct", cstage)):
            reconstructed_post = (
                st["block_input"] + st["attn_update"]
            )
            reconstructed_out = (
                st["post_attn_residual"] + st["mlp_update"]
            )
            row[f"{sid_name}_post_attn_recon_relerr"] = float(
                np.linalg.norm(
                    reconstructed_post - st["post_attn_residual"]
                )
                / max(
                    float(np.linalg.norm(st["post_attn_residual"])),
                    EPS,
                )
            )
            row[f"{sid_name}_block_out_recon_relerr"] = float(
                np.linalg.norm(
                    reconstructed_out - st["block_output"]
                )
                / max(
                    float(np.linalg.norm(st["block_output"])),
                    EPS,
                )
            )

        rows.append(row)

    return rows


# =============================================================================
# Second matching: actual pre-MLP state
# =============================================================================

def premlp_feature(
    stage_cache,
    assets,
    sid: int,
    layer: int,
):
    vec = stage_cache[sid][layer]["pre_mlp_norm"]
    protos = assets["codebooks"][layer]["protos"]
    # No residual-stream center for normalized state.
    return stage_feature(
        vec,
        center=None,
        protos=protos,
        include_norm=True,
    )


def build_premlp_matches(
    *,
    input_matches,
    stage_cache,
    assets,
    with_replacement,
):
    """
    Rematch wrong samples among the same evaluation correct pool represented
    in input_matches. Matching is separate by layer, GT, final foil.
    """
    rows = []

    by_group = defaultdict(lambda: {"wrong": set(), "correct_pool": set()})
    for m in input_matches:
        li = int(m["layer"])
        gt = str(m["gt"])
        foil = str(m["foil"])
        by_group[(li, gt, foil)]["wrong"].add(int(m["wrong_sid"]))

    # Build full correct pool from all captured generation-correct samples with
    # same GT. stage_cache contains all controls used somewhere; this avoids
    # requiring another forward pass.
    captured_correct_by_gt = defaultdict(list)
    for sid in stage_cache:
        gen = assets["generation"].get(sid, {})
        if gen.get("generation_group", "") != "correct":
            continue
        gt = assets["labels"][assets["idx_by_sid"][sid]]
        captured_correct_by_gt[gt].append(sid)

    for (li, gt, foil), d in sorted(by_group.items()):
        wsids = [
            sid for sid in sorted(d["wrong"])
            if sid in stage_cache
        ]
        csids = [
            sid for sid in captured_correct_by_gt.get(gt, [])
            if sid in stage_cache
        ]
        if not wsids or not csids:
            continue

        matches = nearest_matches(
            wrong_sids=wsids,
            correct_sids=csids,
            feature_fn=lambda s, li=li: premlp_feature(
                stage_cache, assets, s, li
            ),
            with_replacement=with_replacement,
        )

        for wsid, csid, dist in matches:
            rows.append({
                "layer": li,
                "gt": gt,
                "foil": foil,
                "wrong_sid": wsid,
                "correct_sid": csid,
                "premlp_match_distance_z": dist,
            })

    return rows


def build_premlp_matched_mlp_rows(
    *,
    matches,
    stage_cache,
    assets,
):
    rows = []

    for m in matches:
        li = int(m["layer"])
        wsid = int(m["wrong_sid"])
        csid = int(m["correct_sid"])
        gt = str(m["gt"])
        foil = str(m["foil"])

        protos = assets["codebooks"][li]["protos"]

        wpre = stage_cache[wsid][li]["pre_mlp_norm"]
        cpre = stage_cache[csid][li]["pre_mlp_norm"]
        wmlp = stage_cache[wsid][li]["mlp_update"]
        cmlp = stage_cache[csid][li]["mlp_update"]

        wpre_scores = fourway(wpre, protos)
        cpre_scores = fourway(cpre, protos)
        gi, fi = REL2ID[gt], REL2ID[foil]

        wm = module_metrics(
            wmlp,
            protos=protos,
            gt=gt,
            foil=foil,
        )
        cm = module_metrics(
            cmlp,
            protos=protos,
            gt=gt,
            foil=foil,
        )

        row = dict(m)
        row.update({
            "wrong_premlp_gt": float(wpre_scores[gi]),
            "correct_premlp_gt": float(cpre_scores[gi]),
            "wrong_premlp_foil": float(wpre_scores[fi]),
            "correct_premlp_foil": float(cpre_scores[fi]),
            "premlp_pair_gap_wrong_minus_correct": float(
                (wpre_scores[gi] - wpre_scores[fi])
                - (cpre_scores[gi] - cpre_scores[fi])
            ),

            "mlp_gt_deficit_correct_minus_wrong":
                cm["u_gt"] - wm["u_gt"],
            "mlp_foil_excess_wrong_minus_correct":
                wm["u_foil"] - cm["u_foil"],
            "mlp_pair_deficit_correct_minus_wrong":
                cm["pair"] - wm["pair"],
        })

        other_excess = {}
        for rel in wm["other_relations"]:
            ri = REL2ID[rel]
            other_excess[rel] = float(
                wm["u"][ri] - cm["u"][ri]
            )
        best_other = max(other_excess, key=other_excess.get)
        row["mlp_max_other_excess_relation"] = best_other
        row["mlp_max_other_excess"] = other_excess[best_other]

        for rel in RELATIONS:
            ri = REL2ID[rel]
            row[f"mlp_wrong_u_{rel}"] = float(wm["u"][ri])
            row[f"mlp_correct_u_{rel}"] = float(cm["u"][ri])
            row[f"mlp_wrong_minus_correct_u_{rel}"] = float(
                wm["u"][ri] - cm["u"][ri]
            )

        rows.append(row)

    return rows


# =============================================================================
# Statistics
# =============================================================================

def paired_bootstrap_mean_ci(
    vals: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(a.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        x = a[rng.integers(0, len(a), len(a))]
        boots[i] = x.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def summarize_input_matched(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed)
    out = []

    for li in selected_layers:
        rr = [r for r in rows if int(r["layer"]) == li]
        row = {
            "layer": li,
            "n_pairs": len(rr),
            "mean_input_match_distance_z":
                safe_mean(r["match_distance_z"] for r in rr),
            "mean_input_pair_gap_wrong_minus_correct":
                safe_mean(
                    r["input_pair_gap_wrong_minus_correct"]
                    for r in rr
                ),
            "mean_post_attn_pair_gap_wrong_minus_correct":
                safe_mean(
                    r["post_attn_residual_pair_gap_wrong_minus_correct"]
                    for r in rr
                ),
            "mean_block_output_pair_gap_wrong_minus_correct":
                safe_mean(
                    r["block_output_pair_gap_wrong_minus_correct"]
                    for r in rr
                ),
            "mean_wrong_post_attn_recon_relerr":
                safe_mean(
                    r["wrong_post_attn_recon_relerr"] for r in rr
                ),
            "mean_wrong_block_out_recon_relerr":
                safe_mean(
                    r["wrong_block_out_recon_relerr"] for r in rr
                ),
        }

        for module in ("attn", "mlp"):
            for metric in (
                "gt_deficit_correct_minus_wrong",
                "foil_excess_wrong_minus_correct",
                "max_other_excess",
                "pair_deficit_correct_minus_wrong",
            ):
                key = f"{module}_{metric}"
                vals = [float(r[key]) for r in rr]
                m, lo, hi = paired_bootstrap_mean_ci(
                    vals, bootstrap, rng
                )
                row[f"mean_{key}"] = m
                row[f"{key}_ci95_lo"] = lo
                row[f"{key}_ci95_hi"] = hi
                row[f"{key}_positive_rate"] = safe_frac(
                    v > 0 for v in vals
                )

        out.append(row)

    return out


def summarize_premlp_matched(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed + 1000)
    out = []

    for li in selected_layers:
        rr = [r for r in rows if int(r["layer"]) == li]
        row = {
            "layer": li,
            "n_pairs": len(rr),
            "mean_premlp_match_distance_z":
                safe_mean(r["premlp_match_distance_z"] for r in rr),
            "mean_premlp_pair_gap_wrong_minus_correct":
                safe_mean(
                    r["premlp_pair_gap_wrong_minus_correct"]
                    for r in rr
                ),
        }

        for metric in (
            "mlp_gt_deficit_correct_minus_wrong",
            "mlp_foil_excess_wrong_minus_correct",
            "mlp_max_other_excess",
            "mlp_pair_deficit_correct_minus_wrong",
        ):
            vals = [float(r[metric]) for r in rr]
            m, lo, hi = paired_bootstrap_mean_ci(
                vals, bootstrap, rng
            )
            row[f"mean_{metric}"] = m
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
            row[f"{metric}_positive_rate"] = safe_frac(
                v > 0 for v in vals
            )

        out.append(row)

    return out


# =============================================================================
# Cause classification
# =============================================================================

def ci_positive(row, prefix: str) -> bool:
    lo = float(row[f"{prefix}_ci95_lo"])
    return math.isfinite(lo) and lo > 0


def classify_causes(
    input_summary,
    premlp_summary,
    selected_layers,
):
    inp = {int(r["layer"]): r for r in input_summary}
    pre = {int(r["layer"]): r for r in premlp_summary}

    rows = []

    for li in selected_layers:
        a = inp.get(li, {})
        b = pre.get(li, {})

        attn_pair = float(
            a.get(
                "mean_attn_pair_deficit_correct_minus_wrong",
                float("nan"),
            )
        )
        mlp_after_input_match = float(
            a.get(
                "mean_mlp_pair_deficit_correct_minus_wrong",
                float("nan"),
            )
        )
        mlp_after_premlp_match = float(
            b.get(
                "mean_mlp_pair_deficit_correct_minus_wrong",
                float("nan"),
            )
        )

        attn_sig = (
            "attn_pair_deficit_correct_minus_wrong_ci95_lo" in a
            and ci_positive(
                a,
                "attn_pair_deficit_correct_minus_wrong",
            )
        )
        mlp_sig = (
            "mlp_pair_deficit_correct_minus_wrong_ci95_lo" in b
            and ci_positive(
                b,
                "mlp_pair_deficit_correct_minus_wrong",
            )
        )

        input_gap = float(
            a.get(
                "mean_input_pair_gap_wrong_minus_correct",
                float("nan"),
            )
        )
        post_attn_gap = float(
            a.get(
                "mean_post_attn_pair_gap_wrong_minus_correct",
                float("nan"),
            )
        )
        output_gap = float(
            a.get(
                "mean_block_output_pair_gap_wrong_minus_correct",
                float("nan"),
            )
        )

        if attn_sig and mlp_sig:
            cause = "attention_and_mlp_independent_divergence"
        elif attn_sig and not mlp_sig:
            cause = "attention_reading_or_routing_primary"
        elif (not attn_sig) and mlp_sig:
            cause = "mlp_transformation_primary"
        else:
            # If matching made module deficits disappear, the observed bad
            # layer update is likely inherited from upstream state differences,
            # or lies outside this Direction-coordinate control.
            cause = "inherited_or_unmeasured_state"

        rows.append({
            "layer": li,
            "classification": cause,

            "input_matched_attention_pair_deficit": attn_pair,
            "attention_pair_deficit_ci95_lo": a.get(
                "attn_pair_deficit_correct_minus_wrong_ci95_lo",
                float("nan"),
            ),
            "attention_pair_deficit_ci95_hi": a.get(
                "attn_pair_deficit_correct_minus_wrong_ci95_hi",
                float("nan"),
            ),
            "attention_significant_positive": int(attn_sig),

            "input_matched_mlp_pair_deficit":
                mlp_after_input_match,

            "premlp_matched_mlp_pair_deficit":
                mlp_after_premlp_match,
            "premlp_mlp_pair_deficit_ci95_lo": b.get(
                "mlp_pair_deficit_correct_minus_wrong_ci95_lo",
                float("nan"),
            ),
            "premlp_mlp_pair_deficit_ci95_hi": b.get(
                "mlp_pair_deficit_correct_minus_wrong_ci95_hi",
                float("nan"),
            ),
            "mlp_significant_after_premlp_matching": int(mlp_sig),

            "matched_input_pair_gap_wrong_minus_correct": input_gap,
            "post_attn_pair_gap_wrong_minus_correct": post_attn_gap,
            "block_output_pair_gap_wrong_minus_correct": output_gap,

            "attention_gt_deficit": a.get(
                "mean_attn_gt_deficit_correct_minus_wrong",
                float("nan"),
            ),
            "attention_foil_excess": a.get(
                "mean_attn_foil_excess_wrong_minus_correct",
                float("nan"),
            ),
            "attention_other_excess": a.get(
                "mean_attn_max_other_excess",
                float("nan"),
            ),

            "mlp_gt_deficit_after_premlp_match": b.get(
                "mean_mlp_gt_deficit_correct_minus_wrong",
                float("nan"),
            ),
            "mlp_foil_excess_after_premlp_match": b.get(
                "mean_mlp_foil_excess_wrong_minus_correct",
                float("nan"),
            ),
            "mlp_other_excess_after_premlp_match": b.get(
                "mean_mlp_max_other_excess",
                float("nan"),
            ),
        })

    return rows


# =============================================================================
# Console
# =============================================================================

def print_input_summary(rows):
    print("\n" + "=" * 180)
    print("BLOCK-INPUT MATCHED: WHERE DOES THE DIVERGENCE APPEAR?")
    print("=" * 180)
    print(
        "layer N  inputGap postAttnGap outputGap | "
        "ATTN: GTdef foilEx otherEx pairDef[95%CI] | "
        "MLP(input-matched): pairDef[95%CI]"
    )
    for r in rows:
        print(
            f"L{int(r['layer']):02d} {int(r['n_pairs']):3d} "
            f"{float(r['mean_input_pair_gap_wrong_minus_correct']):+7.3f} "
            f"{float(r['mean_post_attn_pair_gap_wrong_minus_correct']):+7.3f} "
            f"{float(r['mean_block_output_pair_gap_wrong_minus_correct']):+7.3f} | "
            f"{float(r['mean_attn_gt_deficit_correct_minus_wrong']):+7.3f} "
            f"{float(r['mean_attn_foil_excess_wrong_minus_correct']):+7.3f} "
            f"{float(r['mean_attn_max_other_excess']):+7.3f} "
            f"{float(r['mean_attn_pair_deficit_correct_minus_wrong']):+7.3f}"
            f"[{float(r['attn_pair_deficit_correct_minus_wrong_ci95_lo']):+6.3f},"
            f"{float(r['attn_pair_deficit_correct_minus_wrong_ci95_hi']):+6.3f}] | "
            f"{float(r['mean_mlp_pair_deficit_correct_minus_wrong']):+7.3f}"
            f"[{float(r['mlp_pair_deficit_correct_minus_wrong_ci95_lo']):+6.3f},"
            f"{float(r['mlp_pair_deficit_correct_minus_wrong_ci95_hi']):+6.3f}]"
        )


def print_premlp_summary(rows):
    print("\n" + "=" * 160)
    print("PRE-MLP MATCHED: DOES MLP STILL DIVERGE AFTER CONTROLLING ITS INPUT STATE?")
    print("=" * 160)
    print(
        "layer N preMLPgap | MLP GTdef foilEx otherEx pairDef[95%CI] positiveRate"
    )
    for r in rows:
        print(
            f"L{int(r['layer']):02d} {int(r['n_pairs']):3d} "
            f"{float(r['mean_premlp_pair_gap_wrong_minus_correct']):+8.3f} | "
            f"{float(r['mean_mlp_gt_deficit_correct_minus_wrong']):+7.3f} "
            f"{float(r['mean_mlp_foil_excess_wrong_minus_correct']):+7.3f} "
            f"{float(r['mean_mlp_max_other_excess']):+7.3f} "
            f"{float(r['mean_mlp_pair_deficit_correct_minus_wrong']):+7.3f}"
            f"[{float(r['mlp_pair_deficit_correct_minus_wrong_ci95_lo']):+6.3f},"
            f"{float(r['mlp_pair_deficit_correct_minus_wrong_ci95_hi']):+6.3f}] "
            f"{float(r['mlp_pair_deficit_correct_minus_wrong_positive_rate']):.3f}"
        )


def print_causes(rows):
    print("\n" + "=" * 170)
    print("PROVISIONAL CAUSE CLASSIFICATION")
    print("=" * 170)
    print(
        "layer classification                              "
        "AttnPairDef  MLPpairDef(preMLPmatched)  "
        "AttnGTdef AttnFoilEx | MLPGTdef MLPFoilEx"
    )
    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['classification']):<43s} "
            f"{float(r['input_matched_attention_pair_deficit']):+8.3f} "
            f"{float(r['premlp_matched_mlp_pair_deficit']):+10.3f} "
            f"{float(r['attention_gt_deficit']):+8.3f} "
            f"{float(r['attention_foil_excess']):+8.3f} | "
            f"{float(r['mlp_gt_deficit_after_premlp_match']):+8.3f} "
            f"{float(r['mlp_foil_excess_after_premlp_match']):+8.3f}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_direction_assets(Path(args.direction_dir))

    selected_layers, selection_audit = select_problem_layers(
        Path(args.failure_dir),
        args.layers,
        assets["n_layers"],
        args.min_role_gap,
    )
    write_csv(
        out_dir / "selected_problem_layers.csv",
        selection_audit,
    )
    print("[problem layers]", selected_layers)

    # Build initial matches BEFORE loading the model.
    input_matches = build_block_input_matches(
        assets=assets,
        selected_layers=selected_layers,
        eval_split=args.eval_split,
        max_wrong=args.max_wrong,
        seed=args.seed,
        with_replacement=args.match_with_replacement,
    )
    if not input_matches:
        raise RuntimeError("No input-matched pairs created.")

    write_csv(
        out_dir / "block_input_matches.csv",
        input_matches,
    )

    needed_sids = (
        {int(r["wrong_sid"]) for r in input_matches}
        | {int(r["correct_sid"]) for r in input_matches}
    )
    print(
        f"[matching] pairs={len(input_matches)}, "
        f"unique samples to capture={len(needed_sids)}"
    )

    # Dataset records.
    records, _audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    records_by_sid = {
        int(r.sid): r
        for r in records
        if int(r.sid) in needed_sids
    }

    missing = sorted(needed_sids - set(records_by_sid))
    if missing:
        raise RuntimeError(
            f"{len(missing)} matched SIDs missing dataset records; "
            f"first few={missing[:10]}"
        )

    # Model.
    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)

    kw: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id} on {args.device}")
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, layer_path = resolve_decoder_layers(model)
    if len(decoder_layers) != assets["n_layers"]:
        raise RuntimeError(
            f"Layer mismatch model={len(decoder_layers)} "
            f"direction_cache={assets['n_layers']}"
        )
    print(f"[decoder] {layer_path}")

    stage_cache, capture_errors = capture_needed_samples(
        match_rows=input_matches,
        records_by_sid=records_by_sid,
        model=model,
        processor=processor,
        device=device,
        decoder_layers=decoder_layers,
        selected_layers=selected_layers,
        prompt_template=args.prompt_template,
        out_dir=out_dir,
        save_every=args.save_every,
    )

    # First analysis: match block input -> inspect Attention and MLP.
    pair_rows = build_input_matched_pair_rows(
        matches=input_matches,
        stage_cache=stage_cache,
        assets=assets,
    )
    write_csv(
        out_dir / "block_input_matched_decomposition.csv",
        pair_rows,
    )

    input_summary = summarize_input_matched(
        pair_rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    write_csv(
        out_dir / "block_input_matched_summary.csv",
        input_summary,
    )

    # Second analysis: rematch on actual pre-MLP state -> inspect MLP only.
    premlp_matches = build_premlp_matches(
        input_matches=input_matches,
        stage_cache=stage_cache,
        assets=assets,
        with_replacement=args.match_with_replacement,
    )
    write_csv(
        out_dir / "pre_mlp_matches.csv",
        premlp_matches,
    )

    premlp_rows = build_premlp_matched_mlp_rows(
        matches=premlp_matches,
        stage_cache=stage_cache,
        assets=assets,
    )
    write_csv(
        out_dir / "pre_mlp_matched_mlp_decomposition.csv",
        premlp_rows,
    )

    premlp_summary = summarize_premlp_matched(
        premlp_rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    write_csv(
        out_dir / "pre_mlp_matched_mlp_summary.csv",
        premlp_summary,
    )

    causes = classify_causes(
        input_summary,
        premlp_summary,
        selected_layers,
    )
    write_csv(
        out_dir / "cause_summary.csv",
        causes,
    )

    print_input_summary(input_summary)
    print_premlp_summary(premlp_summary)
    print_causes(causes)

    meta = {
        "experiment":
            "matched within-block diagnostic across automatically selected "
            "problematic spatial-update layers",
        "model": args.model,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "selected_layers": selected_layers,
        "auto_min_role_gap": args.min_role_gap,
        "n_initial_matches": len(input_matches),
        "n_captured_samples": len(stage_cache),
        "n_capture_errors": len(capture_errors),
        "correctness_definition":
            "cached actual model.generate() correct/wrong",
        "input_matching_features":
            "4 Direction scores + vector norm at block input (layer-1 basis)",
        "pre_mlp_matching_features":
            "4 raw Direction-prototype projections + norm of actual "
            "image-minus-noimage pre-MLP normalized state",
        "attention_cause_test":
            "Attention pair deficit remains positive with bootstrap CI > 0 "
            "after block-input matching",
        "mlp_cause_test":
            "MLP pair deficit remains positive with bootstrap CI > 0 "
            "after pre-MLP matching",
        "warning":
            "This controls measured Direction-space state, not the full hidden "
            "state; therefore classification is provisional until causal "
            "component patching validates it.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "selected_problem_layers.csv",
        "block_input_matches.csv",
        "stage_cache.npz",
        "block_input_matched_decomposition.csv",
        "block_input_matched_summary.csv",
        "pre_mlp_matches.csv",
        "pre_mlp_matched_mlp_decomposition.csv",
        "pre_mlp_matched_mlp_summary.csv",
        "cause_summary.csv",
        "capture_errors.json",
        "summary.json",
    ]:
        print(" ", out_dir / name)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
