#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spatial_to_causaltoken_mediation_v1.py

Path-level mediation test:

    upstream spatial state
        -> L25 high-leverage text states
        -> final matching relation logit

This combines the two earlier findings:
  (1) L25 mediation-ranked text states have strong behavioral leverage.
  (2) fixed spatial edits at L20-L25 causally change the corresponding final logit.

The key intervention here is NOT zeroing the full L25 token state.
Instead, for an upstream spatial edit, define the spatial-induced L25 change

    Delta_sp[p] = h_L25^spatial-edit[p] - h_L25^clean[p].

We then route ONLY this spatial-induced change at L25.

Conditions
==========
full
    Keep the ordinary spatial-edited forward pass.

block_top
    At L25, restore Top-K causal-token positions to CLEAN:
        h_edit[p] <- h_clean[p], p in Top-K
    This removes only the spatial-induced update at Top-K.

block_random
    Same operation at category-matched random eligible text positions.

keep_top
    At L25, keep the spatial-induced update only at Top-K eligible text states;
    restore every OTHER eligible text state to CLEAN.
    Visual states and prompt-last are untouched.

keep_random
    Same keep-only routing for matched random positions.

Thus block_top asks whether the spatial effect NEEDS to pass through Top-K,
while keep_top asks how much of the spatial effect can be PRESERVED when,
among eligible text states, only Top-K retain their edit-induced change.

Causal-token ranking
====================
Exactly the earlier single-layer score at receiver layer R (default L25):

    M_p^(r) =
      (h_real[R,p] - h_gray[R,p])^T dJ_r/dh_real[R,p]

where J_r is the same late-writer objective over target layers (default L31-35).
By default only M>0 states are eligible for Top-K ranking.

For this first mediation test, each sample uses its GT relation r.
This directly matches the previous zero/keep causal-token experiment and keeps
the test economical. Across the full COCO set all four relations are covered.

Spatial edit
============
Uses the EXACT Synthetic-400 Real-Gray geometry and SpatialPatch operator from

    eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py

At source layer S < receiver layer R:

    toward left  = -strength * d_H
    toward right = +strength * d_H
    toward under = -strength * d_V
    toward on    = +strength * d_V

and away is the sign reversal.

Primary endpoint
================
Only the sample GT relation's own final relation logit is read.

For every sample / source layer / condition:

    D = logit_r(toward r) - logit_r(away from r)

No opposite-relation logit is used.

Mediation pattern of interest
=============================
If Top-K states mediate spatial control:

    D_block_top    << D_full
    D_block_random ~= D_full

and, as a stronger restricted-sufficiency test:

    D_keep_top     retains a substantial fraction of D_full
    D_keep_random  retains much less.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u \
  eval_spatial_to_causaltoken_mediation_v1.py \
  --source-layers 22,23,24 \
  --receiver-layer 25 \
  --k 7 \
  --strengths 5 \
  --eval-max-samples 80 \
  --conditions full,block_top,block_random,keep_top,keep_random \
  --random-repeats 1 \
  --output-dir output/qwen3b_spatial_to_L25_top7_mediation_n80_v1 \
  --overwrite

You can reuse the geometry cache from the previous full-440 signed-logit run:

  --geometry-cache \
  output/qwen3b_spatial_matching_logit_L20_26_all440_s1_s5_v2/geometry_cache/synthetic400_residual_real_minus_gray.npz

If you have the learned_writers.npz from the prior L25 zero/keep run, also pass:

  --writer-npz /path/to/learned_writers.npz

Otherwise writers are recalibrated with the same 30% protocol.

Outputs
=======
per_sample_effects.csv
    One row per sample/source/strength/condition/repeat with toward, away, D.

mediation_summary.csv
    Mean D, SEM, sign rate, retention/attenuation relative to full.

mediation_by_relation.csv
    Same summaries split by left/right/on/under.

selection_per_sample.csv
    Top-K/random selection counts and token identities.

writer_geometry.csv / learned_writers.npz
    Written only if writers are recalibrated.

synthetic400_HV_geometry.csv
analysis_summary.txt
metadata.json
errors.jsonl
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn

try:
    import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6 as ctrl
except Exception as exc:
    raise SystemExit(
        "Could not import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py.\n"
        "Run from the AdaptVis llava16 repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12
SCRIPT_VERSION = "spatial-to-causaltoken-mediation-v1"


# =============================================================================
# CLI / generic helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--coco-prompt-jsonl",
        "--prompt-jsonl",
        dest="coco_prompt_jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    # Synthetic geometry args expected by v6/head-scan utilities.
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)
    p.add_argument("--target-max-samples", type=int, default=None)
    p.add_argument("--control", default="gray", choices=["gray"])
    p.add_argument("--geometry-control", default="gray", choices=["gray"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--revision", default="main")
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument(
        "--internvl-use-thumbnail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Path test.
    p.add_argument(
        "--source-layers",
        default="22,23,24",
        help="Upstream spatial-edit layers. Must all be < receiver-layer.",
    )
    p.add_argument("--receiver-layer", type=int, default=25)
    p.add_argument("--strengths", default="5")
    p.add_argument("--k", type=int, default=7)
    p.add_argument(
        "--conditions",
        default="full,block_top,block_random,keep_top,keep_random",
    )
    p.add_argument("--random-repeats", type=int, default=1)

    # Causal-token ranking.
    p.add_argument(
        "--target-layers",
        default="31,32,33,34,35",
        help="Late writer layers used in J_r.",
    )
    p.add_argument(
        "--writer-npz",
        default="",
        help="Optional learned_writers.npz from the prior causal-token run.",
    )
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument(
        "--positive-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--eligible-categories",
        default="subject,reference,relation_words,other_text",
    )

    # Evaluation.
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--eval-scope",
        default="all_data",
        choices=["all_data", "test"],
    )
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--eval-seed", type=int, default=17)
    p.add_argument("--seed", type=int, default=17)

    # Geometry cache can contain a superset of requested source layers.
    p.add_argument(
        "--geometry-cache",
        default="",
        help=(
            "Optional v6-compatible Synthetic residual geometry NPZ. "
            "A cache containing a superset of --source-layers is accepted."
        ),
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_floats(text: str) -> List[float]:
    vals = sorted(set(float(x) for x in str(text).replace(" ", "").split(",") if x))
    if not vals or any(x <= 0 for x in vals):
        raise ValueError("--strengths must contain positive values")
    return vals


def parse_names(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def normalize_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v.copy()
    return (v / n).astype(np.float32)


def sem(values: Sequence[float]) -> float:
    a = np.asarray(list(values), dtype=np.float64)
    if len(a) <= 1:
        return 0.0
    return float(a.std(ddof=1) / math.sqrt(len(a)))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# Data / model / writers
# =============================================================================

def load_data(a: argparse.Namespace):
    prompt_path = Path(a.coco_prompt_jsonl)
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(prompt_path)
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta: List[dict] = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "relation": gt,  # v6-compatible key
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    if int(a.max_samples) > 0:
        meta = traj.stratified_cap(meta, int(a.max_samples), int(a.seed))

    train, heldout = traj.stratified_split(
        meta, float(a.train_ratio), int(a.seed)
    )

    if a.eval_scope == "all_data":
        test = list(meta)
    else:
        test = list(heldout)

    if int(a.eval_max_samples) > 0:
        test = traj.stratified_cap(
            test, int(a.eval_max_samples), int(a.eval_seed)
        )

    return two, meta, train, heldout, test, rec_by_sid


def load_model(a: argparse.Namespace, two):
    spec = base.merged_model_specs(two)[a.model]
    cls = getattr(transformers, spec.model_class)
    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    print(f"[MODEL] loading {spec.repo_id}", flush=True)
    try:
        model = cls.from_pretrained(spec.repo_id, **kw)
    except TypeError:
        kw["torch_dtype"] = kw.pop("dtype")
        model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)

    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    return model, processor, decoder_layers, decoder_path, spec


def writer_keys(layer: int, rel: str) -> List[str]:
    surface = DISPLAY[rel]
    return [
        f"L{layer}_{surface}",
        f"L{layer}_{rel}",
        f"{layer}_{surface}",
        f"{layer}_{rel}",
    ]


def load_writer_npz(path: Path, targets: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    z = np.load(path)
    writers: Dict[int, Dict[str, np.ndarray]] = {
        int(T): {} for T in targets
    }
    for T in targets:
        for r in REL:
            found = None
            for key in writer_keys(int(T), r):
                if key in z:
                    found = np.asarray(z[key], dtype=np.float32)
                    break
            if found is None:
                raise RuntimeError(
                    f"{path} missing writer L{T}/{r}; tried {writer_keys(int(T), r)}"
                )
            writers[int(T)][r] = found
    return writers


def calibrate_writers(
    *,
    model,
    processor,
    decoder_layers,
    train,
    rec_by_sid,
    targets,
    device,
    gray_value,
    writer_mode,
):
    q_by_sid: Dict[int, Dict[int, np.ndarray]] = {}

    for m in tqdm(train, desc="CALIBRATE late writers"):
        sid = int(m["sid"])
        real = gray = None
        try:
            real = base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")
            gray = dyn.make_gray_image(real, gray_value)

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

            hr = dyn.capture_cpu(model, decoder_layers, rb, targets)
            hg = dyn.capture_cpu(model, decoder_layers, gb, targets)

            q_by_sid[sid] = {
                int(T): (hr[int(T)][0, -1] - hg[int(T)][0, -1]).astype(np.float32)
                for T in targets
            }
        finally:
            if real is not None:
                with contextlib.suppress(Exception):
                    real.close()
            if gray is not None:
                with contextlib.suppress(Exception):
                    gray.close()

    writers, geometry = dyn.learn_writers(
        train, q_by_sid, targets, writer_mode
    )
    return writers, geometry


# =============================================================================
# Exact v6 geometry
# =============================================================================

def load_geometry_cache_subset(
    path: Path,
    requested_layers: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        cache_layers = [int(x) for x in z["decoder_block_index"].tolist()]
        X = np.asarray(z["relation_vectors"], dtype=np.float32)
        y = np.asarray(
            [ctrl.norm_rel(x) for x in z["relation"].tolist()],
            dtype=object,
        )

    missing = [L for L in requested_layers if int(L) not in cache_layers]
    if missing:
        raise RuntimeError(
            f"Geometry cache {path} layers={cache_layers}; missing {missing}"
        )
    idx = [cache_layers.index(int(L)) for L in requested_layers]
    return X[:, idx, :], y


def relation_axis_coords(
    geom_L: Mapping[str, Any],
    relation: str,
) -> np.ndarray:
    """
    Express semantic +/-d_H or +/-d_V exactly in v6's orthonormal Q coordinates.
    """
    Q = np.asarray(geom_L["control_basis"], dtype=np.float64)
    B = np.asarray(geom_L["B"], dtype=np.float64)
    dH = B[:, 0]
    dV = B[:, 1]

    if relation == "left":
        dz = -dH
    elif relation == "right":
        dz = +dH
    elif relation == "below":
        dz = -dV
    elif relation == "above":
        dz = +dV
    else:
        raise ValueError(relation)

    c = Q.T @ dz
    recon = Q @ c
    err = float(np.linalg.norm(recon - dz))
    if err > 1e-5:
        raise RuntimeError(
            f"v6 semantic-axis reconstruction error {relation}: {err:.3e}"
        )
    return c.astype(np.float64)


# =============================================================================
# Causal-token ranking at receiver layer
# =============================================================================

def causal_rows_for_gt(
    *,
    sid: int,
    gt: str,
    receiver: int,
    ids: Sequence[int],
    cats: Sequence[str],
    toks: Sequence[str],
    real_states: Mapping[int, np.ndarray],
    gray_states: Mapping[int, np.ndarray],
    grad: np.ndarray,
    eligible_categories: Sequence[str],
) -> List[dict]:
    wanted = set(map(str, eligible_categories))

    H = real_states[receiver][0].astype(np.float32)
    Hg = gray_states[receiver][0].astype(np.float32)
    G = grad[0].astype(np.float32)

    npos = min(
        len(ids), len(cats), len(toks),
        H.shape[0], Hg.shape[0], G.shape[0],
    )

    rows: List[dict] = []
    for p in range(max(0, npos - 1)):  # prompt-last excluded
        broad = dyn.broad_category(cats[p])
        if broad in {"visual", "last"}:
            continue
        if wanted and broad not in wanted:
            continue

        delta_rg = (H[p] - Hg[p]).astype(np.float32)
        gp = G[p].astype(np.float32)
        score = float(np.dot(delta_rg, gp))

        rows.append({
            "sid": int(sid),
            "gt": DISPLAY[gt],
            "position": int(p),
            "token_id": int(ids[p]),
            "token": str(toks[p]).replace("\n", "\\n"),
            "category": str(cats[p]),
            "broad_category": str(broad),
            "mediation": score,
            "grad_norm": float(np.linalg.norm(gp)),
            "real_gray_norm": float(np.linalg.norm(delta_rg)),
        })

    return rows


def rank_positive(rows: Sequence[Mapping[str, Any]], positive_only: bool) -> List[dict]:
    z = [dict(r) for r in rows]
    if positive_only:
        z = [r for r in z if float(r["mediation"]) > 0]
    z.sort(key=lambda r: float(r["mediation"]), reverse=True)
    for i, r in enumerate(z, 1):
        r["rank"] = int(i)
    return z


def category_matched_random(
    *,
    all_rows: Sequence[Mapping[str, Any]],
    top_rows: Sequence[Mapping[str, Any]],
    n_select: int,
    seed: int,
) -> List[dict]:
    refs = list(top_rows[: int(n_select)])
    if not refs:
        return []

    excluded = {int(r["position"]) for r in refs}
    candidates = [
        dict(r) for r in all_rows
        if int(r["position"]) not in excluded
    ]

    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in candidates:
        by_cat[str(r["broad_category"])].append(r)

    rng = random.Random(int(seed))
    chosen: List[dict] = []
    used = set()

    for ref in refs:
        cat = str(ref["broad_category"])
        pool = [
            r for r in by_cat.get(cat, [])
            if int(r["position"]) not in used
        ]
        if not pool:
            pool = [
                r for r in candidates
                if int(r["position"]) not in used
            ]
        if not pool:
            break
        r = rng.choice(pool)
        chosen.append(r)
        used.add(int(r["position"]))

    return chosen


# =============================================================================
# Receiver routing: manipulate ONLY the spatial-induced L25 change
# =============================================================================

class ReceiverSpatialDeltaRouter:
    """
    At receiver-layer output, selectively restore positions to the CLEAN state.

    block:
        restore selected positions only.

    keep:
        restore eligible_positions - selected_positions.

    This removes only:
        h_spatial_edit - h_clean
    at the restored positions. It does not zero the full hidden state.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        receiver_layer: int,
        clean_state: np.ndarray,
        eligible_positions: Sequence[int],
        selected_positions: Sequence[int],
        mode: str,
    ):
        self.receiver_layer = int(receiver_layer)
        self.clean_state = np.asarray(clean_state, dtype=np.float32)
        self.eligible = set(map(int, eligible_positions))
        self.selected = set(map(int, selected_positions))
        self.mode = str(mode)
        if self.mode not in {"block", "keep"}:
            raise ValueError(self.mode)

        self.n_restored = 0
        self.handle = decoder_layers[self.receiver_layer].register_forward_hook(
            self._hook
        )

    def _hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        if int(h.shape[0]) != 1:
            raise RuntimeError(f"Expected batch size 1, got {tuple(h.shape)}")

        # We only score prompt prefill, so sequence length should match exactly.
        if int(h.shape[1]) != int(self.clean_state.shape[0]):
            return None

        if self.mode == "block":
            restore = self.selected
        else:
            restore = self.eligible - self.selected

        if not restore:
            self.n_restored = 0
            return None

        clean = torch.as_tensor(
            self.clean_state,
            device=h.device,
            dtype=h.dtype,
        )
        y = h.clone()

        count = 0
        for p in sorted(restore):
            if 0 <= int(p) < int(y.shape[1]):
                y[0, int(p), :] = clean[int(p)]
                count += 1

        self.n_restored = count
        return traj.replace_first_tensor(out, y)

    def close(self):
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def own_relation_logit(
    *,
    model,
    decoder_layers,
    prepared,
    relation_token_ids: Mapping[str, Sequence[int]],
    relation: str,
    source_layer: int,
    geom: Mapping[int, Any],
    source_coords: np.ndarray,
    receiver_layer: int,
    clean_receiver_state: np.ndarray,
    eligible_positions: Sequence[int],
    selected_positions: Sequence[int],
    route_mode: Optional[str],
) -> Tuple[float, int]:
    """
    One forward with upstream spatial edit plus optional receiver routing.
    """
    dev = prepared.batch["input_ids"].device
    c = torch.as_tensor(
        np.asarray(source_coords, dtype=np.float32).reshape(1, 2),
        device=dev,
        dtype=torch.float32,
    )

    router = None
    try:
        with ctrl.SpatialPatch(
            decoder_layers=decoder_layers,
            layers=[int(source_layer)],
            geom=geom,
            sub_pos=prepared.sub_pos,
            ref_pos=prepared.ref_pos,
            coords=c,
        ):
            if route_mode is not None:
                router = ReceiverSpatialDeltaRouter(
                    decoder_layers=decoder_layers,
                    receiver_layer=int(receiver_layer),
                    clean_state=clean_receiver_state,
                    eligible_positions=eligible_positions,
                    selected_positions=selected_positions,
                    mode=route_mode,
                )

            with torch.inference_mode():
                out = model(
                    **prepared.batch,
                    use_cache=False,
                    return_dict=True,
                )

            last = out.logits[0, -1].float()
            score = torch.stack(
                [last[int(tok)] for tok in relation_token_ids[relation]]
            ).max()
            val = float(score.detach().item())

            restored = int(router.n_restored) if router is not None else 0
            del out, last, score
            return val, restored
    finally:
        if router is not None:
            router.close()


def clean_own_relation_logit(
    *,
    model,
    prepared,
    relation_token_ids: Mapping[str, Sequence[int]],
    relation: str,
) -> float:
    with torch.inference_mode():
        out = model(
            **prepared.batch,
            use_cache=False,
            return_dict=True,
        )
    last = out.logits[0, -1].float()
    score = torch.stack(
        [last[int(tok)] for tok in relation_token_ids[relation]]
    ).max()
    val = float(score.detach().item())
    del out, last, score
    return val


# =============================================================================
# Summaries
# =============================================================================

def summarize_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    group_cols = ["source_layer", "strength", "condition"]
    for (L, strength, cond), g in df.groupby(group_cols, sort=True):
        d = g["signed_effect"].to_numpy(dtype=float)
        rows.append({
            "source_layer": int(L),
            "strength": float(strength),
            "condition": str(cond),
            "N_rows": int(len(g)),
            "N_samples": int(g["sid"].nunique()),
            "mean_signed_effect": float(d.mean()),
            "sem_signed_effect": sem(d),
            "median_signed_effect": float(np.median(d)),
            "positive_effect_rate": float(np.mean(d > 0)),
            "mean_selected": float(g["n_selected"].mean()),
            "mean_eligible": float(g["n_eligible"].mean()),
            "mean_restored_toward": float(g["n_restored_toward"].mean()),
            "mean_restored_away": float(g["n_restored_away"].mean()),
        })

    out = pd.DataFrame(rows)

    # Compare each condition to full at the aggregate level.
    ret: List[dict] = []
    for (L, strength), g in out.groupby(
        ["source_layer", "strength"], sort=True
    ):
        full = g[g["condition"] == "full"]
        full_mean = (
            float(full.iloc[0]["mean_signed_effect"])
            if len(full) else float("nan")
        )
        for _, r in g.iterrows():
            mean = float(r["mean_signed_effect"])
            if math.isfinite(full_mean) and abs(full_mean) > 1e-12:
                retention = mean / full_mean
                attenuation = 1.0 - retention
            else:
                retention = float("nan")
                attenuation = float("nan")

            rr = dict(r)
            rr["full_mean_signed_effect"] = full_mean
            rr["retention_vs_full"] = retention
            rr["attenuation_vs_full"] = attenuation
            ret.append(rr)

    return pd.DataFrame(ret)


def summarize_by_relation(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (L, strength, cond, gt), g in df.groupby(
        ["source_layer", "strength", "condition", "gt"],
        sort=True,
    ):
        d = g["signed_effect"].to_numpy(dtype=float)
        rows.append({
            "source_layer": int(L),
            "strength": float(strength),
            "condition": str(cond),
            "gt": str(gt),
            "N_rows": int(len(g)),
            "N_samples": int(g["sid"].nunique()),
            "mean_signed_effect": float(d.mean()),
            "sem_signed_effect": sem(d),
            "positive_effect_rate": float(np.mean(d > 0)),
        })
    return pd.DataFrame(rows)


def report_text(summary: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("=" * 150)
    lines.append(
        "SPATIAL -> L25 CAUSAL-TOKEN MEDIATION | D = own-logit(toward) - own-logit(away)"
    )
    lines.append("=" * 150)

    order = [
        "full",
        "block_top",
        "block_random",
        "keep_top",
        "keep_random",
    ]

    for strength in sorted(summary["strength"].unique()):
        lines.append(f"[strength={strength:g}]")
        sg = summary[summary["strength"] == strength]
        for L in sorted(sg["source_layer"].unique()):
            lines.append(f"L{int(L):02d} -> receiver L25")
            lg = sg[sg["source_layer"] == L]
            by = {str(r["condition"]): r for _, r in lg.iterrows()}
            for cond in order:
                if cond not in by:
                    continue
                r = by[cond]
                lines.append(
                    f"  {cond:12s} "
                    f"D={float(r['mean_signed_effect']):+.6f} "
                    f"SEM={float(r['sem_signed_effect']):.6f} "
                    f"P(D>0)={float(r['positive_effect_rate']):.3f} "
                    f"retain={float(r['retention_vs_full']):+.3f} "
                    f"atten={float(r['attenuation_vs_full']):+.3f}"
                )
            lines.append("")

    lines.append(
        "Desired mediation pattern: block_top attenuates D much more than "
        "block_random; keep_top retains much more D than keep_random."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    a = parse_args()

    # Compatibility alias for utilities that expect args.prompt_jsonl.
    a.prompt_jsonl = a.coco_prompt_jsonl

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    receiver = int(a.receiver_layer)
    targets = parse_ints(a.target_layers)
    strengths = parse_floats(a.strengths)
    conditions = parse_names(a.conditions)
    eligible_categories = parse_names(a.eligible_categories)

    valid_conditions = {
        "full",
        "block_top",
        "block_random",
        "keep_top",
        "keep_random",
    }
    bad = set(conditions) - valid_conditions
    if bad:
        raise ValueError(f"Unknown conditions: {sorted(bad)}")
    if "full" not in conditions:
        raise ValueError("--conditions must include full")
    if any(int(L) >= receiver for L in source_layers):
        raise ValueError(
            f"Every source layer must be strictly upstream of receiver L{receiver}; "
            f"got {source_layers}"
        )
    if receiver < 1:
        raise ValueError("--receiver-layer must be >=1")
    if int(a.k) < 1:
        raise ValueError("--k must be >=1")
    if int(a.random_repeats) < 1:
        raise ValueError("--random-repeats must be >=1")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    two, meta, train, heldout, test, rec_by_sid = load_data(a)

    model = processor = None
    effect_rows: List[dict] = []
    selection_rows: List[dict] = []

    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        needed_layers = (
            source_layers
            + [receiver - 1, receiver]
            + targets
        )
        for L in needed_layers:
            if not 0 <= int(L) < n_layers:
                raise ValueError(
                    f"L{L} invalid; model has L0..L{n_layers - 1}"
                )

        # -------------------------------------------------------------
        # 1) Late writers: same ranking scaffold as prior zero/keep test.
        # -------------------------------------------------------------
        if a.writer_npz:
            writers = load_writer_npz(Path(a.writer_npz), targets)
            writer_source = str(Path(a.writer_npz))
        else:
            writers, writer_geom = calibrate_writers(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                train=train,
                rec_by_sid=rec_by_sid,
                targets=targets,
                device=device,
                gray_value=a.gray_value,
                writer_mode=a.writer_mode,
            )
            writer_source = "recalibrated"
            pd.DataFrame(writer_geom).to_csv(
                outdir / "writer_geometry.csv",
                index=False,
            )
            np.savez_compressed(
                outdir / "learned_writers.npz",
                **{
                    f"L{T}_{DISPLAY[r]}": writers[T][r]
                    for T in targets
                    for r in REL
                },
            )

        # -------------------------------------------------------------
        # 2) Exact v6 Synthetic-400 geometry.
        # -------------------------------------------------------------
        if a.geometry_cache:
            geom_cache_path = Path(a.geometry_cache)
            Xg, yg = load_geometry_cache_subset(
                geom_cache_path, source_layers
            )
            geometry_source = str(geom_cache_path)
            print(
                f"[GEOMETRY CACHE] subset reuse {geom_cache_path} "
                f"-> layers={source_layers} X={Xg.shape}"
            )
        else:
            source_rows = ctrl.hs.load_synthetic(a)
            geom_cache_path = (
                outdir
                / "geometry_cache"
                / "synthetic400_residual_real_minus_gray.npz"
            )
            Xg, yg = ctrl.build_synthetic_geometry_cache(
                model_alias=a.model,
                model=model,
                processor_or_backend=processor,
                decoder_layers=decoder_layers,
                layers=source_layers,
                source_rows=source_rows,
                cache_path=geom_cache_path,
                args=a,
            )
            geometry_source = str(geom_cache_path)

        geom, axis_df = ctrl.fit_geometry(
            Xg, yg, source_layers
        )
        axis_df.to_csv(
            outdir / "synthetic400_HV_geometry.csv",
            index=False,
        )

        # Same final relation-logit convention as recovery v6.
        relation_token_ids, token_details = ctrl.encode_fast_relation_tokens(
            processor.tokenizer, "coco"
        )
        if relation_token_ids is None:
            raise RuntimeError(
                "COCO relations are not all available through v6's fast "
                "single-token logit path:\n"
                + json.dumps(token_details, ensure_ascii=False, indent=2)
            )

        cal_sids = {int(x["sid"]) for x in train}
        eval_sids = {int(x["sid"]) for x in test}
        overlap = len(cal_sids & eval_sids)

        print("\n" + "=" * 154)
        print("SPATIAL -> CAUSAL-TOKEN -> FINAL LOGIT MEDIATION")
        print("=" * 154)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(f"source spatial layers={source_layers}")
        print(f"receiver causal-token layer=L{receiver}")
        print(f"late writer targets={targets}")
        print(f"writer source={writer_source}")
        print(f"geometry source={geometry_source}")
        print(f"K={a.k} positive_only={a.positive_only}")
        print(f"strengths={strengths}")
        print(f"conditions={conditions} random_repeats={a.random_repeats}")
        print(
            f"eval_scope={a.eval_scope} N={len(test)} "
            f"writer-cal/eval overlap={overlap}"
        )
        print(
            "Receiver intervention restores ONLY spatial-induced L25 changes "
            "to the clean L25 state; no full-state zeroing."
        )
        print(
            "Primary endpoint D = GT-logit(toward GT) - GT-logit(away GT)."
        )
        print("=" * 154 + "\n")

        # -------------------------------------------------------------
        # 3) Evaluation.
        # -------------------------------------------------------------
        for m in tqdm(test, desc="Spatial->TopK mediation"):
            sid = int(m["sid"])
            gt = str(m["gt"])
            real = gray = prepared = cap = None

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real, a.gray_value)

                # Use v6 preparation for the real prompt so source spatial patch
                # and final relation-logit readout are exactly recovery-compatible.
                prepared = ctrl.prepare_standard(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    row=m,
                    image=real,
                    args=a,
                )
                rb = prepared.batch

                # Gray prompt with the same question/chat template.
                gb = ctrl.recv.build_batch(
                    probe=ctrl.hs.headprobe,
                    processor=processor,
                    question=str(m["question_text"]),
                    image=gray,
                    device=device,
                )

                real_ids = rb["input_ids"][0].detach().cpu().tolist()
                gray_ids = gb["input_ids"][0].detach().cpu().tolist()
                if real_ids != gray_ids:
                    raise RuntimeError(
                        "Real/gray prompt token IDs differ; cannot align "
                        "Real-Gray causal ranking."
                    )

                ids = real_ids
                cats, toks = dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                # Clean receiver state (plus previous layer only for alignment
                # diagnostics if needed later).
                real_states = dyn.capture_cpu(
                    model,
                    decoder_layers,
                    rb,
                    [receiver - 1, receiver],
                )
                gray_states = dyn.capture_cpu(
                    model,
                    decoder_layers,
                    gb,
                    [receiver],
                )
                clean_receiver_state = real_states[receiver][0].astype(
                    np.float32
                )

                # -----------------------------------------------------
                # Same GT-conditioned causal-token score used before.
                # -----------------------------------------------------
                graph_layers = sorted(set([receiver] + targets))
                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model,
                        decoder_layers,
                        rb,
                        graph_layers,
                        receiver,
                    )

                    objective_terms = []
                    for T in targets:
                        s_hat = torch.as_tensor(
                            normalize_np(writers[T][gt]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        objective_terms.append(
                            torch.dot(
                                cap.states[T][0, -1].float(),
                                s_hat,
                            )
                        )
                    objective = torch.stack(objective_terms).sum()

                    grad = torch.autograd.grad(
                        objective,
                        cap.states[receiver],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                    grad_np = (
                        grad.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                cap.close()
                cap = None

                all_rows = causal_rows_for_gt(
                    sid=sid,
                    gt=gt,
                    receiver=receiver,
                    ids=ids,
                    cats=cats,
                    toks=toks,
                    real_states=real_states,
                    gray_states=gray_states,
                    grad=grad_np,
                    eligible_categories=eligible_categories,
                )
                ranked = rank_positive(
                    all_rows,
                    positive_only=bool(a.positive_only),
                )
                top_rows = ranked[: int(a.k)]
                top_positions = [int(r["position"]) for r in top_rows]
                eligible_positions = [int(r["position"]) for r in all_rows]

                if not top_positions:
                    raise RuntimeError(
                        f"sid={sid}: no positive causal-token state at L{receiver}"
                    )

                selection_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "receiver_layer": receiver,
                    "selection": "top",
                    "repeat": 0,
                    "n_eligible": len(eligible_positions),
                    "n_positive": len(ranked),
                    "n_selected": len(top_positions),
                    "positions": "|".join(map(str, top_positions)),
                    "tokens": "|".join(str(r["token"]) for r in top_rows),
                    "categories": "|".join(
                        str(r["broad_category"]) for r in top_rows
                    ),
                    "scores": "|".join(
                        f"{float(r['mediation']):.8g}" for r in top_rows
                    ),
                })

                random_by_repeat: Dict[int, List[dict]] = {}
                for rep in range(int(a.random_repeats)):
                    rr = category_matched_random(
                        all_rows=all_rows,
                        top_rows=top_rows,
                        n_select=len(top_positions),
                        seed=(
                            int(a.seed) * 1000003
                            + sid * 1009
                            + rep * 9176
                        ),
                    )
                    random_by_repeat[rep] = rr
                    selection_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "receiver_layer": receiver,
                        "selection": "random",
                        "repeat": rep,
                        "n_eligible": len(eligible_positions),
                        "n_positive": len(ranked),
                        "n_selected": len(rr),
                        "positions": "|".join(
                            str(int(r["position"])) for r in rr
                        ),
                        "tokens": "|".join(str(r["token"]) for r in rr),
                        "categories": "|".join(
                            str(r["broad_category"]) for r in rr
                        ),
                        "scores": "|".join(
                            f"{float(r['mediation']):.8g}" for r in rr
                        ),
                    })

                clean_logit = clean_own_relation_logit(
                    model=model,
                    prepared=prepared,
                    relation_token_ids=relation_token_ids,
                    relation=gt,
                )

                # -----------------------------------------------------
                # Upstream signed edit, routed at L25.
                # -----------------------------------------------------
                for source_L in source_layers:
                    unit_coord = relation_axis_coords(
                        geom[source_L], gt
                    )

                    for strength in strengths:
                        toward_coord = (
                            float(strength) * unit_coord
                        )
                        away_coord = -toward_coord

                        # Full has no random-repeat dependence.
                        condition_specs: List[Tuple[str, int, List[int], Optional[str]]] = []
                        if "full" in conditions:
                            condition_specs.append(
                                ("full", 0, [], None)
                            )
                        if "block_top" in conditions:
                            condition_specs.append(
                                ("block_top", 0, top_positions, "block")
                            )
                        if "keep_top" in conditions:
                            condition_specs.append(
                                ("keep_top", 0, top_positions, "keep")
                            )

                        for rep in range(int(a.random_repeats)):
                            random_positions = [
                                int(r["position"])
                                for r in random_by_repeat[rep]
                            ]
                            if "block_random" in conditions:
                                condition_specs.append(
                                    (
                                        "block_random",
                                        rep,
                                        random_positions,
                                        "block",
                                    )
                                )
                            if "keep_random" in conditions:
                                condition_specs.append(
                                    (
                                        "keep_random",
                                        rep,
                                        random_positions,
                                        "keep",
                                    )
                                )

                        for cond, rep, selected, route_mode in condition_specs:
                            toward_logit, restored_t = own_relation_logit(
                                model=model,
                                decoder_layers=decoder_layers,
                                prepared=prepared,
                                relation_token_ids=relation_token_ids,
                                relation=gt,
                                source_layer=source_L,
                                geom=geom,
                                source_coords=toward_coord,
                                receiver_layer=receiver,
                                clean_receiver_state=clean_receiver_state,
                                eligible_positions=eligible_positions,
                                selected_positions=selected,
                                route_mode=route_mode,
                            )
                            away_logit, restored_a = own_relation_logit(
                                model=model,
                                decoder_layers=decoder_layers,
                                prepared=prepared,
                                relation_token_ids=relation_token_ids,
                                relation=gt,
                                source_layer=source_L,
                                geom=geom,
                                source_coords=away_coord,
                                receiver_layer=receiver,
                                clean_receiver_state=clean_receiver_state,
                                eligible_positions=eligible_positions,
                                selected_positions=selected,
                                route_mode=route_mode,
                            )

                            effect_rows.append({
                                "sid": sid,
                                "gt": DISPLAY[gt],
                                "source_layer": int(source_L),
                                "receiver_layer": receiver,
                                "strength": float(strength),
                                "condition": cond,
                                "repeat": int(rep),
                                "clean_logit": float(clean_logit),
                                "toward_logit": float(toward_logit),
                                "away_logit": float(away_logit),
                                "delta_toward_vs_clean": float(
                                    toward_logit - clean_logit
                                ),
                                "delta_away_vs_clean": float(
                                    away_logit - clean_logit
                                ),
                                "signed_effect": float(
                                    toward_logit - away_logit
                                ),
                                "n_eligible": len(eligible_positions),
                                "n_positive": len(ranked),
                                "requested_k": int(a.k),
                                "n_selected": (
                                    len(top_positions)
                                    if cond in {"full", "block_top", "keep_top"}
                                    else len(selected)
                                ),
                                "n_restored_toward": int(restored_t),
                                "n_restored_away": int(restored_a),
                            })

            except Exception as exc:
                ctrl.append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-80:],
                    },
                )
                tqdm.write(
                    f"[ERROR sid={sid}] {type(exc).__name__}: {exc}"
                )
                if a.fail_fast:
                    raise
            finally:
                if cap is not None:
                    with contextlib.suppress(Exception):
                        cap.close()
                if prepared is not None:
                    prepared.close()
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not effect_rows:
            raise RuntimeError("No successful mediation rows")

        write_csv(
            outdir / "per_sample_effects.csv",
            effect_rows,
        )
        write_csv(
            outdir / "selection_per_sample.csv",
            selection_rows,
        )

        df = pd.DataFrame(effect_rows)
        summary = summarize_effects(df)
        by_relation = summarize_by_relation(df)

        summary.to_csv(
            outdir / "mediation_summary.csv",
            index=False,
        )
        by_relation.to_csv(
            outdir / "mediation_by_relation.csv",
            index=False,
        )

        report = report_text(summary)
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(
            report,
            encoding="utf-8",
        )

        meta_out = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "source_layers": source_layers,
            "receiver_layer": receiver,
            "target_writer_layers": targets,
            "writer_source": writer_source,
            "writer_mode": a.writer_mode,
            "causal_token_score": (
                "(h_real_receiver-h_gray_receiver)^T grad J_GT"
            ),
            "positive_only": bool(a.positive_only),
            "k": int(a.k),
            "eligible_categories": eligible_categories,
            "spatial_geometry_source": geometry_source,
            "spatial_geometry": (
                "exact v6 Synthetic-400 Real-Gray H/V subspace"
            ),
            "spatial_patch_operator": (
                "exact v6 SpatialPatch: sub += dz/2, ref -= dz/2"
            ),
            "strengths": strengths,
            "conditions": conditions,
            "random_repeats": int(a.random_repeats),
            "receiver_routing": (
                "restore selected L25 positions from spatial-edited state "
                "to the corresponding clean L25 hidden state"
            ),
            "keep_scope": (
                "eligible text states only; visual and prompt-last are not restored"
            ),
            "primary_endpoint": (
                "GT own-relation logit(toward GT) - "
                "GT own-relation logit(away from GT)"
            ),
            "uses_opposite_relation_logit": False,
            "eval_scope": a.eval_scope,
            "eval_N": len(test),
            "writer_cal_eval_overlap": overlap,
            "relation_token_ids": relation_token_ids,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[SAVED] {outdir / 'per_sample_effects.csv'}")
        print(f"[SAVED] {outdir / 'mediation_summary.csv'}")
        print(f"[SAVED] {outdir / 'mediation_by_relation.csv'}")
        print(f"[SAVED] {outdir / 'selection_per_sample.csv'}")

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
