#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-sample Dynamic Spatial-Direction Repair (No Gradients)
==========================================================

Research question
-----------------
Different wrong samples may fail in different layers and against different
competing spatial relations.  Instead of using one global repair rule, diagnose
and repair every sample dynamically, layer by layer, using ONLY the learned
spatial Direction vectors.

No gradients are used anywhere.

For each active layer l
-----------------------
We use the real image-grounded subject-reference relation vector:

    r_pre  = (h_sub - h_ref)^img_pre  - (h_sub - h_ref)^noimg_pre
    r_post = (h_sub - h_ref)^img_post - (h_sub - h_ref)^noimg_post

The image-grounded Attention update is therefore:

    delta_r_attn = r_post - r_pre

At the CURRENT layer and CURRENT modified trajectory, the Direction codebook
selects the strongest non-GT competitor:

    foil_l = argmax_{r != GT} score(r_post, r)

A pure spatial GT-vs-foil axis is then constructed from the TRAIN Direction
prototypes and projected into the TRAIN 2-D spatial subspace:

    d_l(GT, foil) =
        unit(P_spatial [mu_GT - mu_foil])

The layer's directional spatial contribution is:

    u_l = delta_r_attn dot d_l(GT, foil)

Interpretation:

    u_l > 0:
        Attention pushes the relation representation toward GT relative to
        the strongest local competitor.

    u_l < 0:
        Attention pushes it in the wrong spatial direction.

Repair modes
------------
1) nonnegative
   Only repair genuinely harmful layers:

       target = 0
       if u_l < 0:
           delta = (-u_l) * d_l
       else:
           no edit

   This is the cleanest "keep every layer's spatial update non-negative" test.

2) correct_q25
   Use generation-correct TRAIN samples to define a conservative positive
   floor.  For each layer/GT/foil, estimate the 25th percentile of correct
   u_l values:

       target = max(0, Q25_correct)

   If current u_l is below this normal lower bound:

       delta = (target - u_l) * d_l

   This does NOT force the sample to the correct median.  It only moves an
   abnormal layer back to the lower edge of the correct distribution.

3) correct_q10
   Same idea but even more conservative: max(0, Q10_correct).

4) random_control
   Uses the exact same trigger and norm as correct_q25, but redirects the edit
   into a random direction orthogonal to the learned 2-D spatial subspace.

Dynamic / sample-specific behavior
----------------------------------
- Every sample chooses its own strongest competitor at every layer.
- Every sample chooses its own layers to repair.
- A layer is untouched if its current spatial update is already healthy.
- Earlier repairs change the trajectory; later layers are diagnosed again on
  the modified trajectory.
- No final generated wrong label is needed to choose the local foil.
- GT is used, so this is an ORACLE mechanistic validation experiment.

Primary experiment
------------------
Start with ALL cached generation-wrong test samples.  Run fresh baseline
model.generate(), then run each dynamic repair mode and compare:

    generation accuracy
    W -> C
    C -> W
    net
    number of repaired layers per sample

You can later repeat on generation-correct samples or all samples with
--sample-group correct / all.

Required existing files
-----------------------
--direction-dir:
    vectors.npz
    sample_split_and_generation.csv

--prepost-dir:
    train_stage_vectors.npz

The train_stage_vectors.npz must contain pre_attn and post_attn vectors for all
layers requested by --layers.

Recommended first run
---------------------
CUDA_VISIBLE_DEVICES=0 python eval_per_sample_direction_repair_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 14-19 \
  --sample-group wrong \
  --modes nonnegative,correct_q10,correct_q25,random_control \
  --output-dir output/qwen7b_per_sample_direction_repair_v1 \
  --overwrite

Smoke
-----
CUDA_VISIBLE_DEVICES=0 python eval_per_sample_direction_repair_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 14,16,17,19 \
  --sample-group wrong \
  --modes nonnegative,correct_q25,random_control \
  --max-samples 12 \
  --output-dir output/qwen7b_per_sample_direction_repair_smoke \
  --overwrite

Outputs
-------
correct_direction_update_targets.csv
    TRAIN correct distributions of spatial update margin.

generation_per_sample.csv
    Baseline and final generation for every sample/mode.

layer_diagnostics.csv
    Every sample, every layer, every dynamic decision:
        local foil
        pre/post spatial margin
        current update margin u_l
        correct Q10/Q25/median
        triggered?
        edit norm
        post-repair expected update margin

generation_summary.csv
    Main generation result.

layer_summary.csv
    Which layers are most frequently diagnosed/repaired for baseline-wrong
    samples.

repair_count_summary.csv
    Accuracy / W->C as a function of number of repaired layers.

Important caveat
----------------
This is an oracle validation because GT is known.  Its purpose is not yet a
deployable repair method.  It tests whether spatial Direction vectors can
diagnose sample-specific harmful layer updates and whether minimally removing
those harmful spatial components changes actual generation.
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
# CLI / generic I/O
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
        default="14-19",
        help=(
            "Layers to diagnose sequentially. train_stage_vectors.npz must "
            "contain pre_attn/post_attn vectors for every requested layer."
        ),
    )
    p.add_argument(
        "--sample-group",
        default="wrong",
        choices=["wrong", "correct", "all"],
        help="Cached generation group used to select samples.",
    )
    p.add_argument(
        "--modes",
        default="nonnegative,correct_q10,correct_q25,random_control",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--min-controls-pair",
        type=int,
        default=3,
        help=(
            "Minimum TRAIN correct controls for a layer/GT/local-foil pair. "
            "If fewer exist, fall back to layer/GT pooled local-foil margins."
        ),
    )
    p.add_argument(
        "--max-edit-norm",
        type=float,
        default=0.0,
        help="Optional per-layer cap; <=0 disables.",
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


def safe_median(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(np.median(vals)) if len(vals) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_words(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_layers(text: str, n_layers: int) -> List[int]:
    vals = []
    for piece in parse_words(text):
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            vals.extend(range(a, b + step, step))
        else:
            vals.append(int(piece))

    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No layers selected.")

    bad = [x for x in vals if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(
            f"Invalid layers {bad}; valid range is 0..{n_layers - 1}"
        )
    return vals


# =============================================================================
# Metadata
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
# Model / prompt helpers
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
            if len(layers) > 0 and hasattr(layers[0], "self_attn"):
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
# TRAIN Direction codebooks and healthy-update distributions
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


def project_spatial(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    v64 = np.asarray(v, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)
    return (B64 @ (B64.T @ v64)).astype(np.float32)


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
            raise RuntimeError(f"No TRAIN examples for relation={rel}")
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


def spatial_pair_axis(cb, gt: str, foil: str) -> np.ndarray:
    raw = (
        np.asarray(cb["means"][gt], dtype=np.float32)
        - np.asarray(cb["means"][foil], dtype=np.float32)
    )
    sp = project_spatial(raw, cb["basis"])
    d = unit(sp)
    if float(np.linalg.norm(d)) <= EPS:
        raise RuntimeError(
            f"Degenerate spatial pair axis for {gt} vs {foil}"
        )
    return d


def direction_scores(
    residual_vec: np.ndarray,
    cb,
):
    q = (
        np.asarray(residual_vec, dtype=np.float32)
        - np.asarray(cb["center"], dtype=np.float32)
    )
    return {
        rel: float(q @ cb["protos"][rel])
        for rel in RELATIONS
    }


def strongest_non_gt(scores, gt):
    return max(
        [r for r in RELATIONS if r != gt],
        key=lambda r: scores[r],
    )


def load_codebooks_and_targets(
    *,
    prepost_dir: Path,
    metadata,
    selected_layers,
    min_controls_pair,
):
    path = prepost_dir / "train_stage_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    codebooks = {}
    targets = {}
    rows = []

    with np.load(path, allow_pickle=True) as z:
        train_sids = np.asarray(z["sid"], dtype=np.int64)
        labels = np.asarray(
            [norm_relation(x) for x in z["relation"]],
            dtype=object,
        )

        correct_mask = np.asarray(
            [
                metadata["generation"].get(
                    int(sid), {}
                ).get("generation_group", "")
                == "correct"
                for sid in train_sids.tolist()
            ],
            dtype=bool,
        )

        if not np.any(correct_mask):
            raise RuntimeError(
                "No generation-correct TRAIN controls found."
            )

        for li in selected_layers:
            kpre = f"L{li}_pre_attn"
            kpost = f"L{li}_post_attn"

            if kpre not in z.files or kpost not in z.files:
                raise KeyError(
                    f"{path} missing {kpre}/{kpost}. "
                    f"Rerun pre/post Direction collection with L{li}."
                )

            Xpre = np.asarray(z[kpre], dtype=np.float32)
            Xpost = np.asarray(z[kpost], dtype=np.float32)

            cb = fit_post_codebook(Xpost, labels)
            codebooks[li] = cb

            delta = Xpost - Xpre

            # First collect correct-sample local foil and update margin using
            # exactly the same rule used at runtime.
            per_gt_all = defaultdict(list)
            per_pair = defaultdict(list)

            for i in range(len(train_sids)):
                if not bool(correct_mask[i]):
                    continue

                gt = str(labels[i])
                post_scores = direction_scores(Xpost[i], cb)
                foil = strongest_non_gt(post_scores, gt)
                d = spatial_pair_axis(cb, gt, foil)
                u = float(delta[i] @ d)

                per_gt_all[gt].append(u)
                per_pair[(gt, foil)].append(u)

            for gt in RELATIONS:
                pooled = np.asarray(
                    per_gt_all.get(gt, []),
                    dtype=np.float64,
                )
                if len(pooled) == 0:
                    raise RuntimeError(
                        f"No correct TRAIN healthy updates for L{li} GT={gt}"
                    )

                for foil in RELATIONS:
                    if foil == gt:
                        continue

                    pair = np.asarray(
                        per_pair.get((gt, foil), []),
                        dtype=np.float64,
                    )

                    use_pair = len(pair) >= min_controls_pair
                    vals = pair if use_pair else pooled
                    source = "pair" if use_pair else "gt_pooled"

                    q10 = float(np.quantile(vals, 0.10))
                    q25 = float(np.quantile(vals, 0.25))
                    med = float(np.median(vals))

                    targets[(li, gt, foil)] = {
                        "q10": q10,
                        "q25": q25,
                        "median": med,
                        "n_pair": int(len(pair)),
                        "n_used": int(len(vals)),
                        "source": source,
                    }

                    rows.append({
                        "layer": li,
                        "gt": gt,
                        "foil": foil,
                        "n_pair_controls": int(len(pair)),
                        "n_used_controls": int(len(vals)),
                        "target_source": source,
                        "q10_update_margin": q10,
                        "q25_update_margin": q25,
                        "median_update_margin": med,
                        "nonnegative_floor": 0.0,
                    })

    return codebooks, targets, rows


# =============================================================================
# No-image captures
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

            def make_pre(layer_id):
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

            def make_post(layer_id):
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
                    make_pre(li)
                )
            )

            if not hasattr(block, "post_attention_layernorm"):
                raise RuntimeError(
                    f"L{li} has no post_attention_layernorm."
                )

            self.handles.append(
                block.post_attention_layernorm.register_forward_pre_hook(
                    make_post(li)
                )
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
        li for li in selected_layers
        if li not in cap.pre or li not in cap.post
    ]
    if missing:
        raise RuntimeError(
            f"Missing no-image captures for layers {missing}"
        )

    out = {}
    for li in selected_layers:
        out[li] = {
            "pre": cap.pre[li],
            "post": cap.post[li],
            "attn_delta":
                cap.post[li] - cap.pre[li],
        }

    del batch
    return out


# =============================================================================
# Dynamic per-sample repair
# =============================================================================

def clip_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float32)
    if max_norm <= 0:
        return d

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
):
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

    raise RuntimeError("Failed to create orthogonal random control.")


class DynamicDirectionRepair:
    """
    Diagnose each layer from the CURRENT trajectory and repair only if the
    spatial GT-vs-local-foil Attention update margin is below the chosen floor.
    """

    def __init__(
        self,
        *,
        decoder_layers,
        selected_layers,
        subj_pos,
        ref_pos,
        noimg_pairs,
        codebooks,
        targets,
        gt,
        mode,
        max_edit_norm,
        sid,
        seed,
    ):
        self.decoder_layers = decoder_layers
        self.selected_layers = list(map(int, selected_layers))
        self.subj_pos = list(map(int, subj_pos))
        self.ref_pos = list(map(int, ref_pos))
        self.noimg_pairs = noimg_pairs
        self.codebooks = codebooks
        self.targets = targets
        self.gt = gt
        self.mode = mode
        self.max_edit_norm = float(max_edit_norm)
        self.sid = int(sid)
        self.seed = int(seed)

        self.current_pre_real = {}
        self.applied = set()
        self.logs = []
        self.handles = []

        for li in self.selected_layers:
            block = decoder_layers[li]

            self.handles.append(
                block.register_forward_pre_hook(
                    self._make_block_pre_hook(li)
                )
            )
            self.handles.append(
                block.self_attn.register_forward_hook(
                    self._make_attn_hook(li)
                )
            )

    def _make_block_pre_hook(self, li):
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

            self.current_pre_real[li] = (
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

    def _make_attn_hook(self, li):
        def hook(_module, _args, output):
            if li in self.applied:
                return output

            x = first_tensor(output)
            if x.ndim != 3:
                return output

            seq = x[0]
            if int(seq.shape[0]) <= max(
                self.subj_pos + self.ref_pos
            ):
                return output

            if li not in self.current_pre_real:
                return output

            pre_real = self.current_pre_real[li]

            attn_pair_real = (
                pair_diff(
                    seq,
                    self.subj_pos,
                    self.ref_pos,
                )
                .detach().float().cpu().numpy()
                .astype(np.float32)
            )

            # Current modified real post-attention pair state.
            post_real = pre_real + attn_pair_real

            # Image-grounded relation states.
            r_pre = (
                pre_real
                - self.noimg_pairs[li]["pre"]
            )
            r_post = (
                post_real
                - self.noimg_pairs[li]["post"]
            )

            # Image-grounded Attention update.
            delta_r = r_post - r_pre

            cb = self.codebooks[li]

            post_scores = direction_scores(
                r_post,
                cb,
            )
            foil = strongest_non_gt(
                post_scores,
                self.gt,
            )

            d = spatial_pair_axis(
                cb,
                self.gt,
                foil,
            )

            pre_pair_margin = float(
                (
                    np.asarray(r_pre, dtype=np.float32)
                    - cb["center"]
                ) @ d
            )
            post_pair_margin = float(
                (
                    np.asarray(r_post, dtype=np.float32)
                    - cb["center"]
                ) @ d
            )

            current_u = float(
                np.asarray(delta_r, dtype=np.float32) @ d
            )

            t = self.targets[
                (li, self.gt, foil)
            ]

            if self.mode == "nonnegative":
                target_u = 0.0

            elif self.mode == "correct_q10":
                target_u = max(
                    0.0,
                    float(t["q10"]),
                )

            elif self.mode in (
                "correct_q25",
                "random_control",
            ):
                target_u = max(
                    0.0,
                    float(t["q25"]),
                )

            else:
                raise ValueError(
                    f"Unknown mode: {self.mode}"
                )

            requested = max(
                0.0,
                target_u - current_u,
            )

            targeted_delta = (
                requested * d
            ).astype(np.float32)
            targeted_delta = clip_delta(
                targeted_delta,
                self.max_edit_norm,
            )

            if self.mode == "random_control":
                delta = random_orthogonal_delta(
                    float(np.linalg.norm(targeted_delta)),
                    cb["basis"],
                    len(targeted_delta),
                    seed=(
                        self.seed
                        + self.sid * 100003
                        + li * 1009
                    ),
                )
            else:
                delta = targeted_delta

            delta_norm = float(np.linalg.norm(delta))
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

            achieved_u = float(
                np.asarray(delta, dtype=np.float32) @ d
            )

            self.logs.append({
                "sid": self.sid,
                "layer": li,
                "mode": self.mode,
                "gt": self.gt,
                "local_foil": foil,

                "pre_pair_margin": pre_pair_margin,
                "post_pair_margin_before":
                    post_pair_margin,

                "current_update_margin":
                    current_u,

                "correct_q10":
                    float(t["q10"]),
                "correct_q25":
                    float(t["q25"]),
                "correct_median":
                    float(t["median"]),
                "target_source":
                    str(t["source"]),
                "n_target_controls":
                    int(t["n_used"]),

                "target_update_margin":
                    target_u,
                "requested_correction":
                    requested,

                "edit_norm":
                    delta_norm,
                "achieved_spatial_correction":
                    achieved_u,

                "expected_update_margin_after":
                    current_u + achieved_u,

                "triggered":
                    int(delta_norm > EPS),
                "was_harmful":
                    int(current_u < 0.0),
                "was_below_q10":
                    int(current_u < max(0.0, float(t["q10"]))),
                "was_below_q25":
                    int(current_u < max(0.0, float(t["q25"]))),
            })

            self.applied.add(li)
            return replace_first_tensor(
                output,
                y,
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
# Sample selection
# =============================================================================

def select_eval_sids(
    metadata,
    records,
    split,
    sample_group,
    max_samples,
    seed,
):
    sids = []

    for sid in metadata["sids"]:
        if (
            split != "all"
            and metadata["split"].get(sid, "") != split
        ):
            continue

        if sid not in records:
            continue

        gt = metadata["gt"].get(sid, "")
        if gt not in REL2ID:
            continue

        group = metadata["generation"].get(
            sid, {}
        ).get("generation_group", "")

        if (
            sample_group != "all"
            and group != sample_group
        ):
            continue

        sids.append(sid)

    if max_samples is None or len(sids) <= max_samples:
        return sorted(sids)

    rng = random.Random(seed)
    rng.shuffle(sids)
    return sorted(sids[:max_samples])


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    *,
    model,
    processor,
    decoder_layers,
    selected_layers,
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
    generation_rows = []
    layer_rows = []
    errors = []

    for i, sid in enumerate(
        tqdm(
            eval_sids,
            desc="per-sample Direction repair",
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

            noimg_pairs = capture_noimage_pairs(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                rec=rec,
                device=device,
                prompt_template=prompt_template,
            )

            baseline_text, baseline_pred = generate_answer(
                model,
                processor,
                real_batch,
                max_new_tokens,
            )
            baseline_correct = int(
                baseline_pred == gt
            )

            cached = metadata["generation"].get(
                sid, {}
            )

            generation_rows.append({
                "sid": sid,
                "mode": "baseline",
                "gt": gt,
                "cached_group":
                    cached.get("generation_group", ""),
                "cached_pred":
                    cached.get("generation_pred", ""),

                "baseline_text":
                    baseline_text,
                "baseline_pred":
                    baseline_pred or "",
                "baseline_correct":
                    baseline_correct,

                "edited_text":
                    baseline_text,
                "edited_pred":
                    baseline_pred or "",
                "edited_correct":
                    baseline_correct,

                "n_layers_scanned":
                    len(selected_layers),
                "n_layers_triggered":
                    0,
                "triggered_layers": "",

                "W2C": 0,
                "C2W": 0,
            })

            for mode in modes:
                with DynamicDirectionRepair(
                    decoder_layers=decoder_layers,
                    selected_layers=selected_layers,
                    subj_pos=sp,
                    ref_pos=rp,
                    noimg_pairs=noimg_pairs,
                    codebooks=codebooks,
                    targets=targets,
                    gt=gt,
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

                edited_correct = int(
                    edited_pred == gt
                )

                triggered_layers = [
                    int(r["layer"])
                    for r in repair.logs
                    if int(r["triggered"]) == 1
                ]

                generation_rows.append({
                    "sid": sid,
                    "mode": mode,
                    "gt": gt,
                    "cached_group":
                        cached.get("generation_group", ""),
                    "cached_pred":
                        cached.get("generation_pred", ""),

                    "baseline_text":
                        baseline_text,
                    "baseline_pred":
                        baseline_pred or "",
                    "baseline_correct":
                        baseline_correct,

                    "edited_text":
                        edited_text,
                    "edited_pred":
                        edited_pred or "",
                    "edited_correct":
                        edited_correct,

                    "n_layers_scanned":
                        len(selected_layers),
                    "n_layers_triggered":
                        len(triggered_layers),
                    "triggered_layers":
                        ",".join(
                            str(x) for x in triggered_layers
                        ),

                    "W2C": int(
                        baseline_correct == 0
                        and edited_correct == 1
                    ),
                    "C2W": int(
                        baseline_correct == 1
                        and edited_correct == 0
                    ),
                })

                for row in repair.logs:
                    rr = dict(row)
                    rr.update({
                        "baseline_pred":
                            baseline_pred or "",
                        "baseline_correct":
                            baseline_correct,
                        "edited_pred":
                            edited_pred or "",
                        "edited_correct":
                            edited_correct,
                    })
                    layer_rows.append(rr)

            del real_batch

            if save_every > 0 and i % save_every == 0:
                write_csv(
                    out_dir / "generation_per_sample.csv",
                    generation_rows,
                )
                write_csv(
                    out_dir / "layer_diagnostics.csv",
                    layer_rows,
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
        generation_rows,
    )
    write_csv(
        out_dir / "layer_diagnostics.csv",
        layer_rows,
    )
    write_csv(
        out_dir / "errors.csv",
        errors,
    )

    return generation_rows, layer_rows, errors


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(rows):
    buckets = defaultdict(list)

    for r in rows:
        if r["mode"] == "baseline":
            continue
        buckets[str(r["mode"])].append(r)

    out = []

    for mode, rr in sorted(buckets.items()):
        bacc = safe_mean(
            r["baseline_correct"] for r in rr
        )
        eacc = safe_mean(
            r["edited_correct"] for r in rr
        )

        nw = int(
            sum(
                int(r["baseline_correct"]) == 0
                for r in rr
            )
        )
        nc = int(
            sum(
                int(r["baseline_correct"]) == 1
                for r in rr
            )
        )

        W2C = int(sum(int(r["W2C"]) for r in rr))
        C2W = int(sum(int(r["C2W"]) for r in rr))

        out.append({
            "mode": mode,
            "n": len(rr),
            "baseline_acc": bacc,
            "edited_acc": eacc,
            "acc_gain": eacc - bacc,

            "mean_layers_triggered": safe_mean(
                r["n_layers_triggered"] for r in rr
            ),
            "zero_trigger_rate": safe_frac(
                int(r["n_layers_triggered"]) == 0
                for r in rr
            ),

            "n_baseline_wrong": nw,
            "W2C": W2C,
            "W2C_over_wrong":
                W2C / nw if nw else float("nan"),

            "n_baseline_correct": nc,
            "C2W": C2W,
            "C2W_over_correct":
                C2W / nc if nc else float("nan"),

            "net": W2C - C2W,
        })

    return sorted(
        out,
        key=lambda r: (
            float(r["acc_gain"]),
            int(r["net"]),
        ),
        reverse=True,
    )


def summarize_layers(rows):
    buckets = defaultdict(list)

    for r in rows:
        key = (
            str(r["mode"]),
            int(r["layer"]),
        )
        buckets[key].append(r)

    out = []

    for (mode, li), rr in sorted(buckets.items()):
        out.append({
            "mode": mode,
            "layer": li,
            "n": len(rr),

            "trigger_rate": safe_frac(
                int(r["triggered"]) == 1
                for r in rr
            ),
            "harmful_rate": safe_frac(
                int(r["was_harmful"]) == 1
                for r in rr
            ),
            "below_q10_rate": safe_frac(
                int(r["was_below_q10"]) == 1
                for r in rr
            ),
            "below_q25_rate": safe_frac(
                int(r["was_below_q25"]) == 1
                for r in rr
            ),

            "mean_current_update_margin": safe_mean(
                r["current_update_margin"] for r in rr
            ),
            "median_current_update_margin": safe_median(
                r["current_update_margin"] for r in rr
            ),
            "mean_target_update_margin": safe_mean(
                r["target_update_margin"] for r in rr
            ),

            "mean_edit_norm": safe_mean(
                r["edit_norm"] for r in rr
            ),

            "local_foil_left_rate": safe_frac(
                r["local_foil"] == "left"
                for r in rr
            ),
            "local_foil_right_rate": safe_frac(
                r["local_foil"] == "right"
                for r in rr
            ),
            "local_foil_above_rate": safe_frac(
                r["local_foil"] == "above"
                for r in rr
            ),
            "local_foil_below_rate": safe_frac(
                r["local_foil"] == "below"
                for r in rr
            ),
        })

    return out


def summarize_repair_counts(generation_rows):
    buckets = defaultdict(list)

    for r in generation_rows:
        if r["mode"] == "baseline":
            continue

        key = (
            str(r["mode"]),
            int(r["n_layers_triggered"]),
        )
        buckets[key].append(r)

    out = []

    for (mode, ntrig), rr in sorted(buckets.items()):
        out.append({
            "mode": mode,
            "n_layers_triggered": ntrig,
            "n_samples": len(rr),
            "baseline_acc": safe_mean(
                r["baseline_correct"] for r in rr
            ),
            "edited_acc": safe_mean(
                r["edited_correct"] for r in rr
            ),
            "W2C": int(
                sum(int(r["W2C"]) for r in rr)
            ),
            "C2W": int(
                sum(int(r["C2W"]) for r in rr)
            ),
        })

    return out


def print_generation_summary(rows):
    print("\n" + "=" * 136)
    print("ACTUAL model.generate() — PER-SAMPLE DIRECTION REPAIR")
    print("=" * 136)
    print(
        "mode N | acc base->edit gain | mean repaired layers | "
        "zeroRepair | W2C/wrong C2W/correct net"
    )

    for r in rows:
        print(
            f"{str(r['mode']):18s} "
            f"{int(r['n']):3d} | "
            f"{float(r['baseline_acc']):.4f}->"
            f"{float(r['edited_acc']):.4f} "
            f"{float(r['acc_gain']):+7.4f} | "
            f"{float(r['mean_layers_triggered']):6.3f} | "
            f"{float(r['zero_trigger_rate']):.3f} | "
            f"{int(r['W2C']):3d}/"
            f"{float(r['W2C_over_wrong']):.3f} "
            f"{int(r['C2W']):3d}/"
            f"{float(r['C2W_over_correct']):.3f} "
            f"{int(r['net']):+4d}"
        )


def print_layer_summary(rows):
    print("\n" + "=" * 146)
    print("DIRECTION-DIAGNOSED LAYER FAILURES")
    print("=" * 146)
    print(
        "mode layer N | trigger harmful belowQ10 belowQ25 | "
        "update mean/median target | editNorm"
    )

    for r in rows:
        print(
            f"{str(r['mode']):18s} "
            f"L{int(r['layer']):02d} "
            f"{int(r['n']):3d} | "
            f"{float(r['trigger_rate']):.3f} "
            f"{float(r['harmful_rate']):.3f} "
            f"{float(r['below_q10_rate']):.3f} "
            f"{float(r['below_q25_rate']):.3f} | "
            f"{float(r['mean_current_update_margin']):+7.3f}/"
            f"{float(r['median_current_update_margin']):+7.3f} "
            f"{float(r['mean_target_update_margin']):+7.3f} | "
            f"{float(r['mean_edit_norm']):6.3f}"
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
        f"[decoder] {layer_path}; selected layers={selected_layers}"
    )

    modes = parse_words(args.modes)
    valid_modes = {
        "nonnegative",
        "correct_q10",
        "correct_q25",
        "random_control",
    }
    bad = [m for m in modes if m not in valid_modes]
    if bad:
        raise ValueError(f"Unknown modes: {bad}")

    codebooks, targets, target_rows = load_codebooks_and_targets(
        prepost_dir=Path(args.prepost_dir),
        metadata=metadata,
        selected_layers=selected_layers,
        min_controls_pair=args.min_controls_pair,
    )

    write_csv(
        out_dir / "correct_direction_update_targets.csv",
        target_rows,
    )

    eval_sids = select_eval_sids(
        metadata,
        records,
        args.eval_split,
        args.sample_group,
        args.max_samples,
        args.seed,
    )

    print(
        f"[eval] sample_group={args.sample_group}, "
        f"N={len(eval_sids)}"
    )

    generation_rows, layer_rows, errors = run_experiment(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        selected_layers=selected_layers,
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
        generation_rows
    )
    layer_summary = summarize_layers(
        layer_rows
    )
    repair_count_summary = summarize_repair_counts(
        generation_rows
    )

    write_csv(
        out_dir / "generation_summary.csv",
        generation_summary,
    )
    write_csv(
        out_dir / "layer_summary.csv",
        layer_summary,
    )
    write_csv(
        out_dir / "repair_count_summary.csv",
        repair_count_summary,
    )

    print_generation_summary(
        generation_summary
    )
    print_layer_summary(
        layer_summary
    )

    summary = {
        "experiment":
            "per-sample dynamic Direction-only spatial repair",

        "sample_group":
            args.sample_group,

        "selected_layers":
            selected_layers,

        "modes":
            modes,

        "diagnosis":
            "At every layer, select the strongest current non-GT relation "
            "from the Direction codebook, then measure the image-grounded "
            "Attention update along the pure spatial GT-vs-local-foil axis.",

        "nonnegative_mode":
            "Only cancel a negative spatial update margin.",

        "correct_q10_mode":
            "Raise update margin only to the lower 10th-percentile boundary "
            "of generation-correct TRAIN controls.",

        "correct_q25_mode":
            "Raise update margin only to the lower 25th-percentile boundary "
            "of generation-correct TRAIN controls.",

        "gradient_used":
            False,

        "gt_oracle":
            True,

        "dynamic":
            True,

        "primary_metric":
            "fresh model.generate() W2C/C2W",

        "n_samples":
            len(eval_sids),

        "n_errors":
            len(errors),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "correct_direction_update_targets.csv",
        "generation_per_sample.csv",
        "layer_diagnostics.csv",
        "generation_summary.csv",
        "layer_summary.csv",
        "repair_count_summary.csv",
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
