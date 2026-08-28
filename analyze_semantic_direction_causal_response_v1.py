
# -*- coding: utf-8 -*-

"""
All-layer Semantic Direction -> Causal Response scan.

This experiment changes how the existing Direction vectors are USED.

Instead of assuming:
    "Direction information is decodable at layer L"
therefore
    "the model causally uses that information at layer L",

we explicitly measure TWO quantities at every decoder layer:

(1) Representation availability R
---------------------------------
For GT relation g and competitor c:

    d_l(g,c) = unit(mu_g,l - mu_c,l)

where mu are TRAIN-derived Direction prototypes from the existing
Image-NoImage subject-reference residual vectors.

For a sample:

    q_l = r_l - center_l
    R_l(g,c) = q_l dot d_l(g,c)

Positive R means the cached spatial representation favors GT relative to c.

(2) Causal utilization C
-------------------------
Let output margin be:

    M_out(g,c) = logit_g - logit_c

At layer l, imagine a pair-preserving infinitesimal intervention:

    h_sub <- h_sub + eps/2 * d_l(g,c)
    h_ref <- h_ref - eps/2 * d_l(g,c)

Then:

    C_l(g,c) = d M_out / d eps | eps=0

We compute C exactly as a directional derivative with autograd:

    C_l(g,c)
      = 0.5 * (grad_sub - grad_ref) dot d_l(g,c)

This is first-order CAUSAL SENSITIVITY of the final relation decision to the
semantic Direction axis.

Therefore R and C answer different questions:

    R high, C high : spatial information exists and downstream uses it
    R high, C low  : information exists but is weakly utilized
    R low,  C high : downstream is sensitive, but current state lacks evidence
    R < 0, C high  : wrong-side spatial state lies on a causally important axis
    C < 0          : semantic GT-oriented intervention paradoxically hurts the
                     GT-vs-competitor output margin (semantic/causal mismatch)

The script scans ALL requested layers and ALL 3 GT-vs-competitor axes per
sample. Correct/wrong groups come from cached ACTUAL model.generate() output.

Spatial causal-gradient geometry
--------------------------------
We additionally project the pair gradient into the 2-D spatial subspace
spanned by:

    right - left
    above - below

and report:
    spatial_grad_fraction
    semantic_vs_causal_alignment

This asks whether the direction that is causally important to the output is
aligned with the semantic Direction axis.

Random orthogonal control
-------------------------
For every sample/layer/axis, the same gradient is dotted with multiple random
unit directions orthogonal to the spatial subspace. This requires no extra
model forward/backward and gives a scale-matched specificity control.

Finite-difference validation
----------------------------
Gradient sensitivity is still local. After the all-layer scan, the script can
automatically select candidate layers and validate them with REAL bidirectional
interventions:

    +eps * d
    -eps * d

at the layer output, preserving the subject/reference pair mean.

For each epsilon:
    slope_FD = (M(+eps) - M(-eps)) / (2 eps)

We compare it with the autograd C and check:
    M(+eps) > M(0) > M(-eps)

A norm-matched random direction orthogonal to the spatial subspace is also
tested.

This finite-difference stage is deliberately run only on top candidate layers
and a sample subset because doing +/- interventions for every layer/sample is
expensive.

Inputs
------
--direction-dir must contain:
    vectors.npz
    sample_split_and_generation.csv

Expected vectors.npz arrays:
    sample_index
    relation
    residual        [N, L, D]

The script uses the same TRAIN split to build Direction codebooks.

Recommended full gradient scan
------------------------------
CUDA_VISIBLE_DEVICES=0 python analyze_semantic_direction_causal_response_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --split test \
  --layers all \
  --fd-layers auto \
  --fd-top-k 5 \
  --fd-max-samples 40 \
  --output-dir output/qwen7b_semantic_direction_causal_response_v1 \
  --overwrite

Smoke test
----------
CUDA_VISIBLE_DEVICES=0 python analyze_semantic_direction_causal_response_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --split test \
  --layers 12-20 \
  --max-samples 20 \
  --fd-layers auto \
  --fd-top-k 3 \
  --fd-max-samples 10 \
  --output-dir output/qwen7b_semantic_direction_causal_response_smoke \
  --overwrite

Main outputs
------------
per_sample_layer_axis.csv
    Every sample x layer x GT-vs-competitor axis.

primary_foil_layer_summary.csv
    Correct/wrong all-layer R/C map.
    Wrong primary foil = actual generated wrong relation.
    Correct primary foil = strongest non-GT first-step relation logit.

wrong_failure_map.csv
    For each generation-wrong sample/layer on its ACTUAL final wrong axis,
    compare R and C against generation-correct controls with same GT and same
    competitor axis.

wrong_failure_type_summary.csv
    representation deficit / causal-utilization deficit / both / neither.

layer_causal_map.csv
    Compact layer ranking with representation gap, causal gap, spatial
    alignment, random controls, and wrong failure-type rates.

finite_difference_per_sample.csv
finite_difference_summary.csv
    Bidirectional causal validation on selected layers.

Interpretation warning
----------------------
C is local causal sensitivity of the FIRST relation-token decision margin.
It is much stronger evidence than probe decodability, but it is not identical
to full autoregressive generation behavior. The finite-difference stage
validates non-infinitesimal intervention effects on the same first-step margin.
Use actual model.generate() only after a mechanism candidate has been found.
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
# CLI / generic helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
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
        "--split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--layers",
        default="all",
        help="all, 12-20, or 12,14,16,18.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for gradient scan.",
    )
    p.add_argument(
        "--random-controls",
        type=int,
        default=8,
        help="Random spatial-orthogonal directions per sample/layer/axis.",
    )
    p.add_argument(
        "--control-quantile",
        type=float,
        default=0.25,
        help=(
            "Wrong sample is called representation/utilization deficient when "
            "below this quantile of generation-correct controls matched by "
            "layer, GT, competitor axis."
        ),
    )
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)

    # Finite-difference validation.
    p.add_argument(
        "--fd-layers",
        default="auto",
        help=(
            "none, auto, or explicit layer list. auto selects layers with "
            "strong correct-vs-wrong causal-utilization gaps."
        ),
    )
    p.add_argument("--fd-top-k", type=int, default=5)
    p.add_argument(
        "--fd-max-samples",
        type=int,
        default=40,
        help="Maximum samples used for finite-difference validation.",
    )
    p.add_argument(
        "--fd-eps-scales",
        default="0.25,0.5,1.0",
        help=(
            "eps = scale * TRAIN std of projection on the tested semantic axis."
        ),
    )
    p.add_argument(
        "--fd-random-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

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
                fields.append(str(k))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm_relation(x: Any) -> str:
    return direction.norm_relation(x)


def safe_mean(xs: Iterable[float]) -> float:
    a = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(a.mean()) if len(a) else float("nan")


def safe_median(xs: Iterable[float]) -> float:
    a = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(np.median(a)) if len(a) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_layers(text: str, n_layers: int) -> List[int]:
    t = str(text).strip().lower()
    if t == "all":
        return list(range(n_layers))

    out = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(piece))

    out = sorted(set(out))
    bad = [x for x in out if not 0 <= x < n_layers]
    if bad:
        raise ValueError(
            f"Invalid layers {bad}; valid 0..{n_layers - 1}"
        )
    if not out:
        raise ValueError("No layers selected.")
    return out


def parse_float_list(text: str) -> List[float]:
    vals = [
        float(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]
    if not vals or any(x <= 0 for x in vals):
        raise ValueError("Need positive finite-difference scales.")
    return vals


def bootstrap_gap(
    correct_vals: Sequence[float],
    wrong_vals: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
):
    a = np.asarray(correct_vals, dtype=np.float64)
    b = np.asarray(wrong_vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")

    # positive = correct > wrong
    obs = float(a.mean() - b.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots[i] = aa.mean() - bb.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


# =============================================================================
# Token/logit helpers
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


def relation_token_variants(
    tokenizer: Any,
) -> Dict[str, List[int]]:
    out = {}
    unk = getattr(tokenizer, "unk_token_id", None)

    for rel in RELATIONS:
        ids = []
        for s in (
            rel,
            " " + rel,
            "\n" + rel,
            rel.capitalize(),
            " " + rel.capitalize(),
        ):
            xx = tokenizer_ids(tokenizer, s)
            if len(xx) != 1:
                continue
            tid = int(xx[0])
            if unk is not None and tid == int(unk):
                continue
            ids.append(tid)

        ids = list(dict.fromkeys(ids))
        if not ids:
            raise RuntimeError(
                f"No one-token variants found for relation={rel}"
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

    raise RuntimeError("Could not locate model logits.")


def relation_scores(
    score_vector: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    vals = []
    for rel in RELATIONS:
        ids = torch.as_tensor(
            token_map[rel],
            device=score_vector.device,
            dtype=torch.long,
        )
        vals.append(
            score_vector.index_select(0, ids).max()
        )
    return torch.stack(vals)


# =============================================================================
# Direction cache / train geometry
# =============================================================================

def unit_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def orthonormal_basis(vectors: Sequence[np.ndarray]) -> np.ndarray:
    A = np.stack(
        [np.asarray(v, dtype=np.float64) for v in vectors],
        axis=1,
    )  # [D,K]
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > (1e-8 * max(float(s.max()), 1.0))
    if not np.any(keep):
        raise RuntimeError("Degenerate spatial subspace.")
    return u[:, keep].astype(np.float32)


def fit_codebook(
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    center = X_train.mean(axis=0).astype(np.float32)
    Xc = X_train - center

    means = {}
    protos = {}
    for rel in RELATIONS:
        mask = y_train == rel
        if not np.any(mask):
            raise RuntimeError(f"No train examples for {rel}")
        mu = Xc[mask].mean(axis=0).astype(np.float32)
        means[rel] = mu
        protos[rel] = unit_np(mu)

    proto_arr = np.stack(
        [protos[r] for r in RELATIONS]
    ).astype(np.float32)

    # 2-D physically meaningful spatial basis.
    lr = means["right"] - means["left"]
    ud = means["above"] - means["below"]
    spatial_basis = orthonormal_basis([lr, ud])

    # Train projection scales for finite difference.
    axis_std = {}
    axis_mean = {}
    for gt in RELATIONS:
        for comp in RELATIONS:
            if comp == gt:
                continue
            d = unit_np(protos[gt] - protos[comp])
            proj = Xc @ d
            axis_std[(gt, comp)] = float(np.std(proj))
            axis_mean[(gt, comp)] = float(np.mean(proj))

    return {
        "center": center,
        "means": means,
        "protos": protos,
        "proto_arr": proto_arr,
        "spatial_basis": spatial_basis,
        "axis_std": axis_std,
        "axis_mean": axis_mean,
    }


def load_direction_assets(direction_dir: Path):
    vec_path = direction_dir / "vectors.npz"
    gen_path = direction_dir / "sample_split_and_generation.csv"

    if not vec_path.exists():
        raise FileNotFoundError(vec_path)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    with np.load(vec_path, allow_pickle=True) as z:
        arr = {k: z[k] for k in z.files}

    required = {"sample_index", "relation", "residual"}
    missing = required - set(arr)
    if missing:
        raise KeyError(
            f"{vec_path} missing arrays: {sorted(missing)}"
        )

    sids = arr["sample_index"].astype(np.int64)
    labels = np.asarray(
        [norm_relation(x) for x in arr["relation"]],
        dtype=object,
    )
    residual = np.asarray(arr["residual"], dtype=np.float32)

    if residual.ndim != 3:
        raise ValueError(
            f"Expected residual [N,L,D], got {residual.shape}"
        )

    idx_by_sid = {
        int(s): i for i, s in enumerate(sids.tolist())
    }

    split_by_sid = {}
    gen_by_sid = {}

    for r in read_csv(gen_path):
        sid = int(r["sample_index"])
        split_by_sid[sid] = str(
            r.get("split", "")
        ).strip()

        pred = norm_relation(r.get("generation_pred", ""))
        group = str(
            r.get("generation_group", "")
        ).strip().lower()

        idx = idx_by_sid.get(sid)
        gt = labels[idx] if idx is not None else ""

        if group not in ("correct", "wrong"):
            if gt in REL2ID and pred in REL2ID:
                group = "correct" if pred == gt else "wrong"

        gen_by_sid[sid] = {
            "generation_group": group,
            "generation_pred": pred,
            "generation_text": str(
                r.get("generation_text", "")
            ),
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
        raise RuntimeError("No train split in Direction cache.")

    codebooks = {}
    for li in range(residual.shape[1]):
        codebooks[li] = fit_codebook(
            residual[train_idx, li, :],
            labels[train_idx],
        )

    return {
        "sids": sids,
        "labels": labels,
        "residual": residual,
        "idx_by_sid": idx_by_sid,
        "split": split_by_sid,
        "generation": gen_by_sid,
        "codebooks": codebooks,
        "n_layers": residual.shape[1],
        "hidden_dim": residual.shape[2],
        "train_idx": train_idx,
    }


def semantic_axis(
    codebook: Mapping[str, Any],
    gt: str,
    competitor: str,
) -> np.ndarray:
    return unit_np(
        np.asarray(codebook["protos"][gt])
        - np.asarray(codebook["protos"][competitor])
    )


def representation_metrics(
    residual_vec: np.ndarray,
    codebook: Mapping[str, Any],
    gt: str,
    competitor: str,
):
    q = (
        np.asarray(residual_vec, dtype=np.float32)
        - np.asarray(codebook["center"], dtype=np.float32)
    )
    d = semantic_axis(codebook, gt, competitor)

    r_raw = float(q @ d)
    qn = float(np.linalg.norm(q))
    r_cos = (
        float(q @ d / qn)
        if qn > EPS else float("nan")
    )

    proto_scores = (
        q @ np.asarray(codebook["proto_arr"]).T
    )
    pred = RELATIONS[int(np.argmax(proto_scores))]
    order = np.sort(proto_scores)

    return {
        "semantic_axis": d,
        "R_raw": r_raw,
        "R_cos": r_cos,
        "direction_pred": pred,
        "direction_correct": int(pred == gt),
        "direction_top2_margin":
            float(order[-1] - order[-2]),
        "proto_scores": proto_scores,
    }


# =============================================================================
# Model structure / gradient capture
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
    raise RuntimeError(
        f"Could not find tensor in output type={type(x)}"
    )


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
    raise RuntimeError(
        f"Cannot replace tensor in output type={type(output)}"
    )


def pool_positions(
    x: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    pos = [
        int(p)
        for p in positions
        if 0 <= int(p) < int(x.shape[0])
    ]
    if not pos:
        raise RuntimeError("No valid object token positions.")
    idx = torch.as_tensor(
        pos,
        device=x.device,
        dtype=torch.long,
    )
    return x.index_select(0, idx).mean(dim=0)


class LayerOutputCapture:
    def __init__(self, decoder_layers, layer_ids):
        self.layer_ids = list(map(int, layer_ids))
        self.tensors: Dict[int, torch.Tensor] = {}
        self.handles = []

        for li in self.layer_ids:
            def make_hook(layer_id):
                def hook(_module, _args, output):
                    x = first_tensor(output)
                    self.tensors[layer_id] = x
                    return output
                return hook

            self.handles.append(
                decoder_layers[li].register_forward_hook(
                    make_hook(li)
                )
            )

    def validate(self):
        missing = [
            li for li in self.layer_ids
            if li not in self.tensors
        ]
        if missing:
            raise RuntimeError(
                f"Missing layer output captures: {missing}"
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
        True,
    )
    batch = direction.process_inputs(
        processor,
        rendered,
        image,
        device,
    )
    ids = [
        int(x)
        for x in batch["input_ids"][0]
        .detach().cpu().tolist()
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


def random_orthogonal_units(
    *,
    dim: int,
    spatial_basis: np.ndarray,
    n: int,
    seed: int,
):
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    B = np.asarray(spatial_basis, dtype=np.float64)
    out = []

    tries = 0
    while len(out) < n and tries < n * 20 + 20:
        tries += 1
        v = rng.standard_normal(dim)
        v = v - B @ (B.T @ v)
        nv = float(np.linalg.norm(v))
        if nv <= 1e-8:
            continue
        out.append((v / nv).astype(np.float32))

    return out


def project_to_spatial(
    v: np.ndarray,
    basis: np.ndarray,
):
    B = np.asarray(basis, dtype=np.float64)
    x = np.asarray(v, dtype=np.float64)
    p = B @ (B.T @ x)
    return p.astype(np.float32)


# =============================================================================
# All-layer gradient scan
# =============================================================================

def scan_sample_gradient(
    *,
    model,
    processor,
    token_map,
    decoder_layers,
    selected_layers,
    rec,
    image,
    device,
    assets,
    sid,
    prompt_template,
    random_controls,
    seed,
):
    idx = assets["idx_by_sid"][sid]
    gt = str(assets["labels"][idx])
    gen = assets["generation"][sid]
    gen_group = gen["generation_group"]
    gen_pred = gen["generation_pred"]

    question = prompt_template.format(
        subject=rec.subject,
        reference=rec.reference,
    )
    batch, subj_pos, ref_pos = build_batch(
        processor,
        rec,
        question,
        image,
        device,
    )

    model.zero_grad(set_to_none=True)

    with LayerOutputCapture(
        decoder_layers,
        selected_layers,
    ) as cap:
        outputs = model(
            **batch,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        cap.validate()

        logits = extract_logits(outputs)
        rel_scores = relation_scores(
            logits[0, -1],
            token_map,
        )
        rel_np = (
            rel_scores.detach().float().cpu().numpy()
        )

        gt_i = REL2ID[gt]
        baseline_pred = RELATIONS[int(np.argmax(rel_np))]
        non_gt = [r for r in RELATIONS if r != gt]
        baseline_best_comp = max(
            non_gt,
            key=lambda r: float(rel_np[REL2ID[r]]),
        )

        if (
            gen_group == "wrong"
            and gen_pred in REL2ID
            and gen_pred != gt
        ):
            primary_foil = gen_pred
            primary_kind = "actual_generated_wrong"
        else:
            primary_foil = baseline_best_comp
            primary_kind = "first_step_best_nonGT"

        grad_inputs = [
            cap.tensors[li] for li in selected_layers
        ]

        # 4 backward traversals -> all relation-logit gradients at every layer.
        grad_by_relation: Dict[str, Dict[int, torch.Tensor]] = {}

        for ri, rel in enumerate(RELATIONS):
            grads = torch.autograd.grad(
                rel_scores[ri],
                grad_inputs,
                retain_graph=(ri < len(RELATIONS) - 1),
                create_graph=False,
                allow_unused=True,
            )
            grad_by_relation[rel] = {
                li: g
                for li, g in zip(selected_layers, grads)
            }

        rows = []

        for li in selected_layers:
            residual_vec = assets["residual"][idx, li]
            cb = assets["codebooks"][li]

            # pooled gradient of each relation logit w.r.t. pair difference
            pair_grad_by_rel = {}

            for rel in RELATIONS:
                gfull = grad_by_relation[rel].get(li)
                if gfull is None:
                    pair_grad_by_rel[rel] = None
                    continue

                # Captured layer output is [1, seq, D].
                gseq = gfull[0].float()
                gs = pool_positions(gseq, subj_pos)
                gr = pool_positions(gseq, ref_pos)

                # derivative wrt pair-preserving epsilon edit
                pair_grad_by_rel[rel] = (
                    0.5 * (gs - gr)
                )

            for comp in RELATIONS:
                if comp == gt:
                    continue

                rep = representation_metrics(
                    residual_vec,
                    cb,
                    gt,
                    comp,
                )
                d_np = rep["semantic_axis"]
                d = torch.from_numpy(d_np).to(
                    device=device,
                    dtype=torch.float32,
                )

                gt_grad = pair_grad_by_rel[gt]
                comp_grad = pair_grad_by_rel[comp]

                if gt_grad is None or comp_grad is None:
                    continue

                margin_grad = (gt_grad - comp_grad).float()

                effects = {}
                for rel in RELATIONS:
                    g = pair_grad_by_rel[rel]
                    effects[rel] = (
                        float(torch.dot(g, d).item())
                        if g is not None else float("nan")
                    )

                C = effects[gt] - effects[comp]

                g_np = margin_grad.detach().cpu().numpy().astype(
                    np.float32
                )
                g_norm = float(np.linalg.norm(g_np))

                g_sp = project_to_spatial(
                    g_np,
                    cb["spatial_basis"],
                )
                g_sp_norm = float(np.linalg.norm(g_sp))

                spatial_grad_fraction = (
                    g_sp_norm / g_norm
                    if g_norm > EPS else float("nan")
                )

                semantic_causal_alignment = (
                    float(g_sp @ d_np / g_sp_norm)
                    if g_sp_norm > EPS else float("nan")
                )

                rand_units = random_orthogonal_units(
                    dim=len(d_np),
                    spatial_basis=cb["spatial_basis"],
                    n=random_controls,
                    seed=(
                        seed
                        + sid * 1000003
                        + li * 9176
                        + REL2ID[comp] * 131
                    ),
                )
                rand_effects = [
                    float(g_np @ rv)
                    for rv in rand_units
                ]
                random_abs_mean = safe_mean(
                    abs(x) for x in rand_effects
                )
                random_abs_max = (
                    max(abs(x) for x in rand_effects)
                    if rand_effects else float("nan")
                )
                specificity_ratio = (
                    abs(C) / random_abs_mean
                    if (
                        math.isfinite(random_abs_mean)
                        and random_abs_mean > EPS
                    )
                    else float("nan")
                )

                row = {
                    "sid": sid,
                    "layer": li,
                    "gt": gt,
                    "competitor": comp,
                    "generation_group": gen_group,
                    "generation_pred": gen_pred,
                    "primary_foil": primary_foil,
                    "is_primary_foil": int(comp == primary_foil),
                    "primary_foil_kind": (
                        primary_kind
                        if comp == primary_foil else ""
                    ),

                    "baseline_firststep_pred": baseline_pred,
                    "baseline_firststep_best_competitor":
                        baseline_best_comp,
                    "baseline_firststep_correct":
                        int(baseline_pred == gt),
                    "baseline_output_margin_gt_vs_comp":
                        float(
                            rel_np[gt_i]
                            - rel_np[REL2ID[comp]]
                        ),

                    # Representation availability.
                    "R_raw": rep["R_raw"],
                    "R_cos": rep["R_cos"],
                    "direction_pred": rep["direction_pred"],
                    "direction_correct":
                        rep["direction_correct"],
                    "direction_top2_margin":
                        rep["direction_top2_margin"],

                    # Four relation-logit effects under semantic intervention.
                    "effect_left": effects["left"],
                    "effect_right": effects["right"],
                    "effect_above": effects["above"],
                    "effect_below": effects["below"],
                    "effect_GT": effects[gt],
                    "effect_competitor": effects[comp],

                    # Main causal utilization.
                    "C_margin": C,
                    "C_positive": int(C > 0),

                    # Causal gradient geometry.
                    "margin_gradient_norm": g_norm,
                    "spatial_margin_gradient_norm": g_sp_norm,
                    "spatial_grad_fraction":
                        spatial_grad_fraction,
                    "semantic_vs_causal_alignment":
                        semantic_causal_alignment,

                    # Random orthogonal controls.
                    "random_orthogonal_abs_mean":
                        random_abs_mean,
                    "random_orthogonal_abs_max":
                        random_abs_max,
                    "causal_specificity_ratio":
                        specificity_ratio,
                }

                # Save spatial projected causal gradient coordinates.
                row["causal_spatial_left_axis"] = float(
                    g_sp @ unit_np(
                        cb["means"]["left"]
                        - cb["means"]["right"]
                    )
                )
                row["causal_spatial_above_axis"] = float(
                    g_sp @ unit_np(
                        cb["means"]["above"]
                        - cb["means"]["below"]
                    )
                )

                rows.append(row)

        # capture rows before references are dropped
        del grad_by_relation, grad_inputs, rel_scores, outputs

    del batch
    return rows


# =============================================================================
# Summaries / failure map
# =============================================================================

def primary_rows(per_rows):
    return [
        r for r in per_rows
        if int(r["is_primary_foil"]) == 1
    ]


def summarize_primary_layers(
    per_rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed)
    prim = primary_rows(per_rows)
    out = []

    for li in selected_layers:
        rr = [r for r in prim if int(r["layer"]) == li]
        correct = [
            r for r in rr
            if r["generation_group"] == "correct"
        ]
        wrong = [
            r for r in rr
            if r["generation_group"] == "wrong"
        ]

        row = {
            "layer": li,
            "n_correct": len(correct),
            "n_wrong": len(wrong),
        }

        for metric in (
            "R_raw",
            "R_cos",
            "C_margin",
            "semantic_vs_causal_alignment",
            "spatial_grad_fraction",
            "causal_specificity_ratio",
        ):
            cv = [float(r[metric]) for r in correct]
            wv = [float(r[metric]) for r in wrong]
            gap, lo, hi = bootstrap_gap(
                cv, wv, bootstrap, rng
            )

            row[f"{metric}_correct"] = safe_mean(cv)
            row[f"{metric}_wrong"] = safe_mean(wv)
            row[f"{metric}_correct_minus_wrong"] = gap
            row[f"{metric}_gap_ci95_lo"] = lo
            row[f"{metric}_gap_ci95_hi"] = hi

        row["direction_correct_rate_correct"] = safe_frac(
            int(r["direction_correct"]) == 1
            for r in correct
        )
        row["direction_correct_rate_wrong"] = safe_frac(
            int(r["direction_correct"]) == 1
            for r in wrong
        )
        row["C_positive_rate_correct"] = safe_frac(
            float(r["C_margin"]) > 0
            for r in correct
        )
        row["C_positive_rate_wrong"] = safe_frac(
            float(r["C_margin"]) > 0
            for r in wrong
        )
        row["semantic_causal_mismatch_rate_correct"] = safe_frac(
            (
                float(r["R_raw"]) > 0
                and float(r["C_margin"]) <= 0
            )
            for r in correct
        )
        row["semantic_causal_mismatch_rate_wrong"] = safe_frac(
            (
                float(r["R_raw"]) > 0
                and float(r["C_margin"]) <= 0
            )
            for r in wrong
        )

        out.append(row)

    return out


def build_control_thresholds(
    per_rows,
    quantile,
):
    """
    Correct controls use ALL axis rows, matched by:
        layer, GT, competitor.
    This is better than requiring the competitor to have been the correct
    sample's own primary foil.
    """
    buckets = defaultdict(list)

    for r in per_rows:
        if r["generation_group"] != "correct":
            continue
        key = (
            int(r["layer"]),
            str(r["gt"]),
            str(r["competitor"]),
        )
        buckets[key].append(r)

    targets = {}
    rows = []

    for key, rr in sorted(buckets.items()):
        li, gt, comp = key

        R = np.asarray(
            [float(r["R_raw"]) for r in rr],
            dtype=np.float64,
        )
        C = np.asarray(
            [float(r["C_margin"]) for r in rr],
            dtype=np.float64,
        )
        A = np.asarray(
            [
                float(r["semantic_vs_causal_alignment"])
                for r in rr
            ],
            dtype=np.float64,
        )
        A = A[np.isfinite(A)]

        targets[key] = {
            "n": len(rr),
            "R_q": float(np.quantile(R, quantile)),
            "R_median": float(np.median(R)),
            "R_mean": float(np.mean(R)),
            "C_q": float(np.quantile(C, quantile)),
            "C_median": float(np.median(C)),
            "C_mean": float(np.mean(C)),
            "alignment_median":
                float(np.median(A)) if len(A) else float("nan"),
        }

        rows.append({
            "layer": li,
            "gt": gt,
            "competitor": comp,
            "n_correct_controls": len(rr),
            "quantile": quantile,
            **targets[key],
        })

    return targets, rows


def build_wrong_failure_map(
    per_rows,
    targets,
):
    rows = []

    for r in primary_rows(per_rows):
        if r["generation_group"] != "wrong":
            continue

        key = (
            int(r["layer"]),
            str(r["gt"]),
            str(r["competitor"]),
        )
        if key not in targets:
            continue

        t = targets[key]
        R = float(r["R_raw"])
        C = float(r["C_margin"])

        rep_def = R < float(t["R_q"])
        causal_def = C < float(t["C_q"])

        if rep_def and causal_def:
            failure_type = "both_representation_and_utilization_deficit"
        elif rep_def:
            failure_type = "representation_deficit_only"
        elif causal_def:
            failure_type = "causal_utilization_deficit_only"
        else:
            failure_type = "neither_deficit"

        if R > 0 and C <= 0:
            mismatch = 1
        else:
            mismatch = 0

        row = {
            "sid": int(r["sid"]),
            "layer": int(r["layer"]),
            "gt": r["gt"],
            "final_wrong": r["competitor"],

            "R_raw": R,
            "correct_control_R_q": t["R_q"],
            "correct_control_R_median": t["R_median"],
            "representation_deficit": int(rep_def),
            "R_deficit_from_median":
                float(t["R_median"]) - R,

            "C_margin": C,
            "correct_control_C_q": t["C_q"],
            "correct_control_C_median": t["C_median"],
            "causal_utilization_deficit": int(causal_def),
            "C_deficit_from_median":
                float(t["C_median"]) - C,

            "semantic_vs_causal_alignment":
                r["semantic_vs_causal_alignment"],
            "spatial_grad_fraction":
                r["spatial_grad_fraction"],
            "causal_specificity_ratio":
                r["causal_specificity_ratio"],

            "semantic_causal_sign_mismatch": mismatch,
            "failure_type": failure_type,
        }
        rows.append(row)

    return rows


def summarize_wrong_failure_types(
    wrong_rows,
    selected_layers,
):
    out = []

    types = (
        "representation_deficit_only",
        "causal_utilization_deficit_only",
        "both_representation_and_utilization_deficit",
        "neither_deficit",
    )

    for li in selected_layers:
        rr = [
            r for r in wrong_rows
            if int(r["layer"]) == li
        ]
        n = len(rr)
        counts = Counter(r["failure_type"] for r in rr)

        row = {
            "layer": li,
            "n_wrong": n,
            "representation_deficit_rate": safe_frac(
                int(r["representation_deficit"]) == 1
                for r in rr
            ),
            "causal_utilization_deficit_rate": safe_frac(
                int(r["causal_utilization_deficit"]) == 1
                for r in rr
            ),
            "semantic_causal_sign_mismatch_rate": safe_frac(
                int(r["semantic_causal_sign_mismatch"]) == 1
                for r in rr
            ),
            "mean_R_deficit_from_median": safe_mean(
                r["R_deficit_from_median"] for r in rr
            ),
            "mean_C_deficit_from_median": safe_mean(
                r["C_deficit_from_median"] for r in rr
            ),
        }

        for typ in types:
            row[f"{typ}_rate"] = (
                counts[typ] / n if n else float("nan")
            )

        out.append(row)

    return out


def combine_layer_map(
    primary_summary,
    failure_summary,
):
    f = {
        int(r["layer"]): r
        for r in failure_summary
    }
    out = []

    for r in primary_summary:
        li = int(r["layer"])
        q = f.get(li, {})
        row = dict(r)
        row.update({
            "wrong_representation_deficit_rate":
                q.get(
                    "representation_deficit_rate",
                    float("nan"),
                ),
            "wrong_causal_utilization_deficit_rate":
                q.get(
                    "causal_utilization_deficit_rate",
                    float("nan"),
                ),
            "wrong_both_deficit_rate":
                q.get(
                    "both_representation_and_utilization_deficit_rate",
                    float("nan"),
                ),
            "wrong_semantic_causal_sign_mismatch_rate":
                q.get(
                    "semantic_causal_sign_mismatch_rate",
                    float("nan"),
                ),
        })

        # Descriptive candidate score:
        # high when correct uses semantic Direction more strongly than wrong,
        # while wrong still retains positive representation availability.
        c_gap = float(
            r["C_margin_correct_minus_wrong"]
        )
        wrong_R = float(r["R_raw_wrong"])
        row["candidate_utilization_gap"] = (
            c_gap if wrong_R > 0 else 0.0
        )

        out.append(row)

    out.sort(
        key=lambda x: float(x["candidate_utilization_gap"]),
        reverse=True,
    )
    for i, r in enumerate(out, 1):
        r["rank"] = i

    return out


# =============================================================================
# Finite-difference intervention
# =============================================================================

class PairDirectionIntervention:
    def __init__(
        self,
        module,
        subject_positions,
        reference_positions,
        delta,
    ):
        self.subj = list(map(int, subject_positions))
        self.ref = list(map(int, reference_positions))
        self.delta = delta
        self.applied = False
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        if self.applied:
            return output

        x = first_tensor(output)
        if int(x.shape[1]) <= max(self.subj + self.ref):
            return output

        y = x.clone()
        half = 0.5 * self.delta.to(
            device=y.device,
            dtype=y.dtype,
        )
        y[0, self.subj, :] = (
            y[0, self.subj, :] + half
        )
        y[0, self.ref, :] = (
            y[0, self.ref, :] - half
        )
        self.applied = True
        return replace_first_tensor(output, y)

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


def score_forward(
    model,
    batch,
    token_map,
):
    with torch.inference_mode():
        outputs = model(
            **batch,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        logits = extract_logits(outputs)
        scores = relation_scores(
            logits[0, -1],
            token_map,
        )
    arr = scores.detach().float().cpu().numpy()
    return {
        rel: float(arr[REL2ID[rel]])
        for rel in RELATIONS
    }


def score_with_intervention(
    *,
    model,
    batch,
    token_map,
    module,
    subj_pos,
    ref_pos,
    delta_np,
):
    delta = torch.from_numpy(
        np.asarray(delta_np, dtype=np.float32)
    )
    hook = PairDirectionIntervention(
        module,
        subj_pos,
        ref_pos,
        delta,
    )
    try:
        result = score_forward(
            model,
            batch,
            token_map,
        )
        if not hook.applied:
            raise RuntimeError(
                "Finite-difference hook was not applied."
            )
        return result
    finally:
        hook.close()


def choose_fd_layers(
    fd_arg: str,
    layer_map,
    n_layers,
    top_k,
):
    t = fd_arg.strip().lower()
    if t == "none":
        return []
    if t != "auto":
        return parse_layers(fd_arg, n_layers)

    candidates = [
        r for r in layer_map
        if math.isfinite(
            float(r["candidate_utilization_gap"])
        )
    ]
    candidates.sort(
        key=lambda r: float(
            r["candidate_utilization_gap"]
        ),
        reverse=True,
    )

    selected = [
        int(r["layer"])
        for r in candidates[: int(top_k)]
        if float(r["candidate_utilization_gap"]) > 0
    ]

    if not selected:
        # fallback: strongest mean |C| on correct primary-foil rows
        candidates.sort(
            key=lambda r: abs(float(r["C_margin_correct"])),
            reverse=True,
        )
        selected = [
            int(r["layer"])
            for r in candidates[: int(top_k)]
        ]

    return sorted(set(selected))


def finite_difference_scan(
    *,
    model,
    processor,
    token_map,
    decoder_layers,
    selected_fd_layers,
    records_by_sid,
    assets,
    gradient_rows,
    device,
    prompt_template,
    max_samples,
    eps_scales,
    random_control,
    seed,
):
    if not selected_fd_layers:
        return [], []

    # Use primary-foil gradient rows as sample/axis definition.
    prim = [
        r for r in primary_rows(gradient_rows)
        if int(r["layer"]) in selected_fd_layers
    ]

    # Choose sample SIDs once, stratified correct/wrong as much as possible.
    all_sids = sorted({int(r["sid"]) for r in prim})
    if max_samples is not None and len(all_sids) > max_samples:
        rng = random.Random(seed + 999)
        correct_sids = sorted({
            int(r["sid"]) for r in prim
            if r["generation_group"] == "correct"
        })
        wrong_sids = sorted({
            int(r["sid"]) for r in prim
            if r["generation_group"] == "wrong"
        })
        rng.shuffle(correct_sids)
        rng.shuffle(wrong_sids)

        n_wrong = min(
            len(wrong_sids),
            max_samples // 2,
        )
        n_correct = min(
            len(correct_sids),
            max_samples - n_wrong,
        )
        chosen = wrong_sids[:n_wrong] + correct_sids[:n_correct]

        if len(chosen) < max_samples:
            remaining = [
                s for s in all_sids if s not in set(chosen)
            ]
            rng.shuffle(remaining)
            chosen += remaining[: max_samples - len(chosen)]

        all_sids = sorted(set(chosen))

    grad_lookup = {
        (int(r["sid"]), int(r["layer"])): r
        for r in prim
        if int(r["sid"]) in set(all_sids)
    }

    rows = []

    for sid in tqdm(
        all_sids,
        desc="finite-difference validation",
    ):
        rec = records_by_sid[sid]
        image = None
        try:
            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            image = Image.open(rec.image_path).convert("RGB")
            batch, subj_pos, ref_pos = build_batch(
                processor,
                rec,
                question,
                image,
                device,
            )

            baseline = score_forward(
                model,
                batch,
                token_map,
            )

            for li in selected_fd_layers:
                key = (sid, li)
                if key not in grad_lookup:
                    continue

                gr = grad_lookup[key]
                gt = str(gr["gt"])
                comp = str(gr["competitor"])
                cb = assets["codebooks"][li]
                d = semantic_axis(cb, gt, comp)

                sigma = float(
                    cb["axis_std"][(gt, comp)]
                )
                if not math.isfinite(sigma) or sigma <= EPS:
                    sigma = 1.0

                base_margin = (
                    baseline[gt] - baseline[comp]
                )

                # One deterministic random orthogonal direction for this
                # sample/layer/axis, reused at every epsilon.
                rand_units = random_orthogonal_units(
                    dim=len(d),
                    spatial_basis=cb["spatial_basis"],
                    n=1,
                    seed=(
                        seed
                        + sid * 700001
                        + li * 1009
                        + REL2ID[comp] * 17
                    ),
                )
                rand_d = (
                    rand_units[0] if rand_units else None
                )

                for scale in eps_scales:
                    eps = float(scale) * sigma

                    plus = score_with_intervention(
                        model=model,
                        batch=batch,
                        token_map=token_map,
                        module=decoder_layers[li],
                        subj_pos=subj_pos,
                        ref_pos=ref_pos,
                        delta_np=(eps * d),
                    )
                    minus = score_with_intervention(
                        model=model,
                        batch=batch,
                        token_map=token_map,
                        module=decoder_layers[li],
                        subj_pos=subj_pos,
                        ref_pos=ref_pos,
                        delta_np=(-eps * d),
                    )

                    plus_margin = plus[gt] - plus[comp]
                    minus_margin = minus[gt] - minus[comp]

                    fd_slope = (
                        plus_margin - minus_margin
                    ) / (2.0 * eps)

                    row = {
                        "sid": sid,
                        "layer": li,
                        "gt": gt,
                        "competitor": comp,
                        "generation_group":
                            gr["generation_group"],
                        "generation_pred":
                            gr["generation_pred"],
                        "R_raw": gr["R_raw"],
                        "gradient_C_margin":
                            gr["C_margin"],

                        "eps_scale": scale,
                        "axis_train_sigma": sigma,
                        "eps": eps,

                        "baseline_margin": base_margin,
                        "plus_margin": plus_margin,
                        "minus_margin": minus_margin,
                        "plus_gain":
                            plus_margin - base_margin,
                        "minus_gain":
                            minus_margin - base_margin,
                        "finite_difference_slope":
                            fd_slope,
                        "gradient_fd_same_sign": int(
                            (
                                float(gr["C_margin"]) > 0
                                and fd_slope > 0
                            )
                            or (
                                float(gr["C_margin"]) < 0
                                and fd_slope < 0
                            )
                        ),
                        "semantic_monotonic":
                            int(
                                plus_margin
                                > base_margin
                                > minus_margin
                            ),
                    }

                    if random_control and rand_d is not None:
                        rp = score_with_intervention(
                            model=model,
                            batch=batch,
                            token_map=token_map,
                            module=decoder_layers[li],
                            subj_pos=subj_pos,
                            ref_pos=ref_pos,
                            delta_np=(eps * rand_d),
                        )
                        rm = score_with_intervention(
                            model=model,
                            batch=batch,
                            token_map=token_map,
                            module=decoder_layers[li],
                            subj_pos=subj_pos,
                            ref_pos=ref_pos,
                            delta_np=(-eps * rand_d),
                        )

                        rp_margin = rp[gt] - rp[comp]
                        rm_margin = rm[gt] - rm[comp]
                        rand_slope = (
                            rp_margin - rm_margin
                        ) / (2.0 * eps)

                        row.update({
                            "random_plus_margin": rp_margin,
                            "random_minus_margin": rm_margin,
                            "random_fd_slope": rand_slope,
                            "abs_semantic_minus_random_slope":
                                abs(fd_slope) - abs(rand_slope),
                        })
                    else:
                        row.update({
                            "random_plus_margin": float("nan"),
                            "random_minus_margin": float("nan"),
                            "random_fd_slope": float("nan"),
                            "abs_semantic_minus_random_slope":
                                float("nan"),
                        })

                    rows.append(row)

            del batch

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summaries.
    buckets = defaultdict(list)
    for r in rows:
        buckets[
            (
                int(r["layer"]),
                float(r["eps_scale"]),
                str(r["generation_group"]),
            )
        ].append(r)

    summary = []
    for (li, scale, group), rr in sorted(buckets.items()):
        grad = np.asarray(
            [float(r["gradient_C_margin"]) for r in rr],
            dtype=np.float64,
        )
        fd = np.asarray(
            [float(r["finite_difference_slope"]) for r in rr],
            dtype=np.float64,
        )
        rand = np.asarray(
            [float(r["random_fd_slope"]) for r in rr],
            dtype=np.float64,
        )

        if len(rr) >= 2 and np.std(grad) > 0 and np.std(fd) > 0:
            corr = float(np.corrcoef(grad, fd)[0, 1])
        else:
            corr = float("nan")

        summary.append({
            "layer": li,
            "eps_scale": scale,
            "generation_group": group,
            "n": len(rr),
            "mean_gradient_C": float(grad.mean()),
            "mean_fd_slope": float(fd.mean()),
            "mean_random_fd_slope":
                float(rand.mean()) if len(rand) else float("nan"),
            "mean_abs_fd_slope": float(np.mean(np.abs(fd))),
            "mean_abs_random_fd_slope":
                float(np.mean(np.abs(rand)))
                if len(rand) else float("nan"),
            "gradient_fd_correlation": corr,
            "same_sign_rate": safe_frac(
                int(r["gradient_fd_same_sign"]) == 1
                for r in rr
            ),
            "semantic_monotonic_rate": safe_frac(
                int(r["semantic_monotonic"]) == 1
                for r in rr
            ),
            "semantic_abs_gt_random_rate": safe_frac(
                abs(float(r["finite_difference_slope"]))
                > abs(float(r["random_fd_slope"]))
                for r in rr
                if math.isfinite(float(r["random_fd_slope"]))
            ),
        })

    return rows, summary


# =============================================================================
# Console
# =============================================================================

def print_primary_summary(rows):
    print("\n" + "=" * 178)
    print("SEMANTIC DIRECTION CAUSAL MAP — PRIMARY FOIL")
    print("=" * 178)
    print(
        "layer Ncor Nwr | R cor/wr gap(C-W) | "
        "C cor/wr gap(C-W) 95%CI | align cor/wr | "
        "C>0 cor/wr | R>0,C<=0 cor/wr"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{int(r['n_correct']):3d} {int(r['n_wrong']):3d} | "
            f"{float(r['R_raw_correct']):+7.3f}/"
            f"{float(r['R_raw_wrong']):+7.3f} "
            f"{float(r['R_raw_correct_minus_wrong']):+7.3f} | "
            f"{float(r['C_margin_correct']):+8.4f}/"
            f"{float(r['C_margin_wrong']):+8.4f} "
            f"{float(r['C_margin_correct_minus_wrong']):+8.4f} "
            f"[{float(r['C_margin_gap_ci95_lo']):+7.4f},"
            f"{float(r['C_margin_gap_ci95_hi']):+7.4f}] | "
            f"{float(r['semantic_vs_causal_alignment_correct']):+6.3f}/"
            f"{float(r['semantic_vs_causal_alignment_wrong']):+6.3f} | "
            f"{float(r['C_positive_rate_correct']):.3f}/"
            f"{float(r['C_positive_rate_wrong']):.3f} | "
            f"{float(r['semantic_causal_mismatch_rate_correct']):.3f}/"
            f"{float(r['semantic_causal_mismatch_rate_wrong']):.3f}"
        )


def print_failure_summary(rows):
    print("\n" + "=" * 142)
    print("GENERATION-WRONG FAILURE MAP — SAME GT + SAME FINAL-WRONG AXIS CONTROLS")
    print("=" * 142)
    print(
        "layer N | RepDef UtilDef BOTH RepOnly UtilOnly Neither | "
        "signMismatch meanRdef meanCdef"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} {int(r['n_wrong']):3d} | "
            f"{float(r['representation_deficit_rate']):.3f} "
            f"{float(r['causal_utilization_deficit_rate']):.3f} "
            f"{float(r['both_representation_and_utilization_deficit_rate']):.3f} "
            f"{float(r['representation_deficit_only_rate']):.3f} "
            f"{float(r['causal_utilization_deficit_only_rate']):.3f} "
            f"{float(r['neither_deficit_rate']):.3f} | "
            f"{float(r['semantic_causal_sign_mismatch_rate']):.3f} "
            f"{float(r['mean_R_deficit_from_median']):+7.3f} "
            f"{float(r['mean_C_deficit_from_median']):+8.4f}"
        )


def print_fd_summary(rows):
    if not rows:
        return
    print("\n" + "=" * 158)
    print("FINITE-DIFFERENCE VALIDATION OF SEMANTIC DIRECTION CAUSAL RESPONSE")
    print("=" * 158)
    print(
        "layer eps group N | gradC fdSlope randomSlope corr "
        "sameSign monotonic | |semantic|>|random|"
    )
    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{float(r['eps_scale']):.2f} "
            f"{str(r['generation_group']):>7s} "
            f"{int(r['n']):3d} | "
            f"{float(r['mean_gradient_C']):+8.4f} "
            f"{float(r['mean_fd_slope']):+8.4f} "
            f"{float(r['mean_random_fd_slope']):+8.4f} "
            f"{float(r['gradient_fd_correlation']):+6.3f} "
            f"{float(r['same_sign_rate']):.3f} "
            f"{float(r['semantic_monotonic_rate']):.3f} | "
            f"{float(r['semantic_abs_gt_random_rate']):.3f}"
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

    assets = load_direction_assets(
        Path(args.direction_dir)
    )
    selected_layers = parse_layers(
        args.layers,
        assets["n_layers"],
    )

    # Select SIDs based on cached ACTUAL generation group.
    eval_sids = []
    for sid in assets["idx_by_sid"]:
        if (
            args.split != "all"
            and assets["split"].get(sid, "") != args.split
        ):
            continue

        idx = assets["idx_by_sid"][sid]
        gt = assets["labels"][idx]
        gen = assets["generation"].get(sid, {})
        group = gen.get("generation_group", "")
        pred = gen.get("generation_pred", "")

        if gt not in REL2ID:
            continue
        if group not in ("correct", "wrong"):
            continue
        if group == "wrong" and (
            pred not in REL2ID or pred == gt
        ):
            continue

        eval_sids.append(int(sid))

    if args.max_samples is not None and len(eval_sids) > args.max_samples:
        rng = random.Random(args.seed)
        correct = [
            sid for sid in eval_sids
            if assets["generation"][sid]["generation_group"]
            == "correct"
        ]
        wrong = [
            sid for sid in eval_sids
            if assets["generation"][sid]["generation_group"]
            == "wrong"
        ]
        rng.shuffle(correct)
        rng.shuffle(wrong)

        # Keep wrong samples well represented.
        n_wrong = min(
            len(wrong),
            max(1, args.max_samples // 2),
        )
        n_correct = min(
            len(correct),
            args.max_samples - n_wrong,
        )
        chosen = wrong[:n_wrong] + correct[:n_correct]
        if len(chosen) < args.max_samples:
            remaining = [
                s for s in eval_sids
                if s not in set(chosen)
            ]
            rng.shuffle(remaining)
            chosen += remaining[
                : args.max_samples - len(chosen)
            ]
        eval_sids = sorted(set(chosen))

    if not eval_sids:
        raise RuntimeError("No evaluation samples.")

    group_counts = Counter(
        assets["generation"][sid]["generation_group"]
        for sid in eval_sids
    )

    print(
        f"[data] split={args.split}, N={len(eval_sids)}, "
        f"groups={dict(group_counts)}, layers={selected_layers}"
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
        if int(r.sid) in set(eval_sids)
    }
    missing = sorted(
        set(eval_sids) - set(records_by_sid)
    )
    if missing:
        raise RuntimeError(
            f"Missing records for SIDs {missing[:10]}"
        )

    # Model.
    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)

    common_kw: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        common_kw["attn_implementation"] = args.attn_impl

    dtype = base.resolve_dtype(spec.dtype_name)

    print(f"[model] loading {spec.repo_id} on {args.device}")
    try:
        model = cls.from_pretrained(
            spec.repo_id,
            dtype=dtype,
            **common_kw,
        )
    except TypeError:
        model = cls.from_pretrained(
            spec.repo_id,
            torch_dtype=dtype,
            **common_kw,
        )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, layer_path = resolve_decoder_layers(
        model
    )
    if len(decoder_layers) != assets["n_layers"]:
        raise RuntimeError(
            f"Layer mismatch model={len(decoder_layers)} "
            f"cache={assets['n_layers']}"
        )

    print(f"[decoder] {layer_path}")
    token_map = relation_token_variants(
        processor.tokenizer
    )

    # -------------------------------------------------------------------------
    # Part 1: all-layer gradient causal response
    # -------------------------------------------------------------------------
    per_rows = []
    error_rows = []
    per_path = out_dir / "per_sample_layer_axis.csv"

    for ii, sid in enumerate(
        tqdm(eval_sids, desc="semantic-direction causal gradient"),
        1,
    ):
        rec = records_by_sid[sid]
        image = None
        try:
            image = Image.open(
                rec.image_path
            ).convert("RGB")

            rows = scan_sample_gradient(
                model=model,
                processor=processor,
                token_map=token_map,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                rec=rec,
                image=image,
                device=device,
                assets=assets,
                sid=sid,
                prompt_template=args.prompt_template,
                random_controls=args.random_controls,
                seed=args.seed,
            )
            per_rows.extend(rows)

            if (
                args.save_every > 0
                and ii % args.save_every == 0
            ):
                write_csv(per_path, per_rows)

        except Exception as e:
            error_rows.append({
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
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(per_path, per_rows)
    write_csv(out_dir / "errors.csv", error_rows)

    if not per_rows:
        raise RuntimeError(
            "No successful gradient rows."
        )

    primary_summary = summarize_primary_layers(
        per_rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    write_csv(
        out_dir / "primary_foil_layer_summary.csv",
        primary_summary,
    )

    targets, target_rows = build_control_thresholds(
        per_rows,
        args.control_quantile,
    )
    write_csv(
        out_dir / "correct_axis_control_thresholds.csv",
        target_rows,
    )

    wrong_map = build_wrong_failure_map(
        per_rows,
        targets,
    )
    write_csv(
        out_dir / "wrong_failure_map.csv",
        wrong_map,
    )

    wrong_summary = summarize_wrong_failure_types(
        wrong_map,
        selected_layers,
    )
    write_csv(
        out_dir / "wrong_failure_type_summary.csv",
        wrong_summary,
    )

    layer_map = combine_layer_map(
        primary_summary,
        wrong_summary,
    )
    write_csv(
        out_dir / "layer_causal_map.csv",
        layer_map,
    )

    print_primary_summary(primary_summary)
    print_failure_summary(wrong_summary)

    print("\nTop utilization-gap layers:")
    for r in layer_map[:10]:
        print(
            f"  L{int(r['layer']):02d}: "
            f"R_wrong={float(r['R_raw_wrong']):+.3f}, "
            f"C_correct={float(r['C_margin_correct']):+.4f}, "
            f"C_wrong={float(r['C_margin_wrong']):+.4f}, "
            f"Cgap={float(r['C_margin_correct_minus_wrong']):+.4f}, "
            f"utilDefRate="
            f"{float(r['wrong_causal_utilization_deficit_rate']):.3f}"
        )

    # -------------------------------------------------------------------------
    # Part 2: finite-difference validation
    # -------------------------------------------------------------------------
    fd_layers = choose_fd_layers(
        args.fd_layers,
        layer_map,
        assets["n_layers"],
        args.fd_top_k,
    )
    print(f"\n[finite difference layers] {fd_layers}")

    fd_rows = []
    fd_summary = []

    if fd_layers:
        eps_scales = parse_float_list(
            args.fd_eps_scales
        )

        fd_rows, fd_summary = finite_difference_scan(
            model=model,
            processor=processor,
            token_map=token_map,
            decoder_layers=decoder_layers,
            selected_fd_layers=fd_layers,
            records_by_sid=records_by_sid,
            assets=assets,
            gradient_rows=per_rows,
            device=device,
            prompt_template=args.prompt_template,
            max_samples=args.fd_max_samples,
            eps_scales=eps_scales,
            random_control=args.fd_random_control,
            seed=args.seed,
        )

        write_csv(
            out_dir / "finite_difference_per_sample.csv",
            fd_rows,
        )
        write_csv(
            out_dir / "finite_difference_summary.csv",
            fd_summary,
        )
        print_fd_summary(fd_summary)

    meta = {
        "experiment":
            "Semantic Direction representation availability + causal response",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_eval_requested": len(eval_sids),
        "generation_group_counts": dict(group_counts),
        "selected_layers": selected_layers,
        "finite_difference_layers": fd_layers,
        "random_controls": args.random_controls,
        "control_quantile": args.control_quantile,

        "R_definition":
            "(residual_l-center_l) dot unit(mu_GT,l-mu_comp,l)",

        "C_definition":
            "directional derivative of first-step "
            "(logit_GT-logit_comp) under pair-preserving semantic-axis edit",

        "causal_direction_geometry":
            "pair gradient projected onto 2D span(right-left, above-below)",

        "wrong_primary_foil":
            "cached actual model.generate() wrong relation",

        "correct_primary_foil":
            "highest non-GT first-step relation score",

        "finite_difference_eps":
            "eps_scale * train std of cached residual projection "
            "on tested GT-vs-competitor semantic axis",

        "important_interpretation":
            "R is representation availability. C is local causal utilization. "
            "Do not infer causal use from R alone. Finite difference validates "
            "the gradient response at selected layers.",

        "limitation":
            "Causal outcome is first relation-token margin, not full "
            "autoregressive generation. Full generation should be tested only "
            "after a concrete failure mechanism is localized.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            meta,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "per_sample_layer_axis.csv",
        "primary_foil_layer_summary.csv",
        "correct_axis_control_thresholds.csv",
        "wrong_failure_map.csv",
        "wrong_failure_type_summary.csv",
        "layer_causal_map.csv",
        "finite_difference_per_sample.csv",
        "finite_difference_summary.csv",
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
