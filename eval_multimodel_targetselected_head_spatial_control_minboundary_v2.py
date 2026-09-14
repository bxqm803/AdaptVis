#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_multimodel_targetselected_head_spatial_control_minboundary_v2.py

Cross-model / cross-dataset spatial-readout -> minimum residual spatial control -> generation.

Protocol
========
For EACH model independently:

  A) Direction reader (precomputed by scan_synthetic_frozen_direction_heads_7models_v2.py)
     1. Fit every attention head's four-way relation directions ONLY on Synthetic-400.
     2. Freeze those directions.
     3. Apply them on a target dataset (COCO or Controlled-A).
     4. Use target GT ONLY to choose the single head ID with the highest target
        accuracy under the frozen Synthetic directions.
     5. After the head ID is fixed, each target sample is routed by that head's
        own predicted relation. Per-sample GT is NOT used by the `head` condition.

  B) Minimum-residual spatial actuator
     1. Fit a 2D H/V object-pair spatial SUBSPACE ONLY from Synthetic-400.
        Synthetic labels define the subspace directions; Synthetic class-gap
        magnitudes are NOT used to choose intervention distance.
     2. At a seven-layer mid/late window, edit only the subject/reference
        residual relation state:

            h_sub <- h_sub + delta_z / 2
            h_ref <- h_ref - delta_z / 2

        where delta_z is constrained to that layer's Synthetic H/V subspace.
     3. For the reader-selected target relation, compute target-vs-competitor
        teacher-forced margins and their exact gradients with respect to all
        2*7 spatial coordinates.
     4. Solve the local minimum-norm boundary-crossing problem: find the
        smallest TOTAL residual-space spatial change that makes the target
        relation beat all three competitors under the local linearization.
     5. Apply that minimum step, rerun the model, and relinearize if needed.
        Stop when ACTUAL autoregressive generation matches the control target.

Conditions
==========
  head   : target relation = selected spatial head prediction (non-oracle per sample)
  oracle : target relation = target GT (upper bound only)

Important method label
======================
The `head` condition is NOT fully label-free because target GT is used once to
select the head ID on that target dataset. The per-sample routing after head
selection is GT-free. This is a target-supervised head-selection diagnostic.

The residual H/V subspace is source-only (Synthetic-400). No hidden-space
vectors are shared across models: every model gets its own Synthetic reader
codebook and its own Synthetic residual subspace.

Unlike v1, this controller does NOT use Synthetic half-gap scaling, a fixed
"natural" step, normalized-gradient ascent, or backtracking line search.
Synthetic determines WHERE spatial edits are allowed; the current target
sample's decision margins and gradients determine HOW MUCH each layer changes.
The optimized norm is the actual relation-residual L2 norm inside the fitted
spatial subspaces.

Head-scan dependency
====================
Run first:

  python scan_synthetic_frozen_direction_heads_7models_v2.py \
    --models all --datasets all --head-selection-frac 1.0 --control gray \
    --output-dir output/syn400_direction_head_7models_targetfull_v2

This controller reads, for each model/dataset:
  <headscan-dir>/<model>/<dataset>/summary.json
  <headscan-dir>/<model>/<dataset>/samples_best_head.csv

Default controller window
=========================
Qwen2.5-VL-3B used L20-26 out of 36 decoder blocks. For other architectures
we preserve the same relative depth and the same 7-layer / 14D control budget:

  center_frac = 23 / 35
  center = round(center_frac * (n_layers - 1))
  layers = center-3 ... center+3, clipped to a valid contiguous 7-layer window.

Use --controller-layers to override, e.g.
  --controller-layers qwen-3b=20-26,llava-7b=17-23

Synthetic residual subspace
===========================
Default --geometry-control gray uses:

  q_L = (h_sub^real - h_ref^real) - (h_sub^gray - h_ref^gray)

at each decoder block output. Four GT-conditioned Synthetic means define
Left/Right/Above/Below directions, from which H and V axes are constructed.
Only the span of H/V is retained for control. An orthonormal basis of that span
makes the optimized coordinate norm equal to the actual residual-edit norm.

Resume / runtime
================
Results are appended per sample/condition. Rerun the same command without
--overwrite to resume. Start with --eval-max-samples 80 before all-data runs.

Dependencies (repo root)
========================
  scan_synthetic_frozen_direction_heads_7models_v2.py
  analyze_coco_head_object_residual_direction_probe_v1.py
  scan_coco_receiver_scaling_multimodel_200_v1.py
  eval_synthetic_fixedwindow_centroid_5models_3datasets_v1.py
plus the dependencies already required by the head-scan script.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import csv
import gc
import json
import math
import os
import random
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import scan_synthetic_frozen_direction_heads_7models_v2 as hs
except Exception as exc:
    raise SystemExit(
        "Could not import scan_synthetic_frozen_direction_heads_7models_v2.py. "
        "Run from the AdaptVis llava16 repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import scan_coco_receiver_scaling_multimodel_200_v1 as recv
except Exception as exc:
    raise SystemExit(
        "Could not import scan_coco_receiver_scaling_multimodel_200_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
EPS = 1e-12
SCRIPT_VERSION = "multimodel-targetselected-head-spatial-control-minboundary-v2"
MODEL_ORDER = (
    "qwen2-2b",
    "qwen-3b",
    "qwen-7b",
    "llava-7b",
    "llava-13b",
    "internvl-1b",
    "internvl-2b",
)
DATASET_ORDER = ("coco", "controlled_a")


# =============================================================================
# Generic helpers
# =============================================================================


def norm_rel(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left", "left of": "left", "left_of": "left", "l": "left",
        "right": "right", "right of": "right", "right_of": "right", "r": "right",
        "above": "above", "on": "above", "on top of": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "underneath": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s if s in REL else None)


def parse_generation(text: str) -> Optional[str]:
    s = str(text).lower().replace("_", " ")
    pats = [
        ("above", r"\bon top of\b"),
        ("left", r"\bto the left of\b"),
        ("right", r"\bto the right of\b"),
        ("below", r"\bunderneath\b"),
        ("below", r"\bbeneath\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder\b"),
        ("above", r"\babove\b"),
        ("above", r"\bover\b"),
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\bon\b"),
    ]
    hits: List[Tuple[int, str]] = []
    for lab, pat in pats:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), lab))
    return min(hits, key=lambda z: z[0])[1] if hits else None


def surface_for(rel: str, dataset: str) -> str:
    rel = str(rel)
    if dataset == "controlled_a":
        return {"left": "left", "right": "right", "above": "on", "below": "under"}[rel]
    return rel


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize a near-zero vector")
    return v / n


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def first_tensor(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if torch.is_tensor(x) and x.ndim == 3:
                return x
    if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        return out.last_hidden_state
    raise RuntimeError(f"Could not find 3D hidden tensor in output type={type(out)}")


def replace_first_tensor(out: Any, y: torch.Tensor) -> Any:
    if torch.is_tensor(out):
        return y
    if isinstance(out, tuple):
        xs = list(out)
        for i, x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim == 3:
                xs[i] = y
                return tuple(xs)
    if isinstance(out, list):
        xs = list(out)
        for i, x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim == 3:
                xs[i] = y
                return xs
    raise RuntimeError(f"Could not replace 3D hidden tensor in output type={type(out)}")


def extract_hidden_states(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
        getattr(getattr(outputs, "text_model_output", None), "hidden_states", None),
    ]
    for x in candidates:
        if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
            return tuple(x)
    raise RuntimeError("Forward pass did not expose decoder hidden_states")


def parse_name_list(text: str, allowed: Sequence[str]) -> List[str]:
    if str(text).strip().lower() == "all":
        return list(allowed)
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"Unknown values {bad}; allowed={list(allowed)}")
    return vals


def parse_pairs(text: Optional[str], models: Sequence[str], datasets: Sequence[str]) -> List[Tuple[str, str]]:
    if text:
        out = []
        for item in str(text).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Bad pair {item!r}; expected model:dataset")
            m, d = [x.strip() for x in item.split(":", 1)]
            if m not in MODEL_ORDER or d not in DATASET_ORDER:
                raise ValueError(f"Unsupported pair {m}:{d}")
            out.append((m, d))
        return out
    return [(m, d) for m in models for d in datasets]


def parse_layer_overrides(text: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    if not str(text).strip():
        return out
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad controller override {item!r}")
        model, spec = [x.strip() for x in item.split("=", 1)]
        if model not in MODEL_ORDER:
            raise ValueError(f"Unknown model in controller override: {model}")
        vals: List[int] = []
        for part in spec.split("+"):
            part = part.strip().upper().replace("L", "")
            if "-" in part:
                a, b = [int(x) for x in part.split("-", 1)]
                vals.extend(range(min(a, b), max(a, b) + 1))
            else:
                vals.append(int(part))
        out[model] = sorted(set(vals))
    return out


def relative7_layers(n_layers: int) -> List[int]:
    if n_layers < 7:
        raise RuntimeError(f"Need >=7 decoder layers, got {n_layers}")
    center = int(round((23.0 / 35.0) * (n_layers - 1)))
    start = center - 3
    start = max(0, min(start, n_layers - 7))
    return list(range(start, start + 7))


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", default="all")
    p.add_argument("--datasets", default="all")
    p.add_argument("--pairs", default=None)
    p.add_argument("--headscan-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    # Shared repo/data/model args expected by the head-scan loader.
    p.add_argument("--data-root", default="data")
    p.add_argument("--coco-prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--controlled-json", default="data/controlled_images_dataset.json")
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)
    p.add_argument("--target-max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--revision", default="main")
    p.add_argument("--control", default="gray", choices=["gray"], help="Head-scan control; cross-model protocol is Real-Gray.")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument("--internvl-use-thumbnail", action=argparse.BooleanOptionalAction, default=True)

    # Controller geometry / layers.
    p.add_argument("--geometry-control", default="gray", choices=["gray"], help="Synthetic residual geometry uses Real-Gray for cross-model consistency.")
    p.add_argument(
        "--controller-layers",
        default="",
        help="Optional per-model overrides, e.g. qwen-3b=20-26,llava-7b=17-23. Default is relative 7-layer window.",
    )
    p.add_argument("--geometry-cache-dir", default=None, help="Default: <output-dir>/geometry_cache")

    # Evaluation / minimum-distance optimization.
    p.add_argument("--modes", default="head,oracle", help="Comma-separated: head,oracle")
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all target samples")
    p.add_argument("--eval-seed", type=int, default=17)
    p.add_argument("--max-steps", type=int, default=60, help="Maximum local relinearization steps")
    p.add_argument(
        "--decision-margin-eps", type=float, default=1e-4,
        help="Target teacher-forced margin required over every competitor in each local boundary problem.",
    )
    p.add_argument(
        "--qp-feas-tol", type=float, default=1e-7,
        help="Feasibility tolerance for the tiny minimum-norm half-space QP.",
    )
    p.add_argument(
        "--qp-lambda-tol", type=float, default=1e-9,
        help="Tolerance for nonnegative KKT multipliers in the active-set QP solver.",
    )
    p.add_argument(
        "--min-residual-step", type=float, default=1e-8,
        help="Stop if the minimum local residual-space step is smaller than this.",
    )
    p.add_argument(
        "--max-residual-step", type=float, default=0.0,
        help="Optional trust cap on one local residual step; 0 disables the cap.",
    )
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--sequence-score-reduction", default="mean", choices=["mean", "sum"])
    return p.parse_args()


# =============================================================================
# Head-scan routing inputs
# =============================================================================


def validate_headscan_summary(path: Path, model: str, dataset: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scan_synthetic_frozen_direction_heads_7models_v2.py first."
        )
    s = json.loads(path.read_text(encoding="utf-8"))
    if str(s.get("model")) != model or str(s.get("dataset")) != dataset:
        raise RuntimeError(f"Head-scan summary mismatch at {path}")
    if bool(s.get("direction_fit_uses_target_gt", True)):
        raise RuntimeError(f"{path}: direction vectors are not source-only")
    if not bool(s.get("head_selection_uses_target_gt", False)):
        raise RuntimeError(f"{path}: requested protocol requires target-supervised head ID selection")
    frac = float(s.get("head_selection_frac", 0.0))
    if frac < 1.0 - 1e-9:
        raise RuntimeError(
            f"{path}: head_selection_frac={frac}; requested protocol uses the full target dataset to choose head ID"
        )
    return s


def load_routing(headscan_root: Path, model: str, dataset: str) -> Tuple[dict, Dict[int, dict]]:
    pair = headscan_root / model / dataset
    summary = validate_headscan_summary(pair / "summary.json", model, dataset)
    csv_path = pair / "samples_best_head.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    need = {"sid", "gt", "head_pred"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"{csv_path} missing columns {sorted(need - set(df.columns))}")
    by_sid: Dict[int, dict] = {}
    for r in df.to_dict("records"):
        sid = int(r["sid"])
        gt = norm_rel(r.get("gt"))
        hp = norm_rel(r.get("head_pred"))
        if gt not in REL or hp not in REL:
            continue
        r = dict(r)
        r["gt"] = gt
        r["head_pred"] = hp
        by_sid[sid] = r
    if not by_sid:
        raise RuntimeError(f"No usable routing rows in {csv_path}")
    return summary, by_sid


# =============================================================================
# Synthetic residual H/V geometry
# =============================================================================


def hidden_pair_from_standard(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    rendered = hs.headprobe.build_chat_prompt(processor, question, True)
    batch = hs.headprobe.process_inputs(processor, rendered, image, device)
    try:
        ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
        sub_raw = hs.headprobe.locate_phrase_positions(processor.tokenizer, ids, subject)
        ref_raw = hs.headprobe.locate_phrase_positions(processor.tokenizer, ids, reference)
        with torch.inference_mode():
            out = model(
                **batch,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        hs_tuple = extract_hidden_states(out)
        if len(hs_tuple) < len(decoder_layers) + 1:
            raise RuntimeError(
                f"hidden_states length={len(hs_tuple)} but decoder layers={len(decoder_layers)}"
            )
        merged_len = int(hs_tuple[0].shape[1])
        sub_pos, _ = recv.map_text_positions_to_decoder(
            model=model, processor=processor, input_ids=ids,
            raw_positions=sub_raw, merged_length=merged_len,
        )
        ref_pos, _ = recv.map_text_positions_to_decoder(
            model=model, processor=processor, input_ids=ids,
            raw_positions=ref_raw, merged_length=merged_len,
        )
        vals: List[np.ndarray] = []
        for L in selected_layers:
            h = hs_tuple[int(L) + 1][0]
            si = torch.as_tensor(sub_pos, device=h.device, dtype=torch.long)
            ri = torch.as_tensor(ref_pos, device=h.device, dtype=torch.long)
            pair = h.index_select(0, si).mean(0) - h.index_select(0, ri).mean(0)
            vals.append(pair.detach().float().cpu().numpy().astype(np.float32))
        return np.stack(vals, axis=0)
    finally:
        del batch
        with contextlib.suppress(Exception):
            del out


def internvl_prepared_inputs(backend: Any, image: Image.Image, question: str, subject: str, reference: str, args: argparse.Namespace):
    pixels, _layout = hs.native5.internvl_pixels_and_layout(backend, image, args)
    _query, input_ids, attention_mask, visual_positions = hs.native5.internvl_chat_query(
        backend, question, int(pixels.shape[0])
    )
    ids = input_ids[0].detach().cpu().tolist()
    start = max(visual_positions) + 1 if visual_positions else 0
    sub = hs.native5.token_span_for_phrase(backend.tokenizer, ids, subject, start=start)
    ref = hs.native5.token_span_for_phrase(backend.tokenizer, ids, reference, start=start)
    image_flags = torch.ones(
        (int(pixels.shape[0]), 1), device=backend.device, dtype=torch.long
    )
    batch = {
        "pixel_values": pixels,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_flags": image_flags,
    }
    return batch, list(map(int, sub)), list(map(int, ref))


def hidden_pair_from_internvl(
    *, backend: Any, decoder_layers: Sequence[Any], image: Image.Image, question: str,
    subject: str, reference: str, selected_layers: Sequence[int], args: argparse.Namespace,
) -> np.ndarray:
    batch, sub_pos, ref_pos = internvl_prepared_inputs(
        backend, image, question, subject, reference, args
    )
    try:
        with torch.inference_mode():
            out = backend.model(
                **batch,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        hs_tuple = extract_hidden_states(out)
        if len(hs_tuple) < len(decoder_layers) + 1:
            raise RuntimeError(
                f"InternVL hidden_states length={len(hs_tuple)} but decoder layers={len(decoder_layers)}"
            )
        vals = []
        for L in selected_layers:
            h = hs_tuple[int(L) + 1][0]
            si = torch.as_tensor(sub_pos, device=h.device, dtype=torch.long)
            ri = torch.as_tensor(ref_pos, device=h.device, dtype=torch.long)
            pair = h.index_select(0, si).mean(0) - h.index_select(0, ri).mean(0)
            vals.append(pair.detach().float().cpu().numpy().astype(np.float32))
        return np.stack(vals, axis=0)
    finally:
        for k in list(batch):
            with contextlib.suppress(Exception):
                del batch[k]
        with contextlib.suppress(Exception):
            del out


def build_synthetic_geometry_cache(
    *, model_alias: str, model: Any, processor_or_backend: Any, decoder_layers: Sequence[Any],
    layers: Sequence[int], source_rows: Sequence[Mapping[str, Any]], cache_path: Path,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not args.overwrite:
        with np.load(cache_path, allow_pickle=True) as z:
            old_layers = [int(x) for x in z["decoder_block_index"].tolist()]
            if old_layers != list(map(int, layers)):
                raise RuntimeError(
                    f"Geometry cache {cache_path} layers={old_layers}, expected={list(layers)}"
                )
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            y = np.asarray([norm_rel(x) for x in z["relation"].tolist()], dtype=object)
        print(f"[GEOMETRY CACHE] reuse {cache_path} X={X.shape}")
        return X, y

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    sids: List[int] = []
    errors_path = cache_path.with_suffix(".errors.jsonl")
    if args.overwrite and errors_path.exists():
        errors_path.unlink()

    device = torch.device(args.device)
    for row in tqdm(source_rows, desc=f"{model_alias}:Synthetic residual geometry"):
        img = gray = None
        try:
            img = Image.open(str(row["image_path"])).convert("RGB")
            gray = Image.new("RGB", img.size, (int(args.gray_value),) * 3)
            kwargs = dict(
                question=str(row["question_text"]),
                subject=str(row["subject"]),
                reference=str(row["reference"]),
                selected_layers=layers,
            )
            if model_alias.startswith("internvl-"):
                real = hidden_pair_from_internvl(
                    backend=processor_or_backend, decoder_layers=decoder_layers,
                    image=img, args=args, **kwargs,
                )
                ctrl = hidden_pair_from_internvl(
                    backend=processor_or_backend, decoder_layers=decoder_layers,
                    image=gray, args=args, **kwargs,
                )
            else:
                real = hidden_pair_from_standard(
                    model=model, processor=processor_or_backend, decoder_layers=decoder_layers,
                    image=img, device=device, **kwargs,
                )
                ctrl = hidden_pair_from_standard(
                    model=model, processor=processor_or_backend, decoder_layers=decoder_layers,
                    image=gray, device=device, **kwargs,
                )
            vectors.append((real - ctrl).astype(np.float32))
            labels.append(str(norm_rel(row["relation"])))
            sids.append(int(row["sid"]))
        except Exception as exc:
            append_jsonl(errors_path, {
                "sid": int(row.get("sid", -1)),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-40:],
            })
            tqdm.write(f"[GEOMETRY ERROR] {model_alias} sid={row.get('sid')}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if img is not None:
                img.close()
            if gray is not None:
                gray.close()
            cleanup_cuda()

    if len(vectors) < max(20, int(0.8 * len(source_rows))):
        raise RuntimeError(
            f"Too few successful Synthetic geometry samples for {model_alias}: {len(vectors)}/{len(source_rows)}"
        )
    X = np.stack(vectors, axis=0)
    y = np.asarray(labels, dtype=object)
    np.savez_compressed(
        cache_path,
        relation_vectors=X.astype(np.float16),
        relation=y,
        sample_index=np.asarray(sids, dtype=np.int64),
        decoder_block_index=np.asarray(layers, dtype=np.int16),
        vector_definition=np.asarray(
            "Residual-stream Real-Gray object-role q=(h_sub_real-h_ref_real)-(h_sub_gray-h_ref_gray)",
            dtype=object,
        ),
    )
    print(f"[GEOMETRY CACHE] saved {cache_path} X={X.shape}")
    return X, y


def fit_geometry(X: np.ndarray, y: np.ndarray, layers: Sequence[int]) -> Tuple[dict, pd.DataFrame]:
    """Fit only the 2D Synthetic spatial subspace at each layer.

    The class means define H/V directions, but their class-gap magnitudes do not
    set the intervention scale.  We orthonormalize the H/V span so Euclidean
    distance in the 2D control coordinates equals the actual L2 norm of the
    relation-residual edit at that layer.
    """
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
                raise RuntimeError(f"Synthetic geometry missing class={r}")
            means[r] = Xf[m].mean(0)

        class_dirs = {r: unit(means[r] - center) for r in REL}
        dH = unit(class_dirs["right"] - class_dirs["left"])
        dV = unit(class_dirs["above"] - class_dirs["below"])

        # Preserve the same H/V span, but use an orthonormal control basis.
        # qH is exactly the horizontal axis; qV is the component of the vertical
        # axis orthogonal to qH.  Thus ||Q @ a||_2 == ||a||_2.
        qH = dH.copy()
        v_perp = dV - float(np.dot(dV, qH)) * qH
        qV = unit(v_perp)
        if float(np.dot(qV, dV)) < 0:
            qV = -qV
        Q = np.stack([qH, qV], axis=1)
        B = np.stack([dH, dV], axis=1)

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
            "basis_H_dot_basis_V": float(np.dot(qH, qV)),
            "basis_H_align_axis_H": float(np.dot(qH, dH)),
            "basis_V_align_axis_V": float(np.dot(qV, dV)),
            "orthonormal_check_max_abs": float(np.max(np.abs(Q.T @ Q - np.eye(2)))),
        })
    return geom, pd.DataFrame(rows)


# =============================================================================
# Target sample preparation / sequence scoring
# =============================================================================

@dataclass
class Prepared:
    kind: str
    batch: Dict[str, Any]
    sub_pos: List[int]
    ref_pos: List[int]
    image: Image.Image
    question: str
    tokenizer: Any
    raw_prompt_len: int
    processor_or_backend: Any

    def close(self) -> None:
        for k in list(self.batch.keys()):
            with contextlib.suppress(Exception):
                del self.batch[k]


def prepare_standard(
    *, model: Any, processor: Any, decoder_layers: Sequence[Any], row: Mapping[str, Any],
    image: Image.Image, args: argparse.Namespace,
) -> Prepared:
    device = torch.device(args.device)
    batch = recv.build_batch(
        probe=hs.headprobe, processor=processor, question=str(row["question_text"]),
        image=image, device=device,
    )
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    sub_raw = hs.headprobe.locate_phrase_positions(processor.tokenizer, ids, str(row["subject"]))
    ref_raw = hs.headprobe.locate_phrase_positions(processor.tokenizer, ids, str(row["reference"]))
    # One cheap no-grad forward obtains the actual merged decoder length.
    with torch.inference_mode():
        out = model(
            **batch, output_hidden_states=True, output_attentions=False,
            use_cache=False, return_dict=True,
        )
    hst = extract_hidden_states(out)
    merged_len = int(hst[0].shape[1])
    sub_pos, _ = recv.map_text_positions_to_decoder(
        model=model, processor=processor, input_ids=ids,
        raw_positions=sub_raw, merged_length=merged_len,
    )
    ref_pos, _ = recv.map_text_positions_to_decoder(
        model=model, processor=processor, input_ids=ids,
        raw_positions=ref_raw, merged_length=merged_len,
    )
    del out, hst
    return Prepared(
        kind="standard", batch=dict(batch), sub_pos=sub_pos, ref_pos=ref_pos,
        image=image, question=str(row["question_text"]), tokenizer=processor.tokenizer,
        raw_prompt_len=int(batch["input_ids"].shape[1]), processor_or_backend=processor,
    )


def prepare_internvl(
    *, backend: Any, row: Mapping[str, Any], image: Image.Image, args: argparse.Namespace,
) -> Prepared:
    batch, sub_pos, ref_pos = internvl_prepared_inputs(
        backend, image, str(row["question_text"]), str(row["subject"]), str(row["reference"]), args
    )
    return Prepared(
        kind="internvl", batch=batch, sub_pos=sub_pos, ref_pos=ref_pos,
        image=image, question=str(row["question_text"]), tokenizer=backend.tokenizer,
        raw_prompt_len=int(batch["input_ids"].shape[1]), processor_or_backend=backend,
    )


def encode_candidates(tokenizer: Any, dataset: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for r in REL:
        surface = surface_for(r, dataset)
        ids = list(tokenizer(surface, add_special_tokens=False).input_ids)
        if not ids:
            ids = list(tokenizer(" " + surface, add_special_tokens=False).input_ids)
        if not ids:
            raise RuntimeError(f"Could not tokenize answer surface {surface!r}")
        out[r] = [int(x) for x in ids]
    return out


def extend_batch(batch: Mapping[str, Any], answer_ids: Sequence[int]) -> Dict[str, Any]:
    ids = batch["input_ids"]
    dev = ids.device
    ans = torch.as_tensor([list(map(int, answer_ids))], device=dev, dtype=ids.dtype)
    ext: Dict[str, Any] = {}
    for k, v in batch.items():
        if k in {"position_ids", "cache_position"}:
            continue
        ext[k] = v
    ext["input_ids"] = torch.cat([ids, ans], dim=1)
    if "attention_mask" in batch and torch.is_tensor(batch["attention_mask"]):
        am = batch["attention_mask"]
        extra = torch.ones((am.shape[0], len(answer_ids)), device=am.device, dtype=am.dtype)
        ext["attention_mask"] = torch.cat([am, extra], dim=1)
    return ext


def sequence_score_from_logits(logits: torch.Tensor, answer_ids: Sequence[int], reduction: str) -> torch.Tensor:
    n = len(answer_ids)
    if n <= 0:
        raise ValueError("empty answer ids")
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"Expected logits [1,T,V], got {tuple(logits.shape)}")
    seq_len = int(logits.shape[1])
    start = seq_len - n
    if start <= 0:
        raise RuntimeError(f"Bad candidate alignment seq_len={seq_len} n={n}")
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    vals = []
    for j, tok in enumerate(answer_ids):
        vals.append(logp[start + j - 1, int(tok)])
    s = torch.stack(vals)
    return s.mean() if reduction == "mean" else s.sum()


class SpatialPatch:
    """Patch only object-token residual states in selected decoder blocks."""
    def __init__(
        self, *, decoder_layers: Sequence[Any], layers: Sequence[int], geom: Mapping[int, Any],
        sub_pos: Sequence[int], ref_pos: Sequence[int], coords: torch.Tensor,
    ):
        self.handles = []
        self.layers = list(map(int, layers))
        self.geom = geom
        self.sub_pos = list(map(int, sub_pos))
        self.ref_pos = list(map(int, ref_pos))
        self.coords = coords
        for i, L in enumerate(self.layers):
            self.handles.append(decoder_layers[L].register_forward_hook(self._hook(i, L)))

    def _hook(self, i: int, L: int):
        def hook(_m, _inp, out):
            h = first_tensor(out)
            T = int(h.shape[1])
            if not self.sub_pos or not self.ref_pos:
                return None
            if max(self.sub_pos + self.ref_pos) >= T:
                # Cached autoregressive decode step (usually T=1): no object state exists.
                return None
            Q = torch.as_tensor(self.geom[L]["control_basis"], device=h.device, dtype=h.dtype)
            c = self.coords[i].to(device=h.device, dtype=h.dtype)
            dx = Q @ c
            y = h.clone()
            for p in self.sub_pos:
                y[0, p] = y[0, p] + 0.5 * dx
            for p in self.ref_pos:
                y[0, p] = y[0, p] - 0.5 * dx
            return replace_first_tensor(out, y)
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def score_and_grad(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    answer_ids: Sequence[int], layers: Sequence[int], geom: Mapping[int, Any],
    base_coords: np.ndarray, reduction: str,
) -> Tuple[float, np.ndarray]:
    ext = extend_batch(prepared.batch, answer_ids)
    dev = prepared.batch["input_ids"].device
    delta = torch.zeros((len(layers), 2), device=dev, dtype=torch.float32, requires_grad=True)
    base = torch.as_tensor(base_coords, device=dev, dtype=torch.float32)
    coords = base + delta
    with SpatialPatch(
        decoder_layers=decoder_layers, layers=layers, geom=geom,
        sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=coords,
    ):
        out = model(**ext, use_cache=False, return_dict=True)
        score = sequence_score_from_logits(out.logits, answer_ids, reduction)
        grad = torch.autograd.grad(score, delta, retain_graph=False, create_graph=False)[0]
    return float(score.detach().item()), grad.detach().float().cpu().numpy().astype(np.float64)


def score_only(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    answer_ids: Sequence[int], layers: Sequence[int], geom: Mapping[int, Any],
    coords: np.ndarray, reduction: str,
) -> float:
    ext = extend_batch(prepared.batch, answer_ids)
    dev = prepared.batch["input_ids"].device
    c = torch.as_tensor(coords, device=dev, dtype=torch.float32)
    with SpatialPatch(
        decoder_layers=decoder_layers, layers=layers, geom=geom,
        sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=c,
    ), torch.inference_mode():
        out = model(**ext, use_cache=False, return_dict=True)
        score = sequence_score_from_logits(out.logits, answer_ids, reduction)
    return float(score.detach().item())


def all_scores(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    candidate_ids: Mapping[str, Sequence[int]], layers: Sequence[int], geom: Mapping[int, Any],
    coords: np.ndarray, reduction: str,
) -> Dict[str, float]:
    return {
        r: score_only(
            model=model, decoder_layers=decoder_layers, prepared=prepared,
            answer_ids=candidate_ids[r], layers=layers, geom=geom, coords=coords,
            reduction=reduction,
        )
        for r in REL
    }


def generate_with_coords(
    *, model_alias: str, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    layers: Sequence[int], geom: Mapping[int, Any], coords: np.ndarray, args: argparse.Namespace,
) -> Tuple[Optional[str], str]:
    dev = prepared.batch["input_ids"].device
    c = torch.as_tensor(coords, device=dev, dtype=torch.float32)
    with SpatialPatch(
        decoder_layers=decoder_layers, layers=layers, geom=geom,
        sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=c,
    ):
        if model_alias.startswith("internvl-"):
            # Native chat rebuilds the same deterministic prompt; the absolute object
            # positions prepared above therefore match its prefill sequence.
            text = hs.native5.internvl_generate(
                prepared.processor_or_backend, prepared.image, prepared.question, args
            )
        else:
            with torch.inference_mode():
                generated = model.generate(
                    **prepared.batch, do_sample=False, num_beams=1,
                    max_new_tokens=int(args.max_new_tokens), use_cache=True,
                )
            new = generated[0, prepared.raw_prompt_len:] if int(generated.shape[1]) > prepared.raw_prompt_len else generated[0]
            text = prepared.tokenizer.decode(new, skip_special_tokens=True).strip()
            del generated
    return parse_generation(text), str(text)


def min_margin(scores: Mapping[str, float], target: str) -> float:
    return min(float(scores[target]) - float(scores[r]) for r in REL if r != target)


def argmax_rel(scores: Mapping[str, float]) -> str:
    return max(REL, key=lambda r: float(scores[r]))


def solve_min_norm_halfspaces(
    A: np.ndarray,
    b: np.ndarray,
    *,
    feas_tol: float,
    lambda_tol: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Solve min 0.5*||x||^2 subject to A x >= b.

    There are only three competitor constraints, so an exact active-set
    enumeration is simpler and more transparent than adding a QP dependency.
    For an active set S, KKT gives x=A_S^T lambda and
    (A_S A_S^T) lambda=b_S with lambda>=0.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError(f"Bad QP shapes A={A.shape} b={b.shape}")
    n = int(A.shape[1])
    zero = np.zeros((n,), dtype=np.float64)
    if np.all(A @ zero >= b - float(feas_tol)):
        return zero, {
            "active_constraints": [],
            "predicted_step_norm": 0.0,
            "max_linearized_violation": float(np.max(np.maximum(b, 0.0))) if len(b) else 0.0,
        }

    best_x: Optional[np.ndarray] = None
    best_active: Optional[Tuple[int, ...]] = None
    best_norm = float("inf")
    m = int(A.shape[0])

    for k in range(1, m + 1):
        for active in itertools.combinations(range(m), k):
            idx = np.asarray(active, dtype=np.int64)
            As = A[idx]
            bs = b[idx]
            gram = As @ As.T
            lam = np.linalg.pinv(gram, rcond=1e-12) @ bs
            if np.any(lam < -float(lambda_tol)):
                continue
            lam = np.maximum(lam, 0.0)
            x = As.T @ lam

            # Active constraints should be tight up to numerical tolerance.
            active_err = float(np.max(np.abs(As @ x - bs))) if len(active) else 0.0
            if active_err > max(1e-6, 10.0 * float(feas_tol)):
                continue
            if not np.all(A @ x >= b - float(feas_tol)):
                continue

            nrm = float(np.linalg.norm(x))
            if nrm < best_norm:
                best_norm = nrm
                best_x = x
                best_active = tuple(int(i) for i in active)

    if best_x is None:
        return None, {
            "active_constraints": None,
            "predicted_step_norm": float("nan"),
            "max_linearized_violation": float("nan"),
        }

    violation = np.maximum(b - A @ best_x, 0.0)
    return best_x, {
        "active_constraints": list(best_active or ()),
        "predicted_step_norm": float(best_norm),
        "max_linearized_violation": float(np.max(violation)) if len(violation) else 0.0,
    }


# =============================================================================
# Optimization / evaluation
# =============================================================================


def optimize_to_target(
    *, model_alias: str, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    candidate_ids: Mapping[str, Sequence[int]], layers: Sequence[int], geom: Mapping[int, Any],
    target: str, base_scores: Mapping[str, float], base_gen_pred: Optional[str],
    base_gen_text: str, args: argparse.Namespace,
) -> Dict[str, Any]:
    """Iteratively cross the nearest local target-decision boundary.

    The optimization variable contains two orthonormal spatial-subspace
    coordinates per layer. Because each layer basis is orthonormal, the L2 norm
    of the flattened coordinate update is exactly the total L2 norm of the
    corresponding relation-residual edits across layers.
    """
    coords = np.zeros((len(layers), 2), dtype=np.float64)
    scores = dict(base_scores)
    gen_pred = base_gen_pred
    gen_text = base_gen_text
    steps = 0
    qp_failed = False
    capped_steps = 0
    stop_reason = "max_steps"
    last_active_names: List[str] = []
    last_step_norm = 0.0

    if gen_pred == target:
        stop_reason = "already_target"
    else:
        for step_idx in range(1, int(args.max_steps) + 1):
            # Exact local scores and gradients wrt all layer-wise spatial coords.
            sg: Dict[str, Tuple[float, np.ndarray]] = {}
            for r in REL:
                sg[r] = score_and_grad(
                    model=model, decoder_layers=decoder_layers, prepared=prepared,
                    answer_ids=candidate_ids[r], layers=layers, geom=geom,
                    base_coords=coords, reduction=args.sequence_score_reduction,
                )
            scores = {r: float(sg[r][0]) for r in REL}

            competitors = [r for r in REL if r != target]
            margins = np.asarray(
                [float(scores[target]) - float(scores[r]) for r in competitors],
                dtype=np.float64,
            )
            A = np.stack(
                [
                    (sg[target][1] - sg[r][1]).reshape(-1)
                    for r in competitors
                ],
                axis=0,
            )
            b = float(args.decision_margin_eps) - margins

            # If the differentiable surrogate already prefers the target over
            # every competitor but actual generation does not, there is no
            # nonzero minimum boundary-crossing step under this surrogate.
            # Stop rather than introducing an arbitrary extra steering distance.
            if np.all(b <= float(args.qp_feas_tol)):
                stop_reason = "tf_boundary_crossed_generation_mismatch"
                break

            flat_step, qp_info = solve_min_norm_halfspaces(
                A, b,
                feas_tol=float(args.qp_feas_tol),
                lambda_tol=float(args.qp_lambda_tol),
            )
            if flat_step is None:
                qp_failed = True
                stop_reason = "local_boundary_qp_infeasible"
                break

            step_norm = float(np.linalg.norm(flat_step))
            if step_norm < float(args.min_residual_step):
                stop_reason = "minimum_residual_step_too_small"
                break

            max_step = float(args.max_residual_step)
            if max_step > 0.0 and step_norm > max_step:
                flat_step = flat_step * (max_step / step_norm)
                step_norm = max_step
                capped_steps += 1

            step = flat_step.reshape(len(layers), 2)
            coords = coords + step
            steps = step_idx
            last_step_norm = float(step_norm)
            active_idx = qp_info.get("active_constraints") or []
            last_active_names = [competitors[int(i)] for i in active_idx]

            # The discrete success criterion is always actual autoregressive
            # generation, not merely the teacher-forced boundary surrogate.
            gen_pred, gen_text = generate_with_coords(
                model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                prepared=prepared, layers=layers, geom=geom, coords=coords, args=args,
            )
            if gen_pred == target:
                stop_reason = "generation_target"
                break

    # Always perform one final exact generation and teacher-forced evaluation
    # under the final spatial patch.
    final_pred, final_text = generate_with_coords(
        model_alias=model_alias, model=model, decoder_layers=decoder_layers,
        prepared=prepared, layers=layers, geom=geom, coords=coords, args=args,
    )
    final_scores = all_scores(
        model=model, decoder_layers=decoder_layers, prepared=prepared,
        candidate_ids=candidate_ids, layers=layers, geom=geom,
        coords=coords, reduction=args.sequence_score_reduction,
    )
    layer_norms = np.linalg.norm(coords, axis=1)
    return {
        "patched_prediction": final_pred,
        "patched_generation_text": final_text,
        "final_tf_prediction": argmax_rel(final_scores),
        "final_target_margin": min_margin(final_scores, target),
        "steps_taken": int(steps),
        "final_total_residual_norm": float(np.linalg.norm(coords.reshape(-1))),
        "final_layer_residual_norms": [float(x) for x in layer_norms.tolist()],
        "final_control_coords": coords.astype(np.float64).tolist(),
        "last_boundary_step_norm": float(last_step_norm),
        "last_active_competitors": list(last_active_names),
        "qp_failed": bool(qp_failed),
        "capped_steps": int(capped_steps),
        "stop_reason": stop_reason,
    }


def stratified_cap(rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> List[Mapping[str, Any]]:
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    by: Dict[str, List[int]] = {r: [] for r in REL}
    for i, row in enumerate(rows):
        gt = norm_rel(row.get("relation"))
        if gt in by:
            by[gt].append(i)
    chosen: List[int] = []
    # proportional stratified sampling
    for r in REL:
        idx = np.asarray(by[r], dtype=np.int64)
        if not len(idx):
            continue
        rng.shuffle(idx)
        k = int(round(n * len(idx) / len(rows)))
        chosen.extend(idx[: min(k, len(idx))].tolist())
    chosen = sorted(set(chosen))
    leftovers = [i for i in range(len(rows)) if i not in set(chosen)]
    rng.shuffle(leftovers)
    chosen.extend(leftovers[: max(0, n - len(chosen))])
    return [rows[i] for i in sorted(chosen[:n])]


def summarize_pair(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, g in df.groupby("mode", sort=False):
        base = g["baseline_correct"].astype(bool).to_numpy()
        patch = g["patched_correct"].astype(bool).to_numpy()
        head = g["head_correct"].astype(bool).to_numpy()
        comply = g["patched_matches_control_target"].astype(bool).to_numpy()
        rows.append({
            "mode": mode,
            "N": int(len(g)),
            "baseline_accuracy": float(base.mean()),
            "head_readout_accuracy": float(head.mean()),
            "patched_generation_accuracy": float(patch.mean()),
            "gain_vs_baseline": float(patch.mean() - base.mean()),
            "target_compliance": float(comply.mean()),
            "wrong_to_correct": int(np.sum((~base) & patch)),
            "correct_to_wrong": int(np.sum(base & (~patch))),
            "net": int(np.sum((~base) & patch) - np.sum(base & (~patch))),
            "mean_steps": float(g["steps_taken"].mean()),
            "mean_final_total_residual_norm": float(g["final_total_residual_norm"].mean()),
            "fraction_qp_failed": float(g["qp_failed"].astype(bool).mean()),
            "fraction_tf_boundary_generation_mismatch": float(
                (g["stop_reason"] == "tf_boundary_crossed_generation_mismatch").mean()
            ),
        })
    return pd.DataFrame(rows)


def run_pair(
    *, model_alias: str, dataset: str, model: Any, processor_or_backend: Any,
    decoder_layers: Sequence[Any], layers: Sequence[int], geom: Mapping[int, Any],
    args: argparse.Namespace, root_out: Path,
) -> dict:
    headscan_root = Path(args.headscan_dir)
    hs_summary, routing = load_routing(headscan_root, model_alias, dataset)
    target_rows = hs.load_target(dataset, args)
    target_rows = [r for r in target_rows if int(r["sid"]) in routing]
    target_rows = stratified_cap(target_rows, int(args.eval_max_samples), int(args.eval_seed))
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

    existing = read_jsonl(result_path)
    done = {(int(r["sid"]), str(r["mode"])) for r in existing}
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    bad = [x for x in modes if x not in {"head", "oracle"}]
    if bad:
        raise ValueError(f"Unknown modes {bad}")

    tokenizer = processor_or_backend.tokenizer if model_alias.startswith("internvl-") else processor_or_backend.tokenizer
    candidate_ids = encode_candidates(tokenizer, dataset)

    print("\n" + "=" * 180)
    print(f"CONTROL {model_alias} / {dataset} | selected_head={hs_summary.get('best_head')} | head_acc(all)={float(hs_summary.get('best_all_target_acc', float('nan'))):.4f}")
    print(f"Synthetic residual geometry layers={list(layers)} | eval N={len(target_rows)}")
    print("Synthetic directions -> target GT selects head ID -> head prediction sets target -> minimum residual H/V boundary crossing -> actual generation")
    print("=" * 180, flush=True)

    for row in tqdm(target_rows, desc=f"CONTROL {model_alias}:{dataset}"):
        sid = int(row["sid"])
        rr = routing[sid]
        gt = norm_rel(row["relation"])
        head_pred = norm_rel(rr["head_pred"])
        if gt not in REL or head_pred not in REL:
            continue
        image = prepared = None
        try:
            image = Image.open(str(row["image_path"])).convert("RGB")
            if model_alias.startswith("internvl-"):
                prepared = prepare_internvl(
                    backend=processor_or_backend, row=row, image=image, args=args
                )
            else:
                prepared = prepare_standard(
                    model=model, processor=processor_or_backend,
                    decoder_layers=decoder_layers, row=row, image=image, args=args
                )

            zero = np.zeros((len(layers), 2), dtype=np.float64)
            base_scores = all_scores(
                model=model, decoder_layers=decoder_layers, prepared=prepared,
                candidate_ids=candidate_ids, layers=layers, geom=geom,
                coords=zero, reduction=args.sequence_score_reduction,
            )
            base_pred, base_text = generate_with_coords(
                model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                prepared=prepared, layers=layers, geom=geom, coords=zero, args=args,
            )

            for mode in modes:
                if (sid, mode) in done:
                    continue
                control_target = head_pred if mode == "head" else gt
                res = optimize_to_target(
                    model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                    prepared=prepared, candidate_ids=candidate_ids, layers=layers,
                    geom=geom, target=control_target, base_scores=base_scores,
                    base_gen_pred=base_pred, base_gen_text=base_text, args=args,
                )
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
                    **res,
                }
                append_jsonl(result_path, outrow)
                done.add((sid, mode))
        except Exception as exc:
            append_jsonl(error_path, {
                "sid": sid, "model": model_alias, "dataset": dataset,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-60:],
            })
            tqdm.write(f"[ERROR] {model_alias}/{dataset} sid={sid}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if prepared is not None:
                prepared.close()
            if image is not None:
                image.close()
            cleanup_cuda()

    all_rows = read_jsonl(result_path)
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
        "headscan_dir": str(headscan_root),
        "selected_head": hs_summary.get("best_head"),
        "selected_head_target_acc": hs_summary.get("best_all_target_acc"),
        "direction_source": "Synthetic-400 only",
        "head_id_selection": "full target GT",
        "per_sample_head_routing_uses_target_gt": False,
        "residual_geometry_source": "Synthetic-400 only",
        "geometry_control": args.geometry_control,
        "controller_layers": list(map(int, layers)),
        "controller_dim": int(2 * len(layers)),
        "optimizer": "iterative_minimum_residual_boundary_crossing",
        "spatial_basis": "orthonormal basis spanning Synthetic-400 H/V directions",
        "uses_natural_gap_scaling": False,
        "decision_margin_eps": float(args.decision_margin_eps),
        "qp_feas_tol": float(args.qp_feas_tol),
        "qp_lambda_tol": float(args.qp_lambda_tol),
        "min_residual_step": float(args.min_residual_step),
        "max_residual_step": float(args.max_residual_step),
        "max_steps": int(args.max_steps),
    }
    write_json(outdir / "metadata.json", meta)
    print("\nPAIR SUMMARY")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Return head + head condition + oracle condition in one row for cross-model table.
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


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    seed_all(int(args.seed))
    models = parse_name_list(args.models, MODEL_ORDER)
    datasets = parse_name_list(args.datasets, DATASET_ORDER)
    pairs = parse_pairs(args.pairs, models, datasets)
    overrides = parse_layer_overrides(args.controller_layers)
    root_out = Path(args.output_dir)
    if args.overwrite and root_out.exists():
        shutil.rmtree(root_out)
    root_out.mkdir(parents=True, exist_ok=True)
    geom_root = Path(args.geometry_cache_dir) if args.geometry_cache_dir else root_out / "geometry_cache"

    # Early protocol validation: do not spend GPU hours if a pair was selected with
    # a different head-selection rule.
    valid_pairs: List[Tuple[str, str]] = []
    preflight_failures: List[dict] = []
    for m, d in pairs:
        try:
            validate_headscan_summary(Path(args.headscan_dir) / m / d / "summary.json", m, d)
            valid_pairs.append((m, d))
        except Exception as exc:
            preflight_failures.append({"model": m, "dataset": d, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[PREFLIGHT SKIP] {m}/{d}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
    write_json(root_out / "preflight_failures.json", preflight_failures)
    if not valid_pairs:
        raise RuntimeError("No valid model/dataset pairs after head-scan preflight")

    source_rows = hs.load_synthetic(args)
    summaries: List[dict] = []
    failures: List[dict] = list(preflight_failures)

    for model_alias in MODEL_ORDER:
        model_pairs = [(m, d) for m, d in valid_pairs if m == model_alias]
        if not model_pairs:
            continue
        model = processor = None
        try:
            model, processor, decoder_layers, _nh, _hd, spec = hs.load_model_bundle(model_alias, args)
            # Gradients are needed only for control coordinates, never for weights.
            for p in model.parameters():
                p.requires_grad_(False)
            n_layers = len(decoder_layers)
            layers = overrides.get(model_alias, relative7_layers(n_layers))
            if len(layers) != 7:
                raise RuntimeError(
                    f"{model_alias}: controller must use exactly 7 layers for 14D comparability; got {layers}"
                )
            if min(layers) < 0 or max(layers) >= n_layers:
                raise RuntimeError(f"{model_alias}: invalid controller layers {layers} for n_layers={n_layers}")
            print(f"\n[MODEL CONTROLLER] {model_alias}: n_layers={n_layers}, layers={layers}")

            geom_cache = geom_root / model_alias / "synthetic400_residual_real_minus_gray.npz"
            Xg, yg = build_synthetic_geometry_cache(
                model_alias=model_alias, model=model, processor_or_backend=processor,
                decoder_layers=decoder_layers, layers=layers, source_rows=source_rows,
                cache_path=geom_cache, args=args,
            )
            geom, axis_df = fit_geometry(Xg, yg, layers)
            model_geom_dir = geom_root / model_alias
            model_geom_dir.mkdir(parents=True, exist_ok=True)
            axis_df.to_csv(model_geom_dir / "synthetic400_HV_geometry.csv", index=False)

            for _m, dataset in model_pairs:
                try:
                    s = run_pair(
                        model_alias=model_alias, dataset=dataset, model=model,
                        processor_or_backend=processor, decoder_layers=decoder_layers,
                        layers=layers, geom=geom, args=args, root_out=root_out,
                    )
                    summaries = [x for x in summaries if not (x.get("model") == model_alias and x.get("dataset") == dataset)]
                    summaries.append(s)
                    write_csv(root_out / "cross_model_control_summary.csv", summaries)
                    write_json(root_out / "cross_model_control_summary.json", summaries)
                except Exception as exc:
                    failure = {
                        "model": model_alias, "dataset": dataset,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-60:],
                    }
                    failures.append(failure)
                    write_json(root_out / "failures.json", failures)
                    print(f"[PAIR FAILED] {model_alias}/{dataset}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise
        except Exception as exc:
            failure = {
                "model": model_alias, "dataset": "__model_or_geometry__",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-60:],
            }
            failures.append(failure)
            write_json(root_out / "failures.json", failures)
            print(f"[MODEL FAILED] {model_alias}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            cleanup_cuda()

    write_csv(root_out / "cross_model_control_summary.csv", summaries)
    write_json(root_out / "cross_model_control_summary.json", summaries)
    write_json(root_out / "failures.json", failures)

    print("\n" + "=" * 180)
    print("FINAL CROSS-MODEL CONTROL SUMMARY")
    print("=" * 180)
    if summaries:
        print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("NO SUCCESSFUL PAIRS")
    print(f"\nSaved to: {root_out}")


if __name__ == "__main__":
    main()
