#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_synthetic_frozen_direction_heads_7models_v2.py

Fast multi-model / multi-dataset spatial-head scan.

PURPOSE
-------
This script is the cheap cross-model replacement for the expensive full causal
head ranking.  It implements the protocol agreed for the broad model/dataset
sweep:

  1) SOURCE (Synthetic-400 only): for every attention head, extract a
     vision-dependent subject-reference vector before W_O and fit a frozen
     four-way relation code (left/right/above/below).

  2) TARGET (COCO or Controlled-A): NEVER refit the relation directions.
     Apply the frozen Synthetic-400 code to every head and compute direction
     accuracy on the target dataset.

  3) HEAD SELECTION: target GT is used ONLY to choose the head whose frozen
     Synthetic code has the highest target direction accuracy.  Thus this is a
     target-supervised head-selection diagnostic, NOT a label-free discovery
     method.  The spatial directions themselves remain source-only.

  4) OPTIONAL BASELINE: run ordinary greedy generation on the same target
     samples and report best-head accuracy, baseline accuracy, W2C/C2W/net.


CONTROLLER-COMPATIBLE OUTPUT CONTRACT
-------------------------------------
This file is intended to run before:

  eval_multimodel_targetselected_head_spatial_control_v1.py

For each successful model/dataset pair it writes exactly:

  <output-dir>/<model>/<dataset>/summary.json
  <output-dir>/<model>/<dataset>/samples_best_head.csv

The downstream controller expects:
  * direction_fit_uses_target_gt = false
  * head_selection_uses_target_gt = true
  * head_selection_frac = 1.0
  * samples_best_head.csv contains sid, gt, head_pred

Protocol:
  Synthetic-400 determines each head's frozen L/R/A/B direction code.
  The FULL target dataset (COCO or Controlled-A) determines the head ID.
  Per-sample routing later uses the selected head's own prediction.

Target generation is disabled by default in this scanner. The downstream
controller recomputes actual generation itself, so generation here is redundant.
Keeping it off also prevents a model-specific generate() failure from discarding
an otherwise valid target head-vector sample.

Supported models
----------------
  qwen2-2b    -> Qwen/Qwen2-VL-2B-Instruct
  qwen-3b     -> Qwen/Qwen2.5-VL-3B-Instruct
  qwen-7b     -> Qwen/Qwen2.5-VL-7B-Instruct
  llava-7b    -> llava-hf/llava-1.5-7b-hf
  llava-13b   -> llava-hf/llava-1.5-13b-hf
  internvl-1b -> OpenGVLab/InternVL2_5-1B  (native InternVL2.5)
  internvl-2b -> OpenGVLab/InternVL2_5-2B  (native InternVL2.5)

Supported targets
-----------------
  coco         -> repo COCO_two four-way set
  controlled_a -> Controlled Images A (on->above, under->below internally)

Vector definition
-----------------
For each layer/head and each condition:

  pair = z_preWO(subject) - z_preWO(reference)

Default vision-dependent vector:

  v = pair(real image) - pair(gray image)

Use --control noimage to instead use Real-NoImage, matching older probes.
The same control is always used on Synthetic and target.

IMPORTANT METHOD LABEL
----------------------
If --head-selection-frac=1.0 (default), the same target set is used to rank
heads and report the best-head target accuracy.  This is intentionally an
oracle/supervised HEAD-SELECTION diagnostic, although relation directions are
still frozen from Synthetic-400.  For a cleaner held-out head-selection test,
set e.g. --head-selection-frac 0.15; the head is selected on a stratified 15%
and evaluated frozen on the remaining 85%.

Run from the AdaptVis llava16 repository root.

Examples
--------
# Exact newly requested sweep (Qwen3B+COCO was already run separately):
CUDA_VISIBLE_DEVICES=0 python scan_synthetic_frozen_direction_heads_7models_v2.py \
  --pairs qwen-7b:coco,qwen-3b:controlled_a,qwen-7b:controlled_a,llava-7b:coco,llava-7b:controlled_a,llava-13b:coco,llava-13b:controlled_a \
  --output-dir output/syn400_direction_head_multimodel_v1

# Controller-compatible full 7-model x 2-dataset matrix:
CUDA_VISIBLE_DEVICES=0 python -u scan_synthetic_frozen_direction_heads_7models_v2.py \
  --models all --datasets all \
  --head-selection-frac 1.0 \
  --control gray \
  --no-with-generation \
  --output-dir output/syn400_direction_head_7models_targetfull_v2

# Add the three newly requested model families to an existing v1 output directory:
CUDA_VISIBLE_DEVICES=0 python scan_synthetic_frozen_direction_heads_7models_v2.py \
  --pairs qwen2-2b:coco,qwen2-2b:controlled_a,internvl-1b:coco,internvl-1b:controlled_a,internvl-2b:coco,internvl-2b:controlled_a \
  --output-dir output/syn400_direction_head_multimodel_v1

# Cleaner 15% target-head-selection -> 85% held-out test:
CUDA_VISIBLE_DEVICES=0 python scan_synthetic_frozen_direction_heads_7models_v2.py \
  --models qwen-7b,llava-7b,llava-13b --datasets coco,controlled_a \
  --head-selection-frac 0.15 \
  --output-dir output/syn400_direction_head_multimodel_holdout15
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import random
import re
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

# Repo-native helpers. Run from AdaptVis llava16 root.
try:
    import extract_two_object_relation_states as base
except Exception as exc:
    raise SystemExit(
        "Could not import extract_two_object_relation_states.py. "
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import analyze_coco_head_object_residual_direction_probe_v1 as headprobe
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_head_object_residual_direction_probe_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import analyze_coco_attention_flow_swap_step1_v1 as attncent
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_attention_flow_swap_step1_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import eval_crossdataset_late_causal_qwen25_v4_auto_vgroot as causal
except Exception as exc:
    raise SystemExit(
        "Could not import eval_crossdataset_late_causal_qwen25_v4_auto_vgroot.py.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import eval_synthetic_fixedwindow_centroid_5models_3datasets_v1 as native5
except Exception as exc:
    raise SystemExit(
        "Could not import eval_synthetic_fixedwindow_centroid_5models_3datasets_v1.py; "
        "it is required for native InternVL2.5 loading/preprocessing.\n"
        f"{type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "syn400-frozen-direction-head-7models-v2-controller-compatible"
EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}

MODEL_ORDER = ("qwen2-2b", "qwen-3b", "qwen-7b", "llava-7b", "llava-13b", "internvl-1b", "internvl-2b")
DATASET_ORDER = ("coco", "controlled_a")

REL_ALIASES = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "top": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "bottom": "below",
}

SYN_PROMPT = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with left, right, above, or below."
)

CONTROLLED_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} in the image. "
    "Answer with left, right, on, or under."
)


# =============================================================================
# Generic attention helpers (adds InternLM2 `attention.wo` support)
# =============================================================================

def resolve_self_attention_any(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        obj = getattr(layer, name, None)
        if obj is not None:
            return obj
    raise RuntimeError(f"Could not resolve self-attention for {type(layer).__name__}")


def resolve_o_proj_any(attn: Any) -> torch.nn.Module:
    for name in ("o_proj", "out_proj", "proj", "wo"):
        obj = getattr(attn, name, None)
        if isinstance(obj, torch.nn.Module):
            return obj
    raise RuntimeError(f"Could not resolve attention output projection for {type(attn).__name__}")


def config_num_heads_any(model: Any) -> Optional[int]:
    objs = [
        model,
        getattr(model, "config", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "config", None), "language_config", None),
        getattr(getattr(model, "config", None), "llm_config", None),
    ]
    for obj in objs:
        if obj is None:
            continue
        for attr in ("num_attention_heads", "num_heads", "n_head"):
            value = getattr(obj, attr, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
    return None


def scan_shape_any(model: Any, layers: Sequence[Any]) -> Tuple[int, int]:
    nh_ref = hd_ref = None
    for li, layer in enumerate(layers):
        attn = resolve_self_attention_any(layer)
        op = resolve_o_proj_any(attn)
        nh = None
        for attr in ("num_heads", "num_attention_heads", "n_heads"):
            value = getattr(attn, attr, None)
            if value is not None:
                try:
                    nh = int(value)
                    break
                except Exception:
                    pass
        if nh is None:
            nh = config_num_heads_any(model)
        if nh is None:
            raise RuntimeError(f"L{li}: could not determine attention head count")
        width = getattr(op, "in_features", None)
        if width is None and hasattr(op, "weight"):
            width = int(op.weight.shape[1])
        if width is None:
            raise RuntimeError(f"L{li}: could not infer pre-output-projection width")
        width = int(width)
        if width % nh:
            raise RuntimeError(f"L{li}: output-proj input width={width} not divisible by heads={nh}")
        hd = width // nh
        if nh_ref is None:
            nh_ref, hd_ref = nh, hd
        elif (nh, hd) != (nh_ref, hd_ref):
            raise RuntimeError(
                f"Non-uniform head shape: first={(nh_ref, hd_ref)} L{li}={(nh, hd)}"
            )
    assert nh_ref is not None and hd_ref is not None
    return int(nh_ref), int(hd_ref)


class CapturePreWOAny:
    def __init__(
        self,
        layers: Sequence[Any],
        n_heads: int,
        head_dim: int,
        a_pos: Sequence[int],
        b_pos: Sequence[int],
        pool: str,
    ):
        self.layers = layers
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.a_pos = list(map(int, a_pos))
        self.b_pos = list(map(int, b_pos))
        self.pool = str(pool)
        self.out = torch.empty(
            (len(layers), self.n_heads, 2, self.head_dim), dtype=torch.float32
        )
        self.seen = set()
        self.handles = []

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            op = resolve_o_proj_any(resolve_self_attention_any(layer))

            def make_hook(layer_idx: int):
                def hook(_module, inputs):
                    x = inputs[0]
                    a = headprobe.pool_positions(x, self.a_pos, self.pool).view(
                        self.n_heads, self.head_dim
                    )
                    b = headprobe.pool_positions(x, self.b_pos, self.pool).view(
                        self.n_heads, self.head_dim
                    )
                    self.out[layer_idx, :, 0] = a.detach().float().cpu()
                    self.out[layer_idx, :, 1] = b.detach().float().cpu()
                    self.seen.add(layer_idx)

                return hook

            self.handles.append(op.register_forward_pre_hook(make_hook(li)))
        return self

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __exit__(self, *_):
        self.close()

    def finalize(self) -> np.ndarray:
        missing = [i for i in range(len(self.layers)) if i not in self.seen]
        if missing:
            raise RuntimeError(f"Missing pre-W_O captures: {missing[:10]}")
        return self.out.numpy()


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", default="all",
                   help="all or comma-separated: " + ",".join(MODEL_ORDER))
    p.add_argument("--datasets", default="all",
                   help="all or comma-separated: " + ",".join(DATASET_ORDER))
    p.add_argument(
        "--pairs",
        default=None,
        help=(
            "Optional comma-separated exact model:dataset pairs; overrides --models/--datasets. "
            "Example qwen-7b:coco,llava-13b:controlled_a"
        ),
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Native InternVL2.5 dtype; Qwen/LLaVA keep their repo model-spec dtype.")
    p.add_argument("--revision", default="main")
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument("--internvl-use-thumbnail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--control", default="gray", choices=["gray", "noimage"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--keep-fp32", action="store_true")

    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--coco-prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--controlled-json", default="data/controlled_images_dataset.json")
    p.add_argument("--target-max-samples", type=int, default=None)

    p.add_argument(
        "--head-selection-frac",
        type=float,
        default=1.0,
        help=(
            "Fraction of target GT used to select the best head. 1.0 = scan/select/evaluate on all "
            "target samples (diagnostic). Values <1 use a stratified selection split and report the "
            "chosen head on the held-out remainder."
        ),
    )
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument(
        "--min-target-vector-success-rate",
        type=float,
        default=0.95,
        help=(
            "Minimum fraction of requested target examples that must yield usable "
            "head vectors before head selection is accepted."
        ),
    )

    p.add_argument(
        "--with-generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional old baseline diagnostic. Default=False because "
            "eval_multimodel_targetselected_head_spatial_control_v1.py "
            "recomputes actual generation itself."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument(
        "--cache-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache extracted Synthetic/target head vectors as compressed NPZ files.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If false, save error info and continue to the next pair/model.",
    )
    return p.parse_args()


def norm_relation(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(s, s)


def head_name(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head):02d}"


def seed_all(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_acc(pred: Sequence[Any], gt: Sequence[Any]) -> float:
    if len(gt) == 0:
        return float("nan")
    return float(np.mean(np.asarray(pred, dtype=object) == np.asarray(gt, dtype=object)))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(str(k))
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list(value: str, allowed: Sequence[str]) -> List[str]:
    if str(value).strip().lower() == "all":
        return list(allowed)
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"Unknown values {bad}; allowed={list(allowed)}")
    return out


def canonical_model_alias(name: str) -> str:
    x = str(name).strip().lower()
    return {
        "qwen22b": "qwen2-2b",
        "qwen2vl-2b": "qwen2-2b",
        "internvl2.5-1b": "internvl-1b",
        "internvl2.5-2b": "internvl-2b",
    }.get(x, x)

def resolve_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.pairs:
        out: List[Tuple[str, str]] = []
        for item in args.pairs.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Bad --pairs item {item!r}; expected model:dataset")
            model, dataset = [x.strip() for x in item.split(":", 1)]
            model = canonical_model_alias(model)
            if model not in MODEL_ORDER:
                raise ValueError(f"Unknown model in --pairs: {model}")
            if dataset not in DATASET_ORDER:
                raise ValueError(f"Unknown dataset in --pairs: {dataset}")
            if (model, dataset) not in out:
                out.append((model, dataset))
        if not out:
            raise ValueError("--pairs produced no valid pairs")
        return out
    if str(args.models).strip().lower() == "all":
        models = list(MODEL_ORDER)
    else:
        models = [canonical_model_alias(x) for x in str(args.models).split(",") if x.strip()]
        bad = [x for x in models if x not in MODEL_ORDER]
        if bad:
            raise ValueError(f"Unknown models {bad}; allowed={list(MODEL_ORDER)}")
    datasets = parse_list(args.datasets, DATASET_ORDER)
    return [(m, d) for m in models for d in datasets]


# =============================================================================
# Data
# =============================================================================

def load_synthetic(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.synthetic_dir)
    labels_path = Path(args.synthetic_labels) if args.synthetic_labels else root / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    rows: List[Dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rel = norm_relation(item["relation"])
            if rel not in RELATIONS:
                raise RuntimeError(f"{labels_path}:{line_no}: unsupported relation={item['relation']!r}")
            image_value = Path(str(item["image"]))
            image_path = image_value if image_value.is_absolute() else root / image_value
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()
            rows.append({
                "sid": int(item.get("id", len(rows))),
                "dataset": "synthetic",
                "image_path": str(image_path),
                "subject": subject,
                "reference": reference,
                "relation": rel,
                "question_text": SYN_PROMPT.format(subject=subject, reference=reference),
            })
    rows.sort(key=lambda r: int(r["sid"]))
    if args.source_max_samples is not None:
        rows = rows[: int(args.source_max_samples)]
    counts = Counter(r["relation"] for r in rows)
    missing = [r for r in RELATIONS if counts[r] == 0]
    if missing:
        raise RuntimeError(f"Synthetic source missing classes {missing}; counts={dict(counts)}")
    print(f"[Synthetic] n={len(rows)} counts={dict(counts)}")
    return rows


def load_coco(args: argparse.Namespace) -> List[Dict[str, Any]]:
    records, audit = base.load_records("coco_two", Path(args.data_root), args.target_max_samples)
    prompts = attncent.load_standard_prompts(Path(args.coco_prompt_jsonl))
    rows: List[Dict[str, Any]] = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        rel = norm_relation(rec.relation)
        if rel not in RELATIONS:
            continue
        rows.append({
            "sid": sid,
            "dataset": "coco",
            "image_path": str(rec.image_path),
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "relation": rel,
            "question_text": str(p["question_text"]),
        })
    if args.target_max_samples is not None:
        rows = rows[: int(args.target_max_samples)]
    print(
        f"[COCO] n={len(rows)} counts={dict(Counter(r['relation'] for r in rows))} "
        f"audit={len(audit)}"
    )
    return rows


def load_controlled_a(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw = causal.load_controlled_a(args)
    rows: List[Dict[str, Any]] = []
    for r in raw:
        rel = norm_relation(r["relation"])
        if rel not in RELATIONS:
            continue
        path = Path(str(r["image_path"]))
        if not path.exists():
            continue
        subject = str(r["subject"])
        reference = str(r["reference"])
        rows.append({
            "sid": int(r["sid"]),
            "dataset": "controlled_a",
            "image_path": str(path),
            "subject": subject,
            "reference": reference,
            "relation": rel,
            "question_text": CONTROLLED_PROMPT.format(subject=subject, reference=reference),
        })
    if args.target_max_samples is not None:
        rows = rows[: int(args.target_max_samples)]
    print(f"[Controlled-A] n={len(rows)} counts={dict(Counter(r['relation'] for r in rows))}")
    return rows


def load_target(dataset: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    if dataset == "coco":
        return load_coco(args)
    if dataset == "controlled_a":
        return load_controlled_a(args)
    raise ValueError(dataset)


# =============================================================================
# Model / extraction
# =============================================================================

def load_model_bundle(model_alias: str, args: argparse.Namespace):
    # Native InternVL2.5 uses the already-established repo-native backend from
    # eval_synthetic_fixedwindow_centroid_5models_3datasets_v1.py.  Do NOT use
    # extract_two_object_relation_states.SPECS for InternVL because those old
    # aliases point at InternVL3-HF.
    if model_alias in ("internvl-1b", "internvl-2b"):
        if args.control == "noimage":
            raise ValueError(
                "Native InternVL2.5 in this script supports --control gray only. "
                "Use the default --control gray for the cross-model sweep."
            )
        backend = native5.load_backend(model_alias, args)
        model = backend.model
        layers = backend.layers
        n_heads, head_dim = scan_shape_any(model, layers)
        print(
            f"[MODEL] decoder={backend.decoder_path} layers={len(layers)} "
            f"heads/layer={n_heads} head_dim={head_dim} total_heads={len(layers)*n_heads}"
        )
        # For InternVL, `processor` is deliberately the native Backend object;
        # extraction/generation dispatch on model_alias and use its tokenizer,
        # preprocessing, and chat prompt implementation.
        return model, backend, layers, n_heads, head_dim, {"repo_id": backend.repo_id, "backend": "internvl25"}

    if model_alias not in base.SPECS:
        raise KeyError(f"Model alias {model_alias!r} not present in extract_two_object_relation_states.SPECS")
    spec = base.SPECS[model_alias]
    cls = getattr(transformers, spec.model_class, None)
    if cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class} "
            f"required for {model_alias}"
        )
    kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl
    print(f"\n[LOAD] {model_alias} -> {spec.repo_id}")
    try:
        model = cls.from_pretrained(spec.repo_id, **kwargs)
    except TypeError:
        if "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
        model = cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    # Pin slow processors for reproducibility; this also avoids the Qwen fast-
    # processor default change changing the existing protocol silently.
    try:
        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code, use_fast=False
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
    base.configure_processor(model, processor)
    layers, layer_path = headprobe.resolve_decoder_layers(model)
    n_heads, head_dim = headprobe.scan_shape(model, layers)
    print(
        f"[MODEL] decoder={layer_path} layers={len(layers)} "
        f"heads/layer={n_heads} head_dim={head_dim} total_heads={len(layers)*n_heads}"
    )
    return model, processor, layers, n_heads, head_dim, spec


def internvl_capture_condition(
    backend: Any,
    layers: Sequence[Any],
    n_heads: int,
    head_dim: int,
    question: str,
    subject: str,
    reference: str,
    image: Image.Image,
    pool: str,
    args: argparse.Namespace,
) -> np.ndarray:
    """Capture native InternVL2.5 pre-W_O per-head subject/reference vectors."""
    pixels, _layout = native5.internvl_pixels_and_layout(backend, image, args)
    _query, input_ids, attention_mask, visual_positions = native5.internvl_chat_query(
        backend, question, int(pixels.shape[0])
    )
    ids = input_ids[0].detach().cpu().tolist()
    search_start = max(visual_positions) + 1 if visual_positions else 0
    a_pos = native5.token_span_for_phrase(backend.tokenizer, ids, subject, start=search_start)
    b_pos = native5.token_span_for_phrase(backend.tokenizer, ids, reference, start=search_start)
    cap = CapturePreWOAny(layers, n_heads, head_dim, a_pos, b_pos, pool)
    try:
        with cap:
            with torch.inference_mode():
                image_flags = torch.ones(
                    (int(pixels.shape[0]), 1),
                    device=backend.device,
                    dtype=torch.long,
                )
                backend.model(
                    pixel_values=pixels,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        out = cap.finalize()
    finally:
        cap.close()
        with contextlib.suppress(Exception):
            del pixels, input_ids, attention_mask, image_flags
    return out

def make_gray(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, (v, v, v))


def parse_generated_relation(text: str) -> Optional[str]:
    s = str(text).lower()
    candidates = [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("above", r"\bon\s+top\s+of\b"),
        ("above", r"\bover\b"),
        ("above", r"\bon\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
        ("below", r"\bbeneath\b"),
    ]
    hits: List[Tuple[int, str]] = []
    for rel, pat in candidates:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))
    if not hits:
        return None
    return sorted(hits, key=lambda x: x[0])[0][1]


def greedy_generate(
    model: Any,
    processor: Any,
    device: torch.device,
    question: str,
    image: Image.Image,
    max_new_tokens: int,
    model_alias: str,
    args: argparse.Namespace,
) -> Tuple[Optional[str], str]:
    if model_alias in ("internvl-1b", "internvl-2b"):
        # Here processor is the native5.Backend object.
        text = native5.internvl_generate(processor, image, question, args)
        return parse_generated_relation(text), text

    rendered = headprobe.build_chat_prompt(processor, question, True)
    batch = headprobe.process_inputs(processor, rendered, image, device)
    try:
        with torch.inference_mode():
            out = model.generate(
                **batch,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        input_len = int(batch["input_ids"].shape[1])
        ids = out[0, input_len:]
        text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return parse_generated_relation(text), text
    finally:
        del batch

def cache_metadata_matches(meta: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for k, v in expected.items():
        if meta.get(k) != v:
            return False
    return True


def save_vector_cache(
    path: Path,
    vectors: np.ndarray,
    sids: Sequence[int],
    relations: Sequence[str],
    baseline_pred: Sequence[Optional[str]],
    baseline_text: Sequence[str],
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=vectors,
        sid=np.asarray(sids, dtype=np.int64),
        relation=np.asarray(relations, dtype=object),
        baseline_pred=np.asarray(["" if x is None else str(x) for x in baseline_pred], dtype=object),
        baseline_text=np.asarray(list(baseline_text), dtype=object),
        metadata_json=np.asarray(json.dumps(dict(metadata)), dtype=object),
    )


def load_vector_cache(path: Path, expected: Mapping[str, Any]):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        if not cache_metadata_matches(meta, expected):
            return None
        baseline_pred = [str(x) if str(x) else None for x in z["baseline_pred"].tolist()]
        return {
            "vectors": z["vectors"],
            "sid": z["sid"].astype(np.int64),
            "relation": np.asarray(z["relation"].tolist(), dtype=object),
            "baseline_pred": baseline_pred,
            "baseline_text": [str(x) for x in z["baseline_text"].tolist()],
            "metadata": meta,
        }


def extract_vectors(
    records: Sequence[Mapping[str, Any]],
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    n_heads: int,
    head_dim: int,
    model_alias: str,
    dataset_name: str,
    args: argparse.Namespace,
    cache_path: Optional[Path],
    with_generation: bool,
) -> Dict[str, Any]:
    expected_meta = {
        "model": model_alias,
        "dataset": dataset_name,
        "control": args.control,
        "gray_value": int(args.gray_value),
        "pool": args.pool,
        "n_layers": int(len(layers)),
        "n_heads": int(n_heads),
        "head_dim": int(head_dim),
        "with_generation": bool(with_generation),
        "max_new_tokens": int(args.max_new_tokens) if with_generation else None,
    }
    if args.cache_vectors and not args.overwrite and cache_path is not None:
        cached = load_vector_cache(cache_path, expected_meta)
        if cached is not None:
            print(f"[CACHE] loaded {cache_path} | n={len(cached['relation'])}")
            return cached

    device = torch.device(args.device)
    dtype_np = np.float32 if args.keep_fp32 else np.float16
    vectors: List[np.ndarray] = []
    sids: List[int] = []
    relations: List[str] = []
    baseline_pred: List[Optional[str]] = []
    baseline_text: List[str] = []
    errors: List[Dict[str, Any]] = []

    desc = f"{model_alias}:{dataset_name}:extract"
    for rec in tqdm(records, desc=desc):
        image: Optional[Image.Image] = None
        ctrl_image: Optional[Image.Image] = None
        try:
            image = Image.open(str(rec["image_path"])).convert("RGB")
            question = str(rec["question_text"])
            subject = str(rec["subject"])
            reference = str(rec["reference"])

            if model_alias in ("internvl-1b", "internvl-2b"):
                z_real = internvl_capture_condition(
                    processor, layers, n_heads, head_dim, question, subject, reference,
                    image, args.pool, args
                )
                if args.control != "gray":
                    raise ValueError("InternVL2.5 supports --control gray only in this script")
                ctrl_image = make_gray(image, args.gray_value)
                z_ctrl = internvl_capture_condition(
                    processor, layers, n_heads, head_dim, question, subject, reference,
                    ctrl_image, args.pool, args
                )
            else:
                z_real = headprobe.capture_condition(
                    model, processor, device, layers, n_heads, head_dim,
                    question, subject, reference, image, args.pool,
                )
                if args.control == "gray":
                    ctrl_image = make_gray(image, args.gray_value)
                    z_ctrl = headprobe.capture_condition(
                        model, processor, device, layers, n_heads, head_dim,
                        question, subject, reference, ctrl_image, args.pool,
                    )
                else:
                    z_ctrl = headprobe.capture_condition(
                        model, processor, device, layers, n_heads, head_dim,
                        question, subject, reference, None, args.pool,
                    )

            pair_real = z_real[:, :, 0, :] - z_real[:, :, 1, :]
            pair_ctrl = z_ctrl[:, :, 0, :] - z_ctrl[:, :, 1, :]
            residual = pair_real - pair_ctrl

            pred: Optional[str] = None
            text = ""
            if with_generation:
                pred, text = greedy_generate(
                    model, processor, device, question, image, args.max_new_tokens,
                    model_alias, args
                )

            sids.append(int(rec["sid"]))
            relations.append(norm_relation(rec["relation"]))
            vectors.append(residual.astype(dtype_np, copy=False))
            baseline_pred.append(pred)
            baseline_text.append(text)

            del z_real, z_ctrl, pair_real, pair_ctrl, residual
        except Exception as exc:
            errors.append({
                "sid": int(rec.get("sid", -1)),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            })
            tqdm.write(f"[ERROR] {model_alias}/{dataset_name} sid={rec.get('sid')}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if ctrl_image is not None:
                ctrl_image.close()
            if image is not None:
                image.close()
            cleanup_cuda()

    if not vectors:
        raise RuntimeError(f"No successful samples for {model_alias}/{dataset_name}")

    X = np.stack(vectors, axis=0)
    result = {
        "vectors": X,
        "sid": np.asarray(sids, dtype=np.int64),
        "relation": np.asarray(relations, dtype=object),
        "baseline_pred": baseline_pred,
        "baseline_text": baseline_text,
        "metadata": {**expected_meta, "script_version": SCRIPT_VERSION, "n": int(len(relations)), "errors": len(errors)},
        "errors": errors,
    }
    if args.cache_vectors and cache_path is not None:
        save_vector_cache(
            cache_path, X, sids, relations, baseline_pred, baseline_text,
            result["metadata"],
        )
        write_json(cache_path.with_suffix(".errors.json"), errors)
        print(f"[CACHE] saved {cache_path}")
    return result


# =============================================================================
# Frozen Synthetic codebooks / target ranking
# =============================================================================

def normalize_last(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, EPS)


def fit_source_codebooks(X: np.ndarray, y: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: [N,L,H,D]
    returns:
      center [L,H,D]
      dirs   [L,H,4,D]
    """
    y_arr = np.asarray(y, dtype=object)
    center = np.asarray(X, dtype=np.float32).mean(axis=0)
    L, H, D = center.shape
    dirs = np.empty((L, H, len(RELATIONS), D), dtype=np.float32)
    for ri, rel in enumerate(RELATIONS):
        mask = (y_arr == rel)
        if not np.any(mask):
            raise RuntimeError(f"Synthetic source has no {rel} samples")
        d = np.asarray(X[mask], dtype=np.float32).mean(axis=0) - center
        dirs[:, :, ri, :] = normalize_last(d)
    return center.astype(np.float32), dirs


def predict_one_head(
    X: np.ndarray,
    center: np.ndarray,
    dirs: np.ndarray,
    layer: int,
    head: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(X[:, layer, head, :], dtype=np.float32) - center[layer, head][None, :]
    x = normalize_last(x)
    scores = x @ dirs[layer, head].T
    idx = np.argmax(scores, axis=1)
    pred = np.asarray([RELATIONS[int(i)] for i in idx], dtype=object)
    margin = np.partition(scores, -2, axis=1)[:, -1] - np.partition(scores, -2, axis=1)[:, -2]
    return pred, margin


def make_stratified_split(y: Sequence[str], frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if frac >= 1.0 - 1e-12:
        idx = np.arange(n, dtype=np.int64)
        return idx, idx.copy()
    if not (0.0 < frac < 1.0):
        raise ValueError("--head-selection-frac must be in (0,1]")
    rng = random.Random(seed)
    select: List[int] = []
    test: List[int] = []
    y_arr = np.asarray(y, dtype=object)
    for rel in RELATIONS:
        ids = np.flatnonzero(y_arr == rel).tolist()
        rng.shuffle(ids)
        k = max(1, int(round(len(ids) * frac)))
        if k >= len(ids) and len(ids) > 1:
            k = len(ids) - 1
        select.extend(ids[:k])
        test.extend(ids[k:])
    select = sorted(select)
    test = sorted(test)
    if not test:
        raise RuntimeError("Held-out split is empty; increase dataset size or use frac=1")
    return np.asarray(select, dtype=np.int64), np.asarray(test, dtype=np.int64)


def per_relation_accuracy(pred: np.ndarray, gt: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rel in RELATIONS:
        m = idx[gt[idx] == rel]
        out[rel] = safe_acc(pred[m], gt[m]) if len(m) else float("nan")
    return out


def rank_heads(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    center: np.ndarray,
    dirs: np.ndarray,
    selection_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]]:
    N, L, H, D = X_target.shape
    if center.shape != (L, H, D):
        raise RuntimeError(
            f"Source/target head shape mismatch: source center={center.shape}, target={(L,H,D)}"
        )
    rows: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for l in tqdm(range(L), desc="rank frozen-Synthetic heads", leave=False):
        for h in range(H):
            pred_t, margin_t = predict_one_head(X_target, center, dirs, l, h)
            pred_s, _ = predict_one_head(X_source, center, dirs, l, h)
            predictions[(l, h)] = (pred_t, margin_t)
            pr = per_relation_accuracy(pred_t, y_target, np.arange(N, dtype=np.int64))
            rows.append({
                "layer": l,
                "head": h,
                "head_name": head_name(l, h),
                "syn_self_acc": safe_acc(pred_s, y_source),
                "selection_acc": safe_acc(pred_t[selection_idx], y_target[selection_idx]),
                "test_acc": safe_acc(pred_t[test_idx], y_target[test_idx]),
                "all_target_acc": safe_acc(pred_t, y_target),
                "left_acc": pr["left"],
                "right_acc": pr["right"],
                "above_acc": pr["above"],
                "below_acc": pr["below"],
                "mean_margin": float(np.mean(margin_t)),
                "selection_mean_margin": float(np.mean(margin_t[selection_idx])),
                "test_mean_margin": float(np.mean(margin_t[test_idx])),
            })
    rows.sort(
        key=lambda r: (
            -float(r["selection_acc"]),
            -float(r["syn_self_acc"]),
            -float(r["selection_mean_margin"]),
            int(r["layer"]),
            int(r["head"]),
        )
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, predictions


# =============================================================================
# Evaluation / reporting
# =============================================================================

def compare_with_baseline(
    best_pred: np.ndarray,
    gt: np.ndarray,
    baseline_pred: Sequence[Optional[str]],
    eval_idx: np.ndarray,
) -> Dict[str, Any]:
    if not baseline_pred or all(x is None for x in baseline_pred):
        return {
            "baseline_acc": float("nan"),
            "baseline_parse_coverage": 0.0,
            "head_acc": safe_acc(best_pred[eval_idx], gt[eval_idx]),
            "gain": float("nan"),
            "W2C": 0,
            "C2W": 0,
            "net": 0,
        }
    b = np.asarray(["__none__" if x is None else str(x) for x in baseline_pred], dtype=object)
    base_ok = (b[eval_idx] == gt[eval_idx])
    head_ok = (best_pred[eval_idx] == gt[eval_idx])
    w2c = int(np.sum((~base_ok) & head_ok))
    c2w = int(np.sum(base_ok & (~head_ok)))
    base_acc = float(np.mean(base_ok))
    head_acc = float(np.mean(head_ok))
    coverage = float(np.mean(np.asarray([x is not None for x in baseline_pred], dtype=bool)[eval_idx]))
    return {
        "baseline_acc": base_acc,
        "baseline_parse_coverage": coverage,
        "head_acc": head_acc,
        "gain": head_acc - base_acc,
        "W2C": w2c,
        "C2W": c2w,
        "net": w2c - c2w,
    }


def print_top_heads(rows: Sequence[Mapping[str, Any]], top_k: int) -> None:
    print("\n" + "=" * 132)
    print("TARGET-SUPERVISED HEAD SELECTION -- DIRECTIONS FROZEN FROM SYNTHETIC-400")
    print("=" * 132)
    print(
        f"{'rank':>4s} {'head':>8s} {'select':>8s} {'test':>8s} {'all':>8s} "
        f"{'synAcc':>8s} {'L':>7s} {'R':>7s} {'A':>7s} {'B':>7s} {'margin':>9s}"
    )
    for r in list(rows)[: int(top_k)]:
        print(
            f"{int(r['rank']):4d} {str(r['head_name']):>8s} "
            f"{float(r['selection_acc']):8.4f} {float(r['test_acc']):8.4f} "
            f"{float(r['all_target_acc']):8.4f} {float(r['syn_self_acc']):8.4f} "
            f"{float(r['left_acc']):7.4f} {float(r['right_acc']):7.4f} "
            f"{float(r['above_acc']):7.4f} {float(r['below_acc']):7.4f} "
            f"{float(r['mean_margin']):9.4f}"
        )


def run_pair(
    model_alias: str,
    dataset: str,
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    n_heads: int,
    head_dim: int,
    source_pack: Mapping[str, Any],
    center: np.ndarray,
    dirs: np.ndarray,
    args: argparse.Namespace,
    root_out: Path,
) -> Dict[str, Any]:
    print("\n" + "#" * 132)
    print(f"PAIR: {model_alias} + {dataset}")
    print("#" * 132)
    records = load_target(dataset, args)
    pair_out = root_out / model_alias / dataset
    pair_out.mkdir(parents=True, exist_ok=True)
    cache_path = pair_out / "target_head_vectors.npz"
    target_pack = extract_vectors(
        records, model, processor, layers, n_heads, head_dim,
        model_alias, dataset, args, cache_path,
        with_generation=bool(args.with_generation),
    )

    Xs = np.asarray(source_pack["vectors"])
    ys = np.asarray(source_pack["relation"], dtype=object)
    Xt = np.asarray(target_pack["vectors"])
    yt = np.asarray(target_pack["relation"], dtype=object)

    target_success_rate = float(len(yt) / max(1, len(records)))
    print(
        f"[TARGET VECTOR COVERAGE] {len(yt)}/{len(records)} "
        f"= {target_success_rate:.4f}"
    )
    if target_success_rate < float(args.min_target_vector_success_rate):
        raise RuntimeError(
            f"{model_alias}/{dataset}: target head-vector success rate "
            f"{target_success_rate:.4f} < required "
            f"{float(args.min_target_vector_success_rate):.4f}. "
            "Inspect extraction errors before selecting a head ID."
        )

    selection_idx, test_idx = make_stratified_split(
        yt, float(args.head_selection_frac), int(args.seed)
    )
    print(
        f"[HEAD SELECTION] selection_n={len(selection_idx)} test_n={len(test_idx)} "
        f"frac={args.head_selection_frac:.3f}"
    )
    if args.head_selection_frac >= 1.0 - 1e-12:
        print(
            "[NOTE] target GT is used on the full target set to choose the best head; "
            "this is a supervised head-selection diagnostic. Spatial directions remain Synthetic-only."
        )

    ranking, preds = rank_heads(Xs, ys, Xt, yt, center, dirs, selection_idx, test_idx)
    print_top_heads(ranking, args.top_k)
    write_csv(pair_out / "head_ranking.csv", ranking)

    best = ranking[0]
    key = (int(best["layer"]), int(best["head"]))
    best_pred, best_margin = preds[key]
    eval_idx = test_idx if args.head_selection_frac < 1.0 - 1e-12 else np.arange(len(yt), dtype=np.int64)
    comp = compare_with_baseline(
        best_pred, yt, target_pack.get("baseline_pred", []), eval_idx
    )

    print("\nBEST HEAD")
    print(
        f"{best['head_name']} | select={best['selection_acc']:.4f} "
        f"test={best['test_acc']:.4f} all={best['all_target_acc']:.4f} "
        f"synAcc={best['syn_self_acc']:.4f}"
    )
    if args.with_generation:
        print(
            f"EVAL vs generation | N={len(eval_idx)} | base={comp['baseline_acc']:.4f} "
            f"head={comp['head_acc']:.4f} gain={comp['gain']:+.4f} | "
            f"W2C={comp['W2C']} C2W={comp['C2W']} net={comp['net']} | "
            f"parse_cov={comp['baseline_parse_coverage']:.4f}"
        )

    eval_set = set(int(i) for i in eval_idx.tolist())
    sample_rows: List[Dict[str, Any]] = []
    baseline = target_pack.get("baseline_pred", [None] * len(yt))
    texts = target_pack.get("baseline_text", [""] * len(yt))
    sids = target_pack["sid"]
    for i in range(len(yt)):
        bp = baseline[i] if i < len(baseline) else None
        sample_rows.append({
            "row_index": i,
            "sid": int(sids[i]),
            "split": "eval" if i in eval_set else "selection",
            "gt": str(yt[i]),
            "best_head": str(best["head_name"]),
            "head_pred": str(best_pred[i]),
            "head_correct": int(best_pred[i] == yt[i]),
            "head_margin": float(best_margin[i]),
            "baseline_pred": "" if bp is None else str(bp),
            "baseline_correct": int(bp == yt[i]) if bp is not None else 0,
            "baseline_text": texts[i] if i < len(texts) else "",
        })
    write_csv(pair_out / "samples_best_head.csv", sample_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model_alias,
        "dataset": dataset,
        "control": args.control,
        "source": "synthetic_shapes_4dir_400",
        "direction_fit_uses_target_gt": False,
        "head_selection_uses_target_gt": True,
        "head_selection_frac": float(args.head_selection_frac),
        "controller_compatible": True,
        "controller_expected_consumer": "eval_multimodel_targetselected_head_spatial_control_v1.py",
        "target_requested_n": int(len(records)),
        "target_vector_n": int(len(yt)),
        "target_vector_success_rate": float(target_success_rate),
        "selection_n": int(len(selection_idx)),
        "eval_n": int(len(eval_idx)),
        "target_n": int(len(yt)),
        "best_head": str(best["head_name"]),
        "best_layer": int(best["layer"]),
        "best_head_index": int(best["head"]),
        "best_syn_self_acc": float(best["syn_self_acc"]),
        "best_selection_acc": float(best["selection_acc"]),
        "best_test_acc": float(best["test_acc"]),
        "best_all_target_acc": float(best["all_target_acc"]),
        "baseline_acc_eval": float(comp["baseline_acc"]),
        "best_head_acc_eval": float(comp["head_acc"]),
        "gain_over_generation_eval": float(comp["gain"]),
        "W2C_eval": int(comp["W2C"]),
        "C2W_eval": int(comp["C2W"]),
        "net_eval": int(comp["net"]),
        "baseline_parse_coverage_eval": float(comp["baseline_parse_coverage"]),
        "top_heads": ranking[: int(args.top_k)],
    }
    write_json(pair_out / "summary.json", summary)
    return summary


def summary_row(s: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model": s["model"],
        "dataset": s["dataset"],
        "N": s["target_n"],
        "selection_frac": s["head_selection_frac"],
        "best_head": s["best_head"],
        "syn_acc": s["best_syn_self_acc"],
        "selection_acc": s["best_selection_acc"],
        "test_acc": s["best_test_acc"],
        "all_target_acc": s["best_all_target_acc"],
        "baseline_acc": s["baseline_acc_eval"],
        "head_acc_eval": s["best_head_acc_eval"],
        "gain": s["gain_over_generation_eval"],
        "W2C": s["W2C_eval"],
        "C2W": s["C2W_eval"],
        "net": s["net_eval"],
        "parse_cov": s["baseline_parse_coverage_eval"],
        "control": s["control"],
    }


def print_cross_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    print("\n" + "=" * 170)
    print("CROSS-MODEL / CROSS-DATASET SUMMARY")
    print("=" * 170)
    print(
        f"{'model':<12s} {'dataset':<14s} {'N':>5s} {'head':>8s} {'synAcc':>8s} "
        f"{'select':>8s} {'test':>8s} {'all':>8s} {'base':>8s} {'evalHead':>9s} "
        f"{'gain':>8s} {'W2C':>5s} {'C2W':>5s} {'net':>5s}"
    )
    for r in rows:
        def fnum(k: str) -> str:
            try:
                v = float(r[k])
                return "   nan  " if not math.isfinite(v) else f"{v:8.4f}"
            except Exception:
                return "   nan  "
        print(
            f"{str(r['model']):<12s} {str(r['dataset']):<14s} {int(r['N']):5d} "
            f"{str(r['best_head']):>8s} {fnum('syn_acc')} {fnum('selection_acc')} "
            f"{fnum('test_acc')} {fnum('all_target_acc')} {fnum('baseline_acc')} "
            f"{fnum('head_acc_eval')} {fnum('gain')} "
            f"{int(r['W2C']):5d} {int(r['C2W']):5d} {int(r['net']):5d}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if not (0.0 < float(args.head_selection_frac) <= 1.0):
        raise ValueError("--head-selection-frac must be in (0,1]")

    # Exact protocol required by eval_multimodel_targetselected_head_spatial_control_v1.py.
    if float(args.head_selection_frac) < 1.0 - 1e-12:
        raise ValueError(
            "Controller-compatible scan requires --head-selection-frac 1.0: "
            "the full target dataset must choose the head ID."
        )
    if str(args.control) != "gray":
        raise ValueError(
            "Controller-compatible cross-model scan requires --control gray "
            "to match the Synthetic Real-Gray residual controller."
        )

    seed_all(args.seed)
    root_out = Path(args.output_dir)
    if args.overwrite and root_out.exists():
        shutil.rmtree(root_out)
    root_out.mkdir(parents=True, exist_ok=True)

    pairs = resolve_pairs(args)
    models_in_order = [m for m in MODEL_ORDER if any(pm == m for pm, _ in pairs)]
    print(f"[PLAN] pairs={pairs}")
    print(f"[PROTOCOL] control={args.control} | source directions=Synthetic-400 only | "
          f"target GT used for head selection={args.head_selection_frac:.3f}")

    source_records = load_synthetic(args)
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    # Append mode: preserve prior v1/v2 pair summaries in the same output dir.
    # Re-running a pair replaces its old row below.
    existing_summary = root_out / "cross_model_summary.json"
    if existing_summary.exists() and not args.overwrite:
        try:
            old = json.loads(existing_summary.read_text(encoding="utf-8"))
            if isinstance(old, list):
                summaries = [dict(x) for x in old if isinstance(x, dict)]
                print(f"[APPEND] loaded {len(summaries)} existing pair summaries")
        except Exception as exc:
            print(f"[WARN] could not load existing summary for append: {exc}")

    for model_alias in models_in_order:
        model = processor = None
        try:
            model, processor, layers, n_heads, head_dim, spec = load_model_bundle(model_alias, args)
            model_out = root_out / model_alias
            model_out.mkdir(parents=True, exist_ok=True)

            source_cache = model_out / "synthetic400_head_vectors.npz"
            source_pack = extract_vectors(
                source_records, model, processor, layers, n_heads, head_dim,
                model_alias, "synthetic", args, source_cache,
                with_generation=False,
            )
            Xs = np.asarray(source_pack["vectors"])
            ys = np.asarray(source_pack["relation"], dtype=object)
            center, dirs = fit_source_codebooks(Xs, ys)
            np.savez_compressed(
                model_out / "synthetic400_frozen_codebooks.npz",
                center=center.astype(np.float32),
                directions=dirs.astype(np.float32),
                relations=np.asarray(RELATIONS, dtype=object),
                vector_definition=np.asarray(
                    f"pre-W_O per-head [(subject-reference)_real - (subject-reference)_{args.control}]",
                    dtype=object,
                ),
            )

            model_datasets = [d for m, d in pairs if m == model_alias]
            for dataset in model_datasets:
                try:
                    summary = run_pair(
                        model_alias, dataset, model, processor, layers, n_heads, head_dim,
                        source_pack, center, dirs, args, root_out,
                    )
                    summaries = [s for s in summaries if not (s.get("model") == model_alias and s.get("dataset") == dataset)]
                    summaries.append(summary)
                    rows = [summary_row(s) for s in summaries]
                    write_csv(root_out / "cross_model_summary.csv", rows)
                    write_json(root_out / "cross_model_summary.json", summaries)
                    print_cross_summary(rows)
                except Exception as exc:
                    failure = {
                        "model": model_alias,
                        "dataset": dataset,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-30:],
                    }
                    failures.append(failure)
                    write_json(root_out / "failures.json", failures)
                    print(f"[PAIR FAILED] {model_alias}/{dataset}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise
        except Exception as exc:
            failure = {
                "model": model_alias,
                "dataset": "__model_or_source__",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-30:],
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

    rows = [summary_row(s) for s in summaries]
    write_csv(root_out / "cross_model_summary.csv", rows)
    write_json(root_out / "cross_model_summary.json", summaries)
    write_json(root_out / "failures.json", failures)

    expected_pairs = [{"model": m, "dataset": d} for m, d in pairs]
    successful_pairs = [
        {"model": str(s["model"]), "dataset": str(s["dataset"])}
        for s in summaries
    ]
    successful_keys = {(x["model"], x["dataset"]) for x in successful_pairs}
    missing_pairs = [
        x for x in expected_pairs
        if (x["model"], x["dataset"]) not in successful_keys
    ]
    write_json(
        root_out / "controller_compatibility.json",
        {
            "compatible_with": "eval_multimodel_targetselected_head_spatial_control_v1.py",
            "direction_source": "Synthetic-400 only",
            "head_id_selection": "full target GT",
            "head_selection_frac": float(args.head_selection_frac),
            "control": str(args.control),
            "with_generation": bool(args.with_generation),
            "expected_pairs": expected_pairs,
            "successful_pairs": successful_pairs,
            "missing_pairs": missing_pairs,
            "ready_for_controller": len(missing_pairs) == 0,
        },
    )

    print_cross_summary(rows)
    print(f"\n[DONE] output={root_out} | success_pairs={len(summaries)} failures={len(failures)}")


if __name__ == "__main__":
    main()
