#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_synthetic3d_controlledB_spatial_control_genflip_trust_standalone_v1.py

Standalone v6-style recovery for Controlled-B using the Blender 3D Synthetic-600
source. No AdaptVis Python modules are imported.

Source protocol
===============
The source dataset is synthetic_shapes_6dir_600_3d. For Controlled-B we use
only the four matching relations:

    left, right, front, behind

(above/below remain in the dataset but are not used in this experiment).
The source prompt is rewritten to the same four-answer relation set as
Controlled-B.

Two source-only components are fit independently for each VLM:

1) Reader directions: pre-W_O per-head Real-Gray object-pair residual vectors
   are extracted on the Blender source. Four frozen relation directions are
   fit at every attention head. The single head ID is then chosen using full
   Controlled-B GT, matching the original v6 target-supervised head-selection
   diagnostic. Per-sample routing after head selection uses only that head's
   prediction.

2) Actuator geometry: decoder residual object-pair Real-Gray states are
   extracted on the same Blender source at the seven controller layers. The
   horizontal axis is right-left and the depth axis is front-behind. Their
   span is orthonormalized, yielding the same 2 dimensions/layer and 14D total
   control budget as v6.

Controller
==========
The intervention and search follow v6:

    h_sub <- h_sub + delta_z/2
    h_ref <- h_ref - delta_z/2

where delta_z is restricted to the source-fitted H/D subspace. For the chosen
relation target, the script computes target-vs-competitor relation-score
margins and exact gradients w.r.t. all control coordinates, solves the local
minimum-norm half-space QP, applies a trust-region cap, relinearizes, checks
actual model.generate(), deepens the required margin if needed, and binary
refines the first generation-flipping step.

Conditions:
  head   = target relation from the source-trained / target-selected head
  oracle = Controlled-B GT (upper bound)

This file is self-contained with respect to AdaptVis code. It depends only on
standard Python packages plus torch/transformers/PIL/numpy/pandas/tqdm.

Default source:
  synthetic_shapes_6dir_600_3d/labels.jsonl

Default target:
  data/controlled_clevr_dataset.json
  data/controlled_clevr/
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import AutoProcessor

REL = ("left", "right", "in_front", "behind")
EPS = 1e-12
SCRIPT_VERSION = "synthetic3d-to-controlledB-standalone-v6-controller-v1"
MODEL_ORDER = ("qwen2-2b", "qwen-3b", "qwen-7b", "llava-7b", "llava-13b")
DATASET_ORDER = ("controlled_b",)
MODEL_SPECS = {
    "qwen2-2b": ("Qwen/Qwen2-VL-2B-Instruct", "Qwen2VLForConditionalGeneration", "bfloat16"),
    "qwen-3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2_5_VLForConditionalGeneration", "bfloat16"),
    "qwen-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2_5_VLForConditionalGeneration", "bfloat16"),
    "llava-7b": ("llava-hf/llava-1.5-7b-hf", "LlavaForConditionalGeneration", "float16"),
    "llava-13b": ("llava-hf/llava-1.5-13b-hf", "LlavaForConditionalGeneration", "float16"),
}

# InternVL is intentionally not included in this standalone version.
UNSUPPORTED_INTERNVL = None


# =============================================================================
# Standalone model / prompt / Controlled-B helpers
# =============================================================================

def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[str(name)]


def get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_decoder_layers(model: Any) -> Tuple[Sequence[Any], str]:
    for path in ("language_model.layers", "model.language_model.layers", "language_model.model.layers", "model.layers", "model.model.layers"):
        layers = get_attr_path(model, path)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers, path
    raise RuntimeError("Could not locate decoder layers in model")


def scan_shape(model: Any, layers: Sequence[Any]) -> Tuple[int, int]:
    cfg = getattr(model, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "language_config", None)):
        if c is None:
            continue
        nh = getattr(c, "num_attention_heads", None)
        hs = getattr(c, "hidden_size", None)
        if nh and hs:
            return int(nh), int(hs) // int(nh)
    return 0, 0


def configure_processor(model: Any, processor: Any) -> None:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id
    with contextlib.suppress(Exception):
        if model.generation_config.pad_token_id is None and tok is not None:
            model.generation_config.pad_token_id = tok.pad_token_id


def load_model_bundle(model_alias: str, args: argparse.Namespace):
    if model_alias not in MODEL_SPECS:
        raise KeyError(model_alias)
    repo_id, class_name, default_dtype = MODEL_SPECS[model_alias]
    cls = getattr(transformers, class_name, None)
    if cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {class_name}")
    dtype_name = default_dtype if str(args.dtype) == "bfloat16" else str(args.dtype)
    kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(dtype_name),
        "low_cpu_mem_usage": True,
        "device_map": {"": args.device},
        "revision": args.revision,
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl
    print(f"\n[LOAD] {model_alias} -> {repo_id}")
    try:
        model = cls.from_pretrained(repo_id, **kwargs)
    except TypeError:
        kwargs["dtype"] = kwargs.pop("torch_dtype")
        model = cls.from_pretrained(repo_id, **kwargs)
    model.eval()
    try:
        processor = AutoProcessor.from_pretrained(repo_id, revision=args.revision, use_fast=False)
    except TypeError:
        processor = AutoProcessor.from_pretrained(repo_id, revision=args.revision)
    configure_processor(model, processor)
    layers, path = resolve_decoder_layers(model)
    nh, hd = scan_shape(model, layers)
    print(f"[MODEL] decoder={path} layers={len(layers)} heads/layer={nh} head_dim={hd}")
    return model, processor, layers, nh, hd, {"repo_id": repo_id, "model_class": class_name}


def build_chat_prompt(processor: Any, question: str, include_image: bool = True) -> str:
    if hasattr(processor, "apply_chat_template"):
        content: List[Dict[str, Any]] = []
        if include_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": str(question)})
        messages = [{"role": "user", "content": content}]
        with contextlib.suppress(Exception):
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return ("<image>\n" if include_image else "") + str(question)


def process_inputs(processor: Any, rendered: str, image: Image.Image, device: torch.device) -> Dict[str, Any]:
    attempts = [
        lambda: processor(text=[rendered], images=[image], return_tensors="pt", padding=True),
        lambda: processor(text=rendered, images=image, return_tensors="pt"),
        lambda: processor(images=image, text=rendered, return_tensors="pt"),
    ]
    last = None
    for fn in attempts:
        try:
            batch = fn()
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dict(batch).items()}
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Processor failed to build multimodal batch: {last}")


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    h, n = list(map(int, haystack)), list(map(int, needle))
    return [i for i in range(max(0, len(h)-len(n)+1)) if n and h[i:i+len(n)] == n]


def locate_phrase_positions(tokenizer: Any, input_ids: Sequence[int], phrase: str, start: int = 0) -> List[int]:
    matches: List[Tuple[int, List[int]]] = []
    for text in (str(phrase).strip(), " " + str(phrase).strip()):
        ids = [int(x) for x in tokenizer(text, add_special_tokens=False).input_ids]
        for pos in find_subsequence(input_ids, ids):
            if pos >= int(start):
                matches.append((pos, list(range(pos, pos + len(ids)))))
    if not matches:
        raise RuntimeError(f"Could not locate phrase {phrase!r} in tokenized prompt")
    return max(matches, key=lambda z: z[0])[1]


def image_placeholder_positions(*, model: Any, processor: Any, input_ids: Sequence[int]) -> List[int]:
    token_ids = set()
    cfg = getattr(model, "config", None)
    for attr in ("image_token_index", "image_token_id"):
        v = getattr(cfg, attr, None) if cfg is not None else None
        if v is not None:
            with contextlib.suppress(Exception): token_ids.add(int(v))
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        for token in ("<image>", "<|image_pad|>", "<image_token>", "<IMG_CONTEXT>"):
            with contextlib.suppress(Exception):
                v = tok.convert_tokens_to_ids(token)
                if v is not None and int(v) >= 0: token_ids.add(int(v))
    return [i for i, tid in enumerate(input_ids) if int(tid) in token_ids]


def map_text_positions_to_decoder(*, model: Any, processor: Any, input_ids: Sequence[int], raw_positions: Sequence[int], merged_length: int) -> Tuple[List[int], Dict[str, Any]]:
    raw_length = len(input_ids)
    raw_positions = sorted(set(map(int, raw_positions)))
    if merged_length == raw_length:
        return raw_positions, {"mapping_mode": "identity", "shift": 0}
    if merged_length < raw_length:
        raise RuntimeError(f"Decoder sequence shorter than raw input: raw={raw_length}, merged={merged_length}")
    placeholders = image_placeholder_positions(model=model, processor=processor, input_ids=input_ids)
    if not placeholders:
        raise RuntimeError("Decoder length changed but image placeholder was not found")
    last_image = max(placeholders)
    if any(p <= last_image for p in raw_positions):
        raise RuntimeError("Object text token occurs before image expansion")
    shift = int(merged_length - raw_length)
    return [p + shift for p in raw_positions], {"mapping_mode": "post_image_shift", "shift": shift}


def _clean_obj(s: str) -> str:
    s = re.sub(r"[_-]+", " ", str(s)).strip()
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)
    return s.strip(" .,'\"")


def _parse_controlled_b_caption(caption: str) -> Optional[Tuple[str, str, str]]:
    s = re.sub(r"\s+", " ", str(caption).strip().lower())
    patterns = [
        ("in_front", r"^(?:the )?(.+?)\s+(?:is|are)\s+(?:in front of|in-front of)\s+(?:the )?(.+?)[\.!]?$"),
        ("behind", r"^(?:the )?(.+?)\s+(?:is|are)\s+behind\s+(?:the )?(.+?)[\.!]?$"),
        ("left", r"^(?:the )?(.+?)\s+(?:is|are)\s+(?:to the )?left(?: of)?\s+(?:the )?(.+?)[\.!]?$"),
        ("right", r"^(?:the )?(.+?)\s+(?:is|are)\s+(?:to the )?right(?: of)?\s+(?:the )?(.+?)[\.!]?$"),
    ]
    for rel, pat in patterns:
        m = re.match(pat, s, flags=re.I)
        if m:
            return _clean_obj(m.group(1)), _clean_obj(m.group(2)), rel
    return None


def _parse_controlled_b_filename(path: str) -> Optional[Tuple[str, str, str]]:
    stem = Path(path).stem.lower()
    for marker, rel in (("_in-front_of_", "in_front"), ("_in_front_of_", "in_front"), ("_left_of_", "left"), ("_right_of_", "right"), ("_behind_", "behind")):
        if marker in stem:
            a, b = stem.split(marker, 1)
            return _clean_obj(a), _clean_obj(b), rel
    return None


def load_controlled_b(args: argparse.Namespace, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ann = Path(args.controlled_b_json)
    if not ann.exists():
        raise FileNotFoundError(ann)
    raw = json.loads(ann.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("data", "dataset", "annotations", "items"):
            if isinstance(raw.get(key), list): raw = raw[key]; break
    if not isinstance(raw, list):
        raise RuntimeError(f"Unexpected JSON structure in {ann}")
    image_root = Path(args.controlled_b_image_root)
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for i, item in enumerate(raw):
        if not isinstance(item, dict): continue
        raw_path = str(item.get("image_path") or item.get("image") or "")
        p = Path(raw_path)
        if not p.exists():
            for cand in (image_root / p.name, Path(args.data_root) / raw_path):
                if cand.exists(): p = cand; break
        caps = item.get("caption_options") or item.get("captions") or []
        parsed = _parse_controlled_b_caption(str(caps[0])) if isinstance(caps, list) and caps else None
        if parsed is None: parsed = _parse_controlled_b_filename(raw_path)
        if parsed is None or not p.exists(): skipped += 1; continue
        subject, reference, rel = parsed
        rel = norm_rel(rel)
        if rel not in REL: skipped += 1; continue
        question = f"Where is the {subject} in relation to the {reference}? Answer with left, right, in front, or behind."
        rows.append({"sid": int(item.get("id", i)), "dataset": "controlled_b", "image_path": str(p), "subject": subject, "reference": reference, "relation": rel, "question_text": question})
    rows.sort(key=lambda r: int(r["sid"]))
    if limit is not None: rows = rows[:int(limit)]
    counts = {r: sum(x["relation"] == r for x in rows) for r in REL}
    missing = [r for r in REL if counts[r] == 0]
    if not rows or missing:
        raise RuntimeError(f"Controlled-B rows invalid: n={len(rows)} missing={missing} counts={counts} skipped={skipped}")
    print(f"[Controlled-B] n={len(rows)} counts={counts} skipped={skipped}")
    return rows


def load_synthetic_3d(args: argparse.Namespace, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load Blender Synthetic-600 and keep the four Controlled-B relations.

    The on-disk labels use ``front``; internally it is normalized to
    ``in_front``.  Source prompts are rewritten to exactly the four target
    relation surfaces so the reader is not trained under a six-choice prompt
    and evaluated under a four-choice prompt.
    """
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
            if not subject or not reference:
                raise RuntimeError(f"Synthetic row {i} missing subject/reference")
            rel_img = str(x.get("image", "")).strip()
            if not rel_img:
                raise RuntimeError(f"Synthetic row {i} missing image")
            image_path = Path(rel_img)
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            question = (
                f"Where is the {subject} in relation to the {reference}? "
                "Answer with left, right, in front, or behind."
            )
            rows.append({
                "sid": int(i),
                "dataset": "synthetic3d_controlled_b_source",
                "image_path": str(image_path),
                "subject": subject,
                "reference": reference,
                "relation": rel,
                "question_text": question,
                "source_relation_raw": raw_rel,
            })

    rows.sort(key=lambda r: int(r["sid"]))
    if limit is not None:
        # Keep a deterministic class-balanced prefix rather than the first N
        # rows, which may depend on generation order.
        n = int(limit)
        if n > 0 and n < len(rows):
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
    missing = [r for r in REL if counts[r] == 0]
    if not rows or missing:
        raise RuntimeError(
            f"Synthetic-3D source invalid: n={len(rows)} missing={missing} "
            f"counts={counts} raw_counts={raw_counts}"
        )
    print(f"[Synthetic-3D source] n={len(rows)} counts={counts} raw_counts={raw_counts}")
    return rows


# =============================================================================
# Generic helpers
# =============================================================================


def norm_rel(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = re.sub(r"\s+", " ", str(x).strip().lower().replace("_", " ").replace("-", " "))
    table = {
        "left": "left", "left of": "left", "to the left of": "left", "l": "left",
        "right": "right", "right of": "right", "to the right of": "right", "r": "right",
        "front": "in_front", "in front": "in_front", "in front of": "in_front", "in the front of": "in_front",
        "behind": "behind", "back": "behind", "in back of": "behind",
    }
    key = s.replace(" ", "_")
    return table.get(s, key if key in REL else None)


def parse_generation(text: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", str(text).lower().replace("_", " ").replace("-", " "))
    pats = [
        ("in_front", r"\bin front of\b"), ("in_front", r"\bin front\b"), ("behind", r"\bbehind\b"),
        ("left", r"\bto the left of\b"), ("right", r"\bto the right of\b"),
        ("left", r"\bleft\b"), ("right", r"\bright\b"), ("in_front", r"\bfront\b"),
    ]
    hits=[]
    for lab,pat in pats:
        m=re.search(pat,s)
        if m: hits.append((m.start(),lab))
    return min(hits,key=lambda z:z[0])[1] if hits else None


def surface_for(rel: str, dataset: str) -> str:
    del dataset
    return {"left":"left","right":"right","in_front":"in front","behind":"behind"}[str(rel)]


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
    p.add_argument("--datasets", default="controlled_b")
    p.add_argument("--pairs", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    # Blender 3D Synthetic-600 source. Only L/R/front/behind are used here.
    p.add_argument("--synthetic-3d-dir", default="synthetic_shapes_6dir_600_3d")
    p.add_argument("--synthetic-3d-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)

    # Controlled-B target.
    p.add_argument("--data-root", default="data")
    p.add_argument("--controlled-b-json", default="data/controlled_clevr_dataset.json")
    p.add_argument("--controlled-b-image-root", default="data/controlled_clevr")
    p.add_argument("--target-max-samples", type=int, default=None)

    # Model/runtime.
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--revision", default="main")
    p.add_argument("--control", default="gray", choices=["gray"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--seed", type=int, default=17)

    # Source-trained reader / target-supervised head-ID selection, same v6 protocol.
    p.add_argument("--head-selection-frac", type=float, default=1.0)
    p.add_argument("--top-k-heads", type=int, default=30)
    p.add_argument("--cache-vectors", action=argparse.BooleanOptionalAction, default=True)

    # Controller geometry / optimization.
    p.add_argument("--controller-layers", default="")
    p.add_argument("--geometry-cache-dir", default=None)
    p.add_argument("--modes", default="head,oracle", help="Comma-separated: head,oracle")
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all target samples")
    p.add_argument("--eval-seed", type=int, default=17)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--decision-margin-eps", type=float, default=1e-4)
    p.add_argument("--generation-margin-increment", type=float, default=0.25)
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


# =============================================================================
# Standalone Synthetic-3D frozen head reader
# =============================================================================


def resolve_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise RuntimeError("Could not resolve self-attention module")


def resolve_o_proj(attn: Any) -> torch.nn.Module:
    for name in ("o_proj", "out_proj", "proj"):
        m = getattr(attn, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    raise RuntimeError("Could not resolve attention output projection")


def pool_positions(tensor: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    valid = [int(p) for p in positions if 0 <= int(p) < int(tensor.shape[1])]
    if not valid:
        raise RuntimeError("No valid object positions for head capture")
    if str(mode) == "last":
        return tensor[0, valid[-1]]
    idx = torch.as_tensor(valid, device=tensor.device, dtype=torch.long)
    return tensor[0].index_select(0, idx).mean(dim=0)


class CapturePreWO:
    def __init__(
        self,
        layers: Sequence[Any],
        n_heads: int,
        head_dim: int,
        sub_pos: Sequence[int],
        ref_pos: Sequence[int],
        pool: str,
    ):
        self.layers = layers
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.sub_pos = list(map(int, sub_pos))
        self.ref_pos = list(map(int, ref_pos))
        self.pool = str(pool)
        self.out = torch.empty(
            (len(layers), self.n_heads, 2, self.head_dim), dtype=torch.float32
        )
        self.seen: set[int] = set()
        self.handles: List[Any] = []

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            op = resolve_o_proj(resolve_self_attention(layer))

            def make_hook(layer_idx: int):
                def hook(_module, inputs):
                    x = inputs[0]
                    a = pool_positions(x, self.sub_pos, self.pool).view(self.n_heads, self.head_dim)
                    b = pool_positions(x, self.ref_pos, self.pool).view(self.n_heads, self.head_dim)
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


def capture_head_condition(
    *, model: Any, processor: Any, decoder_layers: Sequence[Any], n_heads: int,
    head_dim: int, row: Mapping[str, Any], image: Image.Image, args: argparse.Namespace,
) -> np.ndarray:
    device = torch.device(args.device)
    rendered = build_chat_prompt(processor, str(row["question_text"]), True)
    batch = process_inputs(processor, rendered, image, device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    sub_pos = locate_phrase_positions(processor.tokenizer, ids, str(row["subject"]))
    ref_pos = locate_phrase_positions(processor.tokenizer, ids, str(row["reference"]))
    cap = CapturePreWO(
        decoder_layers, n_heads, head_dim, sub_pos, ref_pos, str(args.pool)
    )
    try:
        with cap:
            with torch.inference_mode():
                model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        return cap.finalize()
    finally:
        cap.close()
        del batch


def save_head_vector_cache(path: Path, pack: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=np.asarray(pack["vectors"]),
        sid=np.asarray(pack["sid"], dtype=np.int64),
        relation=np.asarray(pack["relation"], dtype=object),
        metadata_json=np.asarray(json.dumps(dict(metadata)), dtype=object),
    )


def load_head_vector_cache(path: Path, expected: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        try:
            meta = json.loads(str(z["metadata_json"].item()))
        except Exception:
            return None
        for k, v in expected.items():
            if meta.get(k) != v:
                return None
        return {
            "vectors": np.asarray(z["vectors"]),
            "sid": np.asarray(z["sid"], dtype=np.int64),
            "relation": np.asarray(z["relation"].tolist(), dtype=object),
            "metadata": meta,
        }


def extract_head_vectors(
    *, records: Sequence[Mapping[str, Any]], model: Any, processor: Any,
    decoder_layers: Sequence[Any], n_heads: int, head_dim: int, model_alias: str,
    dataset_name: str, cache_path: Path, args: argparse.Namespace,
) -> Dict[str, Any]:
    expected = {
        "model": model_alias,
        "dataset": dataset_name,
        "control": str(args.control),
        "gray_value": int(args.gray_value),
        "pool": str(args.pool),
        "n_layers": int(len(decoder_layers)),
        "n_heads": int(n_heads),
        "head_dim": int(head_dim),
        "source_kind": "blender_3d" if "synthetic" in dataset_name else "controlled_b",
    }
    if args.cache_vectors and not args.overwrite:
        cached = load_head_vector_cache(cache_path, expected)
        if cached is not None:
            print(f"[HEAD CACHE] reuse {cache_path} X={cached['vectors'].shape}")
            return cached

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    vectors: List[np.ndarray] = []
    sids: List[int] = []
    labels: List[str] = []
    errors_path = cache_path.with_suffix(".errors.jsonl")
    if args.overwrite and errors_path.exists():
        errors_path.unlink()

    for row in tqdm(records, desc=f"{model_alias}:{dataset_name}:head vectors"):
        img = gray = None
        try:
            img = Image.open(str(row["image_path"])).convert("RGB")
            gray = Image.new("RGB", img.size, (int(args.gray_value),) * 3)
            real = capture_head_condition(
                model=model, processor=processor, decoder_layers=decoder_layers,
                n_heads=n_heads, head_dim=head_dim, row=row, image=img, args=args,
            )
            ctrl = capture_head_condition(
                model=model, processor=processor, decoder_layers=decoder_layers,
                n_heads=n_heads, head_dim=head_dim, row=row, image=gray, args=args,
            )
            pair_real = real[:, :, 0, :] - real[:, :, 1, :]
            pair_ctrl = ctrl[:, :, 0, :] - ctrl[:, :, 1, :]
            vectors.append((pair_real - pair_ctrl).astype(dtype_np, copy=False))
            sids.append(int(row["sid"]))
            labels.append(str(norm_rel(row["relation"])))
        except Exception as exc:
            append_jsonl(errors_path, {
                "sid": int(row.get("sid", -1)),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-40:],
            })
            tqdm.write(
                f"[HEAD VECTOR ERROR] {model_alias}/{dataset_name} sid={row.get('sid')}: "
                f"{type(exc).__name__}: {exc}"
            )
            if args.fail_fast:
                raise
        finally:
            if img is not None:
                img.close()
            if gray is not None:
                gray.close()
            cleanup_cuda()

    if not vectors:
        raise RuntimeError(f"No successful head vectors for {model_alias}/{dataset_name}")
    X = np.stack(vectors, axis=0)
    pack = {
        "vectors": X,
        "sid": np.asarray(sids, dtype=np.int64),
        "relation": np.asarray(labels, dtype=object),
        "metadata": expected,
    }
    if args.cache_vectors:
        save_head_vector_cache(cache_path, pack, expected)
        print(f"[HEAD CACHE] saved {cache_path} X={X.shape}")
    return pack


def normalize_last(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def fit_source_head_codebooks(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """X [N,L,H,D] -> center [L,H,D], dirs [L,H,4,D]."""
    Xf = np.asarray(X, dtype=np.float32)
    center = Xf.mean(axis=0)
    L, H, D = center.shape
    dirs = np.empty((L, H, len(REL), D), dtype=np.float32)
    for ri, rel in enumerate(REL):
        mask = np.asarray(y, dtype=object) == rel
        if not np.any(mask):
            raise RuntimeError(f"Synthetic-3D head source missing relation={rel}")
        d = Xf[mask].mean(axis=0) - center
        dirs[:, :, ri, :] = normalize_last(d)
    return center.astype(np.float32), dirs.astype(np.float32)


def predict_one_head(
    X: np.ndarray, center: np.ndarray, dirs: np.ndarray, layer: int, head: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(X[:, layer, head, :], dtype=np.float32) - center[layer, head][None, :]
    x = normalize_last(x)
    scores = x @ dirs[layer, head].T
    idx = np.argmax(scores, axis=1)
    pred = np.asarray([REL[int(i)] for i in idx], dtype=object)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    return pred, margin


def safe_acc(pred: Sequence[Any], gt: Sequence[Any]) -> float:
    if len(gt) == 0:
        return float("nan")
    return float(np.mean(np.asarray(pred, dtype=object) == np.asarray(gt, dtype=object)))


def build_standalone_routing(
    *, model_alias: str, model: Any, processor: Any, decoder_layers: Sequence[Any],
    n_heads: int, head_dim: int, source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]], root_out: Path, args: argparse.Namespace,
) -> Tuple[dict, Dict[int, dict]]:
    if float(args.head_selection_frac) < 1.0 - 1e-12:
        raise ValueError(
            "This v6-compatible script currently requires --head-selection-frac 1.0 "
            "so the full Controlled-B target chooses the head ID."
        )

    reader_dir = root_out / "reader" / model_alias / "controlled_b"
    reader_dir.mkdir(parents=True, exist_ok=True)
    source_cache = reader_dir / "synthetic3d_LRFB_head_vectors.npz"
    target_cache = reader_dir / "controlledB_head_vectors.npz"

    src = extract_head_vectors(
        records=source_rows, model=model, processor=processor,
        decoder_layers=decoder_layers, n_heads=n_heads, head_dim=head_dim,
        model_alias=model_alias, dataset_name="synthetic3d_LRFB",
        cache_path=source_cache, args=args,
    )
    tgt = extract_head_vectors(
        records=target_rows, model=model, processor=processor,
        decoder_layers=decoder_layers, n_heads=n_heads, head_dim=head_dim,
        model_alias=model_alias, dataset_name="controlled_b",
        cache_path=target_cache, args=args,
    )

    Xs = np.asarray(src["vectors"])
    ys = np.asarray(src["relation"], dtype=object)
    Xt = np.asarray(tgt["vectors"])
    yt = np.asarray(tgt["relation"], dtype=object)
    center, dirs = fit_source_head_codebooks(Xs, ys)
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

    if Xt.shape[1:] != Xs.shape[1:]:
        raise RuntimeError(f"Source/target head shape mismatch: source={Xs.shape}, target={Xt.shape}")
    ranking: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    all_idx = np.arange(len(yt), dtype=np.int64)
    for l in tqdm(range(Xt.shape[1]), desc=f"{model_alias}:rank synthetic3d heads", leave=False):
        for h in range(Xt.shape[2]):
            pred_t, margin_t = predict_one_head(Xt, center, dirs, l, h)
            pred_s, _ = predict_one_head(Xs, center, dirs, l, h)
            predictions[(l, h)] = (pred_t, margin_t)
            per_rel = {}
            for rel in REL:
                m = yt == rel
                per_rel[rel] = safe_acc(pred_t[m], yt[m]) if np.any(m) else float("nan")
            ranking.append({
                "layer": int(l),
                "head": int(h),
                "head_name": f"L{l}H{h:02d}",
                "syn_self_acc": safe_acc(pred_s, ys),
                "selection_acc": safe_acc(pred_t[all_idx], yt[all_idx]),
                "test_acc": safe_acc(pred_t[all_idx], yt[all_idx]),
                "all_target_acc": safe_acc(pred_t, yt),
                "left_acc": per_rel["left"],
                "right_acc": per_rel["right"],
                "front_acc": per_rel["in_front"],
                "behind_acc": per_rel["behind"],
                "mean_margin": float(np.mean(margin_t)),
            })
    ranking.sort(key=lambda r: (
        -float(r["selection_acc"]),
        -float(r["syn_self_acc"]),
        -float(r["mean_margin"]),
        int(r["layer"]), int(r["head"]),
    ))
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
        "dataset": "controlled_b",
        "control": "gray",
        "source": "synthetic_shapes_6dir_600_3d filtered to left/right/front/behind",
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
    write_json(reader_dir / "summary.json", summary)

    print("\n" + "=" * 138)
    print("SYNTHETIC-3D FROZEN HEAD READER -> CONTROLLED-B")
    print("=" * 138)
    print(
        f"best={summary['best_head']} | syn={summary['best_syn_self_acc']:.4f} | "
        f"target={summary['best_all_target_acc']:.4f} | N={summary['target_n']}"
    )
    print(pd.DataFrame(ranking[: min(int(args.top_k_heads), 10)])[
        ["rank", "head_name", "syn_self_acc", "all_target_acc", "left_acc", "right_acc", "front_acc", "behind_acc", "mean_margin"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary, routing


# =============================================================================
# Synthetic-3D residual H/D geometry
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
    rendered = build_chat_prompt(processor, question, True)
    batch = process_inputs(processor, rendered, image, device)
    try:
        ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
        sub_raw = locate_phrase_positions(processor.tokenizer, ids, subject)
        ref_raw = locate_phrase_positions(processor.tokenizer, ids, reference)
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
        sub_pos, _ = map_text_positions_to_decoder(
            model=model, processor=processor, input_ids=ids,
            raw_positions=sub_raw, merged_length=merged_len,
        )
        ref_pos, _ = map_text_positions_to_decoder(
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
    raise RuntimeError("InternVL is not supported by this standalone file; use Qwen/LLaVA.")
    pixels, _layout = UNSUPPORTED_INTERNVL.internvl_pixels_and_layout(backend, image, args)
    _query, input_ids, attention_mask, visual_positions = UNSUPPORTED_INTERNVL.internvl_chat_query(
        backend, question, int(pixels.shape[0])
    )
    ids = input_ids[0].detach().cpu().tolist()
    start = max(visual_positions) + 1 if visual_positions else 0
    sub = UNSUPPORTED_INTERNVL.token_span_for_phrase(backend.tokenizer, ids, subject, start=start)
    ref = UNSUPPORTED_INTERNVL.token_span_for_phrase(backend.tokenizer, ids, reference, start=start)
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
    for row in tqdm(source_rows, desc=f"{model_alias}:Synthetic-3D residual geometry"):
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
            f"Too few successful Synthetic-3D geometry samples for {model_alias}: {len(vectors)}/{len(source_rows)}"
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
    """Fit the 2D Blender-source H/D spatial subspace at each layer.

    The class means define H/D directions, but their class-gap magnitudes do not
    set the intervention scale.  We orthonormalize the H/D span so Euclidean
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
                raise RuntimeError(f"Synthetic-3D geometry missing class={r}")
            means[r] = Xf[m].mean(0)

        class_dirs = {r: unit(means[r] - center) for r in REL}
        dH = unit(class_dirs["right"] - class_dirs["left"])
        dV = unit(class_dirs["in_front"] - class_dirs["behind"])

        # Preserve the same H/D span, but use an orthonormal control basis.
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
            "axis_H_dot_axis_D": float(np.dot(dH, dV)),
            "basis_H_dot_basis_D": float(np.dot(qH, qV)),
            "basis_H_align_axis_H": float(np.dot(qH, dH)),
            "basis_D_align_axis_D": float(np.dot(qV, dV)),
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
    batch = process_inputs(
        processor, build_chat_prompt(processor, str(row["question_text"]), True), image, device
    )
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    sub_raw = locate_phrase_positions(processor.tokenizer, ids, str(row["subject"]))
    ref_raw = locate_phrase_positions(processor.tokenizer, ids, str(row["reference"]))
    # One cheap no-grad forward obtains the actual merged decoder length.
    with torch.inference_mode():
        out = model(
            **batch, output_hidden_states=True, output_attentions=False,
            use_cache=False, return_dict=True,
        )
    hst = extract_hidden_states(out)
    merged_len = int(hst[0].shape[1])
    sub_pos, _ = map_text_positions_to_decoder(
        model=model, processor=processor, input_ids=ids,
        raw_positions=sub_raw, merged_length=merged_len,
    )
    ref_pos, _ = map_text_positions_to_decoder(
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


def encode_fast_relation_tokens(tokenizer: Any, dataset: str) -> Tuple[Optional[Dict[str, List[int]]], Dict[str, Any]]:
    """Collect one-token surface variants for each relation.

    Both ``" left"`` and ``"left"`` are tested because tokenizers differ in
    how they encode a word at the answer boundary.  During each forward pass
    the relation score is the maximum logit over its one-token variants, so the
    fast path is not tied to an arbitrary whitespace convention.  If any
    relation lacks a one-token form, we fall back to exact sequence scoring.
    """
    token_ids: Dict[str, List[int]] = {}
    details: Dict[str, Any] = {}
    for r in REL:
        surface = surface_for(r, dataset)
        choices: List[Tuple[str, List[int]]] = []
        one_ids: List[int] = []
        for text in (" " + surface, surface):
            ids = [int(x) for x in tokenizer(text, add_special_tokens=False).input_ids]
            choices.append((text, ids))
            if len(ids) == 1 and int(ids[0]) not in one_ids:
                one_ids.append(int(ids[0]))
        details[r] = {
            "surface": surface,
            "encodings": [(t, ids) for t, ids in choices],
            "one_token_variants": list(one_ids),
        }
        if not one_ids:
            return None, details
        token_ids[r] = list(one_ids)
    # Distinct relations must not collapse to exactly the same candidate set.
    signatures = [tuple(v) for v in token_ids.values()]
    if len(set(signatures)) != len(REL):
        return None, details
    return token_ids, details


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


def fast_relation_scores_and_grads(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    relation_token_ids: Mapping[str, Sequence[int]], layers: Sequence[int], geom: Mapping[int, Any],
    base_coords: np.ndarray, target: str, need_grads: bool = True,
) -> Tuple[Dict[str, float], Optional[np.ndarray], List[str]]:
    """One prompt forward; optionally three margin backwards.

    For single-token relation surfaces, logit differences equal log-probability
    differences, so no softmax is needed.  The forward graph is shared across
    all three target-vs-competitor gradients.
    """
    dev = prepared.batch["input_ids"].device
    delta = torch.zeros((len(layers), 2), device=dev, dtype=torch.float32, requires_grad=need_grads)
    base = torch.as_tensor(base_coords, device=dev, dtype=torch.float32)
    coords = base + delta
    with SpatialPatch(
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
            g = torch.autograd.grad(
                margin, delta,
                retain_graph=(j + 1 < len(competitors)),
                create_graph=False,
            )[0]
            grads.append(g.detach().float().cpu().numpy().astype(np.float64).reshape(-1))
        A = np.stack(grads, axis=0)
    return scores, A, competitors


def sequence_relation_scores_and_grads(
    *, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    candidate_ids: Mapping[str, Sequence[int]], layers: Sequence[int], geom: Mapping[int, Any],
    base_coords: np.ndarray, target: str, reduction: str,
) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    """Exact multi-token fallback; no nonlinear line search."""
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
            text = UNSUPPORTED_INTERNVL.internvl_generate(
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
    """Project the origin onto {x : A x >= b} without SVD.

    There are only three competitor half-spaces.  We use Dykstra's projection
    algorithm, which is numerically stable for redundant / nearly parallel
    constraints and avoids ``np.linalg.pinv`` (and therefore avoids an SVD on
    a possibly ill-conditioned Gram matrix).

    The row normalization below does *not* change the feasible set; it only
    conditions the projection updates.
    """
    del lambda_tol  # kept in the signature for CLI/backward compatibility
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError(f"Bad QP shapes A={A.shape} b={b.shape}")

    n = int(A.shape[1])
    m = int(A.shape[0])
    zero = np.zeros((n,), dtype=np.float64)

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return None, {
            "active_constraints": None,
            "predicted_step_norm": float("nan"),
            "max_linearized_violation": float("nan"),
            "solver_status": "nonfinite_linearization",
        }

    if m == 0 or np.all(A @ zero >= b - float(feas_tol)):
        return zero, {
            "active_constraints": [],
            "predicted_step_norm": 0.0,
            "max_linearized_violation": float(np.max(np.maximum(b, 0.0))) if len(b) else 0.0,
            "solver_status": "already_feasible",
        }

    # Normalize each half-space normal for numerical conditioning.  This leaves
    # the constraints exactly equivalent because both sides are divided by the
    # same positive scalar.
    row_norm = np.sqrt(np.sum(A * A, axis=1))
    tiny = 1e-14
    impossible = (row_norm <= tiny) & (b > float(feas_tol))
    if np.any(impossible):
        return None, {
            "active_constraints": None,
            "predicted_step_norm": float("nan"),
            "max_linearized_violation": float(np.max(b[impossible])),
            "solver_status": "zero_gradient_infeasible",
        }

    keep = row_norm > tiny
    A_work = A[keep] / row_norm[keep, None]
    b_work = b[keep] / row_norm[keep]
    original_index = np.nonzero(keep)[0]

    if len(b_work) == 0:
        return zero, {
            "active_constraints": [],
            "predicted_step_norm": 0.0,
            "max_linearized_violation": 0.0,
            "solver_status": "degenerate_but_feasible",
        }

    # Dykstra projection of x0=0 onto the intersection of half-spaces.
    x = np.zeros((n,), dtype=np.float64)
    correction = np.zeros((len(b_work), n), dtype=np.float64)
    max_iter = 10000
    tol = max(float(feas_tol), 1e-10)
    converged = False

    for _ in range(max_iter):
        x_before = x.copy()
        for i in range(len(b_work)):
            y = x + correction[i]
            gap = float(b_work[i] - np.dot(A_work[i], y))
            if gap > 0.0:
                # A_work[i] is unit length.
                x_new = y + gap * A_work[i]
            else:
                x_new = y
            correction[i] = y - x_new
            x = x_new

        violation = b_work - A_work @ x
        max_violation = float(np.max(np.maximum(violation, 0.0)))
        movement = float(np.sqrt(np.sum((x - x_before) ** 2)))
        if max_violation <= tol and movement <= max(tol, 1e-12 * (1.0 + float(np.sqrt(np.sum(x * x))))):
            converged = True
            break

    original_violation = b - A @ x
    max_original_violation = float(np.max(np.maximum(original_violation, 0.0))) if len(b) else 0.0
    if (not converged and max_original_violation > 10.0 * tol) or not np.all(np.isfinite(x)):
        return None, {
            "active_constraints": None,
            "predicted_step_norm": float("nan"),
            "max_linearized_violation": max_original_violation,
            "solver_status": "projection_not_converged",
        }

    slack = A @ x - b
    active_tol = max(1e-6, 10.0 * float(feas_tol))
    active = [int(i) for i in range(m) if abs(float(slack[i])) <= active_tol]
    return x, {
        "active_constraints": active,
        "predicted_step_norm": float(np.sqrt(np.sum(x * x))),
        "max_linearized_violation": max_original_violation,
        "solver_status": "ok",
    }



def refine_first_generation_flip(
    *, model_alias: str, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    layers: Sequence[int], geom: Mapping[int, Any], target: str,
    low_coords: np.ndarray, high_coords: np.ndarray, high_text: str,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Optional[str], str, int, float]:
    """Refine the first observed generation flip along one spatial step.

    ``low_coords`` must be a checked non-target state and ``high_coords`` a
    checked target-generating state.  We binary-search only this final segment;
    this is a local refinement, not a claim of a global minimum over all paths.
    """
    low = np.asarray(low_coords, dtype=np.float64).copy()
    high = np.asarray(high_coords, dtype=np.float64).copy()
    best_pred: Optional[str] = target
    best_text = str(high_text)
    checks = 0

    n = max(0, int(args.binary_search_steps))
    for _ in range(n):
        mid = 0.5 * (low + high)
        pred, text = generate_with_coords(
            model_alias=model_alias, model=model, decoder_layers=decoder_layers,
            prepared=prepared, layers=layers, geom=geom, coords=mid, args=args,
        )
        checks += 1
        if pred == target:
            high = mid
            best_pred = pred
            best_text = text
        else:
            low = mid

    full = np.asarray(high_coords, dtype=np.float64) - np.asarray(low_coords, dtype=np.float64)
    refined = high - np.asarray(low_coords, dtype=np.float64)
    denom = float(np.linalg.norm(full.reshape(-1)))
    alpha = float(np.linalg.norm(refined.reshape(-1)) / denom) if denom > 0.0 else 1.0
    return high, best_pred, best_text, checks, alpha


def optimize_to_target(
    *, model_alias: str, model: Any, decoder_layers: Sequence[Any], prepared: Prepared,
    candidate_ids: Mapping[str, Sequence[int]], fast_token_ids: Optional[Mapping[str, Sequence[int]]],
    layers: Sequence[int], geom: Mapping[int, Any], target: str,
    base_gen_pred: Optional[str], base_gen_text: str, args: argparse.Namespace,
) -> Dict[str, Any]:
    """Iterative minimum-residual search whose only success criterion is generation.

    Relation-score margins and their gradients define a cheap local search
    direction in the 2*L spatial coordinates.  The first goal is to cross the
    target-vs-competitor relation-score boundary.  Crucially, that boundary does
    NOT terminate the search.  If actual autoregressive generation still does
    not match ``target``, we increase the required target margin and solve a new
    local minimum-norm step.  Once generation first flips, the final step is
    binary-searched to approximate the smallest generation-flipping point along
    that segment.
    """
    coords = np.zeros((len(layers), 2), dtype=np.float64)
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
            # Current relation scores + exact margin gradients in the spatial
            # coordinates.  The fast path shares one prompt forward graph.
            if fast_token_ids is not None:
                dev = prepared.batch["input_ids"].device
                delta = torch.zeros((len(layers), 2), device=dev, dtype=torch.float32, requires_grad=True)
                base = torch.as_tensor(coords, device=dev, dtype=torch.float32)
                current = base + delta
                with SpatialPatch(
                    decoder_layers=decoder_layers, layers=layers, geom=geom,
                    sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=current,
                ):
                    out = model(**prepared.batch, use_cache=False, return_dict=True)
                    last = out.logits[0, -1].float()
                    score_tensors = {
                        r: torch.stack([last[int(tok)] for tok in fast_token_ids[r]]).max()
                        for r in REL
                    }
                    scores = {r: float(score_tensors[r].detach().item()) for r in REL}
                    competitors = [r for r in REL if r != target]
                    margins = np.asarray(
                        [scores[target] - scores[r] for r in competitors], dtype=np.float64
                    )
                    last_scores = dict(scores)
                    last_pre_step_margin = float(np.min(margins))

                    # A relation-score boundary crossing is only a waypoint.
                    # If generation is still wrong, push the target deeper into
                    # the relation decision region and continue.
                    if last_pre_step_margin >= required_margin - float(args.qp_feas_tol):
                        inc = max(float(args.generation_margin_increment), 1e-8)
                        required_margin = max(required_margin + inc, last_pre_step_margin + inc)
                        max_required_margin = max(max_required_margin, required_margin)
                        n_margin_deepenings += 1

                    b = required_margin - margins
                    grad_rows: List[np.ndarray] = []
                    for j, r in enumerate(competitors):
                        margin_t = score_tensors[target] - score_tensors[r]
                        g = torch.autograd.grad(
                            margin_t, delta,
                            retain_graph=(j + 1 < len(competitors)),
                            create_graph=False,
                        )[0]
                        grad_rows.append(
                            g.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
                        )
                    A = np.stack(grad_rows, axis=0)
                    n_gradient_evals += 1
                del out, last, score_tensors, delta, current, base
            else:
                scores, A, competitors = sequence_relation_scores_and_grads(
                    model=model, decoder_layers=decoder_layers, prepared=prepared,
                    candidate_ids=candidate_ids, layers=layers, geom=geom,
                    base_coords=coords, target=target,
                    reduction=args.sequence_score_reduction,
                )
                margins = np.asarray(
                    [scores[target] - scores[r] for r in competitors], dtype=np.float64
                )
                last_scores = dict(scores)
                last_pre_step_margin = float(np.min(margins))
                if last_pre_step_margin >= required_margin - float(args.qp_feas_tol):
                    inc = max(float(args.generation_margin_increment), 1e-8)
                    required_margin = max(required_margin + inc, last_pre_step_margin + inc)
                    max_required_margin = max(max_required_margin, required_margin)
                    n_margin_deepenings += 1
                b = required_margin - margins
                n_gradient_evals += 1

            flat_step, qp_info = solve_min_norm_halfspaces(
                A, b,
                feas_tol=float(args.qp_feas_tol),
                lambda_tol=float(args.qp_lambda_tol),
            )
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
                # Generation is still wrong, so a numerically zero spatial step
                # means the local score model cannot provide a useful direction.
                stop_reason = "minimum_residual_step_too_small_before_generation_flip"
                break

            # Trust region: the local linearization may ask for an enormous jump
            # when the decision-margin gradient becomes very small.  Preserve the
            # QP direction and its relative layer allocation, but cap the actual
            # residual-space movement and immediately relinearize at the new state.
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

            prev_coords = coords.copy()  # checked non-target state
            step = flat_step.reshape(len(layers), 2)
            proposed_coords = coords + step
            steps = step_idx
            last_step_norm = raw_step_norm
            active_idx = qp_info.get("active_constraints") or []
            last_active_names = [competitors[int(i)] for i in active_idx]
            linearized_post = margins + A @ flat_step
            last_predicted_post_step_margin = float(np.min(linearized_post))

            # Actual generation is the ONLY success test.  We intentionally
            # check every accepted step, regardless of the compatibility flag.
            proposed_pred, proposed_text = generate_with_coords(
                model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                prepared=prepared, layers=layers, geom=geom, coords=proposed_coords, args=args,
            )
            n_generation_checks += 1

            if proposed_pred == target:
                first_flip_total_residual_norm = float(np.linalg.norm(proposed_coords.reshape(-1)))
                # The generation boundary lies somewhere on the final segment
                # from prev_coords (known non-target) to proposed_coords (target).
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

            # No flip: accept the new spatial state and relinearize there.
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
            "fraction_any_step_capped": float((g["capped_steps"].astype(float) > 0).mean()),
            "mean_capped_steps": float(g["capped_steps"].astype(float).mean()),
            "mean_max_requested_boundary_step_norm": float(g["max_requested_boundary_step_norm"].astype(float).mean()),
            "fraction_target_reached": float(comply.mean()),
            "fraction_max_steps_without_target": float(
                (g["stop_reason"] == "max_steps_without_generation_target").mean()
            ),
            "mean_margin_deepenings": float(g["margin_deepenings"].mean()),
            "mean_binary_search_checks": float(g["binary_search_checks"].mean()),
            "mean_gradient_evaluations": float(g["gradient_evaluations"].mean()),
            "mean_generation_checks": float(g["generation_checks"].mean()),
            "mean_runtime_seconds": float(g["runtime_seconds"].mean()) if "runtime_seconds" in g else float("nan"),
        })
    return pd.DataFrame(rows)


def run_pair(
    *, model_alias: str, dataset: str, model: Any, processor_or_backend: Any,
    decoder_layers: Sequence[Any], layers: Sequence[int], geom: Mapping[int, Any],
    hs_summary: Mapping[str, Any], routing: Mapping[int, Mapping[str, Any]],
    target_rows_all: Sequence[Mapping[str, Any]], args: argparse.Namespace, root_out: Path,
) -> dict:
    modes=[x.strip() for x in str(args.modes).split(",") if x.strip()]
    bad=[x for x in modes if x not in {"head","oracle"}]
    if bad: raise ValueError(f"Unknown modes {bad}")
    target_rows = list(target_rows_all)
    if "head" in modes:
        target_rows = [r for r in target_rows if int(r["sid"]) in routing]
    else:
        hs_summary = {"best_head": None, "best_all_target_acc": float("nan")}
        routing = {
            int(r["sid"]): {"sid": int(r["sid"]), "gt": r["relation"], "head_pred": r["relation"]}
            for r in target_rows
        }
    target_rows=stratified_cap(target_rows,int(args.eval_max_samples),int(args.eval_seed))
    if not target_rows: raise RuntimeError(f"No target rows after routing alignment for {model_alias}/{dataset}")

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
    tokenizer = processor_or_backend.tokenizer
    candidate_ids = encode_candidates(tokenizer, dataset)
    fast_token_ids, fast_token_details = encode_fast_relation_tokens(tokenizer, dataset)

    print("\n" + "=" * 180)
    print(f"CONTROL {model_alias} / {dataset} | selected_head={hs_summary.get('best_head')} | head_acc(all)={float(hs_summary.get('best_all_target_acc', float('nan'))):.4f}")
    print(f"Synthetic-3D H/D geometry layers={list(layers)} | eval N={len(target_rows)}")
    if fast_token_ids is not None:
        print(f"FAST SCORE PATH: one prompt forward for four relation scores | token_variants={fast_token_ids}")
    else:
        print("FALLBACK SCORE PATH: at least one relation is multi-token; using exact teacher-forced sequence scores")
        print(json.dumps(fast_token_details, ensure_ascii=False))
    print("Synthetic-3D H/D geometry -> relation target -> minimum residual H/D step -> deepen score margin until ACTUAL generation flips -> trust-cap each local step -> binary-refine final step")
    print("=" * 180, flush=True)

    pbar = tqdm(target_rows, desc=f"CONTROL {model_alias}:{dataset}")
    processed_since_cleanup = 0
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
            base_pred, base_text = generate_with_coords(
                model_alias=model_alias, model=model, decoder_layers=decoder_layers,
                prepared=prepared, layers=layers, geom=geom, coords=zero, args=args,
            )

            # The expensive optimization depends only on the control target.
            # When head prediction == GT, head and oracle are exactly the same
            # intervention and are computed once, then reused.
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
                append_jsonl(result_path, outrow)
                done.add((sid, mode))

            sample_sec = time.perf_counter() - sample_t0
            latest = next(reversed(result_by_target.values())) if result_by_target else {}
            pbar.set_postfix(
                sec=f"{sample_sec:.1f}",
                steps=int(latest.get("steps_taken", 0)),
                stop=str(latest.get("stop_reason", "-"))[:18],
                refresh=True,
            )
        except Exception as exc:
            append_jsonl(error_path, {
                "sid": sid, "model": model_alias, "dataset": dataset,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-60:],
            })
            tqdm.write(f"[ERROR] {model_alias}/{dataset} sid={sid}: {type(exc).__name__}: {exc}")
            cleanup_cuda()
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
            cleanup_cuda()
            processed_since_cleanup = 0

    # One cleanup at pair end, not after every sample.
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
        "reader_dir": str(root_out / "reader" / model_alias / "controlled_b"),
        "selected_head": hs_summary.get("best_head"),
        "selected_head_target_acc": hs_summary.get("best_all_target_acc"),
        "direction_source": "Blender Synthetic-3D left/right/front/behind",
        "head_id_selection": "full target GT",
        "per_sample_head_routing_uses_target_gt": False,
        "residual_geometry_source": "Blender Synthetic-3D left/right/front/behind",
        "geometry_control": "gray",
        "controller_layers": list(map(int, layers)),
        "controller_dim": int(2 * len(layers)),
        "optimizer": "iterative_minimum_residual_generation_flip_with_trust_region",
        "spatial_basis": "orthonormal basis spanning Synthetic-3D horizontal/depth directions",
        "uses_natural_gap_scaling": False,
        "single_token_shared_forward": bool(fast_token_ids is not None),
        "relation_token_ids": fast_token_ids,
        "relation_token_details": fast_token_details,
        "decision_margin_eps": float(args.decision_margin_eps),
        "generation_margin_increment": float(args.generation_margin_increment),
        "binary_search_steps": int(args.binary_search_steps),
        "actual_generation_is_only_success_criterion": True,
        "qp_feas_tol": float(args.qp_feas_tol),
        "min_residual_step": float(args.min_residual_step),
        "max_residual_step": float(args.max_residual_step),
        "trust_region_enabled": bool(float(args.max_residual_step) > 0.0),
        "generation_check_every": int(args.generation_check_every),
        "max_steps": int(args.max_steps),
        "cuda_cleanup_every": int(args.cuda_cleanup_every),
    }
    write_json(outdir / "metadata.json", meta)
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


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if str(args.control) != "gray":
        raise ValueError("This protocol requires Real-Gray source residuals")
    if not (0.0 < float(args.head_selection_frac) <= 1.0):
        raise ValueError("--head-selection-frac must be in (0,1]")

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

    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    bad = [x for x in modes if x not in {"head", "oracle"}]
    if bad:
        raise ValueError(f"Unknown modes {bad}")

    source_rows = load_synthetic_3d(args, limit=args.source_max_samples)
    target_rows_all = load_controlled_b(args, limit=args.target_max_samples)
    print(
        f"[PROTOCOL] source=Blender Synthetic-3D LR/front/behind only | "
        f"source N={len(source_rows)} | Controlled-B N={len(target_rows_all)} | modes={modes}"
    )

    summaries: List[dict] = []
    failures: List[dict] = []

    for model_alias in MODEL_ORDER:
        model_pairs = [(m, d) for m, d in pairs if m == model_alias]
        if not model_pairs:
            continue
        model = processor = None
        try:
            model, processor, decoder_layers, n_heads, head_dim, spec = load_model_bundle(model_alias, args)
            if model_alias.startswith("internvl-"):
                raise RuntimeError(
                    "This standalone Synthetic-3D reader currently supports Qwen/LLaVA only. "
                    "Use qwen2-2b,qwen-3b,qwen-7b,llava-7b,llava-13b."
                )
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

            # A) Source-trained direction reader + target-supervised head ID selection.
            if "head" in modes:
                hs_summary, routing = build_standalone_routing(
                    model_alias=model_alias,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    source_rows=source_rows,
                    target_rows=target_rows_all,
                    root_out=root_out,
                    args=args,
                )
            else:
                hs_summary = {"best_head": None, "best_all_target_acc": float("nan")}
                routing = {
                    int(r["sid"]): {
                        "sid": int(r["sid"]),
                        "gt": str(r["relation"]),
                        "head_pred": str(r["relation"]),
                    }
                    for r in target_rows_all
                }

            # B) Source-only decoder residual H/D actuator geometry.
            geom_cache = geom_root / model_alias / "synthetic3d_LRFB_residual_real_minus_gray.npz"
            Xg, yg = build_synthetic_geometry_cache(
                model_alias=model_alias,
                model=model,
                processor_or_backend=processor,
                decoder_layers=decoder_layers,
                layers=layers,
                source_rows=source_rows,
                cache_path=geom_cache,
                args=args,
            )
            geom, axis_df = fit_geometry(Xg, yg, layers)
            model_geom_dir = geom_root / model_alias
            model_geom_dir.mkdir(parents=True, exist_ok=True)
            axis_df.to_csv(model_geom_dir / "synthetic3d_HD_geometry.csv", index=False)

            # C) Controlled-B generation recovery.
            for _m, dataset in model_pairs:
                try:
                    result = run_pair(
                        model_alias=model_alias,
                        dataset=dataset,
                        model=model,
                        processor_or_backend=processor,
                        decoder_layers=decoder_layers,
                        layers=layers,
                        geom=geom,
                        hs_summary=hs_summary,
                        routing=routing,
                        target_rows_all=target_rows_all,
                        args=args,
                        root_out=root_out,
                    )
                    summaries = [
                        x for x in summaries
                        if not (x.get("model") == model_alias and x.get("dataset") == dataset)
                    ]
                    summaries.append(result)
                    write_csv(root_out / "cross_model_control_summary.csv", summaries)
                    write_json(root_out / "cross_model_control_summary.json", summaries)
                except Exception as exc:
                    failure = {
                        "model": model_alias,
                        "dataset": dataset,
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
                "model": model_alias,
                "dataset": "__model_reader_geometry__",
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
    print("FINAL SYNTHETIC-3D -> CONTROLLED-B CONTROL SUMMARY")
    print("=" * 180)
    if summaries:
        print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("NO SUCCESSFUL PAIRS")
    print(f"\nSaved to: {root_out}")


if __name__ == "__main__":
    main()
