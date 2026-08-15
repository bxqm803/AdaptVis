#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-model / cross-dataset spatial-relation logit-lens scan.

Goal
----
Run the same layer-wise TRUE logit-lens analysis on a matrix of VLMs and
spatial datasets, without per-head scans.

For every decoder layer L and the prompt-last position, capture:
    x_L        : decoder block input
    a_L        : self-attention output AFTER output projection (component only)
    r_L=x_L+a_L: residual state after attention
    m_L        : MLP output (component only)
    y_L        : decoder block output

Primary analysis uses actual residual states:
    x_L -> r_L -> y_L
and reads each state with:
    final language-model norm -> LM head -> {left,right,above,below}

The component-only a_L / m_L lenses are also saved, but they are secondary:
normalizing an isolated residual update is not the state actually consumed by
later layers. Use x->r and r->y gains for the main mechanistic comparison.

Default model matrix (requested aliases):
    llava-7b, llava-13b,
    qwen2-2b, qwen-3b, qwen-7b,
    internvl-1b, internvl-2b, internvl-7b

Model loading reuses extract_two_object_relation_states.py::SPECS from the
AdaptVis repository.  This avoids duplicating repo IDs/model classes here.
Alias resolution is tolerant; e.g. requested internvl-7b can fall back to an
available internvl-8b alias in SPECS if that is how the project names it.

Datasets:
    coco_two
    controlled_a

COCO is loaded with extract_two_object_relation_states.load_records.
Controlled-A first tries the same project loader using several common aliases.
If unavailable, it falls back to a flexible JSON/JSONL manifest loader.  Pass
--controlled-json explicitly if auto-discovery cannot find the manifest.

Two-GPU launcher
----------------
The default --mode matrix launches ONE model process per GPU at a time.  Each
model is loaded only once and then runs all unfinished datasets.  GPU workers
dynamically pull models from a queue, so large/small models are naturally
balanced across the two GPUs.

Example:
    python -u scan_multimodel_spatial_logitlens_matrix_v1.py \
      --mode matrix \
      --gpus 0,1 \
      --data-root data \
      --output-root output/spatial_logitlens_matrix_v1

Known legacy skip
-----------------
By default, qwen-7b / coco_two is skipped IF this directory already exists and
looks completed:
    output/qwen7b_correct_wrong_logitlens_trajectory_v1

This is the Qwen-7B COCO true-logit-lens run already performed previously.
Disable that behavior with --no-default-legacy-skip, or add extra completed
runs with repeated:
    --skip-job model:dataset:path

A new-format job is automatically skipped when <job_dir>/DONE exists.
Use --force to rerun completed new-format jobs.

Outputs per model/dataset
-------------------------
    sample_summary.csv
    layer_trajectory.csv
    layer_summary.csv
    group_layer_summary.csv
    config.json
    report.txt
    errors.jsonl            (only if errors)
    DONE                    (only after successful completion)

Matrix-level outputs
--------------------
    matrix_summary.csv
    matrix_layer_summary.csv
    matrix_status.json

Notes
-----
* No per-head hooks/scans are performed.
* All decoder layers are scanned by default.
* Generation is OFF by default; grouping uses the model's native restricted
  first-step relation prediction at the prompt end.  --run-generation is
  available as an optional slower check.
* The script validates the residual decomposition with
      y_L ~= x_L + a_L + m_L
  and reports relative reconstruction error.  Large error means the model's
  decoder block is not compatible with the assumed Llama/Qwen-style residual
  decomposition, so x/r/y should not be interpreted as attention/MLP stages
  without inspecting that architecture.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


SCRIPT_VERSION = "spatial-logitlens-matrix-v1"
RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}

DEFAULT_MODELS = (
    "llava-7b",
    "llava-13b",
    "qwen2-2b",
    "qwen-3b",
    "qwen-7b",
    "internvl-1b",
    "internvl-2b",
    "internvl-7b",
)
DEFAULT_DATASETS = ("coco_two", "controlled_a")

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)

# User-requested names -> likely aliases in the existing AdaptVis SPECS table.
# Resolution uses the first alias actually present in base.SPECS.
MODEL_ALIAS_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "llava-7b": ("llava-7b", "llava15-7b", "llava-v1.5-7b"),
    "llava-13b": ("llava-13b", "llava15-13b", "llava-v1.5-13b"),
    "qwen2-2b": (
        "qwen2-2b", "qwen2vl-2b", "qwen2-vl-2b", "qwen-2b",
        "qwen2.0-2b", "qwen2_vl_2b",
    ),
    "qwen-3b": ("qwen-3b", "qwen2.5-3b", "qwen25-3b", "qwen2.5-vl-3b"),
    "qwen-7b": ("qwen-7b", "qwen2.5-7b", "qwen25-7b", "qwen2.5-vl-7b"),
    "internvl-1b": ("internvl-1b", "internvl2.5-1b", "internvl25-1b"),
    "internvl-2b": ("internvl-2b", "internvl2.5-2b", "internvl25-2b"),
    # Some InternVL releases are called 8B rather than 7B in local SPECS.
    "internvl-7b": (
        "internvl-7b", "internvl-8b", "internvl2.5-8b", "internvl25-8b",
        "internvl2-8b",
    ),
}

# Completed legacy run known from the current experiment history.
DEFAULT_LEGACY_SKIPS = {
    ("qwen-7b", "coco_two"): Path("output/qwen7b_correct_wrong_logitlens_trajectory_v1"),
}


# =============================================================================
# Generic helpers
# =============================================================================


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_layers(text: str, n_layers: int) -> List[int]:
    text = str(text).strip().lower()
    if text in ("", "all", "*"):
        return list(range(n_layers))
    out: List[int] = []
    seen = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = map(int, chunk.split("-", 1))
            step = 1 if b >= a else -1
            vals = range(a, b + step, step)
        else:
            vals = [int(chunk)]
        for v in vals:
            if v < 0:
                v = n_layers + v
            if not (0 <= v < n_layers):
                raise ValueError(f"Layer L{v} outside [0,{n_layers-1}]")
            if v not in seen:
                out.append(v)
                seen.add(v)
    return sorted(out)


def safe_mean(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            y = float(x)
        except Exception:
            continue
        if math.isfinite(y):
            vals.append(y)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            y = float(x)
        except Exception:
            continue
        if math.isfinite(y):
            vals.append(y)
    return float(np.median(vals)) if vals else float("nan")


def safe_max(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            y = float(x)
        except Exception:
            continue
        if math.isfinite(y):
            vals.append(y)
    return max(vals) if vals else float("nan")


def normalize_relation(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().replace("-", "_")
    aliases = {
        "left": "left", "left_of": "left", "to_the_left_of": "left",
        "right": "right", "right_of": "right", "to_the_right_of": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "on_top_of": "above", "upper": "above",
        "below": "below", "under": "below", "underneath": "below",
        "beneath": "below", "bottom": "below", "lower": "below",
    }
    return aliases.get(s, s if s in RELATIONS else None)


def first_tensor(obj: Any) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for x in obj:
            try:
                return first_tensor(x)
            except Exception:
                pass
    # Some HF outputs expose last_hidden_state / hidden_states-like attrs.
    for name in ("last_hidden_state", "hidden_state", "hidden_states"):
        x = getattr(obj, name, None)
        if torch.is_tensor(x):
            return x
        if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
            return x[0]
    raise TypeError(f"Could not find tensor in output type {type(obj).__name__}")


def to_last_vector(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:      # [B,T,H]
        return t[0, -1]
    if t.ndim == 2:      # [T,H] or [B,H]
        return t[-1]
    if t.ndim == 1:
        return t
    raise RuntimeError(f"Unexpected hidden tensor shape {tuple(t.shape)}")


def get_attr_path(root: Any, path: str) -> Any:
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def relation_metrics(scores: np.ndarray, gt: str) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(4)
    g = RID[gt]
    wrong = [i for i in range(4) if i != g]
    comp = max(wrong, key=lambda i: float(s[i]))
    pred = int(np.argmax(s))
    centered = s - np.max(s)
    p = np.exp(centered)
    p /= p.sum()
    return {
        "pred": RELATIONS[pred],
        "correct": bool(pred == g),
        "decision_margin": float(s[g] - s[comp]),
        "opposite_margin": float(s[g] - s[RID[OPPOSITE[gt]]]),
        "p_gt_4way": float(p[g]),
        "top_competitor": RELATIONS[comp],
    }


def parse_generated_relation(text: str) -> Optional[str]:
    s = str(text).lower()
    hits = []
    for r in RELATIONS:
        m = re.search(rf"\b{re.escape(r)}\b", s)
        if m:
            hits.append((m.start(), r))
    # Accept common above/below aliases in free generation.
    for token, r in (("under", "below"), ("underneath", "below"), ("beneath", "below"), ("on top", "above")):
        pos = s.find(token)
        if pos >= 0:
            hits.append((pos, r))
    return min(hits)[1] if hits else None


# =============================================================================
# Dataset loading
# =============================================================================

@dataclass
class ScanRecord:
    sid: int
    image_path: Path
    relation: str
    subject: str = ""
    reference: str = ""
    question: str = ""


def adapt_base_record(rec: Any) -> Optional[ScanRecord]:
    rel = normalize_relation(getattr(rec, "relation", None))
    if rel not in RELATIONS:
        return None
    image_path = Path(getattr(rec, "image_path"))
    return ScanRecord(
        sid=int(getattr(rec, "sid")),
        image_path=image_path,
        relation=rel,
        subject=str(getattr(rec, "subject", "") or ""),
        reference=str(getattr(rec, "reference", "") or ""),
        question=str(getattr(rec, "question", "") or ""),
    )


def try_base_load(base: Any, aliases: Sequence[str], data_root: Path, max_samples: Optional[int]) -> Tuple[Optional[List[ScanRecord]], List[Any], Optional[str]]:
    last_error = None
    for alias in aliases:
        try:
            recs, audit = base.load_records(alias, data_root, max_samples)
            out = [x for r in recs if (x := adapt_base_record(r)) is not None]
            if out:
                return out, list(audit), alias
        except Exception as exc:
            last_error = exc
    return None, ([{"base_loader_error": repr(last_error)}] if last_error else []), None


def discover_controlled_manifest(data_root: Path, explicit: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    roots = [
        data_root,
        data_root / "Controlled_Images_A",
        data_root / "controlled_images_a",
        data_root / "controlled_a",
        data_root / "controlled",
    ]
    names = [
        "controlled_a.jsonl", "controlled_a.json",
        "Controlled_Images_A.jsonl", "Controlled_Images_A.json",
        "metadata.jsonl", "metadata.json",
        "annotations.jsonl", "annotations.json",
        "data.jsonl", "data.json",
    ]
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return p
    return None


def load_json_rows(path: Path) -> List[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    x = json.loads(line)
                    if isinstance(x, Mapping):
                        rows.append(x)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, Mapping)]
    if isinstance(data, Mapping):
        for key in ("records", "data", "samples", "annotations", "items"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, Mapping)]
        # Dict keyed by sample id.
        if all(isinstance(v, Mapping) for v in data.values()):
            out = []
            for k, v in data.items():
                d = dict(v)
                d.setdefault("sid", k)
                out.append(d)
            return out
    raise RuntimeError(f"Unsupported manifest structure: {path}")


def pick(row: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def resolve_image_path(raw: Any, manifest: Path, data_root: Path) -> Path:
    if raw is None:
        raise RuntimeError("No image/image_path/filename field")
    p = Path(str(raw))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates += [
            manifest.parent / p,
            data_root / p,
            data_root / "Controlled_Images_A" / p,
            data_root / "Controlled_Images_A" / "images" / p,
            data_root / "controlled_a" / p,
            data_root / "controlled_a" / "images" / p,
        ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else p


def load_controlled_manifest(path: Path, data_root: Path, max_samples: Optional[int]) -> Tuple[List[ScanRecord], List[Dict[str, Any]]]:
    rows = load_json_rows(path)
    records: List[ScanRecord] = []
    audit: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            sid_raw = pick(row, ("sid", "id", "sample_id", "index"), idx)
            try:
                sid = int(sid_raw)
            except Exception:
                sid = idx
            rel = normalize_relation(pick(row, ("relation", "answer", "label", "direction", "gt", "gold")))
            if rel not in RELATIONS:
                raise RuntimeError(f"Unsupported relation={pick(row, ('relation','answer','label','direction','gt','gold'))!r}")
            image_raw = pick(row, ("image_path", "image", "img", "file_name", "filename", "path"))
            image_path = resolve_image_path(image_raw, path, data_root)
            subject = str(pick(row, ("subject", "object1", "obj1", "source", "target_object", "a", "A"), "") or "")
            reference = str(pick(row, ("reference", "object2", "obj2", "target", "reference_object", "b", "B"), "") or "")
            question = str(pick(row, ("question", "prompt", "query", "text"), "") or "")
            if not question and not (subject and reference):
                raise RuntimeError("Need either question/prompt or subject+reference")
            records.append(ScanRecord(sid, image_path, rel, subject, reference, question))
        except Exception as exc:
            audit.append({"row": idx, "error": str(exc)})
    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]
    return records, audit


def load_dataset_records(base: Any, dataset: str, data_root: Path, controlled_json: str, max_samples: Optional[int]) -> Tuple[List[ScanRecord], List[Any], Dict[str, Any]]:
    if dataset == "coco_two":
        recs, audit = base.load_records("coco_two", data_root, max_samples)
        out = [x for r in recs if (x := adapt_base_record(r)) is not None]
        return out, list(audit), {"loader": "base.load_records", "base_dataset_alias": "coco_two"}

    if dataset != "controlled_a":
        raise ValueError(f"Unknown dataset: {dataset}")

    aliases = ("controlled_a", "controlled", "controlled_images_a", "controlled_two", "controlled_A")
    out, audit, alias = try_base_load(base, aliases, data_root, max_samples)
    if out:
        return out, audit, {"loader": "base.load_records", "base_dataset_alias": alias}

    manifest = discover_controlled_manifest(data_root, controlled_json)
    if manifest is None:
        raise RuntimeError(
            "Controlled-A could not be loaded by extract_two_object_relation_states.load_records "
            "and no manifest was auto-discovered. Pass --controlled-json /path/to/manifest.json[l]."
        )
    out, manifest_audit = load_controlled_manifest(manifest, data_root, max_samples)
    return out, audit + manifest_audit, {"loader": "flexible_manifest", "manifest": str(manifest)}


def select_records(records: Sequence[ScanRecord], n: int, seed: int) -> List[ScanRecord]:
    records = list(records)
    if n <= 0 or n >= len(records):
        return records
    rng = random.Random(seed)
    # Stratified by relation to keep scans comparable.
    groups: Dict[str, List[ScanRecord]] = defaultdict(list)
    for r in records:
        groups[r.relation].append(r)
    for xs in groups.values():
        rng.shuffle(xs)
    out: List[ScanRecord] = []
    cursors = {r: 0 for r in RELATIONS}
    while len(out) < n:
        moved = False
        for rel in RELATIONS:
            xs = groups.get(rel, [])
            i = cursors[rel]
            if i < len(xs) and len(out) < n:
                out.append(xs[i])
                cursors[rel] += 1
                moved = True
        if not moved:
            break
    return sorted(out, key=lambda r: r.sid)


# =============================================================================
# Model / processor / lens helpers
# =============================================================================


def norm_alias(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def resolve_model_alias(requested: str, specs: Mapping[str, Any]) -> str:
    if requested in specs:
        return requested
    for cand in MODEL_ALIAS_CANDIDATES.get(requested, (requested,)):
        if cand in specs:
            return cand
    # Exact normalized match.
    target_norms = {norm_alias(x) for x in MODEL_ALIAS_CANDIDATES.get(requested, (requested,))}
    matches = [k for k in specs if norm_alias(k) in target_norms]
    if len(matches) == 1:
        return matches[0]
    # Controlled fallback for requested InternVL 7B -> available InternVL 8B.
    if requested == "internvl-7b":
        cand = [k for k in specs if "internvl" in k.lower() and ("8b" in k.lower() or "7b" in k.lower())]
        if len(cand) == 1:
            return cand[0]
    raise KeyError(
        f"Could not resolve requested model alias {requested!r}. "
        f"Available base.SPECS aliases: {sorted(specs)}"
    )


def resolve_decoder_layers(model: Any) -> Tuple[Sequence[Any], str]:
    candidates = [
        ("model.language_model.model.layers", lambda m: m.model.language_model.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("model.model.model.layers", lambda m: m.model.model.model.layers),
        ("llm.model.layers", lambda m: m.llm.model.layers),
        ("model.llm.model.layers", lambda m: m.model.llm.model.layers),
    ]
    for path, fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers):
                return layers, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve language decoder layers")


def resolve_self_attention(layer: Any) -> torch.nn.Module:
    for name in ("self_attn", "attention", "attn"):
        m = getattr(layer, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    raise RuntimeError(f"Could not resolve self-attention in {type(layer).__name__}")


def resolve_mlp(layer: Any) -> torch.nn.Module:
    for name in ("mlp", "feed_forward", "ffn", "feedforward"):
        m = getattr(layer, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    raise RuntimeError(f"Could not resolve MLP in {type(layer).__name__}")


def resolve_final_norm(model: Any, decoder_path: str) -> Tuple[torch.nn.Module, str]:
    parent = decoder_path.rsplit(".", 1)[0] if "." in decoder_path else ""
    candidates = []
    if parent:
        candidates += [f"{parent}.norm", f"{parent}.final_layernorm", f"{parent}.ln_f"]
    candidates += [
        "model.language_model.model.norm",
        "model.language_model.norm",
        "language_model.model.norm",
        "language_model.norm",
        "model.model.norm",
        "model.norm",
        "model.text_model.norm",
        "text_model.norm",
        "llm.model.norm",
        "model.llm.model.norm",
    ]
    for path in dict.fromkeys(candidates):
        m = get_attr_path(model, path)
        if isinstance(m, torch.nn.Module):
            return m, path
    raise RuntimeError("Could not resolve final language-model norm")


def resolve_output_embeddings(model: Any) -> Tuple[torch.nn.Module, str]:
    candidates: List[Tuple[Any, str]] = [(model, "model")]
    for path in (
        "language_model", "model.language_model", "model.model.language_model",
        "llm", "model.llm", "model", "model.model",
    ):
        obj = get_attr_path(model, path)
        if obj is not None:
            candidates.append((obj, path))
    for obj, path in candidates:
        fn = getattr(obj, "get_output_embeddings", None)
        if callable(fn):
            try:
                emb = fn()
                if isinstance(emb, torch.nn.Module) and hasattr(emb, "weight"):
                    return emb, f"{path}.get_output_embeddings()"
            except Exception:
                pass
    for path in (
        "lm_head", "language_model.lm_head", "model.language_model.lm_head",
        "llm.lm_head", "model.llm.lm_head",
    ):
        emb = get_attr_path(model, path)
        if isinstance(emb, torch.nn.Module) and hasattr(emb, "weight"):
            return emb, path
    raise RuntimeError("Could not resolve LM head / output embeddings")


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for relation in RELATIONS:
        ids = set()
        for text in (relation, " " + relation, relation.capitalize(), " " + relation.capitalize()):
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                token_ids = []
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))
        if not ids:
            token_ids = tokenizer.encode(" " + relation, add_special_tokens=False)
            if not token_ids:
                raise RuntimeError(f"No token variant for relation={relation}")
            ids.add(int(token_ids[-1]))
        out[relation] = sorted(ids)
    return out


class RelationLogitLens:
    def __init__(self, final_norm: torch.nn.Module, lm_head: torch.nn.Module, token_map: Mapping[str, Sequence[int]]):
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.token_map = {k: list(map(int, v)) for k, v in token_map.items()}
        weight = lm_head.weight
        union = sorted({i for ids in self.token_map.values() for i in ids})
        self.union = union
        self.lookup = {tid: i for i, tid in enumerate(union)}
        idx = torch.tensor(union, device=weight.device, dtype=torch.long)
        self.weight_rows = weight.index_select(0, idx).detach()
        bias = getattr(lm_head, "bias", None)
        self.bias_rows = bias.index_select(0, idx).detach() if torch.is_tensor(bias) else None
        self.device = weight.device
        self.dtype = weight.dtype

    @torch.inference_mode()
    def scores(self, states: np.ndarray) -> np.ndarray:
        a = np.asarray(states, dtype=np.float32)
        lead = a.shape[:-1]
        flat = a.reshape(-1, a.shape[-1])
        t = torch.as_tensor(flat, device=self.device, dtype=self.dtype)
        n = self.final_norm(t)
        if isinstance(n, (tuple, list)):
            n = n[0]
        logits = n @ self.weight_rows.T
        if self.bias_rows is not None:
            logits = logits + self.bias_rows
        cols = []
        for rel in RELATIONS:
            js = [self.lookup[i] for i in self.token_map[rel]]
            cols.append(logits[:, js].max(dim=-1).values)
        out = torch.stack(cols, dim=-1)
        return out.detach().float().cpu().numpy().reshape(*lead, 4)

    def scores_from_full_logits(self, logits: torch.Tensor) -> np.ndarray:
        # logits shape [V] or [1,V]
        x = logits.detach()
        if x.ndim == 2:
            x = x[0]
        vals = []
        for rel in RELATIONS:
            idx = torch.tensor(self.token_map[rel], device=x.device, dtype=torch.long)
            vals.append(x.index_select(0, idx).max())
        return torch.stack(vals).float().cpu().numpy()


# =============================================================================
# Prompt / batch
# =============================================================================


def render_question(rec: ScanRecord, prompt_template: str, prefer_manifest_question: bool) -> str:
    if prefer_manifest_question and rec.question:
        return rec.question
    if rec.subject and rec.reference:
        return prompt_template.format(subject=rec.subject, reference=rec.reference)
    if rec.question:
        return rec.question
    raise RuntimeError(f"sid={rec.sid}: no usable question")


def build_batch(probe: Any, processor: Any, question: str, image: Image.Image, device: torch.device) -> Any:
    # Reuse the same generic helper already used by the AdaptVis analysis scripts.
    try:
        rendered = probe.build_chat_prompt(processor, question, True)
        return probe.process_inputs(processor, rendered, image, device)
    except Exception as first_exc:
        # Fallback for processors whose chat-template API differs.
        try:
            batch = processor(text=[question], images=[image], padding=True, return_tensors="pt")
            return batch.to(device)
        except Exception:
            raise RuntimeError(
                f"Both project chat-template processing and direct AutoProcessor processing failed. "
                f"First error: {type(first_exc).__name__}: {first_exc}"
            )


# =============================================================================
# Layer tracing
# =============================================================================


class AllLayerResidualTrace:
    """Capture prompt-last x, attention output, MLP output, block output."""

    def __init__(self, decoder_layers: Sequence[Any], layer_indices: Sequence[int]):
        self.decoder_layers = decoder_layers
        self.layer_indices = list(map(int, layer_indices))
        self.x: Dict[int, torch.Tensor] = {}
        self.a: Dict[int, torch.Tensor] = {}
        self.m: Dict[int, torch.Tensor] = {}
        self.y: Dict[int, torch.Tensor] = {}
        self.handles: List[Any] = []

    @staticmethod
    def cpu_vec(obj: Any) -> torch.Tensor:
        return to_last_vector(first_tensor(obj)).detach().float().cpu()

    def __enter__(self):
        for li in self.layer_indices:
            layer = self.decoder_layers[li]
            attn = resolve_self_attention(layer)
            mlp = resolve_mlp(layer)

            def make_pre(li_: int):
                def hook(_m, inputs):
                    self.x[li_] = self.cpu_vec(inputs[0])
                return hook

            def make_attn(li_: int):
                def hook(_m, _inputs, output):
                    self.a[li_] = self.cpu_vec(output)
                return hook

            def make_mlp(li_: int):
                def hook(_m, _inputs, output):
                    self.m[li_] = self.cpu_vec(output)
                return hook

            def make_post(li_: int):
                def hook(_m, _inputs, output):
                    self.y[li_] = self.cpu_vec(output)
                return hook

            self.handles.append(layer.register_forward_pre_hook(make_pre(li)))
            self.handles.append(attn.register_forward_hook(make_attn(li)))
            self.handles.append(mlp.register_forward_hook(make_mlp(li)))
            self.handles.append(layer.register_forward_hook(make_post(li)))
        return self

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def finalize(self) -> Dict[str, np.ndarray]:
        for name, d in (("x", self.x), ("a", self.a), ("m", self.m), ("y", self.y)):
            missing = [li for li in self.layer_indices if li not in d]
            if missing:
                raise RuntimeError(f"Missing {name} captures for layers {missing[:10]}")
        x = np.stack([self.x[li].numpy() for li in self.layer_indices])
        a = np.stack([self.a[li].numpy() for li in self.layer_indices])
        m = np.stack([self.m[li].numpy() for li in self.layer_indices])
        y = np.stack([self.y[li].numpy() for li in self.layer_indices])
        r = x + a
        y_recon = r + m
        denom = np.linalg.norm(y, axis=-1) + 1e-12
        recon_rel = np.linalg.norm(y - y_recon, axis=-1) / denom
        return {"x": x, "attn_delta": a, "rattn": r, "mlp_delta": m, "y": y, "recon_rel": recon_rel}


# =============================================================================
# Summaries
# =============================================================================


def wrong_taxonomy(layer_rows_for_sample: Sequence[Mapping[str, Any]], native_correct: bool) -> Tuple[str, Optional[int], Optional[int]]:
    if native_correct:
        return "native_correct", None, None
    ys = [(int(r["layer"]), bool(r["y_correct"])) for r in layer_rows_for_sample]
    correct_layers = [l for l, c in ys if c]
    if not correct_layers:
        return "never_formed", None, None
    final_layer = ys[-1][0]
    if ys[-1][1]:
        return "formed_and_present_final", min(correct_layers), max(correct_layers)
    # If penultimate scanned layer was correct, call it final-layer loss.
    if len(ys) >= 2 and ys[-2][1] and not ys[-1][1]:
        return "lost_at_final_layer", min(correct_layers), max(correct_layers)
    return "formed_then_lost", min(correct_layers), max(correct_layers)


def summarize_layers(layer_rows: Sequence[Mapping[str, Any]], sample_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_layer: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for r in layer_rows:
        by_layer[int(r["layer"])].append(r)

    layer_summary = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        layer_summary.append({
            "layer": layer,
            "N": len(rows),
            "x_acc": safe_mean(float(r["x_correct"]) for r in rows),
            "rattn_acc": safe_mean(float(r["rattn_correct"]) for r in rows),
            "y_acc": safe_mean(float(r["y_correct"]) for r in rows),
            "attn_delta_acc_component_only": safe_mean(float(r["attn_delta_correct"]) for r in rows),
            "mlp_delta_acc_component_only": safe_mean(float(r["mlp_delta_correct"]) for r in rows),
            "x_margin_mean": safe_mean(r["x_decision_margin"] for r in rows),
            "rattn_margin_mean": safe_mean(r["rattn_decision_margin"] for r in rows),
            "y_margin_mean": safe_mean(r["y_decision_margin"] for r in rows),
            "attn_delta_margin_mean_component_only": safe_mean(r["attn_delta_decision_margin"] for r in rows),
            "mlp_delta_margin_mean_component_only": safe_mean(r["mlp_delta_decision_margin"] for r in rows),
            "attention_decision_gain_mean": safe_mean(r["attention_decision_gain"] for r in rows),
            "mlp_decision_gain_mean": safe_mean(r["mlp_decision_gain"] for r in rows),
            "attention_C_to_W": sum(bool(r["attention_C_to_W"]) for r in rows),
            "attention_W_to_C": sum(bool(r["attention_W_to_C"]) for r in rows),
            "mlp_C_to_W": sum(bool(r["mlp_C_to_W"]) for r in rows),
            "mlp_W_to_C": sum(bool(r["mlp_W_to_C"]) for r in rows),
            "recon_rel_mean": safe_mean(r["recon_rel"] for r in rows),
            "recon_rel_max": safe_max(r["recon_rel"] for r in rows),
        })

    sample_map = {int(r["sid"]): r for r in sample_rows}
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for r in layer_rows:
        s = sample_map[int(r["sid"])]
        groups = ["all", "native_correct" if s["native_correct"] else "native_wrong"]
        if not s["native_correct"]:
            groups.append("wrong_" + str(s["wrong_taxonomy"]))
        if s.get("generation_pred") not in (None, ""):
            groups.append("generation_correct" if s.get("generation_correct") else "generation_wrong")
        for g in groups:
            grouped[(g, int(r["layer"]))].append(r)

    group_summary = []
    for (group, layer), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        group_summary.append({
            "group": group,
            "layer": layer,
            "N": len(rows),
            "x_acc": safe_mean(float(r["x_correct"]) for r in rows),
            "rattn_acc": safe_mean(float(r["rattn_correct"]) for r in rows),
            "y_acc": safe_mean(float(r["y_correct"]) for r in rows),
            "x_margin_mean": safe_mean(r["x_decision_margin"] for r in rows),
            "rattn_margin_mean": safe_mean(r["rattn_decision_margin"] for r in rows),
            "y_margin_mean": safe_mean(r["y_decision_margin"] for r in rows),
            "attention_decision_gain_mean": safe_mean(r["attention_decision_gain"] for r in rows),
            "mlp_decision_gain_mean": safe_mean(r["mlp_decision_gain"] for r in rows),
            "attention_C_to_W": sum(bool(r["attention_C_to_W"]) for r in rows),
            "attention_W_to_C": sum(bool(r["attention_W_to_C"]) for r in rows),
            "mlp_C_to_W": sum(bool(r["mlp_C_to_W"]) for r in rows),
            "mlp_W_to_C": sum(bool(r["mlp_W_to_C"]) for r in rows),
        })
    return layer_summary, group_summary


def build_job_summary(model_requested: str, model_actual: str, dataset: str, sample_rows: Sequence[Mapping[str, Any]], layer_summary: Sequence[Mapping[str, Any]], final_match: int, final_match_n: int) -> Dict[str, Any]:
    native_acc = safe_mean(float(r["native_correct"]) for r in sample_rows)
    wrong = [r for r in sample_rows if not r["native_correct"]]
    ever = [r for r in wrong if r["wrong_taxonomy"] != "never_formed"]
    lost = [r for r in wrong if r["wrong_taxonomy"] in ("formed_then_lost", "lost_at_final_layer")]
    formed_present = [r for r in wrong if r["wrong_taxonomy"] == "formed_and_present_final"]
    best_y_row = max(layer_summary, key=lambda r: float(r["y_acc"])) if layer_summary else None
    final_row = layer_summary[-1] if layer_summary else None
    best_r_row = max(layer_summary, key=lambda r: float(r["rattn_acc"])) if layer_summary else None
    max_attn_gain = max(layer_summary, key=lambda r: float(r["attention_decision_gain_mean"])) if layer_summary else None
    min_attn_gain = min(layer_summary, key=lambda r: float(r["attention_decision_gain_mean"])) if layer_summary else None
    max_mlp_gain = max(layer_summary, key=lambda r: float(r["mlp_decision_gain_mean"])) if layer_summary else None
    min_mlp_gain = min(layer_summary, key=lambda r: float(r["mlp_decision_gain_mean"])) if layer_summary else None
    final_y_acc = float(final_row["y_acc"]) if final_row else float("nan")
    best_y_acc = float(best_y_row["y_acc"]) if best_y_row else float("nan")
    return {
        "model_requested": model_requested,
        "model_actual": model_actual,
        "dataset": dataset,
        "N": len(sample_rows),
        "n_layers": len(layer_summary),
        "native_restricted_firststep_acc": native_acc,
        "best_y_acc": best_y_acc,
        "best_y_layer": int(best_y_row["layer"]) if best_y_row else None,
        "best_rattn_acc": float(best_r_row["rattn_acc"]) if best_r_row else float("nan"),
        "best_rattn_layer": int(best_r_row["layer"]) if best_r_row else None,
        "final_y_acc": final_y_acc,
        "latent_output_gap_bestY_minus_finalY": best_y_acc - final_y_acc,
        "native_wrong_N": len(wrong),
        "wrong_ever_y_correct_N": len(ever),
        "wrong_ever_y_correct_rate": len(ever) / len(wrong) if wrong else float("nan"),
        "wrong_formed_then_lost_N": len(lost),
        "wrong_formed_then_lost_rate": len(lost) / len(wrong) if wrong else float("nan"),
        "wrong_formed_present_final_N": len(formed_present),
        "wrong_formed_present_final_rate": len(formed_present) / len(wrong) if wrong else float("nan"),
        "max_attention_gain_layer": int(max_attn_gain["layer"]) if max_attn_gain else None,
        "max_attention_gain": float(max_attn_gain["attention_decision_gain_mean"]) if max_attn_gain else float("nan"),
        "max_attention_drop_layer": int(min_attn_gain["layer"]) if min_attn_gain else None,
        "max_attention_drop": float(min_attn_gain["attention_decision_gain_mean"]) if min_attn_gain else float("nan"),
        "max_mlp_gain_layer": int(max_mlp_gain["layer"]) if max_mlp_gain else None,
        "max_mlp_gain": float(max_mlp_gain["mlp_decision_gain_mean"]) if max_mlp_gain else float("nan"),
        "max_mlp_drop_layer": int(min_mlp_gain["layer"]) if min_mlp_gain else None,
        "max_mlp_drop": float(min_mlp_gain["mlp_decision_gain_mean"]) if min_mlp_gain else float("nan"),
        "recon_rel_mean": safe_mean(r["recon_rel_mean"] for r in layer_summary),
        "recon_rel_max": safe_max(r["recon_rel_max"] for r in layer_summary),
        "final_lens_native_pred_match": final_match / final_match_n if final_match_n else float("nan"),
    }


# =============================================================================
# Single dataset scan with already-loaded model
# =============================================================================


def run_dataset_scan(
    *,
    args: argparse.Namespace,
    base: Any,
    probe: Any,
    model: Any,
    processor: Any,
    model_requested: str,
    model_actual: str,
    decoder_layers: Sequence[Any],
    decoder_path: str,
    final_norm: torch.nn.Module,
    final_norm_path: str,
    lm_head: torch.nn.Module,
    lm_head_path: str,
    dataset: str,
    out: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "DONE"
    if done.exists() and not args.force:
        return {"status": "skipped_new_done", "output_dir": str(out)}

    errors_path = out / "errors.jsonl"
    with contextlib.suppress(FileNotFoundError):
        errors_path.unlink()

    max_samples = args.max_samples if args.max_samples > 0 else None
    records, audit, loader_info = load_dataset_records(base, dataset, Path(args.data_root), args.controlled_json, max_samples)
    records = [r for r in records if r.relation in RELATIONS and r.image_path.exists()]
    records = select_records(records, args.num_samples, args.seed)
    if not records:
        raise RuntimeError(f"{dataset}: no usable records")

    layers = parse_layers(args.layers, len(decoder_layers))
    token_map = relation_token_variants(processor.tokenizer)
    lens = RelationLogitLens(final_norm, lm_head, token_map)

    config = {
        "script_version": SCRIPT_VERSION,
        "model_requested": model_requested,
        "model_actual": model_actual,
        "dataset": dataset,
        "data_root": str(args.data_root),
        "dataset_loader": loader_info,
        "dataset_audit_count": len(audit),
        "N": len(records),
        "relation_counts": dict(Counter(r.relation for r in records)),
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "layers": layers,
        "final_norm_path": final_norm_path,
        "lm_head_path": lm_head_path,
        "token_map": token_map,
        "prompt_template": args.prompt_template,
        "prefer_manifest_question": args.prefer_manifest_question,
        "run_generation": args.run_generation,
        "max_new_tokens": args.max_new_tokens,
        "transformers_version": transformers.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    final_match = 0
    final_match_n = 0

    print("\n" + "=" * 170)
    print(f"SPATIAL LOGIT-LENS | model={model_requested} -> {model_actual} | dataset={dataset}")
    print(f"N={len(records)} | layers={layers[0]}..{layers[-1]} ({len(layers)}) | output={out}")
    print("=" * 170, flush=True)

    for idx, rec in enumerate(tqdm(records, desc=f"{model_requested}:{dataset}"), start=1):
        image = batch = None
        try:
            question = render_question(rec, args.prompt_template, args.prefer_manifest_question)
            image = Image.open(rec.image_path).convert("RGB")
            batch = build_batch(probe, processor, question, image, device)

            with torch.inference_mode(), AllLayerResidualTrace(decoder_layers, layers) as tr:
                outputs = model(**batch, use_cache=False, return_dict=True)
            trace = tr.finalize()

            native_full_logits = outputs.logits[0, -1]
            native_scores = lens.scores_from_full_logits(native_full_logits)
            native_m = relation_metrics(native_scores, rec.relation)

            scores = {
                key: lens.scores(trace[key])
                for key in ("x", "attn_delta", "rattn", "mlp_delta", "y")
            }

            local_rows: List[Dict[str, Any]] = []
            for j, layer in enumerate(layers):
                mets = {key: relation_metrics(scores[key][j], rec.relation) for key in scores}
                row: Dict[str, Any] = {
                    "sid": rec.sid,
                    "gt": rec.relation,
                    "layer": layer,
                    "recon_rel": float(trace["recon_rel"][j]),
                }
                for key in ("x", "attn_delta", "rattn", "mlp_delta", "y"):
                    m = mets[key]
                    row.update({
                        f"{key}_pred": m["pred"],
                        f"{key}_correct": m["correct"],
                        f"{key}_decision_margin": m["decision_margin"],
                        f"{key}_opposite_margin": m["opposite_margin"],
                        f"{key}_p_gt_4way": m["p_gt_4way"],
                    })
                row["attention_decision_gain"] = row["rattn_decision_margin"] - row["x_decision_margin"]
                row["mlp_decision_gain"] = row["y_decision_margin"] - row["rattn_decision_margin"]
                row["attention_C_to_W"] = bool(row["x_correct"] and not row["rattn_correct"])
                row["attention_W_to_C"] = bool((not row["x_correct"]) and row["rattn_correct"])
                row["mlp_C_to_W"] = bool(row["rattn_correct"] and not row["y_correct"])
                row["mlp_W_to_C"] = bool((not row["rattn_correct"]) and row["y_correct"])
                local_rows.append(row)

            final_y_pred = str(local_rows[-1]["y_pred"])
            final_match_n += 1
            final_match += int(final_y_pred == native_m["pred"])

            generation_text = ""
            generation_pred = None
            generation_correct = None
            if args.run_generation:
                with torch.inference_mode():
                    gen = model.generate(
                        **batch,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                prompt_len = int(batch["input_ids"].shape[1]) if "input_ids" in batch else 0
                new_ids = gen[0, prompt_len:] if prompt_len and gen.shape[1] >= prompt_len else gen[0]
                generation_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                generation_pred = parse_generated_relation(generation_text)
                generation_correct = generation_pred == rec.relation if generation_pred is not None else False

            tax, first_correct_layer, last_correct_layer = wrong_taxonomy(local_rows, native_m["correct"])
            sample_row = {
                "sid": rec.sid,
                "gt": rec.relation,
                "image_path": str(rec.image_path),
                "native_pred": native_m["pred"],
                "native_correct": native_m["correct"],
                "native_decision_margin": native_m["decision_margin"],
                "native_p_gt_4way": native_m["p_gt_4way"],
                "wrong_taxonomy": tax,
                "first_y_correct_layer": first_correct_layer,
                "last_y_correct_layer": last_correct_layer,
                "final_y_pred": final_y_pred,
                "final_y_native_pred_match": final_y_pred == native_m["pred"],
                "generation_pred": generation_pred,
                "generation_correct": generation_correct,
                "generation_text": generation_text,
            }
            sample_rows.append(sample_row)
            layer_rows.extend(local_rows)

        except Exception as exc:
            err = {
                "sid": rec.sid,
                "dataset": dataset,
                "model": model_requested,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
            }
            append_jsonl(errors_path, err)
            tqdm.write(f"[ERROR] {model_requested}/{dataset} sid={rec.sid}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()
            del batch
            if idx % max(1, args.empty_cache_every) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not sample_rows:
        raise RuntimeError(f"{model_requested}/{dataset}: zero successful samples")

    layer_summary, group_summary = summarize_layers(layer_rows, sample_rows)
    job_summary = build_job_summary(model_requested, model_actual, dataset, sample_rows, layer_summary, final_match, final_match_n)

    write_csv(out / "sample_summary.csv", sample_rows)
    write_csv(out / "layer_trajectory.csv", layer_rows)
    write_csv(out / "layer_summary.csv", layer_summary)
    write_csv(out / "group_layer_summary.csv", group_summary)
    (out / "job_summary.json").write_text(json.dumps(job_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "=" * 150,
        "SPATIAL LOGIT-LENS JOB SUMMARY",
        "=" * 150,
        f"model requested / actual          = {model_requested} / {model_actual}",
        f"dataset                           = {dataset}",
        f"N successful                      = {len(sample_rows)}",
        f"native restricted first-step ACC  = {job_summary['native_restricted_firststep_acc']:.4f}",
        f"best y ACC                        = {job_summary['best_y_acc']:.4f} @ L{job_summary['best_y_layer']}",
        f"final y ACC                       = {job_summary['final_y_acc']:.4f}",
        f"best-y minus final-y gap          = {job_summary['latent_output_gap_bestY_minus_finalY']:+.4f}",
        f"native wrong ever y-correct       = {job_summary['wrong_ever_y_correct_N']}/{job_summary['native_wrong_N']} ({job_summary['wrong_ever_y_correct_rate']:.2%})" if job_summary['native_wrong_N'] else "native wrong ever y-correct       = n/a",
        f"native wrong formed-then-lost     = {job_summary['wrong_formed_then_lost_N']}/{job_summary['native_wrong_N']} ({job_summary['wrong_formed_then_lost_rate']:.2%})" if job_summary['native_wrong_N'] else "native wrong formed-then-lost     = n/a",
        f"max mean attention gain           = {job_summary['max_attention_gain']:+.4f} @ L{job_summary['max_attention_gain_layer']}",
        f"max mean attention drop           = {job_summary['max_attention_drop']:+.4f} @ L{job_summary['max_attention_drop_layer']}",
        f"max mean MLP gain                 = {job_summary['max_mlp_gain']:+.4f} @ L{job_summary['max_mlp_gain_layer']}",
        f"max mean MLP drop                 = {job_summary['max_mlp_drop']:+.4f} @ L{job_summary['max_mlp_drop_layer']}",
        f"residual reconstruction rel mean  = {job_summary['recon_rel_mean']:.3e}",
        f"residual reconstruction rel max   = {job_summary['recon_rel_max']:.3e}",
        f"final lens/native pred match      = {job_summary['final_lens_native_pred_match']:.2%}",
        "",
        "Interpretation note:",
        "  Main stage evidence = x -> r_attn -> y.  attn_delta/mlp_delta lenses are component-only diagnostics.",
        "  If residual reconstruction error is not small, inspect that model architecture before interpreting r_attn/MLP stages.",
    ]
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    done.write_text(f"{SCRIPT_VERSION}\n", encoding="utf-8")

    print("\n" + "\n".join(lines), flush=True)
    return {"status": "done", "output_dir": str(out), "summary": job_summary}


# =============================================================================
# Model worker
# =============================================================================


def clear_sampling_defaults(model: Any) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    for name, value in (("do_sample", False), ("temperature", None), ("top_p", None), ("top_k", None)):
        with contextlib.suppress(Exception):
            setattr(cfg, name, value)


def load_model_bundle(base: Any, requested: str, device: str, attn_impl: str):
    actual = resolve_model_alias(requested, base.SPECS)
    spec = base.SPECS[actual]
    cls = getattr(transformers, spec.model_class, None)
    if cls is None:
        raise RuntimeError(f"transformers {transformers.__version__} has no {spec.model_class}")
    kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": device},
    }
    if attn_impl != "none":
        kwargs["attn_implementation"] = attn_impl
    print(f"[load] requested={requested} actual={actual} repo={spec.repo_id}", flush=True)
    try:
        model = cls.from_pretrained(spec.repo_id, **kwargs)
    except TypeError:
        # Compatibility with older transformers expecting torch_dtype.
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    clear_sampling_defaults(model)
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    return actual, spec, model, processor


def parse_skip_jobs(items: Sequence[str], include_default: bool) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = dict(DEFAULT_LEGACY_SKIPS) if include_default else {}
    for item in items:
        parts = str(item).split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"--skip-job must be model:dataset:path, got {item!r}")
        out[(parts[0], parts[1])] = Path(parts[2])
    return out


def legacy_looks_done(path: Path) -> bool:
    if not path.exists():
        return False
    markers = ("report.txt", "config.json", "group_layer_summary.csv", "sample_summary.csv")
    return sum((path / x).exists() for x in markers) >= 2


def run_model_worker(args: argparse.Namespace) -> int:
    base = importlib.import_module(args.base_module)
    probe = importlib.import_module(args.probe_module)
    datasets = parse_csv_list(args.datasets)
    skip_jobs = parse_skip_jobs(args.skip_job, not args.no_default_legacy_skip)
    requested = args.model

    # Check whether anything for this model needs loading at all.
    pending = []
    for dataset in datasets:
        out = Path(args.output_root) / requested / dataset
        if (out / "DONE").exists() and not args.force:
            print(f"[skip new DONE] {requested}/{dataset}: {out}", flush=True)
            continue
        legacy = skip_jobs.get((requested, dataset))
        if legacy is not None and legacy_looks_done(legacy) and not args.force:
            print(f"[skip legacy] {requested}/{dataset}: {legacy}", flush=True)
            continue
        pending.append(dataset)
    if not pending:
        print(f"[skip model] {requested}: all requested datasets already completed", flush=True)
        return 0

    model = processor = None
    try:
        actual, spec, model, processor = load_model_bundle(base, requested, args.device, args.attn_impl)
        device = torch.device(args.device)
        decoder_layers, decoder_path = resolve_decoder_layers(model)
        final_norm, final_norm_path = resolve_final_norm(model, decoder_path)
        lm_head, lm_head_path = resolve_output_embeddings(model)
        print(
            f"[model ready] {requested}->{actual} layers={len(decoder_layers)} "
            f"decoder={decoder_path} final_norm={final_norm_path} lm_head={lm_head_path}",
            flush=True,
        )
        for dataset in pending:
            out = Path(args.output_root) / requested / dataset
            run_dataset_scan(
                args=args,
                base=base,
                probe=probe,
                model=model,
                processor=processor,
                model_requested=requested,
                model_actual=actual,
                decoder_layers=decoder_layers,
                decoder_path=decoder_path,
                final_norm=final_norm,
                final_norm_path=final_norm_path,
                lm_head=lm_head,
                lm_head_path=lm_head_path,
                dataset=dataset,
                out=out,
                device=device,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return 0
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Matrix launcher / collector
# =============================================================================


def script_self_path() -> Path:
    return Path(__file__).resolve()


def model_command(args: argparse.Namespace, model: str) -> List[str]:
    cmd = [
        sys.executable, "-u", str(script_self_path()),
        "--mode", "model",
        "--model", model,
        "--datasets", args.datasets,
        "--data-root", args.data_root,
        "--output-root", args.output_root,
        "--device", "cuda:0",
        "--attn-impl", args.attn_impl,
        "--layers", args.layers,
        "--num-samples", str(args.num_samples),
        "--max-samples", str(args.max_samples),
        "--seed", str(args.seed),
        "--empty-cache-every", str(args.empty_cache_every),
        "--max-new-tokens", str(args.max_new_tokens),
        "--prompt-template", args.prompt_template,
        "--base-module", args.base_module,
        "--probe-module", args.probe_module,
    ]
    if args.controlled_json:
        cmd += ["--controlled-json", args.controlled_json]
    if args.prefer_manifest_question:
        cmd.append("--prefer-manifest-question")
    if args.run_generation:
        cmd.append("--run-generation")
    if args.force:
        cmd.append("--force")
    if args.fail_fast:
        cmd.append("--fail-fast")
    else:
        cmd.append("--no-fail-fast")
    if args.no_default_legacy_skip:
        cmd.append("--no-default-legacy-skip")
    for item in args.skip_job:
        cmd += ["--skip-job", item]
    return cmd


def collect_matrix(output_root: Path, models: Sequence[str], datasets: Sequence[str], skip_jobs: Mapping[Tuple[str, str], Path]) -> Dict[str, Any]:
    matrix_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []

    for model in models:
        for dataset in datasets:
            job = output_root / model / dataset
            summary_path = job / "job_summary.json"
            layer_path = job / "layer_summary.csv"
            if summary_path.exists() and (job / "DONE").exists():
                row = json.loads(summary_path.read_text(encoding="utf-8"))
                matrix_rows.append(row)
                for lr in read_csv(layer_path):
                    lr2 = {"model_requested": model, "dataset": dataset, **lr}
                    layer_rows.append(lr2)
                statuses.append({"model": model, "dataset": dataset, "status": "done", "path": str(job)})
            else:
                legacy = skip_jobs.get((model, dataset))
                if legacy is not None and legacy_looks_done(legacy):
                    statuses.append({"model": model, "dataset": dataset, "status": "legacy_skipped", "path": str(legacy)})
                else:
                    statuses.append({"model": model, "dataset": dataset, "status": "missing_or_failed", "path": str(job)})

    write_csv(output_root / "matrix_summary.csv", matrix_rows)
    write_csv(output_root / "matrix_layer_summary.csv", layer_rows)
    (output_root / "matrix_status.json").write_text(json.dumps(statuses, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summaries": matrix_rows, "statuses": statuses}


def run_matrix(args: argparse.Namespace) -> int:
    models = parse_csv_list(args.models)
    datasets = parse_csv_list(args.datasets)
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise ValueError("--gpus is empty")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "_logs"
    logs.mkdir(parents=True, exist_ok=True)

    # Large models first; dynamic queue prevents one GPU ending much earlier.
    priority = {
        "llava-13b": 100,
        "qwen-7b": 90,
        "internvl-7b": 85,
        "llava-7b": 80,
        "qwen-3b": 60,
        "qwen2-2b": 50,
        "internvl-2b": 40,
        "internvl-1b": 30,
    }
    models = sorted(models, key=lambda m: -priority.get(m, 0))

    q: "queue.Queue[str]" = queue.Queue()
    for m in models:
        q.put(m)

    failures: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def gpu_loop(gpu: str):
        while True:
            try:
                model = q.get_nowait()
            except queue.Empty:
                return
            cmd = model_command(args, model)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = logs / f"{model}.log"
            print(f"[GPU {gpu}] START {model} | log={log_path}", flush=True)
            t0 = time.time()
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            print(f"[GPU {gpu}] END   {model} rc={proc.returncode} time={elapsed/60:.1f}m", flush=True)
            if proc.returncode != 0:
                with lock:
                    failures.append({"gpu": gpu, "model": model, "returncode": proc.returncode, "log": str(log_path)})
            q.task_done()

    threads = [threading.Thread(target=gpu_loop, args=(g,), daemon=False) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    skip_jobs = parse_skip_jobs(args.skip_job, not args.no_default_legacy_skip)
    result = collect_matrix(output_root, parse_csv_list(args.models), datasets, skip_jobs)

    print("\n" + "=" * 150)
    print("MATRIX COMPLETE")
    print("=" * 150)
    for s in result["statuses"]:
        print(f"{s['model']:14s} {s['dataset']:14s} {s['status']:18s} {s['path']}")
    if failures:
        print("\nFAILED MODEL WORKERS:")
        for f in failures:
            print(f"  GPU {f['gpu']} {f['model']} rc={f['returncode']} log={f['log']}")
    print(f"\nsummary: {output_root / 'matrix_summary.csv'}")
    print(f"layers : {output_root / 'matrix_layer_summary.csv'}")
    print(f"status : {output_root / 'matrix_status.json'}")
    return 1 if failures else 0


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("matrix", "model", "collect"), default="matrix")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--model", default="qwen-3b", help="Used only in --mode model")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--gpus", default="0,1", help="Physical GPU IDs used by matrix launcher")
    p.add_argument("--device", default="cuda:0", help="Worker-visible device; matrix launcher remaps each physical GPU to cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--controlled-json", default="")
    p.add_argument("--output-root", default="output/spatial_logitlens_matrix_v1")
    p.add_argument("--layers", default="all")
    p.add_argument("--num-samples", type=int, default=0, help="Stratified cap after loading; 0=all")
    p.add_argument("--max-samples", type=int, default=0, help="Loader/debug cap; 0=all")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--attn-impl", default="eager", choices=("eager", "sdpa", "flash_attention_2", "none"))
    p.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    p.add_argument("--prefer-manifest-question", action="store_true")
    p.add_argument("--run-generation", action="store_true", help="Optional slower full-generation check")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--empty-cache-every", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--base-module", default="extract_two_object_relation_states")
    p.add_argument("--probe-module", default="analyze_coco_head_object_residual_direction_probe_v1")
    p.add_argument("--no-default-legacy-skip", action="store_true")
    p.add_argument(
        "--skip-job", action="append", default=[],
        help="Additional previously completed run: model:dataset:path (repeatable)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode in ("model",) and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if args.mode == "matrix":
        return run_matrix(args)
    if args.mode == "model":
        return run_model_worker(args)
    if args.mode == "collect":
        skip_jobs = parse_skip_jobs(args.skip_job, not args.no_default_legacy_skip)
        collect_matrix(Path(args.output_root), parse_csv_list(args.models), parse_csv_list(args.datasets), skip_jobs)
        return 0
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
