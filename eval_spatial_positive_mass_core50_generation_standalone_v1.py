#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spatial_positive_mass_core50_generation_standalone_v1.py

Standalone Qwen2.5-VL-3B experiment.

NO imports from any AdaptVis project Python file.

Question
========
Can the ORIGINAL Direction-Head spatial source score itself produce a
sample-specific causal-token candidate set strong enough to improve generation?

The script keeps the previously used aligned spatial score:

    c_RG(h,p)
      = [A_real(sub,p)-A_real(ref,p)] V_real_h(p)
        -
        [A_gray(sub,p)-A_gray(ref,p)] V_gray_h(p)

    S(h,p;r) = < c_RG(h,p), d_h,r >

where d_h,r is a relation direction in the SAME PRE-W_O per-head space.

Synthetic-400 direction bank
============================
For every synthetic sample and head:

    q_RG(h)
      = [z_real(sub)-z_real(ref)]
        -
        [z_gray(sub)-z_gray(ref)]

Using the exact COCO prompt wording:

    "Where is the SUBJECT in relation to the REFERENCE?
     Answer with left, right, above, or below."

Fit:

    mu_h = E[q_RG(h)]
    d_h,r = normalize(E[q_RG(h) | relation=r] - mu_h)

COCO head selection
===================
Synthetic gives the four directions, but HEAD SELECTION is from COCO.

On a COCO calibration set (default: all 440 for the current diagnostic run),
for every head:

    pred_h(x) = argmax_r cos(q_RG(h,x)-mu_h, d_h,r)

Compute COCO four-way accuracy and select Top-M heads PER attention layer.
Thus every causal source layer remains represented:

    causal state L  <->  attention head H=L+1.

Spatial positive-mass Core50
============================
For a real COCO sample, for each relation hypothesis r independently:

1) Compute original Real-Gray token spatial score for selected heads.
2) Accuracy-weight selected heads within each aligned layer:

       S_L,p(r) = weighted_mean_h S(h,p;r)

   This remains a RAW original spatial projection, not a new self-alignment score.

3) Candidate domain defaults to TEXT tokens only, prompt-last excluded.

4) Optional global_unique (default ON):
   if token position p is strong at multiple state layers, retain the exact
   state (L,p) with largest positive S_L,p(r), matching the old causal
   global_unique convention.

5) Keep positive states, sort by score, and take the shortest prefix reaching:

       cumulative positive spatial mass >= 50% total positive spatial mass.

Call this:

       SpatialCore50_r(x)

Do this for ALL FOUR relation hypotheses.

Relation-free selection requested by the user
=============================================
The primary non-oracle selector is:

    r_mass(x) = argmax_r TotalPositiveSpatialMass_r(x)

and the edited states are:

    SpatialCore50_{r_mass}(x).

The script also computes diagnostics:
- relation vote from high-accuracy heads;
- GT-relation SpatialCore50 (oracle-relation diagnostic);
- best-of-4 oracle-overlap ceiling:
    among the four independently generated SpatialCore50_r sets, which one
    overlaps the oracle causal Core50 most?
  This is reporting only and is NEVER used for the main edited generation.

Oracle causal Core50 overlap
============================
Load:
    <oracle-bank-dir>/core50_candidates_all440.csv

For mass-selected SpatialCore50 report:
- number / fraction of samples with >=1 EXACT (layer,token) Core50 hit;
- exact micro precision / recall;
- position-only overlap as secondary diagnostic;
- selected-state count distribution.

Also report the same metrics for:
- GT relation candidate;
- best-of-4 oracle-overlap ceiling.

Generation intervention
=======================
After the selection pass, capture decoder block outputs from real and gray:

    Delta h_L,p = h_real_L,p - h_gray_L,p

For selected exact states:

    h'_L,p = h_real_L,p + alpha * Delta h_L,p

Default alpha=1:

    h' = 2*h_real - h_gray

Then run actual greedy model.generate().

Main generation condition:
    spatial_mass4_core50

Optional diagnostic generation:
    spatial_gt_core50
which uses the GT relation ONLY to choose which already-computed spatial
Core50 relation candidate to edit.

Outputs
=======
synthetic_realgray_direction_cache.npz
coco_head_calibration_cache.npz
coco_head_accuracy.csv
selected_heads_per_layer.csv
candidate_relation_summary.csv
selected_spatial_core50_states.csv
core50_overlap_per_sample.csv
core50_overlap_summary.csv
generation_per_sample.csv
generation_summary.csv
analysis_summary.txt
metadata.json
errors.jsonl

Recommended current diagnostic
==============================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_spatial_positive_mass_core50_generation_standalone_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --oracle-bank-dir output/qwen3b_oracle_core50_top10_bank_all440 \
  --head-layers 21,22,23,24,25,26,27 \
  --heads-per-layer 1 \
  --head-calib-frac 1.0 \
  --mass-threshold 0.50 \
  --alpha 1.0 \
  --generation-modes mass4,gt \
  --require-eval-n 440 \
  --output-dir output/qwen3b_spatial_positive_mass_core50_generation_v1 \
  --overwrite

For a cleaner held-out version later:
    --head-calib-frac 0.30 --eval-heldout-only
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import re
import shutil
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "standalone-spatial-positive-mass-core50-generation-v1"

REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12

PROBE_PROMPT = (
    "Where is the {subject} in relation to the {reference}? "
    "Answer with left, right, above, or below."
)

STANDARD_OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?\s*Answer\s+with",
    flags=re.IGNORECASE | re.DOTALL,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument(
        "--model-class",
        default="Qwen2_5_VLForConditionalGeneration",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default="")
    p.add_argument("--synthetic-max-samples", type=int, default=0)

    p.add_argument(
        "--oracle-bank-dir",
        required=True,
        help="Directory containing core50_candidates_all440.csv.",
    )

    p.add_argument(
        "--head-layers",
        default="21,22,23,24,25,26,27",
        help="Attention head layers H; edited decoder output layer is H-1.",
    )
    p.add_argument(
        "--heads-per-layer",
        type=int,
        default=1,
        help="Top COCO-accuracy heads retained independently in EACH head layer.",
    )
    p.add_argument(
        "--vote-top-n",
        type=int,
        default=10,
        help="Global top COCO-accuracy heads used only for relation-vote diagnostics.",
    )

    p.add_argument(
        "--head-calib-frac",
        type=float,
        default=1.0,
        help=(
            "Fraction of COCO used to compute head accuracy. 1.0 uses all 440 "
            "for the current diagnostic macro-head analysis."
        ),
    )
    p.add_argument("--split-seed", type=int, default=17)
    p.add_argument(
        "--eval-heldout-only",
        action="store_true",
        help="Evaluate only SIDs excluded from head calibration.",
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--direction-pool", choices=["mean", "last"], default="mean")
    p.add_argument(
        "--candidate-domain",
        choices=["text", "all"],
        default="text",
    )
    p.add_argument(
        "--global-unique",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep at most one state layer for each token position within a relation candidate.",
    )
    p.add_argument(
        "--mass-threshold",
        type=float,
        default=0.50,
        help="Shortest positive-score prefix reaching this total positive spatial mass.",
    )
    p.add_argument(
        "--head-weight",
        choices=["equal", "accuracy"],
        default="accuracy",
    )
    p.add_argument(
        "--relation-mass-normalization",
        choices=["none", "per_head_delta_norm"],
        default="none",
        help=(
            "Primary requested behavior is none: choose relation by raw total positive "
            "spatial mass. Optional per_head_delta_norm is a robustness control."
        ),
    )

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--generation-modes",
        default="mass4,gt",
        help="Comma separated subset of mass4,gt,vote. baseline is always run.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--require-eval-n", type=int, default=0)

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite-synthetic-cache", action="store_true")
    p.add_argument("--overwrite-head-cache", action="store_true")
    p.add_argument(
        "--resume-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================

def parse_ints(text: str) -> List[int]:
    return sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})


def parse_strs(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def normalize_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def safe_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def hname(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head):02d}"


def canon_rel(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    exact = {
        "left": "left",
        "left of": "left",
        "to the left": "left",
        "to the left of": "left",
        "right": "right",
        "right of": "right",
        "to the right": "right",
        "to the right of": "right",
        "above": "above",
        "over": "above",
        "on": "above",
        "on top of": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "bottom": "below",
    }
    if text in exact:
        return exact[text]

    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(below|under|underneath|beneath|bottom)\b", "below"),
        (r"\b(above|over|on top|top)\b", "above"),
        (r"\bon\b", "above"),
    ]
    hits = []
    for pattern, label in patterns:
        m = re.search(pattern, text)
        if m:
            hits.append((m.start(), label))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def make_gray(real: Image.Image, value: int) -> Image.Image:
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real.size, (v, v, v))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


# =============================================================================
# COCO records / prompt parsing
# =============================================================================

@dataclass(frozen=True)
class CocoRecord:
    sid: int
    image_id: str
    image_path: Path


def load_coco_records(data_root: Path) -> Dict[int, CocoRecord]:
    ann = data_root / "coco_qa_two_obj.json"
    image_dir = data_root / "val2017"

    if not ann.exists():
        raise FileNotFoundError(ann)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    raw = json.loads(ann.read_text(encoding="utf-8"))
    out = {}
    for sid, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) < 1:
            continue
        image_id = row[0]
        image_path = image_dir / f"{int(image_id):012d}.jpg"
        if not image_path.exists():
            continue
        out[sid] = CocoRecord(
            sid=sid,
            image_id=str(image_id),
            image_path=image_path,
        )
    return out


def extract_standard_user_text(raw_question: str) -> str:
    text = str(raw_question).strip()
    text = re.sub(r"^\s*<image>\s*", "", text, flags=re.IGNORECASE)
    m = re.search(
        r"\bUSER\s*:\s*(.*?)(?:\s*\bASSISTANT\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = m.group(1)
    return text.strip()


def parse_standard_objects(question_text: str) -> Tuple[str, str]:
    compact = re.sub(r"\s+", " ", str(question_text)).strip()
    m = STANDARD_OBJECT_RE.search(compact)
    if not m:
        raise ValueError(
            "Could not parse subject/reference from standard question: "
            f"{compact!r}"
        )
    return m.group(1).strip(), m.group(2).strip()


def standard_answer_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def load_standard_prompts(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = int(row["id"])
            q = extract_standard_user_text(str(row["question"]))
            subject, reference = parse_standard_objects(q)
            gt = canon_rel(standard_answer_value(row["answer"]))
            if gt not in REL:
                raise RuntimeError(
                    f"{path}:{line_no} invalid relation answer={row['answer']!r}"
                )
            rows[sid] = {
                "sid": sid,
                "question_text": q,
                "subject": subject,
                "reference": reference,
                "gt": gt,
            }
    return rows


# =============================================================================
# Synthetic rows
# =============================================================================

def load_synthetic_rows(
    synthetic_dir: Path,
    synthetic_labels: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    labels_path = (
        Path(synthetic_labels)
        if synthetic_labels
        else synthetic_dir / "labels.jsonl"
    )
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            relation = canon_rel(item.get("relation"))
            if relation not in REL:
                raise RuntimeError(
                    f"{labels_path}:{line_no} bad relation={item.get('relation')!r}"
                )

            image_value = Path(str(item["image"]))
            image_path = (
                image_value
                if image_value.is_absolute()
                else synthetic_dir / image_value
            )
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()

            rows.append({
                "sid": int(item.get("id", len(rows))),
                "image_path": image_path,
                "subject": subject,
                "reference": reference,
                "relation": relation,
                "question_text": PROBE_PROMPT.format(
                    subject=subject,
                    reference=reference,
                ),
            })

    rows.sort(key=lambda r: r["sid"])
    if max_samples > 0:
        rows = rows[: int(max_samples)]

    counts = Counter(r["relation"] for r in rows)
    for r in REL:
        if counts[r] == 0:
            raise RuntimeError(
                f"Synthetic source has no {r}; counts={dict(counts)}"
            )
    return rows


# =============================================================================
# Oracle causal Core50
# =============================================================================

def boolify(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes", "t"}


def load_oracle_core50(root: Path) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    p = root / "core50_candidates_all440.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    required = {"sid", "source_layer", "position", "category", "broad_category"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{p} missing columns={sorted(missing)}")

    for c in ("sid", "source_layer", "position"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)

    if "is_text" in df.columns:
        df["is_text"] = df["is_text"].map(boolify)
    else:
        df["is_text"] = (
            ~df["category"].astype(str).eq("visual")
            & ~df["broad_category"].astype(str).eq("visual")
        )

    lookup = {}
    for sid, g in df.groupby("sid"):
        states_all = {
            (int(r.source_layer), int(r.position))
            for r in g.itertuples()
        }
        text_g = g[g["is_text"]]
        states_text = {
            (int(r.source_layer), int(r.position))
            for r in text_g.itertuples()
        }
        lookup[int(sid)] = {
            "all_states": states_all,
            "text_states": states_text,
            "all_positions": {p for _, p in states_all},
            "text_positions": {p for _, p in states_text},
        }

    return df, lookup


# =============================================================================
# Processor / token helpers
# =============================================================================

def configure_processor(model: Any, processor: Any) -> None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)

    if (
        vision_config is not None
        and hasattr(processor, "patch_size")
        and hasattr(vision_config, "patch_size")
    ):
        processor.patch_size = int(vision_config.patch_size)

    strategy = getattr(config, "vision_feature_select_strategy", None)
    if (
        strategy is not None
        and hasattr(processor, "vision_feature_select_strategy")
    ):
        processor.vision_feature_select_strategy = str(strategy)


def build_prompt(processor: Any, question_text: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question_text},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question_text


def make_batch(
    processor: Any,
    image: Image.Image,
    question_text: str,
    device: torch.device,
) -> Dict[str, Any]:
    rendered = build_prompt(processor, question_text)
    batch = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
    )
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out.input_ids
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    w = len(needle)
    return [
        i
        for i in range(len(haystack) - w + 1)
        if list(haystack[i:i+w]) == list(needle)
    ]


def phrase_surface_variants(phrase: str) -> List[str]:
    raw = str(phrase).strip()
    xs = [
        raw,
        raw.lower(),
        raw.upper(),
        raw.title(),
        raw.capitalize(),
    ]
    return list(dict.fromkeys(x for x in xs if x))


def find_phrase_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    phrase: str,
    include_article_variants: bool = True,
) -> List[Tuple[int, int]]:
    variants = []
    for surface in phrase_surface_variants(phrase):
        variants.extend([surface, " " + surface])
        if include_article_variants:
            variants.extend(["the " + surface, " the " + surface])

    spans = []
    seen = set()
    for variant in variants:
        ids = tokenizer_ids(tokenizer, variant)
        key = tuple(ids)
        if not ids or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, ids):
            spans.append((start, start + len(ids) - 1))
    return sorted(set(spans))


def locate_object_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    subject: str,
    reference: str,
) -> Tuple[List[int], List[int]]:
    ss = find_phrase_spans(
        tokenizer,
        input_ids,
        subject,
        include_article_variants=True,
    )
    rr = find_phrase_spans(
        tokenizer,
        input_ids,
        reference,
        include_article_variants=True,
    )

    valid = [
        (s, r)
        for s in ss
        for r in rr
        if s[1] < r[0]
    ]
    if not valid:
        raise ValueError(
            f"Could not locate ordered object spans subject={subject!r}, "
            f"reference={reference!r}; subject_spans={ss}, reference_spans={rr}"
        )

    s, r = max(valid, key=lambda pair: (pair[1][0], pair[0][0]))
    return (
        list(range(int(s[0]), int(s[1]) + 1)),
        list(range(int(r[0]), int(r[1]) + 1)),
    )


def candidate_token_id(tokenizer: Any, token: str) -> Optional[int]:
    try:
        idx = tokenizer.convert_tokens_to_ids(token)
        idx = int(idx)
    except Exception:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and idx == int(unk):
        return None
    return idx


def resolve_visual_indices(
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    input_ids: Sequence[int],
) -> List[int]:
    mm = batch.get("mm_token_type_ids")
    if torch.is_tensor(mm) and mm.ndim == 2:
        idx = torch.nonzero(mm[0] == 1, as_tuple=False).flatten().tolist()
        if idx:
            return [int(x) for x in idx]

    tt = batch.get("token_type_ids")
    if torch.is_tensor(tt) and tt.ndim == 2:
        vals = set(int(x) for x in tt[0].detach().cpu().tolist())
        if 1 in vals:
            idx = torch.nonzero(tt[0] == 1, as_tuple=False).flatten().tolist()
            if idx:
                return [int(x) for x in idx]

    token_ids = set()
    objects = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        processor,
        getattr(processor, "tokenizer", None),
    ]
    for obj in objects:
        if obj is None:
            continue
        for name in ("image_token_id", "image_token_index"):
            value = getattr(obj, name, None)
            if isinstance(value, (int, np.integer)) and int(value) >= 0:
                token_ids.add(int(value))

    tokenizer = processor.tokenizer
    for token in (
        "<|image_pad|>",
        "<image>",
        "<image_token>",
        "<IMG_CONTEXT>",
    ):
        idx = candidate_token_id(tokenizer, token)
        if idx is not None:
            token_ids.add(idx)

    idx = [
        i for i, x in enumerate(input_ids)
        if int(x) in token_ids
    ]
    if idx:
        return idx

    start_ids = {
        i
        for t in ("<|vision_start|>", "<image_start>", "<img>")
        if (i := candidate_token_id(tokenizer, t)) is not None
    }
    end_ids = {
        i
        for t in ("<|vision_end|>", "<image_end>", "</img>")
        if (i := candidate_token_id(tokenizer, t)) is not None
    }
    starts = [i for i, x in enumerate(input_ids) if int(x) in start_ids]
    ends = [i for i, x in enumerate(input_ids) if int(x) in end_ids]
    spans = [(s, e) for s in starts for e in ends if s < e]
    if spans:
        s, e = min(spans, key=lambda z: z[1] - z[0])
        return list(range(s + 1, e))

    raise RuntimeError("Could not identify visual-token positions.")


def token_strings(processor: Any, ids: Sequence[int]) -> List[str]:
    tok = processor.tokenizer
    out = []
    for i in ids:
        try:
            s = tok.convert_ids_to_tokens(int(i))
        except Exception:
            s = str(i)
        out.append(str(s).replace("\n", "\\n"))
    return out


# =============================================================================
# Model internals / captures
# =============================================================================

def get_attr_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_decoder_layers(model: Any) -> Tuple[Any, str]:
    preferred = [
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in preferred:
        value = get_attr_path(model, path)
        if isinstance(value, (torch.nn.ModuleList, list, tuple)) and len(value) >= 4:
            return value, path

    candidates = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) >= 4:
            candidates.append((f"{name}.layers" if name else "layers", layers))

    candidates.sort(
        key=lambda item: (
            0 if any(k in item[0].lower() for k in ("language", "text")) else 1,
            1 if any(k in item[0].lower() for k in ("visual", "vision")) else 0,
            -len(item[1]),
        )
    )
    if candidates:
        return candidates[0][1], candidates[0][0]
    raise RuntimeError("Could not resolve decoder layers.")


def get_text_config(model: Any) -> Any:
    cfg = getattr(model, "config", None)
    for c in (
        getattr(cfg, "text_config", None),
        getattr(cfg, "language_config", None),
        cfg,
    ):
        if c is not None and getattr(c, "num_attention_heads", None) is not None:
            return c
    raise RuntimeError("Could not resolve text config.")


def resolve_attn(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        x = getattr(layer, name, None)
        if x is not None:
            return x
    raise RuntimeError("Could not resolve attention module.")


def resolve_o_proj(attn: Any) -> torch.nn.Module:
    for name in ("o_proj", "out_proj", "proj"):
        x = getattr(attn, name, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve o_proj.")


def resolve_v_proj(attn: Any) -> torch.nn.Module:
    for name in ("v_proj", "value", "value_proj"):
        x = getattr(attn, name, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve v_proj.")


def extract_attentions(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "attentions",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "attentions",
            None,
        ),
    ]
    for x in candidates:
        if isinstance(x, (list, tuple)) and len(x):
            return tuple(x)
    raise RuntimeError(
        "No attention tensors returned. Run with --attn-impl eager."
    )


def norm_attn(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"Unexpected attention shape={tuple(x.shape)}")
    return x.detach().float().cpu().numpy().astype(np.float32)


def all_head_pre_o(
    pre_o: np.ndarray,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    x = pre_o[0]  # [seq, hidden]
    return x.reshape(x.shape[0], n_heads, head_dim)


def all_head_v(
    vout: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    x = vout[0]  # [seq, kv_width]
    if x.shape[-1] % head_dim != 0:
        raise RuntimeError(
            f"v width={x.shape[-1]} not divisible by head_dim={head_dim}"
        )
    nkv = x.shape[-1] // head_dim
    x = x.reshape(x.shape[0], nkv, head_dim)

    if nkv == n_heads:
        return x
    if n_heads % nkv != 0:
        raise RuntimeError(
            f"Cannot expand KV heads: query_heads={n_heads}, kv_heads={nkv}"
        )
    repeat = n_heads // nkv
    return np.repeat(x, repeat, axis=1)


class CaptureHooks:
    """
    Capture, in one forward:
      - pre-o_proj head outputs at selected attention layers
      - v_proj outputs at selected attention layers
      - decoder block outputs at selected source layers

    block_outputs are stored on CPU as float16 to limit memory.
    """

    def __init__(
        self,
        model: Any,
        decoder_layers: Sequence[Any],
        head_layers: Sequence[int],
        source_layers: Sequence[int],
        capture_v: bool,
        capture_blocks: bool,
    ):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(
            getattr(cfg, "num_key_value_heads", self.n_heads)
        )
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        if hidden <= 0:
            op = resolve_o_proj(resolve_attn(decoder_layers[0]))
            hidden = int(op.in_features)
        self.hidden_size = hidden
        self.head_dim = hidden // self.n_heads

        self.pre_o: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.block: Dict[int, torch.Tensor] = {}
        self.handles = []

        for H in head_layers:
            attn = resolve_attn(decoder_layers[H])
            op = resolve_o_proj(attn)

            def make_pre_hook(layer_idx: int):
                def hook(_m, inputs):
                    self.pre_o[layer_idx] = (
                        inputs[0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_pre_hook(H))
            )

            if capture_v:
                vp = resolve_v_proj(attn)

                def make_v_hook(layer_idx: int):
                    def hook(_m, _inputs, output):
                        self.v[layer_idx] = (
                            output
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    return hook

                self.handles.append(
                    vp.register_forward_hook(make_v_hook(H))
                )

        if capture_blocks:
            for L in source_layers:
                layer = decoder_layers[L]

                def make_block_hook(layer_idx: int):
                    def hook(_m, _inputs, output):
                        h = output[0] if isinstance(output, tuple) else output
                        self.block[layer_idx] = (
                            h.detach().cpu().to(torch.float16)
                        )
                    return hook

                self.handles.append(
                    layer.register_forward_hook(make_block_hook(L))
                )

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles = []


@torch.inference_mode()
def run_capture(
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    head_layers: Sequence[int],
    source_layers: Sequence[int],
    need_attn: bool,
    need_v: bool,
    need_blocks: bool,
) -> Dict[str, Any]:
    cap = CaptureHooks(
        model=model,
        decoder_layers=decoder_layers,
        head_layers=head_layers,
        source_layers=source_layers,
        capture_v=need_v,
        capture_blocks=need_blocks,
    )
    try:
        kwargs = dict(batch)
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        kwargs["output_attentions"] = bool(need_attn)
        outputs = model(**kwargs)

        attn = {}
        if need_attn:
            aa = extract_attentions(outputs)
            for H in head_layers:
                attn[H] = norm_attn(aa[H])

        return {
            "pre_o": cap.pre_o,
            "v": cap.v,
            "block": cap.block,
            "attn": attn,
            "n_heads": cap.n_heads,
            "n_kv_heads": cap.n_kv_heads,
            "head_dim": cap.head_dim,
            "hidden_size": cap.hidden_size,
        }
    finally:
        cap.close()


def pool_head_rows(
    arr: np.ndarray,
    positions: Sequence[int],
    mode: str,
) -> np.ndarray:
    """
    arr [seq, H, D]
    returns [H,D]
    """
    valid = [int(p) for p in positions if 0 <= int(p) < arr.shape[0]]
    if not valid:
        raise RuntimeError("No valid object positions.")
    if mode == "last":
        return arr[valid[-1]]
    return arr[valid].mean(axis=0)


def head_realgray_residual(
    real_cap: Mapping[str, Any],
    gray_cap: Mapping[str, Any],
    H: int,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    pool: str,
) -> np.ndarray:
    n_heads = int(real_cap["n_heads"])
    head_dim = int(real_cap["head_dim"])

    Hr = all_head_pre_o(
        real_cap["pre_o"][H],
        n_heads,
        head_dim,
    )
    Hg = all_head_pre_o(
        gray_cap["pre_o"][H],
        n_heads,
        head_dim,
    )

    rs = pool_head_rows(Hr, subject_positions, pool)
    rr = pool_head_rows(Hr, reference_positions, pool)
    gs = pool_head_rows(Hg, subject_positions, pool)
    gr = pool_head_rows(Hg, reference_positions, pool)

    return (rs - rr - gs + gr).astype(np.float32)


def direction_token_scores_all_relations(
    real_cap: Mapping[str, Any],
    gray_cap: Mapping[str, Any],
    H: int,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    pool: str,
    dirs_h4d: np.ndarray,
    candidate_positions: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    EXACT original Real-Gray A*V spatial source score.

    Returns:
      projection [H, P, 4]
      head_delta [H,D]
      recon_cos [H]
    """
    Ar = real_cap["attn"][H]
    Ag = gray_cap["attn"][H]

    n_heads = int(real_cap["n_heads"])
    head_dim = int(real_cap["head_dim"])

    Vr = all_head_v(
        real_cap["v"][H],
        n_heads,
        int(real_cap["n_kv_heads"]),
        head_dim,
    )
    Vg = all_head_v(
        gray_cap["v"][H],
        n_heads,
        int(gray_cap["n_kv_heads"]),
        head_dim,
    )

    if pool == "last":
        qs = int(subject_positions[-1])
        qr = int(reference_positions[-1])
        cr = Ar[:, qs, :] - Ar[:, qr, :]
        cg = Ag[:, qs, :] - Ag[:, qr, :]
    else:
        ss = [int(x) for x in subject_positions]
        rr = [int(x) for x in reference_positions]
        cr = Ar[:, ss, :].mean(axis=1) - Ar[:, rr, :].mean(axis=1)
        cg = Ag[:, ss, :].mean(axis=1) - Ag[:, rr, :].mean(axis=1)

    n = min(cr.shape[1], cg.shape[1], Vr.shape[0], Vg.shape[0])
    cr = cr[:, :n]
    cg = cg[:, :n]
    Vr = Vr[:n]
    Vg = Vg[:n]

    # [H,K,D]
    vec = (
        cr[:, :, None] * np.transpose(Vr, (1, 0, 2))
        -
        cg[:, :, None] * np.transpose(Vg, (1, 0, 2))
    ).astype(np.float32)

    valid = [int(p) for p in candidate_positions if 0 <= int(p) < n]
    if valid != list(map(int, candidate_positions)):
        raise RuntimeError(
            "Candidate position outside captured attention/value sequence."
        )

    C = vec[:, np.asarray(valid, dtype=np.int64), :]  # [H,P,D]
    D = np.asarray(dirs_h4d, dtype=np.float32)        # [H,4,D]

    projection = np.einsum(
        "hpd,hrd->hpr",
        C,
        D,
        optimize=True,
    ).astype(np.float32)

    head_delta = head_realgray_residual(
        real_cap,
        gray_cap,
        H,
        subject_positions,
        reference_positions,
        pool,
    )

    recon = vec.sum(axis=1)
    recon_cos = np.full(n_heads, np.nan, dtype=np.float32)
    for h in range(n_heads):
        a = head_delta[h]
        b = recon[h]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na > EPS and nb > EPS:
            recon_cos[h] = float(np.dot(a, b) / (na * nb))

    return projection, head_delta, recon_cos


# =============================================================================
# Synthetic direction cache
# =============================================================================

def fit_direction_bank(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X [N,nL,H,D]
    center [nL,H,D]
    dirs [nL,H,4,D]
    """
    center = X.mean(axis=0).astype(np.float32)
    dirs = np.zeros(
        (X.shape[1], X.shape[2], 4, X.shape[3]),
        dtype=np.float32,
    )
    for ri, relation in enumerate(REL):
        mask = y == relation
        if not np.any(mask):
            raise RuntimeError(f"No source samples for relation={relation}")
        dirs[:, :, ri, :] = normalize_np(
            X[mask].mean(axis=0) - center,
            axis=-1,
        )
    return center, dirs


def score_direction_bank(
    X: np.ndarray,
    center: np.ndarray,
    dirs: np.ndarray,
) -> np.ndarray:
    q = normalize_np(X - center[None, ...], axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", q, dirs, optimize=True)


def save_direction_cache(
    path: Path,
    sids: Sequence[int],
    labels: Sequence[str],
    X: np.ndarray,
    head_layers: Sequence[int],
    center: np.ndarray,
    dirs: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        sample_index=np.asarray(sids, dtype=np.int64),
        relation=np.asarray(labels, dtype=object),
        residual=np.asarray(X, dtype=np.float16),
        head_layers=np.asarray(head_layers, dtype=np.int32),
        center=np.asarray(center, dtype=np.float32),
        directions=np.asarray(dirs, dtype=np.float32),
    )


def load_direction_cache(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return (
        np.asarray(z["sample_index"]).astype(int),
        np.asarray(z["relation"], dtype=object),
        np.asarray(z["residual"], dtype=np.float32),
        np.asarray(z["head_layers"]).astype(int).tolist(),
        np.asarray(z["center"], dtype=np.float32),
        np.asarray(z["directions"], dtype=np.float32),
    )


# =============================================================================
# Split / head calibration
# =============================================================================

def stratified_calibration_sids(
    meta: Sequence[Dict[str, Any]],
    frac: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not (0.0 < frac <= 1.0):
        raise ValueError("--head-calib-frac must be in (0,1].")

    all_sids = [int(m["sid"]) for m in meta]
    if frac >= 1.0:
        return sorted(all_sids), []

    rng = np.random.default_rng(seed)
    calib = []
    heldout = []

    for relation in REL:
        ids = np.asarray(
            [int(m["sid"]) for m in meta if m["gt"] == relation],
            dtype=int,
        )
        rng.shuffle(ids)
        n = max(1, int(round(len(ids) * frac)))
        calib.extend(ids[:n].tolist())
        heldout.extend(ids[n:].tolist())

    return sorted(calib), sorted(heldout)


def compute_head_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    center: np.ndarray,
    dirs: np.ndarray,
    head_layers: Sequence[int],
) -> pd.DataFrame:
    scores = score_direction_bank(X, center, dirs)
    yi = np.asarray([RID[str(r)] for r in y], dtype=int)
    pred = np.argmax(scores, axis=-1)

    rows = []
    for li, H in enumerate(head_layers):
        for h in range(scores.shape[2]):
            ok = pred[:, li, h] == yi
            row = {
                "head_layer": int(H),
                "head": int(h),
                "head_name": hname(H, h),
                "accuracy": float(ok.mean()),
                "N": int(len(ok)),
            }
            for ri, relation in enumerate(REL):
                mask = yi == ri
                row[f"acc_{relation}"] = (
                    float(ok[mask].mean()) if np.any(mask) else np.nan
                )
            rows.append(row)

    return pd.DataFrame(rows)


def select_heads_per_layer(
    accuracy_df: pd.DataFrame,
    head_layers: Sequence[int],
    n: int,
) -> Dict[int, List[Tuple[int, float]]]:
    out = {}
    for H in head_layers:
        q = (
            accuracy_df[accuracy_df["head_layer"] == H]
            .sort_values(["accuracy", "head"], ascending=[False, True])
            .head(int(n))
        )
        out[H] = [
            (int(r.head), float(r.accuracy))
            for r in q.itertuples()
        ]
    return out


def global_vote_heads(
    accuracy_df: pd.DataFrame,
    n: int,
) -> List[Tuple[int, int, float]]:
    q = accuracy_df.sort_values(
        ["accuracy", "head_layer", "head"],
        ascending=[False, True, True],
    ).head(int(n))
    return [
        (int(r.head_layer), int(r.head), float(r.accuracy))
        for r in q.itertuples()
    ]


# =============================================================================
# Spatial positive-mass Core50
# =============================================================================

def aggregate_raw_state_scores(
    projection_by_layer: Mapping[int, np.ndarray],
    selected_heads: Mapping[int, List[Tuple[int, float]]],
    candidate_positions: Sequence[int],
    relation: str,
    head_weight_mode: str,
    head_delta_by_layer: Optional[Mapping[int, np.ndarray]] = None,
    relation_mass_normalization: str = "none",
) -> List[Dict[str, Any]]:
    """
    projection_by_layer[H] = [all_heads, P, 4]

    Returns exact state rows:
       state layer L=H-1, token position p, raw spatial score.

    No percentile transform: preserve original raw spatial projection.
    """
    rid = RID[relation]
    positions = list(map(int, candidate_positions))
    rows = []

    for H, proj in projection_by_layer.items():
        selected = selected_heads.get(int(H), [])
        if not selected:
            continue

        scores = []
        weights = []

        for h, acc in selected:
            x = np.asarray(proj[int(h), :, rid], dtype=np.float64)

            if relation_mass_normalization == "per_head_delta_norm":
                if head_delta_by_layer is None:
                    raise RuntimeError(
                        "per_head_delta_norm requested without head deltas"
                    )
                norm = float(
                    np.linalg.norm(head_delta_by_layer[int(H)][int(h)])
                )
                x = x / max(norm, EPS)

            w = 1.0
            if head_weight_mode == "accuracy":
                w = max(float(acc) - 0.25, EPS)

            scores.append(x)
            weights.append(w)

        S = np.stack(scores, axis=0)
        W = np.asarray(weights, dtype=np.float64)
        agg = np.average(S, axis=0, weights=W)

        source_layer = int(H) - 1
        for j, p in enumerate(positions):
            rows.append({
                "source_layer": source_layer,
                "head_layer": int(H),
                "position": int(p),
                "score": float(agg[j]),
                "relation": relation,
                "heads": ",".join(
                    hname(H, h) for h, _ in selected
                ),
            })

    return rows


def apply_global_unique(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    best = {}
    for row in rows:
        p = int(row["position"])
        if (
            p not in best
            or float(row["score"]) > float(best[p]["score"])
        ):
            best[p] = dict(row)
    return list(best.values())


def positive_mass_core(
    rows: Sequence[Dict[str, Any]],
    threshold: float,
    global_unique: bool,
) -> Dict[str, Any]:
    if global_unique:
        rows = apply_global_unique(rows)
    else:
        rows = [dict(x) for x in rows]

    positive = [
        dict(x)
        for x in rows
        if np.isfinite(float(x["score"])) and float(x["score"]) > 0.0
    ]
    positive.sort(
        key=lambda x: (
            float(x["score"]),
            -int(x["source_layer"]),
            -int(x["position"]),
        ),
        reverse=True,
    )

    total_mass = float(sum(float(x["score"]) for x in positive))
    selected = []

    if total_mass > 0:
        running = 0.0
        for rank, row in enumerate(positive, 1):
            z = dict(row)
            running += float(z["score"])
            z["positive_rank"] = rank
            z["cumulative_positive_mass"] = running
            z["cumulative_positive_mass_fraction"] = running / total_mass
            selected.append(z)
            if running / total_mass >= float(threshold):
                break

    return {
        "selected": selected,
        "total_positive_mass": total_mass,
        "n_positive_states": len(positive),
        "selected_n": len(selected),
        "selected_mass": float(
            sum(float(x["score"]) for x in selected)
        ),
        "selected_mass_fraction": (
            float(
                sum(float(x["score"]) for x in selected)
                / total_mass
            )
            if total_mass > 0
            else np.nan
        ),
    }


def choose_mass_relation(
    relation_cores: Mapping[str, Dict[str, Any]],
) -> str:
    return max(
        REL,
        key=lambda r: (
            float(relation_cores[r]["total_positive_mass"]),
            -RID[r],
        ),
    )


def relation_vote_from_head_residuals(
    head_delta_by_layer: Mapping[int, np.ndarray],
    center: np.ndarray,
    dirs: np.ndarray,
    head_layers: Sequence[int],
    vote_heads: Sequence[Tuple[int, int, float]],
) -> Tuple[str, Dict[str, int], Dict[str, float]]:
    li = {H: i for i, H in enumerate(head_layers)}
    votes = Counter()
    support = np.zeros(4, dtype=np.float64)

    for H, h, _acc in vote_heads:
        q = head_delta_by_layer[int(H)][int(h)]
        z = normalize_np(q - center[li[int(H)], int(h)])
        s4 = np.einsum(
            "d,rd->r",
            z,
            dirs[li[int(H)], int(h)],
        )
        rid = int(np.argmax(s4))
        votes[rid] += 1
        support += s4

    max_vote = max(votes.values())
    tied = [r for r, c in votes.items() if c == max_vote]
    rid = (
        tied[0]
        if len(tied) == 1
        else max(tied, key=lambda r: support[r])
    )

    return (
        REL[rid],
        {REL[r]: int(votes.get(r, 0)) for r in range(4)},
        {REL[r]: float(support[r]) for r in range(4)},
    )


# =============================================================================
# Oracle overlap
# =============================================================================

def state_set(rows: Sequence[Mapping[str, Any]]) -> set:
    return {
        (int(r["source_layer"]), int(r["position"]))
        for r in rows
    }


def overlap_metrics(
    selected_rows: Sequence[Mapping[str, Any]],
    oracle: Mapping[str, Any],
    candidate_domain: str,
) -> Dict[str, Any]:
    pred_states = state_set(selected_rows)
    pred_positions = {p for _, p in pred_states}

    if candidate_domain == "text":
        target_states = set(oracle["text_states"])
        target_positions = set(oracle["text_positions"])
    else:
        target_states = set(oracle["all_states"])
        target_positions = set(oracle["all_positions"])

    exact_hits = len(pred_states & target_states)
    position_hits = len(pred_positions & target_positions)

    return {
        "selected_N": len(pred_states),
        "oracle_core50_N": len(target_states),
        "exact_hits": exact_hits,
        "position_hits": position_hits,
        "any_exact_hit": exact_hits > 0,
        "any_position_hit": position_hits > 0,
        "exact_precision": safe_div(exact_hits, len(pred_states)),
        "exact_recall": safe_div(exact_hits, len(target_states)),
        "position_precision": safe_div(position_hits, len(pred_positions)),
        "position_recall": safe_div(position_hits, len(target_positions)),
    }


def best_of_four_by_oracle_overlap(
    relation_cores: Mapping[str, Dict[str, Any]],
    oracle: Mapping[str, Any],
    candidate_domain: str,
) -> Tuple[str, Dict[str, Any]]:
    scored = []
    for relation in REL:
        met = overlap_metrics(
            relation_cores[relation]["selected"],
            oracle,
            candidate_domain,
        )
        scored.append((
            int(met["exact_hits"]),
            float(met["exact_recall"]) if np.isfinite(met["exact_recall"]) else -1.0,
            float(relation_cores[relation]["total_positive_mass"]),
            relation,
            met,
        ))
    scored.sort(reverse=True)
    _, _, _, relation, met = scored[0]
    return relation, met


# =============================================================================
# Generation editing
# =============================================================================

def decode_new_tokens(
    processor: Any,
    output_ids: torch.Tensor,
    input_length: int,
) -> str:
    ids = output_ids[0, input_length:]
    return processor.tokenizer.decode(
        ids,
        skip_special_tokens=True,
    ).strip()


@torch.inference_mode()
def generate_text(
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    max_new_tokens: int,
) -> str:
    input_length = int(batch["input_ids"].shape[1])
    output_ids = model.generate(
        **dict(batch),
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
    )
    text = decode_new_tokens(
        processor,
        output_ids,
        input_length,
    )
    del output_ids
    return text


def clone_layer_output_with_hidden(
    output: Any,
    new_hidden: torch.Tensor,
) -> Any:
    if torch.is_tensor(output):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    if isinstance(output, list):
        return [new_hidden] + list(output[1:])
    raise TypeError(
        f"Unsupported decoder layer output type={type(output)}"
    )


class EditHooks:
    """
    Edit decoder BLOCK OUTPUT state h_L at exact prompt positions.

    The hook only edits the full prompt-prefill call where sequence length equals
    expected_prompt_len. Cached one-token decoding steps are untouched.
    """

    def __init__(
        self,
        decoder_layers: Sequence[Any],
        edits: Mapping[int, Mapping[int, torch.Tensor]],
        expected_prompt_len: int,
        alpha: float,
    ):
        self.handles = []
        self.edits = edits
        self.expected_prompt_len = int(expected_prompt_len)
        self.alpha = float(alpha)

        for L, pos_to_delta in edits.items():
            layer = decoder_layers[int(L)]

            def make_hook(
                layer_idx: int,
                delta_map: Mapping[int, torch.Tensor],
            ):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if not torch.is_tensor(hidden):
                        return output

                    # Only prefill.
                    if hidden.ndim != 3 or hidden.shape[1] != self.expected_prompt_len:
                        return output

                    valid = [
                        int(p)
                        for p in delta_map
                        if 0 <= int(p) < hidden.shape[1]
                    ]
                    if not valid:
                        return output

                    edited = hidden.clone()
                    for p in valid:
                        delta = delta_map[p].to(
                            device=edited.device,
                            dtype=edited.dtype,
                        )
                        edited[:, p, :] = (
                            edited[:, p, :]
                            + self.alpha * delta
                        )

                    return clone_layer_output_with_hidden(
                        output,
                        edited,
                    )
                return hook

            self.handles.append(
                layer.register_forward_hook(
                    make_hook(int(L), pos_to_delta)
                )
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def build_edits(
    selected_rows: Sequence[Mapping[str, Any]],
    real_blocks: Mapping[int, torch.Tensor],
    gray_blocks: Mapping[int, torch.Tensor],
) -> Dict[int, Dict[int, torch.Tensor]]:
    edits = defaultdict(dict)

    for row in selected_rows:
        L = int(row["source_layer"])
        p = int(row["position"])

        if L not in real_blocks or L not in gray_blocks:
            continue

        rh = real_blocks[L]
        gh = gray_blocks[L]

        if rh.ndim != 3 or gh.ndim != 3:
            raise RuntimeError(
                f"Expected block hidden [1,seq,D], got "
                f"real={tuple(rh.shape)} gray={tuple(gh.shape)}"
            )
        if not (0 <= p < rh.shape[1] and 0 <= p < gh.shape[1]):
            continue

        delta = (
            rh[0, p].float()
            - gh[0, p].float()
        ).cpu()
        edits[L][p] = delta

    return {int(L): dict(v) for L, v in edits.items()}


@torch.inference_mode()
def generate_with_edits(
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    real_blocks: Mapping[int, torch.Tensor],
    gray_blocks: Mapping[int, torch.Tensor],
    alpha: float,
    max_new_tokens: int,
) -> str:
    edits = build_edits(
        selected_rows,
        real_blocks,
        gray_blocks,
    )

    if not edits:
        return generate_text(
            model,
            processor,
            batch,
            max_new_tokens,
        )

    hooks = EditHooks(
        decoder_layers=decoder_layers,
        edits=edits,
        expected_prompt_len=int(batch["input_ids"].shape[1]),
        alpha=alpha,
    )
    try:
        return generate_text(
            model,
            processor,
            batch,
            max_new_tokens,
        )
    finally:
        hooks.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if not (0.0 < a.mass_threshold <= 1.0):
        raise ValueError("--mass-threshold must be in (0,1].")

    generation_modes = parse_strs(a.generation_modes)
    allowed_modes = {"mass4", "gt", "vote"}
    bad = [x for x in generation_modes if x not in allowed_modes]
    if bad:
        raise ValueError(f"Bad --generation-modes={bad}")

    head_layers = parse_ints(a.head_layers)
    source_layers = [H - 1 for H in head_layers]

    outdir = Path(a.output_dir)
    synth_cache = outdir / "synthetic_realgray_direction_cache.npz"
    head_cache = outdir / "coco_head_calibration_cache.npz"
    errors_path = outdir / "errors.jsonl"

    if a.overwrite and outdir.exists():
        for p in list(outdir.iterdir()):
            if p == synth_cache and not a.overwrite_synthetic_cache:
                continue
            if p == head_cache and not a.overwrite_head_cache:
                continue
            if (
                p.name == "generation_per_sample.csv"
                and a.resume_generation
            ):
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    outdir.mkdir(parents=True, exist_ok=True)

    if a.overwrite_synthetic_cache and synth_cache.exists():
        synth_cache.unlink()
    if a.overwrite_head_cache and head_cache.exists():
        head_cache.unlink()

    # -------------------------------------------------------------------------
    # Data.
    # -------------------------------------------------------------------------
    data_root = Path(a.data_root)
    prompts = load_standard_prompts(Path(a.prompt_jsonl))
    records = load_coco_records(data_root)

    oracle_df, oracle_lookup = load_oracle_core50(
        Path(a.oracle_bank_dir)
    )

    all_meta = []
    for sid in sorted(set(prompts) & set(records) & set(oracle_lookup)):
        m = prompts[sid]
        all_meta.append({
            "sid": sid,
            "gt": m["gt"],
            "subject": m["subject"],
            "reference": m["reference"],
            "question_text": m["question_text"],
            "image_path": records[sid].image_path,
        })

    calib_sids, heldout_sids = stratified_calibration_sids(
        all_meta,
        a.head_calib_frac,
        a.split_seed,
    )
    calib_set = set(calib_sids)

    if a.eval_heldout_only:
        if not heldout_sids:
            raise RuntimeError(
                "--eval-heldout-only requested but head-calib-frac leaves no heldout SIDs."
            )
        eval_meta = [
            m for m in all_meta
            if int(m["sid"]) in set(heldout_sids)
        ]
    else:
        eval_meta = list(all_meta)

    if a.max_eval_samples > 0:
        eval_meta = eval_meta[: int(a.max_eval_samples)]

    if a.require_eval_n and len(eval_meta) != int(a.require_eval_n):
        raise RuntimeError(
            f"Expected eval N={a.require_eval_n}, got {len(eval_meta)}"
        )

    synthetic_rows = load_synthetic_rows(
        Path(a.synthetic_dir),
        a.synthetic_labels,
        a.synthetic_max_samples,
    )

    # -------------------------------------------------------------------------
    # Model.
    # -------------------------------------------------------------------------
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    model_cls = getattr(
        transformers,
        a.model_class,
        None,
    )
    if model_cls is None:
        raise RuntimeError(
            f"transformers has no class {a.model_class!r}"
        )

    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "device_map": {"": a.device},
        "dtype": resolve_dtype(a.dtype),
    }
    if a.attn_impl != "none":
        load_kwargs["attn_implementation"] = a.attn_impl

    print(f"Loading {a.model_id}", flush=True)
    try:
        model = model_cls.from_pretrained(
            a.model_id,
            **load_kwargs,
        )
    except TypeError:
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = model_cls.from_pretrained(
            a.model_id,
            **load_kwargs,
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        a.model_id,
        trust_remote_code=False,
    )
    configure_processor(model, processor)

    decoder_layers, decoder_path = resolve_decoder_layers(model)
    cfg = get_text_config(model)
    n_heads = int(cfg.num_attention_heads)
    hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        hidden_size = int(
            resolve_o_proj(resolve_attn(decoder_layers[0])).in_features
        )
    head_dim = hidden_size // n_heads

    for H in head_layers:
        if not (0 <= H < len(decoder_layers)):
            raise ValueError(f"Invalid head layer={H}")

    print("=" * 172)
    print("SPATIAL POSITIVE-MASS CORE50 -> REAL-GRAY GENERATION")
    print("=" * 172)
    print(
        f"model={a.model_id} | decoder={decoder_path} | "
        f"layers={len(decoder_layers)} | heads={n_heads} | head_dim={head_dim}"
    )
    print(
        f"COCO meta={len(all_meta)} | calibration={len(calib_sids)} | "
        f"eval={len(eval_meta)} | synthetic={len(synthetic_rows)}"
    )
    print(
        f"head layers={head_layers} -> state layers={source_layers} | "
        f"heads/layer={a.heads_per_layer}"
    )
    print(
        f"SpatialCore mass={a.mass_threshold:.2f} | alpha={a.alpha} | "
        f"candidate_domain={a.candidate_domain} | global_unique={a.global_unique}"
    )
    print()

    try:
        # =====================================================================
        # 1) Synthetic Real-Gray directions.
        # =====================================================================
        if synth_cache.exists():
            (
                syn_sid,
                syn_y,
                syn_X,
                cache_layers,
                syn_center,
                syn_dirs,
            ) = load_direction_cache(synth_cache)

            if cache_layers != head_layers:
                raise RuntimeError(
                    f"Synthetic cache layers={cache_layers}, requested={head_layers}. "
                    "Use --overwrite-synthetic-cache."
                )
            print(
                f"[synthetic cache] N={len(syn_y)} shape={syn_X.shape}"
            )
        else:
            syn_sids = []
            syn_labels = []
            syn_residuals = []

            for rec in tqdm(
                synthetic_rows,
                desc="SYNTHETIC Real-Gray directions",
            ):
                real = gray = None
                try:
                    real = Image.open(rec["image_path"]).convert("RGB")
                    gray = make_gray(real, a.gray_value)

                    rb = make_batch(
                        processor,
                        real,
                        rec["question_text"],
                        torch.device(a.device),
                    )
                    gb = make_batch(
                        processor,
                        gray,
                        rec["question_text"],
                        torch.device(a.device),
                    )

                    ids = rb["input_ids"][0].detach().cpu().tolist()
                    spos, rpos = locate_object_spans(
                        processor.tokenizer,
                        ids,
                        rec["subject"],
                        rec["reference"],
                    )

                    rc = run_capture(
                        model,
                        decoder_layers,
                        rb,
                        head_layers,
                        source_layers,
                        need_attn=False,
                        need_v=False,
                        need_blocks=False,
                    )
                    gc_ = run_capture(
                        model,
                        decoder_layers,
                        gb,
                        head_layers,
                        source_layers,
                        need_attn=False,
                        need_v=False,
                        need_blocks=False,
                    )

                    arr = []
                    for H in head_layers:
                        arr.append(
                            head_realgray_residual(
                                rc,
                                gc_,
                                H,
                                spos,
                                rpos,
                                a.direction_pool,
                            )
                        )

                    syn_sids.append(int(rec["sid"]))
                    syn_labels.append(rec["relation"])
                    syn_residuals.append(
                        np.stack(arr, axis=0)
                    )

                except Exception as exc:
                    append_jsonl(errors_path, {
                        "stage": "synthetic",
                        "sid": rec["sid"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-12:],
                    })
                    tqdm.write(
                        f"[synthetic ERROR] sid={rec['sid']} "
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    if real is not None:
                        real.close()
                    if gray is not None:
                        gray.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not syn_residuals:
                raise RuntimeError("No synthetic residuals extracted.")

            syn_sid = np.asarray(syn_sids, dtype=int)
            syn_y = np.asarray(syn_labels, dtype=object)
            syn_X = np.stack(syn_residuals).astype(np.float32)
            syn_center, syn_dirs = fit_direction_bank(
                syn_X,
                syn_y,
            )

            save_direction_cache(
                synth_cache,
                syn_sid,
                syn_y,
                syn_X,
                head_layers,
                syn_center,
                syn_dirs,
            )

        # =====================================================================
        # 2) COCO head calibration accuracy.
        # =====================================================================
        if head_cache.exists():
            z = np.load(head_cache, allow_pickle=True)
            cal_sid = np.asarray(z["sample_index"]).astype(int)
            cal_y = np.asarray(z["relation"], dtype=object)
            cal_X = np.asarray(z["residual"], dtype=np.float32)
            cache_layers = np.asarray(z["head_layers"]).astype(int).tolist()

            if cache_layers != head_layers:
                raise RuntimeError(
                    f"Head cache layers={cache_layers}, requested={head_layers}. "
                    "Use --overwrite-head-cache."
                )
            print(
                f"[head calibration cache] N={len(cal_y)} shape={cal_X.shape}"
            )
        else:
            meta_by_sid = {int(m["sid"]): m for m in all_meta}
            cal_resid = []
            cal_labels = []
            cal_ids = []

            for sid in tqdm(
                calib_sids,
                desc="COCO head calibration Real-Gray",
            ):
                m = meta_by_sid[sid]
                real = gray = None
                try:
                    real = Image.open(m["image_path"]).convert("RGB")
                    gray = make_gray(real, a.gray_value)

                    rb = make_batch(
                        processor,
                        real,
                        m["question_text"],
                        torch.device(a.device),
                    )
                    gb = make_batch(
                        processor,
                        gray,
                        m["question_text"],
                        torch.device(a.device),
                    )
                    ids = rb["input_ids"][0].detach().cpu().tolist()
                    spos, rpos = locate_object_spans(
                        processor.tokenizer,
                        ids,
                        m["subject"],
                        m["reference"],
                    )

                    rc = run_capture(
                        model,
                        decoder_layers,
                        rb,
                        head_layers,
                        source_layers,
                        need_attn=False,
                        need_v=False,
                        need_blocks=False,
                    )
                    gc_ = run_capture(
                        model,
                        decoder_layers,
                        gb,
                        head_layers,
                        source_layers,
                        need_attn=False,
                        need_v=False,
                        need_blocks=False,
                    )

                    arr = []
                    for H in head_layers:
                        arr.append(
                            head_realgray_residual(
                                rc,
                                gc_,
                                H,
                                spos,
                                rpos,
                                a.direction_pool,
                            )
                        )

                    cal_ids.append(sid)
                    cal_labels.append(m["gt"])
                    cal_resid.append(
                        np.stack(arr, axis=0)
                    )

                except Exception as exc:
                    append_jsonl(errors_path, {
                        "stage": "head_calibration",
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-12:],
                    })
                    tqdm.write(
                        f"[calib ERROR] sid={sid} {type(exc).__name__}: {exc}"
                    )
                finally:
                    if real is not None:
                        real.close()
                    if gray is not None:
                        gray.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not cal_resid:
                raise RuntimeError("No COCO head calibration residuals extracted.")

            cal_sid = np.asarray(cal_ids, dtype=int)
            cal_y = np.asarray(cal_labels, dtype=object)
            cal_X = np.stack(cal_resid).astype(np.float32)

            np.savez_compressed(
                head_cache,
                sample_index=cal_sid.astype(np.int64),
                relation=cal_y,
                residual=cal_X.astype(np.float16),
                head_layers=np.asarray(head_layers, dtype=np.int32),
            )

        head_acc_df = compute_head_accuracy(
            cal_X,
            cal_y,
            syn_center,
            syn_dirs,
            head_layers,
        )
        head_acc_df.to_csv(
            outdir / "coco_head_accuracy.csv",
            index=False,
        )

        selected_heads = select_heads_per_layer(
            head_acc_df,
            head_layers,
            a.heads_per_layer,
        )
        vote_heads = global_vote_heads(
            head_acc_df,
            a.vote_top_n,
        )

        selected_head_rows = []
        for H in head_layers:
            for rank, (h, acc) in enumerate(selected_heads[H], 1):
                selected_head_rows.append({
                    "head_layer": H,
                    "source_layer": H - 1,
                    "rank_within_layer": rank,
                    "head": h,
                    "head_name": hname(H, h),
                    "coco_calibration_accuracy": acc,
                })
        pd.DataFrame(selected_head_rows).to_csv(
            outdir / "selected_heads_per_layer.csv",
            index=False,
        )

        print("Selected heads per layer:")
        for H in head_layers:
            print(
                f"  H{H:02d}->L{H-1:02d}: "
                + ", ".join(
                    f"{hname(H,h)}={acc:.4f}"
                    for h, acc in selected_heads[H]
                )
            )
        print()

        # =====================================================================
        # 3) Evaluation + generation.
        # =====================================================================
        existing_generation = {}
        gen_path = outdir / "generation_per_sample.csv"

        if a.resume_generation and gen_path.exists():
            old = pd.read_csv(gen_path)
            for r in old.to_dict("records"):
                existing_generation[int(r["sid"])] = r
            print(
                f"[resume] existing generation rows={len(existing_generation)}"
            )

        candidate_relation_rows = []
        selected_state_rows = []
        overlap_rows = []
        generation_rows = list(existing_generation.values())

        for m in tqdm(
            eval_meta,
            desc="COCO spatial-Core50 + generation",
        ):
            sid = int(m["sid"])

            # If generation already finished, still skip all expensive work.
            if sid in existing_generation:
                continue

            real = gray = None
            try:
                real = Image.open(m["image_path"]).convert("RGB")
                gray = make_gray(real, a.gray_value)

                rb = make_batch(
                    processor,
                    real,
                    m["question_text"],
                    torch.device(a.device),
                )
                gb = make_batch(
                    processor,
                    gray,
                    m["question_text"],
                    torch.device(a.device),
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                toks = token_strings(processor, ids)
                spos, rpos = locate_object_spans(
                    processor.tokenizer,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                visual = set(
                    resolve_visual_indices(
                        model,
                        processor,
                        rb,
                        ids,
                    )
                )

                prompt_last = len(ids) - 1
                if a.candidate_domain == "text":
                    candidate_positions = [
                        p
                        for p in range(len(ids))
                        if p != prompt_last and p not in visual
                    ]
                else:
                    candidate_positions = [
                        p
                        for p in range(len(ids))
                        if p != prompt_last
                    ]

                # Full original spatial score and block-state delta sources.
                rc = run_capture(
                    model,
                    decoder_layers,
                    rb,
                    head_layers,
                    source_layers,
                    need_attn=True,
                    need_v=True,
                    need_blocks=True,
                )
                gc_ = run_capture(
                    model,
                    decoder_layers,
                    gb,
                    head_layers,
                    source_layers,
                    need_attn=True,
                    need_v=True,
                    need_blocks=True,
                )

                projection_by_layer = {}
                head_delta_by_layer = {}
                recon_by_layer = {}

                for li, H in enumerate(head_layers):
                    proj, hd, recon = (
                        direction_token_scores_all_relations(
                            real_cap=rc,
                            gray_cap=gc_,
                            H=H,
                            subject_positions=spos,
                            reference_positions=rpos,
                            pool=a.direction_pool,
                            dirs_h4d=syn_dirs[li],
                            candidate_positions=candidate_positions,
                        )
                    )
                    projection_by_layer[H] = proj
                    head_delta_by_layer[H] = hd
                    recon_by_layer[H] = recon

                # Four relation-specific positive-mass spatial cores.
                relation_cores = {}
                for relation in REL:
                    rows = aggregate_raw_state_scores(
                        projection_by_layer=projection_by_layer,
                        selected_heads=selected_heads,
                        candidate_positions=candidate_positions,
                        relation=relation,
                        head_weight_mode=a.head_weight,
                        head_delta_by_layer=head_delta_by_layer,
                        relation_mass_normalization=a.relation_mass_normalization,
                    )
                    core = positive_mass_core(
                        rows,
                        threshold=a.mass_threshold,
                        global_unique=a.global_unique,
                    )
                    relation_cores[relation] = core

                mass_relation = choose_mass_relation(relation_cores)

                vote_relation, vote_counts, vote_support = (
                    relation_vote_from_head_residuals(
                        head_delta_by_layer=head_delta_by_layer,
                        center=syn_center,
                        dirs=syn_dirs,
                        head_layers=head_layers,
                        vote_heads=vote_heads,
                    )
                )

                oracle = oracle_lookup[sid]
                oracle_best_relation, oracle_best_met = (
                    best_of_four_by_oracle_overlap(
                        relation_cores,
                        oracle,
                        a.candidate_domain,
                    )
                )

                relation_selected_map = {
                    "mass4": mass_relation,
                    "gt": m["gt"],
                    "vote": vote_relation,
                    "oracle_best4_reporting_only": oracle_best_relation,
                }

                # Per-relation candidate diagnostics.
                total_mass_sum = sum(
                    float(relation_cores[r]["total_positive_mass"])
                    for r in REL
                )
                for relation in REL:
                    core = relation_cores[relation]
                    met = overlap_metrics(
                        core["selected"],
                        oracle,
                        a.candidate_domain,
                    )

                    candidate_relation_rows.append({
                        "sid": sid,
                        "gt": m["gt"],
                        "relation": relation,
                        "is_gt": relation == m["gt"],
                        "is_mass_selected": relation == mass_relation,
                        "is_vote_selected": relation == vote_relation,
                        "is_oracle_best4": relation == oracle_best_relation,
                        "total_positive_mass": core["total_positive_mass"],
                        "mass_share_over_4": safe_div(
                            core["total_positive_mass"],
                            total_mass_sum,
                        ),
                        "n_positive_states": core["n_positive_states"],
                        "spatial_core50_N": core["selected_n"],
                        "selected_mass_fraction": core["selected_mass_fraction"],
                        **met,
                    })

                    for row in core["selected"]:
                        p = int(row["position"])
                        selected_state_rows.append({
                            "sid": sid,
                            "gt": m["gt"],
                            "candidate_relation": relation,
                            "is_mass_selected": relation == mass_relation,
                            "is_vote_selected": relation == vote_relation,
                            "is_gt_relation": relation == m["gt"],
                            "is_oracle_best4": relation == oracle_best_relation,
                            "source_layer": int(row["source_layer"]),
                            "head_layer": int(row["head_layer"]),
                            "position": p,
                            "token": toks[p] if 0 <= p < len(toks) else "",
                            "spatial_score": float(row["score"]),
                            "positive_rank": int(row["positive_rank"]),
                            "cumulative_positive_mass_fraction": float(
                                row["cumulative_positive_mass_fraction"]
                            ),
                            "in_oracle_core50_exact": (
                                (
                                    int(row["source_layer"]),
                                    p,
                                )
                                in (
                                    oracle["text_states"]
                                    if a.candidate_domain == "text"
                                    else oracle["all_states"]
                                )
                            ),
                            "in_oracle_core50_position": (
                                p
                                in (
                                    oracle["text_positions"]
                                    if a.candidate_domain == "text"
                                    else oracle["all_positions"]
                                )
                            ),
                            "heads": row["heads"],
                        })

                # Selected overlap rows.
                for selector_name, relation in relation_selected_map.items():
                    core = relation_cores[relation]
                    met = overlap_metrics(
                        core["selected"],
                        oracle,
                        a.candidate_domain,
                    )
                    overlap_rows.append({
                        "sid": sid,
                        "gt": m["gt"],
                        "selector": selector_name,
                        "selected_relation": relation,
                        "relation_correct": relation == m["gt"],
                        "total_positive_mass": core["total_positive_mass"],
                        "spatial_core50_N": core["selected_n"],
                        **met,
                    })

                # -------------------------------------------------------------
                # Generation.
                # -------------------------------------------------------------
                baseline_text = generate_text(
                    model,
                    processor,
                    rb,
                    a.max_new_tokens,
                )
                baseline_pred = canon_rel(baseline_text)
                baseline_correct = baseline_pred == m["gt"]

                gen_row = {
                    "sid": sid,
                    "gt": m["gt"],
                    "baseline_text": baseline_text,
                    "baseline_pred": baseline_pred,
                    "baseline_correct": baseline_correct,
                    "mass4_relation": mass_relation,
                    "mass4_relation_correct": mass_relation == m["gt"],
                    "vote_relation": vote_relation,
                    "vote_relation_correct": vote_relation == m["gt"],
                    "oracle_best4_relation": oracle_best_relation,
                    "mass_left": relation_cores["left"]["total_positive_mass"],
                    "mass_right": relation_cores["right"]["total_positive_mass"],
                    "mass_above": relation_cores["above"]["total_positive_mass"],
                    "mass_below": relation_cores["below"]["total_positive_mass"],
                    "mass4_core_N": relation_cores[mass_relation]["selected_n"],
                }

                for mode in generation_modes:
                    relation = relation_selected_map[mode]
                    selected = relation_cores[relation]["selected"]

                    edited_text = generate_with_edits(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        selected_rows=selected,
                        real_blocks=rc["block"],
                        gray_blocks=gc_["block"],
                        alpha=a.alpha,
                        max_new_tokens=a.max_new_tokens,
                    )
                    pred = canon_rel(edited_text)

                    gen_row[f"{mode}_relation"] = relation
                    gen_row[f"{mode}_selected_N"] = len(selected)
                    gen_row[f"{mode}_text"] = edited_text
                    gen_row[f"{mode}_pred"] = pred
                    gen_row[f"{mode}_correct"] = pred == m["gt"]

                generation_rows.append(gen_row)

                # Save incrementally.
                pd.DataFrame(generation_rows).sort_values("sid").to_csv(
                    gen_path,
                    index=False,
                )

                # Free the largest sample tensors immediately.
                del rc, gc_, rb, gb

            except Exception as exc:
                append_jsonl(errors_path, {
                    "stage": "eval_generation",
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-16:],
                })
                tqdm.write(
                    f"[eval ERROR] sid={sid} {type(exc).__name__}: {exc}"
                )
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =====================================================================
        # 4) Save candidate / overlap tables.
        # =====================================================================
        if candidate_relation_rows:
            pd.DataFrame(candidate_relation_rows).to_csv(
                outdir / "candidate_relation_summary.csv",
                index=False,
            )
        if selected_state_rows:
            pd.DataFrame(selected_state_rows).to_csv(
                outdir / "selected_spatial_core50_states.csv",
                index=False,
            )
        if overlap_rows:
            pd.DataFrame(overlap_rows).to_csv(
                outdir / "core50_overlap_per_sample.csv",
                index=False,
            )

        gen_df = pd.DataFrame(generation_rows)
        if len(gen_df):
            gen_df = gen_df.sort_values("sid")
            gen_df.to_csv(gen_path, index=False)

        # =====================================================================
        # 5) Aggregate overlap.
        # =====================================================================
        overlap_df = pd.DataFrame(overlap_rows)
        overlap_summary_rows = []

        if len(overlap_df):
            for selector, g in overlap_df.groupby("selector"):
                exact_hits = int(g["exact_hits"].sum())
                selected_n = int(g["selected_N"].sum())
                oracle_n = int(g["oracle_core50_N"].sum())
                pos_hits = int(g["position_hits"].sum())

                overlap_summary_rows.append({
                    "selector": selector,
                    "N": int(g["sid"].nunique()),
                    "relation_accuracy": float(
                        g["relation_correct"].mean()
                    ),
                    "samples_any_exact_hit": int(
                        g["any_exact_hit"].sum()
                    ),
                    "sample_any_exact_hit_rate": float(
                        g["any_exact_hit"].mean()
                    ),
                    "samples_any_position_hit": int(
                        g["any_position_hit"].sum()
                    ),
                    "sample_any_position_hit_rate": float(
                        g["any_position_hit"].mean()
                    ),
                    "mean_selected_N": safe_mean(g["selected_N"]),
                    "median_selected_N": safe_median(g["selected_N"]),
                    "micro_exact_precision": safe_div(
                        exact_hits, selected_n
                    ),
                    "micro_exact_recall": safe_div(
                        exact_hits, oracle_n
                    ),
                    "micro_position_precision": safe_div(
                        pos_hits, selected_n
                    ),
                    "mean_exact_hits": safe_mean(g["exact_hits"]),
                    "mean_position_hits": safe_mean(g["position_hits"]),
                })

        overlap_summary = pd.DataFrame(overlap_summary_rows)
        if len(overlap_summary):
            overlap_summary = overlap_summary.sort_values(
                "micro_exact_recall",
                ascending=False,
            )
            overlap_summary.to_csv(
                outdir / "core50_overlap_summary.csv",
                index=False,
            )

        # =====================================================================
        # 6) Aggregate generation.
        # =====================================================================
        generation_summary_rows = []

        if len(gen_df):
            valid_gt = gen_df["gt"].isin(REL)
            g = gen_df[valid_gt].copy()

            baseline_acc = float(
                g["baseline_correct"].astype(bool).mean()
            )
            generation_summary_rows.append({
                "condition": "baseline",
                "N": len(g),
                "accuracy": baseline_acc,
                "gain_vs_baseline": 0.0,
                "wrong_to_correct": 0,
                "correct_to_wrong": 0,
                "net": 0,
                "changed_correctness": 0,
                "mean_selected_N": 0.0,
            })

            for mode in generation_modes:
                col = f"{mode}_correct"
                if col not in g.columns:
                    continue

                edited = g[col].map(boolify)
                base = g["baseline_correct"].map(boolify)

                w2c = int((~base & edited).sum())
                c2w = int((base & ~edited).sum())
                acc = float(edited.mean())

                generation_summary_rows.append({
                    "condition": mode,
                    "N": len(g),
                    "accuracy": acc,
                    "gain_vs_baseline": acc - baseline_acc,
                    "wrong_to_correct": w2c,
                    "correct_to_wrong": c2w,
                    "net": w2c - c2w,
                    "changed_correctness": w2c + c2w,
                    "mean_selected_N": safe_mean(
                        g[f"{mode}_selected_N"]
                    ),
                })

        generation_summary = pd.DataFrame(
            generation_summary_rows
        )
        if len(generation_summary):
            generation_summary.to_csv(
                outdir / "generation_summary.csv",
                index=False,
            )

        # =====================================================================
        # 7) Report.
        # =====================================================================
        lines = []
        lines.append("=" * 172)
        lines.append(
            "SPATIAL POSITIVE-MASS CORE50 -> CAUSAL CORE50 + GENERATION"
        )
        lines.append("=" * 172)
        lines.append(
            f"N eval target={len(eval_meta)} | completed generation={len(gen_df)} | "
            f"head calibration N={len(cal_sid)}"
        )
        lines.append(
            f"heads/layer={a.heads_per_layer} | spatial positive-mass threshold="
            f"{a.mass_threshold:.2f} | alpha={a.alpha}"
        )
        lines.append(
            f"candidate_domain={a.candidate_domain} | global_unique={a.global_unique} | "
            f"relation mass normalization={a.relation_mass_normalization}"
        )
        lines.append("")

        lines.append("SELECTED COCO HIGH-ACCURACY HEADS PER LAYER")
        lines.append("-" * 172)
        for H in head_layers:
            lines.append(
                f"H{H:02d}->L{H-1:02d}: "
                + ", ".join(
                    f"{hname(H,h)}={acc:.4f}"
                    for h, acc in selected_heads[H]
                )
            )
        lines.append("")

        lines.append("ORACLE CAUSAL CORE50 OVERLAP")
        lines.append("-" * 172)
        if len(overlap_summary):
            lines.append(
                overlap_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        else:
            lines.append("(no overlap rows)")
        lines.append("")

        lines.append("GENERATION")
        lines.append("-" * 172)
        if len(generation_summary):
            lines.append(
                generation_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        else:
            lines.append("(no generation rows)")
        lines.append("")

        if len(gen_df) and "mass4_relation_correct" in gen_df.columns:
            lines.append(
                f"mass4 relation accuracy = "
                f"{gen_df['mass4_relation_correct'].map(boolify).mean():.4f}"
            )
        if len(gen_df) and "vote_relation_correct" in gen_df.columns:
            lines.append(
                f"head-vote relation accuracy = "
                f"{gen_df['vote_relation_correct'].map(boolify).mean():.4f}"
            )

        report = "\n".join(lines) + "\n"
        (outdir / "analysis_summary.txt").write_text(
            report,
            encoding="utf-8",
        )
        print(report)

        # Metadata.
        metadata = {
            "script_version": SCRIPT_VERSION,
            "model_id": a.model_id,
            "model_class": a.model_class,
            "dtype": a.dtype,
            "head_layers": head_layers,
            "source_layers": source_layers,
            "heads_per_layer": int(a.heads_per_layer),
            "head_calib_frac": float(a.head_calib_frac),
            "calibration_N": int(len(cal_sid)),
            "eval_target_N": int(len(eval_meta)),
            "candidate_domain": a.candidate_domain,
            "global_unique": bool(a.global_unique),
            "mass_threshold": float(a.mass_threshold),
            "head_weight": a.head_weight,
            "relation_mass_normalization": a.relation_mass_normalization,
            "alpha": float(a.alpha),
            "generation_modes": generation_modes,
            "gray_value": int(a.gray_value),
            "direction_pool": a.direction_pool,
            "spatial_score": (
                "c_RG=[A_real(sub)-A_real(ref)]V_real - "
                "[A_gray(sub)-A_gray(ref)]V_gray; "
                "S=<c_RG,d_relation>"
            ),
            "relation_selector_mass4": (
                "Compute independent 50%-positive-mass spatial core for each of "
                "left/right/above/below; choose relation with largest total positive "
                "spatial mass."
            ),
            "generation_edit": (
                "For selected exact states (L,p), edit decoder block output "
                "h_real[L,p] <- h_real[L,p] + alpha*(h_real[L,p]-h_gray[L,p]) "
                "during prompt prefill only, then run greedy model.generate()."
            ),
            "oracle_best4_warning": (
                "oracle_best4_reporting_only uses causal Core50 overlap to choose "
                "the best of four relation candidates; it is reporting only and "
                "never used for main generation."
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("Saved:", outdir)

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
