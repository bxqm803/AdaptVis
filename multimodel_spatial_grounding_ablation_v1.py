#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multimodel_spatial_grounding_ablation_v1.py

Purpose
-------
Cross-model / cross-dataset spatial grounding ablation for:

  Models:
    qwen-3b, qwen-7b, llava-7b, internvl-7b (resolved to repo internvl-8b)

  Datasets:
    coco_two, controlled_a

  Image conditions:
    Real, Gray, Wrong, HFlip, VFlip

  Measurements:
    1) Layer-wise TRUE prob/logit lens at prompt-last token
       hidden_L -> final LM norm -> LM head -> dataset-native relation words.
       COCO:       left/right/above/below
       Controlled: left/right/on/under
       Internally all labels are canonicalized to left/right/above/below.

    2) Per-head object relation direction:
       r_{L,H} = z_{subject,L,H} - z_{reference,L,H}
       where z is the per-head self-attention input immediately before W_O.
       A codebook is fit ONLY on Real-image TRAIN split and then frozen.

    3) Per-head attention centroid:
       object text token -> visual-token attention distribution -> normalized
       visual (x,y) centroid.
       Stores subject/reference centroid coordinates, dx, dy, distance, relation.

Head selection
--------------
All heads are visible in a single eager-attention forward anyway, so using fewer
heads does not materially reduce forward FLOPs. To avoid huge result tables and
test-set head selection leakage, this script:

  * uses Real TRAIN/VAL only;
  * fits the direction codebook on TRAIN;
  * ranks heads separately at EACH layer on VAL by:
        - direction accuracy
        - centroid accuracy
  * selects the UNION of top-K direction and top-K centroid heads per layer;
  * freezes this selected set and reports Real/Gray/Wrong/HFlip/VFlip on TEST.

Default K=2+2, hence <=4 selected heads per layer (less when rankings overlap).

Important semantics
-------------------
Gray:
  same image size, uniform RGB=(128,128,128), prompt unchanged.
  GT remains the original relation. It asks whether original-label spatial
  information survives after useful visual content is removed.

Wrong:
  deterministic mismatched real image from another TEST sample, prompt unchanged.
  GT remains the original relation. This is a negative-control metric, not a claim
  that the donor image has a valid answer to the target prompt.

HFlip:
  horizontally mirrored target image, prompt unchanged.
  Expected canonical relation transforms:
      left <-> right; above/below unchanged.

VFlip:
  vertically mirrored target image, prompt unchanged.
  Expected canonical relation transforms:
      above <-> below; left/right unchanged.

Primary outputs per model/dataset
---------------------------------
  config.json
  split.csv
  selected_heads.csv
  lens_sample.csv.gz
  lens_layer_summary.csv
  head_sample.csv.gz
  head_condition_summary.csv
  centroid_real_gray_pairs.csv.gz
  centroid_flip_pairs.csv.gz
  overall_summary.json
  errors.jsonl
  DONE

Matrix-level outputs
--------------------
  matrix_head_condition_summary.csv
  matrix_lens_layer_summary.csv
  matrix_selected_heads.csv
  matrix_status.json

Run examples
------------
Two-GPU matrix:
  python -u multimodel_spatial_grounding_ablation_v1.py \
    --mode matrix \
    --gpus 0,1 \
    --data-root data \
    --output-root output/spatial_grounding_ablation_v1

Small pilot:
  CUDA_VISIBLE_DEVICES=0 python -u multimodel_spatial_grounding_ablation_v1.py \
    --mode model \
    --model qwen-3b \
    --datasets controlled_a \
    --max-samples 120 \
    --output-root output/spatial_grounding_ablation_pilot \
    --force

Full Qwen-3B / Controlled-A:
  CUDA_VISIBLE_DEVICES=0 python -u multimodel_spatial_grounding_ablation_v1.py \
    --mode model \
    --model qwen-3b \
    --datasets controlled_a \
    --output-root output/spatial_grounding_ablation_v1

Notes
-----
* Run from the AdaptVis llava16 repo root.
* Requires eager attention because centroid analysis needs full attention probs.
* This script intentionally imports/reuses helpers already present in this branch:
      extract_two_object_relation_states.py
      extract_controlled_relation_states_standalone.py
      analyze_coco_centroid_generation_step1_v4.py
      analyze_coco_head_object_residual_direction_probe_v1.py
      scan_multimodel_spatial_logitlens_matrix_v1.py
* Free generation is optional (--run-generation). The core scan does not need it.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import gzip
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import extract_controlled_relation_states_standalone as controlled
import analyze_coco_centroid_generation_step1_v4 as centroid_base
import analyze_controlledA_centroid_step1_v2 as controlled_centroid_base
import analyze_coco_head_object_residual_direction_probe_v1 as direction_base
import scan_multimodel_spatial_logitlens_matrix_v1 as lens_base


SCRIPT_VERSION = "multimodel-spatial-grounding-ablation-v1"

RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
CONDITIONS = ("Real", "Gray", "Wrong", "HFlip", "VFlip")

DEFAULT_MODELS = ("qwen-3b", "qwen-7b", "llava-7b", "internvl-7b")
DEFAULT_DATASETS = ("coco_two", "controlled_a")

COCO_PROMPT = Path("prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
CONTROLLED_PROMPT = Path("prompts/Controlled_Images_A_with_answer_four_options.jsonl")

CANON_REL = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "top": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "bottom": "below",
}

HFLIP_REL = {
    "left": "right",
    "right": "left",
    "above": "above",
    "below": "below",
}
VFLIP_REL = {
    "left": "left",
    "right": "right",
    "above": "below",
    "below": "above",
}


# -----------------------------------------------------------------------------
# Records / data
# -----------------------------------------------------------------------------

@dataclass
class Example:
    sid: int
    relation: str              # canonical
    subject: str
    reference: str
    question: str              # clean user text, no <image>/USER wrapper
    image_path: Optional[Path] = None
    image_obj: Optional[Image.Image] = None

    def image(self) -> Image.Image:
        if self.image_obj is not None:
            return self.image_obj.copy().convert("RGB")
        if self.image_path is None:
            raise RuntimeError(f"sid={self.sid}: no image source")
        return Image.open(self.image_path).convert("RGB")


def canon_relation(x: Any) -> Optional[str]:
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
    key = str(x).strip().lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    return CANON_REL.get(key)


def clean_question(text: str) -> str:
    # Reuse the centroid script's exact prompt cleaner.
    return centroid_base.extract_standard_user_text(str(text))


def load_coco_examples(data_root: Path, max_samples: Optional[int]) -> Tuple[List[Example], List[Any]]:
    records, audit = base.load_records("coco_two", data_root, None)
    prompt_path = COCO_PROMPT
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing {prompt_path}. Run from AdaptVis repo root or pass correct repo files."
        )
    prompts = centroid_base.load_standard_prompts(prompt_path)

    out: List[Example] = []
    for r in records:
        rel = canon_relation(getattr(r, "relation", None))
        if rel not in RELATIONS:
            continue
        sid = int(r.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        out.append(
            Example(
                sid=sid,
                relation=rel,
                subject=str(p["subject"]),
                reference=str(p["reference"]),
                question=str(p["question_text"]),
                image_path=Path(r.image_path),
            )
        )
    if max_samples is not None and max_samples > 0:
        out = stratified_cap(out, max_samples, seed=1)
    return out, list(audit)


def load_controlled_examples(
    max_samples: Optional[int],
    *,
    download: bool,
    num_workers: int,
) -> Tuple[List[Example], List[Any]]:
    prompt_path = CONTROLLED_PROMPT
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing {prompt_path}. Run from AdaptVis repo root."
        )

    records, audit = controlled.load_records(
        prompt_path,
        dataset_key="Controlled_Images_A",
        keep_relations=["left", "right", "on", "under"],
        download=download,
        max_samples=None,
        num_workers=num_workers,
    )
    prompts = controlled_centroid_base.load_standard_prompts(prompt_path)

    out: List[Example] = []
    for r in records:
        rel = canon_relation(r.relation)
        if rel not in RELATIONS:
            continue
        sid = int(r.sid)
        p = prompts.get(sid)
        if p is None:
            continue
        out.append(
            Example(
                sid=sid,
                relation=rel,
                subject=str(p["subject"]),
                reference=str(p["reference"]),
                question=str(p["question_text"]),
                image_obj=r.image.copy().convert("RGB"),
            )
        )

    if max_samples is not None and max_samples > 0:
        out = stratified_cap(out, max_samples, seed=1)
    return out, list(audit)


def load_examples(
    dataset: str,
    data_root: Path,
    max_samples: Optional[int],
    *,
    download: bool,
    num_workers: int,
) -> Tuple[List[Example], List[Any]]:
    if dataset == "coco_two":
        return load_coco_examples(data_root, max_samples)
    if dataset == "controlled_a":
        return load_controlled_examples(
            max_samples,
            download=download,
            num_workers=num_workers,
        )
    raise ValueError(dataset)


def stratified_cap(examples: Sequence[Example], n: int, seed: int) -> List[Example]:
    xs = list(examples)
    if n <= 0 or n >= len(xs):
        return sorted(xs, key=lambda r: r.sid)
    rng = random.Random(seed)
    by: Dict[str, List[Example]] = defaultdict(list)
    for x in xs:
        by[x.relation].append(x)
    for g in by.values():
        rng.shuffle(g)
    out: List[Example] = []
    ptr = {r: 0 for r in RELATIONS}
    while len(out) < n:
        moved = False
        for rel in RELATIONS:
            i = ptr[rel]
            if i < len(by.get(rel, [])) and len(out) < n:
                out.append(by[rel][i])
                ptr[rel] += 1
                moved = True
        if not moved:
            break
    return sorted(out, key=lambda r: r.sid)


def stratified_split(
    examples: Sequence[Example],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Example], List[Example], List[Example]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Need train_ratio>0, val_ratio>0, train+val<1")
    rng = random.Random(seed)
    by: Dict[str, List[Example]] = defaultdict(list)
    for x in examples:
        by[x.relation].append(x)

    train: List[Example] = []
    val: List[Example] = []
    test: List[Example] = []
    for rel in RELATIONS:
        g = list(by.get(rel, []))
        rng.shuffle(g)
        n = len(g)
        nt = max(1, int(round(n * train_ratio)))
        nv = max(1, int(round(n * val_ratio)))
        if nt + nv >= n:
            nv = max(1, n - nt - 1)
        train.extend(g[:nt])
        val.extend(g[nt:nt + nv])
        test.extend(g[nt + nv:])
    return (
        sorted(train, key=lambda x: x.sid),
        sorted(val, key=lambda x: x.sid),
        sorted(test, key=lambda x: x.sid),
    )


def build_wrong_map(test: Sequence[Example], seed: int) -> Dict[int, Example]:
    """Prefer donor with a different relation and different SID."""
    rng = random.Random(seed + 911)
    pool = list(test)
    out: Dict[int, Example] = {}
    for target in test:
        candidates = [x for x in pool if x.sid != target.sid and x.relation != target.relation]
        if not candidates:
            candidates = [x for x in pool if x.sid != target.sid]
        if not candidates:
            raise RuntimeError("Cannot build Wrong-image map from <2 samples")
        out[target.sid] = rng.choice(candidates)
    return out


# -----------------------------------------------------------------------------
# Image conditions
# -----------------------------------------------------------------------------

def make_condition_image(condition: str, target: Example, wrong: Optional[Example]) -> Image.Image:
    real = target.image()
    if condition == "Real":
        return real
    if condition == "Gray":
        return Image.new("RGB", real.size, color=(128, 128, 128))
    if condition == "Wrong":
        if wrong is None:
            raise RuntimeError("Wrong condition requires donor")
        donor = wrong.image()
        # Keep the donor's actual geometry; processor handles resizing.
        return donor
    if condition == "HFlip":
        return ImageOps.mirror(real)
    if condition == "VFlip":
        return ImageOps.flip(real)
    raise ValueError(condition)


def expected_relation(condition: str, original: str) -> str:
    if condition == "HFlip":
        return HFLIP_REL[original]
    if condition == "VFlip":
        return VFLIP_REL[original]
    # Gray/Wrong are negative controls against the original prompt label.
    return original


def transform_prediction(condition: str, pred: str) -> str:
    if condition == "HFlip":
        return HFLIP_REL[pred]
    if condition == "VFlip":
        return VFLIP_REL[pred]
    return pred


# -----------------------------------------------------------------------------
# Model / tokenizer / lens
# -----------------------------------------------------------------------------

def resolve_model(requested: str):
    actual = lens_base.resolve_model_alias(requested, base.SPECS)
    return actual, base.SPECS[actual]


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model_and_processor(requested: str, device: str):
    actual, spec = resolve_model(requested)
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )
    kwargs = dict(
        torch_dtype=resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": device},
        attn_implementation="eager",
    )
    print(f"[load] requested={requested} actual={actual} repo={spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    centroid_base.configure_processor(model, processor)
    return actual, spec, model, processor


def dataset_surface_map(dataset: str) -> Dict[str, str]:
    if dataset == "coco_two":
        return {
            "left": "left",
            "right": "right",
            "above": "above",
            "below": "below",
        }
    if dataset == "controlled_a":
        return {
            "left": "left",
            "right": "right",
            "above": "on",
            "below": "under",
        }
    raise ValueError(dataset)


def relation_token_map(tokenizer: Any, dataset: str) -> Dict[str, List[int]]:
    """Canonical label -> IDs of the dataset-native answer word."""
    surface = dataset_surface_map(dataset)
    out: Dict[str, List[int]] = {}
    for rel in RELATIONS:
        word = surface[rel]
        ids = set()
        for text in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            try:
                tok = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                tok = []
            if len(tok) == 1:
                ids.add(int(tok[0]))
        if not ids:
            tok = tokenizer.encode(" " + word, add_special_tokens=False)
            if not tok:
                raise RuntimeError(f"No token encoding for relation word {word!r}")
            ids.add(int(tok[-1]))
        out[rel] = sorted(ids)
    return out


def relation_stats(scores: np.ndarray, gt: str) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(4)
    g = RID[gt]
    centered = s - np.max(s)
    p = np.exp(centered)
    p /= p.sum()
    pred_i = int(np.argmax(s))
    wrong_i = max((i for i in range(4) if i != g), key=lambda i: float(s[i]))
    return {
        "pred": RELATIONS[pred_i],
        "correct": pred_i == g,
        "p_gt": float(p[g]),
        "margin": float(s[g] - s[wrong_i]),
        "p_left": float(p[0]),
        "p_right": float(p[1]),
        "p_above": float(p[2]),
        "p_below": float(p[3]),
    }


def parse_generated_relation(text: str, dataset: str) -> Optional[str]:
    s = str(text).lower()
    candidates: List[Tuple[int, str]] = []
    mapping = {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
    }
    if dataset == "controlled_a":
        mapping["on"] = "above"
    import re
    for word, rel in mapping.items():
        m = re.search(rf"\b{re.escape(word)}\b", s)
        if m:
            candidates.append((m.start(), rel))
    return min(candidates)[1] if candidates else None


# -----------------------------------------------------------------------------
# Forward extraction
# -----------------------------------------------------------------------------

def hidden_states_from_outputs(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
        getattr(getattr(outputs, "text_model_output", None), "hidden_states", None),
    ]
    for states in candidates:
        if isinstance(states, (tuple, list)) and states and torch.is_tensor(states[-1]):
            return tuple(states)
    raise RuntimeError("No language decoder hidden_states returned")


def attentions_from_outputs(outputs: Any) -> Tuple[Any, ...]:
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(getattr(outputs, "language_model_outputs", None), "attentions", None),
        getattr(getattr(outputs, "text_model_output", None), "attentions", None),
    ]
    for attn in candidates:
        if isinstance(attn, (tuple, list)) and len(attn) > 0:
            return tuple(attn)
    raise RuntimeError(
        "No language attentions returned. This experiment requires attn_implementation='eager'."
    )


def build_batch(
    processor: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
) -> Dict[str, Any]:
    # Use the same generic chat-template path as the existing probe.
    rendered = direction_base.build_chat_prompt(processor, question, True)
    batch = direction_base.process_inputs(processor, rendered, image, device)
    return batch


def attention_centroids_all_layers(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    attentions: Sequence[Any],
    n_layers: int,
    subject_index: int,
    reference_index: int,
) -> Dict[str, np.ndarray]:
    input_ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    input_len = len(input_ids)

    visual_indices = centroid_base.resolve_visual_indices(
        model, processor, batch, input_ids
    )
    coords = centroid_base.visual_coordinates(
        model,
        batch,
        len(visual_indices),
        batch["input_ids"].device,
    )
    if coords is None:
        raise RuntimeError(
            f"Could not construct visual coordinates for n_visual={len(visual_indices)}"
        )

    centers_list: List[np.ndarray] = []
    visual_mass_list: List[np.ndarray] = []
    separation_list: List[np.ndarray] = []
    entropy_list: List[np.ndarray] = []

    for layer in range(n_layers):
        tensor = centroid_base.normalize_attention_tensor(
            attentions[layer],
            expected_query_length=input_len,
        )
        rows = tensor[:, [subject_index, reference_index], :]
        metrics = centroid_base.query_attention_metrics(
            rows,
            visual_indices,
            coords,
            subject_index,
            reference_index,
        )
        maps = metrics["visual_maps"][:, :2, :]
        centers = metrics["centroids"][:, :2, :]
        sep = (
            0.5 * torch.sum(torch.abs(maps[:, 0, :] - maps[:, 1, :]), dim=-1)
        )
        centers_list.append(centers.detach().float().cpu().numpy())
        visual_mass_list.append(
            metrics["visual_mass"][:, :2].detach().float().cpu().numpy()
        )
        separation_list.append(sep.detach().float().cpu().numpy())
        entropy_list.append(
            metrics["entropy_confidence"][:, :2].detach().float().cpu().numpy()
        )

    return {
        "centroids": np.stack(centers_list, axis=0),       # [L,H,2,2]
        "visual_mass": np.stack(visual_mass_list, axis=0),# [L,H,2]
        "separation": np.stack(separation_list, axis=0),  # [L,H]
        "entropy": np.stack(entropy_list, axis=0),        # [L,H,2]
        "visual_indices": np.asarray(visual_indices, dtype=np.int32),
        "visual_coords": coords.detach().float().cpu().numpy(),
    }


def centroid_prediction(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    centers [...,2 objects,2 coords]
    returns pred_id, dx, dy, dist
    """
    dx = centers[..., 0, 0] - centers[..., 1, 0]
    dy = centers[..., 0, 1] - centers[..., 1, 1]
    ax = np.abs(dx)
    ay = np.abs(dy)
    pred = np.where(
        ax >= ay,
        np.where(dx < 0, RID["left"], RID["right"]),
        np.where(dy < 0, RID["above"], RID["below"]),
    ).astype(np.int64)
    dist = np.sqrt(dx * dx + dy * dy)
    return pred, dx, dy, dist


def forward_features(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[Any],
    n_heads: int,
    head_dim: int,
    lens: lens_base.RelationLogitLens,
    question: str,
    subject: str,
    reference: str,
    image: Image.Image,
    dataset: str,
    run_generation: bool,
    max_new_tokens: int,
) -> Dict[str, Any]:
    batch = build_batch(processor, question, image, device)
    input_ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    subject_span, reference_span = centroid_base.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])

    # Direction capture pools all sub-tokens in the phrase, matching the existing
    # direction-probe implementation.
    subject_positions = direction_base.locate_phrase_positions(
        processor.tokenizer, input_ids, subject
    )
    reference_positions = direction_base.locate_phrase_positions(
        processor.tokenizer, input_ids, reference
    )

    cap = direction_base.Capture(
        decoder_layers,
        n_heads,
        head_dim,
        subject_positions,
        reference_positions,
        "mean",
    )
    try:
        with cap:
            with torch.inference_mode():
                outputs = model(
                    **batch,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
        head_obj = cap.finalize()  # [L,H,2,D]
    finally:
        cap.close()

    hidden = hidden_states_from_outputs(outputs)
    attentions = attentions_from_outputs(outputs)
    n_layers = len(decoder_layers)
    if len(hidden) < n_layers + 1:
        raise RuntimeError(
            f"Need >= n_layers+1 hidden states; got {len(hidden)} for {n_layers} layers"
        )
    if len(attentions) < n_layers:
        raise RuntimeError(
            f"Need >= n_layers attentions; got {len(attentions)} for {n_layers} layers"
        )

    # Pre-W_O per-head subject-reference direction.
    direction = (
        head_obj[:, :, 0, :] - head_obj[:, :, 1, :]
    ).astype(np.float32)

    # TRUE prob lens: block output y_L == hidden_states[L+1], prompt-last.
    last_states = np.stack(
        [
            hidden[layer + 1][0, -1].detach().float().cpu().numpy()
            for layer in range(n_layers)
        ],
        axis=0,
    )
    lens_scores = lens.scores(last_states)  # [L,4]

    centroid = attention_centroids_all_layers(
        model=model,
        processor=processor,
        batch=batch,
        attentions=attentions,
        n_layers=n_layers,
        subject_index=subject_index,
        reference_index=reference_index,
    )

    native_scores = lens.scores_from_full_logits(outputs.logits[0, -1])

    generated_text = None
    generation_pred = None
    if run_generation:
        with torch.inference_mode():
            seq = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        input_len = int(batch["input_ids"].shape[1])
        new_ids = seq[0, input_len:].detach().cpu().tolist()
        generated_text = processor.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        )
        generation_pred = parse_generated_relation(generated_text, dataset)

    del outputs, hidden, attentions, batch
    return {
        "direction": direction,
        "centroids": centroid["centroids"],
        "visual_mass": centroid["visual_mass"],
        "separation": centroid["separation"],
        "entropy": centroid["entropy"],
        "lens_scores": lens_scores,
        "native_scores": native_scores,
        "generation_pred": generation_pred,
        "generated_text": generated_text,
    }


# -----------------------------------------------------------------------------
# Direction codebook / head selection
# -----------------------------------------------------------------------------

def fit_direction_codebooks(
    train_dirs: np.ndarray,      # [N,L,H,D]
    train_labels: np.ndarray,    # strings canonical
) -> Tuple[np.ndarray, np.ndarray]:
    N, L, H, D = train_dirs.shape
    centers = np.zeros((L, H, D), dtype=np.float32)
    dirs = np.zeros((L, H, 4, D), dtype=np.float32)
    for l in range(L):
        for h in range(H):
            c, d = direction_base.fit_codebook(
                np.asarray(train_dirs[:, l, h, :], dtype=np.float32),
                train_labels,
            )
            centers[l, h] = c
            dirs[l, h] = d
    return centers, dirs


def direction_scores(
    X: np.ndarray,               # [N,L,H,D] or [L,H,D]
    centers: np.ndarray,
    dirs: np.ndarray,
) -> np.ndarray:
    if X.ndim == 3:
        X = X[None, ...]
        squeeze = True
    else:
        squeeze = False
    Z = X.astype(np.float32) - centers[None, ...]
    norm = np.linalg.norm(Z, axis=-1, keepdims=True)
    Z = Z / np.maximum(norm, 1e-12)
    scores = np.einsum("nlhd,lhkd->nlhk", Z, dirs)
    return scores[0] if squeeze else scores


def val_head_accuracies(
    *,
    val_dirs: np.ndarray,
    val_centers_xy: np.ndarray,
    val_labels: Sequence[str],
    code_center: np.ndarray,
    code_dirs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    gt = np.asarray([RID[x] for x in val_labels], dtype=np.int64)

    ds = direction_scores(val_dirs, code_center, code_dirs)  # [N,L,H,4]
    dpred = np.argmax(ds, axis=-1)
    dacc = np.mean(dpred == gt[:, None, None], axis=0)        # [L,H]

    cpred, _, _, _ = centroid_prediction(val_centers_xy)     # [N,L,H]
    cacc = np.mean(cpred == gt[:, None, None], axis=0)
    return dacc.astype(np.float32), cacc.astype(np.float32)


def select_heads_per_layer(
    direction_val_acc: np.ndarray,
    centroid_val_acc: np.ndarray,
    top_direction: int,
    top_centroid: int,
) -> Tuple[Dict[int, List[int]], List[Dict[str, Any]]]:
    L, H = direction_val_acc.shape
    selected: Dict[int, List[int]] = {}
    rows: List[Dict[str, Any]] = []
    for l in range(L):
        d_order = np.argsort(-direction_val_acc[l])[:max(0, top_direction)]
        c_order = np.argsort(-centroid_val_acc[l])[:max(0, top_centroid)]
        union = sorted(set(map(int, d_order.tolist())) | set(map(int, c_order.tolist())))
        selected[l] = union
        for h in union:
            rows.append(
                {
                    "layer": l,
                    "head": h,
                    "head_name": f"L{l}H{h:02d}",
                    "selected_by_direction": int(h in set(map(int, d_order.tolist()))),
                    "selected_by_centroid": int(h in set(map(int, c_order.tolist()))),
                    "direction_val_acc": float(direction_val_acc[l, h]),
                    "centroid_val_acc": float(centroid_val_acc[l, h]),
                }
            )
    return selected, rows


# -----------------------------------------------------------------------------
# CSV / aggregation
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([dict(r) for r in rows])


def write_csv_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([dict(r) for r in rows])


def safe_mean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def cosine_2d(ax: float, ay: float, bx: float, by: float) -> float:
    na = math.sqrt(ax * ax + ay * ay)
    nb = math.sqrt(bx * bx + by * by)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((ax * bx + ay * by) / (na * nb))


def aggregate_lens_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(str(r["condition"]), int(r["layer"]))].append(r)
    out = []
    for (condition, layer), g in sorted(by.items(), key=lambda x: (CONDITIONS.index(x[0][0]), x[0][1])):
        out.append(
            {
                "condition": condition,
                "layer": layer,
                "N": len(g),
                "acc": safe_mean(float(x["correct"]) for x in g),
                "p_gt_mean": safe_mean(float(x["p_gt"]) for x in g),
                "margin_mean": safe_mean(float(x["margin"]) for x in g),
                "flip_consistency": safe_mean(
                    float(x["flip_consistent"])
                    for x in g
                    if x.get("flip_consistent") not in ("", None)
                ),
            }
        )
    return out


def aggregate_head_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(str(r["condition"]), int(r["layer"]), int(r["head"]))].append(r)

    out = []
    for (condition, layer, head), g in sorted(
        by.items(),
        key=lambda x: (CONDITIONS.index(x[0][0]), x[0][1], x[0][2]),
    ):
        out.append(
            {
                "condition": condition,
                "layer": layer,
                "head": head,
                "head_name": f"L{layer}H{head:02d}",
                "N": len(g),
                "direction_acc": safe_mean(float(x["direction_correct"]) for x in g),
                "direction_p_gt_mean": safe_mean(float(x["direction_p_gt"]) for x in g),
                "direction_margin_mean": safe_mean(float(x["direction_margin"]) for x in g),
                "direction_flip_consistency": safe_mean(
                    float(x["direction_flip_consistent"])
                    for x in g
                    if x.get("direction_flip_consistent") not in ("", None)
                ),
                "centroid_acc": safe_mean(float(x["centroid_correct"]) for x in g),
                "centroid_flip_consistency": safe_mean(
                    float(x["centroid_flip_consistent"])
                    for x in g
                    if x.get("centroid_flip_consistent") not in ("", None)
                ),
                "centroid_sub_x_mean": safe_mean(float(x["sub_x"]) for x in g),
                "centroid_sub_y_mean": safe_mean(float(x["sub_y"]) for x in g),
                "centroid_ref_x_mean": safe_mean(float(x["ref_x"]) for x in g),
                "centroid_ref_y_mean": safe_mean(float(x["ref_y"]) for x in g),
                "centroid_dx_mean": safe_mean(float(x["dx"]) for x in g),
                "centroid_dy_mean": safe_mean(float(x["dy"]) for x in g),
                "centroid_dist_mean": safe_mean(float(x["dist"]) for x in g),
                "object_map_separation_mean": safe_mean(float(x["map_separation"]) for x in g),
                "subject_visual_mass_mean": safe_mean(float(x["subject_visual_mass"]) for x in g),
                "reference_visual_mass_mean": safe_mean(float(x["reference_visual_mass"]) for x in g),
            }
        )
    return out


def add_cross_condition_metrics(
    summaries: List[Dict[str, Any]],
    centroid_gray_pairs: Sequence[Mapping[str, Any]],
    centroid_flip_pairs: Sequence[Mapping[str, Any]],
) -> None:
    # Per-head mean real->gray collapse metrics.
    by_rg: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for r in centroid_gray_pairs:
        by_rg[(int(r["layer"]), int(r["head"]))].append(r)

    by_h: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    by_v: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for r in centroid_flip_pairs:
        key = (int(r["layer"]), int(r["head"]))
        if r["condition"] == "HFlip":
            by_h[key].append(r)
        elif r["condition"] == "VFlip":
            by_v[key].append(r)

    for row in summaries:
        key = (int(row["layer"]), int(row["head"]))
        if row["condition"] == "Gray":
            g = by_rg.get(key, [])
            row["real_gray_sub_shift_mean"] = safe_mean(float(x["sub_shift"]) for x in g)
            row["real_gray_ref_shift_mean"] = safe_mean(float(x["ref_shift"]) for x in g)
            row["real_gray_rel_shift_mean"] = safe_mean(float(x["rel_shift"]) for x in g)
            row["real_gray_collapse_ratio_mean"] = safe_mean(float(x["collapse_ratio"]) for x in g)
            row["real_gray_rel_cos_mean"] = safe_mean(float(x["rel_cos"]) for x in g)
        if row["condition"] == "HFlip":
            g = by_h.get(key, [])
            row["centroid_equivariance_vector_error_mean"] = safe_mean(float(x["equiv_error"]) for x in g)
            row["centroid_equivariance_cos_mean"] = safe_mean(float(x["equiv_cos"]) for x in g)
        if row["condition"] == "VFlip":
            g = by_v.get(key, [])
            row["centroid_equivariance_vector_error_mean"] = safe_mean(float(x["equiv_error"]) for x in g)
            row["centroid_equivariance_cos_mean"] = safe_mean(float(x["equiv_cos"]) for x in g)


def build_lens_grounding_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for r in rows:
        by[(int(r["layer"]), str(r["condition"]))] = r
    layers = sorted({int(r["layer"]) for r in rows})
    out: List[Dict[str, Any]] = []
    for l in layers:
        item: Dict[str, Any] = {"layer": l}
        for cond in CONDITIONS:
            r = by.get((l, cond))
            if r is None:
                continue
            key = cond.lower()
            item[f"{key}_acc"] = float(r["acc"])
            item[f"{key}_p_gt_mean"] = float(r["p_gt_mean"])
            item[f"{key}_margin_mean"] = float(r["margin_mean"])
            if cond in ("HFlip", "VFlip"):
                item[f"{key}_flip_consistency"] = float(r.get("flip_consistency", float("nan")))
        if "real_acc" in item and "gray_acc" in item:
            item["real_minus_gray_acc"] = item["real_acc"] - item["gray_acc"]
        if "real_acc" in item and "wrong_acc" in item:
            item["real_minus_wrong_acc"] = item["real_acc"] - item["wrong_acc"]
        out.append(item)
    return out


def build_head_grounding_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[int, int, str], Mapping[str, Any]] = {}
    for r in rows:
        by[(int(r["layer"]), int(r["head"]), str(r["condition"]))] = r
    keys = sorted({(int(r["layer"]), int(r["head"])) for r in rows})
    out: List[Dict[str, Any]] = []
    for l, h in keys:
        item: Dict[str, Any] = {"layer": l, "head": h, "head_name": f"L{l}H{h:02d}"}
        real = by.get((l, h, "Real"))
        gray = by.get((l, h, "Gray"))
        wrong = by.get((l, h, "Wrong"))
        hf = by.get((l, h, "HFlip"))
        vf = by.get((l, h, "VFlip"))
        for cond, r in (("real", real), ("gray", gray), ("wrong", wrong), ("hflip", hf), ("vflip", vf)):
            if r is None:
                continue
            item[f"direction_{cond}_acc"] = float(r["direction_acc"])
            item[f"direction_{cond}_p_gt_mean"] = float(r["direction_p_gt_mean"])
            item[f"direction_{cond}_margin_mean"] = float(r["direction_margin_mean"])
            item[f"centroid_{cond}_acc"] = float(r["centroid_acc"])
            item[f"centroid_{cond}_dist_mean"] = float(r["centroid_dist_mean"])
            item[f"centroid_{cond}_dx_mean"] = float(r["centroid_dx_mean"])
            item[f"centroid_{cond}_dy_mean"] = float(r["centroid_dy_mean"])
            if cond in ("hflip", "vflip"):
                item[f"direction_{cond}_flip_consistency"] = float(r.get("direction_flip_consistency", float("nan")))
                item[f"centroid_{cond}_flip_consistency"] = float(r.get("centroid_flip_consistency", float("nan")))
                if "centroid_equivariance_vector_error_mean" in r:
                    item[f"centroid_{cond}_equiv_error_mean"] = float(r["centroid_equivariance_vector_error_mean"])
                if "centroid_equivariance_cos_mean" in r:
                    item[f"centroid_{cond}_equiv_cos_mean"] = float(r["centroid_equivariance_cos_mean"])
        if real is not None:
            item["selected_by_direction"] = real.get("selected_by_direction", "")
            item["selected_by_centroid"] = real.get("selected_by_centroid", "")
            item["direction_val_acc"] = real.get("direction_val_acc", "")
            item["centroid_val_acc"] = real.get("centroid_val_acc", "")
        if real is not None and gray is not None:
            item["direction_real_minus_gray_acc"] = float(real["direction_acc"]) - float(gray["direction_acc"])
            item["centroid_real_minus_gray_acc"] = float(real["centroid_acc"]) - float(gray["centroid_acc"])
            item["centroid_dist_gray_over_real"] = float(gray["centroid_dist_mean"]) / (float(real["centroid_dist_mean"]) + 1e-8)
            if "real_gray_collapse_ratio_mean" in gray:
                item["centroid_pairwise_collapse_ratio_mean"] = float(gray["real_gray_collapse_ratio_mean"])
                item["centroid_pairwise_sub_shift_mean"] = float(gray.get("real_gray_sub_shift_mean", float("nan")))
                item["centroid_pairwise_ref_shift_mean"] = float(gray.get("real_gray_ref_shift_mean", float("nan")))
                item["centroid_pairwise_rel_shift_mean"] = float(gray.get("real_gray_rel_shift_mean", float("nan")))
                item["centroid_pairwise_rel_cos_mean"] = float(gray.get("real_gray_rel_cos_mean", float("nan")))
        if real is not None and wrong is not None:
            item["direction_real_minus_wrong_acc"] = float(real["direction_acc"]) - float(wrong["direction_acc"])
            item["centroid_real_minus_wrong_acc"] = float(real["centroid_acc"]) - float(wrong["centroid_acc"])
        out.append(item)
    return out


# -----------------------------------------------------------------------------
# One model/dataset job
# -----------------------------------------------------------------------------

def run_job(args: argparse.Namespace, model_requested: str, dataset: str) -> Dict[str, Any]:
    out = Path(args.output_root) / model_requested / dataset
    done = out / "DONE"
    if done.exists() and not args.force:
        print(f"[skip] {model_requested}/{dataset}: {done}")
        return {"status": "skipped", "model": model_requested, "dataset": dataset, "output": str(out)}
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    max_samples = args.max_samples if args.max_samples > 0 else None
    examples, audit = load_examples(
        dataset,
        Path(args.data_root),
        max_samples,
        download=args.download,
        num_workers=args.num_workers,
    )
    if not examples:
        raise RuntimeError(f"{dataset}: no usable examples")
    train, val, test = stratified_split(
        examples,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    wrong_map = build_wrong_map(test, args.seed)

    print(
        f"\n[{model_requested}/{dataset}] "
        f"N={len(examples)} train={len(train)} val={len(val)} test={len(test)} "
        f"counts={dict(Counter(x.relation for x in examples))}"
    )

    split_rows = []
    for split_name, seq in (("train", train), ("val", val), ("test", test)):
        for x in seq:
            split_rows.append(
                {
                    "sid": x.sid,
                    "split": split_name,
                    "relation": x.relation,
                    "subject": x.subject,
                    "reference": x.reference,
                }
            )
    write_csv(out / "split.csv", split_rows)
    (out / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    actual, spec, model, processor = load_model_and_processor(
        model_requested,
        args.device,
    )
    device = torch.device(args.device)
    decoder_layers, decoder_path = lens_base.resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    n_heads, head_dim = direction_base.scan_shape(model, decoder_layers)
    final_norm, final_norm_path = lens_base.resolve_final_norm(model, decoder_path)
    lm_head, lm_head_path = lens_base.resolve_output_embeddings(model)
    token_map = relation_token_map(processor.tokenizer, dataset)
    lens = lens_base.RelationLogitLens(final_norm, lm_head, token_map)

    config = {
        "script_version": SCRIPT_VERSION,
        "model_requested": model_requested,
        "model_actual": actual,
        "repo_id": spec.repo_id,
        "dataset": dataset,
        "N": len(examples),
        "train_N": len(train),
        "val_N": len(val),
        "test_N": len(test),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "top_direction_per_layer": args.top_direction_per_layer,
        "top_centroid_per_layer": args.top_centroid_per_layer,
        "decoder_path": decoder_path,
        "final_norm_path": final_norm_path,
        "lm_head_path": lm_head_path,
        "dataset_surface_map": dataset_surface_map(dataset),
        "token_map": token_map,
        "run_generation": args.run_generation,
        "wrong_semantics": "mismatched real image; target prompt and original target label retained as negative control",
        "gray_semantics": "same-size uniform RGB 128 image; original target label retained",
        "hflip_semantics": "target image mirrored horizontally; left/right target transformed",
        "vflip_semantics": "target image mirrored vertically; above/below target transformed",
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Stage A: Real train/val, all heads. Used ONLY to fit probes/select heads.
    # -------------------------------------------------------------------------
    real_dirs: Dict[int, np.ndarray] = {}
    real_centroids: Dict[int, np.ndarray] = {}

    selection_examples = train + val
    errors: List[Dict[str, Any]] = []

    for ex in tqdm(selection_examples, desc=f"{model_requested}:{dataset}:select-real"):
        try:
            feat = forward_features(
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                n_heads=n_heads,
                head_dim=head_dim,
                lens=lens,
                question=ex.question,
                subject=ex.subject,
                reference=ex.reference,
                image=ex.image(),
                dataset=dataset,
                run_generation=False,
                max_new_tokens=args.max_new_tokens,
            )
            real_dirs[ex.sid] = feat["direction"].astype(np.float16)
            real_centroids[ex.sid] = feat["centroids"].astype(np.float16)
            del feat
        except Exception as exc:
            errors.append(
                {
                    "stage": "selection_real",
                    "sid": ex.sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                }
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_ok = [x for x in train if x.sid in real_dirs and x.sid in real_centroids]
    val_ok = [x for x in val if x.sid in real_dirs and x.sid in real_centroids]
    if min(len(train_ok), len(val_ok)) < 4:
        raise RuntimeError(
            f"Too few successful Real train/val samples: train={len(train_ok)} val={len(val_ok)}"
        )

    Xtr = np.stack([real_dirs[x.sid] for x in train_ok]).astype(np.float32)
    ytr = np.asarray([x.relation for x in train_ok])
    Xva = np.stack([real_dirs[x.sid] for x in val_ok]).astype(np.float32)
    yva = np.asarray([x.relation for x in val_ok])
    Cva = np.stack([real_centroids[x.sid] for x in val_ok]).astype(np.float32)

    code_center, code_dirs = fit_direction_codebooks(Xtr, ytr)
    direction_val_acc, centroid_val_acc = val_head_accuracies(
        val_dirs=Xva,
        val_centers_xy=Cva,
        val_labels=yva,
        code_center=code_center,
        code_dirs=code_dirs,
    )
    selected_heads, selected_rows = select_heads_per_layer(
        direction_val_acc,
        centroid_val_acc,
        args.top_direction_per_layer,
        args.top_centroid_per_layer,
    )
    for r in selected_rows:
        r["model_requested"] = model_requested
        r["model_actual"] = actual
        r["dataset"] = dataset
    write_csv(out / "selected_heads.csv", selected_rows)

    # Release all-head selection arrays before full 5-condition test.
    real_dirs.clear()
    real_centroids.clear()
    del Xtr, Xva, Cva
    gc.collect()

    # -------------------------------------------------------------------------
    # Stage B: TEST under all five conditions, selected heads only in outputs.
    # -------------------------------------------------------------------------
    lens_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []

    # Cache Real predictions/coords for paired flip/gray diagnostics.
    real_lens_pred: Dict[Tuple[int, int], str] = {}
    real_dir_pred: Dict[Tuple[int, int, int], str] = {}
    real_cent_pred: Dict[Tuple[int, int, int], str] = {}
    real_cent_xy: Dict[Tuple[int, int, int], Tuple[float, float, float, float, float, float, float]] = {}

    for condition in CONDITIONS:
        for ex in tqdm(test, desc=f"{model_requested}:{dataset}:{condition}"):
            donor = wrong_map.get(ex.sid) if condition == "Wrong" else None
            try:
                image = make_condition_image(condition, ex, donor)
                gt = expected_relation(condition, ex.relation)
                feat = forward_features(
                    model=model,
                    processor=processor,
                    device=device,
                    decoder_layers=decoder_layers,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    lens=lens,
                    question=ex.question,
                    subject=ex.subject,
                    reference=ex.reference,
                    image=image,
                    dataset=dataset,
                    run_generation=args.run_generation,
                    max_new_tokens=args.max_new_tokens,
                )

                # ----- prob lens -----
                for l in range(n_layers):
                    met = relation_stats(feat["lens_scores"][l], gt)
                    flip_consistent: Any = ""
                    if condition in ("HFlip", "VFlip"):
                        rp = real_lens_pred.get((ex.sid, l))
                        if rp is not None:
                            flip_consistent = int(met["pred"] == transform_prediction(condition, rp))
                    row = {
                        "sid": ex.sid,
                        "condition": condition,
                        "original_gt": ex.relation,
                        "gt": gt,
                        "layer": l,
                        **met,
                        "flip_consistent": flip_consistent,
                        "wrong_donor_sid": donor.sid if donor is not None else "",
                        "wrong_donor_relation": donor.relation if donor is not None else "",
                    }
                    lens_rows.append(row)
                    if condition == "Real":
                        real_lens_pred[(ex.sid, l)] = met["pred"]

                # ----- selected head direction + centroid -----
                dscore = direction_scores(
                    feat["direction"],
                    code_center,
                    code_dirs,
                )  # [L,H,4]
                cpred, cdx, cdy, cdist = centroid_prediction(feat["centroids"])

                for l in range(n_layers):
                    for h in selected_heads[l]:
                        # Direction
                        dm = relation_stats(dscore[l, h], gt)
                        dflip: Any = ""
                        if condition in ("HFlip", "VFlip"):
                            rp = real_dir_pred.get((ex.sid, l, h))
                            if rp is not None:
                                dflip = int(dm["pred"] == transform_prediction(condition, rp))

                        # Centroid
                        cent = feat["centroids"][l, h]
                        sub_x = float(cent[0, 0])
                        sub_y = float(cent[0, 1])
                        ref_x = float(cent[1, 0])
                        ref_y = float(cent[1, 1])
                        dx = float(cdx[l, h])
                        dy = float(cdy[l, h])
                        dist = float(cdist[l, h])
                        c_pred_name = RELATIONS[int(cpred[l, h])]
                        c_correct = int(c_pred_name == gt)
                        cflip: Any = ""
                        if condition in ("HFlip", "VFlip"):
                            rp = real_cent_pred.get((ex.sid, l, h))
                            if rp is not None:
                                cflip = int(c_pred_name == transform_prediction(condition, rp))

                        row = {
                            "sid": ex.sid,
                            "condition": condition,
                            "original_gt": ex.relation,
                            "gt": gt,
                            "layer": l,
                            "head": h,
                            "head_name": f"L{l}H{h:02d}",
                            "direction_pred": dm["pred"],
                            "direction_correct": int(dm["correct"]),
                            "direction_p_gt": dm["p_gt"],
                            "direction_margin": dm["margin"],
                            "direction_flip_consistent": dflip,
                            "centroid_pred": c_pred_name,
                            "centroid_correct": c_correct,
                            "centroid_flip_consistent": cflip,
                            "sub_x": sub_x,
                            "sub_y": sub_y,
                            "ref_x": ref_x,
                            "ref_y": ref_y,
                            "dx": dx,
                            "dy": dy,
                            "dist": dist,
                            "map_separation": float(feat["separation"][l, h]),
                            "subject_visual_mass": float(feat["visual_mass"][l, h, 0]),
                            "reference_visual_mass": float(feat["visual_mass"][l, h, 1]),
                            "subject_entropy_confidence": float(feat["entropy"][l, h, 0]),
                            "reference_entropy_confidence": float(feat["entropy"][l, h, 1]),
                            "wrong_donor_sid": donor.sid if donor is not None else "",
                            "wrong_donor_relation": donor.relation if donor is not None else "",
                        }
                        head_rows.append(row)

                        if condition == "Real":
                            real_dir_pred[(ex.sid, l, h)] = dm["pred"]
                            real_cent_pred[(ex.sid, l, h)] = c_pred_name
                            real_cent_xy[(ex.sid, l, h)] = (
                                sub_x, sub_y, ref_x, ref_y, dx, dy, dist
                            )

                if args.run_generation:
                    gp = feat["generation_pred"]
                    generation_rows.append(
                        {
                            "sid": ex.sid,
                            "condition": condition,
                            "original_gt": ex.relation,
                            "gt": gt,
                            "prediction": gp or "",
                            "correct": int(gp == gt) if gp is not None else 0,
                            "generated_text": feat["generated_text"] or "",
                        }
                    )

                del feat, image
            except Exception as exc:
                errors.append(
                    {
                        "stage": "test_condition",
                        "condition": condition,
                        "sid": ex.sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-12:],
                    }
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Pair centroid coordinates: Real vs Gray; Real vs H/V flip.
    # -------------------------------------------------------------------------
    by_head: Dict[Tuple[int, int, int, str], Mapping[str, Any]] = {}
    for r in head_rows:
        by_head[(int(r["sid"]), int(r["layer"]), int(r["head"]), str(r["condition"]))] = r

    gray_pairs: List[Dict[str, Any]] = []
    flip_pairs: List[Dict[str, Any]] = []
    for ex in test:
        for l in range(n_layers):
            for h in selected_heads[l]:
                real = by_head.get((ex.sid, l, h, "Real"))
                gray = by_head.get((ex.sid, l, h, "Gray"))
                if real is not None and gray is not None:
                    sub_shift = math.hypot(
                        float(real["sub_x"]) - float(gray["sub_x"]),
                        float(real["sub_y"]) - float(gray["sub_y"]),
                    )
                    ref_shift = math.hypot(
                        float(real["ref_x"]) - float(gray["ref_x"]),
                        float(real["ref_y"]) - float(gray["ref_y"]),
                    )
                    rel_shift = math.hypot(
                        float(real["dx"]) - float(gray["dx"]),
                        float(real["dy"]) - float(gray["dy"]),
                    )
                    collapse_ratio = float(gray["dist"]) / (float(real["dist"]) + 1e-8)
                    gray_pairs.append(
                        {
                            "sid": ex.sid,
                            "layer": l,
                            "head": h,
                            "head_name": f"L{l}H{h:02d}",
                            "gt": ex.relation,
                            "real_sub_x": real["sub_x"],
                            "real_sub_y": real["sub_y"],
                            "real_ref_x": real["ref_x"],
                            "real_ref_y": real["ref_y"],
                            "real_dx": real["dx"],
                            "real_dy": real["dy"],
                            "real_dist": real["dist"],
                            "gray_sub_x": gray["sub_x"],
                            "gray_sub_y": gray["sub_y"],
                            "gray_ref_x": gray["ref_x"],
                            "gray_ref_y": gray["ref_y"],
                            "gray_dx": gray["dx"],
                            "gray_dy": gray["dy"],
                            "gray_dist": gray["dist"],
                            "sub_shift": sub_shift,
                            "ref_shift": ref_shift,
                            "rel_shift": rel_shift,
                            "collapse_ratio": collapse_ratio,
                            "rel_cos": cosine_2d(
                                float(real["dx"]), float(real["dy"]),
                                float(gray["dx"]), float(gray["dy"]),
                            ),
                        }
                    )

                if real is not None:
                    for cond in ("HFlip", "VFlip"):
                        flip = by_head.get((ex.sid, l, h, cond))
                        if flip is None:
                            continue
                        rdx = float(real["dx"])
                        rdy = float(real["dy"])
                        if cond == "HFlip":
                            expected_dx, expected_dy = -rdx, rdy
                        else:
                            expected_dx, expected_dy = rdx, -rdy
                        fdx, fdy = float(flip["dx"]), float(flip["dy"])
                        flip_pairs.append(
                            {
                                "sid": ex.sid,
                                "condition": cond,
                                "layer": l,
                                "head": h,
                                "head_name": f"L{l}H{h:02d}",
                                "real_dx": rdx,
                                "real_dy": rdy,
                                "flip_dx": fdx,
                                "flip_dy": fdy,
                                "expected_dx": expected_dx,
                                "expected_dy": expected_dy,
                                "equiv_error": math.hypot(
                                    fdx - expected_dx,
                                    fdy - expected_dy,
                                ),
                                "equiv_cos": cosine_2d(
                                    fdx, fdy, expected_dx, expected_dy
                                ),
                            }
                        )

    # -------------------------------------------------------------------------
    # Save / summarize.
    # -------------------------------------------------------------------------
    lens_summary = aggregate_lens_rows(lens_rows)
    head_summary = aggregate_head_rows(head_rows)
    add_cross_condition_metrics(head_summary, gray_pairs, flip_pairs)

    # Attach selection validation metrics to test summaries.
    sel_lookup = {
        (int(r["layer"]), int(r["head"])): r
        for r in selected_rows
    }
    for r in head_summary:
        s = sel_lookup[(int(r["layer"]), int(r["head"]))]
        r["selected_by_direction"] = s["selected_by_direction"]
        r["selected_by_centroid"] = s["selected_by_centroid"]
        r["direction_val_acc"] = s["direction_val_acc"]
        r["centroid_val_acc"] = s["centroid_val_acc"]

    lens_grounding_summary = build_lens_grounding_summary(lens_summary)
    head_grounding_summary = build_head_grounding_summary(head_summary)

    write_csv_gz(out / "lens_sample.csv.gz", lens_rows)
    write_csv(out / "lens_layer_summary.csv", lens_summary)
    write_csv(out / "lens_grounding_summary.csv", lens_grounding_summary)
    write_csv_gz(out / "head_sample.csv.gz", head_rows)
    write_csv(out / "head_condition_summary.csv", head_summary)
    write_csv(out / "head_grounding_summary.csv", head_grounding_summary)
    write_csv_gz(out / "centroid_real_gray_pairs.csv.gz", gray_pairs)
    write_csv_gz(out / "centroid_flip_pairs.csv.gz", flip_pairs)
    if generation_rows:
        write_csv(out / "generation.csv", generation_rows)

    # Dataset/model headline numbers.
    headline: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "model_requested": model_requested,
        "model_actual": actual,
        "dataset": dataset,
        "N": len(examples),
        "train_N": len(train_ok),
        "val_N": len(val_ok),
        "test_N": len(test),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "selected_heads_total_layerwise": int(sum(len(v) for v in selected_heads.values())),
        "errors": len(errors),
        "lens": {},
        "direction": {},
        "centroid": {},
    }

    for cond in CONDITIONS:
        lg = [r for r in lens_summary if r["condition"] == cond]
        if lg:
            best = max(lg, key=lambda x: float(x["acc"]))
            final = max(lg, key=lambda x: int(x["layer"]))
            headline["lens"][cond] = {
                "best_acc": float(best["acc"]),
                "best_layer": int(best["layer"]),
                "final_acc": float(final["acc"]),
                "best_p_gt_mean": float(
                    max(lg, key=lambda x: float(x["p_gt_mean"]))["p_gt_mean"]
                ),
            }

        hg = [r for r in head_summary if r["condition"] == cond]
        if hg:
            dbest = max(hg, key=lambda x: float(x["direction_acc"]))
            cbest = max(hg, key=lambda x: float(x["centroid_acc"]))
            headline["direction"][cond] = {
                "best_test_acc": float(dbest["direction_acc"]),
                "best_layer": int(dbest["layer"]),
                "best_head": int(dbest["head"]),
                "best_head_name": str(dbest["head_name"]),
            }
            headline["centroid"][cond] = {
                "best_test_acc": float(cbest["centroid_acc"]),
                "best_layer": int(cbest["layer"]),
                "best_head": int(cbest["head"]),
                "best_head_name": str(cbest["head_name"]),
                "best_head_dist_mean": float(cbest["centroid_dist_mean"]),
            }

    # Global real-vs-gray collapse among selected heads.
    if gray_pairs:
        headline["centroid"]["real_gray_pair_metrics"] = {
            "mean_sub_shift": safe_mean(float(x["sub_shift"]) for x in gray_pairs),
            "mean_ref_shift": safe_mean(float(x["ref_shift"]) for x in gray_pairs),
            "mean_rel_shift": safe_mean(float(x["rel_shift"]) for x in gray_pairs),
            "mean_collapse_ratio": safe_mean(float(x["collapse_ratio"]) for x in gray_pairs),
            "mean_rel_cos": safe_mean(float(x["rel_cos"]) for x in gray_pairs),
        }

    if generation_rows:
        headline["generation"] = {}
        for cond in CONDITIONS:
            g = [x for x in generation_rows if x["condition"] == cond]
            headline["generation"][cond] = {
                "N": len(g),
                "acc": safe_mean(float(x["correct"]) for x in g),
            }

    (out / "overall_summary.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out / "errors.jsonl").open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    done.write_text("done\n", encoding="utf-8")
    print(json.dumps(headline, ensure_ascii=False, indent=2))

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "done",
        "model": model_requested,
        "model_actual": actual,
        "dataset": dataset,
        "output": str(out),
        "errors": len(errors),
    }


# -----------------------------------------------------------------------------
# Matrix collector / launcher
# -----------------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect_matrix(output_root: Path, models: Sequence[str], datasets: Sequence[str]) -> None:
    lens_rows: List[Dict[str, Any]] = []
    lens_grounding_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    head_grounding_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    status: List[Dict[str, Any]] = []

    for model in models:
        for dataset in datasets:
            d = output_root / model / dataset
            ok = (d / "DONE").exists()
            status.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "status": "done" if ok else "missing_or_failed",
                    "output": str(d),
                }
            )
            if not ok:
                continue

            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            actual = cfg["model_actual"]

            for r in read_csv_rows(d / "lens_layer_summary.csv"):
                lens_rows.append(
                    {
                        "model_requested": model,
                        "model_actual": actual,
                        "dataset": dataset,
                        **r,
                    }
                )
            for r in read_csv_rows(d / "lens_grounding_summary.csv"):
                lens_grounding_rows.append(
                    {
                        "model_requested": model,
                        "model_actual": actual,
                        "dataset": dataset,
                        **r,
                    }
                )
            for r in read_csv_rows(d / "head_condition_summary.csv"):
                head_rows.append(
                    {
                        "model_requested": model,
                        "model_actual": actual,
                        "dataset": dataset,
                        **r,
                    }
                )
            for r in read_csv_rows(d / "head_grounding_summary.csv"):
                head_grounding_rows.append(
                    {
                        "model_requested": model,
                        "model_actual": actual,
                        "dataset": dataset,
                        **r,
                    }
                )
            for r in read_csv_rows(d / "selected_heads.csv"):
                selected_rows.append(r)

    write_csv(output_root / "matrix_lens_layer_summary.csv", lens_rows)
    write_csv(output_root / "matrix_lens_grounding_summary.csv", lens_grounding_rows)
    write_csv(output_root / "matrix_head_condition_summary.csv", head_rows)
    write_csv(output_root / "matrix_head_grounding_summary.csv", head_grounding_rows)
    write_csv(output_root / "matrix_selected_heads.csv", selected_rows)
    (output_root / "matrix_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def child_command(args: argparse.Namespace, model: str) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--mode", "model",
        "--model", model,
        "--datasets", *args.datasets,
        "--data-root", args.data_root,
        "--output-root", args.output_root,
        "--device", "cuda:0",
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--seed", str(args.seed),
        "--top-direction-per-layer", str(args.top_direction_per_layer),
        "--top-centroid-per-layer", str(args.top_centroid_per_layer),
        "--max-samples", str(args.max_samples),
        "--num-workers", str(args.num_workers),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if args.download:
        cmd.append("--download")
    if args.force:
        cmd.append("--force")
    if args.run_generation:
        cmd.append("--run-generation")
    return cmd


def run_matrix(args: argparse.Namespace) -> None:
    models = list(args.models)
    datasets = list(args.datasets)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus is empty")

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    pending = list(models)
    active: List[Tuple[subprocess.Popen, str, str, Any]] = []

    while pending or active:
        while pending and len(active) < len(gpus):
            model = pending.pop(0)
            used = {gpu for _, _, gpu, _ in active}
            gpu = next((g for g in gpus if g not in used), gpus[0])
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = log_dir / f"{model}.log"
            log_f = log_path.open("w", encoding="utf-8")
            cmd = child_command(args, model)
            print(f"[launch] GPU {gpu}: {' '.join(cmd)}")
            p = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            active.append((p, model, gpu, log_f))

        time.sleep(2.0)
        still = []
        for p, model, gpu, log_f in active:
            rc = p.poll()
            if rc is None:
                still.append((p, model, gpu, log_f))
                continue
            log_f.close()
            print(f"[finish] GPU {gpu}: {model} rc={rc}")
            if rc != 0:
                print(f"  see {log_dir / (model + '.log')}")
        active = still

    collect_matrix(root, models, datasets)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["matrix", "model"], default="matrix")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--model", default="")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                   choices=list(DEFAULT_DATASETS))
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-root", default="output/spatial_grounding_ablation_v1")
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--top-direction-per-layer", type=int, default=2)
    p.add_argument("--top-centroid-per-layer", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = all usable samples")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--download", action="store_true")
    p.add_argument("--run-generation", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "matrix":
        run_matrix(args)
        return

    if not args.model:
        raise ValueError("--mode model requires --model")
    if args.model not in DEFAULT_MODELS:
        print(
            f"[warning] model={args.model!r} is outside the default four-model set "
            f"{DEFAULT_MODELS}"
        )

    statuses = []
    for dataset in args.datasets:
        try:
            statuses.append(run_job(args, args.model, dataset))
        except Exception as exc:
            statuses.append(
                {
                    "status": "failed",
                    "model": args.model,
                    "dataset": dataset,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(traceback.format_exc(), file=sys.stderr)
    collect_matrix(Path(args.output_root), [args.model], args.datasets)
    if any(x["status"] == "failed" for x in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
