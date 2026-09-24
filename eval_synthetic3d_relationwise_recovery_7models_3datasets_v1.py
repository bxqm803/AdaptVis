#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_synthetic3d_relationwise_recovery_7models_3datasets_v1.py

Quick ablation for the ICLR-2027 recovery section:
  * source: Synthetic-3D (six relations)
  * models: 7 VLMs
  * targets: COCO-two, Controlled-A, Controlled-B
  * relation representation: ONE independent direction per relation
  * NO horizontal/vertical/depth axis pairing

For relation r and decoder layer l, the source direction is

    q_i,l = [(h_sub - h_ref)_real - (h_sub - h_ref)_gray]
    mu_r,l = mean_i(q_i,l | relation=r)
    mu_l   = balanced mean_r(mu_r,l)
    d_r,l  = unit(mu_r,l - mu_l)

At inference, the model's own intermediate readout selects t = r_hat.
The intervention is

    h_sub,l <- h_sub,l + 0.5 * alpha_l * d_t,l
    h_ref,l <- h_ref,l - 0.5 * alpha_l * d_t,l

No relation is required to have an opposite and the directions are not
orthogonalized.

IMPORTANT
---------
This script is intentionally a DIRECT relation-wise ablation.  To make the
axis-vs-relation comparison easy to run before the deadline, it uses a
non-oracle positive scalar search over a fixed list of strengths.  The selector
is the model's own source-direction readout; ground truth is used only for
evaluation.  The first strength whose native generation agrees with the model's
own readout is accepted, with optional binary refinement.

This isolates the change requested here (independent relation directions)
without requiring H/V/D axes.  If you want a strict apples-to-apples replacement
inside the current minimum-distance/QP recovery implementation, transplant the
`fit_relation_directions` and `PairRelationSteer` pieces into that evaluator.

Run from the AdaptVis llava16 repository root.

Examples
--------
# smoke test
CUDA_VISIBLE_DEVICES=0 python eval_synthetic3d_relationwise_recovery_7models_3datasets_v1.py \
  --models qwen-3b --datasets coco --output-dir output/relationwise_3d_smoke \
  --target-max-samples 20 --overwrite

# full 7-model x 3-dataset sweep, sequential
CUDA_VISIBLE_DEVICES=0 python eval_synthetic3d_relationwise_recovery_7models_3datasets_v1.py \
  --models all --datasets all --output-dir output/relationwise_3d_7m3d

# two-GPU split (recommended)
CUDA_VISIBLE_DEVICES=0 python eval_synthetic3d_relationwise_recovery_7models_3datasets_v1.py \
  --models qwen2-2b,qwen-7b,llava-7b,internvl-1b --datasets all \
  --output-dir output/relationwise_3d_7m3d_gpu0 &

CUDA_VISIBLE_DEVICES=1 python eval_synthetic3d_relationwise_recovery_7models_3datasets_v1.py \
  --models qwen-3b,llava-13b,internvl-2b --datasets all \
  --output-dir output/relationwise_3d_7m3d_gpu1 &
wait
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
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

# Reuse tested repository-native model/data plumbing.
try:
    import eval_synthetic_fixedwindow_centroid_5models_3datasets_v1 as base
except Exception as exc:
    raise SystemExit(
        "Could not import eval_synthetic_fixedwindow_centroid_5models_3datasets_v1.py. "
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import extract_two_object_relation_states as twoobj
except Exception as exc:
    raise SystemExit(
        "Could not import extract_two_object_relation_states.py.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import analyze_coco_attention_flow_swap_step1_v1 as attncent
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_attention_flow_swap_step1_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "synthetic3d-relationwise-recovery-7models-3datasets-v1"

SOURCE_RELATIONS = ("left", "right", "above", "below", "front", "behind")
DEFAULT_MODEL_ORDER = (
    "qwen2-2b",
    "qwen-3b",
    "qwen-7b",
    "llava-7b",
    "llava-13b",
    "internvl-1b",
    "internvl-2b",
)
DEFAULT_DATASET_ORDER = ("coco", "controlled_a", "controlled_b")

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "qwen2-2b": {
        "kind": "base",
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct",
    },
    "qwen-3b": {
        "kind": "base",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    },
    "qwen-7b": {
        "kind": "base",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "llava-7b": {
        "kind": "llava",
        "repo_id": "llava-hf/llava-1.5-7b-hf",
        "model_class": "LlavaForConditionalGeneration",
    },
    "llava-13b": {
        "kind": "llava",
        "repo_id": "llava-hf/llava-1.5-13b-hf",
        "model_class": "LlavaForConditionalGeneration",
    },
    "internvl-1b": {
        "kind": "base",
        "repo_id": "OpenGVLab/InternVL2_5-1B",
    },
    "internvl-2b": {
        "kind": "base",
        "repo_id": "OpenGVLab/InternVL2_5-2B",
    },
}

TARGET_SOURCE_LABELS = {
    "coco": ("left", "right", "above", "below"),
    "controlled_a": ("left", "right", "above", "below"),
    "controlled_b": ("left", "right", "front", "behind"),
}

SOURCE_TO_TARGET = {
    "coco": {
        "left": "left", "right": "right", "above": "above", "below": "below",
    },
    "controlled_a": {
        "left": "left", "right": "right", "above": "on", "below": "under",
    },
    "controlled_b": {
        "left": "left", "right": "right", "front": "front", "behind": "behind",
    },
}

TARGET_TO_SOURCE = {
    dataset: {target: source for source, target in mapping.items()}
    for dataset, mapping in SOURCE_TO_TARGET.items()
}

SYN_REL_ALIAS = {
    "left": "left",
    "left_of": "left",
    "right": "right",
    "right_of": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "front": "front",
    "in-front": "front",
    "in_front": "front",
    "in-front-of": "front",
    "in_front_of": "front",
    "behind": "behind",
}


# ---------------------------------------------------------------------------
# CLI / utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", default="all")
    p.add_argument("--datasets", default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--revision", default="main")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--gray-value", type=int, default=128)

    # Qwen2.5-VL-3B paper window is L20--L26 (36 decoder blocks).
    # Other models use a seven-layer contiguous window centered at the same
    # relative decoder depth.
    p.add_argument("--reference-n-layers", type=int, default=36)
    p.add_argument("--reference-layer-start", type=int, default=20)
    p.add_argument("--reference-layer-end", type=int, default=26)

    p.add_argument("--synthetic-dir", default="synthetic_shapes_6dir_600_3d")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--coco-prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--controlled-json", default="data/controlled_images_dataset.json")
    p.add_argument("--controlled-b-json", default="data/controlled_clevr_dataset.json")
    p.add_argument("--target-max-samples", type=int, default=None)

    p.add_argument(
        "--scales",
        default="0.25,0.5,1,2,4,8,16",
        help="Positive relation-direction strengths tried in ascending order.",
    )
    p.add_argument(
        "--binary-refine",
        type=int,
        default=3,
        help="Binary refinements between the last failing and first successful strength.",
    )
    p.add_argument(
        "--skip-if-baseline-agrees-selector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not edit examples whose baseline output already matches the model's relation readout.",
    )

    # Native InternVL2.5 preprocessing, matching the existing repo benchmark.
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument(
        "--internvl-use-thumbnail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def parse_name_list(value: str, allowed: Sequence[str]) -> List[str]:
    if str(value).strip().lower() == "all":
        return list(allowed)
    xs = [x.strip() for x in str(value).split(",") if x.strip()]
    unknown = [x for x in xs if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown values={unknown}; allowed={list(allowed)}")
    return xs


def parse_scales(raw: str) -> List[float]:
    xs = sorted(set(float(x.strip()) for x in str(raw).split(",") if x.strip()))
    if not xs or xs[0] <= 0:
        raise ValueError("--scales must contain positive values")
    return xs


def seed_all(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unit_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    if not math.isfinite(n) or n < eps:
        raise RuntimeError(f"Cannot normalize vector with norm={n}")
    return (x / n).astype(np.float32)


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(den, eps))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, (v, v, v))


def open_image(record: Mapping[str, Any]) -> Image.Image:
    return Image.open(str(record["image_path"])).convert("RGB")


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def load_backend(alias: str, args: argparse.Namespace) -> base.Backend:
    spec = MODEL_SPECS[alias]
    if spec["kind"] == "base":
        # The existing benchmark already has tested Qwen2/2.5 and native
        # InternVL2.5 loading. Keep that code path.
        return base.load_backend(alias, args)

    repo_id = spec["repo_id"]
    model_cls = getattr(transformers, spec["model_class"], None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} lacks {spec['model_class']}"
        )
    dtype = base.dtype_from_name(args.dtype)
    device = torch.device(args.device)
    load_kwargs = dict(
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    try:
        model = model_cls.from_pretrained(repo_id, **load_kwargs)
    except TypeError:
        load_kwargs.pop("torch_dtype", None)
        load_kwargs["dtype"] = dtype
        model = model_cls.from_pretrained(repo_id, **load_kwargs)
    model = model.eval().to(device)

    processor = AutoProcessor.from_pretrained(
        repo_id,
        revision=args.revision,
        trust_remote_code=True,
        use_fast=False,
    )
    twoobj.configure_processor(model, processor)
    base.force_eager_config(model)
    layers, decoder_path = base.resolve_decoder_layers(model)
    tokenizer = processor.tokenizer

    print("\n" + "=" * 120)
    print(f"MODEL {alias} | repo={repo_id} | backend=llava | n_layers={len(layers)}")
    print("=" * 120)

    return base.Backend(
        alias=alias,
        kind="llava",
        repo_id=repo_id,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        layers=layers,
        decoder_path=decoder_path,
        dtype=dtype,
        device=device,
    )


def hf_build_batch(
    backend: base.Backend,
    image: Image.Image,
    question: str,
) -> Dict[str, Any]:
    prompt = attncent.build_prompt(backend.processor, question)
    attempts = (
        lambda: backend.processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        ),
        lambda: backend.processor(
            text=prompt, images=image, return_tensors="pt"
        ),
    )
    errors = []
    for fn in attempts:
        try:
            batch = fn()
            return {
                k: (v.to(backend.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(f"Processor failed: {errors[-1]}")


def generate_text(
    backend: base.Backend,
    image: Image.Image,
    question: str,
    args: argparse.Namespace,
) -> str:
    if backend.kind == "internvl25":
        return base.internvl_generate(backend, image, question, args)
    if backend.kind == "qwen":
        return base.qwen_generate(backend, image, question, args.max_new_tokens)

    batch = hf_build_batch(backend, image, question)
    input_len = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = backend.model.generate(
            **batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(args.max_new_tokens),
            use_cache=True,
        )
    new = out[0, input_len:] if int(out.shape[1]) > input_len else out[0]
    text = backend.tokenizer.decode(new, skip_special_tokens=True).strip()
    del batch, out
    return text


# ---------------------------------------------------------------------------
# Decoder window and object-pair state capture
# ---------------------------------------------------------------------------

def intervention_layers(n_layers: int, args: argparse.Namespace) -> List[int]:
    ref_n = int(args.reference_n_layers)
    lo = int(args.reference_layer_start)
    hi = int(args.reference_layer_end)
    if ref_n < 2 or not (0 <= lo <= hi < ref_n):
        raise ValueError("Invalid reference layer window")
    width = hi - lo + 1
    center_frac = ((lo + hi) / 2.0) / float(ref_n - 1)
    center = int(round(center_frac * (n_layers - 1)))
    start = center - width // 2
    start = max(0, min(start, n_layers - width))
    return list(range(start, start + width))


def _locate_hf_pair(
    backend: base.Backend,
    batch: Mapping[str, Any],
    subject: str,
    reference: str,
) -> Tuple[int, int]:
    ids = batch["input_ids"][0].detach().cpu().tolist()
    try:
        s_span, r_span = attncent.locate_object_spans(
            backend.tokenizer, ids, subject, reference
        )
        return int(s_span[1]), int(r_span[1])
    except Exception:
        s = twoobj.find_phrase_last_token(backend.tokenizer, ids, subject)
        r = twoobj.find_phrase_last_token(backend.tokenizer, ids, reference)
        return int(s), int(r)


def _hf_pair_states(
    backend: base.Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
) -> Dict[int, np.ndarray]:
    batch = hf_build_batch(backend, image, question)
    s_idx, r_idx = _locate_hf_pair(backend, batch, subject, reference)
    with torch.inference_mode():
        outputs = backend.model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    states = twoobj.hidden_tuple(outputs)
    result: Dict[int, np.ndarray] = {}
    for layer in selected_layers:
        state = states[int(layer) + 1]  # state[0] is embeddings
        if int(state.shape[1]) <= max(s_idx, r_idx):
            raise RuntimeError(
                f"Hidden/input token misalignment at L{layer}: "
                f"hidden_len={state.shape[1]}, s={s_idx}, r={r_idx}"
            )
        result[int(layer)] = (
            state[0, s_idx] - state[0, r_idx]
        ).detach().float().cpu().numpy().astype(np.float32)
    del batch, outputs, states
    return result


def _internvl_pair_states(
    backend: base.Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[int, np.ndarray]:
    pixels, _layout = base.internvl_pixels_and_layout(backend, image, args)
    _query, input_ids, attention_mask, visual_positions = base.internvl_chat_query(
        backend, question, int(pixels.shape[0])
    )
    ids = input_ids[0].detach().cpu().tolist()
    start = max(visual_positions) + 1 if visual_positions else 0
    s_span = base.token_span_for_phrase(backend.tokenizer, ids, subject, start=start)
    r_span = base.token_span_for_phrase(backend.tokenizer, ids, reference, start=start)
    s_idx, r_idx = int(s_span[-1]), int(r_span[-1])
    image_flags = torch.ones(
        (int(pixels.shape[0]), 1), device=backend.device, dtype=torch.long
    )
    with torch.inference_mode():
        outputs = backend.model(
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    states = twoobj.hidden_tuple(outputs)
    result: Dict[int, np.ndarray] = {}
    for layer in selected_layers:
        state = states[int(layer) + 1]
        result[int(layer)] = (
            state[0, s_idx] - state[0, r_idx]
        ).detach().float().cpu().numpy().astype(np.float32)
    del pixels, input_ids, attention_mask, image_flags, outputs, states
    return result


def pair_states(
    backend: base.Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[int, np.ndarray]:
    if backend.kind == "internvl25":
        return _internvl_pair_states(
            backend, image, question, subject, reference, selected_layers, args
        )
    return _hf_pair_states(
        backend, image, question, subject, reference, selected_layers
    )


def image_dependent_pair_states(
    backend: base.Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[int, np.ndarray]:
    gray = gray_image(image, args.gray_value)
    try:
        real = pair_states(
            backend, image, question, subject, reference, selected_layers, args
        )
        noimg = pair_states(
            backend, gray, question, subject, reference, selected_layers, args
        )
    finally:
        gray.close()
    return {
        int(layer): (real[int(layer)] - noimg[int(layer)]).astype(np.float32)
        for layer in selected_layers
    }


# ---------------------------------------------------------------------------
# Synthetic-3D loading + relation direction fit
# ---------------------------------------------------------------------------

def _first(item: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return None


def _object_name(item: Mapping[str, Any], role: str) -> str:
    if role == "subject":
        direct = _first(item, ("subject", "subj", "subject_name", "subj_name"))
        shape = _first(item, ("subject_shape", "subj_shape", "shape_subject", "shape_subj"))
        color = _first(item, ("subject_color", "subj_color", "color_subject", "color_subj"))
    else:
        direct = _first(item, ("reference", "ref", "reference_name", "ref_name"))
        shape = _first(item, ("reference_shape", "ref_shape", "shape_reference", "shape_ref"))
        color = _first(item, ("reference_color", "ref_color", "color_reference", "color_ref"))
    if direct is not None and not isinstance(direct, Mapping):
        return str(direct).strip()
    parts = [str(x).strip() for x in (color, shape) if x not in (None, "")]
    if parts:
        return " ".join(parts)
    raise KeyError(f"Could not infer {role} name from annotation keys={sorted(item.keys())}")


def _synthetic_image_path(root: Path, item: Mapping[str, Any], sid: int) -> Path:
    raw = _first(item, ("image", "image_path", "file", "filename"))
    candidates: List[Path] = []
    if raw is not None:
        p = Path(str(raw))
        candidates.extend([p, root / p, root / "images" / p.name])
    image_id = _first(item, ("image_id", "id"))
    if image_id is not None:
        s = str(image_id)
        candidates.extend([
            root / "images" / f"{s}.png",
            root / "images" / f"{int(image_id):06d}.png" if str(image_id).isdigit() else root / "images" / f"{s}.png",
        ])
    candidates.append(root / "images" / f"{sid + 1:06d}.png")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No Synthetic-3D image found; tried={candidates[:8]}")


def load_synthetic3d(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.synthetic_dir)
    if args.synthetic_labels:
        labels = Path(args.synthetic_labels)
    else:
        choices = [root / "annotations.jsonl", root / "labels.jsonl"]
        labels = next((p for p in choices if p.exists()), choices[0])
    if not labels.exists():
        raise FileNotFoundError(
            f"Synthetic-3D labels not found. Tried {labels}. "
            "Pass --synthetic-labels explicitly if needed."
        )

    rows: List[Dict[str, Any]] = []
    with labels.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_rel = str(item["relation"]).strip().lower().replace(" ", "_")
            if raw_rel not in SYN_REL_ALIAS:
                raise RuntimeError(
                    f"{labels}:{line_no}: unsupported relation={raw_rel!r}"
                )
            relation = SYN_REL_ALIAS[raw_rel]
            subject = _object_name(item, "subject")
            reference = _object_name(item, "reference")
            image_path = _synthetic_image_path(root, item, len(rows))
            question = (
                f"Where is the {subject} in relation to the {reference}? "
                "Answer with left, right, above, below, front, or behind."
            )
            rows.append({
                "sid": int(item.get("image_id", item.get("id", len(rows)))),
                "dataset": "synthetic3d",
                "image_path": str(image_path),
                "subject": subject,
                "reference": reference,
                "relation": relation,
                "question_text": question,
            })

    if args.source_max_samples is not None:
        rows = rows[: int(args.source_max_samples)]
    counts = Counter(r["relation"] for r in rows)
    missing = [r for r in SOURCE_RELATIONS if counts[r] == 0]
    if missing:
        raise RuntimeError(
            f"Synthetic-3D lacks required relations={missing}; counts={dict(counts)}"
        )
    print(f"[Synthetic-3D] N={len(rows)} counts={dict(counts)} labels={labels}")
    return rows


def fit_relation_directions(
    backend: base.Backend,
    source_records: Sequence[Mapping[str, Any]],
    selected_layers: Sequence[int],
    args: argparse.Namespace,
    model_outdir: Path,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Return directions[layer][relation] = unit(mu_relation - balanced_global_mu).

    This is explicitly relation-wise. No left/right, above/below, or
    front/behind subtraction is performed.
    """
    cache = model_outdir / "synthetic3d_relation_directions.npz"
    if cache.exists() and not args.overwrite:
        with np.load(cache, allow_pickle=True) as data:
            meta = json.loads(str(data["metadata_json"].item()))
            expected_layers = list(map(int, selected_layers))
            if meta.get("layers") != expected_layers:
                raise RuntimeError(
                    f"Cached layer mismatch {meta.get('layers')} vs {expected_layers}; "
                    "use --overwrite."
                )
            directions = {
                int(layer): {
                    r: np.asarray(data[f"L{layer}_{r}_direction"], dtype=np.float32)
                    for r in SOURCE_RELATIONS
                }
                for layer in selected_layers
            }
        print(f"[directions] loaded {cache}")
        return directions

    bags = {
        int(layer): {r: [] for r in SOURCE_RELATIONS}
        for layer in selected_layers
    }
    errors: List[Dict[str, Any]] = []

    for record in tqdm(source_records, desc=f"fit-3d-directions:{backend.alias}"):
        image = None
        try:
            image = open_image(record)
            q = image_dependent_pair_states(
                backend,
                image,
                str(record["question_text"]),
                str(record["subject"]),
                str(record["reference"]),
                selected_layers,
                args,
            )
            relation = str(record["relation"])
            for layer in selected_layers:
                bags[int(layer)][relation].append(q[int(layer)])
        except Exception as exc:
            errors.append({
                "sid": record.get("sid"),
                "relation": record.get("relation"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]),
            })
            if args.fail_fast:
                raise
        finally:
            if image is not None:
                image.close()
            cleanup()

    directions: Dict[int, Dict[str, np.ndarray]] = {}
    arrays: Dict[str, Any] = {}
    for layer in selected_layers:
        means: Dict[str, np.ndarray] = {}
        for relation in SOURCE_RELATIONS:
            vals = bags[int(layer)][relation]
            if not vals:
                raise RuntimeError(
                    f"No valid Synthetic-3D states for L{layer}/{relation}"
                )
            means[relation] = np.stack(vals, axis=0).mean(axis=0).astype(np.float32)
        # Balanced class mean, so every relation contributes equally.
        global_mean = np.stack(
            [means[r] for r in SOURCE_RELATIONS], axis=0
        ).mean(axis=0).astype(np.float32)
        directions[int(layer)] = {}
        arrays[f"L{layer}_global"] = global_mean
        for relation in SOURCE_RELATIONS:
            d = unit_np(means[relation] - global_mean)
            directions[int(layer)][relation] = d
            arrays[f"L{layer}_{relation}_mean"] = means[relation]
            arrays[f"L{layer}_{relation}_direction"] = d

    meta = {
        "script_version": SCRIPT_VERSION,
        "model": backend.alias,
        "repo_id": backend.repo_id,
        "layers": list(map(int, selected_layers)),
        "relations": list(SOURCE_RELATIONS),
        "direction_definition": "unit(mu_r - balanced_mean_relation_mu)",
        "axis_pairing": False,
        "gray_value": int(args.gray_value),
        "synthetic_dir": str(Path(args.synthetic_dir)),
        "source_n": int(len(source_records)),
        "source_relation_counts": dict(Counter(r["relation"] for r in source_records)),
        "n_errors": len(errors),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(meta), dtype=object)
    model_outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    write_json(model_outdir / "synthetic3d_relation_directions_metadata.json", meta)
    write_csv(model_outdir / "synthetic3d_direction_errors.csv", errors)
    print(f"[directions] saved {cache} | errors={len(errors)}")
    return directions


# ---------------------------------------------------------------------------
# Target data
# ---------------------------------------------------------------------------

def load_controlled_b(args: argparse.Namespace) -> List[Dict[str, Any]]:
    path = Path(args.controlled_b_json)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    skipped = Counter()

    for sid, item in enumerate(data):
        if not isinstance(item, Mapping):
            skipped["malformed"] += 1
            continue
        captions = item.get("caption_options")
        if not isinstance(captions, (list, tuple)) or not captions:
            skipped["no_caption_options"] += 1
            continue
        parsed = twoobj.parse_relation_caption(str(captions[0]))
        if parsed is None:
            skipped["caption_parse_failed"] += 1
            continue
        subject, reference, relation = parsed
        if relation not in ("left", "right", "front", "behind"):
            skipped["not_target_relation"] += 1
            continue

        raw_path = Path(str(item["image_path"]))
        candidates = [
            raw_path,
            Path(args.data_root) / raw_path,
            Path(args.data_root) / "controlled_clevr" / raw_path.name,
        ]
        image_path = next((p for p in candidates if p.exists()), None)
        if image_path is None:
            skipped["missing_image"] += 1
            continue

        rows.append({
            "sid": sid,
            "dataset": "controlled_b",
            "image_path": str(image_path),
            "subject": subject,
            "reference": reference,
            "relation": relation,
            "question_text": (
                f"Where is the {subject} in relation to the {reference}? "
                "Answer with left, right, front, or behind."
            ),
        })

    if args.target_max_samples is not None:
        rows = rows[: int(args.target_max_samples)]
    print(
        f"[Controlled-B] usable={len(rows)} "
        f"relations={dict(Counter(r['relation'] for r in rows))} "
        f"skipped={dict(skipped)}"
    )
    return rows


def load_target(dataset: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    if dataset == "coco":
        return base.load_coco(args)
    if dataset == "controlled_a":
        rows = base.load_controlled_a(args)
        if args.target_max_samples is not None:
            rows = rows[: int(args.target_max_samples)]
        return rows
    if dataset == "controlled_b":
        return load_controlled_b(args)
    raise ValueError(dataset)


def target_question(record: Mapping[str, Any], dataset: str) -> str:
    if record.get("question_text"):
        return str(record["question_text"])
    if dataset == "controlled_a":
        answers = "left, right, on, or under"
    elif dataset == "controlled_b":
        answers = "left, right, front, or behind"
    else:
        answers = "left, right, above, or below"
    return (
        f"Where is the {record['subject']} in relation to the {record['reference']}? "
        f"Answer with {answers}."
    )


def canonical_gt(dataset: str, relation: str) -> str:
    r = str(relation).lower().strip()
    if dataset == "controlled_a":
        aliases = {
            "left": "left", "left_of": "left",
            "right": "right", "right_of": "right",
            "on": "on", "above": "on", "over": "on",
            "under": "under", "below": "under", "underneath": "under",
        }
    elif dataset == "controlled_b":
        aliases = {
            "left": "left", "left_of": "left",
            "right": "right", "right_of": "right",
            "front": "front", "in-front": "front", "in_front": "front",
            "in-front-of": "front", "in_front_of": "front",
            "behind": "behind",
        }
    else:
        aliases = {
            "left": "left", "right": "right",
            "above": "above", "on": "above",
            "below": "below", "under": "below",
        }
    if r not in aliases:
        raise ValueError(f"Unknown {dataset} relation={relation!r}")
    return aliases[r]


def parse_prediction(text: str, dataset: str) -> Optional[str]:
    s = str(text).lower()
    if dataset == "controlled_a":
        pats = [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("on", r"\bon top of\b"),
            ("on", r"\babove\b"),
            ("on", r"\bon\b"),
            ("under", r"\bunder(?:neath)?\b"),
            ("under", r"\bbelow\b"),
            ("under", r"\bbeneath\b"),
        ]
    elif dataset == "controlled_b":
        pats = [
            ("front", r"\bin[\s-]+front(?:\s+of)?\b"),
            ("behind", r"\bbehind\b"),
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
        ]
    else:
        pats = [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("above", r"\babove\b"),
            ("above", r"\bon top of\b"),
            ("below", r"\bbelow\b"),
            ("below", r"\bunder(?:neath)?\b"),
        ]
    hits = []
    for label, pat in pats:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), label))
    return min(hits, key=lambda x: x[0])[1] if hits else None


# ---------------------------------------------------------------------------
# Relation readout
# ---------------------------------------------------------------------------

def relation_readout(
    backend: base.Backend,
    image: Image.Image,
    record: Mapping[str, Any],
    dataset: str,
    directions: Mapping[int, Mapping[str, np.ndarray]],
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question = target_question(record, dataset)
    q = image_dependent_pair_states(
        backend,
        image,
        question,
        str(record["subject"]),
        str(record["reference"]),
        selected_layers,
        args,
    )
    candidates = TARGET_SOURCE_LABELS[dataset]
    scores: Dict[str, float] = {}
    for relation in candidates:
        scores[relation] = float(np.mean([
            cosine_np(q[int(layer)], directions[int(layer)][relation])
            for layer in selected_layers
        ]))
    source_pred = max(candidates, key=lambda r: scores[r])
    target_pred = SOURCE_TO_TARGET[dataset][source_pred]
    return {
        "source_pred": source_pred,
        "target_pred": target_pred,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Relation-wise subject/reference intervention
# ---------------------------------------------------------------------------

def _generation_pair_indices(
    backend: base.Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    if backend.kind == "internvl25":
        pixels, _layout = base.internvl_pixels_and_layout(backend, image, args)
        _query, input_ids, _attention_mask, visual_positions = base.internvl_chat_query(
            backend, question, int(pixels.shape[0])
        )
        ids = input_ids[0].detach().cpu().tolist()
        start = max(visual_positions) + 1 if visual_positions else 0
        s = base.token_span_for_phrase(backend.tokenizer, ids, subject, start=start)[-1]
        r = base.token_span_for_phrase(backend.tokenizer, ids, reference, start=start)[-1]
        del pixels, input_ids
        return int(s), int(r)

    batch = hf_build_batch(backend, image, question)
    try:
        return _locate_hf_pair(backend, batch, subject, reference)
    finally:
        del batch


class PairRelationSteer:
    """
    Add +0.5*a*d_r to the subject and -0.5*a*d_r to the reference.

    The hook is applied once per selected block, on the prompt pass only.
    """

    def __init__(
        self,
        layers: Sequence[Any],
        directions: Mapping[int, Mapping[str, np.ndarray]],
        selected_layers: Sequence[int],
        relation: str,
        scale: float,
        subject_index: int,
        reference_index: int,
    ):
        self.handles = []
        self.done = {int(l): False for l in selected_layers}
        self.directions = directions
        self.relation = str(relation)
        self.scale = float(scale)
        self.subject_index = int(subject_index)
        self.reference_index = int(reference_index)
        for layer in selected_layers:
            self.handles.append(
                layers[int(layer)].register_forward_hook(self._hook(int(layer)))
            )

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if self.done[layer]:
                return output
            hidden, desc = base.output_hidden(output)
            if hidden.ndim != 3:
                return output
            if int(hidden.shape[1]) <= max(self.subject_index, self.reference_index):
                # During cached token-by-token decoding the sequence length is 1.
                # Wait for the full prompt pass.
                return output

            d_np = np.asarray(
                self.directions[layer][self.relation], dtype=np.float32
            )
            d = torch.as_tensor(
                d_np, device=hidden.device, dtype=hidden.dtype
            )
            delta = 0.5 * self.scale * d
            edited = hidden.clone()
            edited[:, self.subject_index, :] += delta
            edited[:, self.reference_index, :] -= delta
            self.done[layer] = True
            return base.replace_output_hidden(output, desc, edited)
        return hook

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def generate_with_relation_steer(
    backend: base.Backend,
    image: Image.Image,
    record: Mapping[str, Any],
    dataset: str,
    relation: str,
    scale: float,
    directions: Mapping[int, Mapping[str, np.ndarray]],
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[str, Optional[str]]:
    question = target_question(record, dataset)
    s_idx, r_idx = _generation_pair_indices(
        backend,
        image,
        question,
        str(record["subject"]),
        str(record["reference"]),
        args,
    )
    with PairRelationSteer(
        backend.layers,
        directions,
        selected_layers,
        relation,
        scale,
        s_idx,
        r_idx,
    ):
        text = generate_text(backend, image, question, args)
    return text, parse_prediction(text, dataset)


# ---------------------------------------------------------------------------
# Recovery evaluation
# ---------------------------------------------------------------------------

def transition(before: bool, after: bool) -> str:
    if not before and after:
        return "W2C"
    if before and not after:
        return "C2W"
    return "C2C" if before else "W2W"


def evaluate_dataset(
    backend: base.Backend,
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    directions: Mapping[int, Mapping[str, np.ndarray]],
    selected_layers: Sequence[int],
    args: argparse.Namespace,
    outdir: Path,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scales)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for record in tqdm(records, desc=f"relationwise:{dataset}:{backend.alias}"):
        image = None
        try:
            image = open_image(record)
            gt = canonical_gt(dataset, str(record["relation"]))
            question = target_question(record, dataset)

            baseline_text = generate_text(backend, image, question, args)
            baseline_pred = parse_prediction(baseline_text, dataset)

            readout = relation_readout(
                backend,
                image,
                record,
                dataset,
                directions,
                selected_layers,
                args,
            )
            source_target = str(readout["source_pred"])
            target_pred = str(readout["target_pred"])

            recovered_text = baseline_text
            recovered_pred = baseline_pred
            accepted_scale = 0.0

            should_search = not (
                args.skip_if_baseline_agrees_selector and baseline_pred == target_pred
            )
            if should_search:
                last_fail = 0.0
                first_success: Optional[float] = None
                first_success_text = None
                first_success_pred = None

                for scale in scales:
                    text, pred = generate_with_relation_steer(
                        backend,
                        image,
                        record,
                        dataset,
                        source_target,
                        float(scale),
                        directions,
                        selected_layers,
                        args,
                    )
                    if pred == target_pred:
                        first_success = float(scale)
                        first_success_text = text
                        first_success_pred = pred
                        break
                    last_fail = float(scale)

                if first_success is not None:
                    lo, hi = float(last_fail), float(first_success)
                    best_text = first_success_text
                    best_pred = first_success_pred
                    for _ in range(int(args.binary_refine)):
                        mid = 0.5 * (lo + hi)
                        text, pred = generate_with_relation_steer(
                            backend,
                            image,
                            record,
                            dataset,
                            source_target,
                            mid,
                            directions,
                            selected_layers,
                            args,
                        )
                        if pred == target_pred:
                            hi = mid
                            best_text, best_pred = text, pred
                        else:
                            lo = mid
                    accepted_scale = float(hi)
                    recovered_text = str(best_text)
                    recovered_pred = best_pred

            baseline_ok = bool(baseline_pred == gt)
            recovered_ok = bool(recovered_pred == gt)
            selector_ok = bool(target_pred == gt)

            row: Dict[str, Any] = {
                "sid": int(record["sid"]),
                "dataset": dataset,
                "model": backend.alias,
                "gt": gt,
                "baseline_pred": baseline_pred or "",
                "baseline_correct": int(baseline_ok),
                "selector_source_relation": source_target,
                "selector_pred": target_pred,
                "selector_correct": int(selector_ok),
                "recovered_pred": recovered_pred or "",
                "recovered_correct": int(recovered_ok),
                "accepted_scale": accepted_scale,
                "transition": transition(baseline_ok, recovered_ok),
                "baseline_text": baseline_text,
                "recovered_text": recovered_text,
            }
            for relation, score in readout["scores"].items():
                row[f"readout_cos_{relation}"] = float(score)
            rows.append(row)

        except Exception as exc:
            errors.append({
                "sid": record.get("sid"),
                "dataset": dataset,
                "model": backend.alias,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-10:]),
            })
            tqdm.write(
                f"[ERROR {backend.alias}/{dataset} sid={record.get('sid')}] "
                f"{type(exc).__name__}: {exc}"
            )
            if args.fail_fast:
                raise
        finally:
            if image is not None:
                image.close()
            cleanup()

    write_csv(outdir / "per_sample.csv", rows)
    write_csv(outdir / "errors.csv", errors)

    n = len(rows)
    baseline_acc = (
        float(np.mean([r["baseline_correct"] for r in rows])) if rows else float("nan")
    )
    selector_acc = (
        float(np.mean([r["selector_correct"] for r in rows])) if rows else float("nan")
    )
    recovered_acc = (
        float(np.mean([r["recovered_correct"] for r in rows])) if rows else float("nan")
    )
    transitions = Counter(r["transition"] for r in rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "model": backend.alias,
        "repo_id": backend.repo_id,
        "dataset": dataset,
        "n": n,
        "n_errors": len(errors),
        "baseline_acc": baseline_acc,
        "selector_acc": selector_acc,
        "recovered_acc": recovered_acc,
        "delta_acc": recovered_acc - baseline_acc,
        "W2C": int(transitions["W2C"]),
        "C2W": int(transitions["C2W"]),
        "C2C": int(transitions["C2C"]),
        "W2W": int(transitions["W2W"]),
        "mean_accepted_scale": (
            float(np.mean([r["accepted_scale"] for r in rows]))
            if rows else float("nan")
        ),
        "layers": list(map(int, selected_layers)),
        "direction_mode": "independent_relation",
        "axis_pairing": False,
        "source": "Synthetic-3D",
    }
    write_json(outdir / "summary.json", summary)

    print(
        f"[{backend.alias}/{dataset}] N={n} "
        f"base={baseline_acc:.4f} selector={selector_acc:.4f} "
        f"recovered={recovered_acc:.4f} delta={recovered_acc-baseline_acc:+.4f} "
        f"W2C={transitions['W2C']} C2W={transitions['C2W']}"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    models = parse_name_list(args.models, DEFAULT_MODEL_ORDER)
    datasets = parse_name_list(args.datasets, DEFAULT_DATASET_ORDER)
    outroot = Path(args.output_dir)
    outroot.mkdir(parents=True, exist_ok=True)

    source_records = load_synthetic3d(args)
    all_summaries: List[Dict[str, Any]] = []

    config = {
        "script_version": SCRIPT_VERSION,
        "models": models,
        "datasets": datasets,
        "source": "Synthetic-3D",
        "source_relations": list(SOURCE_RELATIONS),
        "direction_mode": "independent_relation",
        "axis_pairing": False,
        "scales": parse_scales(args.scales),
        "binary_refine": int(args.binary_refine),
        "seed": int(args.seed),
    }
    write_json(outroot / "config.json", config)

    for model_alias in models:
        backend = None
        try:
            backend = load_backend(model_alias, args)
            selected_layers = intervention_layers(len(backend.layers), args)
            print(
                f"[window] {model_alias}: n_layers={len(backend.layers)} "
                f"selected={selected_layers}"
            )
            model_out = outroot / model_alias
            directions = fit_relation_directions(
                backend,
                source_records,
                selected_layers,
                args,
                model_out,
            )

            for dataset in datasets:
                try:
                    records = load_target(dataset, args)
                    if not records:
                        raise RuntimeError(f"No target records for {dataset}")
                    summary = evaluate_dataset(
                        backend,
                        dataset,
                        records,
                        directions,
                        selected_layers,
                        args,
                        model_out / dataset,
                    )
                    all_summaries.append(summary)
                    write_csv(outroot / "summary.csv", all_summaries)
                    write_json(outroot / "summary.json", all_summaries)
                except Exception as exc:
                    print(
                        f"[DATASET FAILED] {model_alias}/{dataset}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    if args.fail_fast:
                        raise
                    cleanup()
        finally:
            if backend is not None:
                with contextlib.suppress(Exception):
                    backend.close()
            cleanup()

    write_csv(outroot / "summary.csv", all_summaries)
    write_json(outroot / "summary.json", all_summaries)

    print("\n" + "=" * 120)
    print("DONE")
    print(f"output={outroot}")
    for s in all_summaries:
        print(
            f"{s['model']:14s} {s['dataset']:13s} "
            f"{s['baseline_acc']:.4f} -> {s['recovered_acc']:.4f} "
            f"({s['delta_acc']:+.4f}) "
            f"selector={s['selector_acc']:.4f}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
