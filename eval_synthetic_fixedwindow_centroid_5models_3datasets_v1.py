#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_synthetic_fixedwindow_centroid_5models_3datasets_v1.py

Unified synthetic-source spatial steering benchmark for five VLMs and three targets.

Models
------
qwen2-2b    -> Qwen/Qwen2-VL-2B-Instruct
qwen-3b     -> Qwen/Qwen2.5-VL-3B-Instruct
qwen-7b     -> Qwen/Qwen2.5-VL-7B-Instruct
internvl-1b -> OpenGVLab/InternVL2_5-1B   (native InternVL2.5, NOT InternVL3-HF)
internvl-2b -> OpenGVLab/InternVL2_5-2B   (native InternVL2.5, NOT InternVL3-HF)

Target datasets
---------------
coco         : COCO-two, four-way left/right/above/below, full usable set
controlled_a : Controlled Images A, four-way left/right/on/under, full usable set
vg2_lr       : repo-native VG two-object six-option source, filtered to left/right

Source writer
-------------
All relation-specific late directions are fit ONLY from synthetic_shapes_4dir_400.
For each selected late layer l:

    delta_i,l = h_last(real)_i,l - h_last(gray)_i,l
    mu_r,l    = mean(delta_i,l | relation=r)
    mu_g,l    = 1/4 * sum_r mu_r,l
    s_r,l     = mu_r,l - mu_g,l

No target hidden state or target GT is used to fit the writer.

Fixed-window attention-spatial centroid selector
------------------------------------------------
There is NO best-layer/head selection and therefore NO GT-based centroid selection.
For a decoder with N blocks, use every block l satisfying

    centroid_depth_lo <= l/(N-1) <= centroid_depth_hi

Default: [0.45, 0.75].

For each selected (layer, head):
  1) read subject/reference -> visual-token attention on the original question;
  2) run the swapped question;
  3) role-align swapped subject/reference centroids back to the original roles;
  4) average original + aligned-swapped centroids;
  5) convert that head's displacement to a relation vote.

All (layer, head) units in the fixed window receive exactly ONE vote.  No source GT,
target GT, attention-mass weighting, best-head selection, or best-layer selection is
used.  Ties are resolved deterministically by summed geometric axis confidence only.

For VG2-LR, the task is binary, so each head votes from horizontal centroid sign (dx)
rather than allowing vertical labels that are outside the target label space.

Reported target methods
-----------------------
  baseline        : ordinary greedy generation
  oracle          : target GT chooses the frozen synthetic writer (upper bound)
  centroid_select : fixed-window attention centroid prediction, no steering
  centroid_final  : centroid prediction chooses the frozen synthetic writer, then
                    actual greedy generation

Coverage is tracked independently per method.  A centroid failure does NOT discard a
successful baseline/oracle result.

Reproducibility
---------------
All five models use bf16, greedy decoding, num_beams=1, do_sample=False, one seed,
fixed source data, fixed centroid relative-depth window, and eager/non-flash attention.
The script writes environment/config metadata and per-method coverage.

Run examples
------------
# First smoke test requested in development: Qwen2.5-VL-3B on full COCO
CUDA_VISIBLE_DEVICES=0 python eval_synthetic_fixedwindow_centroid_5models_3datasets_v1.py \
  --models qwen-3b --datasets coco --output-dir output/fixedwindow_v1 --overwrite

# Final sequential five-model x three-dataset run
CUDA_VISIBLE_DEVICES=0 python eval_synthetic_fixedwindow_centroid_5models_3datasets_v1.py \
  --models all --datasets all --output-dir output/fixedwindow_v1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.generation import GenerationConfig, GenerationMixin
from packaging import version as packaging_version

try:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except Exception as exc:
    raise SystemExit(f"torchvision is required for native InternVL2.5 preprocessing: {exc}")

# Repo-native helpers.  Run this script from the AdaptVis llava16 repository root.
try:
    import extract_two_object_relation_states as twoobj
except Exception as exc:
    raise SystemExit(
        "Could not import extract_two_object_relation_states.py. "
        "Run from the AdaptVis llava16 repository root.\n"
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


SCRIPT_VERSION = "synthetic-fixedwindow-centroid-5models-3datasets-v1"
SOURCE_RELATIONS = ("left", "right", "above", "below")
SOURCE_REL_TO_ID = {r: i for i, r in enumerate(SOURCE_RELATIONS)}
SOURCE_ID_TO_REL = {i: r for r, i in SOURCE_REL_TO_ID.items()}

SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
}

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "qwen2-2b": {
        "backend": "qwen",
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct",
        "model_class": "Qwen2VLForConditionalGeneration",
        # Tentative final-four window; kept explicit and logged, not claimed established.
        "actuator_layers": [24, 25, 26, 27],
    },
    "qwen-3b": {
        "backend": "qwen",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "actuator_layers": [32, 33, 34, 35],
    },
    "qwen-7b": {
        "backend": "qwen",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "actuator_layers": [25, 26, 27],
    },
    "internvl-1b": {
        "backend": "internvl25",
        "repo_id": "OpenGVLab/InternVL2_5-1B",
        "model_class": "AutoModel",
        "actuator_layers": [20, 21, 22, 23],
    },
    "internvl-2b": {
        "backend": "internvl25",
        "repo_id": "OpenGVLab/InternVL2_5-2B",
        "model_class": "AutoModel",
        "actuator_layers": [21, 22, 23],
    },
}

DEFAULT_MODEL_ORDER = [
    "qwen2-2b",
    "qwen-3b",
    "qwen-7b",
    "internvl-1b",
    "internvl-2b",
]
DEFAULT_DATASET_ORDER = ["coco", "controlled_a", "vg2_lr"]

SYNTHETIC_PROMPT = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with left, right, above, or below."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# CLI / reproducibility
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--models",
        default="all",
        help="all or comma-separated aliases: " + ",".join(DEFAULT_MODEL_ORDER),
    )
    p.add_argument(
        "--datasets",
        default="all",
        help="all or comma-separated subset of coco,controlled_a,vg2_lr",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--revision", default="main")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--scale", type=float, default=1.0)

    p.add_argument("--centroid-depth-lo", type=float, default=0.45)
    p.add_argument("--centroid-depth-hi", type=float, default=0.75)

    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)

    p.add_argument("--data-root", default="data")
    p.add_argument("--coco-prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--controlled-json", default="data/controlled_images_dataset.json")
    p.add_argument("--vg-json", default="data/vg_qa_two_obj.json")
    p.add_argument("--vg-prompt-jsonl", default="prompts/VG_QA_two_obj_with_answer_six_options.jsonl")
    p.add_argument("--vg-image-root", default="auto")
    p.add_argument("--target-max-samples", type=int, default=None)

    # Native InternVL2.5 preprocessing.  Defaults match the project's prior native runs.
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument(
        "--internvl-use-thumbnail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--min-success-rate", type=float, default=0.90)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default is to save the error and continue to the next dataset/model.",
    )
    return p.parse_args()


def parse_name_list(value: str, allowed: Sequence[str]) -> List[str]:
    if str(value).strip().lower() == "all":
        return list(allowed)
    result = [x.strip() for x in str(value).split(",") if x.strip()]
    unknown = [x for x in result if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown values {unknown}; allowed={list(allowed)}")
    return result


def seed_all(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def safe_mean(values: Iterable[Any]) -> float:
    xs: List[float] = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.mean(xs)) if xs else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
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


def environment_report(args: argparse.Namespace) -> Dict[str, Any]:
    report = {
        "script_version": SCRIPT_VERSION,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "seed": int(args.seed),
        "dtype": args.dtype,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "centroid_depth_lo": float(args.centroid_depth_lo),
        "centroid_depth_hi": float(args.centroid_depth_hi),
        "greedy": True,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": int(args.max_new_tokens),
        "tf32_matmul": bool(getattr(getattr(torch.backends, "cuda", object()), "matmul", object()).allow_tf32)
        if hasattr(getattr(torch.backends, "cuda", object()), "matmul") else None,
    }
    if torch.cuda.is_available():
        idx = torch.device(args.device).index
        idx = 0 if idx is None else int(idx)
        report["gpu"] = torch.cuda.get_device_name(idx)
    return report


# =============================================================================
# Data loading
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
            raw = str(item["relation"]).strip().lower()
            if raw not in SYN_REL_MAP:
                raise RuntimeError(f"{labels_path}:{line_no}: unsupported relation={raw!r}")
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
                "relation": SYN_REL_MAP[raw],
                "question_text": SYNTHETIC_PROMPT.format(subject=subject, reference=reference),
            })
    rows.sort(key=lambda r: int(r["sid"]))
    if args.source_max_samples is not None:
        rows = rows[: int(args.source_max_samples)]
    counts = Counter(r["relation"] for r in rows)
    missing = [r for r in SOURCE_RELATIONS if counts[r] == 0]
    if missing:
        raise RuntimeError(f"Synthetic source lacks relations {missing}; counts={dict(counts)}")
    return rows


def load_coco(args: argparse.Namespace) -> List[Dict[str, Any]]:
    records, audit = twoobj.load_records("coco_two", Path(args.data_root), args.target_max_samples)
    prompts = attncent.load_standard_prompts(Path(args.coco_prompt_jsonl))
    result: List[Dict[str, Any]] = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        result.append({
            "sid": sid,
            "dataset": "coco",
            "image_id": str(rec.image_id),
            "image_path": str(rec.image_path),
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "relation": str(rec.relation),
            "question_text": str(p["question_text"]),
        })
    print(f"[COCO] usable={len(result)} | relations={dict(Counter(r['relation'] for r in result))} | audit={len(audit)}")
    return result


def load_controlled_a(args: argparse.Namespace) -> List[Dict[str, Any]]:
    # Keep the repository's established loader and prompt convention.
    old_dataset = getattr(args, "dataset", None)
    args.dataset = "controlled_a"
    try:
        records = causal.load_controlled_a(args)
    finally:
        if old_dataset is None:
            with contextlib.suppress(Exception):
                delattr(args, "dataset")
        else:
            args.dataset = old_dataset
    rows = [dict(r) for r in records]
    for r in rows:
        r["dataset"] = "controlled_a"
        r["question_text"] = target_question(r, "controlled_a")
    if args.target_max_samples is not None:
        rows = rows[: int(args.target_max_samples)]
    print(f"[ControlledA] usable={len(rows)} | relations={dict(Counter(r['relation'] for r in rows))}")
    return rows


def load_vg2_lr(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_path = Path(args.vg_json)
    prompt_path = Path(args.vg_prompt_jsonl)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{data_path} must contain a top-level list")
    prompts = causal.load_vg_standard_prompts(prompt_path)
    if len(data) != len(prompts):
        raise RuntimeError(f"VG data/prompt length mismatch: {len(data)} vs {len(prompts)}")
    old_dataset = getattr(args, "dataset", None)
    args.dataset = "vg2"
    try:
        image_root = causal.resolve_vg_image_root(args, data)
    finally:
        if old_dataset is None:
            with contextlib.suppress(Exception):
                delattr(args, "dataset")
        else:
            args.dataset = old_dataset
    rows: List[Dict[str, Any]] = []
    skipped = Counter()
    for sid, item in enumerate(data):
        if sid not in prompts or not isinstance(item, (list, tuple)) or len(item) < 1:
            skipped["malformed"] += 1
            continue
        p = prompts[sid]
        relation = str(p["relation"])
        if relation not in ("left", "right"):
            skipped["not_lr"] += 1
            continue
        image_id = item[0]
        image_path = causal.find_vg_image(image_root, image_id)
        if image_path is None:
            skipped["missing_image"] += 1
            continue
        rows.append({
            "sid": sid,
            "dataset": "vg2_lr",
            "image_id": str(image_id),
            "image_path": str(image_path),
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "relation": relation,
            "question_text": str(p["question_text"]),
        })
    if args.target_max_samples is not None:
        rows = rows[: int(args.target_max_samples)]
    counts = Counter(r["relation"] for r in rows)
    if not rows or counts["left"] == 0 or counts["right"] == 0:
        raise RuntimeError(f"VG2-LR missing required classes; counts={dict(counts)}")
    print(f"[VG2-LR] usable={len(rows)} | relations={dict(counts)} | skipped={dict(skipped)} | root={image_root}")
    return rows


def load_target(dataset: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    if dataset == "coco":
        return load_coco(args)
    if dataset == "controlled_a":
        return load_controlled_a(args)
    if dataset == "vg2_lr":
        return load_vg2_lr(args)
    raise ValueError(dataset)


def target_labels(dataset: str) -> Tuple[str, ...]:
    if dataset == "coco":
        return ("left", "right", "above", "below")
    if dataset == "controlled_a":
        return ("left", "right", "on", "under")
    if dataset == "vg2_lr":
        return ("left", "right")
    raise ValueError(dataset)


def target_to_source_relation(dataset: str, relation: str) -> str:
    if dataset == "controlled_a":
        return {"left": "left", "right": "right", "on": "above", "under": "below"}[relation]
    return relation


def source_to_target_relation(dataset: str, relation: str) -> Optional[str]:
    if dataset == "controlled_a":
        return {"left": "left", "right": "right", "above": "on", "below": "under"}.get(relation)
    if dataset == "vg2_lr":
        return relation if relation in ("left", "right") else None
    return relation if relation in SOURCE_RELATIONS else None


def target_question(record: Mapping[str, Any], dataset: str) -> str:
    if record.get("question_text") and dataset in ("coco", "vg2_lr"):
        return str(record["question_text"])
    words = target_labels(dataset)
    answer_text = ", ".join(words[:-1]) + f", or {words[-1]}"
    return (
        f"Determine the spatial relation of the {record['subject']} "
        f"to the {record['reference']} in the image. "
        f"Answer with {answer_text}."
    )


def swap_object_mentions(question: str, subject: str, reference: str) -> str:
    if subject == reference:
        raise ValueError("Subject/reference are identical")
    a = "__SUBJECT_SWAP_A_19A7__"
    b = "__REFERENCE_SWAP_B_83D1__"
    if subject in question and reference in question:
        s = question.replace(subject, a)
        s = s.replace(reference, b)
        return s.replace(a, reference).replace(b, subject)
    pa = re.compile(re.escape(subject), flags=re.IGNORECASE)
    pb = re.compile(re.escape(reference), flags=re.IGNORECASE)
    if pa.search(question) is None or pb.search(question) is None:
        raise ValueError(
            f"Could not swap object mentions: subject={subject!r}, reference={reference!r}, question={question!r}"
        )
    s = pa.sub(a, question)
    s = pb.sub(b, s)
    return s.replace(a, reference).replace(b, subject)


def open_record_image(record: Mapping[str, Any]) -> Image.Image:
    return Image.open(record["image_path"]).convert("RGB")


def gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, (v, v, v))


# =============================================================================
# Generic decoder-block hooks for writer extraction / steering
# =============================================================================


def output_hidden(output: Any) -> Tuple[torch.Tensor, Tuple[str, int]]:
    if torch.is_tensor(output):
        return output, ("tensor", 0)
    if isinstance(output, tuple):
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                return item, ("tuple", i)
    if isinstance(output, list):
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                return item, ("list", i)
    raise RuntimeError(f"Cannot extract hidden tensor from output type={type(output)}")


def replace_output_hidden(output: Any, desc: Tuple[str, int], hidden: torch.Tensor) -> Any:
    kind, idx = desc
    if kind == "tensor":
        return hidden
    xs = list(output)
    xs[idx] = hidden
    return tuple(xs) if kind == "tuple" else xs


class CaptureLast:
    def __init__(self, layers: Sequence[Any], selected: Sequence[int]):
        self.states: Dict[int, np.ndarray] = {}
        self.done = {int(l): False for l in selected}
        self.handles = [layers[int(l)].register_forward_hook(self._hook(int(l))) for l in selected]

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if self.done[layer]:
                return output
            hidden, _ = output_hidden(output)
            if hidden.ndim != 3:
                return output
            self.states[layer] = hidden[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            self.done[layer] = True
            return output
        return hook

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SteerLast:
    def __init__(
        self,
        layers: Sequence[Any],
        templates: Mapping[int, Any],
        selected: Sequence[int],
        relation: str,
        scale: float,
    ):
        self.done = {int(l): False for l in selected}
        self.handles = []
        self.layers = layers
        self.templates = templates
        self.relation = relation
        self.scale = float(scale)
        for l in selected:
            self.handles.append(layers[int(l)].register_forward_hook(self._hook(int(l))))

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if self.done[layer]:
                return output
            hidden, desc = output_hidden(output)
            if hidden.ndim != 3:
                return output
            vec = np.asarray(self.templates[layer]["shared"][self.relation], dtype=np.float32)
            v = torch.as_tensor(vec, device=hidden.device, dtype=hidden.dtype) * self.scale
            edited = hidden.clone()
            edited[:, -1, :] += v[None, :]
            self.done[layer] = True
            return replace_output_hidden(output, desc, edited)
        return hook

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Backend abstraction
# =============================================================================


@dataclass
class Backend:
    alias: str
    kind: str
    repo_id: str
    model: Any
    processor: Any
    tokenizer: Any
    layers: Sequence[Any]
    decoder_path: str
    dtype: torch.dtype
    device: torch.device

    def close(self) -> None:
        with contextlib.suppress(Exception):
            del self.model
        cleanup()


def resolve_decoder_layers(model: Any) -> Tuple[Sequence[Any], str]:
    candidates = [
        "language_model.model.layers",
        "model.language_model.model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if len(obj):
                return obj, path
        except Exception:
            pass
    # Qwen repo helper knows additional wrapper paths.
    try:
        return attncent.resolve_decoder_layers(model)
    except Exception as exc:
        raise RuntimeError("Could not resolve language decoder layers") from exc


def ensure_internvl_generation_compat(model: Any, model_id: str) -> Any:
    """Compatibility shim for native InternVL2.5 remote-code LMs on newer Transformers."""
    lm = getattr(model, "language_model", None)
    if lm is None:
        raise RuntimeError("InternVL2.5 model has no language_model attribute")
    original_cls = lm.__class__
    is_internlm2 = "internlm2" in original_cls.__name__.lower()

    if not hasattr(lm, "generate"):
        patched_cls = type(original_cls.__name__ + "WithGenerationMixin", (original_cls, GenerationMixin), {})
        lm.__class__ = patched_cls
        if not hasattr(lm, "generate"):
            raise RuntimeError("Failed to attach GenerationMixin to InternVL language_model")

    try:
        gen_cfg = GenerationConfig.from_pretrained(model_id)
    except Exception:
        gen_cfg = GenerationConfig.from_model_config(lm.config)
    lm.generation_config = gen_cfg
    if hasattr(lm.generation_config, "_from_model_config"):
        lm.generation_config._from_model_config = False
    lm.generation_config.do_sample = False
    lm.generation_config.num_beams = 1

    if is_internlm2:
        for obj in (lm, lm.__class__):
            for attr in ("_supports_cache_class", "_supports_static_cache", "_supports_quantized_cache"):
                with contextlib.suppress(Exception):
                    setattr(obj, attr, False)
        if hasattr(lm.generation_config, "cache_implementation"):
            lm.generation_config.cache_implementation = None
        if hasattr(lm.generation_config, "return_legacy_cache"):
            lm.generation_config.return_legacy_cache = True

    if not hasattr(lm, "prepare_inputs_for_generation"):
        raise RuntimeError(f"{lm.__class__.__name__} lacks prepare_inputs_for_generation")
    return model


def force_eager_config(obj: Any) -> None:
    """Best-effort switch to eager attention without changing checkpoint weights."""
    seen = set()
    queue = [obj, getattr(obj, "config", None), getattr(obj, "language_model", None)]
    while queue:
        cur = queue.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        cfg = getattr(cur, "config", None)
        if cfg is not None and id(cfg) not in seen:
            queue.append(cfg)
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(cur, attr):
                with contextlib.suppress(Exception):
                    setattr(cur, attr, "eager")
        lm = getattr(cur, "language_model", None)
        if lm is not None and id(lm) not in seen:
            queue.append(lm)


def load_backend(alias: str, args: argparse.Namespace) -> Backend:
    if alias not in MODEL_SPECS:
        raise ValueError(alias)
    spec = MODEL_SPECS[alias]
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    repo_id = str(spec["repo_id"])

    print("\n" + "=" * 132)
    print(f"MODEL {alias} | repo={repo_id} | backend={spec['backend']} | dtype={dtype}")
    print(
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"cuda={torch.version.cuda} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        idx = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(idx)
        print(f"GPU={torch.cuda.get_device_name(idx)}")

    if spec["backend"] == "qwen":
        model_cls = getattr(transformers, spec["model_class"], None)
        if model_cls is None:
            raise RuntimeError(
                f"transformers=={transformers.__version__} lacks {spec['model_class']}"
            )
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
        attncent.configure_processor(model, processor)
        force_eager_config(model)
        layers, decoder_path = resolve_decoder_layers(model)
        tokenizer = processor.tokenizer

    else:
        model = AutoModel.from_pretrained(
            repo_id,
            revision=args.revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval().to(device)
        model = ensure_internvl_generation_compat(model, repo_id)
        force_eager_config(model)
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=args.revision,
            trust_remote_code=True,
            use_fast=False,
        )
        processor = None
        layers, decoder_path = resolve_decoder_layers(model)

    if getattr(model, "generation_config", None) is not None:
        with contextlib.suppress(Exception):
            model.generation_config.do_sample = False
        with contextlib.suppress(Exception):
            model.generation_config.num_beams = 1

    print(f"decoder={decoder_path} | n_layers={len(layers)} | device={next(model.parameters()).device}")
    print("=" * 132)
    return Backend(alias, spec["backend"], repo_id, model, processor, tokenizer, layers, decoder_path, dtype, device)


def resolve_actuator_layers(alias: str, n_layers: int) -> List[int]:
    layers = list(MODEL_SPECS[alias]["actuator_layers"])
    bad = [l for l in layers if not (0 <= l < n_layers)]
    if bad:
        raise RuntimeError(f"Actuator preset {bad} invalid for {alias} n_layers={n_layers}")
    return layers


def fixed_relative_layers(n_layers: int, lo: float, hi: float) -> List[int]:
    if n_layers <= 0 or not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"Invalid relative layer range [{lo},{hi}] for n_layers={n_layers}")
    if n_layers == 1:
        return [0]
    layers = [l for l in range(n_layers) if lo <= l / float(n_layers - 1) <= hi]
    if not layers:
        raise RuntimeError(f"Relative range [{lo},{hi}] selected no layers for n_layers={n_layers}")
    return layers


# =============================================================================
# Qwen batch / generation / attention
# =============================================================================


def qwen_build_batch(backend: Backend, image: Image.Image, question: str) -> Dict[str, Any]:
    prompt = attncent.build_prompt(backend.processor, question)
    errors = []
    for fn in (
        lambda: backend.processor(text=[prompt], images=[image], padding=True, return_tensors="pt"),
        lambda: backend.processor(text=prompt, images=image, return_tensors="pt"),
    ):
        try:
            batch = fn()
            break
        except Exception as exc:
            errors.append(exc)
    else:
        raise RuntimeError(f"Qwen processor failed: {errors[-1]}")
    return {k: (v.to(backend.device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def qwen_generate(backend: Backend, image: Image.Image, question: str, max_new_tokens: int) -> str:
    batch = qwen_build_batch(backend, image, question)
    input_len = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = backend.model.generate(
            **batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
        )
    new = out[0, input_len:] if int(out.shape[1]) > input_len else out[0]
    text = backend.tokenizer.decode(new, skip_special_tokens=True).strip()
    del batch, out
    return text


def extract_attentions(outputs: Any) -> Tuple[Any, ...]:
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(getattr(outputs, "language_model_outputs", None), "attentions", None),
        getattr(getattr(outputs, "text_model_output", None), "attentions", None),
    ]
    for x in candidates:
        if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
            return tuple(x)
    raise RuntimeError("Model forward did not return decoder attentions")


def normalize_attn_tensor(x: torch.Tensor, expected_q: int) -> torch.Tensor:
    # Return [heads, query, key].
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise RuntimeError(f"Expected batch=1 attentions, got {tuple(x.shape)}")
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"Unexpected attention tensor shape {tuple(x.shape)}")
    if int(x.shape[-2]) != int(expected_q):
        raise RuntimeError(f"Attention query length {x.shape[-2]} != expected {expected_q}")
    return x


def qwen_object_centroids(
    backend: Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    batch = qwen_build_batch(backend, image, question)
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = attncent.locate_object_spans(
        backend.tokenizer, input_ids, subject, reference
    )
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])
    visual_indices = attncent.resolve_visual_indices(
        backend.model, backend.processor, batch, input_ids
    )
    coords = attncent.visual_coordinates(
        backend.model, batch, len(visual_indices), backend.device
    )
    if coords is None:
        raise RuntimeError(f"Qwen could not construct coordinates for {len(visual_indices)} visual tokens")

    with torch.inference_mode():
        outputs = backend.model(
            **batch,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = extract_attentions(outputs)
    centers: List[np.ndarray] = []
    masses: List[np.ndarray] = []
    vis_idx = torch.as_tensor(visual_indices, device=backend.device, dtype=torch.long)
    for layer in selected_layers:
        a = normalize_attn_tensor(attentions[int(layer)], len(input_ids))
        rows = a[:, [subject_index, reference_index], :]
        visual = rows.index_select(-1, vis_idx)
        mass = visual.sum(dim=-1)
        norm = visual / mass[..., None].clamp_min(1e-12)
        c = torch.einsum("hov,vd->hod", norm.float(), coords.float())
        centers.append(c.detach().cpu().numpy().astype(np.float32))
        masses.append(mass.detach().float().cpu().numpy().astype(np.float32))
    del batch, outputs, attentions
    return np.stack(centers, axis=0), np.stack(masses, axis=0)


# =============================================================================
# Native InternVL2.5 preprocessing / prompt / attention
# =============================================================================


def internvl_build_transform(size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def internvl_closest_ratio(ar: float, ratios: Sequence[Tuple[int, int]], width: int, height: int, size: int) -> Tuple[int, int]:
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in ratios:
        target = ratio[0] / ratio[1]
        diff = abs(ar - target)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * size * size * ratio[0] * ratio[1]:
            best = ratio
    return best


def internvl_dynamic_preprocess(
    image: Image.Image,
    size: int,
    max_num: int,
    use_thumbnail: bool,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    width, height = image.size
    ar = width / height
    ratios = sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    cols, rows = internvl_closest_ratio(ar, ratios, width, height, size)
    target_width, target_height = size * cols, size * rows
    resized = image.resize((target_width, target_height))
    blocks = cols * rows
    tiles: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % cols) * size,
            (i // cols) * size,
            ((i % cols) + 1) * size,
            ((i // cols) + 1) * size,
        )
        tiles.append(resized.crop(box))
    has_thumbnail = bool(use_thumbnail and len(tiles) != 1)
    if has_thumbnail:
        tiles.append(image.resize((size, size)))
    layout = {
        "cols": int(cols),
        "rows": int(rows),
        "num_local_tiles": int(blocks),
        "has_thumbnail": has_thumbnail,
        "num_tiles": len(tiles),
        "original_width": int(width),
        "original_height": int(height),
    }
    return tiles, layout


def internvl_pixels_and_layout(
    backend: Backend,
    image: Image.Image,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tiles, layout = internvl_dynamic_preprocess(
        image,
        size=int(args.internvl_input_size),
        max_num=int(args.internvl_max_num_tiles),
        use_thumbnail=bool(args.internvl_use_thumbnail),
    )
    transform = internvl_build_transform(int(args.internvl_input_size))
    pixels = torch.stack([transform(tile) for tile in tiles]).to(
        device=backend.device, dtype=backend.dtype
    )
    return pixels, layout


def internvl_chat_query(
    backend: Backend,
    question: str,
    num_tiles: int,
) -> Tuple[str, torch.Tensor, torch.Tensor, List[int]]:
    model = backend.model
    tokenizer = backend.tokenizer
    get_conv_template = getattr(getattr(model, "chat", None), "__globals__", {}).get("get_conv_template")
    if get_conv_template is None:
        raise RuntimeError("Could not access InternVL get_conv_template from model.chat")
    template = get_conv_template(model.template)
    if hasattr(model, "system_message"):
        template.system_message = model.system_message
    user_question = "<image>\n" + question
    template.append_message(template.roles[0], user_question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    img_context = "<IMG_CONTEXT>"
    context_id = int(tokenizer.convert_tokens_to_ids(img_context))
    model.img_context_token_id = context_id
    image_tokens = (
        "<img>"
        + img_context * int(model.num_image_token) * int(num_tiles)
        + "</img>"
    )
    query = query.replace("<image>", image_tokens, 1)
    tokenized = tokenizer(query, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(backend.device)
    attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(backend.device)
    ids = input_ids[0].detach().cpu().tolist()
    visual_positions = [i for i, token_id in enumerate(ids) if int(token_id) == context_id]
    return query, input_ids, attention_mask, visual_positions


def token_span_for_phrase(tokenizer: Any, input_ids: Sequence[int], phrase: str, start: int = 0) -> List[int]:
    candidates: List[List[int]] = []
    seen = set()
    for variant in (" " + phrase, phrase, "\n" + phrase, " the " + phrase, "the " + phrase):
        ids = list(tokenizer(variant, add_special_tokens=False).input_ids)
        key = tuple(ids)
        if ids and key not in seen:
            seen.add(key)
            candidates.append(ids)
    matches: List[List[int]] = []
    for ids in candidates:
        n = len(ids)
        for i in range(start, len(input_ids) - n + 1):
            if list(input_ids[i:i+n]) == ids:
                matches.append(list(range(i, i+n)))
    if not matches:
        raise RuntimeError(f"Could not locate phrase tokens for {phrase!r}")
    return max(matches, key=lambda x: x[0])


def internvl_visual_coordinates(
    backend: Backend,
    layout: Mapping[str, Any],
    n_visual: int,
) -> np.ndarray:
    per_tile = int(getattr(backend.model, "num_image_token"))
    if n_visual != per_tile * int(layout["num_tiles"]):
        raise RuntimeError(
            f"InternVL visual-token mismatch: observed={n_visual}, expected={per_tile}*{layout['num_tiles']}"
        )
    grid = int(round(math.sqrt(per_tile)))
    if grid * grid != per_tile:
        raise RuntimeError(
            f"InternVL visual tokens per tile={per_tile} is not a square grid; explicit mapper required"
        )
    cols, rows = int(layout["cols"]), int(layout["rows"])
    num_local = int(layout["num_local_tiles"])
    coords: List[Tuple[float, float]] = []
    for tile_index in range(int(layout["num_tiles"])):
        is_thumb = bool(layout["has_thumbnail"] and tile_index == num_local)
        for iy in range(grid):
            for ix in range(grid):
                lx = (ix + 0.5) / grid
                ly = (iy + 0.5) / grid
                if is_thumb:
                    x, y = lx, ly
                else:
                    tx = tile_index % cols
                    ty = tile_index // cols
                    x = (tx + lx) / cols
                    y = (ty + ly) / rows
                coords.append((float(x), float(y)))
    arr = np.asarray(coords, dtype=np.float32)
    if arr.shape != (n_visual, 2):
        raise RuntimeError(f"InternVL coordinate shape={arr.shape}, expected={(n_visual, 2)}")
    return arr


def internvl_generate(backend: Backend, image: Image.Image, question: str, args: argparse.Namespace) -> str:
    pixels, _layout = internvl_pixels_and_layout(backend, image, args)
    q = "<image>\n" + question
    config = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
    }
    kwargs = dict(history=None, return_history=False)
    try:
        response = backend.model.chat(
            backend.tokenizer,
            pixels,
            q,
            config,
            num_patches_list=[int(pixels.shape[0])],
            verbose=False,
            **kwargs,
        )
    except TypeError:
        response = backend.model.chat(
            backend.tokenizer,
            pixels,
            q,
            config,
            **kwargs,
        )
    del pixels
    return str(response).strip()


def internvl_object_centroids(
    backend: Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    pixels, layout = internvl_pixels_and_layout(backend, image, args)
    _query, input_ids, attention_mask, visual_positions = internvl_chat_query(
        backend, question, int(pixels.shape[0])
    )
    ids = input_ids[0].detach().cpu().tolist()
    search_start = max(visual_positions) + 1 if visual_positions else 0
    subject_span = token_span_for_phrase(backend.tokenizer, ids, subject, start=search_start)
    reference_span = token_span_for_phrase(backend.tokenizer, ids, reference, start=search_start)
    subject_index = int(subject_span[-1])
    reference_index = int(reference_span[-1])
    coords_np = internvl_visual_coordinates(backend, layout, len(visual_positions))
    coords = torch.as_tensor(coords_np, device=backend.device, dtype=torch.float32)

    with torch.inference_mode():
        image_flags = torch.ones(
            (int(pixels.shape[0]), 1),
            device=backend.device,
            dtype=torch.long,
        )
        outputs = backend.model(
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = extract_attentions(outputs)
    vis_idx = torch.as_tensor(visual_positions, device=backend.device, dtype=torch.long)
    centers: List[np.ndarray] = []
    masses: List[np.ndarray] = []
    for layer in selected_layers:
        a = normalize_attn_tensor(attentions[int(layer)], len(ids))
        rows = a[:, [subject_index, reference_index], :]
        visual = rows.index_select(-1, vis_idx)
        mass = visual.sum(dim=-1)
        norm = visual / mass[..., None].clamp_min(1e-12)
        c = torch.einsum("hov,vd->hod", norm.float(), coords)
        centers.append(c.detach().cpu().numpy().astype(np.float32))
        masses.append(mass.detach().float().cpu().numpy().astype(np.float32))
    del pixels, input_ids, attention_mask, image_flags, outputs, attentions
    return np.stack(centers, axis=0), np.stack(masses, axis=0)


# =============================================================================
# Unified generation / centroid reader
# =============================================================================


def generate_text(backend: Backend, image: Image.Image, question: str, args: argparse.Namespace) -> str:
    if backend.kind == "qwen":
        return qwen_generate(backend, image, question, args.max_new_tokens)
    return internvl_generate(backend, image, question, args)


def parse_prediction(text: str, dataset: str) -> Optional[str]:
    s = str(text).lower()
    patterns: List[Tuple[str, str]]
    if dataset == "controlled_a":
        patterns = [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("on", r"\bon\b"),
            ("on", r"\bon top of\b"),
            ("under", r"\bunder(?:neath)?\b"),
            ("under", r"\bbelow\b"),
            ("under", r"\bbeneath\b"),
        ]
    elif dataset == "vg2_lr":
        patterns = [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("other", r"\bfront\b"),
            ("other", r"\bbehind\b"),
            ("other", r"\babove\b"),
            ("other", r"\bbelow\b"),
            ("other", r"\bunder(?:neath)?\b"),
        ]
    else:
        patterns = [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("above", r"\babove\b"),
            ("above", r"\bon top of\b"),
            ("below", r"\bbelow\b"),
            ("below", r"\bunder(?:neath)?\b"),
            ("below", r"\bbeneath\b"),
        ]
    hits = []
    for label, pattern in patterns:
        m = re.search(pattern, s)
        if m:
            hits.append((m.start(), label))
    if not hits:
        return None
    label = min(hits, key=lambda x: x[0])[1]
    return None if label == "other" else label


def object_centroids(
    backend: Backend,
    image: Image.Image,
    question: str,
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    if backend.kind == "qwen":
        return qwen_object_centroids(
            backend, image, question, subject, reference, selected_layers
        )
    return internvl_object_centroids(
        backend, image, question, subject, reference, selected_layers, args
    )


def decode_unit_votes(avg_centroids: np.ndarray, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """avg_centroids: [layers, heads, 2 objects, 2 coordinates]."""
    dx = avg_centroids[..., 0, 0] - avg_centroids[..., 1, 0]
    dy = avg_centroids[..., 0, 1] - avg_centroids[..., 1, 1]
    ax, ay = np.abs(dx), np.abs(dy)
    if dataset == "vg2_lr":
        codes = np.where(dx < 0.0, SOURCE_REL_TO_ID["left"], SOURCE_REL_TO_ID["right"]).astype(np.int8)
        conf = (ax / (ax + ay + 1e-8)).astype(np.float32)
        return codes, conf
    horizontal = ax >= ay
    codes = np.where(
        horizontal,
        np.where(dx < 0.0, SOURCE_REL_TO_ID["left"], SOURCE_REL_TO_ID["right"]),
        np.where(dy < 0.0, SOURCE_REL_TO_ID["above"], SOURCE_REL_TO_ID["below"]),
    ).astype(np.int8)
    conf = (np.abs(ax - ay) / (ax + ay + 1e-8)).astype(np.float32)
    return codes, conf


def aggregate_votes(codes: np.ndarray, confidence: np.ndarray, dataset: str) -> Dict[str, Any]:
    flat_codes = np.asarray(codes).reshape(-1)
    flat_conf = np.asarray(confidence).reshape(-1)
    candidates = ["left", "right"] if dataset == "vg2_lr" else list(SOURCE_RELATIONS)
    counts = {r: int(np.sum(flat_codes == SOURCE_REL_TO_ID[r])) for r in candidates}
    max_count = max(counts.values())
    tied = [r for r in candidates if counts[r] == max_count]
    conf_sums = {
        r: float(flat_conf[flat_codes == SOURCE_REL_TO_ID[r]].sum())
        for r in candidates
    }
    # Deterministic final tie-break: confidence sum, then canonical relation order.
    pred_source = sorted(
        tied,
        key=lambda r: (-conf_sums[r], SOURCE_REL_TO_ID[r]),
    )[0]
    n = max(1, len(flat_codes))
    sorted_counts = sorted(counts.values(), reverse=True)
    vote_margin = (sorted_counts[0] - sorted_counts[1]) / n if len(sorted_counts) > 1 else 1.0
    return {
        "source_pred": pred_source,
        "vote_counts": counts,
        "confidence_sums": conf_sums,
        "n_units": int(len(flat_codes)),
        "vote_fraction": float(counts[pred_source] / n),
        "vote_margin": float(vote_margin),
        "tie": int(len(tied) > 1),
    }


def centroid_predict(
    backend: Backend,
    image: Image.Image,
    record: Mapping[str, Any],
    dataset: str,
    selected_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question = target_question(record, dataset)
    swapped_question = swap_object_mentions(question, str(record["subject"]), str(record["reference"]))

    original_centroids, original_mass = object_centroids(
        backend,
        image,
        question,
        str(record["subject"]),
        str(record["reference"]),
        selected_layers,
        args,
    )
    swapped_centroids, swapped_mass = object_centroids(
        backend,
        image,
        swapped_question,
        str(record["reference"]),
        str(record["subject"]),
        selected_layers,
        args,
    )
    if original_centroids.shape != swapped_centroids.shape:
        raise RuntimeError(
            f"Original/swap centroid shape mismatch: {original_centroids.shape} vs {swapped_centroids.shape}"
        )
    # Swapped query roles are [reference, subject]; align back to [subject, reference].
    swapped_aligned = swapped_centroids[:, :, [1, 0], :]
    avg = 0.5 * (original_centroids + swapped_aligned)
    codes, conf = decode_unit_votes(avg, dataset)
    vote = aggregate_votes(codes, conf, dataset)
    target_pred = source_to_target_relation(dataset, vote["source_pred"])
    return {
        **vote,
        "target_pred": target_pred,
        "mean_original_visual_mass": float(np.mean(original_mass)),
        "mean_swapped_visual_mass": float(np.mean(swapped_mass)),
    }


# =============================================================================
# Synthetic writer fit / cache
# =============================================================================


def capture_source_last(
    backend: Backend,
    image: Image.Image,
    question: str,
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[int, np.ndarray]:
    with CaptureLast(backend.layers, actuator_layers) as cap:
        _ = generate_text(backend, image, question, args)
    missing = [l for l in actuator_layers if l not in cap.states]
    if missing:
        raise RuntimeError(f"Missing late-state captures at layers {missing}")
    return dict(cap.states)


def fit_synthetic_writer(
    backend: Backend,
    source_records: Sequence[Mapping[str, Any]],
    actuator_layers: Sequence[int],
    args: argparse.Namespace,
    model_outdir: Path,
) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
    cache_path = model_outdir / "synthetic_writer.npz"
    if cache_path.exists() and not args.overwrite:
        with np.load(cache_path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            expected = {
                "model": backend.alias,
                "repo_id": backend.repo_id,
                "revision": str(args.revision),
                "actuator_layers": list(map(int, actuator_layers)),
                "gray_value": int(args.gray_value),
                "synthetic_dir": str(Path(args.synthetic_dir)),
                "source_n_requested": int(len(source_records)),
            }
            for k, v in expected.items():
                if metadata.get(k) != v:
                    raise RuntimeError(
                        f"Cached writer mismatch at {k}: cached={metadata.get(k)!r}, expected={v!r}; use --overwrite"
                    )
            templates: Dict[int, Any] = {}
            for layer in actuator_layers:
                rel_mean = {
                    r: np.asarray(data[f"L{layer}_{r}_mean"], dtype=np.float32)
                    for r in SOURCE_RELATIONS
                }
                global_mean = np.asarray(data[f"L{layer}_global"], dtype=np.float32)
                templates[layer] = {
                    "relation_mean": rel_mean,
                    "global": global_mean,
                    "shared": {r: rel_mean[r] - global_mean for r in SOURCE_RELATIONS},
                }
        print(f"[writer] loaded cached synthetic writer: {cache_path}")
        return templates, []

    bags = {
        int(layer): {r: [] for r in SOURCE_RELATIONS}
        for layer in actuator_layers
    }
    errors: List[Dict[str, Any]] = []
    for record in tqdm(source_records, desc=f"source-writer:{backend.alias}"):
        real = gray = None
        try:
            real = open_record_image(record)
            gray = gray_image(real, args.gray_value)
            real_states = capture_source_last(
                backend, real, str(record["question_text"]), actuator_layers, args
            )
            gray_states = capture_source_last(
                backend, gray, str(record["question_text"]), actuator_layers, args
            )
            relation = str(record["relation"])
            for layer in actuator_layers:
                bags[int(layer)][relation].append(
                    (real_states[int(layer)] - gray_states[int(layer)]).astype(np.float32)
                )
        except Exception as exc:
            errors.append({
                "sid": record["sid"],
                "relation": record["relation"],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]),
            })
            tqdm.write(f"[SOURCE WRITER ERROR sid={record['sid']}] {type(exc).__name__}: {exc}")
        finally:
            if real is not None:
                real.close()
            if gray is not None:
                gray.close()
            cleanup()

    templates: Dict[int, Any] = {}
    save_arrays: Dict[str, Any] = {}
    for layer in actuator_layers:
        rel_mean: Dict[str, np.ndarray] = {}
        for relation in SOURCE_RELATIONS:
            values = bags[int(layer)][relation]
            if not values:
                raise RuntimeError(f"No source writer vectors for L{layer}/{relation}")
            rel_mean[relation] = np.stack(values, axis=0).mean(axis=0).astype(np.float32)
        global_mean = np.stack([rel_mean[r] for r in SOURCE_RELATIONS], axis=0).mean(axis=0).astype(np.float32)
        templates[int(layer)] = {
            "relation_mean": rel_mean,
            "global": global_mean,
            "shared": {r: (rel_mean[r] - global_mean).astype(np.float32) for r in SOURCE_RELATIONS},
        }
        save_arrays[f"L{layer}_global"] = global_mean
        for r in SOURCE_RELATIONS:
            save_arrays[f"L{layer}_{r}_mean"] = rel_mean[r]
            save_arrays[f"L{layer}_{r}_shared"] = templates[int(layer)]["shared"][r]

    metadata = {
        "script_version": SCRIPT_VERSION,
        "model": backend.alias,
        "repo_id": backend.repo_id,
        "revision": str(args.revision),
        "actuator_layers": list(map(int, actuator_layers)),
        "gray_value": int(args.gray_value),
        "synthetic_dir": str(Path(args.synthetic_dir)),
        "source_n_requested": int(len(source_records)),
        "source_relation_counts": dict(Counter(r["relation"] for r in source_records)),
        "errors": len(errors),
    }
    save_arrays["metadata_json"] = np.asarray(json.dumps(metadata), dtype=object)
    model_outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **save_arrays)
    write_csv(model_outdir / "synthetic_writer_errors.csv", errors)
    write_json(model_outdir / "synthetic_writer_metadata.json", metadata)
    return templates, errors


# =============================================================================
# Target generation conditions
# =============================================================================


def generate_condition(
    backend: Backend,
    image: Image.Image,
    record: Mapping[str, Any],
    dataset: str,
    args: argparse.Namespace,
    templates: Optional[Mapping[int, Any]] = None,
    actuator_layers: Optional[Sequence[int]] = None,
    writer_relation: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    question = target_question(record, dataset)
    if templates is None:
        text = generate_text(backend, image, question, args)
    else:
        assert actuator_layers is not None and writer_relation is not None
        with SteerLast(
            backend.layers,
            templates,
            actuator_layers,
            writer_relation,
            args.scale,
        ):
            text = generate_text(backend, image, question, args)
    return text, parse_prediction(text, dataset)


def transition(before_ok: bool, after_ok: bool) -> str:
    if (not before_ok) and after_ok:
        return "W2C"
    if before_ok and (not after_ok):
        return "C2W"
    return "C2C" if before_ok else "W2W"


# =============================================================================
# Evaluation: independent coverage for each method
# =============================================================================


def evaluate_dataset(
    backend: Backend,
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    templates: Mapping[int, Any],
    actuator_layers: Sequence[int],
    centroid_layers: Sequence[int],
    args: argparse.Namespace,
    outdir: Path,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    method_rows: Dict[str, List[Dict[str, Any]]] = {
        "baseline": [],
        "oracle": [],
        "centroid_select": [],
        "centroid_final": [],
    }
    errors: List[Dict[str, Any]] = []

    for record in tqdm(records, desc=f"TEST:{dataset}:{backend.alias}"):
        sid = int(record["sid"])
        gt = str(record["relation"])
        image = None
        baseline_pred: Optional[str] = None
        baseline_ok: Optional[bool] = None
        try:
            image = open_record_image(record)

            # Baseline is independent of all other methods.
            try:
                text, pred = generate_condition(backend, image, record, dataset, args)
                baseline_pred = pred
                baseline_ok = bool(pred == gt)
                method_rows["baseline"].append({
                    "sid": sid, "relation": gt, "pred": pred or "",
                    "correct": int(baseline_ok), "text": text,
                })
            except Exception as exc:
                errors.append({
                    "sid": sid, "relation": gt, "method": "baseline",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                tqdm.write(f"[baseline ERROR {dataset} sid={sid}] {type(exc).__name__}: {exc}")
                cleanup()

            # Oracle: target GT only selects the already-frozen synthetic writer.
            try:
                writer_rel = target_to_source_relation(dataset, gt)
                text, pred = generate_condition(
                    backend, image, record, dataset, args,
                    templates=templates,
                    actuator_layers=actuator_layers,
                    writer_relation=writer_rel,
                )
                ok = bool(pred == gt)
                row = {
                    "sid": sid, "relation": gt, "writer_relation": writer_rel,
                    "pred": pred or "", "correct": int(ok), "text": text,
                }
                if baseline_ok is not None:
                    row["baseline_correct"] = int(baseline_ok)
                    row["transition"] = transition(bool(baseline_ok), ok)
                method_rows["oracle"].append(row)
            except Exception as exc:
                errors.append({
                    "sid": sid, "relation": gt, "method": "oracle",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                tqdm.write(f"[oracle ERROR {dataset} sid={sid}] {type(exc).__name__}: {exc}")
                cleanup()

            # Fixed-window centroid selector.  No GT is passed to the reader.
            centroid_result: Optional[Dict[str, Any]] = None
            try:
                centroid_result = centroid_predict(
                    backend, image, record, dataset, centroid_layers, args
                )
                pred = centroid_result["target_pred"]
                ok = bool(pred == gt)
                row = {
                    "sid": sid,
                    "relation": gt,
                    "pred": pred or "",
                    "source_pred": centroid_result["source_pred"],
                    "correct": int(ok),
                    "n_vote_units": centroid_result["n_units"],
                    "vote_fraction": centroid_result["vote_fraction"],
                    "vote_margin": centroid_result["vote_margin"],
                    "tie": centroid_result["tie"],
                    "mean_original_visual_mass": centroid_result["mean_original_visual_mass"],
                    "mean_swapped_visual_mass": centroid_result["mean_swapped_visual_mass"],
                }
                for rel, count in centroid_result["vote_counts"].items():
                    row[f"votes_{rel}"] = int(count)
                for rel, conf in centroid_result["confidence_sums"].items():
                    row[f"confidence_sum_{rel}"] = float(conf)
                method_rows["centroid_select"].append(row)
            except Exception as exc:
                errors.append({
                    "sid": sid, "relation": gt, "method": "centroid_select",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]),
                })
                tqdm.write(f"[centroid ERROR {dataset} sid={sid}] {type(exc).__name__}: {exc}")
                cleanup()

            # Centroid -> frozen synthetic writer.
            if centroid_result is not None and centroid_result.get("source_pred") in SOURCE_RELATIONS:
                try:
                    writer_rel = str(centroid_result["source_pred"])
                    text, pred = generate_condition(
                        backend, image, record, dataset, args,
                        templates=templates,
                        actuator_layers=actuator_layers,
                        writer_relation=writer_rel,
                    )
                    ok = bool(pred == gt)
                    row = {
                        "sid": sid, "relation": gt,
                        "selector_pred": centroid_result.get("target_pred") or "",
                        "writer_relation": writer_rel,
                        "pred": pred or "", "correct": int(ok), "text": text,
                    }
                    if baseline_ok is not None:
                        row["baseline_correct"] = int(baseline_ok)
                        row["transition"] = transition(bool(baseline_ok), ok)
                    method_rows["centroid_final"].append(row)
                except Exception as exc:
                    errors.append({
                        "sid": sid, "relation": gt, "method": "centroid_final",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    tqdm.write(f"[centroid-final ERROR {dataset} sid={sid}] {type(exc).__name__}: {exc}")
                    cleanup()

        finally:
            if image is not None:
                image.close()
            cleanup()

    expected = len(records)
    summary: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "model": backend.alias,
        "repo_id": backend.repo_id,
        "dataset": dataset,
        "expected_n": expected,
        "actuator_layers": list(map(int, actuator_layers)),
        "centroid_relative_depth": [float(args.centroid_depth_lo), float(args.centroid_depth_hi)],
        "centroid_layers": list(map(int, centroid_layers)),
        "centroid_aggregation": "equal vote over every (layer,head); deterministic confidence-only tie break",
        "centroid_gt_selection": False,
        "seed": int(args.seed),
        "scale": float(args.scale),
    }

    for method, rows in method_rows.items():
        coverage = len(rows) / max(1, expected)
        acc = safe_mean(r.get("correct") for r in rows)
        summary[f"{method}_n"] = len(rows)
        summary[f"{method}_coverage"] = coverage
        summary[f"{method}_accuracy"] = acc
        write_csv(outdir / f"{method}_details.csv", rows)

    # Transition counts for edited generation conditions.
    for method in ("oracle", "centroid_final"):
        counts = Counter(r.get("transition") for r in method_rows[method] if r.get("transition"))
        for key in ("W2C", "C2W", "C2C", "W2W"):
            summary[f"{method}_{key}"] = int(counts[key])

    write_csv(outdir / "errors.csv", errors)
    summary["errors"] = len(errors)

    per_relation = []
    for relation in target_labels(dataset):
        row: Dict[str, Any] = {"relation": relation}
        for method, rows in method_rows.items():
            subset = [r for r in rows if r.get("relation") == relation]
            row[f"{method}_n"] = len(subset)
            row[f"{method}_accuracy"] = safe_mean(r.get("correct") for r in subset)
        per_relation.append(row)
    write_csv(outdir / "per_relation.csv", per_relation)
    write_json(outdir / "summary.json", summary)

    print("\n" + "=" * 160)
    print(
        f"{backend.alias} | {dataset} | expected={expected} | "
        f"centroid rel-depth=[{args.centroid_depth_lo:.2f},{args.centroid_depth_hi:.2f}] "
        f"layers={list(centroid_layers)}"
    )
    print("=" * 160)
    for method in ("baseline", "oracle", "centroid_select", "centroid_final"):
        print(
            f"{method:16s}: acc={summary[f'{method}_accuracy']:.4f} | "
            f"N={summary[f'{method}_n']}/{expected} | "
            f"coverage={summary[f'{method}_coverage']:.3f}"
        )
    print("=" * 160)

    low = [
        method for method in method_rows
        if summary[f"{method}_coverage"] < float(args.min_success_rate)
    ]
    if low:
        summary["coverage_failure_methods"] = low
        write_json(outdir / "summary.json", summary)
        raise RuntimeError(
            f"{backend.alias}/{dataset}: method coverage below {args.min_success_rate}: "
            + ", ".join(
                f"{m}={summary[f'{m}_n']}/{expected}={summary[f'{m}_coverage']:.3f}"
                for m in low
            )
        )
    return summary


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.centroid_depth_lo <= args.centroid_depth_hi <= 1.0):
        raise ValueError("Require 0 <= centroid-depth-lo <= centroid-depth-hi <= 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    models = parse_name_list(args.models, DEFAULT_MODEL_ORDER)
    datasets = parse_name_list(args.datasets, DEFAULT_DATASET_ORDER)
    seed_all(args.seed)

    output_root = Path(args.output_dir)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "environment.json", environment_report(args))
    write_json(output_root / "args.json", vars(args))

    source_records = load_synthetic(args)
    print("\n" + "=" * 160)
    print(
        f"SOURCE synthetic={args.synthetic_dir} | N={len(source_records)} | "
        f"relations={dict(Counter(r['relation'] for r in source_records))}"
    )
    print(f"prompt={SYNTHETIC_PROMPT}")
    print("=" * 160)

    # Load target metadata once; model-specific preprocessing happens later.
    target_data: Dict[str, List[Dict[str, Any]]] = {}
    for dataset in datasets:
        target_data[dataset] = load_target(dataset, args)
        if not target_data[dataset]:
            raise RuntimeError(f"No usable target records for {dataset}")

    all_summaries: List[Dict[str, Any]] = []
    run_errors: List[Dict[str, Any]] = []

    for model_alias in models:
        seed_all(args.seed)
        backend: Optional[Backend] = None
        try:
            backend = load_backend(model_alias, args)
            n_layers = len(backend.layers)
            actuator_layers = resolve_actuator_layers(model_alias, n_layers)
            centroid_layers = fixed_relative_layers(
                n_layers,
                float(args.centroid_depth_lo),
                float(args.centroid_depth_hi),
            )
            model_outdir = output_root / model_alias
            model_outdir.mkdir(parents=True, exist_ok=True)
            model_meta = {
                "model": model_alias,
                "repo_id": backend.repo_id,
                "backend": backend.kind,
                "decoder_path": backend.decoder_path,
                "n_layers": n_layers,
                "actuator_layers": actuator_layers,
                "centroid_relative_depth": [args.centroid_depth_lo, args.centroid_depth_hi],
                "centroid_layers": centroid_layers,
                "centroid_head_policy": "ALL query heads, one equal vote each",
                "centroid_cross_layer_policy": "ALL fixed-window layers, one equal vote per head per layer",
                "gt_based_centroid_selection": False,
            }
            write_json(model_outdir / "model_config.json", model_meta)

            print("\n" + "=" * 160)
            print("SOURCE / READER CONFIG")
            print("=" * 160)
            print(f"model={model_alias} repo={backend.repo_id}")
            print(f"late writer layers={actuator_layers}")
            print(
                f"centroid relative depth=[{args.centroid_depth_lo:.2f},{args.centroid_depth_hi:.2f}] "
                f"-> layers={centroid_layers}"
            )
            print("centroid heads=ALL | aggregation=equal vote over all (layer,head) | GT selection=NONE")
            print("=" * 160)

            templates, source_errors = fit_synthetic_writer(
                backend,
                source_records,
                actuator_layers,
                args,
                model_outdir,
            )
            if source_errors:
                rate = 1.0 - len(source_errors) / max(1, len(source_records))
                if rate < float(args.min_success_rate):
                    raise RuntimeError(
                        f"Synthetic writer success rate {rate:.3f} below {args.min_success_rate}"
                    )

            for dataset in datasets:
                seed_all(args.seed)
                try:
                    summary = evaluate_dataset(
                        backend,
                        dataset,
                        target_data[dataset],
                        templates,
                        actuator_layers,
                        centroid_layers,
                        args,
                        model_outdir / dataset,
                    )
                    all_summaries.append(summary)
                    write_csv(output_root / "summary_all.csv", all_summaries)
                except Exception as exc:
                    row = {
                        "model": model_alias,
                        "dataset": dataset,
                        "stage": "evaluate_dataset",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-12:]),
                    }
                    run_errors.append(row)
                    write_csv(output_root / "run_errors.csv", run_errors)
                    print(f"[RUN ERROR] {model_alias}/{dataset}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise

        except Exception as exc:
            row = {
                "model": model_alias,
                "dataset": "*",
                "stage": "model",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-12:]),
            }
            run_errors.append(row)
            write_csv(output_root / "run_errors.csv", run_errors)
            print(f"[MODEL ERROR] {model_alias}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if backend is not None:
                with contextlib.suppress(Exception):
                    backend.close()
                del backend
            cleanup()

    write_csv(output_root / "summary_all.csv", all_summaries)
    write_csv(output_root / "run_errors.csv", run_errors)

    print("\n" + "=" * 160)
    print("FINAL SUMMARY")
    print("=" * 160)
    if not all_summaries:
        print("No valid dataset summaries were produced.")
    else:
        for row in all_summaries:
            print(
                f"{row['model']:12s} | {row['dataset']:12s} | "
                f"base={row['baseline_accuracy']:.4f} | "
                f"oracle={row['oracle_accuracy']:.4f} | "
                f"centroid_sel={row['centroid_select_accuracy']:.4f} | "
                f"centroid_final={row['centroid_final_accuracy']:.4f} | "
                f"coverage={row['centroid_select_coverage']:.3f}"
            )
    if run_errors:
        print(f"run_errors={len(run_errors)} -> {output_root / 'run_errors.csv'}")
    print("=" * 160)


if __name__ == "__main__":
    main()
