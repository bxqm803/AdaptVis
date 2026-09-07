#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_llava_adaptvis_lasttoken_success_v1.py

Mechanistic analysis of a SUCCESSFULLY REPRODUCED LLaVA-1.5 AdaptVis run.

Target repository:
    https://github.com/bxqm803/AdaptVis/tree/llava16

Core rule
=========
Do not analyze an assumed intervention.

For every sample this script first executes the repository's original
generation policy and verifies the actual accuracy:

    SCAL NEUTRAL:
        same custom LlavaForConditionalGenerationScal
        RMSNorm epsilon = 1e-6
        first generation with weight=1.0

    EXACT ADAPTVIS:
        same model, same epsilon
        confidence = round(max softmax(first-step logits), 2)
        if confidence < 0.3:
            weight = 0.5
        else:
            weight = 1.2
        regenerate exactly as repository AdaptVis

Then, using the SAME dynamically selected weight for that sample, run a
full-prompt forward under neutral and AdaptVis conditions and compare the
LAST TOKEN at every decoder layer.

Primary last-token quantities
=============================
For decoder layer l:

    h0_l = neutral last-token hidden state
    h1_l = AdaptVis last-token hidden state
    dh_l = h1_l - h0_l

Report:
    ||h0_l||
    ||h1_l||
    ||dh_l||
    ||dh_l|| / ||h0_l||
    cosine(h0_l, h1_l)
    cosine(dh_l, dh_{l-1})

Direct attention mechanism
==========================
The branch's output_attentions=True returns the FINAL post-softmax attention
probabilities actually used for value aggregation.

Using the exact merged image mask stored by the custom LLaVA model:

    visual_mass_l
      = mean_heads sum_{j in image tokens} A_l[last_query, j]

Report neutral and AdaptVis visual mass and their difference.

Spatial direction readout
=========================
Repeated stratified TRAIN/TEST direction prototypes for:

    neutral_last
    adaptvis_last
    adaptvis_delta = h1 - h0

This asks whether the last-token CHANGE induced by successful AdaptVis becomes
relation-specific over depth.

Generation-transition groups
============================
Every sample is assigned using SAME custom Scal model:

    W2C : neutral wrong -> AdaptVis correct
    C2W : neutral correct -> AdaptVis wrong
    C2C : correct -> correct
    W2W : wrong -> wrong

Layer-wise geometry/attention is also aggregated separately for these groups.

Dynamic-weight groups
=====================
Because original AdaptVis has two qualitatively different interventions:

    selected weight = 0.5
    selected weight = 1.2

the script reports them separately as well.

Optional HF baseline
====================
Pass --run-hf-base to additionally reproduce the standard repository
LLaVA-1.5 baseline. Hidden-state comparisons are NOT made between HF baseline
and custom Scal because their language-model implementations differ.

Environment
===========
Exact generation reproduction requires the branch-pinned environment:
    transformers == 4.39.1
    tokenizers   == 0.15.2

Example
=======
CUDA_VISIBLE_DEVICES=0 python analyze_llava_adaptvis_lasttoken_success_v1.py \
  --dataset COCO_QA_two_obj \
  --option four \
  --weight1 0.5 \
  --weight2 1.2 \
  --threshold 0.3 \
  --rms-eps 1e-6 \
  --repeats 5 \
  --save-vectors \
  --output-dir output/llava_adaptvis_lasttoken_success_v1

Include standard HF baseline accuracy:
CUDA_VISIBLE_DEVICES=0 python analyze_llava_adaptvis_lasttoken_success_v1.py \
  --dataset COCO_QA_two_obj \
  --option four \
  --run-hf-base \
  --save-vectors \
  --output-dir output/llava_adaptvis_lasttoken_success_with_hf_v1

Smoke test:
CUDA_VISIBLE_DEVICES=0 python analyze_llava_adaptvis_lasttoken_success_v1.py \
  --dataset COCO_QA_two_obj \
  --max-samples 20 \
  --repeats 2 \
  --output-dir output/llava_adaptvis_lasttoken_success_smoke
"""

from __future__ import annotations

# Exact original default variant before importing custom attention code.
import os
os.environ["ADAPTVIS_ATTENTION_VARIANT"] = "mul_img"
os.environ.pop("ZERO_SHRINK_VARIANT", None)
os.environ.pop("ZERO_SHRINK_A", None)
os.environ.pop("ZERO_SHRINK_LAMBDA", None)
os.environ["SAVE_ATTN"] = "False"

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import tokenizers
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_zoo import get_dataset
from misc import seed_all
from model_zoo import get_model
from model_zoo.llava15 import (
    _decode_generated,
    _is_correct,
    _norm_gold,
    change_greedy_to_add_weight,
)


RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12

EXPECTED_TRANSFORMERS = "4.39.1"
EXPECTED_TOKENIZERS = "0.15.2"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--dataset",
        default="COCO_QA_two_obj",
        choices=[
            "Controlled_Images_A",
            "Controlled_Images_B",
            "COCO_QA_one_obj",
            "COCO_QA_two_obj",
            "VG_QA_one_obj",
            "VG_QA_two_obj",
        ],
    )
    p.add_argument(
        "--option",
        default="four",
        choices=["two", "four", "six"],
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--root-dir", default="data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--weight1", type=float, default=0.5)
    p.add_argument("--weight2", type=float, default=1.2)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--rms-eps", type=float, default=1e-6)

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Original repository uses 100.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    p.add_argument(
        "--probe-layers",
        default="all",
        help="'all', 'auto', range like 0-12, or comma list.",
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--repeats", type=int, default=5)

    p.add_argument(
        "--run-hf-base",
        action="store_true",
        help="Also evaluate standard repository HF LLaVA baseline.",
    )

    p.add_argument(
        "--adaptvis-max-layers",
        type=int,
        default=32,
        help=(
            "32 = exact original full AdaptVis. "
            "Set 4 to test only L0-L3 while keeping dynamic policy unchanged."
        ),
    )

    p.add_argument(
        "--require-improvement",
        action="store_true",
        help=(
            "Raise an error if AdaptVis accuracy is not higher than "
            "same-model Scal neutral accuracy."
        ),
    )

    p.add_argument("--save-vectors", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--ignore-version-check",
        action="store_true",
    )

    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(values: Iterable[float]) -> float:
    vals = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values: Iterable[float]) -> float:
    vals = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.std(vals)) if vals else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= EPS or nb <= EPS:
        return float("nan")

    return float(
        np.dot(a, b) / (na * nb)
    )


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )
    return x / np.maximum(denom, EPS)


def normalize_relation(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z]+", " ", text)
    toks = text.split()

    for tok in toks[:12]:
        if tok in REL_TO_ID:
            return tok

    if " left " in f" {text} ":
        return "left"
    if " right " in f" {text} ":
        return "right"
    if "above" in toks or "top" in toks:
        return "above"
    if (
        "below" in toks
        or "under" in toks
        or "underneath" in toks
        or "bottom" in toks
    ):
        return "below"

    return None


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    raw = str(text).strip().lower()

    if raw == "all":
        return list(range(n_layers))

    if raw == "auto":
        vals = [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 10, 12, 14, 16, 18, 20,
            22, 24, 26, 28, 30,
            n_layers - 1,
        ]
        return sorted({
            x for x in vals
            if 0 <= x < n_layers
        })

    result: List[int] = []

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            result.extend(range(a, b + step, step))
        else:
            result.append(int(part))

    result = list(dict.fromkeys(result))

    bad = [
        x
        for x in result
        if not (0 <= x < n_layers)
    ]

    if bad:
        raise ValueError(
            f"Invalid layers={bad}; valid 0..{n_layers-1}"
        )

    if not result:
        raise ValueError("No probe layers.")

    return result


def check_versions(args: argparse.Namespace) -> None:
    print("\n" + "=" * 116)
    print("ENVIRONMENT")
    print("=" * 116)
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"tokenizers={tokenizers.__version__}")
    print(
        f"expected transformers={EXPECTED_TRANSFORMERS}, "
        f"tokenizers={EXPECTED_TOKENIZERS}"
    )

    exact = (
        transformers.__version__ == EXPECTED_TRANSFORMERS
        and tokenizers.__version__ == EXPECTED_TOKENIZERS
    )

    print(f"exact_repo_generation_env={exact}")
    print("=" * 116)

    if (
        not exact
        and not args.ignore_version_check
    ):
        raise RuntimeError(
            "Exact reproduction requires the branch-pinned environment "
            f"(transformers={EXPECTED_TRANSFORMERS}, "
            f"tokenizers={EXPECTED_TOKENIZERS})."
        )


# =============================================================================
# Prompt / dataset
# =============================================================================

def read_prompt_answer_file(
    dataset_name: str,
    option: str,
) -> Tuple[List[str], List[Any]]:
    path = Path(
        f"prompts/{dataset_name}_with_answer_{option}_options.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    prompts: List[str] = []
    answers: List[Any] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(str(row["question"]))
            answers.append(row["answer"])

    return prompts, answers


def make_loader(
    dataset: Any,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=None,
    )


def iter_repo_samples(
    dataset: Any,
    prompt_list: Sequence[str],
    answer_list: Sequence[Any],
    *,
    num_workers: int,
    max_samples: Optional[int],
):
    """
    Preserve repository's image_options traversal/order.
    """
    loader = make_loader(
        dataset,
        num_workers,
    )

    sid = 0

    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                if sid >= len(prompt_list):
                    return

                if (
                    max_samples is not None
                    and sid >= int(max_samples)
                ):
                    return

                yield (
                    sid,
                    image,
                    prompt_list[sid],
                    _norm_gold(answer_list[sid]),
                )

                sid += 1


# =============================================================================
# RMSNorm and layer scope
# =============================================================================

def set_rms_eps(
    model: Any,
    eps: float,
) -> int:
    eps = float(eps)
    changed = 0

    for _, module in model.named_modules():
        cls = module.__class__.__name__.lower()

        if "rmsnorm" not in cls:
            continue

        touched = False

        if hasattr(module, "variance_epsilon"):
            module.variance_epsilon = eps
            touched = True

        if hasattr(module, "eps"):
            module.eps = eps
            touched = True

        changed += int(touched)

    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "language_model", None), "config", None),
    ]

    for cfg in configs:
        if cfg is not None and hasattr(cfg, "rms_norm_eps"):
            cfg.rms_norm_eps = eps

    if changed == 0:
        raise RuntimeError("No RMSNorm modules found.")

    return changed


def unique_rms_eps(model: Any) -> List[float]:
    vals = []

    for _, module in model.named_modules():
        cls = module.__class__.__name__.lower()

        if "rmsnorm" not in cls:
            continue

        if hasattr(module, "variance_epsilon"):
            vals.append(float(module.variance_epsilon))
        elif hasattr(module, "eps"):
            vals.append(float(module.eps))

    return sorted(set(vals))


class RestrictAdaptVisLayers:
    """
    Keep repository AdaptVis unchanged except disable weight outside
    idx < max_layers.

    max_layers=32 reproduces original scope exactly and needs no hooks.
    """

    def __init__(
        self,
        scal_model: Any,
        max_layers: int,
    ):
        self.max_layers = int(max_layers)
        self.handles = []

        if self.max_layers >= 32:
            return

        layers = (
            scal_model
            .language_model
            .model
            .layers
        )

        for layer in layers:
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._hook,
                    with_kwargs=True,
                )
            )

    def _hook(self, module, args, kwargs):
        idx = kwargs.get("idx", None)

        if (
            idx is not None
            and int(idx) >= self.max_layers
        ):
            kwargs = dict(kwargs)
            kwargs["weight"] = None

        return args, kwargs

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()

        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Exact preprocessing / generation
# =============================================================================

def preprocess_exact(
    wrapper: Any,
    prompt: str,
    image: Any,
):
    return wrapper.processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=77,
    ).to(wrapper.device)


def repo_keys(
    single_input: Mapping[str, torch.Tensor],
):
    return [
        torch.where(
            input_id == 32001,
            1,
            0,
        )
        for input_id
        in single_input["input_ids"]
    ]


def decode_repo(
    wrapper: Any,
    output: Any,
    single_input: Mapping[str, torch.Tensor],
) -> str:
    return _decode_generated(
        wrapper.processor,
        output,
        len(single_input["input_ids"][-1]),
    )


def neutral_and_adaptvis_generate(
    wrapper: Any,
    single_input: Mapping[str, torch.Tensor],
    *,
    weight1: float,
    weight2: float,
    threshold: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Exact repository AdaptVis two-pass policy.

    The first pass with weight=1.0 is also our same-model neutral generation.
    """
    change_greedy_to_add_weight()

    neutral_output = wrapper.model.generate(
        **single_input,
        weight=1.0,
        max_new_tokens=int(max_new_tokens),
        output_scores=True,
        return_dict_in_generate=True,
    )

    neutral_text = decode_repo(
        wrapper,
        neutral_output,
        single_input,
    )

    confidence = np.round(
        float(
            torch.softmax(
                neutral_output["scores"][0],
                dim=-1,
            )[0].max()
        ),
        2,
    )

    selected_weight = (
        float(weight1)
        if confidence < float(threshold)
        else float(weight2)
    )

    keys = repo_keys(single_input)

    adapt_output = wrapper.model.generate(
        **single_input,
        keys=keys,
        weight=selected_weight,
        max_new_tokens=int(max_new_tokens),
        output_scores=True,
        return_dict_in_generate=True,
    )

    adapt_text = decode_repo(
        wrapper,
        adapt_output,
        single_input,
    )

    return {
        "neutral_text": neutral_text,
        "adapt_text": adapt_text,
        "confidence": float(confidence),
        "selected_weight": selected_weight,
        "keys": keys,
    }


# =============================================================================
# Hidden-state + attention extraction
# =============================================================================

def last_valid_merged_index(
    model: Any,
) -> int:
    mask = getattr(
        model,
        "_adaptvis_last_attention_mask",
        None,
    )

    if mask is None:
        raise RuntimeError(
            "Custom model did not expose _adaptvis_last_attention_mask."
        )

    if mask.ndim != 2 or int(mask.shape[0]) != 1:
        raise RuntimeError(
            f"Unexpected merged attention mask shape={tuple(mask.shape)}"
        )

    valid = torch.where(
        mask[0].bool()
    )[0]

    if len(valid) == 0:
        raise RuntimeError(
            "Merged attention mask has no valid tokens."
        )

    return int(valid[-1].item())


def exact_image_mask(
    model: Any,
) -> torch.Tensor:
    mask = getattr(
        model,
        "_adaptvis_last_image_id",
        None,
    )

    if mask is None:
        raise RuntimeError(
            "Custom model did not expose _adaptvis_last_image_id."
        )

    if mask.ndim != 2 or int(mask.shape[0]) != 1:
        raise RuntimeError(
            f"Unexpected image mask shape={tuple(mask.shape)}"
        )

    return mask[0].bool()


def extract_prompt_forward(
    wrapper: Any,
    single_input: Mapping[str, torch.Tensor],
    *,
    weight: float,
    keys: Optional[Any],
    probe_layers: Sequence[int],
) -> Dict[str, Any]:
    """
    Full prompt forward corresponding to generation's first step.
    """
    kwargs: Dict[str, Any] = {
        **single_input,
        "weight": float(weight),
        "output_hidden_states": True,
        "output_attentions": True,
        "use_cache": False,
        "return_dict": True,
    }

    if keys is not None:
        kwargs["keys"] = keys

    with torch.inference_mode():
        outputs = wrapper.model(
            **kwargs
        )

    merged_last_idx = last_valid_merged_index(
        wrapper.model
    )

    image_mask = exact_image_mask(
        wrapper.model
    ).to(wrapper.device)

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions

    if hidden_states is None:
        raise RuntimeError(
            "No hidden_states returned."
        )

    if attentions is None:
        raise RuntimeError(
            "No attentions returned."
        )

    last_vectors: List[np.ndarray] = []
    visual_masses: List[float] = []

    for layer in probe_layers:
        state_idx = int(layer) + 1

        if state_idx >= len(hidden_states):
            raise RuntimeError(
                f"L{layer}: hidden_states len={len(hidden_states)}"
            )

        h = hidden_states[state_idx]

        last_vectors.append(
            h[
                0,
                merged_last_idx,
                :
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        attn = attentions[int(layer)]

        if attn.ndim != 4:
            raise RuntimeError(
                f"L{layer}: attention shape={tuple(attn.shape)}"
            )

        q_idx = merged_last_idx

        # [heads, kv]
        last_attn = attn[
            0,
            :,
            q_idx,
            :
        ]

        if int(last_attn.shape[-1]) != int(image_mask.numel()):
            raise RuntimeError(
                f"L{layer}: attention kv={last_attn.shape[-1]} "
                f"!= image mask={image_mask.numel()}"
            )

        mass_per_head = last_attn[
            :,
            image_mask,
        ].sum(dim=-1)

        visual_masses.append(
            float(
                mass_per_head
                .float()
                .mean()
                .item()
            )
        )

    return {
        "last_vectors": np.stack(
            last_vectors,
            axis=0,
        ).astype(np.float32),
        "visual_mass": np.asarray(
            visual_masses,
            dtype=np.float32,
        ),
        "merged_last_idx": merged_last_idx,
        "n_image_tokens": int(
            image_mask.sum().item()
        ),
    }


# =============================================================================
# Direction probe
# =============================================================================

def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    train: List[int] = []
    test: List[int] = []

    for relation in RELATIONS:
        ids = np.flatnonzero(
            labels == relation
        ).tolist()

        rng.shuffle(ids)

        if len(ids) < 2:
            raise RuntimeError(
                f"Need >=2 samples for {relation}"
            )

        n_train = int(
            round(len(ids) * train_ratio)
        )

        n_train = max(
            1,
            min(n_train, len(ids) - 1),
        )

        train.extend(ids[:n_train])
        test.extend(ids[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)

    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def fit_codebook(
    X: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)

    center = X.mean(axis=0)
    Xc = X - center

    directions = []

    for relation in RELATIONS:
        mask = labels == relation

        d = Xc[mask].mean(axis=0)

        d = d / max(
            float(np.linalg.norm(d)),
            EPS,
        )

        directions.append(d)

    return (
        center.astype(np.float32),
        np.stack(
            directions,
            axis=0,
        ).astype(np.float32),
    )


def eval_probe(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    center, dirs = fit_codebook(
        X[train_idx],
        labels[train_idx],
    )

    Xt = normalize_rows(
        X[test_idx] - center
    )

    scores = Xt @ dirs.T
    pred = np.argmax(scores, axis=1)

    gt = np.asarray(
        [
            REL_TO_ID[str(x)]
            for x in labels[test_idx]
        ],
        dtype=np.int64,
    )

    return float(
        np.mean(pred == gt)
    )


def run_direction_probes(
    neutral: np.ndarray,
    adapted: np.ndarray,
    labels: np.ndarray,
    layers: Sequence[int],
    *,
    train_ratio: float,
    repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    delta = (
        adapted.astype(np.float32)
        - neutral.astype(np.float32)
    )

    reps = {
        "neutral_last": neutral,
        "adaptvis_last": adapted,
        "adaptvis_delta": delta,
    }

    raw_rows: List[Dict[str, Any]] = []

    for rep in range(repeats):
        tr, te = stratified_split(
            labels,
            train_ratio,
            seed + rep,
        )

        for name, tensor in reps.items():
            for li, layer in enumerate(layers):
                acc = eval_probe(
                    tensor[:, li, :].astype(np.float32),
                    labels,
                    tr,
                    te,
                )

                raw_rows.append({
                    "repeat": rep,
                    "representation": name,
                    "layer": layer,
                    "accuracy": acc,
                })

    summary: List[Dict[str, Any]] = []

    for name in reps:
        for layer in layers:
            rows = [
                r for r in raw_rows
                if (
                    r["representation"] == name
                    and int(r["layer"]) == int(layer)
                )
            ]

            summary.append({
                "representation": name,
                "layer": layer,
                "accuracy_mean": safe_mean(
                    r["accuracy"]
                    for r in rows
                ),
                "accuracy_std": safe_std(
                    r["accuracy"]
                    for r in rows
                ),
            })

    return summary


# =============================================================================
# Optional standard HF baseline generation
# =============================================================================

def run_hf_baseline(
    *,
    dataset: Any,
    prompts: Sequence[str],
    answers: Sequence[Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    wrapper, _ = get_model(
        "llava1.5",
        args.device,
        method="base",
        root_dir=args.root_dir,
    )

    if not hasattr(wrapper.model, "generate"):
        raise RuntimeError(
            "HF baseline model has no generate()."
        )

    correct = 0
    n = 0

    for sid, image, prompt, gold in tqdm(
        iter_repo_samples(
            dataset,
            prompts,
            answers,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        ),
        desc="HF baseline generation",
    ):
        single_input = preprocess_exact(
            wrapper,
            prompt,
            image,
        )

        with torch.inference_mode():
            output = wrapper.model.generate(
                **single_input,
                max_new_tokens=int(args.max_new_tokens),
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen = decode_repo(
            wrapper,
            output,
            single_input,
        )

        correct += int(
            _is_correct(gold, gen)
        )

        n += 1

        del single_input
        del output
        cleanup()

    acc = (
        float(correct / n)
        if n
        else float("nan")
    )

    del wrapper
    cleanup()

    return {
        "N": n,
        "accuracy": acc,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    check_versions(args)

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(
            "--train-ratio must be in (0,1)."
        )

    if not (1 <= args.adaptvis_max_layers <= 32):
        raise ValueError(
            "--adaptvis-max-layers must be 1..32."
        )

    seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)

    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompts, answers = read_prompt_answer_file(
        args.dataset,
        args.option,
    )

    dataset = get_dataset(
        args.dataset,
        image_preprocess=None,
        download=False,
    )

    hf_result = None

    if args.run_hf_base:
        hf_result = run_hf_baseline(
            dataset=dataset,
            prompts=prompts,
            answers=answers,
            args=args,
        )

        print(
            f"\nHF BASELINE: N={hf_result['N']} "
            f"acc={hf_result['accuracy']:.4f}"
        )

    # ---------------------------------------------------------------------
    # Load EXACT custom Scal model used by AdaptVis.
    # ---------------------------------------------------------------------
    wrapper, _ = get_model(
        "llava1.5",
        args.device,
        method="adapt_vis",
        root_dir=args.root_dir,
    )

    if not hasattr(wrapper.model, "generate"):
        raise RuntimeError(
            "Custom Scal model has no .generate(). "
            "Use the branch-pinned transformers==4.39.1 environment."
        )

    n_rms = set_rms_eps(
        wrapper.model,
        args.rms_eps,
    )

    layers_module = (
        wrapper.model
        .language_model
        .model
        .layers
    )

    n_layers = len(layers_module)

    probe_layers = parse_layer_spec(
        args.probe_layers,
        n_layers,
    )

    print("\n" + "=" * 150)
    print("SUCCESS-FIRST ADAPTVIS LAST-TOKEN ANALYSIS")
    print("=" * 150)

    print(
        f"dataset={args.dataset} | prompts={len(prompts)}"
    )

    print(
        f"Scal RMSNorm eps={unique_rms_eps(wrapper.model)} "
        f"(modules={n_rms})"
    )

    print(
        f"dynamic policy: confidence<{args.threshold} -> {args.weight1}, "
        f"else -> {args.weight2}"
    )

    print(
        f"AdaptVis layer scope: L0-L{args.adaptvis_max_layers-1}"
    )

    print(
        f"probe_layers={probe_layers}"
    )

    print("=" * 150)

    generation_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []

    neutral_vectors: List[np.ndarray] = []
    adapt_vectors: List[np.ndarray] = []

    neutral_mass_rows: List[np.ndarray] = []
    adapt_mass_rows: List[np.ndarray] = []

    labels: List[str] = []
    sids: List[int] = []
    selected_weights: List[float] = []
    transition_groups: List[str] = []

    neutral_correct_total = 0
    adapt_correct_total = 0

    weight1_count = 0
    weight2_count = 0

    w2c = 0
    c2w = 0
    c2c = 0
    w2w = 0

    with RestrictAdaptVisLayers(
        wrapper.model,
        args.adaptvis_max_layers,
    ):
        iterator = iter_repo_samples(
            dataset,
            prompts,
            answers,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

        for sid, image, prompt, gold in tqdm(
            iterator,
            desc="exact generation + last-token forwards",
        ):
            relation = normalize_relation(gold)

            if relation not in REL_TO_ID:
                raise RuntimeError(
                    f"sid={sid}: gold={gold!r} is not a four-way relation."
                )

            single_input = preprocess_exact(
                wrapper,
                prompt,
                image,
            )

            # -------------------------------------------------------------
            # 1. EXACT generation first. This verifies the method works.
            # -------------------------------------------------------------
            generation = neutral_and_adaptvis_generate(
                wrapper,
                single_input,
                weight1=args.weight1,
                weight2=args.weight2,
                threshold=args.threshold,
                max_new_tokens=args.max_new_tokens,
            )

            neutral_text = generation["neutral_text"]
            adapt_text = generation["adapt_text"]

            confidence = float(
                generation["confidence"]
            )

            selected_weight = float(
                generation["selected_weight"]
            )

            keys = generation["keys"]

            neutral_correct = bool(
                _is_correct(gold, neutral_text)
            )

            adapt_correct = bool(
                _is_correct(gold, adapt_text)
            )

            neutral_correct_total += int(
                neutral_correct
            )

            adapt_correct_total += int(
                adapt_correct
            )

            if selected_weight == float(args.weight1):
                weight1_count += 1
            elif selected_weight == float(args.weight2):
                weight2_count += 1

            if (not neutral_correct) and adapt_correct:
                group = "W2C"
                w2c += 1
            elif neutral_correct and (not adapt_correct):
                group = "C2W"
                c2w += 1
            elif neutral_correct and adapt_correct:
                group = "C2C"
                c2c += 1
            else:
                group = "W2W"
                w2w += 1

            generation_rows.append({
                "sid": sid,
                "relation": relation,
                "gold": gold,
                "confidence": confidence,
                "selected_weight": selected_weight,
                "neutral_generation": neutral_text,
                "adaptvis_generation": adapt_text,
                "neutral_correct": int(neutral_correct),
                "adaptvis_correct": int(adapt_correct),
                "transition_group": group,
            })

            # -------------------------------------------------------------
            # 2. Full-prompt forward under SAME two conditions.
            #
            # Neutral uses weight=1.0, exactly the first generation condition.
            # AdaptVis uses the dynamically selected weight for THIS sample.
            # -------------------------------------------------------------
            neutral_forward = extract_prompt_forward(
                wrapper,
                single_input,
                weight=1.0,
                keys=None,
                probe_layers=probe_layers,
            )

            adapt_forward = extract_prompt_forward(
                wrapper,
                single_input,
                weight=selected_weight,
                keys=keys,
                probe_layers=probe_layers,
            )

            h0 = neutral_forward[
                "last_vectors"
            ]

            h1 = adapt_forward[
                "last_vectors"
            ]

            m0 = neutral_forward[
                "visual_mass"
            ]

            m1 = adapt_forward[
                "visual_mass"
            ]

            dh = (
                h1 - h0
            ).astype(np.float32)

            dm = (
                m1 - m0
            ).astype(np.float32)

            previous_delta = None

            for li, layer in enumerate(probe_layers):
                base_norm = float(
                    np.linalg.norm(
                        h0[li]
                    )
                )

                delta_norm = float(
                    np.linalg.norm(
                        dh[li]
                    )
                )

                geometry_rows.append({
                    "sid": sid,
                    "relation": relation,
                    "layer": layer,
                    "selected_weight": selected_weight,
                    "transition_group": group,

                    "neutral_correct": int(neutral_correct),
                    "adaptvis_correct": int(adapt_correct),

                    "neutral_last_norm": base_norm,
                    "adaptvis_last_norm": float(
                        np.linalg.norm(
                            h1[li]
                        )
                    ),

                    "delta_last_norm": delta_norm,
                    "relative_delta_norm": (
                        delta_norm
                        / max(base_norm, EPS)
                    ),

                    "cos_neutral_adaptvis": cosine_np(
                        h0[li],
                        h1[li],
                    ),

                    "cos_delta_prev": (
                        cosine_np(
                            dh[li],
                            previous_delta,
                        )
                        if previous_delta is not None
                        else float("nan")
                    ),

                    "neutral_last_visual_mass": float(
                        m0[li]
                    ),

                    "adaptvis_last_visual_mass": float(
                        m1[li]
                    ),

                    "delta_last_visual_mass": float(
                        dm[li]
                    ),

                    "visual_mass_ratio": (
                        float(m1[li])
                        / max(
                            float(m0[li]),
                            EPS,
                        )
                    ),
                })

                previous_delta = dh[li]

            neutral_vectors.append(
                h0.astype(np.float16)
            )

            adapt_vectors.append(
                h1.astype(np.float16)
            )

            neutral_mass_rows.append(
                m0.astype(np.float32)
            )

            adapt_mass_rows.append(
                m1.astype(np.float32)
            )

            labels.append(relation)
            sids.append(int(sid))
            selected_weights.append(
                selected_weight
            )
            transition_groups.append(
                group
            )

            del single_input
            cleanup()

    N = len(sids)

    if N == 0:
        raise RuntimeError(
            "No samples analyzed."
        )

    neutral_acc = (
        neutral_correct_total / N
    )

    adapt_acc = (
        adapt_correct_total / N
    )

    gain = adapt_acc - neutral_acc

    print("\n" + "=" * 150)
    print("ACTUAL REPOSITORY GENERATION — SUCCESS CHECK")
    print("=" * 150)

    if hf_result is not None:
        print(
            f"HF standard baseline : "
            f"{hf_result['accuracy']:.4f}"
        )

    print(
        f"Scal neutral         : "
        f"{neutral_acc:.4f}"
    )

    print(
        f"Exact AdaptVis       : "
        f"{adapt_acc:.4f} "
        f"({gain:+.4f} vs Scal neutral)"
    )

    if hf_result is not None:
        print(
            f"AdaptVis vs HF base  : "
            f"{adapt_acc - hf_result['accuracy']:+.4f}"
        )

    print(
        f"W2C={w2c} C2W={c2w} "
        f"C2C={c2c} W2W={w2w} "
        f"net={w2c-c2w:+d}"
    )

    print(
        f"selected {args.weight1}: {weight1_count} | "
        f"selected {args.weight2}: {weight2_count}"
    )

    print("=" * 150)

    if gain > 0:
        print(
            "[SUCCESS] Exact AdaptVis improved generation accuracy. "
            "The following last-token analysis corresponds to a working intervention."
        )
    else:
        print(
            "[WARNING] AdaptVis did NOT improve over same-model neutral. "
            "Do not interpret the hidden-state differences as the mechanism "
            "of a successful improvement."
        )

        if args.require_improvement:
            raise RuntimeError(
                "AdaptVis generation did not improve; --require-improvement set."
            )

    # ---------------------------------------------------------------------
    # Stack representations.
    # ---------------------------------------------------------------------
    neutral_arr = np.stack(
        neutral_vectors,
        axis=0,
    )

    adapt_arr = np.stack(
        adapt_vectors,
        axis=0,
    )

    neutral_mass = np.stack(
        neutral_mass_rows,
        axis=0,
    )

    adapt_mass = np.stack(
        adapt_mass_rows,
        axis=0,
    )

    labels_arr = np.asarray(
        labels,
        dtype=object,
    )

    selected_weights_arr = np.asarray(
        selected_weights,
        dtype=np.float32,
    )

    transition_arr = np.asarray(
        transition_groups,
        dtype=object,
    )

    # ---------------------------------------------------------------------
    # Direction readout.
    # ---------------------------------------------------------------------
    direction_summary = run_direction_probes(
        neutral_arr,
        adapt_arr,
        labels_arr,
        probe_layers,
        train_ratio=args.train_ratio,
        repeats=args.repeats,
        seed=args.seed,
    )

    direction_lookup = {
        (
            row["representation"],
            int(row["layer"]),
        ): row
        for row in direction_summary
    }

    # ---------------------------------------------------------------------
    # Global layer summary.
    # ---------------------------------------------------------------------
    global_layer_summary: List[
        Dict[str, Any]
    ] = []

    for layer in probe_layers:
        rows = [
            r
            for r in geometry_rows
            if int(r["layer"]) == int(layer)
        ]

        global_layer_summary.append({
            "layer": layer,
            "N": len(rows),

            "neutral_visual_mass": safe_mean(
                r["neutral_last_visual_mass"]
                for r in rows
            ),

            "adaptvis_visual_mass": safe_mean(
                r["adaptvis_last_visual_mass"]
                for r in rows
            ),

            "delta_visual_mass": safe_mean(
                r["delta_last_visual_mass"]
                for r in rows
            ),

            "visual_mass_ratio": safe_mean(
                r["visual_mass_ratio"]
                for r in rows
            ),

            "delta_last_norm": safe_mean(
                r["delta_last_norm"]
                for r in rows
            ),

            "relative_delta_norm": safe_mean(
                r["relative_delta_norm"]
                for r in rows
            ),

            "cos_neutral_adaptvis": safe_mean(
                r["cos_neutral_adaptvis"]
                for r in rows
            ),

            "cos_delta_prev": safe_mean(
                r["cos_delta_prev"]
                for r in rows
            ),

            "neutral_spatial_acc": direction_lookup[
                ("neutral_last", layer)
            ]["accuracy_mean"],

            "adaptvis_spatial_acc": direction_lookup[
                ("adaptvis_last", layer)
            ]["accuracy_mean"],

            "delta_spatial_acc": direction_lookup[
                ("adaptvis_delta", layer)
            ]["accuracy_mean"],
        })

    # ---------------------------------------------------------------------
    # Group summaries: transition group and dynamic selected weight.
    # ---------------------------------------------------------------------
    grouped_layer_summary: List[
        Dict[str, Any]
    ] = []

    grouping_specs = []

    for group in [
        "W2C",
        "C2W",
        "C2C",
        "W2W",
    ]:
        grouping_specs.append(
            (
                "transition",
                group,
                lambda r, g=group:
                    r["transition_group"] == g,
            )
        )

    for weight in [
        float(args.weight1),
        float(args.weight2),
    ]:
        grouping_specs.append(
            (
                "selected_weight",
                str(weight),
                lambda r, w=weight:
                    abs(
                        float(r["selected_weight"])
                        - float(w)
                    ) < 1e-8,
            )
        )

    for grouping, group_name, predicate in grouping_specs:
        for layer in probe_layers:
            rows = [
                r
                for r in geometry_rows
                if (
                    int(r["layer"]) == int(layer)
                    and predicate(r)
                )
            ]

            if not rows:
                continue

            grouped_layer_summary.append({
                "grouping": grouping,
                "group": group_name,
                "layer": layer,
                "N": len(rows),

                "neutral_visual_mass": safe_mean(
                    r["neutral_last_visual_mass"]
                    for r in rows
                ),

                "adaptvis_visual_mass": safe_mean(
                    r["adaptvis_last_visual_mass"]
                    for r in rows
                ),

                "delta_visual_mass": safe_mean(
                    r["delta_last_visual_mass"]
                    for r in rows
                ),

                "visual_mass_ratio": safe_mean(
                    r["visual_mass_ratio"]
                    for r in rows
                ),

                "delta_last_norm": safe_mean(
                    r["delta_last_norm"]
                    for r in rows
                ),

                "relative_delta_norm": safe_mean(
                    r["relative_delta_norm"]
                    for r in rows
                ),

                "cos_neutral_adaptvis": safe_mean(
                    r["cos_neutral_adaptvis"]
                    for r in rows
                ),
            })

    # ---------------------------------------------------------------------
    # Console: central result.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 184)
    print("HOW SUCCESSFUL ADAPTVIS CHANGES THE LAST TOKEN")
    print("=" * 184)

    print(
        f"{'layer':>5s} | "
        f"{'Vmass0':>8s} | "
        f"{'Vmass1':>8s} | "
        f"{'dVmass':>8s} | "
        f"{'Vratio':>7s} | "
        f"{'||dh||':>9s} | "
        f"{'||dh||/||h||':>12s} | "
        f"{'cos(h0,h1)':>10s} | "
        f"{'neutral_sp':>10s} | "
        f"{'adapt_sp':>9s} | "
        f"{'delta_sp':>9s}"
    )

    print("-" * 184)

    for row in global_layer_summary:
        print(
            f"L{int(row['layer']):02d}   | "
            f"{row['neutral_visual_mass']:8.5f} | "
            f"{row['adaptvis_visual_mass']:8.5f} | "
            f"{row['delta_visual_mass']:+8.5f} | "
            f"{row['visual_mass_ratio']:7.3f} | "
            f"{row['delta_last_norm']:9.4f} | "
            f"{row['relative_delta_norm']:12.5f} | "
            f"{row['cos_neutral_adaptvis']:10.6f} | "
            f"{row['neutral_spatial_acc']:10.4f} | "
            f"{row['adaptvis_spatial_acc']:9.4f} | "
            f"{row['delta_spatial_acc']:9.4f}"
        )

    print("=" * 184)

    # Short W2C/C2W view.
    print("\n" + "=" * 150)
    print("W2C vs C2W — LAST-TOKEN CHANGE")
    print("=" * 150)

    for group in ["W2C", "C2W"]:
        group_rows = [
            r
            for r in grouped_layer_summary
            if (
                r["grouping"] == "transition"
                and r["group"] == group
            )
        ]

        if not group_rows:
            print(f"{group}: N=0")
            continue

        print(f"\n{group}:")
        print(
            f"{'layer':>5s} | "
            f"{'N':>4s} | "
            f"{'dVmass':>9s} | "
            f"{'Vratio':>7s} | "
            f"{'||dh||':>9s} | "
            f"{'cos(h0,h1)':>10s}"
        )

        for row in group_rows:
            print(
                f"L{int(row['layer']):02d}   | "
                f"{int(row['N']):4d} | "
                f"{row['delta_visual_mass']:+9.5f} | "
                f"{row['visual_mass_ratio']:7.3f} | "
                f"{row['delta_last_norm']:9.4f} | "
                f"{row['cos_neutral_adaptvis']:10.6f}"
            )

    print("=" * 150)

    # ---------------------------------------------------------------------
    # Save.
    # ---------------------------------------------------------------------
    write_csv(
        outdir / "generation_details.csv",
        generation_rows,
    )

    write_csv(
        outdir / "lasttoken_per_sample_layer.csv",
        geometry_rows,
    )

    write_csv(
        outdir / "lasttoken_layer_summary.csv",
        global_layer_summary,
    )

    write_csv(
        outdir / "lasttoken_grouped_layer_summary.csv",
        grouped_layer_summary,
    )

    write_csv(
        outdir / "direction_probe_summary.csv",
        direction_summary,
    )

    generation_summary = {
        "N": N,
        "hf_base_acc": (
            hf_result["accuracy"]
            if hf_result is not None
            else ""
        ),
        "scal_neutral_acc": neutral_acc,
        "adaptvis_acc": adapt_acc,
        "adaptvis_gain_vs_scal_neutral": gain,
        "adaptvis_gain_vs_hf_base": (
            adapt_acc
            - hf_result["accuracy"]
            if hf_result is not None
            else ""
        ),
        "W2C": w2c,
        "C2W": c2w,
        "C2C": c2c,
        "W2W": w2w,
        "net": w2c - c2w,
        "weight1": args.weight1,
        "weight1_count": weight1_count,
        "weight2": args.weight2,
        "weight2_count": weight2_count,
        "threshold": args.threshold,
        "rms_eps": args.rms_eps,
        "adaptvis_max_layers": args.adaptvis_max_layers,
    }

    write_csv(
        outdir / "generation_summary.csv",
        [generation_summary],
    )

    if args.save_vectors:
        np.savez_compressed(
            outdir / "lasttoken_vectors.npz",
            sample_index=np.asarray(
                sids,
                dtype=np.int64,
            ),
            relation=labels_arr,
            layers=np.asarray(
                probe_layers,
                dtype=np.int32,
            ),
            selected_weight=selected_weights_arr,
            transition_group=transition_arr,
            neutral_last=neutral_arr,
            adaptvis_last=adapt_arr,
            adaptvis_delta=(
                adapt_arr.astype(np.float32)
                - neutral_arr.astype(np.float32)
            ).astype(np.float16),
            neutral_visual_mass=neutral_mass,
            adaptvis_visual_mass=adapt_mass,
        )

    metadata = {
        "dataset": args.dataset,
        "option": args.option,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "torch": torch.__version__,
        "rms_eps": args.rms_eps,
        "weight1": args.weight1,
        "weight2": args.weight2,
        "threshold": args.threshold,
        "adaptvis_max_layers": args.adaptvis_max_layers,
        "probe_layers": probe_layers,
        "generation_is_exact_repo_policy": True,
        "hidden_comparison": (
            "same custom Scal model: weight=1.0 neutral vs "
            "sample-dynamic exact AdaptVis"
        ),
    }

    (
        outdir / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[saved] {outdir / 'generation_summary.csv'}"
    )

    print(
        f"[saved] {outdir / 'lasttoken_layer_summary.csv'}"
    )

    print(
        f"[saved] {outdir / 'lasttoken_grouped_layer_summary.csv'}"
    )

    print(
        f"[saved] {outdir / 'direction_probe_summary.csv'}"
    )

    if args.save_vectors:
        print(
            f"[saved] {outdir / 'lasttoken_vectors.npz'}"
        )


if __name__ == "__main__":
    main()
