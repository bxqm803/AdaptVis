#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer-wise frozen logit lens and trained linear-probe analysis for COCO/VG spatial relations.

This script answers two different questions for each selected decoder block:

1. Frozen logit lens
   Apply the model's final normalization and frozen LM-head relation-token rows to an
   intermediate hidden state. This measures whether the representation is already
   aligned with the model's final readout.

2. Trained linear probe
   Train a regularized four-way linear classifier on frozen hidden states using
   group-stratified cross-validation. This measures whether relation information is
   linearly decodable, even when it is not yet aligned with the final LM head.

The VLM is always frozen. No model parameter is updated.

Example
-------
CUDA_VISIBLE_DEVICES=0 python3 analyze_coco_layerwise_logit_lens_probe_v1.py \
  --models qwen-3b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --relations left,right,above,below \
  --layers all \
  --readouts prompt_last,subject,reference,difference,pair \
  --methods logit_lens,linear_probe \
  --views original \
  --device cuda:0 \
  --attn-impl sdpa \
  --cv-folds 5 \
  --probe-c 0.01 \
  --output-dir output/coco_layerwise_logit_lens_probe \
  --overwrite

Notes
-----
- Decoder block k is read from hidden_states[k + 1], matching the repository's
  existing layer convention.
- ``prompt_last`` is the final input token before answer generation.
- ``pair`` is concat(subject, reference, subject-reference); it is available only
  to the trained probe, not to the frozen LM-head lens.
- Original/swap views from the same image are kept in the same CV fold.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")

try:
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
    )
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for linear probes. "
        f"Import failed: {exc}"
    )


SCRIPT_VERSION = "coco-layerwise-logit-lens-probe-v1"
RELATIONS = ("left", "right", "above", "below")
RELATION_TO_INDEX = {name: i for i, name in enumerate(RELATIONS)}
INDEX_TO_RELATION = np.asarray(RELATIONS, dtype="<U8")
INVERSE_RELATION = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

ACTUAL_TOKEN_READOUTS = (
    "subject",
    "reference",
    "question_last",
    "prompt_last",
)
DERIVED_READOUTS = (
    "difference",
    "sum",
    "pair",
)
ALL_READOUTS = ACTUAL_TOKEN_READOUTS + DERIVED_READOUTS
ALL_METHODS = ("logit_lens", "linear_probe")
ALL_VIEWS = ("original", "swap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
        help="Repository helper script imported for prompts, model specs and token positions.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model aliases, e.g. qwen-3b,qwen-7b.",
    )
    parser.add_argument(
        "--dataset",
        default="coco_two",
        choices=["coco_two", "vg_two"],
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default=None,
        help="Standard prompt JSONL. Uses the base script default when omitted.",
    )
    parser.add_argument(
        "--relations",
        default="left,right,above,below",
        help="Comma-separated relation subset.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated zero-based decoder blocks or 'all'.",
    )
    parser.add_argument(
        "--readouts",
        default="prompt_last,subject,reference,difference,pair",
        help=(
            "Comma-separated readouts: subject,reference,question_last,prompt_last,"
            "difference,sum,pair. pair=concat(subject,reference,subject-reference)."
        ),
    )
    parser.add_argument(
        "--methods",
        default="logit_lens,linear_probe",
        help="Comma-separated methods: logit_lens,linear_probe.",
    )
    parser.add_argument(
        "--views",
        default="original",
        choices=["original", "swap", "both"],
        help=(
            "original: standard question; swap: reversed subject/reference question; "
            "both: include both while grouping them into the same CV fold."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on original dataset records before view expansion.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--probe-c",
        type=float,
        default=0.01,
        help="Inverse L2 regularization strength for logistic regression.",
    )
    parser.add_argument("--probe-max-iter", type=int, default=3000)
    parser.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
    )
    parser.add_argument(
        "--feature-dtype",
        default="float16",
        choices=["float16", "float32"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--reuse-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a compatible features.npz instead of rerunning the VLM.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def comma_subset(raw: str, allowed: Sequence[str], name: str) -> List[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unknown {name}: {invalid}; allowed={list(allowed)}")
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError(f"{name} resolved to an empty list")
    return values


def import_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_attr_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_final_norm(model: Any, decoder_path: str) -> Tuple[Optional[torch.nn.Module], str]:
    parent = decoder_path.rsplit(".", 1)[0] if "." in decoder_path else ""
    candidates = []
    if parent:
        candidates.extend([
            f"{parent}.norm",
            f"{parent}.final_layernorm",
            f"{parent}.ln_f",
        ])
    candidates.extend([
        "model.language_model.norm",
        "model.model.language_model.norm",
        "language_model.model.norm",
        "language_model.norm",
        "model.model.norm",
        "model.norm",
        "model.text_model.norm",
        "text_model.norm",
    ])
    for path in dict.fromkeys(candidates):
        value = get_attr_path(model, path)
        if isinstance(value, torch.nn.Module):
            return value, path
    return None, "identity"


def resolve_output_head(model: Any) -> Tuple[torch.nn.Module, str]:
    try:
        head = model.get_output_embeddings()
    except Exception:
        head = None
    if isinstance(head, torch.nn.Module) and hasattr(head, "weight"):
        return head, "get_output_embeddings()"

    for path in (
        "lm_head",
        "language_model.lm_head",
        "model.lm_head",
        "model.language_model.lm_head",
        "model.model.language_model.lm_head",
    ):
        value = get_attr_path(model, path)
        if isinstance(value, torch.nn.Module) and hasattr(value, "weight"):
            return value, path
    raise RuntimeError("Could not resolve the frozen LM output head.")


def selected_lm_logits(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    """Return four relation scores without constructing the full-vocabulary logits."""
    if hidden.ndim != 1:
        raise ValueError(f"Expected [D] hidden state, got {tuple(hidden.shape)}")
    weight = getattr(head, "weight")
    bias = getattr(head, "bias", None)
    scores = []
    for relation in RELATIONS:
        ids = torch.as_tensor(
            relation_token_map[relation],
            device=weight.device,
            dtype=torch.long,
        )
        rows = weight.index_select(0, ids)
        local = torch.mv(rows.float(), hidden.float())
        if bias is not None:
            local = local + bias.index_select(0, ids).float()
        scores.append(local.max())
    return torch.stack(scores)


def apply_final_norm(hidden: torch.Tensor, norm: Optional[torch.nn.Module]) -> torch.Tensor:
    if norm is None:
        return hidden
    source_dtype = hidden.dtype
    try:
        out = norm(hidden.unsqueeze(0)).squeeze(0)
    except Exception:
        # Some remote-code norms require the module's parameter dtype.
        parameter = next(norm.parameters(), None)
        if parameter is None:
            raise
        out = norm(hidden.to(parameter.dtype).unsqueeze(0)).squeeze(0)
    return out.to(source_dtype)


def infer_final_state_projection(
    *,
    outputs: Any,
    final_hidden: torch.Tensor,
    prompt_last: int,
    norm: Optional[torch.nn.Module],
    head: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Tuple[str, Dict[str, float]]:
    """Detect whether hidden_states[-1] is already final-normalized."""
    output_logits = getattr(outputs, "logits", None)
    if not torch.is_tensor(output_logits):
        return "raw", {"raw_mae": float("nan"), "norm_mae": float("nan")}

    target_values = []
    for relation in RELATIONS:
        ids = torch.as_tensor(
            relation_token_map[relation],
            device=output_logits.device,
            dtype=torch.long,
        )
        target_values.append(output_logits[0, prompt_last].index_select(0, ids).float().max())
    target = torch.stack(target_values)

    raw = selected_lm_logits(final_hidden, head, relation_token_map).to(target.device)
    raw_mae = float(torch.mean(torch.abs(raw - target)).item())

    if norm is None:
        return "raw", {"raw_mae": raw_mae, "norm_mae": float("nan")}

    normed_hidden = apply_final_norm(final_hidden, norm)
    normed = selected_lm_logits(normed_hidden, head, relation_token_map).to(target.device)
    norm_mae = float(torch.mean(torch.abs(normed - target)).item())
    mode = "raw" if raw_mae <= norm_mae else "norm"
    return mode, {"raw_mae": raw_mae, "norm_mae": norm_mae}


def lens_logits_for_state(
    *,
    hidden: torch.Tensor,
    layer: int,
    n_layers: int,
    final_state_mode: str,
    norm: Optional[torch.nn.Module],
    head: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    if layer == n_layers - 1 and final_state_mode == "raw":
        projected = hidden
    else:
        projected = apply_final_norm(hidden, norm)
    return selected_lm_logits(projected, head, relation_token_map)


def safe_metric(callable_obj: Any, *args: Any, **kwargs: Any) -> Optional[float]:
    try:
        value = float(callable_obj(*args, **kwargs))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
        "macro_f1": safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
    }
    if probabilities is not None:
        result["log_loss"] = safe_metric(
            log_loss,
            y_true,
            probabilities,
            labels=np.arange(len(RELATIONS)),
        )
    else:
        result["log_loss"] = None
    for code, relation in enumerate(RELATIONS):
        mask = y_true == code
        result[f"acc_{relation}"] = (
            float(np.mean(y_pred[mask] == y_true[mask])) if np.any(mask) else None
        )
        result[f"n_{relation}"] = int(np.sum(mask))
    return result


def compare_to_baseline(
    y_true: np.ndarray,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
) -> Dict[str, Any]:
    pred_ok = prediction == y_true
    base_ok = baseline_prediction == y_true
    return {
        "baseline_accuracy": float(np.mean(base_ok)),
        "probe_correct_model_wrong": int(np.sum(pred_ok & ~base_ok)),
        "probe_wrong_model_correct": int(np.sum(~pred_ok & base_ok)),
        "both_correct": int(np.sum(pred_ok & base_ok)),
        "both_wrong": int(np.sum(~pred_ok & ~base_ok)),
        "union_accuracy": float(np.mean(pred_ok | base_ok)),
        "probe_accuracy_on_model_wrong": (
            float(np.mean(pred_ok[~base_ok])) if np.any(~base_ok) else None
        ),
    }


def normalize_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def feature_for_readout(arrays: Mapping[str, np.ndarray], readout: str, layer_index: int) -> np.ndarray:
    if readout in ACTUAL_TOKEN_READOUTS:
        key = f"state_{readout}"
        if key not in arrays:
            raise KeyError(f"Feature cache lacks {key}")
        return arrays[key][:, layer_index].astype(np.float32)

    subject = arrays["state_subject"][:, layer_index].astype(np.float32)
    reference = arrays["state_reference"][:, layer_index].astype(np.float32)
    if readout == "difference":
        return subject - reference
    if readout == "sum":
        return subject + reference
    if readout == "pair":
        return np.concatenate([subject, reference, subject - reference], axis=1)
    raise ValueError(readout)


def build_cv_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    if n_splits < 2:
        raise ValueError("--cv-folds must be at least 2")
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Only {len(unique_groups)} unique groups are available for {n_splits} folds"
        )
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
        return splits, "StratifiedGroupKFold"
    except Exception as exc:
        print(
            f"[WARN] StratifiedGroupKFold failed ({type(exc).__name__}: {exc}); "
            "falling back to GroupKFold.",
            flush=True,
        )
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(labels)), labels, groups)), "GroupKFold"


def make_probe(args: argparse.Namespace) -> Pipeline:
    class_weight = None if args.class_weight == "none" else "balanced"
    return Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        (
            "clf",
            LogisticRegression(
                C=float(args.probe_c),
                penalty="l2",
                solver="lbfgs",
                max_iter=int(args.probe_max_iter),
                class_weight=class_weight,
                random_state=int(args.seed),
            ),
        ),
    ])


def extract_features_for_model(
    *,
    args: argparse.Namespace,
    base: Any,
    data_module: Any,
    model_alias: str,
    model_dir: Path,
    selected_relations: Sequence[str],
    selected_readouts: Sequence[str],
    selected_methods: Sequence[str],
) -> Path:
    feature_path = model_dir / "features.npz"
    config_path = model_dir / "feature_config.json"

    views = [args.views] if args.views != "both" else ["original", "swap"]
    prompt_path = (
        Path(args.prompt_jsonl)
        if args.prompt_jsonl
        else base.DEFAULT_PROMPT_FILES[args.dataset]
    )
    feature_config = {
        "script_version": SCRIPT_VERSION,
        "base_script": str(args.base_script),
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "prompt_jsonl": str(prompt_path),
        "model": model_alias,
        "relations": list(selected_relations),
        "layers_request": args.layers,
        "readouts": list(selected_readouts),
        "methods": list(selected_methods),
        "views": views,
        "max_samples": args.max_samples,
        "feature_dtype": args.feature_dtype,
        "seed": args.seed,
    }

    if args.reuse_features and feature_path.exists() and config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        keys = (
            "dataset",
            "data_root",
            "prompt_jsonl",
            "model",
            "relations",
            "layers_request",
            "readouts",
            "methods",
            "views",
            "max_samples",
            "feature_dtype",
        )
        if all(existing.get(key) == feature_config.get(key) for key in keys):
            print(f"[{model_alias}] Reusing {feature_path}", flush=True)
            return feature_path
        print(f"[{model_alias}] Existing feature cache is incompatible; extracting again.", flush=True)

    specs = base.merged_model_specs(data_module)
    if model_alias not in specs:
        raise ValueError(f"Unknown model {model_alias}; available={sorted(specs)}")
    spec = specs[model_alias]

    records, audit = data_module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    prompt_rows = base.load_standard_prompts(prompt_path)
    records = [
        record for record in records
        if int(record.sid) in prompt_rows
        and base.normalize_relation(prompt_rows[int(record.sid)]["answer_raw"]) in selected_relations
    ]
    if not records:
        raise RuntimeError("No records remain after prompt/relation filtering")

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} lacks {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(getattr(spec, "trust_remote_code", False)),
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    model = None
    processor = None
    started = time.time()
    try:
        print(f"[{model_alias}] Loading {spec.repo_id}", flush=True)
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=bool(getattr(spec, "trust_remote_code", False)),
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)
        selected_layers = base.parse_layers(args.layers, n_layers)
        final_norm, norm_path = resolve_final_norm(model, decoder_path)
        output_head, head_path = resolve_output_head(model)
        relation_token_map = base.relation_token_variants(processor.tokenizer)

        print(
            f"[{model_alias}] decoder={decoder_path}, n_layers={n_layers}, "
            f"selected={selected_layers}",
            flush=True,
        )
        print(
            f"[{model_alias}] final_norm={norm_path}, output_head={head_path}, "
            f"relation_token_map={relation_token_map}",
            flush=True,
        )

        needed_actual = set()
        for readout in selected_readouts:
            if readout in ACTUAL_TOKEN_READOUTS:
                needed_actual.add(readout)
            elif readout in ("difference", "sum", "pair"):
                needed_actual.update(("subject", "reference"))
        if "logit_lens" in selected_methods:
            needed_actual.update(
                readout for readout in selected_readouts if readout in ACTUAL_TOKEN_READOUTS
            )

        feature_lists: Dict[str, List[np.ndarray]] = {
            f"state_{name}": [] for name in sorted(needed_actual)
        }
        lens_lists: Dict[str, List[np.ndarray]] = {
            f"lens_{name}": []
            for name in selected_readouts
            if name in ACTUAL_TOKEN_READOUTS and "logit_lens" in selected_methods
        }

        sample_index: List[int] = []
        image_id: List[str] = []
        group_id: List[str] = []
        view_names: List[str] = []
        subjects: List[str] = []
        references: List[str] = []
        labels: List[int] = []
        baseline_logits: List[np.ndarray] = []
        errors: List[Dict[str, Any]] = []
        final_state_mode: Optional[str] = None
        final_projection_diagnostics: Optional[Dict[str, float]] = None
        hidden_size: Optional[int] = None

        total = len(records) * len(views)
        progress = tqdm(total=total, desc=f"lens-features:{model_alias}")
        done = 0

        for record in records:
            sid = int(record.sid)
            prompt_row = prompt_rows[sid]
            original_relation = base.normalize_relation(prompt_row["answer_raw"])
            if original_relation not in selected_relations:
                progress.update(len(views))
                continue

            try:
                image = base.record_image(record)
            except Exception as exc:
                for view in views:
                    errors.append({
                        "sid": sid,
                        "view": view,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    progress.update(1)
                continue

            for view in views:
                try:
                    if view == "original":
                        question_text = str(prompt_row["question_text"])
                        subject = str(prompt_row["subject"])
                        reference = str(prompt_row["reference"])
                        relation = str(original_relation)
                    else:
                        subject = str(prompt_row["reference"])
                        reference = str(prompt_row["subject"])
                        question_text = base.build_swapped_question(
                            str(prompt_row["subject"]),
                            str(prompt_row["reference"]),
                        )
                        relation = INVERSE_RELATION[str(original_relation)]

                    batch = base.make_question_batch(
                        processor=processor,
                        image=image,
                        question_text=question_text,
                        device=device,
                    )
                    input_ids = batch["input_ids"][0].detach().cpu().tolist()
                    subject_span, reference_span = base.locate_object_spans(
                        processor.tokenizer,
                        input_ids,
                        subject,
                        reference,
                    )
                    positions = {
                        "subject": int(subject_span[1]),
                        "reference": int(reference_span[1]),
                        "question_last": int(base.locate_question_last_token(
                            processor.tokenizer,
                            input_ids,
                            question_text,
                        )),
                        "prompt_last": int(len(input_ids) - 1),
                    }

                    with torch.inference_mode():
                        outputs = model(
                            **batch,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                    states = base.hidden_tuple(outputs)
                    if len(states) != n_layers + 1:
                        raise RuntimeError(
                            f"Expected {n_layers + 1} hidden-state tensors, got {len(states)}"
                        )
                    final = states[-1]
                    if final.ndim != 3 or final.shape[0] != 1:
                        raise RuntimeError(f"Unexpected hidden-state shape: {tuple(final.shape)}")
                    if int(final.shape[1]) != len(input_ids):
                        raise RuntimeError(
                            "Token/hidden length mismatch: "
                            f"input={len(input_ids)}, hidden={final.shape[1]}"
                        )
                    if hidden_size is None:
                        hidden_size = int(final.shape[-1])

                    if final_state_mode is None:
                        final_state_mode, final_projection_diagnostics = infer_final_state_projection(
                            outputs=outputs,
                            final_hidden=states[-1][0, positions["prompt_last"]],
                            prompt_last=positions["prompt_last"],
                            norm=final_norm,
                            head=output_head,
                            relation_token_map=relation_token_map,
                        )
                        print(
                            f"[{model_alias}] final hidden projection mode={final_state_mode}, "
                            f"diagnostics={final_projection_diagnostics}",
                            flush=True,
                        )

                    sample_states: Dict[str, np.ndarray] = {}
                    for name in sorted(needed_actual):
                        values = []
                        for layer in selected_layers:
                            values.append(
                                states[layer + 1][0, positions[name]]
                                .detach()
                                .float()
                                .cpu()
                                .numpy()
                            )
                        sample_states[name] = np.stack(values, axis=0)

                    sample_lens: Dict[str, np.ndarray] = {}
                    if "logit_lens" in selected_methods:
                        for name in selected_readouts:
                            if name not in ACTUAL_TOKEN_READOUTS:
                                continue
                            layer_logits = []
                            for layer in selected_layers:
                                logits = lens_logits_for_state(
                                    hidden=states[layer + 1][0, positions[name]],
                                    layer=layer,
                                    n_layers=n_layers,
                                    final_state_mode=str(final_state_mode),
                                    norm=final_norm,
                                    head=output_head,
                                    relation_token_map=relation_token_map,
                                )
                                layer_logits.append(
                                    logits.detach().float().cpu().numpy()
                                )
                            sample_lens[name] = np.stack(layer_logits, axis=0)

                    exact_final = lens_logits_for_state(
                        hidden=states[-1][0, positions["prompt_last"]],
                        layer=n_layers - 1,
                        n_layers=n_layers,
                        final_state_mode=str(final_state_mode),
                        norm=final_norm,
                        head=output_head,
                        relation_token_map=relation_token_map,
                    )

                    feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32
                    for name, values in sample_states.items():
                        feature_lists[f"state_{name}"].append(values.astype(feature_dtype))
                    for name, values in sample_lens.items():
                        lens_lists[f"lens_{name}"].append(values.astype(np.float32))

                    sample_index.append(sid)
                    image_id.append(str(record.image_id))
                    # Group by physical image to prevent image or original/swap leakage.
                    group_id.append(str(record.image_id))
                    view_names.append(view)
                    subjects.append(subject)
                    references.append(reference)
                    labels.append(RELATION_TO_INDEX[relation])
                    baseline_logits.append(
                        exact_final.detach().float().cpu().numpy().astype(np.float32)
                    )
                    done += 1
                    if args.print_every > 0 and done % args.print_every == 0:
                        baseline_pred = RELATIONS[int(torch.argmax(exact_final).item())]
                        tqdm.write(
                            f"[{model_alias}] done={done} sid={sid} view={view} "
                            f"gt={relation} final_lmhead={baseline_pred}"
                        )

                    del outputs, states, batch
                except Exception as exc:
                    errors.append({
                        "sid": sid,
                        "image_id": str(record.image_id),
                        "view": view,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-12:],
                    })
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    progress.update(1)

        progress.close()
        if not labels:
            raise RuntimeError("No features were extracted successfully")
        assert hidden_size is not None
        assert final_state_mode is not None

        arrays: Dict[str, Any] = {
            "metadata_json": np.array(json.dumps({
                **feature_config,
                "repo_id": spec.repo_id,
                "transformers_version": transformers.__version__,
                "decoder_path": decoder_path,
                "decoder_blocks": n_layers,
                "selected_layers": selected_layers,
                "hidden_size": hidden_size,
                "final_norm_path": norm_path,
                "output_head_path": head_path,
                "final_state_mode": final_state_mode,
                "final_projection_diagnostics": final_projection_diagnostics,
                "relation_token_map": relation_token_map,
                "n_saved": len(labels),
                "n_errors": len(errors),
                "elapsed_minutes": (time.time() - started) / 60.0,
            }), dtype=object),
            "sample_index": np.asarray(sample_index, dtype=np.int64),
            "image_id": np.asarray(image_id, dtype=object),
            "group_id": np.asarray(group_id, dtype=object),
            "view": np.asarray(view_names, dtype=object),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "label": np.asarray(labels, dtype=np.int64),
            "selected_layers": np.asarray(selected_layers, dtype=np.int32),
            "baseline_logits": np.stack(baseline_logits, axis=0).astype(np.float32),
        }
        for key, values in feature_lists.items():
            arrays[key] = np.stack(values, axis=0)
        for key, values in lens_lists.items():
            arrays[key] = np.stack(values, axis=0)

        model_dir.mkdir(parents=True, exist_ok=True)
        tmp = feature_path.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(feature_path)
        config_path.write_text(
            json.dumps(feature_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (model_dir / "feature_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{model_alias}] Saved {len(labels)} examples to {feature_path}; "
            f"errors={len(errors)}, elapsed={(time.time() - started) / 60.0:.1f} min",
            flush=True,
        )
        return feature_path
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze_features(
    *,
    args: argparse.Namespace,
    model_alias: str,
    model_dir: Path,
    feature_path: Path,
    selected_readouts: Sequence[str],
    selected_methods: Sequence[str],
) -> None:
    with np.load(feature_path, allow_pickle=True) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    labels = arrays["label"].astype(np.int64)
    groups = arrays["group_id"].astype(str)
    layers = arrays["selected_layers"].astype(np.int64).tolist()
    baseline_logits = arrays["baseline_logits"].astype(np.float32)
    baseline_prediction = np.argmax(baseline_logits, axis=1)
    baseline_probability = normalize_probabilities(baseline_logits)

    splits, split_name = build_cv_splits(
        labels,
        groups,
        int(args.cv_folds),
        int(args.seed),
    )
    fold_assignment = np.full(len(labels), -1, dtype=np.int32)
    for fold, (_, test_index) in enumerate(splits):
        fold_assignment[test_index] = fold
    if np.any(fold_assignment < 0):
        raise RuntimeError("Some samples were not assigned to a CV test fold")

    result_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    probe_probabilities: Dict[str, np.ndarray] = {}

    baseline_metrics = relation_metrics(labels, baseline_prediction, baseline_probability)
    baseline_row = {
        "model": model_alias,
        "method": "final_lm_head",
        "readout": "prompt_last",
        "layer": int(metadata["decoder_blocks"]) - 1,
        "n": len(labels),
        "feature_dim": int(metadata["hidden_size"]),
        **baseline_metrics,
        **compare_to_baseline(labels, baseline_prediction, baseline_prediction),
    }
    result_rows.append(baseline_row)

    if "logit_lens" in selected_methods:
        for readout in selected_readouts:
            if readout not in ACTUAL_TOKEN_READOUTS:
                continue
            key = f"lens_{readout}"
            if key not in arrays:
                continue
            lens = arrays[key].astype(np.float32)
            for layer_index, layer in enumerate(layers):
                logits = lens[:, layer_index]
                probability = normalize_probabilities(logits)
                prediction = np.argmax(logits, axis=1)
                metrics = relation_metrics(labels, prediction, probability)
                result_rows.append({
                    "model": model_alias,
                    "method": "logit_lens",
                    "readout": readout,
                    "layer": int(layer),
                    "n": len(labels),
                    "feature_dim": int(metadata["hidden_size"]),
                    **metrics,
                    **compare_to_baseline(labels, prediction, baseline_prediction),
                })

    if "linear_probe" in selected_methods:
        base_probe = make_probe(args)
        for readout in selected_readouts:
            for layer_index, layer in enumerate(layers):
                features = feature_for_readout(arrays, readout, layer_index)
                oof_probability = np.zeros((len(labels), len(RELATIONS)), dtype=np.float32)
                train_accuracies: List[float] = []
                test_accuracies: List[float] = []

                for fold, (train_index, test_index) in enumerate(splits):
                    probe = clone(base_probe)
                    probe.fit(features[train_index], labels[train_index])
                    probability = probe.predict_proba(features[test_index])
                    # sklearn may omit a class in a pathological training fold.
                    aligned = np.zeros((len(test_index), len(RELATIONS)), dtype=np.float32)
                    classes = probe.named_steps["clf"].classes_.astype(int)
                    aligned[:, classes] = probability.astype(np.float32)
                    oof_probability[test_index] = aligned

                    train_prediction = probe.predict(features[train_index])
                    test_prediction = np.argmax(aligned, axis=1)
                    train_accuracy = float(np.mean(train_prediction == labels[train_index]))
                    test_accuracy = float(np.mean(test_prediction == labels[test_index]))
                    train_accuracies.append(train_accuracy)
                    test_accuracies.append(test_accuracy)
                    fold_rows.append({
                        "model": model_alias,
                        "readout": readout,
                        "layer": int(layer),
                        "fold": fold,
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                    })

                prediction = np.argmax(oof_probability, axis=1)
                metrics = relation_metrics(labels, prediction, oof_probability)
                key = f"{readout}__L{layer}"
                probe_probabilities[key] = oof_probability.astype(np.float16)
                result_rows.append({
                    "model": model_alias,
                    "method": "linear_probe",
                    "readout": readout,
                    "layer": int(layer),
                    "n": len(labels),
                    "feature_dim": int(features.shape[1]),
                    "cv_splitter": split_name,
                    "cv_folds": len(splits),
                    "fold_test_acc_mean": float(np.mean(test_accuracies)),
                    "fold_test_acc_std": float(np.std(test_accuracies, ddof=1)) if len(test_accuracies) > 1 else 0.0,
                    "fold_train_acc_mean": float(np.mean(train_accuracies)),
                    **metrics,
                    **compare_to_baseline(labels, prediction, baseline_prediction),
                })

    result_rows.sort(key=lambda row: (
        str(row.get("method")),
        str(row.get("readout")),
        int(row.get("layer", -1)),
    ))
    write_csv(model_dir / "layerwise_results.csv", result_rows)
    write_csv(model_dir / "probe_fold_results.csv", fold_rows)

    prediction_arrays: Dict[str, Any] = {
        "sample_index": arrays["sample_index"],
        "image_id": arrays["image_id"],
        "group_id": arrays["group_id"],
        "view": arrays["view"],
        "label": labels,
        "fold": fold_assignment,
        "baseline_logits": baseline_logits,
        "baseline_prediction": baseline_prediction,
    }
    for key, value in probe_probabilities.items():
        prediction_arrays[f"probe_probability__{key}"] = value
    np.savez_compressed(model_dir / "oof_predictions.npz", **prediction_arrays)

    best_rows: List[Dict[str, Any]] = []
    for method in ("logit_lens", "linear_probe"):
        for readout in selected_readouts:
            candidates = [
                row for row in result_rows
                if row.get("method") == method
                and row.get("readout") == readout
                and row.get("accuracy") is not None
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: (float(row["accuracy"]), -int(row["layer"])))
            best_rows.append(dict(best))

    report_lines = [
        "=" * 118,
        "LAYER-WISE FROZEN LOGIT LENS AND LINEAR PROBE",
        f"model={model_alias} | n={len(labels)} | groups={len(np.unique(groups))} | "
        f"layers={layers[0]}..{layers[-1]} ({len(layers)}) | views={sorted(set(arrays['view'].astype(str)))}",
        f"baseline final-LM-head accuracy={baseline_metrics['accuracy']:.4f} | "
        f"balanced={baseline_metrics['balanced_accuracy']:.4f} | macro-F1={baseline_metrics['macro_f1']:.4f}",
        f"CV={split_name}, folds={len(splits)}, C={args.probe_c}, class_weight={args.class_weight}",
        "=" * 118,
        "",
        f"{'Method':<16}{'Readout':<16}{'Best L':>8}{'Accuracy':>11}{'Balanced':>11}"
        f"{'Macro-F1':>11}{'Union':>10}{'Fix wrong':>11}{'Damage':>9}",
        "-" * 118,
    ]
    for row in best_rows:
        report_lines.append(
            f"{str(row['method']):<16}{str(row['readout']):<16}{int(row['layer']):>8}"
            f"{float(row['accuracy']):>11.4f}"
            f"{float(row['balanced_accuracy']):>11.4f}"
            f"{float(row['macro_f1']):>11.4f}"
            f"{float(row['union_accuracy']):>10.4f}"
            f"{int(row['probe_correct_model_wrong']):>11}"
            f"{int(row['probe_wrong_model_correct']):>9}"
        )

    report_lines += [
        "",
        "Interpretation:",
        "- High linear-probe accuracy but low frozen-logit-lens accuracy: relation information exists but is not yet aligned with the final LM head.",
        "- High values for both: the layer already contains an LM-head-readable relation state.",
        "- Low values for both: the selected token/readout does not yet contain a strong linearly decodable relation state.",
        "- pair is a diagnostic concat(subject, reference, subject-reference); it is not an actual token and has no frozen logit-lens result.",
        "- All linear-probe accuracies are out-of-fold; original/swap views and repeated records from the same image stay in one fold.",
        "",
        "Saved:",
        f"  {model_dir / 'features.npz'}",
        f"  {model_dir / 'layerwise_results.csv'}",
        f"  {model_dir / 'probe_fold_results.csv'}",
        f"  {model_dir / 'oof_predictions.npz'}",
        f"  {model_dir / 'report.txt'}",
    ]
    report = "\n".join(report_lines) + "\n"
    (model_dir / "report.txt").write_text(report, encoding="utf-8")

    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model_alias,
        "feature_metadata": metadata,
        "n": len(labels),
        "unique_groups": len(np.unique(groups)),
        "label_counts": {
            relation: int(np.sum(labels == code))
            for code, relation in enumerate(RELATIONS)
        },
        "baseline": baseline_row,
        "best_conditions": best_rows,
        "cv_splitter": split_name,
        "cv_folds": len(splits),
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + report, flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if args.probe_c <= 0:
        raise ValueError("--probe-c must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = list(dict.fromkeys(
        part.strip() for part in args.models.split(",") if part.strip()
    ))
    if not models:
        raise ValueError("--models resolved to an empty list")
    selected_relations = comma_subset(args.relations, RELATIONS, "relation")
    selected_readouts = comma_subset(args.readouts, ALL_READOUTS, "readout")
    selected_methods = comma_subset(args.methods, ALL_METHODS, "method")

    base = import_file(Path(args.base_script), "_layerwise_lens_base")
    data_module = base.import_two_object_module()
    available_models = base.merged_model_specs(data_module)
    unknown = sorted(set(models) - set(available_models))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available={sorted(available_models)}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    for model_alias in models:
        model_dir = output_root / model_alias
        if args.overwrite and model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 118, flush=True)
        print(f"MODEL: {model_alias}", flush=True)
        print("=" * 118, flush=True)

        feature_path = extract_features_for_model(
            args=args,
            base=base,
            data_module=data_module,
            model_alias=model_alias,
            model_dir=model_dir,
            selected_relations=selected_relations,
            selected_readouts=selected_readouts,
            selected_methods=selected_methods,
        )
        analyze_features(
            args=args,
            model_alias=model_alias,
            model_dir=model_dir,
            feature_path=feature_path,
            selected_readouts=selected_readouts,
            selected_methods=selected_methods,
        )
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            all_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    (output_root / "combined_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Combined summary: {output_root / 'combined_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
