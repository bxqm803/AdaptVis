#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate multi-layer directional repair with actual autoregressive generation.

This script directly calls model.generate() for:
  1. the unmodified baseline;
  2. each selected intervention variant.

It can also load samples.jsonl from the previous logit-based experiment and
measure sample-level agreement between:
  - relation-logit prediction;
  - parsed free-generation prediction.

Recommended first comparison:
  --variants block_flip_all,attn_flip_all
  --trigger all
  --hook-mode prefill
  --logit-results-dir <previous multilayer repair output>

Why prefill mode?
-----------------
The previous logit experiment modifies the final prompt-token state and reads
the next-token relation logits. During model.generate(), that computation is
the prefill pass. Applying the intervention only on prefill is therefore the
closest generation equivalent of the previous experiment.

Hook modes
----------
prefill:
    Modify only the initial full-prompt forward pass.

every-step:
    Modify the last token at prefill and every autoregressive decoding step.
    This is a stronger intervention and is not directly equivalent to the
    previous next-token logit experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

import eval_directional_head_repair_v1 as core
import eval_multilayer_directional_repair_v2 as repair


SCRIPT_VERSION = "eval-multilayer-directional-generation-all440-v2"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--base-module", default=None)
    p.add_argument("--trace-dir", required=True)
    p.add_argument(
        "--prior-dir",
        default=None,
        help=(
            "Directory produced by eval_topk_attention_centroid_generation_v1.py. "
            "When supplied, load all rows from <prior-dir>/generation.jsonl, "
            "including centroid-wrong samples, instead of the 385-row "
            "centroid-correct trace metadata."
        ),
    )
    p.add_argument("--logit-results-dir", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default=None)
    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--guide",
        choices=["centroid", "oracle"],
        default="centroid",
    )
    p.add_argument(
        "--trigger",
        choices=["conflict", "all", "wrong-only"],
        default="all",
    )
    p.add_argument("--centroid-confidence-threshold", type=float, default=0.0)

    p.add_argument("--start-layer", type=int, default=26)
    p.add_argument("--end-layer", type=int, default=35)
    p.add_argument(
        "--variants",
        default="block_flip_all,attn_flip_all",
    )
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument(
        "--hook-mode",
        choices=["prefill", "every-step"],
        default="prefill",
    )

    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--min-new-tokens", type=int, default=1)
    p.add_argument("--relations", default="left,right,above,below")
    p.add_argument("--max-per-group", type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=1)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def safe_float(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def safe_mean(values: Iterable[float]) -> float:
    finite = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else float("nan")


def load_jsonl_by_sid(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    result: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if "sid" not in row:
                raise RuntimeError(
                    f"{path}:{line_number} contains no sid"
                )
            result[int(row["sid"])] = row
    return result


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# ---------------------------------------------------------------------------
# Generation relation parsing
# ---------------------------------------------------------------------------

RELATION_PATTERNS: Dict[str, Sequence[str]] = {
    "left": (
        r"\bleft\s+of\b",
        r"\bto\s+the\s+left\b",
        r"\bon\s+the\s+left\b",
        r"\bleft\b",
    ),
    "right": (
        r"\bright\s+of\b",
        r"\bto\s+the\s+right\b",
        r"\bon\s+the\s+right\b",
        r"\bright\b",
    ),
    "above": (
        r"\bon\s+top\s+of\b",
        r"\batop\b",
        r"\babove\b",
        r"\bover\b",
        r"\bon\b",
    ),
    "below": (
        r"\bunderneath\b",
        r"\bbeneath\b",
        r"\bbelow\b",
        r"\bunder\b",
    ),
}


def parse_generated_relation(text: str) -> Optional[str]:
    """
    Return the earliest explicit relation mention in generated text.

    "on" is accepted as above because the trace normalizes the vertical
    relation to above/below. Longer phrases are searched before single words.
    """
    normalized = normalize_space(text).lower()
    if not normalized:
        return None

    candidates: List[Tuple[int, int, str]] = []
    for relation, patterns in RELATION_PATTERNS.items():
        for priority, pattern in enumerate(patterns):
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match is not None:
                candidates.append(
                    (int(match.start()), int(priority), relation)
                )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Output replacement and hook intervention
# ---------------------------------------------------------------------------

def correction_delta(
    update: torch.Tensor,
    direction_cpu: torch.Tensor,
    *,
    mode: str,
    strength: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    update_float = update.float()
    direction = direction_cpu.to(
        device=update.device,
        dtype=torch.float32,
    )
    alpha = float(torch.dot(update_float, direction).item())

    if alpha >= 0.0:
        delta_float = torch.zeros_like(update_float)
    elif mode == "remove":
        delta_float = -float(strength) * alpha * direction
    elif mode == "flip":
        delta_float = -2.0 * float(strength) * alpha * direction
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    alpha_delta = float(torch.dot(delta_float, direction).item())
    diagnostics = {
        "alpha_before": alpha,
        "alpha_delta": alpha_delta,
        "alpha_after_estimate": alpha + alpha_delta,
        "was_negative": float(alpha < 0.0),
        "update_norm": float(update_float.norm().item()),
        "delta_norm": float(delta_float.norm().item()),
    }
    return delta_float.to(dtype=update.dtype), diagnostics


def generate_with_intervention(
    *,
    base: Any,
    model: Any,
    batch: Dict[str, Any],
    layers: Sequence[torch.nn.Module],
    attention_modules: Sequence[torch.nn.Module],
    variant: Dict[str, Any],
    direction: torch.Tensor,
    strength: float,
    hook_mode: str,
    prompt_length: int,
    generation_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Run model.generate() while modifying selected module outputs.

    In prefill mode, a layer is modified only when its sequence length equals
    the original prompt length, and only once.

    In every-step mode, the last current token is modified on every forward.
    """
    diagnostics: List[Dict[str, Any]] = []
    handles = []

    modules = (
        attention_modules
        if variant["target"] == "attention"
        else layers
    )

    state: Dict[int, Dict[str, int]] = {
        int(layer): {
            "calls": 0,
            "applications": 0,
        }
        for layer in variant["layers"]
    }

    for layer_index in variant["layers"]:
        module = modules[layer_index]

        def make_hook(index: int):
            def hook(
                _module: torch.nn.Module,
                inputs: Tuple[Any, ...],
                output: Any,
            ) -> Any:
                state[index]["calls"] += 1

                output_tensor = base.first_tensor(output)
                if (
                    output_tensor.ndim != 3
                    or int(output_tensor.shape[0]) != 1
                ):
                    raise RuntimeError(
                        f"Unexpected {variant['target']} output at "
                        f"L{index}: {tuple(output_tensor.shape)}"
                    )

                sequence_length = int(output_tensor.shape[1])

                if hook_mode == "prefill":
                    should_apply = (
                        sequence_length == int(prompt_length)
                        and state[index]["applications"] == 0
                    )
                elif hook_mode == "every-step":
                    should_apply = True
                else:
                    raise ValueError(hook_mode)

                if not should_apply:
                    return output

                target_index = sequence_length - 1

                if variant["target"] == "attention":
                    update = output_tensor[0, target_index, :]
                else:
                    if not inputs:
                        raise RuntimeError(
                            f"Block L{index} hook received no inputs"
                        )
                    input_tensor = base.first_tensor(inputs)
                    if input_tensor.ndim != 3:
                        raise RuntimeError(
                            f"Unexpected block input at L{index}: "
                            f"{tuple(input_tensor.shape)}"
                        )
                    if int(input_tensor.shape[1]) != sequence_length:
                        raise RuntimeError(
                            f"Input/output sequence mismatch at L{index}: "
                            f"{tuple(input_tensor.shape)} vs "
                            f"{tuple(output_tensor.shape)}"
                        )
                    update = (
                        output_tensor[0, target_index, :]
                        - input_tensor[0, target_index, :]
                    )

                delta, info = correction_delta(
                    update,
                    direction,
                    mode=variant["mode"],
                    strength=strength,
                )

                modified = output_tensor.clone()
                modified[0, target_index, :] = (
                    modified[0, target_index, :]
                    + delta.to(
                        device=modified.device,
                        dtype=modified.dtype,
                    )
                )

                state[index]["applications"] += 1
                diagnostics.append({
                    "layer": int(index),
                    "target": variant["target"],
                    "mode": variant["mode"],
                    "hook_mode": hook_mode,
                    "module_call": int(state[index]["calls"]),
                    "application_index": int(
                        state[index]["applications"]
                    ),
                    "sequence_length": sequence_length,
                    "target_index": target_index,
                    **info,
                })

                return core.replace_first_tensor(output, modified)

            return hook

        handles.append(
            module.register_forward_hook(make_hook(int(layer_index)))
        )

    try:
        with torch.inference_mode():
            sequences = model.generate(
                **batch,
                **generation_kwargs,
            )
    finally:
        for handle in handles:
            handle.remove()

    diagnostics.sort(
        key=lambda row: (
            int(row["application_index"]),
            int(row["layer"]),
        )
    )
    return sequences, diagnostics


def decode_new_tokens(
    *,
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_length: int,
) -> Tuple[str, List[int]]:
    if sequences.ndim != 2 or int(sequences.shape[0]) != 1:
        raise RuntimeError(
            f"Expected generation sequences [1,T], "
            f"got {tuple(sequences.shape)}"
        )

    sequence = sequences[0]
    if int(sequence.shape[0]) >= int(prompt_length):
        new_ids_tensor = sequence[int(prompt_length):]
    else:
        # Defensive fallback for a model returning only new tokens.
        new_ids_tensor = sequence

    new_ids = [
        int(token_id)
        for token_id in new_ids_tensor.detach().cpu().tolist()
    ]
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return normalize_space(text), new_ids


def run_baseline_generation(
    *,
    model: Any,
    batch: Dict[str, Any],
    generation_kwargs: Dict[str, Any],
) -> torch.Tensor:
    with torch.inference_mode():
        return model.generate(
            **batch,
            **generation_kwargs,
        )


# ---------------------------------------------------------------------------
# Trigger and summary
# ---------------------------------------------------------------------------

def should_trigger_generation(
    *,
    trigger: str,
    prior_group: str,
    baseline_prediction: Optional[str],
    guide: str,
    confidence: float,
    threshold: float,
    wrong_group_name: str,
) -> bool:
    if not math.isfinite(confidence):
        confidence = -math.inf
    if confidence < threshold:
        return False

    if trigger == "all":
        return True
    if trigger == "wrong-only":
        return (
            prior_group == wrong_group_name
            or "generation_wrong" in str(prior_group)
        )
    if trigger == "conflict":
        return baseline_prediction != guide
    raise ValueError(trigger)


def generation_result(
    *,
    text: str,
    token_ids: Sequence[int],
    gt: str,
) -> Dict[str, Any]:
    prediction = parse_generated_relation(text)
    return {
        "text": text,
        "token_ids": list(token_ids),
        "prediction": prediction,
        "parsed": prediction is not None,
        "correct": bool(prediction == gt),
    }


def summarize_variant(
    rows: Sequence[Dict[str, Any]],
    variant: str,
) -> Dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("status") == "ok"
        and variant in row.get("variants", {})
    ]
    if not usable:
        return {"variant": variant, "n": 0}

    baseline_correct = np.asarray([
        bool(row["baseline_generation"]["correct"])
        for row in usable
    ], dtype=bool)
    intervention_correct = np.asarray([
        bool(row["variants"][variant]["correct"])
        for row in usable
    ], dtype=bool)

    baseline_parsed = np.asarray([
        bool(row["baseline_generation"]["parsed"])
        for row in usable
    ], dtype=bool)
    intervention_parsed = np.asarray([
        bool(row["variants"][variant]["parsed"])
        for row in usable
    ], dtype=bool)
    triggered = np.asarray([
        bool(row["variants"][variant]["triggered"])
        for row in usable
    ], dtype=bool)

    repaired = (~baseline_correct) & intervention_correct
    damaged = baseline_correct & (~intervention_correct)
    changed = np.asarray([
        row["baseline_generation"]["prediction"]
        != row["variants"][variant]["prediction"]
        for row in usable
    ], dtype=bool)

    wrong = ~baseline_correct
    correct = baseline_correct

    logit_baseline_pairs = [
        row
        for row in usable
        if row.get("logit_baseline_prediction") is not None
        and row["baseline_generation"]["prediction"] is not None
    ]
    logit_variant_pairs = [
        row
        for row in usable
        if row.get("logit_variant_predictions", {}).get(variant)
        is not None
        and row["variants"][variant]["prediction"] is not None
    ]

    baseline_logit_agreement = (
        safe_mean(
            float(
                row["baseline_generation"]["prediction"]
                == row["logit_baseline_prediction"]
            )
            for row in logit_baseline_pairs
        )
        if logit_baseline_pairs
        else float("nan")
    )
    intervention_logit_agreement = (
        safe_mean(
            float(
                row["variants"][variant]["prediction"]
                == row["logit_variant_predictions"][variant]
            )
            for row in logit_variant_pairs
        )
        if logit_variant_pairs
        else float("nan")
    )

    logit_change_pairs = [
        row
        for row in usable
        if row.get("logit_baseline_prediction") is not None
        and row.get("logit_variant_predictions", {}).get(variant)
        is not None
        and row["baseline_generation"]["prediction"] is not None
        and row["variants"][variant]["prediction"] is not None
    ]
    change_consistency = (
        safe_mean(
            float(
                (
                    row["baseline_generation"]["prediction"]
                    != row["variants"][variant]["prediction"]
                )
                == (
                    row["logit_baseline_prediction"]
                    != row["logit_variant_predictions"][variant]
                )
            )
            for row in logit_change_pairs
        )
        if logit_change_pairs
        else float("nan")
    )

    return {
        "variant": variant,
        "n": len(usable),
        "n_triggered": int(triggered.sum()),
        "baseline_parse_rate": float(baseline_parsed.mean()),
        "intervention_parse_rate": float(intervention_parsed.mean()),
        "baseline_generation_accuracy": float(
            baseline_correct.mean()
        ),
        "intervention_generation_accuracy": float(
            intervention_correct.mean()
        ),
        "generation_accuracy_change": float(
            intervention_correct.mean()
            - baseline_correct.mean()
        ),
        "repaired_wrong": int(repaired.sum()),
        "damaged_correct": int(damaged.sum()),
        "net_repair": int(repaired.sum() - damaged.sum()),
        "wrong_n": int(wrong.sum()),
        "wrong_repair_rate": (
            float(repaired[wrong].mean())
            if wrong.any()
            else float("nan")
        ),
        "correct_n": int(correct.sum()),
        "correct_damage_rate": (
            float(damaged[correct].mean())
            if correct.any()
            else float("nan")
        ),
        "prediction_changed": int(changed.sum()),
        "baseline_generation_logit_agreement":
            baseline_logit_agreement,
        "intervention_generation_logit_agreement":
            intervention_logit_agreement,
        "generation_logit_change_consistency":
            change_consistency,
        "n_baseline_logit_comparable":
            len(logit_baseline_pairs),
        "n_intervention_logit_comparable":
            len(logit_variant_pairs),
    }


def summarize_by_relation(
    rows: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    relations: Sequence[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for relation in relations:
        subset = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("gt") == relation
        ]
        for variant in variants:
            summary = summarize_variant(subset, variant)
            summary["relation"] = relation
            output.append(summary)
    return output


def summarize_by_centroid_status(
    rows: Sequence[Dict[str, Any]],
    variants: Sequence[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for status, expected in (
        ("centroid_correct", True),
        ("centroid_wrong", False),
    ):
        subset = [
            row
            for row in rows
            if row.get("status") == "ok"
            and bool(
                row.get("centroid_prediction") == row.get("gt")
            ) == expected
        ]
        for variant in variants:
            summary = summarize_variant(subset, variant)
            summary["centroid_status"] = status
            output.append(summary)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.start_layer > args.end_layer:
        raise ValueError("--start-layer must be <= --end-layer")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.min_new_tokens < 0:
        raise ValueError("--min-new-tokens must be non-negative")
    if args.strength < 0:
        raise ValueError("--strength must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = core.load_base(args.base_module)

    relations = [
        base.normalize_relation(item)
        for item in core.csv_list(args.relations)
    ]
    if set(relations) != set(core.OPPOSITE):
        raise ValueError(
            "--relations must resolve to left,right,above,below"
        )

    variant_names = list(dict.fromkeys(
        core.csv_list(args.variants)
    ))
    variants = [
        repair.make_variant(
            name,
            args.start_layer,
            args.end_layer,
        )
        for name in variant_names
    ]

    trace_dir = Path(args.trace_dir)
    trace_summary_path = trace_dir / "summary.json"
    trace_metadata_path = trace_dir / "sample_metadata.jsonl"
    if not trace_summary_path.exists():
        raise FileNotFoundError(trace_summary_path)

    trace_summary = json.loads(
        trace_summary_path.read_text(encoding="utf-8")
    )
    dataset = args.dataset or str(trace_summary["dataset"])
    model_name = args.model or str(trace_summary["model"])

    metadata: List[Dict[str, Any]] = []

    if args.prior_dir:
        prior_dir = Path(args.prior_dir)
        prior_generation_path = prior_dir / "generation.jsonl"
        if not prior_generation_path.exists():
            raise FileNotFoundError(prior_generation_path)

        for original in base.read_jsonl(prior_generation_path):
            gt = base.normalize_relation(original.get("gt"))
            centroid_prediction = base.normalize_relation(
                original.get("centroid_prediction")
            )
            baseline_prediction = base.normalize_relation(
                original.get("baseline_prediction")
            )
            if (
                gt not in relations
                or centroid_prediction not in relations
            ):
                continue

            centroid_state = (
                "centroid_correct"
                if centroid_prediction == gt
                else "centroid_wrong"
            )
            generation_state = (
                "generation_correct"
                if baseline_prediction == gt
                else "generation_wrong"
            )

            row = dict(original)
            row.update({
                "gt": gt,
                "centroid_prediction": centroid_prediction,
                "baseline_prediction": baseline_prediction,
                "baseline_correct": bool(baseline_prediction == gt),
                "centroid_correct": bool(centroid_prediction == gt),
                "group": f"{centroid_state}_{generation_state}",
            })
            metadata.append(row)

        print(
            f"Loaded all-sample prior metadata: {len(metadata)} rows "
            f"from {prior_generation_path}"
        )
    else:
        if not trace_metadata_path.exists():
            raise FileNotFoundError(trace_metadata_path)

        metadata = [
            row
            for row in base.read_jsonl(trace_metadata_path)
            if row.get("group") in (
                base.GROUP_CORRECT,
                base.GROUP_WRONG,
            )
            and base.normalize_relation(row.get("gt")) in relations
        ]
        for row in metadata:
            row["gt"] = base.normalize_relation(row.get("gt"))
            row["centroid_prediction"] = base.normalize_relation(
                row.get("centroid_prediction")
            )
            row["baseline_prediction"] = base.normalize_relation(
                row.get("baseline_prediction")
            )
            row["centroid_correct"] = bool(
                row["centroid_prediction"] == row["gt"]
            )
            row["baseline_correct"] = bool(
                row["baseline_prediction"] == row["gt"]
            )

    metadata = sorted(
        {
            int(row["sid"]): row
            for row in metadata
        }.values(),
        key=lambda row: int(row["sid"]),
    )

    if args.max_per_group is not None:
        rng = random.Random(args.seed)
        selected: List[Dict[str, Any]] = []
        group_names = sorted({
            str(row.get("group", "unknown"))
            for row in metadata
        })
        for group in group_names:
            group_rows = [
                row for row in metadata
                if str(row.get("group", "unknown")) == group
            ]
            rng.shuffle(group_rows)
            selected.extend(group_rows[:args.max_per_group])
        metadata = sorted(
            selected,
            key=lambda row: int(row["sid"]),
        )

    if not metadata:
        raise RuntimeError("No samples selected")

    print("Input quadrants:")
    for key, count in sorted(Counter(
        str(row.get("group", "unknown"))
        for row in metadata
    ).items()):
        print(f"  {key}: {count}")

    logit_rows: Dict[int, Dict[str, Any]] = {}
    if args.logit_results_dir:
        logit_path = (
            Path(args.logit_results_dir) / "samples.jsonl"
        )
        logit_rows = load_jsonl_by_sid(logit_path)
        print(
            f"Loaded logit results: {len(logit_rows)} rows "
            f"from {logit_path}"
        )

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.jsonl"
    errors_path = output_dir / "errors.jsonl"
    for path in (samples_path, errors_path):
        if path.exists():
            path.unlink()

    support = base.import_two_object_module()
    records, audit = support.load_records(
        dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {
        int(record.sid): record
        for record in records
    }

    prompt_args = argparse.Namespace(
        dataset=dataset,
        prompt_jsonl=args.prompt_jsonl,
    )
    prompt_rows = base.load_standard_prompts(
        base.resolve_prompt_path(prompt_args)
    )

    missing = [
        int(row["sid"])
        for row in metadata
        if int(row["sid"]) not in record_by_sid
        or int(row["sid"]) not in prompt_rows
    ]
    if missing:
        raise RuntimeError(
            f"Missing records/prompts for {len(missing)} SIDs; "
            f"first={missing[:10]}"
        )

    if model_name not in support.SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    spec = support.SPECS[model_name]
    model_cls = getattr(
        base.transformers,
        spec.model_class,
        None,
    )
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={base.transformers.__version__} "
            f"has no {spec.model_class}"
        )

    print(f"Loading {model_name}: {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()

    processor = base.AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layers_path = base.resolve_decoder_layers(model)
    if args.end_layer >= len(layers):
        raise RuntimeError(
            f"--end-layer={args.end_layer}, "
            f"model has {len(layers)} layers"
        )

    collector = base.LayerTraceCollector(layers, [])
    attention_modules = list(collector.attention_modules)
    collector.close()

    label_ids = base.label_token_id_variants(
        processor.tokenizer
    )
    relation_vectors = core.readout_vectors(
        model,
        label_ids,
        relations,
    )

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": pad_token_id,
    }
    if processor.tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = (
            processor.tokenizer.eos_token_id
        )

    print("\nGeneration validation")
    print(f"  samples:       {len(metadata)}")
    print(f"  guide:         {args.guide}")
    print(f"  trigger:       {args.trigger}")
    print(f"  hook mode:     {args.hook_mode}")
    print(f"  max new tokens:{args.max_new_tokens}")
    print("  variants:")
    for variant in variants:
        print(
            f"    {variant['name']:24s} "
            f"target={variant['target']:9s} "
            f"mode={variant['mode']:6s} "
            f"layers={variant['layers']}"
        )

    rows: List[Dict[str, Any]] = []
    running_baseline_correct = 0
    running_variant_correct = {
        name: 0 for name in variant_names
    }
    started = time.time()

    for sample_index, metadata_row in enumerate(
        tqdm(
            metadata,
            desc=f"generation-repair:{model_name}",
        ),
        1,
    ):
        sid = int(metadata_row["sid"])
        batch = None
        image = None

        try:
            prompt = prompt_rows[sid]
            gt = base.normalize_relation(
                prompt["answer_raw"]
            )
            guide = (
                metadata_row["centroid_prediction"]
                if args.guide == "centroid"
                else gt
            )
            if guide not in relations:
                raise RuntimeError(
                    f"Invalid guide for sid={sid}: {guide}"
                )

            question = str(prompt["question_text"])
            subject = str(prompt["subject"])
            reference = str(prompt["reference"])

            image = base.record_image(
                record_by_sid[sid]
            )
            batch = base.make_question_batch(
                processor=processor,
                image=image,
                question_text=question,
                device=device,
            )
            prompt_length = int(
                batch["input_ids"].shape[1]
            )

            baseline_sequences = run_baseline_generation(
                model=model,
                batch=batch,
                generation_kwargs=generation_kwargs,
            )
            baseline_text, baseline_token_ids = (
                decode_new_tokens(
                    tokenizer=processor.tokenizer,
                    sequences=baseline_sequences,
                    prompt_length=prompt_length,
                )
            )
            baseline = generation_result(
                text=baseline_text,
                token_ids=baseline_token_ids,
                gt=gt,
            )
            del baseline_sequences

            confidence = safe_float(
                metadata_row.get("centroid_confidence")
            )
            trigger_value = should_trigger_generation(
                trigger=args.trigger,
                prior_group=str(metadata_row["group"]),
                baseline_prediction=baseline["prediction"],
                guide=guide,
                confidence=confidence,
                threshold=args.centroid_confidence_threshold,
                wrong_group_name=base.GROUP_WRONG,
            )

            direction = core.guide_direction(
                relation_vectors,
                guide,
            )

            logit_row = logit_rows.get(sid)
            logit_baseline_prediction = None
            logit_variant_predictions: Dict[str, Optional[str]] = {
                name: None for name in variant_names
            }
            if logit_row is not None:
                logit_baseline_prediction = (
                    logit_row.get("baseline", {})
                    .get("prediction")
                )
                for name in variant_names:
                    logit_variant_predictions[name] = (
                        logit_row.get("variants", {})
                        .get(name, {})
                        .get("prediction")
                    )

            row: Dict[str, Any] = {
                "status": "ok",
                "sid": sid,
                "group": metadata_row["group"],
                "subject": subject,
                "reference": reference,
                "question": question,
                "gt": gt,
                "guide": guide,
                "centroid_prediction":
                    metadata_row["centroid_prediction"],
                "centroid_confidence": confidence,
                "prior_trace_generation_prediction":
                    metadata_row.get("baseline_prediction"),
                "baseline_generation": baseline,
                "logit_baseline_prediction":
                    logit_baseline_prediction,
                "logit_variant_predictions":
                    logit_variant_predictions,
                "variants": {},
            }

            running_baseline_correct += int(
                baseline["correct"]
            )

            for variant in variants:
                name = variant["name"]

                if not trigger_value:
                    result = {
                        **baseline,
                        "triggered": False,
                        "diagnostics": [],
                    }
                else:
                    sequences, diagnostics = (
                        generate_with_intervention(
                            base=base,
                            model=model,
                            batch=batch,
                            layers=layers,
                            attention_modules=attention_modules,
                            variant=variant,
                            direction=direction,
                            strength=args.strength,
                            hook_mode=args.hook_mode,
                            prompt_length=prompt_length,
                            generation_kwargs=generation_kwargs,
                        )
                    )
                    text, token_ids = decode_new_tokens(
                        tokenizer=processor.tokenizer,
                        sequences=sequences,
                        prompt_length=prompt_length,
                    )
                    result = {
                        **generation_result(
                            text=text,
                            token_ids=token_ids,
                            gt=gt,
                        ),
                        "triggered": True,
                        "diagnostics": diagnostics,
                    }
                    del sequences

                row["variants"][name] = result
                running_variant_correct[name] += int(
                    result["correct"]
                )

            rows.append(row)
            append_jsonl(samples_path, row)

            if (
                args.print_every > 0
                and (
                    sample_index == 1
                    or sample_index % args.print_every == 0
                    or sample_index == len(metadata)
                )
            ):
                tqdm.write(
                    f"\n[{sample_index}/{len(metadata)}] "
                    f"sid={sid}\n"
                    f"  Q: {question}\n"
                    f"  GT={gt} | guide={guide}\n"
                    f"  baseline text={baseline['text']!r}\n"
                    f"  baseline pred={baseline['prediction']} | "
                    f"acc="
                    f"{running_baseline_correct/sample_index:.4f}"
                )

                for name in variant_names:
                    result = row["variants"][name]
                    mark = (
                        "REPAIRED"
                        if (
                            not baseline["correct"]
                            and result["correct"]
                        )
                        else "DAMAGED"
                        if (
                            baseline["correct"]
                            and not result["correct"]
                        )
                        else "-"
                    )
                    logit_prediction = (
                        logit_variant_predictions.get(name)
                    )
                    tqdm.write(
                        f"  {name:24s} "
                        f"text={result['text']!r} | "
                        f"pred={str(result['prediction']):5s} | "
                        f"logit={str(logit_prediction):5s} | "
                        f"acc="
                        f"{running_variant_correct[name]/sample_index:.4f} "
                        f"| {mark}"
                    )

        except Exception as exc:
            error = {
                "status": "error",
                "sid": sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            rows.append(error)
            append_jsonl(samples_path, error)
            append_jsonl(errors_path, {
                **error,
                "traceback_tail":
                    traceback.format_exc().splitlines()[-30:],
            })
            tqdm.write(
                f"\n[ERROR] sid={sid}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            if batch is not None:
                del batch
            if image is not None:
                del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_rows = [
        summarize_variant(rows, name)
        for name in variant_names
    ]
    summary_rows.sort(
        key=lambda row: (
            int(row.get("net_repair", -10**9)),
            float(
                row.get(
                    "intervention_generation_accuracy",
                    -math.inf,
                )
            ),
        ),
        reverse=True,
    )

    base.write_csv(
        output_dir / "summary.csv",
        summary_rows,
    )
    base.write_csv(
        output_dir / "summary_by_relation.csv",
        summarize_by_relation(
            rows,
            variant_names,
            relations,
        ),
    )
    base.write_csv(
        output_dir / "summary_by_centroid_status.csv",
        summarize_by_centroid_status(
            rows,
            variant_names,
        ),
    )

    usable = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]
    baseline_generation_accuracy = safe_mean(
        float(row["baseline_generation"]["correct"])
        for row in usable
    )
    baseline_parse_rate = safe_mean(
        float(row["baseline_generation"]["parsed"])
        for row in usable
    )
    trace_generation_agreement = safe_mean(
        float(
            row["baseline_generation"]["prediction"]
            == row["prior_trace_generation_prediction"]
        )
        for row in usable
        if row["baseline_generation"]["prediction"]
        is not None
        and row["prior_trace_generation_prediction"]
        is not None
    )
    centroid_gt_agreement = safe_mean(
        float(row["centroid_prediction"] == row["gt"])
        for row in usable
    )

    summary_json = {
        "script_version": SCRIPT_VERSION,
        "base_module": base.__name__,
        "trace_dir": str(trace_dir),
        "prior_dir": args.prior_dir,
        "logit_results_dir": args.logit_results_dir,
        "dataset": dataset,
        "model": model_name,
        "guide": args.guide,
        "trigger": args.trigger,
        "hook_mode": args.hook_mode,
        "strength": args.strength,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "max_new_tokens": args.max_new_tokens,
        "variants": variants,
        "n_requested": len(metadata),
        "n_successful": len(usable),
        "baseline_generation_parse_rate":
            baseline_parse_rate,
        "baseline_generation_accuracy":
            baseline_generation_accuracy,
        "baseline_current_vs_prior_trace_agreement":
            trace_generation_agreement,
        "centroid_gt_agreement":
            centroid_gt_agreement,
        "best_variant":
            summary_rows[0] if summary_rows else None,
        "variant_summaries": summary_rows,
        "decoder_layers_path": layers_path,
        "elapsed_minutes":
            (time.time() - started) / 60.0,
        "audit": audit,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary_json,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 154)
    print("AUTOREGRESSIVE GENERATION REPAIR VALIDATION")
    print("=" * 154)
    print(
        f"baseline generation parse rate: "
        f"{baseline_parse_rate:.4f}"
    )
    print(
        f"baseline generation accuracy:   "
        f"{baseline_generation_accuracy:.4f}"
    )
    print(
        f"baseline vs prior trace agree:  "
        f"{trace_generation_agreement:.4f}"
    )
    print(
        f"centroid/GT agreement:           "
        f"{centroid_gt_agreement:.4f}"
    )
    print("")
    print(
        "variant                  | base_gen | new_gen  | delta    | "
        "repaired | damaged | net | gen-logit baseline | "
        "gen-logit intervention | change consistency"
    )
    print("-" * 154)

    for row in summary_rows:
        print(
            f"{row['variant']:24s} | "
            f"{row['baseline_generation_accuracy']:.4f}   | "
            f"{row['intervention_generation_accuracy']:.4f}   | "
            f"{row['generation_accuracy_change']:+.4f} | "
            f"{row['repaired_wrong']:8d} | "
            f"{row['damaged_correct']:7d} | "
            f"{row['net_repair']:3d} | "
            f"{row['baseline_generation_logit_agreement']:.4f}             | "
            f"{row['intervention_generation_logit_agreement']:.4f}                 | "
            f"{row['generation_logit_change_consistency']:.4f}"
        )

    print("\nBest variant:")
    print(json.dumps(
        summary_rows[0] if summary_rows else {},
        ensure_ascii=False,
        indent=2,
    ))

    print("\nSaved outputs:")
    for filename in (
        "samples.jsonl",
        "summary.csv",
        "summary_by_relation.csv",
        "summary_by_centroid_status.csv",
        "summary.json",
    ):
        print(f"  {output_dir / filename}")
    if errors_path.exists():
        print(f"  {errors_path}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
