#!/usr/bin/env python3
"""
Full autoregressive-generation grid search for the validated
P_POS7/P_NEG5 -> L26VH0 circuit in Qwen2.5-VL on COCO two-object data.

Unlike prompt-last closed-set scoring, this script runs model.generate() and
parses the complete generated continuation. The intervention is applied to the
L26 V projection during the PREFILL pass only, so the repaired object-token
values are written into the KV cache and remain available to every later
autoregressive decoding step.

For each sample, the script first estimates path-specific receiver states:

    V_base
    V_without_positive
    V_without_negative

using the same residual/MLP-mediated, intermediate-attention-frozen path
ablation as analyze_coco_circuit_failure_repair_v1.py. For every alpha/beta
combination it then constructs

    V_repair = V_base
             + alpha * (V_base - V_without_positive)
             - beta  * (V_base - V_without_negative)

and generates a complete answer with greedy decoding.

This script deliberately supports an exploratory "select the best setting on
all selected samples" mode. That is useful for determining whether the circuit
can improve free generation at all, but the selected accuracy is tuned on the
same data and is not an unbiased test estimate.

Expected companion files in the repository root:
  analyze_coco_circuit_failure_repair_v1.py
  analyze_coco_ioi_backward_circuit_v1.py
  analyze_coco_producer_qk_ov_v1.py
  analyze_coco_receiver_qkv_v1.py
  analyze_spatial_storage_transport_utilization_v3.py
  analyze_coco_centroid_generation_step1_v4.py
  analyze_coco_flip_attention_spatial_vectors_v1.py
  coco_ioi_role_bundles_v1.json

Outputs:
  generation_grid_effect.jsonl       one SID x alpha x beta result per line
  generation_grid_summary.csv        full free-generation accuracy for each pair
  generation_relation_summary.csv    per-relation accuracy for each pair
  best_generation_repair.json        highest-accuracy grid point on selected data
  config.json
  errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import json
import math
import random
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-circuit-generation-repair-grid-v1"
RELATIONS = ("left", "right", "above", "below")
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True)
    p.add_argument("--source-output-dir", required=True)
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", choices=("last", "mean"), default="last")

    p.add_argument("--bundle-json", default="coco_ioi_role_bundles_v1.json")
    p.add_argument("--positive-bundle", default="P_POS7")
    p.add_argument("--negative-bundle", default="P_NEG5")
    p.add_argument("--receiver-layer", type=int, default=26)
    p.add_argument("--receiver-query-head", type=int, default=0)
    p.add_argument("--receiver-channel", choices=("v",), default="v")
    p.add_argument("--receiver-kv-scope", choices=("objects",), default="objects")
    p.add_argument(
        "--sender-object-positions",
        choices=("last", "all"),
        default="last",
    )

    p.add_argument("--sample-status", choices=STATUSES, default="all")
    p.add_argument(
        "--sample-max-samples",
        type=int,
        default=0,
        help="0 means all eligible source rows.",
    )
    p.add_argument(
        "--exclude-sids-from",
        default="",
        help="Optional comma-separated files containing SIDs to exclude.",
    )
    p.add_argument(
        "--include-sids-file",
        default="",
        help="Optional file containing the only SIDs to include.",
    )

    p.add_argument(
        "--alpha-grid",
        default="0,0.5,1.0,1.5,2.0",
        help="Comma-separated positive-circuit amplification coefficients.",
    )
    p.add_argument(
        "--beta-grid",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated negative-circuit suppression coefficients.",
    )

    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--generation-do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument(
        "--expected-source-baseline-accuracy",
        type=float,
        default=0.7114,
        help="Only used to print a warning when generation reproduction differs.",
    )
    p.add_argument("--baseline-warning-tolerance", type=float, default=0.03)

    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument(
        "--failure-script",
        default="analyze_coco_circuit_failure_repair_v1.py",
    )
    p.add_argument(
        "--ioi-script",
        default="analyze_coco_ioi_backward_circuit_v1.py",
    )
    p.add_argument(
        "--producer-script",
        default="analyze_coco_producer_qk_ov_v1.py",
    )
    p.add_argument(
        "--receiver-script",
        default="analyze_coco_receiver_qkv_v1.py",
    )
    p.add_argument(
        "--v3-script",
        default="analyze_spatial_storage_transport_utilization_v3.py",
    )
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument(
        "--attention-helper",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
    )

    # Compatibility with imported helper functions.
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def import_file(path: Path, module_name: str) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def parse_float_grid(value: str, label: str) -> List[float]:
    result: List[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{label} values must be finite and >= 0: {item}")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError(f"No values supplied for {label}")
    if 0.0 not in result:
        result.insert(0, 0.0)
    return result


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def extract_sids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    result: set[int] = set()
    if path.suffix.lower() == ".jsonl":
        for row in read_jsonl(path):
            if "sid" in row:
                result.add(int(row["sid"]))
        return result
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("sid", "")).strip():
                    result.add(int(row["sid"]))
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
                if isinstance(value, Mapping) and "sid" in value:
                    result.add(int(value["sid"]))
                else:
                    result.add(int(value))
            except json.JSONDecodeError:
                result.add(int(text.split(",", 1)[0]))
    return result


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def source_original_correct(row: Mapping[str, Any]) -> Optional[bool]:
    status = str(row.get("generation_pair_status", ""))
    if status in {"both_correct", "original_only"}:
        return True
    if status in {"swapped_only", "both_wrong"}:
        return False
    for key in (
        "original_generation_correct",
        "generation_original_correct",
        "original_correct",
    ):
        if key in row:
            return bool(row[key])
    return None


# -----------------------------------------------------------------------------
# Generated-text parsing
# -----------------------------------------------------------------------------


RELATION_PATTERNS: Mapping[str, Sequence[str]] = {
    "left": (
        r"\bto\s+the\s+left\s+of\b",
        r"\bon\s+the\s+left\s+of\b",
        r"\bleft\s+of\b",
        r"\bleft\b",
    ),
    "right": (
        r"\bto\s+the\s+right\s+of\b",
        r"\bon\s+the\s+right\s+of\b",
        r"\bright\s+of\b",
        r"\bright\b",
    ),
    "above": (
        r"\bon\s+top\s+of\b",
        r"\babove\b",
        r"\bover\b",
    ),
    "below": (
        r"\bunderneath\b",
        r"\bbeneath\b",
        r"\bbelow\b",
        r"\bunder\b",
    ),
}


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "left_of": "left",
        "right_of": "right",
        "on": "above",
        "under": "below",
    }
    if text in RELATIONS:
        return text
    if text in aliases:
        return aliases[text]
    return None


def parse_generated_relation(text: str) -> Optional[str]:
    """Parse the earliest explicit spatial relation in the generated suffix."""
    clean = str(text).lower().replace("_", " ")

    # Prefer text after an explicit answer marker when present.
    marker_matches = list(
        re.finditer(
            r"(?:final\s+answer|answer|relation(?:ship)?)\s*(?:is|:|=)\s*",
            clean,
        )
    )
    regions = []
    if marker_matches:
        regions.append(clean[marker_matches[-1].end() :])
    regions.append(clean)

    for region in regions:
        candidates: List[Tuple[int, int, str]] = []
        for relation, patterns in RELATION_PATTERNS.items():
            for priority, pattern in enumerate(patterns):
                match = re.search(pattern, region)
                if match is not None:
                    candidates.append((int(match.start()), int(priority), relation))
                    break
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], RELATIONS.index(item[2])))
            return candidates[0][2]
    return None


# -----------------------------------------------------------------------------
# Prefill V patch and autoregressive generation
# -----------------------------------------------------------------------------


class PrefillProjectionHeadPatch:
    """Patch one V-head slice only on the full-prompt prefill forward."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        head: int,
        head_dim: int,
        target_to_source: Mapping[int, torch.Tensor],
    ) -> None:
        self.head = int(head)
        self.head_dim = int(head_dim)
        self.target_to_source = {
            int(position): tensor.detach().float().cpu()
            for position, tensor in target_to_source.items()
        }
        if not self.target_to_source:
            raise ValueError("No receiver positions to patch")
        self.max_position = max(self.target_to_source)
        self.prefill_events = 0
        self.decode_events = 0
        self.positions_patched = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not torch.is_tensor(output) or output.ndim != 3:
            raise RuntimeError("V projection output must be [B,S,D]")
        if int(output.shape[0]) != 1:
            raise RuntimeError("Generation patch expects batch size 1")

        sequence_length = int(output.shape[1])
        # Decode steps usually have S=1. Only the prefill sequence contains all
        # original object-token positions.
        if sequence_length <= self.max_position:
            self.decode_events += 1
            return output

        selected_start = self.head * self.head_dim
        selected_stop = selected_start + self.head_dim
        if selected_stop > int(output.shape[-1]):
            raise RuntimeError(
                f"Head slice {selected_start}:{selected_stop} exceeds "
                f"V projection width {int(output.shape[-1])}"
            )

        modified = output.clone()
        count = 0
        for position, full_state in self.target_to_source.items():
            if not 0 <= position < sequence_length:
                raise RuntimeError(
                    f"Receiver position {position} outside prefill length {sequence_length}"
                )
            source = full_state[selected_start:selected_stop].to(
                device=output.device,
                dtype=output.dtype,
            )
            modified[0, position, selected_start:selected_stop] = source
            count += 1
        self.prefill_events += 1
        self.positions_patched += count
        return modified

    def validate(self) -> None:
        if self.prefill_events != 1:
            raise RuntimeError(
                f"Expected exactly one patched prefill event, got {self.prefill_events}"
            )
        if self.positions_patched != len(self.target_to_source):
            raise RuntimeError(
                f"Patched {self.positions_patched} positions; expected "
                f"{len(self.target_to_source)}"
            )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


@torch.inference_mode()
def generate_answer(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
    patch_module: Optional[torch.nn.Module] = None,
    patch_head: Optional[int] = None,
    patch_head_dim: Optional[int] = None,
    patch_states: Optional[Mapping[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    patch: Optional[PrefillProjectionHeadPatch] = None
    if patch_states is not None:
        if patch_module is None or patch_head is None or patch_head_dim is None:
            raise ValueError("Patch states require module/head/head_dim")
        patch = PrefillProjectionHeadPatch(
            module=patch_module,
            head=int(patch_head),
            head_dim=int(patch_head_dim),
            target_to_source=patch_states,
        )

    input_ids = batch.get("input_ids")
    if input_ids is None or not torch.is_tensor(input_ids):
        raise RuntimeError("Generation batch must contain input_ids")
    prompt_length = int(input_ids.shape[1])

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.generation_do_sample),
        "num_beams": int(args.num_beams),
        "use_cache": True,
        "return_dict_in_generate": False,
    }
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.generation_do_sample:
        generation_kwargs["temperature"] = float(args.temperature)
        generation_kwargs["top_p"] = float(args.top_p)
        if int(args.top_k) > 0:
            generation_kwargs["top_k"] = int(args.top_k)

    started = time.perf_counter()
    try:
        generated_ids = model.generate(**dict(batch), **generation_kwargs)
        if patch is not None:
            patch.validate()
    finally:
        if patch is not None:
            patch.close()

    elapsed = time.perf_counter() - started
    if not torch.is_tensor(generated_ids) or generated_ids.ndim != 2:
        raise RuntimeError("model.generate did not return a [B,S] tensor")
    new_ids = generated_ids[0, prompt_length:].detach().cpu().tolist()
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    prediction = parse_generated_relation(text)
    return {
        "text": text,
        "prediction": prediction,
        "parsed": prediction is not None,
        "new_token_count": len(new_ids),
        "generation_seconds": float(elapsed),
    }


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize_grid(
    rows: Sequence[Mapping[str, Any]],
    selected_sids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selected_set = set(map(int, selected_sids))
    groups: Dict[Tuple[float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = int(row["sid"])
        if sid in selected_set:
            groups[(float(row["alpha"]), float(row["beta"]))].append(row)

    baseline_rows = groups.get((0.0, 0.0), [])
    baseline_by_sid = {
        int(row["sid"]): bool(row["correct"])
        for row in baseline_rows
    }
    baseline_prediction_by_sid = {
        int(row["sid"]): row.get("prediction")
        for row in baseline_rows
    }

    summary: List[Dict[str, Any]] = []
    relation_summary: List[Dict[str, Any]] = []
    for (alpha, beta), values in sorted(groups.items()):
        values_by_sid = {int(row["sid"]): row for row in values}
        common_sids = sorted(selected_set & set(values_by_sid) & set(baseline_by_sid))
        fixes = 0
        breaks = 0
        unchanged_correct = 0
        unchanged_wrong = 0
        parsed = 0
        correct = 0
        generation_times = []
        for sid in common_sids:
            row = values_by_sid[sid]
            before = baseline_by_sid[sid]
            after = bool(row["correct"])
            parsed += int(bool(row["parsed"]))
            correct += int(after)
            fixes += int((not before) and after)
            breaks += int(before and (not after))
            unchanged_correct += int(before and after)
            unchanged_wrong += int((not before) and (not after))
            generation_times.append(float(row.get("generation_seconds", float("nan"))))

        n = len(common_sids)
        baseline_correct = sum(int(baseline_by_sid[sid]) for sid in common_sids)
        summary.append(
            {
                "alpha": alpha,
                "beta": beta,
                "N": n,
                "coverage": n / max(len(selected_set), 1),
                "parse_rate": parsed / n if n else float("nan"),
                "baseline_accuracy": baseline_correct / n if n else float("nan"),
                "generation_accuracy": correct / n if n else float("nan"),
                "accuracy_delta": (correct - baseline_correct) / n if n else float("nan"),
                "fixes": fixes,
                "breaks": breaks,
                "net_fixes": fixes - breaks,
                "unchanged_correct": unchanged_correct,
                "unchanged_wrong": unchanged_wrong,
                "fix_rate_among_baseline_wrong": (
                    fixes / max(n - baseline_correct, 1)
                ),
                "preserve_rate_among_baseline_correct": (
                    unchanged_correct / max(baseline_correct, 1)
                ),
                "mean_generation_seconds": safe_mean(generation_times),
                "intervention_magnitude": alpha + beta,
            }
        )

        for relation in RELATIONS:
            subset = [row for row in values if str(row["gt"]) == relation]
            relation_summary.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "relation": relation,
                    "N": len(subset),
                    "parse_rate": safe_mean(int(bool(row["parsed"])) for row in subset),
                    "accuracy": safe_mean(int(bool(row["correct"])) for row in subset),
                }
            )

    if not summary:
        raise RuntimeError("No complete grid rows available for summary")

    max_n = max(int(row["N"]) for row in summary)
    eligible = [row for row in summary if int(row["N"]) == max_n]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["generation_accuracy"]),
            -float(row["parse_rate"]),
            -int(row["net_fixes"]),
            int(row["breaks"]),
            float(row["intervention_magnitude"]),
            float(row["alpha"]),
            float(row["beta"]),
        ),
    )
    for rank, row in enumerate(
        sorted(
            summary,
            key=lambda item: (
                -int(item["N"]),
                -float(item["generation_accuracy"]),
                int(item["breaks"]),
            ),
        ),
        start=1,
    ):
        row["rank"] = rank

    best = dict(ranked[0])
    best.update(
        {
            "script_version": SCRIPT_VERSION,
            "selection_mode": "exploratory maximum on all selected samples",
            "selection_is_unbiased_test_estimate": False,
            "selected_sample_count": len(selected_set),
            "maximum_complete_N": max_n,
            "baseline_prediction_count": len(baseline_prediction_by_sid),
        }
    )
    return summary, relation_summary, best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_beams <= 0:
        raise ValueError("--num-beams must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    alphas = parse_float_grid(args.alpha_grid, "alpha grid")
    betas = parse_float_grid(args.beta_grid, "beta grid")
    grid = [(float(alpha), float(beta)) for alpha in alphas for beta in betas]

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failure = import_file(Path(args.failure_script), "generation_repair_failure")
    ioi = import_file(Path(args.ioi_script), "generation_repair_ioi")
    producer = import_file(Path(args.producer_script), "generation_repair_producer")
    receiver = import_file(Path(args.receiver_script), "generation_repair_receiver")
    v3 = import_file(Path(args.v3_script), "generation_repair_v3")
    base = import_file(Path(args.base_script), "generation_repair_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "generation_repair_attention",
    )

    source_config, source_rows = ioi.load_source_rows(args)

    excluded: set[int] = set()
    for item in str(args.exclude_sids_from).split(","):
        item = item.strip()
        if item:
            excluded.update(extract_sids(Path(item)))
    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(args.include_sids_file.strip()))

    selected_rows = [
        dict(row)
        for row in source_rows
        if (args.sample_status == "all" or str(row.get("generation_pair_status")) == args.sample_status)
        and int(row["sid"]) not in excluded
        and (included is None or int(row["sid"]) in included)
    ]
    selected_rows = failure.deterministic_stratified_limit(
        selected_rows,
        int(args.sample_max_samples),
        int(args.seed),
    )
    selected_rows = sorted(selected_rows, key=lambda row: int(row["sid"]))
    if not selected_rows:
        raise RuntimeError("No eligible samples")

    bundles = failure.load_named_bundles(
        Path(args.bundle_json),
        [args.positive_bundle, args.negative_bundle],
        ioi,
    )
    positive_bundle = bundles[args.positive_bundle]
    negative_bundle = bundles[args.negative_bundle]

    model = None
    processor = None
    try:
        (
            model,
            processor,
            spec,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer.load_model_bundle(args=args, base=base)

        receiver_layer = int(args.receiver_layer)
        writer = ioi.WriterNode("attention", receiver_layer, int(args.receiver_query_head))
        units = ioi.build_receiver_units(
            writers=[writer],
            channels=[str(args.receiver_channel)],
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
        )
        if len(units) != 1:
            raise RuntimeError(f"Expected one receiver unit, got {len(units)}")
        unit = units[0]

        attention = attention_helper.resolve_self_attention(
            decoder_layers[int(unit.layer)]
        )
        shape = receiver.resolve_attention_shape(attention)
        projection = receiver.projection_module(attention, unit.channel)
        patch_head_dim = int(shape.kv_head_dim)

        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        source_baseline_values = [
            source_original_correct(row)
            for row in selected_rows
        ]
        source_baseline_known = [value for value in source_baseline_values if value is not None]
        source_baseline_accuracy = (
            safe_mean(int(value) for value in source_baseline_known)
            if source_baseline_known
            else float("nan")
        )

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "decoder_path": decoder_path,
            "positive_bundle": {
                "name": positive_bundle.name,
                "heads": list(positive_bundle.head_names),
            },
            "negative_bundle": {
                "name": negative_bundle.name,
                "heads": list(negative_bundle.head_names),
            },
            "receiver": {
                "unit": unit.unit,
                "layer": int(unit.layer),
                "kv_head": int(unit.kv_head),
                "unit_head": int(unit.unit_head),
                "shared_query_heads": list(unit.shared_query_heads),
                "channel": unit.channel,
                "kv_scope": args.receiver_kv_scope,
            },
            "repair_formula": "V_base + alpha*(V_base-V_without_POS7) - beta*(V_base-V_without_NEG5)",
            "generation_intervention": "patch V projection on full-prompt prefill only; repaired values enter KV cache",
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.generation_do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
            },
            "alpha_grid": alphas,
            "beta_grid": betas,
            "grid_size": len(grid),
            "sample_status": args.sample_status,
            "selected_samples": len(selected_rows),
            "selected_sids": [int(row["sid"]) for row in selected_rows],
            "excluded_sids": sorted(excluded),
            "source_cached_original_generation_accuracy": source_baseline_accuracy,
            "selection_mode": "best alpha/beta on all selected samples; exploratory/oracle",
            "seed": args.seed,
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        effect_path = output_dir / "generation_grid_effect.jsonl"
        errors_path = output_dir / "errors.jsonl"
        existing = read_jsonl(effect_path) if args.resume else []
        completed = {
            (int(row["sid"]), float(row["alpha"]), float(row["beta"]))
            for row in existing
        }
        grid_set = set(grid)

        pending_rows = []
        for row in selected_rows:
            sid = int(row["sid"])
            if any((sid, alpha, beta) not in completed for alpha, beta in grid_set):
                pending_rows.append(row)

        print(
            "Free-generation circuit grid: "
            f"requested_N={len(selected_rows)}, pending_N={len(pending_rows)}, "
            f"existing_rows={len(existing)}, grid={len(alphas)}x{len(betas)}="
            f"{len(grid)}, expected_total_rows={len(selected_rows) * len(grid)}, "
            f"source_cached_baseline_acc={source_baseline_accuracy:.4f}",
            flush=True,
        )

        capture_layers = list(range(receiver_layer + 1))

        for sample_index, source_row in enumerate(
            tqdm(pending_rows, desc=f"generation-grid:{args.model}"),
            start=1,
        ):
            pair = None
            try:
                sid = int(source_row["sid"])
                missing_grid = [
                    (alpha, beta)
                    for alpha, beta in grid
                    if (sid, alpha, beta) not in completed
                ]
                if not missing_grid:
                    continue

                pair = receiver.prepare_pair(
                    args=args,
                    row=source_row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=torch.device(args.device),
                )

                if args.sender_object_positions == "last":
                    if not pair.original_a_positions or not pair.original_b_positions:
                        raise RuntimeError("Missing original object positions")
                    sender_positions = sorted(
                        {
                            int(pair.original_a_positions[-1]),
                            int(pair.original_b_positions[-1]),
                        }
                    )
                else:
                    sender_positions = list(map(int, pair.original_object_positions))
                receiver_positions = list(map(int, pair.original_object_positions))
                attention_positions = sorted(set(sender_positions) | set(receiver_positions))

                _, clean_capture, baseline_states = failure.capture_clean_original(
                    pair=pair,
                    capture_layers=capture_layers,
                    attention_positions=attention_positions,
                    receiver_positions=receiver_positions,
                    unit=unit,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver,
                    attention_helper=attention_helper,
                    ioi=ioi,
                )
                positive_removed_states = failure.run_bundle_removal_c_pass(
                    bundle=positive_bundle,
                    sender_positions=sender_positions,
                    receiver_positions=receiver_positions,
                    unit=unit,
                    pair=pair,
                    clean_capture=clean_capture,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver,
                    attention_helper=attention_helper,
                    ioi=ioi,
                )
                negative_removed_states = failure.run_bundle_removal_c_pass(
                    bundle=negative_bundle,
                    sender_positions=sender_positions,
                    receiver_positions=receiver_positions,
                    unit=unit,
                    pair=pair,
                    clean_capture=clean_capture,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver,
                    attention_helper=attention_helper,
                    ioi=ioi,
                )

                baseline_v = baseline_states[int(unit.layer)][unit.channel]
                without_positive_v = positive_removed_states[int(unit.layer)][unit.channel]
                without_negative_v = negative_removed_states[int(unit.layer)][unit.channel]

                baseline_generation: Optional[Dict[str, Any]] = None
                if (0.0, 0.0) in missing_grid or any(
                    (sid, 0.0, 0.0) == key for key in completed
                ):
                    if (sid, 0.0, 0.0) in completed:
                        matching = [
                            row
                            for row in existing
                            if int(row["sid"]) == sid
                            and float(row["alpha"]) == 0.0
                            and float(row["beta"]) == 0.0
                        ]
                        if matching:
                            row = matching[-1]
                            baseline_generation = {
                                "text": row.get("generation", ""),
                                "prediction": row.get("prediction"),
                                "parsed": bool(row.get("parsed")),
                                "new_token_count": int(row.get("new_token_count", 0)),
                                "generation_seconds": float(row.get("generation_seconds", 0.0)),
                            }
                    if baseline_generation is None:
                        baseline_generation = generate_answer(
                            model=model,
                            processor=processor,
                            batch=pair.original_batch,
                            args=args,
                        )

                for alpha, beta in missing_grid:
                    key = (sid, float(alpha), float(beta))
                    if key in completed:
                        continue

                    if alpha == 0.0 and beta == 0.0:
                        if baseline_generation is None:
                            baseline_generation = generate_answer(
                                model=model,
                                processor=processor,
                                batch=pair.original_batch,
                                args=args,
                            )
                        generated = baseline_generation
                    else:
                        repaired_states = failure.combine_receiver_states(
                            baseline=baseline_v,
                            without_positive=without_positive_v,
                            without_negative=without_negative_v,
                            alpha=float(alpha),
                            beta=float(beta),
                        )
                        generated = generate_answer(
                            model=model,
                            processor=processor,
                            batch=pair.original_batch,
                            args=args,
                            patch_module=projection,
                            patch_head=int(unit.unit_head),
                            patch_head_dim=patch_head_dim,
                            patch_states=repaired_states,
                        )

                    prediction = normalize_relation(generated["prediction"])
                    gt = str(pair.gt)
                    row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": gt,
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "generation": generated["text"],
                        "prediction": prediction,
                        "parsed": prediction is not None,
                        "correct": bool(prediction == gt),
                        "new_token_count": int(generated["new_token_count"]),
                        "generation_seconds": float(generated["generation_seconds"]),
                        "source_generation_pair_status": source_row.get(
                            "generation_pair_status", "unknown"
                        ),
                        "source_original_correct": source_original_correct(source_row),
                        "sender_positions": sender_positions,
                        "receiver_positions": receiver_positions,
                        "receiver_unit": unit.unit,
                        "cache_intervention": "prefill V projection patch only",
                    }
                    append_jsonl(effect_path, row)
                    existing.append(row)
                    completed.add(key)

                if args.print_every > 0 and sample_index % args.print_every == 0:
                    print(
                        f"[sample {sample_index}/{len(pending_rows)} sid={sid}] "
                        f"saved_rows={len(existing)}",
                        flush=True,
                    )
            except Exception as exc:
                error = {
                    "script_version": SCRIPT_VERSION,
                    "sid": None if pair is None else int(pair.sid),
                    "source_sid": int(source_row.get("sid", -1)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(errors_path, error)
                print(
                    f"[ERROR sid={source_row.get('sid')}] "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    receiver.release_pair(pair)
                gc.collect()
                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        final_rows = read_jsonl(effect_path)
        selected_sids = [int(row["sid"]) for row in selected_rows]
        summary, relation_summary, best = summarize_grid(final_rows, selected_sids)
        write_csv(output_dir / "generation_grid_summary.csv", summary)
        write_csv(output_dir / "generation_relation_summary.csv", relation_summary)

        baseline_rows = [
            row
            for row in final_rows
            if int(row["sid"]) in set(selected_sids)
            and float(row["alpha"]) == 0.0
            and float(row["beta"]) == 0.0
        ]
        baseline_accuracy = safe_mean(int(bool(row["correct"])) for row in baseline_rows)
        baseline_parse_rate = safe_mean(int(bool(row["parsed"])) for row in baseline_rows)
        agreement_rows = [
            row
            for row in baseline_rows
            if row.get("source_original_correct") is not None
        ]
        source_agreement = safe_mean(
            int(bool(row["correct"]) == bool(row["source_original_correct"]))
            for row in agreement_rows
        )
        best.update(
            {
                "generated_baseline_accuracy": baseline_accuracy,
                "generated_baseline_parse_rate": baseline_parse_rate,
                "source_cached_baseline_accuracy": source_baseline_accuracy,
                "generated_vs_source_correctness_agreement": source_agreement,
                "expected_source_baseline_accuracy": args.expected_source_baseline_accuracy,
            }
        )
        write_json(output_dir / "best_generation_repair.json", best)

        print("\n" + "=" * 120)
        print("FREE-GENERATION GRID RESULT")
        print("=" * 120)
        print(
            f"Samples: {len(selected_rows)} | grid: {len(alphas)}x{len(betas)}="
            f"{len(grid)} | rows: {len(final_rows)}"
        )
        print(
            f"Generated baseline: parse={baseline_parse_rate:.4f} "
            f"accuracy={baseline_accuracy:.4f}"
        )
        print(
            f"Source cached baseline accuracy: {source_baseline_accuracy:.4f} | "
            f"correctness agreement={source_agreement:.4f}"
        )
        print(
            f"BEST ON SAME DATA: alpha={best['alpha']:.4f} beta={best['beta']:.4f} "
            f"accuracy={best['generation_accuracy']:.4f} "
            f"delta={best['accuracy_delta']:+.4f} fixes={best['fixes']} "
            f"breaks={best['breaks']} net={best['net_fixes']} "
            f"parse={best['parse_rate']:.4f}"
        )
        print(
            "NOTE: the best pair is selected on these same samples; treat it as "
            "exploratory/oracle until fixed on a separate dataset or split."
        )

        expected = float(args.expected_source_baseline_accuracy)
        tolerance = float(args.baseline_warning_tolerance)
        if math.isfinite(baseline_accuracy) and abs(baseline_accuracy - expected) > tolerance:
            print(
                "WARNING: generated baseline differs from the expected baseline by "
                f"{baseline_accuracy - expected:+.4f}. Check generation settings and parser."
            )

        print(f"\nSaved outputs to {output_dir}", flush=True)
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
