#!/usr/bin/env python3
"""
Free-generation tests for two hypotheses about the validated COCO/Qwen2.5-VL
spatial circuit P_POS7 -> L26VH0.

Experiment A: bundle contribution sign flip
--------------------------------------------
For a sender bundle B, estimate its path-specific contribution to L26VH0:

    delta_B = V_base - V_without_B

Then patch the generation-prefill V cache with:

    V(gamma) = V_base - gamma * delta_B

Interpretation:
    gamma = 0   normal model
    gamma = 1   path-specific bundle ablation
    gamma = 2   full contribution sign flip
    gamma > 2   amplified reverse contribution

The script evaluates complete greedy autoregressive generation, not only the
prompt-last four-way relation logits.

An optional two-bundle grid jointly applies:

    V(gamma_pos, gamma_neg)
        = V_base
        - gamma_pos * delta_POS7
        - gamma_neg * delta_NEG5

Experiment B: query-swapped stage-wise restoration
--------------------------------------------------
For samples where the original query is wrong but the role-swapped query is
correct (source status `swapped_only` / WC), capture original and swapped
states at several stages.  At each stage, construct a reflected state:

    h_restore(lambda) = h_original + lambda * (h_original - h_swapped_aligned)

This moves the original computation away from the correctly-computed swapped
relation state, which represents the opposite relation.  Object-position
states are tested with two alignments:

    identity: original A <- swapped A, original B <- swapped B
    role:     original A <- swapped B, original B <- swapped A

Prompt-last states require no object alignment.  The intervention is applied
on the full-prompt prefill pass and is therefore inherited by subsequent
self-attention KV caches and free generation.

This is a mechanistic analysis script.  The best setting selected on the same
samples is exploratory/oracle, not an unbiased test estimate.

Expected companion files in the repository root:
  analyze_coco_circuit_failure_repair_v1.py
  analyze_coco_circuit_generation_repair_grid_v1.py
  analyze_coco_ioi_backward_circuit_v1.py
  analyze_coco_producer_qk_ov_v1.py
  analyze_coco_receiver_qkv_v1.py
  analyze_spatial_storage_transport_utilization_v3.py
  analyze_coco_centroid_generation_step1_v4.py
  analyze_coco_flip_attention_spatial_vectors_v1.py
  coco_ioi_role_bundles_v1.json

Outputs:
  bundle_signflip_generation.jsonl
  bundle_signflip_summary.csv
  combined_signflip_generation.jsonl
  combined_signflip_summary.csv
  query_swap_restore_generation.jsonl
  query_swap_restore_summary.csv
  query_swap_stage_best.csv
  summary.json
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-bundle-signflip-queryswap-restore-v1"
RELATIONS = ("left", "right", "above", "below")
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}


# -----------------------------------------------------------------------------
# CLI / generic utilities
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

    p.add_argument(
        "--experiments",
        default="bundle_flip,query_swap_restore",
        help="Comma-separated: bundle_flip,combined_flip,query_swap_restore.",
    )
    p.add_argument("--bundle-json", default="coco_ioi_role_bundles_v1.json")
    p.add_argument(
        "--flip-bundles",
        default="P_OLD5,P_NEW2,P_POS7,P_NEG5,P_ALL12",
    )
    p.add_argument(
        "--gamma-grid",
        default="0,0.5,1,1.5,2,2.5,3",
        help="0=baseline, 1=ablation, 2=full contribution flip.",
    )
    p.add_argument("--combined-positive-bundle", default="P_POS7")
    p.add_argument("--combined-negative-bundle", default="P_NEG5")
    p.add_argument("--combined-gamma-pos-grid", default="0,0.5,1,1.5,2,2.5,3")
    p.add_argument("--combined-gamma-neg-grid", default="0,0.5,1,1.5,2,2.5,3")

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
    p.add_argument("--sample-max-samples", type=int, default=0)
    p.add_argument("--exclude-sids-from", default="")
    p.add_argument("--include-sids-file", default="")

    p.add_argument(
        "--query-swap-status",
        choices=STATUSES,
        default="swapped_only",
        help="Default WC: original generation wrong, swapped generation correct.",
    )
    p.add_argument(
        "--query-stage-specs",
        default=(
            "block23_objects,v26_objects,block26_objects,"
            "block26_prompt_last,block27_prompt_last,block28_prompt_last,"
            "block29_prompt_last,block30_prompt_last,block31_prompt_last"
        ),
    )
    p.add_argument(
        "--query-alignments",
        default="identity,role",
        help="Used only for object-position stages.",
    )
    p.add_argument(
        "--query-modes",
        default="reflect",
        help="Comma-separated: reflect,direct.",
    )
    p.add_argument(
        "--query-lambda-grid",
        default="0,0.5,1,1.5,2",
        help="For reflect: h_orig + lambda*(h_orig-h_swap_aligned).",
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
    p.add_argument("--expected-source-baseline-accuracy", type=float, default=0.7114)
    p.add_argument("--baseline-warning-tolerance", type=float, default=0.03)

    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument(
        "--failure-script",
        default="analyze_coco_circuit_failure_repair_v1.py",
    )
    p.add_argument(
        "--generation-helper",
        default="analyze_coco_circuit_generation_repair_grid_v1.py",
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

    # Compatibility with imported helpers.
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
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


def parse_csv_tokens(text: str) -> List[str]:
    result: List[str] = []
    for item in str(text).split(","):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    return result


def parse_float_grid(text: str, label: str) -> List[float]:
    result: List[float] = []
    for item in parse_csv_tokens(text):
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"{label} contains non-finite value {item}")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError(f"{label} is empty")
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def exact_two_sided_binomial_p(successes: int, failures: int) -> float:
    a = int(successes)
    b = int(failures)
    n = a + b
    if n <= 0:
        return 1.0
    k = min(a, b)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * probability))


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {"left_of": "left", "right_of": "right", "on": "above", "under": "below"}
    if text in RELATIONS:
        return text
    return aliases.get(text)


def source_original_correct(row: Mapping[str, Any]) -> Optional[bool]:
    for key in (
        "generation_original_correct",
        "original_generation_correct",
        "original_correct",
    ):
        if key in row and row[key] is not None:
            return bool(row[key])
    return None


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
                payload = json.loads(text)
                if isinstance(payload, Mapping) and "sid" in payload:
                    result.add(int(payload["sid"]))
                else:
                    result.add(int(payload))
            except json.JSONDecodeError:
                result.add(int(text.split(",", 1)[0]))
    return result


def load_bundle_payload(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("bundles", payload)
    if not isinstance(source, Mapping):
        raise ValueError("Bundle JSON must contain an object")
    result: Dict[str, List[str]] = {}
    for name, values in source.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Bundle {name} must be a list")
        names = [str(x) for x in values]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Invalid or duplicate heads in bundle {name}")
        result[str(name)] = names
    return result


# -----------------------------------------------------------------------------
# Stage parsing and generic capture/patch hooks
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str  # "v" or "block"
    layer: int
    scope: str  # "objects" or "prompt_last"


def parse_stage(text: str) -> StageSpec:
    token = str(text).strip().lower()
    match = re.fullmatch(r"v(\d+)_(objects|prompt_last)", token)
    if match:
        return StageSpec(token, "v", int(match.group(1)), match.group(2))
    match = re.fullmatch(r"block(\d+)_(objects|prompt_last)", token)
    if match:
        return StageSpec(token, "block", int(match.group(1)), match.group(2))
    raise ValueError(
        f"Invalid stage {text!r}; use v26_objects, block23_objects, "
        "or block26_prompt_last"
    )


def tensor_from_layer_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported decoder-layer output type: {type(output)}")


def replace_layer_output(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise RuntimeError(f"Unsupported decoder-layer output type: {type(output)}")


class CaptureTensorAtPositions:
    def __init__(self, module: torch.nn.Module, positions: Sequence[int], *, layer_output: bool):
        self.positions = sorted(set(map(int, positions)))
        self.layer_output = bool(layer_output)
        self.states: Dict[int, torch.Tensor] = {}
        self.events = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        tensor = tensor_from_layer_output(output) if self.layer_output else output
        if not torch.is_tensor(tensor) or tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError("Capture expects [1,S,D] tensor")
        seq = int(tensor.shape[1])
        for position in self.positions:
            if not 0 <= position < seq:
                raise RuntimeError(f"Capture position {position} outside sequence length {seq}")
            self.states[position] = tensor[0, position].detach().float().cpu()
        self.events += 1
        return output

    def validate(self) -> None:
        if self.events != 1:
            raise RuntimeError(f"Expected one capture event, got {self.events}")
        if set(self.states) != set(self.positions):
            raise RuntimeError("Capture positions incomplete")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


class PrefillLayerOutputPatch:
    """Patch full hidden vectors at selected positions on generation prefill only."""

    def __init__(self, module: torch.nn.Module, target_to_source: Mapping[int, torch.Tensor]):
        self.target_to_source = {
            int(position): tensor.detach().float().cpu()
            for position, tensor in target_to_source.items()
        }
        if not self.target_to_source:
            raise ValueError("No block-output patch positions")
        self.max_position = max(self.target_to_source)
        self.prefill_events = 0
        self.decode_events = 0
        self.positions_patched = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        hidden = tensor_from_layer_output(output)
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
            raise RuntimeError("Layer patch expects [1,S,H]")
        seq = int(hidden.shape[1])
        if seq <= self.max_position:
            self.decode_events += 1
            return output
        modified = hidden.clone()
        for position, source in self.target_to_source.items():
            if not 0 <= position < seq:
                raise RuntimeError(f"Patch position {position} outside sequence length {seq}")
            if int(source.numel()) != int(hidden.shape[-1]):
                raise RuntimeError(
                    f"Patch width {source.numel()} != hidden width {hidden.shape[-1]}"
                )
            modified[0, position] = source.to(device=hidden.device, dtype=hidden.dtype)
            self.positions_patched += 1
        self.prefill_events += 1
        return replace_layer_output(output, modified)

    def validate(self) -> None:
        if self.prefill_events != 1:
            raise RuntimeError(f"Expected one block prefill patch, got {self.prefill_events}")
        if self.positions_patched != len(self.target_to_source):
            raise RuntimeError(
                f"Patched {self.positions_patched}; expected {len(self.target_to_source)}"
            )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


def run_forward_capture(
    *,
    model: Any,
    batch: Mapping[str, Any],
    captures: Sequence[CaptureTensorAtPositions],
) -> None:
    try:
        with torch.inference_mode():
            model(**dict(batch), use_cache=False, return_dict=True)
        for capture in captures:
            capture.validate()
    finally:
        for capture in reversed(list(captures)):
            capture.close()


def aligned_position_pairs(
    target_positions: Sequence[int],
    source_positions: Sequence[int],
) -> List[Tuple[int, int]]:
    targets = list(map(int, target_positions))
    sources = list(map(int, source_positions))
    if not targets or not sources:
        return []
    if len(targets) == len(sources):
        return list(zip(targets, sources))
    if len(sources) == 1:
        return [(target, sources[0]) for target in targets]
    if len(targets) == 1:
        return [(targets[0], sources[-1])]
    n = min(len(targets), len(sources))
    return list(zip(targets[-n:], sources[-n:]))


def object_alignment_pairs(pair: Any, alignment: str) -> List[Tuple[int, int]]:
    if alignment == "identity":
        pairs = aligned_position_pairs(pair.original_a_positions, pair.swapped_a_positions)
        pairs += aligned_position_pairs(pair.original_b_positions, pair.swapped_b_positions)
    elif alignment == "role":
        pairs = aligned_position_pairs(pair.original_a_positions, pair.swapped_b_positions)
        pairs += aligned_position_pairs(pair.original_b_positions, pair.swapped_a_positions)
    else:
        raise ValueError(f"Unknown alignment {alignment}")
    if not pairs:
        raise RuntimeError(f"No object alignment pairs for {alignment}")
    return pairs


def build_aligned_patch_states(
    *,
    original_states: Mapping[int, torch.Tensor],
    swapped_states: Mapping[int, torch.Tensor],
    position_pairs: Sequence[Tuple[int, int]],
    mode: str,
    lam: float,
) -> Dict[int, torch.Tensor]:
    result: Dict[int, torch.Tensor] = {}
    for target_position, source_position in position_pairs:
        original = original_states[int(target_position)].float()
        swapped = swapped_states[int(source_position)].float()
        if original.shape != swapped.shape:
            raise RuntimeError(
                f"State shape mismatch at target={target_position}, source={source_position}: "
                f"{tuple(original.shape)} vs {tuple(swapped.shape)}"
            )
        if mode == "direct":
            target = swapped
        elif mode == "reflect":
            target = original + float(lam) * (original - swapped)
        else:
            raise ValueError(f"Unknown query mode {mode}")
        result[int(target_position)] = target.detach().float().cpu()
    return result


def combine_signflip_states(
    *,
    baseline: Mapping[int, torch.Tensor],
    without: Mapping[int, torch.Tensor],
    gamma: float,
) -> Dict[int, torch.Tensor]:
    positions = sorted(set(baseline) & set(without))
    if set(positions) != set(baseline) or set(positions) != set(without):
        raise RuntimeError("Sign-flip receiver positions do not match")
    result: Dict[int, torch.Tensor] = {}
    for position in positions:
        base = baseline[position].float()
        delta = base - without[position].float()
        result[position] = (base - float(gamma) * delta).detach().float().cpu()
    return result


def combine_two_signflip_states(
    *,
    baseline: Mapping[int, torch.Tensor],
    without_positive: Mapping[int, torch.Tensor],
    without_negative: Mapping[int, torch.Tensor],
    gamma_positive: float,
    gamma_negative: float,
) -> Dict[int, torch.Tensor]:
    positions = sorted(set(baseline) & set(without_positive) & set(without_negative))
    if set(positions) != set(baseline):
        raise RuntimeError("Combined sign-flip receiver positions do not match")
    result: Dict[int, torch.Tensor] = {}
    for position in positions:
        base = baseline[position].float()
        delta_positive = base - without_positive[position].float()
        delta_negative = base - without_negative[position].float()
        result[position] = (
            base
            - float(gamma_positive) * delta_positive
            - float(gamma_negative) * delta_negative
        ).detach().float().cpu()
    return result


@torch.inference_mode()
def generate_with_block_patch(
    *,
    generation: Any,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
    module: torch.nn.Module,
    patch_states: Mapping[int, torch.Tensor],
) -> Dict[str, Any]:
    patch = PrefillLayerOutputPatch(module, patch_states)
    input_ids = batch.get("input_ids")
    if input_ids is None or not torch.is_tensor(input_ids):
        patch.close()
        raise RuntimeError("Generation batch must contain input_ids")
    prompt_length = int(input_ids.shape[1])
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.generation_do_sample),
        "num_beams": int(args.num_beams),
        "use_cache": True,
        "return_dict_in_generate": False,
    }
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.generation_do_sample:
        kwargs["temperature"] = float(args.temperature)
        kwargs["top_p"] = float(args.top_p)
        if int(args.top_k) > 0:
            kwargs["top_k"] = int(args.top_k)
    started = time.perf_counter()
    try:
        generated_ids = model.generate(**dict(batch), **kwargs)
        patch.validate()
    finally:
        patch.close()
    elapsed = time.perf_counter() - started
    new_ids = generated_ids[0, prompt_length:].detach().cpu().tolist()
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    prediction = generation.parse_generated_relation(text)
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


def summarize_interventions(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    summary: List[Dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        n = len(values)
        baseline_correct = sum(int(bool(row["baseline_correct"])) for row in values)
        correct = sum(int(bool(row["correct"])) for row in values)
        parsed = sum(int(bool(row["parsed"])) for row in values)
        fixes = sum(int(bool(row["fixed"])) for row in values)
        breaks = sum(int(bool(row["broken"])) for row in values)
        output = {field: value for field, value in zip(group_fields, key)}
        output.update(
            {
                "N": n,
                "baseline_accuracy": baseline_correct / n if n else float("nan"),
                "accuracy": correct / n if n else float("nan"),
                "accuracy_delta": (correct - baseline_correct) / n if n else float("nan"),
                "parse_rate": parsed / n if n else float("nan"),
                "fixes": fixes,
                "breaks": breaks,
                "net_fixes": fixes - breaks,
                "fix_rate_among_baseline_wrong": (
                    fixes / (n - baseline_correct) if n > baseline_correct else float("nan")
                ),
                "preserve_rate_among_baseline_correct": (
                    (baseline_correct - breaks) / baseline_correct
                    if baseline_correct else float("nan")
                ),
                "mcnemar_exact_p": exact_two_sided_binomial_p(fixes, breaks),
                "mean_generation_seconds": safe_mean(
                    float(row.get("generation_seconds", float("nan"))) for row in values
                ),
            }
        )
        summary.append(output)
    return summary


def best_rows_by_stage(summary: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary:
        groups[str(row["stage"])].append(row)
    result: List[Dict[str, Any]] = []
    for stage, values in sorted(groups.items()):
        best = sorted(
            values,
            key=lambda row: (
                -float(row["accuracy"]),
                -int(row["net_fixes"]),
                int(row["breaks"]),
                str(row.get("alignment")),
                str(row.get("mode")),
                float(row.get("lambda", 0.0)),
            ),
        )[0]
        result.append(dict(best))
    return result


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

    experiments = set(parse_csv_tokens(args.experiments))
    unknown = experiments - {"bundle_flip", "combined_flip", "query_swap_restore"}
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    if not experiments:
        raise ValueError("No experiment selected")

    gamma_grid = parse_float_grid(args.gamma_grid, "gamma-grid")
    combined_pos_grid = parse_float_grid(
        args.combined_gamma_pos_grid, "combined-gamma-pos-grid"
    )
    combined_neg_grid = parse_float_grid(
        args.combined_gamma_neg_grid, "combined-gamma-neg-grid"
    )
    query_lambdas = parse_float_grid(args.query_lambda_grid, "query-lambda-grid")
    query_modes = parse_csv_tokens(args.query_modes)
    if set(query_modes) - {"reflect", "direct"}:
        raise ValueError("query-modes must contain only reflect,direct")
    query_alignments = parse_csv_tokens(args.query_alignments)
    if set(query_alignments) - {"identity", "role"}:
        raise ValueError("query-alignments must contain only identity,role")
    stages = [parse_stage(text) for text in parse_csv_tokens(args.query_stage_specs)]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failure = import_file(Path(args.failure_script), "signflip_failure")
    generation = import_file(Path(args.generation_helper), "signflip_generation")
    ioi = import_file(Path(args.ioi_script), "signflip_ioi")
    producer = import_file(Path(args.producer_script), "signflip_producer")
    receiver = import_file(Path(args.receiver_script), "signflip_receiver")
    v3 = import_file(Path(args.v3_script), "signflip_v3")
    base = import_file(Path(args.base_script), "signflip_base")
    attention_helper = import_file(Path(args.attention_helper), "signflip_attention")

    source_config, source_rows = ioi.load_source_rows(args)

    excluded: set[int] = set()
    for item in parse_csv_tokens(args.exclude_sids_from):
        excluded.update(extract_sids(Path(item)))
    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(args.include_sids_file.strip()))

    all_eligible = [
        dict(row)
        for row in source_rows
        if int(row["sid"]) not in excluded
        and (included is None or int(row["sid"]) in included)
    ]
    selected_rows = [
        row
        for row in all_eligible
        if args.sample_status == "all"
        or str(row.get("generation_pair_status")) == args.sample_status
    ]
    selected_rows = failure.deterministic_stratified_limit(
        selected_rows, int(args.sample_max_samples), int(args.seed)
    )
    selected_rows = sorted(selected_rows, key=lambda row: int(row["sid"]))
    if not selected_rows:
        raise RuntimeError("No eligible samples for bundle experiments")

    query_rows = [
        row
        for row in all_eligible
        if args.query_swap_status == "all"
        or str(row.get("generation_pair_status")) == args.query_swap_status
    ]
    query_rows = failure.deterministic_stratified_limit(
        query_rows, int(args.sample_max_samples), int(args.seed)
    )
    query_rows = sorted(query_rows, key=lambda row: int(row["sid"]))

    payload = load_bundle_payload(Path(args.bundle_json))
    flip_bundle_names = parse_csv_tokens(args.flip_bundles)
    for name in flip_bundle_names + [
        args.combined_positive_bundle,
        args.combined_negative_bundle,
    ]:
        if name not in payload:
            raise KeyError(f"Bundle {name} missing from {args.bundle_json}")

    required_bundle_names = list(dict.fromkeys(
        flip_bundle_names
        + [args.combined_positive_bundle, args.combined_negative_bundle]
    ))
    named_bundles = failure.load_named_bundles(
        Path(args.bundle_json), required_bundle_names, ioi
    )

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
        receiver_attention = attention_helper.resolve_self_attention(
            decoder_layers[int(unit.layer)]
        )
        receiver_shape = receiver.resolve_attention_shape(receiver_attention)
        receiver_projection = receiver.projection_module(receiver_attention, unit.channel)
        receiver_head_dim = int(receiver_shape.kv_head_dim)

        # Validate all requested stages.
        for stage in stages:
            if not 0 <= stage.layer < len(decoder_layers):
                raise ValueError(f"Stage {stage.name} layer outside model")
            if stage.kind == "v" and stage.layer != int(unit.layer):
                raise ValueError(
                    f"This v1 script supports V-stage patching only at validated "
                    f"receiver layer {unit.layer}; got {stage.name}"
                )

        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        source_known = [source_original_correct(row) for row in selected_rows]
        source_known = [x for x in source_known if x is not None]
        source_baseline_accuracy = safe_mean(int(x) for x in source_known)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "decoder_path": decoder_path,
            "experiments": sorted(experiments),
            "bundle_flip": {
                "bundles": flip_bundle_names,
                "gamma_grid": gamma_grid,
                "formula": "V_base - gamma*(V_base-V_without_bundle)",
            },
            "combined_flip": {
                "positive_bundle": args.combined_positive_bundle,
                "negative_bundle": args.combined_negative_bundle,
                "gamma_positive_grid": combined_pos_grid,
                "gamma_negative_grid": combined_neg_grid,
                "formula": (
                    "V_base - gamma_pos*(V_base-V_without_POS7) "
                    "- gamma_neg*(V_base-V_without_NEG5)"
                ),
            },
            "query_swap_restore": {
                "source_status": args.query_swap_status,
                "stages": [stage.name for stage in stages],
                "alignments": query_alignments,
                "modes": query_modes,
                "lambda_grid": query_lambdas,
                "reflect_formula": "h_original + lambda*(h_original-h_swapped_aligned)",
            },
            "receiver": {
                "unit": unit.unit,
                "layer": int(unit.layer),
                "kv_head": int(unit.kv_head),
                "unit_head": int(unit.unit_head),
                "shared_query_heads": list(unit.shared_query_heads),
                "channel": unit.channel,
            },
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.generation_do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
            },
            "selected_bundle_samples": len(selected_rows),
            "selected_query_swap_samples": len(query_rows),
            "source_cached_original_generation_accuracy": source_baseline_accuracy,
            "selection_note": "best setting is selected on the same data; exploratory/oracle",
            "seed": args.seed,
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        errors_path = output_dir / "errors.jsonl"
        capture_layers = list(range(receiver_layer + 1))

        # ------------------------------------------------------------------
        # Experiment A1: one-bundle sign flips
        # ------------------------------------------------------------------
        bundle_path = output_dir / "bundle_signflip_generation.jsonl"
        bundle_rows = read_jsonl(bundle_path) if args.resume else []
        bundle_done = {
            (int(row["sid"]), str(row["bundle"]), float(row["gamma"]))
            for row in bundle_rows
        }
        if "bundle_flip" in experiments:
            pending = [
                row
                for row in selected_rows
                if any(
                    (int(row["sid"]), name, float(gamma)) not in bundle_done
                    for name in flip_bundle_names
                    for gamma in gamma_grid
                )
            ]
            print(
                "Bundle sign-flip generation: "
                f"N={len(selected_rows)}, pending={len(pending)}, "
                f"bundles={len(flip_bundle_names)}, gammas={len(gamma_grid)}, "
                f"expected_rows={len(selected_rows)*len(flip_bundle_names)*len(gamma_grid)}",
                flush=True,
            )
            for sample_index, source_row in enumerate(
                tqdm(pending, desc=f"bundle-signflip:{args.model}"), start=1
            ):
                pair = None
                try:
                    sid = int(source_row["sid"])
                    missing = [
                        (name, gamma)
                        for name in flip_bundle_names
                        for gamma in gamma_grid
                        if (sid, name, float(gamma)) not in bundle_done
                    ]
                    if not missing:
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
                        sender_positions = sorted({
                            int(pair.original_a_positions[-1]),
                            int(pair.original_b_positions[-1]),
                        })
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
                    baseline_v = baseline_states[int(unit.layer)][unit.channel]
                    needed_names = sorted(set(name for name, _ in missing))
                    without_by_name: Dict[str, Mapping[int, torch.Tensor]] = {}
                    for name in needed_names:
                        removed_states = failure.run_bundle_removal_c_pass(
                            bundle=named_bundles[name],
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
                        without_by_name[name] = removed_states[int(unit.layer)][unit.channel]

                    baseline_generation = generation.generate_answer(
                        model=model,
                        processor=processor,
                        batch=pair.original_batch,
                        args=args,
                    )
                    baseline_prediction = normalize_relation(baseline_generation["prediction"])
                    gt = normalize_relation(pair.gt)
                    if gt is None:
                        raise RuntimeError(f"Invalid GT for SID {sid}")
                    baseline_correct = bool(baseline_prediction == gt)

                    for name, gamma in missing:
                        if float(gamma) == 0.0:
                            generated = baseline_generation
                        else:
                            patch_states = combine_signflip_states(
                                baseline=baseline_v,
                                without=without_by_name[name],
                                gamma=float(gamma),
                            )
                            generated = generation.generate_answer(
                                model=model,
                                processor=processor,
                                batch=pair.original_batch,
                                args=args,
                                patch_module=receiver_projection,
                                patch_head=int(unit.unit_head),
                                patch_head_dim=receiver_head_dim,
                                patch_states=patch_states,
                            )
                        prediction = normalize_relation(generated["prediction"])
                        correct = bool(prediction == gt)
                        row = {
                            "script_version": SCRIPT_VERSION,
                            "model": args.model,
                            "sid": sid,
                            "gt": gt,
                            "bundle": name,
                            "gamma": float(gamma),
                            "operation": (
                                "baseline" if float(gamma) == 0.0
                                else "ablation" if float(gamma) == 1.0
                                else "full_flip" if float(gamma) == 2.0
                                else "scaled_reverse"
                            ),
                            "baseline_generation": baseline_generation["text"],
                            "baseline_prediction": baseline_prediction,
                            "baseline_correct": baseline_correct,
                            "generation": generated["text"],
                            "prediction": prediction,
                            "parsed": prediction is not None,
                            "correct": correct,
                            "fixed": bool((not baseline_correct) and correct),
                            "broken": bool(baseline_correct and (not correct)),
                            "new_token_count": int(generated["new_token_count"]),
                            "generation_seconds": float(generated["generation_seconds"]),
                            "source_generation_pair_status": source_row.get(
                                "generation_pair_status", "unknown"
                            ),
                            "sender_positions": sender_positions,
                            "receiver_positions": receiver_positions,
                            "receiver_unit": unit.unit,
                        }
                        append_jsonl(bundle_path, row)
                        bundle_rows.append(row)
                        bundle_done.add((sid, name, float(gamma)))

                    if args.print_every > 0 and sample_index % args.print_every == 0:
                        print(
                            f"[bundle sample {sample_index}/{len(pending)} sid={sid}] "
                            f"saved={len(bundle_rows)}",
                            flush=True,
                        )
                except Exception as exc:
                    error = {
                        "script_version": SCRIPT_VERSION,
                        "experiment": "bundle_flip",
                        "source_sid": int(source_row.get("sid", -1)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    append_jsonl(errors_path, error)
                    if args.fail_fast:
                        raise
                finally:
                    receiver.release_pair(pair)
                    gc.collect()
                    if torch.cuda.is_available() and (
                        sample_index % max(1, int(args.empty_cache_every)) == 0
                    ):
                        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Experiment A2: joint POS7 / NEG5 sign grid
        # ------------------------------------------------------------------
        combined_path = output_dir / "combined_signflip_generation.jsonl"
        combined_rows = read_jsonl(combined_path) if args.resume else []
        combined_done = {
            (int(row["sid"]), float(row["gamma_positive"]), float(row["gamma_negative"]))
            for row in combined_rows
        }
        if "combined_flip" in experiments:
            combined_grid = [
                (gp, gn) for gp in combined_pos_grid for gn in combined_neg_grid
            ]
            pending = [
                row
                for row in selected_rows
                if any(
                    (int(row["sid"]), float(gp), float(gn)) not in combined_done
                    for gp, gn in combined_grid
                )
            ]
            print(
                "Combined sign-flip generation: "
                f"N={len(selected_rows)}, pending={len(pending)}, grid={len(combined_grid)}, "
                f"expected_rows={len(selected_rows)*len(combined_grid)}",
                flush=True,
            )
            pos_bundle = named_bundles[args.combined_positive_bundle]
            neg_bundle = named_bundles[args.combined_negative_bundle]
            for sample_index, source_row in enumerate(
                tqdm(pending, desc=f"combined-signflip:{args.model}"), start=1
            ):
                pair = None
                try:
                    sid = int(source_row["sid"])
                    missing = [
                        (gp, gn)
                        for gp, gn in combined_grid
                        if (sid, float(gp), float(gn)) not in combined_done
                    ]
                    if not missing:
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
                        sender_positions = sorted({
                            int(pair.original_a_positions[-1]),
                            int(pair.original_b_positions[-1]),
                        })
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
                    pos_removed = failure.run_bundle_removal_c_pass(
                        bundle=pos_bundle,
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
                    )[int(unit.layer)][unit.channel]
                    neg_removed = failure.run_bundle_removal_c_pass(
                        bundle=neg_bundle,
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
                    )[int(unit.layer)][unit.channel]
                    baseline_v = baseline_states[int(unit.layer)][unit.channel]
                    baseline_generation = generation.generate_answer(
                        model=model,
                        processor=processor,
                        batch=pair.original_batch,
                        args=args,
                    )
                    baseline_prediction = normalize_relation(baseline_generation["prediction"])
                    gt = normalize_relation(pair.gt)
                    if gt is None:
                        raise RuntimeError(f"Invalid GT for SID {sid}")
                    baseline_correct = bool(baseline_prediction == gt)
                    for gp, gn in missing:
                        if float(gp) == 0.0 and float(gn) == 0.0:
                            generated = baseline_generation
                        else:
                            patch_states = combine_two_signflip_states(
                                baseline=baseline_v,
                                without_positive=pos_removed,
                                without_negative=neg_removed,
                                gamma_positive=float(gp),
                                gamma_negative=float(gn),
                            )
                            generated = generation.generate_answer(
                                model=model,
                                processor=processor,
                                batch=pair.original_batch,
                                args=args,
                                patch_module=receiver_projection,
                                patch_head=int(unit.unit_head),
                                patch_head_dim=receiver_head_dim,
                                patch_states=patch_states,
                            )
                        prediction = normalize_relation(generated["prediction"])
                        correct = bool(prediction == gt)
                        row = {
                            "script_version": SCRIPT_VERSION,
                            "model": args.model,
                            "sid": sid,
                            "gt": gt,
                            "gamma_positive": float(gp),
                            "gamma_negative": float(gn),
                            "baseline_generation": baseline_generation["text"],
                            "baseline_prediction": baseline_prediction,
                            "baseline_correct": baseline_correct,
                            "generation": generated["text"],
                            "prediction": prediction,
                            "parsed": prediction is not None,
                            "correct": correct,
                            "fixed": bool((not baseline_correct) and correct),
                            "broken": bool(baseline_correct and (not correct)),
                            "new_token_count": int(generated["new_token_count"]),
                            "generation_seconds": float(generated["generation_seconds"]),
                            "source_generation_pair_status": source_row.get(
                                "generation_pair_status", "unknown"
                            ),
                        }
                        append_jsonl(combined_path, row)
                        combined_rows.append(row)
                        combined_done.add((sid, float(gp), float(gn)))
                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "script_version": SCRIPT_VERSION,
                            "experiment": "combined_flip",
                            "source_sid": int(source_row.get("sid", -1)),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if args.fail_fast:
                        raise
                finally:
                    receiver.release_pair(pair)
                    gc.collect()
                    if torch.cuda.is_available() and (
                        sample_index % max(1, int(args.empty_cache_every)) == 0
                    ):
                        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Experiment B: query-swap reflected-state restoration
        # ------------------------------------------------------------------
        query_path = output_dir / "query_swap_restore_generation.jsonl"
        query_result_rows = read_jsonl(query_path) if args.resume else []
        query_done = {
            (
                int(row["sid"]),
                str(row["stage"]),
                str(row["alignment"]),
                str(row["mode"]),
                float(row["lambda"]),
            )
            for row in query_result_rows
        }
        if "query_swap_restore" in experiments:
            query_settings: List[Tuple[StageSpec, str, str, float]] = []
            for stage in stages:
                alignments = query_alignments if stage.scope == "objects" else ["global"]
                for alignment in alignments:
                    for mode in query_modes:
                        lambdas = query_lambdas if mode == "reflect" else [1.0]
                        for lam in lambdas:
                            query_settings.append((stage, alignment, mode, float(lam)))
            pending = [
                row
                for row in query_rows
                if any(
                    (
                        int(row["sid"]), stage.name, alignment, mode, float(lam)
                    ) not in query_done
                    for stage, alignment, mode, lam in query_settings
                )
            ]
            print(
                "Query-swap stage restoration: "
                f"status={args.query_swap_status}, N={len(query_rows)}, "
                f"pending={len(pending)}, settings={len(query_settings)}, "
                f"expected_rows={len(query_rows)*len(query_settings)}",
                flush=True,
            )

            for sample_index, source_row in enumerate(
                tqdm(pending, desc=f"query-swap-restore:{args.model}"), start=1
            ):
                pair = None
                captures_original: List[CaptureTensorAtPositions] = []
                captures_swapped: List[CaptureTensorAtPositions] = []
                try:
                    sid = int(source_row["sid"])
                    missing = [
                        setting
                        for setting in query_settings
                        if (
                            sid,
                            setting[0].name,
                            setting[1],
                            setting[2],
                            float(setting[3]),
                        ) not in query_done
                    ]
                    if not missing:
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
                    gt = normalize_relation(pair.gt)
                    if gt is None:
                        raise RuntimeError(f"Invalid GT for SID {sid}")
                    swapped_gt = OPPOSITE[gt]

                    baseline_original = generation.generate_answer(
                        model=model,
                        processor=processor,
                        batch=pair.original_batch,
                        args=args,
                    )
                    baseline_swapped = generation.generate_answer(
                        model=model,
                        processor=processor,
                        batch=pair.swapped_batch,
                        args=args,
                    )
                    original_prediction = normalize_relation(baseline_original["prediction"])
                    swapped_prediction = normalize_relation(baseline_swapped["prediction"])
                    baseline_correct = bool(original_prediction == gt)
                    swapped_correct = bool(swapped_prediction == swapped_gt)

                    # Build one capture per unique stage/module.  Every capture stores
                    # all semantic positions needed by either alignment.
                    original_stage_states: Dict[str, Dict[int, torch.Tensor]] = {}
                    swapped_stage_states: Dict[str, Dict[int, torch.Tensor]] = {}
                    original_capture_by_stage: Dict[str, CaptureTensorAtPositions] = {}
                    swapped_capture_by_stage: Dict[str, CaptureTensorAtPositions] = {}
                    # Original and swapped hooks must be active in separate forwards.
                    # Original forward.
                    captures_original = []
                    for stage in stages:
                        positions = (
                            pair.original_object_positions
                            if stage.scope == "objects"
                            else [pair.original_prompt_last]
                        )
                        module = receiver_projection if stage.kind == "v" else decoder_layers[stage.layer]
                        cap = CaptureTensorAtPositions(
                            module, positions, layer_output=(stage.kind == "block")
                        )
                        original_capture_by_stage[stage.name] = cap
                        captures_original.append(cap)
                    run_forward_capture(
                        model=model,
                        batch=pair.original_batch,
                        captures=captures_original,
                    )
                    captures_original = []
                    for stage in stages:
                        original_stage_states[stage.name] = dict(
                            original_capture_by_stage[stage.name].states
                        )

                    # Swapped forward.
                    captures_swapped = []
                    for stage in stages:
                        positions = (
                            pair.swapped_object_positions
                            if stage.scope == "objects"
                            else [pair.swapped_prompt_last]
                        )
                        module = receiver_projection if stage.kind == "v" else decoder_layers[stage.layer]
                        cap = CaptureTensorAtPositions(
                            module, positions, layer_output=(stage.kind == "block")
                        )
                        swapped_capture_by_stage[stage.name] = cap
                        captures_swapped.append(cap)
                    run_forward_capture(
                        model=model,
                        batch=pair.swapped_batch,
                        captures=captures_swapped,
                    )
                    captures_swapped = []
                    for stage in stages:
                        swapped_stage_states[stage.name] = dict(
                            swapped_capture_by_stage[stage.name].states
                        )

                    for stage, alignment, mode, lam in missing:
                        if stage.scope == "objects":
                            position_pairs = object_alignment_pairs(pair, alignment)
                        else:
                            position_pairs = [
                                (int(pair.original_prompt_last), int(pair.swapped_prompt_last))
                            ]
                        patch_states = build_aligned_patch_states(
                            original_states=original_stage_states[stage.name],
                            swapped_states=swapped_stage_states[stage.name],
                            position_pairs=position_pairs,
                            mode=mode,
                            lam=float(lam),
                        )
                        # lambda=0 in reflect exactly reproduces the original state.
                        if mode == "reflect" and float(lam) == 0.0:
                            generated = baseline_original
                        elif stage.kind == "v":
                            generated = generation.generate_answer(
                                model=model,
                                processor=processor,
                                batch=pair.original_batch,
                                args=args,
                                patch_module=receiver_projection,
                                patch_head=int(unit.unit_head),
                                patch_head_dim=receiver_head_dim,
                                patch_states=patch_states,
                            )
                        else:
                            generated = generate_with_block_patch(
                                generation=generation,
                                model=model,
                                processor=processor,
                                batch=pair.original_batch,
                                args=args,
                                module=decoder_layers[stage.layer],
                                patch_states=patch_states,
                            )
                        prediction = normalize_relation(generated["prediction"])
                        correct = bool(prediction == gt)
                        row = {
                            "script_version": SCRIPT_VERSION,
                            "model": args.model,
                            "sid": sid,
                            "gt": gt,
                            "swapped_gt": swapped_gt,
                            "source_generation_pair_status": source_row.get(
                                "generation_pair_status", "unknown"
                            ),
                            "stage": stage.name,
                            "stage_kind": stage.kind,
                            "stage_layer": stage.layer,
                            "stage_scope": stage.scope,
                            "alignment": alignment,
                            "mode": mode,
                            "lambda": float(lam),
                            "baseline_original_generation": baseline_original["text"],
                            "baseline_original_prediction": original_prediction,
                            "baseline_correct": baseline_correct,
                            "baseline_swapped_generation": baseline_swapped["text"],
                            "baseline_swapped_prediction": swapped_prediction,
                            "swapped_correct": swapped_correct,
                            "generation": generated["text"],
                            "prediction": prediction,
                            "parsed": prediction is not None,
                            "correct": correct,
                            "fixed": bool((not baseline_correct) and correct),
                            "broken": bool(baseline_correct and (not correct)),
                            "prediction_became_swapped_gt": bool(prediction == swapped_gt),
                            "new_token_count": int(generated["new_token_count"]),
                            "generation_seconds": float(generated["generation_seconds"]),
                            "position_pairs": [list(map(int, pair_)) for pair_ in position_pairs],
                        }
                        append_jsonl(query_path, row)
                        query_result_rows.append(row)
                        query_done.add((sid, stage.name, alignment, mode, float(lam)))

                    if args.print_every > 0 and sample_index % args.print_every == 0:
                        print(
                            f"[query sample {sample_index}/{len(pending)} sid={sid}] "
                            f"saved={len(query_result_rows)}",
                            flush=True,
                        )
                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "script_version": SCRIPT_VERSION,
                            "experiment": "query_swap_restore",
                            "source_sid": int(source_row.get("sid", -1)),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if args.fail_fast:
                        raise
                finally:
                    for capture in captures_original + captures_swapped:
                        with contextlib.suppress(Exception):
                            capture.close()
                    receiver.release_pair(pair)
                    gc.collect()
                    if torch.cuda.is_available() and (
                        sample_index % max(1, int(args.empty_cache_every)) == 0
                    ):
                        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Write summaries
        # ------------------------------------------------------------------
        summary_payload: Dict[str, Any] = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "source_cached_baseline_accuracy": source_baseline_accuracy,
        }

        if bundle_rows:
            bundle_summary = summarize_interventions(bundle_rows, ["bundle", "gamma"])
            write_csv(output_dir / "bundle_signflip_summary.csv", bundle_summary)
            best_bundle = sorted(
                bundle_summary,
                key=lambda row: (
                    -float(row["accuracy"]),
                    -int(row["net_fixes"]),
                    int(row["breaks"]),
                ),
            )[0]
            summary_payload["bundle_flip"] = {
                "rows": len(bundle_rows),
                "best_same_data": best_bundle,
            }

        if combined_rows:
            combined_summary = summarize_interventions(
                combined_rows, ["gamma_positive", "gamma_negative"]
            )
            write_csv(output_dir / "combined_signflip_summary.csv", combined_summary)
            best_combined = sorted(
                combined_summary,
                key=lambda row: (
                    -float(row["accuracy"]),
                    -int(row["net_fixes"]),
                    int(row["breaks"]),
                ),
            )[0]
            summary_payload["combined_flip"] = {
                "rows": len(combined_rows),
                "best_same_data": best_combined,
            }

        if query_result_rows:
            query_summary = summarize_interventions(
                query_result_rows, ["stage", "alignment", "mode", "lambda"]
            )
            write_csv(output_dir / "query_swap_restore_summary.csv", query_summary)
            stage_best = best_rows_by_stage(query_summary)
            write_csv(output_dir / "query_swap_stage_best.csv", stage_best)
            best_query = sorted(
                query_summary,
                key=lambda row: (
                    -float(row["accuracy"]),
                    -int(row["net_fixes"]),
                    int(row["breaks"]),
                ),
            )[0]
            summary_payload["query_swap_restore"] = {
                "rows": len(query_result_rows),
                "best_same_data": best_query,
                "stage_best": stage_best,
            }

        write_json(output_dir / "summary.json", summary_payload)

        print("\n" + "=" * 148)
        print("BUNDLE SIGN-FLIP / QUERY-SWAP RESTORATION RESULT")
        print("=" * 148)
        print(
            f"Bundle samples={len(selected_rows)} | query-swap samples={len(query_rows)} | "
            f"source cached baseline={source_baseline_accuracy:.4f}"
        )
        if "bundle_flip" in summary_payload:
            best = summary_payload["bundle_flip"]["best_same_data"]
            print(
                "BEST BUNDLE FLIP: "
                f"bundle={best['bundle']} gamma={float(best['gamma']):.3f} "
                f"acc={float(best['accuracy']):.4f} "
                f"delta={float(best['accuracy_delta']):+.4f} "
                f"fixes={best['fixes']} breaks={best['breaks']} net={best['net_fixes']}"
            )
        if "combined_flip" in summary_payload:
            best = summary_payload["combined_flip"]["best_same_data"]
            print(
                "BEST COMBINED FLIP: "
                f"gamma_pos={float(best['gamma_positive']):.3f} "
                f"gamma_neg={float(best['gamma_negative']):.3f} "
                f"acc={float(best['accuracy']):.4f} "
                f"delta={float(best['accuracy_delta']):+.4f} "
                f"fixes={best['fixes']} breaks={best['breaks']} net={best['net_fixes']}"
            )
        if "query_swap_restore" in summary_payload:
            best = summary_payload["query_swap_restore"]["best_same_data"]
            print(
                "BEST QUERY-SWAP RESTORE: "
                f"stage={best['stage']} alignment={best['alignment']} "
                f"mode={best['mode']} lambda={float(best['lambda']):.3f} "
                f"acc={float(best['accuracy']):.4f} "
                f"delta={float(best['accuracy_delta']):+.4f} "
                f"fixes={best['fixes']} breaks={best['breaks']} net={best['net_fixes']}"
            )
        print(f"Saved outputs to {output_dir}")

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
