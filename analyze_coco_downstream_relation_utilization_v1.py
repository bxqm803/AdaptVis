#!/usr/bin/env python3
"""
Downstream spatial-relation trajectory and utilization analysis for COCO/Qwen2.5-VL.

This script is deliberately narrower than the earlier sender/receiver searches.
It starts from the already validated producer -> L26VH0 path and asks what
happens after relation information reaches the late decoder.

It explicitly avoids two invalid assumptions:

1. "L26VH0 centroid predicts GT" does NOT automatically mean the model has a
   correct, causally used spatial representation.  A supervised probe may read
   out decodable but unused or spurious information.
2. "Free generation is correct" does NOT automatically mean the internal
   spatial computation is correct.  Some answers may be late recoveries,
   language-prior outputs, or otherwise unsupported by the measured circuit.

The script therefore separates three notions:

Availability
    Is the GT relation cross-fitted-decodable from a state?

Model-aligned readout
    What does the shared final norm + LM head (logit lens) predict from that
    state, and how does the GT-vs-comparison margin change after attention/MLP?

Causal utilization
    Does erasing or replacing the relation subspace at L26VH0, attention/MLP
    updates, block outputs, or final norm change the final relation scores (and,
    optionally, free generation)?

Clean trajectory captured at prompt-last for L26-L31:

    block input
    attention update
    post-attention residual
    MLP update
    block output
    final norm

L26VH0 role states are reused from analyze_coco_signed_routing_shift_v1.py.
Relation probes/planes are cross-fitted using GT labels from ALL training-fold
samples, not only generation-correct samples.  This avoids defining "correct
spatial information" by the model's own output correctness.

Main outputs
------------
clean_stage_states.jsonl
states/sid_XXXXXX.npz
stage_trajectory.csv
stage_accuracy_summary.csv
component_transition_summary.csv
sample_trajectory_summary.csv
causal_relation_interventions.jsonl
causal_intervention_summary.csv
sample_mechanism_summary.csv
first_flip_summary.csv
summary.json
config.json
errors.jsonl

Operational labels end in "_candidate".  They are analysis categories, not
proof that a sample was guessed or that a specific component caused the error.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-downstream-relation-utilization-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {relation: index for index, relation in enumerate(RELATIONS)}
ID_TO_REL = {index: relation for relation, index in REL_TO_ID.items()}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--phase", choices=("extract", "analyze", "causal", "all"), default="all")
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
        "--baseline-generation-jsonl",
        default="",
        help="Baseline free-generation rows. Empty uses <head-output-dir>/baseline_generation.jsonl.",
    )
    p.add_argument(
        "--head-output-dir",
        default="output/coco_ioi_backward/qwen-3b_head_misrouting_pos7_neg5",
    )
    p.add_argument(
        "--signed-routing-dir",
        default="output/coco_ioi_backward/qwen-3b_signed_routing_shift_full440",
        help="Directory containing sample_routing_summary.csv, clean_sample_states.jsonl, and vectors/.",
    )

    p.add_argument("--late-layers", default="26-31")
    p.add_argument("--probe-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--sample-max-samples", type=int, default=0)
    p.add_argument("--include-sids-file", default="")
    p.add_argument("--exclude-sids-from", default="")

    p.add_argument(
        "--causal-components",
        default="l26v,L26_block,L27_block,L28_block,L29_block,L30_block,L31_block,final_norm",
        help=(
            "Comma-separated components. Supported: l26v, final_norm, and "
            "L<layer>_attn_update/L<layer>_mlp_update/L<layer>_block."
        ),
    )
    p.add_argument(
        "--causal-interventions",
        default="erase,gt_replace,counterfactual_replace",
        help="Comma-separated: erase,gt_replace,counterfactual_replace.",
    )
    p.add_argument(
        "--causal-groups",
        default="wrong_prior_l26_gt,wrong_all_l26_gt,correct_all_l26_wrong,correct_all_l26_gt",
        help=(
            "Comma-separated: all,all_wrong,all_correct,wrong_prior_l26_gt," 
            "wrong_all_l26_gt,wrong_all_l26_wrong,correct_all_l26_gt," 
            "correct_all_l26_wrong."
        ),
    )
    p.add_argument(
        "--causal-max-per-group",
        type=int,
        default=0,
        help="0 means all; otherwise stratified limit per requested group.",
    )
    p.add_argument("--patch-strength", type=float, default=1.0)
    p.add_argument(
        "--causal-generate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also rerun natural-stop greedy generation for every causal intervention.",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--effect-threshold",
        type=float,
        default=0.10,
        help="Normalized causal-effect threshold used only for candidate taxonomy.",
    )

    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument("--ioi-script", default="analyze_coco_ioi_backward_circuit_v1.py")
    p.add_argument("--producer-script", default="analyze_coco_producer_qk_ov_v1.py")
    p.add_argument("--receiver-script", default="analyze_coco_receiver_qkv_v1.py")
    p.add_argument("--v3-script", default="analyze_spatial_storage_transport_utilization_v3.py")
    p.add_argument("--base-script", default="analyze_coco_centroid_generation_step1_v4.py")
    p.add_argument("--attention-helper", default="analyze_coco_flip_attention_spatial_vectors_v1.py")
    p.add_argument("--generation-helper", default="analyze_coco_circuit_generation_repair_grid_v1.py")
    p.add_argument("--routing-script", default="analyze_coco_signed_routing_shift_v1.py")

    # Compatibility with imported repository helpers.
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


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
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def deduplicate_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    table: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        table[tuple(row.get(key) for key in keys)] = dict(row)
    return list(table.values())


def safe_mean(values: Iterable[Any]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def safe_std(values: Iterable[Any]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if x.size >= 2 else float("nan")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else float(default)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {"left_of": "left", "right_of": "right", "on": "above", "under": "below"}
    if text in RELATIONS:
        return text
    return aliases.get(text)


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    result: List[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(item)
            result.extend(range(start, stop + 1))
        else:
            result.append(int(item))
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError("No layers selected")
    for layer in result:
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} outside 0..{n_layers - 1}")
    return result


def parse_csv_tokens(text: str) -> List[str]:
    return list(dict.fromkeys(item.strip() for item in str(text).split(",") if item.strip()))


def stable_fold(key: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(folds)


def unit_vector(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    if not math.isfinite(norm) or norm <= eps:
        return np.zeros_like(x, dtype=np.float64)
    return x / norm


def orthonormal_basis(vectors: Sequence[np.ndarray], eps: float = 1e-10) -> np.ndarray:
    basis: List[np.ndarray] = []
    for vector in vectors:
        value = np.asarray(vector, dtype=np.float64).copy()
        for existing in basis:
            value = value - float(np.dot(value, existing)) * existing
        norm = float(np.linalg.norm(value))
        if math.isfinite(norm) and norm > eps:
            basis.append(value / norm)
    if not basis:
        return np.zeros((len(np.asarray(vectors[0])), 0), dtype=np.float64)
    return np.stack(basis, axis=1)


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
        for row in read_csv(path):
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


def relation_score_map(value: Any) -> Dict[str, float]:
    if isinstance(value, Mapping):
        if all(relation in value for relation in RELATIONS):
            return {relation: float(value[relation]) for relation in RELATIONS}
        for key in ("scores", "logits", "relation_scores", "relation_logits"):
            if key in value:
                return relation_score_map(value[key])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return {relation: float(value[index]) for index, relation in enumerate(RELATIONS)}
    raise ValueError(f"Cannot parse relation scores from {type(value)}")


def score_prediction(scores: Mapping[str, float]) -> Tuple[str, float]:
    ranked = sorted(RELATIONS, key=lambda relation: float(scores[relation]), reverse=True)
    return ranked[0], float(scores[ranked[0]] - scores[ranked[1]])


def target_margin(scores: Mapping[str, float], gt: str, comparison: str) -> float:
    return float(scores[gt]) - float(scores[comparison])


def normalized_effect(delta: float, baseline_margin: float, eps: float = 1e-6) -> float:
    return float(delta / max(abs(float(baseline_margin)), eps))


# -----------------------------------------------------------------------------
# Input metadata
# -----------------------------------------------------------------------------


def resolve_baseline_path(args: argparse.Namespace) -> Path:
    if str(args.baseline_generation_jsonl).strip():
        return Path(str(args.baseline_generation_jsonl).strip())
    return Path(args.head_output_dir) / "baseline_generation.jsonl"


def load_baseline_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = deduplicate_rows(read_jsonl(path), ("sid",))
    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        prediction = normalize_relation(
            row.get("prediction", row.get("baseline_generation_prediction"))
        )
        if gt is None:
            raise RuntimeError(f"SID {sid}: invalid GT in baseline file")
        correct = row.get("correct")
        if correct is None:
            correct = prediction == gt
        result[sid] = {
            **dict(row),
            "sid": sid,
            "gt": gt,
            "prediction": prediction,
            "correct": bool(correct),
            "parsed": prediction is not None,
        }
    return result


def load_prior_routing(signed_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    summary_path = signed_dir / "sample_routing_summary.csv"
    clean_path = signed_dir / "clean_sample_states.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)
    summary = {int(row["sid"]): dict(row) for row in read_csv(summary_path)}
    clean = {int(row["sid"]): dict(row) for row in read_jsonl(clean_path)}
    return summary, clean


def comparison_relation(gt: str, generation_prediction: Optional[str]) -> str:
    if generation_prediction in RELATIONS and generation_prediction != gt:
        return str(generation_prediction)
    return OPPOSITE[gt]


# -----------------------------------------------------------------------------
# Decoder/final-readout resolution
# -----------------------------------------------------------------------------


def resolve_attr_path(root: Any, path: str) -> Any:
    value = root
    for token in str(path).split("."):
        if not token:
            continue
        value = getattr(value, token)
    return value


def resolve_final_norm(model: Any, decoder_path: str) -> torch.nn.Module:
    parent_path = str(decoder_path).rsplit(".", 1)[0]
    parent = resolve_attr_path(model, parent_path)
    for name in ("norm", "final_layernorm", "final_layer_norm", "ln_f"):
        module = getattr(parent, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    candidates: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if lowered.endswith(".norm") or lowered.endswith("final_layernorm") or lowered.endswith("ln_f"):
            candidates.append((name, module))
    if candidates:
        candidates.sort(key=lambda item: (len(item[0]), item[0]))
        return candidates[-1][1]
    raise RuntimeError(f"Unable to resolve final norm from decoder path {decoder_path}")


def resolve_output_embedding(model: Any) -> torch.nn.Module:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        module = getter()
        if isinstance(module, torch.nn.Module):
            return module
    for path in ("lm_head", "language_model.lm_head", "model.lm_head"):
        try:
            module = resolve_attr_path(model, path)
        except Exception:
            continue
        if isinstance(module, torch.nn.Module):
            return module
    raise RuntimeError("Unable to resolve LM output embedding")


def module_device_dtype(module: torch.nn.Module, fallback_device: torch.device) -> Tuple[torch.device, torch.dtype]:
    for parameter in module.parameters(recurse=True):
        return parameter.device, parameter.dtype
    return fallback_device, torch.float32


def tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported module output type: {type(output)}")


def replace_tensor_output(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise RuntimeError(f"Unsupported module output type: {type(output)}")


def resolve_post_attention_norm(layer: Any) -> Optional[torch.nn.Module]:
    for name in ("post_attention_layernorm", "post_attention_layer_norm", "ffn_layernorm"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    return None


# -----------------------------------------------------------------------------
# Clean late-stage capture
# -----------------------------------------------------------------------------


class LateStageCapture:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        late_layers: Sequence[int],
        prompt_last: int,
        attention_helper: Any,
        ioi: Any,
        final_norm: torch.nn.Module,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.late_layers = list(map(int, late_layers))
        self.prompt_last = int(prompt_last)
        self.attention_helper = attention_helper
        self.ioi = ioi
        self.final_norm = final_norm
        self.states: Dict[str, torch.Tensor] = {}
        self.events: Counter[str] = Counter()
        self.handles: List[Any] = []

    def _capture_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError(f"{name}: expected [1,S,H], got {tuple(tensor.shape)}")
        if not 0 <= self.prompt_last < int(tensor.shape[1]):
            raise RuntimeError(
                f"{name}: prompt-last {self.prompt_last} outside sequence length {int(tensor.shape[1])}"
            )
        self.states[name] = tensor[0, self.prompt_last].detach().float().cpu()
        self.events[name] += 1

    def __enter__(self) -> "LateStageCapture":
        for layer_index in self.late_layers:
            layer = self.decoder_layers[layer_index]
            attention = self.attention_helper.resolve_self_attention(layer)
            mlp = self.ioi.resolve_mlp(layer)
            post_norm = resolve_post_attention_norm(layer)

            def make_layer_pre(index: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{index} layer pre-hook missing hidden state")
                    self._capture_tensor(f"L{index}_input", inputs[0])
                return hook

            def make_attention_hook(index: int):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    self._capture_tensor(f"L{index}_attn_update", tensor_from_output(output))
                    return output
                return hook

            def make_post_norm_pre(index: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{index} post-attention norm missing residual")
                    self._capture_tensor(f"L{index}_post_attn", inputs[0])
                return hook

            def make_mlp_hook(index: int):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    self._capture_tensor(f"L{index}_mlp_update", tensor_from_output(output))
                    return output
                return hook

            def make_layer_hook(index: int):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    self._capture_tensor(f"L{index}_block", tensor_from_output(output))
                    return output
                return hook

            self.handles.append(layer.register_forward_pre_hook(make_layer_pre(layer_index)))
            self.handles.append(attention.register_forward_hook(make_attention_hook(layer_index)))
            if post_norm is not None:
                self.handles.append(post_norm.register_forward_pre_hook(make_post_norm_pre(layer_index)))
            self.handles.append(mlp.register_forward_hook(make_mlp_hook(layer_index)))
            self.handles.append(layer.register_forward_hook(make_layer_hook(layer_index)))

        def final_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
            self._capture_tensor("final_norm", tensor_from_output(output))
            return output

        self.handles.append(self.final_norm.register_forward_hook(final_hook))
        return self

    def validate(self) -> None:
        required = ["final_norm"]
        for layer in self.late_layers:
            required.extend(
                [
                    f"L{layer}_input",
                    f"L{layer}_attn_update",
                    f"L{layer}_post_attn",
                    f"L{layer}_mlp_update",
                    f"L{layer}_block",
                ]
            )
        missing = [name for name in required if name not in self.states]
        if missing:
            # Some architectures do not expose a post-attention norm. Reconstruct
            # exact residual state from block input + attention update.
            reconstructable: List[str] = []
            for name in list(missing):
                match = re.fullmatch(r"L(\d+)_post_attn", name)
                if match:
                    layer = int(match.group(1))
                    input_name = f"L{layer}_input"
                    update_name = f"L{layer}_attn_update"
                    if input_name in self.states and update_name in self.states:
                        self.states[name] = self.states[input_name] + self.states[update_name]
                        reconstructable.append(name)
            missing = [name for name in missing if name not in reconstructable]
        if missing:
            raise RuntimeError(f"Missing clean capture stages: {missing}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def residual_stage_keys(layers: Sequence[int]) -> List[str]:
    keys: List[str] = []
    for layer in layers:
        keys.extend([f"L{layer}_input", f"L{layer}_post_attn", f"L{layer}_block"])
    keys.append("final_norm")
    return keys


def all_vector_stage_keys(layers: Sequence[int]) -> List[str]:
    keys: List[str] = ["L26V_role"]
    for layer in layers:
        keys.extend(
            [
                f"L{layer}_input",
                f"L{layer}_attn_update",
                f"L{layer}_post_attn",
                f"L{layer}_mlp_update",
                f"L{layer}_block",
            ]
        )
    keys.append("final_norm")
    return keys


def stage_order(layers: Sequence[int]) -> List[str]:
    output: List[str] = ["L26V_role"]
    for layer in layers:
        output.extend([f"L{layer}_input", f"L{layer}_post_attn", f"L{layer}_block"])
    output.append("final_norm")
    return output


@torch.inference_mode()
def relation_scores_from_hidden(
    *,
    hidden: torch.Tensor,
    already_normalized: bool,
    final_norm: torch.nn.Module,
    output_embedding: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    fallback_device: torch.device,
) -> Dict[str, Any]:
    device, dtype = module_device_dtype(final_norm, fallback_device)
    value = hidden.detach().to(device=device, dtype=dtype).view(1, 1, -1)
    if not already_normalized:
        value = final_norm(value)
    logits = output_embedding(value)
    if logits.ndim != 3:
        raise RuntimeError(f"LM output embedding returned shape {tuple(logits.shape)}")
    relation_data = base.relation_scores(logits[0, 0], relation_token_map, gt=None)
    scores = {relation: float(relation_data["scores"][relation]) for relation in RELATIONS}
    prediction, margin = score_prediction(scores)
    return {"scores": scores, "prediction": prediction, "top_margin": margin}


# -----------------------------------------------------------------------------
# Cross-fitted relation models
# -----------------------------------------------------------------------------


@dataclass
class RelationModel:
    center: np.ndarray
    centroids: Dict[str, np.ndarray]
    directions: Dict[str, np.ndarray]
    basis: np.ndarray

    def scores(self, vector: np.ndarray) -> Dict[str, float]:
        centered = np.asarray(vector, dtype=np.float64) - self.center
        centered_unit = unit_vector(centered)
        return {
            relation: float(np.dot(centered_unit, self.directions[relation]))
            for relation in RELATIONS
        }

    def prediction(self, vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        scores = self.scores(vector)
        prediction, margin = score_prediction(scores)
        return prediction, margin, scores

    def project_centered(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if self.basis.shape[1] == 0:
            return np.zeros_like(value)
        centered = value - self.center
        return self.basis @ (self.basis.T @ centered)

    def erase(self, vector: np.ndarray, strength: float = 1.0) -> np.ndarray:
        return np.asarray(vector, dtype=np.float64) - float(strength) * self.project_centered(vector)

    def replace_relation(self, vector: np.ndarray, relation: str, strength: float = 1.0) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if self.basis.shape[1] == 0:
            return value.copy()
        current = self.basis.T @ (value - self.center)
        target = self.basis.T @ (self.centroids[relation] - self.center)
        return value + float(strength) * (self.basis @ (target - current))


def fit_relation_model(vectors: Sequence[np.ndarray], labels: Sequence[str]) -> RelationModel:
    by_relation: Dict[str, List[np.ndarray]] = defaultdict(list)
    values: List[np.ndarray] = []
    for vector, label in zip(vectors, labels):
        relation = normalize_relation(label)
        if relation is None:
            continue
        value = np.asarray(vector, dtype=np.float64)
        by_relation[relation].append(value)
        values.append(value)
    missing = [relation for relation in RELATIONS if not by_relation[relation]]
    if missing:
        raise RuntimeError(f"Relation-model training split misses classes {missing}")
    center = np.mean(np.stack(values, axis=0), axis=0)
    centroids = {
        relation: np.mean(np.stack(by_relation[relation], axis=0), axis=0)
        for relation in RELATIONS
    }
    directions = {
        relation: unit_vector(centroids[relation] - center)
        for relation in RELATIONS
    }
    horizontal = centroids["left"] - centroids["right"]
    vertical = centroids["above"] - centroids["below"]
    basis = orthonormal_basis([horizontal, vertical])
    return RelationModel(
        center=np.asarray(center, dtype=np.float64),
        centroids={key: np.asarray(value, dtype=np.float64) for key, value in centroids.items()},
        directions=directions,
        basis=basis,
    )


def group_key_for_row(source_row: Mapping[str, Any], sid: int) -> str:
    for key in ("image_id", "image_path", "coco_image_id", "uid"):
        value = source_row.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return f"sid:{sid}"


def fit_crossfit_models(
    *,
    sids: Sequence[int],
    gt_by_sid: Mapping[int, str],
    fold_by_sid: Mapping[int, int],
    vectors_by_sid: Mapping[int, Mapping[str, np.ndarray]],
    stage_keys: Sequence[str],
    folds: int,
) -> Dict[Tuple[int, str], RelationModel]:
    models: Dict[Tuple[int, str], RelationModel] = {}
    for fold in range(int(folds)):
        train_sids = [sid for sid in sids if int(fold_by_sid[sid]) != fold]
        for stage in stage_keys:
            vectors = [vectors_by_sid[sid][stage] for sid in train_sids]
            labels = [gt_by_sid[sid] for sid in train_sids]
            models[(fold, stage)] = fit_relation_model(vectors, labels)
    return models


# -----------------------------------------------------------------------------
# Patch hooks
# -----------------------------------------------------------------------------


class PrefillVectorPatch:
    """Patch one full hidden/update vector at prompt-last during prefill only."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        position: int,
        target: torch.Tensor,
    ) -> None:
        self.position = int(position)
        self.target = target.detach().float().cpu()
        self.prefill_events = 0
        self.decode_events = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        tensor = tensor_from_output(output)
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError("Vector patch expects [1,S,H]")
        if int(tensor.shape[1]) <= self.position:
            self.decode_events += 1
            return output
        if int(self.target.numel()) != int(tensor.shape[-1]):
            raise RuntimeError(
                f"Patch width {self.target.numel()} != module width {tensor.shape[-1]}"
            )
        modified = tensor.clone()
        modified[0, self.position] = self.target.to(device=tensor.device, dtype=tensor.dtype)
        self.prefill_events += 1
        return replace_tensor_output(output, modified)

    def validate(self) -> None:
        if self.prefill_events != 1:
            raise RuntimeError(f"Expected one patched prefill event, got {self.prefill_events}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


class PrefillVRolePatch:
    """Patch one shared KV-head Value role state at subject/reference positions."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        subject_position: int,
        reference_position: int,
        unit_head: int,
        head_dim: int,
        subject_target: np.ndarray,
        reference_target: np.ndarray,
    ) -> None:
        self.subject_position = int(subject_position)
        self.reference_position = int(reference_position)
        self.unit_head = int(unit_head)
        self.head_dim = int(head_dim)
        self.subject_target = torch.as_tensor(subject_target, dtype=torch.float32).cpu()
        self.reference_target = torch.as_tensor(reference_target, dtype=torch.float32).cpu()
        self.max_position = max(self.subject_position, self.reference_position)
        self.prefill_events = 0
        self.decode_events = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not torch.is_tensor(output) or output.ndim != 3 or int(output.shape[0]) != 1:
            raise RuntimeError("V-role patch expects [1,S,D]")
        if int(output.shape[1]) <= self.max_position:
            self.decode_events += 1
            return output
        start = self.unit_head * self.head_dim
        stop = start + self.head_dim
        if stop > int(output.shape[-1]):
            raise RuntimeError("V-role head slice outside projection width")
        modified = output.clone()
        modified[0, self.subject_position, start:stop] = self.subject_target.to(
            device=output.device, dtype=output.dtype
        )
        modified[0, self.reference_position, start:stop] = self.reference_target.to(
            device=output.device, dtype=output.dtype
        )
        self.prefill_events += 1
        return modified

    def validate(self) -> None:
        if self.prefill_events != 1:
            raise RuntimeError(f"Expected one V-role prefill patch, got {self.prefill_events}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


def score_model(
    *,
    model: Any,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
) -> Dict[str, Any]:
    with torch.inference_mode():
        outputs = model(**dict(batch), use_cache=False, return_dict=True)
    relation_data = base.relation_scores(outputs.logits[0, -1], relation_token_map, gt=None)
    scores = {relation: float(relation_data["scores"][relation]) for relation in RELATIONS}
    prediction, margin = score_prediction(scores)
    return {"scores": scores, "prediction": prediction, "top_margin": margin}


@torch.inference_mode()
def generate_with_installed_patch(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    max_new_tokens: int,
    generation_helper: Any,
) -> Dict[str, Any]:
    input_ids = batch.get("input_ids")
    if input_ids is None or not torch.is_tensor(input_ids):
        raise RuntimeError("Generation batch lacks input_ids")
    prompt_length = int(input_ids.shape[1])
    tokenizer = processor.tokenizer
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": False,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    started = time.perf_counter()
    generated_ids = model.generate(**dict(batch), **kwargs)
    elapsed = time.perf_counter() - started
    new_ids = generated_ids[0, prompt_length:].detach().cpu().tolist()
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    prediction = generation_helper.parse_generated_relation(text)
    return {
        "text": text,
        "prediction": normalize_relation(prediction),
        "parsed": prediction is not None,
        "new_token_count": len(new_ids),
        "generation_seconds": float(elapsed),
    }


# -----------------------------------------------------------------------------
# Clean extraction and offline trajectory analysis
# -----------------------------------------------------------------------------


def load_vector_state_files(
    *,
    selected_sids: Sequence[int],
    state_dir: Path,
    routing_vector_dir: Path,
    late_layers: Sequence[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    result: Dict[int, Dict[str, np.ndarray]] = {}
    required = all_vector_stage_keys(late_layers)
    for sid in selected_sids:
        state_path = state_dir / f"sid_{sid:06d}.npz"
        routing_path = routing_vector_dir / f"sid_{sid:06d}.npz"
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        if not routing_path.exists():
            raise FileNotFoundError(routing_path)
        values: Dict[str, np.ndarray] = {}
        with np.load(state_path, allow_pickle=False) as data:
            for key in data.files:
                values[str(key)] = np.asarray(data[key], dtype=np.float64)
        with np.load(routing_path, allow_pickle=False) as data:
            values["L26V_role"] = np.asarray(data["baseline_role"], dtype=np.float64)
            values["L26V_subject"] = np.asarray(data["baseline_subject"], dtype=np.float64)
            values["L26V_reference"] = np.asarray(data["baseline_reference"], dtype=np.float64)
        missing = [key for key in required if key not in values]
        if missing:
            raise RuntimeError(f"SID {sid}: missing state vectors {missing}")
        result[int(sid)] = values
    return result


def build_stage_trajectory(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    baseline_by_sid: Mapping[int, Mapping[str, Any]],
    prior_routing: Mapping[int, Mapping[str, Any]],
    clean_meta_by_sid: Mapping[int, Mapping[str, Any]],
    vectors_by_sid: Mapping[int, Mapping[str, np.ndarray]],
    logit_by_sid: Mapping[int, Mapping[str, Mapping[str, Any]]],
    late_layers: Sequence[int],
    probe_folds: int,
    seed: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[Tuple[int, str], RelationModel],
    Dict[int, int],
]:
    selected_sids = [int(row["sid"]) for row in selected_rows]
    source_by_sid = {int(row["sid"]): dict(row) for row in selected_rows}
    gt_by_sid = {sid: str(baseline_by_sid[sid]["gt"]) for sid in selected_sids}
    fold_by_sid = {
        sid: stable_fold(group_key_for_row(source_by_sid[sid], sid), int(probe_folds), int(seed))
        for sid in selected_sids
    }
    stage_keys = all_vector_stage_keys(late_layers)
    models = fit_crossfit_models(
        sids=selected_sids,
        gt_by_sid=gt_by_sid,
        fold_by_sid=fold_by_sid,
        vectors_by_sid=vectors_by_sid,
        stage_keys=stage_keys,
        folds=probe_folds,
    )

    stage_rows: List[Dict[str, Any]] = []
    stage_by_sid: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for sid in selected_sids:
        baseline = baseline_by_sid[sid]
        gt = str(baseline["gt"])
        generation_prediction = normalize_relation(baseline.get("prediction"))
        comparison = comparison_relation(gt, generation_prediction)
        fold = int(fold_by_sid[sid])
        for stage in stage_keys:
            model = models[(fold, stage)]
            prediction, top_margin, scores = model.prediction(vectors_by_sid[sid][stage])
            row: Dict[str, Any] = {
                "sid": sid,
                "gt": gt,
                "generation_prediction": generation_prediction,
                "baseline_correct": bool(baseline["correct"]),
                "comparison_relation": comparison,
                "fold": fold,
                "stage": stage,
                "probe_prediction": prediction,
                "probe_correct": bool(prediction == gt),
                "probe_agrees_generation": bool(
                    generation_prediction is not None and prediction == generation_prediction
                ),
                "probe_top_margin": float(top_margin),
                "probe_GT_vs_comparison_margin": target_margin(scores, gt, comparison),
                **{f"probe_score_{relation}": float(scores[relation]) for relation in RELATIONS},
            }
            if stage in logit_by_sid.get(sid, {}):
                lens = logit_by_sid[sid][stage]
                lens_scores = relation_score_map(lens["scores"])
                row.update(
                    {
                        "logit_prediction": normalize_relation(lens.get("prediction")),
                        "logit_correct": bool(normalize_relation(lens.get("prediction")) == gt),
                        "logit_agrees_generation": bool(
                            generation_prediction is not None
                            and normalize_relation(lens.get("prediction")) == generation_prediction
                        ),
                        "logit_top_margin": float(lens.get("top_margin", score_prediction(lens_scores)[1])),
                        "logit_GT_vs_comparison_margin": target_margin(lens_scores, gt, comparison),
                        **{f"logit_score_{relation}": float(lens_scores[relation]) for relation in RELATIONS},
                    }
                )
            else:
                row.update(
                    {
                        "logit_prediction": "",
                        "logit_correct": False,
                        "logit_agrees_generation": False,
                        "logit_top_margin": float("nan"),
                        "logit_GT_vs_comparison_margin": float("nan"),
                    }
                )
            stage_rows.append(row)
            stage_by_sid[sid][stage] = row

    residual_order = stage_order(late_layers)
    sample_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []

    for sid in selected_sids:
        baseline = baseline_by_sid[sid]
        gt = str(baseline["gt"])
        generation_prediction = normalize_relation(baseline.get("prediction"))
        comparison = comparison_relation(gt, generation_prediction)
        prior = prior_routing.get(sid, {})
        clean_meta = clean_meta_by_sid.get(sid, {})
        stages = stage_by_sid[sid]

        residual_probe = [stages[key]["probe_prediction"] for key in residual_order]
        residual_logit = [stages[key].get("logit_prediction") for key in residual_order]
        probe_gt_flags = [prediction == gt for prediction in residual_probe]
        logit_gt_flags = [prediction == gt for prediction in residual_logit if prediction]
        probe_generation_flags = [
            generation_prediction is not None and prediction == generation_prediction
            for prediction in residual_probe
        ]
        logit_generation_flags = [
            generation_prediction is not None and prediction == generation_prediction
            for prediction in residual_logit if prediction
        ]

        first_probe_gt = next((key for key, flag in zip(residual_order, probe_gt_flags) if flag), "")
        last_probe_gt = next(
            (key for key, flag in reversed(list(zip(residual_order, probe_gt_flags))) if flag),
            "",
        )
        first_probe_switch_to_generation = ""
        seen_gt = False
        for key, prediction in zip(residual_order, residual_probe):
            if prediction == gt:
                seen_gt = True
            elif seen_gt and generation_prediction is not None and prediction == generation_prediction:
                first_probe_switch_to_generation = key
                break

        first_logit_switch_to_generation = ""
        seen_logit_gt = False
        for key, prediction in zip(residual_order, residual_logit):
            if prediction == gt:
                seen_logit_gt = True
            elif seen_logit_gt and generation_prediction is not None and prediction == generation_prediction:
                first_logit_switch_to_generation = key
                break

        mid_keys = [key for key in residual_order if key.startswith(("L26_", "L27_", "L28_"))]
        late_keys = [key for key in residual_order if key.startswith(("L29_", "L30_", "L31_"))]
        mid_probe_gt_fraction = safe_mean(stages[key]["probe_prediction"] == gt for key in mid_keys)
        late_probe_gt_fraction = safe_mean(stages[key]["probe_prediction"] == gt for key in late_keys)
        mid_logit_gt_fraction = safe_mean(
            normalize_relation(stages[key].get("logit_prediction")) == gt for key in mid_keys
        )
        late_logit_gt_fraction = safe_mean(
            normalize_relation(stages[key].get("logit_prediction")) == gt for key in late_keys
        )

        all_l26_prediction = str(stages["L26V_role"]["probe_prediction"])
        prior_channel_prediction = normalize_relation(prior.get("channel_prediction"))
        final_closed_prediction = normalize_relation(
            clean_meta.get(
                "final_closed_prediction",
                clean_meta.get("clean_closed_prediction", baseline.get("closed_prediction")),
            )
        )
        final_closed_scores = None
        raw_final_scores = clean_meta.get(
            "final_closed_scores", clean_meta.get("clean_closed_scores")
        )
        if raw_final_scores not in (None, ""):
            try:
                final_closed_scores = relation_score_map(raw_final_scores)
            except Exception:
                final_closed_scores = None

        provisional = provisional_mechanism_class(
            baseline_correct=bool(baseline["correct"]),
            gt=gt,
            generation_prediction=generation_prediction,
            prior_l26_prediction=prior_channel_prediction,
            all_l26_prediction=all_l26_prediction,
            final_closed_prediction=final_closed_prediction,
            mid_probe_gt_fraction=mid_probe_gt_fraction,
            late_probe_gt_fraction=late_probe_gt_fraction,
            mid_logit_gt_fraction=mid_logit_gt_fraction,
            late_logit_gt_fraction=late_logit_gt_fraction,
            first_probe_switch=first_probe_switch_to_generation,
            first_logit_switch=first_logit_switch_to_generation,
        )

        sample_row = {
            "sid": sid,
            "gt": gt,
            "generation_prediction": generation_prediction,
            "baseline_correct": bool(baseline["correct"]),
            "comparison_relation": comparison,
            "fold": int(fold_by_sid[sid]),
            "prior_l26_prediction": prior_channel_prediction,
            "prior_l26_correct": bool(prior_channel_prediction == gt),
            "all_label_l26_prediction": all_l26_prediction,
            "all_label_l26_correct": bool(all_l26_prediction == gt),
            "l26_probe_agreement": bool(prior_channel_prediction == all_l26_prediction),
            "final_closed_prediction": final_closed_prediction,
            "final_closed_correct": bool(final_closed_prediction == gt),
            "residual_probe_GT_fraction": safe_mean(probe_gt_flags),
            "residual_logit_GT_fraction": safe_mean(logit_gt_flags),
            "residual_probe_generation_fraction": safe_mean(probe_generation_flags),
            "residual_logit_generation_fraction": safe_mean(logit_generation_flags),
            "mid_probe_GT_fraction": mid_probe_gt_fraction,
            "late_probe_GT_fraction": late_probe_gt_fraction,
            "mid_logit_GT_fraction": mid_logit_gt_fraction,
            "late_logit_GT_fraction": late_logit_gt_fraction,
            "first_probe_GT_stage": first_probe_gt,
            "last_probe_GT_stage": last_probe_gt,
            "first_probe_switch_to_generation": first_probe_switch_to_generation,
            "first_logit_switch_to_generation": first_logit_switch_to_generation,
            "provisional_mechanism_class": provisional,
            "subject_position": int(clean_meta.get("subject_position", -1)),
            "reference_position": int(clean_meta.get("reference_position", -1)),
            "prior_failure_class": str(prior.get("heuristic_failure_class", "")),
        }
        if final_closed_scores is not None:
            sample_row["final_closed_GT_vs_comparison_margin"] = target_margin(
                final_closed_scores, gt, comparison
            )
        else:
            sample_row["final_closed_GT_vs_comparison_margin"] = float("nan")
        sample_rows.append(sample_row)

        for layer in late_layers:
            input_row = stages[f"L{layer}_input"]
            post_row = stages[f"L{layer}_post_attn"]
            block_row = stages[f"L{layer}_block"]
            transition_rows.append(
                {
                    "sid": sid,
                    "gt": gt,
                    "generation_prediction": generation_prediction,
                    "baseline_correct": bool(baseline["correct"]),
                    "layer": int(layer),
                    "probe_input_margin": float(input_row["probe_GT_vs_comparison_margin"]),
                    "probe_post_attn_margin": float(post_row["probe_GT_vs_comparison_margin"]),
                    "probe_block_margin": float(block_row["probe_GT_vs_comparison_margin"]),
                    "probe_attention_update": float(
                        post_row["probe_GT_vs_comparison_margin"]
                        - input_row["probe_GT_vs_comparison_margin"]
                    ),
                    "probe_mlp_update": float(
                        block_row["probe_GT_vs_comparison_margin"]
                        - post_row["probe_GT_vs_comparison_margin"]
                    ),
                    "logit_input_margin": float(input_row.get("logit_GT_vs_comparison_margin", float("nan"))),
                    "logit_post_attn_margin": float(post_row.get("logit_GT_vs_comparison_margin", float("nan"))),
                    "logit_block_margin": float(block_row.get("logit_GT_vs_comparison_margin", float("nan"))),
                    "logit_attention_update": float(
                        post_row.get("logit_GT_vs_comparison_margin", float("nan"))
                        - input_row.get("logit_GT_vs_comparison_margin", float("nan"))
                    ),
                    "logit_mlp_update": float(
                        block_row.get("logit_GT_vs_comparison_margin", float("nan"))
                        - post_row.get("logit_GT_vs_comparison_margin", float("nan"))
                    ),
                    "attention_causes_probe_sign_flip": bool(
                        float(input_row["probe_GT_vs_comparison_margin"]) > 0
                        and float(post_row["probe_GT_vs_comparison_margin"]) <= 0
                    ),
                    "mlp_causes_probe_sign_flip": bool(
                        float(post_row["probe_GT_vs_comparison_margin"]) > 0
                        and float(block_row["probe_GT_vs_comparison_margin"]) <= 0
                    ),
                    "attention_causes_logit_sign_flip": bool(
                        float(input_row.get("logit_GT_vs_comparison_margin", float("nan"))) > 0
                        and float(post_row.get("logit_GT_vs_comparison_margin", float("nan"))) <= 0
                    ),
                    "mlp_causes_logit_sign_flip": bool(
                        float(post_row.get("logit_GT_vs_comparison_margin", float("nan"))) > 0
                        and float(block_row.get("logit_GT_vs_comparison_margin", float("nan"))) <= 0
                    ),
                    "provisional_mechanism_class": provisional,
                }
            )

    stage_summary = summarize_stage_accuracy(stage_rows)
    return stage_rows, sample_rows, transition_rows, stage_summary, models, fold_by_sid


def provisional_mechanism_class(
    *,
    baseline_correct: bool,
    gt: str,
    generation_prediction: Optional[str],
    prior_l26_prediction: Optional[str],
    all_l26_prediction: Optional[str],
    final_closed_prediction: Optional[str],
    mid_probe_gt_fraction: float,
    late_probe_gt_fraction: float,
    mid_logit_gt_fraction: float,
    late_logit_gt_fraction: float,
    first_probe_switch: str,
    first_logit_switch: str,
) -> str:
    if baseline_correct:
        if (
            all_l26_prediction == gt
            and mid_probe_gt_fraction >= 2.0 / 3.0
            and late_probe_gt_fraction >= 2.0 / 3.0
            and final_closed_prediction == gt
        ):
            return "internally_supported_correct_candidate"
        if (
            all_l26_prediction != gt
            and mid_probe_gt_fraction < 0.34
            and late_probe_gt_fraction < 0.34
            and final_closed_prediction != gt
        ):
            return "possible_lucky_or_unmeasured_correct_candidate"
        if mid_probe_gt_fraction < 0.5 and late_probe_gt_fraction >= 0.5:
            return "late_recovery_correct_candidate"
        return "mixed_correct_candidate"

    # Wrong generation.
    if prior_l26_prediction == gt and all_l26_prediction != gt:
        return "prior_l26_probe_not_replicated_candidate"
    if all_l26_prediction != gt:
        if generation_prediction is not None and mid_probe_gt_fraction < 0.34:
            return "upstream_representation_failure_candidate"
        return "mixed_upstream_candidate"
    if (
        mid_probe_gt_fraction >= 0.5
        and (first_probe_switch or first_logit_switch)
    ):
        return "downstream_utilization_failure_candidate"
    if (
        mid_probe_gt_fraction >= 0.5
        and late_probe_gt_fraction >= 0.5
        and final_closed_prediction == gt
    ):
        return "free_generation_writer_failure_candidate"
    if mid_probe_gt_fraction < 0.34 and mid_logit_gt_fraction < 0.34:
        return "l26_decodable_but_not_propagated_candidate"
    return "mixed_wrong_candidate"


def summarize_stage_accuracy(stage_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        status = "correct" if bool_value(row["baseline_correct"]) else "wrong"
        groups[(str(row["stage"]), "all")].append(row)
        groups[(str(row["stage"]), status)].append(row)
    output: List[Dict[str, Any]] = []
    for (stage, status), values in sorted(groups.items()):
        output.append(
            {
                "stage": stage,
                "status": status,
                "N": len(values),
                "probe_accuracy": safe_mean(bool(row["probe_correct"]) for row in values),
                "probe_agrees_generation": safe_mean(bool(row["probe_agrees_generation"]) for row in values),
                "probe_GT_vs_comparison_margin": safe_mean(
                    float(row["probe_GT_vs_comparison_margin"]) for row in values
                ),
                "logit_accuracy": safe_mean(bool(row.get("logit_correct", False)) for row in values),
                "logit_agrees_generation": safe_mean(
                    bool(row.get("logit_agrees_generation", False)) for row in values
                ),
                "logit_GT_vs_comparison_margin": safe_mean(
                    float(row.get("logit_GT_vs_comparison_margin", float("nan")))
                    for row in values
                ),
            }
        )
    return output


def summarize_transitions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        status = "correct" if bool_value(row["baseline_correct"]) else "wrong"
        groups[(int(row["layer"]), "all")].append(row)
        groups[(int(row["layer"]), status)].append(row)
        groups[(int(row["layer"]), str(row["provisional_mechanism_class"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (layer, group), values in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        output.append(
            {
                "layer": layer,
                "group": group,
                "N": len(values),
                "probe_attention_update_mean": safe_mean(float(row["probe_attention_update"]) for row in values),
                "probe_mlp_update_mean": safe_mean(float(row["probe_mlp_update"]) for row in values),
                "logit_attention_update_mean": safe_mean(float(row["logit_attention_update"]) for row in values),
                "logit_mlp_update_mean": safe_mean(float(row["logit_mlp_update"]) for row in values),
                "attention_probe_flip_rate": safe_mean(bool(row["attention_causes_probe_sign_flip"]) for row in values),
                "mlp_probe_flip_rate": safe_mean(bool(row["mlp_causes_probe_sign_flip"]) for row in values),
                "attention_logit_flip_rate": safe_mean(bool(row["attention_causes_logit_sign_flip"]) for row in values),
                "mlp_logit_flip_rate": safe_mean(bool(row["mlp_causes_logit_sign_flip"]) for row in values),
            }
        )
    return output


# -----------------------------------------------------------------------------
# Causal component parsing/selection
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: str  # l26v, attn_update, mlp_update, block, final_norm
    layer: Optional[int]
    stage_key: str


def parse_component(value: str, late_layers: Sequence[int]) -> ComponentSpec:
    token = str(value).strip()
    lowered = token.lower()
    if lowered == "l26v":
        return ComponentSpec("l26v", "l26v", 26, "L26V_role")
    if lowered == "final_norm":
        return ComponentSpec("final_norm", "final_norm", None, "final_norm")
    match = re.fullmatch(r"L(\d+)_(attn_update|mlp_update|block)", token, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid causal component {value!r}")
    layer = int(match.group(1))
    kind = match.group(2).lower()
    if layer not in set(map(int, late_layers)):
        raise ValueError(f"Component layer {layer} not in --late-layers")
    return ComponentSpec(f"L{layer}_{kind}", kind, layer, f"L{layer}_{kind}")


def causal_group_membership(row: Mapping[str, Any]) -> set[str]:
    correct = bool_value(row["baseline_correct"])
    gt = str(row["gt"])
    prior = normalize_relation(row.get("prior_l26_prediction"))
    all_l26 = normalize_relation(row.get("all_label_l26_prediction"))
    groups = {"all", "all_correct" if correct else "all_wrong"}
    if not correct and prior == gt:
        groups.add("wrong_prior_l26_gt")
    if not correct and all_l26 == gt:
        groups.add("wrong_all_l26_gt")
    if not correct and all_l26 != gt:
        groups.add("wrong_all_l26_wrong")
    if correct and all_l26 == gt:
        groups.add("correct_all_l26_gt")
    if correct and all_l26 != gt:
        groups.add("correct_all_l26_wrong")
    return groups


def stratified_limit_rows(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    values = [dict(row) for row in rows]
    if limit <= 0 or len(values) <= limit:
        return sorted(values, key=lambda row: int(row["sid"]))
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in values:
        groups[(str(row["gt"]), str(row["provisional_mechanism_class"]))].append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    selected: List[Dict[str, Any]] = []
    cursor = {key: 0 for key in keys}
    while len(selected) < limit:
        progressed = False
        for key in keys:
            index = cursor[key]
            if index < len(groups[key]):
                selected.append(groups[key][index])
                cursor[key] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda row: int(row["sid"]))


def select_causal_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    requested_groups: Sequence[str],
    max_per_group: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
    allowed = {
        "all",
        "all_wrong",
        "all_correct",
        "wrong_prior_l26_gt",
        "wrong_all_l26_gt",
        "wrong_all_l26_wrong",
        "correct_all_l26_gt",
        "correct_all_l26_wrong",
    }
    unknown = [group for group in requested_groups if group not in allowed]
    if unknown:
        raise ValueError(f"Unknown causal groups {unknown}; allowed={sorted(allowed)}")
    selected_by_sid: Dict[int, Dict[str, Any]] = {}
    labels_by_sid: Dict[int, List[str]] = defaultdict(list)
    for group_index, group in enumerate(requested_groups):
        candidates = [row for row in sample_rows if group in causal_group_membership(row)]
        limited = stratified_limit_rows(candidates, max_per_group, seed + group_index)
        for row in limited:
            sid = int(row["sid"])
            selected_by_sid[sid] = dict(row)
            labels_by_sid[sid].append(group)
    return sorted(selected_by_sid.values(), key=lambda row: int(row["sid"])), labels_by_sid


def build_component_module(
    *,
    component: ComponentSpec,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    ioi: Any,
    final_norm: torch.nn.Module,
) -> torch.nn.Module:
    if component.kind == "final_norm":
        return final_norm
    if component.layer is None:
        raise ValueError(component)
    layer = decoder_layers[int(component.layer)]
    if component.kind == "block":
        return layer
    if component.kind == "attn_update":
        return attention_helper.resolve_self_attention(layer)
    if component.kind == "mlp_update":
        return ioi.resolve_mlp(layer)
    raise ValueError(f"No full-vector module for {component}")


def intervention_target(
    *,
    model: RelationModel,
    clean_vector: np.ndarray,
    intervention: str,
    gt: str,
    counterfactual: str,
    strength: float,
) -> np.ndarray:
    if intervention == "erase":
        return model.erase(clean_vector, strength=strength)
    if intervention == "gt_replace":
        return model.replace_relation(clean_vector, gt, strength=strength)
    if intervention == "counterfactual_replace":
        return model.replace_relation(clean_vector, counterfactual, strength=strength)
    raise ValueError(intervention)


def l26v_targets(
    *,
    model: RelationModel,
    subject: np.ndarray,
    reference: np.ndarray,
    intervention: str,
    gt: str,
    counterfactual: str,
    strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    subject = np.asarray(subject, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mean = 0.5 * (subject + reference)
    role = subject - reference
    target_role = intervention_target(
        model=model,
        clean_vector=role,
        intervention=intervention,
        gt=gt,
        counterfactual=counterfactual,
        strength=strength,
    )
    return mean + 0.5 * target_role, mean - 0.5 * target_role


# -----------------------------------------------------------------------------
# Causal summaries and final taxonomy
# -----------------------------------------------------------------------------


def summarize_causal(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        status = "correct" if bool_value(row["baseline_correct"]) else "wrong"
        key_base = (str(row["component"]), str(row["intervention"]))
        groups[(key_base[0], key_base[1], "all")].append(row)
        groups[(key_base[0], key_base[1], status)].append(row)
        groups[(key_base[0], key_base[1], str(row["provisional_mechanism_class"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (component, intervention, group), values in sorted(groups.items()):
        output.append(
            {
                "component": component,
                "intervention": intervention,
                "group": group,
                "N": len(values),
                "baseline_margin_mean": safe_mean(float(row["baseline_GT_vs_comparison_margin"]) for row in values),
                "patched_margin_mean": safe_mean(float(row["patched_GT_vs_comparison_margin"]) for row in values),
                "margin_change_mean": safe_mean(float(row["margin_change"]) for row in values),
                "normalized_margin_change_mean": safe_mean(float(row["normalized_margin_change"]) for row in values),
                "patched_closed_accuracy": safe_mean(bool(row["patched_closed_correct"]) for row in values),
                "closed_prediction_changed_rate": safe_mean(bool(row["closed_prediction_changed"]) for row in values),
                "patched_generation_accuracy": safe_mean(
                    bool(row.get("patched_generation_correct", False))
                    for row in values
                    if row.get("patched_generation_prediction") not in (None, "")
                ),
                "generation_changed_rate": safe_mean(
                    bool(row.get("generation_prediction_changed", False))
                    for row in values
                    if row.get("patched_generation_prediction") not in (None, "")
                ),
            }
        )
    return output


def causal_features_by_sid(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, float]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sid"])].append(row)
    output: Dict[int, Dict[str, float]] = {}
    for sid, values in grouped.items():
        erase = [row for row in values if row["intervention"] == "erase"]
        gt_replace = [row for row in values if row["intervention"] == "gt_replace"]
        counter = [row for row in values if row["intervention"] == "counterfactual_replace"]
        # For erase, positive baseline - patched means the natural component supported GT.
        erase_support = [
            -float(row["normalized_margin_change"])
            for row in erase
        ]
        output[sid] = {
            "max_natural_GT_support": max(erase_support) if erase_support else float("nan"),
            "mean_natural_GT_support": safe_mean(erase_support),
            "max_GT_replace_gain": max(
                [float(row["normalized_margin_change"]) for row in gt_replace],
                default=float("nan"),
            ),
            "mean_GT_replace_gain": safe_mean(
                float(row["normalized_margin_change"]) for row in gt_replace
            ),
            "max_counterfactual_GT_loss": max(
                [-float(row["normalized_margin_change"]) for row in counter],
                default=float("nan"),
            ),
            "mean_counterfactual_GT_loss": safe_mean(
                -float(row["normalized_margin_change"]) for row in counter
            ),
            "causal_component_count": len({str(row["component"]) for row in values}),
        }
    return output


def refine_mechanism_class(
    row: Mapping[str, Any],
    causal: Optional[Mapping[str, float]],
    effect_threshold: float,
) -> str:
    provisional = str(row["provisional_mechanism_class"])
    if causal is None:
        return provisional
    natural = float(causal.get("max_natural_GT_support", float("nan")))
    repair = float(causal.get("max_GT_replace_gain", float("nan")))
    counter = float(causal.get("max_counterfactual_GT_loss", float("nan")))
    natural_ok = math.isfinite(natural) and natural >= effect_threshold
    repair_ok = math.isfinite(repair) and repair >= effect_threshold
    counter_ok = math.isfinite(counter) and counter >= effect_threshold

    if bool_value(row["baseline_correct"]):
        if (
            float(row["residual_probe_GT_fraction"]) >= 0.67
            and natural_ok
            and counter_ok
        ):
            return "causally_supported_correct_candidate"
        if (
            float(row["residual_probe_GT_fraction"]) < 0.34
            and not natural_ok
            and not counter_ok
        ):
            return "possible_lucky_or_unmeasured_correct_candidate"
        if repair_ok and float(row["late_probe_GT_fraction"]) >= 0.5:
            return "late_recovery_correct_candidate"
        return provisional

    if str(row["all_label_l26_prediction"]) == str(row["gt"]):
        if (
            float(row["mid_probe_GT_fraction"]) >= 0.5
            and (row.get("first_probe_switch_to_generation") or row.get("first_logit_switch_to_generation"))
            and (natural_ok or repair_ok or counter_ok)
        ):
            return "causal_downstream_utilization_failure_candidate"
        if (
            float(row["mid_probe_GT_fraction"]) < 0.34
            and not natural_ok
            and not repair_ok
        ):
            return "l26_decodable_but_unused_candidate"
        if repair_ok and not natural_ok:
            return "correct_relation_injectable_but_not_naturally_used_candidate"
    else:
        if repair_ok:
            return "upstream_wrong_but_late_stage_repairable_candidate"
        return "upstream_representation_failure_candidate"
    return provisional


def first_flip_summary(sample_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str]] = Counter()
    for row in sample_rows:
        if bool_value(row["baseline_correct"]):
            continue
        probe = str(row.get("first_probe_switch_to_generation", "") or "none")
        logit = str(row.get("first_logit_switch_to_generation", "") or "none")
        counts[("probe", probe)] += 1
        counts[("logit", logit)] += 1
    return [
        {"trajectory": trajectory, "first_switch_stage": stage, "N": count}
        for (trajectory, stage), count in sorted(counts.items())
    ]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.probe_folds < 2:
        raise ValueError("--probe-folds must be >=2")
    if args.patch_strength < 0:
        raise ValueError("--patch-strength must be >=0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    ioi = import_file(Path(args.ioi_script), "downstream_ioi")
    producer = import_file(Path(args.producer_script), "downstream_producer")
    receiver = import_file(Path(args.receiver_script), "downstream_receiver")
    v3 = import_file(Path(args.v3_script), "downstream_v3")
    base = import_file(Path(args.base_script), "downstream_base")
    attention_helper = import_file(Path(args.attention_helper), "downstream_attention")
    generation_helper = import_file(Path(args.generation_helper), "downstream_generation")
    routing_helper = import_file(Path(args.routing_script), "downstream_routing")

    source_config, source_rows = ioi.load_source_rows(args)
    baseline_path = resolve_baseline_path(args)
    baseline_by_sid = load_baseline_rows(baseline_path)
    prior_routing, prior_clean = load_prior_routing(Path(args.signed_routing_dir))
    routing_vector_dir = Path(args.signed_routing_dir) / "vectors"

    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(str(args.include_sids_file).strip()))
    excluded: set[int] = set()
    for raw in str(args.exclude_sids_from).split(","):
        item = raw.strip()
        if item:
            excluded.update(extract_sids(Path(item)))

    selected_rows = [
        dict(row)
        for row in source_rows
        if int(row["sid"]) in baseline_by_sid
        and int(row["sid"]) in prior_routing
        and int(row["sid"]) in prior_clean
        and int(row["sid"]) not in excluded
        and (included is None or int(row["sid"]) in included)
    ]
    selected_rows = sorted(selected_rows, key=lambda row: int(row["sid"]))
    if args.sample_max_samples > 0:
        selected_rows = producer.stratified_limit(
            selected_rows,
            int(args.sample_max_samples),
            int(args.seed),
        )
    if not selected_rows:
        raise RuntimeError("No eligible samples after source/baseline/routing intersection")

    selected_sids = [int(row["sid"]) for row in selected_rows]
    source_by_sid = {int(row["sid"]): dict(row) for row in selected_rows}

    model = None
    processor = None
    stage_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    stage_summary: List[Dict[str, Any]] = []
    models: Dict[Tuple[int, str], RelationModel] = {}
    fold_by_sid: Dict[int, int] = {}
    vectors_by_sid: Dict[int, Dict[str, np.ndarray]] = {}
    clean_meta_by_sid: Dict[int, Dict[str, Any]] = {}

    try:
        (
            model,
            processor,
            spec,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer.load_model_bundle(args=args, base=base)
        late_layers = parse_layer_spec(args.late_layers, len(decoder_layers))
        final_norm = resolve_final_norm(model, decoder_path)
        output_embedding = resolve_output_embedding(model)
        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "baseline_generation_jsonl": str(baseline_path),
            "signed_routing_dir": args.signed_routing_dir,
            "decoder_path": decoder_path,
            "late_layers": late_layers,
            "probe_folds": args.probe_folds,
            "probe_training_labels": "GT labels from all non-held-out samples, regardless of generation correctness",
            "selected_samples": len(selected_rows),
            "selected_sids": selected_sids,
            "phase": args.phase,
            "causal_components": parse_csv_tokens(args.causal_components),
            "causal_interventions": parse_csv_tokens(args.causal_interventions),
            "causal_groups": parse_csv_tokens(args.causal_groups),
            "causal_generate": bool(args.causal_generate),
            "audit": audit,
            "transformers_version": transformers.__version__,
            "interpretation_limits": [
                "Cross-fitted decodability is availability, not proof of use.",
                "Correct free generation is not assumed to imply correct internal reasoning.",
                "Candidate mechanism labels require causal follow-up and are not ground-truth error causes.",
            ],
        }
        write_json(output_dir / "config.json", config)

        # ------------------------------------------------------------------
        # Phase A: clean extraction
        # ------------------------------------------------------------------
        clean_path = output_dir / "clean_stage_states.jsonl"
        clean_rows = deduplicate_rows(read_jsonl(clean_path), ("sid",)) if args.resume else []
        clean_done = {
            int(row["sid"])
            for row in clean_rows
            if (state_dir / f"sid_{int(row['sid']):06d}.npz").exists()
        }

        if args.phase in ("extract", "all"):
            pending = [row for row in selected_rows if int(row["sid"]) not in clean_done]
            print(
                f"Clean late-stage extraction: samples={len(selected_rows)} pending={len(pending)} "
                f"layers={late_layers}",
                flush=True,
            )
            for index, source_row in enumerate(
                tqdm(pending, desc=f"late-trajectory:{args.model}"), start=1
            ):
                pair = None
                capture = None
                try:
                    sid = int(source_row["sid"])
                    baseline = baseline_by_sid[sid]
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
                    capture = LateStageCapture(
                        decoder_layers=decoder_layers,
                        late_layers=late_layers,
                        prompt_last=pair.original_prompt_last,
                        attention_helper=attention_helper,
                        ioi=ioi,
                        final_norm=final_norm,
                    )
                    with capture:
                        with torch.inference_mode():
                            outputs = model(
                                **dict(pair.original_batch),
                                use_cache=False,
                                return_dict=True,
                            )
                        capture.validate()
                    states = {name: tensor.numpy().astype(np.float32) for name, tensor in capture.states.items()}
                    state_path = state_dir / f"sid_{sid:06d}.npz"
                    np.savez_compressed(state_path, **states)

                    logit_lens: Dict[str, Any] = {}
                    for stage in residual_stage_keys(late_layers):
                        logit_lens[stage] = relation_scores_from_hidden(
                            hidden=torch.as_tensor(states[stage]),
                            already_normalized=(stage == "final_norm"),
                            final_norm=final_norm,
                            output_embedding=output_embedding,
                            relation_token_map=relation_token_map,
                            base=base,
                            fallback_device=torch.device(args.device),
                        )
                    final_scores = {
                        relation: float(
                            base.relation_scores(
                                outputs.logits[0, -1], relation_token_map, gt=None
                            )["scores"][relation]
                        )
                        for relation in RELATIONS
                    }
                    final_prediction, final_top_margin = score_prediction(final_scores)
                    row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": baseline["gt"],
                        "generation_prediction": baseline["prediction"],
                        "baseline_correct": bool(baseline["correct"]),
                        "prompt_last": int(pair.original_prompt_last),
                        "subject_position": int(pair.original_a_positions[-1]),
                        "reference_position": int(pair.original_b_positions[-1]),
                        "state_file": str(state_path),
                        "logit_lens": logit_lens,
                        "final_closed_scores": final_scores,
                        "final_closed_prediction": final_prediction,
                        "final_closed_top_margin": final_top_margin,
                    }
                    append_jsonl(clean_path, row)
                    clean_rows.append(row)
                    if args.print_every > 0 and index % args.print_every == 0:
                        print(
                            f"[extract {index}/{len(pending)} sid={sid}] "
                            f"gen={baseline['prediction']} closed={final_prediction} gt={baseline['gt']}",
                            flush=True,
                        )
                except Exception as exc:
                    error = {
                        "phase": "extract",
                        "sid": int(source_row.get("sid", -1)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    append_jsonl(errors_path, error)
                    print(
                        f"[ERROR extract sid={source_row.get('sid')}] {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                finally:
                    if capture is not None:
                        capture.close()
                    if pair is not None:
                        receiver.release_pair(pair)
                    gc.collect()
                    if (
                        args.device.startswith("cuda")
                        and args.empty_cache_every > 0
                        and index % args.empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()

        if args.phase == "extract":
            print(f"Saved clean late-stage states to {output_dir}")
            return

        # ------------------------------------------------------------------
        # Phase B: offline cross-fitted trajectory
        # ------------------------------------------------------------------
        if args.phase in ("analyze", "causal", "all"):
            clean_rows = deduplicate_rows(read_jsonl(clean_path), ("sid",))
            clean_meta_by_sid = {int(row["sid"]): dict(row) for row in clean_rows}
            missing_clean = [sid for sid in selected_sids if sid not in clean_meta_by_sid]
            if missing_clean:
                raise RuntimeError(f"Missing clean extraction for SIDs {missing_clean[:20]}")
            logit_by_sid = {
                sid: dict(clean_meta_by_sid[sid].get("logit_lens", {}))
                for sid in selected_sids
            }
            vectors_by_sid = load_vector_state_files(
                selected_sids=selected_sids,
                state_dir=state_dir,
                routing_vector_dir=routing_vector_dir,
                late_layers=late_layers,
            )
            (
                stage_rows,
                sample_rows,
                transition_rows,
                stage_summary,
                models,
                fold_by_sid,
            ) = build_stage_trajectory(
                selected_rows=selected_rows,
                baseline_by_sid=baseline_by_sid,
                prior_routing=prior_routing,
                clean_meta_by_sid=clean_meta_by_sid,
                vectors_by_sid=vectors_by_sid,
                logit_by_sid=logit_by_sid,
                late_layers=late_layers,
                probe_folds=args.probe_folds,
                seed=args.seed,
            )
            write_csv(output_dir / "stage_trajectory.csv", stage_rows)
            write_csv(output_dir / "sample_trajectory_summary.csv", sample_rows)
            write_csv(output_dir / "stage_accuracy_summary.csv", stage_summary)
            write_csv(
                output_dir / "component_transition_summary.csv",
                summarize_transitions(transition_rows),
            )
            write_csv(output_dir / "component_transitions.csv", transition_rows)
            write_csv(output_dir / "first_flip_summary.csv", first_flip_summary(sample_rows))

        # ------------------------------------------------------------------
        # Phase C: causal utilization tests
        # ------------------------------------------------------------------
        causal_rows: List[Dict[str, Any]] = []
        if args.phase in ("causal", "all"):
            if not sample_rows:
                sample_rows = read_csv(output_dir / "sample_trajectory_summary.csv")
            requested_components = [
                parse_component(token, late_layers)
                for token in parse_csv_tokens(args.causal_components)
            ]
            interventions = parse_csv_tokens(args.causal_interventions)
            allowed_interventions = {"erase", "gt_replace", "counterfactual_replace"}
            unknown_interventions = [value for value in interventions if value not in allowed_interventions]
            if unknown_interventions:
                raise ValueError(f"Unknown interventions {unknown_interventions}")
            causal_selected, causal_labels = select_causal_rows(
                sample_rows,
                parse_csv_tokens(args.causal_groups),
                int(args.causal_max_per_group),
                int(args.seed),
            )
            print(
                f"Causal utilization: samples={len(causal_selected)} components={len(requested_components)} "
                f"interventions={interventions} planned={len(causal_selected)*len(requested_components)*len(interventions)}",
                flush=True,
            )

            writer = ioi.WriterNode("attention", 26, 0)
            units = ioi.build_receiver_units(
                writers=[writer],
                channels=["v"],
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
                receiver_module=receiver,
            )
            if len(units) != 1:
                raise RuntimeError(f"Expected one L26 V receiver unit, got {len(units)}")
            receiver_unit = units[0]
            receiver_attention = attention_helper.resolve_self_attention(decoder_layers[26])
            receiver_shape = receiver.resolve_attention_shape(receiver_attention)
            v_projection = receiver.projection_module(receiver_attention, "v")
            kv_head_dim = int(receiver_shape.kv_head_dim)

            causal_path = output_dir / "causal_relation_interventions.jsonl"
            causal_rows = read_jsonl(causal_path) if args.resume else []
            done = {
                (int(row["sid"]), str(row["component"]), str(row["intervention"]))
                for row in causal_rows
            }
            sample_by_sid = {int(row["sid"]): dict(row) for row in sample_rows}

            task_index = 0
            for sample_index, sample_row in enumerate(
                tqdm(causal_selected, desc=f"causal-utilization:{args.model}"), start=1
            ):
                sid = int(sample_row["sid"])
                pair = None
                try:
                    source_row = source_by_sid[sid]
                    baseline = baseline_by_sid[sid]
                    gt = str(baseline["gt"])
                    generation_prediction = normalize_relation(baseline.get("prediction"))
                    counterfactual = comparison_relation(gt, generation_prediction)
                    fold = int(fold_by_sid[sid])
                    clean_meta = clean_meta_by_sid[sid]
                    baseline_scores = relation_score_map(clean_meta["final_closed_scores"])
                    baseline_closed_prediction = normalize_relation(clean_meta["final_closed_prediction"])
                    baseline_margin = target_margin(baseline_scores, gt, counterfactual)

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

                    for component in requested_components:
                        for intervention in interventions:
                            task_index += 1
                            key = (sid, component.name, intervention)
                            if key in done:
                                continue
                            patch: Any = None
                            try:
                                relation_model = models[(fold, component.stage_key)]
                                if component.kind == "l26v":
                                    subject = vectors_by_sid[sid]["L26V_subject"]
                                    reference = vectors_by_sid[sid]["L26V_reference"]
                                    subject_target, reference_target = l26v_targets(
                                        model=relation_model,
                                        subject=subject,
                                        reference=reference,
                                        intervention=intervention,
                                        gt=gt,
                                        counterfactual=counterfactual,
                                        strength=args.patch_strength,
                                    )
                                    patch = PrefillVRolePatch(
                                        module=v_projection,
                                        subject_position=int(clean_meta["subject_position"]),
                                        reference_position=int(clean_meta["reference_position"]),
                                        unit_head=int(receiver_unit.unit_head),
                                        head_dim=kv_head_dim,
                                        subject_target=subject_target,
                                        reference_target=reference_target,
                                    )
                                else:
                                    clean_vector = vectors_by_sid[sid][component.stage_key]
                                    target = intervention_target(
                                        model=relation_model,
                                        clean_vector=clean_vector,
                                        intervention=intervention,
                                        gt=gt,
                                        counterfactual=counterfactual,
                                        strength=args.patch_strength,
                                    )
                                    module = build_component_module(
                                        component=component,
                                        decoder_layers=decoder_layers,
                                        attention_helper=attention_helper,
                                        ioi=ioi,
                                        final_norm=final_norm,
                                    )
                                    patch = PrefillVectorPatch(
                                        module=module,
                                        position=int(pair.original_prompt_last),
                                        target=torch.as_tensor(target, dtype=torch.float32),
                                    )

                                patched = score_model(
                                    model=model,
                                    batch=pair.original_batch,
                                    relation_token_map=relation_token_map,
                                    base=base,
                                )
                                patch.validate()
                                patch.close()
                                patch = None

                                patched_scores = relation_score_map(patched["scores"])
                                patched_margin = target_margin(patched_scores, gt, counterfactual)
                                margin_change = patched_margin - baseline_margin
                                row: Dict[str, Any] = {
                                    "script_version": SCRIPT_VERSION,
                                    "model": args.model,
                                    "sid": sid,
                                    "gt": gt,
                                    "generation_prediction": generation_prediction,
                                    "baseline_correct": bool(baseline["correct"]),
                                    "comparison_relation": counterfactual,
                                    "causal_groups": ",".join(causal_labels[sid]),
                                    "provisional_mechanism_class": sample_by_sid[sid]["provisional_mechanism_class"],
                                    "component": component.name,
                                    "component_kind": component.kind,
                                    "component_layer": component.layer,
                                    "stage_key": component.stage_key,
                                    "intervention": intervention,
                                    "patch_strength": float(args.patch_strength),
                                    "baseline_closed_prediction": baseline_closed_prediction,
                                    "baseline_closed_scores": baseline_scores,
                                    "baseline_GT_vs_comparison_margin": float(baseline_margin),
                                    "patched_closed_prediction": normalize_relation(patched["prediction"]),
                                    "patched_closed_scores": patched_scores,
                                    "patched_closed_correct": bool(normalize_relation(patched["prediction"]) == gt),
                                    "patched_GT_vs_comparison_margin": float(patched_margin),
                                    "margin_change": float(margin_change),
                                    "normalized_margin_change": normalized_effect(margin_change, baseline_margin),
                                    "closed_prediction_changed": bool(
                                        normalize_relation(patched["prediction"]) != baseline_closed_prediction
                                    ),
                                }

                                if args.causal_generate:
                                    # Install a fresh identical patch because score_model consumed one prefill event.
                                    if component.kind == "l26v":
                                        patch = PrefillVRolePatch(
                                            module=v_projection,
                                            subject_position=int(clean_meta["subject_position"]),
                                            reference_position=int(clean_meta["reference_position"]),
                                            unit_head=int(receiver_unit.unit_head),
                                            head_dim=kv_head_dim,
                                            subject_target=subject_target,
                                            reference_target=reference_target,
                                        )
                                    else:
                                        module = build_component_module(
                                            component=component,
                                            decoder_layers=decoder_layers,
                                            attention_helper=attention_helper,
                                            ioi=ioi,
                                            final_norm=final_norm,
                                        )
                                        patch = PrefillVectorPatch(
                                            module=module,
                                            position=int(pair.original_prompt_last),
                                            target=torch.as_tensor(target, dtype=torch.float32),
                                        )
                                    generated = generate_with_installed_patch(
                                        model=model,
                                        processor=processor,
                                        batch=pair.original_batch,
                                        max_new_tokens=args.max_new_tokens,
                                        generation_helper=generation_helper,
                                    )
                                    patch.validate()
                                    patch.close()
                                    patch = None
                                    row.update(
                                        {
                                            "patched_generation": generated["text"],
                                            "patched_generation_prediction": generated["prediction"],
                                            "patched_generation_correct": bool(generated["prediction"] == gt),
                                            "generation_prediction_changed": bool(
                                                generated["prediction"] != generation_prediction
                                            ),
                                            "generation_seconds": float(generated["generation_seconds"]),
                                        }
                                    )

                                append_jsonl(causal_path, row)
                                causal_rows.append(row)
                                done.add(key)
                            finally:
                                if patch is not None:
                                    patch.close()

                    if args.print_every > 0 and sample_index % args.print_every == 0:
                        print(
                            f"[causal sample {sample_index}/{len(causal_selected)} sid={sid}] "
                            f"rows={len(causal_rows)}",
                            flush=True,
                        )
                except Exception as exc:
                    error = {
                        "phase": "causal",
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    append_jsonl(errors_path, error)
                    print(
                        f"[ERROR causal sid={sid}] {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                finally:
                    if pair is not None:
                        receiver.release_pair(pair)
                    gc.collect()
                    if (
                        args.device.startswith("cuda")
                        and args.empty_cache_every > 0
                        and sample_index % args.empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()

            causal_rows = deduplicate_rows(
                read_jsonl(causal_path),
                ("sid", "component", "intervention"),
            )
            write_csv(
                output_dir / "causal_intervention_summary.csv",
                summarize_causal(causal_rows),
            )

        # ------------------------------------------------------------------
        # Final candidate taxonomy and report
        # ------------------------------------------------------------------
        if not sample_rows:
            sample_rows = read_csv(output_dir / "sample_trajectory_summary.csv")
        if not causal_rows:
            causal_path = output_dir / "causal_relation_interventions.jsonl"
            causal_rows = deduplicate_rows(
                read_jsonl(causal_path),
                ("sid", "component", "intervention"),
            )
        causal_by_sid = causal_features_by_sid(causal_rows)
        final_rows: List[Dict[str, Any]] = []
        for row in sample_rows:
            sid = int(row["sid"])
            causal = causal_by_sid.get(sid)
            merged = {**dict(row)}
            if causal is not None:
                merged.update(causal)
            merged["refined_mechanism_class"] = refine_mechanism_class(
                row,
                causal,
                float(args.effect_threshold),
            )
            final_rows.append(merged)
        write_csv(output_dir / "sample_mechanism_summary.csv", final_rows)

        class_counts = Counter(str(row["refined_mechanism_class"]) for row in final_rows)
        prior_86 = [
            row
            for row in final_rows
            if (not bool_value(row["baseline_correct"]))
            and normalize_relation(row.get("prior_l26_prediction")) == normalize_relation(row.get("gt"))
        ]
        correct_rows = [row for row in final_rows if bool_value(row["baseline_correct"])]
        wrong_rows = [row for row in final_rows if not bool_value(row["baseline_correct"])]

        summary = {
            "script_version": SCRIPT_VERSION,
            "samples": len(final_rows),
            "correct": len(correct_rows),
            "wrong": len(wrong_rows),
            "baseline_generation_accuracy": safe_mean(bool_value(row["baseline_correct"]) for row in final_rows),
            "prior_l26_GT_generation_wrong_count": len(prior_86),
            "prior_l26_GT_replicated_by_all_label_axis": sum(
                normalize_relation(row.get("all_label_l26_prediction")) == normalize_relation(row.get("gt"))
                for row in prior_86
            ),
            "prior_l26_GT_multi_stage_mid_support": sum(
                float(row.get("mid_probe_GT_fraction", 0.0)) >= 0.5 for row in prior_86
            ),
            "prior_l26_GT_refined_class_counts": dict(
                Counter(str(row["refined_mechanism_class"]) for row in prior_86)
            ),
            "correct_refined_class_counts": dict(
                Counter(str(row["refined_mechanism_class"]) for row in correct_rows)
            ),
            "all_refined_class_counts": dict(class_counts),
            "interpretation": {
                "availability": "cross-fitted GT decodability from all-label training folds",
                "model_aligned": "shared final norm + LM-head logit-lens trajectory",
                "utilization": "relation-subspace erasure/replacement effect on final relation scores",
                "warning": "No single criterion proves correct spatial reasoning or lucky guessing.",
            },
        }
        write_json(output_dir / "summary.json", summary)

        print("\n" + "=" * 152)
        print("DOWNSTREAM RELATION TRAJECTORY / UTILIZATION RESULT")
        print("=" * 152)
        print(
            f"Samples={len(final_rows)} | correct={len(correct_rows)} | wrong={len(wrong_rows)} | "
            f"baseline_acc={summary['baseline_generation_accuracy']:.4f}"
        )
        print(
            "Prior L26-centroid-GT / generation-wrong group: "
            f"N={len(prior_86)} | all-label L26 axis still GT="
            f"{summary['prior_l26_GT_replicated_by_all_label_axis']} | "
            f"mid multi-stage GT support={summary['prior_l26_GT_multi_stage_mid_support']}"
        )
        print("\nREFINED MECHANISM CLASSES")
        for name, count in class_counts.most_common():
            print(f"{name:58s} {count:4d}")
        print("\nPRIOR L26-GT / GENERATION-WRONG SUBGROUP")
        for name, count in Counter(
            str(row["refined_mechanism_class"]) for row in prior_86
        ).most_common():
            print(f"{name:58s} {count:4d}")
        print(f"\nSaved outputs to {output_dir}")

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
