#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-only 2x2 causal factorization for spatial reasoning vs relation relay.

This is a go/no-go experiment for the question:

    Does a causally important spatial path actually COMPOSE visual layout with
    query role, or does it mainly RELAY a relation state formed elsewhere?

The experiment uses only horizontal COCO-two samples (left/right).  For each
sample it constructs four counterfactual cells:

    00: original image   + original query     -> original GT
    01: original image   + role-swapped query -> opposite GT
    10: horizontally flipped image + original query -> opposite GT
    11: horizontally flipped image + swapped query  -> original GT

The two factors are therefore orthogonal:

    V = visual-layout flip
    Q = query-role swap
    I = V x Q interaction / answer parity

For every selected internal component z, the exact 2x2 decomposition is:

    G = (z00 + z01 + z10 + z11) / 4
    V = (-z00 - z01 + z10 + z11) / 4
    Q = (-z00 + z01 - z10 + z11) / 4
    I = ( z00 - z01 - z10 + z11) / 4

The script traces the interaction across:

    early/mid object residuals
    P_POS7 pre-WO bundle output at identity-aligned object positions
    L26 shared Value head 0 at identity-aligned object positions
    L26 prompt-last attention output
    late prompt-last block states
    final norm

It then performs interaction-only interventions.  For a target cell with
interaction sign s_I in {+1,-1}, the interaction-flipped state is:

    z' = z - 2 * s_I * I

This changes only the factorial interaction term while preserving the grand
mean and the two main effects.  Same-norm random, main-effect-only, and whole
query-counterfactual patches are included as controls.

Important interpretation limits
-------------------------------
* This script does not claim that a large interaction norm is itself reasoning.
* A composition locus requires a localized increase plus causal specificity.
* A strong path with interaction already present upstream is evidence for relay.
* Horizontal flip is used only for left/right.  No Controlled-A data and no
  vertical image flip are used.

Main outputs
------------
factorial_cells.jsonl
vectors/sid_XXXXXX.npz
sample_factorization.csv
component_factorization_summary.csv
four_cell_equivariance.csv
causal_interaction_effect.jsonl
causal_interaction_summary.csv
go_no_go.json
config.json
errors.jsonl
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-reasoning-vs-relay-factorial-v2"
RELATIONS = ("left", "right", "above", "below")
HORIZONTAL = ("left", "right")
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
CELL_NAMES = ("00", "01", "10", "11")
CELL_BITS = {
    "00": (0, 0),
    "01": (0, 1),
    "10": (1, 0),
    "11": (1, 1),
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
        help="Optional original-query free-generation rows. Empty uses <head-output-dir>/baseline_generation.jsonl.",
    )
    p.add_argument(
        "--head-output-dir",
        default="output/coco_ioi_backward/qwen-3b_head_misrouting_pos7_neg5",
    )

    p.add_argument("--bundle-json", default="coco_ioi_role_bundles_v1.json")
    p.add_argument("--bundle-name", default="P_POS7")
    p.add_argument("--receiver-layer", type=int, default=26)
    p.add_argument("--receiver-query-head", type=int, default=0)
    p.add_argument("--receiver-kv-head", type=int, default=0)

    p.add_argument(
        "--object-block-layers",
        default="0,8,16,19,23",
        help="Block outputs read at identity-aligned A/B object positions.",
    )
    p.add_argument(
        "--prompt-block-layers",
        default="26,28,30",
        help="Block outputs read at prompt-last.",
    )
    p.add_argument("--probe-folds", type=int, default=5)
    p.add_argument("--bootstrap-repeats", type=int, default=1000)

    p.add_argument(
        "--causal-components",
        default="P_POS7,L26V,L26_attn,L26_block,L28_block,L30_block,final_norm",
    )
    p.add_argument(
        "--causal-interventions",
        default="interaction_flip,interaction_remove,random_matched,whole_query",
        help=(
            "Comma-separated subset of interaction_flip,interaction_remove,"
            "visual_flip,query_flip,random_matched,whole_query."
        ),
    )
    p.add_argument(
        "--causal-target-cells",
        default="00,01",
        help="Target cells to patch. 00,01 gives bidirectional query-role validation on the natural image.",
    )
    p.add_argument("--causal-max-samples", type=int, default=0)
    p.add_argument("--patch-strength", type=float, default=1.0)
    p.add_argument(
        "--min-margin-denominator",
        type=float,
        default=1e-4,
        help="Skip causal target cells whose average full counterfactual margin change is smaller than this value.",
    )

    p.add_argument("--sample-max-samples", type=int, default=0)
    p.add_argument("--include-sids-file", default="")
    p.add_argument("--exclude-sids-from", default="")
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument("--go-min-interaction-accuracy", type=float, default=0.70)
    p.add_argument("--go-min-localization-range", type=float, default=0.10)
    p.add_argument("--go-min-causal-effect", type=float, default=0.15)
    p.add_argument("--go-min-causal-specificity", type=float, default=0.08)
    p.add_argument("--go-min-positive-rate", type=float, default=0.65)

    p.add_argument("--ioi-script", default="analyze_coco_ioi_backward_circuit_v1.py")
    p.add_argument("--producer-script", default="analyze_coco_producer_qk_ov_v1.py")
    p.add_argument("--receiver-script", default="analyze_coco_receiver_qkv_v1.py")
    p.add_argument("--v3-script", default="analyze_spatial_storage_transport_utilization_v3.py")
    p.add_argument("--base-script", default="analyze_coco_centroid_generation_step1_v4.py")
    p.add_argument("--attention-helper", default="analyze_coco_flip_attention_spatial_vectors_v1.py")

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


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if math.isfinite(float(b)) and abs(float(b)) > 1e-12 else float(default)


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {"left_of": "left", "right_of": "right", "on": "above", "under": "below"}
    if text in RELATIONS:
        return text
    return aliases.get(text)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_csv_tokens(text: str) -> List[str]:
    return list(dict.fromkeys(item.strip() for item in str(text).split(",") if item.strip()))


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


def parse_head(value: Any) -> Tuple[int, int]:
    text = str(value).strip()
    if text.startswith("L") and "H" in text:
        layer, head = text[1:].split("H", 1)
        return int(layer), int(head)
    if ":" in text:
        layer, head = text.split(":", 1)
        return int(layer), int(head)
    raise ValueError(f"Invalid head {value!r}")


def head_name(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def load_bundle(path: Path, name: str) -> List[Tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("bundles", payload)
    if name not in source:
        raise KeyError(f"Bundle {name!r} not found; available={sorted(source)}")
    heads = [parse_head(value) for value in source[name]]
    return sorted(set(heads))


def stable_fold(key: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(folds)


def group_key(row: Mapping[str, Any], sid: int) -> str:
    for key in ("image_id", "coco_image_id", "image_path", "uid"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return f"sid:{sid}"


def unit_vector(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    if not math.isfinite(norm) or norm <= eps:
        return np.zeros_like(x)
    return x / norm


def deterministic_orthogonal_control(
    *,
    reference: np.ndarray,
    nuisance_vectors: Sequence[np.ndarray],
    key: str,
    seed: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Same-norm deterministic random control orthogonal to span(V,Q,I)."""
    ref = np.asarray(reference, dtype=np.float64)
    target_norm = float(np.linalg.norm(ref))
    if not math.isfinite(target_norm) or target_norm <= eps:
        return np.zeros_like(ref)

    columns = []
    for nuisance in nuisance_vectors:
        value = np.asarray(nuisance, dtype=np.float64).reshape(-1)
        if float(np.linalg.norm(value)) > eps:
            columns.append(value)
    basis = np.zeros((ref.size, 0), dtype=np.float64)
    if columns:
        matrix = np.stack(columns, axis=1)
        u, singular, _vh = np.linalg.svd(matrix, full_matrices=False)
        keep = singular > eps * max(float(singular[0]), 1.0)
        basis = u[:, keep]

    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(rng_seed)
    value = rng.standard_normal(ref.shape).reshape(-1)
    if basis.shape[1]:
        value = value - basis @ (basis.T @ value)
    value = unit_vector(value, eps=eps)
    if float(np.linalg.norm(value)) <= eps:
        value = np.roll(ref, 1).reshape(-1)
        if basis.shape[1]:
            value = value - basis @ (basis.T @ value)
        value = unit_vector(value, eps=eps)
    return value.reshape(ref.shape) * target_norm


def relation_score_map(value: Any) -> Dict[str, float]:
    if isinstance(value, Mapping):
        if all(relation in value for relation in RELATIONS):
            return {relation: float(value[relation]) for relation in RELATIONS}
        for key in ("scores", "logits", "relation_scores", "relation_logits"):
            if key in value:
                return relation_score_map(value[key])
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        flat = np.asarray(value).reshape(-1)
        if flat.size == 4:
            return {relation: float(flat[index]) for index, relation in enumerate(RELATIONS)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return {relation: float(value[index]) for index, relation in enumerate(RELATIONS)}
    raise ValueError(f"Cannot parse relation scores from {type(value)}")


def score_prediction(scores: Mapping[str, float]) -> Tuple[str, float]:
    ranked = sorted(RELATIONS, key=lambda relation: float(scores[relation]), reverse=True)
    return ranked[0], float(scores[ranked[0]] - scores[ranked[1]])


def horizontal_margin(scores: Mapping[str, float], expected: str) -> float:
    if expected not in HORIZONTAL:
        raise ValueError(expected)
    return float(scores[expected]) - float(scores[OPPOSITE[expected]])


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


def resolve_attr_path(root: Any, path: str) -> Any:
    value = root
    for token in str(path).split("."):
        if token:
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
    if not candidates:
        raise RuntimeError(f"Unable to resolve final norm from decoder path {decoder_path}")
    candidates.sort(key=lambda item: (len(item[0]), item[0]))
    return candidates[-1][1]


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


def stratified_limit(rows: Sequence[Mapping[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    values = [dict(row) for row in rows]
    if limit <= 0 or len(values) <= limit:
        return sorted(values, key=lambda row: int(row["sid"]))
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in values:
        groups[str(row["gt"])].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected: List[Dict[str, Any]] = []
    cursor = {key: 0 for key in groups}
    keys = [key for key in HORIZONTAL if key in groups]
    while len(selected) < limit:
        progressed = False
        for key in keys:
            idx = cursor[key]
            if idx < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][idx])
                cursor[key] += 1
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: int(row["sid"]))


def bootstrap_mean_ci(values: Sequence[float], repeats: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1 or repeats <= 0:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        means[index] = x[rng.integers(0, x.size, size=x.size)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# -----------------------------------------------------------------------------
# Baseline/source metadata
# -----------------------------------------------------------------------------


def resolve_baseline_path(args: argparse.Namespace) -> Path:
    if str(args.baseline_generation_jsonl).strip():
        return Path(str(args.baseline_generation_jsonl).strip())
    return Path(args.head_output_dir) / "baseline_generation.jsonl"


def load_baseline_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = deduplicate_rows(read_jsonl(path), ("sid",))
    output: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        prediction = normalize_relation(row.get("prediction", row.get("baseline_generation_prediction")))
        correct = row.get("correct")
        if correct is None and gt is not None:
            correct = prediction == gt
        output[sid] = {
            **dict(row),
            "sid": sid,
            "gt": gt,
            "prediction": prediction,
            "correct": bool(correct),
        }
    return output


# -----------------------------------------------------------------------------
# Four-cell prompt/image preparation
# -----------------------------------------------------------------------------


@dataclass
class CellData:
    name: str
    visual_flip: int
    query_swap: int
    batch: Dict[str, Any]
    ids: List[int]
    a_positions: List[int]  # original subject identity A, irrespective of query role
    b_positions: List[int]  # original reference identity B
    prompt_last: int
    expected_relation: str

    @property
    def x_visual(self) -> int:
        return -1 if self.visual_flip == 0 else 1

    @property
    def x_query(self) -> int:
        return -1 if self.query_swap == 0 else 1

    @property
    def x_interaction(self) -> int:
        return self.x_visual * self.x_query


@dataclass
class FourCellSample:
    sid: int
    gt: str
    subject: str
    reference: str
    image: Image.Image
    cells: Dict[str, CellData]
    original_pair: Any


def locate_identity_positions(
    *,
    batch: Mapping[str, Any],
    subject: str,
    reference: str,
    query_swap: int,
    tokenizer: Any,
    base: Any,
    v3: Any,
    object_state: str,
) -> Tuple[List[int], List[int], List[int]]:
    ids = batch["input_ids"][0].detach().cpu().tolist()
    if query_swap == 0:
        a_span, b_span = base.locate_object_spans(tokenizer, ids, subject, reference)
    else:
        b_span, a_span = base.locate_object_spans(tokenizer, ids, reference, subject)
    a_positions = list(map(int, v3.span_positions(a_span, object_state)))
    b_positions = list(map(int, v3.span_positions(b_span, object_state)))
    if not a_positions or not b_positions:
        raise RuntimeError("Missing identity-aligned object positions")
    return ids, a_positions, b_positions


def prepare_four_cells(
    *,
    args: argparse.Namespace,
    source_row: Mapping[str, Any],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    base: Any,
    v3: Any,
    receiver: Any,
    processor: Any,
    device: torch.device,
) -> FourCellSample:
    sid = int(source_row["sid"])
    gt = normalize_relation(source_row.get("gt"))
    if gt not in HORIZONTAL:
        raise ValueError(f"SID {sid}: horizontal experiment received GT={gt}")
    pair = receiver.prepare_pair(
        args=args,
        row=source_row,
        records_by_sid=records_by_sid,
        prompt_rows=prompt_rows,
        base=base,
        v3=v3,
        processor=processor,
        device=device,
    )
    prompt = prompt_rows[sid]
    subject = str(prompt["subject"])
    reference = str(prompt["reference"])
    original_question = str(prompt["question_text"])
    swapped_question = base.build_swapped_question(subject, reference)

    cells: Dict[str, CellData] = {}
    original_ids = pair.original_ids
    swapped_ids = pair.swapped_ids
    cells["00"] = CellData(
        name="00",
        visual_flip=0,
        query_swap=0,
        batch=pair.original_batch,
        ids=original_ids,
        a_positions=list(map(int, pair.original_a_positions)),
        b_positions=list(map(int, pair.original_b_positions)),
        prompt_last=int(pair.original_prompt_last),
        expected_relation=gt,
    )
    cells["01"] = CellData(
        name="01",
        visual_flip=0,
        query_swap=1,
        batch=pair.swapped_batch,
        ids=swapped_ids,
        a_positions=list(map(int, pair.swapped_a_positions)),
        b_positions=list(map(int, pair.swapped_b_positions)),
        prompt_last=int(pair.swapped_prompt_last),
        expected_relation=OPPOSITE[gt],
    )

    flipped = pair.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    try:
        flip_original_batch = base.make_question_batch(
            processor=processor,
            image=flipped,
            question_text=original_question,
            device=device,
        )
        flip_swapped_batch = base.make_question_batch(
            processor=processor,
            image=flipped,
            question_text=swapped_question,
            device=device,
        )
    finally:
        with contextlib.suppress(Exception):
            flipped.close()

    ids10, a10, b10 = locate_identity_positions(
        batch=flip_original_batch,
        subject=subject,
        reference=reference,
        query_swap=0,
        tokenizer=processor.tokenizer,
        base=base,
        v3=v3,
        object_state=args.object_state,
    )
    ids11, a11, b11 = locate_identity_positions(
        batch=flip_swapped_batch,
        subject=subject,
        reference=reference,
        query_swap=1,
        tokenizer=processor.tokenizer,
        base=base,
        v3=v3,
        object_state=args.object_state,
    )
    cells["10"] = CellData(
        name="10",
        visual_flip=1,
        query_swap=0,
        batch=flip_original_batch,
        ids=ids10,
        a_positions=a10,
        b_positions=b10,
        prompt_last=len(ids10) - 1,
        expected_relation=OPPOSITE[gt],
    )
    cells["11"] = CellData(
        name="11",
        visual_flip=1,
        query_swap=1,
        batch=flip_swapped_batch,
        ids=ids11,
        a_positions=a11,
        b_positions=b11,
        prompt_last=len(ids11) - 1,
        expected_relation=gt,
    )
    return FourCellSample(
        sid=sid,
        gt=gt,
        subject=subject,
        reference=reference,
        image=pair.image,
        cells=cells,
        original_pair=pair,
    )


def release_four_cells(sample: Optional[FourCellSample], receiver: Any) -> None:
    if sample is None:
        return
    for name in ("10", "11"):
        cell = sample.cells.get(name)
        if cell is not None and isinstance(cell.batch, dict):
            cell.batch.clear()
    receiver.release_pair(sample.original_pair)


# -----------------------------------------------------------------------------
# Capture specification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleHeadSpec:
    layer: int
    head: int
    head_dim: int
    offset_start: int
    offset_stop: int

    @property
    def name(self) -> str:
        return head_name(self.layer, self.head)


@dataclass
class CaptureSpec:
    object_layers: List[int]
    prompt_layers: List[int]
    bundle_heads: List[BundleHeadSpec]
    receiver_layer: int
    receiver_unit_head: int
    receiver_head_dim: int

    @property
    def pair_components(self) -> set[str]:
        return {
            *(f"L{layer}_object" for layer in self.object_layers),
            "P_POS7",
            "L26V",
        }

    @property
    def component_order(self) -> List[str]:
        output = [f"L{layer}_object" for layer in self.object_layers]
        output.extend(["P_POS7", "L26V", "L26_attn"])
        output.extend(f"L{layer}_block" for layer in self.prompt_layers)
        output.append("final_norm")
        return list(dict.fromkeys(output))


def build_capture_spec(
    *,
    args: argparse.Namespace,
    decoder_layers: Sequence[Any],
    bundle_heads: Sequence[Tuple[int, int]],
    attention_helper: Any,
    receiver: Any,
    ioi: Any,
) -> Tuple[CaptureSpec, Any]:
    object_layers = parse_layer_spec(args.object_block_layers, len(decoder_layers))
    prompt_layers = parse_layer_spec(args.prompt_block_layers, len(decoder_layers))
    offset = 0
    head_specs: List[BundleHeadSpec] = []
    for layer, head in bundle_heads:
        attention = attention_helper.resolve_self_attention(decoder_layers[layer])
        shape = receiver.resolve_attention_shape(attention)
        if not 0 <= head < int(shape.n_query_heads):
            raise ValueError(f"{head_name(layer, head)} outside n_query_heads={shape.n_query_heads}")
        width = int(shape.query_head_dim)
        head_specs.append(BundleHeadSpec(layer, head, width, offset, offset + width))
        offset += width

    writer = ioi.WriterNode("attention", int(args.receiver_layer), int(args.receiver_query_head))
    units = ioi.build_receiver_units(
        writers=[writer],
        channels=["v"],
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver,
    )
    if len(units) != 1:
        raise RuntimeError(f"Expected one L26 Value unit, got {len(units)}")
    unit = units[0]
    if int(unit.kv_head) != int(args.receiver_kv_head):
        raise RuntimeError(
            f"Requested receiver KVH{args.receiver_kv_head}, but QH{args.receiver_query_head} maps to KVH{unit.kv_head}"
        )
    attention = attention_helper.resolve_self_attention(decoder_layers[int(unit.layer)])
    shape = receiver.resolve_attention_shape(attention)
    spec = CaptureSpec(
        object_layers=object_layers,
        prompt_layers=prompt_layers,
        bundle_heads=head_specs,
        receiver_layer=int(unit.layer),
        receiver_unit_head=int(unit.unit_head),
        receiver_head_dim=int(shape.kv_head_dim),
    )
    return spec, unit


class FactorCapture:
    def __init__(
        self,
        *,
        cell: CellData,
        spec: CaptureSpec,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver: Any,
        ioi: Any,
        final_norm: torch.nn.Module,
    ) -> None:
        self.cell = cell
        self.spec = spec
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver = receiver
        self.ioi = ioi
        self.final_norm = final_norm
        self.vectors: Dict[str, torch.Tensor] = {}
        self.means: Dict[str, torch.Tensor] = {}
        self.bundle_a: Dict[str, torch.Tensor] = {}
        self.bundle_b: Dict[str, torch.Tensor] = {}
        self.handles: List[Any] = []
        self.events: Counter[str] = Counter()

    def _pair_from_hidden(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError(f"{name}: expected [1,S,H], got {tuple(tensor.shape)}")
        a = tensor[0, int(self.cell.a_positions[-1])]
        b = tensor[0, int(self.cell.b_positions[-1])]
        self.vectors[name] = (a - b).detach().float().cpu()
        self.means[name] = (0.5 * (a + b)).detach().float().cpu()
        self.events[name] += 1

    def _prompt_from_hidden(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError(f"{name}: expected [1,S,H], got {tuple(tensor.shape)}")
        self.vectors[name] = tensor[0, int(self.cell.prompt_last)].detach().float().cpu()
        self.events[name] += 1

    def __enter__(self) -> "FactorCapture":
        # Object identity-aligned block states.
        for layer_index in self.spec.object_layers:
            layer = self.decoder_layers[layer_index]

            def make_object_hook(index: int):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    self._pair_from_hidden(f"L{index}_object", tensor_from_output(output))
                    return output
                return hook

            self.handles.append(layer.register_forward_hook(make_object_hook(layer_index)))

        # P_POS7 pre-WO head outputs at identity A/B positions.
        grouped: Dict[int, List[BundleHeadSpec]] = defaultdict(list)
        for head_spec in self.spec.bundle_heads:
            grouped[int(head_spec.layer)].append(head_spec)
        for layer_index, head_specs in grouped.items():
            attention = self.attention_helper.resolve_self_attention(self.decoder_layers[layer_index])
            module = self.ioi.output_projection_module(attention)
            shape = self.receiver.resolve_attention_shape(attention)

            def make_bundle_pre(index: int, local_specs: List[BundleHeadSpec], head_dim: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{index} P_POS7 pre-WO hook missing tensor")
                    tensor = inputs[0]
                    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                        raise RuntimeError("P_POS7 pre-WO input must be [1,S,D]")
                    a_pos = int(self.cell.a_positions[-1])
                    b_pos = int(self.cell.b_positions[-1])
                    for item in local_specs:
                        start = int(item.head) * int(head_dim)
                        stop = start + int(head_dim)
                        self.bundle_a[item.name] = tensor[0, a_pos, start:stop].detach().float().cpu()
                        self.bundle_b[item.name] = tensor[0, b_pos, start:stop].detach().float().cpu()
                        self.events[f"bundle:{item.name}"] += 1
                return hook

            self.handles.append(
                module.register_forward_pre_hook(
                    make_bundle_pre(layer_index, head_specs, int(shape.query_head_dim))
                )
            )

        # L26 shared Value KV head at identity A/B positions.
        receiver_attention = self.attention_helper.resolve_self_attention(
            self.decoder_layers[self.spec.receiver_layer]
        )
        v_module = self.receiver.projection_module(receiver_attention, "v")

        def v_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
            if not torch.is_tensor(output) or output.ndim != 3 or int(output.shape[0]) != 1:
                raise RuntimeError("L26V output must be [1,S,D]")
            start = self.spec.receiver_unit_head * self.spec.receiver_head_dim
            stop = start + self.spec.receiver_head_dim
            a = output[0, int(self.cell.a_positions[-1]), start:stop]
            b = output[0, int(self.cell.b_positions[-1]), start:stop]
            self.vectors["L26V"] = (a - b).detach().float().cpu()
            self.means["L26V"] = (0.5 * (a + b)).detach().float().cpu()
            self.events["L26V"] += 1
            return output

        self.handles.append(v_module.register_forward_hook(v_hook))

        # L26 prompt-last attention output.
        def attn_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
            self._prompt_from_hidden("L26_attn", tensor_from_output(output))
            return output

        self.handles.append(receiver_attention.register_forward_hook(attn_hook))

        # Prompt-last block states.
        for layer_index in self.spec.prompt_layers:
            layer = self.decoder_layers[layer_index]

            def make_prompt_hook(index: int):
                def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    self._prompt_from_hidden(f"L{index}_block", tensor_from_output(output))
                    return output
                return hook

            self.handles.append(layer.register_forward_hook(make_prompt_hook(layer_index)))

        # Final norm at prompt-last.
        def final_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
            self._prompt_from_hidden("final_norm", tensor_from_output(output))
            return output

        self.handles.append(self.final_norm.register_forward_hook(final_hook))
        return self

    def finalize(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        for item in self.spec.bundle_heads:
            if item.name not in self.bundle_a or item.name not in self.bundle_b:
                raise RuntimeError(f"Missing P_POS7 capture for {item.name}")
        bundle_a = torch.cat([self.bundle_a[item.name] for item in self.spec.bundle_heads], dim=0)
        bundle_b = torch.cat([self.bundle_b[item.name] for item in self.spec.bundle_heads], dim=0)
        self.vectors["P_POS7"] = bundle_a - bundle_b
        self.means["P_POS7"] = 0.5 * (bundle_a + bundle_b)
        self.events["P_POS7"] += 1

        missing = [name for name in self.spec.component_order if name not in self.vectors]
        if missing:
            raise RuntimeError(f"Missing factor capture components {missing}")
        vectors = {
            key: value.detach().float().cpu().numpy().astype(np.float32)
            for key, value in self.vectors.items()
        }
        means = {
            key: value.detach().float().cpu().numpy().astype(np.float32)
            for key, value in self.means.items()
        }
        return vectors, means

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@torch.inference_mode()
def run_scores(
    *,
    model: Any,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
) -> Tuple[Dict[str, float], str, float, Any]:
    outputs = model(**dict(batch), use_cache=False, return_dict=True)
    scores = relation_score_map(base.relation_scores(outputs.logits[0, -1], relation_token_map, gt=None))
    prediction, top_margin = score_prediction(scores)
    return scores, prediction, top_margin, outputs


# -----------------------------------------------------------------------------
# Factor decomposition and cross-fitted direction tests
# -----------------------------------------------------------------------------


def factorial_terms(cell_vectors: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    z00 = np.asarray(cell_vectors["00"], dtype=np.float64)
    z01 = np.asarray(cell_vectors["01"], dtype=np.float64)
    z10 = np.asarray(cell_vectors["10"], dtype=np.float64)
    z11 = np.asarray(cell_vectors["11"], dtype=np.float64)
    return {
        "G": 0.25 * (z00 + z01 + z10 + z11),
        "V": 0.25 * (-z00 - z01 + z10 + z11),
        "Q": 0.25 * (-z00 + z01 - z10 + z11),
        "I": 0.25 * (z00 - z01 - z10 + z11),
    }


def factor_metrics(terms: Mapping[str, np.ndarray]) -> Dict[str, float]:
    norms = {key: float(np.linalg.norm(np.asarray(terms[key], dtype=np.float64))) for key in ("V", "Q", "I")}
    energies = {key: value * value for key, value in norms.items()}
    total = float(sum(energies.values()))
    output: Dict[str, float] = {}
    for key in ("V", "Q", "I"):
        output[f"{key}_norm"] = norms[key]
        output[f"{key}_share"] = safe_divide(energies[key], total, 0.0)
    output["factor_total_energy"] = total
    output["interaction_to_main_ratio"] = safe_divide(norms["I"], norms["V"] + norms["Q"], 0.0)
    return output


def load_extracted_vectors(
    *,
    sids: Sequence[int],
    vector_dir: Path,
    components: Sequence[str],
    pair_components: set[str],
) -> Tuple[
    Dict[int, Dict[str, Dict[str, np.ndarray]]],
    Dict[int, Dict[str, Dict[str, np.ndarray]]],
]:
    vectors: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    means: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for sid in sids:
        path = vector_dir / f"sid_{sid:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        vectors[sid] = {cell: {} for cell in CELL_NAMES}
        means[sid] = {cell: {} for cell in CELL_NAMES}
        with np.load(path, allow_pickle=False) as data:
            for cell in CELL_NAMES:
                for component in components:
                    key = f"{cell}__{component}"
                    if key not in data:
                        raise RuntimeError(f"SID {sid}: missing {key}")
                    vectors[sid][cell][component] = np.asarray(data[key], dtype=np.float64)
                    if component in pair_components:
                        mean_key = f"{cell}__{component}__mean"
                        if mean_key not in data:
                            raise RuntimeError(f"SID {sid}: missing {mean_key}")
                        means[sid][cell][component] = np.asarray(data[mean_key], dtype=np.float64)
    return vectors, means


def crossfit_factor_accuracy(
    *,
    factor_by_sid: Mapping[int, np.ndarray],
    gt_by_sid: Mapping[int, str],
    fold_by_sid: Mapping[int, int],
    folds: int,
) -> Tuple[float, Dict[int, float], Dict[int, str]]:
    scores: Dict[int, float] = {}
    predictions: Dict[int, str] = {}
    for fold in range(int(folds)):
        train_sids = [sid for sid in factor_by_sid if fold_by_sid[sid] != fold]
        test_sids = [sid for sid in factor_by_sid if fold_by_sid[sid] == fold]
        oriented = []
        for sid in train_sids:
            sign = 1.0 if gt_by_sid[sid] == "left" else -1.0
            oriented.append(sign * np.asarray(factor_by_sid[sid], dtype=np.float64))
        if not oriented:
            raise RuntimeError(f"Fold {fold} lacks training vectors")
        direction = unit_vector(np.mean(np.stack(oriented, axis=0), axis=0))
        for sid in test_sids:
            value = float(np.dot(np.asarray(factor_by_sid[sid], dtype=np.float64), direction))
            scores[sid] = value
            predictions[sid] = "left" if value >= 0 else "right"
    accuracy = safe_mean(predictions[sid] == gt_by_sid[sid] for sid in factor_by_sid)
    return accuracy, scores, predictions


def analyze_factorization(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    cell_meta_by_sid: Mapping[int, Mapping[str, Any]],
    vectors_by_sid: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    components: Sequence[str],
    probe_folds: int,
    bootstrap_repeats: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
    source_by_sid = {int(row["sid"]): dict(row) for row in selected_rows}
    sids = sorted(source_by_sid)
    gt_by_sid = {sid: str(source_by_sid[sid]["gt"]) for sid in sids}
    fold_by_sid = {
        sid: stable_fold(group_key(source_by_sid[sid], sid), int(probe_folds), int(seed))
        for sid in sids
    }

    terms_by_sid: Dict[int, Dict[str, Dict[str, np.ndarray]]] = defaultdict(dict)
    sample_rows: List[Dict[str, Any]] = []
    for sid in sids:
        for component in components:
            cells = {cell: vectors_by_sid[sid][cell][component] for cell in CELL_NAMES}
            terms = factorial_terms(cells)
            terms_by_sid[sid][component] = terms
            metrics = factor_metrics(terms)
            meta = cell_meta_by_sid[sid]
            row = {
                "sid": sid,
                "gt": gt_by_sid[sid],
                "baseline_generation_prediction": meta.get("baseline_generation_prediction"),
                "baseline_generation_correct": meta.get("baseline_generation_correct"),
                "component": component,
                "fold": fold_by_sid[sid],
                **metrics,
            }
            sample_rows.append(row)

    component_rows: List[Dict[str, Any]] = []
    component_index = {name: index for index, name in enumerate(components)}
    per_component_predictions: Dict[Tuple[str, str], Dict[int, str]] = {}
    for component in components:
        factor_maps = {
            factor: {sid: terms_by_sid[sid][component][factor] for sid in sids}
            for factor in ("V", "Q", "I")
        }
        cv_results: Dict[str, float] = {}
        for factor in ("V", "Q", "I"):
            acc, _scores, preds = crossfit_factor_accuracy(
                factor_by_sid=factor_maps[factor],
                gt_by_sid=gt_by_sid,
                fold_by_sid=fold_by_sid,
                folds=probe_folds,
            )
            cv_results[factor] = acc
            per_component_predictions[(component, factor)] = preds

        rows = [row for row in sample_rows if row["component"] == component]
        i_shares = [float(row["I_share"]) for row in rows]
        i_norms = [float(row["I_norm"]) for row in rows]
        ci_low, ci_high = bootstrap_mean_ci(i_shares, bootstrap_repeats, seed + component_index[component])
        component_rows.append(
            {
                "component": component,
                "order": component_index[component],
                "N": len(rows),
                "V_norm_mean": safe_mean(float(row["V_norm"]) for row in rows),
                "Q_norm_mean": safe_mean(float(row["Q_norm"]) for row in rows),
                "I_norm_mean": safe_mean(i_norms),
                "V_share_mean": safe_mean(float(row["V_share"]) for row in rows),
                "Q_share_mean": safe_mean(float(row["Q_share"]) for row in rows),
                "I_share_mean": safe_mean(i_shares),
                "I_share_median": safe_median(i_shares),
                "I_share_95CI_low": ci_low,
                "I_share_95CI_high": ci_high,
                "interaction_to_main_ratio_mean": safe_mean(
                    float(row["interaction_to_main_ratio"]) for row in rows
                ),
                "V_direction_cv_accuracy": cv_results["V"],
                "Q_direction_cv_accuracy": cv_results["Q"],
                "I_direction_cv_accuracy": cv_results["I"],
            }
        )

    # Attach cross-fitted factor predictions to sample rows.
    for row in sample_rows:
        sid = int(row["sid"])
        component = str(row["component"])
        for factor in ("V", "Q", "I"):
            prediction = per_component_predictions[(component, factor)][sid]
            row[f"{factor}_direction_prediction"] = prediction
            row[f"{factor}_direction_correct"] = prediction == row["gt"]

    equivariance_rows: List[Dict[str, Any]] = []
    for sid in sids:
        meta = cell_meta_by_sid[sid]
        cell_rows = meta["cells"]
        correct_flags = [bool(cell_rows[cell]["closed_correct"]) for cell in CELL_NAMES]
        predictions = [str(cell_rows[cell]["closed_prediction"]) for cell in CELL_NAMES]
        equivariance_rows.append(
            {
                "sid": sid,
                "gt": gt_by_sid[sid],
                "all_four_closed_correct": all(correct_flags),
                "closed_equivariance_accuracy": safe_mean(correct_flags),
                "prediction_pattern": "/".join(predictions),
                "expected_pattern": "/".join(str(cell_rows[cell]["expected_relation"]) for cell in CELL_NAMES),
                **{
                    f"{cell}_closed_prediction": cell_rows[cell]["closed_prediction"]
                    for cell in CELL_NAMES
                },
                **{
                    f"{cell}_closed_correct": cell_rows[cell]["closed_correct"]
                    for cell in CELL_NAMES
                },
            }
        )

    component_rows.sort(key=lambda row: int(row["order"]))
    return sample_rows, component_rows, equivariance_rows, terms_by_sid


# -----------------------------------------------------------------------------
# Causal patch classes
# -----------------------------------------------------------------------------


class FullVectorPatch:
    def __init__(self, *, module: torch.nn.Module, position: int, target: np.ndarray) -> None:
        self.position = int(position)
        self.target = torch.as_tensor(target, dtype=torch.float32).cpu()
        self.applied = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        tensor = tensor_from_output(output)
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError("FullVectorPatch expects [1,S,H]")
        if int(tensor.shape[1]) <= self.position:
            return output
        if int(tensor.shape[-1]) != int(self.target.numel()):
            raise RuntimeError("FullVectorPatch width mismatch")
        modified = tensor.clone()
        modified[0, self.position] = self.target.to(device=tensor.device, dtype=tensor.dtype)
        self.applied += 1
        return replace_tensor_output(output, modified)

    def validate(self) -> None:
        if self.applied != 1:
            raise RuntimeError(f"FullVectorPatch expected one prefill event, got {self.applied}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


class PairVectorPatch:
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        a_position: int,
        b_position: int,
        a_target: np.ndarray,
        b_target: np.ndarray,
    ) -> None:
        self.a_position = int(a_position)
        self.b_position = int(b_position)
        self.a_target = torch.as_tensor(a_target, dtype=torch.float32).cpu()
        self.b_target = torch.as_tensor(b_target, dtype=torch.float32).cpu()
        self.applied = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        tensor = tensor_from_output(output)
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError("PairVectorPatch expects [1,S,H]")
        if int(tensor.shape[1]) <= max(self.a_position, self.b_position):
            return output
        modified = tensor.clone()
        modified[0, self.a_position] = self.a_target.to(device=tensor.device, dtype=tensor.dtype)
        modified[0, self.b_position] = self.b_target.to(device=tensor.device, dtype=tensor.dtype)
        self.applied += 1
        return replace_tensor_output(output, modified)

    def validate(self) -> None:
        if self.applied != 1:
            raise RuntimeError(f"PairVectorPatch expected one prefill event, got {self.applied}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


class VHeadPairPatch:
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        a_position: int,
        b_position: int,
        unit_head: int,
        head_dim: int,
        a_target: np.ndarray,
        b_target: np.ndarray,
    ) -> None:
        self.a_position = int(a_position)
        self.b_position = int(b_position)
        self.unit_head = int(unit_head)
        self.head_dim = int(head_dim)
        self.a_target = torch.as_tensor(a_target, dtype=torch.float32).cpu()
        self.b_target = torch.as_tensor(b_target, dtype=torch.float32).cpu()
        self.applied = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not torch.is_tensor(output) or output.ndim != 3 or int(output.shape[0]) != 1:
            raise RuntimeError("VHeadPairPatch expects [1,S,D]")
        if int(output.shape[1]) <= max(self.a_position, self.b_position):
            return output
        start = self.unit_head * self.head_dim
        stop = start + self.head_dim
        modified = output.clone()
        modified[0, self.a_position, start:stop] = self.a_target.to(output.device, output.dtype)
        modified[0, self.b_position, start:stop] = self.b_target.to(output.device, output.dtype)
        self.applied += 1
        return modified

    def validate(self) -> None:
        if self.applied != 1:
            raise RuntimeError(f"VHeadPairPatch expected one prefill event, got {self.applied}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


class BundlePairPatch:
    def __init__(
        self,
        *,
        spec: CaptureSpec,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver: Any,
        ioi: Any,
        a_position: int,
        b_position: int,
        a_target: np.ndarray,
        b_target: np.ndarray,
    ) -> None:
        self.spec = spec
        self.a_position = int(a_position)
        self.b_position = int(b_position)
        self.a_target = np.asarray(a_target, dtype=np.float64)
        self.b_target = np.asarray(b_target, dtype=np.float64)
        self.applied: Counter[str] = Counter()
        self.handles: List[Any] = []
        grouped: Dict[int, List[BundleHeadSpec]] = defaultdict(list)
        for item in spec.bundle_heads:
            grouped[item.layer].append(item)
        for layer_index, items in grouped.items():
            attention = attention_helper.resolve_self_attention(decoder_layers[layer_index])
            module = ioi.output_projection_module(attention)
            shape = receiver.resolve_attention_shape(attention)

            def make_pre_hook(index: int, local_items: List[BundleHeadSpec], head_dim: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError("BundlePairPatch missing pre-WO tensor")
                    tensor = inputs[0]
                    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                        raise RuntimeError("BundlePairPatch expects [1,S,D]")
                    if int(tensor.shape[1]) <= max(self.a_position, self.b_position):
                        return inputs
                    modified = tensor.clone()
                    for item in local_items:
                        hs = int(item.head) * int(head_dim)
                        he = hs + int(head_dim)
                        a_chunk = self.a_target[item.offset_start:item.offset_stop]
                        b_chunk = self.b_target[item.offset_start:item.offset_stop]
                        modified[0, self.a_position, hs:he] = torch.as_tensor(
                            a_chunk, device=tensor.device, dtype=tensor.dtype
                        )
                        modified[0, self.b_position, hs:he] = torch.as_tensor(
                            b_chunk, device=tensor.device, dtype=tensor.dtype
                        )
                        self.applied[item.name] += 1
                    return (modified, *inputs[1:])
                return hook

            self.handles.append(
                module.register_forward_pre_hook(make_pre_hook(layer_index, items, int(shape.query_head_dim)))
            )

    def validate(self) -> None:
        missing = [item.name for item in self.spec.bundle_heads if self.applied[item.name] != 1]
        if missing:
            raise RuntimeError(f"BundlePairPatch missing/duplicate events {missing}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


class VDiffCapture:
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        a_position: int,
        b_position: int,
        unit_head: int,
        head_dim: int,
    ) -> None:
        self.a_position = int(a_position)
        self.b_position = int(b_position)
        self.unit_head = int(unit_head)
        self.head_dim = int(head_dim)
        self.value: Optional[np.ndarray] = None
        self.events = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not torch.is_tensor(output) or output.ndim != 3 or int(output.shape[0]) != 1:
            raise RuntimeError("VDiffCapture expects [1,S,D]")
        if int(output.shape[1]) <= max(self.a_position, self.b_position):
            return output
        start = self.unit_head * self.head_dim
        stop = start + self.head_dim
        diff = output[0, self.a_position, start:stop] - output[0, self.b_position, start:stop]
        self.value = diff.detach().float().cpu().numpy().astype(np.float64)
        self.events += 1
        return output

    def validate(self) -> None:
        if self.events != 1 or self.value is None:
            raise RuntimeError(f"VDiffCapture expected one event, got {self.events}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


# -----------------------------------------------------------------------------
# Causal targets and execution
# -----------------------------------------------------------------------------


def target_vector_for_intervention(
    *,
    intervention: str,
    cell_name: str,
    cell_vectors: Mapping[str, np.ndarray],
    terms: Mapping[str, np.ndarray],
    random_interaction: Optional[np.ndarray],
    strength: float,
) -> np.ndarray:
    current = np.asarray(cell_vectors[cell_name], dtype=np.float64)
    visual_flip, query_swap = CELL_BITS[cell_name]
    x_v = -1 if visual_flip == 0 else 1
    x_q = -1 if query_swap == 0 else 1
    x_i = x_v * x_q
    if intervention == "interaction_flip":
        return current - float(strength) * 2.0 * x_i * np.asarray(terms["I"], dtype=np.float64)
    if intervention == "interaction_remove":
        return current - float(strength) * x_i * np.asarray(terms["I"], dtype=np.float64)
    if intervention == "visual_flip":
        return current - float(strength) * 2.0 * x_v * np.asarray(terms["V"], dtype=np.float64)
    if intervention == "query_flip":
        return current - float(strength) * 2.0 * x_q * np.asarray(terms["Q"], dtype=np.float64)
    if intervention == "random_matched":
        if random_interaction is None:
            raise RuntimeError("random_matched requires donor interaction")
        donor = unit_vector(np.asarray(random_interaction, dtype=np.float64))
        donor = donor * float(np.linalg.norm(np.asarray(terms["I"], dtype=np.float64)))
        return current - float(strength) * 2.0 * x_i * donor
    if intervention == "whole_query":
        donor_cell = f"{visual_flip}{1 - query_swap}"
        donor = np.asarray(cell_vectors[donor_cell], dtype=np.float64)
        return current + float(strength) * (donor - current)
    raise ValueError(intervention)


def pair_targets(
    *,
    target_diff: np.ndarray,
    current_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(current_mean, dtype=np.float64)
    diff = np.asarray(target_diff, dtype=np.float64)
    return mean + 0.5 * diff, mean - 0.5 * diff


def build_patch(
    *,
    component: str,
    cell: CellData,
    target_vector: np.ndarray,
    target_mean: Optional[np.ndarray],
    spec: CaptureSpec,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver: Any,
    ioi: Any,
    final_norm: torch.nn.Module,
) -> Any:
    if component == "P_POS7":
        if target_mean is None:
            raise RuntimeError("P_POS7 patch requires mean")
        a_target, b_target = pair_targets(target_diff=target_vector, current_mean=target_mean)
        return BundlePairPatch(
            spec=spec,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver=receiver,
            ioi=ioi,
            a_position=int(cell.a_positions[-1]),
            b_position=int(cell.b_positions[-1]),
            a_target=a_target,
            b_target=b_target,
        )
    if component == "L26V":
        if target_mean is None:
            raise RuntimeError("L26V patch requires mean")
        a_target, b_target = pair_targets(target_diff=target_vector, current_mean=target_mean)
        attention = attention_helper.resolve_self_attention(decoder_layers[spec.receiver_layer])
        module = receiver.projection_module(attention, "v")
        return VHeadPairPatch(
            module=module,
            a_position=int(cell.a_positions[-1]),
            b_position=int(cell.b_positions[-1]),
            unit_head=spec.receiver_unit_head,
            head_dim=spec.receiver_head_dim,
            a_target=a_target,
            b_target=b_target,
        )
    match_object = re.fullmatch(r"L(\d+)_object", component)
    if match_object:
        if target_mean is None:
            raise RuntimeError(f"{component} patch requires mean")
        layer = int(match_object.group(1))
        a_target, b_target = pair_targets(target_diff=target_vector, current_mean=target_mean)
        return PairVectorPatch(
            module=decoder_layers[layer],
            a_position=int(cell.a_positions[-1]),
            b_position=int(cell.b_positions[-1]),
            a_target=a_target,
            b_target=b_target,
        )
    if component == "L26_attn":
        module = attention_helper.resolve_self_attention(decoder_layers[spec.receiver_layer])
        return FullVectorPatch(module=module, position=cell.prompt_last, target=target_vector)
    match_block = re.fullmatch(r"L(\d+)_block", component)
    if match_block:
        return FullVectorPatch(
            module=decoder_layers[int(match_block.group(1))],
            position=cell.prompt_last,
            target=target_vector,
        )
    if component == "final_norm":
        return FullVectorPatch(module=final_norm, position=cell.prompt_last, target=target_vector)
    raise ValueError(f"Unsupported causal component {component}")


def desired_receiver_target(
    *,
    intervention: str,
    cell_name: str,
    receiver_cells: Mapping[str, np.ndarray],
    receiver_terms: Mapping[str, np.ndarray],
    random_receiver_interaction: Optional[np.ndarray],
    strength: float,
) -> np.ndarray:
    return target_vector_for_intervention(
        intervention=intervention,
        cell_name=cell_name,
        cell_vectors=receiver_cells,
        terms=receiver_terms,
        random_interaction=random_receiver_interaction,
        strength=strength,
    )


def run_causal_analysis(
    *,
    args: argparse.Namespace,
    selected_rows: Sequence[Mapping[str, Any]],
    cell_meta_by_sid: Mapping[int, Mapping[str, Any]],
    vectors_by_sid: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    means_by_sid: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    terms_by_sid: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    spec: CaptureSpec,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    final_norm: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    base: Any,
    v3: Any,
    receiver: Any,
    attention_helper: Any,
    ioi: Any,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    allowed_interventions = {
        "interaction_flip",
        "interaction_remove",
        "visual_flip",
        "query_flip",
        "random_matched",
        "whole_query",
    }
    interventions = parse_csv_tokens(args.causal_interventions)
    unknown = [name for name in interventions if name not in allowed_interventions]
    if unknown:
        raise ValueError(f"Unknown interventions {unknown}")
    components = parse_csv_tokens(args.causal_components)
    available_components = set(spec.component_order)
    missing_components = [name for name in components if name not in available_components]
    if missing_components:
        raise ValueError(f"Causal components not captured: {missing_components}")
    target_cells = parse_csv_tokens(args.causal_target_cells)
    if any(cell not in CELL_NAMES for cell in target_cells):
        raise ValueError(f"Invalid target cells {target_cells}")

    rows_to_run = [dict(row) for row in selected_rows]
    rows_to_run = stratified_limit(rows_to_run, int(args.causal_max_samples), int(args.seed) + 1000)
    sids = [int(row["sid"]) for row in rows_to_run]
    if len(sids) < 2 and "random_matched" in interventions:
        raise RuntimeError("random_matched control requires at least two samples")
    donor_by_sid = {sid: sids[(index + 1) % len(sids)] for index, sid in enumerate(sids)}

    output_path = output_dir / "causal_interaction_effect.jsonl"
    errors_path = output_dir / "errors.jsonl"
    existing = deduplicate_rows(read_jsonl(output_path), ("sid", "target_cell", "component", "intervention")) if args.resume else []
    completed = {
        (int(row["sid"]), str(row["target_cell"]), str(row["component"]), str(row["intervention"]))
        for row in existing
    }
    all_rows = list(existing)

    receiver_attention = attention_helper.resolve_self_attention(decoder_layers[spec.receiver_layer])
    receiver_v_module = receiver.projection_module(receiver_attention, "v")

    expected_total = len(rows_to_run) * len(target_cells) * len(components) * len(interventions)
    print(
        f"Causal factorial scan: samples={len(rows_to_run)} cells={target_cells} "
        f"components={components} interventions={interventions} expected={expected_total}",
        flush=True,
    )

    for sample_index, source_row in enumerate(
        tqdm(rows_to_run, desc=f"factorial-causal:{args.model}"), start=1
    ):
        sample = None
        try:
            sid = int(source_row["sid"])
            sample = prepare_four_cells(
                args=args,
                source_row=source_row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                receiver=receiver,
                processor=processor,
                device=torch.device(args.device),
            )
            donor_sid = donor_by_sid[sid]
            for target_cell_name in target_cells:
                cell = sample.cells[target_cell_name]
                cell_meta = cell_meta_by_sid[sid]["cells"]
                baseline_scores = relation_score_map(cell_meta[target_cell_name]["closed_scores"])
                expected = str(cell.expected_relation)
                baseline_margin = horizontal_margin(baseline_scores, expected)
                v, q = CELL_BITS[target_cell_name]
                q_donor_cell = f"{v}{1-q}"
                v_donor_cell = f"{1-v}{q}"
                q_donor_margin = horizontal_margin(
                    relation_score_map(cell_meta[q_donor_cell]["closed_scores"]), expected
                )
                v_donor_margin = horizontal_margin(
                    relation_score_map(cell_meta[v_donor_cell]["closed_scores"]), expected
                )
                denominator = 0.5 * (
                    abs(baseline_margin - q_donor_margin)
                    + abs(baseline_margin - v_donor_margin)
                )
                if denominator < float(args.min_margin_denominator):
                    continue

                for component in components:
                    current_cells = {
                        name: vectors_by_sid[sid][name][component] for name in CELL_NAMES
                    }
                    terms = terms_by_sid[sid][component]
                    current_mean = means_by_sid[sid][target_cell_name].get(component)
                    receiver_cells = {
                        name: vectors_by_sid[sid][name]["L26V"] for name in CELL_NAMES
                    }
                    receiver_terms = terms_by_sid[sid]["L26V"]
                    for intervention in interventions:
                        key = (sid, target_cell_name, component, intervention)
                        if key in completed:
                            continue
                        random_i = None
                        random_receiver_i = None
                        if intervention == "random_matched":
                            random_i = deterministic_orthogonal_control(
                                reference=np.asarray(terms["I"], dtype=np.float64),
                                nuisance_vectors=[terms["V"], terms["Q"], terms["I"]],
                                key=f"{sid}:{target_cell_name}:{component}",
                                seed=int(args.seed),
                            )
                            receiver_terms_local = terms_by_sid[sid]["L26V"]
                            random_receiver_i = deterministic_orthogonal_control(
                                reference=np.asarray(receiver_terms_local["I"], dtype=np.float64),
                                nuisance_vectors=[
                                    receiver_terms_local["V"],
                                    receiver_terms_local["Q"],
                                    receiver_terms_local["I"],
                                ],
                                key=f"{sid}:{target_cell_name}:L26V",
                                seed=int(args.seed) + 1,
                            )
                        target_vector = target_vector_for_intervention(
                            intervention=intervention,
                            cell_name=target_cell_name,
                            cell_vectors=current_cells,
                            terms=terms,
                            random_interaction=random_i,
                            strength=float(args.patch_strength),
                        )
                        target_mean = current_mean
                        if intervention == "whole_query" and component in spec.pair_components:
                            target_mean = means_by_sid[sid][q_donor_cell][component]

                        patch = build_patch(
                            component=component,
                            cell=cell,
                            target_vector=target_vector,
                            target_mean=target_mean,
                            spec=spec,
                            decoder_layers=decoder_layers,
                            attention_helper=attention_helper,
                            receiver=receiver,
                            ioi=ioi,
                            final_norm=final_norm,
                        )
                        v_capture = None
                        if component == "P_POS7":
                            v_capture = VDiffCapture(
                                module=receiver_v_module,
                                a_position=int(cell.a_positions[-1]),
                                b_position=int(cell.b_positions[-1]),
                                unit_head=spec.receiver_unit_head,
                                head_dim=spec.receiver_head_dim,
                            )
                        try:
                            patched_scores, patched_prediction, _top, _outputs = run_scores(
                                model=model,
                                batch=cell.batch,
                                relation_token_map=relation_token_map,
                                base=base,
                            )
                            patch.validate()
                            if v_capture is not None:
                                v_capture.validate()
                        finally:
                            patch.close()
                            if v_capture is not None:
                                v_capture.close()

                        patched_margin = horizontal_margin(patched_scores, expected)
                        desired_effect = baseline_margin - patched_margin
                        normalized_effect = safe_divide(desired_effect, denominator, float("nan"))
                        receiver_recovery = float("nan")
                        if component == "P_POS7" and v_capture is not None and v_capture.value is not None:
                            desired_receiver = desired_receiver_target(
                                intervention=intervention,
                                cell_name=target_cell_name,
                                receiver_cells=receiver_cells,
                                receiver_terms=receiver_terms,
                                random_receiver_interaction=random_receiver_i,
                                strength=float(args.patch_strength),
                            )
                            base_receiver = receiver_cells[target_cell_name]
                            desired_delta = np.asarray(desired_receiver) - np.asarray(base_receiver)
                            actual_delta = np.asarray(v_capture.value) - np.asarray(base_receiver)
                            receiver_recovery = safe_divide(
                                float(np.dot(actual_delta, desired_delta)),
                                float(np.dot(desired_delta, desired_delta)),
                                float("nan"),
                            )

                        row = {
                            "script_version": SCRIPT_VERSION,
                            "model": args.model,
                            "sid": sid,
                            "gt": sample.gt,
                            "baseline_generation_prediction": cell_meta_by_sid[sid].get("baseline_generation_prediction"),
                            "baseline_generation_correct": cell_meta_by_sid[sid].get("baseline_generation_correct"),
                            "target_cell": target_cell_name,
                            "visual_flip": int(cell.visual_flip),
                            "query_swap": int(cell.query_swap),
                            "expected_relation": expected,
                            "component": component,
                            "intervention": intervention,
                            "patch_strength": float(args.patch_strength),
                            "random_donor_sid": None,
                            "random_control": (
                                "deterministic_same_norm_orthogonal_to_VQI"
                                if intervention == "random_matched"
                                else None
                            ),
                            "baseline_closed_prediction": cell_meta[target_cell_name]["closed_prediction"],
                            "patched_closed_prediction": patched_prediction,
                            "baseline_closed_correct": bool(cell_meta[target_cell_name]["closed_correct"]),
                            "patched_closed_correct": bool(patched_prediction == expected),
                            "baseline_expected_vs_opposite_margin": baseline_margin,
                            "patched_expected_vs_opposite_margin": patched_margin,
                            "desired_flip_effect": desired_effect,
                            "total_effect_denominator": denominator,
                            "normalized_effect": normalized_effect,
                            "positive_effect": bool(desired_effect > 0),
                            "crossed_to_opposite": bool(baseline_margin > 0 >= patched_margin),
                            "patched_predicts_opposite": bool(patched_prediction == OPPOSITE[expected]),
                            "patch_delta_norm": float(
                                np.linalg.norm(
                                    np.asarray(target_vector, dtype=np.float64)
                                    - np.asarray(current_cells[target_cell_name], dtype=np.float64)
                                )
                            ),
                            "natural_interaction_norm": float(np.linalg.norm(np.asarray(terms["I"]))),
                            "p_pos7_to_l26v_desired_recovery": receiver_recovery,
                        }
                        append_jsonl(output_path, row)
                        all_rows.append(row)
                        completed.add(key)
        except Exception as exc:
            error = {
                "phase": "causal",
                "sid": int(source_row.get("sid", -1)),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error)
            print(
                f"[ERROR causal sid={source_row.get('sid')}] {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            release_four_cells(sample, receiver)
            gc.collect()
            if (
                args.device.startswith("cuda")
                and args.empty_cache_every > 0
                and sample_index % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    all_rows = deduplicate_rows(read_jsonl(output_path), ("sid", "target_cell", "component", "intervention"))
    summary_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in all_rows:
        summary_groups[(str(row["component"]), str(row["intervention"]))].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for (component, intervention), values in sorted(summary_groups.items()):
        normalized = [float(row["normalized_effect"]) for row in values]
        ci_low, ci_high = bootstrap_mean_ci(normalized, int(args.bootstrap_repeats), int(args.seed) + 3000)
        summary_rows.append(
            {
                "component": component,
                "intervention": intervention,
                "N": len(values),
                "desired_flip_effect_mean": safe_mean(float(row["desired_flip_effect"]) for row in values),
                "normalized_effect_mean": safe_mean(normalized),
                "normalized_effect_median": safe_median(normalized),
                "normalized_effect_95CI_low": ci_low,
                "normalized_effect_95CI_high": ci_high,
                "positive_effect_rate": safe_mean(bool_value(row["positive_effect"]) for row in values),
                "crossed_to_opposite_rate": safe_mean(bool_value(row["crossed_to_opposite"]) for row in values),
                "opposite_prediction_rate": safe_mean(bool_value(row["patched_predicts_opposite"]) for row in values),
                "patched_closed_accuracy": safe_mean(bool_value(row["patched_closed_correct"]) for row in values),
                "patch_delta_norm_mean": safe_mean(float(row["patch_delta_norm"]) for row in values),
                "p_pos7_to_l26v_desired_recovery_mean": safe_mean(
                    float(row["p_pos7_to_l26v_desired_recovery"]) for row in values
                ),
            }
        )
    write_csv(output_dir / "causal_interaction_summary.csv", summary_rows)
    return all_rows, summary_rows


# -----------------------------------------------------------------------------
# Go / no-go assessment
# -----------------------------------------------------------------------------


def build_go_no_go(
    *,
    args: argparse.Namespace,
    component_summary: Sequence[Mapping[str, Any]],
    causal_summary: Sequence[Mapping[str, Any]],
    equivariance_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ordered_all = sorted(component_summary, key=lambda row: int(row["order"]))
    ordered = [row for row in ordered_all if str(row["component"]) != "relation_logits"]
    if not ordered:
        raise RuntimeError("No internal components available for go/no-go assessment")
    interaction_acc = [float(row["I_direction_cv_accuracy"]) for row in ordered]
    interaction_share = [float(row["I_share_mean"]) for row in ordered]
    peak_acc_row = max(ordered, key=lambda row: float(row["I_direction_cv_accuracy"]))
    peak_share_row = max(ordered, key=lambda row: float(row["I_share_mean"]))
    localization_range = max(interaction_acc) - min(interaction_acc) if interaction_acc else float("nan")

    causal_by_key = {
        (str(row["component"]), str(row["intervention"])): row
        for row in causal_summary
    }
    causal_candidates: List[Dict[str, Any]] = []
    for component in {str(row["component"]) for row in causal_summary}:
        interaction = causal_by_key.get((component, "interaction_flip"))
        random_control = causal_by_key.get((component, "random_matched"))
        if interaction is None:
            continue
        effect = float(interaction["normalized_effect_mean"])
        positive = float(interaction["positive_effect_rate"])
        random_effect = (
            float(random_control["normalized_effect_mean"])
            if random_control is not None
            else float("nan")
        )
        specificity = effect - random_effect if math.isfinite(random_effect) else float("nan")
        causal_candidates.append(
            {
                "component": component,
                "interaction_effect": effect,
                "positive_rate": positive,
                "random_effect": random_effect,
                "specificity": specificity,
            }
        )
    best_causal = max(
        causal_candidates,
        key=lambda row: (
            float(row["specificity"]) if math.isfinite(float(row["specificity"])) else -1e9,
            float(row["interaction_effect"]),
        ),
        default={
            "component": "",
            "interaction_effect": float("nan"),
            "positive_rate": float("nan"),
            "random_effect": float("nan"),
            "specificity": float("nan"),
        },
    )

    p_pos7_recovery = float("nan")
    p_row = causal_by_key.get(("P_POS7", "interaction_flip"))
    if p_row is not None:
        p_pos7_recovery = float(p_row.get("p_pos7_to_l26v_desired_recovery_mean", float("nan")))

    criteria = {
        "interaction_decodable": bool(
            float(peak_acc_row["I_direction_cv_accuracy"]) >= float(args.go_min_interaction_accuracy)
        ),
        "interaction_localized": bool(
            math.isfinite(localization_range)
            and localization_range >= float(args.go_min_localization_range)
        ),
        "causal_interaction_specific": bool(
            math.isfinite(float(best_causal["interaction_effect"]))
            and float(best_causal["interaction_effect"]) >= float(args.go_min_causal_effect)
            and float(best_causal["positive_rate"]) >= float(args.go_min_positive_rate)
            and math.isfinite(float(best_causal["specificity"]))
            and float(best_causal["specificity"]) >= float(args.go_min_causal_specificity)
        ),
        "p_pos7_interaction_reaches_l26v": bool(
            math.isfinite(p_pos7_recovery) and p_pos7_recovery >= 0.15
        ),
    }
    passed = sum(criteria.values())
    decision = "GO" if passed >= 2 and criteria["causal_interaction_specific"] else "NO_GO_OR_REDESIGN"
    return {
        "script_version": SCRIPT_VERSION,
        "decision": decision,
        "criteria_passed": passed,
        "criteria": criteria,
        "peak_interaction_accuracy_component": str(peak_acc_row["component"]),
        "peak_interaction_accuracy": float(peak_acc_row["I_direction_cv_accuracy"]),
        "peak_interaction_share_component": str(peak_share_row["component"]),
        "peak_interaction_share": float(peak_share_row["I_share_mean"]),
        "interaction_accuracy_localization_range": localization_range,
        "best_causal_component": best_causal,
        "p_pos7_to_l26v_interaction_recovery": p_pos7_recovery,
        "four_cell_closed_equivariance_accuracy_mean": safe_mean(
            float(row["closed_equivariance_accuracy"]) for row in equivariance_rows
        ),
        "all_four_closed_correct_rate": safe_mean(
            bool_value(row["all_four_closed_correct"]) for row in equivariance_rows
        ),
        "interpretation": {
            "GO": (
                "A localized interaction representation has directionally specific causal effect. "
                "Proceed to exact path isolation and a second VLM."
            ),
            "NO_GO_OR_REDESIGN": (
                "The 2x2 interaction is either diffuse, non-specific, or no stronger than matched perturbations. "
                "Do not build the paper around reasoning-vs-relay without a new hypothesis."
            ),
        },
    }


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
        raise ValueError("--patch-strength must be nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    ioi = import_file(Path(args.ioi_script), "factorial_ioi")
    producer = import_file(Path(args.producer_script), "factorial_producer")
    receiver = import_file(Path(args.receiver_script), "factorial_receiver")
    v3 = import_file(Path(args.v3_script), "factorial_v3")
    base = import_file(Path(args.base_script), "factorial_base")
    attention_helper = import_file(Path(args.attention_helper), "factorial_attention")

    source_config, source_rows = ioi.load_source_rows(args)
    baseline_by_sid = load_baseline_rows(resolve_baseline_path(args))

    included: Optional[set[int]] = None
    if str(args.include_sids_file).strip():
        included = extract_sids(Path(str(args.include_sids_file).strip()))
    excluded: set[int] = set()
    for raw in str(args.exclude_sids_from).split(","):
        item = raw.strip()
        if item:
            excluded.update(extract_sids(Path(item)))

    selected_rows = []
    for row in source_rows:
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        if gt not in HORIZONTAL:
            continue
        if sid in excluded or (included is not None and sid not in included):
            continue
        selected_rows.append({**dict(row), "sid": sid, "gt": gt})
    selected_rows = stratified_limit(selected_rows, int(args.sample_max_samples), int(args.seed))
    if not selected_rows:
        raise RuntimeError("No horizontal COCO samples selected")
    selected_sids = [int(row["sid"]) for row in selected_rows]

    model = None
    processor = None
    try:
        (
            model,
            processor,
            spec_model,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer.load_model_bundle(args=args, base=base)
        final_norm = resolve_final_norm(model, decoder_path)
        bundle_heads = load_bundle(Path(args.bundle_json), args.bundle_name)
        capture_spec, receiver_unit = build_capture_spec(
            args=args,
            decoder_layers=decoder_layers,
            bundle_heads=bundle_heads,
            attention_helper=attention_helper,
            receiver=receiver,
            ioi=ioi,
        )
        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec_model.repo_id,
            "dataset": "coco_two_horizontal_only",
            "relations": list(HORIZONTAL),
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "decoder_path": decoder_path,
            "n_layers": len(decoder_layers),
            "bundle_name": args.bundle_name,
            "bundle_heads": [head_name(layer, head) for layer, head in bundle_heads],
            "receiver_unit": {
                "layer": int(receiver_unit.layer),
                "query_head": int(receiver_unit.query_head),
                "kv_head": int(receiver_unit.kv_head),
                "unit_head": int(receiver_unit.unit_head),
                "head_dim": int(capture_spec.receiver_head_dim),
            },
            "components": capture_spec.component_order + ["relation_logits"],
            "pair_components": sorted(capture_spec.pair_components),
            "four_cells": {
                "00": "original image + original query",
                "01": "original image + role-swapped query",
                "10": "horizontal-flip image + original query",
                "11": "horizontal-flip image + role-swapped query",
            },
            "selected_samples": len(selected_rows),
            "selected_sids": selected_sids,
            "phase": args.phase,
            "min_margin_denominator": float(args.min_margin_denominator),
            "audit": audit,
            "transformers_version": transformers.__version__,
            "limits": [
                "COCO left/right only; no Controlled-A and no vertical flip.",
                "Factor norms are descriptive; composition claims require causal specificity.",
                "P_POS7 patch is bundle-output intervention, not yet an exact frozen-intermediate path patch.",
                "random_matched is a deterministic same-norm direction orthogonalized to the sample V/Q/I subspace.",
            ],
        }
        write_json(output_dir / "config.json", config)

        cells_path = output_dir / "factorial_cells.jsonl"
        cell_rows = deduplicate_rows(read_jsonl(cells_path), ("sid",)) if args.resume else []
        done = {
            int(row["sid"])
            for row in cell_rows
            if (vector_dir / f"sid_{int(row['sid']):06d}.npz").exists()
        }

        if args.phase in ("extract", "all"):
            pending = [row for row in selected_rows if int(row["sid"]) not in done]
            print(
                f"COCO 2x2 extraction: horizontal samples={len(selected_rows)} pending={len(pending)} "
                f"four-cell-forwards={len(pending)*4}",
                flush=True,
            )
            for index, source_row in enumerate(
                tqdm(pending, desc=f"factorial-extract:{args.model}"), start=1
            ):
                sample = None
                try:
                    sid = int(source_row["sid"])
                    sample = prepare_four_cells(
                        args=args,
                        source_row=source_row,
                        records_by_sid=records_by_sid,
                        prompt_rows=prompt_rows,
                        base=base,
                        v3=v3,
                        receiver=receiver,
                        processor=processor,
                        device=torch.device(args.device),
                    )
                    arrays: Dict[str, np.ndarray] = {}
                    cells_meta: Dict[str, Dict[str, Any]] = {}
                    for cell_name in CELL_NAMES:
                        cell = sample.cells[cell_name]
                        capture = FactorCapture(
                            cell=cell,
                            spec=capture_spec,
                            decoder_layers=decoder_layers,
                            attention_helper=attention_helper,
                            receiver=receiver,
                            ioi=ioi,
                            final_norm=final_norm,
                        )
                        with capture:
                            scores, prediction, top_margin, _outputs = run_scores(
                                model=model,
                                batch=cell.batch,
                                relation_token_map=relation_token_map,
                                base=base,
                            )
                        vectors, means = capture.finalize()
                        for component, vector in vectors.items():
                            arrays[f"{cell_name}__{component}"] = vector
                        for component, mean in means.items():
                            arrays[f"{cell_name}__{component}__mean"] = mean
                        arrays[f"{cell_name}__relation_logits"] = np.asarray(
                            [scores["left"], scores["right"]], dtype=np.float32
                        )
                        cells_meta[cell_name] = {
                            "visual_flip": int(cell.visual_flip),
                            "query_swap": int(cell.query_swap),
                            "expected_relation": cell.expected_relation,
                            "closed_scores": scores,
                            "closed_prediction": prediction,
                            "closed_top_margin": top_margin,
                            "closed_correct": bool(prediction == cell.expected_relation),
                            "expected_vs_opposite_margin": horizontal_margin(scores, cell.expected_relation),
                            "a_position": int(cell.a_positions[-1]),
                            "b_position": int(cell.b_positions[-1]),
                            "prompt_last": int(cell.prompt_last),
                        }
                    vector_path = vector_dir / f"sid_{sid:06d}.npz"
                    np.savez_compressed(vector_path, **arrays)
                    baseline = baseline_by_sid.get(sid, {})
                    row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": sample.gt,
                        "subject": sample.subject,
                        "reference": sample.reference,
                        "baseline_generation_prediction": normalize_relation(baseline.get("prediction")),
                        "baseline_generation_correct": bool(baseline.get("correct", False)) if baseline else None,
                        "cells": cells_meta,
                        "vector_file": str(vector_path),
                    }
                    append_jsonl(cells_path, row)
                    cell_rows.append(row)
                    if args.print_every > 0 and index % args.print_every == 0:
                        pattern = "/".join(cells_meta[cell]["closed_prediction"] for cell in CELL_NAMES)
                        expected = "/".join(cells_meta[cell]["expected_relation"] for cell in CELL_NAMES)
                        print(f"[extract {index}/{len(pending)} sid={sid}] pred={pattern} expected={expected}", flush=True)
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
                    release_four_cells(sample, receiver)
                    gc.collect()
                    if (
                        args.device.startswith("cuda")
                        and args.empty_cache_every > 0
                        and index % args.empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()

        if args.phase == "extract":
            complete = deduplicate_rows(read_jsonl(cells_path), ("sid",))
            complete_sids = {int(row["sid"]) for row in complete}
            missing = [sid for sid in selected_sids if sid not in complete_sids]
            print(
                f"Factorial extraction completed: built={len(complete_sids)}/{len(selected_sids)} "
                f"failed={len(missing)} output={output_dir}",
                flush=True,
            )
            if missing:
                raise RuntimeError(f"Incomplete extraction SIDs {missing[:20]}; inspect {errors_path}")
            return

        # Offline factor analysis.
        cell_rows = deduplicate_rows(read_jsonl(cells_path), ("sid",))
        cell_meta_by_sid = {int(row["sid"]): dict(row) for row in cell_rows}
        missing = [sid for sid in selected_sids if sid not in cell_meta_by_sid]
        if missing:
            raise RuntimeError(f"Missing factorial extraction for SIDs {missing[:20]}")
        analysis_components = capture_spec.component_order + ["relation_logits"]
        vectors_by_sid, means_by_sid = load_extracted_vectors(
            sids=selected_sids,
            vector_dir=vector_dir,
            components=analysis_components,
            pair_components=capture_spec.pair_components,
        )
        sample_factor_rows, component_summary, equivariance_rows, terms_by_sid = analyze_factorization(
            selected_rows=selected_rows,
            cell_meta_by_sid=cell_meta_by_sid,
            vectors_by_sid=vectors_by_sid,
            components=analysis_components,
            probe_folds=int(args.probe_folds),
            bootstrap_repeats=int(args.bootstrap_repeats),
            seed=int(args.seed),
        )
        write_csv(output_dir / "sample_factorization.csv", sample_factor_rows)
        write_csv(output_dir / "component_factorization_summary.csv", component_summary)
        write_csv(output_dir / "four_cell_equivariance.csv", equivariance_rows)

        causal_rows: List[Dict[str, Any]] = []
        causal_summary: List[Dict[str, Any]] = []
        if args.phase in ("causal", "all"):
            causal_rows, causal_summary = run_causal_analysis(
                args=args,
                selected_rows=selected_rows,
                cell_meta_by_sid=cell_meta_by_sid,
                vectors_by_sid=vectors_by_sid,
                means_by_sid=means_by_sid,
                terms_by_sid=terms_by_sid,
                spec=capture_spec,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                final_norm=final_norm,
                relation_token_map=relation_token_map,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                receiver=receiver,
                attention_helper=attention_helper,
                ioi=ioi,
                output_dir=output_dir,
            )
        elif (output_dir / "causal_interaction_summary.csv").exists():
            causal_summary = read_csv(output_dir / "causal_interaction_summary.csv")

        go_no_go = build_go_no_go(
            args=args,
            component_summary=component_summary,
            causal_summary=causal_summary,
            equivariance_rows=equivariance_rows,
        )
        write_json(output_dir / "go_no_go.json", go_no_go)

        print("\n" + "=" * 144)
        print("COCO REASONING-vs-RELAY FACTORIAL RESULT")
        print("=" * 144)
        print(
            f"Samples={len(selected_rows)} left/right only | four-cell closed equivariance="
            f"{go_no_go['four_cell_closed_equivariance_accuracy_mean']:.4f} | "
            f"all-four-correct={go_no_go['all_four_closed_correct_rate']:.4f}"
        )
        print("\nCOMPONENT FACTORIZATION")
        print(f"{'component':>16} {'Vshare':>9} {'Qshare':>9} {'Ishare':>9} {'Vacc':>8} {'Qacc':>8} {'Iacc':>8}")
        for row in component_summary:
            print(
                f"{str(row['component']):>16} "
                f"{float(row['V_share_mean']):9.4f} "
                f"{float(row['Q_share_mean']):9.4f} "
                f"{float(row['I_share_mean']):9.4f} "
                f"{float(row['V_direction_cv_accuracy']):8.4f} "
                f"{float(row['Q_direction_cv_accuracy']):8.4f} "
                f"{float(row['I_direction_cv_accuracy']):8.4f}"
            )
        if causal_summary:
            print("\nCAUSAL INTERACTION PATCH")
            print(f"{'component':>16} {'intervention':>20} {'effect':>10} {'positive':>10} {'crossed':>10} {'L26rec':>10}")
            for row in causal_summary:
                if str(row["intervention"]) not in {"interaction_flip", "random_matched", "whole_query"}:
                    continue
                print(
                    f"{str(row['component']):>16} {str(row['intervention']):>20} "
                    f"{float(row['normalized_effect_mean']):10.4f} "
                    f"{float(row['positive_effect_rate']):10.4f} "
                    f"{float(row['crossed_to_opposite_rate']):10.4f} "
                    f"{float(row['p_pos7_to_l26v_desired_recovery_mean']):10.4f}"
                )
        print("\nGO / NO-GO")
        print(
            f"decision={go_no_go['decision']} | passed={go_no_go['criteria_passed']}/4 | "
            f"peak-I-acc={go_no_go['peak_interaction_accuracy']:.4f} "
            f"at {go_no_go['peak_interaction_accuracy_component']} | "
            f"localization-range={go_no_go['interaction_accuracy_localization_range']:.4f}"
        )
        print(f"criteria={go_no_go['criteria']}")
        print(f"Saved outputs to {output_dir}", flush=True)

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
