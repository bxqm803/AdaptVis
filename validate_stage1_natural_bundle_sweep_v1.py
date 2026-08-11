#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-I natural head-state BUNDLE sweep.

This version intentionally DOES NOT run individual-head interventions.

It ranks Direction heads from the existing per-head probe `head_results.csv`
(default metric: raw-image `img_accuracy_mean`) and evaluates cumulative
natural-state replacement bundles:

    Top10
    Top20
    Top25
    Top30

For every K, three conditions are run:

    topK_role
        Real swapped/source subject/reference head states are copied into the
        target branch with ROLE alignment:

            source subject B   -> target subject A
            source reference A -> target reference B

    topK_identity
        Same real source activations, but object identity is preserved:

            source A -> target A
            source B -> target B

    topK_random_role
        Same ROLE replacement applied to a nested same-layer matched-random
        bundle with exactly K heads and the same per-layer head-count profile
        as the corresponding Direction TopK.

The tested bundles are cumulative/nested.  If Top30 is:
    h1, h2, ..., h30
then Top10 = h1..h10, Top20 = h1..h20, etc.

Main question
=============
If the spatial relation code is distributed/redundant across multiple
Direction heads, source-follow should increase systematically as K grows:

    Top10 < Top20 < Top25 < Top30

while identity and matched-random controls should remain much smaller.

No synthetic relation vector is constructed.  Every replacement uses real
pre-W_O head activations from the swapped branch.

Recommended full run
====================

CUDA_VISIBLE_DEVICES=0 python -u validate_stage1_natural_bundle_sweep_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --direction-results output/coco_head_object_residual_direction_probe_v1/qwen-3b/head_results.csv \
  --rank-metric img_accuracy_mean \
  --bundle-sizes 10,20,25,30 \
  --max-samples 0 \
  --pair-status all \
  --object-probe-layers 18,19,20,21,22,23,24,25,26,27,28 \
  --last-probe-layers 18,19,20,21,22,23,24,25,26,27,28,35 \
  --report-layers 19,20,21,22,23,24,25,26,27,28,35 \
  --probe-train-ratio 0.15 \
  --probe-repeats 5 \
  --probe-seed 1 \
  --replacement-mode tokenwise_resample \
  --device cuda:0 \
  --no-run-generation \
  --output-dir output/qwen3b_stage1_natural_bundle_sweep_all440 \
  --overwrite
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
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


VERSION = "stage1-natural-head-state-bundle-sweep-v1"

RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
ID_TO_REL = {i: r for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

DEFAULT_HEADS = "23:5,23:1,19:13,26:3,23:0,20:0,20:8,19:12,20:6,19:8"
DEFAULT_OBJECT_LAYERS = "18,19,20,21,23,24,25,26,27"
DEFAULT_LAST_LAYERS = "23,24,25,26,27,35"

REL_PATTERNS: Dict[str, Sequence[str]] = {
    "left": (r"\bleft\s+of\b", r"\bto\s+the\s+left\b", r"\bleft\b"),
    "right": (r"\bright\s+of\b", r"\bto\s+the\s+right\b", r"\bright\b"),
    "above": (r"\bon\s+top\s+of\b", r"\batop\b", r"\babove\b", r"\bover\b"),
    "below": (r"\bunderneath\b", r"\bbeneath\b", r"\bbelow\b", r"\bunder\b"),
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", default="mean", choices=("mean", "last"))

    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
    )

    p.add_argument(
        "--direction-results",
        default="output/coco_head_object_residual_direction_probe_v1/qwen-3b/head_results.csv",
        help="Per-head Direction probe head_results.csv. Heads are re-sorted by --rank-metric.",
    )
    p.add_argument(
        "--rank-metric",
        default="img_accuracy_mean",
        help="Metric column used to rank Direction heads; raw-image Direction uses img_accuracy_mean.",
    )
    p.add_argument(
        "--bundle-sizes",
        default="10,20,25,30",
        help="Cumulative Direction bundle sizes. Example: 10,20,25,30",
    )
    p.add_argument("--object-probe-layers", default=DEFAULT_OBJECT_LAYERS)
    p.add_argument("--last-probe-layers", default=DEFAULT_LAST_LAYERS)
    p.add_argument(
        "--report-layers",
        default="23,24,26,27,35",
        help="Layers printed in compact per-head summary when available.",
    )

    p.add_argument(
        "--pair-status",
        default="both_correct",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
        help="Use both_correct first so the source branch is a valid opposite-relation source.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="0 = all eligible source-cache samples.",
    )
    p.add_argument("--sample-seed", type=int, default=17)

    p.add_argument("--probe-train-ratio", type=float, default=0.15)
    p.add_argument("--probe-repeats", type=int, default=5)
    p.add_argument("--probe-seed", type=int, default=1)

    p.add_argument(
        "--replacement-mode",
        default="tokenwise_resample",
        choices=("tokenwise_resample", "pooled_broadcast", "mean_shift"),
        help=(
            "How real source object-token head states are placed into target object-token slots. "
            "tokenwise_resample copies exact source rows when counts match and nearest-neighbor "
            "resamples actual source rows otherwise."
        ),
    )


    p.add_argument(
        "--run-generation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--generation-max-samples",
        type=int,
        default=100,
        help="0 = all analyzed samples.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--min-new-tokens", type=int, default=1)

    p.add_argument(
        "--feature-dtype",
        default="float16",
        choices=("float16", "float32"),
        help="CPU storage dtype for captured hidden states.",
    )

    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--empty-cache-every", type=int, default=5)

    # Same repository helper scripts used by previous experiments.
    p.add_argument(
        "--three-stage-script",
        default="validate_three_stage_spatial_circuit_v1.py",
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

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


# =============================================================================
# Generic utilities
# =============================================================================

def import_file(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def parse_head(text: str) -> Tuple[int, int]:
    s = str(text).strip().upper()
    s = s.replace("L", "").replace("H", ":")
    while "::" in s:
        s = s.replace("::", ":")
    if ":" not in s:
        raise ValueError(f"Bad head {text!r}; use 23:1 or L23H1")
    a, b = s.split(":", 1)
    return int(a), int(b)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for item in str(text).split(","):
        if not item.strip():
            continue
        head = parse_head(item)
        if head not in seen:
            seen.add(head)
            out.append(head)
    if not out:
        raise ValueError("No --heads selected")
    return out



def parse_bundle_sizes(text: str) -> List[int]:
    vals: List[int] = []
    seen = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"Bundle size must be > 0, got {k}")
        if k not in seen:
            seen.add(k)
            vals.append(k)
    if not vals:
        raise ValueError("No --bundle-sizes selected")
    return sorted(vals)


def load_ranked_direction_heads(
    path: Path,
    metric: str,
    max_k: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    if not path.exists():
        # Do not silently guess among multiple experiments.
        candidates = sorted(Path("output").glob("**/head_results.csv"))
        msg = [f"Direction results not found: {path}"]
        if candidates:
            msg.append("Available head_results.csv candidates:")
            msg.extend(f"  {p}" for p in candidates[:30])
        raise FileNotFoundError("\n".join(msg))

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty Direction results: {path}")
    if metric not in rows[0]:
        raise KeyError(
            f"Metric {metric!r} not in {path}. Columns: {list(rows[0].keys())}"
        )

    parsed: List[Dict[str, Any]] = []
    for row in rows:
        try:
            layer = int(row["layer"])
            head = int(row["head"])
            score = float(row[metric])
        except Exception:
            continue
        if not np.isfinite(score):
            continue
        rr = dict(row)
        rr["_layer"] = layer
        rr["_head"] = head
        rr["_score"] = score
        parsed.append(rr)

    parsed.sort(key=lambda r: float(r["_score"]), reverse=True)
    if len(parsed) < max_k:
        raise RuntimeError(
            f"Need Top{max_k}, but only {len(parsed)} valid rows for metric={metric}"
        )

    top_rows = parsed[:max_k]
    heads = [(int(r["_layer"]), int(r["_head"])) for r in top_rows]
    if len(set(heads)) != len(heads):
        raise RuntimeError("Duplicate heads in ranking")
    return heads, top_rows


def bundle_condition_name(k: int, kind: str) -> str:
    return f"top{k}_{kind}"


def hname(head: Tuple[int, int]) -> str:
    return f"L{int(head[0])}H{int(head[1]):02d}"


def parse_layers(text: str, n_layers: int) -> List[int]:
    result: List[int] = []
    seen = set()
    for piece in str(text).split(","):
        piece = piece.strip().upper().replace("L", "")
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            start, stop = int(a), int(b)
            values = range(min(start, stop), max(start, stop) + 1)
        else:
            values = [int(piece)]
        for layer in values:
            if not 0 <= layer < n_layers:
                raise ValueError(f"Layer {layer} outside 0..{n_layers-1}")
            if layer not in seen:
                seen.add(layer)
                result.append(layer)
    return sorted(result)


def stratified_subset(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in rows]
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda x: int(x["sid"]))

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected: List[Dict[str, Any]] = []
    cursors = {r: 0 for r in RELATIONS}
    while len(selected) < limit:
        moved = False
        for relation in RELATIONS:
            group = groups.get(relation, [])
            i = cursors[relation]
            if i < len(group) and len(selected) < limit:
                selected.append(group[i])
                cursors[relation] += 1
                moved = True
        if not moved:
            break

    return sorted(selected, key=lambda x: int(x["sid"]))


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_generation(text: str) -> Optional[str]:
    normalized = one_line(text).lower()
    found: List[Tuple[int, int, str]] = []
    for relation, patterns in REL_PATTERNS.items():
        for priority, pattern in enumerate(patterns):
            m = re.search(pattern, normalized)
            if m:
                found.append((m.start(), priority, relation))
                break
    if not found:
        return None
    found.sort()
    return found[0][2]


def safe_mean(xs: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size else float("nan")


# =============================================================================
# Head geometry / matched random controls
# =============================================================================

def validate_heads(
    heads: Sequence[Tuple[int, int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
) -> None:
    for layer, head in heads:
        if not 0 <= layer < len(decoder_layers):
            raise ValueError(f"{hname((layer, head))}: bad layer")
        attn = attention_helper.resolve_self_attention(decoder_layers[layer])
        shape = receiver_module.resolve_attention_shape(attn)
        if not 0 <= head < int(shape.n_query_heads):
            raise ValueError(
                f"{hname((layer, head))}: head outside 0..{int(shape.n_query_heads)-1}"
            )


def matched_random_heads(
    target_heads: Sequence[Tuple[int, int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    excluded: set[Tuple[int, int]],
    seed: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    out: List[Tuple[int, int]] = []
    used = set(excluded)
    for layer, _ in target_heads:
        attn = attention_helper.resolve_self_attention(decoder_layers[layer])
        nh = int(receiver_module.resolve_attention_shape(attn).n_query_heads)
        choices = [(layer, h) for h in range(nh) if (layer, h) not in used]
        if not choices:
            raise RuntimeError(f"No matched random head left at L{layer}")
        pick = rng.choice(choices)
        out.append(pick)
        used.add(pick)
    return out


# =============================================================================
# Tensor output helpers
# =============================================================================

def first_3d_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output) and output.ndim == 3:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("Could not find 3D hidden tensor")


# =============================================================================
# Source head-state capture
# =============================================================================

class SourceHeadCapture:
    """
    Capture REAL swapped/source pre-W_O head slices at every object token.

    Source prompt asks original B relative to original A, so:
        source subject   = B
        source reference = A

    States are retained by OBJECT IDENTITY (A/B).  ROLE vs IDENTITY mapping
    is decided later when constructing an intervention condition.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        heads: Sequence[Tuple[int, int]],
        source_a_positions: Sequence[int],
        source_b_positions: Sequence[int],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver_module = receiver_module
        self.by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer, head in heads:
            self.by_layer[int(layer)].append(int(head))
        self.a_positions = sorted(set(map(int, source_a_positions)))
        self.b_positions = sorted(set(map(int, source_b_positions)))
        if not self.a_positions or not self.b_positions:
            raise ValueError("Empty source object-token positions")
        self.handles: List[Any] = []
        self.states: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    def __enter__(self) -> "SourceHeadCapture":
        for layer, heads in self.by_layer.items():
            attn = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"L{layer} attention lacks o_proj")
            shape = self.receiver_module.resolve_attention_shape(attn)
            nh = int(shape.n_query_heads)
            width = int(o_proj.weight.shape[1])
            if width % nh != 0:
                raise RuntimeError(f"L{layer}: o_proj width/head mismatch")
            hd = width // nh

            def make_hook(layer_index: int, head_ids: Sequence[int], head_dim: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        return None

                    ap = torch.as_tensor(
                        self.a_positions, device=x.device, dtype=torch.long
                    )
                    bp = torch.as_tensor(
                        self.b_positions, device=x.device, dtype=torch.long
                    )
                    for head in head_ids:
                        lo = int(head) * head_dim
                        hi = lo + head_dim
                        a_tokens = x[0].index_select(0, ap)[:, lo:hi]
                        b_tokens = x[0].index_select(0, bp)[:, lo:hi]
                        self.states[(layer_index, int(head))] = {
                            "A_tokens": a_tokens.detach().float().cpu().numpy().astype(np.float32),
                            "B_tokens": b_tokens.detach().float().cpu().numpy().astype(np.float32),
                            "A": a_tokens.mean(dim=0).detach().float().cpu().numpy().astype(np.float32),
                            "B": b_tokens.mean(dim=0).detach().float().cpu().numpy().astype(np.float32),
                        }
                    return None
                return hook

            self.handles.append(
                o_proj.register_forward_pre_hook(
                    make_hook(layer, heads, hd)
                )
            )
        return self

    def validate(self, heads: Sequence[Tuple[int, int]]) -> None:
        missing = [hname(h) for h in heads if h not in self.states]
        if missing:
            raise RuntimeError(f"Missing source head states: {missing}")

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _map_source_rows_to_target(
    *,
    source_rows: torch.Tensor,
    target_rows: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, str]:
    """Return replacement rows with shape == target_rows.shape."""
    if source_rows.ndim != 2 or target_rows.ndim != 2:
        raise ValueError("Expected [tokens, head_dim] tensors")
    ns, ds = int(source_rows.shape[0]), int(source_rows.shape[1])
    nt, dt = int(target_rows.shape[0]), int(target_rows.shape[1])
    if ns < 1 or nt < 1 or ds != dt:
        raise ValueError(
            f"Bad source/target row shapes source={tuple(source_rows.shape)} "
            f"target={tuple(target_rows.shape)}"
        )

    if mode == "tokenwise_resample":
        if ns == nt:
            return source_rows, "exact_tokenwise"
        # Nearest-neighbor sequence resampling.  Every inserted row is one
        # actual source activation; no artificial vector interpolation.
        idx = torch.linspace(
            0, ns - 1, steps=nt, device=source_rows.device,
            dtype=torch.float32,
        ).round().long()
        return source_rows.index_select(0, idx), "nearest_source_row_resample"

    if mode == "pooled_broadcast":
        mean = source_rows.mean(dim=0, keepdim=True)
        return mean.expand(nt, -1), "source_pooled_broadcast"

    if mode == "mean_shift":
        source_mean = source_rows.mean(dim=0, keepdim=True)
        target_mean = target_rows.mean(dim=0, keepdim=True)
        return target_rows + (source_mean - target_mean), "target_cloud_shift_to_source_mean"

    raise ValueError(mode)


# =============================================================================
# Target natural-state replacement + hidden-state capture
# =============================================================================

class TargetNaturalStateReplacementAndCapture:
    """
    Replace selected target pre-W_O head object states by REAL source states.

    replacements maps:
        (layer, head) -> {
            "A_tokens": source rows to place at target A,
            "B_tokens": source rows to place at target B,
            "alignment": "role" | "identity" | "random_role"
        }

    No source-target delta vector is constructed.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        replacements: Mapping[Tuple[int, int], Mapping[str, Any]],
        target_a_positions: Sequence[int],
        target_b_positions: Sequence[int],
        prompt_last: int,
        object_layers: Sequence[int],
        last_layers: Sequence[int],
        replacement_mode: str,
        storage_dtype: np.dtype,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver_module = receiver_module
        self.by_layer: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for (layer, head), payload in replacements.items():
            self.by_layer[int(layer)][int(head)] = {
                "A_tokens": np.asarray(payload["A_tokens"], dtype=np.float32),
                "B_tokens": np.asarray(payload["B_tokens"], dtype=np.float32),
                "alignment": str(payload.get("alignment", "")),
            }

        self.a_positions = sorted(set(map(int, target_a_positions)))
        self.b_positions = sorted(set(map(int, target_b_positions)))
        if not self.a_positions or not self.b_positions:
            raise ValueError("Empty target object-token positions")
        self.prompt_last = int(prompt_last)
        self.object_layers = set(map(int, object_layers))
        self.last_layers = set(map(int, last_layers))
        self.replacement_mode = str(replacement_mode)
        self.storage_dtype = storage_dtype

        self.handles: List[Any] = []
        self.object_features: Dict[int, np.ndarray] = {}
        self.last_features: Dict[int, np.ndarray] = {}
        self.patch_events: Dict[Tuple[int, int], int] = defaultdict(int)
        self.patch_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def __enter__(self) -> "TargetNaturalStateReplacementAndCapture":
        for layer, head_map in self.by_layer.items():
            attn = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"L{layer} attention lacks o_proj")
            shape = self.receiver_module.resolve_attention_shape(attn)
            nh = int(shape.n_query_heads)
            width = int(o_proj.weight.shape[1])
            if width % nh != 0:
                raise RuntimeError(f"L{layer}: o_proj width/head mismatch")
            hd = width // nh

            def make_patch_hook(
                layer_index: int,
                payloads: Mapping[int, Mapping[str, Any]],
                head_dim: int,
            ):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Any:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        return None

                    modified = x.clone()
                    ap = torch.as_tensor(
                        self.a_positions, device=x.device, dtype=torch.long
                    )
                    bp = torch.as_tensor(
                        self.b_positions, device=x.device, dtype=torch.long
                    )

                    for head, payload in payloads.items():
                        lo = int(head) * head_dim
                        hi = lo + head_dim
                        target_a = modified[0].index_select(0, ap)[:, lo:hi]
                        target_b = modified[0].index_select(0, bp)[:, lo:hi]

                        source_a = torch.as_tensor(
                            payload["A_tokens"], device=x.device, dtype=x.dtype
                        )
                        source_b = torch.as_tensor(
                            payload["B_tokens"], device=x.device, dtype=x.dtype
                        )

                        repl_a, map_a = _map_source_rows_to_target(
                            source_rows=source_a,
                            target_rows=target_a,
                            mode=self.replacement_mode,
                        )
                        repl_b, map_b = _map_source_rows_to_target(
                            source_rows=source_b,
                            target_rows=target_b,
                            mode=self.replacement_mode,
                        )

                        before_a_mean = target_a.float().mean(dim=0)
                        before_b_mean = target_b.float().mean(dim=0)
                        source_a_mean = source_a.float().mean(dim=0)
                        source_b_mean = source_b.float().mean(dim=0)

                        modified[0, ap, lo:hi] = repl_a
                        modified[0, bp, lo:hi] = repl_b

                        after_a = modified[0].index_select(0, ap)[:, lo:hi].float()
                        after_b = modified[0].index_select(0, bp)[:, lo:hi].float()

                        self.patch_events[(layer_index, int(head))] += 1
                        self.patch_stats[(layer_index, int(head))] = {
                            "alignment": str(payload.get("alignment", "")),
                            "replacement_mode": self.replacement_mode,
                            "A_mapping": map_a,
                            "B_mapping": map_b,
                            "target_A_tokens": int(target_a.shape[0]),
                            "target_B_tokens": int(target_b.shape[0]),
                            "source_for_A_tokens": int(source_a.shape[0]),
                            "source_for_B_tokens": int(source_b.shape[0]),
                            "A_exact_count_match": bool(int(target_a.shape[0]) == int(source_a.shape[0])),
                            "B_exact_count_match": bool(int(target_b.shape[0]) == int(source_b.shape[0])),
                            "A_before_to_source_mean_l2": float((before_a_mean - source_a_mean).norm().item()),
                            "B_before_to_source_mean_l2": float((before_b_mean - source_b_mean).norm().item()),
                            "A_after_to_source_mean_l2": float((after_a.mean(dim=0) - source_a_mean).norm().item()),
                            "B_after_to_source_mean_l2": float((after_b.mean(dim=0) - source_b_mean).norm().item()),
                        }

                    return (modified, *inputs[1:])
                return hook

            self.handles.append(
                o_proj.register_forward_pre_hook(
                    make_patch_hook(layer, head_map, hd)
                )
            )

        # Capture true decoder residual hidden states after selected blocks.
        for layer in sorted(self.object_layers | self.last_layers):
            block = self.decoder_layers[layer]

            def make_block_hook(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_3d_tensor(output)
                    if int(hidden.shape[0]) != 1:
                        return

                    if layer_index in self.object_layers:
                        ap = torch.as_tensor(
                            self.a_positions, device=hidden.device, dtype=torch.long
                        )
                        bp = torch.as_tensor(
                            self.b_positions, device=hidden.device, dtype=torch.long
                        )
                        a = hidden[0].index_select(0, ap).mean(dim=0)
                        b = hidden[0].index_select(0, bp).mean(dim=0)
                        self.object_features[layer_index] = (
                            (a - b).detach().float().cpu().numpy().astype(self.storage_dtype)
                        )

                    if layer_index in self.last_layers:
                        if 0 <= self.prompt_last < int(hidden.shape[1]):
                            self.last_features[layer_index] = (
                                hidden[0, self.prompt_last]
                                .detach().float().cpu().numpy().astype(self.storage_dtype)
                            )
                return hook

            self.handles.append(block.register_forward_hook(make_block_hook(layer)))

        return self

    def by_layer_as_heads(self) -> List[Tuple[int, int]]:
        return [
            (layer, head)
            for layer, head_map in self.by_layer.items()
            for head in head_map
        ]

    def validate(self) -> None:
        missing_patches = [
            hname(h) for h in self.by_layer_as_heads()
            if self.patch_events[h] < 1
        ]
        if missing_patches:
            raise RuntimeError(f"Natural-state replacement hook did not fire: {missing_patches}")

        missing_object = [
            layer for layer in self.object_layers
            if layer not in self.object_features
        ]
        missing_last = [
            layer for layer in self.last_layers
            if layer not in self.last_features
        ]
        if missing_object or missing_last:
            raise RuntimeError(
                f"Missing captures object={missing_object} last={missing_last}"
            )

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# =============================================================================
# Generation-safe prefill natural-state replacement
# =============================================================================

class PrefillNaturalStateReplacement:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        replacements: Mapping[Tuple[int, int], Mapping[str, Any]],
        target_a_positions: Sequence[int],
        target_b_positions: Sequence[int],
        prompt_length: int,
        replacement_mode: str,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.receiver_module = receiver_module
        self.by_layer: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for (layer, head), payload in replacements.items():
            self.by_layer[int(layer)][int(head)] = {
                "A_tokens": np.asarray(payload["A_tokens"], dtype=np.float32),
                "B_tokens": np.asarray(payload["B_tokens"], dtype=np.float32),
                "alignment": str(payload.get("alignment", "")),
            }
        self.a_positions = sorted(set(map(int, target_a_positions)))
        self.b_positions = sorted(set(map(int, target_b_positions)))
        self.prompt_length = int(prompt_length)
        self.replacement_mode = str(replacement_mode)
        self.handles: List[Any] = []
        self.applications = 0

    def __enter__(self) -> "PrefillNaturalStateReplacement":
        for layer, head_map in self.by_layer.items():
            attn = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )
            o_proj = attn.o_proj
            shape = self.receiver_module.resolve_attention_shape(attn)
            nh = int(shape.n_query_heads)
            width = int(o_proj.weight.shape[1])
            if width % nh != 0:
                raise RuntimeError(f"L{layer}: o_proj width/head mismatch")
            hd = width // nh

            def make_hook(payloads: Mapping[int, Mapping[str, Any]], head_dim: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Any:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        return None
                    # Prefill only. Decode steps normally have sequence length 1.
                    if int(x.shape[1]) < self.prompt_length:
                        return None

                    modified = x.clone()
                    ap = torch.as_tensor(
                        self.a_positions, device=x.device, dtype=torch.long
                    )
                    bp = torch.as_tensor(
                        self.b_positions, device=x.device, dtype=torch.long
                    )

                    for head, payload in payloads.items():
                        lo = int(head) * head_dim
                        hi = lo + head_dim
                        target_a = modified[0].index_select(0, ap)[:, lo:hi]
                        target_b = modified[0].index_select(0, bp)[:, lo:hi]
                        source_a = torch.as_tensor(
                            payload["A_tokens"], device=x.device, dtype=x.dtype
                        )
                        source_b = torch.as_tensor(
                            payload["B_tokens"], device=x.device, dtype=x.dtype
                        )
                        repl_a, _ = _map_source_rows_to_target(
                            source_rows=source_a,
                            target_rows=target_a,
                            mode=self.replacement_mode,
                        )
                        repl_b, _ = _map_source_rows_to_target(
                            source_rows=source_b,
                            target_rows=target_b,
                            mode=self.replacement_mode,
                        )
                        modified[0, ap, lo:hi] = repl_a
                        modified[0, bp, lo:hi] = repl_b

                    self.applications += 1
                    return (modified, *inputs[1:])
                return hook

            self.handles.append(
                o_proj.register_forward_pre_hook(make_hook(head_map, hd))
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


# =============================================================================
# Forward / generation
# =============================================================================

@torch.inference_mode()
def capture_source_states(
    *,
    model: Any,
    batch: Mapping[str, Any],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    heads: Sequence[Tuple[int, int]],
    source_a_positions: Sequence[int],
    source_b_positions: Sequence[int],
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    capture = SourceHeadCapture(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        heads=heads,
        source_a_positions=source_a_positions,
        source_b_positions=source_b_positions,
    )
    with capture:
        out = model(**batch, use_cache=False, return_dict=True)
    capture.validate(heads)
    del out
    return capture.states


@torch.inference_mode()
def run_target_forward(
    *,
    model: Any,
    batch: Mapping[str, Any],
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    attention_helper: Any,
    receiver_module: Any,
    target_a_positions: Sequence[int],
    target_b_positions: Sequence[int],
    prompt_last: int,
    object_layers: Sequence[int],
    last_layers: Sequence[int],
    replacements: Mapping[Tuple[int, int], Mapping[str, Any]],
    replacement_mode: str,
    storage_dtype: np.dtype,
) -> Dict[str, Any]:
    capture = TargetNaturalStateReplacementAndCapture(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        replacements=replacements,
        target_a_positions=target_a_positions,
        target_b_positions=target_b_positions,
        prompt_last=prompt_last,
        object_layers=object_layers,
        last_layers=last_layers,
        replacement_mode=replacement_mode,
        storage_dtype=storage_dtype,
    )
    with capture:
        outputs = model(**batch, use_cache=False, return_dict=True)

    capture.validate()
    relation = base.relation_scores(
        outputs.logits[0, -1],
        dict(relation_token_map),
        gt=None,
    )
    pred = str(relation["prediction"])
    logits = np.asarray(relation["logits"], dtype=np.float32)
    del outputs

    return {
        "prediction": pred,
        "relation_logits": logits,
        "object": capture.object_features,
        "last": capture.last_features,
        "patch_stats": {
            hname(k): v for k, v in capture.patch_stats.items()
        },
    }


def generation_kwargs(processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    tok = processor.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kw: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "pad_token_id": pad,
    }
    if tok.eos_token_id is not None:
        kw["eos_token_id"] = tok.eos_token_id
    return kw


@torch.inference_mode()
def run_generation(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    replacements: Mapping[Tuple[int, int], Mapping[str, Any]],
    target_a_positions: Sequence[int],
    target_b_positions: Sequence[int],
    replacement_mode: str,
    gen_kw: Mapping[str, Any],
) -> Dict[str, Any]:
    prompt_length = int(batch["input_ids"].shape[1])
    patcher = PrefillNaturalStateReplacement(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        replacements=replacements,
        target_a_positions=target_a_positions,
        target_b_positions=target_b_positions,
        prompt_length=prompt_length,
        replacement_mode=replacement_mode,
    )
    with patcher:
        out = model.generate(**batch, **dict(gen_kw))

    seq = out.sequences[0]
    new = seq[prompt_length:]
    ids = [int(x) for x in new.detach().cpu().tolist()]
    text = processor.tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    pred = parse_generation(text)
    del out
    return {
        "prediction": pred,
        "text": one_line(text),
        "token_ids": ids,
    }


# =============================================================================
# Frozen probes
# =============================================================================

def fit_direction_probe(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    center = x.mean(axis=0)
    centered = x - center[None, :]

    dirs = []
    for label in range(len(RELATIONS)):
        rows = centered[y == label]
        if len(rows) == 0:
            raise RuntimeError(f"No train samples for label={label}")
        d = rows.mean(axis=0)
        norm = np.linalg.norm(d)
        if norm <= 1e-12:
            raise RuntimeError(f"Zero prototype norm for label={label}")
        dirs.append(d / norm)
    return center, np.stack(dirs, axis=0)


def predict_direction_probe(
    x: np.ndarray,
    center: np.ndarray,
    dirs: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = x - center[None, :]
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.clip(norms, 1e-12, None)
    scores = z @ np.asarray(dirs, dtype=np.float64).T
    return np.argmax(scores, axis=1).astype(np.int64)


def split_indices(
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = max(4, int(round(n * train_ratio)))
    n_train = min(n_train, n - 4)

    # Retry a few random permutations until every class appears in train/test.
    for _ in range(100):
        train = indices[:n_train]
        test = indices[n_train:]
        if (
            len(np.unique(labels[train])) == len(RELATIONS)
            and len(np.unique(labels[test])) == len(RELATIONS)
        ):
            return train, test
        rng.shuffle(indices)

    raise RuntimeError("Could not build 4-class train/test split")


def probe_rows_for_layer(
    *,
    layer: int,
    clean_x: np.ndarray,
    condition_x: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_ratio: float,
    repeats: int,
    seed: int,
    feature_type: str,
) -> List[Dict[str, Any]]:
    stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for repeat in range(repeats):
        train, test = split_indices(
            labels,
            train_ratio=train_ratio,
            seed=seed + 1009 * repeat + 17 * layer,
        )
        center, dirs = fit_direction_probe(clean_x[train], labels[train])
        source_labels = np.asarray(
            [REL_TO_ID[OPPOSITE[ID_TO_REL[int(v)]]] for v in labels[test]],
            dtype=np.int64,
        )

        for condition, x in condition_x.items():
            pred = predict_direction_probe(x[test], center, dirs)
            target_acc = float(np.mean(pred == labels[test]))
            source_follow = float(np.mean(pred == source_labels))
            other = float(np.mean(
                (pred != labels[test]) & (pred != source_labels)
            ))
            stats[condition].append({
                "target_accuracy": target_acc,
                "source_follow_accuracy": source_follow,
                "other_accuracy": other,
            })

    rows: List[Dict[str, Any]] = []
    for condition, values in stats.items():
        row: Dict[str, Any] = {
            "feature_type": feature_type,
            "layer": int(layer),
            "condition": condition,
            "N": int(len(labels)),
            "repeats": int(repeats),
        }
        for metric in (
            "target_accuracy",
            "source_follow_accuracy",
            "other_accuracy",
        ):
            vals = [v[metric] for v in values]
            row[f"{metric}_mean"] = safe_mean(vals)
            row[f"{metric}_std"] = safe_std(vals)
        rows.append(row)
    return rows


# =============================================================================
# Direct prediction summaries
# =============================================================================

def prediction_summary_rows(
    *,
    predictions: Mapping[str, Sequence[Optional[str]]],
    labels: Sequence[str],
    title: str,
) -> List[Dict[str, Any]]:
    labels = list(labels)
    clean = list(predictions["clean"])
    rows: List[Dict[str, Any]] = []

    for condition, pred_seq in predictions.items():
        preds = list(pred_seq)
        target = np.asarray(
            [p == g for p, g in zip(preds, labels)],
            dtype=np.float64,
        )
        source = np.asarray(
            [p == OPPOSITE[g] for p, g in zip(preds, labels)],
            dtype=np.float64,
        )
        parsed = np.asarray(
            [p in RELATIONS for p in preds],
            dtype=np.float64,
        )

        clean_target_to_source = sum(
            (c == g) and (p == OPPOSITE[g])
            for c, p, g in zip(clean, preds, labels)
        )
        clean_target_correct = sum(c == g for c, g in zip(clean, labels))

        rows.append({
            "metric": title,
            "condition": condition,
            "N": len(labels),
            "target_accuracy": float(target.mean()),
            "source_follow_accuracy": float(source.mean()),
            "parse_rate": float(parsed.mean()),
            "clean_target_correct_to_source": int(clean_target_to_source),
            "clean_target_correct_N": int(clean_target_correct),
            "clean_target_correct_to_source_rate": (
                float(clean_target_to_source / clean_target_correct)
                if clean_target_correct else float("nan")
            ),
            "prediction_change_rate_vs_clean": float(
                np.mean([p != c for p, c in zip(preds, clean)])
            ),
        })
    return rows


def print_prediction_table(
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    print("\n" + "=" * 114)
    print(title)
    print("=" * 114)
    print(
        f"{'condition':<28s} {'N':>5s} {'target':>9s} {'SOURCE':>9s} "
        f"{'cleanGT->src':>13s} {'change':>9s} {'parse':>8s}"
    )
    print("-" * 114)
    for row in rows:
        print(
            f"{str(row['condition']):<28s} "
            f"{int(row['N']):>5d} "
            f"{100*float(row['target_accuracy']):>8.2f}% "
            f"{100*float(row['source_follow_accuracy']):>8.2f}% "
            f"{100*float(row['clean_target_correct_to_source_rate']):>12.2f}% "
            f"{100*float(row['prediction_change_rate_vs_clean']):>8.2f}% "
            f"{100*float(row['parse_rate']):>7.2f}%"
        )


# =============================================================================
# Condition construction
# =============================================================================

def source_object_rows(
    source_state: Mapping[str, np.ndarray],
    identity: str,
) -> np.ndarray:
    key = f"{identity}_tokens"
    if key not in source_state:
        raise KeyError(key)
    return np.asarray(source_state[key], dtype=np.float32)


def replacement_payload(
    *,
    source_state: Mapping[str, np.ndarray],
    alignment: str,
) -> Dict[str, Any]:
    if alignment == "role":
        # Source/swapped asks B relative A:
        # source subject B -> target subject A
        # source reference A -> target reference B
        return {
            "A_tokens": source_object_rows(source_state, "B"),
            "B_tokens": source_object_rows(source_state, "A"),
            "alignment": "role",
        }
    if alignment == "identity":
        return {
            "A_tokens": source_object_rows(source_state, "A"),
            "B_tokens": source_object_rows(source_state, "B"),
            "alignment": "identity",
        }
    raise ValueError(alignment)


def build_condition_replacements(
    *,
    condition: str,
    ranked_heads: Sequence[Tuple[int, int]],
    random_heads: Sequence[Tuple[int, int]],
    bundle_sizes: Sequence[int],
    source_states: Mapping[Tuple[int, int], Mapping[str, np.ndarray]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if condition == "clean":
        return {}

    m = re.fullmatch(r"top(\d+)_(role|identity|random_role)", condition)
    if not m:
        raise ValueError(f"Unknown bundle condition: {condition}")

    k = int(m.group(1))
    kind = str(m.group(2))
    if k not in set(map(int, bundle_sizes)):
        raise ValueError(f"Bundle size {k} not selected")
    if k > len(ranked_heads) or k > len(random_heads):
        raise ValueError(f"Bundle size {k} exceeds available heads")

    if kind == "role":
        heads = list(ranked_heads[:k])
        alignment = "role"
    elif kind == "identity":
        heads = list(ranked_heads[:k])
        alignment = "identity"
    else:
        heads = list(random_heads[:k])
        alignment = "role"

    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for h in heads:
        payload = replacement_payload(
            source_state=source_states[h],
            alignment=alignment,
        )
        if kind == "random_role":
            payload["alignment"] = "random_role"
        out[h] = payload
    return out


# =============================================================================
# Compact per-head report
# =============================================================================

def build_individual_summary(
    *,
    object_rows: Sequence[Mapping[str, Any]],
    last_rows: Sequence[Mapping[str, Any]],
    tested_heads: Sequence[Tuple[int, int]],
    random_heads: Sequence[Tuple[int, int]],
    report_layers: Sequence[int],
) -> List[Dict[str, Any]]:
    obj = {
        (int(r["layer"]), str(r["condition"])): r
        for r in object_rows
    }
    last = {
        (int(r["layer"]), str(r["condition"])): r
        for r in last_rows
    }

    rows: List[Dict[str, Any]] = []
    for i, head in enumerate(tested_heads):
        name = hname(head)
        role = f"{name}__role"
        identity = f"{name}__identity"
        random_c = f"{name}__random_role"

        for layer in report_layers:
            if (layer, role) not in obj and (layer, role) not in last:
                continue
            row: Dict[str, Any] = {
                "head": name,
                "matched_random_head": hname(random_heads[i]),
                "intervention_layer": int(head[0]),
                "report_layer": int(layer),
            }

            if (layer, role) in obj:
                clean = obj[(layer, "clean")]
                rr = obj[(layer, role)]
                ii = obj[(layer, identity)]
                rc = obj[(layer, random_c)]
                row.update({
                    "object_clean_source_follow": clean["source_follow_accuracy_mean"],
                    "object_role_source_follow": rr["source_follow_accuracy_mean"],
                    "object_identity_source_follow": ii["source_follow_accuracy_mean"],
                    "object_random_source_follow": rc["source_follow_accuracy_mean"],
                    "object_role_excess_vs_random": (
                        float(rr["source_follow_accuracy_mean"])
                        - float(rc["source_follow_accuracy_mean"])
                    ),
                    "object_role_excess_vs_identity": (
                        float(rr["source_follow_accuracy_mean"])
                        - float(ii["source_follow_accuracy_mean"])
                    ),
                    "object_role_target_accuracy": rr["target_accuracy_mean"],
                })

            if (layer, role) in last:
                clean = last[(layer, "clean")]
                rr = last[(layer, role)]
                ii = last[(layer, identity)]
                rc = last[(layer, random_c)]
                row.update({
                    "last_clean_source_follow": clean["source_follow_accuracy_mean"],
                    "last_role_source_follow": rr["source_follow_accuracy_mean"],
                    "last_identity_source_follow": ii["source_follow_accuracy_mean"],
                    "last_random_source_follow": rc["source_follow_accuracy_mean"],
                    "last_role_excess_vs_random": (
                        float(rr["source_follow_accuracy_mean"])
                        - float(rc["source_follow_accuracy_mean"])
                    ),
                    "last_role_excess_vs_identity": (
                        float(rr["source_follow_accuracy_mean"])
                        - float(ii["source_follow_accuracy_mean"])
                    ),
                    "last_role_target_accuracy": rr["target_accuracy_mean"],
                })

            rows.append(row)
    return rows


def print_compact_head_summary(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    interesting = [
        r for r in rows
        if "object_role_source_follow" in r
        and int(r["report_layer"]) >= int(r["intervention_layer"])
    ]
    if not interesting:
        return

    print("\n" + "=" * 128)
    print("INDIVIDUAL HEAD: DOWNSTREAM OBJECT SOURCE-FOLLOW")
    print("=" * 128)
    print(
        f"{'head':<9s} {'random':<9s} {'layer':>6s} "
        f"{'clean':>9s} {'ROLE':>9s} {'identity':>10s} {'random':>9s} "
        f"{'role-rnd':>10s} {'role-id':>9s}"
    )
    print("-" * 128)
    for r in interesting:
        print(
            f"{str(r['head']):<9s} {str(r['matched_random_head']):<9s} "
            f"L{int(r['report_layer']):<5d} "
            f"{100*float(r['object_clean_source_follow']):>8.2f}% "
            f"{100*float(r['object_role_source_follow']):>8.2f}% "
            f"{100*float(r['object_identity_source_follow']):>9.2f}% "
            f"{100*float(r['object_random_source_follow']):>8.2f}% "
            f"{100*float(r['object_role_excess_vs_random']):>+9.2f} "
            f"{100*float(r['object_role_excess_vs_identity']):>+8.2f}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0 < args.probe_train_ratio < 1:
        raise ValueError("--probe-train-ratio must be in (0,1)")
    if args.probe_repeats < 1:
        raise ValueError("--probe-repeats must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output directory not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    three = import_file(Path(args.three_stage_script), "_s1tx_three")
    ioi = import_file(Path(args.ioi_script), "_s1tx_ioi")
    producer = import_file(Path(args.producer_script), "_s1tx_producer")
    receiver = import_file(Path(args.receiver_script), "_s1tx_receiver")
    v3 = import_file(Path(args.v3_script), "_s1tx_v3")
    base = import_file(Path(args.base_script), "_s1tx_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "_s1tx_attention",
    )

    source_dir = Path(args.source_output_dir)
    extraction = source_dir / "extraction.jsonl"
    config_path = source_dir / "config.json"
    if not extraction.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Need config.json + extraction.jsonl in {source_dir}"
        )

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(source_config.get("model")) != args.model:
        raise RuntimeError(
            f"Source cache model={source_config.get('model')} != --model={args.model}"
        )

    rows = read_jsonl(extraction)
    rows = [r for r in rows if str(r.get("gt")) in RELATIONS]
    if args.pair_status != "all":
        rows = [
            r for r in rows
            if str(r.get("generation_pair_status", "")) == args.pair_status
        ]
    rows = stratified_subset(rows, args.max_samples, args.sample_seed)
    if not rows:
        raise RuntimeError("No eligible samples")

    model = processor = None
    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )

        # prepare_data_helpers interprets args.max_samples as dataset truncation.
        saved_max = args.max_samples
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(
                args, base
            )
        finally:
            args.max_samples = saved_max

        bundle_sizes = parse_bundle_sizes(args.bundle_sizes)
        max_k = max(bundle_sizes)
        ranked_heads, ranked_rows = load_ranked_direction_heads(
            Path(args.direction_results),
            args.rank_metric,
            max_k=max_k,
        )
        validate_heads(
            ranked_heads, decoder_layers, attention_helper, receiver
        )

        # One nested matched-random list for TopMaxK.
        # Prefix k therefore matches the per-layer profile of Direction TopK.
        random_heads = matched_random_heads(
            ranked_heads,
            decoder_layers,
            attention_helper,
            receiver,
            excluded=set(ranked_heads),
            seed=args.seed + 991,
        )
        validate_heads(
            random_heads, decoder_layers, attention_helper, receiver
        )

        object_layers = parse_layers(
            args.object_probe_layers, len(decoder_layers)
        )
        last_layers = parse_layers(
            args.last_probe_layers, len(decoder_layers)
        )
        report_layers = parse_layers(
            args.report_layers, len(decoder_layers)
        )

        all_source_heads = list(
            dict.fromkeys(ranked_heads + random_heads)
        )

        # Bundle-only: no individual head conditions.
        conditions: List[str] = ["clean"]
        for k in bundle_sizes:
            conditions.extend([
                bundle_condition_name(k, "role"),
                bundle_condition_name(k, "identity"),
                bundle_condition_name(k, "random_role"),
            ])

        storage_dtype = (
            np.float16 if args.feature_dtype == "float16" else np.float32
        )

        config = {
            "version": VERSION,
            "model": args.model,
            "repo_id": getattr(spec, "repo_id", ""),
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "N": len(rows),
            "pair_status": args.pair_status,
            "ranked_heads_top_maxk": [hname(h) for h in ranked_heads],
            "matched_random_heads_top_maxk": [hname(h) for h in random_heads],
            "rank_metric": args.rank_metric,
            "direction_results": str(Path(args.direction_results)),
            "bundle_sizes": bundle_sizes,
            "ranked_head_scores": [
                {
                    "rank": i + 1,
                    "head": hname(ranked_heads[i]),
                    "score": float(ranked_rows[i]["_score"]),
                }
                for i in range(len(ranked_heads))
            ],
            "ranked_to_random": {
                hname(h): hname(random_heads[i])
                for i, h in enumerate(ranked_heads)
            },
            "object_probe_layers": object_layers,
            "last_probe_layers": last_layers,
            "report_layers": report_layers,
            "conditions": conditions,
            "replacement_mode": args.replacement_mode,
            "probe_train_ratio": args.probe_train_ratio,
            "probe_repeats": args.probe_repeats,
            "probe_seed": args.probe_seed,
            "run_generation": args.run_generation,
            "feature_dtype": args.feature_dtype,
            "intervention": (
                "direct replacement of target pre-W_O object-head states by real "
                "swapped/source object-head states; no constructed relation delta"
            ),
            "source_relation": (
                "same image, swapped query roles: B relative A = opposite(target GT)"
            ),
            "audit": audit,
        }
        write_json(out_dir / "config.json", config)

        print("\n" + "=" * 120)
        print("STAGE-I NATURAL HEAD-STATE REPLACEMENT")
        print("=" * 120)
        print("N               :", len(rows))
        print("pair status     :", args.pair_status)
        print("replacement mode:", args.replacement_mode)
        print("rank metric     :", args.rank_metric)
        print("direction file  :", args.direction_results)
        print("bundle sizes    :", bundle_sizes)
        print("TopMax heads    :", ", ".join(hname(h) for h in ranked_heads))
        print("matched random  :", ", ".join(hname(h) for h in random_heads))
        print("object layers   :", object_layers)
        print("last layers     :", last_layers)
        print("conditions      :", len(conditions))
        print("=" * 120, flush=True)

        # condition -> layer -> list(feature)
        object_features: Dict[str, Dict[int, List[np.ndarray]]] = {
            c: {l: [] for l in object_layers} for c in conditions
        }
        last_features: Dict[str, Dict[int, List[np.ndarray]]] = {
            c: {l: [] for l in last_layers} for c in conditions
        }
        next_predictions: Dict[str, List[Optional[str]]] = {
            c: [] for c in conditions
        }

        labels_text: List[str] = []
        successful_sids: List[int] = []
        sample_path = out_dir / "sample_predictions.jsonl"
        error_path = out_dir / "errors.jsonl"

        # Generation subset is selected by sid up front.
        generation_rows = stratified_subset(
            rows,
            args.generation_max_samples if args.run_generation else 0,
            args.sample_seed + 404,
        )
        generation_sids = (
            {int(r["sid"]) for r in generation_rows}
            if args.run_generation else set()
        )
        generation_predictions: Dict[str, List[Optional[str]]] = defaultdict(list)
        generation_labels: List[str] = []
        generation_sid_order: List[int] = []
        gen_kw = generation_kwargs(processor, args)

        for sample_index, source_row in enumerate(
            tqdm(rows, desc=f"stage1-natural-swap:{args.model}"),
            start=1,
        ):
            pair = None
            try:
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

                gt = str(pair.gt)
                source_gt = OPPOSITE[gt]

                source_states = capture_source_states(
                    model=model,
                    batch=pair.swapped_batch,
                    decoder_layers=decoder_layers,
                    attention_helper=attention_helper,
                    receiver_module=receiver,
                    heads=all_source_heads,
                    source_a_positions=pair.swapped_a_positions,
                    source_b_positions=pair.swapped_b_positions,
                )

                sample_record: Dict[str, Any] = {
                    "sid": int(pair.sid),
                    "gt_target": gt,
                    "gt_source": source_gt,
                    "subject": pair.subject,
                    "reference": pair.reference,
                }

                condition_replacements_cache: Dict[
                    str, Dict[Tuple[int, int], Dict[str, Any]]
                ] = {}

                for condition in conditions:
                    replacements = build_condition_replacements(
                        condition=condition,
                        ranked_heads=ranked_heads,
                        random_heads=random_heads,
                        bundle_sizes=bundle_sizes,
                        source_states=source_states,
                    )
                    condition_replacements_cache[condition] = replacements

                    result = run_target_forward(
                        model=model,
                        batch=pair.original_batch,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        base=base,
                        attention_helper=attention_helper,
                        receiver_module=receiver,
                        target_a_positions=pair.original_a_positions,
                        target_b_positions=pair.original_b_positions,
                        prompt_last=pair.original_prompt_last,
                        object_layers=object_layers,
                        last_layers=last_layers,
                        replacements=replacements,
                        replacement_mode=args.replacement_mode,
                        storage_dtype=storage_dtype,
                    )

                    next_predictions[condition].append(result["prediction"])
                    sample_record[f"{condition}__next"] = result["prediction"]

                    for layer in object_layers:
                        object_features[condition][layer].append(
                            result["object"][layer]
                        )
                    for layer in last_layers:
                        last_features[condition][layer].append(
                            result["last"][layer]
                        )

                    if condition != "clean":
                        sample_record[
                            f"{condition}__patch_stats"
                        ] = result["patch_stats"]

                # Optional full generation.
                if args.run_generation and int(pair.sid) in generation_sids:
                    generation_sid_order.append(int(pair.sid))
                    generation_labels.append(gt)

                    gen_conditions = list(conditions)

                    for condition in gen_conditions:
                        g = run_generation(
                            model=model,
                            processor=processor,
                            batch=pair.original_batch,
                            decoder_layers=decoder_layers,
                            attention_helper=attention_helper,
                            receiver_module=receiver,
                            replacements=condition_replacements_cache[condition],
                            target_a_positions=pair.original_a_positions,
                            target_b_positions=pair.original_b_positions,
                            replacement_mode=args.replacement_mode,
                            gen_kw=gen_kw,
                        )
                        generation_predictions[condition].append(g["prediction"])
                        sample_record[f"{condition}__generation"] = g["prediction"]
                        sample_record[f"{condition}__generation_text"] = g["text"]

                labels_text.append(gt)
                successful_sids.append(int(pair.sid))
                append_jsonl(sample_path, sample_record)

                if (
                    args.print_every > 0
                    and sample_index % args.print_every == 0
                ):
                    compact = []
                    for k in bundle_sizes:
                        c = bundle_condition_name(k, "role")
                        compact.append(f"Top{k}:{gt}->{next_predictions[c][-1]}")
                    tqdm.write(
                        f"sid={pair.sid} target={gt} source={source_gt} | "
                        + " ".join(compact)
                    )

            except Exception as exc:
                append_jsonl(error_path, {
                    "sid": int(source_row["sid"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    f"\n[ERROR sid={source_row['sid']}] "
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

        if not successful_sids:
            raise RuntimeError("No successful samples")

        labels = np.asarray(
            [REL_TO_ID[g] for g in labels_text],
            dtype=np.int64,
        )

        # Stack CPU feature arrays once.
        object_arrays: Dict[str, Dict[int, np.ndarray]] = {
            c: {
                l: np.stack(object_features[c][l], axis=0)
                for l in object_layers
            }
            for c in conditions
        }
        last_arrays: Dict[str, Dict[int, np.ndarray]] = {
            c: {
                l: np.stack(last_features[c][l], axis=0)
                for l in last_layers
            }
            for c in conditions
        }

        object_probe_rows: List[Dict[str, Any]] = []
        for layer in object_layers:
            condition_x = {
                c: object_arrays[c][layer] for c in conditions
            }
            object_probe_rows.extend(
                probe_rows_for_layer(
                    layer=layer,
                    clean_x=object_arrays["clean"][layer],
                    condition_x=condition_x,
                    labels=labels,
                    train_ratio=args.probe_train_ratio,
                    repeats=args.probe_repeats,
                    seed=args.probe_seed,
                    feature_type="object_hidden_A_minus_B",
                )
            )

        last_probe_rows: List[Dict[str, Any]] = []
        for layer in last_layers:
            condition_x = {
                c: last_arrays[c][layer] for c in conditions
            }
            last_probe_rows.extend(
                probe_rows_for_layer(
                    layer=layer,
                    clean_x=last_arrays["clean"][layer],
                    condition_x=condition_x,
                    labels=labels,
                    train_ratio=args.probe_train_ratio,
                    repeats=args.probe_repeats,
                    seed=args.probe_seed + 7001,
                    feature_type="prompt_last_hidden",
                )
            )

        write_csv(out_dir / "object_hidden_probe.csv", object_probe_rows)
        write_csv(out_dir / "last_hidden_probe.csv", last_probe_rows)

        next_rows = prediction_summary_rows(
            predictions=next_predictions,
            labels=labels_text,
            title="next_token",
        )
        write_csv(out_dir / "next_token_summary.csv", next_rows)
        print_prediction_table("NEXT-TOKEN TARGET vs SOURCE-FOLLOW", next_rows)

        if args.run_generation and generation_labels:
            gen_rows = prediction_summary_rows(
                predictions=generation_predictions,
                labels=generation_labels,
                title="generation",
            )
            write_csv(out_dir / "generation_summary.csv", gen_rows)
            print_prediction_table(
                "FULL GENERATION TARGET vs SOURCE-FOLLOW",
                gen_rows,
            )

        # Bundle-only compact tables.
        obj_map = {
            (int(r["layer"]), str(r["condition"])): r
            for r in object_probe_rows
        }
        last_map = {
            (int(r["layer"]), str(r["condition"])): r
            for r in last_probe_rows
        }

        bundle_summary_rows: List[Dict[str, Any]] = []

        print("\n" + "=" * 132)
        print("BUNDLE SWEEP: DOWNSTREAM OBJECT SOURCE-FOLLOW")
        print("=" * 132)
        print(
            f"{'K':>4s} {'layer':>6s} {'clean':>9s} {'ROLE':>9s} "
            f"{'identity':>10s} {'random':>9s} {'role-rnd':>10s} {'role-id':>9s}"
        )
        print("-" * 132)

        for k in bundle_sizes:
            role_c = bundle_condition_name(k, "role")
            ident_c = bundle_condition_name(k, "identity")
            rnd_c = bundle_condition_name(k, "random_role")
            for layer in report_layers:
                key = (layer, role_c)
                if key not in obj_map:
                    continue
                clean = obj_map[(layer, "clean")]
                role = obj_map[(layer, role_c)]
                ident = obj_map[(layer, ident_c)]
                rnd = obj_map[(layer, rnd_c)]

                row = {
                    "feature_type": "object_hidden_A_minus_B",
                    "K": int(k),
                    "layer": int(layer),
                    "clean_source_follow": float(clean["source_follow_accuracy_mean"]),
                    "role_source_follow": float(role["source_follow_accuracy_mean"]),
                    "identity_source_follow": float(ident["source_follow_accuracy_mean"]),
                    "random_source_follow": float(rnd["source_follow_accuracy_mean"]),
                    "role_excess_vs_random": (
                        float(role["source_follow_accuracy_mean"])
                        - float(rnd["source_follow_accuracy_mean"])
                    ),
                    "role_excess_vs_identity": (
                        float(role["source_follow_accuracy_mean"])
                        - float(ident["source_follow_accuracy_mean"])
                    ),
                    "role_target_accuracy": float(role["target_accuracy_mean"]),
                }
                bundle_summary_rows.append(row)
                print(
                    f"{k:>4d} L{layer:<5d} "
                    f"{100*row['clean_source_follow']:>8.2f}% "
                    f"{100*row['role_source_follow']:>8.2f}% "
                    f"{100*row['identity_source_follow']:>9.2f}% "
                    f"{100*row['random_source_follow']:>8.2f}% "
                    f"{100*row['role_excess_vs_random']:>+9.2f} "
                    f"{100*row['role_excess_vs_identity']:>+8.2f}"
                )

        print("\n" + "=" * 132)
        print("BUNDLE SWEEP: PROMPT-LAST SOURCE-FOLLOW")
        print("=" * 132)
        print(
            f"{'K':>4s} {'layer':>6s} {'clean':>9s} {'ROLE':>9s} "
            f"{'identity':>10s} {'random':>9s} {'role-rnd':>10s} {'role-id':>9s}"
        )
        print("-" * 132)

        for k in bundle_sizes:
            role_c = bundle_condition_name(k, "role")
            ident_c = bundle_condition_name(k, "identity")
            rnd_c = bundle_condition_name(k, "random_role")
            for layer in report_layers:
                key = (layer, role_c)
                if key not in last_map:
                    continue
                clean = last_map[(layer, "clean")]
                role = last_map[(layer, role_c)]
                ident = last_map[(layer, ident_c)]
                rnd = last_map[(layer, rnd_c)]

                row = {
                    "feature_type": "prompt_last_hidden",
                    "K": int(k),
                    "layer": int(layer),
                    "clean_source_follow": float(clean["source_follow_accuracy_mean"]),
                    "role_source_follow": float(role["source_follow_accuracy_mean"]),
                    "identity_source_follow": float(ident["source_follow_accuracy_mean"]),
                    "random_source_follow": float(rnd["source_follow_accuracy_mean"]),
                    "role_excess_vs_random": (
                        float(role["source_follow_accuracy_mean"])
                        - float(rnd["source_follow_accuracy_mean"])
                    ),
                    "role_excess_vs_identity": (
                        float(role["source_follow_accuracy_mean"])
                        - float(ident["source_follow_accuracy_mean"])
                    ),
                    "role_target_accuracy": float(role["target_accuracy_mean"]),
                }
                bundle_summary_rows.append(row)
                print(
                    f"{k:>4d} L{layer:<5d} "
                    f"{100*row['clean_source_follow']:>8.2f}% "
                    f"{100*row['role_source_follow']:>8.2f}% "
                    f"{100*row['identity_source_follow']:>9.2f}% "
                    f"{100*row['random_source_follow']:>8.2f}% "
                    f"{100*row['role_excess_vs_random']:>+9.2f} "
                    f"{100*row['role_excess_vs_identity']:>+8.2f}"
                )

        write_csv(
            out_dir / "bundle_sweep_summary.csv",
            bundle_summary_rows,
        )

        report = [
            f"version: {VERSION}",
            f"model: {args.model}",
            f"N successful: {len(successful_sids)}",
            f"pair_status: {args.pair_status}",
            f"replacement_mode: {args.replacement_mode}",
            "",
            "MAIN TEST",
            "A distributed Stage-I bundle is supported if cumulative natural ROLE replacement:",
            "  1) yields source-follow well above identity and matched-random bundles;",
            "  2) strengthens as K grows from Top10 to Top20/25/30;",
            "  3) propagates into later object and prompt-last hidden states;",
            "  4) ideally also increases next-token / generation source-follow.",
            "",
            "IMPORTANT CONTROL INTERPRETATION",
            "role replacement    : real source subject(B)->target A and source reference(A)->target B",
            "identity replacement: real source A->target A and source B->target B",
            "random role         : same natural ROLE replacement on a same-layer random head",
            "",
            "A convincing result looks like:",
            "  downstream object source-follow:",
            "      role >> random",
            "      role >> identity",
            "  with little/no effect in layers before the intervention.",
            "",
            "FILES",
            "object_hidden_probe.csv",
            "last_hidden_probe.csv",
            "next_token_summary.csv",
            "generation_summary.csv (if enabled)",
            "bundle_sweep_summary.csv",
            "sample_predictions.jsonl",
        ]
        (out_dir / "report.txt").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "config.json",
            "object_hidden_probe.csv",
            "last_hidden_probe.csv",
            "next_token_summary.csv",
            "bundle_sweep_summary.csv",
            "sample_predictions.jsonl",
            "report.txt",
        ):
            print(" ", out_dir / name)
        if args.run_generation:
            print(" ", out_dir / "generation_summary.csv")

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
