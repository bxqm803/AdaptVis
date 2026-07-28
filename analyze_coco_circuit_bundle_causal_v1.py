#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundle-level causal analysis for the COCO two-object spatial task.

This script is intended to follow:

    analyze_spatial_storage_transport_utilization_v3.py
    analyze_coco_producer_qk_ov_v1.py
    analyze_coco_receiver_qkv_v1.py

Motivation
----------
Single-head interventions can underestimate causal importance when a Transformer
uses parallel heads, grouped-query attention, repeated cross-layer writes, and
residual bypasses. This script evaluates functional bundles rather than treating
one head as the entire circuit.

The script supports four complementary tests.

1. Dynamic whole-head bundle ablation
   ----------------------------------
   At selected target tokens, zero selected head slices immediately before W_O.

   Producer bundle:
       selected heads are zeroed at subject/reference object-token positions.

   Receiver bundle:
       selected heads are zeroed at prompt-last.

   Combined bundle:
       producer and receiver bundles are zeroed in the same forward pass.

   This is a dynamic intervention. Downstream layers recompute after upstream
   heads have been removed.

2. Component-specific bundle ablation
   ----------------------------------
   Subtract the clean-run component previously measured by the producer/receiver
   scripts:

       producer: visual-token -> object-token A V W_O write
       receiver: object-token -> prompt-last A V W_O write

   This is more function-specific than whole-head zeroing, but the subtracted
   vector is computed from the clean baseline trace.

3. Bundle Q/K/V projection patching
   --------------------------------
   Patch multiple receiver projection-head slices together.

       Q: prompt-last Q
       K: object-token K
       V: object-token V

   K/V support two alignments:

       identity:
           original A -> swapped A
           original B -> swapped B

       role:
           original subject A   -> swapped subject B
           original reference B -> swapped reference A

   In grouped-query attention, K/V heads are automatically deduplicated.

4. Redundancy diagnostics
   ----------------------
   Optional singleton and leave-one-out bundles allow the script to report:

       bundle effect
       sum of singleton effects
       bundle - sum(singletons)
       bundle / sum(singletons)
       full bundle - leave-one-out bundle

   Optional cumulative top-k receiver bundles and layer-matched random controls
   can be generated from receiver_head_scan.csv.

Important interpretation
------------------------
* A large whole-head bundle effect shows that the selected head set is jointly
  necessary at the selected token locations.
* A large component-specific effect shows that the selected visual/object write
  is jointly necessary.
* A positive role-aligned K/V recovery is evidence that subject/reference source
  binding matters.
* These tests identify bundle-level causal nodes. They do not by themselves
  establish an exact producer -> receiver edge. Exact edge patching requires
  freezing the corrupt run and allowing only a specified upstream contribution
  to enter a specified downstream Q/K/V channel.

Default qwen-3b preset
----------------------
Producer:
    P_L19 = 19:13
    P_L21 = 21:1
    P_L22 = 22:14
    P_L23 = 23:1,23:5
    P_ALL = union

Receiver:
    R24       = 24:0,24:4,24:5,24:7
    R26_KVH0  = 26:0..7
    R27       = 27:0,27:3,27:4,27:5
    R28       = 28:13
    R_CORE    = R24 + R26_KVH0 + R27
    R_CORE28  = R_CORE + R28

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_coco_circuit_bundle_causal_v1.py \
  --model qwen-3b \
  --source-output-dir \
    output/spatial_storage_transport_utilization/coco/qwen-3b \
  --producer-output-dir \
    output/coco_producer_qk_ov/qwen-3b \
  --receiver-output-dir \
    output/coco_receiver_qkv/qwen-3b \
  --experiments \
    producer_ablate,receiver_ablate,combined_ablate,qkv_patch \
  --ablation-modes head_zero,component_ablate \
  --producer-bundles P_ALL \
  --receiver-bundles R24,R26_KVH0,R27,R_CORE \
  --combined-bundles P_ALL+R24,P_ALL+R26_KVH0,P_ALL+R27,P_ALL+R_CORE \
  --qkv-bundles R26_KVH0,R27,R_CORE \
  --qkv-channels q,k,v \
  --qkv-alignments identity,role \
  --qkv-conditions restore_on_swapped,corrupt_on_original \
  --include-singletons \
  --leave-one-out P_ALL,R_CORE \
  --topk 1,2,4,8,12,16 \
  --random-controls 3 \
  --causal-status both_correct \
  --causal-max-samples 100 \
  --object-state last \
  --trace-layer-chunk 4 \
  --device cuda:0 \
  --output-dir \
    output/coco_circuit_bundle_causal/qwen-3b \
  --overwrite

Outputs
-------
    config.json
    bundle_definitions.json
    bundle_causal.jsonl
    bundle_causal_summary.csv
    bundle_redundancy_summary.csv
    bundle_leave_one_out_summary.csv
    report.txt
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
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-circuit-bundle-causal-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")
EXPERIMENTS = (
    "producer_ablate",
    "receiver_ablate",
    "combined_ablate",
    "qkv_patch",
)
ABLATION_MODES = ("head_zero", "component_ablate")
QKV_CHANNELS = ("q", "k", "v")
QKV_ALIGNMENTS = ("identity", "role")
QKV_CONDITIONS = ("restore_on_swapped", "corrupt_on_original")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source-output-dir",
        required=True,
        help="Existing v3 storage/transport output directory.",
    )
    parser.add_argument(
        "--producer-output-dir",
        required=True,
        help="Existing producer QK/OV output directory.",
    )
    parser.add_argument(
        "--receiver-output-dir",
        required=True,
        help="Existing receiver scan/QKV output directory.",
    )
    parser.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="eager", choices=("eager",))
    parser.add_argument(
        "--object-state",
        choices=("last", "mean"),
        default="last",
    )

    parser.add_argument(
        "--experiments",
        default="producer_ablate,receiver_ablate,combined_ablate",
        help="Comma-separated subset of producer_ablate,receiver_ablate,"
             "combined_ablate,qkv_patch.",
    )
    parser.add_argument(
        "--ablation-modes",
        default="head_zero,component_ablate",
        help="Comma-separated subset of head_zero,component_ablate.",
    )

    parser.add_argument(
        "--bundle-json",
        default=None,
        help=(
            "Optional JSON overriding or extending the built-in preset. Schema: "
            '{"producer":{"NAME":["19:13"]},"receiver":{"NAME":["26:4"]}}'
        ),
    )
    parser.add_argument(
        "--producer-bundles",
        default="P_ALL",
        help="Comma-separated producer bundle names.",
    )
    parser.add_argument(
        "--receiver-bundles",
        default="R24,R26_KVH0,R27,R_CORE",
        help="Comma-separated receiver bundle names.",
    )
    parser.add_argument(
        "--combined-bundles",
        default="P_ALL+R24,P_ALL+R26_KVH0,P_ALL+R27,P_ALL+R_CORE",
        help="Comma-separated PRODUCER+RECEIVER bundle pairs.",
    )
    parser.add_argument(
        "--qkv-bundles",
        default="R26_KVH0,R27,R_CORE",
        help="Receiver bundles used for Q/K/V projection patching.",
    )
    parser.add_argument(
        "--qkv-channels",
        default="q,k,v",
        help="Comma-separated subset of q,k,v.",
    )
    parser.add_argument(
        "--qkv-alignments",
        default="identity,role",
        help="Comma-separated subset of identity,role. Q ignores this distinction.",
    )
    parser.add_argument(
        "--qkv-conditions",
        default="restore_on_swapped,corrupt_on_original",
        help="Comma-separated patch conditions.",
    )

    parser.add_argument(
        "--include-singletons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add one-head bundles for redundancy analysis.",
    )
    parser.add_argument(
        "--leave-one-out",
        default="",
        help="Comma-separated producer or receiver bundles for LOO variants.",
    )
    parser.add_argument(
        "--topk",
        default="",
        help=(
            "Optional cumulative receiver top-k bundles from receiver_head_scan.csv, "
            "for example 1,2,4,8,12,16."
        ),
    )
    parser.add_argument(
        "--random-controls",
        type=int,
        default=0,
        help="Layer-matched random receiver controls per top-k bundle.",
    )

    parser.add_argument("--causal-status", choices=STATUSES, default="both_correct")
    parser.add_argument(
        "--causal-max-samples",
        type=int,
        default=100,
        help="0 or negative means all eligible samples.",
    )
    parser.add_argument("--min-margin-denominator", type=float, default=1e-4)
    parser.add_argument(
        "--causal-require-margin-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trace-layer-chunk", type=int, default=4)
    parser.add_argument("--replay-tolerance", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--empty-cache-every", type=int, default=10)

    parser.add_argument(
        "--producer-script",
        default="analyze_coco_producer_qk_ov_v1.py",
    )
    parser.add_argument(
        "--receiver-script",
        default="analyze_coco_receiver_qkv_v1.py",
    )
    parser.add_argument(
        "--v3-script",
        default="analyze_spatial_storage_transport_utilization_v3.py",
    )
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    parser.add_argument(
        "--attention-helper",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
    )

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


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
                fields.append(str(key))
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_mean(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def safe_std(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if array.size >= 2 else float("nan")


def parse_subset(value: str, allowed: Sequence[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    result: List[str] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(
                f"Unsupported {label}={item}; allowed={sorted(allowed_set)}"
            )
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError(f"No {label} selected")
    return result


def parse_name_list(value: str) -> List[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )
    )


def parse_int_list(value: str) -> List[int]:
    result: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return list(dict.fromkeys(result))


def parse_head(text: str) -> Tuple[int, int]:
    layer_text, head_text = str(text).strip().split(":", 1)
    return int(layer_text), int(head_text)


def head_text(head: Tuple[int, int]) -> str:
    return f"{int(head[0])}:{int(head[1])}"


def unique_heads(heads: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return list(dict.fromkeys((int(layer), int(head)) for layer, head in heads))


def relation_margin(logits: Sequence[float], gt: str) -> float:
    values = np.asarray(logits, dtype=np.float64)
    return float(values[REL_TO_ID[gt]] - values[REL_TO_ID[OPPOSITE[gt]]])


def scores_to_logits(scores: Mapping[str, float]) -> List[float]:
    return [float(scores[relation]) for relation in RELATIONS]


def status_matches(row: Mapping[str, Any], status: str) -> bool:
    return status == "all" or str(row["generation_pair_status"]) == status


def stratified_limit(
    rows: Sequence[Mapping[str, Any]],
    limit: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if limit is None or limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda item: int(item["sid"]))
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)
    for values in groups.values():
        rng.shuffle(values)
    labels = [relation for relation in RELATIONS if relation in groups]
    selected: List[Dict[str, Any]] = []
    cursors = {label: 0 for label in labels}
    while len(selected) < limit:
        progressed = False
        for label in labels:
            cursor = cursors[label]
            if cursor < len(groups[label]) and len(selected) < limit:
                selected.append(groups[label][cursor])
                cursors[label] += 1
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda item: int(item["sid"]))


# -----------------------------------------------------------------------------
# Bundle definitions
# -----------------------------------------------------------------------------


def default_bundle_config(model: str) -> Dict[str, Dict[str, List[str]]]:
    # The current preset is intentionally explicit. For another model, pass
    # --bundle-json rather than silently reusing qwen-3b heads.
    if model != "qwen-3b":
        return {"producer": {}, "receiver": {}}

    producer = {
        "P_L19": ["19:13"],
        "P_L21": ["21:1"],
        "P_L22": ["22:14"],
        "P_L23": ["23:1", "23:5"],
        "P_ALL": ["19:13", "21:1", "22:14", "23:1", "23:5"],
    }
    receiver = {
        "R24": ["24:0", "24:4", "24:5", "24:7"],
        "R26_KVH0": [f"26:{head}" for head in range(8)],
        "R27": ["27:0", "27:3", "27:4", "27:5"],
        "R28": ["28:13"],
    }
    receiver["R_CORE"] = (
        receiver["R24"] + receiver["R26_KVH0"] + receiver["R27"]
    )
    receiver["R_CORE28"] = receiver["R_CORE"] + receiver["R28"]
    return {"producer": producer, "receiver": receiver}


def load_bundle_config(args: argparse.Namespace) -> Dict[str, Dict[str, List[str]]]:
    config = default_bundle_config(args.model)
    if args.bundle_json:
        external = json.loads(Path(args.bundle_json).read_text(encoding="utf-8"))
        for family in ("producer", "receiver"):
            values = external.get(family, {})
            if not isinstance(values, dict):
                raise ValueError(f"{family} bundle section must be an object")
            config.setdefault(family, {})
            for name, heads in values.items():
                if not isinstance(heads, list):
                    raise ValueError(f"Bundle {family}.{name} must be a list")
                config[family][str(name)] = [str(item) for item in heads]

    parsed: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        "producer": {},
        "receiver": {},
    }
    for family in ("producer", "receiver"):
        for name, values in config.get(family, {}).items():
            parsed[family][name] = unique_heads(parse_head(item) for item in values)
    return parsed  # type: ignore[return-value]


def require_bundle_names(
    bundles: Mapping[str, Sequence[Tuple[int, int]]],
    names: Sequence[str],
    family: str,
) -> None:
    missing = [name for name in names if name not in bundles]
    if missing:
        raise KeyError(
            f"Unknown {family} bundles {missing}; available={sorted(bundles)}"
        )


def add_singletons(
    bundles: MutableMapping[str, List[Tuple[int, int]]],
    selected_names: Sequence[str],
    prefix: str,
) -> None:
    heads = unique_heads(
        head
        for name in selected_names
        for head in bundles[name]
    )
    for layer, head in heads:
        bundles.setdefault(
            f"{prefix}_SINGLE_L{layer}H{head}",
            [(layer, head)],
        )


def add_leave_one_out(
    bundles: MutableMapping[str, List[Tuple[int, int]]],
    selected_names: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    for name in selected_names:
        if name not in bundles:
            continue
        heads = list(bundles[name])
        if len(heads) <= 1:
            continue
        for removed in heads:
            variant = f"{name}__MINUS_L{removed[0]}H{removed[1]}"
            bundles[variant] = [head for head in heads if head != removed]
            metadata[variant] = {
                "parent": name,
                "removed_head": head_text(removed),
            }
    return metadata


def load_receiver_scan(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["rank"]))
    return rows


def add_topk_and_controls(
    *,
    receiver_bundles: MutableMapping[str, List[Tuple[int, int]]],
    scan_rows: Sequence[Mapping[str, Any]],
    topk_values: Sequence[int],
    random_controls: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    ranked = [
        (int(row["layer"]), int(row["head"]))
        for row in scan_rows
    ]
    all_by_layer: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for head in ranked:
        all_by_layer[head[0]].append(head)

    rng = random.Random(seed)
    for k in topk_values:
        if k <= 0:
            continue
        selected = unique_heads(ranked[:k])
        if len(selected) < k:
            raise RuntimeError(f"Only {len(selected)} receiver heads available for top-{k}")
        name = f"TOPK{k}"
        receiver_bundles[name] = selected
        metadata[name] = {
            "kind": "topk",
            "k": k,
            "layer_histogram": dict(Counter(layer for layer, _ in selected)),
        }

        if random_controls <= 0:
            continue
        layer_hist = Counter(layer for layer, _ in selected)
        selected_set = set(selected)
        for control_index in range(random_controls):
            control: List[Tuple[int, int]] = []
            for layer, count in sorted(layer_hist.items()):
                pool = [
                    head
                    for head in all_by_layer[layer]
                    if head not in selected_set
                ]
                if len(pool) < count:
                    raise RuntimeError(
                        f"Not enough non-top heads in L{layer} for a matched "
                        f"top-{k} random control: need={count}, available={len(pool)}"
                    )
                control.extend(rng.sample(pool, count))
            control_name = f"RANDK{k}_R{control_index + 1}"
            receiver_bundles[control_name] = unique_heads(control)
            metadata[control_name] = {
                "kind": "random_control",
                "matched_to": name,
                "k": k,
                "layer_histogram": dict(layer_hist),
            }
    return metadata


@dataclass(frozen=True)
class AblationUnit:
    experiment: str
    family: str
    bundle: str
    producer_bundle: Optional[str]
    receiver_bundle: Optional[str]
    producer_heads: Tuple[Tuple[int, int], ...]
    receiver_heads: Tuple[Tuple[int, int], ...]
    ablation_mode: str


@dataclass(frozen=True)
class QKVPatchUnit:
    experiment: str
    bundle: str
    heads: Tuple[Tuple[int, int], ...]
    channel: str
    alignment: str
    condition: str


def build_ablation_units(
    *,
    experiments: Sequence[str],
    ablation_modes: Sequence[str],
    producer_bundles: Mapping[str, Sequence[Tuple[int, int]]],
    receiver_bundles: Mapping[str, Sequence[Tuple[int, int]]],
    selected_producer: Sequence[str],
    selected_receiver: Sequence[str],
    selected_combined: Sequence[str],
) -> List[AblationUnit]:
    units: List[AblationUnit] = []

    if "producer_ablate" in experiments:
        for name in selected_producer:
            for mode in ablation_modes:
                units.append(
                    AblationUnit(
                        experiment="producer_ablate",
                        family="producer",
                        bundle=name,
                        producer_bundle=name,
                        receiver_bundle=None,
                        producer_heads=tuple(producer_bundles[name]),
                        receiver_heads=tuple(),
                        ablation_mode=mode,
                    )
                )

    if "receiver_ablate" in experiments:
        for name in selected_receiver:
            for mode in ablation_modes:
                units.append(
                    AblationUnit(
                        experiment="receiver_ablate",
                        family="receiver",
                        bundle=name,
                        producer_bundle=None,
                        receiver_bundle=name,
                        producer_heads=tuple(),
                        receiver_heads=tuple(receiver_bundles[name]),
                        ablation_mode=mode,
                    )
                )

    if "combined_ablate" in experiments:
        for pair in selected_combined:
            if "+" not in pair:
                raise ValueError(
                    f"Combined bundle must be PRODUCER+RECEIVER, got {pair}"
                )
            producer_name, receiver_name = pair.split("+", 1)
            require_bundle_names(producer_bundles, [producer_name], "producer")
            require_bundle_names(receiver_bundles, [receiver_name], "receiver")
            for mode in ablation_modes:
                units.append(
                    AblationUnit(
                        experiment="combined_ablate",
                        family="combined",
                        bundle=pair,
                        producer_bundle=producer_name,
                        receiver_bundle=receiver_name,
                        producer_heads=tuple(producer_bundles[producer_name]),
                        receiver_heads=tuple(receiver_bundles[receiver_name]),
                        ablation_mode=mode,
                    )
                )
    return units


def build_qkv_units(
    *,
    experiments: Sequence[str],
    receiver_bundles: Mapping[str, Sequence[Tuple[int, int]]],
    selected_bundles: Sequence[str],
    channels: Sequence[str],
    alignments: Sequence[str],
    conditions: Sequence[str],
) -> List[QKVPatchUnit]:
    if "qkv_patch" not in experiments:
        return []
    units: List[QKVPatchUnit] = []
    for bundle in selected_bundles:
        for channel in channels:
            effective_alignments = ["identity"] if channel == "q" else list(alignments)
            for alignment in effective_alignments:
                for condition in conditions:
                    units.append(
                        QKVPatchUnit(
                            experiment="qkv_patch",
                            bundle=bundle,
                            heads=tuple(receiver_bundles[bundle]),
                            channel=channel,
                            alignment=alignment,
                            condition=condition,
                        )
                    )
    return units


# -----------------------------------------------------------------------------
# Dynamic whole-head output zeroing
# -----------------------------------------------------------------------------


def output_projection_module(attention: Any) -> torch.nn.Module:
    for name in (
        "o_proj",
        "out_proj",
        "dense",
        "proj",
        "c_proj",
    ):
        module = getattr(attention, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise AttributeError(
        f"Unable to locate attention output projection on {type(attention).__name__}"
    )


class HeadOutputZeroPatch:
    """Zero selected query-head slices in the input to W_O."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        heads: Sequence[int],
        head_dim: int,
        positions: Sequence[int],
    ) -> None:
        self.heads = sorted(set(map(int, heads)))
        self.head_dim = int(head_dim)
        self.positions = sorted(set(map(int, positions)))
        self.applied = False
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not inputs:
            raise RuntimeError("W_O pre-hook received no input")
        hidden = inputs[0]
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError(
                f"W_O input must be [B,S,D], got {type(hidden).__name__}"
            )
        if int(hidden.shape[0]) != 1:
            raise RuntimeError("Head output zeroing expects batch size 1")

        modified = hidden.clone()
        applied = 0
        for position in self.positions:
            if not 0 <= position < int(hidden.shape[1]):
                raise RuntimeError(
                    f"Position {position} outside sequence length {int(hidden.shape[1])}"
                )
            for head in self.heads:
                start = head * self.head_dim
                stop = start + self.head_dim
                if stop > int(hidden.shape[-1]):
                    raise RuntimeError(
                        f"Head slice {start}:{stop} exceeds W_O input "
                        f"dimension {int(hidden.shape[-1])}"
                    )
                modified[0, position, start:stop] = 0
                applied += 1

        self.applied = applied > 0
        return (modified, *inputs[1:])

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


def group_heads_by_layer(
    heads: Sequence[Tuple[int, int]],
) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    return {
        layer: sorted(set(values))
        for layer, values in grouped.items()
    }


def install_head_zero_patches(
    *,
    producer_heads: Sequence[Tuple[int, int]],
    receiver_heads: Sequence[Tuple[int, int]],
    pair: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
) -> List[HeadOutputZeroPatch]:
    patches: List[HeadOutputZeroPatch] = []

    for family, heads, positions in (
        ("producer", producer_heads, pair.original_object_positions),
        ("receiver", receiver_heads, [pair.original_prompt_last]),
    ):
        for layer, layer_heads in group_heads_by_layer(heads).items():
            attention = attention_helper.resolve_self_attention(
                decoder_layers[int(layer)]
            )
            shape = receiver_module.resolve_attention_shape(attention)
            for head in layer_heads:
                if not 0 <= head < shape.n_query_heads:
                    raise ValueError(
                        f"{family} L{layer}H{head} outside "
                        f"0..{shape.n_query_heads - 1}"
                    )
            patches.append(
                HeadOutputZeroPatch(
                    module=output_projection_module(attention),
                    heads=layer_heads,
                    head_dim=shape.query_head_dim,
                    positions=positions,
                )
            )
    return patches


# -----------------------------------------------------------------------------
# Component-specific bundle ablation
# -----------------------------------------------------------------------------


def add_tensor_delta(
    store: MutableMapping[int, MutableMapping[int, torch.Tensor]],
    layer: int,
    position: int,
    delta: torch.Tensor,
) -> None:
    layer_map = store.setdefault(int(layer), {})
    if int(position) in layer_map:
        layer_map[int(position)] = layer_map[int(position)] + delta
    else:
        layer_map[int(position)] = delta


def producer_component_deltas(
    *,
    heads: Sequence[Tuple[int, int]],
    traces: Mapping[int, Any],
    object_positions: Sequence[int],
    visual_positions: Sequence[int],
    producer_module: Any,
) -> Dict[int, Dict[int, torch.Tensor]]:
    deltas: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer, layer_heads in group_heads_by_layer(heads).items():
        trace = traces[int(layer)]
        for position in object_positions:
            total: Optional[torch.Tensor] = None
            for head in layer_heads:
                pre = producer_module.visual_pre_vector(
                    trace=trace,
                    head=head,
                    target_position=int(position),
                    visual_positions=visual_positions,
                )
                write = producer_module.project_one_head(trace, head, pre)
                total = write if total is None else total + write
            if total is None:
                continue
            add_tensor_delta(deltas, layer, int(position), -total)
    return deltas


def receiver_component_deltas(
    *,
    heads: Sequence[Tuple[int, int]],
    traces: Mapping[int, Any],
    prompt_last: int,
    object_positions: Sequence[int],
    receiver_module: Any,
) -> Dict[int, Dict[int, torch.Tensor]]:
    deltas: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer, layer_heads in group_heads_by_layer(heads).items():
        trace = traces[int(layer)]
        total: Optional[torch.Tensor] = None
        for head in layer_heads:
            write, _mass = receiver_module.object_head_write(
                trace=trace,
                head=head,
                target_position=int(prompt_last),
                source_positions=object_positions,
            )
            total = write if total is None else total + write
        if total is not None:
            add_tensor_delta(deltas, layer, int(prompt_last), -total)
    return deltas


def merge_layer_deltas(
    *collections: Mapping[int, Mapping[int, torch.Tensor]],
) -> Dict[int, Dict[int, torch.Tensor]]:
    merged: Dict[int, Dict[int, torch.Tensor]] = {}
    for collection in collections:
        for layer, position_map in collection.items():
            for position, delta in position_map.items():
                add_tensor_delta(merged, int(layer), int(position), delta)
    return merged


def install_component_delta_patches(
    *,
    layer_deltas: Mapping[int, Mapping[int, torch.Tensor]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    producer_module: Any,
) -> List[Any]:
    patches = []
    for layer, position_map in sorted(layer_deltas.items()):
        attention = attention_helper.resolve_self_attention(
            decoder_layers[int(layer)]
        )
        numpy_map = {
            int(position): delta.detach().float().cpu().numpy()
            for position, delta in position_map.items()
        }
        patches.append(
            producer_module.AttentionTargetDelta(attention, numpy_map)
        )
    return patches


# -----------------------------------------------------------------------------
# Multi-head Q/K/V patching
# -----------------------------------------------------------------------------


def aligned_pairs(
    source_positions: Sequence[int],
    target_positions: Sequence[int],
    label: str,
) -> List[Tuple[int, int]]:
    source = list(map(int, source_positions))
    target = list(map(int, target_positions))
    if len(source) != len(target):
        raise RuntimeError(
            f"{label} token lengths differ: {len(source)} vs {len(target)}. "
            "Use --object-state last or provide compatible object spans."
        )
    return list(zip(source, target))


def projection_position_mapping(
    *,
    pair: Any,
    channel: str,
    alignment: str,
    condition: str,
    original_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    swapped_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    layer: int,
) -> Tuple[str, Dict[int, torch.Tensor]]:
    if channel == "q":
        if condition == "restore_on_swapped":
            return "swapped", {
                int(pair.swapped_prompt_last): original_states[layer]["q"][
                    int(pair.original_prompt_last)
                ]
            }
        return "original", {
            int(pair.original_prompt_last): swapped_states[layer]["q"][
                int(pair.swapped_prompt_last)
            ]
        }

    if alignment == "identity":
        pairs = (
            aligned_pairs(
                pair.original_a_positions,
                pair.swapped_a_positions,
                "identity A",
            )
            + aligned_pairs(
                pair.original_b_positions,
                pair.swapped_b_positions,
                "identity B",
            )
        )
    elif alignment == "role":
        pairs = (
            aligned_pairs(
                pair.original_a_positions,
                pair.swapped_b_positions,
                "role subject",
            )
            + aligned_pairs(
                pair.original_b_positions,
                pair.swapped_a_positions,
                "role reference",
            )
        )
    else:
        raise ValueError(alignment)

    mapping: Dict[int, torch.Tensor] = {}
    if condition == "restore_on_swapped":
        for original_position, swapped_target in pairs:
            mapping[int(swapped_target)] = original_states[layer][channel][
                int(original_position)
            ]
        return "swapped", mapping

    for original_target, swapped_source in pairs:
        mapping[int(original_target)] = swapped_states[layer][channel][
            int(swapped_source)
        ]
    return "original", mapping


def projection_patch_specs(
    *,
    heads: Sequence[Tuple[int, int]],
    channel: str,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    seen = set()
    for layer, query_head in heads:
        attention = attention_helper.resolve_self_attention(
            decoder_layers[int(layer)]
        )
        shape = receiver_module.resolve_attention_shape(attention)
        if not 0 <= int(query_head) < shape.n_query_heads:
            raise ValueError(
                f"L{layer} query head {query_head} outside "
                f"0..{shape.n_query_heads - 1}"
            )
        if channel == "q":
            unit_head = int(query_head)
            head_dim = int(shape.query_head_dim)
        else:
            unit_head = int(shape.kv_head_for_query(int(query_head)))
            head_dim = int(shape.kv_head_dim)
        key = (int(layer), channel, int(unit_head))
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "layer": int(layer),
                "channel": channel,
                "unit_head": int(unit_head),
                "query_head": int(query_head),
                "kv_head": int(shape.kv_head_for_query(int(query_head))),
                "shared_query_heads": (
                    [int(query_head)]
                    if channel == "q"
                    else shape.shared_query_heads(unit_head)
                ),
                "head_dim": head_dim,
            }
        )
    return specs


@torch.inference_mode()
def run_qkv_bundle_patch(
    *,
    unit: QKVPatchUnit,
    pair: Any,
    original_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    swapped_states: Mapping[int, Mapping[str, Mapping[int, torch.Tensor]]],
    model: Any,
    base: Any,
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
) -> Dict[str, Any]:
    specs = projection_patch_specs(
        heads=unit.heads,
        channel=unit.channel,
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
    )
    patches: List[Any] = []
    target_side: Optional[str] = None
    try:
        for spec in specs:
            layer = int(spec["layer"])
            side, mapping = projection_position_mapping(
                pair=pair,
                channel=unit.channel,
                alignment=unit.alignment,
                condition=unit.condition,
                original_states=original_states,
                swapped_states=swapped_states,
                layer=layer,
            )
            if target_side is None:
                target_side = side
            elif target_side != side:
                raise RuntimeError("Mixed target sides in one QKV bundle")
            attention = attention_helper.resolve_self_attention(
                decoder_layers[layer]
            )
            module = receiver_module.projection_module(attention, unit.channel)
            patches.append(
                receiver_module.ProjectionHeadPatch(
                    module=module,
                    head=int(spec["unit_head"]),
                    head_dim=int(spec["head_dim"]),
                    target_to_source=mapping,
                )
            )

        if target_side is None:
            raise RuntimeError("No QKV patch specs generated")
        batch = pair.swapped_batch if target_side == "swapped" else pair.original_batch
        result = receiver_module.run_scores(
            model=model,
            batch=batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        if not all(patch.applied for patch in patches):
            raise RuntimeError("At least one QKV bundle hook did not fire")
        result["target_side"] = target_side
        result["patch_units"] = specs
        return result
    finally:
        for patch in reversed(patches):
            patch.close()


# -----------------------------------------------------------------------------
# Model runs and per-sample evaluation
# -----------------------------------------------------------------------------


@torch.inference_mode()
def run_with_patches(
    *,
    model: Any,
    batch: Mapping[str, Any],
    base: Any,
    relation_token_map: Mapping[str, Sequence[int]],
    patches: Sequence[Any],
) -> Dict[str, Any]:
    try:
        outputs = model(
            **batch,
            use_cache=False,
            return_dict=True,
        )
        for patch in patches:
            if not getattr(patch, "applied", False):
                raise RuntimeError(
                    f"Intervention hook {type(patch).__name__} did not fire"
                )
        relation = base.relation_scores(
            outputs.logits[0, -1],
            dict(relation_token_map),
            gt=None,
        )
        logits = np.asarray(relation["logits"], dtype=np.float64)
        return {
            "logits": logits.tolist(),
            "prediction": str(relation["prediction"]),
        }
    finally:
        for patch in reversed(list(patches)):
            with contextlib.suppress(Exception):
                patch.close()
        with contextlib.suppress(Exception):
            del outputs


def normalized_result(
    *,
    original_margin: float,
    swapped_margin: float,
    intervention_margin: float,
    target_side: str,
) -> Tuple[float, float, bool, str]:
    denominator = float(original_margin - swapped_margin)
    if target_side == "original":
        raw = float(original_margin - intervention_margin)
        normalized = float(raw / denominator)
        crossed = bool(original_margin > 0 >= intervention_margin)
        metric = "normalized_damage"
    else:
        raw = float(intervention_margin - swapped_margin)
        normalized = float(raw / denominator)
        crossed = bool(swapped_margin < 0 <= intervention_margin)
        metric = "normalized_recovery"
    return raw, normalized, crossed, metric


def selected_trace_layers(
    ablation_units: Sequence[AblationUnit],
) -> List[int]:
    layers = set()
    for unit in ablation_units:
        if unit.ablation_mode != "component_ablate":
            continue
        layers.update(layer for layer, _ in unit.producer_heads)
        layers.update(layer for layer, _ in unit.receiver_heads)
    return sorted(layers)


def selected_projection_layers(
    qkv_units: Sequence[QKVPatchUnit],
) -> List[int]:
    return sorted(
        {
            int(layer)
            for unit in qkv_units
            for layer, _ in unit.heads
        }
    )


def evaluate_sample(
    *,
    args: argparse.Namespace,
    source_row: Mapping[str, Any],
    ablation_units: Sequence[AblationUnit],
    qkv_units: Sequence[QKVPatchUnit],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    base: Any,
    v3: Any,
    producer_module: Any,
    receiver_module: Any,
    attention_helper: Any,
    completed: set,
    output_path: Path,
) -> List[Dict[str, Any]]:
    device = torch.device(args.device)
    pair = receiver_module.prepare_pair(
        args=args,
        row=source_row,
        records_by_sid=records_by_sid,
        prompt_rows=prompt_rows,
        base=base,
        v3=v3,
        processor=processor,
        device=device,
    )
    sid = int(pair.sid)
    gt = str(pair.gt)
    rows: List[Dict[str, Any]] = []

    try:
        original_baseline = receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        swapped_baseline = receiver_module.run_scores(
            model=model,
            batch=pair.swapped_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        original_margin = relation_margin(original_baseline["logits"], gt)
        swapped_margin = relation_margin(swapped_baseline["logits"], gt)
        denominator = float(original_margin - swapped_margin)
        if abs(denominator) < args.min_margin_denominator:
            return rows
        if args.causal_require_margin_sign and not (
            original_margin > 0 and swapped_margin < 0
        ):
            return rows

        component_layers = selected_trace_layers(ablation_units)
        original_traces: Dict[int, Any] = {}
        max_replay_error = 0.0
        original_visual: List[int] = []

        if component_layers:
            original_targets = sorted(
                set(pair.original_object_positions + [pair.original_prompt_last])
            )
            original_visual = base.resolve_visual_indices(
                model,
                processor,
                dict(pair.original_batch),
                pair.original_ids,
            )
            traced_baseline, original_traces = v3.trace_prompt_chunks(
                attention_helper=attention_helper,
                model=model,
                batch=pair.original_batch,
                relation_token_map=relation_token_map,
                decoder_layers=decoder_layers,
                layers=component_layers,
                target_positions=original_targets,
                chunk_size=args.trace_layer_chunk,
            )
            traced_margin = relation_margin(
                [traced_baseline["scores"][relation] for relation in RELATIONS],
                gt,
            )
            if abs(traced_margin - original_margin) > 1e-3:
                raise RuntimeError(
                    f"Trace baseline margin mismatch: normal={original_margin}, "
                    f"trace={traced_margin}"
                )
            max_replay_error = max(
                float(original_traces[layer].replay_relative_error)
                for layer in component_layers
            )

        projection_layers = selected_projection_layers(qkv_units)
        original_states = None
        swapped_states = None
        if projection_layers:
            (
                projection_original,
                projection_swapped,
                original_states,
                swapped_states,
            ) = receiver_module.capture_pair_projections(
                pair=pair,
                layers=projection_layers,
                model=model,
                base=base,
                relation_token_map=relation_token_map,
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
            )
            projection_original_margin = relation_margin(
                projection_original["logits"], gt
            )
            projection_swapped_margin = relation_margin(
                projection_swapped["logits"], gt
            )
            if abs(projection_original_margin - original_margin) > 1e-3:
                raise RuntimeError("Projection capture changed original margin")
            if abs(projection_swapped_margin - swapped_margin) > 1e-3:
                raise RuntimeError("Projection capture changed swapped margin")

        for unit in ablation_units:
            key = (
                sid,
                unit.experiment,
                unit.bundle,
                unit.ablation_mode,
                "ablate_on_original",
                "",
                "",
            )
            if key in completed:
                continue

            patches: List[Any]
            if unit.ablation_mode == "head_zero":
                patches = install_head_zero_patches(
                    producer_heads=unit.producer_heads,
                    receiver_heads=unit.receiver_heads,
                    pair=pair,
                    decoder_layers=decoder_layers,
                    attention_helper=attention_helper,
                    receiver_module=receiver_module,
                )
            elif unit.ablation_mode == "component_ablate":
                producer_deltas = producer_component_deltas(
                    heads=unit.producer_heads,
                    traces=original_traces,
                    object_positions=pair.original_object_positions,
                    visual_positions=original_visual,
                    producer_module=producer_module,
                )
                receiver_deltas = receiver_component_deltas(
                    heads=unit.receiver_heads,
                    traces=original_traces,
                    prompt_last=pair.original_prompt_last,
                    object_positions=pair.original_object_positions,
                    receiver_module=receiver_module,
                )
                patches = install_component_delta_patches(
                    layer_deltas=merge_layer_deltas(
                        producer_deltas,
                        receiver_deltas,
                    ),
                    decoder_layers=decoder_layers,
                    attention_helper=attention_helper,
                    producer_module=producer_module,
                )
            else:
                raise ValueError(unit.ablation_mode)

            intervention = run_with_patches(
                model=model,
                batch=pair.original_batch,
                base=base,
                relation_token_map=relation_token_map,
                patches=patches,
            )
            intervention_margin = relation_margin(intervention["logits"], gt)
            raw, normalized, crossed, metric = normalized_result(
                original_margin=original_margin,
                swapped_margin=swapped_margin,
                intervention_margin=intervention_margin,
                target_side="original",
            )
            row = {
                "script_version": SCRIPT_VERSION,
                "model": args.model,
                "sid": sid,
                "gt": gt,
                "generation_pair_status": source_row["generation_pair_status"],
                "experiment": unit.experiment,
                "family": unit.family,
                "bundle": unit.bundle,
                "producer_bundle": unit.producer_bundle,
                "receiver_bundle": unit.receiver_bundle,
                "ablation_mode": unit.ablation_mode,
                "condition": "ablate_on_original",
                "channel": "",
                "alignment": "",
                "target_side": "original",
                "producer_heads": [list(head) for head in unit.producer_heads],
                "receiver_heads": [list(head) for head in unit.receiver_heads],
                "n_producer_heads": len(unit.producer_heads),
                "n_receiver_heads": len(unit.receiver_heads),
                "original_margin": original_margin,
                "swapped_margin_fixed_axis": swapped_margin,
                "intervention_margin_fixed_axis": intervention_margin,
                "margin_denominator": denominator,
                "raw_effect": raw,
                "normalized_effect": normalized,
                "normalized_metric": metric,
                "expected_positive": bool(raw > 0),
                "crossed_decision_boundary": crossed,
                "original_prediction": original_baseline["prediction"],
                "swapped_prediction": swapped_baseline["prediction"],
                "intervention_prediction": intervention["prediction"],
                "max_replay_relative_error": max_replay_error,
                "replay_within_tolerance": bool(
                    max_replay_error <= args.replay_tolerance
                ),
            }
            append_jsonl(output_path, row)
            rows.append(row)
            completed.add(key)

        if qkv_units:
            assert original_states is not None
            assert swapped_states is not None

        for unit in qkv_units:
            key = (
                sid,
                unit.experiment,
                unit.bundle,
                "",
                unit.condition,
                unit.channel,
                unit.alignment,
            )
            if key in completed:
                continue
            intervention = run_qkv_bundle_patch(
                unit=unit,
                pair=pair,
                original_states=original_states,
                swapped_states=swapped_states,
                model=model,
                base=base,
                relation_token_map=relation_token_map,
                decoder_layers=decoder_layers,
                attention_helper=attention_helper,
                receiver_module=receiver_module,
            )
            intervention_margin = relation_margin(intervention["logits"], gt)
            raw, normalized, crossed, metric = normalized_result(
                original_margin=original_margin,
                swapped_margin=swapped_margin,
                intervention_margin=intervention_margin,
                target_side=intervention["target_side"],
            )
            row = {
                "script_version": SCRIPT_VERSION,
                "model": args.model,
                "sid": sid,
                "gt": gt,
                "generation_pair_status": source_row["generation_pair_status"],
                "experiment": unit.experiment,
                "family": "receiver_qkv",
                "bundle": unit.bundle,
                "producer_bundle": None,
                "receiver_bundle": unit.bundle,
                "ablation_mode": "",
                "condition": unit.condition,
                "channel": unit.channel,
                "alignment": unit.alignment,
                "target_side": intervention["target_side"],
                "producer_heads": [],
                "receiver_heads": [list(head) for head in unit.heads],
                "n_producer_heads": 0,
                "n_receiver_heads": len(unit.heads),
                "qkv_patch_units": intervention["patch_units"],
                "original_margin": original_margin,
                "swapped_margin_fixed_axis": swapped_margin,
                "intervention_margin_fixed_axis": intervention_margin,
                "margin_denominator": denominator,
                "raw_effect": raw,
                "normalized_effect": normalized,
                "normalized_metric": metric,
                "expected_positive": bool(raw > 0),
                "crossed_decision_boundary": crossed,
                "original_prediction": original_baseline["prediction"],
                "swapped_prediction": swapped_baseline["prediction"],
                "intervention_prediction": intervention["prediction"],
                "max_replay_relative_error": 0.0,
                "replay_within_tolerance": True,
            }
            append_jsonl(output_path, row)
            rows.append(row)
            completed.add(key)

        return rows
    finally:
        receiver_module.release_pair(pair)


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summary_group_key(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return (
        str(row["experiment"]),
        str(row["family"]),
        str(row["bundle"]),
        str(row.get("ablation_mode", "")),
        str(row.get("condition", "")),
        str(row.get("channel", "")),
        str(row.get("alignment", "")),
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[summary_group_key(row)].append(row)

    summary: List[Dict[str, Any]] = []
    for key, values in groups.items():
        (
            experiment,
            family,
            bundle,
            ablation_mode,
            condition,
            channel,
            alignment,
        ) = key
        summary.append(
            {
                "experiment": experiment,
                "family": family,
                "bundle": bundle,
                "ablation_mode": ablation_mode,
                "condition": condition,
                "channel": channel,
                "alignment": alignment,
                "N": len(values),
                "n_producer_heads": int(values[0]["n_producer_heads"]),
                "n_receiver_heads": int(values[0]["n_receiver_heads"]),
                "mean_raw_effect": safe_mean(
                    value["raw_effect"] for value in values
                ),
                "median_raw_effect": safe_median(
                    value["raw_effect"] for value in values
                ),
                "std_raw_effect": safe_std(
                    value["raw_effect"] for value in values
                ),
                "mean_normalized_effect": safe_mean(
                    value["normalized_effect"] for value in values
                ),
                "median_normalized_effect": safe_median(
                    value["normalized_effect"] for value in values
                ),
                "std_normalized_effect": safe_std(
                    value["normalized_effect"] for value in values
                ),
                "positive_effect_rate": safe_mean(
                    int(bool(value["expected_positive"])) for value in values
                ),
                "crossed_decision_boundary_rate": safe_mean(
                    int(bool(value["crossed_decision_boundary"])) for value in values
                ),
                "replay_within_tolerance_rate": safe_mean(
                    int(bool(value["replay_within_tolerance"])) for value in values
                ),
            }
        )

    summary.sort(
        key=lambda row: (
            str(row["experiment"]),
            str(row["ablation_mode"]),
            str(row["channel"]),
            str(row["alignment"]),
            -float(row["mean_normalized_effect"]),
            str(row["bundle"]),
        )
    )
    return summary


def singleton_name(family: str, head: Tuple[int, int]) -> str:
    prefix = "P" if family == "producer" else "R"
    return f"{prefix}_SINGLE_L{head[0]}H{head[1]}"


def redundancy_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    producer_bundles: Mapping[str, Sequence[Tuple[int, int]]],
    receiver_bundles: Mapping[str, Sequence[Tuple[int, int]]],
) -> List[Dict[str, Any]]:
    # Only ablation rows are meaningful for singleton additivity.
    by_key_sid: Dict[Tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if str(row["experiment"]) not in {
            "producer_ablate",
            "receiver_ablate",
        }:
            continue
        key = (
            str(row["family"]),
            str(row["bundle"]),
            str(row["ablation_mode"]),
            int(row["sid"]),
        )
        by_key_sid[key] = row

    results: List[Dict[str, Any]] = []
    for family, bundles in (
        ("producer", producer_bundles),
        ("receiver", receiver_bundles),
    ):
        for bundle, heads in bundles.items():
            if len(heads) <= 1 or "SINGLE_" in bundle or "__MINUS_" in bundle:
                continue
            for mode in ABLATION_MODES:
                sample_rows = [
                    row
                    for (fam, name, row_mode, _sid), row in by_key_sid.items()
                    if fam == family and name == bundle and row_mode == mode
                ]
                interactions: List[float] = []
                ratios: List[float] = []
                bundle_effects: List[float] = []
                singleton_sums: List[float] = []
                for row in sample_rows:
                    sid = int(row["sid"])
                    singleton_values: List[float] = []
                    missing = False
                    for head in heads:
                        name = singleton_name(family, head)
                        singleton_row = by_key_sid.get(
                            (family, name, mode, sid)
                        )
                        if singleton_row is None:
                            missing = True
                            break
                        singleton_values.append(
                            float(singleton_row["normalized_effect"])
                        )
                    if missing:
                        continue
                    bundle_effect = float(row["normalized_effect"])
                    singleton_sum = float(sum(singleton_values))
                    interactions.append(bundle_effect - singleton_sum)
                    if abs(singleton_sum) > 1e-12:
                        ratios.append(bundle_effect / singleton_sum)
                    bundle_effects.append(bundle_effect)
                    singleton_sums.append(singleton_sum)
                if not bundle_effects:
                    continue
                results.append(
                    {
                        "family": family,
                        "bundle": bundle,
                        "ablation_mode": mode,
                        "N": len(bundle_effects),
                        "n_heads": len(heads),
                        "mean_bundle_effect": safe_mean(bundle_effects),
                        "mean_sum_singletons": safe_mean(singleton_sums),
                        "mean_bundle_minus_singletons": safe_mean(interactions),
                        "median_bundle_minus_singletons": safe_median(interactions),
                        "mean_bundle_over_singletons": safe_mean(ratios),
                    }
                )
    results.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["ablation_mode"]),
            -float(row["mean_bundle_effect"]),
        )
    )
    return results


def leave_one_out_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    loo_metadata: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    by_key_sid: Dict[Tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if str(row["experiment"]) not in {
            "producer_ablate",
            "receiver_ablate",
        }:
            continue
        by_key_sid[
            (
                str(row["family"]),
                str(row["bundle"]),
                str(row["ablation_mode"]),
                int(row["sid"]),
            )
        ] = row

    output: List[Dict[str, Any]] = []
    for variant, meta in loo_metadata.items():
        parent = str(meta["parent"])
        removed_head = str(meta["removed_head"])
        for family in ("producer", "receiver"):
            for mode in ABLATION_MODES:
                differences: List[float] = []
                full_values: List[float] = []
                loo_values: List[float] = []
                for (fam, name, row_mode, sid), loo_row in by_key_sid.items():
                    if fam != family or name != variant or row_mode != mode:
                        continue
                    full_row = by_key_sid.get((family, parent, mode, sid))
                    if full_row is None:
                        continue
                    full_effect = float(full_row["normalized_effect"])
                    loo_effect = float(loo_row["normalized_effect"])
                    differences.append(full_effect - loo_effect)
                    full_values.append(full_effect)
                    loo_values.append(loo_effect)
                if differences:
                    output.append(
                        {
                            "family": family,
                            "parent_bundle": parent,
                            "loo_bundle": variant,
                            "removed_head": removed_head,
                            "ablation_mode": mode,
                            "N": len(differences),
                            "mean_full_effect": safe_mean(full_values),
                            "mean_loo_effect": safe_mean(loo_values),
                            "mean_unique_contribution": safe_mean(differences),
                            "median_unique_contribution": safe_median(differences),
                            "positive_unique_rate": safe_mean(
                                int(value > 0) for value in differences
                            ),
                        }
                    )
    output.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["parent_bundle"]),
            str(row["ablation_mode"]),
            -float(row["mean_unique_contribution"]),
        )
    )
    return output


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    summary: Sequence[Mapping[str, Any]],
    redundancy: Sequence[Mapping[str, Any]],
    loo: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        f"script_version: {SCRIPT_VERSION}",
        f"model: {args.model}",
        f"dataset: {args.dataset}",
        f"causal_status: {args.causal_status}",
        f"causal_max_samples: {args.causal_max_samples}",
        "",
        "BUNDLE CAUSAL SUMMARY",
    ]
    for row in summary:
        lines.append(
            f"{row['experiment']} {row['bundle']} "
            f"mode={row['ablation_mode'] or '-'} "
            f"channel={row['channel'] or '-'} "
            f"alignment={row['alignment'] or '-'} "
            f"condition={row['condition']} "
            f"N={int(row['N'])} "
            f"effect={float(row['mean_normalized_effect']):+.6f} "
            f"median={float(row['median_normalized_effect']):+.6f} "
            f"positive={float(row['positive_effect_rate']):.4f} "
            f"crossed={float(row['crossed_decision_boundary_rate']):.4f}"
        )

    if redundancy:
        lines.extend(["", "REDUNDANCY / ADDITIVITY"])
        for row in redundancy:
            lines.append(
                f"{row['family']} {row['bundle']} "
                f"mode={row['ablation_mode']} "
                f"N={int(row['N'])} "
                f"bundle={float(row['mean_bundle_effect']):+.6f} "
                f"sumSingles={float(row['mean_sum_singletons']):+.6f} "
                f"interaction={float(row['mean_bundle_minus_singletons']):+.6f} "
                f"ratio={float(row['mean_bundle_over_singletons']):+.6f}"
            )

    if loo:
        lines.extend(["", "LEAVE-ONE-OUT"])
        for row in loo:
            lines.append(
                f"{row['family']} {row['parent_bundle']} "
                f"minus={row['removed_head']} "
                f"mode={row['ablation_mode']} "
                f"unique={float(row['mean_unique_contribution']):+.6f} "
                f"positive={float(row['positive_unique_rate']):.4f}"
            )

    lines.extend(
        [
            "",
            "Interpretation:",
            "  head_zero is a dynamic whole-head intervention before W_O.",
            "  component_ablate subtracts the clean baseline visual/object write.",
            "  bundle > sum(singletons) indicates positive interaction/synergy.",
            "  bundle < sum(singletons) indicates overlapping or redundant effects.",
            "  QKV role alignment swaps subject/reference source content rather than",
            "  matching the same object identity.",
            "  Bundle results do not establish an exact producer->receiver edge.",
        ]
    )
    (output_dir / "report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.trace_layer_chunk < 1:
        raise ValueError("--trace-layer-chunk must be >= 1")
    if args.random_controls < 0:
        raise ValueError("--random-controls must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    experiments = parse_subset(args.experiments, EXPERIMENTS, "experiments")
    ablation_modes = parse_subset(
        args.ablation_modes,
        ABLATION_MODES,
        "ablation modes",
    )
    qkv_channels = parse_subset(
        args.qkv_channels,
        QKV_CHANNELS,
        "QKV channels",
    )
    qkv_alignments = parse_subset(
        args.qkv_alignments,
        QKV_ALIGNMENTS,
        "QKV alignments",
    )
    qkv_conditions = parse_subset(
        args.qkv_conditions,
        QKV_CONDITIONS,
        "QKV conditions",
    )

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_output_dir)
    producer_dir = Path(args.producer_output_dir)
    receiver_dir = Path(args.receiver_output_dir)
    source_config_path = source_dir / "config.json"
    source_extraction_path = source_dir / "extraction.jsonl"
    receiver_scan_path = receiver_dir / "receiver_head_scan.csv"
    for path in (source_config_path, source_extraction_path):
        if not path.exists():
            raise FileNotFoundError(path)

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if str(source_config.get("model")) != args.model:
        raise RuntimeError(
            f"Source model={source_config.get('model')} but --model={args.model}"
        )
    if str(source_config.get("dataset")) != args.dataset:
        raise RuntimeError(
            f"Source dataset={source_config.get('dataset')} "
            f"but --dataset={args.dataset}"
        )
    source_object_state = str(source_config.get("object_state", args.object_state))
    if source_object_state != args.object_state:
        raise RuntimeError(
            f"Source object_state={source_object_state}, "
            f"but --object-state={args.object_state}"
        )

    source_rows = read_jsonl(source_extraction_path)
    if args.max_samples is not None:
        source_rows = source_rows[: int(args.max_samples)]
    eligible = [
        dict(row)
        for row in source_rows
        if status_matches(row, args.causal_status)
    ]
    filtered = []
    for row in eligible:
        original_cached = float(row["baseline_lm_margin"])
        swapped_cached = relation_margin(
            row["swapped_relation_logits"],
            str(row["gt"]),
        )
        if abs(original_cached - swapped_cached) < args.min_margin_denominator:
            continue
        if args.causal_require_margin_sign and not (
            original_cached > 0 and swapped_cached < 0
        ):
            continue
        filtered.append(row)
    eligible = stratified_limit(
        filtered,
        args.causal_max_samples,
        args.seed,
    )
    if not eligible:
        raise RuntimeError("No samples eligible for bundle causal analysis")

    producer_module = import_file(
        Path(args.producer_script),
        "bundle_causal_producer",
    )
    receiver_module = import_file(
        Path(args.receiver_script),
        "bundle_causal_receiver",
    )
    v3 = import_file(Path(args.v3_script), "bundle_causal_v3")
    base = import_file(Path(args.base_script), "bundle_causal_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "bundle_causal_attention",
    )

    bundle_config = load_bundle_config(args)
    producer_bundles: Dict[str, List[Tuple[int, int]]] = {
        name: list(heads)
        for name, heads in bundle_config["producer"].items()
    }
    receiver_bundles: Dict[str, List[Tuple[int, int]]] = {
        name: list(heads)
        for name, heads in bundle_config["receiver"].items()
    }

    selected_producer = parse_name_list(args.producer_bundles)
    selected_receiver = parse_name_list(args.receiver_bundles)
    selected_combined = parse_name_list(args.combined_bundles)
    selected_qkv = parse_name_list(args.qkv_bundles)
    loo_names = parse_name_list(args.leave_one_out)
    topk_values = parse_int_list(args.topk)

    require_bundle_names(producer_bundles, selected_producer, "producer")
    require_bundle_names(receiver_bundles, selected_receiver, "receiver")
    require_bundle_names(receiver_bundles, selected_qkv, "receiver")

    topk_metadata: Dict[str, Dict[str, Any]] = {}
    if topk_values:
        scan_rows = load_receiver_scan(receiver_scan_path)
        topk_metadata = add_topk_and_controls(
            receiver_bundles=receiver_bundles,
            scan_rows=scan_rows,
            topk_values=topk_values,
            random_controls=args.random_controls,
            seed=args.seed,
        )
        generated_names = [
            name
            for name in receiver_bundles
            if name.startswith("TOPK") or name.startswith("RANDK")
        ]
        selected_receiver = list(dict.fromkeys(selected_receiver + generated_names))

    if args.include_singletons:
        add_singletons(
            producer_bundles,
            selected_producer,
            "P",
        )
        add_singletons(
            receiver_bundles,
            selected_receiver,
            "R",
        )
        selected_producer = list(
            dict.fromkeys(
                selected_producer
                + [
                    name
                    for name in producer_bundles
                    if name.startswith("P_SINGLE_")
                ]
            )
        )
        selected_receiver = list(
            dict.fromkeys(
                selected_receiver
                + [
                    name
                    for name in receiver_bundles
                    if name.startswith("R_SINGLE_")
                ]
            )
        )

    producer_loo_names = [name for name in loo_names if name in producer_bundles]
    receiver_loo_names = [name for name in loo_names if name in receiver_bundles]
    unknown_loo = [
        name
        for name in loo_names
        if name not in producer_bundles and name not in receiver_bundles
    ]
    if unknown_loo:
        raise KeyError(f"Unknown leave-one-out bundles: {unknown_loo}")

    loo_metadata = {}
    loo_metadata.update(
        add_leave_one_out(producer_bundles, producer_loo_names)
    )
    loo_metadata.update(
        add_leave_one_out(receiver_bundles, receiver_loo_names)
    )
    selected_producer = list(
        dict.fromkeys(
            selected_producer
            + [
                name
                for name, meta in loo_metadata.items()
                if meta["parent"] in producer_loo_names
            ]
        )
    )
    selected_receiver = list(
        dict.fromkeys(
            selected_receiver
            + [
                name
                for name, meta in loo_metadata.items()
                if meta["parent"] in receiver_loo_names
            ]
        )
    )

    ablation_units = build_ablation_units(
        experiments=experiments,
        ablation_modes=ablation_modes,
        producer_bundles=producer_bundles,
        receiver_bundles=receiver_bundles,
        selected_producer=selected_producer,
        selected_receiver=selected_receiver,
        selected_combined=selected_combined,
    )
    qkv_units = build_qkv_units(
        experiments=experiments,
        receiver_bundles=receiver_bundles,
        selected_bundles=selected_qkv,
        channels=qkv_channels,
        alignments=qkv_alignments,
        conditions=qkv_conditions,
    )

    bundle_output = {
        "producer": {
            name: [head_text(head) for head in heads]
            for name, heads in producer_bundles.items()
        },
        "receiver": {
            name: [head_text(head) for head in heads]
            for name, heads in receiver_bundles.items()
        },
        "selected_producer": selected_producer,
        "selected_receiver": selected_receiver,
        "selected_combined": selected_combined,
        "selected_qkv": selected_qkv,
        "topk_metadata": topk_metadata,
        "leave_one_out_metadata": loo_metadata,
    }
    write_json(output_dir / "bundle_definitions.json", bundle_output)

    model = None
    processor = None
    output_path = output_dir / "bundle_causal.jsonl"
    errors_path = output_dir / "errors.jsonl"

    existing = read_jsonl(output_path) if args.resume else []
    completed = {
        (
            int(row["sid"]),
            str(row["experiment"]),
            str(row["bundle"]),
            str(row.get("ablation_mode", "")),
            str(row.get("condition", "")),
            str(row.get("channel", "")),
            str(row.get("alignment", "")),
        )
        for row in existing
    }
    all_rows = list(existing)

    try:
        (
            model,
            processor,
            spec,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer_module.load_model_bundle(args=args, base=base)

        all_heads = unique_heads(
            head
            for unit in ablation_units
            for head in tuple(unit.producer_heads) + tuple(unit.receiver_heads)
        )
        all_heads += unique_heads(
            head
            for unit in qkv_units
            for head in unit.heads
        )
        for layer, head in unique_heads(all_heads):
            if not 0 <= int(layer) < len(decoder_layers):
                raise ValueError(
                    f"Bundle head L{layer}H{head} outside decoder layers "
                    f"0..{len(decoder_layers) - 1}"
                )
            attention = attention_helper.resolve_self_attention(
                decoder_layers[int(layer)]
            )
            shape = receiver_module.resolve_attention_shape(attention)
            if not 0 <= int(head) < shape.n_query_heads:
                raise ValueError(
                    f"Bundle head L{layer}H{head} outside query heads "
                    f"0..{shape.n_query_heads - 1}"
                )

        two_object = base.import_two_object_module()
        records, audit = two_object.load_records(
            args.dataset,
            Path(args.data_root),
            args.max_samples,
        )
        records_by_sid = {int(record.sid): record for record in records}
        prompt_rows = base.load_standard_prompts(Path(args.prompt_jsonl))

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": str(source_dir),
            "producer_output_dir": str(producer_dir),
            "receiver_output_dir": str(receiver_dir),
            "source_script_version": source_config.get("script_version"),
            "decoder_path": decoder_path,
            "object_state": args.object_state,
            "experiments": experiments,
            "ablation_modes": ablation_modes,
            "qkv_channels": qkv_channels,
            "qkv_alignments": qkv_alignments,
            "qkv_conditions": qkv_conditions,
            "causal_status": args.causal_status,
            "causal_max_samples": args.causal_max_samples,
            "n_eligible": len(eligible),
            "include_singletons": args.include_singletons,
            "leave_one_out": loo_names,
            "topk": topk_values,
            "random_controls": args.random_controls,
            "trace_layer_chunk": args.trace_layer_chunk,
            "patch_locations": {
                "head_zero": "input_to_attention_WO",
                "component_ablate": "attention_output_residual_delta",
                "qkv_patch": "pre_rope_qkv_projection_output",
            },
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        print(
            f"Bundle causal: N={len(eligible)}, "
            f"ablation_units={len(ablation_units)}, "
            f"qkv_units={len(qkv_units)}",
            flush=True,
        )

        processed = 0
        for source_row in tqdm(
            eligible,
            desc=f"bundle-causal:{args.model}",
        ):
            sid = int(source_row["sid"])
            try:
                new_rows = evaluate_sample(
                    args=args,
                    source_row=source_row,
                    ablation_units=ablation_units,
                    qkv_units=qkv_units,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    producer_module=producer_module,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                    completed=completed,
                    output_path=output_path,
                )
                all_rows.extend(new_rows)
                processed += 1
                if args.print_every > 0 and processed % args.print_every == 0:
                    print(
                        f"[bundle {processed}/{len(eligible)}] sid={sid} "
                        f"new_rows={len(new_rows)}",
                        flush=True,
                    )
            except Exception as exc:
                error = {
                    "phase": "bundle_causal",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(errors_path, error)
                print(
                    f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available() and (
                    args.empty_cache_every > 0
                    and max(processed, 1) % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        # Stable rewrite removes duplicate rows left by interrupted runs.
        deduplicated: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for row in all_rows:
            key = (
                int(row["sid"]),
                str(row["experiment"]),
                str(row["bundle"]),
                str(row.get("ablation_mode", "")),
                str(row.get("condition", "")),
                str(row.get("channel", "")),
                str(row.get("alignment", "")),
            )
            deduplicated[key] = dict(row)
        all_rows = sorted(
            deduplicated.values(),
            key=lambda row: (
                str(row["experiment"]),
                str(row["bundle"]),
                str(row.get("ablation_mode", "")),
                str(row.get("channel", "")),
                str(row.get("alignment", "")),
                str(row.get("condition", "")),
                int(row["sid"]),
            ),
        )
        output_path.unlink(missing_ok=True)
        for row in all_rows:
            append_jsonl(output_path, row)

        summary = summarize_rows(all_rows)
        redundancy = redundancy_summary(
            rows=all_rows,
            producer_bundles=producer_bundles,
            receiver_bundles=receiver_bundles,
        )
        loo = leave_one_out_summary(
            rows=all_rows,
            loo_metadata=loo_metadata,
        )
        write_csv(output_dir / "bundle_causal_summary.csv", summary)
        write_csv(
            output_dir / "bundle_redundancy_summary.csv",
            redundancy,
        )
        write_csv(
            output_dir / "bundle_leave_one_out_summary.csv",
            loo,
        )
        write_report(
            output_dir=output_dir,
            args=args,
            summary=summary,
            redundancy=redundancy,
            loo=loo,
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
