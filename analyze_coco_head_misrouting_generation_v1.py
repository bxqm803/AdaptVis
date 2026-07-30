#!/usr/bin/env python3
"""
Sample-level head-misrouting analysis for the validated COCO/Qwen2.5-VL
spatial-relation circuit.

Question answered
-----------------
When free generation is wrong, is the error caused by one or more sender heads
supporting the wrong relation through the validated receiver path?

For every selected sample and every selected sender head h, this script:

  1. reproduces the normal free-generation answer;
  2. captures the clean L26 shared KV-head-0 Value state at object positions;
  3. removes h only at the sender object positions while freezing intermediate
     attention writes to their clean values;
  4. captures the resulting L26VH0 state;
  5. patches that state into the PREFILL V projection and runs full greedy
     autoregressive generation;
  6. compares clean and ablated four-relation logits and complete generated
     answers.

Thus the intervention is PATH-SPECIFIC:

    sender head h at object positions
        -> residual / token-wise MLP path
        -> L26 shared KV-head 0 Value channel
        -> autoregressive generation

It is not a whole-head ablation over every token and every downstream route.
That distinction is written into every output row.

Default scan
------------
The union of P_POS7 and P_NEG5 from coco_ioi_role_bundles_v1.json (12 heads).
You can instead scan explicit heads or all heads in selected layers.

Main outputs
------------
  baseline_generation.jsonl
      One normal free-generation row per SID.

  head_path_ablation_generation.jsonl
      One SID x sender-head row with free generation, logit effects, whether the
      ablation fixed/broke the sample, and whether the head favored the wrong
      generated relation over GT.

  head_summary.csv
      Per-head fixes, breaks, net fixes, free-generation accuracy, wrong-sample
      misleading scores, and exact paired McNemar p-values.

  head_status_summary.csv
      Per-head results split by source pair status (CC/CW/WC/WW aliases).

  sample_summary.csv
      Per-sample list/count of heads that fix the error or favor the wrong
      relation, plus a no-ablation-or-best-single-head oracle upper bound.

  summary.json, config.json, errors.jsonl

Expected companion scripts in repository root
---------------------------------------------
  analyze_coco_circuit_failure_repair_v1.py
  analyze_coco_circuit_generation_repair_grid_v1.py
  analyze_coco_ioi_backward_circuit_v1.py
  analyze_coco_producer_qk_ov_v1.py
  analyze_coco_receiver_qkv_v1.py
  analyze_spatial_storage_transport_utilization_v3.py
  analyze_coco_centroid_generation_step1_v4.py
  analyze_coco_flip_attention_spatial_vectors_v1.py
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


SCRIPT_VERSION = "coco-head-misrouting-generation-v1"
RELATIONS = ("left", "right", "above", "below")
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")
STATUS_ALIASES = {
    "both_correct": "CC",
    "original_only": "CW",
    "swapped_only": "WC",
    "both_wrong": "WW",
}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


# -----------------------------------------------------------------------------
# CLI and utilities
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
    p.add_argument(
        "--scan-bundles",
        default="P_POS7,P_NEG5",
        help="Union of these named bundles is scanned when --scan-heads and "
        "--scan-layers are empty.",
    )
    p.add_argument(
        "--scan-heads",
        default="",
        help="Explicit comma-separated heads, e.g. 19:8,19:13,22:9. "
        "Overrides --scan-bundles.",
    )
    p.add_argument(
        "--scan-layers",
        default="",
        help="Comma-separated layers whose every query head is scanned, e.g. "
        "19,20,21,22,23. Overrides bundles but not explicit --scan-heads.",
    )

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
    )
    p.add_argument("--baseline-warning-tolerance", type=float, default=0.03)

    p.add_argument(
        "--effect-epsilon",
        type=float,
        default=1e-6,
        help="Numerical threshold for classifying positive/negative head effects.",
    )
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=1)
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


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def parse_int_list(text: str) -> List[int]:
    result: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value not in result:
            result.append(value)
    return result


def parse_head_text(value: Any) -> Tuple[int, int]:
    text = str(value).strip()
    if text.startswith("L") and "H" in text:
        layer, head = text[1:].split("H", 1)
        return int(layer), int(head)
    if ":" in text:
        layer, head = text.split(":", 1)
        return int(layer), int(head)
    raise ValueError(f"Invalid head: {value!r}")


def head_name(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


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
    return aliases.get(text)


def relation_margin(scores: Mapping[str, float], target: Optional[str]) -> float:
    target = normalize_relation(target)
    if target is None:
        return float("nan")
    if target not in scores:
        return float("nan")
    others = [float(scores[r]) for r in RELATIONS if r != target]
    return float(scores[target]) - max(others)


def exact_two_sided_binomial_p(successes: int, failures: int) -> float:
    """Exact two-sided sign/McNemar test for discordant paired outcomes."""
    a = int(successes)
    b = int(failures)
    n = a + b
    if n <= 0:
        return 1.0
    k = min(a, b)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * probability))


@dataclass(frozen=True)
class ScanHead:
    layer: int
    head: int
    node: Any
    memberships: Tuple[str, ...]

    @property
    def name(self) -> str:
        return head_name(self.layer, self.head)

    @property
    def group(self) -> str:
        memberships = set(self.memberships)
        if "P_POS7" in memberships and "P_NEG5" in memberships:
            return "both"
        if "P_POS7" in memberships:
            return "positive"
        if "P_NEG5" in memberships:
            return "negative"
        return "other"


# -----------------------------------------------------------------------------
# Head selection
# -----------------------------------------------------------------------------


def load_bundle_payload(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("bundles", payload)
    if not isinstance(source, Mapping):
        raise ValueError("Bundle JSON must contain an object")
    result: Dict[str, List[Tuple[int, int]]] = {}
    for name, values in source.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Bundle {name} must be a list")
        parsed = [parse_head_text(value) for value in values]
        if len(parsed) != len(set(parsed)):
            raise ValueError(f"Duplicate heads in bundle {name}")
        result[str(name)] = parsed
    return result


def build_scan_heads(
    *,
    args: argparse.Namespace,
    bundle_payload: Mapping[str, Sequence[Tuple[int, int]]],
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    receiver_module: Any,
    ioi: Any,
) -> List[ScanHead]:
    memberships_by_head: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for bundle_name, values in bundle_payload.items():
        for pair in values:
            memberships_by_head[(int(pair[0]), int(pair[1]))].append(str(bundle_name))

    selected: List[Tuple[int, int]] = []
    explicit = [
        parse_head_text(item)
        for item in str(args.scan_heads).split(",")
        if item.strip()
    ]
    layers = parse_int_list(args.scan_layers)
    if explicit:
        selected = explicit
    elif layers:
        for layer_index in layers:
            if not 0 <= int(layer_index) < len(decoder_layers):
                raise ValueError(f"Layer {layer_index} outside decoder")
            attention = attention_helper.resolve_self_attention(
                decoder_layers[int(layer_index)]
            )
            shape = receiver_module.resolve_attention_shape(attention)
            selected.extend(
                (int(layer_index), int(head))
                for head in range(int(shape.n_query_heads))
            )
    else:
        bundle_names = [
            item.strip()
            for item in str(args.scan_bundles).split(",")
            if item.strip()
        ]
        if not bundle_names:
            raise ValueError("No scan heads, layers, or bundles supplied")
        for name in bundle_names:
            if name not in bundle_payload:
                raise KeyError(
                    f"Missing bundle {name}; available={list(bundle_payload)}"
                )
            selected.extend(bundle_payload[name])

    selected = sorted(set((int(layer), int(head)) for layer, head in selected))
    if not selected:
        raise RuntimeError("No heads selected")

    result: List[ScanHead] = []
    for layer, head in selected:
        if layer >= int(args.receiver_layer):
            raise ValueError(
                f"Sender {head_name(layer, head)} must be earlier than "
                f"receiver layer {args.receiver_layer}"
            )
        attention = attention_helper.resolve_self_attention(decoder_layers[layer])
        shape = receiver_module.resolve_attention_shape(attention)
        if not 0 <= head < int(shape.n_query_heads):
            raise ValueError(
                f"Head {head_name(layer, head)} outside 0..{int(shape.n_query_heads)-1}"
            )
        result.append(
            ScanHead(
                layer=layer,
                head=head,
                node=ioi.SenderNode("attention", layer, head),
                memberships=tuple(sorted(memberships_by_head.get((layer, head), ()))),
            )
        )
    return result


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
    selected_sids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selected = set(map(int, selected_sids))
    baseline_by_sid = {
        int(row["sid"]): row
        for row in baseline_rows
        if int(row["sid"]) in selected
    }
    heads_by_name: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in head_rows:
        if int(row["sid"]) in selected:
            heads_by_name[str(row["head"])].append(row)

    head_summary: List[Dict[str, Any]] = []
    status_summary: List[Dict[str, Any]] = []

    for name, values in sorted(heads_by_name.items()):
        by_sid = {int(row["sid"]): row for row in values}
        common = sorted(selected & set(baseline_by_sid) & set(by_sid))
        subset = [by_sid[sid] for sid in common]
        if not subset:
            continue
        baseline_correct = sum(int(bool(row["baseline_correct"])) for row in subset)
        ablated_correct = sum(int(bool(row["ablated_correct"])) for row in subset)
        fixes = sum(int(bool(row["fixed"])) for row in subset)
        breaks = sum(int(bool(row["broken"])) for row in subset)
        wrong_rows = [row for row in subset if not bool(row["baseline_correct"])]
        correct_rows = [row for row in subset if bool(row["baseline_correct"])]
        changed = sum(int(bool(row["generation_prediction_changed"])) for row in subset)
        membership = str(subset[0].get("head_group", "other"))
        memberships = str(subset[0].get("head_memberships", ""))

        row = {
            "head": name,
            "layer": int(subset[0]["head_layer"]),
            "head_index": int(subset[0]["head_index"]),
            "head_group": membership,
            "head_memberships": memberships,
            "N": len(subset),
            "parse_rate": safe_mean(int(bool(x["ablated_parsed"])) for x in subset),
            "baseline_accuracy": baseline_correct / len(subset),
            "ablated_accuracy": ablated_correct / len(subset),
            "accuracy_delta": (ablated_correct - baseline_correct) / len(subset),
            "fixes": fixes,
            "breaks": breaks,
            "net_fixes": fixes - breaks,
            "discordant": fixes + breaks,
            "fix_precision_among_changed_correctness": (
                fixes / max(fixes + breaks, 1)
            ),
            "mcnemar_exact_p": exact_two_sided_binomial_p(fixes, breaks),
            "prediction_change_rate": changed / len(subset),
            "wrong_N": len(wrong_rows),
            "fix_rate_among_baseline_wrong": fixes / max(len(wrong_rows), 1),
            "correct_N": len(correct_rows),
            "break_rate_among_baseline_correct": breaks / max(len(correct_rows), 1),
            "mean_E_GT_all": safe_mean(float(x["E_GT"]) for x in subset),
            "mean_E_GT_correct": safe_mean(float(x["E_GT"]) for x in correct_rows),
            "mean_E_GT_wrong": safe_mean(float(x["E_GT"]) for x in wrong_rows),
            "mean_E_pred_wrong": safe_mean(float(x["E_pred"]) for x in wrong_rows),
            "mean_wrong_over_gt_contribution_wrong": safe_mean(
                float(x["wrong_over_gt_contribution"]) for x in wrong_rows
            ),
            "misleading_rate_wrong": safe_mean(
                int(bool(x["misleading_for_generation_error"])) for x in wrong_rows
            ),
            "strict_misleading_rate_wrong": safe_mean(
                int(bool(x["strict_misleading_for_generation_error"]))
                for x in wrong_rows
            ),
            "preferred_generation_pred_rate_wrong": safe_mean(
                int(bool(x["preferred_relation_is_generation_pred"]))
                for x in wrong_rows
            ),
            "preferred_gt_rate_wrong": safe_mean(
                int(bool(x["preferred_relation_is_gt"])) for x in wrong_rows
            ),
            "preferred_gt_rate_correct": safe_mean(
                int(bool(x["preferred_relation_is_gt"])) for x in correct_rows
            ),
            "mean_receiver_delta_norm": safe_mean(
                float(x["receiver_delta_norm"]) for x in subset
            ),
            "mean_receiver_delta_ratio": safe_mean(
                float(x["receiver_delta_ratio"]) for x in subset
            ),
        }
        head_summary.append(row)

        statuses = sorted(set(str(x["source_generation_pair_status"]) for x in subset))
        for status in statuses:
            part = [x for x in subset if str(x["source_generation_pair_status"]) == status]
            fixes_part = sum(int(bool(x["fixed"])) for x in part)
            breaks_part = sum(int(bool(x["broken"])) for x in part)
            status_summary.append(
                {
                    "head": name,
                    "head_group": membership,
                    "source_generation_pair_status": status,
                    "status_alias": STATUS_ALIASES.get(status, status),
                    "N": len(part),
                    "baseline_accuracy": safe_mean(
                        int(bool(x["baseline_correct"])) for x in part
                    ),
                    "ablated_accuracy": safe_mean(
                        int(bool(x["ablated_correct"])) for x in part
                    ),
                    "fixes": fixes_part,
                    "breaks": breaks_part,
                    "net_fixes": fixes_part - breaks_part,
                    "mean_E_GT": safe_mean(float(x["E_GT"]) for x in part),
                    "mean_E_pred": safe_mean(float(x["E_pred"]) for x in part),
                    "mean_wrong_over_gt_contribution": safe_mean(
                        float(x["wrong_over_gt_contribution"]) for x in part
                    ),
                    "misleading_rate": safe_mean(
                        int(bool(x["misleading_for_generation_error"])) for x in part
                    ),
                }
            )

    sample_summary: List[Dict[str, Any]] = []
    rows_by_sid: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in head_rows:
        if int(row["sid"]) in selected:
            rows_by_sid[int(row["sid"])].append(row)

    for sid in sorted(selected & set(baseline_by_sid)):
        baseline = baseline_by_sid[sid]
        values = rows_by_sid.get(sid, [])
        fixed_heads = sorted(str(x["head"]) for x in values if bool(x["fixed"]))
        broken_heads = sorted(str(x["head"]) for x in values if bool(x["broken"]))
        misleading_heads = sorted(
            str(x["head"])
            for x in values
            if bool(x["misleading_for_generation_error"])
        )
        strict_misleading_heads = sorted(
            str(x["head"])
            for x in values
            if bool(x["strict_misleading_for_generation_error"])
        )
        ranked_misleading = sorted(
            values,
            key=lambda x: float(x["wrong_over_gt_contribution"]),
            reverse=True,
        )
        top = ranked_misleading[0] if ranked_misleading else None
        sample_summary.append(
            {
                "sid": sid,
                "gt": baseline.get("gt"),
                "source_generation_pair_status": baseline.get(
                    "source_generation_pair_status", "unknown"
                ),
                "status_alias": STATUS_ALIASES.get(
                    str(baseline.get("source_generation_pair_status", "")),
                    str(baseline.get("source_generation_pair_status", "unknown")),
                ),
                "baseline_generation_prediction": baseline.get("prediction"),
                "baseline_generation_correct": bool(baseline.get("correct")),
                "baseline_closed_prediction": baseline.get("closed_prediction"),
                "scanned_head_count": len(values),
                "any_single_head_fix": bool(fixed_heads),
                "single_head_fix_count": len(fixed_heads),
                "fixed_heads": ",".join(fixed_heads),
                "single_head_break_count": len(broken_heads),
                "broken_heads": ",".join(broken_heads),
                "misleading_head_count": len(misleading_heads),
                "misleading_heads": ",".join(misleading_heads),
                "strict_misleading_head_count": len(strict_misleading_heads),
                "strict_misleading_heads": ",".join(strict_misleading_heads),
                "most_wrong_favoring_head": None if top is None else top["head"],
                "max_wrong_over_gt_contribution": (
                    float("nan")
                    if top is None
                    else float(top["wrong_over_gt_contribution"])
                ),
                "oracle_correct_with_no_ablation_or_one_head": bool(
                    baseline.get("correct") or fixed_heads
                ),
            }
        )

    n = len(sample_summary)
    baseline_correct_n = sum(
        int(bool(row["baseline_generation_correct"])) for row in sample_summary
    )
    oracle_correct_n = sum(
        int(bool(row["oracle_correct_with_no_ablation_or_one_head"]))
        for row in sample_summary
    )
    wrong_samples = [row for row in sample_summary if not row["baseline_generation_correct"]]
    wrong_with_fix = sum(int(bool(row["any_single_head_fix"])) for row in wrong_samples)
    wrong_with_misleading = sum(
        int(int(row["misleading_head_count"]) > 0) for row in wrong_samples
    )
    wrong_with_strict = sum(
        int(int(row["strict_misleading_head_count"]) > 0) for row in wrong_samples
    )

    ranked_heads = sorted(
        head_summary,
        key=lambda row: (
            -int(row["net_fixes"]),
            -float(row["accuracy_delta"]),
            int(row["breaks"]),
            -int(row["fixes"]),
            str(row["head"]),
        ),
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "N": n,
        "baseline_parse_rate": safe_mean(
            int(bool(row.get("parsed"))) for row in baseline_by_sid.values()
        ),
        "baseline_accuracy": baseline_correct_n / n if n else float("nan"),
        "baseline_correct": baseline_correct_n,
        "baseline_wrong": n - baseline_correct_n,
        "scanned_head_count": len(head_summary),
        "wrong_samples_fixable_by_at_least_one_single_head": wrong_with_fix,
        "wrong_sample_single_head_fix_rate": (
            wrong_with_fix / max(len(wrong_samples), 1)
        ),
        "wrong_samples_with_at_least_one_wrong_favoring_head": wrong_with_misleading,
        "wrong_sample_wrong_favoring_head_rate": (
            wrong_with_misleading / max(len(wrong_samples), 1)
        ),
        "wrong_samples_with_at_least_one_strict_misleading_head": wrong_with_strict,
        "wrong_sample_strict_misleading_head_rate": (
            wrong_with_strict / max(len(wrong_samples), 1)
        ),
        "single_head_oracle_correct": oracle_correct_n,
        "single_head_oracle_accuracy": oracle_correct_n / n if n else float("nan"),
        "single_head_oracle_delta": (
            (oracle_correct_n - baseline_correct_n) / n if n else float("nan")
        ),
        "best_fixed_head": None if not ranked_heads else ranked_heads[0],
        "ablation_definition": (
            "path-specific sender removal at object positions; resulting "
            "L26VH0 V state patched into generation prefill"
        ),
    }
    return head_summary, status_summary, sample_summary, summary


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

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failure = import_file(Path(args.failure_script), "head_misrouting_failure")
    generation = import_file(Path(args.generation_helper), "head_misrouting_generation")
    ioi = import_file(Path(args.ioi_script), "head_misrouting_ioi")
    producer = import_file(Path(args.producer_script), "head_misrouting_producer")
    receiver = import_file(Path(args.receiver_script), "head_misrouting_receiver")
    v3 = import_file(Path(args.v3_script), "head_misrouting_v3")
    base = import_file(Path(args.base_script), "head_misrouting_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "head_misrouting_attention",
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
        if (
            args.sample_status == "all"
            or str(row.get("generation_pair_status")) == args.sample_status
        )
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

    bundle_payload = load_bundle_payload(Path(args.bundle_json))

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

        scan_heads = build_scan_heads(
            args=args,
            bundle_payload=bundle_payload,
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver,
            ioi=ioi,
        )

        receiver_layer = int(args.receiver_layer)
        writer = ioi.WriterNode(
            "attention",
            receiver_layer,
            int(args.receiver_query_head),
        )
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

        source_baseline_values = [source_original_correct(row) for row in selected_rows]
        source_baseline_known = [x for x in source_baseline_values if x is not None]
        source_baseline_accuracy = (
            safe_mean(int(x) for x in source_baseline_known)
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
            "scan_heads": [
                {
                    "head": item.name,
                    "layer": item.layer,
                    "head_index": item.head,
                    "memberships": list(item.memberships),
                    "group": item.group,
                }
                for item in scan_heads
            ],
            "scan_head_count": len(scan_heads),
            "receiver": {
                "unit": unit.unit,
                "layer": int(unit.layer),
                "kv_head": int(unit.kv_head),
                "unit_head": int(unit.unit_head),
                "shared_query_heads": list(unit.shared_query_heads),
                "channel": unit.channel,
                "kv_scope": args.receiver_kv_scope,
            },
            "ablation_definition": (
                "single sender-head output is zeroed only at object-token sender "
                "positions in a C-pass with intermediate attention frozen to "
                "clean; the resulting L26VH0 V state is patched into full-prompt "
                "prefill and enters the generation KV cache"
            ),
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.generation_do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
            },
            "sender_object_positions": args.sender_object_positions,
            "sample_status": args.sample_status,
            "selected_samples": len(selected_rows),
            "selected_sids": [int(row["sid"]) for row in selected_rows],
            "excluded_sids": sorted(excluded),
            "source_cached_original_generation_accuracy": source_baseline_accuracy,
            "effect_epsilon": args.effect_epsilon,
            "seed": args.seed,
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        baseline_path = output_dir / "baseline_generation.jsonl"
        effect_path = output_dir / "head_path_ablation_generation.jsonl"
        errors_path = output_dir / "errors.jsonl"

        baseline_rows = read_jsonl(baseline_path) if args.resume else []
        effect_rows = read_jsonl(effect_path) if args.resume else []
        baseline_done = {int(row["sid"]) for row in baseline_rows}
        effect_done = {
            (int(row["sid"]), str(row["head"])) for row in effect_rows
        }
        required_head_names = {item.name for item in scan_heads}

        pending = []
        for row in selected_rows:
            sid = int(row["sid"])
            if sid not in baseline_done or any(
                (sid, name) not in effect_done for name in required_head_names
            ):
                pending.append(row)

        print(
            "Head-misrouting free-generation scan: "
            f"requested_N={len(selected_rows)}, pending_N={len(pending)}, "
            f"heads={len(scan_heads)}, expected_effect_rows="
            f"{len(selected_rows) * len(scan_heads)}, "
            f"existing_baseline_rows={len(baseline_rows)}, "
            f"existing_effect_rows={len(effect_rows)}, "
            f"source_cached_baseline_acc={source_baseline_accuracy:.4f}",
            flush=True,
        )

        capture_layers = list(range(receiver_layer + 1))
        baseline_by_sid = {int(row["sid"]): row for row in baseline_rows}

        for sample_index, source_row in enumerate(
            tqdm(pending, desc=f"head-misrouting:{args.model}"),
            start=1,
        ):
            pair = None
            try:
                sid = int(source_row["sid"])
                missing_heads = [
                    item for item in scan_heads if (sid, item.name) not in effect_done
                ]
                need_baseline = sid not in baseline_done
                if not need_baseline and not missing_heads:
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
                attention_positions = sorted(
                    set(sender_positions) | set(receiver_positions)
                )

                clean_scores, clean_capture, baseline_states = (
                    failure.capture_clean_original(
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
                )
                baseline_v = baseline_states[int(unit.layer)][unit.channel]

                baseline_row = baseline_by_sid.get(sid)
                if baseline_row is None:
                    generated = generation.generate_answer(
                        model=model,
                        processor=processor,
                        batch=pair.original_batch,
                        args=args,
                    )
                    prediction = normalize_relation(generated["prediction"])
                    gt = normalize_relation(pair.gt)
                    if gt is None:
                        raise RuntimeError(f"Invalid GT relation: {pair.gt!r}")
                    baseline_row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": gt,
                        "generation": generated["text"],
                        "prediction": prediction,
                        "parsed": prediction is not None,
                        "correct": bool(prediction == gt),
                        "new_token_count": int(generated["new_token_count"]),
                        "generation_seconds": float(generated["generation_seconds"]),
                        "closed_prediction": normalize_relation(
                            clean_scores.get("prediction")
                        ),
                        "closed_scores": {
                            relation: float(clean_scores["scores"][relation])
                            for relation in RELATIONS
                        },
                        "closed_GT_margin": relation_margin(
                            clean_scores["scores"], gt
                        ),
                        "source_generation_pair_status": source_row.get(
                            "generation_pair_status", "unknown"
                        ),
                        "status_alias": STATUS_ALIASES.get(
                            str(source_row.get("generation_pair_status", "")),
                            str(source_row.get("generation_pair_status", "unknown")),
                        ),
                        "source_original_correct": source_original_correct(source_row),
                        "sender_positions": sender_positions,
                        "receiver_positions": receiver_positions,
                    }
                    append_jsonl(baseline_path, baseline_row)
                    baseline_rows.append(baseline_row)
                    baseline_by_sid[sid] = baseline_row
                    baseline_done.add(sid)

                gt = normalize_relation(baseline_row["gt"])
                baseline_generation_prediction = normalize_relation(
                    baseline_row.get("prediction")
                )
                baseline_closed_prediction = normalize_relation(
                    clean_scores.get("prediction")
                )
                pred_target = (
                    baseline_generation_prediction
                    if baseline_generation_prediction is not None
                    else baseline_closed_prediction
                )
                clean_score_map = {
                    relation: float(clean_scores["scores"][relation])
                    for relation in RELATIONS
                }

                for scan_head in missing_heads:
                    single_bundle = failure.HeadBundle(
                        name=scan_head.name,
                        heads=(scan_head.node,),
                    )
                    ablated_states = failure.run_bundle_removal_c_pass(
                        bundle=single_bundle,
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
                    ablated_v = ablated_states[int(unit.layer)][unit.channel]

                    ablated_scores = failure.run_receiver_state(
                        unit=unit,
                        full_states_by_position=ablated_v,
                        pair=pair,
                        model=model,
                        decoder_layers=decoder_layers,
                        relation_token_map=relation_token_map,
                        base=base,
                        receiver_module=receiver,
                        attention_helper=attention_helper,
                    )
                    ablated_score_map = {
                        relation: float(ablated_scores["scores"][relation])
                        for relation in RELATIONS
                    }

                    generated = generation.generate_answer(
                        model=model,
                        processor=processor,
                        batch=pair.original_batch,
                        args=args,
                        patch_module=projection,
                        patch_head=int(unit.unit_head),
                        patch_head_dim=patch_head_dim,
                        patch_states=ablated_v,
                    )
                    ablated_prediction = normalize_relation(generated["prediction"])
                    ablated_correct = bool(ablated_prediction == gt)
                    baseline_correct = bool(baseline_row["correct"])

                    contribution = {
                        relation: clean_score_map[relation] - ablated_score_map[relation]
                        for relation in RELATIONS
                    }
                    preferred_relation = max(
                        RELATIONS,
                        key=lambda relation: contribution[relation],
                    )
                    e_gt = relation_margin(clean_score_map, gt) - relation_margin(
                        ablated_score_map, gt
                    )
                    e_pred = relation_margin(clean_score_map, pred_target) - relation_margin(
                        ablated_score_map, pred_target
                    )
                    if pred_target is None or gt is None:
                        wrong_over_gt = float("nan")
                    else:
                        wrong_over_gt = (
                            contribution[pred_target] - contribution[gt]
                        )
                    eps = float(args.effect_epsilon)
                    baseline_wrong = not baseline_correct
                    misleading = bool(
                        baseline_wrong
                        and pred_target is not None
                        and pred_target != gt
                        and math.isfinite(wrong_over_gt)
                        and wrong_over_gt > eps
                    )
                    strict_misleading = bool(
                        misleading
                        and math.isfinite(e_pred)
                        and math.isfinite(e_gt)
                        and e_pred > eps
                        and e_gt < -eps
                    )
                    delta_norm, delta_ratio = failure.state_delta_norms(
                        baseline_v,
                        ablated_v,
                        unit,
                        decoder_layers,
                        attention_helper,
                        receiver,
                    )

                    row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": gt,
                        "head": scan_head.name,
                        "head_layer": int(scan_head.layer),
                        "head_index": int(scan_head.head),
                        "head_group": scan_head.group,
                        "head_memberships": ",".join(scan_head.memberships),
                        "baseline_generation": baseline_row.get("generation", ""),
                        "baseline_generation_prediction": baseline_generation_prediction,
                        "baseline_parsed": bool(baseline_row.get("parsed")),
                        "baseline_correct": baseline_correct,
                        "ablated_generation": generated["text"],
                        "ablated_generation_prediction": ablated_prediction,
                        "ablated_parsed": ablated_prediction is not None,
                        "ablated_correct": ablated_correct,
                        "fixed": bool((not baseline_correct) and ablated_correct),
                        "broken": bool(baseline_correct and (not ablated_correct)),
                        "generation_prediction_changed": bool(
                            ablated_prediction != baseline_generation_prediction
                        ),
                        "ablated_new_token_count": int(generated["new_token_count"]),
                        "ablated_generation_seconds": float(
                            generated["generation_seconds"]
                        ),
                        "baseline_closed_prediction": baseline_closed_prediction,
                        "ablated_closed_prediction": normalize_relation(
                            ablated_scores.get("prediction")
                        ),
                        "baseline_closed_scores": clean_score_map,
                        "ablated_closed_scores": ablated_score_map,
                        "head_logit_contribution": contribution,
                        "head_preferred_relation": preferred_relation,
                        "preferred_relation_is_gt": bool(preferred_relation == gt),
                        "preferred_relation_is_generation_pred": bool(
                            pred_target is not None
                            and preferred_relation == pred_target
                        ),
                        "prediction_target_for_effect": pred_target,
                        "E_GT": float(e_gt),
                        "E_pred": float(e_pred),
                        "wrong_over_gt_contribution": float(wrong_over_gt),
                        "misleading_for_generation_error": misleading,
                        "strict_misleading_for_generation_error": strict_misleading,
                        "receiver_delta_norm": float(delta_norm),
                        "receiver_delta_ratio": float(delta_ratio),
                        "source_generation_pair_status": source_row.get(
                            "generation_pair_status", "unknown"
                        ),
                        "status_alias": STATUS_ALIASES.get(
                            str(source_row.get("generation_pair_status", "")),
                            str(source_row.get("generation_pair_status", "unknown")),
                        ),
                        "sender_positions": sender_positions,
                        "receiver_positions": receiver_positions,
                        "receiver_unit": unit.unit,
                        "ablation_type": "path_specific_single_sender_to_L26VH0",
                        "cache_intervention": "prefill V projection patch only",
                    }
                    append_jsonl(effect_path, row)
                    effect_rows.append(row)
                    effect_done.add((sid, scan_head.name))

                if args.print_every > 0 and sample_index % args.print_every == 0:
                    baseline_correct_count = sum(
                        int(bool(row["correct"])) for row in baseline_rows
                    )
                    print(
                        f"[sample {sample_index}/{len(pending)} sid={sid}] "
                        f"baseline_rows={len(baseline_rows)} "
                        f"effect_rows={len(effect_rows)} "
                        f"baseline_acc_so_far="
                        f"{baseline_correct_count/max(len(baseline_rows),1):.4f}",
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
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    del pair
                gc.collect()
                if (
                    args.device.startswith("cuda")
                    and args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        selected_sids = [int(row["sid"]) for row in selected_rows]
        head_summary, status_summary, sample_summary, summary = summarize(
            baseline_rows=baseline_rows,
            head_rows=effect_rows,
            selected_sids=selected_sids,
        )
        write_csv(output_dir / "head_summary.csv", head_summary)
        write_csv(output_dir / "head_status_summary.csv", status_summary)
        write_csv(output_dir / "sample_summary.csv", sample_summary)

        current_baseline_rows = [
            row for row in baseline_rows if int(row["sid"]) in set(selected_sids)
        ]
        source_agreement_values = []
        for row in current_baseline_rows:
            source_value = row.get("source_original_correct")
            if source_value is not None:
                source_agreement_values.append(
                    int(bool(source_value) == bool(row["correct"]))
                )
        generated_baseline_accuracy = safe_mean(
            int(bool(row["correct"])) for row in current_baseline_rows
        )
        summary["source_cached_baseline_accuracy"] = source_baseline_accuracy
        summary["generated_vs_source_correctness_agreement"] = safe_mean(
            source_agreement_values
        )
        summary["generated_baseline_accuracy"] = generated_baseline_accuracy
        summary["complete_effect_rows"] = len(
            [
                row
                for row in effect_rows
                if int(row["sid"]) in set(selected_sids)
                and str(row["head"]) in required_head_names
            ]
        )
        summary["expected_effect_rows"] = len(selected_sids) * len(scan_heads)
        write_json(output_dir / "summary.json", summary)

        ranked = sorted(
            head_summary,
            key=lambda row: (
                -int(row["net_fixes"]),
                -float(row["accuracy_delta"]),
                int(row["breaks"]),
                -int(row["fixes"]),
            ),
        )

        print("\n" + "=" * 136)
        print("HEAD-MISROUTING FREE-GENERATION RESULT")
        print("=" * 136)
        print(
            f"Samples: {summary['N']} | heads: {summary['scanned_head_count']} | "
            f"effect rows: {summary['complete_effect_rows']}/"
            f"{summary['expected_effect_rows']}"
        )
        print(
            f"Baseline: parse={summary['baseline_parse_rate']:.4f} "
            f"accuracy={summary['baseline_accuracy']:.4f} | "
            f"source cached={source_baseline_accuracy:.4f} | "
            f"agreement={summary['generated_vs_source_correctness_agreement']:.4f}"
        )
        print(
            "Wrong samples fixable by at least one single-head path ablation: "
            f"{summary['wrong_samples_fixable_by_at_least_one_single_head']}/"
            f"{summary['baseline_wrong']} = "
            f"{summary['wrong_sample_single_head_fix_rate']:.4f}"
        )
        print(
            "Wrong samples with >=1 head favoring generated wrong relation over GT: "
            f"{summary['wrong_samples_with_at_least_one_wrong_favoring_head']}/"
            f"{summary['baseline_wrong']} = "
            f"{summary['wrong_sample_wrong_favoring_head_rate']:.4f}"
        )
        print(
            "Wrong samples with >=1 strict misleading head "
            "(E_pred>0 and E_GT<0): "
            f"{summary['wrong_samples_with_at_least_one_strict_misleading_head']}/"
            f"{summary['baseline_wrong']} = "
            f"{summary['wrong_sample_strict_misleading_head_rate']:.4f}"
        )
        print(
            "No-ablation-or-best-single-head oracle: "
            f"{summary['single_head_oracle_accuracy']:.4f} "
            f"delta={summary['single_head_oracle_delta']:+.4f}"
        )
        print("\nTOP HEADS BY NET FREE-GENERATION FIXES")
        print(
            f"{'head':>8} {'group':>9} {'acc':>8} {'delta':>8} "
            f"{'fix':>5} {'break':>6} {'net':>5} {'wrongMis':>10} "
            f"{'EgtWrong':>10} {'EpredWrong':>11} {'p':>10}"
        )
        for row in ranked[: min(20, len(ranked))]:
            print(
                f"{str(row['head']):>8} {str(row['head_group']):>9} "
                f"{float(row['ablated_accuracy']):8.4f} "
                f"{float(row['accuracy_delta']):+8.4f} "
                f"{int(row['fixes']):5d} {int(row['breaks']):6d} "
                f"{int(row['net_fixes']):5d} "
                f"{float(row['misleading_rate_wrong']):10.4f} "
                f"{float(row['mean_E_GT_wrong']):+10.4f} "
                f"{float(row['mean_E_pred_wrong']):+11.4f} "
                f"{float(row['mcnemar_exact_p']):10.4g}"
            )

        if math.isfinite(generated_baseline_accuracy) and abs(
            generated_baseline_accuracy - float(args.expected_source_baseline_accuracy)
        ) > float(args.baseline_warning_tolerance):
            print(
                "WARNING: regenerated baseline differs from expected source "
                f"accuracy {args.expected_source_baseline_accuracy:.4f} by more "
                f"than {args.baseline_warning_tolerance:.4f}.",
                file=sys.stderr,
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
