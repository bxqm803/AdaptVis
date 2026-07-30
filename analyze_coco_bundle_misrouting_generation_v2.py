#!/usr/bin/env python3
"""
Joint-bundle misrouting analysis for the validated COCO/Qwen2.5-VL spatial circuit.

This script follows the single-head experiment in
`analyze_coco_head_misrouting_generation_v1.py` and directly tests whether
free-generation errors are maintained by DISTRIBUTED sender bundles rather
than by one individually decisive head.

Two experiments are implemented.

1. Fixed negative-subset scan
   Exhaustively enumerate every non-empty subset of P_NEG5 (31 subsets for
   five heads).  For every COCO sample, jointly remove all sender heads in the
   subset only at the object-token positions, capture the resulting L26VH0
   Value state, patch that state into generation prefill, and run complete
   greedy autoregressive generation.

   This gives actual fixes, breaks, net fixes, McNemar p-values, and nonlinear
   subset synergy.  Single-head effects are never added to approximate the
   joint result; every subset is recomputed by a fresh causal forward pass.

2. Sample-adaptive misleading-head oracle
   Read the preceding single-head scan and, for each baseline-wrong sample,
   jointly remove the heads classified as:

       misleading:        contribution(predicted wrong) > contribution(GT)
       strict misleading: E_pred > 0 and E_GT < 0

   Top-1, top-2, top-3, and all qualifying heads can be tested.  Head selection
   uses GT and knowledge that the baseline answer is wrong, so these rows are
   explicitly labelled ANALYSIS ORACLE, not a deployable repair method.  The
   result answers the mechanistic question: if the sample-specific misleading
   sender set were known, would joint removal repair the answer?

Path-specific intervention
--------------------------

    selected sender heads at object positions
        -> residual / token-wise MLP path
        -> L26 shared KV-head 0 Value channel
        -> repaired generation prefill KV cache
        -> full autoregressive generation

The script does not zero whole heads at all tokens and does not patch final
relation logits.  It modifies the validated sender-to-L26VH0 path and evaluates
natural free generation with the same parser as the baseline pipeline.

Required preceding output
-------------------------

`--single-head-output-dir` must contain the completed outputs from
`analyze_coco_head_misrouting_generation_v1.py`:

    baseline_generation.jsonl
    head_path_ablation_generation.jsonl
    head_summary.csv

Main outputs
------------

    fixed_subset_generation.jsonl
    fixed_subset_summary.csv
    fixed_subset_size_summary.csv
    fixed_subset_status_summary.csv

    adaptive_oracle_generation.jsonl
    adaptive_oracle_summary.csv
    adaptive_oracle_status_summary.csv

    summary.json
    config.json
    errors.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import itertools
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


SCRIPT_VERSION = "coco-bundle-misrouting-generation-v2"
RELATIONS = ("left", "right", "above", "below")
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")
STATUS_ALIASES = {
    "both_correct": "CC",
    "original_only": "CW",
    "swapped_only": "WC",
    "both_wrong": "WW",
}


# -----------------------------------------------------------------------------
# CLI / basic I/O
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
    p.add_argument("--negative-bundle", default="P_NEG5")
    p.add_argument(
        "--oracle-candidate-bundles",
        default="P_POS7,P_NEG5",
        help="Union of heads eligible for sample-adaptive oracle bundles.",
    )
    p.add_argument(
        "--single-head-output-dir",
        required=True,
        help="Completed output directory from the v1 single-head scan.",
    )

    p.add_argument(
        "--experiments",
        default="fixed_subsets,adaptive_oracle",
        help="Comma-separated: fixed_subsets,adaptive_oracle.",
    )
    p.add_argument("--subset-min-size", type=int, default=1)
    p.add_argument(
        "--subset-max-size",
        type=int,
        default=0,
        help="0 means the full negative-bundle size.",
    )
    p.add_argument(
        "--oracle-kinds",
        default="strict,misleading",
        help="Comma-separated: strict,misleading.",
    )
    p.add_argument(
        "--oracle-k-values",
        default="1,2,3,all",
        help="Top-k qualifying heads to remove jointly for each oracle kind.",
    )
    p.add_argument(
        "--oracle-min-score",
        type=float,
        default=0.0,
        help="Minimum wrong_over_gt_contribution for an oracle-selected head.",
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
    p.add_argument("--exclude-sids-from", default="")
    p.add_argument("--include-sids-file", default="")

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
    p.add_argument(
        "--compute-closed-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run a closed four-relation scoring pass for every joint bundle. "
        "Disabled by default because it adds one full forward per row.",
    )

    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument(
        "--single-head-script",
        default="analyze_coco_head_misrouting_generation_v1.py",
    )
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

    # Compatibility with imported helper scripts.
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


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
    if target is None or target not in scores:
        return float("nan")
    return float(scores[target]) - max(float(scores[x]) for x in RELATIONS if x != target)


def exact_two_sided_binomial_p(successes: int, failures: int) -> float:
    a = int(successes)
    b = int(failures)
    n = a + b
    if n <= 0:
        return 1.0
    k = min(a, b)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * probability))


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


def parse_csv_tokens(text: str) -> List[str]:
    result: List[str] = []
    for item in str(text).split(","):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    return result


def parse_oracle_k_values(text: str) -> List[Optional[int]]:
    result: List[Optional[int]] = []
    for item in parse_csv_tokens(text):
        if item.lower() == "all":
            value: Optional[int] = None
        else:
            value = int(item)
            if value <= 0:
                raise ValueError("Oracle k values must be positive or 'all'")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("No oracle k values")
    return result


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
        if not parsed or len(parsed) != len(set(parsed)):
            raise ValueError(f"Invalid or duplicate heads in bundle {name}")
        result[str(name)] = parsed
    return result


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


# -----------------------------------------------------------------------------
# Bundle specifications
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedBundleSpec:
    bundle_id: str
    heads: Tuple[Tuple[int, int], ...]

    @property
    def head_names(self) -> Tuple[str, ...]:
        return tuple(head_name(layer, head) for layer, head in self.heads)

    @property
    def size(self) -> int:
        return len(self.heads)


@dataclass(frozen=True)
class OracleMode:
    mode: str
    kind: str
    k: Optional[int]


def build_fixed_subsets(
    heads: Sequence[Tuple[int, int]],
    min_size: int,
    max_size: int,
) -> List[FixedBundleSpec]:
    ordered = tuple((int(layer), int(head)) for layer, head in heads)
    if min_size <= 0:
        raise ValueError("--subset-min-size must be positive")
    if max_size <= 0:
        max_size = len(ordered)
    if min_size > max_size or max_size > len(ordered):
        raise ValueError(
            f"Invalid subset range [{min_size}, {max_size}] for {len(ordered)} heads"
        )
    result: List[FixedBundleSpec] = []
    for size in range(min_size, max_size + 1):
        for combination in itertools.combinations(ordered, size):
            names = tuple(head_name(layer, head) for layer, head in combination)
            result.append(
                FixedBundleSpec(
                    bundle_id=f"NEG{size}__" + "__".join(names),
                    heads=tuple(combination),
                )
            )
    return result


def build_oracle_modes(kinds: Sequence[str], k_values: Sequence[Optional[int]]) -> List[OracleMode]:
    result: List[OracleMode] = []
    for kind in kinds:
        if kind not in {"strict", "misleading"}:
            raise ValueError(f"Unknown oracle kind: {kind}")
        for k in k_values:
            suffix = "all" if k is None else f"top{k}"
            result.append(OracleMode(mode=f"{kind}_{suffix}", kind=kind, k=k))
    return result


def canonical_head_tuple(heads: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted(set((int(layer), int(head)) for layer, head in heads)))


def nodes_for_heads(heads: Sequence[Tuple[int, int]], ioi: Any) -> Tuple[Any, ...]:
    return tuple(
        ioi.SenderNode("attention", int(layer), int(head))
        for layer, head in heads
    )


# -----------------------------------------------------------------------------
# Single-head scan loading and oracle selection
# -----------------------------------------------------------------------------


def load_single_head_artifacts(
    directory: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline_path = directory / "baseline_generation.jsonl"
    effect_path = directory / "head_path_ablation_generation.jsonl"
    summary_path = directory / "head_summary.csv"
    for path in (baseline_path, effect_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing preceding single-head output: {path}"
            )
    return read_jsonl(baseline_path), read_jsonl(effect_path), read_csv(summary_path)


def select_oracle_heads(
    *,
    rows: Sequence[Mapping[str, Any]],
    mode: OracleMode,
    candidate_names: set[str],
    min_score: float,
) -> List[Tuple[int, int]]:
    flag = (
        "strict_misleading_for_generation_error"
        if mode.kind == "strict"
        else "misleading_for_generation_error"
    )
    eligible = [
        row
        for row in rows
        if str(row.get("head")) in candidate_names
        and bool(row.get(flag))
        and math.isfinite(float(row.get("wrong_over_gt_contribution", float("nan"))))
        and float(row.get("wrong_over_gt_contribution", float("nan"))) > float(min_score)
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["wrong_over_gt_contribution"]),
            str(row["head"]),
        )
    )
    if mode.k is not None:
        eligible = eligible[: int(mode.k)]
    return [parse_head_text(row["head"]) for row in eligible]


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize_fixed(
    *,
    rows: Sequence[Mapping[str, Any]],
    baseline_by_sid: Mapping[int, Mapping[str, Any]],
    selected_sids: Sequence[int],
    single_head_summary: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    selected = set(map(int, selected_sids))
    single_by_head = {str(row["head"]): row for row in single_head_summary}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["sid"]) in selected:
            grouped[str(row["bundle_id"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for bundle_id, values in sorted(grouped.items()):
        by_sid = {int(row["sid"]): row for row in values}
        common = sorted(selected & set(by_sid) & set(baseline_by_sid))
        subset = [by_sid[sid] for sid in common]
        if not subset:
            continue
        baseline_correct = sum(int(bool(row["baseline_correct"])) for row in subset)
        repaired_correct = sum(int(bool(row["joint_correct"])) for row in subset)
        fixes = sum(int(bool(row["fixed"])) for row in subset)
        breaks = sum(int(bool(row["broken"])) for row in subset)
        changed = sum(int(bool(row["generation_prediction_changed"])) for row in subset)
        head_names = tuple(str(subset[0]["head_names"]).split(","))
        head_names = tuple(x for x in head_names if x)
        additive_net = sum(
            int(float(single_by_head[name].get("net_fixes", 0)))
            for name in head_names
            if name in single_by_head
        )
        additive_delta = sum(
            float(single_by_head[name].get("accuracy_delta", 0.0))
            for name in head_names
            if name in single_by_head
        )
        row = {
            "bundle_id": bundle_id,
            "subset_size": int(subset[0]["subset_size"]),
            "head_names": ",".join(head_names),
            "N": len(subset),
            "parse_rate": safe_mean(int(bool(x["joint_parsed"])) for x in subset),
            "baseline_accuracy": baseline_correct / len(subset),
            "joint_accuracy": repaired_correct / len(subset),
            "accuracy_delta": (repaired_correct - baseline_correct) / len(subset),
            "fixes": fixes,
            "breaks": breaks,
            "net_fixes": fixes - breaks,
            "discordant": fixes + breaks,
            "fix_precision_among_changed_correctness": fixes / max(fixes + breaks, 1),
            "mcnemar_exact_p": exact_two_sided_binomial_p(fixes, breaks),
            "prediction_change_rate": changed / len(subset),
            "wrong_N": sum(int(not bool(x["baseline_correct"])) for x in subset),
            "fix_rate_among_baseline_wrong": fixes
            / max(sum(int(not bool(x["baseline_correct"])) for x in subset), 1),
            "correct_N": sum(int(bool(x["baseline_correct"])) for x in subset),
            "break_rate_among_baseline_correct": breaks
            / max(sum(int(bool(x["baseline_correct"])) for x in subset), 1),
            "sum_single_net_fixes": additive_net,
            "net_fix_synergy": (fixes - breaks) - additive_net,
            "sum_single_accuracy_delta": additive_delta,
            "accuracy_delta_synergy": (
                (repaired_correct - baseline_correct) / len(subset) - additive_delta
            ),
            "mean_receiver_delta_norm": safe_mean(
                float(x.get("receiver_delta_norm", float("nan"))) for x in subset
            ),
            "mean_receiver_delta_ratio": safe_mean(
                float(x.get("receiver_delta_ratio", float("nan"))) for x in subset
            ),
            "mean_E_GT_wrong": safe_mean(
                float(x.get("E_GT", float("nan")))
                for x in subset
                if not bool(x["baseline_correct"])
            ),
            "mean_E_pred_wrong": safe_mean(
                float(x.get("E_pred", float("nan")))
                for x in subset
                if not bool(x["baseline_correct"])
            ),
        }
        summary_rows.append(row)

        statuses = sorted(set(str(x["source_generation_pair_status"]) for x in subset))
        for status in statuses:
            part = [x for x in subset if str(x["source_generation_pair_status"]) == status]
            part_fixes = sum(int(bool(x["fixed"])) for x in part)
            part_breaks = sum(int(bool(x["broken"])) for x in part)
            status_rows.append(
                {
                    "bundle_id": bundle_id,
                    "subset_size": row["subset_size"],
                    "head_names": row["head_names"],
                    "source_generation_pair_status": status,
                    "status_alias": STATUS_ALIASES.get(status, status),
                    "N": len(part),
                    "baseline_accuracy": safe_mean(
                        int(bool(x["baseline_correct"])) for x in part
                    ),
                    "joint_accuracy": safe_mean(int(bool(x["joint_correct"])) for x in part),
                    "fixes": part_fixes,
                    "breaks": part_breaks,
                    "net_fixes": part_fixes - part_breaks,
                }
            )

    summary_rows.sort(
        key=lambda row: (
            -float(row["joint_accuracy"]),
            -int(row["net_fixes"]),
            int(row["breaks"]),
            int(row["subset_size"]),
            str(row["bundle_id"]),
        )
    )

    size_rows: List[Dict[str, Any]] = []
    by_size: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_size[int(row["subset_size"])].append(row)
    for size, values in sorted(by_size.items()):
        best = sorted(
            values,
            key=lambda row: (
                -float(row["joint_accuracy"]),
                -int(row["net_fixes"]),
                int(row["breaks"]),
            ),
        )[0]
        size_rows.append(
            {
                "subset_size": size,
                "subset_count": len(values),
                "mean_joint_accuracy": safe_mean(float(x["joint_accuracy"]) for x in values),
                "mean_accuracy_delta": safe_mean(float(x["accuracy_delta"]) for x in values),
                "mean_net_fixes": safe_mean(float(x["net_fixes"]) for x in values),
                "best_bundle_id": best["bundle_id"],
                "best_head_names": best["head_names"],
                "best_joint_accuracy": best["joint_accuracy"],
                "best_accuracy_delta": best["accuracy_delta"],
                "best_fixes": best["fixes"],
                "best_breaks": best["breaks"],
                "best_net_fixes": best["net_fixes"],
            }
        )

    best = summary_rows[0] if summary_rows else None
    return summary_rows, size_rows, status_rows, best


def summarize_oracle(
    *,
    rows: Sequence[Mapping[str, Any]],
    baseline_by_sid: Mapping[int, Mapping[str, Any]],
    selected_sids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    selected = set(map(int, selected_sids))
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["sid"]) in selected:
            grouped[str(row["oracle_mode"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for mode, values in sorted(grouped.items()):
        by_sid = {int(row["sid"]): row for row in values}
        common = sorted(selected & set(by_sid) & set(baseline_by_sid))
        subset = [by_sid[sid] for sid in common]
        if not subset:
            continue
        baseline_correct = sum(int(bool(x["baseline_correct"])) for x in subset)
        oracle_correct = sum(int(bool(x["oracle_correct"])) for x in subset)
        fixes = sum(int(bool(x["fixed"])) for x in subset)
        breaks = sum(int(bool(x["broken"])) for x in subset)
        wrong = [x for x in subset if not bool(x["baseline_correct"])]
        eligible_wrong = [x for x in wrong if bool(x["intervened"])]
        row = {
            "oracle_mode": mode,
            "oracle_kind": subset[0]["oracle_kind"],
            "oracle_k": subset[0]["oracle_k"],
            "analysis_oracle": True,
            "N": len(subset),
            "parse_rate": safe_mean(int(bool(x["oracle_parsed"])) for x in subset),
            "baseline_accuracy": baseline_correct / len(subset),
            "oracle_accuracy": oracle_correct / len(subset),
            "accuracy_delta": (oracle_correct - baseline_correct) / len(subset),
            "fixes": fixes,
            "breaks": breaks,
            "net_fixes": fixes - breaks,
            "wrong_N": len(wrong),
            "eligible_wrong_N": len(eligible_wrong),
            "eligible_wrong_rate": len(eligible_wrong) / max(len(wrong), 1),
            "fix_rate_among_all_baseline_wrong": fixes / max(len(wrong), 1),
            "fix_rate_among_intervened_wrong": fixes / max(len(eligible_wrong), 1),
            "mean_selected_head_count_all": safe_mean(
                float(x["selected_head_count"]) for x in subset
            ),
            "mean_selected_head_count_intervened": safe_mean(
                float(x["selected_head_count"]) for x in subset if bool(x["intervened"])
            ),
            "median_selected_head_count_intervened": safe_median(
                float(x["selected_head_count"]) for x in subset if bool(x["intervened"])
            ),
            "mcnemar_exact_p": exact_two_sided_binomial_p(fixes, breaks),
        }
        summary_rows.append(row)

        statuses = sorted(set(str(x["source_generation_pair_status"]) for x in subset))
        for status in statuses:
            part = [x for x in subset if str(x["source_generation_pair_status"]) == status]
            part_fixes = sum(int(bool(x["fixed"])) for x in part)
            part_breaks = sum(int(bool(x["broken"])) for x in part)
            status_rows.append(
                {
                    "oracle_mode": mode,
                    "source_generation_pair_status": status,
                    "status_alias": STATUS_ALIASES.get(status, status),
                    "N": len(part),
                    "baseline_accuracy": safe_mean(
                        int(bool(x["baseline_correct"])) for x in part
                    ),
                    "oracle_accuracy": safe_mean(int(bool(x["oracle_correct"])) for x in part),
                    "intervention_rate": safe_mean(int(bool(x["intervened"])) for x in part),
                    "fixes": part_fixes,
                    "breaks": part_breaks,
                    "net_fixes": part_fixes - part_breaks,
                }
            )

    summary_rows.sort(
        key=lambda row: (
            -float(row["oracle_accuracy"]),
            -int(row["fixes"]),
            float(row["mean_selected_head_count_intervened"]),
            str(row["oracle_mode"]),
        )
    )
    best = summary_rows[0] if summary_rows else None
    return summary_rows, status_rows, best


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
    unknown_experiments = experiments - {"fixed_subsets", "adaptive_oracle"}
    if unknown_experiments:
        raise ValueError(f"Unknown experiments: {sorted(unknown_experiments)}")
    if not experiments:
        raise ValueError("No experiment selected")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_helper = import_file(Path(args.single_head_script), "bundle_misrouting_single")
    failure = import_file(Path(args.failure_script), "bundle_misrouting_failure")
    generation = import_file(Path(args.generation_helper), "bundle_misrouting_generation")
    ioi = import_file(Path(args.ioi_script), "bundle_misrouting_ioi")
    producer = import_file(Path(args.producer_script), "bundle_misrouting_producer")
    receiver = import_file(Path(args.receiver_script), "bundle_misrouting_receiver")
    v3 = import_file(Path(args.v3_script), "bundle_misrouting_v3")
    base = import_file(Path(args.base_script), "bundle_misrouting_base")
    attention_helper = import_file(
        Path(args.attention_helper),
        "bundle_misrouting_attention",
    )

    source_config, source_rows = ioi.load_source_rows(args)

    excluded: set[int] = set()
    for item in parse_csv_tokens(args.exclude_sids_from):
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
    selected_sids = [int(row["sid"]) for row in selected_rows]
    selected_sid_set = set(selected_sids)

    bundle_payload = load_bundle_payload(Path(args.bundle_json))
    if args.negative_bundle not in bundle_payload:
        raise KeyError(
            f"Missing negative bundle {args.negative_bundle}; "
            f"available={list(bundle_payload)}"
        )
    negative_heads = tuple(bundle_payload[args.negative_bundle])
    fixed_specs = build_fixed_subsets(
        negative_heads,
        int(args.subset_min_size),
        int(args.subset_max_size),
    )

    candidate_bundle_names = parse_csv_tokens(args.oracle_candidate_bundles)
    candidate_heads: List[Tuple[int, int]] = []
    for name in candidate_bundle_names:
        if name not in bundle_payload:
            raise KeyError(f"Missing oracle candidate bundle {name}")
        candidate_heads.extend(bundle_payload[name])
    candidate_heads = list(canonical_head_tuple(candidate_heads))
    candidate_names = {head_name(layer, head) for layer, head in candidate_heads}

    oracle_modes = build_oracle_modes(
        parse_csv_tokens(args.oracle_kinds),
        parse_oracle_k_values(args.oracle_k_values),
    )

    single_dir = Path(args.single_head_output_dir)
    single_baseline_rows, single_effect_rows, single_head_summary = (
        load_single_head_artifacts(single_dir)
    )
    baseline_by_sid = {
        int(row["sid"]): row
        for row in single_baseline_rows
        if int(row["sid"]) in selected_sid_set
    }
    missing_baselines = sorted(selected_sid_set - set(baseline_by_sid))
    if missing_baselines:
        raise RuntimeError(
            f"Single-head baseline output is missing {len(missing_baselines)} selected SIDs; "
            f"first={missing_baselines[:10]}"
        )

    single_rows_by_sid: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    single_keys = set()
    for row in single_effect_rows:
        sid = int(row["sid"])
        name = str(row["head"])
        if sid in selected_sid_set and name in candidate_names:
            single_rows_by_sid[sid].append(row)
            single_keys.add((sid, name))
    missing_single_keys = [
        (sid, name)
        for sid in selected_sids
        for name in sorted(candidate_names)
        if (sid, name) not in single_keys
    ]
    if missing_single_keys:
        raise RuntimeError(
            f"Single-head effect output is missing {len(missing_single_keys)} SID/head rows; "
            f"first={missing_single_keys[:10]}"
        )

    baseline_accuracy = safe_mean(
        int(bool(baseline_by_sid[sid]["correct"])) for sid in selected_sids
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

        # Validate sender heads against the loaded decoder.
        all_required_heads = canonical_head_tuple(list(negative_heads) + candidate_heads)
        for layer, head in all_required_heads:
            if layer >= int(args.receiver_layer):
                raise ValueError(
                    f"Sender {head_name(layer, head)} must precede receiver layer "
                    f"{args.receiver_layer}"
                )
            attention = attention_helper.resolve_self_attention(decoder_layers[layer])
            shape = receiver.resolve_attention_shape(attention)
            if not 0 <= head < int(shape.n_query_heads):
                raise ValueError(
                    f"Head {head_name(layer, head)} outside 0..{int(shape.n_query_heads)-1}"
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

        receiver_attention = attention_helper.resolve_self_attention(
            decoder_layers[int(unit.layer)]
        )
        receiver_shape = receiver.resolve_attention_shape(receiver_attention)
        projection = receiver.projection_module(receiver_attention, unit.channel)
        patch_head_dim = int(receiver_shape.kv_head_dim)

        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "single_head_output_dir": str(single_dir),
            "decoder_path": decoder_path,
            "experiments": sorted(experiments),
            "negative_bundle": args.negative_bundle,
            "negative_heads": [head_name(*head) for head in negative_heads],
            "fixed_subset_count": len(fixed_specs),
            "fixed_subsets": [
                {
                    "bundle_id": spec_.bundle_id,
                    "heads": list(spec_.head_names),
                    "size": spec_.size,
                }
                for spec_ in fixed_specs
            ],
            "oracle_candidate_bundles": candidate_bundle_names,
            "oracle_candidate_heads": sorted(candidate_names),
            "oracle_modes": [
                {
                    "mode": mode.mode,
                    "kind": mode.kind,
                    "k": "all" if mode.k is None else mode.k,
                }
                for mode in oracle_modes
            ],
            "oracle_definition": (
                "analysis-only sample-adaptive joint removal selected using GT, "
                "baseline error status, and single-head wrong-over-GT effects"
            ),
            "receiver": {
                "unit": unit.unit,
                "layer": int(unit.layer),
                "kv_head": int(unit.kv_head),
                "unit_head": int(unit.unit_head),
                "shared_query_heads": list(unit.shared_query_heads),
                "channel": unit.channel,
                "kv_scope": args.receiver_kv_scope,
            },
            "intervention_definition": (
                "joint sender-head removal at object positions; resulting L26VH0 "
                "V state patched into full-prompt generation prefill KV cache"
            ),
            "compute_closed_scores": bool(args.compute_closed_scores),
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
            "selected_sids": selected_sids,
            "baseline_accuracy_from_single_head_output": baseline_accuracy,
            "seed": args.seed,
            "audit": audit,
            "transformers_version": transformers.__version__,
        }
        write_json(output_dir / "config.json", config)

        # Keep a selected baseline reference in the new output directory.
        baseline_reference = [baseline_by_sid[sid] for sid in selected_sids]
        baseline_reference_path = output_dir / "baseline_reference.jsonl"
        if args.overwrite or not baseline_reference_path.exists():
            baseline_reference_path.write_text("", encoding="utf-8")
            for row in baseline_reference:
                append_jsonl(baseline_reference_path, row)

        fixed_path = output_dir / "fixed_subset_generation.jsonl"
        oracle_path = output_dir / "adaptive_oracle_generation.jsonl"
        errors_path = output_dir / "errors.jsonl"

        fixed_rows = read_jsonl(fixed_path) if args.resume else []
        oracle_rows = read_jsonl(oracle_path) if args.resume else []
        fixed_done = {
            (int(row["sid"]), str(row["bundle_id"])) for row in fixed_rows
        }
        oracle_done = {
            (int(row["sid"]), str(row["oracle_mode"])) for row in oracle_rows
        }
        required_fixed_ids = {spec_.bundle_id for spec_ in fixed_specs}
        required_oracle_modes = {mode.mode for mode in oracle_modes}

        pending_rows = []
        for source_row in selected_rows:
            sid = int(source_row["sid"])
            need_fixed = (
                "fixed_subsets" in experiments
                and any((sid, bundle_id) not in fixed_done for bundle_id in required_fixed_ids)
            )
            need_oracle = (
                "adaptive_oracle" in experiments
                and any((sid, mode) not in oracle_done for mode in required_oracle_modes)
            )
            if need_fixed or need_oracle:
                pending_rows.append(source_row)

        expected_fixed = (
            len(selected_rows) * len(fixed_specs)
            if "fixed_subsets" in experiments
            else 0
        )
        expected_oracle = (
            len(selected_rows) * len(oracle_modes)
            if "adaptive_oracle" in experiments
            else 0
        )
        print(
            "Joint-bundle free-generation scan: "
            f"requested_N={len(selected_rows)}, pending_N={len(pending_rows)}, "
            f"fixed_subsets={len(fixed_specs) if 'fixed_subsets' in experiments else 0}, "
            f"oracle_modes={len(oracle_modes) if 'adaptive_oracle' in experiments else 0}, "
            f"expected_fixed_rows={expected_fixed}, expected_oracle_rows={expected_oracle}, "
            f"existing_fixed_rows={len(fixed_rows)}, existing_oracle_rows={len(oracle_rows)}, "
            f"baseline_acc={baseline_accuracy:.4f}",
            flush=True,
        )

        capture_layers = list(range(receiver_layer + 1))

        for sample_index, source_row in enumerate(
            tqdm(pending_rows, desc=f"bundle-misrouting:{args.model}"),
            start=1,
        ):
            pair = None
            try:
                sid = int(source_row["sid"])
                baseline_row = baseline_by_sid[sid]
                gt = normalize_relation(baseline_row.get("gt"))
                baseline_prediction = normalize_relation(baseline_row.get("prediction"))
                baseline_correct = bool(baseline_row.get("correct"))
                if gt is None:
                    raise RuntimeError(f"Invalid baseline GT for SID {sid}")

                missing_fixed = (
                    [
                        spec_
                        for spec_ in fixed_specs
                        if (sid, spec_.bundle_id) not in fixed_done
                    ]
                    if "fixed_subsets" in experiments
                    else []
                )
                missing_oracle = (
                    [
                        mode
                        for mode in oracle_modes
                        if (sid, mode.mode) not in oracle_done
                    ]
                    if "adaptive_oracle" in experiments
                    else []
                )
                if not missing_fixed and not missing_oracle:
                    continue

                oracle_heads_by_mode: Dict[str, List[Tuple[int, int]]] = {}
                for mode in missing_oracle:
                    # Correct baseline samples are intentionally left untouched in this
                    # analysis oracle.  On wrong samples, selection uses GT-derived
                    # single-head causal effects.
                    selected_heads = []
                    if not baseline_correct:
                        selected_heads = select_oracle_heads(
                            rows=single_rows_by_sid[sid],
                            mode=mode,
                            candidate_names=candidate_names,
                            min_score=float(args.oracle_min_score),
                        )
                    oracle_heads_by_mode[mode.mode] = selected_heads

                intervention_head_sets: List[Tuple[Tuple[int, int], ...]] = []
                intervention_head_sets.extend(
                    canonical_head_tuple(spec_.heads) for spec_ in missing_fixed
                )
                intervention_head_sets.extend(
                    canonical_head_tuple(heads)
                    for heads in oracle_heads_by_mode.values()
                    if heads
                )
                unique_interventions = sorted(
                    set(intervention_head_sets),
                    key=lambda heads: (len(heads), heads),
                )

                result_cache: Dict[Tuple[Tuple[int, int], ...], Dict[str, Any]] = {}
                if unique_interventions:
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
                    clean_score_map = {
                        relation: float(clean_scores["scores"][relation])
                        for relation in RELATIONS
                    }
                    pred_target = (
                        baseline_prediction
                        if baseline_prediction is not None
                        else normalize_relation(clean_scores.get("prediction"))
                    )

                    for head_tuple in unique_interventions:
                        names = tuple(head_name(*head) for head in head_tuple)
                        bundle = failure.HeadBundle(
                            name="JOINT__" + "__".join(names),
                            heads=nodes_for_heads(head_tuple, ioi),
                        )
                        ablated_states = failure.run_bundle_removal_c_pass(
                            bundle=bundle,
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
                        prediction = normalize_relation(generated["prediction"])
                        correct = bool(prediction == gt)

                        closed_score_map: Optional[Dict[str, float]] = None
                        closed_prediction: Optional[str] = None
                        e_gt = float("nan")
                        e_pred = float("nan")
                        preferred_relation: Optional[str] = None
                        if args.compute_closed_scores:
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
                            closed_score_map = {
                                relation: float(ablated_scores["scores"][relation])
                                for relation in RELATIONS
                            }
                            closed_prediction = normalize_relation(
                                ablated_scores.get("prediction")
                            )
                            contribution = {
                                relation: clean_score_map[relation]
                                - closed_score_map[relation]
                                for relation in RELATIONS
                            }
                            preferred_relation = max(
                                RELATIONS,
                                key=lambda relation: contribution[relation],
                            )
                            e_gt = relation_margin(clean_score_map, gt) - relation_margin(
                                closed_score_map, gt
                            )
                            e_pred = relation_margin(
                                clean_score_map, pred_target
                            ) - relation_margin(closed_score_map, pred_target)

                        delta_norm, delta_ratio = failure.state_delta_norms(
                            baseline_v,
                            ablated_v,
                            unit,
                            decoder_layers,
                            attention_helper,
                            receiver,
                        )
                        result_cache[head_tuple] = {
                            "generation": generated["text"],
                            "prediction": prediction,
                            "parsed": prediction is not None,
                            "correct": correct,
                            "new_token_count": int(generated["new_token_count"]),
                            "generation_seconds": float(generated["generation_seconds"]),
                            "closed_prediction": closed_prediction,
                            "closed_scores": closed_score_map,
                            "E_GT": float(e_gt),
                            "E_pred": float(e_pred),
                            "preferred_relation": preferred_relation,
                            "receiver_delta_norm": float(delta_norm),
                            "receiver_delta_ratio": float(delta_ratio),
                            "sender_positions": sender_positions,
                            "receiver_positions": receiver_positions,
                        }

                # Fixed subset rows.
                for spec_ in missing_fixed:
                    head_tuple = canonical_head_tuple(spec_.heads)
                    result = result_cache[head_tuple]
                    row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": gt,
                        "bundle_id": spec_.bundle_id,
                        "subset_size": spec_.size,
                        "head_names": ",".join(spec_.head_names),
                        "baseline_generation": baseline_row.get("generation", ""),
                        "baseline_generation_prediction": baseline_prediction,
                        "baseline_parsed": bool(baseline_row.get("parsed")),
                        "baseline_correct": baseline_correct,
                        "joint_generation": result["generation"],
                        "joint_generation_prediction": result["prediction"],
                        "joint_parsed": result["parsed"],
                        "joint_correct": result["correct"],
                        "fixed": bool((not baseline_correct) and result["correct"]),
                        "broken": bool(baseline_correct and (not result["correct"])),
                        "generation_prediction_changed": bool(
                            result["prediction"] != baseline_prediction
                        ),
                        "joint_new_token_count": result["new_token_count"],
                        "joint_generation_seconds": result["generation_seconds"],
                        "joint_closed_prediction": result["closed_prediction"],
                        "joint_closed_scores": result["closed_scores"],
                        "E_GT": result["E_GT"],
                        "E_pred": result["E_pred"],
                        "bundle_preferred_relation": result["preferred_relation"],
                        "receiver_delta_norm": result["receiver_delta_norm"],
                        "receiver_delta_ratio": result["receiver_delta_ratio"],
                        "source_generation_pair_status": source_row.get(
                            "generation_pair_status", "unknown"
                        ),
                        "status_alias": STATUS_ALIASES.get(
                            str(source_row.get("generation_pair_status", "")),
                            str(source_row.get("generation_pair_status", "unknown")),
                        ),
                        "sender_positions": result["sender_positions"],
                        "receiver_positions": result["receiver_positions"],
                        "receiver_unit": unit.unit,
                        "ablation_type": "path_specific_joint_sender_subset_to_L26VH0",
                        "cache_intervention": "prefill V projection patch only",
                    }
                    append_jsonl(fixed_path, row)
                    fixed_rows.append(row)
                    fixed_done.add((sid, spec_.bundle_id))

                # Sample-adaptive oracle rows.
                for mode in missing_oracle:
                    selected_heads = oracle_heads_by_mode[mode.mode]
                    head_tuple = canonical_head_tuple(selected_heads)
                    if head_tuple:
                        result = result_cache[head_tuple]
                        prediction = result["prediction"]
                        parsed = result["parsed"]
                        correct = result["correct"]
                        generation_text = result["generation"]
                        new_token_count = result["new_token_count"]
                        generation_seconds = result["generation_seconds"]
                        delta_norm = result["receiver_delta_norm"]
                        delta_ratio = result["receiver_delta_ratio"]
                        sender_positions_value = result["sender_positions"]
                        receiver_positions_value = result["receiver_positions"]
                        intervened = True
                    else:
                        prediction = baseline_prediction
                        parsed = bool(baseline_row.get("parsed"))
                        correct = baseline_correct
                        generation_text = baseline_row.get("generation", "")
                        new_token_count = int(baseline_row.get("new_token_count", 0))
                        generation_seconds = 0.0
                        delta_norm = 0.0
                        delta_ratio = 0.0
                        sender_positions_value = baseline_row.get("sender_positions", [])
                        receiver_positions_value = baseline_row.get("receiver_positions", [])
                        intervened = False
                    selected_names = [head_name(*head) for head in selected_heads]
                    oracle_row = {
                        "script_version": SCRIPT_VERSION,
                        "model": args.model,
                        "sid": sid,
                        "gt": gt,
                        "oracle_mode": mode.mode,
                        "oracle_kind": mode.kind,
                        "oracle_k": "all" if mode.k is None else int(mode.k),
                        "analysis_oracle": True,
                        "uses_gt_for_head_selection": True,
                        "uses_baseline_correctness_for_gating": True,
                        "selected_head_count": len(selected_heads),
                        "selected_head_names": ",".join(selected_names),
                        "intervened": intervened,
                        "baseline_generation": baseline_row.get("generation", ""),
                        "baseline_generation_prediction": baseline_prediction,
                        "baseline_parsed": bool(baseline_row.get("parsed")),
                        "baseline_correct": baseline_correct,
                        "oracle_generation": generation_text,
                        "oracle_generation_prediction": prediction,
                        "oracle_parsed": parsed,
                        "oracle_correct": correct,
                        "fixed": bool((not baseline_correct) and correct),
                        "broken": bool(baseline_correct and (not correct)),
                        "generation_prediction_changed": bool(
                            prediction != baseline_prediction
                        ),
                        "oracle_new_token_count": new_token_count,
                        "oracle_generation_seconds": generation_seconds,
                        "receiver_delta_norm": delta_norm,
                        "receiver_delta_ratio": delta_ratio,
                        "source_generation_pair_status": source_row.get(
                            "generation_pair_status", "unknown"
                        ),
                        "status_alias": STATUS_ALIASES.get(
                            str(source_row.get("generation_pair_status", "")),
                            str(source_row.get("generation_pair_status", "unknown")),
                        ),
                        "sender_positions": sender_positions_value,
                        "receiver_positions": receiver_positions_value,
                        "receiver_unit": unit.unit,
                        "ablation_type": (
                            "analysis_oracle_sample_adaptive_joint_misleading_sender_"
                            "removal_to_L26VH0"
                        ),
                        "cache_intervention": (
                            "none" if not intervened else "prefill V projection patch only"
                        ),
                    }
                    append_jsonl(oracle_path, oracle_row)
                    oracle_rows.append(oracle_row)
                    oracle_done.add((sid, mode.mode))

                if args.print_every > 0 and sample_index % args.print_every == 0:
                    print(
                        f"[sample {sample_index}/{len(pending_rows)} sid={sid}] "
                        f"fixed_rows={len(fixed_rows)} oracle_rows={len(oracle_rows)}",
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

        fixed_summary: List[Dict[str, Any]] = []
        fixed_size_summary: List[Dict[str, Any]] = []
        fixed_status_summary: List[Dict[str, Any]] = []
        best_fixed: Optional[Dict[str, Any]] = None
        if "fixed_subsets" in experiments:
            (
                fixed_summary,
                fixed_size_summary,
                fixed_status_summary,
                best_fixed,
            ) = summarize_fixed(
                rows=fixed_rows,
                baseline_by_sid=baseline_by_sid,
                selected_sids=selected_sids,
                single_head_summary=single_head_summary,
            )
            write_csv(output_dir / "fixed_subset_summary.csv", fixed_summary)
            write_csv(
                output_dir / "fixed_subset_size_summary.csv",
                fixed_size_summary,
            )
            write_csv(
                output_dir / "fixed_subset_status_summary.csv",
                fixed_status_summary,
            )

        oracle_summary: List[Dict[str, Any]] = []
        oracle_status_summary: List[Dict[str, Any]] = []
        best_oracle: Optional[Dict[str, Any]] = None
        if "adaptive_oracle" in experiments:
            oracle_summary, oracle_status_summary, best_oracle = summarize_oracle(
                rows=oracle_rows,
                baseline_by_sid=baseline_by_sid,
                selected_sids=selected_sids,
            )
            write_csv(output_dir / "adaptive_oracle_summary.csv", oracle_summary)
            write_csv(
                output_dir / "adaptive_oracle_status_summary.csv",
                oracle_status_summary,
            )

        summary = {
            "script_version": SCRIPT_VERSION,
            "N": len(selected_sids),
            "baseline_accuracy": baseline_accuracy,
            "baseline_correct": sum(
                int(bool(baseline_by_sid[sid]["correct"])) for sid in selected_sids
            ),
            "baseline_wrong": sum(
                int(not bool(baseline_by_sid[sid]["correct"])) for sid in selected_sids
            ),
            "negative_bundle": args.negative_bundle,
            "negative_heads": [head_name(*head) for head in negative_heads],
            "fixed_subset_count": len(fixed_specs),
            "complete_fixed_rows": len(
                [
                    row
                    for row in fixed_rows
                    if int(row["sid"]) in selected_sid_set
                    and str(row["bundle_id"]) in required_fixed_ids
                ]
            ),
            "expected_fixed_rows": expected_fixed,
            "oracle_mode_count": len(oracle_modes),
            "complete_oracle_rows": len(
                [
                    row
                    for row in oracle_rows
                    if int(row["sid"]) in selected_sid_set
                    and str(row["oracle_mode"]) in required_oracle_modes
                ]
            ),
            "expected_oracle_rows": expected_oracle,
            "best_fixed_subset": best_fixed,
            "best_adaptive_oracle": best_oracle,
            "oracle_is_analysis_only": True,
            "intervention_definition": (
                "joint sender removal at object positions; resulting L26VH0 V "
                "state patched into free-generation prefill cache"
            ),
        }
        write_json(output_dir / "summary.json", summary)

        print("\n" + "=" * 148)
        print("JOINT-BUNDLE MISROUTING FREE-GENERATION RESULT")
        print("=" * 148)
        print(
            f"Samples: {summary['N']} | baseline={summary['baseline_accuracy']:.4f} "
            f"({summary['baseline_correct']}/{summary['N']})"
        )

        if fixed_summary:
            print(
                f"Fixed P_NEG5 subset rows: {summary['complete_fixed_rows']}/"
                f"{summary['expected_fixed_rows']} | subsets={len(fixed_summary)}"
            )
            print("\nTOP FIXED NEGATIVE SUBSETS")
            print(
                f"{'size':>4} {'acc':>8} {'delta':>8} {'fix':>5} {'break':>6} "
                f"{'net':>5} {'synNet':>7} {'p':>10}  heads"
            )
            for row in fixed_summary[: min(15, len(fixed_summary))]:
                print(
                    f"{int(row['subset_size']):4d} "
                    f"{float(row['joint_accuracy']):8.4f} "
                    f"{float(row['accuracy_delta']):+8.4f} "
                    f"{int(row['fixes']):5d} {int(row['breaks']):6d} "
                    f"{int(row['net_fixes']):5d} "
                    f"{int(row['net_fix_synergy']):+7d} "
                    f"{float(row['mcnemar_exact_p']):10.4g}  "
                    f"{row['head_names']}"
                )
            print("\nBEST FIXED SUBSET BY SIZE")
            print(
                f"{'size':>4} {'count':>6} {'bestAcc':>9} {'delta':>8} "
                f"{'fix':>5} {'break':>6} {'net':>5}  heads"
            )
            for row in fixed_size_summary:
                print(
                    f"{int(row['subset_size']):4d} "
                    f"{int(row['subset_count']):6d} "
                    f"{float(row['best_joint_accuracy']):9.4f} "
                    f"{float(row['best_accuracy_delta']):+8.4f} "
                    f"{int(row['best_fixes']):5d} "
                    f"{int(row['best_breaks']):6d} "
                    f"{int(row['best_net_fixes']):5d}  "
                    f"{row['best_head_names']}"
                )

        if oracle_summary:
            print(
                f"\nAdaptive oracle rows: {summary['complete_oracle_rows']}/"
                f"{summary['expected_oracle_rows']} | modes={len(oracle_summary)}"
            )
            print("ANALYSIS ORACLE: SAMPLE-SPECIFIC JOINT MISLEADING-HEAD REMOVAL")
            print(
                f"{'mode':>18} {'acc':>8} {'delta':>8} {'fix':>5} "
                f"{'eligible':>9} {'fixAllWrong':>12} {'meanHeads':>10}"
            )
            for row in oracle_summary:
                print(
                    f"{str(row['oracle_mode']):>18} "
                    f"{float(row['oracle_accuracy']):8.4f} "
                    f"{float(row['accuracy_delta']):+8.4f} "
                    f"{int(row['fixes']):5d} "
                    f"{int(row['eligible_wrong_N']):9d} "
                    f"{float(row['fix_rate_among_all_baseline_wrong']):12.4f} "
                    f"{float(row['mean_selected_head_count_intervened']):10.3f}"
                )

        if abs(baseline_accuracy - float(args.expected_source_baseline_accuracy)) > float(
            args.baseline_warning_tolerance
        ):
            print(
                "WARNING: baseline from the preceding single-head output differs "
                f"from expected {args.expected_source_baseline_accuracy:.4f} by more "
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
