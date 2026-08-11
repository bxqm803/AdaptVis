#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate functional specialization in the COCO spatial-reasoning circuit.

This is a focused orchestration/analysis script.  It intentionally REUSES the
already-tested IOI-style exact sender->receiver path-patching implementation in:

    analyze_coco_ioi_backward_circuit_v1.py

rather than re-implementing fragile Qwen GQA/Q/K/V hooks.

Hypothesis
----------
Different head families perform different jobs:

    attention geometry / centroid heads
              |
              v
    relation-representation / direction heads
              |
              v
    downstream receiver / utilization heads
              |
              v
             answer

This script directly tests the middle arrow:

    sender head  --->  receiver Q/V  --->  relation-logit margin

using exact IOI-style path patching.

The underlying upstream_path intervention is:

C pass
    * Run original prompt.
    * Freeze intermediate attention outputs to CLEAN original activations.
    * At selected object-token positions, patch ONE sender head from the
      subject/reference-swapped prompt.
    * Recompute MLPs.
    * Capture resulting receiver Q/K/V projection state.

D pass
    * Run original prompt normally.
    * Patch only the selected receiver Q/K/V channel with the state captured
      in C.
    * Recompute receiver + downstream layers.
    * Measure GT-vs-opposite relation-logit effect.

Therefore intermediate attention-head mediated paths are excluded; the measured
effect is the direct residual/MLP-mediated sender->receiver path effect.

Groups
------
1. Direction Top-K:
   selected from the current per-head image A-B direction-probe CSV by
   img_accuracy_mean (NO no-image subtraction).

2. Centroid Top-K:
   supplied explicitly; defaults to the current Qwen-3B COCO centroid Top10.

3. Layer-matched random controls:
   for each Direction Top-K sender, sample a different head from the same layer
   when possible, excluding Direction/Centroid/Receiver heads.

Default receivers
-----------------
    L26H4, L26H2, L26H6, L26H0, L27H5

Important Qwen GQA note:
----------------------
For V/K, several L26 query heads share the same KV head.  The old exact-path
implementation correctly deduplicates those into a shared receiver unit
(e.g. L26 VH0).  Q remains query-head-specific.

Primary output
--------------
    sender_path_comparison.csv
        one row per sender, with:
          direction img accuracy/rank
          centroid rank
          max/mean absolute exact path effect
          best V receiver path
          best Q receiver path

    group_channel_summary.csv
        Direction vs Centroid vs matched-random comparisons by V/Q.

    exact_path/upstream_path_summary.csv
        untouched exact output from the old IOI implementation.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u validate_coco_spatial_specialization_path_v1.py \
  --model qwen-3b \
  --direction-results \
    output/qwen3b_coco_head_object_residual_direction_probe/head_results.csv \
  --source-output-dir \
    output/spatial_storage_transport_utilization/coco/qwen-3b \
  --direction-top-k 10 \
  --causal-max-samples 30 \
  --channels v,q \
  --sender-position-scopes objects_role \
  --output-dir output/qwen3b_spatial_specialization_path_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_VERSION = "coco-spatial-specialization-path-v1"

DEFAULT_CENTROID_TOP10 = (
    "27:10,24:5,28:8,20:5,31:7,22:13,22:0,21:1,20:0,28:15"
)

# The four L26 receiver candidates recovered previously + L27H5, whose Q channel
# had strong bidirectional receiver evidence.
DEFAULT_RECEIVERS = "26:4,26:2,26:6,26:0,27:5"


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b")
    p.add_argument(
        "--direction-results",
        required=True,
        help="head_results.csv from analyze_coco_head_object_residual_direction_probe_v1.py",
    )
    p.add_argument(
        "--direction-metric",
        default="img_accuracy_mean",
        help="Main direction-head ranking metric. Use img_accuracy_mean for raw image A-B.",
    )
    p.add_argument("--direction-top-k", type=int, default=10)
    p.add_argument(
        "--centroid-heads",
        default=DEFAULT_CENTROID_TOP10,
        help="Comma-separated L:H centroid Top-K heads.",
    )
    p.add_argument(
        "--receiver-heads",
        default=DEFAULT_RECEIVERS,
        help="Comma-separated L:H downstream receiver heads.",
    )
    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
        help="Existing source directory required by the old exact IOI path script.",
    )
    p.add_argument(
        "--ioi-script",
        default="analyze_coco_ioi_backward_circuit_v1.py",
        help="Existing exact IOI-style path-patching script in AdaptVis root.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--channels",
        default="v,q",
        help="Receiver channels. Recommended v,q. V tests transported content; Q tests receiver routing.",
    )
    p.add_argument(
        "--sender-position-scopes",
        default="objects_role",
        help="Recommended objects_role; this was the strong old sender->L26 V path.",
    )
    p.add_argument(
        "--receiver-kv-scope",
        default="objects",
        choices=("objects", "all"),
    )
    p.add_argument(
        "--causal-status",
        default="both_correct",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
    )
    p.add_argument("--causal-max-samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch the old exact-path script.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only aggregate an already existing exact_path/upstream_path_summary.csv.",
    )
    return p.parse_args()


def parse_head(text: str) -> Tuple[int, int]:
    s = str(text).strip().upper()
    s = s.replace("L", "").replace("H", ":")
    while "::" in s:
        s = s.replace("::", ":")
    if ":" not in s:
        raise ValueError(f"Bad head spec: {text!r}")
    a, b = s.split(":", 1)
    return int(a), int(b)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    output: List[Tuple[int, int]] = []
    seen = set()
    for item in str(text).split(","):
        if not item.strip():
            continue
        head = parse_head(item)
        if head not in seen:
            seen.add(head)
            output.append(head)
    return output


def hname(head: Tuple[int, int]) -> str:
    return f"L{int(head[0])}H{int(head[1]):02d}"


def old_head_spec(head: Tuple[int, int]) -> str:
    return f"{int(head[0])}:{int(head[1])}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


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


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x
    except Exception:
        return default


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def safe_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size else float("nan")


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small scipy-free average-rank implementation."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def corr_pair(x: Sequence[float], y: Sequence[float]) -> Tuple[int, float, float]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    n = len(a)
    if n < 3 or np.std(a) <= 0 or np.std(b) <= 0:
        return n, float("nan"), float("nan")
    pearson = float(np.corrcoef(a, b)[0, 1])
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    return n, pearson, spearman


# =============================================================================
# Groups
# =============================================================================

def load_direction_table(
    path: Path,
    metric: str,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    raw = read_csv(path)
    rows: List[Dict[str, Any]] = []
    for row in raw:
        if row.get("layer", "") == "" or row.get("head", "") == "":
            continue
        head = (int(row["layer"]), int(row["head"]))
        score = fnum(row.get(metric))
        if not math.isfinite(score):
            continue
        item = dict(row)
        item["_head"] = head
        item["_direction_score"] = score
        rows.append(item)

    rows.sort(key=lambda row: -float(row["_direction_score"]))
    for rank, row in enumerate(rows, 1):
        row["_direction_rank"] = rank

    by_head = {row["_head"]: row for row in rows}
    return rows, by_head


def choose_layer_matched_random(
    *,
    direction_heads: Sequence[Tuple[int, int]],
    all_heads: Sequence[Tuple[int, int]],
    exclude: set[Tuple[int, int]],
    seed: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    by_layer: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for head in all_heads:
        if head in exclude:
            continue
        by_layer[int(head[0])].append(head)
    for candidates in by_layer.values():
        rng.shuffle(candidates)

    chosen: List[Tuple[int, int]] = []
    used = set()
    for target in direction_heads:
        layer = int(target[0])
        candidates = [h for h in by_layer.get(layer, []) if h not in used]
        if candidates:
            pick = candidates[0]
            chosen.append(pick)
            used.add(pick)
            continue

        # Rare fallback: nearest layer, still excluding target groups.
        pool = [h for h in all_heads if h not in exclude and h not in used]
        if not pool:
            break
        dist = min(abs(int(h[0]) - layer) for h in pool)
        near = [h for h in pool if abs(int(h[0]) - layer) == dist]
        rng.shuffle(near)
        pick = near[0]
        chosen.append(pick)
        used.add(pick)

    return chosen


# =============================================================================
# Exact IOI path run
# =============================================================================

def make_manual_writer_top(
    receiver_heads: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "selection_metric": "manual_receiver_set_for_specialization_test",
        "positive_attention": [
            {
                "node": hname(head),
                "kind": "attention",
                "layer": int(head[0]),
                "head": int(head[1]),
                "mean_normalized_effect": 1.0,
                "median_normalized_effect": 1.0,
                "positive_effect_rate": 1.0,
                "N": 1,
            }
            for head in receiver_heads
        ],
        "negative_attention": [],
        "mlps": [],
    }


def run_exact_path(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    all_senders: Sequence[Tuple[int, int]],
    receiver_heads: Sequence[Tuple[int, int]],
) -> None:
    ioi_script = Path(args.ioi_script)
    source_dir = Path(args.source_output_dir)
    if not ioi_script.exists():
        raise FileNotFoundError(
            f"Missing {ioi_script}. Put this wrapper in AdaptVis root or pass --ioi-script."
        )
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Missing source-output-dir: {source_dir}. "
            "This exact-path test reuses your previous storage/transport extraction."
        )

    exact_dir = output_dir / "exact_path"
    exact_dir.mkdir(parents=True, exist_ok=True)

    writer_top_path = output_dir / "manual_receivers_writer_top.json"
    writer_top_path.write_text(
        json.dumps(make_manual_writer_top(receiver_heads), indent=2),
        encoding="utf-8",
    )

    sender_text = ",".join(old_head_spec(h) for h in all_senders)
    cmd = [
        str(args.python), "-u", str(ioi_script),
        "--phase", "upstream_path",
        "--model", str(args.model),
        "--source-output-dir", str(source_dir),
        "--writer-top-file", str(writer_top_path),
        "--receiver-writers", "positive_attention",
        "--max-receivers", str(len(receiver_heads)),
        "--sender-heads", sender_text,
        "--sender-mlps", "none",
        "--sender-position-scopes", str(args.sender_position_scopes),
        "--upstream-channels", str(args.channels),
        "--receiver-kv-scope", str(args.receiver_kv_scope),
        "--causal-status", str(args.causal_status),
        "--causal-max-samples", str(int(args.causal_max_samples)),
        "--device", str(args.device),
        "--output-dir", str(exact_dir),
        "--overwrite",
    ]

    print("\n" + "=" * 120)
    print("EXACT IOI PATH PATCH")
    print("=" * 120)
    print("Senders :", ", ".join(hname(h) for h in all_senders))
    print("Receivers:", ", ".join(hname(h) for h in receiver_heads))
    print("Channels:", args.channels)
    print("Scope   :", args.sender_position_scopes)
    print("Command :")
    print(" ".join(cmd))
    print("=" * 120, flush=True)

    subprocess.run(cmd, check=True)


# =============================================================================
# Aggregate exact path
# =============================================================================

def canonical_sender_from_summary(row: Mapping[str, str]) -> Tuple[int, int]:
    if row.get("sender_layer", "") != "" and row.get("sender_head", "") != "":
        return int(row["sender_layer"]), int(float(row["sender_head"]))

    s = str(row.get("sender", "")).strip()
    return parse_head(s)


def summarize_sender_paths(
    *,
    path_rows: Sequence[Mapping[str, str]],
    scanned_senders: Sequence[Tuple[int, int]],
    direction_by_head: Mapping[Tuple[int, int], Mapping[str, Any]],
    direction_set: set[Tuple[int, int]],
    centroid_set: set[Tuple[int, int]],
    random_set: set[Tuple[int, int]],
    receiver_set: set[Tuple[int, int]],
    direction_metric: str,
    centroid_heads: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    centroid_rank = {h: i + 1 for i, h in enumerate(centroid_heads)}

    grouped: Dict[Tuple[int, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in path_rows:
        try:
            grouped[canonical_sender_from_summary(row)].append(row)
        except Exception:
            continue

    output: List[Dict[str, Any]] = []
    for sender in scanned_senders:
        rows = grouped.get(sender, [])
        drow = direction_by_head.get(sender, {})
        base: Dict[str, Any] = {
            "sender": hname(sender),
            "layer": sender[0],
            "head": sender[1],
            "is_direction_top": int(sender in direction_set),
            "is_centroid_top": int(sender in centroid_set),
            "is_random_control": int(sender in random_set),
            "is_receiver_head": int(sender in receiver_set),
            "direction_rank": drow.get("_direction_rank", ""),
            "direction_img_accuracy": drow.get(direction_metric, ""),
            "n_exact_path_rows": len(rows),
        }
        if sender in centroid_rank:
            base["centroid_rank"] = centroid_rank[sender]
        else:
            base["centroid_rank"] = ""

        effects = [fnum(r.get("mean_normalized_effect")) for r in rows]
        finite = [v for v in effects if math.isfinite(v)]
        base["mean_path_effect"] = safe_mean(finite)
        base["mean_abs_path_effect"] = safe_mean(abs(v) for v in finite)
        base["max_abs_path_effect"] = max((abs(v) for v in finite), default=float("nan"))

        # Best overall path.
        best = None
        best_abs = -1.0
        for row in rows:
            effect = fnum(row.get("mean_normalized_effect"))
            if math.isfinite(effect) and abs(effect) > best_abs:
                best_abs = abs(effect)
                best = row
        if best is not None:
            base["best_receiver_unit"] = best.get("receiver_unit", "")
            base["best_channel"] = best.get("channel", "")
            base["best_scope"] = best.get("sender_position_scope", "")
            base["best_effect"] = fnum(best.get("mean_normalized_effect"))
            base["best_positive_rate"] = fnum(best.get("positive_effect_rate"))
            base["best_cross_rate"] = fnum(best.get("crossed_decision_boundary_rate"))
            base["best_N"] = int(float(best.get("N", 0) or 0))
        else:
            base.update({
                "best_receiver_unit": "",
                "best_channel": "",
                "best_scope": "",
                "best_effect": float("nan"),
                "best_positive_rate": float("nan"),
                "best_cross_rate": float("nan"),
                "best_N": 0,
            })

        # V / Q separate.
        for channel in ("v", "q", "k"):
            crows = [r for r in rows if str(r.get("channel", "")).lower() == channel]
            vals = [fnum(r.get("mean_normalized_effect")) for r in crows]
            vals = [v for v in vals if math.isfinite(v)]
            base[f"{channel}_mean_abs_effect"] = safe_mean(abs(v) for v in vals)
            base[f"{channel}_max_abs_effect"] = max((abs(v) for v in vals), default=float("nan"))

            cbest = None
            cbest_abs = -1.0
            for row in crows:
                effect = fnum(row.get("mean_normalized_effect"))
                if math.isfinite(effect) and abs(effect) > cbest_abs:
                    cbest_abs = abs(effect)
                    cbest = row
            if cbest is not None:
                base[f"{channel}_best_receiver"] = cbest.get("receiver_unit", "")
                base[f"{channel}_best_effect"] = fnum(cbest.get("mean_normalized_effect"))
                base[f"{channel}_best_positive_rate"] = fnum(cbest.get("positive_effect_rate"))
            else:
                base[f"{channel}_best_receiver"] = ""
                base[f"{channel}_best_effect"] = float("nan")
                base[f"{channel}_best_positive_rate"] = float("nan")

        output.append(base)

    # Rank by strongest exact path effect.
    output.sort(
        key=lambda r: (
            -fnum(r.get("max_abs_path_effect"), default=-1.0)
            if math.isfinite(fnum(r.get("max_abs_path_effect"))) else float("inf")
        )
    )
    for i, row in enumerate(output, 1):
        row["path_rank"] = i
    return output


def group_channel_summary(
    sender_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups = {
        "direction_top": lambda r: int(r["is_direction_top"]) == 1,
        "centroid_top": lambda r: int(r["is_centroid_top"]) == 1,
        "matched_random": lambda r: int(r["is_random_control"]) == 1,
    }
    rows: List[Dict[str, Any]] = []

    for gname, pred in groups.items():
        members = [r for r in sender_rows if pred(r)]
        for channel in ("all", "v", "q", "k"):
            key = (
                "max_abs_path_effect"
                if channel == "all"
                else f"{channel}_max_abs_effect"
            )
            vals = [fnum(r.get(key)) for r in members]
            vals = [v for v in vals if math.isfinite(v)]
            rows.append({
                "group": gname,
                "channel": channel,
                "requested_heads": len(members),
                "heads_with_path_result": len(vals),
                "mean_max_abs_effect": safe_mean(vals),
                "median_max_abs_effect": safe_median(vals),
                "std_max_abs_effect": safe_std(vals),
                "max_head_effect": max(vals, default=float("nan")),
            })
    return rows


def print_results(
    sender_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    direction_metric: str,
) -> None:
    print("\n" + "=" * 150)
    print("EXACT SENDER -> RECEIVER PATH EFFECTS")
    print("=" * 150)
    print(
        "rank sender    groups  dirAcc  dirRank  centRank   max|CE|    "
        "Vmax|CE|   Vbest          Qmax|CE|   Qbest"
    )
    print("-" * 150)
    for row in sender_rows:
        groups = ""
        if int(row["is_direction_top"]): groups += "D"
        if int(row["is_centroid_top"]): groups += "C"
        if int(row["is_random_control"]): groups += "R"
        dacc = fnum(row.get("direction_img_accuracy"))
        dr = row.get("direction_rank", "")
        cr = row.get("centroid_rank", "")
        print(
            f"{int(row['path_rank']):>4d} "
            f"{str(row['sender']):<9s} "
            f"{groups:<6s} "
            f"{dacc:>6.3f} "
            f"{str(dr):>7s} "
            f"{str(cr):>9s} "
            f"{fnum(row.get('max_abs_path_effect')):>9.4f} "
            f"{fnum(row.get('v_max_abs_effect')):>10.4f} "
            f"{str(row.get('v_best_receiver','')):<13s} "
            f"{fnum(row.get('q_max_abs_effect')):>10.4f} "
            f"{str(row.get('q_best_receiver','')):<13s}"
        )

    print("\n" + "=" * 105)
    print("GROUP SUMMARY")
    print("=" * 105)
    print("group             channel  Npath/requested   mean max|CE|   median max|CE|   max")
    print("-" * 105)
    for row in group_rows:
        print(
            f"{str(row['group']):<17s} "
            f"{str(row['channel']):<7s} "
            f"{int(row['heads_with_path_result']):>3d}/{int(row['requested_heads']):<3d} "
            f"{fnum(row['mean_max_abs_effect']):>14.4f} "
            f"{fnum(row['median_max_abs_effect']):>16.4f} "
            f"{fnum(row['max_head_effect']):>9.4f}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    direction_path = Path(args.direction_results)
    out = Path(args.output_dir)
    if args.overwrite and out.exists() and not args.skip_run:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    direction_rows, direction_by_head = load_direction_table(
        direction_path, args.direction_metric
    )
    if len(direction_rows) < args.direction_top_k:
        raise RuntimeError(
            f"Only {len(direction_rows)} valid direction rows, requested top-k={args.direction_top_k}"
        )

    direction_heads = [
        row["_head"] for row in direction_rows[: int(args.direction_top_k)]
    ]
    centroid_heads = parse_heads(args.centroid_heads)
    receiver_heads = parse_heads(args.receiver_heads)

    direction_set = set(direction_heads)
    centroid_set = set(centroid_heads)
    receiver_set = set(receiver_heads)

    all_available_heads = [row["_head"] for row in direction_rows]
    random_heads = choose_layer_matched_random(
        direction_heads=direction_heads,
        all_heads=all_available_heads,
        exclude=direction_set | centroid_set | receiver_set,
        seed=args.seed,
    )
    random_set = set(random_heads)

    # Union preserves meaningful group order.
    all_senders: List[Tuple[int, int]] = []
    seen = set()
    for group in (direction_heads, centroid_heads, random_heads):
        for head in group:
            if head not in seen:
                seen.add(head)
                all_senders.append(head)

    config = {
        "script_version": SCRIPT_VERSION,
        "direction_results": str(direction_path),
        "direction_metric": args.direction_metric,
        "direction_top_k": args.direction_top_k,
        "direction_heads": [hname(h) for h in direction_heads],
        "centroid_heads": [hname(h) for h in centroid_heads],
        "random_layer_matched_heads": [hname(h) for h in random_heads],
        "receiver_heads": [hname(h) for h in receiver_heads],
        "channels": args.channels,
        "sender_position_scopes": args.sender_position_scopes,
        "receiver_kv_scope": args.receiver_kv_scope,
        "causal_status": args.causal_status,
        "causal_max_samples": args.causal_max_samples,
        "source_output_dir": args.source_output_dir,
        "ioi_script": args.ioi_script,
        "seed": args.seed,
        "important_note": (
            "For Qwen GQA, multiple L26 query heads can share one KV/V receiver unit. "
            "V/K results are therefore shared-KV path effects; Q results are query-head-specific."
        ),
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nSelected groups")
    print("Direction Top :", ", ".join(hname(h) for h in direction_heads))
    print("Centroid Top  :", ", ".join(hname(h) for h in centroid_heads))
    print("Matched random:", ", ".join(hname(h) for h in random_heads))
    print("Receivers     :", ", ".join(hname(h) for h in receiver_heads))

    if not args.skip_run:
        run_exact_path(
            args=args,
            output_dir=out,
            all_senders=all_senders,
            receiver_heads=receiver_heads,
        )

    summary_path = out / "exact_path" / "upstream_path_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Expected exact-path summary not found: {summary_path}"
        )

    path_rows = read_csv(summary_path)
    sender_rows = summarize_sender_paths(
        path_rows=path_rows,
        scanned_senders=all_senders,
        direction_by_head=direction_by_head,
        direction_set=direction_set,
        centroid_set=centroid_set,
        random_set=random_set,
        receiver_set=receiver_set,
        direction_metric=args.direction_metric,
        centroid_heads=centroid_heads,
    )
    group_rows = group_channel_summary(sender_rows)

    write_csv(out / "sender_path_comparison.csv", sender_rows)
    write_csv(out / "group_channel_summary.csv", group_rows)

    # Correlation across every scanned sender that actually got a path result.
    dir_acc = []
    path_abs = []
    v_abs = []
    q_abs = []
    for row in sender_rows:
        da = fnum(row.get("direction_img_accuracy"))
        pa = fnum(row.get("max_abs_path_effect"))
        vv = fnum(row.get("v_max_abs_effect"))
        qq = fnum(row.get("q_max_abs_effect"))
        if math.isfinite(da) and math.isfinite(pa):
            dir_acc.append(da)
            path_abs.append(pa)
        if math.isfinite(da) and math.isfinite(vv):
            v_abs.append((da, vv))
        if math.isfinite(da) and math.isfinite(qq):
            q_abs.append((da, qq))

    n_all, p_all, s_all = corr_pair(dir_acc, path_abs)
    n_v, p_v, s_v = corr_pair(
        [x for x, _ in v_abs], [y for _, y in v_abs]
    )
    n_q, p_q, s_q = corr_pair(
        [x for x, _ in q_abs], [y for _, y in q_abs]
    )

    correlations = [
        {
            "comparison": "direction_img_acc_vs_max_abs_exact_path",
            "N": n_all,
            "pearson": p_all,
            "spearman": s_all,
        },
        {
            "comparison": "direction_img_acc_vs_V_max_abs_exact_path",
            "N": n_v,
            "pearson": p_v,
            "spearman": s_v,
        },
        {
            "comparison": "direction_img_acc_vs_Q_max_abs_exact_path",
            "N": n_q,
            "pearson": p_q,
            "spearman": s_q,
        },
    ]
    write_csv(out / "correlations.csv", correlations)

    print_results(sender_rows, group_rows, args.direction_metric)

    print("\nCorrelations across scanned senders:")
    for row in correlations:
        print(
            f"{row['comparison']:<52s} "
            f"N={row['N']:>2d} pearson={row['pearson']:+.4f} "
            f"spearman={row['spearman']:+.4f}"
        )

    # Compact interpretation checklist; no scientific conclusion is hard-coded.
    report = [
        f"script_version: {SCRIPT_VERSION}",
        "",
        "TESTED HYPOTHESIS",
        "Direction-high heads should have stronger exact sender->receiver path effects",
        "than layer-matched random heads if they are genuine relation-representation senders.",
        "",
        "STRONG SUPPORT PATTERN",
        "1. Direction Top-K > matched random on V max|normalized effect|.",
        "2. Known direction heads such as L23H5/L23H1/L19H13 rank high on paths into L26 VH0.",
        "3. Q effects may be receiver-head-specific; V effects may be shared across L26 query heads due to GQA.",
        "4. Direction accuracy correlates with exact path effect across the scanned sender pool.",
        "",
        "WEAK / NEGATIVE PATTERN",
        "Direction Top-K ~= random, or effects are dominated by centroid controls with no direction representation.",
        "",
        "IMPORTANT",
        "This script validates Direction -> Receiver specialization. It does not yet prove",
        "Centroid -> Direction causality; that should be tested separately by perturbing",
        "centroid/localization heads and measuring downstream direction-head representation.",
        "",
        "FILES",
        f"exact path summary: {summary_path}",
        f"sender comparison: {out / 'sender_path_comparison.csv'}",
        f"group summary: {out / 'group_channel_summary.csv'}",
        f"correlations: {out / 'correlations.csv'}",
    ]
    (out / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(f"  {out / 'sender_path_comparison.csv'}")
    print(f"  {out / 'group_channel_summary.csv'}")
    print(f"  {out / 'correlations.csv'}")
    print(f"  {out / 'report.txt'}")


if __name__ == "__main__":
    main()
