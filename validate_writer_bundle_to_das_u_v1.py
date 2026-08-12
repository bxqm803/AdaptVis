#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Causal Writer-Bundle -> DAS-U validation.

This is the family-level follow-up to validate_writer_to_das_u_v1.py.

Question
========
Single candidate writers move the learned causal relation subspace only a
little (e.g. source_progress ~0.03-0.09).  Are those effects small because the
relation representation is jointly written by a distributed FAMILY of heads,
or because the candidate heads are not actually important?

This script answers that directly by simultaneously applying natural
counterfactual state replacement to nested bundles selected by the previous
Writer->DAS-U causal ranking:

    Top1 / Top2 / Top4 / Top6 / Top10 / Top20

For every K it runs:

    topK_role
        Every selected candidate writer receives its own real swapped/source
        ROLE-aligned pre-W_O object state.

    topK_identity
        The same selected writers receive identity-aligned source states.

    topK_random_role
        A nested same-layer matched-random bundle of K heads receives its own
        real swapped/source ROLE-aligned states.

No head is ablated and no synthetic direction vector is added.

Main measurements
=================
At all requested learned causal relation subspaces U (default L22/L23/L24/L25,
D16), compare:

    t = clean target/original U state
    s = clean source/swapped ROLE-aligned U state
    p = target U state after the writer bundle intervention

source_progress:
    dot(p-t, s-t) / ||s-t||^2

    0 = no causal movement toward source
    1 = reaches the source state along the target->source direction

fraction_distance_closed:
    1 - ||p-s|| / ||t-s||

The script also measures final next-token source/opposite follow.

Crucial interpretation
======================
Strong distributed-writer evidence would look like:

    K increases
      -> ROLE U source_progress rises strongly
      -> final source-follow rises
      -> identity/random remain near zero

For example:
    Top1  progress .09
    Top2           .16
    Top4           .30
    Top6           .45
    Top10          .60

If even Top10/Top20 causal-ranked writer bundles remain weak while direct DAS-U
interchange has ~70% IIA, then the currently identified heads should NOT be
described as the main/important writer family.  They would be better described
as information-carrying causal contributors.

Layer detail
============
Bundles can contain heads at several layers.  The intervention is cumulative:
an L19 writer is patched at L19, an L21 writer at L21, an L23 writer at L23.

Therefore U@L22 only reflects selected writers at layers <=22.  U@L24 reflects
all selected writers at layers <=24.  Output reports `active_heads` for every
(K,U-layer) pair.

Dependency
==========
This script reuses tested hooking/data helpers from:

    validate_writer_to_das_u_v1.py

Keep that file in the repository root, or pass --writer-helper explicitly.

Recommended run
===============

CUDA_VISIBLE_DEVICES=0 python -u validate_writer_bundle_to_das_u_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --writer-results output/qwen3b_writer_to_das_u_top20/writer_summary.csv \
  --writer-helper validate_writer_to_das_u_v1.py \
  --das-dir output/qwen3b_relation_das_full \
  --bundle-sizes 1,2,4,6,10,20 \
  --u-layers 22,23,24,25 \
  --u-dim 16 \
  --sample-scope das_eval \
  --pair-status both_correct \
  --max-samples 0 \
  --replacement-mode tokenwise_resample \
  --device cuda:0 \
  --output-dir output/qwen3b_writer_bundle_to_das_u \
  --overwrite

Smoke test
==========

CUDA_VISIBLE_DEVICES=0 python -u validate_writer_bundle_to_das_u_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --writer-results output/qwen3b_writer_to_das_u_top20/writer_summary.csv \
  --das-dir output/qwen3b_relation_das_full \
  --bundle-sizes 1,2,4,6 \
  --u-layers 22,23,24,25 \
  --u-dim 16 \
  --sample-scope das_eval \
  --pair-status both_correct \
  --max-samples 60 \
  --device cuda:0 \
  --output-dir output/qwen3b_writer_bundle_to_das_u_smoke \
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


VERSION = "writer-bundle-to-das-u-v1"

RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
ID_TO_REL = {i: r for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
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
        "--writer-results",
        default="output/qwen3b_writer_to_das_u_top20/writer_summary.csv",
        help="writer_summary.csv from validate_writer_to_das_u_v1.py.",
    )
    p.add_argument(
        "--writer-helper",
        default="validate_writer_to_das_u_v1.py",
        help="Previous Writer->DAS-U script; reused for tested hooks/data helpers.",
    )
    p.add_argument(
        "--bundle-sizes",
        default="1,2,4,6,10,20",
        help="Nested bundle sizes from the causal Writer->U ranking.",
    )

    p.add_argument(
        "--das-dir",
        default="output/qwen3b_relation_das_full",
    )
    p.add_argument(
        "--u-layers",
        default="22,23,24,25",
    )
    p.add_argument("--u-dim", type=int, default=16)

    p.add_argument(
        "--sample-scope",
        default="das_eval",
        choices=("das_eval", "all"),
    )
    p.add_argument(
        "--pair-status",
        default="both_correct",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=17)

    p.add_argument(
        "--replacement-mode",
        default="tokenwise_resample",
        choices=("tokenwise_resample", "pooled_broadcast", "mean_shift"),
    )

    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--empty-cache-every", type=int, default=5)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


# =============================================================================
# Basic helpers
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
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


def parse_int_list(text: str) -> List[int]:
    vals: List[int] = []
    seen = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError(f"Expected positive integer, got {v}")
        if v not in seen:
            vals.append(v)
            seen.add(v)
    if not vals:
        raise ValueError("Empty integer list")
    return sorted(vals)


def hname(head: Tuple[int, int]) -> str:
    return f"L{int(head[0])}H{int(head[1]):02d}"


def parse_hname(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"L(\d+)H(\d+)", str(text).strip(), flags=re.I)
    if not m:
        raise ValueError(f"Bad head name: {text!r}")
    return int(m.group(1)), int(m.group(2))


def safe_mean(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def first_3d_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output) and output.ndim == 3:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("Could not find 3D hidden tensor")


def load_ranked_writers(
    path: Path,
    max_k: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    if not path.exists():
        candidates = sorted(Path("output").glob("**/writer_summary.csv"))
        msg = [f"Writer ranking not found: {path}"]
        if candidates:
            msg.append("Available writer_summary.csv candidates:")
            msg.extend(f"  {p}" for p in candidates[:30])
        raise FileNotFoundError("\n".join(msg))

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty writer ranking: {path}")

    parsed: List[Tuple[int, Tuple[int, int], Dict[str, Any]]] = []
    for fallback_rank, row in enumerate(rows, start=1):
        head = parse_hname(row["writer"])
        raw_rank = row.get("writer_rank_by_U", "")
        rank = int(float(raw_rank)) if str(raw_rank).strip() else fallback_rank
        parsed.append((rank, head, dict(row)))

    parsed.sort(key=lambda x: x[0])
    if len(parsed) < max_k:
        raise RuntimeError(
            f"Requested Top{max_k}, but writer ranking has only {len(parsed)} heads"
        )

    heads = [h for _, h, _ in parsed[:max_k]]
    ranked_rows = [r for _, _, r in parsed[:max_k]]

    if len(set(heads)) != len(heads):
        raise RuntimeError("Duplicate heads in writer ranking")

    return heads, ranked_rows


def stratified_subset(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in rows]
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda r: int(r["sid"]))

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gt"])].append(row)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    cursors = {r: 0 for r in RELATIONS}
    chosen: List[Dict[str, Any]] = []
    while len(chosen) < limit:
        moved = False
        for rel in RELATIONS:
            group = groups.get(rel, [])
            i = cursors[rel]
            if i < len(group) and len(chosen) < limit:
                chosen.append(group[i])
                cursors[rel] += 1
                moved = True
        if not moved:
            break
    return sorted(chosen, key=lambda r: int(r["sid"]))


# =============================================================================
# Multi-head natural patch
# =============================================================================

class MultiWriterPatchAndCapture:
    """
    Simultaneously patch a bundle of pre-W_O attention-head object states,
    possibly across multiple decoder layers, then capture DAS-U residual states.

    `bundle_heads` and source_states keys must match.
    """

    def __init__(
        self,
        *,
        helper: Any,
        decoder_layers: Sequence[Any],
        bundle_heads: Sequence[Tuple[int, int]],
        source_states: Mapping[Tuple[int, int], Mapping[str, np.ndarray]],
        alignment: str,
        target_a_positions: Sequence[int],
        target_b_positions: Sequence[int],
        capture_u_layers: Sequence[int],
        replacement_mode: str,
    ) -> None:
        self.helper = helper
        self.decoder_layers = decoder_layers
        self.bundle_heads = list(bundle_heads)
        self.source_states = source_states
        self.alignment = str(alignment)
        self.a_positions = sorted(set(map(int, target_a_positions)))
        self.b_positions = sorted(set(map(int, target_b_positions)))
        self.capture_u_layers = sorted(set(map(int, capture_u_layers)))
        self.replacement_mode = str(replacement_mode)

        self.by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer, head in self.bundle_heads:
            self.by_layer[int(layer)].append(int(head))

        self.handles: List[Any] = []
        self.residual_states: Dict[int, Dict[str, np.ndarray]] = {}
        self.patch_counts: Dict[Tuple[int, int], int] = {
            h: 0 for h in self.bundle_heads
        }

    def __enter__(self) -> "MultiWriterPatchAndCapture":
        # Register all writer patch hooks.
        for layer, heads in self.by_layer.items():
            attn = self.helper.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"L{layer} attention lacks o_proj")

            nh = int(
                self.helper.receiver.resolve_attention_shape(attn).n_query_heads
            )
            width = int(o_proj.weight.shape[1])
            if width % nh != 0:
                raise RuntimeError(
                    f"L{layer}: o_proj input width={width} not divisible by n_heads={nh}"
                )
            hd = width // nh

            def make_patch_hook(
                layer_index: int,
                head_ids: Sequence[int],
                head_dim: int,
            ):
                def patch_hook(_module: Any, inputs: Tuple[Any, ...]) -> Any:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        return None

                    modified = x.clone()

                    ap = torch.as_tensor(
                        self.a_positions,
                        device=x.device,
                        dtype=torch.long,
                    )
                    bp = torch.as_tensor(
                        self.b_positions,
                        device=x.device,
                        dtype=torch.long,
                    )

                    for head in head_ids:
                        key = (layer_index, int(head))
                        state = self.source_states[key]

                        if self.alignment == "role":
                            src_a_np = np.asarray(
                                state["B_tokens"], dtype=np.float32
                            )
                            src_b_np = np.asarray(
                                state["A_tokens"], dtype=np.float32
                            )
                        elif self.alignment == "identity":
                            src_a_np = np.asarray(
                                state["A_tokens"], dtype=np.float32
                            )
                            src_b_np = np.asarray(
                                state["B_tokens"], dtype=np.float32
                            )
                        else:
                            raise ValueError(self.alignment)

                        lo = int(head) * head_dim
                        hi = lo + head_dim

                        ta = modified[0].index_select(0, ap)[:, lo:hi]
                        tb = modified[0].index_select(0, bp)[:, lo:hi]

                        sa = torch.as_tensor(
                            src_a_np,
                            device=x.device,
                            dtype=x.dtype,
                        )
                        sb = torch.as_tensor(
                            src_b_np,
                            device=x.device,
                            dtype=x.dtype,
                        )

                        ra, _ = self.helper.mod.map_source_rows_to_target(
                            source_rows=sa,
                            target_rows=ta,
                            mode=self.replacement_mode,
                        )
                        rb, _ = self.helper.mod.map_source_rows_to_target(
                            source_rows=sb,
                            target_rows=tb,
                            mode=self.replacement_mode,
                        )

                        modified[0, ap, lo:hi] = ra
                        modified[0, bp, lo:hi] = rb
                        self.patch_counts[key] += 1

                    return (modified, *inputs[1:])

                return patch_hook

            self.handles.append(
                o_proj.register_forward_pre_hook(
                    make_patch_hook(layer, heads, hd)
                )
            )

        # Capture post-block residual states at requested U layers.
        for u_layer in self.capture_u_layers:
            block = self.decoder_layers[u_layer]

            def make_capture(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    hidden = first_3d_tensor(output)
                    if int(hidden.shape[0]) != 1:
                        return

                    ap = torch.as_tensor(
                        self.a_positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    bp = torch.as_tensor(
                        self.b_positions,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    a = hidden[0].index_select(0, ap).mean(dim=0)
                    b = hidden[0].index_select(0, bp).mean(dim=0)

                    self.residual_states[layer_index] = {
                        "A": a.detach().float().cpu().numpy().astype(np.float32),
                        "B": b.detach().float().cpu().numpy().astype(np.float32),
                    }

                return hook

            self.handles.append(
                block.register_forward_hook(make_capture(u_layer))
            )

        return self

    def validate(self) -> None:
        missing_heads = [
            hname(h)
            for h, count in self.patch_counts.items()
            if count < 1
        ]
        missing_u = [
            l for l in self.capture_u_layers
            if l not in self.residual_states
        ]
        if missing_heads or missing_u:
            raise RuntimeError(
                f"Missing patch/capture: heads={missing_heads}, U={missing_u}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


# =============================================================================
# Bundle evaluation
# =============================================================================

def run_bundle_condition(
    *,
    args: argparse.Namespace,
    helper: Any,
    condition: str,
    bundle_heads: Sequence[Tuple[int, int]],
    alignment: str,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[int, Mapping[str, Any]],
    u_layers: Sequence[int],
    bases: Mapping[int, np.ndarray],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Any,
    base: Any,
    v3: Any,
    error_path: Path,
    sample_log_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_u: Dict[int, List[Dict[str, float]]] = defaultdict(list)

    n = 0
    source_hits = 0
    target_hits = 0
    changes = 0
    clean_correct_to_source = 0
    clean_correct_n = 0
    margins: List[float] = []

    for i, row in enumerate(
        tqdm(rows, desc=condition, leave=False),
        start=1,
    ):
        sid = int(row["sid"])
        if sid not in cache:
            continue

        pair = None
        try:
            pair = helper.receiver.prepare_pair(
                args=args,
                row=row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base,
                v3=v3,
                processor=processor,
                device=torch.device(args.device),
            )

            source_states = {
                h: cache[sid]["head_states"][h]
                for h in bundle_heads
            }

            cap = MultiWriterPatchAndCapture(
                helper=helper,
                decoder_layers=decoder_layers,
                bundle_heads=bundle_heads,
                source_states=source_states,
                alignment=alignment,
                target_a_positions=pair.original_a_positions,
                target_b_positions=pair.original_b_positions,
                capture_u_layers=u_layers,
                replacement_mode=args.replacement_mode,
            )

            with cap:
                out = model(
                    **pair.original_batch,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            cap.validate()

            pred, scores = helper.mod.relation_scores(
                out.logits[0, -1],
                relation_token_map,
            )
            del out

            gt = str(pair.gt)
            source_gt = OPPOSITE[gt]
            clean_pred = str(cache[sid]["clean_prediction"])

            source_hits += int(pred == source_gt)
            target_hits += int(pred == gt)
            changes += int(pred != clean_pred)

            if clean_pred == gt:
                clean_correct_n += 1
                clean_correct_to_source += int(pred == source_gt)

            margins.append(
                float(
                    scores[REL_TO_ID[source_gt]]
                    - scores[REL_TO_ID[gt]]
                )
            )
            n += 1

            sample_row: Dict[str, Any] = {
                "sid": sid,
                "condition": condition,
                "gt_target": gt,
                "gt_source": source_gt,
                "clean_prediction": clean_pred,
                "patched_prediction": pred,
            }

            for u_layer in u_layers:
                q = bases[u_layer]

                target_coords = helper.mod.role_pair_coords(
                    cache[sid]["clean_u_states"][u_layer],
                    q,
                    source_branch=False,
                )
                source_coords = helper.mod.role_pair_coords(
                    cache[sid]["source_u_states"][u_layer],
                    q,
                    source_branch=True,
                )
                patched_coords = helper.mod.role_pair_coords(
                    cap.residual_states[u_layer],
                    q,
                    source_branch=False,
                )

                geom = helper.mod.u_geometry(
                    target_coords,
                    source_coords,
                    patched_coords,
                )
                per_u[u_layer].append(geom)

                sample_row[f"U{u_layer}_progress"] = geom["source_progress"]
                sample_row[f"U{u_layer}_closed"] = geom[
                    "fraction_distance_closed"
                ]
                sample_row[f"U{u_layer}_source_side"] = geom["source_side"]

            append_jsonl(sample_log_path, sample_row)

        except Exception as exc:
            append_jsonl(error_path, {
                "phase": "bundle_eval",
                "condition": condition,
                "sid": sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            if args.fail_fast:
                raise
        finally:
            if pair is not None:
                helper.receiver.release_pair(pair)

            if (
                torch.cuda.is_available()
                and args.empty_cache_every > 0
                and i % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    u_rows: List[Dict[str, Any]] = []
    for u_layer in u_layers:
        vals = per_u[u_layer]
        active = [h for h in bundle_heads if int(h[0]) <= int(u_layer)]
        u_rows.append({
            "condition": condition,
            "u_layer": int(u_layer),
            "bundle_size": len(bundle_heads),
            "active_heads": len(active),
            "active_head_names": ",".join(hname(h) for h in active),
            "N": len(vals),
            "source_progress_mean": safe_mean(
                v["source_progress"] for v in vals
            ),
            "source_progress_std": safe_std(
                v["source_progress"] for v in vals
            ),
            "fraction_distance_closed_mean": safe_mean(
                v["fraction_distance_closed"] for v in vals
            ),
            "source_side_rate": safe_mean(
                v["source_side"] for v in vals
            ),
            "off_axis_ratio_mean": safe_mean(
                v["off_axis_ratio"] for v in vals
            ),
            "move_norm_ratio_mean": safe_mean(
                v["move_norm_ratio"] for v in vals
            ),
        })

    final_row = {
        "condition": condition,
        "bundle_size": len(bundle_heads),
        "N": n,
        "next_token_source_follow": source_hits / n if n else float("nan"),
        "next_token_target_accuracy": target_hits / n if n else float("nan"),
        "next_token_change_vs_clean": changes / n if n else float("nan"),
        "clean_correct_to_source_rate": (
            clean_correct_to_source / clean_correct_n
            if clean_correct_n else float("nan")
        ),
        "clean_correct_N": clean_correct_n,
        "source_minus_target_margin_mean": safe_mean(margins),
        "bundle_heads": ",".join(hname(h) for h in bundle_heads),
    }
    return u_rows, final_row


# =============================================================================
# DAS direct-intervention reference
# =============================================================================

def load_das_reference(
    das_dir: Path,
    u_layers: Sequence[int],
    u_dim: int,
) -> Dict[int, float]:
    """
    Read heldout_both_correct learned_role IIA from the existing DAS summary.
    This is only a reference benchmark; no new DAS intervention is run here.
    """
    path = das_dir / "summary.csv"
    if not path.exists():
        return {}

    refs: Dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        try:
            layer = int(row["layer"])
            dim = int(row["dim"])
        except Exception:
            continue
        if layer not in set(u_layers) or dim != int(u_dim):
            continue
        if str(row.get("condition")) != "learned_role":
            continue
        if str(row.get("eval_subset")) != "heldout_both_correct":
            continue

        value = row.get("source_follow_iia", "")
        if str(value).strip():
            refs[layer] = float(value)

    return refs


# =============================================================================
# Compact reporting
# =============================================================================

def build_comparison_rows(
    *,
    all_u_rows: Sequence[Mapping[str, Any]],
    bundle_sizes: Sequence[int],
    u_layers: Sequence[int],
) -> List[Dict[str, Any]]:
    lookup = {
        (int(r["bundle_size"]), int(r["u_layer"]), str(r["condition"]).split("_", 1)[1]):
            r
        for r in all_u_rows
    }

    rows: List[Dict[str, Any]] = []
    for k in bundle_sizes:
        for u in u_layers:
            role = lookup.get((k, u, "role"))
            identity = lookup.get((k, u, "identity"))
            random_role = lookup.get((k, u, "random_role"))
            if role is None or identity is None or random_role is None:
                continue

            rows.append({
                "K": k,
                "u_layer": u,
                "active_heads": int(role["active_heads"]),
                "role_progress": float(role["source_progress_mean"]),
                "identity_progress": float(identity["source_progress_mean"]),
                "random_progress": float(random_role["source_progress_mean"]),
                "role_minus_identity": (
                    float(role["source_progress_mean"])
                    - float(identity["source_progress_mean"])
                ),
                "role_minus_random": (
                    float(role["source_progress_mean"])
                    - float(random_role["source_progress_mean"])
                ),
                "role_fraction_closed": float(
                    role["fraction_distance_closed_mean"]
                ),
                "identity_fraction_closed": float(
                    identity["fraction_distance_closed_mean"]
                ),
                "random_fraction_closed": float(
                    random_role["fraction_distance_closed_mean"]
                ),
                "role_source_side": float(role["source_side_rate"]),
                "identity_source_side": float(identity["source_side_rate"]),
                "random_source_side": float(random_role["source_side_rate"]),
                "active_head_names": str(role["active_head_names"]),
            })
    return rows


def build_final_comparison(
    *,
    finals: Sequence[Mapping[str, Any]],
    bundle_sizes: Sequence[int],
) -> List[Dict[str, Any]]:
    lookup = {
        (int(r["bundle_size"]), str(r["condition"]).split("_", 1)[1]): r
        for r in finals
    }
    rows: List[Dict[str, Any]] = []
    for k in bundle_sizes:
        role = lookup.get((k, "role"))
        identity = lookup.get((k, "identity"))
        random_role = lookup.get((k, "random_role"))
        if role is None or identity is None or random_role is None:
            continue

        rows.append({
            "K": k,
            "N": int(role["N"]),
            "role_source_follow": float(role["next_token_source_follow"]),
            "identity_source_follow": float(identity["next_token_source_follow"]),
            "random_source_follow": float(random_role["next_token_source_follow"]),
            "role_target_accuracy": float(role["next_token_target_accuracy"]),
            "identity_target_accuracy": float(identity["next_token_target_accuracy"]),
            "random_target_accuracy": float(random_role["next_token_target_accuracy"]),
            "role_change": float(role["next_token_change_vs_clean"]),
            "identity_change": float(identity["next_token_change_vs_clean"]),
            "random_change": float(random_role["next_token_change_vs_clean"]),
            "role_cleanGT_to_source": float(role["clean_correct_to_source_rate"]),
            "identity_cleanGT_to_source": float(identity["clean_correct_to_source_rate"]),
            "random_cleanGT_to_source": float(random_role["clean_correct_to_source_rate"]),
            "role_margin": float(role["source_minus_target_margin_mean"]),
            "identity_margin": float(identity["source_minus_target_margin_mean"]),
            "random_margin": float(random_role["source_minus_target_margin_mean"]),
            "bundle_heads": str(role["bundle_heads"]),
        })
    return rows


def print_u_table(
    rows: Sequence[Mapping[str, Any]],
    das_ref: Mapping[int, float],
) -> None:
    print("\n" + "=" * 146)
    print("WRITER BUNDLE -> DAS-U")
    print("=" * 146)
    print(
        f"{'K':>4s} {'U':>5s} {'active':>7s} "
        f"{'ROLE':>9s} {'identity':>10s} {'random':>9s} "
        f"{'role-id':>10s} {'role-rnd':>10s} "
        f"{'closed':>9s} {'srcSide':>9s} {'DAS IIA ref':>12s}"
    )
    print("-" * 146)
    for r in rows:
        ref = das_ref.get(int(r["u_layer"]), float("nan"))
        ref_text = (
            f"{100*ref:>10.2f}%"
            if np.isfinite(ref)
            else f"{'n/a':>11s}"
        )
        print(
            f"{int(r['K']):>4d} "
            f"L{int(r['u_layer']):<4d} "
            f"{int(r['active_heads']):>7d} "
            f"{float(r['role_progress']):>+8.3f} "
            f"{float(r['identity_progress']):>+9.3f} "
            f"{float(r['random_progress']):>+8.3f} "
            f"{float(r['role_minus_identity']):>+9.3f} "
            f"{float(r['role_minus_random']):>+9.3f} "
            f"{100*float(r['role_fraction_closed']):>8.2f}% "
            f"{100*float(r['role_source_side']):>8.2f}% "
            f"{ref_text}"
        )


def print_final_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 138)
    print("WRITER BUNDLE -> FINAL NEXT-TOKEN")
    print("=" * 138)
    print(
        f"{'K':>4s} {'N':>5s} "
        f"{'ROLE src':>10s} {'ID src':>9s} {'RND src':>9s} "
        f"{'ROLE tgt':>10s} {'change':>9s} {'cleanGT->src':>13s} "
        f"{'ROLE margin':>12s}"
    )
    print("-" * 138)
    for r in rows:
        print(
            f"{int(r['K']):>4d} "
            f"{int(r['N']):>5d} "
            f"{100*float(r['role_source_follow']):>9.2f}% "
            f"{100*float(r['identity_source_follow']):>8.2f}% "
            f"{100*float(r['random_source_follow']):>8.2f}% "
            f"{100*float(r['role_target_accuracy']):>9.2f}% "
            f"{100*float(r['role_change']):>8.2f}% "
            f"{100*float(r['role_cleanGT_to_source']):>12.2f}% "
            f"{float(r['role_margin']):>+11.4f}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundle_sizes = parse_int_list(args.bundle_sizes)
    max_k = max(bundle_sizes)
    u_layers = parse_int_list(args.u_layers)

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output directory not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    error_path = out_dir / "errors.jsonl"

    # Reuse the previous validated Writer->DAS-U implementation.
    mod = import_file(Path(args.writer_helper), "_writer_bundle_helper")

    class Helper:
        pass

    helper = Helper()
    helper.mod = mod
    helper.receiver = mod.receiver if hasattr(mod, "receiver") else None
    helper.attention_helper = (
        mod.attention_helper if hasattr(mod, "attention_helper") else None
    )

    # The previous script imports repo helper modules only inside main(), so
    # import them here using its own CLI-default filenames.
    helper.receiver = import_file(
        Path("analyze_coco_receiver_qkv_v1.py"),
        "_wb_receiver",
    )
    helper.attention_helper = import_file(
        Path("analyze_coco_flip_attention_spatial_vectors_v1.py"),
        "_wb_attention",
    )
    ioi = import_file(
        Path("analyze_coco_ioi_backward_circuit_v1.py"),
        "_wb_ioi",
    )
    producer = import_file(
        Path("analyze_coco_producer_qk_ov_v1.py"),
        "_wb_producer",
    )
    v3 = import_file(
        Path("analyze_spatial_storage_transport_utilization_v3.py"),
        "_wb_v3",
    )
    base = import_file(
        Path("analyze_coco_centroid_generation_step1_v4.py"),
        "_wb_base",
    )

    ranked_heads, ranked_rows = load_ranked_writers(
        Path(args.writer_results),
        max_k=max_k,
    )

    bases = mod.load_das_bases(
        das_dir=Path(args.das_dir),
        u_layers=u_layers,
        u_dim=args.u_dim,
    )

    source_dir = Path(args.source_output_dir)
    extraction_path = source_dir / "extraction.jsonl"
    if not extraction_path.exists():
        raise FileNotFoundError(extraction_path)

    rows = [
        r for r in read_jsonl(extraction_path)
        if str(r.get("gt")) in RELATIONS
    ]

    das_config_path = Path(args.das_dir) / "config.json"
    if args.sample_scope == "das_eval":
        if not das_config_path.exists():
            raise FileNotFoundError(das_config_path)
        das_config = json.loads(
            das_config_path.read_text(encoding="utf-8")
        )
        eval_sids = set(map(int, das_config.get("eval_sids", [])))
        if not eval_sids:
            raise RuntimeError(
                f"{das_config_path} contains no eval_sids"
            )
        rows = [r for r in rows if int(r["sid"]) in eval_sids]

    if args.pair_status != "all":
        rows = [
            r for r in rows
            if str(r.get("generation_pair_status", "")) == args.pair_status
        ]

    rows = stratified_subset(
        rows,
        args.max_samples,
        args.sample_seed,
    )
    if not rows:
        raise RuntimeError("No samples after filtering")

    model = processor = None

    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )
        model.eval()

        # Validate ranked heads.
        mod.validate_heads(
            ranked_heads,
            decoder_layers,
            helper.attention_helper,
            helper.receiver,
        )

        random_heads = mod.matched_random_heads(
            target_heads=ranked_heads,
            decoder_layers=decoder_layers,
            attention_helper=helper.attention_helper,
            receiver=helper.receiver,
            seed=args.seed + 991,
        )
        mod.validate_heads(
            random_heads,
            decoder_layers,
            helper.attention_helper,
            helper.receiver,
        )

        # prepare_data_helpers expects args.max_samples in the style of the
        # earlier scripts. Temporarily disable it to preserve sid lookup.
        saved_max = getattr(args, "max_samples", None)
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(
                args,
                base,
            )
        finally:
            args.max_samples = saved_max

        all_source_heads = list(
            dict.fromkeys(ranked_heads + random_heads)
        )

        print("\n" + "=" * 132)
        print("CAUSAL WRITER-BUNDLE -> DAS-U")
        print("=" * 132)
        print("model          :", args.model)
        print("N samples      :", len(rows))
        print("sample scope   :", args.sample_scope)
        print("pair status    :", args.pair_status)
        print("bundle sizes   :", bundle_sizes)
        print("U layers       :", u_layers)
        print("U dim          :", args.u_dim)
        print("\nWriter ranking used:")
        for i, head in enumerate(ranked_heads, start=1):
            row = ranked_rows[i - 1]
            best_prog = row.get("best_source_progress", "")
            best_u = row.get("best_u_layer", "")
            print(
                f"  {i:02d}. {hname(head)} "
                f"best_U=L{best_u} best_progress={best_prog}"
            )
        print("=" * 132, flush=True)

        config = {
            "version": VERSION,
            "model": args.model,
            "writer_results": args.writer_results,
            "writer_helper": args.writer_helper,
            "bundle_sizes": bundle_sizes,
            "ranked_writers": [hname(h) for h in ranked_heads],
            "matched_random": [hname(h) for h in random_heads],
            "u_layers": u_layers,
            "u_dim": args.u_dim,
            "sample_scope": args.sample_scope,
            "pair_status": args.pair_status,
            "N_samples_requested": len(rows),
            "sample_sids": [int(r["sid"]) for r in rows],
            "replacement_mode": args.replacement_mode,
            "audit": audit,
        }
        write_json(out_dir / "config.json", config)

        # Reuse previous source/clean cache function.
        # It expects a namespace with attributes used by prepare_pair.
        cache, successful_rows = mod.prepare_source_and_clean_cache(
            args=args,
            rows=rows,
            all_heads=all_source_heads,
            u_layers=u_layers,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            relation_token_map=relation_token_map,
            records_by_sid=records_by_sid,
            prompt_rows=prompt_rows,
            base=base,
            v3=v3,
            receiver=helper.receiver,
            attention_helper=helper.attention_helper,
            error_path=error_path,
        )
        rows = successful_rows
        if not rows:
            raise RuntimeError("No successful samples after cache phase")

        all_u_rows: List[Dict[str, Any]] = []
        all_final_rows: List[Dict[str, Any]] = []

        for k in bundle_sizes:
            role_heads = list(ranked_heads[:k])
            id_heads = list(ranked_heads[:k])
            rnd_heads = list(random_heads[:k])

            print("\n" + "-" * 120)
            print(
                f"Top{k} ROLE bundle: "
                + ", ".join(hname(h) for h in role_heads)
            )
            print("-" * 120, flush=True)

            conditions = [
                (f"top{k}_role", role_heads, "role"),
                (f"top{k}_identity", id_heads, "identity"),
                (f"top{k}_random_role", rnd_heads, "role"),
            ]

            for condition, heads, alignment in conditions:
                sample_log = out_dir / f"samples_{condition}.jsonl"

                u_rows, final_row = run_bundle_condition(
                    args=args,
                    helper=helper,
                    condition=condition,
                    bundle_heads=heads,
                    alignment=alignment,
                    rows=rows,
                    cache=cache,
                    u_layers=u_layers,
                    bases=bases,
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    error_path=error_path,
                    sample_log_path=sample_log,
                )
                all_u_rows.extend(u_rows)
                all_final_rows.append(final_row)

                write_csv(
                    out_dir / "bundle_u_effects_all.csv",
                    all_u_rows,
                )
                write_csv(
                    out_dir / "bundle_final_effects_all.csv",
                    all_final_rows,
                )

            # Print partial results after each K.
            partial_u = build_comparison_rows(
                all_u_rows=all_u_rows,
                bundle_sizes=[k],
                u_layers=u_layers,
            )
            partial_final = build_final_comparison(
                finals=all_final_rows,
                bundle_sizes=[k],
            )
            print_u_table(
                partial_u,
                load_das_reference(
                    Path(args.das_dir),
                    u_layers,
                    args.u_dim,
                ),
            )
            print_final_table(partial_final)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comparison_rows = build_comparison_rows(
            all_u_rows=all_u_rows,
            bundle_sizes=bundle_sizes,
            u_layers=u_layers,
        )
        final_comparison = build_final_comparison(
            finals=all_final_rows,
            bundle_sizes=bundle_sizes,
        )

        write_csv(
            out_dir / "bundle_u_comparison.csv",
            comparison_rows,
        )
        write_csv(
            out_dir / "bundle_final_comparison.csv",
            final_comparison,
        )

        das_ref = load_das_reference(
            Path(args.das_dir),
            u_layers,
            args.u_dim,
        )

        print_u_table(comparison_rows, das_ref)
        print_final_table(final_comparison)

        # Monotonic/trend diagnostics.
        trend_rows: List[Dict[str, Any]] = []
        for u in u_layers:
            relevant = [
                r for r in comparison_rows
                if int(r["u_layer"]) == int(u)
            ]
            relevant.sort(key=lambda r: int(r["K"]))
            if len(relevant) >= 2:
                xs = np.asarray(
                    [float(r["active_heads"]) for r in relevant],
                    dtype=np.float64,
                )
                ys = np.asarray(
                    [float(r["role_progress"]) for r in relevant],
                    dtype=np.float64,
                )

                if np.std(xs) > 0 and np.std(ys) > 0:
                    corr = float(np.corrcoef(xs, ys)[0, 1])
                else:
                    corr = float("nan")

                increments = [
                    ys[i] - ys[i - 1]
                    for i in range(1, len(ys))
                ]
                trend_rows.append({
                    "u_layer": u,
                    "pearson_active_heads_vs_role_progress": corr,
                    "role_progress_first": float(ys[0]),
                    "role_progress_last": float(ys[-1]),
                    "net_gain": float(ys[-1] - ys[0]),
                    "positive_step_fraction": (
                        sum(v > 0 for v in increments) / len(increments)
                        if increments else float("nan")
                    ),
                })

        write_csv(out_dir / "bundle_trends.csv", trend_rows)

        report = [
            f"version: {VERSION}",
            f"model: {args.model}",
            f"N successful samples: {len(rows)}",
            f"writer ranking: {args.writer_results}",
            f"bundle sizes: {bundle_sizes}",
            f"DAS U layers: {u_layers}, dim={args.u_dim}",
            "",
            "DECISION RULE",
            "Evidence for an important distributed writer family requires:",
            "  1) ROLE U source_progress rises substantially as causal-ranked K grows;",
            "  2) ROLE is clearly above identity and same-layer random bundles;",
            "  3) final source-follow rises in the same direction;",
            "  4) the largest bundle closes a meaningful fraction of the gap toward",
            "     the direct DAS-U causal intervention reference.",
            "",
            "If Top10/Top20 remain weak in U progress and final source-follow,",
            "do NOT call these heads the main/important relation writers.",
            "Call them causal contributors and search for additional mechanisms",
            "(MLPs, non-Direction heads, routing/QK, or distributed residual computation).",
            "",
            "FILES",
            "bundle_u_comparison.csv",
            "bundle_final_comparison.csv",
            "bundle_trends.csv",
            "bundle_u_effects_all.csv",
            "bundle_final_effects_all.csv",
            "samples_top*_*.jsonl",
            "config.json",
        ]
        (out_dir / "report.txt").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "bundle_u_comparison.csv",
            "bundle_final_comparison.csv",
            "bundle_trends.csv",
            "bundle_u_effects_all.csv",
            "bundle_final_effects_all.csv",
            "config.json",
            "report.txt",
        ):
            print(" ", out_dir / name)

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
