#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_spatial_head_write_post_downstream_v1.py

Question
========
Which attention heads contain / inject the most spatial information, and how
does that compare with heads that have the highest spatial relation accuracy?

This script deliberately separates FOUR different notions that are easy to
conflate:

(A) HEAD WRITE STRENGTH
    How much Synthetic-defined H/V spatial signal does this head itself write?

        delta_h = W_O^(h) * [(head_pair_REAL) - (head_pair_GRAY)]

        write_strength(h) = RMS || spatial_coord(delta_h) ||

(B) LOCAL POST-HEAD SPATIAL STATE
    Starting from the visual residual state entering the layer, what would the
    spatial state look like after adding ONLY this head's visual write?

        r_pre = pair(layer_input_REAL) - pair(layer_input_GRAY)

        r_after_h = r_pre + delta_h

    We report:
        local_post_strength
        local_gain = local_post_strength - pre_strength
        mean per-sample post-minus-pre magnitude
        spatial alignment between r_pre and delta_h

(C) FULL MULTI-HEAD POST-ATTENTION STATE
    For reference:

        r_after_all = r_pre + sum_h delta_h

    This quantifies mixing / cancellation across heads.

(D) DOWNSTREAM CAUSAL SURVIVAL (optional second pass)
    For a shortlist of heads, start from the GRAY-image computation and inject
    ONLY that head's clean REAL-GRAY pre-W_O contribution at its source layer.
    Then measure the induced subject-reference residual at a fixed later
    readout layer.

    This is an actual intervention:

        do(gray source-layer head h += real-minus-gray head contribution)

    and answers:
        "If this head alone injects its visual spatial write, how much spatial
         information survives to the later residual state?"

Head selection for the causal shortlist is the UNION of:
    - top-K write strength
    - top-K local post strength
    - top-K local gain
    - top-K prior COCO spatial-accuracy heads (if --accuracy-csv exists)
    - --known-heads

The script also joins the earlier head-accuracy scan, e.g.

    output/qwen3b_spatial_heads_syn400_vs_coco440_v1/all_head_scores.csv

so you can directly compare:
    high-accuracy spatial heads
vs
    high-write heads
vs
    high-post-state heads
vs
    high-downstream-causal-survival heads.

IMPORTANT
=========
COCO GT relation labels are NEVER used to rank the non-accuracy metrics.
GT-derived accuracies are reported only as validation columns.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u compare_spatial_head_write_post_downstream_v1.py \
  --model qwen-3b \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --source-layers 16-26 \
  --readout-layer 27 \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --accuracy-csv \
    output/qwen3b_spatial_heads_syn400_vs_coco440_v1/all_head_scores.csv \
  --pool mean \
  --causal-n 80 \
  --causal-top-k 8 \
  --causal-head-batch 4 \
  --device cuda:0 \
  --output-dir output/qwen3b_spatial_head_write_post_downstream_v1 \
  --overwrite

If you only want the cheap all-head local analysis, add:
    --skip-causal

Main outputs
============
  all_head_metrics.csv
  top_by_write_strength.csv
  top_by_local_post_strength.csv
  top_by_local_gain.csv
  top_by_prior_coco_accuracy.csv
  metric_vs_accuracy_correlation.csv
  topk_overlap_with_accuracy.csv
  per_layer_state_summary.csv
  causal_shortlist.csv
  causal_downstream_metrics.csv
  causal_vs_accuracy_correlation.csv
  spatial_basis.csv
  errors.csv
  config.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    import analyze_coco_centroid_generation_step1_v4 as base
    import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
except Exception as exc:
    raise SystemExit(
        "Run from the AdaptVis llava16 repo root next to "
        "analyze_coco_centroid_generation_step1_v4.py and "
        "eval_coco_multilayer_relation_trajectory_repair_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
EPS = 1e-12


# ============================================================================
# CLI / generic helpers
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--source-layers", default="16-26")
    p.add_argument("--readout-layer", type=int, default=27)

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument(
        "--accuracy-csv",
        default="output/qwen3b_spatial_heads_syn400_vs_coco440_v1/all_head_scores.csv",
        help=(
            "Optional earlier head-accuracy scan. If present, its "
            "coco_in_accuracy/ranks are joined for direct comparison."
        ),
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument("--known-heads", default="23:1,23:5,26:3,19:13,20:8,23:0")
    p.add_argument("--top-print", type=int, default=25)
    p.add_argument("--top-k-overlap", default="5,10,20,30")

    # Causal downstream pass.
    p.add_argument("--skip-causal", action="store_true")
    p.add_argument(
        "--causal-n",
        type=int,
        default=80,
        help="Number of COCO samples for actual downstream intervention.",
    )
    p.add_argument(
        "--causal-top-k",
        type=int,
        default=8,
        help="Top K from each selection metric added to causal shortlist.",
    )
    p.add_argument(
        "--causal-head-batch",
        type=int,
        default=4,
        help="How many head interventions to evaluate in one repeated-image batch.",
    )
    p.add_argument("--causal-seed", type=int, default=20260912)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).lower().replace("l", "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"No layers parsed from {text!r}")
    return sorted(out)


def parse_int_list(text: str) -> List[int]:
    return sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})


def parse_known_heads(text: str) -> List[Tuple[int, int]]:
    out = []
    for item in str(text).split(","):
        item = item.strip().upper().replace("L", "").replace("H", ":")
        if not item:
            continue
        while "::" in item:
            item = item.replace("::", ":")
        if ":" not in item:
            raise ValueError(f"Bad head {item!r}; expected L:H")
        a, b = item.split(":", 1)
        out.append((int(a), int(b)))
    return out


def hname(L: int, h: int) -> str:
    return f"L{int(L)}H{int(h):02d}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v)
    return v / n


def average_ranks(values: Sequence[float], descending: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if descending:
        x = -x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)  # one-indexed average rank
        ranks[order[i:j]] = rank
        i = j
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= EPS:
        return float("nan")
    return float(np.dot(x, y) / den)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    return pearson(average_ranks(x), average_ranks(y))


def topk_set(rows: Sequence[Mapping[str, Any]], field: str, k: int) -> set:
    valid = [r for r in rows if np.isfinite(float(r.get(field, float("nan"))))]
    valid = sorted(
        valid,
        key=lambda r: (
            -float(r[field]),
            int(r["layer"]),
            int(r["head_index"]),
        ),
    )
    return {
        (int(r["layer"]), int(r["head_index"]))
        for r in valid[: min(int(k), len(valid))]
    }


def make_gray(real: Image.Image, value: int) -> Image.Image:
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real.size, (v, v, v))


def norm_rel(v: Any) -> str:
    s = str(v).strip().lower().replace("_", " ").replace("-", " ")
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    if "above" in s or "over" in s or s == "on":
        return "above"
    if "below" in s or "under" in s:
        return "below"
    raise ValueError(f"Cannot normalize relation {v!r}")


def relation_from_coord(h: float, v: float) -> str:
    scores = {
        "left": -float(h),
        "right": float(h),
        "above": float(v),
        "below": -float(v),
    }
    return max(REL, key=lambda r: (scores[r], -REL.index(r)))


def stratified_subset(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in rows]
    if n <= 0 or n >= len(rows):
        return rows

    rng = random.Random(int(seed))
    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_rel[str(r["relation"])].append(r)
    for rel in REL:
        rng.shuffle(by_rel[rel])

    out = []
    while len(out) < int(n):
        moved = False
        for rel in REL:
            if len(out) >= int(n):
                break
            if by_rel[rel]:
                out.append(by_rel[rel].pop())
                moved = True
        if not moved:
            break
    rng.shuffle(out)
    return out


# ============================================================================
# Synthetic residual-space basis
# ============================================================================

def load_state_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            definition = (
                str(z["vector_definition"].item())
                if "vector_definition" in keys
                else "relation_vectors"
            )
        elif {"img", "no_image"}.issubset(keys):
            X = (
                np.asarray(z["img"], dtype=np.float32)
                - np.asarray(z["no_image"], dtype=np.float32)
            )
            definition = "img_minus_no_image"
        else:
            raise RuntimeError(f"Bad NPZ keys: {sorted(keys)}")

        if "decoder_block_index" not in keys:
            raise RuntimeError("NPZ missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]

        if "relation" not in keys:
            raise RuntimeError("NPZ missing relation")
        y = np.asarray(
            [norm_rel(v) for v in z["relation"].tolist()],
            dtype=object,
        )

    if X.ndim != 3:
        raise RuntimeError(f"Expected [N,L,D], got {X.shape}")
    if len(y) != len(X):
        raise RuntimeError("relation length mismatch")
    return X, y, layers, definition


def fit_spatial_basis(
    X: np.ndarray,
    y: np.ndarray,
    source_layers: Sequence[int],
    wanted_layers: Sequence[int],
):
    lmap = {int(L): i for i, L in enumerate(source_layers)}
    basis = {}
    rows = []

    for L in wanted_layers:
        if int(L) not in lmap:
            raise RuntimeError(f"Source cache missing requested L{L}")

        Xf = np.asarray(X[:, lmap[int(L)]], dtype=np.float64)
        center = Xf.mean(axis=0)
        mus = {r: Xf[y == r].mean(axis=0) for r in REL}
        dirs = {r: unit(mus[r] - center) for r in REL}

        dH = unit(dirs["right"] - dirs["left"])
        dV = unit(dirs["above"] - dirs["below"])

        gapH = float(np.dot(mus["right"] - mus["left"], dH))
        gapV = float(np.dot(mus["above"] - mus["below"], dV))
        if gapH < 0:
            dH, gapH = -dH, -gapH
        if gapV < 0:
            dV, gapV = -dV, -gapV

        B = np.stack([dH, dV], axis=1)  # [D,2]
        gram = B.T @ B
        cond = float(np.linalg.cond(gram))
        if not np.isfinite(cond) or cond > 1e8:
            raise RuntimeError(f"Ill-conditioned H/V basis at L{L}: {cond}")

        dual = B @ np.linalg.inv(gram)
        Q, _ = np.linalg.qr(B)
        Q = Q[:, :2]

        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)

        basis[int(L)] = {
            "Q": Q.astype(np.float32),
            "dual": dual.astype(np.float32),
            "halfH": float(halfH),
            "halfV": float(halfV),
            "hidden_dim": int(Xf.shape[1]),
        }
        rows.append(
            {
                "layer": int(L),
                "source_N": int(len(Xf)),
                "hidden_dim": int(Xf.shape[1]),
                "axis_H_dot_axis_V": float(np.dot(dH, dV)),
                "full_gap_H": float(gapH),
                "full_gap_V": float(gapV),
                "half_gap_H": float(halfH),
                "half_gap_V": float(halfV),
                "gram_condition_number": cond,
            }
        )

    return basis, rows


# ============================================================================
# Data / model helpers
# ============================================================================

def load_coco_rows(args: argparse.Namespace):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(args.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(args.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    rows = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        try:
            rel = traj.normalize_relation(base, p["answer_raw"])
        except Exception:
            rel = norm_rel(p["answer_raw"])
        if rel not in REL:
            continue
        rows.append(
            {
                "sid": sid,
                "relation": rel,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "question_text": str(p["question_text"]),
            }
        )

    rows.sort(key=lambda r: int(r["sid"]))
    if args.max_samples and int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
        keep = {int(r["sid"]) for r in rows}
        rec_by_sid = {k: v for k, v in rec_by_sid.items() if k in keep}

    return rows, rec_by_sid, two


def load_accuracy_csv(path: Optional[str]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warning] accuracy CSV not found; continuing without it: {p}")
        return {}

    out = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            if row.get("layer") not in (None, "") and row.get("head_index") not in (None, ""):
                key = (int(float(row["layer"])), int(float(row["head_index"])))
            elif row.get("head"):
                text = row["head"].upper().replace("L", "").replace("H", ":")
                a, b = text.split(":", 1)
                key = (int(a), int(b))
            else:
                continue

            keep = {}
            for k, v in row.items():
                if k in {"head", "layer", "head_index"}:
                    continue
                if v is None or v == "":
                    continue
                try:
                    keep[k] = float(v)
                except Exception:
                    keep[k] = v
            out[key] = keep
    return out


def get_text_config(model: Any) -> Any:
    cfg = getattr(model, "config", None)
    for c in (
        getattr(cfg, "text_config", None),
        getattr(cfg, "language_config", None),
        cfg,
    ):
        if c is not None and getattr(c, "num_attention_heads", None) is not None:
            return c
    raise RuntimeError("Could not resolve text config")


def resolve_attn(layer: Any) -> Any:
    for n in ("self_attn", "attention", "attn"):
        x = getattr(layer, n, None)
        if x is not None:
            return x
    raise RuntimeError("Could not resolve attention module")


def resolve_o_proj(attn: Any) -> torch.nn.Module:
    for n in ("o_proj", "out_proj", "proj"):
        x = getattr(attn, n, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Could not resolve o_proj")


def span_positions(span: Sequence[int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


def locate_positions(
    processor: Any,
    input_ids: Sequence[int],
    subject: str,
    reference: str,
) -> Tuple[List[int], List[int]]:
    ss, rr = base.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    return span_positions(ss), span_positions(rr)


def pool_pair_from_hidden(
    x: torch.Tensor,
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
) -> torch.Tensor:
    """
    x: [B,T,D] -> [B,D]
    """
    T = int(x.shape[1])
    ss = [int(p) for p in spos if 0 <= int(p) < T]
    rr = [int(p) for p in rpos if 0 <= int(p) < T]
    if not ss or not rr:
        raise RuntimeError("No valid subject/reference positions")
    if pool == "last":
        return x[:, ss[-1], :] - x[:, rr[-1], :]
    return x[:, ss, :].mean(dim=1) - x[:, rr, :].mean(dim=1)


# ============================================================================
# First pass clean capture
# ============================================================================

class CleanLocalCapture:
    """
    For each selected layer captures:
      - decoder-layer INPUT subject-reference pair (pre-attention residual)
      - pre-W_O head subject-reference pair [H,Dh]
    """

    def __init__(
        self,
        model: Any,
        decoder_layers: Sequence[Any],
        layers: Sequence[int],
        spos: Sequence[int],
        rpos: Sequence[int],
        pool: str,
    ):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.hidden_size = int(getattr(cfg, "hidden_size"))
        self.head_dim = self.hidden_size // self.n_heads
        self.spos = list(map(int, spos))
        self.rpos = list(map(int, rpos))
        self.pool = str(pool)

        self.pre_residual_pair: Dict[int, torch.Tensor] = {}
        self.head_pair: Dict[int, torch.Tensor] = {}
        self.handles = []

        for L in layers:
            layer = decoder_layers[int(L)]
            op = resolve_o_proj(resolve_attn(layer))

            def make_layer_pre(layer_idx: int):
                def hook(_module, inputs):
                    x = inputs[0]
                    pair = pool_pair_from_hidden(
                        x, self.spos, self.rpos, self.pool
                    )
                    self.pre_residual_pair[int(layer_idx)] = (
                        pair[0].detach().float().clone()
                    )
                return hook

            def make_o_pre(layer_idx: int):
                def hook(_module, inputs):
                    x = inputs[0]  # [1,T,H*Dh]
                    T = int(x.shape[1])
                    H = x[0].reshape(T, self.n_heads, self.head_dim)
                    ss = [p for p in self.spos if 0 <= p < T]
                    rr = [p for p in self.rpos if 0 <= p < T]
                    if self.pool == "last":
                        pair = H[ss[-1]] - H[rr[-1]]
                    else:
                        pair = H[ss].mean(dim=0) - H[rr].mean(dim=0)
                    self.head_pair[int(layer_idx)] = (
                        pair.detach().float().clone()
                    )
                return hook

            self.handles.append(
                layer.register_forward_pre_hook(make_layer_pre(int(L)))
            )
            self.handles.append(
                op.register_forward_pre_hook(make_o_pre(int(L)))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_clean_local_capture(
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    layers: Sequence[int],
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
):
    cap = CleanLocalCapture(
        model,
        decoder_layers,
        layers,
        spos,
        rpos,
        pool,
    )
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = False
        _ = model(**kw)

        missing_r = sorted(set(layers) - set(cap.pre_residual_pair))
        missing_h = sorted(set(layers) - set(cap.head_pair))
        if missing_r or missing_h:
            raise RuntimeError(
                f"Missing captures residual={missing_r}, head={missing_h}"
            )

        return {
            "pre_residual_pair": cap.pre_residual_pair,
            "head_pair": cap.head_pair,
            "n_heads": cap.n_heads,
            "head_dim": cap.head_dim,
            "hidden_size": cap.hidden_size,
        }
    finally:
        cap.close()


# ============================================================================
# Spatial coordinate helpers
# ============================================================================

def spatial_coord_torch(
    v: torch.Tensor,
    basis_t: Mapping[str, Any],
) -> torch.Tensor:
    """
    v [...,D] -> normalized H/V coordinates [...,2]
    """
    hv = v @ basis_t["dual"]
    scale = torch.tensor(
        [basis_t["halfH"], basis_t["halfV"]],
        device=v.device,
        dtype=v.dtype,
    )
    return hv / scale


def spatial_projection_energy_torch(
    v: torch.Tensor,
    basis_t: Mapping[str, Any],
) -> torch.Tensor:
    q = v @ basis_t["Q"]
    return (q * q).sum(dim=-1)


def cosine_2d(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ============================================================================
# Causal downstream intervention helpers
# ============================================================================

class CausalCleanTokenCapture:
    """
    Captures exact pre-W_O head vectors at selected subject/reference token
    positions for source layers, plus the pre-attention residual pair at the
    fixed readout layer.
    """

    def __init__(
        self,
        model: Any,
        decoder_layers: Sequence[Any],
        source_layers: Sequence[int],
        readout_layer: int,
        positions: Sequence[int],
        spos: Sequence[int],
        rpos: Sequence[int],
        pool: str,
    ):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.hidden_size = int(getattr(cfg, "hidden_size"))
        self.head_dim = self.hidden_size // self.n_heads

        self.positions = sorted(set(int(x) for x in positions))
        self.spos = list(map(int, spos))
        self.rpos = list(map(int, rpos))
        self.pool = str(pool)

        self.head_tokens: Dict[int, torch.Tensor] = {}
        self.readout_pair: Optional[torch.Tensor] = None
        self.handles = []

        for L in source_layers:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(L)]))

            def make_o_pre(layer_idx: int):
                def hook(_module, inputs):
                    x = inputs[0]
                    T = int(x.shape[1])
                    H = x[0].reshape(T, self.n_heads, self.head_dim)
                    pos = [p for p in self.positions if 0 <= p < T]
                    if len(pos) != len(self.positions):
                        raise RuntimeError(
                            f"Position out of range at L{layer_idx}: T={T}"
                        )
                    self.head_tokens[int(layer_idx)] = (
                        H[pos].detach().float().clone()
                    )  # [P,H,Dh]
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_o_pre(int(L)))
            )

        read_layer = decoder_layers[int(readout_layer)]

        def readout_pre(_module, inputs):
            x = inputs[0]
            pair = pool_pair_from_hidden(
                x, self.spos, self.rpos, self.pool
            )
            self.readout_pair = pair[0].detach().float().clone()

        self.handles.append(read_layer.register_forward_pre_hook(readout_pre))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_causal_clean_capture(
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    source_layers: Sequence[int],
    readout_layer: int,
    positions: Sequence[int],
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
):
    cap = CausalCleanTokenCapture(
        model=model,
        decoder_layers=decoder_layers,
        source_layers=source_layers,
        readout_layer=readout_layer,
        positions=positions,
        spos=spos,
        rpos=rpos,
        pool=pool,
    )
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = False
        _ = model(**kw)
        missing = sorted(set(source_layers) - set(cap.head_tokens))
        if missing:
            raise RuntimeError(f"Missing source token capture layers: {missing}")
        if cap.readout_pair is None:
            raise RuntimeError("Missing readout pair capture")
        return cap.head_tokens, cap.readout_pair
    finally:
        cap.close()


def repeat_batch(batch: Mapping[str, Any], n: int) -> Dict[str, Any]:
    """
    Repeat a single-example Qwen multimodal batch n times.

    Qwen2-VL commonly stores:
      input_ids / attention_mask : [1,T]
      image_grid_thw             : [1,3]
      pixel_values               : [num_patches, patch_dim]

    Therefore pixel_values does NOT have a leading batch dimension of 1.
    Repeating one image n times means concatenating its patch block n times
    while also repeating image_grid_thw n times.
    """
    out = {}
    patch_keys = {
        "pixel_values",
        "pixel_values_videos",
        "pixel_values_video",
    }

    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue

        if v.ndim == 0:
            out[k] = v
            continue

        if k in patch_keys:
            reps = [int(n)] + [1] * (v.ndim - 1)
            out[k] = v.repeat(*reps)
            continue

        if int(v.shape[0]) == 1:
            reps = [int(n)] + [1] * (v.ndim - 1)
            out[k] = v.repeat(*reps)
            continue

        # Unknown tensor with non-unit leading dimension. Refuse to guess.
        raise RuntimeError(
            f"Cannot safely repeat tensor {k} with shape {tuple(v.shape)}. "
            "Known Qwen patch tensors are handled explicitly."
        )

    return out


class InjectionAndReadout:
    """
    Adds a precomputed [B,T,D] residual injection after source-layer W_O and
    captures the subject-reference residual at readout-layer input.
    """

    def __init__(
        self,
        decoder_layers: Sequence[Any],
        source_layer: int,
        readout_layer: int,
        injection: torch.Tensor,
        spos: Sequence[int],
        rpos: Sequence[int],
        pool: str,
    ):
        self.injection = injection
        self.spos = list(map(int, spos))
        self.rpos = list(map(int, rpos))
        self.pool = str(pool)
        self.readout_pair: Optional[torch.Tensor] = None
        self.handles = []

        op = resolve_o_proj(resolve_attn(decoder_layers[int(source_layer)]))

        def o_post(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"Unexpected o_proj output type {type(output).__name__}"
                )
            inj = self.injection.to(
                device=output.device,
                dtype=output.dtype,
            )
            if tuple(inj.shape) != tuple(output.shape):
                raise RuntimeError(
                    f"Injection shape {tuple(inj.shape)} != "
                    f"o_proj output {tuple(output.shape)}"
                )
            return output + inj

        self.handles.append(op.register_forward_hook(o_post))

        read_layer = decoder_layers[int(readout_layer)]

        def read_pre(_module, inputs):
            x = inputs[0]
            self.readout_pair = pool_pair_from_hidden(
                x, self.spos, self.rpos, self.pool
            ).detach().float().clone()  # [B,D]

        self.handles.append(read_layer.register_forward_pre_hook(read_pre))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_injected_readout(
    model: Any,
    decoder_layers: Sequence[Any],
    batch_repeated: Mapping[str, Any],
    source_layer: int,
    readout_layer: int,
    injection: torch.Tensor,
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
) -> torch.Tensor:
    cap = InjectionAndReadout(
        decoder_layers=decoder_layers,
        source_layer=source_layer,
        readout_layer=readout_layer,
        injection=injection,
        spos=spos,
        rpos=rpos,
        pool=pool,
    )
    try:
        kw = dict(batch_repeated)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = False
        _ = model(**kw)
        if cap.readout_pair is None:
            raise RuntimeError("Missing injected readout capture")
        return cap.readout_pair
    finally:
        cap.close()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    source_layers = parse_layers(args.source_layers)
    readout_layer = int(args.readout_layer)
    known_heads = parse_known_heads(args.known_heads)
    topk_overlap = parse_int_list(args.top_k_overlap)

    if any(L >= readout_layer for L in source_layers):
        raise ValueError(
            f"All source layers must be < readout layer {readout_layer}; "
            f"got {source_layers}"
        )

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    Xsrc, ysrc, src_layers, src_definition = load_state_npz(
        Path(args.source_spatial_npz)
    )
    wanted_basis_layers = sorted(set(source_layers + [readout_layer]))
    basis, basis_rows = fit_spatial_basis(
        Xsrc,
        ysrc,
        src_layers,
        wanted_basis_layers,
    )

    coco_rows, rec_by_sid, two = load_coco_rows(args)
    prior_acc = load_accuracy_csv(args.accuracy_csv)

    specs = base.merged_model_specs(two)
    spec = specs[args.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    model = processor = None
    errors: List[Dict[str, Any]] = []

    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        cfg = get_text_config(model)
        n_heads = int(cfg.num_attention_heads)
        hidden_size = int(getattr(cfg, "hidden_size"))
        head_dim = hidden_size // n_heads

        for L in wanted_basis_layers:
            if L < 0 or L >= len(decoder_layers):
                raise ValueError(f"Bad layer L{L}")
            if int(basis[L]["hidden_dim"]) != hidden_size:
                raise RuntimeError(
                    f"Basis L{L} dim={basis[L]['hidden_dim']} "
                    f"!= model hidden={hidden_size}"
                )

        device = torch.device(args.device)

        # W_O reshaped into per-head blocks.
        Wo: Dict[int, torch.Tensor] = {}
        for L in source_layers:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(L)]))
            W = op.weight.detach().float()
            if int(W.shape[1]) != n_heads * head_dim:
                raise RuntimeError(
                    f"L{L} W_O input dim mismatch: {tuple(W.shape)}"
                )
            Wo[L] = W.reshape(int(W.shape[0]), n_heads, head_dim)

        # Basis tensors.
        Bt = {}
        for L in wanted_basis_layers:
            Bt[L] = {
                "Q": torch.from_numpy(basis[L]["Q"]).to(
                    device=device, dtype=torch.float32
                ),
                "dual": torch.from_numpy(basis[L]["dual"]).to(
                    device=device, dtype=torch.float32
                ),
                "halfH": float(basis[L]["halfH"]),
                "halfV": float(basis[L]["halfV"]),
            }

        nL = len(source_layers)
        li = {L: i for i, L in enumerate(source_layers)}

        # ------------------------------------------------------------------
        # First pass accumulators.
        # ------------------------------------------------------------------
        # Per-layer pre/full.
        pre_coord_sq_sum = np.zeros(nL, dtype=np.float64)
        pre_coord_norm_sum = np.zeros(nL, dtype=np.float64)
        full_coord_sq_sum = np.zeros(nL, dtype=np.float64)
        full_coord_norm_sum = np.zeros(nL, dtype=np.float64)
        layer_N = np.zeros(nL, dtype=np.int64)

        # Per-head.
        shape = (nL, n_heads)
        N = np.zeros(shape, dtype=np.int64)

        write_coord_sq_sum = np.zeros(shape, dtype=np.float64)
        write_coord_norm_sum = np.zeros(shape, dtype=np.float64)
        write_spatial_energy_sum = np.zeros(shape, dtype=np.float64)
        write_total_energy_sum = np.zeros(shape, dtype=np.float64)

        post_coord_sq_sum = np.zeros(shape, dtype=np.float64)
        post_coord_norm_sum = np.zeros(shape, dtype=np.float64)
        sample_gain_sum = np.zeros(shape, dtype=np.float64)
        sample_gain_positive_N = np.zeros(shape, dtype=np.int64)

        align_sum = np.zeros(shape, dtype=np.float64)
        align_valid_N = np.zeros(shape, dtype=np.int64)

        gt_write_correct = np.zeros(shape, dtype=np.int64)
        gt_local_post_correct = np.zeros(shape, dtype=np.int64)

        print("=" * 150)
        print("ALL-HEAD SPATIAL WRITE vs LOCAL POST-STATE")
        print("=" * 150)
        print(f"model={args.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(f"source_layers={source_layers}, readout_layer={readout_layer}")
        print(f"heads/layer={n_heads}, head_dim={head_dim}, hidden={hidden_size}")
        print(f"COCO N={len(coco_rows)}")
        print(f"source cache={args.source_spatial_npz}")
        print(f"source definition={src_definition}")
        print(
            "Non-accuracy rankings use NO COCO GT. "
            "Prior accuracy is joined only for comparison."
        )
        print()

        for row in tqdm(coco_rows, desc="PASS1 write/post metrics"):
            real = gray = rb = gb = None
            try:
                rec = rec_by_sid[int(row["sid"])]
                real = base.record_image(rec)
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray(real, args.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=str(row["question_text"]),
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=str(row["question_text"]),
                    device=device,
                )

                rids = rb["input_ids"][0].detach().cpu().tolist()
                gids = gb["input_ids"][0].detach().cpu().tolist()
                rs, rr = locate_positions(
                    processor,
                    rids,
                    str(row["subject"]),
                    str(row["reference"]),
                )
                gs, gr = locate_positions(
                    processor,
                    gids,
                    str(row["subject"]),
                    str(row["reference"]),
                )

                rc = run_clean_local_capture(
                    model,
                    decoder_layers,
                    rb,
                    source_layers,
                    rs,
                    rr,
                    args.pool,
                )
                gc_ = run_clean_local_capture(
                    model,
                    decoder_layers,
                    gb,
                    source_layers,
                    gs,
                    gr,
                    args.pool,
                )

                gt = str(row["relation"])

                for L in source_layers:
                    i = li[L]

                    r_pre = (
                        rc["pre_residual_pair"][L]
                        - gc_["pre_residual_pair"][L]
                    )  # [D]

                    q_head = (
                        rc["head_pair"][L]
                        - gc_["head_pair"][L]
                    )  # [H,Dh]

                    # Head write after each head's own W_O block.
                    delta = torch.einsum(
                        "hd,ohd->ho",
                        q_head,
                        Wo[L],
                    )  # [H,D]

                    # Synthetic-normalized spatial coordinates.
                    pre_coord = spatial_coord_torch(
                        r_pre[None, :],
                        Bt[L],
                    )[0]  # [2]
                    write_coord = spatial_coord_torch(
                        delta,
                        Bt[L],
                    )  # [H,2]
                    post_coord = pre_coord[None, :] + write_coord
                    full_coord = pre_coord + write_coord.sum(dim=0)

                    pre_norm = float(torch.linalg.vector_norm(pre_coord).item())
                    full_norm = float(torch.linalg.vector_norm(full_coord).item())
                    write_norm = torch.linalg.vector_norm(
                        write_coord, dim=1
                    )
                    post_norm = torch.linalg.vector_norm(
                        post_coord, dim=1
                    )

                    # Residual-space purity of each write.
                    sp_e = spatial_projection_energy_torch(delta, Bt[L])
                    tot_e = (delta * delta).sum(dim=1)

                    pre_np = pre_coord.detach().cpu().numpy().astype(np.float64)
                    write_np = write_coord.detach().cpu().numpy().astype(np.float64)
                    post_np = post_coord.detach().cpu().numpy().astype(np.float64)
                    write_norm_np = write_norm.detach().cpu().numpy().astype(np.float64)
                    post_norm_np = post_norm.detach().cpu().numpy().astype(np.float64)
                    sp_e_np = sp_e.detach().cpu().numpy().astype(np.float64)
                    tot_e_np = tot_e.detach().cpu().numpy().astype(np.float64)

                    pre_coord_sq_sum[i] += pre_norm ** 2
                    pre_coord_norm_sum[i] += pre_norm
                    full_coord_sq_sum[i] += full_norm ** 2
                    full_coord_norm_sum[i] += full_norm
                    layer_N[i] += 1

                    write_coord_sq_sum[i] += write_norm_np ** 2
                    write_coord_norm_sum[i] += write_norm_np
                    write_spatial_energy_sum[i] += sp_e_np
                    write_total_energy_sum[i] += tot_e_np

                    post_coord_sq_sum[i] += post_norm_np ** 2
                    post_coord_norm_sum[i] += post_norm_np

                    gains = post_norm_np - pre_norm
                    sample_gain_sum[i] += gains
                    sample_gain_positive_N[i] += (gains > 0).astype(np.int64)

                    # Alignment in normalized H/V coordinate space.
                    for h in range(n_heads):
                        c = cosine_2d(pre_np, write_np[h])
                        if np.isfinite(c):
                            align_sum[i, h] += c
                            align_valid_N[i, h] += 1

                        pred_w = relation_from_coord(
                            write_np[h, 0], write_np[h, 1]
                        )
                        pred_p = relation_from_coord(
                            post_np[h, 0], post_np[h, 1]
                        )
                        if pred_w == gt:
                            gt_write_correct[i, h] += 1
                        if pred_p == gt:
                            gt_local_post_correct[i, h] += 1

                    N[i] += 1

            except Exception as exc:
                errors.append(
                    {
                        "phase": "pass1",
                        "sid": row.get("sid"),
                        "relation": row.get("relation"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()

        if int(N.sum()) <= 0:
            raise RuntimeError("No successful first-pass samples")

        # ------------------------------------------------------------------
        # Aggregate all-head metrics.
        # ------------------------------------------------------------------
        all_rows: List[Dict[str, Any]] = []
        layer_rows: List[Dict[str, Any]] = []

        for L in source_layers:
            i = li[L]
            n_layer = max(int(layer_N[i]), 1)
            pre_rms = math.sqrt(pre_coord_sq_sum[i] / n_layer)
            full_rms = math.sqrt(full_coord_sq_sum[i] / n_layer)

            layer_rows.append(
                {
                    "layer": int(L),
                    "N": int(layer_N[i]),
                    "pre_state_spatial_coord_rms": float(pre_rms),
                    "pre_state_mean_coord_norm": float(
                        pre_coord_norm_sum[i] / n_layer
                    ),
                    "full_all_heads_post_spatial_coord_rms": float(full_rms),
                    "full_all_heads_mean_coord_norm": float(
                        full_coord_norm_sum[i] / n_layer
                    ),
                    "full_minus_pre_rms_gain": float(full_rms - pre_rms),
                }
            )

            for h in range(n_heads):
                n = int(N[i, h])
                if n <= 0:
                    continue

                write_rms = math.sqrt(write_coord_sq_sum[i, h] / n)
                post_rms = math.sqrt(post_coord_sq_sum[i, h] / n)
                purity = float(
                    write_spatial_energy_sum[i, h]
                    / max(write_total_energy_sum[i, h], EPS)
                )
                mean_align = float(
                    align_sum[i, h] / max(int(align_valid_N[i, h]), 1)
                )

                row = {
                    "head": hname(L, h),
                    "layer": int(L),
                    "head_index": int(h),
                    "N": n,

                    # A. head write
                    "write_spatial_coord_rms": float(write_rms),
                    "write_mean_coord_norm": float(
                        write_coord_norm_sum[i, h] / n
                    ),
                    "write_energy_purity": purity,

                    # B. local post state
                    "pre_state_spatial_coord_rms": float(pre_rms),
                    "local_post_spatial_coord_rms": float(post_rms),
                    "local_post_rms_gain_vs_pre": float(post_rms - pre_rms),
                    "mean_sample_post_minus_pre_norm": float(
                        sample_gain_sum[i, h] / n
                    ),
                    "sample_gain_positive_rate": float(
                        sample_gain_positive_N[i, h] / n
                    ),
                    "mean_pre_write_spatial_alignment": mean_align,

                    # C. full all-head state reference
                    "full_all_heads_post_spatial_coord_rms": float(full_rms),
                    "full_all_heads_rms_gain_vs_pre": float(full_rms - pre_rms),

                    # GT diagnostics only
                    "gt_write_direction_accuracy": float(
                        gt_write_correct[i, h] / n
                    ),
                    "gt_local_post_direction_accuracy": float(
                        gt_local_post_correct[i, h] / n
                    ),
                }

                # Join prior accuracy scan if available.
                prior = prior_acc.get((int(L), int(h)), {})
                for k, v in prior.items():
                    row[f"prior_{k}"] = v

                all_rows.append(row)

        # Add ranks.
        rank_fields = {
            "write_strength_rank": "write_spatial_coord_rms",
            "write_purity_rank": "write_energy_purity",
            "local_post_strength_rank": "local_post_spatial_coord_rms",
            "local_post_rms_gain_rank": "local_post_rms_gain_vs_pre",
            "mean_sample_gain_rank": "mean_sample_post_minus_pre_norm",
            "alignment_rank": "mean_pre_write_spatial_alignment",
            "gt_write_acc_rank": "gt_write_direction_accuracy",
            "gt_local_post_acc_rank": "gt_local_post_direction_accuracy",
        }

        # Prior COCO accuracy gets a rank too if loaded.
        if any("prior_coco_in_accuracy" in r for r in all_rows):
            rank_fields["prior_coco_accuracy_rank_recomputed"] = (
                "prior_coco_in_accuracy"
            )

        for rank_name, metric in rank_fields.items():
            vals = [
                float(r.get(metric, float("nan")))
                for r in all_rows
            ]
            rr = average_ranks(
                [
                    (-1e30 if not np.isfinite(v) else v)
                    for v in vals
                ],
                descending=True,
            )
            for row, rank in zip(all_rows, rr):
                row[rank_name] = float(rank)

        def sorted_by(field: str):
            return sorted(
                all_rows,
                key=lambda r: (
                    -float(r.get(field, -1e30))
                    if np.isfinite(float(r.get(field, float("nan"))))
                    else 1e30,
                    int(r["layer"]),
                    int(r["head_index"]),
                ),
            )

        top_write = sorted_by("write_spatial_coord_rms")
        top_post = sorted_by("local_post_spatial_coord_rms")
        top_gain = sorted_by("local_post_rms_gain_vs_pre")
        top_sample_gain = sorted_by("mean_sample_post_minus_pre_norm")

        if any("prior_coco_in_accuracy" in r for r in all_rows):
            top_acc = sorted_by("prior_coco_in_accuracy")
        else:
            top_acc = []

        # ------------------------------------------------------------------
        # Metric correlation / overlap with prior accuracy.
        # ------------------------------------------------------------------
        corr_rows = []
        overlap_rows = []

        if top_acc:
            acc = np.asarray(
                [
                    float(r.get("prior_coco_in_accuracy", float("nan")))
                    for r in all_rows
                ],
                dtype=np.float64,
            )
            compare_metrics = (
                "write_spatial_coord_rms",
                "write_energy_purity",
                "local_post_spatial_coord_rms",
                "local_post_rms_gain_vs_pre",
                "mean_sample_post_minus_pre_norm",
                "sample_gain_positive_rate",
                "mean_pre_write_spatial_alignment",
                "gt_write_direction_accuracy",
                "gt_local_post_direction_accuracy",
            )

            for metric in compare_metrics:
                x = np.asarray(
                    [float(r.get(metric, float("nan"))) for r in all_rows],
                    dtype=np.float64,
                )
                corr_rows.append(
                    {
                        "metric": metric,
                        "reference": "prior_coco_in_accuracy",
                        "N_heads": int(
                            (np.isfinite(x) & np.isfinite(acc)).sum()
                        ),
                        "pearson": pearson(x, acc),
                        "spearman": spearman(x, acc),
                    }
                )

            for k in topk_overlap:
                A = topk_set(all_rows, "prior_coco_in_accuracy", k)
                for metric in (
                    "write_spatial_coord_rms",
                    "write_energy_purity",
                    "local_post_spatial_coord_rms",
                    "local_post_rms_gain_vs_pre",
                    "mean_sample_post_minus_pre_norm",
                ):
                    B = topk_set(all_rows, metric, k)
                    inter = sorted(A & B)
                    union = A | B
                    overlap_rows.append(
                        {
                            "K": int(k),
                            "metric": metric,
                            "accuracy_topK_N": len(A),
                            "metric_topK_N": len(B),
                            "intersection_N": len(inter),
                            "overlap_fraction_of_K": (
                                len(inter) / max(1, min(len(A), len(B)))
                            ),
                            "jaccard": (
                                len(inter) / max(1, len(union))
                            ),
                            "intersection_heads": ";".join(
                                hname(*x) for x in inter
                            ),
                        }
                    )

        # ------------------------------------------------------------------
        # Build causal shortlist.
        # ------------------------------------------------------------------
        shortlist_reason: Dict[Tuple[int, int], set] = defaultdict(set)

        def add_top(rows_sorted, reason: str, k: int):
            for r in rows_sorted[: int(k)]:
                key = (int(r["layer"]), int(r["head_index"]))
                shortlist_reason[key].add(reason)

        add_top(top_write, "top_write", args.causal_top_k)
        add_top(top_post, "top_local_post", args.causal_top_k)
        add_top(top_gain, "top_local_gain", args.causal_top_k)
        add_top(top_sample_gain, "top_mean_sample_gain", args.causal_top_k)
        if top_acc:
            add_top(top_acc, "top_prior_coco_acc", args.causal_top_k)

        for key in known_heads:
            if key[0] in source_layers and 0 <= key[1] < n_heads:
                shortlist_reason[key].add("known_head")

        shortlist = sorted(shortlist_reason)
        row_map = {
            (int(r["layer"]), int(r["head_index"])): r
            for r in all_rows
        }
        shortlist_rows = []
        for key in shortlist:
            r = dict(row_map[key])
            r["causal_selection_reasons"] = ";".join(
                sorted(shortlist_reason[key])
            )
            shortlist_rows.append(r)

        # Save cheap first-pass outputs immediately.
        write_csv(outdir / "all_head_metrics.csv", all_rows)
        write_csv(outdir / "top_by_write_strength.csv", top_write)
        write_csv(outdir / "top_by_local_post_strength.csv", top_post)
        write_csv(outdir / "top_by_local_gain.csv", top_gain)
        write_csv(outdir / "top_by_mean_sample_gain.csv", top_sample_gain)
        write_csv(outdir / "top_by_prior_coco_accuracy.csv", top_acc)
        write_csv(outdir / "metric_vs_accuracy_correlation.csv", corr_rows)
        write_csv(outdir / "topk_overlap_with_accuracy.csv", overlap_rows)
        write_csv(outdir / "per_layer_state_summary.csv", layer_rows)
        write_csv(outdir / "causal_shortlist.csv", shortlist_rows)
        write_csv(outdir / "spatial_basis.csv", basis_rows)

        # ------------------------------------------------------------------
        # Console first-pass tables.
        # ------------------------------------------------------------------
        def print_top(title: str, rows_sorted, metric: str, n: int):
            print()
            print("=" * 150)
            print(title)
            print("=" * 150)
            print(
                f"{'rank':>4} {'head':>8} {metric:>13} "
                f"{'write':>9} {'post':>9} {'gain':>9} "
                f"{'align':>8} {'COCOacc':>9}"
            )
            for rank, r in enumerate(rows_sorted[: int(n)], 1):
                accv = r.get("prior_coco_in_accuracy", float("nan"))
                print(
                    f"{rank:4d} {r['head']:>8} "
                    f"{float(r[metric]):13.5f} "
                    f"{float(r['write_spatial_coord_rms']):9.4f} "
                    f"{float(r['local_post_spatial_coord_rms']):9.4f} "
                    f"{float(r['local_post_rms_gain_vs_pre']):9.4f} "
                    f"{float(r['mean_pre_write_spatial_alignment']):8.4f} "
                    f"{float(accv):9.4f}"
                )

        print_top(
            "TOP BY HEAD SPATIAL WRITE",
            top_write,
            "write_spatial_coord_rms",
            args.top_print,
        )
        print_top(
            "TOP BY LOCAL POST-HEAD SPATIAL STATE",
            top_post,
            "local_post_spatial_coord_rms",
            args.top_print,
        )
        print_top(
            "TOP BY LOCAL POST-HEAD RMS GAIN",
            top_gain,
            "local_post_rms_gain_vs_pre",
            args.top_print,
        )
        print_top(
            "TOP BY MEAN PER-SAMPLE POST-PRE GAIN",
            top_sample_gain,
            "mean_sample_post_minus_pre_norm",
            args.top_print,
        )

        if top_acc:
            print()
            print("=" * 150)
            print("PRIOR HIGH-ACCURACY SPATIAL HEADS: DIRECT COMPARISON")
            print("=" * 150)
            print(
                f"{'rank':>4} {'head':>8} {'COCOacc':>9} "
                f"{'write':>9} {'post':>9} {'gain':>9} "
                f"{'align':>8} {'gtPost*':>9}"
            )
            for rank, r in enumerate(top_acc[: int(args.top_print)], 1):
                print(
                    f"{rank:4d} {r['head']:>8} "
                    f"{float(r['prior_coco_in_accuracy']):9.4f} "
                    f"{float(r['write_spatial_coord_rms']):9.4f} "
                    f"{float(r['local_post_spatial_coord_rms']):9.4f} "
                    f"{float(r['local_post_rms_gain_vs_pre']):9.4f} "
                    f"{float(r['mean_pre_write_spatial_alignment']):8.4f} "
                    f"{float(r['gt_local_post_direction_accuracy']):9.4f}"
                )

        print()
        print("=" * 150)
        print("PER-LAYER PRE vs FULL-ALL-HEAD POST SPATIAL STATE")
        print("=" * 150)
        print(
            f"{'L':>3} {'preRMS':>10} {'fullPost':>10} "
            f"{'gain':>10}"
        )
        for r in layer_rows:
            print(
                f"{int(r['layer']):3d} "
                f"{float(r['pre_state_spatial_coord_rms']):10.4f} "
                f"{float(r['full_all_heads_post_spatial_coord_rms']):10.4f} "
                f"{float(r['full_minus_pre_rms_gain']):10.4f}"
            )

        # ------------------------------------------------------------------
        # Actual downstream causal intervention pass.
        # ------------------------------------------------------------------
        causal_rows: List[Dict[str, Any]] = []

        if not args.skip_causal and shortlist:
            causal_samples = stratified_subset(
                coco_rows,
                args.causal_n,
                args.causal_seed,
            )
            candidate_by_layer: Dict[int, List[int]] = defaultdict(list)
            for L, h in shortlist:
                candidate_by_layer[int(L)].append(int(h))
            for L in candidate_by_layer:
                candidate_by_layer[L] = sorted(set(candidate_by_layer[L]))

            # Accumulators keyed by head.
            cN = defaultdict(int)
            c_coord_sq_sum = defaultdict(float)
            c_coord_norm_sum = defaultdict(float)
            c_gt_correct = defaultdict(int)
            c_h2 = defaultdict(float)
            c_v2 = defaultdict(float)

            print()
            print("=" * 150)
            print("ACTUAL DOWNSTREAM CAUSAL HEAD INJECTION")
            print("=" * 150)
            print(
                f"readout_layer=L{readout_layer}, "
                f"causal_samples={len(causal_samples)}, "
                f"candidate_heads={len(shortlist)}, "
                f"head_batch={args.causal_head_batch}"
            )
            print(
                "Intervention: start from GRAY computation and inject ONLY "
                "one head's clean REAL-GRAY pre-W_O contribution."
            )

            for row in tqdm(causal_samples, desc="PASS2 causal downstream"):
                real = gray = rb = gb = None
                try:
                    rec = rec_by_sid[int(row["sid"])]
                    real = base.record_image(rec)
                    if hasattr(real, "convert"):
                        real = real.convert("RGB")
                    gray = make_gray(real, args.gray_value)

                    rb = base.make_question_batch(
                        processor=processor,
                        image=real,
                        question_text=str(row["question_text"]),
                        device=device,
                    )
                    gb = base.make_question_batch(
                        processor=processor,
                        image=gray,
                        question_text=str(row["question_text"]),
                        device=device,
                    )

                    rids = rb["input_ids"][0].detach().cpu().tolist()
                    gids = gb["input_ids"][0].detach().cpu().tolist()
                    rs, rr = locate_positions(
                        processor,
                        rids,
                        str(row["subject"]),
                        str(row["reference"]),
                    )
                    gs, gr = locate_positions(
                        processor,
                        gids,
                        str(row["subject"]),
                        str(row["reference"]),
                    )

                    if rs != gs or rr != gr:
                        raise RuntimeError(
                            "REAL/GRAY object token positions differ; "
                            f"real={rs,rr}, gray={gs,gr}"
                        )

                    positions = sorted(set(rs + rr))
                    active_layers = sorted(candidate_by_layer)

                    real_tokens, _real_read = run_causal_clean_capture(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        source_layers=active_layers,
                        readout_layer=readout_layer,
                        positions=positions,
                        spos=rs,
                        rpos=rr,
                        pool=args.pool,
                    )
                    gray_tokens, gray_read = run_causal_clean_capture(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=gb,
                        source_layers=active_layers,
                        readout_layer=readout_layer,
                        positions=positions,
                        spos=gs,
                        rpos=gr,
                        pool=args.pool,
                    )

                    # Exact per-token real-minus-gray head activity.
                    token_delta = {
                        L: real_tokens[L] - gray_tokens[L]
                        for L in active_layers
                    }  # each [P,H,Dh]

                    T = int(gb["input_ids"].shape[1])
                    D = hidden_size
                    pos_to_local = {p: j for j, p in enumerate(positions)}
                    gt = str(row["relation"])

                    for L in active_layers:
                        heads = candidate_by_layer[L]
                        for start in range(
                            0,
                            len(heads),
                            max(1, int(args.causal_head_batch)),
                        ):
                            chunk = heads[
                                start : start + max(
                                    1, int(args.causal_head_batch)
                                )
                            ]
                            B = len(chunk)

                            # [B,T,D] sparse injection.
                            inj = torch.zeros(
                                (B, T, D),
                                device=device,
                                dtype=torch.float32,
                            )

                            W = Wo[L]  # [D,H,Dh]
                            td = token_delta[L]  # [P,H,Dh]

                            for b, h in enumerate(chunk):
                                for p in positions:
                                    j = pos_to_local[p]
                                    vec = torch.einsum(
                                        "d,od->o",
                                        td[j, h],
                                        W[:, h, :],
                                    )
                                    inj[b, p, :] = vec

                            gb_rep = repeat_batch(gb, B)
                            read_pairs = run_injected_readout(
                                model=model,
                                decoder_layers=decoder_layers,
                                batch_repeated=gb_rep,
                                source_layer=L,
                                readout_layer=readout_layer,
                                injection=inj,
                                spos=gs,
                                rpos=gr,
                                pool=args.pool,
                            )  # [B,D]

                            effect = (
                                read_pairs
                                - gray_read[None, :]
                            ).float()

                            coord = spatial_coord_torch(
                                effect,
                                Bt[readout_layer],
                            )  # [B,2]
                            coord_np = (
                                coord.detach().cpu().numpy().astype(np.float64)
                            )

                            for b, h in enumerate(chunk):
                                key = (int(L), int(h))
                                hv = coord_np[b]
                                norm = float(np.linalg.norm(hv))
                                cN[key] += 1
                                c_coord_sq_sum[key] += norm ** 2
                                c_coord_norm_sum[key] += norm
                                c_h2[key] += float(hv[0] ** 2)
                                c_v2[key] += float(hv[1] ** 2)
                                if relation_from_coord(hv[0], hv[1]) == gt:
                                    c_gt_correct[key] += 1

                except Exception as exc:
                    errors.append(
                        {
                            "phase": "causal",
                            "sid": row.get("sid"),
                            "relation": row.get("relation"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )
                finally:
                    if real is not None:
                        with contextlib.suppress(Exception):
                            real.close()
                    if gray is not None:
                        with contextlib.suppress(Exception):
                            gray.close()

            for key in shortlist:
                n = int(cN.get(key, 0))
                base_row = dict(row_map[key])
                base_row["causal_selection_reasons"] = ";".join(
                    sorted(shortlist_reason[key])
                )
                base_row["causal_readout_layer"] = int(readout_layer)
                base_row["causal_N"] = n
                if n > 0:
                    base_row["causal_downstream_spatial_coord_rms"] = float(
                        math.sqrt(c_coord_sq_sum[key] / n)
                    )
                    base_row["causal_downstream_mean_coord_norm"] = float(
                        c_coord_norm_sum[key] / n
                    )
                    base_row["causal_downstream_H_rms"] = float(
                        math.sqrt(c_h2[key] / n)
                    )
                    base_row["causal_downstream_V_rms"] = float(
                        math.sqrt(c_v2[key] / n)
                    )
                    base_row["causal_downstream_gt_direction_accuracy"] = float(
                        c_gt_correct[key] / n
                    )
                else:
                    base_row["causal_downstream_spatial_coord_rms"] = float("nan")
                    base_row["causal_downstream_mean_coord_norm"] = float("nan")
                    base_row["causal_downstream_H_rms"] = float("nan")
                    base_row["causal_downstream_V_rms"] = float("nan")
                    base_row["causal_downstream_gt_direction_accuracy"] = float("nan")
                causal_rows.append(base_row)

            causal_rows.sort(
                key=lambda r: (
                    -float(
                        r.get(
                            "causal_downstream_spatial_coord_rms",
                            -1e30,
                        )
                    )
                    if np.isfinite(
                        float(
                            r.get(
                                "causal_downstream_spatial_coord_rms",
                                float("nan"),
                            )
                        )
                    )
                    else 1e30,
                    int(r["layer"]),
                    int(r["head_index"]),
                )
            )

            # Causal rank.
            valid_vals = [
                float(
                    r.get(
                        "causal_downstream_spatial_coord_rms",
                        float("nan"),
                    )
                )
                for r in causal_rows
            ]
            rr = average_ranks(
                [(-1e30 if not np.isfinite(v) else v) for v in valid_vals],
                descending=True,
            )
            for r, rank in zip(causal_rows, rr):
                r["causal_downstream_strength_rank"] = float(rank)

            causal_corr = []
            if any("prior_coco_in_accuracy" in r for r in causal_rows):
                x = [
                    float(
                        r.get(
                            "causal_downstream_spatial_coord_rms",
                            float("nan"),
                        )
                    )
                    for r in causal_rows
                ]
                y = [
                    float(r.get("prior_coco_in_accuracy", float("nan")))
                    for r in causal_rows
                ]
                causal_corr.append(
                    {
                        "metric": "causal_downstream_spatial_coord_rms",
                        "reference": "prior_coco_in_accuracy",
                        "N_heads": int(
                            (
                                np.isfinite(np.asarray(x))
                                & np.isfinite(np.asarray(y))
                            ).sum()
                        ),
                        "pearson": pearson(x, y),
                        "spearman": spearman(x, y),
                    }
                )

            write_csv(
                outdir / "causal_downstream_metrics.csv",
                causal_rows,
            )
            write_csv(
                outdir / "causal_vs_accuracy_correlation.csv",
                causal_corr,
            )

            print()
            print("=" * 150)
            print("CAUSAL DOWNSTREAM SPATIAL SURVIVAL RANKING")
            print("=" * 150)
            print(
                f"{'rank':>4} {'head':>8} {'downRMS':>9} "
                f"{'H-rms':>8} {'V-rms':>8} "
                f"{'GTacc*':>8} {'COCOacc':>9} "
                f"{'write':>8} {'localGain':>10} reasons"
            )
            for rank, r in enumerate(
                causal_rows[: int(args.top_print)],
                1,
            ):
                print(
                    f"{rank:4d} {r['head']:>8} "
                    f"{float(r.get('causal_downstream_spatial_coord_rms', float('nan'))):9.4f} "
                    f"{float(r.get('causal_downstream_H_rms', float('nan'))):8.4f} "
                    f"{float(r.get('causal_downstream_V_rms', float('nan'))):8.4f} "
                    f"{float(r.get('causal_downstream_gt_direction_accuracy', float('nan'))):8.4f} "
                    f"{float(r.get('prior_coco_in_accuracy', float('nan'))):9.4f} "
                    f"{float(r['write_spatial_coord_rms']):8.4f} "
                    f"{float(r['local_post_rms_gain_vs_pre']):10.4f} "
                    f"{r['causal_selection_reasons']}"
                )

        else:
            write_csv(outdir / "causal_downstream_metrics.csv", [])
            write_csv(outdir / "causal_vs_accuracy_correlation.csv", [])

        # ------------------------------------------------------------------
        # Final save.
        # ------------------------------------------------------------------
        write_csv(outdir / "errors.csv", errors)

        config = vars(args).copy()
        config.update(
            {
                "resolved_source_layers": source_layers,
                "resolved_readout_layer": readout_layer,
                "source_vector_definition": src_definition,
                "source_N": int(len(Xsrc)),
                "source_hidden_dim": int(Xsrc.shape[-1]),
                "coco_requested_N": int(len(coco_rows)),
                "n_heads": n_heads,
                "head_dim": head_dim,
                "hidden_size": hidden_size,
                "prior_accuracy_heads_loaded": int(len(prior_acc)),
                "causal_shortlist_N": int(len(shortlist)),
                "ranking_uses_coco_gt": False,
            }
        )
        (outdir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print()
        print("=" * 150)
        print("FILES")
        print("=" * 150)
        print(outdir / "all_head_metrics.csv")
        print(outdir / "top_by_write_strength.csv")
        print(outdir / "top_by_local_post_strength.csv")
        print(outdir / "top_by_local_gain.csv")
        print(outdir / "top_by_prior_coco_accuracy.csv")
        print(outdir / "metric_vs_accuracy_correlation.csv")
        print(outdir / "topk_overlap_with_accuracy.csv")
        print(outdir / "per_layer_state_summary.csv")
        print(outdir / "causal_shortlist.csv")
        print(outdir / "causal_downstream_metrics.csv")
        print(outdir / "errors.csv")
        print()
        print(
            f"[success] all_heads={len(all_rows)} "
            f"causal_candidates={len(shortlist)} "
            f"errors={len(errors)}"
        )

    finally:
        if model is not None:
            try:
                del model
            except Exception:
                pass
        if processor is not None:
            try:
                del processor
            except Exception:
                pass
        cleanup()


if __name__ == "__main__":
    main()
