#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two Head Selection -- source-defined spatial basis, label-free candidate pool,
actual single-head causal injection, and downstream RMS ranking.

Derived from compare_spatial_head_write_post_downstream_v1.py (the user's original
experiment). Exact Qwen3B historical intervention: Gray + Real-Gray pre-W_O head
message at subject/ref token positions; read spatial pair at input of L27
(i.e. after L26). No target GT / readout accuracy used for selection.

Default: Qwen2.5-VL-3B; all 440 COCO-two QA examples; source Synthetic400;
Use --candidate-policy legacy23 to reproduce the historical N=80 23-head pool
on a different number of COCO examples (e.g. 20). This exploratory mode is
not label-free in its original pool construction, unlike the default.

source heads L16-L26, readout L27. Other supported aliases: qwen-7b,
llava-7b, llava-13b, internvl-1b (native InternVL2.5 backend; deferred smoke tests).

NOTE: full COCO selection is unsupervised on target inputs, NOT an independent
held-out model-selection test. Report any accuracy on an independent split to
avoid selection bias. 24GB GPUs may not hold 7B/13B checkpoints.

Usage:
  python -u eval_coco440_head_causal_rms_selection_v2.py \
    --model qwen-3b --phase all --max-samples 0 --causal-n 0 \
    --candidate-top-k 6 --causal-head-batch 1

Pilot:
  python -u eval_coco440_head_causal_rms_selection_v2.py \
    --model qwen-3b --phase all --max-samples 8 --causal-n 8 \
    --candidate-top-k 2 --causal-head-batch 1 \
    --output-dir output/head_rms_qwen3b_pilot8

All output files saved as per-sample NPZ checkpoints; subsequent runs resume.
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
        getattr(cfg, "llm_config", None),
        getattr(getattr(model, "language_model", None), "config", None),
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
    for n in ("o_proj", "out_proj", "wo", "W_o", "proj"):
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
        "image_flags",
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



# =============================================================================
# Revised v2: source reconstruction, model registry, resumable per-sample passes
# =============================================================================

SUPPORTED = ("qwen2-2b", "qwen-3b", "qwen-7b", "llava-7b", "llava-13b", "internvl-1b")

# Frozen candidates recovered from the historical N=80 causal experiment log.
# These 23 heads were chosen by a UNION that included prior COCO accuracy
# and known-head priors. This is a *historical reproduction* mode and NOT an
# unbiased, target-label-free head-selection protocol.
LEGACY_23_HEADS = ('L26H03', 'L21H05', 'L19H13', 'L19H08', 'L23H00', 'L22H14', 'L21H14', 'L23H01', 'L23H05', 'L20H08', 'L20H05', 'L21H01', 'L17H10', 'L17H11', 'L26H02', 'L16H01', 'L16H03', 'L16H02', 'L16H00', 'L16H07', 'L16H05', 'L16H15', 'L16H13')
LEGACY_80_DOWNSTREAM_RMS = {'L26H03': 0.2788, 'L21H05': 0.2482, 'L19H13': 0.21, 'L19H08': 0.1996, 'L23H00': 0.192, 'L22H14': 0.1831, 'L21H14': 0.1801, 'L23H01': 0.1426, 'L23H05': 0.1394, 'L20H08': 0.1265, 'L20H05': 0.1214, 'L21H01': 0.1207, 'L17H10': 0.1132, 'L17H11': 0.0681, 'L26H02': 0.0525, 'L16H01': 0.0325, 'L16H03': 0.0174, 'L16H02': 0.0143, 'L16H00': 0.013, 'L16H07': 0.0121, 'L16H05': 0.0118, 'L16H15': 0.0111, 'L16H13': 0.0101}

def parse_legacy_head_name(name: str) -> tuple[int, int]:
    m = __import__('re').fullmatch(r'L(\d+)H(\d+)', name)
    if not m:
        raise ValueError(f'Bad historical head name: {name}')
    return int(m.group(1)), int(m.group(2))



def new_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", choices=SUPPORTED, default="qwen-3b")
    p.add_argument("--phase", choices=("all", "source", "prescreen", "causal"), default="all")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--source-spatial-npz", default=None, help="Optional precomputed NPZ containing relation_vectors [N,L,D], relation, decoder_block_index")
    p.add_argument("--source-layers", default="auto", help="For Qwen3B: 16-26. Other models use 40-75%% depth; override explicitly")
    p.add_argument("--readout-layer", default="auto", help="For Qwen3B: L27 input (L26 output). Other models use source_max + 1")
    p.add_argument("--max-samples", type=int, default=0, help="COCO N=0 means all 440; first N for pilot")
    p.add_argument("--source-max-samples", type=int, default=0, help="Synthetic N=0 means all 400")
    p.add_argument("--causal-n", type=int, default=0, help="0 means every eligible COCO sample, otherwise fixed stratified subset")
    p.add_argument("--candidate-policy", choices=("unlabeled", "legacy23"), default="unlabeled",
                   help="legacy23 reproduces the exact historical 23-head candidate pool (previously COCO-accuracy-influenced); unlabeled is the clean prescreen")
    p.add_argument("--candidate-top-k", type=int, default=6, help="Top K from EACH GT-free write/post/gain prescreen metric")
    p.add_argument("--causal-head-batch", type=int, default=1, help="Batch of simultaneous head injections; 1 for 24GB GPU")
    p.add_argument("--selection-seed", type=int, default=20260912)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", choices=("mean", "last"), default="mean")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=("eager", "sdpa", "flash_attention_2", "none"), default="sdpa")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--accuracy-csv", default="", help="OPTIONAL previously evaluated readout accuracy CSV, joined AFTER ranking; NEVER used for selection")
    p.add_argument("--min-success-fraction", type=float, default=0.95)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Delete current output folder checkpoints (dangerous). Without this resumes.")
    # InternVL2.5 native settings, matching repo's native backend.
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--revision", default="main")
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument("--internvl-use-thumbnail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--control", default="gray", choices=("gray",))
    p.add_argument("--top-print", type=int, default=30)
    return p.parse_args()


def atomic_save_npz(path: Path, **arrays):
    """An interrupted Pod never leaves a misleading completed checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+".partial")
    with tmp.open("wb") as fout:
        np.savez_compressed(fout,**arrays)
    tmp.replace(path)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_error(path: Path, phase: str, sid: Any, e: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"phase": phase, "sid": sid, "error": f"{type(e).__name__}: {e}",
           "traceback": traceback.format_exc(limit=8)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ERROR] {phase} sid={sid}: {type(e).__name__}: {e}", flush=True)


def unlabeled_subset(rows: Sequence[Mapping[str,Any]], n: int, seed: int) -> List[dict]:
    """Fixed-seed subset using only sample ID; never GT relation."""
    rows=[dict(r) for r in rows]
    if n <= 0 or n >= len(rows):
        return rows
    rng=random.Random(int(seed))
    chosen=rng.sample(rows,int(n))
    return sorted(chosen,key=lambda r:int(r["sid"]))


def source_records(root: Path, nmax: int) -> List[Dict[str, Any]]:
    path = root / "labels.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing Synthetic400 source: {path}. Run generate_synthetic_shapes_4dir_v1.py first")
    records = []
    for k, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        img = Path(rec["image"])
        if not img.is_absolute():
            img = root / img
        if not img.exists():
            raise FileNotFoundError(img)
        relation = norm_rel(rec["relation"])
        sub, ref = str(rec["subject"]), str(rec["reference"])
        records.append({"sid": int(rec.get("id", k)), "relation": relation,
                        "image_path": str(img), "subject": sub, "reference": ref,
                        "question_text": f"Where is the {sub} in relation to the {ref}? Answer with left, right, above, or below."})
    records.sort(key=lambda x: x["sid"])
    return records[:nmax] if nmax else records


class ModelRuntime:
    """Shared HF forward and repo-native InternVL2.5 forward, with honest shape checks."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.native = args.model == "internvl-1b"
        self.backend = None
        self.processor = None
        self.tokenizer = None
        if self.native:
            # Native InternVL2.5, not InternVL3-HF alias found in older SPECS.
            import eval_synthetic_fixedwindow_centroid_5models_3datasets_v1 as native5
            self.native5 = native5
            self.backend = native5.load_backend("internvl-1b", args)
            self.model = self.backend.model
            self.layers = list(self.backend.layers)
            self.tokenizer = self.backend.tokenizer
            self.repo_id = self.backend.repo_id
        else:
            two = base.import_two_object_module()
            spec = base.merged_model_specs(two)[args.model]
            self.repo_id = spec.repo_id
            cls = getattr(transformers, spec.model_class)
            kw = {"torch_dtype": base.resolve_dtype(spec.dtype_name),
                  "low_cpu_mem_usage": True, "trust_remote_code": spec.trust_remote_code,
                  "device_map": {"": args.device}}
            if args.attn_impl != "none":
                kw["attn_implementation"] = args.attn_impl
            try:
                self.model = cls.from_pretrained(spec.repo_id, **kw)
            except TypeError:
                kw["dtype"] = kw.pop("torch_dtype")
                self.model = cls.from_pretrained(spec.repo_id, **kw)
            try:
                self.processor = AutoProcessor.from_pretrained(spec.repo_id,
                    trust_remote_code=spec.trust_remote_code, use_fast=False)
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(spec.repo_id,
                    trust_remote_code=spec.trust_remote_code)
            base.configure_processor(self.model, self.processor)
            self.layers, _ = base.resolve_decoder_layers(self.model)
            self.tokenizer = self.processor.tokenizer
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = list(self.layers)
        self.cfg = get_text_config(self.model)
        self.nheads = int(self.cfg.num_attention_heads)
        self.hidden_size = int(self.cfg.hidden_size)
        self.head_dim = self.hidden_size // self.nheads
        if self.hidden_size % self.nheads:
            raise RuntimeError("head_dim is not integral")
        # A full-width W_O is required for the exact pre-W_O per-head decomposition.
        for L in range(len(self.layers)):
            a = resolve_attn(self.layers[L])
            o = resolve_o_proj(a)
            if o.weight.ndim != 2 or tuple(o.weight.shape) != (self.hidden_size, self.hidden_size):
                raise RuntimeError(f"L{L} head output projection shape {tuple(o.weight.shape)} is not supported")
        print(f"[MODEL] {args.model} ({self.repo_id}), layers={len(self.layers)}, "
              f"heads/layer={self.nheads}, hidden={self.hidden_size}, device={args.device}", flush=True)

    def batch_and_positions(self, image: Image.Image, row: Mapping[str, Any]):
        question = str(row["question_text"])
        if self.native:
            n5 = self.native5
            pixels, layout = n5.internvl_pixels_and_layout(self.backend, image, self.args)
            _, ids, attention_mask, vis_positions = n5.internvl_chat_query(self.backend, question, int(pixels.shape[0]))
            raw_ids = ids[0].detach().cpu().tolist()
            start = max(vis_positions) + 1 if vis_positions else 0
            spos = n5.token_span_for_phrase(self.tokenizer, raw_ids, str(row["subject"]), start=start)
            rpos = n5.token_span_for_phrase(self.tokenizer, raw_ids, str(row["reference"]), start=start)
            batch = {"pixel_values": pixels, "input_ids": ids, "attention_mask": attention_mask,
                     "image_flags": torch.ones((int(pixels.shape[0]), 1), device=ids.device, dtype=torch.long)}
            return batch, spos, rpos
        batch = base.make_question_batch(
            processor=self.processor,
            image=image,
            question_text=question,
            device=torch.device(self.args.device),
        )
        ids = batch["input_ids"][0].detach().cpu().tolist()
        spos, rpos = locate_positions(self.processor, ids, str(row["subject"]), str(row["reference"]))
        if self.args.model.startswith("llava"):
            # LLaVA-1.5 may expand one <image> token into ~576 decoder tokens.
            # Resolve actual decoder sequence length using only the first-layer pre-hook.
            seen = {}
            class _LengthCaptured(Exception):
                pass
            def _capture_length(_module, inputs):
                seen["merged_len"] = int(inputs[0].shape[1])
                raise _LengthCaptured()
            handle = self.layers[0].register_forward_pre_hook(_capture_length)
            try:
                with torch.inference_mode():
                    kw = dict(batch)
                    kw.update(use_cache=False, output_attentions=False, return_dict=True)
                    try:
                        self.model(**kw)
                    except _LengthCaptured:
                        pass
            finally:
                handle.remove()
            if "merged_len" not in seen:
                raise RuntimeError("Unable to measure LLaVA merged decoder sequence length")
            self._last_decoder_length = seen["merged_len"]
            raw_len = len(ids)
            shift = seen["merged_len"] - raw_len
            if shift < 0:
                raise RuntimeError(f"Invalid LLaVA merge: {seen['merged_len']} < raw {raw_len}")
            if shift:
                image_id = getattr(self.model.config, "image_token_index", None)
                if image_id is not None:
                    locations = [j for j,t in enumerate(ids) if int(t)==int(image_id)]
                    if locations and min(spos+rpos) <= max(locations):
                        raise RuntimeError("LLaVA image marker follows object positions; cannot safely map")
                spos = [p+shift for p in spos]
                rpos = [p+shift for p in rpos]
                if max(spos+rpos) >= seen["merged_len"]:
                    raise RuntimeError("Mapped LLaVA object position outside decoder sequence")
        return batch, spos, rpos


def choose_layers(args: argparse.Namespace, total: int) -> Tuple[List[int], int]:
    if args.source_layers == "auto":
        if args.model == "qwen-3b":
            layers = list(range(16, 27))
        else:
            # Comparable middle-to-late window on models of different depths.
            low, high = round(total * 0.40), round(total * 0.73)
            layers = list(range(max(0, low), min(total - 2, high) + 1))
    else:
        layers = parse_layers(args.source_layers)
    readout = (27 if args.model == "qwen-3b" else max(layers) + 1) if args.readout_layer == "auto" else int(args.readout_layer)
    if not layers or min(layers) < 0 or max(layers) >= readout or readout + 1 >= total:
        raise ValueError(f"Layers invalid for {total} blocks: source={layers}, readout={readout}. "
                         "Readout L means the input of block L and requires block L+1 for source extraction")
    return layers, readout


def load_basis_from_cache(path: Path, layers: List[int]) -> Tuple[Dict[int, dict], List[dict]]:
    X, y, loaded_layers, definition = load_state_npz(path)
    if len(X) < 20:
        raise RuntimeError("Source basis has fewer than 20 samples")
    if any(not np.any(y == r) for r in REL):
        raise RuntimeError("Source basis missing one or more of left/right/above/below")
    print(f"[SOURCE BASIS] file={path} N={len(X)} definition={definition}")
    return fit_spatial_basis(X, y, loaded_layers, layers)


def extract_source_basis(rt: ModelRuntime, args: argparse.Namespace, requested: List[int], dest: Path, error_path: Path):
    source = source_records(Path(args.synthetic_dir), int(args.source_max_samples))
    sample_dir = dest.parent / "source_checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    n_layers = len(rt.layers)
    if max(requested) + 1 >= n_layers:
        raise ValueError("Need a subsequent decoder block to read each requested block output")
    head_layers = [L for L in requested if L < max(requested)]
    hooks = sorted(set([L+1 for L in requested] + head_layers))
    print(f"[SOURCE] Synthetic N={len(source)}, Real-Gray block outputs={requested}", flush=True)
    for row in tqdm(source, desc="SOURCE Real-Gray spatial basis"):
        sid = int(row["sid"])
        f = sample_dir / f"{sid:06d}.npz"
        if f.exists():
            continue
        try:
            with Image.open(row["image_path"]) as im:
                real = im.convert("RGB")
            gray = make_gray(real, args.gray_value)
            rb, rs, rr = rt.batch_and_positions(real, row)
            gb, gs, gr = rt.batch_and_positions(gray, row)
            with torch.inference_mode():
                rc = run_clean_local_capture(rt.model, rt.layers, rb, hooks, rs, rr, args.pool)
                gc_ = run_clean_local_capture(rt.model, rt.layers, gb, hooks, gs, gr, args.pool)
            arr = np.stack([(rc["pre_residual_pair"][L+1] - gc_["pre_residual_pair"][L+1]).cpu().numpy()
                            for L in requested], axis=0).astype(np.float32)
            hq = np.stack([(rc["head_pair"][L] - gc_["head_pair"][L]).cpu().numpy()
                           for L in head_layers], axis=0).astype(np.float16)
            atomic_save_npz(f, vectors=arr, head_vectors=hq, sid=sid, relation=row["relation"])
            real.close(); gray.close()
        except Exception as e:
            append_error(error_path, "source", sid, e)
            if args.fail_fast:
                raise
        finally:
            cleanup()
    X, y, Xheads, ids = [], [], [], []
    for row in source:
        f = sample_dir / f"{int(row['sid']):06d}.npz"
        if f.exists():
            with np.load(f) as z:
                X.append(z["vectors"])
                y.append(str(z["relation"].item()))
                ids.append(int(row["sid"]))
                if "head_vectors" in z:
                    Xheads.append(z["head_vectors"])
    if len(X) < max(20, math.ceil(len(source)*args.min_success_fraction)):
        raise RuntimeError(f"Source extraction incomplete: {len(X)}/{len(source)}; see {error_path}")
    atomic_save_npz(dest, relation_vectors=np.stack(X).astype(np.float16),
                        relation=np.asarray(y), decoder_block_index=np.asarray(requested),
                        vector_definition="synthetic400_real_minus_gray")
    if len(Xheads)==len(X):
        head_file=dest.parent/"synthetic400_real_gray_head_vectors.npz"
        atomic_save_npz(head_file,head_vectors=np.stack(Xheads),relation=np.asarray(y),
                            sid=np.asarray(ids),head_layers=np.asarray(head_layers),
                            definition="pre_WO_head_pair_real_minus_gray")
        print(f"[SOURCE] saved head readout cache: {head_file}", flush=True)
    print(f"[SOURCE] saved {dest} N={len(X)}", flush=True)


def pass1_scan(rt: ModelRuntime, args: argparse.Namespace, coco_rows, rec_by_sid,
               source_layers, Bt, outdir: Path, error_path: Path):
    folder = outdir / "pass1_local"
    folder.mkdir(parents=True, exist_ok=True)
    Wo = {L: resolve_o_proj(resolve_attn(rt.layers[L])).weight.detach().float().reshape(rt.hidden_size, rt.nheads, rt.head_dim)
          for L in source_layers}
    for row in tqdm(coco_rows, desc="PASS1 all-head local spatial scan"):
        sid = int(row["sid"])
        f = folder / f"{sid:06d}.npz"
        if f.exists():
            continue
        try:
            with Image.open(rec_by_sid[sid].image_path) as im:
                real = im.convert("RGB")
            gray = make_gray(real, args.gray_value)
            rb, rs, rr = rt.batch_and_positions(real, row)
            gb, gs, gr = rt.batch_and_positions(gray, row)
            if rt.native is False and (rs != gs or rr != gr):
                raise RuntimeError("Real/Gray token positions differ")
            rc = run_clean_local_capture(rt.model, rt.layers, rb, source_layers, rs, rr, args.pool)
            gc_ = run_clean_local_capture(rt.model, rt.layers, gb, source_layers, gs, gr, args.pool)
            pre_all, write_all, energy_sp, energy_total, full_all, head_all = [], [], [], [], [], []
            for L in source_layers:
                rp = rc["pre_residual_pair"][L] - gc_["pre_residual_pair"][L]
                hd = rc["head_pair"][L] - gc_["head_pair"][L]
                d = torch.einsum("hd,ohd->ho", hd, Wo[L])
                head_all.append(hd.detach().cpu().numpy().astype(np.float16))
                pre = spatial_coord_torch(rp[None], Bt[L])[0]
                w = spatial_coord_torch(d, Bt[L])
                full = pre + w.sum(0)
                pre_all.append(pre.detach().cpu().numpy())
                write_all.append(w.detach().cpu().numpy())
                energy_sp.append(spatial_projection_energy_torch(d, Bt[L]).detach().cpu().numpy())
                energy_total.append(d.square().sum(-1).detach().cpu().numpy())
                full_all.append(full.detach().cpu().numpy())
            atomic_save_npz(f, pre=np.stack(pre_all), write=np.stack(write_all),
                                spatial_energy=np.stack(energy_sp), total_energy=np.stack(energy_total),
                                full=np.stack(full_all),head_vectors=np.stack(head_all),
                                relation=row["relation"], sid=sid)
            real.close(); gray.close()
        except Exception as e:
            append_error(error_path, "prescreen", sid, e)
            if args.fail_fast:
                raise
        finally:
            cleanup()
    del Wo


def pass1_aggregate(args: argparse.Namespace, rows: List[dict], source_layers: List[int],
                    rt: ModelRuntime, outdir: Path):
    folder = outdir / "pass1_local"
    files = []
    gt = []
    for row in rows:
        f = folder / f"{int(row['sid']):06d}.npz"
        if f.exists():
            files.append(f)
            gt.append(row["relation"])
    if len(files) < max(1, math.ceil(len(rows) * args.min_success_fraction)):
        raise RuntimeError(f"Too few prescreen results: {len(files)}/{len(rows)}")
    nL, nH = len(source_layers), rt.nheads
    wsq = np.zeros((nL, nH)); psq = np.zeros((nL, nH))
    wmag = np.zeros((nL, nH)); gain = np.zeros((nL, nH))
    spE = np.zeros((nL, nH)); totE = np.zeros((nL, nH))
    gt_write_ok = np.zeros((nL, nH)); gt_post_ok = np.zeros((nL, nH))
    fullsq = np.zeros(nL); presq = np.zeros(nL)
    for f, rel in tqdm(zip(files, gt), total=len(files), desc="PASS1 aggregate"):
        with np.load(f) as z:
            pre = np.asarray(z["pre"], dtype=np.float64) # [L,2]
            w = np.asarray(z["write"], dtype=np.float64) # [L,H,2]
            full = np.asarray(z["full"], dtype=np.float64)
            e_sp = np.asarray(z["spatial_energy"], dtype=np.float64)
            e_total = np.asarray(z["total_energy"], dtype=np.float64)
        post = pre[:, None, :] + w
        wnorm2 = np.sum(w*w, axis=-1)
        postnorm2 = np.sum(post*post, axis=-1)
        pre_norm = np.linalg.norm(pre, axis=-1)
        wsq += wnorm2
        psq += postnorm2
        wmag += np.sqrt(wnorm2)
        gain += np.sqrt(postnorm2) - pre_norm[:, None]
        spE += e_sp
        totE += e_total
        fullsq += np.sum(full*full, axis=-1)
        presq += np.sum(pre*pre, axis=-1)
        for lidx in range(nL):
            for h in range(nH):
                gt_write_ok[lidx, h] += int(relation_from_coord(*w[lidx, h]) == rel)
                gt_post_ok[lidx, h] += int(relation_from_coord(*post[lidx, h]) == rel)
    n = len(files)
    allhead = []
    for j,L in enumerate(source_layers):
        pre_rms = math.sqrt(presq[j] / n)
        full_rms = math.sqrt(fullsq[j] / n)
        for h in range(nH):
            allhead.append({"head": hname(L,h), "layer":L, "head_index":h, "N":n,
                "write_spatial_coord_rms":math.sqrt(wsq[j,h]/n),
                "local_post_spatial_coord_rms":math.sqrt(psq[j,h]/n),
                "local_post_rms_gain_vs_pre":math.sqrt(psq[j,h]/n) - pre_rms,
                "mean_sample_post_minus_pre_norm":gain[j,h]/n,
                "write_mean_coord_norm":wmag[j,h]/n,
                "write_energy_purity":spE[j,h]/max(totE[j,h], EPS),
                "gt_write_direction_accuracy_audit_only":gt_write_ok[j,h]/n,
                "gt_local_post_direction_accuracy_audit_only":gt_post_ok[j,h]/n,
                "pre_state_spatial_coord_rms":pre_rms,
                "full_all_heads_post_spatial_coord_rms":full_rms})
    write_csv(outdir/"all_head_metrics.csv", allhead)
    metrics = {"top_write": "write_spatial_coord_rms",
               "top_local_post": "local_post_spatial_coord_rms",
               "top_local_gain": "local_post_rms_gain_vs_pre",
               "top_mean_sample_gain": "mean_sample_post_minus_pre_norm"}
    selected = {}
    for name,field in metrics.items():
        ranked = sorted(allhead, key=lambda r:(-r[field],r["layer"],r["head_index"]))
        write_csv(outdir/f"{name}.csv",ranked)
        for r in ranked[:args.candidate_top_k]:
            k=(int(r["layer"]),int(r["head_index"]))
            selected.setdefault(k, set()).add(name)
    idx = {(int(r["layer"]),int(r["head_index"])):r for r in allhead}
    if args.candidate_policy == "legacy23":
        # Historical candidates are frozen independently of the current 20
        # samples; downstream ranking is the SAME real-gray injection as v2.
        # These heads were not originally selected without COCO label influence.
        historical_keys = [parse_legacy_head_name(h) for h in LEGACY_23_HEADS]
        missing = [hname(*k) for k in historical_keys if k not in idx]
        if missing:
            raise ValueError(
                f"Historical Qwen3B L16-26 candidates missing: {missing}. "
                "legacy23 is currently supported only with --model qwen-3b "
                "--source-layers 16-26."
            )
        selected = {key: {"historical_fixed_pool"} for key in historical_keys}
        write_csv(outdir/"historical_N80_reference.csv", [
            {"head": h, "historical_N80_downstream_rms": LEGACY_80_DOWNSTREAM_RMS[h],
             "historical_rank": i+1} for i,h in enumerate(LEGACY_23_HEADS)
        ])
        print("[LEGACY23] Using FROZEN original 23 Head IDs (historical N=80 pool).",flush=True)
        print("[LEGACY23] This historical pool included COCO accuracy/known-head priors; diagnostic only.",flush=True)
    candidates=[]
    for L,h in sorted(selected):
        row=dict(idx[L,h])
        row["causal_selection_reasons"]=";".join(sorted(selected[L,h]))
        candidates.append(row)
    if not candidates:
        raise RuntimeError("Empty GT-free candidate pool")
    path=outdir/"causal_shortlist.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            old=[(int(r["layer"]),int(r["head_index"])) for r in csv.DictReader(f)]
        if old != [(r["layer"],r["head_index"]) for r in candidates]:
            raise RuntimeError("Candidate shortlist changed; refusing to reuse old causal results. Use a new --output-dir")
    write_csv(path,candidates)
    write_json(outdir/"selection_protocol.json",{
        "model":args.model,"prescreen_samples":n,"source":str(args.synthetic_dir),
        "used_coco_labels_for_prescreen":False,
        "used_prior_coco_accuracy":args.candidate_policy == "legacy23",
        "used_manual_known_heads":args.candidate_policy == "legacy23",
        "candidate_policy":args.candidate_policy,
        "historical_23_previously_coco_accuracy_influenced":args.candidate_policy == "legacy23",
        "candidate_top_k_each_metric":args.candidate_top_k,
        "candidate_union_n":len(candidates), "metric_names":metrics,
        "readout_accuracy_is_not_same_as_gt_write_direction_accuracy_audit_only":True})
    print(f"[PRESCREEN] n={n}, full_heads={len(allhead)}, candidate_union={len(candidates)}, policy={args.candidate_policy}",flush=True)
    print("[PRESCREEN] candidates:",", ".join(r["head"] for r in candidates))
    return candidates



def audit_source_head_readout(args:argparse.Namespace, coco_rows:List[dict], outdir:Path,
                             source_layers:List[int], n_heads:int):
    """Four-way head readout audit only: Synthetic-only frozen direction fitting;
    COCO labels used only to score predictions AFTER shortlist is frozen.
    No readout accuracy contributes to candidate selection/ranking.
    """
    src=outdir/"synthetic400_real_gray_head_vectors.npz"
    if not src.exists() and args.source_spatial_npz:
        src=Path(args.source_spatial_npz).parent/"synthetic400_real_gray_head_vectors.npz"
    if not src.exists():
        print("[AUDIT] Source head vector cache unavailable; skipping readout accuracy audit")
        return
    with np.load(src) as z:
        X=np.asarray(z["head_vectors"],dtype=np.float32)
        y=np.asarray(z["relation"]).astype(str)
        stored_layers=np.asarray(z["head_layers"],dtype=int).tolist()
    layeridx={int(L):i for i,L in enumerate(stored_layers)}
    if not set(source_layers).issubset(layeridx):
        raise RuntimeError("Source head vectors do not cover requested candidate layers")
    if len(X)<20 or X.shape[2]!=n_heads:
        raise RuntimeError("Invalid source head vector cache")
    # Source-only 4-way linear directional codebook (centered class means).
    code={}
    for L in source_layers:
        hX=np.asarray(X[:,layeridx[L]],dtype=np.float64) # [N,H,Dh]
        mu=hX.mean(axis=0)
        directions=[]
        for rel in REL:
            m=hX[y==rel].mean(axis=0)-mu
            m=m/np.maximum(np.linalg.norm(m,axis=-1,keepdims=True),1e-12)
            directions.append(m)
        code[L]=(mu,np.stack(directions,axis=1)) # [H,Dh],[H,4,Dh]
    n={(L,h):0 for L in source_layers for h in range(n_heads)}
    correct={(L,h):0 for L in source_layers for h in range(n_heads)}
    predrows=[]
    for row in tqdm(coco_rows,desc="AUDIT Synthetic->COCO head readout"):
        f=outdir/"pass1_local"/f"{int(row['sid']):06d}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            if "head_vectors" not in z:
                raise RuntimeError("Target pass1 cache was created by older version without head vectors")
            V=np.asarray(z["head_vectors"],dtype=np.float64)
        for j,L in enumerate(source_layers):
            mu,dirs=code[L]
            z=V[j]-mu
            z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
            scores=np.einsum("hd,hkd->hk",z,dirs)
            predictions=np.argmax(scores,axis=1)
            for h in range(n_heads):
                pred=REL[int(predictions[h])]
                correct[L,h]+=int(pred==row["relation"])
                n[L,h]+=1
                predrows.append({"sid":int(row["sid"]),"head":hname(L,h),"pred":pred,
                                 "correct":int(pred==row["relation"]),"gt":row["relation"]})
    results=[{"head":hname(L,h),"layer":L,"head_index":h,
              "source_frozen_readout_accuracy":correct[L,h]/max(1,n[L,h]),
              "N":n[L,h]} for L in source_layers for h in range(n_heads)]
    write_csv(outdir/"readout_accuracy_ALL_HEADS_AUDIT_ONLY.csv",results)
    # The per-sample CSV is compact enough for Qwen3b: 440*176 rows.
    write_csv(outdir/"readout_predictions_AUDIT_ONLY.csv",predrows)
    if (outdir/"head_ranking.csv").exists():
        accmap={(r["layer"],r["head_index"]):r for r in results}
        with (outdir/"head_ranking.csv").open(newline="",encoding="utf-8") as f:
            ranked=list(csv.DictReader(f))
        for r in ranked:
            v=accmap.get((int(r["layer"]),int(r["head_index"])))
            r["source_frozen_readout_accuracy"]=(None if v is None else v["source_frozen_readout_accuracy"])
        write_csv(outdir/"head_ranking_with_readout_AUDIT_ONLY.csv",ranked)
        aa=[r for r in ranked if r["source_frozen_readout_accuracy"] is not None]
        if len(aa)>=3:
            write_json(outdir/"causal_rms_vs_readout_accuracy_AUDIT_ONLY.json",{
                "heads":len(aa),
                "pearson":pearson([float(r["causal_downstream_spatial_coord_rms"]) for r in aa],
                                  [float(r["source_frozen_readout_accuracy"]) for r in aa]),
                "spearman":spearman([float(r["causal_downstream_spatial_coord_rms"]) for r in aa],
                                   [float(r["source_frozen_readout_accuracy"]) for r in aa]),
                "selection_used_GT":False,"readout_accuracy_audit_after_selection":True})
    print("[AUDIT] Head readout accuracy saved; never used to select candidates")


def pass2_causal(rt: ModelRuntime,args:argparse.Namespace,coco_rows,rec_by_sid,readout:int,
                 candidates:List[dict],Bt,outdir:Path,error_path:Path):
    folder=outdir/"pass2_causal"
    folder.mkdir(parents=True,exist_ok=True)
    by_layer=defaultdict(list)
    for r in candidates:
        by_layer[int(r["layer"])].append(int(r["head_index"]))
    active_layers=sorted(by_layer)
    Wo={L:resolve_o_proj(resolve_attn(rt.layers[L])).weight.detach().float().reshape(rt.hidden_size,rt.nheads,rt.head_dim)
        for L in active_layers}
    order=[(int(r["layer"]),int(r["head_index"])) for r in candidates]
    sample_rows=unlabeled_subset(coco_rows,args.causal_n,args.selection_seed)
    write_json(outdir/"causal_manifest.json",{
       "sids":[int(r["sid"]) for r in sample_rows], "head_order":[hname(*t) for t in order],
       "readout_block_input":readout,"source_block_output":readout-1,
       "selection_uses_coco_gt":False,"head_metric":"downstream_spatial_coord_rms",
       "causal_n_requested":len(sample_rows)})
    print(f"[CAUSAL] n={len(sample_rows)}, candidates={len(order)}, readout=L{readout} INPUT",flush=True)
    for row in tqdm(sample_rows,desc="PASS2 causal single-head Real-Gray injection"):
        sid=int(row["sid"])
        f=folder/f"{sid:06d}.npz"
        if f.exists():
            continue
        try:
            with Image.open(rec_by_sid[sid].image_path) as im:
                real=im.convert("RGB")
            gray=make_gray(real,args.gray_value)
            rb,rs,rr=rt.batch_and_positions(real,row)
            gb,gs,gr=rt.batch_and_positions(gray,row)
            if not rt.native and (rs!=gs or rr!=gr):
                raise RuntimeError("Real/Gray object positions differ")
            positions=sorted(set(rs+rr))
            rtoken,_=run_causal_clean_capture(rt.model,rt.layers,rb,active_layers,readout,positions,rs,rr,args.pool)
            gtoken,gread=run_causal_clean_capture(rt.model,rt.layers,gb,active_layers,readout,positions,gs,gr,args.pool)
            delta={L:rtoken[L]-gtoken[L] for L in active_layers}
            T=int(rt._last_decoder_length if args.model.startswith("llava") else gb["input_ids"].shape[1]); D=rt.hidden_size
            index={p:j for j,p in enumerate(positions)}
            scores=np.full((len(order),2),np.nan,dtype=np.float32)
            candidate_index={k:i for i,k in enumerate(order)}
            for L in active_layers:
                heads=by_layer[L]
                for start in range(0,len(heads),max(1,args.causal_head_batch)):
                    hs=heads[start:start+max(1,args.causal_head_batch)]
                    B=len(hs)
                    inj=torch.zeros((B,T,D),device=torch.device(args.device),dtype=torch.float32)
                    for bi,h in enumerate(hs):
                        for p in positions:
                            inj[bi,p]=torch.einsum("d,od->o",delta[L][index[p],h],Wo[L][:,h,:])
                    repeated=repeat_batch(gb,B)
                    read=run_injected_readout(rt.model,rt.layers,repeated,L,readout,inj,gs,gr,args.pool)
                    coord=spatial_coord_torch(read-gread[None],Bt[readout]).cpu().numpy()
                    for bi,h in enumerate(hs):
                        scores[candidate_index[L,h]]=coord[bi]
                    del inj,repeated,read,coord
            if not np.isfinite(scores).all():
                raise RuntimeError(f"Nonfinite causal head scores for sid={sid}")
            atomic_save_npz(f,scores=scores,sid=sid)
            real.close();gray.close()
        except Exception as e:
            append_error(error_path,"causal",sid,e)
            if args.fail_fast:
                raise
        finally:
            cleanup()


def pass2_aggregate(args,coco_rows,candidates,outdir:Path,readout:int):
    folder=outdir/"pass2_causal"
    sample_rows=unlabeled_subset(coco_rows,args.causal_n,args.selection_seed)
    files=[folder/f"{int(r['sid']):06d}.npz" for r in sample_rows]
    valid=[(f,r) for f,r in zip(files,sample_rows) if f.exists()]
    if len(valid)<max(1,math.ceil(len(files)*args.min_success_fraction)):
        raise RuntimeError(f"Causal outputs incomplete: {len(valid)}/{len(files)}; see errors.jsonl")
    total=np.zeros((len(candidates),2),dtype=np.float64)
    direction_ok=np.zeros(len(candidates),dtype=np.int64)
    for f,r in tqdm(valid,desc="PASS2 aggregate"):
        with np.load(f) as z:
            arr=np.asarray(z["scores"],dtype=np.float64)
        if arr.shape!=(len(candidates),2):
            raise ValueError(f"Bad cached causal array {f} shape={arr.shape}")
        total+=arr*arr
        for k,hv in enumerate(arr):
            direction_ok[k]+=int(relation_from_coord(*hv)==r["relation"])
    n=len(valid)
    out=[]
    for j,c in enumerate(candidates):
        r=dict(c)
        r.update({"causal_N":n,"causal_downstream_spatial_coord_rms":float(math.sqrt(total[j].sum()/n)),
                  "causal_downstream_H_rms":float(math.sqrt(total[j,0]/n)),
                  "causal_downstream_V_rms":float(math.sqrt(total[j,1]/n)),
                  "causal_gt_direction_acc_audit_only":float(direction_ok[j]/n)})
        out.append(r)
    out.sort(key=lambda r:(-r["causal_downstream_spatial_coord_rms"],r["layer"],r["head_index"]))
    for rank,r in enumerate(out,1):
        r["causal_rank"]=rank
    write_csv(outdir/"causal_downstream_metrics.csv",out)
    write_csv(outdir/"head_ranking.csv",out)
    # Optional external readout accuracy can ONLY be joined after the final causal ranking.
    if args.accuracy_csv:
        lookup=load_accuracy_csv(args.accuracy_csv)
        scored=[]
        for row in out:
            t=dict(row)
            for name,val in lookup.get((int(row["layer"]),int(row["head_index"])),{}).items():
                t["prior_"+name]=val
            scored.append(t)
        write_csv(outdir/"causal_ranking_with_accuracy_AUDIT_ONLY.csv",scored)
        validacc=[r for r in scored if isinstance(r.get("prior_coco_in_accuracy"),float)]
        if len(validacc)>=3:
            write_json(outdir/"causal_accuracy_correlation_AUDIT_ONLY.json",{
                "N":len(validacc),"pearson":pearson([x["causal_downstream_spatial_coord_rms"] for x in validacc],[x["prior_coco_in_accuracy"] for x in validacc]),
                "spearman":spearman([x["causal_downstream_spatial_coord_rms"] for x in validacc],[x["prior_coco_in_accuracy"] for x in validacc])})
    write_json(outdir/"result_summary.json",{
        "model":args.model,"prescreen_candidates":len(candidates),"causal_samples_valid":n,
        "causal_samples_requested":len(sample_rows),"readout_input_layer":int(readout),
        "winner":out[0]["head"],"winner_rms":out[0]["causal_downstream_spatial_coord_rms"],
        "warning":"Full-440 selection uses unlabeled target inputs; downstream predictive accuracy on the same inputs is not independent heldout validation."})
    print("\nRANK HEAD DOWNRMS   H-rms   V-rms  N",flush=True)
    for r in out[:args.top_print]:
        print(f"{r['causal_rank']:>4} {r['head']:>8} {r['causal_downstream_spatial_coord_rms']:>8.4f} "
              f"{r['causal_downstream_H_rms']:>8.4f} {r['causal_downstream_V_rms']:>8.4f} {r['causal_N']}",flush=True)
    return out


def main_v2():
    args=new_args()
    if args.output_dir is None:
        args.output_dir=f"output/coco440_causal_head_rms_{args.model.replace('-','_')}_v2"
    outdir=Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    if args.candidate_policy == "legacy23" and args.model != "qwen-3b":
        raise ValueError("Frozen historical 23-head pool is Qwen3B-only; for other models use --candidate-policy unlabeled")
    if args.causal_head_batch<1 or args.candidate_top_k<1:
        raise ValueError("candidate-top-k and causal-head-batch must be positive")
    if args.causal_n<0 or args.max_samples<0:
        raise ValueError("sample counts must be nonnegative")
    error_path=outdir/"errors.jsonl"
    rt=ModelRuntime(args)
    source_layers,readout=choose_layers(args,len(rt.layers))
    requested=sorted(set(source_layers+[readout]))
    write_json(outdir/"run_config.json",{
        **vars(args),"torch":torch.__version__,"transformers":transformers.__version__,
        "source_layers_resolved":source_layers,"readout_input_layer":readout,
        "readout_is_input_of_block_L":True,"corresponding_upstream_output":readout-1,
        "candidate_pool_excludes_GT_accuracy":args.candidate_policy == "unlabeled",
        "candidate_pool_excludes_manual_known_heads":args.candidate_policy == "unlabeled",
        "MIG_GPU":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    cache=Path(args.source_spatial_npz) if args.source_spatial_npz else outdir/"synthetic400_real_gray_basis.npz"
    if not cache.exists():
        extract_source_basis(rt,args,requested,cache,error_path)
    basis,basis_rows=load_basis_from_cache(cache,requested)
    write_csv(outdir/"source_spatial_basis.csv",basis_rows)
    Bt={L:{"Q":torch.from_numpy(basis[L]["Q"]).to(args.device,dtype=torch.float32),
           "dual":torch.from_numpy(basis[L]["dual"]).to(args.device,dtype=torch.float32),
           "halfH":basis[L]["halfH"],"halfV":basis[L]["halfV"]} for L in requested}
    if args.phase=="source":
        print("[DONE] source basis",cache)
        return
    coco_rows,rec_by_sid,_=load_coco_rows(args)
    if not coco_rows:
        raise RuntimeError("No COCO rows; check data/coco_qa_two_obj.json, data/val2017 and prompts")
    if len(coco_rows)!=440 and args.max_samples==0:
        raise RuntimeError(f"Full run must have exactly 440 COCO examples, found {len(coco_rows)}")
    if args.phase in ("all","prescreen"):
        pass1_scan(rt,args,coco_rows,rec_by_sid,source_layers,Bt,outdir,error_path)
    candidates=pass1_aggregate(args,coco_rows,source_layers,rt,outdir)
    if args.phase=="prescreen":
        audit_source_head_readout(args,coco_rows,outdir,source_layers,rt.nheads)
        print("[DONE] prescreen, candidates saved to",outdir/"causal_shortlist.csv")
        return
    pass2_causal(rt,args,coco_rows,rec_by_sid,readout,candidates,Bt,outdir,error_path)
    pass2_aggregate(args,coco_rows,candidates,outdir,readout)
    audit_source_head_readout(args,coco_rows,outdir,source_layers,rt.nheads)
    print("[DONE] ranking:",outdir/"head_ranking.csv")




# =============================================================================
# Two-stage, GT-free selection -- all window heads on 50; Top20 on 440.
# =============================================================================

import hashlib

STAGE_ALGORITHM_VERSION = "twostage-rms-v1"


def two_stage_args():
    p = argparse.ArgumentParser(
        description="50 unlabeled COCO samples x ALL window heads -> Top20 -> all 440 x Top20 -> Top1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=SUPPORTED)
    p.add_argument("--phase", choices=("all", "stage1", "stage2", "report"), default="all")
    p.add_argument("--calib-n", type=int, default=50)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--selection-seed", type=int, default=20260912)
    p.add_argument("--source-layers", default="auto", help="auto = decoder depth 30%%-70%%, independent of GT and known reader heads")
    p.add_argument("--readout-layer", default="auto", help="auto = last source layer + 1 (input of that decoder block)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--source-spatial-npz", default=None)
    p.add_argument("--source-max-samples", type=int, default=0)
    p.add_argument("--causal-head-batch", type=int, default=1)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", choices=("mean", "last"), default="mean")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=("eager", "sdpa", "flash_attention_2", "none"), default="sdpa")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--revision", default="main")
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument("--internvl-use-thumbnail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--control", choices=("gray",), default="gray")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-print", type=int, default=25)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-success-fraction", type=float, default=1.0)
    # Extra fields read by imported v3 helpers, not used as selection metrics.
    args = p.parse_args()
    args.max_samples = 0
    args.causal_n = args.calib_n
    args.accuracy_csv = ""
    args.candidate_policy = "unlabeled"
    args.candidate_top_k = 1
    args.overwrite = False
    if args.calib_n < 1 or args.calib_n > 440:
        p.error("--calib-n must be between 1 and 440")
    if args.top_k < 1 or args.causal_head_batch < 1:
        p.error("--top-k and --causal-head-batch must be positive")
    if not (0 < args.min_success_fraction <= 1):
        p.error("--min-success-fraction must be in (0,1]")
    if args.output_dir is None:
        args.output_dir = f"output/head_twostage_50_to_440_{args.model.replace('-', '_')}"
    return args


def fixed_source_window(args, total):
    if args.source_layers == "auto":
        # Protocol fixed in advance: normalized decoder-depth window [30%,70%].
        # This is NOT tuned to the previously selected target-GT reader head.
        low = max(0, int(math.floor(0.30 * total)))
        high = min(total - 3, int(math.floor(0.70 * total)))
        layers = list(range(low, high + 1))
    else:
        layers = parse_layers(args.source_layers)
    readout = max(layers) + 1 if args.readout_layer == "auto" else int(args.readout_layer)
    if not layers or min(layers) < 0 or max(layers) >= readout or readout + 1 >= total:
        raise ValueError(f"Invalid source={layers} readout={readout} for num_layers={total}")
    return layers, readout


def digest_obj(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def assert_write_manifest(path, obj):
    obj = dict(obj)
    obj["sha256"] = digest_obj(obj)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != obj:
            raise RuntimeError(f"Protocol changed for {path}; do not mix checkpoints. Use a NEW output directory.")
    else:
        write_json(path, obj)


def read_npz_score(path, expected_n):
    with np.load(path, allow_pickle=False) as z:
        a = np.asarray(z["scores"], dtype=np.float64)
    if a.shape != (expected_n, 2) or not np.isfinite(a).all():
        raise RuntimeError(f"Bad checkpoint {path}: shape={a.shape}, expected {(expected_n,2)}")
    return a


def rank_from_causal_files(outdir, sample_sids, candidates, label):
    headnames = [r["head"] for r in candidates]
    if len(set(headnames)) != len(headnames):
        raise RuntimeError("Duplicate candidate head")
    sq = np.zeros((len(candidates), 2), np.float64)
    for sid in sample_sids:
        file = outdir / "pass2_causal" / f"{sid:06d}.npz"
        if not file.is_file():
            raise RuntimeError(f"Missing {label} checkpoint {file}; run previous stage again")
        a = read_npz_score(file, len(candidates))
        sq += a*a
    n = len(sample_sids)
    result=[]
    for j,r in enumerate(candidates):
        result.append({"head":r["head"],"layer":int(r["layer"]),"head_index":int(r["head_index"]),
                       "N":n,"rms":float(np.sqrt(sq[j].sum()/n)),
                       "h_rms":float(np.sqrt(sq[j,0]/n)),"v_rms":float(np.sqrt(sq[j,1]/n))})
    result.sort(key=lambda r:(-r["rms"],r["layer"],r["head_index"]))
    for i,row in enumerate(result,1):
        row["rank"] = i
    write_csv(outdir / "head_ranking.csv", result)
    print(f"[{label}] scanned {n} samples x {len(candidates)} Heads",flush=True)
    print("RANK HEAD RMS H-RMS V-RMS",flush=True)
    for row in result[:min(25,len(result))]:
        print(f"{row['rank']:4d} {row['head']:>8} {row['rms']:.5f} {row['h_rms']:.5f} {row['v_rms']:.5f}",flush=True)
    return result


def selected_candidates(stage1_ranks, k):
    # Retain rank order; downstream caches use exactly this index order.
    return [{"head":r["head"],"layer":r["layer"],"head_index":r["head_index"]}
            for r in stage1_ranks[:k]]


def reuse_first_stage_scores(stage1_dir, stage2_dir, calib_sids, allheads, top20):
    pos = {r["head"]:j for j,r in enumerate(allheads)}
    idx = [pos[r["head"]] for r in top20]
    dest = stage2_dir / "pass2_causal"
    dest.mkdir(parents=True,exist_ok=True)
    copied=0
    for sid in calib_sids:
        source_file=stage1_dir/"pass2_causal"/f"{sid:06d}.npz"
        dest_file=dest/f"{sid:06d}.npz"
        source_score=read_npz_score(source_file,len(allheads))
        wanted=np.asarray(source_score[idx],dtype=np.float32)
        if dest_file.exists():
            existing=read_npz_score(dest_file,len(top20))
            if not np.allclose(existing,wanted,rtol=1e-5,atol=1e-6):
                raise RuntimeError(f"Stage-2 cached score disagrees with stage-1: {dest_file}")
        else:
            atomic_save_npz(dest_file,scores=wanted,sid=sid)
            copied+=1
    print(f"[CACHE] reused {len(calib_sids)} stage-1 calibration samples in Stage 2 (new files={copied})",flush=True)


def load_target_without_labels(args):
    """Load only target images/IDs/prompts. Do not inspect COCO answer labels."""
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(args.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(args.data_root), None)
    by_id = {int(r.sid): r for r in records}
    rows = []
    for sid in sorted(by_id):
        if sid not in prompts:
            continue
        prompt = prompts[sid]
        rows.append({"sid":sid,"subject":str(prompt["subject"]),
                     "reference":str(prompt["reference"]),
                     "question_text":str(prompt["question_text"])})
    return rows, by_id


def main_twostage():
    args = two_stage_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    # Do not load models at all for reporting completed rankings.
    if args.phase == "report":
        for st in ("stage1", "stage2"):
            f=root/st/"head_ranking.csv"
            print(f"{st}: {f} exists={f.exists()}")
            if f.exists():
                with f.open(newline="",encoding="utf-8") as fp:
                    for r in list(csv.DictReader(fp))[:5]:
                        print(f"  {r['rank']} {r['head']} {r['rms']}")
        return

    err = root/"errors.jsonl"
    rt = ModelRuntime(args)
    source_layers, readout = fixed_source_window(args, len(rt.layers))
    sample_rows, rec_by_sid = load_target_without_labels(args)
    if len(sample_rows) != 440:
        raise RuntimeError(f"Expected all 440 COCO-two examples, found {len(sample_rows)}")
    calib = unlabeled_subset(sample_rows,args.calib_n,args.selection_seed)
    calib_sids = [int(r["sid"]) for r in calib]
    all_sids=[int(r["sid"]) for r in sample_rows]
    allheads=[{"head":hname(L,h),"layer":L,"head_index":h}
              for L in source_layers for h in range(rt.nheads)]
    top_k = min(args.top_k,len(allheads))
    proto={"version":STAGE_ALGORITHM_VERSION,"model":args.model,"repo_id":rt.repo_id,
           "source_layers":source_layers,"readout_input":readout,"all_window_head_count":len(allheads),
           "calibration_sids":calib_sids,"all_sids":all_sids,"top_k":top_k,
           "gray_value":args.gray_value,"pool":args.pool,"attn_impl":args.attn_impl,
           "source_spatial_npz":str(args.source_spatial_npz or "AUTOGENERATE"),
           "synthetic_dir":args.synthetic_dir,"source_max_samples":args.source_max_samples,
           "spatial_basis_definition":"source_Synthetic400_real_gray",
           "selection_metric":"downstream_causal_spatial_RMS",
           "selection_target_GT_used":False,
           "calibration_seed":args.selection_seed}
    assert_write_manifest(root/"protocol.json",proto)
    print(f"[PROTOCOL] model={args.model}; source L{source_layers[0]}-L{source_layers[-1]}; "
          f"heads/layer={rt.nheads}; all heads={len(allheads)}; readout=L{readout} INPUT; "
          f"calib={len(calib_sids)}; final=440; top-k={top_k}",flush=True)

    basis_file=Path(args.source_spatial_npz) if args.source_spatial_npz else root/"synthetic400_real_gray_basis.npz"
    needed=sorted(set(source_layers+[readout]))
    if not basis_file.exists():
        print("[SOURCE] no precomputed Synthetic400 cache; extracting full source basis",flush=True)
        extract_source_basis(rt,args,needed,basis_file,err)
    basis,basis_rows=load_basis_from_cache(basis_file,needed)
    write_csv(root/"source_spatial_basis.csv",basis_rows)
    Bt={L:{"Q":torch.from_numpy(basis[L]["Q"]).to(args.device,dtype=torch.float32),
           "dual":torch.from_numpy(basis[L]["dual"]).to(args.device,dtype=torch.float32),
           "halfH":basis[L]["halfH"],"halfV":basis[L]["halfV"]} for L in needed}

    stage1=root/"stage1"
    stage2=root/"stage2"
    stage1.mkdir(exist_ok=True)
    stage2.mkdir(exist_ok=True)
    assert_write_manifest(stage1/"candidate_order.json",{
        "protocol":digest_obj(proto),"heads":[r["head"] for r in allheads]})
    stage1_args=argparse.Namespace(**vars(args))
    stage1_args.causal_n=args.calib_n
    if args.phase in ("all","stage1"):
        pass2_causal(rt,stage1_args,sample_rows,rec_by_sid,readout,allheads,Bt,stage1,err)
    stage1_rank=rank_from_causal_files(stage1,calib_sids,allheads,"STAGE1")
    shortlist=selected_candidates(stage1_rank,top_k)
    manifest={"protocol":digest_obj(proto),"top20_order":[r["head"] for r in shortlist]}
    assert_write_manifest(root/"frozen_top20.json",manifest)
    write_csv(root/"frozen_top20.csv",shortlist)
    if args.phase == "stage1":
        print("[DONE] Stage 1 complete. Run --phase stage2 to continue.",flush=True)
        return

    assert_write_manifest(stage2/"candidate_order.json",manifest)
    reuse_first_stage_scores(stage1,stage2,calib_sids,allheads,shortlist)
    stage2_args=argparse.Namespace(**vars(args))
    stage2_args.causal_n=0
    pass2_causal(rt,stage2_args,sample_rows,rec_by_sid,readout,shortlist,Bt,stage2,err)
    stage2_rank=rank_from_causal_files(stage2,all_sids,shortlist,"STAGE2")
    summary={"model":args.model,"full_window_head_count":len(allheads),"stage1_N":len(calib_sids),
             "stage2_N":len(all_sids),"TopK":top_k,"final_head":stage2_rank[0]["head"],
             "stage1_head":stage1_rank[0]["head"],"stage2_top1_rms":stage2_rank[0]["rms"],
             "selection_uses_COCO_GT":False,"readout_block_input":readout,
             "overlap_calibration_stage2":True,
             "qualification":"440 target inputs used in unlabeled final model selection; accuracy on same inputs is not held-out validation."}
    write_json(root/"result_summary.json",summary)
    print("[DONE] Top-1 Head:",stage2_rank[0]["head"],"RMS:",stage2_rank[0]["rms"],flush=True)


if __name__ == "__main__":
    main_twostage()
