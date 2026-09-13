#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
measure_head_spatial_strength_from_source_basis_v1.py

Purpose
=======
Identify spatial heads WITHOUT ranking heads by COCO relation accuracy.

The only labeled spatial supervision used for head identification is an
INDEPENDENT Synthetic-400 residual-space cache, which defines a per-layer
H/V spatial subspace.  On COCO, GT labels are NOT used to rank heads.

For decoder layer L and attention head h:

  1) capture the head's pre-W_O subject-reference vector on REAL image;
  2) subtract the same vector on a GRAY control image;
  3) apply the head's own block of W_O so the head contribution is mapped
     back into the common residual space;
  4) measure how much of that contribution lies inside the Synthetic-defined
     H/V spatial subspace for layer L.

This gives label-free-on-COCO head metrics:

  spatial_coord_rms
      RMS H/V coordinate magnitude, normalized by Synthetic class half-gaps.
      This is the default "spatial strength" ranking because it is comparable
      across layers in units of the source spatial geometry.

  spatial_rms_residual
      RMS norm of the head's projected contribution in residual units.

  energy_purity
      sum spatial energy / sum total head-write energy.
      "What fraction of this head's visual object-pair write is spatial?"

  mean_sample_purity
      Per-sample version of the same fraction.

  layer_spatial_share
      Fraction of the layer's summed per-head spatial-write energy carried by
      this head.

COCO GT is used ONLY after ranking as an optional diagnostic:
  gt_direction_accuracy_from_projection
This checks whether a head selected by spatial strength also points toward the
correct relation; it never affects the ranking.

The script also quantifies multi-head mixing:
  - sum of per-head spatial energies
  - spatial energy after all head contributions are summed
  - coherence ratio = whole-layer spatial energy / summed head spatial energy

coherence < 1 indicates cancellation among head spatial components;
coherence > 1 indicates constructive alignment.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u \
  measure_head_spatial_strength_from_source_basis_v1.py \
  --model qwen-3b \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --layers 16-27 \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --pool mean \
  --device cuda:0 \
  --output-dir output/qwen3b_head_spatial_strength_synbasis_coco440_v1 \
  --overwrite

Main outputs
============
  all_head_spatial_strength.csv
  top_by_spatial_strength.csv
  top_by_spatial_purity.csv
  per_layer_concentration.csv
  known_head_metrics.csv
  metric_vs_gt_validation_correlation.csv
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
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
        "Run this script from the AdaptVis llava16 repo root next to "
        "analyze_coco_centroid_generation_step1_v4.py and "
        "eval_coco_multilayer_relation_trajectory_repair_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
EPS = 1e-12


# ---------------------------------------------------------------------
# CLI / utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--layers", default="16-27")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--known-heads", default="23:1,23:5,26:3,19:13,20:8,23:0")
    p.add_argument("--top-print", type=int, default=30)
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
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
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
        rank = 0.5 * ((i + 1) + j)  # 1-indexed average rank
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


# ---------------------------------------------------------------------
# Synthetic residual-space H/V basis
# ---------------------------------------------------------------------

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
            raise RuntimeError(
                f"Bad NPZ keys in {path}: {sorted(keys)}"
            )

        if "decoder_block_index" not in keys:
            raise RuntimeError(f"{path} missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]

        if "relation" not in keys:
            raise RuntimeError(f"{path} missing relation labels")
        y = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)

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
    """
    Same H/V construction used in the recent layer-wise spatial diagnostics,
    plus:
      B      : [D,2] non-orthogonal H/V basis
      Q      : [D,2] orthonormal basis for the same 2-D span
      dual   : [D,2] dual basis for H/V coordinate readout
    """
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
            raise RuntimeError(f"Ill-conditioned H/V basis at L{L}: cond={cond}")

        dual = B @ np.linalg.inv(gram)
        Q, _ = np.linalg.qr(B)
        Q = Q[:, :2]

        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)

        basis[int(L)] = {
            "B": B.astype(np.float32),
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
                "basis_gram_condition_number": cond,
            }
        )

    return basis, rows


# ---------------------------------------------------------------------
# COCO data / model helpers
# ---------------------------------------------------------------------

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

    rows.sort(key=lambda x: int(x["sid"]))
    if args.max_samples and int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
        keep = {int(x["sid"]) for x in rows}
        rec_by_sid = {k: v for k, v in rec_by_sid.items() if k in keep}

    return rows, rec_by_sid, two


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
    for name in ("self_attn", "attention", "attn"):
        obj = getattr(layer, name, None)
        if obj is not None:
            return obj
    raise RuntimeError("Could not resolve attention module")


def resolve_o_proj(attn: Any) -> torch.nn.Module:
    for name in ("o_proj", "out_proj", "proj"):
        obj = getattr(attn, name, None)
        if isinstance(obj, torch.nn.Module):
            return obj
    raise RuntimeError("Could not resolve attention output projection")


def span_positions(span: Sequence[int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


def locate_positions(
    processor: Any,
    input_ids: Sequence[int],
    subject: str,
    reference: str,
):
    ss, rr = base.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    return span_positions(ss), span_positions(rr)


class PairHeadCapture:
    """
    Captures only the subject-reference pair vector for every head at selected
    layers, immediately before W_O.  This avoids copying full [T,H,D] tensors.
    """

    def __init__(
        self,
        model: Any,
        decoder_layers: Sequence[Any],
        layers: Sequence[int],
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        pool: str,
    ):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)

        first_op = resolve_o_proj(resolve_attn(decoder_layers[int(layers[0])]))
        if self.hidden_size <= 0:
            self.hidden_size = int(first_op.in_features)
        if self.hidden_size % self.n_heads != 0:
            raise RuntimeError(
                f"hidden_size={self.hidden_size} not divisible by "
                f"num_heads={self.n_heads}"
            )
        self.head_dim = self.hidden_size // self.n_heads
        self.spos = [int(x) for x in subject_positions]
        self.rpos = [int(x) for x in reference_positions]
        self.pool = str(pool)
        self.pair: Dict[int, torch.Tensor] = {}
        self.handles = []

        for L in layers:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(L)]))

            def make_hook(layer_idx: int):
                def hook(_module, inputs):
                    x = inputs[0]
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        raise RuntimeError(
                            f"Unexpected pre-W_O tensor shape at L{layer_idx}: "
                            f"{tuple(x.shape)}"
                        )
                    T = int(x.shape[1])
                    ss = [p for p in self.spos if 0 <= p < T]
                    rr = [p for p in self.rpos if 0 <= p < T]
                    if not ss or not rr:
                        raise RuntimeError(
                            f"No valid object positions at L{layer_idx}: "
                            f"T={T}, subject={self.spos}, reference={self.rpos}"
                        )

                    H = x[0].reshape(T, self.n_heads, self.head_dim)
                    if self.pool == "last":
                        pair = H[ss[-1]] - H[rr[-1]]
                    else:
                        pair = H[ss].mean(dim=0) - H[rr].mean(dim=0)
                    self.pair[int(layer_idx)] = pair.detach().float().clone()

                return hook

            self.handles.append(op.register_forward_pre_hook(make_hook(int(L))))

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_pair_capture(
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    layers: Sequence[int],
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
):
    cap = PairHeadCapture(
        model=model,
        decoder_layers=decoder_layers,
        layers=layers,
        subject_positions=spos,
        reference_positions=rpos,
        pool=pool,
    )
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = False
        _ = model(**kw)
        if set(cap.pair) != set(int(x) for x in layers):
            missing = sorted(set(int(x) for x in layers) - set(cap.pair))
            raise RuntimeError(f"Missing captured layers: {missing}")
        return cap.pair, cap.n_heads, cap.head_dim, cap.hidden_size
    finally:
        cap.close()


def relation_from_coord(h: float, v: float) -> str:
    scores = {
        "left": -float(h),
        "right": float(h),
        "above": float(v),
        "below": -float(v),
    }
    return max(REL, key=lambda r: (scores[r], -REL.index(r)))


# ---------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    layers = parse_layers(args.layers)
    known_heads = parse_known_heads(args.known_heads)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    Xsrc, ysrc, src_layers, src_definition = load_state_npz(
        Path(args.source_spatial_npz)
    )
    basis, basis_rows = fit_spatial_basis(
        Xsrc, ysrc, src_layers, layers
    )

    coco_rows, rec_by_sid, two = load_coco_rows(args)

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
    errors = []

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

        bad = [L for L in layers if L < 0 or L >= len(decoder_layers)]
        if bad:
            raise ValueError(
                f"Invalid layers {bad}; decoder blocks are 0.."
                f"{len(decoder_layers)-1}"
            )
        for L in layers:
            if int(basis[L]["hidden_dim"]) != hidden_size:
                raise RuntimeError(
                    f"Source basis L{L} hidden_dim={basis[L]['hidden_dim']} "
                    f"but model hidden_size={hidden_size}"
                )

        device = torch.device(args.device)

        # Pre-cache each W_O in [out_dim, H, head_dim] form on GPU.
        Wo = {}
        for L in layers:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(L)]))
            W = op.weight.detach().float()
            if int(W.shape[1]) != n_heads * head_dim:
                raise RuntimeError(
                    f"L{L} W_O input dim {W.shape[1]} != "
                    f"heads*head_dim {n_heads*head_dim}"
                )
            Wo[int(L)] = W.reshape(int(W.shape[0]), n_heads, head_dim)

        # Basis tensors on GPU.
        Bt = {}
        for L in layers:
            Bt[int(L)] = {
                "Q": torch.from_numpy(basis[L]["Q"]).to(device=device, dtype=torch.float32),
                "dual": torch.from_numpy(basis[L]["dual"]).to(device=device, dtype=torch.float32),
                "halfH": float(basis[L]["halfH"]),
                "halfV": float(basis[L]["halfV"]),
            }

        nL = len(layers)
        li = {L: i for i, L in enumerate(layers)}

        # Per-head accumulators [L,H]
        sum_total_e = np.zeros((nL, n_heads), dtype=np.float64)
        sum_spatial_e = np.zeros((nL, n_heads), dtype=np.float64)
        sum_sample_purity = np.zeros((nL, n_heads), dtype=np.float64)
        sum_h2 = np.zeros((nL, n_heads), dtype=np.float64)
        sum_v2 = np.zeros((nL, n_heads), dtype=np.float64)
        sum_abs_h = np.zeros((nL, n_heads), dtype=np.float64)
        sum_abs_v = np.zeros((nL, n_heads), dtype=np.float64)
        gt_correct = np.zeros((nL, n_heads), dtype=np.int64)
        count = np.zeros((nL, n_heads), dtype=np.int64)

        # Whole-layer accumulators.
        whole_total_e = np.zeros(nL, dtype=np.float64)
        whole_spatial_e = np.zeros(nL, dtype=np.float64)
        whole_sum_head_spatial_e = np.zeros(nL, dtype=np.float64)
        whole_count = np.zeros(nL, dtype=np.int64)

        print("=" * 150)
        print("HEAD SPATIAL STRENGTH FROM INDEPENDENT SYNTHETIC H/V BASIS")
        print("=" * 150)
        print(f"model={args.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(f"layers={layers}")
        print(f"heads/layer={n_heads}, head_dim={head_dim}, hidden={hidden_size}")
        print(f"COCO N={len(coco_rows)}")
        print(f"source cache={args.source_spatial_npz}")
        print(f"source vector definition={src_definition}")
        print(
            "Ranking uses NO COCO GT.  COCO GT is diagnostic only after ranking."
        )
        print(
            f"Random 2-D subspace expected energy purity ~= "
            f"{2.0/hidden_size:.6f}"
        )
        print()

        for row in tqdm(coco_rows, desc="MEASURE head spatial strength"):
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

                real_pair, H1, D1, hidden1 = run_pair_capture(
                    model, decoder_layers, rb, layers, rs, rr, args.pool
                )
                gray_pair, H2, D2, hidden2 = run_pair_capture(
                    model, decoder_layers, gb, layers, gs, gr, args.pool
                )
                if (H1, D1, hidden1) != (H2, D2, hidden2):
                    raise RuntimeError("REAL/GRAY capture shape mismatch")

                gt = str(row["relation"])

                for L in layers:
                    i = li[L]
                    q = real_pair[L] - gray_pair[L]  # [H,Dh]

                    # Individual head contributions after each head's own W_O block:
                    # contribution[h, o] = sum_d q[h,d] * W_O[o,h,d]
                    contrib = torch.einsum(
                        "hd,ohd->ho",
                        q,
                        Wo[L],
                    )  # [H, hidden]

                    Q = Bt[L]["Q"]              # [hidden,2], orthonormal
                    dual = Bt[L]["dual"]        # [hidden,2]

                    # Projection energy onto the 2-D spatial span.
                    qcoord = contrib @ Q        # [H,2]
                    sp_e = (qcoord * qcoord).sum(dim=1)
                    tot_e = (contrib * contrib).sum(dim=1)
                    sample_purity = sp_e / torch.clamp(tot_e, min=EPS)

                    # H/V coordinates in Synthetic half-gap units.
                    hv = contrib @ dual         # [H,2]
                    hv[:, 0] /= float(Bt[L]["halfH"])
                    hv[:, 1] /= float(Bt[L]["halfV"])

                    sp_e_np = sp_e.detach().cpu().numpy().astype(np.float64)
                    tot_e_np = tot_e.detach().cpu().numpy().astype(np.float64)
                    pur_np = sample_purity.detach().cpu().numpy().astype(np.float64)
                    hv_np = hv.detach().cpu().numpy().astype(np.float64)

                    sum_spatial_e[i] += sp_e_np
                    sum_total_e[i] += tot_e_np
                    sum_sample_purity[i] += pur_np
                    sum_h2[i] += hv_np[:, 0] ** 2
                    sum_v2[i] += hv_np[:, 1] ** 2
                    sum_abs_h[i] += np.abs(hv_np[:, 0])
                    sum_abs_v[i] += np.abs(hv_np[:, 1])
                    count[i] += 1

                    # GT validation only -- never used in ranking.
                    for h in range(n_heads):
                        pred = relation_from_coord(hv_np[h, 0], hv_np[h, 1])
                        if pred == gt:
                            gt_correct[i, h] += 1

                    # Multi-head mixing / cancellation diagnostic.
                    whole = contrib.sum(dim=0)            # [hidden]
                    whole_q = whole @ Q                   # [2]
                    whole_sp_e = float((whole_q * whole_q).sum().item())
                    whole_tot_e = float((whole * whole).sum().item())

                    whole_spatial_e[i] += whole_sp_e
                    whole_total_e[i] += whole_tot_e
                    whole_sum_head_spatial_e[i] += float(sp_e.sum().item())
                    whole_count[i] += 1

            except Exception as exc:
                errors.append(
                    {
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

        if int(count.sum()) == 0:
            raise RuntimeError("No successful head measurements")

        # Aggregate head metrics.
        rows = []
        random_purity = 2.0 / float(hidden_size)

        for L in layers:
            i = li[L]
            layer_spatial_total = float(sum_spatial_e[i].sum())
            if layer_spatial_total <= EPS:
                layer_spatial_total = EPS

            for h in range(n_heads):
                n = int(count[i, h])
                if n <= 0:
                    continue

                mean_total_e = float(sum_total_e[i, h] / n)
                mean_spatial_e = float(sum_spatial_e[i, h] / n)
                energy_purity = float(
                    sum_spatial_e[i, h] / max(sum_total_e[i, h], EPS)
                )
                mean_sample_purity = float(sum_sample_purity[i, h] / n)
                rms_h = float(math.sqrt(sum_h2[i, h] / n))
                rms_v = float(math.sqrt(sum_v2[i, h] / n))
                spatial_coord_rms = float(
                    math.sqrt((sum_h2[i, h] + sum_v2[i, h]) / n)
                )
                balance = float(
                    2.0 * min(rms_h, rms_v) / max(rms_h + rms_v, EPS)
                )
                share = float(sum_spatial_e[i, h] / layer_spatial_total)

                rows.append(
                    {
                        "head": hname(L, h),
                        "layer": int(L),
                        "head_index": int(h),
                        "N": n,
                        "spatial_coord_rms": spatial_coord_rms,
                        "rms_H_coord_halfgap_units": rms_h,
                        "rms_V_coord_halfgap_units": rms_v,
                        "HV_balance": balance,
                        "mean_abs_H_coord": float(sum_abs_h[i, h] / n),
                        "mean_abs_V_coord": float(sum_abs_v[i, h] / n),
                        "spatial_rms_residual": float(math.sqrt(mean_spatial_e)),
                        "total_rms_residual": float(math.sqrt(mean_total_e)),
                        "energy_purity": energy_purity,
                        "mean_sample_purity": mean_sample_purity,
                        "purity_enrichment_over_random_2D": float(
                            energy_purity / max(random_purity, EPS)
                        ),
                        "layer_spatial_share": share,
                        "gt_direction_accuracy_from_projection": float(
                            gt_correct[i, h] / n
                        ),
                        "gt_direction_correct_N": int(gt_correct[i, h]),
                    }
                )

        # Global ranks.
        for metric, field in (
            ("spatial_strength_rank", "spatial_coord_rms"),
            ("spatial_rms_rank", "spatial_rms_residual"),
            ("purity_rank", "energy_purity"),
            ("gt_validation_rank", "gt_direction_accuracy_from_projection"),
        ):
            vals = [float(r[field]) for r in rows]
            rr = average_ranks(vals, descending=True)
            for r, rank in zip(rows, rr):
                r[metric] = float(rank)

        # Within-layer share rank.
        for L in layers:
            subset = [r for r in rows if int(r["layer"]) == int(L)]
            if not subset:
                continue
            rr = average_ranks(
                [float(r["layer_spatial_share"]) for r in subset],
                descending=True,
            )
            for r, rank in zip(subset, rr):
                r["layer_spatial_share_rank"] = float(rank)

        rows_by_strength = sorted(
            rows,
            key=lambda r: (
                -float(r["spatial_coord_rms"]),
                -float(r["energy_purity"]),
                int(r["layer"]),
                int(r["head_index"]),
            ),
        )
        rows_by_purity = sorted(
            rows,
            key=lambda r: (
                -float(r["energy_purity"]),
                -float(r["spatial_coord_rms"]),
                int(r["layer"]),
                int(r["head_index"]),
            ),
        )

        # Per-layer concentration and mixing.
        layer_rows = []
        for L in layers:
            i = li[L]
            subset = [r for r in rows if int(r["layer"]) == int(L)]
            if not subset:
                continue

            shares = np.asarray(
                [float(r["layer_spatial_share"]) for r in subset],
                dtype=np.float64,
            )
            shares = shares / max(float(shares.sum()), EPS)
            ss = np.sort(shares)[::-1]
            effective_n = float(1.0 / max(float(np.sum(shares ** 2)), EPS))
            entropy_n = float(
                math.exp(
                    -float(
                        np.sum(
                            shares
                            * np.log(np.maximum(shares, EPS))
                        )
                    )
                )
            )

            mean_whole_sp = float(
                whole_spatial_e[i] / max(int(whole_count[i]), 1)
            )
            mean_whole_tot = float(
                whole_total_e[i] / max(int(whole_count[i]), 1)
            )
            mean_sum_head_sp = float(
                whole_sum_head_spatial_e[i] / max(int(whole_count[i]), 1)
            )
            coherence = float(
                mean_whole_sp / max(mean_sum_head_sp, EPS)
            )
            whole_purity = float(
                mean_whole_sp / max(mean_whole_tot, EPS)
            )

            best_strength = max(
                subset, key=lambda r: float(r["spatial_coord_rms"])
            )
            best_share = max(
                subset, key=lambda r: float(r["layer_spatial_share"])
            )

            layer_rows.append(
                {
                    "layer": int(L),
                    "N_samples": int(whole_count[i]),
                    "N_heads": len(subset),
                    "best_strength_head": best_strength["head"],
                    "best_strength_spatial_coord_rms": float(
                        best_strength["spatial_coord_rms"]
                    ),
                    "best_share_head": best_share["head"],
                    "best_share": float(best_share["layer_spatial_share"]),
                    "top1_spatial_share": float(ss[:1].sum()),
                    "top3_spatial_share": float(ss[:3].sum()),
                    "top5_spatial_share": float(ss[:5].sum()),
                    "effective_spatial_head_count_IPR": effective_n,
                    "effective_spatial_head_count_entropy": entropy_n,
                    "mean_sum_individual_head_spatial_energy": mean_sum_head_sp,
                    "mean_whole_attention_spatial_energy": mean_whole_sp,
                    "spatial_coherence_ratio_whole_over_sum_heads": coherence,
                    "spatial_cancellation_indicator_1_minus_coherence": float(
                        1.0 - coherence
                    ),
                    "whole_attention_spatial_purity": whole_purity,
                }
            )

        # Known heads.
        row_map = {(int(r["layer"]), int(r["head_index"])): r for r in rows}
        known_rows = []
        for key in known_heads:
            if key in row_map:
                known_rows.append(dict(row_map[key]))
            else:
                known_rows.append(
                    {
                        "head": hname(*key),
                        "layer": int(key[0]),
                        "head_index": int(key[1]),
                        "present": False,
                    }
                )

        # Does label-free spatial strength correlate with GT validation?
        gtacc = [float(r["gt_direction_accuracy_from_projection"]) for r in rows]
        corr_rows = []
        for metric in (
            "spatial_coord_rms",
            "spatial_rms_residual",
            "energy_purity",
            "mean_sample_purity",
            "layer_spatial_share",
        ):
            vals = [float(r[metric]) for r in rows]
            corr_rows.append(
                {
                    "selection_metric": metric,
                    "validation_metric": "gt_direction_accuracy_from_projection",
                    "N_heads": len(rows),
                    "pearson": pearson(vals, gtacc),
                    "spearman": spearman(vals, gtacc),
                }
            )

        # Save.
        write_csv(outdir / "all_head_spatial_strength.csv", rows_by_strength)
        write_csv(outdir / "top_by_spatial_strength.csv", rows_by_strength)
        write_csv(outdir / "top_by_spatial_purity.csv", rows_by_purity)
        write_csv(outdir / "per_layer_concentration.csv", layer_rows)
        write_csv(outdir / "known_head_metrics.csv", known_rows)
        write_csv(
            outdir / "metric_vs_gt_validation_correlation.csv",
            corr_rows,
        )
        write_csv(outdir / "spatial_basis.csv", basis_rows)
        write_csv(outdir / "errors.csv", errors)

        config = vars(args).copy()
        config.update(
            {
                "resolved_layers": layers,
                "source_vector_definition": src_definition,
                "source_N": int(len(Xsrc)),
                "source_hidden_dim": int(Xsrc.shape[-1]),
                "coco_requested_N": int(len(coco_rows)),
                "successful_sample_N": int(max(whole_count) if len(whole_count) else 0),
                "n_heads": n_heads,
                "head_dim": head_dim,
                "hidden_size": hidden_size,
                "random_2d_subspace_expected_purity": random_purity,
                "ranking_uses_coco_gt": False,
            }
        )
        (outdir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Console summaries.
        print()
        print("=" * 150)
        print("TOP HEADS BY LABEL-FREE COCO SPATIAL STRENGTH")
        print("(ranking = Synthetic-defined H/V coordinate RMS; COCO GT not used)")
        print("=" * 150)
        print(
            f"{'rank':>4} {'head':>8} {'coordRMS':>9} {'H-rms':>8} "
            f"{'V-rms':>8} {'purity':>9} {'xRand':>8} "
            f"{'share':>8} {'GTacc*':>8}"
        )
        for rank, r in enumerate(rows_by_strength[: int(args.top_print)], 1):
            print(
                f"{rank:4d} {r['head']:>8} "
                f"{float(r['spatial_coord_rms']):9.4f} "
                f"{float(r['rms_H_coord_halfgap_units']):8.4f} "
                f"{float(r['rms_V_coord_halfgap_units']):8.4f} "
                f"{float(r['energy_purity']):9.5f} "
                f"{float(r['purity_enrichment_over_random_2D']):8.1f} "
                f"{float(r['layer_spatial_share']):8.4f} "
                f"{float(r['gt_direction_accuracy_from_projection']):8.4f}"
            )
        print("* GTacc is diagnostic only; it is not used for ranking.")

        print()
        print("=" * 150)
        print("TOP HEADS BY SPATIAL PURITY")
        print("=" * 150)
        print(
            f"{'rank':>4} {'head':>8} {'purity':>9} {'coordRMS':>9} "
            f"{'spRMS':>9} {'totalRMS':>9} {'share':>8} {'GTacc*':>8}"
        )
        for rank, r in enumerate(rows_by_purity[: int(args.top_print)], 1):
            print(
                f"{rank:4d} {r['head']:>8} "
                f"{float(r['energy_purity']):9.5f} "
                f"{float(r['spatial_coord_rms']):9.4f} "
                f"{float(r['spatial_rms_residual']):9.4f} "
                f"{float(r['total_rms_residual']):9.4f} "
                f"{float(r['layer_spatial_share']):8.4f} "
                f"{float(r['gt_direction_accuracy_from_projection']):8.4f}"
            )

        print()
        print("=" * 150)
        print("PER-LAYER SPATIAL CONCENTRATION / MULTI-HEAD MIXING")
        print("=" * 150)
        print(
            f"{'L':>3} {'best':>8} {'top1share':>10} {'top3share':>10} "
            f"{'top5share':>10} {'effN':>7} {'coherence':>10} "
            f"{'wholePur':>10}"
        )
        for r in layer_rows:
            print(
                f"{int(r['layer']):3d} {r['best_strength_head']:>8} "
                f"{float(r['top1_spatial_share']):10.4f} "
                f"{float(r['top3_spatial_share']):10.4f} "
                f"{float(r['top5_spatial_share']):10.4f} "
                f"{float(r['effective_spatial_head_count_IPR']):7.2f} "
                f"{float(r['spatial_coherence_ratio_whole_over_sum_heads']):10.4f} "
                f"{float(r['whole_attention_spatial_purity']):10.5f}"
            )

        print()
        print("=" * 150)
        print("KNOWN HEADS")
        print("=" * 150)
        for r in known_rows:
            if r.get("present") is False:
                print(f"{r['head']}: not scanned")
                continue
            print(
                f"{r['head']}: strength={float(r['spatial_coord_rms']):.4f} "
                f"(rank {float(r['spatial_strength_rank']):.1f}), "
                f"purity={float(r['energy_purity']):.5f} "
                f"(rank {float(r['purity_rank']):.1f}), "
                f"layer_share={float(r['layer_spatial_share']):.4f}, "
                f"GTacc*={float(r['gt_direction_accuracy_from_projection']):.4f}"
            )

        print()
        print("=" * 150)
        print("LABEL-FREE METRIC vs GT VALIDATION")
        print("=" * 150)
        for r in corr_rows:
            print(
                f"{r['selection_metric']:<28} "
                f"Pearson={float(r['pearson']):.4f} "
                f"Spearman={float(r['spearman']):.4f}"
            )

        print()
        print(f"[saved] {outdir}")
        print(
            f"[success] rows={len(rows)} "
            f"successful_samples={int(max(whole_count)) if len(whole_count) else 0} "
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
