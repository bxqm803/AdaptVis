#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_spatial_heads_synthetic400_coco440_v1.py

Goal
====
Independently identify "spatial heads" on:

  1) synthetic_shapes_4dir_400
  2) COCO_two / original-prompt COCO-440

using the SAME head-level spatial diagnostic, then ask whether the SAME heads
are spatially strong in both datasets.

Crucially, this script does NOT use downstream answer-improvement / repair
results to select heads.

Head representation
===================
For attention head h at decoder layer L, capture the head slice immediately
before W_O at subject/reference text-token positions:

    g_real(L,h) =
        z_real(L,h,subject) - z_real(L,h,reference)

and subtract a gray-image control:

    q(L,h) = g_real(L,h) - g_gray(L,h)

Pooling over multi-token object phrases is controlled by --pool (mean/last).

Each dataset is split independently into a calibration split and a held-out
evaluation split. On the calibration split, for every (L,h), fit a 4-way
relation code from class means:

    mu_left, mu_right, mu_above, mu_below

with a calibration center c. Prediction uses cosine similarity between
q-c and the normalized class-mean directions.

Four maps are reported
======================
  SYN -> SYN    : fit on synthetic calibration, test on synthetic held-out
  COCO -> COCO  : fit on COCO calibration,      test on COCO held-out
  SYN -> COCO   : freeze synthetic code,         test on COCO held-out
  COCO -> SYN   : freeze COCO code,              test on synthetic held-out

Main question
=============
Do independently discovered high-spatial heads overlap?

Outputs include:
  - all_head_scores.csv
  - top_heads_syn_in.csv
  - top_heads_coco_in.csv
  - top_heads_syn_to_coco.csv
  - top_heads_coco_to_syn.csv
  - rank_correlations.csv
  - topk_overlap.csv
  - known_head_ranks.csv
  - per_layer_summary.csv
  - split_manifest.csv
  - config.json

Recommended full run
====================
CUDA_VISIBLE_DEVICES=0 python -u compare_spatial_heads_synthetic400_coco440_v1.py \
  --model qwen-3b \
  --layers 16-27 \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --train-ratio 0.30 \
  --pool mean \
  --device cuda:0 \
  --output-dir output/qwen3b_spatial_heads_syn400_vs_coco440_v1 \
  --overwrite

Interpretation
==============
Strong evidence for dataset-independent spatial specialization would be:

  * positive rank correlation between SYN->SYN and COCO->COCO head maps;
  * substantial Top-K overlap, especially for small K;
  * known spatial heads (e.g. L23H01/L23H05/L26H03) rank high in both;
  * ideally, the same heads also transfer cross-dataset without refitting.

A high within-dataset overlap but weaker cross-dataset transfer means the same
heads may be spatially specialized while their exact coordinate geometry is
dataset/prompt dependent.

This is a diagnostic, not a downstream correction experiment.
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
from collections import Counter, defaultdict
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
        "Run this script from the AdaptVis llava16 repository root next to "
        "analyze_coco_centroid_generation_step1_v4.py and "
        "eval_coco_multilayer_relation_trajectory_repair_v1.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
}
SYNTHETIC_PROMPT = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with left, right, above, or below."
)
EPS = 1e-12


# ---------------------------------------------------------------------
# CLI / small helpers
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
        help="No attention matrices are needed, but eager is the safest default.",
    )

    p.add_argument("--layers", default="16-27")
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--synthetic-max-samples", type=int, default=0)

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--coco-max-samples", type=int, default=0)

    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--top-k", default="5,10,20,30,50")
    p.add_argument("--top-heads-print", type=int, default=20)
    p.add_argument(
        "--known-heads",
        default="23:1,23:5,26:3",
        help="Heads whose ranks should be highlighted, format L:H,L:H,...",
    )

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
            raise ValueError(f"Bad known head {item!r}; expected L:H")
        a, b = item.split(":", 1)
        out.append((int(a), int(b)))
    return out


def hname(L: int, h: int) -> str:
    return f"L{int(L)}H{int(h):02d}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
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


def normalize_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def make_gray(real: Image.Image, value: int) -> Image.Image:
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real.size, (v, v, v))


def span_positions(span: Sequence[int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


def object_positions(
    processor: Any,
    input_ids: Sequence[int],
    subject: str,
    reference: str,
) -> Tuple[List[int], List[int]]:
    ss, rr = base.locate_object_spans(
        processor.tokenizer, input_ids, subject, reference
    )
    return span_positions(ss), span_positions(rr)


def stratified_split(
    rows: Sequence[Mapping[str, Any]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0.0 < float(train_ratio) < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    rng = random.Random(int(seed))
    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_rel[str(r["relation"])].append(dict(r))

    train, test = [], []
    for rel in REL:
        xs = list(by_rel[rel])
        rng.shuffle(xs)
        if len(xs) < 2:
            raise RuntimeError(f"Need >=2 samples for relation={rel}, got {len(xs)}")
        n = int(round(len(xs) * float(train_ratio)))
        n = max(1, min(len(xs) - 1, n))
        train.extend(xs[:n])
        test.extend(xs[n:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_cap(
    rows: Sequence[Mapping[str, Any]],
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(x) for x in rows]
    if not max_samples or int(max_samples) <= 0 or len(rows) <= int(max_samples):
        return rows
    rng = random.Random(int(seed))
    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_rel[str(r["relation"])].append(r)
    for rel in REL:
        rng.shuffle(by_rel[rel])

    out = []
    # round-robin keeps the cap approximately balanced
    while len(out) < int(max_samples):
        progress = False
        for rel in REL:
            if by_rel[rel] and len(out) < int(max_samples):
                out.append(by_rel[rel].pop())
                progress = True
        if not progress:
            break
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------

def norm_syn_rel(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in SYN_REL_MAP:
        raise ValueError(f"Unsupported synthetic relation: {value!r}")
    return SYN_REL_MAP[key]


def load_synthetic_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.synthetic_dir)
    labels = (
        Path(args.synthetic_labels)
        if args.synthetic_labels
        else root / "labels.jsonl"
    )
    if not labels.exists():
        raise FileNotFoundError(f"Synthetic labels not found: {labels}")

    rows = []
    with labels.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rel = norm_syn_rel(item["relation"])
            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()

            image_value = Path(str(item["image"]))
            image_path = image_value if image_value.is_absolute() else root / image_value
            if not image_path.exists():
                raise FileNotFoundError(
                    f"{labels}:{line_no}: missing image {image_path}"
                )

            sid = int(item.get("id", len(rows)))
            rows.append(
                {
                    "dataset": "synthetic",
                    "sid": sid,
                    "relation": rel,
                    "subject": subject,
                    "reference": reference,
                    "question_text": SYNTHETIC_PROMPT.format(
                        subject=subject, reference=reference
                    ),
                    "image_path": str(image_path),
                }
            )

    ids = [int(x["sid"]) for x in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate synthetic ids")
    rows.sort(key=lambda x: int(x["sid"]))
    return stratified_cap(rows, args.synthetic_max_samples, args.seed + 101)


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
        rel = traj.normalize_relation(base, p["answer_raw"])
        if rel not in REL:
            continue
        rows.append(
            {
                "dataset": "coco",
                "sid": sid,
                "relation": rel,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "question_text": str(p["question_text"]),
            }
        )

    rows = stratified_cap(rows, args.coco_max_samples, args.seed + 202)
    keep = {int(x["sid"]) for x in rows}
    rec_by_sid = {k: v for k, v in rec_by_sid.items() if k in keep}
    return rows, rec_by_sid, two


# ---------------------------------------------------------------------
# Model / head capture
# ---------------------------------------------------------------------

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


class LayerCapture:
    def __init__(
        self,
        model: Any,
        decoder_layers: Sequence[Any],
        layers: Sequence[int],
    ):
        cfg = get_text_config(model)
        self.n_heads = int(cfg.num_attention_heads)
        self.hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        if self.hidden_size <= 0:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(layers[0])]))
            self.hidden_size = int(op.in_features)
        if self.hidden_size % self.n_heads != 0:
            raise RuntimeError(
                f"hidden_size={self.hidden_size} not divisible by heads={self.n_heads}"
            )
        self.head_dim = self.hidden_size // self.n_heads
        self.pre_o: Dict[int, np.ndarray] = {}
        self.handles = []

        for L in layers:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(L)]))

            def make_hook(layer_idx: int):
                def hook(_module, inputs):
                    self.pre_o[int(layer_idx)] = (
                        inputs[0].detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_hook(int(L)))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_capture(
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    layers: Sequence[int],
):
    cap = LayerCapture(model, decoder_layers, layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = False
        _ = model(**kw)
        return {
            "pre_o": cap.pre_o,
            "n_heads": cap.n_heads,
            "head_dim": cap.head_dim,
        }
    finally:
        cap.close()


def all_head_pre_o(
    pre_o: Mapping[int, np.ndarray],
    L: int,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    x = pre_o[int(L)][0]
    return x.reshape(x.shape[0], n_heads, head_dim)


def pooled_pair(
    H: np.ndarray,
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
) -> np.ndarray:
    ss = [int(x) for x in spos if 0 <= int(x) < H.shape[0]]
    rr = [int(x) for x in rpos if 0 <= int(x) < H.shape[0]]
    if not ss or not rr:
        raise RuntimeError("No valid subject/reference positions")
    if pool == "last":
        return H[ss[-1]] - H[rr[-1]]
    return H[ss].mean(axis=0) - H[rr].mean(axis=0)


def open_row_image(
    row: Mapping[str, Any],
    rec_by_sid: Mapping[int, Any],
) -> Image.Image:
    if row["dataset"] == "synthetic":
        im = Image.open(str(row["image_path"]))
        return im.convert("RGB")
    im = base.record_image(rec_by_sid[int(row["sid"])])
    if hasattr(im, "convert"):
        im = im.convert("RGB")
    return im


def extract_dataset_vectors(
    *,
    rows: Sequence[Mapping[str, Any]],
    rec_by_sid: Mapping[int, Any],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    device: torch.device,
    pool: str,
    gray_value: int,
    desc: str,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      vectors[L] = [N,H,D] Real-Gray subject-reference pre-W_O head vectors
      kept_meta  = metadata rows aligned to first dimension
      errors
    """
    collected: Dict[int, List[np.ndarray]] = {int(L): [] for L in layers}
    kept_meta: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc=desc):
        real = gray = rb = gb = None
        try:
            real = open_row_image(row, rec_by_sid)
            gray = make_gray(real, gray_value)

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

            ids = rb["input_ids"][0].detach().cpu().tolist()
            spos, rpos = object_positions(
                processor,
                ids,
                str(row["subject"]),
                str(row["reference"]),
            )

            rc = run_capture(model, decoder_layers, rb, layers)
            gc_ = run_capture(model, decoder_layers, gb, layers)

            H = int(rc["n_heads"])
            D = int(rc["head_dim"])
            per_layer = {}
            for L in layers:
                Hr = all_head_pre_o(rc["pre_o"], L, H, D)
                Hg = all_head_pre_o(gc_["pre_o"], L, H, D)
                q = (
                    pooled_pair(Hr, spos, rpos, pool)
                    - pooled_pair(Hg, spos, rpos, pool)
                )
                per_layer[int(L)] = q.astype(np.float32)

            for L in layers:
                collected[int(L)].append(per_layer[int(L)])
            kept_meta.append(dict(row))

        except Exception as exc:
            errors.append(
                {
                    "dataset": row.get("dataset"),
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
            cleanup()

    if not kept_meta:
        raise RuntimeError(f"{desc}: no samples extracted successfully")

    out = {}
    for L in layers:
        out[int(L)] = np.stack(collected[int(L)]).astype(np.float32)
    return out, kept_meta, errors


# ---------------------------------------------------------------------
# Spatial-code fitting / evaluation
# ---------------------------------------------------------------------

def index_by_sid(meta: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    return {int(m["sid"]): i for i, m in enumerate(meta)}


def rows_to_indices(
    split_rows: Sequence[Mapping[str, Any]],
    extracted_meta: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    idx = index_by_sid(extracted_meta)
    missing = [int(x["sid"]) for x in split_rows if int(x["sid"]) not in idx]
    if missing:
        # failed extraction samples are simply dropped from that split
        pass
    out = [idx[int(x["sid"])] for x in split_rows if int(x["sid"]) in idx]
    return np.asarray(out, dtype=np.int64)


def fit_codes(
    vectors: Mapping[int, np.ndarray],
    meta: Sequence[Mapping[str, Any]],
    train_idx: np.ndarray,
    layers: Sequence[int],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Match the existing direction-head diagnostic:
      center = mean of all calibration examples
      dir_r  = unit(mean_r - center)
    """
    rels = np.asarray([str(x["relation"]) for x in meta], dtype=object)
    codes: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for L in layers:
        X = np.asarray(vectors[int(L)], dtype=np.float32)  # [N,H,D]
        H = X.shape[1]
        for h in range(H):
            A = X[train_idx, h, :]
            center = A.mean(axis=0).astype(np.float32)
            dirs = {}
            counts = {}
            ok = True
            for rel in REL:
                mask = rels[train_idx] == rel
                counts[rel] = int(mask.sum())
                if not mask.any():
                    ok = False
                    break
                mu = A[mask].mean(axis=0).astype(np.float32)
                dirs[rel] = normalize_np(mu - center)
            if ok:
                codes[(int(L), int(h))] = {
                    "center": center,
                    "dirs": dirs,
                    "counts": counts,
                }
    return codes


def predict_code_batch(
    X: np.ndarray,
    code: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X [N,D] -> pred relation strings, score matrix [N,4]
    """
    Z = np.asarray(X, dtype=np.float32) - np.asarray(code["center"], dtype=np.float32)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(n, EPS)
    D = np.stack([np.asarray(code["dirs"][r], dtype=np.float32) for r in REL])
    scores = Z @ D.T
    pred_idx = scores.argmax(axis=1)
    pred = np.asarray([REL[int(i)] for i in pred_idx], dtype=object)
    return pred, scores.astype(np.float32)


def evaluate_code_map(
    *,
    vectors: Mapping[int, np.ndarray],
    meta: Sequence[Mapping[str, Any]],
    eval_idx: np.ndarray,
    layers: Sequence[int],
    codes: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    gt = np.asarray([str(x["relation"]) for x in meta], dtype=object)[eval_idx]
    out = {}
    for L in layers:
        X = np.asarray(vectors[int(L)], dtype=np.float32)
        H = X.shape[1]
        for h in range(H):
            code = codes.get((int(L), int(h)))
            if code is None:
                continue
            pred, scores = predict_code_batch(X[eval_idx, h, :], code)
            correct = pred == gt
            margins = []
            for i, g in enumerate(gt):
                gi = REL.index(str(g))
                best_wrong = max(
                    float(scores[i, j]) for j in range(len(REL)) if j != gi
                )
                margins.append(float(scores[i, gi]) - best_wrong)
            out[(int(L), int(h))] = {
                "N": int(len(eval_idx)),
                "correct_N": int(correct.sum()),
                "accuracy": float(correct.mean()) if len(correct) else float("nan"),
                "mean_gt_margin": float(np.mean(margins)) if margins else float("nan"),
                "median_gt_margin": float(np.median(margins)) if margins else float("nan"),
            }
    return out


# ---------------------------------------------------------------------
# Ranking / correlation / overlap
# ---------------------------------------------------------------------

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


def rank_desc(scores: Mapping[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    keys = list(scores)
    vals = [float(scores[k]) for k in keys]
    rr = average_ranks(vals, descending=True)
    return {k: float(r) for k, r in zip(keys, rr)}


def topk_set(scores: Mapping[Tuple[int, int], float], k: int):
    items = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return {x[0] for x in items[: min(int(k), len(items))]}


def hypergeom_expected_overlap(total_heads: int, k: int) -> float:
    kk = min(int(k), int(total_heads))
    if total_heads <= 0:
        return float("nan")
    return float(kk * kk / total_heads)


def leaderboard_rows(
    metric: str,
    score_map: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for (L, h), d in score_map.items():
        rows.append(
            {
                "head": hname(L, h),
                "layer": int(L),
                "head_index": int(h),
                "metric": metric,
                **dict(d),
            }
        )
    return sorted(rows, key=lambda r: (-float(r["accuracy"]), r["layer"], r["head_index"]))


def print_leaderboard(title: str, rows: Sequence[Mapping[str, Any]], n: int):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)
    print(f"{'rank':>4}  {'head':>8}  {'acc':>8}  {'N':>6}  {'margin':>10}")
    for i, r in enumerate(list(rows)[: int(n)], 1):
        print(
            f"{i:4d}  {r['head']:>8}  {float(r['accuracy']):8.4f}  "
            f"{int(r['N']):6d}  {float(r['mean_gt_margin']):10.4f}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    layers = parse_layers(args.layers)
    top_ks = parse_int_list(args.top_k)
    known_heads = parse_known_heads(args.known_heads)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    syn_rows = load_synthetic_rows(args)
    coco_rows, rec_by_sid, two = load_coco_rows(args)

    syn_train_rows, syn_test_rows = stratified_split(
        syn_rows, args.train_ratio, args.seed
    )
    coco_train_rows, coco_test_rows = stratified_split(
        coco_rows, args.train_ratio, args.seed + 1
    )

    # Model
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
    all_errors = []

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
        head_dim = int(getattr(cfg, "hidden_size")) // n_heads

        bad = [L for L in layers if L < 0 or L >= len(decoder_layers)]
        if bad:
            raise ValueError(
                f"Invalid layers {bad}; decoder blocks are 0..{len(decoder_layers)-1}"
            )

        print("=" * 140)
        print("SYNTHETIC-400 vs COCO-440: INDEPENDENT SPATIAL-HEAD CONSISTENCY SCAN")
        print("=" * 140)
        print(f"model={args.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(f"layers={layers}")
        print(f"heads/layer={n_heads} head_dim={head_dim}")
        print(f"total scanned heads={len(layers) * n_heads}")
        print(
            f"synthetic N={len(syn_rows)} "
            f"cal/test={len(syn_train_rows)}/{len(syn_test_rows)}"
        )
        print(
            f"COCO N={len(coco_rows)} "
            f"cal/test={len(coco_train_rows)}/{len(coco_test_rows)}"
        )
        print(
            "Head vector = (REAL subject-reference) - "
            "(GRAY subject-reference), pre-W_O"
        )
        print("No downstream generation / repair result is used for head selection.")
        print()

        device = torch.device(args.device)

        syn_vec, syn_meta, syn_err = extract_dataset_vectors(
            rows=syn_rows,
            rec_by_sid={},
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            layers=layers,
            device=device,
            pool=args.pool,
            gray_value=args.gray_value,
            desc="EXTRACT synthetic head vectors",
        )
        coco_vec, coco_meta, coco_err = extract_dataset_vectors(
            rows=coco_rows,
            rec_by_sid=rec_by_sid,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            layers=layers,
            device=device,
            pool=args.pool,
            gray_value=args.gray_value,
            desc="EXTRACT COCO head vectors",
        )
        all_errors.extend(syn_err)
        all_errors.extend(coco_err)

        syn_train_idx = rows_to_indices(syn_train_rows, syn_meta)
        syn_test_idx = rows_to_indices(syn_test_rows, syn_meta)
        coco_train_idx = rows_to_indices(coco_train_rows, coco_meta)
        coco_test_idx = rows_to_indices(coco_test_rows, coco_meta)

        if min(
            len(syn_train_idx), len(syn_test_idx),
            len(coco_train_idx), len(coco_test_idx)
        ) == 0:
            raise RuntimeError("A calibration/evaluation split became empty after extraction.")

        # Make sure every class survived extraction in every calibration split.
        def count_split(meta, idx):
            c = Counter(str(meta[int(i)]["relation"]) for i in idx)
            return {r: int(c.get(r, 0)) for r in REL}

        print("Effective split counts after extraction:")
        print("  syn cal :", count_split(syn_meta, syn_train_idx))
        print("  syn test:", count_split(syn_meta, syn_test_idx))
        print("  coco cal :", count_split(coco_meta, coco_train_idx))
        print("  coco test:", count_split(coco_meta, coco_test_idx))

        syn_codes = fit_codes(
            syn_vec, syn_meta, syn_train_idx, layers
        )
        coco_codes = fit_codes(
            coco_vec, coco_meta, coco_train_idx, layers
        )

        syn_in = evaluate_code_map(
            vectors=syn_vec,
            meta=syn_meta,
            eval_idx=syn_test_idx,
            layers=layers,
            codes=syn_codes,
        )
        coco_in = evaluate_code_map(
            vectors=coco_vec,
            meta=coco_meta,
            eval_idx=coco_test_idx,
            layers=layers,
            codes=coco_codes,
        )
        syn_to_coco = evaluate_code_map(
            vectors=coco_vec,
            meta=coco_meta,
            eval_idx=coco_test_idx,
            layers=layers,
            codes=syn_codes,
        )
        coco_to_syn = evaluate_code_map(
            vectors=syn_vec,
            meta=syn_meta,
            eval_idx=syn_test_idx,
            layers=layers,
            codes=coco_codes,
        )

        maps = {
            "syn_in": syn_in,
            "coco_in": coco_in,
            "syn_to_coco": syn_to_coco,
            "coco_to_syn": coco_to_syn,
        }

        leaderboards = {
            name: leaderboard_rows(name, mp)
            for name, mp in maps.items()
        }

        for name, title in (
            ("syn_in", "SYN -> SYN spatial-head leaderboard"),
            ("coco_in", "COCO -> COCO spatial-head leaderboard"),
            ("syn_to_coco", "SYN -> COCO frozen-code leaderboard"),
            ("coco_to_syn", "COCO -> SYN frozen-code leaderboard"),
        ):
            print_leaderboard(
                title,
                leaderboards[name],
                args.top_heads_print,
            )

        common_heads = sorted(
            set(syn_in) & set(coco_in) & set(syn_to_coco) & set(coco_to_syn)
        )
        if not common_heads:
            raise RuntimeError("No common heads across four maps")

        # Ranks in each map.
        acc_maps = {
            name: {k: float(v["accuracy"]) for k, v in mp.items()}
            for name, mp in maps.items()
        }
        ranks = {name: rank_desc(acc_maps[name]) for name in acc_maps}

        # One master table.
        all_head_rows = []
        for L, h in common_heads:
            row = {
                "head": hname(L, h),
                "layer": int(L),
                "head_index": int(h),
            }
            for name in ("syn_in", "coco_in", "syn_to_coco", "coco_to_syn"):
                row[f"{name}_accuracy"] = float(maps[name][(L, h)]["accuracy"])
                row[f"{name}_correct_N"] = int(maps[name][(L, h)]["correct_N"])
                row[f"{name}_N"] = int(maps[name][(L, h)]["N"])
                row[f"{name}_mean_gt_margin"] = float(
                    maps[name][(L, h)]["mean_gt_margin"]
                )
                row[f"{name}_rank"] = float(ranks[name][(L, h)])

            row["within_dataset_mean_accuracy"] = 0.5 * (
                row["syn_in_accuracy"] + row["coco_in_accuracy"]
            )
            row["cross_dataset_mean_accuracy"] = 0.5 * (
                row["syn_to_coco_accuracy"] + row["coco_to_syn_accuracy"]
            )
            row["all_four_mean_accuracy"] = 0.25 * sum(
                row[f"{x}_accuracy"]
                for x in ("syn_in", "coco_in", "syn_to_coco", "coco_to_syn")
            )
            all_head_rows.append(row)

        all_head_rows.sort(
            key=lambda r: (
                -float(r["within_dataset_mean_accuracy"]),
                -float(r["cross_dataset_mean_accuracy"]),
                int(r["layer"]),
                int(r["head_index"]),
            )
        )

        # Correlations.
        metric_pairs = [
            ("syn_in", "coco_in"),
            ("syn_in", "syn_to_coco"),
            ("coco_in", "coco_to_syn"),
            ("syn_to_coco", "coco_to_syn"),
            ("coco_in", "syn_to_coco"),
            ("syn_in", "coco_to_syn"),
        ]
        corr_rows = []
        for a, b in metric_pairs:
            xa = [acc_maps[a][k] for k in common_heads]
            xb = [acc_maps[b][k] for k in common_heads]
            corr_rows.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "N_heads": len(common_heads),
                    "pearson_accuracy": pearson(xa, xb),
                    "spearman_rank": spearman(xa, xb),
                }
            )

        # Top-K overlap between independently discovered within-dataset maps.
        overlap_rows = []
        total_heads = len(common_heads)
        for k in top_ks:
            A = topk_set(acc_maps["syn_in"], k)
            B = topk_set(acc_maps["coco_in"], k)
            inter = sorted(A & B)
            union = A | B
            expected = hypergeom_expected_overlap(total_heads, k)
            overlap_rows.append(
                {
                    "K": int(k),
                    "total_heads": total_heads,
                    "syn_topK_N": len(A),
                    "coco_topK_N": len(B),
                    "intersection_N": len(inter),
                    "overlap_fraction_of_K": (
                        len(inter) / max(1, min(int(k), total_heads))
                    ),
                    "jaccard": len(inter) / max(1, len(union)),
                    "random_expected_intersection": expected,
                    "enrichment_over_random": (
                        len(inter) / expected if expected > 0 else float("nan")
                    ),
                    "intersection_heads": ";".join(hname(*x) for x in inter),
                }
            )

        # Highlight known heads.
        known_rows = []
        for k in known_heads:
            L, h = k
            row = {"head": hname(L, h), "layer": L, "head_index": h}
            if k not in common_heads:
                row["present"] = False
                known_rows.append(row)
                continue
            row["present"] = True
            for name in ("syn_in", "coco_in", "syn_to_coco", "coco_to_syn"):
                row[f"{name}_accuracy"] = acc_maps[name][k]
                row[f"{name}_rank"] = ranks[name][k]
            known_rows.append(row)

        # Per-layer aggregate: are the same layers enriched?
        per_layer_rows = []
        for L in layers:
            ks = [k for k in common_heads if int(k[0]) == int(L)]
            if not ks:
                continue
            r = {
                "layer": int(L),
                "N_heads": len(ks),
            }
            for name in ("syn_in", "coco_in", "syn_to_coco", "coco_to_syn"):
                vals = np.asarray([acc_maps[name][k] for k in ks], dtype=np.float64)
                best = max(ks, key=lambda k: acc_maps[name][k])
                r[f"{name}_mean_accuracy"] = float(vals.mean())
                r[f"{name}_median_accuracy"] = float(np.median(vals))
                r[f"{name}_best_accuracy"] = float(acc_maps[name][best])
                r[f"{name}_best_head"] = hname(*best)
            per_layer_rows.append(r)

        # Split manifest.
        split_rows = []
        extracted_syn = index_by_sid(syn_meta)
        extracted_coco = index_by_sid(coco_meta)
        split_sets = [
            ("synthetic", "calibration", syn_train_rows, extracted_syn),
            ("synthetic", "heldout", syn_test_rows, extracted_syn),
            ("coco", "calibration", coco_train_rows, extracted_coco),
            ("coco", "heldout", coco_test_rows, extracted_coco),
        ]
        for ds, split, rows, extracted in split_sets:
            for r in rows:
                split_rows.append(
                    {
                        "dataset": ds,
                        "split": split,
                        "sid": int(r["sid"]),
                        "relation": str(r["relation"]),
                        "extracted_successfully": int(r["sid"]) in extracted,
                    }
                )

        # Save.
        write_csv(outdir / "all_head_scores.csv", all_head_rows)
        write_csv(outdir / "top_heads_syn_in.csv", leaderboards["syn_in"])
        write_csv(outdir / "top_heads_coco_in.csv", leaderboards["coco_in"])
        write_csv(
            outdir / "top_heads_syn_to_coco.csv",
            leaderboards["syn_to_coco"],
        )
        write_csv(
            outdir / "top_heads_coco_to_syn.csv",
            leaderboards["coco_to_syn"],
        )
        write_csv(outdir / "rank_correlations.csv", corr_rows)
        write_csv(outdir / "topk_overlap.csv", overlap_rows)
        write_csv(outdir / "known_head_ranks.csv", known_rows)
        write_csv(outdir / "per_layer_summary.csv", per_layer_rows)
        write_csv(outdir / "split_manifest.csv", split_rows)
        write_csv(outdir / "errors.csv", all_errors)

        config = vars(args).copy()
        config.update(
            {
                "resolved_layers": layers,
                "n_decoder_blocks": len(decoder_layers),
                "n_heads": n_heads,
                "head_dim": head_dim,
                "n_total_scanned_heads": len(common_heads),
                "synthetic_extracted_N": len(syn_meta),
                "coco_extracted_N": len(coco_meta),
                "synthetic_calibration_effective_N": len(syn_train_idx),
                "synthetic_heldout_effective_N": len(syn_test_idx),
                "coco_calibration_effective_N": len(coco_train_idx),
                "coco_heldout_effective_N": len(coco_test_idx),
            }
        )
        (outdir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Compact console summaries.
        print()
        print("=" * 140)
        print("RANK CORRELATIONS")
        print("=" * 140)
        print(
            f"{'metric_a':>14}  {'metric_b':>14}  "
            f"{'pearson':>9}  {'spearman':>9}"
        )
        for r in corr_rows:
            print(
                f"{r['metric_a']:>14}  {r['metric_b']:>14}  "
                f"{float(r['pearson_accuracy']):9.4f}  "
                f"{float(r['spearman_rank']):9.4f}"
            )

        print()
        print("=" * 140)
        print("TOP-K OVERLAP: INDEPENDENT SYN vs COCO SPATIAL-HEAD DISCOVERY")
        print("=" * 140)
        print(
            f"{'K':>5}  {'overlap':>8}  {'frac':>8}  "
            f"{'jaccard':>9}  {'expected':>9}  {'enrich':>9}  heads"
        )
        for r in overlap_rows:
            print(
                f"{int(r['K']):5d}  {int(r['intersection_N']):8d}  "
                f"{float(r['overlap_fraction_of_K']):8.4f}  "
                f"{float(r['jaccard']):9.4f}  "
                f"{float(r['random_expected_intersection']):9.3f}  "
                f"{float(r['enrichment_over_random']):9.3f}  "
                f"{r['intersection_heads']}"
            )

        print()
        print("=" * 140)
        print("KNOWN SPATIAL HEADS")
        print("=" * 140)
        for r in known_rows:
            if not r.get("present"):
                print(f"{r['head']}: not in scanned layers")
                continue
            print(
                f"{r['head']}: "
                f"SYN {r['syn_in_accuracy']:.4f} (rank {r['syn_in_rank']:.1f}), "
                f"COCO {r['coco_in_accuracy']:.4f} (rank {r['coco_in_rank']:.1f}), "
                f"SYN->COCO {r['syn_to_coco_accuracy']:.4f} "
                f"(rank {r['syn_to_coco_rank']:.1f}), "
                f"COCO->SYN {r['coco_to_syn_accuracy']:.4f} "
                f"(rank {r['coco_to_syn_rank']:.1f})"
            )

        print()
        print("=" * 140)
        print("TOP HEADS BY WITHIN-DATASET MEAN ACCURACY")
        print("=" * 140)
        print(
            f"{'rank':>4} {'head':>8} {'syn':>7} {'coco':>7} "
            f"{'s->c':>7} {'c->s':>7} {'within':>8} {'cross':>8}"
        )
        for i, r in enumerate(all_head_rows[: args.top_heads_print], 1):
            print(
                f"{i:4d} {r['head']:>8} "
                f"{r['syn_in_accuracy']:7.4f} "
                f"{r['coco_in_accuracy']:7.4f} "
                f"{r['syn_to_coco_accuracy']:7.4f} "
                f"{r['coco_to_syn_accuracy']:7.4f} "
                f"{r['within_dataset_mean_accuracy']:8.4f} "
                f"{r['cross_dataset_mean_accuracy']:8.4f}"
            )

        print()
        print(f"[saved] {outdir}")
        print(
            f"[success] scanned_heads={len(common_heads)} "
            f"errors={len(all_errors)}"
        )

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        cleanup()


if __name__ == "__main__":
    main()
