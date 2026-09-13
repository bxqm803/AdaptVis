#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
causal_spatial_head_consensus_selector_all440_v2.py

Purpose
=======
Build a TRAINING-FREE / NON-ORACLE spatial selector from causally relevant
spatial heads.

Pipeline
--------
1) Synthetic-400 defines each head's OWN 4-way spatial code.
   No COCO labels are used here.

2) Use ALL COCO-440 samples, WITHOUT relation GT, to causally rank EVERY
   head in source layers L16-L26:
      start from GRAY computation
      inject one head's REAL-GRAY pre-W_O contribution
      continue forward to fixed readout layer L27
      measure downstream H/V spatial-effect magnitude

   IMPORTANT:
      causal ranking uses magnitude ONLY.
      No COCO GT relation is used for head selection.

3) Freeze Top-K heads, K in {1,3,5,10,20} by default.

4) On the same 440 target samples, each selected head predicts relation using
   its own frozen Synthetic spatial code, then aggregate head evidence by:

      majority
      causal_vote
      causal_margin_vote
      mean_cosine_score
      causal_cosine_score
      borda
      causal_borda

5) Only AFTER all predictions are fixed, reveal COCO GT to report selector
   accuracy, plus W2C/C2W relative to the generation baseline if
   --baseline-csv can be parsed.

This cleanly separates:
    causal relevance  -> which heads matter downstream
from
    spatial code      -> what relation each head says

No downstream answer accuracy is used to select the heads.

Recommended run
===============
CUDA_VISIBLE_DEVICES=0 python -u causal_spatial_head_consensus_selector_all440_v2.py \
  --model qwen-3b \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --source-layers 16-26 \
  --readout-layer 27 \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --baseline-csv \
    output/qwen3b_synthetic400_to_coco440_originalprompt_mean_v2/per_sample_candidate_repair.csv \
  --prior-accuracy-csv \
    output/qwen3b_spatial_heads_syn400_vs_coco440_v1/all_head_scores.csv \
  --top-k 1,3,5,10,20 \
  --causal-head-batch 4 \
  --pool mean \
  --device cuda:0 \
  --output-dir output/qwen3b_causal_spatial_head_consensus_selector_v1 \
  --overwrite

If memory allows, --causal-head-batch 8 or 16 is much faster.

Outputs
=======
  causal_head_ranking.csv
  selected_heads_by_k.csv
  selector_summary.csv
  selector_per_sample.csv
  causal_vs_prior_accuracy.csv
  split_manifest.csv
  synthetic_head_code_sanity.csv
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
from collections import Counter, defaultdict
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
REL_TO_I = {r: i for i, r in enumerate(REL)}
EPS = 1e-12

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


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-spatial-npz", required=True)

    p.add_argument("--source-layers", default="16-26")
    p.add_argument("--readout-layer", type=int, default=27)

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--baseline-csv", default=None)
    p.add_argument("--prior-accuracy-csv", default=None)

    # Kept only for backward CLI compatibility. This all-440 version ignores
    # these split arguments and uses every COCO sample for unlabeled causal ranking.
    p.add_argument("--calibration-ratio", type=float, default=1.0)
    p.add_argument("--split-seed", type=int, default=20260912)
    p.add_argument("--top-k", default="1,3,5,10,20")

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument(
        "--causal-head-batch",
        type=int,
        default=4,
        help="Number of head interventions evaluated per forward pass.",
    )
    p.add_argument(
        "--known-heads",
        default="23:1,23:5,26:3,19:13,20:8,23:0,19:8",
    )
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
        rank = 0.5 * ((i + 1) + j)
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
    x -= x.mean()
    y -= y.mean()
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


def exact_stratified_split(
    rows: Sequence[Mapping[str, Any]],
    ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Exact total calibration size round(N*ratio), allocated across classes by
    largest remainder, then sampled independently within each class.
    """
    if not (0.0 < float(ratio) < 1.0):
        raise ValueError("--calibration-ratio must be in (0,1)")

    rows = [dict(r) for r in rows]
    N = len(rows)
    target_total = int(round(N * float(ratio)))
    target_total = max(len(REL), min(N - len(REL), target_total))

    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_rel[str(r["relation"])].append(r)

    raw = {rel: len(by_rel[rel]) * target_total / N for rel in REL}
    alloc = {rel: int(math.floor(raw[rel])) for rel in REL}

    # Guarantee at least one calib and one test example per relation.
    for rel in REL:
        alloc[rel] = max(1, min(len(by_rel[rel]) - 1, alloc[rel]))

    while sum(alloc.values()) < target_total:
        candidates = [
            rel for rel in REL
            if alloc[rel] < len(by_rel[rel]) - 1
        ]
        if not candidates:
            break
        rel = max(
            candidates,
            key=lambda r: (raw[r] - math.floor(raw[r]), len(by_rel[r]), -REL_TO_I[r]),
        )
        alloc[rel] += 1
        raw[rel] = math.floor(raw[rel])  # don't repeatedly favor same remainder

    while sum(alloc.values()) > target_total:
        candidates = [rel for rel in REL if alloc[rel] > 1]
        if not candidates:
            break
        rel = min(
            candidates,
            key=lambda r: (raw[r] - math.floor(raw[r]), -len(by_rel[r]), REL_TO_I[r]),
        )
        alloc[rel] -= 1

    rng = random.Random(int(seed))
    calib, test = [], []
    for rel in REL:
        xs = list(by_rel[rel])
        rng.shuffle(xs)
        n = alloc[rel]
        calib.extend(xs[:n])
        test.extend(xs[n:])

    rng.shuffle(calib)
    rng.shuffle(test)
    return calib, test


# =============================================================================
# Synthetic data and head-specific codes
# =============================================================================

def norm_syn_rel(v: Any) -> str:
    key = str(v).strip().lower().replace("-", "_")
    if key not in SYN_REL_MAP:
        raise ValueError(f"Unsupported synthetic relation: {v!r}")
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
            image_path = (
                image_value if image_value.is_absolute()
                else root / image_value
            )
            if not image_path.exists():
                raise FileNotFoundError(
                    f"{labels}:{line_no}: missing image {image_path}"
                )

            rows.append(
                {
                    "dataset": "synthetic",
                    "sid": int(item.get("id", len(rows))),
                    "relation": rel,
                    "subject": subject,
                    "reference": reference,
                    "question_text": SYNTHETIC_PROMPT.format(
                        subject=subject,
                        reference=reference,
                    ),
                    "image_path": str(image_path),
                }
            )

    rows.sort(key=lambda r: int(r["sid"]))
    return rows


def open_synthetic_image(row: Mapping[str, Any]) -> Image.Image:
    return Image.open(str(row["image_path"])).convert("RGB")


# =============================================================================
# Synthetic residual-space readout basis for causal downstream effect
# =============================================================================

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
    return X, y, layers, definition


def fit_residual_spatial_basis(
    X: np.ndarray,
    y: np.ndarray,
    source_layers: Sequence[int],
    wanted_layer: int,
):
    lmap = {int(L): i for i, L in enumerate(source_layers)}
    if int(wanted_layer) not in lmap:
        raise RuntimeError(f"Source cache missing L{wanted_layer}")

    Xf = np.asarray(X[:, lmap[int(wanted_layer)]], dtype=np.float64)
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

    B = np.stack([dH, dV], axis=1)
    gram = B.T @ B
    cond = float(np.linalg.cond(gram))
    if not np.isfinite(cond) or cond > 1e8:
        raise RuntimeError(f"Ill-conditioned readout basis: cond={cond}")

    dual = B @ np.linalg.inv(gram)
    Q, _ = np.linalg.qr(B)
    Q = Q[:, :2]

    return {
        "Q": Q.astype(np.float32),
        "dual": dual.astype(np.float32),
        "halfH": float(max(gapH / 2.0, EPS)),
        "halfV": float(max(gapV / 2.0, EPS)),
        "hidden_dim": int(Xf.shape[1]),
        "gapH": float(gapH),
        "gapV": float(gapV),
        "condition": cond,
    }


# =============================================================================
# COCO data / baseline / prior accuracy
# =============================================================================

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
                "dataset": "coco",
                "sid": sid,
                "relation": rel,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "question_text": str(p["question_text"]),
            }
        )

    rows.sort(key=lambda r: int(r["sid"]))
    return rows, rec_by_sid, two


def load_prior_accuracy_csv(
    path: Optional[str],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warning] prior accuracy CSV not found: {p}")
        return {}

    out = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            if row.get("layer") not in (None, "") and row.get("head_index") not in (None, ""):
                key = (int(float(row["layer"])), int(float(row["head_index"])))
            elif row.get("head"):
                t = row["head"].upper().replace("L", "").replace("H", ":")
                a, b = t.split(":", 1)
                key = (int(a), int(b))
            else:
                continue

            d = {}
            for k, v in row.items():
                if k in {"head", "layer", "head_index"} or v in (None, ""):
                    continue
                try:
                    d[k] = float(v)
                except Exception:
                    d[k] = v
            out[key] = d
    return out


def load_baseline_correctness(
    path: Optional[str],
) -> Dict[int, bool]:
    """
    Best-effort parser. It does NOT affect selector predictions.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warning] baseline CSV not found: {p}")
        return {}

    correctness_candidates = (
        "baseline_correct",
        "base_correct",
        "baseline_is_correct",
        "baseline_generation_correct",
        "is_baseline_correct",
        "correct",
    )
    pred_candidates = (
        "baseline_relation",
        "baseline_pred_relation",
        "baseline_prediction",
        "base_prediction",
        "prediction",
    )
    gt_candidates = ("relation", "gt_relation", "label", "answer")

    out = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    sid_key = None
    for k in ("sid", "id", "sample_id", "index"):
        if k in rows[0]:
            sid_key = k
            break
    if sid_key is None:
        print("[warning] baseline CSV has no recognizable sid column")
        return {}

    def parse_bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "correct"}:
            return True
        if s in {"0", "false", "no", "n", "wrong", "incorrect"}:
            return False
        try:
            return bool(int(float(s)))
        except Exception:
            return None

    for row in rows:
        try:
            sid = int(float(row[sid_key]))
        except Exception:
            continue

        val = None
        for k in correctness_candidates:
            if k in row and row[k] not in (None, ""):
                val = parse_bool(row[k])
                if val is not None:
                    break

        if val is None:
            pred = None
            gt = None
            for k in pred_candidates:
                if k in row and row[k] not in (None, ""):
                    try:
                        pred = norm_rel(row[k])
                    except Exception:
                        pred = None
                    if pred is not None:
                        break
            for k in gt_candidates:
                if k in row and row[k] not in (None, ""):
                    try:
                        gt = norm_rel(row[k])
                    except Exception:
                        gt = None
                    if gt is not None:
                        break
            if pred is not None and gt is not None:
                val = pred == gt

        if val is not None:
            out[sid] = bool(val)

    if not out:
        print(
            "[warning] could not infer baseline correctness columns; "
            "W2C/C2W will be omitted"
        )
    return out


# =============================================================================
# Model helpers
# =============================================================================

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


def pool_head_pair(
    H: torch.Tensor,
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
) -> torch.Tensor:
    """
    H: [T, n_heads, head_dim] -> [n_heads, head_dim]
    """
    T = int(H.shape[0])
    ss = [int(p) for p in spos if 0 <= int(p) < T]
    rr = [int(p) for p in rpos if 0 <= int(p) < T]
    if not ss or not rr:
        raise RuntimeError("No valid subject/reference positions")
    if pool == "last":
        return H[ss[-1]] - H[rr[-1]]
    return H[ss].mean(dim=0) - H[rr].mean(dim=0)


def pool_hidden_pair(
    x: torch.Tensor,
    spos: Sequence[int],
    rpos: Sequence[int],
    pool: str,
) -> torch.Tensor:
    """
    x [B,T,D] -> [B,D]
    """
    T = int(x.shape[1])
    ss = [int(p) for p in spos if 0 <= int(p) < T]
    rr = [int(p) for p in rpos if 0 <= int(p) < T]
    if not ss or not rr:
        raise RuntimeError("No valid subject/reference positions")
    if pool == "last":
        return x[:, ss[-1], :] - x[:, rr[-1], :]
    return x[:, ss, :].mean(dim=1) - x[:, rr, :].mean(dim=1)


def repeat_batch(batch: Mapping[str, Any], n: int) -> Dict[str, Any]:
    """
    Repeat one Qwen multimodal sample n times.
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
        raise RuntimeError(
            f"Cannot safely repeat tensor {k} shape={tuple(v.shape)}"
        )
    return out


# =============================================================================
# Pair capture for Synthetic code fitting and COCO test evidence
# =============================================================================

class PairHeadCapture:
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
        self.head_pair: Dict[int, torch.Tensor] = {}
        self.handles = []

        for L in layers:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(L)]))

            def make_hook(layer_idx: int):
                def hook(_module, inputs):
                    x = inputs[0]
                    T = int(x.shape[1])
                    H = x[0].reshape(T, self.n_heads, self.head_dim)
                    self.head_pair[int(layer_idx)] = (
                        pool_head_pair(
                            H, self.spos, self.rpos, self.pool
                        )
                        .detach()
                        .float()
                        .clone()
                    )
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
        model, decoder_layers, layers, spos, rpos, pool
    )
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = False
        _ = model(**kw)
        missing = sorted(set(layers) - set(cap.head_pair))
        if missing:
            raise RuntimeError(f"Missing head-pair capture layers: {missing}")
        return cap.head_pair
    finally:
        cap.close()


def extract_head_vectors(
    *,
    rows: Sequence[Mapping[str, Any]],
    image_loader,
    rec_by_sid: Mapping[int, Any],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    device: torch.device,
    pool: str,
    gray_value: int,
    desc: str,
):
    """
    vectors[L] -> [N, H, Dh] REAL-GRAY subject-reference pre-W_O
    """
    coll = {int(L): [] for L in layers}
    kept = []
    errors = []

    for row in tqdm(rows, desc=desc):
        real = gray = rb = gb = None
        try:
            real = image_loader(row, rec_by_sid)
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

            rids = rb["input_ids"][0].detach().cpu().tolist()
            gids = gb["input_ids"][0].detach().cpu().tolist()
            rs, rr = locate_positions(
                processor, rids, str(row["subject"]), str(row["reference"])
            )
            gs, gr = locate_positions(
                processor, gids, str(row["subject"]), str(row["reference"])
            )

            rc = run_pair_capture(
                model, decoder_layers, rb, layers, rs, rr, pool
            )
            gc_ = run_pair_capture(
                model, decoder_layers, gb, layers, gs, gr, pool
            )

            for L in layers:
                coll[L].append(
                    (rc[L] - gc_[L]).detach().cpu().numpy().astype(np.float32)
                )
            kept.append(dict(row))

        except Exception as exc:
            errors.append(
                {
                    "phase": desc,
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

    if not kept:
        raise RuntimeError(f"{desc}: no successful samples")

    vec = {
        L: np.stack(coll[L]).astype(np.float32)
        for L in layers
    }
    return vec, kept, errors


def synthetic_image_loader(row, _rec_by_sid):
    return open_synthetic_image(row)


def coco_image_loader(row, rec_by_sid):
    im = base.record_image(rec_by_sid[int(row["sid"])])
    if hasattr(im, "convert"):
        im = im.convert("RGB")
    return im


def fit_head_codes(
    vectors: Mapping[int, np.ndarray],
    meta: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
):
    """
    Fit each head's own 4-way code on ALL Synthetic-400.

    code[(L,h)]:
      center [Dh]
      dirs   [4,Dh] normalized class-mean directions
    """
    y = np.asarray([str(r["relation"]) for r in meta], dtype=object)
    codes = {}

    for L in layers:
        X = np.asarray(vectors[L], dtype=np.float64)  # [N,H,Dh]
        H = int(X.shape[1])

        for h in range(H):
            A = X[:, h, :]
            center = A.mean(axis=0)
            dirs = []
            ok = True
            for rel in REL:
                mask = y == rel
                if not mask.any():
                    ok = False
                    break
                mu = A[mask].mean(axis=0)
                dirs.append(unit(mu - center))
            if not ok:
                continue
            codes[(int(L), int(h))] = {
                "center": center.astype(np.float32),
                "dirs": np.stack(dirs).astype(np.float32),
            }
    return codes


def head_code_scores(
    x: np.ndarray,
    code: Mapping[str, np.ndarray],
) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) - np.asarray(code["center"], dtype=np.float64)
    n = float(np.linalg.norm(z))
    if n <= EPS:
        return np.zeros(4, dtype=np.float64)
    z = z / n
    D = np.asarray(code["dirs"], dtype=np.float64)
    return z @ D.T


def synthetic_code_sanity(
    vectors: Mapping[int, np.ndarray],
    meta: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    codes: Mapping[Tuple[int, int], Mapping[str, np.ndarray]],
):
    y = [str(r["relation"]) for r in meta]
    rows = []
    for L in layers:
        X = vectors[L]
        H = X.shape[1]
        for h in range(H):
            code = codes[(L, h)]
            correct = 0
            margins = []
            for i in range(len(meta)):
                s = head_code_scores(X[i, h], code)
                pred = REL[int(np.argmax(s))]
                correct += int(pred == y[i])
                order = np.argsort(-s)
                margins.append(float(s[order[0]] - s[order[1]]))
            rows.append(
                {
                    "head": hname(L, h),
                    "layer": L,
                    "head_index": h,
                    "synthetic_resub_accuracy": correct / len(meta),
                    "mean_margin": float(np.mean(margins)),
                }
            )
    return rows


# =============================================================================
# Causal calibration ranking
# =============================================================================

class CausalCleanCapture:
    """
    Captures exact pre-W_O head vectors at subject/reference positions for all
    source layers, plus readout-layer input residual pair.
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
                    )
                return hook

            self.handles.append(op.register_forward_pre_hook(make_o_pre(int(L))))

        read_layer = decoder_layers[int(readout_layer)]

        def read_pre(_module, inputs):
            self.readout_pair = (
                pool_hidden_pair(
                    inputs[0],
                    self.spos,
                    self.rpos,
                    self.pool,
                )[0]
                .detach()
                .float()
                .clone()
            )

        self.handles.append(read_layer.register_forward_pre_hook(read_pre))

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
    cap = CausalCleanCapture(
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
            raise RuntimeError(f"Missing source captures: {missing}")
        if cap.readout_pair is None:
            raise RuntimeError("Missing readout pair")
        return cap.head_tokens, cap.readout_pair
    finally:
        cap.close()


class InjectionReadout:
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
                raise RuntimeError(f"Unexpected o_proj output type {type(output)}")
            inj = self.injection.to(
                device=output.device,
                dtype=output.dtype,
            )
            if tuple(inj.shape) != tuple(output.shape):
                raise RuntimeError(
                    f"Injection {tuple(inj.shape)} != output {tuple(output.shape)}"
                )
            return output + inj

        self.handles.append(op.register_forward_hook(o_post))

        read_layer = decoder_layers[int(readout_layer)]

        def read_pre(_module, inputs):
            self.readout_pair = (
                pool_hidden_pair(
                    inputs[0],
                    self.spos,
                    self.rpos,
                    self.pool,
                )
                .detach()
                .float()
                .clone()
            )  # [B,D]

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
):
    cap = InjectionReadout(
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
            raise RuntimeError("Missing injected readout")
        return cap.readout_pair
    finally:
        cap.close()


def spatial_coord_torch(
    v: torch.Tensor,
    readout_basis_t: Mapping[str, Any],
) -> torch.Tensor:
    hv = v @ readout_basis_t["dual"]
    scale = torch.tensor(
        [
            readout_basis_t["halfH"],
            readout_basis_t["halfV"],
        ],
        device=v.device,
        dtype=v.dtype,
    )
    return hv / scale


def causal_rank_all_heads(
    *,
    calib_rows: Sequence[Mapping[str, Any]],
    rec_by_sid: Mapping[int, Any],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    source_layers: Sequence[int],
    readout_layer: int,
    Wo: Mapping[int, torch.Tensor],
    readout_basis_t: Mapping[str, Any],
    device: torch.device,
    pool: str,
    gray_value: int,
    head_batch: int,
):
    cfg = get_text_config(model)
    n_heads = int(cfg.num_attention_heads)
    hidden_size = int(getattr(cfg, "hidden_size"))

    N = defaultdict(int)
    norm2 = defaultdict(float)
    norm1 = defaultdict(float)
    h2 = defaultdict(float)
    v2 = defaultdict(float)
    errors = []

    for row in tqdm(calib_rows, desc="ALL440 causal rank all heads"):
        real = gray = rb = gb = None
        try:
            rec = rec_by_sid[int(row["sid"])]
            real = base.record_image(rec)
            if hasattr(real, "convert"):
                real = real.convert("RGB")
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

            rids = rb["input_ids"][0].detach().cpu().tolist()
            gids = gb["input_ids"][0].detach().cpu().tolist()
            rs, rr = locate_positions(
                processor, rids, str(row["subject"]), str(row["reference"])
            )
            gs, gr = locate_positions(
                processor, gids, str(row["subject"]), str(row["reference"])
            )

            if rs != gs or rr != gr:
                raise RuntimeError(
                    f"REAL/GRAY object positions differ: real={rs,rr}, gray={gs,gr}"
                )

            positions = sorted(set(rs + rr))
            pos_to_local = {p: j for j, p in enumerate(positions)}

            real_tokens, _ = run_causal_clean_capture(
                model=model,
                decoder_layers=decoder_layers,
                batch=rb,
                source_layers=source_layers,
                readout_layer=readout_layer,
                positions=positions,
                spos=rs,
                rpos=rr,
                pool=pool,
            )
            gray_tokens, gray_read = run_causal_clean_capture(
                model=model,
                decoder_layers=decoder_layers,
                batch=gb,
                source_layers=source_layers,
                readout_layer=readout_layer,
                positions=positions,
                spos=gs,
                rpos=gr,
                pool=pool,
            )

            T = int(gb["input_ids"].shape[1])

            for L in source_layers:
                td = real_tokens[L] - gray_tokens[L]  # [P,H,Dh]
                W = Wo[L]                             # [D,H,Dh]

                for start in range(0, n_heads, max(1, int(head_batch))):
                    heads = list(
                        range(
                            start,
                            min(n_heads, start + max(1, int(head_batch))),
                        )
                    )
                    B = len(heads)

                    inj = torch.zeros(
                        (B, T, hidden_size),
                        device=device,
                        dtype=torch.float32,
                    )

                    for b, h in enumerate(heads):
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
                        pool=pool,
                    )

                    effect = read_pairs - gray_read[None, :]
                    coord = spatial_coord_torch(effect.float(), readout_basis_t)
                    arr = coord.detach().cpu().numpy().astype(np.float64)

                    for b, h in enumerate(heads):
                        key = (int(L), int(h))
                        hv = arr[b]
                        nrm = float(np.linalg.norm(hv))
                        N[key] += 1
                        norm2[key] += nrm ** 2
                        norm1[key] += nrm
                        h2[key] += float(hv[0] ** 2)
                        v2[key] += float(hv[1] ** 2)

        except Exception as exc:
            errors.append(
                {
                    "phase": "causal_all440_ranking",
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

    rows = []
    for L in source_layers:
        for h in range(n_heads):
            key = (L, h)
            n = int(N.get(key, 0))
            if n <= 0:
                continue
            rows.append(
                {
                    "head": hname(L, h),
                    "layer": L,
                    "head_index": h,
                    "calibration_N": n,
                    "causal_downstream_spatial_rms": float(
                        math.sqrt(norm2[key] / n)
                    ),
                    "causal_downstream_mean_norm": float(norm1[key] / n),
                    "causal_downstream_H_rms": float(math.sqrt(h2[key] / n)),
                    "causal_downstream_V_rms": float(math.sqrt(v2[key] / n)),
                }
            )

    rows.sort(
        key=lambda r: (
            -float(r["causal_downstream_spatial_rms"]),
            int(r["layer"]),
            int(r["head_index"]),
        )
    )
    for i, r in enumerate(rows, 1):
        r["causal_rank"] = i

    return rows, errors


# =============================================================================
# Test-time aggregation
# =============================================================================

def stable_argmax(scores: np.ndarray) -> int:
    """
    Deterministic REL-order tie-break.
    """
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def zscore4(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    x = x - x.mean()
    sd = float(x.std())
    if sd <= EPS:
        return x
    return x / sd


def borda_points(scores: np.ndarray) -> np.ndarray:
    """
    Highest relation gets 3, then 2,1,0.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    out = np.zeros(4, dtype=np.float64)
    for rank, idx in enumerate(order):
        out[int(idx)] = 3 - rank
    return out


def aggregate_head_scores(
    head_scores: Sequence[np.ndarray],
    causal_weights: Sequence[float],
):
    H = len(head_scores)
    if H == 0:
        raise ValueError("No heads supplied")

    scores = [np.asarray(s, dtype=np.float64) for s in head_scores]
    w = np.asarray(causal_weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if float(w.sum()) <= EPS:
        w = np.ones_like(w)
    w = w / w.sum()

    pred_idx = np.asarray([stable_argmax(s) for s in scores], dtype=np.int64)

    # Per-head confidence margin.
    margins = []
    for s in scores:
        order = np.argsort(-s, kind="mergesort")
        margins.append(max(0.0, float(s[order[0]] - s[order[1]])))
    margins = np.asarray(margins, dtype=np.float64)

    # 1) Majority.
    majority = np.zeros(4, dtype=np.float64)
    for p in pred_idx:
        majority[int(p)] += 1.0

    # 2) Causal weighted hard vote.
    causal_vote = np.zeros(4, dtype=np.float64)
    for p, wi in zip(pred_idx, w):
        causal_vote[int(p)] += float(wi)

    # 3) Causal x confidence-margin hard vote.
    causal_margin_vote = np.zeros(4, dtype=np.float64)
    wm = w * margins
    if float(wm.sum()) <= EPS:
        wm = w.copy()
    for p, wi in zip(pred_idx, wm):
        causal_margin_vote[int(p)] += float(wi)

    # 4) Mean normalized cosine scores.
    Z = np.stack([zscore4(s) for s in scores])
    mean_cos = Z.mean(axis=0)

    # 5) Causal weighted normalized cosine scores.
    causal_cos = (Z * w[:, None]).sum(axis=0)

    # 6/7 Borda.
    B = np.stack([borda_points(s) for s in scores])
    borda = B.mean(axis=0)
    causal_borda = (B * w[:, None]).sum(axis=0)

    return {
        "majority": majority,
        "causal_vote": causal_vote,
        "causal_margin_vote": causal_margin_vote,
        "mean_cosine_score": mean_cos,
        "causal_cosine_score": causal_cos,
        "borda": borda,
        "causal_borda": causal_borda,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    source_layers = parse_layers(args.source_layers)
    readout_layer = int(args.readout_layer)
    top_ks = parse_int_list(args.top_k)
    known_heads = parse_known_heads(args.known_heads)

    if any(L >= readout_layer for L in source_layers):
        raise ValueError(
            f"All source layers must be < readout L{readout_layer}"
        )

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Independent source residual basis at readout layer.
    Xsrc, ysrc, src_layers, src_definition = load_state_npz(
        Path(args.source_spatial_npz)
    )
    readout_basis = fit_residual_spatial_basis(
        Xsrc, ysrc, src_layers, readout_layer
    )

    synthetic_rows = load_synthetic_rows(args)
    coco_rows, rec_by_sid, two = load_coco_rows(args)

    # ALL target samples are used for unlabeled causal head ranking.
    # No relation GT enters the ranking statistic. The same samples are then
    # evaluated only after Top-K head identities and predictions are fixed.
    calib_rows = [dict(r) for r in coco_rows]
    test_rows = [dict(r) for r in coco_rows]

    baseline_correct = load_baseline_correctness(args.baseline_csv)
    prior_acc = load_prior_accuracy_csv(args.prior_accuracy_csv)

    split_manifest = []
    for r in coco_rows:
        split_manifest.append(
            {
                "role": "unlabeled_causal_ranking_and_final_evaluation",
                "sid": int(r["sid"]),
                "relation": str(r["relation"]),
            }
        )
    write_csv(outdir / "split_manifest.csv", split_manifest)

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
        hidden_size = int(getattr(cfg, "hidden_size"))
        head_dim = hidden_size // n_heads

        if int(readout_basis["hidden_dim"]) != hidden_size:
            raise RuntimeError(
                f"Readout basis dim={readout_basis['hidden_dim']} "
                f"!= hidden={hidden_size}"
            )

        for L in source_layers + [readout_layer]:
            if L < 0 or L >= len(decoder_layers):
                raise ValueError(f"Invalid layer L{L}")

        device = torch.device(args.device)

        print("=" * 150)
        print("CAUSAL SPATIAL HEAD CONSENSUS SELECTOR")
        print("=" * 150)
        print(f"model={args.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(f"source_layers={source_layers}")
        print(f"readout_layer=L{readout_layer}")
        print(f"heads/layer={n_heads}")
        print(f"total candidate heads={len(source_layers)*n_heads}")
        print(
            f"COCO total={len(coco_rows)} "
            f"unlabeled causal-ranking N={len(calib_rows)} "
            f"final evaluation N={len(test_rows)}"
        )
        print(
            "target relation counts (printed for audit only; not used in ranking):",
            dict(Counter(r["relation"] for r in coco_rows)),
        )
        print(
            "Head selection = downstream causal spatial magnitude on ALL "
            "target samples; no COCO GT used."
        )
        print(
            "Relation code = each head's own frozen Synthetic-400 code."
        )
        print()

        # ------------------------------------------------------------------
        # A) Synthetic head-specific spatial codes.
        # ------------------------------------------------------------------
        syn_vec, syn_meta, syn_errors = extract_head_vectors(
            rows=synthetic_rows,
            image_loader=synthetic_image_loader,
            rec_by_sid={},
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            layers=source_layers,
            device=device,
            pool=args.pool,
            gray_value=args.gray_value,
            desc="SYNTHETIC fit head codes",
        )
        all_errors.extend(syn_errors)

        head_codes = fit_head_codes(
            syn_vec, syn_meta, source_layers
        )
        syn_sanity = synthetic_code_sanity(
            syn_vec,
            syn_meta,
            source_layers,
            head_codes,
        )
        write_csv(
            outdir / "synthetic_head_code_sanity.csv",
            syn_sanity,
        )

        # W_O blocks for all causal interventions.
        Wo = {}
        for L in source_layers:
            op = resolve_o_proj(resolve_attn(decoder_layers[int(L)]))
            W = op.weight.detach().float()
            if int(W.shape[1]) != n_heads * head_dim:
                raise RuntimeError(
                    f"L{L} W_O input dim mismatch {tuple(W.shape)}"
                )
            Wo[L] = W.reshape(hidden_size, n_heads, head_dim)

        readout_basis_t = {
            "Q": torch.from_numpy(readout_basis["Q"]).to(
                device=device, dtype=torch.float32
            ),
            "dual": torch.from_numpy(readout_basis["dual"]).to(
                device=device, dtype=torch.float32
            ),
            "halfH": float(readout_basis["halfH"]),
            "halfV": float(readout_basis["halfV"]),
        }

        # ------------------------------------------------------------------
        # B) Full all-head causal ranking on 15% calibration.
        # ------------------------------------------------------------------
        causal_rows, causal_errors = causal_rank_all_heads(
            calib_rows=calib_rows,
            rec_by_sid=rec_by_sid,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            source_layers=source_layers,
            readout_layer=readout_layer,
            Wo=Wo,
            readout_basis_t=readout_basis_t,
            device=device,
            pool=args.pool,
            gray_value=args.gray_value,
            head_batch=args.causal_head_batch,
        )
        all_errors.extend(causal_errors)

        if len(causal_rows) != len(source_layers) * n_heads:
            print(
                f"[warning] causal ranking has {len(causal_rows)} heads, "
                f"expected {len(source_layers)*n_heads}"
            )

        # Join prior accuracy only for AFTER-THE-FACT comparison.
        for r in causal_rows:
            key = (int(r["layer"]), int(r["head_index"]))
            prior = prior_acc.get(key, {})
            for k, v in prior.items():
                r[f"prior_{k}"] = v

            synr = next(
                (
                    x for x in syn_sanity
                    if int(x["layer"]) == key[0]
                    and int(x["head_index"]) == key[1]
                ),
                None,
            )
            if synr is not None:
                r["synthetic_resub_accuracy"] = float(
                    synr["synthetic_resub_accuracy"]
                )
                r["synthetic_mean_margin"] = float(synr["mean_margin"])

        write_csv(outdir / "causal_head_ranking.csv", causal_rows)

        # Correlation with prior COCO accuracy, diagnostic only.
        corr_rows = []
        if any("prior_coco_in_accuracy" in r for r in causal_rows):
            x = [
                float(r["causal_downstream_spatial_rms"])
                for r in causal_rows
            ]
            y = [
                float(r.get("prior_coco_in_accuracy", float("nan")))
                for r in causal_rows
            ]
            corr_rows.append(
                {
                    "metric": "causal_downstream_spatial_rms",
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
        write_csv(outdir / "causal_vs_prior_accuracy.csv", corr_rows)

        print()
        print("=" * 150)
        print("FULL CAUSAL HEAD RANKING -- ALL 440, NO COCO GT")
        print("=" * 150)
        print(
            f"{'rank':>4} {'head':>8} {'downRMS':>9} "
            f"{'H-rms':>8} {'V-rms':>8} "
            f"{'synAcc':>8} {'priorCOCOacc*':>13}"
        )
        for r in causal_rows[: int(args.top_print)]:
            print(
                f"{int(r['causal_rank']):4d} "
                f"{r['head']:>8} "
                f"{float(r['causal_downstream_spatial_rms']):9.4f} "
                f"{float(r['causal_downstream_H_rms']):8.4f} "
                f"{float(r['causal_downstream_V_rms']):8.4f} "
                f"{float(r.get('synthetic_resub_accuracy', float('nan'))):8.4f} "
                f"{float(r.get('prior_coco_in_accuracy', float('nan'))):13.4f}"
            )
        print("* priorCOCOacc is comparison only; never used for ranking.")

        known_map = {
            (int(r["layer"]), int(r["head_index"])): r
            for r in causal_rows
        }
        print()
        print("KNOWN HEADS IN CAUSAL RANKING")
        for key in known_heads:
            r = known_map.get(key)
            if r is None:
                continue
            print(
                f"{r['head']}: causal rank={r['causal_rank']} "
                f"downRMS={r['causal_downstream_spatial_rms']:.4f} "
                f"priorCOCOacc={float(r.get('prior_coco_in_accuracy', float('nan'))):.4f}"
            )

        # ------------------------------------------------------------------
        # C) Freeze Top-K.
        # ------------------------------------------------------------------
        selected_rows = []
        top_by_k = {}
        for K in top_ks:
            k_eff = min(int(K), len(causal_rows))
            selected = causal_rows[:k_eff]
            top_by_k[int(K)] = [
                (int(r["layer"]), int(r["head_index"]))
                for r in selected
            ]
            for r in selected:
                selected_rows.append(
                    {
                        "K": int(K),
                        "within_K_rank": int(r["causal_rank"]),
                        **dict(r),
                    }
                )
        write_csv(outdir / "selected_heads_by_k.csv", selected_rows)

        # ------------------------------------------------------------------
        # D) Extract TEST head evidence only once.
        # ------------------------------------------------------------------
        test_vec, test_meta, test_errors = extract_head_vectors(
            rows=test_rows,
            image_loader=coco_image_loader,
            rec_by_sid=rec_by_sid,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            layers=source_layers,
            device=device,
            pool=args.pool,
            gray_value=args.gray_value,
            desc="ALL440 extract head evidence",
        )
        all_errors.extend(test_errors)

        test_sid_to_i = {
            int(r["sid"]): i for i, r in enumerate(test_meta)
        }
        effective_test_rows = [
            r for r in test_rows
            if int(r["sid"]) in test_sid_to_i
        ]

        causal_weight = {
            (int(r["layer"]), int(r["head_index"])):
                float(r["causal_downstream_spatial_rms"])
            for r in causal_rows
        }

        methods = (
            "majority",
            "causal_vote",
            "causal_margin_vote",
            "mean_cosine_score",
            "causal_cosine_score",
            "borda",
            "causal_borda",
        )

        # Per method/K accumulators.
        agg = {
            (K, m): {
                "N": 0,
                "correct": 0,
                "w2c": 0,
                "c2w": 0,
                "baseline_known_N": 0,
                "changed_vs_baseline_correctness": 0,
                "sum_margin": 0.0,
            }
            for K in top_ks
            for m in methods
        }
        per_sample_rows = []

        for row in tqdm(effective_test_rows, desc="ALL440 aggregate selectors"):
            sid = int(row["sid"])
            i = test_sid_to_i[sid]
            gt = str(row["relation"])
            gt_i = REL_TO_I[gt]
            base_ok = baseline_correct.get(sid, None)

            for K in top_ks:
                selected = top_by_k[K]
                hscores = []
                weights = []
                head_preds = []

                for L, h in selected:
                    code = head_codes[(L, h)]
                    s = head_code_scores(test_vec[L][i, h], code)
                    hscores.append(s)
                    weights.append(causal_weight[(L, h)])
                    head_preds.append(REL[stable_argmax(s)])

                outputs = aggregate_head_scores(hscores, weights)

                for method, final_scores in outputs.items():
                    pred_i = stable_argmax(final_scores)
                    pred = REL[pred_i]
                    correct = pred_i == gt_i

                    order = np.argsort(-final_scores, kind="mergesort")
                    margin = float(
                        final_scores[order[0]] - final_scores[order[1]]
                    )

                    a = agg[(K, method)]
                    a["N"] += 1
                    a["correct"] += int(correct)
                    a["sum_margin"] += margin

                    if base_ok is not None:
                        a["baseline_known_N"] += 1
                        if (not base_ok) and correct:
                            a["w2c"] += 1
                        if base_ok and (not correct):
                            a["c2w"] += 1
                        if bool(base_ok) != bool(correct):
                            a["changed_vs_baseline_correctness"] += 1

                    per_sample_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "K": K,
                            "method": method,
                            "prediction": pred,
                            "correct": int(correct),
                            "selector_margin": margin,
                            "baseline_correct": (
                                "" if base_ok is None else int(base_ok)
                            ),
                            "selected_heads": ";".join(
                                hname(L, h) for L, h in selected
                            ),
                            "head_predictions": ";".join(head_preds),
                        }
                    )

        summary_rows = []

        baseline_test_known = [
            baseline_correct[int(r["sid"])]
            for r in effective_test_rows
            if int(r["sid"]) in baseline_correct
        ]
        baseline_test_acc = (
            float(np.mean(baseline_test_known))
            if baseline_test_known
            else float("nan")
        )

        for K in top_ks:
            for method in methods:
                a = agg[(K, method)]
                n = max(int(a["N"]), 1)
                acc = float(a["correct"] / n)
                summary_rows.append(
                    {
                        "K": int(K),
                        "method": method,
                        "test_N": int(a["N"]),
                        "selector_accuracy": acc,
                        "baseline_accuracy_on_known_test": baseline_test_acc,
                        "gain_vs_baseline_known": (
                            acc - baseline_test_acc
                            if np.isfinite(baseline_test_acc)
                            and int(a["baseline_known_N"]) == int(a["N"])
                            else float("nan")
                        ),
                        "baseline_known_N": int(a["baseline_known_N"]),
                        "wrong_to_correct": int(a["w2c"]),
                        "correct_to_wrong": int(a["c2w"]),
                        "net_w2c_minus_c2w": int(a["w2c"] - a["c2w"]),
                        "mean_selector_margin": float(
                            a["sum_margin"] / n
                        ),
                        "topK_heads": ";".join(
                            hname(L, h) for L, h in top_by_k[K]
                        ),
                    }
                )

        summary_rows.sort(
            key=lambda r: (
                -float(r["selector_accuracy"]),
                int(r["K"]),
                str(r["method"]),
            )
        )

        write_csv(outdir / "selector_summary.csv", summary_rows)
        write_csv(outdir / "selector_per_sample.csv", per_sample_rows)
        write_csv(outdir / "errors.csv", all_errors)

        # ------------------------------------------------------------------
        # E) Console summary.
        # ------------------------------------------------------------------
        print()
        print("=" * 150)
        print("ALL-440 SELECTOR RESULTS -- HEADS FROZEN FROM UNLABELED CAUSAL RANKING")
        print("=" * 150)
        print(
            f"{'rank':>4} {'K':>3} {'method':>23} "
            f"{'acc':>8} {'base':>8} {'gain':>8} "
            f"{'W2C':>5} {'C2W':>5} {'net':>5} {'margin':>9}"
        )
        for rank, r in enumerate(summary_rows, 1):
            print(
                f"{rank:4d} {int(r['K']):3d} "
                f"{r['method']:>23} "
                f"{float(r['selector_accuracy']):8.4f} "
                f"{float(r['baseline_accuracy_on_known_test']):8.4f} "
                f"{float(r['gain_vs_baseline_known']):8.4f} "
                f"{int(r['wrong_to_correct']):5d} "
                f"{int(r['correct_to_wrong']):5d} "
                f"{int(r['net_w2c_minus_c2w']):5d} "
                f"{float(r['mean_selector_margin']):9.4f}"
            )

        print()
        print("=" * 150)
        print("TOP-K HEAD SETS")
        print("=" * 150)
        for K in top_ks:
            print(
                f"K={K:2d}: "
                + ", ".join(hname(L, h) for L, h in top_by_k[K])
            )

        # Config.
        config = vars(args).copy()
        config.update(
            {
                "resolved_source_layers": source_layers,
                "resolved_readout_layer": readout_layer,
                "n_heads": n_heads,
                "head_dim": head_dim,
                "hidden_size": hidden_size,
                "candidate_head_N": len(source_layers) * n_heads,
                "coco_total_N": len(coco_rows),
                "unlabeled_causal_ranking_N": len(calib_rows),
                "evaluation_requested_N": len(test_rows),
                "evaluation_effective_N": len(effective_test_rows),
                "protocol": "all-target unlabeled transductive causal head ranking",
                "synthetic_N": len(syn_meta),
                "readout_source_vector_definition": src_definition,
                "readout_half_gap_H": readout_basis["halfH"],
                "readout_half_gap_V": readout_basis["halfV"],
                "head_selection_uses_coco_gt": False,
                "selector_uses_coco_gt": False,
                "prior_accuracy_used_for_selection": False,
                "target_inputs_used_for_head_selection": True,
                "target_labels_used_for_head_selection": False,
            }
        )
        (outdir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print()
        print(f"[saved] {outdir}")
        print(
            f"[success] causal_heads={len(causal_rows)} "
            f"test_N={len(effective_test_rows)} "
            f"errors={len(all_errors)}"
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
