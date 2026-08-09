#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal validation of GroundingDINO-grounded horizontal / vertical hidden-space axes.

The script is designed for the AdaptVis llava16 branch and reuses the model / dataset /
prompt utilities from:

    eval_coco_centroid_guided_generation_v1.py
    extract_two_object_relation_states.py

It learns beta_x / beta_y ONLY from a TRAIN split of saved raw-question hidden states +
GroundingDINO bboxes, then evaluates interventions ONLY on the held-out TEST split.

Primary interventions at one residual-stream site (default: B token span):

  x_plus / x_minus
      h <- h +/- strength * scale_x * unit(beta_x)

  y_plus / y_minus
      h <- h +/- strength * scale_y * unit(beta_y)

  remove_x
      r = h_img - h_noimg(saved)
      h <- h - <r, unit(beta_x)> unit(beta_x)

  remove_y
      analogous for beta_y

  random_x_plus / random_x_minus
  random_y_plus / random_y_minus
      matched-norm steering along a deterministic random direction orthogonal to beta_x,beta_y

  remove_random
      remove the image-conditioned residual component along that random direction

Optional diagnostic variants:

  oracle_axis
      use GT only to choose +x/-x/+y/-y. This is NOT a deployable method; it tests
      whether the discovered axis can causally control the correct answer.

  anti_oracle_axis
      opposite of oracle_axis, a falsification / reversibility control.

  remove_xy
      remove both beta_x and beta_y residual components.

Two evaluation modes are supported:

  fixed
      one normal forward pass; score the four one-token answer labels at the generation
      boundary. Reports horizontal margin (right-left), vertical margin (below-above),
      accuracy, repair/damage, main-axis and cross-axis effects.

  generation
      true greedy model.generate() with the hidden-state hook applied on the prefill pass.
      Reports parsed autoregressive answer accuracy and repair/damage.

  both
      run both.

Why saved no-image states are used for removal
---------------------------------------------
The learned axes come from image-conditioned residuals:

    r = h_img - h_noimg

For remove_x/remove_y, we therefore remove the beta component of r, not the absolute
projection of raw h_img. This avoids deleting a large task/text baseline that happens to
project onto the same direction.

Layer alignment check
---------------------
Saved NPZ `decoder_block_index=L25` may correspond either to block-25 output or to the
residual stream entering block 25, depending on the state-extraction implementation.
Before causal evaluation, the script captures candidate block outputs around L25 and
compares them with the saved raw vectors. With --hook-layer auto, it automatically picks
the candidate with highest mean cosine. If the best alignment is below
--min-state-alignment, the run aborts by default because a prompt/layer mismatch would
invalidate the intervention.

Important interpretation
------------------------
+/- steering shows SUFFICIENCY / directional controllability.
remove_x/remove_y shows NECESSITY only if removal selectively harms the corresponding
axis and random controls do not. This is stronger than probe/cosine evidence, but still
intervenes at one chosen residual-stream site rather than proving a complete circuit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import random
import re
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")

try:
    import eval_coco_centroid_guided_generation_v1 as core
except Exception as exc:
    raise RuntimeError(
        "This script expects eval_coco_centroid_guided_generation_v1.py in the AdaptVis root/PYTHONPATH."
    ) from exc


SCRIPT_VERSION = "eval-coco-gdino-spatial-axis-causal-v1"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
HORIZONTAL = {"left", "right"}
VERTICAL = {"above", "below"}
EPS = 1e-12

DEFAULT_VARIANTS = (
    "x_plus",
    "x_minus",
    "y_plus",
    "y_minus",
    "remove_x",
    "remove_y",
    "random_x_plus",
    "random_x_minus",
    "random_y_plus",
    "random_y_minus",
    "remove_random",
    "remove_xy",
    "oracle_axis",
    "anti_oracle_axis",
)

GEOM_FEATURES = (
    "pair_cx", "pair_cy", "dx", "dy", "wA", "hA", "wB", "hB"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--state-dir", required=True,
                   help="Contains raw__correct__all_layers.npz and raw__no_image__all_layers.npz")
    p.add_argument("--correct-npz", default=None)
    p.add_argument("--noimage-npz", default=None)
    p.add_argument("--bbox-jsonl", default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default=None,
                   help="Must be the SAME question file used when the saved states were extracted.")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2", "none"])
    p.add_argument("--fixed-layer", type=int, default=25,
                   help="Layer label in the saved state NPZ used to learn beta_x/beta_y.")
    p.add_argument("--hook-layer", default="auto",
                   help="Actual decoder block whose output is patched. 'auto' checks L-1/L/L+1 against saved states.")
    p.add_argument("--target", choices=["A", "B", "last"], default="B",
                   help="Token/span whose residual-stream state is patched and from which axes are learned.")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="lambda = ridge * trace(X'X)/P")
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument("--require-gt-consistent", action="store_true",
                   help="Sensitivity analysis only; primary analysis should leave this OFF.")
    p.add_argument("--mode", choices=["fixed", "generation", "both"], default="both")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Steering magnitude in units of TRAIN residual projection std along each axis.")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--max-eval-samples", type=int, default=None,
                   help="Optional stratified subset of held-out samples. Useful for smoke tests.")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--alignment-samples", type=int, default=8)
    p.add_argument("--min-state-alignment", type=float, default=0.90)
    p.add_argument("--allow-low-alignment", action="store_true")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def csv_list(x: str) -> List[str]:
    return [s.strip() for s in str(x).split(",") if s.strip()]


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < EPS:
        return float("nan")
    return float(np.dot(a, b) / den)


def norm_relation(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().replace("-", "_")
    aliases = {
        "left": "left", "leftward": "left",
        "right": "right", "rightward": "right",
        "above": "above", "over": "above", "top": "above", "on": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return aliases.get(s, s if s in RELATIONS else None)


def parse_generated_relation(text: str) -> Optional[str]:
    text = str(text).strip().lower()
    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(below|under|beneath|bottom)\b", "below"),
        (r"\b(above|over|on top|top)\b", "above"),
        (r"\bon\b", "above"),
    ]
    hits = []
    for pattern, rel in patterns:
        m = re.search(pattern, text)
        if m:
            hits.append((m.start(), rel))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def canonical_phrase(x: Any) -> str:
    s = " ".join(str(x).strip().split())
    return s


def extract_state_metadata(obj: Mapping[str, Any]) -> Dict[str, Any]:
    raw = obj.get("metadata_json")
    if raw is None:
        return {}
    try:
        arr = np.asarray(raw, dtype=object)
        value = arr.item() if arr.ndim == 0 or arr.size == 1 else arr.reshape(-1)[0]
        return json.loads(str(value))
    except Exception:
        return {}


def recursive_prompt_candidates(obj: Any) -> List[str]:
    found: List[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            lk = str(k).lower()
            if "prompt" in lk and isinstance(v, str) and v.endswith((".jsonl", ".json")):
                found.append(v)
            found.extend(recursive_prompt_candidates(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(recursive_prompt_candidates(v))
    return found


def align_state_files(correct: Mapping[str, Any], noimg: Mapping[str, Any]) -> Dict[str, Any]:
    required = {
        "sample_index", "relation", "subject", "reference", "decoder_block_index",
        "A_vectors", "B_vectors", "last_vectors",
    }
    for name, obj in (("correct", correct), ("noimg", noimg)):
        miss = sorted(required - set(obj.keys()))
        if miss:
            raise KeyError(f"{name} state NPZ missing keys: {miss}")

    cids = np.asarray(correct["sample_index"], dtype=np.int64)
    nids = np.asarray(noimg["sample_index"], dtype=np.int64)
    cm = {int(s): i for i, s in enumerate(cids.tolist())}
    nm = {int(s): i for i, s in enumerate(nids.tolist())}
    common = [int(s) for s in cids.tolist() if int(s) in nm]
    ci = np.asarray([cm[s] for s in common], dtype=np.int64)
    ni = np.asarray([nm[s] for s in common], dtype=np.int64)

    layers_c = np.asarray(correct["decoder_block_index"], dtype=np.int64)
    layers_n = np.asarray(noimg["decoder_block_index"], dtype=np.int64)
    if not np.array_equal(layers_c, layers_n):
        raise ValueError("correct/no-image decoder_block_index differ")

    result: Dict[str, Any] = {
        "sids": np.asarray(common, dtype=np.int64),
        "layers": layers_c,
        "relation": np.asarray([norm_relation(x) for x in np.asarray(correct["relation"], dtype=object)[ci]], dtype=object),
        "subject": np.asarray([canonical_phrase(x) for x in np.asarray(correct["subject"], dtype=object)[ci]], dtype=object),
        "reference": np.asarray([canonical_phrase(x) for x in np.asarray(correct["reference"], dtype=object)[ci]], dtype=object),
        "metadata": extract_state_metadata(correct),
    }
    for slot, key in (("A", "A_vectors"), ("B", "B_vectors"), ("last", "last_vectors")):
        c = np.asarray(correct[key][ci], dtype=np.float32)
        n = np.asarray(noimg[key][ni], dtype=np.float32)
        result[f"{slot}_raw"] = c
        result[f"{slot}_noimg"] = n
        result[f"{slot}_residual"] = c - n
    return result


def load_gdino_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows[int(row["sid"])] = row
            except Exception as exc:
                raise ValueError(f"Bad JSONL {path}:{line_no}: {exc}") from exc
    return rows


def selected_box(obj: Mapping[str, Any]) -> Tuple[np.ndarray, float, bool]:
    sel = obj.get("selected")
    if not isinstance(sel, Mapping):
        raise ValueError("missing selected bbox")
    box = sel.get("box_xyxy_normalized")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("missing box_xyxy_normalized")
    b = np.asarray([float(v) for v in box], dtype=np.float64)
    if not np.all(np.isfinite(b)):
        raise ValueError("non-finite bbox")
    x1, y1, x2, y2 = b.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate bbox")
    score = float(sel.get("score", float("nan")))
    ambiguous = bool(obj.get("ambiguous", False))
    return b, score, ambiguous


def box_stats(b: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1)


def relation_consistent(rel: str, dx: float, dy: float) -> bool:
    if rel == "left":
        return dx < 0
    if rel == "right":
        return dx > 0
    if rel == "above":
        return dy < 0
    if rel == "below":
        return dy > 0
    return False


def build_geometry_cohort(
    states: Mapping[str, Any],
    bbox_rows: Mapping[int, Mapping[str, Any]],
    fixed_layer: int,
    target: str,
    min_score: float,
    include_ambiguous: bool,
    require_gt_consistent: bool,
) -> Dict[str, Any]:
    layers = np.asarray(states["layers"], dtype=np.int64)
    hits = np.where(layers == int(fixed_layer))[0]
    if len(hits) != 1:
        raise ValueError(f"fixed layer L{fixed_layer} occurs {len(hits)} times in {layers.tolist()}")
    li = int(hits[0])

    state_sids = np.asarray(states["sids"], dtype=np.int64)
    relations = np.asarray(states["relation"], dtype=object)
    subjects = np.asarray(states["subject"], dtype=object)
    references = np.asarray(states["reference"], dtype=object)
    raw_all = np.asarray(states[f"{target}_raw"], dtype=np.float32)
    no_all = np.asarray(states[f"{target}_noimg"], dtype=np.float32)
    res_all = np.asarray(states[f"{target}_residual"], dtype=np.float32)

    rows: List[Dict[str, Any]] = []
    X: List[List[float]] = []
    Yraw: List[np.ndarray] = []
    Yno: List[np.ndarray] = []
    Yres: List[np.ndarray] = []
    skipped = defaultdict(int)

    for i, sid in enumerate(state_sids.tolist()):
        rel = relations[i]
        if rel not in RELATIONS:
            skipped["bad_relation"] += 1
            continue
        gd = bbox_rows.get(int(sid))
        if gd is None:
            skipped["bbox_missing"] += 1
            continue
        try:
            if not bool(gd.get("both_found", True)):
                raise ValueError("both_found=false")
            a_box, a_score, a_amb = selected_box(gd["subject"])
            b_box, b_score, b_amb = selected_box(gd["reference"])
            amb = bool(gd.get("either_ambiguous", False)) or a_amb or b_amb
            if not include_ambiguous and amb:
                skipped["ambiguous"] += 1
                continue
            if not (math.isfinite(a_score) and math.isfinite(b_score)):
                skipped["score_nonfinite"] += 1
                continue
            if min(a_score, b_score) < min_score:
                skipped["score_low"] += 1
                continue
            ax, ay, aw, ah = box_stats(a_box)
            bx, by, bw, bh = box_stats(b_box)
            dx, dy = ax - bx, ay - by
            consistent = relation_consistent(str(rel), dx, dy)
            if require_gt_consistent and not consistent:
                skipped["gt_inconsistent"] += 1
                continue
            feat = [0.5 * (ax + bx), 0.5 * (ay + by), dx, dy, aw, ah, bw, bh]
        except Exception:
            skipped["bbox_invalid"] += 1
            continue

        rows.append({
            "sid": int(sid), "relation": str(rel), "subject": str(subjects[i]), "reference": str(references[i]),
            "A_score": a_score, "B_score": b_score, "ambiguous": amb, "gt_consistent": consistent,
            **{name: float(v) for name, v in zip(GEOM_FEATURES, feat)},
        })
        X.append(feat)
        Yraw.append(raw_all[i, li].astype(np.float32, copy=False))
        Yno.append(no_all[i, li].astype(np.float32, copy=False))
        Yres.append(res_all[i, li].astype(np.float32, copy=False))

    if len(rows) < 40:
        raise RuntimeError(f"Only {len(rows)} usable bbox/state samples; skipped={dict(skipped)}")

    return {
        "rows": rows,
        "sids": np.asarray([r["sid"] for r in rows], dtype=np.int64),
        "labels": np.asarray([r["relation"] for r in rows], dtype=object),
        "subjects": np.asarray([r["subject"] for r in rows], dtype=object),
        "references": np.asarray([r["reference"] for r in rows], dtype=object),
        "X": np.asarray(X, dtype=np.float64),
        "Yraw": np.stack(Yraw).astype(np.float32),
        "Yno": np.stack(Yno).astype(np.float32),
        "Yres": np.stack(Yres).astype(np.float32),
        "skipped": dict(skipped),
        "layer_index": li,
    }


def stratified_split(labels: np.ndarray, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.1 <= train_ratio <= 0.9):
        raise ValueError("--train-ratio must be between 0.1 and 0.9")
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for rel in RELATIONS:
        idx = np.where(labels == rel)[0]
        rng.shuffle(idx)
        ntr = int(round(len(idx) * train_ratio))
        ntr = min(max(ntr, 1), len(idx) - 1)
        tr.extend(idx[:ntr].tolist())
        te.extend(idx[ntr:].tolist())
    tr = np.asarray(sorted(tr), dtype=np.int64)
    te = np.asarray(sorted(te), dtype=np.int64)
    return tr, te


def stratified_limit(indices: np.ndarray, labels: np.ndarray, max_n: Optional[int], seed: int) -> np.ndarray:
    if max_n is None or len(indices) <= max_n:
        return indices
    rng = np.random.default_rng(seed + 991)
    by_rel = {r: indices[labels[indices] == r].copy() for r in RELATIONS}
    for arr in by_rel.values():
        rng.shuffle(arr)
    selected: List[int] = []
    # round robin keeps proportions roughly balanced even for tiny smoke tests
    while len(selected) < max_n and any(len(v) for v in by_rel.values()):
        for rel in RELATIONS:
            if len(selected) >= max_n:
                break
            arr = by_rel[rel]
            if len(arr):
                selected.append(int(arr[0]))
                by_rel[rel] = arr[1:]
    return np.asarray(sorted(selected), dtype=np.int64)


def fit_axes(X: np.ndarray, Yres: np.ndarray, Yraw: np.ndarray, tr: np.ndarray, ridge: float, seed: int) -> Dict[str, Any]:
    Xtr = np.asarray(X[tr], dtype=np.float64)
    Ytr = np.asarray(Yres[tr], dtype=np.float64)
    xmu = Xtr.mean(axis=0)
    xsd = Xtr.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xz = (Xtr - xmu) / xsd
    ymu = Ytr.mean(axis=0)
    Yc = Ytr - ymu
    gram = Xz.T @ Xz
    lam = float(ridge) * float(np.trace(gram) / max(Xz.shape[1], 1))
    W = np.linalg.solve(gram + lam * np.eye(Xz.shape[1]), Xz.T @ Yc)
    beta_x = np.asarray(W[2], dtype=np.float64)
    beta_y = np.asarray(W[3], dtype=np.float64)
    nx, ny = np.linalg.norm(beta_x), np.linalg.norm(beta_y)
    if nx < EPS or ny < EPS:
        raise RuntimeError("Degenerate beta_x/beta_y")
    ux, uy = beta_x / nx, beta_y / ny

    rng = np.random.default_rng(seed + 12345)
    ur = rng.normal(size=ux.shape).astype(np.float64)
    ur = ur - np.dot(ur, ux) * ux - np.dot(ur, uy) * uy
    nr = np.linalg.norm(ur)
    if nr < EPS:
        raise RuntimeError("Degenerate random control direction")
    ur = ur / nr

    proj_x = np.asarray(Yres[tr], dtype=np.float64) @ ux
    proj_y = np.asarray(Yres[tr], dtype=np.float64) @ uy
    scale_x = float(np.std(proj_x))
    scale_y = float(np.std(proj_y))
    if scale_x < 1e-8 or scale_y < 1e-8:
        raise RuntimeError(f"Degenerate axis scale: x={scale_x}, y={scale_y}")

    # Raw-state means are diagnostic only; removal uses per-sample no-image state.
    raw_mean = np.asarray(Yraw[tr], dtype=np.float64).mean(axis=0)

    return {
        "W": W.astype(np.float32), "beta_x": beta_x.astype(np.float32), "beta_y": beta_y.astype(np.float32),
        "ux": ux.astype(np.float32), "uy": uy.astype(np.float32), "urand": ur.astype(np.float32),
        "scale_x": scale_x, "scale_y": scale_y, "xmu": xmu, "xsd": xsd,
        "residual_mean": ymu.astype(np.float32), "raw_mean": raw_mean.astype(np.float32),
        "axis_cosine": float(np.dot(ux, uy)), "random_x_cosine": float(np.dot(ur, ux)),
        "random_y_cosine": float(np.dot(ur, uy)), "ridge_lambda": lam,
    }


def load_prompt_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "question" not in row:
                raise ValueError(f"{path}:{line_no}: need id/question")
            sid = int(row["id"])
            rawq = str(row["question"])
            qtext = core.extract_standard_user_text(rawq) if hasattr(core, "extract_standard_user_text") else rawq
            ans = row.get("answer")
            if isinstance(ans, (list, tuple)):
                ans = ans[0] if ans else None
            rows[sid] = {"question_text": qtext, "raw_question": rawq, "answer": ans}
    return rows


def resolve_prompt_path(args: argparse.Namespace, state_meta: Mapping[str, Any]) -> Path:
    if args.prompt_jsonl:
        return Path(args.prompt_jsonl)
    for cand in recursive_prompt_candidates(state_meta):
        p = Path(cand)
        if p.exists():
            print(f"Using prompt path recovered from state metadata: {p}")
            return p
    default = Path("prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    if default.exists():
        print(f"WARNING: --prompt-jsonl omitted; falling back to {default}")
        return default
    raise FileNotFoundError(
        "Could not infer prompt JSONL. Pass --prompt-jsonl with the EXACT prompt file used to extract states."
    )


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Cannot find tensor in output type {type(output)}")


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for i, x in enumerate(items):
            if torch.is_tensor(x):
                items[i] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, x in enumerate(items):
            if torch.is_tensor(x):
                items[i] = replacement
                return items
    raise TypeError(f"Cannot replace tensor in output type {type(output)}")


def target_indices_from_batch(processor: Any, batch: Mapping[str, Any], subject: str, reference: str, target: str) -> List[int]:
    ids = batch["input_ids"][0].detach().cpu().tolist()
    if target == "last":
        return [len(ids) - 1]
    s_span, r_span = core.locate_object_spans(processor.tokenizer, ids, subject, reference)
    span = s_span if target == "A" else r_span
    return list(range(int(span[0]), int(span[1]) + 1))


def make_batch(processor: Any, record: Any, question_text: str, device: torch.device) -> Tuple[Dict[str, Any], Any]:
    rendered = core.build_prompt(processor, question_text)
    image = core.record_image(record)
    batch = processor(text=[rendered], images=[image], return_tensors="pt")
    batch = core.move_batch(batch, device)
    return batch, image


def relation_scores(logits: torch.Tensor, label_ids: Mapping[str, Sequence[int]]) -> np.ndarray:
    if logits.ndim == 2:
        logits = logits[-1]
    vals = []
    for rel in RELATIONS:
        ids = torch.as_tensor(list(label_ids[rel]), device=logits.device, dtype=torch.long)
        vals.append(logits.index_select(0, ids).max())
    return torch.stack(vals).detach().float().cpu().numpy().astype(np.float32)


def score_diagnostics(scores: np.ndarray, gt: str) -> Dict[str, Any]:
    pred = RELATIONS[int(np.argmax(scores))]
    gi = RELATIONS.index(gt)
    other = scores.copy()
    other[gi] = -np.inf
    gt_margin = float(scores[gi] - np.max(other))
    return {
        "prediction": pred,
        "correct": bool(pred == gt),
        "gt_margin": gt_margin,
        "x_margin": float(scores[1] - scores[0]),   # right - left
        "y_margin": float(scores[3] - scores[2]),   # below - above
        **{f"logit_{r}": float(scores[i]) for i, r in enumerate(RELATIONS)},
    }


def decode_new_tokens(processor: Any, output_ids: torch.Tensor, input_length: int) -> str:
    return processor.tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def capture_alignment_for_sample(
    model: Any,
    batch: Dict[str, Any],
    layers: Sequence[torch.nn.Module],
    candidates: Sequence[int],
    target_indices: Sequence[int],
) -> Dict[int, np.ndarray]:
    captured: Dict[int, np.ndarray] = {}
    handles = []
    for li in candidates:
        def make_hook(layer_index: int):
            def hook(_m: Any, _inp: Any, out: Any) -> None:
                t = first_tensor(out)
                idx = torch.as_tensor(list(target_indices), device=t.device, dtype=torch.long)
                v = t[0].index_select(0, idx).float().mean(0).detach().cpu().numpy().astype(np.float32)
                captured[layer_index] = v
            return hook
        handles.append(layers[li].register_forward_hook(make_hook(li)))
    try:
        with torch.inference_mode():
            _ = model(**batch, output_hidden_states=False, output_attentions=False, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def choose_hook_layer(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    layers: Sequence[torch.nn.Module],
    records_by_sid: Mapping[int, Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    cohort: Mapping[str, Any],
    eval_idx: np.ndarray,
    device: torch.device,
) -> Tuple[int, Dict[str, Any]]:
    if args.hook_layer != "auto":
        chosen = int(args.hook_layer)
        candidates = [chosen]
    else:
        candidates = [x for x in (args.fixed_layer - 1, args.fixed_layer, args.fixed_layer + 1) if 0 <= x < len(layers)]
    if not candidates:
        raise ValueError("No candidate hook layers")

    n = min(args.alignment_samples, len(eval_idx))
    # spread across labels rather than taking first sorted examples
    selected = stratified_limit(eval_idx, cohort["labels"], n, args.seed + 700)
    per_layer: Dict[int, List[float]] = {li: [] for li in candidates}
    per_sample = []
    for idx in selected:
        sid = int(cohort["sids"][idx])
        record = records_by_sid.get(sid)
        prompt = prompt_rows.get(sid)
        if record is None or prompt is None:
            continue
        batch = image = None
        try:
            batch, image = make_batch(processor, record, str(prompt["question_text"]), device)
            target_indices = target_indices_from_batch(
                processor, batch, str(cohort["subjects"][idx]), str(cohort["references"][idx]), args.target
            )
            caps = capture_alignment_for_sample(model, batch, layers, candidates, target_indices)
            saved = np.asarray(cohort["Yraw"][idx], dtype=np.float32)
            row = {"sid": sid}
            for li in candidates:
                c = cosine(caps[li], saved) if li in caps else float("nan")
                row[f"L{li}"] = c
                if math.isfinite(c):
                    per_layer[li].append(c)
            per_sample.append(row)
        finally:
            if batch is not None:
                del batch
            if image is not None:
                try: image.close()
                except Exception: pass
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    means = {li: safe_mean(vals) for li, vals in per_layer.items()}
    if args.hook_layer == "auto":
        chosen = max(candidates, key=lambda x: means.get(x, -math.inf))
    best = means.get(chosen, float("nan"))
    report = {"candidates": candidates, "mean_cosine": means, "chosen_hook_layer": chosen, "samples": per_sample}
    print("\n" + "=" * 112)
    print("STATE ↔ FORWARD-HOOK ALIGNMENT")
    print("=" * 112)
    print(f"saved state layer label: L{args.fixed_layer} | target={args.target}")
    for li in candidates:
        print(f"hook block L{li:2d} output -> saved state cosine = {means[li]:+.4f}")
    print(f"chosen hook layer: L{chosen} | mean cosine={best:+.4f}")
    if not math.isfinite(best) or best < args.min_state_alignment:
        msg = (
            f"State alignment {best:.4f} < --min-state-alignment={args.min_state_alignment:.4f}. "
            "Likely prompt/model/layer mismatch. Pass the exact --prompt-jsonl or inspect layer indexing."
        )
        if args.allow_low_alignment:
            print("WARNING: " + msg)
        else:
            raise RuntimeError(msg)
    return chosen, report


def variant_axis(variant: str) -> Optional[str]:
    if variant.startswith("x_") or variant.startswith("random_x") or variant == "remove_x":
        return "x"
    if variant.startswith("y_") or variant.startswith("random_y") or variant == "remove_y":
        return "y"
    return None


def compute_delta(
    variant: str,
    current_mean: torch.Tensor,
    noimg_cpu: torch.Tensor,
    axes: Mapping[str, Any],
    gt: str,
    strength: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    device = current_mean.device
    dtype = current_mean.dtype
    ux = torch.as_tensor(axes["ux"], device=device, dtype=torch.float32)
    uy = torch.as_tensor(axes["uy"], device=device, dtype=torch.float32)
    ur = torch.as_tensor(axes["urand"], device=device, dtype=torch.float32)
    h = current_mean.float()
    no = noimg_cpu.to(device=device, dtype=torch.float32)
    residual = h - no
    px = float(torch.dot(residual, ux).item())
    py = float(torch.dot(residual, uy).item())
    pr = float(torch.dot(residual, ur).item())
    sx, sy = float(axes["scale_x"]), float(axes["scale_y"])

    uses_gt = False
    if variant == "x_plus":
        delta = +strength * sx * ux
    elif variant == "x_minus":
        delta = -strength * sx * ux
    elif variant == "y_plus":
        delta = +strength * sy * uy
    elif variant == "y_minus":
        delta = -strength * sy * uy
    elif variant == "remove_x":
        delta = -px * ux
    elif variant == "remove_y":
        delta = -py * uy
    elif variant == "remove_xy":
        # Sequential orthogonal projection is virtually identical here because ux,uy are near orthogonal.
        # Use the exact 2D least-squares projection to avoid relying on perfect orthogonality.
        U = torch.stack([ux, uy], dim=1)  # [D,2]
        gram = U.T @ U
        coeff = torch.linalg.solve(gram, U.T @ residual)
        delta = -(U @ coeff)
    elif variant == "random_x_plus":
        delta = +strength * sx * ur
    elif variant == "random_x_minus":
        delta = -strength * sx * ur
    elif variant == "random_y_plus":
        delta = +strength * sy * ur
    elif variant == "random_y_minus":
        delta = -strength * sy * ur
    elif variant == "remove_random":
        delta = -pr * ur
    elif variant in ("oracle_axis", "anti_oracle_axis"):
        uses_gt = True
        sign = -1.0 if variant == "anti_oracle_axis" else 1.0
        if gt == "right":
            delta = sign * strength * sx * ux
        elif gt == "left":
            delta = -sign * strength * sx * ux
        elif gt == "below":
            delta = sign * strength * sy * uy
        elif gt == "above":
            delta = -sign * strength * sy * uy
        else:
            raise ValueError(gt)
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    after = residual + delta
    info = {
        "variant": variant,
        "uses_gt": uses_gt,
        "residual_norm_before": float(residual.norm().item()),
        "delta_norm": float(delta.norm().item()),
        "proj_x_before": px,
        "proj_y_before": py,
        "proj_random_before": pr,
        "proj_x_after_est": float(torch.dot(after, ux).item()),
        "proj_y_after_est": float(torch.dot(after, uy).item()),
        "proj_random_after_est": float(torch.dot(after, ur).item()),
    }
    return delta.to(dtype=dtype), info


def run_fixed_intervention(
    model: Any,
    batch: Dict[str, Any],
    layer_module: torch.nn.Module,
    target_indices: Sequence[int],
    variant: str,
    noimg_vec: np.ndarray,
    axes: Mapping[str, Any],
    gt: str,
    strength: float,
) -> Tuple[Any, Dict[str, Any]]:
    diag: Dict[str, Any] = {}
    noimg_cpu = torch.as_tensor(noimg_vec, dtype=torch.float32, device="cpu")
    applied = False

    def hook(_m: Any, _inp: Any, out: Any) -> Any:
        nonlocal diag, applied
        t = first_tensor(out)
        if applied:
            return out
        if t.ndim != 3 or int(t.shape[0]) != 1:
            raise RuntimeError(f"Unexpected block output shape {tuple(t.shape)}")
        if max(target_indices) >= int(t.shape[1]):
            return out
        idx = torch.as_tensor(list(target_indices), device=t.device, dtype=torch.long)
        mean = t[0].index_select(0, idx).float().mean(0)
        delta, diag = compute_delta(variant, mean, noimg_cpu, axes, gt, strength)
        modified = t.clone()
        modified[0, idx, :] += delta.to(device=modified.device, dtype=modified.dtype).unsqueeze(0)
        applied = True
        return replace_first_tensor(out, modified)

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=False, output_attentions=False, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if not applied:
        raise RuntimeError(f"Intervention hook never applied for {variant}")
    return outputs, diag


def run_generation_intervention(
    model: Any,
    processor: Any,
    batch: Dict[str, Any],
    layer_module: torch.nn.Module,
    target_indices: Sequence[int],
    variant: str,
    noimg_vec: np.ndarray,
    axes: Mapping[str, Any],
    gt: str,
    strength: float,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    diag: Dict[str, Any] = {}
    noimg_cpu = torch.as_tensor(noimg_vec, dtype=torch.float32, device="cpu")
    applied = False
    input_length = int(batch["input_ids"].shape[1])

    def hook(_m: Any, _inp: Any, out: Any) -> Any:
        nonlocal diag, applied
        t = first_tensor(out)
        if applied:
            return out
        if t.ndim != 3 or int(t.shape[0]) != 1:
            return out
        # Apply only on prefill. During cached decode seq_len is usually 1.
        if int(t.shape[1]) < input_length or max(target_indices) >= int(t.shape[1]):
            return out
        idx = torch.as_tensor(list(target_indices), device=t.device, dtype=torch.long)
        mean = t[0].index_select(0, idx).float().mean(0)
        delta, diag = compute_delta(variant, mean, noimg_cpu, axes, gt, strength)
        modified = t.clone()
        modified[0, idx, :] += delta.to(device=modified.device, dtype=modified.dtype).unsqueeze(0)
        applied = True
        return replace_first_tensor(out, modified)

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        handle.remove()
    if not applied:
        raise RuntimeError(f"Generation intervention hook never applied for {variant}")
    text = decode_new_tokens(processor, output_ids, input_length)
    del output_ids
    return text, diag


def baseline_fixed(model: Any, batch: Dict[str, Any]) -> Any:
    with torch.inference_mode():
        return model(**batch, output_hidden_states=False, output_attentions=False, use_cache=False, return_dict=True)


def baseline_generate(model: Any, processor: Any, batch: Dict[str, Any], max_new_tokens: int) -> str:
    n = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    text = decode_new_tokens(processor, out, n)
    del out
    return text


def gt_signed_axis_margin(diag: Mapping[str, Any], gt: str) -> float:
    if gt == "right": return float(diag["x_margin"])
    if gt == "left": return -float(diag["x_margin"])
    if gt == "below": return float(diag["y_margin"])
    if gt == "above": return -float(diag["y_margin"])
    return float("nan")


def summarize_fixed(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_variant: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    baseline = {int(r["sid"]): r for r in by_variant.get("baseline", [])}
    out = []
    for variant, vals in by_variant.items():
        hs = [r for r in vals if r["gt"] in HORIZONTAL]
        vs = [r for r in vals if r["gt"] in VERTICAL]
        paired = [(r, baseline.get(int(r["sid"]))) for r in vals if int(r["sid"]) in baseline]
        changed = repaired = damaged = 0
        dxs = []; dys = []; dax = []; day = []; dgt = []
        for r, b in paired:
            if b is None: continue
            changed += int(r["prediction"] != b["prediction"])
            repaired += int((not bool(b["correct"])) and bool(r["correct"]))
            damaged += int(bool(b["correct"]) and (not bool(r["correct"])))
            dxs.append(float(r["x_margin"]) - float(b["x_margin"]))
            dys.append(float(r["y_margin"]) - float(b["y_margin"]))
            dax.append(abs(float(r["x_margin"])) - abs(float(b["x_margin"])))
            day.append(abs(float(r["y_margin"])) - abs(float(b["y_margin"])))
            dgt.append(gt_signed_axis_margin(r, str(r["gt"])) - gt_signed_axis_margin(b, str(b["gt"])))
        item = {
            "variant": variant,
            "n": len(vals),
            "accuracy": safe_mean(float(r["correct"]) for r in vals),
            "horizontal_accuracy": safe_mean(float(r["correct"]) for r in hs),
            "vertical_accuracy": safe_mean(float(r["correct"]) for r in vs),
            "changed_vs_baseline": changed,
            "repaired": repaired,
            "damaged": damaged,
            "net_repair": repaired - damaged,
            "mean_delta_x_margin": safe_mean(dxs),
            "mean_delta_y_margin": safe_mean(dys),
            "mean_delta_abs_x_margin": safe_mean(dax),
            "mean_delta_abs_y_margin": safe_mean(day),
            "mean_delta_gt_axis_margin": safe_mean(dgt),
            "mean_delta_norm": safe_mean(safe_float(r.get("delta_norm")) for r in vals),
        }
        axis = variant_axis(variant)
        if axis == "x":
            item["main_axis_margin_delta"] = item["mean_delta_x_margin"]
            item["cross_axis_margin_delta"] = item["mean_delta_y_margin"]
        elif axis == "y":
            item["main_axis_margin_delta"] = item["mean_delta_y_margin"]
            item["cross_axis_margin_delta"] = item["mean_delta_x_margin"]
        else:
            item["main_axis_margin_delta"] = float("nan")
            item["cross_axis_margin_delta"] = float("nan")
        out.append(item)
    order = {"baseline": -1, **{v: i for i, v in enumerate(DEFAULT_VARIANTS)}}
    out.sort(key=lambda r: order.get(r["variant"], 999))
    return out


def summarize_generation(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_variant: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    baseline = {int(r["sid"]): r for r in by_variant.get("baseline", [])}
    out = []
    for variant, vals in by_variant.items():
        parsed = [r for r in vals if r.get("prediction") in RELATIONS]
        hs = [r for r in vals if r["gt"] in HORIZONTAL]
        vs = [r for r in vals if r["gt"] in VERTICAL]
        changed = repaired = damaged = 0
        for r in vals:
            b = baseline.get(int(r["sid"]))
            if b is None: continue
            changed += int(r.get("prediction") != b.get("prediction"))
            repaired += int((not bool(b.get("correct"))) and bool(r.get("correct")))
            damaged += int(bool(b.get("correct")) and (not bool(r.get("correct"))))
        out.append({
            "variant": variant,
            "n": len(vals),
            "parse_rate": len(parsed) / max(len(vals), 1),
            "accuracy": safe_mean(float(r.get("correct", False)) for r in vals),
            "horizontal_accuracy": safe_mean(float(r.get("correct", False)) for r in hs),
            "vertical_accuracy": safe_mean(float(r.get("correct", False)) for r in vs),
            "changed_vs_baseline": changed,
            "repaired": repaired,
            "damaged": damaged,
            "net_repair": repaired - damaged,
            "mean_delta_norm": safe_mean(safe_float(r.get("delta_norm")) for r in vals),
        })
    order = {"baseline": -1, **{v: i for i, v in enumerate(DEFAULT_VARIANTS)}}
    out.sort(key=lambda r: order.get(r["variant"], 999))
    return out


def print_fixed_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 146)
    print("FIXED-STATE CAUSAL AXIS SUMMARY")
    print("right-left margin = x_margin; below-above margin = y_margin")
    print("=" * 146)
    print("variant                | acc    | H acc  | V acc  | Δx margin | Δy margin | ΔGT-axis | repaired | damaged | net")
    print("-" * 146)
    for r in rows:
        print(
            f"{r['variant']:22s} | {r['accuracy']:.4f} | {r['horizontal_accuracy']:.4f} | {r['vertical_accuracy']:.4f} | "
            f"{r['mean_delta_x_margin']:+.4f}   | {r['mean_delta_y_margin']:+.4f}   | {r['mean_delta_gt_axis_margin']:+.4f}   | "
            f"{int(r['repaired']):8d} | {int(r['damaged']):7d} | {int(r['net_repair']):+4d}"
        )


def print_generation_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 118)
    print("AUTOREGRESSIVE GENERATION CAUSAL AXIS SUMMARY")
    print("=" * 118)
    print("variant                | parse  | acc    | H acc  | V acc  | repaired | damaged | net | changed")
    print("-" * 118)
    for r in rows:
        print(
            f"{r['variant']:22s} | {r['parse_rate']:.4f} | {r['accuracy']:.4f} | {r['horizontal_accuracy']:.4f} | "
            f"{r['vertical_accuracy']:.4f} | {int(r['repaired']):8d} | {int(r['damaged']):7d} | "
            f"{int(r['net_repair']):+4d} | {int(r['changed_vs_baseline']):7d}"
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.strength < 0:
        raise ValueError("--strength must be >= 0")
    variants = csv_list(args.variants)
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; allowed={list(DEFAULT_VARIANTS)}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fixed_path = outdir / "fixed_samples.jsonl"
    gen_path = outdir / "generation_samples.jsonl"
    errors_path = outdir / "errors.jsonl"
    for p in (fixed_path, gen_path, errors_path):
        if p.exists(): p.unlink()

    state_dir = Path(args.state_dir)
    correct_path = Path(args.correct_npz) if args.correct_npz else state_dir / "raw__correct__all_layers.npz"
    noimg_path = Path(args.noimage_npz) if args.noimage_npz else state_dir / "raw__no_image__all_layers.npz"
    states = align_state_files(load_npz(correct_path), load_npz(noimg_path))
    bbox_rows = load_gdino_rows(Path(args.bbox_jsonl))
    cohort = build_geometry_cohort(
        states, bbox_rows, args.fixed_layer, args.target, args.min_score,
        args.include_ambiguous, args.require_gt_consistent,
    )

    train_idx, test_idx_all = stratified_split(cohort["labels"], args.train_ratio, args.seed)
    test_idx = stratified_limit(test_idx_all, cohort["labels"], args.max_eval_samples, args.seed)
    axes = fit_axes(cohort["X"], cohort["Yres"], cohort["Yraw"], train_idx, args.ridge, args.seed)

    sign_consistency = safe_mean(float(r["gt_consistent"]) for r in cohort["rows"])
    print("\n" + "=" * 112)
    print("AXIS TRAIN / HELD-OUT COHORT")
    print("=" * 112)
    print(f"usable bbox/state samples={len(cohort['sids'])} | train={len(train_idx)} | held-out={len(test_idx_all)} | eval={len(test_idx)}")
    print(f"target={args.target} | saved layer=L{args.fixed_layer} | bbox sign consistency={sign_consistency*100:.2f}%")
    print(f"beta_x/beta_y cosine={axes['axis_cosine']:+.4f}")
    print(f"steering scale_x={axes['scale_x']:.4f} | scale_y={axes['scale_y']:.4f} (1.0 = one TRAIN residual-projection std)")
    print(f"random control cos(x)={axes['random_x_cosine']:+.5f} | cos(y)={axes['random_y_cosine']:+.5f}")

    prompt_path = resolve_prompt_path(args, states.get("metadata", {}))
    prompt_rows = load_prompt_rows(prompt_path)

    support = importlib.import_module("extract_two_object_relation_states")
    records, audit = support.load_records(args.dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}

    missing = [int(cohort["sids"][i]) for i in test_idx if int(cohort["sids"][i]) not in record_by_sid or int(cohort["sids"][i]) not in prompt_rows]
    if missing:
        raise RuntimeError(f"Held-out sids missing record/prompt: n={len(missing)} first={missing[:10]}")

    if args.model not in support.SPECS:
        raise ValueError(f"Model {args.model!r} not in extract_two_object_relation_states.SPECS")
    spec = support.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers {transformers.__version__} has no {spec.model_class}")
    load_kwargs: Dict[str, Any] = {
        "dtype": core.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl
    print(f"\nLoading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    core.configure_processor(model, processor)
    device = torch.device(args.device)
    decoder_layers, decoder_path = core.resolve_decoder_layers(model)
    label_ids = core.label_token_id_variants(processor.tokenizer)

    hook_layer, alignment = choose_hook_layer(
        args, model, processor, decoder_layers, record_by_sid, prompt_rows, cohort, test_idx, device
    )
    layer_module = decoder_layers[hook_layer]

    # Save axes now; useful for later head/circuit tracing.
    np.savez_compressed(
        outdir / "spatial_axes.npz",
        beta_x=np.asarray(axes["beta_x"], dtype=np.float32),
        beta_y=np.asarray(axes["beta_y"], dtype=np.float32),
        unit_x=np.asarray(axes["ux"], dtype=np.float32),
        unit_y=np.asarray(axes["uy"], dtype=np.float32),
        unit_random=np.asarray(axes["urand"], dtype=np.float32),
        scale_x=np.asarray(axes["scale_x"], dtype=np.float32),
        scale_y=np.asarray(axes["scale_y"], dtype=np.float32),
        train_sids=cohort["sids"][train_idx],
        test_sids=cohort["sids"][test_idx],
        saved_layer=np.asarray(args.fixed_layer, dtype=np.int64),
        hook_layer=np.asarray(hook_layer, dtype=np.int64),
        target=np.asarray(args.target),
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "prompt_jsonl": str(prompt_path),
        "state_dir": str(state_dir),
        "correct_npz": str(correct_path),
        "noimage_npz": str(noimg_path),
        "bbox_jsonl": str(args.bbox_jsonl),
        "target": args.target,
        "saved_layer": args.fixed_layer,
        "hook_layer": hook_layer,
        "decoder_layers_path": decoder_path,
        "train_ratio": args.train_ratio,
        "train_n": int(len(train_idx)),
        "heldout_n": int(len(test_idx_all)),
        "eval_n": int(len(test_idx)),
        "strength": args.strength,
        "variants": variants,
        "mode": args.mode,
        "axis_cosine": axes["axis_cosine"],
        "scale_x": axes["scale_x"],
        "scale_y": axes["scale_y"],
        "bbox_sign_consistency": sign_consistency,
        "alignment": alignment,
        "uses_gt_variants": [v for v in variants if v in ("oracle_axis", "anti_oracle_axis")],
        "audit": audit,
    }
    save_json(outdir / "config.json", config)

    fixed_rows: List[Dict[str, Any]] = []
    gen_rows: List[Dict[str, Any]] = []
    started = time.time()

    for done, idx in enumerate(tqdm(test_idx, desc=f"axis-causal:{args.model}:{args.target}"), 1):
        sid = int(cohort["sids"][idx])
        gt = str(cohort["labels"][idx])
        subject = str(cohort["subjects"][idx])
        reference = str(cohort["references"][idx])
        prompt = prompt_rows[sid]
        qtext = str(prompt["question_text"])
        prompt_gt = norm_relation(prompt.get("answer"))
        if prompt_gt is not None and prompt_gt != gt:
            raise RuntimeError(f"sid={sid}: prompt answer {prompt_gt} != state relation {gt}")

        batch = image = None
        try:
            batch, image = make_batch(processor, record_by_sid[sid], qtext, device)
            target_indices = target_indices_from_batch(processor, batch, subject, reference, args.target)
            noimg_vec = np.asarray(cohort["Yno"][idx], dtype=np.float32)

            baseline_fixed_diag = None
            if args.mode in ("fixed", "both"):
                outputs = baseline_fixed(model, batch)
                scores = relation_scores(outputs.logits[0, -1], label_ids)
                baseline_fixed_diag = score_diagnostics(scores, gt)
                base_row = {
                    "sid": sid, "gt": gt, "subject": subject, "reference": reference,
                    "variant": "baseline", **baseline_fixed_diag,
                    "delta_norm": 0.0, "hook_layer": hook_layer, "target": args.target,
                }
                fixed_rows.append(base_row); append_jsonl(fixed_path, base_row)
                del outputs

                for variant in variants:
                    outputs, diag = run_fixed_intervention(
                        model, batch, layer_module, target_indices, variant, noimg_vec,
                        axes, gt, args.strength,
                    )
                    scores = relation_scores(outputs.logits[0, -1], label_ids)
                    sd = score_diagnostics(scores, gt)
                    row = {
                        "sid": sid, "gt": gt, "subject": subject, "reference": reference,
                        "variant": variant, **sd, **diag,
                        "hook_layer": hook_layer, "target": args.target,
                    }
                    fixed_rows.append(row); append_jsonl(fixed_path, row)
                    del outputs

            baseline_gen_pred = None
            if args.mode in ("generation", "both"):
                text = baseline_generate(model, processor, batch, args.max_new_tokens)
                pred = parse_generated_relation(text)
                baseline_gen_pred = pred
                base_row = {
                    "sid": sid, "gt": gt, "subject": subject, "reference": reference,
                    "variant": "baseline", "prediction": pred, "correct": bool(pred == gt),
                    "generated_text": text, "delta_norm": 0.0, "hook_layer": hook_layer, "target": args.target,
                }
                gen_rows.append(base_row); append_jsonl(gen_path, base_row)
                for variant in variants:
                    text, diag = run_generation_intervention(
                        model, processor, batch, layer_module, target_indices, variant,
                        noimg_vec, axes, gt, args.strength, args.max_new_tokens,
                    )
                    pred = parse_generated_relation(text)
                    row = {
                        "sid": sid, "gt": gt, "subject": subject, "reference": reference,
                        "variant": variant, "prediction": pred, "correct": bool(pred == gt),
                        "generated_text": text, **diag,
                        "hook_layer": hook_layer, "target": args.target,
                    }
                    gen_rows.append(row); append_jsonl(gen_path, row)

            if args.print_every > 0 and (done == 1 or done % args.print_every == 0 or done == len(test_idx)):
                msg = f"[{done}/{len(test_idx)}] sid={sid} GT={gt}"
                if baseline_fixed_diag is not None:
                    msg += f" fixed={baseline_fixed_diag['prediction']}"
                if args.mode in ("generation", "both"):
                    msg += f" gen={baseline_gen_pred}"
                tqdm.write(msg)

        except Exception as exc:
            err = {
                "sid": sid, "gt": gt, "error_type": type(exc).__name__, "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-25:],
            }
            append_jsonl(errors_path, err)
            tqdm.write(f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}")
        finally:
            if batch is not None: del batch
            if image is not None:
                try: image.close()
                except Exception: pass
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    fixed_summary: List[Dict[str, Any]] = []
    generation_summary: List[Dict[str, Any]] = []
    if fixed_rows:
        fixed_summary = summarize_fixed(fixed_rows)
        write_csv(outdir / "fixed_summary.csv", fixed_summary)
        print_fixed_summary(fixed_summary)
    if gen_rows:
        generation_summary = summarize_generation(gen_rows)
        write_csv(outdir / "generation_summary.csv", generation_summary)
        print_generation_summary(generation_summary)

    final = {
        **config,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "fixed_summary": fixed_summary,
        "generation_summary": generation_summary,
    }
    save_json(outdir / "summary.json", final)

    print("\nSaved:")
    print(f"  {outdir / 'spatial_axes.npz'}")
    if fixed_rows:
        print(f"  {fixed_path}")
        print(f"  {outdir / 'fixed_summary.csv'}")
    if gen_rows:
        print(f"  {gen_path}")
        print(f"  {outdir / 'generation_summary.csv'}")
    print(f"  {outdir / 'summary.json'}")
    print(f"  {errors_path}")


if __name__ == "__main__":
    main()
