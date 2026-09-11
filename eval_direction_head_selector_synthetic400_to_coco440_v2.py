#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_direction_head_selector_synthetic400_to_coco440_v2.py

Corrected end-to-end cross-dataset Direction-Head selector.

SOURCE
------
Read synthetic_shapes_4dir_400 DIRECTLY from:
    <synthetic-dir>/labels.jsonl
and images referenced by each JSONL row.

For every synthetic sample and every attention head, extract exactly the same
per-head object residual used by:
    analyze_coco_head_object_residual_direction_probe_v1.py

Definition for head h at layer L:
    r_img     = z_img(subject)   - z_img(reference)
    r_noimage = z_noimg(subject) - z_noimg(reference)
    r_resid   = r_img - r_noimage

where z is the PRE-W_O head slice captured from the input to o_proj.

IMPORTANT:
The default source prompt is intentionally the SAME prompt template as the COCO
Direction-Head probe, so the synthetic source vectors are directly comparable to
an existing COCO relation_vectors.npz produced by that probe.

SOURCE-ONLY HEAD SELECTION
--------------------------
1. Run stratified K-fold CV on Synthetic-400.
2. For each fold, fit the four relation directions only on the source-train fold:
       center_h = mean(z_h)
       d_{h,r}  = normalize(mean(z_h | r) - center_h)
3. Classify held-out source examples by cosine similarity.
4. Rank heads by source OOF accuracy.
5. Freeze the Top-K heads.
6. Refit directions on ALL Synthetic-400.
7. Apply the frozen source codebook + heads to ALL COCO target examples.

No COCO GT is used to fit relation directions or select heads.

TARGET
------
The target is an existing COCO Direction-Head cache:
    <target-direction-dir>/relation_vectors.npz

This should be the output of:
    analyze_coco_head_object_residual_direction_probe_v1.py

Typical run
-----------
CUDA_VISIBLE_DEVICES=0 python -u eval_direction_head_selector_synthetic400_to_coco440_v2.py \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --model qwen-3b \
  --target-direction-dir output/qwen3b_head_object_residual_direction \
  --cv-folds 5 \
  --topks 1,3,5,10,20 \
  --output-dir output/qwen3b_direction_selector_syn400_to_coco440 \
  --overwrite-source-cache

After the first run, omit --overwrite-source-cache to reuse the extracted
Synthetic-400 head residual cache and skip all source model forward passes.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Reuse the repository's exact Direction-Head extraction implementation.
try:
    import analyze_coco_head_object_residual_direction_probe_v1 as dh
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_head_object_residual_direction_probe_v1.py.\n"
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import extract_two_object_relation_states as base
except Exception as exc:
    raise SystemExit(
        "Could not import extract_two_object_relation_states.py.\n"
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "direction-head-selector-synthetic400-to-coco440-v2"
REL = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(REL)}
EPS = 1e-12

SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "top": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "bottom": "below",
}

# Match analyze_coco_head_object_residual_direction_probe_v1.py exactly.
DEFAULT_PROBE_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)

# Repository's original synthetic benchmark prompt, kept only as an ablation.
ORIGINAL_SYNTHETIC_PROMPT = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with left, right, above, or below."
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Raw synthetic source.
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument(
        "--synthetic-labels",
        default=None,
        help="Default: <synthetic-dir>/labels.jsonl",
    )
    p.add_argument("--source-max-samples", type=int, default=None)

    # Must match target extractor settings.
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument(
        "--source-prompt",
        default="probe",
        choices=["probe", "synthetic"],
        help=(
            "probe = exact COCO Direction-Head probe prompt (recommended); "
            "synthetic = repository's original synthetic benchmark prompt"
        ),
    )

    # Existing target cache.
    p.add_argument(
        "--target-direction-dir",
        required=True,
        help="Directory containing target relation_vectors.npz, or NPZ path itself.",
    )

    # Source-only head selection.
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=17)
    p.add_argument("--topks", default="1,3,5,10,20")
    p.add_argument(
        "--candidate-layers",
        default="",
        help="Optional comma-separated decoder layers; empty means all layers.",
    )
    p.add_argument("--temperature", type=float, default=1.0)

    # Optional baseline metadata for correct/wrong subset diagnostics.
    p.add_argument(
        "--target-baseline-csv",
        default="",
        help=(
            "Optional CSV containing sid and either baseline_correct or "
            "gt+baseline_prediction. Not needed for overall selector accuracy."
        ),
    )

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--overwrite-source-cache",
        action="store_true",
        help="Re-extract Synthetic-400 residuals even if source cache exists.",
    )
    return p.parse_args()


# =============================================================================
# Generic utilities
# =============================================================================

def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return SYN_REL_MAP.get(s, s)


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def head_name(l, h):
    return f"L{int(l)}H{int(h):02d}"


def normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def softmax(x, axis=-1, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(axis=axis, keepdims=True), EPS)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def resolve_npz(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    q = p / "relation_vectors.npz"
    if not q.exists():
        raise FileNotFoundError(q)
    return q


# =============================================================================
# Synthetic loader: mirrors repository's load_synthetic()
# =============================================================================

def load_synthetic_rows(args):
    root = Path(args.synthetic_dir)
    labels_path = (
        Path(args.synthetic_labels)
        if args.synthetic_labels
        else root / "labels.jsonl"
    )
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    prompt_template = (
        DEFAULT_PROBE_PROMPT
        if args.source_prompt == "probe"
        else ORIGINAL_SYNTHETIC_PROMPT
    )

    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            raw = str(item["relation"]).strip().lower()
            if raw not in SYN_REL_MAP:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: unsupported relation={raw!r}"
                )

            image_value = Path(str(item["image"]))
            image_path = (
                image_value if image_value.is_absolute() else root / image_value
            )
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()

            rows.append(
                {
                    "sid": int(item.get("id", len(rows))),
                    "image_path": str(image_path),
                    "subject": subject,
                    "reference": reference,
                    "relation": SYN_REL_MAP[raw],
                    "question": prompt_template.format(
                        subject=subject,
                        reference=reference,
                    ),
                }
            )

    rows.sort(key=lambda r: int(r["sid"]))
    if args.source_max_samples is not None:
        rows = rows[: int(args.source_max_samples)]

    counts = Counter(r["relation"] for r in rows)
    missing = [r for r in REL if counts[r] == 0]
    if missing:
        raise RuntimeError(
            f"Synthetic source lacks relations {missing}; counts={dict(counts)}"
        )

    return rows, labels_path


# =============================================================================
# Synthetic extraction: exact same head residual definition as COCO probe
# =============================================================================

def extract_synthetic_source(args, rows, cache_path):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model not in base.SPECS:
        raise KeyError(
            f"Unknown model={args.model!r}; available={sorted(base.SPECS.keys())}"
        )

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)

    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        load_kw["attn_implementation"] = args.attn_impl

    print(f"[source] loading {spec.repo_id}", flush=True)
    model = cls.from_pretrained(spec.repo_id, **load_kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)

    device = torch.device(args.device)
    layers, decoder_path = dh.resolve_decoder_layers(model)
    n_heads, head_dim = dh.scan_shape(model, layers)

    print(
        f"[source] decoder={decoder_path} | layers={len(layers)} | "
        f"heads/layer={n_heads} | head_dim={head_dim}"
    )

    dtype_np = np.float32 if args.keep_fp32 else np.float16

    sids = []
    labels = []
    residuals = []
    errors = []

    for rec in tqdm(rows, desc="extract synthetic direction-head residuals"):
        image = None
        try:
            image = Image.open(rec["image_path"]).convert("RGB")

            # EXACT same function used by the COCO Direction-Head probe.
            z_img = dh.capture_condition(
                model=model,
                processor=processor,
                device=device,
                layers=layers,
                n_heads=n_heads,
                head_dim=head_dim,
                question=rec["question"],
                subject=rec["subject"],
                reference=rec["reference"],
                image=image,
                pool=args.pool,
            )

            z_noimg = dh.capture_condition(
                model=model,
                processor=processor,
                device=device,
                layers=layers,
                n_heads=n_heads,
                head_dim=head_dim,
                question=rec["question"],
                subject=rec["subject"],
                reference=rec["reference"],
                image=None,
                pool=args.pool,
            )

            r_img = z_img[:, :, 0] - z_img[:, :, 1]
            r_noimg = z_noimg[:, :, 0] - z_noimg[:, :, 1]
            r_resid = r_img - r_noimg

            sids.append(int(rec["sid"]))
            labels.append(str(rec["relation"]))
            residuals.append(r_resid.astype(dtype_np))

            del z_img, z_noimg, r_img, r_noimg, r_resid

        except Exception as exc:
            errors.append(
                {
                    "sid": int(rec["sid"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                }
            )
            tqdm.write(
                f"[ERROR] sid={rec['sid']}: {type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not residuals:
        raise RuntimeError("Synthetic extraction produced zero successful samples.")

    X = np.stack(residuals)
    y = np.asarray(labels, dtype=object)
    sid = np.asarray(sids, dtype=int)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_index=sid,
        relation=y,
        residual=X,
        decoder_block_index=np.arange(X.shape[1]),
        head_index=np.arange(X.shape[2]),
    )

    (cache_path.parent / "synthetic_extraction_errors.json").write_text(
        json.dumps(errors, indent=2),
        encoding="utf-8",
    )

    print(
        f"[source] extracted N={len(y)} | shape={X.shape} | "
        f"counts={dict(Counter(y.tolist()))}"
    )
    print(f"[source] cache saved: {cache_path}")

    return sid, y, X


def load_source_cache(cache_path):
    z = np.load(cache_path, allow_pickle=True)
    required = {"sample_index", "relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(
            f"{cache_path} missing arrays: {sorted(missing)}"
        )
    sid = np.asarray(z["sample_index"]).astype(int)
    y = np.asarray([canon_rel(x) for x in z["relation"]], dtype=object)
    X = np.asarray(z["residual"], dtype=np.float32)
    return sid, y, X


# =============================================================================
# Target cache
# =============================================================================

def load_target_cache(path_or_dir):
    p = resolve_npz(path_or_dir)
    z = np.load(p, allow_pickle=True)

    required = {"relation", "residual"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{p} missing arrays: {sorted(missing)}")

    X = np.asarray(z["residual"], dtype=np.float32)
    y = np.asarray([canon_rel(x) for x in z["relation"]], dtype=object)

    if "sample_index" in z.files:
        sid = np.asarray(z["sample_index"]).astype(int)
    else:
        sid = np.arange(len(y), dtype=int)

    valid = np.isin(y, np.asarray(REL, dtype=object))
    return p, sid[valid], y[valid], X[valid]


# =============================================================================
# Source-only codebook fitting / CV
# =============================================================================

def fit_codebook_all_heads(X, y):
    """
    X: [N,L,H,D]
    Returns:
      center [L,H,D]
      dirs   [L,H,R,D]
    """
    center = X.mean(axis=0)
    dirs = np.zeros(
        (X.shape[1], X.shape[2], len(REL), X.shape[3]),
        dtype=np.float32,
    )

    for ri, rel in enumerate(REL):
        m = y == rel
        if not np.any(m):
            raise RuntimeError(f"No source samples for relation={rel}")
        d = X[m].mean(axis=0) - center
        dirs[:, :, ri, :] = normalize(d, axis=-1)

    return center.astype(np.float32), dirs


def score_all_heads(X, center, dirs):
    """
    Frozen source codebook applied to samples X.
    returns [N,L,H,R]
    """
    Xc = X - center[None, :, :, :]
    Xn = normalize(Xc, axis=-1)
    return np.einsum("nlhd,lhrd->nlhr", Xn, dirs, optimize=True)


def stratified_folds(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_folds)]

    for rel in REL:
        idx = np.where(y == rel)[0].copy()
        rng.shuffle(idx)
        pieces = np.array_split(idx, n_folds)
        for f, piece in enumerate(pieces):
            buckets[f].extend(piece.tolist())

    return [np.asarray(sorted(x), dtype=int) for x in buckets]


def source_oof_head_accuracy(X, y, n_folds, seed):
    N, L, H, D = X.shape
    yi = np.asarray([REL_TO_ID[r] for r in y], dtype=int)
    folds = stratified_folds(y, n_folds, seed)
    all_idx = np.arange(N)

    correct = np.zeros((L, H), dtype=np.int64)
    fold_rows = []

    for f, te in enumerate(folds):
        keep = np.ones(N, dtype=bool)
        keep[te] = False
        tr = all_idx[keep]

        center, dirs = fit_codebook_all_heads(X[tr], y[tr])
        scores = score_all_heads(X[te], center, dirs)
        pred = np.argmax(scores, axis=-1)
        ok = pred == yi[te, None, None]

        correct += ok.sum(axis=0)

        fold_acc = ok.mean(axis=0)
        for l in range(L):
            for h in range(H):
                fold_rows.append(
                    {
                        "fold": f,
                        "N_train": len(tr),
                        "N_val": len(te),
                        "layer": l,
                        "head": h,
                        "head_name": head_name(l, h),
                        "fold_accuracy": float(fold_acc[l, h]),
                    }
                )

    return correct.astype(np.float32) / float(N), pd.DataFrame(fold_rows)


def rank_heads(acc, candidate_layers):
    allowed = set(candidate_layers) if candidate_layers else None
    rows = []

    for l in range(acc.shape[0]):
        if allowed is not None and l not in allowed:
            continue
        for h in range(acc.shape[1]):
            rows.append((float(acc[l, h]), int(l), int(h)))

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


# =============================================================================
# Target selector
# =============================================================================

def ensemble_probs(scores, heads, reliability, temperature, weighted, layer_balanced):
    probs = softmax(scores, axis=-1, temperature=temperature)
    N = scores.shape[0]

    if not layer_balanced:
        out = np.zeros((N, len(REL)), dtype=np.float64)
        denom = 0.0
        for l, h in heads:
            w = (
                max(float(reliability[l, h]) - 0.25, 0.0)
                if weighted
                else 1.0
            )
            if w <= 0:
                continue
            out += w * probs[:, l, h, :]
            denom += w

        if denom <= EPS:
            return np.mean(
                np.stack([probs[:, l, h, :] for l, h in heads], axis=0),
                axis=0,
            )
        return out / denom

    by_layer = defaultdict(list)
    for l, h in heads:
        by_layer[l].append(h)

    layer_outputs = []
    for l, hs in sorted(by_layer.items()):
        out = np.zeros((N, len(REL)), dtype=np.float64)
        denom = 0.0
        for h in hs:
            w = (
                max(float(reliability[l, h]) - 0.25, 0.0)
                if weighted
                else 1.0
            )
            if w <= 0:
                continue
            out += w * probs[:, l, h, :]
            denom += w

        if denom <= EPS:
            out = np.mean(probs[:, l, hs, :], axis=1)
        else:
            out /= denom
        layer_outputs.append(out)

    return np.mean(np.stack(layer_outputs, axis=0), axis=0)


def evaluate_prediction(pred_idx, y):
    yi = np.asarray([REL_TO_ID[r] for r in y], dtype=int)
    correct = pred_idx == yi

    row = {
        "N": len(y),
        "acc_all": float(correct.mean()),
    }

    for ri, rel in enumerate(REL):
        m = yi == ri
        row[f"N_{rel}"] = int(m.sum())
        row[f"acc_{rel}"] = (
            float(correct[m].mean()) if np.any(m) else np.nan
        )

    return row, correct


def load_baseline_csv(path):
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    if "sid" not in df.columns:
        raise RuntimeError(f"{p}: missing sid")

    df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)

    if "baseline_correct" in df.columns:
        df["baseline_correct"] = df["baseline_correct"].map(boolify)
    elif {"gt", "baseline_prediction"} <= set(df.columns):
        df["gt"] = df["gt"].map(canon_rel)
        df["baseline_prediction"] = df["baseline_prediction"].map(canon_rel)
        df["baseline_correct"] = df["gt"] == df["baseline_prediction"]
    else:
        raise RuntimeError(
            f"{p}: need baseline_correct or gt+baseline_prediction"
        )

    if "baseline_prediction" in df.columns:
        df["baseline_prediction"] = df["baseline_prediction"].map(canon_rel)

    return df.drop_duplicates("sid").set_index("sid")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    topks = parse_ints(args.topks)
    if not topks:
        raise ValueError("--topks is empty")
    candidate_layers = (
        parse_ints(args.candidate_layers)
        if args.candidate_layers
        else None
    )

    # -------------------------------------------------------------------------
    # 1) Load raw Synthetic-400 and extract/reuse source head residuals.
    # -------------------------------------------------------------------------
    source_rows, labels_path = load_synthetic_rows(args)

    source_cache = outdir / "synthetic_direction_head_relation_vectors.npz"

    if source_cache.exists() and not args.overwrite_source_cache:
        print(f"[source] reusing cache: {source_cache}")
        source_sid, source_y, source_X = load_source_cache(source_cache)
    else:
        source_sid, source_y, source_X = extract_synthetic_source(
            args,
            source_rows,
            source_cache,
        )

    source_y = np.asarray([canon_rel(x) for x in source_y], dtype=object)
    valid_source = np.isin(source_y, np.asarray(REL, dtype=object))
    source_sid = source_sid[valid_source]
    source_y = source_y[valid_source]
    source_X = source_X[valid_source]

    print(
        f"[source] N={len(source_y)} | shape={source_X.shape} | "
        f"counts={dict(Counter(source_y.tolist()))}"
    )

    # -------------------------------------------------------------------------
    # 2) Load ALL COCO target head residuals.
    # -------------------------------------------------------------------------
    target_path, target_sid, target_y, target_X = load_target_cache(
        args.target_direction_dir
    )

    print(
        f"[target] {target_path} | N={len(target_y)} | shape={target_X.shape} | "
        f"counts={dict(Counter(target_y.tolist()))}"
    )

    if source_X.shape[1:] != target_X.shape[1:]:
        raise RuntimeError(
            "Source/target residual geometry mismatch.\n"
            f"source={source_X.shape}\n"
            f"target={target_X.shape}\n"
            "Use the same VLM and the same Direction-Head extraction definition."
        )

    if min(Counter(source_y.tolist()).values()) < args.cv_folds:
        raise RuntimeError(
            f"Cannot run {args.cv_folds}-fold source CV; "
            f"source counts={dict(Counter(source_y.tolist()))}"
        )

    # -------------------------------------------------------------------------
    # 3) Source-only OOF reliability -> Top heads.
    # -------------------------------------------------------------------------
    print(
        f"[source] computing {args.cv_folds}-fold stratified OOF head accuracy..."
    )
    source_cv_acc, fold_df = source_oof_head_accuracy(
        source_X,
        source_y,
        args.cv_folds,
        args.cv_seed,
    )

    ranked = rank_heads(source_cv_acc, candidate_layers)

    reliability_rows = []
    for rank, (acc, l, h) in enumerate(ranked, 1):
        reliability_rows.append(
            {
                "rank": rank,
                "layer": l,
                "head": h,
                "head_name": head_name(l, h),
                "source_oof_accuracy": acc,
                "weight_over_chance": max(acc - 0.25, 0.0),
            }
        )
    reliability_df = pd.DataFrame(reliability_rows)

    # -------------------------------------------------------------------------
    # 4) Refit codebook on ALL Synthetic-400 and freeze it.
    # -------------------------------------------------------------------------
    source_center, source_dirs = fit_codebook_all_heads(
        source_X,
        source_y,
    )

    np.savez_compressed(
        outdir / "synthetic_fitted_direction_codebook.npz",
        center=source_center,
        directions=source_dirs,
        relations=np.asarray(REL, dtype=object),
    )

    # -------------------------------------------------------------------------
    # 5) Frozen transfer to ALL target samples.
    # -------------------------------------------------------------------------
    target_scores = score_all_heads(
        target_X,
        source_center,
        source_dirs,
    )

    baseline = load_baseline_csv(args.target_baseline_csv)

    result_rows = []
    detail = pd.DataFrame(
        {
            "sid": target_sid,
            "gt": target_y,
        }
    )

    for K in topks:
        selected = [(l, h) for _, l, h in ranked[:K]]
        if len(selected) < K:
            raise RuntimeError(
                f"Requested Top{K}, but only {len(selected)} candidate heads available."
            )

        selected_names = ",".join(head_name(l, h) for l, h in selected)

        for weighted in (False, True):
            for layer_balanced in (False, True):
                probs = ensemble_probs(
                    scores=target_scores,
                    heads=selected,
                    reliability=source_cv_acc,
                    temperature=args.temperature,
                    weighted=weighted,
                    layer_balanced=layer_balanced,
                )

                pred_idx = np.argmax(probs, axis=-1)
                pred_rel = np.asarray(
                    [REL[int(i)] for i in pred_idx],
                    dtype=object,
                )

                method = (
                    f"top{K}_"
                    + ("weighted" if weighted else "equal")
                    + ("_layerbalanced" if layer_balanced else "")
                )

                metrics, correct = evaluate_prediction(pred_idx, target_y)

                row = {
                    "method": method,
                    **metrics,
                    "source_cv_mean_selected_head_acc": float(
                        np.mean([source_cv_acc[l, h] for l, h in selected])
                    ),
                    "heads": selected_names,
                }

                if baseline is not None:
                    has = np.asarray(
                        [int(sid) in baseline.index for sid in target_sid],
                        dtype=bool,
                    )
                    bc = np.zeros(len(target_sid), dtype=bool)

                    for i, sid in enumerate(target_sid):
                        sid = int(sid)
                        if sid not in baseline.index:
                            continue
                        v = baseline.loc[sid, "baseline_correct"]
                        if isinstance(v, pd.Series):
                            v = v.iloc[0]
                        bc[i] = bool(v)

                    c_mask = has & bc
                    w_mask = has & (~bc)

                    row["N_with_baseline"] = int(has.sum())
                    row["N_baseline_correct"] = int(c_mask.sum())
                    row["N_baseline_wrong"] = int(w_mask.sum())
                    row["acc_baseline_correct"] = (
                        float(correct[c_mask].mean())
                        if np.any(c_mask)
                        else np.nan
                    )
                    row["acc_baseline_wrong"] = (
                        float(correct[w_mask].mean())
                        if np.any(w_mask)
                        else np.nan
                    )

                    if "baseline_prediction" in baseline.columns:
                        bp = np.asarray(
                            [
                                canon_rel(
                                    baseline.loc[int(sid), "baseline_prediction"]
                                )
                                if int(sid) in baseline.index
                                else ""
                                for sid in target_sid
                            ],
                            dtype=object,
                        )
                        row["match_baseline_wrong"] = (
                            float(np.mean(pred_rel[w_mask] == bp[w_mask]))
                            if np.any(w_mask)
                            else np.nan
                        )

                result_rows.append(row)

                detail[f"pred_{method}"] = pred_rel
                detail[f"correct_{method}"] = correct
                detail[f"confidence_{method}"] = probs.max(axis=1)
                sorted_probs = np.sort(probs, axis=1)
                detail[f"margin_{method}"] = (
                    sorted_probs[:, -1] - sorted_probs[:, -2]
                )

    results = pd.DataFrame(result_rows).sort_values(
        ["acc_all", "source_cv_mean_selected_head_acc"],
        ascending=[False, False],
    )

    # -------------------------------------------------------------------------
    # Source head stability across source CV folds.
    # -------------------------------------------------------------------------
    stability_rows = []
    for K in sorted(set(topks + [3])):
        counts = Counter()

        for fold, g in fold_df.groupby("fold"):
            gg = g
            if candidate_layers:
                gg = gg[gg["layer"].isin(candidate_layers)]
            top = gg.sort_values(
                "fold_accuracy",
                ascending=False,
            ).head(K)
            counts.update(top["head_name"].tolist())

        for hn, n in counts.most_common():
            stability_rows.append(
                {
                    "K": K,
                    "head_name": hn,
                    "selected_in_source_folds": int(n),
                    "selection_frequency": float(n / args.cv_folds),
                }
            )

    stability_df = pd.DataFrame(stability_rows)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    reliability_df.to_csv(
        outdir / "source_oof_head_reliability.csv",
        index=False,
    )
    fold_df.to_csv(
        outdir / "source_fold_head_accuracy.csv",
        index=False,
    )
    stability_df.to_csv(
        outdir / "source_head_selection_stability.csv",
        index=False,
    )
    results.to_csv(
        outdir / "synthetic400_to_coco440_selector_summary.csv",
        index=False,
    )
    detail.to_csv(
        outdir / "synthetic400_to_coco440_predictions.csv",
        index=False,
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "source_labels": str(labels_path),
        "source_cache": str(source_cache),
        "source_prompt_mode": args.source_prompt,
        "source_prompt_template": (
            DEFAULT_PROBE_PROMPT
            if args.source_prompt == "probe"
            else ORIGINAL_SYNTHETIC_PROMPT
        ),
        "pool": args.pool,
        "source_N": int(len(source_y)),
        "source_counts": dict(Counter(source_y.tolist())),
        "target_path": str(target_path),
        "target_N": int(len(target_y)),
        "target_counts": dict(Counter(target_y.tolist())),
        "source_cv_folds": int(args.cv_folds),
        "source_cv_seed": int(args.cv_seed),
        "head_selection": "source-only stratified OOF accuracy",
        "direction_fit": "all synthetic source samples after source-only head ranking",
        "target_gt_usage": "evaluation only",
        "vector_definition": (
            "[(z_img_subject-z_img_reference) - "
            "(z_noimage_subject-z_noimage_reference)] pre-W_O per head"
        ),
        "topks": topks,
        "candidate_layers": candidate_layers,
    }

    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------
    print()
    print("=" * 160)
    print("SOURCE-ONLY TOP DIRECTION HEADS")
    print("=" * 160)
    print(
        reliability_df.head(20)[
            [
                "rank",
                "head_name",
                "source_oof_accuracy",
                "weight_over_chance",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 160)
    print("FROZEN SYNTHETIC-400 -> COCO TARGET RESULTS")
    print("=" * 160)

    show = [
        "method",
        "acc_all",
        "acc_left",
        "acc_right",
        "acc_above",
        "acc_below",
    ]
    if baseline is not None:
        show += [
            "N_with_baseline",
            "acc_baseline_correct",
            "acc_baseline_wrong",
        ]

    print(
        results[show].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 160)
    print("SOURCE TOP-3 HEAD STABILITY")
    print("=" * 160)

    s3 = stability_df[stability_df["K"] == 3].sort_values(
        ["selected_in_source_folds", "head_name"],
        ascending=[False, True],
    )
    print(s3.head(20).to_string(index=False))

    print()
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
