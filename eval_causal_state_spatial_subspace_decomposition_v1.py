#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_causal_state_spatial_subspace_decomposition_v1.py

Question
========
Why are writer-causal states behaviorally high-leverage?

This script tests two hypotheses without assuming that a causal state must be a
Direction-Head state.

For a causal state (L,p):

    delta_h = h_real[L,p] - h_gray[L,p]

and the existing oracle writer score is:

    M = delta_h^T g
    g = d J_writer / d h[L,p]

so:

    M = ||delta_h|| ||g|| cos(delta_h, g)

Part A: geometry decomposition
------------------------------
Read the existing L1..26 oracle ranking CSV and compare the selected causal
Top-K states against same-sample / same-layer / same-token-category lower-rank
positive controls:

    ||delta_h||
    ||g||
    cos(delta_h,g)
    M

This asks whether causal states are special because of:
    1) large image-induced displacement,
    2) large downstream decision sensitivity,
    3) unusually strong alignment between the two.

Part B: spatial-subspace causal decomposition
---------------------------------------------
Fit a residual-stream spatial subspace at EACH source layer using Synthetic-400
only, with Real-Gray object-role residuals:

    q_i,L =
        [(h_sub^real - h_ref^real)
         -
         (h_sub^gray - h_ref^gray)]_L

Default spatial basis is the 2D physical-axis basis:

    dx_L = mean(q|left)  - mean(q|right)
    dy_L = mean(q|above) - mean(q|below)

orthonormalized with SVD.

For every selected COCO causal state:

    delta_sp   = P_sp delta_h
    delta_perp = delta_h - delta_sp

Then run ACTUAL greedy generation under:

    baseline
    full:
        h <- h + alpha * delta_h

    spatial_raw:
        h <- h + alpha * delta_sp

    spatial_normmatched:
        h <- h + alpha * delta_sp * ||delta_h||/||delta_sp||

    orthogonal:
        h <- h + alpha * delta_perp

Interpretation
==============
Strong evidence that linearly decodable residual spatial content carries the
causal effect:

    spatial_raw ~= full
or at least
    spatial_normmatched ~= full
while orthogonal is weak.

Strong evidence AGAINST that account:

    spatial_raw << full
    spatial_normmatched << full
    orthogonal ~= full

The norm-matched spatial condition is important: it tests whether a weak raw
spatial effect is merely because the spatial projection has small norm.

This is still an ORACLE MECHANISM DIAGNOSTIC because the input causal ranking
was created using the GT-selected late writer.  Synthetic labels are used only
to define the spatial subspace.

Expected repository files
=========================
Run from AdaptVis/llava16 repository root.  Reuses only stable repository
utilities for model IO / token editing:

    analyze_coco_centroid_generation_step1_v4.py
    eval_coco_multilayer_relation_trajectory_repair_v1.py
    eval_qwen_dynamic_k24_all440_v1.py

It does NOT reuse any previously computed spatial vectors.

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_causal_state_spatial_subspace_decomposition_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --causal-top-k 7 \
  --causal-domain all \
  --source-layers 1-26 \
  --subspace-mode axis2 \
  --alpha 1.0 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_causal_spatial_decomp_top7_n80_v1 \
  --overwrite

Then full 440:
    --eval-max-samples 0

If you only want the text causal states:
    --causal-domain text

Outputs
=======
synthetic_spatial_basis.npz
synthetic_spatial_basis_summary.csv

selected_causal_states.csv
matched_control_states.csv
geometry_selected_vs_control.csv

causal_state_spatial_projection.csv
generation_per_sample.csv
generation_summary.csv
generation_by_category_presence.csv

analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument("--ranked-causal", required=True)
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument(
        "--causal-domain",
        choices=["all", "text", "visual"],
        default="all",
        help=(
            "Select first K eligible states from each sample's GLOBAL causal ranking. "
            "'text' excludes broad_category=visual; 'visual' keeps only visual."
        ),
    )
    p.add_argument(
        "--source-layers",
        default="1-26",
        help="Allowed causal source layers, e.g. 1-26 or 20,21,22,23,24,25,26.",
    )

    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default="")
    p.add_argument(
        "--subspace-mode",
        choices=["axis2", "classspan"],
        default="axis2",
        help=(
            "axis2: span(mu_left-mu_right, mu_above-mu_below). "
            "classspan: span of the four centered relation means (rank <=3)."
        ),
    )
    p.add_argument(
        "--svd-rel-tol",
        type=float,
        default=1e-5,
        help="Relative singular-value tolerance for spatial basis rank.",
    )
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument(
        "--conditions",
        default="full,spatial_raw,spatial_normmatched,orthogonal",
        help="Comma-separated edited conditions. Baseline is always run.",
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="0 = all available COCO samples.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# =============================================================================
# Generic
# =============================================================================

def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_conditions(text: str) -> List[str]:
    allowed = {
        "full",
        "spatial_raw",
        "spatial_normmatched",
        "orthogonal",
    }
    xs = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [x for x in xs if x not in allowed]
    if bad:
        raise ValueError(f"Bad --conditions: {bad}; allowed={sorted(allowed)}")
    return xs


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_median(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def cosine_from_row(row):
    m = float(row["mediation"])
    dn = float(row["delta_h_norm"])
    gn = float(row["grad_norm"])
    den = dn * gn
    if not np.isfinite(den) or den <= EPS:
        return float("nan")
    return float(np.clip(m / den, -1.0, 1.0))


def write_json(path: Path, x):
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def orthonormal_basis(vectors: np.ndarray, rel_tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    vectors: [K,D]
    returns basis [D,r], singular_values
    """
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(X.shape)
    if X.shape[0] == 0:
        return np.zeros((X.shape[1], 0), np.float32), np.zeros((0,), np.float32)

    # Row vectors. Right singular vectors are residual-space basis vectors.
    _u, s, vt = np.linalg.svd(X, full_matrices=False)
    if len(s) == 0 or float(s[0]) <= EPS:
        return np.zeros((X.shape[1], 0), np.float32), s.astype(np.float32)

    keep = s > float(rel_tol) * float(s[0])
    B = vt[keep].T.astype(np.float32)
    return B, s.astype(np.float32)


def project(B: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if B.size == 0 or B.shape[1] == 0:
        return np.zeros_like(v)
    return (B @ (B.T @ v)).astype(np.float32)


def normmatch(part: np.ndarray, full: np.ndarray) -> np.ndarray:
    pn = float(np.linalg.norm(part))
    fn = float(np.linalg.norm(full))
    if pn <= EPS or fn <= EPS:
        return np.zeros_like(full, dtype=np.float32)
    return (part * (fn / pn)).astype(np.float32)


# =============================================================================
# Synthetic data
# =============================================================================

def normalize_relation(x: str) -> str:
    s = str(x).strip().lower()
    mp = {
        "left": "left",
        "right": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
    }
    if s not in mp:
        raise ValueError(f"Unknown relation {x!r}")
    return mp[s]


def load_synthetic_records(root: Path, labels_path: Optional[Path]) -> List[Dict[str, Any]]:
    labels = labels_path if labels_path else root / "labels.jsonl"
    if not labels.exists():
        raise FileNotFoundError(labels)

    rows = []
    with labels.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            x = json.loads(line)
            rel = normalize_relation(x["relation"])
            subject = str(x["subject"]).strip()
            reference = str(x["reference"]).strip()
            ip = Path(str(x["image"]))
            if not ip.is_absolute():
                ip = root / ip
            if not ip.exists():
                raise FileNotFoundError(ip)

            # Match the existing synthetic steering prompt.
            question = (
                f"Where is the {subject} relative to the {reference}? "
                "Answer with left, right, above, or below."
            )
            rows.append({
                "sid": int(x.get("id", i)),
                "relation": rel,
                "subject": subject,
                "reference": reference,
                "image_path": ip,
                "question_text": question,
            })
    rows.sort(key=lambda z: int(z["sid"]))
    return rows


def phrase_positions(processor, ids, subject, reference):
    try:
        sspan, rspan = base.locate_object_spans(
            processor.tokenizer, ids, subject, reference
        )
        spos = sorted(dyn.span_positions(sspan))
        rpos = sorted(dyn.span_positions(rspan))
    except Exception:
        spos = dyn.find_text_positions(processor.tokenizer, ids, subject)
        rpos = dyn.find_text_positions(processor.tokenizer, ids, reference)

    if not spos or not rpos:
        raise RuntimeError(
            f"Could not locate subject/reference: subject={subject!r} reference={reference!r}"
        )
    return spos, rpos


def pool_positions(H: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    ps = [int(p) for p in positions if 0 <= int(p) < H.shape[0]]
    if not ps:
        raise RuntimeError("No valid phrase positions after hidden-length clipping.")
    return H[np.asarray(ps, dtype=int)].mean(axis=0).astype(np.float32)


@torch.inference_mode()
def fit_synthetic_spatial_bases(
    model,
    processor,
    decoder_layers,
    source_layers: Sequence[int],
    records: Sequence[Mapping[str, Any]],
    gray_value: int,
    subspace_mode: str,
    svd_rel_tol: float,
    device: torch.device,
    cache_path: Path,
    summary_path: Path,
):
    """
    Fit residual-stream Real-Gray object-role spatial basis at each source layer.
    """
    # Cache stores q: [N,L,D], labels [N].
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_layers = [int(x) for x in z["layers"].tolist()]
        if cached_layers != list(source_layers):
            raise RuntimeError(
                f"Cached spatial layers={cached_layers}, requested={list(source_layers)}. "
                "Delete cache or use a new output dir."
            )
        q = z["q"].astype(np.float32)
        labels = z["relation"].astype(str)
        sids = z["sid"].astype(int)
        print(f"[synthetic] reuse {cache_path} q={q.shape}")
    else:
        qs, labels, sids = [], [], []
        errors = []

        for rec in tqdm(records, desc="Synthetic residual spatial subspace"):
            real = gray = rb = gb = None
            try:
                real = Image.open(rec["image_path"]).convert("RGB")
                gray = dyn.make_gray_image(real, gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=rec["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=rec["question_text"],
                    device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                spos, rpos = phrase_positions(
                    processor,
                    ids,
                    rec["subject"],
                    rec["reference"],
                )

                hr = dyn.capture_cpu(model, decoder_layers, rb, source_layers)
                hg = dyn.capture_cpu(model, decoder_layers, gb, source_layers)

                arr = []
                for L in source_layers:
                    R = hr[L][0].astype(np.float32)
                    G = hg[L][0].astype(np.float32)
                    n = min(R.shape[0], G.shape[0])
                    D = (R[:n] - G[:n]).astype(np.float32)
                    qL = (
                        pool_positions(D, spos)
                        - pool_positions(D, rpos)
                    ).astype(np.float32)
                    arr.append(qL)

                qs.append(np.stack(arr, axis=0))
                labels.append(rec["relation"])
                sids.append(int(rec["sid"]))

            except Exception as exc:
                errors.append({
                    "sid": int(rec["sid"]),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                tqdm.write(
                    f"[synthetic ERROR] sid={rec['sid']} {type(exc).__name__}: {exc}"
                )
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not qs:
            raise RuntimeError("No valid synthetic spatial examples.")

        q = np.stack(qs, axis=0).astype(np.float32)
        labels = np.asarray(labels, dtype=object)
        sids = np.asarray(sids, dtype=np.int64)

        np.savez_compressed(
            cache_path,
            sid=sids,
            relation=labels,
            layers=np.asarray(source_layers, dtype=np.int64),
            q=q,
        )
        if errors:
            write_json(cache_path.with_suffix(".errors.json"), errors)

    # Fit means and bases.
    bases = {}
    rows = []
    basis_arrays = {}

    for li, L in enumerate(source_layers):
        means = {}
        for r in REL:
            idx = np.where(labels == r)[0]
            if len(idx) == 0:
                raise RuntimeError(f"No synthetic relation={r}")
            means[r] = q[idx, li].mean(axis=0).astype(np.float32)

        global_mu = np.stack([means[r] for r in REL], axis=0).mean(axis=0)

        if subspace_mode == "axis2":
            dx = (means["left"] - means["right"]).astype(np.float32)
            dy = (means["above"] - means["below"]).astype(np.float32)
            raw = np.stack([dx, dy], axis=0)
        else:
            raw = np.stack(
                [(means[r] - global_mu).astype(np.float32) for r in REL],
                axis=0,
            )

        B, svals = orthonormal_basis(raw, svd_rel_tol)
        bases[int(L)] = B
        basis_arrays[f"basis_L{int(L)}"] = B

        rows.append({
            "source_layer": int(L),
            "mode": subspace_mode,
            "basis_rank": int(B.shape[1]),
            "s1": float(svals[0]) if len(svals) > 0 else np.nan,
            "s2": float(svals[1]) if len(svals) > 1 else np.nan,
            "s3": float(svals[2]) if len(svals) > 2 else np.nan,
            "dx_norm": float(np.linalg.norm(means["left"] - means["right"])),
            "dy_norm": float(np.linalg.norm(means["above"] - means["below"])),
            "n_left": int(np.sum(labels == "left")),
            "n_right": int(np.sum(labels == "right")),
            "n_above": int(np.sum(labels == "above")),
            "n_below": int(np.sum(labels == "below")),
        })

    np.savez_compressed(
        cache_path.parent / "synthetic_spatial_basis.npz",
        layers=np.asarray(source_layers, dtype=np.int64),
        mode=np.asarray(subspace_mode),
        **basis_arrays,
    )
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return bases, pd.DataFrame(rows)


# =============================================================================
# Causal ranking selection / matched controls
# =============================================================================

def load_ranking(path: Path, allowed_layers: Sequence[int]) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = {
        "sid", "rank", "source_layer", "position", "mediation",
        "delta_h_norm", "grad_norm", "broad_category", "category",
    }
    missing = need - set(d.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    for c in ("sid", "rank", "source_layer", "position"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)
    for c in ("mediation", "delta_h_norm", "grad_norm"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[d["source_layer"].isin(set(map(int, allowed_layers)))].copy()
    d["alignment_cos"] = [
        cosine_from_row(r)
        for _, r in d.iterrows()
    ]
    return d


def eligible_domain(row, domain: str) -> bool:
    broad = str(row["broad_category"])
    if domain == "all":
        return True
    if domain == "text":
        return broad != "visual"
    if domain == "visual":
        return broad == "visual"
    raise ValueError(domain)


def select_topk_per_sample(
    ranking: pd.DataFrame,
    top_k: int,
    domain: str,
    allowed_sids: Optional[set] = None,
) -> pd.DataFrame:
    rows = []
    for sid, g in ranking.groupby("sid"):
        sid = int(sid)
        if allowed_sids is not None and sid not in allowed_sids:
            continue
        g = g.sort_values("rank")
        g = g[g.apply(lambda r: eligible_domain(r, domain), axis=1)]
        g = g.head(int(top_k)).copy()
        g["selected_domain_rank"] = np.arange(1, len(g) + 1)
        rows.append(g)
    if not rows:
        return ranking.iloc[:0].copy()
    return pd.concat(rows, ignore_index=True)


def match_controls(
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """
    One lower-ranked positive control for each selected state when possible.
    Match: same sid, same source layer, same broad category.
    No selected position may be reused as control.
    """
    rng = random.Random(seed)
    selected_keys = {
        (int(r.sid), int(r.source_layer), int(r.position))
        for r in selected.itertuples()
    }
    rank_by_sid = {
        int(sid): g.sort_values("rank").copy()
        for sid, g in ranking.groupby("sid")
    }

    rows = []
    used = set()

    for s in selected.sort_values(["sid", "selected_domain_rank"]).itertuples():
        sid = int(s.sid)
        L = int(s.source_layer)
        broad = str(s.broad_category)

        g = rank_by_sid[sid]
        pool = g[
            (g["source_layer"] == L)
            & (g["broad_category"].astype(str) == broad)
            & (g["rank"] > int(s.rank))
        ]

        candidates = []
        for r in pool.itertuples():
            key = (sid, int(r.source_layer), int(r.position))
            if key in selected_keys or key in used:
                continue
            candidates.append(r)

        if not candidates:
            continue

        # Prefer controls not absurdly far away, but randomize within lower-ranked pool.
        # Choose among the first 25 candidates after rank sorting.
        candidates = candidates[:25]
        c = rng.choice(candidates)
        key = (sid, int(c.source_layer), int(c.position))
        used.add(key)

        row = {k: getattr(c, k) for k in ranking.columns if hasattr(c, k)}
        row.update({
            "matched_to_selected_rank": int(s.rank),
            "matched_to_selected_domain_rank": int(s.selected_domain_rank),
            "matched_to_position": int(s.position),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def geometry_summary(selected: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, df in (("causal_topk", selected), ("matched_control", controls)):
        if len(df) == 0:
            continue

        scopes = [("all", df)]
        for broad, g in df.groupby("broad_category"):
            scopes.append((f"category:{broad}", g))

        for scope, g in scopes:
            parts.append({
                "group": name,
                "scope": scope,
                "N": len(g),
                "mean_M": safe_mean(g["mediation"]),
                "median_M": safe_median(g["mediation"]),
                "mean_delta_h_norm": safe_mean(g["delta_h_norm"]),
                "median_delta_h_norm": safe_median(g["delta_h_norm"]),
                "mean_grad_norm": safe_mean(g["grad_norm"]),
                "median_grad_norm": safe_median(g["grad_norm"]),
                "mean_alignment_cos": safe_mean(g["alignment_cos"]),
                "median_alignment_cos": safe_median(g["alignment_cos"]),
            })
    return pd.DataFrame(parts)


# =============================================================================
# COCO metadata / model
# =============================================================================

def load_coco_meta(args):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(args.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(args.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, args.max_samples, args.seed)
    return two, meta, rec_by_sid


def load_model(args, two):
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

    print(f"Loading {spec.repo_id}", flush=True)
    try:
        model = cls.from_pretrained(spec.repo_id, **kw)
    except TypeError:
        # compatibility with older transformers
        kw["torch_dtype"] = kw.pop("dtype")
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
    return model, processor, decoder_layers, decoder_path, spec


# =============================================================================
# Generation intervention
# =============================================================================

@torch.inference_mode()
def run_generation(
    model,
    processor,
    decoder_layers,
    batch,
    max_new_tokens: int,
    token_specs: Optional[Dict[int, List[Tuple[int, np.ndarray]]]] = None,
    alpha: float = 1.0,
):
    editor = None
    try:
        if token_specs:
            prompt_len = int(batch["input_ids"].shape[1])
            editor = dyn.TokenDeltaEditor(
                decoder_layers,
                token_specs,
                float(alpha),
                prompt_len,
            )

        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )
        pred = traj.normalize_relation(base, text)
        return pred, text, (
            dict(editor.applied) if editor is not None else {}
        )
    finally:
        if editor is not None:
            editor.close()


def make_specs(
    rows_for_sid: pd.DataFrame,
    hr: Mapping[int, np.ndarray],
    hg: Mapping[int, np.ndarray],
    bases: Mapping[int, np.ndarray],
):
    """
    Return specs by condition and per-state projection diagnostics.
    """
    specs = {
        "full": defaultdict(list),
        "spatial_raw": defaultdict(list),
        "spatial_normmatched": defaultdict(list),
        "orthogonal": defaultdict(list),
    }
    diag = []

    for r in rows_for_sid.sort_values("selected_domain_rank").itertuples():
        L = int(r.source_layer)
        p = int(r.position)

        R = hr[L][0].astype(np.float32)
        G = hg[L][0].astype(np.float32)
        n = min(R.shape[0], G.shape[0])
        if not (0 <= p < n):
            raise RuntimeError(f"state L{L} p{p} outside hidden length {n}")

        delta = (R[p] - G[p]).astype(np.float32)
        B = bases[L]
        dsp = project(B, delta)
        dperp = (delta - dsp).astype(np.float32)
        dsnm = normmatch(dsp, delta)

        fn = float(np.linalg.norm(delta))
        sn = float(np.linalg.norm(dsp))
        pn = float(np.linalg.norm(dperp))

        specs["full"][L].append((p, delta))
        specs["spatial_raw"][L].append((p, dsp))
        specs["spatial_normmatched"][L].append((p, dsnm))
        specs["orthogonal"][L].append((p, dperp))

        diag.append({
            "sid": int(r.sid),
            "gt": str(getattr(r, "gt", "")),
            "selected_domain_rank": int(r.selected_domain_rank),
            "global_causal_rank": int(r.rank),
            "source_layer": L,
            "position": p,
            "token": str(r.token) if hasattr(r, "token") else "",
            "category": str(r.category),
            "broad_category": str(r.broad_category),
            "mediation": float(r.mediation),
            "delta_h_norm_csv": float(r.delta_h_norm),
            "grad_norm_csv": float(r.grad_norm),
            "alignment_cos_csv": float(r.alignment_cos),
            "delta_norm_recomputed": fn,
            "spatial_norm": sn,
            "orthogonal_norm": pn,
            "spatial_norm_fraction": safe_div(sn, fn),
            "spatial_energy_fraction": safe_div(sn * sn, fn * fn),
            "orthogonal_norm_fraction": safe_div(pn, fn),
            "basis_rank": int(B.shape[1]),
        })

    return {k: dict(v) for k, v in specs.items()}, diag


# =============================================================================
# Summaries
# =============================================================================

def generation_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "baseline"].set_index("sid")
    if len(base) == 0:
        return pd.DataFrame()

    rows = []
    base_acc = float(base["correct"].mean())

    for cond, g in df.groupby("condition"):
        gg = g.set_index("sid")
        common = sorted(set(base.index) & set(gg.index))
        b = base.loc[common]
        c = gg.loc[common]

        bcorrect = b["correct"].astype(bool).to_numpy()
        ccorrect = c["correct"].astype(bool).to_numpy()

        w2c = int(np.sum((~bcorrect) & ccorrect))
        c2w = int(np.sum(bcorrect & (~ccorrect)))

        rows.append({
            "condition": cond,
            "N": len(common),
            "accuracy": float(ccorrect.mean()) if len(common) else np.nan,
            "gain_vs_baseline": (
                float(ccorrect.mean()) - base_acc if len(common) else np.nan
            ),
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "changed_correctness": int(np.sum(bcorrect != ccorrect)),
        })

    order = {
        "baseline": 0,
        "full": 1,
        "spatial_raw": 2,
        "spatial_normmatched": 3,
        "orthogonal": 4,
    }
    out = pd.DataFrame(rows)
    out["_o"] = out["condition"].map(order).fillna(999)
    return out.sort_values("_o").drop(columns="_o")


def category_presence_summary(
    gen_df: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if len(gen_df) == 0 or len(selected) == 0:
        return pd.DataFrame()

    rows = []
    sid_categories = {
        int(sid): set(g["broad_category"].astype(str))
        for sid, g in selected.groupby("sid")
    }

    for cat in sorted(set(selected["broad_category"].astype(str))):
        sids = {sid for sid, cats in sid_categories.items() if cat in cats}
        z = gen_df[gen_df["sid"].isin(sids)]
        if len(z) == 0:
            continue
        sm = generation_summary(z)
        for _, r in sm.iterrows():
            row = r.to_dict()
            row["causal_category_present"] = cat
            row["sample_N_with_category"] = len(sids)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_layers(a.source_layers)
    conditions = parse_conditions(a.conditions)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    errors_path = outdir / "errors.jsonl"

    # Load ranking first.
    ranking = load_ranking(Path(a.ranked_causal), source_layers)

    # COCO metadata.
    two, meta, rec_by_sid = load_coco_meta(a)
    meta_by_sid = {int(x["sid"]): x for x in meta}

    valid_sids = sorted(set(ranking["sid"].astype(int)) & set(meta_by_sid))
    if a.eval_max_samples > 0:
        # Relation-stratified cap, but only over ranking-available samples.
        tmp = [meta_by_sid[sid] for sid in valid_sids]
        tmp = traj.stratified_cap(tmp, a.eval_max_samples, a.seed + 1)
        valid_sids = sorted(int(x["sid"]) for x in tmp)

    selected = select_topk_per_sample(
        ranking,
        a.causal_top_k,
        a.causal_domain,
        allowed_sids=set(valid_sids),
    )

    # Add GT for convenience.
    selected["gt"] = selected["sid"].map(
        lambda sid: meta_by_sid[int(sid)]["gt"]
    )

    controls = match_controls(
        ranking[ranking["sid"].isin(valid_sids)].copy(),
        selected,
        a.seed,
    )
    if len(controls):
        controls["gt"] = controls["sid"].map(
            lambda sid: meta_by_sid[int(sid)]["gt"]
        )

    selected.to_csv(outdir / "selected_causal_states.csv", index=False)
    controls.to_csv(outdir / "matched_control_states.csv", index=False)

    geom = geometry_summary(selected, controls)
    geom.to_csv(outdir / "geometry_selected_vs_control.csv", index=False)

    # Model.
    model = processor = None
    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(a, two)
        n_layers = len(decoder_layers)
        for L in source_layers:
            if not (0 <= L < n_layers):
                raise ValueError(
                    f"source L{L} invalid; model has L0..L{n_layers-1}"
                )

        device = torch.device(a.device)

        # Synthetic source.
        syn_records = load_synthetic_records(
            Path(a.synthetic_dir),
            Path(a.synthetic_labels) if a.synthetic_labels else None,
        )

        print("=" * 170)
        print("CAUSAL STATE: SPATIAL SUBSPACE vs ORTHOGONAL CAUSAL EFFECT")
        print("=" * 170)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(f"source layers={source_layers}")
        print(
            f"causal ranking={a.ranked_causal} | domain={a.causal_domain} | "
            f"TopK={a.causal_top_k}"
        )
        print(
            f"eval N={len(valid_sids)} | selected states={len(selected)} | "
            f"matched controls={len(controls)}"
        )
        print(
            f"Synthetic N={len(syn_records)} | subspace={a.subspace_mode} | "
            f"gray={a.gray_value}"
        )
        print(f"conditions={conditions} | alpha={a.alpha}")
        print()

        bases, basis_summary = fit_synthetic_spatial_bases(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            source_layers=source_layers,
            records=syn_records,
            gray_value=a.gray_value,
            subspace_mode=a.subspace_mode,
            svd_rel_tol=a.svd_rel_tol,
            device=device,
            cache_path=outdir / "synthetic_residual_spatial_cache.npz",
            summary_path=outdir / "synthetic_spatial_basis_summary.csv",
        )

        # Evaluate.
        gen_rows = []
        proj_rows = []

        selected_by_sid = {
            int(sid): g.copy()
            for sid, g in selected.groupby("sid")
        }

        for sid in tqdm(valid_sids, desc="COCO causal component generation"):
            real = gray = rb = gb = None
            try:
                m = meta_by_sid[sid]
                if sid not in selected_by_sid or len(selected_by_sid[sid]) == 0:
                    continue

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                rows_sid = selected_by_sid[sid]
                layers_sid = sorted(
                    set(int(x) for x in rows_sid["source_layer"].tolist())
                )

                hr = dyn.capture_cpu(model, decoder_layers, rb, layers_sid)
                hg = dyn.capture_cpu(model, decoder_layers, gb, layers_sid)

                specs, diags = make_specs(
                    rows_sid,
                    hr,
                    hg,
                    bases,
                )
                proj_rows.extend(diags)

                # Baseline.
                pred, text, applied = run_generation(
                    model,
                    processor,
                    decoder_layers,
                    rb,
                    a.max_new_tokens,
                    token_specs=None,
                    alpha=a.alpha,
                )
                base_correct = pred == m["gt"]
                gen_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "baseline",
                    "prediction": pred,
                    "correct": base_correct,
                    "text": text,
                    "selected_state_N": len(rows_sid),
                    "edit_applied_N": 0,
                })

                # Component interventions.
                for cond in conditions:
                    pred, text, applied = run_generation(
                        model,
                        processor,
                        decoder_layers,
                        rb,
                        a.max_new_tokens,
                        token_specs=specs[cond],
                        alpha=a.alpha,
                    )
                    gen_rows.append({
                        "sid": sid,
                        "gt": m["gt"],
                        "condition": cond,
                        "prediction": pred,
                        "correct": pred == m["gt"],
                        "text": text,
                        "selected_state_N": len(rows_sid),
                        "edit_applied_N": int(sum(applied.values()))
                        if applied else 0,
                    })

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid} {type(exc).__name__}: {exc}"
                )
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gen_df = pd.DataFrame(gen_rows)
        proj_df = pd.DataFrame(proj_rows)

        gen_df.to_csv(outdir / "generation_per_sample.csv", index=False)
        proj_df.to_csv(
            outdir / "causal_state_spatial_projection.csv",
            index=False,
        )

        gen_sum = generation_summary(gen_df)
        gen_sum.to_csv(outdir / "generation_summary.csv", index=False)

        cat_sum = category_presence_summary(gen_df, selected)
        cat_sum.to_csv(
            outdir / "generation_by_category_presence.csv",
            index=False,
        )

        # Additional projection summary.
        proj_summary_rows = []
        if len(proj_df):
            scopes = [("all", proj_df)]
            for cat, g in proj_df.groupby("broad_category"):
                scopes.append((f"category:{cat}", g))
            for scope, g in scopes:
                proj_summary_rows.append({
                    "scope": scope,
                    "N": len(g),
                    "mean_spatial_norm_fraction": safe_mean(
                        g["spatial_norm_fraction"]
                    ),
                    "median_spatial_norm_fraction": safe_median(
                        g["spatial_norm_fraction"]
                    ),
                    "mean_spatial_energy_fraction": safe_mean(
                        g["spatial_energy_fraction"]
                    ),
                    "median_spatial_energy_fraction": safe_median(
                        g["spatial_energy_fraction"]
                    ),
                    "mean_orthogonal_norm_fraction": safe_mean(
                        g["orthogonal_norm_fraction"]
                    ),
                })
        proj_summary = pd.DataFrame(proj_summary_rows)
        proj_summary.to_csv(
            outdir / "spatial_projection_summary.csv",
            index=False,
        )

        # Console/report.
        report = []
        report.append("=" * 170)
        report.append("CAUSAL STATE SPATIAL-SUBSPACE DECOMPOSITION")
        report.append("=" * 170)
        report.append(
            f"N target samples={len(valid_sids)} | completed="
            f"{gen_df[gen_df.condition == 'baseline'].sid.nunique() if len(gen_df) else 0}"
        )
        report.append(
            f"TopK={a.causal_top_k} domain={a.causal_domain} "
            f"subspace={a.subspace_mode} alpha={a.alpha}"
        )
        report.append("")

        report.append("A) M = ||delta|| ||grad|| cos(delta,grad)")
        report.append("-" * 170)
        if len(geom):
            report.append(
                geom.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append("B) HOW MUCH OF EACH CAUSAL DELTA LIES IN SYNTHETIC SPATIAL SUBSPACE?")
        report.append("-" * 170)
        if len(proj_summary):
            report.append(
                proj_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append("C) ACTUAL GENERATION")
        report.append("-" * 170)
        if len(gen_sum):
            report.append(
                gen_sum.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        report.append("")

        # Automatic compact diagnostic, not a claim.
        if len(gen_sum):
            dct = {
                str(r.condition): r
                for r in gen_sum.itertuples()
            }
            if all(k in dct for k in ("baseline", "full")):
                full_gain = float(dct["full"].gain_vs_baseline)
                report.append(f"full gain = {full_gain:+.4f}")

            if all(k in dct for k in ("full", "spatial_raw", "orthogonal")):
                f = float(dct["full"].gain_vs_baseline)
                s = float(dct["spatial_raw"].gain_vs_baseline)
                o = float(dct["orthogonal"].gain_vs_baseline)
                report.append(
                    "gain retention vs full: "
                    f"spatial_raw={safe_div(s, f):+.3f} | "
                    f"orthogonal={safe_div(o, f):+.3f}"
                )

            if all(k in dct for k in ("full", "spatial_normmatched")):
                f = float(dct["full"].gain_vs_baseline)
                sn = float(dct["spatial_normmatched"].gain_vs_baseline)
                report.append(
                    "normmatched-spatial gain retention vs full: "
                    f"{safe_div(sn, f):+.3f}"
                )

        report_text = "\n".join(report) + "\n"
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )
        print(report_text)

        metadata = {
            "script": "eval_causal_state_spatial_subspace_decomposition_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "ranked_causal": str(a.ranked_causal),
            "source_layers": source_layers,
            "causal_top_k": a.causal_top_k,
            "causal_domain": a.causal_domain,
            "synthetic_dir": str(a.synthetic_dir),
            "synthetic_N": len(syn_records),
            "spatial_subspace_definition": (
                "Residual-stream Real-Gray object-role q = "
                "(h_sub_real-h_ref_real)-(h_sub_gray-h_ref_gray)"
            ),
            "subspace_mode": a.subspace_mode,
            "subspace_axis2": (
                "span(mean(q|left)-mean(q|right), "
                "mean(q|above)-mean(q|below))"
            ),
            "alpha": a.alpha,
            "conditions": conditions,
            "oracle_warning": (
                "Input causal ranking uses GT-selected late writer; this is an "
                "oracle mechanism diagnostic."
            ),
            "matched_control_definition": (
                "Lower-ranked positive global-unique state, same sample, same source "
                "layer and same broad token category."
            ),
        }
        write_json(outdir / "metadata.json", metadata)

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
