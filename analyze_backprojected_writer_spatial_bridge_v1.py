#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_backprojected_writer_spatial_bridge_v1.py

Question
========
The previous causal-update decomposition showed that removing the original
object-token spatial subspace from K36 causal updates leaves essentially all of
the repair effect.  That rules out the simplest hypothesis that the same
object-token spatial direction is literally carried forward unchanged.

This script tests the stronger / more appropriate bridge hypothesis:

    object-token spatial code  --Transformer-->  late decision/writer code

Instead of comparing vectors that live at different layers/tokens directly, we
BACK-PROJECT each late writer objective into the subject/reference token space.

For source layer L and relation r, define a late writer objective

    J_r = mean_T < h[T,last], unit(v[T,r]) >,   T in target layers.

By default we remove the four-way common mode:

    Jc_r = J_r - mean_s J_s.

Then differentiate w.r.t. the source-layer object-token states:

    g_sub[L,r] = d Jc_r / d h[L,subject]
    g_ref[L,r] = d Jc_r / d h[L,reference]

Because the spatial representation is

    relation_state = h(subject) - h(reference),

its corresponding back-projected decision direction (up to an irrelevant
factor of 1/2) is

    b[L,r] = g_sub[L,r] - g_ref[L,r].

We compare b[L,r] with the frozen REAL-NoImage object-token spatial directions
from --spatial-states-npz.

Primary diagnostics
===================
1) Four-by-four relation matrix

    M_L[i,j] = cos( spatial_dir[L,i], backproj_writer[L,j] )

   If the identity semantic mapping is real, diagonal / opposite-pair structure
   should beat shuffled relation mappings.

2) Axis bridge (main, more robust)

    d_H = unit(d_right - d_left)
    d_V = unit(d_above - d_below)

    b_H = unit(b_right - b_left)
    b_V = unit(b_above - b_below)

   We want matched alignment to dominate crossed alignment:

    cos(d_H,b_H) > cos(d_H,b_V)
    cos(d_V,b_V) > cos(d_V,b_H)

3) Pair-contrast score

    horizontal = M[L,left,left] + M[L,right,right]
                 - M[L,left,right] - M[L,right,left]

    vertical   = analogous above/below.

This is a MECHANISM diagnostic.  It does not use the causal-token WHERE scaffold
and it does not perform an intervention.  GT relation labels are used only to
fit/evaluate the global spatial codebook and to report stratified summaries;
the back-projected writer directions themselves are relation-conditioned writer
objectives, not sample-GT-selected objectives.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u analyze_backprojected_writer_spatial_bridge_v1.py \
  --model qwen-3b \
  --spatial-states-npz output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz \
  --writer-npz output/qwen3b_coco_dynamic_k24_all440/learned_writers.npz \
  --source-layers 20-26 \
  --target-layers 32,34,35 \
  --spatial-fit-ratio 0.30 \
  --eval-scope heldout \
  --eval-max-samples 80 \
  --objective-mode centered \
  --output-dir output/qwen3b_backproject_writer_spatial_bridge_n80_v1 \
  --overwrite

If that writer file is not the one used by your causal scan, pass the exact
learned_writers.npz from that run.  When --writer-npz is omitted, this script
tries to auto-resolve a compatible file and prints what it chose.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import eval_real_causal_token_update_gating_v1 as old
except Exception as exc:
    raise SystemExit(
        "Could not import eval_real_causal_token_update_gating_v1.py.\n"
        "Run this script from the AdaptVis repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# =============================================================================
# CLI / generic helpers
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
    p.add_argument("--spatial-states-npz", required=True)
    p.add_argument(
        "--writer-npz",
        default="",
        help=(
            "learned_writers.npz used by the late-writer causal experiments. "
            "If omitted, try to auto-resolve a compatible file under output/."
        ),
    )
    p.add_argument("--source-layers", default="20-26")
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument(
        "--objective-mode",
        default="centered",
        choices=["raw", "centered"],
        help="centered subtracts the four-way common writer objective before backprop.",
    )
    p.add_argument(
        "--target-combine",
        default="mean",
        choices=["mean", "sum"],
        help="Combine the selected late writer layers into one relation objective.",
    )

    p.add_argument(
        "--spatial-fit-ratio",
        type=float,
        default=0.30,
        help="Per-relation fraction of residual-state NPZ used to fit spatial directions.",
    )
    p.add_argument("--spatial-fit-seed", type=int, default=1)
    p.add_argument(
        "--eval-scope",
        default="heldout",
        choices=["heldout", "all_data"],
        help="heldout excludes the samples used to fit the object spatial codebook.",
    )
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all in eval scope")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if not (0.0 < a.spatial_fit_ratio < 1.0):
        p.error("--spatial-fit-ratio must be in (0,1)")
    return a


def parse_layers(s: str) -> List[int]:
    out = set()
    for part in str(s).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"No layers parsed from {s!r}")
    return sorted(out)


def norm_rel(x) -> str:
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s)


def normalize_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def stratified_cap(rows: Sequence[dict], n: int, seed: int) -> List[dict]:
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    # Reuse repository helper when available; it stratifies by gt.
    try:
        return list(old.traj.stratified_cap(rows, int(n), int(seed)))
    except Exception:
        rng = random.Random(seed)
        by = defaultdict(list)
        for row in rows:
            by[row["gt"]].append(row)
        for v in by.values():
            rng.shuffle(v)
        picked = []
        while len(picked) < n:
            moved = False
            for r in REL:
                if by[r] and len(picked) < n:
                    picked.append(by[r].pop())
                    moved = True
            if not moved:
                break
        return sorted(picked, key=lambda x: int(x["sid"]))


# =============================================================================
# Spatial codebook
# =============================================================================

def load_spatial_npz(path: Path):
    with np.load(path, allow_pickle=True) as z:
        need = {"relation_vectors", "relation", "decoder_block_index"}
        miss = need - set(z.files)
        if miss:
            raise RuntimeError(f"{path} missing keys: {sorted(miss)}; keys={z.files}")
        X = np.asarray(z["relation_vectors"], dtype=np.float32)
        y = np.asarray([norm_rel(x) for x in z["relation"].tolist()], dtype=object)
        layers = [int(x) for x in z["decoder_block_index"].tolist()]
        if "sample_index" in z.files:
            sids = np.asarray(z["sample_index"], dtype=np.int64)
        else:
            sids = np.arange(len(y), dtype=np.int64)
    if X.ndim != 3 or X.shape[0] != len(y) or X.shape[1] != len(layers):
        raise RuntimeError(
            f"Bad spatial NPZ shapes: X={X.shape}, y={y.shape}, layers={len(layers)}"
        )
    return X, y, layers, sids


def make_spatial_split(y: np.ndarray, sids: np.ndarray, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    fit_idx = []
    held_idx = []
    for r in REL:
        idx = np.where(y == r)[0]
        if len(idx) == 0:
            raise RuntimeError(f"Spatial NPZ has no relation {r}")
        idx = idx.copy()
        rng.shuffle(idx)
        nfit = max(1, int(math.floor(len(idx) * ratio)))
        nfit = min(nfit, len(idx) - 1) if len(idx) > 1 else 1
        fit_idx.extend(idx[:nfit].tolist())
        held_idx.extend(idx[nfit:].tolist())
    fit_idx = np.asarray(sorted(fit_idx), dtype=np.int64)
    held_idx = np.asarray(sorted(held_idx), dtype=np.int64)
    return fit_idx, held_idx, set(map(int, sids[fit_idx])), set(map(int, sids[held_idx]))


def fit_spatial_directions(
    X: np.ndarray,
    y: np.ndarray,
    layers: Sequence[int],
    fit_idx: np.ndarray,
    requested_layers: Sequence[int],
):
    layer_to_col = {int(L): i for i, L in enumerate(layers)}
    missing = [L for L in requested_layers if L not in layer_to_col]
    if missing:
        raise RuntimeError(f"Spatial NPZ does not contain decoder layers {missing}; has {layers}")

    dirs: Dict[int, Dict[str, np.ndarray]] = {}
    axes: Dict[int, Dict[str, np.ndarray]] = {}
    rows = []

    for L in requested_layers:
        li = layer_to_col[L]
        Z = X[fit_idx, li, :].astype(np.float64)
        yy = y[fit_idx]
        center = Z.mean(axis=0)
        dirs[L] = {}
        for r in REL:
            mask = yy == r
            if not np.any(mask):
                raise RuntimeError(f"No fit samples for relation={r} at L{L}")
            v = Z[mask].mean(axis=0) - center
            dirs[L][r] = normalize_np(v)
            rows.append({
                "source_layer": L,
                "relation": r,
                "fit_N": int(mask.sum()),
                "centroid_offset_norm": float(np.linalg.norm(v)),
            })

        # These are semantic contrast axes; use the relation codebook directions,
        # not a separately fitted classifier.
        h = normalize_np(dirs[L]["right"] - dirs[L]["left"])
        v = normalize_np(dirs[L]["above"] - dirs[L]["below"])
        axes[L] = {"horizontal": h, "vertical": v}

    return dirs, axes, pd.DataFrame(rows)


# =============================================================================
# Writers / objectives
# =============================================================================

def writer_key_candidates(T: int, r: str) -> List[str]:
    disp = DISPLAY[r]
    return [
        f"L{T}_{r}",
        f"L{T}_{disp}",
        f"{T}_{r}",
        f"{T}_{disp}",
    ]


def npz_supports_writers(path: Path, targets: Sequence[int]) -> bool:
    try:
        with np.load(path, allow_pickle=True) as z:
            keys = set(z.files)
            for T in targets:
                for r in REL:
                    if not any(k in keys for k in writer_key_candidates(T, r)):
                        return False
        return True
    except Exception:
        return False


def recursive_find_writer_path_in_json(obj) -> List[str]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(recursive_find_writer_path_in_json(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(recursive_find_writer_path_in_json(v))
    elif isinstance(obj, str) and obj.endswith("learned_writers.npz"):
        out.append(obj)
    return out


def resolve_writer_npz(a, targets: Sequence[int]) -> Path:
    if a.writer_npz:
        p = Path(a.writer_npz)
        if not p.exists():
            raise FileNotFoundError(p)
        if not npz_supports_writers(p, targets):
            raise RuntimeError(f"{p} does not contain writers for targets={targets}")
        return p

    candidates: List[Path] = []

    # Common / historically used paths first.
    preferred = [
        Path("output/qwen3b_coco_dynamic_L20_26_K36_all440/learned_writers.npz"),
        Path("output/qwen3b_coco_dynamic_k24_all440/learned_writers.npz"),
        Path("output/qwen3b_oracle_k7_k15_all440/learned_writers.npz"),
    ]
    candidates.extend([p for p in preferred if p.exists()])

    # Try metadata near known causal scan if present in the standard location.
    for meta_path in [
        Path("output/qwen3b_oracle_causal_scan_L1_L26_all440/metadata.json"),
        Path("output/qwen3b_oracle_causal_scan_L1_L26_all440/config.json"),
    ]:
        if meta_path.exists():
            try:
                obj = json.loads(meta_path.read_text(encoding="utf-8"))
                for s in recursive_find_writer_path_in_json(obj):
                    p = Path(s)
                    if p.exists():
                        candidates.append(p)
            except Exception:
                pass

    # Last resort: scan output tree.
    outroot = Path("output")
    if outroot.exists():
        candidates.extend(sorted(outroot.glob("**/learned_writers.npz")))

    uniq = []
    seen = set()
    for p in candidates:
        rp = str(p)
        if rp not in seen and npz_supports_writers(p, targets):
            uniq.append(p)
            seen.add(rp)

    if not uniq:
        raise FileNotFoundError(
            "Could not auto-resolve a compatible learned_writers.npz. "
            "Pass --writer-npz explicitly."
        )

    chosen = uniq[0]
    print("[writer] auto-selected:", chosen, flush=True)
    if len(uniq) > 1:
        print("[writer] other compatible candidates:", flush=True)
        for p in uniq[1:10]:
            print("   ", p, flush=True)
        print("[writer] For exact causal-scan comparability, pass --writer-npz explicitly if needed.", flush=True)
    return chosen


def load_writers(path: Path, targets: Sequence[int]):
    writers = {int(T): {} for T in targets}
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        for T in targets:
            for r in REL:
                key = next((k for k in writer_key_candidates(T, r) if k in keys), None)
                if key is None:
                    raise RuntimeError(
                        f"{path}: missing writer for L{T}/{r}; tried {writer_key_candidates(T,r)}"
                    )
                writers[T][r] = normalize_np(np.asarray(z[key], dtype=np.float32))
    return writers


# =============================================================================
# Backprojection
# =============================================================================

def find_object_positions(two, processor, batch, subject: str, reference: str):
    tok = old.tokenizer_of(processor)
    ids = batch["input_ids"][0].detach().cpu().tolist()
    sub = int(two.find_phrase_last_token(tok, ids, subject))
    ref = int(two.find_phrase_last_token(tok, ids, reference))
    if sub == ref:
        raise RuntimeError(f"subject/reference positions collide at {sub}")
    return sub, ref


def build_writer_objectives(cap, writers, targets, mode: str, combine: str):
    raw = {}
    for r in REL:
        terms = []
        for T in targets:
            v = torch.as_tensor(
                writers[T][r],
                dtype=torch.float32,
                device=cap.states[T].device,
            )
            terms.append(torch.dot(cap.states[T][0, -1].float(), v))
        stack = torch.stack(terms)
        raw[r] = stack.mean() if combine == "mean" else stack.sum()

    if mode == "raw":
        return raw, raw

    common = torch.stack([raw[r] for r in REL]).mean()
    centered = {r: raw[r] - common for r in REL}
    return raw, centered


def backproject_one_sample(
    *,
    model,
    decoder_layers,
    batch,
    source_layers,
    target_layers,
    writers,
    subject_pos,
    reference_pos,
    objective_mode,
    target_combine,
):
    graph_layers = sorted(set(source_layers + target_layers))
    cap = old.GraphBlockCapture(decoder_layers, graph_layers)
    try:
        with torch.enable_grad():
            kw = dict(batch)
            kw["use_cache"] = False
            kw["return_dict"] = True
            _out = model(**kw)

            missing = [L for L in graph_layers if L not in cap.states]
            if missing:
                raise RuntimeError(f"Graph capture missing layers {missing}")

            seq_len = int(cap.states[source_layers[0]].shape[1])
            if not (0 <= subject_pos < seq_len and 0 <= reference_pos < seq_len):
                raise RuntimeError(
                    f"Object positions out of range: sub={subject_pos}, ref={reference_pos}, seq={seq_len}"
                )

            raw_obj, obj = build_writer_objectives(
                cap, writers, target_layers, objective_mode, target_combine
            )

            src_tensors = [cap.states[L] for L in source_layers]
            back = {L: {} for L in source_layers}
            grad_meta = []

            for ri, r in enumerate(REL):
                grads = torch.autograd.grad(
                    obj[r],
                    src_tensors,
                    retain_graph=(ri < len(REL) - 1),
                    create_graph=False,
                    allow_unused=True,
                )
                for L, g in zip(source_layers, grads):
                    if g is None:
                        vec = np.zeros(cap.states[L].shape[-1], dtype=np.float32)
                        gs = np.zeros_like(vec)
                        gr = np.zeros_like(vec)
                    else:
                        gs = g[0, subject_pos].detach().float().cpu().numpy().astype(np.float32)
                        gr = g[0, reference_pos].detach().float().cpu().numpy().astype(np.float32)
                        vec = (gs - gr).astype(np.float32)
                    back[L][r] = vec
                    grad_meta.append({
                        "source_layer": int(L),
                        "decision_relation": r,
                        "subject_grad_norm": float(np.linalg.norm(gs)),
                        "reference_grad_norm": float(np.linalg.norm(gr)),
                        "backproject_norm": float(np.linalg.norm(vec)),
                    })

            obj_values = {
                f"raw_writer_objective_{r}": float(raw_obj[r].detach().item())
                for r in REL
            }
            obj_values.update({
                f"used_writer_objective_{r}": float(obj[r].detach().item())
                for r in REL
            })
            return back, grad_meta, obj_values
    finally:
        cap.close()


# =============================================================================
# Summaries
# =============================================================================

def relation_matrix_from_mean_vectors(spatial_dirs, mean_back, L: int):
    rows = []
    M = np.full((4, 4), np.nan, dtype=np.float64)
    for i, sr in enumerate(REL):
        for j, dr in enumerate(REL):
            c = cosine_np(spatial_dirs[L][sr], mean_back[L][dr])
            M[i, j] = c
            rows.append({
                "source_layer": int(L),
                "spatial_relation": sr,
                "decision_relation": dr,
                "cosine_of_mean_backproject": c,
            })
    return M, rows


def matrix_structure_rows(M: np.ndarray, L: int):
    def m(i, j):
        return float(M[RID[i], RID[j]])

    diag = np.nanmean(np.diag(M))
    off = np.nanmean(M[~np.eye(4, dtype=bool)])
    horizontal = m("left", "left") + m("right", "right") - m("left", "right") - m("right", "left")
    vertical = m("above", "above") + m("below", "below") - m("above", "below") - m("below", "above")
    return {
        "source_layer": int(L),
        "diag_mean": float(diag),
        "offdiag_mean": float(off),
        "diag_minus_offdiag": float(diag - off),
        "horizontal_pair_contrast": float(horizontal),
        "vertical_pair_contrast": float(vertical),
        "mean_pair_contrast": float((horizontal + vertical) / 2.0),
    }


def exact_permutation_scores(M: np.ndarray, L: int):
    rows = []
    # mapping: spatial row i is paired with decision column perm[i]
    for perm in itertools.permutations(range(4)):
        score = float(np.nanmean([M[i, perm[i]] for i in range(4)]))
        rows.append({
            "source_layer": int(L),
            "mapping": ",".join(f"{REL[i]}->{REL[perm[i]]}" for i in range(4)),
            "score": score,
            "is_identity": bool(all(perm[i] == i for i in range(4))),
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["percentile_vs_24"] = 1.0 - (rank - 1) / 23.0
    return rows


def axis_summary(spatial_axes, mean_back, L: int):
    b_h = normalize_np(mean_back[L]["right"] - mean_back[L]["left"])
    b_v = normalize_np(mean_back[L]["above"] - mean_back[L]["below"])
    d_h = spatial_axes[L]["horizontal"]
    d_v = spatial_axes[L]["vertical"]

    hh = cosine_np(d_h, b_h)
    hv = cosine_np(d_h, b_v)
    vv = cosine_np(d_v, b_v)
    vh = cosine_np(d_v, b_h)
    matched = np.nanmean([hh, vv])
    crossed = np.nanmean([hv, vh])
    return {
        "source_layer": int(L),
        "cos_spatialH_backH": hh,
        "cos_spatialH_backV": hv,
        "cos_spatialV_backV": vv,
        "cos_spatialV_backH": vh,
        "matched_mean": float(matched),
        "crossed_mean": float(crossed),
        "matched_minus_crossed": float(matched - crossed),
        "backH_norm": float(np.linalg.norm(mean_back[L]["right"] - mean_back[L]["left"])),
        "backV_norm": float(np.linalg.norm(mean_back[L]["above"] - mean_back[L]["below"])),
    }, b_h, b_v


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_layers(a.source_layers)
    target_layers = parse_layers(a.target_layers)
    if min(target_layers) <= min(source_layers):
        raise ValueError("Target writer layers should be downstream of source layers")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    err_path = outdir / "errors.jsonl"

    # -------------------------------------------------------------------------
    # Frozen spatial codebook from REAL-NoImage residual states.
    # -------------------------------------------------------------------------
    spatial_path = Path(a.spatial_states_npz)
    X, y, spatial_layers, spatial_sids = load_spatial_npz(spatial_path)
    fit_idx, held_idx, fit_sids, heldout_sids = make_spatial_split(
        y, spatial_sids, a.spatial_fit_ratio, a.spatial_fit_seed
    )
    spatial_dirs, spatial_axes, codebook_df = fit_spatial_directions(
        X, y, spatial_layers, fit_idx, source_layers
    )
    codebook_df.to_csv(outdir / "spatial_codebook.csv", index=False)

    # -------------------------------------------------------------------------
    # Dataset metadata. Load all first, then apply heldout/all scope ourselves.
    # -------------------------------------------------------------------------
    # old.load_data expects these attributes.
    a.max_samples = 0
    old_eval_max = a.eval_max_samples
    a.eval_max_samples = 0
    two, all_meta, rec_by_sid = old.load_data(a)
    a.eval_max_samples = old_eval_max

    spatial_sid_set = set(map(int, spatial_sids.tolist()))
    meta = [m for m in all_meta if int(m["sid"]) in spatial_sid_set]
    if a.eval_scope == "heldout":
        meta = [m for m in meta if int(m["sid"]) in heldout_sids]
    meta = stratified_cap(meta, a.eval_max_samples, a.seed)
    if not meta:
        raise RuntimeError("No evaluation samples after spatial/eval-scope filtering")

    # -------------------------------------------------------------------------
    # Writers / model.
    # -------------------------------------------------------------------------
    writer_path = resolve_writer_npz(a, target_layers)
    writers = load_writers(writer_path, target_layers)

    model = processor = None
    per_rows = []
    sample_rows = []
    accum = {
        L: {r: np.zeros(X.shape[-1], dtype=np.float64) for r in REL}
        for L in source_layers
    }
    accum_n = {L: {r: 0 for r in REL} for L in source_layers}

    try:
        model, processor, decoder_layers, decoder_path, spec = old.load_model(a, two)
        device = torch.device(a.device)
        if max(target_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested target L{max(target_layers)} but model has {len(decoder_layers)} decoder layers"
            )

        print("\n" + "=" * 150)
        print("BACK-PROJECTED WRITER -> OBJECT SPATIAL BRIDGE")
        print("=" * 150)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"source layers={source_layers} target writers={target_layers}")
        print(f"writer={writer_path}")
        print(f"spatial={spatial_path}")
        print(f"spatial fit N={len(fit_idx)} heldout N={len(held_idx)} ratio={a.spatial_fit_ratio}")
        print(f"eval scope={a.eval_scope} N={len(meta)}")
        print(f"objective={a.objective_mode}; target combine={a.target_combine}")
        print(f"decoder={decoder_path}")
        print("=" * 150, flush=True)

        for m in tqdm(meta, desc="BACKPROJECT writer->objects"):
            sid = int(m["sid"])
            image = batch = None
            try:
                image = old.base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = old.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                sub_pos, ref_pos = find_object_positions(
                    two, processor, batch, m["subject"], m["reference"]
                )

                back, grad_meta, obj_values = backproject_one_sample(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    source_layers=source_layers,
                    target_layers=target_layers,
                    writers=writers,
                    subject_pos=sub_pos,
                    reference_pos=ref_pos,
                    objective_mode=a.objective_mode,
                    target_combine=a.target_combine,
                )

                sample_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "subject": m["subject"],
                    "reference": m["reference"],
                    "subject_position": sub_pos,
                    "reference_position": ref_pos,
                    **obj_values,
                })

                grad_lookup = {
                    (int(r["source_layer"]), str(r["decision_relation"])): r
                    for r in grad_meta
                }

                for L in source_layers:
                    for dr in REL:
                        b = back[L][dr]
                        accum[L][dr] += b.astype(np.float64)
                        accum_n[L][dr] += 1
                        gm = grad_lookup[(L, dr)]
                        row = {
                            "sid": sid,
                            "gt": m["gt"],
                            "source_layer": L,
                            "decision_relation": dr,
                            **gm,
                        }
                        for sr in REL:
                            row[f"cos_to_spatial_{sr}"] = cosine_np(b, spatial_dirs[L][sr])
                        # Per-sample axis-oriented views from this relation-specific b are
                        # kept in the 4x4 columns; actual H/V contrasts are summarized after
                        # subtracting relation gradients below.
                        per_rows.append(row)

                # Per-sample explicit H/V bridge rows.
                for L in source_layers:
                    b_h = back[L]["right"] - back[L]["left"]
                    b_v = back[L]["above"] - back[L]["below"]
                    per_rows.append({
                        "sid": sid,
                        "gt": m["gt"],
                        "source_layer": L,
                        "decision_relation": "__axis_horizontal__",
                        "subject_grad_norm": float("nan"),
                        "reference_grad_norm": float("nan"),
                        "backproject_norm": float(np.linalg.norm(b_h)),
                        "cos_to_spatial_left": float("nan"),
                        "cos_to_spatial_right": float("nan"),
                        "cos_to_spatial_above": float("nan"),
                        "cos_to_spatial_below": float("nan"),
                        "axis_match_cos": cosine_np(b_h, spatial_axes[L]["horizontal"]),
                        "axis_cross_cos": cosine_np(b_h, spatial_axes[L]["vertical"]),
                    })
                    per_rows.append({
                        "sid": sid,
                        "gt": m["gt"],
                        "source_layer": L,
                        "decision_relation": "__axis_vertical__",
                        "subject_grad_norm": float("nan"),
                        "reference_grad_norm": float("nan"),
                        "backproject_norm": float(np.linalg.norm(b_v)),
                        "cos_to_spatial_left": float("nan"),
                        "cos_to_spatial_right": float("nan"),
                        "cos_to_spatial_above": float("nan"),
                        "cos_to_spatial_below": float("nan"),
                        "axis_match_cos": cosine_np(b_v, spatial_axes[L]["vertical"]),
                        "axis_cross_cos": cosine_np(b_v, spatial_axes[L]["horizontal"]),
                    })

            except Exception as exc:
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-12:],
                    }, ensure_ascii=False) + "\n")
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                batch = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not sample_rows:
            raise RuntimeError(f"No successful samples. See {err_path}")

        per_df = pd.DataFrame(per_rows)
        sample_df = pd.DataFrame(sample_rows)
        per_df.to_csv(outdir / "per_sample_backproject_alignment.csv", index=False)
        sample_df.to_csv(outdir / "sample_objectives.csv", index=False)

        # Mean backprojected vectors.
        mean_back = {L: {} for L in source_layers}
        for L in source_layers:
            for r in REL:
                n = accum_n[L][r]
                if n <= 0:
                    raise RuntimeError(f"No backproject vectors accumulated for L{L}/{r}")
                mean_back[L][r] = (accum[L][r] / n).astype(np.float32)

        matrix_rows = []
        structure_rows = []
        perm_rows = []
        axis_rows = []
        vector_npz = {}

        for L in source_layers:
            M, mr = relation_matrix_from_mean_vectors(spatial_dirs, mean_back, L)
            matrix_rows.extend(mr)
            structure_rows.append(matrix_structure_rows(M, L))
            perm_rows.extend(exact_permutation_scores(M, L))
            ar, b_h, b_v = axis_summary(spatial_axes, mean_back, L)
            axis_rows.append(ar)

            for r in REL:
                vector_npz[f"L{L}_spatial_{r}"] = spatial_dirs[L][r].astype(np.float32)
                vector_npz[f"L{L}_backproj_{r}"] = mean_back[L][r].astype(np.float32)
            vector_npz[f"L{L}_spatial_horizontal"] = spatial_axes[L]["horizontal"].astype(np.float32)
            vector_npz[f"L{L}_spatial_vertical"] = spatial_axes[L]["vertical"].astype(np.float32)
            vector_npz[f"L{L}_backproj_horizontal"] = b_h.astype(np.float32)
            vector_npz[f"L{L}_backproj_vertical"] = b_v.astype(np.float32)

        matrix_df = pd.DataFrame(matrix_rows)
        structure_df = pd.DataFrame(structure_rows)
        perm_df = pd.DataFrame(perm_rows)
        axis_df = pd.DataFrame(axis_rows)

        # Add mean per-sample cosine to the 4x4 table (distinct from cosine of mean vector).
        rel_per = per_df[per_df["decision_relation"].isin(REL)].copy()
        sample_matrix_rows = []
        for L in source_layers:
            z = rel_per[rel_per["source_layer"] == L]
            for sr in REL:
                for dr in REL:
                    vals = pd.to_numeric(
                        z.loc[z["decision_relation"] == dr, f"cos_to_spatial_{sr}"],
                        errors="coerce",
                    ).dropna()
                    sample_matrix_rows.append({
                        "source_layer": L,
                        "spatial_relation": sr,
                        "decision_relation": dr,
                        "mean_per_sample_cosine": float(vals.mean()) if len(vals) else float("nan"),
                        "std_per_sample_cosine": float(vals.std(ddof=0)) if len(vals) else float("nan"),
                        "N": int(len(vals)),
                    })
        sample_matrix_df = pd.DataFrame(sample_matrix_rows)
        matrix_df = matrix_df.merge(
            sample_matrix_df,
            on=["source_layer", "spatial_relation", "decision_relation"],
            how="left",
        )

        # Per-sample axis summary by layer.
        axis_per = per_df[per_df["decision_relation"].str.startswith("__axis_")].copy()
        axis_sample_summary = []
        for L in source_layers:
            zh = axis_per[(axis_per["source_layer"] == L) & (axis_per["decision_relation"] == "__axis_horizontal__")]
            zv = axis_per[(axis_per["source_layer"] == L) & (axis_per["decision_relation"] == "__axis_vertical__")]
            hm = pd.to_numeric(zh["axis_match_cos"], errors="coerce").dropna()
            hc = pd.to_numeric(zh["axis_cross_cos"], errors="coerce").dropna()
            vm = pd.to_numeric(zv["axis_match_cos"], errors="coerce").dropna()
            vc = pd.to_numeric(zv["axis_cross_cos"], errors="coerce").dropna()
            axis_sample_summary.append({
                "source_layer": L,
                "mean_sample_H_match": float(hm.mean()) if len(hm) else float("nan"),
                "mean_sample_H_cross": float(hc.mean()) if len(hc) else float("nan"),
                "mean_sample_V_match": float(vm.mean()) if len(vm) else float("nan"),
                "mean_sample_V_cross": float(vc.mean()) if len(vc) else float("nan"),
                "mean_sample_matched": float(np.nanmean([hm.mean() if len(hm) else np.nan, vm.mean() if len(vm) else np.nan])),
                "mean_sample_crossed": float(np.nanmean([hc.mean() if len(hc) else np.nan, vc.mean() if len(vc) else np.nan])),
                "N_horizontal": int(len(hm)),
                "N_vertical": int(len(vm)),
            })
        axis_sample_df = pd.DataFrame(axis_sample_summary)
        axis_sample_df["mean_sample_matched_minus_crossed"] = (
            axis_sample_df["mean_sample_matched"] - axis_sample_df["mean_sample_crossed"]
        )
        axis_df = axis_df.merge(axis_sample_df, on="source_layer", how="left")

        matrix_df.to_csv(outdir / "alignment_matrix_by_layer.csv", index=False)
        structure_df.to_csv(outdir / "matrix_structure_by_layer.csv", index=False)
        perm_df.to_csv(outdir / "relation_permutation_scores.csv", index=False)
        axis_df.to_csv(outdir / "axis_alignment_by_layer.csv", index=False)
        np.savez_compressed(outdir / "mean_bridge_vectors.npz", **vector_npz)

        # Global summaries across source layers.
        identity = perm_df[perm_df["is_identity"]].copy()
        best_layer_axis = axis_df.loc[axis_df["matched_minus_crossed"].idxmax()]
        best_layer_pair = structure_df.loc[structure_df["mean_pair_contrast"].idxmax()]

        # Global 4x4 by averaging mean-vector matrices layerwise.
        global_matrix = (
            matrix_df.groupby(["spatial_relation", "decision_relation"], as_index=False)
            ["cosine_of_mean_backproject"].mean()
        )
        G = np.full((4, 4), np.nan, dtype=np.float64)
        for row in global_matrix.itertuples():
            G[RID[row.spatial_relation], RID[row.decision_relation]] = row.cosine_of_mean_backproject
        global_perm = exact_permutation_scores(G, -1)
        global_identity = next(r for r in global_perm if r["is_identity"])

        report = [
            "=" * 150,
            "BACK-PROJECTED WRITER -> OBJECT SPATIAL BRIDGE",
            "=" * 150,
            f"model={a.model} repo={spec.repo_id}",
            f"N successful={len(sample_df)} / requested={len(meta)}",
            f"source layers={source_layers}",
            f"target writer layers={target_layers}",
            f"writer={writer_path}",
            f"spatial states={spatial_path}",
            f"spatial fit/eval={len(fit_idx)}/{len(held_idx)}; eval_scope={a.eval_scope}",
            f"objective_mode={a.objective_mode}; target_combine={a.target_combine}",
            "",
            "AXIS BRIDGE (PRIMARY)",
            "-" * 150,
            axis_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "",
            "4x4 STRUCTURE",
            "-" * 150,
            structure_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "",
            "IDENTITY RELATION-MAPPING RANK (1=best among all 24 permutations)",
            "-" * 150,
            identity[["source_layer", "score", "rank", "percentile_vs_24"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            ),
            "",
            f"Best axis layer: L{int(best_layer_axis.source_layer)} "
            f"matched-crossed={best_layer_axis.matched_minus_crossed:.4f} "
            f"(H/H={best_layer_axis.cos_spatialH_backH:.4f}, "
            f"V/V={best_layer_axis.cos_spatialV_backV:.4f})",
            f"Best pair-contrast layer: L{int(best_layer_pair.source_layer)} "
            f"mean_pair_contrast={best_layer_pair.mean_pair_contrast:.4f}",
            f"Global averaged 4x4 identity mapping rank: {global_identity['rank']}/24 "
            f"score={global_identity['score']:.4f}",
            "",
            "Interpretation:",
            "  Strong evidence for a spatial->decision transport bridge requires structure, not merely nonzero cosine:",
            "  (1) matched H/H and V/V exceed crossed H/V and V/H;",
            "  (2) left/right and above/below pair contrasts are positive;",
            "  (3) identity relation mapping ranks near the top of the 24 possible permutations.",
            "  If these fail despite strong late causal steering, the current object-token codebook is not the right upstream",
            "  coordinate system for the writer computation, or the bridge is mediated by a different token/head representation.",
        ]
        text = "\n".join(report) + "\n"
        print("\n" + text, flush=True)
        (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

        old.write_json(outdir / "metadata.json", {
            "script": "analyze_backprojected_writer_spatial_bridge_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "spatial_states_npz": str(spatial_path),
            "writer_npz": str(writer_path),
            "source_layers": source_layers,
            "target_layers": target_layers,
            "objective_mode": a.objective_mode,
            "target_combine": a.target_combine,
            "spatial_fit_ratio": a.spatial_fit_ratio,
            "spatial_fit_seed": a.spatial_fit_seed,
            "spatial_fit_N": int(len(fit_idx)),
            "spatial_heldout_N": int(len(held_idx)),
            "eval_scope": a.eval_scope,
            "eval_requested_N": int(len(meta)),
            "eval_successful_N": int(len(sample_df)),
            "backproject_definition": "grad_subject(J_r) - grad_reference(J_r)",
            "axis_definition": {
                "spatial_horizontal": "unit(d_right-d_left)",
                "spatial_vertical": "unit(d_above-d_below)",
                "backproj_horizontal": "unit(b_right-b_left)",
                "backproj_vertical": "unit(b_above-b_below)",
            },
        })

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
