#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_qwen_spatial_to_decision_v1.py

Goal
----
Test whether the middle states selected by writer-guided causal attribution are
themselves "explicit spatial information", or whether they support the conversion
from explicit spatial representations into a late decision.

This script reuses the repo's current Qwen/COCO pipeline:

    analyze_coco_centroid_generation_step1_v4.py
    eval_coco_multilayer_relation_trajectory_repair_v1.py
    eval_qwen_dynamic_k24_all440_v1.py

It performs three analyses in one run:

A) Spatial-subspace x causal-attribution map
   Learn, on the calibration split, a layer-wise explicit spatial subspace from
   object-pair residuals

       r_L = mean(h_subject,L) - mean(h_reference,L)

   using four relation centroids. For every candidate middle-token Real-Gray
   delta, measure:
       - writer-aligned mediation M = delta_h dot grad(J)
       - fraction of delta_h lying in the explicit spatial subspace
       - cosine to the GT relation direction in that subspace

B) Full vs spatial-parallel vs spatial-orthogonal causal intervention
   For the SAME oracle-selected positions, compare actual model.generate():

       full:        + alpha * delta_h
       parallel:    + alpha * P_spatial(delta_h)
       parallel_nm: + alpha * norm-matched P_spatial(delta_h)
       orthogonal:  + alpha * (I-P_spatial)(delta_h)

   IMPORTANT:
   "orthogonal" means orthogonal to THIS measured explicit object-pair spatial
   subspace. It does NOT mean mathematically "non-spatial in every possible code".

C) Spatial-to-decision trajectory
   During each generation intervention, record:
       - downstream object-pair spatial readout at chosen layers
       - late writer/decision projection at target layers
       - actual generated answer

This lets us distinguish, empirically:
    1) explicit spatial-content states,
    2) states that alter downstream spatial representation,
    3) states that mainly alter late decision/readout without strongly changing
       the explicit spatial code.

The writer-guided token selection remains ORACLE because the GT relation selects
the late writer target. This is a mechanistic diagnostic, not a deployable selector.

Recommended quick run
---------------------
python analyze_qwen_spatial_to_decision_v1.py \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --spatial-readout-layers 20,21,22,23,24,25,26,28,30,32 \
  --target-layers 32,34,35 \
  --k 36 \
  --alpha 1.0 \
  --eval-scope test \
  --eval-max-samples 80 \
  --output-dir outputs/spatial_to_decision_v1_n80 \
  --overwrite

Full held-out run:
python analyze_qwen_spatial_to_decision_v1.py \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --spatial-readout-layers 20,21,22,23,24,25,26,27,28,29,30,31,32 \
  --target-layers 32,34,35 \
  --k 36 \
  --alpha 1.0 \
  --eval-scope test \
  --eval-max-samples 0 \
  --output-dir outputs/spatial_to_decision_v1_heldout \
  --overwrite

Notes
-----
* Use --eval-scope test first. That keeps calibration and evaluation disjoint.
* --eval-scope all_data is only a full-dataset mechanistic diagnostic and overlaps
  with the calibration samples, matching the caveat in the current dynamic script.
* Raw parallel effects can be small simply because a 2-3D measured spatial subspace
  captures little delta_h norm. Therefore parallel_normmatched is included as a
  direction-control, but very large norm-matching factors are capped and should be
  interpreted cautiously.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# ---------------------------------------------------------------------
# Args / generic helpers
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])

    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument(
        "--spatial-readout-layers",
        default="20,21,22,23,24,25,26,28,30,32",
        help=(
            "Layers where explicit spatial object-pair readout is learned and "
            "tracked under intervention."
        ),
    )
    p.add_argument("--target-layers", default="32,34,35")

    p.add_argument("--k", type=int, default=36)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--norm-match-cap",
        type=float,
        default=10.0,
        help=(
            "Maximum per-token multiplier for norm-matched spatial-parallel "
            "control. Prevents exploding vectors when raw spatial projection is tiny."
        ),
    )
    p.add_argument(
        "--conditions",
        default="full,spatial_parallel_raw,spatial_parallel_normmatched,spatial_orthogonal_raw",
        help=(
            "Comma separated subset of: full,spatial_parallel_raw,"
            "spatial_parallel_normmatched,spatial_orthogonal_raw"
        ),
    )

    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--eval-scope", default="test", choices=["test", "all_data"])

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return sorted(
        {
            int(x.strip().upper().replace("L", ""))
            for x in str(text).split(",")
            if x.strip()
        }
    )


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    vals = np.asarray([float(x) for x in xs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def safe_corr(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    sx, sy = float(x.std()), float(y.std())
    if sx < EPS or sy < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v / n).astype(np.float32)


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def mean_positions(H: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    good = [int(p) for p in positions if 0 <= int(p) < H.shape[0]]
    if not good:
        raise RuntimeError("No valid object-token positions")
    return H[good].mean(axis=0).astype(np.float32)


def object_pair_residual(
    H: np.ndarray,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
) -> np.ndarray:
    return (
        mean_positions(H, subject_positions)
        - mean_positions(H, reference_positions)
    ).astype(np.float32)


def get_object_positions(processor, ids, subject, reference):
    tokenizer = processor.tokenizer
    try:
        sspan, rspan = base.locate_object_spans(
            tokenizer, ids, subject, reference
        )
        spos = sorted(dyn.span_positions(sspan))
        rpos = sorted(dyn.span_positions(rspan))
    except Exception:
        spos = dyn.find_text_positions(tokenizer, ids, subject)
        rpos = dyn.find_text_positions(tokenizer, ids, reference)
    if not spos or not rpos:
        raise RuntimeError(
            f"Could not locate object tokens: subject={subject!r}, reference={reference!r}"
        )
    return spos, rpos


# ---------------------------------------------------------------------
# Explicit spatial code:
# relation centroids in object-pair residual space
# ---------------------------------------------------------------------

def learn_explicit_spatial_codes(
    train_pair_by_sid: Dict[int, Dict[int, np.ndarray]],
    train_meta,
    spatial_layers: Sequence[int],
):
    """
    For each layer:
        r_i = mean(h_subject) - mean(h_reference)

    Learn four class means, center by their common mean, and define:
      - relation direction d_{L,r}
      - explicit spatial subspace = span(centered class means)

    Since there are four centered class means, rank <= 3.
    """
    codes = {}
    geom_rows = []

    for L in spatial_layers:
        means = {}
        for r in REL:
            xs = [
                train_pair_by_sid[int(m["sid"])][L]
                for m in train_meta
                if m["gt"] == r and int(m["sid"]) in train_pair_by_sid
            ]
            if not xs:
                raise RuntimeError(f"No calibration pair residuals for {r} at L{L}")
            means[r] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        common = np.mean(np.stack([means[r] for r in REL]), axis=0).astype(np.float32)
        centered = {
            r: (means[r] - common).astype(np.float32)
            for r in REL
        }

        # Basis of the centered class-mean row space.
        A = np.stack([centered[r] for r in REL], axis=0).astype(np.float64)
        _u, s, vt = np.linalg.svd(A, full_matrices=False)
        tol = max(A.shape) * np.finfo(np.float64).eps * (float(s[0]) if s.size else 1.0)
        rank = int(np.sum(s > tol))
        rank = max(1, min(rank, 3))
        Q = vt[:rank].T.astype(np.float32)  # [d, rank], orthonormal columns

        dirs = {r: normalize_np(centered[r]) for r in REL}
        codes[L] = {
            "means": means,
            "common": common,
            "centered": centered,
            "dirs": dirs,
            "basis": Q,
            "rank": rank,
            "singular_values": s.astype(np.float32),
        }

        for r in REL:
            geom_rows.append({
                "layer": L,
                "relation": DISPLAY[r],
                "basis_rank": rank,
                "centered_norm": float(np.linalg.norm(centered[r])),
                "cos_left": cosine_np(centered[r], centered["left"]),
                "cos_right": cosine_np(centered[r], centered["right"]),
                "cos_on": cosine_np(centered[r], centered["above"]),
                "cos_under": cosine_np(centered[r], centered["below"]),
            })

    return codes, geom_rows


def spatial_readout(pair_vec: np.ndarray, code, gt: str):
    """
    Cosine nearest-centroid readout in the explicit object-pair relation code.
    """
    scores = {
        r: cosine_np(pair_vec, code["centered"][r])
        for r in REL
    }
    pred = max(REL, key=lambda r: -1e9 if not np.isfinite(scores[r]) else scores[r])
    gt_score = float(scores[gt])
    others = [float(scores[r]) for r in REL if r != gt and np.isfinite(scores[r])]
    margin = gt_score - max(others) if others and np.isfinite(gt_score) else float("nan")
    return scores, pred, gt_score, margin


def project_spatial(delta: np.ndarray, code):
    Q = code["basis"]
    d = np.asarray(delta, np.float32)
    par = (Q @ (Q.T @ d)).astype(np.float32)
    orth = (d - par).astype(np.float32)
    return par, orth


def norm_match(vec: np.ndarray, target_norm: float, cap: float):
    v = np.asarray(vec, np.float32)
    n = float(np.linalg.norm(v))
    if n < EPS or target_norm < EPS:
        return np.zeros_like(v), 0.0
    scale = min(float(cap), float(target_norm) / n)
    return (v * scale).astype(np.float32), float(scale)


# ---------------------------------------------------------------------
# Recording downstream spatial code + late decision writer during generate
# ---------------------------------------------------------------------

class AnalysisRecorder:
    def __init__(
        self,
        decoder_layers,
        prompt_len: int,
        spatial_layers: Sequence[int],
        target_layers: Sequence[int],
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
    ):
        self.prompt_len = int(prompt_len)
        self.spatial_layers = set(map(int, spatial_layers))
        self.target_layers = set(map(int, target_layers))
        self.subject_positions = list(map(int, subject_positions))
        self.reference_positions = list(map(int, reference_positions))
        self.pair = {}
        self.last = {}
        self.handles = []

        for L in sorted(self.spatial_layers | self.target_layers):
            self.handles.append(
                decoder_layers[L].register_forward_hook(self._hook(L))
            )

    def _hook(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            # Only the prefill pass; cached decoding steps have seq_len=1.
            if int(h.shape[1]) != self.prompt_len:
                return out

            H = h[0].detach().float().cpu().numpy().astype(np.float32)

            if L in self.spatial_layers and L not in self.pair:
                try:
                    self.pair[L] = object_pair_residual(
                        H, self.subject_positions, self.reference_positions
                    )
                except Exception:
                    pass

            if L in self.target_layers and L not in self.last:
                self.last[L] = H[-1].copy()

            return out

        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def run_generation_analysis(
    model,
    processor,
    decoder_layers,
    batch,
    gt,
    spatial_codes,
    spatial_layers,
    target_layers,
    writers_for_relation,
    subject_positions,
    reference_positions,
    max_new_tokens,
    token_specs=None,
    token_alpha=None,
):
    """
    Register editor first, recorder second, so recorder sees the edited states.
    """
    prompt_len = int(batch["input_ids"].shape[1])
    editor = None
    recorder = None

    try:
        if token_specs is not None:
            editor = dyn.TokenDeltaEditor(
                decoder_layers, token_specs, float(token_alpha), prompt_len
            )

        recorder = AnalysisRecorder(
            decoder_layers=decoder_layers,
            prompt_len=prompt_len,
            spatial_layers=spatial_layers,
            target_layers=target_layers,
            subject_positions=subject_positions,
            reference_positions=reference_positions,
        )

        text = base.generate_text(
            model, processor, batch, max_new_tokens=max_new_tokens
        )
        pred = traj.normalize_relation(base, text)

        spatial = {}
        for L in spatial_layers:
            if L not in recorder.pair:
                continue
            scores, spred, gt_score, margin = spatial_readout(
                recorder.pair[L], spatial_codes[L], gt
            )
            spatial[L] = {
                "pred": spred,
                "gt_score": gt_score,
                "margin": margin,
                "scores": scores,
            }

        writer = {}
        for T in target_layers:
            if T not in recorder.last:
                continue
            w = writers_for_relation[T]
            writer[T] = {
                "projection": float(np.dot(recorder.last[T], normalize_np(w))),
                "cosine": cosine_np(recorder.last[T], w),
            }

        return {
            "text": text,
            "prediction": pred,
            "correct": pred == gt,
            "spatial": spatial,
            "writer": writer,
            "edit_applied": dict(editor.applied) if editor is not None else {},
        }

    finally:
        if recorder is not None:
            recorder.close()
        if editor is not None:
            editor.close()


# ---------------------------------------------------------------------
# Build intervention specs from exactly the same selected positions
# ---------------------------------------------------------------------

def row_lookup(rows_by_layer):
    out = {}
    for L, rows in rows_by_layer.items():
        for r in rows:
            out[(int(L), int(r["position"]))] = r
    return out


def selected_to_specs(
    selected_export,
    rows_by_layer,
    spatial_codes,
    condition: str,
    norm_match_cap: float,
):
    lookup = row_lookup(rows_by_layer)
    specs = defaultdict(list)
    stat_rows = []

    for sel in selected_export:
        L = int(sel["source_layer"])
        pos = int(sel["position"])
        row = lookup[(L, pos)]
        delta = np.asarray(row["delta_h"], np.float32)

        par, orth = project_spatial(delta, spatial_codes[L])
        full_norm = float(np.linalg.norm(delta))
        par_norm = float(np.linalg.norm(par))
        orth_norm = float(np.linalg.norm(orth))
        frac = par_norm / max(full_norm, EPS)

        if condition == "full":
            use = delta
            nm_scale = 1.0
        elif condition == "spatial_parallel_raw":
            use = par
            nm_scale = 1.0
        elif condition == "spatial_parallel_normmatched":
            use, nm_scale = norm_match(par, full_norm, norm_match_cap)
        elif condition == "spatial_orthogonal_raw":
            use = orth
            nm_scale = 1.0
        else:
            raise ValueError(condition)

        specs[L].append((pos, use.astype(np.float32)))
        stat_rows.append({
            "source_layer": L,
            "position": pos,
            "condition": condition,
            "full_delta_norm": full_norm,
            "parallel_norm": par_norm,
            "orthogonal_norm": orth_norm,
            "parallel_fraction": frac,
            "normmatch_scale": nm_scale,
            "edit_norm": float(np.linalg.norm(use)),
        })

    return dict(specs), stat_rows


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_behavior(baseline_rows, condition_rows):
    base = {int(r["sid"]): r for r in baseline_rows}
    grouped = defaultdict(list)
    for r in condition_rows:
        grouped[r["condition"]].append(r)

    out = []
    for cond, rows in sorted(grouped.items()):
        by_sid = {int(r["sid"]): r for r in rows}
        common = sorted(set(base) & set(by_sid))
        if not common:
            continue

        bacc = safe_mean(float(base[s]["correct"]) for s in common)
        eacc = safe_mean(float(by_sid[s]["correct"]) for s in common)
        w2c = sum((not bool(base[s]["correct"])) and bool(by_sid[s]["correct"]) for s in common)
        c2w = sum(bool(base[s]["correct"]) and (not bool(by_sid[s]["correct"])) for s in common)

        out.append({
            "condition": cond,
            "N": len(common),
            "baseline_acc": bacc,
            "edited_acc": eacc,
            "delta_acc": eacc - bacc,
            "W2C": int(w2c),
            "C2W": int(c2w),
            "net": int(w2c - c2w),
        })
    return out


def summarize_trajectory(trajectory_rows):
    """
    Summarize spatial readout / writer trajectories condition x layer.
    """
    grouped = defaultdict(list)
    for r in trajectory_rows:
        grouped[(r["signal_type"], r["condition"], int(r["layer"]))].append(r)

    out = []
    for (sig, cond, L), rows in sorted(grouped.items()):
        row = {
            "signal_type": sig,
            "condition": cond,
            "layer": L,
            "N": len(rows),
        }
        if sig == "spatial":
            row.update({
                "mean_gt_score": safe_mean(r["gt_score"] for r in rows),
                "mean_margin": safe_mean(r["margin"] for r in rows),
                "readout_acc": safe_mean(float(r["spatial_correct"]) for r in rows),
            })
        else:
            row.update({
                "mean_writer_projection": safe_mean(r["writer_projection"] for r in rows),
                "mean_writer_cosine": safe_mean(r["writer_cosine"] for r in rows),
            })
        out.append(row)

    # Add gain relative to baseline at same signal/layer.
    baseline_map = {}
    for r in out:
        if r["condition"] == "baseline":
            baseline_map[(r["signal_type"], r["layer"])] = r

    for r in out:
        b = baseline_map.get((r["signal_type"], r["layer"]))
        if not b:
            continue
        if r["signal_type"] == "spatial":
            r["delta_mean_margin_vs_baseline"] = (
                float(r["mean_margin"]) - float(b["mean_margin"])
            )
            r["delta_readout_acc_vs_baseline"] = (
                float(r["readout_acc"]) - float(b["readout_acc"])
            )
        else:
            r["delta_writer_projection_vs_baseline"] = (
                float(r["mean_writer_projection"])
                - float(b["mean_writer_projection"])
            )
    return out


def summarize_candidate_relation(candidate_rows, selected_rows):
    out = []

    for name, rows in [
        ("all_candidates", candidate_rows),
        ("selected", selected_rows),
    ]:
        if not rows:
            continue
        out.append({
            "subset": name,
            "N": len(rows),
            "mean_mediation": safe_mean(r["mediation"] for r in rows),
            "mean_abs_mediation": safe_mean(abs(float(r["mediation"])) for r in rows),
            "mean_spatial_parallel_fraction": safe_mean(
                r["spatial_parallel_fraction"] for r in rows
            ),
            "mean_abs_gt_spatial_cos": safe_mean(
                abs(float(r["gt_spatial_cos"])) for r in rows
            ),
            "corr_mediation_vs_parallel_fraction": safe_corr(
                [r["mediation"] for r in rows],
                [r["spatial_parallel_fraction"] for r in rows],
            ),
            "corr_absmed_vs_parallel_fraction": safe_corr(
                [abs(float(r["mediation"])) for r in rows],
                [r["spatial_parallel_fraction"] for r in rows],
            ),
            "corr_mediation_vs_gt_spatial_cos": safe_corr(
                [r["mediation"] for r in rows],
                [r["gt_spatial_cos"] for r in rows],
            ),
        })

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    spatial_layers = parse_ints(a.spatial_readout_layers)
    target_layers = parse_ints(a.target_layers)
    conditions = [x.strip() for x in a.conditions.split(",") if x.strip()]
    allowed_conditions = {
        "full",
        "spatial_parallel_raw",
        "spatial_parallel_normmatched",
        "spatial_orthogonal_raw",
    }
    bad = [x for x in conditions if x not in allowed_conditions]
    if bad:
        raise ValueError(f"Unknown conditions: {bad}")

    # We need a spatial basis at every source layer for delta decomposition.
    spatial_layers = sorted(set(spatial_layers) | set(source_layers))
    calibration_layers = sorted(set(spatial_layers) | set(target_layers))

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
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

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, heldout = traj.stratified_split(meta, a.train_ratio, a.seed)

    if a.eval_scope == "all_data":
        test = list(meta)
    else:
        test = list(heldout)

    test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)

    train_sids = {int(x["sid"]) for x in train}
    test_sids = {int(x["sid"]) for x in test}
    overlap = len(train_sids & test_sids)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    model = processor = None

    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)
        for L in sorted(set(calibration_layers) | set(source_layers)):
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid; model has L0...L{n_layers - 1}")

        cut = min(source_layers)
        graph_layers = sorted(set(source_layers) | set(target_layers))
        device = torch.device(a.device)

        print("=" * 132)
        print("SPATIAL REPRESENTATION -> MIDDLE CAUSAL SUPPORT -> LATE DECISION")
        print("=" * 132)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(
            f"calibration N={len(train)} | eval_scope={a.eval_scope} "
            f"| eval N={len(test)} | overlap={overlap}"
        )
        print(f"source layers={source_layers}")
        print(f"spatial readout layers={spatial_layers}")
        print(f"late writer layers={target_layers}")
        print(f"K={a.k} alpha={a.alpha}")
        print(f"conditions={conditions}")
        print("selection remains ORACLE: GT relation chooses late writer target")
        print()

        # -------------------------------------------------------------
        # 1) Calibration: learn late writers + explicit spatial codes.
        # -------------------------------------------------------------
        q_by_sid = {}
        pair_by_sid = {}

        for m in tqdm(train, desc="CALIBRATE writers + spatial codes"):
            sid = int(m["sid"])
            real = gray = rb = gb = None

            try:
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

                ids = rb["input_ids"][0].detach().cpu().tolist()
                spos, rpos = get_object_positions(
                    processor, ids, m["subject"], m["reference"]
                )

                # One real forward captures spatial + late target layers.
                hr = dyn.capture_cpu(
                    model, decoder_layers, rb, calibration_layers
                )
                # Gray only needed for late Real-Gray writer calibration.
                hg = dyn.capture_cpu(
                    model, decoder_layers, gb, target_layers
                )

                q_by_sid[sid] = {
                    T: (
                        hr[T][0, -1] - hg[T][0, -1]
                    ).astype(np.float32)
                    for T in target_layers
                }

                pair_by_sid[sid] = {
                    L: object_pair_residual(
                        hr[L][0].astype(np.float32), spos, rpos
                    )
                    for L in spatial_layers
                }

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()

        writers, writer_geom = dyn.learn_writers(
            train, q_by_sid, target_layers, a.writer_mode
        )
        spatial_codes, spatial_geom = learn_explicit_spatial_codes(
            pair_by_sid, train, spatial_layers
        )

        write_csv(outdir / "writer_geometry.csv", writer_geom)
        write_csv(outdir / "explicit_spatial_geometry.csv", spatial_geom)

        np.savez_compressed(
            outdir / "learned_explicit_spatial_codes.npz",
            **{
                **{
                    f"L{L}_{DISPLAY[r]}_centered": spatial_codes[L]["centered"][r]
                    for L in spatial_layers for r in REL
                },
                **{
                    f"L{L}_basis": spatial_codes[L]["basis"]
                    for L in spatial_layers
                },
            }
        )

        # -------------------------------------------------------------
        # 2) Evaluation.
        # -------------------------------------------------------------
        baseline_rows = []
        condition_rows = []
        trajectory_rows = []
        candidate_rows = []
        selected_rows = []
        intervention_vector_rows = []

        for m in tqdm(test, desc="EVAL spatial -> causal -> decision"):
            sid = int(m["sid"])
            gt = m["gt"]
            writers_r = {T: writers[T][gt] for T in target_layers}

            real = gray = rb = gb = cap = None

            try:
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

                ids = rb["input_ids"][0].detach().cpu().tolist()
                spos, rpos = get_object_positions(
                    processor, ids, m["subject"], m["reference"]
                )
                cats, toks = dyn.build_categories(
                    model, processor, rb, ids, m["subject"], m["reference"]
                )

                # -----------------------------------------------------
                # Baseline actual generation + trajectories.
                # -----------------------------------------------------
                baseline = run_generation_analysis(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    gt=gt,
                    spatial_codes=spatial_codes,
                    spatial_layers=spatial_layers,
                    target_layers=target_layers,
                    writers_for_relation=writers_r,
                    subject_positions=spos,
                    reference_positions=rpos,
                    max_new_tokens=a.max_new_tokens,
                )

                baseline_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "prediction": DISPLAY.get(
                        baseline["prediction"], baseline["prediction"]
                    ),
                    "correct": baseline["correct"],
                    "text": baseline["text"],
                })

                for L, info in baseline["spatial"].items():
                    trajectory_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "baseline_correct": baseline["correct"],
                        "condition": "baseline",
                        "signal_type": "spatial",
                        "layer": L,
                        "spatial_pred": DISPLAY.get(info["pred"], info["pred"]),
                        "spatial_correct": info["pred"] == gt,
                        "gt_score": info["gt_score"],
                        "margin": info["margin"],
                        **{
                            f"score_{DISPLAY[r]}": info["scores"][r]
                            for r in REL
                        },
                    })

                for T, info in baseline["writer"].items():
                    trajectory_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "baseline_correct": baseline["correct"],
                        "condition": "baseline",
                        "signal_type": "writer",
                        "layer": T,
                        "writer_projection": info["projection"],
                        "writer_cosine": info["cosine"],
                    })

                # -----------------------------------------------------
                # Build writer-guided candidate attribution exactly as
                # current dynamic K24/K36 script.
                # -----------------------------------------------------
                hgray = dyn.capture_cpu(
                    model, decoder_layers, gb, source_layers
                )

                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model, decoder_layers, rb, graph_layers, cut
                    )

                    objective_terms = []
                    for T in target_layers:
                        s_hat = torch.as_tensor(
                            normalize_np(writers_r[T]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        objective_terms.append(
                            torch.dot(
                                cap.states[T][0, -1].float(),
                                s_hat,
                            )
                        )
                    objective = torch.stack(objective_terms).sum()

                    grads = torch.autograd.grad(
                        objective,
                        [cap.states[S] for S in source_layers],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                    rows_by_layer = {}
                    for S, g in zip(source_layers, grads):
                        Hreal = (
                            cap.states[S][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        Hgray = hgray[S][0].astype(np.float32)
                        G = (
                            g[0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                        npos = min(
                            len(ids),
                            len(cats),
                            len(toks),
                            Hreal.shape[0],
                            Hgray.shape[0],
                            G.shape[0],
                        )

                        rowsS = []
                        # Exclude source last token, matching current K36 logic.
                        for pos in range(max(0, npos - 1)):
                            delta = (
                                Hreal[pos] - Hgray[pos]
                            ).astype(np.float32)
                            grad = G[pos]
                            med = float(np.dot(delta, grad))

                            par, orth = project_spatial(
                                delta, spatial_codes[S]
                            )
                            full_norm = float(np.linalg.norm(delta))
                            par_norm = float(np.linalg.norm(par))
                            orth_norm = float(np.linalg.norm(orth))

                            gt_dir = spatial_codes[S]["centered"][gt]
                            gt_spatial_cos = cosine_np(delta, gt_dir)

                            row = {
                                "sid": sid,
                                "relation": DISPLAY[gt],
                                "source_layer": S,
                                "position": pos,
                                "token_id": int(ids[pos]),
                                "token": str(toks[pos]).replace("\n", "\\n"),
                                "category": cats[pos],
                                "broad_category": dyn.broad_category(cats[pos]),
                                "mediation": med,
                                "abs_mediation": abs(med),
                                "delta_h_norm": full_norm,
                                "grad_norm": float(np.linalg.norm(grad)),
                                "spatial_parallel_norm": par_norm,
                                "spatial_orthogonal_norm": orth_norm,
                                "spatial_parallel_fraction": (
                                    par_norm / max(full_norm, EPS)
                                ),
                                "gt_spatial_cos": gt_spatial_cos,
                                "delta_h": delta,  # in-memory only
                            }
                            rowsS.append(row)

                            candidate_rows.append({
                                k: v for k, v in row.items()
                                if k != "delta_h"
                            })

                        rowsS.sort(
                            key=lambda x: x["mediation"], reverse=True
                        )
                        rows_by_layer[S] = rowsS

                cap.close()
                cap = None

                # Oracle global_unique Top-K, same selection principle as K36.
                _full_specs, selected_export = dyn.make_specs(
                    tuple(source_layers),
                    rows_by_layer,
                    mode="positive",
                    k=a.k,
                    sid=sid,
                    seed=a.seed,
                    strategy="global_unique",
                )

                lookup = row_lookup(rows_by_layer)
                for rank, sel in enumerate(selected_export, 1):
                    L = int(sel["source_layer"])
                    pos = int(sel["position"])
                    src = lookup[(L, pos)]
                    selected_rows.append({
                        "sid": sid,
                        "relation": DISPLAY[gt],
                        "rank": rank,
                        "source_layer": L,
                        "position": pos,
                        "token": sel["token"],
                        "category": sel["category"],
                        "broad_category": sel["broad_category"],
                        "mediation": float(src["mediation"]),
                        "delta_h_norm": float(src["delta_h_norm"]),
                        "spatial_parallel_norm": float(src["spatial_parallel_norm"]),
                        "spatial_parallel_fraction": float(
                            src["spatial_parallel_fraction"]
                        ),
                        "gt_spatial_cos": float(src["gt_spatial_cos"]),
                    })

                # -----------------------------------------------------
                # Same positions, different intervention content.
                # -----------------------------------------------------
                for condition in conditions:
                    token_specs, vector_stats = selected_to_specs(
                        selected_export=selected_export,
                        rows_by_layer=rows_by_layer,
                        spatial_codes=spatial_codes,
                        condition=condition,
                        norm_match_cap=a.norm_match_cap,
                    )

                    for vr in vector_stats:
                        intervention_vector_rows.append({
                            "sid": sid,
                            "relation": DISPLAY[gt],
                            **vr,
                        })

                    edited = run_generation_analysis(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        gt=gt,
                        spatial_codes=spatial_codes,
                        spatial_layers=spatial_layers,
                        target_layers=target_layers,
                        writers_for_relation=writers_r,
                        subject_positions=spos,
                        reference_positions=rpos,
                        max_new_tokens=a.max_new_tokens,
                        token_specs=token_specs,
                        token_alpha=a.alpha,
                    )

                    condition_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "condition": condition,
                        "prediction": DISPLAY.get(
                            edited["prediction"], edited["prediction"]
                        ),
                        "correct": edited["correct"],
                        "text": edited["text"],
                        "n_selected": len(selected_export),
                    })

                    for L, info in edited["spatial"].items():
                        trajectory_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "baseline_correct": baseline["correct"],
                            "condition": condition,
                            "signal_type": "spatial",
                            "layer": L,
                            "spatial_pred": DISPLAY.get(
                                info["pred"], info["pred"]
                            ),
                            "spatial_correct": info["pred"] == gt,
                            "gt_score": info["gt_score"],
                            "margin": info["margin"],
                            **{
                                f"score_{DISPLAY[r]}": info["scores"][r]
                                for r in REL
                            },
                        })

                    for T, info in edited["writer"].items():
                        trajectory_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "baseline_correct": baseline["correct"],
                            "condition": condition,
                            "signal_type": "writer",
                            "layer": T,
                            "writer_projection": info["projection"],
                            "writer_cosine": info["cosine"],
                        })

            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # 3) Save.
        # -------------------------------------------------------------
        behavior_summary = summarize_behavior(
            baseline_rows, condition_rows
        )
        trajectory_summary = summarize_trajectory(
            trajectory_rows
        )
        candidate_summary = summarize_candidate_relation(
            candidate_rows, selected_rows
        )

        write_csv(outdir / "baseline_generation.csv", baseline_rows)
        write_csv(outdir / "generation_conditions.csv", condition_rows)
        write_csv(outdir / "behavior_summary.csv", behavior_summary)
        write_csv(outdir / "candidate_spatial_causal.csv", candidate_rows)
        write_csv(outdir / "selected_spatial_causal.csv", selected_rows)
        write_csv(outdir / "candidate_summary.csv", candidate_summary)
        write_csv(outdir / "trajectory_per_sample.csv", trajectory_rows)
        write_csv(outdir / "trajectory_summary.csv", trajectory_summary)
        write_csv(
            outdir / "intervention_vector_stats.csv",
            intervention_vector_rows,
        )

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "eval_scope": a.eval_scope,
            "calibration_N": len(train),
            "eval_N": len(test),
            "calibration_eval_overlap": overlap,
            "source_layers": source_layers,
            "spatial_readout_layers": spatial_layers,
            "target_layers": target_layers,
            "k": a.k,
            "alpha": a.alpha,
            "conditions": conditions,
            "norm_match_cap": a.norm_match_cap,
            "writer_mode": a.writer_mode,
            "explicit_spatial_code": (
                "layerwise four-relation centered centroids of "
                "mean(subject hidden)-mean(reference hidden)"
            ),
            "explicit_spatial_subspace": (
                "SVD row-space of four centered relation centroids; rank <= 3"
            ),
            "selection_objective": (
                "sum_T <h_real_T,last, normalized relation-conditioned late writer>"
            ),
            "mediation_score": (
                "(h_real_source-h_gray_source) dot grad(selection_objective)"
            ),
            "selection_strategy": "positive global_unique Top-K",
            "oracle_relation_used_for_selection": True,
            "important_caveat": (
                "orthogonal = orthogonal only to the measured explicit "
                "object-pair centroid subspace; it does not imply globally non-spatial"
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        # -------------------------------------------------------------
        # Console report.
        # -------------------------------------------------------------
        print("\n" + "=" * 132)
        print("BEHAVIOR: SAME SELECTED POSITIONS, DIFFERENT VECTOR CONTENT")
        print("=" * 132)

        baseline_acc = safe_mean(float(r["correct"]) for r in baseline_rows)
        print(f"Baseline: N={len(baseline_rows)} acc={baseline_acc:.4f}")
        for r in behavior_summary:
            print(
                f"{r['condition']:<34s} "
                f"acc={float(r['edited_acc']):.4f} "
                f"gain={float(r['delta_acc']):+.4f} "
                f"W2C/C2W={int(r['W2C'])}/{int(r['C2W'])} "
                f"net={int(r['net']):+d}"
            )

        print("\n" + "=" * 132)
        print("CAUSAL ATTRIBUTION x EXPLICIT SPATIAL SUBSPACE")
        print("=" * 132)
        for r in candidate_summary:
            print(
                f"{r['subset']:<18s} N={int(r['N']):>7d} "
                f"mean spatial fraction={float(r['mean_spatial_parallel_fraction']):.4f} "
                f"corr(M, frac)={float(r['corr_mediation_vs_parallel_fraction']):+.4f} "
                f"corr(|M|, frac)={float(r['corr_absmed_vs_parallel_fraction']):+.4f} "
                f"corr(M, gtcos)={float(r['corr_mediation_vs_gt_spatial_cos']):+.4f}"
            )

        print("\n" + "=" * 132)
        print("WHAT TO LOOK FOR")
        print("=" * 132)
        print(
            "1) full >> spatial_parallel_raw: most repair energy is not contained "
            "in the measured explicit object-pair spatial subspace."
        )
        print(
            "2) spatial_parallel_normmatched still weak: not merely a norm/energy "
            "issue; the full high-dimensional direction carries extra causal structure."
        )
        print(
            "3) spatial_orthogonal_raw retains substantial repair: evidence that "
            "decision-supporting computation exists outside this explicit spatial code."
        )
        print(
            "4) full intervention first raises downstream spatial margin, then late "
            "writer projection: candidate evidence for causal construction of spatial code."
        )
        print(
            "5) full raises late writer but barely moves downstream spatial margin: "
            "candidate routing/readout-support states rather than explicit spatial content."
        )
        print(
            "6) High-M selected states have low spatial_parallel_fraction / weak "
            "M-vs-spatial correlation: explicit spatialness and decision leverage dissociate."
        )
        print("\nSaved:", outdir)

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
