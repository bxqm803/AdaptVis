#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_qwen_token_to_spatial_formation_v1.py

Question
--------
Do the middle-layer tokens that causally support the final spatial decision also
CAUSE a later explicit spatial representation to form?

This script compares two attribution targets for the SAME source token state:

    M_writer(L,p)
      = Delta h(L,p)^T * d J_writer / d h(L,p)

    M_spatial(L,p -> T)
      = Delta h(L,p)^T * d J_spatial(T) / d h(L,p)

where:
    Delta h = h_real - h_gray

J_writer is the current late decision/writer objective.
J_spatial(T) is a later explicit object-pair spatial readout:

    r_T = mean(h_subject,T) - mean(h_reference,T)

A four-relation centroid code is learned on the calibration split, and

    J_spatial(T)
      = cosine(r_T, d_GT,T) - mean_{r != GT} cosine(r_T, d_r,T)

Thus a source token can be:

    high M_spatial, high M_writer
        candidate "spatial constructor / pathway" state

    low M_spatial, high M_writer
        candidate decision-support / routing state that does not strongly
        operate through THIS measured explicit spatial readout

    high M_spatial, low M_writer
        state that helps form explicit spatial representation but has weak
        late-decision leverage

IMPORTANT
---------
M_spatial is FIRST-ORDER attribution, not causal proof.

For causal validation, the script also takes writer-selected Top-K states and
splits them into:
    - writer_high_spatial_high
    - writer_high_spatial_low

It amplifies their ORIGINAL Real-Gray Delta h and performs a new forward pass,
measuring the actual change in downstream spatial margin and late writer
projection.

By default the spatial target layers are later than every source layer:
    sources: 20..26
    spatial targets: 27,28,30,32
so the measured spatial change is downstream, not a same-layer edit artifact.

This script reuses the current repo infrastructure:
    analyze_coco_centroid_generation_step1_v4.py
    eval_coco_multilayer_relation_trajectory_repair_v1.py
    eval_qwen_dynamic_k24_all440_v1.py

Recommended first run
---------------------
python analyze_qwen_token_to_spatial_formation_v1.py \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --spatial-target-layers 27,28,30,32 \
  --writer-target-layers 32,34,35 \
  --writer-k 36 \
  --subgroup-k 12 \
  --alpha 1.0 \
  --eval-scope test \
  --eval-max-samples 80 \
  --output-dir outputs/token_to_spatial_formation_v1_n80 \
  --overwrite

If you specifically want early formation (e.g. source L20-22 -> spatial L23-28):
python analyze_qwen_token_to_spatial_formation_v1.py \
  --source-layers 20,21,22 \
  --spatial-target-layers 23,24,25,26,27,28 \
  --writer-target-layers 32,34,35 \
  --writer-k 18 \
  --subgroup-k 6 \
  --eval-scope test \
  --eval-max-samples 80 \
  --output-dir outputs/token_to_spatial_formation_early_v1 \
  --overwrite
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
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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
# Generic helpers
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
        "--spatial-target-layers",
        default="27,28,30,32",
        help=(
            "Layers where downstream explicit object-pair spatial representation "
            "is measured. For the cleanest causal-formation test, keep every target "
            "strictly later than max(source-layers)."
        ),
    )
    p.add_argument("--writer-target-layers", default="32,34,35")

    p.add_argument("--writer-k", type=int, default=36)
    p.add_argument(
        "--subgroup-k",
        type=int,
        default=12,
        help=(
            "Within writer Top-K, size of high-spatial and low-spatial groups "
            "used for actual forward intervention validation."
        ),
    )
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--eval-scope", default="test", choices=["test", "all_data"])

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--skip-group-validation",
        action="store_true",
        help="Only compute attribution matrices; skip extra edited forward passes.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted(
        {
            int(x.strip().upper().replace("L", ""))
            for x in str(s).split(",")
            if x.strip()
        }
    )


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    arr = np.asarray([float(x) for x in xs], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_median(xs):
    arr = np.asarray([float(x) for x in xs], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def safe_corr(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or float(x.std()) < EPS or float(y.std()) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v / n).astype(np.float32)


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def mean_positions_np(H: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    good = [int(p) for p in positions if 0 <= int(p) < H.shape[0]]
    if not good:
        raise RuntimeError("No valid object-token positions")
    return H[good].mean(axis=0).astype(np.float32)


def pair_np(H, subject_positions, reference_positions):
    return (
        mean_positions_np(H, subject_positions)
        - mean_positions_np(H, reference_positions)
    ).astype(np.float32)


def pair_torch(H, subject_positions, reference_positions):
    """
    H: [batch=1, seq, d] or [seq, d]
    returns differentiable [d] object-pair residual.
    """
    if H.ndim == 3:
        H = H[0]
    spos = [p for p in subject_positions if 0 <= int(p) < H.shape[0]]
    rpos = [p for p in reference_positions if 0 <= int(p) < H.shape[0]]
    if not spos or not rpos:
        raise RuntimeError("No valid object-token positions in differentiable state")
    sidx = torch.as_tensor(spos, device=H.device, dtype=torch.long)
    ridx = torch.as_tensor(rpos, device=H.device, dtype=torch.long)
    return H.index_select(0, sidx).mean(0) - H.index_select(0, ridx).mean(0)


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
# Learn the explicit relation representation at target layers
# ---------------------------------------------------------------------

def learn_spatial_codes(pair_by_sid, train_meta, layers):
    """
    d_{T,r} = relation centroid - common centroid
    Spatial readout uses cosine to these centered relation centroids.
    """
    codes = {}
    geometry = []

    for T in layers:
        means = {}
        for r in REL:
            xs = [
                pair_by_sid[int(m["sid"])][T]
                for m in train_meta
                if m["gt"] == r and int(m["sid"]) in pair_by_sid
            ]
            if not xs:
                raise RuntimeError(f"No calibration spatial states for {r} at L{T}")
            means[r] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        common = np.mean(np.stack([means[r] for r in REL]), axis=0).astype(np.float32)
        centered = {r: (means[r] - common).astype(np.float32) for r in REL}
        dirs = {r: normalize_np(centered[r]) for r in REL}
        codes[T] = {"means": means, "common": common, "centered": centered, "dirs": dirs}

        A = np.stack([centered[r] for r in REL], axis=0).astype(np.float64)
        _u, s, _vt = np.linalg.svd(A, full_matrices=False)
        energy = (s ** 2) / max(float(np.sum(s ** 2)), EPS)

        for r in REL:
            geometry.append({
                "target_layer": T,
                "relation": DISPLAY[r],
                "centered_norm": float(np.linalg.norm(centered[r])),
                "cos_left": cosine_np(centered[r], centered["left"]),
                "cos_right": cosine_np(centered[r], centered["right"]),
                "cos_on": cosine_np(centered[r], centered["above"]),
                "cos_under": cosine_np(centered[r], centered["below"]),
                "sv1": float(s[0]) if len(s) > 0 else float("nan"),
                "sv2": float(s[1]) if len(s) > 1 else float("nan"),
                "sv3": float(s[2]) if len(s) > 2 else float("nan"),
                "sv1_energy": float(energy[0]) if len(energy) > 0 else float("nan"),
                "sv12_energy": float(np.sum(energy[:2])) if len(energy) > 1 else float("nan"),
                "sv123_energy": float(np.sum(energy[:3])) if len(energy) > 2 else float("nan"),
            })

    return codes, geometry


def spatial_scores_np(pair_vec, code):
    return {
        r: cosine_np(pair_vec, code["centered"][r])
        for r in REL
    }


def spatial_objective_np(pair_vec, code, gt):
    """
    Same scalar semantics as the differentiable objective:
        GT cosine - mean other cosines
    """
    scores = spatial_scores_np(pair_vec, code)
    gt_score = float(scores[gt])
    other = [float(scores[r]) for r in REL if r != gt]
    return gt_score - float(np.mean(other)), scores


def spatial_objective_torch(pair_vec, code, gt):
    """
    Differentiable explicit-spatial objective.
    Directions are fixed from the calibration split.
    """
    v = pair_vec.float()
    v = v / (torch.linalg.vector_norm(v) + 1e-8)

    vals = {}
    for r in REL:
        d = torch.as_tensor(
            code["dirs"][r], device=v.device, dtype=torch.float32
        )
        vals[r] = torch.dot(v, d)

    gt_score = vals[gt]
    others = torch.stack([vals[r] for r in REL if r != gt]).mean()
    return gt_score - others


# ---------------------------------------------------------------------
# Forward recorder for actual intervention validation
# ---------------------------------------------------------------------

class DownstreamRecorder:
    def __init__(
        self,
        decoder_layers,
        prompt_len,
        spatial_targets,
        writer_targets,
        subject_positions,
        reference_positions,
    ):
        self.prompt_len = int(prompt_len)
        self.spatial_targets = set(map(int, spatial_targets))
        self.writer_targets = set(map(int, writer_targets))
        self.subject_positions = list(map(int, subject_positions))
        self.reference_positions = list(map(int, reference_positions))
        self.pairs = {}
        self.last = {}
        self.handles = []

        for L in sorted(self.spatial_targets | self.writer_targets):
            self.handles.append(
                decoder_layers[L].register_forward_hook(self._hook(L))
            )

    def _hook(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) != self.prompt_len:
                return out
            H = h[0].detach().float().cpu().numpy().astype(np.float32)

            if L in self.spatial_targets and L not in self.pairs:
                self.pairs[L] = pair_np(
                    H, self.subject_positions, self.reference_positions
                )

            if L in self.writer_targets and L not in self.last:
                self.last[L] = H[-1].copy()

            return out
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def run_forward_readouts(
    model,
    decoder_layers,
    batch,
    spatial_targets,
    writer_targets,
    spatial_codes,
    writers_for_relation,
    gt,
    subject_positions,
    reference_positions,
    token_specs=None,
    alpha=1.0,
):
    """
    Actual causal forward validation, no generation.
    Editors are registered before the recorder so the recorder sees edited states.
    """
    prompt_len = int(batch["input_ids"].shape[1])
    editor = None
    rec = None

    try:
        if token_specs:
            editor = dyn.TokenDeltaEditor(
                decoder_layers, token_specs, float(alpha), prompt_len
            )

        rec = DownstreamRecorder(
            decoder_layers,
            prompt_len,
            spatial_targets,
            writer_targets,
            subject_positions,
            reference_positions,
        )

        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)

        spatial = {}
        for T in spatial_targets:
            if T not in rec.pairs:
                continue
            obj, scores = spatial_objective_np(
                rec.pairs[T], spatial_codes[T], gt
            )
            pred = max(
                REL,
                key=lambda r: -1e9 if not np.isfinite(scores[r]) else scores[r],
            )
            spatial[T] = {
                "objective": float(obj),
                "scores": scores,
                "pred": pred,
                "correct": pred == gt,
            }

        writer = {}
        for T in writer_targets:
            if T not in rec.last:
                continue
            w = writers_for_relation[T]
            writer[T] = {
                "projection": float(np.dot(rec.last[T], normalize_np(w))),
                "cosine": cosine_np(rec.last[T], w),
            }

        return {
            "spatial": spatial,
            "writer": writer,
            "edit_applied": dict(editor.applied) if editor is not None else {},
        }

    finally:
        if rec is not None:
            rec.close()
        if editor is not None:
            editor.close()


# ---------------------------------------------------------------------
# Generic global-unique selection
# ---------------------------------------------------------------------

def global_unique_top(rows, score_key, k, positive_only=True):
    vals = []
    for r in rows:
        s = float(r.get(score_key, float("nan")))
        if not np.isfinite(s):
            continue
        if positive_only and s <= 0:
            continue
        vals.append(r)

    best_by_pos = {}
    for r in vals:
        pos = int(r["position"])
        if pos not in best_by_pos or float(r[score_key]) > float(best_by_pos[pos][score_key]):
            best_by_pos[pos] = r

    out = sorted(
        best_by_pos.values(),
        key=lambda r: float(r[score_key]),
        reverse=True,
    )[: int(k)]
    return out


def rows_to_specs(rows):
    specs = defaultdict(list)
    for r in rows:
        specs[int(r["source_layer"])].append(
            (int(r["position"]), np.asarray(r["delta_h"], np.float32))
        )
    return dict(specs)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    spatial_targets = parse_ints(a.spatial_target_layers)
    writer_targets = parse_ints(a.writer_target_layers)

    if not source_layers or not spatial_targets or not writer_targets:
        raise ValueError("source/spatial/writer layer lists must be non-empty")

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

        all_layers = sorted(set(source_layers + spatial_targets + writer_targets))
        for L in all_layers:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid; model has L0...L{n_layers - 1}")

        cut = min(source_layers)
        graph_layers = all_layers
        device = torch.device(a.device)

        strictly_downstream = all(T > max(source_layers) for T in spatial_targets)

        print("=" * 136)
        print("MIDDLE TOKEN -> EXPLICIT SPATIAL REPRESENTATION -> LATE DECISION")
        print("=" * 136)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(
            f"calibration N={len(train)} | eval_scope={a.eval_scope} "
            f"| eval N={len(test)} | overlap={overlap}"
        )
        print(f"source layers={source_layers}")
        print(f"spatial target layers={spatial_targets}")
        print(f"writer target layers={writer_targets}")
        print(f"writer K={a.writer_k} | subgroup K={a.subgroup_k} | alpha={a.alpha}")
        print(f"all spatial targets strictly downstream of all sources: {strictly_downstream}")
        print("GT relation is used for BOTH writer and spatial objectives (oracle analysis).")
        print()

        # -------------------------------------------------------------
        # 1) Calibration:
        #    A. late writer directions
        #    B. explicit object-pair relation code at spatial targets
        # -------------------------------------------------------------
        q_by_sid = {}
        pair_by_sid = {}

        calib_capture_layers = sorted(set(spatial_targets + writer_targets))

        for m in tqdm(train, desc="CALIBRATE writer + explicit spatial code"):
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

                hr = dyn.capture_cpu(
                    model, decoder_layers, rb, calib_capture_layers
                )
                hg = dyn.capture_cpu(
                    model, decoder_layers, gb, writer_targets
                )

                q_by_sid[sid] = {
                    T: (hr[T][0, -1] - hg[T][0, -1]).astype(np.float32)
                    for T in writer_targets
                }
                pair_by_sid[sid] = {
                    T: pair_np(hr[T][0], spos, rpos)
                    for T in spatial_targets
                }

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()

        writers, writer_geometry = dyn.learn_writers(
            train, q_by_sid, writer_targets, a.writer_mode
        )
        spatial_codes, spatial_geometry = learn_spatial_codes(
            pair_by_sid, train, spatial_targets
        )

        write_csv(outdir / "writer_geometry.csv", writer_geometry)
        write_csv(outdir / "spatial_code_geometry.csv", spatial_geometry)

        # -------------------------------------------------------------
        # 2) Evaluation.
        # -------------------------------------------------------------
        all_candidate_rows = []
        writer_selected_rows = []
        overlap_rows = []
        readout_rows = []
        group_validation_rows = []

        for m in tqdm(test, desc="EVAL token -> spatial formation"):
            sid = int(m["sid"])
            gt = m["gt"]
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

                # Gray source states define the natural image-conditioned delta.
                hgray = dyn.capture_cpu(
                    model, decoder_layers, gb, source_layers
                )

                # One differentiable real forward.
                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model, decoder_layers, rb, graph_layers, cut
                    )

                    # ---------------- writer objective ----------------
                    writer_terms = []
                    for T in writer_targets:
                        s_hat = torch.as_tensor(
                            normalize_np(writers[T][gt]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        writer_terms.append(
                            torch.dot(cap.states[T][0, -1].float(), s_hat)
                        )
                    J_writer = torch.stack(writer_terms).sum()

                    grad_writer = torch.autograd.grad(
                        J_writer,
                        [cap.states[S] for S in source_layers],
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )

                    # ---------------- spatial objectives --------------
                    spatial_objectives = {}
                    spatial_grads = {}

                    for idx_T, T in enumerate(spatial_targets):
                        pairT = pair_torch(cap.states[T], spos, rpos)
                        Jsp = spatial_objective_torch(
                            pairT, spatial_codes[T], gt
                        )
                        spatial_objectives[T] = float(
                            Jsp.detach().float().cpu().item()
                        )

                        valid_sources = [S for S in source_layers if S < T]
                        if not valid_sources:
                            spatial_grads[T] = {}
                            continue

                        # Retain graph except after the last spatial target.
                        retain = idx_T < (len(spatial_targets) - 1)
                        gs = torch.autograd.grad(
                            Jsp,
                            [cap.states[S] for S in valid_sources],
                            retain_graph=retain,
                            create_graph=False,
                            allow_unused=True,
                        )
                        spatial_grads[T] = {
                            S: g for S, g in zip(valid_sources, gs)
                        }

                    # ---------------- held-out readout quality --------
                    for T in spatial_targets:
                        H = (
                            cap.states[T][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        pvec = pair_np(H, spos, rpos)
                        obj, scores = spatial_objective_np(
                            pvec, spatial_codes[T], gt
                        )
                        pred = max(
                            REL,
                            key=lambda r: -1e9 if not np.isfinite(scores[r]) else scores[r],
                        )
                        readout_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "target_layer": T,
                            "spatial_objective": obj,
                            "spatial_prediction": DISPLAY[pred],
                            "spatial_correct": pred == gt,
                            **{
                                f"score_{DISPLAY[r]}": scores[r]
                                for r in REL
                            },
                        })

                    # ---------------- candidate scores ----------------
                    rows_this_sample = []

                    for iS, S in enumerate(source_layers):
                        Hreal = (
                            cap.states[S][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        Hgray = hgray[S][0].astype(np.float32)

                        gw = grad_writer[iS]
                        if gw is None:
                            Gw = np.zeros_like(Hreal, dtype=np.float32)
                        else:
                            Gw = (
                                gw[0]
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
                            Gw.shape[0],
                        )

                        # source last token excluded, matching current K36.
                        for pos in range(max(0, npos - 1)):
                            delta = (Hreal[pos] - Hgray[pos]).astype(np.float32)
                            mw = float(np.dot(delta, Gw[pos]))

                            row = {
                                "sid": sid,
                                "gt": DISPLAY[gt],
                                "source_layer": S,
                                "position": pos,
                                "token_id": int(ids[pos]),
                                "token": str(toks[pos]).replace("\n", "\\n"),
                                "category": cats[pos],
                                "broad_category": dyn.broad_category(cats[pos]),
                                "writer_m": mw,
                                # dyn.make_specs expects these legacy keys:
                                "mediation": mw,
                                "delta_h_norm": float(np.linalg.norm(delta)),
                                "grad_norm": float(np.linalg.norm(Gw[pos])),
                                "delta_h": delta,
                            }

                            valid_spatial_ms = []
                            positive_spatial_ms = []

                            for T in spatial_targets:
                                key = f"spatial_m_L{T}"
                                gT = spatial_grads.get(T, {}).get(S)
                                if gT is None:
                                    row[key] = float("nan")
                                    continue

                                Gt_np = (
                                    gT[0, pos]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )
                                ms = float(np.dot(delta, Gt_np))
                                row[key] = ms
                                valid_spatial_ms.append(ms)
                                positive_spatial_ms.append(max(ms, 0.0))

                            row["spatial_m_mean"] = (
                                safe_mean(valid_spatial_ms)
                                if valid_spatial_ms else float("nan")
                            )
                            row["spatial_m_max"] = (
                                max(valid_spatial_ms)
                                if valid_spatial_ms else float("nan")
                            )
                            row["spatial_m_positive_sum"] = (
                                float(sum(positive_spatial_ms))
                                if positive_spatial_ms else float("nan")
                            )
                            rows_this_sample.append(row)

                cap.close()
                cap = None

                # -----------------------------------------------------
                # Normalize each spatial target inside this sample.
                #
                # spatial_mass_score:
                #   sum_T positive M_spatial(token,T) /
                #         total positive M_spatial(all candidates,T)
                #
                # This makes different target-layer gradient scales comparable.
                # -----------------------------------------------------
                pos_mass_by_T = {}
                for T in spatial_targets:
                    vals = [
                        max(float(r[f"spatial_m_L{T}"]), 0.0)
                        for r in rows_this_sample
                        if np.isfinite(float(r[f"spatial_m_L{T}"]))
                    ]
                    pos_mass_by_T[T] = float(sum(vals))

                for r in rows_this_sample:
                    mass_score = 0.0
                    nvalid = 0
                    for T in spatial_targets:
                        v = float(r[f"spatial_m_L{T}"])
                        denom = pos_mass_by_T[T]
                        if not np.isfinite(v):
                            continue
                        nvalid += 1
                        if denom > EPS and v > 0:
                            mass_score += v / denom
                    r["spatial_mass_score"] = (
                        mass_score if nvalid > 0 else float("nan")
                    )

                # Writer oracle Top-K = current causal-support set.
                rows_by_layer = {
                    S: [r for r in rows_this_sample if int(r["source_layer"]) == S]
                    for S in source_layers
                }
                _specs, selected_export = dyn.make_specs(
                    tuple(source_layers),
                    rows_by_layer,
                    mode="positive",
                    k=a.writer_k,
                    sid=sid,
                    seed=a.seed,
                    strategy="global_unique",
                )

                lookup = {
                    (int(r["source_layer"]), int(r["position"])): r
                    for r in rows_this_sample
                }
                writer_selected = [
                    lookup[(int(s["source_layer"]), int(s["position"]))]
                    for s in selected_export
                ]

                selected_keys = {
                    (int(r["source_layer"]), int(r["position"]))
                    for r in writer_selected
                }

                # Spatial Top-K of same size, using normalized spatial mass.
                spatial_selected = global_unique_top(
                    rows_this_sample,
                    "spatial_mass_score",
                    a.writer_k,
                    positive_only=True,
                )
                spatial_keys = {
                    (int(r["source_layer"]), int(r["position"]))
                    for r in spatial_selected
                }

                inter = selected_keys & spatial_keys
                union = selected_keys | spatial_keys

                overlap_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "writer_k_actual": len(selected_keys),
                    "spatial_k_actual": len(spatial_keys),
                    "intersection": len(inter),
                    "jaccard": (
                        len(inter) / len(union) if union else float("nan")
                    ),
                    "writer_recall_by_spatial_topk": (
                        len(inter) / len(selected_keys)
                        if selected_keys else float("nan")
                    ),
                })

                for r in rows_this_sample:
                    export = {k: v for k, v in r.items() if k != "delta_h"}
                    export["writer_selected"] = (
                        (int(r["source_layer"]), int(r["position"])) in selected_keys
                    )
                    export["spatial_selected"] = (
                        (int(r["source_layer"]), int(r["position"])) in spatial_keys
                    )
                    all_candidate_rows.append(export)

                for rank, r in enumerate(
                    sorted(writer_selected, key=lambda x: x["writer_m"], reverse=True),
                    1,
                ):
                    writer_selected_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "writer_rank": rank,
                        **{k: v for k, v in r.items() if k not in {"delta_h", "mediation"}},
                    })

                # -----------------------------------------------------
                # 3) Actual causal validation:
                #    same writer-selected states, but split by how much
                #    they are predicted to CREATE later explicit spatial code.
                # -----------------------------------------------------
                if not a.skip_group_validation and writer_selected:
                    ksub = min(a.subgroup_k, len(writer_selected))

                    high_spatial = sorted(
                        writer_selected,
                        key=lambda r: float(r["spatial_mass_score"])
                        if np.isfinite(float(r["spatial_mass_score"])) else -1e30,
                        reverse=True,
                    )[:ksub]

                    low_spatial = sorted(
                        writer_selected,
                        key=lambda r: float(r["spatial_mass_score"])
                        if np.isfinite(float(r["spatial_mass_score"])) else 1e30,
                    )[:ksub]

                    top_writer_same_size = sorted(
                        writer_selected,
                        key=lambda r: float(r["writer_m"]),
                        reverse=True,
                    )[:ksub]

                    groups = {
                        "baseline": [],
                        "writer_topK_full": writer_selected,
                        "writer_topK_top_writer_subgroup": top_writer_same_size,
                        "writer_high_spatial_high": high_spatial,
                        "writer_high_spatial_low": low_spatial,
                        "spatial_topK": spatial_selected[:ksub],
                    }

                    writers_r = {
                        T: writers[T][gt]
                        for T in writer_targets
                    }

                    # Baseline once.
                    baseline_read = run_forward_readouts(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        spatial_targets=spatial_targets,
                        writer_targets=writer_targets,
                        spatial_codes=spatial_codes,
                        writers_for_relation=writers_r,
                        gt=gt,
                        subject_positions=spos,
                        reference_positions=rpos,
                        token_specs=None,
                        alpha=a.alpha,
                    )

                    base_sp = {
                        T: baseline_read["spatial"][T]["objective"]
                        for T in baseline_read["spatial"]
                    }
                    base_wr = {
                        T: baseline_read["writer"][T]["projection"]
                        for T in baseline_read["writer"]
                    }

                    # Save baseline rows.
                    for T, info in baseline_read["spatial"].items():
                        group_validation_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "group": "baseline",
                            "group_size": 0,
                            "signal": "spatial",
                            "target_layer": T,
                            "value": info["objective"],
                            "delta_vs_baseline": 0.0,
                            "spatial_correct": info["correct"],
                        })
                    for T, info in baseline_read["writer"].items():
                        group_validation_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "group": "baseline",
                            "group_size": 0,
                            "signal": "writer",
                            "target_layer": T,
                            "value": info["projection"],
                            "delta_vs_baseline": 0.0,
                        })

                    for gname, grows in groups.items():
                        if gname == "baseline":
                            continue
                        edit_specs = rows_to_specs(grows)
                        edited = run_forward_readouts(
                            model=model,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            spatial_targets=spatial_targets,
                            writer_targets=writer_targets,
                            spatial_codes=spatial_codes,
                            writers_for_relation=writers_r,
                            gt=gt,
                            subject_positions=spos,
                            reference_positions=rpos,
                            token_specs=edit_specs,
                            alpha=a.alpha,
                        )

                        for T, info in edited["spatial"].items():
                            group_validation_rows.append({
                                "sid": sid,
                                "gt": DISPLAY[gt],
                                "group": gname,
                                "group_size": len(grows),
                                "signal": "spatial",
                                "target_layer": T,
                                "value": info["objective"],
                                "delta_vs_baseline": (
                                    info["objective"] - base_sp[T]
                                ),
                                "spatial_correct": info["correct"],
                            })

                        for T, info in edited["writer"].items():
                            group_validation_rows.append({
                                "sid": sid,
                                "gt": DISPLAY[gt],
                                "group": gname,
                                "group_size": len(grows),
                                "signal": "writer",
                                "target_layer": T,
                                "value": info["projection"],
                                "delta_vs_baseline": (
                                    info["projection"] - base_wr[T]
                                ),
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
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # 4) Summaries
        # -------------------------------------------------------------
        readout_summary = []
        for T in spatial_targets:
            rowsT = [r for r in readout_rows if int(r["target_layer"]) == T]
            readout_summary.append({
                "target_layer": T,
                "N": len(rowsT),
                "heldout_readout_acc": safe_mean(
                    float(r["spatial_correct"]) for r in rowsT
                ),
                "mean_spatial_objective": safe_mean(
                    r["spatial_objective"] for r in rowsT
                ),
            })

        attribution_summary = []
        for T in spatial_targets:
            valid = [
                r for r in all_candidate_rows
                if np.isfinite(float(r[f"spatial_m_L{T}"]))
            ]
            selected = [r for r in valid if bool(r["writer_selected"])]

            attribution_summary.append({
                "spatial_target_layer": T,
                "all_N": len(valid),
                "all_corr_writerM_spatialM": safe_corr(
                    [r["writer_m"] for r in valid],
                    [r[f"spatial_m_L{T}"] for r in valid],
                ),
                "writer_selected_N": len(selected),
                "writer_selected_corr_writerM_spatialM": safe_corr(
                    [r["writer_m"] for r in selected],
                    [r[f"spatial_m_L{T}"] for r in selected],
                ),
                "writer_selected_positive_spatial_fraction": safe_mean(
                    float(float(r[f"spatial_m_L{T}"]) > 0)
                    for r in selected
                ),
                "writer_selected_mean_spatialM": safe_mean(
                    r[f"spatial_m_L{T}"] for r in selected
                ),
            })

        overall_selected = [
            r for r in all_candidate_rows if bool(r["writer_selected"])
        ]
        overall_candidate_summary = [{
            "subset": "writer_selected",
            "N": len(overall_selected),
            "corr_writerM_spatial_mass_score": safe_corr(
                [r["writer_m"] for r in overall_selected],
                [r["spatial_mass_score"] for r in overall_selected],
            ),
            "mean_spatial_mass_score": safe_mean(
                r["spatial_mass_score"] for r in overall_selected
            ),
            "median_spatial_mass_score": safe_median(
                r["spatial_mass_score"] for r in overall_selected
            ),
            "mean_topK_jaccard_writer_vs_spatial": safe_mean(
                r["jaccard"] for r in overlap_rows
            ),
            "mean_writer_recall_by_spatial_topK": safe_mean(
                r["writer_recall_by_spatial_topk"] for r in overlap_rows
            ),
        }]

        group_summary = []
        if group_validation_rows:
            keys = sorted({
                (r["group"], r["signal"], int(r["target_layer"]))
                for r in group_validation_rows
            })
            for g, sig, T in keys:
                rr = [
                    r for r in group_validation_rows
                    if r["group"] == g
                    and r["signal"] == sig
                    and int(r["target_layer"]) == T
                ]
                row = {
                    "group": g,
                    "signal": sig,
                    "target_layer": T,
                    "N": len(rr),
                    "mean_group_size": safe_mean(r["group_size"] for r in rr),
                    "mean_value": safe_mean(r["value"] for r in rr),
                    "mean_delta_vs_baseline": safe_mean(
                        r["delta_vs_baseline"] for r in rr
                    ),
                }
                if sig == "spatial":
                    row["spatial_readout_acc"] = safe_mean(
                        float(r["spatial_correct"]) for r in rr
                    )
                group_summary.append(row)

        # -------------------------------------------------------------
        # 5) Save
        # -------------------------------------------------------------
        write_csv(outdir / "spatial_readout_per_sample.csv", readout_rows)
        write_csv(outdir / "spatial_readout_summary.csv", readout_summary)
        write_csv(outdir / "all_token_writer_vs_spatial.csv", all_candidate_rows)
        write_csv(outdir / "writer_topk_with_spatial_scores.csv", writer_selected_rows)
        write_csv(outdir / "writer_spatial_topk_overlap.csv", overlap_rows)
        write_csv(outdir / "attribution_summary_by_target.csv", attribution_summary)
        write_csv(outdir / "overall_selected_summary.csv", overall_candidate_summary)
        write_csv(outdir / "group_validation_per_sample.csv", group_validation_rows)
        write_csv(outdir / "group_validation_summary.csv", group_summary)

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "source_layers": source_layers,
            "spatial_target_layers": spatial_targets,
            "writer_target_layers": writer_targets,
            "writer_k": a.writer_k,
            "subgroup_k": a.subgroup_k,
            "alpha": a.alpha,
            "calibration_N": len(train),
            "eval_N": len(test),
            "eval_scope": a.eval_scope,
            "calibration_eval_overlap": overlap,
            "strictly_downstream_spatial_targets": strictly_downstream,
            "M_writer": "Delta h dot grad(J_writer)",
            "M_spatial": (
                "Delta h dot grad(J_spatial_T), only defined for source layer < target layer"
            ),
            "J_spatial": (
                "cos(object_pair_residual, GT centered relation centroid) "
                "- mean cosine to the other three relation centroids"
            ),
            "important_interpretation": (
                "High M_spatial means the natural Real-Gray change at this source "
                "state is locally aligned with increasing a LATER measured explicit "
                "spatial readout. It does not prove mediation until intervention/blocking."
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        # -------------------------------------------------------------
        # 6) Console report
        # -------------------------------------------------------------
        print("\n" + "=" * 136)
        print("1) IS THE DOWNSTREAM TARGET ACTUALLY A DECODABLE SPATIAL REPRESENTATION?")
        print("=" * 136)
        for r in readout_summary:
            print(
                f"L{int(r['target_layer']):02d} "
                f"N={int(r['N']):3d} "
                f"readout_acc={float(r['heldout_readout_acc']):.4f} "
                f"mean_obj={float(r['mean_spatial_objective']):+.4f}"
            )

        print("\n" + "=" * 136)
        print("2) DO THE CURRENT WRITER-CAUSAL TOKENS ALSO PROMOTE LATER SPATIAL REPRESENTATION?")
        print("=" * 136)
        for r in attribution_summary:
            print(
                f"spatial target L{int(r['spatial_target_layer']):02d} | "
                f"corr all={float(r['all_corr_writerM_spatialM']):+.4f} | "
                f"corr writerTopK={float(r['writer_selected_corr_writerM_spatialM']):+.4f} | "
                f"TopK spatial-positive={float(r['writer_selected_positive_spatial_fraction']):.3f} | "
                f"TopK mean M_spatial={float(r['writer_selected_mean_spatialM']):+.6f}"
            )

        s = overall_candidate_summary[0]
        print(
            f"\nwriterTopK vs spatialTopK: "
            f"mean Jaccard={float(s['mean_topK_jaccard_writer_vs_spatial']):.4f}, "
            f"writer recall={float(s['mean_writer_recall_by_spatial_topK']):.4f}"
        )
        print(
            f"Within writer-selected states: "
            f"corr(writer M, normalized spatial mass)="
            f"{float(s['corr_writerM_spatial_mass_score']):+.4f}"
        )

        if group_summary:
            print("\n" + "=" * 136)
            print("3) ACTUAL FORWARD VALIDATION: DOES AMPLIFYING THESE STATES CHANGE LATER SPATIAL CODE?")
            print("=" * 136)

            for T in spatial_targets:
                print(f"\nSpatial target L{T}:")
                rowsT = [
                    r for r in group_summary
                    if r["signal"] == "spatial"
                    and int(r["target_layer"]) == T
                ]
                for r in rowsT:
                    print(
                        f"  {r['group']:<34s} "
                        f"size={float(r['mean_group_size']):5.1f} "
                        f"delta_spatial_obj={float(r['mean_delta_vs_baseline']):+.5f} "
                        f"readout_acc={float(r.get('spatial_readout_acc', float('nan'))):.4f}"
                    )

            print("\nLate writer:")
            for T in writer_targets:
                rowsT = [
                    r for r in group_summary
                    if r["signal"] == "writer"
                    and int(r["target_layer"]) == T
                ]
                print(f"  L{T}:")
                for r in rowsT:
                    print(
                        f"    {r['group']:<32s} "
                        f"delta_writer={float(r['mean_delta_vs_baseline']):+.5f}"
                    )

        print("\n" + "=" * 136)
        print("HOW TO READ THE RESULT")
        print("=" * 136)
        print(
            "A) writer Top-K has high positive M_spatial AND its actual amplification "
            "raises later spatial objective -> evidence that part of K36 helps CONSTRUCT "
            "the later explicit spatial representation."
        )
        print(
            "B) writer_high_spatial_high raises spatial objective much more than "
            "writer_high_spatial_low (same group size) -> the split has functional meaning."
        )
        print(
            "C) writer_high_spatial_low barely changes spatial objective but still strongly "
            "raises late writer projection -> candidate routing/integration/decision-support path."
        )
        print(
            "D) writer Top-K strongly raises writer but has little M_spatial / little actual "
            "spatial change -> the measured explicit object-pair spatial code is probably NOT "
            "the main mediator of the current repair effect."
        )
        print(
            "E) High M_spatial alone is not proof of mediation. The next strong test is "
            "source intervention + downstream spatial-component blocking."
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
