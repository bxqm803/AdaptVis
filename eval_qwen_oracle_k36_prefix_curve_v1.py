#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_qwen_oracle_k36_prefix_curve_v1.py

Purpose
-------
Shrink the current oracle K36 causal set by contribution magnitude and measure
how many of the highest-contribution states are actually needed for behavior.

For EACH sample:

1) Compute the current writer-guided score on source layers L20-L26:

       M(L,p) = (h_real(L,p) - h_gray(L,p))^T grad_h J_writer

2) Apply the same current `global_unique` rule:
   - keep only positive-M candidates
   - for each token POSITION, keep the source layer with the largest M
   - sort the remaining candidates by M descending

This creates ONE nested ranked K36 list:

       rank 1 >= rank 2 >= ... >= rank 36

3) Evaluate prefixes of EXACTLY that same K36 ranking:

       K5, K7, K10, K20, K36

   using the existing intervention:

       h <- h + alpha * (h_real - h_gray)

   followed by actual `model.generate()`.

Thus K5 is literally ranks 1..5 of the same K36, K7 is ranks 1..7, etc.
This avoids the ambiguity of independently re-running a Top-K selector.

Outputs
-------
baseline.csv
    Baseline generation result for every sample.

ranked_k36_tokens.csv
    The full per-sample K36 ranking, including:
      rank, layer, position, token, category, M,
      M share within K36, cumulative M share.

generation_by_k.csv
    Actual generation result for every sample and K.

summary.csv
    Baseline -> K5/K7/K10/K20/K36:
      accuracy, gain, W2C, C2W, repair rate, preserve rate,
      changed predictions, mean writer-projection gain.

rank_mass_summary.csv
    Across samples, how much of the K36 positive attribution mass is already
    contained in the first K tokens.

Default run
-----------
The defaults intentionally use `all_data` / all 440 samples so the curve is
directly comparable with the existing full-dataset K36 mechanistic diagnostic.
This DOES overlap the 30% writer-calibration split and is therefore an ORACLE
mechanistic diagnostic, not the final OOF paper number.

Run:
python eval_qwen_oracle_k36_prefix_curve_v1.py \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --target-layers 32,34,35 \
  --ks 5,7,10,20,36 \
  --alpha 1.0 \
  --eval-scope all_data \
  --eval-max-samples 0 \
  --output-dir outputs/oracle_k36_prefix_curve_v1 \
  --overwrite

Quick held-out test:
python eval_qwen_oracle_k36_prefix_curve_v1.py \
  --eval-scope test \
  --eval-max-samples 80 \
  --output-dir outputs/oracle_k36_prefix_curve_v1_n80 \
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
from typing import Dict, List

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
    p.add_argument("--target-layers", default="32,34,35")

    p.add_argument(
        "--ks",
        default="5,7,10,20,36",
        help="Nested prefixes of one per-sample ranked K36.",
    )
    p.add_argument(
        "--max-rank",
        type=int,
        default=36,
        help="Build the ranked oracle set up to this many states.",
    )
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument(
        "--eval-scope",
        default="all_data",
        choices=["test", "all_data"],
        help=(
            "all_data reproduces the current full-dataset oracle diagnostic "
            "but overlaps writer calibration; test is the held-out 70%% split."
        ),
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 means all samples in the chosen eval scope.",
    )

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


def parse_ints(s: str) -> List[int]:
    return sorted({
        int(x.strip().upper().replace("L", ""))
        for x in str(s).split(",") if x.strip()
    })


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


def safe_median(xs):
    vals = np.asarray([float(x) for x in xs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v / n).astype(np.float32)


def build_ranked_global_unique(rows_by_layer, max_rank: int):
    """
    Exact positive global_unique logic:
      1) gather all positive-M (layer, position) candidates
      2) for each POSITION retain only the layer with maximal M
      3) globally sort by M descending
      4) return top max_rank

    Returned rows retain delta_h so prefixes can be edited directly.
    """
    best_by_pos = {}

    for L, rows in rows_by_layer.items():
        for row in rows:
            m = float(row["mediation"])
            if not np.isfinite(m) or m <= 0:
                continue
            rr = dict(row)
            rr["source_layer"] = int(L)
            pos = int(rr["position"])
            cur = best_by_pos.get(pos)
            if cur is None or m > float(cur["mediation"]):
                best_by_pos[pos] = rr

    ranked = sorted(
        best_by_pos.values(),
        key=lambda r: float(r["mediation"]),
        reverse=True,
    )
    return ranked[: int(max_rank)]


def prefix_to_specs(ranked, k):
    use = ranked[: min(int(k), len(ranked))]
    specs = defaultdict(list)
    for r in use:
        specs[int(r["source_layer"])].append(
            (int(r["position"]), np.asarray(r["delta_h"], np.float32))
        )
    return dict(specs), use


def summarize(baseline_rows, generation_rows, ks):
    base = {int(r["sid"]): r for r in baseline_rows}
    out = []

    base_acc = safe_mean(float(r["baseline_correct"]) for r in baseline_rows)
    n_wrong = sum(not bool(r["baseline_correct"]) for r in baseline_rows)
    n_correct = sum(bool(r["baseline_correct"]) for r in baseline_rows)

    for k in ks:
        rows = [r for r in generation_rows if int(r["k"]) == int(k)]
        by_sid = {int(r["sid"]): r for r in rows}
        common = sorted(set(base) & set(by_sid))

        edited_correct = [bool(by_sid[s]["correct"]) for s in common]
        edited_acc = safe_mean(float(x) for x in edited_correct)

        w2c = sum(
            (not bool(base[s]["baseline_correct"])) and bool(by_sid[s]["correct"])
            for s in common
        )
        c2w = sum(
            bool(base[s]["baseline_correct"]) and (not bool(by_sid[s]["correct"]))
            for s in common
        )
        changed = sum(
            str(base[s]["baseline_prediction"]) != str(by_sid[s]["prediction"])
            for s in common
        )

        proj_gain = [
            float(by_sid[s]["mean_projection"]) -
            float(base[s]["baseline_mean_projection"])
            for s in common
        ]

        out.append({
            "k": int(k),
            "N": len(common),
            "baseline_acc": base_acc,
            "edited_acc": edited_acc,
            "delta_acc": edited_acc - base_acc,
            "W2C": int(w2c),
            "C2W": int(c2w),
            "net": int(w2c - c2w),
            "changed": int(changed),
            "wrong_N": int(n_wrong),
            "correct_N": int(n_correct),
            "repair_rate_W2C_over_wrong": (
                w2c / n_wrong if n_wrong else float("nan")
            ),
            "preserve_rate_correct_stays_correct": (
                1.0 - c2w / n_correct if n_correct else float("nan")
            ),
            "mean_writer_projection_gain": safe_mean(proj_gain),
            "mean_selected_tokens": safe_mean(
                by_sid[s]["n_selected"] for s in common
            ),
        })

    return out


def summarize_mass(ranked_rows, ks):
    """
    Per sample ranked rows already contain cumulative share.
    Report mean/median attribution mass captured by first K of K36.
    """
    by_sid = defaultdict(list)
    for r in ranked_rows:
        by_sid[int(r["sid"])].append(r)

    out = []
    for k in ks:
        vals = []
        raw = []
        counts = []
        for sid, rows in by_sid.items():
            rows = sorted(rows, key=lambda r: int(r["rank"]))
            use = [r for r in rows if int(r["rank"]) <= int(k)]
            if not rows:
                continue
            total = sum(float(r["mediation"]) for r in rows)
            part = sum(float(r["mediation"]) for r in use)
            vals.append(part / total if total > EPS else float("nan"))
            raw.append(part)
            counts.append(len(use))
        out.append({
            "k": int(k),
            "N": len(vals),
            "mean_K36_mediation_mass_fraction": safe_mean(vals),
            "median_K36_mediation_mass_fraction": safe_median(vals),
            "mean_raw_mediation_sum": safe_mean(raw),
            "mean_actual_selected": safe_mean(counts),
        })
    return out


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    targets = parse_ints(a.target_layers)
    ks = parse_ints(a.ks)

    if not ks:
        raise ValueError("No K values")
    if max(ks) > int(a.max_rank):
        raise ValueError(
            f"max requested K={max(ks)} exceeds --max-rank={a.max_rank}"
        )
    if int(a.max_rank) <= 0:
        raise ValueError("--max-rank must be positive")

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

    if int(a.eval_max_samples) > 0:
        test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)

    train_sids = {int(x["sid"]) for x in train}
    eval_sids = {int(x["sid"]) for x in test}
    overlap = len(train_sids & eval_sids)

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
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        for L in source_layers + targets:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid; model has L0...L{n_layers-1}")

        cut = min(source_layers)
        graph_layers = sorted(set(source_layers + targets))
        device = torch.device(a.device)

        print("=" * 136)
        print("ORACLE K36 CONTRIBUTION RANK -> NESTED PREFIX BEHAVIOR CURVE")
        print("=" * 136)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(
            f"writer calibration N={len(train)} | eval_scope={a.eval_scope} "
            f"| eval N={len(test)} | calibration/eval overlap={overlap}"
        )
        if overlap:
            print(
                "NOTE: evaluation overlaps writer calibration; this is an oracle "
                "mechanistic diagnostic, not an OOF paper estimate."
            )
        print(f"source layers={source_layers}")
        print(f"writer target layers={targets}")
        print(f"rank max={a.max_rank} | prefixes={ks} | alpha={a.alpha}")
        print("ranking = positive global_unique by writer-aligned M")
        print("K5/K7/K10/K20 are strict prefixes of the SAME per-sample K36.")
        print()

        # -------------------------------------------------------------
        # 1) Learn late Real-Gray writers.
        # -------------------------------------------------------------
        q_by_sid = {}

        for m in tqdm(train, desc="CALIBRATE late writers"):
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

                hr = dyn.capture_cpu(model, decoder_layers, rb, targets)
                hg = dyn.capture_cpu(model, decoder_layers, gb, targets)

                q_by_sid[sid] = {
                    T: (hr[T][0, -1] - hg[T][0, -1]).astype(np.float32)
                    for T in targets
                }

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()

        writers, writer_geom = dyn.learn_writers(
            train, q_by_sid, targets, a.writer_mode
        )
        write_csv(outdir / "writer_geometry.csv", writer_geom)

        # -------------------------------------------------------------
        # 2) Per sample: build ONE K36 ranking, then evaluate prefixes.
        # -------------------------------------------------------------
        baseline_rows = []
        ranked_rows = []
        generation_rows = []

        for m in tqdm(test, desc="EVAL ranked K36 prefixes"):
            sid = int(m["sid"])
            gt = m["gt"]
            writers_r = {T: writers[T][gt] for T in targets}

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
                cats, toks = dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                # Baseline actual generation.
                clean = dyn.run_generation_with_hooks(
                    model,
                    processor,
                    decoder_layers,
                    rb,
                    targets,
                    writers_r,
                    a.max_new_tokens,
                )
                clean_pred = clean["prediction"]
                clean_correct = clean_pred == gt

                baseline_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "baseline_prediction": DISPLAY.get(clean_pred, clean_pred),
                    "baseline_correct": clean_correct,
                    "baseline_text": clean["text"],
                    "baseline_mean_projection": clean["mean_projection"],
                    **{
                        f"baseline_proj_L{T}": clean["projection_by_target"][T]
                        for T in targets
                    },
                })

                # Gray source activations.
                hgray = dyn.capture_cpu(
                    model, decoder_layers, gb, source_layers
                )

                # Real graph + writer-guided source attribution.
                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model,
                        decoder_layers,
                        rb,
                        graph_layers,
                        cut,
                    )

                    objective_terms = []
                    for T in targets:
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
                        # Match current K36: source last token excluded.
                        for pos in range(max(0, npos - 1)):
                            delta = (Hreal[pos] - Hgray[pos]).astype(np.float32)
                            grad = G[pos]
                            med = float(np.dot(delta, grad))

                            rowsS.append({
                                "sid": sid,
                                "relation": DISPLAY[gt],
                                "source_layer": S,
                                "position": pos,
                                "token_id": int(ids[pos]),
                                "token": str(toks[pos]).replace("\n", "\\n"),
                                "category": cats[pos],
                                "broad_category": dyn.broad_category(cats[pos]),
                                "mediation": med,
                                "delta_h_norm": float(np.linalg.norm(delta)),
                                "grad_norm": float(np.linalg.norm(grad)),
                                "delta_h": delta,
                            })

                        rows_by_layer[S] = rowsS

                cap.close()
                cap = None

                # Build exactly ONE ranked K36 set.
                ranked = build_ranked_global_unique(
                    rows_by_layer, a.max_rank
                )

                total_m = sum(float(r["mediation"]) for r in ranked)
                cumulative_m = 0.0

                for rank, r in enumerate(ranked, 1):
                    mval = float(r["mediation"])
                    cumulative_m += mval

                    ranked_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "baseline_correct": clean_correct,
                        "rank": rank,
                        "source_layer": int(r["source_layer"]),
                        "position": int(r["position"]),
                        "token_id": int(r["token_id"]),
                        "token": r["token"],
                        "category": r["category"],
                        "broad_category": r["broad_category"],
                        "mediation": mval,
                        "mediation_share_within_K36": (
                            mval / total_m if total_m > EPS else float("nan")
                        ),
                        "cumulative_mediation": cumulative_m,
                        "cumulative_K36_mediation_fraction": (
                            cumulative_m / total_m
                            if total_m > EPS else float("nan")
                        ),
                        "delta_h_norm": float(r["delta_h_norm"]),
                        "grad_norm": float(r["grad_norm"]),
                    })

                # Actual generation for nested prefixes.
                for k in ks:
                    token_specs, selected = prefix_to_specs(ranked, k)

                    edited = dyn.run_generation_with_hooks(
                        model,
                        processor,
                        decoder_layers,
                        rb,
                        targets,
                        writers_r,
                        a.max_new_tokens,
                        token_specs=token_specs,
                        token_alpha=a.alpha,
                    )

                    pred = edited["prediction"]

                    generation_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "k": int(k),
                        "alpha": float(a.alpha),
                        "prediction": DISPLAY.get(pred, pred),
                        "correct": pred == gt,
                        "text": edited["text"],
                        "n_selected": len(selected),
                        "mean_projection": edited["mean_projection"],
                        "mean_cosine": edited["mean_cosine"],
                        **{
                            f"proj_L{T}": edited["projection_by_target"][T]
                            for T in targets
                        },
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
        # 3) Summaries.
        # -------------------------------------------------------------
        summary_rows = summarize(
            baseline_rows, generation_rows, ks
        )
        mass_rows = summarize_mass(
            ranked_rows, ks
        )

        write_csv(outdir / "baseline.csv", baseline_rows)
        write_csv(outdir / "ranked_k36_tokens.csv", ranked_rows)
        write_csv(outdir / "generation_by_k.csv", generation_rows)
        write_csv(outdir / "summary.csv", summary_rows)
        write_csv(outdir / "rank_mass_summary.csv", mass_rows)

        # Merge behavior + attribution-mass summary for convenient reading.
        mass_by_k = {int(r["k"]): r for r in mass_rows}
        combined_rows = []
        for r in summary_rows:
            k = int(r["k"])
            mr = mass_by_k.get(k, {})
            combined_rows.append({
                **r,
                "mean_K36_mediation_mass_fraction": mr.get(
                    "mean_K36_mediation_mass_fraction", float("nan")
                ),
                "median_K36_mediation_mass_fraction": mr.get(
                    "median_K36_mediation_mass_fraction", float("nan")
                ),
            })
        write_csv(outdir / "prefix_curve.csv", combined_rows)

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "source_layers": source_layers,
            "target_layers": targets,
            "max_rank": int(a.max_rank),
            "prefix_ks": ks,
            "alpha": float(a.alpha),
            "writer_mode": a.writer_mode,
            "writer_calibration_N": len(train),
            "eval_scope": a.eval_scope,
            "eval_N": len(test),
            "calibration_eval_overlap": overlap,
            "selection": (
                "One per-sample positive global_unique ranking by "
                "M=(h_real-h_gray)^T grad J_writer; K values are nested prefixes."
            ),
            "oracle_relation_used": True,
            "source_last_token_excluded": True,
        }
        (outdir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        # -------------------------------------------------------------
        # 4) Console report.
        # -------------------------------------------------------------
        baseline_acc = safe_mean(
            float(r["baseline_correct"]) for r in baseline_rows
        )

        print("\n" + "=" * 136)
        print("ORACLE K36 PREFIX CURVE — ACTUAL model.generate()")
        print("=" * 136)
        print(f"Baseline: N={len(baseline_rows)} acc={baseline_acc:.4f}")
        print()
        print(
            f"{'K':>4s} | {'M-mass':>8s} | {'acc':>8s} | {'gain':>8s} | "
            f"{'W2C':>5s} | {'C2W':>5s} | {'repair':>8s} | {'preserve':>8s} | {'writer Δ':>9s}"
        )
        print("-" * 100)

        for r in combined_rows:
            print(
                f"{int(r['k']):4d} | "
                f"{float(r['mean_K36_mediation_mass_fraction']):8.3f} | "
                f"{float(r['edited_acc']):8.4f} | "
                f"{float(r['delta_acc']):+8.4f} | "
                f"{int(r['W2C']):5d} | "
                f"{int(r['C2W']):5d} | "
                f"{float(r['repair_rate_W2C_over_wrong']):8.3f} | "
                f"{float(r['preserve_rate_correct_stays_correct']):8.3f} | "
                f"{float(r['mean_writer_projection_gain']):+9.3f}"
            )

        print("\nInterpretation:")
        print(
            "  - If K5/K7 already approaches K36 accuracy, K36 contains substantial "
            "behavioral redundancy and the strong causal core is much smaller."
        )
        print(
            "  - Compare M-mass vs accuracy: attribution mass concentration and "
            "behavioral sufficiency are not assumed to be the same."
        )
        print(
            "  - K values are nested prefixes of one ranking, so non-monotonic "
            "accuracy directly reveals that lower-ranked additions can help or hurt."
        )
        print(
            "  - ranked_k36_tokens.csv shows exactly which layer/token enters at "
            "each rank and its cumulative contribution share."
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
