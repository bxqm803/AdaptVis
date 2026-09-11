#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_direction_head_causal_enhancement_coco440_v1.py

Non-oracle COCO-440 causal enhancement using the frozen Synthetic-400
Direction-Head selector produced by:

    eval_direction_head_selector_synthetic400_to_coco440_v2.py

This script intentionally fixes the routing method to:
    pred_top10_equal

and sweeps only the number of selected causal states:
    K_state = 7,10,15,20,30

Default intervention:
    source layers      = L20..L26
    target writers     = L32,L34,L35
    selector           = frozen top10_equal Direction Heads
    state selection    = positive + global_unique
    alpha              = 1.0
    edit               = h_real + alpha * (h_real - h_gray)
    evaluation         = ALL COCO target samples in the selector CSV

IMPORTANT:
- COCO GT is NEVER used to choose the relation for intervention.
- The selected relation is always pred_top10_equal from the frozen selector CSV.
- GT is used only for evaluation, except for optional COCO writer calibration.
- If no --writer-npz is supplied, writers are calibrated exactly like
  eval_qwen_dynamic_k24_all440_v1.py on the 30% COCO calibration split.
  Therefore that version is "cross-domain selector + calibrated actuator",
  not fully source-only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import random
import shutil
from collections import Counter, defaultdict
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
DISPLAY_INV = {"left": "left", "right": "right", "on": "above", "under": "below",
               "above": "above", "below": "below"}


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

    # Frozen selector output from eval_direction_head_selector...v2.py
    p.add_argument(
        "--selector-csv",
        default=(
            "output/qwen3b_direction_selector_syn400_to_coco440/"
            "synthetic400_to_coco440_predictions.csv"
        ),
    )
    p.add_argument("--selector-column", default="pred_top10_equal")

    # Fixed causal actuator configuration; exposed only for reproducibility.
    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument("--ks", default="7,10,15,20,30")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)

    # Same writer calibration as eval_qwen_dynamic_k24_all440_v1.py.
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument(
        "--writer-npz",
        default="",
        help=(
            "Optional precomputed learned_writers.npz. If empty, calibrate writers "
            "on the COCO calibration split exactly as in the old dynamic script."
        ),
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    # Full 440 is the default. Nonzero is only for debugging.
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all selector-covered COCO samples; >0 = stratified debug cap.",
    )
    p.add_argument(
        "--require-n",
        type=int,
        default=440,
        help="Require this many samples when --max-samples=0; set <=0 to disable.",
    )
    p.add_argument(
        "--save-all-mediation",
        action="store_true",
        help="Save every L20-L26 mediation state; can produce a large CSV.",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    aliases = {
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
    return aliases.get(s, s)


def parse_ints(text: str) -> List[int]:
    return sorted(
        {int(x.strip().upper().replace("L", ""))
         for x in str(text).split(",") if x.strip()}
    )


def write_csv(path: Path, rows):
    rows = list(rows)
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


def load_selector_csv(path: Path, pred_col: str):
    if not path.exists():
        raise FileNotFoundError(path)

    selector = {}
    selector_gt = {}
    selector_conf = {}
    selector_margin = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols = set(rd.fieldnames or [])
        need = {"sid", pred_col}
        missing = need - cols
        if missing:
            raise RuntimeError(
                f"{path} missing columns {sorted(missing)}; "
                f"available={sorted(cols)}"
            )

        conf_col = pred_col.replace("pred_", "confidence_", 1)
        margin_col = pred_col.replace("pred_", "margin_", 1)

        for row in rd:
            sid = int(row["sid"])
            if sid in selector:
                raise RuntimeError(f"Duplicate sid={sid} in {path}")
            pred = canon_rel(row[pred_col])
            if pred not in REL:
                raise RuntimeError(
                    f"sid={sid}: invalid selector relation {row[pred_col]!r}"
                )
            selector[sid] = pred

            if "gt" in row and row["gt"] != "":
                selector_gt[sid] = canon_rel(row["gt"])
            if conf_col in row and row[conf_col] != "":
                selector_conf[sid] = float(row[conf_col])
            if margin_col in row and row[margin_col] != "":
                selector_margin[sid] = float(row[margin_col])

    return selector, selector_gt, selector_conf, selector_margin


def load_writers_npz(path: Path, targets):
    z = np.load(path, allow_pickle=True)
    writers = {T: {} for T in targets}
    for T in targets:
        for r in REL:
            candidates = [
                f"L{T}_{r}",
                f"L{T}_{DISPLAY[r]}",
            ]
            key = next((k for k in candidates if k in z.files), None)
            if key is None:
                raise RuntimeError(
                    f"{path}: missing writer for L{T}/{r}; tried {candidates}"
                )
            writers[T][r] = np.asarray(z[key], dtype=np.float32)
    return writers


def summarize_k(eval_rows, generation_rows, ks):
    baseline = {int(r["sid"]): bool(r["baseline_correct"]) for r in eval_rows}
    baseline_pred = {int(r["sid"]): r["baseline_prediction"] for r in eval_rows}
    selector_ok = {int(r["sid"]): bool(r["selector_correct"]) for r in eval_rows}

    out = []
    for k in ks:
        rows = [r for r in generation_rows if int(r["k"]) == int(k)]
        by_sid = {int(r["sid"]): r for r in rows}

        missing = sorted(set(baseline) - set(by_sid))
        if missing:
            raise RuntimeError(
                f"K={k}: missing edited rows for {len(missing)} sids, "
                f"first={missing[:10]}"
            )

        def subset_stats(name, sids):
            sids = list(sids)
            if not sids:
                return {
                    f"N_{name}": 0,
                    f"baseline_acc_{name}": float("nan"),
                    f"edited_acc_{name}": float("nan"),
                    f"W2C_{name}": 0,
                    f"C2W_{name}": 0,
                }
            b = [baseline[s] for s in sids]
            e = [bool(by_sid[s]["correct"]) for s in sids]
            w2c = sum((not baseline[s]) and bool(by_sid[s]["correct"]) for s in sids)
            c2w = sum(baseline[s] and (not bool(by_sid[s]["correct"])) for s in sids)
            return {
                f"N_{name}": len(sids),
                f"baseline_acc_{name}": safe_mean(b),
                f"edited_acc_{name}": safe_mean(e),
                f"W2C_{name}": int(w2c),
                f"C2W_{name}": int(c2w),
            }

        all_sids = sorted(baseline)
        sel_correct_sids = [s for s in all_sids if selector_ok[s]]
        sel_wrong_sids = [s for s in all_sids if not selector_ok[s]]

        edited_correct = {s: bool(by_sid[s]["correct"]) for s in all_sids}
        w2c = sum((not baseline[s]) and edited_correct[s] for s in all_sids)
        c2w = sum(baseline[s] and (not edited_correct[s]) for s in all_sids)
        changed = sum(
            str(by_sid[s]["prediction"]) != str(baseline_pred[s]) for s in all_sids
        )

        row = {
            "k": int(k),
            "N": len(all_sids),
            "baseline_acc": safe_mean(baseline.values()),
            "edited_acc": safe_mean(edited_correct.values()),
            "delta_acc": (
                safe_mean(edited_correct.values()) - safe_mean(baseline.values())
            ),
            "W2C": int(w2c),
            "C2W": int(c2w),
            "net": int(w2c - c2w),
            "changed": int(changed),
            "mean_selected_tokens": safe_mean(
                float(by_sid[s]["n_selected_total"]) for s in all_sids
            ),
            **subset_stats("selector_correct", sel_correct_sids),
            **subset_stats("selector_wrong", sel_wrong_sids),
        }

        for rel in REL:
            rel_sids = [
                int(r["sid"]) for r in eval_rows if canon_rel(r["gt_internal"]) == rel
            ]
            row[f"N_{rel}"] = len(rel_sids)
            row[f"edited_acc_{rel}"] = safe_mean(
                bool(by_sid[s]["correct"]) for s in rel_sids
            )

        out.append(row)
    return out


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    sources = parse_ints(a.source_layers)
    targets = parse_ints(a.target_layers)
    ks = parse_ints(a.ks)
    if not sources:
        raise ValueError("--source-layers is empty")
    if not targets:
        raise ValueError("--target-layers is empty")
    if not ks:
        raise ValueError("--ks is empty")

    # This experiment is intentionally a single actuator configuration.
    bundle = tuple(sources)
    strategy = "global_unique"
    mode = "positive"
    alpha = float(a.alpha)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    selector_path = Path(a.selector_csv)
    selector, selector_gt, selector_conf, selector_margin = load_selector_csv(
        selector_path, a.selector_column
    )

    # -------------------------------------------------------------------------
    # Load COCO metadata in the same SID convention used by the old dynamic run.
    # -------------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    gt_mismatches = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts or sid not in selector:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue

        if sid in selector_gt and selector_gt[sid] != gt:
            gt_mismatches.append((sid, gt, selector_gt[sid]))

        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
            "selector_relation": selector[sid],
            "selector_correct": selector[sid] == gt,
            "selector_confidence": selector_conf.get(sid, float("nan")),
            "selector_margin": selector_margin.get(sid, float("nan")),
        })

    meta.sort(key=lambda x: int(x["sid"]))

    if gt_mismatches:
        raise RuntimeError(
            "Selector CSV / COCO GT SID alignment mismatch. First mismatches: "
            + repr(gt_mismatches[:10])
        )

    missing_selector = sorted(
        sid for sid in rec_by_sid
        if sid in prompts and sid not in selector
    )
    if missing_selector:
        print(
            f"[warning] {len(missing_selector)} COCO prompt SIDs are absent from "
            f"selector CSV; first={missing_selector[:10]}",
            flush=True,
        )

    if a.max_samples > 0:
        meta = traj.stratified_cap(meta, a.max_samples, a.seed + 11)
    elif a.require_n > 0 and len(meta) != a.require_n:
        raise RuntimeError(
            f"Expected N={a.require_n} full-data samples but resolved N={len(meta)}. "
            "Check selector CSV, prompt file, and COCO SID alignment. "
            "Use --require-n 0 only if a non-440 target is intentional."
        )

    if not meta:
        raise RuntimeError("No evaluation samples resolved.")

    # Writer calibration split follows the prior dynamic script. Evaluation,
    # however, remains all meta rows.
    writer_train, _ = traj.stratified_split(meta, a.train_ratio, a.seed)
    eval_rows_meta = list(meta)
    writer_train_sids = {int(x["sid"]) for x in writer_train}
    eval_sids = {int(x["sid"]) for x in eval_rows_meta}
    calibration_eval_overlap = len(writer_train_sids & eval_sids)

    print("=" * 150)
    print("FROZEN TOP10 DIRECTION SELECTOR -> CAUSAL STATE AMPLIFICATION -> ACTUAL GENERATION")
    print("=" * 150)
    print(f"selector_csv={selector_path}")
    print(f"selector_column={a.selector_column}")
    print(
        f"selector accuracy on resolved target = "
        f"{safe_mean(float(x['selector_correct']) for x in meta):.4f} "
        f"({sum(x['selector_correct'] for x in meta)}/{len(meta)})"
    )
    print(f"eval N={len(eval_rows_meta)} (ALL DATA)")
    print(f"source layers={sources}")
    print(f"target writer layers={targets}")
    print(f"K_state={ks}")
    print(f"selection={mode}+{strategy} | alpha={alpha}")
    if a.writer_npz:
        print(f"writers=precomputed {a.writer_npz}")
    else:
        print(
            f"writers=COCO calibrated | train_ratio={a.train_ratio} "
            f"| calibration N={len(writer_train)} "
            f"| calibration/eval overlap={calibration_eval_overlap}"
        )
    print("COCO GT is evaluation-only for ROUTING; selector relation chooses writer target.")
    print()

    # -------------------------------------------------------------------------
    # Load model.
    # -------------------------------------------------------------------------
    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)

    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    model = processor = None
    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)
        for L in sources + targets:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid; model has L0...L{n_layers-1}")
        for S in sources:
            if S >= max(targets):
                raise ValueError(
                    f"Source L{S} must be earlier than useful target writers {targets}"
                )

        cut = min(sources)
        graph_layers = sorted(set(sources + targets))
        device = torch.device(a.device)

        print(f"decoder={decoder_path} | n_layers={n_layers}")

        # ---------------------------------------------------------------------
        # Writers: either load precomputed ones or calibrate as before.
        # ---------------------------------------------------------------------
        if a.writer_npz:
            writers = load_writers_npz(Path(a.writer_npz), targets)
            writer_geom = []
        else:
            q_by_sid = {}
            for m in tqdm(writer_train, desc="CALIBRATE late writers"):
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
                    rb = None
                    gb = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            writers, writer_geom = dyn.learn_writers(
                writer_train, q_by_sid, targets, a.writer_mode
            )
            dyn.write_csv(outdir / "writer_geometry.csv", writer_geom)
            np.savez_compressed(
                outdir / "learned_writers.npz",
                **{
                    f"L{T}_{DISPLAY[r]}": writers[T][r]
                    for T in targets for r in REL
                },
            )

        # ---------------------------------------------------------------------
        # Full-data non-oracle evaluation.
        # ---------------------------------------------------------------------
        baseline_rows = []
        generation_rows = []
        selection_rows = []
        mediation_rows = []

        for m in tqdm(eval_rows_meta, desc="ALL COCO select + trace + generate"):
            sid = int(m["sid"])
            gt = m["gt"]
            r_hat = m["selector_relation"]

            # NON-ORACLE ROUTING: Direction Head prediction chooses writer.
            writers_r = {T: writers[T][r_hat] for T in targets}

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
                    model, processor, rb, ids, m["subject"], m["reference"]
                )

                # Baseline generation once per sample.
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
                    "gt_internal": gt,
                    "gt": DISPLAY[gt],
                    "selector_relation_internal": r_hat,
                    "selector_relation": DISPLAY[r_hat],
                    "selector_correct": r_hat == gt,
                    "selector_confidence": m["selector_confidence"],
                    "selector_margin": m["selector_margin"],
                    "baseline_prediction_internal": clean_pred,
                    "baseline_prediction": DISPLAY.get(clean_pred, clean_pred),
                    "baseline_correct": clean_correct,
                    "baseline_text": clean["text"],
                    "baseline_mean_projection_to_selected_writer": clean["mean_projection"],
                })

                # Natural Real-Gray visual displacement at all source layers.
                hgray = dyn.capture_cpu(model, decoder_layers, gb, sources)

                # Writer-conditioned causal tracing for the SELECTED relation.
                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model, decoder_layers, rb, graph_layers, cut
                    )

                    objective_terms = []
                    for T in targets:
                        s_hat = torch.as_tensor(
                            dyn.normalize_np(writers_r[T]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        objective_terms.append(
                            torch.dot(cap.states[T][0, -1].float(), s_hat)
                        )
                    objective = torch.stack(objective_terms).sum()

                    grads = torch.autograd.grad(
                        objective,
                        [cap.states[S] for S in sources],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                    rows_by_layer = {}
                    for S, g in zip(sources, grads):
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
                        # Match old experiment: do not edit source last token.
                        for pos in range(max(0, npos - 1)):
                            delta = (Hreal[pos] - Hgray[pos]).astype(np.float32)
                            grad = G[pos]
                            med = float(np.dot(delta, grad))
                            row = {
                                "sid": sid,
                                "gt": DISPLAY[gt],
                                "selector_relation": DISPLAY[r_hat],
                                "selector_correct": r_hat == gt,
                                "source_layer": S,
                                "position": pos,
                                "token_id": int(ids[pos]),
                                "token": str(toks[pos]).replace("\n", "\\n"),
                                "category": cats[pos],
                                "broad_category": dyn.broad_category(cats[pos]),
                                "mediation": med,
                                "abs_mediation": abs(med),
                                "delta_h_norm": float(np.linalg.norm(delta)),
                                "grad_norm": float(np.linalg.norm(grad)),
                                # in-memory only
                                "delta_h": delta,
                            }
                            rowsS.append(row)

                        rowsS.sort(
                            key=lambda x: x["mediation"], reverse=True
                        )
                        rows_by_layer[S] = rowsS

                        if a.save_all_mediation:
                            for rank, r0 in enumerate(rowsS, 1):
                                export = {
                                    k: v for k, v in r0.items()
                                    if k != "delta_h"
                                }
                                export["positive_rank_within_layer"] = rank
                                mediation_rows.append(export)

                cap.close()
                cap = None

                # One fixed ranking, multiple prefix sizes K.
                for k in ks:
                    token_specs, selected = dyn.make_specs(
                        bundle=bundle,
                        rows_by_layer=rows_by_layer,
                        mode=mode,
                        k=k,
                        sid=sid,
                        seed=a.seed,
                        strategy=strategy,
                    )

                    for sel in selected:
                        selection_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "selector_relation": DISPLAY[r_hat],
                            "selector_correct": r_hat == gt,
                            "selection_strategy": strategy,
                            "source_bundle": dyn.bundle_name(bundle),
                            "k": k,
                            **sel,
                        })

                    if selected:
                        edited = dyn.run_generation_with_hooks(
                            model,
                            processor,
                            decoder_layers,
                            rb,
                            targets,
                            writers_r,
                            a.max_new_tokens,
                            token_specs=token_specs,
                            token_alpha=alpha,
                        )
                        pred = edited["prediction"]
                        text = edited["text"]
                        mean_proj = edited["mean_projection"]
                    else:
                        # Keep summary complete even if a sample has no positive state.
                        pred = clean_pred
                        text = clean["text"]
                        mean_proj = clean["mean_projection"]

                    generation_rows.append({
                        "sid": sid,
                        "gt_internal": gt,
                        "gt": DISPLAY[gt],
                        "selector_relation_internal": r_hat,
                        "selector_relation": DISPLAY[r_hat],
                        "selector_correct": r_hat == gt,
                        "selector_confidence": m["selector_confidence"],
                        "selector_margin": m["selector_margin"],
                        "k": k,
                        "alpha": alpha,
                        "selection_strategy": strategy,
                        "prediction_internal": pred,
                        "prediction": DISPLAY.get(pred, pred),
                        "correct": pred == gt,
                        "text": text,
                        "mean_projection_to_selected_writer": mean_proj,
                        "n_selected_total": len(selected),
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
                rb = None
                gb = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Save and report.
        # ---------------------------------------------------------------------
        summary = summarize_k(baseline_rows, generation_rows, ks)

        write_csv(outdir / "baseline.csv", baseline_rows)
        write_csv(outdir / "generation_by_k.csv", generation_rows)
        write_csv(outdir / "selected_states.csv", selection_rows)
        write_csv(outdir / "summary_by_k.csv", summary)
        if a.save_all_mediation:
            write_csv(outdir / "all_mediation_states.csv", mediation_rows)

        metadata = {
            "script": "eval_direction_head_causal_enhancement_coco440_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "selector_csv": str(selector_path),
            "selector_column": a.selector_column,
            "selector_N": len(meta),
            "selector_accuracy": safe_mean(
                float(x["selector_correct"]) for x in meta
            ),
            "routing_uses_coco_gt": False,
            "source_layers": sources,
            "target_layers": targets,
            "ks": ks,
            "alpha": alpha,
            "condition": mode,
            "selection_strategy": strategy,
            "edit_definition": (
                "h_edit = h_real + alpha * (h_real - h_gray)"
            ),
            "mediation_definition": (
                "(h_real-h_gray)^T * gradient of "
                "sum_T <h_T,last, normalized selected-relation writer_T>"
            ),
            "writer_mode": a.writer_mode,
            "writer_npz": a.writer_npz or None,
            "writer_calibration_uses_coco_gt": not bool(a.writer_npz),
            "writer_calibration_N": 0 if a.writer_npz else len(writer_train),
            "eval_N": len(eval_rows_meta),
            "calibration_eval_overlap": (
                0 if a.writer_npz else calibration_eval_overlap
            ),
            "source_last_excluded": True,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        print()
        print("=" * 170)
        print("FULL COCO NON-ORACLE DIRECTION-HEAD CAUSAL ENHANCEMENT")
        print("=" * 170)
        print(
            f"Baseline: N={len(baseline_rows)} "
            f"acc={safe_mean(float(r['baseline_correct']) for r in baseline_rows):.4f}"
        )
        print(
            f"Selector: acc="
            f"{safe_mean(float(r['selector_correct']) for r in baseline_rows):.4f}"
        )
        print()
        header = (
            f"{'K':>4s} {'acc':>8s} {'gain':>8s} "
            f"{'W2C':>5s} {'C2W':>5s} {'net':>5s} {'changed':>8s} "
            f"{'nEdit':>7s} "
            f"{'selOK acc':>10s} {'selBAD acc':>11s} "
            f"{'L':>7s} {'R':>7s} {'A':>7s} {'B':>7s}"
        )
        print(header)
        print("-" * len(header))
        for r in summary:
            print(
                f"{int(r['k']):>4d} "
                f"{float(r['edited_acc']):>8.4f} "
                f"{float(r['delta_acc']):>+8.4f} "
                f"{int(r['W2C']):>5d} "
                f"{int(r['C2W']):>5d} "
                f"{int(r['net']):>5d} "
                f"{int(r['changed']):>8d} "
                f"{float(r['mean_selected_tokens']):>7.2f} "
                f"{float(r['edited_acc_selector_correct']):>10.4f} "
                f"{float(r['edited_acc_selector_wrong']):>11.4f} "
                f"{float(r['edited_acc_left']):>7.4f} "
                f"{float(r['edited_acc_right']):>7.4f} "
                f"{float(r['edited_acc_above']):>7.4f} "
                f"{float(r['edited_acc_below']):>7.4f}"
            )

        print()
        print("Saved:", outdir)
        print("  summary_by_k.csv")
        print("  generation_by_k.csv")
        print("  baseline.csv")
        print("  selected_states.csv")
        print("  metadata.json")
        if a.save_all_mediation:
            print("  all_mediation_states.csv")

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
