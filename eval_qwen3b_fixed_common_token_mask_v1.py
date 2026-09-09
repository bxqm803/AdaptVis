#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed COMMON-POSITION middle-token amplification.

Goal
----
Test whether the common token positions discovered from the oracle mediation
analysis are useful WITHOUT using:
  - the late writer s_r at inference,
  - the ground-truth relation,
  - gradients,
  - per-sample Top-K selection.

For each evaluation sample we only compute each fixed token's own Real-Gray
activation displacement:

    delta_{S,p} = h_real[S,p] - h_gray[S,p]

and amplify the fixed canonical token position:

    h_edit[S,p] = h_real[S,p] + alpha * delta_{S,p]

This is therefore a writer-free / relation-free fixed-mask intervention.

Default fixed masks are derived from the current Qwen3B K=24 commonality result:

  M1_ref24
      L24::<REF>

  M2_ref24_subj22
      L24::<REF>
      L22::<SUBJ>

  M3_plus_to24
      + L24::C023:'Ġto'

  M5_core_text
      + L22::C024:'Ġthe'
      + L22::C021:'Ġin'

  M7_common_text
      + L24::C024:'Ġthe'
      + L24::C021:'Ġin'

The recommended evaluation is on UNSEEN test samples, excluding the N=32
samples used to discover these common positions.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen3b_fixed_common_token_mask_v1.py \
  --model qwen-3b \
  --selected-csv output/qwen3b_middle_token_multilayer_search_v2/selected_tokens.csv \
  --eval-scope unseen \
  --eval-max-samples 80 \
  --alphas 0.25,0.5,0.75,1.0 \
  --output-dir output/qwen3b_fixed_common_mask_unseen80 \
  --overwrite

For the original N=32 exploratory set:
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen3b_fixed_common_token_mask_v1.py \
  --model qwen-3b \
  --selected-csv output/qwen3b_middle_token_multilayer_search_v2/selected_tokens.csv \
  --eval-scope selected \
  --eval-max-samples 32 \
  --alphas 0.25,0.5,0.75,1.0 \
  --output-dir output/qwen3b_fixed_common_mask_seen32 \
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}


# ---------------------------------------------------------------------
# Fixed canonical masks from the current K=24 commonality analysis.
# ---------------------------------------------------------------------
MASKS = {
    "M1_ref24": [
        ("semantic", 24, "<REF>"),
    ],
    "M2_ref24_subj22": [
        ("semantic", 24, "<REF>"),
        ("semantic", 22, "<SUBJ>"),
    ],
    "M3_plus_to24": [
        ("semantic", 24, "<REF>"),
        ("semantic", 22, "<SUBJ>"),
        ("exact", 24, "C023:Ġto"),
    ],
    "M5_core_text": [
        ("semantic", 24, "<REF>"),
        ("semantic", 22, "<SUBJ>"),
        ("exact", 24, "C023:Ġto"),
        ("exact", 22, "C024:Ġthe"),
        ("exact", 22, "C021:Ġin"),
    ],
    "M7_common_text": [
        ("semantic", 24, "<REF>"),
        ("semantic", 22, "<SUBJ>"),
        ("exact", 24, "C023:Ġto"),
        ("exact", 22, "C024:Ġthe"),
        ("exact", 22, "C021:Ġin"),
        ("exact", 24, "C024:Ġthe"),
        ("exact", 24, "C021:Ġin"),
    ],
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument(
        "--selected-csv",
        required=True,
        help=(
            "selected_tokens.csv used only to identify the discovery sample IDs "
            "so unseen evaluation can exclude them. It is NOT used to select tokens "
            "for an evaluation sample."
        ),
    )
    p.add_argument(
        "--eval-scope",
        default="unseen",
        choices=["unseen", "selected", "all_test"],
        help=(
            "unseen: test split excluding discovery samples from selected_tokens.csv; "
            "selected: evaluate the original discovery samples; "
            "all_test: entire held-out test split."
        ),
    )
    p.add_argument("--masks", default="M1_ref24,M2_ref24_subj22,M3_plus_to24,M5_core_text,M7_common_text")
    p.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--max-samples", type=int, default=0)
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


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def make_gray_image(real_image, value):
    from PIL import Image
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


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
    vals = np.asarray([float(x) for x in xs], np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def build_canonical_mapping(tokens, subject_pos, reference_pos, visual_pos):
    """
    Collapse:
      subject span -> one <SUBJ> canonical slot
      reference span -> one <REF> canonical slot
      contiguous visual block -> one <VISUAL_BLOCK> slot
    Keep every other text token as an exact canonical slot Cxxx:token.
    """
    mapping = {}
    n = len(tokens)
    canonical_index = 0
    i = 0

    while i < n:
        if i in visual_pos:
            while i < n and i in visual_pos:
                mapping[i] = {
                    "kind": "visual",
                    "semantic": "<VISUAL>",
                    "exact": None,
                }
                i += 1
            canonical_index += 1
            continue

        if i in subject_pos:
            label = f"C{canonical_index:03d}:<SUBJ>"
            while i < n and i in subject_pos:
                mapping[i] = {
                    "kind": "subject",
                    "semantic": "<SUBJ>",
                    "exact": label,
                }
                i += 1
            canonical_index += 1
            continue

        if i in reference_pos:
            label = f"C{canonical_index:03d}:<REF>"
            while i < n and i in reference_pos:
                mapping[i] = {
                    "kind": "reference",
                    "semantic": "<REF>",
                    "exact": label,
                }
                i += 1
            canonical_index += 1
            continue

        tok = str(tokens[i]).replace("\n", "\\n")
        label = f"C{canonical_index:03d}:{tok}"
        mapping[i] = {
            "kind": "fixed_text",
            "semantic": tok,
            "exact": label,
        }
        canonical_index += 1
        i += 1

    return mapping


def resolve_mask_positions(mask_specs, mapping):
    """
    Returns layer -> sorted unique original token positions.
    For semantic <SUBJ>/<REF>, all subtokens in that object span are edited.
    """
    out = defaultdict(set)
    hits = []

    for mode, layer, key in mask_specs:
        found = []
        for pos, info in mapping.items():
            if mode == "semantic" and info["semantic"] == key:
                found.append(pos)
            elif mode == "exact" and info["exact"] == key:
                found.append(pos)

        for pos in found:
            out[int(layer)].add(int(pos))

        hits.append({
            "mode": mode,
            "layer": int(layer),
            "key": key,
            "n_positions": len(found),
            "positions": ",".join(str(x) for x in found),
        })

    return {L: sorted(ps) for L, ps in out.items()}, hits


class Capture:
    def __init__(self, decoder_layers, layers):
        self.states = {}
        self.handles = []
        for L in layers:
            self.handles.append(
                decoder_layers[L].register_forward_hook(self._hook(L))
            )

    def _hook(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            self.states[L] = h.detach().float().cpu()
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_states(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing captured layers: {missing}")
        return {
            L: cap.states[L][0].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


class FixedDeltaEditor:
    """
    At fixed canonical positions:
        h_real <- h_real + alpha * (h_real - h_gray)
    Deltas were captured from the clean real/gray forward passes.
    """
    def __init__(self, decoder_layers, positions_by_layer, deltas_by_layer, alpha, prompt_len):
        self.handles = []
        self.applied = defaultdict(int)
        self.prompt_len = int(prompt_len)
        self.alpha = float(alpha)

        for L, positions in positions_by_layer.items():
            if not positions:
                continue
            self.handles.append(
                decoder_layers[L].register_forward_hook(
                    self._hook(L, positions, deltas_by_layer[L])
                )
            )

    def _hook(self, L, positions, delta_matrix):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            # Only edit the prefill pass, not cached 1-token decode steps.
            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()
            for pos in positions:
                if 0 <= pos < y.shape[1] and pos < delta_matrix.shape[0]:
                    d = torch.as_tensor(
                        delta_matrix[pos],
                        device=y.device,
                        dtype=torch.float32,
                    )
                    y[:, pos, :] = y[:, pos, :] + self.alpha * d
                    self.applied[L] += 1
            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def generate_with_fixed_edit(
    model,
    processor,
    decoder_layers,
    batch,
    positions_by_layer,
    deltas_by_layer,
    alpha,
    max_new_tokens,
):
    prompt_len = int(batch["input_ids"].shape[1])
    editor = FixedDeltaEditor(
        decoder_layers,
        positions_by_layer,
        deltas_by_layer,
        alpha,
        prompt_len,
    )
    try:
        text = base.generate_text(
            model, processor, batch, max_new_tokens=max_new_tokens
        )
        pred = traj.normalize_relation(base, text)
        return text, pred, dict(editor.applied)
    finally:
        editor.close()


def stratified_take(rows, n, seed):
    if n <= 0 or n >= len(rows):
        return list(rows)
    return traj.stratified_cap(rows, n, seed)


def summarize(eval_rows, condition_rows):
    base_correct = {int(r["sid"]): bool(r["baseline_correct"]) for r in eval_rows}
    base_pred = {int(r["sid"]): r["baseline_prediction"] for r in eval_rows}
    base_acc = safe_mean(base_correct.values())

    groups = defaultdict(list)
    for r in condition_rows:
        groups[(r["mask"], float(r["alpha"]))].append(r)

    out = []
    for (mask, alpha), rows in groups.items():
        cur = {int(r["sid"]): r for r in rows}
        edited_correct = {
            sid: bool(cur[sid]["correct"])
            for sid in base_correct if sid in cur
        }
        edited_pred = {
            sid: cur[sid]["prediction"]
            for sid in base_correct if sid in cur
        }

        common_sids = sorted(set(base_correct) & set(edited_correct))
        acc = safe_mean(edited_correct[s] for s in common_sids)
        w2c = sum((not base_correct[s]) and edited_correct[s] for s in common_sids)
        c2w = sum(base_correct[s] and (not edited_correct[s]) for s in common_sids)
        changed = sum(base_pred[s] != edited_pred[s] for s in common_sids)

        out.append({
            "mask": mask,
            "alpha": alpha,
            "N": len(common_sids),
            "baseline_acc": base_acc,
            "edited_acc": acc,
            "gain": acc - base_acc,
            "W2C": int(w2c),
            "C2W": int(c2w),
            "net": int(w2c - c2w),
            "changed": int(changed),
            "mean_n_edited_positions": safe_mean(
                r["n_edited_positions"] for r in rows
            ),
        })

    return sorted(
        out,
        key=lambda x: (x["edited_acc"], x["net"]),
        reverse=True,
    )


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    requested_masks = [x.strip() for x in a.masks.split(",") if x.strip()]
    bad = [m for m in requested_masks if m not in MASKS]
    if bad:
        raise ValueError(f"Unknown masks={bad}; available={list(MASKS)}")
    alphas = parse_floats(a.alphas)

    selected = pd.read_csv(a.selected_csv)
    discovery_sids = sorted(set(pd.to_numeric(selected["sid"]).astype(int)))

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
    _train, test = traj.stratified_split(meta, a.train_ratio, a.seed)

    if a.eval_scope == "unseen":
        pool = [r for r in test if int(r["sid"]) not in set(discovery_sids)]
    elif a.eval_scope == "selected":
        disc = set(discovery_sids)
        pool = [r for r in test if int(r["sid"]) in disc]
    else:
        pool = list(test)

    eval_set = stratified_take(pool, a.eval_max_samples, a.seed + 101)

    if not eval_set:
        raise RuntimeError("Evaluation set is empty.")

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

    print("=" * 132)
    print("FIXED COMMON-POSITION MIDDLE AMPLIFICATION — NO WRITER / NO GT / NO GRADIENT")
    print("=" * 132)
    print(f"model={a.model} repo={spec.repo_id}")
    print(f"eval_scope={a.eval_scope}")
    print(f"discovery sample ids={len(discovery_sids)}")
    print(f"eval N={len(eval_set)}")
    print(f"masks={requested_masks}")
    print(f"alphas={alphas}")
    print()

    model = processor = None
    try:
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

        source_layers = sorted({
            int(layer)
            for m in requested_masks
            for _, layer, _ in MASKS[m]
        })
        for L in source_layers:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid for {a.model}, n_layers={n_layers}")

        eval_rows = []
        condition_rows = []
        resolved_rows = []

        device = torch.device(a.device)

        for m in tqdm(eval_set, desc="Fixed-mask eval"):
            sid = int(m["sid"])
            gt = m["gt"]
            real = gray = rb = gb = None

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

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
                toks = [
                    str(x)
                    for x in processor.tokenizer.convert_ids_to_tokens(ids)
                ]

                sspan, rspan = base.locate_object_spans(
                    processor.tokenizer,
                    ids,
                    m["subject"],
                    m["reference"],
                )
                subject_pos = set(range(int(sspan[0]), int(sspan[1]) + 1))
                reference_pos = set(range(int(rspan[0]), int(rspan[1]) + 1))
                visual_pos = {
                    i for i, tok in enumerate(toks)
                    if "image_pad" in tok or "video_pad" in tok
                }

                mapping = build_canonical_mapping(
                    toks, subject_pos, reference_pos, visual_pos
                )

                # Clean Real / Gray states only at source layers.
                hreal = capture_states(
                    model, decoder_layers, rb, source_layers
                )
                hgray = capture_states(
                    model, decoder_layers, gb, source_layers
                )
                deltas = {
                    L: (hreal[L] - hgray[L]).astype(np.float32)
                    for L in source_layers
                }

                # Baseline generation.
                baseline_text = base.generate_text(
                    model, processor, rb,
                    max_new_tokens=a.max_new_tokens
                )
                baseline_pred = traj.normalize_relation(base, baseline_text)
                baseline_correct = baseline_pred == gt

                eval_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "baseline_prediction": DISPLAY.get(
                        baseline_pred, baseline_pred
                    ),
                    "baseline_correct": baseline_correct,
                    "baseline_text": baseline_text,
                })

                for mask_name in requested_masks:
                    positions_by_layer, hits = resolve_mask_positions(
                        MASKS[mask_name], mapping
                    )

                    # Save how the fixed canonical mask resolved in this sample.
                    for hit in hits:
                        resolved_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "mask": mask_name,
                            **hit,
                        })

                    n_positions = sum(
                        len(ps) for ps in positions_by_layer.values()
                    )

                    for alpha in alphas:
                        text, pred, applied = generate_with_fixed_edit(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            positions_by_layer=positions_by_layer,
                            deltas_by_layer=deltas,
                            alpha=alpha,
                            max_new_tokens=a.max_new_tokens,
                        )
                        condition_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "mask": mask_name,
                            "alpha": alpha,
                            "prediction": DISPLAY.get(pred, pred),
                            "correct": pred == gt,
                            "text": text,
                            "n_edited_positions": n_positions,
                            "applied_by_layer": json.dumps(
                                {str(k): int(v) for k, v in applied.items()}
                            ),
                        })

            finally:
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

        summary = summarize(eval_rows, condition_rows)

        # Per-relation summaries.
        per_rel = []
        base_lookup = {int(r["sid"]): r for r in eval_rows}
        for mask_name in requested_masks:
            for alpha in alphas:
                cur = [
                    r for r in condition_rows
                    if r["mask"] == mask_name and float(r["alpha"]) == float(alpha)
                ]
                for rel_disp in ("left", "right", "on", "under"):
                    rows = [r for r in cur if r["gt"] == rel_disp]
                    if not rows:
                        continue
                    sids = [int(r["sid"]) for r in rows]
                    base_acc = safe_mean(
                        base_lookup[s]["baseline_correct"] for s in sids
                    )
                    edit_acc = safe_mean(r["correct"] for r in rows)
                    w2c = sum(
                        (not base_lookup[int(r["sid"])]["baseline_correct"])
                        and bool(r["correct"])
                        for r in rows
                    )
                    c2w = sum(
                        bool(base_lookup[int(r["sid"])]["baseline_correct"])
                        and (not bool(r["correct"]))
                        for r in rows
                    )
                    per_rel.append({
                        "mask": mask_name,
                        "alpha": alpha,
                        "relation": rel_disp,
                        "N": len(rows),
                        "baseline_acc": base_acc,
                        "edited_acc": edit_acc,
                        "gain": edit_acc - base_acc,
                        "W2C": w2c,
                        "C2W": c2w,
                        "net": w2c - c2w,
                    })

        write_csv(outdir / "baseline.csv", eval_rows)
        write_csv(outdir / "generation_conditions.csv", condition_rows)
        write_csv(outdir / "resolved_fixed_mask_positions.csv", resolved_rows)
        write_csv(outdir / "summary.csv", summary)
        write_csv(outdir / "per_relation_summary.csv", per_rel)

        print("\n" + "=" * 132)
        print("RESULTS")
        print("=" * 132)
        baseline_acc = safe_mean(r["baseline_correct"] for r in eval_rows)
        print(f"Baseline: N={len(eval_rows)} acc={baseline_acc:.4f}")
        print(
            f"{'mask':<24s} {'alpha':>7s} {'nEdit':>7s} "
            f"{'acc':>8s} {'gain':>8s} {'W2C':>5s} {'C2W':>5s} {'net':>5s}"
        )
        print("-" * 132)
        for r in summary:
            print(
                f"{r['mask']:<24s} "
                f"{float(r['alpha']):>7.3f} "
                f"{float(r['mean_n_edited_positions']):>7.2f} "
                f"{float(r['edited_acc']):>8.4f} "
                f"{float(r['gain']):>+8.4f} "
                f"{int(r['W2C']):>5d} "
                f"{int(r['C2W']):>5d} "
                f"{int(r['net']):>5d}"
            )

        print("\nMask definitions:")
        for name in requested_masks:
            print(f"  {name}")
            for mode, layer, key in MASKS[name]:
                print(f"    L{layer:02d} {mode:<8s} {key}")

        print("\nInterpretation:")
        print("  These masks are FIXED before each evaluation sample.")
        print("  No GT relation, late writer, gradient, or per-sample token ranking is used.")
        print("  Each chosen token only receives its own sample-specific Real-Gray delta.")
        if a.eval_scope == "unseen":
            print("  Evaluation samples exclude the sample IDs used to discover the common mask.")

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "eval_scope": a.eval_scope,
            "discovery_N": len(discovery_sids),
            "eval_N": len(eval_set),
            "masks": {m: MASKS[m] for m in requested_masks},
            "alphas": alphas,
            "seed": a.seed,
            "train_ratio": a.train_ratio,
            "writer_used_at_inference": False,
            "gt_relation_used_for_edit": False,
            "gradient_used_for_edit": False,
            "per_sample_topk_used_for_edit": False,
            "edit": "h_real[S,p] += alpha * (h_real[S,p] - h_gray[S,p])",
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
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
