#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT-selected CORRECT RELATION TOKEN amplification.

Question
--------
Is most of the middle-layer repair effect concentrated on the four answer
relation tokens {left, right, above, below}?

For each sample, use the GT relation ONLY to choose which relation-word token
to edit:

    GT left  -> token "left"
    GT right -> token "right"
    GT on    -> token "above"
    GT under -> token "below"

No late writer, no gradient, no per-sample Top-K.

At chosen source layer(s) S, amplify only that token's own Real-Gray delta:

    delta_{S,p} = h_real[S,p] - h_gray[S,p]
    h_edit[S,p] = h_real[S,p] + alpha * delta_{S,p}

We test the same GT-selected relation token at:
    L22
    L24
    L26
    L22+L24
    L24+L26
    L22+L24+L26

This is an oracle relation-routing experiment because GT chooses WHICH of the
four relation tokens is edited.  It directly tests whether the large Top-K
effect can be reproduced mostly through the correct relation-word position.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen3b_gt_relation_token_amplify_v1.py \
  --model qwen-3b \
  --eval-max-samples 80 \
  --alphas 0.125,0.25,0.5,0.75,1.0,1.5,2.0 \
  --layer-bundles "22;24;26;22+24;24+26;22+24+26" \
  --output-dir output/qwen3b_gt_relation_token_amplify_v1 \
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

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
GT_TOKEN = {
    "left": "left",
    "right": "right",
    "above": "above",
    "below": "below",
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
        "--layer-bundles",
        default="22;24;26;22+24;24+26;22+24+26",
        help='Semicolon-separated bundles, e.g. "22;24;26;22+24+26".',
    )
    p.add_argument("--alphas", default="0.125,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--occurrence",
        default="answer_option",
        choices=["answer_option", "all"],
        help=(
            "answer_option: choose the canonical answer-option occurrence "
            "(left/right/above/below slots); all: edit all exact occurrences "
            "of the GT relation word."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_bundles(s):
    out = []
    for raw in str(s).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        vals = tuple(sorted({int(x.strip().replace("L", "").replace("l", ""))
                             for x in raw.split("+") if x.strip()}))
        out.append(vals)
    return out


def bundle_name(b):
    return "+".join(f"L{x}" for x in b)


def safe_mean(xs):
    vals = np.asarray([float(x) for x in xs], np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


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


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def all_subsequence_positions(ids, pat):
    if not pat or len(pat) > len(ids):
        return []
    out = []
    n = len(pat)
    for i in range(len(ids) - n + 1):
        if ids[i:i+n] == pat:
            out.append(tuple(range(i, i+n)))
    return out


def find_word_occurrences(tokenizer, ids, word):
    """
    Return candidate spans for exact tokenization of word / leading-space word.
    """
    occ = set()
    for variant in (word, " " + word):
        pat = tokenizer.encode(variant, add_special_tokens=False)
        for span in all_subsequence_positions(ids, pat):
            occ.add(tuple(span))
    return sorted(occ)


def build_canonical_slots(tokens, subject_pos, reference_pos, visual_pos):
    """
    Same collapsing convention used in the previous canonical-position script.
    Returns original position -> canonical exact label.
    """
    mapping = {}
    n = len(tokens)
    c = 0
    i = 0
    while i < n:
        if i in visual_pos:
            while i < n and i in visual_pos:
                mapping[i] = f"C{c:03d}:<VISUAL_BLOCK>"
                i += 1
            c += 1
            continue

        if i in subject_pos:
            while i < n and i in subject_pos:
                mapping[i] = f"C{c:03d}:<SUBJ>"
                i += 1
            c += 1
            continue

        if i in reference_pos:
            while i < n and i in reference_pos:
                mapping[i] = f"C{c:03d}:<REF>"
                i += 1
            c += 1
            continue

        tok = str(tokens[i]).replace("\n", "\\n")
        mapping[i] = f"C{c:03d}:{tok}"
        i += 1
        c += 1

    return mapping


ANSWER_OPTION_CANON = {
    # Observed canonical answer-option slots in the current prompt template.
    "left": "C029:Ġleft",
    "right": "C031:Ġright",
    "above": "C033:Ġabove",
    "below": "C035:Ġbelow",
}


def resolve_gt_relation_positions(
    tokenizer,
    ids,
    tokens,
    subject_pos,
    reference_pos,
    visual_pos,
    gt,
    occurrence_mode,
):
    """
    Resolve the GT relation word to original token positions.

    answer_option mode:
      Prefer the exact canonical answer-option slot from the current template.
      If unavailable, fall back to the last exact occurrence of the GT word.

    all mode:
      Use every exact occurrence.
    """
    word = GT_TOKEN[gt]
    occs = find_word_occurrences(tokenizer, ids, word)

    if occurrence_mode == "all":
        positions = sorted({p for span in occs for p in span})
        return positions, "all_exact_occurrences"

    cmap = build_canonical_slots(tokens, subject_pos, reference_pos, visual_pos)
    target_label = ANSWER_OPTION_CANON[gt]
    positions = [p for p, label in cmap.items() if label == target_label]
    if positions:
        return sorted(positions), target_label

    # Robust fallback for small prompt-format changes: use the latest exact
    # occurrence, which in this multiple-choice template should be the option.
    if occs:
        return list(occs[-1]), "fallback_last_occurrence"

    return [], "not_found"


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
            raise RuntimeError(f"Missing layers {missing}")
        return {
            L: cap.states[L][0].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


class RelationTokenEditor:
    def __init__(
        self,
        decoder_layers,
        bundle,
        positions,
        deltas_by_layer,
        alpha,
        prompt_len,
    ):
        self.handles = []
        self.applied = defaultdict(int)
        self.prompt_len = int(prompt_len)
        self.alpha = float(alpha)

        for L in bundle:
            self.handles.append(
                decoder_layers[L].register_forward_hook(
                    self._hook(L, positions, deltas_by_layer[L])
                )
            )

    def _hook(self, L, positions, delta_matrix):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) != self.prompt_len:
                return out
            y = h.float().clone()
            for p in positions:
                if 0 <= p < y.shape[1] and p < delta_matrix.shape[0]:
                    delta = torch.as_tensor(
                        delta_matrix[p],
                        device=y.device,
                        dtype=torch.float32,
                    )
                    y[:, p, :] = y[:, p, :] + self.alpha * delta
                    self.applied[L] += 1
            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def generate_edit(
    model,
    processor,
    decoder_layers,
    batch,
    bundle,
    positions,
    deltas_by_layer,
    alpha,
    max_new_tokens,
):
    editor = RelationTokenEditor(
        decoder_layers,
        bundle,
        positions,
        deltas_by_layer,
        alpha,
        int(batch["input_ids"].shape[1]),
    )
    try:
        text = base.generate_text(
            model, processor, batch, max_new_tokens=max_new_tokens
        )
        pred = traj.normalize_relation(base, text)
        return text, pred, dict(editor.applied)
    finally:
        editor.close()


def summarize(eval_rows, cond_rows):
    base = {int(r["sid"]): r for r in eval_rows}
    base_acc = safe_mean(r["baseline_correct"] for r in eval_rows)

    groups = defaultdict(list)
    for r in cond_rows:
        groups[(r["bundle"], float(r["alpha"]))].append(r)

    out = []
    for (bundle, alpha), rows in groups.items():
        w2c = c2w = changed = 0
        corr = []
        for r in rows:
            sid = int(r["sid"])
            b = base[sid]
            c = bool(r["correct"])
            corr.append(c)
            w2c += int((not bool(b["baseline_correct"])) and c)
            c2w += int(bool(b["baseline_correct"]) and (not c))
            changed += int(r["prediction"] != b["baseline_prediction"])

        acc = safe_mean(corr)
        out.append({
            "bundle": bundle,
            "alpha": alpha,
            "N": len(rows),
            "baseline_acc": base_acc,
            "edited_acc": acc,
            "gain": acc - base_acc,
            "W2C": w2c,
            "C2W": c2w,
            "net": w2c - c2w,
            "changed": changed,
            "mean_edited_positions": safe_mean(r["n_edited_positions"] for r in rows),
        })

    return sorted(
        out,
        key=lambda x: (x["edited_acc"], x["net"]),
        reverse=True
    )


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    bundles = parse_bundles(a.layer_bundles)
    all_layers = sorted({L for b in bundles for L in b})
    alphas = parse_floats(a.alphas)

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
    _train, test = traj.stratified_split(meta, a.train_ratio, a.seed)
    test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)

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
    print("GT-SELECTED CORRECT RELATION TOKEN AMPLIFICATION")
    print("=" * 132)
    print(f"model={a.model} repo={spec.repo_id}")
    print(f"N={len(test)}")
    print(f"layers={[bundle_name(b) for b in bundles]}")
    print(f"alphas={alphas}")
    print(f"occurrence={a.occurrence}")
    print("NO late writer / NO gradient / NO Top-K")
    print("GT is used only to choose one of: left/right/above/below")
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
        for L in all_layers:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid for model with {n_layers} layers")

        device = torch.device(a.device)
        eval_rows = []
        cond_rows = []
        resolved_rows = []

        for m in tqdm(test, desc="GT relation token eval"):
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
                    str(x) for x in processor.tokenizer.convert_ids_to_tokens(ids)
                ]

                sspan, rspan = base.locate_object_spans(
                    processor.tokenizer,
                    ids,
                    m["subject"],
                    m["reference"],
                )
                spos = set(range(int(sspan[0]), int(sspan[1]) + 1))
                rpos = set(range(int(rspan[0]), int(rspan[1]) + 1))
                vpos = {
                    i for i, tok in enumerate(toks)
                    if "image_pad" in tok or "video_pad" in tok
                }

                positions, resolution = resolve_gt_relation_positions(
                    processor.tokenizer,
                    ids,
                    toks,
                    spos,
                    rpos,
                    vpos,
                    gt,
                    a.occurrence,
                )
                if not positions:
                    raise RuntimeError(
                        f"sid={sid}: failed to locate GT relation token {GT_TOKEN[gt]!r}"
                    )

                resolved_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "gt_relation_token": GT_TOKEN[gt],
                    "resolution": resolution,
                    "positions": ",".join(str(x) for x in positions),
                    "tokens": ",".join(str(toks[p]) for p in positions),
                    "n_positions": len(positions),
                })

                hreal = capture_states(model, decoder_layers, rb, all_layers)
                hgray = capture_states(model, decoder_layers, gb, all_layers)
                deltas = {
                    L: (hreal[L] - hgray[L]).astype(np.float32)
                    for L in all_layers
                }

                baseline_text = base.generate_text(
                    model, processor, rb,
                    max_new_tokens=a.max_new_tokens
                )
                baseline_pred = traj.normalize_relation(base, baseline_text)
                eval_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "baseline_prediction": DISPLAY.get(
                        baseline_pred, baseline_pred
                    ),
                    "baseline_correct": baseline_pred == gt,
                    "baseline_text": baseline_text,
                })

                for bundle in bundles:
                    for alpha in alphas:
                        text, pred, applied = generate_edit(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            bundle=bundle,
                            positions=positions,
                            deltas_by_layer=deltas,
                            alpha=alpha,
                            max_new_tokens=a.max_new_tokens,
                        )
                        cond_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "gt_relation_token": GT_TOKEN[gt],
                            "bundle": bundle_name(bundle),
                            "alpha": alpha,
                            "prediction": DISPLAY.get(pred, pred),
                            "correct": pred == gt,
                            "text": text,
                            "n_edited_positions": sum(applied.values()),
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

        summary = summarize(eval_rows, cond_rows)

        # Per-relation results.
        base_lookup = {int(r["sid"]): r for r in eval_rows}
        per_rel = []
        for bundle in bundles:
            bname = bundle_name(bundle)
            for alpha in alphas:
                rows = [
                    r for r in cond_rows
                    if r["bundle"] == bname and float(r["alpha"]) == float(alpha)
                ]
                for rel_disp in ("left", "right", "on", "under"):
                    rr = [r for r in rows if r["gt"] == rel_disp]
                    if not rr:
                        continue
                    sids = [int(r["sid"]) for r in rr]
                    base_acc = safe_mean(
                        base_lookup[s]["baseline_correct"] for s in sids
                    )
                    edit_acc = safe_mean(r["correct"] for r in rr)
                    w2c = sum(
                        (not base_lookup[int(r["sid"])]["baseline_correct"])
                        and bool(r["correct"])
                        for r in rr
                    )
                    c2w = sum(
                        base_lookup[int(r["sid"])]["baseline_correct"]
                        and (not bool(r["correct"]))
                        for r in rr
                    )
                    per_rel.append({
                        "bundle": bname,
                        "alpha": alpha,
                        "relation": rel_disp,
                        "N": len(rr),
                        "baseline_acc": base_acc,
                        "edited_acc": edit_acc,
                        "gain": edit_acc - base_acc,
                        "W2C": w2c,
                        "C2W": c2w,
                        "net": w2c - c2w,
                    })

        write_csv(outdir / "baseline.csv", eval_rows)
        write_csv(outdir / "resolved_gt_relation_tokens.csv", resolved_rows)
        write_csv(outdir / "generation_conditions.csv", cond_rows)
        write_csv(outdir / "summary.csv", summary)
        write_csv(outdir / "per_relation_summary.csv", per_rel)

        print("\n" + "=" * 132)
        print("RESULTS")
        print("=" * 132)
        base_acc = safe_mean(r["baseline_correct"] for r in eval_rows)
        print(f"Baseline: N={len(eval_rows)} acc={base_acc:.4f}")
        print(
            f"{'bundle':<20s} {'alpha':>7s} {'nEdit':>7s} "
            f"{'acc':>8s} {'gain':>8s} {'W2C':>5s} {'C2W':>5s} {'net':>5s}"
        )
        print("-" * 132)
        for r in summary:
            print(
                f"{r['bundle']:<20s} "
                f"{float(r['alpha']):>7.3f} "
                f"{float(r['mean_edited_positions']):>7.2f} "
                f"{float(r['edited_acc']):>8.4f} "
                f"{float(r['gain']):>+8.4f} "
                f"{int(r['W2C']):>5d} "
                f"{int(r['C2W']):>5d} "
                f"{int(r['net']):>5d}"
            )

        print("\nInterpretation:")
        print("  Large gain here -> correct relation-word token is a major causal carrier.")
        print("  Small gain here but large Top-K gain -> repair is distributed across other tokens.")
        print("  This experiment is still oracle because GT chooses which relation word to amplify.")

        (outdir / "metadata.json").write_text(
            json.dumps({
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "N": len(test),
                "layer_bundles": [list(b) for b in bundles],
                "alphas": alphas,
                "occurrence": a.occurrence,
                "gt_used_only_for_relation_token_choice": True,
                "late_writer_used": False,
                "gradient_used": False,
                "topk_used": False,
                "edit": "h_real[S,p] += alpha * (h_real[S,p] - h_gray[S,p])",
            }, indent=2),
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
