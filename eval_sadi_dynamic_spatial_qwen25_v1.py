#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_sadi_dynamic_spatial_qwen25_v1.py

SADI-style training-free dynamic intervention for AdaptVis Real-vs-Gray spatial setup.

No per-sample L/R/A/B selector, no cosine router, no external classifier.

TRAIN contrastive identification:
    Delta_i,l = h_last(real)_i,l - h_last(gray)_i,l
    D_l       = mean_i Delta_i,l

Select global top-K hidden dimensions across the chosen late layers:
    M[l,d] = 1 for top-K dimensions, else 0.

TEST dynamic intervention (SADI Eq.-5 style):
    A'_q,l = A_q,l + delta * (A_q,l * M_l)

Thus the direction/sign comes from the current sample's own activation rather
than selecting one of four fixed relation vectors.

Protocol:
  FIT: identify mask
  CAL: choose top-K and strength using actual model.generate()
  FULL TRAIN: refit mask with selected top-K
  TEST: evaluate once

Requires this file in the same repository directory as:
    eval_cosine_confidence_selector_qwen25_v1.py

Example:
CUDA_VISIBLE_DEVICES=0 python eval_sadi_dynamic_spatial_qwen25_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --prior-output-dir output/qwen3b_last_causal_auto_v1 \
  --cache-deltas output/qwen3b_real_gray_increment_similarity_v1/real_gray_last_deltas.npz \
  --layers auto \
  --topk-counts 64,128,256 \
  --strengths 0.5,1.0,2.0 \
  --mask-score positive \
  --output-dir output/qwen3b_sadi_dynamic_spatial_v1 \
  --overwrite

Quick fixed-parameter run:
  --fixed-topk 128 --fixed-strength 1.0

mask-score:
  positive : rank by signed mean Real-Gray difference D (closest to SADI)
  abs      : rank by |D| (diagnostic if opposite relations cancel in the mean)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import eval_cosine_confidence_selector_qwen25_v1 as base


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model-id", required=True)
    p.add_argument("--prior-output-dir", required=True)
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--annotation-json", default="data/coco_qa_two_obj.json")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument(
        "--layers",
        default="auto",
        help="auto = known late actuator window; otherwise e.g. 32-35",
    )
    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )
    p.add_argument(
        "--mask-score",
        default="positive",
        choices=["positive", "abs"],
    )
    p.add_argument(
        "--topk-counts",
        default="64,128,256",
        help="Global top-K hidden dimensions scanned on CAL.",
    )
    p.add_argument(
        "--strengths",
        default="0.5,1.0,2.0",
        help="SADI delta; selected dims become (1+delta)*activation.",
    )
    p.add_argument("--fixed-topk", type=int, default=0)
    p.add_argument("--fixed-strength", type=float, default=float("nan"))
    p.add_argument("--cal-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--cache-deltas",
        default="",
        help="Existing Real-Gray last-token npz cache; may contain extra layers.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(spec):
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def parse_floats(spec):
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def resolve_selected_layers(args, decoder_layers):
    if str(args.layers).strip().lower() == "auto":
        selected = list(base.model_preset(args.model_id)["actuator_layers"])
    else:
        selected = base.parse_layer_spec(args.layers)
    if not selected:
        raise ValueError("No layers selected.")
    bad = [x for x in selected if x < 0 or x >= len(decoder_layers)]
    if bad:
        raise ValueError(
            f"Invalid layers {bad}; decoder blocks are 0..{len(decoder_layers)-1}"
        )
    return sorted(set(selected))


def load_cached_subset(cache_path, wanted_layers, wanted_sids):
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        cached_layers = np.asarray(z["layers"], dtype=np.int64).tolist()
        deltas = np.asarray(z["deltas"], dtype=np.float32)

    layer_to_col = {int(layer): i for i, layer in enumerate(cached_layers)}
    missing_layers = [x for x in wanted_layers if x not in layer_to_col]
    if missing_layers:
        raise RuntimeError(
            f"Cache {path} lacks layers={missing_layers}; available={cached_layers}"
        )

    sid_to_row = {int(sid): i for i, sid in enumerate(sids.tolist())}
    missing_sids = [int(sid) for sid in wanted_sids if int(sid) not in sid_to_row]
    if missing_sids:
        raise RuntimeError(
            f"Cache {path} lacks {len(missing_sids)} requested TRAIN samples; "
            f"first={missing_sids[:10]}"
        )

    cols = [layer_to_col[x] for x in wanted_layers]
    result = {
        int(sid): deltas[sid_to_row[int(sid)]][cols].astype(np.float32)
        for sid in wanted_sids
    }
    print(
        f"[reuse] delta cache={path} | cached_layers={cached_layers} "
        f"-> selected_layers={wanted_layers} | TRAIN N={len(result)}"
    )
    return result


def build_train_delta_cache(
    model,
    processor,
    decoder_layers,
    records,
    train_sids,
    selected_layers,
    args,
    outdir,
):
    cached = load_cached_subset(args.cache_deltas, selected_layers, train_sids)
    if cached is not None:
        return cached

    train_records = {sid: records[sid] for sid in train_sids}
    cache_path = outdir / "train_real_gray_last_deltas.npz"
    return base.build_or_load_delta_cache(
        model=model,
        processor=processor,
        layers=decoder_layers,
        records=train_records,
        selected_layers=selected_layers,
        args=args,
        cache_path=cache_path,
    )


def allowed_sids(sids, train_generation, requested_filter):
    filter_sequence = {
        "real_correct_gray_wrong": [
            "real_correct_gray_wrong",
            "real_correct",
            "all",
        ],
        "real_correct": ["real_correct", "all"],
        "all": ["all"],
    }[requested_filter]

    for mode in filter_sequence:
        chosen = [
            sid
            for sid in sids
            if base.sample_allowed(sid, train_generation, mode)
        ]
        if chosen:
            return chosen, mode
    raise RuntimeError("No TRAIN samples available for SADI mask fitting.")


def fit_mean_difference(
    delta_cache,
    sids,
    selected_layers,
    train_generation,
    requested_filter,
):
    used_sids, filter_used = allowed_sids(
        sids, train_generation, requested_filter
    )
    stack = np.stack(
        [delta_cache[sid] for sid in used_sids], axis=0
    ).astype(np.float32)
    if stack.ndim != 3:
        raise RuntimeError(f"Unexpected delta stack shape={stack.shape}")
    if stack.shape[1] != len(selected_layers):
        raise RuntimeError(
            f"Delta stack layer dim={stack.shape[1]} "
            f"!= selected_layers={len(selected_layers)}"
        )
    mean_diff = stack.mean(axis=0).astype(np.float32)
    return mean_diff, used_sids, filter_used


def make_global_topk_mask(mean_diff, selected_layers, topk, mask_score):
    if mask_score == "positive":
        score = mean_diff.astype(np.float64)
    elif mask_score == "abs":
        score = np.abs(mean_diff.astype(np.float64))
    else:
        raise ValueError(mask_score)

    flat = score.reshape(-1)
    total = flat.size
    k = max(1, min(int(topk), total))
    chosen = np.argpartition(flat, total - k)[total - k:]

    mask_flat = np.zeros(total, dtype=np.float32)
    mask_flat[chosen] = 1.0
    mask = mask_flat.reshape(score.shape)

    layer_masks = {
        layer: mask[i].astype(np.float32)
        for i, layer in enumerate(selected_layers)
    }
    layer_counts = {
        layer: int(mask[i].sum())
        for i, layer in enumerate(selected_layers)
    }
    threshold = float(np.min(flat[chosen]))
    return layer_masks, layer_counts, threshold


class SADIAdaptiveLast:
    """A'_q = A_q + delta * (A_q * M), first/prefill pass only."""

    def __init__(self, decoder_layers, masks, selected_layers, strength):
        self.handles = []
        self.masks = masks
        self.strength = float(strength)
        self.done = {layer: False for layer in selected_layers}
        for layer in selected_layers:
            self.handles.append(
                decoder_layers[layer].register_forward_hook(self._hook(layer))
            )

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            if self.done[layer]:
                return output

            hidden, descriptor = base.extract_hidden(output)
            if hidden.ndim != 3:
                return output

            mask = torch.as_tensor(
                self.masks[layer], device=hidden.device, dtype=hidden.dtype
            )
            edited = hidden.clone()
            current = hidden[:, -1, :]
            dynamic_vector = current * mask[None, :]
            edited[:, -1, :] = current + self.strength * dynamic_vector
            self.done[layer] = True
            return base.replace_hidden(output, descriptor, edited)

        return hook

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_sadi_generation(
    model,
    processor,
    decoder_layers,
    masks,
    selected_layers,
    strength,
    records,
    sids,
    baseline_map,
    args,
    desc,
):
    rows = []
    for sid in tqdm(sids, desc=desc):
        rec = records[sid]
        image = None
        base_item = baseline_map.get(sid, {})
        baseline_pred = base_item.get("pred")
        baseline_correct = int(
            base_item.get("correct", baseline_pred == rec["gt"])
        )

        try:
            image = Image.open(rec["image_path"]).convert("RGB")
            batch = base.build_batch(processor, image, rec, args)

            with SADIAdaptiveLast(
                decoder_layers=decoder_layers,
                masks=masks,
                selected_layers=selected_layers,
                strength=strength,
            ):
                text, pred = base.generate(model, processor, batch, args)

            edit_correct = int(pred == rec["gt"])
            rows.append(
                {
                    "sid": sid,
                    "gt": rec["gt"],
                    "baseline_pred": baseline_pred or "",
                    "edit_pred": pred or "",
                    "baseline_correct": baseline_correct,
                    "edit_correct": edit_correct,
                    "W2C": int(baseline_correct == 0 and edit_correct == 1),
                    "C2W": int(baseline_correct == 1 and edit_correct == 0),
                    "changed": int((baseline_pred or "") != (pred or "")),
                    "edit_text": text,
                }
            )
            del batch
        finally:
            if image is not None:
                image.close()
            cleanup()
    return rows


def summarize(rows, condition, topk, strength):
    n = len(rows)
    base_acc = (
        sum(int(r["baseline_correct"]) for r in rows) / n
        if n
        else float("nan")
    )
    edit_acc = (
        sum(int(r["edit_correct"]) for r in rows) / n
        if n
        else float("nan")
    )
    w2c = sum(int(r["W2C"]) for r in rows)
    c2w = sum(int(r["C2W"]) for r in rows)
    return {
        "condition": condition,
        "N": n,
        "topk": int(topk),
        "strength": float(strength),
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "W2C": w2c,
        "C2W": c2w,
        "net": w2c - c2w,
        "changed": sum(int(r["changed"]) for r in rows),
    }


def choose_cal_best(rows):
    return max(
        rows,
        key=lambda r: (
            float(r["edit_acc"]),
            int(r["net"]),
            -int(r["C2W"]),
            -int(r["topk"]),
            -abs(float(r["strength"])),
        ),
    )


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    existing_test, baseline_path = base.load_existing_test_baseline(
        args.prior_output_dir
    )
    if existing_test is None:
        raise RuntimeError(
            "Could not reuse TEST baseline from --prior-output-dir."
        )

    train_generation, train_generation_path = (
        base.load_existing_train_generation(args.prior_output_dir)
    )

    records = base.load_all_records(args)
    train_sids, test_sids = base.derive_train_test_ids(records, existing_test)

    model, processor = base.load_model(args)
    decoder_layers, decoder_path = base.resolve_layers(model)
    selected_layers = resolve_selected_layers(args, decoder_layers)

    print("\n" + "=" * 138)
    print("SADI-STYLE DYNAMIC SPATIAL INTERVENTION — NO FOUR-WAY SELECTOR")
    print("=" * 138)
    print(f"model={args.model_id}")
    print(f"decoder={decoder_path} | blocks={len(decoder_layers)}")
    print(f"selected late layers={selected_layers}")
    print(f"TRAIN={len(train_sids)} TEST={len(test_sids)}")
    print(f"mask_score={args.mask_score}")
    print(f"template_filter={args.template_filter}")
    print(f"reused TEST baseline={baseline_path}")
    print(f"reused TRAIN Real/Gray labels={train_generation_path}")
    print("=" * 138)

    delta_cache = build_train_delta_cache(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        records=records,
        train_sids=train_sids,
        selected_layers=selected_layers,
        args=args,
        outdir=outdir,
    )

    fixed_mode = args.fixed_topk > 0 and math.isfinite(args.fixed_strength)

    if fixed_mode:
        selected_topk = int(args.fixed_topk)
        selected_strength = float(args.fixed_strength)
        full_mean_diff, used, filter_used = fit_mean_difference(
            delta_cache,
            train_sids,
            selected_layers,
            train_generation,
            args.template_filter,
        )
        final_masks, counts, threshold = make_global_topk_mask(
            full_mean_diff,
            selected_layers,
            selected_topk,
            args.mask_score,
        )
        print(
            f"\n[fixed] topk={selected_topk} strength={selected_strength} "
            f"| fit_N={len(used)} filter={filter_used} "
            f"threshold={threshold:.6g}"
        )
        print(f"[fixed] selected dims per layer={counts}")
        cal_summary_rows = []

    else:
        fit_sids, cal_sids = base.stratified_fit_cal_split(
            records=records,
            train_sids=train_sids,
            cal_frac=args.cal_frac,
            seed=args.seed,
        )
        print(f"\n[FIT/CAL] FIT={len(fit_sids)} CAL={len(cal_sids)}")

        fit_mean_diff, fit_used, fit_filter_used = fit_mean_difference(
            delta_cache,
            fit_sids,
            selected_layers,
            train_generation,
            args.template_filter,
        )
        print(
            f"[mask FIT] used_N={len(fit_used)} filter={fit_filter_used}"
        )

        cal_baseline = base.generate_baseline_for_sids(
            model=model,
            processor=processor,
            records=records,
            sids=cal_sids,
            args=args,
            desc="CAL baseline",
        )

        topk_counts = sorted(set(parse_ints(args.topk_counts)))
        strengths = sorted(set(parse_floats(args.strengths)))
        if not topk_counts:
            raise ValueError("No --topk-counts.")
        if not strengths:
            raise ValueError("No --strengths.")

        cal_summary_rows, cal_detail_rows = [], []

        for topk in topk_counts:
            masks, counts, threshold = make_global_topk_mask(
                fit_mean_diff,
                selected_layers,
                topk,
                args.mask_score,
            )
            print(
                f"\n[mask] topk={topk} threshold={threshold:.6g} "
                f"per_layer={counts}"
            )

            for strength in strengths:
                condition = f"cal_sadi_k{topk}_delta{strength:g}"
                rows = run_sadi_generation(
                    model,
                    processor,
                    decoder_layers,
                    masks,
                    selected_layers,
                    strength,
                    records,
                    cal_sids,
                    cal_baseline,
                    args,
                    condition,
                )

                for row in rows:
                    row["condition"] = condition
                    row["topk"] = topk
                    row["strength"] = strength
                    cal_detail_rows.append(row)

                summary = summarize(rows, condition, topk, strength)
                summary["mask_threshold"] = threshold
                cal_summary_rows.append(summary)

                print(
                    f"{condition:30s} | "
                    f"{summary['base_acc']:.4f}->{summary['edit_acc']:.4f} "
                    f"{summary['gain']:+.4f} | "
                    f"W2C={summary['W2C']} C2W={summary['C2W']} "
                    f"net={summary['net']:+d}"
                )

        write_csv(outdir / "cal_sadi_summary.csv", cal_summary_rows)
        write_csv(outdir / "cal_sadi_details.csv", cal_detail_rows)

        best = choose_cal_best(cal_summary_rows)
        selected_topk = int(best["topk"])
        selected_strength = float(best["strength"])

        print("\nSELECTED ON CAL")
        print("-" * 138)
        print(
            f"topk={selected_topk} | strength={selected_strength:g} | "
            f"CAL {float(best['base_acc']):.4f}->{float(best['edit_acc']):.4f} "
            f"{float(best['gain']):+.4f} | "
            f"W2C={int(best['W2C'])} C2W={int(best['C2W'])} "
            f"net={int(best['net']):+d}"
        )

        full_mean_diff, full_used, full_filter_used = fit_mean_difference(
            delta_cache,
            train_sids,
            selected_layers,
            train_generation,
            args.template_filter,
        )
        final_masks, counts, threshold = make_global_topk_mask(
            full_mean_diff,
            selected_layers,
            selected_topk,
            args.mask_score,
        )
        print(
            f"[refit FULL TRAIN] used_N={len(full_used)} "
            f"filter={full_filter_used} | threshold={threshold:.6g}"
        )
        print(f"[final mask] selected dims per layer={counts}")

    mask_rows = []
    for layer in selected_layers:
        indices = np.flatnonzero(final_masks[layer] > 0).tolist()
        mask_rows.append(
            {
                "layer": layer,
                "selected_count": len(indices),
                "selected_indices": ",".join(map(str, indices)),
            }
        )
    write_csv(outdir / "final_sadi_mask.csv", mask_rows)

    np.savez_compressed(
        outdir / "final_sadi_mask.npz",
        layers=np.asarray(selected_layers, dtype=np.int64),
        masks=np.stack(
            [final_masks[layer] for layer in selected_layers], axis=0
        ).astype(np.float32),
        topk=np.asarray([selected_topk], dtype=np.int64),
        strength=np.asarray([selected_strength], dtype=np.float32),
        mean_difference=full_mean_diff.astype(np.float32),
    )

    test_baseline = base.prepare_test_baseline(
        existing=existing_test,
        records=records,
        test_sids=test_sids,
    )
    if test_baseline is None:
        test_baseline = base.generate_baseline_for_sids(
            model=model,
            processor=processor,
            records=records,
            sids=test_sids,
            args=args,
            desc="TEST baseline fallback",
        )

    test_rows = run_sadi_generation(
        model,
        processor,
        decoder_layers,
        final_masks,
        selected_layers,
        selected_strength,
        records,
        test_sids,
        test_baseline,
        args,
        "TEST SADI dynamic steering",
    )

    test_summary = summarize(
        test_rows,
        "test_sadi_dynamic",
        selected_topk,
        selected_strength,
    )
    test_summary["mask_score"] = args.mask_score
    test_summary["layers"] = ",".join(map(str, selected_layers))

    write_csv(outdir / "test_sadi_summary.csv", [test_summary])
    write_csv(outdir / "test_sadi_details.csv", test_rows)

    print("\n" + "=" * 138)
    print("ACTUAL model.generate() — SADI DYNAMIC SPATIAL INTERVENTION")
    print("=" * 138)
    print(
        f"layers={selected_layers} | topk={selected_topk} | "
        f"strength={selected_strength:g} | mask={args.mask_score}"
    )
    print(
        f"acc {test_summary['base_acc']:.4f}->{test_summary['edit_acc']:.4f} "
        f"{test_summary['gain']:+.4f} | "
        f"W2C={test_summary['W2C']} C2W={test_summary['C2W']} "
        f"net={test_summary['net']:+d} | "
        f"changed={test_summary['changed']}"
    )
    print("=" * 138)
    print(f"[saved] {outdir / 'cal_sadi_summary.csv'}")
    print(f"[saved] {outdir / 'test_sadi_summary.csv'}")
    print(f"[saved] {outdir / 'test_sadi_details.csv'}")
    print(f"[saved] {outdir / 'final_sadi_mask.npz'}")


if __name__ == "__main__":
    main()
