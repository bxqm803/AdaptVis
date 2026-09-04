#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_real_gray_increment_similarity_v1.py

Training-free diagnostic:
  Delta_i,l = h_last(real)_i,l - h_last(gray)_i,l

Use TRAIN-derived Real-Gray relation vectors:
  mu_r,l      = mean(Delta_i,l | relation=r)
  mu_global,l = balanced mean_r(mu_r,l)
  s_r,l       = mu_r,l - mu_global,l

Then test on TEST whether the sample's own Real-Gray increment selects the
correct relation by cosine similarity. No selector classifier is trained and
no steering generate() is run.

Modes:
  centered_shared:
      cos(Delta_i,l - mu_global,l, s_r,l)
  raw_mu:
      cos(Delta_i,l, mu_r,l)
  raw_shared:
      cos(Delta_i,l, s_r,l)

This script reuses utilities from:
  eval_cosine_confidence_selector_qwen25_v1.py

NOTE:
The best TEST window printed here is diagnostic only. If this works, fix/select
the window on TRAIN/CAL and then evaluate once on TEST.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np

import eval_cosine_confidence_selector_qwen25_v1 as base

RELS = base.RELS
RELSET = set(RELS)
EPS = 1e-12


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
        help="e.g. 18-35, 28-35, 32-35, all; auto: 3B=18-35, 7B=14-27",
    )
    p.add_argument("--window-max", type=int, default=4)
    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--cache-deltas", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def get_auto_layers(model_id):
    low = model_id.lower()
    if "3b" in low:
        return list(range(18, 36))
    if "7b" in low:
        return list(range(14, 28))
    raise ValueError("For non-Qwen3B/7B, pass --layers explicitly.")


def resolve_requested_layers(spec, n_layers, model_id):
    spec = str(spec).strip().lower()
    if spec == "all":
        layers = list(range(n_layers))
    elif spec == "auto":
        layers = get_auto_layers(model_id)
    else:
        layers = base.parse_layer_spec(spec)
    bad = [x for x in layers if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Invalid layers={bad}; model has 0..{n_layers-1}")
    return sorted(set(layers))


def cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_sample(delta_cache, layer_index, templates, sid, window, mode):
    scores = {r: [] for r in RELS}
    for layer in window:
        delta = delta_cache[sid][layer_index[layer]].astype(np.float64)
        global_mu = templates[layer]["global"].astype(np.float64)

        if mode == "centered_shared":
            query = delta - global_mu
        elif mode in ("raw_mu", "raw_shared"):
            query = delta
        else:
            raise ValueError(mode)

        for r in RELS:
            shared = templates[layer]["shared"][r].astype(np.float64)
            proto = global_mu + shared if mode == "raw_mu" else shared
            scores[r].append(cos(query, proto))

    agg = {r: float(np.mean(scores[r])) for r in RELS}
    ordered = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    pred = ordered[0][0]
    margin = float(ordered[0][1] - ordered[1][1])
    return pred, margin, agg


def candidate_windows(layers, window_max):
    allowed = set(layers)
    out = []
    for length in range(1, max(1, window_max) + 1):
        for start in layers:
            w = tuple(range(start, start + length))
            if all(x in allowed for x in w):
                out.append(w)
    return out


def baseline_info(existing_test, records, sid):
    pred = existing_test.get(sid, {}).get("pred")
    correct = int(pred in RELSET and pred == records[sid]["gt"])
    return pred, correct


def evaluate_window(
    delta_cache, layer_index, templates, records,
    test_sids, existing_test, window, mode,
):
    details = []
    for sid in test_sids:
        pred, margin, scores = score_sample(
            delta_cache, layer_index, templates, sid, window, mode
        )
        gt = records[sid]["gt"]
        base_pred, base_correct = baseline_info(existing_test, records, sid)
        details.append({
            "sid": sid,
            "gt": gt,
            "baseline_pred": base_pred,
            "baseline_correct": base_correct,
            "selector_pred": pred,
            "selector_correct": int(pred == gt),
            "selector_vs_baseline_conflict": int(
                pred in RELSET and base_pred in RELSET and pred != base_pred
            ),
            "margin": margin,
            **{f"score_{r}": scores[r] for r in RELS},
        })

    wrong = [x for x in details if x["baseline_correct"] == 0]
    correct = [x for x in details if x["baseline_correct"] == 1]
    conflict = [x for x in details if x["selector_vs_baseline_conflict"] == 1]
    conflict_wrong = [x for x in conflict if x["baseline_correct"] == 0]

    row = {
        "mode": mode,
        "layers": ",".join(map(str, window)),
        "length": len(window),
        "N": len(details),
        "selector_acc": safe_mean(x["selector_correct"] for x in details),
        "mean_margin": safe_mean(x["margin"] for x in details),
        "baseline_wrong_N": len(wrong),
        "selector_acc_on_baseline_wrong": safe_mean(
            x["selector_correct"] for x in wrong
        ),
        "baseline_correct_N": len(correct),
        "selector_acc_on_baseline_correct": safe_mean(
            x["selector_correct"] for x in correct
        ),
        "conflict_N": len(conflict),
        "conflict_rate": len(conflict) / len(details) if details else float("nan"),
        "selector_acc_on_conflict": safe_mean(
            x["selector_correct"] for x in conflict
        ),
        "conflict_baseline_wrong_N": len(conflict_wrong),
        "selector_acc_on_conflict_baseline_wrong": safe_mean(
            x["selector_correct"] for x in conflict_wrong
        ),
    }
    return row, details


def confusion_rows(details, mode, window):
    rows = []
    for gt in RELS:
        subset = [x for x in details if x["gt"] == gt]
        row = {
            "mode": mode,
            "layers": ",".join(map(str, window)),
            "gt": gt,
            "N": len(subset),
        }
        for pred in RELS:
            row[f"pred_{pred}"] = sum(
                int(x["selector_pred"] == pred) for x in subset
            )
        rows.append(row)
    return rows


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
            "Cannot recover TEST split. Point --prior-output-dir to the earlier "
            "causal-steering output containing test_baseline.csv."
        )

    train_generation, train_gen_path = base.load_existing_train_generation(
        args.prior_output_dir
    )
    records = base.load_all_records(args)
    train_sids, test_sids = base.derive_train_test_ids(records, existing_test)

    print("\n" + "=" * 130)
    print("REAL-GRAY LAST-TOKEN INCREMENT SIMILARITY")
    print("=" * 130)
    print(f"model={args.model_id}")
    print(f"TRAIN={len(train_sids)} TEST={len(test_sids)}")
    print(f"baseline={baseline_path}")
    print(f"train_real_gray_labels={train_gen_path}")
    print(f"template_filter={args.template_filter}")

    model, processor = base.load_model(args)
    decoder_layers, layer_path = base.resolve_layers(model)
    selected_layers = resolve_requested_layers(
        args.layers, len(decoder_layers), args.model_id
    )
    layer_index = {layer: i for i, layer in enumerate(selected_layers)}

    print(f"decoder={layer_path} | blocks={len(decoder_layers)}")
    print(f"tested_layers={selected_layers}")

    cache_path = (
        Path(args.cache_deltas)
        if args.cache_deltas
        else outdir / "real_gray_last_deltas.npz"
    )
    delta_cache = base.build_or_load_delta_cache(
        model=model,
        processor=processor,
        layers=decoder_layers,
        records=records,
        selected_layers=selected_layers,
        args=args,
        cache_path=cache_path,
    )

    # Same TRAIN-derived relation vectors as the causal actuator family.
    # No external selector is trained here.
    templates, filter_used = base.fit_templates(
        delta_cache=delta_cache,
        layer_index=layer_index,
        records=records,
        sids=train_sids,
        selected_layers=selected_layers,
        train_generation=train_generation,
        requested_filter=args.template_filter,
    )
    print(f"template_filter_used={filter_used}")

    modes = ("centered_shared", "raw_mu", "raw_shared")
    scan_rows = []
    best = {}

    for mode in modes:
        candidates = []
        details_bank = {}

        for window in candidate_windows(selected_layers, args.window_max):
            row, details = evaluate_window(
                delta_cache, layer_index, templates, records,
                test_sids, existing_test, window, mode,
            )
            scan_rows.append(row)
            candidates.append(row)
            details_bank[tuple(window)] = details

        best_row = max(
            candidates,
            key=lambda r: (
                float(r["selector_acc"]),
                float(r["selector_acc_on_baseline_wrong"])
                if math.isfinite(float(r["selector_acc_on_baseline_wrong"]))
                else -1.0,
                float(r["mean_margin"]),
                -int(r["length"]),
            ),
        )
        best_window = tuple(int(x) for x in best_row["layers"].split(","))
        best[mode] = (best_row, best_window, details_bank[best_window])

    write_csv(outdir / "test_similarity_scan.csv", scan_rows)

    summary_rows = []
    conf_rows = []
    for mode in modes:
        row, window, details = best[mode]
        out = dict(row)
        out["NOTE"] = "best TEST window is diagnostic only"
        summary_rows.append(out)
        write_csv(outdir / f"test_best_{mode}_predictions.csv", details)
        conf_rows.extend(confusion_rows(details, mode, window))

    write_csv(outdir / "test_best_similarity_summary.csv", summary_rows)
    write_csv(outdir / "test_best_confusion.csv", conf_rows)

    # Also report the already-known actuator window if it is inside tested layers.
    try:
        actuator_window = tuple(base.model_preset(args.model_id)["actuator_layers"])
    except Exception:
        actuator_window = tuple()

    print("\nRESULT")
    print("-" * 130)
    for mode in modes:
        row, _, _ = best[mode]
        print(
            f"{mode:18s} | best_window={row['layers']:>14s} "
            f"| acc={float(row['selector_acc']):.4f} "
            f"| wrong_acc={float(row['selector_acc_on_baseline_wrong']):.4f} "
            f"| correct_acc={float(row['selector_acc_on_baseline_correct']):.4f} "
            f"| conflictN={int(row['conflict_N']):3d} "
            f"| conflict_wrong_acc="
            f"{float(row['selector_acc_on_conflict_baseline_wrong']):.4f}"
        )

        if actuator_window and all(x in selected_layers for x in actuator_window):
            target = ",".join(map(str, actuator_window))
            hits = [
                r for r in scan_rows
                if r["mode"] == mode and r["layers"] == target
            ]
            if hits:
                r = hits[0]
                print(
                    f"{'':18s} | actuator={target:>17s} "
                    f"| acc={float(r['selector_acc']):.4f} "
                    f"| wrong_acc={float(r['selector_acc_on_baseline_wrong']):.4f} "
                    f"| correct_acc={float(r['selector_acc_on_baseline_correct']):.4f}"
                )

    print("-" * 130)
    print(f"[saved] {outdir / 'test_similarity_scan.csv'}")
    print(f"[saved] {outdir / 'test_best_similarity_summary.csv'}")
    print(f"[saved] {outdir / 'test_best_confusion.csv'}")
    print(
        "\nPriority:\n"
        "  centered_shared = (Delta - mu_global) vs s_r   [main]\n"
        "  raw_mu          = Delta vs mean Delta_r        [literal increment similarity]\n"
        "  raw_shared      = Delta vs s_r\n"
    )


if __name__ == "__main__":
    main()
