#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-3B + COCO random-map visualization:
Does the SAME prompt-final Real-Gray hidden state change from relation-organized
geometry in middle layers to answer-option-organized geometry in late layers?

This is a visualization / diagnostic script, not a causal intervention script.

Protocol
--------
1) COCO two-object samples.
2) Per-example randomized, balanced relation -> A/B/C/D mapping.
3) Same prompt format and mapping protocol as
   eval_qwen_coco_randmap_late_spatial_deconfound_quick.py.
4) At every requested decoder layer capture the SAME state:
       q_i,l = h_real[i,l,last] - h_gray[i,l,last]
5) Run actual generation on held-out TEST samples.
6) Main visualization uses only correctly generated TEST samples.
7) Quantify relation-vs-option organization at every layer using TRAIN-fitted
   centroids, evaluated on held-out TEST:
       - relation nearest-centroid accuracy
       - option nearest-centroid accuracy
       - between/within geometry ratio
8) Automatically choose:
       middle layer = max(relation_acc - option_acc) on correct TEST samples
       late layer   = max(option_acc - relation_acc) on correct TEST samples
9) PCA figure:
       A) middle q_last, colored by GT relation
       B) late q_last, colored by GT relation
       C) EXACT SAME late PCA coordinates as B, colored by correct A/B/C/D option

Important
---------
Panels B and C use exactly the same points / PCA projection. Only the labels
used for coloring change.

Default run
-----------
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u qwen3b_coco_randmap_relation_to_option_pca_v1.py \
  --max-samples 0 \
  --eval-max-samples 0 \
  --output-dir output/qwen3b_coco_randmap_relation_to_option_pca_v1 \
  --overwrite

Quick preview
-------------
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u qwen3b_coco_randmap_relation_to_option_pca_v1.py \
  --max-samples 240 \
  --eval-max-samples 120 \
  --output-dir output/qwen3b_coco_randmap_relation_to_option_pca_quick \
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
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

# Reuse the repository's already-debugged random-mapping / COCO / model helpers.
import eval_qwen_coco_randmap_late_spatial_deconfound_quick as rm

base = rm.base
traj = rm.traj
REL = rm.REL
LETTERS = rm.LETTERS


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument(
        "--layers",
        default="all",
        help="all = every decoder block; otherwise comma list such as 12,16,20,24,28,32,35",
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all COCO-two samples before train/test split",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all held-out TEST samples",
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--pca-balance",
        default="relation_option",
        choices=["relation_option", "none"],
        help="Balance the PCA visualization over relation x correct-option cells when possible.",
    )
    p.add_argument(
        "--pca-max-per-cell",
        type=int,
        default=0,
        help="0 = use the maximum balanced count available; otherwise cap each relation-option cell.",
    )
    p.add_argument(
        "--middle-layer",
        type=int,
        default=-1,
        help="-1 = auto choose max(correct relation_acc - option_acc)",
    )
    p.add_argument(
        "--late-layer",
        type=int,
        default=-1,
        help="-1 = auto choose max(correct option_acc - relation_acc)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_layers(text: str, n_layers: int) -> List[int]:
    if text.strip().lower() == "all":
        return list(range(n_layers))
    out = sorted(
        {
            int(x.strip().upper().replace("L", ""))
            for x in text.split(",")
            if x.strip()
        }
    )
    bad = [x for x in out if not (0 <= x < n_layers)]
    if bad:
        raise ValueError(f"Invalid layers {bad}; model has L0-L{n_layers-1}")
    return out


def geometry_ratio(X: np.ndarray, y: List[str]) -> float:
    """Between-class / within-class squared variance ratio."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 2:
        return float("nan")
    grand = X.mean(axis=0)
    between_num = 0.0
    between_den = 0
    within_num = 0.0
    within_den = 0
    for cls in sorted(set(y.tolist())):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        Xc = X[idx]
        mu = Xc.mean(axis=0)
        between_num += len(idx) * float(np.sum((mu - grand) ** 2))
        between_den += len(idx)
        within_num += float(np.sum((Xc - mu) ** 2))
        within_den += len(idx)
    between = between_num / max(between_den, 1)
    within = within_num / max(within_den, 1)
    return float(between / max(within, 1e-12))


def nearest_centroid(x: np.ndarray, centroids: Dict[str, np.ndarray]) -> str:
    labels = list(centroids)
    vals = [rm.cosine_np(x, centroids[k]) for k in labels]
    vals = [-1e30 if not np.isfinite(v) else v for v in vals]
    return labels[int(np.argmax(vals))]


def fit_label_centroids(items, maps, q_by_sid, layers):
    rel_cent = {L: {} for L in layers}
    opt_cent = {L: {} for L in layers}
    for L in layers:
        for r in REL:
            xs = [
                q_by_sid[int(m["sid"])][L]
                for m in items
                if int(m["sid"]) in q_by_sid and m["gt"] == r
            ]
            if not xs:
                raise RuntimeError(f"No training states for relation={r} L{L}")
            rel_cent[L][r] = np.mean(np.stack(xs), axis=0).astype(np.float32)
        for a in LETTERS:
            xs = [
                q_by_sid[int(m["sid"])][L]
                for m in items
                if int(m["sid"]) in q_by_sid
                and maps[int(m["sid"])][m["gt"]] == a
            ]
            if not xs:
                raise RuntimeError(f"No training states for option={a} L{L}")
            opt_cent[L][a] = np.mean(np.stack(xs), axis=0).astype(np.float32)
    return rel_cent, opt_cent


def pca_2d(X: np.ndarray):
    """Plain PCA via numpy SVD; no sklearn dependency."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    if len(X) < 3:
        raise RuntimeError("Need at least 3 samples for PCA")
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    ev = S ** 2
    evr = ev[:2] / max(ev.sum(), 1e-12)
    return coords.astype(np.float32), evr.astype(np.float64)


def choose_pca_ids(correct_rows, maps, mode, max_per_cell, seed):
    ids = [int(r["sid"]) for r in correct_rows]
    if mode == "none":
        return ids, {"mode": "none", "n": len(ids)}

    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in correct_rows:
        sid = int(r["sid"])
        rel = str(r["gt"])
        opt = maps[sid][rel]
        cells[(rel, opt)].append(sid)

    all_cells = [(r, a) for r in REL for a in LETTERS]
    counts = {f"{r}_{a}": len(cells[(r, a)]) for r, a in all_cells}
    min_n = min(len(cells[c]) for c in all_cells)

    if min_n <= 0:
        return ids, {
            "mode": "fallback_all_correct_empty_cell",
            "n": len(ids),
            "cell_counts": counts,
        }

    take = min_n
    if max_per_cell > 0:
        take = min(take, max_per_cell)

    chosen = []
    for c in all_cells:
        vals = list(cells[c])
        rng.shuffle(vals)
        chosen.extend(vals[:take])

    rng.shuffle(chosen)
    return chosen, {
        "mode": "relation_option",
        "n_per_cell": take,
        "n": len(chosen),
        "cell_counts_before_balance": counts,
    }


def plot_layer_curves(rows: List[dict], out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [int(r["layer"]) for r in rows]
    rel_c = [float(r["correct_relation_decode_acc"]) for r in rows]
    opt_c = [float(r["correct_option_decode_acc"]) for r in rows]
    rel_g = [float(r["correct_relation_geometry_ratio"]) for r in rows]
    opt_g = [float(r["correct_option_geometry_ratio"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(layers, rel_c, marker="o", label="Relation decode (correct samples)")
    ax.plot(layers, opt_c, marker="o", label="Option decode (correct samples)")
    ax.axhline(0.25, linestyle=":", linewidth=1, label="4-way chance")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Held-out nearest-centroid accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Prompt-final Real-Gray state: relation vs answer identity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(layers, rel_g, marker="o", label="Relation geometry")
    ax.plot(layers, opt_g, marker="o", label="Option geometry")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Between / within class variance")
    ax.set_title("Geometry organization of the same prompt-final state")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png.with_name(out_png.stem + "_geometry.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pca_triptych(q_by_sid, rows_by_sid, maps, ids, middle_layer, late_layer, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Xm = np.stack([q_by_sid[sid][middle_layer] for sid in ids], axis=0)
    Xl = np.stack([q_by_sid[sid][late_layer] for sid in ids], axis=0)

    Cm, evm = pca_2d(Xm)
    Cl, evl = pca_2d(Xl)

    rel_labels = [rows_by_sid[sid]["gt"] for sid in ids]
    opt_labels = [maps[sid][rows_by_sid[sid]["gt"]] for sid in ids]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    for r in REL:
        idx = [i for i, x in enumerate(rel_labels) if x == r]
        axes[0].scatter(Cm[idx, 0], Cm[idx, 1], s=24, alpha=0.72, label=r)
    axes[0].set_title(
        f"Middle L{middle_layer}: relation\n"
        f"PCA var={100*evm[0]:.1f}%+{100*evm[1]:.1f}%"
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False, fontsize=8)

    # EXACT SAME late coordinates in panels B and C.
    for r in REL:
        idx = [i for i, x in enumerate(rel_labels) if x == r]
        axes[1].scatter(Cl[idx, 0], Cl[idx, 1], s=24, alpha=0.72, label=r)
    axes[1].set_title(
        f"Late L{late_layer}: relation\n"
        f"PCA var={100*evl[0]:.1f}%+{100*evl[1]:.1f}%"
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(frameon=False, fontsize=8)

    for aa in LETTERS:
        idx = [i for i, x in enumerate(opt_labels) if x == aa]
        axes[2].scatter(Cl[idx, 0], Cl[idx, 1], s=24, alpha=0.72, label=aa)
    axes[2].set_title(f"Same late L{late_layer} points: option")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Correctly generated samples only (N={len(ids)}): relation-to-option geometry",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    point_rows = []
    for i, sid in enumerate(ids):
        point_rows.append({
            "sid": sid,
            "gt_relation": rel_labels[i],
            "correct_option": opt_labels[i],
            "middle_layer": middle_layer,
            "middle_pc1": float(Cm[i, 0]),
            "middle_pc2": float(Cm[i, 1]),
            "late_layer": late_layer,
            "late_pc1": float(Cl[i, 0]),
            "late_pc2": float(Cl[i, 1]),
        })
    return point_rows, evm, evl


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    err_path = out / "errors.jsonl"

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for r in records:
        sid = int(r.sid)
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
        })

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, test = traj.stratified_split(meta, a.train_ratio, a.seed)
    test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)

    train_map = rm.assign_relation_balanced_mappings(train, a.seed + 1001)
    test_map = rm.assign_relation_balanced_mappings(test, a.seed + 2003)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": a.device},
    }
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    model = processor = None
    q_by_sid: Dict[int, Dict[int, np.ndarray]] = {}
    baseline_rows: List[dict] = []

    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        if getattr(model, "generation_config", None) is not None:
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = model.generation_config.eos_token_id

        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        layers = parse_layers(a.layers, len(decoder_layers))

        print("=" * 120)
        print("QWEN3B + COCO RANDOM-MAP RELATION -> OPTION GEOMETRY")
        print("=" * 120)
        print(f"decoder={decoder_path} | n_layers={len(decoder_layers)}")
        print(f"captured layers={layers}")
        print(f"train/test={len(train)}/{len(test)}")
        print("state at EVERY layer: q_last = h_real(prompt-last) - h_gray(prompt-last)")
        print("main PCA subset: final generation CORRECT only")
        print()

        mapping_rows = []
        for split_name, items, maps in (("train", train, train_map), ("test", test, test_map)):
            for r in REL:
                for opt in LETTERS:
                    mapping_rows.append({
                        "split": split_name,
                        "relation": r,
                        "correct_option": opt,
                        "n": sum(
                            1 for m in items
                            if m["gt"] == r and maps[int(m["sid"])][r] == opt
                        ),
                    })
        write_csv(out / "mapping_balance.csv", mapping_rows)

        # Capture same prompt-final Real-Gray state at every layer.
        for split_name, items, maps in (("TRAIN", train, train_map), ("TEST", test, test_map)):
            test_mode = split_name == "TEST"
            for m in tqdm(items, desc=f"{split_name} q_last Real-Gray"):
                sid = int(m["sid"])
                mp = maps[sid]
                prompt = rm.build_randmap_prompt(m["subject"], m["reference"], mp)

                real = gray = rb = gb = None
                try:
                    real = rm.make_real_image(rec_by_sid[sid])
                    gray = rm.make_gray_image(real, a.gray_value)
                    rb = rm.make_batch(processor, device, real, prompt)
                    gb = rm.make_batch(processor, device, gray, prompt)

                    hr = rm.capture_prefix_last(model, decoder_layers, rb, layers)
                    hg = rm.capture_prefix_last(model, decoder_layers, gb, layers)
                    q_by_sid[sid] = {
                        L: (hr[L] - hg[L]).astype(np.float32)
                        for L in layers
                    }

                    if test_mode:
                        pred, text, parse_mode, _ = rm.generate_one(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            mapping=mp,
                            max_new_tokens=a.max_new_tokens,
                        )
                        correct_option = mp[m["gt"]]
                        baseline_rows.append({
                            "sid": sid,
                            "gt": m["gt"],
                            "correct_option": correct_option,
                            "pred_option": pred,
                            "baseline_correct": pred == correct_option,
                            "parse_mode": parse_mode,
                            "text": text,
                            "mapping": rm.mapping_string(mp),
                        })

                except Exception as e:
                    traj.append_jsonl(
                        err_path,
                        {
                            "phase": split_name.lower(),
                            "sid": sid,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise
                finally:
                    for im in (real, gray):
                        if im is not None:
                            with contextlib.suppress(Exception):
                                im.close()
                    if rb is not None:
                        del rb
                    if gb is not None:
                        del gb
                    gc.collect()

        write_csv(out / "baseline.csv", baseline_rows)

        train = [m for m in train if int(m["sid"]) in q_by_sid]
        test = [m for m in test if int(m["sid"]) in q_by_sid]
        test_meta_by_sid = {int(m["sid"]): m for m in test}
        baseline_by_sid = {
            int(r["sid"]): r
            for r in baseline_rows
            if int(r["sid"]) in test_meta_by_sid
        }

        valid_test = [m for m in test if int(m["sid"]) in baseline_by_sid]
        correct_rows = [
            baseline_by_sid[int(m["sid"])]
            for m in valid_test
            if bool(baseline_by_sid[int(m["sid"])]["baseline_correct"])
        ]

        N = len(valid_test)
        Nc = len(correct_rows)
        print(f"\nRandom-map generation: N={N} correct={Nc} acc={Nc/max(N,1):.4f}")
        print("Correct relation counts:", dict(Counter(r["gt"] for r in correct_rows)))
        print("Correct option counts:", dict(Counter(r["correct_option"] for r in correct_rows)))

        if Nc < 16:
            raise RuntimeError(f"Only {Nc} correct TEST samples; too few for a useful 4x4 visualization.")

        rel_cent, opt_cent = fit_label_centroids(train, train_map, q_by_sid, layers)

        # Held-out readout + geometry at every layer.
        layer_rows = []
        correct_sids = {int(r["sid"]) for r in correct_rows}

        for L in layers:
            all_rel_ok = all_opt_ok = 0
            cor_rel_ok = cor_opt_ok = 0
            all_rel_y, all_opt_y, all_X = [], [], []
            cor_rel_y, cor_opt_y, cor_X = [], [], []

            for m in valid_test:
                sid = int(m["sid"])
                q = q_by_sid[sid][L]
                gt = m["gt"]
                opt = test_map[sid][gt]

                pr = nearest_centroid(q, rel_cent[L])
                po = nearest_centroid(q, opt_cent[L])

                all_rel_ok += int(pr == gt)
                all_opt_ok += int(po == opt)
                all_X.append(q)
                all_rel_y.append(gt)
                all_opt_y.append(opt)

                if sid in correct_sids:
                    cor_rel_ok += int(pr == gt)
                    cor_opt_ok += int(po == opt)
                    cor_X.append(q)
                    cor_rel_y.append(gt)
                    cor_opt_y.append(opt)

            Xall = np.stack(all_X)
            Xcor = np.stack(cor_X)

            row = {
                "layer": L,
                "N_all": len(all_X),
                "all_relation_decode_acc": all_rel_ok / max(len(all_X), 1),
                "all_option_decode_acc": all_opt_ok / max(len(all_X), 1),
                "all_relation_geometry_ratio": geometry_ratio(Xall, all_rel_y),
                "all_option_geometry_ratio": geometry_ratio(Xall, all_opt_y),
                "N_correct": len(cor_X),
                "correct_relation_decode_acc": cor_rel_ok / max(len(cor_X), 1),
                "correct_option_decode_acc": cor_opt_ok / max(len(cor_X), 1),
                "correct_relation_geometry_ratio": geometry_ratio(Xcor, cor_rel_y),
                "correct_option_geometry_ratio": geometry_ratio(Xcor, cor_opt_y),
            }
            row["correct_relation_minus_option_acc"] = (
                row["correct_relation_decode_acc"] - row["correct_option_decode_acc"]
            )
            row["correct_option_minus_relation_acc"] = -row["correct_relation_minus_option_acc"]
            layer_rows.append(row)

            print(
                f"L{L:02d} | correct N={len(cor_X):3d} "
                f"rel={row['correct_relation_decode_acc']:.3f} "
                f"opt={row['correct_option_decode_acc']:.3f} "
                f"delta(rel-opt)={row['correct_relation_minus_option_acc']:+.3f} | "
                f"geom rel/opt={row['correct_relation_geometry_ratio']:.3g}/{row['correct_option_geometry_ratio']:.3g}"
            )

        write_csv(out / "layerwise_relation_vs_option.csv", layer_rows)
        plot_layer_curves(layer_rows, out / "layerwise_relation_vs_option_correct.png")

        if a.middle_layer >= 0:
            middle_layer = int(a.middle_layer)
        else:
            middle_layer = max(
                layer_rows,
                key=lambda r: float(r["correct_relation_minus_option_acc"]),
            )["layer"]

        if a.late_layer >= 0:
            late_layer = int(a.late_layer)
        else:
            layer_candidates = [r for r in layer_rows if int(r["layer"]) != int(middle_layer)]
            late_layer = max(
                layer_candidates,
                key=lambda r: float(r["correct_option_minus_relation_acc"]),
            )["layer"]

        if middle_layer not in layers or late_layer not in layers:
            raise ValueError(f"Selected middle/late layer not captured: {middle_layer}, {late_layer}")

        print(f"\nSelected visualization layers: middle=L{middle_layer}, late=L{late_layer}")

        pca_ids, pca_balance_info = choose_pca_ids(
            correct_rows=correct_rows,
            maps=test_map,
            mode=a.pca_balance,
            max_per_cell=a.pca_max_per_cell,
            seed=a.seed + 9901,
        )

        point_rows, evm, evl = plot_pca_triptych(
            q_by_sid=q_by_sid,
            rows_by_sid=baseline_by_sid,
            maps=test_map,
            ids=pca_ids,
            middle_layer=middle_layer,
            late_layer=late_layer,
            out_png=out / "pca_correct_relation_to_option.png",
        )
        write_csv(out / "pca_correct_points.csv", point_rows)

        metadata = {
            "model_alias": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "captured_layers": layers,
            "state_definition": "q_last = h_real(prompt-final) - h_gray(prompt-final)",
            "random_mapping": "per-example balanced relation -> A/B/C/D",
            "train_ratio": a.train_ratio,
            "seed": a.seed,
            "N_train": len(train),
            "N_test": N,
            "N_test_correct": Nc,
            "test_random_map_accuracy": Nc / max(N, 1),
            "pca_subset": pca_balance_info,
            "middle_layer": int(middle_layer),
            "late_layer": int(late_layer),
            "middle_pca_explained_variance_ratio": evm.tolist(),
            "late_pca_explained_variance_ratio": evl.tolist(),
            "middle_selection_rule": (
                "manual" if a.middle_layer >= 0 else "max correct(relation_decode - option_decode)"
            ),
            "late_selection_rule": (
                "manual" if a.late_layer >= 0 else "max correct(option_decode - relation_decode), excluding middle"
            ),
            "important_visualization_note": (
                "Late relation-colored and late option-colored panels use identical PCA coordinates."
            ),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("\nSaved:")
        for name in (
            "baseline.csv",
            "mapping_balance.csv",
            "layerwise_relation_vs_option.csv",
            "layerwise_relation_vs_option_correct.png",
            "layerwise_relation_vs_option_correct_geometry.png",
            "pca_correct_relation_to_option.png",
            "pca_correct_points.csv",
            "metadata.json",
        ):
            print(" ", out / name)

        print("\nFirst things to inspect:")
        print("  1) pca_correct_relation_to_option.png")
        print("  2) layerwise_relation_vs_option_correct.png")
        print("  3) layerwise_relation_vs_option.csv")
        print("\nInterpret only if the pattern is actually present; this script does not force a crossover.")

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
