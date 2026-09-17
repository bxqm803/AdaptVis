#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-3B + COCO random-map layerwise analysis.

Goal
----
At each decoder layer, measure the strongest available spatial-relation readout
across TWO candidate carriers:

    1) object-pair text state
       q_obj(l) =
           (h_real[sub] - h_real[ref])
         - (h_gray[sub] - h_gray[ref])

    2) prompt-final state
       q_last(l) =
           h_real[last] - h_gray[last]

For each layer we compute held-out relation decoding from BOTH carriers and define

    best_spatial(l) = max(
        relation_acc_from_q_obj(l),
        relation_acc_from_q_last(l)
    )

IMPORTANT:
"best_spatial" is a descriptive best-of-two ENVELOPE at the layer level.
It is NOT a per-example oracle: we do not choose obj-vs-last separately for
individual samples.

We compare this spatial envelope with answer-option decoding from q_last(l).

Outputs
-------
A) all held-out TEST samples:
   best_spatial_vs_option_all.png

B) correctly generated held-out TEST samples only:
   best_spatial_vs_option_correct.png

For the all-sample plot, we show both:
   - GT-option decode: whether q_last predicts the option assigned to the GT relation
   - generated-option decode: whether q_last predicts the option actually generated
     by the model (only among samples whose generated A/B/C/D option is parseable)

For correct samples GT option == generated option, so the correct-only plot uses
a single option-decoding curve.

Audit outputs also preserve the two underlying spatial curves separately so the
best-of-two envelope is fully transparent.

Protocol
--------
- Dataset: COCO two-object 4-way spatial relation.
- Model: Qwen2.5-VL-3B.
- Per-example randomized balanced relation -> A/B/C/D mapping.
- Real-Gray residuals.
- TRAIN split fits relation / option centroids.
- TEST split is used for all reported readout accuracies.
- Actual generation is run on TEST to define correct-only and generated-option labels.
- Same decoder forward provides q_obj and q_last at every layer.

Run
---
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u qwen3b_coco_best_spatial_readout_vs_option_v1.py \
  --max-samples 0 \
  --eval-max-samples 0 \
  --output-dir output/qwen3b_coco_best_spatial_readout_vs_option_v1 \
  --overwrite

Quick test
----------
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u qwen3b_coco_best_spatial_readout_vs_option_v1.py \
  --max-samples 160 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_coco_best_spatial_readout_vs_option_quick \
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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

# Existing random A/B/C/D COCO protocol.
import eval_qwen_coco_randmap_late_spatial_deconfound_quick as rm

# Reuse the repo's already-debugged multimodal text-position -> decoder-position
# mapping and hidden-state extraction utilities.
import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6 as v6

base = rm.base
traj = rm.traj
REL = tuple(rm.REL)
LETTERS = tuple(rm.LETTERS)


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
        help="all = every decoder block; otherwise comma list such as 20,21,...,35",
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all available COCO-two samples before train/test split",
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
        raise ValueError(f"Invalid layers {bad}; model has L0-L{n_layers - 1}")
    return out


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("-inf")
    return float(np.dot(a, b) / den)


def nearest_centroid(x: np.ndarray, centroids: Dict[str, np.ndarray]) -> str:
    labels = list(centroids)
    vals = [cosine_np(x, centroids[k]) for k in labels]
    vals = [-1e30 if not np.isfinite(v) else v for v in vals]
    return labels[int(np.argmax(vals))]


def make_gray_like(image, value: int):
    from PIL import Image
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, (v, v, v))


def capture_obj_and_last(
    *,
    model,
    processor,
    decoder_layers: Sequence,
    batch: Dict[str, torch.Tensor],
    subject: str,
    reference: str,
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    One no-grad multimodal forward.

    Returns:
      obj[L]  = mean(h_sub) - mean(h_ref)
      last[L] = h_prompt-final

    Both are decoder-block output states (hidden_states[L+1]).
    """
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]

    sub_raw = v6.hs.headprobe.locate_phrase_positions(
        processor.tokenizer, ids, subject
    )
    ref_raw = v6.hs.headprobe.locate_phrase_positions(
        processor.tokenizer, ids, reference
    )

    with torch.inference_mode():
        out = model(
            **batch,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    hst = v6.extract_hidden_states(out)
    if len(hst) < len(decoder_layers) + 1:
        raise RuntimeError(
            f"hidden_states length={len(hst)} "
            f"but decoder layers={len(decoder_layers)}"
        )

    merged_len = int(hst[0].shape[1])

    sub_pos, _ = v6.recv.map_text_positions_to_decoder(
        model=model,
        processor=processor,
        input_ids=ids,
        raw_positions=sub_raw,
        merged_length=merged_len,
    )
    ref_pos, _ = v6.recv.map_text_positions_to_decoder(
        model=model,
        processor=processor,
        input_ids=ids,
        raw_positions=ref_raw,
        merged_length=merged_len,
    )

    if not sub_pos or not ref_pos:
        raise RuntimeError(
            f"Could not map subject/reference positions: "
            f"subject={subject!r} ref={reference!r}"
        )

    obj = {}
    last = {}

    for L in layers:
        h = hst[int(L) + 1][0]
        si = torch.as_tensor(sub_pos, device=h.device, dtype=torch.long)
        ri = torch.as_tensor(ref_pos, device=h.device, dtype=torch.long)

        pair = (
            h.index_select(0, si).mean(0)
            - h.index_select(0, ri).mean(0)
        )

        # Batch size = 1, no padding on the right in this protocol.
        # This matches the prompt-final state used by the existing random-map script.
        h_last = h[-1]

        obj[int(L)] = (
            pair.detach().float().cpu().numpy().astype(np.float32)
        )
        last[int(L)] = (
            h_last.detach().float().cpu().numpy().astype(np.float32)
        )

    del out, hst
    return obj, last


def fit_centroids(
    items: List[dict],
    maps: Dict[int, Dict[str, str]],
    obj_q: Dict[int, Dict[int, np.ndarray]],
    last_q: Dict[int, Dict[int, np.ndarray]],
    layers: Sequence[int],
):
    obj_rel = {L: {} for L in layers}
    last_rel = {L: {} for L in layers}
    last_opt = {L: {} for L in layers}

    for L in layers:
        for r in REL:
            xo = [
                obj_q[int(m["sid"])][L]
                for m in items
                if int(m["sid"]) in obj_q and m["gt"] == r
            ]
            xl = [
                last_q[int(m["sid"])][L]
                for m in items
                if int(m["sid"]) in last_q and m["gt"] == r
            ]
            if not xo or not xl:
                raise RuntimeError(f"No TRAIN states for relation={r} L{L}")
            obj_rel[L][r] = np.mean(np.stack(xo), axis=0).astype(np.float32)
            last_rel[L][r] = np.mean(np.stack(xl), axis=0).astype(np.float32)

        for a in LETTERS:
            xs = [
                last_q[int(m["sid"])][L]
                for m in items
                if int(m["sid"]) in last_q
                and maps[int(m["sid"])][m["gt"]] == a
            ]
            if not xs:
                raise RuntimeError(f"No TRAIN states for option={a} L{L}")
            last_opt[L][a] = np.mean(np.stack(xs), axis=0).astype(np.float32)

    return obj_rel, last_rel, last_opt


def evaluate_subset(
    *,
    subset_name: str,
    items: List[dict],
    maps: Dict[int, Dict[str, str]],
    baseline_by_sid: Dict[int, dict],
    obj_q,
    last_q,
    obj_rel_cent,
    last_rel_cent,
    last_opt_cent,
    layers,
):
    """
    Compute aggregate held-out accuracies for this subset.

    best_spatial is max at the LAYER-LEVEL between:
      object-pair relation accuracy
      prompt-final relation accuracy

    It is NOT per-sample max/oracle selection.
    """
    rows = []

    for L in layers:
        obj_rel_ok = 0
        last_rel_ok = 0
        gt_opt_ok = 0

        gen_opt_ok = 0
        gen_opt_n = 0

        n = 0

        for m in items:
            sid = int(m["sid"])
            if sid not in obj_q or sid not in last_q or sid not in baseline_by_sid:
                continue

            gt = m["gt"]
            gt_opt = maps[sid][gt]

            pred_obj_rel = nearest_centroid(obj_q[sid][L], obj_rel_cent[L])
            pred_last_rel = nearest_centroid(last_q[sid][L], last_rel_cent[L])
            pred_last_opt = nearest_centroid(last_q[sid][L], last_opt_cent[L])

            obj_rel_ok += int(pred_obj_rel == gt)
            last_rel_ok += int(pred_last_rel == gt)
            gt_opt_ok += int(pred_last_opt == gt_opt)

            generated_opt = baseline_by_sid[sid].get("pred_option")
            if generated_opt in LETTERS:
                gen_opt_n += 1
                gen_opt_ok += int(pred_last_opt == generated_opt)

            n += 1

        if n == 0:
            raise RuntimeError(f"No evaluable samples in subset={subset_name}")

        obj_acc = obj_rel_ok / n
        last_acc = last_rel_ok / n

        if obj_acc >= last_acc:
            best_acc = obj_acc
            best_source = "object_pair"
        else:
            best_acc = last_acc
            best_source = "prompt_final"

        rows.append(
            {
                "subset": subset_name,
                "layer": int(L),
                "N": int(n),
                "object_pair_relation_acc": float(obj_acc),
                "prompt_final_relation_acc": float(last_acc),
                "best_spatial_relation_acc": float(best_acc),
                "best_spatial_source": best_source,
                "gt_option_acc_from_prompt_final": float(gt_opt_ok / n),
                "generated_option_N": int(gen_opt_n),
                "generated_option_acc_from_prompt_final": (
                    float(gen_opt_ok / gen_opt_n) if gen_opt_n > 0 else float("nan")
                ),
            }
        )

    return rows


def plot_main(rows: List[dict], out_png: Path, subset_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [int(r["layer"]) for r in rows]
    spatial = [float(r["best_spatial_relation_acc"]) for r in rows]
    gt_opt = [float(r["gt_option_acc_from_prompt_final"]) for r in rows]
    gen_opt = [float(r["generated_option_acc_from_prompt_final"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    ax.plot(
        layers,
        spatial,
        marker="o",
        label="Best spatial readout (max: object-pair, prompt-final)",
    )

    if subset_name == "all":
        ax.plot(
            layers,
            gt_opt,
            marker="o",
            label="GT option readout (prompt-final)",
        )
        if any(np.isfinite(x) for x in gen_opt):
            ax.plot(
                layers,
                gen_opt,
                marker="o",
                label="Generated option readout (prompt-final)",
            )
    else:
        # On correct samples GT option == generated option.
        ax.plot(
            layers,
            gt_opt,
            marker="o",
            label="Option readout (prompt-final)",
        )

    ax.axhline(0.25, linestyle=":", linewidth=1, label="4-way chance")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Held-out nearest-centroid accuracy")
    ax.set_ylim(0.0, 1.02)

    if subset_name == "all":
        title = "Layer-wise spatial vs decision readout: all held-out samples"
    else:
        title = "Layer-wise spatial vs decision readout: correct samples only"
    ax.set_title(title)

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_components(rows: List[dict], out_png: Path, subset_name: str):
    """
    Audit figure showing what produced the spatial envelope.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [int(r["layer"]) for r in rows]
    obj = [float(r["object_pair_relation_acc"]) for r in rows]
    last = [float(r["prompt_final_relation_acc"]) for r in rows]
    best = [float(r["best_spatial_relation_acc"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(layers, obj, marker="o", label="Object-pair relation readout")
    ax.plot(layers, last, marker="o", label="Prompt-final relation readout")
    ax.plot(
        layers,
        best,
        marker="o",
        linewidth=2.4,
        label="Best-of-two spatial envelope",
    )
    ax.axhline(0.25, linestyle=":", linewidth=1, label="4-way chance")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Held-out nearest-centroid accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        "Spatial readout carriers: "
        + ("all held-out samples" if subset_name == "all" else "correct samples only")
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}; use --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
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
        meta.append(
            {
                "sid": sid,
                "gt": gt,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
            }
        )

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

    model = None
    processor = None

    obj_q: Dict[int, Dict[int, np.ndarray]] = {}
    last_q: Dict[int, Dict[int, np.ndarray]] = {}
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
                model.generation_config.pad_token_id = (
                    model.generation_config.eos_token_id
                )

        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        layers = parse_layers(a.layers, len(decoder_layers))

        print("=" * 120)
        print("QWEN3B + COCO: BEST AVAILABLE SPATIAL READOUT VS OPTION READOUT")
        print("=" * 120)
        print(f"decoder={decoder_path} | n_layers={len(decoder_layers)}")
        print(f"layers={layers}")
        print(f"train/test={len(train)}/{len(test)}")
        print(
            "best spatial at each layer = max("
            "object-pair Real-Gray relation acc, "
            "prompt-final Real-Gray relation acc)"
        )
        print(
            "NOTE: this is a layer-level best-of-two envelope, "
            "NOT a per-sample oracle."
        )
        print()

        # ------------------------------------------------------------
        # Capture TRAIN + TEST object-pair and prompt-final Real-Gray.
        # ------------------------------------------------------------
        for split_name, items, maps in (
            ("TRAIN", train, train_map),
            ("TEST", test, test_map),
        ):
            do_generation = split_name == "TEST"

            for m in tqdm(items, desc=f"{split_name} obj+last Real-Gray"):
                sid = int(m["sid"])
                mp = maps[sid]
                prompt = rm.build_randmap_prompt(
                    m["subject"],
                    m["reference"],
                    mp,
                )

                real = gray = rb = gb = None
                try:
                    real = rm.make_real_image(rec_by_sid[sid])
                    gray = make_gray_like(real, a.gray_value)

                    rb = rm.make_batch(processor, device, real, prompt)
                    gb = rm.make_batch(processor, device, gray, prompt)

                    obj_r, last_r = capture_obj_and_last(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        subject=m["subject"],
                        reference=m["reference"],
                        layers=layers,
                    )
                    obj_g, last_g = capture_obj_and_last(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=gb,
                        subject=m["subject"],
                        reference=m["reference"],
                        layers=layers,
                    )

                    obj_q[sid] = {
                        L: (obj_r[L] - obj_g[L]).astype(np.float32)
                        for L in layers
                    }
                    last_q[sid] = {
                        L: (last_r[L] - last_g[L]).astype(np.float32)
                        for L in layers
                    }

                    if do_generation:
                        pred, text, parse_mode, _ = rm.generate_one(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            mapping=mp,
                            max_new_tokens=a.max_new_tokens,
                        )
                        correct_option = mp[m["gt"]]

                        baseline_rows.append(
                            {
                                "sid": sid,
                                "gt": m["gt"],
                                "correct_option": correct_option,
                                "pred_option": pred,
                                "baseline_correct": pred == correct_option,
                                "parse_mode": parse_mode,
                                "text": text,
                                "mapping": rm.mapping_string(mp),
                            }
                        )

                except Exception as exc:
                    row = {
                        "phase": split_name.lower(),
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    with error_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raise
                finally:
                    for im in (real, gray):
                        if im is not None:
                            with contextlib.suppress(Exception):
                                im.close()
                    del rb, gb
                    gc.collect()

        write_csv(outdir / "baseline.csv", baseline_rows)

        train = [
            m for m in train
            if int(m["sid"]) in obj_q and int(m["sid"]) in last_q
        ]
        test = [
            m for m in test
            if int(m["sid"]) in obj_q and int(m["sid"]) in last_q
        ]

        baseline_by_sid = {
            int(r["sid"]): r for r in baseline_rows
        }

        test = [
            m for m in test
            if int(m["sid"]) in baseline_by_sid
        ]

        correct_test = [
            m for m in test
            if bool(baseline_by_sid[int(m["sid"])]["baseline_correct"])
        ]

        print(
            f"\nRandom-map generation: "
            f"N_test={len(test)} "
            f"N_correct={len(correct_test)} "
            f"acc={len(correct_test) / max(len(test), 1):.4f}"
        )
        print(
            "Correct relation counts:",
            dict(Counter(m["gt"] for m in correct_test)),
        )

        # ------------------------------------------------------------
        # Fit TRAIN centroids for three fixed readout families.
        # ------------------------------------------------------------
        obj_rel_cent, last_rel_cent, last_opt_cent = fit_centroids(
            train,
            train_map,
            obj_q,
            last_q,
            layers,
        )

        # ------------------------------------------------------------
        # Evaluate ALL held-out test samples.
        # ------------------------------------------------------------
        all_rows = evaluate_subset(
            subset_name="all",
            items=test,
            maps=test_map,
            baseline_by_sid=baseline_by_sid,
            obj_q=obj_q,
            last_q=last_q,
            obj_rel_cent=obj_rel_cent,
            last_rel_cent=last_rel_cent,
            last_opt_cent=last_opt_cent,
            layers=layers,
        )

        # ------------------------------------------------------------
        # Evaluate CORRECT-generation test samples only.
        # ------------------------------------------------------------
        correct_rows = evaluate_subset(
            subset_name="correct",
            items=correct_test,
            maps=test_map,
            baseline_by_sid=baseline_by_sid,
            obj_q=obj_q,
            last_q=last_q,
            obj_rel_cent=obj_rel_cent,
            last_rel_cent=last_rel_cent,
            last_opt_cent=last_opt_cent,
            layers=layers,
        )

        write_csv(
            outdir / "layerwise_best_spatial_all.csv",
            all_rows,
        )
        write_csv(
            outdir / "layerwise_best_spatial_correct.csv",
            correct_rows,
        )

        plot_main(
            all_rows,
            outdir / "best_spatial_vs_option_all.png",
            "all",
        )
        plot_main(
            correct_rows,
            outdir / "best_spatial_vs_option_correct.png",
            "correct",
        )

        plot_components(
            all_rows,
            outdir / "spatial_components_all.png",
            "all",
        )
        plot_components(
            correct_rows,
            outdir / "spatial_components_correct.png",
            "correct",
        )

        # ------------------------------------------------------------
        # Print compact layerwise table.
        # ------------------------------------------------------------
        print("\n" + "=" * 140)
        print("ALL TEST SAMPLES")
        print("=" * 140)
        for r in all_rows:
            print(
                f"L{int(r['layer']):02d} | "
                f"obj-rel={r['object_pair_relation_acc']:.3f} "
                f"last-rel={r['prompt_final_relation_acc']:.3f} "
                f"BEST={r['best_spatial_relation_acc']:.3f}"
                f"[{r['best_spatial_source']}] | "
                f"GT-opt={r['gt_option_acc_from_prompt_final']:.3f} "
                f"GEN-opt={r['generated_option_acc_from_prompt_final']:.3f} "
                f"(Ngen={int(r['generated_option_N'])})"
            )

        print("\n" + "=" * 140)
        print("CORRECT TEST SAMPLES ONLY")
        print("=" * 140)
        for r in correct_rows:
            print(
                f"L{int(r['layer']):02d} | "
                f"obj-rel={r['object_pair_relation_acc']:.3f} "
                f"last-rel={r['prompt_final_relation_acc']:.3f} "
                f"BEST={r['best_spatial_relation_acc']:.3f}"
                f"[{r['best_spatial_source']}] | "
                f"option={r['gt_option_acc_from_prompt_final']:.3f}"
            )

        metadata = {
            "model_alias": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "n_decoder_layers": len(decoder_layers),
            "layers": list(map(int, layers)),
            "seed": int(a.seed),
            "train_ratio": float(a.train_ratio),
            "N_train": len(train),
            "N_test": len(test),
            "N_correct_test": len(correct_test),
            "random_map_generation_accuracy": (
                len(correct_test) / max(len(test), 1)
            ),
            "object_pair_state": (
                "q_obj=(h_real_sub-h_real_ref)"
                "-(h_gray_sub-h_gray_ref)"
            ),
            "prompt_final_state": (
                "q_last=h_real_last-h_gray_last"
            ),
            "best_spatial_definition": (
                "layer-level max of held-out aggregate relation decoding "
                "accuracy from q_obj and q_last; not per-example oracle"
            ),
            "option_readout_state": "q_last",
            "option_centroid_fit_labels": (
                "TRAIN correct A/B/C/D option induced by balanced random relation-option mapping"
            ),
            "all_sample_option_metrics": {
                "gt_option": (
                    "compare q_last option prediction with option mapped from GT relation"
                ),
                "generated_option": (
                    "compare the same q_last option prediction with actual generated A/B/C/D option; "
                    "only parseable generated options are counted"
                ),
            },
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "best_spatial_vs_option_all.png",
            "best_spatial_vs_option_correct.png",
            "spatial_components_all.png",
            "spatial_components_correct.png",
            "layerwise_best_spatial_all.csv",
            "layerwise_best_spatial_correct.csv",
            "baseline.csv",
            "metadata.json",
        ):
            print(" ", outdir / name)

        print(
            "\nPrimary figures:\n"
            "  1) best_spatial_vs_option_all.png\n"
            "  2) best_spatial_vs_option_correct.png\n"
            "\nAudit figures preserve the underlying object-pair and prompt-final "
            "spatial readouts separately."
        )

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
