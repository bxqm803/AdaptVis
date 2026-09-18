#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find a stronger illustrative sample for Figure A.

This script uses the SAME protocol as:
    figureA_qwen3b_single_sample_spatial_steering_layers_v1.py

It does NOT create new evidence for the paper. It only pre-specifies a ranking
rule to choose a visually clear illustrative example after the dataset-level
experiment is defined separately.

Preferred illustrative pattern
==============================
For a held-out sample i with GT relation g and semantic opposite o:

  1) Original first-step prediction is wrong.
  2) Preferably, the wrong prediction is exactly pi_i(o).
  3) At the SAME middle layer L, steering toward g:
       - increases log P(pi_i(g))
       - decreases log P(pi_i(o))
  4) Steering toward o does the reverse:
       - decreases log P(pi_i(g))
       - increases log P(pi_i(o))
  5) The two intervention directions create a large decision-margin "opening".

For each layer:

    margin_orig = logP(GT option) - logP(opposite option)
    margin_gt   = same margin after steer -> GT
    margin_opp  = same margin after steer -> opposite

    opening(L) = margin_gt - margin_opp

This equals:

    [margin_gt - margin_orig] + [margin_orig - margin_opp]

so it rewards BOTH directions at the same layer. The ranking also records the
four directional component tests separately and whether steer->GT flips the
first-step decision to the GT-mapped option.

To avoid selecting a one-layer spike, final rank_score mixes the best-layer
opening with local neighbor support:

    rank_score = 0.70 * best_opening
               + 0.30 * local_opening_mean
               + 0.25 * gt_flip_at_best
               + 0.10 * opp_prediction_at_best
               + 0.20 * four_way_consistent_at_best

where local_opening_mean averages the best layer and adjacent scanned layers.

Outputs
=======
  sample_ranking.csv
      One row per scanned candidate, sorted best first.

  layer_details.csv
      Per sample x layer causal steering details.

  top_samples.json
      Compact metadata for the top-ranked samples.

  previews/rank01_sidXXXX.png ...
      Figure-A-format previews for the top-K samples.

Recommended run
===============
Put this script next to:
  figureA_qwen3b_single_sample_spatial_steering_layers_v1.py

Then run:

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u find_figureA_best_sample_v1.py \
  --model qwen-3b \
  --scan-layers 18-26 \
  --alpha 1.0 \
  --candidate-mode opposite \
  --top-k 10 \
  --preview-k 5 \
  --output-dir output/figureA_best_sample_scan_v1 \
  --overwrite

After choosing a fixed SID, regenerate the final Figure A with:

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u figureA_qwen3b_single_sample_spatial_steering_layers_v1.py \
  --sid <SID> --layers all --alpha 1.0 \
  --output-dir output/figureA_sid<SID> --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoProcessor

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj

try:
    import figureA_qwen3b_single_sample_spatial_steering_layers_v1 as figA
except Exception as exc:
    raise SystemExit(
        "Could not import figureA_qwen3b_single_sample_spatial_steering_layers_v1.py.\n"
        "Put this scanner next to the Figure-A script in the AdaptVis repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )

REL = tuple(figA.REL)
LETTERS = tuple(figA.LETTERS)
OPP = dict(figA.OPP)
EPS = float(figA.EPS)
SCRIPT_VERSION = "find-figureA-best-sample-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument(
        "--scan-layers",
        default="18-26",
        help="Middle-layer window used ONLY for ranking illustrative samples.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all COCO-two samples before train/test split.",
    )
    p.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="Optional TRAIN cap for quick scans; 0 = all train.",
    )
    p.add_argument(
        "--candidate-mode",
        default="opposite",
        choices=["opposite", "wrong", "all"],
        help=(
            "opposite = scan only samples whose original first-step prediction is the mapped semantic opposite; "
            "wrong = any original error; all = every held-out sample."
        ),
    )
    p.add_argument(
        "--candidate-max-samples",
        type=int,
        default=0,
        help="Optional cap AFTER baseline candidate filtering; 0 = all candidates.",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--preview-k", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    traj.write_csv(path, rows)


def local_mean(opening_by_layer: Mapping[int, float], best_layer: int) -> float:
    vals = [
        float(opening_by_layer[L])
        for L in (best_layer - 1, best_layer, best_layer + 1)
        if L in opening_by_layer
    ]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    a = parse_args()
    if a.alpha < 0:
        raise ValueError("--alpha must be >= 0")
    if not 0 <= a.gray_value <= 255:
        raise ValueError("--gray-value must be in [0,255]")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    preview_dir = out / "previews"
    preview_dir.mkdir(exist_ok=True)
    err_path = out / "errors.jsonl"

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta: List[Dict[str, Any]] = []
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
    if a.train_max_samples > 0:
        train = traj.stratified_cap(train, a.train_max_samples, a.seed + 31)

    train_map = figA.assign_relation_balanced_mappings(train, a.seed + 1001)
    test_map = figA.assign_relation_balanced_mappings(test, a.seed + 2003)

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
        print(f"[LOAD] {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        scan_layers = figA.parse_layers(a.scan_layers, len(decoder_layers))
        relation_token_map = base.relation_token_variants(processor.tokenizer)
        option_token_map = figA.build_option_token_map(processor.tokenizer)

        print("=" * 108)
        print("FIGURE A — BEST ILLUSTRATIVE SAMPLE SCAN")
        print("=" * 108)
        print(
            f"decoder={decoder_path} | scan_layers={scan_layers} | alpha={a.alpha:g} | "
            f"candidate_mode={a.candidate_mode}"
        )
        print(f"TRAIN={len(train)} TEST={len(test)}")
        print()

        # ------------------------------------------------------------------
        # 1) Fit TRAIN spatial centroids on exactly the scan layers.
        # ------------------------------------------------------------------
        train_q: Dict[int, Dict[int, np.ndarray]] = {}
        for m in tqdm(train, desc="TRAIN Real-Gray spatial states"):
            sid = int(m["sid"])
            prompt = figA.build_randmap_prompt(
                m["subject"], m["reference"], train_map[sid]
            )
            try:
                q, _sp, _rp = figA.collect_realgray_pair_states(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    rec=rec_by_sid[sid],
                    prompt=prompt,
                    subject=m["subject"],
                    reference=m["reference"],
                    layers=scan_layers,
                    object_state=a.object_state,
                    relation_token_map=relation_token_map,
                    gray_value=a.gray_value,
                    device=device,
                )
                train_q[sid] = q
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "train_realgray",
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                gc.collect()

        train_valid = [m for m in train if int(m["sid"]) in train_q]
        cent = traj.fit_centroids(train_valid, train_q, scan_layers)
        np.savez_compressed(
            out / "train_relation_centroids_realgray.npz",
            **{f"L{L}_{r}": cent[L][r] for L in scan_layers for r in REL},
        )

        # ------------------------------------------------------------------
        # 2) Cheap baseline pass: keep only requested candidate type.
        # ------------------------------------------------------------------
        candidates: List[Dict[str, Any]] = []
        for m in tqdm(test, desc="baseline candidate filter"):
            sid = int(m["sid"])
            mapping = test_map[sid]
            gt = str(m["gt"])
            opp = OPP[gt]
            gt_letter = mapping[gt]
            opp_letter = mapping[opp]
            prompt = figA.build_randmap_prompt(m["subject"], m["reference"], mapping)
            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = figA.make_batch(processor, device, image, prompt)
                sc = figA.first_step_scores(model, batch, option_token_map)
                pred = sc["prediction"]
                keep = (
                    a.candidate_mode == "all"
                    or (a.candidate_mode == "wrong" and pred != gt_letter)
                    or (a.candidate_mode == "opposite" and pred == opp_letter)
                )
                if keep:
                    candidates.append({
                        **m,
                        "mapping": mapping,
                        "prompt": prompt,
                        "gt_letter": gt_letter,
                        "opp_letter": opp_letter,
                        "orig_prediction": pred,
                        "orig_logprob": dict(sc["logprob"]),
                    })
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "baseline_filter",
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

        if not candidates:
            raise RuntimeError(f"No candidates found for mode={a.candidate_mode}")

        if a.candidate_max_samples > 0 and len(candidates) > a.candidate_max_samples:
            rng = random.Random(a.seed + 4049)
            rng.shuffle(candidates)
            candidates = candidates[: a.candidate_max_samples]

        print(f"[CANDIDATES] {len(candidates)} samples after baseline filter")

        # ------------------------------------------------------------------
        # 3) Causal scan and deterministic ranking.
        # ------------------------------------------------------------------
        layer_details: List[Dict[str, Any]] = []
        sample_rows: List[Dict[str, Any]] = []
        rows_by_sid: Dict[int, List[Dict[str, Any]]] = {}

        for c in tqdm(candidates, desc="causal sample ranking"):
            sid = int(c["sid"])
            gt = str(c["gt"])
            opp = OPP[gt]
            mapping = c["mapping"]
            gt_letter = str(c["gt_letter"])
            opp_letter = str(c["opp_letter"])
            orig_lp = c["orig_logprob"]
            orig_margin = float(orig_lp[gt_letter] - orig_lp[opp_letter])

            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = figA.make_batch(processor, device, image, c["prompt"])

                clean = traj.clean_forward(
                    base, model, processor, decoder_layers, batch,
                    c["subject"], c["reference"], scan_layers,
                    a.object_state, relation_token_map,
                )
                subject_positions = tuple(clean["subject_positions"])
                reference_positions = tuple(clean["reference_positions"])

                fig_rows: List[Dict[str, Any]] = []
                opening_by_layer: Dict[int, float] = {}
                detail_for_sample: List[Dict[str, Any]] = []

                for L in scan_layers:
                    raw_axis = (
                        np.asarray(cent[L][gt], np.float32)
                        - np.asarray(cent[L][opp], np.float32)
                    )
                    if float(np.linalg.norm(raw_axis)) < EPS:
                        continue

                    sc_gt = figA.patched_first_step_scores(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=batch,
                        token_map=option_token_map,
                        layer=L,
                        subject_positions=subject_positions,
                        reference_positions=reference_positions,
                        delta_pair=(a.alpha * raw_axis).astype(np.float32),
                    )
                    sc_opp = figA.patched_first_step_scores(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=batch,
                        token_map=option_token_map,
                        layer=L,
                        subject_positions=subject_positions,
                        reference_positions=reference_positions,
                        delta_pair=(-a.alpha * raw_axis).astype(np.float32),
                    )

                    gt_gt = float(sc_gt["logprob"][gt_letter])
                    gt_opp = float(sc_gt["logprob"][opp_letter])
                    opp_gt = float(sc_opp["logprob"][gt_letter])
                    opp_opp = float(sc_opp["logprob"][opp_letter])
                    orig_gt = float(orig_lp[gt_letter])
                    orig_opp = float(orig_lp[opp_letter])

                    margin_gt = gt_gt - gt_opp
                    margin_opp = opp_gt - opp_opp
                    opening = margin_gt - margin_opp
                    gt_gain = gt_gt - orig_gt
                    opp_suppress = orig_opp - gt_opp
                    opp_gain = opp_opp - orig_opp
                    gt_suppress = orig_gt - opp_gt
                    four_way = int(
                        gt_gain > 0 and opp_suppress > 0 and opp_gain > 0 and gt_suppress > 0
                    )
                    gt_flip = int(sc_gt["prediction"] == gt_letter)
                    opp_pred = int(sc_opp["prediction"] == opp_letter)

                    det = {
                        "sid": sid,
                        "layer": int(L),
                        "gt_relation": gt,
                        "opposite_relation": opp,
                        "gt_letter": gt_letter,
                        "opposite_letter": opp_letter,
                        "orig_prediction": c["orig_prediction"],
                        "toward_gt_prediction": sc_gt["prediction"],
                        "toward_opp_prediction": sc_opp["prediction"],
                        "orig_margin": orig_margin,
                        "toward_gt_margin": margin_gt,
                        "toward_opp_margin": margin_opp,
                        "gt_margin_gain": margin_gt - orig_margin,
                        "opp_margin_drop": orig_margin - margin_opp,
                        "opening": opening,
                        "gt_option_gain": gt_gain,
                        "opp_option_suppression": opp_suppress,
                        "opp_option_gain": opp_gain,
                        "gt_option_suppression": gt_suppress,
                        "four_way_consistent": four_way,
                        "gt_flip": gt_flip,
                        "opp_prediction": opp_pred,
                    }
                    detail_for_sample.append(det)
                    layer_details.append(det)
                    opening_by_layer[int(L)] = opening

                    # rows compatible with Figure-A plot helper
                    for cond, sc in (
                        ("orig", {"prediction": c["orig_prediction"], "logprob": orig_lp}),
                        ("toward_gt", sc_gt),
                        ("toward_opp", sc_opp),
                    ):
                        rr: Dict[str, Any] = {
                            "sid": sid,
                            "layer": int(L),
                            "condition": cond,
                            "alpha": float(a.alpha),
                            "gt_relation": gt,
                            "opposite_relation": opp,
                            "gt_letter": gt_letter,
                            "opposite_letter": opp_letter,
                            "prediction": sc["prediction"],
                        }
                        for letter in LETTERS:
                            rr[f"logp_{letter}"] = float(sc["logprob"][letter])
                        rr["gt_vs_opp_margin"] = float(
                            sc["logprob"][gt_letter] - sc["logprob"][opp_letter]
                        )
                        fig_rows.append(rr)

                if not detail_for_sample:
                    continue

                best = max(detail_for_sample, key=lambda x: float(x["opening"]))
                best_layer = int(best["layer"])
                local_opening = local_mean(opening_by_layer, best_layer)
                rank_score = (
                    0.70 * float(best["opening"])
                    + 0.30 * float(local_opening)
                    + 0.25 * int(best["gt_flip"])
                    + 0.10 * int(best["opp_prediction"])
                    + 0.20 * int(best["four_way_consistent"])
                )

                sample_rows.append({
                    "sid": sid,
                    "subject": c["subject"],
                    "reference": c["reference"],
                    "gt_relation": gt,
                    "opposite_relation": opp,
                    "mapping": figA.mapping_string(mapping),
                    "gt_letter": gt_letter,
                    "opposite_letter": opp_letter,
                    "orig_prediction": c["orig_prediction"],
                    "orig_margin": orig_margin,
                    "best_layer": best_layer,
                    "best_opening": float(best["opening"]),
                    "local_opening_mean": float(local_opening),
                    "gt_margin_gain_best": float(best["gt_margin_gain"]),
                    "opp_margin_drop_best": float(best["opp_margin_drop"]),
                    "gt_option_gain_best": float(best["gt_option_gain"]),
                    "opp_option_suppression_best": float(best["opp_option_suppression"]),
                    "opp_option_gain_best": float(best["opp_option_gain"]),
                    "gt_option_suppression_best": float(best["gt_option_suppression"]),
                    "four_way_consistent_best": int(best["four_way_consistent"]),
                    "gt_flip_best": int(best["gt_flip"]),
                    "opp_prediction_best": int(best["opp_prediction"]),
                    "rank_score": float(rank_score),
                })
                rows_by_sid[sid] = fig_rows

            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "candidate_scan",
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

        if not sample_rows:
            raise RuntimeError("No candidate completed the causal scan")

        sample_rows.sort(key=lambda x: float(x["rank_score"]), reverse=True)
        for rank, row in enumerate(sample_rows, 1):
            row["rank"] = rank

        write_csv(out / "sample_ranking.csv", sample_rows)
        write_csv(out / "layer_details.csv", layer_details)

        top = sample_rows[: max(1, int(a.top_k))]
        top_json = []
        for row in top:
            top_json.append({k: row[k] for k in row})
        (out / "top_samples.json").write_text(
            json.dumps(top_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ------------------------------------------------------------------
        # 4) Render Figure-A-format previews for the top K.
        # ------------------------------------------------------------------
        test_by_sid = {int(m["sid"]): m for m in test}
        for row in top[: max(0, int(a.preview_k))]:
            sid = int(row["sid"])
            m = test_by_sid[sid]
            image = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                rank = int(row["rank"])
                figA.plot_figure(
                    output_path=preview_dir / f"rank{rank:02d}_sid{sid}.png",
                    image=image,
                    subject=m["subject"],
                    reference=m["reference"],
                    gt=m["gt"],
                    opp=OPP[m["gt"]],
                    mapping=test_map[sid],
                    sid=sid,
                    rows=rows_by_sid[sid],
                    alpha=a.alpha,
                )
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": spec.repo_id,
            "decoder_path": decoder_path,
            "scan_layers": scan_layers,
            "alpha": a.alpha,
            "candidate_mode": a.candidate_mode,
            "candidate_count": len(candidates),
            "completed_count": len(sample_rows),
            "ranking_rule": (
                "0.70*best_opening + 0.30*local_opening_mean + 0.25*gt_flip + "
                "0.10*opp_prediction + 0.20*four_way_consistent"
            ),
            "opening_definition": "(margin toward GT) - (margin toward opposite)",
            "train_n": len(train_valid),
            "test_n": len(test),
            "seed": a.seed,
        }
        traj.write_json(out / "metadata.json", metadata)

        print("\n" + "=" * 108)
        print("TOP FIGURE-A CANDIDATES")
        print("=" * 108)
        for row in top:
            print(
                f"#{int(row['rank']):02d} sid={int(row['sid']):3d} "
                f"{row['gt_relation']}->{row['gt_letter']} "
                f"orig={row['orig_prediction']} "
                f"best=L{int(row['best_layer']):02d} "
                f"opening={float(row['best_opening']):+.4f} "
                f"local={float(row['local_opening_mean']):+.4f} "
                f"4way={int(row['four_way_consistent_best'])} "
                f"GTflip={int(row['gt_flip_best'])} "
                f"score={float(row['rank_score']):+.4f}"
            )
        print(f"\n[SAVED] {out / 'sample_ranking.csv'}")
        print(f"[SAVED] {out / 'layer_details.csv'}")
        print(f"[SAVED] {preview_dir}")

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
