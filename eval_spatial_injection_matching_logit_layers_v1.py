#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spatial_injection_matching_logit_layers_v1.py

Question
========
For each source layer L, does a causal edit of the explicit spatial relation
state move the FINAL matching relation decision in the expected direction?

For each relation r independently:

    toward-r edit  -> logit(r) should increase
    away-from-r    -> logit(r) should decrease

We intentionally DO NOT use an opposite-relation logit contrast in the primary
metric.  LEFT only reads the LEFT decision logit, RIGHT only RIGHT, ON only ON,
and UNDER only UNDER.

Spatial basis
=============
Use Synthetic-400 only.  At every tested layer L:

    q_L = (h_real_sub - h_real_ref) - (h_gray_sub - h_gray_ref)

Fit four relation centroids and construct the same paper-style 2-D dual basis
used by figureB_qwen3b_synthetic400_dualbasis_bothaxes_v1.py:

    horizontal coordinate: left <-> right
    vertical coordinate:   under <-> on

The dual basis is used only to produce a pure spatial-coordinate edit.  The
readout is relation-specific, not an axis contrast.

Intervention
============
At source layer L:

    h_sub += 0.5 * delta_r
    h_ref -= 0.5 * delta_r

For strength a > 0:

    toward left   = -a * steer_H
    away left     = +a * steer_H

    toward right  = +a * steer_H
    away right    = -a * steer_H

    toward under  = -a * steer_V
    away under    = +a * steer_V

    toward on     = +a * steer_V
    away on       = -a * steer_V

Decision readout
================
To avoid a fixed relation-word lexical confound, every COCO example receives an
independently balanced random relation -> A/B/C/D mapping, using the existing
Figure-B helper.  "LEFT logit" below therefore means the FINAL option logit
currently assigned to LEFT for that sample.

For relation r:

    d_toward(r) = logit_r(toward-r edit) - logit_r(clean)
    d_away(r)   = logit_r(away-r edit)   - logit_r(clean)

Expected principle:

    d_toward(r) > 0
    d_away(r)   < 0

for r in {left, right, on, under}.

Outputs
=======
per_sample_effects.csv
    Every sample x layer x relation x strength x direction.

layer_relation_summary.csv
    Mean/SEM and sign-consistency rates for each layer/relation/strength.

layer_principle_summary.csv
    Compact per-layer report:
      - how many of four relations pass mean(toward)>0 & mean(away)<0
      - mean toward effect
      - mean away effect
      - mean paired contrast
      - sample-level both-sign consistency

synthetic_basis_diagnostics.csv
    Per-layer basis geometry.

analysis_summary.txt
metadata.json
errors.jsonl

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u \
  eval_spatial_injection_matching_logit_layers_v1.py \
  --layers 20-26 \
  --strengths 1.0 \
  --coco-max-samples 80 \
  --output-dir output/qwen3b_spatial_to_matching_logit_L20_26_n80_v1 \
  --overwrite

Stronger dose check
===================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u \
  eval_spatial_injection_matching_logit_layers_v1.py \
  --layers 20-26 \
  --strengths 0.5,1.0,1.5 \
  --coco-max-samples 80 \
  --output-dir output/qwen3b_spatial_to_matching_logit_L20_26_dose_n80_v1 \
  --overwrite

Notes
=====
- This establishes spatial-state edit -> matching final-decision-logit control.
- It does NOT by itself prove that L25 Top-K high-leverage states mediate that
  effect.  The next mediation experiment should repeat the same spatial edit
  while zeroing the L25 Top-K states.
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
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj

try:
    import figureB_qwen3b_synthetic400_left_right_split_v1 as src
except Exception as exc:
    raise SystemExit(
        "Could not import figureB_qwen3b_synthetic400_left_right_split_v1.py.\n"
        "Put this script in the AdaptVis repo root next to that helper.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import figureB_qwen3b_synthetic400_dualbasis_bothaxes_v1 as dual
except Exception as exc:
    raise SystemExit(
        "Could not import figureB_qwen3b_synthetic400_dualbasis_bothaxes_v1.py.\n"
        "Put this script in the AdaptVis repo root next to that helper.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
SCRIPT_VERSION = "spatial-injection-matching-logit-layers-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--synthetic-root",
        default="synthetic_shapes_4dir_400",
    )
    p.add_argument("--synthetic-labels", default="labels.jsonl")

    p.add_argument(
        "--layers",
        default="20-26",
        help="Source decoder layers, e.g. 20-26 or 20,22,24,25.",
    )
    p.add_argument(
        "--strengths",
        default="1.0",
        help="Positive magnitudes; comma-separated, e.g. 0.5,1.0,1.5.",
    )
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument("--synthetic-max-samples", type=int, default=0)
    p.add_argument("--coco-max-samples", type=int, default=80)
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_int_spec(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            lo, hi = min(a, b), max(a, b)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_float_list(text: str) -> List[float]:
    vals = [float(x) for x in str(text).replace(" ", "").split(",") if x]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No --strengths supplied")
    if any(x <= 0 for x in vals):
        raise ValueError("--strengths must contain positive magnitudes only")
    return vals


def sem(x: Sequence[float]) -> float:
    a = np.asarray(list(x), np.float64)
    if len(a) <= 1:
        return 0.0
    return float(a.std(ddof=1) / math.sqrt(len(a)))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def relation_vector(
    relation: str,
    basis: Mapping[str, Any],
    magnitude: float,
    toward: bool,
) -> np.ndarray:
    """
    Return pair-state delta for moving toward / away from one relation.
    The metric later reads ONLY that relation's final mapped-option logit.
    """
    if relation == "left":
        sign = -1.0
        axis = np.asarray(basis["steer_h"], np.float32)
    elif relation == "right":
        sign = +1.0
        axis = np.asarray(basis["steer_h"], np.float32)
    elif relation == "below":  # display = under
        sign = -1.0
        axis = np.asarray(basis["steer_v"], np.float32)
    elif relation == "above":  # display = on
        sign = +1.0
        axis = np.asarray(basis["steer_v"], np.float32)
    else:
        raise ValueError(relation)

    if not toward:
        sign *= -1.0
    return (sign * float(magnitude) * axis).astype(np.float32)


def summarize_relation_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    for r in rows:
        key = (
            int(r["layer"]),
            str(r["relation"]),
            float(r["strength"]),
        )
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (layer, relation, strength), chunk in sorted(groups.items()):
        toward = np.asarray(
            [float(x["delta_target_logit_toward"]) for x in chunk],
            np.float64,
        )
        away = np.asarray(
            [float(x["delta_target_logit_away"]) for x in chunk],
            np.float64,
        )
        toward_lp = np.asarray(
            [float(x["delta_target_logprob_toward"]) for x in chunk],
            np.float64,
        )
        away_lp = np.asarray(
            [float(x["delta_target_logprob_away"]) for x in chunk],
            np.float64,
        )
        contrast = toward - away
        both = (toward > 0) & (away < 0)

        out.append({
            "layer": layer,
            "relation": relation,
            "relation_display": DISPLAY[relation],
            "strength": strength,
            "N": len(chunk),
            "toward_logit_mean": float(toward.mean()),
            "toward_logit_sem": sem(toward),
            "away_logit_mean": float(away.mean()),
            "away_logit_sem": sem(away),
            "paired_logit_contrast_mean": float(contrast.mean()),
            "paired_logit_contrast_sem": sem(contrast),
            "toward_positive_rate": float(np.mean(toward > 0)),
            "away_negative_rate": float(np.mean(away < 0)),
            "both_sign_rate": float(np.mean(both)),
            "toward_logprob_mean": float(toward_lp.mean()),
            "away_logprob_mean": float(away_lp.mean()),
            "mean_sign_principle_pass": int(
                float(toward.mean()) > 0 and float(away.mean()) < 0
            ),
        })
    return out


def summarize_layers(
    relation_summary: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    for r in relation_summary:
        key = (int(r["layer"]), float(r["strength"]))
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (layer, strength), chunk in sorted(groups.items()):
        by_rel = {str(r["relation"]): r for r in chunk}
        relation_passes = sum(
            int(r["mean_sign_principle_pass"]) for r in chunk
        )

        row: Dict[str, Any] = {
            "layer": layer,
            "strength": strength,
            "relations_present": len(chunk),
            "relations_passing_mean_sign_rule": relation_passes,
            "all_four_pass": int(relation_passes == 4 and len(chunk) == 4),
            "mean_toward_logit": float(
                np.mean([float(r["toward_logit_mean"]) for r in chunk])
            ),
            "mean_away_logit": float(
                np.mean([float(r["away_logit_mean"]) for r in chunk])
            ),
            "mean_paired_logit_contrast": float(
                np.mean([
                    float(r["paired_logit_contrast_mean"]) for r in chunk
                ])
            ),
            "mean_toward_positive_rate": float(
                np.mean([float(r["toward_positive_rate"]) for r in chunk])
            ),
            "mean_away_negative_rate": float(
                np.mean([float(r["away_negative_rate"]) for r in chunk])
            ),
            "mean_both_sign_rate": float(
                np.mean([float(r["both_sign_rate"]) for r in chunk])
            ),
        }

        for rel in REL:
            rr = by_rel.get(rel)
            prefix = DISPLAY[rel]
            row[f"{prefix}_toward"] = (
                float(rr["toward_logit_mean"]) if rr else float("nan")
            )
            row[f"{prefix}_away"] = (
                float(rr["away_logit_mean"]) if rr else float("nan")
            )
            row[f"{prefix}_both_sign_rate"] = (
                float(rr["both_sign_rate"]) if rr else float("nan")
            )

        out.append(row)
    return out


def main() -> None:
    a = parse_args()
    layers = parse_int_spec(a.layers)
    strengths = parse_float_list(a.strengths)

    if not layers:
        raise ValueError("No --layers supplied")
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
    err_path = out / "errors.jsonl"

    # ------------------------------------------------------------------
    # Synthetic-400 source
    # ------------------------------------------------------------------
    synth_root = Path(a.synthetic_root)
    synth_labels = src.resolve_synthetic_labels(
        synth_root, a.synthetic_labels
    )
    synthetic = src.load_synthetic400(synth_root, synth_labels)
    synthetic = src.stratified_cap(
        synthetic, a.synthetic_max_samples, a.seed
    )
    synth_counts = {
        r: sum(1 for m in synthetic if m["gt"] == r) for r in REL
    }
    synth_map = src.assign_relation_balanced_mappings(
        synthetic, a.seed + 1001
    )

    # ------------------------------------------------------------------
    # COCO target
    # ------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two", Path(a.data_root), None
    )
    rec_by_sid = {int(r.sid): r for r in records}

    coco_all: List[Dict[str, Any]] = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        coco_all.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
        })

    coco_all = src.stratified_cap(
        coco_all, a.coco_max_samples, a.seed + 53
    )
    coco_map = src.assign_relation_balanced_mappings(
        coco_all, a.seed + 2003
    )
    coco_counts = {
        r: sum(1 for m in coco_all if m["gt"] == r) for r in REL
    }

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
        try:
            model = cls.from_pretrained(spec.repo_id, **load_kw)
        except TypeError:
            load_kw["torch_dtype"] = load_kw.pop("dtype")
            model = cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)

        for L in layers:
            if not 0 <= L < len(decoder_layers):
                raise ValueError(
                    f"Invalid L{L}; decoder has "
                    f"L0..L{len(decoder_layers)-1}"
                )

        relation_token_map = base.relation_token_variants(
            processor.tokenizer
        )
        option_token_map = src.build_option_token_map(
            processor.tokenizer
        )

        print("\n" + "=" * 128)
        print("SPATIAL INJECTION -> MATCHING FINAL-DECISION LOGIT")
        print("=" * 128)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(f"layers={layers}")
        print(f"strengths={strengths}")
        print(
            f"synthetic N={len(synthetic)} counts={synth_counts} "
            "(basis only)"
        )
        print(
            f"COCO N={len(coco_all)} counts={coco_counts} "
            "(evaluation only)"
        )
        print(
            "Primary rule per relation: toward edit -> its own logit UP; "
            "away edit -> its own logit DOWN."
        )
        print(
            "No opposite-relation logit is used in the primary metric."
        )
        print("=" * 128 + "\n")

        # ==============================================================
        # 1) Fit a Synthetic-400 spatial basis at EVERY tested layer.
        #    All layers are captured in the same Real/Gray runs.
        # ==============================================================
        synth_q: Dict[int, Dict[int, np.ndarray]] = {}
        synth_valid: List[Dict[str, Any]] = []

        for m in tqdm(
            synthetic,
            desc=f"Synthetic-400 bases @ L{layers[0]}..L{layers[-1]}",
        ):
            sid = int(m["sid"])
            prompt = src.build_randmap_prompt(
                m["subject"], m["reference"], synth_map[sid]
            )
            image = None
            try:
                image = Image.open(m["image_path"]).convert("RGB")
                q, _sp, _rp = src.collect_realgray_pair_states_from_image(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    image=image,
                    prompt=prompt,
                    subject=m["subject"],
                    reference=m["reference"],
                    layers=layers,
                    object_state=a.object_state,
                    relation_token_map=relation_token_map,
                    gray_value=a.gray_value,
                    device=device,
                )
                synth_q[sid] = q
                synth_valid.append(m)
            except Exception as exc:
                traj.append_jsonl(
                    err_path,
                    {
                        "phase": "synthetic_basis",
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

        cent = traj.fit_centroids(synth_valid, synth_q, layers)

        basis_by_layer: Dict[int, Dict[str, Any]] = {}
        basis_rows: List[Dict[str, Any]] = []

        for L in layers:
            cent_L = {
                r: np.asarray(cent[L][r], np.float32) for r in REL
            }
            b = dual.build_dual_basis(cent_L)
            basis_by_layer[L] = b

            basis_rows.append({
                "layer": L,
                "gap_H": float(b["gap_h"]),
                "gap_V": float(b["gap_v"]),
                "half_H": float(b["half_h"]),
                "half_V": float(b["half_v"]),
                "cos_dH_dV": float(b["cos_hv"]),
                "gram_condition_number": float(b["gram_cond"]),
                "dot_steerH_dV": float(b["cross_h_to_v"]),
                "dot_steerV_dH": float(b["cross_v_to_h"]),
            })

        write_csv(
            out / "synthetic_basis_diagnostics.csv",
            basis_rows,
        )

        # ==============================================================
        # 2) COCO: every sample gets all 4 relation edits at every layer.
        #    Read ONLY the target relation's mapped option logit.
        # ==============================================================
        effect_rows: List[Dict[str, Any]] = []

        for m in tqdm(
            coco_all,
            desc="COCO spatial +/- edits",
        ):
            sid = int(m["sid"])
            mapping = coco_map[sid]
            prompt = src.build_randmap_prompt(
                m["subject"], m["reference"], mapping
            )

            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = src.make_batch(
                    processor, device, image, prompt
                )

                base_sc = src.first_step_scores(
                    model, batch, option_token_map
                )

                # Token positions do not depend on layer; collect once.
                # We request all layers because the helper also returns
                # the Real-Gray pair states, though here only positions
                # are needed downstream.
                _q, subject_positions, reference_positions = (
                    src.collect_coco_realgray_pair_states(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        rec=rec_by_sid[sid],
                        prompt=prompt,
                        subject=m["subject"],
                        reference=m["reference"],
                        layers=layers,
                        object_state=a.object_state,
                        relation_token_map=relation_token_map,
                        gray_value=a.gray_value,
                        device=device,
                    )
                )

                for L in layers:
                    basis = basis_by_layer[L]

                    for rel in REL:
                        target_letter = mapping[rel]
                        base_logit = float(
                            base_sc["letter_logits"][target_letter]
                        )
                        base_logprob = float(
                            base_sc["letter_logprobs"][target_letter]
                        )

                        for strength in strengths:
                            delta_toward = relation_vector(
                                rel, basis, strength, toward=True
                            )
                            delta_away = relation_vector(
                                rel, basis, strength, toward=False
                            )

                            toward_sc = src.patched_first_step_scores(
                                model=model,
                                decoder_layers=decoder_layers,
                                batch=batch,
                                token_map=option_token_map,
                                layer=L,
                                subject_positions=subject_positions,
                                reference_positions=reference_positions,
                                delta_pair=delta_toward,
                            )
                            away_sc = src.patched_first_step_scores(
                                model=model,
                                decoder_layers=decoder_layers,
                                batch=batch,
                                token_map=option_token_map,
                                layer=L,
                                subject_positions=subject_positions,
                                reference_positions=reference_positions,
                                delta_pair=delta_away,
                            )

                            toward_logit = float(
                                toward_sc["letter_logits"][target_letter]
                            )
                            away_logit = float(
                                away_sc["letter_logits"][target_letter]
                            )
                            toward_logprob = float(
                                toward_sc["letter_logprobs"][target_letter]
                            )
                            away_logprob = float(
                                away_sc["letter_logprobs"][target_letter]
                            )

                            d_t = toward_logit - base_logit
                            d_a = away_logit - base_logit

                            effect_rows.append({
                                "sid": sid,
                                "gt": m["gt"],
                                "gt_display": DISPLAY[m["gt"]],
                                "mapping": src.mapping_string(mapping),
                                "layer": L,
                                "relation": rel,
                                "relation_display": DISPLAY[rel],
                                "target_letter": target_letter,
                                "strength": float(strength),
                                "base_target_logit": base_logit,
                                "toward_target_logit": toward_logit,
                                "away_target_logit": away_logit,
                                "delta_target_logit_toward": d_t,
                                "delta_target_logit_away": d_a,
                                "paired_logit_contrast": d_t - d_a,
                                "toward_logit_positive": int(d_t > 0),
                                "away_logit_negative": int(d_a < 0),
                                "both_sign_rule": int(d_t > 0 and d_a < 0),
                                "base_target_logprob": base_logprob,
                                "delta_target_logprob_toward": (
                                    toward_logprob - base_logprob
                                ),
                                "delta_target_logprob_away": (
                                    away_logprob - base_logprob
                                ),
                                "base_prediction_letter": base_sc["prediction"],
                                "toward_prediction_letter": toward_sc["prediction"],
                                "away_prediction_letter": away_sc["prediction"],
                            })

            except Exception as exc:
                traj.append_jsonl(
                    err_path,
                    {
                        "phase": "coco_eval",
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                if batch is not None:
                    del batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not effect_rows:
            raise RuntimeError("No COCO effect rows produced")

        write_csv(
            out / "per_sample_effects.csv",
            effect_rows,
        )

        relation_summary = summarize_relation_rows(effect_rows)
        layer_summary = summarize_layers(relation_summary)

        write_csv(
            out / "layer_relation_summary.csv",
            relation_summary,
        )
        write_csv(
            out / "layer_principle_summary.csv",
            layer_summary,
        )

        # ==============================================================
        # Readable console report
        # ==============================================================
        lines: List[str] = []
        lines.append("=" * 128)
        lines.append(
            "SPATIAL EDIT SIGN RULE: TOWARD r -> logit(r) UP; "
            "AWAY FROM r -> logit(r) DOWN"
        )
        lines.append("=" * 128)

        for strength in strengths:
            lines.append(f"[strength={strength:g}]")
            for row in [
                x for x in layer_summary
                if float(x["strength"]) == float(strength)
            ]:
                L = int(row["layer"])
                n_pass = int(row["relations_passing_mean_sign_rule"])
                lines.append(
                    f"L{L:02d} | pass={n_pass}/4 | "
                    f"mean toward={row['mean_toward_logit']:+.5f} | "
                    f"mean away={row['mean_away_logit']:+.5f} | "
                    f"contrast={row['mean_paired_logit_contrast']:+.5f} | "
                    f"sample both-sign={row['mean_both_sign_rate']:.3f}"
                )
                rel_chunks = []
                for name in ("left", "right", "on", "under"):
                    rel_chunks.append(
                        f"{name}({row[name + '_toward']:+.4f}/"
                        f"{row[name + '_away']:+.4f})"
                    )
                lines.append("      " + " ".join(rel_chunks))
            lines.append("")

        lines.append(
            "Each relation tuple is (toward delta logit / away delta logit)."
        )
        lines.append(
            "Desired sign pattern is (+ / -) independently for all four relations."
        )
        lines.append(
            "No opposite-relation logit enters this test."
        )

        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (out / "analysis_summary.txt").write_text(
            report, encoding="utf-8"
        )

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "layers": layers,
            "strengths": strengths,
            "seed": a.seed,
            "object_state": a.object_state,
            "gray_value": a.gray_value,
            "synthetic_n": len(synthetic),
            "synthetic_counts": synth_counts,
            "coco_n": len(coco_all),
            "coco_counts": coco_counts,
            "basis_source": "Synthetic-400 only",
            "basis": (
                "paper-style 2-D dual basis: horizontal left<->right; "
                "vertical under<->on"
            ),
            "intervention": (
                "h_sub += 0.5*delta_pair; "
                "h_ref -= 0.5*delta_pair"
            ),
            "decision_readout": (
                "first-step full-vocabulary logit/logprob of the per-sample "
                "randomly mapped A/B/C/D option assigned to the edited relation"
            ),
            "primary_rule": (
                "for each relation independently: toward edit should raise "
                "its own decision logit; away edit should lower its own logit"
            ),
            "uses_opposite_relation_logit": False,
            "note": (
                "This tests spatial edit -> final matching-logit control. "
                "It is not yet the L25 Top-K mediation/blocking experiment."
            ),
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[SAVED] {out / 'per_sample_effects.csv'}")
        print(f"[SAVED] {out / 'layer_relation_summary.csv'}")
        print(f"[SAVED] {out / 'layer_principle_summary.csv'}")
        print(f"[SAVED] {out / 'analysis_summary.txt'}")

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
