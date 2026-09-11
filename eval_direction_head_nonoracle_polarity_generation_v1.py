#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_direction_head_nonoracle_polarity_generation_v1.py

Purpose
=======
Take the polarity predictions already produced by:

    diagnose_nonoracle_update_polarity_v1.py

and test them with ACTUAL greedy generation.

This isolates the current HOW question:

    fixed global WHERE scaffold
        +
    non-oracle sample-specific Direction-Head polarity
        ->
    generation accuracy

The fixed WHERE positions are read from the completed global-scaffold run:

    resolved_global_slots_per_sample.csv

The non-oracle sign for every (sample, layer, position) is read from:

    per_update_sign_predictions.csv

Specifically, generation uses ONLY:

    pred_sign_direction_head

plus optional non-oracle sample-level gates:

    selector_margin
    selector_confidence
    route_direction_head != baseline_prediction

No GT relation / oracle B is used by the non-oracle generation conditions.

Important caveat
================
If the global scaffold was discovered with --split-mode all_data, WHERE still
contains discovery-set GT-writer leakage.  Therefore this script isolates
NON-ORACLE HOW, not a fully non-oracle end-to-end method.

Main conditions
===============
dir_signed_all
    Apply Direction-Head predicted +/- sign to every fixed-scaffold update.

dir_margin_<t>
    Apply the signed trajectory only when the Direction-Head selector margin
    for that sample is >= t.  Otherwise leave the sample untouched.

dir_conf_<t>
    Same, gated by selector confidence.

dir_disagree
    Edit only when Direction-Head predicted relation differs from the baseline
    generated relation.  This is a fully non-oracle error-disagreement trigger.

dir_disagree_margin_<t>
    Require BOTH disagreement with baseline and Direction-Head margin >= t.

oracle_signed_reference
    Optional diagnostic ceiling: uses oracle_sign from the diagnostic CSV.
    It should reproduce the previous global_fixed_signed result closely.
    This is explicitly ORACLE and is not part of the proposed method.

Recommended N80 run
===================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_direction_head_nonoracle_polarity_generation_v1.py \
  --model qwen-3b \
  --global-run-dir output/qwen3b_global_fixed_l26_top10_traj_n80_v1 \
  --polarity-dir output/qwen3b_global_fixed_polarity_diag_n80_v1 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --update-layers 20-26 \
  --scale 0.5 \
  --margin-thresholds 0.02,0.05,0.10 \
  --confidence-thresholds 0.30,0.35 \
  --include-oracle-reference \
  --output-dir output/qwen3b_direction_polarity_generation_n80_v1 \
  --overwrite

Outputs
=======
generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
condition_trigger_summary.csv
sign_replay_check.csv
analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import random
import re
import shutil
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_real_causal_token_update_gating_v1 as gate
import eval_l26_horizontal_top7_real_update_trajectory_v1 as l26


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    p.add_argument("--global-run-dir", required=True)
    p.add_argument("--polarity-dir", required=True)
    p.add_argument("--prior-real-update-dir", required=True)

    p.add_argument("--update-layers", default="20-26")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument(
        "--margin-thresholds",
        default="0.02,0.05,0.10",
    )
    p.add_argument(
        "--confidence-thresholds",
        default="0.30,0.35",
    )
    p.add_argument(
        "--include-oracle-reference",
        action="store_true",
        help="Also rerun oracle sign as a ceiling/replay check.",
    )

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all samples in polarity diagnostic.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    out = set()
    for x in str(text).split(","):
        x = x.strip().upper().replace("L", "")
        if not x:
            continue
        if "-" in x:
            a, b = x.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(x))
    return sorted(out)


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return {
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
    }.get(s, s)


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def threshold_tag(x: float):
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


def load_inputs(global_dir: Path, polarity_dir: Path, update_layers):
    resolved_path = global_dir / "resolved_global_slots_per_sample.csv"
    sign_path = polarity_dir / "per_update_sign_predictions.csv"
    route_path = polarity_dir / "sample_relation_routes.csv"

    for p in (resolved_path, sign_path, route_path):
        if not p.exists():
            raise FileNotFoundError(p)

    resolved = pd.read_csv(resolved_path)
    signs = pd.read_csv(sign_path)
    routes = pd.read_csv(route_path)

    resolved["sid"] = pd.to_numeric(resolved["sid"], errors="raise").astype(int)
    resolved["position"] = pd.to_numeric(
        resolved["position"], errors="coerce"
    ).fillna(-1).astype(int)

    if "resolved" in resolved.columns:
        if resolved["resolved"].dtype != bool:
            resolved["resolved"] = (
                resolved["resolved"].astype(str).str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
            )
        resolved = resolved[resolved["resolved"]].copy()

    resolved = resolved[resolved["position"] >= 0].copy()

    signs["sid"] = pd.to_numeric(signs["sid"], errors="raise").astype(int)
    signs["update_layer"] = pd.to_numeric(
        signs["update_layer"], errors="raise"
    ).astype(int)
    signs["position"] = pd.to_numeric(
        signs["position"], errors="raise"
    ).astype(int)
    signs = signs[signs["update_layer"].isin(set(update_layers))].copy()

    for c in ("pred_sign_direction_head", "oracle_sign"):
        if c in signs.columns:
            signs[c] = pd.to_numeric(signs[c], errors="coerce").fillna(0).astype(int)

    routes["sid"] = pd.to_numeric(routes["sid"], errors="raise").astype(int)
    routes["gt"] = routes["gt"].map(canon_rel)
    routes["baseline_prediction"] = routes["baseline_prediction"].map(canon_rel)
    routes["route_direction_head"] = routes["route_direction_head"].map(canon_rel)
    routes["selector_margin"] = pd.to_numeric(
        routes["selector_margin"], errors="coerce"
    )
    routes["selector_confidence"] = pd.to_numeric(
        routes["selector_confidence"], errors="coerce"
    )

    return resolved, signs, routes


def build_specs(resolved_sid: pd.DataFrame, max_layer: int):
    specs = []
    seen = set()

    sort_cols = [c for c in ("global_rank", "position") if c in resolved_sid.columns]
    x = resolved_sid.sort_values(sort_cols) if sort_cols else resolved_sid

    for r in x.itertuples():
        p = int(r.position)
        if p in seen:
            continue
        seen.add(p)

        specs.append(
            {
                "real_position": p,
                "max_target_layer": int(max_layer),
                "min_target_layer": int(max_layer),
                "best_rank": int(getattr(r, "global_rank", len(specs) + 1)),
                "token": str(getattr(r, "token", "")),
                "category": str(getattr(r, "category", "")),
                "broad_category": str(getattr(r, "broad_category", "")),
            }
        )
    return specs


def build_sign_lookup(signs_sid: pd.DataFrame, sign_col: str):
    lookup = {}
    for r in signs_sid.itertuples():
        key = (int(r.update_layer), int(r.position))
        lookup[key] = int(getattr(r, sign_col))
    return lookup


def signed_entries_from_lookup(entries, lookup):
    """
    Reuse gate.build_real_update_patch_map by writing the predicted sign into
    real_update_decision_score.  Only the sign matters for real_signed.
    """
    out = []
    missing = []

    for e in entries:
        key = (int(e["update_layer"]), int(e["real_position"]))
        if key not in lookup:
            missing.append(key)
            continue

        sgn = int(lookup[key])
        ee = dict(e)
        ee["real_update_decision_score"] = float(sgn)
        out.append(ee)

    return out, missing


def should_trigger(condition, route_row):
    margin = float(route_row["selector_margin"])
    conf = float(route_row["selector_confidence"])
    dir_rel = canon_rel(route_row["route_direction_head"])
    base_rel = canon_rel(route_row["baseline_prediction"])

    if condition == "dir_signed_all":
        return True

    if condition == "dir_disagree":
        return dir_rel in REL and base_rel in REL and dir_rel != base_rel

    m = re.fullmatch(r"dir_margin_(.+)", condition)
    if m:
        t = float(m.group(1).replace("p", "."))
        return np.isfinite(margin) and margin >= t

    m = re.fullmatch(r"dir_conf_(.+)", condition)
    if m:
        t = float(m.group(1).replace("p", "."))
        return np.isfinite(conf) and conf >= t

    m = re.fullmatch(r"dir_disagree_margin_(.+)", condition)
    if m:
        t = float(m.group(1).replace("p", "."))
        return (
            dir_rel in REL
            and base_rel in REL
            and dir_rel != base_rel
            and np.isfinite(margin)
            and margin >= t
        )

    raise ValueError(condition)


def condition_list(margin_thresholds, confidence_thresholds, include_oracle):
    out = ["dir_signed_all", "dir_disagree"]

    for t in margin_thresholds:
        tag = threshold_tag(t)
        out.append(f"dir_margin_{tag}")
        out.append(f"dir_disagree_margin_{tag}")

    for t in confidence_thresholds:
        tag = threshold_tag(t)
        out.append(f"dir_conf_{tag}")

    if include_oracle:
        out.append("oracle_signed_reference")

    # dedupe, preserve order
    return list(dict.fromkeys(out))


def summarize_trigger(gen_df):
    rows = []
    x = gen_df[gen_df["condition"] != "baseline"].copy()

    for cond, g in x.groupby("condition"):
        rows.append(
            {
                "condition": cond,
                "N": int(g["sid"].nunique()),
                "triggered_samples": int(
                    g.loc[g["triggered"], "sid"].nunique()
                ),
                "trigger_rate": float(
                    g.groupby("sid")["triggered"].first().mean()
                ),
                "mean_patched_updates": float(g["n_patched_updates"].mean()),
                "mean_positive_updates": float(g["n_positive_updates"].mean()),
                "mean_negative_updates": float(g["n_negative_updates"].mean()),
            }
        )

    return pd.DataFrame(rows)


def load_prior_reference(global_dir: Path, sids):
    p = global_dir / "generation_per_sample.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
    df = df[df["sid"].isin(set(map(int, sids)))].copy()

    if "correct" in df.columns and df["correct"].dtype != bool:
        df["correct"] = (
            df["correct"].astype(str).str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    rows = []
    for cond in [
        "global_fixed_signed",
        "global_fixed_all_amplify",
        "global_fixed_positive",
        "global_fixed_negative_cancel",
        "global_fixed_l26_only_signed",
        "sample_l26_signed_reference",
    ]:
        g = df[df["condition"].astype(str) == cond]
        if len(g):
            rows.append(
                {
                    "condition": cond,
                    "N": int(len(g)),
                    "accuracy": float(g["correct"].astype(bool).mean()),
                }
            )

    return pd.DataFrame(rows)


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    update_layers = parse_ints(a.update_layers)
    margin_thresholds = parse_floats(a.margin_thresholds)
    confidence_thresholds = parse_floats(a.confidence_thresholds)

    if not update_layers:
        raise ValueError("No update layers")

    conditions = condition_list(
        margin_thresholds,
        confidence_thresholds,
        a.include_oracle_reference,
    )

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    global_dir = Path(a.global_run_dir)
    polarity_dir = Path(a.polarity_dir)
    prior_dir = Path(a.prior_real_update_dir)

    resolved, signs, routes = load_inputs(
        global_dir,
        polarity_dir,
        update_layers,
    )

    # Common diagnostic cohort only.
    sids = sorted(
        set(resolved["sid"])
        & set(signs["sid"])
        & set(routes["sid"])
    )

    # Optional stratified debug cap.
    prior_cohort, _, prior_metadata, _ = l26.load_prior_run(prior_dir)
    prior_cohort = prior_cohort[prior_cohort["sid"].isin(sids)].copy()

    if int(a.eval_max_samples) > 0:
        prior_cohort = l26.stratified_cap_df(
            prior_cohort,
            int(a.eval_max_samples),
            int(a.seed) + 211,
        )
        sids = sorted(prior_cohort["sid"].astype(int).tolist())
    else:
        sids = sorted(prior_cohort["sid"].astype(int).tolist())

    resolved = resolved[resolved["sid"].isin(sids)].copy()
    signs = signs[signs["sid"].isin(sids)].copy()
    routes = routes[routes["sid"].isin(sids)].copy()

    two, eval_meta, rec_by_sid, prompts, records = l26.load_dataset(
        a,
        prior_cohort,
    )
    eval_meta = [m for m in eval_meta if int(m["sid"]) in set(sids)]

    route_by_sid = routes.set_index("sid").to_dict("index")

    model = processor = None
    generation_rows = []
    replay_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = l26.load_model(a, two)
        device = torch.device(a.device)

        print("=" * 190)
        print("DIRECTION-HEAD NON-ORACLE POLARITY -> ACTUAL GENERATION")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(eval_meta)}")
        print(f"update_layers={update_layers}")
        print(f"scale={a.scale}")
        print(f"conditions={conditions}")
        print()
        print(
            "Non-oracle conditions use ONLY pred_sign_direction_head plus optional "
            "selector margin/confidence/baseline-disagreement gates."
        )
        print(
            "GT is used only to score final accuracy. "
            "oracle_signed_reference, if requested, is an explicit ceiling."
        )
        print()

        for m in tqdm(eval_meta, desc="NONORACLE generation"):
            sid = int(m["sid"])
            real = None

            # Baseline row from prior run, not re-generated.
            generation_rows.append(
                {
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "baseline",
                    "scale": 0.0,
                    "prediction": m["baseline_prediction"],
                    "correct": bool(m["baseline_correct"]),
                    "triggered": False,
                    "route_direction_head": route_by_sid[sid]["route_direction_head"],
                    "selector_margin": route_by_sid[sid]["selector_margin"],
                    "selector_confidence": route_by_sid[sid]["selector_confidence"],
                    "n_patched_updates": 0,
                    "n_positive_updates": 0,
                    "n_negative_updates": 0,
                    "text": "",
                }
            )

            try:
                rs = resolved[resolved["sid"] == sid].copy()
                ss = signs[signs["sid"] == sid].copy()
                route_row = route_by_sid[sid]

                specs = build_specs(rs, max(update_layers))
                if not specs:
                    raise RuntimeError("No resolved global scaffold positions")

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )

                capture_layers = sorted(
                    set(update_layers + [L - 1 for L in update_layers])
                )
                real_states = gate.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                entries = gate.build_real_updates(
                    sid=sid,
                    gt=m["gt"],
                    baseline_correct=bool(m["baseline_correct"]),
                    specs=specs,
                    r2n={},
                    real_states=real_states,
                    no_states={},
                    update_layers=update_layers,
                    exclude_target_layer=False,
                )
                if not entries:
                    raise RuntimeError("No REAL updates")

                dir_lookup = build_sign_lookup(
                    ss,
                    "pred_sign_direction_head",
                )
                dir_entries, missing_dir = signed_entries_from_lookup(
                    entries,
                    dir_lookup,
                )

                if missing_dir:
                    raise RuntimeError(
                        f"Missing Direction sign predictions for "
                        f"{len(missing_dir)} updates, e.g. {missing_dir[:5]}"
                    )

                oracle_entries = []
                missing_oracle = []
                if a.include_oracle_reference:
                    oracle_lookup = build_sign_lookup(ss, "oracle_sign")
                    oracle_entries, missing_oracle = signed_entries_from_lookup(
                        entries,
                        oracle_lookup,
                    )
                    if missing_oracle:
                        raise RuntimeError(
                            f"Missing oracle signs for {len(missing_oracle)} updates"
                        )

                # Sign replay bookkeeping.
                for e in dir_entries:
                    key = (
                        int(e["update_layer"]),
                        int(e["real_position"]),
                    )
                    source = ss[
                        (ss["update_layer"] == key[0])
                        & (ss["position"] == key[1])
                    ].iloc[0]

                    replay_rows.append(
                        {
                            "sid": sid,
                            "update_layer": key[0],
                            "position": key[1],
                            "pred_sign_direction_head": int(
                                source["pred_sign_direction_head"]
                            ),
                            "oracle_sign": int(source["oracle_sign"]),
                            "oracle_B": float(source["oracle_B"]),
                            "Bhat_direction_head": float(
                                source["Bhat_direction_head"]
                            ),
                            "selector_margin": float(
                                source["selector_margin"]
                            ),
                            "selector_confidence": float(
                                source["selector_confidence"]
                            ),
                        }
                    )

                # ---------------------------------------------------------
                # Generate each condition.
                # ---------------------------------------------------------
                for cond in conditions:
                    if cond == "oracle_signed_reference":
                        active = True
                        signed = oracle_entries
                    else:
                        active = should_trigger(cond, route_row)
                        signed = dir_entries

                    if active:
                        patch_map, counts = gate.build_real_update_patch_map(
                            signed,
                            "real_signed",
                            float(a.scale),
                            0.0,
                        )
                    else:
                        patch_map = {}
                        counts = {
                            "patched": 0,
                            "positive": 0,
                            "negative": 0,
                            "neutral": len(signed),
                        }

                    # If abstained, reuse prior baseline prediction. This is
                    # behaviorally identical to generating with empty patch_map
                    # and avoids wasting GPU time.
                    if not active:
                        pred = m["baseline_prediction"]
                        text = ""
                    else:
                        pred, text = gate.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=patch_map,
                            max_new_tokens=a.max_new_tokens,
                        )

                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": m["gt"],
                            "condition": cond,
                            "scale": float(a.scale),
                            "prediction": pred,
                            "correct": pred == m["gt"],
                            "triggered": bool(active),
                            "route_direction_head": route_row["route_direction_head"],
                            "selector_margin": float(route_row["selector_margin"]),
                            "selector_confidence": float(
                                route_row["selector_confidence"]
                            ),
                            "n_patched_updates": int(counts["patched"]),
                            "n_positive_updates": int(counts["positive"]),
                            "n_negative_updates": int(counts["negative"]),
                            "text": text,
                        }
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-80:],
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
                )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gen_df = pd.DataFrame(generation_rows)
        replay_df = pd.DataFrame(replay_rows)

        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )
        replay_df.to_csv(
            outdir / "sign_replay_check.csv",
            index=False,
        )

        gen_summary = gate.summarize_generation(gen_df)
        gen_rel = gate.summarize_generation_by_relation(gen_df)
        trigger_summary = summarize_trigger(gen_df)
        prior_ref = load_prior_reference(
            global_dir,
            gen_df["sid"].unique().tolist(),
        )

        gen_summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )
        gen_rel.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )
        trigger_summary.to_csv(
            outdir / "condition_trigger_summary.csv",
            index=False,
        )
        prior_ref.to_csv(
            outdir / "prior_global_run_reference.csv",
            index=False,
        )

        # Compare sign accuracy on the exact generated cohort.
        if len(replay_df):
            replay_df["sign_correct"] = (
                replay_df["pred_sign_direction_head"].astype(int)
                == replay_df["oracle_sign"].astype(int)
            )
            sign_acc = float(replay_df["sign_correct"].mean())
            weights = np.abs(replay_df["oracle_B"].to_numpy(float))
            correct = replay_df["sign_correct"].to_numpy(float)
            weighted_sign_acc = (
                float(np.sum(weights * correct) / np.sum(weights))
                if np.sum(weights) > 0 else np.nan
            )
        else:
            sign_acc = np.nan
            weighted_sign_acc = np.nan

        report = [
            "=" * 190,
            "DIRECTION-HEAD NON-ORACLE POLARITY -> ACTUAL GENERATION",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"N={gen_df['sid'].nunique()}",
            f"update layers={update_layers}",
            f"scale={a.scale}",
            f"Direction sign accuracy on exact cohort={sign_acc:.4f}",
            f"Direction |B|-weighted sign accuracy={weighted_sign_acc:.4f}",
            "",
            "GENERATION",
            "-" * 190,
            gen_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "TRIGGER / COVERAGE",
            "-" * 190,
            trigger_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "PREVIOUS GLOBAL-RUN REFERENCES",
            "-" * 190,
            (
                prior_ref.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(prior_ref)
                else "No prior generation reference found."
            ),
            "",
            "Interpretation:",
            "  dir_signed_all = pure non-oracle HOW test on the fixed scaffold.",
            "  dir_margin_* = confidence-abstaining variants.",
            "  dir_disagree* = edit only when independent spatial belief disagrees",
            "  with the model's baseline answer; this is a non-oracle repair trigger.",
            "  oracle_signed_reference is a ceiling only.",
            "",
            "Caveat:",
            "  If the global scaffold came from split_mode=all_data, WHERE is still",
            "  GT-writer-discovered on this cohort. This run isolates HOW only.",
        ]

        report_text = "\n".join(report) + "\n"
        print(report_text)
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        metadata = {
            "script": "eval_direction_head_nonoracle_polarity_generation_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "global_run_dir": str(global_dir),
            "polarity_dir": str(polarity_dir),
            "prior_real_update_dir": str(prior_dir),
            "update_layers": update_layers,
            "scale": float(a.scale),
            "margin_thresholds": margin_thresholds,
            "confidence_thresholds": confidence_thresholds,
            "conditions": conditions,
            "nonoracle_how": (
                "Direction-Head relation route + predicted update polarity "
                "precomputed by diagnose_nonoracle_update_polarity_v1.py"
            ),
            "gt_usage_nonoracle_conditions": "final accuracy evaluation only",
            "oracle_reference_enabled": bool(a.include_oracle_reference),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
