#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_nonoracle_hard_globalK_all440_v1.py

Focused experiment
==================
Evaluate the CURRENT BEST non-oracle HOW method on ALL COCO_two samples while
varying only the size of the shared global canonical WHERE scaffold.

Default K sweep:
    global K = 7, 10, 15

The K sets are NESTED PREFIXES of ONE frozen global ranking:

    K7  = ranks 1..7
    K10 = ranks 1..10
    K15 = ranks 1..15

The ranking is read from:
    <global-template-dir>/global_slot_ranking.csv

This is preferable to independently rediscovering Top7/Top10/Top15 because the
experiment then isolates ONE variable only:

    "How many shared canonical positions do we keep?"

The script also verifies that ranking[:10] exactly matches the old
global_topk_slots.csv, so K10 is the same WHERE used by the original Top10 run.

Current best non-oracle HOW
===========================

1) Frozen Direction-Head selector gives one sample-level spatial belief:

       r_hat in {left,right,above,below}

   This script reads the ALREADY SAVED selector prediction directly from:

       synthetic400_to_coco440_predictions.csv
       pred_top10_equal           (default)

   Therefore no selector reconstruction is needed and no COCO GT is used.

2) Only when the independent spatial belief DISAGREES with the baseline
   generated relation do we repair the sample:

       r_hat != r_baseline

   If they agree, all K conditions reuse the baseline answer and the model is
   not rerun.

3) For every resolved global canonical position p and update layer L:

       a_REAL[L,p] = h_REAL[L,p] - h_REAL[L-1,p]

4) For the Direction route r_hat, find the strongest clean sequence-score
   competitor among the other three candidates:

       foil = argmax_{r != r_hat} S_r

5) Use the model's own downstream Jacobian to decide the sign of EACH actual
   block update:

       B_hat[L,p]
           = < a_REAL[L,p],
               grad_h (S_r_hat - S_foil) >

6) Actual generation patch, default alpha=0.5:

       B_hat > 0  -> + alpha * a_REAL
       B_hat < 0  -> - alpha * a_REAL

This is the same hard Direction -> candidate-answer-gradient HOW method that
previously reached 71.25% -> 81.25% on the original N80 experiment.

Efficiency
==========
For the default `disagree` trigger, expensive gradient + generation work is
performed ONLY on samples where Direction prediction != baseline prediction.
The remaining samples are copied from baseline for K7/K10/K15.

For each triggered sample, the script:
    * resolves the UNION Top15 scaffold once;
    * captures REAL states once;
    * computes candidate sequence gradients once;
    * reuses those results for K7/K10/K15.

Thus K is the only changing intervention variable.

Seen / unseen generalization
============================
If the frozen global ranking came from the old N80 all_data discovery run, the
full-440 result is additionally split into:

    template_seen
    template_unseen

so we can tell whether a K only works on samples that helped discover WHERE.

Recommended full-440 run
========================
CUDA_VISIBLE_DEVICES=0 python -u eval_nonoracle_hard_globalK_all440_v1.py \
  --model qwen-3b \
  --global-template-dir output/qwen3b_global_fixed_l26_top10_traj_n80_v1 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --selector-csv output/qwen3b_direction_selector_syn400_to_coco440/synthetic400_to_coco440_predictions.csv \
  --selector-method top10_equal \
  --global-ks 7,10,15 \
  --update-layers 20-26 \
  --scale 0.5 \
  --eval-max-samples 0 \
  --output-dir output/qwen3b_nonoracle_hard_globalK_7_10_15_all440_v1 \
  --overwrite

Optional:
    --include-hard-all
        Also run the same signed intervention even when Direction == baseline.
        This is more expensive.

    --include-oracle-reference
        Also compute/generate oracle signed references for every K. This is
        substantially more expensive and is NOT needed for the main non-oracle
        K comparison.

Outputs
=======
generation_summary.csv
generation_by_relation.csv
generation_seen_vs_unseen.csv
generation_per_sample.csv
globalK_triggered_coverage.csv
per_update_nonoracle_B.csv
route_summary.csv
globalK_template_slots.csv
analysis_summary.txt
errors.jsonl
metadata.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import random
import shutil
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_real_causal_token_update_gating_v1 as gate
import eval_l26_horizontal_top7_real_update_trajectory_v1 as l26
import eval_global_fixed_l26_top10_trajectory_gating_v1 as gfix


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


# =============================================================================
# CLI / utility
# =============================================================================

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

    p.add_argument("--global-template-dir", required=True)
    p.add_argument("--prior-real-update-dir", required=True)

    p.add_argument(
        "--selector-csv",
        required=True,
        help="synthetic400_to_coco440_predictions.csv",
    )
    p.add_argument("--selector-method", default="top10_equal")

    p.add_argument("--global-ks", default="7,10,15")
    p.add_argument("--update-layers", default="20-26")
    p.add_argument("--scale", type=float, default=0.5)

    p.add_argument(
        "--include-hard-all",
        action="store_true",
        help="Also edit samples where Direction route == baseline.",
    )
    p.add_argument(
        "--include-oracle-reference",
        action="store_true",
        help="Also run GT-sign ceiling for each K; expensive.",
    )

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all samples from prior all440 run.",
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


# =============================================================================
# Frozen global ranking
# =============================================================================

def load_global_ranking(template_dir: Path, ks: List[int]):
    ranking_path = template_dir / "global_slot_ranking.csv"
    legacy_path = template_dir / "global_topk_slots.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)

    ranking = pd.read_csv(ranking_path)

    if "global_rank" not in ranking.columns:
        raise RuntimeError(
            f"{ranking_path} missing global_rank"
        )

    ranking["global_rank"] = pd.to_numeric(
        ranking["global_rank"], errors="raise"
    ).astype(int)
    ranking = ranking.sort_values("global_rank").reset_index(drop=True)

    max_k = max(ks)
    if len(ranking) < max_k:
        raise RuntimeError(
            f"Frozen ranking has only {len(ranking)} slots, cannot evaluate K={max_k}."
        )

    # Verify that K10 is exactly the old Top10 template when possible.
    legacy_check = {
        "available": False,
        "exact_prefix_match": None,
        "legacy_K": 0,
    }

    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path).sort_values("global_rank")
        legacy_check["available"] = True
        legacy_check["legacy_K"] = int(len(legacy))

        n = min(10, len(legacy), len(ranking))
        old_slots = legacy.head(n)["canonical_slot"].astype(str).tolist()
        rank_slots = ranking.head(n)["canonical_slot"].astype(str).tolist()
        legacy_check["exact_prefix_match"] = old_slots == rank_slots

        if len(legacy) >= 10 and 10 in ks and not legacy_check["exact_prefix_match"]:
            raise RuntimeError(
                "global_slot_ranking[:10] does NOT match old global_topk_slots.csv. "
                "Refusing to call K10 the original WHERE template."
            )

    discovery_sids = set()
    disc_path = template_dir / "discovery_sample_l26_topk.csv"
    if disc_path.exists():
        d = pd.read_csv(disc_path)
        if "sid" in d.columns:
            discovery_sids = set(
                pd.to_numeric(d["sid"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

    return ranking, discovery_sids, legacy_check


# =============================================================================
# Frozen Direction routes
# =============================================================================

def load_selector_routes(path: Path, method: str):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "sid" not in df.columns:
        raise RuntimeError(f"{path} missing sid")

    col = f"pred_{method}"
    if col not in df.columns:
        candidates = [
            c for c in df.columns
            if c.startswith("pred_")
        ]
        raise RuntimeError(
            f"{path} missing {col}. Available prediction columns: {candidates[:50]}"
        )

    df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
    df["route"] = df[col].map(canon_rel)

    bad = df[~df["route"].isin(REL)]
    if len(bad):
        raise RuntimeError(
            f"Invalid selector relations in {col}: "
            f"{sorted(bad['route'].astype(str).unique().tolist())}"
        )

    keep = ["sid", "route"]
    for c in (f"confidence_{method}", f"margin_{method}", "gt"):
        if c in df.columns:
            keep.append(c)

    return df[keep].drop_duplicates("sid")


# =============================================================================
# Candidate gradients and B_hat
# =============================================================================

def strongest_other(scores: Dict[str, float], route: str):
    return max(
        (r for r in REL if r != route),
        key=lambda r: float(scores[r]),
    )


def attach_nonoracle_scores(entries, grad_route, grad_foil):
    scored = gate.attach_decision_scores(
        entries,
        grad_route,
        grad_foil,
    )
    # attach_decision_scores writes real_update_decision_score, exactly what
    # build_real_update_patch_map expects.
    return scored


def subset_scored_by_global_k(scored, pos_to_rank, K):
    out = []
    for e in scored:
        p = int(e["real_position"])
        rank = pos_to_rank.get(p, None)
        if rank is None:
            continue
        if int(rank) <= int(K):
            ee = dict(e)
            ee["global_rank"] = int(rank)
            out.append(ee)
    return out


# =============================================================================
# Summary
# =============================================================================

def summarize_generation(gen_df):
    base = (
        gen_df[gen_df["condition"] == "baseline"]
        .drop_duplicates("sid")
        .set_index("sid")
    )

    rows = []
    for cond, g in gen_df[
        gen_df["condition"] != "baseline"
    ].groupby("condition"):
        gg = g.drop_duplicates("sid").set_index("sid")
        sids = sorted(set(base.index) & set(gg.index))
        if not sids:
            continue

        b = base.loc[sids]
        p = gg.loc[sids]

        bc = b["correct"].astype(bool).to_numpy()
        pc = p["correct"].astype(bool).to_numpy()

        w2c = int(np.sum((~bc) & pc))
        c2w = int(np.sum(bc & (~pc)))

        rows.append(
            {
                "condition": cond,
                "global_k": int(p["global_k"].iloc[0]),
                "N": len(sids),
                "baseline_accuracy": float(np.mean(bc)),
                "patched_accuracy": float(np.mean(pc)),
                "gain": float(np.mean(pc) - np.mean(bc)),
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "net": w2c - c2w,
                "changed": int(
                    np.sum(
                        b["prediction"].astype(str).to_numpy()
                        != p["prediction"].astype(str).to_numpy()
                    )
                ),
                "repair_rate_on_wrong": (
                    w2c / max(int(np.sum(~bc)), 1)
                ),
                "preserve_rate_on_correct": (
                    1.0 - c2w / max(int(np.sum(bc)), 1)
                ),
                "trigger_rate": float(
                    p["triggered"].astype(bool).mean()
                ),
                "mean_resolved_positions": float(
                    p["resolved_global_positions"].mean()
                ),
                "mean_patched_updates": float(
                    p["n_patched_updates"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["patched_accuracy", "global_k"],
        ascending=[False, True],
    )


def grouped_summary(gen_df, group_col):
    rows = []
    for value, g in gen_df.groupby(group_col):
        s = summarize_generation(g)
        if len(s):
            s.insert(1, group_col, value)
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    ks = parse_ints(a.global_ks)
    update_layers = parse_ints(a.update_layers)

    if not ks:
        raise ValueError("--global-ks is empty")
    if not update_layers:
        raise ValueError("--update-layers is empty")

    max_k = max(ks)

    outdir = Path(a.output_dir)
    ensure_outdir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    template_dir = Path(a.global_template_dir)
    prior_dir = Path(a.prior_real_update_dir)

    ranking, discovery_sids, legacy_check = load_global_ranking(
        template_dir,
        ks,
    )
    ranking.head(max_k).to_csv(
        outdir / "globalK_template_slots.csv",
        index=False,
    )

    routes_df = load_selector_routes(
        Path(a.selector_csv),
        a.selector_method,
    )
    route_map = dict(zip(routes_df["sid"], routes_df["route"]))

    prior_cohort, _, prior_metadata, _ = l26.load_prior_run(
        prior_dir
    )
    prior_cohort = prior_cohort[
        prior_cohort["sid"].isin(route_map)
    ].copy()

    if int(a.eval_max_samples) > 0:
        prior_cohort = l26.stratified_cap_df(
            prior_cohort,
            int(a.eval_max_samples),
            int(a.seed) + 517,
        )

    two, eval_meta, rec_by_sid, prompts, records = l26.load_dataset(
        a,
        prior_cohort,
    )
    eval_meta = [
        m for m in eval_meta
        if int(m["sid"]) in route_map
    ]

    texts = prior_metadata.get(
        "candidate_texts",
        {
            "left": "left",
            "right": "right",
            "above": "above",
            "below": "below",
        },
    )
    texts = {canon_rel(k): str(v) for k, v in texts.items()}
    for r in REL:
        if r not in texts:
            raise RuntimeError(f"candidate_texts missing {r}: {texts}")

    reduction = str(
        prior_metadata.get("sequence_score_reduction", "mean")
    )
    if reduction not in ("mean", "sum"):
        reduction = "mean"

    model = processor = None
    generation_rows = []
    update_rows = []
    coverage_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = (
            l26.load_model(a, two)
        )
        device = torch.device(a.device)
        candidate_ids = gate.encode_candidate_ids(
            processor,
            texts,
        )

        print("=" * 190)
        print("FULL-COHORT NON-ORACLE HARD HOW: GLOBAL K SWEEP")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N eval={len(eval_meta)}")
        print(f"global Ks={ks} | maxK={max_k}")
        print(f"update layers={update_layers}")
        print(f"scale={a.scale}")
        print(f"selector={a.selector_method}")
        print(f"legacy K10 prefix check={legacy_check}")
        print(f"template discovery SIDs={len(discovery_sids)}")
        print(
            "Main method: Direction != baseline -> hard candidate-gradient signed repair"
        )
        print()

        for m in tqdm(eval_meta, desc="GLOBAL-K sweep"):
            sid = int(m["sid"])
            gt = canon_rel(m["gt"])
            baseline = canon_rel(m["baseline_prediction"])
            route = canon_rel(route_map[sid])
            disagree = route != baseline
            split = (
                "template_seen"
                if sid in discovery_sids
                else "template_unseen"
            )

            # baseline once.
            generation_rows.append(
                {
                    "sid": sid,
                    "gt": gt,
                    "condition": "baseline",
                    "global_k": 0,
                    "prediction": baseline,
                    "correct": bool(m["baseline_correct"]),
                    "triggered": False,
                    "route": route,
                    "route_correct": route == gt,
                    "baseline_prediction": baseline,
                    "route_disagrees_baseline": disagree,
                    "template_split": split,
                    "resolved_global_positions": 0,
                    "n_patched_updates": 0,
                    "text": "",
                }
            )

            # For the main disagree-triggered method, no model work is needed if
            # selector and baseline already agree, unless hard-all/oracle requested.
            need_model = (
                disagree
                or bool(a.include_hard_all)
                or bool(a.include_oracle_reference)
            )

            if not need_model:
                for K in ks:
                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "condition": f"K{K}_hard_disagree",
                            "global_k": K,
                            "prediction": baseline,
                            "correct": bool(m["baseline_correct"]),
                            "triggered": False,
                            "route": route,
                            "route_correct": route == gt,
                            "baseline_prediction": baseline,
                            "route_disagrees_baseline": disagree,
                            "template_split": split,
                            "resolved_global_positions": 0,
                            "n_patched_updates": 0,
                            "text": "",
                        }
                    )
                continue

            real = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = gfix.dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                mapping, canon_seq, sig = gfix.canonical_mapping(
                    model=model,
                    processor=processor,
                    batch=rb,
                    ids=ids,
                    subject=m["subject"],
                    reference=m["reference"],
                )

                # Resolve union Top-maxK once.
                top_union = ranking.head(max_k).copy()
                specs, resolved = gfix.make_specs_from_global_slots(
                    global_top=top_union,
                    mapping=mapping,
                    ids=ids,
                    cats=cats,
                    toks=toks,
                    selector_layer=max(update_layers),
                )

                if not specs:
                    raise RuntimeError(
                        "None of the global slots resolved"
                    )

                pos_to_rank = {
                    int(s["real_position"]): int(s["global_rank"])
                    for s in specs
                }

                resolved_by_k = {}
                for K in ks:
                    resolved_by_k[K] = int(
                        sum(
                            bool(rr.get("resolved", False))
                            and int(rr["global_rank"]) <= K
                            for rr in resolved
                        )
                    )
                    coverage_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "global_k": K,
                            "template_split": split,
                            "triggered_main_method": bool(disagree),
                            "resolved_positions": resolved_by_k[K],
                            "full_resolution": resolved_by_k[K] == K,
                        }
                    )

                capture_layers = sorted(
                    set(
                        update_layers
                        + [L - 1 for L in update_layers]
                    )
                )
                real_states = gate.capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                entries = gate.build_real_updates(
                    sid=sid,
                    gt=gt,
                    baseline_correct=bool(m["baseline_correct"]),
                    specs=specs,
                    r2n={},
                    real_states=real_states,
                    no_states={},
                    update_layers=update_layers,
                    exclude_target_layer=False,
                )
                if not entries:
                    raise RuntimeError(
                        "No REAL trajectory updates"
                    )

                grad_layers = sorted(
                    set(int(e["update_layer"]) for e in entries)
                )

                # Compute all candidate scores/gradients ONCE. This also lets us
                # optionally form the oracle reference with no extra backward.
                scores = {}
                grads = {}
                for r in REL:
                    sc, gr = gate.sequence_score_and_grads(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        answer_ids=candidate_ids[r],
                        reduction=reduction,
                        grad_layers=grad_layers,
                    )
                    scores[r] = float(sc)
                    grads[r] = gr

                foil = strongest_other(scores, route)
                scored_nonoracle = attach_nonoracle_scores(
                    entries,
                    grads[route],
                    grads[foil],
                )

                if not scored_nonoracle:
                    raise RuntimeError(
                        "No non-oracle signed updates"
                    )

                oracle_scored = None
                oracle_foil = None
                if a.include_oracle_reference:
                    oracle_foil = strongest_other(scores, gt)
                    oracle_scored = gate.attach_decision_scores(
                        entries,
                        grads[gt],
                        grads[oracle_foil],
                    )

                # Export B_hat once for union K.
                for e in scored_nonoracle:
                    p = int(e["real_position"])
                    rank = pos_to_rank.get(p, -1)
                    update_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "baseline_correct": bool(m["baseline_correct"]),
                            "baseline_prediction": baseline,
                            "route": route,
                            "route_correct": route == gt,
                            "route_disagrees_baseline": disagree,
                            "template_split": split,
                            "route_foil": foil,
                            "global_rank": int(rank),
                            "update_layer": int(e["update_layer"]),
                            "position": p,
                            "token": str(e["token"]),
                            "category": str(e["category"]),
                            "B_hat": float(
                                e["real_update_decision_score"]
                            ),
                            "B_hat_sign": int(
                                np.sign(
                                    float(
                                        e["real_update_decision_score"]
                                    )
                                )
                            ),
                            "real_update_norm": float(
                                e["real_update_norm"]
                            ),
                        }
                    )

                # -----------------------------------------------------
                # K7 / K10 / K15 main generation.
                # -----------------------------------------------------
                for K in ks:
                    sub = subset_scored_by_global_k(
                        scored_nonoracle,
                        pos_to_rank,
                        K,
                    )

                    # Main best method: edit only disagreement samples.
                    if disagree and sub:
                        patch_map, counts = gate.build_real_update_patch_map(
                            sub,
                            "real_signed",
                            float(a.scale),
                            0.0,
                        )
                        pred, text = gate.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=patch_map,
                            max_new_tokens=a.max_new_tokens,
                        )
                    else:
                        counts = {"patched": 0}
                        pred, text = baseline, ""

                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "condition": f"K{K}_hard_disagree",
                            "global_k": K,
                            "prediction": pred,
                            "correct": pred == gt,
                            "triggered": bool(disagree),
                            "route": route,
                            "route_correct": route == gt,
                            "baseline_prediction": baseline,
                            "route_disagrees_baseline": disagree,
                            "template_split": split,
                            "resolved_global_positions": resolved_by_k[K],
                            "n_patched_updates": int(counts["patched"]),
                            "text": text,
                        }
                    )

                    # Optional exact same HOW without disagreement trigger.
                    if a.include_hard_all:
                        if sub:
                            patch_map, counts = gate.build_real_update_patch_map(
                                sub,
                                "real_signed",
                                float(a.scale),
                                0.0,
                            )
                            pred2, text2 = gate.generate_with_patch(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                patch_map=patch_map,
                                max_new_tokens=a.max_new_tokens,
                            )
                        else:
                            counts = {"patched": 0}
                            pred2, text2 = baseline, ""

                        generation_rows.append(
                            {
                                "sid": sid,
                                "gt": gt,
                                "condition": f"K{K}_hard_all",
                                "global_k": K,
                                "prediction": pred2,
                                "correct": pred2 == gt,
                                "triggered": True,
                                "route": route,
                                "route_correct": route == gt,
                                "baseline_prediction": baseline,
                                "route_disagrees_baseline": disagree,
                                "template_split": split,
                                "resolved_global_positions": resolved_by_k[K],
                                "n_patched_updates": int(counts["patched"]),
                                "text": text2,
                            }
                        )

                    # Optional oracle ceiling for same fixed K.
                    if a.include_oracle_reference:
                        osub = subset_scored_by_global_k(
                            oracle_scored,
                            pos_to_rank,
                            K,
                        )
                        if osub:
                            patch_map, counts = gate.build_real_update_patch_map(
                                osub,
                                "real_signed",
                                float(a.scale),
                                0.0,
                            )
                            pred3, text3 = gate.generate_with_patch(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                patch_map=patch_map,
                                max_new_tokens=a.max_new_tokens,
                            )
                        else:
                            counts = {"patched": 0}
                            pred3, text3 = baseline, ""

                        generation_rows.append(
                            {
                                "sid": sid,
                                "gt": gt,
                                "condition": f"K{K}_oracle_signed",
                                "global_k": K,
                                "prediction": pred3,
                                "correct": pred3 == gt,
                                "triggered": True,
                                "route": route,
                                "route_correct": route == gt,
                                "baseline_prediction": baseline,
                                "route_disagrees_baseline": disagree,
                                "template_split": split,
                                "resolved_global_positions": resolved_by_k[K],
                                "n_patched_updates": int(counts["patched"]),
                                "text": text3,
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
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =================================================================
        # Save
        # =================================================================
        gen_df = pd.DataFrame(generation_rows)
        upd_df = pd.DataFrame(update_rows)
        cov_df = pd.DataFrame(coverage_rows)

        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )
        upd_df.to_csv(
            outdir / "per_update_nonoracle_B.csv",
            index=False,
        )
        cov_df.to_csv(
            outdir / "globalK_triggered_coverage.csv",
            index=False,
        )

        summary = summarize_generation(gen_df)
        by_rel = grouped_summary(gen_df, "gt")
        by_split = grouped_summary(gen_df, "template_split")

        summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )
        by_rel.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )
        by_split.to_csv(
            outdir / "generation_seen_vs_unseen.csv",
            index=False,
        )

        # Route statistics across all eval samples.
        base_rows = (
            gen_df[gen_df["condition"] == "baseline"]
            .drop_duplicates("sid")
        )
        route_rows = []
        for name, g in [
            ("all", base_rows),
            ("baseline_wrong", base_rows[~base_rows["correct"]]),
            ("baseline_correct", base_rows[base_rows["correct"]]),
            (
                "template_unseen",
                base_rows[
                    base_rows["template_split"] == "template_unseen"
                ],
            ),
        ]:
            if not len(g):
                continue
            route_rows.append(
                {
                    "cohort": name,
                    "N": int(len(g)),
                    "direction_relation_accuracy": float(
                        g["route_correct"].mean()
                    ),
                    "disagreement_rate_with_baseline": float(
                        g["route_disagrees_baseline"].mean()
                    ),
                }
            )

        route_summary = pd.DataFrame(route_rows)
        route_summary.to_csv(
            outdir / "route_summary.csv",
            index=False,
        )

        # Coverage summary for triggered samples.
        cov_summary = pd.DataFrame()
        if len(cov_df):
            cov_summary = (
                cov_df[cov_df["triggered_main_method"]]
                .groupby("global_k", as_index=False)
                .agg(
                    N_triggered=("sid", "nunique"),
                    mean_resolved_positions=("resolved_positions", "mean"),
                    full_resolution_fraction=("full_resolution", "mean"),
                )
            )
            cov_summary.to_csv(
                outdir / "globalK_triggered_coverage_summary.csv",
                index=False,
            )

        print("=" * 190)
        print("FULL-COHORT GLOBAL-K RESULTS")
        print("=" * 190)
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\nROUTE SUMMARY")
        print("-" * 190)
        print(
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        if len(cov_summary):
            print("\nTRIGGERED-SAMPLE SLOT COVERAGE")
            print("-" * 190)
            print(
                cov_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

        if len(by_split):
            print("\nSEEN vs UNSEEN")
            print("-" * 190)
            print(
                by_split.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

        report = [
            "=" * 190,
            "FULL-COHORT NON-ORACLE HARD HOW: GLOBAL K SWEEP",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"N eval={base_rows['sid'].nunique()}",
            f"global Ks={ks}",
            f"scale={a.scale}",
            f"update layers={update_layers}",
            f"selector method={a.selector_method}",
            f"legacy K10 check={legacy_check}",
            f"template discovery N={len(discovery_sids)}",
            "",
            "GENERATION SUMMARY",
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "ROUTE SUMMARY",
            route_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "Interpretation:",
            "  K7/K10/K15 are nested prefixes of the SAME frozen canonical ranking.",
            "  Thus differences isolate the number of shared WHERE positions.",
            "  Main condition uses the current best non-oracle HOW:",
            "      Direction != baseline -> signed candidate-gradient trajectory repair.",
            "  Samples where Direction == baseline are intentionally untouched.",
            "",
            "  Check generation_seen_vs_unseen.csv to separate template-discovery",
            "  samples from samples that never participated in WHERE discovery.",
        ]

        (outdir / "analysis_summary.txt").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "script": "eval_nonoracle_hard_globalK_all440_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "global_template_dir": str(template_dir),
            "global_ranking_file": str(
                template_dir / "global_slot_ranking.csv"
            ),
            "global_ks": ks,
            "nested_prefix_design": True,
            "legacy_k10_check": legacy_check,
            "prior_real_update_dir": str(prior_dir),
            "selector_csv": str(a.selector_csv),
            "selector_method": a.selector_method,
            "update_layers": update_layers,
            "scale": float(a.scale),
            "trigger": "direction_route != baseline_prediction",
            "nonoracle_B": (
                "dot(actual REAL block update, grad(S_direction_route - "
                "S_strongest_clean_nonroute_candidate))"
            ),
            "include_hard_all": bool(a.include_hard_all),
            "include_oracle_reference": bool(a.include_oracle_reference),
            "N_template_discovery_sids": len(discovery_sids),
            "candidate_texts": texts,
            "sequence_score_reduction": reduction,
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
