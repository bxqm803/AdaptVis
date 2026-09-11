#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_global_fixed_l26_top10_trajectory_gating_v1.py

Hypothesis
==========
Maybe the exact causal token position is not very important.  Instead, there may
be a SMALL GLOBAL SET of text slots that are useful for every sample, while the
important sample-specific quantity is WHAT each layer injects into those slots:

    fixed WHERE across samples
        +
    sample-specific positive / negative update polarity.

This script tests exactly that hypothesis.

A. Discover ONE GLOBAL L26 token template
=========================================
For every discovery sample, compute the usual L26 GT-writer mediation:

    M_i,26(p)
      = < h_real_i[26,p] - h_gray_i[26,p],
          d J_GT_i / d h_real_i[26,p] >

Take the sample's positive L26 Top-K (default K=10), then canonicalize token
positions across samples:

    subject span   -> one <SUBJ> slot
    reference span -> one <REF> slot
    visual block   -> collapsed placeholder (visual slots are excluded here)
    fixed prompt text -> exact canonical text slot

This is necessary because raw merged positions such as p=410 and p=360 are not
comparable across images/prompts.

Pool the discovery samples and select ONE GLOBAL Top-K canonical slot list
(default K=10).  The same canonical slots are then used for EVERY evaluation
sample.  There is NO per-sample causal-token selection during evaluation.

The default global ranking favors universality across relations:
    1) high minimum selection rate across LEFT/RIGHT/ABOVE/BELOW
    2) high mean selection rate across relations
    3) high mean positive mediation when selected

B. Sample-specific injection sign
==================================
For every evaluation sample, resolve the SAME global canonical slots back to
concrete token positions p_i.

For every selected position and every update layer (default L20..L26):

    a_REAL[L,p] = h_REAL[L,p] - h_REAL[L-1,p]

Use the same oracle sequence-margin sign from
eval_real_causal_token_update_gating_v1.py:

    B_i[L,p]
      = < a_REAL_i[L,p],
          d(S_GT - S_best_nonGT) / d h_REAL_i[L,p] >

Main intervention:

    global_fixed_signed:
        B > 0 -> +alpha * a_REAL
        B < 0 -> -alpha * a_REAL

Thus ALL samples use the SAME WHERE template, while each sample/layer/token gets
its own positive/negative injection decision.

Controls
========
global_fixed_all_amplify
    Same global positions, blindly amplify every update.

global_fixed_positive
    Only amplify positive updates.

global_fixed_negative_cancel
    Only cancel negative updates.

global_fixed_l26_only_signed
    Same global positions, but edit only L26.

sample_l26_signed_reference
    Per-sample L26 Top-K WHERE (oracle) + signed trajectory.  This measures the
    loss caused specifically by forcing WHERE to be shared across samples.

Oracle status
=============
This remains a mechanism diagnostic:
  * discovery of the global slots uses GT writers;
  * injection polarity uses the GT sequence margin.

However, in heldout mode the GLOBAL WHERE template is discovered only on the
30% calibration split and frozen for the 70% evaluation split.  No evaluation
sample chooses its own causal token positions.

Recommended quick diagnostic (same-sample discovery, N=80)
===========================================================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_global_fixed_l26_top10_trajectory_gating_v1.py \
  --model qwen-3b \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --writer-npz output/qwen3b_coco_dynamic_L20_26_K36_all440/learned_writers.npz \
  --selector-layer 26 \
  --per-sample-discovery-k 10 \
  --global-k 10 \
  --update-layers 20-26 \
  --scale 0.5 \
  --split-mode all_data \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_global_fixed_l26_top10_traj_n80_v1 \
  --overwrite

Cleaner heldout test
====================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_global_fixed_l26_top10_trajectory_gating_v1.py \
  --model qwen-3b \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --writer-npz output/qwen3b_coco_dynamic_L20_26_K36_all440/learned_writers.npz \
  --selector-layer 26 \
  --per-sample-discovery-k 10 \
  --global-k 10 \
  --update-layers 20-26 \
  --scale 0.5 \
  --split-mode heldout \
  --discovery-frac 0.30 \
  --eval-max-samples 0 \
  --output-dir output/qwen3b_global_fixed_l26_top10_traj_heldout_v1 \
  --overwrite

If you want direct comparability to the earlier 8..26 result, simply change:
    --update-layers 8-26

Main outputs
============
global_slot_ranking.csv
global_topk_slots.csv
    The ONE frozen WHERE template shared by all evaluation samples.

discovery_sample_l26_topk.csv
    Per-sample L26 Top-K used only to discover the global template.

resolved_global_slots_per_sample.csv
    How each frozen canonical slot maps to a concrete position in each sample.

per_real_update_decision_score.csv
    Sample-specific B[L,p] on the fixed global slots.

generation_summary.csv
generation_by_relation.csv
generation_per_sample.csv

global_slot_coverage.csv
    Whether the global slots actually resolve in every sample.

analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn
import eval_real_causal_token_update_gating_v1 as gate
import eval_l26_horizontal_top7_real_update_trajectory_v1 as l26


REL = ("left", "right", "above", "below")
DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
EPS = 1e-12


# =============================================================================
# CLI
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

    p.add_argument("--prior-real-update-dir", required=True)
    p.add_argument("--writer-npz", default="")
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--target-layers", default="32,34,35")

    p.add_argument("--selector-layer", type=int, default=26)
    p.add_argument(
        "--per-sample-discovery-k",
        type=int,
        default=10,
        help="Positive L26 Top-K used to estimate stable global canonical slots.",
    )
    p.add_argument(
        "--global-k",
        type=int,
        default=10,
        help="Number of frozen canonical token slots used for EVERY eval sample.",
    )

    p.add_argument(
        "--split-mode",
        default="all_data",
        choices=["all_data", "heldout"],
        help=(
            "all_data = discover the global template and evaluate on the same cohort "
            "(mechanism diagnostic). heldout = discover on calibration split and "
            "freeze the template for heldout evaluation."
        ),
    )
    p.add_argument("--discovery-frac", type=float, default=0.30)

    p.add_argument(
        "--update-layers",
        default="20-26",
        help="Actual REAL updates to regulate on the shared token trajectories.",
    )
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--decision-threshold", type=float, default=0.0)
    p.add_argument(
        "--conditions",
        default=(
            "global_fixed_signed,"
            "global_fixed_all_amplify,"
            "global_fixed_positive,"
            "global_fixed_negative_cancel,"
            "global_fixed_l26_only_signed,"
            "sample_l26_signed_reference"
        ),
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)

    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help=(
            "0 = all samples after split. In all_data mode, >0 caps the shared "
            "discovery/evaluation cohort. In heldout mode, >0 caps eval only."
        ),
    )
    p.add_argument(
        "--discovery-max-samples",
        type=int,
        default=0,
        help="Optional cap for discovery samples; 0 = all discovery samples.",
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


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def safe_mean(xs):
    vals = pd.to_numeric(pd.Series(list(xs)), errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def safe_median(xs):
    vals = pd.to_numeric(pd.Series(list(xs)), errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else float("nan")


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ensure_output_dir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def stratified_cap_meta(meta, n, seed):
    if n <= 0 or len(meta) <= n:
        return list(meta)

    df = pd.DataFrame(meta)
    if "baseline_correct" not in df.columns:
        # fallback: relation-only cap
        rng = random.Random(seed)
        ids = list(range(len(meta)))
        rng.shuffle(ids)
        return [meta[i] for i in ids[:n]]

    capped = l26.stratified_cap_df(
        df,
        int(n),
        int(seed),
    )
    wanted = set(capped["sid"].astype(int))
    return [m for m in meta if int(m["sid"]) in wanted]


# =============================================================================
# Canonical text slots shared across samples
# =============================================================================

def canonical_mapping(
    *,
    model,
    processor,
    batch,
    ids,
    subject,
    reference,
):
    """
    Collapse variable-length pieces into a comparable canonical prompt sequence:
      all image patches -> one <VISUAL_BLOCK> slot
      subject subtokens -> one <SUBJ> slot
      reference subtokens -> one <REF> slot
      every other text token -> its canonical prompt slot

    Returns:
      mapping[position] -> metadata
      canonical sequence
      signature hash
    """
    tokenizer = processor.tokenizer
    toks = [str(t) for t in tokenizer.convert_ids_to_tokens(ids)]

    # Object spans.
    try:
        sspan, rspan = base.locate_object_spans(
            tokenizer,
            ids,
            subject,
            reference,
        )
        spos = set(range(int(sspan[0]), int(sspan[1]) + 1))
        rpos = set(range(int(rspan[0]), int(rspan[1]) + 1))
    except Exception:
        # fall back to dyn's category labels
        cats, _ = dyn.build_categories(
            model,
            processor,
            batch,
            ids,
            subject,
            reference,
        )
        spos = {i for i, c in enumerate(cats) if c == "subject"}
        rpos = {i for i, c in enumerate(cats) if c == "reference"}

    # Actual merged visual positions.
    try:
        vpos = set(
            map(
                int,
                base.resolve_visual_indices(
                    model,
                    processor,
                    batch,
                    ids,
                ),
            )
        )
    except Exception:
        vpos = {
            i
            for i, tok in enumerate(toks)
            if "image_pad" in tok or "video_pad" in tok
        }

    mapping = {}
    canonical_seq = []
    c = 0
    i = 0
    n = len(ids)

    while i < n:
        if i in vpos:
            block = []
            while i < n and i in vpos:
                block.append(i)
                i += 1

            label = f"C{c:03d}:<VISUAL_BLOCK>"
            canonical_seq.append("<VISUAL_BLOCK>")
            for p in block:
                mapping[p] = {
                    "canonical_slot": label,
                    "canonical_kind": "visual",
                    "semantic_slot": "<VISUAL_BLOCK>",
                    "token": toks[p],
                }
            c += 1
            continue

        if i in spos:
            block = []
            while i < n and i in spos:
                block.append(i)
                i += 1

            label = f"C{c:03d}:<SUBJ>"
            canonical_seq.append("<SUBJ>")
            for p in block:
                mapping[p] = {
                    "canonical_slot": label,
                    "canonical_kind": "subject",
                    "semantic_slot": "<SUBJ>",
                    "token": toks[p],
                }
            c += 1
            continue

        if i in rpos:
            block = []
            while i < n and i in rpos:
                block.append(i)
                i += 1

            label = f"C{c:03d}:<REF>"
            canonical_seq.append("<REF>")
            for p in block:
                mapping[p] = {
                    "canonical_slot": label,
                    "canonical_kind": "reference",
                    "semantic_slot": "<REF>",
                    "token": toks[p],
                }
            c += 1
            continue

        tok = str(toks[i]).replace("\n", "\\n")
        label = f"C{c:03d}:{tok}"
        canonical_seq.append(tok)
        mapping[i] = {
            "canonical_slot": label,
            "canonical_kind": "fixed_text",
            "semantic_slot": tok,
            "token": tok,
        }
        c += 1
        i += 1

    sig = "\t".join(canonical_seq)
    sig_hash = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
    return mapping, canonical_seq, sig_hash


def resolve_slot(mapping, slot: str):
    """
    Resolve one frozen canonical slot into one concrete token position.
    For multi-subtoken <SUBJ>/<REF>, use the center token so one global slot gives
    exactly one concrete trajectory.
    """
    ps = sorted(
        int(p)
        for p, info in mapping.items()
        if str(info["canonical_slot"]) == str(slot)
    )
    if not ps:
        return None
    return ps[len(ps) // 2]


# =============================================================================
# Discover global fixed slots
# =============================================================================

def discover_global_slots(
    *,
    a,
    discovery_meta,
    rec_by_sid,
    model,
    processor,
    decoder_layers,
    writers,
    targets,
    device,
    outdir,
    error_path,
):
    per_sample_rows = []
    signature_rows = []

    for m in tqdm(
        discovery_meta,
        desc=f"DISCOVER global L{a.selector_layer} Top{a.per_sample_discovery_k}",
    ):
        sid = int(m["sid"])
        gt = m["gt"]
        real = gray = None

        try:
            writers_r = {T: writers[T][gt] for T in targets}

            real = base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")
            gray = dyn.make_gray_image(real, a.gray_value)

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

            selected, _ = l26.select_horizontal_positions(
                sid=sid,
                gt=gt,
                selector_layer=int(a.selector_layer),
                top_k=int(a.per_sample_discovery_k),
                categories_allowed=set(),
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                rb=rb,
                gb=gb,
                subject=m["subject"],
                reference=m["reference"],
                writers_r=writers_r,
                targets=targets,
            )

            ids = rb["input_ids"][0].detach().cpu().tolist()
            mapping, canon_seq, sig = canonical_mapping(
                model=model,
                processor=processor,
                batch=rb,
                ids=ids,
                subject=m["subject"],
                reference=m["reference"],
            )

            signature_rows.append(
                {
                    "sid": sid,
                    "gt": gt,
                    "signature_hash": sig,
                    "n_canonical_slots": len(canon_seq),
                }
            )

            # Multiple object subtokens may collapse to one canonical slot.
            # Keep only the strongest L26 mediation for that slot in this sample.
            sample_slot_best = {}

            for r in selected:
                pos = int(r["position"])
                info = mapping.get(pos)
                if info is None:
                    continue
                if info["canonical_kind"] == "visual":
                    continue

                slot = str(info["canonical_slot"])
                rr = dict(r)
                rr.update(
                    {
                        "canonical_slot": slot,
                        "canonical_kind": str(info["canonical_kind"]),
                        "semantic_slot": str(info["semantic_slot"]),
                        "signature_hash": sig,
                    }
                )

                cur = sample_slot_best.get(slot)
                if cur is None or float(rr["mediation"]) > float(cur["mediation"]):
                    sample_slot_best[slot] = rr

            ranked_unique = sorted(
                sample_slot_best.values(),
                key=lambda z: float(z["mediation"]),
                reverse=True,
            )

            for urank, rr in enumerate(ranked_unique, 1):
                rr["canonical_rank_within_sample"] = int(urank)
                rr["rank_weight"] = 1.0 / float(urank)
                per_sample_rows.append(rr)

        except Exception as exc:
            append_jsonl(
                error_path,
                {
                    "sid": sid,
                    "phase": "global_slot_discovery",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-80:],
                },
            )
            tqdm.write(
                f"[DISCOVERY ERROR] sid={sid}: {type(exc).__name__}: {exc}"
            )

        finally:
            if real is not None:
                with contextlib.suppress(Exception):
                    real.close()
            if gray is not None:
                with contextlib.suppress(Exception):
                    gray.close()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    per_sample = pd.DataFrame(per_sample_rows)
    signatures = pd.DataFrame(signature_rows)

    per_sample.to_csv(
        outdir / "discovery_sample_l26_topk.csv",
        index=False,
    )
    signatures.to_csv(
        outdir / "discovery_canonical_signatures.csv",
        index=False,
    )

    if not len(per_sample):
        raise RuntimeError("Global slot discovery produced no rows")

    # One presence per sample/slot.
    one = (
        per_sample.sort_values(
            ["sid", "canonical_slot", "mediation"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["sid", "canonical_slot"], keep="first")
        .copy()
    )

    n_by_rel = {
        rel: int(sum(str(m["gt"]) == rel for m in discovery_meta))
        for rel in REL
    }

    rows = []
    for slot, g in one.groupby("canonical_slot"):
        base_row = {
            "canonical_slot": str(slot),
            "canonical_kind": str(g.iloc[0]["canonical_kind"]),
            "semantic_slot": str(g.iloc[0]["semantic_slot"]),
            "N_samples_selected": int(g["sid"].nunique()),
            "overall_selection_rate": (
                int(g["sid"].nunique()) / max(len(discovery_meta), 1)
            ),
            "mean_positive_mediation_when_selected": safe_mean(g["mediation"]),
            "median_positive_mediation_when_selected": safe_median(g["mediation"]),
            "mean_rank_when_selected": safe_mean(
                g["canonical_rank_within_sample"]
            ),
            "mean_reciprocal_rank_when_selected": safe_mean(g["rank_weight"]),
        }

        rates = []
        for rel in REL:
            denom = max(n_by_rel.get(rel, 0), 1)
            nrel = int(g[g["gt"] == rel]["sid"].nunique())
            rate = nrel / denom
            base_row[f"selection_rate_{rel}"] = rate
            rates.append(rate)

        base_row["min_relation_selection_rate"] = float(min(rates))
        base_row["mean_relation_selection_rate"] = float(np.mean(rates))
        base_row["max_relation_selection_rate"] = float(max(rates))
        base_row["relation_rate_range"] = float(max(rates) - min(rates))

        rows.append(base_row)

    ranking = pd.DataFrame(rows)

    # "All samples share the same causal token slots" -> reward universality first.
    ranking = ranking.sort_values(
        [
            "min_relation_selection_rate",
            "mean_relation_selection_rate",
            "overall_selection_rate",
            "mean_reciprocal_rank_when_selected",
            "mean_positive_mediation_when_selected",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    ranking["global_rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected_global_topk"] = (
        ranking["global_rank"] <= int(a.global_k)
    )

    global_top = ranking.head(int(a.global_k)).copy()

    if len(global_top) < int(a.global_k):
        raise RuntimeError(
            f"Only {len(global_top)} canonical slots available, "
            f"cannot build global K={a.global_k}"
        )

    ranking.to_csv(
        outdir / "global_slot_ranking.csv",
        index=False,
    )
    global_top.to_csv(
        outdir / "global_topk_slots.csv",
        index=False,
    )

    return global_top, per_sample, signatures


# =============================================================================
# Fixed global WHERE -> sample-specific HOW
# =============================================================================

def make_specs_from_global_slots(
    *,
    global_top,
    mapping,
    ids,
    cats,
    toks,
    selector_layer,
):
    specs = []
    resolved_rows = []

    for r in global_top.itertuples():
        slot = str(r.canonical_slot)
        p = resolve_slot(mapping, slot)

        if p is None:
            resolved_rows.append(
                {
                    "global_rank": int(r.global_rank),
                    "canonical_slot": slot,
                    "canonical_kind": str(r.canonical_kind),
                    "semantic_slot": str(r.semantic_slot),
                    "resolved": False,
                    "position": -1,
                    "token": "",
                    "category": "",
                    "broad_category": "",
                }
            )
            continue

        cat = str(cats[p])
        broad = str(dyn.broad_category(cat))
        tok = str(toks[p]).replace("\n", "\\n")

        # Defensive: keep the experiment text-only.
        if broad in ("visual", "last"):
            resolved_rows.append(
                {
                    "global_rank": int(r.global_rank),
                    "canonical_slot": slot,
                    "canonical_kind": str(r.canonical_kind),
                    "semantic_slot": str(r.semantic_slot),
                    "resolved": False,
                    "position": int(p),
                    "token": tok,
                    "category": cat,
                    "broad_category": broad,
                    "reason": "resolved_to_excluded_category",
                }
            )
            continue

        specs.append(
            {
                "real_position": int(p),
                "max_target_layer": int(selector_layer),
                "min_target_layer": int(selector_layer),
                "best_rank": int(r.global_rank),
                "token": tok,
                "category": cat,
                "broad_category": broad,
                "canonical_slot": slot,
                "global_rank": int(r.global_rank),
            }
        )

        resolved_rows.append(
            {
                "global_rank": int(r.global_rank),
                "canonical_slot": slot,
                "canonical_kind": str(r.canonical_kind),
                "semantic_slot": str(r.semantic_slot),
                "resolved": True,
                "position": int(p),
                "token": tok,
                "category": cat,
                "broad_category": broad,
            }
        )

    return specs, resolved_rows


def condition_patch(scored, condition, scale, threshold, selector_layer):
    if condition == "global_fixed_signed":
        return gate.build_real_update_patch_map(
            scored, "real_signed", scale, threshold
        )
    if condition == "global_fixed_positive":
        return gate.build_real_update_patch_map(
            scored, "real_positive", scale, threshold
        )
    if condition == "global_fixed_negative_cancel":
        return gate.build_real_update_patch_map(
            scored, "real_negative_cancel", scale, threshold
        )
    if condition == "global_fixed_all_amplify":
        return gate.build_real_update_patch_map(
            scored, "real_all_amplify", scale, threshold
        )
    if condition == "global_fixed_l26_only_signed":
        rows = [
            x
            for x in scored
            if int(x["update_layer"]) == int(selector_layer)
        ]
        return gate.build_real_update_patch_map(
            rows, "real_signed", scale, threshold
        )
    raise ValueError(condition)


def summarize_coverage(resolved_df, global_k):
    if not len(resolved_df):
        return pd.DataFrame()

    per_sample = (
        resolved_df.groupby(["sid", "gt"], as_index=False)
        .agg(
            resolved_N=("resolved", "sum"),
            attempted_N=("canonical_slot", "size"),
        )
    )
    per_sample["full_global_k"] = (
        per_sample["resolved_N"] >= int(global_k)
    )

    rows = [
        {
            "relation": "ALL",
            "N": int(len(per_sample)),
            "mean_resolved_N": float(per_sample["resolved_N"].mean()),
            "full_global_k_fraction": float(
                per_sample["full_global_k"].mean()
            ),
        }
    ]

    for gt, g in per_sample.groupby("gt"):
        rows.append(
            {
                "relation": DISPLAY.get(gt, gt),
                "N": int(len(g)),
                "mean_resolved_N": float(g["resolved_N"].mean()),
                "full_global_k_fraction": float(
                    g["full_global_k"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    update_layers = parse_ints(a.update_layers)
    targets = parse_ints(a.target_layers)
    conditions = parse_list(a.conditions)

    allowed_conditions = {
        "global_fixed_signed",
        "global_fixed_all_amplify",
        "global_fixed_positive",
        "global_fixed_negative_cancel",
        "global_fixed_l26_only_signed",
        "sample_l26_signed_reference",
    }
    unknown = set(conditions) - allowed_conditions
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    if min(update_layers) < 1:
        raise ValueError("update layers must be >=1")
    if max(update_layers) > int(a.selector_layer):
        raise ValueError(
            "update layers cannot exceed selector layer in this experiment"
        )

    outdir = Path(a.output_dir)
    ensure_output_dir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    # -------------------------------------------------------------------------
    # Reuse exact baseline/competitor cohort from prior REAL-update run.
    # -------------------------------------------------------------------------
    (
        cohort,
        _old_sel,
        prior_metadata,
        _prior_nonbase,
    ) = l26.load_prior_run(Path(a.prior_real_update_dir))

    two, all_eval_meta, rec_by_sid, prompts, records = l26.load_dataset(
        a,
        cohort,
    )

    # Split global-template discovery from evaluation if requested.
    if a.split_mode == "heldout":
        discovery_meta, eval_meta = traj.stratified_split(
            list(all_eval_meta),
            float(a.discovery_frac),
            int(a.seed),
        )
    else:
        eval_meta = list(all_eval_meta)
        if int(a.eval_max_samples) > 0:
            eval_meta = stratified_cap_meta(
                eval_meta,
                int(a.eval_max_samples),
                int(a.seed) + 101,
            )
        discovery_meta = list(eval_meta)

    if a.split_mode == "heldout" and int(a.eval_max_samples) > 0:
        eval_meta = stratified_cap_meta(
            eval_meta,
            int(a.eval_max_samples),
            int(a.seed) + 102,
        )

    if int(a.discovery_max_samples) > 0:
        discovery_meta = stratified_cap_meta(
            discovery_meta,
            int(a.discovery_max_samples),
            int(a.seed) + 103,
        )

    model = processor = None

    try:
        (
            model,
            processor,
            decoder_layers,
            decoder_path,
            spec,
        ) = l26.load_model(a, two)

        device = torch.device(a.device)

        # Writer.
        if a.writer_npz:
            writers = l26.load_writers_npz(
                Path(a.writer_npz),
                targets,
            )
            writer_source = str(Path(a.writer_npz))
        else:
            # Build metadata required by the existing calibration helper.
            all_meta_for_writer = []
            for rec in records:
                sid = int(rec.sid)
                if sid not in prompts:
                    continue
                p = prompts[sid]
                gt = l26.canon_rel(
                    traj.normalize_relation(base, p["answer_raw"])
                )
                if gt not in REL:
                    continue
                all_meta_for_writer.append(
                    {
                        "sid": sid,
                        "gt": gt,
                        "subject": str(p["subject"]),
                        "reference": str(p["reference"]),
                        "question_text": str(p["question_text"]),
                    }
                )

            writers, _writer_train = l26.calibrate_writers(
                a=a,
                all_meta=all_meta_for_writer,
                rec_by_sid=rec_by_sid,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                targets=targets,
                device=device,
                outdir=outdir,
            )
            writer_source = "recalibrated_in_this_run"

        # Same candidate answer strings / reduction as prior run.
        texts = prior_metadata.get(
            "candidate_texts",
            {
                "left": "left",
                "right": "right",
                "above": "above",
                "below": "below",
            },
        )
        texts = {
            l26.canon_rel(k): str(v)
            for k, v in texts.items()
        }
        for r in REL:
            if r not in texts:
                raise RuntimeError(
                    f"candidate_texts missing {r}: {texts}"
                )

        candidate_ids = gate.encode_candidate_ids(processor, texts)
        reduction = str(
            prior_metadata.get(
                "sequence_score_reduction",
                "mean",
            )
        )
        if reduction not in ("mean", "sum"):
            reduction = "mean"

        print("=" * 190)
        print("GLOBAL FIXED L26 TOP10 TOKEN TEMPLATE -> SAMPLE-SPECIFIC SIGNED TRAJECTORY")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"split_mode={a.split_mode}")
        print(f"N discovery={len(discovery_meta)}")
        print(f"N eval={len(eval_meta)}")
        print(
            f"selector=L{a.selector_layer} "
            f"per_sample_discovery_k={a.per_sample_discovery_k} "
            f"global_k={a.global_k}"
        )
        print(f"update_layers={update_layers}")
        print(f"scale={a.scale}")
        print(f"writer_source={writer_source}")
        print()

        # =====================================================================
        # 1) Discover ONE global relation-invariant WHERE template.
        # =====================================================================
        global_top, discovery_rows, signatures = discover_global_slots(
            a=a,
            discovery_meta=discovery_meta,
            rec_by_sid=rec_by_sid,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            writers=writers,
            targets=targets,
            device=device,
            outdir=outdir,
            error_path=error_path,
        )

        print("\nGLOBAL TOP SLOTS")
        print(
            global_top[
                [
                    "global_rank",
                    "canonical_slot",
                    "semantic_slot",
                    "canonical_kind",
                    "min_relation_selection_rate",
                    "mean_relation_selection_rate",
                    "overall_selection_rate",
                    "mean_positive_mediation_when_selected",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print()

        # =====================================================================
        # 2) Evaluate frozen global WHERE; only HOW is sample-specific.
        # =====================================================================
        resolved_all = []
        update_rows_all = []
        generation_rows = []

        for m in tqdm(
            eval_meta,
            desc=f"EVAL global fixed K{a.global_k}",
        ):
            sid = int(m["sid"])
            gt = m["gt"]
            real = gray = None

            generation_rows.append(
                {
                    "sid": sid,
                    "gt": gt,
                    "condition": "baseline",
                    "scale": 0.0,
                    "prediction": m["baseline_prediction"],
                    "correct": bool(m["baseline_correct"]),
                    "n_patched_updates": 0,
                    "n_positive_updates": 0,
                    "n_negative_updates": 0,
                    "resolved_global_positions": 0,
                    "text": "",
                }
            )

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real, a.gray_value)

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
                cats, toks = dyn.build_categories(
                    model,
                    processor,
                    rb,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                mapping, canon_seq, sig = canonical_mapping(
                    model=model,
                    processor=processor,
                    batch=rb,
                    ids=ids,
                    subject=m["subject"],
                    reference=m["reference"],
                )

                # SAME global slots for every sample.
                specs, resolved = make_specs_from_global_slots(
                    global_top=global_top,
                    mapping=mapping,
                    ids=ids,
                    cats=cats,
                    toks=toks,
                    selector_layer=int(a.selector_layer),
                )

                for rr in resolved:
                    resolved_all.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "signature_hash": sig,
                            **rr,
                        }
                    )

                if not specs:
                    raise RuntimeError(
                        "None of the global slots resolved in this sample"
                    )

                # Fixed global positions -> actual REAL trajectories.
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
                        "No trajectory updates for resolved global slots"
                    )

                grad_layers = sorted(
                    set(int(e["update_layer"]) for e in entries)
                )

                gt_score, grad_gt = gate.sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    answer_ids=candidate_ids[gt],
                    reduction=reduction,
                    grad_layers=grad_layers,
                )

                comp = m["competitor"]
                comp_score, grad_comp = gate.sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    answer_ids=candidate_ids[comp],
                    reduction=reduction,
                    grad_layers=grad_layers,
                )

                replay_margin = float(gt_score - comp_score)
                if abs(
                    replay_margin - float(m["prior_sequence_margin"])
                ) > 1e-2:
                    raise RuntimeError(
                        "Sequence margin replay mismatch: "
                        f"prior={m['prior_sequence_margin']:.6f}, "
                        f"replay={replay_margin:.6f}"
                    )

                scored = gate.attach_decision_scores(
                    entries,
                    grad_gt,
                    grad_comp,
                )

                if not scored:
                    raise RuntimeError("No signed REAL updates")

                spec_by_pos = {
                    int(s["real_position"]): s
                    for s in specs
                }

                for e in scored:
                    s = spec_by_pos[int(e["real_position"])]
                    export = {
                        k: v
                        for k, v in e.items()
                        if k not in (
                            "_real_update",
                            "_noimage_update",
                            "_rn_update",
                            "_decision_grad",
                        )
                    }
                    export["canonical_slot"] = s["canonical_slot"]
                    export["global_rank"] = int(s["global_rank"])
                    export["competitor"] = comp
                    export["replayed_sequence_margin"] = replay_margin
                    update_rows_all.append(export)

                # -------------------------------------------------------------
                # Frozen-global conditions.
                # -------------------------------------------------------------
                for cond in conditions:
                    if cond == "sample_l26_signed_reference":
                        continue

                    pmap, counts = condition_patch(
                        scored,
                        cond,
                        float(a.scale),
                        float(a.decision_threshold),
                        int(a.selector_layer),
                    )

                    pred, text = gate.generate_with_patch(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        patch_map=pmap,
                        max_new_tokens=a.max_new_tokens,
                    )

                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "condition": cond,
                            "scale": float(a.scale),
                            "prediction": pred,
                            "correct": pred == gt,
                            "n_patched_updates": int(counts["patched"]),
                            "n_positive_updates": int(counts["positive"]),
                            "n_negative_updates": int(counts["negative"]),
                            "resolved_global_positions": len(specs),
                            "text": text,
                        }
                    )

                # -------------------------------------------------------------
                # Reference: sample-specific L26 Top-K WHERE, same HOW.
                # -------------------------------------------------------------
                if "sample_l26_signed_reference" in conditions:
                    writers_r = {
                        T: writers[T][gt]
                        for T in targets
                    }

                    selected_ref, _ = l26.select_horizontal_positions(
                        sid=sid,
                        gt=gt,
                        selector_layer=int(a.selector_layer),
                        top_k=int(a.global_k),
                        categories_allowed=set(),
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        rb=rb,
                        gb=gb,
                        subject=m["subject"],
                        reference=m["reference"],
                        writers_r=writers_r,
                        targets=targets,
                    )

                    ref_specs = l26.selected_to_specs(
                        selected_ref,
                        int(a.selector_layer),
                    )

                    ref_entries = gate.build_real_updates(
                        sid=sid,
                        gt=gt,
                        baseline_correct=bool(m["baseline_correct"]),
                        specs=ref_specs,
                        r2n={},
                        real_states=real_states,
                        no_states={},
                        update_layers=update_layers,
                        exclude_target_layer=False,
                    )

                    ref_scored = gate.attach_decision_scores(
                        ref_entries,
                        grad_gt,
                        grad_comp,
                    )

                    ref_pmap, ref_counts = gate.build_real_update_patch_map(
                        ref_scored,
                        "real_signed",
                        float(a.scale),
                        float(a.decision_threshold),
                    )

                    pred, text = gate.generate_with_patch(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        batch=rb,
                        patch_map=ref_pmap,
                        max_new_tokens=a.max_new_tokens,
                    )

                    generation_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "condition": "sample_l26_signed_reference",
                            "scale": float(a.scale),
                            "prediction": pred,
                            "correct": pred == gt,
                            "n_patched_updates": int(ref_counts["patched"]),
                            "n_positive_updates": int(ref_counts["positive"]),
                            "n_negative_updates": int(ref_counts["negative"]),
                            "resolved_global_positions": len(ref_specs),
                            "text": text,
                        }
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "phase": "eval",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-80:],
                    },
                )
                tqdm.write(
                    f"[EVAL ERROR] sid={sid}: {type(exc).__name__}: {exc}"
                )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =====================================================================
        # Save / summarize
        # =====================================================================
        resolved_df = pd.DataFrame(resolved_all)
        update_df = pd.DataFrame(update_rows_all)
        gen_df = pd.DataFrame(generation_rows)

        resolved_df.to_csv(
            outdir / "resolved_global_slots_per_sample.csv",
            index=False,
        )
        update_df.to_csv(
            outdir / "per_real_update_decision_score.csv",
            index=False,
        )
        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )

        coverage = summarize_coverage(
            resolved_df,
            int(a.global_k),
        )
        coverage.to_csv(
            outdir / "global_slot_coverage.csv",
            index=False,
        )

        if len(update_df):
            layer_summary = gate.summarize_layers(update_df)
            layer_summary.to_csv(
                outdir / "layer_real_update_summary.csv",
                index=False,
            )

            layer_rel = gate.summarize_layers(
                update_df,
                extra_keys=["gt"],
            )
            layer_rel.to_csv(
                outdir / "layer_real_update_by_relation.csv",
                index=False,
            )

            # Per fixed canonical slot: how often is its realized update + vs -?
            slot_sign = (
                update_df.groupby(
                    ["canonical_slot", "global_rank", "update_layer"],
                    as_index=False,
                )
                .agg(
                    N=("sid", "size"),
                    mean_B=("real_update_decision_score", "mean"),
                    mean_abs_B=("real_update_decision_score", lambda s: np.mean(np.abs(s))),
                    positive_fraction=(
                        "real_update_decision_score",
                        lambda s: np.mean(np.asarray(s, float) > 0),
                    ),
                    negative_fraction=(
                        "real_update_decision_score",
                        lambda s: np.mean(np.asarray(s, float) < 0),
                    ),
                )
                .sort_values(["global_rank", "update_layer"])
            )
            slot_sign.to_csv(
                outdir / "global_slot_layer_sign_summary.csv",
                index=False,
            )
        else:
            layer_summary = pd.DataFrame()
            slot_sign = pd.DataFrame()

        gen_summary = gate.summarize_generation(gen_df)
        gen_rel = gate.summarize_generation_by_relation(gen_df)

        gen_summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )
        gen_rel.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )

        # How different is fixed-global WHERE from per-sample oracle L26 WHERE?
        # Compare concrete positions only for reporting.
        where_overlap_rows = []
        if "sample_l26_signed_reference" in conditions:
            # We did not save ref positions above; reconstruct from discovery rows only
            # when all_data and sample belongs to discovery.  Main causal comparison is
            # the generation reference, so this overlap is optional.
            pass

        # Main rows.
        fixed_row = gen_summary[
            gen_summary["condition"] == "global_fixed_signed"
        ] if len(gen_summary) else pd.DataFrame()

        ref_row = gen_summary[
            gen_summary["condition"] == "sample_l26_signed_reference"
        ] if len(gen_summary) else pd.DataFrame()

        report = [
            "=" * 190,
            "GLOBAL FIXED L26 TOKEN TEMPLATE -> SAMPLE-SPECIFIC SIGNED TRAJECTORY",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"split_mode={a.split_mode}",
            f"N discovery={len(discovery_meta)}",
            f"N eval requested={len(eval_meta)}",
            f"selector layer=L{a.selector_layer}",
            f"per-sample discovery K={a.per_sample_discovery_k}",
            f"GLOBAL shared K={a.global_k}",
            f"update layers={update_layers}",
            f"scale={a.scale}",
            f"writer source={writer_source}",
            "",
            "GLOBAL SHARED WHERE TEMPLATE",
            "-" * 190,
            global_top[
                [
                    "global_rank",
                    "canonical_slot",
                    "semantic_slot",
                    "canonical_kind",
                    "min_relation_selection_rate",
                    "mean_relation_selection_rate",
                    "overall_selection_rate",
                    "mean_positive_mediation_when_selected",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "GLOBAL SLOT COVERAGE ON EVAL",
            "-" * 190,
            (
                coverage.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(coverage)
                else "EMPTY"
            ),
            "",
            "GENERATION",
            "-" * 190,
            (
                gen_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(gen_summary)
                else "EMPTY"
            ),
            "",
            "PRIMARY COMPARISON",
            "-" * 190,
            (
                "global_fixed_signed:\n"
                + fixed_row.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(fixed_row)
                else "global_fixed_signed missing"
            ),
            "",
            (
                "sample_l26_signed_reference:\n"
                + ref_row.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(ref_row)
                else "sample_l26_signed_reference missing"
            ),
            "",
            "Interpretation:",
            "  If global_fixed_signed is close to sample_l26_signed_reference,",
            "  exact per-sample token localization contributes little. A shared token",
            "  scaffold is enough; sample-specific behavior is mainly in the signed",
            "  information injected along those trajectories.",
            "",
            "  If global_fixed_all_amplify is weak while global_fixed_signed is strong,",
            "  that is especially strong evidence that HOW (polarity of injected update)",
            "  matters much more than WHERE.",
            "",
            "  If heldout mode also works, the fixed WHERE template is not an",
            "  evaluation-set artifact.",
            "",
            "Oracle caveat:",
            "  The global template is discovered using GT-writer mediation on the",
            "  discovery split, and HOW still uses GT sequence-margin gradients.",
        ]

        report_text = "\n".join(report) + "\n"
        print(report_text)
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        metadata = {
            "script": "eval_global_fixed_l26_top10_trajectory_gating_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "prior_real_update_dir": str(a.prior_real_update_dir),
            "selector_layer": int(a.selector_layer),
            "per_sample_discovery_k": int(a.per_sample_discovery_k),
            "global_k": int(a.global_k),
            "split_mode": a.split_mode,
            "discovery_frac": float(a.discovery_frac),
            "N_discovery": len(discovery_meta),
            "N_eval_requested": len(eval_meta),
            "update_layers": update_layers,
            "target_layers": targets,
            "scale": float(a.scale),
            "decision_threshold": float(a.decision_threshold),
            "conditions": conditions,
            "writer_source": writer_source,
            "sequence_score_reduction": reduction,
            "candidate_texts": texts,
            "global_where_definition": (
                "One relation-balanced canonical text-slot Top-K template discovered "
                f"from per-sample positive L{a.selector_layer} writer-mediated Top-K; "
                "frozen and reused for every evaluation sample."
            ),
            "sample_specific_how_definition": (
                "For each fixed slot, sample/layer-specific B = "
                "dot(hR[L,p]-hR[L-1,p], grad[S_GT-S_competitor]); "
                "positive amplify, negative cancel."
            ),
            "oracle_note": (
                "GT writer is used only to discover the global WHERE template; "
                "GT sequence margin is still used for sample-specific HOW."
            ),
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
