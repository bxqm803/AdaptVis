#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_l26_horizontal_top7_real_update_trajectory_v1.py

Question
========
Can we collapse the old cross-layer causal-state selector

    WHERE = (layer L, token position p)

into a single horizontal token-position selector at one late layer (default L26):

    WHERE = token position p selected only at L26

and then treat the earlier layers only as the trajectory of that SAME token position?

Primary experiment
==================
1) Use the SAME late GT writer objective as the old oracle causal ranking, but
   restrict source attribution to selector layer C=26 only.

For each text position p at L26:

    M26(p)
      = < h_real[26,p] - h_gray[26,p],
          d J_GT / d h_real[26,p] >

Take the Top-K POSITIVE text positions at L26:

    P26 = TopK_p M26(p)

This is the unified "horizontal cross-section".

2) Keep those SAME token positions p fixed for all earlier layers.

For each p in P26 and each L in update layers (default 8..26):

    a_REAL[L,p] = h_REAL[L,p] - h_REAL[L-1,p]

3) Use the same sequence-level oracle decision sign as
   eval_real_causal_token_update_gating_v1.py:

    M_seq = score(GT) - score(best non-GT competitor)
    B[L,p] = < a_REAL[L,p], dM_seq / d h_REAL[L,p] >

4) Main intervention:

    l26_trajectory_signed:
        B > 0  -> h[L,p] += alpha * a_REAL[L,p]
        B < 0  -> h[L,p] -= alpha * a_REAL[L,p]

Thus:
    * token positions are selected ONCE at L26;
    * all layers reuse the same token positions;
    * layers only determine positive / negative update polarity.

Controls:
    l26_only_signed
        Same L26 Top-K positions, but modify ONLY L26.
        Tests whether the cross-layer trajectory matters.

    l26_trajectory_positive
        Only amplify positive updates on selected trajectories.

    l26_trajectory_negative_cancel
        Only suppress negative updates on selected trajectories.

    l26_trajectory_all_amplify
        Blindly amplify every update on selected trajectories.

Oracle status
=============
This remains an ORACLE mechanism diagnostic:
    WHERE:
      L26 Top-K uses the GT late writer.
    HOW:
      positive / negative sign uses the GT sequence margin.

The purpose is NOT to claim a non-oracle method yet.
It isolates whether the difficult cross-layer state selection can be reduced to:

    one late horizontal token-position selection
        +
    layerwise trajectory polarity.

Reuse of the previous full-440 run
==================================
--prior-real-update-dir should point to the completed
eval_real_causal_token_update_gating_v1.py run.

The script reuses:
    generation_per_sample.csv
        clean baseline predictions
    sequence_score_summary.csv
        strongest non-GT competitor for each sample
    selected_causal_states.csv
        old cross-layer Top-7, ONLY for overlap diagnostics
    metadata.json
        exact candidate answer strings / sequence score reduction

So it does NOT re-run baseline generation or four-way competitor search.

Writer
======
For exact comparability with the original writer-guided causal ranking, pass
the old learned_writers.npz if available:

    --writer-npz output/qwen3b_coco_dynamic_L20_26_K36_all440/learned_writers.npz

If omitted, the script recalibrates the same centered Real-Gray writers on the
usual 30% stratified calibration split.

Recommended smoke test
======================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_l26_horizontal_top7_real_update_trajectory_v1.py \
  --model qwen-3b \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --writer-npz output/qwen3b_coco_dynamic_L20_26_K36_all440/learned_writers.npz \
  --selector-layer 26 \
  --top-k 7 \
  --update-layers 8-26 \
  --scale 0.5 \
  --conditions l26_trajectory_signed,l26_only_signed \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_l26_horizontal_top7_traj_n80_v1 \
  --overwrite

Full 440
========
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_l26_horizontal_top7_real_update_trajectory_v1.py \
  --model qwen-3b \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --writer-npz output/qwen3b_coco_dynamic_L20_26_K36_all440/learned_writers.npz \
  --selector-layer 26 \
  --top-k 7 \
  --update-layers 8-26 \
  --scale 0.5 \
  --conditions l26_trajectory_signed,l26_only_signed,l26_trajectory_positive,l26_trajectory_negative_cancel,l26_trajectory_all_amplify \
  --eval-max-samples 0 \
  --output-dir output/qwen3b_l26_horizontal_top7_traj_all440_v1 \
  --overwrite

Main outputs
============
l26_selected_positions.csv
    The L26 horizontal Top-K positions for every sample.

old_vs_l26_overlap_per_sample.csv
old_vs_l26_overlap_summary.csv
    Position-only overlap with the OLD cross-layer Top-7, plus exact (L,p)
    overlap.  High position overlap with lower state overlap supports the
    "same token trajectory, different strongest layer" hypothesis.

per_real_update_decision_score.csv
    B[L,p] for every selected L26 position trajectory.

generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
    Main behavioral test.

selection_category_summary.csv
selection_relation_category_summary.csv
    What kinds of tokens occupy the L26 Top-K.

analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn
import eval_real_causal_token_update_gating_v1 as gate


REL = ("left", "right", "above", "below")
DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
EPS = 1e-12


# =============================================================================
# CLI / utilities
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

    p.add_argument(
        "--prior-real-update-dir",
        required=True,
        help=(
            "Completed eval_real_causal_token_update_gating_v1.py output dir. "
            "Used for baseline, competitor, old Top-K overlap and sequence config."
        ),
    )

    p.add_argument(
        "--writer-npz",
        default="",
        help=(
            "Optional old learned_writers.npz. Strongly recommended for exact "
            "comparability. If empty, writers are recalibrated."
        ),
    )
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--target-layers", default="32,34,35")

    p.add_argument(
        "--selector-layer",
        type=int,
        default=26,
        help="Single horizontal layer used to select token positions.",
    )
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument(
        "--causal-categories",
        default="",
        help=(
            "Optional comma-separated broad categories. Empty means all TEXT "
            "categories except visual and last."
        ),
    )

    p.add_argument(
        "--update-layers",
        default="8-26",
        help="Layers whose actual REAL updates are followed on fixed L26 positions.",
    )

    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--decision-threshold", type=float, default=0.0)
    p.add_argument(
        "--conditions",
        default=(
            "l26_trajectory_signed,l26_only_signed,"
            "l26_trajectory_positive,l26_trajectory_negative_cancel,"
            "l26_trajectory_all_amplify"
        ),
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)

    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all samples present in the prior run; >0 = stratified debug cap.",
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
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_csv_set(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    aliases = {
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
    }
    return aliases.get(s, s)


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


def safe_median(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        return v.copy()
    return (v / n).astype(np.float32)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_output_dir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Reuse prior 440 result
# =============================================================================

def load_prior_run(run_dir: Path):
    gen_path = run_dir / "generation_per_sample.csv"
    seq_path = run_dir / "sequence_score_summary.csv"
    old_sel_path = run_dir / "selected_causal_states.csv"
    meta_path = run_dir / "metadata.json"

    for p in (gen_path, seq_path):
        if not p.exists():
            raise FileNotFoundError(p)

    gen = pd.read_csv(gen_path)
    seq = pd.read_csv(seq_path)

    baseline = gen[gen["condition"].astype(str) == "baseline"].copy()
    baseline["sid"] = pd.to_numeric(baseline["sid"], errors="raise").astype(int)
    baseline["gt"] = baseline["gt"].map(canon_rel)
    baseline["prediction"] = baseline["prediction"].map(canon_rel)
    if baseline["correct"].dtype != bool:
        baseline["correct"] = (
            baseline["correct"].astype(str).str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    baseline = baseline.sort_values("sid").drop_duplicates("sid")

    seq["sid"] = pd.to_numeric(seq["sid"], errors="raise").astype(int)
    seq["gt"] = seq["gt"].map(canon_rel)
    seq["competitor"] = seq["competitor"].map(canon_rel)

    cohort = baseline[
        ["sid", "gt", "prediction", "correct"]
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "correct": "baseline_correct",
        }
    ).merge(
        seq[["sid", "competitor", "sequence_margin"]],
        on="sid",
        how="inner",
    )

    old_sel = pd.DataFrame()
    if old_sel_path.exists():
        old_sel = pd.read_csv(old_sel_path)
        if len(old_sel):
            for c in ("sid", "rank", "source_layer", "position"):
                if c in old_sel.columns:
                    old_sel[c] = pd.to_numeric(
                        old_sel[c], errors="raise"
                    ).astype(int)
            if "gt" in old_sel.columns:
                old_sel["gt"] = old_sel["gt"].map(canon_rel)

    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    # Reference old oracle-signed results from exactly the prior run.
    prior_nonbase = gen[gen["condition"].astype(str) != "baseline"].copy()
    prior_nonbase["sid"] = pd.to_numeric(
        prior_nonbase["sid"], errors="raise"
    ).astype(int)
    prior_nonbase["gt"] = prior_nonbase["gt"].map(canon_rel)
    prior_nonbase["prediction"] = prior_nonbase["prediction"].map(canon_rel)
    if prior_nonbase["correct"].dtype != bool:
        prior_nonbase["correct"] = (
            prior_nonbase["correct"].astype(str).str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    return cohort, old_sel, metadata, prior_nonbase


def stratified_cap_df(df: pd.DataFrame, n: int, seed: int):
    if n <= 0 or len(df) <= n:
        return df.copy()

    rng = random.Random(seed)
    picked = []

    groups = list(df.groupby(["gt", "baseline_correct"], dropna=False))
    for _, g in groups:
        ids = sorted(g["sid"].astype(int).tolist())
        rng.shuffle(ids)
        target = max(1, int(round(n * len(g) / len(df))))
        picked.extend(ids[:target])

    picked = list(dict.fromkeys(picked))
    if len(picked) > n:
        rng.shuffle(picked)
        picked = picked[:n]
    elif len(picked) < n:
        rest = [x for x in df["sid"].astype(int).tolist() if x not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])

    return df[df["sid"].isin(set(picked))].copy()


# =============================================================================
# Dataset / model / writer
# =============================================================================

def load_dataset(a, cohort):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    cohort_map = cohort.set_index("sid").to_dict("index")

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts or sid not in cohort_map:
            continue

        p = prompts[sid]
        gt = canon_rel(traj.normalize_relation(base, p["answer_raw"]))
        if gt not in REL:
            continue

        old = cohort_map[sid]
        meta.append(
            {
                "sid": sid,
                "gt": gt,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "question_text": str(p["question_text"]),
                "baseline_prediction": canon_rel(old["baseline_prediction"]),
                "baseline_correct": bool(old["baseline_correct"]),
                "competitor": canon_rel(old["competitor"]),
                "prior_sequence_margin": float(old["sequence_margin"]),
            }
        )

    return two, meta, rec_by_sid, prompts, records


def load_model(a, two):
    spec = base.merged_model_specs(two)[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    print(f"[model] loading {spec.repo_id}", flush=True)

    try:
        model = cls.from_pretrained(spec.repo_id, **kw)
    except TypeError:
        kw["torch_dtype"] = kw.pop("dtype")
        model = cls.from_pretrained(spec.repo_id, **kw)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)

    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    return model, processor, decoder_layers, decoder_path, spec


def load_writers_npz(path: Path, targets):
    if not path.exists():
        raise FileNotFoundError(path)

    z = np.load(path, allow_pickle=True)
    writers = {T: {} for T in targets}

    for T in targets:
        for r in REL:
            candidates = [
                f"L{T}_{r}",
                f"L{T}_{DISPLAY[r]}",
            ]
            key = next((k for k in candidates if k in z.files), None)
            if key is None:
                raise RuntimeError(
                    f"{path}: missing writer for L{T}/{r}; tried {candidates}"
                )
            writers[T][r] = np.asarray(z[key], dtype=np.float32)

    return writers


def calibrate_writers(
    *,
    a,
    all_meta,
    rec_by_sid,
    model,
    processor,
    decoder_layers,
    targets,
    device,
    outdir,
):
    # Match original 30% stratified writer calibration.
    train, _heldout = traj.stratified_split(
        list(all_meta),
        a.train_ratio,
        a.seed,
    )

    q_by_sid = {}

    for m in tqdm(train, desc="CALIBRATE late writers"):
        sid = int(m["sid"])
        real = gray = None

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

            hr = dyn.capture_cpu(model, decoder_layers, rb, targets)
            hg = dyn.capture_cpu(model, decoder_layers, gb, targets)

            q_by_sid[sid] = {
                T: (hr[T][0, -1] - hg[T][0, -1]).astype(np.float32)
                for T in targets
            }

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

    writers, writer_geom = dyn.learn_writers(
        train,
        q_by_sid,
        targets,
        a.writer_mode,
    )

    pd.DataFrame(writer_geom).to_csv(
        outdir / "writer_geometry.csv",
        index=False,
    )
    np.savez_compressed(
        outdir / "learned_writers.npz",
        **{
            f"L{T}_{DISPLAY[r]}": writers[T][r]
            for T in targets
            for r in REL
        },
    )

    return writers, train


# =============================================================================
# L26 horizontal selection
# =============================================================================

def select_horizontal_positions(
    *,
    sid,
    gt,
    selector_layer,
    top_k,
    categories_allowed,
    model,
    processor,
    decoder_layers,
    rb,
    gb,
    subject,
    reference,
    writers_r,
    targets,
):
    """
    Same writer-guided mediation as the old oracle ranking, but ONLY at one
    source layer. Return Top-K positive TEXT positions.
    """
    ids = rb["input_ids"][0].detach().cpu().tolist()
    cats, toks = dyn.build_categories(
        model,
        processor,
        rb,
        ids,
        subject,
        reference,
    )

    # Gray source activation at selector layer.
    hgray = dyn.capture_cpu(
        model,
        decoder_layers,
        gb,
        [selector_layer],
    )[selector_layer][0].astype(np.float32)

    graph_layers = sorted(set([selector_layer] + list(targets)))
    cap = None

    try:
        with torch.enable_grad():
            cap = dyn.forward_graph(
                model,
                decoder_layers,
                rb,
                graph_layers,
                selector_layer,
            )

            objective_terms = []
            for T in targets:
                s_hat = torch.as_tensor(
                    normalize_np(writers_r[T]),
                    device=cap.states[T].device,
                    dtype=torch.float32,
                )
                objective_terms.append(
                    torch.dot(
                        cap.states[T][0, -1].float(),
                        s_hat,
                    )
                )
            objective = torch.stack(objective_terms).sum()

            grad26 = torch.autograd.grad(
                objective,
                cap.states[selector_layer],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            hreal = (
                cap.states[selector_layer][0]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            G = (
                grad26[0]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        npos = min(
            len(ids),
            len(cats),
            len(toks),
            hreal.shape[0],
            hgray.shape[0],
            G.shape[0],
        )

        candidates = []

        # Match the old causal ranking: exclude the final text/answer-prompt state.
        for pos in range(max(0, npos - 1)):
            cat = str(cats[pos])
            broad = str(dyn.broad_category(cat))

            # This experiment is about causal TEXT token trajectories.
            if broad in ("visual", "last"):
                continue
            if categories_allowed and broad not in categories_allowed:
                continue

            delta = (hreal[pos] - hgray[pos]).astype(np.float32)
            grad = G[pos].astype(np.float32)
            med = float(np.dot(delta, grad))

            candidates.append(
                {
                    "sid": int(sid),
                    "gt": str(gt),
                    "selector_layer": int(selector_layer),
                    "position": int(pos),
                    "token_id": int(ids[pos]),
                    "token": str(toks[pos]).replace("\n", "\\n"),
                    "category": cat,
                    "broad_category": broad,
                    "mediation": med,
                    "delta_h_norm": float(np.linalg.norm(delta)),
                    "writer_grad_norm": float(np.linalg.norm(grad)),
                }
            )

        positive = [
            r for r in candidates
            if np.isfinite(float(r["mediation"]))
            and float(r["mediation"]) > 0
        ]
        positive.sort(
            key=lambda r: float(r["mediation"]),
            reverse=True,
        )
        selected = positive[: int(top_k)]

        for rank, r in enumerate(selected, 1):
            r["rank"] = int(rank)

        return selected, candidates

    finally:
        if cap is not None:
            cap.close()


def selected_to_specs(selected, selector_layer):
    """
    Convert one-layer selected positions into fixed-position trajectories whose
    latest target layer is selector_layer.
    """
    specs = []
    for r in selected:
        specs.append(
            {
                "real_position": int(r["position"]),
                "max_target_layer": int(selector_layer),
                "min_target_layer": int(selector_layer),
                "best_rank": int(r["rank"]),
                "token": str(r["token"]),
                "category": str(r["category"]),
                "broad_category": str(r["broad_category"]),
            }
        )
    return specs


# =============================================================================
# Old-vs-L26 position overlap
# =============================================================================

def old_selected_by_sid(old_sel):
    if old_sel is None or len(old_sel) == 0:
        return {}

    out = {}
    for sid, g in old_sel.groupby("sid"):
        out[int(sid)] = g.sort_values("rank").copy()
    return out


def overlap_row(sid, gt, selected_l26, old_rows):
    new_states = {
        (int(r["selector_layer"]), int(r["position"]))
        for r in selected_l26
    }
    new_pos = {int(r["position"]) for r in selected_l26}

    if old_rows is None or len(old_rows) == 0:
        return {
            "sid": int(sid),
            "gt": gt,
            "old_N": 0,
            "l26_N": len(new_pos),
            "position_intersection": np.nan,
            "position_recall_old_by_l26": np.nan,
            "position_precision_l26_vs_old": np.nan,
            "position_jaccard": np.nan,
            "exact_state_intersection": np.nan,
            "exact_state_recall_old_by_l26": np.nan,
        }

    old_states = {
        (int(r.source_layer), int(r.position))
        for r in old_rows.itertuples()
    }
    old_pos = {
        int(r.position)
        for r in old_rows.itertuples()
    }

    pinter = old_pos & new_pos
    sinter = old_states & new_states
    punion = old_pos | new_pos

    return {
        "sid": int(sid),
        "gt": gt,
        "old_N": len(old_pos),
        "l26_N": len(new_pos),
        "position_intersection": len(pinter),
        "position_recall_old_by_l26": (
            len(pinter) / len(old_pos)
            if old_pos else np.nan
        ),
        "position_precision_l26_vs_old": (
            len(pinter) / len(new_pos)
            if new_pos else np.nan
        ),
        "position_jaccard": (
            len(pinter) / len(punion)
            if punion else np.nan
        ),
        "exact_state_intersection": len(sinter),
        "exact_state_recall_old_by_l26": (
            len(sinter) / len(old_states)
            if old_states else np.nan
        ),
    }


# =============================================================================
# Conditions / generation
# =============================================================================

def build_condition_patch(scored, condition, scale, threshold, selector_layer):
    if condition == "l26_trajectory_signed":
        return gate.build_real_update_patch_map(
            scored,
            "real_signed",
            scale,
            threshold,
        )

    if condition == "l26_trajectory_positive":
        return gate.build_real_update_patch_map(
            scored,
            "real_positive",
            scale,
            threshold,
        )

    if condition == "l26_trajectory_negative_cancel":
        return gate.build_real_update_patch_map(
            scored,
            "real_negative_cancel",
            scale,
            threshold,
        )

    if condition == "l26_trajectory_all_amplify":
        return gate.build_real_update_patch_map(
            scored,
            "real_all_amplify",
            scale,
            threshold,
        )

    if condition == "l26_only_signed":
        rows = [
            e for e in scored
            if int(e["update_layer"]) == int(selector_layer)
        ]
        return gate.build_real_update_patch_map(
            rows,
            "real_signed",
            scale,
            threshold,
        )

    if condition == "l26_only_all_amplify":
        rows = [
            e for e in scored
            if int(e["update_layer"]) == int(selector_layer)
        ]
        return gate.build_real_update_patch_map(
            rows,
            "real_all_amplify",
            scale,
            threshold,
        )

    raise ValueError(f"Unknown condition: {condition}")


# =============================================================================
# Summaries
# =============================================================================

def summarize_overlap(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    rows = []

    for relation, g in list(df.groupby("gt")) + [("ALL", df)]:
        rows.append(
            {
                "relation": DISPLAY.get(relation, relation),
                "N": int(len(g)),
                "mean_old_N": safe_mean(g["old_N"]),
                "mean_l26_N": safe_mean(g["l26_N"]),
                "mean_position_intersection": safe_mean(
                    g["position_intersection"]
                ),
                "mean_position_recall_old_by_l26": safe_mean(
                    g["position_recall_old_by_l26"]
                ),
                "median_position_recall_old_by_l26": safe_median(
                    g["position_recall_old_by_l26"]
                ),
                "mean_position_precision_l26_vs_old": safe_mean(
                    g["position_precision_l26_vs_old"]
                ),
                "mean_position_jaccard": safe_mean(
                    g["position_jaccard"]
                ),
                "mean_exact_state_recall_old_by_l26": safe_mean(
                    g["exact_state_recall_old_by_l26"]
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_selection_categories(sel_df, by_relation=False):
    if sel_df is None or len(sel_df) == 0:
        return pd.DataFrame()

    keys = ["broad_category"]
    if by_relation:
        keys = ["gt", "broad_category"]

    agg = (
        sel_df.groupby(keys, as_index=False)
        .agg(
            selected_rows=("sid", "size"),
            N_samples=("sid", "nunique"),
            mean_rank=("rank", "mean"),
            mean_mediation=("mediation", "mean"),
            median_mediation=("mediation", "median"),
        )
    )

    if by_relation:
        denom = (
            sel_df.groupby("gt")["sid"].nunique()
            .to_dict()
        )
        agg["sample_presence_rate"] = [
            int(n) / max(int(denom.get(gt, 0)), 1)
            for gt, n in zip(agg["gt"], agg["N_samples"])
        ]
        agg["relation"] = agg["gt"].map(
            lambda x: DISPLAY.get(x, x)
        )
    else:
        denom = int(sel_df["sid"].nunique())
        agg["sample_presence_rate"] = (
            agg["N_samples"] / max(denom, 1)
        )

    return agg


def prior_reference_summary(prior_nonbase, cohort_sids, scale):
    if prior_nonbase is None or len(prior_nonbase) == 0:
        return pd.DataFrame()

    x = prior_nonbase[
        prior_nonbase["sid"].isin(set(map(int, cohort_sids)))
    ].copy()

    # Prefer exact old real_signed scale.
    x = x[
        (x["condition"].astype(str) == "real_signed")
        & (
            np.isclose(
                pd.to_numeric(x["scale"], errors="coerce").astype(float),
                float(scale),
            )
        )
    ].copy()

    if len(x) == 0:
        return pd.DataFrame()

    rows = [
        {
            "condition": "old_crosslayer_real_signed_reference",
            "scale": float(scale),
            "N": int(len(x)),
            "patched_accuracy": float(x["correct"].astype(bool).mean()),
        }
    ]

    for gt, g in x.groupby("gt"):
        rows.append(
            {
                "condition": "old_crosslayer_real_signed_reference",
                "scale": float(scale),
                "N": int(len(g)),
                "relation": DISPLAY.get(gt, gt),
                "patched_accuracy": float(g["correct"].astype(bool).mean()),
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
    conditions = parse_csv_set(a.conditions)
    categories_allowed = set(parse_csv_set(a.causal_categories))

    valid_conditions = {
        "l26_trajectory_signed",
        "l26_only_signed",
        "l26_trajectory_positive",
        "l26_trajectory_negative_cancel",
        "l26_trajectory_all_amplify",
        "l26_only_all_amplify",
    }
    unknown = set(conditions) - valid_conditions
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    if not update_layers:
        raise ValueError("No update layers")
    if min(update_layers) < 1:
        raise ValueError("--update-layers must be >=1")
    if max(update_layers) > int(a.selector_layer):
        raise ValueError(
            "--update-layers cannot exceed --selector-layer in this trajectory test"
        )

    outdir = Path(a.output_dir)
    ensure_output_dir(outdir, a.overwrite)
    error_path = outdir / "errors.jsonl"

    prior_dir = Path(a.prior_real_update_dir)
    (
        cohort,
        old_sel,
        prior_metadata,
        prior_nonbase,
    ) = load_prior_run(prior_dir)

    cohort = stratified_cap_df(
        cohort,
        int(a.eval_max_samples),
        a.seed + 1,
    )

    two, eval_meta, rec_by_sid, prompts, records = load_dataset(
        a,
        cohort,
    )

    # Full metadata is needed only if writer recalibration is required.
    all_meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = canon_rel(traj.normalize_relation(base, p["answer_raw"]))
        if gt not in REL:
            continue
        all_meta.append(
            {
                "sid": sid,
                "gt": gt,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "question_text": str(p["question_text"]),
            }
        )

    old_by_sid = old_selected_by_sid(old_sel)

    model = processor = None

    selected_rows_all = []
    update_rows_all = []
    overlap_rows = []
    generation_rows = []

    try:
        (
            model,
            processor,
            decoder_layers,
            decoder_path,
            spec,
        ) = load_model(a, two)

        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        needed_layers = sorted(
            set(
                [int(a.selector_layer)]
                + list(targets)
                + list(update_layers)
                + [L - 1 for L in update_layers]
            )
        )
        bad = [L for L in needed_layers if not 0 <= L < n_layers]
        if bad:
            raise ValueError(
                f"Requested layers outside 0..{n_layers-1}: {bad}"
            )

        # -------------------------------------------------------------
        # Writer: preferably reuse exact old writer.
        # -------------------------------------------------------------
        if a.writer_npz:
            writers = load_writers_npz(
                Path(a.writer_npz),
                targets,
            )
            writer_source = str(Path(a.writer_npz))
            writer_train_N = None
        else:
            writers, writer_train = calibrate_writers(
                a=a,
                all_meta=all_meta,
                rec_by_sid=rec_by_sid,
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                targets=targets,
                device=device,
                outdir=outdir,
            )
            writer_source = "recalibrated_in_this_run"
            writer_train_N = len(writer_train)

        # -------------------------------------------------------------
        # Exact sequence answer strings / reduction from prior run.
        # -------------------------------------------------------------
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
            canon_rel(k): str(v)
            for k, v in texts.items()
        }
        for r in REL:
            if r not in texts:
                raise RuntimeError(
                    f"Prior metadata candidate_texts missing {r}: {texts}"
                )

        candidate_ids = gate.encode_candidate_ids(
            processor,
            texts,
        )
        reduction = str(
            prior_metadata.get(
                "sequence_score_reduction",
                "mean",
            )
        )
        if reduction not in ("mean", "sum"):
            reduction = "mean"

        print("=" * 190)
        print("L26 HORIZONTAL TOP-K -> FIXED TOKEN TRAJECTORY REAL-UPDATE GATING")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"prior_run={prior_dir}")
        print(f"N={len(eval_meta)}")
        print(f"selector_layer=L{a.selector_layer} topK={a.top_k}")
        print(f"update_layers={update_layers}")
        print(f"target_layers={targets}")
        print(f"scale={a.scale} threshold={a.decision_threshold}")
        print(f"conditions={conditions}")
        print(f"writer_source={writer_source}")
        print(f"sequence_reduction={reduction}")
        print()
        print(
            "IMPORTANT: L26 Top-K uses GT writer and layerwise sign uses GT "
            "sequence margin. This tests WHERE dimensionality, not non-oracle use."
        )
        print()

        # -------------------------------------------------------------
        # Per sample
        # -------------------------------------------------------------
        for m in tqdm(
            eval_meta,
            desc=f"L{a.selector_layer} HORIZONTAL TOP{a.top_k}",
        ):
            sid = int(m["sid"])
            gt = m["gt"]
            real = gray = None

            # Reuse exact old baseline row for common-sample comparisons.
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
                    "text": "",
                }
            )

            try:
                writers_r = {
                    T: writers[T][gt]
                    for T in targets
                }

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(
                    real,
                    a.gray_value,
                )

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

                # -----------------------------------------------------
                # A) L26-only horizontal GT-writer causal selection
                # -----------------------------------------------------
                selected, _all_candidates = select_horizontal_positions(
                    sid=sid,
                    gt=gt,
                    selector_layer=int(a.selector_layer),
                    top_k=int(a.top_k),
                    categories_allowed=categories_allowed,
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

                if not selected:
                    raise RuntimeError(
                        f"No positive L{a.selector_layer} text positions"
                    )

                selected_rows_all.extend(selected)

                overlap_rows.append(
                    overlap_row(
                        sid,
                        gt,
                        selected,
                        old_by_sid.get(sid),
                    )
                )

                # -----------------------------------------------------
                # B) Follow SAME selected positions from L8..L26
                # -----------------------------------------------------
                specs = selected_to_specs(
                    selected,
                    int(a.selector_layer),
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

                # NoImage is intentionally absent here; primary experiment
                # is actual REAL update only.
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
                        "No REAL updates on L26-selected trajectories"
                    )

                grad_layers = sorted(
                    set(int(e["update_layer"]) for e in entries)
                )

                # -----------------------------------------------------
                # C) Same oracle HOW sign as prior full-440 experiment
                # -----------------------------------------------------
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
                        f"prior={m['prior_sequence_margin']:.6f} "
                        f"replay={replay_margin:.6f}"
                    )

                scored = gate.attach_decision_scores(
                    entries,
                    grad_gt,
                    grad_comp,
                )

                if not scored:
                    raise RuntimeError(
                        "No selected trajectory update received decision gradient"
                    )

                # Annotate selector rank / selector mediation onto each layer row.
                sel_lookup = {
                    int(r["position"]): r
                    for r in selected
                }

                for e in scored:
                    srow = sel_lookup[int(e["real_position"])]
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
                    export["selector_layer"] = int(a.selector_layer)
                    export["selector_rank"] = int(srow["rank"])
                    export["selector_mediation"] = float(
                        srow["mediation"]
                    )
                    export["competitor"] = comp
                    export["replayed_sequence_margin"] = replay_margin
                    update_rows_all.append(export)

                # -----------------------------------------------------
                # D) Actual generation
                # -----------------------------------------------------
                for cond in conditions:
                    pmap, counts = build_condition_patch(
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
                            "text": text,
                        }
                    )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "phase": "sample",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": (
                            traceback.format_exc().splitlines()[-80:]
                        ),
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
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =================================================================
        # Save raw tables
        # =================================================================
        sel_df = pd.DataFrame(selected_rows_all)
        upd_df = pd.DataFrame(update_rows_all)
        overlap_df = pd.DataFrame(overlap_rows)
        gen_df = pd.DataFrame(generation_rows)

        sel_df.to_csv(
            outdir / "l26_selected_positions.csv",
            index=False,
        )
        upd_df.to_csv(
            outdir / "per_real_update_decision_score.csv",
            index=False,
        )
        overlap_df.to_csv(
            outdir / "old_vs_l26_overlap_per_sample.csv",
            index=False,
        )
        gen_df.to_csv(
            outdir / "generation_per_sample.csv",
            index=False,
        )

        # =================================================================
        # Summaries
        # =================================================================
        overlap_summary = summarize_overlap(overlap_df)
        overlap_summary.to_csv(
            outdir / "old_vs_l26_overlap_summary.csv",
            index=False,
        )

        cat_summary = summarize_selection_categories(
            sel_df,
            by_relation=False,
        )
        rel_cat_summary = summarize_selection_categories(
            sel_df,
            by_relation=True,
        )
        cat_summary.to_csv(
            outdir / "selection_category_summary.csv",
            index=False,
        )
        rel_cat_summary.to_csv(
            outdir / "selection_relation_category_summary.csv",
            index=False,
        )

        if len(upd_df):
            layer_summary = gate.summarize_layers(upd_df)
            layer_summary.to_csv(
                outdir / "layer_real_update_summary.csv",
                index=False,
            )

            layer_rel_summary = gate.summarize_layers(
                upd_df,
                extra_keys=["gt"],
            )
            layer_rel_summary.to_csv(
                outdir / "layer_real_update_by_relation.csv",
                index=False,
            )
        else:
            layer_summary = pd.DataFrame()
            layer_rel_summary = pd.DataFrame()

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

        common_sids = (
            gen_df[
                gen_df["condition"] != "baseline"
            ]["sid"].drop_duplicates().astype(int).tolist()
            if len(gen_df)
            else []
        )

        prior_ref = prior_reference_summary(
            prior_nonbase,
            common_sids,
            a.scale,
        )
        prior_ref.to_csv(
            outdir / "old_crosslayer_oracle_reference.csv",
            index=False,
        )

        # Selector count diagnostics.
        if len(sel_df):
            selected_count_by_sid = sel_df.groupby("sid").size()
            mean_selected = float(selected_count_by_sid.mean())
            full_k_fraction = float(
                np.mean(selected_count_by_sid.to_numpy() >= int(a.top_k))
            )
        else:
            mean_selected = float("nan")
            full_k_fraction = float("nan")

        # Main condition lookup.
        primary_row = pd.DataFrame()
        if len(gen_summary):
            primary_row = gen_summary[
                gen_summary["condition"] == "l26_trajectory_signed"
            ]

        # =================================================================
        # Report
        # =================================================================
        report = [
            "=" * 190,
            f"L{a.selector_layer} HORIZONTAL TOP-{a.top_k} -> FIXED TOKEN TRAJECTORY",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested from prior cohort={len(eval_meta)}",
            f"N successful selector samples={sel_df['sid'].nunique() if len(sel_df) else 0}",
            f"N successful generation samples={len(common_sids)}",
            f"selector layer=L{a.selector_layer}",
            f"topK={a.top_k}",
            f"mean actual selected positions={mean_selected:.4f}",
            f"fraction with full K positions={full_k_fraction:.4f}",
            f"update layers={update_layers}",
            f"scale={a.scale}",
            f"writer source={writer_source}",
            "",
            "PRIMARY TEST",
            "-" * 190,
            (
                primary_row.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(primary_row)
                else "No l26_trajectory_signed result."
            ),
            "",
            "ALL GENERATION CONDITIONS",
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
            "OLD CROSS-LAYER ORACLE REFERENCE FROM PRIOR RUN",
            "-" * 190,
            (
                prior_ref.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(prior_ref)
                else "No matching old real_signed reference found."
            ),
            "",
            "OLD TOP7 vs L26 TOP7 OVERLAP",
            "-" * 190,
            (
                overlap_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(overlap_summary)
                else "EMPTY"
            ),
            "",
            "L26 TOP-K TOKEN CATEGORY SUMMARY",
            "-" * 190,
            (
                cat_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(cat_summary)
                else "EMPTY"
            ),
            "",
            "LAYERWISE REAL-UPDATE SCORE ON FIXED L26-SELECTED TRAJECTORIES",
            "-" * 190,
            (
                layer_summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
                if len(layer_summary)
                else "EMPTY"
            ),
            "",
            "Interpretation:",
            "  If l26_trajectory_signed stays close to the old cross-layer real_signed",
            "  oracle, then cross-layer WHERE can be collapsed to one late horizontal",
            "  token-position selection. Layers mainly describe how those same token",
            "  trajectories are updated (+/-), rather than which token is causal.",
            "",
            "  If l26_only_signed is much weaker than l26_trajectory_signed, the gain",
            "  genuinely uses the accumulated trajectory, not merely an L26 endpoint patch.",
            "",
            "  If position overlap with old Top7 is high while exact-state overlap is",
            "  lower, that directly supports 'same token position, different strongest layer'.",
            "",
            "Oracle caveat:",
            "  L26 WHERE still uses GT writer; HOW still uses GT sequence margin.",
            "  This experiment only tests the dimensionality/simplification of WHERE.",
        ]

        report_text = "\n".join(report) + "\n"
        print(report_text)
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(
            outdir / "metadata.json",
            {
                "script": "eval_l26_horizontal_top7_real_update_trajectory_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "prior_real_update_dir": str(prior_dir),
                "selector_layer": int(a.selector_layer),
                "top_k": int(a.top_k),
                "update_layers": update_layers,
                "target_layers": targets,
                "scale": float(a.scale),
                "decision_threshold": float(a.decision_threshold),
                "conditions": conditions,
                "writer_source": writer_source,
                "writer_train_N_if_recalibrated": writer_train_N,
                "writer_mode": a.writer_mode,
                "sequence_score_reduction": reduction,
                "candidate_texts": texts,
                "where_definition": (
                    "Top-K positive GT-writer Real-Gray mediation at one "
                    f"horizontal selector layer L{a.selector_layer}"
                ),
                "how_definition": (
                    "same token positions reused across update layers; "
                    "B=dot(hR[L,p]-hR[L-1,p], grad[S_GT-S_competitor])"
                ),
                "oracle_note": (
                    "WHERE uses GT writer and HOW uses GT sequence margin. "
                    "This is a mechanism simplification test, not non-oracle."
                ),
            },
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
