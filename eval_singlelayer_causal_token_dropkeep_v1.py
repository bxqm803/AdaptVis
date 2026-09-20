#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_singlelayer_causal_token_dropkeep_v1.py

Purpose
=======
Validate whether the decision-relevant TEXT-token updates at a single decoder
layer form an important route to the final generation.

We test L25 and L26 independently by default.

For each sample and each tested layer L:

  1) Rank eligible TEXT token positions using the existing writer-based causal
     score (default):

        M_L,p
          = (h_real[L,p] - h_gray[L,p])^T dJ_r/dh_real[L,p]

     where J_r is the GT relation's late writer objective.

     For diagnostics we also export:

        B_L,p
          = (h_real[L,p] - h_real[L-1,p])^T dJ_r/dh_real[L,p]

     and gradient norm.

  2) Define the actual block update:

        a_L,p = h_real[L,p] - h_real[L-1,p]

  3) Run ACTUAL model.generate() under:

     drop_top:
        remove the current layer's update only at Top-K positions
        h_L,p <- h_{L-1,p}

     keep_top:
        keep the current layer's update only at Top-K eligible text positions;
        remove the update from all other eligible text positions.

     Optional category-matched random controls:
        drop_random / keep_random

     Optional layer reference:
        drop_all_text removes the L-th block update from every eligible text
        position.

Important
=========
This does NOT zero the whole hidden state.  It only deletes the NEW update
written by the tested block.  Information accumulated through L-1 is preserved.

Visual tokens and the prompt-final token are excluded from the eligible set by
default and are never suppressed by keep_top/drop_top.

The experiment is an ORACLE mechanistic diagnostic because GT selects the
relation-specific late writer.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_singlelayer_causal_token_dropkeep_v1.py \
  --layers 25,26 \
  --ks 1,2,3,5,7,10,15,20 \
  --rank-by mediation \
  --eval-max-samples 80 \
  --conditions drop_top,keep_top,drop_random,keep_random \
  --random-repeats 1 \
  --output-dir output/qwen3b_singlelayer_L25_L26_dropkeep_n80_v1 \
  --overwrite

Faster smoke test
=================
CUDA_VISIBLE_DEVICES=0 python -u eval_singlelayer_causal_token_dropkeep_v1.py \
  --layers 25,26 \
  --ks 1,3,5,7,10,15 \
  --rank-by mediation \
  --eval-max-samples 40 \
  --conditions drop_top,keep_top \
  --output-dir output/qwen3b_singlelayer_L25_L26_dropkeep_n40_v1 \
  --overwrite

Outputs
=======
per_token_scores.csv
    Every eligible text position at L25/L26 with M, B, grad norm, rank.

eligible_counts.csv
    Number of eligible text tokens and positive-score tokens per sample/layer.

generation_per_sample.csv
    Actual generation for clean, drop_all_text, and every layer/K/condition.

generation_summary.csv
    Accuracy, gain vs clean, W2C/C2W, repair/preserve, mean selected fraction.

generation_by_baseline_status.csv
    Same analysis split into baseline-correct / baseline-wrong cohorts.

selection_summary.csv
    Composition and score statistics of selected Top-K states.

metadata.json
analysis_summary.txt
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
UNDISPLAY = {
    "left": "left", "right": "right",
    "on": "above", "under": "below",
    "above": "above", "below": "below",
}
EPS = 1e-12


# =============================================================================
# CLI / generic
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
        "--layers",
        default="25,26",
        help="Single receiver layers tested independently.",
    )
    p.add_argument(
        "--target-layers",
        default="31,32,33,34,35",
        help="Late writer layers used by J_r.",
    )
    p.add_argument(
        "--writer-npz",
        default="",
        help=(
            "Optional learned_writers.npz. If omitted, writers are recalibrated "
            "on the usual stratified 30%% calibration split."
        ),
    )
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)

    p.add_argument(
        "--rank-by",
        default="mediation",
        choices=["mediation", "real_update", "grad_norm"],
        help=(
            "mediation: (Real-Gray)^T grad J (matches old causal-state definition); "
            "real_update: (h_L-h_{L-1})^T grad J; "
            "grad_norm: ||grad J||."
        ),
    )
    p.add_argument(
        "--positive-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For mediation/real_update ranking, restrict Top-K to positive scores. "
            "Ignored for grad_norm."
        ),
    )

    p.add_argument("--ks", default="1,2,3,5,7,10,15,20")
    p.add_argument(
        "--conditions",
        default="drop_top,keep_top,drop_random,keep_random",
        help=(
            "Comma-separated subset of "
            "drop_top,keep_top,drop_random,keep_random."
        ),
    )
    p.add_argument("--random-repeats", type=int, default=1)
    p.add_argument(
        "--run-drop-all-text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--eligible-categories",
        default="subject,reference,relation_words,other_text",
        help=(
            "Broad categories allowed in the single-layer ranking. "
            "visual and last are excluded regardless."
        ),
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--eval-scope",
        default="all_data",
        choices=["all_data", "test"],
        help=(
            "all_data is the same oracle mechanistic setting used by the old "
            "full-440 causal diagnostics; test uses the held-out 70%% split."
        ),
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="0 = all samples in eval scope.",
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
    out = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_set(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def safe_mean(xs):
    z = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(z)) if z else float("nan")


def safe_median(xs):
    z = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.median(z)) if z else float("nan")


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row: Mapping):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def normalize_np(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v.copy()
    return (v / n).astype(np.float32)


# =============================================================================
# Data / model
# =============================================================================

def load_data(a):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
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
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, heldout = traj.stratified_split(meta, a.train_ratio, a.seed)

    if a.eval_scope == "all_data":
        test = list(meta)
    else:
        test = list(heldout)

    if int(a.eval_max_samples) > 0:
        test = traj.stratified_cap(test, int(a.eval_max_samples), a.seed + 1)

    return two, meta, train, heldout, test, rec_by_sid


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

    print(f"[MODEL] loading {spec.repo_id}", flush=True)
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


# =============================================================================
# Writers
# =============================================================================

def _writer_key_candidates(layer: int, rel: str) -> List[str]:
    surface = DISPLAY[rel]
    return [
        f"L{layer}_{surface}",
        f"L{layer}_{rel}",
        f"{layer}_{surface}",
        f"{layer}_{rel}",
    ]


def load_writer_npz(path: Path, targets: Sequence[int]):
    z = np.load(path)
    writers = {int(T): {} for T in targets}

    for T in targets:
        for r in REL:
            found = None
            for k in _writer_key_candidates(int(T), r):
                if k in z:
                    found = np.asarray(z[k], dtype=np.float32)
                    break
            if found is None:
                raise RuntimeError(
                    f"{path} has no writer for L{T}/{r}. "
                    f"Tried {_writer_key_candidates(int(T), r)}"
                )
            writers[int(T)][r] = found

    return writers


def calibrate_writers(
    *,
    model,
    processor,
    decoder_layers,
    train,
    rec_by_sid,
    targets,
    device,
    gray_value,
    writer_mode,
):
    q_by_sid = {}

    for m in tqdm(train, desc="CALIBRATE late writers"):
        sid = int(m["sid"])
        real = gray = None
        try:
            real = base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")
            gray = dyn.make_gray_image(real, gray_value)

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

    writers, geom = dyn.learn_writers(train, q_by_sid, targets, writer_mode)
    return writers, geom


# =============================================================================
# Single-layer update suppression
# =============================================================================

class SingleLayerUpdateSuppressor:
    """
    Remove the tested block's NEW residual update at selected prompt positions:

        h_L[p] <- h_{L-1}[p]

    The entire pre-L representation is preserved.

    mode="drop":
        suppress exactly selected_positions

    mode="keep":
        suppress eligible_positions - selected_positions
    """

    def __init__(
        self,
        decoder_layers,
        layer: int,
        prompt_len: int,
        eligible_positions: Sequence[int],
        selected_positions: Sequence[int],
        mode: str,
    ):
        self.layer = int(layer)
        self.prompt_len = int(prompt_len)
        self.eligible = set(map(int, eligible_positions))
        self.selected = set(map(int, selected_positions))
        self.mode = str(mode)
        self.prev_state = None
        self.n_suppressed = 0
        self.handles = []

        if self.layer < 1:
            raise ValueError("SingleLayerUpdateSuppressor requires L>=1")
        if self.mode not in {"drop", "keep"}:
            raise ValueError(self.mode)

        self.handles.append(
            decoder_layers[self.layer - 1].register_forward_hook(self._prev_hook)
        )
        self.handles.append(
            decoder_layers[self.layer].register_forward_hook(self._layer_hook)
        )

    def _prev_hook(self, _m, _inp, out):
        x = traj.first_tensor(out)
        if int(x.shape[1]) == self.prompt_len:
            self.prev_state = x.detach().clone()
        return None

    def _layer_hook(self, _m, _inp, out):
        x = traj.first_tensor(out)
        if int(x.shape[1]) != self.prompt_len:
            return None
        if self.prev_state is None:
            raise RuntimeError(
                f"L{self.layer}: previous-layer state was not captured."
            )

        if self.mode == "drop":
            suppress = self.selected
        else:
            suppress = self.eligible - self.selected

        if not suppress:
            self.n_suppressed = 0
            return None

        y = x.clone()
        count = 0
        for p in sorted(suppress):
            if (
                0 <= p < int(y.shape[1])
                and p < int(self.prev_state.shape[1])
            ):
                y[0, p] = self.prev_state[0, p].to(
                    device=y.device, dtype=y.dtype
                )
                count += 1

        self.n_suppressed = count
        return traj.replace_first_tensor(out, y)

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []
        self.prev_state = None


@torch.inference_mode()
def generate_condition(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    max_new_tokens,
    layer=None,
    eligible_positions=None,
    selected_positions=None,
    mode=None,
):
    editor = None
    try:
        if layer is not None:
            editor = SingleLayerUpdateSuppressor(
                decoder_layers=decoder_layers,
                layer=int(layer),
                prompt_len=int(batch["input_ids"].shape[1]),
                eligible_positions=eligible_positions or [],
                selected_positions=selected_positions or [],
                mode=str(mode),
            )

        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )
        pred = traj.normalize_relation(base, text)

        return {
            "prediction": pred,
            "text": text,
            "n_suppressed": (
                int(editor.n_suppressed) if editor is not None else 0
            ),
        }
    finally:
        if editor is not None:
            editor.close()


# =============================================================================
# Ranking
# =============================================================================

def build_per_layer_scores(
    *,
    sid: int,
    gt: str,
    layers: Sequence[int],
    ids,
    cats,
    toks,
    real_states,
    gray_states,
    grad_by_layer,
    eligible_categories: Sequence[str],
):
    wanted = set(map(str, eligible_categories))
    out = {}

    for L in layers:
        H_L = real_states[L][0].astype(np.float32)
        H_prev = real_states[L - 1][0].astype(np.float32)
        H_gray = gray_states[L][0].astype(np.float32)
        G = grad_by_layer[L][0].astype(np.float32)

        npos = min(
            len(ids),
            len(cats),
            len(toks),
            H_L.shape[0],
            H_prev.shape[0],
            H_gray.shape[0],
            G.shape[0],
        )

        rows = []
        for p in range(max(0, npos - 1)):  # prompt-last excluded
            broad = dyn.broad_category(cats[p])
            if broad in {"visual", "last"}:
                continue
            if wanted and broad not in wanted:
                continue

            delta_rg = (H_L[p] - H_gray[p]).astype(np.float32)
            update = (H_L[p] - H_prev[p]).astype(np.float32)
            grad = G[p].astype(np.float32)

            mediation = float(np.dot(delta_rg, grad))
            real_update_score = float(np.dot(update, grad))

            rows.append({
                "sid": int(sid),
                "gt": DISPLAY[gt],
                "layer": int(L),
                "position": int(p),
                "token_id": int(ids[p]),
                "token": str(toks[p]).replace("\n", "\\n"),
                "category": str(cats[p]),
                "broad_category": str(broad),
                "mediation": mediation,
                "real_update_decision_score": real_update_score,
                "grad_norm": float(np.linalg.norm(grad)),
                "real_gray_norm": float(np.linalg.norm(delta_rg)),
                "real_update_norm": float(np.linalg.norm(update)),
                "mediation_cosine": (
                    mediation
                    / max(
                        float(np.linalg.norm(delta_rg))
                        * float(np.linalg.norm(grad)),
                        EPS,
                    )
                ),
                "real_update_cosine": (
                    real_update_score
                    / max(
                        float(np.linalg.norm(update))
                        * float(np.linalg.norm(grad)),
                        EPS,
                    )
                ),
            })

        out[int(L)] = rows

    return out


def ranking_value(row, rank_by: str) -> float:
    if rank_by == "mediation":
        return float(row["mediation"])
    if rank_by == "real_update":
        return float(row["real_update_decision_score"])
    if rank_by == "grad_norm":
        return float(row["grad_norm"])
    raise ValueError(rank_by)


def rank_rows(rows, rank_by: str, positive_only: bool):
    z = list(rows)

    if positive_only and rank_by in {"mediation", "real_update"}:
        z = [r for r in z if ranking_value(r, rank_by) > 0]

    z.sort(key=lambda r: ranking_value(r, rank_by), reverse=True)

    ranked = []
    for i, r in enumerate(z, 1):
        rr = dict(r)
        rr["rank_by"] = rank_by
        rr["ranking_score"] = ranking_value(r, rank_by)
        rr["rank"] = int(i)
        ranked.append(rr)

    return ranked


def category_matched_random(
    *,
    all_rows,
    top_rows,
    k: int,
    seed: int,
):
    """
    Match Top-K broad-category composition as closely as possible.
    Prefer positions outside Top-K.  Fall back to any unused eligible position.
    """
    refs = list(top_rows[: int(k)])
    if not refs:
        return []

    top_pos = {int(r["position"]) for r in refs}
    candidates = [r for r in all_rows if int(r["position"]) not in top_pos]

    by_cat = defaultdict(list)
    for r in candidates:
        by_cat[str(r["broad_category"])].append(r)

    rng = random.Random(int(seed))
    chosen = []
    used = set()

    for ref in refs:
        cat = str(ref["broad_category"])
        pool = [
            r for r in by_cat.get(cat, [])
            if int(r["position"]) not in used
        ]
        if not pool:
            pool = [
                r for r in candidates
                if int(r["position"]) not in used
            ]
        if not pool:
            break

        r = rng.choice(pool)
        chosen.append(r)
        used.add(int(r["position"]))

    return chosen


# =============================================================================
# Summaries
# =============================================================================

def summarize_generation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    baseline = (
        df[df["condition"] == "clean"]
        .drop_duplicates("sid")
        .set_index("sid")
    )
    rows = []

    group_cols = ["layer", "k", "condition", "repeat"]

    for key, g in df[df["condition"] != "clean"].groupby(
        group_cols, dropna=False
    ):
        layer, k, cond, rep = key
        z = g.copy()

        clean_correct = baseline.loc[z["sid"], "correct"].to_numpy(dtype=bool)
        edited_correct = z["correct"].to_numpy(dtype=bool)

        w2c = int(np.sum((~clean_correct) & edited_correct))
        c2w = int(np.sum(clean_correct & (~edited_correct)))
        n_wrong = int(np.sum(~clean_correct))
        n_correct = int(np.sum(clean_correct))

        rows.append({
            "layer": int(layer),
            "k": int(k),
            "condition": str(cond),
            "repeat": int(rep),
            "N": len(z),
            "baseline_acc": float(np.mean(clean_correct)),
            "edited_acc": float(np.mean(edited_correct)),
            "gain": float(np.mean(edited_correct) - np.mean(clean_correct)),
            "W2C": w2c,
            "C2W": c2w,
            "net": w2c - c2w,
            "repair_rate": w2c / n_wrong if n_wrong else float("nan"),
            "preserve_rate": (
                1.0 - c2w / n_correct if n_correct else float("nan")
            ),
            "changed_prediction_rate": float(
                np.mean(
                    baseline.loc[z["sid"], "prediction"].astype(str).to_numpy()
                    != z["prediction"].astype(str).to_numpy()
                )
            ),
            "mean_n_eligible": safe_mean(z["n_eligible"]),
            "mean_n_selected": safe_mean(z["n_selected"]),
            "mean_selected_fraction": safe_mean(z["selected_fraction"]),
            "mean_n_suppressed": safe_mean(z["n_suppressed"]),
            "mean_selected_positive_fraction": safe_mean(
                z["selected_positive_fraction"]
            ),
            "mean_selected_score": safe_mean(z["selected_mean_score"]),
        })

    # Average repeated random controls into repeat=-1 convenience rows.
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    avg_rows = []
    for (L, k, cond), g in raw.groupby(
        ["layer", "k", "condition"], dropna=False
    ):
        if len(g) <= 1:
            continue
        avg_rows.append({
            "layer": int(L),
            "k": int(k),
            "condition": str(cond) + "_mean",
            "repeat": -1,
            "N": int(round(safe_mean(g["N"]))),
            **{
                c: safe_mean(g[c])
                for c in raw.columns
                if c not in {"layer", "k", "condition", "repeat", "N"}
            },
        })

    if avg_rows:
        raw = pd.concat([raw, pd.DataFrame(avg_rows)], ignore_index=True)

    return raw.sort_values(["layer", "k", "condition", "repeat"])


def summarize_by_baseline_status(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    baseline = (
        df[df["condition"] == "clean"]
        .drop_duplicates("sid")
        .set_index("sid")
    )

    rows = []
    zall = df[df["condition"] != "clean"].copy()

    for (L, k, cond, rep), g in zall.groupby(
        ["layer", "k", "condition", "repeat"], dropna=False
    ):
        for status in (True, False):
            keep = [
                sid for sid in g["sid"].tolist()
                if bool(baseline.loc[sid, "correct"]) == status
            ]
            if not keep:
                continue
            z = g[g["sid"].isin(keep)]
            rows.append({
                "layer": int(L),
                "k": int(k),
                "condition": str(cond),
                "repeat": int(rep),
                "baseline_correct": bool(status),
                "N": len(z),
                "edited_acc": float(z["correct"].mean()),
                "prediction_changed_rate": float(
                    np.mean(
                        baseline.loc[z["sid"], "prediction"].astype(str).to_numpy()
                        != z["prediction"].astype(str).to_numpy()
                    )
                ),
                "mean_n_selected": safe_mean(z["n_selected"]),
                "mean_selected_fraction": safe_mean(z["selected_fraction"]),
            })

    return pd.DataFrame(rows).sort_values(
        ["layer", "k", "condition", "repeat", "baseline_correct"]
    )


def summarize_selection(score_df: pd.DataFrame, rank_by: str, ks: Sequence[int]):
    rows = []

    if score_df.empty:
        return pd.DataFrame()

    for (sid, L), g in score_df.groupby(["sid", "layer"]):
        g = g.sort_values("rank")
        for k in ks:
            z = g.head(int(k))
            if z.empty:
                continue
            counts = Counter(z["broad_category"].astype(str))
            rows.append({
                "sid": int(sid),
                "layer": int(L),
                "k": int(k),
                "n_selected": len(z),
                "mean_ranking_score": safe_mean(z["ranking_score"]),
                "min_ranking_score": float(z["ranking_score"].min()),
                "max_ranking_score": float(z["ranking_score"].max()),
                "positive_fraction": float(np.mean(z["ranking_score"] > 0)),
                "subject": counts.get("subject", 0),
                "reference": counts.get("reference", 0),
                "relation_words": counts.get("relation_words", 0),
                "other_text": counts.get("other_text", 0),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    layers = parse_ints(a.layers)
    targets = parse_ints(a.target_layers)
    ks = parse_ints(a.ks)
    conditions = parse_set(a.conditions)
    eligible_categories = parse_set(a.eligible_categories)

    valid_conditions = {
        "drop_top",
        "keep_top",
        "drop_random",
        "keep_random",
    }
    bad = set(conditions) - valid_conditions
    if bad:
        raise ValueError(f"Unknown conditions: {sorted(bad)}")
    if any(L < 1 for L in layers):
        raise ValueError("Every tested layer must be >=1")
    if not ks:
        raise ValueError("No K values")
    if a.random_repeats < 1:
        raise ValueError("--random-repeats must be >=1")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    two, meta, train, heldout, test, rec_by_sid = load_data(a)

    model = processor = None

    token_score_rows = []
    eligible_count_rows = []
    generation_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        for L in layers + targets + [L - 1 for L in layers]:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid; model has L0..L{n_layers-1}")

        # -------------------------------------------------------------
        # Writers
        # -------------------------------------------------------------
        if a.writer_npz:
            writers = load_writer_npz(Path(a.writer_npz), targets)
            writer_geom = []
            writer_source = str(Path(a.writer_npz))
        else:
            writers, writer_geom = calibrate_writers(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                train=train,
                rec_by_sid=rec_by_sid,
                targets=targets,
                device=device,
                gray_value=a.gray_value,
                writer_mode=a.writer_mode,
            )
            writer_source = "recalibrated"
            pd.DataFrame(writer_geom).to_csv(
                outdir / "writer_geometry.csv", index=False
            )
            np.savez_compressed(
                outdir / "learned_writers.npz",
                **{
                    f"L{T}_{DISPLAY[r]}": writers[T][r]
                    for T in targets for r in REL
                },
            )

        cal_sids = {int(x["sid"]) for x in train}
        eval_sids = {int(x["sid"]) for x in test}
        overlap = len(cal_sids & eval_sids)

        print("\n" + "=" * 132)
        print("SINGLE-LAYER DECISION-PATH TEST: DROP / KEEP TOP-K TEXT UPDATES")
        print("=" * 132)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(f"tested layers={layers}")
        print(f"late writer targets={targets}")
        print(f"writer source={writer_source}")
        print(
            f"rank_by={a.rank_by} positive_only={a.positive_only} "
            f"eligible={eligible_categories}"
        )
        print(f"K={ks}")
        print(f"conditions={conditions} random_repeats={a.random_repeats}")
        print(
            f"eval_scope={a.eval_scope} N={len(test)} "
            f"writer-cal/eval overlap={overlap}"
        )
        print(
            "Intervention removes ONLY the tested block update: "
            "h_L[p] <- h_{L-1}[p]."
        )
        print("visual and prompt-last positions are untouched.")
        print("=" * 132 + "\n")

        capture_layers = sorted(set(layers + [L - 1 for L in layers]))

        # -------------------------------------------------------------
        # Eval
        # -------------------------------------------------------------
        for m in tqdm(test, desc="L25/L26 drop-keep"):
            sid = int(m["sid"])
            gt = str(m["gt"])
            writers_r = {T: writers[T][gt] for T in targets}

            real = gray = None
            cap = None

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

                # Clean actual generation once per sample.
                clean = generate_condition(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    max_new_tokens=a.max_new_tokens,
                )
                clean_pred = clean["prediction"]
                clean_correct = clean_pred == gt

                generation_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "layer": -1,
                    "k": 0,
                    "condition": "clean",
                    "repeat": 0,
                    "prediction": DISPLAY.get(clean_pred, clean_pred),
                    "correct": bool(clean_correct),
                    "text": clean["text"],
                    "n_eligible": 0,
                    "n_selected": 0,
                    "selected_fraction": 0.0,
                    "n_suppressed": 0,
                    "selected_positive_fraction": float("nan"),
                    "selected_mean_score": float("nan"),
                })

                # Prompt states for real updates and Real-Gray terms.
                real_states = dyn.capture_cpu(
                    model, decoder_layers, rb, capture_layers
                )
                gray_states = dyn.capture_cpu(
                    model, decoder_layers, gb, layers
                )

                # One graph gives gradients at both L25 and L26.
                graph_layers = sorted(set(layers + targets))
                cut = min(layers)

                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model,
                        decoder_layers,
                        rb,
                        graph_layers,
                        cut,
                    )

                    objective_terms = []
                    for T in targets:
                        s_hat = torch.as_tensor(
                            normalize_np(writers_r[T]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        objective_terms.append(
                            torch.dot(cap.states[T][0, -1].float(), s_hat)
                        )
                    objective = torch.stack(objective_terms).sum()

                    grads = torch.autograd.grad(
                        objective,
                        [cap.states[L] for L in layers],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                    grad_by_layer = {
                        int(L): g.detach().float().cpu().numpy().astype(np.float32)
                        for L, g in zip(layers, grads)
                    }

                cap.close()
                cap = None

                rows_by_layer = build_per_layer_scores(
                    sid=sid,
                    gt=gt,
                    layers=layers,
                    ids=ids,
                    cats=cats,
                    toks=toks,
                    real_states=real_states,
                    gray_states=gray_states,
                    grad_by_layer=grad_by_layer,
                    eligible_categories=eligible_categories,
                )

                for L in layers:
                    all_rows = rows_by_layer[L]
                    ranked = rank_rows(
                        all_rows,
                        rank_by=a.rank_by,
                        positive_only=bool(a.positive_only),
                    )

                    for r in ranked:
                        token_score_rows.append(r)

                    eligible_positions = [
                        int(r["position"]) for r in all_rows
                    ]

                    positive_available = int(
                        np.sum(
                            [
                                ranking_value(r, a.rank_by) > 0
                                for r in all_rows
                            ]
                        )
                    )

                    eligible_count_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "layer": int(L),
                        "n_prompt_tokens": len(ids),
                        "n_eligible_text": len(all_rows),
                        "n_positive_score": positive_available,
                        "positive_score_fraction": (
                            positive_available / len(all_rows)
                            if all_rows else float("nan")
                        ),
                        "n_rankable": len(ranked),
                        "baseline_correct": bool(clean_correct),
                    })

                    # Layer-level reference: suppress ALL eligible text updates.
                    if a.run_drop_all_text:
                        edited = generate_condition(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            max_new_tokens=a.max_new_tokens,
                            layer=L,
                            eligible_positions=eligible_positions,
                            selected_positions=eligible_positions,
                            mode="drop",
                        )
                        pred = edited["prediction"]

                        generation_rows.append({
                            "sid": sid,
                            "gt": DISPLAY[gt],
                            "layer": int(L),
                            "k": len(eligible_positions),
                            "condition": "drop_all_text",
                            "repeat": 0,
                            "prediction": DISPLAY.get(pred, pred),
                            "correct": pred == gt,
                            "text": edited["text"],
                            "n_eligible": len(eligible_positions),
                            "n_selected": len(eligible_positions),
                            "selected_fraction": 1.0 if eligible_positions else 0.0,
                            "n_suppressed": edited["n_suppressed"],
                            "selected_positive_fraction": float("nan"),
                            "selected_mean_score": float("nan"),
                        })

                    for k in ks:
                        top = ranked[: int(k)]
                        top_pos = [int(r["position"]) for r in top]
                        top_scores = [
                            float(r["ranking_score"]) for r in top
                        ]

                        top_pos_frac = (
                            float(np.mean(np.asarray(top_scores) > 0))
                            if top_scores else float("nan")
                        )
                        top_mean = safe_mean(top_scores)
                        selected_fraction = (
                            len(top_pos) / len(eligible_positions)
                            if eligible_positions else float("nan")
                        )

                        for cond in [c for c in conditions if c in {"drop_top", "keep_top"}]:
                            mode = "drop" if cond == "drop_top" else "keep"
                            edited = generate_condition(
                                model=model,
                                processor=processor,
                                decoder_layers=decoder_layers,
                                batch=rb,
                                max_new_tokens=a.max_new_tokens,
                                layer=L,
                                eligible_positions=eligible_positions,
                                selected_positions=top_pos,
                                mode=mode,
                            )
                            pred = edited["prediction"]

                            generation_rows.append({
                                "sid": sid,
                                "gt": DISPLAY[gt],
                                "layer": int(L),
                                "k": int(k),
                                "condition": cond,
                                "repeat": 0,
                                "prediction": DISPLAY.get(pred, pred),
                                "correct": pred == gt,
                                "text": edited["text"],
                                "n_eligible": len(eligible_positions),
                                "n_selected": len(top_pos),
                                "selected_fraction": selected_fraction,
                                "n_suppressed": edited["n_suppressed"],
                                "selected_positive_fraction": top_pos_frac,
                                "selected_mean_score": top_mean,
                            })

                        if any(
                            c in conditions
                            for c in {"drop_random", "keep_random"}
                        ):
                            for rep in range(int(a.random_repeats)):
                                rnd = category_matched_random(
                                    all_rows=all_rows,
                                    top_rows=top,
                                    k=int(k),
                                    seed=(
                                        a.seed * 1000003
                                        + sid * 1009
                                        + int(L) * 97
                                        + int(k) * 17
                                        + rep
                                    ),
                                )
                                rnd_pos = [int(r["position"]) for r in rnd]
                                rnd_scores = [
                                    ranking_value(r, a.rank_by) for r in rnd
                                ]
                                rnd_pos_frac = (
                                    float(np.mean(np.asarray(rnd_scores) > 0))
                                    if rnd_scores else float("nan")
                                )
                                rnd_mean = safe_mean(rnd_scores)
                                rnd_frac = (
                                    len(rnd_pos) / len(eligible_positions)
                                    if eligible_positions else float("nan")
                                )

                                for cond in [
                                    c for c in conditions
                                    if c in {"drop_random", "keep_random"}
                                ]:
                                    mode = (
                                        "drop"
                                        if cond == "drop_random"
                                        else "keep"
                                    )

                                    edited = generate_condition(
                                        model=model,
                                        processor=processor,
                                        decoder_layers=decoder_layers,
                                        batch=rb,
                                        max_new_tokens=a.max_new_tokens,
                                        layer=L,
                                        eligible_positions=eligible_positions,
                                        selected_positions=rnd_pos,
                                        mode=mode,
                                    )
                                    pred = edited["prediction"]

                                    generation_rows.append({
                                        "sid": sid,
                                        "gt": DISPLAY[gt],
                                        "layer": int(L),
                                        "k": int(k),
                                        "condition": cond,
                                        "repeat": int(rep),
                                        "prediction": DISPLAY.get(pred, pred),
                                        "correct": pred == gt,
                                        "text": edited["text"],
                                        "n_eligible": len(eligible_positions),
                                        "n_selected": len(rnd_pos),
                                        "selected_fraction": rnd_frac,
                                        "n_suppressed": edited["n_suppressed"],
                                        "selected_positive_fraction": rnd_pos_frac,
                                        "selected_mean_score": rnd_mean,
                                    })

                del rb, gb, real_states, gray_states, grad_by_layer

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}",
                    flush=True,
                )

            finally:
                if cap is not None:
                    with contextlib.suppress(Exception):
                        cap.close()
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Save raw outputs
        # -------------------------------------------------------------
        score_df = pd.DataFrame(token_score_rows)
        count_df = pd.DataFrame(eligible_count_rows)
        gen_df = pd.DataFrame(generation_rows)

        score_df.to_csv(outdir / "per_token_scores.csv", index=False)
        count_df.to_csv(outdir / "eligible_counts.csv", index=False)
        gen_df.to_csv(outdir / "generation_per_sample.csv", index=False)

        summary_df = summarize_generation(gen_df)
        by_status_df = summarize_by_baseline_status(gen_df)
        sel_summary_df = summarize_selection(score_df, a.rank_by, ks)

        summary_df.to_csv(outdir / "generation_summary.csv", index=False)
        by_status_df.to_csv(
            outdir / "generation_by_baseline_status.csv", index=False
        )
        sel_summary_df.to_csv(
            outdir / "selection_summary.csv", index=False
        )

        # -------------------------------------------------------------
        # Human-readable report
        # -------------------------------------------------------------
        lines = []
        lines.append("=" * 132)
        lines.append("SINGLE-LAYER CAUSAL-TOKEN DROP / KEEP SUMMARY")
        lines.append("=" * 132)

        clean_rows = gen_df[gen_df["condition"] == "clean"].drop_duplicates("sid")
        clean_acc = (
            float(clean_rows["correct"].mean())
            if len(clean_rows) else float("nan")
        )
        lines.append(f"Clean generation accuracy: {clean_acc:.4f}")
        lines.append("")

        if not count_df.empty:
            for L, g in count_df.groupby("layer"):
                lines.append(
                    f"L{int(L)} eligible text tokens: "
                    f"mean={g['n_eligible_text'].mean():.2f}, "
                    f"median={g['n_eligible_text'].median():.1f}, "
                    f"positive-score mean={g['n_positive_score'].mean():.2f}"
                )
            lines.append("")

        if not summary_df.empty:
            show = summary_df[
                ~summary_df["condition"].str.contains("_mean$", regex=True)
            ].copy()
            for L in layers:
                lines.append(f"[L{L}]")
                zz = show[show["layer"] == L].copy()
                if zz.empty:
                    lines.append("  no results")
                    continue

                # Print Top conditions first; compact.
                for k in ks:
                    z = zz[zz["k"] == k]
                    if z.empty:
                        continue
                    chunks = []
                    for cond in ("drop_top", "keep_top", "drop_random", "keep_random"):
                        q = z[z["condition"] == cond]
                        if q.empty:
                            continue
                        r = q.iloc[0]
                        chunks.append(
                            f"{cond}:acc={r.edited_acc:.4f},"
                            f"gain={r.gain:+.4f},"
                            f"W2C/C2W={int(r.W2C)}/{int(r.C2W)},"
                            f"sel={r.mean_n_selected:.1f}/"
                            f"{r.mean_n_eligible:.1f}"
                        )
                    if chunks:
                        lines.append(f"  K={k:>2d} | " + " | ".join(chunks))
                lines.append("")

        lines.append(
            "Interpretation:"
        )
        lines.append(
            "  drop_top << clean and drop_top < drop_random supports necessity "
            "of the ranked token updates at that layer."
        )
        lines.append(
            "  keep_top >> keep_random, especially at small/moderate K, supports "
            "that the ranking captures a disproportionately important route."
        )
        lines.append(
            "  Do not call K 'sparse' without reporting K / eligible-text count."
        )
        lines.append(
            "  keep/drop manipulate ONLY the tested block update, not the whole "
            "token hidden state."
        )

        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(
            report, encoding="utf-8"
        )

        write_json(
            outdir / "metadata.json",
            {
                "script": "eval_singlelayer_causal_token_dropkeep_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "layers": layers,
                "target_layers": targets,
                "writer_source": writer_source,
                "writer_mode": a.writer_mode,
                "rank_by": a.rank_by,
                "positive_only": bool(a.positive_only),
                "score_definition_mediation": (
                    "(h_real[L,p]-h_gray[L,p])^T grad_h J_GT_writer"
                ),
                "score_definition_real_update": (
                    "(h_real[L,p]-h_real[L-1,p])^T grad_h J_GT_writer"
                ),
                "intervention": "h_L[p] <- h_{L-1}[p]",
                "eligible_categories": eligible_categories,
                "visual_tokens_edited": False,
                "prompt_last_edited": False,
                "ks": ks,
                "conditions": conditions,
                "random_repeats": int(a.random_repeats),
                "eval_scope": a.eval_scope,
                "eval_N_requested": len(test),
                "writer_calibration_N": len(train),
                "writer_calibration_eval_overlap": overlap,
                "oracle_note": (
                    "GT selects the relation-specific late writer used for token ranking."
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
