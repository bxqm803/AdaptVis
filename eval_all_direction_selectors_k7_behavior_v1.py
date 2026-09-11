#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_all_direction_selectors_k7_behavior_v1.py

Behavioral evaluation of ALL previously computed Direction-Head selectors.

Key idea
--------
You already have, for each sample, four strict writer-guided candidate K7 sets:

    K7^left, K7^right, K7^on, K7^under

and a Direction-Head selector file containing many GT-free predictions:

    pred_single_L23H01
    pred_top3_equal
    pred_top10_equal
    ...

This script runs the expensive intervention ONLY ONCE PER CANDIDATE RELATION
(max 4 edited generations / sample), caches those four outputs, and then
evaluates ALL Direction-Head selector variants offline.

Therefore one run gives:
  1) stored baseline generation
  2) oracle GT strict-K7 intervention
  3) every Direction-Head selector -> selected strict-K7 intervention
  4) baseline-relation K7 intervention
  5) random-candidate K7 control (many random policies, no extra generation)
  6) full 4x candidate intervention response table

Intervention
------------
For selected exact state (L,p):

    h_edit[L,p] = h_real[L,p] + alpha * (h_real[L,p] - h_gray[L,p])

Default alpha=1, so:

    h_edit = 2*h_real - h_gray

Only selected exact (layer, token) states are edited during generation prefill.

Important
---------
- This tests STRICT fixed Top-K from selected_topk_all_directions.csv.
- It does NOT test Core50 unless K happens to match it.
- Candidate relation selection is separated from intervention:
  all four candidate outputs are generated once, then arbitrary selector
  policies can be scored without rerunning the model.
- Existing per-sample chunks make the run resumable.

Typical command
---------------
CUDA_VISIBLE_DEVICES=0 python -u eval_all_direction_selectors_k7_behavior_v1.py \
  --fourway-dir output/qwen3b_four_writer_k7_N80 \
  --direction-selector-dir output/qwen3b_direction_head_selector_N80 \
  --model qwen-3b \
  --k 7 \
  --alpha 1.0 \
  --output-dir output/qwen3b_direction_selectors_K7_behavior_N80 \
  --overwrite

If interrupted, rerun WITHOUT --overwrite to resume.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "on", "under")
EPS = 1e-12


# =============================================================================
# Args / generic utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fourway-dir", required=True)
    p.add_argument("--direction-selector-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--random-repeats",
        type=int,
        default=1000,
        help="Random selector policies are evaluated offline after all 4 candidates are generated.",
    )
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="0 = use all samples shared by fourway and direction-selector detail.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return sorted({
        int(x.strip().upper().replace("L", ""))
        for x in str(s).split(",") if x.strip()
    })


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    mp = {
        "left": "left", "left_of": "left", "left of": "left", "l": "left",
        "right": "right", "right_of": "right", "right of": "right", "r": "right",
        "above": "on", "on": "on", "over": "on", "top": "on",
        "below": "under", "under": "under", "beneath": "under", "bottom": "under",
    }
    return mp.get(s, s)


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def make_gray_image(img, value):
    v = int(np.clip(value, 0, 255))
    return Image.new("RGB", img.size, (v, v, v))


def safe_div(a, b):
    return float(a / b) if float(b) != 0.0 else np.nan


def safe_mean(x):
    a = pd.to_numeric(pd.Series(list(x)), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def first_tensor(out):
    return traj.first_tensor(out)


def replace_first_tensor(out, y):
    return traj.replace_first_tensor(out, y)


# =============================================================================
# Capture natural Real-Gray state deltas
# =============================================================================

class Capture:
    def __init__(self, decoder_layers, layers):
        self.states = {}
        self.handles = []
        for L in layers:
            self.handles.append(
                decoder_layers[L].register_forward_hook(self._hook(L))
            )

    def _hook(self, L):
        def hook(_m, _inp, out):
            h = first_tensor(out)
            self.states[int(L)] = h.detach().float().cpu()
            return out
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_cpu(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing captured layers: {missing}")
        return {
            int(L): cap.states[int(L)].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


def delta_at(hreal, hgray, L, pos):
    if L not in hreal or L not in hgray:
        return None
    R = hreal[L]
    G = hgray[L]
    if R.ndim == 3:
        R = R[0]
    if G.ndim == 3:
        G = G[0]
    p = int(pos)
    if not (0 <= p < min(R.shape[0], G.shape[0])):
        return None
    return (R[p] - G[p]).astype(np.float32)


# =============================================================================
# Exact-state prefill editor
# =============================================================================

class TokenDeltaEditor:
    def __init__(self, decoder_layers, specs, alpha, prompt_len):
        self.handles = []
        self.applied = defaultdict(int)
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)

        for L, entries in specs.items():
            if not entries:
                continue
            self.handles.append(
                decoder_layers[int(L)].register_forward_hook(
                    self._hook(int(L), entries)
                )
            )

    def _hook(self, L, entries):
        def hook(_m, _inp, out):
            h = first_tensor(out)

            # Apply ONLY during full prompt prefill, not token-by-token decoding.
            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()
            for pos, delta_np in entries:
                p = int(pos)
                if not (0 <= p < y.shape[1]):
                    continue
                delta = torch.as_tensor(
                    delta_np,
                    device=y.device,
                    dtype=torch.float32,
                )
                y[:, p, :] = y[:, p, :] + self.alpha * delta
                self.applied[L] += 1

            return replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def generate_edited(
    model,
    processor,
    decoder_layers,
    real_batch,
    specs,
    alpha,
    max_new_tokens,
):
    prompt_len = int(real_batch["input_ids"].shape[1])
    editor = TokenDeltaEditor(
        decoder_layers=decoder_layers,
        specs=specs,
        alpha=alpha,
        prompt_len=prompt_len,
    )
    try:
        text = base.generate_text(
            model,
            processor,
            real_batch,
            max_new_tokens=max_new_tokens,
        )
        pred = canon_rel(traj.normalize_relation(base, text))
        return {
            "prediction": pred,
            "text": text,
            "n_edit_pairs": int(sum(editor.applied.values())),
        }
    finally:
        editor.close()


def build_specs(chosen_rows, hreal, hgray):
    specs = defaultdict(list)
    export = []

    chosen = chosen_rows.sort_values("rank")
    for r in chosen.itertuples(index=False):
        L = int(r.source_layer)
        p = int(r.position)
        d = delta_at(hreal, hgray, L, p)
        if d is None:
            continue
        specs[L].append((p, d))
        export.append({
            "rank": int(r.rank),
            "source_layer": L,
            "position": p,
            "selection_score": float(r.selection_score)
                if hasattr(r, "selection_score") else np.nan,
        })
    return dict(specs), export


# =============================================================================
# Evaluation metrics
# =============================================================================

def policy_metrics(df, policy_name):
    bcorrect = df["baseline_correct"].to_numpy(bool)
    edited_correct = df["correct"].to_numpy(bool)
    baseline_pred = df["baseline_prediction"].to_numpy(object)
    edited_pred = df["prediction"].to_numpy(object)
    gt = df["gt"].to_numpy(object)
    cand = df["selected_candidate"].to_numpy(object)

    wrong = ~bcorrect
    correct = bcorrect

    w2c = int(np.sum(wrong & edited_correct))
    c2w = int(np.sum(correct & (~edited_correct)))
    changed = int(np.sum(edited_pred != baseline_pred))

    candidate_acc = float(np.mean(cand == gt))
    target_follow = float(np.mean(edited_pred == cand))

    return {
        "policy": policy_name,
        "N": len(df),
        "candidate_acc": candidate_acc,
        "behavior_acc": float(np.mean(edited_correct)),
        "gain_vs_baseline": float(np.mean(edited_correct) - np.mean(bcorrect)),
        "W2C": w2c,
        "C2W": c2w,
        "net": w2c - c2w,
        "repair_rate": safe_div(w2c, int(wrong.sum())),
        "preserve_rate": safe_div(int(np.sum(correct & edited_correct)), int(correct.sum())),
        "changed": changed,
        "target_follow": target_follow,
        "behavior_acc_when_candidate_correct": (
            float(np.mean(edited_correct[cand == gt])) if np.any(cand == gt) else np.nan
        ),
        "behavior_acc_when_candidate_wrong": (
            float(np.mean(edited_correct[cand != gt])) if np.any(cand != gt) else np.nan
        ),
        "mean_edit_pairs": safe_mean(df["n_edit_pairs"]),
    }


def baseline_metrics(base_df):
    return {
        "policy": "BASELINE_GENERATION",
        "N": len(base_df),
        "candidate_acc": np.nan,
        "behavior_acc": float(base_df["baseline_correct"].mean()),
        "gain_vs_baseline": 0.0,
        "W2C": 0,
        "C2W": 0,
        "net": 0,
        "repair_rate": 0.0,
        "preserve_rate": 1.0,
        "changed": 0,
        "target_follow": np.nan,
        "behavior_acc_when_candidate_correct": np.nan,
        "behavior_acc_when_candidate_wrong": np.nan,
        "mean_edit_pairs": 0.0,
    }


def random_policy_distribution(candidate_outputs, baseline_by_sid, repeats, seed):
    """
    No model calls. Each repeat independently picks one of four candidate
    relations per sample, then reads the already-generated candidate output.
    """
    rng = np.random.default_rng(seed)
    sids = sorted(candidate_outputs["sid"].unique().tolist())
    by_key = {
        (int(r.sid), str(r.candidate_relation)): r
        for r in candidate_outputs.itertuples(index=False)
    }

    rows = []
    for rep in range(int(repeats)):
        sim = []
        for sid in sids:
            rel = REL[int(rng.integers(0, len(REL)))]
            r = by_key[(int(sid), rel)]
            b = baseline_by_sid[int(sid)]
            sim.append({
                "sid": sid,
                "gt": b["gt"],
                "baseline_prediction": b["baseline_prediction"],
                "baseline_correct": b["baseline_correct"],
                "selected_candidate": rel,
                "prediction": str(r.prediction),
                "correct": str(r.prediction) == b["gt"],
                "n_edit_pairs": int(r.n_edit_pairs),
            })
        m = policy_metrics(pd.DataFrame(sim), f"random_rep_{rep}")
        m["repeat"] = rep
        rows.append(m)

    dist = pd.DataFrame(rows)
    summary = {
        "policy": f"RANDOM_CANDIDATE_MEAN_{repeats}x",
        "N": len(sids),
        "candidate_acc": safe_mean(dist["candidate_acc"]),
        "behavior_acc": safe_mean(dist["behavior_acc"]),
        "gain_vs_baseline": safe_mean(dist["gain_vs_baseline"]),
        "W2C": safe_mean(dist["W2C"]),
        "C2W": safe_mean(dist["C2W"]),
        "net": safe_mean(dist["net"]),
        "repair_rate": safe_mean(dist["repair_rate"]),
        "preserve_rate": safe_mean(dist["preserve_rate"]),
        "changed": safe_mean(dist["changed"]),
        "target_follow": safe_mean(dist["target_follow"]),
        "behavior_acc_when_candidate_correct": safe_mean(
            dist["behavior_acc_when_candidate_correct"]
        ),
        "behavior_acc_when_candidate_wrong": safe_mean(
            dist["behavior_acc_when_candidate_wrong"]
        ),
        "mean_edit_pairs": safe_mean(dist["mean_edit_pairs"]),
    }
    return dist, summary


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    outdir = Path(a.output_dir)

    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dir = outdir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    four = Path(a.fourway_dir)
    dsel_dir = Path(a.direction_selector_dir)

    selected_path = four / "selected_topk_all_directions.csv"
    sample_path = four / "per_sample_direction_scores.csv"
    selector_detail_path = dsel_dir / "direction_head_selector_detail.csv"
    selector_summary_path = dsel_dir / "direction_head_selector_summary.csv"

    for p in (selected_path, sample_path, selector_detail_path):
        if not p.exists():
            raise FileNotFoundError(p)

    selected = pd.read_csv(selected_path)
    samples = pd.read_csv(sample_path)
    selector_detail = pd.read_csv(selector_detail_path)

    selected["sid"] = pd.to_numeric(selected["sid"], errors="raise").astype(int)
    selected["rank"] = pd.to_numeric(selected["rank"], errors="raise").astype(int)
    selected["source_layer"] = pd.to_numeric(
        selected["source_layer"], errors="raise"
    ).astype(int)
    selected["position"] = pd.to_numeric(
        selected["position"], errors="raise"
    ).astype(int)
    selected["candidate_relation"] = selected["candidate_relation"].map(canon_rel)

    if "selection_type" in selected.columns:
        selected = selected[selected["selection_type"] == "raw"].copy()
    selected = selected[selected["rank"] <= int(a.k)].copy()

    samples["sid"] = pd.to_numeric(samples["sid"], errors="raise").astype(int)
    samples["gt"] = samples["gt"].map(canon_rel)
    samples["baseline_prediction"] = samples["baseline_prediction"].map(canon_rel)
    samples["baseline_correct"] = samples["baseline_correct"].map(boolify)

    selector_detail["sid"] = pd.to_numeric(
        selector_detail["sid"], errors="raise"
    ).astype(int)
    selector_detail["gt"] = selector_detail["gt"].map(canon_rel)
    selector_detail["baseline_prediction"] = selector_detail[
        "baseline_prediction"
    ].map(canon_rel)
    selector_detail["baseline_correct"] = selector_detail[
        "baseline_correct"
    ].map(boolify)

    pred_cols = [
        c for c in selector_detail.columns
        if c.startswith("pred_")
    ]
    if not pred_cols:
        raise RuntimeError(
            f"No pred_* columns in {selector_detail_path}"
        )
    for c in pred_cols:
        selector_detail[c] = selector_detail[c].map(canon_rel)

    valid_sids = sorted(
        set(samples["sid"])
        & set(selector_detail["sid"])
        & set(selected["sid"])
    )
    if a.max_eval_samples and a.max_eval_samples > 0:
        valid_sids = valid_sids[: int(a.max_eval_samples)]

    samples = samples[samples["sid"].isin(valid_sids)].copy()
    selector_detail = selector_detail[
        selector_detail["sid"].isin(valid_sids)
    ].copy()

    # Require all four candidate K sets for each sample.
    bad = []
    for sid in valid_sids:
        g = selected[selected["sid"] == sid]
        have = set(g["candidate_relation"].unique())
        missing = set(REL) - have
        if missing:
            bad.append((sid, sorted(missing)))
    if bad:
        raise RuntimeError(
            f"Missing candidate K sets for {len(bad)} samples; first: {bad[:5]}"
        )

    baseline_by_sid = {}
    for r in samples.itertuples(index=False):
        baseline_by_sid[int(r.sid)] = {
            "gt": canon_rel(r.gt),
            "baseline_prediction": canon_rel(r.baseline_prediction),
            "baseline_correct": bool(r.baseline_correct),
        }

    selector_by_sid = selector_detail.set_index("sid")

    # -------------------------------------------------------------------------
    # Load repo-native data/model.
    # -------------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _ = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    missing = [
        sid for sid in valid_sids
        if sid not in prompts or sid not in rec_by_sid
    ]
    if missing:
        raise RuntimeError(
            f"Missing prompt/record for SIDs: {missing[:20]}"
        )

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    model_cls = getattr(transformers, spec.model_class)

    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    print(f"Loading {spec.repo_id}", flush=True)
    model = model_cls.from_pretrained(spec.repo_id, **load_kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)

    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)

    print("=" * 160)
    print("ALL DIRECTION-HEAD SELECTORS -> STRICT K BEHAVIOR")
    print("=" * 160)
    print(f"N={len(valid_sids)} | K={a.k} | alpha={a.alpha}")
    print(f"source layers={source_layers}")
    print(f"direction selector variants={len(pred_cols)}")
    print("candidate generations/sample = 4")
    print("baseline is reused from four-way tracing output.")
    print(f"decoder={decoder_path}")
    print()

    all_candidate_rows = []
    all_selected_state_rows = []
    errors = []

    try:
        for sid in tqdm(valid_sids, desc="4-candidate K behavior"):
            sid = int(sid)
            cache = chunk_dir / f"sid_{sid}.pkl.gz"

            if cache.exists():
                obj = pd.read_pickle(cache, compression="gzip")
                all_candidate_rows.extend(obj.get("candidate_rows", []))
                all_selected_state_rows.extend(obj.get("selected_states", []))
                continue

            real = gray = rb = gb = None
            sid_candidate_rows = []
            sid_selected_states = []

            try:
                b = baseline_by_sid[sid]
                gt = b["gt"]

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                device = torch.device(a.device)
                question_text = str(prompts[sid]["question_text"])

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=question_text,
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=question_text,
                    device=device,
                )

                # Real and Gray prompts must align token-wise.
                ids_r = rb["input_ids"][0].detach().cpu().tolist()
                ids_g = gb["input_ids"][0].detach().cpu().tolist()
                if ids_r != ids_g:
                    raise RuntimeError(
                        "Real/Gray tokenization mismatch; cannot apply aligned state delta."
                    )

                hreal = capture_cpu(
                    model, decoder_layers, rb, source_layers
                )
                hgray = capture_cpu(
                    model, decoder_layers, gb, source_layers
                )

                # Generate all 4 candidate K interventions exactly once.
                for cand in REL:
                    chosen = selected[
                        (selected["sid"] == sid)
                        & (selected["candidate_relation"] == cand)
                    ].sort_values("rank").head(int(a.k))

                    if len(chosen) == 0:
                        raise RuntimeError(
                            f"sid={sid} candidate={cand}: no selected states"
                        )

                    specs_edit, exported = build_specs(
                        chosen, hreal, hgray
                    )
                    out = generate_edited(
                        model=model,
                        processor=processor,
                        decoder_layers=decoder_layers,
                        real_batch=rb,
                        specs=specs_edit,
                        alpha=a.alpha,
                        max_new_tokens=a.max_new_tokens,
                    )

                    sid_candidate_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "baseline_prediction": b["baseline_prediction"],
                        "baseline_correct": b["baseline_correct"],
                        "candidate_relation": cand,
                        "candidate_is_gt": cand == gt,
                        "prediction": out["prediction"],
                        "correct": out["prediction"] == gt,
                        "target_follow": out["prediction"] == cand,
                        "n_selected_states": len(chosen),
                        "n_edit_pairs": out["n_edit_pairs"],
                        "text": out["text"],
                    })

                    for e in exported:
                        sid_selected_states.append({
                            "sid": sid,
                            "candidate_relation": cand,
                            **e,
                        })

                pd.to_pickle(
                    {
                        "candidate_rows": sid_candidate_rows,
                        "selected_states": sid_selected_states,
                    },
                    cache,
                    compression="gzip",
                )

                all_candidate_rows.extend(sid_candidate_rows)
                all_selected_state_rows.extend(sid_selected_states)

            except Exception as e:
                import traceback
                errors.append({
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback_tail": traceback.format_exc().splitlines()[-15:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(e).__name__}: {e}"
                )

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()

                rb = None
                gb = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidate_df = pd.DataFrame(all_candidate_rows)
    selected_state_df = pd.DataFrame(all_selected_state_rows)

    if not len(candidate_df):
        raise RuntimeError(
            "No candidate intervention outputs produced. Check errors.json."
        )

    success_sids = sorted(candidate_df["sid"].unique().tolist())
    samples_success = samples[samples["sid"].isin(success_sids)].copy()

    # Require complete four-way behavior matrix.
    counts = candidate_df.groupby("sid")["candidate_relation"].nunique()
    incomplete = counts[counts < 4]
    if len(incomplete):
        print(
            f"[WARN] excluding {len(incomplete)} incomplete SIDs from policy evaluation"
        )
        keep = sorted(counts[counts == 4].index.astype(int).tolist())
        candidate_df = candidate_df[candidate_df["sid"].isin(keep)].copy()
        samples_success = samples_success[samples_success["sid"].isin(keep)].copy()
        success_sids = keep

    # -------------------------------------------------------------------------
    # Build lookup from (sid, candidate relation) -> actual edited generation.
    # -------------------------------------------------------------------------
    cand_lookup = {}
    for r in candidate_df.itertuples(index=False):
        cand_lookup[(int(r.sid), str(r.candidate_relation))] = {
            "prediction": str(r.prediction),
            "correct": bool(r.correct),
            "n_edit_pairs": int(r.n_edit_pairs),
            "target_follow": bool(r.target_follow),
        }

    def evaluate_policy(policy_name, selected_relation_by_sid):
        rows = []
        for sid in success_sids:
            sid = int(sid)
            b = baseline_by_sid[sid]
            cand = canon_rel(selected_relation_by_sid[sid])
            o = cand_lookup[(sid, cand)]
            rows.append({
                "sid": sid,
                "gt": b["gt"],
                "baseline_prediction": b["baseline_prediction"],
                "baseline_correct": b["baseline_correct"],
                "selected_candidate": cand,
                "prediction": o["prediction"],
                "correct": o["correct"],
                "n_edit_pairs": o["n_edit_pairs"],
                "target_follow": o["target_follow"],
            })
        df = pd.DataFrame(rows)
        return df, policy_metrics(df, policy_name)

    policy_details = []
    summary_rows = [baseline_metrics(samples_success)]

    # Oracle GT strict K.
    oracle_rel = {
        sid: baseline_by_sid[sid]["gt"]
        for sid in success_sids
    }
    df, met = evaluate_policy("ORACLE_GT_STRICT_K", oracle_rel)
    policy_details.append(df.assign(policy="ORACLE_GT_STRICT_K"))
    summary_rows.append(met)

    # Reinforce the model's own baseline relation.
    base_rel = {
        sid: baseline_by_sid[sid]["baseline_prediction"]
        for sid in success_sids
    }
    df, met = evaluate_policy("BASELINE_RELATION_STRICT_K", base_rel)
    policy_details.append(df.assign(policy="BASELINE_RELATION_STRICT_K"))
    summary_rows.append(met)

    # Every Direction-Head selector prediction column.
    for c in pred_cols:
        policy_name = c[len("pred_"):]
        mapping = {}
        for sid in success_sids:
            if sid not in selector_by_sid.index:
                continue
            val = selector_by_sid.loc[sid, c]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            mapping[int(sid)] = canon_rel(val)

        if len(mapping) != len(success_sids):
            continue
        if any(mapping[sid] not in REL for sid in success_sids):
            continue

        df, met = evaluate_policy(
            f"DH_{policy_name}",
            mapping,
        )
        policy_details.append(
            df.assign(policy=f"DH_{policy_name}")
        )
        summary_rows.append(met)

    # Random control from already-generated four candidates.
    random_dist, random_summary = random_policy_distribution(
        candidate_outputs=candidate_df,
        baseline_by_sid=baseline_by_sid,
        repeats=a.random_repeats,
        seed=a.seed + 991,
    )
    summary_rows.append(random_summary)

    summary_df = pd.DataFrame(summary_rows)

    # Sort with baseline/oracle first, then behavior accuracy.
    special_order = {
        "BASELINE_GENERATION": 0,
        "ORACLE_GT_STRICT_K": 1,
        "BASELINE_RELATION_STRICT_K": 2,
    }
    summary_df["_special"] = summary_df["policy"].map(
        lambda x: special_order.get(str(x), 10)
    )
    summary_df = summary_df.sort_values(
        ["_special", "behavior_acc"],
        ascending=[True, False],
    ).drop(columns=["_special"])

    detail_df = (
        pd.concat(policy_details, ignore_index=True)
        if policy_details else pd.DataFrame()
    )

    # -------------------------------------------------------------------------
    # Candidate intervention matrix diagnostics.
    # -------------------------------------------------------------------------
    candidate_summary = (
        candidate_df.groupby("candidate_relation")
        .agg(
            N=("sid", "nunique"),
            behavior_acc=("correct", "mean"),
            target_follow=("target_follow", "mean"),
            mean_edit_pairs=("n_edit_pairs", "mean"),
        )
        .reset_index()
    )

    # How does each candidate intervention behave conditional on candidate truth?
    candidate_truth_summary = (
        candidate_df.groupby(["candidate_relation", "candidate_is_gt"])
        .agg(
            N=("sid", "nunique"),
            behavior_acc=("correct", "mean"),
            target_follow=("target_follow", "mean"),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    candidate_df.to_csv(
        outdir / "all_four_candidate_K_behavior.csv",
        index=False,
    )
    candidate_summary.to_csv(
        outdir / "candidate_intervention_summary.csv",
        index=False,
    )
    candidate_truth_summary.to_csv(
        outdir / "candidate_intervention_by_truth.csv",
        index=False,
    )
    selected_state_df.to_csv(
        outdir / "selected_exact_states_used.csv",
        index=False,
    )
    summary_df.to_csv(
        outdir / "all_selector_behavior_summary.csv",
        index=False,
    )
    detail_df.to_csv(
        outdir / "all_selector_behavior_detail.csv",
        index=False,
    )
    random_dist.to_csv(
        outdir / "random_candidate_policy_distribution.csv",
        index=False,
    )
    (outdir / "errors.json").write_text(
        json.dumps(errors, indent=2),
        encoding="utf-8",
    )

    # Include original direction-selector classification summary for reference.
    if selector_summary_path.exists():
        orig = pd.read_csv(selector_summary_path)
        orig.to_csv(
            outdir / "original_direction_selector_summary.csv",
            index=False,
        )

    metadata = {
        "N_success": len(success_sids),
        "K": int(a.k),
        "alpha": float(a.alpha),
        "source_layers": source_layers,
        "random_repeats": int(a.random_repeats),
        "direction_selector_variants": [
            c[len("pred_"):] for c in pred_cols
        ],
        "intervention": (
            "h_edit[L,p] = h_real[L,p] + alpha*(h_real[L,p]-h_gray[L,p]), "
            "exact selected layer/token states, prefill only"
        ),
        "candidate_generation_strategy": (
            "all four candidate K sets generated once per sample; all selector "
            "policies evaluated offline from the same four outputs"
        ),
        "baseline_source": str(sample_path),
        "selected_k_source": str(selected_path),
        "direction_selector_source": str(selector_detail_path),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------
    print()
    print("=" * 182)
    print("STRICT-K BEHAVIOR — ALL DIRECTION-HEAD SELECTORS")
    print("=" * 182)
    print(
        f"N={len(success_sids)} | baseline={samples_success['baseline_correct'].mean():.4f} "
        f"| K={a.k} | alpha={a.alpha}"
    )
    print()

    cols = [
        "policy", "candidate_acc", "behavior_acc", "gain_vs_baseline",
        "W2C", "C2W", "net", "repair_rate", "preserve_rate",
        "target_follow",
    ]

    # Print baseline/oracle + top 20 non-special by behavioral accuracy.
    head = summary_df[
        summary_df["policy"].isin([
            "BASELINE_GENERATION",
            "ORACLE_GT_STRICT_K",
            "BASELINE_RELATION_STRICT_K",
        ])
    ]
    rest = summary_df[
        ~summary_df["policy"].isin([
            "BASELINE_GENERATION",
            "ORACLE_GT_STRICT_K",
            "BASELINE_RELATION_STRICT_K",
        ])
    ].sort_values(
        ["behavior_acc", "candidate_acc"],
        ascending=False,
    ).head(25)
    shown = pd.concat([head, rest], ignore_index=True)

    print(
        f"{'policy':<48s} {'selAcc':>7s} {'behAcc':>7s} {'gain':>7s} "
        f"{'W2C':>5s} {'C2W':>5s} {'net':>5s} "
        f"{'repair':>7s} {'pres':>7s} {'follow':>7s}"
    )
    print("-" * 182)

    for r in shown.itertuples(index=False):
        ca = getattr(r, "candidate_acc")
        tf = getattr(r, "target_follow")
        print(
            f"{str(r.policy):<48s} "
            f"{ca:7.3f} " if np.isfinite(ca) else f"{str(r.policy):<48s} {'-':>7s} ",
            end=""
        )
        print(
            f"{float(r.behavior_acc):7.3f} "
            f"{float(r.gain_vs_baseline):+7.3f} "
            f"{float(r.W2C):5.1f} "
            f"{float(r.C2W):5.1f} "
            f"{float(r.net):5.1f} "
            f"{float(r.repair_rate):7.3f} "
            f"{float(r.preserve_rate):7.3f} "
            + (f"{tf:7.3f}" if np.isfinite(tf) else f"{'-':>7s}")
        )

    # Best Direction-Head policy only.
    dh = summary_df[
        summary_df["policy"].astype(str).str.startswith("DH_")
    ].sort_values(
        ["behavior_acc", "candidate_acc"],
        ascending=False,
    )

    if len(dh):
        b = dh.iloc[0]
        print()
        print("BEST DIRECTION-HEAD SELECTOR AFTER ACTUAL K INTERVENTION")
        print("-" * 182)
        print(
            f"{b['policy']} | selector_acc={b['candidate_acc']:.3f} | "
            f"behavior={b['behavior_acc']:.3f} "
            f"({b['gain_vs_baseline']:+.3f}) | "
            f"W2C/C2W={int(b['W2C'])}/{int(b['C2W'])} | "
            f"repair={b['repair_rate']:.3f} preserve={b['preserve_rate']:.3f} | "
            f"targetFollow={b['target_follow']:.3f}"
        )

    print()
    oracle = summary_df[
        summary_df["policy"] == "ORACLE_GT_STRICT_K"
    ].iloc[0]
    print(
        "Oracle strict-K ceiling: "
        f"{oracle['behavior_acc']:.3f} "
        f"({oracle['gain_vs_baseline']:+.3f}), "
        f"W2C/C2W={int(oracle['W2C'])}/{int(oracle['C2W'])}"
    )
    print(
        "Random-candidate behavior mean: "
        f"{random_summary['behavior_acc']:.3f} "
        f"({random_summary['gain_vs_baseline']:+.3f})"
    )
    print()
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
