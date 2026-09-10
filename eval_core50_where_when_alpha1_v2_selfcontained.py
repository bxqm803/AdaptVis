#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Core50 WHERE-vs-WHEN behavioral diagnostic.

This file embeds all helper code from eval_self_selected_core_behavior_v1.py,
so no external eval_self_selected_core_behavior_v1.py is required.

alpha is fixed to 1.0.
"""

# ===== Embedded behavioral helper code =====
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
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


REL_DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
    "on": "on",
    "under": "under",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument(
        "--bundle",
        default="L20+L21+L22+L23+L24+L25+L26[global_unique]",
    )
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument("--oracle-k", type=int, default=36)
    p.add_argument("--core-masses", default="0.5,0.7,0.8")
    p.add_argument("--alphas", default="1.0")
    p.add_argument(
        "--diagnostic-alpha",
        type=float,
        default=1.0,
        help="Oracle diagnostic controls are run only at this alpha.",
    )
    p.add_argument(
        "--methods",
        default=(
            "attn_delta_max,"
            "attn_delta_top2,"
            "attn_delta_mean,"
            "attn_posdelta_max,"
            "attn_ensemble,"
            "attn75_prior25,"
            "attn_delta_top2_pm1split,"
            "random_pos_self_layer"
        ),
        help="Comma-separated deployable/relation-free methods.",
    )
    p.add_argument(
        "--diagnostics",
        default=(
            "oracle_core_exact,"
            "oracle_pos_self_layer,"
            "self_pos_oracle_layer"
        ),
        help="Comma-separated oracle diagnostics; set empty string to disable.",
    )
    p.add_argument(
        "--self-pos-diagnostic-method",
        default="attn_delta_top2",
        help="Self position selector used in self_pos_oracle_layer diagnostic.",
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_methods(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def infer_source_layers(bundle):
    raw = str(bundle).split("[", 1)[0]
    vals = []
    for x in raw.split("+"):
        x = x.strip().upper().replace("L", "")
        if x:
            vals.append(int(x))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError(f"Could not infer source layers from bundle={bundle!r}")
    return vals


def fmt_core(t):
    return f"core{int(round(100*t))}"


def canon_rel(x):
    x = str(x).strip().lower()
    return {
        "above": "on",
        "below": "under",
        "on": "on",
        "under": "under",
        "left": "left",
        "right": "right",
    }.get(x, x)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def percentile_rank_high(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(method="average", pct=True)


# =============================================================================
# Existing oracle run and core labels
# =============================================================================

def load_existing_run(a, source_layers):
    run_dir = Path(a.run_dir)

    base_path = run_dir / "baseline.csv"
    med_path = run_dir / "mediation_tokens.csv"
    sel_path = run_dir / "selected_tokens.csv"

    for p in [base_path, med_path, sel_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    baseline = pd.read_csv(base_path)
    med = pd.read_csv(med_path)
    sel = pd.read_csv(sel_path)

    for df in [baseline, med, sel]:
        df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)

    med["source_layer"] = pd.to_numeric(
        med["source_layer"], errors="raise"
    ).astype(int)
    med["position"] = pd.to_numeric(
        med["position"], errors="raise"
    ).astype(int)
    med["mediation"] = pd.to_numeric(
        med["mediation"], errors="coerce"
    )
    med["relation"] = med["relation"].map(canon_rel)

    sel["source_layer"] = pd.to_numeric(
        sel["source_layer"], errors="raise"
    ).astype(int)
    sel["position"] = pd.to_numeric(
        sel["position"], errors="raise"
    ).astype(int)
    sel["mediation"] = pd.to_numeric(
        sel["mediation"], errors="coerce"
    )
    sel["k"] = pd.to_numeric(sel["k"], errors="raise").astype(int)
    sel["relation"] = sel["relation"].map(canon_rel)

    filt = (
        (sel["condition"].astype(str) == str(a.condition))
        & (sel["selection_strategy"].astype(str) == str(a.selection_strategy))
        & (sel["source_bundle"].astype(str) == str(a.bundle))
        & (sel["k"] == int(a.oracle_k))
    )
    sel = sel[filt].copy()

    if not len(sel):
        avail = pd.read_csv(sel_path)[
            ["condition", "selection_strategy", "source_bundle", "k"]
        ].drop_duplicates()
        raise RuntimeError(
            "No rows matched requested oracle K36 configuration.\n"
            + avail.head(50).to_string(index=False)
        )

    valid_sids = sorted(sel["sid"].unique().tolist())

    baseline = baseline[baseline["sid"].isin(valid_sids)].copy()
    if "gt" in baseline.columns:
        baseline["gt"] = baseline["gt"].map(canon_rel)
    if "baseline_prediction" in baseline.columns:
        baseline["baseline_prediction"] = baseline["baseline_prediction"].map(canon_rel)
    if "baseline_correct" in baseline.columns:
        baseline["baseline_correct"] = baseline["baseline_correct"].astype(bool)

    med = med[
        med["sid"].isin(valid_sids)
        & med["source_layer"].isin(source_layers)
    ].copy()

    med = (
        med.sort_values(
            ["sid", "source_layer", "position", "mediation"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["sid", "source_layer", "position"], keep="first")
    )

    sel = (
        sel.sort_values(
            ["sid", "source_layer", "position", "mediation"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["sid", "source_layer", "position"], keep="first")
    )

    return baseline, med, sel, valid_sids


def build_core_truth(sel, core_masses):
    truth = {fmt_core(t): {} for t in core_masses}
    sizes = []

    for sid, g0 in sel.groupby("sid", sort=True):
        g = g0.sort_values("mediation", ascending=False).reset_index(drop=True)
        m = pd.to_numeric(g["mediation"], errors="coerce").fillna(0).to_numpy(float)
        pm = np.maximum(m, 0.0)
        total = float(pm.sum())
        cum = np.cumsum(pm) / total if total > 0 else np.zeros(len(pm))

        row = {"sid": int(sid), "relation": canon_rel(g["relation"].iloc[0])}
        for t in core_masses:
            name = fmt_core(t)
            hit = np.where(cum >= t)[0]
            kk = int(hit[0] + 1) if len(hit) else len(g)
            core = g.iloc[:kk]
            truth[name][int(sid)] = [
                {
                    "source_layer": int(r.source_layer),
                    "position": int(r.position),
                    "mediation": float(r.mediation),
                }
                for r in core.itertuples()
            ]
            row[f"{name}_N"] = kk
        sizes.append(row)

    return truth, pd.DataFrame(sizes)


# =============================================================================
# Feature table and self scores
# =============================================================================

def load_features(a, valid_sids, source_layers):
    p = Path(a.feature_dir) / "all_candidate_self_features.pkl.gz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\nRun eval_self_predict_causal_core_v2_selfcontained.py first."
        )

    df = pd.read_pickle(p, compression="gzip")
    need = {"sid", "source_layer", "position", "broad_category"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise RuntimeError("Feature cache missing columns: " + ", ".join(missing))

    df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)
    df["source_layer"] = pd.to_numeric(
        df["source_layer"], errors="raise"
    ).astype(int)
    df["position"] = pd.to_numeric(
        df["position"], errors="raise"
    ).astype(int)
    if "relation" in df.columns:
        df["relation"] = df["relation"].map(canon_rel)

    df = df[
        df["sid"].isin(valid_sids)
        & df["source_layer"].isin(source_layers)
    ].copy()

    # Exclude source last token if collector left one in.
    if "broad_category" in df.columns:
        df = df[df["broad_category"].astype(str) != "last"].copy()

    # Make sure the key rank/ensemble scores exist even if an older cache was used.
    if (
        "RANK_last_attn_next_delta_mean" not in df.columns
        and "last_attn_next_delta_mean" in df.columns
    ):
        df["RANK_last_attn_next_delta_mean"] = (
            df.groupby(["sid", "source_layer"], group_keys=False)[
                "last_attn_next_delta_mean"
            ].transform(percentile_rank_high)
        )

    if (
        "RANK_last_attn_next_max_positive_delta" not in df.columns
        and "last_attn_next_max_positive_delta" in df.columns
    ):
        df["RANK_last_attn_next_max_positive_delta"] = (
            df.groupby(["sid", "source_layer"], group_keys=False)[
                "last_attn_next_max_positive_delta"
            ].transform(percentile_rank_high)
        )

    if "SCORE_attn_ensemble" not in df.columns:
        raw = [
            "last_attn_next_max_positive_delta",
            "last_attn_next_delta_mean",
            "last_attn_next_real_mean",
            "last_attn_next_real_std_heads",
            "last_attn_same_real_mean",
            "last_attn_same_real_max",
        ]
        rank_cols = []
        for c in raw:
            if c not in df.columns or df[c].notna().sum() == 0:
                continue
            rc = "RANK_" + c
            if rc not in df.columns:
                df[rc] = (
                    df.groupby(["sid", "source_layer"], group_keys=False)[c]
                    .transform(percentile_rank_high)
                )
            rank_cols.append(rc)

        if rank_cols:
            df["SCORE_attn_ensemble"] = df[rank_cols].mean(axis=1, skipna=True)

    return df


def core_exact_sets(truth_for_target):
    return {
        int(sid): {
            (int(r["source_layer"]), int(r["position"]))
            for r in rows
        }
        for sid, rows in truth_for_target.items()
    }


def add_loo_role_prior(df, truth_for_target, target_name):
    """
    Relation-free calibration prior:
      P(core | source_layer, broad_category)
    estimated leave-one-sample-out.
    """
    out = df.copy()
    truth_sets = core_exact_sets(truth_for_target)

    is_core = np.array(
        [
            (int(L), int(pos)) in truth_sets.get(int(sid), set())
            for sid, L, pos in zip(
                out["sid"], out["source_layer"], out["position"]
            )
        ],
        dtype=np.int8,
    )
    bucket = (
        "L" + out["source_layer"].astype(int).astype(str)
        + ":" + out["broad_category"].astype(str)
    )

    tmp = pd.DataFrame(
        {
            "sid": out["sid"].astype(int).to_numpy(),
            "bucket": bucket.to_numpy(),
            "is_core": is_core,
        }
    )

    global_stats = tmp.groupby("bucket")["is_core"].agg(["sum", "count"])
    sid_stats = tmp.groupby(["sid", "bucket"])["is_core"].agg(["sum", "count"])

    prior = np.full(len(out), np.nan, dtype=np.float64)
    for i, r in enumerate(tmp.itertuples(index=False)):
        gs = global_stats.loc[r.bucket]
        try:
            ss = sid_stats.loc[(r.sid, r.bucket)]
            num = float(gs["sum"] - ss["sum"])
            den = float(gs["count"] - ss["count"])
        except KeyError:
            num = float(gs["sum"])
            den = float(gs["count"])
        prior[i] = num / den if den > 0 else np.nan

    pcol = f"SCORE_{target_name}_loo_role_prior"
    out[pcol] = prior

    # Match earlier self-predictor behavior: rank prior within sample.
    out[pcol + "_RANK"] = (
        out.groupby("sid", group_keys=False)[pcol]
        .transform(percentile_rank_high)
    )

    # Use attention ensemble when available; otherwise next-delta rank.
    if "SCORE_attn_ensemble" in out.columns:
        attn_base = "SCORE_attn_ensemble"
    else:
        attn_base = "RANK_last_attn_next_delta_mean"

    # Re-rank attention within SID x layer before mixture.
    acol = f"SCORE_{target_name}_attn_rank_for_mix"
    out[acol] = (
        out.groupby(["sid", "source_layer"], group_keys=False)[attn_base]
        .transform(percentile_rank_high)
    )

    mix = f"SCORE_{target_name}_attn75_prior25"
    out[mix] = 0.75 * out[acol] + 0.25 * out[pcol + "_RANK"]
    return out, mix


# =============================================================================
# Position selectors
# =============================================================================

METHOD_SPECS = {
    "attn_delta_max": {
        "score": "RANK_last_attn_next_delta_mean",
        "position_agg": "max",
        "layer_rule": "peak",
        "relation_free": True,
    },
    "attn_delta_top2": {
        "score": "RANK_last_attn_next_delta_mean",
        "position_agg": "top2",
        "layer_rule": "peak",
        "relation_free": True,
    },
    "attn_delta_mean": {
        "score": "RANK_last_attn_next_delta_mean",
        "position_agg": "mean",
        "layer_rule": "peak",
        "relation_free": True,
    },
    "attn_posdelta_max": {
        "score": "last_attn_next_max_positive_delta",
        "position_agg": "max",
        "layer_rule": "peak",
        "relation_free": True,
    },
    "attn_ensemble": {
        "score": "SCORE_attn_ensemble",
        "position_agg": "max",
        "layer_rule": "peak",
        "relation_free": True,
    },
    "attn75_prior25": {
        "score": "__TARGET_PRIOR_MIX__",
        "position_agg": "max",
        "layer_rule": "peak",
        "relation_free": True,
        "calibrated": True,
    },
    "attn_delta_top2_pm1split": {
        "score": "RANK_last_attn_next_delta_mean",
        "position_agg": "top2",
        "layer_rule": "pm1_split",
        "relation_free": True,
    },
    "random_pos_self_layer": {
        "score": "RANK_last_attn_next_delta_mean",
        "position_agg": "random",
        "layer_rule": "peak",
        "relation_free": True,
    },
}


def aggregate_position_score(vals, mode):
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    x = np.sort(x)[::-1]
    if mode == "max":
        return float(x[0])
    if mode == "top2":
        return float(x[: min(2, len(x))].mean())
    if mode == "mean":
        return float(x.mean())
    raise ValueError(mode)


def rank_positions(g, score_col, agg, sid, seed):
    """
    Returns one row per position:
      position_score
      peak_layer
      peak_layer_score
      broad_category
    """
    if score_col not in g.columns:
        raise RuntimeError(f"Missing required feature column: {score_col}")

    rows = []
    for pos, q in g.groupby("position", sort=False):
        score = pd.to_numeric(q[score_col], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(score)
        if not finite.any():
            continue

        q2 = q.loc[finite].copy()
        sc = pd.to_numeric(q2[score_col], errors="coerce").to_numpy(np.float64)
        j = int(np.argmax(sc))
        peak = q2.iloc[j]

        rows.append(
            {
                "position": int(pos),
                "position_score": aggregate_position_score(sc, agg) if agg != "random" else 0.0,
                "peak_layer": int(peak["source_layer"]),
                "peak_layer_score": float(sc[j]),
                "broad_category": str(peak["broad_category"]),
            }
        )

    out = pd.DataFrame(rows)
    if not len(out):
        return out

    if agg == "random":
        rng = np.random.default_rng(int(seed) * 1000003 + int(sid) * 1009)
        out["_random"] = rng.random(len(out))
        out = out.sort_values("_random", ascending=False).drop(columns="_random")
    else:
        out = out.sort_values(
            ["position_score", "peak_layer_score", "position"],
            ascending=[False, False, True],
        )
    return out.reset_index(drop=True)


def select_self_positions(
    g,
    method,
    k,
    sid,
    seed,
    target_prior_mix_col=None,
):
    spec = METHOD_SPECS[method]
    score_col = spec["score"]
    if score_col == "__TARGET_PRIOR_MIX__":
        if target_prior_mix_col is None:
            raise RuntimeError("attn75_prior25 requires target-specific prior mix")
        score_col = target_prior_mix_col

    ranked = rank_positions(
        g,
        score_col=score_col,
        agg=spec["position_agg"],
        sid=sid,
        seed=seed,
    )
    chosen = ranked.iloc[: min(int(k), len(ranked))].copy()
    chosen["method"] = method
    chosen["score_col"] = score_col
    chosen["layer_rule"] = spec["layer_rule"]
    return chosen


# =============================================================================
# Model hooks
# =============================================================================

def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def first_tensor(out):
    return traj.first_tensor(out)


def replace_first_tensor(out, y):
    return traj.replace_first_tensor(out, y)


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
            self.states[L] = h.detach().float().cpu()
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
            L: cap.states[L].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


class WeightedTokenDeltaEditor:
    """
    Edit source positions with:
      h[L,pos] += alpha * weight * (h_real[L,pos] - h_gray[L,pos])
    during the generation prefill only.
    """
    def __init__(self, decoder_layers, specs, alpha, prompt_len):
        self.handles = []
        self.applied = defaultdict(int)
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)

        for L, entries in specs.items():
            if entries:
                self.handles.append(
                    decoder_layers[L].register_forward_hook(
                        self._hook(L, entries)
                    )
                )

    def _hook(self, L, entries):
        def hook(_m, _inp, out):
            h = first_tensor(out)
            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()
            for pos, delta_np, weight in entries:
                pos = int(pos)
                if 0 <= pos < y.shape[1]:
                    delta = torch.as_tensor(
                        delta_np,
                        device=y.device,
                        dtype=torch.float32,
                    )
                    y[:, pos, :] = (
                        y[:, pos, :]
                        + self.alpha * float(weight) * delta
                    )
                    self.applied[L] += 1
            return replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def generate_with_specs(
    model,
    processor,
    decoder_layers,
    real_batch,
    specs,
    alpha,
    max_new_tokens,
):
    prompt_len = int(real_batch["input_ids"].shape[1])
    editor = WeightedTokenDeltaEditor(
        decoder_layers, specs, alpha, prompt_len
    )
    try:
        text = base.generate_text(
            model,
            processor,
            real_batch,
            max_new_tokens=max_new_tokens,
        )
        pred = traj.normalize_relation(base, text)
        return {
            "text": text,
            "prediction": canon_rel(pred),
            "n_edit_pairs": int(sum(editor.applied.values())),
        }
    finally:
        editor.close()


# =============================================================================
# Turn selected positions into actual intervention specs
# =============================================================================

def delta_at(hreal, hgray, L, pos):
    if L not in hreal or L not in hgray:
        return None
    R = hreal[L]
    G = hgray[L]
    if R.ndim == 3:
        R = R[0]
    if G.ndim == 3:
        G = G[0]
    if not (0 <= int(pos) < min(R.shape[0], G.shape[0])):
        return None
    return (R[int(pos)] - G[int(pos)]).astype(np.float32)


def build_specs_from_selected(
    chosen,
    hreal,
    hgray,
    source_layers,
    layer_rule,
):
    specs = defaultdict(list)
    export = []

    layer_set = set(int(x) for x in source_layers)

    for rank, r in enumerate(chosen.itertuples(), 1):
        pos = int(r.position)
        peak = int(r.peak_layer)

        if layer_rule == "peak":
            layers = [peak]
            weights = [1.0]

        elif layer_rule == "pm1_split":
            layers = [
                L for L in [peak - 1, peak, peak + 1]
                if L in layer_set
            ]
            if not layers:
                continue
            weights = [1.0 / len(layers)] * len(layers)

        else:
            raise ValueError(layer_rule)

        for L, w in zip(layers, weights):
            d = delta_at(hreal, hgray, L, pos)
            if d is None:
                continue
            specs[L].append((pos, d, float(w)))
            export.append(
                {
                    "rank": rank,
                    "position": pos,
                    "peak_layer": peak,
                    "edit_layer": int(L),
                    "edit_weight": float(w),
                    "position_score": float(r.position_score),
                    "broad_category": str(r.broad_category),
                }
            )

    return dict(specs), export


def build_oracle_exact_specs(core_rows, hreal, hgray):
    specs = defaultdict(list)
    export = []

    for rank, r in enumerate(core_rows, 1):
        L = int(r["source_layer"])
        pos = int(r["position"])
        d = delta_at(hreal, hgray, L, pos)
        if d is None:
            continue
        specs[L].append((pos, d, 1.0))
        export.append(
            {
                "rank": rank,
                "position": pos,
                "peak_layer": L,
                "edit_layer": L,
                "edit_weight": 1.0,
                "position_score": float(r["mediation"]),
                "broad_category": "oracle",
            }
        )
    return dict(specs), export


def choose_self_layer_for_positions(
    g,
    positions,
    score_col,
):
    rows = []
    for rank, pos in enumerate(positions, 1):
        q = g[g["position"] == int(pos)].copy()
        if not len(q):
            continue
        q["_score"] = pd.to_numeric(q[score_col], errors="coerce")
        q = q[np.isfinite(q["_score"])]
        if not len(q):
            continue
        best = q.sort_values(
            ["_score", "source_layer"],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            {
                "position": int(pos),
                "position_score": float(best["_score"]),
                "peak_layer": int(best["source_layer"]),
                "peak_layer_score": float(best["_score"]),
                "broad_category": str(best["broad_category"]),
            }
        )
    return pd.DataFrame(rows)


def choose_oracle_layer_for_self_positions(
    chosen_positions,
    med_sid,
):
    rows = []
    for rank, r in enumerate(chosen_positions.itertuples(), 1):
        pos = int(r.position)
        q = med_sid[med_sid["position"] == pos].copy()
        if not len(q):
            continue
        q = q.sort_values("mediation", ascending=False)
        best = q.iloc[0]
        rows.append(
            {
                "position": pos,
                "position_score": float(r.position_score),
                "peak_layer": int(best["source_layer"]),
                "peak_layer_score": float(best["mediation"]),
                "broad_category": str(
                    best["broad_category"]
                    if "broad_category" in best.index
                    else r.broad_category
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Summaries
# =============================================================================

def summarize_behavior(rows, baseline_by_sid):
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame()

    out = []
    keys = [
        "target_core",
        "method",
        "method_kind",
        "alpha",
    ]

    for key, g in df.groupby(keys, dropna=False):
        target_core, method, method_kind, alpha = key
        sids = sorted(g["sid"].astype(int).unique().tolist())

        bcorrect = {
            sid: bool(baseline_by_sid[sid]["baseline_correct"])
            for sid in sids
        }
        epred = {
            int(r.sid): canon_rel(r.prediction)
            for r in g.itertuples()
        }
        ecorrect = {
            int(r.sid): bool(r.correct)
            for r in g.itertuples()
        }
        bpred = {
            sid: canon_rel(baseline_by_sid[sid]["baseline_prediction"])
            for sid in sids
        }

        base_acc = safe_mean(float(bcorrect[s]) for s in sids)
        edit_acc = safe_mean(float(ecorrect[s]) for s in sids)
        wrong_n = sum(not bcorrect[s] for s in sids)
        correct_n = sum(bcorrect[s] for s in sids)
        w2c = sum((not bcorrect[s]) and ecorrect[s] for s in sids)
        c2w = sum(bcorrect[s] and (not ecorrect[s]) for s in sids)
        changed = sum(epred[s] != bpred[s] for s in sids)

        out.append(
            {
                "target_core": target_core,
                "method": method,
                "method_kind": method_kind,
                "alpha": alpha,
                "N": len(sids),
                "baseline_acc": base_acc,
                "edited_acc": edit_acc,
                "gain": edit_acc - base_acc,
                "baseline_wrong_N": wrong_n,
                "baseline_correct_N": correct_n,
                "W2C": w2c,
                "C2W": c2w,
                "net": w2c - c2w,
                "repair_rate_given_wrong": safe_div(w2c, wrong_n),
                "preserve_rate_given_correct": safe_div(correct_n - c2w, correct_n),
                "changed": changed,
                "mean_selected_positions": safe_mean(g["n_selected_positions"]),
                "mean_edit_pairs": safe_mean(g["n_edit_pairs"]),
            }
        )

    return pd.DataFrame(out).sort_values(
        ["target_core", "edited_acc", "net"],
        ascending=[True, False, False],
    )


def summarize_by_relation(rows, baseline_by_sid):
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame()

    out = []
    for key, g in df.groupby(
        ["target_core", "method", "method_kind", "alpha", "gt"],
        dropna=False,
    ):
        target_core, method, method_kind, alpha, gt = key
        sids = sorted(g["sid"].astype(int).unique().tolist())
        b = {
            sid: bool(baseline_by_sid[sid]["baseline_correct"])
            for sid in sids
        }
        e = {
            int(r.sid): bool(r.correct)
            for r in g.itertuples()
        }
        base_acc = safe_mean(float(b[s]) for s in sids)
        edit_acc = safe_mean(float(e[s]) for s in sids)
        w2c = sum((not b[s]) and e[s] for s in sids)
        c2w = sum(b[s] and (not e[s]) for s in sids)

        out.append(
            {
                "target_core": target_core,
                "method": method,
                "method_kind": method_kind,
                "alpha": alpha,
                "relation": gt,
                "N": len(sids),
                "baseline_acc": base_acc,
                "edited_acc": edit_acc,
                "gain": edit_acc - base_acc,
                "W2C": w2c,
                "C2W": c2w,
                "net": w2c - c2w,
            }
        )
    return pd.DataFrame(out)


def render_summary(summary, core_sizes, a):
    lines = []
    lines.append("=" * 138)
    lines.append("SELF-SELECTED CAUSAL-CORE -> ACTUAL GENERATION")
    lines.append("=" * 138)

    for c in sorted(x for x in core_sizes.columns if x.endswith("_N")):
        x = core_sizes[c]
        lines.append(
            f"{c[:-2]}: true oracle core size mean={x.mean():.2f}, "
            f"median={x.median():.1f}"
        )

    lines.append("")
    lines.append(
        "Deployable methods use NO LEFT/RIGHT/ON/UNDER for selection. "
        "GT is used only to score generation correctness."
    )
    lines.append(
        "Oracle diagnostics are explicitly marked diagnostic_oracle and must "
        "not be reported as a deployable selector."
    )

    for target in sorted(summary["target_core"].unique()):
        lines.append("")
        lines.append("-" * 138)
        lines.append(target.upper())
        lines.append("-" * 138)
        g = summary[summary["target_core"] == target].copy()
        g = g.sort_values(
            ["method_kind", "edited_acc"],
            ascending=[True, False],
        )
        for r in g.itertuples():
            lines.append(
                f"{r.method:<34s} "
                f"{r.method_kind:<22s} "
                f"a={r.alpha:<5.2f} "
                f"acc={r.edited_acc:.4f} "
                f"gain={r.gain:+.4f} "
                f"W2C/C2W={int(r.W2C):3d}/{int(r.C2W):3d} "
                f"repair={r.repair_rate_given_wrong:.3f} "
                f"preserve={r.preserve_rate_given_correct:.3f} "
                f"pos/edit={r.mean_selected_positions:.1f}/{r.mean_edit_pairs:.1f}"
            )

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  The primary result is edited_acc / W2C / C2W, not overlap with oracle K36."
    )
    lines.append(
        "  If a relation-free selector produces positive net W2C-C2W, its selected "
        "tokens are behaviorally useful even when exact oracle-set recall is modest."
    )
    lines.append(
        "  Compare oracle_pos_self_layer vs oracle_core_exact to isolate layer-assignment cost."
    )
    lines.append(
        "  Compare self_pos_oracle_layer vs oracle_core_exact to isolate position-selection cost."
    )
    lines.append(
        "  Compare attn_delta_top2_pm1split vs attn_delta_top2 to test whether a small "
        "layer window compensates for uncertain exact layer without increasing total alpha."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = infer_source_layers(a.bundle)
    core_masses = parse_floats(a.core_masses)
    alphas = parse_floats(a.alphas)
    methods = parse_methods(a.methods)
    diagnostics = parse_methods(a.diagnostics)

    bad_methods = [m for m in methods if m not in METHOD_SPECS]
    if bad_methods:
        raise ValueError(
            f"Unknown methods: {bad_methods}\nAvailable: {sorted(METHOD_SPECS)}"
        )
    valid_diag = {
        "oracle_core_exact",
        "oracle_pos_self_layer",
        "self_pos_oracle_layer",
    }
    bad_diag = [m for m in diagnostics if m not in valid_diag]
    if bad_diag:
        raise ValueError(f"Unknown diagnostics: {bad_diag}")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dir = outdir / "generation_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    baseline, med, sel, valid_sids = load_existing_run(a, source_layers)

    if a.max_eval_samples and a.max_eval_samples > 0:
        rng = np.random.default_rng(a.seed)
        keep = sorted(
            rng.choice(
                valid_sids,
                size=min(a.max_eval_samples, len(valid_sids)),
                replace=False,
            ).tolist()
        )
        valid_sids = keep
        baseline = baseline[baseline["sid"].isin(keep)].copy()
        med = med[med["sid"].isin(keep)].copy()
        sel = sel[sel["sid"].isin(keep)].copy()

    truth, core_sizes = build_core_truth(sel, core_masses)
    features = load_features(a, valid_sids, source_layers)

    # Target-specific calibrated score tables.
    features_by_target = {}
    prior_mix_cols = {}
    for t in core_masses:
        name = fmt_core(t)
        f, mix = add_loo_role_prior(features, truth[name], name)
        features_by_target[name] = f
        prior_mix_cols[name] = mix

    # Fixed deployable K = median true core size across calibration dataset.
    fixed_k = {
        fmt_core(t): int(round(float(core_sizes[fmt_core(t) + "_N"].median())))
        for t in core_masses
    }

    # Baseline lookup.
    baseline_by_sid = {}
    for r in baseline.itertuples():
        gt = canon_rel(getattr(r, "gt", ""))
        baseline_by_sid[int(r.sid)] = {
            "gt": gt,
            "baseline_prediction": canon_rel(r.baseline_prediction),
            "baseline_correct": bool(r.baseline_correct),
        }

    missing_base = [sid for sid in valid_sids if sid not in baseline_by_sid]
    if missing_base:
        raise RuntimeError(f"Missing baseline rows for sids: {missing_base[:10]}")

    # Data.
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    # Model.
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
    n_layers = len(decoder_layers)
    for L in source_layers:
        if not (0 <= L < n_layers):
            raise ValueError(f"L{L} invalid; model has L0..L{n_layers-1}")

    print("=" * 138)
    print("RELATION-FREE SELF-SELECTED CORE -> ACTUAL GENERATION")
    print("=" * 138)
    print("model:", a.model, spec.repo_id)
    print("decoder:", decoder_path)
    print("N:", len(valid_sids))
    print("source layers:", source_layers)
    print("core fixed K:", fixed_k)
    print("methods:", methods)
    print("alphas:", alphas)
    print("diagnostics:", diagnostics)
    print("diagnostic alpha:", a.diagnostic_alpha)
    print("NOTE: GT relation is NOT used by deployable selectors.")
    print()

    all_rows = []
    selected_export = []

    try:
        for sid in tqdm(valid_sids, desc="Self-select + generate"):
            sid = int(sid)
            chunk_path = chunk_dir / f"sid_{sid}.pkl.gz"

            if chunk_path.exists():
                cached = pd.read_pickle(chunk_path, compression="gzip")
                all_rows.extend(cached["rows"])
                selected_export.extend(cached["selected"])
                continue

            if sid not in prompts or sid not in rec_by_sid:
                raise RuntimeError(f"sid={sid} missing prompt/data record")

            pr = prompts[sid]
            gt = baseline_by_sid[sid]["gt"]
            if not gt:
                # Fallback from feature/run relation if baseline gt is absent.
                qrel = sel[sel["sid"] == sid]["relation"]
                gt = canon_rel(qrel.iloc[0])

            real = gray = rb = gb = None
            sid_rows = []
            sid_selected = []

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                device = torch.device(a.device)
                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=str(pr["question_text"]),
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=str(pr["question_text"]),
                    device=device,
                )

                hreal = capture_cpu(
                    model, decoder_layers, rb, source_layers
                )
                hgray = capture_cpu(
                    model, decoder_layers, gb, source_layers
                )

                med_sid = med[med["sid"] == sid].copy()

                # ---------------------------------------------------------
                # Deployable / relation-free selectors.
                # ---------------------------------------------------------
                for t in core_masses:
                    target = fmt_core(t)
                    k = fixed_k[target]
                    f_sid = features_by_target[target]
                    f_sid = f_sid[f_sid["sid"] == sid].copy()

                    for method in methods:
                        chosen = select_self_positions(
                            f_sid,
                            method=method,
                            k=k,
                            sid=sid,
                            seed=a.seed,
                            target_prior_mix_col=prior_mix_cols[target],
                        )
                        if not len(chosen):
                            continue

                        layer_rule = METHOD_SPECS[method]["layer_rule"]
                        specs_mid, exp = build_specs_from_selected(
                            chosen,
                            hreal,
                            hgray,
                            source_layers,
                            layer_rule,
                        )

                        for e in exp:
                            sid_selected.append(
                                {
                                    "sid": sid,
                                    "gt": gt,
                                    "target_core": target,
                                    "method": method,
                                    "method_kind": (
                                        "calibrated_no_relation"
                                        if method == "attn75_prior25"
                                        else (
                                            "random_control"
                                            if method == "random_pos_self_layer"
                                            else "pure_self"
                                        )
                                    ),
                                    **e,
                                }
                            )

                        for alpha in alphas:
                            out = generate_with_specs(
                                model,
                                processor,
                                decoder_layers,
                                rb,
                                specs_mid,
                                alpha,
                                a.max_new_tokens,
                            )
                            kind = (
                                "calibrated_no_relation"
                                if method == "attn75_prior25"
                                else (
                                    "random_control"
                                    if method == "random_pos_self_layer"
                                    else "pure_self"
                                )
                            )
                            sid_rows.append(
                                {
                                    "sid": sid,
                                    "gt": gt,
                                    "baseline_prediction": baseline_by_sid[sid][
                                        "baseline_prediction"
                                    ],
                                    "baseline_correct": baseline_by_sid[sid][
                                        "baseline_correct"
                                    ],
                                    "target_core": target,
                                    "method": method,
                                    "method_kind": kind,
                                    "alpha": float(alpha),
                                    "prediction": out["prediction"],
                                    "correct": out["prediction"] == gt,
                                    "text": out["text"],
                                    "n_selected_positions": len(chosen),
                                    "n_edit_pairs": out["n_edit_pairs"],
                                }
                            )

                    # ---------------------------------------------------------
                    # Oracle diagnostics: run only at diagnostic-alpha.
                    # ---------------------------------------------------------
                    score_col = "RANK_last_attn_next_delta_mean"
                    if score_col not in f_sid.columns:
                        raise RuntimeError(
                            f"Diagnostic layer selector needs {score_col}"
                        )

                    if "oracle_core_exact" in diagnostics:
                        specs_mid, exp = build_oracle_exact_specs(
                            truth[target][sid],
                            hreal,
                            hgray,
                        )
                        out = generate_with_specs(
                            model,
                            processor,
                            decoder_layers,
                            rb,
                            specs_mid,
                            a.diagnostic_alpha,
                            a.max_new_tokens,
                        )
                        sid_rows.append(
                            {
                                "sid": sid,
                                "gt": gt,
                                "baseline_prediction": baseline_by_sid[sid][
                                    "baseline_prediction"
                                ],
                                "baseline_correct": baseline_by_sid[sid][
                                    "baseline_correct"
                                ],
                                "target_core": target,
                                "method": "oracle_core_exact",
                                "method_kind": "diagnostic_oracle",
                                "alpha": float(a.diagnostic_alpha),
                                "prediction": out["prediction"],
                                "correct": out["prediction"] == gt,
                                "text": out["text"],
                                "n_selected_positions": len(
                                    {int(x["position"]) for x in truth[target][sid]}
                                ),
                                "n_edit_pairs": out["n_edit_pairs"],
                            }
                        )

                    if "oracle_pos_self_layer" in diagnostics:
                        oracle_positions = [
                            int(x["position"]) for x in truth[target][sid]
                        ]
                        chosen = choose_self_layer_for_positions(
                            f_sid,
                            oracle_positions,
                            score_col=score_col,
                        )
                        specs_mid, exp = build_specs_from_selected(
                            chosen,
                            hreal,
                            hgray,
                            source_layers,
                            "peak",
                        )
                        out = generate_with_specs(
                            model,
                            processor,
                            decoder_layers,
                            rb,
                            specs_mid,
                            a.diagnostic_alpha,
                            a.max_new_tokens,
                        )
                        sid_rows.append(
                            {
                                "sid": sid,
                                "gt": gt,
                                "baseline_prediction": baseline_by_sid[sid][
                                    "baseline_prediction"
                                ],
                                "baseline_correct": baseline_by_sid[sid][
                                    "baseline_correct"
                                ],
                                "target_core": target,
                                "method": "oracle_pos_self_layer",
                                "method_kind": "diagnostic_oracle",
                                "alpha": float(a.diagnostic_alpha),
                                "prediction": out["prediction"],
                                "correct": out["prediction"] == gt,
                                "text": out["text"],
                                "n_selected_positions": len(chosen),
                                "n_edit_pairs": out["n_edit_pairs"],
                            }
                        )

                    if "self_pos_oracle_layer" in diagnostics:
                        diag_method = a.self_pos_diagnostic_method
                        if diag_method not in METHOD_SPECS:
                            raise ValueError(
                                f"Bad --self-pos-diagnostic-method={diag_method}"
                            )
                        chosen_self = select_self_positions(
                            f_sid,
                            method=diag_method,
                            k=k,
                            sid=sid,
                            seed=a.seed,
                            target_prior_mix_col=prior_mix_cols[target],
                        )
                        chosen = choose_oracle_layer_for_self_positions(
                            chosen_self,
                            med_sid,
                        )
                        specs_mid, exp = build_specs_from_selected(
                            chosen,
                            hreal,
                            hgray,
                            source_layers,
                            "peak",
                        )
                        out = generate_with_specs(
                            model,
                            processor,
                            decoder_layers,
                            rb,
                            specs_mid,
                            a.diagnostic_alpha,
                            a.max_new_tokens,
                        )
                        sid_rows.append(
                            {
                                "sid": sid,
                                "gt": gt,
                                "baseline_prediction": baseline_by_sid[sid][
                                    "baseline_prediction"
                                ],
                                "baseline_correct": baseline_by_sid[sid][
                                    "baseline_correct"
                                ],
                                "target_core": target,
                                "method": "self_pos_oracle_layer",
                                "method_kind": "diagnostic_oracle",
                                "alpha": float(a.diagnostic_alpha),
                                "prediction": out["prediction"],
                                "correct": out["prediction"] == gt,
                                "text": out["text"],
                                "n_selected_positions": len(chosen),
                                "n_edit_pairs": out["n_edit_pairs"],
                            }
                        )

                pd.to_pickle(
                    {"rows": sid_rows, "selected": sid_selected},
                    chunk_path,
                    compression="gzip",
                )
                all_rows.extend(sid_rows)
                selected_export.extend(sid_selected)

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_behavior(all_rows, baseline_by_sid)
    by_relation = summarize_by_relation(all_rows, baseline_by_sid)

    write_csv(outdir / "behavior_per_sample.csv", all_rows)
    summary.to_csv(outdir / "behavior_summary.csv", index=False)
    by_relation.to_csv(outdir / "behavior_by_relation.csv", index=False)
    write_csv(outdir / "selected_positions.csv", selected_export)
    core_sizes.to_csv(outdir / "oracle_core_sizes.csv", index=False)

    text = render_summary(summary, core_sizes, a)
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    metadata = {
        "run_dir": a.run_dir,
        "feature_dir": a.feature_dir,
        "model": a.model,
        "bundle": a.bundle,
        "source_layers": source_layers,
        "oracle_k": a.oracle_k,
        "core_masses": core_masses,
        "fixed_k_by_core": fixed_k,
        "alphas": alphas,
        "methods": methods,
        "diagnostics": diagnostics,
        "diagnostic_alpha": a.diagnostic_alpha,
        "N": len(valid_sids),
        "selection_relation_free_for_method_kind": {
            "pure_self": True,
            "calibrated_no_relation": True,
            "random_control": True,
            "diagnostic_oracle": False,
        },
        "note": (
            "GT relation is used only to score prediction correctness for "
            "relation-free methods. Oracle diagnostics use mediation/core labels "
            "and are not deployable."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print(text)
    print("Saved:", outdir)



# ===== Focused Core50 WHERE/WHEN experiment =====

import argparse
import contextlib
import csv
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
from transformers import AutoProcessor
from tqdm import tqdm

import sys as _sys
E = _sys.modules[__name__]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument(
        "--bundle",
        default="L20+L21+L22+L23+L24+L25+L26[global_unique]",
    )
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument("--oracle-k", type=int, default=36)
    p.add_argument(
        "--core-mass",
        type=float,
        default=0.5,
        help="Default Core50; keep 0.5 for this diagnostic.",
    )
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
    p.add_argument("--max-eval-samples", type=int, default=80)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def canon_rel(x):
    return E.canon_rel(x)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


# -----------------------------------------------------------------------------
# Layer-rule helpers
# -----------------------------------------------------------------------------

def ensure_feature(df, col):
    if col not in df.columns or df[col].notna().sum() == 0:
        raise RuntimeError(
            f"Required feature {col!r} is missing/all-NaN in feature cache."
        )


def choose_layer_rows_for_oracle_positions(
    feature_sid,
    oracle_core_rows,
    rule,
    source_layers,
):
    """
    Return rows:
      position, [(edit_layer, weight), ...], score
    Oracle is used ONLY to provide position identity.
    Layer selection itself uses relation-free cached features.
    """
    if rule in {
        "next_delta_peak",
        "next_delta_top2_split",
        "next_delta_top3_split",
        "next_delta_pm1_split",
        "next_delta_soft_all",
    }:
        score_col = "RANK_last_attn_next_delta_mean"
    elif rule == "next_posdelta_peak":
        score_col = "last_attn_next_max_positive_delta"
    elif rule == "attn_ensemble_peak":
        score_col = "SCORE_attn_ensemble"
    else:
        raise ValueError(rule)

    ensure_feature(feature_sid, score_col)
    allowed = set(int(x) for x in source_layers)
    out = []

    for core_rank, cr in enumerate(oracle_core_rows, 1):
        pos = int(cr["position"])
        q = feature_sid[feature_sid["position"] == pos].copy()
        q["_score"] = pd.to_numeric(q[score_col], errors="coerce")
        q = q[
            q["source_layer"].astype(int).isin(allowed)
            & np.isfinite(q["_score"])
        ].copy()
        if not len(q):
            continue

        q = q.sort_values(
            ["_score", "source_layer"],
            ascending=[False, True],
        ).reset_index(drop=True)

        peak_layer = int(q.iloc[0]["source_layer"])
        peak_score = float(q.iloc[0]["_score"])

        if rule in {
            "next_delta_peak",
            "next_posdelta_peak",
            "attn_ensemble_peak",
        }:
            layers_weights = [(peak_layer, 1.0)]

        elif rule == "next_delta_top2_split":
            qq = q.drop_duplicates("source_layer").head(2)
            layers = [int(x) for x in qq["source_layer"]]
            layers_weights = [(L, 1.0 / len(layers)) for L in layers]

        elif rule == "next_delta_top3_split":
            qq = q.drop_duplicates("source_layer").head(3)
            layers = [int(x) for x in qq["source_layer"]]
            layers_weights = [(L, 1.0 / len(layers)) for L in layers]

        elif rule == "next_delta_pm1_split":
            layers = [
                L for L in [peak_layer - 1, peak_layer, peak_layer + 1]
                if L in allowed
            ]
            layers_weights = [(L, 1.0 / len(layers)) for L in layers]

        elif rule == "next_delta_soft_all":
            # Score is a within-sample/layer percentile rank in [0,1].
            # Normalize across available layers for this position.
            qq = q.drop_duplicates("source_layer").copy()
            vals = np.maximum(
                pd.to_numeric(qq["_score"], errors="coerce").to_numpy(float),
                0.0,
            )
            if vals.sum() <= 1e-12:
                vals = np.ones(len(vals), dtype=float)
            vals = vals / vals.sum()
            layers_weights = [
                (int(L), float(w))
                for L, w in zip(qq["source_layer"], vals)
            ]
        else:
            raise AssertionError(rule)

        out.append(
            {
                "core_rank": core_rank,
                "position": pos,
                "oracle_layer": int(cr["source_layer"]),
                "oracle_mediation": float(cr["mediation"]),
                "peak_self_layer": peak_layer,
                "peak_self_score": peak_score,
                "score_col": score_col,
                "layers_weights": layers_weights,
            }
        )
    return out


def specs_from_layer_rows(layer_rows, hreal, hgray):
    specs = defaultdict(list)
    export = []
    for r in layer_rows:
        pos = int(r["position"])
        for L, w in r["layers_weights"]:
            d = E.delta_at(hreal, hgray, int(L), pos)
            if d is None:
                continue
            specs[int(L)].append((pos, d, float(w)))
            export.append(
                {
                    "position": pos,
                    "core_rank": int(r["core_rank"]),
                    "oracle_layer": int(r["oracle_layer"]),
                    "oracle_mediation": float(r["oracle_mediation"]),
                    "peak_self_layer": int(r["peak_self_layer"]),
                    "edit_layer": int(L),
                    "edit_weight": float(w),
                    "self_score": float(r["peak_self_score"]),
                    "score_col": str(r["score_col"]),
                }
            )
    return dict(specs), export


# -----------------------------------------------------------------------------
# Behavioral summaries
# -----------------------------------------------------------------------------

def summarize(rows):
    df = pd.DataFrame(rows)
    out = []
    if not len(df):
        return pd.DataFrame()

    for key, g in df.groupby(
        ["axis", "method", "method_kind", "alpha"],
        dropna=False,
    ):
        axis, method, kind, alpha = key
        base = g["baseline_correct"].astype(bool).to_numpy()
        edit = g["correct"].astype(bool).to_numpy()
        bpred = g["baseline_prediction"].astype(str).to_numpy()
        epred = g["prediction"].astype(str).to_numpy()

        wrong_n = int((~base).sum())
        correct_n = int(base.sum())
        w2c = int(((~base) & edit).sum())
        c2w = int((base & (~edit)).sum())

        out.append(
            {
                "axis": axis,
                "method": method,
                "method_kind": kind,
                "alpha": float(alpha),
                "N": len(g),
                "baseline_acc": float(base.mean()),
                "edited_acc": float(edit.mean()),
                "gain": float(edit.mean() - base.mean()),
                "W2C": w2c,
                "C2W": c2w,
                "net": w2c - c2w,
                "repair_rate_given_wrong": safe_div(w2c, wrong_n),
                "preserve_rate_given_correct": safe_div(correct_n - c2w, correct_n),
                "changed": int(np.sum(bpred != epred)),
                "mean_selected_positions": safe_mean(g["n_selected_positions"]),
                "mean_edit_pairs": safe_mean(g["n_edit_pairs"]),
            }
        )

    return pd.DataFrame(out).sort_values(
        ["axis", "edited_acc", "net"],
        ascending=[True, False, False],
    )


def summarize_by_relation(rows):
    df = pd.DataFrame(rows)
    out = []
    if not len(df):
        return pd.DataFrame()

    for key, g in df.groupby(
        ["axis", "method", "method_kind", "alpha", "gt"],
        dropna=False,
    ):
        axis, method, kind, alpha, gt = key
        base = g["baseline_correct"].astype(bool).to_numpy()
        edit = g["correct"].astype(bool).to_numpy()
        out.append(
            {
                "axis": axis,
                "method": method,
                "method_kind": kind,
                "alpha": float(alpha),
                "relation": gt,
                "N": len(g),
                "baseline_acc": float(base.mean()),
                "edited_acc": float(edit.mean()),
                "gain": float(edit.mean() - base.mean()),
                "W2C": int(((~base) & edit).sum()),
                "C2W": int((base & (~edit)).sum()),
            }
        )
    return pd.DataFrame(out)


def render_summary(summary, core_sizes, fixed_k, a):
    lines = []
    lines.append("=" * 144)
    lines.append("CORE POSITION vs LAYER DIAGNOSTIC — ALPHA FIXED TO 1")
    lines.append("=" * 144)
    lines.append(
        f"core={E.fmt_core(a.core_mass)} | "
        f"true core size mean={core_sizes[E.fmt_core(a.core_mass)+'_N'].mean():.2f}, "
        f"median={core_sizes[E.fmt_core(a.core_mass)+'_N'].median():.1f} | "
        f"self position K={fixed_k}"
    )
    lines.append("")
    lines.append(
        "Axis POSITION: self-select position + ORACLE best layer. "
        "Higher acc means better relation-free position localization."
    )
    lines.append(
        "Axis LAYER: ORACLE core positions + self-selected layer rule. "
        "Higher acc means better layer assignment."
    )
    lines.append(
        "Multi-layer layer rules preserve total per-position intervention weight = alpha=1."
    )

    for axis in ["reference", "position", "layer"]:
        g = summary[summary["axis"] == axis].copy()
        if not len(g):
            continue
        lines.append("")
        lines.append("-" * 144)
        lines.append(axis.upper())
        lines.append("-" * 144)
        for r in g.sort_values("edited_acc", ascending=False).itertuples():
            lines.append(
                f"{r.method:<34s} "
                f"acc={r.edited_acc:.4f} gain={r.gain:+.4f} "
                f"W2C/C2W={int(r.W2C):3d}/{int(r.C2W):3d} "
                f"net={int(r.net):+3d} "
                f"repair={r.repair_rate_given_wrong:.3f} "
                f"preserve={r.preserve_rate_given_correct:.3f} "
                f"pos/edit={r.mean_selected_positions:.1f}/{r.mean_edit_pairs:.1f}"
            )

    lines.append("")
    lines.append("Readout:")
    lines.append(
        "  Best POSITION method tells us which self signal finds behaviorally useful token positions "
        "when exact layer is no longer the bottleneck."
    )
    lines.append(
        "  Best LAYER method tells us how to intervene on a known causal position without knowing its oracle layer."
    )
    lines.append(
        "  If multi-layer split beats hard peak, exact layer should be treated as a trajectory/window rather than "
        "a single discrete layer."
    )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    a = parse_args()
    if abs(float(a.alpha) - 1.0) > 1e-12:
        raise ValueError(
            "This focused script intentionally fixes alpha=1.0. "
            "Run with --alpha 1.0."
        )

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = E.infer_source_layers(a.bundle)
    target = E.fmt_core(a.core_mass)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dir = outdir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    baseline, med, sel, valid_sids = E.load_existing_run(a, source_layers)

    if a.max_eval_samples and a.max_eval_samples > 0:
        rng = np.random.default_rng(a.seed)
        valid_sids = sorted(
            rng.choice(
                valid_sids,
                size=min(a.max_eval_samples, len(valid_sids)),
                replace=False,
            ).tolist()
        )
        baseline = baseline[baseline["sid"].isin(valid_sids)].copy()
        med = med[med["sid"].isin(valid_sids)].copy()
        sel = sel[sel["sid"].isin(valid_sids)].copy()

    truth, core_sizes = E.build_core_truth(sel, [a.core_mass])
    fixed_k = int(round(float(core_sizes[target + "_N"].median())))

    features = E.load_features(a, valid_sids, source_layers)
    features_prior, prior_mix_col = E.add_loo_role_prior(
        features, truth[target], target
    )

    # Baseline lookup.
    baseline_by_sid = {}
    for r in baseline.itertuples():
        baseline_by_sid[int(r.sid)] = {
            "gt": canon_rel(getattr(r, "gt", "")),
            "baseline_prediction": canon_rel(r.baseline_prediction),
            "baseline_correct": bool(r.baseline_correct),
        }

    # Data/model.
    two = E.base.import_two_object_module()
    prompts = E.base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _ = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    specs = E.base.merged_model_specs(two)
    spec = specs[a.model]
    model_cls = getattr(transformers, spec.model_class)

    load_kw = dict(
        dtype=E.base.resolve_dtype(spec.dtype_name),
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
    E.base.configure_processor(model, processor)
    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = E.base.resolve_decoder_layers(model)

    position_methods = [
        "attn_delta_max",
        "attn_delta_top2",
        "attn_delta_mean",
        "attn_posdelta_max",
        "attn_ensemble",
        "attn75_prior25",
        "random_pos_self_layer",
    ]
    layer_methods = [
        "next_delta_peak",
        "next_posdelta_peak",
        "attn_ensemble_peak",
        "next_delta_top2_split",
        "next_delta_top3_split",
        "next_delta_pm1_split",
        "next_delta_soft_all",
    ]

    print("=" * 144)
    print("CORE50 WHERE vs WHEN — alpha=1")
    print("=" * 144)
    print("N:", len(valid_sids))
    print("source layers:", source_layers)
    print("self position K:", fixed_k)
    print("position methods:", position_methods)
    print("layer methods:", layer_methods)
    print("generations/sample:", 1 + len(position_methods) + len(layer_methods))
    print()

    all_rows = []
    selected_rows = []

    try:
        for sid in tqdm(valid_sids, desc="WHERE/WHEN"):
            sid = int(sid)
            cache = chunk_dir / f"sid_{sid}.pkl.gz"
            if cache.exists():
                obj = pd.read_pickle(cache, compression="gzip")
                all_rows.extend(obj["rows"])
                selected_rows.extend(obj["selected"])
                continue

            gt = baseline_by_sid[sid]["gt"]
            if not gt:
                gt = canon_rel(sel[sel["sid"] == sid]["relation"].iloc[0])

            pr = prompts[sid]
            real = gray = rb = gb = None
            sid_rows, sid_selected = [], []

            try:
                real = E.base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = E.make_gray_image(real, a.gray_value)

                device = torch.device(a.device)
                rb = E.base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=str(pr["question_text"]),
                    device=device,
                )
                gb = E.base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=str(pr["question_text"]),
                    device=device,
                )

                hreal = E.capture_cpu(model, decoder_layers, rb, source_layers)
                hgray = E.capture_cpu(model, decoder_layers, gb, source_layers)

                f_sid = features[features["sid"] == sid].copy()
                fp_sid = features_prior[features_prior["sid"] == sid].copy()
                med_sid = med[med["sid"] == sid].copy()
                oracle_core = truth[target][sid]

                def add_result(axis, method, kind, gen_out, npos):
                    sid_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "baseline_prediction": baseline_by_sid[sid][
                                "baseline_prediction"
                            ],
                            "baseline_correct": baseline_by_sid[sid][
                                "baseline_correct"
                            ],
                            "axis": axis,
                            "method": method,
                            "method_kind": kind,
                            "alpha": 1.0,
                            "prediction": gen_out["prediction"],
                            "correct": gen_out["prediction"] == gt,
                            "text": gen_out["text"],
                            "n_selected_positions": int(npos),
                            "n_edit_pairs": int(gen_out["n_edit_pairs"]),
                        }
                    )

                # ---------------------------------------------------------
                # Reference: oracle Core50 exact.
                # ---------------------------------------------------------
                specs_mid, exp = E.build_oracle_exact_specs(
                    oracle_core, hreal, hgray
                )
                out = E.generate_with_specs(
                    model, processor, decoder_layers, rb,
                    specs_mid, 1.0, a.max_new_tokens
                )
                add_result(
                    "reference", "oracle_core_exact", "diagnostic_oracle",
                    out, len(oracle_core)
                )

                # ---------------------------------------------------------
                # Axis A: self position + oracle layer.
                # ---------------------------------------------------------
                for method in position_methods:
                    use_df = fp_sid if method == "attn75_prior25" else f_sid
                    chosen_self = E.select_self_positions(
                        use_df,
                        method=method,
                        k=fixed_k,
                        sid=sid,
                        seed=a.seed,
                        target_prior_mix_col=(
                            prior_mix_col if method == "attn75_prior25" else None
                        ),
                    )

                    chosen_oracle_layer = E.choose_oracle_layer_for_self_positions(
                        chosen_self,
                        med_sid,
                    )
                    specs_mid, exp = E.build_specs_from_selected(
                        chosen_oracle_layer,
                        hreal,
                        hgray,
                        source_layers,
                        "peak",
                    )
                    out = E.generate_with_specs(
                        model, processor, decoder_layers, rb,
                        specs_mid, 1.0, a.max_new_tokens
                    )

                    kind = (
                        "random_control_oracle_layer"
                        if method == "random_pos_self_layer"
                        else (
                            "calibrated_self_pos_oracle_layer"
                            if method == "attn75_prior25"
                            else "self_pos_oracle_layer"
                        )
                    )
                    add_result(
                        "position",
                        method.replace("random_pos_self_layer",
                                       "random_pos_oracle_layer"),
                        kind,
                        out,
                        len(chosen_oracle_layer),
                    )

                    for rank, r in enumerate(chosen_oracle_layer.itertuples(), 1):
                        sid_selected.append(
                            {
                                "sid": sid,
                                "axis": "position",
                                "method": method,
                                "rank": rank,
                                "position": int(r.position),
                                "oracle_edit_layer": int(r.peak_layer),
                                "position_score": float(r.position_score),
                            }
                        )

                # ---------------------------------------------------------
                # Axis B: oracle positions + self layer rule.
                # ---------------------------------------------------------
                for rule in layer_methods:
                    layer_rows = choose_layer_rows_for_oracle_positions(
                        f_sid,
                        oracle_core,
                        rule,
                        source_layers,
                    )
                    specs_mid, exp = specs_from_layer_rows(
                        layer_rows, hreal, hgray
                    )
                    out = E.generate_with_specs(
                        model, processor, decoder_layers, rb,
                        specs_mid, 1.0, a.max_new_tokens
                    )
                    add_result(
                        "layer",
                        rule,
                        "oracle_pos_self_layer",
                        out,
                        len(layer_rows),
                    )

                    for e in exp:
                        sid_selected.append(
                            {
                                "sid": sid,
                                "axis": "layer",
                                "method": rule,
                                **e,
                            }
                        )

                pd.to_pickle(
                    {"rows": sid_rows, "selected": sid_selected},
                    cache,
                    compression="gzip",
                )
                all_rows.extend(sid_rows)
                selected_rows.extend(sid_selected)

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(all_rows)
    by_relation = summarize_by_relation(all_rows)

    write_csv(outdir / "behavior_per_sample.csv", all_rows)
    summary.to_csv(outdir / "behavior_summary.csv", index=False)
    by_relation.to_csv(outdir / "behavior_by_relation.csv", index=False)
    write_csv(outdir / "selected_details.csv", selected_rows)
    core_sizes.to_csv(outdir / "core_sizes.csv", index=False)

    text = render_summary(summary, core_sizes, fixed_k, a)
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    meta = {
        "alpha": 1.0,
        "core_mass": a.core_mass,
        "target": target,
        "N": len(valid_sids),
        "fixed_self_position_k": fixed_k,
        "source_layers": source_layers,
        "position_methods": position_methods,
        "layer_methods": layer_methods,
        "run_dir": a.run_dir,
        "feature_dir": a.feature_dir,
        "note": (
            "Position-axis experiments use oracle layer; layer-axis experiments "
            "use oracle core positions. Only oracle_core_exact uses both oracle "
            "position and oracle layer. This script is diagnostic, not deployable."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print()
    print(text)
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
