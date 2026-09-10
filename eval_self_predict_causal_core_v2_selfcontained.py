#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELF-PREDICT each sample's causal-core tokens WITHOUT relation routing.

SELF-CONTAINED VERSION.
No analyze_dynamic_k24_predictive_correlates_v*.py dependency is required.

The first half of this file embeds the feature collector used by the earlier
correlation experiment; the second half evaluates relation-free prediction of
Core50/Core70/Core80 from the L20..L26 K36 oracle run.
"""

# ===== Embedded feature collector (v3) =====
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL_ORDER = ["left", "right", "on", "under"]


# =============================================================================
# CLI / generic helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])

    p.add_argument("--bundle", default="L22+L24+L26[global_unique]")
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument("--target-layers", default="32,34,35")

    p.add_argument(
        "--with-attention-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request decoder attention matrices and derive last->token features.",
    )
    p.add_argument(
        "--attention-head-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also test each source-layer attention head separately.",
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="0 = use every sample represented in mediation_tokens.csv.",
    )

    p.add_argument(
        "--save-all-token-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save concatenated token_features.pkl.gz; feature chunks are always cached.",
    )
    p.add_argument(
        "--top-summary-n",
        type=int,
        default=30,
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def canon_rel(x):
    x = str(x).strip().lower()
    if x == "above":
        return "on"
    if x == "below":
        return "under"
    return x


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def first_tensor(out):
    try:
        return traj.first_tensor(out)
    except Exception:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            for x in out:
                if torch.is_tensor(x):
                    return x
        raise


def cosine_rows_to_vec(X, v, eps=1e-8):
    X = np.asarray(X, np.float32)
    v = np.asarray(v, np.float32)
    xn = np.linalg.norm(X, axis=1)
    vn = float(np.linalg.norm(v))
    den = xn * vn
    out = np.full(len(X), np.nan, np.float32)
    good = den > eps
    if vn > eps and np.any(good):
        out[good] = (X[good] @ v) / den[good]
    return out


def cosine_rows(A, B, eps=1e-8):
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    an = np.linalg.norm(A, axis=1)
    bn = np.linalg.norm(B, axis=1)
    den = an * bn
    out = np.full(len(A), np.nan, np.float32)
    good = den > eps
    if np.any(good):
        out[good] = np.sum(A[good] * B[good], axis=1) / den[good]
    return out


def safe_spearman(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 4:
        return float("nan"), float("nan"), int(good.sum())
    if np.nanstd(x[good]) < 1e-12 or np.nanstd(y[good]) < 1e-12:
        return float("nan"), float("nan"), int(good.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(x[good], y[good])
    return float(r), float(p), int(good.sum())


def safe_auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score, np.float64)
    good = np.isfinite(score) & np.isfinite(y.astype(float))
    y = y[good].astype(int)
    score = score[good]
    if len(y) < 4 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y, score):
    y = np.asarray(y)
    score = np.asarray(score, np.float64)
    good = np.isfinite(score) & np.isfinite(y.astype(float))
    y = y[good].astype(int)
    score = score[good]
    if len(y) < 4 or y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def safe_mean(values):
    """Mean over finite numeric values; NaN if none are finite."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def infer_source_layers(bundle):
    # Only parse the pre-[strategy] part.
    left = str(bundle).split("[", 1)[0]
    vals = [int(x) for x in re.findall(r"L?(\d+)", left)]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError(f"Could not parse source layers from bundle={bundle!r}")
    return vals


def to_bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float) != 0
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


# =============================================================================
# Load dynamic-run labels / targets
# =============================================================================

def load_run_tables(a, source_layers):
    run_dir = Path(a.run_dir)
    med_path = run_dir / "mediation_tokens.csv"
    sel_path = run_dir / "selected_tokens.csv"
    base_path = run_dir / "baseline.csv"
    gen_path = run_dir / "generation_conditions.csv"

    for p in [med_path, sel_path, base_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    med = pd.read_csv(med_path)
    sel = pd.read_csv(sel_path)
    baseline = pd.read_csv(base_path)

    for df in [med, sel, baseline]:
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
    med["delta_h_norm"] = pd.to_numeric(
        med["delta_h_norm"], errors="coerce"
    )
    med["grad_norm"] = pd.to_numeric(
        med["grad_norm"], errors="coerce"
    )
    med["relation"] = med["relation"].map(canon_rel)

    med = med[
        med["source_layer"].isin(source_layers)
    ].copy()

    # Selected K24 labels.
    sel["k"] = pd.to_numeric(sel["k"], errors="raise").astype(int)
    sel["source_layer"] = pd.to_numeric(
        sel["source_layer"], errors="raise"
    ).astype(int)
    sel["position"] = pd.to_numeric(
        sel["position"], errors="raise"
    ).astype(int)
    sel["rank"] = pd.to_numeric(sel["rank"], errors="raise").astype(int)
    sel["relation"] = sel["relation"].map(canon_rel)

    filt = (
        (sel["condition"].astype(str) == str(a.condition))
        & (
            sel["selection_strategy"].astype(str)
            == str(a.selection_strategy)
        )
        & (sel["source_bundle"].astype(str) == str(a.bundle))
        & (sel["k"] == int(a.k))
    )
    sel = sel[filt].copy()
    if not len(sel):
        raise RuntimeError(
            "No selected rows match: "
            f"condition={a.condition}, strategy={a.selection_strategy}, "
            f"bundle={a.bundle}, k={a.k}"
        )

    sel = (
        sel.sort_values(["sid", "rank"])
        .drop_duplicates(["sid", "source_layer", "position"], keep="first")
    )

    selected_key = set(
        zip(
            sel["sid"].astype(int),
            sel["source_layer"].astype(int),
            sel["position"].astype(int),
        )
    )
    selected_rank = {
        (int(r.sid), int(r.source_layer), int(r.position)): int(r.rank)
        for r in sel.itertuples()
    }

    med["selected_k24"] = [
        (int(sid), int(L), int(p)) in selected_key
        for sid, L, p in zip(
            med["sid"], med["source_layer"], med["position"]
        )
    ]
    med["selected_rank"] = [
        selected_rank.get((int(sid), int(L), int(p)), np.nan)
        for sid, L, p in zip(
            med["sid"], med["source_layer"], med["position"]
        )
    ]

    baseline["gt"] = baseline["gt"].map(canon_rel)
    baseline["baseline_correct"] = to_bool_series(
        baseline["baseline_correct"]
    )

    generation = None
    best_alpha = None
    if gen_path.exists():
        generation = pd.read_csv(gen_path)
        needed = {
            "sid",
            "condition",
            "selection_strategy",
            "source_bundle",
            "k",
            "alpha",
            "correct",
        }
        if needed.issubset(generation.columns):
            generation["sid"] = pd.to_numeric(
                generation["sid"], errors="raise"
            ).astype(int)
            generation["k"] = pd.to_numeric(
                generation["k"], errors="raise"
            ).astype(int)
            generation["alpha"] = pd.to_numeric(
                generation["alpha"], errors="coerce"
            )
            generation["correct"] = to_bool_series(generation["correct"])

            gf = (
                (
                    generation["condition"].astype(str)
                    == f"middle_{a.condition}"
                )
                & (
                    generation["selection_strategy"].astype(str)
                    == str(a.selection_strategy)
                )
                & (
                    generation["source_bundle"].astype(str)
                    == str(a.bundle)
                )
                & (generation["k"] == int(a.k))
            )
            generation = generation[gf].copy()

            if len(generation):
                best_alpha = float(
                    generation.groupby("alpha")["correct"]
                    .mean()
                    .sort_values(ascending=False)
                    .index[0]
                )
                generation = generation[
                    np.isclose(generation["alpha"], best_alpha)
                ].copy()
            else:
                generation = None

    # Restrict to samples with both candidate universe and K24 labels.
    valid_sids = sorted(
        set(med["sid"].unique())
        & set(sel["sid"].unique())
    )
    if a.max_eval_samples and a.max_eval_samples > 0:
        rng = np.random.default_rng(a.seed)
        if a.max_eval_samples < len(valid_sids):
            valid_sids = sorted(
                rng.choice(
                    valid_sids,
                    size=a.max_eval_samples,
                    replace=False,
                ).tolist()
            )

    med = med[med["sid"].isin(valid_sids)].copy()
    sel = sel[sel["sid"].isin(valid_sids)].copy()
    baseline = baseline[baseline["sid"].isin(valid_sids)].copy()
    if generation is not None:
        generation = generation[
            generation["sid"].isin(valid_sids)
        ].copy()

    med = med.sort_values(
        ["sid", "source_layer", "position"]
    ).reset_index(drop=True)

    return med, sel, baseline, generation, best_alpha, valid_sids


# =============================================================================
# Rich forward capture
# =============================================================================

class RichCapture:
    def __init__(self, decoder_layers, hidden_layers, module_layers):
        self.hidden = {}
        self.mlp = {}
        self.attn_out = {}
        self.handles = []

        for L in hidden_layers:
            self.handles.append(
                decoder_layers[L].register_forward_hook(
                    self._hidden_hook(L)
                )
            )

        for L in module_layers:
            layer = decoder_layers[L]

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                self.handles.append(
                    mlp.register_forward_hook(
                        self._module_hook(self.mlp, L)
                    )
                )

            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                self.handles.append(
                    attn.register_forward_hook(
                        self._module_hook(self.attn_out, L)
                    )
                )

    def _hidden_hook(self, L):
        def hook(_m, _inp, out):
            x = first_tensor(out)
            self.hidden[L] = (
                x.detach().float().cpu().numpy().astype(np.float32)
            )
            return out
        return hook

    def _module_hook(self, store, L):
        def hook(_m, _inp, out):
            try:
                x = first_tensor(out)
                store[L] = (
                    x.detach().float().cpu().numpy().astype(np.float32)
                )
            except Exception:
                pass
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


def extract_decoder_attentions(model_output):
    # Most Qwen2/Qwen2.5 VL causal-LM outputs expose .attentions directly.
    for obj in [
        model_output,
        getattr(model_output, "language_model_output", None),
        getattr(model_output, "model_output", None),
    ]:
        if obj is None:
            continue
        att = getattr(obj, "attentions", None)
        if att is not None:
            return att
    return None


def logits_summary(logits_last):
    x = logits_last.detach().float()
    top2 = torch.topk(x, k=2).values
    logp = torch.log_softmax(x, dim=-1)
    p = torch.exp(logp)
    entropy = -(p * logp).sum()
    return {
        "next_logit_top1": float(top2[0].cpu()),
        "next_logit_margin": float((top2[0] - top2[1]).cpu()),
        "next_entropy": float(entropy.cpu()),
        "next_max_prob": float(torch.exp(logp.max()).cpu()),
    }


@torch.inference_mode()
def rich_forward(
    model,
    decoder_layers,
    batch,
    hidden_layers,
    module_layers,
    requested_attn_layers,
    with_attention_weights,
):
    cap = RichCapture(
        decoder_layers,
        hidden_layers,
        module_layers,
    )
    out = None
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = bool(with_attention_weights)

        out = model(**kw)

        last_attn = {}
        attn_entropy = {}

        if with_attention_weights:
            attentions = extract_decoder_attentions(out)
            if attentions is not None:
                for L in requested_attn_layers:
                    if L < 0 or L >= len(attentions):
                        continue
                    A = attentions[L]
                    if A is None or not torch.is_tensor(A):
                        continue

                    # [B, heads, query, key] -> [heads, key] for the last query.
                    row = (
                        A[0, :, -1, :]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    last_attn[L] = row

                    q = np.clip(row, 1e-12, 1.0)
                    ent = -(q * np.log(q)).sum(axis=1)
                    attn_entropy[L] = {
                        "mean": float(np.mean(ent)),
                        "std": float(np.std(ent)),
                    }

        logits = getattr(out, "logits", None)
        lsum = {}
        if logits is not None:
            lsum = logits_summary(logits[0, -1])

        return {
            "hidden": cap.hidden,
            "mlp": cap.mlp,
            "attn_out": cap.attn_out,
            "last_attn": last_attn,
            "attn_entropy": attn_entropy,
            "logits_summary": lsum,
        }
    finally:
        cap.close()
        del out


# =============================================================================
# Feature construction
# =============================================================================

def row_norm(X):
    return np.linalg.norm(X, axis=1).astype(np.float32)


def add_module_features(
    feat,
    positions,
    real_store,
    gray_store,
    L,
    prefix,
):
    if L not in real_store or L not in gray_store:
        return
    R = real_store[L][0]
    G = gray_store[L][0]
    pos = np.asarray(positions, dtype=int)
    good = (pos >= 0) & (pos < min(len(R), len(G)))
    if not np.all(good):
        return

    R = R[pos]
    G = G[pos]
    D = R - G

    rn = row_norm(R)
    gn = row_norm(G)
    dn = row_norm(D)

    feat[f"{prefix}_real_norm"] = rn
    feat[f"{prefix}_gray_norm"] = gn
    feat[f"{prefix}_delta_norm"] = dn
    feat[f"{prefix}_relative_delta"] = dn / (rn + 1e-8)
    feat[f"{prefix}_real_gray_cos"] = cosine_rows(R, G)
    feat[f"{prefix}_norm_change"] = rn - gn


def role_centroid_cos_features(D, broad_categories):
    out = {}
    cats = np.asarray(broad_categories, dtype=object)

    for role in [
        "visual",
        "subject",
        "reference",
        "relation_words",
        "other_text",
    ]:
        mask = cats == role
        name = f"delta_cos_{role}_centroid"

        if mask.sum() == 0:
            out[name] = np.full(len(D), np.nan, np.float32)
            continue

        centroid = D[mask].mean(axis=0)

        # Leave-one-out centroid for members of the same role to avoid a
        # trivial self-similarity boost.
        vals = cosine_rows_to_vec(D, centroid)

        if mask.sum() > 1:
            total = D[mask].sum(axis=0)
            idxs = np.where(mask)[0]
            for idx in idxs:
                loo = (total - D[idx]) / (mask.sum() - 1)
                vals[idx] = cosine_rows_to_vec(
                    D[idx : idx + 1], loo
                )[0]

        out[name] = vals

    return out


def max_relation_word_cos(D, broad_categories):
    cats = np.asarray(broad_categories, dtype=object)
    rel_idx = np.where(cats == "relation_words")[0]
    out = np.full(len(D), np.nan, np.float32)

    if len(rel_idx) == 0:
        return out

    norms = row_norm(D)
    Dn = D / (norms[:, None] + 1e-8)
    rel = Dn[rel_idx]

    sims = Dn @ rel.T

    # For a relation-word candidate, remove its exact self-comparison.
    index_to_col = {int(idx): j for j, idx in enumerate(rel_idx)}
    for i in range(len(D)):
        row = sims[i].copy()
        if i in index_to_col and len(row) > 1:
            row[index_to_col[i]] = -np.inf
        out[i] = np.max(row) if len(row) else np.nan

    return out


def global_selected_centroid_features(
    D_by_layer,
    selected_masks_by_layer,
):
    vecs = []
    locs = []

    for L, D in D_by_layer.items():
        mask = selected_masks_by_layer[L]
        for local_i in np.where(mask)[0]:
            vecs.append(D[local_i])
            locs.append((L, int(local_i)))

    if not vecs:
        return {
            L: np.full(len(D), np.nan, np.float32)
            for L, D in D_by_layer.items()
        }

    V = np.stack(vecs)
    total = V.sum(axis=0)
    n = len(V)

    selected_loc_set = set(locs)
    result = {}

    for L, D in D_by_layer.items():
        vals = cosine_rows_to_vec(D, total / n)
        if n > 1:
            for local_i in range(len(D)):
                if (L, local_i) in selected_loc_set:
                    loo = (total - D[local_i]) / (n - 1)
                    vals[local_i] = cosine_rows_to_vec(
                        D[local_i : local_i + 1], loo
                    )[0]
        result[L] = vals

    return result


def attention_features_for_positions(
    real_attn,
    gray_attn,
    positions,
    prefix,
    head_scan,
):
    out = {}
    if real_attn is None or gray_attn is None:
        return out

    # [heads, key]
    max_key = min(real_attn.shape[1], gray_attn.shape[1])
    pos = np.asarray(positions, dtype=int)
    if np.any(pos < 0) or np.any(pos >= max_key):
        return out

    R = real_attn[:, pos].T  # [tokens, heads]
    G = gray_attn[:, pos].T
    D = R - G

    out[f"{prefix}_real_mean"] = R.mean(axis=1).astype(np.float32)
    out[f"{prefix}_real_max"] = R.max(axis=1).astype(np.float32)
    out[f"{prefix}_real_std_heads"] = R.std(axis=1).astype(np.float32)
    out[f"{prefix}_gray_mean"] = G.mean(axis=1).astype(np.float32)
    out[f"{prefix}_delta_mean"] = D.mean(axis=1).astype(np.float32)
    out[f"{prefix}_abs_delta_mean"] = np.abs(D).mean(axis=1).astype(np.float32)
    out[f"{prefix}_max_positive_delta"] = D.max(axis=1).astype(np.float32)

    if head_scan:
        for h in range(R.shape[1]):
            out[f"{prefix}_head{h:02d}_real"] = R[:, h].astype(np.float32)
            out[f"{prefix}_head{h:02d}_delta"] = D[:, h].astype(np.float32)

    return out


def build_sample_features(
    sid_med,
    source_layers,
    target_layers,
    real_cap,
    gray_cap,
    with_attention_weights,
    attention_head_scan,
):
    """
    Return:
      token_feature_df
      sample_feature_dict
    """
    chunks = []
    D_by_layer = {}
    selected_masks_by_layer = {}
    local_frames = {}

    # First pass: hidden-state deltas for all source-layer candidates.
    for L in source_layers:
        g = (
            sid_med[sid_med["source_layer"] == L]
            .sort_values("position")
            .copy()
        )
        if not len(g):
            continue

        pos = g["position"].astype(int).to_numpy()
        Hreal = real_cap["hidden"][L][0]
        Hgray = gray_cap["hidden"][L][0]
        if pos.max() >= min(len(Hreal), len(Hgray)):
            raise RuntimeError(
                f"sid={int(g['sid'].iloc[0])} L{L}: candidate position "
                f"{pos.max()} exceeds captured sequence length "
                f"{min(len(Hreal), len(Hgray))}"
            )

        R = Hreal[pos]
        G = Hgray[pos]
        D = (R - G).astype(np.float32)

        D_by_layer[L] = D
        selected_masks_by_layer[L] = g["selected_k24"].astype(bool).to_numpy()
        local_frames[L] = g

    selected_centroid_cos = global_selected_centroid_features(
        D_by_layer,
        selected_masks_by_layer,
    )

    # Global current-sample delta centroid: relation-free.
    all_D = np.concatenate(
        [D_by_layer[L] for L in source_layers if L in D_by_layer],
        axis=0,
    )
    sample_delta_centroid = all_D.mean(axis=0)

    sample_feat = {
        "sid": int(sid_med["sid"].iloc[0]),
    }
    sample_feat.update(
        {
            f"real_{k}": v
            for k, v in real_cap["logits_summary"].items()
        }
    )
    sample_feat.update(
        {
            f"gray_{k}": v
            for k, v in gray_cap["logits_summary"].items()
        }
    )

    # Last-state sample features.
    for L in sorted(set(source_layers + target_layers)):
        if L not in real_cap["hidden"] or L not in gray_cap["hidden"]:
            continue
        hr = real_cap["hidden"][L][0, -1]
        hg = gray_cap["hidden"][L][0, -1]
        d = hr - hg
        sample_feat[f"last_delta_norm_L{L}"] = float(np.linalg.norm(d))
        sample_feat[f"last_real_norm_L{L}"] = float(np.linalg.norm(hr))
        sample_feat[f"last_real_gray_cos_L{L}"] = float(
            cosine_rows(
                hr[None, :], hg[None, :]
            )[0]
        )

        if with_attention_weights:
            er = real_cap["attn_entropy"].get(L, {})
            eg = gray_cap["attn_entropy"].get(L, {})
            if "mean" in er:
                sample_feat[f"last_attn_entropy_real_L{L}"] = er["mean"]
            if "mean" in eg:
                sample_feat[f"last_attn_entropy_gray_L{L}"] = eg["mean"]
            if "mean" in er and "mean" in eg:
                sample_feat[f"last_attn_entropy_delta_L{L}"] = (
                    er["mean"] - eg["mean"]
                )

    # Late target delta vectors for token-level similarities.
    late_delta = {}
    late_real_last = {}
    for T in target_layers:
        if T in real_cap["hidden"] and T in gray_cap["hidden"]:
            late_real_last[T] = real_cap["hidden"][T][0, -1]
            late_delta[T] = (
                real_cap["hidden"][T][0, -1]
                - gray_cap["hidden"][T][0, -1]
            ).astype(np.float32)

    for L in source_layers:
        if L not in D_by_layer:
            continue

        g = local_frames[L].copy()
        pos = g["position"].astype(int).to_numpy()

        Hreal_all = real_cap["hidden"][L][0]
        Hgray_all = gray_cap["hidden"][L][0]
        R = Hreal_all[pos]
        G = Hgray_all[pos]
        D = D_by_layer[L]

        feat = pd.DataFrame(index=g.index)

        rn = row_norm(R)
        gn = row_norm(G)
        dn = row_norm(D)

        feat["hidden_real_norm"] = rn
        feat["hidden_gray_norm"] = gn
        feat["hidden_delta_norm_recomputed"] = dn
        feat["hidden_relative_delta"] = dn / (rn + 1e-8)
        feat["hidden_real_gray_cos"] = cosine_rows(R, G)
        feat["hidden_norm_change"] = rn - gn

        # Within-sample-layer standardized magnitudes/ranks.
        feat["hidden_delta_norm_z_within_layer"] = (
            (dn - dn.mean()) / (dn.std() + 1e-8)
        )
        feat["hidden_delta_norm_percentile_within_layer"] = (
            pd.Series(dn)
            .rank(pct=True)
            .to_numpy()
            .astype(np.float32)
        )

        # Relation-free similarity to current sample's own mean delta.
        feat["delta_cos_sample_global_centroid"] = cosine_rows_to_vec(
            D, sample_delta_centroid
        )

        # Circular diagnostic: does the TRUE K24 form a coherent vector cluster?
        feat["delta_cos_true_k24_centroid_ORACLE_DIAGNOSTIC"] = (
            selected_centroid_cos[L]
        )

        # Same-layer last state.
        source_last_delta = (
            Hreal_all[-1] - Hgray_all[-1]
        ).astype(np.float32)
        feat["delta_cos_source_last_delta"] = cosine_rows_to_vec(
            D, source_last_delta
        )
        feat["real_cos_source_last_real"] = cosine_rows_to_vec(
            R, Hreal_all[-1]
        )

        # Similarity to sample's own late Real-Gray last-state displacement.
        late_cos_cols = []
        late_dot_cols = []
        for T in target_layers:
            if T not in late_delta:
                continue
            c = cosine_rows_to_vec(D, late_delta[T])
            dot = (D @ late_delta[T]).astype(np.float32)
            feat[f"delta_cos_late_last_delta_L{T}"] = c
            feat[f"delta_dot_late_last_delta_L{T}"] = dot
            late_cos_cols.append(c)
            late_dot_cols.append(dot)

        if late_cos_cols:
            C = np.stack(late_cos_cols, axis=1)
            valid = np.isfinite(C)
            nvalid = valid.sum(axis=1)

            # Warning-free row-wise reductions. Some candidate tokens can have
            # zero Real-Gray displacement, making cosine undefined against all
            # late targets. Preserve those rows as NaN instead of emitting an
            # "All-NaN slice" warning for every sample.
            csum = np.where(valid, C, 0.0).sum(axis=1)
            cmean = np.full(C.shape[0], np.nan, dtype=np.float32)
            np.divide(
                csum,
                nvalid,
                out=cmean,
                where=nvalid > 0,
            )

            cmax_src = np.where(valid, C, -np.inf)
            cmax = cmax_src.max(axis=1).astype(np.float32)
            cmax[nvalid == 0] = np.nan

            cabs_src = np.where(valid, np.abs(C), -np.inf)
            cmaxabs = cabs_src.max(axis=1).astype(np.float32)
            cmaxabs[nvalid == 0] = np.nan

            feat["delta_cos_late_mean"] = cmean
            feat["delta_cos_late_max"] = cmax
            feat["delta_cos_late_maxabs"] = cmaxabs
            feat["delta_cos_late_nvalid"] = nvalid.astype(np.float32)

        if late_dot_cols:
            X = np.stack(late_dot_cols, axis=1)
            valid_x = np.isfinite(X)
            nx = valid_x.sum(axis=1)
            xsum = np.where(valid_x, X, 0.0).sum(axis=1)
            xmean = np.full(X.shape[0], np.nan, dtype=np.float32)
            np.divide(
                xsum,
                nx,
                out=xmean,
                where=nx > 0,
            )
            feat["delta_dot_late_mean"] = xmean

        # Role-centroid similarities at the same source layer.
        role_feats = role_centroid_cos_features(
            D,
            g["broad_category"].astype(str).to_numpy(),
        )
        for name, vals in role_feats.items():
            feat[name] = vals

        feat["delta_cos_any_relation_word_max"] = max_relation_word_cos(
            D,
            g["broad_category"].astype(str).to_numpy(),
        )

        # MLP and attention-output vector activity.
        add_module_features(
            feat,
            pos,
            real_cap["mlp"],
            gray_cap["mlp"],
            L,
            "mlp_out",
        )
        add_module_features(
            feat,
            pos,
            real_cap["attn_out"],
            gray_cap["attn_out"],
            L,
            "attn_out",
        )

        # Attention weights: last query -> candidate token.
        if with_attention_weights:
            same = attention_features_for_positions(
                real_cap["last_attn"].get(L),
                gray_cap["last_attn"].get(L),
                pos,
                "last_attn_same",
                attention_head_scan,
            )
            for name, vals in same.items():
                feat[name] = vals

            if (L + 1) in real_cap["last_attn"]:
                nxt = attention_features_for_positions(
                    real_cap["last_attn"].get(L + 1),
                    gray_cap["last_attn"].get(L + 1),
                    pos,
                    "last_attn_next",
                    False,
                )
                for name, vals in nxt.items():
                    feat[name] = vals

            # Average last-query attention to this position across late targets.
            late_real = []
            late_gray = []
            for T in target_layers:
                Ar = real_cap["last_attn"].get(T)
                Ag = gray_cap["last_attn"].get(T)
                if Ar is None or Ag is None:
                    continue
                if pos.max() >= min(Ar.shape[1], Ag.shape[1]):
                    continue
                late_real.append(Ar[:, pos].T.mean(axis=1))
                late_gray.append(Ag[:, pos].T.mean(axis=1))

            if late_real:
                LR = np.stack(late_real, axis=1)
                LG = np.stack(late_gray, axis=1)
                feat["last_attn_late_real_mean"] = LR.mean(axis=1)
                feat["last_attn_late_gray_mean"] = LG.mean(axis=1)
                feat["last_attn_late_delta_mean"] = (
                    LR - LG
                ).mean(axis=1)
                feat["last_attn_late_abs_delta_mean"] = (
                    np.abs(LR - LG)
                ).mean(axis=1)

        # Join identifying / oracle target columns from mediation_tokens.csv.
        base_cols = [
            "sid",
            "relation",
            "source_layer",
            "position",
            "token",
            "category",
            "broad_category",
            "mediation",
            "abs_mediation",
            "delta_h_norm",
            "grad_norm",
            "positive_rank",
            "selected_k24",
            "selected_rank",
        ]
        base_cols = [c for c in base_cols if c in g.columns]

        out = pd.concat(
            [
                g[base_cols].reset_index(drop=True),
                feat.reset_index(drop=True),
            ],
            axis=1,
        )
        chunks.append(out)

    token_df = pd.concat(chunks, ignore_index=True)
    return token_df, sample_feat


# =============================================================================
# Predictor evaluation
# =============================================================================

META_COLS = {
    "sid",
    "relation",
    "source_layer",
    "position",
    "token",
    "category",
    "broad_category",
    "selected_k24",
    "selected_rank",
    "positive_rank",
    "mediation",
    "abs_mediation",
}


def feature_kind(name):
    if name == "grad_norm":
        return "oracle_gradient_diagnostic"
    if "TRUE_K24" in name.upper() or "ORACLE_DIAGNOSTIC" in name.upper():
        return "circular_k24_diagnostic"
    return "relation_free_current_forward"


def select_global_unique_topk(g, feature, k, ascending=False):
    q = g[
        np.isfinite(pd.to_numeric(g[feature], errors="coerce"))
    ].copy()
    if not len(q):
        return set()

    q = q.sort_values(
        feature,
        ascending=ascending,
    )

    chosen = set()
    used_pos = set()

    for r in q.itertuples():
        pos = int(r.position)
        if pos in used_pos:
            continue
        used_pos.add(pos)
        chosen.add((int(r.source_layer), pos))
        if len(chosen) >= k:
            break

    return chosen


def evaluate_feature(token_df, feature, k):
    x = pd.to_numeric(
        token_df[feature],
        errors="coerce",
    ).to_numpy(np.float64)
    y = token_df["selected_k24"].astype(int).to_numpy()
    m = pd.to_numeric(
        token_df["mediation"],
        errors="coerce",
    ).to_numpy(np.float64)

    rho_m, p_m, n_m = safe_spearman(x, m)

    auc = safe_auc(y, x)
    ap = safe_ap(y, x)
    auc_low = safe_auc(y, -x)
    ap_low = safe_ap(y, -x)

    per_sample_rho = []
    per_sample_auc = []
    recall_high = []
    recall_low = []
    jacc_high = []
    jacc_low = []

    for sid, g in token_df.groupby("sid", sort=False):
        gx = pd.to_numeric(g[feature], errors="coerce").to_numpy(np.float64)
        gm = pd.to_numeric(g["mediation"], errors="coerce").to_numpy(np.float64)
        gy = g["selected_k24"].astype(int).to_numpy()

        rho, _, _ = safe_spearman(gx, gm)
        if np.isfinite(rho):
            per_sample_rho.append(rho)

        av = safe_auc(gy, gx)
        if np.isfinite(av):
            per_sample_auc.append(av)

        truth = set(
            (int(r.source_layer), int(r.position))
            for r in g[g["selected_k24"]].itertuples()
        )
        if not truth:
            continue

        hi = select_global_unique_topk(
            g, feature, k, ascending=False
        )
        lo = select_global_unique_topk(
            g, feature, k, ascending=True
        )

        for pred, recs, jacs in [
            (hi, recall_high, jacc_high),
            (lo, recall_low, jacc_low),
        ]:
            inter = len(truth & pred)
            recs.append(inter / len(truth))
            union = len(truth | pred)
            jacs.append(inter / union if union else np.nan)

    hi_recall = safe_mean(recall_high)
    lo_recall = safe_mean(recall_low)
    hi_jacc = safe_mean(jacc_high)
    lo_jacc = safe_mean(jacc_low)

    if (
        np.isfinite(hi_recall)
        and np.isfinite(lo_recall)
        and lo_recall > hi_recall
    ):
        best_orientation = "low"
        best_recall = lo_recall
        best_jacc = lo_jacc
    else:
        best_orientation = "high"
        best_recall = hi_recall
        best_jacc = hi_jacc

    return {
        "feature": feature,
        "feature_kind": feature_kind(feature),
        "N_rows": int(np.isfinite(x).sum()),
        "selected_prevalence": float(y.mean()),
        "spearman_vs_mediation": rho_m,
        "spearman_p": p_m,
        "mean_within_sample_spearman_vs_mediation": safe_mean(
            per_sample_rho
        ),
        "pooled_AUROC_selected_high": auc,
        "pooled_AP_selected_high": ap,
        "pooled_AUROC_selected_low": auc_low,
        "pooled_AP_selected_low": ap_low,
        "mean_within_sample_AUROC_high": safe_mean(per_sample_auc),
        "mean_recall_at_k_high": hi_recall,
        "mean_jaccard_at_k_high": hi_jacc,
        "mean_recall_at_k_low": lo_recall,
        "mean_jaccard_at_k_low": lo_jacc,
        "best_orientation": best_orientation,
        "best_mean_recall_at_k": best_recall,
        "best_mean_jaccard_at_k": best_jacc,
    }


def feature_columns(token_df):
    cols = []
    for c in token_df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(token_df[c]):
            if token_df[c].notna().sum() >= 100:
                cols.append(c)
    return cols


def relation_stratified_feature_eval(token_df, features, k):
    rows = []
    for rel in REL_ORDER:
        g = token_df[token_df["relation"] == rel]
        if not len(g):
            continue
        for f in features:
            r = evaluate_feature(g, f, k)
            r["relation"] = rel
            rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# Sample-level K24 composition correlations
# =============================================================================

def add_k24_composition_targets(sample_df, selected_df, source_layers):
    rows = []
    for sid, g in selected_df.groupby("sid"):
        row = {"sid": int(sid)}
        N = len(g)
        for L in source_layers:
            for role in [
                "visual",
                "subject",
                "reference",
                "relation_words",
                "other_text",
            ]:
                n = int(
                    (
                        (g["source_layer"] == L)
                        & (g["broad_category"] == role)
                    ).sum()
                )
                key = f"L{L}_{role}"
                row[f"k24_count_{key}"] = n
                row[f"k24_frac_{key}"] = n / N if N else np.nan

        row["k24_mean_mediation"] = float(g["mediation"].mean())
        row["k24_sum_mediation"] = float(g["mediation"].sum())
        row["k24_mean_delta_norm"] = float(g["delta_h_norm"].mean())
        rows.append(row)

    comp = pd.DataFrame(rows)
    return sample_df.merge(comp, on="sid", how="left")


def sample_feature_correlations(sample_df):
    targets = [
        c
        for c in sample_df.columns
        if c.startswith("k24_")
    ]

    excluded = {
        "sid",
        "relation",
        "baseline_prediction",
        "edited_prediction",
        "repair_outcome",
    } | set(targets)

    features = [
        c
        for c in sample_df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(sample_df[c])
        and sample_df[c].nunique(dropna=True) > 2
    ]

    rows = []
    for f in features:
        for t in targets:
            rho, p, n = safe_spearman(
                pd.to_numeric(sample_df[f], errors="coerce"),
                pd.to_numeric(sample_df[t], errors="coerce"),
            )
            if np.isfinite(rho):
                rows.append({
                    "sample_feature": f,
                    "k24_target": t,
                    "spearman": rho,
                    "abs_spearman": abs(rho),
                    "p": p,
                    "N": n,
                })

    return pd.DataFrame(rows).sort_values(
        "abs_spearman",
        ascending=False,
    )


def sample_outcome_predictors(sample_df):
    features = [
        c
        for c in sample_df.columns
        if pd.api.types.is_numeric_dtype(sample_df[c])
        and not c.startswith("k24_")
        and c
        not in {
            "sid",
            "baseline_correct",
            "edited_correct",
            "repair_w2c",
            "damage_c2w",
        }
        and sample_df[c].nunique(dropna=True) > 2
    ]

    rows = []

    # Can sample-level quantities predict repair among baseline-wrong samples?
    wrong = sample_df[sample_df["baseline_correct"] == False]
    if "repair_w2c" in wrong.columns:
        y = wrong["repair_w2c"].astype(int)
        for f in features:
            x = pd.to_numeric(wrong[f], errors="coerce")
            rows.append({
                "outcome": "W2C_given_baseline_wrong",
                "feature": f,
                "AUROC_high": safe_auc(y, x),
                "AUROC_low": safe_auc(y, -x),
                "spearman": safe_spearman(x, y)[0],
            })

    correct = sample_df[sample_df["baseline_correct"] == True]
    if "damage_c2w" in correct.columns:
        y = correct["damage_c2w"].astype(int)
        if y.sum() > 0:
            for f in features:
                x = pd.to_numeric(correct[f], errors="coerce")
                rows.append({
                    "outcome": "C2W_given_baseline_correct",
                    "feature": f,
                    "AUROC_high": safe_auc(y, x),
                    "AUROC_low": safe_auc(y, -x),
                    "spearman": safe_spearman(x, y)[0],
                })

    return pd.DataFrame(rows)


# =============================================================================
# Summary text
# =============================================================================

def render_summary(
    overall,
    per_rel,
    sample_corr,
    outcome_df,
    baseline_recall,
    a,
):
    lines = []
    lines.append("=" * 110)
    lines.append("K24 PREDICTIVE CORRELATES")
    lines.append("=" * 110)
    lines.append(
        "Question: can a relation-free quantity from the current sample recover "
        "the oracle writer-guided K24?"
    )
    lines.append(
        f"Random Top-{a.k} overlap baseline (approx prevalence among unique "
        f"candidate positions): {baseline_recall:.4f}"
    )
    lines.append("")

    rf = overall[
        overall["feature_kind"] == "relation_free_current_forward"
    ].copy()

    lines.append("Top relation-free features by K24 recall:")
    q = rf.sort_values(
        "best_mean_recall_at_k",
        ascending=False,
    ).head(a.top_summary_n)
    for r in q.itertuples():
        lines.append(
            f"  {r.feature:<48s} "
            f"orient={r.best_orientation:<4s} "
            f"recall@{a.k}={r.best_mean_recall_at_k:.3f} "
            f"J={r.best_mean_jaccard_at_k:.3f} "
            f"within-rho(M)={r.mean_within_sample_spearman_vs_mediation:+.3f} "
            f"AUC={r.mean_within_sample_AUROC_high:.3f}"
        )

    lines.append("")
    lines.append("Top relation-free features by |within-sample Spearman vs mediation|:")
    q = rf.assign(
        abs_wrho=np.abs(
            rf["mean_within_sample_spearman_vs_mediation"]
        )
    ).sort_values("abs_wrho", ascending=False).head(a.top_summary_n)
    for r in q.itertuples():
        lines.append(
            f"  {r.feature:<48s} "
            f"within-rho(M)={r.mean_within_sample_spearman_vs_mediation:+.3f} "
            f"recall@{a.k}={r.best_mean_recall_at_k:.3f}"
        )

    oracle_diag = overall[
        overall["feature_kind"] != "relation_free_current_forward"
    ].copy()
    if len(oracle_diag):
        lines.append("")
        lines.append("Diagnostic/non-deployable features:")
        q = oracle_diag.sort_values(
            "best_mean_recall_at_k",
            ascending=False,
        ).head(15)
        for r in q.itertuples():
            lines.append(
                f"  [{r.feature_kind}] {r.feature:<40s} "
                f"recall@{a.k}={r.best_mean_recall_at_k:.3f} "
                f"within-rho(M)={r.mean_within_sample_spearman_vs_mediation:+.3f}"
            )

    if len(per_rel):
        lines.append("")
        lines.append("Best relation-free feature separately by relation:")
        for rel in REL_ORDER:
            g = per_rel[
                (per_rel["relation"] == rel)
                & (
                    per_rel["feature_kind"]
                    == "relation_free_current_forward"
                )
            ].sort_values(
                "best_mean_recall_at_k",
                ascending=False,
            )
            if len(g):
                r = g.iloc[0]
                lines.append(
                    f"  {rel:<5s}: {r['feature']} | "
                    f"recall@{a.k}={r['best_mean_recall_at_k']:.3f}, "
                    f"within-rho(M)="
                    f"{r['mean_within_sample_spearman_vs_mediation']:+.3f}"
                )

    if len(sample_corr):
        lines.append("")
        lines.append("Strongest sample-state -> K24-composition correlations:")
        for r in sample_corr.head(25).itertuples():
            lines.append(
                f"  {r.sample_feature:<40s} -> {r.k24_target:<35s} "
                f"rho={r.spearman:+.3f}"
            )

    if len(outcome_df):
        lines.append("")
        lines.append("Best sample-level correlates of repairability:")
        for outcome in outcome_df["outcome"].unique():
            g = outcome_df[outcome_df["outcome"] == outcome].copy()
            g["best_auc"] = g[["AUROC_high", "AUROC_low"]].max(axis=1)
            g = g.sort_values("best_auc", ascending=False).head(10)
            lines.append(f"  {outcome}:")
            for r in g.itertuples():
                lines.append(
                    f"    {r.feature:<42s} bestAUC={r.best_auc:.3f} "
                    f"rho={r.spearman:+.3f}"
                )

    lines.append("")
    lines.append(
        "Interpretation: correlation/recovery does not establish necessity. "
        "A useful next-stage selector candidate should be relation-free, show "
        "within-sample correlation with mediation, and recover substantially "
        "more true K24 tokens than the random-overlap baseline across relations."
    )

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = infer_source_layers(a.bundle)
    target_layers = parse_ints(a.target_layers)

    outdir = Path(a.output_dir)
    cache_dir = outdir / "feature_chunks"

    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    med, sel, baseline, generation, best_alpha, valid_sids = load_run_tables(
        a,
        source_layers,
    )

    print("=" * 120)
    print("DYNAMIC K24 PREDICTIVE-CORRELATE ANALYSIS")
    print("=" * 120)
    print(f"N samples={len(valid_sids)}")
    print(f"source layers={source_layers}")
    print(f"target layers={target_layers}")
    print(f"K={a.k} | bundle={a.bundle}")
    print(f"attention weights={a.with_attention_weights}")
    print(f"attention head scan={a.attention_head_scan}")
    print(f"best dynamic-K24 alpha in run={best_alpha}")
    print()

    # Build sample metadata from the exact run SIDs.
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
    rec_by_sid = {int(r.sid): r for r in records}

    meta_by_sid = {}
    for sid in valid_sids:
        if sid not in prompts or sid not in rec_by_sid:
            raise RuntimeError(f"sid={sid} missing from prompts or dataset")
        p = prompts[sid]
        meta_by_sid[sid] = {
            "sid": sid,
            "question_text": str(p["question_text"]),
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
        }

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )

    model_cls = getattr(transformers, spec.model_class)
    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    model = None
    attention_available_any = False

    try:
        model = model_cls.from_pretrained(
            spec.repo_id,
            **load_kw,
        )
        model.eval()
        base.configure_processor(model, processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        hidden_layers = sorted(
            set(source_layers + target_layers)
        )
        module_layers = list(source_layers)
        requested_attn_layers = sorted(
            {
                L
                for L in (
                    source_layers
                    + [x + 1 for x in source_layers]
                    + target_layers
                )
                if 0 <= L < n_layers
            }
        )

        for L in hidden_layers:
            if L < 0 or L >= n_layers:
                raise ValueError(
                    f"Layer L{L} invalid; model has {n_layers} decoder layers"
                )

        sample_rows = []

        for sid in tqdm(valid_sids, desc="Collect relation-free features"):
            chunk_path = cache_dir / f"sid_{sid}.pkl.gz"
            sample_path = cache_dir / f"sid_{sid}_sample.json"

            if chunk_path.exists() and sample_path.exists():
                try:
                    sample_rows.append(
                        json.loads(sample_path.read_text(encoding="utf-8"))
                    )
                    continue
                except Exception:
                    pass

            m = meta_by_sid[sid]
            real = gray = rb = gb = None

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                device = torch.device(a.device)
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

                real_cap = rich_forward(
                    model,
                    decoder_layers,
                    rb,
                    hidden_layers,
                    module_layers,
                    requested_attn_layers,
                    a.with_attention_weights,
                )
                gray_cap = rich_forward(
                    model,
                    decoder_layers,
                    gb,
                    hidden_layers,
                    module_layers,
                    requested_attn_layers,
                    a.with_attention_weights,
                )

                if real_cap["last_attn"]:
                    attention_available_any = True

                sid_med = med[med["sid"] == sid].copy()
                token_chunk, sample_feat = build_sample_features(
                    sid_med,
                    source_layers,
                    target_layers,
                    real_cap,
                    gray_cap,
                    a.with_attention_weights,
                    a.attention_head_scan,
                )

                token_chunk.to_pickle(
                    chunk_path,
                    compression="gzip",
                )
                sample_path.write_text(
                    json.dumps(sample_feat, indent=2),
                    encoding="utf-8",
                )
                sample_rows.append(sample_feat)

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

        if a.with_attention_weights and not attention_available_any:
            print(
                "WARNING: model did not expose decoder attention matrices. "
                "Hidden/MLP/attention-output-vector features were still collected."
            )

        # Concatenate feature chunks.
        token_parts = []
        for sid in valid_sids:
            p = cache_dir / f"sid_{sid}.pkl.gz"
            if not p.exists():
                raise RuntimeError(f"Missing feature cache: {p}")
            token_parts.append(pd.read_pickle(p, compression="gzip"))
        token_df = pd.concat(token_parts, ignore_index=True)

        sample_df = pd.DataFrame(sample_rows).drop_duplicates(
            "sid", keep="last"
        )

        # Add baseline / repair outcome.
        bcols = [
            c
            for c in [
                "sid",
                "gt",
                "baseline_prediction",
                "baseline_correct",
            ]
            if c in baseline.columns
        ]
        sample_df = sample_df.merge(
            baseline[bcols],
            on="sid",
            how="left",
        )
        sample_df["relation"] = sample_df["gt"].map(canon_rel)
        sample_df["baseline_correct"] = to_bool_series(
            sample_df["baseline_correct"]
        )

        if generation is not None and len(generation):
            gcols = ["sid", "prediction", "correct"]
            gg = generation[gcols].copy()
            gg = gg.rename(
                columns={
                    "prediction": "edited_prediction",
                    "correct": "edited_correct",
                }
            )
            sample_df = sample_df.merge(
                gg,
                on="sid",
                how="left",
            )
            sample_df["edited_correct"] = to_bool_series(
                sample_df["edited_correct"]
            )
            sample_df["repair_w2c"] = (
                (~sample_df["baseline_correct"])
                & sample_df["edited_correct"]
            )
            sample_df["damage_c2w"] = (
                sample_df["baseline_correct"]
                & (~sample_df["edited_correct"])
            )

        # Merge mediation back into selected rows so composition targets are exact.
        sel_comp = sel.merge(
            med[
                [
                    "sid",
                    "source_layer",
                    "position",
                    "broad_category",
                    "mediation",
                    "delta_h_norm",
                ]
            ],
            on=["sid", "source_layer", "position"],
            how="left",
            suffixes=("", "_med"),
        )
        if "broad_category_med" in sel_comp.columns:
            sel_comp["broad_category"] = sel_comp[
                "broad_category_med"
            ].fillna(sel_comp["broad_category"])
        if "mediation_med" in sel_comp.columns:
            sel_comp["mediation"] = sel_comp[
                "mediation_med"
            ].fillna(sel_comp["mediation"])
        if "delta_h_norm_med" in sel_comp.columns:
            sel_comp["delta_h_norm"] = sel_comp[
                "delta_h_norm_med"
            ].fillna(sel_comp["delta_h_norm"])

        sample_df = add_k24_composition_targets(
            sample_df,
            sel_comp,
            source_layers,
        )

        # Candidate feature performance.
        features = feature_columns(token_df)
        print(f"Evaluating {len(features)} scalar token features...")

        overall_rows = []
        for f in tqdm(features, desc="Token feature metrics"):
            overall_rows.append(
                evaluate_feature(
                    token_df,
                    f,
                    a.k,
                )
            )
        overall = pd.DataFrame(overall_rows)

        per_rel = relation_stratified_feature_eval(
            token_df,
            features,
            a.k,
        )

        sample_corr = sample_feature_correlations(
            sample_df
        )

        outcome_df = sample_outcome_predictors(
            sample_df
        )

        # Approx random unique-position TopK recall baseline.
        per_sample_random = []
        for sid, g in token_df.groupby("sid"):
            n_unique_pos = g["position"].nunique()
            n_truth = int(g["selected_k24"].sum())
            if n_unique_pos > 0:
                per_sample_random.append(
                    min(a.k, n_unique_pos)
                    / n_unique_pos
                )
        random_recall = safe_mean(per_sample_random)

        overall = overall.sort_values(
            [
                "feature_kind",
                "best_mean_recall_at_k",
            ],
            ascending=[True, False],
        )

        # Save outputs.
        overall.to_csv(
            outdir / "token_feature_predictor_metrics.csv",
            index=False,
        )
        per_rel.to_csv(
            outdir / "token_feature_predictor_metrics_by_relation.csv",
            index=False,
        )
        sample_df.to_csv(
            outdir / "sample_features_and_k24_composition.csv",
            index=False,
        )
        sample_corr.to_csv(
            outdir / "sample_feature_to_k24_composition_correlations.csv",
            index=False,
        )
        outcome_df.to_csv(
            outdir / "sample_feature_outcome_predictors.csv",
            index=False,
        )

        # Selected-only compact view is convenient for vector/activity inspection.
        token_df[token_df["selected_k24"]].to_csv(
            outdir / "selected_k24_rich_features.csv",
            index=False,
        )

        if a.save_all_token_features:
            token_df.to_pickle(
                outdir / "token_features.pkl.gz",
                compression="gzip",
            )

        summary = render_summary(
            overall,
            per_rel,
            sample_corr,
            outcome_df,
            random_recall,
            a,
        )
        (outdir / "analysis_summary.txt").write_text(
            summary,
            encoding="utf-8",
        )

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "run_dir": str(a.run_dir),
            "bundle": a.bundle,
            "k": a.k,
            "condition": a.condition,
            "selection_strategy": a.selection_strategy,
            "source_layers": source_layers,
            "target_layers": target_layers,
            "N_samples": len(valid_sids),
            "N_candidate_rows": len(token_df),
            "best_dynamic_alpha": best_alpha,
            "with_attention_weights_requested": a.with_attention_weights,
            "attention_weights_available": attention_available_any,
            "attention_head_scan": a.attention_head_scan,
            "random_recall_at_k_baseline": random_recall,
            "feature_count": len(features),
            "important": (
                "The K24 labels and mediation targets are oracle writer-guided. "
                "Only features tagged relation_free_current_forward are candidates "
                "for a no-relation-routing selection rule."
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print()
        print(summary)
        print("Saved:", outdir)

    finally:
        if model is not None:
            del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



# ===== Self-predict causal-core evaluator =====

import argparse
import contextlib
import gc
import importlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor
from tqdm import tqdm


# =============================================================================
# Embedded collector
# =============================================================================

# The feature-collection helpers from predictive_correlates_v3 are embedded
# above in this same file.  No external analyze_dynamic_k24_predictive_*.py
# file is required.
import sys as _sys
CORR = _sys.modules[__name__]
CORR_NAME = "embedded_predictive_correlates_v3"

# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--run-dir", required=True)
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
    p.add_argument("--k", type=int, default=36)
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument("--core-masses", default="0.5,0.7,0.8")
    p.add_argument(
        "--with-attention-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--attention-head-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument(
        "--max-pred-k",
        type=int,
        default=36,
        help="Upper bound when a self-gap selector estimates its own K.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def fmt_mass(x):
    return f"core{int(round(100*x))}"


def canon_rel(x):
    x = str(x).strip().lower()
    return {"above": "on", "below": "under"}.get(x, x)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    good = np.isfinite(score)
    y = y[good]
    score = score[good]
    if len(y) < 4 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def percentile_rank_high(s):
    """
    0..1, larger raw value -> larger rank. NaNs stay NaN.
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(method="average", pct=True)


def within_sid_layer_rank(df, feature):
    return (
        df.groupby(["sid", "source_layer"], group_keys=False)[feature]
        .transform(percentile_rank_high)
        .astype(float)
    )


def unique_position_sorted(g, score_col):
    """
    global_unique-compatible ranking:
      sort all (layer,position) candidates by score, then keep only the first
      occurrence of each token position across layers.
    """
    q = g[np.isfinite(pd.to_numeric(g[score_col], errors="coerce"))].copy()
    if not len(q):
        return q
    q = q.sort_values(
        [score_col, "source_layer", "position"],
        ascending=[False, True, True],
    )
    q = q.drop_duplicates("position", keep="first")
    return q.reset_index(drop=True)


def largest_relative_gap_k(q, score_col, max_k=36, min_k=2):
    """
    Fully self-derived token count. Uses the largest relative drop among the
    first max_k unique-position scores. Diagnostic, because score distributions
    can be smooth.
    """
    if not len(q):
        return 0
    vals = pd.to_numeric(q[score_col], errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    n = min(len(vals), int(max_k))
    vals = vals[:n]
    if n <= min_k:
        return n

    # Shift to positive only for stable relative-drop computation.
    shifted = vals - np.min(vals) + 1e-8
    drops = (shifted[:-1] - shifted[1:]) / (np.abs(shifted[:-1]) + 1e-8)

    start = max(1, int(min_k)) - 1
    if start >= len(drops):
        return n
    j = start + int(np.argmax(drops[start:]))
    return int(j + 1)


# =============================================================================
# Existing run + core ground truth
# =============================================================================

def load_run(a):
    source_layers = CORR.infer_source_layers(a.bundle)
    target_layers = parse_ints(a.target_layers)

    ns = SimpleNamespace(
        run_dir=a.run_dir,
        condition=a.condition,
        selection_strategy=a.selection_strategy,
        bundle=a.bundle,
        k=a.k,
        max_eval_samples=a.max_eval_samples,
        seed=a.seed,
    )
    med, sel, baseline, generation, best_alpha, valid_sids = (
        CORR.load_run_tables(ns, source_layers)
    )

    med["relation"] = med["relation"].map(canon_rel)
    sel["relation"] = sel["relation"].map(canon_rel)

    return (
        med, sel, baseline, generation, best_alpha,
        valid_sids, source_layers, target_layers
    )


def build_core_truth(sel, core_masses):
    """
    Returns:
      truth_by_target[target][sid] = set((layer,pos))
      truth_pos_by_target[target][sid] = set(pos)
      size summary
      selected positive-M lookup
    """
    truth = {fmt_mass(t): {} for t in core_masses}
    truth_pos = {fmt_mass(t): {} for t in core_masses}
    size_rows = []
    selected_m_lookup = {}
    selected_total_mass = {}

    for sid, g0 in sel.groupby("sid", sort=True):
        g = g0.sort_values("mediation", ascending=False).copy()
        m = pd.to_numeric(g["mediation"], errors="coerce").to_numpy(np.float64)
        posm = np.maximum(np.where(np.isfinite(m), m, 0.0), 0.0)
        total = float(posm.sum())
        cum = np.cumsum(posm) / total if total > 0 else np.zeros(len(g))

        selected_total_mass[int(sid)] = total

        for r in g.itertuples():
            selected_m_lookup[
                (int(sid), int(r.source_layer), int(r.position))
            ] = max(float(r.mediation), 0.0)

        row = {
            "sid": int(sid),
            "relation": canon_rel(g["relation"].iloc[0]),
            "selected_positive_mass": total,
        }

        for t in core_masses:
            name = fmt_mass(t)
            idx = np.where(cum >= t)[0]
            kk = int(idx[0] + 1) if len(idx) else len(g)
            core = g.iloc[:kk]

            exact = set(
                (int(r.source_layer), int(r.position))
                for r in core.itertuples()
            )
            positions = set(int(r.position) for r in core.itertuples())

            truth[name][int(sid)] = exact
            truth_pos[name][int(sid)] = positions
            row[f"{name}_N"] = len(exact)
            row[f"{name}_actual_mass_fraction"] = (
                float(posm[:kk].sum() / total) if total > 0 else np.nan
            )

        size_rows.append(row)

    return (
        truth,
        truth_pos,
        pd.DataFrame(size_rows),
        selected_m_lookup,
        selected_total_mass,
    )


# =============================================================================
# Relation-free feature collection
# =============================================================================

def collect_or_load_features(
    a,
    outdir,
    med,
    sel,
    baseline,
    valid_sids,
    source_layers,
    target_layers,
):
    cache_dir = outdir / "feature_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If all chunks exist, skip model load entirely.
    all_cached = all(
        (cache_dir / f"sid_{sid}.pkl.gz").exists()
        for sid in valid_sids
    )
    if all_cached:
        print("All per-sample feature chunks already cached; skipping VLM forward.")
        parts = [
            pd.read_pickle(cache_dir / f"sid_{sid}.pkl.gz", compression="gzip")
            for sid in valid_sids
        ]
        return pd.concat(parts, ignore_index=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
    rec_by_sid = {int(r.sid): r for r in records}

    specs = base.merged_model_specs(two)
    spec = specs[a.model]

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    model_cls = getattr(transformers, spec.model_class)

    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    model = None
    try:
        model = model_cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        base.configure_processor(model, processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        hidden_layers = sorted(set(source_layers + target_layers))
        module_layers = list(source_layers)
        requested_attn_layers = sorted(
            {
                L
                for L in (
                    source_layers
                    + [x + 1 for x in source_layers]
                    + target_layers
                )
                if 0 <= L < n_layers
            }
        )

        print("Feature collector:", CORR_NAME)
        print("decoder:", decoder_path)
        print("source layers:", source_layers)
        print("attention layers requested:", requested_attn_layers)

        for sid in tqdm(valid_sids, desc="Collect self-prediction features"):
            chunk_path = cache_dir / f"sid_{sid}.pkl.gz"
            if chunk_path.exists():
                continue

            if sid not in prompts or sid not in rec_by_sid:
                raise RuntimeError(f"sid={sid} missing prompt or data record")

            pr = prompts[sid]
            real = gray = rb = gb = None
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

                real_cap = CORR.rich_forward(
                    model,
                    decoder_layers,
                    rb,
                    hidden_layers,
                    module_layers,
                    requested_attn_layers,
                    a.with_attention_weights,
                )
                gray_cap = CORR.rich_forward(
                    model,
                    decoder_layers,
                    gb,
                    hidden_layers,
                    module_layers,
                    requested_attn_layers,
                    a.with_attention_weights,
                )

                sid_med = med[med["sid"] == sid].copy()
                token_chunk, _sample_feat = CORR.build_sample_features(
                    sid_med,
                    source_layers,
                    target_layers,
                    real_cap,
                    gray_cap,
                    a.with_attention_weights,
                    a.attention_head_scan,
                )
                token_chunk.to_pickle(chunk_path, compression="gzip")

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
        if model is not None:
            del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parts = [
        pd.read_pickle(cache_dir / f"sid_{sid}.pkl.gz", compression="gzip")
        for sid in valid_sids
    ]
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# Build pure self scores
# =============================================================================

def available(df, name):
    return name in df.columns and df[name].notna().sum() > 0


def mean_existing(df, cols, out_name):
    valid = [c for c in cols if available(df, c)]
    if not valid:
        return None

    X = np.stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) for c in valid],
        axis=1,
    )
    finite = np.isfinite(X)
    n = finite.sum(axis=1)
    val = np.full(len(df), np.nan, dtype=np.float64)
    sums = np.where(finite, X, 0.0).sum(axis=1)
    np.divide(sums, n, out=val, where=n > 0)
    df[out_name] = val
    return out_name


def add_self_scores(token_df):
    """
    Add relation-free scores. Raw features are converted to within-SID-layer
    percentile ranks before ensembles so different layers/scales are comparable.
    """
    df = token_df.copy()

    raw_candidates = [
        "last_attn_next_max_positive_delta",
        "last_attn_next_delta_mean",
        "last_attn_next_real_mean",
        "last_attn_next_real_std_heads",
        "last_attn_same_real_mean",
        "last_attn_same_real_max",
        "hidden_delta_norm_recomputed",
        "attn_out_delta_norm",
        "mlp_out_delta_norm",
        "delta_cos_source_last_delta",
        "delta_cos_late_mean",
        "delta_cos_late_max",
    ]

    rank_cols = []
    for c in raw_candidates:
        if available(df, c):
            rc = "RANK_" + c
            df[rc] = within_sid_layer_rank(df, c)
            rank_cols.append(rc)

    selectors = []

    # Raw attention score (the strongest previous single feature).
    if available(df, "last_attn_next_max_positive_delta"):
        selectors.append(
            ("attn_next_posdelta_raw",
             "last_attn_next_max_positive_delta",
             "pure_self")
        )
        selectors.append(
            ("attn_next_posdelta_rank",
             "RANK_last_attn_next_max_positive_delta",
             "pure_self")
        )

    if available(df, "last_attn_next_delta_mean"):
        selectors.append(
            ("attn_next_delta_rank",
             "RANK_last_attn_next_delta_mean",
             "pure_self")
        )

    if available(df, "last_attn_next_real_mean"):
        selectors.append(
            ("attn_next_real_rank",
             "RANK_last_attn_next_real_mean",
             "pure_self")
        )

    # Attention ensemble.
    attn_rank_features = [
        "RANK_last_attn_next_max_positive_delta",
        "RANK_last_attn_next_delta_mean",
        "RANK_last_attn_next_real_mean",
        "RANK_last_attn_next_real_std_heads",
        "RANK_last_attn_same_real_mean",
        "RANK_last_attn_same_real_max",
    ]
    if mean_existing(df, attn_rank_features, "SCORE_attn_ensemble") is not None:
        selectors.append(
            ("attn_ensemble", "SCORE_attn_ensemble", "pure_self")
        )

    geom_rank_features = [
        "RANK_delta_cos_source_last_delta",
        "RANK_delta_cos_late_mean",
        "RANK_delta_cos_late_max",
    ]
    if mean_existing(df, geom_rank_features, "SCORE_geometry_ensemble") is not None:
        selectors.append(
            ("geometry_ensemble", "SCORE_geometry_ensemble", "pure_self")
        )

    activity_rank_features = [
        "RANK_hidden_delta_norm_recomputed",
        "RANK_attn_out_delta_norm",
        "RANK_mlp_out_delta_norm",
    ]
    if mean_existing(df, activity_rank_features, "SCORE_activity_ensemble") is not None:
        selectors.append(
            ("activity_ensemble", "SCORE_activity_ensemble", "pure_self")
        )

    all_components = [
        c for c in [
            "SCORE_attn_ensemble",
            "SCORE_geometry_ensemble",
            "SCORE_activity_ensemble",
        ]
        if available(df, c)
    ]
    if mean_existing(df, all_components, "SCORE_all_ensemble") is not None:
        selectors.append(
            ("all_ensemble", "SCORE_all_ensemble", "pure_self")
        )

    return df, selectors


# =============================================================================
# Leave-one-sample-out layer-role prior (no relation labels)
# =============================================================================

def add_loo_structural_prior(
    token_df,
    truth_exact_by_sid,
    target_name,
):
    """
    P(core | layer, broad_category) estimated from OTHER samples.
    This is relation-free but calibration-dependent.
    """
    df = token_df.copy()
    is_core = np.array(
        [
            (int(L), int(pos)) in truth_exact_by_sid.get(int(sid), set())
            for sid, L, pos in zip(df["sid"], df["source_layer"], df["position"])
        ],
        dtype=np.int8,
    )
    temp = pd.DataFrame({
        "sid": df["sid"].astype(int).to_numpy(),
        "bucket": (
            "L" + df["source_layer"].astype(int).astype(str)
            + ":" + df["broad_category"].astype(str)
        ).to_numpy(),
        "is_core": is_core,
    })

    global_stats = temp.groupby("bucket")["is_core"].agg(["sum", "count"])
    sid_stats = temp.groupby(["sid", "bucket"])["is_core"].agg(["sum", "count"])

    priors = np.empty(len(temp), dtype=np.float64)

    for i, r in enumerate(temp.itertuples(index=False)):
        gs = global_stats.loc[r.bucket]
        try:
            ss = sid_stats.loc[(r.sid, r.bucket)]
            num = float(gs["sum"] - ss["sum"])
            den = float(gs["count"] - ss["count"])
        except KeyError:
            num = float(gs["sum"])
            den = float(gs["count"])
        priors[i] = num / den if den > 0 else np.nan

    col = f"SCORE_loo_structural_prior_{target_name}"
    df[col] = priors
    return df, col


# =============================================================================
# Core localization metrics
# =============================================================================

def evaluate_one_prediction(
    sid,
    pred_rows,
    truth_exact,
    truth_pos,
    mediation_lookup,
    true_core_mass,
):
    pred_exact = set(
        (int(r.source_layer), int(r.position))
        for r in pred_rows.itertuples()
    )
    pred_pos = set(int(r.position) for r in pred_rows.itertuples())

    inter = pred_exact & truth_exact

    # +/-1-layer tolerance for the SAME position.
    tolerant_hits = 0
    for Lt, pt in truth_exact:
        ok = any(
            (pp == pt and abs(Lp - Lt) <= 1)
            for Lp, pp in pred_exact
        )
        tolerant_hits += int(ok)

    exact_recall = safe_div(len(inter), len(truth_exact))
    exact_precision = safe_div(len(inter), len(pred_exact))
    union = truth_exact | pred_exact
    exact_jaccard = safe_div(len(inter), len(union))

    pos_inter = pred_pos & truth_pos
    pos_recall = safe_div(len(pos_inter), len(truth_pos))
    pos_precision = safe_div(len(pos_inter), len(pred_pos))
    pos_union = pred_pos | truth_pos
    pos_jaccard = safe_div(len(pos_inter), len(pos_union))

    # How much oracle positive M is carried by the actually predicted rows?
    pred_mass = 0.0
    for L, p in pred_exact:
        pred_mass += max(
            float(mediation_lookup.get((int(sid), int(L), int(p)), 0.0)),
            0.0,
        )

    return {
        "pred_N": len(pred_exact),
        "true_N": len(truth_exact),
        "exact_recall": exact_recall,
        "exact_precision": exact_precision,
        "exact_jaccard": exact_jaccard,
        "position_recall": pos_recall,
        "position_precision": pos_precision,
        "position_jaccard": pos_jaccard,
        "layer_pm1_recall": safe_div(tolerant_hits, len(truth_exact)),
        "pred_positive_M": pred_mass,
        "pred_M_over_true_core_M": safe_div(pred_mass, true_core_mass),
    }


def core_mass_for_truth(sid, truth_set, mediation_lookup):
    return float(
        sum(
            max(float(mediation_lookup.get((sid, L, p), 0.0)), 0.0)
            for L, p in truth_set
        )
    )


def evaluate_selector(
    token_df,
    selector_name,
    score_col,
    selector_kind,
    target_name,
    truth,
    truth_pos,
    global_fixed_k,
    mediation_lookup,
    max_pred_k,
):
    rows = []

    for sid, g in token_df.groupby("sid", sort=True):
        sid = int(sid)
        true = truth[sid]
        true_pos = truth_pos[sid]
        true_mass = core_mass_for_truth(
            sid, true, mediation_lookup
        )

        ranked = unique_position_sorted(g, score_col)
        if not len(ranked):
            continue

        # 1. Deployable fixed global core-size prior.
        k_fixed = min(int(global_fixed_k), len(ranked))
        pred = ranked.iloc[:k_fixed]
        m = evaluate_one_prediction(
            sid, pred, true, true_pos, mediation_lookup, true_mass
        )
        m.update({
            "sid": sid,
            "relation": canon_rel(g["relation"].iloc[0]),
            "target_core": target_name,
            "selector": selector_name,
            "selector_kind": selector_kind,
            "size_mode": "fixed_global_median",
            "score_col": score_col,
        })
        rows.append(m)

        # 2. Diagnostic: exact number of core tokens is revealed, identities are not.
        k_true = min(len(true), len(ranked))
        pred = ranked.iloc[:k_true]
        m = evaluate_one_prediction(
            sid, pred, true, true_pos, mediation_lookup, true_mass
        )
        m.update({
            "sid": sid,
            "relation": canon_rel(g["relation"].iloc[0]),
            "target_core": target_name,
            "selector": selector_name,
            "selector_kind": selector_kind,
            "size_mode": "oracle_core_size_DIAGNOSTIC",
            "score_col": score_col,
        })
        rows.append(m)

        # 3. Fully self-derived K from score-gap.
        k_gap = largest_relative_gap_k(
            ranked, score_col, max_k=max_pred_k, min_k=2
        )
        k_gap = max(1, min(k_gap, len(ranked)))
        pred = ranked.iloc[:k_gap]
        m = evaluate_one_prediction(
            sid, pred, true, true_pos, mediation_lookup, true_mass
        )
        m.update({
            "sid": sid,
            "relation": canon_rel(g["relation"].iloc[0]),
            "target_core": target_name,
            "selector": selector_name,
            "selector_kind": selector_kind,
            "size_mode": "self_largest_score_gap",
            "score_col": score_col,
        })
        rows.append(m)

    return rows


def summarize_metrics(detail):
    group_cols = [
        "target_core", "selector", "selector_kind", "size_mode"
    ]
    metrics = [
        "pred_N",
        "true_N",
        "exact_recall",
        "exact_precision",
        "exact_jaccard",
        "position_recall",
        "position_precision",
        "position_jaccard",
        "layer_pm1_recall",
        "pred_M_over_true_core_M",
    ]
    return (
        detail.groupby(group_cols, as_index=False)[metrics]
        .mean()
        .sort_values(
            ["target_core", "size_mode", "position_recall"],
            ascending=[True, True, False],
        )
    )


def summarize_by_relation(detail):
    group_cols = [
        "target_core", "selector", "selector_kind", "size_mode", "relation"
    ]
    metrics = [
        "pred_N",
        "true_N",
        "exact_recall",
        "position_recall",
        "layer_pm1_recall",
        "pred_M_over_true_core_M",
    ]
    return (
        detail.groupby(group_cols, as_index=False)[metrics]
        .mean()
        .sort_values(
            ["target_core", "size_mode", "relation", "position_recall"],
            ascending=[True, True, True, False],
        )
    )


def render_summary(summary, core_sizes, selector_meta):
    lines = []
    lines.append("=" * 126)
    lines.append("RELATION-FREE SELF-PREDICTION OF SAMPLE-SPECIFIC CAUSAL CORE")
    lines.append("=" * 126)

    for c in sorted(
        [x for x in core_sizes.columns if x.endswith("_N")]
    ):
        target = c[:-2]
        x = core_sizes[c]
        lines.append(
            f"{target}: mean true N={x.mean():.2f}, "
            f"median={x.median():.1f}, p25={x.quantile(.25):.1f}, "
            f"p75={x.quantile(.75):.1f}"
        )

    lines.append("")
    lines.append(
        "Primary metric: position_recall = did the selector find the same token "
        "position, even if it chose an adjacent layer?"
    )
    lines.append(
        "Exact_recall requires the exact same (layer,position); layer_pm1_recall "
        "allows +/-1 layer at the same position."
    )

    for target in sorted(summary["target_core"].unique()):
        lines.append("")
        lines.append("-" * 126)
        lines.append(target.upper())
        lines.append("-" * 126)

        for mode in [
            "fixed_global_median",
            "oracle_core_size_DIAGNOSTIC",
            "self_largest_score_gap",
        ]:
            g = summary[
                (summary["target_core"] == target)
                & (summary["size_mode"] == mode)
            ].sort_values(
                ["position_recall", "exact_recall"],
                ascending=False,
            )
            if not len(g):
                continue
            lines.append(f"  [{mode}]")
            for r in g.head(12).itertuples():
                lines.append(
                    f"    {r.selector:<34s} "
                    f"kind={r.selector_kind:<20s} "
                    f"N={r.pred_N:5.2f} | "
                    f"exactR={r.exact_recall:.3f} "
                    f"posR={r.position_recall:.3f} "
                    f"±1R={r.layer_pm1_recall:.3f} "
                    f"J={r.position_jaccard:.3f} "
                    f"M/coreM={r.pred_M_over_true_core_M:.3f}"
                )

    lines.append("")
    lines.append("Interpretation guardrails:")
    lines.append(
        "  pure_self selectors use only the current sample's relation-free forward-pass quantities."
    )
    lines.append(
        "  calibrated_no_relation selectors use a leave-one-sample-out layer-role prior from other samples; "
        "they still do NOT use LEFT/RIGHT/ON/UNDER, but they are calibration-dependent."
    )
    lines.append(
        "  oracle_core_size_DIAGNOSTIC is NOT deployable; it isolates localization quality from core-size prediction."
    )
    lines.append(
        "  A large jump from exactR to posR/±1R means the self signal often finds the right token trajectory "
        "but chooses a neighboring layer."
    )
    lines.append(
        "  M/coreM can exceed exact overlap: that means 'wrong' predicted positions may still carry substantial "
        "positive oracle mediation, consistent with redundant causal carriers."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    core_masses = parse_floats(a.core_masses)
    for t in core_masses:
        if not (0 < t <= 1):
            raise ValueError(f"core mass must be in (0,1], got {t}")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (
        med, sel, baseline, generation, best_alpha,
        valid_sids, source_layers, target_layers
    ) = load_run(a)

    (
        truth,
        truth_pos,
        core_sizes,
        selected_m_lookup,
        selected_total_mass,
    ) = build_core_truth(sel, core_masses)

    core_sizes.to_csv(outdir / "oracle_core_size_by_sample.csv", index=False)

    print("=" * 120)
    print("SELF-PREDICT SAMPLE-SPECIFIC CORE")
    print("=" * 120)
    print("N samples:", len(valid_sids))
    print("source layers:", source_layers)
    print("target layers:", target_layers)
    print("forced oracle K:", a.k)
    print("core targets:", [fmt_mass(x) for x in core_masses])
    for t in core_masses:
        c = fmt_mass(t) + "_N"
        print(
            f"  {fmt_mass(t)}: mean={core_sizes[c].mean():.2f}, "
            f"median={core_sizes[c].median():.1f}"
        )
    print()

    token_df = collect_or_load_features(
        a, outdir, med, sel, baseline, valid_sids,
        source_layers, target_layers
    )

    # Ensure numeric mediation lookup includes ALL candidate rows, not just K36.
    mediation_lookup = {
        (int(r.sid), int(r.source_layer), int(r.position)): max(float(r.mediation), 0.0)
        for r in med.itertuples()
        if np.isfinite(float(r.mediation))
    }

    token_df, base_selectors = add_self_scores(token_df)

    # Save candidate features once.
    token_df.to_pickle(
        outdir / "all_candidate_self_features.pkl.gz",
        compression="gzip",
    )

    all_detail_rows = []
    selector_meta = []

    for t in core_masses:
        target = fmt_mass(t)
        sizes = core_sizes[target + "_N"]
        fixed_k = int(round(float(sizes.median())))

        # Pure self selectors.
        for selector_name, score_col, kind in base_selectors:
            selector_meta.append({
                "target_core": target,
                "selector": selector_name,
                "score_col": score_col,
                "selector_kind": kind,
            })
            all_detail_rows.extend(
                evaluate_selector(
                    token_df,
                    selector_name,
                    score_col,
                    kind,
                    target,
                    truth[target],
                    truth_pos[target],
                    fixed_k,
                    mediation_lookup,
                    a.max_pred_k,
                )
            )

        # LOO structural prior; no relation label.
        prior_df, prior_col = add_loo_structural_prior(
            token_df,
            truth[target],
            target,
        )

        prior_selectors = [
            (f"{target}_loo_role_prior", prior_col, "calibrated_no_relation")
        ]

        # Combine prior with the strongest self attention family.
        attn_col = None
        for candidate in [
            "SCORE_attn_ensemble",
            "RANK_last_attn_next_max_positive_delta",
            "last_attn_next_max_positive_delta",
        ]:
            if available(prior_df, candidate):
                attn_col = candidate
                break

        if attn_col is not None:
            # Convert both to within-sample ranks before mixing.
            prior_rank = f"RANK_{prior_col}"
            prior_df[prior_rank] = (
                prior_df.groupby("sid", group_keys=False)[prior_col]
                .transform(percentile_rank_high)
                .astype(float)
            )

            attn_mix_rank = f"RANK_MIXBASE_{target}"
            prior_df[attn_mix_rank] = (
                prior_df.groupby(["sid","source_layer"], group_keys=False)[attn_col]
                .transform(percentile_rank_high)
                .astype(float)
            )

            for w_attn in [0.50, 0.75]:
                w_prior = 1.0 - w_attn
                col = f"SCORE_{target}_attn{int(w_attn*100)}_prior{int(w_prior*100)}"
                prior_df[col] = (
                    w_attn * prior_df[attn_mix_rank]
                    + w_prior * prior_df[prior_rank]
                )
                prior_selectors.append(
                    (
                        f"{target}_attn{int(w_attn*100)}_roleprior{int(w_prior*100)}",
                        col,
                        "calibrated_no_relation",
                    )
                )

        for selector_name, score_col, kind in prior_selectors:
            selector_meta.append({
                "target_core": target,
                "selector": selector_name,
                "score_col": score_col,
                "selector_kind": kind,
            })
            all_detail_rows.extend(
                evaluate_selector(
                    prior_df,
                    selector_name,
                    score_col,
                    kind,
                    target,
                    truth[target],
                    truth_pos[target],
                    fixed_k,
                    mediation_lookup,
                    a.max_pred_k,
                )
            )

    detail = pd.DataFrame(all_detail_rows)
    summary = summarize_metrics(detail)
    by_relation = summarize_by_relation(detail)

    detail.to_csv(outdir / "self_core_prediction_per_sample.csv", index=False)
    summary.to_csv(outdir / "self_core_prediction_summary.csv", index=False)
    by_relation.to_csv(
        outdir / "self_core_prediction_by_relation.csv",
        index=False,
    )
    pd.DataFrame(selector_meta).drop_duplicates().to_csv(
        outdir / "selector_definitions.csv",
        index=False,
    )

    text = render_summary(
        summary,
        core_sizes,
        pd.DataFrame(selector_meta),
    )
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    metadata = {
        "feature_collector_module": CORR_NAME,
        "run_dir": a.run_dir,
        "bundle": a.bundle,
        "k": a.k,
        "source_layers": source_layers,
        "target_layers": target_layers,
        "core_masses": core_masses,
        "N_samples": len(valid_sids),
        "best_dynamic_alpha_original_run": best_alpha,
        "important": (
            "Core labels and mediation are oracle evaluation targets only. "
            "Selectors tagged pure_self never use GT relation or writer direction. "
            "Selectors tagged calibrated_no_relation use leave-one-sample-out "
            "layer-role core-frequency priors but no relation labels."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print(text)
    print("Saved outputs to:", outdir)


if __name__ == "__main__":
    main()
