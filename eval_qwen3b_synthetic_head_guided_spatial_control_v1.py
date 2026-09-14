
# -*- coding: utf-8 -*-
"""
eval_qwen3b_synthetic_head_guided_spatial_control_v1.py

Qwen2.5-VL-3B / COCO_two: use a Synthetic-400 selected spatial attention head
as the NON-ORACLE target for the old multi-layer residual spatial controller.

Main question
=============
The previous oracle controller showed that editing ONLY the mid-layer
subject/reference spatial H/V subspace (L20-26) can drive actual generation
from ~66% to ~95% on an N=80 diagnostic subset.  That experiment used the
COCO ground-truth relation as the optimization target.

This script changes only the target selection:

    Synthetic-400
        -> choose the best direction head using SOURCE accuracy only
        -> refit that head's four frozen relation directions on all Synthetic-400

    COCO image
        -> selected head predicts r_hat in {left,right,above,below}
        -> replace oracle GT target by r_hat
        -> optimize ONLY the L20-26 residual spatial H/V coordinates
        -> continue the original network and call model.generate()

So the main condition is:

    spatial head (READ) -> residual spatial controller (WRITE) -> generation

No COCO GT is used to select the head or choose the intervention target.
COCO GT is used only for evaluation metrics.  For a strict source-only method,
use --geometry-mode synthetic_source.  The default --geometry-mode coco_calib
reuses the old controller's COCO-calibrated H/V geometry, which is useful for
an apples-to-apples mechanistic diagnostic but is not fully target-label-free.

Head selection
==============
Default is source_cv: stratified K-fold CV entirely within Synthetic-400.
The head with the highest Synthetic CV accuracy is chosen, then its codebook is
refit on all 400 synthetic samples before COCO inference.

Use --head-select-mode source_self to reproduce the simpler "highest Synthetic
self-accuracy" rule.  That rule uses the same source points to fit and rank the
prototype code, so source_cv is cleaner for a paper.

Controller profiles
===================
old:
    step=0.75, max_steps=20, max_total_natural=12
    approximately the previous bounded controller.

wide:
    step=1.50, max_steps=60, max_total_natural=0 (NO global cap)
    no per-layer cap; backtracking still requires real objective improvement.

verywide:
    step=3.00, max_steps=80, max_total_natural=0
    stronger diagnostic upper bound.

The optimizer uses the v2 smooth multi-competitor objective inside the 14D
spatial coordinate system; it never edits arbitrary hidden dimensions.

Recommended first diagnostic
============================
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen3b_synthetic_head_guided_spatial_control_v1.py \
  --geometry-mode coco_calib \
  --spatial-states-npz output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz \
  --head-select-mode source_cv \
  --modes head,oracle \
  --profiles old,wide \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_synhead_spatial_control_n80_v1 \
  --overwrite

Strict source-only spatial geometry
===================================
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen3b_synthetic_head_guided_spatial_control_v1.py \
  --geometry-mode synthetic_source \
  --source-spatial-npz output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --head-select-mode source_cv \
  --modes head,oracle \
  --profiles wide \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_synhead_spatial_control_syngeom_n80_v1 \
  --overwrite

Notes
=====
* "oracle" is diagnostic only: it uses the COCO GT relation as the controller
  target and asks whether the wider spatial budget can push 95% closer to 100%.
* "head" is the actual non-oracle target-routing experiment.
* Selecting the best profile after inspecting COCO GT would be target tuning.
  Treat old/wide/verywide comparisons as diagnostics until a profile is frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import eval_real_causal_token_update_gating_v1 as gate
    import eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp as ora
    import eval_nonoracle_direct_self_spatial_amplification_v1 as selfamp
    import scan_synthetic_frozen_direction_heads_multimodel_v1 as headscan
except Exception as exc:
    raise SystemExit(
        "Could not import AdaptVis dependencies. Run from the llava16 repo root and keep:\n"
        "  eval_real_causal_token_update_gating_v1.py\n"
        "  eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp.py\n"
        "  eval_nonoracle_direct_self_spatial_amplification_v1.py\n"
        "  scan_synthetic_frozen_direction_heads_multimodel_v1.py\n"
        f"available.\n{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
EPS = 1e-12
SCRIPT_VERSION = "qwen3b-synthetic-head-guided-spatial-control-v1"


@dataclass(frozen=True)
class Profile:
    name: str
    step_natural: float
    max_steps: int
    max_total_natural: float
    max_layer_natural: float
    line_search_tries: int


PROFILE_SPECS: Dict[str, Profile] = {
    "old": Profile(
        name="old",
        step_natural=0.75,
        max_steps=20,
        max_total_natural=12.0,
        max_layer_natural=0.0,
        line_search_tries=8,
    ),
    "wide": Profile(
        name="wide",
        step_natural=1.50,
        max_steps=60,
        max_total_natural=0.0,
        max_layer_natural=0.0,
        line_search_tries=10,
    ),
    "verywide": Profile(
        name="verywide",
        step_natural=3.00,
        max_steps=80,
        max_total_natural=0.0,
        max_layer_natural=0.0,
        line_search_tries=12,
    ),
}


def parse_csv_list(text: str, allowed: Sequence[str], name: str) -> List[str]:
    vals: List[str] = []
    allowed_set = set(allowed)
    for raw in str(text).split(","):
        v = raw.strip().lower()
        if not v:
            continue
        if v not in allowed_set:
            raise ValueError(f"Unknown {name}={v!r}; choose from {sorted(allowed_set)}")
        if v not in vals:
            vals.append(v)
    if not vals:
        raise ValueError(f"No {name} parsed")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Fixed first target requested by the user.
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    # headscan expects this alias.
    p.add_argument(
        "--coco-prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )

    # Synthetic direction-head source.
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)
    p.add_argument("--target-max-samples", type=int, default=None)
    p.add_argument("--head-control", dest="control", default="gray", choices=["gray", "noimage"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument(
        "--head-select-mode",
        default="source_cv",
        choices=["source_cv", "source_self"],
        help="Select the head using Synthetic-400 only; source_cv is recommended.",
    )
    p.add_argument("--source-cv-folds", type=int, default=5)
    p.add_argument("--head-source-cache", default=None)
    p.add_argument("--head-target-cache", default=None)
    p.add_argument(
        "--reuse-head-cache-any-version",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a supplied cache contains vectors/sid/relation, reuse it without script-version checks.",
    )

    # Residual spatial geometry.
    p.add_argument(
        "--geometry-mode",
        default="coco_calib",
        choices=["coco_calib", "synthetic_source"],
        help=(
            "coco_calib exactly mirrors the old controller geometry (uses a COCO calibration split); "
            "synthetic_source defines H/V geometry only from Synthetic-400 residual states."
        ),
    )
    p.add_argument(
        "--spatial-states-npz",
        default="output/qwen3b_coco_spatial_real_noimage_v1/states/raw__correct_minus_noimage.npz",
        help="COCO residual relation states for coco_calib geometry.",
    )
    p.add_argument(
        "--source-spatial-npz",
        default="output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz",
        help="Synthetic residual relation states for synthetic_source geometry.",
    )
    p.add_argument("--spatial-fit-ratio", type=float, default=0.30)
    p.add_argument("--spatial-fit-seed", type=int, default=1)
    p.add_argument(
        "--layer-groups",
        default="20-26",
        help='Semicolon-separated groups, e.g. "25;23-26;20-26".',
    )

    # Conditions and controller strength.
    p.add_argument("--modes", default="head,oracle", help="Comma-separated: head,oracle")
    p.add_argument("--profiles", default="old,wide", help="Comma-separated: old,wide,verywide")
    p.add_argument("--competitor-temperature", type=float, default=0.25)
    p.add_argument("--line-search-shrink", type=float, default=0.5)
    p.add_argument("--min-objective-improvement", type=float, default=1e-6)
    p.add_argument(
        "--generation-check-every",
        type=int,
        default=1,
        help="Run actual generation every N accepted spatial updates.",
    )
    p.add_argument(
        "--min-head-margin",
        type=float,
        default=0.0,
        help=(
            "Optional source-frozen head confidence gate. If margin is below this threshold, "
            "the head condition leaves baseline unchanged. Default 0 edits every baseline/head disagreement."
        ),
    )

    # Evaluation / model generation.
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all eligible COCO samples")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--answer-surface", default="above_below", choices=["above_below", "on_under"])
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--answer-suffix", default="")
    p.add_argument("--sequence-score-reduction", default="mean", choices=["mean", "sum"])
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl", default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--cache-vectors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    a = p.parse_args()
    try:
        a.modes_parsed = parse_csv_list(a.modes, ["head", "oracle"], "mode")
        a.profiles_parsed = parse_csv_list(
            a.profiles, list(PROFILE_SPECS.keys()), "profile"
        )
    except ValueError as exc:
        p.error(str(exc))

    if a.source_cv_folds < 2:
        p.error("--source-cv-folds must be >=2")
    if not (0.0 < a.spatial_fit_ratio < 1.0):
        p.error("--spatial-fit-ratio must be in (0,1)")
    if a.competitor_temperature <= 0:
        p.error("--competitor-temperature must be >0")
    if not (0.0 < a.line_search_shrink < 1.0):
        p.error("--line-search-shrink must be in (0,1)")
    if a.min_objective_improvement < 0:
        p.error("--min-objective-improvement must be >=0")
    if a.generation_check_every <= 0:
        p.error("--generation-check-every must be >=1")
    if a.min_head_margin < 0:
        p.error("--min-head-margin must be >=0")
    return a


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def safe_acc(pred: Sequence[Any], gt: Sequence[Any]) -> float:
    p = np.asarray(pred, dtype=object)
    y = np.asarray(gt, dtype=object)
    return float(np.mean(p == y)) if len(y) else float("nan")


def load_head_cache_any(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        required = {"vectors", "sid", "relation"}
        missing = required - keys
        if missing:
            raise RuntimeError(f"{path} missing head-cache keys: {sorted(missing)}")
        return {
            "vectors": np.asarray(z["vectors"]),
            "sid": np.asarray(z["sid"], dtype=np.int64),
            "relation": np.asarray([headscan.norm_relation(x) for x in z["relation"].tolist()], dtype=object),
            "baseline_pred": [None] * len(z["sid"]),
            "baseline_text": [""] * len(z["sid"]),
            "metadata": {"loaded_any_version": True, "path": str(path)},
        }


def load_or_extract_head_pack(
    *,
    records: Sequence[Mapping[str, Any]],
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    n_heads: int,
    head_dim: int,
    dataset_name: str,
    cache_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if (
        args.reuse_head_cache_any_version
        and cache_path.exists()
        and not args.overwrite
    ):
        try:
            pack = load_head_cache_any(cache_path)
            print(f"[HEAD CACHE] generic reuse {cache_path} | N={len(pack['sid'])}")
            return pack
        except Exception as exc:
            print(f"[HEAD CACHE] generic reuse failed: {type(exc).__name__}: {exc}")

    return headscan.extract_vectors(
        records=records,
        model=model,
        processor=processor,
        layers=decoder_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        model_alias="qwen-3b",
        dataset_name=dataset_name,
        args=args,
        cache_path=cache_path,
        with_generation=False,
    )


def stratified_folds(y: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: List[List[int]] = [[] for _ in range(k)]
    for rel in REL:
        idx = np.flatnonzero(y == rel).astype(np.int64)
        idx = idx.copy()
        rng.shuffle(idx)
        for j, ix in enumerate(idx.tolist()):
            buckets[j % k].append(int(ix))
    folds = [np.asarray(sorted(b), dtype=np.int64) for b in buckets if b]
    if len(folds) < 2:
        raise RuntimeError("Could not form >=2 non-empty source CV folds")
    return folds


def rank_heads_source_only(
    Xs: np.ndarray,
    ys: np.ndarray,
    mode: str,
    folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, Tuple[int, int], np.ndarray, np.ndarray]:
    """Return source-only ranking, selected (layer,head), and all-source codebook."""
    Xs = np.asarray(Xs)
    ys = np.asarray(ys, dtype=object)
    _, L, H, _ = Xs.shape

    center_all, dirs_all = headscan.fit_source_codebooks(Xs, ys)
    pred_self: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    self_acc = np.zeros((L, H), dtype=np.float64)
    self_margin = np.zeros((L, H), dtype=np.float64)
    for l in range(L):
        for h in range(H):
            p, m = headscan.predict_one_head(Xs, center_all, dirs_all, l, h)
            pred_self[(l, h)] = (p, m)
            self_acc[l, h] = safe_acc(p, ys)
            self_margin[l, h] = float(np.mean(m))

    if mode == "source_self":
        rows = []
        for l in range(L):
            for h in range(H):
                rows.append({
                    "layer": l,
                    "head": h,
                    "head_name": headscan.head_name(l, h),
                    "source_select_acc": float(self_acc[l, h]),
                    "source_cv_acc": np.nan,
                    "source_self_acc": float(self_acc[l, h]),
                    "source_mean_margin": float(self_margin[l, h]),
                })
    else:
        fold_idx = stratified_folds(ys, folds, seed)
        correct = np.zeros((L, H), dtype=np.float64)
        total = np.zeros((L, H), dtype=np.float64)
        margin_sum = np.zeros((L, H), dtype=np.float64)
        margin_n = np.zeros((L, H), dtype=np.float64)
        all_idx = np.arange(len(ys), dtype=np.int64)

        for fi, val_idx in enumerate(fold_idx):
            val_set = set(map(int, val_idx.tolist()))
            train_idx = np.asarray([i for i in all_idx.tolist() if i not in val_set], dtype=np.int64)
            c, d = headscan.fit_source_codebooks(Xs[train_idx], ys[train_idx])
            for l in range(L):
                for h in range(H):
                    p, m = headscan.predict_one_head(Xs[val_idx], c, d, l, h)
                    correct[l, h] += float(np.sum(p == ys[val_idx]))
                    total[l, h] += float(len(val_idx))
                    margin_sum[l, h] += float(np.sum(m))
                    margin_n[l, h] += float(len(m))
            print(f"[SOURCE CV] fold {fi+1}/{len(fold_idx)} n_train={len(train_idx)} n_val={len(val_idx)}")

        rows = []
        for l in range(L):
            for h in range(H):
                cv_acc = float(correct[l, h] / max(total[l, h], 1.0))
                cv_margin = float(margin_sum[l, h] / max(margin_n[l, h], 1.0))
                rows.append({
                    "layer": l,
                    "head": h,
                    "head_name": headscan.head_name(l, h),
                    "source_select_acc": cv_acc,
                    "source_cv_acc": cv_acc,
                    "source_self_acc": float(self_acc[l, h]),
                    "source_mean_margin": cv_margin,
                })

    rows = sorted(
        rows,
        key=lambda r: (
            -float(r["source_select_acc"]),
            -float(r["source_self_acc"]),
            -float(r["source_mean_margin"]),
            int(r["layer"]),
            int(r["head"]),
        ),
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    ranking = pd.DataFrame(rows)
    best = rows[0]
    key = (int(best["layer"]), int(best["head"]))
    return ranking, key, center_all, dirs_all


def fit_controller_geometry(
    args: argparse.Namespace,
    needed_layers: Sequence[int],
    outdir: Path,
) -> Tuple[dict, Optional[set], Dict[str, Any]]:
    """Fit residual H/V geometry. Returns geom, optional heldout sid set, metadata."""
    if args.geometry_mode == "coco_calib":
        X, y, layers, sids = ora.load_spatial_npz(Path(args.spatial_states_npz))
        fit_idx, held_idx, fit_sids, held_sids = ora.make_spatial_split(
            y, sids, args.spatial_fit_ratio, args.spatial_fit_seed
        )
        geom, axis_df = ora.fit_spatial_geometry(X, y, layers, fit_idx, needed_layers)
        axis_df.to_csv(outdir / "controller_spatial_geometry.csv", index=False)
        meta = {
            "geometry_mode": "coco_calib",
            "npz": str(args.spatial_states_npz),
            "fit_N": int(len(fit_idx)),
            "heldout_N": int(len(held_idx)),
            "fit_ratio": float(args.spatial_fit_ratio),
            "fit_seed": int(args.spatial_fit_seed),
            "uses_target_labels_to_define_axes": True,
        }
        return geom, held_sids, meta

    Xs, ys, source_layers, source_sids, source_def = selfamp.load_state_npz(
        Path(args.source_spatial_npz), require_labels=True
    )
    geom, axis_df = selfamp.fit_source_geometry(Xs, ys, source_layers, needed_layers)
    axis_df.to_csv(outdir / "controller_spatial_geometry.csv", index=False)
    meta = {
        "geometry_mode": "synthetic_source",
        "npz": str(args.source_spatial_npz),
        "fit_N": int(len(Xs)),
        "vector_definition": source_def,
        "uses_target_labels_to_define_axes": False,
    }
    return geom, None, meta


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if len(df) == 0:
        return pd.DataFrame()
    for (mode, profile, group), g in df.groupby(
        ["mode", "profile", "layer_group"], sort=False
    ):
        base = g["baseline_correct"].astype(bool).to_numpy()
        patch = g["patched_correct"].astype(bool).to_numpy()
        target_ok = g["patched_matches_control_target"].astype(bool).to_numpy()
        head_ok = g["head_correct"].astype(bool).to_numpy()
        rows.append({
            "mode": mode,
            "profile": profile,
            "layer_group": group,
            "N": int(len(g)),
            "baseline_accuracy": float(np.mean(base)),
            "head_readout_accuracy": float(np.mean(head_ok)),
            "patched_generation_accuracy": float(np.mean(patch)),
            "gain_vs_baseline": float(np.mean(patch) - np.mean(base)),
            "target_compliance": float(np.mean(target_ok)),
            "wrong_to_correct": int(np.sum((~base) & patch)),
            "correct_to_wrong": int(np.sum(base & (~patch))),
            "net": int(np.sum((~base) & patch) - np.sum(base & (~patch))),
            "changed": int(np.sum(g["baseline_prediction"] != g["patched_prediction"])),
            "mean_steps": float(g["steps_taken"].mean()),
            "median_steps": float(g["steps_taken"].median()),
            "mean_final_total_natural": float(g["final_total_natural_norm"].mean()),
            "max_final_total_natural": float(g["final_total_natural_norm"].max()),
            "fraction_line_search_failed": float(g["line_search_failed"].astype(bool).mean()),
            "fraction_head_gated": float(g["head_margin_gated"].astype(bool).mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["patched_generation_accuracy", "target_compliance"], ascending=[False, False]
    )


def optimize_spatial_to_target(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, torch.Tensor],
    candidate_ids: Mapping[str, Sequence[int]],
    layers: Sequence[int],
    geom: dict,
    sub_pos: int,
    ref_pos: int,
    target_rel: str,
    base_scores: Dict[str, float],
    base_gen_pred: str,
    base_gen_text: str,
    profile: Profile,
    args: argparse.Namespace,
    common: Mapping[str, Any],
    step_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    coords = np.zeros((len(layers), 2), dtype=np.float64)
    current_scores = dict(base_scores)
    current_tf_pred = ora.argmax_relation(current_scores)
    current_gen_pred = base_gen_pred
    current_gen_text = base_gen_text
    steps_taken = 0
    hit_total_cap = False
    line_search_failed = False
    stop_reason = "max_steps"

    if current_gen_pred == target_rel:
        stop_reason = "already_target"
    else:
        for step_idx in range(1, profile.max_steps + 1):
            if current_gen_pred == target_rel:
                stop_reason = "generation_target"
                break

            current_margin = ora.min_gt_margin(current_scores, target_rel)
            current_objective = ora.smooth_gt_objective(
                current_scores, target_rel, args.competitor_temperature
            )

            # Exact gradient of the smooth target-vs-all-competitors objective,
            # but ONLY with respect to the 2*|layers| spatial coordinates.
            score_grad: Dict[str, Tuple[float, np.ndarray]] = {}
            for r in REL:
                sr, gr = ora.sequence_score_and_spatial_grad(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    answer_ids=candidate_ids[r],
                    reduction=args.sequence_score_reduction,
                    layers=layers,
                    geom=geom,
                    sub_pos=sub_pos,
                    ref_pos=ref_pos,
                    base_coords=coords,
                )
                score_grad[r] = (float(sr), np.asarray(gr, dtype=np.float64))

            grad_scores = {r: score_grad[r][0] for r in REL}
            comp_names, comp_weights = ora.smooth_competitor_weights(
                grad_scores, target_rel, args.competitor_temperature
            )
            g_target = score_grad[target_rel][1]
            weighted_comp_grad = np.zeros_like(g_target, dtype=np.float64)
            for r, w in zip(comp_names, comp_weights):
                weighted_comp_grad += float(w) * score_grad[r][1]
            grad2d = g_target - weighted_comp_grad

            direction, raw_grad_norm = ora.normalize_direction(grad2d)
            if direction is None:
                stop_reason = "zero_spatial_gradient"
                break

            accepted = False
            accepted_scores = None
            accepted_coords = None
            accepted_step = None
            accepted_hit_cap = False
            trial_logs: List[Dict[str, Any]] = []

            for ls in range(profile.line_search_tries):
                step_size = float(
                    profile.step_natural * (args.line_search_shrink ** ls)
                )
                cand_coords, trial_hit_cap = ora.apply_control_step(
                    coords=coords,
                    direction=direction,
                    step=step_size,
                    mode="free2d",
                    gt=target_rel,  # ignored by free2d, kept for API compatibility
                    max_total=profile.max_total_natural,
                    max_layer=profile.max_layer_natural,
                )
                delta = cand_coords - coords
                step_norm = float(np.linalg.norm(delta.reshape(-1)))
                if step_norm < 1e-10:
                    trial_logs.append({
                        "ls": int(ls),
                        "step_size": step_size,
                        "step_norm": step_norm,
                        "accepted": False,
                        "reason": "zero_step",
                    })
                    continue

                patch = ora.natural_coords_to_patch(
                    layers, geom, sub_pos, ref_pos, cand_coords
                )
                cand_scores = ora.all_scores(
                    model,
                    batch,
                    candidate_ids,
                    args.sequence_score_reduction,
                    decoder_layers,
                    patch,
                )
                cand_obj = ora.smooth_gt_objective(
                    cand_scores, target_rel, args.competitor_temperature
                )
                obj_gain = float(cand_obj - current_objective)
                improved = obj_gain >= args.min_objective_improvement
                trial_logs.append({
                    "ls": int(ls),
                    "step_size": step_size,
                    "step_norm": step_norm,
                    "objective": float(cand_obj),
                    "objective_gain": obj_gain,
                    "accepted": bool(improved),
                })
                if improved:
                    accepted = True
                    accepted_scores = cand_scores
                    accepted_coords = cand_coords
                    accepted_step = (ls, step_size, step_norm)
                    accepted_hit_cap = bool(trial_hit_cap)
                    break

            if not accepted:
                line_search_failed = True
                stop_reason = "line_search_failed"
                step_rows.append({
                    **common,
                    "step": int(step_idx),
                    "accepted": False,
                    "target_rel": target_rel,
                    "target_margin_before": float(current_margin),
                    "smooth_objective_before": float(current_objective),
                    "raw_spatial_gradient_norm": float(raw_grad_norm),
                    "accepted_step_natural": 0.0,
                    "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                    "tf_prediction_after": current_tf_pred,
                    "generation_prediction_after": current_gen_pred,
                    "line_search_trials": json.dumps(trial_logs),
                })
                break

            prev_coords = coords.copy()
            prev_obj = float(current_objective)
            coords = np.asarray(accepted_coords, dtype=np.float64)
            current_scores = dict(accepted_scores)
            current_tf_pred = ora.argmax_relation(current_scores)
            steps_taken = step_idx
            hit_total_cap = bool(hit_total_cap or accepted_hit_cap)

            run_generation = (
                step_idx % args.generation_check_every == 0
                or current_tf_pred == target_rel
                or step_idx == profile.max_steps
                or hit_total_cap
            )
            if run_generation:
                patch_now = ora.natural_coords_to_patch(
                    layers, geom, sub_pos, ref_pos, coords
                )
                current_gen_pred, current_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map=patch_now,
                    max_new_tokens=args.max_new_tokens,
                )

            dc = coords - prev_coords
            step_rows.append({
                **common,
                "step": int(step_idx),
                "accepted": True,
                "target_rel": target_rel,
                "target_margin_before": float(current_margin),
                "target_margin_after": float(ora.min_gt_margin(current_scores, target_rel)),
                "smooth_objective_before": prev_obj,
                "smooth_objective_after": float(
                    ora.smooth_gt_objective(
                        current_scores, target_rel, args.competitor_temperature
                    )
                ),
                "predicted_local_objective_gain": float(np.sum(grad2d * dc)),
                "raw_spatial_gradient_norm": float(raw_grad_norm),
                "accepted_line_search_index": int(accepted_step[0]),
                "accepted_nominal_step": float(accepted_step[1]),
                "accepted_step_natural": float(accepted_step[2]),
                "total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
                "tf_prediction_after": current_tf_pred,
                "generation_prediction_after": current_gen_pred if run_generation else None,
                "generation_target_after": bool(current_gen_pred == target_rel) if run_generation else None,
                "hit_total_cap": bool(hit_total_cap),
                "line_search_trials": json.dumps(trial_logs),
            })

            if current_gen_pred == target_rel:
                stop_reason = "generation_target"
                break
            if (
                profile.max_total_natural > 0
                and float(np.linalg.norm(coords.reshape(-1)))
                >= profile.max_total_natural - 1e-8
            ):
                stop_reason = "total_cap_reached"
                break

    # Final exact generation under the final patch.
    final_patch = (
        ora.natural_coords_to_patch(layers, geom, sub_pos, ref_pos, coords)
        if float(np.linalg.norm(coords.reshape(-1))) > EPS
        else {}
    )
    final_scores = ora.all_scores(
        model,
        batch,
        candidate_ids,
        args.sequence_score_reduction,
        decoder_layers,
        final_patch,
    )
    final_tf_pred = ora.argmax_relation(final_scores)
    final_gen_pred, final_gen_text = gate.generate_with_patch(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        batch=batch,
        patch_map=final_patch,
        max_new_tokens=args.max_new_tokens,
    )

    return {
        "patched_prediction": final_gen_pred,
        "patched_generation_text": final_gen_text,
        "final_tf_prediction": final_tf_pred,
        "final_target_margin": float(ora.min_gt_margin(final_scores, target_rel)),
        "steps_taken": int(steps_taken),
        "final_total_natural_norm": float(np.linalg.norm(coords.reshape(-1))),
        "hit_total_cap": bool(hit_total_cap),
        "line_search_failed": bool(line_search_failed),
        "stop_reason": stop_reason,
        **{f"final_score_{r}": float(final_scores[r]) for r in REL},
        **{
            f"final_x_L{L}_{axis}": float(coords[i, ai])
            for i, L in enumerate(layers)
            for ai, axis in enumerate(("H", "V"))
        },
    }


def main() -> None:
    a = parse_args()
    seed_all(a.seed)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    groups = ora.parse_layer_groups(a.layer_groups)
    needed_layers = sorted(set(L for _, ls in groups for L in ls))
    profiles = [PROFILE_SPECS[x] for x in a.profiles_parsed]

    # Fit residual spatial geometry first.
    geom, geometry_heldout_sids, geometry_meta = fit_controller_geometry(
        a, needed_layers, outdir
    )

    # Load repo COCO metadata used by the old controller.
    two, all_meta, rec_by_sid = ora.load_all_data(a)

    model = processor = None
    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        if max(needed_layers) >= len(decoder_layers):
            raise RuntimeError(
                f"Requested L{max(needed_layers)} but model has {len(decoder_layers)} decoder layers"
            )

        n_heads, head_dim = headscan.headprobe.scan_shape(model, decoder_layers)
        print(
            f"[MODEL] {spec.repo_id} decoder={decoder_path} layers={len(decoder_layers)} "
            f"heads/layer={n_heads} head_dim={head_dim}"
        )

        # ---------------------------------------------------------------
        # SOURCE-ONLY spatial-head selection.
        # ---------------------------------------------------------------
        syn_records = headscan.load_synthetic(a)
        coco_records = headscan.load_coco(a)

        source_cache = (
            Path(a.head_source_cache)
            if a.head_source_cache
            else outdir / "head_source_synthetic_vectors.npz"
        )
        target_cache = (
            Path(a.head_target_cache)
            if a.head_target_cache
            else outdir / "head_target_coco_vectors.npz"
        )

        source_pack = load_or_extract_head_pack(
            records=syn_records,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            dataset_name="synthetic",
            cache_path=source_cache,
            args=a,
        )
        target_pack = load_or_extract_head_pack(
            records=coco_records,
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            dataset_name="coco",
            cache_path=target_cache,
            args=a,
        )

        Xs = np.asarray(source_pack["vectors"])
        ys = np.asarray(source_pack["relation"], dtype=object)
        Xt = np.asarray(target_pack["vectors"])
        yt = np.asarray(target_pack["relation"], dtype=object)
        target_sids = np.asarray(target_pack["sid"], dtype=np.int64)

        ranking_df, (best_layer, best_head), center_all, dirs_all = rank_heads_source_only(
            Xs=Xs,
            ys=ys,
            mode=a.head_select_mode,
            folds=a.source_cv_folds,
            seed=a.seed,
        )
        ranking_df.to_csv(outdir / "synthetic_only_head_ranking.csv", index=False)
        best_row = ranking_df.iloc[0].to_dict()

        head_pred, head_margin = headscan.predict_one_head(
            Xt, center_all, dirs_all, best_layer, best_head
        )
        head_by_sid: Dict[int, Dict[str, Any]] = {}
        for i, sid in enumerate(target_sids.tolist()):
            head_by_sid[int(sid)] = {
                "pred": str(head_pred[i]),
                "margin": float(head_margin[i]),
                "gt_eval": str(yt[i]),
                "correct_eval": bool(head_pred[i] == yt[i]),
            }

        head_pred_df = pd.DataFrame([
            {
                "sid": sid,
                "head_pred": d["pred"],
                "head_margin": d["margin"],
                "gt_eval_only": d["gt_eval"],
                "head_correct_eval_only": d["correct_eval"],
            }
            for sid, d in sorted(head_by_sid.items())
        ])
        head_pred_df.to_csv(outdir / "selected_head_coco_predictions.csv", index=False)

        print("\n" + "=" * 180)
        print("SYNTHETIC-ONLY HEAD SELECTION")
        print("=" * 180)
        print(ranking_df.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(
            f"\nSELECTED HEAD = {headscan.head_name(best_layer, best_head)} | "
            f"source_select_acc={float(best_row['source_select_acc']):.4f} | "
            f"source_self_acc={float(best_row['source_self_acc']):.4f}"
        )
        print(
            f"COCO head accuracy (EVAL ONLY; not used for selection) = "
            f"{safe_acc(head_pred, yt):.4f}"
        )

        # ---------------------------------------------------------------
        # Select evaluation rows. For coco_calib geometry, use heldout only
        # to match the old 95% controller protocol.
        # ---------------------------------------------------------------
        eligible = [m for m in all_meta if int(m["sid"]) in head_by_sid]
        if a.geometry_mode == "coco_calib" and geometry_heldout_sids is not None:
            eligible = [m for m in eligible if int(m["sid"]) in geometry_heldout_sids]
        eval_meta = ora.stratified_cap(eligible, a.eval_max_samples, a.seed)
        if not eval_meta:
            raise RuntimeError("No evaluation samples after head/geometry alignment")

        candidate_texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, candidate_texts)
        device = torch.device(a.device)

        baseline_rows: List[Dict[str, Any]] = []
        result_rows: List[Dict[str, Any]] = []
        step_rows: List[Dict[str, Any]] = []

        print("\n" + "=" * 180)
        print("SYNTHETIC HEAD -> MID-LAYER SPATIAL CONTROLLER -> ACTUAL GENERATION")
        print("=" * 180)
        print(f"eval_N={len(eval_meta)} geometry={a.geometry_mode}")
        print(f"layer_groups={[x for x, _ in groups]}")
        print(f"modes={a.modes_parsed} profiles={a.profiles_parsed}")
        print(f"selected_head={headscan.head_name(best_layer, best_head)}")
        print(f"min_head_margin={a.min_head_margin}")
        print("HEAD mode never uses COCO GT to choose target relation.")
        print("ORACLE mode is diagnostic only.")
        print("=" * 180, flush=True)

        for m in tqdm(eval_meta, desc="EVAL head-guided spatial control"):
            sid = int(m["sid"])
            gt = str(m["gt"])
            image = batch = None
            try:
                hinfo = head_by_sid[sid]
                hpred = str(hinfo["pred"])
                hmargin = float(hinfo["margin"])
                hcorrect = bool(hpred == gt)  # evaluation only

                image = gate.base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = gate.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )
                sub_pos, ref_pos = ora.find_object_positions(
                    two, processor, batch, m["subject"], m["reference"]
                )

                base_scores = ora.all_scores(
                    model,
                    batch,
                    candidate_ids,
                    a.sequence_score_reduction,
                    decoder_layers,
                    {},
                )
                base_tf_pred = ora.argmax_relation(base_scores)
                base_gen_pred, base_gen_text = gate.generate_with_patch(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    patch_map={},
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = bool(base_gen_pred == gt)

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "head_pred": hpred,
                    "head_margin": hmargin,
                    "head_correct": hcorrect,
                    "baseline_prediction": base_gen_pred,
                    "baseline_correct": base_correct,
                    "baseline_tf_prediction": base_tf_pred,
                    "baseline_generation_text": base_gen_text,
                    **{f"baseline_score_{r}": float(base_scores[r]) for r in REL},
                })

                for group_label, layers in groups:
                    for mode in a.modes_parsed:
                        control_target = hpred if mode == "head" else gt
                        head_margin_gated = bool(
                            mode == "head" and hmargin < a.min_head_margin
                        )

                        for profile in profiles:
                            common = {
                                "sid": sid,
                                "gt": gt,
                                "head_pred": hpred,
                                "head_margin": hmargin,
                                "head_correct": hcorrect,
                                "mode": mode,
                                "profile": profile.name,
                                "layer_group": group_label,
                                "layers": ",".join(map(str, layers)),
                                "control_target": control_target,
                                "head_margin_gated": head_margin_gated,
                            }

                            if head_margin_gated:
                                res = {
                                    "patched_prediction": base_gen_pred,
                                    "patched_generation_text": base_gen_text,
                                    "final_tf_prediction": base_tf_pred,
                                    "final_target_margin": float(
                                        ora.min_gt_margin(base_scores, control_target)
                                    ),
                                    "steps_taken": 0,
                                    "final_total_natural_norm": 0.0,
                                    "hit_total_cap": False,
                                    "line_search_failed": False,
                                    "stop_reason": "head_margin_gate",
                                    **{f"final_score_{r}": float(base_scores[r]) for r in REL},
                                }
                                for L in layers:
                                    res[f"final_x_L{L}_H"] = 0.0
                                    res[f"final_x_L{L}_V"] = 0.0
                            else:
                                res = optimize_spatial_to_target(
                                    model=model,
                                    processor=processor,
                                    decoder_layers=decoder_layers,
                                    batch=batch,
                                    candidate_ids=candidate_ids,
                                    layers=layers,
                                    geom=geom,
                                    sub_pos=sub_pos,
                                    ref_pos=ref_pos,
                                    target_rel=control_target,
                                    base_scores=base_scores,
                                    base_gen_pred=base_gen_pred,
                                    base_gen_text=base_gen_text,
                                    profile=profile,
                                    args=a,
                                    common=common,
                                    step_rows=step_rows,
                                )

                            patched_pred = res["patched_prediction"]
                            result_rows.append({
                                **common,
                                "baseline_prediction": base_gen_pred,
                                "baseline_correct": base_correct,
                                "baseline_tf_prediction": base_tf_pred,
                                "patched_prediction": patched_pred,
                                "patched_correct": bool(patched_pred == gt),
                                "patched_matches_control_target": bool(
                                    patched_pred == control_target
                                ),
                                "baseline_matches_control_target": bool(
                                    base_gen_pred == control_target
                                ),
                                "baseline_generation_text": base_gen_text,
                                **res,
                            })

            except Exception as exc:
                err = {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, err)
                print(f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}", flush=True)
                if a.fail_fast:
                    raise
            finally:
                if batch is not None:
                    del batch
                if image is not None and hasattr(image, "close"):
                    try:
                        image.close()
                    except Exception:
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        baseline_df = pd.DataFrame(baseline_rows)
        result_df = pd.DataFrame(result_rows)
        step_df = pd.DataFrame(step_rows)
        summary_df = summarize_results(result_df)

        baseline_df.to_csv(outdir / "baseline.csv", index=False)
        result_df.to_csv(outdir / "per_condition.csv", index=False)
        step_df.to_csv(outdir / "per_step.csv", index=False)
        summary_df.to_csv(outdir / "summary.csv", index=False)

        byrel_rows: List[Dict[str, Any]] = []
        if len(result_df):
            for (mode, profile, group, rel), g in result_df.groupby(
                ["mode", "profile", "layer_group", "gt"], sort=False
            ):
                b = g["baseline_correct"].astype(bool).to_numpy()
                p = g["patched_correct"].astype(bool).to_numpy()
                byrel_rows.append({
                    "mode": mode,
                    "profile": profile,
                    "layer_group": group,
                    "relation": rel,
                    "N": int(len(g)),
                    "baseline_accuracy": float(np.mean(b)),
                    "patched_accuracy": float(np.mean(p)),
                    "gain": float(np.mean(p) - np.mean(b)),
                    "W2C": int(np.sum((~b) & p)),
                    "C2W": int(np.sum(b & (~p))),
                    "target_compliance": float(
                        g["patched_matches_control_target"].astype(bool).mean()
                    ),
                })
        pd.DataFrame(byrel_rows).to_csv(outdir / "by_relation.csv", index=False)

        # Cross-tab explains exactly where head routing helps/hurts relative to baseline.
        cross_rows: List[Dict[str, Any]] = []
        if len(baseline_df):
            for b_ok in (False, True):
                for h_ok in (False, True):
                    gg = baseline_df[
                        (baseline_df["baseline_correct"].astype(bool) == b_ok)
                        & (baseline_df["head_correct"].astype(bool) == h_ok)
                    ]
                    cross_rows.append({
                        "baseline_correct": b_ok,
                        "head_correct": h_ok,
                        "N": int(len(gg)),
                        "fraction": float(len(gg) / max(len(baseline_df), 1)),
                    })
        pd.DataFrame(cross_rows).to_csv(outdir / "baseline_vs_head_crosstab.csv", index=False)

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo": spec.repo_id,
            "decoder_path": decoder_path,
            "selected_head": headscan.head_name(best_layer, best_head),
            "selected_head_layer": int(best_layer),
            "selected_head_index": int(best_head),
            "head_select_mode": a.head_select_mode,
            "head_selection_uses_coco_gt": False,
            "source_cv_folds": int(a.source_cv_folds),
            "source_select_acc": float(best_row["source_select_acc"]),
            "source_self_acc": float(best_row["source_self_acc"]),
            "coco_head_acc_eval_only_all_extracted": float(safe_acc(head_pred, yt)),
            "head_control": a.control,
            "geometry": geometry_meta,
            "layer_groups": [{"label": label, "layers": layers} for label, layers in groups],
            "modes": a.modes_parsed,
            "profiles": [PROFILE_SPECS[x].__dict__ for x in a.profiles_parsed],
            "eval_N": int(len(baseline_df)),
            "eval_max_samples": int(a.eval_max_samples),
            "seed": int(a.seed),
            "min_head_margin": float(a.min_head_margin),
            "head_mode_uses_coco_gt_for_target": False,
            "oracle_mode_uses_coco_gt_for_target": "oracle" in a.modes_parsed,
            "controller_space": "2D H/V per layer; free2d across controlled layers",
            "arbitrary_hidden_dimensions_editable": False,
            "generation_is_actual_model_generate": True,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\n" + "=" * 180)
        print("FINAL SUMMARY")
        print("=" * 180)
        if len(baseline_df):
            print(
                f"baseline={baseline_df['baseline_correct'].astype(bool).mean():.4f} | "
                f"selected-head={baseline_df['head_correct'].astype(bool).mean():.4f} | "
                f"N={len(baseline_df)}"
            )
        if len(summary_df):
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("EMPTY")
        print(f"\nSaved to: {outdir}")

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
