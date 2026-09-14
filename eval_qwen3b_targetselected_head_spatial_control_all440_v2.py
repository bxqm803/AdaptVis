#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_qwen3b_targetselected_head_spatial_control_all440_v2.py

Qwen2.5-VL-3B / COCO_two: learn the relation directions ONLY on
Synthetic-400, transfer those frozen directions to every COCO attention head,
select the COCO head with the highest transferred direction accuracy, and then
use that head prediction to drive the mid-layer residual H/V spatial controller.

This is the exact diagnostic requested here:

    Synthetic-400
        -> fit a frozen L/R/A/B direction code for EVERY head

    COCO (all target samples)
        -> apply those frozen Synthetic directions to EVERY head
        -> use COCO GT only to choose the best transferred head
        -> selected head predicts r_hat for each sample

    Synthetic-400 residual states
        -> define the L20-26 residual H/V control geometry

    COCO generation
        -> use r_hat as the target of the H/V controller
        -> edit ONLY subject/reference residual spatial coordinates
        -> continue the original model and call model.generate()

Thus the spatial DIRECTIONS and residual H/V GEOMETRY are source-only, but the
HEAD ID is target-supervised.  This is intentionally a diagnostic / oracle head
selection experiment, not a fully label-free method.  Per-sample intervention
routing in `head` mode does NOT use COCO GT after the head has been selected.

Default run is the requested all-440 experiment:

CUDA_VISIBLE_DEVICES=0 python -u eval_qwen3b_targetselected_head_spatial_control_all440_v2.py \
  --head-source-cache output/qwen3b_synhead_spatial_control_syngeom_n80_v1/head_source_synthetic_vectors.npz \
  --head-target-cache output/qwen3b_synhead_spatial_control_syngeom_n80_v1/head_target_coco_vectors.npz \
  --source-spatial-npz output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --output-dir output/qwen3b_targetselected_head_spatial_control_all440_v2 \
  --overwrite

Expected head-selection protocol matches the earlier multi-model table:
Synthetic-frozen relation code + full-target supervised head selection.
For Qwen3B/COCO this should recover the same best head as that scan if the same
cache/control/pooling conventions are used.
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
    import scan_synthetic_frozen_direction_heads_multimodel_v1 as headscan
except Exception as exc:
    raise SystemExit(
        "Could not import AdaptVis dependencies. Run from the llava16 repo root and keep:\n"
        "  eval_real_causal_token_update_gating_v1.py\n"
        "  eval_oracle_multilayer_spatial_logit_optimization_v2_multicomp.py\n"
        "  scan_synthetic_frozen_direction_heads_multimodel_v1.py\n"
        f"available.\n{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
EPS = 1e-12
SCRIPT_VERSION = "qwen3b-targetselected-head-spatial-control-all440-v2"


def _norm_rel_local(x) -> str:
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s)


def _unit_local(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return v / n


def load_state_npz_local(path: Path, *, require_labels: bool):
    """Load relation_vectors or old img/no_image cache as a relation state."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            vector_definition = (
                str(z["vector_definition"].item())
                if "vector_definition" in keys else "relation_vectors"
            )
        elif {"img", "no_image"}.issubset(keys):
            X = np.asarray(z["img"], dtype=np.float32) - np.asarray(z["no_image"], dtype=np.float32)
            vector_definition = "img_minus_no_image"
        else:
            raise RuntimeError(
                f"{path} must contain relation_vectors OR both img and no_image; keys={sorted(keys)}"
            )

        if "decoder_block_index" not in keys:
            raise RuntimeError(f"{path} missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]
        sids = (
            np.asarray(z["sample_index"], dtype=np.int64)
            if "sample_index" in keys else np.arange(X.shape[0], dtype=np.int64)
        )
        labels = None
        if "relation" in keys:
            labels = np.asarray([_norm_rel_local(v) for v in z["relation"].tolist()], dtype=object)
        elif require_labels:
            raise RuntimeError(f"{path} requires relation labels for source geometry")

    if X.ndim != 3:
        raise RuntimeError(f"Bad state shape {X.shape}; expected [N,L,D]")
    if X.shape[0] != len(sids):
        raise RuntimeError("X/sample_index length mismatch")
    if labels is not None and len(labels) != X.shape[0]:
        raise RuntimeError("X/relation length mismatch")
    return X, labels, layers, sids, vector_definition


def fit_source_geometry_local(X, y, layers, fit_layers: Sequence[int]):
    """Fit source-only H/V natural-coordinate geometry without external helpers."""
    if y is None:
        raise RuntimeError("Source labels required")
    layer_to_i = {int(L): i for i, L in enumerate(layers)}
    missing = [int(L) for L in fit_layers if int(L) not in layer_to_i]
    if missing:
        raise RuntimeError(f"Source NPZ missing layers {missing}; has {layers}")

    geom = {}
    rows = []
    for L in fit_layers:
        li = layer_to_i[int(L)]
        Xf = X[:, li].astype(np.float64)
        center = Xf.mean(axis=0)
        means = {}
        for r in REL:
            mask = y == r
            if not np.any(mask):
                raise RuntimeError(f"Source has no samples for relation={r}")
            means[r] = Xf[mask].mean(axis=0)

        class_dirs = {r: _unit_local(means[r] - center) for r in REL}
        dH = _unit_local(class_dirs["right"] - class_dirs["left"])
        dV = _unit_local(class_dirs["above"] - class_dirs["below"])
        gapH = float(np.dot(means["right"] - means["left"], dH))
        gapV = float(np.dot(means["above"] - means["below"], dV))
        if gapH < 0:
            dH, gapH = -dH, -gapH
        if gapV < 0:
            dV, gapV = -dV, -gapV

        B = np.stack([dH, dV], axis=1)
        dual = B @ np.linalg.inv(B.T @ B)
        halfH = max(gapH / 2.0, EPS)
        halfV = max(gapV / 2.0, EPS)
        natural_M = dual @ np.diag([halfH, halfV])

        geom[int(L)] = {
            "layer_index": int(li),
            "center": center.astype(np.float32),
            "B": B.astype(np.float32),
            "dual": dual.astype(np.float32),
            "natural_M": natural_M.astype(np.float32),
            "natural_half_H": float(halfH),
            "natural_half_V": float(halfV),
        }
        rows.append({
            "source_layer": int(L),
            "fit_N": int(X.shape[0]),
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "natural_full_gap_H": float(gapH),
            "natural_half_gap_H": float(halfH),
            "natural_full_gap_V": float(gapV),
            "natural_half_gap_V": float(halfV),
            "dual_check_max_abs": float(np.max(np.abs(B.T @ dual - np.eye(2)))),
        })
    return geom, pd.DataFrame(rows)


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
        default="target_full",
        choices=["target_full", "source_cv", "source_self"],
        help=("target_full: fit directions on Synthetic-400, then use full COCO GT only to choose "
              "the best transferred head; source_cv/source_self keep the older source-only selection."),
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
        default="synthetic_source",
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
    p.add_argument("--profiles", default="wide", help="Comma-separated: old,wide,verywide")
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
    p.add_argument("--eval-max-samples", type=int, default=0, help="0 = all eligible COCO samples")
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

    Xs, ys, source_layers, source_sids, source_def = load_state_npz_local(
        Path(args.source_spatial_npz), require_labels=True
    )
    geom, axis_df = fit_source_geometry_local(Xs, ys, source_layers, needed_layers)
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
        # Synthetic-frozen direction code; head ID may be selected on COCO GT.
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

        if a.head_select_mode == "target_full":
            # Fit relation directions ONLY on Synthetic-400.  Then apply those
            # frozen directions to every COCO head and use COCO GT only to pick
            # the best transferred head.  This exactly matches the broad-sweep
            # diagnostic used for the current Generation-vs-Spatial-Head table.
            center_all, dirs_all = headscan.fit_source_codebooks(Xs, ys)
            all_idx = np.arange(len(yt), dtype=np.int64)
            ranking_rows, pred_map = headscan.rank_heads(
                X_source=Xs,
                y_source=ys,
                X_target=Xt,
                y_target=yt,
                center=center_all,
                dirs=dirs_all,
                selection_idx=all_idx,
                test_idx=all_idx,
            )
            ranking_df = pd.DataFrame(ranking_rows)
            ranking_df.to_csv(outdir / "synthetic_frozen_coco_head_ranking.csv", index=False)
            best_row = ranking_rows[0]
            best_layer = int(best_row["layer"])
            best_head = int(best_row["head"])
            head_pred, head_margin = pred_map[(best_layer, best_head)]

            print("\n" + "=" * 180)
            print("SYNTHETIC-FROZEN DIRECTIONS -> FULL-COCO HEAD SELECTION")
            print("=" * 180)
            cols = [
                "rank", "head_name", "syn_self_acc", "selection_acc",
                "all_target_acc", "left_acc", "right_acc", "above_acc",
                "below_acc", "mean_margin",
            ]
            print(ranking_df[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(
                f"\nSELECTED HEAD = {headscan.head_name(best_layer, best_head)} | "
                f"COCO transferred-direction acc={float(best_row['all_target_acc']):.4f} | "
                f"Synthetic self acc={float(best_row['syn_self_acc']):.4f}"
            )
            print(
                "[NOTE] COCO GT is used ONLY to choose the head ID. "
                "The four relation directions remain frozen from Synthetic-400."
            )
        else:
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

        if a.head_select_mode != "target_full":
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
        print("HEAD mode uses the selected head prediction as the per-sample target; it does not use per-sample COCO GT for routing.")
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
            "direction_fit_uses_coco_gt": False,
            "head_selection_uses_coco_gt": bool(a.head_select_mode == "target_full"),
            "head_selection_target_fraction": 1.0 if a.head_select_mode == "target_full" else 0.0,
            "source_cv_folds": int(a.source_cv_folds),
            "source_select_acc": (
                float(best_row["source_select_acc"])
                if "source_select_acc" in best_row else None
            ),
            "source_self_acc": float(
                best_row.get("source_self_acc", best_row.get("syn_self_acc", float("nan")))
            ),
            "selected_head_coco_transfer_acc": float(safe_acc(head_pred, yt)),
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
