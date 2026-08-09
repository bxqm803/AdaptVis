#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic linear decomposition of COCO raw-token hidden states using GroundingDINO bboxes.

Requested component table
-------------------------
  subject_id    : A category one-hot
  reference_id  : B category one-hot
  abs_A         : [cx_A, cy_A]
  abs_B         : [cx_B, cy_B]
  relative      : [dx, dy] = [cx_A-cx_B, cy_A-cy_B]
  size_A        : [w_A, h_A]
  size_B        : [w_B, h_B]
  relation      : one-hot(left/right/above/below)

Targets
-------
  A_raw / A_noimg / A_residual
  B_raw / B_noimg / B_residual
  last_raw / last_noimg / last_residual
  Diff_raw / Diff_noimg / Diff_residual, where Diff = A-B

For every target, on held-out test samples, report:
  * full reconstruction: cosine, centered cosine, R^2, relative error
  * group-only R^2
  * drop-one unique R^2 = R^2(full) - R^2(full without group)
  * permutation-Shapley R^2 allocation across the eight requested groups
  * component mean norm, norm ratio, component-to-target cosine
  * matched-minus-shuffled cosine

IMPORTANT IDENTIFIABILITY WARNING
---------------------------------
abs_A, abs_B and relative are exactly linearly dependent because

    relative = abs_A - abs_B.

Therefore their individual ridge coefficients / unique R^2 / Shapley credit are
NOT a unique physical decomposition.  The script still reports them because they
are the requested components, but it ALSO reports the combined group:

    position_family = abs_A + abs_B + relative

which is much safer to interpret.  If you want a non-redundant geometric basis,
use pair_center + relative instead of abs_A + abs_B + relative.

This is a correlational encoding/decomposition analysis on natural images, not a
causal image intervention.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_ALIASES = {
    "left": "left", "right": "right", "above": "above", "below": "below",
    "on": "above", "top": "above", "under": "below", "underneath": "below",
    "bottom": "below",
}
SLOTS = ("A", "B", "last")
REQUESTED_GROUPS = (
    "subject_id", "reference_id", "abs_A", "abs_B",
    "relative", "size_A", "size_B", "relation",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--state-dir", required=True,
                   help="Directory containing raw__correct__all_layers.npz and raw__no_image__all_layers.npz")
    p.add_argument("--bbox-jsonl", default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fixed-layer", type=int, default=None,
                   help="Decoder block index. If omitted, choose from raw A-B relation centroid ACC.")
    p.add_argument("--layer-select-train-ratio", type=float, default=0.15)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="lambda = ridge * trace(X'X)/p")
    p.add_argument("--shapley-permutations", type=int, default=32,
                   help="Monte-Carlo permutations per split for Shapley R2. Set 0 to skip.")
    p.add_argument("--shuffle-repeats", type=int, default=10)
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument("--require-gt-consistent", action="store_true",
                   help="Sensitivity analysis only; conditions on GT relation.")
    return p.parse_args()


def norm_relation(x: Any) -> str:
    k = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(k, k)


def canonical_phrase(x: Any) -> str:
    s = " ".join(str(x).lower().strip().split())
    for prefix in ("the ", "a ", "an "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip(" .,!?:;\t\n")


def load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def align_states(correct: Mapping[str, Any], noimg: Mapping[str, Any]):
    required = {
        "sample_index", "relation", "subject", "reference", "decoder_block_index",
        "A_vectors", "B_vectors", "last_vectors",
    }
    for name, obj in (("correct", correct), ("noimg", noimg)):
        missing = sorted(required - set(obj.keys()))
        if missing:
            raise KeyError(f"{name} state NPZ missing keys: {missing}")

    cids = np.asarray(correct["sample_index"], dtype=np.int64)
    nids = np.asarray(noimg["sample_index"], dtype=np.int64)
    cm = {int(s): i for i, s in enumerate(cids.tolist())}
    nm = {int(s): i for i, s in enumerate(nids.tolist())}
    common = np.asarray([int(s) for s in cids.tolist() if int(s) in nm], dtype=np.int64)
    ci = np.asarray([cm[int(s)] for s in common], dtype=np.int64)
    ni = np.asarray([nm[int(s)] for s in common], dtype=np.int64)

    layers_c = np.asarray(correct["decoder_block_index"], dtype=np.int64)
    layers_n = np.asarray(noimg["decoder_block_index"], dtype=np.int64)
    if not np.array_equal(layers_c, layers_n):
        raise ValueError("correct/noimg layer lists differ")
    layer_to_idx = {int(v): i for i, v in enumerate(layers_c.tolist())}

    states: Dict[str, np.ndarray] = {}
    for slot, key in (("A", "A_vectors"), ("B", "B_vectors"), ("last", "last_vectors")):
        c = np.asarray(correct[key][ci], dtype=np.float32)
        n = np.asarray(noimg[key][ni], dtype=np.float32)
        states[f"{slot}_raw"] = c
        states[f"{slot}_noimg"] = n
        states[f"{slot}_residual"] = c - n

    relation = np.asarray([norm_relation(x) for x in np.asarray(correct["relation"], dtype=object)[ci]], dtype=object)
    subject = np.asarray([canonical_phrase(x) for x in np.asarray(correct["subject"], dtype=object)[ci]], dtype=object)
    reference = np.asarray([canonical_phrase(x) for x in np.asarray(correct["reference"], dtype=object)[ci]], dtype=object)
    image_id = np.asarray(correct.get("image_id", np.asarray([""] * len(cids), dtype=object)), dtype=object)[ci]
    return common, states, relation, subject, reference, image_id, layers_c, layer_to_idx


def load_gdino_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows[int(row["sid"])] = row
            except Exception as exc:
                raise ValueError(f"Bad bbox JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def parse_selected_box(obj: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    selected = obj.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("missing selected bbox")
    box = selected.get("box_xyxy_normalized")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("selected bbox has no box_xyxy_normalized[4]")
    b = np.asarray([float(v) for v in box], dtype=np.float64)
    if not np.all(np.isfinite(b)):
        raise ValueError("bbox contains non-finite values")
    b = np.clip(b, 0.0, 1.0)
    x1, y1, x2, y2 = b.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate bbox: {b.tolist()}")
    return b, float(selected.get("score", float("nan")))


def box_stats(b: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    return 0.5*(x1+x2), 0.5*(y1+y2), x2-x1, y2-y1


def relation_geometry_consistent(rel: str, dx: float, dy: float) -> bool:
    if rel == "left": return dx < 0
    if rel == "right": return dx > 0
    if rel == "above": return dy < 0
    if rel == "below": return dy > 0
    return False


def build_geometry_table(
    sids: np.ndarray,
    relation: np.ndarray,
    subject: np.ndarray,
    reference: np.ndarray,
    image_id: np.ndarray,
    gdino: Mapping[int, Mapping[str, Any]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows, audit = [], []
    for i, sid0 in enumerate(sids.tolist()):
        sid = int(sid0)
        g = gdino.get(sid)
        if g is None:
            audit.append({"sid": sid, "status": "skip", "reason": "bbox_sid_missing"})
            continue
        try:
            if not bool(g.get("both_found", False)):
                raise ValueError("both_found=false")
            so, ro = g.get("subject", {}), g.get("reference", {})
            if not isinstance(so, Mapping) or not isinstance(ro, Mapping):
                raise ValueError("subject/reference block missing")
            s_amb, r_amb = bool(so.get("ambiguous", False)), bool(ro.get("ambiguous", False))
            if (s_amb or r_amb) and not args.include_ambiguous:
                raise ValueError("ambiguous_bbox")
            ba, score_a = parse_selected_box(so)
            bb, score_b = parse_selected_box(ro)
            if not np.isfinite(score_a) or not np.isfinite(score_b):
                raise ValueError("nonfinite score")
            if score_a < args.min_score or score_b < args.min_score:
                raise ValueError(f"score_below_min:A={score_a:.4f},B={score_b:.4f}")
            cxA, cyA, wA, hA = box_stats(ba)
            cxB, cyB, wB, hB = box_stats(bb)
            dx, dy = cxA-cxB, cyA-cyB
            rel = norm_relation(relation[i])
            consistent = relation_geometry_consistent(rel, dx, dy)
            if args.require_gt_consistent and not consistent:
                raise ValueError("bbox_center_sign_disagrees_with_gt")
            rows.append({
                "sid": sid,
                "image_id": str(image_id[i]),
                "subject": canonical_phrase(subject[i]),
                "reference": canonical_phrase(reference[i]),
                "relation": rel,
                "subject_score": score_a,
                "reference_score": score_b,
                "subject_ambiguous": s_amb,
                "reference_ambiguous": r_amb,
                "bbox_relation_consistent": consistent,
                "cxA": cxA, "cyA": cyA, "wA": wA, "hA": hA,
                "cxB": cxB, "cyB": cyB, "wB": wB, "hB": hB,
                "dx": dx, "dy": dy,
                "pair_cx": 0.5*(cxA+cxB), "pair_cy": 0.5*(cyA+cyB),
            })
        except Exception as exc:
            audit.append({"sid": sid, "status": "skip", "reason": str(exc)})
    return pd.DataFrame(rows), audit


def one_hot(values: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    vals = [str(v) for v in values]
    vocab = sorted(set(vals))
    lookup = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(vals), len(vocab)), dtype=np.float64)
    for i, v in enumerate(vals):
        X[i, lookup[v]] = 1.0
    return X, vocab


def build_requested_design(df: pd.DataFrame):
    Xs, svocab = one_hot(df["subject"].tolist())
    Xr, rvocab = one_hot(df["reference"].tolist())
    Xrel, relvocab = one_hot(df["relation"].tolist())

    blocks: List[np.ndarray] = []
    groups: Dict[str, List[int]] = {}
    names: List[str] = []
    offset = 0

    def add_group(name: str, arr: np.ndarray, cols: Sequence[str]):
        nonlocal offset
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        blocks.append(arr)
        groups[name] = list(range(offset, offset + arr.shape[1]))
        names.extend([f"{name}:{c}" for c in cols])
        offset += arr.shape[1]

    add_group("subject_id", Xs, svocab)
    add_group("reference_id", Xr, rvocab)
    add_group("abs_A", df[["cxA", "cyA"]].to_numpy(), ("cxA", "cyA"))
    add_group("abs_B", df[["cxB", "cyB"]].to_numpy(), ("cxB", "cyB"))
    add_group("relative", df[["dx", "dy"]].to_numpy(), ("dx", "dy"))
    add_group("size_A", df[["wA", "hA"]].to_numpy(), ("wA", "hA"))
    add_group("size_B", df[["wB", "hB"]].to_numpy(), ("wB", "hB"))
    add_group("relation", Xrel, relvocab)

    X = np.concatenate(blocks, axis=1)
    vocabs = {"subject": svocab, "reference": rvocab, "relation": relvocab}
    return X, groups, names, vocabs


def make_stratified_splits(y: Sequence[str], train_ratio: float, repeats: int, seed: int):
    y = np.asarray(y, dtype=object)
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(repeats):
        tr_parts = []
        for c in RELATIONS:
            idx = np.where(y == c)[0].copy()
            if len(idx) < 2:
                raise ValueError(f"Need >=2 samples for {c}")
            rng.shuffle(idx)
            ntr = int(round(train_ratio * len(idx)))
            ntr = max(1, min(len(idx)-1, ntr))
            tr_parts.append(idx[:ntr])
        tr = np.sort(np.concatenate(tr_parts))
        mask = np.ones(len(y), dtype=bool)
        mask[tr] = False
        te = np.where(mask)[0]
        splits.append((tr, te))
    return splits


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    return np.sum(a*b, axis=1) / np.maximum(an*bn, EPS)


def relative_error(pred_c: np.ndarray, target_c: np.ndarray) -> float:
    return float(np.linalg.norm(pred_c-target_c) / max(float(np.linalg.norm(target_c)), EPS))


def global_r2(pred: np.ndarray, target: np.ndarray, train_mean: np.ndarray) -> float:
    sse = float(np.sum((target-pred)**2))
    sst = float(np.sum((target-train_mean)**2))
    return 1.0 - sse/max(sst, EPS)


def centroid_relation_acc(X: np.ndarray, y: np.ndarray, splits) -> Tuple[float, float]:
    vals = []
    for tr, te in splits:
        center = X[tr].mean(axis=0, keepdims=True)
        dirs = []
        for c in RELATIONS:
            v = X[tr][y[tr] == c].mean(axis=0) - center[0]
            v = v / max(float(np.linalg.norm(v)), EPS)
            dirs.append(v)
        D = np.stack(dirs, axis=0)
        pred = np.argmax(normalize_rows(X[te]-center) @ D.T, axis=1)
        gt = np.asarray([RELATIONS.index(str(v)) for v in y[te]], dtype=np.int64)
        vals.append(float(np.mean(pred == gt)))
    return float(np.mean(vals)), float(np.std(vals))


def select_layer(states: Mapping[str, np.ndarray], y: np.ndarray, layers: np.ndarray, args: argparse.Namespace):
    diff = np.asarray(states["A_raw"] - states["B_raw"], dtype=np.float64)
    splits = make_stratified_splits(y, args.layer_select_train_ratio, args.repeats, args.seed+300)
    rows = []
    for j, L in enumerate(layers.tolist()):
        m, s = centroid_relation_acc(diff[:, j], y, splits)
        rows.append({"layer": int(L), "acc_mean": m, "acc_std": s})
    df = pd.DataFrame(rows)
    best = int(df.loc[df["acc_mean"].idxmax(), "layer"])
    fixed = best if args.fixed_layer is None else int(args.fixed_layer)
    if fixed not in set(map(int, layers.tolist())):
        raise ValueError(f"L{fixed} unavailable; layers={layers.tolist()}")
    return fixed, df


def group_indices(groups: Mapping[str, Sequence[int]], names: Iterable[str]) -> List[int]:
    out: List[int] = []
    for name in names:
        out.extend(groups[name])
    return sorted(out)


def ridge_fit_predict(
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    ridge: float,
    feature_indices: Sequence[int],
):
    idx = np.asarray(list(feature_indices), dtype=np.int64)
    ymu = Y[tr].mean(axis=0, keepdims=True)
    Ytr_c = Y[tr] - ymu
    Yte = Y[te]
    Yte_c = Yte - ymu

    if len(idx) == 0:
        pred_c = np.zeros_like(Yte_c)
        return {
            "pred": np.repeat(ymu, len(te), axis=0), "pred_c": pred_c,
            "target": Yte, "target_c": Yte_c, "ymu": ymu,
            "W": np.zeros((0, Y.shape[1]), dtype=np.float64),
            "xmu": np.zeros((0,), dtype=np.float64), "xsd": np.ones((0,), dtype=np.float64),
            "idx": idx,
        }

    raw_tr, raw_te = X[tr][:, idx], X[te][:, idx]
    xmu = raw_tr.mean(axis=0)
    xsd = raw_tr.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xtr, Xte = (raw_tr-xmu)/xsd, (raw_te-xmu)/xsd

    gram = Xtr.T @ Xtr
    lam = float(ridge) * float(np.trace(gram) / max(len(idx), 1))
    A = gram + lam*np.eye(len(idx), dtype=np.float64)
    try:
        W = np.linalg.solve(A, Xtr.T @ Ytr_c)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ Xtr.T @ Ytr_c
    pred_c = Xte @ W
    return {
        "pred": ymu + pred_c, "pred_c": pred_c,
        "target": Yte, "target_c": Yte_c, "ymu": ymu,
        "W": W, "xmu": xmu, "xsd": xsd, "idx": idx,
    }


def fit_metrics(f: Mapping[str, Any]) -> Dict[str, float]:
    pred, pc = np.asarray(f["pred"]), np.asarray(f["pred_c"])
    target, tc, ymu = np.asarray(f["target"]), np.asarray(f["target_c"]), np.asarray(f["ymu"])
    return {
        "full_cos": float(np.mean(cosine_rows(pred, target))),
        "centered_cos": float(np.mean(cosine_rows(pc, tc))),
        "r2": global_r2(pred, target, ymu),
        "relative_error": relative_error(pc, tc),
    }


def component_from_fit(f: Mapping[str, Any], X: np.ndarray, te: np.ndarray, selected_feature_indices: Sequence[int]) -> np.ndarray:
    fit_idx = np.asarray(f["idx"], dtype=np.int64)
    local = {int(global_i): local_i for local_i, global_i in enumerate(fit_idx.tolist())}
    chosen_global = [int(i) for i in selected_feature_indices if int(i) in local]
    if not chosen_global:
        return np.zeros_like(f["target_c"])
    chosen_local = np.asarray([local[i] for i in chosen_global], dtype=np.int64)
    raw = X[te][:, fit_idx]
    z = (raw-np.asarray(f["xmu"]))/np.asarray(f["xsd"])
    return z[:, chosen_local] @ np.asarray(f["W"])[chosen_local]


def shuffled_delta(comp: np.ndarray, target_c: np.ndarray, repeats: int, rng: np.random.Generator) -> float:
    matched = float(np.mean(cosine_rows(comp, target_c)))
    if len(comp) < 2 or repeats <= 0:
        return float("nan")
    sh = []
    for _ in range(repeats):
        sh.append(float(np.mean(cosine_rows(comp[rng.permutation(len(comp))], target_c))))
    return matched - float(np.mean(sh))


def design_diagnostics(X: np.ndarray, groups: Mapping[str, Sequence[int]], tr: np.ndarray) -> Dict[str, Any]:
    xmu = X[tr].mean(axis=0)
    xsd = X[tr].std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Z = (X[tr]-xmu)/xsd
    rank = int(np.linalg.matrix_rank(Z))
    p = int(Z.shape[1])
    # requested geometry dependency: relative == abs_A - abs_B, tested in raw coordinates
    a = X[tr][:, groups["abs_A"]]
    b = X[tr][:, groups["abs_B"]]
    r = X[tr][:, groups["relative"]]
    dep_err = float(np.max(np.abs((a-b)-r)))
    return {"n_train": int(len(tr)), "p": p, "rank": rank, "rank_deficiency": p-rank,
            "max_abs_relative_minus_absA_plus_absB_error": dep_err}


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(a)), float(np.nanstd(a))


def mc_shapley_r2(
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    ridge: float,
    groups: Mapping[str, Sequence[int]],
    group_names: Sequence[str],
    permutations: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if permutations <= 0:
        return {g: float("nan") for g in group_names}

    # Cache subset R2 inside this target/split. A subset is represented by sorted group names.
    cache: Dict[Tuple[str, ...], float] = {}

    def score(subset: Sequence[str]) -> float:
        key = tuple(sorted(subset))
        if key not in cache:
            idx = group_indices(groups, key)
            cache[key] = fit_metrics(ridge_fit_predict(X, Y, tr, te, ridge, idx))["r2"]
        return cache[key]

    phi = {g: 0.0 for g in group_names}
    for _ in range(permutations):
        order = list(group_names)
        rng.shuffle(order)
        S: List[str] = []
        prev = score(S)
        for g in order:
            S.append(g)
            cur = score(S)
            phi[g] += cur-prev
            prev = cur
    return {g: phi[g]/permutations for g in group_names}


def analyze_target(
    target_name: str,
    Y: np.ndarray,
    X: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    splits,
    args: argparse.Namespace,
):
    full_idx = list(range(X.shape[1]))
    summary_rows, component_rows, split_rows = [], [], []
    rng = np.random.default_rng(args.seed + 1009 + abs(hash(target_name)) % 100000)

    tracked_groups = list(REQUESTED_GROUPS)
    # This aggregate is NOT an extra model feature; it is the union of the three collinear position groups.
    aggregate_groups = {"position_family": group_indices(groups, ("abs_A", "abs_B", "relative"))}

    per_metric: Dict[str, List[float]] = {k: [] for k in ("full_cos", "centered_cos", "r2", "relative_error", "composition_error")}
    group_stats: Dict[str, Dict[str, List[float]]] = {
        g: {k: [] for k in ("group_only_r2", "drop_one_unique_r2", "shapley_r2", "component_cos",
                                 "component_norm", "component_norm_ratio", "matched_minus_shuffled")}
        for g in tracked_groups
    }
    agg_stats = {g: {k: [] for k in ("group_only_r2", "drop_one_unique_r2", "component_cos",
                                     "component_norm", "component_norm_ratio", "matched_minus_shuffled")}
                 for g in aggregate_groups}

    for rep, (tr, te) in enumerate(splits):
        full = ridge_fit_predict(X, Y, tr, te, args.ridge, full_idx)
        fm = fit_metrics(full)
        for k in ("full_cos", "centered_cos", "r2", "relative_error"):
            per_metric[k].append(fm[k])

        # Full-model additive components. Their sum must reconstruct pred_c up to numerical precision.
        components = {g: component_from_fit(full, X, te, groups[g]) for g in tracked_groups}
        comp_sum = np.sum(np.stack([components[g] for g in tracked_groups], axis=0), axis=0)
        comp_err = float(np.linalg.norm(comp_sum-full["pred_c"]) / max(float(np.linalg.norm(full["pred_c"])), EPS))
        per_metric["composition_error"].append(comp_err)

        shap = mc_shapley_r2(X, Y, tr, te, args.ridge, groups, tracked_groups,
                             args.shapley_permutations, rng)

        for g in tracked_groups:
            only = ridge_fit_predict(X, Y, tr, te, args.ridge, groups[g])
            without = [i for i in full_idx if i not in set(groups[g])]
            wo = ridge_fit_predict(X, Y, tr, te, args.ridge, without)
            c = components[g]
            tc = np.asarray(full["target_c"])
            norms = np.linalg.norm(c, axis=1)
            tnorms = np.linalg.norm(tc, axis=1)
            group_stats[g]["group_only_r2"].append(fit_metrics(only)["r2"])
            group_stats[g]["drop_one_unique_r2"].append(fm["r2"] - fit_metrics(wo)["r2"])
            group_stats[g]["shapley_r2"].append(shap[g])
            group_stats[g]["component_cos"].append(float(np.mean(cosine_rows(c, tc))))
            group_stats[g]["component_norm"].append(float(np.mean(norms)))
            group_stats[g]["component_norm_ratio"].append(float(np.mean(norms / np.maximum(tnorms, EPS))))
            group_stats[g]["matched_minus_shuffled"].append(shuffled_delta(c, tc, args.shuffle_repeats, rng))

        for g, idxs in aggregate_groups.items():
            only = ridge_fit_predict(X, Y, tr, te, args.ridge, idxs)
            without = [i for i in full_idx if i not in set(idxs)]
            wo = ridge_fit_predict(X, Y, tr, te, args.ridge, without)
            c = component_from_fit(full, X, te, idxs)
            tc = np.asarray(full["target_c"])
            norms = np.linalg.norm(c, axis=1)
            tnorms = np.linalg.norm(tc, axis=1)
            agg_stats[g]["group_only_r2"].append(fit_metrics(only)["r2"])
            agg_stats[g]["drop_one_unique_r2"].append(fm["r2"] - fit_metrics(wo)["r2"])
            agg_stats[g]["component_cos"].append(float(np.mean(cosine_rows(c, tc))))
            agg_stats[g]["component_norm"].append(float(np.mean(norms)))
            agg_stats[g]["component_norm_ratio"].append(float(np.mean(norms / np.maximum(tnorms, EPS))))
            agg_stats[g]["matched_minus_shuffled"].append(shuffled_delta(c, tc, args.shuffle_repeats, rng))

        split_rows.append({
            "target": target_name, "repeat": rep, "n_train": len(tr), "n_test": len(te),
            **{k: per_metric[k][-1] for k in per_metric},
        })

    srow = {"target": target_name}
    for k, vals in per_metric.items():
        m, s = mean_std(vals)
        srow[f"{k}_mean"] = m
        srow[f"{k}_std"] = s
    summary_rows.append(srow)

    for g in tracked_groups:
        row = {"target": target_name, "group": g, "group_type": "requested"}
        for k, vals in group_stats[g].items():
            m, s = mean_std(vals)
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s
        component_rows.append(row)

    for g in aggregate_groups:
        row = {"target": target_name, "group": g, "group_type": "aggregate_nonunique_position_family"}
        for k, vals in agg_stats[g].items():
            m, s = mean_std(vals)
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s
        row["shapley_r2_mean"] = float("nan")
        row["shapley_r2_std"] = float("nan")
        component_rows.append(row)

    return summary_rows, component_rows, split_rows


def main():
    args = parse_args()
    state_dir = Path(args.state_dir)
    bbox_path = Path(args.bbox_jsonl)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    correct = load_npz(state_dir / "raw__correct__all_layers.npz")
    noimg = load_npz(state_dir / "raw__no_image__all_layers.npz")
    sids, states, relation, subject, reference, image_id, layers, layer_to_idx = align_states(correct, noimg)

    valid_rel = np.asarray([r in RELATIONS for r in relation], dtype=bool)
    sids, relation, subject, reference, image_id = sids[valid_rel], relation[valid_rel], subject[valid_rel], reference[valid_rel], image_id[valid_rel]
    for k in list(states.keys()):
        states[k] = states[k][valid_rel]

    fixed, scan_df = select_layer(states, relation, layers, args)
    scan_df.to_csv(out / "baseline_layer_scan.csv", index=False)
    li = layer_to_idx[fixed]

    gdino = load_gdino_rows(bbox_path)
    geom_df, audit = build_geometry_table(sids, relation, subject, reference, image_id, gdino, args)
    if len(geom_df) < 40:
        raise RuntimeError(f"Too few usable bbox samples: {len(geom_df)}")

    sid_to_state = {int(s): i for i, s in enumerate(sids.tolist())}
    ridx = np.asarray([sid_to_state[int(s)] for s in geom_df["sid"].tolist()], dtype=np.int64)
    y = relation[ridx]
    states_keep = {k: v[ridx] for k, v in states.items()}

    X, groups, feature_names, vocabs = build_requested_design(geom_df)
    splits = make_stratified_splits(y, args.train_ratio, args.repeats, args.seed)

    # Diagnostics on first split are enough to expose exact dependency.
    diag = design_diagnostics(X, groups, splits[0][0])

    targets: Dict[str, np.ndarray] = {}
    for cond in ("raw", "noimg", "residual"):
        A = np.asarray(states_keep[f"A_{cond}"][:, li], dtype=np.float64)
        B = np.asarray(states_keep[f"B_{cond}"][:, li], dtype=np.float64)
        L = np.asarray(states_keep[f"last_{cond}"][:, li], dtype=np.float64)
        targets[f"A_{cond}"] = A
        targets[f"B_{cond}"] = B
        targets[f"last_{cond}"] = L
        targets[f"Diff_{cond}"] = A-B

    all_summary, all_components, all_splits = [], [], []
    for target_name, Y in targets.items():
        sr, cr, rr = analyze_target(target_name, Y, X, groups, splits, args)
        all_summary.extend(sr)
        all_components.extend(cr)
        all_splits.extend(rr)

    summary_df = pd.DataFrame(all_summary)
    comp_df = pd.DataFrame(all_components)
    split_df = pd.DataFrame(all_splits)
    summary_df.to_csv(out / "semantic_decomposition_summary.csv", index=False)
    comp_df.to_csv(out / "semantic_component_contributions.csv", index=False)
    split_df.to_csv(out / "semantic_decomposition_splits.csv", index=False)
    geom_df.to_csv(out / "gdino_geometry_used.csv", index=False)
    (out / "bbox_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compact comparison: raw -> noimg -> residual for each slot/group.
    compact_rows = []
    for slot in ("A", "B", "last", "Diff"):
        for g in list(REQUESTED_GROUPS) + ["position_family"]:
            row = {"slot": slot, "group": g}
            for cond in ("raw", "noimg", "residual"):
                z = comp_df[(comp_df.target == f"{slot}_{cond}") & (comp_df.group == g)]
                if len(z):
                    r = z.iloc[0]
                    row[f"{cond}_group_only_r2"] = r.get("group_only_r2_mean", np.nan)
                    row[f"{cond}_unique_r2"] = r.get("drop_one_unique_r2_mean", np.nan)
                    row[f"{cond}_shapley_r2"] = r.get("shapley_r2_mean", np.nan)
                    row[f"{cond}_component_cos"] = r.get("component_cos_mean", np.nan)
                    row[f"{cond}_norm_ratio"] = r.get("component_norm_ratio_mean", np.nan)
            compact_rows.append(row)
    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(out / "raw_noimg_residual_component_comparison.csv", index=False)

    bbox_cons = float(np.mean(geom_df["bbox_relation_consistent"].to_numpy(dtype=bool)))
    print("\n" + "="*126)
    print("SEMANTIC LINEAR DECOMPOSITION")
    print("="*126)
    print(f"usable bbox samples={len(geom_df)} | layer=L{fixed} | train_ratio={args.train_ratio:.2f} | repeats={args.repeats}")
    print(f"bbox relation sign consistency={100*bbox_cons:.2f}%")
    print(f"design shape X={X.shape}; first-split rank={diag['rank']}/{diag['p']} (deficiency={diag['rank_deficiency']})")
    print(f"check relative == abs_A - abs_B: max abs error={diag['max_abs_relative_minus_absA_plus_absB_error']:.3e}")
    print("WARNING: abs_A, abs_B, relative are exactly redundant; interpret their separate credit cautiously.")
    print("         position_family = union(abs_A, abs_B, relative) is safer.\n")

    print("FULL HELD-OUT RECONSTRUCTION")
    for _, r in summary_df.iterrows():
        print(
            f"{r.target:14s} | centered cos={r.centered_cos_mean:+.4f}±{r.centered_cos_std:.4f} "
            f"| R2={r.r2_mean:+.4f}±{r.r2_std:.4f} | relerr={r.relative_error_mean:.4f} "
            f"| additivity err={r.composition_error_mean:.2e}"
        )

    print("\n" + "="*126)
    print("RESIDUAL COMPONENTS: requested 8 groups")
    print("group-only R2 = group by itself; unique = full - without-group; Shapley shares correlated credit by convention")
    print("="*126)
    for slot in ("A", "B", "last", "Diff"):
        target = f"{slot}_residual"
        print(f"\n{target}")
        d = comp_df[(comp_df.target == target) & (comp_df.group_type == "requested")]
        for _, r in d.iterrows():
            print(
                f"  {r.group:12s} | onlyR2={r.group_only_r2_mean:+.4f} "
                f"| uniqueΔR2={r.drop_one_unique_r2_mean:+.4f} "
                f"| ShapleyR2={r.shapley_r2_mean:+.4f} "
                f"| compCos={r.component_cos_mean:+.4f} "
                f"| norm/target={r.component_norm_ratio_mean:.4f}"
            )
        p = comp_df[(comp_df.target == target) & (comp_df.group == "position_family")].iloc[0]
        print(
            f"  {'position_family':12s} | onlyR2={p.group_only_r2_mean:+.4f} "
            f"| uniqueΔR2={p.drop_one_unique_r2_mean:+.4f} "
            f"| compCos={p.component_cos_mean:+.4f} | norm/target={p.component_norm_ratio_mean:.4f}"
        )

    print("\n" + "="*126)
    print("RAW -> NOIMAGE -> RESIDUAL: Shapley R2 comparison")
    print("="*126)
    for slot in ("A", "B", "last", "Diff"):
        print(f"\n{slot}")
        for g in REQUESTED_GROUPS:
            row = compact_df[(compact_df.slot == slot) & (compact_df.group == g)].iloc[0]
            print(
                f"  {g:12s} | raw={row.raw_shapley_r2:+.4f} "
                f"| noimg={row.noimg_shapley_r2:+.4f} | residual={row.residual_shapley_r2:+.4f}"
            )

    payload = {
        "script": "analyze_coco_gdino_semantic_linear_decomposition_v1.py",
        "state_dir": str(state_dir), "bbox_jsonl": str(bbox_path), "fixed_layer": int(fixed),
        "n_usable": int(len(geom_df)), "bbox_sign_consistency": bbox_cons,
        "requested_groups": list(REQUESTED_GROUPS),
        "group_feature_indices": {k: list(v) for k, v in groups.items()},
        "feature_names": feature_names,
        "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        "design_diagnostics": diag,
        "identifiability_warning": "relative = abs_A - abs_B exactly; separate abs_A/abs_B/relative attribution is non-unique",
        "settings": vars(args),
        "summary": summary_df.to_dict(orient="records"),
        "components": comp_df.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:")
    for name in (
        "baseline_layer_scan.csv", "gdino_geometry_used.csv", "bbox_audit.json",
        "semantic_decomposition_summary.csv", "semantic_component_contributions.csv",
        "semantic_decomposition_splits.csv", "raw_noimg_residual_component_comparison.csv", "summary.json",
    ):
        print(f"  {out/name}")


if __name__ == "__main__":
    main()
