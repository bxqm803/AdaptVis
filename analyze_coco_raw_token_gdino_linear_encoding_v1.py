#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear encoding analysis of RAW A / B / generation-boundary hidden states using
GroundingDINO subject/reference boxes produced by:

    extract_coco_groundingdino_subject_reference_bboxes_v2.py

This script DOES NOT rerun the VLM and DOES NOT need COCO instance annotations.
It reuses the raw-question state files produced by the prior prompt-readout run:

    raw__correct__all_layers.npz
    raw__no_image__all_layers.npz

and directly joins them by sid to:

    output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl

Main questions
--------------
1. How much of h_A, h_B, h_last (and Image-NoImage residuals) is linearly
   explained by continuous object geometry?
2. Which geometric factor matters most?
      pair_center = [(cx_A+cx_B)/2, (cy_A+cy_B)/2]
      relative    = [cx_A-cx_B, cy_A-cy_B]
      size_A      = [w_A, h_A]
      size_B      = [w_B, h_B]
3. After controlling subject identity, reference identity and categorical
   relation label, does continuous relative geometry still explain hidden-state
   variation?
4. Does the encoding direction learned from continuous dx/dy align with the
   old categorical relation axes (right-left / below-above)?
5. If the representation is additively decomposable by the fitted encoding
   components, do the component vectors recombine to reconstruct held-out
   hidden-state variation?

Important interpretation
------------------------
This is still a CORRELATIONAL analysis on natural COCO images. GroundingDINO
boxes provide measured geometry, not a causal intervention. Strong dx/dy
encoding is evidence for a continuous geometric representation, but controlled
counterfactual image-position changes are still needed for a causal claim.

GroundingDINO selection
-----------------------
The bbox extractor chooses the highest-score candidate WITHOUT using GT spatial
relation. By default this analysis keeps only samples for which both objects are
found, both selected scores pass --min-score, and neither object is flagged
ambiguous. Use --include-ambiguous only as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
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

GEOM_FEATURES = (
    "pair_cx", "pair_cy",
    "dx", "dy",
    "wA", "hA",
    "wB", "hB",
)
GEOM_GROUPS = ("pair_center", "relative", "size_A", "size_B")
GEOM_GROUP_IDXS = {
    "pair_center": (0, 1),
    "relative": (2, 3),
    "size_A": (4, 5),
    "size_B": (6, 7),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--state-dir", required=True,
        help="Directory containing raw__correct__all_layers.npz and raw__no_image__all_layers.npz",
    )
    p.add_argument(
        "--bbox-jsonl",
        default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl",
        help="GroundingDINO bbox JSONL from extract_coco_groundingdino_subject_reference_bboxes_v2.py",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--fixed-layer", type=int, default=None,
        help="Decoder block index. If omitted, select once from raw h_A-h_B relation ACC.",
    )
    p.add_argument(
        "--layer-select-train-ratio", type=float, default=0.15,
        help="Train fraction used only for choosing L* when --fixed-layer is omitted.",
    )
    p.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Train fraction for encoding models. 0.7 is safer than 0.15 once identity one-hots are included.",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--ridge", type=float, default=1e-3,
        help="Relative ridge strength: lambda = ridge * trace(X'X)/p.",
    )
    p.add_argument("--shuffle-repeats", type=int, default=10)
    p.add_argument(
        "--min-score", type=float, default=0.25,
        help="Minimum selected GroundingDINO score for BOTH subject and reference.",
    )
    p.add_argument(
        "--include-ambiguous", action="store_true",
        help="Include samples flagged ambiguous by the bbox extractor. Default is strict exclusion.",
    )
    p.add_argument(
        "--require-gt-consistent", action="store_true",
        help=(
            "Sensitivity analysis only: require bbox-center sign to agree with the GT relation. "
            "Do NOT use as the primary analysis because it conditions on the target label."
        ),
    )
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
        raise ValueError(f"correct/noimg layer lists differ: {layers_c.tolist()} vs {layers_n.tolist()}")
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

    meta = {}
    try:
        meta = json.loads(str(np.asarray(correct.get("metadata_json", np.array("{}", dtype=object))).item()))
    except Exception:
        meta = {}

    return common, states, relation, subject, reference, image_id, layers_c, layer_to_idx, meta


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
        raise ValueError(f"degenerate normalized bbox: {b.tolist()}")
    score = float(selected.get("score", float("nan")))
    return b, score


def box_stats(b: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ab = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + ab - inter, EPS)


def relation_geometry_consistent(rel: str, dx: float, dy: float) -> bool:
    # Image coordinates: +x is right, +y is down.
    if rel == "left":
        return dx < 0
    if rel == "right":
        return dx > 0
    if rel == "above":
        return dy < 0
    if rel == "below":
        return dy > 0
    return False


def relation_signed_margin(rel: str, dx: float, dy: float) -> float:
    if rel == "left": return -dx
    if rel == "right": return dx
    if rel == "above": return -dy
    if rel == "below": return dy
    return float("nan")


def build_geometry_table(
    sids: np.ndarray,
    relation: np.ndarray,
    subject: np.ndarray,
    reference: np.ndarray,
    image_id: np.ndarray,
    gdino: Mapping[int, Mapping[str, Any]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    for i, sid0 in enumerate(sids.tolist()):
        sid = int(sid0)
        g = gdino.get(sid)
        if g is None:
            audit.append({"sid": sid, "status": "skip", "reason": "bbox_sid_missing"})
            continue
        try:
            if not bool(g.get("both_found", False)):
                raise ValueError("both_found=false")
            so = g.get("subject", {})
            ro = g.get("reference", {})
            if not isinstance(so, Mapping) or not isinstance(ro, Mapping):
                raise ValueError("subject/reference object block missing")

            s_amb = bool(so.get("ambiguous", False))
            r_amb = bool(ro.get("ambiguous", False))
            if (s_amb or r_amb) and not args.include_ambiguous:
                raise ValueError("ambiguous_bbox")

            ba, score_a = parse_selected_box(so)
            bb, score_b = parse_selected_box(ro)
            if not np.isfinite(score_a) or not np.isfinite(score_b):
                raise ValueError("nonfinite_selected_score")
            if score_a < args.min_score or score_b < args.min_score:
                raise ValueError(f"score_below_min:A={score_a:.4f},B={score_b:.4f}")

            cxA, cyA, wA, hA = box_stats(ba)
            cxB, cyB, wB, hB = box_stats(bb)
            pair_cx = 0.5 * (cxA + cxB)
            pair_cy = 0.5 * (cyA + cyB)
            dx = cxA - cxB
            dy = cyA - cyB
            rel = norm_relation(relation[i])
            consistent = relation_geometry_consistent(rel, dx, dy)
            if args.require_gt_consistent and not consistent:
                raise ValueError("bbox_center_sign_disagrees_with_gt")

            gd_s = canonical_phrase(so.get("phrase", ""))
            gd_r = canonical_phrase(ro.get("phrase", ""))
            state_s = canonical_phrase(subject[i])
            state_r = canonical_phrase(reference[i])
            phrase_match = (gd_s == state_s and gd_r == state_r)
            gd_rel = norm_relation(g.get("relation_gt_metadata_only", rel))
            rel_match = (gd_rel == rel)

            rows.append({
                "sid": sid,
                "image_id": str(image_id[i]),
                "subject": state_s,
                "reference": state_r,
                "relation": rel,
                "subject_score": score_a,
                "reference_score": score_b,
                "subject_ambiguous": s_amb,
                "reference_ambiguous": r_amb,
                "phrase_match": phrase_match,
                "relation_metadata_match": rel_match,
                "bbox_relation_consistent": consistent,
                "relation_signed_margin": relation_signed_margin(rel, dx, dy),
                "bbox_iou": box_iou(ba, bb),
                "Ax1": float(ba[0]), "Ay1": float(ba[1]), "Ax2": float(ba[2]), "Ay2": float(ba[3]),
                "Bx1": float(bb[0]), "By1": float(bb[1]), "Bx2": float(bb[2]), "By2": float(bb[3]),
                "cxA": cxA, "cyA": cyA, "wA": wA, "hA": hA,
                "cxB": cxB, "cyB": cyB, "wB": wB, "hB": hB,
                "pair_cx": pair_cx, "pair_cy": pair_cy,
                "dx": dx, "dy": dy,
                "abs_dx": abs(dx), "abs_dy": abs(dy),
                "center_distance": math.sqrt(dx * dx + dy * dy),
            })
        except Exception as exc:
            audit.append({
                "sid": sid,
                "status": "skip",
                "reason": str(exc),
                "subject": str(subject[i]),
                "reference": str(reference[i]),
                "relation": str(relation[i]),
            })

    return pd.DataFrame(rows), audit


def make_stratified_splits(y: Sequence[str], train_ratio: float, repeats: int, seed: int):
    y = np.asarray(y, dtype=object)
    rng = np.random.default_rng(seed)
    splits = []
    for rep in range(repeats):
        tr_parts = []
        for c in RELATIONS:
            idx = np.where(y == c)[0].copy()
            if len(idx) < 2:
                raise ValueError(f"Need >=2 samples for relation {c}; got {len(idx)}")
            rng.shuffle(idx)
            ntr = int(round(train_ratio * len(idx)))
            ntr = max(1, min(len(idx) - 1, ntr))
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
    return np.sum(a * b, axis=1) / np.maximum(an * bn, EPS)


def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def relative_error(pred_c: np.ndarray, target_c: np.ndarray) -> float:
    return float(np.linalg.norm(pred_c - target_c) / max(float(np.linalg.norm(target_c)), EPS))


def global_r2(pred: np.ndarray, target: np.ndarray, train_mean: np.ndarray) -> float:
    sse = float(np.sum((target - pred) ** 2))
    sst = float(np.sum((target - train_mean) ** 2))
    return 1.0 - sse / max(sst, EPS)


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
        pred = np.argmax(normalize_rows(X[te] - center) @ D.T, axis=1)
        gt = np.asarray([RELATIONS.index(str(v)) for v in y[te]], dtype=np.int64)
        vals.append(float(np.mean(pred == gt)))
    return float(np.mean(vals)), float(np.std(vals))


def select_layer(states: Mapping[str, np.ndarray], y: np.ndarray, layers: np.ndarray, args: argparse.Namespace):
    diff = np.asarray(states["A_raw"] - states["B_raw"], dtype=np.float64)
    splits = make_stratified_splits(y, args.layer_select_train_ratio, args.repeats, args.seed + 300)
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


def one_hot(values: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    vals = [str(v) for v in values]
    vocab = sorted(set(vals))
    lookup = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(vals), len(vocab)), dtype=np.float64)
    for i, v in enumerate(vals):
        X[i, lookup[v]] = 1.0
    return X, vocab


def build_controlled_design(df: pd.DataFrame):
    Xg = df[list(GEOM_FEATURES)].to_numpy(dtype=np.float64)
    Xs, svocab = one_hot(df["subject"].tolist())
    Xr, rvocab = one_hot(df["reference"].tolist())
    Xrel, relvocab = one_hot(df["relation"].tolist())

    blocks = []
    groups: Dict[str, List[int]] = {}
    names: List[str] = []
    offset = 0

    def add_group(name: str, arr: np.ndarray, cols: Sequence[str]):
        nonlocal offset
        blocks.append(arr)
        groups[name] = list(range(offset, offset + arr.shape[1]))
        names.extend([f"{name}:{c}" for c in cols])
        offset += arr.shape[1]

    add_group("subject_id", Xs, svocab)
    add_group("reference_id", Xr, rvocab)
    add_group("pair_center", Xg[:, 0:2], ("pair_cx", "pair_cy"))
    add_group("relative", Xg[:, 2:4], ("dx", "dy"))
    add_group("size_A", Xg[:, 4:6], ("wA", "hA"))
    add_group("size_B", Xg[:, 6:8], ("wB", "hB"))
    add_group("relation_label", Xrel, relvocab)
    return np.concatenate(blocks, axis=1), groups, names, {
        "subject_vocab": svocab, "reference_vocab": rvocab, "relation_vocab": relvocab,
    }


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
            "pred": np.repeat(ymu, len(te), axis=0),
            "pred_c": pred_c,
            "target": Yte,
            "target_c": Yte_c,
            "ymu": ymu,
            "W": np.zeros((0, Y.shape[1]), dtype=np.float64),
            "xmu": np.zeros((0,), dtype=np.float64),
            "xsd": np.ones((0,), dtype=np.float64),
            "idx": idx,
        }

    raw_tr = X[tr][:, idx]
    raw_te = X[te][:, idx]
    xmu = raw_tr.mean(axis=0)
    xsd = raw_tr.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xtr = (raw_tr - xmu) / xsd
    Xte = (raw_te - xmu) / xsd

    gram = Xtr.T @ Xtr
    lam = float(ridge) * float(np.trace(gram) / max(len(idx), 1))
    A = gram + lam * np.eye(len(idx), dtype=np.float64)
    W = np.linalg.solve(A, Xtr.T @ Ytr_c)
    pred_c = Xte @ W
    return {
        "pred": ymu + pred_c,
        "pred_c": pred_c,
        "target": Yte,
        "target_c": Yte_c,
        "ymu": ymu,
        "W": W,
        "xmu": xmu,
        "xsd": xsd,
        "idx": idx,
    }


def fit_metrics(f: Mapping[str, Any]) -> Dict[str, float]:
    pred = np.asarray(f["pred"])
    pc = np.asarray(f["pred_c"])
    target = np.asarray(f["target"])
    tc = np.asarray(f["target_c"])
    ymu = np.asarray(f["ymu"])
    return {
        "full_cos": float(np.mean(cosine_rows(pred, target))),
        "centered_cos": float(np.mean(cosine_rows(pc, tc))),
        "r2": global_r2(pred, target, ymu),
        "relative_error": relative_error(pc, tc),
    }


def group_indices(groups: Mapping[str, Sequence[int]], names: Iterable[str]) -> List[int]:
    out: List[int] = []
    for name in names:
        out.extend(groups[name])
    return sorted(out)


def component_from_fit(
    f: Mapping[str, Any],
    X: np.ndarray,
    te: np.ndarray,
    selected_feature_indices: Sequence[int],
) -> np.ndarray:
    # f coefficient rows correspond exactly to f['idx']; find local positions.
    fit_idx = np.asarray(f["idx"], dtype=np.int64)
    local = {int(global_i): local_i for local_i, global_i in enumerate(fit_idx.tolist())}
    chosen_global = [int(i) for i in selected_feature_indices if int(i) in local]
    if not chosen_global:
        return np.zeros_like(f["target_c"])
    chosen_local = np.asarray([local[i] for i in chosen_global], dtype=np.int64)
    raw = X[te][:, fit_idx]
    z = (raw - np.asarray(f["xmu"])) / np.asarray(f["xsd"])
    return z[:, chosen_local] @ np.asarray(f["W"])[chosen_local]


def shuffled_delta(comp: np.ndarray, target_c: np.ndarray, repeats: int, rng: np.random.Generator) -> float:
    matched = float(np.mean(cosine_rows(comp, target_c)))
    sh = []
    for _ in range(repeats):
        sh.append(float(np.mean(cosine_rows(comp[rng.permutation(len(comp))], target_c))))
    return matched - float(np.mean(sh))


def shapley_geometry_r2(Xg: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray, ridge: float):
    subset_r2: Dict[frozenset, float] = {}
    for r in range(len(GEOM_GROUPS) + 1):
        for comb in itertools.combinations(GEOM_GROUPS, r):
            S = frozenset(comb)
            idx: List[int] = []
            for g in S:
                idx.extend(GEOM_GROUP_IDXS[g])
            fit = ridge_fit_predict(Xg, Y, tr, te, ridge, sorted(idx))
            subset_r2[S] = fit_metrics(fit)["r2"]
    phi = {g: 0.0 for g in GEOM_GROUPS}
    m = len(GEOM_GROUPS)
    for g in GEOM_GROUPS:
        others = [x for x in GEOM_GROUPS if x != g]
        for r in range(len(others) + 1):
            for comb in itertools.combinations(others, r):
                S = frozenset(comb)
                weight = math.factorial(len(S)) * math.factorial(m - len(S) - 1) / math.factorial(m)
                phi[g] += weight * (subset_r2[S | {g}] - subset_r2[S])
    return phi, subset_r2


def ms(vals: Sequence[float]) -> Tuple[float, float]:
    return float(np.mean(vals)), float(np.std(vals))


def analyze_geometry_only(
    Xg: np.ndarray,
    Y: np.ndarray,
    splits,
    args: argparse.Namespace,
    target_name: str,
):
    rng = np.random.default_rng(args.seed + 101)
    all_idx = list(range(Xg.shape[1]))
    metrics = []
    comps = defaultdict(list)
    comp_norms = defaultdict(list)
    comp_shuffle = defaultdict(list)
    drop = defaultdict(list)
    shapley = defaultdict(list)
    gains = []
    split_rows = []

    for si, (tr, te) in enumerate(splits):
        full = ridge_fit_predict(Xg, Y, tr, te, args.ridge, all_idx)
        fm = fit_metrics(full)
        metrics.append(fm)
        cur_comps: Dict[str, np.ndarray] = {}
        for g in GEOM_GROUPS:
            idx = list(GEOM_GROUP_IDXS[g])
            c = component_from_fit(full, Xg, te, idx)
            cur_comps[g] = c
            comps[g].append(float(np.mean(cosine_rows(c, full["target_c"]))))
            comp_norms[g].append(float(np.mean(np.linalg.norm(c, axis=1))))
            comp_shuffle[g].append(shuffled_delta(c, full["target_c"], args.shuffle_repeats, rng))
            keep = [i for i in all_idx if i not in idx]
            red = ridge_fit_predict(Xg, Y, tr, te, args.ridge, keep)
            drop[g].append(fm["r2"] - fit_metrics(red)["r2"])
        phi, _ = shapley_geometry_r2(Xg, Y, tr, te, args.ridge)
        for g, v in phi.items():
            shapley[g].append(float(v))

        recon = sum(cur_comps.values())
        recon_cos = float(np.mean(cosine_rows(recon, full["target_c"])))
        best_single = max(float(np.mean(cosine_rows(c, full["target_c"]))) for c in cur_comps.values())
        gains.append(recon_cos - best_single)
        split_rows.append({
            "target": target_name, "split": si, **fm,
            "component_sum_cos": recon_cos,
            "best_single_component_cos": best_single,
            "composition_gain": recon_cos - best_single,
            "algebra_max_abs": float(np.max(np.abs(recon - full["pred_c"]))),
        })

    summary = {"target": target_name}
    for k in ("full_cos", "centered_cos", "r2", "relative_error"):
        m, s = ms([x[k] for x in metrics])
        summary[k + "_mean"] = m; summary[k + "_std"] = s
    gm, gs = ms(gains)
    summary["composition_gain_mean"] = gm; summary["composition_gain_std"] = gs

    component_rows = []
    for g in GEOM_GROUPS:
        cm, cs = ms(comps[g]); nm, ns = ms(comp_norms[g]); sm, ss = ms(comp_shuffle[g])
        dm, ds = ms(drop[g]); pm, ps = ms(shapley[g])
        component_rows.append({
            "target": target_name,
            "group": g,
            "component_cos_mean": cm, "component_cos_std": cs,
            "component_norm_mean": nm, "component_norm_std": ns,
            "matched_minus_shuffled_mean": sm, "matched_minus_shuffled_std": ss,
            "drop_one_delta_r2_mean": dm, "drop_one_delta_r2_std": ds,
            "shapley_r2_mean": pm, "shapley_r2_std": ps,
        })
    return summary, component_rows, split_rows


def analyze_controlled(
    X: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    Y: np.ndarray,
    splits,
    args: argparse.Namespace,
    target_name: str,
):
    all_groups = list(groups.keys())
    all_idx = group_indices(groups, all_groups)
    geometry_groups = ["pair_center", "relative", "size_A", "size_B"]
    identity_groups = ["subject_id", "reference_id"]

    model_specs = {
        "full": all_groups,
        "identity_only": identity_groups,
        "geometry_only": geometry_groups,
        "relation_only": ["relation_label"],
        "identity_plus_relation": identity_groups + ["relation_label"],
        "identity_plus_geometry": identity_groups + geometry_groups,
        "geometry_plus_relation": geometry_groups + ["relation_label"],
        "full_without_relative": [g for g in all_groups if g != "relative"],
        "full_without_relation_label": [g for g in all_groups if g != "relation_label"],
        "full_without_identity": [g for g in all_groups if g not in identity_groups],
        "full_without_geometry": [g for g in all_groups if g not in geometry_groups],
    }

    per_model = defaultdict(list)
    drop_one = defaultdict(list)
    comp_cos = defaultdict(list)
    rng = np.random.default_rng(args.seed + 202)
    comp_shuffle = defaultdict(list)
    split_rows = []

    for si, (tr, te) in enumerate(splits):
        fits = {}
        for model_name, gs in model_specs.items():
            f = ridge_fit_predict(X, Y, tr, te, args.ridge, group_indices(groups, gs))
            fits[model_name] = f
            per_model[model_name].append(fit_metrics(f))

        full = fits["full"]
        full_r2 = fit_metrics(full)["r2"]
        for g in all_groups:
            keep = [x for x in all_groups if x != g]
            red = ridge_fit_predict(X, Y, tr, te, args.ridge, group_indices(groups, keep))
            drop_one[g].append(full_r2 - fit_metrics(red)["r2"])
            c = component_from_fit(full, X, te, groups[g])
            comp_cos[g].append(float(np.mean(cosine_rows(c, full["target_c"]))))
            comp_shuffle[g].append(shuffled_delta(c, full["target_c"], args.shuffle_repeats, rng))

        row = {"target": target_name, "split": si}
        for model_name, f in fits.items():
            met = fit_metrics(f)
            row[f"{model_name}_r2"] = met["r2"]
            row[f"{model_name}_cos"] = met["centered_cos"]
        split_rows.append(row)

    model_rows = []
    for model_name in model_specs:
        r2m, r2s = ms([x["r2"] for x in per_model[model_name]])
        cm, cs = ms([x["centered_cos"] for x in per_model[model_name]])
        model_rows.append({
            "target": target_name, "model": model_name,
            "r2_mean": r2m, "r2_std": r2s,
            "centered_cos_mean": cm, "centered_cos_std": cs,
        })

    group_rows = []
    for g in all_groups:
        dm, ds = ms(drop_one[g]); cm, cs = ms(comp_cos[g]); sm, ss = ms(comp_shuffle[g])
        group_rows.append({
            "target": target_name, "group": g,
            "drop_one_delta_r2_mean": dm, "drop_one_delta_r2_std": ds,
            "component_cos_mean": cm, "component_cos_std": cs,
            "matched_minus_shuffled_mean": sm, "matched_minus_shuffled_std": ss,
        })

    # Explicit nested-model effects that answer the continuous-vs-categorical question.
    nested = []
    for si in range(len(splits)):
        vals = {m: per_model[m][si] for m in model_specs}
        nested.append({
            "target": target_name,
            "split": si,
            "relative_unique_in_full_r2": vals["full"]["r2"] - vals["full_without_relative"]["r2"],
            "relation_label_unique_in_full_r2": vals["full"]["r2"] - vals["full_without_relation_label"]["r2"],
            "all_geometry_after_identity_plus_relation_r2": vals["full"]["r2"] - vals["identity_plus_relation"]["r2"],
            "relation_after_identity_plus_geometry_r2": vals["full"]["r2"] - vals["identity_plus_geometry"]["r2"],
            "identity_after_geometry_plus_relation_r2": vals["full"]["r2"] - vals["geometry_plus_relation"]["r2"],
        })
    return model_rows, group_rows, split_rows, nested


def axis_alignment(
    Xg: np.ndarray,
    Y: np.ndarray,
    y: np.ndarray,
    splits,
    args: argparse.Namespace,
    target_name: str,
):
    rows = []
    betas_dx = []
    betas_dy = []
    for si, (tr, te) in enumerate(splits):
        fit = ridge_fit_predict(Xg, Y, tr, te, args.ridge, list(range(Xg.shape[1])))
        # Full geometry feature order is fixed; W rows correspond GEOM_FEATURES order.
        W = np.asarray(fit["W"])
        beta_dx = W[GEOM_FEATURES.index("dx")]
        beta_dy = W[GEOM_FEATURES.index("dy")]
        betas_dx.append(beta_dx / max(float(np.linalg.norm(beta_dx)), EPS))
        betas_dy.append(beta_dy / max(float(np.linalg.norm(beta_dy)), EPS))

        # Categorical axes on training hidden states. Image y grows downward, so
        # +dy should align with BELOW - ABOVE.
        if not all(np.any(y[tr] == c) for c in RELATIONS):
            continue
        d_rl = Y[tr][y[tr] == "right"].mean(axis=0) - Y[tr][y[tr] == "left"].mean(axis=0)
        d_ba = Y[tr][y[tr] == "below"].mean(axis=0) - Y[tr][y[tr] == "above"].mean(axis=0)
        rows.append({
            "target": target_name,
            "split": si,
            "cos_beta_dx_vs_right_minus_left": cosine_vec(beta_dx, d_rl),
            "cos_beta_dy_vs_below_minus_above": cosine_vec(beta_dy, d_ba),
            "cross_cos_beta_dx_vs_vertical_axis": cosine_vec(beta_dx, d_ba),
            "cross_cos_beta_dy_vs_horizontal_axis": cosine_vec(beta_dy, d_rl),
            "cos_beta_dx_vs_beta_dy": cosine_vec(beta_dx, beta_dy),
        })

    def mean_pairwise_cos(vecs: List[np.ndarray]) -> float:
        if len(vecs) < 2:
            return float("nan")
        vals = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                vals.append(cosine_vec(vecs[i], vecs[j]))
        return float(np.mean(vals))

    stability = {
        "target": target_name,
        "beta_dx_split_stability": mean_pairwise_cos(betas_dx),
        "beta_dy_split_stability": mean_pairwise_cos(betas_dy),
    }
    return rows, stability


def summarize_relation_geometry(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n"] = int(len(df))
    out["consistency_rate"] = float(df["bbox_relation_consistent"].mean()) if len(df) else float("nan")
    out["mean_subject_score"] = float(df["subject_score"].mean()) if len(df) else float("nan")
    out["mean_reference_score"] = float(df["reference_score"].mean()) if len(df) else float("nan")
    out["mean_bbox_iou"] = float(df["bbox_iou"].mean()) if len(df) else float("nan")
    out["by_relation"] = {}
    for rel in RELATIONS:
        d = df[df["relation"] == rel]
        out["by_relation"][rel] = {
            "n": int(len(d)),
            "consistency_rate": float(d["bbox_relation_consistent"].mean()) if len(d) else float("nan"),
            "mean_signed_margin": float(d["relation_signed_margin"].mean()) if len(d) else float("nan"),
            "mean_dx": float(d["dx"].mean()) if len(d) else float("nan"),
            "mean_dy": float(d["dy"].mean()) if len(d) else float("nan"),
        }
    return out


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if not (0.0 < args.layer_select_train_ratio < 1.0):
        raise ValueError("--layer-select-train-ratio must be in (0,1)")
    if args.repeats < 1 or args.shuffle_repeats < 1:
        raise ValueError("--repeats and --shuffle-repeats must be >=1")

    state_dir = Path(args.state_dir)
    correct_path = state_dir / "raw__correct__all_layers.npz"
    noimg_path = state_dir / "raw__no_image__all_layers.npz"
    bbox_path = Path(args.bbox_jsonl)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    correct = load_npz(correct_path)
    noimg = load_npz(noimg_path)
    sids, states, relation, subject, reference, image_id, layers, layer_to_idx, state_meta = align_states(correct, noimg)

    valid_relation = np.asarray([r in RELATIONS for r in relation], dtype=bool)
    if not np.all(valid_relation):
        sids = sids[valid_relation]
        relation = relation[valid_relation]
        subject = subject[valid_relation]
        reference = reference[valid_relation]
        image_id = image_id[valid_relation]
        for k in list(states.keys()):
            states[k] = states[k][valid_relation]

    gdino = load_gdino_rows(bbox_path)
    geom_df, bbox_audit = build_geometry_table(sids, relation, subject, reference, image_id, gdino, args)
    if len(geom_df) < 40:
        raise RuntimeError(
            f"Only {len(geom_df)} usable GroundingDINO pairs after filtering. "
            f"See {out/'bbox_audit.json'}; try lowering --min-score or inspect ambiguity/missing rates."
        )

    # Align state rows to retained bbox sids.
    sid_to_state = {int(s): i for i, s in enumerate(sids.tolist())}
    ridx = np.asarray([sid_to_state[int(s)] for s in geom_df["sid"].tolist()], dtype=np.int64)
    y = relation[ridx]
    subj = subject[ridx]
    ref = reference[ridx]
    # Overwrite with exactly aligned canonical state values.
    geom_df["subject"] = subj
    geom_df["reference"] = ref
    geom_df["relation"] = y

    # Choose L* using only retained bbox cohort so downstream analyses share samples.
    states_keep = {k: v[ridx] for k, v in states.items()}
    fixed, scan_df = select_layer(states_keep, y, layers, args)
    li = layer_to_idx[fixed]
    scan_df.to_csv(out / "baseline_layer_scan.csv", index=False)

    bbox_stats = summarize_relation_geometry(geom_df)
    bbox_audit_payload = {
        "bbox_jsonl": str(bbox_path),
        "n_state_common": int(len(sids)),
        "n_bbox_jsonl": int(len(gdino)),
        "n_usable": int(len(geom_df)),
        "n_skipped": int(len(bbox_audit)),
        "skip_reasons": dict(Counter(x.get("reason", "unknown") for x in bbox_audit)),
        "first_100_skips": bbox_audit[:100],
        "filter": {
            "min_score": args.min_score,
            "include_ambiguous": args.include_ambiguous,
            "require_gt_consistent": args.require_gt_consistent,
        },
        "geometry_diagnostics": bbox_stats,
    }
    (out / "bbox_audit.json").write_text(json.dumps(bbox_audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    geom_df.to_csv(out / "gdino_geometry.csv", index=False)

    print("\n" + "=" * 118)
    print("GROUNDINGDINO GEOMETRY COHORT")
    print("=" * 118)
    print(f"state sids={len(sids)} | bbox jsonl={len(gdino)} | usable={len(geom_df)} | skipped={len(bbox_audit)}")
    print(f"strict ambiguity filter={not args.include_ambiguous} | min score={args.min_score:.3f}")
    print(f"bbox-center vs GT relation consistency={100*bbox_stats['consistency_rate']:.2f}%")
    print(f"mean detector scores: A={bbox_stats['mean_subject_score']:.3f}, B={bbox_stats['mean_reference_score']:.3f}")
    for rel in RELATIONS:
        z = bbox_stats["by_relation"][rel]
        print(f"  {rel:5s}: n={z['n']:3d} | sign-consistency={100*z['consistency_rate']:.1f}% | mean signed margin={z['mean_signed_margin']:+.4f}")

    best = scan_df.loc[scan_df["acc_mean"].idxmax()]
    selected = scan_df[scan_df["layer"] == fixed].iloc[0]
    print("\n" + "=" * 118)
    print("FIXED LAYER")
    print("=" * 118)
    print(f"raw h_A-h_B best : L{int(best.layer)} acc={100*float(best.acc_mean):.2f}%±{100*float(best.acc_std):.2f}%")
    print(f"analysis layer    : L{fixed} acc={100*float(selected.acc_mean):.2f}%±{100*float(selected.acc_std):.2f}%")

    Xg = geom_df[list(GEOM_FEATURES)].to_numpy(dtype=np.float64)
    Xc, controlled_groups, controlled_feature_names, vocabs = build_controlled_design(geom_df)
    if args.train_ratio < 0.5 and (len(vocabs["subject_vocab"]) + len(vocabs["reference_vocab"]) > 40):
        print("[warning] --train-ratio < 0.5 with many identity one-hot columns; controlled identity estimates may be unstable.")

    splits = make_stratified_splits(y, args.train_ratio, args.repeats, args.seed)
    targets: Dict[str, np.ndarray] = {}
    for slot in SLOTS:
        targets[f"{slot}_raw"] = np.asarray(states_keep[f"{slot}_raw"][:, li], dtype=np.float64)
        targets[f"{slot}_residual"] = np.asarray(states_keep[f"{slot}_residual"][:, li], dtype=np.float64)

    geom_summaries = []
    geom_components = []
    geom_splits = []
    controlled_models = []
    controlled_groups_rows = []
    controlled_splits = []
    controlled_nested = []
    axis_rows = []
    axis_stability = []

    for target_name, Y in targets.items():
        s, c, sp = analyze_geometry_only(Xg, Y, splits, args, target_name)
        geom_summaries.append(s); geom_components.extend(c); geom_splits.extend(sp)

        mr, gr, sr, nr = analyze_controlled(Xc, controlled_groups, Y, splits, args, target_name)
        controlled_models.extend(mr); controlled_groups_rows.extend(gr); controlled_splits.extend(sr); controlled_nested.extend(nr)

        ar, stab = axis_alignment(Xg, Y, y, splits, args, target_name)
        axis_rows.extend(ar); axis_stability.append(stab)

    geom_summary_df = pd.DataFrame(geom_summaries)
    geom_comp_df = pd.DataFrame(geom_components)
    geom_split_df = pd.DataFrame(geom_splits)
    controlled_model_df = pd.DataFrame(controlled_models)
    controlled_group_df = pd.DataFrame(controlled_groups_rows)
    controlled_split_df = pd.DataFrame(controlled_splits)
    controlled_nested_df = pd.DataFrame(controlled_nested)
    axis_df = pd.DataFrame(axis_rows)
    axis_stab_df = pd.DataFrame(axis_stability)

    geom_summary_df.to_csv(out / "geometry_encoding_summary.csv", index=False)
    geom_comp_df.to_csv(out / "geometry_encoding_components.csv", index=False)
    geom_split_df.to_csv(out / "geometry_encoding_splits.csv", index=False)
    controlled_model_df.to_csv(out / "controlled_model_comparison.csv", index=False)
    controlled_group_df.to_csv(out / "controlled_group_contributions.csv", index=False)
    controlled_split_df.to_csv(out / "controlled_model_splits.csv", index=False)
    controlled_nested_df.to_csv(out / "controlled_nested_effects.csv", index=False)
    axis_df.to_csv(out / "continuous_axis_alignment.csv", index=False)
    axis_stab_df.to_csv(out / "continuous_axis_stability.csv", index=False)

    print("\n" + "=" * 118)
    print(f"GEOMETRY-ONLY LINEAR ENCODING @ L{fixed}")
    print("basis = pair_center + relative(dx,dy) + size_A + size_B; all metrics held out")
    print("=" * 118)
    for row in geom_summaries:
        print(
            f"{row['target']:14s} | cos={row['centered_cos_mean']:+.4f}±{row['centered_cos_std']:.4f} "
            f"| R2={row['r2_mean']:+.4f}±{row['r2_std']:.4f} "
            f"| relerr={row['relative_error_mean']:.4f} "
            f"| composition_gain={row['composition_gain_mean']:+.4f}"
        )

    print("\n" + "=" * 118)
    print("GEOMETRY COMPONENTS (Image-NoImage targets)")
    print("Shapley R2 is exact over the four continuous geometry groups")
    print("=" * 118)
    for slot in SLOTS:
        target = f"{slot}_residual"
        print(f"\n{target}")
        d = geom_comp_df[geom_comp_df["target"] == target]
        for _, r in d.iterrows():
            print(
                f"  {r['group']:12s} | cos={r['component_cos_mean']:+.4f} "
                f"| match-shuf={r['matched_minus_shuffled_mean']:+.4f} "
                f"| drop-one ΔR2={r['drop_one_delta_r2_mean']:+.4f} "
                f"| ShapleyR2={r['shapley_r2_mean']:+.4f}"
            )

    print("\n" + "=" * 118)
    print("CONTROLLED: CONTINUOUS GEOMETRY vs CATEGORICAL RELATION")
    print("full = subject_id + reference_id + center + dxdy + sizes + relation_label")
    print("Positive unique R2 means that factor adds held-out explanatory power after the others.")
    print("=" * 118)
    for slot in SLOTS:
        target = f"{slot}_residual"
        d = controlled_nested_df[controlled_nested_df["target"] == target]
        vals = {
            k: (float(d[k].mean()), float(d[k].std()))
            for k in (
                "relative_unique_in_full_r2",
                "relation_label_unique_in_full_r2",
                "all_geometry_after_identity_plus_relation_r2",
                "relation_after_identity_plus_geometry_r2",
            )
        }
        print(
            f"{target:14s} | relative unique={vals['relative_unique_in_full_r2'][0]:+.4f} "
            f"| label unique={vals['relation_label_unique_in_full_r2'][0]:+.4f} "
            f"| all geometry after id+label={vals['all_geometry_after_identity_plus_relation_r2'][0]:+.4f} "
            f"| label after id+geometry={vals['relation_after_identity_plus_geometry_r2'][0]:+.4f}"
        )

    print("\n" + "=" * 118)
    print("CONTINUOUS GEOMETRIC AXIS vs OLD CATEGORICAL AXIS")
    print("dx = cx_A-cx_B, so +dx corresponds right; +dy corresponds below in image coordinates")
    print("=" * 118)
    if len(axis_df):
        for slot in SLOTS:
            target = f"{slot}_residual"
            d = axis_df[axis_df["target"] == target]
            st = axis_stab_df[axis_stab_df["target"] == target].iloc[0]
            print(
                f"{target:14s} | cos(beta_dx, right-left)={d['cos_beta_dx_vs_right_minus_left'].mean():+.4f} "
                f"| cos(beta_dy, below-above)={d['cos_beta_dy_vs_below_minus_above'].mean():+.4f} "
                f"| beta dx/dy cos={d['cos_beta_dx_vs_beta_dy'].mean():+.4f} "
                f"| split stability dx={float(st.beta_dx_split_stability):+.4f}, dy={float(st.beta_dy_split_stability):+.4f}"
            )

    summary = {
        "script": "analyze_coco_raw_token_gdino_linear_encoding_v1.py",
        "state_dir": str(state_dir),
        "bbox_jsonl": str(bbox_path),
        "model_metadata": state_meta,
        "fixed_layer": int(fixed),
        "n_state_common": int(len(sids)),
        "n_usable_bbox": int(len(geom_df)),
        "bbox_diagnostics": bbox_stats,
        "filters": {
            "min_score": args.min_score,
            "include_ambiguous": args.include_ambiguous,
            "require_gt_consistent": args.require_gt_consistent,
        },
        "encoding": {
            "train_ratio": args.train_ratio,
            "repeats": args.repeats,
            "ridge": args.ridge,
            "geometry_features": list(GEOM_FEATURES),
            "geometry_groups": {k: list(v) for k, v in GEOM_GROUP_IDXS.items()},
            "controlled_groups": {k: list(v) for k, v in controlled_groups.items()},
            "controlled_feature_names": controlled_feature_names,
            "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        },
        "geometry_summary": geom_summary_df.to_dict(orient="records"),
        "geometry_components": geom_comp_df.to_dict(orient="records"),
        "controlled_models": controlled_model_df.to_dict(orient="records"),
        "controlled_nested_effects": controlled_nested_df.groupby("target").mean(numeric_only=True).reset_index().to_dict(orient="records"),
        "axis_alignment_mean": axis_df.groupby("target").mean(numeric_only=True).reset_index().to_dict(orient="records") if len(axis_df) else [],
        "axis_stability": axis_stab_df.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:")
    for name in (
        "baseline_layer_scan.csv",
        "gdino_geometry.csv",
        "bbox_audit.json",
        "geometry_encoding_summary.csv",
        "geometry_encoding_components.csv",
        "geometry_encoding_splits.csv",
        "controlled_model_comparison.csv",
        "controlled_group_contributions.csv",
        "controlled_model_splits.csv",
        "controlled_nested_effects.csv",
        "continuous_axis_alignment.csv",
        "continuous_axis_stability.csv",
        "summary.json",
    ):
        print(f"  {out / name}")


if __name__ == "__main__":
    main()
