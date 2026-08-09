#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared + unique analysis for COCO-two semantic spatial representations.

Main question
-------------
For a target hidden representation Y (default: Diff_residual), compare

    Identity family I = subject_id + reference_id
    Position family P = pair_center(x,y) + relative(dx,dy)

without assuming that I and P occupy orthogonal hidden subspaces.

The script reports three complementary analyses:

1) HELD-OUT VARIANCE COMMONALITY
   Fit I-only, P-only, and I+P linear encoding models on train only:

       R2_I, R2_P, R2_joint

   and compute

       unique_I = R2_joint - R2_P
       unique_P = R2_joint - R2_I
       shared   = R2_I + R2_P - R2_joint

   "shared" here means shared explanatory credit, not a literal vector-space
   intersection. It can be slightly negative under finite-sample/suppression effects.

2) PRINCIPAL ANGLES BETWEEN LEARNED HIDDEN SUBSPACES
   On each train split:
     - fit I -> centered Y, SVD the fitted identity contribution -> U_I [D,kI]
     - fit P -> centered Y, SVD the fitted position contribution -> U_P [D,kP]
     - SVD(U_I^T U_P)
   Singular values sigma_j = cos(theta_j) quantify overlap of the two learned
   hidden subspaces. No test hidden state is used to learn these bases.

3) APPROXIMATE SHARED / REMAINDER COMPONENTS
   For thresholds such as 0.5,0.7,0.8,0.9, principal pairs with
   sigma >= threshold are treated as approximate shared directions:

       u_shared,j = normalize(u_I,j + u_P,j)

   We then evaluate:
     - shared projection
     - identity candidate from non-shared principal directions
     - position candidate from non-shared principal directions
   on held-out samples.

   Because finite-data subspaces need not decompose into an exact orthogonal
   direct sum, the non-shared identity/position candidates may still overlap
   when their remaining principal cosines are below but near the threshold.
   The script explicitly reports that residual overlap.

Default target
--------------
    Diff_residual =
      (h_A^img - h_A^noimg) - (h_B^img - h_B^noimg)

Expected state files
--------------------
<state-dir>/
  raw__correct__all_layers.npz
  raw__no_image__all_layers.npz

Expected GroundingDINO file
---------------------------
output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_ALIASES = {
    "left": "left", "right": "right", "above": "above", "below": "below",
    "on": "above", "top": "above", "under": "below", "underneath": "below",
    "bottom": "below",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--state-dir", required=True)
    p.add_argument(
        "--bbox-jsonl",
        default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fixed-layer", type=int, default=25)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="lambda = ridge * trace(X'X)/p",
    )
    p.add_argument(
        "--targets",
        default="Diff_residual",
        help=(
            "Comma-separated targets among "
            "A_raw,A_noimg,A_residual,B_raw,B_noimg,B_residual,"
            "last_raw,last_noimg,last_residual,"
            "Diff_raw,Diff_noimg,Diff_residual"
        ),
    )
    p.add_argument(
        "--thresholds",
        default="0.5,0.7,0.8,0.9",
        help="Principal-cosine thresholds used to define approximate shared directions.",
    )
    p.add_argument(
        "--subspace-energy",
        type=float,
        default=0.999,
        help="Fraction of fitted-contribution SVD energy retained in U_I/U_P.",
    )
    p.add_argument(
        "--max-subspace-rank",
        type=int,
        default=128,
        help="Upper bound on learned identity/position subspace rank.",
    )
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument(
        "--require-gt-consistent",
        action="store_true",
        help="Sensitivity analysis only; filters bbox pairs using GT relation.",
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
        "sample_index", "relation", "subject", "reference",
        "decoder_block_index", "A_vectors", "B_vectors", "last_vectors",
    }
    for name, obj in (("correct", correct), ("noimg", noimg)):
        missing = sorted(required - set(obj.keys()))
        if missing:
            raise KeyError(f"{name} state NPZ missing keys: {missing}")

    cids = np.asarray(correct["sample_index"], dtype=np.int64)
    nids = np.asarray(noimg["sample_index"], dtype=np.int64)
    cm = {int(s): i for i, s in enumerate(cids.tolist())}
    nm = {int(s): i for i, s in enumerate(nids.tolist())}
    common = np.asarray(
        [int(s) for s in cids.tolist() if int(s) in nm], dtype=np.int64
    )
    ci = np.asarray([cm[int(s)] for s in common], dtype=np.int64)
    ni = np.asarray([nm[int(s)] for s in common], dtype=np.int64)

    layers_c = np.asarray(correct["decoder_block_index"], dtype=np.int64)
    layers_n = np.asarray(noimg["decoder_block_index"], dtype=np.int64)
    if not np.array_equal(layers_c, layers_n):
        raise ValueError("correct/noimg layer lists differ")
    layer_to_idx = {int(v): i for i, v in enumerate(layers_c.tolist())}

    states: Dict[str, np.ndarray] = {}
    for slot, key in (
        ("A", "A_vectors"),
        ("B", "B_vectors"),
        ("last", "last_vectors"),
    ):
        c = np.asarray(correct[key][ci], dtype=np.float32)
        n = np.asarray(noimg[key][ni], dtype=np.float32)
        states[f"{slot}_raw"] = c
        states[f"{slot}_noimg"] = n
        states[f"{slot}_residual"] = c - n

    relation = np.asarray(
        [norm_relation(x) for x in np.asarray(correct["relation"], dtype=object)[ci]],
        dtype=object,
    )
    subject = np.asarray(
        [canonical_phrase(x) for x in np.asarray(correct["subject"], dtype=object)[ci]],
        dtype=object,
    )
    reference = np.asarray(
        [canonical_phrase(x) for x in np.asarray(correct["reference"], dtype=object)[ci]],
        dtype=object,
    )
    image_id = np.asarray(
        correct.get("image_id", np.asarray([""] * len(cids), dtype=object)),
        dtype=object,
    )[ci]

    return (
        common, states, relation, subject, reference,
        image_id, layers_c, layer_to_idx
    )


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
                raise ValueError(
                    f"Bad bbox JSONL at {path}:{line_no}: {exc}"
                ) from exc
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
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1


def relation_geometry_consistent(rel: str, dx: float, dy: float) -> bool:
    if rel == "left":
        return dx < 0
    if rel == "right":
        return dx > 0
    if rel == "above":
        return dy < 0
    if rel == "below":
        return dy > 0
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
            so, ro = g.get("subject", {}), g.get("reference", {})
            if not isinstance(so, Mapping) or not isinstance(ro, Mapping):
                raise ValueError("subject/reference block missing")

            s_amb = bool(so.get("ambiguous", False))
            r_amb = bool(ro.get("ambiguous", False))
            if (s_amb or r_amb) and not args.include_ambiguous:
                raise ValueError("ambiguous_bbox")

            ba, score_a = parse_selected_box(so)
            bb, score_b = parse_selected_box(ro)
            if not np.isfinite(score_a) or not np.isfinite(score_b):
                raise ValueError("nonfinite score")
            if score_a < args.min_score or score_b < args.min_score:
                raise ValueError(
                    f"score_below_min:A={score_a:.4f},B={score_b:.4f}"
                )

            cxA, cyA, wA, hA = box_stats(ba)
            cxB, cyB, wB, hB = box_stats(bb)
            dx, dy = cxA - cxB, cyA - cyB
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
                "bbox_relation_consistent": consistent,
                "cxA": cxA, "cyA": cyA, "wA": wA, "hA": hA,
                "cxB": cxB, "cyB": cyB, "wB": wB, "hB": hB,
                "dx": dx, "dy": dy,
                "pair_cx": 0.5 * (cxA + cxB),
                "pair_cy": 0.5 * (cyA + cyB),
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


def make_stratified_splits(
    y: Sequence[str], train_ratio: float, repeats: int, seed: int
):
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b)
        / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS)
    )


def ridge_map_fit(Xtr: np.ndarray, Ytr: np.ndarray, ridge: float):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)

    xmu = Xtr.mean(axis=0)
    xsd = Xtr.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xz = (Xtr - xmu) / xsd

    ymu = Ytr.mean(axis=0, keepdims=True)
    Yc = Ytr - ymu

    p = Xz.shape[1]
    gram = Xz.T @ Xz
    lam = float(ridge) * float(np.trace(gram) / max(p, 1))
    A = gram + lam * np.eye(p, dtype=np.float64)

    try:
        W = np.linalg.solve(A, Xz.T @ Yc)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ Xz.T @ Yc

    Ctr = Xz @ W
    return {
        "xmu": xmu,
        "xsd": xsd,
        "ymu": ymu,
        "W": W,
        "Ctr": Ctr,
    }


def ridge_map_predict(model: Mapping[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xz = (
        np.asarray(X, dtype=np.float64) - model["xmu"]
    ) / model["xsd"]
    return Xz @ model["W"]


def heldout_r2(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    Ytr: np.ndarray,
    Yte: np.ndarray,
    ridge: float,
) -> float:
    model = ridge_map_fit(Xtr, Ytr, ridge)
    pred = model["ymu"] + ridge_map_predict(model, Xte)
    sse = float(np.sum((Yte - pred) ** 2))
    sst = float(np.sum((Yte - model["ymu"]) ** 2))
    return 1.0 - sse / max(sst, EPS)


def spatial_axes(Ytr: np.ndarray, ytr: np.ndarray):
    def mean(c: str) -> np.ndarray:
        return Ytr[ytr == c].mean(axis=0)

    dx = mean("right") - mean("left")
    dy = mean("below") - mean("above")
    dx = dx / max(float(np.linalg.norm(dx)), EPS)
    dy = dy / max(float(np.linalg.norm(dy)), EPS)
    return dx, dy


def centroid_predict(
    Ytr: np.ndarray, ytr: np.ndarray, Yte: np.ndarray
) -> np.ndarray:
    center = Ytr.mean(axis=0, keepdims=True)
    dirs = []
    for c in RELATIONS:
        v = Ytr[ytr == c].mean(axis=0) - center[0]
        dirs.append(v / max(float(np.linalg.norm(v)), EPS))
    D = np.stack(dirs, axis=0)
    scores = normalize_rows(Yte - center) @ D.T
    pred = np.argmax(scores, axis=1)
    return np.asarray([RELATIONS[i] for i in pred], dtype=object)


def spatial_acc_metrics(
    Ytr: np.ndarray, ytr: np.ndarray, Yte: np.ndarray, yte: np.ndarray
) -> Dict[str, float]:
    if Ytr.shape[1] == 0:
        return {
            "spatial_acc4": float("nan"),
            "horizontal_acc": float("nan"),
            "vertical_acc": float("nan"),
        }

    pred = centroid_predict(Ytr, ytr, Yte)
    acc = float(np.mean(pred == yte))
    hmask = np.isin(yte, ["left", "right"])
    vmask = np.isin(yte, ["above", "below"])

    def pair_acc(classes: List[str], mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        center = Ytr.mean(axis=0)
        dirs = []
        for c in classes:
            v = Ytr[ytr == c].mean(axis=0) - center
            dirs.append(v / max(float(np.linalg.norm(v)), EPS))
        D = np.stack(dirs, axis=0)
        sc = normalize_rows(Yte[mask] - center) @ D.T
        pp = np.argmax(sc, axis=1)
        gt = np.asarray(
            [classes.index(str(z)) for z in yte[mask]], dtype=np.int64
        )
        return float(np.mean(pp == gt))

    return {
        "spatial_acc4": acc,
        "horizontal_acc": pair_acc(["left", "right"], hmask),
        "vertical_acc": pair_acc(["above", "below"], vmask),
    }


def parse_thresholds(text: str) -> List[float]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if not (0.0 <= v <= 1.0):
            raise ValueError("thresholds must be in [0,1]")
        vals.append(v)
    if not vals:
        raise ValueError("No thresholds requested")
    return vals


def svd_hidden_basis(
    Ctr: np.ndarray, energy: float, max_rank: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Ctr: [N,D] fitted centered contribution.
    Return U_hidden [D,k], singular values kept, k, numerical_rank.
    """
    Ctr = np.asarray(Ctr, dtype=np.float64)
    if Ctr.size == 0 or np.linalg.norm(Ctr) < EPS:
        return np.zeros((Ctr.shape[1], 0), dtype=np.float64), np.zeros(0), 0, 0

    _, s, Vt = np.linalg.svd(Ctr, full_matrices=False)
    if len(s) == 0 or s[0] <= 0:
        return np.zeros((Ctr.shape[1], 0), dtype=np.float64), s, 0, 0

    tol = max(Ctr.shape) * np.finfo(np.float64).eps * float(s[0])
    numerical_rank = int(np.sum(s > tol))
    numerical_rank = max(numerical_rank, 0)

    e = np.cumsum(s ** 2)
    total = float(e[-1])
    if total <= EPS:
        k_energy = 0
    else:
        k_energy = int(np.searchsorted(e / total, energy, side="left") + 1)

    k = min(numerical_rank, k_energy, int(max_rank))
    if k <= 0:
        return np.zeros((Ctr.shape[1], 0), dtype=np.float64), s, 0, numerical_rank

    Uhid = Vt[:k].T.copy()
    return Uhid, s[:k].copy(), k, numerical_rank


def orthonormalize_columns(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=np.float64)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if len(s) == 0 or s[0] <= EPS:
        return np.zeros((A.shape[0], 0), dtype=np.float64)
    keep = s > max(tol, max(A.shape) * np.finfo(np.float64).eps * float(s[0]))
    return U[:, keep]


def project_rows(Yc: np.ndarray, U: np.ndarray) -> np.ndarray:
    if U.shape[1] == 0:
        return np.zeros_like(Yc)
    return (Yc @ U) @ U.T


def energy_ratio(component: np.ndarray, target_centered: np.ndarray) -> float:
    return float(
        np.linalg.norm(component)
        / max(float(np.linalg.norm(target_centered)), EPS)
    )


def principal_geometry(Ui: np.ndarray, Up: np.ndarray):
    """
    Canonical/principal coordinates for the two learned subspaces.

    Returns
    -------
    sigma : [m]
        Principal cosines, m=min(kI,kP).
    Pi : [D,kI]
        Full orthonormal identity basis rotated into principal coordinates.
        Pi[:,:m] are the paired principal identity directions.
    Pp : [D,kP]
        Full orthonormal position basis rotated into principal coordinates.
        Pp[:,:m] are the paired principal position directions.

    For j < m:
        Pi[:,j]^T Pp[:,j] = sigma[j]
    and cross-pair inner products are zero.
    """
    if Ui.shape[1] == 0 or Up.shape[1] == 0:
        D = Ui.shape[0]
        return (
            np.zeros(0, dtype=np.float64),
            Ui.copy(),
            Up.copy(),
        )

    M = Ui.T @ Up
    # full_matrices=True is important: it preserves unpaired directions when
    # kI != kP, so they can remain in the unique candidate subspaces.
    A, sigma, Bt = np.linalg.svd(M, full_matrices=True)
    sigma = np.clip(sigma, 0.0, 1.0)
    Pi = Ui @ A
    Pp = Up @ Bt.T
    return sigma, Pi, Pp


def approximate_shared_basis(
    sigma: np.ndarray,
    Pi: np.ndarray,
    Pp: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Approximate common directions from high-cosine principal pairs.

    For a selected pair j, use the normalized midpoint
        normalize(Pi_j + Pp_j).
    """
    keep = np.where(sigma >= threshold)[0]
    if len(keep) == 0:
        return np.zeros((Pi.shape[0], 0), dtype=np.float64)

    cols = []
    for j in keep.tolist():
        v = Pi[:, j] + Pp[:, j]
        nv = float(np.linalg.norm(v))
        if nv > EPS:
            cols.append(v / nv)

    if not cols:
        return np.zeros((Pi.shape[0], 0), dtype=np.float64)
    return orthonormalize_columns(np.stack(cols, axis=1))


def approximate_unique_bases(
    sigma: np.ndarray,
    Pi: np.ndarray,
    Pp: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate identity-only / position-only candidate subspaces.

    High-overlap principal pairs (sigma >= threshold) are assigned to SHARED
    and are excluded from both unique candidates.

    Low-overlap paired directions (sigma < threshold) remain in their
    respective identity/position candidate subspaces. Any unpaired directions
    (when kI != kP) also remain unique to their own family.

    This avoids the artifact from subtracting the shared midpoint from both
    members of a high-overlap pair, which would leave an anti-parallel
    difference direction in both remainders.
    """
    m = len(sigma)
    keep_nonshared = [j for j in range(m) if sigma[j] < threshold]

    id_cols = [Pi[:, j] for j in keep_nonshared]
    pos_cols = [Pp[:, j] for j in keep_nonshared]

    # Unpaired principal-coordinate directions are family-specific.
    id_cols.extend(Pi[:, j] for j in range(m, Pi.shape[1]))
    pos_cols.extend(Pp[:, j] for j in range(m, Pp.shape[1]))

    if id_cols:
        Ui_unique = orthonormalize_columns(np.stack(id_cols, axis=1))
    else:
        Ui_unique = np.zeros((Pi.shape[0], 0), dtype=np.float64)

    if pos_cols:
        Up_unique = orthonormalize_columns(np.stack(pos_cols, axis=1))
    else:
        Up_unique = np.zeros((Pp.shape[0], 0), dtype=np.float64)

    return Ui_unique, Up_unique


def max_subspace_overlap(Ua: np.ndarray, Ub: np.ndarray) -> float:
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return float(np.max(s)) if len(s) else 0.0


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    state_dir = Path(args.state_dir)
    correct = load_npz(state_dir / "raw__correct__all_layers.npz")
    noimg = load_npz(state_dir / "raw__no_image__all_layers.npz")

    (
        sids, states, relation, subject, reference,
        image_id, layers, layer_to_idx
    ) = align_states(correct, noimg)

    if args.fixed_layer not in layer_to_idx:
        raise ValueError(
            f"L{args.fixed_layer} unavailable; layers={layers.tolist()}"
        )
    li = layer_to_idx[args.fixed_layer]

    valid_rel = np.asarray([r in RELATIONS for r in relation], dtype=bool)
    sids = sids[valid_rel]
    relation = relation[valid_rel]
    subject = subject[valid_rel]
    reference = reference[valid_rel]
    image_id = image_id[valid_rel]
    for k in list(states.keys()):
        states[k] = states[k][valid_rel]

    gdino = load_gdino_rows(Path(args.bbox_jsonl))
    geom_df, audit = build_geometry_table(
        sids, relation, subject, reference, image_id, gdino, args
    )
    if len(geom_df) < 40:
        raise RuntimeError(f"Too few usable bbox samples: {len(geom_df)}")

    sid_to_state = {int(s): i for i, s in enumerate(sids.tolist())}
    ridx = np.asarray(
        [sid_to_state[int(s)] for s in geom_df["sid"].tolist()],
        dtype=np.int64,
    )
    y = relation[ridx]

    # Fixed-layer targets.
    Araw = np.asarray(states["A_raw"][ridx, li], dtype=np.float64)
    Ano = np.asarray(states["A_noimg"][ridx, li], dtype=np.float64)
    Ares = np.asarray(states["A_residual"][ridx, li], dtype=np.float64)
    Braw = np.asarray(states["B_raw"][ridx, li], dtype=np.float64)
    Bno = np.asarray(states["B_noimg"][ridx, li], dtype=np.float64)
    Bres = np.asarray(states["B_residual"][ridx, li], dtype=np.float64)
    Lraw = np.asarray(states["last_raw"][ridx, li], dtype=np.float64)
    Lno = np.asarray(states["last_noimg"][ridx, li], dtype=np.float64)
    Lres = np.asarray(states["last_residual"][ridx, li], dtype=np.float64)

    targets_all = {
        "A_raw": Araw,
        "A_noimg": Ano,
        "A_residual": Ares,
        "B_raw": Braw,
        "B_noimg": Bno,
        "B_residual": Bres,
        "last_raw": Lraw,
        "last_noimg": Lno,
        "last_residual": Lres,
        "Diff_raw": Araw - Braw,
        "Diff_noimg": Ano - Bno,
        "Diff_residual": Ares - Bres,
    }

    wanted = [x.strip() for x in args.targets.split(",") if x.strip()]
    bad = [x for x in wanted if x not in targets_all]
    if bad:
        raise ValueError(
            f"Unsupported targets {bad}; available={list(targets_all)}"
        )
    targets = {k: targets_all[k] for k in wanted}

    # Identity family.
    Xs, subject_vocab = one_hot(geom_df["subject"].tolist())
    Xr, reference_vocab = one_hot(geom_df["reference"].tolist())
    Xid = np.concatenate([Xs, Xr], axis=1)

    # Nonredundant position family.
    # We intentionally use pair_center + relative rather than
    # abs_A + abs_B + relative, because the latter is exactly redundant.
    Xpos = geom_df[
        ["pair_cx", "pair_cy", "dx", "dy"]
    ].to_numpy(dtype=np.float64)

    Xjoint = np.concatenate([Xid, Xpos], axis=1)

    splits = make_stratified_splits(
        y, args.train_ratio, args.repeats, args.seed
    )
    thresholds = parse_thresholds(args.thresholds)

    common_rows: List[Dict[str, Any]] = []
    angle_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []

    print("=" * 126)
    print("SHARED + UNIQUE: IDENTITY vs POSITION")
    print("=" * 126)
    print(
        f"usable bbox samples={len(geom_df)} | layer=L{args.fixed_layer} | "
        f"repeats={args.repeats} | train_ratio={args.train_ratio:.2f}"
    )
    print(
        "identity = subject_id + reference_id | "
        "position = pair_center(x,y) + relative(dx,dy)"
    )
    print(
        f"identity design shape={Xid.shape} | position design shape={Xpos.shape} | "
        f"hidden D={next(iter(targets.values())).shape[1]}"
    )
    print(
        "bbox relation sign consistency="
        f"{100.0 * float(np.mean(geom_df['bbox_relation_consistent'].to_numpy(dtype=bool))):.2f}%"
    )

    for target_name, Y in targets.items():
        for rep, (tr, te) in enumerate(splits):
            Ytr, Yte = Y[tr], Y[te]
            ytr, yte = y[tr], y[te]

            Xid_tr, Xid_te = Xid[tr], Xid[te]
            Xpos_tr, Xpos_te = Xpos[tr], Xpos[te]
            Xjoint_tr, Xjoint_te = Xjoint[tr], Xjoint[te]

            # 1) Held-out commonality.
            r2_i = heldout_r2(
                Xid_tr, Xid_te, Ytr, Yte, args.ridge
            )
            r2_p = heldout_r2(
                Xpos_tr, Xpos_te, Ytr, Yte, args.ridge
            )
            r2_j = heldout_r2(
                Xjoint_tr, Xjoint_te, Ytr, Yte, args.ridge
            )

            unique_i = r2_j - r2_p
            unique_p = r2_j - r2_i
            shared = r2_i + r2_p - r2_j

            common_rows.append({
                "target": target_name,
                "repeat": rep,
                "identity_only_r2": r2_i,
                "position_only_r2": r2_p,
                "joint_r2": r2_j,
                "identity_unique_r2": unique_i,
                "position_unique_r2": unique_p,
                "shared_commonality_r2": shared,
                "joint_unexplained_fraction": 1.0 - r2_j,
            })

            # 2) Learn hidden subspaces on TRAIN ONLY.
            mid = ridge_map_fit(Xid_tr, Ytr, args.ridge)
            mpos = ridge_map_fit(Xpos_tr, Ytr, args.ridge)

            Ui, si, ki, nri = svd_hidden_basis(
                mid["Ctr"], args.subspace_energy, args.max_subspace_rank
            )
            Up, sp, kp, nrp = svd_hidden_basis(
                mpos["Ctr"], args.subspace_energy, args.max_subspace_rank
            )

            sigma, Pi, Pp = principal_geometry(Ui, Up)

            for j, sig in enumerate(sigma.tolist()):
                angle_rows.append({
                    "target": target_name,
                    "repeat": rep,
                    "principal_index": j + 1,
                    "sigma_cos": float(sig),
                    "angle_deg": float(
                        np.degrees(np.arccos(np.clip(sig, 0.0, 1.0)))
                    ),
                    "identity_rank_kept": ki,
                    "position_rank_kept": kp,
                    "identity_numerical_rank": nri,
                    "position_numerical_rank": nrp,
                })

            # Original centered target and spatial axes for capture diagnostics.
            mu = Ytr.mean(axis=0, keepdims=True)
            Ytrc = Ytr - mu
            Ytec = Yte - mu
            dx_axis, dy_axis = spatial_axes(Ytr, ytr)

            # Baseline row, useful for direct comparison.
            base_acc = spatial_acc_metrics(Ytr, ytr, Yte, yte)
            component_rows.append({
                "target": target_name,
                "repeat": rep,
                "threshold": -1.0,
                "component": "baseline",
                "shared_rank": 0,
                "component_rank": Y.shape[1],
                "energy_ratio": 1.0,
                "spatial_acc4": base_acc["spatial_acc4"],
                "horizontal_acc": base_acc["horizontal_acc"],
                "vertical_acc": base_acc["vertical_acc"],
                "x_axis_capture": 1.0,
                "y_axis_capture": 1.0,
                "identity_position_remainder_max_sigma": float(
                    np.max(sigma) if len(sigma) else 0.0
                ),
            })

            # 3) Thresholded approximate shared/remainders.
            for thr in thresholds:
                Us = approximate_shared_basis(sigma, Pi, Pp, thr)
                Ui_rem, Up_rem = approximate_unique_bases(sigma, Pi, Pp, thr)
                rem_overlap = max_subspace_overlap(Ui_rem, Up_rem)

                blocks = [
                    ("shared", Us),
                    ("identity_unique_candidate", Ui_rem),
                    ("position_unique_candidate", Up_rem),
                ]

                for cname, Uc in blocks:
                    Ctr = project_rows(Ytrc, Uc)
                    Cte = project_rows(Ytec, Uc)

                    if Uc.shape[1] == 0:
                        accs = {
                            "spatial_acc4": float("nan"),
                            "horizontal_acc": float("nan"),
                            "vertical_acc": float("nan"),
                        }
                        e = 0.0
                        xcap = 0.0
                        ycap = 0.0
                    else:
                        accs = spatial_acc_metrics(Ctr, ytr, Cte, yte)
                        e = energy_ratio(Cte, Ytec)
                        xcap = float(np.sum((Uc.T @ dx_axis) ** 2))
                        ycap = float(np.sum((Uc.T @ dy_axis) ** 2))

                    component_rows.append({
                        "target": target_name,
                        "repeat": rep,
                        "threshold": float(thr),
                        "component": cname,
                        "shared_rank": int(Us.shape[1]),
                        "component_rank": int(Uc.shape[1]),
                        "energy_ratio": e,
                        "spatial_acc4": accs["spatial_acc4"],
                        "horizontal_acc": accs["horizontal_acc"],
                        "vertical_acc": accs["vertical_acc"],
                        "x_axis_capture": xcap,
                        "y_axis_capture": ycap,
                        "identity_position_remainder_max_sigma": rem_overlap,
                    })

    common_df = pd.DataFrame(common_rows)
    angle_df = pd.DataFrame(angle_rows)
    component_df = pd.DataFrame(component_rows)

    common_df.to_csv(outdir / "variance_commonality_by_split.csv", index=False)
    angle_df.to_csv(outdir / "principal_angles_by_split.csv", index=False)
    component_df.to_csv(outdir / "shared_unique_components_by_split.csv", index=False)

    with (outdir / "audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "state_dir": str(args.state_dir),
        "bbox_jsonl": str(args.bbox_jsonl),
        "fixed_layer": int(args.fixed_layer),
        "usable_samples": int(len(geom_df)),
        "train_ratio": float(args.train_ratio),
        "repeats": int(args.repeats),
        "ridge": float(args.ridge),
        "targets": wanted,
        "thresholds": thresholds,
        "subspace_energy": float(args.subspace_energy),
        "max_subspace_rank": int(args.max_subspace_rank),
        "identity_feature_dim": int(Xid.shape[1]),
        "position_feature_dim": int(Xpos.shape[1]),
        "subject_vocab_size": int(len(subject_vocab)),
        "reference_vocab_size": int(len(reference_vocab)),
        "bbox_relation_sign_consistency": float(
            np.mean(geom_df["bbox_relation_consistent"].to_numpy(dtype=bool))
        ),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Human-readable summaries
    # -------------------------
    print()
    print("=" * 126)
    print("1) HELD-OUT VARIANCE COMMONALITY")
    print("shared = R2(identity) + R2(position) - R2(joint)")
    print("unique identity = R2(joint)-R2(position); unique position = R2(joint)-R2(identity)")
    print("=" * 126)

    for target_name in wanted:
        d = common_df[common_df["target"] == target_name]
        vals = {}
        for col in (
            "identity_only_r2",
            "position_only_r2",
            "joint_r2",
            "identity_unique_r2",
            "position_unique_r2",
            "shared_commonality_r2",
        ):
            vals[col] = mean_std(d[col].to_numpy(dtype=float))

        print(f"\n{target_name}")
        print(
            f"  identity only R2 = {vals['identity_only_r2'][0]:+.4f}±{vals['identity_only_r2'][1]:.4f}"
        )
        print(
            f"  position only R2 = {vals['position_only_r2'][0]:+.4f}±{vals['position_only_r2'][1]:.4f}"
        )
        print(
            f"  joint R2         = {vals['joint_r2'][0]:+.4f}±{vals['joint_r2'][1]:.4f}"
        )
        print(
            f"  identity unique  = {vals['identity_unique_r2'][0]:+.4f}±{vals['identity_unique_r2'][1]:.4f}"
        )
        print(
            f"  position unique  = {vals['position_unique_r2'][0]:+.4f}±{vals['position_unique_r2'][1]:.4f}"
        )
        print(
            f"  SHARED           = {vals['shared_commonality_r2'][0]:+.4f}±{vals['shared_commonality_r2'][1]:.4f}"
        )

    print()
    print("=" * 126)
    print("2) PRINCIPAL-ANGLE SPECTRUM: learned identity subspace vs learned position subspace")
    print("sigma=1 -> same direction; sigma=0 -> orthogonal")
    print("=" * 126)

    if len(angle_df):
        for target_name in wanted:
            d = angle_df[angle_df["target"] == target_name]
            if len(d) == 0:
                continue
            print(f"\n{target_name}")
            for j in sorted(d["principal_index"].unique().tolist()):
                q = d[d["principal_index"] == j]
                sm, ss = mean_std(q["sigma_cos"].to_numpy(dtype=float))
                am, ast = mean_std(q["angle_deg"].to_numpy(dtype=float))
                print(
                    f"  PC{int(j):02d} | sigma={sm:.4f}±{ss:.4f} | angle={am:.2f}°±{ast:.2f}°"
                )
            ki = d.groupby("repeat")["identity_rank_kept"].first().to_numpy(dtype=float)
            kp = d.groupby("repeat")["position_rank_kept"].first().to_numpy(dtype=float)
            print(
                f"  retained ranks: identity={np.mean(ki):.1f}, position={np.mean(kp):.1f}"
            )
    else:
        print("No non-empty principal-angle spectrum.")

    print()
    print("=" * 126)
    print("3) APPROXIMATE SHARED / UNIQUE-CANDIDATE COMPONENTS")
    print("energy = ||projected held-out component|| / ||centered held-out target||")
    print("x/y capture = fraction of original categorical spatial-axis energy lying in that component subspace")
    print("remainder-overlap = max principal cosine between the non-shared identity/position candidates")
    print("=" * 126)

    for target_name in wanted:
        print(f"\n{target_name}")

        b = component_df[
            (component_df["target"] == target_name)
            & (component_df["component"] == "baseline")
        ]
        if len(b):
            am, ast = mean_std(b["spatial_acc4"].to_numpy(dtype=float))
            hm, hs = mean_std(b["horizontal_acc"].to_numpy(dtype=float))
            vm, vs = mean_std(b["vertical_acc"].to_numpy(dtype=float))
            print(
                f"  baseline | spatial={am:.4f}±{ast:.4f} | H={hm:.4f} | V={vm:.4f}"
            )

        for thr in thresholds:
            print(f"\n  threshold sigma >= {thr:.2f}")
            td = component_df[
                (component_df["target"] == target_name)
                & np.isclose(component_df["threshold"], thr)
            ]
            if len(td) == 0:
                continue

            # Shared rank and remainder overlap are repeated across the 3 component rows.
            shared_rank_by_rep = (
                td.groupby("repeat")["shared_rank"].first().to_numpy(dtype=float)
            )
            overlap_by_rep = (
                td.groupby("repeat")["identity_position_remainder_max_sigma"]
                .first()
                .to_numpy(dtype=float)
            )
            print(
                f"    shared rank={np.mean(shared_rank_by_rep):.2f} | "
                f"remainder-overlap max sigma={np.mean(overlap_by_rep):.3f}"
            )

            for cname in ("shared", "identity_unique_candidate", "position_unique_candidate"):
                q = td[td["component"] == cname]
                if len(q) == 0:
                    continue
                e, _ = mean_std(q["energy_ratio"].to_numpy(dtype=float))
                a, _ = mean_std(q["spatial_acc4"].to_numpy(dtype=float))
                h, _ = mean_std(q["horizontal_acc"].to_numpy(dtype=float))
                v, _ = mean_std(q["vertical_acc"].to_numpy(dtype=float))
                xc, _ = mean_std(q["x_axis_capture"].to_numpy(dtype=float))
                yc, _ = mean_std(q["y_axis_capture"].to_numpy(dtype=float))
                rk = float(np.nanmean(q["component_rank"].to_numpy(dtype=float)))
                print(
                    f"    {cname:19s} | rank={rk:5.1f} | energy={e:.3f} | "
                    f"spatial={a:.4f} | H={h:.4f} | V={v:.4f} | "
                    f"xcap={xc:.3f} | ycap={yc:.3f}"
                )

    print()
    print("=" * 126)
    print("OUTPUT FILES")
    print("=" * 126)
    for name in (
        "variance_commonality_by_split.csv",
        "principal_angles_by_split.csv",
        "shared_unique_components_by_split.csv",
        "summary.json",
        "audit.jsonl",
    ):
        print(outdir / name)


if __name__ == "__main__":
    main()
