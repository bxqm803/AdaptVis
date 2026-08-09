#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO-two coefficient-space decomposition of a hidden representation.

Main decomposition
------------------
For target hidden states H [N,D] (default Diff_residual), each train split learns:

  Identity features I = subject_id + reference_id
  Position features P = pair_center(x,y) + relative(dx,dy)

First learn identity and position hidden subspaces from TRAIN only and obtain
high-overlap principal directions. Their midpoint span defines an orthonormal
shared basis:

    U_shared [D,k]

Center hidden states by the TRAIN mean:

    Hc = H - mean_train(H)

Project into the shared coefficient space:

    Z = Hc @ U_shared                  [N,k]

Then fit one joint ridge model on TRAIN only:

    Z ~= Z_identity + Z_position
      = I_std @ W_identity + P_std @ W_position

For every sample define:

    H_identity = Z_identity @ U_shared.T

    H_position = Z_position @ U_shared.T

    H_shared_residual
        = (Z - Z_identity - Z_position) @ U_shared.T

    H_outside
        = Hc - (Z @ U_shared.T)

Therefore, up to floating-point precision:

    Hc =
        H_identity
      + H_position
      + H_shared_residual
      + H_outside

Important
---------
These are LINEAR MODEL ATTRIBUTIONS, not claims that the model contains four
naturally orthogonal internal modules. In particular H_identity and H_position
reuse the same U_shared and need not be orthogonal.

Held-out evaluation
-------------------
For each component, report:
  - 4-class spatial centroid ACC
  - horizontal / vertical pair ACC
  - identity-family encoding R2
  - position-family encoding R2
  - relative(dx,dy)-only encoding R2
  - subject/reference seen-class centroid accuracy
  - component norm ratio

Also report coefficient-space joint held-out R2 and exact additivity error.

Expected files
--------------
<state-dir>/
  raw__correct__all_layers.npz
  raw__no_image__all_layers.npz

bbox:
  output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_ALIASES = {
    "left": "left", "right": "right",
    "above": "above", "below": "below",
    "on": "above", "top": "above",
    "under": "below", "underneath": "below", "bottom": "below",
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
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
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument(
        "--target",
        default="Diff_residual",
        choices=[
            "A_raw", "A_noimg", "A_residual",
            "B_raw", "B_noimg", "B_residual",
            "last_raw", "last_noimg", "last_residual",
            "Diff_raw", "Diff_noimg", "Diff_residual",
        ],
    )
    p.add_argument("--shared-threshold", type=float, default=0.90)
    p.add_argument("--subspace-energy", type=float, default=0.999)
    p.add_argument("--max-identity-rank", type=int, default=128)
    p.add_argument("--max-position-rank", type=int, default=4)
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument("--require-gt-consistent", action="store_true")
    p.add_argument(
        "--no-save-components",
        action="store_true",
        help="Do not save per-repeat NPZ decompositions.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

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
    req = {
        "sample_index", "relation", "subject", "reference",
        "decoder_block_index", "A_vectors", "B_vectors", "last_vectors",
    }
    for name, obj in (("correct", correct), ("noimg", noimg)):
        missing = sorted(req - set(obj.keys()))
        if missing:
            raise KeyError(f"{name} state NPZ missing keys: {missing}")

    cids = np.asarray(correct["sample_index"], dtype=np.int64)
    nids = np.asarray(noimg["sample_index"], dtype=np.int64)
    cmap = {int(s): i for i, s in enumerate(cids.tolist())}
    nmap = {int(s): i for i, s in enumerate(nids.tolist())}

    common = np.asarray(
        [int(s) for s in cids.tolist() if int(s) in nmap],
        dtype=np.int64,
    )
    ci = np.asarray([cmap[int(s)] for s in common], dtype=np.int64)
    ni = np.asarray([nmap[int(s)] for s in common], dtype=np.int64)

    lc = np.asarray(correct["decoder_block_index"], dtype=np.int64)
    ln = np.asarray(noimg["decoder_block_index"], dtype=np.int64)
    if not np.array_equal(lc, ln):
        raise ValueError("correct/no_image layer lists differ")
    layer_to_idx = {int(v): i for i, v in enumerate(lc.tolist())}

    states = {}
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
        [norm_relation(x)
         for x in np.asarray(correct["relation"], dtype=object)[ci]],
        dtype=object,
    )
    subject = np.asarray(
        [canonical_phrase(x)
         for x in np.asarray(correct["subject"], dtype=object)[ci]],
        dtype=object,
    )
    reference = np.asarray(
        [canonical_phrase(x)
         for x in np.asarray(correct["reference"], dtype=object)[ci]],
        dtype=object,
    )
    image_id = np.asarray(
        correct.get("image_id", np.asarray([""] * len(cids), dtype=object)),
        dtype=object,
    )[ci]

    return (
        common, states, relation, subject, reference,
        image_id, lc, layer_to_idx,
    )


def load_bbox_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                out[int(row["sid"])] = row
            except Exception as exc:
                raise ValueError(
                    f"Bad bbox JSONL {path}:{line_no}: {exc}"
                ) from exc
    return out


def selected_box(obj: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    s = obj.get("selected")
    if not isinstance(s, Mapping):
        raise ValueError("missing selected bbox")
    b = s.get("box_xyxy_normalized")
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError("selected bbox has no normalized xyxy[4]")

    arr = np.asarray([float(v) for v in b], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("bbox contains nonfinite values")
    arr = np.clip(arr, 0.0, 1.0)

    x1, y1, x2, y2 = arr.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate bbox")
    return arr, float(s.get("score", np.nan))


def box_stats(b: np.ndarray):
    x1, y1, x2, y2 = [float(v) for v in b]
    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        x2 - x1,
        y2 - y1,
    )


def relation_consistent(rel: str, dx: float, dy: float) -> bool:
    if rel == "left":
        return dx < 0
    if rel == "right":
        return dx > 0
    if rel == "above":
        return dy < 0
    if rel == "below":
        return dy > 0
    return False


def build_table(
    sids, relation, subject, reference, image_id, bbox_rows, args
):
    rows, audit = [], []

    for i, sid0 in enumerate(sids.tolist()):
        sid = int(sid0)
        g = bbox_rows.get(sid)

        if g is None:
            audit.append({"sid": sid, "reason": "bbox_sid_missing"})
            continue

        try:
            if not bool(g.get("both_found", False)):
                raise ValueError("both_found=false")

            so = g.get("subject", {})
            ro = g.get("reference", {})

            if not args.include_ambiguous:
                if bool(so.get("ambiguous", False)) or bool(ro.get("ambiguous", False)):
                    raise ValueError("ambiguous_bbox")

            ba, score_a = selected_box(so)
            bb, score_b = selected_box(ro)

            if not np.isfinite(score_a) or not np.isfinite(score_b):
                raise ValueError("nonfinite score")
            if score_a < args.min_score or score_b < args.min_score:
                raise ValueError("score_below_min")

            cxA, cyA, wA, hA = box_stats(ba)
            cxB, cyB, wB, hB = box_stats(bb)
            dx, dy = cxA - cxB, cyA - cyB
            rel = norm_relation(relation[i])
            ok = relation_consistent(rel, dx, dy)

            if args.require_gt_consistent and not ok:
                raise ValueError("bbox_relation_inconsistent")

            rows.append({
                "sid": sid,
                "image_id": str(image_id[i]),
                "subject": canonical_phrase(subject[i]),
                "reference": canonical_phrase(reference[i]),
                "relation": rel,
                "pair_cx": 0.5 * (cxA + cxB),
                "pair_cy": 0.5 * (cyA + cyB),
                "dx": dx,
                "dy": dy,
                "wA": wA, "hA": hA,
                "wB": wB, "hB": hB,
                "bbox_relation_consistent": ok,
            })
        except Exception as exc:
            audit.append({"sid": sid, "reason": str(exc)})

    return pd.DataFrame(rows), audit


# ---------------------------------------------------------------------
# Features / splits
# ---------------------------------------------------------------------

def one_hot(values: Sequence[str]):
    vals = [str(x) for x in values]
    vocab = sorted(set(vals))
    lookup = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(vals), len(vocab)), dtype=np.float64)
    for i, v in enumerate(vals):
        X[i, lookup[v]] = 1.0
    return X, vocab


def stratified_splits(y, train_ratio, repeats, seed):
    y = np.asarray(y, dtype=object)
    rng = np.random.default_rng(seed)
    out = []

    for _ in range(repeats):
        parts = []
        for c in RELATIONS:
            idx = np.where(y == c)[0].copy()
            if len(idx) < 2:
                raise RuntimeError(f"too few samples for class={c}")
            rng.shuffle(idx)
            ntr = int(round(train_ratio * len(idx)))
            ntr = max(1, min(len(idx) - 1, ntr))
            parts.append(idx[:ntr])

        tr = np.sort(np.concatenate(parts))
        mask = np.ones(len(y), dtype=bool)
        mask[tr] = False
        te = np.where(mask)[0]
        out.append((tr, te))

    return out


# ---------------------------------------------------------------------
# Ridge helpers
# ---------------------------------------------------------------------

def ridge_lambda(Xz: np.ndarray, ridge: float):
    p = Xz.shape[1]
    G = Xz.T @ Xz
    lam = float(ridge) * float(np.trace(G) / max(p, 1))
    return G, lam


def fit_ridge_matrix(Xtr, Ytr, ridge):
    """
    Standardized-X ridge with intercept through centering Y.
    Y can be [N,D].
    """
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)

    xmu = Xtr.mean(axis=0)
    xsd = Xtr.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xz = (Xtr - xmu) / xsd

    ymu = Ytr.mean(axis=0, keepdims=True)
    Yc = Ytr - ymu

    G, lam = ridge_lambda(Xz, ridge)
    A = G + lam * np.eye(Xz.shape[1], dtype=np.float64)
    try:
        W = np.linalg.solve(A, Xz.T @ Yc)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ Xz.T @ Yc

    return {
        "xmu": xmu,
        "xsd": xsd,
        "ymu": ymu,
        "W": W,
        "Ctr": Xz @ W,
    }


def predict_ridge_matrix(model, X):
    Xz = (
        np.asarray(X, dtype=np.float64) - model["xmu"]
    ) / model["xsd"]
    return model["ymu"] + Xz @ model["W"]


def heldout_vector_r2(Xtr, Xte, Ytr, Yte, ridge):
    m = fit_ridge_matrix(Xtr, Ytr, ridge)
    pred = predict_ridge_matrix(m, Xte)
    sse = float(np.sum((Yte - pred) ** 2))
    sst = float(np.sum((Yte - m["ymu"]) ** 2))
    return 1.0 - sse / max(sst, EPS)


def fit_joint_groups(Xi_tr, Xp_tr, Ztr, ridge):
    """
    Joint coefficient model:
        Z ~= zmu + I_std @ Wi + P_std @ Wp

    Group contributions are kept separate.
    """
    Xi_tr = np.asarray(Xi_tr, dtype=np.float64)
    Xp_tr = np.asarray(Xp_tr, dtype=np.float64)
    Ztr = np.asarray(Ztr, dtype=np.float64)

    imu, isd = Xi_tr.mean(axis=0), Xi_tr.std(axis=0)
    pmu, psd = Xp_tr.mean(axis=0), Xp_tr.std(axis=0)
    isd = np.where(isd < 1e-8, 1.0, isd)
    psd = np.where(psd < 1e-8, 1.0, psd)

    Iz = (Xi_tr - imu) / isd
    Pz = (Xp_tr - pmu) / psd
    X = np.concatenate([Iz, Pz], axis=1)

    zmu = Ztr.mean(axis=0, keepdims=True)
    Zc = Ztr - zmu

    G, lam = ridge_lambda(X, ridge)
    A = G + lam * np.eye(X.shape[1], dtype=np.float64)
    try:
        W = np.linalg.solve(A, X.T @ Zc)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ X.T @ Zc

    pi = Xi_tr.shape[1]
    return {
        "imu": imu, "isd": isd,
        "pmu": pmu, "psd": psd,
        "zmu": zmu,
        "Wi": W[:pi],
        "Wp": W[pi:],
    }


def joint_group_contributions(model, Xi, Xp):
    Iz = (
        np.asarray(Xi, dtype=np.float64) - model["imu"]
    ) / model["isd"]
    Pz = (
        np.asarray(Xp, dtype=np.float64) - model["pmu"]
    ) / model["psd"]

    Zi = Iz @ model["Wi"]
    Zp = Pz @ model["Wp"]
    Zhat = model["zmu"] + Zi + Zp
    return Zi, Zp, Zhat


# ---------------------------------------------------------------------
# Hidden subspaces
# ---------------------------------------------------------------------

def svd_hidden_basis(Ctr, energy, max_rank):
    """
    Ctr [N,D] fitted centered contribution -> hidden basis [D,k].
    """
    Ctr = np.asarray(Ctr, dtype=np.float64)
    if Ctr.size == 0 or np.linalg.norm(Ctr) < EPS:
        return np.zeros((Ctr.shape[1], 0)), np.zeros(0)

    _, s, Vt = np.linalg.svd(Ctr, full_matrices=False)
    if len(s) == 0 or s[0] <= EPS:
        return np.zeros((Ctr.shape[1], 0)), s

    tol = max(Ctr.shape) * np.finfo(np.float64).eps * float(s[0])
    numerical_rank = int(np.sum(s > tol))

    frac = np.cumsum(s ** 2) / max(float(np.sum(s ** 2)), EPS)
    k_energy = int(np.searchsorted(frac, energy, side="left") + 1)

    k = min(numerical_rank, k_energy, int(max_rank))
    return Vt[:k].T.copy(), s[:k].copy()


def principal_pairs(Ui, Up):
    if Ui.shape[1] == 0 or Up.shape[1] == 0:
        D = Ui.shape[0]
        return (
            np.zeros(0),
            np.zeros((D, 0)),
            np.zeros((D, 0)),
        )

    A, sigma, Bt = np.linalg.svd(Ui.T @ Up, full_matrices=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    Pi = Ui @ A
    Pp = Up @ Bt.T
    return sigma, Pi, Pp


def orthonormalize_columns(A, tol=1e-10):
    A = np.asarray(A, dtype=np.float64)
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=np.float64)

    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if len(s) == 0 or s[0] <= EPS:
        return np.zeros((A.shape[0], 0), dtype=np.float64)

    keep = s > max(
        tol,
        max(A.shape) * np.finfo(np.float64).eps * float(s[0]),
    )
    return U[:, keep]


def shared_basis_from_principal(sigma, Pi, Pp, threshold):
    cols = []
    selected = []

    for j, sig in enumerate(sigma.tolist()):
        if sig < threshold:
            continue
        v = Pi[:, j] + Pp[:, j]
        nv = float(np.linalg.norm(v))
        if nv <= EPS:
            continue
        cols.append(v / nv)
        selected.append(j)

    if not cols:
        return np.zeros((Pi.shape[0], 0)), selected

    U = orthonormalize_columns(np.stack(cols, axis=1))
    return U, selected


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def normalize_rows(X):
    return X / np.maximum(
        np.linalg.norm(X, axis=1, keepdims=True),
        EPS,
    )


def spatial_centroid_metrics(Ytr, ytr, Yte, yte):
    """
    Same centered-cosine centroid classifier for every component.
    """
    Ytr = np.asarray(Ytr, dtype=np.float64)
    Yte = np.asarray(Yte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=object)
    yte = np.asarray(yte, dtype=object)

    if Ytr.shape[1] == 0 or np.linalg.norm(Ytr) < EPS:
        return {
            "spatial_acc4": float("nan"),
            "horizontal_acc": float("nan"),
            "vertical_acc": float("nan"),
        }

    center = Ytr.mean(axis=0, keepdims=True)

    dirs = []
    for c in RELATIONS:
        if not np.any(ytr == c):
            return {
                "spatial_acc4": float("nan"),
                "horizontal_acc": float("nan"),
                "vertical_acc": float("nan"),
            }
        v = Ytr[ytr == c].mean(axis=0) - center[0]
        dirs.append(v / max(float(np.linalg.norm(v)), EPS))

    D = np.stack(dirs, axis=0)
    scores = normalize_rows(Yte - center) @ D.T
    pred = np.asarray(
        [RELATIONS[i] for i in np.argmax(scores, axis=1)],
        dtype=object,
    )
    acc4 = float(np.mean(pred == yte))

    def pair_acc(classes):
        mask = np.isin(yte, classes)
        if not np.any(mask):
            return float("nan")

        dd = []
        for c in classes:
            v = Ytr[ytr == c].mean(axis=0) - center[0]
            dd.append(v / max(float(np.linalg.norm(v)), EPS))
        dd = np.stack(dd, axis=0)

        sc = normalize_rows(Yte[mask] - center) @ dd.T
        pp = np.argmax(sc, axis=1)
        gt = np.asarray(
            [classes.index(str(z)) for z in yte[mask]],
            dtype=np.int64,
        )
        return float(np.mean(pp == gt))

    return {
        "spatial_acc4": acc4,
        "horizontal_acc": pair_acc(["left", "right"]),
        "vertical_acc": pair_acc(["above", "below"]),
    }


def seen_class_centroid_acc(Ytr, labels_tr, Yte, labels_te):
    """
    Cosine nearest-centroid accuracy, evaluated only on TEST labels that
    appeared at least once in TRAIN.
    """
    labels_tr = np.asarray(labels_tr, dtype=object)
    labels_te = np.asarray(labels_te, dtype=object)
    classes = sorted(set(str(x) for x in labels_tr.tolist()))

    if not classes or np.linalg.norm(Ytr) < EPS:
        return float("nan"), 0

    center = np.asarray(Ytr, dtype=np.float64).mean(axis=0, keepdims=True)
    cents = []
    kept_classes = []

    for c in classes:
        m = labels_tr == c
        if not np.any(m):
            continue
        v = np.asarray(Ytr[m], dtype=np.float64).mean(axis=0) - center[0]
        nv = float(np.linalg.norm(v))
        if nv <= EPS:
            continue
        cents.append(v / nv)
        kept_classes.append(c)

    if not cents:
        return float("nan"), 0

    keep_set = set(kept_classes)
    mask = np.asarray([str(x) in keep_set for x in labels_te], dtype=bool)
    if not np.any(mask):
        return float("nan"), 0

    C = np.stack(cents, axis=0)
    scores = normalize_rows(np.asarray(Yte[mask]) - center) @ C.T
    pred_idx = np.argmax(scores, axis=1)
    pred = np.asarray([kept_classes[i] for i in pred_idx], dtype=object)
    gt = np.asarray([str(x) for x in labels_te[mask]], dtype=object)

    return float(np.mean(pred == gt)), int(np.sum(mask))


def fro_norm_ratio(C, H):
    return float(np.linalg.norm(C)) / max(float(np.linalg.norm(H)), EPS)


def mean_sample_norm_ratio(C, H):
    cn = np.linalg.norm(C, axis=1)
    hn = np.maximum(np.linalg.norm(H, axis=1), EPS)
    return float(np.mean(cn / hn))


def flattened_cosine(A, B):
    a = np.asarray(A, dtype=np.float64).reshape(-1)
    b = np.asarray(B, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b)) / max(
        float(np.linalg.norm(a) * np.linalg.norm(b)),
        EPS,
    )


def numerical_rank(X):
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0 or np.linalg.norm(X) < EPS:
        return 0
    s = np.linalg.svd(X, compute_uv=False)
    if len(s) == 0 or s[0] <= EPS:
        return 0
    tol = max(X.shape) * np.finfo(np.float64).eps * float(s[0])
    return int(np.sum(s > tol))


def vector_r2_from_prediction(Y, pred, baseline_mean):
    sse = float(np.sum((Y - pred) ** 2))
    sst = float(np.sum((Y - baseline_mean) ** 2))
    return 1.0 - sse / max(sst, EPS)


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) == 0 or np.all(~np.isfinite(a)):
        return float("nan"), float("nan")
    return float(np.nanmean(a)), float(np.nanstd(a))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if not (0.0 <= args.shared_threshold <= 1.0):
        raise ValueError("--shared-threshold must be in [0,1]")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    state_dir = Path(args.state_dir)
    correct = load_npz(state_dir / "raw__correct__all_layers.npz")
    noimg = load_npz(state_dir / "raw__no_image__all_layers.npz")

    (
        sids, states, relation, subject, reference,
        image_id, layers, layer_to_idx,
    ) = align_states(correct, noimg)

    if args.fixed_layer not in layer_to_idx:
        raise ValueError(
            f"L{args.fixed_layer} unavailable; layers={layers.tolist()}"
        )
    li = layer_to_idx[args.fixed_layer]

    valid = np.asarray([r in RELATIONS for r in relation], dtype=bool)
    sids = sids[valid]
    relation = relation[valid]
    subject = subject[valid]
    reference = reference[valid]
    image_id = image_id[valid]
    for k in list(states):
        states[k] = states[k][valid]

    bbox_rows = load_bbox_jsonl(Path(args.bbox_jsonl))
    df, audit = build_table(
        sids, relation, subject, reference, image_id,
        bbox_rows, args,
    )

    if len(df) < 40:
        raise RuntimeError(f"Too few usable samples: {len(df)}")

    sid_to_idx = {int(s): i for i, s in enumerate(sids.tolist())}
    ridx = np.asarray(
        [sid_to_idx[int(s)] for s in df["sid"].tolist()],
        dtype=np.int64,
    )

    y = relation[ridx]
    subj = subject[ridx]
    ref = reference[ridx]

    Araw = np.asarray(states["A_raw"][ridx, li], dtype=np.float64)
    Ano = np.asarray(states["A_noimg"][ridx, li], dtype=np.float64)
    Ares = np.asarray(states["A_residual"][ridx, li], dtype=np.float64)

    Braw = np.asarray(states["B_raw"][ridx, li], dtype=np.float64)
    Bno = np.asarray(states["B_noimg"][ridx, li], dtype=np.float64)
    Bres = np.asarray(states["B_residual"][ridx, li], dtype=np.float64)

    Lraw = np.asarray(states["last_raw"][ridx, li], dtype=np.float64)
    Lno = np.asarray(states["last_noimg"][ridx, li], dtype=np.float64)
    Lres = np.asarray(states["last_residual"][ridx, li], dtype=np.float64)

    target_map = {
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

    H = target_map[args.target]
    N, D = H.shape

    Xs, subject_vocab = one_hot(df["subject"].tolist())
    Xr, reference_vocab = one_hot(df["reference"].tolist())
    Xi = np.concatenate([Xs, Xr], axis=1)

    pos_names = ["pair_cx", "pair_cy", "dx", "dy"]
    Xp = df[pos_names].to_numpy(dtype=np.float64)
    Xrel = df[["dx", "dy"]].to_numpy(dtype=np.float64)

    splits = stratified_splits(
        y, args.train_ratio, args.repeats, args.seed
    )

    metrics_rows = []
    principal_rows = []
    coefficient_rows = []
    additivity_rows = []

    print("=" * 132)
    print("COEFFICIENT-SPACE HIDDEN DECOMPOSITION")
    print("=" * 132)
    print(
        f"target={args.target} | usable={N} | layer=L{args.fixed_layer} | "
        f"D={D} | repeats={args.repeats} | train_ratio={args.train_ratio:.2f}"
    )
    print(
        f"identity={Xi.shape} | position={Xp.shape} | "
        f"shared threshold sigma>={args.shared_threshold:.2f}"
    )
    print(
        "Decomposition: centered_hidden = identity + position + "
        "shared_residual + outside_shared"
    )
    print(
        f"bbox relation sign consistency="
        f"{100.0 * float(np.mean(df['bbox_relation_consistent'])):.2f}%"
    )

    for rep, (tr, te) in enumerate(splits):
        Htr, Hte = H[tr], H[te]
        ytr, yte = y[tr], y[te]
        str_, ste = subj[tr], subj[te]
        rtr, rte = ref[tr], ref[te]

        Xi_tr, Xi_te = Xi[tr], Xi[te]
        Xp_tr, Xp_te = Xp[tr], Xp[te]
        Xrel_tr, Xrel_te = Xrel[tr], Xrel[te]

        # -------------------------------------------------------------
        # 1) Learn identity/position hidden subspaces on TRAIN only.
        # -------------------------------------------------------------
        mid_vec = fit_ridge_matrix(Xi_tr, Htr, args.ridge)
        mpos_vec = fit_ridge_matrix(Xp_tr, Htr, args.ridge)

        Ui, _ = svd_hidden_basis(
            mid_vec["Ctr"],
            args.subspace_energy,
            args.max_identity_rank,
        )
        Up, _ = svd_hidden_basis(
            mpos_vec["Ctr"],
            args.subspace_energy,
            args.max_position_rank,
        )

        sigma, Pi, Pp = principal_pairs(Ui, Up)
        Ushared, selected = shared_basis_from_principal(
            sigma, Pi, Pp, args.shared_threshold
        )

        if Ushared.shape[1] == 0:
            raise RuntimeError(
                f"repeat={rep}: no shared direction passes "
                f"sigma>={args.shared_threshold}"
            )

        for j, sig in enumerate(sigma.tolist()):
            principal_rows.append({
                "repeat": rep,
                "principal_index": j + 1,
                "sigma": float(sig),
                "angle_deg": float(
                    np.degrees(np.arccos(np.clip(sig, 0.0, 1.0)))
                ),
                "selected": bool(j in selected),
                "identity_rank": int(Ui.shape[1]),
                "position_rank": int(Up.shape[1]),
                "shared_rank": int(Ushared.shape[1]),
            })

        # -------------------------------------------------------------
        # 2) Project actual hidden state into shared coefficient space.
        # -------------------------------------------------------------
        muH = Htr.mean(axis=0, keepdims=True)
        Htrc = Htr - muH
        Htec = Hte - muH
        Hallc = H - muH

        Ztr = Htrc @ Ushared
        Zte = Htec @ Ushared
        Zall = Hallc @ Ushared

        # -------------------------------------------------------------
        # 3) Joint coefficient regression:
        #       Z = identity_coeff + position_coeff + residual
        # -------------------------------------------------------------
        mj = fit_joint_groups(Xi_tr, Xp_tr, Ztr, args.ridge)

        Zi_tr, Zp_tr, Zhat_tr = joint_group_contributions(
            mj, Xi_tr, Xp_tr
        )
        Zi_te, Zp_te, Zhat_te = joint_group_contributions(
            mj, Xi_te, Xp_te
        )
        Zi_all, Zp_all, Zhat_all = joint_group_contributions(
            mj, Xi, Xp
        )

        # We fold any tiny intercept plus unexplained coefficient into Zres
        # so that the decomposition is exactly additive.
        Zres_tr = Ztr - Zi_tr - Zp_tr
        Zres_te = Zte - Zi_te - Zp_te
        Zres_all = Zall - Zi_all - Zp_all

        # -------------------------------------------------------------
        # 4) Map each coefficient attribution back to D dimensions.
        # -------------------------------------------------------------
        def back(Z):
            return Z @ Ushared.T

        Ctr = {}
        Cte = {}
        Call = {}

        Ctr["original_centered"] = Htrc
        Cte["original_centered"] = Htec
        Call["original_centered"] = Hallc

        Ctr["shared_projection"] = back(Ztr)
        Cte["shared_projection"] = back(Zte)
        Call["shared_projection"] = back(Zall)

        Ctr["identity_component"] = back(Zi_tr)
        Cte["identity_component"] = back(Zi_te)
        Call["identity_component"] = back(Zi_all)

        Ctr["position_component"] = back(Zp_tr)
        Cte["position_component"] = back(Zp_te)
        Call["position_component"] = back(Zp_all)

        Ctr["shared_residual"] = back(Zres_tr)
        Cte["shared_residual"] = back(Zres_te)
        Call["shared_residual"] = back(Zres_all)

        Ctr["outside_shared"] = Htrc - Ctr["shared_projection"]
        Cte["outside_shared"] = Htec - Cte["shared_projection"]
        Call["outside_shared"] = Hallc - Call["shared_projection"]

        Ctr["identity_plus_position"] = (
            Ctr["identity_component"] + Ctr["position_component"]
        )
        Cte["identity_plus_position"] = (
            Cte["identity_component"] + Cte["position_component"]
        )
        Call["identity_plus_position"] = (
            Call["identity_component"] + Call["position_component"]
        )

        # Exact reconstruction check.
        rec_te = (
            Cte["identity_component"]
            + Cte["position_component"]
            + Cte["shared_residual"]
            + Cte["outside_shared"]
        )
        abs_err = float(np.linalg.norm(Htec - rec_te))
        rel_err = abs_err / max(float(np.linalg.norm(Htec)), EPS)
        max_abs = float(np.max(np.abs(Htec - rec_te)))

        # Shared-basis spatial-axis capture.
        def relation_mean(c):
            return Htrc[ytr == c].mean(axis=0)

        xaxis = relation_mean("right") - relation_mean("left")
        yaxis = relation_mean("below") - relation_mean("above")
        xaxis = xaxis / max(float(np.linalg.norm(xaxis)), EPS)
        yaxis = yaxis / max(float(np.linalg.norm(yaxis)), EPS)
        xcap = float(np.sum((Ushared.T @ xaxis) ** 2))
        ycap = float(np.sum((Ushared.T @ yaxis) ** 2))

        # Coefficient-space held-out R2.
        coeff_r2 = vector_r2_from_prediction(
            Zte, Zhat_te, np.mean(Ztr, axis=0, keepdims=True)
        )
        coeff_id_only = heldout_vector_r2(
            Xi_tr, Xi_te, Ztr, Zte, args.ridge
        )
        coeff_pos_only = heldout_vector_r2(
            Xp_tr, Xp_te, Ztr, Zte, args.ridge
        )

        coeff_common = (
            coeff_id_only + coeff_pos_only - coeff_r2
        )
        coeff_unique_id = coeff_r2 - coeff_pos_only
        coeff_unique_pos = coeff_r2 - coeff_id_only

        coefficient_rows.append({
            "repeat": rep,
            "shared_rank": int(Ushared.shape[1]),
            "mean_selected_sigma": float(
                np.mean([sigma[j] for j in selected])
            ),
            "joint_coeff_r2": coeff_r2,
            "identity_only_coeff_r2": coeff_id_only,
            "position_only_coeff_r2": coeff_pos_only,
            "unique_identity_coeff_r2": coeff_unique_id,
            "unique_position_coeff_r2": coeff_unique_pos,
            "shared_commonality_coeff_r2": coeff_common,
            "identity_coeff_norm_over_actual": fro_norm_ratio(
                Zi_te, Zte
            ),
            "position_coeff_norm_over_actual": fro_norm_ratio(
                Zp_te, Zte
            ),
            "residual_coeff_norm_over_actual": fro_norm_ratio(
                Zres_te, Zte
            ),
            "identity_position_flat_cos": flattened_cosine(
                Zi_te, Zp_te
            ),
            "shared_x_axis_capture": xcap,
            "shared_y_axis_capture": ycap,
        })

        additivity_rows.append({
            "repeat": rep,
            "absolute_error_fro": abs_err,
            "relative_error_fro": rel_err,
            "max_absolute_element_error": max_abs,
        })

        # -------------------------------------------------------------
        # 5) Held-out probes for each decomposition component.
        # -------------------------------------------------------------
        for cname in (
            "original_centered",
            "shared_projection",
            "identity_component",
            "position_component",
            "identity_plus_position",
            "shared_residual",
            "outside_shared",
        ):
            Atr = Ctr[cname]
            Ate = Cte[cname]

            spa = spatial_centroid_metrics(Atr, ytr, Ate, yte)

            id_r2 = heldout_vector_r2(
                Xi_tr, Xi_te, Atr, Ate, args.ridge
            )
            pos_r2 = heldout_vector_r2(
                Xp_tr, Xp_te, Atr, Ate, args.ridge
            )
            rel_r2 = heldout_vector_r2(
                Xrel_tr, Xrel_te, Atr, Ate, args.ridge
            )

            sacc, sn = seen_class_centroid_acc(
                Atr, str_, Ate, ste
            )
            racc, rn = seen_class_centroid_acc(
                Atr, rtr, Ate, rte
            )

            metrics_rows.append({
                "repeat": rep,
                "component": cname,
                "component_train_rank": numerical_rank(Atr),
                "spatial_acc4": spa["spatial_acc4"],
                "horizontal_acc": spa["horizontal_acc"],
                "vertical_acc": spa["vertical_acc"],
                "identity_encode_r2": id_r2,
                "position_encode_r2": pos_r2,
                "relative_encode_r2": rel_r2,
                "subject_centroid_acc_seen": sacc,
                "subject_seen_n": sn,
                "reference_centroid_acc_seen": racc,
                "reference_seen_n": rn,
                "fro_norm_over_original": fro_norm_ratio(Ate, Htec),
                "mean_sample_norm_over_original": mean_sample_norm_ratio(
                    Ate, Htec
                ),
            })

        # -------------------------------------------------------------
        # 6) Save per-repeat decomposition for downstream analysis.
        # -------------------------------------------------------------
        if not args.no_save_components:
            is_train = np.zeros(N, dtype=bool)
            is_train[tr] = True

            np.savez_compressed(
                outdir / f"repeat_{rep:02d}_components.npz",
                sid=df["sid"].to_numpy(dtype=np.int64),
                relation=np.asarray(y, dtype=object),
                subject=np.asarray(subj, dtype=object),
                reference=np.asarray(ref, dtype=object),
                is_train=is_train,
                train_indices=tr.astype(np.int64),
                test_indices=te.astype(np.int64),
                train_hidden_mean=muH.astype(np.float32),
                shared_basis=Ushared.astype(np.float32),
                principal_sigma=sigma.astype(np.float32),
                selected_principal_indices=np.asarray(
                    selected, dtype=np.int64
                ),
                Z_actual=Zall.astype(np.float32),
                Z_identity=Zi_all.astype(np.float32),
                Z_position=Zp_all.astype(np.float32),
                Z_shared_residual=Zres_all.astype(np.float32),
                original_centered=Call["original_centered"].astype(np.float32),
                shared_projection=Call["shared_projection"].astype(np.float32),
                identity_component=Call["identity_component"].astype(np.float32),
                position_component=Call["position_component"].astype(np.float32),
                identity_plus_position=Call["identity_plus_position"].astype(np.float32),
                shared_residual=Call["shared_residual"].astype(np.float32),
                outside_shared=Call["outside_shared"].astype(np.float32),
            )

    # -----------------------------------------------------------------
    # Save CSVs / metadata
    # -----------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    principal_df = pd.DataFrame(principal_rows)
    coefficient_df = pd.DataFrame(coefficient_rows)
    additivity_df = pd.DataFrame(additivity_rows)

    metrics_df.to_csv(
        outdir / "component_metrics_by_split.csv", index=False
    )
    principal_df.to_csv(
        outdir / "principal_directions_by_split.csv", index=False
    )
    coefficient_df.to_csv(
        outdir / "coefficient_model_by_split.csv", index=False
    )
    additivity_df.to_csv(
        outdir / "additivity_check_by_split.csv", index=False
    )

    # Aggregated summary CSV.
    summary_rows = []
    for cname in metrics_df["component"].unique():
        d = metrics_df[metrics_df["component"] == cname]
        row = {"component": cname}
        for col in (
            "component_train_rank",
            "spatial_acc4",
            "horizontal_acc",
            "vertical_acc",
            "identity_encode_r2",
            "position_encode_r2",
            "relative_encode_r2",
            "subject_centroid_acc_seen",
            "reference_centroid_acc_seen",
            "fro_norm_over_original",
            "mean_sample_norm_over_original",
        ):
            m, s = mean_std(d[col])
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        outdir / "component_metrics_summary.csv", index=False
    )

    with (outdir / "audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "target": args.target,
        "fixed_layer": int(args.fixed_layer),
        "usable_samples": int(N),
        "hidden_dim": int(D),
        "identity_dim": int(Xi.shape[1]),
        "position_dim": int(Xp.shape[1]),
        "relative_dim": int(Xrel.shape[1]),
        "subject_vocab_size": int(len(subject_vocab)),
        "reference_vocab_size": int(len(reference_vocab)),
        "shared_threshold": float(args.shared_threshold),
        "subspace_energy": float(args.subspace_energy),
        "repeats": int(args.repeats),
        "train_ratio": float(args.train_ratio),
        "ridge": float(args.ridge),
        "bbox_relation_sign_consistency": float(
            np.mean(df["bbox_relation_consistent"].to_numpy(dtype=bool))
        ),
        "decomposition_equation": (
            "centered_hidden = identity_component + position_component + "
            "shared_residual + outside_shared"
        ),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------
    # Human-readable result
    # -----------------------------------------------------------------
    print()
    print("=" * 132)
    print("1) PRINCIPAL / COEFFICIENT SUMMARY")
    print("=" * 132)

    for j in sorted(principal_df["principal_index"].unique()):
        d = principal_df[principal_df["principal_index"] == j]
        sm, ss = mean_std(d["sigma"])
        am, ast = mean_std(d["angle_deg"])
        sel = float(np.mean(d["selected"].astype(float)))
        print(
            f"PC{int(j):02d} | sigma={sm:.4f}±{ss:.4f} | "
            f"angle={am:.2f}°±{ast:.2f}° | selected={sel:.2f}"
        )

    print()
    for col in (
        "shared_rank",
        "mean_selected_sigma",
        "joint_coeff_r2",
        "identity_only_coeff_r2",
        "position_only_coeff_r2",
        "unique_identity_coeff_r2",
        "unique_position_coeff_r2",
        "shared_commonality_coeff_r2",
        "identity_coeff_norm_over_actual",
        "position_coeff_norm_over_actual",
        "residual_coeff_norm_over_actual",
        "identity_position_flat_cos",
        "shared_x_axis_capture",
        "shared_y_axis_capture",
    ):
        m, s = mean_std(coefficient_df[col])
        print(f"{col:36s} = {m:+.4f}±{s:.4f}")

    print()
    print("=" * 132)
    print("2) DECOMPOSED COMPONENTS: HELD-OUT METRICS")
    print("=" * 132)
    print(
        f"{'component':26s} | {'spatial':>7s} | {'H':>7s} | {'V':>7s} | "
        f"{'idR2':>8s} | {'posR2':>8s} | {'relR2':>8s} | "
        f"{'subj':>7s} | {'ref':>7s} | {'norm':>7s}"
    )
    print("-" * 132)

    order = [
        "original_centered",
        "shared_projection",
        "identity_component",
        "position_component",
        "identity_plus_position",
        "shared_residual",
        "outside_shared",
    ]

    for cname in order:
        d = metrics_df[metrics_df["component"] == cname]
        vals = {}
        for col in (
            "spatial_acc4",
            "horizontal_acc",
            "vertical_acc",
            "identity_encode_r2",
            "position_encode_r2",
            "relative_encode_r2",
            "subject_centroid_acc_seen",
            "reference_centroid_acc_seen",
            "fro_norm_over_original",
        ):
            vals[col] = mean_std(d[col])[0]

        print(
            f"{cname:26s} | "
            f"{vals['spatial_acc4']:7.4f} | "
            f"{vals['horizontal_acc']:7.4f} | "
            f"{vals['vertical_acc']:7.4f} | "
            f"{vals['identity_encode_r2']:+8.4f} | "
            f"{vals['position_encode_r2']:+8.4f} | "
            f"{vals['relative_encode_r2']:+8.4f} | "
            f"{vals['subject_centroid_acc_seen']:7.4f} | "
            f"{vals['reference_centroid_acc_seen']:7.4f} | "
            f"{vals['fro_norm_over_original']:7.3f}"
        )

    print()
    print("=" * 132)
    print("3) EXACT ADDITIVITY CHECK")
    print("=" * 132)
    for col in (
        "relative_error_fro",
        "max_absolute_element_error",
    ):
        m, s = mean_std(additivity_df[col])
        print(f"{col:32s} = {m:.6e}±{s:.6e}")

    print()
    print("Interpretation target:")
    print(
        "  A useful position-attributed component should retain high spatial/relative "
        "metrics while showing substantially lower identity metrics than the "
        "identity-attributed component."
    )
    print(
        "  Do NOT require identity_component and position_component to be orthogonal; "
        "they intentionally reuse the same shared basis."
    )

    print()
    print("OUTPUTS:")
    for fn in (
        "component_metrics_by_split.csv",
        "component_metrics_summary.csv",
        "principal_directions_by_split.csv",
        "coefficient_model_by_split.csv",
        "additivity_check_by_split.csv",
        "summary.json",
        "audit.jsonl",
    ):
        print(outdir / fn)

    if not args.no_save_components:
        print(outdir / "repeat_XX_components.npz")


if __name__ == "__main__":
    main()
