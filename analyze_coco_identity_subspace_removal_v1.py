#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identity-subspace removal for COCO-two raw-token spatial representations.

Goal
----
Start from image-conditioned residual hidden states, especially

    Diff_residual = (h_A^img-h_A^noimg) - (h_B^img-h_B^noimg)

and ask whether subject/reference identity can be removed while preserving or
improving spatial information.

The identity subspace is learned ONLY on each training split:

  1) fit subject_id + reference_id -> centered hidden state with ridge;
  2) collect the fitted identity contribution C_train [N_train, D];
  3) SVD(C_train) to obtain ranked hidden-space identity directions;
  4) remove the first k directions from centered train/test hidden states.

For each k, report held-out:
  * 4-way spatial centroid accuracy + horizontal/vertical accuracy
  * subject-only / reference-only / identity-union reconstruction R2
  * relative(dx,dy)-only R2
  * nonredundant position(pair_center + dxdy) R2
  * relation-label-only R2
  * amount of hidden-state energy removed
  * overlap of the removed identity subspace with right-left / below-above axes
  * cosine of spatial axes before vs after cleaning

Also report a conservative comparison:

    regression_residualization = Y - predicted_identity_component

which subtracts only the identity component predicted for that sample rather
than projecting away the whole hidden-space identity span.

No test hidden states are used to learn the identity subspace.
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--state-dir", required=True,
                   help="Directory with raw__correct__all_layers.npz and raw__no_image__all_layers.npz")
    p.add_argument("--bbox-jsonl", default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fixed-layer", type=int, default=25)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="lambda = ridge * trace(X'X)/p")
    p.add_argument("--ranks", default="1,2,4,8,16,32,64,full",
                   help="Identity-subspace ranks to remove. 'full' uses numerical rank.")
    p.add_argument("--targets", default="A_residual,B_residual,last_residual,Diff_residual",
                   help="Comma-separated targets")
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument("--require-gt-consistent", action="store_true",
                   help="Sensitivity analysis only; conditions on GT relation")
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a)*np.linalg.norm(b)), EPS))


def ridge_map_fit(Xtr: np.ndarray, Ytr: np.ndarray, ridge: float):
    """Fit centered ridge X -> Y and return training parameters and fitted centered contribution."""
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
    A = gram + lam*np.eye(p, dtype=np.float64)
    try:
        W = np.linalg.solve(A, Xz.T @ Yc)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ Xz.T @ Yc
    Ctr = Xz @ W
    return {"xmu": xmu, "xsd": xsd, "ymu": ymu, "W": W, "Ctr": Ctr}


def ridge_map_predict(model: Mapping[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xz = (np.asarray(X, dtype=np.float64)-model["xmu"]) / model["xsd"]
    return Xz @ model["W"]


def global_r2_from_train_test(Xtr, Xte, Ytr, Yte, ridge: float) -> float:
    model = ridge_map_fit(Xtr, Ytr, ridge)
    Cte = ridge_map_predict(model, Xte)
    pred = model["ymu"] + Cte
    sse = float(np.sum((Yte-pred)**2))
    sst = float(np.sum((Yte-model["ymu"])**2))
    return 1.0 - sse/max(sst, EPS)


def spatial_axes(Ytr: np.ndarray, ytr: np.ndarray):
    def mean(c):
        return Ytr[ytr == c].mean(axis=0)
    dx = mean("right") - mean("left")
    dy = mean("below") - mean("above")
    dx = dx / max(float(np.linalg.norm(dx)), EPS)
    dy = dy / max(float(np.linalg.norm(dy)), EPS)
    return dx, dy


def centroid_predict(Ytr: np.ndarray, ytr: np.ndarray, Yte: np.ndarray):
    center = Ytr.mean(axis=0, keepdims=True)
    dirs = []
    for c in RELATIONS:
        v = Ytr[ytr == c].mean(axis=0) - center[0]
        dirs.append(v / max(float(np.linalg.norm(v)), EPS))
    D = np.stack(dirs, axis=0)
    scores = normalize_rows(Yte-center) @ D.T
    pred = np.argmax(scores, axis=1)
    return np.asarray([RELATIONS[i] for i in pred], dtype=object), scores


def spatial_acc_metrics(Ytr: np.ndarray, ytr: np.ndarray, Yte: np.ndarray, yte: np.ndarray):
    pred, _ = centroid_predict(Ytr, ytr, Yte)
    acc = float(np.mean(pred == yte))
    hmask = np.isin(yte, ["left", "right"])
    vmask = np.isin(yte, ["above", "below"])
    # Pair-specific classifier avoids penalizing H/V with irrelevant opposite axis labels.
    def pair_acc(classes, mask):
        if not np.any(mask):
            return float("nan")
        center = Ytr.mean(axis=0)
        ds = []
        for c in classes:
            v = Ytr[ytr == c].mean(axis=0)-center
            ds.append(v/max(float(np.linalg.norm(v)), EPS))
        D = np.stack(ds, axis=0)
        sc = normalize_rows(Yte[mask]-center) @ D.T
        pp = np.argmax(sc, axis=1)
        gt = np.asarray([classes.index(str(z)) for z in yte[mask]], dtype=np.int64)
        return float(np.mean(pp == gt))
    return {
        "spatial_acc4": acc,
        "horizontal_acc": pair_acc(["left", "right"], hmask),
        "vertical_acc": pair_acc(["above", "below"], vmask),
    }


def numeric_rank_from_s(s: np.ndarray) -> int:
    if len(s) == 0 or float(s[0]) <= 0:
        return 0
    tol = max(1e-8, max(s.shape)*np.finfo(np.float64).eps*float(s[0]))
    return int(np.sum(s > tol))


def parse_rank_specs(text: str) -> List[str]:
    out = []
    for t in text.split(","):
        t = t.strip().lower()
        if not t:
            continue
        if t == "full":
            out.append("full")
        else:
            k = int(t)
            if k <= 0:
                raise ValueError("All numeric ranks must be > 0")
            out.append(str(k))
    if not out:
        raise ValueError("No ranks requested")
    return out


def evaluate_variant(
    target: str,
    repeat: int,
    variant: str,
    rank_removed: int,
    Ytr_orig: np.ndarray,
    Yte_orig: np.ndarray,
    Ytr: np.ndarray,
    Yte: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    Xs_tr: np.ndarray,
    Xs_te: np.ndarray,
    Xr_tr: np.ndarray,
    Xr_te: np.ndarray,
    Xid_tr: np.ndarray,
    Xid_te: np.ndarray,
    Xrel_tr: np.ndarray,
    Xrel_te: np.ndarray,
    Xpos_tr: np.ndarray,
    Xpos_te: np.ndarray,
    Xlab_tr: np.ndarray,
    Xlab_te: np.ndarray,
    ridge: float,
    Uremoved: np.ndarray | None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "target": target, "repeat": repeat, "variant": variant,
        "rank_removed": rank_removed,
    }
    out.update(spatial_acc_metrics(Ytr, ytr, Yte, yte))
    out["subject_only_r2"] = global_r2_from_train_test(Xs_tr, Xs_te, Ytr, Yte, ridge)
    out["reference_only_r2"] = global_r2_from_train_test(Xr_tr, Xr_te, Ytr, Yte, ridge)
    out["identity_union_r2"] = global_r2_from_train_test(Xid_tr, Xid_te, Ytr, Yte, ridge)
    out["relative_only_r2"] = global_r2_from_train_test(Xrel_tr, Xrel_te, Ytr, Yte, ridge)
    out["position_nonredundant_r2"] = global_r2_from_train_test(Xpos_tr, Xpos_te, Ytr, Yte, ridge)
    out["relation_label_r2"] = global_r2_from_train_test(Xlab_tr, Xlab_te, Ytr, Yte, ridge)

    mu = Ytr_orig.mean(axis=0, keepdims=True)
    orig_c = Yte_orig-mu
    removed = Yte_orig-Yte
    out["removed_energy_ratio"] = float(np.linalg.norm(removed) / max(float(np.linalg.norm(orig_c)), EPS))

    dx0, dy0 = spatial_axes(Ytr_orig, ytr)
    dxc, dyc = spatial_axes(Ytr, ytr)
    out["cos_xaxis_clean_vs_original"] = cosine(dxc, dx0)
    out["cos_yaxis_clean_vs_original"] = cosine(dyc, dy0)

    if Uremoved is None or Uremoved.size == 0:
        out["xaxis_energy_in_removed_subspace"] = 0.0
        out["yaxis_energy_in_removed_subspace"] = 0.0
    else:
        out["xaxis_energy_in_removed_subspace"] = float(np.sum((Uremoved.T @ dx0)**2))
        out["yaxis_energy_in_removed_subspace"] = float(np.sum((Uremoved.T @ dy0)**2))
    return out


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
    sids, states, relation, subject, reference, image_id, layers, layer_to_idx = align_states(correct, noimg)
    if args.fixed_layer not in layer_to_idx:
        raise ValueError(f"L{args.fixed_layer} unavailable; layers={layers.tolist()}")
    li = layer_to_idx[args.fixed_layer]

    valid_rel = np.asarray([r in RELATIONS for r in relation], dtype=bool)
    sids, relation, subject, reference, image_id = (
        sids[valid_rel], relation[valid_rel], subject[valid_rel], reference[valid_rel], image_id[valid_rel]
    )
    for k in list(states.keys()):
        states[k] = states[k][valid_rel]

    gdino = load_gdino_rows(Path(args.bbox_jsonl))
    geom_df, audit = build_geometry_table(sids, relation, subject, reference, image_id, gdino, args)
    if len(geom_df) < 40:
        raise RuntimeError(f"Too few usable bbox samples: {len(geom_df)}")

    sid_to_state = {int(s): i for i, s in enumerate(sids.tolist())}
    ridx = np.asarray([sid_to_state[int(s)] for s in geom_df["sid"].tolist()], dtype=np.int64)
    y = relation[ridx]

    # Build residual targets at fixed layer.
    A = np.asarray(states["A_residual"][ridx, li], dtype=np.float64)
    B = np.asarray(states["B_residual"][ridx, li], dtype=np.float64)
    L = np.asarray(states["last_residual"][ridx, li], dtype=np.float64)
    targets_all = {
        "A_residual": A,
        "B_residual": B,
        "last_residual": L,
        "Diff_residual": A-B,
    }
    wanted = [x.strip() for x in args.targets.split(",") if x.strip()]
    bad = [x for x in wanted if x not in targets_all]
    if bad:
        raise ValueError(f"Unsupported targets {bad}; available={list(targets_all)}")
    targets = {k: targets_all[k] for k in wanted}

    # Design matrices. Identity is deliberately separate from geometry.
    Xs, svocab = one_hot(geom_df["subject"].tolist())
    Xr, rvocab = one_hot(geom_df["reference"].tolist())
    Xid = np.concatenate([Xs, Xr], axis=1)
    Xrel = geom_df[["dx", "dy"]].to_numpy(dtype=np.float64)
    # Nonredundant position family: pair center + relative displacement.
    Xpos = geom_df[["pair_cx", "pair_cy", "dx", "dy"]].to_numpy(dtype=np.float64)
    Xlab, relvocab = one_hot(geom_df["relation"].tolist())

    splits = make_stratified_splits(y, args.train_ratio, args.repeats, args.seed)
    rank_specs = parse_rank_specs(args.ranks)
    rows: List[Dict[str, Any]] = []
    svd_rows: List[Dict[str, Any]] = []

    for target_name, Y in targets.items():
        D = Y.shape[1]
        for rep, (tr, te) in enumerate(splits):
            ytr, yte = y[tr], y[te]
            Ytr0, Yte0 = Y[tr], Y[te]

            # Learn identity mapping using TRAIN ONLY.
            idmodel = ridge_map_fit(Xid[tr], Ytr0, args.ridge)
            Ctr = np.asarray(idmodel["Ctr"], dtype=np.float64)  # [Ntr, D]
            Cte = ridge_map_predict(idmodel, Xid[te])           # [Nte, D]

            # Ranked hidden-space identity basis from fitted identity contribution.
            _, svals, Vt = np.linalg.svd(Ctr, full_matrices=False)
            nrank = numeric_rank_from_s(svals)
            energy = svals**2
            energy_frac = energy / max(float(np.sum(energy)), EPS)
            cumulative = np.cumsum(energy_frac)
            for j in range(min(nrank, len(svals))):
                svd_rows.append({
                    "target": target_name, "repeat": rep, "component": j+1,
                    "singular_value": float(svals[j]),
                    "identity_energy_fraction": float(energy_frac[j]),
                    "identity_energy_cumulative": float(cumulative[j]),
                    "numerical_rank": nrank,
                })

            # Baseline.
            rows.append(evaluate_variant(
                target_name, rep, "baseline", 0,
                Ytr0, Yte0, Ytr0, Yte0, ytr, yte,
                Xs[tr], Xs[te], Xr[tr], Xr[te], Xid[tr], Xid[te],
                Xrel[tr], Xrel[te], Xpos[tr], Xpos[te], Xlab[tr], Xlab[te],
                args.ridge, None,
            ))

            mu = Ytr0.mean(axis=0, keepdims=True)
            Ytrc, Ytec = Ytr0-mu, Yte0-mu

            for rs in rank_specs:
                req = nrank if rs == "full" else int(rs)
                k = min(req, nrank)
                if k <= 0:
                    continue
                U = Vt[:k].T  # [D,k], orthonormal columns
                Ytr_clean = mu + (Ytrc - (Ytrc @ U) @ U.T)
                Yte_clean = mu + (Ytec - (Ytec @ U) @ U.T)
                rows.append(evaluate_variant(
                    target_name, rep, f"project_rank_{rs}", k,
                    Ytr0, Yte0, Ytr_clean, Yte_clean, ytr, yte,
                    Xs[tr], Xs[te], Xr[tr], Xr[te], Xid[tr], Xid[te],
                    Xrel[tr], Xrel[te], Xpos[tr], Xpos[te], Xlab[tr], Xlab[te],
                    args.ridge, U,
                ))

            # Conservative sample-specific residualization: subtract only predicted identity effect.
            Ytr_reg = Ytr0 - Ctr
            Yte_reg = Yte0 - Cte
            rows.append(evaluate_variant(
                target_name, rep, "regression_residualization", nrank,
                Ytr0, Yte0, Ytr_reg, Yte_reg, ytr, yte,
                Xs[tr], Xs[te], Xr[tr], Xr[te], Xid[tr], Xid[te],
                Xrel[tr], Xrel[te], Xpos[tr], Xpos[te], Xlab[tr], Xlab[te],
                args.ridge, None,
            ))

    per_split = pd.DataFrame(rows)
    per_split.to_csv(outdir / "identity_removal_per_split.csv", index=False)
    pd.DataFrame(svd_rows).to_csv(outdir / "identity_subspace_spectrum.csv", index=False)
    geom_df.to_csv(outdir / "gdino_geometry_used.csv", index=False)
    (outdir / "bbox_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_cols = [
        "spatial_acc4", "horizontal_acc", "vertical_acc",
        "subject_only_r2", "reference_only_r2", "identity_union_r2",
        "relative_only_r2", "position_nonredundant_r2", "relation_label_r2",
        "removed_energy_ratio", "xaxis_energy_in_removed_subspace", "yaxis_energy_in_removed_subspace",
        "cos_xaxis_clean_vs_original", "cos_yaxis_clean_vs_original",
    ]
    summary_rows = []
    for (target, variant), d in per_split.groupby(["target", "variant"], sort=False):
        row = {
            "target": target,
            "variant": variant,
            "rank_removed_mean": float(d["rank_removed"].mean()),
        }
        for c in metric_cols:
            m, s = mean_std(d[c].to_numpy())
            row[c+"_mean"] = m
            row[c+"_std"] = s
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Deltas relative to the baseline for each target.
    delta_metrics = [
        "spatial_acc4", "horizontal_acc", "vertical_acc",
        "subject_only_r2", "reference_only_r2", "identity_union_r2",
        "relative_only_r2", "position_nonredundant_r2", "relation_label_r2",
    ]
    for target in summary["target"].unique():
        b = summary[(summary.target == target) & (summary.variant == "baseline")]
        if len(b) != 1:
            continue
        b = b.iloc[0]
        mask = summary.target == target
        for c in delta_metrics:
            summary.loc[mask, "delta_"+c] = summary.loc[mask, c+"_mean"] - float(b[c+"_mean"])

    summary.to_csv(outdir / "identity_removal_summary.csv", index=False)

    bbox_cons = float(np.mean(geom_df["bbox_relation_consistent"].to_numpy(dtype=bool)))
    print("\n" + "="*132)
    print("IDENTITY SUBSPACE REMOVAL")
    print("="*132)
    print(f"usable bbox samples={len(geom_df)} | layer=L{args.fixed_layer} | train_ratio={args.train_ratio:.2f} | repeats={args.repeats}")
    print(f"hidden dim={next(iter(targets.values())).shape[1]} | identity design={Xid.shape} (subject={Xs.shape[1]}, reference={Xr.shape[1]})")
    print(f"bbox relation sign consistency={100*bbox_cons:.2f}%")
    print("identity basis is learned on TRAIN ONLY from ridge-predicted subject+reference effects, then SVD-ranked.")
    print("position metric uses nonredundant [pair_cx,pair_cy,dx,dy].\n")

    print("HOW TO READ")
    print("  wanted: identity_union_R2 DOWN, subject/reference_R2 DOWN")
    print("          relative/position R2 SAME or UP, spatial ACC SAME or UP")
    print("  x/y axis-energy-in-removed-subspace tells how much the identity subspace overlaps spatial axes.\n")

    for target in wanted:
        print("-"*132)
        print(target)
        print("-"*132)
        d = summary[summary.target == target]
        for _, r in d.iterrows():
            print(
                f"{r.variant:28s} | k={r.rank_removed_mean:5.1f} "
                f"| spatial={r.spatial_acc4_mean:.4f} ({r.get('delta_spatial_acc4', np.nan):+.4f}) "
                f"| H={r.horizontal_acc_mean:.4f} | V={r.vertical_acc_mean:.4f} "
                f"| idR2={r.identity_union_r2_mean:+.4f} ({r.get('delta_identity_union_r2', np.nan):+.4f}) "
                f"| relR2={r.relative_only_r2_mean:+.4f} ({r.get('delta_relative_only_r2', np.nan):+.4f}) "
                f"| posR2={r.position_nonredundant_r2_mean:+.4f} ({r.get('delta_position_nonredundant_r2', np.nan):+.4f}) "
                f"| removed={r.removed_energy_ratio_mean:.3f}"
            )
        print("\n  spatial-axis overlap with removed identity subspace (projection variants):")
        for _, r in d[d.variant.str.startswith("project_rank_")].iterrows():
            print(
                f"    {r.variant:24s} | x-overlap={r.xaxis_energy_in_removed_subspace_mean:.3f} "
                f"| y-overlap={r.yaxis_energy_in_removed_subspace_mean:.3f} "
                f"| cos x(clean,orig)={r.cos_xaxis_clean_vs_original_mean:.3f} "
                f"| cos y(clean,orig)={r.cos_yaxis_clean_vs_original_mean:.3f}"
            )

    print("\nSaved:")
    for name in (
        "identity_removal_summary.csv", "identity_removal_per_split.csv",
        "identity_subspace_spectrum.csv", "gdino_geometry_used.csv", "bbox_audit.json",
    ):
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()
