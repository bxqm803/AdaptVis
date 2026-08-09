#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate coefficient superposition on shared identity-position directions.

For each train/test split:
1) Learn identity-only and position-only vector encoders on TRAIN.
2) Compute principal directions between their learned hidden subspaces.
3) For each high-overlap principal pair, define one shared hidden direction u_j.
4) Project held-out hidden states onto u_j:
       z_ij = (h_i - mean_train(h)) @ u_j
5) Fit scalar encoders on TRAIN:
       z_j ~ identity
       z_j ~ position
       z_j ~ identity + position
6) Decompose the joint prediction:
       z_hat = mean_z + a_identity + a_position

This tests whether the SAME hidden direction carries complementary,
sample-dependent identity and position coefficients.

Default target:
    Diff_residual
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
    req = {
        "sample_index", "relation", "subject", "reference",
        "decoder_block_index", "A_vectors", "B_vectors", "last_vectors",
    }
    for name, obj in (("correct", correct), ("noimg", noimg)):
        missing = sorted(req - set(obj.keys()))
        if missing:
            raise KeyError(f"{name} NPZ missing keys: {missing}")

    cids = np.asarray(correct["sample_index"], dtype=np.int64)
    nids = np.asarray(noimg["sample_index"], dtype=np.int64)
    cmap = {int(s): i for i, s in enumerate(cids.tolist())}
    nmap = {int(s): i for i, s in enumerate(nids.tolist())}
    common = np.asarray(
        [int(s) for s in cids.tolist() if int(s) in nmap], dtype=np.int64
    )
    ci = np.asarray([cmap[int(s)] for s in common], dtype=np.int64)
    ni = np.asarray([nmap[int(s)] for s in common], dtype=np.int64)

    lc = np.asarray(correct["decoder_block_index"], dtype=np.int64)
    ln = np.asarray(noimg["decoder_block_index"], dtype=np.int64)
    if not np.array_equal(lc, ln):
        raise ValueError("correct/noimg layer lists differ")
    layer_to_idx = {int(v): i for i, v in enumerate(lc.tolist())}

    out = {}
    for slot, key in (
        ("A", "A_vectors"),
        ("B", "B_vectors"),
        ("last", "last_vectors"),
    ):
        c = np.asarray(correct[key][ci], dtype=np.float32)
        n = np.asarray(noimg[key][ni], dtype=np.float32)
        out[f"{slot}_raw"] = c
        out[f"{slot}_noimg"] = n
        out[f"{slot}_residual"] = c - n

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
    return common, out, relation, subject, reference, image_id, lc, layer_to_idx


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
                raise ValueError(f"Bad JSONL {path}:{line_no}: {exc}") from exc
    return out


def selected_box(obj: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    s = obj.get("selected")
    if not isinstance(s, Mapping):
        raise ValueError("missing selected bbox")
    b = s.get("box_xyxy_normalized")
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError("bad normalized bbox")
    arr = np.asarray([float(v) for v in b], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("nonfinite bbox")
    arr = np.clip(arr, 0.0, 1.0)
    x1, y1, x2, y2 = arr.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate bbox")
    return arr, float(s.get("score", np.nan))


def box_stats(b: np.ndarray):
    x1, y1, x2, y2 = [float(v) for v in b]
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


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


def build_table(sids, relation, subject, reference, bbox_rows, args):
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
            so, ro = g.get("subject", {}), g.get("reference", {})
            if not args.include_ambiguous:
                if bool(so.get("ambiguous", False)) or bool(ro.get("ambiguous", False)):
                    raise ValueError("ambiguous_bbox")
            ba, sa = selected_box(so)
            bb, sb = selected_box(ro)
            if not np.isfinite(sa) or not np.isfinite(sb):
                raise ValueError("nonfinite score")
            if sa < args.min_score or sb < args.min_score:
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
                "subject": canonical_phrase(subject[i]),
                "reference": canonical_phrase(reference[i]),
                "relation": rel,
                "pair_cx": (cxA + cxB) / 2,
                "pair_cy": (cyA + cyB) / 2,
                "dx": dx,
                "dy": dy,
                "bbox_relation_consistent": ok,
            })
        except Exception as exc:
            audit.append({"sid": sid, "reason": str(exc)})
    return pd.DataFrame(rows), audit


def one_hot(values: Sequence[str]):
    vals = [str(x) for x in values]
    vocab = sorted(set(vals))
    m = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(vals), len(vocab)), dtype=np.float64)
    for i, v in enumerate(vals):
        X[i, m[v]] = 1.0
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
                raise RuntimeError(f"too few samples for {c}")
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


def ridge_lambda(Xz, ridge):
    p = Xz.shape[1]
    G = Xz.T @ Xz
    lam = float(ridge) * float(np.trace(G) / max(p, 1))
    return G, lam


def fit_ridge_matrix(Xtr, Ytr, ridge):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)

    xmu = Xtr.mean(axis=0)
    xsd = Xtr.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xz = (Xtr - xmu) / xsd

    ymu = Ytr.mean(axis=0, keepdims=True)
    Yc = Ytr - ymu

    G, lam = ridge_lambda(Xz, ridge)
    A = G + lam * np.eye(Xz.shape[1])
    try:
        W = np.linalg.solve(A, Xz.T @ Yc)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ Xz.T @ Yc

    return {"xmu": xmu, "xsd": xsd, "ymu": ymu, "W": W, "Ctr": Xz @ W}


def matrix_predict(model, X):
    Xz = (np.asarray(X, dtype=np.float64) - model["xmu"]) / model["xsd"]
    return model["ymu"] + Xz @ model["W"]


def svd_basis(Ctr, energy, max_rank):
    Ctr = np.asarray(Ctr, dtype=np.float64)
    _, s, Vt = np.linalg.svd(Ctr, full_matrices=False)
    if len(s) == 0 or s[0] <= EPS:
        return np.zeros((Ctr.shape[1], 0)), np.zeros(0)
    tol = max(Ctr.shape) * np.finfo(np.float64).eps * float(s[0])
    rank = int(np.sum(s > tol))
    frac = np.cumsum(s ** 2) / max(float(np.sum(s ** 2)), EPS)
    k_energy = int(np.searchsorted(frac, energy, side="left") + 1)
    k = min(rank, k_energy, int(max_rank))
    return Vt[:k].T.copy(), s[:k].copy()


def principal_pairs(Ui, Up):
    if Ui.shape[1] == 0 or Up.shape[1] == 0:
        return np.zeros(0), np.zeros((Ui.shape[0], 0)), np.zeros((Up.shape[0], 0))
    A, sigma, Bt = np.linalg.svd(Ui.T @ Up, full_matrices=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    return sigma, Ui @ A, Up @ Bt.T


def midpoint_direction(a, b):
    u = a + b
    n = float(np.linalg.norm(u))
    if n < EPS:
        return None
    return u / n


def fit_scalar(Xtr, ztr, ridge):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    ztr = np.asarray(ztr, dtype=np.float64).reshape(-1)

    xmu = Xtr.mean(axis=0)
    xsd = Xtr.std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xz = (Xtr - xmu) / xsd

    zmu = float(np.mean(ztr))
    zc = ztr - zmu

    G, lam = ridge_lambda(Xz, ridge)
    A = G + lam * np.eye(Xz.shape[1])
    try:
        w = np.linalg.solve(A, Xz.T @ zc)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ Xz.T @ zc

    return {"xmu": xmu, "xsd": xsd, "zmu": zmu, "w": w}


def pred_scalar(model, X):
    Xz = (np.asarray(X, dtype=np.float64) - model["xmu"]) / model["xsd"]
    return model["zmu"] + Xz @ model["w"]


def r2_from_pred(z, pred, baseline_mean):
    z = np.asarray(z, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    sse = float(np.sum((z - pred) ** 2))
    sst = float(np.sum((z - baseline_mean) ** 2))
    return 1.0 - sse / max(sst, EPS)


def fit_joint_group_scalar(Xi_tr, Xp_tr, ztr, ridge):
    Xi_tr = np.asarray(Xi_tr, dtype=np.float64)
    Xp_tr = np.asarray(Xp_tr, dtype=np.float64)
    ztr = np.asarray(ztr, dtype=np.float64).reshape(-1)

    imu, isd = Xi_tr.mean(axis=0), Xi_tr.std(axis=0)
    pmu, psd = Xp_tr.mean(axis=0), Xp_tr.std(axis=0)
    isd = np.where(isd < 1e-8, 1.0, isd)
    psd = np.where(psd < 1e-8, 1.0, psd)

    Iz = (Xi_tr - imu) / isd
    Pz = (Xp_tr - pmu) / psd
    X = np.concatenate([Iz, Pz], axis=1)

    zmu = float(np.mean(ztr))
    zc = ztr - zmu

    G, lam = ridge_lambda(X, ridge)
    A = G + lam * np.eye(X.shape[1])
    try:
        w = np.linalg.solve(A, X.T @ zc)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ X.T @ zc

    pi = Iz.shape[1]
    return {
        "imu": imu, "isd": isd, "pmu": pmu, "psd": psd,
        "zmu": zmu, "wi": w[:pi], "wp": w[pi:],
    }


def joint_group_contrib(model, Xi, Xp):
    Iz = (np.asarray(Xi, dtype=np.float64) - model["imu"]) / model["isd"]
    Pz = (np.asarray(Xp, dtype=np.float64) - model["pmu"]) / model["psd"]
    ai = Iz @ model["wi"]
    ap = Pz @ model["wp"]
    return ai, ap, model["zmu"] + ai + ap


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 2 or np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def vector_r2(Z, Pred, train_mean):
    Z = np.asarray(Z, dtype=np.float64)
    Pred = np.asarray(Pred, dtype=np.float64)
    sse = float(np.sum((Z - Pred) ** 2))
    sst = float(np.sum((Z - train_mean) ** 2))
    return 1.0 - sse / max(sst, EPS)


def mean_std(x):
    a = np.asarray(x, dtype=np.float64)
    if len(a) == 0 or np.all(~np.isfinite(a)):
        return float("nan"), float("nan")
    return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    args = parse_args()
    if not 0 <= args.shared_threshold <= 1:
        raise ValueError("--shared-threshold must be in [0,1]")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    state_dir = Path(args.state_dir)
    correct = load_npz(state_dir / "raw__correct__all_layers.npz")
    noimg = load_npz(state_dir / "raw__no_image__all_layers.npz")

    sids, states, relation, subject, reference, image_id, layers, layer_to_idx = \
        align_states(correct, noimg)

    if args.fixed_layer not in layer_to_idx:
        raise ValueError(f"L{args.fixed_layer} unavailable; layers={layers.tolist()}")
    li = layer_to_idx[args.fixed_layer]

    valid = np.asarray([r in RELATIONS for r in relation], dtype=bool)
    sids, relation = sids[valid], relation[valid]
    subject, reference = subject[valid], reference[valid]
    for k in list(states):
        states[k] = states[k][valid]

    bbox_rows = load_bbox_jsonl(Path(args.bbox_jsonl))
    df, audit = build_table(sids, relation, subject, reference, bbox_rows, args)
    if len(df) < 40:
        raise RuntimeError(f"Too few usable bbox samples: {len(df)}")

    sid_to_idx = {int(s): i for i, s in enumerate(sids.tolist())}
    ridx = np.asarray([sid_to_idx[int(s)] for s in df["sid"]], dtype=np.int64)
    y = relation[ridx]

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
        "A_raw": Araw, "A_noimg": Ano, "A_residual": Ares,
        "B_raw": Braw, "B_noimg": Bno, "B_residual": Bres,
        "last_raw": Lraw, "last_noimg": Lno, "last_residual": Lres,
        "Diff_raw": Araw - Braw,
        "Diff_noimg": Ano - Bno,
        "Diff_residual": Ares - Bres,
    }
    Y = target_map[args.target]
    D = Y.shape[1]

    Xs, _ = one_hot(df["subject"].tolist())
    Xr, _ = one_hot(df["reference"].tolist())
    Xi = np.concatenate([Xs, Xr], axis=1)

    pos_names = ["pair_cx", "pair_cy", "dx", "dy"]
    Xp = df[pos_names].to_numpy(dtype=np.float64)

    splits = stratified_splits(y, args.train_ratio, args.repeats, args.seed)

    direction_rows, coeff_rows, sample_rows, aggregate_rows = [], [], [], []

    print("=" * 130)
    print("SHARED-DIRECTION COEFFICIENT SUPERPOSITION TEST")
    print("=" * 130)
    print(
        f"target={args.target} | samples={len(df)} | layer=L{args.fixed_layer} | "
        f"hidden D={D} | repeats={args.repeats}"
    )
    print(
        f"identity shape={Xi.shape} | position shape={Xp.shape} | "
        f"shared threshold sigma>={args.shared_threshold:.2f}"
    )

    for rep, (tr, te) in enumerate(splits):
        Ytr, Yte = Y[tr], Y[te]
        Xi_tr, Xi_te = Xi[tr], Xi[te]
        Xp_tr, Xp_te = Xp[tr], Xp[te]

        mi = fit_ridge_matrix(Xi_tr, Ytr, args.ridge)
        mp = fit_ridge_matrix(Xp_tr, Ytr, args.ridge)

        Ui, _ = svd_basis(mi["Ctr"], args.subspace_energy, args.max_identity_rank)
        Up, _ = svd_basis(mp["Ctr"], args.subspace_energy, args.max_position_rank)
        sigma, Pi, Pp = principal_pairs(Ui, Up)

        selected = [j for j, s in enumerate(sigma.tolist())
                    if s >= args.shared_threshold]
        if not selected:
            raise RuntimeError(
                f"repeat {rep}: no principal direction passes "
                f"sigma>={args.shared_threshold}"
            )

        muY = Ytr.mean(axis=0, keepdims=True)
        Ztr_cols, Zte_cols, AI_cols, AP_cols, SUM_cols = [], [], [], [], []

        for j in range(len(sigma)):
            direction_rows.append({
                "repeat": rep,
                "principal_index": j + 1,
                "sigma": float(sigma[j]),
                "angle_deg": float(np.degrees(np.arccos(np.clip(sigma[j], 0, 1)))),
                "identity_rank": int(Ui.shape[1]),
                "position_rank": int(Up.shape[1]),
                "selected_as_shared": bool(j in selected),
            })

        for j in selected:
            u = midpoint_direction(Pi[:, j], Pp[:, j])
            if u is None:
                continue

            ztr = (Ytr - muY) @ u
            zte = (Yte - muY) @ u

            m_i = fit_scalar(Xi_tr, ztr, args.ridge)
            m_p = fit_scalar(Xp_tr, ztr, args.ridge)
            Xj_tr = np.concatenate([Xi_tr, Xp_tr], axis=1)
            Xj_te = np.concatenate([Xi_te, Xp_te], axis=1)
            m_j = fit_scalar(Xj_tr, ztr, args.ridge)

            pred_i = pred_scalar(m_i, Xi_te)
            pred_p = pred_scalar(m_p, Xp_te)
            pred_j = pred_scalar(m_j, Xj_te)

            zmu = float(np.mean(ztr))
            r2_i = r2_from_pred(zte, pred_i, zmu)
            r2_p = r2_from_pred(zte, pred_p, zmu)
            r2_j = r2_from_pred(zte, pred_j, zmu)

            mg = fit_joint_group_scalar(Xi_tr, Xp_tr, ztr, args.ridge)
            ai_te, ap_te, sum_te = joint_group_contrib(mg, Xi_te, Xp_te)

            std_z = max(float(np.std(zte)), EPS)
            pos_corrs = {
                name: corr(ap_te, Xp_te[:, k])
                for k, name in enumerate(pos_names)
            }

            coeff_rows.append({
                "repeat": rep,
                "principal_index": j + 1,
                "sigma": float(sigma[j]),
                "angle_deg": float(np.degrees(np.arccos(np.clip(sigma[j], 0, 1)))),
                "r2_identity": r2_i,
                "r2_position": r2_p,
                "r2_joint": r2_j,
                "unique_identity": r2_j - r2_p,
                "unique_position": r2_j - r2_i,
                "shared_commonality": r2_i + r2_p - r2_j,
                "joint_group_sum_r2": r2_from_pred(zte, sum_te, mg["zmu"]),
                "corr_z_identity_component": corr(zte, ai_te),
                "corr_z_position_component": corr(zte, ap_te),
                "corr_z_sum": corr(zte, sum_te),
                "corr_identity_position_components": corr(ai_te, ap_te),
                "std_identity_over_z": float(np.std(ai_te)) / std_z,
                "std_position_over_z": float(np.std(ap_te)) / std_z,
                "corr_position_component_pair_cx": pos_corrs["pair_cx"],
                "corr_position_component_pair_cy": pos_corrs["pair_cy"],
                "corr_position_component_dx": pos_corrs["dx"],
                "corr_position_component_dy": pos_corrs["dy"],
            })

            Ztr_cols.append(ztr)
            Zte_cols.append(zte)
            AI_cols.append(ai_te)
            AP_cols.append(ap_te)
            SUM_cols.append(sum_te)

            for q, sample_idx in enumerate(te.tolist()):
                sample_rows.append({
                    "repeat": rep,
                    "principal_index": j + 1,
                    "sid": int(df.iloc[sample_idx]["sid"]),
                    "relation": str(y[sample_idx]),
                    "subject": str(df.iloc[sample_idx]["subject"]),
                    "reference": str(df.iloc[sample_idx]["reference"]),
                    "z_actual": float(zte[q]),
                    "a_identity": float(ai_te[q]),
                    "a_position": float(ap_te[q]),
                    "z_pred_joint": float(sum_te[q]),
                    "pred_residual": float(zte[q] - sum_te[q]),
                })

        Ztr = np.stack(Ztr_cols, axis=1)
        Zte = np.stack(Zte_cols, axis=1)
        AI = np.stack(AI_cols, axis=1)
        AP = np.stack(AP_cols, axis=1)
        SUM = np.stack(SUM_cols, axis=1)

        miZ = fit_ridge_matrix(Xi_tr, Ztr, args.ridge)
        mpZ = fit_ridge_matrix(Xp_tr, Ztr, args.ridge)
        mjZ = fit_ridge_matrix(
            np.concatenate([Xi_tr, Xp_tr], axis=1), Ztr, args.ridge
        )
        pred_iZ = matrix_predict(miZ, Xi_te)
        pred_pZ = matrix_predict(mpZ, Xp_te)
        pred_jZ = matrix_predict(
            mjZ, np.concatenate([Xi_te, Xp_te], axis=1)
        )

        r2_iZ = vector_r2(Zte, pred_iZ, miZ["ymu"])
        r2_pZ = vector_r2(Zte, pred_pZ, mpZ["ymu"])
        r2_jZ = vector_r2(Zte, pred_jZ, mjZ["ymu"])

        aggregate_rows.append({
            "repeat": rep,
            "shared_rank": int(Ztr.shape[1]),
            "mean_sigma": float(np.mean([sigma[j] for j in selected])),
            "r2_identity": r2_iZ,
            "r2_position": r2_pZ,
            "r2_joint": r2_jZ,
            "unique_identity": r2_jZ - r2_pZ,
            "unique_position": r2_jZ - r2_iZ,
            "shared_commonality": r2_iZ + r2_pZ - r2_jZ,
            "joint_group_sum_r2": vector_r2(
                Zte, SUM, np.mean(Ztr, axis=0, keepdims=True)
            ),
            "identity_component_norm_over_z": float(np.linalg.norm(AI))
                / max(float(np.linalg.norm(Zte)), EPS),
            "position_component_norm_over_z": float(np.linalg.norm(AP))
                / max(float(np.linalg.norm(Zte)), EPS),
            "corr_flat_identity_position": corr(AI.ravel(), AP.ravel()),
            "corr_flat_actual_sum": corr(Zte.ravel(), SUM.ravel()),
        })

    direction_df = pd.DataFrame(direction_rows)
    coeff_df = pd.DataFrame(coeff_rows)
    sample_df = pd.DataFrame(sample_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)

    direction_df.to_csv(outdir / "principal_directions.csv", index=False)
    coeff_df.to_csv(outdir / "coefficient_decomposition_by_direction.csv", index=False)
    sample_df.to_csv(outdir / "heldout_sample_coefficients.csv", index=False)
    aggregate_df.to_csv(outdir / "aggregate_shared_coefficient_space.csv", index=False)

    with (outdir / "audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "target": args.target,
            "fixed_layer": args.fixed_layer,
            "usable_samples": len(df),
            "hidden_dim": D,
            "identity_dim": Xi.shape[1],
            "position_dim": Xp.shape[1],
            "shared_threshold": args.shared_threshold,
            "repeats": args.repeats,
            "train_ratio": args.train_ratio,
            "bbox_relation_sign_consistency": float(
                np.mean(df["bbox_relation_consistent"].to_numpy(dtype=bool))
            ),
        }, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 130)
    print("1) PRINCIPAL DIRECTIONS")
    print("=" * 130)
    for j in sorted(direction_df["principal_index"].unique()):
        d = direction_df[direction_df["principal_index"] == j]
        sm, ss = mean_std(d["sigma"])
        am, ast = mean_std(d["angle_deg"])
        sel = bool(np.mean(d["selected_as_shared"].astype(float)) >= 0.5)
        print(
            f"PC{int(j):02d} | sigma={sm:.4f}±{ss:.4f} | "
            f"angle={am:.2f}°±{ast:.2f}° | shared={sel}"
        )

    print()
    print("=" * 130)
    print("2) SAME-DIRECTION HELD-OUT COEFFICIENT DECOMPOSITION")
    print("=" * 130)
    for j in sorted(coeff_df["principal_index"].unique()):
        d = coeff_df[coeff_df["principal_index"] == j]
        sigma_m, _ = mean_std(d["sigma"])
        ri, ris = mean_std(d["r2_identity"])
        rp, rps = mean_std(d["r2_position"])
        rj, rjs = mean_std(d["r2_joint"])
        ui, _ = mean_std(d["unique_identity"])
        up, _ = mean_std(d["unique_position"])
        sh, _ = mean_std(d["shared_commonality"])
        rsum, _ = mean_std(d["joint_group_sum_r2"])
        csum, _ = mean_std(d["corr_z_sum"])
        cip, _ = mean_std(d["corr_identity_position_components"])
        si, _ = mean_std(d["std_identity_over_z"])
        sp, _ = mean_std(d["std_position_over_z"])

        print(f"\nPC{int(j):02d} | sigma={sigma_m:.4f}")
        print(
            f"  held-out R2: identity={ri:+.4f}±{ris:.4f} | "
            f"position={rp:+.4f}±{rps:.4f} | joint={rj:+.4f}±{rjs:.4f}"
        )
        print(
            f"  commonality: unique-I={ui:+.4f} | unique-P={up:+.4f} | "
            f"shared-info={sh:+.4f}"
        )
        print(
            f"  explicit sum: R2={rsum:+.4f} | corr(z,mean+aI+aP)={csum:+.4f}"
        )
        print(
            f"  parts: corr(aI,aP)={cip:+.4f} | "
            f"std(aI)/std(z)={si:.3f} | std(aP)/std(z)={sp:.3f}"
        )
        vals = []
        for name in pos_names:
            m, _ = mean_std(d[f"corr_position_component_{name}"])
            vals.append(f"{name}={m:+.3f}")
        print("  position signature: " + " | ".join(vals))

    print()
    print("=" * 130)
    print("3) AGGREGATE SHARED-COEFFICIENT SPACE")
    print("=" * 130)
    for col in [
        "shared_rank", "mean_sigma",
        "r2_identity", "r2_position", "r2_joint",
        "unique_identity", "unique_position", "shared_commonality",
        "joint_group_sum_r2",
        "identity_component_norm_over_z",
        "position_component_norm_over_z",
        "corr_flat_identity_position",
        "corr_flat_actual_sum",
    ]:
        m, s = mean_std(aggregate_df[col])
        print(f"{col:34s} = {m:+.4f}±{s:.4f}")

    print()
    print("=" * 130)
    print("INTERPRETATION")
    print("=" * 130)
    print(
        "Strong support for coefficient superposition requires:\n"
        "  1) high principal sigma;\n"
        "  2) positive held-out unique-I AND unique-P on the same z_j;\n"
        "  3) joint R2 larger than either identity-only or position-only;\n"
        "  4) explicit mean+a_identity+a_position predicts held-out z_j well;\n"
        "  5) corr(a_identity,a_position) is not ~±1 everywhere.\n"
        "If unique-P is near zero, high subspace overlap alone is not enough."
    )

    print()
    print("OUTPUTS:")
    for fn in [
        "principal_directions.csv",
        "coefficient_decomposition_by_direction.csv",
        "heldout_sample_coefficients.csv",
        "aggregate_shared_coefficient_space.csv",
        "summary.json",
        "audit.jsonl",
    ]:
        print(outdir / fn)


if __name__ == "__main__":
    main()
