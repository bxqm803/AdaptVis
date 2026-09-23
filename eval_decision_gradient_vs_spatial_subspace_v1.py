#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_decision_gradient_vs_spatial_subspace_v1.py

Quick diagnostic: are the strongest decision-sensitive directions at L25
mostly outside the H/V spatial subspace?

This reuses the exact causal-state objective/ranking infrastructure from
    eval_singlelayer_causal_token_zero_dropkeep_v1.py
and fits the H/V spatial subspace from Real-Gray object-pair states on the
calibration split:

    q = (h_sub^real - h_ref^real) - (h_sub^gray - h_ref^gray)

For each relation r, let
    d_r = unit(mu_r - mean_r mu_r)
and
    d_H = unit(d_right - d_left)
    d_V = unit(d_above - d_below).

We QR-orthonormalize [d_H,d_V] to obtain Q_sp. For every eligible text state
p at layer L, the causal script already computes

    g_p = d J_r / d h_{L,p}
    M_p = (h_real - h_gray)^T g_p.

This script adds the projection-energy ratio

    rho_grad = || Q_sp Q_sp^T g_p ||^2 / ||g_p||^2.

Interpretation:
  rho_grad ~ 0   : the local decision-sensitive direction is mostly outside
                   the H/V spatial subspace.
  rho_grad ~ 1   : it is mostly inside that subspace.

Important: low pairwise cosine to one spatial vector is NOT enough. This test
projects onto the full 2-D H/V span.

No generation is run. The script only performs state capture + one backward
pass per evaluation sample, so it is much faster than Drop/Keep or recovery.

Recommended quick run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_decision_gradient_vs_spatial_subspace_v1.py \
  --layer 25 \
  --ks 1,3,5,7 \
  --eval-scope test \
  --eval-max-samples 80 \
  --random-repeats 20 \
  --output-dir output/qwen3b_L25_grad_vs_spatial_n80_v1 \
  --overwrite

If you already have the writer file, add for speed:
  --writer-npz output/<writer_run>/learned_writers.npz

Outputs
=======
spatial_basis_geometry.csv
per_token_projection.csv
topk_projection_summary.csv
sample_topk_projection.csv
analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import eval_singlelayer_causal_token_zero_dropkeep_v1 as causal
except Exception as exc:
    raise SystemExit(
        "Could not import eval_singlelayer_causal_token_zero_dropkeep_v1.py.\n"
        "Put this script in the AdaptVis repo root and run it there.\n"
        f"{type(exc).__name__}: {exc}"
    )

REL = ("left", "right", "above", "below")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--layer", type=int, default=25)
    p.add_argument("--target-layers", default="31,32,33,34,35")
    p.add_argument("--writer-npz", default="")
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)

    p.add_argument(
        "--eligible-categories",
        default="subject,reference,relation_words,other_text",
    )
    p.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ks", default="1,3,5,7")
    p.add_argument("--random-repeats", type=int, default=20)

    p.add_argument(
        "--basis-max-per-relation",
        type=int,
        default=0,
        help="0 uses the whole calibration split; otherwise cap each relation.",
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-scope", default="test", choices=["all_data", "test"])
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all")

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return (v / n).astype(np.float32)


def tokenizer_of(processor):
    tok = getattr(processor, "tokenizer", None)
    return tok if tok is not None else processor


def find_object_positions(two, processor, batch, subject: str, reference: str):
    tok = tokenizer_of(processor)
    ids = batch["input_ids"][0].detach().cpu().tolist()
    sub = int(two.find_phrase_last_token(tok, ids, subject))
    ref = int(two.find_phrase_last_token(tok, ids, reference))
    if sub == ref:
        raise RuntimeError(f"subject/reference positions collide at {sub}")
    return sub, ref


def append_jsonl(path: Path, row: Mapping):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def stratified_basis_rows(train: Sequence[dict], cap_per_relation: int, seed: int):
    rows = list(train)
    if cap_per_relation <= 0:
        return rows
    rng = random.Random(seed)
    out = []
    for r in REL:
        z = [x for x in rows if str(x["gt"]) == r]
        rng.shuffle(z)
        out.extend(z[:cap_per_relation])
    return sorted(out, key=lambda x: int(x["sid"]))


def fit_spatial_basis(
    *, model, processor, decoder_layers, two, train, rec_by_sid,
    layer: int, device, gray_value: int, cap_per_relation: int, seed: int,
    error_path: Path,
):
    dyn = causal.dyn
    by_rel = defaultdict(list)
    basis_rows = stratified_basis_rows(train, cap_per_relation, seed)

    for m in tqdm(basis_rows, desc=f"FIT H/V spatial basis L{layer}"):
        sid = int(m["sid"])
        real = gray = None
        try:
            real = causal.base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")
            gray = dyn.make_gray_image(real, gray_value)

            rb = causal.base.make_question_batch(
                processor=processor,
                image=real,
                question_text=m["question_text"],
                device=device,
            )
            gb = causal.base.make_question_batch(
                processor=processor,
                image=gray,
                question_text=m["question_text"],
                device=device,
            )
            sub, ref = find_object_positions(
                two, processor, rb, m["subject"], m["reference"]
            )
            hr = dyn.capture_cpu(model, decoder_layers, rb, [layer])[layer][0]
            hg = dyn.capture_cpu(model, decoder_layers, gb, [layer])[layer][0]
            npos = min(hr.shape[0], hg.shape[0])
            if not (0 <= sub < npos and 0 <= ref < npos):
                raise RuntimeError(
                    f"object position out of bounds: sub={sub} ref={ref} n={npos}"
                )
            q = ((hr[sub] - hr[ref]) - (hg[sub] - hg[ref])).astype(np.float32)
            by_rel[str(m["gt"])].append(q)
        except Exception as exc:
            append_jsonl(error_path, {
                "stage": "fit_spatial_basis", "sid": sid,
                "type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        finally:
            with contextlib.suppress(Exception):
                if real is not None: real.close()
            with contextlib.suppress(Exception):
                if gray is not None: gray.close()

    missing = [r for r in REL if len(by_rel[r]) == 0]
    if missing:
        raise RuntimeError(f"No usable basis samples for relations: {missing}")

    means = {r: np.mean(np.stack(by_rel[r]), axis=0).astype(np.float32) for r in REL}
    global_mean = np.mean(np.stack([means[r] for r in REL]), axis=0).astype(np.float32)
    d = {r: unit(means[r] - global_mean) for r in REL}
    dH = unit(d["right"] - d["left"])
    dV = unit(d["above"] - d["below"])

    # Orthonormal basis spanning exactly the same H/V plane.
    A = np.stack([dH, dV], axis=1).astype(np.float64)  # [D,2]
    Q, _ = np.linalg.qr(A)
    Q = Q[:, :2].astype(np.float32)

    geom = pd.DataFrame([{
        "layer": int(layer),
        "N_left": len(by_rel["left"]),
        "N_right": len(by_rel["right"]),
        "N_above": len(by_rel["above"]),
        "N_below": len(by_rel["below"]),
        "dH_dot_dV_before_QR": float(np.dot(dH, dV)),
        "Q0_dot_Q1": float(np.dot(Q[:, 0], Q[:, 1])),
        "Q0_norm": float(np.linalg.norm(Q[:, 0])),
        "Q1_norm": float(np.linalg.norm(Q[:, 1])),
    }])
    return Q, geom


def projection_ratio(v: np.ndarray, Q: np.ndarray):
    v = np.asarray(v, dtype=np.float32)
    den = float(np.dot(v, v))
    if den <= EPS:
        return float("nan")
    coeff = Q.T @ v
    num = float(np.dot(coeff, coeff))
    return num / den


def summarize_topk(per_token: pd.DataFrame, ks: Sequence[int], random_repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    sample_rows = []

    for (sid, layer), g0 in per_token.groupby(["sid", "layer"]):
        g = g0.copy()
        rankable = g[g["mediation"] > 0].copy()
        rankable = rankable.sort_values("mediation", ascending=False)
        pool = g.copy()

        for k in ks:
            kk = min(int(k), len(rankable))
            if kk <= 0:
                continue
            top = rankable.head(kk)
            sample_rows.append({
                "sid": int(sid), "layer": int(layer), "k": int(k),
                "selection": "top", "repeat": 0, "n_selected": kk,
                "mean_rho_grad": float(top["rho_grad_spatial"].mean()),
                "median_rho_grad": float(top["rho_grad_spatial"].median()),
                "mean_grad_norm": float(top["grad_norm"].mean()),
                "mean_mediation": float(top["mediation"].mean()),
            })

            # Matched random text positions, independent of mediation rank.
            if len(pool) >= kk:
                idx = np.arange(len(pool))
                for rep in range(int(random_repeats)):
                    chosen = rng.choice(idx, size=kk, replace=False)
                    z = pool.iloc[chosen]
                    sample_rows.append({
                        "sid": int(sid), "layer": int(layer), "k": int(k),
                        "selection": "random", "repeat": int(rep), "n_selected": kk,
                        "mean_rho_grad": float(z["rho_grad_spatial"].mean()),
                        "median_rho_grad": float(z["rho_grad_spatial"].median()),
                        "mean_grad_norm": float(z["grad_norm"].mean()),
                        "mean_mediation": float(z["mediation"].mean()),
                    })

    sdf = pd.DataFrame(sample_rows)
    summary_rows = []
    if not sdf.empty:
        for (layer, k, selection), g in sdf.groupby(["layer", "k", "selection"]):
            # Random repeats are first averaged within sample so samples remain equal-weighted.
            z = g.groupby("sid", as_index=False).agg(
                mean_rho_grad=("mean_rho_grad", "mean"),
                median_rho_grad=("median_rho_grad", "mean"),
                mean_grad_norm=("mean_grad_norm", "mean"),
                mean_mediation=("mean_mediation", "mean"),
            )
            vals = z["mean_rho_grad"].to_numpy(float)
            summary_rows.append({
                "layer": int(layer), "k": int(k), "selection": str(selection),
                "N_samples": len(z),
                "mean_spatial_projection_ratio": float(np.nanmean(vals)),
                "median_spatial_projection_ratio": float(np.nanmedian(vals)),
                "p25_spatial_projection_ratio": float(np.nanpercentile(vals, 25)),
                "p75_spatial_projection_ratio": float(np.nanpercentile(vals, 75)),
                "mean_outside_ratio": float(1.0 - np.nanmean(vals)),
                "mean_grad_norm": float(z["mean_grad_norm"].mean()),
                "mean_mediation": float(z["mean_mediation"].mean()),
            })
    return sdf, pd.DataFrame(summary_rows)


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    L = int(a.layer)
    targets = causal.parse_ints(a.target_layers)
    ks = causal.parse_ints(a.ks)
    eligible_categories = causal.parse_set(a.eligible_categories)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    # causal.load_data expects these attributes.
    two, meta, train, heldout, test, rec_by_sid = causal.load_data(a)

    model = processor = None
    token_rows = []
    try:
        model, processor, decoder_layers, decoder_path, spec = causal.load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)
        for x in [L, L - 1] + targets:
            if not (0 <= x < n_layers):
                raise ValueError(f"L{x} invalid; model has L0..L{n_layers-1}")

        # Fit H/V plane from the calibration split only.
        Qsp, geom = fit_spatial_basis(
            model=model, processor=processor, decoder_layers=decoder_layers,
            two=two, train=train, rec_by_sid=rec_by_sid,
            layer=L, device=device, gray_value=a.gray_value,
            cap_per_relation=int(a.basis_max_per_relation), seed=a.seed,
            error_path=error_path,
        )
        geom.to_csv(outdir / "spatial_basis_geometry.csv", index=False)

        # Same relation-specific late-writer objective used in causal ranking.
        if a.writer_npz:
            writers = causal.load_writer_npz(Path(a.writer_npz), targets)
            writer_source = str(Path(a.writer_npz))
        else:
            writers, writer_geom = causal.calibrate_writers(
                model=model, processor=processor, decoder_layers=decoder_layers,
                train=train, rec_by_sid=rec_by_sid, targets=targets,
                device=device, gray_value=a.gray_value,
                writer_mode=a.writer_mode,
            )
            writer_source = "recalibrated"
            pd.DataFrame(writer_geom).to_csv(outdir / "writer_geometry.csv", index=False)
            np.savez_compressed(
                outdir / "learned_writers.npz",
                **{f"L{T}_{causal.DISPLAY[r]}": writers[T][r] for T in targets for r in REL},
            )

        print("\n" + "=" * 110)
        print("DECISION GRADIENT vs H/V SPATIAL SUBSPACE")
        print("=" * 110)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"layer=L{L}; writer targets={targets}")
        print(f"eval_scope={a.eval_scope}; N={len(test)}")
        print(f"writer source={writer_source}")
        print(f"eligible categories={eligible_categories}")
        print("rho = ||P_spatial g||^2 / ||g||^2")
        print("No generation is run.\n")

        capture_layers = [L - 1, L]
        for m in tqdm(test, desc=f"PROJECT gradients L{L}"):
            sid = int(m["sid"])
            gt = str(m["gt"])
            real = gray = cap = None
            try:
                real = causal.base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = causal.dyn.make_gray_image(real, a.gray_value)

                rb = causal.base.make_question_batch(
                    processor=processor, image=real,
                    question_text=m["question_text"], device=device,
                )
                gb = causal.base.make_question_batch(
                    processor=processor, image=gray,
                    question_text=m["question_text"], device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = causal.dyn.build_categories(
                    model, processor, rb, ids, m["subject"], m["reference"]
                )
                real_states = causal.dyn.capture_cpu(
                    model, decoder_layers, rb, capture_layers
                )
                gray_states = causal.dyn.capture_cpu(
                    model, decoder_layers, gb, [L]
                )

                graph_layers = sorted(set([L] + targets))
                with torch.enable_grad():
                    cap = causal.dyn.forward_graph(
                        model, decoder_layers, rb, graph_layers, L
                    )
                    terms = []
                    for T in targets:
                        s_hat = torch.as_tensor(
                            causal.normalize_np(writers[T][gt]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        terms.append(torch.dot(cap.states[T][0, -1].float(), s_hat))
                    objective = torch.stack(terms).sum()
                    grad = torch.autograd.grad(
                        objective, cap.states[L], retain_graph=False,
                        create_graph=False, allow_unused=False,
                    )[0].detach().float().cpu().numpy().astype(np.float32)
                cap.close(); cap = None

                H = real_states[L][0].astype(np.float32)
                Hprev = real_states[L - 1][0].astype(np.float32)
                Hg = gray_states[L][0].astype(np.float32)
                G = grad[0].astype(np.float32)
                npos = min(len(ids), len(cats), len(toks), H.shape[0], Hprev.shape[0], Hg.shape[0], G.shape[0])
                wanted = set(map(str, eligible_categories))

                for p in range(max(0, npos - 1)):
                    broad = causal.dyn.broad_category(cats[p])
                    if broad in {"visual", "last"}:
                        continue
                    if wanted and broad not in wanted:
                        continue
                    delta = (H[p] - Hg[p]).astype(np.float32)
                    upd = (H[p] - Hprev[p]).astype(np.float32)
                    g = G[p].astype(np.float32)
                    M = float(np.dot(delta, g))
                    if a.positive_only is False or M > 0:
                        rankable = True
                    else:
                        rankable = False
                    token_rows.append({
                        "sid": sid, "gt": causal.DISPLAY[gt], "layer": L,
                        "position": int(p), "token_id": int(ids[p]),
                        "token": str(toks[p]).replace("\n", "\\n"),
                        "category": str(cats[p]), "broad_category": str(broad),
                        "mediation": M,
                        "rankable_positive": bool(rankable),
                        "grad_norm": float(np.linalg.norm(g)),
                        "rho_grad_spatial": projection_ratio(g, Qsp),
                        "rho_real_gray_spatial": projection_ratio(delta, Qsp),
                        "rho_block_update_spatial": projection_ratio(upd, Qsp),
                        "grad_cos_Q0": float(np.dot(g, Qsp[:, 0]) / max(np.linalg.norm(g), EPS)),
                        "grad_cos_Q1": float(np.dot(g, Qsp[:, 1]) / max(np.linalg.norm(g), EPS)),
                    })

                del real_states, gray_states, grad
            except Exception as exc:
                append_jsonl(error_path, {
                    "stage": "eval", "sid": sid,
                    "type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}", flush=True)
            finally:
                if cap is not None:
                    with contextlib.suppress(Exception): cap.close()
                with contextlib.suppress(Exception):
                    if real is not None: real.close()
                with contextlib.suppress(Exception):
                    if gray is not None: gray.close()
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        df = pd.DataFrame(token_rows)
        if df.empty:
            raise RuntimeError("No token rows produced; inspect errors.jsonl")

        # Add within-sample mediation rank for convenience.
        df["mediation_rank"] = df.groupby("sid")["mediation"].rank(method="first", ascending=False)
        df.to_csv(outdir / "per_token_projection.csv", index=False)

        sdf, summary = summarize_topk(df, ks, a.random_repeats, a.seed + 999)
        sdf.to_csv(outdir / "sample_topk_projection.csv", index=False)
        summary.to_csv(outdir / "topk_projection_summary.csv", index=False)

        # Overall positive-M vs all eligible context.
        all_rho = df["rho_grad_spatial"].to_numpy(float)
        pos = df[df["mediation"] > 0]
        pos_rho = pos["rho_grad_spatial"].to_numpy(float)

        lines = []
        lines.append("=" * 110)
        lines.append("DECISION GRADIENT vs H/V SPATIAL SUBSPACE")
        lines.append("=" * 110)
        lines.append(f"model={a.model} layer=L{L} eval_N={df['sid'].nunique()}")
        lines.append(f"rho = ||P_spatial g||^2 / ||g||^2")
        lines.append(f"All eligible states: mean rho={np.nanmean(all_rho):.4f}, median={np.nanmedian(all_rho):.4f}")
        lines.append(f"Positive-M states:  mean rho={np.nanmean(pos_rho):.4f}, median={np.nanmedian(pos_rho):.4f}")
        lines.append("")
        if not summary.empty:
            for k in ks:
                z = summary[summary["k"] == int(k)]
                if z.empty: continue
                chunks = []
                for sel in ["top", "random"]:
                    q = z[z["selection"] == sel]
                    if q.empty: continue
                    r = q.iloc[0]
                    chunks.append(
                        f"{sel}: rho={r['mean_spatial_projection_ratio']:.4f} "
                        f"(outside={r['mean_outside_ratio']:.4f})"
                    )
                lines.append(f"K={k}: " + " | ".join(chunks))
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  rho is energy in the FULL 2-D H/V spatial span, not cosine to one axis.")
        lines.append("  Very small rho for Top-K supports 'decision-sensitive directions lie largely outside the spatial subspace'.")
        lines.append("  Do NOT call the subspaces orthogonal unless principal-angle/subspace analysis is also reported.")
        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(report, encoding="utf-8")

        (outdir / "metadata.json").write_text(json.dumps({
            "script": "eval_decision_gradient_vs_spatial_subspace_v1.py",
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "layer": L,
            "target_layers": targets,
            "eval_scope": a.eval_scope,
            "eval_N_requested": len(test),
            "writer_source": writer_source,
            "spatial_basis": "QR(span(d_right-d_left, d_above-d_below)) from calibration Real-Gray object-pair relation states",
            "rho_definition": "||Q_sp Q_sp^T g||^2 / ||g||^2",
            "ranking": "M=(h_real-h_gray)^T grad J_GT_writer",
            "ks": ks,
            "random_repeats": int(a.random_repeats),
        }, indent=2), encoding="utf-8")

    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
