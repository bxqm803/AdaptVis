#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_decision_gradient_vs_relation_span_multilayer_v1.py

Multi-layer diagnostic for the geometric relation between high-leverage
decision states and the spatial representation.

Differences from eval_decision_gradient_vs_spatial_subspace_v1.py
=================================================================
1) Evaluate multiple decoder layers in one run (default L20--L26).
2) Select Top-K token POSITIONS once at an anchor layer (default L25), using
   M = (h_real-h_gray)^T grad J, then evaluate the SAME token positions at
   every requested layer.
3) Do NOT build horizontal/vertical axes. At each layer L, construct

       d_{r,L} = unit(mu_{r,L} - mean_r mu_{r,L})

   for r in {left,right,above,below}, and define

       S_sp,L = span{d_left,L,d_right,L,d_above,L,d_below,L}.

   An SVD gives an orthonormal basis Q_sp,L for the actual numerical column
   span. No opposite-pair subtraction and no H/V axis construction is used.

For every eligible text state p at layer L,

    g_{i,L,p} = d J_r / d h_{i,L,p}

and

    rho_{i,L,p}
      = ||Q_sp,L Q_sp,L^T g_{i,L,p}||^2 / ||g_{i,L,p}||^2.

Recommended full-COCO run
=========================
CUDA_VISIBLE_DEVICES=0 python -u eval_decision_gradient_vs_relation_span_multilayer_v1.py \
  --layers 20-26 \
  --anchor-layer 25 \
  --ks 1,3,5,7 \
  --eval-scope all_data \
  --eval-max-samples 0 \
  --random-repeats 20 \
  --output-dir output/qwen3b_L20_26_grad_vs_relation_span_all440_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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


def parse_layer_spec(text: str) -> List[int]:
    vals: List[int] = []
    for raw in str(text).split(","):
        part = raw.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = [int(x.strip()) for x in part.split("-", 1)]
            vals.extend(range(min(a, b), max(a, b) + 1))
        else:
            vals.append(int(part))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError(f"No layers parsed from {text!r}")
    return vals


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
    p.add_argument("--layers", default="20-26")
    p.add_argument("--anchor-layer", type=int, default=25)
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
    p.add_argument("--basis-max-per-relation", type=int, default=0)
    p.add_argument("--span-rank-rtol", type=float, default=1e-6)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-scope", default="all_data", choices=["all_data", "test"])
    p.add_argument("--eval-max-samples", type=int, default=0, help="0 = all")
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


def orthonormal_span_from_relation_dirs(
    relation_dirs: Mapping[str, np.ndarray],
    rank_rtol: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    A = np.stack(
        [np.asarray(relation_dirs[r], dtype=np.float64) for r in REL], axis=1
    )
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if len(s) == 0 or float(s[0]) <= EPS:
        raise RuntimeError("Degenerate relation-direction matrix")
    rank = int(np.sum(s > float(rank_rtol) * float(s[0])))
    rank = max(1, rank)
    Q = U[:, :rank].astype(np.float32)
    return Q, s.astype(np.float64), rank


def fit_spatial_bases(
    *,
    model,
    processor,
    decoder_layers,
    two,
    train,
    rec_by_sid,
    layers: Sequence[int],
    device,
    gray_value: int,
    cap_per_relation: int,
    seed: int,
    rank_rtol: float,
    error_path: Path,
):
    dyn = causal.dyn
    layers = list(map(int, layers))
    by_layer_rel = {L: {r: [] for r in REL} for L in layers}
    basis_rows = stratified_basis_rows(train, cap_per_relation, seed)

    for m in tqdm(
        basis_rows,
        desc=f"FIT relation-span spatial bases L{layers[0]}-L{layers[-1]}",
    ):
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

            hr_all = dyn.capture_cpu(model, decoder_layers, rb, layers)
            hg_all = dyn.capture_cpu(model, decoder_layers, gb, layers)
            rel = str(m["gt"])

            for L in layers:
                hr = hr_all[L][0]
                hg = hg_all[L][0]
                npos = min(hr.shape[0], hg.shape[0])
                if not (0 <= sub < npos and 0 <= ref < npos):
                    raise RuntimeError(
                        f"L{L}: object position out of bounds: "
                        f"sub={sub} ref={ref} n={npos}"
                    )
                q = ((hr[sub] - hr[ref]) - (hg[sub] - hg[ref])).astype(np.float32)
                by_layer_rel[L][rel].append(q)

            del hr_all, hg_all

        except Exception as exc:
            append_jsonl(error_path, {
                "stage": "fit_spatial_bases",
                "sid": sid,
                "type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        finally:
            with contextlib.suppress(Exception):
                if real is not None:
                    real.close()
            with contextlib.suppress(Exception):
                if gray is not None:
                    gray.close()

    bases: Dict[int, np.ndarray] = {}
    geom_rows: List[dict] = []

    for L in layers:
        missing = [r for r in REL if len(by_layer_rel[L][r]) == 0]
        if missing:
            raise RuntimeError(f"L{L}: no usable basis samples for relations: {missing}")

        means = {
            r: np.mean(np.stack(by_layer_rel[L][r]), axis=0).astype(np.float32)
            for r in REL
        }
        global_mean = np.mean(
            np.stack([means[r] for r in REL]), axis=0
        ).astype(np.float32)
        relation_dirs = {r: unit(means[r] - global_mean) for r in REL}

        Q, s, rank = orthonormal_span_from_relation_dirs(
            relation_dirs, rank_rtol
        )
        bases[L] = Q
        gram = Q.T @ Q

        geom_rows.append({
            "layer": int(L),
            "N_left": len(by_layer_rel[L]["left"]),
            "N_right": len(by_layer_rel[L]["right"]),
            "N_above": len(by_layer_rel[L]["above"]),
            "N_below": len(by_layer_rel[L]["below"]),
            "span_rank": int(rank),
            "span_input_directions": 4,
            "sv0": float(s[0]) if len(s) > 0 else float("nan"),
            "sv1": float(s[1]) if len(s) > 1 else float("nan"),
            "sv2": float(s[2]) if len(s) > 2 else float("nan"),
            "sv3": float(s[3]) if len(s) > 3 else float("nan"),
            "orthonormal_check_max_abs": float(
                np.max(np.abs(gram - np.eye(rank)))
            ),
        })

    return bases, pd.DataFrame(geom_rows)


def projection_ratio(v: np.ndarray, Q: np.ndarray):
    v = np.asarray(v, dtype=np.float32)
    den = float(np.dot(v, v))
    if den <= EPS:
        return float("nan")
    coeff = Q.T @ v
    return float(np.dot(coeff, coeff)) / den


def summarize_anchor_selected_multilayer(
    per_token: pd.DataFrame,
    layers: Sequence[int],
    anchor_layer: int,
    ks: Sequence[int],
    random_repeats: int,
    seed: int,
    positive_only: bool,
):
    rng = np.random.default_rng(seed)
    sample_rows: List[dict] = []
    selection_rows: List[dict] = []
    layers = list(map(int, layers))

    for sid, sample_df in per_token.groupby("sid", sort=False):
        anchor = sample_df[sample_df["layer"] == int(anchor_layer)].copy()
        if anchor.empty:
            continue

        if positive_only:
            rankable = anchor[anchor["mediation"] > 0].copy()
        else:
            rankable = anchor.copy()
        rankable = rankable.sort_values("mediation", ascending=False)

        pool_positions = anchor["position"].astype(int).to_numpy()
        if len(pool_positions) == 0:
            continue

        layer_lookup = {
            int(L): sample_df[sample_df["layer"] == int(L)].set_index("position", drop=False)
            for L in layers
        }

        for k in ks:
            kk = min(int(k), len(rankable))
            if kk <= 0:
                continue

            top_positions = rankable.head(kk)["position"].astype(int).tolist()

            for rank, p in enumerate(top_positions, 1):
                ar = anchor[anchor["position"] == p].iloc[0]
                selection_rows.append({
                    "sid": int(sid),
                    "k": int(k),
                    "selection": "top",
                    "repeat": 0,
                    "anchor_layer": int(anchor_layer),
                    "anchor_rank": int(rank),
                    "position": int(p),
                    "token": str(ar["token"]),
                    "category": str(ar["category"]),
                    "anchor_mediation": float(ar["mediation"]),
                })

            for L in layers:
                lookup = layer_lookup[L]
                available = [p for p in top_positions if p in lookup.index]
                if not available:
                    continue
                z = lookup.loc[available]
                if isinstance(z, pd.Series):
                    z = z.to_frame().T
                sample_rows.append({
                    "sid": int(sid),
                    "layer": int(L),
                    "k": int(k),
                    "selection": "top",
                    "repeat": 0,
                    "n_selected": int(len(z)),
                    "mean_rho_grad": float(z["rho_grad_spatial"].astype(float).mean()),
                    "median_rho_grad": float(z["rho_grad_spatial"].astype(float).median()),
                    "mean_grad_norm": float(z["grad_norm"].astype(float).mean()),
                    "mean_mediation_at_layer": float(z["mediation"].astype(float).mean()),
                })

            if len(pool_positions) >= kk:
                for rep in range(int(random_repeats)):
                    random_positions = rng.choice(
                        pool_positions, size=kk, replace=False
                    ).astype(int).tolist()

                    for p in random_positions:
                        ar = anchor[anchor["position"] == p].iloc[0]
                        selection_rows.append({
                            "sid": int(sid),
                            "k": int(k),
                            "selection": "random",
                            "repeat": int(rep),
                            "anchor_layer": int(anchor_layer),
                            "anchor_rank": None,
                            "position": int(p),
                            "token": str(ar["token"]),
                            "category": str(ar["category"]),
                            "anchor_mediation": float(ar["mediation"]),
                        })

                    for L in layers:
                        lookup = layer_lookup[L]
                        available = [p for p in random_positions if p in lookup.index]
                        if not available:
                            continue
                        z = lookup.loc[available]
                        if isinstance(z, pd.Series):
                            z = z.to_frame().T
                        sample_rows.append({
                            "sid": int(sid),
                            "layer": int(L),
                            "k": int(k),
                            "selection": "random",
                            "repeat": int(rep),
                            "n_selected": int(len(z)),
                            "mean_rho_grad": float(z["rho_grad_spatial"].astype(float).mean()),
                            "median_rho_grad": float(z["rho_grad_spatial"].astype(float).median()),
                            "mean_grad_norm": float(z["grad_norm"].astype(float).mean()),
                            "mean_mediation_at_layer": float(z["mediation"].astype(float).mean()),
                        })

    sdf = pd.DataFrame(sample_rows)
    sel_df = pd.DataFrame(selection_rows)

    summary_rows: List[dict] = []
    if not sdf.empty:
        for (layer, k, selection), g in sdf.groupby(
            ["layer", "k", "selection"], sort=True
        ):
            z = g.groupby("sid", as_index=False).agg(
                mean_rho_grad=("mean_rho_grad", "mean"),
                median_rho_grad=("median_rho_grad", "mean"),
                mean_grad_norm=("mean_grad_norm", "mean"),
                mean_mediation_at_layer=("mean_mediation_at_layer", "mean"),
            )
            vals = z["mean_rho_grad"].to_numpy(float)
            summary_rows.append({
                "layer": int(layer),
                "k": int(k),
                "selection": str(selection),
                "N_samples": int(len(z)),
                "mean_spatial_projection_ratio": float(np.nanmean(vals)),
                "median_spatial_projection_ratio": float(np.nanmedian(vals)),
                "std_spatial_projection_ratio": float(np.nanstd(vals, ddof=1))
                    if len(vals) > 1 else float("nan"),
                "p25_spatial_projection_ratio": float(np.nanpercentile(vals, 25)),
                "p75_spatial_projection_ratio": float(np.nanpercentile(vals, 75)),
                "mean_outside_ratio": float(1.0 - np.nanmean(vals)),
                "mean_grad_norm": float(z["mean_grad_norm"].mean()),
                "mean_mediation_at_layer": float(z["mean_mediation_at_layer"].mean()),
            })

    return sdf, pd.DataFrame(summary_rows), sel_df


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    layers = parse_layer_spec(a.layers)
    anchor_layer = int(a.anchor_layer)
    if anchor_layer not in layers:
        raise ValueError(
            f"--anchor-layer L{anchor_layer} must be included in --layers {layers}"
        )

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

    two, meta, train, heldout, test, rec_by_sid = causal.load_data(a)

    model = processor = None
    token_rows: List[dict] = []

    try:
        model, processor, decoder_layers, decoder_path, spec = causal.load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        needed = sorted(set(layers + [L - 1 for L in layers] + targets))
        for x in needed:
            if not (0 <= x < n_layers):
                raise ValueError(
                    f"L{x} invalid; model has L0..L{n_layers-1}. "
                    f"Requested layers={layers}, writer targets={targets}"
                )

        Qsp_by_layer, geom = fit_spatial_bases(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            two=two,
            train=train,
            rec_by_sid=rec_by_sid,
            layers=layers,
            device=device,
            gray_value=a.gray_value,
            cap_per_relation=int(a.basis_max_per_relation),
            seed=a.seed,
            rank_rtol=float(a.span_rank_rtol),
            error_path=error_path,
        )
        geom.to_csv(outdir / "spatial_basis_geometry.csv", index=False)

        if a.writer_npz:
            writers = causal.load_writer_npz(Path(a.writer_npz), targets)
            writer_source = str(Path(a.writer_npz))
        else:
            writers, writer_geom = causal.calibrate_writers(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                train=train,
                rec_by_sid=rec_by_sid,
                targets=targets,
                device=device,
                gray_value=a.gray_value,
                writer_mode=a.writer_mode,
            )
            writer_source = "recalibrated"
            pd.DataFrame(writer_geom).to_csv(
                outdir / "writer_geometry.csv", index=False
            )
            np.savez_compressed(
                outdir / "learned_writers.npz",
                **{
                    f"L{T}_{causal.DISPLAY[r]}": writers[T][r]
                    for T in targets
                    for r in REL
                },
            )

        print("\n" + "=" * 120)
        print("DECISION GRADIENT vs RELATION-DIRECTION SPAN")
        print("=" * 120)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"layers={layers}; anchor=L{anchor_layer}; writer targets={targets}")
        print(f"eval_scope={a.eval_scope}; N={len(test)}")
        print(f"writer source={writer_source}")
        print(f"eligible categories={eligible_categories}")
        print("spatial basis = orth(span[d_left,d_right,d_above,d_below])")
        print(
            f"Top-K positions selected once at L{anchor_layer} "
            "and reused at all layers"
        )
        print("rho = ||P_spatial g||^2 / ||g||^2")
        print("No generation is run.\n")

        capture_real_layers = sorted(set(layers + [L - 1 for L in layers]))
        capture_gray_layers = list(layers)

        for m in tqdm(test, desc=f"PROJECT gradients L{layers[0]}-L{layers[-1]}"):
            sid = int(m["sid"])
            gt = str(m["gt"])
            real = gray = None
            rb = gb = None
            try:
                real = causal.base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = causal.dyn.make_gray_image(real, a.gray_value)

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

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = causal.dyn.build_categories(
                    model, processor, rb, ids, m["subject"], m["reference"]
                )
                real_states = causal.dyn.capture_cpu(
                    model, decoder_layers, rb, capture_real_layers
                )
                gray_states = causal.dyn.capture_cpu(
                    model, decoder_layers, gb, capture_gray_layers
                )
                wanted = set(map(str, eligible_categories))

                for L in layers:
                    cap = None
                    try:
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
                                terms.append(
                                    torch.dot(
                                        cap.states[T][0, -1].float(),
                                        s_hat,
                                    )
                                )
                            objective = torch.stack(terms).sum()
                            grad = torch.autograd.grad(
                                objective,
                                cap.states[L],
                                retain_graph=False,
                                create_graph=False,
                                allow_unused=False,
                            )[0].detach().float().cpu().numpy().astype(np.float32)

                        cap.close()
                        cap = None

                        H = real_states[L][0].astype(np.float32)
                        Hprev = real_states[L - 1][0].astype(np.float32)
                        Hg = gray_states[L][0].astype(np.float32)
                        G = grad[0].astype(np.float32)
                        Qsp = Qsp_by_layer[L]

                        npos = min(
                            len(ids), len(cats), len(toks),
                            H.shape[0], Hprev.shape[0],
                            Hg.shape[0], G.shape[0],
                        )

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
                            rankable = True if not a.positive_only else bool(M > 0)

                            token_rows.append({
                                "sid": sid,
                                "gt": causal.DISPLAY[gt],
                                "layer": int(L),
                                "anchor_layer": int(anchor_layer),
                                "position": int(p),
                                "token_id": int(ids[p]),
                                "token": str(toks[p]).replace("\n", "\\n"),
                                "category": str(cats[p]),
                                "broad_category": str(broad),
                                "mediation": M,
                                "rankable_positive": bool(rankable),
                                "grad_norm": float(np.linalg.norm(g)),
                                "spatial_span_rank": int(Qsp.shape[1]),
                                "rho_grad_spatial": projection_ratio(g, Qsp),
                                "rho_real_gray_spatial": projection_ratio(delta, Qsp),
                                "rho_block_update_spatial": projection_ratio(upd, Qsp),
                            })

                        del grad

                    except Exception as exc:
                        append_jsonl(error_path, {
                            "stage": "eval_layer",
                            "sid": sid,
                            "layer": int(L),
                            "type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                        print(
                            f"\n[ERROR sid={sid} L{L}] "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    finally:
                        if cap is not None:
                            with contextlib.suppress(Exception):
                                cap.close()

                del real_states, gray_states

            except Exception as exc:
                append_jsonl(error_path, {
                    "stage": "eval_sample",
                    "sid": sid,
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                with contextlib.suppress(Exception):
                    if real is not None:
                        real.close()
                with contextlib.suppress(Exception):
                    if gray is not None:
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        df = pd.DataFrame(token_rows)
        if df.empty:
            raise RuntimeError("No token rows produced; inspect errors.jsonl")

        anchor_df = df[df["layer"] == anchor_layer].copy()
        if a.positive_only:
            anchor_rankable = anchor_df[anchor_df["mediation"] > 0].copy()
        else:
            anchor_rankable = anchor_df.copy()

        anchor_rankable["anchor_mediation_rank"] = (
            anchor_rankable.groupby("sid")["mediation"]
            .rank(method="first", ascending=False)
        )
        rank_map = {
            (int(r.sid), int(r.position)): float(r.anchor_mediation_rank)
            for r in anchor_rankable.itertuples()
        }
        df["anchor_mediation_rank"] = [
            rank_map.get((int(sid), int(pos)), float("nan"))
            for sid, pos in zip(df["sid"], df["position"])
        ]
        df["layer_mediation_rank"] = (
            df.groupby(["sid", "layer"])["mediation"]
            .rank(method="first", ascending=False)
        )
        df.to_csv(outdir / "per_token_projection.csv", index=False)

        sdf, summary, sel_df = summarize_anchor_selected_multilayer(
            df,
            layers=layers,
            anchor_layer=anchor_layer,
            ks=ks,
            random_repeats=int(a.random_repeats),
            seed=int(a.seed) + 999,
            positive_only=bool(a.positive_only),
        )
        sel_df.to_csv(outdir / "anchor_selection.csv", index=False)
        sdf.to_csv(outdir / "sample_topk_projection.csv", index=False)
        summary.to_csv(outdir / "topk_projection_summary.csv", index=False)

        lines: List[str] = []
        lines.append("=" * 120)
        lines.append("DECISION GRADIENT vs RELATION-DIRECTION SPAN")
        lines.append("=" * 120)
        lines.append(
            f"model={a.model} layers={layers} anchor=L{anchor_layer} "
            f"eval_N={df['sid'].nunique()}"
        )
        lines.append(
            "spatial basis = orth(span[d_left,d_right,d_above,d_below]) at each layer"
        )
        lines.append(
            f"Top-K positions selected at L{anchor_layer} and reused at all layers"
        )
        lines.append("rho = ||P_spatial g||^2 / ||g||^2")
        lines.append("")

        if not summary.empty:
            for k in ks:
                lines.append(f"K={k}")
                zk = summary[summary["k"] == int(k)]
                for L in layers:
                    zL = zk[zk["layer"] == int(L)]
                    chunks = []
                    for sel in ["top", "random"]:
                        q = zL[zL["selection"] == sel]
                        if q.empty:
                            continue
                        r = q.iloc[0]
                        chunks.append(
                            f"{sel}: rho={r['mean_spatial_projection_ratio']:.4f} "
                            f"(outside={r['mean_outside_ratio']:.4f}, "
                            f"N={int(r['N_samples'])})"
                        )
                    if chunks:
                        lines.append(f"  L{L}: " + " | ".join(chunks))
                lines.append("")

        lines.append("Interpretation:")
        lines.append(
            "  rho is energy in the span of ALL four relation directions, "
            "not cosine to one direction and not an H/V axis plane."
        )
        lines.append(
            "  Small rho for L25-selected Top-K positions means the local "
            "decision-sensitive direction is not confined to the identified "
            "relation-defined spatial subspace."
        )
        lines.append(
            "  The same token positions are evaluated at every layer."
        )
        lines.append(
            "  Do not call the spaces orthogonal solely from low rho."
        )

        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(report, encoding="utf-8")

        (outdir / "metadata.json").write_text(
            json.dumps({
                "script": "eval_decision_gradient_vs_relation_span_multilayer_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "layers": layers,
                "anchor_layer": anchor_layer,
                "target_layers": targets,
                "eval_scope": a.eval_scope,
                "eval_N_requested": len(test),
                "writer_source": writer_source,
                "spatial_basis": (
                    "SVD orthonormal basis of span("
                    "unit(mu_left-mu), unit(mu_right-mu), "
                    "unit(mu_above-mu), unit(mu_below-mu))"
                ),
                "axis_pairing": False,
                "span_rank_rtol": float(a.span_rank_rtol),
                "rho_definition": "||Q_sp Q_sp^T g||^2 / ||g||^2",
                "ranking": (
                    f"Top-K token positions selected only at L{anchor_layer} by "
                    "M=(h_real-h_gray)^T grad J_GT_writer; same positions reused "
                    "for every requested layer"
                ),
                "ks": ks,
                "random_repeats": int(a.random_repeats),
            }, indent=2),
            encoding="utf-8",
        )

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
