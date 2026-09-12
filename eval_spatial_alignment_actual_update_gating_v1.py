#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spatial_alignment_actual_update_gating_v1.py

Replace the GT-defined sign of actual causal-token updates with a continuous,
source-only spatial-alignment objective.

NO relation argmax is used for routing.
NO COCO GT is used to define the update sign.

Synthetic source defines, at each read layer L, two continuous axes:

    dH_L = unit(mu_right - mu_left)
    dV_L = unit(mu_above - mu_below)

For a COCO sample relation state x_L = (h_sub-h_ref)_REAL-NoImage,
source-calibrated natural coordinates are

    zH_L = <x_L-center_L, dH_L> / half_gap_H_L
    zV_L = <x_L-center_L, dV_L> / half_gap_V_L

and the sample readout is the mean across read layers:

    z = (zH, zV).

No discretization to left/right/above/below occurs.

The downstream continuous decision coordinates are

    yH = S_right - S_left
    yV = S_above - S_below

where S_r is the teacher-forced sequence score of answer r.
Define

    J_align = zH * yH + zV * yV.

For each actual REAL causal-token block update

    a_{L,p} = h_REAL[L,p] - h_REAL[L-1,p],

compute

    B_align[L,p] = <a_{L,p}, d J_align / d h_REAL[L,p]>.

Generation interventions:

    align_positive:
        B_align > tau  -> +alpha * a

    align_negative_cancel:
        B_align < -tau -> -alpha * a

    align_signed:
        positive -> +alpha*a; negative -> -alpha*a

    all_amplify:
        +alpha*a regardless of sign (matched control)

Important caveat:
The causal-token POSITIONS are still supplied by --ranked-causal. If that file
comes from the oracle K36 causal scan, position selection remains oracle. This
script isolates only the replacement of the GT decision sign by model-internal
continuous spatial information.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import eval_real_causal_token_update_gating_v1 as gate

REL = ("left", "right", "above", "below")
EPS = 1e-12


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v)
    return v / n


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

    p.add_argument("--ranked-causal", required=True)
    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument("--causal-categories", default="")
    p.add_argument("--update-layers", default="8-26")
    p.add_argument("--exclude-target-layer", action="store_true")

    p.add_argument(
        "--source-spatial-npz",
        required=True,
        help="Synthetic-400 spatial cache; relation_vectors or img/no_image.",
    )
    p.add_argument(
        "--target-spatial-npz",
        required=True,
        help="COCO original-prompt spatial cache; relation_vectors or img/no_image.",
    )
    p.add_argument(
        "--read-layers",
        default="20-26",
        help="Layers whose continuous H/V coordinates are averaged into z.",
    )
    p.add_argument(
        "--z-normalization",
        default="unit",
        choices=["unit", "none"],
        help="unit preserves H/V ratio but removes overall readout magnitude.",
    )
    p.add_argument(
        "--z-clip",
        type=float,
        default=3.0,
        help="Clip each per-layer natural coordinate before layer averaging; <=0 disables.",
    )

    p.add_argument("--scales", default="0.1,0.25,0.5")
    p.add_argument("--alignment-threshold", type=float, default=0.0)
    p.add_argument(
        "--conditions",
        default="align_positive,align_negative_cancel,align_signed,all_amplify",
    )

    p.add_argument(
        "--answer-surface", default="above_below", choices=["above_below", "on_under"]
    )
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--answer-suffix", default="")
    p.add_argument(
        "--sequence-score-reduction", default="mean", choices=["mean", "sum"]
    )
    p.add_argument("--max-new-tokens", type=int, default=6)

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
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


def load_spatial_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        req = {"sample_index", "relation", "decoder_block_index"}
        missing = req - set(z.files)
        if missing:
            raise RuntimeError(f"{path} missing keys {sorted(missing)}; has={sorted(z.files)}")

        if "relation_vectors" in z.files:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            mode = "relation_vectors"
        elif "img" in z.files and "no_image" in z.files:
            img = np.asarray(z["img"], dtype=np.float32)
            no = np.asarray(z["no_image"], dtype=np.float32)
            if img.shape != no.shape:
                raise RuntimeError(f"img/no_image mismatch {img.shape} vs {no.shape}")
            X = (img - no).astype(np.float32)
            mode = "img_minus_no_image"
        else:
            raise RuntimeError(f"{path}: need relation_vectors or img+no_image")

        y = np.asarray([gate.traj.normalize_relation(gate.base, v) for v in z["relation"].tolist()], dtype=object)
        layers = [int(v) for v in z["decoder_block_index"].tolist()]
        sids = np.asarray(z["sample_index"], dtype=np.int64)

        meta = {}
        for k in ("prompt_template", "pool", "vector_definition", "model"):
            if k in z.files:
                try:
                    meta[k] = str(z[k].item())
                except Exception:
                    meta[k] = str(z[k])

    if X.ndim != 3:
        raise RuntimeError(f"Expected [N,L,D], got {X.shape}")
    if X.shape[0] != len(y) or X.shape[0] != len(sids):
        raise RuntimeError(f"Bad cache shapes X={X.shape}, y={y.shape}, sids={sids.shape}")
    return X, y, layers, sids, mode, meta


def fit_continuous_axes(Xs, ys, cache_layers, read_layers):
    l2i = {int(L): i for i, L in enumerate(cache_layers)}
    missing = [L for L in read_layers if L not in l2i]
    if missing:
        raise RuntimeError(f"Source cache missing read layers {missing}")

    axes = {}
    rows = []
    for L in read_layers:
        li = l2i[L]
        X = Xs[:, li].astype(np.float64)
        center = X.mean(axis=0)
        means = {}
        for r in REL:
            mask = ys == r
            if not np.any(mask):
                raise RuntimeError(f"No source examples for relation={r}")
            means[r] = X[mask].mean(axis=0)

        dH = unit(means["right"] - means["left"])
        dV = unit(means["above"] - means["below"])
        gapH = float(np.dot(means["right"] - means["left"], dH))
        gapV = float(np.dot(means["above"] - means["below"], dV))
        if gapH <= EPS or gapV <= EPS:
            raise RuntimeError(f"Degenerate source axis at L{L}: gapH={gapH}, gapV={gapV}")

        halfH = gapH / 2.0
        halfV = gapV / 2.0
        axes[L] = {
            "layer_index": li,
            "center": center.astype(np.float32),
            "dH": dH.astype(np.float32),
            "dV": dV.astype(np.float32),
            "halfH": float(halfH),
            "halfV": float(halfV),
        }
        rows.append({
            "read_layer": L,
            "source_N": len(X),
            "dH_dot_dV": float(np.dot(dH, dV)),
            "full_gap_H": gapH,
            "full_gap_V": gapV,
            "half_gap_H": halfH,
            "half_gap_V": halfV,
        })
    return axes, pd.DataFrame(rows)


def continuous_spatial_readout(Xrow, axes, read_layers, z_clip, normalization):
    per_layer = []
    for L in read_layers:
        a = axes[L]
        v = Xrow[int(a["layer_index"])].astype(np.float64) - a["center"].astype(np.float64)
        zH = float(np.dot(v, a["dH"])) / max(float(a["halfH"]), EPS)
        zV = float(np.dot(v, a["dV"])) / max(float(a["halfV"]), EPS)
        if z_clip > 0:
            zH = float(np.clip(zH, -z_clip, z_clip))
            zV = float(np.clip(zV, -z_clip, z_clip))
        per_layer.append((L, zH, zV))

    zH = float(np.mean([x[1] for x in per_layer]))
    zV = float(np.mean([x[2] for x in per_layer]))
    raw_norm = float(math.sqrt(zH * zH + zV * zV))

    if normalization == "unit" and raw_norm > EPS:
        zH /= raw_norm
        zV /= raw_norm

    return {
        "zH": zH,
        "zV": zV,
        "raw_norm": raw_norm,
        "per_layer": per_layer,
    }


def alignment_objective(scores, zH, zV):
    yH = float(scores["right"] - scores["left"])
    yV = float(scores["above"] - scores["below"])
    J = float(zH * yH + zV * yV)
    return J, yH, yV


def score_and_four_grads(model, decoder_layers, batch, candidate_ids, reduction, grad_layers):
    scores = {}
    grads = {}
    for r in REL:
        s, g = gate.sequence_score_and_grads(
            model=model,
            decoder_layers=decoder_layers,
            batch=batch,
            answer_ids=candidate_ids[r],
            reduction=reduction,
            grad_layers=grad_layers,
        )
        scores[r] = float(s)
        grads[r] = g
    return scores, grads


def attach_alignment_scores(entries, grads, zH, zV):
    out = []
    for e in entries:
        L = int(e["update_layer"])
        p = int(e["real_position"])
        gs = {r: grads[r].get(L, None) for r in REL}
        if any(gs[r] is None for r in REL):
            continue
        if any(not (0 <= p < gs[r].shape[1]) for r in REL):
            continue

        gH = (gs["right"][0, p] - gs["left"][0, p]).astype(np.float32)
        gV = (gs["above"][0, p] - gs["below"][0, p]).astype(np.float32)
        g = (float(zH) * gH + float(zV) * gV).astype(np.float32)
        a = np.asarray(e["_real_update"], dtype=np.float32)

        BH = float(np.dot(a, gH))
        BV = float(np.dot(a, gV))
        B = float(np.dot(a, g))
        an = float(np.linalg.norm(a))
        gn = float(np.linalg.norm(g))

        q = dict(e)
        q.update({
            "zH": float(zH),
            "zV": float(zV),
            "update_effect_H": BH,
            "update_effect_V": BV,
            "alignment_update_score": B,
            "alignment_grad_norm": gn,
            "alignment_update_score_per_update_norm": B / max(an, EPS),
            "alignment_update_score_per_grad_norm": B / max(gn, EPS),
            "alignment_update_cosine": gate.cosine_np(a, g),
            "_alignment_grad": g,
        })
        out.append(q)
    return out


def build_patch(entries, condition, scale, threshold):
    pmap = {}
    counts = {"positive": 0, "negative": 0, "neutral": 0, "patched": 0}
    for e in entries:
        B = float(e["alignment_update_score"])
        u = np.asarray(e["_real_update"], dtype=np.float32)
        vec = None

        if B > threshold:
            counts["positive"] += 1
        elif B < -threshold:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

        if condition == "align_positive":
            if B > threshold:
                vec = float(scale) * u
        elif condition == "align_negative_cancel":
            if B < -threshold:
                vec = -float(scale) * u
        elif condition == "align_signed":
            if B > threshold:
                vec = float(scale) * u
            elif B < -threshold:
                vec = -float(scale) * u
        elif condition == "all_amplify":
            vec = float(scale) * u
        else:
            raise ValueError(condition)

        if vec is not None:
            gate.add_patch(pmap, e["update_layer"], e["real_position"], vec)
            counts["patched"] += 1
    return pmap, counts


def summarize_alignment_updates(df):
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    for cohort, g in [("all", df), ("baseline_wrong", df[~df["baseline_correct"].astype(bool)]), ("baseline_correct", df[df["baseline_correct"].astype(bool)])]:
        if len(g) == 0:
            continue
        B = pd.to_numeric(g["alignment_update_score"], errors="coerce").to_numpy(float)
        rows.append({
            "cohort": cohort,
            "N_updates": len(g),
            "positive_fraction": float(np.mean(B > 0)),
            "negative_fraction": float(np.mean(B < 0)),
            "mean_B_align": float(np.nanmean(B)),
            "mean_abs_B_align": float(np.nanmean(np.abs(B))),
            "median_B_align": float(np.nanmedian(B)),
        })
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers = gate.parse_layers(a.causal_layers)
    update_layers = gate.parse_layers(a.update_layers)
    read_layers = gate.parse_layers(a.read_layers)
    scales = gate.parse_floats(a.scales)
    categories = gate.parse_set(a.causal_categories)
    conditions = gate.parse_set(a.conditions)
    valid = {"align_positive", "align_negative_cancel", "align_signed", "all_amplify"}
    bad = set(conditions) - valid
    if bad:
        raise ValueError(f"Unknown conditions {sorted(bad)}")
    if any(L < 1 for L in update_layers):
        raise ValueError("--update-layers must be >=1")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    err_path = outdir / "errors.jsonl"

    Xs, ys, src_cache_layers, src_sids, src_mode, src_meta = load_spatial_cache(Path(a.source_spatial_npz))
    Xt, yt, tgt_cache_layers, tgt_sids, tgt_mode, tgt_meta = load_spatial_cache(Path(a.target_spatial_npz))
    if src_cache_layers != tgt_cache_layers:
        raise RuntimeError("Source/target decoder_block_index mismatch")
    if Xs.shape[2] != Xt.shape[2]:
        raise RuntimeError(f"Hidden dim mismatch source={Xs.shape[2]} target={Xt.shape[2]}")

    axes, axis_df = fit_continuous_axes(Xs, ys, src_cache_layers, read_layers)
    axis_df.to_csv(outdir / "synthetic_continuous_axes.csv", index=False)
    sid_to_tgt = {int(sid): i for i, sid in enumerate(tgt_sids.tolist())}

    two, meta, rec_by_sid = gate.load_data(a)
    meta = [m for m in meta if int(m["sid"]) in sid_to_tgt]
    allowed_sids = {int(m["sid"]) for m in meta}
    causal_sel = gate.load_causal_selection(
        Path(a.ranked_causal), allowed_sids, causal_layers, a.causal_top_k, categories
    )
    causal_sel.to_csv(outdir / "selected_causal_states.csv", index=False)
    causal_by_sid = {int(sid): g.sort_values("rank").copy() for sid, g in causal_sel.groupby("sid")}

    model = processor = None
    generation_rows = []
    sample_rows = []
    update_rows = []
    z_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = gate.load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)
        capture_layers = sorted(set(update_layers + [L - 1 for L in update_layers]))
        outside = [L for L in capture_layers if not 0 <= L < n_layers]
        if outside:
            raise RuntimeError(f"Capture layers outside model: {outside}")

        texts = gate.candidate_texts(a)
        candidate_ids = gate.encode_candidate_ids(processor, texts)

        print("=" * 180)
        print("CONTINUOUS SPATIAL ALIGNMENT -> ACTUAL UPDATE SIGN")
        print("=" * 180)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(meta)} sourceN={len(Xs)} target-cacheN={len(Xt)}")
        print(f"read_layers={read_layers} z_norm={a.z_normalization} z_clip={a.z_clip}")
        print(f"causal_layers={causal_layers} topK={a.causal_top_k} update_layers={update_layers}")
        print(f"scales={scales} threshold={a.alignment_threshold}")
        print(f"conditions={conditions}")
        print("NO relation argmax is used for routing/sign.")
        print("NO COCO GT is used to define J_align or B_align.")
        print("CAUTION: causal positions may still be oracle if --ranked-causal is oracle-ranked.")
        print("=" * 180, flush=True)

        for m in tqdm(meta, desc="SPATIAL-ALIGN UPDATE GATING"):
            sid = int(m["sid"])
            if sid not in causal_by_sid:
                continue
            image = None
            batch = None
            try:
                ti = sid_to_tgt[sid]
                cache_gt = str(yt[ti])
                if cache_gt in REL and cache_gt != str(m["gt"]):
                    raise RuntimeError(f"GT mismatch metadata={m['gt']} target-cache={cache_gt}")

                z = continuous_spatial_readout(
                    Xt[ti], axes, read_layers, a.z_clip, a.z_normalization
                )
                zH, zV = float(z["zH"]), float(z["zV"])
                if abs(zH) + abs(zV) < 1e-10:
                    raise RuntimeError("Degenerate zero continuous spatial readout")

                zrow = {
                    "sid": sid,
                    "gt": m["gt"],
                    "zH": zH,
                    "zV": zV,
                    "z_raw_norm": float(z["raw_norm"]),
                }
                for L, h, v in z["per_layer"]:
                    zrow[f"zH_L{L}"] = h
                    zrow[f"zV_L{L}"] = v
                z_rows.append(zrow)

                image = gate.base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = gate.base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )

                real_states = gate.capture_prompt_blocks(
                    model, decoder_layers, batch, capture_layers
                )

                base_text = gate.base.generate_text(
                    model, processor, batch, max_new_tokens=a.max_new_tokens
                )
                base_pred = gate.traj.normalize_relation(gate.base, base_text)
                base_correct = base_pred == m["gt"]
                generation_rows.append({
                    "sid": sid, "gt": m["gt"], "condition": "baseline", "scale": 0.0,
                    "prediction": base_pred, "correct": base_correct,
                    "n_patched_updates": 0, "n_positive_updates": 0,
                    "n_negative_updates": 0, "text": base_text,
                })

                clean_scores = gate.all_sequence_scores(
                    model, batch, candidate_ids, a.sequence_score_reduction
                )
                J, yH, yV = alignment_objective(clean_scores, zH, zV)
                seq_pred = max(REL, key=lambda r: clean_scores[r])

                specs = gate.causal_position_specs(causal_by_sid[sid])
                entries = gate.build_real_updates(
                    sid=sid,
                    gt=m["gt"],
                    baseline_correct=base_correct,
                    specs=specs,
                    r2n={},
                    real_states=real_states,
                    no_states={},
                    update_layers=update_layers,
                    exclude_target_layer=a.exclude_target_layer,
                )
                if not entries:
                    raise RuntimeError("No actual REAL updates")

                grad_layers = sorted(set(int(e["update_layer"]) for e in entries))
                graph_scores, grads = score_and_four_grads(
                    model, decoder_layers, batch, candidate_ids,
                    a.sequence_score_reduction, grad_layers
                )
                replay_err = max(abs(graph_scores[r] - clean_scores[r]) for r in REL)
                if replay_err > 5e-3:
                    raise RuntimeError(f"Sequence score replay mismatch maxerr={replay_err:.6f}")

                scored = attach_alignment_scores(entries, grads, zH, zV)
                if not scored:
                    raise RuntimeError("No updates received alignment gradient")

                B = np.asarray([float(e["alignment_update_score"]) for e in scored], dtype=np.float64)
                pos_mass = float(np.maximum(B, 0).sum())
                neg_mass = float(np.maximum(-B, 0).sum())
                sample_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "baseline_prediction": base_pred,
                    "baseline_correct": base_correct,
                    "sequence_prediction": seq_pred,
                    "zH": zH,
                    "zV": zV,
                    "decision_yH": yH,
                    "decision_yV": yV,
                    "J_align": J,
                    "N_updates": len(scored),
                    "positive_fraction": float(np.mean(B > 0)),
                    "negative_fraction": float(np.mean(B < 0)),
                    "positive_mass": pos_mass,
                    "negative_mass": neg_mass,
                    "net_alignment_update_mass": float(B.sum()),
                    "mean_abs_alignment_update_score": float(np.mean(np.abs(B))),
                    **{f"score_{r}": float(clean_scores[r]) for r in REL},
                })

                for e in scored:
                    update_rows.append({
                        k: v for k, v in e.items()
                        if k not in ("_real_update", "_noimage_update", "_rn_update", "_alignment_grad")
                    })

                for cond in conditions:
                    for scale in scales:
                        pmap, counts = build_patch(
                            scored, cond, scale, a.alignment_threshold
                        )
                        pred, text = gate.generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=batch,
                            patch_map=pmap,
                            max_new_tokens=a.max_new_tokens,
                        )
                        generation_rows.append({
                            "sid": sid,
                            "gt": m["gt"],
                            "condition": cond,
                            "scale": float(scale),
                            "prediction": pred,
                            "correct": pred == m["gt"],
                            "n_patched_updates": int(counts["patched"]),
                            "n_positive_updates": int(counts["positive"]),
                            "n_negative_updates": int(counts["negative"]),
                            "text": text,
                        })

            except Exception as exc:
                gate.append_jsonl(err_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                print(f"\n[ERROR sid={sid}] {type(exc).__name__}: {exc}", flush=True)
            finally:
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gen_df = pd.DataFrame(generation_rows)
        sample_df = pd.DataFrame(sample_rows)
        upd_df = pd.DataFrame(update_rows)
        z_df = pd.DataFrame(z_rows)

        gen_df.to_csv(outdir / "generation_per_sample.csv", index=False)
        sample_df.to_csv(outdir / "sample_alignment_summary.csv", index=False)
        upd_df.to_csv(outdir / "per_real_update_spatial_alignment_score.csv", index=False)
        z_df.to_csv(outdir / "continuous_spatial_readout.csv", index=False)

        gen_summary = gate.summarize_generation(gen_df)
        gen_by_rel = gate.summarize_generation_by_relation(gen_df)
        upd_summary = summarize_alignment_updates(upd_df)
        gen_summary.to_csv(outdir / "generation_summary.csv", index=False)
        gen_by_rel.to_csv(outdir / "generation_by_relation.csv", index=False)
        upd_summary.to_csv(outdir / "alignment_update_summary.csv", index=False)

        baseline = gen_df[gen_df["condition"] == "baseline"].drop_duplicates("sid")
        baseline_acc = float(baseline["correct"].astype(bool).mean()) if len(baseline) else float("nan")

        print("\n" + "=" * 180)
        print("CONTINUOUS SPATIAL-ALIGNMENT SIGN: GENERATION")
        print("=" * 180)
        print(f"baseline_accuracy={baseline_acc:.4f} N={len(baseline)}")
        print(gen_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(gen_summary) else "EMPTY")
        print("\nALIGNMENT UPDATE SIGN DISTRIBUTION")
        print("-" * 180)
        print(upd_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(upd_summary) else "EMPTY")
        print("\nBY RELATION")
        print("-" * 180)
        print(gen_by_rel.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(gen_by_rel) else "EMPTY")

        metadata = {
            "script": "eval_spatial_alignment_actual_update_gating_v1.py",
            "model": a.model,
            "repo": spec.repo_id,
            "decoder_path": decoder_path,
            "source_spatial_npz": a.source_spatial_npz,
            "target_spatial_npz": a.target_spatial_npz,
            "source_cache_mode": src_mode,
            "target_cache_mode": tgt_mode,
            "source_cache_meta": src_meta,
            "target_cache_meta": tgt_meta,
            "read_layers": read_layers,
            "causal_layers": causal_layers,
            "causal_top_k": a.causal_top_k,
            "update_layers": update_layers,
            "z_normalization": a.z_normalization,
            "z_clip": a.z_clip,
            "alignment_objective": "zH*(S_right-S_left)+zV*(S_above-S_below)",
            "uses_coco_gt_for_sign": False,
            "uses_relation_argmax_for_sign": False,
            "causal_position_oracle_caveat": True,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
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
