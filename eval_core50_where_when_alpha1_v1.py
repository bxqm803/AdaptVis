#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare WHERE vs WHEN for Core50 causal repair, with alpha fixed to 1.0.

This is a focused follow-up to eval_self_selected_core_behavior_v1.py.

Axis A: POSITION selector quality
---------------------------------
Self-select positions, but give each selected position its ORACLE best layer
(max oracle mediation across L20..L26).  This asks:

    "If layer assignment were solved, which relation-free position selector
     produces the most behaviorally useful positions?"

Tested:
  - attn_delta_max
  - attn_delta_top2
  - attn_delta_mean
  - attn_posdelta_max
  - attn_ensemble
  - attn75_prior25
  - random_pos_oracle_layer  [control]

Axis B: LAYER assignment quality
--------------------------------
Use ORACLE Core50 positions, but choose intervention layer(s) without relation
labels.  This asks:

    "If position selection were solved, which self layer rule works best?"

Tested:
  - next_delta_peak
  - next_posdelta_peak
  - attn_ensemble_peak
  - next_delta_top2_split
  - next_delta_top3_split
  - next_delta_pm1_split
  - next_delta_soft_all

Reference:
  - oracle_core_exact

All edited methods use alpha = 1.0 exactly.
Multi-layer rules split total weight across layers, so a selected position has
total intervention weight 1.0 rather than multiplying strength by #layers.

Requirements:
  eval_self_selected_core_behavior_v1.py must be in the same directory.
  The previous feature cache must exist:
    <feature-dir>/all_candidate_self_features.pkl.gz

Recommended quick run:
CUDA_VISIBLE_DEVICES=0 python -u eval_core50_where_when_alpha1_v1.py \
  --run-dir output/qwen3b_coco_dynamic_L20_26_K36_all440 \
  --feature-dir output/qwen3b_coco_L20_26_self_predict_core \
  --model qwen-3b \
  --bundle "L20+L21+L22+L23+L24+L25+L26[global_unique]" \
  --max-eval-samples 80 \
  --output-dir output/qwen3b_core50_where_when_alpha1_N80 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import eval_self_selected_core_behavior_v1 as E


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument(
        "--bundle",
        default="L20+L21+L22+L23+L24+L25+L26[global_unique]",
    )
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument("--oracle-k", type=int, default=36)
    p.add_argument(
        "--core-mass",
        type=float,
        default=0.5,
        help="Default Core50; keep 0.5 for this diagnostic.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-eval-samples", type=int, default=80)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def canon_rel(x):
    return E.canon_rel(x)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


# -----------------------------------------------------------------------------
# Layer-rule helpers
# -----------------------------------------------------------------------------

def ensure_feature(df, col):
    if col not in df.columns or df[col].notna().sum() == 0:
        raise RuntimeError(
            f"Required feature {col!r} is missing/all-NaN in feature cache."
        )


def choose_layer_rows_for_oracle_positions(
    feature_sid,
    oracle_core_rows,
    rule,
    source_layers,
):
    """
    Return rows:
      position, [(edit_layer, weight), ...], score
    Oracle is used ONLY to provide position identity.
    Layer selection itself uses relation-free cached features.
    """
    if rule in {
        "next_delta_peak",
        "next_delta_top2_split",
        "next_delta_top3_split",
        "next_delta_pm1_split",
        "next_delta_soft_all",
    }:
        score_col = "RANK_last_attn_next_delta_mean"
    elif rule == "next_posdelta_peak":
        score_col = "last_attn_next_max_positive_delta"
    elif rule == "attn_ensemble_peak":
        score_col = "SCORE_attn_ensemble"
    else:
        raise ValueError(rule)

    ensure_feature(feature_sid, score_col)
    allowed = set(int(x) for x in source_layers)
    out = []

    for core_rank, cr in enumerate(oracle_core_rows, 1):
        pos = int(cr["position"])
        q = feature_sid[feature_sid["position"] == pos].copy()
        q["_score"] = pd.to_numeric(q[score_col], errors="coerce")
        q = q[
            q["source_layer"].astype(int).isin(allowed)
            & np.isfinite(q["_score"])
        ].copy()
        if not len(q):
            continue

        q = q.sort_values(
            ["_score", "source_layer"],
            ascending=[False, True],
        ).reset_index(drop=True)

        peak_layer = int(q.iloc[0]["source_layer"])
        peak_score = float(q.iloc[0]["_score"])

        if rule in {
            "next_delta_peak",
            "next_posdelta_peak",
            "attn_ensemble_peak",
        }:
            layers_weights = [(peak_layer, 1.0)]

        elif rule == "next_delta_top2_split":
            qq = q.drop_duplicates("source_layer").head(2)
            layers = [int(x) for x in qq["source_layer"]]
            layers_weights = [(L, 1.0 / len(layers)) for L in layers]

        elif rule == "next_delta_top3_split":
            qq = q.drop_duplicates("source_layer").head(3)
            layers = [int(x) for x in qq["source_layer"]]
            layers_weights = [(L, 1.0 / len(layers)) for L in layers]

        elif rule == "next_delta_pm1_split":
            layers = [
                L for L in [peak_layer - 1, peak_layer, peak_layer + 1]
                if L in allowed
            ]
            layers_weights = [(L, 1.0 / len(layers)) for L in layers]

        elif rule == "next_delta_soft_all":
            # Score is a within-sample/layer percentile rank in [0,1].
            # Normalize across available layers for this position.
            qq = q.drop_duplicates("source_layer").copy()
            vals = np.maximum(
                pd.to_numeric(qq["_score"], errors="coerce").to_numpy(float),
                0.0,
            )
            if vals.sum() <= 1e-12:
                vals = np.ones(len(vals), dtype=float)
            vals = vals / vals.sum()
            layers_weights = [
                (int(L), float(w))
                for L, w in zip(qq["source_layer"], vals)
            ]
        else:
            raise AssertionError(rule)

        out.append(
            {
                "core_rank": core_rank,
                "position": pos,
                "oracle_layer": int(cr["source_layer"]),
                "oracle_mediation": float(cr["mediation"]),
                "peak_self_layer": peak_layer,
                "peak_self_score": peak_score,
                "score_col": score_col,
                "layers_weights": layers_weights,
            }
        )
    return out


def specs_from_layer_rows(layer_rows, hreal, hgray):
    specs = defaultdict(list)
    export = []
    for r in layer_rows:
        pos = int(r["position"])
        for L, w in r["layers_weights"]:
            d = E.delta_at(hreal, hgray, int(L), pos)
            if d is None:
                continue
            specs[int(L)].append((pos, d, float(w)))
            export.append(
                {
                    "position": pos,
                    "core_rank": int(r["core_rank"]),
                    "oracle_layer": int(r["oracle_layer"]),
                    "oracle_mediation": float(r["oracle_mediation"]),
                    "peak_self_layer": int(r["peak_self_layer"]),
                    "edit_layer": int(L),
                    "edit_weight": float(w),
                    "self_score": float(r["peak_self_score"]),
                    "score_col": str(r["score_col"]),
                }
            )
    return dict(specs), export


# -----------------------------------------------------------------------------
# Behavioral summaries
# -----------------------------------------------------------------------------

def summarize(rows):
    df = pd.DataFrame(rows)
    out = []
    if not len(df):
        return pd.DataFrame()

    for key, g in df.groupby(
        ["axis", "method", "method_kind", "alpha"],
        dropna=False,
    ):
        axis, method, kind, alpha = key
        base = g["baseline_correct"].astype(bool).to_numpy()
        edit = g["correct"].astype(bool).to_numpy()
        bpred = g["baseline_prediction"].astype(str).to_numpy()
        epred = g["prediction"].astype(str).to_numpy()

        wrong_n = int((~base).sum())
        correct_n = int(base.sum())
        w2c = int(((~base) & edit).sum())
        c2w = int((base & (~edit)).sum())

        out.append(
            {
                "axis": axis,
                "method": method,
                "method_kind": kind,
                "alpha": float(alpha),
                "N": len(g),
                "baseline_acc": float(base.mean()),
                "edited_acc": float(edit.mean()),
                "gain": float(edit.mean() - base.mean()),
                "W2C": w2c,
                "C2W": c2w,
                "net": w2c - c2w,
                "repair_rate_given_wrong": safe_div(w2c, wrong_n),
                "preserve_rate_given_correct": safe_div(correct_n - c2w, correct_n),
                "changed": int(np.sum(bpred != epred)),
                "mean_selected_positions": safe_mean(g["n_selected_positions"]),
                "mean_edit_pairs": safe_mean(g["n_edit_pairs"]),
            }
        )

    return pd.DataFrame(out).sort_values(
        ["axis", "edited_acc", "net"],
        ascending=[True, False, False],
    )


def summarize_by_relation(rows):
    df = pd.DataFrame(rows)
    out = []
    if not len(df):
        return pd.DataFrame()

    for key, g in df.groupby(
        ["axis", "method", "method_kind", "alpha", "gt"],
        dropna=False,
    ):
        axis, method, kind, alpha, gt = key
        base = g["baseline_correct"].astype(bool).to_numpy()
        edit = g["correct"].astype(bool).to_numpy()
        out.append(
            {
                "axis": axis,
                "method": method,
                "method_kind": kind,
                "alpha": float(alpha),
                "relation": gt,
                "N": len(g),
                "baseline_acc": float(base.mean()),
                "edited_acc": float(edit.mean()),
                "gain": float(edit.mean() - base.mean()),
                "W2C": int(((~base) & edit).sum()),
                "C2W": int((base & (~edit)).sum()),
            }
        )
    return pd.DataFrame(out)


def render_summary(summary, core_sizes, fixed_k, a):
    lines = []
    lines.append("=" * 144)
    lines.append("CORE POSITION vs LAYER DIAGNOSTIC — ALPHA FIXED TO 1")
    lines.append("=" * 144)
    lines.append(
        f"core={E.fmt_core(a.core_mass)} | "
        f"true core size mean={core_sizes[E.fmt_core(a.core_mass)+'_N'].mean():.2f}, "
        f"median={core_sizes[E.fmt_core(a.core_mass)+'_N'].median():.1f} | "
        f"self position K={fixed_k}"
    )
    lines.append("")
    lines.append(
        "Axis POSITION: self-select position + ORACLE best layer. "
        "Higher acc means better relation-free position localization."
    )
    lines.append(
        "Axis LAYER: ORACLE core positions + self-selected layer rule. "
        "Higher acc means better layer assignment."
    )
    lines.append(
        "Multi-layer layer rules preserve total per-position intervention weight = alpha=1."
    )

    for axis in ["reference", "position", "layer"]:
        g = summary[summary["axis"] == axis].copy()
        if not len(g):
            continue
        lines.append("")
        lines.append("-" * 144)
        lines.append(axis.upper())
        lines.append("-" * 144)
        for r in g.sort_values("edited_acc", ascending=False).itertuples():
            lines.append(
                f"{r.method:<34s} "
                f"acc={r.edited_acc:.4f} gain={r.gain:+.4f} "
                f"W2C/C2W={int(r.W2C):3d}/{int(r.C2W):3d} "
                f"net={int(r.net):+3d} "
                f"repair={r.repair_rate_given_wrong:.3f} "
                f"preserve={r.preserve_rate_given_correct:.3f} "
                f"pos/edit={r.mean_selected_positions:.1f}/{r.mean_edit_pairs:.1f}"
            )

    lines.append("")
    lines.append("Readout:")
    lines.append(
        "  Best POSITION method tells us which self signal finds behaviorally useful token positions "
        "when exact layer is no longer the bottleneck."
    )
    lines.append(
        "  Best LAYER method tells us how to intervene on a known causal position without knowing its oracle layer."
    )
    lines.append(
        "  If multi-layer split beats hard peak, exact layer should be treated as a trajectory/window rather than "
        "a single discrete layer."
    )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    a = parse_args()
    if abs(float(a.alpha) - 1.0) > 1e-12:
        raise ValueError(
            "This focused script intentionally fixes alpha=1.0. "
            "Run with --alpha 1.0."
        )

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = E.infer_source_layers(a.bundle)
    target = E.fmt_core(a.core_mass)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dir = outdir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    baseline, med, sel, valid_sids = E.load_existing_run(a, source_layers)

    if a.max_eval_samples and a.max_eval_samples > 0:
        rng = np.random.default_rng(a.seed)
        valid_sids = sorted(
            rng.choice(
                valid_sids,
                size=min(a.max_eval_samples, len(valid_sids)),
                replace=False,
            ).tolist()
        )
        baseline = baseline[baseline["sid"].isin(valid_sids)].copy()
        med = med[med["sid"].isin(valid_sids)].copy()
        sel = sel[sel["sid"].isin(valid_sids)].copy()

    truth, core_sizes = E.build_core_truth(sel, [a.core_mass])
    fixed_k = int(round(float(core_sizes[target + "_N"].median())))

    features = E.load_features(a, valid_sids, source_layers)
    features_prior, prior_mix_col = E.add_loo_role_prior(
        features, truth[target], target
    )

    # Baseline lookup.
    baseline_by_sid = {}
    for r in baseline.itertuples():
        baseline_by_sid[int(r.sid)] = {
            "gt": canon_rel(getattr(r, "gt", "")),
            "baseline_prediction": canon_rel(r.baseline_prediction),
            "baseline_correct": bool(r.baseline_correct),
        }

    # Data/model.
    two = E.base.import_two_object_module()
    prompts = E.base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _ = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    specs = E.base.merged_model_specs(two)
    spec = specs[a.model]
    model_cls = getattr(transformers, spec.model_class)

    load_kw = dict(
        dtype=E.base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    print(f"Loading {spec.repo_id}", flush=True)
    model = model_cls.from_pretrained(spec.repo_id, **load_kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    E.base.configure_processor(model, processor)
    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = E.base.resolve_decoder_layers(model)

    position_methods = [
        "attn_delta_max",
        "attn_delta_top2",
        "attn_delta_mean",
        "attn_posdelta_max",
        "attn_ensemble",
        "attn75_prior25",
        "random_pos_self_layer",
    ]
    layer_methods = [
        "next_delta_peak",
        "next_posdelta_peak",
        "attn_ensemble_peak",
        "next_delta_top2_split",
        "next_delta_top3_split",
        "next_delta_pm1_split",
        "next_delta_soft_all",
    ]

    print("=" * 144)
    print("CORE50 WHERE vs WHEN — alpha=1")
    print("=" * 144)
    print("N:", len(valid_sids))
    print("source layers:", source_layers)
    print("self position K:", fixed_k)
    print("position methods:", position_methods)
    print("layer methods:", layer_methods)
    print("generations/sample:", 1 + len(position_methods) + len(layer_methods))
    print()

    all_rows = []
    selected_rows = []

    try:
        for sid in tqdm(valid_sids, desc="WHERE/WHEN"):
            sid = int(sid)
            cache = chunk_dir / f"sid_{sid}.pkl.gz"
            if cache.exists():
                obj = pd.read_pickle(cache, compression="gzip")
                all_rows.extend(obj["rows"])
                selected_rows.extend(obj["selected"])
                continue

            gt = baseline_by_sid[sid]["gt"]
            if not gt:
                gt = canon_rel(sel[sel["sid"] == sid]["relation"].iloc[0])

            pr = prompts[sid]
            real = gray = rb = gb = None
            sid_rows, sid_selected = [], []

            try:
                real = E.base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = E.make_gray_image(real, a.gray_value)

                device = torch.device(a.device)
                rb = E.base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=str(pr["question_text"]),
                    device=device,
                )
                gb = E.base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=str(pr["question_text"]),
                    device=device,
                )

                hreal = E.capture_cpu(model, decoder_layers, rb, source_layers)
                hgray = E.capture_cpu(model, decoder_layers, gb, source_layers)

                f_sid = features[features["sid"] == sid].copy()
                fp_sid = features_prior[features_prior["sid"] == sid].copy()
                med_sid = med[med["sid"] == sid].copy()
                oracle_core = truth[target][sid]

                def add_result(axis, method, kind, gen_out, npos):
                    sid_rows.append(
                        {
                            "sid": sid,
                            "gt": gt,
                            "baseline_prediction": baseline_by_sid[sid][
                                "baseline_prediction"
                            ],
                            "baseline_correct": baseline_by_sid[sid][
                                "baseline_correct"
                            ],
                            "axis": axis,
                            "method": method,
                            "method_kind": kind,
                            "alpha": 1.0,
                            "prediction": gen_out["prediction"],
                            "correct": gen_out["prediction"] == gt,
                            "text": gen_out["text"],
                            "n_selected_positions": int(npos),
                            "n_edit_pairs": int(gen_out["n_edit_pairs"]),
                        }
                    )

                # ---------------------------------------------------------
                # Reference: oracle Core50 exact.
                # ---------------------------------------------------------
                specs_mid, exp = E.build_oracle_exact_specs(
                    oracle_core, hreal, hgray
                )
                out = E.generate_with_specs(
                    model, processor, decoder_layers, rb,
                    specs_mid, 1.0, a.max_new_tokens
                )
                add_result(
                    "reference", "oracle_core_exact", "diagnostic_oracle",
                    out, len(oracle_core)
                )

                # ---------------------------------------------------------
                # Axis A: self position + oracle layer.
                # ---------------------------------------------------------
                for method in position_methods:
                    use_df = fp_sid if method == "attn75_prior25" else f_sid
                    chosen_self = E.select_self_positions(
                        use_df,
                        method=method,
                        k=fixed_k,
                        sid=sid,
                        seed=a.seed,
                        target_prior_mix_col=(
                            prior_mix_col if method == "attn75_prior25" else None
                        ),
                    )

                    chosen_oracle_layer = E.choose_oracle_layer_for_self_positions(
                        chosen_self,
                        med_sid,
                    )
                    specs_mid, exp = E.build_specs_from_selected(
                        chosen_oracle_layer,
                        hreal,
                        hgray,
                        source_layers,
                        "peak",
                    )
                    out = E.generate_with_specs(
                        model, processor, decoder_layers, rb,
                        specs_mid, 1.0, a.max_new_tokens
                    )

                    kind = (
                        "random_control_oracle_layer"
                        if method == "random_pos_self_layer"
                        else (
                            "calibrated_self_pos_oracle_layer"
                            if method == "attn75_prior25"
                            else "self_pos_oracle_layer"
                        )
                    )
                    add_result(
                        "position",
                        method.replace("random_pos_self_layer",
                                       "random_pos_oracle_layer"),
                        kind,
                        out,
                        len(chosen_oracle_layer),
                    )

                    for rank, r in enumerate(chosen_oracle_layer.itertuples(), 1):
                        sid_selected.append(
                            {
                                "sid": sid,
                                "axis": "position",
                                "method": method,
                                "rank": rank,
                                "position": int(r.position),
                                "oracle_edit_layer": int(r.peak_layer),
                                "position_score": float(r.position_score),
                            }
                        )

                # ---------------------------------------------------------
                # Axis B: oracle positions + self layer rule.
                # ---------------------------------------------------------
                for rule in layer_methods:
                    layer_rows = choose_layer_rows_for_oracle_positions(
                        f_sid,
                        oracle_core,
                        rule,
                        source_layers,
                    )
                    specs_mid, exp = specs_from_layer_rows(
                        layer_rows, hreal, hgray
                    )
                    out = E.generate_with_specs(
                        model, processor, decoder_layers, rb,
                        specs_mid, 1.0, a.max_new_tokens
                    )
                    add_result(
                        "layer",
                        rule,
                        "oracle_pos_self_layer",
                        out,
                        len(layer_rows),
                    )

                    for e in exp:
                        sid_selected.append(
                            {
                                "sid": sid,
                                "axis": "layer",
                                "method": rule,
                                **e,
                            }
                        )

                pd.to_pickle(
                    {"rows": sid_rows, "selected": sid_selected},
                    cache,
                    compression="gzip",
                )
                all_rows.extend(sid_rows)
                selected_rows.extend(sid_selected)

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(all_rows)
    by_relation = summarize_by_relation(all_rows)

    write_csv(outdir / "behavior_per_sample.csv", all_rows)
    summary.to_csv(outdir / "behavior_summary.csv", index=False)
    by_relation.to_csv(outdir / "behavior_by_relation.csv", index=False)
    write_csv(outdir / "selected_details.csv", selected_rows)
    core_sizes.to_csv(outdir / "core_sizes.csv", index=False)

    text = render_summary(summary, core_sizes, fixed_k, a)
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")

    meta = {
        "alpha": 1.0,
        "core_mass": a.core_mass,
        "target": target,
        "N": len(valid_sids),
        "fixed_self_position_k": fixed_k,
        "source_layers": source_layers,
        "position_methods": position_methods,
        "layer_methods": layer_methods,
        "run_dir": a.run_dir,
        "feature_dir": a.feature_dir,
        "note": (
            "Position-axis experiments use oracle layer; layer-axis experiments "
            "use oracle core positions. Only oracle_core_exact uses both oracle "
            "position and oracle layer. This script is diagnostic, not deployable."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print()
    print(text)
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
