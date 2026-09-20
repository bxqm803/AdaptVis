#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_qwen3b_coco_sparse_causal_bottleneck_v1.py

Experiment 1 for the paper: Sparse Causal Decision Bottleneck
==============================================================
Target setting:
    model   = Qwen/Qwen2.5-VL-3B-Instruct (repo alias qwen-3b)
    dataset = COCO-two, full 440 samples by default

This script intentionally REUSES the repository's established causal-state
experiments instead of reimplementing them:

A) Sparsity / behavioral sufficiency
   eval_qwen_oracle_k36_prefix_curve_v1.py

   One per-sample writer-guided K36 ranking is constructed with

       M(L,p) = (h_real(L,p)-h_gray(L,p))^T grad_h J_GT

   under the existing positive global_unique rule.  Nested prefixes of that
   SAME ranking are evaluated with actual model.generate():

       K = 1,3,5,7,10,20,36    (configurable)

   Main output used here:
       <prefix-dir>/summary.csv

B) Relation-specific causal-set overlap
   eval_four_writer_k7_competition_v1.py

   For every sample, all four candidate writers are traced independently:

       T_left^K, T_right^K, T_above^K, T_below^K

   GT is NOT used to choose which writer to trace.  We analyze the raw
   positive global_unique Top-K sets.

   Main output used here:
       <fourway-dir>/selected_topk_all_directions.csv

Primary figure
--------------
A two-panel paper figure:

  (a) Actual generation accuracy vs causal-set size K.
      Baseline accuracy is shown as a horizontal reference.

  (b) Mean within-sample POSITION overlap of relation-specific Top-K causal
      token sets.  For each sample and relation pair (r,s):

          overlap_pos(r,s) = |P_r ∩ P_s| / K

      Only samples for which both candidate sets contain exactly K selected
      positive positions are included for that pair.

We additionally export an exact token-layer STATE overlap matrix:

          overlap_state(r,s)
            = |{(L,p)}_r ∩ {(L,p)}_s| / K

This distinguishes "same token position, possibly strongest at a different
layer" from exact state identity.

Important interpretation
------------------------
This is an ORACLE mechanistic diagnostic for sparsity, matching the existing
K36 experiment.  It establishes that decision control can be concentrated in
few states and that relation-specific causal sets are partially shared.  It
DOES NOT by itself prove middle-spatial-state -> causal-state mediation.

Typical full run
----------------
CUDA_VISIBLE_DEVICES=0 python -u run_qwen3b_coco_sparse_causal_bottleneck_v1.py \
  --oracle-run-dir output/qwen3b_coco_dynamic_L20_26_K36_all440 \
  --prefix-dir output/qwen3b_sparse_bottleneck_prefix_all440_v1 \
  --fourway-dir output/qwen3b_sparse_bottleneck_fourway_k7_all440_v1 \
  --output-dir output/qwen3b_sparse_causal_bottleneck_figure_v1

If both component runs already exist, this command only post-processes/plots.
To ONLY plot and never launch model jobs:

python -u run_qwen3b_coco_sparse_causal_bottleneck_v1.py \
  --plot-only \
  --prefix-dir output/qwen3b_sparse_bottleneck_prefix_all440_v1 \
  --fourway-dir output/qwen3b_sparse_bottleneck_fourway_k7_all440_v1 \
  --output-dir output/qwen3b_sparse_causal_bottleneck_figure_v1

Quick N=80 smoke test
---------------------
CUDA_VISIBLE_DEVICES=0 python -u run_qwen3b_coco_sparse_causal_bottleneck_v1.py \
  --eval-max-samples 80 \
  --prefix-dir output/qwen3b_sparse_bottleneck_prefix_N80_v1 \
  --fourway-dir output/qwen3b_sparse_bottleneck_fourway_k7_N80_v1 \
  --output-dir output/qwen3b_sparse_causal_bottleneck_figure_N80_v1
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend on Slurm/headless nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REL = ("left", "right", "above", "below")
REL_DISPLAY = ("Left", "Right", "Above", "Below")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Existing canonical full-440 oracle run.  The four-way tracer reuses its
    # baseline.csv + learned_writers.npz so writer definitions stay identical.
    p.add_argument(
        "--oracle-run-dir",
        default="output/qwen3b_coco_dynamic_L20_26_K36_all440",
    )

    p.add_argument(
        "--prefix-script",
        default="eval_qwen_oracle_k36_prefix_curve_v1.py",
    )
    p.add_argument(
        "--fourway-script",
        default="eval_four_writer_k7_competition_v1.py",
    )

    p.add_argument(
        "--prefix-dir",
        default="output/qwen3b_sparse_bottleneck_prefix_all440_v1",
    )
    p.add_argument(
        "--fourway-dir",
        default="output/qwen3b_sparse_bottleneck_fourway_k7_all440_v1",
    )
    p.add_argument(
        "--output-dir",
        default="output/qwen3b_sparse_causal_bottleneck_figure_v1",
    )

    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument("--ks", default="1,3,5,7,10,20,36")
    p.add_argument("--max-rank", type=int, default=36)
    p.add_argument("--overlap-k", type=int, default=7)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all COCO samples; >0 = smoke-test cap for BOTH component runs.",
    )

    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Never launch model jobs; require both component outputs to exist.",
    )
    p.add_argument(
        "--overwrite-runs",
        action="store_true",
        help="Pass --overwrite to missing component runs. Existing COMPLETE runs are never rerun.",
    )
    p.add_argument(
        "--overwrite-figure",
        action="store_true",
        help="Allow replacing files in --output-dir.",
    )
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    vals = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError(f"No integers parsed from {text!r}")
    return vals


def canon_rel(x) -> str:
    s = str(x).strip().lower().replace("-", "_")
    return {
        "left": "left",
        "left_of": "left",
        "right": "right",
        "right_of": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
        "bottom": "below",
    }.get(s, s)


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def run_command(cmd: Sequence[str], label: str):
    print("\n" + "=" * 140, flush=True)
    print(label, flush=True)
    print("=" * 140, flush=True)
    print(" ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def maybe_run_prefix(a, ks: List[int]):
    out = Path(a.prefix_dir)
    summary = out / "summary.csv"
    generation = out / "generation_by_k.csv"

    if summary.exists() and generation.exists():
        print(f"[reuse] prefix curve: {out}", flush=True)
        return

    if a.plot_only:
        raise FileNotFoundError(
            f"--plot-only but prefix result is incomplete: {out}\n"
            f"Expected {summary} and {generation}"
        )

    script = Path(a.prefix_script)
    require_file(script, "prefix evaluator script")

    if out.exists() and any(out.iterdir()) and not a.overwrite_runs:
        raise RuntimeError(
            f"Prefix output exists but is incomplete: {out}\n"
            "The underlying evaluator does not resume partial runs. Remove that "
            "directory or rerun this wrapper with --overwrite-runs."
        )

    cmd = [
        sys.executable, "-u", str(script),
        "--model", a.model,
        "--data-root", a.data_root,
        "--prompt-jsonl", a.prompt_jsonl,
        "--source-layers", a.source_layers,
        "--target-layers", a.target_layers,
        "--ks", ",".join(map(str, ks)),
        "--max-rank", str(a.max_rank),
        "--alpha", str(a.alpha),
        "--eval-scope", "all_data",
        "--eval-max-samples", str(a.eval_max_samples),
        "--device", a.device,
        "--attn-impl", a.attn_impl,
        "--seed", str(a.seed),
        "--output-dir", str(out),
    ]
    if a.overwrite_runs:
        cmd.append("--overwrite")
    run_command(cmd, "RUN A: ORACLE K36 NESTED-PREFIX GENERATION CURVE")


def maybe_run_fourway(a):
    out = Path(a.fourway_dir)
    selected = out / "selected_topk_all_directions.csv"
    per_sample = out / "per_sample_direction_scores.csv"

    if selected.exists() and per_sample.exists():
        print(f"[reuse] four-way causal sets: {out}", flush=True)
        return

    if a.plot_only:
        raise FileNotFoundError(
            f"--plot-only but four-way result is incomplete: {out}\n"
            f"Expected {selected} and {per_sample}"
        )

    script = Path(a.fourway_script)
    require_file(script, "four-way writer tracing script")

    oracle = Path(a.oracle_run_dir)
    require_file(oracle / "baseline.csv", "oracle baseline.csv")
    require_file(oracle / "learned_writers.npz", "oracle learned_writers.npz")

    # This script is chunk-resumable.  Do not force overwrite unless explicitly
    # requested; if interrupted, rerunning this wrapper continues cached SIDs.
    cmd = [
        sys.executable, "-u", str(script),
        "--run-dir", str(oracle),
        "--model", a.model,
        "--data-root", a.data_root,
        "--prompt-jsonl", a.prompt_jsonl,
        "--source-layers", a.source_layers,
        "--target-layers", a.target_layers,
        "--k", str(a.overlap_k),
        "--max-eval-samples", str(a.eval_max_samples),
        "--device", a.device,
        "--attn-impl", a.attn_impl,
        "--seed", str(a.seed),
        "--output-dir", str(out),
    ]
    if a.overwrite_runs:
        cmd.append("--overwrite")
    run_command(cmd, "RUN B: FOUR-WAY RELATION-SPECIFIC CAUSAL SETS")


def load_prefix_summary(prefix_dir: Path, ks: Sequence[int]) -> pd.DataFrame:
    p = prefix_dir / "summary.csv"
    require_file(p, "prefix summary.csv")
    df = pd.read_csv(p)

    needed = {"k", "baseline_acc", "edited_acc"}
    miss = needed - set(df.columns)
    if miss:
        raise RuntimeError(f"{p} missing columns: {sorted(miss)}")

    df["k"] = pd.to_numeric(df["k"], errors="raise").astype(int)
    for c in ["baseline_acc", "edited_acc", "delta_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    wanted = list(map(int, ks))
    q = df[df["k"].isin(wanted)].copy().sort_values("k")
    missing_k = sorted(set(wanted) - set(q["k"].tolist()))
    if missing_k:
        raise RuntimeError(
            f"{p} does not contain requested K values {missing_k}. "
            "Rerun the prefix experiment with the same --ks used here."
        )
    return q


def load_fourway_selected(fourway_dir: Path, k: int) -> pd.DataFrame:
    p = fourway_dir / "selected_topk_all_directions.csv"
    require_file(p, "four-way selected_topk_all_directions.csv")
    x = pd.read_csv(p)

    needed = {
        "sid", "candidate_relation", "selection_type",
        "source_layer", "position", "rank",
    }
    miss = needed - set(x.columns)
    if miss:
        raise RuntimeError(f"{p} missing columns: {sorted(miss)}")

    x = x[x["selection_type"].astype(str) == "raw"].copy()
    x["sid"] = pd.to_numeric(x["sid"], errors="raise").astype(int)
    x["source_layer"] = pd.to_numeric(
        x["source_layer"], errors="raise"
    ).astype(int)
    x["position"] = pd.to_numeric(x["position"], errors="raise").astype(int)
    x["rank"] = pd.to_numeric(x["rank"], errors="raise").astype(int)
    x["relation"] = x["candidate_relation"].map(canon_rel)
    x = x[x["relation"].isin(REL)].copy()
    x = x[x["rank"] <= int(k)].copy()

    # Defensive deduplication consistent with global_unique semantics.
    x = (
        x.sort_values(["sid", "relation", "rank"])
        .drop_duplicates(["sid", "relation", "position"], keep="first")
        .copy()
    )
    return x


def compute_overlap(
    selected: pd.DataFrame,
    k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute WITHIN-SAMPLE overlap between four relation-specific causal sets.

    Primary token-position overlap:
        |P_r ∩ P_s| / K

    Exact state overlap:
        |{(L,p)}_r ∩ {(L,p)}_s| / K

    To avoid conflating overlap with failure to obtain K positive states, a
    sample contributes to pair (r,s) only when BOTH candidate sets contain
    exactly K unique token positions.
    """
    rows = []

    grouped: Dict[Tuple[int, str], pd.DataFrame] = {
        (int(sid), str(rel)): g.copy()
        for (sid, rel), g in selected.groupby(["sid", "relation"])
    }
    sids = sorted(selected["sid"].unique().tolist())

    completeness = []
    for sid in sids:
        for r in REL:
            g = grouped.get((sid, r), pd.DataFrame())
            npos = int(g["position"].nunique()) if len(g) else 0
            nexact = (
                int(g[["source_layer", "position"]].drop_duplicates().shape[0])
                if len(g) else 0
            )
            completeness.append({
                "sid": int(sid),
                "relation": r,
                "n_positions": npos,
                "n_exact_states": nexact,
                "complete_k": bool(npos == int(k)),
            })

    for sid in sids:
        sets_pos = {}
        sets_exact = {}
        complete = {}
        for r in REL:
            g = grouped.get((sid, r), pd.DataFrame())
            if len(g):
                sets_pos[r] = set(map(int, g["position"].tolist()))
                sets_exact[r] = {
                    (int(z.source_layer), int(z.position))
                    for z in g[["source_layer", "position"]].itertuples(index=False)
                }
            else:
                sets_pos[r] = set()
                sets_exact[r] = set()
            complete[r] = len(sets_pos[r]) == int(k)

        for r in REL:
            for s in REL:
                if not (complete[r] and complete[s]):
                    continue
                ip = len(sets_pos[r] & sets_pos[s])
                ie = len(sets_exact[r] & sets_exact[s])
                up = len(sets_pos[r] | sets_pos[s])
                ue = len(sets_exact[r] | sets_exact[s])
                rows.append({
                    "sid": int(sid),
                    "relation_a": r,
                    "relation_b": s,
                    "K": int(k),
                    "position_intersection": int(ip),
                    "position_overlap_fraction": float(ip / k),
                    "position_jaccard": float(ip / up) if up else np.nan,
                    "exact_intersection": int(ie),
                    "exact_overlap_fraction": float(ie / k),
                    "exact_jaccard": float(ie / ue) if ue else np.nan,
                })

    pair = pd.DataFrame(rows)
    if not len(pair):
        raise RuntimeError(
            "No complete relation-pair Top-K sets found. Check four-way output "
            f"and --overlap-k={k}."
        )

    def aggregate(metric: str) -> pd.DataFrame:
        g = (
            pair.groupby(["relation_a", "relation_b"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
        return g

    return (
        pair,
        aggregate("position_overlap_fraction"),
        aggregate("exact_overlap_fraction"),
        pd.DataFrame(completeness),
    )


def agg_to_matrix(agg: pd.DataFrame, value="mean") -> np.ndarray:
    lookup = {
        (str(r.relation_a), str(r.relation_b)): float(getattr(r, value))
        for r in agg.itertuples(index=False)
    }
    M = np.full((4, 4), np.nan, dtype=np.float64)
    for i, r in enumerate(REL):
        for j, s in enumerate(REL):
            M[i, j] = lookup.get((r, s), np.nan)
    return M


def matrix_csv(M: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(M, index=REL_DISPLAY, columns=REL_DISPLAY)


def save_main_figure(
    prefix: pd.DataFrame,
    position_M: np.ndarray,
    outdir: Path,
    overlap_k: int,
):
    # Deliberately use Matplotlib defaults rather than paper-specific colors so
    # the figure can inherit the user's final style later.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.15))

    ax = axes[0]
    k = prefix["k"].to_numpy(int)
    acc = prefix["edited_acc"].to_numpy(float) * 100.0
    base = float(prefix["baseline_acc"].dropna().iloc[0]) * 100.0

    ax.plot(k, acc, marker="o", linewidth=2)
    ax.axhline(base, linestyle="--", linewidth=1.4)
    for xx, yy in zip(k, acc):
        ax.annotate(
            f"{yy:.1f}",
            (xx, yy),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    ax.text(
        0.98, base,
        f" baseline {base:.1f}% ",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8,
    )
    ax.set_xlabel("Number of causal states (K)")
    ax.set_ylabel("Generation accuracy (%)")
    ax.set_title("(a) Decision control is sparse")
    ax.set_xticks(k)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    im = ax.imshow(position_M, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(4), labels=REL_DISPLAY)
    ax.set_yticks(np.arange(4), labels=REL_DISPLAY)
    ax.set_xlabel("Causal set $T_{r'}$")
    ax.set_ylabel("Causal set $T_r$")
    ax.set_title(f"(b) Relation-specific Top-{overlap_k} tokens overlap")
    for i in range(4):
        for j in range(4):
            v = position_M[i, j]
            txt = "--" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Shared token positions / K")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figure_sparse_causal_bottleneck.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_exact_figure(exact_M: np.ndarray, outdir: Path, overlap_k: int):
    fig, ax = plt.subplots(figsize=(5.0, 4.35))
    im = ax.imshow(exact_M, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(4), labels=REL_DISPLAY)
    ax.set_yticks(np.arange(4), labels=REL_DISPLAY)
    ax.set_xlabel("Causal set $T_{r'}$")
    ax.set_ylabel("Causal set $T_r$")
    ax.set_title(f"Exact token-layer overlap, Top-{overlap_k}")
    for i in range(4):
        for j in range(4):
            v = exact_M[i, j]
            txt = "--" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Shared exact states (L,p) / K")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figure_exact_state_overlap_k{overlap_k}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_summary(
    prefix: pd.DataFrame,
    pos_agg: pd.DataFrame,
    exact_agg: pd.DataFrame,
    completeness: pd.DataFrame,
    overlap_k: int,
) -> str:
    lines = []
    lines.append("QWEN2.5-VL-3B / COCO — SPARSE CAUSAL BOTTLENECK")
    lines.append("=" * 92)
    lines.append("")
    lines.append("A. NESTED CAUSAL PREFIX -> ACTUAL GENERATION")
    lines.append("-" * 92)
    cols = [c for c in [
        "k", "N", "baseline_acc", "edited_acc", "delta_acc",
        "W2C", "C2W", "net", "mean_selected_tokens",
    ] if c in prefix.columns]
    lines.append(prefix[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    lines.append(f"B. WITHIN-SAMPLE FOUR-RELATION OVERLAP AT K={overlap_k}")
    lines.append("-" * 92)
    pm = agg_to_matrix(pos_agg)
    em = agg_to_matrix(exact_agg)
    lines.append("Position overlap fraction |P_r ∩ P_r'| / K:")
    lines.append(matrix_csv(pm).to_string(float_format=lambda x: f"{x:.3f}"))
    lines.append("")
    lines.append("Exact state overlap fraction |(L,p)_r ∩ (L,p)_r'| / K:")
    lines.append(matrix_csv(em).to_string(float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    comp = (
        completeness.groupby("relation", as_index=False)
        .agg(
            N_samples=("sid", "nunique"),
            complete_K_fraction=("complete_k", "mean"),
            mean_selected_positions=("n_positions", "mean"),
        )
    )
    lines.append("Top-K completeness by candidate relation:")
    lines.append(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("Interpretation guardrail:")
    lines.append(
        "  This experiment establishes sparsity and partial relation-specific sharing of causal "
        "decision states. It does not establish spatial-state -> causal-state mediation; that is "
        "the next experiment."
    )
    return "\n".join(lines) + "\n"


def main():
    a = parse_args()
    ks = parse_ints(a.ks)
    if a.overlap_k <= 0:
        raise ValueError("--overlap-k must be positive")
    if max(ks) > a.max_rank:
        raise ValueError("max(--ks) must be <= --max-rank")

    # Component experiments.
    maybe_run_prefix(a, ks)
    maybe_run_fourway(a)

    # Analysis outputs.
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not a.overwrite_figure:
        occupied = [
            outdir / "figure_sparse_causal_bottleneck.png",
            outdir / "figure_sparse_causal_bottleneck.pdf",
        ]
        if any(p.exists() for p in occupied):
            print(
                f"[note] figure files already exist in {outdir}; they will be refreshed "
                "from the current CSVs. Use --overwrite-figure only as an explicit audit flag.",
                flush=True,
            )

    prefix = load_prefix_summary(Path(a.prefix_dir), ks)
    selected = load_fourway_selected(Path(a.fourway_dir), a.overlap_k)
    pair, pos_agg, exact_agg, completeness = compute_overlap(selected, a.overlap_k)

    pos_M = agg_to_matrix(pos_agg)
    exact_M = agg_to_matrix(exact_agg)

    # Save machine-readable paper statistics.
    prefix.to_csv(outdir / "k_recovery_summary.csv", index=False)
    pair.to_csv(outdir / "overlap_per_sample_pair.csv", index=False)
    pos_agg.to_csv(outdir / "position_overlap_summary_long.csv", index=False)
    exact_agg.to_csv(outdir / "exact_state_overlap_summary_long.csv", index=False)
    completeness.to_csv(outdir / "causal_set_completeness.csv", index=False)
    matrix_csv(pos_M).to_csv(outdir / "position_overlap_matrix.csv")
    matrix_csv(exact_M).to_csv(outdir / "exact_state_overlap_matrix.csv")

    save_main_figure(prefix, pos_M, outdir, a.overlap_k)
    save_exact_figure(exact_M, outdir, a.overlap_k)

    summary = render_summary(prefix, pos_agg, exact_agg, completeness, a.overlap_k)
    (outdir / "analysis_summary.txt").write_text(summary, encoding="utf-8")

    metadata = {
        "model": a.model,
        "dataset": "COCO-two",
        "prefix_dir": str(Path(a.prefix_dir)),
        "fourway_dir": str(Path(a.fourway_dir)),
        "oracle_run_dir": str(Path(a.oracle_run_dir)),
        "source_layers": parse_ints(a.source_layers),
        "target_layers": parse_ints(a.target_layers),
        "ks": ks,
        "overlap_k": int(a.overlap_k),
        "alpha": float(a.alpha),
        "eval_max_samples": int(a.eval_max_samples),
        "position_overlap_definition": (
            "within-sample mean |TopK token positions for relation r ∩ TopK token positions "
            "for relation r'| / K; pair included only if both have exactly K positions"
        ),
        "exact_overlap_definition": (
            "within-sample mean |TopK exact (source_layer,position) states for relation r ∩ "
            "TopK exact states for relation r'| / K"
        ),
        "selection": "raw positive global_unique writer-guided causal Top-K",
        "causal_score": "(h_real-h_gray)^T grad_h J_relation_writer",
        "behavioral_test": "actual autoregressive model.generate for nested GT-writer K36 prefixes",
        "interpretation": (
            "oracle mechanism diagnostic; sparsity + relation-specific sharing, not mediation"
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + summary, flush=True)
    print("Saved:", flush=True)
    for name in [
        "figure_sparse_causal_bottleneck.png",
        "figure_sparse_causal_bottleneck.pdf",
        f"figure_exact_state_overlap_k{a.overlap_k}.png",
        f"figure_exact_state_overlap_k{a.overlap_k}.pdf",
        "k_recovery_summary.csv",
        "position_overlap_matrix.csv",
        "exact_state_overlap_matrix.csv",
        "overlap_per_sample_pair.csv",
        "analysis_summary.txt",
        "metadata.json",
    ]:
        print(" ", outdir / name, flush=True)


if __name__ == "__main__":
    main()
