#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_oracle_core50_top10_candidate_bank_all440_v1.py

Pure post-processing of an EXISTING full-440 oracle K36 run.

Purpose
-------
Build a fixed candidate bank for later spatial-head token recovery experiments.

For every COCO sample:
  1) take the oracle writer-guided positive global_unique K36 states
     from selected_tokens.csv;
  2) sort by mediation;
  3) reconstruct the same sample-specific Core50 definition used by
     analyze_dynamic_k36_causal_core_v1.py:
         the shortest prefix reaching 50% of positive mediation mass
         inside the selected K36;
  4) export the first Top-10 exact states (layer, token position);
  5) mark which Top-10 states belong to Core50.

NO model loading.
NO forward pass.
NO gradients.
NO generation.

This is intended so future spatial-head experiments can read ONE frozen CSV
instead of re-running oracle writer tracing.

Text-token statistics
---------------------
Statistics are computed only for TEXT candidates:
  - SUBJECT and REFERENCE are canonicalized across samples.
  - relation words are canonicalized by relation-word identity.
  - all other fixed prompt text is aggregated by tokenizer token identity/text.
  - visual tokens are excluded from the text-only statistics.

Outputs include:
  top10_candidates_all440.csv
  core50_candidates_all440.csv
  top10_candidates_by_sample.jsonl
  sample_core50_summary.csv
  text_top10_layer_summary.csv
  text_top10_token_summary.csv
  text_top10_layer_token_counts.csv
  text_top10_layer_role_summary.csv
  text_core50_layer_summary.csv
  text_core50_token_summary.csv
  text_core50_layer_token_counts.csv
  text_core50_layer_role_summary.csv
  top10_rank_role_summary.csv
  analysis_summary.txt
  metadata.json

Typical
-------
python -u build_oracle_core50_top10_candidate_bank_all440_v1.py \
  --run-dir output/qwen3b_coco_dynamic_L20_26_K36_all440 \
  --require-n 440 \
  --topk 10 \
  --output-dir output/qwen3b_oracle_core50_top10_bank_all440 \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BUNDLE = "L20+L21+L22+L23+L24+L25+L26[global_unique]"
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="Existing full-440 oracle run containing selected_tokens.csv.",
    )
    p.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE,
        help="Exact source_bundle string to analyze.",
    )
    p.add_argument("--k-source", type=int, default=36)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--mass-threshold", type=float, default=0.50)
    p.add_argument("--condition", default="positive")
    p.add_argument("--selection-strategy", default="global_unique")
    p.add_argument(
        "--require-n",
        type=int,
        default=440,
        help="Fail unless this many unique SIDs are present; 0 disables.",
    )
    p.add_argument(
        "--positive-eps",
        type=float,
        default=0.0,
        help="Mediation must exceed this value to contribute positive mass.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def canon_rel(x):
    s = str(x).strip().lower()
    return {
        "above": "on",
        "below": "under",
        "on": "on",
        "under": "under",
        "left": "left",
        "right": "right",
    }.get(s, s)


def clean_token(tok):
    """
    Make tokenizer pieces easier to read/group while preserving special tokens.
    This is only a display/grouping field; original token is always retained.
    """
    s = str(tok)
    s = s.replace("\\n", "↵")
    s = s.replace("Ċ", "↵")
    s = s.replace("Ġ", " ")
    s = s.replace("▁", " ")
    s = s.strip()
    return s if s else "<SPACE>"


def canonical_token_fields(row):
    cat = str(row.get("category", ""))
    broad = str(row.get("broad_category", ""))
    tok = clean_token(row.get("token", ""))

    if cat == "subject" or broad == "subject":
        return "subject", "SUBJECT", "SUBJECT"

    if cat == "reference" or broad == "reference":
        return "reference", "REFERENCE", "REFERENCE"

    if cat.startswith("relation_word:"):
        word = cat.split(":", 1)[1]
        return "relation_word", f"RELATION_WORD:{word}", word

    if broad == "relation_words":
        return "relation_word", f"RELATION_WORD:{tok}", tok

    # Everything else in text space is a fixed prompt token/special text token.
    return "other_text", f"TEXT:{tok}", tok


def load_selected(a):
    run_dir = Path(a.run_dir)
    selected_path = run_dir / "selected_tokens.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)

    sel = pd.read_csv(selected_path)

    required = {
        "sid",
        "relation",
        "condition",
        "selection_strategy",
        "source_bundle",
        "k",
        "source_layer",
        "rank",
        "position",
        "token",
        "category",
        "broad_category",
        "mediation",
    }
    missing = sorted(required - set(sel.columns))
    if missing:
        raise RuntimeError(
            f"{selected_path} missing required columns: {missing}"
        )

    sel["sid"] = pd.to_numeric(sel["sid"], errors="raise").astype(int)
    sel["k"] = pd.to_numeric(sel["k"], errors="raise").astype(int)
    sel["source_layer"] = pd.to_numeric(
        sel["source_layer"], errors="raise"
    ).astype(int)
    sel["position"] = pd.to_numeric(
        sel["position"], errors="raise"
    ).astype(int)
    sel["mediation"] = pd.to_numeric(
        sel["mediation"], errors="coerce"
    )
    sel["relation"] = sel["relation"].map(canon_rel)

    mask = (
        (sel["condition"].astype(str) == str(a.condition))
        & (
            sel["selection_strategy"].astype(str)
            == str(a.selection_strategy)
        )
        & (sel["source_bundle"].astype(str) == str(a.bundle))
        & (sel["k"] == int(a.k_source))
    )
    q = sel[mask].copy()

    if not len(q):
        avail = (
            sel[
                [
                    "condition",
                    "selection_strategy",
                    "source_bundle",
                    "k",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                ["source_bundle", "selection_strategy", "condition", "k"]
            )
        )
        raise RuntimeError(
            "No selected_tokens rows match the requested configuration.\n"
            f"Requested bundle={a.bundle!r}, strategy={a.selection_strategy!r}, "
            f"condition={a.condition!r}, k={a.k_source}\n\n"
            "Available configurations:\n"
            + avail.head(100).to_string(index=False)
        )

    # global_unique is supposed to use each token position at most once.
    # Re-enforce it defensively by keeping the strongest layer for a position.
    q = q.sort_values(
        ["sid", "mediation", "source_layer", "position"],
        ascending=[True, False, True, True],
    )
    q = q.drop_duplicates(
        ["sid", "position"],
        keep="first",
    )

    return selected_path, q


def maybe_merge_token_id(run_dir, q):
    """
    selected_tokens.csv historically omits token_id, while mediation_tokens.csv
    contains it. Merge token_id back by exact (sid, layer, position) when possible.
    """
    med_path = Path(run_dir) / "mediation_tokens.csv"
    if not med_path.exists():
        q["token_id"] = np.nan
        return q, None

    med = pd.read_csv(
        med_path,
        usecols=lambda c: c
        in {
            "sid",
            "source_layer",
            "position",
            "token_id",
            "token",
            "category",
            "broad_category",
            "mediation",
        },
    )

    needed = {"sid", "source_layer", "position"}
    if not needed <= set(med.columns):
        q["token_id"] = np.nan
        return q, med_path

    med["sid"] = pd.to_numeric(med["sid"], errors="coerce").astype("Int64")
    med["source_layer"] = pd.to_numeric(
        med["source_layer"], errors="coerce"
    ).astype("Int64")
    med["position"] = pd.to_numeric(
        med["position"], errors="coerce"
    ).astype("Int64")

    keep_cols = ["sid", "source_layer", "position"]
    if "token_id" in med.columns:
        keep_cols.append("token_id")

    mm = med[keep_cols].drop_duplicates(
        ["sid", "source_layer", "position"]
    )

    out = q.merge(
        mm,
        on=["sid", "source_layer", "position"],
        how="left",
        validate="one_to_one",
    )
    return out, med_path


def maybe_merge_baseline(run_dir, sample_df):
    p = Path(run_dir) / "baseline.csv"
    if not p.exists():
        return sample_df, None

    b = pd.read_csv(p)
    if "sid" not in b.columns:
        return sample_df, p

    b["sid"] = pd.to_numeric(b["sid"], errors="raise").astype(int)

    cols = ["sid"]
    for c in (
        "gt",
        "baseline_prediction",
        "baseline_correct",
        "text",
    ):
        if c in b.columns:
            cols.append(c)

    bb = b[cols].drop_duplicates("sid").copy()

    if "baseline_correct" in bb.columns:
        bb["baseline_correct"] = bb["baseline_correct"].map(boolify)

    rename = {}
    if "gt" in bb.columns:
        rename["gt"] = "baseline_gt"
    if "text" in bb.columns:
        rename["text"] = "baseline_text"
    bb = bb.rename(columns=rename)

    return sample_df.merge(bb, on="sid", how="left"), p


def build_bank(q, topk, mass_threshold, positive_eps):
    sample_rows = []
    candidate_chunks = []
    core_chunks = []

    for sid, g0 in q.groupby("sid", sort=True):
        g = (
            g0.sort_values(
                ["mediation", "source_layer", "position"],
                ascending=[False, True, True],
            )
            .reset_index(drop=True)
            .copy()
        )

        g["oracle_rank_k36"] = np.arange(1, len(g) + 1)

        med = g["mediation"].to_numpy(np.float64)
        positive = np.where(
            np.isfinite(med) & (med > positive_eps),
            med,
            0.0,
        )
        total_pos = float(positive.sum())

        if total_pos > EPS:
            cum = np.cumsum(positive) / total_pos
            hit = np.where(cum >= float(mass_threshold))[0]
            core_size = int(hit[0] + 1) if len(hit) else int(len(g))
        else:
            cum = np.full(len(g), np.nan)
            core_size = 0

        g["positive_mediation"] = positive
        g["cum_positive_mass_k36"] = cum
        g["core50_size"] = core_size
        g["in_core50"] = (
            g["oracle_rank_k36"] <= core_size
            if core_size > 0
            else False
        )
        g["in_top10_bank"] = g["oracle_rank_k36"] <= int(topk)

        # Text/visual identity.
        g["is_visual"] = (
            g["broad_category"].astype(str).eq("visual")
            | g["category"].astype(str).eq("visual")
        )
        g["is_text"] = ~g["is_visual"]

        token_classes = []
        canonical_keys = []
        display_tokens = []
        cleaned = []
        for _, r in g.iterrows():
            cl, key, disp = canonical_token_fields(r)
            token_classes.append(cl)
            canonical_keys.append(key)
            display_tokens.append(disp)
            cleaned.append(clean_token(r["token"]))

        g["token_clean"] = cleaned
        g["canonical_token_class"] = token_classes
        g["canonical_token_key"] = canonical_keys
        g["canonical_token_display"] = display_tokens

        top = g[g["in_top10_bank"]].copy()
        core = g[g["in_core50"]].copy()

        candidate_chunks.append(top)
        core_chunks.append(core)

        top_mass = (
            float(top["positive_mediation"].sum() / total_pos)
            if total_pos > EPS
            else np.nan
        )
        n_top = int(len(top))
        n_top_text = int(top["is_text"].sum())
        n_top_visual = int(top["is_visual"].sum())
        n_core_text = int(core["is_text"].sum())
        n_core_visual = int(core["is_visual"].sum())

        sample_rows.append(
            {
                "sid": int(sid),
                "relation": str(g["relation"].iloc[0]),
                "k36_available": int(len(g)),
                "positive_mass_total_k36": total_pos,
                "core50_size": int(core_size),
                "core50_within_top10": bool(core_size <= topk)
                if core_size > 0
                else False,
                "top10_count": n_top,
                "top10_positive_mass_fraction": top_mass,
                "top10_text_count": n_top_text,
                "top10_visual_count": n_top_visual,
                "core50_text_count": n_core_text,
                "core50_visual_count": n_core_visual,
            }
        )

    candidates = (
        pd.concat(candidate_chunks, ignore_index=True)
        if candidate_chunks
        else pd.DataFrame()
    )
    core = (
        pd.concat(core_chunks, ignore_index=True)
        if core_chunks
        else pd.DataFrame()
    )
    sample_df = pd.DataFrame(sample_rows)

    return candidates, core, sample_df


def add_fractions(df, count_col, denom):
    if len(df):
        df["fraction"] = df[count_col] / float(denom) if denom else np.nan
    return df


def layer_summary(text_df, total_text, n_samples):
    if not len(text_df):
        return pd.DataFrame()

    out = (
        text_df.groupby("source_layer", as_index=False)
        .agg(
            candidate_count=("sid", "size"),
            sample_presence_N=("sid", "nunique"),
            mean_rank=("oracle_rank_k36", "mean"),
            median_rank=("oracle_rank_k36", "median"),
            mean_mediation=("mediation", "mean"),
            median_mediation=("mediation", "median"),
        )
        .sort_values("candidate_count", ascending=False)
    )
    out["fraction_of_text_candidates"] = (
        out["candidate_count"] / float(total_text)
        if total_text
        else np.nan
    )
    out["sample_presence_fraction"] = (
        out["sample_presence_N"] / float(n_samples)
        if n_samples
        else np.nan
    )
    return out


def token_summary(text_df, total_text, n_samples):
    if not len(text_df):
        return pd.DataFrame()

    out = (
        text_df.groupby(
            [
                "canonical_token_class",
                "canonical_token_key",
                "canonical_token_display",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            candidate_count=("sid", "size"),
            sample_presence_N=("sid", "nunique"),
            layer_presence_N=("source_layer", "nunique"),
            mean_rank=("oracle_rank_k36", "mean"),
            median_rank=("oracle_rank_k36", "median"),
            mean_mediation=("mediation", "mean"),
            median_mediation=("mediation", "median"),
        )
        .sort_values(
            ["candidate_count", "sample_presence_N"],
            ascending=[False, False],
        )
    )
    out["fraction_of_text_candidates"] = (
        out["candidate_count"] / float(total_text)
        if total_text
        else np.nan
    )
    out["sample_presence_fraction"] = (
        out["sample_presence_N"] / float(n_samples)
        if n_samples
        else np.nan
    )
    return out


def layer_role_summary(text_df, total_text):
    if not len(text_df):
        return pd.DataFrame()

    out = (
        text_df.groupby(
            ["source_layer", "canonical_token_class"],
            as_index=False,
        )
        .agg(
            candidate_count=("sid", "size"),
            sample_presence_N=("sid", "nunique"),
            mean_rank=("oracle_rank_k36", "mean"),
            mean_mediation=("mediation", "mean"),
        )
        .sort_values(
            ["candidate_count", "source_layer"],
            ascending=[False, True],
        )
    )
    out["fraction_of_text_candidates"] = (
        out["candidate_count"] / float(total_text)
        if total_text
        else np.nan
    )
    return out


def layer_token_matrix(text_df):
    if not len(text_df):
        return pd.DataFrame()

    counts = (
        text_df.groupby(
            ["canonical_token_key", "source_layer"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )
    piv = counts.pivot(
        index="canonical_token_key",
        columns="source_layer",
        values="count",
    ).fillna(0)

    piv.columns = [f"L{int(c)}" for c in piv.columns]
    piv["total"] = piv.sum(axis=1)
    piv = piv.sort_values("total", ascending=False).reset_index()
    return piv


def rank_role_summary(candidates):
    if not len(candidates):
        return pd.DataFrame()

    return (
        candidates.groupby(
            ["oracle_rank_k36", "broad_category"],
            as_index=False,
        )
        .agg(
            candidate_count=("sid", "size"),
            sample_presence_N=("sid", "nunique"),
        )
        .sort_values(
            ["oracle_rank_k36", "candidate_count"],
            ascending=[True, False],
        )
    )


def write_jsonl(path, candidates, sample_df):
    sample_lookup = sample_df.set_index("sid").to_dict("index")

    with Path(path).open("w", encoding="utf-8") as f:
        for sid, g in candidates.groupby("sid", sort=True):
            meta = sample_lookup[int(sid)]
            rows = []
            for r in g.sort_values("oracle_rank_k36").itertuples():
                rows.append(
                    {
                        "rank": int(r.oracle_rank_k36),
                        "source_layer": int(r.source_layer),
                        "position": int(r.position),
                        "token_id": (
                            None
                            if not hasattr(r, "token_id")
                            or pd.isna(r.token_id)
                            else int(r.token_id)
                        ),
                        "token": str(r.token),
                        "token_clean": str(r.token_clean),
                        "category": str(r.category),
                        "broad_category": str(r.broad_category),
                        "canonical_token_class": str(
                            r.canonical_token_class
                        ),
                        "canonical_token_key": str(
                            r.canonical_token_key
                        ),
                        "mediation": float(r.mediation),
                        "cum_positive_mass_k36": (
                            None
                            if pd.isna(r.cum_positive_mass_k36)
                            else float(r.cum_positive_mass_k36)
                        ),
                        "in_core50": bool(r.in_core50),
                        "is_text": bool(r.is_text),
                    }
                )

            obj = {
                "sid": int(sid),
                "relation": str(meta["relation"]),
                "core50_size": int(meta["core50_size"]),
                "top10_positive_mass_fraction": (
                    None
                    if pd.isna(meta["top10_positive_mass_fraction"])
                    else float(meta["top10_positive_mass_fraction"])
                ),
                "candidates": rows,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def render_summary(
    candidates,
    core,
    sample_df,
    top_layer,
    top_token,
    core_layer,
    core_token,
    a,
):
    n = int(sample_df["sid"].nunique())
    n_cand = len(candidates)
    n_text = int(candidates["is_text"].sum()) if n_cand else 0
    n_vis = int(candidates["is_visual"].sum()) if n_cand else 0

    core_sizes = sample_df["core50_size"].to_numpy(float)
    finite_core = core_sizes[np.isfinite(core_sizes) & (core_sizes > 0)]

    lines = []
    lines.append("=" * 150)
    lines.append("ORACLE CORE50 -> TOP10 CANDIDATE BANK (ALL-440)")
    lines.append("=" * 150)
    lines.append(
        f"N={n} | source K={a.k_source} | exported TopK={a.topk} "
        f"| mass threshold={a.mass_threshold:.2f}"
    )
    lines.append(
        "Definition: positive global_unique K36 -> sort by writer mediation -> "
        "Core50 is shortest prefix reaching 50% K36 positive mediation mass."
    )
    lines.append("")

    if len(finite_core):
        lines.append(
            "Core50 size: "
            f"mean={np.mean(finite_core):.2f} "
            f"median={np.median(finite_core):.1f} "
            f"p90={np.percentile(finite_core, 90):.1f} "
            f"max={np.max(finite_core):.0f}"
        )

    within = sample_df["core50_within_top10"].astype(bool)
    lines.append(
        f"Core50 fully contained in Top{a.topk}: "
        f"{int(within.sum())}/{n} = {within.mean():.4f}"
    )
    lines.append(
        f"Mean positive mediation mass covered by Top{a.topk}: "
        f"{sample_df['top10_positive_mass_fraction'].mean():.4f}"
    )
    lines.append(
        f"Top{a.topk} candidates: total={n_cand} | "
        f"text={n_text} ({n_text/max(n_cand,1):.3f}) | "
        f"visual={n_vis} ({n_vis/max(n_cand,1):.3f})"
    )
    lines.append("")

    lines.append("TEXT TOP10: LAYER DISTRIBUTION")
    lines.append("-" * 150)
    if len(top_layer):
        lines.append(
            top_layer.head(20).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    else:
        lines.append("(no text candidates)")
    lines.append("")

    lines.append("TEXT TOP10: MOST FREQUENT CANONICAL TOKENS")
    lines.append("-" * 150)
    if len(top_token):
        show = [
            "canonical_token_class",
            "canonical_token_key",
            "candidate_count",
            "sample_presence_N",
            "layer_presence_N",
            "fraction_of_text_candidates",
            "mean_rank",
        ]
        lines.append(
            top_token.head(40)[show].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    else:
        lines.append("(no text candidates)")
    lines.append("")

    lines.append("TEXT CORE50: LAYER DISTRIBUTION")
    lines.append("-" * 150)
    if len(core_layer):
        lines.append(
            core_layer.head(20).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    else:
        lines.append("(no text Core50 candidates)")
    lines.append("")

    lines.append("TEXT CORE50: MOST FREQUENT CANONICAL TOKENS")
    lines.append("-" * 150)
    if len(core_token):
        show = [
            "canonical_token_class",
            "canonical_token_key",
            "candidate_count",
            "sample_presence_N",
            "layer_presence_N",
            "fraction_of_text_candidates",
            "mean_rank",
        ]
        lines.append(
            core_token.head(40)[show].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    else:
        lines.append("(no text Core50 candidates)")

    return "\n".join(lines) + "\n"


def main():
    a = parse_args()

    if a.topk <= 0:
        raise ValueError("--topk must be > 0")
    if not (0 < a.mass_threshold <= 1):
        raise ValueError("--mass-threshold must be in (0,1]")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_path, q = load_selected(a)
    q, med_path = maybe_merge_token_id(a.run_dir, q)

    n_sid = int(q["sid"].nunique())
    if a.require_n and n_sid != int(a.require_n):
        raise RuntimeError(
            f"Expected N={a.require_n} unique SIDs, got {n_sid}. "
            "Check --run-dir / --bundle / --k-source."
        )

    candidates, core, sample_df = build_bank(
        q,
        topk=a.topk,
        mass_threshold=a.mass_threshold,
        positive_eps=a.positive_eps,
    )

    sample_df, baseline_path = maybe_merge_baseline(
        a.run_dir,
        sample_df,
    )

    # Stable exact-state key for later spatial-head matching.
    for df in (candidates, core):
        if len(df):
            df["state_key"] = (
                "L"
                + df["source_layer"].astype(str)
                + ":P"
                + df["position"].astype(str)
            )
            # Head layer that immediately consumes this block-output state.
            df["aligned_head_layer"] = df["source_layer"] + 1

    # Text-only views.
    text_top = candidates[candidates["is_text"]].copy()
    text_core = core[core["is_text"]].copy()

    n_samples = int(sample_df["sid"].nunique())

    top_layer = layer_summary(
        text_top,
        total_text=len(text_top),
        n_samples=n_samples,
    )
    top_token = token_summary(
        text_top,
        total_text=len(text_top),
        n_samples=n_samples,
    )
    top_layer_role = layer_role_summary(
        text_top,
        total_text=len(text_top),
    )
    top_matrix = layer_token_matrix(text_top)

    core_layer = layer_summary(
        text_core,
        total_text=len(text_core),
        n_samples=n_samples,
    )
    core_token = token_summary(
        text_core,
        total_text=len(text_core),
        n_samples=n_samples,
    )
    core_layer_role = layer_role_summary(
        text_core,
        total_text=len(text_core),
    )
    core_matrix = layer_token_matrix(text_core)

    rank_role = rank_role_summary(candidates)

    # Save candidate bank.
    candidates.sort_values(
        ["sid", "oracle_rank_k36"]
    ).to_csv(
        outdir / "top10_candidates_all440.csv",
        index=False,
    )
    core.sort_values(
        ["sid", "oracle_rank_k36"]
    ).to_csv(
        outdir / "core50_candidates_all440.csv",
        index=False,
    )
    sample_df.sort_values("sid").to_csv(
        outdir / "sample_core50_summary.csv",
        index=False,
    )

    write_jsonl(
        outdir / "top10_candidates_by_sample.jsonl",
        candidates,
        sample_df,
    )

    # Text statistics.
    top_layer.to_csv(
        outdir / "text_top10_layer_summary.csv",
        index=False,
    )
    top_token.to_csv(
        outdir / "text_top10_token_summary.csv",
        index=False,
    )
    top_layer_role.to_csv(
        outdir / "text_top10_layer_role_summary.csv",
        index=False,
    )
    top_matrix.to_csv(
        outdir / "text_top10_layer_token_counts.csv",
        index=False,
    )

    core_layer.to_csv(
        outdir / "text_core50_layer_summary.csv",
        index=False,
    )
    core_token.to_csv(
        outdir / "text_core50_token_summary.csv",
        index=False,
    )
    core_layer_role.to_csv(
        outdir / "text_core50_layer_role_summary.csv",
        index=False,
    )
    core_matrix.to_csv(
        outdir / "text_core50_layer_token_counts.csv",
        index=False,
    )

    rank_role.to_csv(
        outdir / "top10_rank_role_summary.csv",
        index=False,
    )

    summary = render_summary(
        candidates,
        core,
        sample_df,
        top_layer,
        top_token,
        core_layer,
        core_token,
        a,
    )
    (outdir / "analysis_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    metadata = {
        "run_dir": str(Path(a.run_dir)),
        "selected_tokens_path": str(selected_path),
        "mediation_tokens_path": (
            str(med_path) if med_path is not None else None
        ),
        "baseline_path": (
            str(baseline_path) if baseline_path is not None else None
        ),
        "source_bundle": a.bundle,
        "source_k": int(a.k_source),
        "topk_exported": int(a.topk),
        "mass_threshold": float(a.mass_threshold),
        "condition": a.condition,
        "selection_strategy": a.selection_strategy,
        "N_samples": n_samples,
        "source_layers": sorted(
            int(x) for x in candidates["source_layer"].unique()
        )
        if len(candidates)
        else [],
        "candidate_definition": (
            "Top-K prefix of oracle GT-writer positive global_unique K36 "
            "ranked by Real-Gray x writer-gradient mediation"
        ),
        "core50_definition": (
            "Shortest K36 prefix reaching requested fraction of positive "
            "mediation mass, matching analyze_dynamic_k36_causal_core_v1.py"
        ),
        "future_spatial_head_alignment": (
            "decoder block-output source_layer L is consumed by attention "
            "head layer L+1"
        ),
        "text_statistics": (
            "visual excluded; subject/reference canonicalized across samples; "
            "relation words canonicalized by word; other text grouped by "
            "clean tokenizer token"
        ),
        "note": (
            "Pure post-processing. No new model forward/gradient/generation."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(summary)
    print("Saved:")
    for name in [
        "top10_candidates_all440.csv",
        "core50_candidates_all440.csv",
        "top10_candidates_by_sample.jsonl",
        "sample_core50_summary.csv",
        "text_top10_layer_summary.csv",
        "text_top10_token_summary.csv",
        "text_top10_layer_token_counts.csv",
        "text_top10_layer_role_summary.csv",
        "text_core50_layer_summary.csv",
        "text_core50_token_summary.csv",
        "text_core50_layer_token_counts.csv",
        "text_core50_layer_role_summary.csv",
        "top10_rank_role_summary.csv",
        "analysis_summary.txt",
        "metadata.json",
    ]:
        print(" ", outdir / name)


if __name__ == "__main__":
    main()
