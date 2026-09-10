#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process self-selected Top-K positions and summarize what token types they
focus on for LEFT / RIGHT / ON / UNDER.

No VLM forward pass is needed.

Expected inputs:
  <selection-dir>/selected_positions.csv
  <run-dir>/mediation_tokens.csv
  <run-dir>/baseline.csv

Typical use:
python analyze_self_k7_token_focus_by_relation_v1.py \
  --selection-dir output/qwen3b_self_topk_coverage_alpha1_N80 \
  --run-dir output/qwen3b_coco_dynamic_L20_26_K36_all440 \
  --k 7 \
  --selector attn_ensemble \
  --output-dir output/qwen3b_self_k7_token_focus_N80
"""

from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd


REL_ORDER = ["left", "right", "on", "under"]


def canon_rel(x):
    s = str(x).strip().lower()
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "on": "on", "above": "on", "over": "on", "top": "on",
        "under": "under", "below": "under", "beneath": "under", "bottom": "under",
    }
    return table.get(s, s)


def clean_token(x):
    s = str(x)
    s = s.replace("\\n", " ").replace("Ġ", " ").replace("▁", " ")
    return re.sub(r"\s+", " ", s).strip()


def relation_word_class(tok):
    s = clean_token(tok).lower()
    # remove common punctuation/brackets while preserving words
    words = re.findall(r"[a-z]+", s)
    if not words:
        return ""
    # For BPE pieces, exact single word is most reliable.
    joined = " ".join(words)
    if joined in {"left", "left of"} or words == ["left"]:
        return "left"
    if joined in {"right", "right of"} or words == ["right"]:
        return "right"
    if joined in {"above", "on", "over"} or words in (["above"], ["on"], ["over"]):
        return "on"
    if joined in {"under", "below", "beneath"} or words in (["under"], ["below"], ["beneath"]):
        return "under"
    return ""


def write_csv(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selection-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--selector", default="attn_ensemble")
    p.add_argument("--top-token-n", type=int, default=15)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def count_share(df, group_cols, value_col, value_name):
    c = (
        df.groupby(group_cols + [value_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        df.groupby(group_cols, dropna=False)
        .size()
        .rename("total")
        .reset_index()
    )
    c = c.merge(totals, on=group_cols, how="left")
    c["share"] = c["count"] / c["total"]
    c = c.rename(columns={value_col: value_name})
    return c


def main():
    a = parse_args()
    selection_dir = Path(a.selection_dir)
    run_dir = Path(a.run_dir)
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    spath = selection_dir / "selected_positions.csv"
    mpath = run_dir / "mediation_tokens.csv"
    bpath = run_dir / "baseline.csv"

    for p in [spath, mpath, bpath]:
        if not p.exists():
            raise FileNotFoundError(p)

    selected = pd.read_csv(spath)
    med = pd.read_csv(mpath)
    baseline = pd.read_csv(bpath)

    # Normalize keys.
    for df in [selected, med, baseline]:
        df["sid"] = pd.to_numeric(df["sid"], errors="raise").astype(int)

    selected["K"] = pd.to_numeric(selected["K"], errors="raise").astype(int)
    selected["rank"] = pd.to_numeric(selected["rank"], errors="raise").astype(int)
    selected["position"] = pd.to_numeric(selected["position"], errors="raise").astype(int)
    if "peak_selector_layer" in selected.columns:
        selected["peak_selector_layer"] = pd.to_numeric(
            selected["peak_selector_layer"], errors="coerce"
        )

    # Exact requested K/selector only.
    selected = selected[
        (selected["K"] == int(a.k))
        & (selected["selector"].astype(str) == str(a.selector))
    ].copy()
    if not len(selected):
        avail = (
            pd.read_csv(spath)[["selector", "K"]]
            .drop_duplicates()
            .sort_values(["selector", "K"])
        )
        raise RuntimeError(
            f"No rows for selector={a.selector!r}, K={a.k}.\nAvailable:\n"
            + avail.to_string(index=False)
        )

    # One token metadata row per (sid, position). Token identity/category is the
    # same across L20..L26; use the first non-null representative.
    med["position"] = pd.to_numeric(med["position"], errors="raise").astype(int)
    med["source_layer"] = pd.to_numeric(med["source_layer"], errors="coerce")
    if "relation" in med.columns:
        med["relation"] = med["relation"].map(canon_rel)

    meta_cols = ["sid", "position"]
    for c in ["token_id", "token", "category", "broad_category", "relation"]:
        if c in med.columns:
            meta_cols.append(c)

    token_meta = (
        med[meta_cols]
        .sort_values(["sid", "position"])
        .drop_duplicates(["sid", "position"], keep="first")
    )

    # Ground-truth relation from baseline, fallback to mediation table.
    gt_col = None
    for c in ["gt", "relation", "gold", "gold_relation"]:
        if c in baseline.columns:
            gt_col = c
            break
    if gt_col is not None:
        gt = baseline[["sid", gt_col]].copy()
        gt["gt_relation"] = gt[gt_col].map(canon_rel)
        gt = gt[["sid", "gt_relation"]].drop_duplicates("sid")
    else:
        gt = (
            token_meta[["sid", "relation"]]
            .rename(columns={"relation": "gt_relation"})
            .drop_duplicates("sid")
        )

    x = selected.merge(token_meta, on=["sid", "position"], how="left", suffixes=("", "_med"))
    x = x.merge(gt, on="sid", how="left")

    # Prefer selected broad_category if present because it corresponds to the
    # selector's peak layer row; otherwise use mediation metadata.
    if "broad_category_med" in x.columns:
        if "broad_category" not in x.columns:
            x["broad_category"] = x["broad_category_med"]
        else:
            x["broad_category"] = x["broad_category"].fillna(x["broad_category_med"])

    x["gt_relation"] = x["gt_relation"].map(canon_rel)
    x["token_clean"] = x["token"].map(clean_token) if "token" in x.columns else ""
    x["relation_word"] = x["token_clean"].map(relation_word_class)
    x["is_relation_word"] = x["relation_word"].ne("")
    x["is_gt_relation_word"] = (
        x["is_relation_word"] & (x["relation_word"] == x["gt_relation"])
    )
    x["is_other_relation_word"] = (
        x["is_relation_word"] & (x["relation_word"] != x["gt_relation"])
    )

    # Keep canonical relation order where possible.
    x = x[x["gt_relation"].isin(REL_ORDER)].copy()

    # ------------------------------------------------------------------
    # 1) broad category distribution by relation
    # ------------------------------------------------------------------
    broad = count_share(x, ["gt_relation"], "broad_category", "broad_category")
    broad["rel_order"] = broad["gt_relation"].map({r:i for i,r in enumerate(REL_ORDER)})
    broad = broad.sort_values(["rel_order", "count"], ascending=[True, False]).drop(columns="rel_order")
    write_csv(outdir / "broad_category_by_relation.csv", broad)

    # Fine category if available.
    if "category" in x.columns:
        fine = count_share(x, ["gt_relation"], "category", "category")
        fine["rel_order"] = fine["gt_relation"].map({r:i for i,r in enumerate(REL_ORDER)})
        fine = fine.sort_values(["rel_order", "count"], ascending=[True, False]).drop(columns="rel_order")
        write_csv(outdir / "fine_category_by_relation.csv", fine)
    else:
        fine = pd.DataFrame()

    # ------------------------------------------------------------------
    # 2) exact text tokens by relation
    # ------------------------------------------------------------------
    tok = (
        x.groupby(["gt_relation", "token_clean"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    tok["share_of_selected"] = tok["count"] / tok.groupby("gt_relation")["count"].transform("sum")
    tok["rel_order"] = tok["gt_relation"].map({r:i for i,r in enumerate(REL_ORDER)})
    tok = tok.sort_values(["rel_order", "count", "token_clean"], ascending=[True, False, True]).drop(columns="rel_order")
    write_csv(outdir / "exact_token_by_relation.csv", tok)

    # ------------------------------------------------------------------
    # 3) selected relation-word confusion matrix
    # ------------------------------------------------------------------
    relword = x[x["is_relation_word"]].copy()
    if len(relword):
        relword_mat = pd.crosstab(
            relword["gt_relation"],
            relword["relation_word"],
            dropna=False,
        ).reindex(index=REL_ORDER, columns=REL_ORDER, fill_value=0)
        relword_mat.insert(0, "gt_relation", relword_mat.index)
        relword_mat.to_csv(outdir / "relation_word_matrix.csv", index=False)

        relword_summary = (
            x.groupby("gt_relation")
            .agg(
                N_selected=("position", "size"),
                relation_word_count=("is_relation_word", "sum"),
                correct_relation_word_count=("is_gt_relation_word", "sum"),
                other_relation_word_count=("is_other_relation_word", "sum"),
            )
            .reset_index()
        )
        relword_summary["relation_word_share"] = (
            relword_summary["relation_word_count"] / relword_summary["N_selected"]
        )
        relword_summary["correct_relation_word_share_all_selected"] = (
            relword_summary["correct_relation_word_count"] / relword_summary["N_selected"]
        )
        relword_summary["correct_given_relation_word"] = np.where(
            relword_summary["relation_word_count"] > 0,
            relword_summary["correct_relation_word_count"] / relword_summary["relation_word_count"],
            np.nan,
        )
        write_csv(outdir / "relation_word_summary.csv", relword_summary)
    else:
        relword_mat = pd.DataFrame()
        relword_summary = pd.DataFrame()

    # ------------------------------------------------------------------
    # 4) role/category by rank 1..K, per relation
    # ------------------------------------------------------------------
    rank_role = (
        x.groupby(["gt_relation", "rank", "broad_category"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    rank_tot = (
        x.groupby(["gt_relation", "rank"])
        .size()
        .rename("total")
        .reset_index()
    )
    rank_role = rank_role.merge(rank_tot, on=["gt_relation", "rank"], how="left")
    rank_role["share"] = rank_role["count"] / rank_role["total"]
    write_csv(outdir / "rank_broad_category_by_relation.csv", rank_role)

    # ------------------------------------------------------------------
    # 5) selector peak-layer distribution by relation
    # ------------------------------------------------------------------
    if "peak_selector_layer" in x.columns:
        layer = count_share(
            x.dropna(subset=["peak_selector_layer"]),
            ["gt_relation"],
            "peak_selector_layer",
            "peak_selector_layer",
        )
        layer = layer.sort_values(["gt_relation", "peak_selector_layer"])
        write_csv(outdir / "peak_layer_by_relation.csv", layer)
    else:
        layer = pd.DataFrame()

    # ------------------------------------------------------------------
    # 6) Core50 hit rate by relation (if selected file has it)
    # ------------------------------------------------------------------
    if "is_core50_position" in x.columns:
        x["is_core50_position"] = pd.to_numeric(
            x["is_core50_position"], errors="coerce"
        ).fillna(0).astype(int)
        corehit = (
            x.groupby("gt_relation")
            .agg(
                N_selected=("position", "size"),
                core50_hits=("is_core50_position", "sum"),
            )
            .reset_index()
        )
        corehit["core50_hit_share_selected"] = (
            corehit["core50_hits"] / corehit["N_selected"]
        )
        write_csv(outdir / "core50_hit_by_relation.csv", corehit)
    else:
        corehit = pd.DataFrame()

    write_csv(outdir / "selected_k7_enriched.csv", x)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 110)
    lines.append(f"SELF-SELECTED TOP-{a.k} TOKEN FOCUS BY RELATION — selector={a.selector}")
    lines.append("=" * 110)
    lines.append(
        f"N samples={x['sid'].nunique()} | selected rows={len(x)} "
        f"(expected about {a.k} per sample)"
    )
    lines.append("")

    for rel in REL_ORDER:
        g = x[x["gt_relation"] == rel]
        if not len(g):
            continue
        lines.append("-" * 110)
        lines.append(f"{rel.upper()}  samples={g['sid'].nunique()}  selected={len(g)}")
        lines.append("-" * 110)

        # Broad roles.
        bc = (
            g["broad_category"]
            .fillna("NA")
            .astype(str)
            .value_counts()
        )
        top_bc = ", ".join(
            f"{name}={cnt} ({cnt/len(g):.1%})"
            for name, cnt in bc.head(8).items()
        )
        lines.append("broad roles: " + top_bc)

        # Top exact tokens.
        tc = (
            g["token_clean"]
            .replace("", np.nan)
            .dropna()
            .value_counts()
        )
        top_tc = ", ".join(
            f"{tok!r}={cnt}"
            for tok, cnt in tc.head(a.top_token_n).items()
        )
        lines.append("top exact tokens: " + (top_tc if top_tc else "(token text unavailable)"))

        # Relation-word diagnostic.
        nrel = int(g["is_relation_word"].sum())
        ngt = int(g["is_gt_relation_word"].sum())
        nother = int(g["is_other_relation_word"].sum())
        lines.append(
            f"relation words: any={nrel}/{len(g)} ({nrel/len(g):.1%}), "
            f"GT-matching={ngt}/{len(g)} ({ngt/len(g):.1%}), "
            f"other relation words={nother}/{len(g)} ({nother/len(g):.1%})"
        )

        if "is_core50_position" in g.columns:
            nh = int(g["is_core50_position"].sum())
            lines.append(
                f"oracle Core50 hits among selected: {nh}/{len(g)} ({nh/len(g):.1%})"
            )

        # Rank-1 pattern.
        g1 = g[g["rank"] == 1]
        if len(g1):
            rc = g1["broad_category"].fillna("NA").astype(str).value_counts()
            rt = g1["token_clean"].replace("", np.nan).dropna().value_counts()
            lines.append(
                "rank-1 roles: "
                + ", ".join(f"{n}={c}" for n, c in rc.head(5).items())
            )
            if len(rt):
                lines.append(
                    "rank-1 exact tokens: "
                    + ", ".join(f"{t!r}={c}" for t, c in rt.head(10).items())
                )
        lines.append("")

    lines.append("Interpretation checks:")
    lines.append(
        "1) If GT-matching relation words dominate, the self selector may partly exploit answer/relation-token cues."
    )
    lines.append(
        "2) If subject/reference/visual dominate and differ by relation, this supports relation-conditioned carrier structure beyond literal answer words."
    )
    lines.append(
        "3) Compare LEFT/RIGHT/ON/UNDER broad-role profiles; exact nouns are sample-dependent, so category-level structure is more meaningful than raw token frequency."
    )

    text = "\n".join(lines) + "\n"
    (outdir / "analysis_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
