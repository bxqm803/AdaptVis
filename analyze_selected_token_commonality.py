#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze whether oracle-selected middle tokens share a common structure across
LEFT / RIGHT / ON / UNDER.

Input
-----
The `selected_tokens.csv` produced by:
    eval_qwen_middle_token_amplify_multilayer_search_v2.py

Recommended use on the current best Qwen3B configuration:
    bundle = L22+L24+L26[global_unique]
    K      = 24 or 32
    condition = positive

This script DOES NOT run the model. It only analyzes the already-selected
concrete tokens.

Questions answered
------------------
1) Do the four relations select the same layer x token-role scaffold?
2) Which layer x role combinations are common to all four relations?
3) Are selection distributions for LEFT/RIGHT/ON/UNDER similar?
4) For selected relation-word tokens, are they usually:
       - the CORRECT relation word?
       - the OPPOSITE relation word?
       - an ORTHOGONAL relation word?
5) Which concrete token strings recur most often?
6) Is the commonality strong enough to motivate a relation-agnostic fixed mask?

Main outputs
------------
layer_role_by_relation.csv
    Mean per-sample count/share for every relation x layer x broad role.

common_layer_role.csv
    Commonality of each layer x role across the four directions:
        mean_share
        min_share       <- "common score"
        max_share
        std_share
        CV
        range
        relations_present

pairwise_relation_similarity.csv
    Cosine / Jensen-Shannon similarity between the four relation distributions
    over layer x role features.

relation_word_semantics.csv
    For relation-word tokens: correct / opposite / orthogonal / unknown.

layer_by_relation.csv
role_by_relation.csv
    Marginal layer and role distributions.

token_string_by_relation.csv
    Frequently selected literal token strings.

sample_overlap_summary.csv
    Per-sample structural signature statistics.

summary.txt
    Compact human-readable verdict.

Example
-------
python analyze_selected_token_commonality.py \
  --selected-csv output/qwen3b_middle_token_multilayer_search_v2/selected_tokens.csv \
  --bundle "L22+L24+L26[global_unique]" \
  --k 24 \
  --condition positive \
  --output-dir output/qwen3b_middle_token_commonality_K24 \
  --overwrite
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REL_ORDER = ["left", "right", "on", "under"]

REL_CANON = {
    "left": "left",
    "right": "right",
    "above": "on",
    "on": "on",
    "below": "under",
    "under": "under",
}

OPPOSITE = {
    "left": "right",
    "right": "left",
    "on": "under",
    "under": "on",
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--selected-csv", required=True)
    p.add_argument("--bundle", default="L22+L24+L26[global_unique]")
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--condition", default="positive")
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--top-token-strings",
        type=int,
        default=30,
        help="How many literal token strings to report per relation.",
    )
    p.add_argument(
        "--common-threshold",
        type=float,
        default=0.05,
        help=(
            "A layer-role feature is called broadly shared when its minimum "
            "selection share across the four relations is at least this value."
        ),
    )
    return p.parse_args()


def canon_relation(x):
    x = str(x).strip().lower()
    return REL_CANON.get(x, x)


def canon_relation_word_from_category(cat):
    cat = str(cat)
    if not cat.startswith("relation_word:"):
        return None
    word = cat.split(":", 1)[1].strip().lower()
    return REL_CANON.get(word, word)


def safe_cv(vals):
    vals = np.asarray(vals, dtype=np.float64)
    m = float(np.mean(vals))
    if abs(m) < 1e-12:
        return float("nan")
    return float(np.std(vals, ddof=0) / abs(m))


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def js_similarity(a, b):
    """
    Jensen-Shannon similarity = 1 - JSD/log(2), in [0,1].
    1 means identical distributions.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.maximum(a, 0)
    b = np.maximum(b, 0)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    a = a / a.sum()
    b = b / b.sum()
    m = 0.5 * (a + b)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-300))))

    jsd = 0.5 * kl(a, m) + 0.5 * kl(b, m)
    return float(1.0 - jsd / math.log(2.0))


def relation_word_semantic_role(gt, selected_word):
    gt = canon_relation(gt)
    selected_word = canon_relation(selected_word)
    if selected_word not in REL_ORDER or gt not in REL_ORDER:
        return "unknown"
    if selected_word == gt:
        return "correct"
    if selected_word == OPPOSITE[gt]:
        return "opposite"
    return "orthogonal"


def per_sample_feature_table(df, feature_cols, total_col_name="selected_total"):
    """
    Return rows:
      relation, sid, feature..., count, share
    Share is normalized by number of selected tokens in that sample/config.
    """
    total = (
        df.groupby(["relation", "sid"])
          .size()
          .rename(total_col_name)
          .reset_index()
    )

    cnt = (
        df.groupby(["relation", "sid"] + feature_cols)
          .size()
          .rename("count")
          .reset_index()
    )

    cnt = cnt.merge(total, on=["relation", "sid"], how="left")
    cnt["share"] = cnt["count"] / cnt[total_col_name]

    # Fill missing feature cells with zero for every sample.
    if len(feature_cols) == 1:
        features = [(x,) for x in sorted(df[feature_cols[0]].dropna().unique())]
    else:
        features = sorted(
            set(tuple(x) for x in df[feature_cols].drop_duplicates().itertuples(index=False, name=None))
        )

    sample_keys = df[["relation", "sid"]].drop_duplicates()
    rows = []
    lookup = {
        (r.relation, r.sid, *tuple(getattr(r, c) for c in feature_cols)):
            (float(r.count), float(r.share), float(getattr(r, total_col_name)))
        for r in cnt.itertuples(index=False)
    }

    totals_lookup = {
        (r.relation, r.sid): float(getattr(r, total_col_name))
        for r in total.itertuples(index=False)
    }

    for sk in sample_keys.itertuples(index=False):
        for feat in features:
            key = (sk.relation, sk.sid, *feat)
            count, share, _ = lookup.get(
                key, (0.0, 0.0, totals_lookup[(sk.relation, sk.sid)])
            )
            row = {
                "relation": sk.relation,
                "sid": sk.sid,
                total_col_name: totals_lookup[(sk.relation, sk.sid)],
                "count": count,
                "share": share,
            }
            for c, v in zip(feature_cols, feat):
                row[c] = v
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_relation_feature(per_sample, feature_cols):
    gcols = ["relation"] + feature_cols
    out = (
        per_sample.groupby(gcols, as_index=False)
        .agg(
            N=("sid", "nunique"),
            mean_count=("count", "mean"),
            mean_share=("share", "mean"),
            std_share=("share", "std"),
            median_share=("share", "median"),
            present_rate=("count", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
    )
    out["std_share"] = out["std_share"].fillna(0.0)
    return out


def layer_role_commonality(layer_role_rel):
    features = sorted(
        set(
            (int(r.source_layer), str(r.broad_category))
            for r in layer_role_rel.itertuples(index=False)
        )
    )
    rows = []
    for layer, role in features:
        rel_vals = {}
        rel_present = {}
        for rel in REL_ORDER:
            sub = layer_role_rel[
                (layer_role_rel["relation"] == rel)
                & (layer_role_rel["source_layer"] == layer)
                & (layer_role_rel["broad_category"] == role)
            ]
            if len(sub):
                rel_vals[rel] = float(sub.iloc[0]["mean_share"])
                rel_present[rel] = float(sub.iloc[0]["present_rate"])
            else:
                rel_vals[rel] = 0.0
                rel_present[rel] = 0.0

        vals = [rel_vals[r] for r in REL_ORDER]
        pres = [rel_present[r] for r in REL_ORDER]
        rows.append({
            "source_layer": layer,
            "broad_category": role,
            **{f"share_{r}": rel_vals[r] for r in REL_ORDER},
            **{f"present_{r}": rel_present[r] for r in REL_ORDER},
            "mean_share": float(np.mean(vals)),
            "min_share": float(np.min(vals)),
            "max_share": float(np.max(vals)),
            "std_share": float(np.std(vals)),
            "cv_share": safe_cv(vals),
            "range_share": float(np.max(vals) - np.min(vals)),
            "mean_present_rate": float(np.mean(pres)),
            "min_present_rate": float(np.min(pres)),
            "relations_present": int(sum(v > 0 for v in vals)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["min_share", "mean_share", "mean_present_rate"],
            ascending=[False, False, False],
        )
    return out


def pairwise_similarity(layer_role_rel):
    features = sorted(
        set(
            (int(r.source_layer), str(r.broad_category))
            for r in layer_role_rel.itertuples(index=False)
        )
    )
    vecs = {}
    for rel in REL_ORDER:
        sub = layer_role_rel[layer_role_rel["relation"] == rel]
        lookup = {
            (int(r.source_layer), str(r.broad_category)): float(r.mean_share)
            for r in sub.itertuples(index=False)
        }
        vecs[rel] = np.array([lookup.get(f, 0.0) for f in features], dtype=np.float64)

    rows = []
    for i, a in enumerate(REL_ORDER):
        for b in REL_ORDER[i + 1:]:
            va, vb = vecs[a], vecs[b]
            rows.append({
                "relation_a": a,
                "relation_b": b,
                "cosine_similarity": cosine(va, vb),
                "js_similarity": js_similarity(va, vb),
                "l1_distance": float(np.sum(np.abs(
                    va / max(va.sum(), 1e-12) -
                    vb / max(vb.sum(), 1e-12)
                ))),
            })
    return pd.DataFrame(rows)


def relation_word_analysis(df):
    rw = df[df["category"].astype(str).str.startswith("relation_word:")].copy()
    if not len(rw):
        return pd.DataFrame(), pd.DataFrame()

    rw["selected_relation_word"] = rw["category"].map(canon_relation_word_from_category)
    rw["semantic_role"] = [
        relation_word_semantic_role(gt, sw)
        for gt, sw in zip(rw["relation"], rw["selected_relation_word"])
    ]

    # Per-sample fractions among selected relation-word tokens.
    cnt = (
        rw.groupby(["relation", "sid", "semantic_role"])
          .size()
          .rename("count")
          .reset_index()
    )
    total = (
        rw.groupby(["relation", "sid"])
          .size()
          .rename("relation_word_total")
          .reset_index()
    )
    cnt = cnt.merge(total, on=["relation", "sid"], how="left")
    cnt["share_among_relation_words"] = cnt["count"] / cnt["relation_word_total"]

    # Zero-fill semantic roles for samples that have at least one relation word.
    semantics = ["correct", "opposite", "orthogonal", "unknown"]
    sample_keys = total[["relation", "sid", "relation_word_total"]]
    lookup = {
        (r.relation, r.sid, r.semantic_role):
            (float(r.count), float(r.share_among_relation_words))
        for r in cnt.itertuples(index=False)
    }
    rows = []
    for sk in sample_keys.itertuples(index=False):
        for sem in semantics:
            c, s = lookup.get((sk.relation, sk.sid, sem), (0.0, 0.0))
            rows.append({
                "relation": sk.relation,
                "sid": sk.sid,
                "relation_word_total": sk.relation_word_total,
                "semantic_role": sem,
                "count": c,
                "share_among_relation_words": s,
            })

    per_sample = pd.DataFrame(rows)

    summary = (
        per_sample.groupby(["relation", "semantic_role"], as_index=False)
        .agg(
            N=("sid", "nunique"),
            mean_count=("count", "mean"),
            mean_share=("share_among_relation_words", "mean"),
            present_rate=("count", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
    )
    return per_sample, summary


def token_string_stats(df, top_n):
    # Literal token strings are not expected to be universal, but this reveals
    # recurring relation/task tokens and common structural tokens.
    g = (
        df.groupby(["relation", "source_layer", "token", "broad_category"], as_index=False)
          .agg(
              count=("sid", "size"),
              sample_count=("sid", "nunique"),
              mean_mediation=("mediation", "mean"),
              median_mediation=("mediation", "median"),
          )
    )

    out = []
    for rel in REL_ORDER:
        sub = g[g["relation"] == rel].copy()
        sub = sub.sort_values(
            ["sample_count", "mean_mediation"],
            ascending=[False, False]
        ).head(top_n)
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sample_signature_stats(df):
    rows = []
    for (rel, sid), sub in df.groupby(["relation", "sid"]):
        counts = Counter(
            (int(r.source_layer), str(r.broad_category))
            for r in sub.itertuples(index=False)
        )
        k = len(sub)
        shares = {f"L{L}_{role}": c / max(k, 1) for (L, role), c in counts.items()}
        rows.append({
            "relation": rel,
            "sid": sid,
            "selected_total": k,
            "n_unique_layers": sub["source_layer"].nunique(),
            "n_unique_roles": sub["broad_category"].nunique(),
            **shares,
        })
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    in_path = Path(a.selected_csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    required = {
        "sid", "relation", "condition", "source_bundle", "k",
        "source_layer", "position", "token", "category",
        "broad_category", "mediation"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"selected_tokens.csv missing columns: {missing}")

    df["relation"] = df["relation"].map(canon_relation)
    df["source_layer"] = pd.to_numeric(df["source_layer"], errors="raise").astype(int)
    df["k"] = pd.to_numeric(df["k"], errors="raise").astype(int)
    df["mediation"] = pd.to_numeric(df["mediation"], errors="coerce")

    sub = df[
        (df["condition"].astype(str) == str(a.condition))
        & (df["source_bundle"].astype(str) == str(a.bundle))
        & (df["k"] == int(a.k))
        & (df["relation"].isin(REL_ORDER))
    ].copy()

    if not len(sub):
        avail = (
            df[["condition", "source_bundle", "k"]]
            .drop_duplicates()
            .sort_values(["condition", "source_bundle", "k"])
        )
        raise RuntimeError(
            "No rows match requested configuration.\n"
            f"Requested: condition={a.condition}, bundle={a.bundle}, k={a.k}\n"
            "Available configs include:\n"
            + avail.head(50).to_string(index=False)
        )

    # De-duplicate defensively in case a selection file was concatenated.
    unique_cols = [
        "sid", "relation", "condition", "source_bundle", "k",
        "source_layer", "position"
    ]
    before = len(sub)
    sub = sub.sort_values("mediation", ascending=False).drop_duplicates(
        unique_cols, keep="first"
    )
    deduped = before - len(sub)

    # Basic sample counts.
    relation_sample_counts = (
        sub[["relation", "sid"]]
        .drop_duplicates()
        .groupby("relation")
        .size()
        .reindex(REL_ORDER, fill_value=0)
    )

    # Layer x role.
    ps_lr = per_sample_feature_table(sub, ["source_layer", "broad_category"])
    lr = aggregate_relation_feature(ps_lr, ["source_layer", "broad_category"])
    lr.to_csv(outdir / "layer_role_by_relation.csv", index=False)

    common = layer_role_commonality(lr)
    common["broadly_shared"] = common["min_share"] >= float(a.common_threshold)
    common.to_csv(outdir / "common_layer_role.csv", index=False)

    sim = pairwise_similarity(lr)
    sim.to_csv(outdir / "pairwise_relation_similarity.csv", index=False)

    # Marginals.
    ps_layer = per_sample_feature_table(sub, ["source_layer"])
    layer_rel = aggregate_relation_feature(ps_layer, ["source_layer"])
    layer_rel.to_csv(outdir / "layer_by_relation.csv", index=False)

    ps_role = per_sample_feature_table(sub, ["broad_category"])
    role_rel = aggregate_relation_feature(ps_role, ["broad_category"])
    role_rel.to_csv(outdir / "role_by_relation.csv", index=False)

    # Relation-word semantics.
    rw_ps, rw_summary = relation_word_analysis(sub)
    if len(rw_ps):
        rw_ps.to_csv(outdir / "relation_word_semantics_per_sample.csv", index=False)
        rw_summary.to_csv(outdir / "relation_word_semantics.csv", index=False)

    # Literal token string recurrence.
    tok_stats = token_string_stats(sub, a.top_token_strings)
    tok_stats.to_csv(outdir / "token_string_by_relation.csv", index=False)

    # Sample signatures.
    sig = sample_signature_stats(sub)
    sig.to_csv(outdir / "sample_overlap_summary.csv", index=False)

    # Overall common features.
    shared = common[common["broadly_shared"]].copy()

    mean_cos = float(sim["cosine_similarity"].mean()) if len(sim) else float("nan")
    min_cos = float(sim["cosine_similarity"].min()) if len(sim) else float("nan")
    mean_js = float(sim["js_similarity"].mean()) if len(sim) else float("nan")
    min_js = float(sim["js_similarity"].min()) if len(sim) else float("nan")

    # A deliberately descriptive—not inferential—verdict.
    if np.isfinite(min_cos) and min_cos >= 0.90 and len(shared) >= 2:
        verdict = "STRONG_SHARED_SCAFFOLD"
    elif np.isfinite(min_cos) and min_cos >= 0.75 and len(shared) >= 1:
        verdict = "MODERATE_SHARED_SCAFFOLD"
    else:
        verdict = "WEAK_OR_RELATION_SPECIFIC_SCAFFOLD"

    lines = []
    lines.append("=" * 110)
    lines.append("SELECTED MIDDLE-TOKEN COMMONALITY ACROSS LEFT / RIGHT / ON / UNDER")
    lines.append("=" * 110)
    lines.append(f"input       : {in_path}")
    lines.append(f"condition   : {a.condition}")
    lines.append(f"bundle      : {a.bundle}")
    lines.append(f"K           : {a.k}")
    lines.append(f"rows        : {len(sub)} (deduplicated {deduped})")
    lines.append(
        "samples     : " +
        ", ".join(f"{r}={int(relation_sample_counts[r])}" for r in REL_ORDER)
    )
    lines.append("")

    lines.append("PAIRWISE RELATION DISTRIBUTION SIMILARITY (layer x role)")
    lines.append("-" * 110)
    if len(sim):
        for r in sim.itertuples(index=False):
            lines.append(
                f"{r.relation_a:>5s} vs {r.relation_b:<5s} | "
                f"cos={r.cosine_similarity:.4f} | "
                f"JS-sim={r.js_similarity:.4f} | "
                f"L1={r.l1_distance:.4f}"
            )
        lines.append(
            f"mean cosine={mean_cos:.4f}, min cosine={min_cos:.4f}, "
            f"mean JS-sim={mean_js:.4f}, min JS-sim={min_js:.4f}"
        )
    lines.append("")

    lines.append("MOST COMMON LAYER x ROLE FEATURES")
    lines.append("-" * 110)
    for r in common.head(20).itertuples(index=False):
        lines.append(
            f"L{int(r.source_layer):02d} {r.broad_category:<16s} | "
            f"left={r.share_left:.3f} right={r.share_right:.3f} "
            f"on={r.share_on:.3f} under={r.share_under:.3f} | "
            f"mean={r.mean_share:.3f} min/common={r.min_share:.3f} "
            f"CV={r.cv_share:.3f} present(min)={r.min_present_rate:.3f}"
        )
    lines.append("")

    if len(rw_summary):
        lines.append("RELATION-WORD SEMANTICS")
        lines.append("-" * 110)
        pivot = rw_summary.pivot(
            index="relation", columns="semantic_role", values="mean_share"
        ).fillna(0.0)
        for rel in REL_ORDER:
            if rel not in pivot.index:
                continue
            vals = pivot.loc[rel]
            lines.append(
                f"{rel:>5s} | "
                f"correct={float(vals.get('correct', 0.0)):.3f} "
                f"opposite={float(vals.get('opposite', 0.0)):.3f} "
                f"orthogonal={float(vals.get('orthogonal', 0.0)):.3f} "
                f"unknown={float(vals.get('unknown', 0.0)):.3f}"
            )
        lines.append("")

    lines.append("VERDICT")
    lines.append("-" * 110)
    lines.append(verdict)
    lines.append(
        f"shared features with min_share >= {a.common_threshold:.3f}: {len(shared)}"
    )
    if len(shared):
        for r in shared.head(12).itertuples(index=False):
            lines.append(
                f"  L{int(r.source_layer):02d} {r.broad_category:<16s} "
                f"common={r.min_share:.3f}, mean={r.mean_share:.3f}, CV={r.cv_share:.3f}"
            )
    lines.append("")
    lines.append(
        "Interpretation rule: high pairwise cosine/JS similarity plus high min_share and low CV "
        "supports a relation-invariant layer/role scaffold. This does NOT yet show that a fixed "
        "writer-free mask will preserve the generation gain; that requires a separate intervention."
    )

    report = "\n".join(lines)
    print(report)
    (outdir / "summary.txt").write_text(report, encoding="utf-8")

    print("\nSaved:")
    for name in [
        "summary.txt",
        "layer_role_by_relation.csv",
        "common_layer_role.csv",
        "pairwise_relation_similarity.csv",
        "layer_by_relation.csv",
        "role_by_relation.csv",
        "relation_word_semantics.csv",
        "token_string_by_relation.csv",
        "sample_overlap_summary.csv",
    ]:
        p = outdir / name
        if p.exists():
            print(" ", p)


if __name__ == "__main__":
    main()
