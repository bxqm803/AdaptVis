#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process sample-specific dynamic K24 selections.

This script DOES NOT run the VLM and DOES NOT choose new tokens.
It analyzes the tokens that each sample already selected for itself in
selected_tokens.csv from the dynamic writer-guided global_unique experiment.

Main outputs
------------
1) sample_layer_role_composition.csv
   Per-sample K24 layer x role composition.

2) correct_wrong_layer_role.csv
   Baseline-correct vs baseline-wrong comparison.

3) relation_layer_role.csv
   LEFT/RIGHT/ON/UNDER comparison and enrichment over the global distribution.

4) rank_cutoff_composition.csv
   Top1/4/8/12/24 composition, overall + by relation + by baseline correctness.

5) pair_cooccurrence.csv / triple_cooccurrence.csv
   Frequent layer-role motifs within the same sample, with support and lift.

6) jaccard_pairwise.csv / jaccard_summary.csv
   Cross-sample similarity of K24:
     - layer-role set Jaccard
     - layer-role multiset/composition Jaccard
     - layer+semantic-token set Jaccard

7) cluster_full_*.csv
   Unsupervised KMeans on sample K24 composition.

8) cluster_no_relation_words_*.csv
   Same clustering after removing relation-word motifs, to test whether any
   cluster structure is more than trivial left/right/above/below grouping.

9) top_concrete_tokens.csv
   Recurrent concrete carriers within layer/role/relation.

10) analysis_summary.txt
    Compact text report of the strongest patterns.

Repairability
-------------
If generation_conditions.csv exists, the script chooses the alpha with highest
overall edited accuracy for the requested dynamic-K24 configuration unless
--repair-alpha is supplied. Then each sample is labeled:
    W2C = baseline wrong -> edited correct
    C2W = baseline correct -> edited wrong
    preserved_correct
    remained_wrong

Example
-------
python analyze_dynamic_k24_patterns_v1.py \
  --run-dir output/qwen3b_coco_dynamic_k24_all440 \
  --bundle "L22+L24+L26[global_unique]" \
  --k 24 \
  --condition positive \
  --rank-cutoffs 1,4,8,12,24 \
  --n-clusters 0 \
  --output-dir output/qwen3b_coco_dynamic_k24_patterns \
  --overwrite

The script is generic in source layers. If a later run uses
L22+L23+L24+L25+L26[global_unique], just pass that bundle string instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise RuntimeError(
        "This analysis requires scikit-learn (sklearn). "
        f"Import failed with: {e}"
    )


REL_ORDER = ["left", "right", "on", "under"]
ROLE_ORDER = ["visual", "subject", "reference", "relation_words", "other_text"]


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--bundle",
        default="L22+L24+L26[global_unique]",
        help="Exact source_bundle label in selected_tokens.csv.",
    )
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--condition", default="positive")
    p.add_argument(
        "--selection-strategy",
        default="global_unique",
    )
    p.add_argument(
        "--rank-cutoffs",
        default="1,4,8,12,24",
    )
    p.add_argument(
        "--repair-alpha",
        type=float,
        default=None,
        help=(
            "Alpha used to define repairability from generation_conditions.csv. "
            "If omitted, choose the alpha with highest edited accuracy."
        ),
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help=(
            "KMeans k. 0 = choose k automatically by best silhouette over "
            "--cluster-k-min..--cluster-k-max."
        ),
    )
    p.add_argument("--cluster-k-min", type=int, default=2)
    p.add_argument("--cluster-k-max", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--pair-min-support",
        type=float,
        default=0.05,
        help="Minimum sample support for pair output.",
    )
    p.add_argument(
        "--triple-min-support",
        type=float,
        default=0.03,
        help="Minimum sample support for triple output.",
    )
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--max-triples", type=int, default=200)
    p.add_argument("--top-concrete-n", type=int, default=100)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def canon_rel(x):
    x = str(x).strip().lower()
    if x == "above":
        return "on"
    if x == "below":
        return "under"
    return x


def to_bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float) != 0
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def role_sort_key(role):
    try:
        return ROLE_ORDER.index(role)
    except ValueError:
        return len(ROLE_ORDER)


def relation_sort_key(rel):
    try:
        return REL_ORDER.index(rel)
    except ValueError:
        return len(REL_ORDER)


def semantic_signature(row):
    """
    Cross-sample semantic signature.

    We deliberately collapse object identities to <SUBJ>/<REF> and all visual
    positions to VISUAL. Relation words retain their lexical identity.
    Other text retains the token string.
    """
    layer = int(row["source_layer"])
    role = str(row["broad_category"])
    tok = str(row.get("token", ""))

    if role == "subject":
        unit = "<SUBJ>"
    elif role == "reference":
        unit = "<REF>"
    elif role == "visual":
        unit = "<VISUAL>"
    elif role == "relation_words":
        unit = tok.replace("Ġ", "").strip().lower()
    else:
        unit = tok.replace("\n", "\\n")

    return f"L{layer}:{role}:{unit}"


def layer_role_signature(row):
    return f"L{int(row['source_layer'])}:{str(row['broad_category'])}"


def validate_inputs(sel, baseline):
    required_sel = {
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
    missing = sorted(required_sel - set(sel.columns))
    if missing:
        raise RuntimeError(
            "selected_tokens.csv is missing required columns: "
            + ", ".join(missing)
        )

    required_base = {
        "sid",
        "gt",
        "baseline_prediction",
        "baseline_correct",
    }
    missing = sorted(required_base - set(baseline.columns))
    if missing:
        raise RuntimeError(
            "baseline.csv is missing required columns: "
            + ", ".join(missing)
        )


def load_and_filter(a):
    run_dir = Path(a.run_dir)

    selected_path = run_dir / "selected_tokens.csv"
    baseline_path = run_dir / "baseline.csv"
    generation_path = run_dir / "generation_conditions.csv"

    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)

    sel = pd.read_csv(selected_path)
    baseline = pd.read_csv(baseline_path)
    validate_inputs(sel, baseline)

    sel["sid"] = pd.to_numeric(sel["sid"], errors="raise").astype(int)
    sel["k"] = pd.to_numeric(sel["k"], errors="raise").astype(int)
    sel["rank"] = pd.to_numeric(sel["rank"], errors="raise").astype(int)
    sel["source_layer"] = pd.to_numeric(
        sel["source_layer"], errors="raise"
    ).astype(int)
    sel["position"] = pd.to_numeric(sel["position"], errors="raise").astype(int)
    sel["mediation"] = pd.to_numeric(sel["mediation"], errors="coerce")
    if "delta_h_norm" in sel.columns:
        sel["delta_h_norm"] = pd.to_numeric(
            sel["delta_h_norm"], errors="coerce"
        )
    if "grad_norm" in sel.columns:
        sel["grad_norm"] = pd.to_numeric(
            sel["grad_norm"], errors="coerce"
        )
    sel["relation"] = sel["relation"].map(canon_rel)

    filt = (
        (sel["condition"].astype(str) == str(a.condition))
        & (
            sel["selection_strategy"].astype(str)
            == str(a.selection_strategy)
        )
        & (sel["source_bundle"].astype(str) == str(a.bundle))
        & (sel["k"] == int(a.k))
    )
    sel = sel[filt].copy()

    if not len(sel):
        avail = (
            pd.read_csv(selected_path)[
                ["condition", "selection_strategy", "source_bundle", "k"]
            ]
            .drop_duplicates()
            .head(50)
        )
        raise RuntimeError(
            "No selected-token rows matched the requested configuration.\n"
            f"Requested: condition={a.condition!r}, "
            f"strategy={a.selection_strategy!r}, "
            f"bundle={a.bundle!r}, k={a.k}\n"
            "Available examples:\n"
            + avail.to_string(index=False)
        )

    # Keep one row per selected (sample, layer, position), highest M if duplicated.
    sel = (
        sel.sort_values(["sid", "rank", "mediation"], ascending=[True, True, False])
        .drop_duplicates(
            ["sid", "source_layer", "position"],
            keep="first",
        )
        .copy()
    )

    # Re-rank within each sample so downstream Top1/4/... is unambiguous.
    sel = sel.sort_values(
        ["sid", "mediation"],
        ascending=[True, False],
    )
    sel["analysis_rank"] = sel.groupby("sid").cumcount() + 1
    sel["layer_role"] = sel.apply(layer_role_signature, axis=1)
    sel["semantic_signature"] = sel.apply(semantic_signature, axis=1)

    baseline["sid"] = pd.to_numeric(
        baseline["sid"], errors="raise"
    ).astype(int)
    baseline["gt"] = baseline["gt"].map(canon_rel)
    baseline["baseline_correct"] = to_bool_series(
        baseline["baseline_correct"]
    )

    # Restrict baseline to samples represented in this selected config.
    sids = sorted(sel["sid"].unique())
    baseline = baseline[baseline["sid"].isin(sids)].copy()

    generation = None
    chosen_alpha = None
    if generation_path.exists():
        generation = pd.read_csv(generation_path)
        needed = {
            "sid",
            "condition",
            "selection_strategy",
            "source_bundle",
            "k",
            "alpha",
            "correct",
            "prediction",
        }
        if needed.issubset(generation.columns):
            generation["sid"] = pd.to_numeric(
                generation["sid"], errors="raise"
            ).astype(int)
            generation["k"] = pd.to_numeric(
                generation["k"], errors="raise"
            ).astype(int)
            generation["alpha"] = pd.to_numeric(
                generation["alpha"], errors="coerce"
            )
            generation["correct"] = to_bool_series(generation["correct"])

            gfilt = (
                (
                    generation["condition"].astype(str)
                    == f"middle_{a.condition}"
                )
                & (
                    generation["selection_strategy"].astype(str)
                    == str(a.selection_strategy)
                )
                & (
                    generation["source_bundle"].astype(str)
                    == str(a.bundle)
                )
                & (generation["k"] == int(a.k))
                & generation["sid"].isin(sids)
            )
            generation = generation[gfilt].copy()

            if len(generation):
                if a.repair_alpha is None:
                    alpha_acc = (
                        generation.groupby("alpha")["correct"]
                        .mean()
                        .sort_values(ascending=False)
                    )
                    chosen_alpha = float(alpha_acc.index[0])
                else:
                    chosen_alpha = float(a.repair_alpha)

                generation = generation[
                    np.isclose(generation["alpha"], chosen_alpha)
                ].copy()
            else:
                generation = None

    return sel, baseline, generation, chosen_alpha


def sample_composition(sel, baseline, generation):
    layers = sorted(sel["source_layer"].unique())
    roles = sorted(sel["broad_category"].unique(), key=role_sort_key)

    base_lookup = baseline.set_index("sid").to_dict("index")
    gen_lookup = (
        generation.set_index("sid").to_dict("index")
        if generation is not None and len(generation)
        else {}
    )

    rows = []

    for sid, g in sel.groupby("sid"):
        sid = int(sid)
        g = g.sort_values("analysis_rank")
        b = base_lookup.get(sid, {})
        e = gen_lookup.get(sid, {})

        baseline_correct = bool(b.get("baseline_correct", False))
        edited_correct = (
            bool(e.get("correct"))
            if sid in gen_lookup
            else None
        )

        if edited_correct is None:
            outcome = "unknown"
        elif (not baseline_correct) and edited_correct:
            outcome = "W2C"
        elif baseline_correct and (not edited_correct):
            outcome = "C2W"
        elif baseline_correct and edited_correct:
            outcome = "preserved_correct"
        else:
            outcome = "remained_wrong"

        row = {
            "sid": sid,
            "relation": canon_rel(b.get("gt", g["relation"].iloc[0])),
            "baseline_prediction": b.get("baseline_prediction", ""),
            "baseline_correct": baseline_correct,
            "edited_prediction": e.get("prediction", "") if sid in gen_lookup else "",
            "edited_correct": edited_correct,
            "repair_outcome": outcome,
            "n_selected": len(g),
            "mean_mediation": float(g["mediation"].mean()),
            "sum_mediation": float(g["mediation"].sum()),
            "top1_layer_role": g.iloc[0]["layer_role"] if len(g) else "",
            "top1_token": str(g.iloc[0]["token"]) if len(g) else "",
            "n_unique_layer_roles": int(g["layer_role"].nunique()),
        }

        total_m = float(g["mediation"].clip(lower=0).sum())

        for L in layers:
            gl = g[g["source_layer"] == L]
            row[f"count_L{L}"] = int(len(gl))
            row[f"frac_L{L}"] = safe_div(len(gl), len(g))

        for role in roles:
            gr = g[g["broad_category"] == role]
            row[f"count_role_{role}"] = int(len(gr))
            row[f"frac_role_{role}"] = safe_div(len(gr), len(g))

        for L in layers:
            for role in roles:
                unit = f"L{L}:{role}"
                gu = g[g["layer_role"] == unit]
                c = len(gu)
                m = float(gu["mediation"].clip(lower=0).sum())
                rr = float((1.0 / gu["analysis_rank"]).sum()) if c else 0.0
                row[f"count_{unit}"] = int(c)
                row[f"frac_{unit}"] = safe_div(c, len(g))
                row[f"rr_{unit}"] = rr
                row[f"mshare_{unit}"] = safe_div(m, total_m) if total_m > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows), layers, roles


def layer_role_group_summary(sel, sample_df, group_col, group_values=None):
    meta = sample_df[
        ["sid", "relation", "baseline_correct", "repair_outcome"]
    ].copy()
    x = sel.merge(meta, on="sid", how="left", suffixes=("", "_meta"))

    if group_values is None:
        group_values = sorted(x[group_col].dropna().unique())

    units = sorted(x["layer_role"].unique())
    rows = []

    for gv in group_values:
        g = x[x[group_col] == gv]
        n_samples = int(g["sid"].nunique())
        n_tokens = len(g)
        if not n_samples:
            continue

        for unit in units:
            u = g[g["layer_role"] == unit]
            sample_presence = int(u["sid"].nunique())
            rows.append({
                "group_col": group_col,
                "group": gv,
                "layer_role": unit,
                "N_samples": n_samples,
                "N_selected_tokens": n_tokens,
                "token_count": len(u),
                "token_fraction": safe_div(len(u), n_tokens),
                "sample_presence_N": sample_presence,
                "sample_presence_rate": safe_div(sample_presence, n_samples),
                "mean_count_when_present": (
                    safe_div(len(u), sample_presence)
                    if sample_presence
                    else 0.0
                ),
                "mean_rank": safe_mean(u["analysis_rank"]) if len(u) else float("nan"),
                "mean_mediation": safe_mean(u["mediation"]) if len(u) else float("nan"),
            })

    return pd.DataFrame(rows)


def relation_summary_with_enrichment(sel, sample_df):
    rel_df = layer_role_group_summary(
        sel,
        sample_df,
        "relation",
        REL_ORDER,
    )

    global_frac = (
        sel["layer_role"].value_counts(normalize=True).to_dict()
    )
    global_presence = {}
    N = sel["sid"].nunique()
    for unit, g in sel.groupby("layer_role"):
        global_presence[unit] = g["sid"].nunique() / N

    rel_df["global_token_fraction"] = rel_df["layer_role"].map(global_frac)
    rel_df["token_enrichment_vs_global"] = (
        rel_df["token_fraction"] / rel_df["global_token_fraction"]
    )
    rel_df["global_sample_presence_rate"] = rel_df["layer_role"].map(
        global_presence
    )
    rel_df["presence_enrichment_vs_global"] = (
        rel_df["sample_presence_rate"]
        / rel_df["global_sample_presence_rate"]
    )
    return rel_df


def rank_cutoff_summary(sel, sample_df, cutoffs):
    meta = sample_df[
        ["sid", "relation", "baseline_correct"]
    ].copy()
    x = sel.merge(meta, on="sid", how="left", suffixes=("", "_meta"))

    rows = []

    slices = [("overall", "all", x)]
    for rel in REL_ORDER:
        slices.append(("relation", rel, x[x["relation_meta"] == rel]))
    slices.append(
        ("baseline_status", "correct", x[x["baseline_correct"] == True])
    )
    slices.append(
        ("baseline_status", "wrong", x[x["baseline_correct"] == False])
    )

    for cutoff in cutoffs:
        for slice_type, slice_value, base_x in slices:
            g = base_x[base_x["analysis_rank"] <= cutoff]
            n_samples = int(g["sid"].nunique())
            n_tokens = len(g)
            if not n_samples or not n_tokens:
                continue

            for unit, u in g.groupby("layer_role"):
                rows.append({
                    "cutoff": cutoff,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "layer_role": unit,
                    "N_samples": n_samples,
                    "token_count": len(u),
                    "token_fraction": len(u) / n_tokens,
                    "sample_presence_rate": u["sid"].nunique() / n_samples,
                    "mean_rank": safe_mean(u["analysis_rank"]),
                    "mean_mediation": safe_mean(u["mediation"]),
                })

    return pd.DataFrame(rows)


def cooccurrence(sel, n_samples, pair_min_support, triple_min_support, max_pairs, max_triples):
    sample_units = {
        int(sid): sorted(set(g["layer_role"]))
        for sid, g in sel.groupby("sid")
    }

    unit_counts = Counter()
    pair_counts = Counter()
    triple_counts = Counter()

    for units in sample_units.values():
        for u in units:
            unit_counts[u] += 1
        for pair in itertools.combinations(units, 2):
            pair_counts[pair] += 1
        for triple in itertools.combinations(units, 3):
            triple_counts[triple] += 1

    pair_rows = []
    for (a, b), count in pair_counts.items():
        support = count / n_samples
        if support < pair_min_support:
            continue
        pa = unit_counts[a] / n_samples
        pb = unit_counts[b] / n_samples
        expected = pa * pb
        pair_rows.append({
            "unit_a": a,
            "unit_b": b,
            "count": count,
            "support": support,
            "presence_a": pa,
            "presence_b": pb,
            "expected_if_independent": expected,
            "lift": safe_div(support, expected),
        })

    pair_rows = sorted(
        pair_rows,
        key=lambda r: (r["support"], r["lift"]),
        reverse=True,
    )[:max_pairs]

    triple_rows = []
    for (a, b, c), count in triple_counts.items():
        support = count / n_samples
        if support < triple_min_support:
            continue
        pa = unit_counts[a] / n_samples
        pb = unit_counts[b] / n_samples
        pc = unit_counts[c] / n_samples
        expected = pa * pb * pc
        triple_rows.append({
            "unit_a": a,
            "unit_b": b,
            "unit_c": c,
            "count": count,
            "support": support,
            "presence_a": pa,
            "presence_b": pb,
            "presence_c": pc,
            "expected_if_independent": expected,
            "lift": safe_div(support, expected),
        })

    triple_rows = sorted(
        triple_rows,
        key=lambda r: (r["support"], r["lift"]),
        reverse=True,
    )[:max_triples]

    return pd.DataFrame(pair_rows), pd.DataFrame(triple_rows)


def set_jaccard(a, b):
    a = set(a)
    b = set(b)
    u = a | b
    return len(a & b) / len(u) if u else float("nan")


def multiset_jaccard(ca, cb):
    keys = set(ca) | set(cb)
    num = sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    den = sum(max(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    return num / den if den else float("nan")


def jaccard_analysis(sel, sample_df):
    info = sample_df.set_index("sid").to_dict("index")
    per_sid = {}

    for sid, g in sel.groupby("sid"):
        sid = int(sid)
        per_sid[sid] = {
            "layer_role_set": set(g["layer_role"]),
            "layer_role_count": Counter(g["layer_role"]),
            "semantic_set": set(g["semantic_signature"]),
        }

    sids = sorted(per_sid)
    rows = []

    for i, sa in enumerate(sids):
        ia = info[sa]
        for sb in sids[i + 1:]:
            ib = info[sb]

            a = per_sid[sa]
            b = per_sid[sb]

            ba = bool(ia["baseline_correct"])
            bb = bool(ib["baseline_correct"])
            if ba and bb:
                status_pair = "correct_correct"
            elif (not ba) and (not bb):
                status_pair = "wrong_wrong"
            else:
                status_pair = "mixed"

            rows.append({
                "sid_a": sa,
                "sid_b": sb,
                "relation_a": ia["relation"],
                "relation_b": ib["relation"],
                "same_relation": ia["relation"] == ib["relation"],
                "baseline_status_pair": status_pair,
                "layer_role_set_jaccard": set_jaccard(
                    a["layer_role_set"],
                    b["layer_role_set"],
                ),
                "layer_role_multiset_jaccard": multiset_jaccard(
                    a["layer_role_count"],
                    b["layer_role_count"],
                ),
                "semantic_set_jaccard": set_jaccard(
                    a["semantic_set"],
                    b["semantic_set"],
                ),
            })

    pair_df = pd.DataFrame(rows)

    summary_rows = []

    def add_summary(label, sub):
        if not len(sub):
            return
        summary_rows.append({
            "comparison": label,
            "N_pairs": len(sub),
            "mean_layer_role_set_jaccard": sub[
                "layer_role_set_jaccard"
            ].mean(),
            "mean_layer_role_multiset_jaccard": sub[
                "layer_role_multiset_jaccard"
            ].mean(),
            "mean_semantic_set_jaccard": sub[
                "semantic_set_jaccard"
            ].mean(),
            "median_layer_role_multiset_jaccard": sub[
                "layer_role_multiset_jaccard"
            ].median(),
        })

    add_summary("all_pairs", pair_df)
    add_summary("same_relation", pair_df[pair_df["same_relation"]])
    add_summary("different_relation", pair_df[~pair_df["same_relation"]])

    for rel in REL_ORDER:
        add_summary(
            f"same_relation:{rel}",
            pair_df[
                (pair_df["relation_a"] == rel)
                & (pair_df["relation_b"] == rel)
            ],
        )

    for status in ["correct_correct", "wrong_wrong", "mixed"]:
        add_summary(
            status,
            pair_df[pair_df["baseline_status_pair"] == status],
        )

    return pair_df, pd.DataFrame(summary_rows)


def top_concrete_tokens(sel, sample_df, top_n):
    meta = sample_df[["sid", "relation", "baseline_correct"]]
    x = sel.merge(meta, on="sid", how="left", suffixes=("", "_meta"))

    rows = []

    slice_specs = [("overall", "all", x)]
    for rel in REL_ORDER:
        slice_specs.append(
            ("relation", rel, x[x["relation_meta"] == rel])
        )

    for slice_type, slice_value, g in slice_specs:
        n_samples = g["sid"].nunique()
        if not n_samples:
            continue

        # One semantic signature can appear only once per sample for presence stats.
        agg = (
            g.groupby(
                [
                    "source_layer",
                    "broad_category",
                    "semantic_signature",
                ],
                as_index=False,
            )
            .agg(
                token_count=("sid", "size"),
                sample_presence_N=("sid", "nunique"),
                mean_rank=("analysis_rank", "mean"),
                mean_mediation=("mediation", "mean"),
                max_mediation=("mediation", "max"),
            )
        )
        agg["sample_presence_rate"] = agg["sample_presence_N"] / n_samples
        agg["slice_type"] = slice_type
        agg["slice_value"] = slice_value
        agg["N_samples"] = n_samples

        agg = agg.sort_values(
            ["sample_presence_rate", "mean_mediation"],
            ascending=[False, False],
        ).head(top_n)

        rows.extend(agg.to_dict("records"))

    return pd.DataFrame(rows)


def make_cluster_features(sel, sample_df, exclude_relation_words):
    x = sel.copy()
    if exclude_relation_words:
        x = x[x["broad_category"] != "relation_words"].copy()

    units = sorted(x["layer_role"].unique())
    sids = sorted(sample_df["sid"].astype(int).unique())

    records = []

    for sid in sids:
        g = x[x["sid"] == sid]
        total = len(g)
        total_m = float(g["mediation"].clip(lower=0).sum())

        row = {"sid": sid}

        for unit in units:
            u = g[g["layer_role"] == unit]
            c = len(u)
            row[f"frac::{unit}"] = c / total if total else 0.0
            row[f"rr::{unit}"] = (
                float((1.0 / u["analysis_rank"]).sum())
                if c
                else 0.0
            )
            m = float(u["mediation"].clip(lower=0).sum())
            row[f"mshare::{unit}"] = (
                m / total_m if total_m > 0 else 0.0
            )

        records.append(row)

    feat_df = pd.DataFrame(records).set_index("sid").sort_index()

    # Drop all-zero / constant columns before clustering.
    keep = [
        c for c in feat_df.columns
        if feat_df[c].nunique(dropna=False) > 1
    ]
    feat_df = feat_df[keep].fillna(0.0)

    return feat_df


def choose_cluster_k(Xz, a):
    n = Xz.shape[0]

    if a.n_clusters and a.n_clusters >= 2:
        k = min(int(a.n_clusters), max(2, n - 1))
        return k, pd.DataFrame([{
            "k": k,
            "silhouette": float("nan"),
            "selected": True,
            "mode": "user_fixed",
        }])

    kmin = max(2, int(a.cluster_k_min))
    kmax = min(int(a.cluster_k_max), n - 1)

    rows = []
    best_k = None
    best_score = -np.inf

    for k in range(kmin, kmax + 1):
        km = KMeans(
            n_clusters=k,
            random_state=a.seed,
            n_init=20,
        )
        labels = km.fit_predict(Xz)

        if len(set(labels)) < 2:
            score = float("nan")
        else:
            score = float(silhouette_score(Xz, labels))

        rows.append({
            "k": k,
            "silhouette": score,
            "selected": False,
            "mode": "auto_silhouette",
        })

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        best_k = kmin

    for r in rows:
        r["selected"] = r["k"] == best_k

    return best_k, pd.DataFrame(rows)


def cluster_analysis(sel, sample_df, a, exclude_relation_words, prefix, outdir):
    feat_df = make_cluster_features(
        sel,
        sample_df,
        exclude_relation_words=exclude_relation_words,
    )

    if feat_df.shape[1] == 0:
        raise RuntimeError(
            f"No non-constant cluster features for {prefix}"
        )

    scaler = StandardScaler()
    Xz = scaler.fit_transform(feat_df.values)

    k, silhouette_df = choose_cluster_k(Xz, a)

    km = KMeans(
        n_clusters=k,
        random_state=a.seed,
        n_init=50,
    )
    labels = km.fit_predict(Xz)

    assign = pd.DataFrame({
        "sid": feat_df.index.astype(int),
        "cluster": labels.astype(int),
    }).merge(
        sample_df[
            [
                "sid",
                "relation",
                "baseline_correct",
                "edited_correct",
                "repair_outcome",
                "n_selected",
            ]
        ],
        on="sid",
        how="left",
    )

    # Centroids in ORIGINAL feature units for interpretability.
    centroids_z = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_z)
    centroid_df = pd.DataFrame(
        centroids,
        columns=feat_df.columns,
    )
    centroid_df.insert(0, "cluster", np.arange(k))

    # Long-form dominant centroid features.
    centroid_long = []
    for c in range(k):
        vals = centroid_df[centroid_df["cluster"] == c].iloc[0]
        pairs = [
            (col, float(vals[col]))
            for col in feat_df.columns
        ]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        for rank, (feature, value) in enumerate(pairs, 1):
            centroid_long.append({
                "cluster": c,
                "rank": rank,
                "feature": feature,
                "value": value,
            })
    centroid_long_df = pd.DataFrame(centroid_long)

    summary_rows = []

    for c, g in assign.groupby("cluster"):
        N = len(g)
        wrong = g[g["baseline_correct"] == False]
        correct = g[g["baseline_correct"] == True]

        rel_counts = g["relation"].value_counts(normalize=True)
        outcome_counts = g["repair_outcome"].value_counts(normalize=True)

        top_features = (
            centroid_long_df[centroid_long_df["cluster"] == c]
            .head(10)
        )
        top_feature_text = " | ".join(
            f"{r.feature}={r.value:.3f}"
            for r in top_features.itertuples()
        )

        summary_rows.append({
            "cluster": int(c),
            "N": N,
            "fraction_of_samples": N / len(assign),
            "baseline_acc": safe_mean(g["baseline_correct"]),
            "edited_acc": (
                safe_mean(g["edited_correct"].dropna())
                if g["edited_correct"].notna().any()
                else float("nan")
            ),
            "repair_rate_given_baseline_wrong": (
                float((wrong["repair_outcome"] == "W2C").mean())
                if len(wrong)
                else float("nan")
            ),
            "damage_rate_given_baseline_correct": (
                float((correct["repair_outcome"] == "C2W").mean())
                if len(correct)
                else float("nan")
            ),
            "left_fraction": float(rel_counts.get("left", 0.0)),
            "right_fraction": float(rel_counts.get("right", 0.0)),
            "on_fraction": float(rel_counts.get("on", 0.0)),
            "under_fraction": float(rel_counts.get("under", 0.0)),
            "W2C_fraction_all": float(outcome_counts.get("W2C", 0.0)),
            "C2W_fraction_all": float(outcome_counts.get("C2W", 0.0)),
            "top_centroid_features": top_feature_text,
        })

    assign.to_csv(outdir / f"{prefix}_assignments.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        outdir / f"{prefix}_summary.csv",
        index=False,
    )
    centroid_df.to_csv(
        outdir / f"{prefix}_centroids.csv",
        index=False,
    )
    centroid_long_df.to_csv(
        outdir / f"{prefix}_centroids_long.csv",
        index=False,
    )
    silhouette_df.to_csv(
        outdir / f"{prefix}_silhouette.csv",
        index=False,
    )

    return {
        "k": k,
        "assignments": assign,
        "summary": pd.DataFrame(summary_rows),
        "silhouette": silhouette_df,
    }


def make_analysis_summary(
    sel,
    sample_df,
    cw_df,
    rel_df,
    rank_df,
    pair_df,
    triple_df,
    jaccard_summary,
    clusters_full,
    clusters_no_rel,
    chosen_alpha,
):
    lines = []

    N = sample_df["sid"].nunique()
    lines.append("=" * 100)
    lines.append("DYNAMIC K24 SAMPLE-SPECIFIC PATTERN ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"N samples: {N}")
    lines.append(f"Mean selected/sample: {sample_df['n_selected'].mean():.3f}")
    lines.append(
        f"Baseline accuracy: {sample_df['baseline_correct'].mean():.4f}"
    )
    if chosen_alpha is not None:
        edited_valid = sample_df["edited_correct"].dropna()
        lines.append(f"Repair alpha used: {chosen_alpha:.4f}")
        if len(edited_valid):
            lines.append(
                f"Edited accuracy at chosen alpha: {edited_valid.mean():.4f}"
            )
            wrong = sample_df[sample_df["baseline_correct"] == False]
            correct = sample_df[sample_df["baseline_correct"] == True]
            if len(wrong):
                lines.append(
                    "Repair rate W2C | baseline wrong: "
                    f"{(wrong['repair_outcome'] == 'W2C').mean():.4f}"
                )
            if len(correct):
                lines.append(
                    "Damage rate C2W | baseline correct: "
                    f"{(correct['repair_outcome'] == 'C2W').mean():.4f}"
                )

    lines.append("")
    lines.append("Top global layer-role units by sample presence:")
    global_presence = (
        sel.groupby("layer_role")["sid"]
        .nunique()
        .sort_values(ascending=False)
    )
    for unit, n in global_presence.head(15).items():
        lines.append(f"  {unit:<28s} {n/N:.3f} ({n}/{N})")

    lines.append("")
    lines.append("Top global layer-role units by token fraction:")
    vc = sel["layer_role"].value_counts(normalize=True)
    for unit, frac in vc.head(15).items():
        lines.append(f"  {unit:<28s} {frac:.3f}")

    lines.append("")
    lines.append("Correct vs wrong largest token-fraction differences:")
    if len(cw_df):
        piv = cw_df.pivot_table(
            index="layer_role",
            columns="group",
            values="token_fraction",
            aggfunc="first",
        ).fillna(0.0)
        if "correct" in piv.columns and "wrong" in piv.columns:
            piv["correct_minus_wrong"] = piv["correct"] - piv["wrong"]
            for unit, r in piv.sort_values(
                "correct_minus_wrong", ascending=False
            ).head(8).iterrows():
                lines.append(
                    f"  correct-heavy {unit:<24s} "
                    f"Δ={r['correct_minus_wrong']:+.3f}"
                )
            for unit, r in piv.sort_values(
                "correct_minus_wrong", ascending=True
            ).head(8).iterrows():
                lines.append(
                    f"  wrong-heavy   {unit:<24s} "
                    f"Δ={r['correct_minus_wrong']:+.3f}"
                )

    lines.append("")
    lines.append("Most relation-enriched layer-role units:")
    if len(rel_df):
        for rel in REL_ORDER:
            q = rel_df[rel_df["group"] == rel].sort_values(
                "presence_enrichment_vs_global",
                ascending=False,
            ).head(6)
            lines.append(f"  {rel.upper()}:")
            for r in q.itertuples():
                lines.append(
                    f"    {r.layer_role:<26s} "
                    f"presence={r.sample_presence_rate:.3f} "
                    f"enrichment={r.presence_enrichment_vs_global:.2f}x"
                )

    lines.append("")
    lines.append("Rank evolution (overall dominant units):")
    if len(rank_df):
        for cutoff in sorted(rank_df["cutoff"].unique()):
            q = rank_df[
                (rank_df["cutoff"] == cutoff)
                & (rank_df["slice_type"] == "overall")
            ].sort_values("token_fraction", ascending=False).head(5)
            text = ", ".join(
                f"{r.layer_role}={r.token_fraction:.2f}"
                for r in q.itertuples()
            )
            lines.append(f"  Top{int(cutoff):>2}: {text}")

    lines.append("")
    lines.append("Top pair co-occurrences:")
    for r in pair_df.head(12).itertuples():
        lines.append(
            f"  {r.unit_a} + {r.unit_b} | "
            f"support={r.support:.3f}, lift={r.lift:.2f}"
        )

    lines.append("")
    lines.append("Top triple co-occurrences:")
    for r in triple_df.head(12).itertuples():
        lines.append(
            f"  {r.unit_a} + {r.unit_b} + {r.unit_c} | "
            f"support={r.support:.3f}, lift={r.lift:.2f}"
        )

    lines.append("")
    lines.append("K24 cross-sample Jaccard:")
    if len(jaccard_summary):
        for r in jaccard_summary.itertuples():
            if r.comparison in [
                "all_pairs",
                "same_relation",
                "different_relation",
                "correct_correct",
                "wrong_wrong",
            ]:
                lines.append(
                    f"  {r.comparison:<20s} "
                    f"role-multiset={r.mean_layer_role_multiset_jaccard:.3f}, "
                    f"semantic-set={r.mean_semantic_set_jaccard:.3f}"
                )

    lines.append("")
    lines.append(
        f"Unsupervised full-feature clustering: k={clusters_full['k']}"
    )
    for r in clusters_full["summary"].itertuples():
        lines.append(
            f"  C{r.cluster}: N={r.N}, base={r.baseline_acc:.3f}, "
            f"edited={r.edited_acc:.3f}, "
            f"repair|wrong={r.repair_rate_given_baseline_wrong:.3f}, "
            f"L/R/ON/U={r.left_fraction:.2f}/"
            f"{r.right_fraction:.2f}/{r.on_fraction:.2f}/{r.under_fraction:.2f}"
        )

    lines.append("")
    lines.append(
        "Unsupervised clustering WITHOUT relation-word motifs: "
        f"k={clusters_no_rel['k']}"
    )
    for r in clusters_no_rel["summary"].itertuples():
        lines.append(
            f"  C{r.cluster}: N={r.N}, base={r.baseline_acc:.3f}, "
            f"edited={r.edited_acc:.3f}, "
            f"repair|wrong={r.repair_rate_given_baseline_wrong:.3f}, "
            f"L/R/ON/U={r.left_fraction:.2f}/"
            f"{r.right_fraction:.2f}/{r.on_fraction:.2f}/{r.under_fraction:.2f}"
        )

    lines.append("")
    lines.append(
        "Interpretation guardrail: high recurrence of a layer-role motif means "
        "a reusable structural pattern; it does NOT by itself prove that every "
        "member of that motif is necessary. Necessity requires deletion/leave-one-out."
    )

    return "\n".join(lines) + "\n"


def main():
    a = parse_args()
    cutoffs = parse_ints(a.rank_cutoffs)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    sel, baseline, generation, chosen_alpha = load_and_filter(a)

    sample_df, layers, roles = sample_composition(
        sel,
        baseline,
        generation,
    )
    sample_df.to_csv(
        outdir / "sample_layer_role_composition.csv",
        index=False,
    )

    # Baseline correct vs wrong.
    sample_df["baseline_status"] = np.where(
        sample_df["baseline_correct"],
        "correct",
        "wrong",
    )
    cw_df = layer_role_group_summary(
        sel,
        sample_df,
        "baseline_status",
        ["correct", "wrong"],
    )
    cw_df.to_csv(
        outdir / "correct_wrong_layer_role.csv",
        index=False,
    )

    # Four relations.
    rel_df = relation_summary_with_enrichment(
        sel,
        sample_df,
    )
    rel_df.to_csv(
        outdir / "relation_layer_role.csv",
        index=False,
    )

    # Rank-cutoff composition.
    rank_df = rank_cutoff_summary(
        sel,
        sample_df,
        cutoffs,
    )
    rank_df.to_csv(
        outdir / "rank_cutoff_composition.csv",
        index=False,
    )

    # Pair / triple motif co-occurrence.
    pair_df, triple_df = cooccurrence(
        sel,
        n_samples=sample_df["sid"].nunique(),
        pair_min_support=a.pair_min_support,
        triple_min_support=a.triple_min_support,
        max_pairs=a.max_pairs,
        max_triples=a.max_triples,
    )
    pair_df.to_csv(
        outdir / "pair_cooccurrence.csv",
        index=False,
    )
    triple_df.to_csv(
        outdir / "triple_cooccurrence.csv",
        index=False,
    )

    # Cross-sample K24 Jaccard.
    jaccard_pairs, jaccard_summary = jaccard_analysis(
        sel,
        sample_df,
    )
    jaccard_pairs.to_csv(
        outdir / "jaccard_pairwise.csv",
        index=False,
    )
    jaccard_summary.to_csv(
        outdir / "jaccard_summary.csv",
        index=False,
    )

    # Recurrent concrete token carriers.
    concrete_df = top_concrete_tokens(
        sel,
        sample_df,
        a.top_concrete_n,
    )
    concrete_df.to_csv(
        outdir / "top_concrete_tokens.csv",
        index=False,
    )

    # Unsupervised clustering, with and without relation-word motifs.
    clusters_full = cluster_analysis(
        sel,
        sample_df,
        a,
        exclude_relation_words=False,
        prefix="cluster_full",
        outdir=outdir,
    )
    clusters_no_rel = cluster_analysis(
        sel,
        sample_df,
        a,
        exclude_relation_words=True,
        prefix="cluster_no_relation_words",
        outdir=outdir,
    )

    summary_text = make_analysis_summary(
        sel=sel,
        sample_df=sample_df,
        cw_df=cw_df,
        rel_df=rel_df,
        rank_df=rank_df,
        pair_df=pair_df,
        triple_df=triple_df,
        jaccard_summary=jaccard_summary,
        clusters_full=clusters_full,
        clusters_no_rel=clusters_no_rel,
        chosen_alpha=chosen_alpha,
    )

    (outdir / "analysis_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    metadata = {
        "run_dir": str(a.run_dir),
        "bundle": a.bundle,
        "k": a.k,
        "condition": a.condition,
        "selection_strategy": a.selection_strategy,
        "N_samples": int(sample_df["sid"].nunique()),
        "layers": [int(x) for x in layers],
        "roles": [str(x) for x in roles],
        "rank_cutoffs": cutoffs,
        "repair_alpha": chosen_alpha,
        "repair_alpha_mode": (
            "user_fixed"
            if a.repair_alpha is not None
            else "auto_best_overall"
        ),
        "full_cluster_k": clusters_full["k"],
        "no_relation_words_cluster_k": clusters_no_rel["k"],
        "note": (
            "Pure post-processing of each sample's already-selected dynamic K24; "
            "no VLM forward and no new selector."
        ),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(summary_text)
    print("Saved outputs to:", outdir)


if __name__ == "__main__":
    main()
