
# -*- coding: utf-8 -*-

"""
analyze_causal_core_vs_spatial_heads_from_csv_v1.py

GPU-free post-hoc analysis.

Goal
----
Test the current hypothesis using the CSVs we already produced:

    strongest writer-causal core (Rank1 / Top3 / Top5 / Top7 ...)
        vs
    source tokens used by known Direction / Centroid spatial heads.

This script DOES NOT recompute the model. It joins:

1) ranked causal states:
   outputs/.../ranked_k36_tokens.csv
   from eval_qwen_oracle_rank_marginal_v1.py
   (or eval_qwen_oracle_k36_prefix_curve_v1.py)

2) per-token spatial-head source scores:
   outputs/.../token_scores.csv
   from analyze_qwen_k36_vs_spatial_heads_v1.py

The two runs may contain different numbers of samples. We automatically use
their SID intersection and print the common N.

Definitions
-----------
For causal core K:

    C_K(i) = global writer-ranked states with rank <= K.

For head H at attention layer L, the aligned causal/source state layer is the
`aligned_source_layer` already stored in token_scores.csv.

Direction head:
    all token positions scored by its A*V spatial contribution.

Centroid head:
    only visual token positions are eligible, because its source mechanism is
    visual attention geometry.

For each causal state that is eligible for a given head, we ask:

A) MATCHED-BUDGET OVERLAP
   If a sample has c causal states on this head's aligned source layer, take
   the head's Top-c spatial source positions and measure overlap.

   This gives:
       micro_recall
       expected_random_recall
       enrichment_over_random

B) SPATIAL PERCENTILE (budget-independent)
   Rank every eligible token by the head's spatial_source_score.
   Best token -> percentile 1.0
   Worst token -> percentile 0.0

   Random expectation ~= 0.5.

C) TOP-10% / TOP-5% HIT RATE
   Does the causal state fall in the head's strongest 10% / 5% sources?
   Random expectations are ~10% / ~5%.

D) GLOBAL CAUSAL RANK -> SPATIALNESS
   For Rank1, Rank2, ... directly report the mean spatial percentile whenever
   that causal state lands on the head's aligned source layer.

This directly tests:
    "As we shrink K36 toward the strongest causal core, does it become more
     aligned with Direction heads (especially L26H03), while Centroid alignment
     remains weak?"

Example
-------
python analyze_causal_core_vs_spatial_heads_from_csv_v1.py \
  --ranked-causal outputs/oracle_rank_marginal_v1_r10/ranked_k36_tokens.csv \
  --head-token-scores outputs/k36_vs_spatial_heads_v1_n80/token_scores.csv \
  --cores 1,3,5,7,10,20,36 \
  --rank-max 10 \
  --output-dir outputs/causal_core_vs_spatial_heads_posthoc_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--ranked-causal", required=True,
                   help="ranked_k36_tokens.csv")
    p.add_argument("--head-token-scores", required=True,
                   help="token_scores.csv from K36-vs-spatial-head analysis")
    p.add_argument("--cores", default="1,3,5,7,10,20,36")
    p.add_argument("--rank-max", type=int, default=10,
                   help="Direct rank-by-rank spatialness table up to this rank.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def median(xs):
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def corr(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def spearman(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    return corr(rankdata(x), rankdata(y))


def f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def i(row, key):
    return int(float(row[key]))


def percentile_map(score_by_pos):
    """
    Descending score ranking.
    Best -> 1.0, worst -> 0.0.
    Average rank is used for ties.
    """
    items = sorted(
        score_by_pos.items(),
        key=lambda kv: (-float(kv[1]), int(kv[0]))
    )
    n = len(items)
    if n == 0:
        return {}, {}, []

    # Average rank for tied score, 1-indexed.
    rank_by_pos = {}
    j = 0
    while j < n:
        k = j + 1
        while k < n and float(items[k][1]) == float(items[j][1]):
            k += 1
        avg_rank = 0.5 * ((j + 1) + k)
        for t in range(j, k):
            rank_by_pos[int(items[t][0])] = avg_rank
        j = k

    pct = {}
    for pos, r in rank_by_pos.items():
        pct[pos] = 1.0 if n == 1 else float((n - r) / (n - 1))

    ordered_positions = [int(pos) for pos, _ in items]
    return pct, rank_by_pos, ordered_positions


def main():
    a = parse_args()
    cores = parse_ints(a.cores)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    causal_raw = read_csv(a.ranked_causal)
    head_raw = read_csv(a.head_token_scores)

    # ------------------------------------------------------------------
    # Normalize causal ranking.
    # ------------------------------------------------------------------
    causal_by_sid = defaultdict(list)
    causal_sids = set()

    for r in causal_raw:
        sid = i(r, "sid")
        row = {
            "sid": sid,
            "rank": i(r, "rank"),
            "source_layer": i(r, "source_layer"),
            "position": i(r, "position"),
            "token": r.get("token", ""),
            "category": r.get("category", ""),
            "broad_category": r.get("broad_category", ""),
            "mediation": f(r, "mediation"),
            "M_over_M1": f(r, "M_over_M1"),
        }
        causal_by_sid[sid].append(row)
        causal_sids.add(sid)

    for sid in causal_by_sid:
        causal_by_sid[sid].sort(key=lambda x: x["rank"])

    # ------------------------------------------------------------------
    # Normalize spatial-head token scores.
    #
    # head_scores[(family, head_name, head_layer, aligned_source_layer)][sid]
    #     = {position: score}
    # ------------------------------------------------------------------
    head_scores = defaultdict(lambda: defaultdict(dict))
    head_meta = {}
    head_sids = set()

    for r in head_raw:
        sid = i(r, "sid")
        family = r["family"].strip()
        name = r["head_name"].strip()
        head_layer = i(r, "head_layer")
        source_layer = i(r, "aligned_source_layer")
        pos = i(r, "position")
        score = f(r, "spatial_source_score")

        key = (family, name, head_layer, source_layer)
        head_scores[key][sid][pos] = score
        head_meta[key] = {
            "family": family,
            "head_name": name,
            "head_layer": head_layer,
            "aligned_source_layer": source_layer,
        }
        head_sids.add(sid)

    common_sids = sorted(causal_sids & head_sids)
    if not common_sids:
        raise RuntimeError(
            "No common SIDs between ranked causal CSV and head token-score CSV."
        )

    print("=" * 146)
    print("STRONG CAUSAL CORE vs KNOWN SPATIAL-HEAD SOURCE TOKENS")
    print("=" * 146)
    print(f"causal CSV samples={len(causal_sids)}")
    print(f"head-score CSV samples={len(head_sids)}")
    print(f"COMMON samples used={len(common_sids)}")
    print(f"cores={cores} | direct rank table=1..{a.rank_max}")
    print()

    # Precompute head spatial rank/percentile information.
    head_rank = {}
    for key, by_sid in head_scores.items():
        for sid, score_by_pos in by_sid.items():
            pct, ranks, ordered = percentile_map(score_by_pos)
            n = len(ordered)
            top10_n = max(1, int(math.ceil(0.10 * n))) if n else 0
            top05_n = max(1, int(math.ceil(0.05 * n))) if n else 0

            head_rank[(key, sid)] = {
                "N": n,
                "pct": pct,
                "rank": ranks,
                "ordered": ordered,
                "top10": set(ordered[:top10_n]),
                "top05": set(ordered[:top05_n]),
                "top10_fraction": top10_n / n if n else float("nan"),
                "top05_fraction": top05_n / n if n else float("nan"),
            }

    # ------------------------------------------------------------------
    # 1) HEAD x CORE
    # ------------------------------------------------------------------
    core_rows = []
    detail_rows = []

    for key in sorted(head_meta):
        meta = head_meta[key]
        family = meta["family"]
        source_layer = meta["aligned_source_layer"]

        for K in cores:
            total_core_states = 0
            layer_core_states = 0
            eligible_states = 0
            overlap = 0
            expected_overlap = 0.0
            sample_recalls = []
            percentiles = []
            top10_hits = 0
            top05_hits = 0
            expected_top10_hits = 0.0
            expected_top05_hits = 0.0
            samples_with_eligible = 0

            for sid in common_sids:
                core = [r for r in causal_by_sid[sid] if r["rank"] <= K]
                total_core_states += len(core)

                on_layer = [
                    r for r in core if r["source_layer"] == source_layer
                ]
                layer_core_states += len(on_layer)

                info = head_rank.get((key, sid))
                if info is None or info["N"] == 0:
                    continue

                # Direction: all positions in token_scores are eligible.
                # Centroid: token_scores itself contains only visual source positions,
                # so membership in info["pct"] automatically performs the visual-only
                # restriction.
                elig = [r for r in on_layer if r["position"] in info["pct"]]
                c = len(elig)
                if c == 0:
                    continue

                samples_with_eligible += 1
                eligible_states += c

                # Fair matched-budget spatial Top-c for THIS sample/head/core.
                spatial_top = set(info["ordered"][: min(c, info["N"])])
                hit = sum(r["position"] in spatial_top for r in elig)
                overlap += hit
                sample_recalls.append(hit / c)

                expected_overlap += c * min(c, info["N"]) / info["N"]

                for r in elig:
                    pos = r["position"]
                    pct = float(info["pct"][pos])
                    in10 = pos in info["top10"]
                    in05 = pos in info["top05"]

                    percentiles.append(pct)
                    top10_hits += int(in10)
                    top05_hits += int(in05)
                    expected_top10_hits += float(info["top10_fraction"])
                    expected_top05_hits += float(info["top05_fraction"])

                    detail_rows.append({
                        "sid": sid,
                        "core_k": K,
                        "family": family,
                        "head_name": meta["head_name"],
                        "head_layer": meta["head_layer"],
                        "aligned_source_layer": source_layer,
                        "causal_rank": r["rank"],
                        "causal_M": r["mediation"],
                        "causal_M_over_M1": r["M_over_M1"],
                        "position": pos,
                        "token": r["token"],
                        "category": r["category"],
                        "broad_category": r["broad_category"],
                        "spatial_source_score": head_scores[key][sid][pos],
                        "spatial_percentile": pct,
                        "in_matched_budget_spatial_top": pos in spatial_top,
                        "in_spatial_top10pct": in10,
                        "in_spatial_top05pct": in05,
                    })

            micro_recall = (
                overlap / eligible_states
                if eligible_states else float("nan")
            )
            expected_random_recall = (
                expected_overlap / eligible_states
                if eligible_states else float("nan")
            )
            enrichment = (
                overlap / expected_overlap
                if expected_overlap > EPS else float("nan")
            )
            top10_rate = (
                top10_hits / eligible_states
                if eligible_states else float("nan")
            )
            top05_rate = (
                top05_hits / eligible_states
                if eligible_states else float("nan")
            )

            core_rows.append({
                "core_k": K,
                "family": family,
                "head_name": meta["head_name"],
                "head_layer": meta["head_layer"],
                "aligned_source_layer": source_layer,
                "common_sample_N": len(common_sids),
                "samples_with_eligible_causal_state": samples_with_eligible,
                "total_causal_core_states": total_core_states,
                "causal_states_on_aligned_layer": layer_core_states,
                "eligible_causal_states": eligible_states,
                "eligible_fraction_of_full_core": (
                    eligible_states / total_core_states
                    if total_core_states else float("nan")
                ),
                "matched_budget_overlap_count": overlap,
                "micro_recall": micro_recall,
                "mean_sample_recall": mean(sample_recalls),
                "expected_random_recall": expected_random_recall,
                "enrichment_over_random": enrichment,
                "mean_spatial_percentile": mean(percentiles),
                "median_spatial_percentile": median(percentiles),
                "spatial_top10_rate": top10_rate,
                "expected_top10_rate": (
                    expected_top10_hits / eligible_states
                    if eligible_states else float("nan")
                ),
                "top10_enrichment": (
                    top10_hits / expected_top10_hits
                    if expected_top10_hits > EPS else float("nan")
                ),
                "spatial_top05_rate": top05_rate,
                "expected_top05_rate": (
                    expected_top05_hits / eligible_states
                    if eligible_states else float("nan")
                ),
                "top05_enrichment": (
                    top05_hits / expected_top05_hits
                    if expected_top05_hits > EPS else float("nan")
                ),
            })

    # ------------------------------------------------------------------
    # 2) DIRECT GLOBAL CAUSAL RANK -> HEAD SPATIAL PERCENTILE
    # ------------------------------------------------------------------
    rank_rows = []

    for key in sorted(head_meta):
        meta = head_meta[key]
        source_layer = meta["aligned_source_layer"]

        all_rank_values = []
        all_pct_values = []

        for rank in range(1, int(a.rank_max) + 1):
            pcts = []
            top10 = 0
            top05 = 0
            expected10 = 0.0
            expected05 = 0.0
            N = 0

            for sid in common_sids:
                state = next(
                    (
                        r for r in causal_by_sid[sid]
                        if r["rank"] == rank
                        and r["source_layer"] == source_layer
                    ),
                    None,
                )
                if state is None:
                    continue

                info = head_rank.get((key, sid))
                if info is None or state["position"] not in info["pct"]:
                    continue

                pos = state["position"]
                pct = float(info["pct"][pos])

                N += 1
                pcts.append(pct)
                top10 += int(pos in info["top10"])
                top05 += int(pos in info["top05"])
                expected10 += float(info["top10_fraction"])
                expected05 += float(info["top05_fraction"])

                all_rank_values.append(rank)
                all_pct_values.append(pct)

            rank_rows.append({
                "family": meta["family"],
                "head_name": meta["head_name"],
                "head_layer": meta["head_layer"],
                "aligned_source_layer": source_layer,
                "causal_rank": rank,
                "eligible_state_N": N,
                "mean_spatial_percentile": mean(pcts),
                "median_spatial_percentile": median(pcts),
                "top10_rate": top10 / N if N else float("nan"),
                "top10_enrichment": (
                    top10 / expected10 if expected10 > EPS else float("nan")
                ),
                "top05_rate": top05 / N if N else float("nan"),
                "top05_enrichment": (
                    top05 / expected05 if expected05 > EPS else float("nan")
                ),
            })

        # Add one trend row per head.
        rank_rows.append({
            "family": meta["family"],
            "head_name": meta["head_name"],
            "head_layer": meta["head_layer"],
            "aligned_source_layer": source_layer,
            "causal_rank": "TREND",
            "eligible_state_N": len(all_pct_values),
            "mean_spatial_percentile": float("nan"),
            "median_spatial_percentile": float("nan"),
            "spearman_causal_rank_vs_spatial_percentile": spearman(
                all_rank_values, all_pct_values
            ),
        })

    # ------------------------------------------------------------------
    # 3) FAMILY-LEVEL FIXED TOP-10% UNION
    #
    # This avoids unstable "Top-c" budgets when K is tiny.
    # A causal state is linked to a family if ANY configured head applicable
    # to that source layer puts that token in its spatial Top10%.
    # ------------------------------------------------------------------
    family_rows = []

    for K in cores:
        for family in ("direction", "centroid", "either"):
            eligible = 0
            linked = 0
            pct_best = []
            total_core = 0

            for sid in common_sids:
                core = [r for r in causal_by_sid[sid] if r["rank"] <= K]
                total_core += len(core)

                for state in core:
                    candidates = []

                    for key, meta in head_meta.items():
                        if family != "either" and meta["family"] != family:
                            continue
                        if state["source_layer"] != meta["aligned_source_layer"]:
                            continue

                        info = head_rank.get((key, sid))
                        if info is None:
                            continue
                        pos = state["position"]
                        if pos not in info["pct"]:
                            continue

                        candidates.append((
                            float(info["pct"][pos]),
                            pos in info["top10"],
                        ))

                    if not candidates:
                        continue

                    eligible += 1
                    pct_best.append(max(x[0] for x in candidates))
                    if any(x[1] for x in candidates):
                        linked += 1

            family_rows.append({
                "core_k": K,
                "family": family,
                "total_causal_core_states": total_core,
                "eligible_for_configured_family_heads": eligible,
                "eligible_fraction": (
                    eligible / total_core if total_core else float("nan")
                ),
                "top10_linked_states": linked,
                "top10_link_rate_among_eligible": (
                    linked / eligible if eligible else float("nan")
                ),
                "mean_best_spatial_percentile": mean(pct_best),
                "median_best_spatial_percentile": median(pct_best),
            })

    write_csv(outdir / "head_by_core.csv", core_rows)
    write_csv(outdir / "eligible_state_details.csv", detail_rows)
    write_csv(outdir / "causal_rank_spatialness.csv", rank_rows)
    write_csv(outdir / "family_union_top10.csv", family_rows)

    metadata = {
        "ranked_causal": str(Path(a.ranked_causal)),
        "head_token_scores": str(Path(a.head_token_scores)),
        "causal_samples": len(causal_sids),
        "head_score_samples": len(head_sids),
        "common_samples": len(common_sids),
        "cores": cores,
        "rank_max": a.rank_max,
        "important": [
            "Direction comparisons use all token positions scored by the direction head.",
            "Centroid comparisons are visual-only because token_scores.csv contains only centroid visual sources.",
            "Matched-budget overlap uses Top-c spatial tokens where c is the number of causal states eligible for that sample/head/core.",
            "Spatial percentile is budget-independent; random expectation is approximately 0.5.",
            "Top10/Top5 metrics provide a stable fixed-threshold spatialness check.",
        ],
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console: compact result table.
    # ------------------------------------------------------------------
    print("=" * 146)
    print("HEAD x CAUSAL CORE")
    print("microRecall = matched-budget overlap / eligible causal states")
    print("spPct: 1=strongest spatial source, random expectation ~0.5")
    print("=" * 146)
    print(
        f"{'core':>5s} | {'family':<9s} | {'head':<8s} | {'src':>4s} | "
        f"{'elig':>5s} | {'microRec':>8s} | {'randRec':>8s} | "
        f"{'enrich':>7s} | {'spPct':>6s} | {'top10':>6s} | {'top10x':>7s}"
    )
    print("-" * 126)

    for K in cores:
        rr = [r for r in core_rows if int(r["core_k"]) == K]
        # direction before centroid; within family by layer/head
        rr.sort(key=lambda r: (
            0 if r["family"] == "direction" else 1,
            int(r["head_layer"]),
            r["head_name"],
        ))
        for r in rr:
            print(
                f"{K:5d} | "
                f"{r['family']:<9s} | "
                f"{r['head_name']:<8s} | "
                f"L{int(r['aligned_source_layer']):02d} | "
                f"{int(r['eligible_causal_states']):5d} | "
                f"{float(r['micro_recall']):8.3f} | "
                f"{float(r['expected_random_recall']):8.3f} | "
                f"{float(r['enrichment_over_random']):7.2f} | "
                f"{float(r['mean_spatial_percentile']):6.3f} | "
                f"{float(r['spatial_top10_rate']):6.3f} | "
                f"{float(r['top10_enrichment']):7.2f}"
            )
        print("-" * 126)

    print("\n" + "=" * 146)
    print("FAMILY UNION — FIXED SPATIAL TOP10%")
    print("=" * 146)
    print(
        f"{'core':>5s} | {'family':<9s} | {'eligible':>8s} | "
        f"{'eligible%':>9s} | {'top10 link':>10s} | {'best spPct':>10s}"
    )
    print("-" * 76)
    for r in family_rows:
        print(
            f"{int(r['core_k']):5d} | "
            f"{r['family']:<9s} | "
            f"{int(r['eligible_for_configured_family_heads']):8d} | "
            f"{float(r['eligible_fraction']):9.3f} | "
            f"{float(r['top10_link_rate_among_eligible']):10.3f} | "
            f"{float(r['mean_best_spatial_percentile']):10.3f}"
        )

    # Focus on L26H03 if present.
    focus = [r for r in core_rows if r["head_name"] == "L26H03"]
    if focus:
        print("\n" + "=" * 146)
        print("FOCUS: L26H03")
        print("=" * 146)
        print(
            "Hypothesis support would look like: as K shrinks toward Top1/Top3, "
            "spatial percentile / Top10 hit / enrichment rises above the K36 value."
        )
        for K in cores:
            z = next((r for r in focus if int(r["core_k"]) == K), None)
            if z is None:
                continue
            print(
                f"K{K:<2d}: eligible={int(z['eligible_causal_states']):3d} "
                f"microRecall={float(z['micro_recall']):.3f} "
                f"enrich={float(z['enrichment_over_random']):.2f}x "
                f"spPct={float(z['mean_spatial_percentile']):.3f} "
                f"top10={float(z['spatial_top10_rate']):.3f} "
                f"top10x={float(z['top10_enrichment']):.2f}x"
            )

    print("\nHow to judge the guess:")
    print(
        "  1) Strong-core -> Direction: L26H03 K1/K3/K5 spatial percentile and "
        "Top10 rate should be clearly ABOVE its K36 values."
    )
    print(
        "  2) Direction > Centroid: Direction family should show stronger fixed-Top10 "
        "link rates / spatial percentiles; remember Centroid is visual-only."
    )
    print(
        "  3) If K1/K3 do NOT become more spatial than K36, reject the claim that "
        "shrinking to the strongest causal core explains the previous low overlap."
    )
    print(
        "  4) Do not rely on enrichment alone when eligible N is tiny. Use N, "
        "spatial percentile, and fixed Top10 rate together."
    )
    print("\nSaved:", outdir)


if __name__ == "__main__":
    main()
