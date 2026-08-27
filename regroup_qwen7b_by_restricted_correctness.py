#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regroup direction-conditioned causal results using REAL restricted correctness.

No GPU/model run is needed.

Inputs (default):
  output/qwen7b_direction_conditioned_failure_v1/direction_groups.csv
  output/qwen7b_direction_conditioned_failure_v1/causal/baseline_samples.csv
  output/qwen7b_direction_conditioned_failure_v1/per_sample_layer_ED_states.csv

Outputs:
  restricted_direction_groups.csv
  per_sample_layer_ED_states_restricted.csv
  restricted_group_layer_summary.csv
  restricted_wrong_repr_strong_per_sample.csv
  restricted_wrong_repr_strong_taxonomy.csv
"""

import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def as_bool(x):
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def stable_onset(mask, layers, k=2):
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= k:
            return layers[i-k+1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="output/qwen7b_direction_conditioned_failure_v1"
    )
    ap.add_argument("--stable-k", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root)
    direction_path = root / "direction_groups.csv"
    baseline_path = root / "causal" / "baseline_samples.csv"
    ed_path = root / "per_sample_layer_ED_states.csv"

    for p in (direction_path, baseline_path, ed_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    direction = read_csv(direction_path)
    baseline = read_csv(baseline_path)

    d_by_sid = {int(r["sid"]): r for r in direction}
    b_by_sid = {int(r["sid"]): r for r in baseline}

    # 1) Re-label using causal baseline REAL restricted correctness.
    joined = []
    for sid, d in sorted(d_by_sid.items()):
        if sid not in b_by_sid:
            continue
        br = b_by_sid[sid]
        restricted_correct = as_bool(br["real_correct"])
        rep = d["representation_group"]
        restricted_group = (
            ("restricted_correct_" if restricted_correct else "restricted_wrong_")
            + rep
        )

        joined.append({
            **d,
            "real_restricted_correct": int(restricted_correct),
            "real_restricted_pred": br.get("real_pred", ""),
            "real_restricted_margin": br.get("real_margin", ""),
            "restricted_group": restricted_group,
        })

    write_csv(root / "restricted_direction_groups.csv", joined)

    print("=" * 100)
    print("RESTRICTED CORRECTNESS x DIRECTION REPRESENTATION")
    print("=" * 100)
    counts = Counter(r["restricted_group"] for r in joined)
    for k, v in counts.most_common():
        print(f"{k:<36s} {v:4d}")

    # 2) Re-group the already-computed E/D layer states.
    ed = read_csv(ed_path)
    rg_by_sid = {int(r["sid"]): r["restricted_group"] for r in joined}

    regrouped = []
    for r in ed:
        sid = int(r["sid"])
        if sid not in rg_by_sid:
            continue
        rr = dict(r)
        rr["restricted_group"] = rg_by_sid[sid]
        regrouped.append(rr)

    write_csv(root / "per_sample_layer_ED_states_restricted.csv", regrouped)

    buckets = defaultdict(list)
    for r in regrouped:
        buckets[(r["restricted_group"], int(r["layer"]))].append(r)

    summary = []
    for (group, layer), rs in sorted(buckets.items()):
        E = np.asarray([int(r["E_all_text_correct"]) for r in rs], dtype=float)
        D = np.asarray([int(r["D_last_correct"]) for r in rs], dtype=float)
        states = [r["state"] for r in rs]

        summary.append({
            "restricted_group": group,
            "layer": layer,
            "n": len(rs),
            "E_acc": float(E.mean()),
            "D_acc": float(D.mean()),
            "E_margin": float(np.mean([float(r["E_all_text_margin"]) for r in rs])),
            "D_margin": float(np.mean([float(r["D_last_margin"]) for r in rs])),
            "state_E-D-": float(np.mean([s == "E-D-" for s in states])),
            "state_E+D-": float(np.mean([s == "E+D-" for s in states])),
            "state_E+D+": float(np.mean([s == "E+D+" for s in states])),
            "state_E-D+": float(np.mean([s == "E-D+" for s in states])),
        })

    write_csv(root / "restricted_group_layer_summary.csv", summary)

    print("\n" + "=" * 100)
    print("RESTRICTED DIRECTION-CONDITIONED TEXT/LAST SUMMARY")
    print("=" * 100)
    print("group                              layer    N    Eacc    Dacc    E+D-    E+D+    E-D-")

    show_groups = [
        "restricted_correct_repr_strong",
        "restricted_wrong_repr_strong",
        "restricted_wrong_repr_weak",
    ]
    for group in show_groups:
        for r in summary:
            if r["restricted_group"] != group:
                continue
            print(
                f"{group:<34s} "
                f"L{int(r['layer']):02d} "
                f"{int(r['n']):4d}  "
                f"{r['E_acc']:.3f}   "
                f"{r['D_acc']:.3f}   "
                f"{r['state_E+D-']:.3f}   "
                f"{r['state_E+D+']:.3f}   "
                f"{r['state_E-D-']:.3f}"
            )

    # 3) Recompute taxonomy ONLY for restricted-wrong + repr-strong.
    target_sids = {
        int(r["sid"])
        for r in joined
        if r["restricted_group"] == "restricted_wrong_repr_strong"
    }

    by_sid = defaultdict(list)
    for r in regrouped:
        sid = int(r["sid"])
        if sid in target_sids:
            by_sid[sid].append(r)

    tax_rows = []
    for sid, rs in sorted(by_sid.items()):
        rs = sorted(rs, key=lambda x: int(x["layer"]))
        layers = [int(r["layer"]) for r in rs]
        E = [as_bool(r["E_all_text_correct"]) for r in rs]
        D = [as_bool(r["D_last_correct"]) for r in rs]

        e_on = stable_onset(E, layers, args.stable_k)
        d_on = stable_onset(D, layers, args.stable_k)

        if e_on is None and d_on is None:
            stage = "text_evidence_failure_candidate"
        elif e_on is not None and d_on is None:
            stage = "text_to_last_failure_candidate"
        elif e_on is not None and d_on is not None:
            stage = "late_or_output_failure_candidate"
        else:
            stage = "bypass_or_unmeasured_carrier"

        tax_rows.append({
            "sid": sid,
            "E_stable_onset": -1 if e_on is None else e_on,
            "D_stable_onset": -1 if d_on is None else d_on,
            "candidate_failure_stage": stage,
            "ED_trajectory": "|".join(
                f"L{int(r['layer'])}:{r['state']}" for r in rs
            ),
        })

    write_csv(root / "restricted_wrong_repr_strong_per_sample.csv", tax_rows)

    tcounts = Counter(r["candidate_failure_stage"] for r in tax_rows)
    agg = [{
        "candidate_failure_stage": k,
        "count": v,
        "fraction": v / max(1, len(tax_rows)),
    } for k, v in tcounts.most_common()]
    write_csv(root / "restricted_wrong_repr_strong_taxonomy.csv", agg)

    print("\nrestricted_wrong_repr_strong taxonomy:")
    for r in agg:
        print(
            f"  {r['candidate_failure_stage']:<38s} "
            f"{int(r['count']):4d} ({float(r['fraction']):.3f})"
        )

    print("\nSaved:")
    for name in [
        "restricted_direction_groups.csv",
        "per_sample_layer_ED_states_restricted.csv",
        "restricted_group_layer_summary.csv",
        "restricted_wrong_repr_strong_per_sample.csv",
        "restricted_wrong_repr_strong_taxonomy.csv",
    ]:
        print(" ", root / name)


if __name__ == "__main__":
    main()
