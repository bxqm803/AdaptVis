#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def load_run(path: str) -> Dict[int, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for i, row in enumerate(data):
        sid = int(row.get("sample_id", i))
        out[sid] = row
    return out


def is_correct(row: Dict[str, Any]) -> bool:
    if "RawGenerationCorrect" in row:
        return bool(row["RawGenerationCorrect"])
    if "Correct" in row:
        return bool(row["Correct"])
    raise KeyError(f"Cannot find correctness field in row keys: {list(row.keys())}")


def generation(row: Dict[str, Any]) -> str:
    return str(row.get("RawGeneration", row.get("Generation", "")))


def add(stats: Dict[str, List[int]], key: str, sid: int) -> None:
    stats.setdefault(key, []).append(sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="JSON from w1=w2=1.0 run")
    ap.add_argument("--low", required=True, help="JSON from w1=w2=0.5 run")
    ap.add_argument("--high", required=True, help="JSON from w1=w2=1.5 run")
    ap.add_argument("--out", default="output/alpha_generation_compare_summary.json")
    ap.add_argument("--csv", default="output/alpha_generation_compare_per_sample.csv")
    args = ap.parse_args()

    base = load_run(args.base)
    low = load_run(args.low)
    high = load_run(args.high)

    ids = sorted(set(base) & set(low) & set(high))
    missing = {
        "missing_in_base": sorted((set(low) | set(high)) - set(base)),
        "missing_in_low": sorted((set(base) | set(high)) - set(low)),
        "missing_in_high": sorted((set(base) | set(low)) - set(high)),
    }
    if not ids:
        raise RuntimeError("No overlapping sample ids among the three runs.")

    stats = {
        "low_wrong_to_correct": [],
        "high_wrong_to_correct": [],
        "always_correct": [],
        "low_correct_to_wrong": [],
        "high_correct_to_wrong": [],
        "always_wrong": [],
    }
    exclusive = {
        "low_only_wrong_to_correct": [],
        "high_only_wrong_to_correct": [],
        "both_wrong_to_correct": [],
        "low_only_correct_to_wrong": [],
        "high_only_correct_to_wrong": [],
        "both_correct_to_wrong": [],
        "always_correct": [],
        "always_wrong": [],
        "other_mixed": [],
    }

    rows = []
    for sid in ids:
        b = is_correct(base[sid])
        l = is_correct(low[sid])
        h = is_correct(high[sid])
        gold = base[sid].get("Golden", low[sid].get("Golden", high[sid].get("Golden", "")))

        # Six non-exclusive categories requested by the user.
        if (not b) and l:
            add(stats, "low_wrong_to_correct", sid)
        if (not b) and h:
            add(stats, "high_wrong_to_correct", sid)
        if b and l and h:
            add(stats, "always_correct", sid)
        if b and (not l):
            add(stats, "low_correct_to_wrong", sid)
        if b and (not h):
            add(stats, "high_correct_to_wrong", sid)
        if (not b) and (not l) and (not h):
            add(stats, "always_wrong", sid)

        # Mutually exclusive version for cleaner interpretation.
        if b and l and h:
            ex = "always_correct"
        elif (not b) and (not l) and (not h):
            ex = "always_wrong"
        elif (not b) and l and (not h):
            ex = "low_only_wrong_to_correct"
        elif (not b) and (not l) and h:
            ex = "high_only_wrong_to_correct"
        elif (not b) and l and h:
            ex = "both_wrong_to_correct"
        elif b and (not l) and h:
            ex = "low_only_correct_to_wrong"
        elif b and l and (not h):
            ex = "high_only_correct_to_wrong"
        elif b and (not l) and (not h):
            ex = "both_correct_to_wrong"
        else:
            ex = "other_mixed"
        exclusive[ex].append(sid)

        rows.append({
            "sample_id": sid,
            "gold": gold,
            "base_correct_w1w2_1": b,
            "low_correct_w1w2_0p5": l,
            "high_correct_w1w2_1p5": h,
            "exclusive_case": ex,
            "base_generation": generation(base[sid]),
            "low_generation": generation(low[sid]),
            "high_generation": generation(high[sid]),
        })

    n = len(ids)
    summary = {
        "num_samples": n,
        "files": {"base": args.base, "low": args.low, "high": args.high},
        "accuracy": {
            "base_w1w2_1": sum(is_correct(base[i]) for i in ids) / n,
            "low_w1w2_0p5": sum(is_correct(low[i]) for i in ids) / n,
            "high_w1w2_1p5": sum(is_correct(high[i]) for i in ids) / n,
        },
        "six_categories_nonexclusive": {
            k: {"count": len(v), "sample_ids": v} for k, v in stats.items()
        },
        "exclusive_categories": {
            k: {"count": len(v), "sample_ids": v} for k, v in exclusive.items()
        },
        "missing_ids": missing,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[ACCURACY]")
    for k, v in summary["accuracy"].items():
        print(f"  {k}: {v:.6f}")

    print("\n[SIX CATEGORIES | non-exclusive]")
    for k, v in summary["six_categories_nonexclusive"].items():
        print(f"  {k}: {v['count']} | sample_ids={v['sample_ids']}")

    print("\n[EXCLUSIVE CATEGORIES]")
    for k, v in summary["exclusive_categories"].items():
        print(f"  {k}: {v['count']} | sample_ids={v['sample_ids']}")

    print("\nSaved summary:", out_path)
    print("Saved per-sample CSV:", csv_path)


if __name__ == "__main__":
    main()
