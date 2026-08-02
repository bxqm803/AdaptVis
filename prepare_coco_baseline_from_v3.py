#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert v3 storage/transport extraction rows into baseline_generation.jsonl.

Input rows are produced by:
    analyze_spatial_storage_transport_utilization_v3.py

Only parsed four-relation generations are written. Unparsed SIDs are saved to a
sidecar text file so downstream detector sample counts remain explicit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

RELATIONS = ("left", "right", "above", "below")
ALIASES = {
    "left": "left",
    "right": "right",
    "above": "above",
    "below": "below",
    "on": "above",
    "under": "below",
}


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ALIASES:
        return ALIASES[text]
    tokens = (
        text.replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("\n", " ")
        .split()
    )
    found = [ALIASES[token] for token in tokens if token in ALIASES]
    return found[0] if found else None


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="v3 extraction.jsonl")
    parser.add_argument("--output", required=True, help="baseline_generation.jsonl")
    parser.add_argument(
        "--unparsed-output",
        default=None,
        help="Optional path for one unparsed SID per line. Defaults beside --output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    unparsed_path = (
        Path(args.unparsed_output)
        if args.unparsed_output
        else output_path.with_name("baseline_generation_unparsed_sids.txt")
    )

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    by_sid: Dict[int, Dict[str, Any]] = {}
    unparsed: list[int] = []
    total = 0

    for row in read_jsonl(input_path):
        total += 1
        sid = int(row["sid"])
        gt = normalize_relation(row.get("gt"))
        prediction = normalize_relation(
            row.get("baseline_generated_prediction", row.get("baseline_generation_prediction"))
        )

        if gt not in RELATIONS:
            raise RuntimeError(f"SID {sid}: invalid GT {row.get('gt')!r}")

        if prediction not in RELATIONS:
            unparsed.append(sid)
            continue

        correct = bool(prediction == gt)
        stored_correct = row.get("baseline_generation_correct")
        if stored_correct is not None and bool(stored_correct) != correct:
            raise RuntimeError(
                f"SID {sid}: correctness mismatch: stored={stored_correct}, "
                f"computed={correct}, gt={gt}, prediction={prediction}"
            )

        by_sid[sid] = {
            "sid": sid,
            "gt": gt,
            "prediction": prediction,
            "correct": correct,
            "parsed": True,
            "generated_text": row.get("baseline_generated_text", ""),
            "source": str(input_path),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sid in sorted(by_sid):
            handle.write(json.dumps(by_sid[sid], ensure_ascii=False) + "\n")

    unparsed_path.parent.mkdir(parents=True, exist_ok=True)
    unparsed_path.write_text(
        "".join(f"{sid}\n" for sid in sorted(set(unparsed))),
        encoding="utf-8",
    )

    rows = list(by_sid.values())
    relation_counts = Counter(row["gt"] for row in rows)
    correct_counts = Counter(bool(row["correct"]) for row in rows)

    print(f"Input rows      : {total}")
    print(f"Parsed rows     : {len(rows)}")
    print(f"Unparsed rows   : {len(set(unparsed))}")
    print(f"Generation ACC  : {correct_counts[True] / max(len(rows), 1):.6f}")
    print(f"Relation counts : {dict(relation_counts)}")
    print(f"Saved baseline  : {output_path}")
    print(f"Saved unparsed  : {unparsed_path}")


if __name__ == "__main__":
    main()
