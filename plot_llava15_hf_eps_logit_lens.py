#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot per-sample LLaVA epsilon logit-lens traces written by
run_llava15_hf_eps_logit_lens.py.

Creates, for every SID:
  1. active-lens four-way gold margin across depth;
  2. active-lens four-way gold probability across depth;
  3. final four-way probabilities for each epsilon.

No external package beyond matplotlib/pandas is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

RELATIONS = ("left", "right", "on", "under")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lens",
        default="active",
        choices=["active", "fixed_reference"],
        help="Which stored lens to plot.",
    )
    parser.add_argument(
        "--sids",
        default=None,
        help="Optional comma-separated subset of SIDs.",
    )
    return parser.parse_args()


def parse_sids(text: str | None):
    if text is None or not text.strip():
        return None
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def lens_key(name: str) -> str:
    return "active_lens" if name == "active" else "fixed_reference_lens"


def eps_sort_key(text: str) -> float:
    return float(text)


def stage_depth(entry: Dict[str, Any]) -> int:
    return int(entry["layer_index"]) + 1


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted_sids = parse_sids(args.sids)
    key = lens_key(args.lens)

    by_sid: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for eps, records in payload["eps_runs"].items():
        for record in records:
            sid = int(record["sid"])
            if wanted_sids is not None and sid not in wanted_sids:
                continue
            by_sid.setdefault(sid, {})[eps] = record

    summary_rows: List[Dict[str, Any]] = []
    for sid in sorted(by_sid):
        eps_records = by_sid[sid]
        eps_values = sorted(eps_records, key=eps_sort_key)
        first_record = eps_records[eps_values[0]]
        gold = str(first_record["gold"]).lower()

        # Figure 1: four-way gold margin by depth.
        plt.figure(figsize=(9, 4.5))
        for eps in eps_values:
            entries = eps_records[eps].get(key)
            if entries is None:
                continue
            x = [stage_depth(entry) for entry in entries]
            y = [entry["gold_margin_vs_best_other"] for entry in entries]
            plt.plot(x, y, marker="o", markersize=2.6, linewidth=1.5, label=f"eps={eps}")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("Depth (0=input to layer 0; 32=after final decoder layer)")
        plt.ylabel(f"Gold '{gold}' logit − best other relation logit")
        plt.title(f"SID {sid}: four-way gold margin ({args.lens} lens)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"sid{sid:03d}_{args.lens}_gold_margin.png", dpi=200)
        plt.close()

        # Figure 2: four-way gold probability by depth.
        plt.figure(figsize=(9, 4.5))
        for eps in eps_values:
            entries = eps_records[eps].get(key)
            if entries is None:
                continue
            x = [stage_depth(entry) for entry in entries]
            y = [entry["gold_prob_4way"] for entry in entries]
            plt.plot(x, y, marker="o", markersize=2.6, linewidth=1.5, label=f"eps={eps}")
        plt.xlabel("Depth (0=input to layer 0; 32=after final decoder layer)")
        plt.ylabel(f"Four-way probability of gold '{gold}'")
        plt.ylim(-0.02, 1.02)
        plt.title(f"SID {sid}: four-way gold probability ({args.lens} lens)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"sid{sid:03d}_{args.lens}_gold_probability.png", dpi=200)
        plt.close()

        # Figure 3: final four-way category probabilities per epsilon.
        plt.figure(figsize=(9, 4.5))
        x_positions = list(range(len(eps_values)))
        width = 0.18
        for relation_index, relation in enumerate(RELATIONS):
            x = [pos + (relation_index - 1.5) * width for pos in x_positions]
            y = [
                eps_records[eps]["final_output"]["fourway_probs"][relation]
                for eps in eps_values
            ]
            plt.bar(x, y, width=width, label=relation)
        plt.xticks(x_positions, [f"eps={eps}" for eps in eps_values])
        plt.ylim(0.0, 1.02)
        plt.ylabel("Four-way probability")
        plt.title(f"SID {sid}: final four-way distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"sid{sid:03d}_final_fourway_probs.png", dpi=200)
        plt.close()

        for eps in eps_values:
            final = eps_records[eps]["final_output"]
            summary_rows.append(
                {
                    "sid": sid,
                    "gold": gold,
                    "eps": eps,
                    "final_fourway_prediction": final["fourway_prediction"],
                    "final_gold_margin": final["gold_margin_vs_best_other"],
                    "final_gold_prob_4way": final["gold_prob_4way"],
                    "final_gold_candidate_vocab_prob": final["gold_candidate_vocab_prob"],
                    "final_lens_max_abs_error": eps_records[eps]["final_lens_match"]["max_abs_error"],
                }
            )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved {len(summary_rows)} epsilon-by-SID summaries to {output_dir / 'summary.csv'}")
    print(f"Saved plots for {len(by_sid)} SIDs under {output_dir}")


if __name__ == "__main__":
    main()
