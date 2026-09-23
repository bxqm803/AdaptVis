#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-style Introduction figure for the matched-option preference trajectory.

Standalone plotting script: it does NOT import any AdaptVis experiment script.
It only reads the per-layer CSV produced by the hidden-state logit-lens run.

Expected CSV columns (minimum):
    sid, layer, prob_gap
Optional columns used only for validation / labels:
    gt, final_prediction_relation, target_relation, opposite_relation

Default figure semantics:
    prob_gap = P(option mapped to left) - P(option mapped to right)
where P is the A/B/C/D softmax obtained from:
    layer hidden state at prompt-last token -> final norm -> LM head.

Example:
    python plot_intro_mapped_option_preference_paper_v1.py \
      --csv output/intro_hiddenstate_decision_pair_v2/per_sample_layer.csv \
      --correct-sid 288 \
      --wrong-sid 63 \
      --output-dir output/intro_hiddenstate_decision_pair_v2/paper_figure
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default="output/intro_hiddenstate_decision_pair_v2/per_sample_layer.csv",
        help="Per-sample, per-layer CSV from the hidden-state -> final norm -> LM-head experiment.",
    )
    p.add_argument("--correct-sid", type=int, default=288)
    p.add_argument("--wrong-sid", type=int, default=63)
    p.add_argument("--target-relation", default="left")
    p.add_argument("--opposite-relation", default="right")
    p.add_argument(
        "--transition-after",
        type=float,
        default=26.5,
        help="Draw a subtle vertical marker here. Set <0 to disable.",
    )
    p.add_argument(
        "--show-transition-label",
        action="store_true",
        help="Add a small 'decision transition' label next to the marker.",
    )
    p.add_argument("--x-min", type=float, default=17.5)
    p.add_argument("--x-max", type=float, default=32.5)
    p.add_argument("--y-min", type=float, default=-1.05)
    p.add_argument("--y-max", type=float, default=1.05)
    p.add_argument(
        "--output-dir",
        default="output/intro_hiddenstate_decision_pair_v2/paper_figure",
    )
    p.add_argument("--basename", default="intro_mapped_option_preference")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_for_sid(
    rows: List[Dict[str, str]],
    sid: int,
) -> List[Dict[str, str]]:
    out = [r for r in rows if int(r["sid"]) == int(sid)]
    if not out:
        raise RuntimeError(f"sid={sid} not found in {len(rows)} rows")
    out.sort(key=lambda r: int(r["layer"]))
    return out


def validate_pair(
    correct_rows: List[Dict[str, str]],
    wrong_rows: List[Dict[str, str]],
    target: str,
    opposite: str,
) -> None:
    """Best-effort sanity checks; works even if optional columns are absent."""

    for name, rr in [
        ("correct", correct_rows),
        ("wrong", wrong_rows),
    ]:
        if "target_relation" in rr[0] and rr[0]["target_relation"]:
            got = rr[0]["target_relation"]
            if got != target:
                print(
                    f"[WARN] {name} sample target_relation={got!r}, "
                    f"requested {target!r}"
                )

        if "opposite_relation" in rr[0] and rr[0]["opposite_relation"]:
            got = rr[0]["opposite_relation"]
            if got != opposite:
                print(
                    f"[WARN] {name} sample opposite_relation={got!r}, "
                    f"requested {opposite!r}"
                )

    if "final_prediction_relation" in correct_rows[0]:
        got = correct_rows[0]["final_prediction_relation"]
        if got and got != target:
            print(
                f"[WARN] correct sid final_prediction_relation={got!r}, "
                f"expected {target!r}"
            )

    if "final_prediction_relation" in wrong_rows[0]:
        got = wrong_rows[0]["final_prediction_relation"]
        if got and got != opposite:
            print(
                f"[WARN] wrong sid final_prediction_relation={got!r}, "
                f"expected {opposite!r}"
            )


def arrow_label(target: str, final_rel: str) -> str:
    return f"GT {target} \u2192 final {final_rel}"


def main() -> None:
    a = parse_args()

    csv_path = Path(a.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_csv(csv_path)

    cr = rows_for_sid(rows, a.correct_sid)
    wr = rows_for_sid(rows, a.wrong_sid)

    validate_pair(
        cr,
        wr,
        a.target_relation,
        a.opposite_relation,
    )

    c_layers = np.asarray(
        [int(r["layer"]) for r in cr],
        dtype=int,
    )
    w_layers = np.asarray(
        [int(r["layer"]) for r in wr],
        dtype=int,
    )

    if not np.array_equal(c_layers, w_layers):
        raise RuntimeError(
            "The two samples do not share the same layer set."
        )

    c_gap = np.asarray(
        [float(r["prob_gap"]) for r in cr],
        dtype=float,
    )
    w_gap = np.asarray(
        [float(r["prob_gap"]) for r in wr],
        dtype=float,
    )

    outdir = Path(a.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.3, 3.65)
    )

    ax.plot(
        c_layers,
        c_gap,
        marker="o",
        markersize=5.0,
        linewidth=2.2,
        label=arrow_label(
            a.target_relation,
            a.target_relation,
        ),
    )

    ax.plot(
        w_layers,
        w_gap,
        marker="o",
        markersize=5.0,
        linewidth=2.2,
        linestyle="--",
        label=arrow_label(
            a.target_relation,
            a.opposite_relation,
        ),
    )

    # Neutral reference line.
    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.25,
        alpha=0.65,
    )

    # Optional transition marker.
    if a.transition_after >= 0:
        ax.axvline(
            a.transition_after,
            linestyle=":",
            linewidth=1.25,
            alpha=0.55,
        )

        if a.show_transition_label:
            ax.text(
                a.transition_after + 0.12,
                a.y_max - 0.08 * (a.y_max - a.y_min),
                "decision transition",
                fontsize=16,
                va="top",
                ha="left",
                alpha=0.8,
            )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------

    ax.set_xlim(
        a.x_min,
        a.x_max,
    )
    ax.set_ylim(
        a.y_min,
        a.y_max,
    )

    # Axis labels: 16 pt
    ax.set_xlabel(
        "Decoder layer",
        fontsize=16,
    )
    ax.set_ylabel(
        "Mapped-option preference",
        fontsize=16,
    )

    # Use even-numbered layers for a cleaner paper figure.
    xticks = [
        int(x)
        for x in c_layers
        if int(x) % 2 == 0
    ]
    ax.set_xticks(xticks)

    # Tick labels: 18 pt
    ax.tick_params(
        axis="both",
        labelsize=18,
    )

    # Very light grid.
    ax.grid(
        True,
        axis="both",
        linewidth=0.55,
        alpha=0.20,
    )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=16,
        handlelength=2.8,
        borderaxespad=0.55,
    )

    # No title: figure caption carries the message in the paper.
    fig.tight_layout(
        pad=0.7
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    png = outdir / f"{a.basename}.png"
    pdf = outdir / f"{a.basename}.pdf"

    fig.savefig(
        png,
        dpi=a.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[OK] correct sid : {a.correct_sid}"
    )
    print(
        f"[OK] wrong sid   : {a.wrong_sid}"
    )
    print(
        f"[OK] PNG         : {png}"
    )
    print(
        f"[OK] PDF         : {pdf}"
    )

    print()

    print(
        "Suggested caption definition:"
    )

    print(
        f"Mapped-option preference = "
        f"P(pi({a.target_relation})) - "
        f"P(pi({a.opposite_relation})), "
        "where probabilities are obtained by applying "
        "the model's final normalization and LM head "
        "to the prompt-final hidden state at each decoder layer "
        "and softmaxing over A/B/C/D."
    )


if __name__ == "__main__":
    main()
