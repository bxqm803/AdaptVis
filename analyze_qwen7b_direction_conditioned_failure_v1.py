#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direction-conditioned causal failure analysis for Qwen-7B.

Inputs
------
1) Existing direction scan:
   <direction-dir>/per_sample_layer_predictions.csv
2) Existing repo script:
   analyze_text_stream_visual_causal_transfer_v1.py

Pipeline
--------
A. Use residual direction correctness in a chosen window (default L14-L20)
   to label each test sample as repr_strong / repr_mixed / repr_weak.
B. Select all generation-wrong samples plus matched generation-correct controls.
C. Optionally call the existing text-stream causal script with roles all_text,last.
D. Join direction groups with causal sufficiency results:
      E = all_text sufficiency correct
      D = last sufficiency correct
   and summarize E/D trajectories for correct-vs-wrong samples.

Example
-------
CUDA_VISIBLE_DEVICES=0 python analyze_qwen7b_direction_conditioned_failure_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --spatial-window 14-20 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --causal-layers 14,16,18,20,22,24,26,27 \
  --output-dir output/qwen7b_direction_conditioned_failure_v1 \
  --run-causal \
  --overwrite-causal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--spatial-window", default="14-20")
    p.add_argument("--strong-frac", type=float, default=0.70)
    p.add_argument("--weak-frac", type=float, default=0.30)
    p.add_argument("--direction-stable-k", type=int, default=3)

    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--causal-layers", default="14,16,18,20,22,24,26,27")
    p.add_argument("--causal-stable-k", type=int, default=2)
    p.add_argument("--causal-script", default="analyze_text_stream_visual_causal_transfer_v1.py")

    p.add_argument("--control-ratio", type=float, default=1.0)
    p.add_argument("--max-wrong", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--run-causal", action="store_true")
    p.add_argument("--overwrite-causal", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def parse_layers(s: str) -> List[int]:
    s = str(s).strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        if b < a:
            raise ValueError(s)
        return list(range(a, b + 1))
    out = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    if not out:
        raise ValueError("No layers selected")
    return out


def longest_run(mask: Sequence[bool]) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def stable_onset(mask: Sequence[bool], layers: Sequence[int], k: int) -> Optional[int]:
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= k:
            return int(layers[i - k + 1])
    return None


def build_direction_groups(direction_dir: Path, window: Sequence[int], strong_frac: float,
                           weak_frac: float, stable_k: int) -> List[Dict[str, Any]]:
    path = direction_dir / "per_sample_layer_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    rows = read_csv(path)
    wanted = set(window)
    rows = [r for r in rows if r.get("metric") == "residual" and int(r["layer"]) in wanted]
    if not rows:
        raise RuntimeError("No residual direction rows found in requested window")

    by_sid = defaultdict(list)
    for r in rows:
        by_sid[int(r["sample_index"])].append(r)

    out = []
    for sid, rs in sorted(by_sid.items()):
        by_l = {int(r["layer"]): r for r in rs}
        layers = [l for l in window if l in by_l]
        mask = [to_bool(by_l[l]["probe_correct"]) for l in layers]
        margins = [float(by_l[l]["margin"]) for l in layers]
        frac = float(np.mean(mask))
        run = longest_run(mask)

        if frac >= strong_frac and run >= stable_k:
            rep = "repr_strong"
        elif frac <= weak_frac:
            rep = "repr_weak"
        else:
            rep = "repr_mixed"

        first = by_l[layers[0]]
        gen = str(first["generation_group"])
        diag = f"{gen}_{rep}"

        out.append({
            "sid": sid,
            "gt": first["relation"],
            "generation_group": gen,
            "generation_pred": first.get("generation_pred", ""),
            "representation_group": rep,
            "diagnostic_group": diag,
            "direction_window": ",".join(map(str, layers)),
            "direction_correct_fraction": frac,
            "direction_longest_correct_run": run,
            "direction_mean_margin": finite_mean(margins),
            "direction_min_margin": min(margins),
            "direction_max_margin": max(margins),
            "direction_correct_trajectory": "".join("1" if x else "0" for x in mask),
            "direction_pred_trajectory": "|".join(by_l[l]["probe_pred"] for l in layers),
            "direction_margin_trajectory": "|".join(f"{float(by_l[l]['margin']):+.4f}" for l in layers),
        })
    return out


def select_causal_samples(groups: Sequence[Mapping[str, Any]], control_ratio: float,
                          max_wrong: Optional[int], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    wrong = [dict(r) for r in groups if r["generation_group"] == "wrong"]
    if max_wrong is not None and len(wrong) > max_wrong:
        rng.shuffle(wrong)
        wrong = wrong[:max_wrong]

    n_control = int(round(len(wrong) * control_ratio))
    correct = [dict(r) for r in groups if r["generation_group"] == "correct"]
    strong = [r for r in correct if r["representation_group"] == "repr_strong"]
    rest = [r for r in correct if r["representation_group"] != "repr_strong"]

    selected_controls, used = [], set()
    wrong_counts = Counter(r["gt"] for r in wrong)
    relations = ("left", "right", "above", "below")

    for rel in relations:
        quota = int(round(n_control * wrong_counts[rel] / max(1, len(wrong))))
        cand = [r for r in strong if r["gt"] == rel and r["sid"] not in used]
        rng.shuffle(cand)
        take = cand[:quota]
        selected_controls.extend(take)
        used.update(r["sid"] for r in take)

    if len(selected_controls) < n_control:
        pool = [r for r in strong + rest if r["sid"] not in used]
        rng.shuffle(pool)
        selected_controls.extend(pool[:n_control - len(selected_controls)])

    selected = wrong + selected_controls[:n_control]
    return sorted(selected, key=lambda r: int(r["sid"]))


def run_causal(args, split_csv: Path, causal_dir: Path):
    cmd = [
        sys.executable, args.causal_script,
        "--dataset", args.dataset,
        "--data-root", args.data_root,
        "--model", args.model,
        "--device", args.device,
        "--attn-impl", args.attn_impl,
        "--layers", args.causal_layers,
        "--roles", "all_text,last",
        "--split-csv", str(split_csv),
        "--split", "test",
        "--output-dir", str(causal_dir),
    ]
    if args.overwrite_causal:
        cmd.append("--overwrite")
    print("\n[RUN CAUSAL]\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def postprocess(groups: Sequence[Mapping[str, Any]], causal_dir: Path,
                out_dir: Path, stable_k: int):
    base_path = causal_dir / "baseline_samples.csv"
    int_path = causal_dir / "per_sample_interventions.csv"
    if not base_path.exists() or not int_path.exists():
        print("\n[INFO] causal outputs not found; grouping files were still created")
        return

    group_by_sid = {int(r["sid"]): dict(r) for r in groups}
    base_rows = [r for r in read_csv(base_path) if int(r["sid"]) in group_by_sid]
    int_rows = [r for r in read_csv(int_path) if int(r["sid"]) in group_by_sid]
    baselines = {int(r["sid"]): r for r in base_rows}

    by_key, layer_set = {}, set()
    for r in int_rows:
        key = (int(r["sid"]), int(r["layer"]), r["condition"], r["role"])
        by_key[key] = r
        layer_set.add(int(r["layer"]))
    layers = sorted(layer_set)

    per_layer, per_sample = [], []

    for sid, b in sorted(baselines.items()):
        g = group_by_sid[sid]
        used, E, D, states = [], [], [], []

        for li in layers:
            er = by_key.get((sid, li, "sufficiency", "all_text"))
            dr = by_key.get((sid, li, "sufficiency", "last"))
            if er is None or dr is None:
                continue

            e, d = to_bool(er["correct"]), to_bool(dr["correct"])
            state = "E+D+" if (e and d) else "E+D-" if e else "E-D+" if d else "E-D-"
            used.append(li); E.append(e); D.append(d); states.append(state)

            per_layer.append({
                "sid": sid,
                "gt": g["gt"],
                "generation_group": g["generation_group"],
                "representation_group": g["representation_group"],
                "diagnostic_group": g["diagnostic_group"],
                "layer": li,
                "E_all_text_correct": int(e),
                "E_all_text_margin": float(er["margin"]),
                "D_last_correct": int(d),
                "D_last_margin": float(dr["margin"]),
                "state": state,
                "real_correct": int(to_bool(b["real_correct"])),
                "real_margin": float(b["real_margin"]),
            })

        if not used:
            continue

        e_on = stable_onset(E, used, stable_k)
        d_on = stable_onset(D, used, stable_k)

        if g["diagnostic_group"] == "wrong_repr_strong":
            if e_on is None and d_on is None:
                stage = "object_to_text_failure_candidate"
            elif e_on is not None and d_on is None:
                stage = "text_to_last_failure_candidate"
            elif e_on is not None and d_on is not None:
                stage = "downstream_or_output_failure_candidate"
            else:
                stage = "bypass_or_unmeasured_carrier"
        else:
            stage = "not_primary_group"

        per_sample.append({
            "sid": sid,
            "gt": g["gt"],
            "generation_group": g["generation_group"],
            "generation_pred": g["generation_pred"],
            "representation_group": g["representation_group"],
            "diagnostic_group": g["diagnostic_group"],
            "direction_correct_fraction": g["direction_correct_fraction"],
            "direction_mean_margin": g["direction_mean_margin"],
            "E_stable_onset": -1 if e_on is None else e_on,
            "D_stable_onset": -1 if d_on is None else d_on,
            "E_correct_fraction": float(np.mean(E)),
            "D_correct_fraction": float(np.mean(D)),
            "ED_trajectory": "|".join(f"L{li}:{st}" for li, st in zip(used, states)),
            "candidate_failure_stage": stage,
        })

    write_csv(out_dir / "per_sample_layer_ED_states.csv", per_layer)
    write_csv(out_dir / "per_sample_direction_conditioned_failure.csv", per_sample)

    buckets = defaultdict(list)
    for r in per_layer:
        buckets[(r["diagnostic_group"], r["layer"])].append(r)

    summary = []
    for (grp, li), rs in sorted(buckets.items()):
        summary.append({
            "diagnostic_group": grp,
            "layer": li,
            "n": len(rs),
            "E_acc": float(np.mean([r["E_all_text_correct"] for r in rs])),
            "D_acc": float(np.mean([r["D_last_correct"] for r in rs])),
            "E_margin": finite_mean(r["E_all_text_margin"] for r in rs),
            "D_margin": finite_mean(r["D_last_margin"] for r in rs),
            "state_E-D-": float(np.mean([r["state"] == "E-D-" for r in rs])),
            "state_E+D-": float(np.mean([r["state"] == "E+D-" for r in rs])),
            "state_E+D+": float(np.mean([r["state"] == "E+D+" for r in rs])),
            "state_E-D+": float(np.mean([r["state"] == "E-D+" for r in rs])),
        })
    write_csv(out_dir / "group_layer_summary.csv", summary)

    focus = [r for r in per_sample if r["diagnostic_group"] == "wrong_repr_strong"]
    counts = Counter(r["candidate_failure_stage"] for r in focus)
    tax = [{
        "candidate_failure_stage": k,
        "count": v,
        "fraction": v / max(1, len(focus)),
    } for k, v in counts.most_common()]
    write_csv(out_dir / "wrong_repr_strong_taxonomy.csv", tax)

    print("\n" + "=" * 100)
    print("DIRECTION-CONDITIONED TEXT/LAST SUMMARY")
    print("=" * 100)
    print("group                  layer    N    Eacc    Dacc    E+D-    E+D+    E-D-")
    for grp in ("correct_repr_strong", "wrong_repr_strong", "wrong_repr_weak"):
        for r in summary:
            if r["diagnostic_group"] == grp:
                print(
                    f"{grp:<22s} L{int(r['layer']):02d} {int(r['n']):4d}  "
                    f"{r['E_acc']:.3f}   {r['D_acc']:.3f}   "
                    f"{r['state_E+D-']:.3f}   {r['state_E+D+']:.3f}   {r['state_E-D-']:.3f}"
                )

    print("\nwrong_repr_strong candidate taxonomy:")
    for r in tax:
        print(f"  {r['candidate_failure_stage']:<40s} {int(r['count']):4d} ({r['fraction']:.3f})")


def main():
    args = parse_args()
    if not (0 <= args.weak_frac <= args.strong_frac <= 1):
        raise ValueError("Need 0 <= weak_frac <= strong_frac <= 1")
    if args.direction_stable_k < 1 or args.causal_stable_k < 1:
        raise ValueError("stable-k must be >= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window = parse_layers(args.spatial_window)

    groups = build_direction_groups(
        Path(args.direction_dir), window,
        args.strong_frac, args.weak_frac, args.direction_stable_k,
    )
    write_csv(out_dir / "direction_groups.csv", groups)

    print("\n" + "=" * 100)
    print("DIRECTION WINDOW GROUPS")
    print("=" * 100)
    counts = Counter(r["diagnostic_group"] for r in groups)
    for k, v in counts.most_common():
        print(f"{k:<24s} {v:4d}")

    selected = select_causal_samples(groups, args.control_ratio, args.max_wrong, args.seed)
    write_csv(out_dir / "causal_selected_samples.csv", selected)

    split_csv = out_dir / "causal_split.csv"
    write_csv(split_csv, [{
        "sid": int(r["sid"]),
        "split": "test",
        "gt": r["gt"],
        "diagnostic_group": r["diagnostic_group"],
    } for r in selected])

    nw = sum(r["generation_group"] == "wrong" for r in selected)
    nc = sum(r["generation_group"] == "correct" for r in selected)
    print(f"\nSelected causal samples: total={len(selected)} wrong={nw} correct_controls={nc}")
    print("split:", split_csv)

    causal_dir = out_dir / "causal"
    if args.run_causal:
        run_causal(args, split_csv, causal_dir)

    postprocess(groups, causal_dir, out_dir, args.causal_stable_k)

    (out_dir / "summary.json").write_text(json.dumps({
        "spatial_window": window,
        "strong_frac": args.strong_frac,
        "weak_frac": args.weak_frac,
        "direction_stable_k": args.direction_stable_k,
        "causal_layers": parse_layers(args.causal_layers),
        "direction_group_counts": dict(counts),
        "selected_total": len(selected),
        "selected_wrong": nw,
        "selected_correct_controls": nc,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
