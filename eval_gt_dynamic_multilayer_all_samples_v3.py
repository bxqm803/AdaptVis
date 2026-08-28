#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-7B oracle dynamic multi-layer Direction intervention on ALL test samples.

Goal
----
Use the known GT relation to test whether repeated multi-layer correction of
Direction-space competition can improve prediction, and whether it harms
originally-correct samples.

Two main interventions are run:

1) multi_suppress
   At every selected layer, find the strongest NON-GT relation coordinate and,
   only if it is above the normal correct-control target, suppress the excess.
   The GT coordinate is held fixed at that layer.

2) multi_both
   At every selected layer:
     - boost the GT coordinate if it is below its normal correct-control target;
     - suppress the strongest NON-GT coordinate if it is above its normal
       correct-control target.
   Both requested coordinate changes are solved jointly with a minimum-L2 edit.

Both are DYNAMIC:
- after an edit at L14, the model runs normally to L15;
- at L15 we recompute the current Direction vector from the already-edited run;
- choose the current strongest competitor again;
- correct only the deficit/excess that remains;
- repeat through all requested layers.

This directly tests the hypothesis that wrong spatial evidence may be
re-generated / amplified across multiple layers.

IMPORTANT
---------
This is an ORACLE mechanism experiment, not a deployable method:
- GT is explicitly used to decide which relation is correct.
- Targets are learned from the previous restricted_correct_repr_strong controls.

The script evaluates ALL samples in the requested split (default: test),
including samples that are already correct. It therefore reports:
- Wrong -> Correct (W2C)
- Correct -> Wrong (C2W)
- net gain
- overall accuracy
- margin changes
- per-layer edit/trigger traces

Random controls
---------------
For multi_suppress and multi_both, a cumulative norm-matched random control is
also run. At each layer it computes the same intended edit norm but applies a
random vector orthogonal to span{GT prototype, current competitor prototype}.
Thus it has similar perturbation magnitude without directly changing those two
relation coordinates.

Expected existing files
-----------------------
<direction-dir>/vectors.npz
<direction-dir>/sample_split_and_generation.csv
<group-root>/restricted_direction_groups.csv

Expected helper scripts in repo root:
  extract_two_object_relation_states.py
  analyze_layerwise_direction_failure_scan_v1.py
  analyze_text_stream_visual_causal_transfer_v1.py
  analyze_direction_geometry_and_causality_v1.py

Recommended run
---------------
CUDA_VISIBLE_DEVICES=0 python eval_gt_dynamic_multilayer_all_samples_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --group-root output/qwen7b_direction_conditioned_failure_v1 \
  --layers 14,15,16,17,18,19,20 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --split test \
  --output-dir output/qwen7b_gt_dynamic_multilayer_all_v1 \
  --overwrite

Smoke test
----------
CUDA_VISIBLE_DEVICES=0 python eval_gt_dynamic_multilayer_all_samples_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --group-root output/qwen7b_direction_conditioned_failure_v1 \
  --layers 14,16,18,20 \
  --model qwen-7b \
  --device cuda:0 \
  --split test \
  --max-samples 20 \
  --output-dir output/qwen7b_gt_dynamic_multilayer_all_smoke \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as dirscan
import analyze_text_stream_visual_causal_transfer_v1 as causal
import analyze_direction_geometry_and_causality_v1 as geom


RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-10


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--group-root", required=True)
    p.add_argument("--layers", default="14,15,16,17,18,19,20")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "all"],
        help="Which cached direction split to evaluate. Default test = all held-out samples.",
    )
    p.add_argument(
        "--control-group",
        default="restricted_correct_repr_strong",
        help="Previous group used to define normal correct-state coordinate targets.",
    )
    p.add_argument(
        "--target-stat",
        default="median",
        choices=["median", "mean"],
    )
    p.add_argument("--min-controls", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--max-edit-norm",
        type=float,
        default=0.0,
        help="Optional per-layer edit-norm cap; <=0 disables clipping.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def parse_layers(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("No layers selected.")
    vals = list(dict.fromkeys(vals))
    if vals != sorted(vals):
        raise ValueError("Layers must be in ascending order.")
    return vals


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def frac(xs: Iterable[bool]) -> float:
    vals = [bool(x) for x in xs]
    return float(np.mean(vals)) if vals else float("nan")


def norm_rel(x: Any) -> str:
    return dirscan.norm_relation(str(x))


# =============================================================================
# Load assets and previous control targets
# =============================================================================

def load_assets(
    direction_dir: Path,
    group_root: Path,
    layers: Sequence[int],
):
    arr, groups, idx_by_sid, labels, residual, codebooks = geom.load_assets(
        direction_dir, group_root, layers
    )

    split_rows = read_csv(direction_dir / "sample_split_and_generation.csv")
    split_by_sid = {
        int(r["sample_index"]): str(r["split"]).strip()
        for r in split_rows
    }

    return (
        arr,
        groups,
        idx_by_sid,
        labels,
        residual,
        codebooks,
        split_rows,
        split_by_sid,
    )


def _target_stat(vals, which):
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        raise RuntimeError("Cannot compute target statistic from zero samples.")
    return float(np.median(x) if which == "median" else np.mean(x))


def build_targets(
    *,
    groups,
    idx_by_sid,
    labels,
    residual,
    codebooks,
    layers,
    control_group,
    target_stat,
    min_controls,
):
    """
    Build the same correct-control targets used by the prior single-layer test.

    gt_targets[(layer, gt)]:
        typical q dot prototype_gt among restricted-correct + repr-strong
        controls with the same GT.

    wrong_targets[(layer, gt, foil)]:
        typical q dot prototype_foil among those same correct controls.
    """
    controls = [
        r for r in groups
        if str(r["restricted_group"]) == control_group
        and int(r["sid"]) in idx_by_sid
    ]
    if not controls:
        raise RuntimeError(f"No controls found for {control_group}")

    gt_targets = {}
    wrong_targets = {}
    audit = []

    for li in layers:
        cb = codebooks[li]
        center = cb["center"]
        protos = cb["protos"]

        for gt in RELATIONS:
            same_gt = []
            for r in controls:
                sid = int(r["sid"])
                idx = idx_by_sid[sid]
                row_gt = norm_rel(r.get("gt", labels[idx]))
                if row_gt == gt:
                    same_gt.append(r)

            # Keep same-GT matching. min_controls is diagnostic only here;
            # falling back across GTs would mix relation-specific scales.
            if len(same_gt) == 0:
                continue

            q_list = []
            for r in same_gt:
                idx = idx_by_sid[int(r["sid"])]
                q_list.append(residual[idx, li, :] - center)

            gt_proto = protos[REL2ID[gt]]
            gt_vals = [float(q @ gt_proto) for q in q_list]
            gt_targets[(li, gt)] = _target_stat(gt_vals, target_stat)

            audit.append({
                "layer": li,
                "gt": gt,
                "foil": "GT",
                "n_controls": len(q_list),
                "below_min_controls": int(len(q_list) < min_controls),
                "target_coordinate": gt_targets[(li, gt)],
                "statistic": target_stat,
            })

            for foil in RELATIONS:
                if foil == gt:
                    continue
                foil_proto = protos[REL2ID[foil]]
                vals = [float(q @ foil_proto) for q in q_list]
                wrong_targets[(li, gt, foil)] = _target_stat(
                    vals, target_stat
                )
                audit.append({
                    "layer": li,
                    "gt": gt,
                    "foil": foil,
                    "n_controls": len(q_list),
                    "below_min_controls": int(len(q_list) < min_controls),
                    "target_coordinate": wrong_targets[(li, gt, foil)],
                    "statistic": target_stat,
                })

    return gt_targets, wrong_targets, audit


def dual_edit_matrix(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Minimum-L2 coordinate edit basis.

    Compute in float32 because torch.linalg.inv does not support BF16/FP16
    reliably on CUDA. Cast back to the model dtype afterward.
    """
    out_dtype = g.dtype
    A32 = torch.stack([g.float(), w.float()], dim=1)  # [D, 2]
    gram32 = A32.transpose(0, 1) @ A32
    eye32 = torch.eye(2, device=A32.device, dtype=torch.float32)
    inv32 = torch.linalg.inv(gram32 + 1e-6 * eye32)
    return (A32 @ inv32).to(dtype=out_dtype)


def random_orthogonal_delta(
    *,
    true_delta: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """
    Norm-matched random edit orthogonal to span{g,w}.
    Linear solve is performed in float32 for BF16 compatibility.
    """
    target_norm = true_delta.float().norm()
    if float(target_norm.detach().cpu()) <= EPS:
        return torch.zeros_like(true_delta)

    rng = np.random.default_rng(seed)
    rv32 = torch.from_numpy(
        rng.standard_normal(true_delta.numel()).astype(np.float32)
    ).to(device=true_delta.device, dtype=torch.float32)

    g32 = g.float()
    w32 = w.float()
    A32 = torch.stack([g32, w32], dim=1)
    gram32 = A32.transpose(0, 1) @ A32
    eye32 = torch.eye(2, device=A32.device, dtype=torch.float32)

    coeff32 = torch.linalg.solve(
        gram32 + 1e-6 * eye32,
        A32.transpose(0, 1) @ rv32,
    )
    rv32 = rv32 - A32 @ coeff32
    nr = rv32.norm()

    if float(nr.detach().cpu()) <= EPS:
        return torch.zeros_like(true_delta)

    return (rv32 * (target_norm / nr)).to(dtype=true_delta.dtype)


# =============================================================================
# Dynamic multi-layer edit
# =============================================================================

class DynamicLayerEdit:
    """
    One dynamic edit hook at one layer.

    The current relation vector is recomputed from the actual hidden state that
    reaches this layer, so previous-layer edits are automatically incorporated.
    """

    def __init__(
        self,
        *,
        module,
        layer: int,
        sid: int,
        gt: str,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        noimg_diff_cpu: torch.Tensor,
        center_cpu: torch.Tensor,
        protos_cpu: torch.Tensor,
        target_gt: float,
        wrong_targets_for_gt: Mapping[str, float],
        mode: str,
        seed: int,
        max_edit_norm: float,
    ):
        self.layer = int(layer)
        self.sid = int(sid)
        self.gt = gt
        self.subj = [int(x) for x in subject_positions]
        self.ref = [int(x) for x in reference_positions]
        self.noimg_diff_cpu = noimg_diff_cpu
        self.center_cpu = center_cpu
        self.protos_cpu = protos_cpu
        self.target_gt = float(target_gt)
        self.wrong_targets_for_gt = {
            str(k): float(v) for k, v in wrong_targets_for_gt.items()
        }
        self.mode = str(mode)
        self.seed = int(seed)
        self.max_edit_norm = float(max_edit_norm)

        self.applied = 0
        self.trace: Dict[str, Any] = {}
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        if self.applied:
            return output

        x = causal.first_tensor(output)
        hs = geom.pool_hidden(x, self.subj)
        hr = geom.pool_hidden(x, self.ref)

        noimg = self.noimg_diff_cpu.to(x.device, dtype=x.dtype)
        center = self.center_cpu.to(x.device, dtype=x.dtype)
        protos = self.protos_cpu.to(x.device, dtype=x.dtype)

        q = (hs - hr) - noimg - center

        gt_i = REL2ID[self.gt]
        gt_proto = protos[gt_i]
        gt_coord = gt_proto @ q

        # Strongest current non-GT relation coordinate.
        coords = torch.stack([protos[i] @ q for i in range(4)])
        wrong_ids = [i for i in range(4) if i != gt_i]
        wrong_vals = coords[wrong_ids]
        local_j = int(torch.argmax(wrong_vals).detach().cpu())
        comp_i = wrong_ids[local_j]
        competitor = RELATIONS[comp_i]
        comp_proto = protos[comp_i]
        comp_coord = coords[comp_i]

        wrong_target = float(self.wrong_targets_for_gt[competitor])

        gt_deficit_t = torch.clamp(
            torch.as_tensor(
                self.target_gt, device=x.device, dtype=x.dtype
            ) - gt_coord,
            min=0.0,
        )
        wrong_excess_t = torch.clamp(
            comp_coord - torch.as_tensor(
                wrong_target, device=x.device, dtype=x.dtype
            ),
            min=0.0,
        )

        gt_deficit = float(gt_deficit_t.detach().float().cpu())
        wrong_excess = float(wrong_excess_t.detach().float().cpu())

        is_random = self.mode.startswith("random_")
        true_mode = self.mode.replace("random_", "") if is_random else self.mode

        # Solve the requested coordinate changes jointly. This lets suppress-only
        # hold the GT coordinate fixed rather than accidentally damaging it.
        B = dual_edit_matrix(gt_proto, comp_proto)

        if true_mode == "suppress":
            c = torch.stack([
                torch.zeros_like(wrong_excess_t),
                -wrong_excess_t,
            ])
        elif true_mode == "both":
            c = torch.stack([
                gt_deficit_t,
                -wrong_excess_t,
            ])
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        true_delta = B @ c

        if is_random:
            delta = random_orthogonal_delta(
                true_delta=true_delta,
                g=gt_proto,
                w=comp_proto,
                seed=self.seed,
            )
        else:
            delta = true_delta

        unclipped_norm = float(delta.detach().float().norm().cpu())

        clipped = False
        if self.max_edit_norm > 0 and unclipped_norm > self.max_edit_norm:
            scale = self.max_edit_norm / max(unclipped_norm, EPS)
            delta = delta * scale
            clipped = True

        delta_norm = float(delta.detach().float().norm().cpu())

        q_after = q + delta
        post_gt = float((gt_proto @ q_after).detach().float().cpu())
        post_comp = float((comp_proto @ q_after).detach().float().cpu())

        # Preserve object-pair mean:
        #   h_sub += delta/2
        #   h_ref -= delta/2
        # so their difference changes by exactly +delta.
        y = x.clone()
        half = delta / 2.0
        y[0, self.subj, :] = y[0, self.subj, :] + half
        y[0, self.ref, :] = y[0, self.ref, :] - half

        self.trace = {
            "sid": self.sid,
            "layer": self.layer,
            "mode": self.mode,
            "gt": self.gt,
            "competitor": competitor,
            "pre_gt_coordinate": float(gt_coord.detach().float().cpu()),
            "gt_target": self.target_gt,
            "gt_deficit": gt_deficit,
            "gt_boost_triggered": int(
                true_mode == "both" and gt_deficit > 1e-8
            ),
            "pre_competitor_coordinate": float(
                comp_coord.detach().float().cpu()
            ),
            "competitor_target": wrong_target,
            "wrong_excess": wrong_excess,
            "wrong_suppress_triggered": int(wrong_excess > 1e-8),
            "post_gt_coordinate": post_gt,
            "post_competitor_coordinate": post_comp,
            "delta_norm": delta_norm,
            "unclipped_delta_norm": unclipped_norm,
            "clipped": int(clipped),
            "random_control": int(is_random),
        }

        self.applied += 1
        return causal.replace_first_tensor(output, y)

    def validate(self):
        if self.applied != 1:
            raise RuntimeError(
                f"sid={self.sid} L{self.layer} {self.mode}: "
                f"hook applied {self.applied} times"
            )

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


class MultiLayerDynamicIntervention:
    def __init__(
        self,
        *,
        decoder_layers,
        selected_layers: Sequence[int],
        sid: int,
        gt: str,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        noimg_diffs: Mapping[int, torch.Tensor],
        codebooks: Mapping[int, Mapping[str, np.ndarray]],
        gt_targets: Mapping[Tuple[int, str], float],
        wrong_targets: Mapping[Tuple[int, str, str], float],
        mode: str,
        seed: int,
        max_edit_norm: float,
    ):
        self.patches: List[DynamicLayerEdit] = []

        for li in selected_layers:
            wrong_for_gt = {
                r: wrong_targets[(li, gt, r)]
                for r in RELATIONS
                if r != gt and (li, gt, r) in wrong_targets
            }
            if len(wrong_for_gt) != 3:
                raise RuntimeError(
                    f"Missing wrong targets for L{li}, GT={gt}: "
                    f"{sorted(wrong_for_gt)}"
                )

            if (li, gt) not in gt_targets:
                raise RuntimeError(f"Missing GT target for L{li}, GT={gt}")

            patch = DynamicLayerEdit(
                module=decoder_layers[li],
                layer=li,
                sid=sid,
                gt=gt,
                subject_positions=subject_positions,
                reference_positions=reference_positions,
                noimg_diff_cpu=noimg_diffs[li],
                center_cpu=torch.from_numpy(codebooks[li]["center"]),
                protos_cpu=torch.from_numpy(codebooks[li]["protos"]),
                target_gt=gt_targets[(li, gt)],
                wrong_targets_for_gt=wrong_for_gt,
                mode=mode,
                seed=seed + li * 10007,
                max_edit_norm=max_edit_norm,
            )
            self.patches.append(patch)

    def validate(self):
        for p in self.patches:
            p.validate()

    def traces(self) -> List[Dict[str, Any]]:
        return [dict(p.trace) for p in self.patches]

    def close(self):
        for p in self.patches:
            p.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def forward_dynamic(
    *,
    model,
    batch,
    token_map,
    gt,
    decoder_layers,
    layers,
    sid,
    subj_pos,
    ref_pos,
    noimg_diffs,
    codebooks,
    gt_targets,
    wrong_targets,
    mode,
    seed,
    max_edit_norm,
):
    patcher = MultiLayerDynamicIntervention(
        decoder_layers=decoder_layers,
        selected_layers=layers,
        sid=sid,
        gt=gt,
        subject_positions=subj_pos,
        reference_positions=ref_pos,
        noimg_diffs=noimg_diffs,
        codebooks=codebooks,
        gt_targets=gt_targets,
        wrong_targets=wrong_targets,
        mode=mode,
        seed=seed,
        max_edit_norm=max_edit_norm,
    )
    try:
        with patcher:
            result = causal.score_forward(model, batch, token_map, gt)
        patcher.validate()
        traces = patcher.traces()
        return result, traces
    finally:
        patcher.close()


# =============================================================================
# Summaries
# =============================================================================

def summarize_final(rows: Sequence[Mapping[str, Any]], out_dir: Path):
    modes = (
        "multi_suppress",
        "multi_both",
        "random_multi_suppress",
        "random_multi_both",
    )

    total = len(rows)
    baseline_correct_n = sum(int(r["baseline_correct"]) for r in rows)
    baseline_wrong_n = total - baseline_correct_n

    summary = []

    for mode in modes:
        correct_n = sum(int(r[f"{mode}_correct"]) for r in rows)
        w2c = sum(int(r[f"{mode}_W_to_C"]) for r in rows)
        c2w = sum(int(r[f"{mode}_C_to_W"]) for r in rows)
        unchanged_correct = sum(
            int(r["baseline_correct"]) and int(r[f"{mode}_correct"])
            for r in rows
        )
        unchanged_wrong = sum(
            (not int(r["baseline_correct"])) and
            (not int(r[f"{mode}_correct"]))
            for r in rows
        )

        summary.append({
            "mode": mode,
            "n": total,
            "baseline_acc": baseline_correct_n / total if total else float("nan"),
            "intervention_acc": correct_n / total if total else float("nan"),
            "accuracy_gain": (
                correct_n - baseline_correct_n
            ) / total if total else float("nan"),
            "baseline_correct_n": baseline_correct_n,
            "baseline_wrong_n": baseline_wrong_n,
            "W_to_C": w2c,
            "W_to_C_rate_among_wrong": (
                w2c / baseline_wrong_n if baseline_wrong_n else float("nan")
            ),
            "C_to_W": c2w,
            "C_to_W_rate_among_correct": (
                c2w / baseline_correct_n if baseline_correct_n else float("nan")
            ),
            "net_correct_gain": w2c - c2w,
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
            "mean_margin_gain_all": mean(
                r[f"{mode}_margin_gain"] for r in rows
            ),
            "mean_margin_gain_baseline_correct": mean(
                r[f"{mode}_margin_gain"]
                for r in rows if int(r["baseline_correct"])
            ),
            "mean_margin_gain_baseline_wrong": mean(
                r[f"{mode}_margin_gain"]
                for r in rows if not int(r["baseline_correct"])
            ),
            "pred_change_rate": frac(
                int(r[f"{mode}_pred_changed"]) == 1 for r in rows
            ),
        })

    write_csv(out_dir / "final_summary.csv", summary)

    print("\n" + "=" * 150)
    print("ALL-SAMPLE DYNAMIC MULTI-LAYER RESULT")
    print("=" * 150)
    print(
        f"N={total}  baseline_correct={baseline_correct_n}  "
        f"baseline_wrong={baseline_wrong_n}  "
        f"baseline_acc={baseline_correct_n/total:.4f}"
    )
    print(
        "mode                    acc      gain      W2C   W2C/wrong   "
        "C2W   C2W/correct   net    marginGain(correct/wrong)"
    )
    for r in summary:
        print(
            f"{r['mode']:<23s} "
            f"{r['intervention_acc']:.4f}  "
            f"{r['accuracy_gain']:+.4f}   "
            f"{int(r['W_to_C']):3d}   "
            f"{r['W_to_C_rate_among_wrong']:.3f}       "
            f"{int(r['C_to_W']):3d}   "
            f"{r['C_to_W_rate_among_correct']:.3f}        "
            f"{int(r['net_correct_gain']):+4d}   "
            f"{r['mean_margin_gain_baseline_correct']:+.3f}/"
            f"{r['mean_margin_gain_baseline_wrong']:+.3f}"
        )

    # Specific effect vs random counterpart.
    by_mode = {r["mode"]: r for r in summary}
    specific_rows = []
    for real_mode, rand_mode in [
        ("multi_suppress", "random_multi_suppress"),
        ("multi_both", "random_multi_both"),
    ]:
        a = by_mode[real_mode]
        b = by_mode[rand_mode]
        specific_rows.append({
            "mode": real_mode,
            "accuracy_gain_minus_random": (
                a["accuracy_gain"] - b["accuracy_gain"]
            ),
            "W_to_C_minus_random": a["W_to_C"] - b["W_to_C"],
            "C_to_W_minus_random": a["C_to_W"] - b["C_to_W"],
            "net_gain_minus_random": (
                a["net_correct_gain"] - b["net_correct_gain"]
            ),
            "margin_gain_all_minus_random": (
                a["mean_margin_gain_all"] - b["mean_margin_gain_all"]
            ),
            "margin_gain_wrong_minus_random": (
                a["mean_margin_gain_baseline_wrong"]
                - b["mean_margin_gain_baseline_wrong"]
            ),
            "margin_gain_correct_minus_random": (
                a["mean_margin_gain_baseline_correct"]
                - b["mean_margin_gain_baseline_correct"]
            ),
        })
    write_csv(out_dir / "specific_vs_random.csv", specific_rows)

    print("\nSpecific effect vs norm-matched cumulative random control:")
    for r in specific_rows:
        print(
            f"{r['mode']:<18s} "
            f"acc_specific={r['accuracy_gain_minus_random']:+.4f}  "
            f"W2C_specific={int(r['W_to_C_minus_random']):+d}  "
            f"C2W_specific={int(r['C_to_W_minus_random']):+d}  "
            f"net_specific={int(r['net_gain_minus_random']):+d}  "
            f"wrongMarginSpecific={r['margin_gain_wrong_minus_random']:+.4f}"
        )

    return summary


def summarize_layer_traces(
    traces: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
):
    baseline_correct_by_sid = {
        int(r["sid"]): bool(int(r["baseline_correct"])) for r in final_rows
    }

    buckets = defaultdict(list)
    for t in traces:
        buckets[(str(t["mode"]), int(t["layer"]))].append(t)

    summary = []
    for (mode, li), rs in sorted(buckets.items()):
        corr = [
            r for r in rs if baseline_correct_by_sid.get(int(r["sid"]), False)
        ]
        wrong = [
            r for r in rs if not baseline_correct_by_sid.get(int(r["sid"]), False)
        ]

        competitor_counts = Counter(str(r["competitor"]) for r in rs)

        summary.append({
            "mode": mode,
            "layer": li,
            "n": len(rs),
            "wrong_suppress_trigger_rate": frac(
                int(r["wrong_suppress_triggered"]) == 1 for r in rs
            ),
            "gt_boost_trigger_rate": frac(
                int(r["gt_boost_triggered"]) == 1 for r in rs
            ),
            "mean_wrong_excess": mean(r["wrong_excess"] for r in rs),
            "mean_gt_deficit": mean(r["gt_deficit"] for r in rs),
            "mean_delta_norm": mean(r["delta_norm"] for r in rs),
            "clip_rate": frac(int(r["clipped"]) == 1 for r in rs),
            "correct_samples_wrong_suppress_trigger_rate": frac(
                int(r["wrong_suppress_triggered"]) == 1 for r in corr
            ),
            "wrong_samples_wrong_suppress_trigger_rate": frac(
                int(r["wrong_suppress_triggered"]) == 1 for r in wrong
            ),
            "correct_samples_gt_boost_trigger_rate": frac(
                int(r["gt_boost_triggered"]) == 1 for r in corr
            ),
            "wrong_samples_gt_boost_trigger_rate": frac(
                int(r["gt_boost_triggered"]) == 1 for r in wrong
            ),
            "competitor_left": competitor_counts.get("left", 0),
            "competitor_right": competitor_counts.get("right", 0),
            "competitor_above": competitor_counts.get("above", 0),
            "competitor_below": competitor_counts.get("below", 0),
        })

    write_csv(out_dir / "layer_trace_summary.csv", summary)

    print("\n" + "=" * 145)
    print("PER-LAYER DYNAMIC EDIT TRACE")
    print("=" * 145)
    print(
        "mode                    layer  suppressTrig  boostTrig  "
        "wrongEx  gtDef  deltaNorm  suppressTrig(correct/wrong)"
    )
    for r in summary:
        if r["mode"].startswith("random_"):
            continue
        print(
            f"{r['mode']:<23s} L{int(r['layer']):02d}    "
            f"{r['wrong_suppress_trigger_rate']:.3f}       "
            f"{r['gt_boost_trigger_rate']:.3f}    "
            f"{r['mean_wrong_excess']:.3f}   "
            f"{r['mean_gt_deficit']:.3f}   "
            f"{r['mean_delta_norm']:.3f}      "
            f"{r['correct_samples_wrong_suppress_trigger_rate']:.3f}/"
            f"{r['wrong_samples_wrong_suppress_trigger_rate']:.3f}"
        )

    return summary


def summarize_relation(rows: Sequence[Mapping[str, Any]], out_dir: Path):
    modes = ("multi_suppress", "multi_both")
    buckets = defaultdict(list)
    for r in rows:
        buckets[str(r["gt"])].append(r)

    out = []
    for gt, rs in sorted(buckets.items()):
        for mode in modes:
            base_acc = frac(int(r["baseline_correct"]) == 1 for r in rs)
            new_acc = frac(int(r[f"{mode}_correct"]) == 1 for r in rs)
            out.append({
                "gt": gt,
                "mode": mode,
                "n": len(rs),
                "baseline_acc": base_acc,
                "intervention_acc": new_acc,
                "accuracy_gain": new_acc - base_acc,
                "W_to_C": sum(int(r[f"{mode}_W_to_C"]) for r in rs),
                "C_to_W": sum(int(r[f"{mode}_C_to_W"]) for r in rs),
                "net": (
                    sum(int(r[f"{mode}_W_to_C"]) for r in rs)
                    - sum(int(r[f"{mode}_C_to_W"]) for r in rs)
                ),
            })
    write_csv(out_dir / "summary_by_relation.csv", out)


# =============================================================================
# Main causal experiment
# =============================================================================

def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    direction_dir = Path(args.direction_dir)
    group_root = Path(args.group_root)
    out_dir = Path(args.output_dir)

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)

    (
        arr,
        groups,
        idx_by_sid,
        labels,
        residual,
        codebooks,
        split_rows,
        split_by_sid,
    ) = load_assets(direction_dir, group_root, layers)

    gt_targets, wrong_targets, target_audit = build_targets(
        groups=groups,
        idx_by_sid=idx_by_sid,
        labels=labels,
        residual=residual,
        codebooks=codebooks,
        layers=layers,
        control_group=args.control_group,
        target_stat=args.target_stat,
        min_controls=args.min_controls,
    )
    write_csv(out_dir / "control_targets.csv", target_audit)

    # Select ALL records in requested cached split.
    if args.split == "all":
        keep_sids = set(split_by_sid)
    else:
        keep_sids = {
            sid for sid, sp in split_by_sid.items() if sp == args.split
        }

    records, _audit = base.load_records(
        args.dataset, Path(args.data_root), None
    )
    records = [
        rec for rec in records
        if int(rec.sid) in keep_sids
        and norm_rel(rec.relation) in REL2ID
    ]

    if args.max_samples is not None and len(records) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[: int(args.max_samples)]

    if not records:
        raise RuntimeError("No evaluation records selected.")

    print(
        f"[data] split={args.split}  N={len(records)}  "
        f"layers={layers}"
    )

    # Load model.
    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    kw: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id} on {args.device}")
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, layer_path = causal.direction_base.resolve_decoder_layers(
        model
    )
    token_map = causal.relation_token_variants(processor.tokenizer)
    print(f"[decoder] {layer_path}")

    modes = (
        "multi_suppress",
        "multi_both",
        "random_multi_suppress",
        "random_multi_both",
    )
    internal_mode = {
        "multi_suppress": "suppress",
        "multi_both": "both",
        "random_multi_suppress": "random_suppress",
        "random_multi_both": "random_both",
    }

    final_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    final_path = out_dir / "per_sample_final.csv"
    trace_path = out_dir / "per_layer_trace.csv"

    for rec in tqdm(records, desc="dynamic multilayer all-sample"):
        img = None
        batch = None
        try:
            sid = int(rec.sid)
            gt = norm_rel(rec.relation)

            question = args.prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )

            img = Image.open(rec.image_path).convert("RGB")
            batch, _, subj_pos, ref_pos = causal.build_batch(
                processor, rec, question, img, device
            )

            baseline = causal.score_forward(
                model, batch, token_map, gt
            )

            # Matched no-image object differences are fixed references for all
            # dynamic edited REAL runs.
            noimg_diffs = geom.capture_noimg_diffs(
                model,
                processor,
                device,
                decoder_layers,
                layers,
                question,
                str(rec.subject),
                str(rec.reference),
            )

            row: Dict[str, Any] = {
                "sid": sid,
                "split": split_by_sid.get(sid, ""),
                "gt": gt,
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "baseline_pred": baseline["pred"],
                "baseline_correct": int(baseline["correct"]),
                "baseline_margin": baseline["margin"],
            }
            for rel in RELATIONS:
                row[f"baseline_logit_{rel}"] = baseline[f"logit_{rel}"]

            for mi, mode_name in enumerate(modes):
                result, traces = forward_dynamic(
                    model=model,
                    batch=batch,
                    token_map=token_map,
                    gt=gt,
                    decoder_layers=decoder_layers,
                    layers=layers,
                    sid=sid,
                    subj_pos=subj_pos,
                    ref_pos=ref_pos,
                    noimg_diffs=noimg_diffs,
                    codebooks=codebooks,
                    gt_targets=gt_targets,
                    wrong_targets=wrong_targets,
                    mode=internal_mode[mode_name],
                    seed=args.seed + sid * 1009 + mi * 99991,
                    max_edit_norm=args.max_edit_norm,
                )

                row[f"{mode_name}_pred"] = result["pred"]
                row[f"{mode_name}_correct"] = int(result["correct"])
                row[f"{mode_name}_margin"] = result["margin"]
                row[f"{mode_name}_margin_gain"] = (
                    result["margin"] - baseline["margin"]
                )
                row[f"{mode_name}_pred_changed"] = int(
                    result["pred"] != baseline["pred"]
                )
                row[f"{mode_name}_W_to_C"] = int(
                    (not baseline["correct"]) and result["correct"]
                )
                row[f"{mode_name}_C_to_W"] = int(
                    baseline["correct"] and (not result["correct"])
                )

                for rel in RELATIONS:
                    row[f"{mode_name}_logit_{rel}"] = result[
                        f"logit_{rel}"
                    ]

                for t in traces:
                    t = dict(t)
                    t["public_mode"] = mode_name
                    t["baseline_correct"] = int(baseline["correct"])
                    t["baseline_pred"] = baseline["pred"]
                    trace_rows.append(t)

            final_rows.append(row)

            if len(final_rows) % 20 == 0:
                write_csv(final_path, final_rows)
                write_csv(trace_path, trace_rows)

        except Exception as exc:
            tqdm.write(
                f"[ERROR sid={getattr(rec, 'sid', '?')}] "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            if img is not None:
                img.close()
            del batch
