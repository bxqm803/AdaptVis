#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-7B oracle intervention:
    GT+  vs  Wrong-  vs  GT+Wrong-

Purpose
-------
Distinguish two hypotheses for restricted-wrong + Direction-strong samples:

H1: insufficient correct spatial evidence
    -> boosting the GT-specific relation coordinate should repair errors.

H2: excessive competing/wrong spatial evidence
    -> suppressing the model's actually predicted wrong-relation coordinate
       should repair errors.

H3: both
    -> the combined intervention should be stronger than either one alone.

This is an ORACLE MECHANISM TEST:
- GT relation is known.
- The model's baseline restricted wrong prediction is known.
It is NOT intended as a deployable inference method.

Representation
--------------
The existing Direction probe uses:

    residual_l = (h_sub^img - h_ref^img)
                 - (h_sub^noimg - h_ref^noimg)

and a train-set center plus four relation prototypes.

For a wrong sample:
    g = prototype(GT)
    w = prototype(model's final wrong prediction)
    q = residual_l - train_center

We measure two raw relation coordinates:
    s_g = q · g
    s_w = q · w

From restricted-correct + Direction-strong controls with the same GT, we learn:
    target_g(GT, layer)       = median(q_control · g)
    target_w(GT, wrong, layer)= median(q_control · w)

Then:

GT+
    raise s_g to the normal correct-control median,
    while keeping s_w unchanged.

Wrong-
    lower s_w to the normal correct-control median,
    while keeping s_g unchanged.

Both
    do both simultaneously.

The edit is the minimum-L2 delta satisfying the requested dot-product changes:

    A = [g, w]
    delta = A (A^T A)^(-1) c

where c is:
    GT+      : [target_g - s_g, 0]
    Wrong-   : [0, target_w - s_w]
    Both     : [target_g - s_g, target_w - s_w]

Only deficits/excesses are corrected:
    GT is boosted only if s_g < target_g.
    Wrong is suppressed only if s_w > target_w.

The subject/reference pair mean is preserved:

    h_sub <- h_sub + delta/2
    h_ref <- h_ref - delta/2

so their difference changes by exactly delta.

Controls
--------
For each intervention we also apply a norm-matched random edit orthogonal to
span{g,w}. Therefore it has the same perturbation magnitude but does not
directly alter either GT or wrong prototype coordinate.

Required existing files
-----------------------
From previous runs:

<direction-dir>/vectors.npz
<direction-dir>/sample_split_and_generation.csv
<group-root>/restricted_direction_groups.csv

Also place these previous scripts in the repo root:
    analyze_layerwise_direction_failure_scan_v1.py
    analyze_text_stream_visual_causal_transfer_v1.py
    analyze_direction_geometry_and_causality_v1.py

Recommended run
---------------
CUDA_VISIBLE_DEVICES=0 python eval_gt_boost_vs_wrong_suppress_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --group-root output/qwen7b_direction_conditioned_failure_v1 \
  --layers 16,18 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --output-dir output/qwen7b_gt_vs_wrong_intervention_v1 \
  --overwrite

Smoke test
----------
CUDA_VISIBLE_DEVICES=0 python eval_gt_boost_vs_wrong_suppress_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --group-root output/qwen7b_direction_conditioned_failure_v1 \
  --layers 16,18 \
  --model qwen-7b \
  --device cuda:0 \
  --max-samples 8 \
  --output-dir output/qwen7b_gt_vs_wrong_intervention_smoke \
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
from collections import defaultdict
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
# CLI / I/O
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--group-root", required=True)
    p.add_argument("--layers", default="16,18")
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
        "--target-group",
        default="restricted_wrong_repr_strong",
        help="Samples on which oracle repair is tested.",
    )
    p.add_argument(
        "--control-group",
        default="restricted_correct_repr_strong",
        help="Controls used to define normal relation-coordinate medians.",
    )
    p.add_argument(
        "--target-stat",
        choices=["median", "mean"],
        default="median",
        help="Statistic of correct controls used as target strength.",
    )
    p.add_argument(
        "--min-controls",
        type=int,
        default=5,
        help="Minimum same-GT controls required for a target. "
             "Falls back to all correct controls of the same GT.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
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
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_layers(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("No layers selected.")
    return list(dict.fromkeys(vals))


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def frac(xs: Iterable[bool]) -> float:
    vals = [bool(x) for x in xs]
    return float(np.mean(vals)) if vals else float("nan")


def stat(vals: Sequence[float], which: str) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if which == "median":
        return float(np.median(x))
    return float(np.mean(x))


# =============================================================================
# Assets / target statistics
# =============================================================================

def load_assets(
    direction_dir: Path,
    group_root: Path,
    layers: Sequence[int],
):
    # Reuse exactly the same codebook construction as the previous geometry run.
    arr, groups, idx_by_sid, labels, residual, codebooks = geom.load_assets(
        direction_dir, group_root, layers
    )
    return arr, groups, idx_by_sid, labels, residual, codebooks


def normalize_rel(x: str) -> str:
    return dirscan.norm_relation(x)


def build_control_targets(
    *,
    groups: Sequence[Mapping[str, str]],
    idx_by_sid: Mapping[int, int],
    labels: np.ndarray,
    residual: np.ndarray,
    codebooks: Mapping[int, Mapping[str, np.ndarray]],
    layers: Sequence[int],
    control_group: str,
    target_stat: str,
    min_controls: int,
) -> Tuple[
    Dict[Tuple[int, str], float],
    Dict[Tuple[int, str, str], float],
    List[Dict[str, Any]],
]:
    """
    target_gt[(layer, gt)] = typical correct-control q dot proto_gt
    target_wrong[(layer, gt, foil)] = typical correct-control q dot proto_foil
    """
    controls = [
        r for r in groups
        if str(r["restricted_group"]) == control_group
        and int(r["sid"]) in idx_by_sid
    ]
    if not controls:
        raise RuntimeError(f"No controls found for {control_group}")

    gt_targets: Dict[Tuple[int, str], float] = {}
    wrong_targets: Dict[Tuple[int, str, str], float] = {}
    audit: List[Dict[str, Any]] = []

    for li in layers:
        cb = codebooks[li]
        center = cb["center"]
        protos = cb["protos"]

        for gt in RELATIONS:
            same_gt = [
                r for r in controls
                if normalize_rel(r.get("gt", labels[idx_by_sid[int(r["sid"])]])) == gt
            ]
            if len(same_gt) < min_controls:
                # Same fallback is intentionally conservative: we still require
                # same GT, because relation scales can differ.
                same_gt = [
                    r for r in controls
                    if labels[idx_by_sid[int(r["sid"])]] == gt
                ]

            if not same_gt:
                continue

            q_list = []
            for r in same_gt:
                idx = idx_by_sid[int(r["sid"])]
                q_list.append(residual[idx, li, :] - center)

            gt_proto = protos[REL2ID[gt]]
            gt_vals = [float(q @ gt_proto) for q in q_list]
            gt_targets[(li, gt)] = stat(gt_vals, target_stat)

            audit.append({
                "layer": li,
                "gt": gt,
                "foil": "GT",
                "n_controls": len(q_list),
                "target_coordinate": gt_targets[(li, gt)],
                "statistic": target_stat,
            })

            for foil in RELATIONS:
                if foil == gt:
                    continue
                foil_proto = protos[REL2ID[foil]]
                foil_vals = [float(q @ foil_proto) for q in q_list]
                wrong_targets[(li, gt, foil)] = stat(foil_vals, target_stat)
                audit.append({
                    "layer": li,
                    "gt": gt,
                    "foil": foil,
                    "n_controls": len(q_list),
                    "target_coordinate": wrong_targets[(li, gt, foil)],
                    "statistic": target_stat,
                })

    return gt_targets, wrong_targets, audit


# =============================================================================
# Linear algebra for independent GT / wrong coordinate edits
# =============================================================================

def dual_edit_matrix(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    A=[g,w] and B=A(A^T A)^(-1), so A^T B = I.

    BF16/FP16 linear algebra is not supported by torch.linalg.inv on CUDA,
    so solve the tiny 2x2 system in float32 and cast the result back.
    """
    orig_dtype = g.dtype
    A32 = torch.stack([g.float(), w.float()], dim=1)  # [D,2], fp32
    gram32 = A32.transpose(0, 1) @ A32
    eye32 = torch.eye(2, device=A32.device, dtype=torch.float32)
    inv32 = torch.linalg.inv(gram32 + 1e-6 * eye32)
    B32 = A32 @ inv32
    return B32.to(dtype=orig_dtype)  # [D,2]


def random_orthogonal_delta(
    *,
    true_delta: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """
    Same norm as true_delta, but orthogonal to span{g,w}.

    The orthogonalization solve is done in float32 because torch.linalg.solve
    does not support BF16/FP16 reliably here.
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
    A32 = torch.stack([g32, w32], dim=1)  # [D,2]
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

    out32 = rv32 * (target_norm / nr)
    return out32.to(dtype=true_delta.dtype)


# =============================================================================
# Intervention hook
# =============================================================================

class RelationCoordinateEdit:
    def __init__(
        self,
        *,
        module,
        layer: int,
        sid: int,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        noimg_diff_cpu: torch.Tensor,
        center_cpu: torch.Tensor,
        gt_proto_cpu: torch.Tensor,
        wrong_proto_cpu: torch.Tensor,
        target_gt: float,
        target_wrong: float,
        mode: str,
        seed: int,
    ):
        self.layer = int(layer)
        self.sid = int(sid)
        self.subj = [int(x) for x in subject_positions]
        self.ref = [int(x) for x in reference_positions]
        self.noimg_diff_cpu = noimg_diff_cpu
        self.center_cpu = center_cpu
        self.gt_proto_cpu = gt_proto_cpu
        self.wrong_proto_cpu = wrong_proto_cpu
        self.target_gt = float(target_gt)
        self.target_wrong = float(target_wrong)
        self.mode = str(mode)
        self.seed = int(seed)

        self.applied = 0
        self.pre_gt = float("nan")
        self.pre_wrong = float("nan")
        self.post_gt = float("nan")
        self.post_wrong = float("nan")
        self.gt_deficit = float("nan")
        self.wrong_excess = float("nan")
        self.delta_norm = float("nan")

        self.handle = module.register_forward_hook(self._hook)

    def _base_true_delta(
        self,
        q: torch.Tensor,
        g: torch.Tensor,
        w: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, float, float]:
        pre_gt = g @ q
        pre_wrong = w @ q

        gt_deficit = torch.clamp(
            torch.as_tensor(self.target_gt, device=q.device, dtype=q.dtype) - pre_gt,
            min=0.0,
        )
        wrong_excess = torch.clamp(
            pre_wrong - torch.as_tensor(
                self.target_wrong, device=q.device, dtype=q.dtype
            ),
            min=0.0,
        )

        if mode == "gt_plus":
            c = torch.stack([gt_deficit, torch.zeros_like(gt_deficit)])
        elif mode == "wrong_minus":
            c = torch.stack([torch.zeros_like(wrong_excess), -wrong_excess])
        elif mode == "both":
            c = torch.stack([gt_deficit, -wrong_excess])
        else:
            raise ValueError(mode)

        B = dual_edit_matrix(g, w)
        delta = B @ c
        return delta, float(gt_deficit.detach().float().cpu()), float(
            wrong_excess.detach().float().cpu()
        )

    def _hook(self, _module, _args, output):
        if self.applied:
            return output

        x = causal.first_tensor(output)
        hs = geom.pool_hidden(x, self.subj)
        hr = geom.pool_hidden(x, self.ref)

        noimg = self.noimg_diff_cpu.to(x.device, dtype=x.dtype)
        center = self.center_cpu.to(x.device, dtype=x.dtype)
        g = self.gt_proto_cpu.to(x.device, dtype=x.dtype)
        w = self.wrong_proto_cpu.to(x.device, dtype=x.dtype)

        # q is exactly the centered residual Direction vector under this run.
        q = (hs - hr) - noimg - center

        self.pre_gt = float((g @ q).detach().float().cpu())
        self.pre_wrong = float((w @ q).detach().float().cpu())

        is_random = self.mode.startswith("random_")
        true_mode = self.mode.replace("random_", "") if is_random else self.mode

        true_delta, gt_def, wrong_exc = self._base_true_delta(
            q, g, w, true_mode
        )
        self.gt_deficit = gt_def
        self.wrong_excess = wrong_exc

        if is_random:
            delta = random_orthogonal_delta(
                true_delta=true_delta,
                g=g,
                w=w,
                seed=self.seed,
            )
        else:
            delta = true_delta

        self.delta_norm = float(delta.detach().float().norm().cpu())

        q_after = q + delta
        self.post_gt = float((g @ q_after).detach().float().cpu())
        self.post_wrong = float((w @ q_after).detach().float().cpu())

        # Preserve pair mean; change only subject-reference difference.
        y = x.clone()
        half = delta / 2.0
        y[0, self.subj, :] = y[0, self.subj, :] + half
        y[0, self.ref, :] = y[0, self.ref, :] - half

        self.applied += 1
        return causal.replace_first_tensor(output, y)

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


# =============================================================================
# Offline deficit / excess diagnostic
# =============================================================================

def offline_deficit_excess(
    *,
    groups,
    idx_by_sid,
    labels,
    residual,
    codebooks,
    gt_targets,
    wrong_targets,
    layers,
    target_group,
    out_dir,
):
    rows = []

    for r in groups:
        if str(r["restricted_group"]) != target_group:
            continue
        sid = int(r["sid"])
        if sid not in idx_by_sid:
            continue

        gt = normalize_rel(r.get("gt", labels[idx_by_sid[sid]]))
        wrong = normalize_rel(r.get("real_restricted_pred", ""))
        if gt not in REL2ID or wrong not in REL2ID or wrong == gt:
            continue

        idx = idx_by_sid[sid]

        for li in layers:
            if (li, gt) not in gt_targets or (li, gt, wrong) not in wrong_targets:
                continue

            cb = codebooks[li]
            q = residual[idx, li, :] - cb["center"]
            g = cb["protos"][REL2ID[gt]]
            w = cb["protos"][REL2ID[wrong]]

            sg = float(q @ g)
            sw = float(q @ w)
            tg = float(gt_targets[(li, gt)])
            tw = float(wrong_targets[(li, gt, wrong)])

            rows.append({
                "sid": sid,
                "layer": li,
                "gt": gt,
                "wrong_pred": wrong,
                "gt_coordinate": sg,
                "gt_target": tg,
                "gt_deficit": max(0.0, tg - sg),
                "has_gt_deficit": int(sg < tg),
                "wrong_coordinate": sw,
                "wrong_target": tw,
                "wrong_excess": max(0.0, sw - tw),
                "has_wrong_excess": int(sw > tw),
                "has_both": int((sg < tg) and (sw > tw)),
            })

    write_csv(out_dir / "offline_deficit_excess_per_sample.csv", rows)

    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r["layer"])].append(r)

    summary = []
    for li, rs in sorted(buckets.items()):
        summary.append({
            "layer": li,
            "n": len(rs),
            "gt_deficit_fraction": frac(bool(int(r["has_gt_deficit"])) for r in rs),
            "mean_gt_deficit": mean(r["gt_deficit"] for r in rs),
            "wrong_excess_fraction": frac(bool(int(r["has_wrong_excess"])) for r in rs),
            "mean_wrong_excess": mean(r["wrong_excess"] for r in rs),
            "both_fraction": frac(bool(int(r["has_both"])) for r in rs),
        })

    write_csv(out_dir / "offline_deficit_excess_summary.csv", summary)

    print("\n" + "=" * 110)
    print("OFFLINE: INSUFFICIENT GT vs EXCESS WRONG COORDINATE")
    print("=" * 110)
    print("layer   N   GT-def%  meanGTdef   Wrong-excess%  meanWrongExcess  both%")
    for r in summary:
        print(
            f"L{int(r['layer']):02d}   {int(r['n']):3d}   "
            f"{r['gt_deficit_fraction']:.3f}    "
            f"{r['mean_gt_deficit']:.4f}       "
            f"{r['wrong_excess_fraction']:.3f}          "
            f"{r['mean_wrong_excess']:.4f}          "
            f"{r['both_fraction']:.3f}"
        )

    return rows, summary


# =============================================================================
# Causal run
# =============================================================================

def run_causal(
    *,
    args,
    groups,
    codebooks,
    gt_targets,
    wrong_targets,
    layers,
    out_dir,
):
    target_rows = [
        r for r in groups
        if str(r["restricted_group"]) == args.target_group
    ]

    rng = random.Random(args.seed)
    if args.max_samples is not None and len(target_rows) > args.max_samples:
        rng.shuffle(target_rows)
        target_rows = target_rows[: int(args.max_samples)]

    keep_sids = {int(r["sid"]) for r in target_rows}
    group_by_sid = {int(r["sid"]): r for r in target_rows}

    records, _ = base.load_records(args.dataset, Path(args.data_root), None)
    records = [
        rec for rec in records
        if int(rec.sid) in keep_sids
        and normalize_rel(rec.relation) in REL2ID
    ]
    if not records:
        raise RuntimeError("No selected dataset records found.")

    # Model.
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

    print(f"\n[model] loading {spec.repo_id} on {args.device}")
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, layer_path = causal.direction_base.resolve_decoder_layers(model)
    token_map = causal.relation_token_variants(processor.tokenizer)
    print(f"[decoder] {layer_path}; layers={layers}; N={len(records)}")

    modes = (
        "gt_plus",
        "wrong_minus",
        "both",
        "random_gt_plus",
        "random_wrong_minus",
        "random_both",
    )

    rows: List[Dict[str, Any]] = []
    per_path = out_dir / "causal_per_sample.csv"

    for rec in tqdm(records, desc="GT+ vs Wrong-"):
        img = None
        batch = None
        try:
            sid = int(rec.sid)
            meta = group_by_sid[sid]
            gt = normalize_rel(rec.relation)
            wrong = normalize_rel(meta.get("real_restricted_pred", ""))

            if wrong not in REL2ID or wrong == gt:
                tqdm.write(
                    f"[skip sid={sid}] invalid restricted wrong prediction: {wrong!r}"
                )
                continue

            question = args.prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            img = Image.open(rec.image_path).convert("RGB")
            batch, _, subj_pos, ref_pos = causal.build_batch(
                processor, rec, question, img, device
            )

            baseline = causal.score_forward(model, batch, token_map, gt)

            # Safety: this experiment is intended only for baseline restricted-wrong.
            if baseline["correct"]:
                tqdm.write(
                    f"[warn sid={sid}] current baseline is correct although saved group is wrong; "
                    "processor/model version may differ. Keeping row but mark mismatch."
                )

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

            for li in layers:
                if (li, gt) not in gt_targets:
                    continue
                if (li, gt, wrong) not in wrong_targets:
                    continue

                cb = codebooks[li]
                gt_proto = torch.from_numpy(cb["protos"][REL2ID[gt]])
                wrong_proto = torch.from_numpy(cb["protos"][REL2ID[wrong]])
                center = torch.from_numpy(cb["center"])
                target_gt = gt_targets[(li, gt)]
                target_wrong = wrong_targets[(li, gt, wrong)]

                mode_results = {}
                mode_meta = {}

                for mi, mode in enumerate(modes):
                    patch = RelationCoordinateEdit(
                        module=decoder_layers[li],
                        layer=li,
                        sid=sid,
                        subject_positions=subj_pos,
                        reference_positions=ref_pos,
                        noimg_diff_cpu=noimg_diffs[li],
                        center_cpu=center,
                        gt_proto_cpu=gt_proto,
                        wrong_proto_cpu=wrong_proto,
                        target_gt=target_gt,
                        target_wrong=target_wrong,
                        mode=mode,
                        seed=args.seed + sid * 1009 + li * 10007 + mi * 9176,
                    )
                    try:
                        with patch:
                            result = causal.score_forward(
                                model, batch, token_map, gt
                            )
                        if patch.applied != 1:
                            raise RuntimeError(
                                f"sid={sid} L{li} {mode}: hook applied "
                                f"{patch.applied} times"
                            )
                    finally:
                        patch.close()

                    mode_results[mode] = result
                    mode_meta[mode] = {
                        "pre_gt": patch.pre_gt,
                        "pre_wrong": patch.pre_wrong,
                        "post_gt": patch.post_gt,
                        "post_wrong": patch.post_wrong,
                        "gt_deficit": patch.gt_deficit,
                        "wrong_excess": patch.wrong_excess,
                        "delta_norm": patch.delta_norm,
                    }

                row: Dict[str, Any] = {
                    "sid": sid,
                    "layer": li,
                    "gt": gt,
                    "baseline_wrong_pred_saved": wrong,
                    "baseline_pred": baseline["pred"],
                    "baseline_correct": int(baseline["correct"]),
                    "baseline_margin": baseline["margin"],
                    "target_gt_coordinate": target_gt,
                    "target_wrong_coordinate": target_wrong,
                    "pre_gt_coordinate": mode_meta["both"]["pre_gt"],
                    "pre_wrong_coordinate": mode_meta["both"]["pre_wrong"],
                    "gt_deficit": mode_meta["both"]["gt_deficit"],
                    "wrong_excess": mode_meta["both"]["wrong_excess"],
                    "has_gt_deficit": int(mode_meta["both"]["gt_deficit"] > 1e-8),
                    "has_wrong_excess": int(
                        mode_meta["both"]["wrong_excess"] > 1e-8
                    ),
                }

                for mode in modes:
                    res = mode_results[mode]
                    mm = mode_meta[mode]
                    row[f"{mode}_pred"] = res["pred"]
                    row[f"{mode}_correct"] = int(res["correct"])
                    row[f"{mode}_margin"] = res["margin"]
                    row[f"{mode}_margin_gain"] = (
                        res["margin"] - baseline["margin"]
                    )
                    row[f"{mode}_pred_changed"] = int(
                        res["pred"] != baseline["pred"]
                    )
                    row[f"{mode}_W_to_C"] = int(
                        (not baseline["correct"]) and res["correct"]
                    )
                    row[f"{mode}_delta_norm"] = mm["delta_norm"]
                    row[f"{mode}_post_gt_coordinate"] = mm["post_gt"]
                    row[f"{mode}_post_wrong_coordinate"] = mm["post_wrong"]

                rows.append(row)

            if len(rows) % 20 == 0:
                write_csv(per_path, rows)

        except Exception as exc:
            tqdm.write(
                f"[ERROR sid={getattr(rec, 'sid', '?')}] "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            if img is not None:
                img.close()
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(per_path, rows)
    summarize_causal(rows, out_dir)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


# =============================================================================
# Summary
# =============================================================================

def summarize_causal(rows: Sequence[Mapping[str, Any]], out_dir: Path):
    modes = (
        "gt_plus",
        "wrong_minus",
        "both",
        "random_gt_plus",
        "random_wrong_minus",
        "random_both",
    )

    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r["layer"])].append(r)

    summary: List[Dict[str, Any]] = []

    for li, rs in sorted(buckets.items()):
        base = {
            "layer": li,
            "n": len(rs),
            "baseline_acc": frac(bool(int(r["baseline_correct"])) for r in rs),
            "baseline_margin": mean(r["baseline_margin"] for r in rs),
            "gt_deficit_fraction": frac(
                bool(int(r["has_gt_deficit"])) for r in rs
            ),
            "mean_gt_deficit": mean(r["gt_deficit"] for r in rs),
            "wrong_excess_fraction": frac(
                bool(int(r["has_wrong_excess"])) for r in rs
            ),
            "mean_wrong_excess": mean(r["wrong_excess"] for r in rs),
        }

        for mode in modes:
            base[f"{mode}_acc"] = frac(
                bool(int(r[f"{mode}_correct"])) for r in rs
            )
            base[f"{mode}_W_to_C"] = sum(
                int(r[f"{mode}_W_to_C"]) for r in rs
            )
            base[f"{mode}_margin_gain"] = mean(
                r[f"{mode}_margin_gain"] for r in rs
            )
            base[f"{mode}_pred_change"] = frac(
                bool(int(r[f"{mode}_pred_changed"])) for r in rs
            )
            base[f"{mode}_mean_delta_norm"] = mean(
                r[f"{mode}_delta_norm"] for r in rs
            )

        base["gt_plus_specific_margin_gain"] = (
            base["gt_plus_margin_gain"] - base["random_gt_plus_margin_gain"]
        )
        base["wrong_minus_specific_margin_gain"] = (
            base["wrong_minus_margin_gain"]
            - base["random_wrong_minus_margin_gain"]
        )
        base["both_specific_margin_gain"] = (
            base["both_margin_gain"] - base["random_both_margin_gain"]
        )

        summary.append(base)

    write_csv(out_dir / "causal_summary.csv", summary)

    print("\n" + "=" * 150)
    print("ORACLE REPAIR: GT+ vs WRONG- vs BOTH")
    print("=" * 150)
    print(
        "layer N  GTdef% WrongEx% | "
        "GT+ acc W2C gain spec | "
        "Wrong- acc W2C gain spec | "
        "Both acc W2C gain spec"
    )

    for r in summary:
        print(
            f"L{int(r['layer']):02d} {int(r['n']):2d}  "
            f"{r['gt_deficit_fraction']:.2f}   "
            f"{r['wrong_excess_fraction']:.2f}   | "
            f"{r['gt_plus_acc']:.3f} "
            f"{int(r['gt_plus_W_to_C']):2d} "
            f"{r['gt_plus_margin_gain']:+.3f} "
            f"{r['gt_plus_specific_margin_gain']:+.3f} | "
            f"{r['wrong_minus_acc']:.3f} "
            f"{int(r['wrong_minus_W_to_C']):2d} "
            f"{r['wrong_minus_margin_gain']:+.3f} "
            f"{r['wrong_minus_specific_margin_gain']:+.3f} | "
            f"{r['both_acc']:.3f} "
            f"{int(r['both_W_to_C']):2d} "
            f"{r['both_margin_gain']:+.3f} "
            f"{r['both_specific_margin_gain']:+.3f}"
        )

    print("\nInterpretation:")
    print("  GT+ >> Wrong-  : insufficient correct evidence is more important.")
    print("  Wrong- >> GT+  : excessive competing/wrong evidence is more important.")
    print("  Both >> either : both mechanisms contribute / interaction matters.")
    print("  All ~ random   : prototype-coordinate edit is not the causal mechanism.")


# =============================================================================
# Main
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

    arr, groups, idx_by_sid, labels, residual, codebooks = load_assets(
        direction_dir, group_root, layers
    )

    gt_targets, wrong_targets, target_audit = build_control_targets(
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

    offline_deficit_excess(
        groups=groups,
        idx_by_sid=idx_by_sid,
        labels=labels,
        residual=residual,
        codebooks=codebooks,
        gt_targets=gt_targets,
        wrong_targets=wrong_targets,
        layers=layers,
        target_group=args.target_group,
        out_dir=out_dir,
    )

    run_causal(
        args=args,
        groups=groups,
        codebooks=codebooks,
        gt_targets=gt_targets,
        wrong_targets=wrong_targets,
        layers=layers,
        out_dir=out_dir,
    )

    meta = {
        "layers": layers,
        "target_group": args.target_group,
        "control_group": args.control_group,
        "target_stat": args.target_stat,
        "edit": {
            "gt_plus": "raise GT prototype coordinate to correct-control target while holding wrong coordinate fixed",
            "wrong_minus": "lower final-wrong prototype coordinate to correct-control target while holding GT coordinate fixed",
            "both": "apply both constraints simultaneously",
            "random_controls": "norm-matched random edit orthogonal to GT/wrong prototype span",
        },
        "oracle": True,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved to:", out_dir)
    print("  control_targets.csv")
    print("  offline_deficit_excess_per_sample.csv")
    print("  offline_deficit_excess_summary.csv")
    print("  causal_per_sample.csv")
    print("  causal_summary.csv")


if __name__ == "__main__":
    main()
