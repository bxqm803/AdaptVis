#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic multi-layer Direction repair with object-pair + last-token states.
No gradients.

Requires in the same repo:
  - extract_two_object_relation_states.py
  - analyze_layerwise_direction_failure_scan_v1.py
  - eval_single_layer_direction_state_rescue_v2.py

Also requires the existing all-layer object TRAIN cache:
  train_all_layer_direction_states.npz
from eval_per_sample_direction_repair_all_layers_v2.py.

Variants:
  object_only
  last_only
  joint
  last_random

For each selected layer on the CURRENT modified trajectory:

  q_obj  = (h_sub-h_ref)^img - (h_sub-h_ref)^noimg
  q_last = h_last^img - h_last^noimg

For each channel, learn layer-specific four-way Direction codebooks from TRAIN.
Generation-correct TRAIN samples define q10/q25/median GT-vs-maxNonGT margins.

If current margin is below target, apply the exact minimum one-axis Direction
correction:
  object: subject += delta/2, reference -= delta/2
  last:   last_token += delta

Earlier edits affect later-layer diagnosis, so repair is truly dynamic.

Recommended:
CUDA_VISIBLE_DEVICES=0 python eval_dynamic_multilayer_object_last_direction_repair_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --object-train-cache output/qwen7b_per_sample_direction_repair_all_layers_v2/train_all_layer_direction_states.npz \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers all \
  --sample-group wrong \
  --target q10 \
  --modes object_only,last_only,joint,last_random \
  --output-dir output/qwen7b_dynamic_object_last_direction_repair_v1 \
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as direction
import eval_single_layer_direction_state_rescue_v2 as single


RELATIONS = ("left", "right", "above", "below")
EPS = 1e-10


# =============================================================================
# CLI / I/O
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--object-train-cache", required=True)
    p.add_argument("--last-train-cache", default="")
    p.add_argument("--rebuild-last-train-cache", action="store_true")

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
    p.add_argument("--layers", default="all")
    p.add_argument(
        "--sample-group",
        default="wrong",
        choices=["wrong", "correct", "all"],
    )
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--target",
        default="q10",
        choices=["zero", "q10", "q25", "median"],
    )
    p.add_argument(
        "--modes",
        default="object_only,last_only,joint,last_random",
    )
    p.add_argument(
        "--project-spatial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--object-max-edit-norm", type=float, default=0.0)
    p.add_argument("--last-max-edit-norm", type=float, default=0.0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs: Iterable[float]) -> float:
    vals = [
        float(x)
        for x in xs
        if math.isfinite(float(x))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_words(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_layers(text: str, n_layers: int):
    if str(text).strip().lower() == "all":
        return list(range(n_layers))
    vals = []
    for piece in parse_words(text):
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            vals.extend(range(a, b + step, step))
        else:
            vals.append(int(piece))
    vals = sorted(set(vals))
    bad = [x for x in vals if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Invalid layers: {bad}")
    return vals


# =============================================================================
# Batch / capture helpers
# =============================================================================

def build_batch(processor, rec, question, image, device):
    rendered = direction.build_chat_prompt(
        processor, question, image is not None
    )
    batch = direction.process_inputs(
        processor, rendered, image, device
    )
    ids = [
        int(x)
        for x in batch["input_ids"][0].detach().cpu().tolist()
    ]
    sp = direction.locate_phrase_positions(
        processor.tokenizer, ids, str(rec.subject)
    )
    rp = direction.locate_phrase_positions(
        processor.tokenizer, ids, str(rec.reference)
    )
    if "attention_mask" in batch:
        nz = torch.nonzero(
            batch["attention_mask"][0],
            as_tuple=False,
        ).flatten()
        last_pos = (
            int(nz[-1].item())
            if len(nz)
            else int(batch["input_ids"].shape[1] - 1)
        )
    else:
        last_pos = int(batch["input_ids"].shape[1] - 1)
    return batch, sp, rp, last_pos


class PreAttnCapture:
    def __init__(
        self,
        layers,
        selected_layers,
        subj_pos,
        ref_pos,
        last_pos,
        capture_obj=True,
        capture_last=True,
    ):
        self.obj = {}
        self.last = {}
        self.sp = list(map(int, subj_pos))
        self.rp = list(map(int, ref_pos))
        self.last_pos = int(last_pos)
        self.capture_obj = capture_obj
        self.capture_last = capture_last
        self.handles = []

        for li in selected_layers:
            self.handles.append(
                layers[li].register_forward_pre_hook(
                    self._make_hook(li)
                )
            )

    def _make_hook(self, li):
        def hook(_module, args):
            if not args:
                return None
            x = single.first_tensor(args)
            if x.ndim != 3:
                return None
            seq = x[0]
            if self.last_pos >= int(seq.shape[0]):
                return None

            if self.capture_obj:
                self.obj[li] = (
                    single.pair_diff(seq, self.sp, self.rp)
                    .detach().float().cpu().numpy()
                    .astype(np.float32)
                )

            if self.capture_last:
                self.last[li] = (
                    seq[self.last_pos]
                    .detach().float().cpu().numpy()
                    .astype(np.float32)
                )
            return None
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def capture_states(
    *,
    model,
    processor,
    layers,
    selected_layers,
    rec,
    image,
    device,
    prompt_template,
    capture_obj=True,
    capture_last=True,
):
    q = prompt_template.format(
        subject=rec.subject,
        reference=rec.reference,
    )
    batch, sp, rp, last_pos = build_batch(
        processor, rec, q, image, device
    )

    with PreAttnCapture(
        layers,
        selected_layers,
        sp,
        rp,
        last_pos,
        capture_obj=capture_obj,
        capture_last=capture_last,
    ) as cap:
        with torch.inference_mode():
            _ = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

    if capture_obj:
        missing = [li for li in selected_layers if li not in cap.obj]
        if missing:
            raise RuntimeError(f"Missing object states: {missing}")
    if capture_last:
        missing = [li for li in selected_layers if li not in cap.last]
        if missing:
            raise RuntimeError(f"Missing last states: {missing}")

    del batch
    return {
        "obj": dict(cap.obj),
        "last": dict(cap.last),
        "sp": sp,
        "rp": rp,
        "last_pos": last_pos,
    }


# =============================================================================
# Last-token TRAIN cache
# =============================================================================

def read_object_cache_sids(path: Path):
    with np.load(path, allow_pickle=True) as z:
        return [int(x) for x in z["sid"].tolist()]


def validate_object_cache(path: Path, selected_layers):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        avail = set(int(x) for x in z["selected_layers"].tolist())
        missing = [
            li for li in selected_layers
            if li not in avail or f"L{li}_pre_attn" not in z.files
        ]
    if missing:
        raise RuntimeError(f"Object cache missing layers {missing}")


def validate_last_cache(path: Path, selected_layers):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        avail = set(int(x) for x in z["selected_layers"].tolist())
        missing = [
            li for li in selected_layers
            if li not in avail or f"L{li}_pre_attn_last" not in z.files
        ]
    if missing:
        raise RuntimeError(f"Last cache missing layers {missing}")


def collect_last_cache(
    *,
    cache_path,
    object_cache_path,
    model,
    processor,
    layers,
    selected_layers,
    records,
    metadata,
    device,
    prompt_template,
):
    sids = read_object_cache_sids(object_cache_path)

    vals = {li: [] for li in selected_layers}
    kept_sid, kept_rel, kept_group = [], [], []
    errors = []

    for sid in tqdm(sids, desc="collect TRAIN last-token Direction"):
        rec = records.get(sid)
        if rec is None:
            continue

        image = None
        try:
            image = Image.open(rec.image_path).convert("RGB")

            real = capture_states(
                model=model,
                processor=processor,
                layers=layers,
                selected_layers=selected_layers,
                rec=rec,
                image=image,
                device=device,
                prompt_template=prompt_template,
                capture_obj=False,
                capture_last=True,
            )
            noimg = capture_states(
                model=model,
                processor=processor,
                layers=layers,
                selected_layers=selected_layers,
                rec=rec,
                image=None,
                device=device,
                prompt_template=prompt_template,
                capture_obj=False,
                capture_last=True,
            )

            for li in selected_layers:
                vals[li].append(
                    (
                        real["last"][li]
                        - noimg["last"][li]
                    ).astype(np.float32)
                )

            kept_sid.append(sid)
            kept_rel.append(metadata["gt"][sid])
            kept_group.append(
                metadata["generation"].get(sid, {}).get(
                    "generation_group", ""
                )
            )

        except Exception as e:
            errors.append(
                {
                    "sid": sid,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            tqdm.write(f"[last-cache sid={sid}] {type(e).__name__}: {e}")
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "sid": np.asarray(kept_sid, dtype=np.int64),
        "relation": np.asarray(kept_rel, dtype=object),
        "generation_group": np.asarray(kept_group, dtype=object),
        "selected_layers": np.asarray(selected_layers, dtype=np.int64),
    }
    for li in selected_layers:
        payload[f"L{li}_pre_attn_last"] = np.stack(
            vals[li], axis=0
        ).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    write_csv(cache_path.with_suffix(".errors.csv"), errors)
    print(f"[last-cache] saved {cache_path}, N={len(kept_sid)}")


# =============================================================================
# Codebooks / normal trajectories
# =============================================================================

def fit_channel(cache_path: Path, selected_layers, channel: str):
    codebooks = {}
    targets = {}
    rows = []

    with np.load(cache_path, allow_pickle=True) as z:
        labels = np.asarray(
            [single.norm_relation(x) for x in z["relation"]],
            dtype=object,
        )
        groups = np.asarray(
            [str(x).strip().lower() for x in z["generation_group"]],
            dtype=object,
        )
        correct = groups == "correct"

        for li in selected_layers:
            key = (
                f"L{li}_pre_attn"
                if channel == "object"
                else f"L{li}_pre_attn_last"
            )
            X = np.asarray(z[key], dtype=np.float32)
            cb = single.fit_codebook(X, labels)
            codebooks[li] = cb

            for gt in RELATIONS:
                mask = correct & (labels == gt)
                margins = []
                for i in np.where(mask)[0]:
                    _, _, m = single.scores_and_margin(
                        X[i], cb, gt
                    )
                    margins.append(m)
                if not margins:
                    raise RuntimeError(
                        f"No correct TRAIN controls: {channel} L{li} {gt}"
                    )
                a = np.asarray(margins, dtype=np.float64)
                t = {
                    "zero": 0.0,
                    "q10": float(np.quantile(a, 0.10)),
                    "q25": float(np.quantile(a, 0.25)),
                    "median": float(np.median(a)),
                    "n": int(len(a)),
                }
                targets[(li, gt)] = t
                rows.append(
                    {
                        "channel": channel,
                        "layer": li,
                        "gt": gt,
                        "basis_rank": int(cb.get("basis_rank", cb["basis"].shape[1])),
                        "n_correct_train": len(a),
                        "q10_margin": t["q10"],
                        "q25_margin": t["q25"],
                        "median_margin": t["median"],
                    }
                )
    return codebooks, targets, rows


def exact_delta(
    cb,
    gt,
    foil,
    current_margin,
    target_margin,
    project_spatial,
    max_edit_norm,
):
    # Reuse the corrected exact-margin implementation from v2.
    (
        delta,
        achieved,
        raw_norm,
        denom,
        used_spatial,
        used_raw,
    ) = single.make_margin_correction(
        cb=cb,
        gt=gt,
        foil=foil,
        current_margin=current_margin,
        target_margin=target_margin,
        project_to_spatial=project_spatial,
        max_edit_norm=max_edit_norm,
    )
    return {
        "delta": delta,
        "achieved": achieved,
        "raw_norm": raw_norm,
        "edit_norm": float(np.linalg.norm(delta)),
        "denom": denom,
        "used_spatial": int(bool(used_spatial)),
        "used_raw": int(bool(used_raw)),
    }


def random_orthogonal(norm, basis, dim, seed):
    if norm <= EPS:
        return np.zeros(dim, dtype=np.float32)
    rng = np.random.default_rng(seed)
    B = np.asarray(basis, dtype=np.float64)
    for _ in range(100):
        v = rng.standard_normal(dim)
        if B.ndim == 2 and B.shape[1] > 0:
            v = v - B @ (B.T @ v)
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            return (v / n * norm).astype(np.float32)
    raise RuntimeError("Could not sample orthogonal random control")


# =============================================================================
# Dynamic hook
# =============================================================================

class DynamicRepair:
    def __init__(
        self,
        *,
        layers,
        selected_layers,
        sp,
        rp,
        last_pos,
        noimg_obj,
        noimg_last,
        obj_codebooks,
        obj_targets,
        last_codebooks,
        last_targets,
        gt,
        target_name,
        mode,
        project_spatial,
        object_max_edit_norm,
        last_max_edit_norm,
        sid,
        seed,
    ):
        self.sp = list(map(int, sp))
        self.rp = list(map(int, rp))
        self.last_pos = int(last_pos)
        self.noimg_obj = noimg_obj
        self.noimg_last = noimg_last
        self.obj_cb = obj_codebooks
        self.obj_t = obj_targets
        self.last_cb = last_codebooks
        self.last_t = last_targets
        self.gt = gt
        self.target_name = target_name
        self.mode = mode
        self.project_spatial = project_spatial
        self.obj_cap = object_max_edit_norm
        self.last_cap = last_max_edit_norm
        self.sid = int(sid)
        self.seed = int(seed)
        self.done = set()
        self.logs = []
        self.handles = []

        for li in selected_layers:
            self.handles.append(
                layers[li].register_forward_pre_hook(
                    self._make_hook(li)
                )
            )

    def diagnose(self, q, cb, target, li):
        scores, foil, margin = single.scores_and_margin(
            q, cb, self.gt
        )
        tgt = float(target[(li, self.gt)][self.target_name])
        return scores, foil, margin, tgt, int(margin < tgt)

    def _make_hook(self, li):
        def hook(_module, args):
            if li in self.done or not args:
                return None

            vals = list(args)
            idx = None
            x = None
            for i, item in enumerate(vals):
                if torch.is_tensor(item):
                    idx, x = i, item
                    break

            if x is None or x.ndim != 3:
                return None
            if self.last_pos >= int(x.shape[1]):
                return None  # decode step, not prompt prefill

            seq = x[0]

            obj_real = (
                single.pair_diff(seq, self.sp, self.rp)
                .detach().float().cpu().numpy()
                .astype(np.float32)
            )
            last_real = (
                seq[self.last_pos]
                .detach().float().cpu().numpy()
                .astype(np.float32)
            )

            q_obj = obj_real - self.noimg_obj[li]
            q_last = last_real - self.noimg_last[li]

            oscore, ofoil, om, ot, oab = self.diagnose(
                q_obj, self.obj_cb[li], self.obj_t, li
            )
            lscore, lfoil, lm, lt, lab = self.diagnose(
                q_last, self.last_cb[li], self.last_t, li
            )

            od = exact_delta(
                self.obj_cb[li],
                self.gt,
                ofoil,
                om,
                ot,
                self.project_spatial,
                self.obj_cap,
            )
            ld = exact_delta(
                self.last_cb[li],
                self.gt,
                lfoil,
                lm,
                lt,
                self.project_spatial,
                self.last_cap,
            )

            obj_delta = np.zeros_like(od["delta"])
            last_delta = np.zeros_like(ld["delta"])

            if self.mode in ("object_only", "joint") and oab:
                obj_delta = od["delta"]

            if self.mode in ("last_only", "joint") and lab:
                last_delta = ld["delta"]

            if self.mode == "last_random" and lab:
                last_delta = random_orthogonal(
                    ld["edit_norm"],
                    self.last_cb[li]["basis"],
                    len(ld["delta"]),
                    self.seed + self.sid * 100003 + li * 1009,
                )

            onorm = float(np.linalg.norm(obj_delta))
            lnorm = float(np.linalg.norm(last_delta))

            y = x.clone()

            if onorm > EPS:
                half = 0.5 * torch.from_numpy(obj_delta).to(
                    device=y.device, dtype=y.dtype
                )
                y[0, self.sp, :] = y[0, self.sp, :] + half
                y[0, self.rp, :] = y[0, self.rp, :] - half

            if lnorm > EPS:
                dl = torch.from_numpy(last_delta).to(
                    device=y.device, dtype=y.dtype
                )
                y[0, self.last_pos, :] = y[0, self.last_pos, :] + dl

            self.logs.append(
                {
                    "sid": self.sid,
                    "mode": self.mode,
                    "layer": li,
                    "gt": self.gt,

                    "object_foil": ofoil,
                    "object_margin": om,
                    "object_target": ot,
                    "object_abnormal": oab,
                    "object_triggered": int(onorm > EPS),
                    "object_edit_norm": onorm,

                    "last_foil": lfoil,
                    "last_margin": lm,
                    "last_target": lt,
                    "last_abnormal": lab,
                    "last_triggered": int(lnorm > EPS),
                    "last_edit_norm": lnorm,

                    "obj_score_left": oscore["left"],
                    "obj_score_right": oscore["right"],
                    "obj_score_above": oscore["above"],
                    "obj_score_below": oscore["below"],

                    "last_score_left": lscore["left"],
                    "last_score_right": lscore["right"],
                    "last_score_above": lscore["above"],
                    "last_score_below": lscore["below"],
                }
            )

            self.done.add(li)

            if onorm <= EPS and lnorm <= EPS:
                return None

            vals[idx] = y
            return tuple(vals)

        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Eval
# =============================================================================

def select_eval_sids(
    metadata,
    records,
    split,
    sample_group,
    max_samples,
    seed,
):
    sids = []
    for sid in metadata["sids"]:
        if split != "all" and metadata["split"].get(sid, "") != split:
            continue
        if sid not in records:
            continue
        group = metadata["generation"].get(sid, {}).get(
            "generation_group", ""
        )
        if sample_group != "all" and group != sample_group:
            continue
        sids.append(sid)

    if max_samples is not None and len(sids) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(sids)
        sids = sids[:max_samples]
    return sorted(sids)


def run_eval(
    *,
    model,
    processor,
    layers,
    selected_layers,
    records,
    metadata,
    eval_sids,
    modes,
    obj_codebooks,
    obj_targets,
    last_codebooks,
    last_targets,
    args,
    out_dir,
):
    grows = []
    lrows = []
    errors = []

    for n, sid in enumerate(
        tqdm(eval_sids, desc="dynamic object+last repair"),
        1,
    ):
        rec = records[sid]
        image = None

        try:
            gt = metadata["gt"][sid]
            image = Image.open(rec.image_path).convert("RGB")

            question = args.prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )

            real_batch, sp, rp, last_pos = build_batch(
                processor,
                rec,
                question,
                image,
                torch.device(args.device),
            )

            noimg = capture_states(
                model=model,
                processor=processor,
                layers=layers,
                selected_layers=selected_layers,
                rec=rec,
                image=None,
                device=torch.device(args.device),
                prompt_template=args.prompt_template,
                capture_obj=True,
                capture_last=True,
            )

            btext, bpred = single.generate_answer(
                model,
                processor,
                real_batch,
                args.max_new_tokens,
            )
            bcorr = int(bpred == gt)

            for mode in modes:
                with DynamicRepair(
                    layers=layers,
                    selected_layers=selected_layers,
                    sp=sp,
                    rp=rp,
                    last_pos=last_pos,
                    noimg_obj=noimg["obj"],
                    noimg_last=noimg["last"],
                    obj_codebooks=obj_codebooks,
                    obj_targets=obj_targets,
                    last_codebooks=last_codebooks,
                    last_targets=last_targets,
                    gt=gt,
                    target_name=args.target,
                    mode=mode,
                    project_spatial=args.project_spatial,
                    object_max_edit_norm=args.object_max_edit_norm,
                    last_max_edit_norm=args.last_max_edit_norm,
                    sid=sid,
                    seed=args.seed,
                ) as rep:
                    etext, epred = single.generate_answer(
                        model,
                        processor,
                        real_batch,
                        args.max_new_tokens,
                    )

                ecorr = int(epred == gt)
                obj_layers = [
                    int(r["layer"])
                    for r in rep.logs
                    if int(r["object_triggered"])
                ]
                last_layers = [
                    int(r["layer"])
                    for r in rep.logs
                    if int(r["last_triggered"])
                ]

                grows.append(
                    {
                        "sid": sid,
                        "mode": mode,
                        "gt": gt,
                        "baseline_text": btext,
                        "baseline_pred": bpred or "",
                        "baseline_correct": bcorr,
                        "edited_text": etext,
                        "edited_pred": epred or "",
                        "edited_correct": ecorr,
                        "n_object_repairs": len(obj_layers),
                        "n_last_repairs": len(last_layers),
                        "object_repair_layers": ",".join(map(str, obj_layers)),
                        "last_repair_layers": ",".join(map(str, last_layers)),
                        "W2C": int(bcorr == 0 and ecorr == 1),
                        "C2W": int(bcorr == 1 and ecorr == 0),
                    }
                )

                for r in rep.logs:
                    rr = dict(r)
                    rr.update(
                        {
                            "baseline_pred": bpred or "",
                            "baseline_correct": bcorr,
                            "edited_pred": epred or "",
                            "edited_correct": ecorr,
                        }
                    )
                    lrows.append(rr)

            if args.save_every > 0 and n % args.save_every == 0:
                write_csv(out_dir / "generation_per_sample.csv", grows)
                write_csv(out_dir / "layer_diagnostics.csv", lrows)

            del real_batch

        except Exception as e:
            errors.append(
                {
                    "sid": sid,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            tqdm.write(f"[ERROR sid={sid}] {type(e).__name__}: {e}")
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(out_dir / "generation_per_sample.csv", grows)
    write_csv(out_dir / "layer_diagnostics.csv", lrows)
    write_csv(out_dir / "errors.csv", errors)
    return grows, lrows, errors


def summarize_generation(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["mode"]].append(r)

    out = []
    for mode, rr in sorted(buckets.items()):
        bacc = safe_mean(r["baseline_correct"] for r in rr)
        eacc = safe_mean(r["edited_correct"] for r in rr)
        wrong = sum(int(r["baseline_correct"]) == 0 for r in rr)
        correct = sum(int(r["baseline_correct"]) == 1 for r in rr)
        w2c = sum(int(r["W2C"]) for r in rr)
        c2w = sum(int(r["C2W"]) for r in rr)

        out.append(
            {
                "mode": mode,
                "n": len(rr),
                "baseline_acc": bacc,
                "edited_acc": eacc,
                "gain": eacc - bacc,
                "mean_object_repairs": safe_mean(
                    r["n_object_repairs"] for r in rr
                ),
                "mean_last_repairs": safe_mean(
                    r["n_last_repairs"] for r in rr
                ),
                "W2C": w2c,
                "W2C_over_wrong": w2c / wrong if wrong else float("nan"),
                "C2W": c2w,
                "C2W_over_correct": c2w / correct if correct else float("nan"),
                "net": w2c - c2w,
            }
        )
    return out


def print_summary(rows):
    print("\n" + "=" * 145)
    print("ACTUAL model.generate() — DYNAMIC MULTI-LAYER OBJECT + LAST DIRECTION REPAIR")
    print("=" * 145)
    print(
        "mode N | acc base->edit gain | mean obj/last repairs | "
        "W2C/wrong C2W/correct net"
    )
    for r in rows:
        print(
            f"{r['mode']:12s} {int(r['n']):3d} | "
            f"{r['baseline_acc']:.4f}->{r['edited_acc']:.4f} "
            f"{r['gain']:+.4f} | "
            f"{r['mean_object_repairs']:.2f}/{r['mean_last_repairs']:.2f} | "
            f"{int(r['W2C']):3d}/{r['W2C_over_wrong']:.3f} "
            f"{int(r['C2W']):3d}/{r['C2W_over_correct']:.3f} "
            f"{int(r['net']):+d}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = single.load_metadata(Path(args.direction_dir))

    records_list, _ = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    records = {int(r.sid): r for r in records_list}

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    dtype = base.resolve_dtype(spec.dtype_name)

    kw: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id} on {args.device}")
    try:
        model = cls.from_pretrained(
            spec.repo_id,
            dtype=dtype,
            **kw,
        )
    except TypeError:
        model = cls.from_pretrained(
            spec.repo_id,
            torch_dtype=dtype,
            **kw,
        )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)

    layers, layer_path = single.resolve_decoder_layers(model)
    selected_layers = parse_layers(args.layers, len(layers))

    print(
        f"[decoder] {layer_path}; n={len(layers)}; "
        f"selected={selected_layers}"
    )

    modes = parse_words(args.modes)
    valid = {"object_only", "last_only", "joint", "last_random"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"Unknown modes: {bad}")

    object_cache = Path(args.object_train_cache)
    validate_object_cache(object_cache, selected_layers)

    last_cache = (
        Path(args.last_train_cache)
        if args.last_train_cache
        else out_dir / "train_all_layer_last_states.npz"
    )

    if args.rebuild_last_train_cache or not last_cache.exists():
        collect_last_cache(
            cache_path=last_cache,
            object_cache_path=object_cache,
            model=model,
            processor=processor,
            layers=layers,
            selected_layers=selected_layers,
            records=records,
            metadata=metadata,
            device=torch.device(args.device),
            prompt_template=args.prompt_template,
        )

    validate_last_cache(last_cache, selected_layers)

    obj_cb, obj_t, obj_rows = fit_channel(
        object_cache, selected_layers, "object"
    )
    last_cb, last_t, last_rows = fit_channel(
        last_cache, selected_layers, "last"
    )
    write_csv(
        out_dir / "correct_direction_trajectories.csv",
        obj_rows + last_rows,
    )

    eval_sids = select_eval_sids(
        metadata,
        records,
        args.eval_split,
        args.sample_group,
        args.max_samples,
        args.seed,
    )

    print(
        f"[eval] cached group={args.sample_group}, N={len(eval_sids)}, "
        f"target={args.target}, modes={modes}"
    )

    grows, lrows, errors = run_eval(
        model=model,
        processor=processor,
        layers=layers,
        selected_layers=selected_layers,
        records=records,
        metadata=metadata,
        eval_sids=eval_sids,
        modes=modes,
        obj_codebooks=obj_cb,
        obj_targets=obj_t,
        last_codebooks=last_cb,
        last_targets=last_t,
        args=args,
        out_dir=out_dir,
    )

    summary = summarize_generation(grows)
    write_csv(out_dir / "generation_summary.csv", summary)
    print_summary(summary)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "gradient_used": False,
                "gt_oracle": True,
                "dynamic": True,
                "target": args.target,
                "modes": modes,
                "layers": selected_layers,
                "n_eval": len(eval_sids),
                "n_errors": len(errors),
                "object_train_cache": str(object_cache),
                "last_train_cache": str(last_cache),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    for p in [
        last_cache,
        out_dir / "correct_direction_trajectories.csv",
        out_dir / "generation_per_sample.csv",
        out_dir / "layer_diagnostics.csv",
        out_dir / "generation_summary.csv",
        out_dir / "errors.csv",
        out_dir / "summary.json",
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
