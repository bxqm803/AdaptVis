#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle L16-MLP spatial correction evaluated with ACTUAL model.generate().

What this tests
---------------
The previous diagnostic scan found L16 MLP as the strongest Qwen-7B candidate:
generation-wrong samples often receive an L16 MLP subject-reference update that
pushes toward a competing / finally generated wrong spatial relation.

This script asks whether correcting THAT COMPUTATION improves actual generation.

The experiment is oracle because GT is used to define the correct spatial
relation. It is a mechanism test, not a deployable method.

Main interventions
------------------
At decoder block L16, let

    m = [(MLP_img_sub - MLP_img_ref)
         - (MLP_noimg_sub - MLP_noimg_ref)]

and let the existing Direction prototypes be

    mu_left, mu_right, mu_above, mu_below.

For GT relation g and competitor c:

    wrong_drive(g,c) = m dot (mu_c - mu_g)

Positive = this MLP update pushes toward c relative to GT.

Correct-control targets are learned ONLY from cached TRAIN samples whose
actual model.generate() result was correct.

1) suppress
   Find the strongest current non-GT wrong_drive at L16.
   If it is above the corresponding correct-control median, reduce it to that
   median with a minimum-norm edit.

2) gt_region
   For ALL three non-GT competitors, require the GT-vs-competitor MLP margin
   to be at least the correct-control median:

       (m + delta) dot (mu_GT - mu_c) >= target_margin(GT,c)

   Find the minimum-L2 delta satisfying all violated constraints.

3) random_suppress / random_region
   Apply a norm-matched random MLP edit orthogonal to the relation-difference
   subspace, as perturbation controls.

The edit changes ONLY the L16 MLP output at subject/reference prompt tokens:

    MLP_sub <- MLP_sub + delta/2
    MLP_ref <- MLP_ref - delta/2

so the pair mean is preserved and the subject-reference MLP update changes by
exactly delta.

Evaluation
----------
Every condition is evaluated with a fresh full:

    model.generate(...)

and then parses left/right/above/below from the generated suffix.

The final summary reports:
    generation accuracy
    parsed rate
    W->C
    C->W
    net gain

on ALL selected samples, including originally-correct samples.

Required existing files
-----------------------
<direction-dir>/vectors.npz
<direction-dir>/sample_split_and_generation.csv

Required helper scripts in repo root
------------------------------------
extract_two_object_relation_states.py
analyze_layerwise_direction_failure_scan_v1.py

Recommended run
---------------
CUDA_VISIBLE_DEVICES=0 python eval_l16_mlp_oracle_generation_repair_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layer 16 \
  --split test \
  --output-dir output/qwen7b_l16_mlp_generation_repair_v1 \
  --overwrite

Smoke test
----------
CUDA_VISIBLE_DEVICES=0 python eval_l16_mlp_oracle_generation_repair_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layer 16 \
  --split test \
  --max-samples 20 \
  --output-dir output/qwen7b_l16_mlp_generation_repair_smoke \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import itertools
import json
import math
import random
import re
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
import analyze_layerwise_direction_failure_scan_v1 as direction


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
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
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
        "--target-stat",
        default="median",
        choices=["median", "mean", "q25", "q75"],
        help="Correct-control statistic used as normal L16-MLP target.",
    )
    p.add_argument(
        "--modes",
        default="suppress,gt_region,random_suppress,random_region",
        help="Comma-separated intervention modes.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--max-edit-norm",
        type=float,
        default=0.0,
        help="Optional per-sample MLP edit-norm cap; <=0 disables clipping.",
    )
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
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


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def frac(xs: Iterable[bool]) -> float:
    vals = [bool(x) for x in xs]
    return float(np.mean(vals)) if vals else float("nan")


def target_stat(vals: Sequence[float], kind: str) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        raise RuntimeError("Cannot compute target from zero controls.")
    if kind == "median":
        return float(np.median(x))
    if kind == "mean":
        return float(np.mean(x))
    if kind == "q25":
        return float(np.quantile(x, 0.25))
    if kind == "q75":
        return float(np.quantile(x, 0.75))
    raise ValueError(kind)


def norm_relation(x: Any) -> str:
    return direction.norm_relation(x)


# =============================================================================
# Model structure / tensor helpers
# =============================================================================

def get_attr_path(obj: Any, path: str):
    cur = obj
    for piece in path.split("."):
        cur = getattr(cur, piece)
    return cur


def resolve_decoder_layers(model):
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
    ]
    for path in candidates:
        try:
            layers = get_attr_path(model, path)
            if len(layers) > 0:
                block = layers[0]
                if hasattr(block, "self_attn") and hasattr(block, "mlp"):
                    return layers, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers.")


def first_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            if torch.is_tensor(item):
                return item
    raise RuntimeError(f"No tensor found in output type={type(x)}")


def replace_first_tensor(output: Any, new_tensor: torch.Tensor):
    if torch.is_tensor(output):
        return new_tensor
    if isinstance(output, tuple):
        out = list(output)
        for i, item in enumerate(out):
            if torch.is_tensor(item):
                out[i] = new_tensor
                return tuple(out)
    if isinstance(output, list):
        out = list(output)
        for i, item in enumerate(out):
            if torch.is_tensor(item):
                out[i] = new_tensor
                return out
    raise RuntimeError(f"Cannot replace tensor in output type={type(output)}")


def pool_positions(
    tensor: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    valid = [int(p) for p in positions if 0 <= int(p) < int(tensor.shape[1])]
    if not valid:
        raise RuntimeError("No valid object-token positions.")
    idx = torch.as_tensor(valid, device=tensor.device, dtype=torch.long)
    return tensor[0].index_select(0, idx).mean(dim=0)


# =============================================================================
# Cached Direction codebook
# =============================================================================

def fit_codebook(X_train: np.ndarray, y_train: np.ndarray):
    center = X_train.mean(axis=0)
    Xc = X_train - center
    protos = []
    for rel in RELATIONS:
        mask = y_train == rel
        if not np.any(mask):
            raise RuntimeError(f"No train samples for relation {rel}")
        p = Xc[mask].mean(axis=0)
        p = p / max(float(np.linalg.norm(p)), EPS)
        protos.append(p)
    return center.astype(np.float32), np.stack(protos).astype(np.float32)


def load_direction_assets(direction_dir: Path, layer: int):
    vec_path = direction_dir / "vectors.npz"
    split_path = direction_dir / "sample_split_and_generation.csv"
    if not vec_path.exists():
        raise FileNotFoundError(vec_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    with np.load(vec_path, allow_pickle=True) as z:
        arr = {k: z[k] for k in z.files}

    rows = read_csv(split_path)

    sids = arr["sample_index"].astype(np.int64)
    labels = np.asarray([norm_relation(x) for x in arr["relation"]])
    residual = np.asarray(arr["residual"], dtype=np.float32)

    if not 0 <= layer < residual.shape[1]:
        raise ValueError(
            f"Requested L{layer}, cache has {residual.shape[1]} layers."
        )

    idx_by_sid = {int(s): i for i, s in enumerate(sids.tolist())}
    split_by_sid = {
        int(r["sample_index"]): str(r["split"]).strip()
        for r in rows
    }
    cached_gen = {
        int(r["sample_index"]): {
            "generation_group": str(r.get("generation_group", "")).strip(),
            "generation_pred": norm_relation(r.get("generation_pred", "")),
            "generation_text": str(r.get("generation_text", "")),
        }
        for r in rows
    }

    train_idx = np.asarray(
        [
            idx_by_sid[int(r["sample_index"])]
            for r in rows
            if str(r["split"]).strip() == "train"
            and int(r["sample_index"]) in idx_by_sid
        ],
        dtype=np.int64,
    )
    center, protos = fit_codebook(
        residual[train_idx, layer, :],
        labels[train_idx],
    )

    return {
        "rows": rows,
        "labels": labels,
        "residual": residual,
        "idx_by_sid": idx_by_sid,
        "split_by_sid": split_by_sid,
        "cached_generation": cached_gen,
        "center": center,
        "protos": protos,
    }


# =============================================================================
# Prompt / forward / generation
# =============================================================================

def build_batch_and_positions(
    *,
    processor,
    device,
    question: str,
    subject: str,
    reference: str,
    image: Optional[Image.Image],
):
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
    subj_pos = direction.locate_phrase_positions(
        processor.tokenizer, ids, subject
    )
    ref_pos = direction.locate_phrase_positions(
        processor.tokenizer, ids, reference
    )
    return batch, subj_pos, ref_pos


class MLPDiffCollector:
    def __init__(
        self,
        module,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
    ):
        self.subj = list(map(int, subject_positions))
        self.ref = list(map(int, reference_positions))
        self.diff: Optional[torch.Tensor] = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        x = first_tensor(output)
        # Only collect on a sequence containing prompt object positions.
        if (
            int(x.shape[1]) > max(self.subj + self.ref)
            and self.diff is None
        ):
            hs = pool_positions(x, self.subj)
            hr = pool_positions(x, self.ref)
            self.diff = (hs - hr).detach().float().cpu()
        return output

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


def capture_noimg_mlp_diff(
    *,
    model,
    processor,
    device,
    mlp_module,
    question,
    subject,
    reference,
):
    batch, subj_pos, ref_pos = build_batch_and_positions(
        processor=processor,
        device=device,
        question=question,
        subject=subject,
        reference=reference,
        image=None,
    )
    col = MLPDiffCollector(mlp_module, subj_pos, ref_pos)
    try:
        with torch.inference_mode():
            _ = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        if col.diff is None:
            raise RuntimeError("No no-image MLP diff captured.")
        return col.diff
    finally:
        col.close()
        del batch


def parse_relation(text: str) -> str:
    t = str(text).lower()
    matches = []
    for rel in RELATIONS:
        m = re.search(rf"\b{re.escape(rel)}\b", t)
        if m:
            matches.append((m.start(), rel))
    if not matches:
        return ""
    matches.sort()
    return matches[0][1]


def generate_answer(
    *,
    model,
    processor,
    batch,
    max_new_tokens: int,
):
    input_len = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    new_ids = output_ids[:, input_len:]

    tokenizer = getattr(processor, "tokenizer", processor)
    texts = tokenizer.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    text = texts[0] if texts else ""
    pred = parse_relation(text)
    return {
        "text": text,
        "pred": pred,
        "parsed": int(pred in REL2ID),
    }


# =============================================================================
# Correct-control L16 MLP targets
# =============================================================================

def collect_train_control_targets(
    *,
    model,
    processor,
    device,
    mlp_module,
    records_by_sid,
    assets,
    layer,
    prompt_template,
    target_kind,
    out_dir,
):
    """
    Correct controls are TRAIN samples whose cached ACTUAL model.generate()
    result is correct. We only use them to define normal L16 MLP pairwise
    contributions.
    """
    control_sids = [
        sid
        for sid, split in assets["split_by_sid"].items()
        if split == "train"
        and assets["cached_generation"].get(sid, {}).get(
            "generation_group", ""
        ) == "correct"
        and sid in records_by_sid
    ]

    if not control_sids:
        raise RuntimeError(
            "No cached train generation-correct controls available."
        )

    cache_csv = out_dir / "train_correct_control_mlp_vectors.csv"
    vector_npz = out_dir / "train_correct_control_mlp_vectors.npz"

    rows = []
    vecs = []
    sids_done = []

    print(
        f"[controls] collecting L{layer} MLP residual vectors from "
        f"{len(control_sids)} cached train-generation-correct samples"
    )

    for sid in tqdm(control_sids, desc="train correct controls"):
        rec = records_by_sid[sid]
        image = None
        real_batch = None
        try:
            gt = norm_relation(rec.relation)
            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            image = Image.open(rec.image_path).convert("RGB")

            real_batch, subj_pos, ref_pos = build_batch_and_positions(
                processor=processor,
                device=device,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
                image=image,
            )

            real_col = MLPDiffCollector(
                mlp_module, subj_pos, ref_pos
            )
            try:
                with torch.inference_mode():
                    _ = model(
                        **real_batch,
                        output_attentions=False,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                if real_col.diff is None:
                    raise RuntimeError("No real MLP diff captured.")
                real_diff = real_col.diff
            finally:
                real_col.close()

            noimg_diff = capture_noimg_mlp_diff(
                model=model,
                processor=processor,
                device=device,
                mlp_module=mlp_module,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
            )

            m = (real_diff - noimg_diff).numpy().astype(np.float32)
            vecs.append(m)
            sids_done.append(sid)
            rows.append({
                "sid": sid,
                "gt": gt,
                "vector_norm": float(np.linalg.norm(m)),
            })

        finally:
            if image is not None:
                image.close()
            del real_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(cache_csv, rows)
    np.savez_compressed(
        vector_npz,
        sid=np.asarray(sids_done, dtype=np.int64),
        vector=np.stack(vecs).astype(np.float32),
    )

    protos = assets["protos"]
    targets = {}
    target_rows = []

    for gt in RELATIONS:
        gt_vecs = [
            v for v, row in zip(vecs, rows)
            if row["gt"] == gt
        ]
        if not gt_vecs:
            continue

        for comp in RELATIONS:
            if comp == gt:
                continue
            d = (
                protos[REL2ID[gt]]
                - protos[REL2ID[comp]]
            )
            margins = [float(v @ d) for v in gt_vecs]
            t = target_stat(margins, target_kind)
            targets[(gt, comp)] = t
            target_rows.append({
                "layer": layer,
                "gt": gt,
                "competitor": comp,
                "n_controls": len(gt_vecs),
                "target_margin": t,
                "target_stat": target_kind,
                "control_margin_mean": float(np.mean(margins)),
                "control_margin_median": float(np.median(margins)),
                "control_margin_std": float(np.std(margins)),
            })

    write_csv(out_dir / "control_pairwise_targets.csv", target_rows)

    missing = [
        (g, c)
        for g in RELATIONS
        for c in RELATIONS
        if c != g and (g, c) not in targets
    ]
    if missing:
        raise RuntimeError(f"Missing control targets: {missing}")

    return targets


# =============================================================================
# Edit construction
# =============================================================================

def minimum_norm_halfspace_edit(
    *,
    directions: Sequence[torch.Tensor],
    deficits: Sequence[float],
) -> torch.Tensor:
    """
    Solve approximately/exactly:
        min ||delta||_2
        s.t. delta dot d_j >= deficit_j

    There are only 3 constraints. Enumerate all non-empty active subsets,
    solve the equality-constrained minimum-norm candidate for each subset,
    retain feasible candidates, and choose the smallest norm.

    If no equality-subset candidate is feasible due to numerical issues,
    fall back to cyclic halfspace projections.
    """
    if len(directions) == 0:
        raise ValueError("No directions.")

    device = directions[0].device
    dtype = directions[0].dtype
    D = directions[0].numel()

    dirs32 = [d.float() for d in directions]
    deficits32 = [
        max(0.0, float(x)) for x in deficits
    ]

    if max(deficits32) <= EPS:
        return torch.zeros(D, device=device, dtype=dtype)

    best = None
    best_norm = float("inf")
    n = len(dirs32)

    for k in range(1, n + 1):
        for subset in itertools.combinations(range(n), k):
            A = torch.stack(
                [dirs32[j] for j in subset],
                dim=1,
            )  # [D,k]
            b = torch.tensor(
                [deficits32[j] for j in subset],
                device=device,
                dtype=torch.float32,
            )

            gram = A.transpose(0, 1) @ A
            pinv = torch.linalg.pinv(gram)
            delta = A @ (pinv @ b)

            feasible = True
            for dj, bj in zip(dirs32, deficits32):
                if float((delta @ dj).detach().cpu()) + 1e-4 < bj:
                    feasible = False
                    break

            if feasible:
                nrm = float(delta.norm().detach().cpu())
                if nrm < best_norm:
                    best_norm = nrm
                    best = delta

    if best is None:
        delta = torch.zeros(D, device=device, dtype=torch.float32)
        for _ in range(20):
            changed = False
            for dj, bj in zip(dirs32, deficits32):
                cur = float((delta @ dj).detach().cpu())
                if cur + 1e-5 < bj:
                    step = (bj - cur) / max(
                        float((dj @ dj).detach().cpu()), EPS
                    )
                    delta = delta + step * dj
                    changed = True
            if not changed:
                break
        best = delta

    return best.to(dtype=dtype)


def random_orthogonal_delta(
    *,
    true_delta: torch.Tensor,
    relation_dirs: Sequence[torch.Tensor],
    seed: int,
) -> torch.Tensor:
    target_norm = true_delta.float().norm()
    if float(target_norm.detach().cpu()) <= EPS:
        return torch.zeros_like(true_delta)

    rng = np.random.default_rng(seed)
    rv = torch.from_numpy(
        rng.standard_normal(true_delta.numel()).astype(np.float32)
    ).to(device=true_delta.device)

    A = torch.stack(
        [d.float() for d in relation_dirs],
        dim=1,
    )
    # Remove projection onto the relation-difference span.
    coeff = torch.linalg.pinv(A) @ rv
    rv = rv - A @ coeff

    nr = rv.norm()
    if float(nr.detach().cpu()) <= EPS:
        return torch.zeros_like(true_delta)

    rv = rv * (target_norm / nr)
    return rv.to(dtype=true_delta.dtype)


class L16MLPOracleEdit:
    """
    Edit only the first/prefill call to block.mlp during model.generate().
    Decode-step MLP calls are untouched.
    """

    def __init__(
        self,
        *,
        module,
        sid: int,
        gt: str,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        noimg_diff_cpu: torch.Tensor,
        protos_cpu: torch.Tensor,
        targets: Mapping[Tuple[str, str], float],
        mode: str,
        seed: int,
        max_edit_norm: float,
    ):
        self.sid = int(sid)
        self.gt = gt
        self.subj = list(map(int, subject_positions))
        self.ref = list(map(int, reference_positions))
        self.noimg_diff_cpu = noimg_diff_cpu
        self.protos_cpu = protos_cpu
        self.targets = targets
        self.mode = mode
        self.seed = int(seed)
        self.max_edit_norm = float(max_edit_norm)

        self.applied = 0
        self.trace: Dict[str, Any] = {}
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        x = first_tensor(output)

        # During generation, only edit the prompt prefill. Later decode calls
        # usually have sequence length 1.
        if self.applied:
            return output
        if int(x.shape[1]) <= max(self.subj + self.ref):
            return output

        hs = pool_positions(x, self.subj)
        hr = pool_positions(x, self.ref)
        noimg = self.noimg_diff_cpu.to(
            x.device, dtype=x.dtype
        )
        protos = self.protos_cpu.to(
            x.device, dtype=x.dtype
        )

        m = (hs - hr) - noimg

        gt_i = REL2ID[self.gt]
        g = protos[gt_i]

        competitors = [
            r for r in RELATIONS if r != self.gt
        ]
        relation_dirs = [
            g - protos[REL2ID[c]]
            for c in competitors
        ]

        current_margins = [
            float((m @ d).detach().float().cpu())
            for d in relation_dirs
        ]
        target_margins = [
            float(self.targets[(self.gt, c)])
            for c in competitors
        ]
        deficits = [
            max(0.0, t - cur)
            for cur, t in zip(current_margins, target_margins)
        ]

        if self.mode in ("suppress", "random_suppress"):
            # "Strongest wrong" = largest target deficit / most below the
            # normal GT-vs-competitor margin.
            j = int(np.argmax(deficits))
            if deficits[j] <= EPS:
                true_delta = torch.zeros_like(m)
            else:
                d = relation_dirs[j]
                denom = d.float() @ d.float()
                true_delta = (
                    deficits[j] / max(
                        float(denom.detach().cpu()), EPS
                    )
                ) * d

            chosen = competitors[j]
            active = [chosen] if deficits[j] > EPS else []

        elif self.mode in ("gt_region", "random_region"):
            true_delta = minimum_norm_halfspace_edit(
                directions=relation_dirs,
                deficits=deficits,
            )
            chosen = ";".join(
                c for c, df in zip(competitors, deficits)
                if df > EPS
            )
            active = [
                c for c, df in zip(competitors, deficits)
                if df > EPS
            ]
        else:
            raise ValueError(self.mode)

        is_random = self.mode.startswith("random_")
        if is_random:
            delta = random_orthogonal_delta(
                true_delta=true_delta,
                relation_dirs=relation_dirs,
                seed=self.seed,
            )
        else:
            delta = true_delta

        unclipped_norm = float(
            delta.detach().float().norm().cpu()
        )
        clipped = False
        if (
            self.max_edit_norm > 0
            and unclipped_norm > self.max_edit_norm
        ):
            delta = delta * (
                self.max_edit_norm
                / max(unclipped_norm, EPS)
            )
            clipped = True

        delta_norm = float(
            delta.detach().float().norm().cpu()
        )

        y = x.clone()
        half = delta / 2.0
        y[0, self.subj, :] = y[0, self.subj, :] + half
        y[0, self.ref, :] = y[0, self.ref, :] - half

        m_after = m + delta
        post_margins = [
            float((m_after @ d).detach().float().cpu())
            for d in relation_dirs
        ]

        self.trace = {
            "sid": self.sid,
            "mode": self.mode,
            "gt": self.gt,
            "chosen_or_active_competitors": chosen,
            "n_active_constraints": len(active),
            "triggered": int(unclipped_norm > EPS),
            "delta_norm": delta_norm,
            "unclipped_delta_norm": unclipped_norm,
            "clipped": int(clipped),
        }
        for c, cur, tar, post, df in zip(
            competitors,
            current_margins,
            target_margins,
            post_margins,
            deficits,
        ):
            self.trace[f"pre_margin_{c}"] = cur
            self.trace[f"target_margin_{c}"] = tar
            self.trace[f"deficit_{c}"] = df
            self.trace[f"post_margin_{c}"] = post

        self.applied += 1
        return replace_first_tensor(output, y)

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()

    def validate(self):
        # A zero-norm/no-op intervention still goes through the hook once.
        if self.applied != 1:
            raise RuntimeError(
                f"sid={self.sid} {self.mode}: hook applied {self.applied} times"
            )


# =============================================================================
# Evaluation summary
# =============================================================================

def summarize(
    rows: Sequence[Mapping[str, Any]],
    modes: Sequence[str],
    out_dir: Path,
):
    n = len(rows)
    baseline_correct = sum(
        int(r["baseline_correct"]) for r in rows
    )
    baseline_parsed = sum(
        int(r["baseline_parsed"]) for r in rows
    )

    summary = []
    for mode in modes:
        correct = sum(
            int(r[f"{mode}_correct"]) for r in rows
        )
        parsed = sum(
            int(r[f"{mode}_parsed"]) for r in rows
        )
        w2c = sum(
            int(r[f"{mode}_W_to_C"]) for r in rows
        )
        c2w = sum(
            int(r[f"{mode}_C_to_W"]) for r in rows
        )

        base_wrong = n - baseline_correct

        summary.append({
            "mode": mode,
            "n": n,
            "baseline_acc_all": baseline_correct / n if n else float("nan"),
            "intervention_acc_all": correct / n if n else float("nan"),
            "accuracy_gain": (
                correct - baseline_correct
            ) / n if n else float("nan"),
            "baseline_parsed_rate": baseline_parsed / n if n else float("nan"),
            "intervention_parsed_rate": parsed / n if n else float("nan"),
            "W_to_C": w2c,
            "W_to_C_rate_among_baseline_wrong":
                w2c / base_wrong if base_wrong else float("nan"),
            "C_to_W": c2w,
            "C_to_W_rate_among_baseline_correct":
                c2w / baseline_correct if baseline_correct else float("nan"),
            "net_correct_gain": w2c - c2w,
            "prediction_change_rate": frac(
                str(r[f"{mode}_pred"]) != str(r["baseline_pred"])
                for r in rows
            ),
            "trigger_rate": frac(
                int(r[f"{mode}_triggered"]) == 1
                for r in rows
            ),
            "mean_edit_norm": mean(
                r[f"{mode}_delta_norm"] for r in rows
            ),
            "trigger_rate_baseline_correct": frac(
                int(r[f"{mode}_triggered"]) == 1
                for r in rows
                if int(r["baseline_correct"])
            ),
            "trigger_rate_baseline_wrong": frac(
                int(r[f"{mode}_triggered"]) == 1
                for r in rows
                if not int(r["baseline_correct"])
            ),
        })

    write_csv(out_dir / "generation_summary.csv", summary)

    print("\n" + "=" * 160)
    print("ACTUAL model.generate() — L16 MLP ORACLE REPAIR")
    print("=" * 160)
    print(
        f"N={n} baseline_acc(all)={baseline_correct/n:.4f} "
        f"baseline_parsed={baseline_parsed/n:.4f}"
    )
    print(
        "mode              acc      gain      parsed   W2C  W2C/wrong  "
        "C2W  C2W/correct  net   trigger(correct/wrong)  editNorm"
    )
    for r in summary:
        print(
            f"{r['mode']:<17s} "
            f"{r['intervention_acc_all']:.4f}  "
            f"{r['accuracy_gain']:+.4f}   "
            f"{r['intervention_parsed_rate']:.4f}   "
            f"{int(r['W_to_C']):3d}   "
            f"{r['W_to_C_rate_among_baseline_wrong']:.3f}      "
            f"{int(r['C_to_W']):3d}   "
            f"{r['C_to_W_rate_among_baseline_correct']:.3f}       "
            f"{int(r['net_correct_gain']):+4d}   "
            f"{r['trigger_rate_baseline_correct']:.3f}/"
            f"{r['trigger_rate_baseline_wrong']:.3f}              "
            f"{r['mean_edit_norm']:.3f}"
        )

    by_mode = {r["mode"]: r for r in summary}
    specific = []
    for real, rand in [
        ("suppress", "random_suppress"),
        ("gt_region", "random_region"),
    ]:
        if real not in by_mode or rand not in by_mode:
            continue
        a, b = by_mode[real], by_mode[rand]
        specific.append({
            "mode": real,
            "accuracy_gain_minus_random":
                a["accuracy_gain"] - b["accuracy_gain"],
            "net_gain_minus_random":
                a["net_correct_gain"] - b["net_correct_gain"],
            "W_to_C_minus_random":
                a["W_to_C"] - b["W_to_C"],
            "C_to_W_minus_random":
                a["C_to_W"] - b["C_to_W"],
        })

    write_csv(out_dir / "specific_vs_random.csv", specific)

    if specific:
        print("\nSpecific effect vs norm-matched random:")
        for r in specific:
            print(
                f"{r['mode']:<12s} "
                f"accSpecific={r['accuracy_gain_minus_random']:+.4f} "
                f"netSpecific={int(r['net_gain_minus_random']):+d} "
                f"W2Cspecific={int(r['W_to_C_minus_random']):+d} "
                f"C2Wspecific={int(r['C_to_W_minus_random']):+d}"
            )

    # Baseline parsed-only accuracy is also useful to compare with old cached
    # generation experiments that excluded parse failures.
    parsed_rows = [r for r in rows if int(r["baseline_parsed"])]
    parsed_base_acc = frac(
        int(r["baseline_correct"]) == 1 for r in parsed_rows
    )
    print(
        f"\nBaseline parsed-only: N={len(parsed_rows)}, "
        f"acc={parsed_base_acc:.4f}"
    )

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    modes = [
        x.strip() for x in args.modes.split(",")
        if x.strip()
    ]
    valid_modes = {
        "suppress",
        "gt_region",
        "random_suppress",
        "random_region",
    }
    unknown = [m for m in modes if m not in valid_modes]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_direction_assets(
        Path(args.direction_dir),
        args.layer,
    )

    # Load all dataset records once.
    all_records, _audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    records_by_sid = {
        int(r.sid): r for r in all_records
        if norm_relation(r.relation) in REL2ID
    }

    # Evaluation split.
    if args.split == "all":
        eval_sids = [
            sid for sid in assets["split_by_sid"]
            if sid in records_by_sid
        ]
    else:
        eval_sids = [
            sid
            for sid, sp in assets["split_by_sid"].items()
            if sp == args.split and sid in records_by_sid
        ]

    eval_sids.sort()
    if args.max_samples is not None and len(eval_sids) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(eval_sids)
        eval_sids = eval_sids[: int(args.max_samples)]

    eval_records = [records_by_sid[sid] for sid in eval_sids]
    if not eval_records:
        raise RuntimeError("No evaluation records.")

    print(
        f"[data] eval split={args.split} N={len(eval_records)} "
        f"layer=L{args.layer}"
    )

    # Model.
    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    kw: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
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

    decoder_layers, layer_path = resolve_decoder_layers(model)
    if not 0 <= args.layer < len(decoder_layers):
        raise ValueError(
            f"L{args.layer} invalid for {len(decoder_layers)} decoder blocks"
        )
    mlp_module = decoder_layers[args.layer].mlp
    print(f"[decoder] {layer_path}; using L{args.layer}.mlp")

    # Train-generation-correct controls define normal MLP pairwise margins.
    targets = collect_train_control_targets(
        model=model,
        processor=processor,
        device=device,
        mlp_module=mlp_module,
        records_by_sid=records_by_sid,
        assets=assets,
        layer=args.layer,
        prompt_template=args.prompt_template,
        target_kind=args.target_stat,
        out_dir=out_dir,
    )

    # Save prototype diagnostics.
    proto_rows = []
    for gt in RELATIONS:
        for comp in RELATIONS:
            if comp == gt:
                continue
            g = assets["protos"][REL2ID[gt]]
            c = assets["protos"][REL2ID[comp]]
            proto_rows.append({
                "gt": gt,
                "competitor": comp,
                "prototype_cosine": float(g @ c),
                "difference_norm": float(np.linalg.norm(g - c)),
                "target_margin": targets[(gt, comp)],
            })
    write_csv(out_dir / "prototype_target_diagnostics.csv", proto_rows)

    rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    per_path = out_dir / "per_sample_generation.csv"
    trace_path = out_dir / "per_sample_edit_trace.csv"

    for rec in tqdm(eval_records, desc="L16 MLP generation repair"):
        image = None
        batch = None
        try:
            sid = int(rec.sid)
            gt = norm_relation(rec.relation)
            question = args.prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )

            image = Image.open(rec.image_path).convert("RGB")
            batch, subj_pos, ref_pos = build_batch_and_positions(
                processor=processor,
                device=device,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
                image=image,
            )

            # Fresh baseline generation in the same runtime/configuration.
            baseline = generate_answer(
                model=model,
                processor=processor,
                batch=batch,
                max_new_tokens=args.max_new_tokens,
            )
            baseline_correct = int(baseline["pred"] == gt)

            noimg_diff = capture_noimg_mlp_diff(
                model=model,
                processor=processor,
                device=device,
                mlp_module=mlp_module,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
            )

            row: Dict[str, Any] = {
                "sid": sid,
                "gt": gt,
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "baseline_pred": baseline["pred"],
                "baseline_text": baseline["text"],
                "baseline_parsed": baseline["parsed"],
                "baseline_correct": baseline_correct,
                "cached_generation_group":
                    assets["cached_generation"].get(sid, {}).get(
                        "generation_group", ""
                    ),
                "cached_generation_pred":
                    assets["cached_generation"].get(sid, {}).get(
                        "generation_pred", ""
                    ),
            }

            for mi, mode in enumerate(modes):
                patch = L16MLPOracleEdit(
                    module=mlp_module,
                    sid=sid,
                    gt=gt,
                    subject_positions=subj_pos,
                    reference_positions=ref_pos,
                    noimg_diff_cpu=noimg_diff,
                    protos_cpu=torch.from_numpy(assets["protos"]),
                    targets=targets,
                    mode=mode,
                    seed=args.seed + sid * 1009 + mi * 100003,
                    max_edit_norm=args.max_edit_norm,
                )
                try:
                    result = generate_answer(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=args.max_new_tokens,
                    )
                    patch.validate()
                finally:
                    patch.close()

                correct = int(result["pred"] == gt)

                row[f"{mode}_pred"] = result["pred"]
                row[f"{mode}_text"] = result["text"]
                row[f"{mode}_parsed"] = result["parsed"]
                row[f"{mode}_correct"] = correct
                row[f"{mode}_W_to_C"] = int(
                    (not baseline_correct) and correct
                )
                row[f"{mode}_C_to_W"] = int(
                    baseline_correct and (not correct)
                )
                row[f"{mode}_triggered"] = int(
                    patch.trace.get("triggered", 0)
                )
                row[f"{mode}_delta_norm"] = float(
                    patch.trace.get("delta_norm", float("nan"))
                )

                tr = dict(patch.trace)
                tr["baseline_correct"] = baseline_correct
                tr["baseline_pred"] = baseline["pred"]
                tr["result_pred"] = result["pred"]
                tr["result_correct"] = correct
                trace_rows.append(tr)

            rows.append(row)

            if len(rows) % args.save_every == 0:
                write_csv(per_path, rows)
                write_csv(trace_path, trace_rows)

        except Exception as e:
            errors.append({
                "sid": int(getattr(rec, "sid", -1)),
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[ERROR sid={getattr(rec, 'sid', '?')}] "
                f"{type(e).__name__}: {e}"
            )
        finally:
            if image is not None:
                image.close()
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(per_path, rows)
    write_csv(trace_path, trace_rows)
    (out_dir / "errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summarize(rows, modes, out_dir)

    # Relation-specific harm/benefit.
    rel_rows = []
    for gt in RELATIONS:
        rr = [r for r in rows if r["gt"] == gt]
        if not rr:
            continue
        base_acc = frac(int(r["baseline_correct"]) == 1 for r in rr)
        for mode in modes:
            new_acc = frac(int(r[f"{mode}_correct"]) == 1 for r in rr)
            rel_rows.append({
                "gt": gt,
                "mode": mode,
                "n": len(rr),
                "baseline_acc": base_acc,
                "intervention_acc": new_acc,
                "gain": new_acc - base_acc,
                "W_to_C": sum(int(r[f"{mode}_W_to_C"]) for r in rr),
                "C_to_W": sum(int(r[f"{mode}_C_to_W"]) for r in rr),
            })
    write_csv(out_dir / "summary_by_relation.csv", rel_rows)

    meta = {
        "experiment":
            "oracle L16 MLP correction evaluated with actual model.generate()",
        "model": args.model,
        "dataset": args.dataset,
        "layer": args.layer,
        "split": args.split,
        "n_success": len(rows),
        "n_errors": len(errors),
        "target_stat": args.target_stat,
        "modes": modes,
        "correct_control_source":
            "cached TRAIN samples with actual generation_group=correct",
        "evaluation":
            "fresh model.generate() for baseline and every intervention",
        "oracle": True,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved to:", out_dir)
    print("  control_pairwise_targets.csv")
    print("  prototype_target_diagnostics.csv")
    print("  per_sample_generation.csv")
    print("  per_sample_edit_trace.csv")
    print("  generation_summary.csv")
    print("  specific_vs_random.csv")
    print("  summary_by_relation.csv")
    print("  errors.json")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
