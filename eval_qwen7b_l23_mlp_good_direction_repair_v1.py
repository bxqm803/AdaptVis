#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-7B L23 MLP GOOD-vs-BAD DIRECTION REPAIR.

Goal
====
The earlier correct-vs-wrong trajectory analysis found the largest group-level
bifurcation at L23.  This script tests a more specific hypothesis:

    "At L23, successful samples receive a systematically different MLP update
     from failure samples.  If we learn that successful update direction on a
     TRAIN split, can a held-out TEST sample be repaired by moving ONLY the L23
     prompt-last MLP contribution toward the successful trajectory?"

Natural L23 block
=================
    x23
      -> attention contribution a23
      -> r23 = x23 + a23
      -> MLP contribution m23
      -> y23 = r23 + m23

The script first traces natural x23/r23/m23/y23 and true Logit-Lens predictions.
It then learns relation-conditioned MLP update directions on TRAIN only.

Two definitions of GOOD/BAD are supported simultaneously:

(1) final grouping  (matches the original Qwen-7B correct-vs-wrong analysis)
    GOOD: clean native first-step prediction is GT
    BAD : clean native first-step prediction is not GT

(2) transition grouping  (cleaner module-boundary diagnosis)
    GOOD: r23 is Logit-Lens correct AND y23 remains correct
    BAD : r23 is Logit-Lens correct BUT y23 becomes wrong

For each relation r:
    d_r = mean(m23 | GOOD, GT=r) - mean(m23 | BAD, GT=r)
    u_r = d_r / ||d_r||

and TRAIN estimates the successful projection target:
    t_r = mean(<m23, u_r> | GOOD, GT=r)

Repair
======
On a held-out sample, the natural L23 MLP output m is changed only along u_r:

    m' = m + lambda * (t_r - <m, u_r>) * u_r

Thus the orthogonal component of the natural MLP update is preserved.
All layers after L23 recompute normally.

This is intentionally different from replacing the whole hidden state with a
correct centroid.

Data split
==========
Default: stratified by GT relation
    TRAIN 50%
    VAL   25%
    TEST  25%

TRAIN:
    learn directions / targets.
VAL:
    select ONE global lambda per fit mode from --lambda-grid using an oracle
    gate + GT relation.  This is discovery/tuning only.
TEST:
    directions and lambda are frozen.

Primary TEST modes
==================
clean
final_oracle_gt
    final-group direction; patch only clean first-step WRONG samples; GT picks
    relation direction.  Oracle diagnostic.
final_oracle_predrel
    same oracle gate, but relation direction is selected from clean r23's own
    Logit-Lens prediction (GT-free relation selector).
transition_oracle_gt
    transition direction; patch only r23-correct -> y23-wrong samples; GT
    selects relation direction.  Oracle diagnostic.
transition_oracle_predrel
    same transition oracle gate, but relation direction is selected from r23.

Additional first-step controls are also run:
final_all_predrel
    no correctness gate; patch every sample using r23 predicted relation.
final_oracle_gt_shuffled_relation
    same oracle gate but deliberately use another relation's learned direction.
final_oracle_gt_random_direction
    same scalar correction magnitude as the learned direction, but apply it in
    a fixed random unit direction.  This controls for intervention norm.

Important
=========
This v1 is a MECHANISTIC DISCOVERY experiment, not yet a deployable GT-free
repair.  GT is allowed for TRAIN labels and for explicit oracle TEST modes.
If held-out oracle-direction repair works, the next step is to learn a failure
controller from the pre-intervention state r23 only.

Required repo helper modules
============================
    analyze_qwen7b_l26_l27_attention_overwrite_v1.py
    analyze_coco_head_object_residual_direction_probe_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py

Example
=======
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_qwen7b_l23_mlp_good_direction_repair_v1.py \
  --model qwen-7b \
  --num-samples 0 \
  --device cuda:0 \
  --output-dir output/qwen7b_l23_mlp_good_direction_repair_v1 \
  --overwrite

Quick check:
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_qwen7b_l23_mlp_good_direction_repair_v1.py \
  --model qwen-7b \
  --num-samples 120 \
  --device cuda:0 \
  --no-run-generation \
  --output-dir output/qwen7b_l23_mlp_good_direction_repair_n120_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


SCRIPT_VERSION = "qwen7b-l23-mlp-good-direction-repair-v1"
RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
SHUFFLE_RELATION = {
    "left": "above",
    "above": "right",
    "right": "below",
    "below": "left",
}

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)
DEFAULT_LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
EPS = 1e-12


# =============================================================================
# CLI / IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--layer", type=int, default=23)

    p.add_argument("--train-ratio", type=float, default=0.50)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument(
        "--lambda-grid",
        default=",".join(str(x) for x in DEFAULT_LAMBDA_GRID),
    )
    p.add_argument(
        "--fit-modes",
        default="final,transition",
        help="Comma-separated subset of: final,transition",
    )
    p.add_argument(
        "--min-good-per-relation",
        type=int,
        default=3,
        help="Below this count, relation-specific direction falls back to global.",
    )
    p.add_argument(
        "--min-bad-per-relation",
        type=int,
        default=2,
        help="Below this count, relation-specific direction falls back to global.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument(
        "--run-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--generation-modes",
        default=(
            "clean,final_oracle_gt,final_oracle_predrel,"
            "transition_oracle_gt,transition_oracle_predrel"
        ),
        help="Comma-separated TEST modes for full model.generate().",
    )
    p.add_argument("--prompt-template", default=DEFAULT_PROMPT)

    p.add_argument(
        "--helper-module",
        default="analyze_qwen7b_l26_l27_attention_overwrite_v1",
    )
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def parse_float_grid(text: str) -> List[float]:
    out: List[float] = []
    seen = set()
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value not in seen:
            out.append(value)
            seen.add(value)
    if not out:
        raise ValueError("Empty lambda grid.")
    return out


def parse_names(text: str, allowed: Sequence[str]) -> List[str]:
    allowed_set = set(allowed)
    out: List[str] = []
    seen = set()
    for chunk in str(text).split(","):
        name = chunk.strip()
        if not name:
            continue
        if name not in allowed_set:
            raise ValueError(f"Unsupported name {name!r}; allowed={sorted(allowed_set)}")
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def safe_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.mean(xs)) if xs else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


# =============================================================================
# Relation scoring
# =============================================================================

def relation_scores_from_vocab_logits(
    vocab_logits: torch.Tensor,
    relation_token_map: Mapping[str, Sequence[int]],
) -> np.ndarray:
    scores: List[float] = []
    for relation in RELATIONS:
        ids = torch.as_tensor(
            list(map(int, relation_token_map[relation])),
            device=vocab_logits.device,
            dtype=torch.long,
        )
        values = vocab_logits.index_select(0, ids)
        scores.append(float(values.max().item()))
    return np.asarray(scores, dtype=np.float32)


def relation_metrics(scores: np.ndarray, gt: str) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64)
    gt_id = RID[gt]
    wrong_ids = [i for i in range(len(RELATIONS)) if i != gt_id]
    competitor = max(wrong_ids, key=lambda i: float(s[i]))
    pred_id = int(np.argmax(s))

    shifted = s - np.max(s)
    p = np.exp(shifted)
    p = p / max(float(p.sum()), EPS)

    return {
        "prediction": RELATIONS[pred_id],
        "correct": pred_id == gt_id,
        "decision_margin": float(s[gt_id] - s[competitor]),
        "opposite_margin": float(s[gt_id] - s[RID[OPPOSITE[gt]]]),
        "p_gt_4way": float(p[gt_id]),
        "top_competitor": RELATIONS[competitor],
    }


def prediction_from_scores(scores: np.ndarray) -> str:
    return RELATIONS[int(np.argmax(np.asarray(scores)))]


# =============================================================================
# Module-output patching
# =============================================================================

def first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(f"Expected [B,S,D], got {tuple(output.shape)}")
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("Could not find [B,S,D] tensor in module output.")


def replace_first_3d(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return items
    raise RuntimeError("Could not replace [B,S,D] tensor in module output.")


class PromptLastProjectionRepairHook:
    """
    Modify ONLY the prompt-last MLP output on the full multimodal prefill.

    Learned repair:
        scalar = lambda * (target - <m, u_true>)
        m' = m + scalar * u_patch

    Usually u_patch == u_true.
    For the random-direction norm control, u_patch is random but scalar is still
    computed from the learned u_true, preserving the intervention magnitude.
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        prompt_length: int,
        prompt_last: int,
        true_direction: np.ndarray,
        patch_direction: np.ndarray,
        target_projection: float,
        lambda_value: float,
        label: str,
    ) -> None:
        self.module = module
        self.prompt_length = int(prompt_length)
        self.prompt_last = int(prompt_last)
        self.true_direction_np = np.asarray(true_direction, dtype=np.float32)
        self.patch_direction_np = np.asarray(patch_direction, dtype=np.float32)
        self.target_projection = float(target_projection)
        self.lambda_value = float(lambda_value)
        self.label = str(label)
        self.handle = None
        self.applications = 0
        self.last_natural_projection = float("nan")
        self.last_scalar = float("nan")
        self.last_delta_norm = float("nan")

    def __enter__(self) -> "PromptLastProjectionRepairHook":
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = first_3d(output)

            # Full prompt prefill only; ignore autoregressive q_len=1 calls.
            if int(hidden.shape[1]) != self.prompt_length:
                return None
            if not (0 <= self.prompt_last < int(hidden.shape[1])):
                raise RuntimeError(
                    f"{self.label}: prompt_last={self.prompt_last}, "
                    f"q_len={hidden.shape[1]}"
                )

            current = hidden[0, self.prompt_last]
            u_true = torch.as_tensor(
                self.true_direction_np,
                device=current.device,
                dtype=torch.float32,
            )
            u_patch = torch.as_tensor(
                self.patch_direction_np,
                device=current.device,
                dtype=torch.float32,
            )
            current32 = current.float()
            projection = torch.dot(current32, u_true)
            scalar = self.lambda_value * (self.target_projection - projection)
            delta32 = scalar * u_patch

            modified = hidden.clone()
            modified[0, self.prompt_last] = (
                current32 + delta32
            ).to(dtype=current.dtype)

            self.applications += 1
            self.last_natural_projection = float(projection.item())
            self.last_scalar = float(scalar.item())
            self.last_delta_norm = float(delta32.norm().item())

            return replace_first_3d(output, modified)

        self.handle = self.module.register_forward_hook(hook)
        return self

    def validate(self, expected: int = 1) -> None:
        if self.applications != int(expected):
            raise RuntimeError(
                f"{self.label}: expected {expected} prefill patch application(s), "
                f"got {self.applications}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# =============================================================================
# Natural tracing
# =============================================================================

def run_trace_l23(
    *,
    ah: Any,
    helper: Any,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[torch.nn.Module],
    layer: int,
    lens: Any,
    gt: str,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    prompt_last = int(batch["input_ids"].shape[1]) - 1
    trace_layers = [layer - 1, layer]

    baseline, traces = ah.run_and_trace(
        model=model,
        batch=batch,
        token_map=token_map,
        decoder_layers=decoder_layers,
        layer_indices=trace_layers,
        target_positions=[prompt_last],
    )

    x = helper.trace_block_state(traces[layer - 1], prompt_last).astype(np.float32)
    attn = helper.trace_attention_state(traces[layer], prompt_last).astype(np.float32)
    r = (x + attn).astype(np.float32)
    y = helper.trace_block_state(traces[layer], prompt_last).astype(np.float32)
    mlp = (y - r).astype(np.float32)

    scores = lens.scores(np.stack([x, r, y], axis=0))
    xm = relation_metrics(scores[0], gt)
    rm = relation_metrics(scores[1], gt)
    ym = relation_metrics(scores[2], gt)

    first_pred = helper.normalize_relation(baseline["prediction"])
    if first_pred not in RELATIONS:
        raise RuntimeError(f"Bad native first-step prediction: {baseline['prediction']!r}")

    meta = {
        "firststep_pred": first_pred,
        "firststep_correct": first_pred == gt,
        "x_pred": xm["prediction"],
        "x_correct": xm["correct"],
        "x_margin": xm["decision_margin"],
        "r_pred": rm["prediction"],
        "r_correct": rm["correct"],
        "r_margin": rm["decision_margin"],
        "y_pred": ym["prediction"],
        "y_correct": ym["correct"],
        "y_margin": ym["decision_margin"],
        "mlp_gain": ym["decision_margin"] - rm["decision_margin"],
        "mlp_norm": float(np.linalg.norm(mlp)),
        "r_norm": float(np.linalg.norm(r)),
    }
    vectors = {"x": x, "r": r, "mlp": mlp, "y": y}

    del traces
    return meta, vectors


# =============================================================================
# Splitting
# =============================================================================

def stratified_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[int]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = random.Random(seed)
    by_rel: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        by_rel[str(row["gt"])].append(int(row["sid"]))

    out = {"train": [], "val": [], "test": []}
    for relation in RELATIONS:
        ids = list(by_rel.get(relation, []))
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Keep at least one TEST sample when relation has enough examples.
        if n >= 3 and n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        out["train"].extend(ids[:n_train])
        out["val"].extend(ids[n_train:n_train + n_val])
        out["test"].extend(ids[n_train + n_val:])

    for key in out:
        rng.shuffle(out[key])
    return out


# =============================================================================
# Direction fitting
# =============================================================================

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def is_good_bad(row: Mapping[str, Any], fit_mode: str) -> Tuple[bool, bool]:
    if fit_mode == "final":
        return bool(row["firststep_correct"]), not bool(row["firststep_correct"])
    if fit_mode == "transition":
        r_ok = bool(row["r_correct"])
        y_ok = bool(row["y_correct"])
        return (r_ok and y_ok), (r_ok and (not y_ok))
    raise ValueError(f"Unknown fit mode {fit_mode}")


def fit_one_direction(
    good_vectors: Sequence[np.ndarray],
    bad_vectors: Sequence[np.ndarray],
) -> Optional[Dict[str, Any]]:
    if not good_vectors or not bad_vectors:
        return None
    g = np.stack(good_vectors, axis=0).astype(np.float32)
    b = np.stack(bad_vectors, axis=0).astype(np.float32)
    mu_g = g.mean(axis=0)
    mu_b = b.mean(axis=0)
    d = (mu_g - mu_b).astype(np.float32)
    dnorm = float(np.linalg.norm(d))
    if not math.isfinite(dnorm) or dnorm <= 1e-8:
        return None
    u = (d / dnorm).astype(np.float32)
    proj_g = g @ u
    proj_b = b @ u
    target = float(np.mean(proj_g))
    return {
        "direction": u,
        "target_projection": target,
        "mu_good": mu_g,
        "mu_bad": mu_b,
        "n_good": int(g.shape[0]),
        "n_bad": int(b.shape[0]),
        "mean_norm_good": float(np.linalg.norm(g, axis=1).mean()),
        "mean_norm_bad": float(np.linalg.norm(b, axis=1).mean()),
        "norm_mu_good": float(np.linalg.norm(mu_g)),
        "norm_mu_bad": float(np.linalg.norm(mu_b)),
        "cos_mu_good_bad": cosine(mu_g, mu_b),
        "direction_norm_before_normalize": dnorm,
        "projection_good_mean": float(np.mean(proj_g)),
        "projection_good_std": float(np.std(proj_g)),
        "projection_bad_mean": float(np.mean(proj_b)),
        "projection_bad_std": float(np.std(proj_b)),
        "projection_gap": float(np.mean(proj_g) - np.mean(proj_b)),
    }


def fit_directions(
    *,
    fit_mode: str,
    train_rows: Sequence[Mapping[str, Any]],
    vectors_by_sid: Mapping[int, Mapping[str, np.ndarray]],
    min_good: int,
    min_bad: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    grouped_good: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_bad: Dict[str, List[np.ndarray]] = defaultdict(list)
    all_good: List[np.ndarray] = []
    all_bad: List[np.ndarray] = []

    for row in train_rows:
        good, bad = is_good_bad(row, fit_mode)
        sid = int(row["sid"])
        relation = str(row["gt"])
        m = vectors_by_sid[sid]["mlp"]
        if good:
            grouped_good[relation].append(m)
            all_good.append(m)
        if bad:
            grouped_bad[relation].append(m)
            all_bad.append(m)

    directions: Dict[str, Dict[str, Any]] = {}
    diagnostics: List[Dict[str, Any]] = []

    global_fit = fit_one_direction(all_good, all_bad)
    if global_fit is None:
        raise RuntimeError(
            f"Cannot fit global {fit_mode} direction: "
            f"N_good={len(all_good)} N_bad={len(all_bad)}"
        )
    directions["__global__"] = global_fit
    diagnostics.append({
        "fit_mode": fit_mode,
        "relation": "__global__",
        "used_fallback": False,
        **{k: v for k, v in global_fit.items() if not isinstance(v, np.ndarray)},
    })

    for relation in RELATIONS:
        gs = grouped_good.get(relation, [])
        bs = grouped_bad.get(relation, [])
        relation_fit = None
        if len(gs) >= min_good and len(bs) >= min_bad:
            relation_fit = fit_one_direction(gs, bs)

        if relation_fit is None:
            directions[relation] = global_fit
            diagnostics.append({
                "fit_mode": fit_mode,
                "relation": relation,
                "used_fallback": True,
                "raw_n_good": len(gs),
                "raw_n_bad": len(bs),
                **{k: v for k, v in global_fit.items() if not isinstance(v, np.ndarray)},
            })
        else:
            directions[relation] = relation_fit
            diagnostics.append({
                "fit_mode": fit_mode,
                "relation": relation,
                "used_fallback": False,
                "raw_n_good": len(gs),
                "raw_n_bad": len(bs),
                **{k: v for k, v in relation_fit.items() if not isinstance(v, np.ndarray)},
            })

    return directions, diagnostics


def save_directions_npz(
    path: Path,
    all_directions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    payload: Dict[str, np.ndarray] = {}
    for fit_mode, directions in all_directions.items():
        for relation, spec in directions.items():
            prefix = f"{fit_mode}__{relation}"
            payload[prefix + "__direction"] = np.asarray(spec["direction"], dtype=np.float32)
            payload[prefix + "__target"] = np.asarray(
                [float(spec["target_projection"])], dtype=np.float32
            )
            payload[prefix + "__mu_good"] = np.asarray(spec["mu_good"], dtype=np.float32)
            payload[prefix + "__mu_bad"] = np.asarray(spec["mu_bad"], dtype=np.float32)
    np.savez_compressed(path, **payload)


# =============================================================================
# Batch / patched forward / generation
# =============================================================================

def build_batch_for_record(
    *,
    record: Any,
    args: argparse.Namespace,
    helper: Any,
    probe: Any,
    processor: Any,
    device: torch.device,
) -> Tuple[Image.Image, Mapping[str, torch.Tensor]]:
    question = args.prompt_template.format(
        subject=record.subject,
        reference=record.reference,
    )
    image = Image.open(record.image_path).convert("RGB")
    batch = helper.build_batch(
        probe=probe,
        processor=processor,
        question=question,
        image=image,
        device=device,
    )
    return image, batch


def should_patch(row: Mapping[str, Any], gate: str) -> bool:
    if gate == "none":
        return False
    if gate == "all":
        return True
    if gate == "final_oracle":
        return not bool(row["firststep_correct"])
    if gate == "transition_oracle":
        return bool(row["r_correct"]) and (not bool(row["y_correct"]))
    raise ValueError(f"Unknown gate {gate}")


def relation_for_mode(row: Mapping[str, Any], selector: str) -> str:
    if selector == "gt":
        return str(row["gt"])
    if selector == "r_pred":
        value = str(row["r_pred"])
        if value not in RELATIONS:
            raise RuntimeError(f"Bad r_pred: {value}")
        return value
    if selector == "gt_shuffled":
        return SHUFFLE_RELATION[str(row["gt"])]
    raise ValueError(f"Unknown relation selector {selector}")


def random_unit_vectors(hidden_dim: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for relation in RELATIONS:
        v = rng.standard_normal(hidden_dim).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), EPS)
        out[relation] = v
    return out


def build_patch_context(
    *,
    row: Mapping[str, Any],
    mode: str,
    mlp_module: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    directions_by_mode: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selected_lambdas: Mapping[str, float],
    random_dirs: Mapping[str, np.ndarray],
) -> Tuple[contextlib.AbstractContextManager, Dict[str, Any]]:
    """Return context manager + patch metadata for one evaluation mode."""
    mode_specs = {
        "clean": (None, "none", "gt", "learned"),
        "final_oracle_gt": ("final", "final_oracle", "gt", "learned"),
        "final_oracle_predrel": ("final", "final_oracle", "r_pred", "learned"),
        "final_all_predrel": ("final", "all", "r_pred", "learned"),
        "final_oracle_gt_shuffled_relation": (
            "final", "final_oracle", "gt_shuffled", "learned"
        ),
        "final_oracle_gt_random_direction": (
            "final", "final_oracle", "gt", "random"
        ),
        "transition_oracle_gt": (
            "transition", "transition_oracle", "gt", "learned"
        ),
        "transition_oracle_predrel": (
            "transition", "transition_oracle", "r_pred", "learned"
        ),
    }
    if mode not in mode_specs:
        raise ValueError(f"Unknown evaluation mode {mode}")

    fit_mode, gate, relation_selector, patch_kind = mode_specs[mode]
    if fit_mode is None or not should_patch(row, gate):
        return contextlib.nullcontext(), {
            "patched": False,
            "fit_mode": fit_mode,
            "gate": gate,
            "relation_selector": relation_selector,
            "selected_relation": None,
            "lambda": 0.0,
            "patch_kind": patch_kind,
        }

    if fit_mode not in directions_by_mode:
        return contextlib.nullcontext(), {
            "patched": False,
            "fit_mode": fit_mode,
            "gate": gate,
            "relation_selector": relation_selector,
            "selected_relation": None,
            "lambda": 0.0,
            "patch_kind": patch_kind,
            "missing_fit_mode": True,
        }

    relation = relation_for_mode(row, relation_selector)
    spec = directions_by_mode[fit_mode][relation]
    true_u = np.asarray(spec["direction"], dtype=np.float32)
    patch_u = true_u if patch_kind == "learned" else np.asarray(
        random_dirs[relation], dtype=np.float32
    )
    lambda_value = float(selected_lambdas[fit_mode])

    prompt_length = int(batch["input_ids"].shape[1])
    prompt_last = prompt_length - 1
    manager = PromptLastProjectionRepairHook(
        module=mlp_module,
        prompt_length=prompt_length,
        prompt_last=prompt_last,
        true_direction=true_u,
        patch_direction=patch_u,
        target_projection=float(spec["target_projection"]),
        lambda_value=lambda_value,
        label=mode,
    )
    return manager, {
        "patched": True,
        "fit_mode": fit_mode,
        "gate": gate,
        "relation_selector": relation_selector,
        "selected_relation": relation,
        "lambda": lambda_value,
        "patch_kind": patch_kind,
    }


@torch.inference_mode()
def forward_firststep(
    *,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    relation_token_map: Mapping[str, Sequence[int]],
    gt: str,
    manager: contextlib.AbstractContextManager,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    patch_stats: Dict[str, Any] = {}
    with manager as active:
        outputs = model(**batch, use_cache=False, return_dict=True)
        vocab_logits = outputs.logits[0, -1]
        scores = relation_scores_from_vocab_logits(vocab_logits, relation_token_map)
        metrics = relation_metrics(scores, gt)

        if isinstance(active, PromptLastProjectionRepairHook):
            active.validate(expected=1)
            patch_stats = {
                "natural_projection": active.last_natural_projection,
                "correction_scalar": active.last_scalar,
                "delta_norm": active.last_delta_norm,
            }
        del outputs
    return metrics, patch_stats


@torch.inference_mode()
def greedy_generation(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, torch.Tensor],
    helper: Any,
    max_new_tokens: int,
    manager: contextlib.AbstractContextManager,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    prompt_length = int(batch["input_ids"].shape[1])
    patch_stats: Dict[str, Any] = {}

    with manager as active:
        generated = model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )
        if isinstance(active, PromptLastProjectionRepairHook):
            active.validate(expected=1)
            patch_stats = {
                "natural_projection": active.last_natural_projection,
                "correction_scalar": active.last_scalar,
                "delta_norm": active.last_delta_norm,
            }

    text = processor.tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    pred = helper.normalize_relation(text)
    del generated
    return pred, text, patch_stats


# =============================================================================
# Validation / Test summaries
# =============================================================================

def tune_lambda(
    *,
    fit_mode: str,
    val_rows: Sequence[Mapping[str, Any]],
    record_by_sid: Mapping[int, Any],
    args: argparse.Namespace,
    helper: Any,
    probe: Any,
    processor: Any,
    device: torch.device,
    model: Any,
    mlp_module: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
    directions_by_mode: Mapping[str, Mapping[str, Mapping[str, Any]]],
    lambda_grid: Sequence[float],
    random_dirs: Mapping[str, np.ndarray],
    errors_path: Path,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Tune with oracle gate + GT relation on VAL, using final first-step accuracy."""
    mode = "final_oracle_gt" if fit_mode == "final" else "transition_oracle_gt"
    sweep_rows: List[Dict[str, Any]] = []

    for lambda_value in lambda_grid:
        selected_lambdas = {fit_mode: float(lambda_value)}
        sample_metrics = []

        for row in tqdm(
            val_rows,
            desc=f"VAL {fit_mode} lambda={lambda_value:g}",
            leave=False,
        ):
            sid = int(row["sid"])
            image = batch = None
            try:
                image, batch = build_batch_for_record(
                    record=record_by_sid[sid],
                    args=args,
                    helper=helper,
                    probe=probe,
                    processor=processor,
                    device=device,
                )
                manager, patch_meta = build_patch_context(
                    row=row,
                    mode=mode,
                    mlp_module=mlp_module,
                    batch=batch,
                    directions_by_mode=directions_by_mode,
                    selected_lambdas=selected_lambdas,
                    random_dirs=random_dirs,
                )
                metrics, patch_stats = forward_firststep(
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    gt=str(row["gt"]),
                    manager=manager,
                )
                sample_metrics.append({
                    "correct": bool(metrics["correct"]),
                    "margin": float(metrics["decision_margin"]),
                    "patched": bool(patch_meta["patched"]),
                    "baseline_correct": bool(row["firststep_correct"]),
                    "delta_norm": float(patch_stats.get("delta_norm", 0.0)),
                })
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "validation",
                    "fit_mode": fit_mode,
                    "lambda": float(lambda_value),
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                if batch is not None:
                    del batch

        if not sample_metrics:
            continue
        acc = safe_mean(float(x["correct"]) for x in sample_metrics)
        base_acc = safe_mean(float(x["baseline_correct"]) for x in sample_metrics)
        mean_margin = safe_mean(x["margin"] for x in sample_metrics)
        patch_fraction = safe_mean(float(x["patched"]) for x in sample_metrics)
        mean_delta_norm = safe_mean(
            x["delta_norm"] for x in sample_metrics if x["patched"]
        )
        w_to_c = sum(
            (not x["baseline_correct"]) and x["correct"] for x in sample_metrics
        )
        c_to_w = sum(
            x["baseline_correct"] and (not x["correct"]) for x in sample_metrics
        )
        sweep_rows.append({
            "fit_mode": fit_mode,
            "lambda": float(lambda_value),
            "N": len(sample_metrics),
            "baseline_acc": base_acc,
            "patched_acc": acc,
            "delta_acc": acc - base_acc,
            "mean_decision_margin": mean_margin,
            "patch_fraction": patch_fraction,
            "mean_delta_norm_on_patched": mean_delta_norm,
            "wrong_to_correct": w_to_c,
            "correct_to_wrong": c_to_w,
            "net_repairs": w_to_c - c_to_w,
        })

    if not sweep_rows:
        raise RuntimeError(f"No validation results for {fit_mode}")

    # Primary: max ACC. Secondary: max margin. Tertiary: smaller |lambda|.
    best = max(
        sweep_rows,
        key=lambda r: (
            float(r["patched_acc"]),
            float(r["mean_decision_margin"]),
            -abs(float(r["lambda"])),
        ),
    )
    return float(best["lambda"]), sweep_rows


def summarize_mode_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)

    out = []
    mode_order = [
        "clean",
        "final_oracle_gt",
        "final_oracle_predrel",
        "final_all_predrel",
        "final_oracle_gt_shuffled_relation",
        "final_oracle_gt_random_direction",
        "transition_oracle_gt",
        "transition_oracle_predrel",
    ]
    order = {m: i for i, m in enumerate(mode_order)}

    clean_by_sid = {
        int(r["sid"]): bool(r["correct"])
        for r in grouped.get("clean", [])
    }

    for mode, rs in grouped.items():
        acc = safe_mean(float(r["correct"]) for r in rs)
        patched_fraction = safe_mean(float(r.get("patched", False)) for r in rs)
        delta_norm = safe_mean(
            r.get("delta_norm", float("nan"))
            for r in rs
            if bool(r.get("patched", False))
        )
        w_to_c = 0
        c_to_w = 0
        matched = 0
        for r in rs:
            sid = int(r["sid"])
            if sid not in clean_by_sid:
                continue
            matched += 1
            clean_ok = clean_by_sid[sid]
            now_ok = bool(r["correct"])
            w_to_c += int((not clean_ok) and now_ok)
            c_to_w += int(clean_ok and (not now_ok))

        out.append({
            "mode": mode,
            "N": len(rs),
            "accuracy": acc,
            "patch_fraction": patched_fraction,
            "mean_delta_norm_on_patched": delta_norm,
            "wrong_to_correct": w_to_c,
            "correct_to_wrong": c_to_w,
            "net_repairs": w_to_c - c_to_w,
            "matched_to_clean": matched,
        })

    out.sort(key=lambda r: order.get(str(r["mode"]), 99))
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.model != "qwen-7b":
        raise ValueError("v1 intentionally supports --model qwen-7b only.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    fit_modes = parse_names(args.fit_modes, ("final", "transition"))
    generation_modes = parse_names(
        args.generation_modes,
        (
            "clean",
            "final_oracle_gt",
            "final_oracle_predrel",
            "final_all_predrel",
            "final_oracle_gt_shuffled_relation",
            "final_oracle_gt_random_direction",
            "transition_oracle_gt",
            "transition_oracle_predrel",
        ),
    )
    lambda_grid = parse_float_grid(args.lambda_grid)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    helper = importlib.import_module(args.helper_module)
    probe = importlib.import_module(args.probe_module)
    ah = importlib.import_module(args.attention_helper_module)
    base = probe.base

    records, audit = base.load_records(args.dataset, Path(args.data_root), None)
    selected_records = helper.select_records(
        records,
        n=args.num_samples,
        seed=args.seed,
    )
    record_by_sid = {int(r.sid): r for r in selected_records}

    spec = base.SPECS[args.model]
    model_class = getattr(transformers, spec.model_class)
    model = None
    processor = None

    try:
        print(f"Loading {args.model}: {spec.repo_id}", flush=True)
        model = model_class.from_pretrained(
            spec.repo_id,
            dtype=base.resolve_dtype(spec.dtype_name),
            low_cpu_mem_usage=True,
            trust_remote_code=spec.trust_remote_code,
            device_map={"": args.device},
            attn_implementation=args.attn_impl,
        )
        model.eval()
        helper.clear_sampling_defaults(model)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        decoder_layers, decoder_path = probe.resolve_decoder_layers(model)
        if len(decoder_layers) != 28:
            raise RuntimeError(
                f"Expected Qwen-7B 28 decoder layers; got {len(decoder_layers)} "
                f"at {decoder_path}"
            )
        layer = int(args.layer)
        if not (1 <= layer < len(decoder_layers)):
            raise ValueError(f"Bad layer L{layer}")
        if layer != 23:
            print(f"[warning] experiment motivated by L23, running L{layer}.", flush=True)

        block = decoder_layers[layer]
        mlp_module = getattr(block, "mlp", None)
        if not isinstance(mlp_module, torch.nn.Module):
            raise RuntimeError(f"Could not resolve L{layer} block.mlp")

        final_norm, final_norm_path = helper.resolve_final_norm(model, decoder_path)
        if final_norm is None:
            raise RuntimeError("Could not resolve final norm.")
        token_map = helper.relation_token_variants(processor.tokenizer)
        lens = helper.RelationLogitLens(
            model=model,
            final_norm=final_norm,
            token_map=token_map,
        )
        relation_token_map = token_map

        print("\n" + "=" * 190)
        print("QWEN-7B L23 MLP GOOD-vs-BAD DIRECTION REPAIR")
        print("=" * 190)
        print("N selected      :", len(selected_records))
        print("layer           :", layer)
        print("fit modes       :", fit_modes)
        print("split           :", args.train_ratio, args.val_ratio, 1-args.train_ratio-args.val_ratio)
        print("lambda grid     :", lambda_grid)
        print("final norm      :", final_norm_path)
        print("run generation  :", args.run_generation)
        print("=" * 190, flush=True)

        # ---------------------------------------------------------------------
        # Phase 1: natural trace on ALL selected samples.
        # ---------------------------------------------------------------------
        natural_rows: List[Dict[str, Any]] = []
        vectors_by_sid: Dict[int, Dict[str, np.ndarray]] = {}

        for index, record in enumerate(
            tqdm(selected_records, desc="natural-L23-trace"), start=1
        ):
            image = batch = None
            try:
                sid = int(record.sid)
                gt = helper.normalize_relation(record.relation)
                if gt not in RELATIONS:
                    raise RuntimeError(f"Unsupported GT: {record.relation!r}")

                image, batch = build_batch_for_record(
                    record=record,
                    args=args,
                    helper=helper,
                    probe=probe,
                    processor=processor,
                    device=device,
                )
                meta, vectors = run_trace_l23(
                    ah=ah,
                    helper=helper,
                    model=model,
                    batch=batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    layer=layer,
                    lens=lens,
                    gt=gt,
                )
                natural_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "subject": record.subject,
                    "reference": record.reference,
                    **meta,
                })
                vectors_by_sid[sid] = vectors
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "natural_trace",
                    "sid": int(getattr(record, "sid", -1)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                if batch is not None:
                    del batch
                gc.collect()
                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        if len(natural_rows) < 8:
            raise RuntimeError(f"Too few natural rows: {len(natural_rows)}")

        write_csv(output_dir / "natural_l23_states.csv", natural_rows)

        split = stratified_split(
            natural_rows,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        (output_dir / "split_sids.json").write_text(
            json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        row_by_sid = {int(r["sid"]): r for r in natural_rows}
        train_rows = [row_by_sid[sid] for sid in split["train"] if sid in row_by_sid]
        val_rows = [row_by_sid[sid] for sid in split["val"] if sid in row_by_sid]
        test_rows = [row_by_sid[sid] for sid in split["test"] if sid in row_by_sid]

        # ---------------------------------------------------------------------
        # Phase 2: fit train-only directions.
        # ---------------------------------------------------------------------
        directions_by_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}
        diagnostics: List[Dict[str, Any]] = []

        for fit_mode in fit_modes:
            directions, diag = fit_directions(
                fit_mode=fit_mode,
                train_rows=train_rows,
                vectors_by_sid=vectors_by_sid,
                min_good=args.min_good_per_relation,
                min_bad=args.min_bad_per_relation,
            )
            directions_by_mode[fit_mode] = directions
            diagnostics.extend(diag)

        write_csv(output_dir / "direction_diagnostics.csv", diagnostics)
        save_directions_npz(output_dir / "learned_directions.npz", directions_by_mode)

        hidden_dim = int(next(iter(vectors_by_sid.values()))["mlp"].shape[0])
        random_dirs = random_unit_vectors(hidden_dim, seed=args.seed + 99173)

        print("\nTRAIN DIRECTION DIAGNOSTICS")
        for row in diagnostics:
            if row["relation"] == "__global__" or not row.get("used_fallback", False):
                print(
                    f"  {row['fit_mode']:<10s} {row['relation']:<10s} "
                    f"Ng={int(row['n_good']):3d} Nb={int(row['n_bad']):3d} "
                    f"cos(muG,muB)={float(row['cos_mu_good_bad']):+.4f} "
                    f"||m|| G/B={float(row['mean_norm_good']):.3f}/"
                    f"{float(row['mean_norm_bad']):.3f} "
                    f"projGap={float(row['projection_gap']):+.4f}"
                )

        # ---------------------------------------------------------------------
        # Phase 3: VAL chooses one lambda per fit mode.
        # ---------------------------------------------------------------------
        selected_lambdas: Dict[str, float] = {}
        validation_rows: List[Dict[str, Any]] = []

        for fit_mode in fit_modes:
            best_lambda, sweep = tune_lambda(
                fit_mode=fit_mode,
                val_rows=val_rows,
                record_by_sid=record_by_sid,
                args=args,
                helper=helper,
                probe=probe,
                processor=processor,
                device=device,
                model=model,
                mlp_module=mlp_module,
                relation_token_map=relation_token_map,
                directions_by_mode=directions_by_mode,
                lambda_grid=lambda_grid,
                random_dirs=random_dirs,
                errors_path=errors_path,
            )
            selected_lambdas[fit_mode] = best_lambda
            validation_rows.extend(sweep)
            print(f"VAL selected lambda [{fit_mode}] = {best_lambda:g}")

        write_csv(output_dir / "validation_lambda_sweep.csv", validation_rows)
        (output_dir / "selected_lambdas.json").write_text(
            json.dumps(selected_lambdas, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ---------------------------------------------------------------------
        # Phase 4: frozen TEST evaluation.
        # ---------------------------------------------------------------------
        test_modes = ["clean"]
        if "final" in fit_modes:
            test_modes += [
                "final_oracle_gt",
                "final_oracle_predrel",
                "final_all_predrel",
                "final_oracle_gt_shuffled_relation",
                "final_oracle_gt_random_direction",
            ]
        if "transition" in fit_modes:
            test_modes += [
                "transition_oracle_gt",
                "transition_oracle_predrel",
            ]

        firststep_test_rows: List[Dict[str, Any]] = []
        generation_test_rows: List[Dict[str, Any]] = []

        for index, row in enumerate(tqdm(test_rows, desc="TEST repair"), start=1):
            sid = int(row["sid"])
            record = record_by_sid[sid]
            gt = str(row["gt"])
            image = batch = None
            try:
                image, batch = build_batch_for_record(
                    record=record,
                    args=args,
                    helper=helper,
                    probe=probe,
                    processor=processor,
                    device=device,
                )

                for mode in test_modes:
                    manager, patch_meta = build_patch_context(
                        row=row,
                        mode=mode,
                        mlp_module=mlp_module,
                        batch=batch,
                        directions_by_mode=directions_by_mode,
                        selected_lambdas=selected_lambdas,
                        random_dirs=random_dirs,
                    )
                    metrics, patch_stats = forward_firststep(
                        model=model,
                        batch=batch,
                        relation_token_map=relation_token_map,
                        gt=gt,
                        manager=manager,
                    )
                    firststep_test_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "mode": mode,
                        "clean_firststep_pred": row["firststep_pred"],
                        "clean_firststep_correct": row["firststep_correct"],
                        "clean_r_pred": row["r_pred"],
                        "clean_r_correct": row["r_correct"],
                        "clean_y_pred": row["y_pred"],
                        "clean_y_correct": row["y_correct"],
                        "prediction": metrics["prediction"],
                        "correct": metrics["correct"],
                        "decision_margin": metrics["decision_margin"],
                        **patch_meta,
                        **patch_stats,
                    })

                if args.run_generation:
                    for mode in generation_modes:
                        # Skip modes whose required fit direction was not fitted.
                        if mode.startswith("final_") and "final" not in fit_modes:
                            continue
                        if mode.startswith("transition_") and "transition" not in fit_modes:
                            continue

                        manager, patch_meta = build_patch_context(
                            row=row,
                            mode=mode,
                            mlp_module=mlp_module,
                            batch=batch,
                            directions_by_mode=directions_by_mode,
                            selected_lambdas=selected_lambdas,
                            random_dirs=random_dirs,
                        )
                        pred, text, patch_stats = greedy_generation(
                            model=model,
                            processor=processor,
                            batch=batch,
                            helper=helper,
                            max_new_tokens=args.max_new_tokens,
                            manager=manager,
                        )
                        generation_test_rows.append({
                            "sid": sid,
                            "gt": gt,
                            "mode": mode,
                            "prediction": pred,
                            "text": text,
                            "correct": pred == gt,
                            **patch_meta,
                            **patch_stats,
                        })
                        append_jsonl(
                            output_dir / "test_generation_results.jsonl",
                            generation_test_rows[-1],
                        )

            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "test",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                if batch is not None:
                    del batch
                gc.collect()
                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(output_dir / "test_firststep_results.csv", firststep_test_rows)
        firststep_summary = summarize_mode_rows(firststep_test_rows)
        write_csv(output_dir / "test_firststep_summary.csv", firststep_summary)

        generation_summary: List[Dict[str, Any]] = []
        if generation_test_rows:
            generation_summary = summarize_mode_rows(generation_test_rows)
            write_csv(output_dir / "test_generation_summary.csv", generation_summary)

        # ---------------------------------------------------------------------
        # Console/report.
        # ---------------------------------------------------------------------
        print("\n" + "=" * 190)
        print("HELD-OUT TEST FIRST-STEP")
        print("=" * 190)
        clean_acc = next(
            (float(r["accuracy"]) for r in firststep_summary if r["mode"] == "clean"),
            float("nan"),
        )
        print(f"clean TEST first-step ACC = {100*clean_acc:.2f}%")
        print(
            f"  {'mode':<40s} {'ACC':>8s} {'delta':>9s} {'patch%':>8s} "
            f"{'W->C':>6s} {'C->W':>6s} {'net':>6s} {'dNorm':>9s}"
        )
        for r in firststep_summary:
            acc = float(r["accuracy"])
            print(
                f"  {str(r['mode']):<40s} {100*acc:7.2f}% "
                f"{100*(acc-clean_acc):+8.2f} "
                f"{100*float(r['patch_fraction']):7.2f}% "
                f"{int(r['wrong_to_correct']):6d} "
                f"{int(r['correct_to_wrong']):6d} "
                f"{int(r['net_repairs']):+6d} "
                f"{float(r['mean_delta_norm_on_patched']):9.3f}"
            )

        if generation_summary:
            print("\nHELD-OUT TEST FULL GENERATION")
            clean_gen = next(
                (float(r["accuracy"]) for r in generation_summary if r["mode"] == "clean"),
                float("nan"),
            )
            print(f"clean TEST generation ACC = {100*clean_gen:.2f}%")
            print(
                f"  {'mode':<40s} {'ACC':>8s} {'delta':>9s} {'patch%':>8s} "
                f"{'W->C':>6s} {'C->W':>6s} {'net':>6s}"
            )
            for r in generation_summary:
                acc = float(r["accuracy"])
                print(
                    f"  {str(r['mode']):<40s} {100*acc:7.2f}% "
                    f"{100*(acc-clean_gen):+8.2f} "
                    f"{100*float(r['patch_fraction']):7.2f}% "
                    f"{int(r['wrong_to_correct']):6d} "
                    f"{int(r['correct_to_wrong']):6d} "
                    f"{int(r['net_repairs']):+6d}"
                )
        print("=" * 190)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "N_natural": len(natural_rows),
            "layer": layer,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "split_sizes": {k: len(v) for k, v in split.items()},
            "fit_modes": fit_modes,
            "lambda_grid": lambda_grid,
            "selected_lambdas": selected_lambdas,
            "direction_formula": "u_r = normalize(mean(m|GOOD,r)-mean(m|BAD,r))",
            "repair_formula": "m' = m + lambda*(target_good_proj - <m,u_r>)*u_r",
            "validation_selection": "max held-out VAL final first-step ACC under oracle gate+GT relation; then margin; then smaller |lambda|",
            "primary_warning": (
                "Oracle modes use GT for failure gating and/or relation selection. "
                "This v1 tests whether a TRAIN-learned update direction transfers; "
                "it is not yet deployable GT-free repair."
            ),
            "final_norm_path": final_norm_path,
            "audit": audit,
        }
        (output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        report = [
            f"script_version: {SCRIPT_VERSION}",
            f"N={len(natural_rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}",
            f"selected_lambdas={selected_lambdas}",
            "",
            "TRAIN DIRECTION DIAGNOSTICS",
        ]
        for r in diagnostics:
            report.append(
                f"{r['fit_mode']} {r['relation']} fallback={r.get('used_fallback', False)} "
                f"Ng={int(r['n_good'])} Nb={int(r['n_bad'])} "
                f"cosMu={float(r['cos_mu_good_bad']):+.5f} "
                f"normG={float(r['mean_norm_good']):.5f} "
                f"normB={float(r['mean_norm_bad']):.5f} "
                f"projGap={float(r['projection_gap']):+.5f}"
            )
        report += ["", "TEST FIRST-STEP"]
        for r in firststep_summary:
            acc = float(r["accuracy"])
            report.append(
                f"{r['mode']}: acc={100*acc:.2f}% delta={100*(acc-clean_acc):+.2f}pp "
                f"W->C={int(r['wrong_to_correct'])} C->W={int(r['correct_to_wrong'])} "
                f"net={int(r['net_repairs']):+d} patch={100*float(r['patch_fraction']):.2f}%"
            )
        if generation_summary:
            report += ["", "TEST GENERATION"]
            clean_gen = next(
                (float(r["accuracy"]) for r in generation_summary if r["mode"] == "clean"),
                float("nan"),
            )
            for r in generation_summary:
                acc = float(r["accuracy"])
                report.append(
                    f"{r['mode']}: acc={100*acc:.2f}% delta={100*(acc-clean_gen):+.2f}pp "
                    f"W->C={int(r['wrong_to_correct'])} C->W={int(r['correct_to_wrong'])} "
                    f"net={int(r['net_repairs']):+d} patch={100*float(r['patch_fraction']):.2f}%"
                )
        report += [
            "",
            "READING THE RESULT",
            "1) If final/transition learned direction beats clean on held-out TEST, the Correct-vs-Wrong update difference transfers beyond the TRAIN samples.",
            "2) If shuffled/random controls do not improve, the effect is not explained by generic perturbation norm.",
            "3) If oracle_gt works but oracle_predrel collapses, relation selection is a bottleneck.",
            "4) If oracle modes work but final_all_predrel harms clean samples, a GT-free failure gate is necessary.",
            "5) If all held-out direction repairs fail while the earlier scalar oracle helped, the failure is more sample-specific than a single relation-conditioned mean direction.",
        ]
        (output_dir / "report.txt").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )

        print("\nSaved:")
        for name in (
            "natural_l23_states.csv",
            "split_sids.json",
            "direction_diagnostics.csv",
            "learned_directions.npz",
            "validation_lambda_sweep.csv",
            "selected_lambdas.json",
            "test_firststep_results.csv",
            "test_firststep_summary.csv",
            "test_generation_results.jsonl",
            "test_generation_summary.csv",
            "config.json",
            "report.txt",
            "errors.jsonl",
        ):
            path = output_dir / name
            if path.exists():
                print(" ", path)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
