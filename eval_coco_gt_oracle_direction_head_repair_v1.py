#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GT-oracle causal repair of high-accuracy direction heads.

Purpose
=======
This is intentionally NOT a deployable ACC-improvement method.  It is an oracle
diagnostic answering only:

    If the correct relation is known, can high direction-ACC attention heads
    causally steer an originally WRONG generation to the GT relation?

If not, these heads are decodable relation carriers/passengers rather than useful
causal actuators for repair, and there is no reason yet to connect an attention-
centroid detector to them.

Representation
==============
Reuse outputs of:

    analyze_coco_head_object_residual_direction_probe_v1.py

Required direction directory:
    head_results.csv
    relation_vectors.npz

That probe defines, for each layer/head:

    d_img = z_img(A) - z_img(B)                      [pre-W_O head space]
    d_no  = z_noimg(A) - z_noimg(B)
    d_res = d_img - d_no

where z is one query-head slice of the input to attention.o_proj.

This script supports:
    --vector-mode residual   (default)
    --vector-mode img

For a selected head h, TRAIN data give class means mu_h[c].  On a baseline-WRONG
eval sample with generation source=s and GT target=t:

    u_h = normalize(mu_h[t] - mu_h[s])

At runtime, for the current image forward:

    if vector_mode == img:
        d_current = z_A - z_B

    if vector_mode == residual:
        d_current = (z_A - z_B) - cached_d_noimg

Pull only the source->target coordinate toward the TRAIN target prototype:

    delta_h =
        alpha * ( <mu_h[t],u_h> - <d_current,u_h> ) * u_h

and modify only the selected pre-W_O head slice at the two object spans:

    z_A <- z_A + 0.5 * delta_h
    z_B <- z_B - 0.5 * delta_h

The attention output projection W_O, all later blocks, receiver heads, prompt-last,
and LM head then run naturally.

Crucially:
    * GT chooses target ONLY because this is an oracle actuator test.
    * Only baseline-WRONG eval samples are patched.
    * Baseline-correct samples are untouched.
    * Head prototypes are fit on a held-out TRAIN split.
    * Head ranking can come from existing head_results.csv or explicit --heads.
    * Cumulative Top1/Top3/Top5/Top10 bundles are tested.

The script uses the SAME prompt template as the original direction-probe script by
default, so cached pre-W_O vectors and online patch states are prompt-compatible.

Example: residual vectors
=========================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_coco_gt_oracle_direction_head_repair_v1.py \
  --model qwen-3b \
  --direction-dir output/qwen3b_coco_head_object_residual_direction_probe \
  --vector-mode residual \
  --bundle-sizes 1,3,5,10 \
  --alphas 0.25,0.5,1.0 \
  --train-ratio 0.15 \
  --device cuda:0 \
  --output-dir output/qwen3b_gt_oracle_direction_repair_residual_v1 \
  --overwrite

Then compare raw-image head vectors:
====================================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_coco_gt_oracle_direction_head_repair_v1.py \
  --model qwen-3b \
  --direction-dir output/qwen3b_coco_head_object_residual_direction_probe \
  --vector-mode img \
  --bundle-sizes 1,3,5,10 \
  --alphas 0.25,0.5,1.0 \
  --train-ratio 0.15 \
  --device cuda:0 \
  --output-dir output/qwen3b_gt_oracle_direction_repair_img_v1 \
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
import re
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

SCRIPT_VERSION = "coco-gt-oracle-direction-head-repair-v1"
RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-12

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager", "sdpa", "flash_attention_2", "none"),
    )
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
        help="Existing direction-probe module in AdaptVis root/PYTHONPATH.",
    )
    p.add_argument(
        "--direction-dir",
        required=True,
        help="Directory containing head_results.csv and relation_vectors.npz.",
    )
    p.add_argument(
        "--vector-mode",
        default="residual",
        choices=("residual", "img"),
        help="Which direction vector/prototype space to use.",
    )
    p.add_argument(
        "--rank-metric",
        default="auto",
        help="auto -> residual_accuracy_mean or img_accuracy_mean according to vector-mode.",
    )
    p.add_argument(
        "--heads",
        default="",
        help="Optional explicit comma list, e.g. 23:5,23:1,19:13. Overrides ranking.",
    )
    p.add_argument(
        "--bundle-sizes",
        default="1,3,5,10",
        help="Cumulative Top-K bundle sizes.",
    )
    p.add_argument(
        "--alphas",
        default="0.25,0.5,1.0",
        help="Pull fraction toward target prototype coordinate.",
    )
    p.add_argument(
        "--pool",
        default="mean",
        choices=("mean", "last"),
        help="Must match original direction probe cache.",
    )
    p.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT,
        help="Must match prompt used to build relation_vectors.npz.",
    )
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all cache samples before train/eval split.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="0 = all held-out eval samples.",
    )
    p.add_argument(
        "--only-opposite-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, patch only errors where baseline prediction is GT opposite.",
    )
    p.add_argument(
        "--include-random-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Matched same-layer random-head bundle, using that head's own prototypes.",
    )
    p.add_argument("--random-seed", type=int, default=991)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--empty-cache-every", type=int, default=10)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================

def parse_int_list(text: str) -> List[int]:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v <= 0:
            raise ValueError("bundle sizes must be positive")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("No bundle sizes")
    return out


def parse_float_list(text: str) -> List[float]:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if v < 0:
            raise ValueError("alpha must be non-negative")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("No alphas")
    return out


def parse_head(text: str) -> Tuple[int, int]:
    s = str(text).strip().upper()
    s = s.replace("L", "").replace("H", ":")
    while "::" in s:
        s = s.replace("::", ":")
    if ":" not in s:
        raise ValueError(f"Bad head spec: {text!r}")
    a, b = s.split(":", 1)
    return int(a), int(b)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out = []
    seen = set()
    for item in str(text).split(","):
        if not item.strip():
            continue
        h = parse_head(item)
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def hname(h: Tuple[int, int]) -> str:
    return f"L{int(h[0])}H{int(h[1]):02d}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "left": "left",
        "leftward": "left",
        "right": "right",
        "rightward": "right",
        "above": "above",
        "over": "above",
        "on top": "above",
        "top": "above",
        "on": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "bottom": "below",
    }
    if text in aliases:
        return aliases[text]
    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(below|under|underneath|beneath|bottom)\b", "below"),
        (r"\b(above|over|on top|top)\b", "above"),
        (r"\bon\b", "above"),
    ]
    hits = []
    for pat, rel in patterns:
        m = re.search(pat, text)
        if m:
            hits.append((m.start(), rel))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def stratified_limit_indices(
    indices: Sequence[int],
    labels: np.ndarray,
    limit: int,
    seed: int,
) -> List[int]:
    ids = list(map(int, indices))
    if limit <= 0 or limit >= len(ids):
        return sorted(ids)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i in ids:
        groups[str(labels[i])].append(i)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    order = [r for r in RELATIONS if r in groups]
    ptr = {r: 0 for r in order}
    out: List[int] = []
    while len(out) < limit:
        moved = False
        for r in order:
            if len(out) >= limit:
                break
            j = ptr[r]
            if j < len(groups[r]):
                out.append(groups[r][j])
                ptr[r] += 1
                moved = True
        if not moved:
            break
    return sorted(out)


def stratified_split(
    labels: np.ndarray,
    ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not 0 < ratio < 1:
        raise ValueError("--train-ratio must be in (0,1)")
    rng = random.Random(seed)
    train, eval_ = [], []
    for rel in RELATIONS:
        ids = [i for i, y in enumerate(labels) if str(y) == rel]
        rng.shuffle(ids)
        if len(ids) < 2:
            raise RuntimeError(f"Not enough samples for {rel}")
        n = max(1, int(round(len(ids) * ratio)))
        n = min(n, len(ids) - 1)
        train.extend(ids[:n])
        eval_.extend(ids[n:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def clear_sampling_defaults(model: Any) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    for field in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, field):
            setattr(cfg, field, None)


# =============================================================================
# Cache / head selection
# =============================================================================

def load_direction_cache(direction_dir: Path) -> Dict[str, Any]:
    npz_path = direction_dir / "relation_vectors.npz"
    csv_path = direction_dir / "head_results.csv"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with np.load(npz_path, allow_pickle=True) as z:
        data = {k: np.asarray(z[k]) for k in z.files}

    required = {"sample_index", "relation", "decoder_block_index", "head_index"}
    missing = required - set(data)
    if missing:
        raise RuntimeError(f"direction cache missing keys: {sorted(missing)}")
    return {
        "arrays": data,
        "head_rows": read_csv(csv_path),
        "npz_path": str(npz_path),
        "csv_path": str(csv_path),
    }


def select_ranked_heads(
    *,
    rows: Sequence[Mapping[str, str]],
    metric: str,
    explicit: Sequence[Tuple[int, int]],
    max_k: int,
) -> List[Tuple[int, int]]:
    if explicit:
        if len(explicit) < max_k:
            raise ValueError(
                f"Explicit --heads has {len(explicit)} heads but max bundle K={max_k}"
            )
        return list(explicit[:max_k])

    vals = []
    for row in rows:
        try:
            score = float(row[metric])
            layer = int(row["layer"])
            head = int(row["head"])
        except Exception:
            continue
        if np.isfinite(score):
            vals.append((score, layer, head))
    if len(vals) < max_k:
        raise RuntimeError(
            f"Only {len(vals)} heads have ranking metric {metric!r}"
        )
    vals.sort(key=lambda x: x[0], reverse=True)
    return [(l, h) for _, l, h in vals[:max_k]]


def matched_random_heads(
    *,
    selected: Sequence[Tuple[int, int]],
    n_heads: int,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    excluded = set(selected)
    used = set()
    out = []
    for layer, _head in selected:
        candidates = [
            (layer, h)
            for h in range(n_heads)
            if (layer, h) not in excluded and (layer, h) not in used
        ]
        if not candidates:
            # Allow reuse only as last resort within same layer.
            candidates = [
                (layer, h)
                for h in range(n_heads)
                if (layer, h) not in excluded
            ]
        if not candidates:
            raise RuntimeError(f"No random control candidates at L{layer}")
        choice = rng.choice(candidates)
        out.append(choice)
        used.add(choice)
    return out


def fit_class_means(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: Sequence[int],
    heads: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    means: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    tr = np.asarray(train_idx, dtype=np.int64)
    for layer, head in heads:
        means[(layer, head)] = {}
        for rel in RELATIONS:
            mask = labels[tr] == rel
            if not np.any(mask):
                raise RuntimeError(f"No TRAIN examples for {rel}")
            xs = np.asarray(X[tr[mask], layer, head, :], dtype=np.float32)
            means[(layer, head)][rel] = xs.mean(axis=0).astype(np.float32)
    return means


def heldout_head_accuracy(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
    heads: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    rows = []
    tr = np.asarray(train_idx, dtype=np.int64)
    te = np.asarray(eval_idx, dtype=np.int64)
    for layer, head in heads:
        Xt = np.asarray(X[tr, layer, head, :], dtype=np.float32)
        center = Xt.mean(axis=0)
        dirs = []
        for rel in RELATIONS:
            m = labels[tr] == rel
            d = (Xt[m] - center).mean(axis=0)
            d = d / max(float(np.linalg.norm(d)), EPS)
            dirs.append(d)
        dirs = np.stack(dirs, axis=0)
        Xe = np.asarray(X[te, layer, head, :], dtype=np.float32) - center
        Xe = Xe / np.maximum(
            np.linalg.norm(Xe, axis=-1, keepdims=True), EPS
        )
        pred = np.argmax(Xe @ dirs.T, axis=1)
        gt = np.asarray([RID[str(labels[i])] for i in te])
        rows.append({
            "head": hname((layer, head)),
            "layer": layer,
            "head_index": head,
            "N_eval": len(te),
            "heldout_accuracy": float(np.mean(pred == gt)),
        })
    return rows


# =============================================================================
# Online patching
# =============================================================================

def batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def build_batch(
    *,
    probe: Any,
    processor: Any,
    question: str,
    image: Any,
    device: torch.device,
) -> Dict[str, Any]:
    rendered = probe.build_chat_prompt(processor, question, True)
    return probe.process_inputs(processor, rendered, image, device)


def relation_scores_from_logits(
    logits_last: torch.Tensor,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Tuple[str, np.ndarray]:
    vals = []
    for rel in RELATIONS:
        ids = [int(x) for x in relation_token_map[rel]]
        vals.append(
            float(logits_last[ids].detach().float().max().item())
        )
    arr = np.asarray(vals, dtype=np.float32)
    return RELATIONS[int(np.argmax(arr))], arr


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
    out = {}
    for rel in RELATIONS:
        ids = set()
        for text in (rel, " " + rel, rel.capitalize(), " " + rel.capitalize()):
            tok = tokenizer.encode(text, add_special_tokens=False)
            if len(tok) == 1:
                ids.add(int(tok[0]))
        if not ids:
            tok = tokenizer.encode(" " + rel, add_special_tokens=False)
            if not tok:
                raise RuntimeError(f"No token ids for relation {rel}")
            ids.add(int(tok[-1]))
        out[rel] = sorted(ids)
    return out


class DirectionHeadOraclePatch:
    """
    Modifies selected query-head slices of attention.o_proj INPUT at A/B positions.

    For residual mode:
        current = (zA-zB) - cached_noimage_direction

    For img mode:
        current = zA-zB

    delta = alpha * (target_coord-current_coord) * source_to_target_axis
    """

    def __init__(
        self,
        *,
        probe: Any,
        decoder_layers: Sequence[Any],
        heads: Sequence[Tuple[int, int]],
        n_heads: int,
        head_dim: int,
        a_positions: Sequence[int],
        b_positions: Sequence[int],
        pool: str,
        class_means: Mapping[Tuple[int, int], Mapping[str, np.ndarray]],
        source_relation: str,
        target_relation: str,
        alpha: float,
        vector_mode: str,
        noimage_by_head: Optional[Mapping[Tuple[int, int], np.ndarray]],
    ) -> None:
        self.probe = probe
        self.decoder_layers = decoder_layers
        self.heads = list(heads)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.a_positions = tuple(map(int, a_positions))
        self.b_positions = tuple(map(int, b_positions))
        self.pool = str(pool)
        self.class_means = class_means
        self.source_relation = str(source_relation)
        self.target_relation = str(target_relation)
        self.alpha = float(alpha)
        self.vector_mode = str(vector_mode)
        self.noimage_by_head = noimage_by_head or {}
        self.handles = []
        self.applied: Counter = Counter()
        self.meta: Dict[str, Dict[str, float]] = {}

        grouped: Dict[int, List[int]] = defaultdict(list)
        for layer, head in self.heads:
            grouped[int(layer)].append(int(head))

        for layer, local_heads in grouped.items():
            attn = probe.resolve_self_attention(decoder_layers[layer])
            oproj = probe.resolve_o_proj(attn)
            handle = oproj.register_forward_pre_hook(
                self._make_hook(layer, sorted(local_heads))
            )
            self.handles.append(handle)

    def _make_hook(self, layer: int, local_heads: Sequence[int]):
        def hook(_module: Any, inputs: Tuple[Any, ...]):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"L{layer} o_proj pre-hook missing tensor")
            x = inputs[0]
            max_pos = max(self.a_positions + self.b_positions)
            # Skip cached decode steps (sequence length typically 1).
            if x.ndim != 3 or int(x.shape[0]) != 1 or int(x.shape[1]) <= max_pos:
                return None

            y = x.float().clone()
            for head in local_heads:
                start = head * self.head_dim
                stop = start + self.head_dim

                za = self.probe.pool_positions(
                    y[:, :, start:stop],
                    self.a_positions,
                    self.pool,
                ).float()
                zb = self.probe.pool_positions(
                    y[:, :, start:stop],
                    self.b_positions,
                    self.pool,
                ).float()
                d_img = za - zb

                key = (layer, head)
                if self.vector_mode == "residual":
                    if key not in self.noimage_by_head:
                        raise RuntimeError(f"Missing no-image vector for {hname(key)}")
                    d_no = torch.as_tensor(
                        self.noimage_by_head[key],
                        device=y.device,
                        dtype=torch.float32,
                    )
                    current = d_img - d_no
                else:
                    current = d_img

                mu_s = torch.as_tensor(
                    self.class_means[key][self.source_relation],
                    device=y.device,
                    dtype=torch.float32,
                )
                mu_t = torch.as_tensor(
                    self.class_means[key][self.target_relation],
                    device=y.device,
                    dtype=torch.float32,
                )
                axis = mu_t - mu_s
                axis_norm = axis.norm()
                if float(axis_norm.item()) <= EPS:
                    continue
                u = axis / axis_norm
                current_coord = torch.dot(current, u)
                target_coord = torch.dot(mu_t, u)
                scalar = self.alpha * (target_coord - current_coord)
                delta = scalar * u

                # Apply the same head-local vector to each token in the span.
                # Mean pooling then moves by exactly +/- delta/2.
                for pos in self.a_positions:
                    y[:, pos, start:stop] += 0.5 * delta
                for pos in self.b_positions:
                    y[:, pos, start:stop] -= 0.5 * delta

                name = hname(key)
                self.applied[name] += 1
                self.meta[name] = {
                    "current_coord": float(current_coord.item()),
                    "target_coord": float(target_coord.item()),
                    "scalar": float(scalar.item()),
                    "delta_norm": float(delta.norm().item()),
                }

            new_inputs = (y.to(dtype=x.dtype), *inputs[1:])
            return new_inputs
        return hook

    def validate(self) -> None:
        missing = [hname(h) for h in self.heads if self.applied[hname(h)] < 1]
        if missing:
            raise RuntimeError(f"Patch hooks did not fire: {missing}")

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


@torch.inference_mode()
def first_step_with_patch(
    *,
    model: Any,
    batch: Mapping[str, Any],
    patch: DirectionHeadOraclePatch,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Tuple[str, np.ndarray]:
    with patch:
        out = model(
            **batch,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        patch.validate()
        pred, scores = relation_scores_from_logits(
            out.logits[0, -1],
            relation_token_map,
        )
    del out
    return pred, scores


@torch.inference_mode()
def generate_with_patch(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    patch: DirectionHeadOraclePatch,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    input_len = int(batch["input_ids"].shape[1])
    with patch:
        ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        patch.validate()
    text = processor.tokenizer.decode(
        ids[0, input_len:],
        skip_special_tokens=True,
    ).strip()
    del ids
    return normalize_relation(text), text


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    direction_dir = Path(args.direction_dir)
    out = Path(args.output_dir)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output dir is non-empty: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    errors_path = out / "errors.jsonl"

    cache = load_direction_cache(direction_dir)
    arr = cache["arrays"]

    vector_key = args.vector_mode
    if vector_key not in arr:
        raise RuntimeError(
            f"relation_vectors.npz has no {vector_key!r}; keys={sorted(arr)}"
        )
    if args.vector_mode == "residual" and "no_image" not in arr:
        raise RuntimeError("residual intervention requires cached no_image vectors")

    X = np.asarray(arr[vector_key], dtype=np.float32)
    X_no = (
        np.asarray(arr["no_image"], dtype=np.float32)
        if "no_image" in arr else None
    )
    sids = np.asarray(arr["sample_index"]).astype(np.int64)
    labels = np.asarray(
        [normalize_relation(x) for x in arr["relation"].tolist()],
        dtype=object,
    )

    if X.ndim != 4:
        raise RuntimeError(f"{vector_key} array must be [N,L,H,D], got {X.shape}")
    N, n_layers_cache, n_heads_cache, head_dim_cache = X.shape

    valid = np.asarray([x in RELATIONS for x in labels], dtype=bool)
    if not np.all(valid):
        X = X[valid]
        if X_no is not None:
            X_no = X_no[valid]
        sids = sids[valid]
        labels = labels[valid]
        N = len(labels)

    if args.max_samples > 0 and args.max_samples < N:
        keep = stratified_limit_indices(
            list(range(N)),
            labels,
            args.max_samples,
            args.seed,
        )
        X = X[keep]
        if X_no is not None:
            X_no = X_no[keep]
        sids = sids[keep]
        labels = labels[keep]
        N = len(labels)

    train_idx, eval_idx = stratified_split(
        labels,
        args.train_ratio,
        args.seed,
    )
    if args.eval_max_samples > 0:
        eval_idx = stratified_limit_indices(
            eval_idx,
            labels,
            args.eval_max_samples,
            args.seed + 1,
        )

    rank_metric = args.rank_metric
    if rank_metric == "auto":
        rank_metric = (
            "residual_accuracy_mean"
            if args.vector_mode == "residual"
            else "img_accuracy_mean"
        )

    bundle_sizes = parse_int_list(args.bundle_sizes)
    alphas = parse_float_list(args.alphas)
    max_k = max(bundle_sizes)
    explicit = parse_heads(args.heads)
    selected_top = select_ranked_heads(
        rows=cache["head_rows"],
        metric=rank_metric,
        explicit=explicit,
        max_k=max_k,
    )

    if any(l < 0 or l >= n_layers_cache or h < 0 or h >= n_heads_cache
           for l, h in selected_top):
        raise RuntimeError("Selected head outside cache dimensions")

    random_top = matched_random_heads(
        selected=selected_top,
        n_heads=n_heads_cache,
        seed=args.random_seed,
    )

    all_fit_heads = list(dict.fromkeys(selected_top + random_top))
    class_means = fit_class_means(
        X=X,
        labels=labels,
        train_idx=train_idx,
        heads=all_fit_heads,
    )

    heldout_selected = heldout_head_accuracy(
        X, labels, train_idx, eval_idx, selected_top
    )
    heldout_random = heldout_head_accuracy(
        X, labels, train_idx, eval_idx, random_top
    )
    for r in heldout_selected:
        r["family"] = "selected"
    for r in heldout_random:
        r["family"] = "random"
    write_csv(out / "heldout_head_accuracy.csv", heldout_selected + heldout_random)

    # Load repo probe module and dataset/model.
    probe = importlib.import_module(args.probe_module)
    base = probe.base

    records, audit = base.load_records(args.dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    cache_index_by_sid = {int(sid): i for i, sid in enumerate(sids.tolist())}

    eval_sids = [
        int(sids[i])
        for i in eval_idx
        if int(sids[i]) in record_by_sid
    ]
    if not eval_sids:
        raise RuntimeError("No held-out eval SIDs found in dataset")

    spec = base.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class)
    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    model = processor = None
    try:
        print(f"Loading {args.model}: {spec.repo_id}", flush=True)
        model = model_cls.from_pretrained(spec.repo_id, **kw)
        model.eval()
        clear_sampling_defaults(model)
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)
        decoder_layers, decoder_path = probe.resolve_decoder_layers(model)
        n_heads, head_dim = probe.scan_shape(model, decoder_layers)

        if n_heads != n_heads_cache or head_dim != head_dim_cache:
            raise RuntimeError(
                f"Cache/model head shape mismatch: cache H={n_heads_cache},D={head_dim_cache}; "
                f"model H={n_heads},D={head_dim}"
            )
        if len(decoder_layers) != n_layers_cache:
            raise RuntimeError(
                f"Cache/model layer mismatch: {n_layers_cache} vs {len(decoder_layers)}"
            )

        relation_token_map = relation_token_variants(processor.tokenizer)

        print("\n" + "=" * 128)
        print("GT-ORACLE DIRECTION-HEAD ACTUATOR TEST")
        print("=" * 128)
        print("vector mode       :", args.vector_mode)
        print("rank metric       :", rank_metric)
        print("train/eval cache  :", len(train_idx), "/", len(eval_idx))
        print("selected Top      :", ", ".join(hname(h) for h in selected_top))
        print("random matched    :", ", ".join(hname(h) for h in random_top))
        print("bundles           :", bundle_sizes)
        print("alphas            :", alphas)
        print("prompt            :", args.prompt_template)
        print("=" * 128)

        print("\nHeld-out direction accuracies:")
        for srow, rrow in zip(heldout_selected[:max_k], heldout_random[:max_k]):
            print(
                f"{srow['head']:<8s} {100*float(srow['heldout_accuracy']):6.2f}%"
                f"    random {rrow['head']:<8s} {100*float(rrow['heldout_accuracy']):6.2f}%"
            )

        # ---------------------------------------------------------------------
        # Baseline generation and positions for held-out samples.
        # ---------------------------------------------------------------------
        baseline_rows: List[Dict[str, Any]] = []
        runtime_meta: Dict[int, Dict[str, Any]] = {}

        for idx, sid in enumerate(
            tqdm(eval_sids, desc="baseline-generation"),
            start=1,
        ):
            rec = record_by_sid[sid]
            image = batch = None
            try:
                question = args.prompt_template.format(
                    subject=rec.subject,
                    reference=rec.reference,
                )
                from PIL import Image
                image = Image.open(rec.image_path).convert("RGB")
                batch = build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )
                ids_list = [
                    int(x) for x in batch["input_ids"][0].detach().cpu().tolist()
                ]
                apos = probe.locate_phrase_positions(
                    processor.tokenizer, ids_list, str(rec.subject)
                )
                bpos = probe.locate_phrase_positions(
                    processor.tokenizer, ids_list, str(rec.reference)
                )

                # First-step baseline.
                bout = model(
                    **batch,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                first_pred, first_scores = relation_scores_from_logits(
                    bout.logits[0, -1], relation_token_map
                )
                del bout

                input_len = int(batch["input_ids"].shape[1])
                gids = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                text = processor.tokenizer.decode(
                    gids[0, input_len:],
                    skip_special_tokens=True,
                ).strip()
                del gids

                pred = normalize_relation(text)
                ci = cache_index_by_sid[sid]
                gt = str(labels[ci])
                correct = pred == gt

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "baseline_generation_text": text,
                    "baseline_generation_prediction": pred,
                    "baseline_generation_correct": correct,
                    "baseline_first_step_prediction": first_pred,
                    "baseline_first_step_scores": json.dumps(
                        first_scores.tolist()
                    ),
                })
                runtime_meta[sid] = {
                    "a_positions": apos,
                    "b_positions": bpos,
                    "question": question,
                }
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "baseline",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if image is not None:
                    image.close()
                del batch
                gc.collect()
                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and idx % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(out / "baseline_eval.csv", baseline_rows)
        baseline_by_sid = {int(r["sid"]): r for r in baseline_rows}
        baseline_acc = safe_mean(
            float(bool(r["baseline_generation_correct"])) for r in baseline_rows
        )

        wrong_sids = []
        for r in baseline_rows:
            pred = r["baseline_generation_prediction"]
            gt = str(r["gt"])
            if bool(r["baseline_generation_correct"]):
                continue
            if pred not in RELATIONS:
                continue
            if args.only_opposite_errors and pred != OPPOSITE[gt]:
                continue
            wrong_sids.append(int(r["sid"]))

        print("\nBaseline:")
        print(
            f"  generation ACC = {100*baseline_acc:.2f}% "
            f"({sum(bool(r['baseline_generation_correct']) for r in baseline_rows)}/"
            f"{len(baseline_rows)})"
        )
        print(
            f"  eligible baseline-WRONG oracle targets = "
            f"{len(wrong_sids)}/{len(baseline_rows)}"
        )

        if not wrong_sids:
            raise RuntimeError("No eligible baseline-wrong samples")

        # ---------------------------------------------------------------------
        # Oracle intervention.
        # ---------------------------------------------------------------------
        result_rows: List[Dict[str, Any]] = []

        conditions = []
        for k in bundle_sizes:
            if k > len(selected_top):
                continue
            for alpha in alphas:
                conditions.append(("selected", k, alpha, selected_top[:k]))
                if args.include_random_control:
                    conditions.append(("random", k, alpha, random_top[:k]))

        for ci, (family, k, alpha, heads) in enumerate(conditions, start=1):
            print(
                f"\n[{ci}/{len(conditions)}] {family} Top{k} "
                f"alpha={alpha:g}: {','.join(hname(h) for h in heads)}",
                flush=True,
            )

            for si, sid in enumerate(
                tqdm(wrong_sids, desc=f"{family}:K{k}:a{alpha:g}"),
                start=1,
            ):
                rec = record_by_sid[sid]
                image = batch = None
                try:
                    brow = baseline_by_sid[sid]
                    source = str(brow["baseline_generation_prediction"])
                    target = str(brow["gt"])
                    ci_cache = cache_index_by_sid[sid]

                    from PIL import Image
                    image = Image.open(rec.image_path).convert("RGB")
                    batch = build_batch(
                        probe=probe,
                        processor=processor,
                        question=runtime_meta[sid]["question"],
                        image=image,
                        device=device,
                    )
                    ids_list = [
                        int(x) for x in batch["input_ids"][0].detach().cpu().tolist()
                    ]
                    apos = probe.locate_phrase_positions(
                        processor.tokenizer, ids_list, str(rec.subject)
                    )
                    bpos = probe.locate_phrase_positions(
                        processor.tokenizer, ids_list, str(rec.reference)
                    )

                    no_by_head = {}
                    if args.vector_mode == "residual":
                        for head_key in heads:
                            l, h = head_key
                            no_by_head[head_key] = np.asarray(
                                X_no[ci_cache, l, h, :],
                                dtype=np.float32,
                            )

                    # First-step intervention.
                    patch1 = DirectionHeadOraclePatch(
                        probe=probe,
                        decoder_layers=decoder_layers,
                        heads=heads,
                        n_heads=n_heads,
                        head_dim=head_dim,
                        a_positions=apos,
                        b_positions=bpos,
                        pool=args.pool,
                        class_means=class_means,
                        source_relation=source,
                        target_relation=target,
                        alpha=alpha,
                        vector_mode=args.vector_mode,
                        noimage_by_head=no_by_head,
                    )
                    first_pred, first_scores = first_step_with_patch(
                        model=model,
                        batch=batch,
                        patch=patch1,
                        relation_token_map=relation_token_map,
                    )
                    first_meta = dict(patch1.meta)

                    # Greedy generation intervention.
                    patch2 = DirectionHeadOraclePatch(
                        probe=probe,
                        decoder_layers=decoder_layers,
                        heads=heads,
                        n_heads=n_heads,
                        head_dim=head_dim,
                        a_positions=apos,
                        b_positions=bpos,
                        pool=args.pool,
                        class_means=class_means,
                        source_relation=source,
                        target_relation=target,
                        alpha=alpha,
                        vector_mode=args.vector_mode,
                        noimage_by_head=no_by_head,
                    )
                    gen_pred, gen_text = generate_with_patch(
                        model=model,
                        processor=processor,
                        batch=batch,
                        patch=patch2,
                        max_new_tokens=args.max_new_tokens,
                    )

                    base_scores = np.asarray(
                        json.loads(brow["baseline_first_step_scores"]),
                        dtype=np.float32,
                    )
                    base_margin = float(
                        base_scores[RID[target]] - base_scores[RID[source]]
                    )
                    patch_margin = float(
                        first_scores[RID[target]] - first_scores[RID[source]]
                    )

                    result = {
                        "sid": sid,
                        "family": family,
                        "K": k,
                        "alpha": float(alpha),
                        "heads": ",".join(hname(h) for h in heads),
                        "vector_mode": args.vector_mode,
                        "source_relation": source,
                        "gt_target": target,
                        "error_type": (
                            "opposite"
                            if source == OPPOSITE[target]
                            else "cross_axis"
                        ),
                        "baseline_generation_prediction": source,
                        "baseline_generation_correct": False,
                        "patched_first_step_prediction": first_pred,
                        "patched_first_step_target_margin": patch_margin,
                        "baseline_first_step_target_margin": base_margin,
                        "first_step_margin_change": patch_margin - base_margin,
                        "patched_generation_text": gen_text,
                        "patched_generation_prediction": gen_pred,
                        "wrong_to_correct": gen_pred == target,
                        "generation_follow_gt": gen_pred == target,
                        "generation_changed": gen_pred != source,
                        "total_delta_norm": float(
                            sum(
                                float(m.get("delta_norm", 0.0))
                                for m in first_meta.values()
                            )
                        ),
                        "head_patch_meta": json.dumps(
                            first_meta,
                            ensure_ascii=False,
                        ),
                    }
                    result_rows.append(result)
                    append_jsonl(out / "oracle_patch_results.jsonl", result)

                except Exception as exc:
                    append_jsonl(errors_path, {
                        "phase": "oracle_patch",
                        "sid": sid,
                        "family": family,
                        "K": k,
                        "alpha": alpha,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    if args.fail_fast:
                        raise
                finally:
                    if image is not None:
                        image.close()
                    del batch
                    gc.collect()
                    if (
                        torch.cuda.is_available()
                        and args.empty_cache_every > 0
                        and si % args.empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()

        write_csv(out / "oracle_patch_results.csv", result_rows)

        # ---------------------------------------------------------------------
        # Summary: baseline-correct samples untouched, only wrong eligible patched.
        # ---------------------------------------------------------------------
        groups: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = defaultdict(list)
        for r in result_rows:
            groups[(str(r["family"]), int(r["K"]), float(r["alpha"]))].append(r)

        summary = []
        N_eval = len(baseline_rows)
        n_base_correct = sum(bool(r["baseline_generation_correct"]) for r in baseline_rows)

        for (family, k, alpha), rows in sorted(groups.items()):
            w2c = sum(bool(r["wrong_to_correct"]) for r in rows)
            oracle_acc = (n_base_correct + w2c) / max(N_eval, 1)
            opposite_rows = [r for r in rows if r["error_type"] == "opposite"]
            cross_rows = [r for r in rows if r["error_type"] == "cross_axis"]
            summary.append({
                "family": family,
                "K": k,
                "alpha": alpha,
                "N_eval": N_eval,
                "N_baseline_wrong_patched": len(rows),
                "baseline_generation_acc": baseline_acc,
                "wrong_to_correct": w2c,
                "wrong_to_correct_rate": w2c / max(len(rows), 1),
                "oracle_generation_acc": oracle_acc,
                "oracle_delta_acc": oracle_acc - baseline_acc,
                "generation_changed_rate": safe_mean(
                    float(bool(r["generation_changed"])) for r in rows
                ),
                "first_step_follow_gt_rate": safe_mean(
                    float(r["patched_first_step_prediction"] == r["gt_target"])
                    for r in rows
                ),
                "mean_first_step_margin_change": safe_mean(
                    float(r["first_step_margin_change"]) for r in rows
                ),
                "opposite_N": len(opposite_rows),
                "opposite_W2C_rate": safe_mean(
                    float(bool(r["wrong_to_correct"])) for r in opposite_rows
                ),
                "cross_axis_N": len(cross_rows),
                "cross_axis_W2C_rate": safe_mean(
                    float(bool(r["wrong_to_correct"])) for r in cross_rows
                ),
                "mean_total_delta_norm": safe_mean(
                    float(r["total_delta_norm"]) for r in rows
                ),
            })

        write_csv(out / "summary.csv", summary)

        print("\n" + "=" * 150)
        print("GT-ORACLE DIRECTION-HEAD REPAIR SUMMARY")
        print("=" * 150)
        print(
            f"{'family':<9s} {'K':>3s} {'alpha':>6s} {'Nwrong':>7s} "
            f"{'W->C':>6s} {'W->C%':>8s} {'baseACC':>9s} {'oracleACC':>10s} "
            f"{'delta':>8s} {'changed':>8s} {'1stGT':>8s} {'marginΔ':>9s}"
        )
        print("-" * 150)
        for r in summary:
            print(
                f"{r['family']:<9s} "
                f"{int(r['K']):>3d} "
                f"{float(r['alpha']):>6.2f} "
                f"{int(r['N_baseline_wrong_patched']):>7d} "
                f"{int(r['wrong_to_correct']):>6d} "
                f"{100*float(r['wrong_to_correct_rate']):>7.2f}% "
                f"{100*float(r['baseline_generation_acc']):>8.2f}% "
                f"{100*float(r['oracle_generation_acc']):>9.2f}% "
                f"{100*float(r['oracle_delta_acc']):>+7.2f} "
                f"{100*float(r['generation_changed_rate']):>7.2f}% "
                f"{100*float(r['first_step_follow_gt_rate']):>7.2f}% "
                f"{float(r['mean_first_step_margin_change']):>+9.3f}"
            )
        print("=" * 150)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "direction_dir": str(direction_dir),
            "direction_npz": cache["npz_path"],
            "direction_csv": cache["csv_path"],
            "vector_mode": args.vector_mode,
            "rank_metric": rank_metric,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "selected_top_heads": [hname(h) for h in selected_top],
            "random_matched_heads": [hname(h) for h in random_top],
            "bundle_sizes": bundle_sizes,
            "alphas": alphas,
            "prompt_template": args.prompt_template,
            "pool": args.pool,
            "N_cache": N,
            "N_train": len(train_idx),
            "N_eval_requested": len(eval_idx),
            "N_eval_completed": len(baseline_rows),
            "N_baseline_wrong_patched": len(wrong_sids),
            "baseline_generation_acc": baseline_acc,
            "only_opposite_errors": args.only_opposite_errors,
            "uses_gt_for_target": True,
            "baseline_correct_samples_are_untouched": True,
            "patch_site": "attention.o_proj input / pre-W_O query-head slice at object spans",
            "patch_formula": (
                "u=normalize(mu_GT-mu_generation); "
                "current=d_img[-d_noimg]; "
                "delta=alpha*(<mu_GT,u>-<current,u>)*u; "
                "zA+=delta/2; zB-=delta/2"
            ),
            "audit": audit,
        }
        (out / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        selected_summary = [
            r for r in summary if r["family"] == "selected"
        ]
        selected_summary.sort(
            key=lambda r: (
                -float(r["wrong_to_correct_rate"]),
                -float(r["oracle_delta_acc"]),
            )
        )
        lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"vector_mode: {args.vector_mode}",
            f"rank_metric: {rank_metric}",
            f"baseline_generation_acc: {100*baseline_acc:.2f}%",
            f"baseline_wrong_oracle_patched: {len(wrong_sids)}",
            "",
            "SELECTED HEADS",
            ", ".join(hname(h) for h in selected_top),
            "",
            "BEST SELECTED SETTINGS",
        ]
        for r in selected_summary[:10]:
            lines.append(
                f"K={int(r['K'])} alpha={float(r['alpha']):.2f}: "
                f"W->C={int(r['wrong_to_correct'])}/"
                f"{int(r['N_baseline_wrong_patched'])} "
                f"({100*float(r['wrong_to_correct_rate']):.2f}%), "
                f"oracle ACC={100*float(r['oracle_generation_acc']):.2f}% "
                f"(delta={100*float(r['oracle_delta_acc']):+.2f} pp)"
            )
        lines += [
            "",
            "READING THE RESULT",
            "Large selected W->C with selected >> same-layer random means high-ACC",
            "direction heads are useful causal actuators when the target relation is known.",
            "Low W->C despite GT oracle means their probe accuracy is mostly decodable",
            "information and is not sufficient for generation repair through this coordinate.",
            "This run does NOT validate a GT-free detector; that should only be connected",
            "after the actuator test succeeds.",
        ]
        (out / "report.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "config.json",
            "heldout_head_accuracy.csv",
            "baseline_eval.csv",
            "oracle_patch_results.csv",
            "oracle_patch_results.jsonl",
            "summary.csv",
            "report.txt",
        ):
            print(" ", out / name)

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
