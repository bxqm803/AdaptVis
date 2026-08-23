#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT-free spatial redeployment feasibility test.

Purpose
-------
The previous feasibility experiment showed that many generation errors have a
high-confidence multi-head Image-NoImage Direction consensus that points to the
correct spatial relation.  This script tests the next question:

    Can that INTERNAL consensus be re-deployed into the prompt-last state and
    improve actual free generation, without knowing which samples are wrong?

Key protocol
------------
1) Reuse the exact train/dev/test split, selected Direction heads, and consensus
   operating point from validate_grounded_spatial_consensus_v1.py.
2) Rebuild multi-head residual consensus from relation_vectors.npz.
3) On TRAIN only, learn a low-dimensional last-token spatial coordinate at a
   small set of candidate decoder layers using

       h_residual = h_image(last) - h_noimage(last)

   and relation centroids for left/right/above/below.
4) On DEV only, choose intervention layer + alpha.
5) On untouched TEST, the intervention guide is ONLY the internal consensus.
   Ground-truth labels are never used to decide whether/how to repair a sample.
6) Report two GT-free trigger policies from the same repaired generations:

   all_covered:
       repair every sample with high-confidence internal consensus.
       This intentionally includes baseline-correct samples, so damage is visible.

   conflict_only:
       repair only when high-confidence internal consensus disagrees with the
       baseline generation prediction.  This still does NOT use ground truth.

Intervention
------------
For a chosen layer L and internal guide relation r, learn from TRAIN residual
last-token states an axis u_L and target coordinate c_L(r).  For a test sample,
first obtain the no-image last-token state at L. During the image prefill, patch
only the prompt-last block output:

    residual_current = h_image - h_noimage
    current_coord    = <residual_current, u_L>
    delta            = alpha * (target_coord(r) - current_coord) * u_L
    h_image'          = h_image + delta

Only this 1-D spatial component is altered; the orthogonal hidden state is left
unchanged.  A max-delta-ratio clip prevents an excessively large edit.

This is a feasibility experiment, not a claim that the learned axis is the
unique causal spatial mechanism.

Expected repository context
---------------------------
Run from AdaptVis/tree/llava16 with these files present:
    extract_two_object_relation_states.py
    analyze_coco_head_object_residual_direction_probe_v1.py
    validate_grounded_spatial_consensus_v1.py

Example
-------
python eval_consensus_spatial_redeployment_v1.py \
  --feasibility-dir output/qwen3b_coco_grounded_consensus_v1 \
  --layers 22,24,26,28 \
  --alpha-grid 0.25,0.5,1.0,1.5 \
  --device cuda:0 \
  --output-dir output/qwen3b_coco_consensus_redeployment_v1 \
  --overwrite

For a smoke test only:
    --max-dev-samples 12 --max-test-samples 24
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
import time
import traceback
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
import analyze_coco_head_object_residual_direction_probe_v1 as direction_base
import validate_grounded_spatial_consensus_v1 as feas


SCRIPT_VERSION = "eval-consensus-spatial-redeployment-v1"
RELATIONS: Tuple[str, ...] = ("left", "right", "above", "below")
REL_TO_IDX = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--feasibility-dir",
        required=True,
        help="Output directory of validate_grounded_spatial_consensus_v1.py",
    )
    p.add_argument("--dataset", default=None, help="Default: read from feasibility summary")
    p.add_argument("--model", default=None, help="Default: read from feasibility summary")
    p.add_argument("--data-root", default="data")
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
        help="Must match the prompt used by the feasibility run / Direction NPZ.",
    )
    p.add_argument(
        "--baseline-generation-jsonl",
        default=None,
        help="Default: generation_jsonl path recorded in feasibility summary.",
    )
    p.add_argument(
        "--layers",
        default="22,24,26,28",
        help="Candidate decoder block indices. Supports comma values and a-b ranges.",
    )
    p.add_argument(
        "--alpha-grid",
        default="0.25,0.5,1.0,1.5",
        help="DEV-only strength grid for target-coordinate correction.",
    )
    p.add_argument(
        "--max-delta-ratio",
        type=float,
        default=0.15,
        help="Clip ||delta|| to this fraction of current prompt-last hidden norm; <=0 disables.",
    )
    p.add_argument(
        "--tune-trigger",
        choices=["conflict_only", "all_covered"],
        default="conflict_only",
        help="Which GT-free trigger policy chooses layer/alpha on DEV.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--max-dev-samples", type=int, default=None, help="Smoke test only")
    p.add_argument("--max-test-samples", type=int, default=None, help="Smoke test only")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--reuse-noimage-cache",
        action="store_true",
        help="Reuse output-dir/noimage_last_states.pt when signature matches.",
    )
    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    out: List[int] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(piece))
    return list(dict.fromkeys(out))


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Cannot locate tensor in output type {type(output)}")


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for i, x in enumerate(items):
            if torch.is_tensor(x):
                items[i] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, x in enumerate(items):
            if torch.is_tensor(x):
                items[i] = replacement
                return items
    raise TypeError(f"Cannot replace tensor in output type {type(output)}")


def load_split_indices(
    split_csv: Path,
    sid_to_row: Mapping[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: Dict[str, List[int]] = {"train": [], "dev": [], "test": []}
    with split_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = str(row["split"])
            if name not in groups:
                continue
            sid = int(row["sid"])
            if sid not in sid_to_row:
                raise RuntimeError(f"split.csv sid={sid} absent from relation_vectors.npz")
            groups[name].append(int(sid_to_row[sid]))
    for name in groups:
        if not groups[name]:
            raise RuntimeError(f"split.csv has no {name} samples")
    return tuple(np.asarray(groups[x], dtype=np.int64) for x in ("train", "dev", "test"))  # type: ignore


def balanced_accuracy_per_head(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # pred [N,H], gt [N]
    out = np.zeros(pred.shape[1], dtype=np.float64)
    for h in range(pred.shape[1]):
        cls = []
        for r in range(len(RELATIONS)):
            mask = gt == r
            if mask.any():
                cls.append(float(np.mean(pred[mask, h] == gt[mask])))
        out[h] = float(np.mean(cls)) if cls else float("nan")
    return out


def parse_head_name(name: str) -> Tuple[int, int]:
    m = re.fullmatch(r"L(\d+)H(\d+)", str(name).strip())
    if not m:
        raise ValueError(f"Bad head name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def rebuild_consensus(
    *,
    feasibility_summary: Mapping[str, Any],
    npz_path: Path,
    split_csv: Path,
) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        required = {"sample_index", "relation", "residual"}
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"{npz_path} missing {sorted(missing)}")
        sids = np.asarray(data["sample_index"], dtype=np.int64)
        labels = np.asarray([feas.normalize_relation(x) for x in data["relation"]], dtype=object)
        residual4 = np.asarray(data["residual"], dtype=np.float32)
        layer_ids = np.asarray(
            data["decoder_block_index"] if "decoder_block_index" in data.files else np.arange(residual4.shape[1]),
            dtype=np.int64,
        )
        head_ids = np.asarray(
            data["head_index"] if "head_index" in data.files else np.arange(residual4.shape[2]),
            dtype=np.int64,
        )

    valid = np.asarray([x in REL_TO_IDX for x in labels], dtype=bool)
    sids = sids[valid]
    labels = labels[valid]
    residual4 = residual4[valid]
    sid_to_row = {int(sid): i for i, sid in enumerate(sids.tolist())}
    if len(sid_to_row) != len(sids):
        raise RuntimeError("Duplicate SIDs in direction NPZ")

    train_idx, dev_idx, test_idx = load_split_indices(split_csv, sid_to_row)
    residual = feas.flatten_heads(residual4)
    center, proto = feas.fit_prototypes(residual, labels, train_idx)
    dev_scores = feas.score_prototypes(residual, dev_idx, center, proto)
    dev_gt = np.asarray([REL_TO_IDX[str(labels[i])] for i in dev_idx], dtype=np.int64)
    dev_pred, dev_margin = feas.top_prediction_and_margin(dev_scores)
    dev_bal = balanced_accuracy_per_head(dev_pred, dev_gt)
    reliability = np.maximum(dev_bal - 0.25, 1e-4)

    n_layers, n_heads = residual4.shape[1], residual4.shape[2]
    name_to_flat: Dict[str, int] = {}
    for lp in range(n_layers):
        for hp in range(n_heads):
            name = f"L{int(layer_ids[lp])}H{int(head_ids[hp]):02d}"
            name_to_flat[name] = lp * n_heads + hp

    selected_names = list(feasibility_summary["selected_heads"])
    selected = []
    for name in selected_names:
        if name not in name_to_flat:
            raise RuntimeError(f"Selected head {name} absent from current NPZ")
        selected.append(name_to_flat[name])
    selected = np.asarray(selected, dtype=np.int64)

    chosen = dict(feasibility_summary["selected_operating_point"])
    q = float(chosen["head_margin_quantile"])
    thresholds = np.zeros(residual.shape[1], dtype=np.float32)
    thresholds[selected] = np.quantile(dev_margin[:, selected], q, axis=0).astype(np.float32)

    result_by_split = {}
    for name, idx in (("dev", dev_idx), ("test", test_idx)):
        scores = feas.score_prototypes(residual, idx, center, proto)
        result = feas.consensus_from_scores(
            scores=scores[:, selected, :],
            selected_global_heads=selected,
            reliability_global=reliability,
            per_head_margin_threshold_global=thresholds,
            min_active=int(chosen["min_active"]),
            min_support=float(chosen["min_support"]),
        )
        result_by_split[name] = result

    return {
        "sids": sids,
        "labels": labels,
        "train_idx": train_idx,
        "dev_idx": dev_idx,
        "test_idx": test_idx,
        "result_dev": result_by_split["dev"],
        "result_test": result_by_split["test"],
        "selected_names": selected_names,
        "chosen": chosen,
    }


class MultiLayerLastCapture:
    """Persistent hooks that capture block-output last-token states."""

    def __init__(self, decoder_layers: Sequence[torch.nn.Module], layer_ids: Sequence[int]):
        self.layer_ids = [int(x) for x in layer_ids]
        self.states: Dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_id in self.layer_ids:
            module = decoder_layers[layer_id]

            def make_hook(li: int):
                def hook(_module: Any, _args: Any, output: Any) -> Any:
                    tensor = first_tensor(output)
                    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                        raise RuntimeError(f"Unexpected L{li} block output {tuple(tensor.shape)}")
                    self.states[li] = tensor[0, -1].detach().float().cpu()
                    return output
                return hook

            self.handles.append(module.register_forward_hook(make_hook(layer_id)))

    def reset(self) -> None:
        self.states = {}

    def get(self) -> Dict[int, torch.Tensor]:
        missing = [x for x in self.layer_ids if x not in self.states]
        if missing:
            raise RuntimeError(f"Missing captured layers: {missing}")
        return {k: v.clone() for k, v in self.states.items()}

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


class SpatialCoordinatePatch:
    """Patch one decoder block output on the generation prefill only."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        layer: int,
        noimage_state: torch.Tensor,
        axis: torch.Tensor,
        target_coord: float,
        alpha: float,
        max_delta_ratio: float,
    ) -> None:
        self.layer = int(layer)
        self.noimage_state = noimage_state.detach().float().cpu()
        self.axis = axis.detach().float().cpu()
        self.target_coord = float(target_coord)
        self.alpha = float(alpha)
        self.max_delta_ratio = float(max_delta_ratio)
        self.applied = 0
        self.decode_events = 0
        self.meta: Dict[str, Any] = {}
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        tensor = first_tensor(output)
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise RuntimeError(f"Unexpected L{self.layer} output {tuple(tensor.shape)}")

        # generate() prefill has the full prompt; cached decode steps normally S=1.
        # Apply only once, at the first non-trivial sequence event.
        if self.applied > 0 or int(tensor.shape[1]) <= 1:
            self.decode_events += 1
            return output

        current = tensor[0, -1].float()
        noimg = self.noimage_state.to(device=current.device, dtype=torch.float32)
        axis = self.axis.to(device=current.device, dtype=torch.float32)
        residual = current - noimg
        current_coord = float(torch.dot(residual, axis).item())
        raw_scalar = self.alpha * (self.target_coord - current_coord)
        delta = raw_scalar * axis

        clipped = False
        current_norm = float(current.norm().item())
        delta_norm = float(delta.norm().item())
        if self.max_delta_ratio > 0 and current_norm > EPS:
            max_norm = self.max_delta_ratio * current_norm
            if delta_norm > max_norm and delta_norm > EPS:
                delta = delta * (max_norm / delta_norm)
                clipped = True
                delta_norm = float(delta.norm().item())

        modified = tensor.clone()
        modified[0, -1] = modified[0, -1] + delta.to(modified.dtype)
        self.applied += 1
        self.meta = {
            "layer": self.layer,
            "current_coord": current_coord,
            "target_coord": self.target_coord,
            "raw_scalar": float(raw_scalar),
            "delta_norm": delta_norm,
            "current_hidden_norm": current_norm,
            "delta_ratio": delta_norm / max(current_norm, EPS),
            "clipped": bool(clipped),
        }
        return replace_first_tensor(output, modified)

    def validate(self) -> None:
        if self.applied != 1:
            raise RuntimeError(f"L{self.layer} patch expected 1 prefill application, got {self.applied}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


def capture_last_states(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    capture: MultiLayerLastCapture,
    question: str,
    image: Optional[Image.Image],
) -> Dict[int, torch.Tensor]:
    rendered = direction_base.build_chat_prompt(processor, question, image is not None)
    batch = direction_base.process_inputs(processor, rendered, image, device)
    capture.reset()
    try:
        with torch.inference_mode():
            model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        return capture.get()
    finally:
        del batch


def decode_generated(
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_length: int,
) -> Dict[str, Any]:
    text, token_ids = feas.decode_new_tokens(tokenizer, sequences, prompt_length)
    pred = feas.parse_generated_relation(text)
    return {"prediction": pred, "text": text, "token_ids": token_ids}


def generate_with_patch(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[torch.nn.Module],
    question: str,
    image: Image.Image,
    layer: int,
    noimage_state: torch.Tensor,
    axis: torch.Tensor,
    target_coord: float,
    alpha: float,
    max_delta_ratio: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    rendered = direction_base.build_chat_prompt(processor, question, True)
    batch = direction_base.process_inputs(processor, rendered, image, device)
    prompt_length = int(batch["input_ids"].shape[1])
    tok = processor.tokenizer
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = eos
    kwargs: Dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
    }
    if pad is not None:
        kwargs["pad_token_id"] = int(pad)

    patch = SpatialCoordinatePatch(
        module=decoder_layers[int(layer)],
        layer=int(layer),
        noimage_state=noimage_state,
        axis=axis,
        target_coord=float(target_coord),
        alpha=float(alpha),
        max_delta_ratio=float(max_delta_ratio),
    )
    try:
        with torch.inference_mode():
            seq = model.generate(**batch, **kwargs)
        patch.validate()
        result = decode_generated(tok, seq, prompt_length)
        result["patch"] = dict(patch.meta)
        return result
    finally:
        patch.close()
        del batch


def make_question(rec: Any, prompt_template: str) -> str:
    return prompt_template.format(subject=rec.subject, reference=rec.reference)


def relation_axis_and_targets(
    centroids: Mapping[int, Mapping[str, torch.Tensor]],
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, float]]]:
    axes: Dict[int, Dict[str, torch.Tensor]] = {}
    targets: Dict[int, Dict[str, float]] = {}
    for layer, by_rel in centroids.items():
        left = by_rel["left"].float()
        right = by_rel["right"].float()
        above = by_rel["above"].float()
        below = by_rel["below"].float()
        lr = left - right
        ud = above - below
        lr = lr / max(float(lr.norm().item()), EPS)
        ud = ud / max(float(ud.norm().item()), EPS)
        axes[layer] = {
            "left": lr,
            "right": lr,
            "above": ud,
            "below": ud,
        }
        targets[layer] = {
            "left": float(torch.dot(left, lr).item()),
            "right": float(torch.dot(right, lr).item()),
            "above": float(torch.dot(above, ud).item()),
            "below": float(torch.dot(below, ud).item()),
        }
    return axes, targets


def fit_train_last_residual_centroids(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[torch.nn.Module],
    candidate_layers: Sequence[int],
    train_sids: Sequence[int],
    label_by_sid: Mapping[int, str],
    record_by_sid: Mapping[int, Any],
    prompt_template: str,
) -> Dict[int, Dict[str, torch.Tensor]]:
    sums: Dict[int, Dict[str, Optional[torch.Tensor]]] = {
        L: {r: None for r in RELATIONS} for L in candidate_layers
    }
    counts: Dict[str, int] = {r: 0 for r in RELATIONS}
    capture = MultiLayerLastCapture(decoder_layers, candidate_layers)
    try:
        for sid in tqdm(train_sids, desc="train-last-residual"):
            rec = record_by_sid[int(sid)]
            gt = str(label_by_sid[int(sid)])
            question = make_question(rec, prompt_template)
            image = None
            try:
                image = Image.open(rec.image_path).convert("RGB")
                img_states = capture_last_states(
                    model=model, processor=processor, device=device,
                    capture=capture, question=question, image=image,
                )
                no_states = capture_last_states(
                    model=model, processor=processor, device=device,
                    capture=capture, question=question, image=None,
                )
                for L in candidate_layers:
                    residual = img_states[L] - no_states[L]
                    if sums[L][gt] is None:
                        sums[L][gt] = residual.clone()
                    else:
                        sums[L][gt] = sums[L][gt] + residual
                counts[gt] += 1
            finally:
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        capture.close()

    for rel in RELATIONS:
        if counts[rel] <= 0:
            raise RuntimeError(f"No TRAIN examples for relation {rel}")
    centroids: Dict[int, Dict[str, torch.Tensor]] = {}
    for L in candidate_layers:
        centroids[L] = {}
        for rel in RELATIONS:
            value = sums[L][rel]
            if value is None:
                raise RuntimeError(f"Missing centroid L{L} {rel}")
            centroids[L][rel] = value / float(counts[rel])
    return centroids


def noimage_cache_signature(
    *,
    model: str,
    dataset: str,
    layers: Sequence[int],
    sids: Sequence[int],
    prompt_template: str,
) -> Dict[str, Any]:
    return {
        "model": model,
        "dataset": dataset,
        "layers": list(map(int, layers)),
        "sids": list(map(int, sids)),
        "prompt_template": prompt_template,
    }


def build_noimage_cache(
    *,
    cache_path: Path,
    reuse: bool,
    signature: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[torch.nn.Module],
    candidate_layers: Sequence[int],
    sids: Sequence[int],
    record_by_sid: Mapping[int, Any],
    prompt_template: str,
) -> Dict[int, Dict[int, torch.Tensor]]:
    if reuse and cache_path.exists():
        loaded = torch.load(cache_path, map_location="cpu")
        if loaded.get("signature") == dict(signature):
            print(f"[noimage] reused {cache_path}")
            return loaded["states"]
        print("[noimage] cache signature mismatch; rebuilding")

    states: Dict[int, Dict[int, torch.Tensor]] = {}
    capture = MultiLayerLastCapture(decoder_layers, candidate_layers)
    try:
        for sid in tqdm(sids, desc="noimage-last-state"):
            rec = record_by_sid[int(sid)]
            question = make_question(rec, prompt_template)
            got = capture_last_states(
                model=model, processor=processor, device=device,
                capture=capture, question=question, image=None,
            )
            states[int(sid)] = {int(L): got[int(L)].half().cpu() for L in candidate_layers}
    finally:
        capture.close()
    torch.save({"signature": dict(signature), "states": states}, cache_path)
    return states


def load_baseline_generation(path: Path) -> Dict[int, Dict[str, Any]]:
    return feas.load_generation_cache(path)


def subset_indices(indices: np.ndarray, limit: Optional[int], seed: int) -> np.ndarray:
    if limit is None or int(limit) >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    picked = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(picked)
    return picked[: int(limit)]


def summarize_policy(
    *,
    gt: Sequence[str],
    baseline_pred: Sequence[Optional[str]],
    repaired_pred: Sequence[Optional[str]],
    guide: Sequence[str],
    covered: Sequence[bool],
    policy: str,
) -> Dict[str, Any]:
    gt = np.asarray(list(gt), dtype=object)
    basep = np.asarray(list(baseline_pred), dtype=object)
    repp = np.asarray(list(repaired_pred), dtype=object)
    guide = np.asarray(list(guide), dtype=object)
    covered = np.asarray(list(covered), dtype=bool)

    base_correct = basep == gt
    if policy == "all_covered":
        triggered = covered
    elif policy == "conflict_only":
        triggered = covered & (basep != guide)
    else:
        raise ValueError(policy)

    finalp = basep.copy()
    finalp[triggered] = repp[triggered]
    final_correct = finalp == gt
    repaired_wrong = (~base_correct) & final_correct
    damaged_correct = base_correct & (~final_correct)
    changed = finalp != basep
    guide_correct = guide == gt

    return {
        "policy": policy,
        "n": int(len(gt)),
        "triggered_n": int(triggered.sum()),
        "trigger_rate": float(triggered.mean()) if len(gt) else float("nan"),
        "baseline_accuracy": float(base_correct.mean()) if len(gt) else float("nan"),
        "intervention_accuracy": float(final_correct.mean()) if len(gt) else float("nan"),
        "accuracy_change": float(final_correct.mean() - base_correct.mean()) if len(gt) else float("nan"),
        "baseline_correct_n": int(base_correct.sum()),
        "baseline_wrong_n": int((~base_correct).sum()),
        "repaired_wrong": int(repaired_wrong.sum()),
        "damaged_correct": int(damaged_correct.sum()),
        "net_repair": int(repaired_wrong.sum() - damaged_correct.sum()),
        "wrong_repair_rate": float(repaired_wrong[~base_correct].mean()) if (~base_correct).any() else float("nan"),
        "correct_damage_rate": float(damaged_correct[base_correct].mean()) if base_correct.any() else float("nan"),
        "prediction_changed": int(changed.sum()),
        "guide_correct_rate_triggered": float(guide_correct[triggered].mean()) if triggered.any() else float("nan"),
        "triggered_baseline_correct_n": int((triggered & base_correct).sum()),
        "triggered_baseline_wrong_n": int((triggered & (~base_correct)).sum()),
        "damaged_correct_among_triggered_correct": (
            float(damaged_correct[triggered & base_correct].mean())
            if (triggered & base_correct).any() else float("nan")
        ),
        "repaired_wrong_among_triggered_wrong": (
            float(repaired_wrong[triggered & (~base_correct)].mean())
            if (triggered & (~base_correct)).any() else float("nan")
        ),
    }


def run_repaired_subset(
    *,
    sids: Sequence[int],
    consensus_pred: Mapping[int, str],
    consensus_covered: Mapping[int, bool],
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[torch.nn.Module],
    record_by_sid: Mapping[int, Any],
    noimage_states: Mapping[int, Mapping[int, torch.Tensor]],
    axes: Mapping[int, Mapping[str, torch.Tensor]],
    targets: Mapping[int, Mapping[str, float]],
    layer: int,
    alpha: float,
    max_delta_ratio: float,
    prompt_template: str,
    max_new_tokens: int,
    desc: str,
    print_every: int = 0,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    covered_sids = [int(sid) for sid in sids if bool(consensus_covered[int(sid)])]
    for pos, sid in enumerate(tqdm(covered_sids, desc=desc), 1):
        rec = record_by_sid[int(sid)]
        guide = str(consensus_pred[int(sid)])
        question = make_question(rec, prompt_template)
        image = None
        try:
            image = Image.open(rec.image_path).convert("RGB")
            result = generate_with_patch(
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                question=question,
                image=image,
                layer=int(layer),
                noimage_state=noimage_states[int(sid)][int(layer)],
                axis=axes[int(layer)][guide],
                target_coord=targets[int(layer)][guide],
                alpha=float(alpha),
                max_delta_ratio=float(max_delta_ratio),
                max_new_tokens=int(max_new_tokens),
            )
            results[int(sid)] = result
        except Exception as exc:
            results[int(sid)] = {
                "prediction": None,
                "text": "",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            }
            tqdm.write(f"[repair ERROR] sid={sid}: {type(exc).__name__}: {exc}")
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if print_every > 0 and pos % print_every == 0:
            changed = sum(1 for x in results.values() if x.get("prediction") is not None)
            tqdm.write(f"[{desc}] {pos}/{len(covered_sids)} completed={changed}")
    return results


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out = Path(args.output_dir)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    feasibility_dir = Path(args.feasibility_dir)
    fsum_path = feasibility_dir / "summary.json"
    if not fsum_path.exists():
        raise FileNotFoundError(fsum_path)
    fsum = json.loads(fsum_path.read_text(encoding="utf-8"))
    dataset = args.dataset or str(fsum["dataset"])
    model_alias = args.model or str(fsum["model"])

    direction_dir = Path(fsum["direction_dir"])
    npz_path = direction_dir / "relation_vectors.npz"
    split_csv = feasibility_dir / "split.csv"
    if args.baseline_generation_jsonl:
        baseline_path = Path(args.baseline_generation_jsonl)
    else:
        baseline_path = Path(fsum["generation_jsonl"])
    for path in (npz_path, split_csv, baseline_path):
        if not path.exists():
            raise FileNotFoundError(path)

    consensus = rebuild_consensus(
        feasibility_summary=fsum,
        npz_path=npz_path,
        split_csv=split_csv,
    )
    sids = consensus["sids"]
    labels = consensus["labels"]
    train_idx = consensus["train_idx"]
    dev_idx_full = consensus["dev_idx"]
    test_idx_full = consensus["test_idx"]

    # Smoke caps are applied only after the official split/consensus is reconstructed.
    dev_idx = subset_indices(dev_idx_full, args.max_dev_samples, args.seed + 101)
    test_idx = subset_indices(test_idx_full, args.max_test_samples, args.seed + 202)
    if len(dev_idx) != len(dev_idx_full) or len(test_idx) != len(test_idx_full):
        print("[SMOKE MODE] DEV/TEST capped; do not report these numbers as final")

    # Consensus results were produced in dev_idx_full/test_idx_full local order.
    def consensus_maps(split_name: str, official_idx: np.ndarray) -> Tuple[Dict[int, str], Dict[int, bool], Dict[int, float]]:
        result = consensus[f"result_{split_name}"]
        pred_map: Dict[int, str] = {}
        cov_map: Dict[int, bool] = {}
        support_map: Dict[int, float] = {}
        for local, global_i in enumerate(official_idx.tolist()):
            sid = int(sids[global_i])
            pred_map[sid] = RELATIONS[int(result["prediction"][local])]
            cov_map[sid] = bool(result["covered"][local])
            support_map[sid] = float(result["support"][local])
        return pred_map, cov_map, support_map

    dev_cons_pred, dev_cons_cov, dev_cons_support = consensus_maps("dev", dev_idx_full)
    test_cons_pred, test_cons_cov, test_cons_support = consensus_maps("test", test_idx_full)

    label_by_sid = {int(sid): str(label) for sid, label in zip(sids.tolist(), labels.tolist())}
    train_sids = [int(sids[i]) for i in train_idx.tolist()]
    dev_sids = [int(sids[i]) for i in dev_idx.tolist()]
    test_sids = [int(sids[i]) for i in test_idx.tolist()]

    baseline = load_baseline_generation(baseline_path)
    for sid in [*dev_sids, *test_sids]:
        if sid not in baseline:
            raise RuntimeError(f"Baseline generation cache missing sid={sid}")

    candidate_layers = parse_layers(args.layers)
    alpha_grid = parse_floats(args.alpha_grid)
    if not candidate_layers:
        raise ValueError("No candidate layers")
    if not alpha_grid:
        raise ValueError("No alpha values")

    # Load model using the same repository model specs as the existing experiments.
    spec = base.SPECS[model_alias]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")
    kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl
    print(f"[model] loading {model_alias} from {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)
    decoder_layers, decoder_path = direction_base.resolve_decoder_layers(model)
    print(f"[model] decoder={decoder_path} n_layers={len(decoder_layers)}")
    bad_layers = [x for x in candidate_layers if not 0 <= int(x) < len(decoder_layers)]
    if bad_layers:
        raise ValueError(f"Candidate layers outside decoder: {bad_layers}")

    records, _audit = base.load_records(dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    missing_records = [sid for sid in [*train_sids, *dev_sids, *test_sids] if sid not in record_by_sid]
    if missing_records:
        raise RuntimeError(f"Dataset loader missing SIDs: {missing_records[:10]}")

    # ------------------------------------------------------------------
    # Stage 1: TRAIN-only last-token Image-NoImage spatial centroids.
    # ------------------------------------------------------------------
    print("\nStage 1/4: fit TRAIN last-token residual spatial coordinates")
    centroids = fit_train_last_residual_centroids(
        model=model,
        processor=processor,
        device=device,
        decoder_layers=decoder_layers,
        candidate_layers=candidate_layers,
        train_sids=train_sids,
        label_by_sid=label_by_sid,
        record_by_sid=record_by_sid,
        prompt_template=args.prompt_template,
    )
    axes, targets = relation_axis_and_targets(centroids)

    # Save small reusable prototype file.
    proto_arrays: Dict[str, np.ndarray] = {}
    for L in candidate_layers:
        for rel in RELATIONS:
            proto_arrays[f"centroid_L{L}_{rel}"] = centroids[L][rel].numpy().astype(np.float32)
            proto_arrays[f"axis_L{L}_{rel}"] = axes[L][rel].numpy().astype(np.float32)
            proto_arrays[f"target_L{L}_{rel}"] = np.asarray(targets[L][rel], dtype=np.float32)
    np.savez_compressed(out / "train_last_residual_spatial_coordinates.npz", **proto_arrays)

    # ------------------------------------------------------------------
    # Stage 2: no-image references for DEV + TEST. Required by the repair.
    # ------------------------------------------------------------------
    print("\nStage 2/4: capture no-image prompt-last references")
    noimg_sids = list(dict.fromkeys([*dev_sids, *test_sids]))
    noimg_sig = noimage_cache_signature(
        model=model_alias,
        dataset=dataset,
        layers=candidate_layers,
        sids=noimg_sids,
        prompt_template=args.prompt_template,
    )
    noimage_states = build_noimage_cache(
        cache_path=out / "noimage_last_states.pt",
        reuse=args.reuse_noimage_cache,
        signature=noimg_sig,
        model=model,
        processor=processor,
        device=device,
        decoder_layers=decoder_layers,
        candidate_layers=candidate_layers,
        sids=noimg_sids,
        record_by_sid=record_by_sid,
        prompt_template=args.prompt_template,
    )

    # ------------------------------------------------------------------
    # Stage 3: DEV-only layer/alpha tuning. We generate the repaired answer for
    # every COVERED sample. From the same generated outputs we evaluate both
    # all_covered and conflict_only policies; neither uses GT for triggering.
    # ------------------------------------------------------------------
    print("\nStage 3/4: DEV repair grid")
    dev_grid_rows: List[Dict[str, Any]] = []
    dev_gt = [label_by_sid[sid] for sid in dev_sids]
    dev_base_pred = [feas.normalize_relation(baseline[sid].get("prediction")) for sid in dev_sids]
    dev_guide = [dev_cons_pred[sid] for sid in dev_sids]
    dev_cov = [dev_cons_cov[sid] for sid in dev_sids]

    for L in candidate_layers:
        for alpha in alpha_grid:
            repaired = run_repaired_subset(
                sids=dev_sids,
                consensus_pred=dev_cons_pred,
                consensus_covered=dev_cons_cov,
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                record_by_sid=record_by_sid,
                noimage_states=noimage_states,
                axes=axes,
                targets=targets,
                layer=L,
                alpha=alpha,
                max_delta_ratio=args.max_delta_ratio,
                prompt_template=args.prompt_template,
                max_new_tokens=args.max_new_tokens,
                desc=f"DEV L{L} a={alpha:g}",
                print_every=0,
            )
            repaired_pred = [
                feas.normalize_relation(repaired[sid].get("prediction"))
                if dev_cons_cov[sid] else dev_base_pred[i]
                for i, sid in enumerate(dev_sids)
            ]
            for policy in ("all_covered", "conflict_only"):
                summ = summarize_policy(
                    gt=dev_gt,
                    baseline_pred=dev_base_pred,
                    repaired_pred=repaired_pred,
                    guide=dev_guide,
                    covered=dev_cov,
                    policy=policy,
                )
                dev_grid_rows.append({
                    "layer": int(L),
                    "alpha": float(alpha),
                    "max_delta_ratio": float(args.max_delta_ratio),
                    **summ,
                })
            del repaired
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(out / "dev_repair_grid.csv", dev_grid_rows)
    tune_rows = [r for r in dev_grid_rows if r["policy"] == args.tune_trigger]
    best = max(
        tune_rows,
        key=lambda r: (
            float(r["intervention_accuracy"]),
            int(r["net_repair"]),
            -int(r["damaged_correct"]),
            -float(r["alpha"]),
        ),
    )
    best_layer = int(best["layer"])
    best_alpha = float(best["alpha"])
    print("[DEV best]")
    print(json.dumps(best, indent=2))

    # ------------------------------------------------------------------
    # Stage 4: untouched TEST. Run repair on every consensus-covered sample,
    # INCLUDING baseline-correct samples. This one set of repaired generations
    # allows us to evaluate both all_covered and conflict_only without GT leakage.
    # ------------------------------------------------------------------
    print("\nStage 4/4: TEST free-generation repair")
    test_repaired = run_repaired_subset(
        sids=test_sids,
        consensus_pred=test_cons_pred,
        consensus_covered=test_cons_cov,
        model=model,
        processor=processor,
        device=device,
        decoder_layers=decoder_layers,
        record_by_sid=record_by_sid,
        noimage_states=noimage_states,
        axes=axes,
        targets=targets,
        layer=best_layer,
        alpha=best_alpha,
        max_delta_ratio=args.max_delta_ratio,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        desc=f"TEST L{best_layer} a={best_alpha:g}",
        print_every=args.print_every,
    )

    test_gt = [label_by_sid[sid] for sid in test_sids]
    test_base_pred = [feas.normalize_relation(baseline[sid].get("prediction")) for sid in test_sids]
    test_guide = [test_cons_pred[sid] for sid in test_sids]
    test_cov = [test_cons_cov[sid] for sid in test_sids]
    test_rep_pred = [
        feas.normalize_relation(test_repaired[sid].get("prediction"))
        if test_cons_cov[sid] else test_base_pred[i]
        for i, sid in enumerate(test_sids)
    ]

    test_summaries = {
        policy: summarize_policy(
            gt=test_gt,
            baseline_pred=test_base_pred,
            repaired_pred=test_rep_pred,
            guide=test_guide,
            covered=test_cov,
            policy=policy,
        )
        for policy in ("all_covered", "conflict_only")
    }

    # External consensus override is a useful upper reference, not a model repair.
    ext_pred = []
    for i, sid in enumerate(test_sids):
        b = test_base_pred[i]
        g = test_guide[i]
        if test_cov[i] and b != g:
            ext_pred.append(g)
        else:
            ext_pred.append(b)
    external_override_acc = float(np.mean(np.asarray(ext_pred, dtype=object) == np.asarray(test_gt, dtype=object)))

    samples_path = out / "test_samples.jsonl"
    if samples_path.exists():
        samples_path.unlink()
    for i, sid in enumerate(test_sids):
        basep = test_base_pred[i]
        repp = test_rep_pred[i]
        gt = test_gt[i]
        guide = test_guide[i]
        covered = bool(test_cov[i])
        all_trigger = covered
        conflict_trigger = covered and (basep != guide)
        all_final = repp if all_trigger else basep
        conflict_final = repp if conflict_trigger else basep
        rr = test_repaired.get(sid, {})
        append_jsonl(samples_path, {
            "sid": int(sid),
            "gt": gt,
            "baseline_prediction": basep,
            "baseline_correct": bool(basep == gt),
            "consensus_prediction": guide,
            "consensus_covered": covered,
            "consensus_support": float(test_cons_support[sid]),
            "consensus_correct_posthoc": bool(guide == gt),
            "repaired_generation_prediction": repp,
            "repaired_generation_text": rr.get("text", ""),
            "patch": rr.get("patch"),
            "repair_error": rr.get("error"),
            "all_covered_triggered": all_trigger,
            "all_covered_final_prediction": all_final,
            "all_covered_correct": bool(all_final == gt),
            "conflict_only_triggered": conflict_trigger,
            "conflict_only_final_prediction": conflict_final,
            "conflict_only_correct": bool(conflict_final == gt),
        })

    final_summary = {
        "script_version": SCRIPT_VERSION,
        "feasibility_dir": str(feasibility_dir),
        "dataset": dataset,
        "model": model_alias,
        "decoder_path": decoder_path,
        "prompt_template": args.prompt_template,
        "direction_dir": str(direction_dir),
        "baseline_generation_jsonl": str(baseline_path),
        "selected_direction_heads": consensus["selected_names"],
        "consensus_operating_point": consensus["chosen"],
        "candidate_layers": candidate_layers,
        "alpha_grid": alpha_grid,
        "max_delta_ratio": float(args.max_delta_ratio),
        "tune_trigger": args.tune_trigger,
        "dev_best": best,
        "n_train": len(train_sids),
        "n_dev_evaluated": len(dev_sids),
        "n_test_evaluated": len(test_sids),
        "test": test_summaries,
        "external_consensus_override_reference_accuracy": external_override_acc,
        "important_note": (
            "No test-time trigger uses ground truth. GT is used only after generation to score "
            "wrong->correct and correct->wrong. all_covered intentionally repairs baseline-correct "
            "samples too, so collateral damage is directly measured."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("CONSENSUS-GUIDED SPATIAL REDEPLOYMENT: TEST GENERATION")
    print("=" * 100)
    print(f"TEST N                         : {len(test_sids)}")
    print(f"DEV-selected layer / alpha    : L{best_layer} / {best_alpha:g}")
    print(f"external override reference   : {external_override_acc:.4f}  (NOT model repair)")
    for policy in ("all_covered", "conflict_only"):
        s = test_summaries[policy]
        print("-")
        print(f"policy                         : {policy}")
        print(f"triggered                      : {s['triggered_n']} ({s['trigger_rate']:.4f})")
        print(f"baseline generation ACC        : {s['baseline_accuracy']:.4f}")
        print(f"repaired generation ACC        : {s['intervention_accuracy']:.4f}")
        print(f"ACC change                     : {s['accuracy_change']:+.4f}")
        print(f"wrong -> correct               : {s['repaired_wrong']}")
        print(f"correct -> wrong               : {s['damaged_correct']}")
        print(f"net repair                     : {s['net_repair']:+d}")
        print(f"wrong repair rate              : {s['wrong_repair_rate']:.4f}")
        print(f"correct damage rate            : {s['correct_damage_rate']:.4f}")
        print(f"guide correct | triggered      : {s['guide_correct_rate_triggered']:.4f}")

    print("\nSaved:")
    for filename in (
        "summary.json",
        "dev_repair_grid.csv",
        "test_samples.jsonl",
        "train_last_residual_spatial_coordinates.npz",
        "noimage_last_states.pt",
    ):
        print(f"  {out / filename}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
