#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_llava_visual_entry_flow_v1.py

LLaVA visual-entry / visual-to-text transfer diagnostic.

Main hypothesis:
  spatial information can already be present in the VISUAL stream near the
  projector / early LLM layers, while transfer into TEXT object tokens is weak.

Stages probed:
  projector_pre
  projector_post
  llm_input_visual
  visual_Lk
  transfer_Lk
  text_residual_Lk

Visual-side object pooling deliberately does NOT depend on early text hidden
states. A fixed soft subject/reference visual locator is obtained from
subject/reference -> visual attention at one configurable later decoder layer.
The same visual weights are then reused to pool projector-pre, projector-post,
and every scanned LLM visual-token state.

For visual stage S:
  v_sub^S = sum_j w_sub[j] * v_j^S
  v_ref^S = sum_j w_ref[j] * v_j^S
  q_visual^S = v_sub^S - v_ref^S

For actual visual -> text transfer at layer k:
  c_sub<-V = O_k(sum_{j in V} A_k[sub,j] V_k[j])
  c_ref<-V = O_k(sum_{j in V} A_k[ref,j] V_k[j])
  q_transfer^k = c_sub<-V - c_ref<-V

For text carrier comparison:
  q_text^k = [(h_sub-h_ref)_REAL - (h_sub-h_ref)_NOIMAGE]

Every q is evaluated with the same TRAIN-only four-relation direction codebook
using repeated stratified 30/70 splits.

Interpretation:
  projector_pre high -> projector_post low
      => projector bottleneck

  projector_post / early visual_L high,
  early transfer_L + text_residual_L low,
  later transfer/text rise
      => early visual-to-text transfer bottleneck

  early visual_L already low
      => failure occurred before / at visual entry into LLM

No intervention is performed in this v1 diagnostic.

Run from AdaptVis/llava16 repository root. Requires:
  extract_two_object_relation_states.py
  trace_centroid_generation_groups_v2_1.py

Example:
CUDA_VISIBLE_DEVICES=0 python diagnose_llava_visual_entry_flow_v1.py \
  --dataset coco_two \
  --model llava-7b \
  --device cuda:0 \
  --layers auto \
  --localize-layer auto \
  --train-ratio 0.30 \
  --repeats 5 \
  --output-dir output/llava7b_visual_entry_flow_v1 \
  --overwrite

Smoke test:
CUDA_VISIBLE_DEVICES=0 python diagnose_llava_visual_entry_flow_v1.py \
  --dataset coco_two \
  --model llava-7b \
  --device cuda:0 \
  --layers 0,1,2,3,4,6,8,10,12 \
  --localize-layer 12 \
  --max-samples 40 \
  --repeats 2 \
  --output-dir output/llava7b_visual_entry_flow_smoke \
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
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import trace_centroid_generation_groups_v2_1 as trace


SCRIPT_VERSION = "llava-visual-entry-flow-v1"
RELATIONS = ("left", "right", "above", "below")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="llava-7b", choices=["llava-7b", "llava-13b"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl", default="eager", choices=["eager"],
        help="Transfer reconstruction requires returned eager attention weights.",
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument(
        "--layers", default="auto",
        help="Decoder blocks, e.g. 0,1,2,3,4,6,8,10,12 or auto/all.",
    )
    p.add_argument(
        "--localize-layer", default="auto",
        help=(
            "Layer used ONLY to obtain fixed subject/reference soft visual locators. "
            "auto=L12 for llava-7b, L16 for llava-13b."
        ),
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-vectors", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def norm_relation(value: Any) -> Optional[str]:
    return trace.normalize_relation(value)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_layers(text: str, n_layers: int) -> List[int]:
    raw = str(text).strip().lower()
    if raw == "all":
        return list(range(n_layers))
    if raw == "auto":
        points = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28]
        points += [
            int(round(0.75 * (n_layers - 1))),
            int(round(0.875 * (n_layers - 1))),
            n_layers - 1,
        ]
        return sorted({x for x in points if 0 <= x < n_layers})
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
    out = list(dict.fromkeys(out))
    bad = [x for x in out if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Invalid layers={bad}; valid 0..{n_layers-1}")
    if not out:
        raise ValueError("No layers selected")
    return out


def resolve_localize_layer(model_alias: str, text: str, n_layers: int) -> int:
    raw = str(text).strip().lower()
    if raw == "auto":
        layer = 12 if model_alias == "llava-7b" else 16
    else:
        layer = int(raw)
    if not 0 <= layer < n_layers:
        raise ValueError(f"localize layer {layer} outside 0..{n_layers-1}")
    return layer


def get_attr_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_projector(model: Any) -> Tuple[torch.nn.Module, str]:
    preferred = [
        "multi_modal_projector",
        "model.multi_modal_projector",
        "mm_projector",
        "model.mm_projector",
        "model.model.mm_projector",
    ]
    for path in preferred:
        module = get_attr_path(model, path)
        if isinstance(module, torch.nn.Module):
            return module, path
    candidates = []
    for name, module in model.named_modules():
        low = name.lower()
        if any(k in low for k in ("multi_modal_projector", "multimodal_projector", "mm_projector")):
            candidates.append((name, module))
    if candidates:
        candidates.sort(key=lambda x: len(x[0]))
        return candidates[0][1], candidates[0][0]
    raise RuntimeError("Could not resolve LLaVA multimodal projector")


def first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    for name in ("last_hidden_state", "hidden_states", "image_features"):
        item = getattr(value, name, None)
        if torch.is_tensor(item):
            return item
    raise TypeError(f"Could not extract tensor from {type(value)}")


class ProjectorCapture:
    def __init__(self, projector: torch.nn.Module):
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.handle = projector.register_forward_hook(self._hook)

    def _hook(self, _module, args, output):
        if not args:
            raise RuntimeError("Projector hook received no positional input")
        x = first_tensor(args[0])
        y = first_tensor(output)
        self.input = x.detach().float().cpu()
        self.output = y.detach().float().cpu()
        return output

    def validate(self):
        if self.input is None or self.output is None:
            raise RuntimeError("Projector hook did not capture input/output")

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def build_prompt(processor: Any, question: str, with_image: bool) -> str:
    content = ([{"type": "image"}] if with_image else []) + [
        {"type": "text", "text": question}
    ]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_real_batch(
    model: Any,
    processor: Any,
    rec: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
):
    rendered = build_prompt(processor, question, True)
    batch = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    batch = trace.move_batch(batch, device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    s_span, r_span = trace.locate_object_spans(
        processor.tokenizer, ids, str(rec.subject), str(rec.reference)
    )
    s_pos = list(range(s_span[0], s_span[1] + 1))
    r_pos = list(range(r_span[0], r_span[1] + 1))
    visual_pos = trace.resolve_visual_indices(model, processor, batch, ids)
    return batch, ids, s_pos, r_pos, visual_pos


def make_noimage_batch(
    processor: Any,
    rec: Any,
    question: str,
    device: torch.device,
):
    rendered = build_prompt(processor, question, False)
    batch = processor(text=[rendered], padding=True, return_tensors="pt")
    batch = trace.move_batch(batch, device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    s_span, r_span = trace.locate_object_spans(
        processor.tokenizer, ids, str(rec.subject), str(rec.reference)
    )
    s_pos = list(range(s_span[0], s_span[1] + 1))
    r_pos = list(range(r_span[0], r_span[1] + 1))
    return batch, ids, s_pos, r_pos


def pool_text(state: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    valid = [int(x) for x in positions if 0 <= int(x) < int(state.shape[1])]
    if not valid:
        raise RuntimeError("No valid text positions")
    if mode == "last":
        return state[0, valid[-1]]
    idx = torch.as_tensor(valid, device=state.device, dtype=torch.long)
    return state[0].index_select(0, idx).mean(dim=0)


def object_visual_attention_weights(
    attn: torch.Tensor,
    query_positions: Sequence[int],
    visual_positions: Sequence[int],
) -> Tuple[torch.Tensor, float]:
    if attn.ndim != 4 or int(attn.shape[0]) != 1:
        raise RuntimeError(f"Unexpected attention shape={tuple(attn.shape)}")
    q_idx = torch.as_tensor(list(query_positions), device=attn.device, dtype=torch.long)
    v_idx = torch.as_tensor(list(visual_positions), device=attn.device, dtype=torch.long)
    selected = attn[0].index_select(1, q_idx).index_select(2, v_idx).float()  # H,Q,V
    raw_mass = selected.sum(dim=-1)
    normalized = selected / raw_mass[..., None].clamp_min(EPS)
    valid = raw_mass > EPS
    if bool(valid.any()):
        w = normalized[valid].mean(dim=0)
    else:
        w = torch.ones(len(visual_positions), device=attn.device, dtype=torch.float32)
    w = w / w.sum().clamp_min(EPS)
    return w, float(raw_mass.mean().detach().cpu().item())


def weighted_visual_pair(
    visual_matrix: torch.Tensor,
    w_sub: torch.Tensor,
    w_ref: torch.Tensor,
) -> torch.Tensor:
    if visual_matrix.ndim != 2:
        raise RuntimeError(f"Expected [V,D], got {tuple(visual_matrix.shape)}")
    if int(visual_matrix.shape[0]) != int(w_sub.numel()):
        raise RuntimeError(
            f"Visual count mismatch: features={visual_matrix.shape[0]} locator={w_sub.numel()}"
        )
    x = visual_matrix.float()
    ws = w_sub.to(device=x.device, dtype=torch.float32)
    wr = w_ref.to(device=x.device, dtype=torch.float32)
    sub = torch.einsum("v,vd->d", ws, x)
    ref = torch.einsum("v,vd->d", wr, x)
    return sub - ref


def resolve_input_norm(layer: torch.nn.Module) -> torch.nn.Module:
    for name in ("input_layernorm", "self_attn_layer_norm", "attention_norm", "ln_1"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise RuntimeError(f"Could not resolve input norm in {type(layer)}")


def resolve_num_heads(model: Any, attn_module: torch.nn.Module) -> int:
    for name in ("num_heads", "num_attention_heads", "n_heads"):
        value = getattr(attn_module, name, None)
        if value is not None:
            return int(value)
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and getattr(cfg, "num_attention_heads", None) is not None:
            return int(cfg.num_attention_heads)
    raise RuntimeError("Could not determine num_attention_heads")


def resolve_head_dim(attn_module: torch.nn.Module, num_heads: int) -> int:
    value = getattr(attn_module, "head_dim", None)
    if value is not None:
        return int(value)
    q_proj = trace.resolve_projection(attn_module, ["q_proj", "query_proj"])
    if q_proj is not None:
        out_features = getattr(q_proj, "out_features", None)
        if out_features is None and hasattr(q_proj, "weight"):
            out_features = int(q_proj.weight.shape[0])
        if out_features is not None and int(out_features) % num_heads == 0:
            return int(out_features) // num_heads
    raise RuntimeError("Could not determine head_dim")


def linear_no_bias(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight):
        return F.linear(x, weight, bias=None)
    y = module(x)
    bias = getattr(module, "bias", None)
    if torch.is_tensor(bias):
        y = y - bias
    return y


def visual_contribution_to_role(
    *,
    model: Any,
    layer: torch.nn.Module,
    layer_input_state: torch.Tensor,
    attention_weights: torch.Tensor,
    query_positions: Sequence[int],
    visual_positions: Sequence[int],
) -> torch.Tensor:
    """Reconstruct visual-only self-attention contribution after o_proj."""
    attn_module, _ = trace.resolve_self_attention(layer)
    v_proj = trace.resolve_projection(attn_module, ["v_proj", "value_proj"])
    o_proj = trace.resolve_projection(attn_module, ["o_proj", "out_proj", "output_proj"])
    if v_proj is None or o_proj is None:
        raise RuntimeError("Could not resolve v_proj/o_proj")
    num_heads = resolve_num_heads(model, attn_module)
    head_dim = resolve_head_dim(attn_module, num_heads)
    input_norm = resolve_input_norm(layer)
    with torch.inference_mode():
        normed = input_norm(layer_input_state)
        values_flat = v_proj(normed)  # [1,T,Hkv*Dh]
    vdim = int(values_flat.shape[-1])
    if vdim % head_dim != 0:
        raise RuntimeError(f"v_proj width={vdim} not divisible by head_dim={head_dim}")
    n_kv_heads = vdim // head_dim
    v_idx = torch.as_tensor(list(visual_positions), device=values_flat.device, dtype=torch.long)
    values = (
        values_flat[0]
        .index_select(0, v_idx)
        .view(len(visual_positions), n_kv_heads, head_dim)
        .permute(1, 0, 2)
        .contiguous()
    )  # Hkv,V,Dh
    if n_kv_heads != num_heads:
        if num_heads % n_kv_heads != 0:
            raise RuntimeError(f"num_heads={num_heads}, n_kv_heads={n_kv_heads}")
        values = values.repeat_interleave(num_heads // n_kv_heads, dim=0)
    attn = attention_weights.to(device=values.device, dtype=torch.float32)
    if int(attn.shape[1]) != num_heads:
        raise RuntimeError(f"attention heads={attn.shape[1]} != expected {num_heads}")
    q_idx = torch.as_tensor(list(query_positions), device=attn.device, dtype=torch.long)
    av_idx = torch.as_tensor(list(visual_positions), device=attn.device, dtype=torch.long)
    selected = attn[0].index_select(1, q_idx).index_select(2, av_idx).float()  # H,Q,V
    head_role = torch.einsum("hqv,hvd->hqd", selected, values.float())
    head_mean = head_role.mean(dim=1)  # H,Dh
    flat = head_mean.reshape(1, num_heads * head_dim)
    weight = getattr(o_proj, "weight", None)
    if torch.is_tensor(weight):
        flat = flat.to(device=weight.device, dtype=weight.dtype)
    with torch.inference_mode():
        projected = linear_no_bias(o_proj, flat)
    return projected[0].detach().float().cpu()


def stratified_split(y: np.ndarray, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train: List[int] = []
    test: List[int] = []
    for relation in RELATIONS:
        ids = np.flatnonzero(y == relation).tolist()
        rng.shuffle(ids)
        n_train = int(round(len(ids) * train_ratio))
        n_train = min(max(n_train, 1), max(len(ids) - 1, 1))
        train.extend(ids[:n_train])
        test.extend(ids[n_train:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def fit_codebook(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    center = X.mean(axis=0)
    Xc = X - center
    directions = []
    for relation in RELATIONS:
        mask = y == relation
        if not bool(mask.any()):
            raise RuntimeError(f"No TRAIN samples for {relation}")
        d = Xc[mask].mean(axis=0)
        d = d / max(float(np.linalg.norm(d)), EPS)
        directions.append(d)
    return center, np.stack(directions, axis=0)


def evaluate_stage(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    center, dirs = fit_codebook(X[train_idx], y[train_idx])
    Xt = normalize_rows(X[test_idx] - center)
    pred = np.argmax(Xt @ dirs.T, axis=1)
    gt = np.asarray([RELATIONS.index(str(x)) for x in y[test_idx]], dtype=np.int64)
    row: Dict[str, Any] = {"accuracy": float(np.mean(pred == gt)), "n_test": int(len(test_idx))}
    for ri, relation in enumerate(RELATIONS):
        mask = gt == ri
        row[f"{relation}_accuracy"] = float(np.mean(pred[mask] == gt[mask])) if bool(mask.any()) else float("nan")
    row["lr_direction_cosine"] = float(np.dot(dirs[0], dirs[1]))
    row["ab_direction_cosine"] = float(np.dot(dirs[2], dirs[3]))
    return row


def evaluate_all_stages(
    stage_arrays: Mapping[str, np.ndarray], labels: np.ndarray, train_ratio: float, repeats: int, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    repeat_rows: List[Dict[str, Any]] = []
    for rep in range(repeats):
        tr, te = stratified_split(labels, train_ratio, seed + rep)
        for stage, X in stage_arrays.items():
            metrics = evaluate_stage(X, labels, tr, te)
            repeat_rows.append({"repeat": rep, "stage": stage, "train_n": len(tr), "test_n": len(te), **metrics})
    summary_rows: List[Dict[str, Any]] = []
    for stage in stage_arrays.keys():
        rows = [r for r in repeat_rows if r["stage"] == stage]
        summary_rows.append({
            "stage": stage,
            "accuracy_mean": safe_mean(r["accuracy"] for r in rows),
            "accuracy_std": safe_std(r["accuracy"] for r in rows),
            "left_accuracy": safe_mean(r["left_accuracy"] for r in rows),
            "right_accuracy": safe_mean(r["right_accuracy"] for r in rows),
            "above_accuracy": safe_mean(r["above_accuracy"] for r in rows),
            "below_accuracy": safe_mean(r["below_accuracy"] for r in rows),
            "lr_direction_cosine": safe_mean(r["lr_direction_cosine"] for r in rows),
            "ab_direction_cosine": safe_mean(r["ab_direction_cosine"] for r in rows),
        })
    return repeat_rows, summary_rows


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
    return trace.label_token_id_variants(tokenizer)


def restricted_relation_prediction(logits_last: torch.Tensor, token_map: Mapping[str, Sequence[int]]) -> str:
    scores = []
    for relation in RELATIONS:
        ids = torch.as_tensor(list(token_map[relation]), device=logits_last.device, dtype=torch.long)
        scores.append(float(logits_last.index_select(0, ids).max().detach().float().cpu().item()))
    return RELATIONS[int(np.argmax(np.asarray(scores)))]


def squeeze_visual_matrix(x: torch.Tensor, expected_visual: int, name: str) -> torch.Tensor:
    work = x
    if work.ndim == 3:
        if int(work.shape[0]) != 1:
            raise RuntimeError(f"{name}: expected batch=1, got {tuple(work.shape)}")
        work = work[0]
    if work.ndim != 2:
        raise RuntimeError(f"{name}: expected [V,D], got {tuple(work.shape)}")
    if int(work.shape[0]) != expected_visual:
        raise RuntimeError(
            f"{name}: projector visual count={work.shape[0]} != LLM visual positions={expected_visual}. "
            "Do not silently align mismatched visual tokenizations."
        )
    return work


def relation_from_locator(
    model: Any,
    batch: Mapping[str, Any],
    w_sub: torch.Tensor,
    w_ref: torch.Tensor,
    gt: str,
) -> Dict[str, Any]:
    coords = trace.visual_coordinates(model, dict(batch), int(w_sub.numel()), w_sub.device)
    if coords is None:
        return {
            "locator_pred": "",
            "locator_correct": float("nan"),
            "locator_axis_conf": float("nan"),
            "locator_dx": float("nan"),
            "locator_dy": float("nan"),
        }
    sxy = torch.einsum("v,vd->d", w_sub.float(), coords.float())
    rxy = torch.einsum("v,vd->d", w_ref.float(), coords.float())
    dx = float((sxy[0] - rxy[0]).detach().cpu().item())
    dy = float((sxy[1] - rxy[1]).detach().cpu().item())
    pred, conf = trace.relation_from_centroids(dx, dy)
    return {
        "locator_pred": pred,
        "locator_correct": float(pred == gt),
        "locator_axis_conf": float(conf),
        "locator_dx": dx,
        "locator_dy": dy,
    }


def extract_one(
    *,
    model: Any,
    processor: Any,
    projector: torch.nn.Module,
    decoder_layers: Sequence[torch.nn.Module],
    selected_layers: Sequence[int],
    localize_layer: int,
    rec: Any,
    device: torch.device,
    token_map: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    gt = norm_relation(rec.relation)
    if gt not in RELATIONS:
        raise RuntimeError(f"Invalid relation={rec.relation!r}")
    question = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
    image = Image.open(rec.image_path).convert("RGB")
    real_batch = no_batch = None
    try:
        real_batch, real_ids, real_sub, real_ref, visual_pos = make_real_batch(
            model, processor, rec, question, image, device
        )
        no_batch, no_ids, no_sub, no_ref = make_noimage_batch(
            processor, rec, question, device
        )

        with ProjectorCapture(projector) as pc:
            with torch.inference_mode():
                real_outputs = model(
                    **real_batch,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
            pc.validate()
            projector_pre = pc.input
            projector_post = pc.output

        with torch.inference_mode():
            no_outputs = model(
                **no_batch,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )

        real_states = base.hidden_tuple(real_outputs)
        no_states = base.hidden_tuple(no_outputs)
        attentions = trace.attention_tuple(real_outputs)
        n_layers = len(decoder_layers)
        if len(real_states) < n_layers + 1 or len(no_states) < n_layers + 1:
            raise RuntimeError(
                f"hidden states too short: real={len(real_states)} noimg={len(no_states)} decoder={n_layers}"
            )
        if len(attentions) != n_layers:
            raise RuntimeError(f"attentions={len(attentions)} != decoder={n_layers}")
        if int(real_states[0].shape[1]) != len(real_ids):
            raise RuntimeError(
                f"real hidden seq={real_states[0].shape[1]} != input_ids={len(real_ids)}"
            )

        n_visual = len(visual_pos)
        loc_attn = attentions[localize_layer]
        w_sub, loc_sub_mass = object_visual_attention_weights(loc_attn, real_sub, visual_pos)
        w_ref, loc_ref_mass = object_visual_attention_weights(loc_attn, real_ref, visual_pos)
        locator_diag = relation_from_locator(model, real_batch, w_sub, w_ref, gt)

        pre = squeeze_visual_matrix(projector_pre, n_visual, "projector_pre")
        post = squeeze_visual_matrix(projector_post, n_visual, "projector_post")
        vectors: Dict[str, np.ndarray] = {}
        vectors["projector_pre"] = weighted_visual_pair(pre, w_sub.cpu(), w_ref.cpu()).numpy().astype(np.float32)
        vectors["projector_post"] = weighted_visual_pair(post, w_sub.cpu(), w_ref.cpu()).numpy().astype(np.float32)

        v_idx = torch.as_tensor(visual_pos, device=real_states[0].device, dtype=torch.long)
        llm_input_visual = real_states[0][0].index_select(0, v_idx)
        vectors["llm_input_visual"] = (
            weighted_visual_pair(llm_input_visual, w_sub, w_ref)
            .detach().float().cpu().numpy().astype(np.float32)
        )

        layer_rows: List[Dict[str, Any]] = []
        for li in selected_layers:
            state_in = real_states[li]
            state_out = real_states[li + 1]
            no_state_out = no_states[li + 1]
            visual_matrix = state_out[0].index_select(0, v_idx)
            q_visual = weighted_visual_pair(visual_matrix, w_sub, w_ref)
            vectors[f"visual_L{li}"] = q_visual.detach().float().cpu().numpy().astype(np.float32)

            c_sub = visual_contribution_to_role(
                model=model,
                layer=decoder_layers[li],
                layer_input_state=state_in,
                attention_weights=attentions[li],
                query_positions=real_sub,
                visual_positions=visual_pos,
            )
            c_ref = visual_contribution_to_role(
                model=model,
                layer=decoder_layers[li],
                layer_input_state=state_in,
                attention_weights=attentions[li],
                query_positions=real_ref,
                visual_positions=visual_pos,
            )
            q_transfer = c_sub - c_ref
            vectors[f"transfer_L{li}"] = q_transfer.numpy().astype(np.float32)

            real_pair = pool_text(state_out, real_sub, args.pool) - pool_text(state_out, real_ref, args.pool)
            no_pair = pool_text(no_state_out, no_sub, args.pool) - pool_text(no_state_out, no_ref, args.pool)
            q_text = real_pair.detach().float().cpu() - no_pair.detach().float().cpu()
            vectors[f"text_residual_L{li}"] = q_text.numpy().astype(np.float32)

            _, sub_mass = object_visual_attention_weights(attentions[li], real_sub, visual_pos)
            _, ref_mass = object_visual_attention_weights(attentions[li], real_ref, visual_pos)
            layer_rows.append({
                "sid": int(rec.sid),
                "gt": gt,
                "layer": li,
                "subject_visual_attention_mass": sub_mass,
                "reference_visual_attention_mass": ref_mass,
                "mean_visual_attention_mass": 0.5 * (sub_mass + ref_mass),
                "visual_vector_norm": float(torch.linalg.vector_norm(q_visual.float()).detach().cpu().item()),
                "transfer_vector_norm": float(torch.linalg.vector_norm(q_transfer.float()).detach().cpu().item()),
                "text_residual_norm": float(torch.linalg.vector_norm(q_text.float()).detach().cpu().item()),
            })

        real_logits = real_outputs.logits
        no_logits = no_outputs.logits
        real_pred = restricted_relation_prediction(real_logits[0, -1], token_map)
        no_pred = restricted_relation_prediction(no_logits[0, -1], token_map)
        sample_row = {
            "sid": int(rec.sid),
            "image_id": str(rec.image_id),
            "subject": str(rec.subject),
            "reference": str(rec.reference),
            "gt": gt,
            "real_restricted_pred": real_pred,
            "real_restricted_correct": int(real_pred == gt),
            "noimage_restricted_pred": no_pred,
            "noimage_restricted_correct": int(no_pred == gt),
            "n_visual_tokens": n_visual,
            "localize_layer": localize_layer,
            "localizer_subject_visual_mass": loc_sub_mass,
            "localizer_reference_visual_mass": loc_ref_mass,
            **locator_diag,
        }
        return vectors, sample_row, layer_rows
    finally:
        if real_batch is not None:
            del real_batch
        if no_batch is not None:
            del no_batch
        image.close()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must lie in (0,1)")

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records, audit = base.load_records(args.dataset, Path(args.data_root), args.max_samples)
    records = [r for r in records if norm_relation(r.relation) in RELATIONS]
    if not records:
        raise RuntimeError("No usable 4-way spatial records")
    print(f"[{args.dataset}] N={len(records)} counts={dict(Counter(norm_relation(r.relation) for r in records))}")

    spec = base.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers={transformers.__version__} has no {spec.model_class}")
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": "eager",
    }
    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = trace.resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    selected_layers = parse_layers(args.layers, n_layers)
    localize_layer = resolve_localize_layer(args.model, args.localize_layer, n_layers)
    projector, projector_path = resolve_projector(model)
    token_map = relation_token_variants(processor.tokenizer)

    print("\n" + "=" * 150)
    print("LLaVA VISUAL ENTRY / VISUAL->TEXT TRANSFER DIAGNOSTIC")
    print("=" * 150)
    print(f"model={args.model} repo={spec.repo_id}")
    print(f"decoder={decoder_path} layers={n_layers}")
    print(f"projector={projector_path}")
    print(f"scan_layers={selected_layers}")
    print(f"fixed_visual_locator_layer=L{localize_layer}")
    print(f"split={args.train_ratio:.2f}/{1.0-args.train_ratio:.2f} repeats={args.repeats}")
    print("=" * 150)

    sample_rows: List[Dict[str, Any]] = []
    layer_metric_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    stage_lists: Dict[str, List[np.ndarray]] = {}
    labels: List[str] = []
    sids: List[int] = []

    for rec in tqdm(records, desc=f"extract:{args.model}"):
        try:
            vectors, sample_row, layer_rows = extract_one(
                model=model,
                processor=processor,
                projector=projector,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                localize_layer=localize_layer,
                rec=rec,
                device=device,
                token_map=token_map,
                args=args,
            )
            if not stage_lists:
                stage_lists = {name: [] for name in vectors.keys()}
            if set(vectors.keys()) != set(stage_lists.keys()):
                raise RuntimeError("Stage set changed between samples")
            for name, vec in vectors.items():
                stage_lists[name].append(np.asarray(vec, dtype=np.float32))
            sample_rows.append(sample_row)
            layer_metric_rows.extend(layer_rows)
            labels.append(str(sample_row["gt"]))
            sids.append(int(sample_row["sid"]))
        except Exception as exc:
            errors.append({
                "sid": int(rec.sid),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            })
            tqdm.write(f"[ERROR sid={rec.sid}] {type(exc).__name__}: {exc}")
        finally:
            cleanup()

    if not sample_rows:
        raise RuntimeError("No samples extracted successfully")

    y = np.asarray(labels, dtype=object)
    stage_arrays = {name: np.stack(vs, axis=0).astype(np.float32) for name, vs in stage_lists.items()}
    repeat_rows, summary_rows = evaluate_all_stages(
        stage_arrays, y, args.train_ratio, args.repeats, args.seed
    )

    layer_summary_rows: List[Dict[str, Any]] = []
    for li in selected_layers:
        rows = [r for r in layer_metric_rows if int(r["layer"]) == li]
        layer_summary_rows.append({
            "layer": li,
            "N": len(rows),
            "subject_visual_attention_mass": safe_mean(r["subject_visual_attention_mass"] for r in rows),
            "reference_visual_attention_mass": safe_mean(r["reference_visual_attention_mass"] for r in rows),
            "mean_visual_attention_mass": safe_mean(r["mean_visual_attention_mass"] for r in rows),
            "visual_vector_norm": safe_mean(r["visual_vector_norm"] for r in rows),
            "transfer_vector_norm": safe_mean(r["transfer_vector_norm"] for r in rows),
            "text_residual_norm": safe_mean(r["text_residual_norm"] for r in rows),
        })

    write_csv(outdir / "samples.csv", sample_rows)
    write_csv(outdir / "probe_repeats.csv", repeat_rows)
    write_csv(outdir / "probe_summary.csv", summary_rows)
    write_csv(outdir / "layer_transfer_metrics.csv", layer_metric_rows)
    write_csv(outdir / "layer_transfer_summary.csv", layer_summary_rows)
    (outdir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_vectors:
        arrays: Dict[str, Any] = {
            "sample_index": np.asarray(sids, dtype=np.int64),
            "relation": y,
            "selected_layers": np.asarray(selected_layers, dtype=np.int32),
            "localize_layer": np.asarray([localize_layer], dtype=np.int32),
        }
        for name, arr in stage_arrays.items():
            arrays[name] = arr.astype(np.float16)
        np.savez_compressed(outdir / "stage_vectors.npz", **arrays)

    summary = {row["stage"]: row for row in summary_rows}
    ordered = ["projector_pre", "projector_post", "llm_input_visual"]
    for li in selected_layers:
        ordered += [f"visual_L{li}", f"transfer_L{li}", f"text_residual_L{li}"]

    print("\n" + "=" * 150)
    print("STAGE-WISE DIRECTION READOUT")
    print("=" * 150)
    for stage in ordered:
        if stage not in summary:
            continue
        r = summary[stage]
        print(
            f"{stage:24s} | acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} | "
            f"L/R/A/B={r['left_accuracy']:.3f}/{r['right_accuracy']:.3f}/"
            f"{r['above_accuracy']:.3f}/{r['below_accuracy']:.3f}"
        )

    loc_vals = [float(r["locator_correct"]) for r in sample_rows if math.isfinite(float(r["locator_correct"]))]
    real_acc = safe_mean(r["real_restricted_correct"] for r in sample_rows)
    no_acc = safe_mean(r["noimage_restricted_correct"] for r in sample_rows)
    print("-" * 150)
    print(f"fixed locator L{localize_layer} centroid acc = {safe_mean(loc_vals):.4f} (N={len(loc_vals)})")
    print(f"REAL restricted first-token acc    = {real_acc:.4f}")
    print(f"NOIMAGE restricted first-token acc = {no_acc:.4f}")

    print("\nATTENTION / TRANSFER MAGNITUDE")
    print("-" * 150)
    for r in layer_summary_rows:
        print(
            f"L{int(r['layer']):02d} | visual_mass={r['mean_visual_attention_mass']:.5f} | "
            f"||q_visual||={r['visual_vector_norm']:.4f} | "
            f"||q_transfer||={r['transfer_vector_norm']:.4f} | "
            f"||q_text_res||={r['text_residual_norm']:.4f}"
        )
    print("=" * 150)

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "model": args.model,
        "repo_id": spec.repo_id,
        "n_success": len(sample_rows),
        "n_errors": len(errors),
        "decoder_path": decoder_path,
        "decoder_layers": n_layers,
        "projector_path": projector_path,
        "selected_layers": selected_layers,
        "localize_layer": localize_layer,
        "train_ratio": args.train_ratio,
        "repeats": args.repeats,
        "seed": args.seed,
        "locator_centroid_accuracy": safe_mean(loc_vals),
        "real_restricted_accuracy": real_acc,
        "noimage_restricted_accuracy": no_acc,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[saved] {outdir / 'probe_summary.csv'}")
    print(f"[saved] {outdir / 'layer_transfer_summary.csv'}")
    print(f"[saved] {outdir / 'samples.csv'}")
    if args.save_vectors:
        print(f"[saved] {outdir / 'stage_vectors.npz'}")


if __name__ == "__main__":
    main()
