#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
逐层分析 object token 的视觉 grounding。

每层计算两套 subject/reference → visual map：
1) hidden similarity map：object hidden 与 visual hidden 的 cosine similarity；
2) attention map：object text query 对 visual token keys 的注意力。

输出：
- hidden-sim centroid accuracy / macro accuracy；
- attention-centroid accuracy / macro accuracy；
- entropy、top-k mass、compactness、subject/reference overlap、
  centroid separation、相邻层 drift、visual attention mass、head agreement；
- 自动在中层范围内选择 centroid accuracy 峰值，并打印峰值前后若干层；
- A/B/C 分组对比和 A-B Cohen's d。

依赖同目录中的：
    run_spatial_repair_three_experiments_v1.py
    trace_centroid_generation_groups_v2_1.py

不使用 GT box。GT relation 只用于事后计算 accuracy 和分组统计。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

try:
    import run_spatial_repair_three_experiments_v1 as base
except Exception as exc:
    raise SystemExit(
        "Place run_spatial_repair_three_experiments_v1.py in the same directory.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "object-visual-layer-analysis-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: i for i, name in enumerate(RELATIONS)}
GROUP_A = base.GROUP_A
GROUP_B = base.GROUP_B
GROUP_C = base.GROUP_C
GROUP_D = base.GROUP_D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", default="qwen-3b,qwen-7b")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--input-root",
        default="output/three_group_transfer_fresh/coco",
    )
    p.add_argument(
        "--output-root",
        default="output/object_visual_attention_layer_analysis/coco",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--groups", default="A,B,C")
    p.add_argument("--max-per-group", type=int, default=None)
    p.add_argument("--sid", type=int, default=None)

    p.add_argument("--similarity-map", choices=["softmax", "relu"], default="softmax")
    p.add_argument("--similarity-temperature", type=float, default=0.07)
    p.add_argument("--top-fraction", type=float, default=0.10)

    p.add_argument("--middle-start", type=float, default=0.30)
    p.add_argument("--middle-end", type=float, default=0.80)
    p.add_argument(
        "--peak-source",
        choices=[
            "similarity_macro",
            "similarity_accuracy",
            "attention_macro",
            "attention_accuracy",
        ],
        default="similarity_macro",
    )
    p.add_argument("--neighbor-radius", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=20)
    p.add_argument(
        "--core-module",
        default="trace_centroid_generation_groups_v2_1",
    )
    return p.parse_args()


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (
        (len(a) - 1) * float(np.var(a, ddof=1))
        + (len(b) - 1) * float(np.var(b, ddof=1))
    ) / max(1, len(a) + len(b) - 2)
    std = math.sqrt(max(pooled, 0.0))
    return 0.0 if std <= 1e-12 else float((np.mean(a) - np.mean(b)) / std)


def macro_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    values = []
    for code in range(4):
        mask = gt == code
        if np.any(mask):
            values.append(float(np.mean(pred[mask] == gt[mask])))
    return float(np.mean(values)) if values else float("nan")


def entropy(p: torch.Tensor) -> float:
    p = p.float() / p.float().sum().clamp_min(1e-12)
    n = int(p.numel())
    if n <= 1:
        return 0.0
    h = -(p * torch.log(p.clamp_min(1e-12))).sum() / math.log(n)
    return float(h.detach().cpu())


def topk_mass(p: torch.Tensor, fraction: float) -> float:
    k = max(1, min(int(p.numel()), int(math.ceil(p.numel() * fraction))))
    return float(torch.topk(p.float(), k=k).values.sum().detach().cpu())


def centroid(p: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    p = p.float() / p.float().sum().clamp_min(1e-12)
    return (p[:, None] * xy.float()).sum(dim=0)


def compactness(p: torch.Tensor, xy: torch.Tensor, c: torch.Tensor) -> float:
    p = p.float() / p.float().sum().clamp_min(1e-12)
    d2 = ((xy.float() - c[None, :].float()) ** 2).sum(dim=-1)
    return float(((p * d2).sum() / 2.0).detach().cpu())


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() / a.float().sum().clamp_min(1e-12)
    b = b.float() / b.float().sum().clamp_min(1e-12)
    return float(torch.minimum(a, b).sum().detach().cpu())


def relation_from_centroids(s: torch.Tensor, r: torch.Tensor) -> int:
    dx = float(s[0] - r[0])
    dy = float(s[1] - r[1])
    if abs(dx) >= abs(dy):
        return REL_TO_ID["left"] if dx < 0 else REL_TO_ID["right"]
    return REL_TO_ID["above"] if dy < 0 else REL_TO_ID["below"]


def gt_axis_margin(s: torch.Tensor, r: torch.Tensor, gt: int) -> float:
    sx, sy = float(s[0]), float(s[1])
    rx, ry = float(r[0]), float(r[1])
    if gt == REL_TO_ID["left"]:
        return rx - sx
    if gt == REL_TO_ID["right"]:
        return sx - rx
    if gt == REL_TO_ID["above"]:
        return ry - sy
    return sy - ry


def factor_pairs(n: int) -> List[Tuple[int, int]]:
    pairs = []
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            pairs.append((h, n // h))
            if h != n // h:
                pairs.append((n // h, h))
    return pairs


def merge_sizes(model: Any, processor: Any) -> List[int]:
    out = []
    for obj in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        getattr(processor, "image_processor", None),
    ):
        if obj is None:
            continue
        for name in ("spatial_merge_size", "merge_size", "spatial_merge"):
            value = getattr(obj, name, None)
            if isinstance(value, (int, np.integer)) and int(value) > 0:
                out.append(int(value))
    out.extend([1, 2, 4])
    return list(dict.fromkeys(out))


def infer_grid(
    visual_count: int,
    batch: Mapping[str, Any],
    image_size: Tuple[int, int],
    model: Any,
    processor: Any,
) -> Tuple[int, int, str]:
    grid = batch.get("image_grid_thw")
    if torch.is_tensor(grid) and grid.numel() >= 3:
        _, raw_h, raw_w = [int(v) for v in grid.detach().cpu().reshape(-1, 3)[0].tolist()]
        for m in merge_sizes(model, processor):
            if raw_h % m == 0 and raw_w % m == 0:
                h, w = raw_h // m, raw_w // m
                if h * w == visual_count:
                    return h, w, f"image_grid_thw/merge{m}"
        if raw_h * raw_w == visual_count:
            return raw_h, raw_w, "image_grid_thw"

    width_px, height_px = image_size
    target = float(width_px) / max(float(height_px), 1.0)
    pairs = factor_pairs(visual_count)
    if not pairs:
        raise RuntimeError(f"Cannot factor visual_count={visual_count}")
    h, w = min(
        pairs,
        key=lambda pair: abs(
            math.log(max(pair[1] / max(pair[0], 1), 1e-8))
            - math.log(max(target, 1e-8))
        ),
    )
    return int(h), int(w), "factor/aspect"


def make_xy(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def hidden_map(
    hidden: torch.Tensor,
    query_positions: Sequence[int],
    visual_positions: Sequence[int],
    mode: str,
    temperature: float,
) -> torch.Tensor:
    q_idx = torch.as_tensor(query_positions, device=hidden.device, dtype=torch.long)
    v_idx = torch.as_tensor(visual_positions, device=hidden.device, dtype=torch.long)
    q = hidden[0].index_select(0, q_idx).mean(dim=0)
    v = hidden[0].index_select(0, v_idx)
    scores = F.normalize(v.float(), dim=-1) @ F.normalize(q.float(), dim=-1)
    if mode == "softmax":
        return torch.softmax(scores / max(temperature, 1e-6), dim=-1)
    weights = torch.relu(scores)
    if float(weights.sum()) <= 1e-12:
        weights = torch.ones_like(weights)
    return weights / weights.sum().clamp_min(1e-12)


def attention_map(
    weights: torch.Tensor,
    query_positions: Sequence[int],
    visual_positions: Sequence[int],
) -> Tuple[torch.Tensor, float, float]:
    q_idx = torch.as_tensor(query_positions, device=weights.device, dtype=torch.long)
    v_idx = torch.as_tensor(visual_positions, device=weights.device, dtype=torch.long)
    selected = weights[0].index_select(1, q_idx).index_select(2, v_idx).float()  # H,T,V
    raw_mass = selected.sum(dim=-1)
    normalized = selected / raw_mass[..., None].clamp_min(1e-12)
    valid = raw_mass > 1e-12
    if torch.any(valid):
        agg = normalized[valid].mean(dim=0)
    else:
        agg = torch.ones(len(visual_positions), device=weights.device)
    agg = agg / agg.sum().clamp_min(1e-12)
    per_head = F.normalize(normalized.mean(dim=1), dim=-1)
    agreement = (per_head * F.normalize(agg, dim=-1)[None, :]).sum(dim=-1).mean()
    return agg, float(raw_mass.mean().detach().cpu()), float(agreement.detach().cpu())


@dataclass
class LayerMetrics:
    sim_pred: int
    attn_pred: int
    sim_s_centroid: np.ndarray
    sim_r_centroid: np.ndarray
    attn_s_centroid: np.ndarray
    attn_r_centroid: np.ndarray
    values: Dict[str, float]


class Capture:
    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        model: Any,
        processor: Any,
        sim_mode: str,
        sim_temperature: float,
        top_fraction: float,
    ) -> None:
        self.layers = list(layers)
        self.model = model
        self.processor = processor
        self.sim_mode = sim_mode
        self.sim_temperature = sim_temperature
        self.top_fraction = top_fraction
        self.handles = []
        self.reset()

        for layer_id, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise RuntimeError(f"Layer {layer_id} has no self_attn")
            self.handles.append(attn.register_forward_hook(
                self._attn_hook(layer_id), with_kwargs=True
            ))
            self.handles.append(layer.register_forward_hook(
                self._layer_hook(layer_id), with_kwargs=True
            ))

    def reset(self) -> None:
        self.prompt_spec = None
        self.batch = None
        self.image_size = None
        self.gt_code = None
        self.positions = None
        self.xy = None
        self.grid_shape = None
        self.grid_source = None
        self.pending_attn = {}
        self.results = {}

    def configure(
        self,
        prompt_spec: base.PromptPositionSpec,
        batch: Mapping[str, Any],
        image_size: Tuple[int, int],
        gt_code: int,
    ) -> None:
        self.reset()
        self.prompt_spec = prompt_spec
        self.batch = batch
        self.image_size = image_size
        self.gt_code = gt_code

    def close(self) -> None:
        for handle in self.handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()
        self.reset()

    def _ensure_geometry(self, seq_len: int, device: torch.device) -> None:
        if self.positions is not None:
            return
        if self.prompt_spec is None or self.batch is None or self.image_size is None:
            raise RuntimeError("Capture not configured")
        self.positions = base.expand_positions(self.prompt_spec, hidden_length=seq_len)
        n_visual = len(self.positions.visual_positions)
        h, w, source = infer_grid(
            n_visual, self.batch, self.image_size, self.model, self.processor
        )
        if h * w != n_visual:
            raise RuntimeError(f"Grid {h}x{w} != visual_count {n_visual}")
        self.xy = make_xy(h, w, device)
        self.grid_shape = (h, w)
        self.grid_source = source

    @staticmethod
    def _find_attn(output: Any) -> Optional[torch.Tensor]:
        if not isinstance(output, (tuple, list)):
            return None
        tensors = [x for x in output[1:] if torch.is_tensor(x) and x.ndim == 4]
        if not tensors:
            return None
        return max(tensors, key=lambda x: int(x.shape[-2]) * int(x.shape[-1]))

    def _attn_hook(self, layer_id: int):
        def hook(module, args, kwargs, output):
            weights = self._find_attn(output)
            if weights is None:
                raise RuntimeError(
                    f"L{layer_id} did not expose eager attention weights"
                )
            self._ensure_geometry(int(weights.shape[-1]), weights.device)
            assert self.positions is not None
            s_map, s_mass, s_agree = attention_map(
                weights, self.positions.subject_positions, self.positions.visual_positions
            )
            r_map, r_mass, r_agree = attention_map(
                weights, self.positions.reference_positions, self.positions.visual_positions
            )
            self.pending_attn[layer_id] = {
                "s_map": s_map.detach(),
                "r_map": r_map.detach(),
                "s_mass": s_mass,
                "r_mass": r_mass,
                "s_agree": s_agree,
                "r_agree": r_agree,
            }
            return output
        return hook

    def _layer_hook(self, layer_id: int):
        def hook(module, args, kwargs, output):
            hidden = base.output_first_tensor(output)
            self._ensure_geometry(int(hidden.shape[1]), hidden.device)
            assert self.positions is not None and self.xy is not None
            if layer_id not in self.pending_attn:
                raise RuntimeError(f"Missing attention map at L{layer_id}")

            sim_s = hidden_map(
                hidden, self.positions.subject_positions, self.positions.visual_positions,
                self.sim_mode, self.sim_temperature
            )
            sim_r = hidden_map(
                hidden, self.positions.reference_positions, self.positions.visual_positions,
                self.sim_mode, self.sim_temperature
            )
            attn_data = self.pending_attn.pop(layer_id)
            attn_s = attn_data["s_map"].to(hidden.device)
            attn_r = attn_data["r_map"].to(hidden.device)
            xy = self.xy.to(hidden.device)

            sim_sc, sim_rc = centroid(sim_s, xy), centroid(sim_r, xy)
            attn_sc, attn_rc = centroid(attn_s, xy), centroid(attn_r, xy)
            gt = int(self.gt_code)

            values = {
                "sim_s_entropy": entropy(sim_s),
                "sim_r_entropy": entropy(sim_r),
                "sim_s_topk": topk_mass(sim_s, self.top_fraction),
                "sim_r_topk": topk_mass(sim_r, self.top_fraction),
                "sim_s_compact": compactness(sim_s, xy, sim_sc),
                "sim_r_compact": compactness(sim_r, xy, sim_rc),
                "sim_overlap": overlap(sim_s, sim_r),
                "sim_separation": float(torch.linalg.vector_norm(sim_sc - sim_rc).detach().cpu()),
                "sim_gt_margin": gt_axis_margin(sim_sc, sim_rc, gt),
                "attn_s_entropy": entropy(attn_s),
                "attn_r_entropy": entropy(attn_r),
                "attn_s_topk": topk_mass(attn_s, self.top_fraction),
                "attn_r_topk": topk_mass(attn_r, self.top_fraction),
                "attn_s_compact": compactness(attn_s, xy, attn_sc),
                "attn_r_compact": compactness(attn_r, xy, attn_rc),
                "attn_overlap": overlap(attn_s, attn_r),
                "attn_separation": float(torch.linalg.vector_norm(attn_sc - attn_rc).detach().cpu()),
                "attn_gt_margin": gt_axis_margin(attn_sc, attn_rc, gt),
                "attn_s_visual_mass": float(attn_data["s_mass"]),
                "attn_r_visual_mass": float(attn_data["r_mass"]),
                "attn_s_head_agreement": float(attn_data["s_agree"]),
                "attn_r_head_agreement": float(attn_data["r_agree"]),
            }
            self.results[layer_id] = LayerMetrics(
                sim_pred=relation_from_centroids(sim_sc, sim_rc),
                attn_pred=relation_from_centroids(attn_sc, attn_rc),
                sim_s_centroid=sim_sc.detach().cpu().numpy().astype(np.float32),
                sim_r_centroid=sim_rc.detach().cpu().numpy().astype(np.float32),
                attn_s_centroid=attn_sc.detach().cpu().numpy().astype(np.float32),
                attn_r_centroid=attn_rc.detach().cpu().numpy().astype(np.float32),
                values=values,
            )
            return output
        return hook


VALUE_KEYS = (
    "sim_s_entropy", "sim_r_entropy", "sim_s_topk", "sim_r_topk",
    "sim_s_compact", "sim_r_compact", "sim_overlap", "sim_separation",
    "sim_gt_margin", "attn_s_entropy", "attn_r_entropy", "attn_s_topk",
    "attn_r_topk", "attn_s_compact", "attn_r_compact", "attn_overlap",
    "attn_separation", "attn_gt_margin", "attn_s_visual_mass",
    "attn_r_visual_mass", "attn_s_head_agreement", "attn_r_head_agreement",
)


def drift(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.full(s.shape[0], np.nan, dtype=np.float32)
    if s.shape[0] > 1:
        out[1:] = 0.5 * (
            np.linalg.norm(s[1:] - s[:-1], axis=-1)
            + np.linalg.norm(r[1:] - r[:-1], axis=-1)
        )
    return out


def sample_to_arrays(results: Mapping[int, LayerMetrics], n_layers: int) -> Dict[str, np.ndarray]:
    missing = [i for i in range(n_layers) if i not in results]
    if missing:
        raise RuntimeError(f"Missing layer metrics: {missing[:10]}")
    sim_s = np.stack([results[i].sim_s_centroid for i in range(n_layers)])
    sim_r = np.stack([results[i].sim_r_centroid for i in range(n_layers)])
    attn_s = np.stack([results[i].attn_s_centroid for i in range(n_layers)])
    attn_r = np.stack([results[i].attn_r_centroid for i in range(n_layers)])
    out = {
        "sim_pred": np.asarray([results[i].sim_pred for i in range(n_layers)], dtype=np.int16),
        "attn_pred": np.asarray([results[i].attn_pred for i in range(n_layers)], dtype=np.int16),
        "sim_s_centroid": sim_s,
        "sim_r_centroid": sim_r,
        "attn_s_centroid": attn_s,
        "attn_r_centroid": attn_r,
        "sim_drift": drift(sim_s, sim_r),
        "attn_drift": drift(attn_s, attn_r),
    }
    for key in VALUE_KEYS:
        out[key] = np.asarray([results[i].values[key] for i in range(n_layers)], dtype=np.float32)
    return out


def stack_samples(samples: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = set(samples[0])
    for sample in samples[1:]:
        keys &= set(sample)
    return {key: np.stack([sample[key] for sample in samples]) for key in sorted(keys)}


def pair_metrics(arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "sim_entropy": 0.5 * (arrays["sim_s_entropy"] + arrays["sim_r_entropy"]),
        "sim_topk": 0.5 * (arrays["sim_s_topk"] + arrays["sim_r_topk"]),
        "sim_compact": 0.5 * (arrays["sim_s_compact"] + arrays["sim_r_compact"]),
        "sim_overlap": arrays["sim_overlap"],
        "sim_separation": arrays["sim_separation"],
        "sim_gt_margin": arrays["sim_gt_margin"],
        "sim_drift": arrays["sim_drift"],
        "attn_entropy": 0.5 * (arrays["attn_s_entropy"] + arrays["attn_r_entropy"]),
        "attn_topk": 0.5 * (arrays["attn_s_topk"] + arrays["attn_r_topk"]),
        "attn_compact": 0.5 * (arrays["attn_s_compact"] + arrays["attn_r_compact"]),
        "attn_overlap": arrays["attn_overlap"],
        "attn_separation": arrays["attn_separation"],
        "attn_gt_margin": arrays["attn_gt_margin"],
        "attn_drift": arrays["attn_drift"],
        "attn_visual_mass": 0.5 * (arrays["attn_s_visual_mass"] + arrays["attn_r_visual_mass"]),
        "attn_head_agreement": 0.5 * (
            arrays["attn_s_head_agreement"] + arrays["attn_r_head_agreement"]
        ),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_summaries(
    arrays: Mapping[str, np.ndarray],
    gt: np.ndarray,
    groups: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics = pair_metrics(arrays)
    n_layers = arrays["sim_pred"].shape[1]
    layer_rows, group_rows, effect_rows = [], [], []

    for layer in range(n_layers):
        sim_pred = arrays["sim_pred"][:, layer]
        attn_pred = arrays["attn_pred"][:, layer]
        row = {
            "layer": layer,
            "similarity_accuracy": float(np.mean(sim_pred == gt)),
            "similarity_macro_accuracy": macro_accuracy(sim_pred, gt),
            "attention_accuracy": float(np.mean(attn_pred == gt)),
            "attention_macro_accuracy": macro_accuracy(attn_pred, gt),
        }
        for name, values in metrics.items():
            row[name] = safe_mean(values[:, layer])
        layer_rows.append(row)

        for group_name, short in (
            (GROUP_A, "A"), (GROUP_B, "B"), (GROUP_C, "C"), (GROUP_D, "D")
        ):
            mask = groups == group_name
            if not np.any(mask):
                continue
            grow = {
                "layer": layer,
                "group": short,
                "group_full": group_name,
                "n": int(mask.sum()),
                "similarity_accuracy": float(np.mean(sim_pred[mask] == gt[mask])),
                "similarity_macro_accuracy": macro_accuracy(sim_pred[mask], gt[mask]),
                "attention_accuracy": float(np.mean(attn_pred[mask] == gt[mask])),
                "attention_macro_accuracy": macro_accuracy(attn_pred[mask], gt[mask]),
            }
            for name, values in metrics.items():
                grow[name] = safe_mean(values[mask, layer])
            group_rows.append(grow)

        a_mask, b_mask = groups == GROUP_A, groups == GROUP_B
        if np.any(a_mask) and np.any(b_mask):
            for name, values in metrics.items():
                a = values[a_mask, layer]
                b = values[b_mask, layer]
                effect_rows.append({
                    "layer": layer,
                    "metric": name,
                    "A_mean": safe_mean(a),
                    "B_mean": safe_mean(b),
                    "A_minus_B": safe_mean(a) - safe_mean(b),
                    "cohen_d_A_minus_B": cohen_d(a, b),
                })

    return layer_rows, group_rows, effect_rows


def peak_key(source: str) -> str:
    return {
        "similarity_macro": "similarity_macro_accuracy",
        "similarity_accuracy": "similarity_accuracy",
        "attention_macro": "attention_macro_accuracy",
        "attention_accuracy": "attention_accuracy",
    }[source]


def choose_peak(rows: Sequence[Mapping[str, Any]], source: str, start_f: float, end_f: float):
    n = len(rows)
    start = max(0, min(n - 1, int(math.floor(start_f * n))))
    end = max(start + 1, min(n, int(math.ceil(end_f * n))))
    key = peak_key(source)
    best = max(rows[start:end], key=lambda row: (float(row[key]), -int(row["layer"])))
    return int(best["layer"]), start, end


def load_saved_centroid_layer(model_root: Path) -> Optional[int]:
    path = model_root / "fresh_group_cache" / "config.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        value = obj.get("centroid_selection", {}).get("layer")
        return None if value is None else int(value)
    except Exception:
        return None


def print_report(
    model_name: str,
    layer_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    peak: int,
    search_start: int,
    search_end: int,
    radius: int,
    saved_layer: Optional[int],
) -> str:
    left, right = max(0, peak - radius), min(len(layer_rows) - 1, peak + radius)
    lines = [
        "=" * 142,
        f"MODEL: {model_name}",
        f"middle search=L{search_start}-L{search_end-1}, selected peak=L{peak}, "
        f"saved centroid layer={'unknown' if saved_layer is None else 'L'+str(saved_layer)}",
        "=" * 142,
        "Layer SimAcc SimMacro AttnAcc AttnMacro SimEnt AttnEnt AttnTop10 "
        "AttnCompact AttnOverlap AttnSep AttnDrift",
    ]
    for layer in range(left, right + 1):
        row = layer_rows[layer]
        mark = "*" if layer == peak else " "
        lines.append(
            f"{mark}L{layer:02d}  {row['similarity_accuracy']:.3f}   "
            f"{row['similarity_macro_accuracy']:.3f}    {row['attention_accuracy']:.3f}   "
            f"{row['attention_macro_accuracy']:.3f}    {row['sim_entropy']:.3f}  "
            f"{row['attn_entropy']:.3f}    {row['attn_topk']:.3f}      "
            f"{row['attn_compact']:.4f}       {row['attn_overlap']:.3f}      "
            f"{row['attn_separation']:.3f}    {row['attn_drift']:.4f}"
        )

    lookup = {(int(r["layer"]), str(r["group"])): r for r in group_rows}
    lines += [
        "",
        "A/B/C around selected peak",
        "Layer Group N SimAcc AttnAcc Entropy Top10 Compact Overlap Separation Drift GTmargin VisMass HeadAgree",
    ]
    for layer in range(left, right + 1):
        for group in ("A", "B", "C"):
            row = lookup.get((layer, group))
            if row is None:
                continue
            lines.append(
                f"L{layer:02d}  {group:>2s}  {int(row['n']):3d}  "
                f"{row['similarity_accuracy']:.3f}  {row['attention_accuracy']:.3f}  "
                f"{row['attn_entropy']:.3f}  {row['attn_topk']:.3f}  "
                f"{row['attn_compact']:.4f}  {row['attn_overlap']:.3f}  "
                f"{row['attn_separation']:.3f}  {row['attn_drift']:.4f}  "
                f"{row['attn_gt_margin']:.4f}  {row['attn_visual_mass']:.4f}  "
                f"{row['attn_head_agreement']:.3f}"
            )

    effect_lookup = {(int(r["layer"]), str(r["metric"])): r for r in effect_rows}
    lines += ["", "A-B effects at selected peak", "Metric A_mean B_mean A-B Cohen_d"]
    for metric in (
        "attn_entropy", "attn_topk", "attn_compact", "attn_overlap",
        "attn_separation", "attn_drift", "attn_visual_mass",
        "attn_head_agreement", "attn_gt_margin",
    ):
        row = effect_lookup.get((peak, metric))
        if row is not None:
            lines.append(
                f"{metric:22s} {row['A_mean']:+.4f} {row['B_mean']:+.4f} "
                f"{row['A_minus_B']:+.4f} {row['cohen_d_A_minus_B']:+.3f}"
            )

    lines += [
        "",
        "Reading:",
        "- B entropy/compactness higher: B attention is more diffuse.",
        "- B overlap higher and separation lower: subject/reference binding is less distinct.",
        "- B drift higher: grounding is less stable across adjacent layers.",
        "- Similar concentration but different GT margin: concentration is not the main bottleneck.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    return report


def run_model(args, core, backend, model_name: str, groups_selected: Sequence[str]) -> None:
    model_input_root = Path(args.input_root) / model_name
    metadata_path = model_input_root / "pass2_transfer_trace" / "sample_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    prior = base.cap_rows(
        base.read_jsonl(metadata_path),
        selected_groups=groups_selected,
        max_per_group=args.max_per_group,
        seed=args.seed,
        sid=args.sid,
    )
    records, _ = backend.load_records(args.dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    prompt_rows = core.load_standard_prompts(core.resolve_prompt_path(args))

    spec = backend.SPECS[model_name]
    model_cls = getattr(transformers, spec.model_class)
    print("\n" + "=" * 142)
    print(f"LOADING {model_name}: {spec.repo_id}")
    print("=" * 142)
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=core.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation=args.attn_impl,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    core.configure_processor(model, processor)
    device = torch.device(args.device)
    layers, layers_path = core.resolve_decoder_layers(model)
    n_layers = len(layers)

    out_dir = Path(args.output_root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        out_dir / "sample_layer_metrics.npz",
        out_dir / "sample_metadata.jsonl",
        out_dir / "layer_summary.csv",
        out_dir / "group_layer_summary.csv",
        out_dir / "ab_effect_summary.csv",
        out_dir / "peak_report.txt",
        out_dir / "run_config.json",
        out_dir / "errors.jsonl",
    ]
    if args.overwrite:
        for path in files:
            if path.exists():
                path.unlink()
    elif any(path.exists() for path in files[:-1]):
        raise FileExistsError(f"Results exist in {out_dir}; use --overwrite")

    capture = Capture(
        layers, model, processor, args.similarity_map,
        args.similarity_temperature, args.top_fraction
    )
    successful_meta, sample_arrays, grids, grid_sources = [], [], [], []
    progress = tqdm(prior, desc=f"layer-grounding:{model_name}", unit="sample", dynamic_ncols=True)

    try:
        for index, meta in enumerate(progress, 1):
            sid = int(meta["sid"])
            image = None
            batch = None
            try:
                record = record_by_sid[sid]
                prompt = prompt_rows[sid]
                image = core.record_image(record)
                gt_name = base.normalize_relation(prompt["answer_raw"])
                if gt_name not in REL_TO_ID:
                    raise RuntimeError(f"Invalid GT: {prompt['answer_raw']!r}")
                batch = core.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=str(prompt["question_text"]),
                    device=device,
                )
                batch = base.move_batch_to_device(batch, device)
                prompt_spec = base.build_prompt_position_spec(
                    model=model,
                    tokenizer=processor.tokenizer,
                    input_ids=batch["input_ids"],
                    subject=str(prompt["subject"]),
                    reference=str(prompt["reference"]),
                )
                capture.configure(prompt_spec, batch, tuple(image.size), REL_TO_ID[gt_name])
                with torch.inference_mode():
                    model(
                        **batch,
                        use_cache=False,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                sample_arrays.append(sample_to_arrays(capture.results, n_layers))
                if capture.grid_shape is None or capture.grid_source is None:
                    raise RuntimeError("Grid unresolved")
                grids.append(capture.grid_shape)
                grid_sources.append(capture.grid_source)
                successful_meta.append({
                    "model": model_name,
                    "sid": sid,
                    "group": str(meta["group"]),
                    "group_short": base.group_short(str(meta["group"])),
                    "gt": gt_name,
                    "gt_code": REL_TO_ID[gt_name],
                    "subject": str(prompt["subject"]),
                    "reference": str(prompt["reference"]),
                    "grid_height": int(capture.grid_shape[0]),
                    "grid_width": int(capture.grid_shape[1]),
                    "grid_source": capture.grid_source,
                })
            except Exception as exc:
                with (out_dir / "errors.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "model": model_name,
                        "sid": sid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-25:],
                    }, ensure_ascii=False) + "\n")
                raise
            finally:
                capture.reset()
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

            if args.print_every > 0 and index % args.print_every == 0:
                progress.set_postfix_str(f"success={len(sample_arrays)}", refresh=False)
            if (
                args.empty_cache_every > 0
                and index % args.empty_cache_every == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
    finally:
        progress.close()
        capture.close()

    if not sample_arrays:
        raise RuntimeError("No successful samples")

    arrays = stack_samples(sample_arrays)
    gt = np.asarray([m["gt_code"] for m in successful_meta], dtype=np.int16)
    group_arr = np.asarray([m["group"] for m in successful_meta], dtype="U64")
    sids = np.asarray([m["sid"] for m in successful_meta], dtype=np.int64)
    np.savez_compressed(out_dir / "sample_layer_metrics.npz", sid=sids, gt_code=gt, group=group_arr, **arrays)

    with (out_dir / "sample_metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in successful_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    layer_rows, group_rows, effect_rows = make_summaries(arrays, gt, group_arr)
    write_csv(out_dir / "layer_summary.csv", layer_rows)
    write_csv(out_dir / "group_layer_summary.csv", group_rows)
    write_csv(out_dir / "ab_effect_summary.csv", effect_rows)

    peak, search_start, search_end = choose_peak(
        layer_rows, args.peak_source, args.middle_start, args.middle_end
    )
    saved = load_saved_centroid_layer(model_input_root)
    report = print_report(
        model_name, layer_rows, group_rows, effect_rows, peak,
        search_start, search_end, args.neighbor_radius, saved
    )
    (out_dir / "peak_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "model": model_name,
        "repo_id": spec.repo_id,
        "decoder_path": layers_path,
        "n_layers": n_layers,
        "sample_count": len(successful_meta),
        "similarity_map": args.similarity_map,
        "similarity_temperature": args.similarity_temperature,
        "top_fraction": args.top_fraction,
        "middle_start": args.middle_start,
        "middle_end": args.middle_end,
        "peak_source": args.peak_source,
        "peak_layer": peak,
        "saved_centroid_layer": saved,
        "neighbor_radius": args.neighbor_radius,
        "grid_shapes": sorted([list(x) for x in set(grids)]),
        "grid_sources": sorted(set(grid_sources)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved to {out_dir}")
    print(f"  {out_dir / 'peak_report.txt'}")
    print(f"  {out_dir / 'layer_summary.csv'}")
    print(f"  {out_dir / 'group_layer_summary.csv'}")
    print(f"  {out_dir / 'ab_effect_summary.csv'}")
    print(f"  {out_dir / 'sample_layer_metrics.npz'}")

    del model, processor, layers, arrays, sample_arrays
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0,1]")
    if not 0.0 <= args.middle_start < args.middle_end <= 1.0:
        raise ValueError("Require 0 <= middle-start < middle-end <= 1")
    if args.similarity_temperature <= 0:
        raise ValueError("--similarity-temperature must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = base.parse_models(args.models)
    groups = base.parse_groups(args.groups)
    core = base.import_core(args.core_module)
    backend = core.import_two_object_module()

    print("=" * 142)
    print("LAYER-WISE OBJECT→VISUAL GROUNDING ANALYSIS")
    print("=" * 142)
    print(f"models={models}")
    print(f"groups={[base.group_short(x) for x in groups]}")
    print(
        f"peak_source={args.peak_source}, middle=[{args.middle_start:.2f},{args.middle_end:.2f}), "
        f"neighbor_radius={args.neighbor_radius}"
    )
    print("No GT box is used; GT relation is used only for post-hoc accuracy/statistics.")

    completed, failures = 0, []
    for model_name in models:
        try:
            run_model(args, core, backend, model_name, groups)
            completed += 1
        except Exception as exc:
            failures.append((model_name, f"{type(exc).__name__}: {exc}"))
            print(f"\n[ERROR] {model_name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 142)
    print(f"COMPLETE: {completed}/{len(models)} models")
    for model_name, error in failures:
        print(f"  failed {model_name}: {error}")
    if completed == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
