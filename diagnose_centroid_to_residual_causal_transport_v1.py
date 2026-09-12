
# -*- coding: utf-8 -*-
"""
diagnose_centroid_to_residual_causal_transport_v1.py

Mechanistic causal diagnostic for Qwen2.5-VL-3B on COCO_two:

    L24 attention-centroid geometry  --->  downstream object-pair residual spatial state

This script does NOT optimize or repair final answers.  It asks only whether a
controlled change to a centroid head causes a direction-specific change in the
already-established H/V residual spatial coordinates.

Primary intervention
--------------------
For one attention head (default L24H05), take the CLEAN subject/reference query
attention to visual tokens.  Within the visual tokens only, reweight the
attention distribution by an exponential spatial tilt while keeping:

  * total visual attention mass for that query fixed;
  * all non-visual attention unchanged;
  * all model weights unchanged.

For query q and visual token j:

    p'_j \propto p_j exp(beta * s * coord_j)
    a'_j = M_visual * p'_j

The exact change to this head's pre-W_O output is then

    delta o_q = sum_j (a'_j - a_j) V_j.

We inject this exact delta only into the selected head slice at the selected
object-token query position, immediately before W_O.  Thus downstream layers
receive exactly the selected head output that the counterfactual visual
attention distribution would have produced, while the rest of the attention
computation is held fixed.

Coordinate convention
---------------------
Causal centroid coordinates are aligned to the residual H/V convention:

    c_H = x_subject - x_reference       (+ = right)
    c_V = -(y_subject - y_reference)    (+ = above)

Interventions:

    H+ : subject right, reference left
    H- : subject left,  reference right
    V+ : subject up,    reference down
    V- : subject down,  reference up

Controls:

    common_H+ : move both object centroids right; relative c_H should barely move
    common_V+ : move both object centroids up;    relative c_V should barely move
    control head: same antisymmetric H+/H-/V+/V- intervention on a fixed
                  same-layer non-target head.

Residual readout
----------------
The source Synthetic cache defines the H/V geometry at each requested layer.
The target COCO cache supplies the clean Real-NoImage object-pair residual.
Because the intervention changes only the REAL forward, the intervened residual
is exactly:

    r_int = r_clean_cache + [(h_sub-h_ref)_int - (h_sub-h_ref)_clean].

This avoids rerunning NoImage and keeps the readout definition identical to the
existing all-440 residual analysis.

Primary causal statistic
------------------------
For each downstream layer L, finite differences estimate

                 [ dz_H / dc_H    dz_H / dc_V ]
    J_C->R(L) =  [                            ]
                 [ dz_V / dc_H    dz_V / dc_V ]

using H+/H- and V+/V- interventions.

Evidence for axis-specific centroid -> residual transport is:

    diagonal entries positive,
    |diagonal| >> |cross|,
    common-mode effects small,
    target-head effects > same-layer control-head effects.

Recommended pilot
-----------------
CUDA_VISIBLE_DEVICES=0 python -u diagnose_centroid_to_residual_causal_transport_v1.py \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --intervention-layer 24 \
  --target-head 5 \
  --control-head 15 \
  --readout-layers 23,24,25,26 \
  --beta 2.0 \
  --max-samples 80 \
  --output-dir output/qwen3b_centroid_to_residual_causal_transport_n80_v1 \
  --overwrite

Then run all 440 by setting --max-samples 0.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import eval_coco_centroid_select_late_direction_qwen25_v1 as centexp
    import analyze_coco_centroid_generation_step1_v4 as cent
except Exception as exc:
    raise SystemExit(
        "Run this script from the AdaptVis llava16 repository root next to "
        "eval_coco_centroid_select_late_direction_qwen25_v1.py and "
        "analyze_coco_centroid_generation_step1_v4.py.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
EPS = 1e-12


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_layers(text: str) -> List[int]:
    out: List[int] = []
    for piece in str(text).split(","):
        piece = piece.strip().lower().replace("l", "")
        if not piece:
            continue
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(piece))
    result: List[int] = []
    for x in out:
        if x not in result:
            result.append(x)
    if not result:
        raise ValueError("empty layer list")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--target-spatial-npz", required=True)
    p.add_argument("--intervention-layer", type=int, default=24)
    p.add_argument("--target-head", type=int, default=5)
    p.add_argument(
        "--control-head",
        type=int,
        default=15,
        help="Same-layer control head; must differ from --target-head.",
    )
    p.add_argument("--readout-layers", default="23,24,25,26")
    p.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Exponential spatial tilt strength in normalized visual coordinates.",
    )
    p.add_argument(
        "--object-pool",
        default="mean",
        choices=["mean", "last"],
        help="Pooling used for h_sub/h_ref downstream residual readout.",
    )
    p.add_argument("--max-samples", type=int, default=80, help="0 = all 440")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.readout_layers_parsed = parse_layers(args.readout_layers)
    if args.beta <= 0:
        p.error("--beta must be > 0")
    if args.target_head == args.control_head:
        p.error("--control-head must differ from --target-head")
    return args


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return x / max(n, EPS)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.std(vals)) if vals else float("nan")


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def norm_rel(x: Any) -> str:
    s = str(x).strip().lower()
    mapping = {
        "on": "above",
        "over": "above",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
    }
    return mapping.get(s, s)


# =============================================================================
# Existing residual H/V geometry
# =============================================================================

def load_state_npz(path: Path, require_labels: bool):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            definition = (
                str(z["vector_definition"].item())
                if "vector_definition" in keys
                else "relation_vectors"
            )
        elif {"img", "no_image"}.issubset(keys):
            X = np.asarray(z["img"], dtype=np.float32) - np.asarray(
                z["no_image"], dtype=np.float32
            )
            definition = "img_minus_no_image"
        else:
            raise RuntimeError(f"Bad NPZ keys in {path}: {sorted(keys)}")
        if "decoder_block_index" not in keys:
            raise RuntimeError(f"{path} missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]
        sids = (
            np.asarray(z["sample_index"], dtype=np.int64)
            if "sample_index" in keys
            else np.arange(len(X), dtype=np.int64)
        )
        labels = None
        if "relation" in keys:
            labels = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        elif require_labels:
            raise RuntimeError(f"{path} requires relation labels")
    if X.ndim != 3:
        raise RuntimeError(f"Expected [N,L,D], got {X.shape}")
    return X, labels, layers, sids, definition


def fit_hv_geometry(
    X: np.ndarray,
    y: np.ndarray,
    source_layers: Sequence[int],
    wanted_layers: Sequence[int],
):
    lmap = {int(L): i for i, L in enumerate(source_layers)}
    geom: Dict[int, Dict[str, Any]] = {}
    rows = []
    for L in wanted_layers:
        if L not in lmap:
            raise RuntimeError(f"Source cache missing L{L}")
        Xf = X[:, lmap[L]].astype(np.float64)
        center = Xf.mean(axis=0)
        mus = {r: Xf[y == r].mean(axis=0) for r in REL}
        dirs = {r: unit(mus[r] - center) for r in REL}
        dH = unit(dirs["right"] - dirs["left"])
        dV = unit(dirs["above"] - dirs["below"])
        gapH = float(np.dot(mus["right"] - mus["left"], dH))
        gapV = float(np.dot(mus["above"] - mus["below"], dV))
        if gapH < 0:
            dH, gapH = -dH, -gapH
        if gapV < 0:
            dV, gapV = -dV, -gapV
        B = np.stack([dH, dV], axis=1)
        gram = B.T @ B
        dual = B @ np.linalg.inv(gram)
        geom[int(L)] = {
            "center": center,
            "dual": dual,
            "halfH": max(gapH / 2.0, EPS),
            "halfV": max(gapV / 2.0, EPS),
        }
        rows.append(
            {
                "layer": int(L),
                "axis_H_dot_axis_V": float(np.dot(dH, dV)),
                "full_gap_H": gapH,
                "full_gap_V": gapV,
            }
        )
    return geom, pd.DataFrame(rows)


def read_coord(x: np.ndarray, g: Mapping[str, Any]) -> np.ndarray:
    res = np.asarray(x, dtype=np.float64) - np.asarray(g["center"], dtype=np.float64)
    c = np.asarray(g["dual"], dtype=np.float64).T @ res
    return np.asarray(
        [c[0] / float(g["halfH"]), c[1] / float(g["halfV"])],
        dtype=np.float64,
    )


# =============================================================================
# Attention helpers
# =============================================================================

def resolve_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise RuntimeError(f"Cannot locate self-attention in {type(layer).__name__}")


def resolve_hidden_tuple(outputs: Any) -> Sequence[torch.Tensor]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_output", None), "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
    ]
    for value in candidates:
        if isinstance(value, (tuple, list)) and len(value) > 0:
            return value
    raise RuntimeError("Forward did not return decoder hidden_states")


def locate_attention_hidden(args: Sequence[Any], kwargs: Mapping[str, Any]) -> torch.Tensor:
    value = kwargs.get("hidden_states")
    if torch.is_tensor(value) and value.ndim == 3:
        return value
    for item in args:
        if torch.is_tensor(item) and item.ndim == 3:
            return item
    raise RuntimeError("Could not locate attention hidden_states")


class CaptureAttentionInput:
    def __init__(self, attention: Any):
        self.attention = attention
        self.handle = None
        self.hidden: Optional[torch.Tensor] = None
        self.events = 0

    def __enter__(self):
        def hook(_module, args, kwargs):
            hidden = locate_attention_hidden(args, kwargs)
            self.hidden = hidden.detach()
            self.events += 1

        self.handle = self.attention.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_num_heads(attention: Any, attention_tensor: torch.Tensor) -> int:
    # prompt attention tensor is already [H,Q,K]
    return int(attention_tensor.shape[0])


def project_value_states(
    attention: Any,
    hidden_states: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """Return V after GQA repetition as [B,H,S,Dh]."""
    v_proj = getattr(attention, "v_proj", None)
    if v_proj is None:
        raise RuntimeError(f"{type(attention).__name__} has no v_proj")
    values = v_proj(hidden_states)
    if values.ndim != 3:
        raise RuntimeError(f"v_proj shape={tuple(values.shape)}, expected [B,S,D]")
    n_kv = getattr(attention, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = getattr(getattr(attention, "config", None), "num_key_value_heads", None)
    if n_kv is None:
        n_kv = n_heads
    n_kv = int(n_kv)
    if values.shape[-1] % n_kv != 0:
        raise RuntimeError(
            f"v_proj dim={values.shape[-1]} not divisible by num_key_value_heads={n_kv}"
        )
    dh = int(values.shape[-1] // n_kv)
    values = values.view(values.shape[0], values.shape[1], n_kv, dh).transpose(1, 2)
    if n_kv != n_heads:
        if n_heads % n_kv != 0:
            raise RuntimeError(f"Cannot repeat KV heads {n_kv} -> query heads {n_heads}")
        values = values.repeat_interleave(n_heads // n_kv, dim=1)
    return values.contiguous()


class PatchHeadPreWO:
    """Add precomputed deltas to one head slice at selected token positions."""

    def __init__(
        self,
        attention: Any,
        head: int,
        deltas_by_position: Mapping[int, torch.Tensor],
        n_heads: int,
    ):
        self.attention = attention
        self.head = int(head)
        self.deltas = {int(k): v.detach() for k, v in deltas_by_position.items()}
        self.n_heads = int(n_heads)
        self.handle = None
        self.events = 0

    def __enter__(self):
        o_proj = getattr(self.attention, "o_proj", None)
        if o_proj is None:
            raise RuntimeError(f"{type(self.attention).__name__} has no o_proj")

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError("o_proj pre-hook received no tensor input")
            x = inputs[0]
            if x.ndim != 3:
                raise RuntimeError(f"o_proj input shape={tuple(x.shape)}, expected [B,S,D]")
            if x.shape[-1] % self.n_heads != 0:
                raise RuntimeError(
                    f"o_proj input dim={x.shape[-1]} not divisible by n_heads={self.n_heads}"
                )
            dh = int(x.shape[-1] // self.n_heads)
            if not (0 <= self.head < self.n_heads):
                raise RuntimeError(f"head {self.head} outside 0..{self.n_heads-1}")
            start, end = self.head * dh, (self.head + 1) * dh
            y = x.clone()
            for position, delta in self.deltas.items():
                if not (0 <= position < y.shape[1]):
                    raise RuntimeError(
                        f"patch position {position} outside seq len {y.shape[1]}"
                    )
                d = delta.to(device=y.device, dtype=y.dtype)
                if d.numel() != dh:
                    raise RuntimeError(f"delta dim={d.numel()} != head_dim={dh}")
                y[0, position, start:end] = y[0, position, start:end] + d
            self.events += 1
            return (y,) + tuple(inputs[1:])

        self.handle = o_proj.register_forward_pre_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# =============================================================================
# Counterfactual centroid construction
# =============================================================================

@dataclass
class ConditionSpec:
    name: str
    head_kind: str       # target / control
    axis: str            # H / V
    mode: str            # antisym / common
    direction: int       # +1 / -1 in aligned c_H/c_V convention


def conditions() -> List[ConditionSpec]:
    out = []
    for axis in ("H", "V"):
        for direction in (+1, -1):
            out.append(
                ConditionSpec(
                    name=f"target_{axis}{'+' if direction > 0 else '-'}",
                    head_kind="target",
                    axis=axis,
                    mode="antisym",
                    direction=direction,
                )
            )
    out.extend(
        [
            ConditionSpec("common_H+", "target", "H", "common", +1),
            ConditionSpec("common_V+", "target", "V", "common", +1),
        ]
    )
    for axis in ("H", "V"):
        for direction in (+1, -1):
            out.append(
                ConditionSpec(
                    name=f"control_{axis}{'+' if direction > 0 else '-'}",
                    head_kind="control",
                    axis=axis,
                    mode="antisym",
                    direction=direction,
                )
            )
    return out


def normalized_visual_map(full_row: torch.Tensor, visual_idx: torch.Tensor):
    a = full_row.index_select(0, visual_idx).float()
    mass = a.sum()
    if not torch.isfinite(mass) or float(mass.item()) <= EPS:
        raise RuntimeError("visual attention mass is zero/non-finite")
    p = a / mass
    return a, p, mass


def tilted_distribution(
    p: torch.Tensor,
    coord: torch.Tensor,
    signed_beta: float,
) -> torch.Tensor:
    # p is already normalized over visual tokens; multiplication keeps exact zeros zero.
    c = coord.float()
    c = c - c.mean()
    scale = torch.exp(torch.clamp(float(signed_beta) * c, min=-40.0, max=40.0))
    q = p.float() * scale
    denom = q.sum()
    if float(denom.item()) <= EPS or not torch.isfinite(denom):
        raise RuntimeError("tilted visual distribution degenerated")
    return q / denom


def condition_query_signs(spec: ConditionSpec) -> Tuple[int, int, int]:
    """Return coordinate index and subject/ref tilt signs in raw image coords."""
    d = int(spec.direction)
    if spec.axis == "H":
        coord_idx = 0
        if spec.mode == "antisym":
            return coord_idx, +d, -d
        return coord_idx, +d, +d
    if spec.axis == "V":
        coord_idx = 1
        # c_V = -(y_sub-y_ref): +V(above) requires sub y down-sign negative.
        if spec.mode == "antisym":
            return coord_idx, -d, +d
        return coord_idx, -d, -d
    raise ValueError(spec.axis)


def build_condition_patch(
    *,
    spec: ConditionSpec,
    beta: float,
    head: int,
    prompt_attention: torch.Tensor,   # [H,Q,K]
    values: torch.Tensor,             # [B,H,S,Dh]
    visual_indices: Sequence[int],
    coords: torch.Tensor,              # [V,2]
    subject_index: int,
    reference_index: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, float]]:
    device = prompt_attention.device
    vidx = torch.as_tensor(visual_indices, device=device, dtype=torch.long)
    coords = coords.to(device=device, dtype=torch.float32)
    if len(visual_indices) != coords.shape[0]:
        raise RuntimeError("visual index / coordinate length mismatch")
    if head >= prompt_attention.shape[0]:
        raise RuntimeError(f"head={head} >= n_heads={prompt_attention.shape[0]}")

    coord_idx, s_sign, r_sign = condition_query_signs(spec)
    query_data = []
    deltas: Dict[int, torch.Tensor] = {}

    for role, q_index, sign in (
        ("subject", subject_index, s_sign),
        ("reference", reference_index, r_sign),
    ):
        full_row = prompt_attention[head, q_index, :]
        a, p, mass = normalized_visual_map(full_row, vidx)
        p_new = tilted_distribution(p, coords[:, coord_idx], float(beta) * float(sign))
        a_new = mass * p_new
        delta_a = a_new - a

        # Values are from the clean input to this attention layer; L24 input is
        # unchanged in every L24 intervention, so this is the exact A'V-AV delta.
        v = values[0, head].index_select(0, vidx.to(values.device)).float()
        delta_vec = torch.matmul(delta_a.to(v.device).unsqueeze(0), v).squeeze(0)
        deltas[int(q_index)] = delta_vec.detach()

        c_old = torch.sum(p.unsqueeze(1) * coords, dim=0)
        c_new = torch.sum(p_new.unsqueeze(1) * coords, dim=0)
        query_data.append((role, c_old, c_new, float(mass.item())))

    old = {role: c for role, c, _, _ in query_data}
    new = {role: c for role, _, c, _ in query_data}
    masses = {role: m for role, _, _, m in query_data}

    old_dx = float((old["subject"][0] - old["reference"][0]).item())
    old_dy = float((old["subject"][1] - old["reference"][1]).item())
    new_dx = float((new["subject"][0] - new["reference"][0]).item())
    new_dy = float((new["subject"][1] - new["reference"][1]).item())

    old_cH = old_dx
    old_cV = -old_dy
    new_cH = new_dx
    new_cV = -new_dy

    obj_shift_sub = float(torch.linalg.vector_norm(new["subject"] - old["subject"]).item())
    obj_shift_ref = float(torch.linalg.vector_norm(new["reference"] - old["reference"]).item())

    metrics = {
        "clean_cH": old_cH,
        "clean_cV": old_cV,
        "cf_cH": new_cH,
        "cf_cV": new_cV,
        "delta_cH": new_cH - old_cH,
        "delta_cV": new_cV - old_cV,
        "subject_centroid_shift": obj_shift_sub,
        "reference_centroid_shift": obj_shift_ref,
        "mean_object_centroid_shift": 0.5 * (obj_shift_sub + obj_shift_ref),
        "subject_visual_mass": masses["subject"],
        "reference_visual_mass": masses["reference"],
        "delta_head_norm_subject": float(torch.linalg.vector_norm(deltas[subject_index].float()).item()),
        "delta_head_norm_reference": float(torch.linalg.vector_norm(deltas[reference_index].float()).item()),
    }
    return deltas, metrics


# =============================================================================
# Object pair states
# =============================================================================

def span_positions(span: Sequence[int]) -> List[int]:
    if len(span) < 2:
        raise RuntimeError(f"bad span={span}")
    a, b = int(span[0]), int(span[1])
    if b < a:
        a, b = b, a
    # Existing centroid code uses span[1] as the final object token, so spans are
    # treated as inclusive endpoints here.
    return list(range(a, b + 1))


def pooled_state(hidden: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    idx = torch.as_tensor(positions, device=hidden.device, dtype=torch.long)
    selected = hidden[0].index_select(0, idx)
    if mode == "last":
        return selected[-1]
    return selected.mean(dim=0)


def pair_state(
    hidden_tuple: Sequence[torch.Tensor],
    layer: int,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    pool: str,
) -> torch.Tensor:
    # HF hidden_states[0] is embedding output; block L output is hidden_states[L+1].
    index = int(layer) + 1
    if index >= len(hidden_tuple):
        raise RuntimeError(
            f"Requested block L{layer}, but hidden_states has {len(hidden_tuple)} entries"
        )
    h = hidden_tuple[index]
    sub = pooled_state(h, subject_positions, pool)
    ref = pooled_state(h, reference_positions, pool)
    return (sub - ref).detach().float().cpu()


# =============================================================================
# Per-sample clean trace and intervention
# =============================================================================

def clean_trace(
    *,
    model: Any,
    processor: Any,
    layers: Sequence[Any],
    image: Any,
    record: Mapping[str, Any],
    intervention_layer: int,
    args: argparse.Namespace,
):
    device = torch.device(args.device)
    batch = centexp.make_batch(processor, image, record, device)
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    input_length = len(input_ids)
    sub_span, ref_span = cent.locate_object_spans(
        processor.tokenizer,
        input_ids,
        record["subject"],
        record["reference"],
    )
    subject_index = int(sub_span[1])
    reference_index = int(ref_span[1])
    subject_positions = span_positions(sub_span)
    reference_positions = span_positions(ref_span)

    visual_indices = cent.resolve_visual_indices(model, processor, batch, input_ids)
    coords = cent.visual_coordinates(
        model,
        batch,
        len(visual_indices),
        batch["input_ids"].device,
    )
    if coords is None:
        raise RuntimeError("Could not construct visual coordinates")

    attention = resolve_self_attention(layers[int(intervention_layer)])
    with CaptureAttentionInput(attention) as capture:
        with torch.inference_mode():
            outputs = model(
                **batch,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
    if capture.events != 1 or capture.hidden is None:
        raise RuntimeError(
            f"L{intervention_layer} attention input capture events={capture.events}"
        )
    attentions = centexp.resolve_attention_tuple(outputs)
    hidden_tuple = resolve_hidden_tuple(outputs)
    prompt_attention = cent.normalize_attention_tensor(
        attentions[int(intervention_layer)],
        expected_query_length=input_length,
    )
    n_heads = get_num_heads(attention, prompt_attention)
    values = project_value_states(attention, capture.hidden, n_heads)

    clean_pairs = {
        int(L): pair_state(
            hidden_tuple,
            int(L),
            subject_positions,
            reference_positions,
            args.object_pool,
        )
        for L in args.readout_layers_parsed
    }

    return {
        "batch": batch,
        "subject_index": subject_index,
        "reference_index": reference_index,
        "subject_positions": subject_positions,
        "reference_positions": reference_positions,
        "visual_indices": visual_indices,
        "coords": coords.detach(),
        "attention": attention,
        "prompt_attention": prompt_attention.detach(),
        "values": values.detach(),
        "n_heads": n_heads,
        "clean_pairs": clean_pairs,
        "input_length": input_length,
    }


def run_intervention(
    *,
    model: Any,
    clean: Mapping[str, Any],
    head: int,
    deltas: Mapping[int, torch.Tensor],
    args: argparse.Namespace,
) -> Dict[int, torch.Tensor]:
    attention = clean["attention"]
    with PatchHeadPreWO(attention, head, deltas, clean["n_heads"]) as patch:
        with torch.inference_mode():
            outputs = model(
                **clean["batch"],
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
    if patch.events != 1:
        raise RuntimeError(f"o_proj patch fired {patch.events} times, expected 1")
    hidden_tuple = resolve_hidden_tuple(outputs)
    result = {
        int(L): pair_state(
            hidden_tuple,
            int(L),
            clean["subject_positions"],
            clean["reference_positions"],
            args.object_pool,
        )
        for L in args.readout_layers_parsed
    }
    del outputs
    return result


# =============================================================================
# Aggregation
# =============================================================================

def finite_difference_summary(df: pd.DataFrame, head_kind: str) -> pd.DataFrame:
    rows = []
    prefix = "target" if head_kind == "target" else "control"
    for L in sorted(df["readout_layer"].unique()):
        sub = df[df["readout_layer"] == L]
        piv = sub.pivot_table(
            index="sid",
            columns="condition",
            values=["delta_cH", "delta_cV", "z_H", "z_V"],
            aggfunc="first",
        )
        needed = [
            f"{prefix}_H+", f"{prefix}_H-", f"{prefix}_V+", f"{prefix}_V-"
        ]
        ok = np.ones(len(piv), dtype=bool)
        for cond in needed:
            for metric in ("delta_cH", "delta_cV", "z_H", "z_V"):
                ok &= (metric, cond) in piv.columns
        piv = piv.iloc[np.where(ok)[0]] if len(piv) else piv
        per = []
        for sid in piv.index:
            try:
                dch = float(
                    piv.loc[sid, ("delta_cH", f"{prefix}_H+")]
                    - piv.loc[sid, ("delta_cH", f"{prefix}_H-")]
                )
                dcv = float(
                    piv.loc[sid, ("delta_cV", f"{prefix}_V+")]
                    - piv.loc[sid, ("delta_cV", f"{prefix}_V-")]
                )
                dzH_H = float(
                    piv.loc[sid, ("z_H", f"{prefix}_H+")]
                    - piv.loc[sid, ("z_H", f"{prefix}_H-")]
                )
                dzV_H = float(
                    piv.loc[sid, ("z_V", f"{prefix}_H+")]
                    - piv.loc[sid, ("z_V", f"{prefix}_H-")]
                )
                dzH_V = float(
                    piv.loc[sid, ("z_H", f"{prefix}_V+")]
                    - piv.loc[sid, ("z_H", f"{prefix}_V-")]
                )
                dzV_V = float(
                    piv.loc[sid, ("z_V", f"{prefix}_V+")]
                    - piv.loc[sid, ("z_V", f"{prefix}_V-")]
                )
                if abs(dch) <= EPS or abs(dcv) <= EPS:
                    continue
                per.append(
                    {
                        "J_HH": dzH_H / dch,
                        "J_VH": dzV_H / dch,
                        "J_HV": dzH_V / dcv,
                        "J_VV": dzV_V / dcv,
                        "H_delta_c": dch,
                        "V_delta_c": dcv,
                        "H_dzH": dzH_H,
                        "H_dzV": dzV_H,
                        "V_dzH": dzH_V,
                        "V_dzV": dzV_V,
                    }
                )
            except Exception:
                continue
        if not per:
            continue
        p = pd.DataFrame(per)
        diag_abs = 0.5 * (np.abs(p.J_HH) + np.abs(p.J_VV))
        cross_abs = 0.5 * (np.abs(p.J_HV) + np.abs(p.J_VH))
        rows.append(
            {
                "head_kind": head_kind,
                "readout_layer": int(L),
                "N": int(len(p)),
                "mean_J_HH": float(p.J_HH.mean()),
                "mean_J_HV": float(p.J_HV.mean()),
                "mean_J_VH": float(p.J_VH.mean()),
                "mean_J_VV": float(p.J_VV.mean()),
                "median_J_HH": float(p.J_HH.median()),
                "median_J_HV": float(p.J_HV.median()),
                "median_J_VH": float(p.J_VH.median()),
                "median_J_VV": float(p.J_VV.median()),
                "H_same_axis_sign_rate": float(np.mean(p.H_dzH > 0)),
                "V_same_axis_sign_rate": float(np.mean(p.V_dzV > 0)),
                "mean_abs_diag": float(diag_abs.mean()),
                "mean_abs_cross": float(cross_abs.mean()),
                "diag_over_cross": float(diag_abs.mean() / max(cross_abs.mean(), EPS)),
                "corr_H_deltaC_deltaZ": safe_corr(p.H_delta_c, p.H_dzH),
                "corr_V_deltaC_deltaZ": safe_corr(p.V_delta_c, p.V_dzV),
                "mean_H_delta_c": float(p.H_delta_c.mean()),
                "mean_V_delta_c": float(p.V_delta_c.mean()),
            }
        )
    return pd.DataFrame(rows)


def common_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for L in sorted(df["readout_layer"].unique()):
        for cond in ("common_H+", "common_V+"):
            sub = df[(df.readout_layer == L) & (df.condition == cond)]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "readout_layer": int(L),
                    "condition": cond,
                    "N": int(len(sub)),
                    "mean_abs_delta_cH": float(np.abs(sub.delta_cH).mean()),
                    "mean_abs_delta_cV": float(np.abs(sub.delta_cV).mean()),
                    "mean_object_centroid_shift": float(sub.mean_object_centroid_shift.mean()),
                    "mean_abs_delta_zH": float(np.abs(sub.delta_z_H).mean()),
                    "mean_abs_delta_zV": float(np.abs(sub.delta_z_V).mean()),
                    "mean_delta_z_norm": float(
                        np.sqrt(sub.delta_z_H.to_numpy() ** 2 + sub.delta_z_V.to_numpy() ** 2).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def condition_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["condition", "readout_layer"], sort=False)
    rows = []
    for (cond, L), sub in g:
        rows.append(
            {
                "condition": cond,
                "readout_layer": int(L),
                "N": int(len(sub)),
                "mean_delta_cH": float(sub.delta_cH.mean()),
                "mean_delta_cV": float(sub.delta_cV.mean()),
                "mean_abs_delta_cH": float(np.abs(sub.delta_cH).mean()),
                "mean_abs_delta_cV": float(np.abs(sub.delta_cV).mean()),
                "mean_object_centroid_shift": float(sub.mean_object_centroid_shift.mean()),
                "mean_delta_zH": float(sub.delta_z_H.mean()),
                "mean_delta_zV": float(sub.delta_z_V.mean()),
                "mean_abs_delta_zH": float(np.abs(sub.delta_z_H).mean()),
                "mean_abs_delta_zV": float(np.abs(sub.delta_z_V).mean()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Xs, ys, Ls, sids_s, source_def = load_state_npz(
        Path(args.source_spatial_npz), require_labels=True
    )
    Xt, yt, Lt, sids_t, target_def = load_state_npz(
        Path(args.target_spatial_npz), require_labels=False
    )
    geom, geom_df = fit_hv_geometry(Xs, ys, Ls, args.readout_layers_parsed)
    geom_df.to_csv(outdir / "hv_geometry.csv", index=False)
    target_lmap = {int(L): i for i, L in enumerate(Lt)}
    target_sid_to_i = {int(sid): i for i, sid in enumerate(sids_t)}
    for L in args.readout_layers_parsed:
        if L not in target_lmap:
            raise RuntimeError(f"Target cache missing readout L{L}")

    # centexp expects these fields in its loader helpers.
    args.max_samples = None if int(args.max_samples) == 0 else int(args.max_samples)
    records, audit = centexp.load_coco_records(args)
    if args.max_samples is not None:
        # load_records may already limit. Keep deterministic relation-agnostic first-N
        # only if the helper returned more than requested.
        records = records[: int(args.max_samples)]

    model, processor, layers, decoder_path, spec = centexp.load_model_and_processor(args)
    if not (0 <= args.intervention_layer < len(layers)):
        raise RuntimeError(
            f"intervention L{args.intervention_layer} outside 0..{len(layers)-1}"
        )

    cond_specs = conditions()
    detail_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print("\n" + "=" * 176)
    print("CENTROID -> RESIDUAL CAUSAL TRANSPORT")
    print("=" * 176)
    print(
        f"model={args.model} | N={len(records)} | intervention=L{args.intervention_layer}H{args.target_head} "
        f"| control=H{args.control_head} | beta={args.beta}"
    )
    print(f"readout_layers={args.readout_layers_parsed} | pool={args.object_pool}")
    print(f"source_definition={source_def} | target_definition={target_def}")
    print("No generation / no correction objective. Internal causal diagnostic only.")
    print("=" * 176, flush=True)

    for record in tqdm(records, desc="centroid->residual causal"):
        sid = int(record["sid"])
        image = None
        clean = None
        try:
            if sid not in target_sid_to_i:
                raise RuntimeError(f"sid={sid} missing from target spatial cache")
            ti = target_sid_to_i[sid]
            image = centexp.open_record_image(record)
            clean = clean_trace(
                model=model,
                processor=processor,
                layers=layers,
                image=image,
                record=record,
                intervention_layer=args.intervention_layer,
                args=args,
            )
            if args.target_head >= clean["n_heads"] or args.control_head >= clean["n_heads"]:
                raise RuntimeError(
                    f"n_heads={clean['n_heads']} but target/control="
                    f"{args.target_head}/{args.control_head}"
                )

            # Clean residual z from the exact cached Real-NoImage representation.
            clean_residual: Dict[int, np.ndarray] = {}
            clean_z: Dict[int, np.ndarray] = {}
            for L in args.readout_layers_parsed:
                r = Xt[ti, target_lmap[L]].astype(np.float64)
                clean_residual[int(L)] = r
                clean_z[int(L)] = read_coord(r, geom[int(L)])

            for cs in cond_specs:
                head = args.target_head if cs.head_kind == "target" else args.control_head
                deltas, cmetrics = build_condition_patch(
                    spec=cs,
                    beta=args.beta,
                    head=head,
                    prompt_attention=clean["prompt_attention"],
                    values=clean["values"],
                    visual_indices=clean["visual_indices"],
                    coords=clean["coords"],
                    subject_index=clean["subject_index"],
                    reference_index=clean["reference_index"],
                )
                int_pairs = run_intervention(
                    model=model,
                    clean=clean,
                    head=head,
                    deltas=deltas,
                    args=args,
                )

                for L in args.readout_layers_parsed:
                    pair_delta = (
                        int_pairs[int(L)] - clean["clean_pairs"][int(L)]
                    ).numpy().astype(np.float64)
                    r_int = clean_residual[int(L)] + pair_delta
                    z_int = read_coord(r_int, geom[int(L)])
                    z0 = clean_z[int(L)]
                    detail_rows.append(
                        {
                            "sid": sid,
                            "gt": record["relation"],
                            "condition": cs.name,
                            "head_kind": cs.head_kind,
                            "head": int(head),
                            "intervention_layer": int(args.intervention_layer),
                            "readout_layer": int(L),
                            "beta": float(args.beta),
                            **cmetrics,
                            "clean_z_H": float(z0[0]),
                            "clean_z_V": float(z0[1]),
                            "z_H": float(z_int[0]),
                            "z_V": float(z_int[1]),
                            "delta_z_H": float(z_int[0] - z0[0]),
                            "delta_z_V": float(z_int[1] - z0[1]),
                            "pair_delta_l2": float(np.linalg.norm(pair_delta)),
                        }
                    )

            # Incremental persistence for long HPC runs.
            if len(detail_rows) % max(10 * len(cond_specs) * len(args.readout_layers_parsed), 1) == 0:
                pd.DataFrame(detail_rows).to_csv(outdir / "sample_condition_layer.csv", index=False)

        except Exception as exc:
            errors.append(
                {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": " | ".join(traceback.format_exc().splitlines()[-10:]),
                }
            )
            tqdm.write(f"[ERROR sid={sid}] {type(exc).__name__}: {exc}")
        finally:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
            if clean is not None:
                try:
                    del clean
                except Exception:
                    pass
            cleanup()

    if not detail_rows:
        raise RuntimeError("No successful intervention samples")

    df = pd.DataFrame(detail_rows)
    df.to_csv(outdir / "sample_condition_layer.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(outdir / "errors.csv", index=False)

    effect = condition_effect_summary(df)
    target_fd = finite_difference_summary(df, "target")
    control_fd = finite_difference_summary(df, "control")
    fd = pd.concat([target_fd, control_fd], ignore_index=True)
    common = common_mode_summary(df)

    effect.to_csv(outdir / "condition_effect_summary.csv", index=False)
    fd.to_csv(outdir / "causal_transport_matrix.csv", index=False)
    common.to_csv(outdir / "common_mode_control.csv", index=False)

    # Target/control causal effect ratio using mean finite-difference diagonal magnitude.
    compare_rows = []
    for L in args.readout_layers_parsed:
        t = target_fd[target_fd.readout_layer == L]
        c = control_fd[control_fd.readout_layer == L]
        if len(t) and len(c):
            tr, cr = t.iloc[0], c.iloc[0]
            compare_rows.append(
                {
                    "readout_layer": int(L),
                    "target_mean_abs_diag": float(tr.mean_abs_diag),
                    "control_mean_abs_diag": float(cr.mean_abs_diag),
                    "target_over_control_diag": float(
                        tr.mean_abs_diag / max(float(cr.mean_abs_diag), EPS)
                    ),
                    "target_diag_over_cross": float(tr.diag_over_cross),
                    "control_diag_over_cross": float(cr.diag_over_cross),
                    "target_H_sign_rate": float(tr.H_same_axis_sign_rate),
                    "target_V_sign_rate": float(tr.V_same_axis_sign_rate),
                }
            )
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(outdir / "target_vs_control_summary.csv", index=False)

    print("\n" + "=" * 176)
    print("CAUSAL TRANSPORT MATRIX: finite differences normalized by achieved centroid displacement")
    print("=" * 176)
    if len(fd):
        print(fd.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCOMMON-MODE CONTROL")
    print("-" * 176)
    if len(common):
        print(common.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTARGET HEAD vs SAME-LAYER CONTROL HEAD")
    print("-" * 176)
    if len(compare):
        print(compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Compact interpretation-oriented sanity numbers.
    print("\nINTERVENTION SANITY: achieved centroid shifts")
    print("-" * 176)
    sanity = effect[
        effect.condition.isin(
            ["target_H+", "target_H-", "target_V+", "target_V-", "common_H+", "common_V+"]
        )
    ]
    if len(sanity):
        cols = [
            "condition", "readout_layer", "N", "mean_delta_cH", "mean_delta_cV",
            "mean_object_centroid_shift", "mean_delta_zH", "mean_delta_zV",
        ]
        print(sanity[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    config = {
        "script": "diagnose_centroid_to_residual_causal_transport_v1.py",
        "model": args.model,
        "repo_id": getattr(spec, "repo_id", ""),
        "decoder_path": decoder_path,
        "N_requested": len(records),
        "N_success": int(df.sid.nunique()),
        "N_errors": len(errors),
        "intervention_layer": args.intervention_layer,
        "target_head": args.target_head,
        "control_head": args.control_head,
        "readout_layers": args.readout_layers_parsed,
        "beta": args.beta,
        "object_pool": args.object_pool,
        "source_spatial_npz": args.source_spatial_npz,
        "target_spatial_npz": args.target_spatial_npz,
        "source_definition": source_def,
        "target_definition": target_def,
        "intervention_definition": (
            "within-visual attention redistribution with fixed visual mass; exact "
            "counterfactual A'V-AV added to selected head pre-W_O slice at subject/reference queries"
        ),
        "coordinate_convention": "cH=dx (+right), cV=-dy (+above)",
        "conditions": [vars(c) for c in cond_specs],
        "audit": audit,
    }
    (outdir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (outdir / "DONE").write_text("ok\n", encoding="utf-8")

    print(f"\n[saved] {outdir / 'sample_condition_layer.csv'}")
    print(f"[saved] {outdir / 'causal_transport_matrix.csv'}")
    print(f"[saved] {outdir / 'common_mode_control.csv'}")
    print(f"[saved] {outdir / 'target_vs_control_summary.csv'}")
    print(f"[success] N={df.sid.nunique()} errors={len(errors)}")


if __name__ == "__main__":
    main()
