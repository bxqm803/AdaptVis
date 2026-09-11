
# -*- coding: utf-8 -*-
"""
eval_spatial_head_to_causal_text_message_injection_v1.py

Mechanistic question
====================
Does a spatially informative attention head fail because it does not write enough
image-derived spatial message into behaviorally causal TEXT tokens?

For a fixed spatial head (L,h), a causal text token position p, and visual source
positions V, extract the sample-specific post-W_O visual message:

    m_real(L,h,p)
      = W_O^h sum_{s in V} A_real[L,h,p,s] V_real[L,h,s]

    m_gray(L,h,p)
      = W_O^h sum_{s in V} A_gray[L,h,p,s] V_gray[L,h,s]

    m_RG(L,h,p)
      = m_real - m_gray

Then, during a fresh REAL-image generation pass, add an extra copy scaled by alpha
to the attention-module output at the SAME text position p:

    attn_out'_L[p]
      = attn_out_L[p] + alpha * m_RG(L,h,p)

If multiple selected spatial heads are temporally upstream of the same causal text
state, each head writes its own message at its own layer.  Heads later than the
causal state are NOT allowed to target that state.

This does NOT:
    * edit the causal hidden state directly,
    * edit attention probabilities,
    * inject a learned LEFT/RIGHT/ABOVE/BELOW prototype,
    * choose a relation-specific direction at intervention time.

It amplifies the CURRENT SAMPLE'S OWN Real-Gray visual A·V·W_O message through
heads that were independently identified as spatially informative.

Important indexing
==================
A causal state in ranked_k36_tokens.csv is the OUTPUT of decoder block C.
An attention head at layer L can causally affect that state if L <= C:
    - L == C: direct write into the same block before its MLP/output;
    - L <  C: upstream write which must propagate through later blocks.
L > C is impossible and is excluded.

Default spatial heads
=====================
The default bundle is the current Top-10 Qwen3B heads under the Synthetic-400
Real-Gray direction codebook transferred to COCO (ordered by COCO accuracy):

    L26H03, L23H01, L23H05, L26H02, L22H09,
    L23H00, L22H13, L22H02, L21H14, L23H10

Override with:
    --spatial-heads "23:1,23:5,22:9"

Causal targets
==============
The input ranking is the existing oracle writer-guided causal ranking:

    M(L,p) = (h_real-h_gray)^T grad_h J_writer

By default this script:
    * keeps TEXT states only (visual excluded),
    * restricts causal layers to L20-L26,
    * takes the first K=7 states after those filters per sample.

Therefore this remains an ORACLE MECHANISM DIAGNOSTIC: GT was used upstream to
construct the causal ranking.  GT is NOT used to choose the spatial head message.

Causal-state movement diagnostic
================================
For each selected causal state (C,p), also record:

    delta_causal = h_real[C,p] - h_gray[C,p]
    move_alpha   = h_patched[C,p] - h_real[C,p]

and report:

    cos(move_alpha, delta_causal)
    <move_alpha, normalize(delta_causal)>

This tells us whether stronger spatial-head communication actually moves the
downstream causal state along its own image-induced useful direction.

Recommended N=80 run
====================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_spatial_head_to_causal_text_message_injection_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --spatial-heads \
    "26:3,23:1,23:5,26:2,22:9,23:0,22:13,22:2,21:14,23:10" \
  --scales 0.25,0.5,1,2,4 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_spatial_to_causal_text_injection_n80_v1 \
  --overwrite

Useful narrower bundle:
    --spatial-heads "23:1,23:5,22:9,22:2,21:14"

Outputs
=======
selected_causal_text_states.csv
message_edges.csv
generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
causal_state_movement.csv
movement_summary.csv
sample_message_summary.csv
analysis_summary.txt
metadata.json
errors.jsonl

Dependencies
============
Run from the AdaptVis llava16 repository root.  Reuses:
    analyze_coco_centroid_generation_step1_v4.py
    eval_coco_multilayer_relation_trajectory_repair_v1.py
    eval_qwen_dynamic_k24_all440_v1.py
    analyze_coco_flip_attention_spatial_vectors_v1.py
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import math
import random
import re
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


REL = ("left", "right", "above", "below")
EPS = 1e-12

DEFAULT_QWEN3B_SPATIAL_HEADS = (
    "26:3,23:1,23:5,26:2,22:9,23:0,22:13,22:2,21:14,23:10"
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--ranked-causal", required=True)

    p.add_argument(
        "--causal-layers",
        default="20-26",
        help="Eligible causal TEXT state layers.",
    )
    p.add_argument(
        "--causal-top-k",
        type=int,
        default=7,
        help="First K text causal states after layer/domain filtering, per sample.",
    )
    p.add_argument(
        "--causal-categories",
        default="",
        help=(
            "Optional comma-separated broad categories, e.g. "
            "'reference,subject,relation_words'. Empty = all non-visual text."
        ),
    )

    p.add_argument(
        "--spatial-heads",
        default=DEFAULT_QWEN3B_SPATIAL_HEADS,
        help='Comma-separated attention heads, e.g. "23:1,23:5,22:9".',
    )
    p.add_argument(
        "--scales",
        default="0.25,0.5,1,2,4",
        help=(
            "Extra Real-Gray message coefficients. alpha=1 adds one extra "
            "copy of the extracted Real-Gray visual head message."
        ),
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="0 = all available samples.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument(
        "--replay-relative-tolerance",
        type=float,
        default=1e-3,
        help="Warn if attention replay relative error exceeds this.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# =============================================================================
# Parsing / generic
# =============================================================================

def parse_layers(text: str) -> List[int]:
    out = set()
    for item in str(text).split(","):
        item = item.strip().upper().replace("L", "")
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(item))
    return sorted(out)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    out = []
    seen = set()
    for item in str(text).split(","):
        item = item.strip().upper().replace("L", "").replace("H", ":")
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad head spec {item!r}; expected layer:head")
        a, b = item.split(":", 1)
        pair = (int(a), int(b))
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    if not out:
        raise ValueError("No --spatial-heads")
    return out


def parse_floats(text: str) -> List[float]:
    xs = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            xs.append(float(item))
    if not xs:
        raise ValueError("No scales")
    return xs


def parse_categories(text: str) -> Optional[set]:
    xs = {x.strip() for x in str(text).split(",") if x.strip()}
    return xs if xs else None


def hname(L: int, h: int) -> str:
    return f"L{int(L):02d}H{int(h):02d}"


def safe_mean(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            z = float(x)
        except Exception:
            continue
        if math.isfinite(z):
            vals.append(z)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            z = float(x)
        except Exception:
            continue
        if math.isfinite(z):
            vals.append(z)
    return float(np.median(vals)) if vals else float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def unit_projection(a: np.ndarray, direction: np.ndarray) -> float:
    a = np.asarray(a, np.float32)
    direction = np.asarray(direction, np.float32)
    n = float(np.linalg.norm(direction))
    if n <= EPS:
        return float("nan")
    return float(np.dot(a, direction / n))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


# =============================================================================
# Causal ranking
# =============================================================================

def load_causal_selection(
    path: Path,
    causal_layers: Sequence[int],
    top_k: int,
    categories: Optional[set],
    allowed_sids: Optional[set] = None,
) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = {
        "sid", "rank", "source_layer", "position",
        "token", "category", "broad_category", "mediation",
    }
    missing = need - set(d.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    for c in ("sid", "rank", "source_layer", "position"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)
    d["mediation"] = pd.to_numeric(d["mediation"], errors="coerce")

    d = d[d["source_layer"].isin(set(map(int, causal_layers)))].copy()
    d = d[d["broad_category"].astype(str) != "visual"].copy()
    d = d[d["broad_category"].astype(str) != "last"].copy()

    if categories is not None:
        d = d[d["broad_category"].astype(str).isin(categories)].copy()

    if allowed_sids is not None:
        d = d[d["sid"].isin(allowed_sids)].copy()

    rows = []
    for sid, g in d.groupby("sid"):
        g = g.sort_values("rank").head(int(top_k)).copy()
        g["causal_text_rank"] = np.arange(1, len(g) + 1)
        rows.append(g)

    if not rows:
        return d.iloc[:0].copy()
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Dataset/model
# =============================================================================

def load_coco_meta(args):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(args.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(args.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, args.max_samples, args.seed)
    return two, meta, rec_by_sid


def load_model(args, two):
    specs = base.merged_model_specs(two)
    spec = specs[args.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"Loading {spec.repo_id}", flush=True)
    try:
        model = cls.from_pretrained(spec.repo_id, **kw)
    except TypeError:
        kw["torch_dtype"] = kw.pop("dtype")
        model = cls.from_pretrained(spec.repo_id, **kw)

    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    return model, processor, decoder_layers, decoder_path, spec


def relation_token_variants(tokenizer) -> Dict[str, List[int]]:
    out = {}
    for rel in REL:
        ids = set()
        for s in (rel, " " + rel, rel.capitalize(), " " + rel.capitalize()):
            z = tokenizer.encode(s, add_special_tokens=False)
            if len(z) == 1:
                ids.add(int(z[0]))
        if not ids:
            z = tokenizer.encode(" " + rel, add_special_tokens=False)
            if not z:
                raise RuntimeError(f"No token id for {rel}")
            ids.add(int(z[-1]))
        out[rel] = sorted(ids)
    return out


# =============================================================================
# Attention trace -> per-head visual write
# =============================================================================

def trace_target_index(trace: Any, target_position: int) -> int:
    lookup = {
        int(global_p): int(local_p)
        for local_p, global_p in enumerate(trace.target_positions)
    }
    if int(target_position) not in lookup:
        raise RuntimeError(
            f"target p={target_position} absent from trace targets "
            f"{trace.target_positions}"
        )
    return lookup[int(target_position)]


def one_head_source_write(
    *,
    trace: Any,
    head: int,
    target_position: int,
    source_positions: Sequence[int],
) -> np.ndarray:
    """
    post-W_O contribution from source_positions -> one target position for one head.

        W_O^h sum_s A[h,target,s] V[h,s]
    """
    src = sorted(set(map(int, source_positions)))
    if not src:
        raise RuntimeError("No source positions")

    local = trace_target_index(trace, target_position)
    H = int(trace.attention_weights.shape[0])
    if not (0 <= int(head) < H):
        raise RuntimeError(f"head={head} outside 0..{H-1}")

    source = torch.as_tensor(src, dtype=torch.long)

    klen = int(trace.value_states.shape[1])
    if int(source.max()) >= klen:
        raise RuntimeError(
            f"source max={int(source.max())} exceeds value length={klen}"
        )

    weights = (
        trace.attention_weights[int(head), local, :]
        .index_select(0, source)
        .float()
    )                                                   # [S]
    values = (
        trace.value_states[int(head)]
        .index_select(0, source)
        .float()
    )                                                   # [S,Dh]

    pre = torch.einsum("s,sd->d", weights, values)       # [Dh]
    Wo = trace.o_proj_weight[:, int(head), :].float()    # [Dmodel,Dh]
    post = torch.einsum("d,od->o", pre, Wo)              # [Dmodel]
    return post.detach().cpu().numpy().astype(np.float32)


def extract_rg_edges(
    *,
    attention_helper: Any,
    model: Any,
    rb: Mapping[str, Any],
    gb: Mapping[str, Any],
    relation_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    spatial_heads: Sequence[Tuple[int, int]],
    causal_rows: pd.DataFrame,
    visual_positions: Sequence[int],
    replay_tolerance: float,
):
    """
    For every eligible spatial-head -> causal-token pair, compute Real-Gray visual
    post-W_O message.

    Eligibility:
        spatial head layer L <= causal block-output layer C.
    """
    # Which causal positions need tracing?
    target_positions = sorted(set(int(x) for x in causal_rows["position"]))
    if not target_positions:
        return [], {}, {}

    head_layers = sorted(set(int(L) for L, _ in spatial_heads))

    _, rt = attention_helper.run_and_trace(
        model=model,
        batch=rb,
        token_map=relation_map,
        decoder_layers=decoder_layers,
        layer_indices=head_layers,
        target_positions=target_positions,
    )
    _, gt = attention_helper.run_and_trace(
        model=model,
        batch=gb,
        token_map=relation_map,
        decoder_layers=decoder_layers,
        layer_indices=head_layers,
        target_positions=target_positions,
    )

    replay = {}
    for L in head_layers:
        replay[L] = max(
            float(rt[L].replay_relative_error),
            float(gt[L].replay_relative_error),
        )
        if replay[L] > replay_tolerance:
            print(
                f"[replay warning] L{L}: relative_error={replay[L]:.3e}",
                flush=True,
            )

    # edge rows and injection map:
    # patch_map[layer][position] = sum over eligible selected spatial heads at layer
    patch_map: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    edge_rows = []

    for cr in causal_rows.itertuples():
        C = int(cr.source_layer)
        p = int(cr.position)

        for L, h in spatial_heads:
            L, h = int(L), int(h)
            if L > C:
                continue

            mr = one_head_source_write(
                trace=rt[L],
                head=h,
                target_position=p,
                source_positions=visual_positions,
            )
            mg = one_head_source_write(
                trace=gt[L],
                head=h,
                target_position=p,
                source_positions=visual_positions,
            )
            msg = (mr - mg).astype(np.float32)

            if p not in patch_map[L]:
                patch_map[L][p] = np.zeros_like(msg, dtype=np.float32)
            patch_map[L][p] += msg

            edge_rows.append({
                "causal_layer": C,
                "causal_position": p,
                "causal_text_rank": int(cr.causal_text_rank),
                "causal_global_rank": int(cr.rank),
                "causal_token": str(cr.token),
                "causal_category": str(cr.category),
                "causal_broad_category": str(cr.broad_category),
                "causal_mediation": float(cr.mediation),
                "spatial_head_layer": L,
                "spatial_head": h,
                "spatial_head_name": hname(L, h),
                "real_visual_message_norm": float(np.linalg.norm(mr)),
                "gray_visual_message_norm": float(np.linalg.norm(mg)),
                "realgray_visual_message_norm": float(np.linalg.norm(msg)),
                "real_vs_gray_message_cos": cosine(mr, mg),
                "replay_relative_error": replay[L],
            })

    return edge_rows, {L: dict(v) for L, v in patch_map.items()}, replay


# =============================================================================
# Multiple attention-output edits + causal-state recording
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
    raise RuntimeError("Could not find 3D output")


def replace_first_3d(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        xs = list(output)
        for i, item in enumerate(xs):
            if torch.is_tensor(item) and item.ndim == 3:
                xs[i] = replacement
                return tuple(xs)
    if isinstance(output, list):
        xs = list(output)
        for i, item in enumerate(xs):
            if torch.is_tensor(item) and item.ndim == 3:
                xs[i] = replacement
                return xs
    raise RuntimeError("Could not replace first 3D output")


class MultiAttentionPositionDelta:
    """
    patch_map[layer][position] = residual-space post-W_O vector.
    Applies only on full prompt PREFILL.
    """

    def __init__(
        self,
        *,
        decoder_layers,
        attention_helper,
        patch_map: Dict[int, Dict[int, np.ndarray]],
        prompt_len: int,
        scale: float,
    ):
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.patch_map = patch_map
        self.prompt_len = int(prompt_len)
        self.scale = float(scale)
        self.handles = []
        self.applications = defaultdict(int)

    def __enter__(self):
        for L, by_pos in self.patch_map.items():
            attention = self.attention_helper.resolve_self_attention(
                self.decoder_layers[int(L)]
            )

            def make_hook(layer: int, pos_map: Dict[int, np.ndarray]):
                def hook(_module, _inputs, output):
                    hidden = first_3d(output)
                    if int(hidden.shape[1]) != self.prompt_len:
                        return None

                    modified = hidden.clone()
                    for p, vec in pos_map.items():
                        p = int(p)
                        if not (0 <= p < int(hidden.shape[1])):
                            raise RuntimeError(
                                f"L{layer} target p={p} outside q_len={hidden.shape[1]}"
                            )
                        v = torch.as_tensor(
                            vec,
                            device=hidden.device,
                            dtype=hidden.dtype,
                        )
                        if int(v.numel()) != int(hidden.shape[-1]):
                            raise RuntimeError(
                                f"L{layer} delta dim={v.numel()} hidden={hidden.shape[-1]}"
                            )
                        modified[0, p] += self.scale * v
                        self.applications[(layer, p)] += 1

                    return replace_first_3d(output, modified)
                return hook

            self.handles.append(
                attention.register_forward_hook(make_hook(int(L), by_pos))
            )
        return self

    def validate(self):
        expected = {
            (int(L), int(p))
            for L, by_pos in self.patch_map.items()
            for p in by_pos
        }
        bad = []
        for key in expected:
            if int(self.applications.get(key, 0)) != 1:
                bad.append((key, int(self.applications.get(key, 0))))
        if bad:
            raise RuntimeError(f"Patch application mismatch: {bad[:10]}")

    def __exit__(self, exc_type, exc, tb):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


class CausalStateRecorder:
    """
    Record decoder block outputs h_C[p] for selected causal states on PREFILL.
    """
    def __init__(
        self,
        decoder_layers,
        causal_rows: pd.DataFrame,
        prompt_len: int,
    ):
        self.decoder_layers = decoder_layers
        self.prompt_len = int(prompt_len)
        self.by_layer = defaultdict(list)
        for r in causal_rows.itertuples():
            self.by_layer[int(r.source_layer)].append(int(r.position))
        self.by_layer = {
            L: sorted(set(ps)) for L, ps in self.by_layer.items()
        }
        self.states = {}
        self.handles = []

    def __enter__(self):
        for L, positions in self.by_layer.items():
            def make_hook(layer: int, pos_list: List[int]):
                def hook(_module, _inputs, output):
                    hidden = first_3d(output)
                    if int(hidden.shape[1]) != self.prompt_len:
                        return None
                    for p in pos_list:
                        if 0 <= p < int(hidden.shape[1]):
                            self.states[(layer, p)] = (
                                hidden[0, p].detach().float().cpu().numpy()
                                .astype(np.float32)
                            )
                    return None
                return hook
            self.handles.append(
                self.decoder_layers[L].register_forward_hook(
                    make_hook(L, positions)
                )
            )
        return self

    def validate(self):
        expected = {
            (int(L), int(p))
            for L, ps in self.by_layer.items()
            for p in ps
        }
        missing = sorted(expected - set(self.states))
        if missing:
            raise RuntimeError(f"Missing causal states: {missing[:10]}")

    def __exit__(self, exc_type, exc, tb):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def generate_with_patch_and_record(
    *,
    model,
    processor,
    decoder_layers,
    attention_helper,
    batch,
    causal_rows,
    patch_map,
    scale,
    max_new_tokens,
):
    prompt_len = int(batch["input_ids"].shape[1])
    patcher = MultiAttentionPositionDelta(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        patch_map=patch_map,
        prompt_len=prompt_len,
        scale=scale,
    )
    recorder = CausalStateRecorder(
        decoder_layers=decoder_layers,
        causal_rows=causal_rows,
        prompt_len=prompt_len,
    )

    with patcher, recorder:
        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )

    patcher.validate()
    recorder.validate()

    pred = traj.normalize_relation(base, text)
    return pred, text, dict(recorder.states), dict(patcher.applications)


@torch.inference_mode()
def generate_clean_and_record(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    causal_rows,
    max_new_tokens,
):
    prompt_len = int(batch["input_ids"].shape[1])
    recorder = CausalStateRecorder(
        decoder_layers=decoder_layers,
        causal_rows=causal_rows,
        prompt_len=prompt_len,
    )
    with recorder:
        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )
    recorder.validate()
    pred = traj.normalize_relation(base, text)
    return pred, text, dict(recorder.states)


# =============================================================================
# Summary
# =============================================================================

def summarize_generation(d: pd.DataFrame) -> pd.DataFrame:
    if len(d) == 0:
        return pd.DataFrame()

    b = d[d["condition"] == "baseline"].set_index("sid")
    rows = []

    for cond, g in d.groupby("condition"):
        x = g.set_index("sid")
        common = sorted(set(b.index) & set(x.index))
        if not common:
            continue

        bb = b.loc[common]["correct"].astype(bool).to_numpy()
        cc = x.loc[common]["correct"].astype(bool).to_numpy()

        rows.append({
            "condition": cond,
            "scale": (
                0.0 if cond == "baseline"
                else float(g["scale"].iloc[0])
            ),
            "N": len(common),
            "accuracy": float(cc.mean()),
            "gain_vs_baseline": float(cc.mean() - bb.mean()),
            "wrong_to_correct": int(np.sum((~bb) & cc)),
            "correct_to_wrong": int(np.sum(bb & (~cc))),
            "net": int(np.sum((~bb) & cc) - np.sum(bb & (~cc))),
            "prediction_changed": int(
                np.sum(
                    b.loc[common]["prediction"].astype(str).to_numpy()
                    != x.loc[common]["prediction"].astype(str).to_numpy()
                )
            ),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["scale", "condition"]).reset_index(drop=True)


def summarize_by_relation(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gt, g in d.groupby("gt"):
        z = summarize_generation(g)
        if len(z):
            z.insert(0, "gt", gt)
            rows.append(z)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_movement(d: pd.DataFrame) -> pd.DataFrame:
    if len(d) == 0:
        return pd.DataFrame()

    rows = []
    for scale, g in d.groupby("scale"):
        rows.append({
            "scale": float(scale),
            "N_states": len(g),
            "N_samples": g["sid"].nunique(),
            "mean_move_norm": safe_mean(g["move_norm"]),
            "median_move_norm": safe_median(g["move_norm"]),
            "mean_cos_move_vs_own_RG_delta": safe_mean(
                g["cos_move_vs_own_RG_delta"]
            ),
            "median_cos_move_vs_own_RG_delta": safe_median(
                g["cos_move_vs_own_RG_delta"]
            ),
            "mean_projection_on_own_RG_delta": safe_mean(
                g["projection_on_own_RG_delta"]
            ),
            "median_projection_on_own_RG_delta": safe_median(
                g["projection_on_own_RG_delta"]
            ),
        })

        for cat, gg in g.groupby("causal_broad_category"):
            rows.append({
                "scale": float(scale),
                "scope": f"category:{cat}",
                "N_states": len(gg),
                "N_samples": gg["sid"].nunique(),
                "mean_move_norm": safe_mean(gg["move_norm"]),
                "median_move_norm": safe_median(gg["move_norm"]),
                "mean_cos_move_vs_own_RG_delta": safe_mean(
                    gg["cos_move_vs_own_RG_delta"]
                ),
                "median_cos_move_vs_own_RG_delta": safe_median(
                    gg["cos_move_vs_own_RG_delta"]
                ),
                "mean_projection_on_own_RG_delta": safe_mean(
                    gg["projection_on_own_RG_delta"]
                ),
                "median_projection_on_own_RG_delta": safe_median(
                    gg["projection_on_own_RG_delta"]
                ),
            })

    out = pd.DataFrame(rows)
    if "scope" not in out:
        out["scope"] = "all"
    else:
        out["scope"] = out["scope"].fillna("all")
    return out.sort_values(["scale", "scope"]).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers = parse_layers(a.causal_layers)
    spatial_heads = parse_heads(a.spatial_heads)
    scales = parse_floats(a.scales)
    categories = parse_categories(a.causal_categories)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    errors_path = outdir / "errors.jsonl"

    two, meta, rec_by_sid = load_coco_meta(a)
    meta_by_sid = {int(x["sid"]): x for x in meta}

    ranking_all = pd.read_csv(a.ranked_causal)
    ranking_sids = set(
        pd.to_numeric(ranking_all["sid"], errors="coerce")
        .dropna().astype(int).tolist()
    )
    del ranking_all

    eval_meta = [x for x in meta if int(x["sid"]) in ranking_sids]
    if a.eval_max_samples > 0:
        eval_meta = traj.stratified_cap(
            eval_meta,
            a.eval_max_samples,
            a.seed + 1,
        )
    eval_sids = {int(x["sid"]) for x in eval_meta}

    selected = load_causal_selection(
        Path(a.ranked_causal),
        causal_layers=causal_layers,
        top_k=a.causal_top_k,
        categories=categories,
        allowed_sids=eval_sids,
    )
    selected["gt"] = selected["sid"].map(
        lambda sid: meta_by_sid[int(sid)]["gt"]
    )
    selected.to_csv(outdir / "selected_causal_text_states.csv", index=False)

    selected_by_sid = {
        int(sid): g.copy()
        for sid, g in selected.groupby("sid")
    }

    attention_helper = importlib.import_module(a.attention_helper_module)

    model = processor = None
    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        for C in causal_layers:
            if not (0 <= C < n_layers):
                raise ValueError(f"causal L{C} outside 0..{n_layers-1}")

        # Validate spatial heads against actual attention modules / query-head counts.
        n_heads_by_layer = {}
        for L, h in spatial_heads:
            if not (0 <= L < n_layers):
                raise ValueError(f"spatial head layer L{L} outside model")
            attn = attention_helper.resolve_self_attention(decoder_layers[L])
            q_proj = getattr(attn, "q_proj", None)
            head_dim = getattr(attn, "head_dim", None)
            nh = getattr(attn, "num_heads", None)
            if nh is None:
                nh = getattr(getattr(attn, "config", None), "num_attention_heads", None)
            if nh is None:
                if q_proj is None or head_dim is None:
                    raise RuntimeError(f"Cannot infer query heads at L{L}")
                nh = int(q_proj.out_features) // int(head_dim)
            n_heads_by_layer[L] = int(nh)
            if not (0 <= h < int(nh)):
                raise ValueError(
                    f"{hname(L,h)} invalid; L{L} has {nh} query heads"
                )

        relation_map = relation_token_variants(processor.tokenizer)

        print("=" * 170)
        print("SPATIAL-HEAD -> CAUSAL TEXT TOKEN REAL-GRAY MESSAGE INJECTION")
        print("=" * 170)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(f"causal layers={causal_layers} TopK text={a.causal_top_k}")
        print(
            "causal categories=",
            sorted(categories) if categories is not None else "all non-visual text",
        )
        print(
            "spatial heads=",
            ",".join(hname(L,h) for L,h in spatial_heads),
        )
        print(f"scales={scales}")
        print(
            f"eval samples={len(eval_meta)} selected causal states={len(selected)}"
        )
        print(
            "message = Real-Gray visual-source A·V·W_O at the causal token position"
        )
        print(
            "eligibility = spatial_head_layer <= causal_state_layer; later heads excluded"
        )
        print()

        gen_rows = []
        edge_rows_all = []
        movement_rows = []
        sample_message_rows = []

        for m in tqdm(eval_meta, desc="COCO spatial-message injection"):
            sid = int(m["sid"])
            if sid not in selected_by_sid:
                continue

            real = gray = rb = gb = None
            try:
                causal_rows = selected_by_sid[sid]

                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                ids_r = rb["input_ids"][0].detach().cpu().tolist()
                ids_g = gb["input_ids"][0].detach().cpu().tolist()
                if ids_r != ids_g:
                    raise RuntimeError(
                        "Real/Gray token sequences differ; cannot align token positions."
                    )

                visual = sorted(
                    set(
                        map(
                            int,
                            base.resolve_visual_indices(
                                model,
                                processor,
                                rb,
                                ids_r,
                            ),
                        )
                    )
                )
                if not visual:
                    raise RuntimeError("No visual token positions resolved.")

                # Only trace spatial head layers that can reach at least one selected
                # causal state in this sample.  Still keep head identity fixed globally.
                max_causal_layer = int(causal_rows["source_layer"].max())
                heads_sid = [
                    (L, h)
                    for L, h in spatial_heads
                    if int(L) <= max_causal_layer
                ]
                if not heads_sid:
                    raise RuntimeError(
                        f"No spatial head is upstream of selected causal states; "
                        f"max causal layer={max_causal_layer}"
                    )

                edges, patch_map, replay = extract_rg_edges(
                    attention_helper=attention_helper,
                    model=model,
                    rb=rb,
                    gb=gb,
                    relation_map=relation_map,
                    decoder_layers=decoder_layers,
                    spatial_heads=heads_sid,
                    causal_rows=causal_rows,
                    visual_positions=visual,
                    replay_tolerance=a.replay_relative_tolerance,
                )

                for e in edges:
                    e.update({
                        "sid": sid,
                        "gt": m["gt"],
                        "baseline_unknown_yet": True,
                    })
                    edge_rows_all.append(e)

                # Some early causal text states can have zero eligible spatial heads.
                eligible_positions = {
                    int(e["causal_position"]) for e in edges
                }
                n_eligible_states = sum(
                    int(p) in eligible_positions
                    for p in causal_rows["position"].tolist()
                )

                total_edge_norm = sum(
                    float(e["realgray_visual_message_norm"])
                    for e in edges
                )
                sample_message_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "selected_causal_state_N": len(causal_rows),
                    "eligible_causal_state_N": n_eligible_states,
                    "eligible_edge_N": len(edges),
                    "visual_token_N": len(visual),
                    "sum_edge_message_norm": total_edge_norm,
                    "mean_edge_message_norm": safe_mean(
                        e["realgray_visual_message_norm"] for e in edges
                    ),
                    "max_replay_relative_error": (
                        max(replay.values()) if replay else np.nan
                    ),
                })

                # Clean Real and Gray causal states.
                causal_layers_sid = sorted(
                    set(map(int, causal_rows["source_layer"].tolist()))
                )
                hr = dyn.capture_cpu(
                    model,
                    decoder_layers,
                    rb,
                    causal_layers_sid,
                )
                hg = dyn.capture_cpu(
                    model,
                    decoder_layers,
                    gb,
                    causal_layers_sid,
                )

                own_delta = {}
                for cr in causal_rows.itertuples():
                    C, p = int(cr.source_layer), int(cr.position)
                    R = hr[C][0].astype(np.float32)
                    G = hg[C][0].astype(np.float32)
                    if p >= min(R.shape[0], G.shape[0]):
                        raise RuntimeError(
                            f"causal L{C} p{p} outside hidden sequence"
                        )
                    own_delta[(C,p)] = (R[p] - G[p]).astype(np.float32)

                # Baseline generation with causal-state recorder.
                base_pred, base_text, clean_recorded = generate_clean_and_record(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    causal_rows=causal_rows,
                    max_new_tokens=a.max_new_tokens,
                )
                base_correct = base_pred == m["gt"]
                gen_rows.append({
                    "sid": sid,
                    "gt": m["gt"],
                    "condition": "baseline",
                    "scale": 0.0,
                    "prediction": base_pred,
                    "correct": base_correct,
                    "text": base_text,
                    "selected_causal_state_N": len(causal_rows),
                    "eligible_causal_state_N": n_eligible_states,
                    "eligible_edge_N": len(edges),
                })

                # Verify recorder agrees with plain Real forward closely enough for
                # movement baseline; generate prefill should be same prompt computation.
                for cr in causal_rows.itertuples():
                    C, p = int(cr.source_layer), int(cr.position)
                    if (C,p) not in clean_recorded:
                        continue
                    reference = hr[C][0, p].astype(np.float32)
                    err = float(
                        np.linalg.norm(clean_recorded[(C,p)] - reference)
                        / max(np.linalg.norm(reference), EPS)
                    )
                    if err > 1e-3:
                        print(
                            f"[prefill warning] sid={sid} L{C} p{p} "
                            f"generate-vs-forward relerr={err:.3e}",
                            flush=True,
                        )

                # Strength sweep.
                for alpha in scales:
                    pred, text, patched_states, applications = (
                        generate_with_patch_and_record(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            attention_helper=attention_helper,
                            batch=rb,
                            causal_rows=causal_rows,
                            patch_map=patch_map,
                            scale=alpha,
                            max_new_tokens=a.max_new_tokens,
                        )
                    )

                    correct = pred == m["gt"]
                    gen_rows.append({
                        "sid": sid,
                        "gt": m["gt"],
                        "condition": f"spatial_message_x{alpha:g}",
                        "scale": float(alpha),
                        "prediction": pred,
                        "correct": correct,
                        "text": text,
                        "wrong_to_correct": (not base_correct) and correct,
                        "correct_to_wrong": base_correct and (not correct),
                        "selected_causal_state_N": len(causal_rows),
                        "eligible_causal_state_N": n_eligible_states,
                        "eligible_edge_N": len(edges),
                        "patch_location_N": len(applications),
                    })

                    for cr in causal_rows.itertuples():
                        C, p = int(cr.source_layer), int(cr.position)
                        key = (C,p)
                        if key not in patched_states or key not in clean_recorded:
                            continue

                        move = (
                            patched_states[key] - clean_recorded[key]
                        ).astype(np.float32)
                        delta = own_delta[key]

                        incoming_heads = [
                            e for e in edges
                            if int(e["causal_layer"]) == C
                            and int(e["causal_position"]) == p
                        ]

                        movement_rows.append({
                            "sid": sid,
                            "gt": m["gt"],
                            "baseline_correct": base_correct,
                            "patched_correct": correct,
                            "scale": float(alpha),
                            "causal_layer": C,
                            "causal_position": p,
                            "causal_text_rank": int(cr.causal_text_rank),
                            "causal_global_rank": int(cr.rank),
                            "causal_token": str(cr.token),
                            "causal_category": str(cr.category),
                            "causal_broad_category": str(cr.broad_category),
                            "causal_mediation": float(cr.mediation),
                            "eligible_spatial_head_N": len(incoming_heads),
                            "own_RG_delta_norm": float(np.linalg.norm(delta)),
                            "move_norm": float(np.linalg.norm(move)),
                            "cos_move_vs_own_RG_delta": cosine(move, delta),
                            "projection_on_own_RG_delta": unit_projection(
                                move, delta
                            ),
                        })

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid} {type(exc).__name__}: {exc}"
                )
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gen_df = pd.DataFrame(gen_rows)
        edge_df = pd.DataFrame(edge_rows_all)
        move_df = pd.DataFrame(movement_rows)
        sample_msg_df = pd.DataFrame(sample_message_rows)

        gen_df.to_csv(outdir / "generation_per_sample.csv", index=False)
        edge_df.to_csv(outdir / "message_edges.csv", index=False)
        move_df.to_csv(outdir / "causal_state_movement.csv", index=False)
        sample_msg_df.to_csv(outdir / "sample_message_summary.csv", index=False)

        gen_sum = summarize_generation(gen_df)
        gen_sum.to_csv(outdir / "generation_summary.csv", index=False)

        rel_sum = summarize_by_relation(gen_df)
        rel_sum.to_csv(outdir / "generation_by_relation.csv", index=False)

        move_sum = summarize_movement(move_df)
        move_sum.to_csv(outdir / "movement_summary.csv", index=False)

        # Compact console report.
        report = []
        report.append("=" * 170)
        report.append("SPATIAL-HEAD -> CAUSAL TEXT TOKEN MESSAGE INJECTION")
        report.append("=" * 170)
        completed = (
            gen_df[gen_df["condition"] == "baseline"]["sid"].nunique()
            if len(gen_df) else 0
        )
        report.append(
            f"requested N={len(eval_meta)} | completed N={completed} | "
            f"selected causal states={len(selected)}"
        )
        report.append(
            "spatial heads="
            + ",".join(hname(L,h) for L,h in spatial_heads)
        )
        report.append(f"scales={scales}")
        report.append("")

        report.append("GENERATION")
        report.append("-" * 170)
        if len(gen_sum):
            report.append(
                gen_sum.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        report.append("")

        report.append("CAUSAL-STATE MOVEMENT")
        report.append("-" * 170)
        if len(move_sum):
            z = move_sum[move_sum["scope"] == "all"]
            report.append(
                z.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append("MESSAGE EDGE STATS")
        report.append("-" * 170)
        if len(edge_df):
            report.append(
                "N edges={} | mean ||m_RG||={:.5f} | median ||m_RG||={:.5f}".format(
                    len(edge_df),
                    safe_mean(edge_df["realgray_visual_message_norm"]),
                    safe_median(edge_df["realgray_visual_message_norm"]),
                )
            )
            by_head = (
                edge_df.groupby("spatial_head_name")
                .agg(
                    edges=("sid", "size"),
                    samples=("sid", "nunique"),
                    mean_msg_norm=("realgray_visual_message_norm", "mean"),
                    median_msg_norm=("realgray_visual_message_norm", "median"),
                )
                .sort_values("mean_msg_norm", ascending=False)
            )
            report.append(by_head.to_string())
        report.append("")

        # Wrong-only / correct-only compact effect.
        if len(gen_df):
            baseline = (
                gen_df[gen_df["condition"] == "baseline"]
                [["sid", "correct"]]
                .rename(columns={"correct": "baseline_correct"})
            )
            patched = gen_df[gen_df["condition"] != "baseline"].merge(
                baseline, on="sid", how="left"
            )
            report.append("BASELINE-WRONG SAMPLES")
            report.append("-" * 170)
            rows = []
            for scale, g in patched.groupby("scale"):
                w = g[~g["baseline_correct"].astype(bool)]
                if len(w):
                    rows.append({
                        "scale": float(scale),
                        "N_wrong": len(w),
                        "repaired": int(w["correct"].astype(bool).sum()),
                        "repair_rate": float(w["correct"].astype(bool).mean()),
                    })
            if rows:
                report.append(
                    pd.DataFrame(rows).to_string(
                        index=False,
                        float_format=lambda x: f"{x:.4f}",
                    )
                )

        report_text = "\n".join(report) + "\n"
        print(report_text)
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(
            outdir / "metadata.json",
            {
                "script": "eval_spatial_head_to_causal_text_message_injection_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "ranked_causal": str(a.ranked_causal),
                "causal_layers": causal_layers,
                "causal_top_k": a.causal_top_k,
                "causal_categories": (
                    sorted(categories) if categories is not None else None
                ),
                "spatial_heads": [
                    {"layer": L, "head": h, "name": hname(L,h)}
                    for L,h in spatial_heads
                ],
                "scales": scales,
                "message_definition": (
                    "post-W_O Real-Gray visual-source message at each selected "
                    "causal text token: W_O^h sum_visual A_real V_real - "
                    "W_O^h sum_visual A_gray V_gray"
                ),
                "intervention_definition": (
                    "During REAL prefill, attention_out[L,p] += alpha*m_RG for "
                    "each fixed spatial head L,h with L <= causal_state_layer."
                ),
                "oracle_warning": (
                    "Causal token ranking is GT-writer-guided.  Spatial message "
                    "itself is sample-specific Real-Gray and does not use GT."
                ),
            },
        )

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
