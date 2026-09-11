#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_all_positive_suppliers_reconstruct_causal_rg_v1.py

Question
========
For EACH selected causal text state (C,p), take ALL attention heads whose
first-order contribution to that state's OWN Real-Gray direction is positive:

    M_RG[h -> (C,p)] > threshold

Amplify each of those heads by one extra copy of its sample-specific
PRE-W_O Real-Gray change at the SAME token position p:

    delta_z[L,h,p] = z_real[L,h,p] - z_gray[L,h,p]

    m_RG[L,h,p] = W_O^{L,h} delta_z[L,h,p]

    attention_out_L[p] += alpha * m_RG[L,h,p]

Only heads with L <= C are eligible.

Crucially, this script patches ONE causal state at a time.
That avoids contamination from simultaneously patching the other six causal
token positions.

Then measure:

    MOVE = h_patch[C,p] - h_real[C,p]
    RG   = h_real[C,p]  - h_gray[C,p]

    cosine(MOVE, RG)
    projection of MOVE on normalized RG
    ||MOVE|| / ||RG||

If ALL positive attention-head suppliers approximately reconstruct the causal
state's original Real-Gray direction, then with alpha=1 we would expect:

    cosine(MOVE,RG) -> high
    ||MOVE||/||RG|| -> around 1

and ideally MOVE ~= RG.

Input
=====
Use the per-state decomposition produced by:

    analyze_head_contribution_to_causal_rg_percent_v2.py

Specifically:
    head_to_causal_rg_per_state.csv

Recommended N=80
================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_all_positive_suppliers_reconstruct_causal_rg_v1.py \
  --model qwen-3b \
  --head-contrib \
    output/qwen3b_head_to_causal_rg_percent_n80_v2/head_to_causal_rg_per_state.csv \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --scales 0.5,1,2 \
  --positive-threshold 0 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_all_positive_suppliers_reconstruct_rg_n80_v1 \
  --overwrite

Fastest test:
    --scales 1

Outputs
=======
per_state_results.csv
summary.csv
summary_by_causal_layer.csv
summary_by_category.csv
positive_supplier_counts.csv
analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
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
import scan_qwen_spatial_heads_vs_causal_core_v1 as spatialscan


EPS = 1e-12
REL = ("left", "right", "above", "below")


# =============================================================================
# CLI / helpers
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

    p.add_argument(
        "--head-contrib",
        required=True,
        help="head_to_causal_rg_per_state.csv from the per-state contribution scan.",
    )
    p.add_argument("--ranked-causal", required=True)

    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument("--causal-categories", default="")

    p.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help=(
            "Select heads with rg_mediation > this value. "
            "Default 0 means every positive supplier."
        ),
    )
    p.add_argument(
        "--scales",
        default="1",
        help="Extra copies of each selected head's Real-Gray message.",
    )

    p.add_argument("--gray-value", type=int, default=128)
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

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_categories(text: str) -> Optional[set]:
    xs = {x.strip() for x in str(text).split(",") if x.strip()}
    return xs if xs else None


def hname(L: int, h: int) -> str:
    return f"L{int(L):02d}H{int(h):02d}"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def safe_mean(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Data/model
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


def load_selected_causal(
    ranked_path: Path,
    allowed_sids: set,
    causal_layers: Sequence[int],
    top_k: int,
    categories: Optional[set],
) -> pd.DataFrame:
    d = pd.read_csv(ranked_path)

    needed = {
        "sid", "rank", "source_layer", "position",
        "token", "category", "broad_category", "mediation",
    }
    missing = needed - set(d.columns)
    if missing:
        raise RuntimeError(
            f"{ranked_path} missing columns: {sorted(missing)}"
        )

    for c in ("sid", "rank", "source_layer", "position"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)

    d = d[d["sid"].isin(allowed_sids)].copy()
    d = d[d["source_layer"].isin(set(map(int, causal_layers)))].copy()
    d = d[d["broad_category"].astype(str) != "visual"].copy()
    d = d[d["broad_category"].astype(str) != "last"].copy()

    if categories is not None:
        d = d[d["broad_category"].astype(str).isin(categories)].copy()

    rows = []
    for sid, g in d.groupby("sid"):
        z = g.sort_values("rank").head(int(top_k)).copy()
        z["causal_text_rank"] = np.arange(1, len(z) + 1)
        rows.append(z)

    if not rows:
        return d.iloc[:0].copy()

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Captures
# =============================================================================

class StatePreOCapture:
    def __init__(self, decoder_layers, state_layers, head_layers):
        self.states = {}
        self.pre_o = {}
        self.handles = []

        for L in sorted(set(map(int, state_layers))):
            def make_state_hook(layer):
                def hook(_m, _inp, out):
                    x = traj.first_tensor(out)
                    self.states[layer] = (
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook

            self.handles.append(
                decoder_layers[L].register_forward_hook(make_state_hook(L))
            )

        for L in sorted(set(map(int, head_layers))):
            attn = spatialscan.resolve_attn(decoder_layers[L])
            op = spatialscan.resolve_o_proj(attn)

            def make_pre_hook(layer):
                def hook(_m, inputs):
                    self.pre_o[layer] = (
                        inputs[0].detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            self.handles.append(
                op.register_forward_pre_hook(make_pre_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def capture_states_pre_o(
    model,
    decoder_layers,
    batch,
    state_layers,
    head_layers,
):
    cap = StatePreOCapture(
        decoder_layers,
        state_layers=state_layers,
        head_layers=head_layers,
    )
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)

        ms = [L for L in state_layers if L not in cap.states]
        mh = [L for L in head_layers if L not in cap.pre_o]
        if ms or mh:
            raise RuntimeError(
                f"Missing states={ms}, PRE-W_O={mh}"
            )

        return cap.states, cap.pre_o
    finally:
        cap.close()


class StateCapture:
    def __init__(self, decoder_layers, layers):
        self.states = {}
        self.handles = []

        for L in sorted(set(map(int, layers))):
            def make_hook(layer):
                def hook(_m, _inp, out):
                    x = traj.first_tensor(out)
                    self.states[layer] = (
                        x.detach().float().cpu().numpy().astype(np.float32)
                    )
                    return None
                return hook

            self.handles.append(
                decoder_layers[L].register_forward_hook(make_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


# =============================================================================
# Head geometry / message construction
# =============================================================================

def infer_head_geometry(model, decoder_layers, layers):
    cfg = spatialscan.get_text_config(model)
    fallback_heads = int(cfg.num_attention_heads)

    out = {}
    for L in layers:
        attn = spatialscan.resolve_attn(decoder_layers[L])
        op = spatialscan.resolve_o_proj(attn)

        H = getattr(attn, "num_heads", None)
        if H is None:
            H = getattr(
                getattr(attn, "config", None),
                "num_attention_heads",
                None,
            )
        if H is None:
            H = fallback_heads

        H = int(H)
        width = int(op.in_features)
        if width % H != 0:
            raise RuntimeError(
                f"L{L}: o_proj input={width}, heads={H}"
            )

        out[L] = {
            "n_heads": H,
            "head_dim": width // H,
        }

    return out


def build_single_state_patch_map(
    *,
    decoder_layers,
    real_pre_o: Mapping[int, np.ndarray],
    gray_pre_o: Mapping[int, np.ndarray],
    head_geometry: Mapping[int, Mapping[str, int]],
    heads: Sequence[Tuple[int, int]],
    causal_layer: int,
    causal_position: int,
):
    """
    For ONE causal state only:
        patch_map[L][p] = sum selected heads at layer L of W_O^h delta_z_h[p]
    """
    C = int(causal_layer)
    p = int(causal_position)

    patch_map: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    message_norms = []

    for L, h in heads:
        L, h = int(L), int(h)

        if L > C:
            continue
        if L not in real_pre_o or L not in gray_pre_o:
            continue

        zr = np.asarray(real_pre_o[L], dtype=np.float32)
        zg = np.asarray(gray_pre_o[L], dtype=np.float32)
        n = min(zr.shape[1], zg.shape[1])

        if not (0 <= p < n):
            continue

        H = int(head_geometry[L]["n_heads"])
        D = int(head_geometry[L]["head_dim"])

        if not (0 <= h < H):
            raise RuntimeError(
                f"{hname(L,h)} invalid; L{L} has {H} heads"
            )

        delta_z = (
            zr[0, p, h*D:(h+1)*D]
            - zg[0, p, h*D:(h+1)*D]
        ).astype(np.float32)

        attn = spatialscan.resolve_attn(decoder_layers[L])
        op = spatialscan.resolve_o_proj(attn)
        W = op.weight.detach().float().cpu().numpy().astype(np.float32)
        Wh = W[:, h*D:(h+1)*D]

        msg = (Wh @ delta_z).astype(np.float32)

        if p not in patch_map[L]:
            patch_map[L][p] = np.zeros_like(msg)
        patch_map[L][p] += msg

        message_norms.append(float(np.linalg.norm(msg)))

    return {L: dict(x) for L, x in patch_map.items()}, message_norms


# =============================================================================
# Patch hook
# =============================================================================

def first_3d(output):
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected attention output [B,S,D], got {tuple(output.shape)}"
            )
        return output

    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x) and x.ndim == 3:
                return x

    raise RuntimeError("Could not locate 3D attention output")


def replace_first_3d(output, replacement):
    if torch.is_tensor(output):
        return replacement

    if isinstance(output, tuple):
        xs = list(output)
        for i, x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim == 3:
                xs[i] = replacement
                return tuple(xs)

    if isinstance(output, list):
        xs = list(output)
        for i, x in enumerate(xs):
            if torch.is_tensor(x) and x.ndim == 3:
                xs[i] = replacement
                return xs

    raise RuntimeError("Could not replace attention output")


class MultiLayerPatch:
    def __init__(
        self,
        decoder_layers,
        patch_map,
        prompt_len,
        scale,
    ):
        self.decoder_layers = decoder_layers
        self.patch_map = patch_map
        self.prompt_len = int(prompt_len)
        self.scale = float(scale)
        self.handles = []
        self.counts = defaultdict(int)

    def __enter__(self):
        for L, pos_map in self.patch_map.items():
            attn = spatialscan.resolve_attn(self.decoder_layers[L])

            def make_hook(layer, by_pos):
                def hook(_m, _inp, out):
                    x = first_3d(out)

                    # prefill only
                    if int(x.shape[1]) != self.prompt_len:
                        return None

                    y = x.clone()

                    for p, delta in by_pos.items():
                        p = int(p)
                        v = torch.as_tensor(
                            delta,
                            device=y.device,
                            dtype=y.dtype,
                        )
                        y[0, p] += self.scale * v
                        self.counts[(layer, p)] += 1

                    return replace_first_3d(out, y)

                return hook

            self.handles.append(
                attn.register_forward_hook(
                    make_hook(int(L), dict(pos_map))
                )
            )

        return self

    def validate(self):
        expected = {
            (int(L), int(p))
            for L, by_pos in self.patch_map.items()
            for p in by_pos
        }

        bad = [
            (k, int(self.counts.get(k, 0)))
            for k in expected
            if int(self.counts.get(k, 0)) != 1
        ]

        if bad:
            raise RuntimeError(
                f"Patch application count mismatch: {bad[:10]}"
            )

    def __exit__(self, exc_type, exc, tb):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def patched_state(
    *,
    model,
    decoder_layers,
    batch,
    causal_layer,
    patch_map,
    scale,
):
    sc = StateCapture(decoder_layers, [causal_layer])

    try:
        with MultiLayerPatch(
            decoder_layers=decoder_layers,
            patch_map=patch_map,
            prompt_len=int(batch["input_ids"].shape[1]),
            scale=scale,
        ) as ph:
            kw = dict(batch)
            kw["use_cache"] = False
            _ = model(**kw)

        ph.validate()

        if causal_layer not in sc.states:
            raise RuntimeError(
                f"Missing patched causal layer L{causal_layer}"
            )

        return sc.states[causal_layer]
    finally:
        sc.close()


# =============================================================================
# Contribution CSV
# =============================================================================

def load_head_contrib(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)

    required = {
        "sid",
        "causal_text_rank",
        "causal_layer",
        "causal_position",
        "supplier_layer",
        "head",
        "rg_mediation",
    }
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(
            f"{path} missing columns: {sorted(missing)}"
        )

    for c in (
        "sid",
        "causal_text_rank",
        "causal_layer",
        "causal_position",
        "supplier_layer",
        "head",
    ):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)

    d["rg_mediation"] = pd.to_numeric(
        d["rg_mediation"],
        errors="coerce",
    )

    return d


# =============================================================================
# Summaries
# =============================================================================

def summarize(df):
    if len(df) == 0:
        return pd.DataFrame()

    rows = []

    for scale, g in df.groupby("scale"):
        rows.append({
            "scale": float(scale),
            "N_states": len(g),
            "N_samples": int(g["sid"].nunique()),

            "mean_positive_head_N": safe_mean(g["positive_head_N"]),
            "median_positive_head_N": safe_median(g["positive_head_N"]),

            "mean_cos_move_vs_RG": safe_mean(g["cos_move_vs_RG"]),
            "median_cos_move_vs_RG": safe_median(g["cos_move_vs_RG"]),

            "mean_projection_on_RG": safe_mean(g["projection_on_RG"]),
            "median_projection_on_RG": safe_median(g["projection_on_RG"]),

            "mean_move_norm": safe_mean(g["move_norm"]),
            "median_move_norm": safe_median(g["move_norm"]),

            "mean_RG_norm": safe_mean(g["RG_norm"]),
            "median_RG_norm": safe_median(g["RG_norm"]),

            "mean_move_over_RG_norm": safe_mean(g["move_over_RG_norm"]),
            "median_move_over_RG_norm": safe_median(g["move_over_RG_norm"]),

            "positive_cos_fraction": float(
                np.mean(g["cos_move_vs_RG"] > 0)
            ),
            "cos_gt_0p25_fraction": float(
                np.mean(g["cos_move_vs_RG"] > 0.25)
            ),
            "cos_gt_0p5_fraction": float(
                np.mean(g["cos_move_vs_RG"] > 0.5)
            ),
            "cos_gt_0p75_fraction": float(
                np.mean(g["cos_move_vs_RG"] > 0.75)
            ),
        })

    return pd.DataFrame(rows).sort_values("scale")


def grouped_summary(df, key):
    outs = []

    for val, g in df.groupby(key):
        z = summarize(g)
        if len(z):
            z.insert(0, key, val)
            outs.append(z)

    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers = parse_layers(a.causal_layers)
    categories = parse_categories(a.causal_categories)
    scales = parse_floats(a.scales)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    error_path = outdir / "errors.jsonl"

    contrib = load_head_contrib(Path(a.head_contrib))

    two, meta, rec_by_sid = load_coco_meta(a)
    meta_by_sid = {int(x["sid"]): x for x in meta}

    ranking_sids = set(
        pd.to_numeric(
            pd.read_csv(a.ranked_causal, usecols=["sid"])["sid"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )

    contrib_sids = set(contrib["sid"].astype(int).tolist())

    eval_meta = [
        x for x in meta
        if int(x["sid"]) in ranking_sids
        and int(x["sid"]) in contrib_sids
    ]

    if a.eval_max_samples > 0:
        eval_meta = traj.stratified_cap(
            eval_meta,
            a.eval_max_samples,
            a.seed + 1,
        )

    eval_sids = {int(x["sid"]) for x in eval_meta}

    causal = load_selected_causal(
        Path(a.ranked_causal),
        allowed_sids=eval_sids,
        causal_layers=causal_layers,
        top_k=a.causal_top_k,
        categories=categories,
    )

    causal.to_csv(
        outdir / "selected_causal_text_states.csv",
        index=False,
    )

    # Keep exact same states as selected causal ranking.
    causal_keys = set(
        (
            int(r.sid),
            int(r.causal_text_rank),
            int(r.source_layer),
            int(r.position),
        )
        for r in causal.itertuples()
    )

    contrib = contrib[
        contrib.apply(
            lambda r: (
                int(r.sid),
                int(r.causal_text_rank),
                int(r.causal_layer),
                int(r.causal_position),
            ) in causal_keys,
            axis=1,
        )
    ].copy()

    positive = contrib[
        contrib["rg_mediation"] > float(a.positive_threshold)
    ].copy()

    positive.to_csv(
        outdir / "selected_positive_supplier_heads.csv",
        index=False,
    )

    pos_by_state = {}
    for key, g in positive.groupby(
        [
            "sid",
            "causal_text_rank",
            "causal_layer",
            "causal_position",
        ]
    ):
        pos_by_state[tuple(map(int, key))] = [
            (int(r.supplier_layer), int(r.head))
            for r in g.itertuples()
        ]

    all_head_layers = sorted(
        set(
            int(x)
            for x in positive["supplier_layer"].tolist()
        )
    )

    model = processor = None

    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(a, two)
        device = torch.device(a.device)

        geom = infer_head_geometry(
            model,
            decoder_layers,
            all_head_layers,
        )

        print("=" * 180)
        print("ALL POSITIVE HEAD SUPPLIERS -> RECONSTRUCT EACH CAUSAL RG")
        print("=" * 180)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"eval N={len(eval_meta)}")
        print(
            f"positive criterion: rg_mediation > {a.positive_threshold}"
        )
        print(f"scales={scales}")
        print(
            "IMPORTANT: each causal state is patched independently; "
            "other causal token positions are not patched."
        )
        print()

        rows = []
        count_rows = []

        causal_by_sid = {
            int(sid): g.copy()
            for sid, g in causal.groupby("sid")
        }

        for m in tqdm(eval_meta, desc="ALL POSITIVE SUPPLIERS"):
            sid = int(m["sid"])
            if sid not in causal_by_sid:
                continue

            real = gray = rb = gb = None

            try:
                crows = causal_by_sid[sid].sort_values("causal_text_rank")

                state_layers = sorted(
                    set(crows["source_layer"].astype(int).tolist())
                )
                max_C = max(state_layers)

                head_layers_sid = [
                    L for L in all_head_layers if L <= max_C
                ]

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
                    raise RuntimeError("Real/Gray tokenization mismatch")

                hr, pre_r = capture_states_pre_o(
                    model,
                    decoder_layers,
                    rb,
                    state_layers=state_layers,
                    head_layers=head_layers_sid,
                )

                hg, pre_g = capture_states_pre_o(
                    model,
                    decoder_layers,
                    gb,
                    state_layers=state_layers,
                    head_layers=head_layers_sid,
                )

                for crow in crows.itertuples():
                    C = int(crow.source_layer)
                    p = int(crow.position)
                    cr = int(crow.causal_text_rank)

                    key = (sid, cr, C, p)
                    heads = pos_by_state.get(key, [])

                    # Enforce causal ordering.
                    heads = [(L,h) for L,h in heads if L <= C]

                    count_rows.append({
                        "sid": sid,
                        "gt": m["gt"],
                        "causal_text_rank": cr,
                        "causal_layer": C,
                        "causal_position": p,
                        "causal_token": str(crow.token),
                        "causal_broad_category": str(crow.broad_category),
                        "positive_head_N": len(heads),
                    })

                    if not heads:
                        continue

                    patch_map, msg_norms = build_single_state_patch_map(
                        decoder_layers=decoder_layers,
                        real_pre_o=pre_r,
                        gray_pre_o=pre_g,
                        head_geometry=geom,
                        heads=heads,
                        causal_layer=C,
                        causal_position=p,
                    )

                    if not patch_map:
                        continue

                    real_vec = hr[C][0, p].astype(np.float32)
                    gray_vec = hg[C][0, p].astype(np.float32)

                    rg = real_vec - gray_vec
                    rg_norm = float(np.linalg.norm(rg))

                    for alpha in scales:
                        hp = patched_state(
                            model=model,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            causal_layer=C,
                            patch_map=patch_map,
                            scale=alpha,
                        )

                        patch_vec = hp[0, p].astype(np.float32)
                        move = patch_vec - real_vec

                        move_norm = float(np.linalg.norm(move))
                        cos = cosine(move, rg)

                        proj = (
                            float(np.dot(move, rg / rg_norm))
                            if rg_norm > EPS
                            else np.nan
                        )

                        residual = move - rg
                        residual_norm = float(np.linalg.norm(residual))

                        rows.append({
                            "sid": sid,
                            "gt": m["gt"],

                            "causal_text_rank": cr,
                            "global_rank": int(crow.rank),
                            "causal_layer": C,
                            "causal_position": p,
                            "causal_token": str(crow.token),
                            "causal_category": str(crow.category),
                            "causal_broad_category": str(crow.broad_category),

                            "scale": float(alpha),

                            "positive_head_N": len(heads),
                            "positive_head_names": ",".join(
                                hname(L,h) for L,h in heads
                            ),

                            "mean_selected_message_norm": safe_mean(msg_norms),
                            "sum_selected_message_norm": float(np.sum(msg_norms)),

                            "RG_norm": rg_norm,
                            "move_norm": move_norm,
                            "move_over_RG_norm": (
                                move_norm / rg_norm
                                if rg_norm > EPS else np.nan
                            ),

                            "cos_move_vs_RG": cos,
                            "projection_on_RG": proj,

                            # Exact vector reconstruction error:
                            # ||MOVE - RG|| / ||RG||
                            "relative_reconstruction_error": (
                                residual_norm / rg_norm
                                if rg_norm > EPS else np.nan
                            ),
                        })

            except Exception as exc:
                append_jsonl(error_path, {
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

        df = pd.DataFrame(rows)
        counts = pd.DataFrame(count_rows)

        df.to_csv(
            outdir / "per_state_results.csv",
            index=False,
        )
        counts.to_csv(
            outdir / "positive_supplier_counts.csv",
            index=False,
        )

        sm = summarize(df)
        sm.to_csv(
            outdir / "summary.csv",
            index=False,
        )

        by_layer = grouped_summary(df, "causal_layer")
        by_layer.to_csv(
            outdir / "summary_by_causal_layer.csv",
            index=False,
        )

        by_cat = grouped_summary(df, "causal_broad_category")
        by_cat.to_csv(
            outdir / "summary_by_category.csv",
            index=False,
        )

        report = []
        report.append("=" * 180)
        report.append("ALL POSITIVE SUPPLIERS -> CAUSAL RG RECONSTRUCTION")
        report.append("=" * 180)
        report.append(
            f"completed samples={df['sid'].nunique() if len(df) else 0} | "
            f"states={len(counts)}"
        )
        report.append(
            f"positive head count: mean={safe_mean(counts['positive_head_N']):.2f} "
            f"median={safe_median(counts['positive_head_N']):.2f}"
            if len(counts) else "no state counts"
        )
        report.append("")

        report.append("SUMMARY")
        report.append("-" * 180)
        if len(sm):
            report.append(
                sm.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )
        report.append("")

        report.append("INTERPRETATION")
        report.append("-" * 180)
        report.append(
            "If alpha=1 gives cos~1, norm ratio~1, reconstruction error~0, "
            "then positive attention-head suppliers nearly reconstruct the "
            "causal token's full Real-Gray displacement."
        )
        report.append(
            "If cosine remains low despite large norm ratio, the positive "
            "first-order head contributions explain projection on RG but not "
            "the full vector; MLP/residual/nonlinear and orthogonal components "
            "remain essential."
        )

        report_text = "\n".join(report) + "\n"
        print()
        print(report_text)

        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(
            outdir / "metadata.json",
            {
                "script": "eval_all_positive_suppliers_reconstruct_causal_rg_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "head_contrib": str(a.head_contrib),
                "ranked_causal": str(a.ranked_causal),
                "causal_layers": causal_layers,
                "causal_top_k": a.causal_top_k,
                "positive_threshold": a.positive_threshold,
                "scales": scales,
                "eval_N_requested": a.eval_max_samples,
                "patch_mode": "one_causal_state_at_a_time",
                "selection": "all heads with per-state rg_mediation > threshold",
                "comparison": (
                    "MOVE=h_patch[C,p]-h_real[C,p] vs "
                    "RG=h_real[C,p]-h_gray[C,p]"
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
