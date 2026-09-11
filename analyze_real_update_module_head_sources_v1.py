
# -*- coding: utf-8 -*-

"""
analyze_real_update_module_head_sources_v1.py

Goal
====
Decompose the ACTUAL REAL causal-token update at each layer into:

    REAL block update
        = Attention residual contribution
        + MLP residual contribution

and then decompose the attention residual exactly into per-head post-W_O
messages.

This directly extends:
    eval_real_causal_token_update_gating_v1.py

It can READ that run's all-440 outputs to reuse:
  * exact evaluated sample IDs
  * exact Top-K selected causal states
  * baseline prediction/correctness
  * exact GT competitor used by the sequence-margin analysis
  * previous per-update B_REAL values for replay validation

But the old CSVs do NOT contain attention/MLP activations or pre-W_O head
outputs. Therefore Attention-vs-MLP and per-head decomposition require NEW
model forward/backward passes. No new training is performed.

Primary definitions
===================
For sample i, causal-token position p, decoder block L:

    a_REAL[L,p] = block_out[L,p] - block_in[L,p]

For Qwen2.5 decoder blocks this is exactly:

    a_REAL[L,p] = a_ATTN[L,p] + a_MLP[L,p]

where:
    a_ATTN = self-attention module output after W_O, before residual addition
    a_MLP  = MLP module output before residual addition

Use the same sequence-level decision margin as the prior experiment:

    M = S_GT - S_competitor

and the gradient at the block output:

    g[L,p] = dM / d block_out[L,p]

Decision contributions:

    B_REAL = <a_REAL, g>
    B_ATTN = <a_ATTN, g>
    B_MLP  = <a_MLP,  g>

Thus:

    B_REAL ~= B_ATTN + B_MLP

Interpretation:
    B > 0: realized residual contribution locally supports GT
    B < 0: realized residual contribution locally supports competitor

Per-head decomposition
======================
Let z_h be head h's pre-W_O vector at query position p and W_O,h the
corresponding slice of the output projection:

    m_h = W_O,h z_h

Then:

    a_ATTN ~= sum_h m_h (+ optional o_proj bias)

and:

    B_HEAD[h] = <m_h, g>

so:

    B_ATTN ~= sum_h B_HEAD[h] (+ bias contribution)

This is an exact additive decomposition of the REALIZED residual update.
It is attribution of the realized block update, not yet a causal ablation:
attention also influences the MLP input inside the same block.

Spatial-head annotations
========================
Default direction-head list:
  L26H03,L23H01,L23H05,L26H02,L22H09,L23H00,L22H13,L22H02,
  L21H14,L23H10,L21H05,L22H12,L21H01,L26H01,L27H02,L27H01,
  L21H11,L22H14,L21H03,L22H10

Default centroid-head list:
  L27H10,L24H05,L28H08,L20H05,L31H07

Only heads in analyzed layers are used. Both lists can be overridden.

Recommended run using the existing full-440 experiment
=======================================================
CUDA_VISIBLE_DEVICES=0 python -u analyze_real_update_module_head_sources_v1.py \
  --model qwen-3b \
  --real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --trace-layers 8-26 \
  --output-dir output/qwen3b_real_update_module_head_sources_all440_v1 \
  --overwrite

For a quick N=40 validation first:
  --max-samples 40

Outputs
=======
per_module_decision_contribution.csv
per_head_decision_contribution.csv
module_summary_by_layer.csv
module_summary_by_relation_layer.csv
module_summary_by_correctness_layer.csv
head_summary_global.csv
head_summary_by_relation.csv
head_summary_by_layer.csv
spatial_head_group_summary.csv
spatial_head_group_by_relation.csv
top_positive_heads.csv
top_negative_heads.csv
replay_validation.csv
analysis_summary.txt
metadata.json
errors.jsonl

No interventions are performed here. This first asks WHERE the positive and
negative realized causal-token updates come from.
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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import scan_qwen_spatial_heads_vs_causal_core_v1 as spatialscan


REL = ("left", "right", "above", "below")
EPS = 1e-12

DEFAULT_DIRECTION_HEADS = (
    "L26H03,L23H01,L23H05,L26H02,L22H09,"
    "L23H00,L22H13,L22H02,L21H14,L23H10,"
    "L21H05,L22H12,L21H01,L26H01,L27H02,"
    "L27H01,L21H11,L22H14,L21H03,L22H10"
)

DEFAULT_CENTROID_HEADS = (
    "L27H10,L24H05,L28H08,L20H05,L31H07"
)


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
        "--real-update-dir",
        required=True,
        help=(
            "Output directory from eval_real_causal_token_update_gating_v1.py. "
            "Must contain selected_causal_states.csv, generation_per_sample.csv, "
            "sequence_score_summary.csv. per_real_update_decision_score.csv is "
            "optional but strongly recommended for replay validation."
        ),
    )

    p.add_argument(
        "--trace-layers",
        default="8-26",
        help="Decoder layers whose realized causal-token updates are decomposed.",
    )
    p.add_argument(
        "--exclude-target-layer",
        action="store_true",
        help="Analyze only L < latest selected causal target layer for each token position.",
    )

    p.add_argument("--direction-heads", default=DEFAULT_DIRECTION_HEADS)
    p.add_argument("--centroid-heads", default=DEFAULT_CENTROID_HEADS)

    p.add_argument(
        "--top-heads",
        type=int,
        default=30,
        help="Number of globally strongest positive/negative heads to save.",
    )
    p.add_argument(
        "--top-heads-per-relation",
        type=int,
        default=15,
        help="Number of strongest positive/negative heads per relation.",
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 uses every successfully evaluated sample in real-update-dir.",
    )
    p.add_argument("--seed", type=int, default=17)

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


def parse_head_set(text: str):
    return {
        x.strip().upper()
        for x in str(text).split(",")
        if x.strip()
    }


def hname(layer, head):
    return f"L{int(layer)}H{int(head):02d}"


def normalize_rel(x):
    s = str(x).strip().lower()
    aliases = {
        "left": "left",
        "right": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "below": "below",
        "under": "below",
    }
    return aliases.get(s, s)


def display_rel(x):
    r = normalize_rel(x)
    return {
        "left": "left",
        "right": "right",
        "above": "on",
        "below": "under",
    }.get(r, r)


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def safe_ratio(a, b):
    return float(a) / max(float(b), EPS)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_output_dir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load prior all-440 result metadata
# =============================================================================

def load_prior_run(run_dir: Path, max_samples: int, seed: int):
    selected_path = run_dir / "selected_causal_states.csv"
    generation_path = run_dir / "generation_per_sample.csv"
    seq_path = run_dir / "sequence_score_summary.csv"
    prior_update_path = run_dir / "per_real_update_decision_score.csv"
    metadata_path = run_dir / "metadata.json"

    for p in (selected_path, generation_path, seq_path):
        if not p.exists():
            raise FileNotFoundError(p)

    selected = pd.read_csv(selected_path)
    gen = pd.read_csv(generation_path)
    seq = pd.read_csv(seq_path)

    required_selected = {
        "sid", "rank", "source_layer", "position",
        "token", "category", "broad_category",
    }
    miss = required_selected - set(selected.columns)
    if miss:
        raise RuntimeError(
            f"{selected_path} missing columns {sorted(miss)}"
        )

    for c in ("sid", "rank", "source_layer", "position"):
        selected[c] = pd.to_numeric(
            selected[c], errors="raise"
        ).astype(int)

    # Exact baseline cohort from the successful prior run.
    basegen = gen[gen["condition"].astype(str) == "baseline"].copy()
    basegen["sid"] = pd.to_numeric(
        basegen["sid"], errors="raise"
    ).astype(int)
    basegen = basegen.sort_values("sid").drop_duplicates("sid")
    basegen["gt"] = basegen["gt"].map(normalize_rel)
    basegen["prediction"] = basegen["prediction"].map(normalize_rel)

    if basegen["correct"].dtype != bool:
        basegen["correct"] = (
            basegen["correct"].astype(str).str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    seq["sid"] = pd.to_numeric(seq["sid"], errors="raise").astype(int)
    seq["gt"] = seq["gt"].map(normalize_rel)
    seq["competitor"] = seq["competitor"].map(normalize_rel)

    cohort = basegen.merge(
        seq[["sid", "competitor", "sequence_margin"]],
        on="sid",
        how="inner",
    )

    # Optional deterministic cap for quick checks.
    if max_samples > 0 and len(cohort) > max_samples:
        rng = random.Random(seed)
        keep = []
        for _, g in cohort.groupby(["gt", "correct"], dropna=False):
            sids = sorted(g["sid"].astype(int).tolist())
            rng.shuffle(sids)
            frac = max_samples / len(cohort)
            n = max(1, int(round(len(sids) * frac)))
            keep.extend(sids[:n])
        keep = sorted(set(keep))
        if len(keep) > max_samples:
            rng.shuffle(keep)
            keep = keep[:max_samples]
        elif len(keep) < max_samples:
            remaining = [
                x for x in cohort["sid"].astype(int).tolist()
                if x not in set(keep)
            ]
            rng.shuffle(remaining)
            keep.extend(remaining[: max_samples - len(keep)])
        cohort = cohort[cohort["sid"].isin(set(keep))].copy()

    sids = set(cohort["sid"].astype(int))
    selected = selected[selected["sid"].isin(sids)].copy()

    # Keep original Top-K selection exactly as saved.
    selected_by_sid = {
        int(sid): g.sort_values("rank").copy()
        for sid, g in selected.groupby("sid")
    }

    prior_update = None
    if prior_update_path.exists():
        prior_update = pd.read_csv(prior_update_path)
        for c in ("sid", "update_layer", "real_position"):
            if c in prior_update.columns:
                prior_update[c] = pd.to_numeric(
                    prior_update[c], errors="raise"
                ).astype(int)
        prior_update = prior_update[
            prior_update["sid"].isin(sids)
        ].copy()

    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except Exception:
            metadata = {}

    return cohort, selected_by_sid, prior_update, metadata


def causal_position_specs(causal_rows):
    specs = {}

    for r in causal_rows.itertuples():
        p = int(r.position)
        C = int(r.source_layer)

        if p not in specs:
            specs[p] = {
                "real_position": p,
                "max_target_layer": C,
                "min_target_layer": C,
                "best_rank": int(r.rank),
                "token": str(r.token),
                "category": str(r.category),
                "broad_category": str(r.broad_category),
            }
        else:
            specs[p]["max_target_layer"] = max(
                specs[p]["max_target_layer"], C
            )
            specs[p]["min_target_layer"] = min(
                specs[p]["min_target_layer"], C
            )
            if int(r.rank) < specs[p]["best_rank"]:
                specs[p]["best_rank"] = int(r.rank)
                specs[p]["token"] = str(r.token)
                specs[p]["category"] = str(r.category)
                specs[p]["broad_category"] = str(r.broad_category)

    return list(specs.values())


# =============================================================================
# Data / model
# =============================================================================

def load_dataset_for_sids(a, cohort):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for row in cohort.itertuples():
        sid = int(row.sid)
        if sid not in prompts or sid not in rec_by_sid:
            continue

        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        gt = normalize_rel(gt)

        meta.append(
            {
                "sid": sid,
                "gt": gt,
                "question_text": str(p["question_text"]),
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "baseline_prediction": normalize_rel(row.prediction),
                "baseline_correct": bool(row.correct),
                "competitor": normalize_rel(row.competitor),
                "prior_sequence_margin": float(row.sequence_margin),
            }
        )

    return two, meta, rec_by_sid


def load_model(a, two):
    spec = base.merged_model_specs(two)[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    print(f"[model] loading {spec.repo_id}", flush=True)
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


# =============================================================================
# Model internals
# =============================================================================

def resolve_mlp(layer):
    for name in ("mlp", "feed_forward", "ffn"):
        x = getattr(layer, name, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError(
        f"Could not resolve MLP module on {type(layer).__name__}"
    )


def first_tensor_any(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            if torch.is_tensor(y):
                return y
    if isinstance(x, dict):
        for y in x.values():
            if torch.is_tensor(y):
                return y
    raise RuntimeError(f"No tensor found in {type(x).__name__}")


def replace_first_tensor(output, tensor):
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise RuntimeError(
        f"Unsupported output type: {type(output).__name__}"
    )


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
                f"L{L}: o_proj.in_features={width} incompatible "
                f"with heads={H}"
            )

        out[L] = {
            "n_heads": H,
            "head_dim": width // H,
            "pre_o_width": width,
        }

    return out


# =============================================================================
# Candidate sequence score
# =============================================================================

def tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def get_candidate_texts(metadata):
    x = metadata.get("candidate_texts", None)
    if isinstance(x, dict) and all(r in x for r in REL):
        return {r: str(x[r]) for r in REL}

    # Fallback matches the prior recommended run.
    return {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
    }


def encode_candidate_ids(processor, texts):
    tok = tokenizer_of(processor)
    out = {}

    print("\nCandidate answer tokenizations:")
    for r in REL:
        ids = tok.encode(texts[r], add_special_tokens=False)
        if not ids:
            raise RuntimeError(
                f"Empty tokenization for relation {r}: {texts[r]!r}"
            )
        out[r] = list(map(int, ids))
        print(
            f"  {r:>5s}: text={texts[r]!r} "
            f"ids={out[r]} decoded={tok.decode(out[r])!r}"
        )
    return out


def extend_batch_with_candidate(batch, answer_ids):
    out = {}
    T = int(batch["input_ids"].shape[1])
    dev = batch["input_ids"].device

    add = torch.tensor(
        [list(map(int, answer_ids))],
        dtype=batch["input_ids"].dtype,
        device=dev,
    )
    out["input_ids"] = torch.cat(
        [batch["input_ids"], add],
        dim=1,
    )

    for k, v in batch.items():
        if k == "input_ids":
            continue
        if k in ("position_ids", "cache_position"):
            continue

        if k == "attention_mask" and torch.is_tensor(v):
            ones = torch.ones(
                (v.shape[0], len(answer_ids)),
                dtype=v.dtype,
                device=v.device,
            )
            out[k] = torch.cat([v, ones], dim=1)

        elif k == "token_type_ids" and torch.is_tensor(v):
            tail = v[:, -1:].expand(
                v.shape[0],
                len(answer_ids),
            )
            out[k] = torch.cat([v, tail], dim=1)

        else:
            out[k] = v

    return out, T


def sequence_score_from_logits(
    logits,
    prompt_len,
    answer_ids,
    reduction,
):
    terms = []
    for j, tid in enumerate(answer_ids):
        idx = prompt_len - 1 + j
        lp = torch.log_softmax(
            logits[0, idx].float(),
            dim=-1,
        )[int(tid)]
        terms.append(lp)

    score = torch.stack(terms).sum()
    if reduction == "mean":
        score = score / max(len(terms), 1)
    return score


# =============================================================================
# Combined gradient + realized module/head capture
# =============================================================================

class DecisionFormationCapture:
    """
    During the GT answer forward:
      - capture block inputs at selected positions
      - capture attention residual outputs after W_O
      - capture MLP residual outputs
      - capture pre-W_O attention vectors
      - keep block-output graph tensors for gradient

    Autograd is cut at the earliest requested block output to avoid carrying
    graph through vision and early decoder blocks.
    """

    def __init__(
        self,
        decoder_layers,
        layers,
        positions,
        capture_modules=True,
    ):
        self.layers = sorted(set(map(int, layers)))
        self.positions = sorted(set(map(int, positions)))
        self.capture_modules = bool(capture_modules)

        self.block_in = {}
        self.attn_out = {}
        self.mlp_out = {}
        self.pre_o = {}
        self.block_out_cpu = {}
        self.block_out_graph = {}
        self.handles = []

        if not self.layers:
            raise ValueError("No layers requested")

        cut = min(self.layers)

        def take_positions(x):
            if x.ndim != 3 or int(x.shape[0]) != 1:
                raise RuntimeError(
                    f"Expected [1,S,D], got {tuple(x.shape)}"
                )

            good = [
                p for p in self.positions
                if 0 <= p < int(x.shape[1])
            ]
            if len(good) != len(self.positions):
                bad = sorted(set(self.positions) - set(good))
                raise RuntimeError(
                    f"Positions outside sequence: {bad}; S={x.shape[1]}"
                )

            idx = torch.as_tensor(
                good,
                device=x.device,
                dtype=torch.long,
            )
            return (
                x[0].index_select(0, idx)
                .detach().float().cpu().numpy().astype(np.float32)
            )

        # Cut first. Later block-output capture hooks see the replaced tensor.
        def cut_hook(_m, _inp, out):
            x = first_tensor_any(out)
            y = x.detach().clone().requires_grad_(True)
            return replace_first_tensor(out, y)

        self.handles.append(
            decoder_layers[cut].register_forward_hook(cut_hook)
        )

        for L in self.layers:
            block = decoder_layers[L]

            if self.capture_modules:
                attn = spatialscan.resolve_attn(block)
                op = spatialscan.resolve_o_proj(attn)
                mlp = resolve_mlp(block)

                def make_block_pre(layer):
                    def hook(_m, inputs):
                        x = first_tensor_any(inputs)
                        self.block_in[layer] = take_positions(x)
                    return hook

                def make_attn_hook(layer):
                    def hook(_m, _inp, out):
                        x = first_tensor_any(out)
                        self.attn_out[layer] = take_positions(x)
                        return None
                    return hook

                def make_pre_o_hook(layer):
                    def hook(_m, inputs):
                        x = first_tensor_any(inputs)
                        self.pre_o[layer] = take_positions(x)
                    return hook

                def make_mlp_hook(layer):
                    def hook(_m, _inp, out):
                        x = first_tensor_any(out)
                        self.mlp_out[layer] = take_positions(x)
                        return None
                    return hook

                self.handles.append(
                    block.register_forward_pre_hook(make_block_pre(L))
                )
                self.handles.append(
                    attn.register_forward_hook(make_attn_hook(L))
                )
                self.handles.append(
                    op.register_forward_pre_hook(make_pre_o_hook(L))
                )
                self.handles.append(
                    mlp.register_forward_hook(make_mlp_hook(L))
                )

            def make_block_graph(layer):
                def hook(_m, _inp, out):
                    x = first_tensor_any(out)
                    self.block_out_graph[layer] = x
                    if self.capture_modules:
                        self.block_out_cpu[layer] = take_positions(x)
                    return None
                return hook

            self.handles.append(
                block.register_forward_hook(make_block_graph(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def run_answer_with_grad_and_optional_capture(
    *,
    model,
    decoder_layers,
    batch,
    answer_ids,
    reduction,
    layers,
    positions,
    capture_modules,
):
    ext, T = extend_batch_with_candidate(
        batch,
        answer_ids,
    )

    cap = DecisionFormationCapture(
        decoder_layers=decoder_layers,
        layers=layers,
        positions=positions,
        capture_modules=capture_modules,
    )

    try:
        with torch.enable_grad():
            kw = dict(ext)
            kw["use_cache"] = False
            kw["return_dict"] = True
            out = model(**kw)

            missing_graph = [
                L for L in layers
                if L not in cap.block_out_graph
            ]
            if missing_graph:
                raise RuntimeError(
                    f"Missing graph outputs: {missing_graph}"
                )

            if capture_modules:
                missing_mod = []
                for L in layers:
                    for name, store in (
                        ("block_in", cap.block_in),
                        ("attn_out", cap.attn_out),
                        ("mlp_out", cap.mlp_out),
                        ("pre_o", cap.pre_o),
                        ("block_out_cpu", cap.block_out_cpu),
                    ):
                        if L not in store:
                            missing_mod.append((L, name))
                if missing_mod:
                    raise RuntimeError(
                        f"Missing module captures: {missing_mod[:20]}"
                    )

            score = sequence_score_from_logits(
                out.logits,
                T,
                answer_ids,
                reduction,
            )

            tensors = [
                cap.block_out_graph[L]
                for L in layers
            ]
            grads = torch.autograd.grad(
                score,
                tensors,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            grad_by_layer = {}
            for L, g in zip(layers, grads):
                grad_by_layer[L] = (
                    None
                    if g is None
                    else g.detach().float().cpu().numpy().astype(np.float32)
                )

            module_caps = None
            if capture_modules:
                module_caps = {
                    "block_in": dict(cap.block_in),
                    "attn_out": dict(cap.attn_out),
                    "mlp_out": dict(cap.mlp_out),
                    "pre_o": dict(cap.pre_o),
                    "block_out": dict(cap.block_out_cpu),
                    "positions": list(cap.positions),
                }

            return (
                float(score.detach().item()),
                grad_by_layer,
                module_caps,
            )

    finally:
        cap.close()


# =============================================================================
# Per-sample decomposition
# =============================================================================

def position_index(module_caps, position):
    return module_caps["positions"].index(int(position))


def row_module_class(b_attn, b_mlp):
    ap = max(float(b_attn), 0.0)
    mp = max(float(b_mlp), 0.0)
    an = max(-float(b_attn), 0.0)
    mn = max(-float(b_mlp), 0.0)

    if ap + mp > EPS:
        positive_source = (
            "attention" if ap > mp
            else "mlp" if mp > ap
            else "tie"
        )
    else:
        positive_source = "none"

    if an + mn > EPS:
        negative_source = (
            "attention" if an > mn
            else "mlp" if mn > an
            else "tie"
        )
    else:
        negative_source = "none"

    return positive_source, negative_source


def analyze_sample(
    *,
    sid,
    gt,
    baseline_prediction,
    baseline_correct,
    competitor,
    prior_sequence_margin,
    causal_rows,
    trace_layers,
    exclude_target_layer,
    decoder_layers,
    geom,
    direction_heads,
    centroid_heads,
    gt_score,
    comp_score,
    grad_gt,
    grad_comp,
    module_caps,
):
    specs = causal_position_specs(causal_rows)

    module_rows = []
    head_rows = []

    score_margin = float(gt_score - comp_score)

    for s in specs:
        p = int(s["real_position"])
        Cmax = int(s["max_target_layer"])
        pi = position_index(module_caps, p)

        for L in trace_layers:
            L = int(L)

            if exclude_target_layer:
                if L >= Cmax:
                    continue
            else:
                if L > Cmax:
                    continue

            gg = grad_gt.get(L, None)
            gc = grad_comp.get(L, None)
            if gg is None or gc is None:
                continue
            if not (
                0 <= p < gg.shape[1]
                and 0 <= p < gc.shape[1]
            ):
                continue

            g = (
                gg[0, p].astype(np.float32)
                - gc[0, p].astype(np.float32)
            )

            x_in = module_caps["block_in"][L][pi].astype(np.float32)
            x_out = module_caps["block_out"][L][pi].astype(np.float32)
            a_attn = module_caps["attn_out"][L][pi].astype(np.float32)
            a_mlp = module_caps["mlp_out"][L][pi].astype(np.float32)

            # Primary definition: the ACTUAL realized block update.
            a_real = (x_out - x_in).astype(np.float32)
            a_module_sum = (a_attn + a_mlp).astype(np.float32)
            vector_closure_rel = float(
                np.linalg.norm(a_real - a_module_sum)
                / max(float(np.linalg.norm(a_real)), EPS)
            )
            B_real = float(np.dot(a_real, g))
            B_attn = float(np.dot(a_attn, g))
            B_mlp = float(np.dot(a_mlp, g))

            positive_source, negative_source = row_module_class(
                B_attn,
                B_mlp,
            )

            module_rows.append(
                {
                    "sid": int(sid),
                    "gt": gt,
                    "relation": display_rel(gt),
                    "baseline_prediction": baseline_prediction,
                    "baseline_prediction_display": display_rel(
                        baseline_prediction
                    ),
                    "baseline_correct": bool(baseline_correct),
                    "competitor": competitor,
                    "competitor_display": display_rel(competitor),
                    "prior_sequence_margin": float(prior_sequence_margin),
                    "replayed_sequence_margin": score_margin,

                    "real_position": p,
                    "token": str(s["token"]),
                    "category": str(s["category"]),
                    "broad_category": str(s["broad_category"]),
                    "max_target_layer": Cmax,
                    "update_layer": L,
                    "distance_to_latest_target": Cmax - L,

                    "grad_norm": float(np.linalg.norm(g)),
                    "real_update_norm": float(np.linalg.norm(a_real)),
                    "attn_update_norm": float(np.linalg.norm(a_attn)),
                    "mlp_update_norm": float(np.linalg.norm(a_mlp)),

                    "B_real": B_real,
                    "B_attn": B_attn,
                    "B_mlp": B_mlp,
                    "B_module_sum": B_attn + B_mlp,
                    "B_module_closure_error": B_real - (B_attn + B_mlp),
                    "module_vector_closure_relative_error": vector_closure_rel,

                    "positive_module_source": positive_source,
                    "negative_module_source": negative_source,
                    "attention_positive_mass": max(B_attn, 0.0),
                    "mlp_positive_mass": max(B_mlp, 0.0),
                    "attention_negative_mass": max(-B_attn, 0.0),
                    "mlp_negative_mass": max(-B_mlp, 0.0),
                    "attention_abs_share": (
                        abs(B_attn) /
                        max(abs(B_attn) + abs(B_mlp), EPS)
                    ),
                    "mlp_abs_share": (
                        abs(B_mlp) /
                        max(abs(B_attn) + abs(B_mlp), EPS)
                    ),
                }
            )

            # -------------------------------------------------------------
            # Exact per-head decomposition of attention residual
            # -------------------------------------------------------------
            z = module_caps["pre_o"][L][pi].astype(np.float32)

            H = int(geom[L]["n_heads"])
            D = int(geom[L]["head_dim"])

            attn = spatialscan.resolve_attn(decoder_layers[L])
            op = spatialscan.resolve_o_proj(attn)

            W = (
                op.weight.detach().float().cpu()
                .numpy().astype(np.float32)
            )
            bias = getattr(op, "bias", None)
            if bias is None:
                bias_np = None
                bias_B = 0.0
            else:
                bias_np = (
                    bias.detach().float().cpu()
                    .numpy().astype(np.float32)
                )
                bias_B = float(np.dot(bias_np, g))

            head_sum = np.zeros_like(a_attn, dtype=np.float32)
            head_B_sum = 0.0

            for h in range(H):
                hh = hname(L, h)
                zh = z[h * D:(h + 1) * D]
                msg = (
                    W[:, h * D:(h + 1) * D] @ zh
                ).astype(np.float32)

                b = float(np.dot(msg, g))
                head_sum += msg
                head_B_sum += b

                is_direction = hh in direction_heads
                is_centroid = hh in centroid_heads

                if is_direction and is_centroid:
                    group = "direction+centroid"
                elif is_direction:
                    group = "direction"
                elif is_centroid:
                    group = "centroid"
                else:
                    group = "other"

                head_rows.append(
                    {
                        "sid": int(sid),
                        "gt": gt,
                        "relation": display_rel(gt),
                        "baseline_prediction": baseline_prediction,
                        "baseline_prediction_display": display_rel(
                            baseline_prediction
                        ),
                        "baseline_correct": bool(baseline_correct),
                        "competitor": competitor,
                        "competitor_display": display_rel(competitor),

                        "real_position": p,
                        "token": str(s["token"]),
                        "category": str(s["category"]),
                        "broad_category": str(s["broad_category"]),
                        "max_target_layer": Cmax,
                        "update_layer": L,
                        "distance_to_latest_target": Cmax - L,

                        "head": h,
                        "head_name": hh,
                        "head_group": group,
                        "is_direction_head": bool(is_direction),
                        "is_centroid_head": bool(is_centroid),
                        "is_spatial_head": bool(
                            is_direction or is_centroid
                        ),

                        "head_pre_o_norm": float(np.linalg.norm(zh)),
                        "head_message_norm": float(np.linalg.norm(msg)),
                        "head_decision_score": b,
                        "head_decision_cosine": (
                            float(
                                np.dot(msg, g) /
                                max(
                                    float(np.linalg.norm(msg))
                                    * float(np.linalg.norm(g)),
                                    EPS,
                                )
                            )
                        ),
                        "head_positive_mass": max(b, 0.0),
                        "head_negative_mass": max(-b, 0.0),

                        "layer_B_attn": B_attn,
                        "layer_B_mlp": B_mlp,
                        "layer_B_real": B_real,
                        "o_proj_bias_decision_score": bias_B,
                    }
                )

            reconstructed = head_sum.copy()
            if bias_np is not None:
                reconstructed += bias_np

            module_rows[-1].update(
                {
                    "head_sum_attn_relative_error": (
                        float(
                            np.linalg.norm(reconstructed - a_attn)
                        ) / max(float(np.linalg.norm(a_attn)), EPS)
                    ),
                    "head_B_sum": float(head_B_sum),
                    "o_proj_bias_B": float(bias_B),
                    "head_B_plus_bias": float(head_B_sum + bias_B),
                    "head_B_vs_attn_error": float(
                        (head_B_sum + bias_B) - B_attn
                    ),
                }
            )

    return module_rows, head_rows


# =============================================================================
# Prior replay validation
# =============================================================================

def make_prior_score_map(prior_update):
    if prior_update is None or len(prior_update) == 0:
        return {}

    needed = {
        "sid",
        "update_layer",
        "real_position",
        "real_update_decision_score",
    }
    if not needed.issubset(prior_update.columns):
        return {}

    out = {}
    for r in prior_update.itertuples():
        key = (
            int(r.sid),
            int(r.update_layer),
            int(r.real_position),
        )
        out[key] = float(r.real_update_decision_score)
    return out


def add_replay_validation(module_df, prior_map):
    if module_df.empty:
        return module_df, pd.DataFrame()

    z = module_df.copy()
    prior_vals = []
    for r in z.itertuples():
        prior_vals.append(
            prior_map.get(
                (
                    int(r.sid),
                    int(r.update_layer),
                    int(r.real_position),
                ),
                float("nan"),
            )
        )

    z["prior_B_real"] = prior_vals
    z["prior_B_real_error"] = (
        z["B_real"] - z["prior_B_real"]
    )

    good = z[np.isfinite(z["prior_B_real"])].copy()

    if len(good):
        summary = pd.DataFrame(
            [
                {
                    "N_matched_rows": len(good),
                    "mean_abs_B_error": float(
                        np.mean(np.abs(good["prior_B_real_error"]))
                    ),
                    "max_abs_B_error": float(
                        np.max(np.abs(good["prior_B_real_error"]))
                    ),
                    "sign_agreement": float(
                        np.mean(
                            np.sign(good["B_real"])
                            == np.sign(good["prior_B_real"])
                        )
                    ),
                    "pearson_B": float(
                        np.corrcoef(
                            good["B_real"].astype(float),
                            good["prior_B_real"].astype(float),
                        )[0, 1]
                    )
                    if len(good) > 1
                    else float("nan"),
                }
            ]
        )
    else:
        summary = pd.DataFrame()

    return z, summary


# =============================================================================
# Summaries
# =============================================================================

def module_summary(df, group_keys):
    if df.empty:
        return pd.DataFrame()

    rows = []

    for key, g in df.groupby(group_keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            k: v for k, v in zip(group_keys, key)
        }

        br = g["B_real"].astype(float).to_numpy()
        ba = g["B_attn"].astype(float).to_numpy()
        bm = g["B_mlp"].astype(float).to_numpy()

        real_pos = br > 0
        real_neg = br < 0

        row.update(
            {
                "N_samples": int(g["sid"].nunique()),
                "N_updates": int(len(g)),

                "mean_B_real": float(np.mean(br)),
                "mean_B_attn": float(np.mean(ba)),
                "mean_B_mlp": float(np.mean(bm)),

                "real_positive_fraction": float(np.mean(real_pos)),
                "real_negative_fraction": float(np.mean(real_neg)),
                "attn_positive_fraction": float(np.mean(ba > 0)),
                "attn_negative_fraction": float(np.mean(ba < 0)),
                "mlp_positive_fraction": float(np.mean(bm > 0)),
                "mlp_negative_fraction": float(np.mean(bm < 0)),

                "mean_attn_positive_mass": float(
                    np.mean(np.maximum(ba, 0))
                ),
                "mean_mlp_positive_mass": float(
                    np.mean(np.maximum(bm, 0))
                ),
                "mean_attn_negative_mass": float(
                    np.mean(np.maximum(-ba, 0))
                ),
                "mean_mlp_negative_mass": float(
                    np.mean(np.maximum(-bm, 0))
                ),

                # Conditioned on the sign of the REAL update.
                "positive_real_attn_positive_mass_share": (
                    float(
                        np.sum(np.maximum(ba[real_pos], 0))
                        / max(
                            np.sum(np.maximum(ba[real_pos], 0))
                            + np.sum(np.maximum(bm[real_pos], 0)),
                            EPS,
                        )
                    )
                    if np.any(real_pos)
                    else float("nan")
                ),
                "negative_real_attn_negative_mass_share": (
                    float(
                        np.sum(np.maximum(-ba[real_neg], 0))
                        / max(
                            np.sum(np.maximum(-ba[real_neg], 0))
                            + np.sum(np.maximum(-bm[real_neg], 0)),
                            EPS,
                        )
                    )
                    if np.any(real_neg)
                    else float("nan")
                ),

                "attention_dominates_positive_source_fraction": float(
                    np.mean(
                        g["positive_module_source"].astype(str)
                        == "attention"
                    )
                ),
                "mlp_dominates_positive_source_fraction": float(
                    np.mean(
                        g["positive_module_source"].astype(str)
                        == "mlp"
                    )
                ),
                "attention_dominates_negative_source_fraction": float(
                    np.mean(
                        g["negative_module_source"].astype(str)
                        == "attention"
                    )
                ),
                "mlp_dominates_negative_source_fraction": float(
                    np.mean(
                        g["negative_module_source"].astype(str)
                        == "mlp"
                    )
                ),

                "mean_module_vector_closure_error": safe_mean(
                    g["module_vector_closure_relative_error"]
                ),
                "mean_head_attn_reconstruction_error": safe_mean(
                    g["head_sum_attn_relative_error"]
                ),
                "mean_head_B_attn_error": safe_mean(
                    np.abs(g["head_B_vs_attn_error"])
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def head_summary(df, group_keys):
    if df.empty:
        return pd.DataFrame()

    rows = []

    keys = list(group_keys) + ["head_name"]

    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            k: v for k, v in zip(keys, key)
        }

        b = g["head_decision_score"].astype(float).to_numpy()

        first = g.iloc[0]
        row.update(
            {
                "update_layer": int(first["update_layer"]),
                "head": int(first["head"]),
                "head_group": str(first["head_group"]),
                "is_direction_head": bool(first["is_direction_head"]),
                "is_centroid_head": bool(first["is_centroid_head"]),
                "is_spatial_head": bool(first["is_spatial_head"]),

                "N_samples": int(g["sid"].nunique()),
                "N_messages": int(len(g)),
                "mean_head_decision_score": float(np.mean(b)),
                "median_head_decision_score": float(np.median(b)),
                "positive_fraction": float(np.mean(b > 0)),
                "negative_fraction": float(np.mean(b < 0)),
                "mean_positive_mass": float(
                    np.mean(np.maximum(b, 0))
                ),
                "mean_negative_mass": float(
                    np.mean(np.maximum(-b, 0))
                ),
                "mean_abs_score": float(np.mean(np.abs(b))),
                "mean_head_message_norm": safe_mean(
                    g["head_message_norm"]
                ),
                "mean_head_decision_cosine": safe_mean(
                    g["head_decision_cosine"]
                ),
            }
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            "mean_head_decision_score",
            ascending=False,
        ).reset_index(drop=True)
    return out


def spatial_group_summary(df, group_keys):
    if df.empty:
        return pd.DataFrame()

    rows = []
    keys = list(group_keys) + ["head_group"]

    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            k: v for k, v in zip(keys, key)
        }
        b = g["head_decision_score"].astype(float).to_numpy()

        row.update(
            {
                "N_unique_heads": int(g["head_name"].nunique()),
                "N_samples": int(g["sid"].nunique()),
                "N_messages": int(len(g)),
                "mean_score": float(np.mean(b)),
                "positive_fraction": float(np.mean(b > 0)),
                "negative_fraction": float(np.mean(b < 0)),
                "mean_positive_mass": float(
                    np.mean(np.maximum(b, 0))
                ),
                "mean_negative_mass": float(
                    np.mean(np.maximum(-b, 0))
                ),
                "mean_abs_score": float(np.mean(np.abs(b))),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def top_head_tables(
    head_global,
    head_by_relation,
    top_global,
    top_rel,
):
    if head_global.empty:
        return pd.DataFrame(), pd.DataFrame()

    pos_global = (
        head_global.sort_values(
            "mean_head_decision_score",
            ascending=False,
        )
        .head(top_global)
        .assign(rank_type="global_positive")
    )

    neg_global = (
        head_global.sort_values(
            "mean_head_decision_score",
            ascending=True,
        )
        .head(top_global)
        .assign(rank_type="global_negative")
    )

    pos_parts = [pos_global]
    neg_parts = [neg_global]

    if not head_by_relation.empty:
        for rel, g in head_by_relation.groupby("relation"):
            pos_parts.append(
                g.sort_values(
                    "mean_head_decision_score",
                    ascending=False,
                )
                .head(top_rel)
                .assign(rank_type=f"{rel}_positive")
            )
            neg_parts.append(
                g.sort_values(
                    "mean_head_decision_score",
                    ascending=True,
                )
                .head(top_rel)
                .assign(rank_type=f"{rel}_negative")
            )

    return (
        pd.concat(pos_parts, ignore_index=True),
        pd.concat(neg_parts, ignore_index=True),
    )


def enrichment_summary(head_df):
    """
    Compare direction/centroid prevalence among strongest positive/negative
    message rows against their prevalence among all head-message rows.
    """
    if head_df.empty:
        return pd.DataFrame()

    rows = []

    for scope_name, g in [("all", head_df)] + [
        (f"relation={r}", z)
        for r, z in head_df.groupby("relation")
    ]:
        base_dir = float(g["is_direction_head"].mean())
        base_cent = float(g["is_centroid_head"].mean())
        base_spat = float(g["is_spatial_head"].mean())

        for k in (1, 3, 5, 10):
            # Top-k per concrete (sample, layer, causal position) update.
            pos_parts = []
            neg_parts = []
            for _, q in g.groupby(
                ["sid", "update_layer", "real_position"]
            ):
                pos_parts.append(
                    q.nlargest(min(k, len(q)), "head_decision_score")
                )
                neg_parts.append(
                    q.nsmallest(min(k, len(q)), "head_decision_score")
                )

            pos = pd.concat(pos_parts, ignore_index=True)
            neg = pd.concat(neg_parts, ignore_index=True)

            for polarity, z in (
                ("positive", pos),
                ("negative", neg),
            ):
                for typ, col, base_rate in (
                    ("direction", "is_direction_head", base_dir),
                    ("centroid", "is_centroid_head", base_cent),
                    ("spatial_union", "is_spatial_head", base_spat),
                ):
                    rate = float(z[col].mean())
                    rows.append(
                        {
                            "scope": scope_name,
                            "polarity": polarity,
                            "top_k_per_update": k,
                            "head_type": typ,
                            "base_rate": base_rate,
                            "topk_rate": rate,
                            "enrichment": (
                                rate / base_rate
                                if base_rate > EPS
                                else float("nan")
                            ),
                            "N_topk_rows": len(z),
                        }
                    )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    run_dir = Path(a.real_update_dir)
    outdir = Path(a.output_dir)
    ensure_output_dir(outdir, a.overwrite)

    trace_layers = parse_layers(a.trace_layers)
    direction_heads = parse_head_set(a.direction_heads)
    centroid_heads = parse_head_set(a.centroid_heads)

    (
        cohort,
        selected_by_sid,
        prior_update,
        prior_metadata,
    ) = load_prior_run(
        run_dir,
        a.max_samples,
        a.seed,
    )

    two, meta, rec_by_sid = load_dataset_for_sids(
        a,
        cohort,
    )

    prior_map = make_prior_score_map(prior_update)

    error_path = outdir / "errors.jsonl"
    module_rows_all = []
    head_rows_all = []

    model = processor = None

    try:
        (
            model,
            processor,
            decoder_layers,
            decoder_path,
            spec,
        ) = load_model(a, two)

        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        bad = [
            L for L in trace_layers
            if not 0 <= L < n_layers
        ]
        if bad:
            raise ValueError(
                f"trace layers outside 0..{n_layers-1}: {bad}"
            )

        geom = infer_head_geometry(
            model,
            decoder_layers,
            trace_layers,
        )

        candidate_texts = get_candidate_texts(
            prior_metadata
        )
        candidate_ids = encode_candidate_ids(
            processor,
            candidate_texts,
        )

        reduction = str(
            prior_metadata.get(
                "sequence_score_reduction",
                "mean",
            )
        )
        if reduction not in ("mean", "sum"):
            reduction = "mean"

        print("=" * 190)
        print("REAL CAUSAL-TOKEN UPDATE -> ATTENTION / MLP / HEAD SOURCES")
        print("=" * 190)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"prior_run={run_dir}")
        print(f"N={len(meta)}")
        print(f"trace_layers={trace_layers}")
        print(f"sequence_score_reduction={reduction}")
        print(
            f"direction heads={sorted(direction_heads)}"
        )
        print(
            f"centroid heads={sorted(centroid_heads)}"
        )
        print()

        for m in tqdm(
            meta,
            desc="MODULE/HEAD SOURCE DECOMPOSITION",
        ):
            sid = int(m["sid"])

            if sid not in selected_by_sid:
                continue

            image = None

            try:
                causal_rows = selected_by_sid[sid]
                specs = causal_position_specs(causal_rows)
                positions = sorted(
                    set(int(s["real_position"]) for s in specs)
                )

                # Only layers that can lie on at least one selected token's
                # trajectory are needed.
                sample_layers = []
                for L in trace_layers:
                    include = False
                    for s in specs:
                        Cmax = int(s["max_target_layer"])
                        if a.exclude_target_layer:
                            ok = L < Cmax
                        else:
                            ok = L <= Cmax
                        if ok:
                            include = True
                            break
                    if include:
                        sample_layers.append(int(L))

                if not sample_layers:
                    continue

                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )

                # GT pass: gradients + all module/head activations.
                (
                    gt_score,
                    grad_gt,
                    module_caps,
                ) = run_answer_with_grad_and_optional_capture(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    answer_ids=candidate_ids[m["gt"]],
                    reduction=reduction,
                    layers=sample_layers,
                    positions=positions,
                    capture_modules=True,
                )

                # Competitor pass: only block-output gradients.
                (
                    comp_score,
                    grad_comp,
                    _,
                ) = run_answer_with_grad_and_optional_capture(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=batch,
                    answer_ids=candidate_ids[m["competitor"]],
                    reduction=reduction,
                    layers=sample_layers,
                    positions=positions,
                    capture_modules=False,
                )

                replay_margin = gt_score - comp_score
                if abs(
                    replay_margin
                    - float(m["prior_sequence_margin"])
                ) > 5e-3:
                    raise RuntimeError(
                        f"sequence margin mismatch: "
                        f"prior={m['prior_sequence_margin']:.6f} "
                        f"replay={replay_margin:.6f}"
                    )

                module_rows, head_rows = analyze_sample(
                    sid=sid,
                    gt=m["gt"],
                    baseline_prediction=m["baseline_prediction"],
                    baseline_correct=m["baseline_correct"],
                    competitor=m["competitor"],
                    prior_sequence_margin=m["prior_sequence_margin"],
                    causal_rows=causal_rows,
                    trace_layers=sample_layers,
                    exclude_target_layer=a.exclude_target_layer,
                    decoder_layers=decoder_layers,
                    geom=geom,
                    direction_heads=direction_heads,
                    centroid_heads=centroid_heads,
                    gt_score=gt_score,
                    comp_score=comp_score,
                    grad_gt=grad_gt,
                    grad_comp=grad_comp,
                    module_caps=module_caps,
                )

                module_rows_all.extend(module_rows)
                head_rows_all.extend(head_rows)

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": (
                            traceback.format_exc().splitlines()[-60:]
                        ),
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =================================================================
        # DataFrames / replay validation
        # =================================================================
        module_df = pd.DataFrame(module_rows_all)
        head_df = pd.DataFrame(head_rows_all)

        module_df, replay_summary = add_replay_validation(
            module_df,
            prior_map,
        )

        module_df.to_csv(
            outdir / "per_module_decision_contribution.csv",
            index=False,
        )
        head_df.to_csv(
            outdir / "per_head_decision_contribution.csv",
            index=False,
        )
        replay_summary.to_csv(
            outdir / "replay_validation.csv",
            index=False,
        )

        # =================================================================
        # Module summaries
        # =================================================================
        mod_layer = module_summary(
            module_df,
            ["update_layer"],
        )
        mod_rel_layer = module_summary(
            module_df,
            ["relation", "update_layer"],
        )
        mod_corr_layer = module_summary(
            module_df,
            ["baseline_correct", "update_layer"],
        )
        mod_rel_corr_layer = module_summary(
            module_df,
            ["relation", "baseline_correct", "update_layer"],
        )

        mod_layer.to_csv(
            outdir / "module_summary_by_layer.csv",
            index=False,
        )
        mod_rel_layer.to_csv(
            outdir / "module_summary_by_relation_layer.csv",
            index=False,
        )
        mod_corr_layer.to_csv(
            outdir / "module_summary_by_correctness_layer.csv",
            index=False,
        )
        mod_rel_corr_layer.to_csv(
            outdir / "module_summary_by_relation_correctness_layer.csv",
            index=False,
        )

        # =================================================================
        # Head summaries
        # =================================================================
        head_global = head_summary(
            head_df,
            [],
        )
        head_rel = head_summary(
            head_df,
            ["relation"],
        )
        head_layer = head_summary(
            head_df,
            ["update_layer"],
        )
        head_rel_layer = head_summary(
            head_df,
            ["relation", "update_layer"],
        )

        head_global.to_csv(
            outdir / "head_summary_global.csv",
            index=False,
        )
        head_rel.to_csv(
            outdir / "head_summary_by_relation.csv",
            index=False,
        )
        head_layer.to_csv(
            outdir / "head_summary_by_layer.csv",
            index=False,
        )
        head_rel_layer.to_csv(
            outdir / "head_summary_by_relation_layer.csv",
            index=False,
        )

        group_global = spatial_group_summary(
            head_df,
            [],
        )
        group_rel = spatial_group_summary(
            head_df,
            ["relation"],
        )
        group_layer = spatial_group_summary(
            head_df,
            ["update_layer"],
        )

        group_global.to_csv(
            outdir / "spatial_head_group_summary.csv",
            index=False,
        )
        group_rel.to_csv(
            outdir / "spatial_head_group_by_relation.csv",
            index=False,
        )
        group_layer.to_csv(
            outdir / "spatial_head_group_by_layer.csv",
            index=False,
        )

        top_pos, top_neg = top_head_tables(
            head_global,
            head_rel,
            a.top_heads,
            a.top_heads_per_relation,
        )
        top_pos.to_csv(
            outdir / "top_positive_heads.csv",
            index=False,
        )
        top_neg.to_csv(
            outdir / "top_negative_heads.csv",
            index=False,
        )

        enrich = enrichment_summary(head_df)
        enrich.to_csv(
            outdir / "spatial_head_enrichment.csv",
            index=False,
        )

        # =================================================================
        # Human-readable report
        # =================================================================
        replay_text = (
            replay_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
            if len(replay_summary)
            else "No prior per-update score file available."
        )

        module_text = (
            mod_layer.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(mod_layer)
            else "EMPTY"
        )

        group_text = (
            group_global.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
            if len(group_global)
            else "EMPTY"
        )

        pos_cols = [
            c for c in (
                "head_name",
                "head_group",
                "mean_head_decision_score",
                "positive_fraction",
                "negative_fraction",
                "mean_abs_score",
                "N_samples",
            )
            if c in top_pos.columns
        ]
        neg_cols = [
            c for c in (
                "head_name",
                "head_group",
                "mean_head_decision_score",
                "positive_fraction",
                "negative_fraction",
                "mean_abs_score",
                "N_samples",
            )
            if c in top_neg.columns
        ]

        pos_global_only = (
            top_pos[top_pos["rank_type"] == "global_positive"]
            if len(top_pos)
            else pd.DataFrame()
        )
        neg_global_only = (
            top_neg[top_neg["rank_type"] == "global_negative"]
            if len(top_neg)
            else pd.DataFrame()
        )

        report = [
            "=" * 190,
            "REAL CAUSAL-TOKEN UPDATE -> ATTENTION / MLP / HEAD SOURCES",
            "=" * 190,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested from prior run={len(meta)}",
            f"N successful module samples={module_df['sid'].nunique() if len(module_df) else 0}",
            f"N module-update rows={len(module_df)}",
            f"N head-message rows={len(head_df)}",
            f"trace layers={trace_layers}",
            "",
            "REPLAY VALIDATION AGAINST PRIOR REAL-UPDATE RUN",
            "-" * 190,
            replay_text,
            "",
            "ATTENTION vs MLP BY LAYER",
            "-" * 190,
            module_text,
            "",
            "SPATIAL / CENTROID HEAD GROUPS",
            "-" * 190,
            group_text,
            "",
            f"TOP {a.top_heads} GLOBAL POSITIVE HEADS",
            "-" * 190,
            (
                pos_global_only[pos_cols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}",
                )
                if len(pos_global_only)
                else "EMPTY"
            ),
            "",
            f"TOP {a.top_heads} GLOBAL NEGATIVE HEADS",
            "-" * 190,
            (
                neg_global_only[neg_cols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}",
                )
                if len(neg_global_only)
                else "EMPTY"
            ),
            "",
            "Definitions:",
            "  a_REAL = a_ATTN + a_MLP",
            "  B_REAL = <a_REAL, grad margin>",
            "  B_ATTN = <a_ATTN, grad margin>",
            "  B_MLP  = <a_MLP,  grad margin>",
            "  m_h    = W_O,h z_h",
            "  B_HEAD = <m_h, grad margin>",
            "",
            "How to interpret:",
            "  Positive REAL update: ask whether its positive mass comes mainly",
            "  from attention or MLP.",
            "  Negative REAL update: ask whether its negative mass comes mainly",
            "  from attention or MLP.",
            "  If attention dominates, inspect per-head scores and whether known",
            "  direction / centroid heads are enriched among top contributors.",
            "",
            "Important causal caveat:",
            "  This is an exact additive attribution of the REALIZED residual update.",
            "  It does not yet prove causal mediation by a module/head. Attention",
            "  also changes the MLP input in the same block. Follow-up ablation or",
            "  signed intervention is needed for causal source claims.",
        ]

        report_text = "\n".join(report) + "\n"
        print(report_text)
        (outdir / "analysis_summary.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(
            outdir / "metadata.json",
            {
                "script": "analyze_real_update_module_head_sources_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "prior_real_update_dir": str(run_dir),
                "N_prior_cohort": len(meta),
                "trace_layers": trace_layers,
                "exclude_target_layer": a.exclude_target_layer,
                "direction_heads": sorted(direction_heads),
                "centroid_heads": sorted(centroid_heads),
                "candidate_texts": candidate_texts,
                "sequence_score_reduction": reduction,
                "primary_definition": (
                    "a_REAL=attention_residual+mlp_residual; "
                    "B_component=dot(component, grad sequence margin)"
                ),
                "head_definition": (
                    "m_h = W_O[:,h] @ pre_o_head_h; "
                    "B_head=dot(m_h, grad sequence margin)"
                ),
                "note": (
                    "Old 440 CSVs are reused for cohort, causal states, "
                    "competitor and validation, but module/head activations "
                    "require new forward/backward passes."
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
