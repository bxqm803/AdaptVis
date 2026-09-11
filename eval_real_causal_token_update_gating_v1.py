#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_real_causal_token_update_gating_v1.py

Question
========
For a selected causal-token position p, what does EACH REAL decoder layer
actually add to that token, and is that real update helpful or harmful to the
correct decision?

Primary update definition
=========================
This script does NOT define the layer update with Real-NoImage.

For REAL only:

    a_L(p) = h_real[L,p] - h_real[L-1,p]

where h_real[L,p] is the decoder BLOCK OUTPUT at layer L.

Thus a_L is the actual residual-stream update received by causal token p from
decoder block L.

Decision-aware sign
===================
Use a four-way teacher-forced sequence margin:

    M = S_GT - S_competitor

where S_r is the sequence log-probability of answer r and competitor is the
strongest non-GT answer on the clean REAL prompt.

At each layer/token:

    B_L,p = < a_L(p), dM / d h_real[L,p] >

Interpretation:

    B > 0 : the actual REAL block update locally supports GT.
    B < 0 : the actual REAL block update locally supports the competing decision.

Generation interventions
========================
real_positive:
    if B > tau:
        h_real[L,p] <- h_real[L,p] + alpha * a_L(p)

real_negative_cancel:
    if B < -tau:
        h_real[L,p] <- h_real[L,p] - alpha * a_L(p)

real_signed:
    positive -> +alpha*a_L
    negative -> -alpha*a_L

real_all_amplify:
    h_real[L,p] <- h_real[L,p] + alpha*a_L
    regardless of sign

direct_rn:
    reference only. At selected final causal states:
        h_real[C,p] <- h_real[C,p] + beta*(h_real[C,p]-h_noimage[C,q])

The sign/gating of the main experiment uses ONLY the REAL update a_L and the
GT decision gradient. NoImage is not used to define the update or its sign.

Secondary decomposition
=======================
NoImage is retained only as a diagnostic. For aligned q:

    a_N,L = h_noimage[L,q] - h_noimage[L-1,q]
    a_RN,L = a_R,L - a_N,L

and therefore:

    B_REAL = B_NOIMAGE + B_RN

up to floating-point error.

This lets us ask AFTER finding a helpful/harmful REAL update whether its
decision effect is mainly shared/text-like or image-conditioned.

Oracle status
=============
This is an oracle MECHANISM experiment:
  1) causal-token positions come from the prior oracle causal ranking;
  2) GT defines the decision margin and therefore the positive/negative sign.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_real_causal_token_update_gating_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --update-layers 8-26 \
  --scales 0.1,0.25,0.5 \
  --decision-threshold 0 \
  --answer-surface above_below \
  --sequence-score-reduction mean \
  --eval-max-samples 40 \
  --output-dir output/qwen3b_real_causal_token_updates_n40_v1 \
  --overwrite

Optional local sign validation:
  --finite-probe-k 4 --finite-probe-scale 0.05

Outputs
=======
selected_causal_states.csv
alignment_summary.csv
sequence_score_summary.csv
per_real_update_decision_score.csv
sample_real_update_summary.csv
real_update_summary_by_generation_correctness.csv
layer_real_update_summary.csv
layer_real_update_by_generation_correctness.csv
finite_probe_validation.csv
generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
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
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
EPS = 1e-12


# =============================================================================
# CLI / utilities
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

    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument("--causal-categories", default="")

    p.add_argument(
        "--update-layers",
        default="8-26",
        help="REAL block updates a_L=hR_L-hR_{L-1}; every L must be >=1.",
    )
    p.add_argument(
        "--exclude-target-layer",
        action="store_true",
        help="For each causal token position, only use L before its latest selected causal layer.",
    )

    p.add_argument(
        "--scales",
        default="0.1,0.25,0.5",
        help="alpha values for actual-REAL-update interventions.",
    )
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=0.0,
        help="Gate only if |B_REAL| is greater than this raw score.",
    )
    p.add_argument(
        "--conditions",
        default="real_positive,real_negative_cancel,real_signed,real_all_amplify,direct_rn",
    )
    p.add_argument("--direct-beta", type=float, default=1.0)

    p.add_argument(
        "--answer-surface",
        default="above_below",
        choices=["above_below", "on_under"],
    )
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--answer-suffix", default="")
    p.add_argument(
        "--sequence-score-reduction",
        default="mean",
        choices=["mean", "sum"],
    )

    p.add_argument(
        "--finite-probe-k",
        type=int,
        default=0,
        help="Validate top-|B_REAL| updates per sample with +eps*a_L sequence-margin probes.",
    )
    p.add_argument("--finite-probe-scale", type=float, default=0.05)

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=40)

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


def parse_set(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


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


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Data / model
# =============================================================================

def load_data(a):
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
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
        meta.append(
            {
                "sid": sid,
                "gt": gt,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "question_text": str(p["question_text"]),
            }
        )

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    if a.eval_max_samples > 0:
        meta = traj.stratified_cap(meta, a.eval_max_samples, a.seed + 1)

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


def load_causal_selection(path, allowed_sids, causal_layers, top_k, categories):
    d = pd.read_csv(path)

    required = {
        "sid",
        "rank",
        "source_layer",
        "position",
        "token",
        "category",
        "broad_category",
    }
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")

    for c in ("sid", "rank", "source_layer", "position"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)

    d = d[d["sid"].isin(allowed_sids)].copy()
    d = d[d["source_layer"].isin(set(map(int, causal_layers)))].copy()
    d = d[d["broad_category"].astype(str) != "visual"].copy()
    d = d[d["broad_category"].astype(str) != "last"].copy()

    if categories:
        wanted = set(map(str, categories))
        d = d[d["broad_category"].astype(str).isin(wanted)].copy()

    rows = []
    for sid, g in d.groupby("sid"):
        z = g.sort_values("rank").head(int(top_k)).copy()
        z["causal_text_rank"] = np.arange(1, len(z) + 1)
        rows.append(z)

    return pd.concat(rows, ignore_index=True) if rows else d.iloc[:0].copy()


# =============================================================================
# NoImage / LCS -- secondary decomposition and direct-RN reference only
# =============================================================================

def move_batch(batch, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def build_noimage_batch(processor, question_text, device):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": str(question_text)}],
        }
    ]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = str(question_text)

    last_error = None
    for fn in [
        lambda: processor(text=[prompt], padding=True, return_tensors="pt"),
        lambda: processor(text=prompt, return_tensors="pt"),
    ]:
        try:
            return move_batch(fn(), device)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"NoImage processor failed: {last_error}")


def lcs_token_map(real_ids: List[int], no_ids: List[int]) -> Dict[int, int]:
    a = list(map(int, real_ids))
    b = list(map(int, no_ids))
    n, m = len(a), len(b)

    dp = np.zeros((n + 1, m + 1), dtype=np.uint16)

    for i in range(n - 1, -1, -1):
        ai = a[i]
        row = dp[i]
        below = dp[i + 1]
        for j in range(m - 1, -1, -1):
            if ai == b[j]:
                row[j] = 1 + below[j + 1]
            else:
                x = below[j]
                y = row[j + 1]
                row[j] = x if x >= y else y

    out = {}
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j] and dp[i, j] == 1 + dp[i + 1, j + 1]:
            out[i] = j
            i += 1
            j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1

    return out


# =============================================================================
# Block output capture / patch
# =============================================================================

def first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported layer output type: {type(output).__name__}")


def replace_first_tensor(output, tensor):
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise RuntimeError(f"Unsupported layer output type: {type(output).__name__}")


class CpuBlockCapture:
    def __init__(self, decoder_layers, layers):
        self.states = {}
        self.handles = []

        for L in sorted(set(map(int, layers))):
            def make_hook(layer):
                def hook(_m, _inp, out):
                    x = first_tensor(out)
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


@torch.inference_mode()
def capture_prompt_blocks(model, decoder_layers, batch, layers):
    cap = CpuBlockCapture(decoder_layers, layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        _ = model(**kw)

        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing block captures: {missing}")

        return dict(cap.states)
    finally:
        cap.close()


class GraphBlockCapture:
    """
    Capture block OUTPUTS and cut autograd at the earliest requested block output.
    """
    def __init__(self, decoder_layers, layers):
        layers = sorted(set(map(int, layers)))
        if not layers:
            raise ValueError("GraphBlockCapture requires non-empty layers")

        self.states = {}
        self.handles = []
        cut = min(layers)

        def cut_hook(_m, _inp, out):
            x = first_tensor(out)
            y = x.detach().clone().requires_grad_(True)
            return replace_first_tensor(out, y)

        self.handles.append(
            decoder_layers[cut].register_forward_hook(cut_hook)
        )

        for L in layers:
            def make_capture(layer):
                def hook(_m, _inp, out):
                    self.states[layer] = first_tensor(out)
                    return None
                return hook

            self.handles.append(
                decoder_layers[L].register_forward_hook(make_capture(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


class MultiLayerResidualAdd:
    """
    patch_map:
        layer -> {prompt_position -> vector}

    prompt_len is the full sequence length of the current prefill / teacher-forced
    forward. Decode-token steps are not patched.
    """
    def __init__(self, decoder_layers, patch_map, prompt_len):
        self.handles = []
        self.prompt_len = int(prompt_len)

        for L, pos_map in sorted(patch_map.items()):
            if not pos_map:
                continue

            def make_hook(local_pos_map):
                def hook(_m, _inp, out):
                    x = first_tensor(out)
                    if int(x.shape[1]) != self.prompt_len:
                        return None

                    y = x.clone()
                    for p, vec in local_pos_map.items():
                        p = int(p)
                        if 0 <= p < int(y.shape[1]):
                            y[0, p] += torch.as_tensor(
                                vec,
                                device=y.device,
                                dtype=y.dtype,
                            )
                    return replace_first_tensor(out, y)
                return hook

            self.handles.append(
                decoder_layers[int(L)].register_forward_hook(
                    make_hook(dict(pos_map))
                )
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


# =============================================================================
# Candidate sequence score
# =============================================================================

def tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def answer_words(surface):
    if surface == "on_under":
        return {
            "left": "left",
            "right": "right",
            "above": "on",
            "below": "under",
        }
    return {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
    }


def candidate_texts(a):
    words = answer_words(a.answer_surface)
    return {
        r: f"{a.answer_prefix}{words[r]}{a.answer_suffix}"
        for r in REL
    }


def encode_candidate_ids(processor, texts):
    tok = tokenizer_of(processor)
    result = {}

    print("\nCandidate answer tokenizations:")
    for r in REL:
        ids = tok.encode(texts[r], add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Empty candidate tokenization for {r}: {texts[r]!r}")
        result[r] = list(map(int, ids))
        print(
            f"  {r:>5s}: text={texts[r]!r} ids={result[r]} "
            f"decoded={tok.decode(result[r])!r}"
        )

    return result


def extend_batch_with_candidate(batch, answer_ids):
    out = {}
    T = int(batch["input_ids"].shape[1])
    dev = batch["input_ids"].device

    add = torch.tensor(
        [list(map(int, answer_ids))],
        dtype=batch["input_ids"].dtype,
        device=dev,
    )
    out["input_ids"] = torch.cat([batch["input_ids"], add], dim=1)

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
            tail = v[:, -1:].expand(v.shape[0], len(answer_ids))
            out[k] = torch.cat([v, tail], dim=1)
        else:
            out[k] = v

    return out, T


def sequence_score_from_logits(logits, prompt_len, answer_ids, reduction):
    terms = []
    for j, tid in enumerate(answer_ids):
        idx = prompt_len - 1 + j
        lp = torch.log_softmax(logits[0, idx].float(), dim=-1)[int(tid)]
        terms.append(lp)

    score = torch.stack(terms).sum()
    if reduction == "mean":
        score = score / max(len(terms), 1)
    return score


@torch.inference_mode()
def sequence_score(
    *,
    model,
    batch,
    answer_ids,
    reduction,
    decoder_layers=None,
    patch_map=None,
):
    ext, T = extend_batch_with_candidate(batch, answer_ids)
    full_len = T + len(answer_ids)

    ctx = (
        MultiLayerResidualAdd(
            decoder_layers=decoder_layers,
            patch_map=patch_map,
            prompt_len=full_len,
        )
        if patch_map
        else contextlib.nullcontext()
    )

    with ctx:
        kw = dict(ext)
        kw["use_cache"] = False
        kw["return_dict"] = True
        out = model(**kw)
        s = sequence_score_from_logits(
            out.logits,
            T,
            answer_ids,
            reduction,
        )

    return float(s.item())


def all_sequence_scores(model, batch, candidate_ids, reduction):
    return {
        r: sequence_score(
            model=model,
            batch=batch,
            answer_ids=candidate_ids[r],
            reduction=reduction,
        )
        for r in REL
    }


def sequence_score_and_grads(
    *,
    model,
    decoder_layers,
    batch,
    answer_ids,
    reduction,
    grad_layers,
):
    ext, T = extend_batch_with_candidate(batch, answer_ids)
    cap = GraphBlockCapture(decoder_layers, grad_layers)

    try:
        with torch.enable_grad():
            kw = dict(ext)
            kw["use_cache"] = False
            kw["return_dict"] = True
            out = model(**kw)

            missing = [L for L in grad_layers if L not in cap.states]
            if missing:
                raise RuntimeError(f"Gradient capture missing layers {missing}")

            score = sequence_score_from_logits(
                out.logits,
                T,
                answer_ids,
                reduction,
            )

            tensors = [cap.states[L] for L in grad_layers]
            grads = torch.autograd.grad(
                score,
                tensors,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            grad_by_layer = {}
            for L, g in zip(grad_layers, grads):
                grad_by_layer[L] = (
                    None
                    if g is None
                    else g.detach().float().cpu().numpy().astype(np.float32)
                )

            return float(score.detach().item()), grad_by_layer

    finally:
        cap.close()


# =============================================================================
# Causal-token trajectory + ACTUAL REAL block update
# =============================================================================

def causal_position_specs(causal_rows):
    """
    Unique causal token positions. A token can be selected at more than one
    target layer; scan its trajectory up to the latest selected causal layer.
    """
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
            specs[p]["max_target_layer"] = max(specs[p]["max_target_layer"], C)
            specs[p]["min_target_layer"] = min(specs[p]["min_target_layer"], C)

            if int(r.rank) < specs[p]["best_rank"]:
                specs[p]["best_rank"] = int(r.rank)
                specs[p]["token"] = str(r.token)
                specs[p]["category"] = str(r.category)
                specs[p]["broad_category"] = str(r.broad_category)

    return list(specs.values())


def build_real_updates(
    *,
    sid,
    gt,
    baseline_correct,
    specs,
    r2n,
    real_states,
    no_states,
    update_layers,
    exclude_target_layer,
):
    """
    PRIMARY:
        a_R,L = hR_L - hR_{L-1}

    SECONDARY only:
        a_N,L  = hN_L - hN_{L-1}
        a_RN,L = a_R,L - a_N,L
    """
    entries = []

    for s in specs:
        p = int(s["real_position"])
        q = r2n.get(p, None)
        Cmax = int(s["max_target_layer"])

        for L in update_layers:
            L = int(L)

            if L < 1:
                continue

            if exclude_target_layer:
                if L >= Cmax:
                    continue
            else:
                if L > Cmax:
                    continue

            if L not in real_states or L - 1 not in real_states:
                continue

            if not (
                0 <= p < real_states[L].shape[1]
                and 0 <= p < real_states[L - 1].shape[1]
            ):
                continue

            a_real = (
                real_states[L][0, p].astype(np.float32)
                - real_states[L - 1][0, p].astype(np.float32)
            )

            # Secondary NoImage decomposition if this position is alignable.
            a_no = None
            a_rn = None
            if (
                q is not None
                and L in no_states
                and L - 1 in no_states
                and 0 <= q < no_states[L].shape[1]
                and 0 <= q < no_states[L - 1].shape[1]
            ):
                a_no = (
                    no_states[L][0, q].astype(np.float32)
                    - no_states[L - 1][0, q].astype(np.float32)
                )
                a_rn = (a_real - a_no).astype(np.float32)

            entries.append(
                {
                    "sid": int(sid),
                    "gt": str(gt),
                    "baseline_correct": bool(baseline_correct),
                    "update_layer": L,
                    "real_position": p,
                    "noimage_position": int(q) if q is not None else -1,
                    "token": str(s["token"]),
                    "category": str(s["category"]),
                    "broad_category": str(s["broad_category"]),
                    "max_target_layer": Cmax,
                    "distance_to_latest_target": Cmax - L,
                    "real_update_norm": float(np.linalg.norm(a_real)),
                    "noimage_update_norm": (
                        float(np.linalg.norm(a_no))
                        if a_no is not None
                        else float("nan")
                    ),
                    "rn_update_norm": (
                        float(np.linalg.norm(a_rn))
                        if a_rn is not None
                        else float("nan")
                    ),
                    "rn_update_fraction_of_real": (
                        float(np.linalg.norm(a_rn)) / max(float(np.linalg.norm(a_real)), EPS)
                        if a_rn is not None
                        else float("nan")
                    ),
                    "real_rn_update_cosine": (
                        cosine_np(a_real, a_rn)
                        if a_rn is not None
                        else float("nan")
                    ),
                    "_real_update": a_real,
                    "_noimage_update": a_no,
                    "_rn_update": a_rn,
                }
            )

    return entries


def attach_decision_scores(entries, grad_gt, grad_comp):
    scored = []

    for e in entries:
        L = int(e["update_layer"])
        p = int(e["real_position"])

        gg = grad_gt.get(L, None)
        gc = grad_comp.get(L, None)

        if gg is None or gc is None:
            continue
        if not (0 <= p < gg.shape[1] and 0 <= p < gc.shape[1]):
            continue

        grad = (gg[0, p] - gc[0, p]).astype(np.float32)
        a_real = np.asarray(e["_real_update"], np.float32)

        B_real = float(np.dot(a_real, grad))
        grad_norm = float(np.linalg.norm(grad))
        real_norm = float(np.linalg.norm(a_real))

        a_no = e["_noimage_update"]
        a_rn = e["_rn_update"]

        if a_no is not None and a_rn is not None:
            B_no = float(np.dot(np.asarray(a_no, np.float32), grad))
            B_rn = float(np.dot(np.asarray(a_rn, np.float32), grad))
            decomp_error = abs(B_real - (B_no + B_rn))
        else:
            B_no = float("nan")
            B_rn = float("nan")
            decomp_error = float("nan")

        z = dict(e)
        z.update(
            {
                "decision_grad_norm": grad_norm,
                "real_update_decision_score": B_real,
                "real_update_decision_score_per_update_norm": B_real / max(real_norm, EPS),
                "real_update_decision_score_per_grad_norm": B_real / max(grad_norm, EPS),
                "real_update_decision_cosine": cosine_np(a_real, grad),

                # Secondary decomposition only.
                "noimage_update_decision_score": B_no,
                "rn_update_decision_score": B_rn,
                "decision_score_decomposition_error": decomp_error,
                "_decision_grad": grad,
            }
        )
        scored.append(z)

    return scored


# =============================================================================
# Patch maps
# =============================================================================

def add_patch(patch_map, layer, position, vec):
    L = int(layer)
    p = int(position)
    patch_map.setdefault(L, {})
    v = np.asarray(vec, np.float32)

    if p in patch_map[L]:
        patch_map[L][p] = (patch_map[L][p] + v).astype(np.float32)
    else:
        patch_map[L][p] = v.copy()


def build_real_update_patch_map(entries, condition, scale, threshold):
    patch_map = {}
    counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "patched": 0,
    }

    for e in entries:
        s = float(e["real_update_decision_score"])
        u = np.asarray(e["_real_update"], np.float32)
        vec = None

        if s > threshold:
            counts["positive"] += 1
        elif s < -threshold:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

        if condition == "real_positive":
            if s > threshold:
                vec = float(scale) * u

        elif condition == "real_negative_cancel":
            if s < -threshold:
                vec = -float(scale) * u

        elif condition == "real_signed":
            if s > threshold:
                vec = float(scale) * u
            elif s < -threshold:
                vec = -float(scale) * u

        elif condition == "real_all_amplify":
            vec = float(scale) * u

        else:
            raise ValueError(condition)

        if vec is not None:
            add_patch(
                patch_map,
                e["update_layer"],
                e["real_position"],
                vec,
            )
            counts["patched"] += 1

    return patch_map, counts


def build_direct_rn_patch_map(causal_rows, r2n, real_states, no_states, beta):
    patch_map = {}
    seen = set()

    for r in causal_rows.itertuples():
        C = int(r.source_layer)
        p = int(r.position)
        q = r2n.get(p, None)
        key = (C, p)

        if q is None or key in seen:
            continue
        if C not in real_states or C not in no_states:
            continue
        if not (
            0 <= p < real_states[C].shape[1]
            and 0 <= q < no_states[C].shape[1]
        ):
            continue

        t = (
            real_states[C][0, p].astype(np.float32)
            - no_states[C][0, q].astype(np.float32)
        )

        add_patch(patch_map, C, p, float(beta) * t)
        seen.add(key)

    return patch_map


@torch.inference_mode()
def generate_with_patch(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    patch_map,
    max_new_tokens,
):
    prompt_len = int(batch["input_ids"].shape[1])

    ctx = (
        MultiLayerResidualAdd(
            decoder_layers=decoder_layers,
            patch_map=patch_map,
            prompt_len=prompt_len,
        )
        if patch_map
        else contextlib.nullcontext()
    )

    with ctx:
        text = base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )

    pred = traj.normalize_relation(base, text)
    return pred, text


# =============================================================================
# Finite local validation
# =============================================================================

def sequence_margin_with_patch(
    *,
    model,
    decoder_layers,
    batch,
    candidate_ids,
    reduction,
    gt,
    competitor,
    patch_map,
):
    s_gt = sequence_score(
        model=model,
        batch=batch,
        answer_ids=candidate_ids[gt],
        reduction=reduction,
        decoder_layers=decoder_layers,
        patch_map=patch_map,
    )
    s_comp = sequence_score(
        model=model,
        batch=batch,
        answer_ids=candidate_ids[competitor],
        reduction=reduction,
        decoder_layers=decoder_layers,
        patch_map=patch_map,
    )
    return float(s_gt - s_comp)


# =============================================================================
# Summaries
# =============================================================================

def summarize_samples(sample_df):
    if len(sample_df) == 0:
        return pd.DataFrame()

    rows = []

    for correct, g in sample_df.groupby("baseline_correct", dropna=False):
        rows.append(
            {
                "baseline_correct": bool(correct),
                "N": len(g),
                "mean_positive_fraction": safe_mean(g["positive_fraction"]),
                "mean_negative_fraction": safe_mean(g["negative_fraction"]),
                "mean_positive_mass": safe_mean(g["positive_mass"]),
                "mean_negative_mass": safe_mean(g["negative_mass"]),
                "mean_net_real_update_decision_mass": safe_mean(
                    g["net_real_update_decision_mass"]
                ),
                "mean_decision_efficiency": safe_mean(g["decision_efficiency"]),
                "median_decision_efficiency": safe_median(g["decision_efficiency"]),
                "mean_sequence_margin": safe_mean(g["sequence_margin"]),
                "sequence_margin_correct_fraction": float(
                    np.mean(g["sequence_margin_correct"].astype(bool))
                ),
                "mean_rn_share_of_positive_mass": safe_mean(
                    g["rn_share_of_positive_mass"]
                ),
                "mean_rn_share_of_negative_mass": safe_mean(
                    g["rn_share_of_negative_mass"]
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_layers(update_df, extra_keys=None):
    if len(update_df) == 0:
        return pd.DataFrame()

    keys = list(extra_keys or []) + ["update_layer"]
    rows = []

    for key, g in update_df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {k: v for k, v in zip(keys, key)}

        s = g["real_update_decision_score"].astype(float).to_numpy()

        row.update(
            {
                "N_samples": g["sid"].nunique(),
                "N_updates": len(g),
                "mean_real_update_decision_score": float(np.mean(s)),
                "median_real_update_decision_score": float(np.median(s)),
                "positive_fraction": float(np.mean(s > 0)),
                "negative_fraction": float(np.mean(s < 0)),
                "mean_positive_score": safe_mean(x for x in s if x > 0),
                "mean_negative_score": safe_mean(x for x in s if x < 0),
                "mean_abs_score": float(np.mean(np.abs(s))),
                "mean_real_update_decision_cosine": safe_mean(
                    g["real_update_decision_cosine"]
                ),
                "mean_real_update_norm": safe_mean(g["real_update_norm"]),
                "mean_grad_norm": safe_mean(g["decision_grad_norm"]),

                # Secondary source diagnostics.
                "mean_noimage_update_decision_score": safe_mean(
                    g["noimage_update_decision_score"]
                ),
                "mean_rn_update_decision_score": safe_mean(
                    g["rn_update_decision_score"]
                ),
                "mean_rn_update_fraction_of_real": safe_mean(
                    g["rn_update_fraction_of_real"]
                ),
                "mean_real_rn_update_cosine": safe_mean(
                    g["real_rn_update_cosine"]
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def summarize_generation(gen_df):
    if len(gen_df) == 0:
        return pd.DataFrame()

    b = (
        gen_df[gen_df["condition"] == "baseline"]
        .drop_duplicates("sid")
        .set_index("sid")
    )

    rows = []

    for (cond, scale), g in gen_df[gen_df["condition"] != "baseline"].groupby(
        ["condition", "scale"],
        dropna=False,
    ):
        x = g.set_index("sid")
        common = sorted(set(b.index) & set(x.index))
        if not common:
            continue

        bc = b.loc[common, "correct"].astype(bool).to_numpy()
        pc = x.loc[common, "correct"].astype(bool).to_numpy()
        bp = b.loc[common, "prediction"].astype(str).to_numpy()
        pp = x.loc[common, "prediction"].astype(str).to_numpy()

        w2c = int(np.sum((~bc) & pc))
        c2w = int(np.sum(bc & (~pc)))

        rows.append(
            {
                "condition": str(cond),
                "scale": float(scale),
                "N": len(common),
                "baseline_accuracy": float(np.mean(bc)),
                "patched_accuracy": float(np.mean(pc)),
                "gain": float(np.mean(pc) - np.mean(bc)),
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "net": w2c - c2w,
                "changed": int(np.sum(bp != pp)),
                "repair_rate_on_wrong": w2c / max(int((~bc).sum()), 1),
                "preserve_rate_on_correct": 1 - c2w / max(int(bc.sum()), 1),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["patched_accuracy", "condition", "scale"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def summarize_generation_by_relation(gen_df):
    if len(gen_df) == 0:
        return pd.DataFrame()

    rows = []

    for (cond, scale, gt), g in gen_df.groupby(
        ["condition", "scale", "gt"],
        dropna=False,
    ):
        rows.append(
            {
                "condition": str(cond),
                "scale": float(scale),
                "relation": str(gt),
                "N": len(g),
                "accuracy": float(g["correct"].astype(bool).mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["condition", "scale", "relation"])
        .reset_index(drop=True)
    )


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers = parse_layers(a.causal_layers)
    update_layers = parse_layers(a.update_layers)
    scales = parse_floats(a.scales)
    categories = parse_set(a.causal_categories)
    conditions = parse_set(a.conditions)

    valid_conditions = {
        "real_positive",
        "real_negative_cancel",
        "real_signed",
        "real_all_amplify",
        "direct_rn",
    }
    unknown = set(conditions) - valid_conditions
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    if any(L < 1 for L in update_layers):
        raise ValueError("--update-layers must be >=1 because a_L uses L-1")

    outdir = Path(a.output_dir)

    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")

    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    two, meta, rec_by_sid = load_data(a)
    eval_sids = {int(m["sid"]) for m in meta}

    causal_sel = load_causal_selection(
        Path(a.ranked_causal),
        eval_sids,
        causal_layers,
        a.causal_top_k,
        categories,
    )
    causal_sel.to_csv(outdir / "selected_causal_states.csv", index=False)

    causal_by_sid = {
        int(sid): g.sort_values("rank").copy()
        for sid, g in causal_sel.groupby("sid")
    }

    model = processor = None

    alignment_rows = []
    sequence_rows = []
    update_rows_all = []
    sample_rows = []
    finite_rows = []
    generation_rows = []

    try:
        model, processor, decoder_layers, decoder_path, spec = load_model(a, two)
        device = torch.device(a.device)
        n_layers = len(decoder_layers)

        capture_layers = sorted(
            set(
                causal_layers
                + update_layers
                + [L - 1 for L in update_layers]
            )
        )

        bad = [L for L in capture_layers if not 0 <= L < n_layers]
        if bad:
            raise ValueError(
                f"Requested/capture layers outside 0..{n_layers-1}: {bad}"
            )

        texts = candidate_texts(a)
        candidate_ids = encode_candidate_ids(processor, texts)

        print("=" * 202)
        print("ACTUAL REAL CAUSAL-TOKEN LAYER UPDATE -> DECISION")
        print("=" * 202)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"N={len(meta)}")
        print(f"causal_layers={causal_layers} topK={a.causal_top_k}")
        print(f"update_layers={update_layers}")
        print(f"exclude_target_layer={a.exclude_target_layer}")
        print(f"scales={scales}")
        print(f"decision_threshold={a.decision_threshold}")
        print(f"conditions={conditions}")
        print(
            f"sequence answers={texts} reduction={a.sequence_score_reduction}"
        )
        print()

        for m in tqdm(meta, desc="REAL CAUSAL-TOKEN UPDATES"):
            sid = int(m["sid"])
            if sid not in causal_by_sid:
                continue

            image = None

            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                rb = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m["question_text"],
                    device=device,
                )

                # NoImage is not part of the primary update/sign. It is used only
                # for secondary decomposition and direct-RN control.
                nb = build_noimage_batch(
                    processor,
                    m["question_text"],
                    device,
                )

                rid = rb["input_ids"][0].detach().cpu().tolist()
                nid = nb["input_ids"][0].detach().cpu().tolist()
                r2n = lcs_token_map(rid, nid)

                real_states = capture_prompt_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )
                no_states = capture_prompt_blocks(
                    model,
                    decoder_layers,
                    nb,
                    capture_layers,
                )

                # ---------------------------------------------------------
                # Baseline free generation
                # ---------------------------------------------------------
                base_text = base.generate_text(
                    model,
                    processor,
                    rb,
                    max_new_tokens=a.max_new_tokens,
                )
                base_pred = traj.normalize_relation(base, base_text)
                base_correct = base_pred == m["gt"]

                generation_rows.append(
                    {
                        "sid": sid,
                        "gt": m["gt"],
                        "condition": "baseline",
                        "scale": 0.0,
                        "prediction": base_pred,
                        "correct": base_correct,
                        "n_patched_updates": 0,
                        "n_positive_updates": 0,
                        "n_negative_updates": 0,
                        "text": base_text,
                    }
                )

                # ---------------------------------------------------------
                # Sequence-level decision proxy
                # ---------------------------------------------------------
                clean_scores = all_sequence_scores(
                    model,
                    rb,
                    candidate_ids,
                    a.sequence_score_reduction,
                )
                seq_pred = max(REL, key=lambda r: clean_scores[r])
                competitor = max(
                    (r for r in REL if r != m["gt"]),
                    key=lambda r: clean_scores[r],
                )
                seq_margin = float(clean_scores[m["gt"]] - clean_scores[competitor])
                seq_correct = seq_pred == m["gt"]

                sequence_rows.append(
                    {
                        "sid": sid,
                        "gt": m["gt"],
                        "generation_prediction": base_pred,
                        "generation_correct": base_correct,
                        "sequence_prediction": seq_pred,
                        "sequence_correct": seq_correct,
                        "sequence_generation_agree": seq_pred == base_pred,
                        "competitor": competitor,
                        "sequence_margin": seq_margin,
                        **{f"score_{r}": clean_scores[r] for r in REL},
                    }
                )

                # ---------------------------------------------------------
                # ACTUAL REAL updates at selected causal-token trajectories
                # ---------------------------------------------------------
                specs = causal_position_specs(causal_by_sid[sid])

                entries = build_real_updates(
                    sid=sid,
                    gt=m["gt"],
                    baseline_correct=base_correct,
                    specs=specs,
                    r2n=r2n,
                    real_states=real_states,
                    no_states=no_states,
                    update_layers=update_layers,
                    exclude_target_layer=a.exclude_target_layer,
                )

                if not entries:
                    raise RuntimeError("No REAL causal-token layer updates")

                grad_layers = sorted(
                    set(int(e["update_layer"]) for e in entries)
                )

                gt_score_graph, grad_gt = sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    answer_ids=candidate_ids[m["gt"]],
                    reduction=a.sequence_score_reduction,
                    grad_layers=grad_layers,
                )

                comp_score_graph, grad_comp = sequence_score_and_grads(
                    model=model,
                    decoder_layers=decoder_layers,
                    batch=rb,
                    answer_ids=candidate_ids[competitor],
                    reduction=a.sequence_score_reduction,
                    grad_layers=grad_layers,
                )

                graph_margin = gt_score_graph - comp_score_graph

                if abs(graph_margin - seq_margin) > 5e-3:
                    raise RuntimeError(
                        f"Sequence margin replay mismatch: "
                        f"clean={seq_margin:.6f} graph={graph_margin:.6f}"
                    )

                scored = attach_decision_scores(
                    entries,
                    grad_gt,
                    grad_comp,
                )

                if not scored:
                    raise RuntimeError("No REAL update received a decision gradient")

                B = np.asarray(
                    [float(e["real_update_decision_score"]) for e in scored],
                    dtype=np.float64,
                )

                pos_mass = float(np.maximum(B, 0).sum())
                neg_mass = float(np.maximum(-B, 0).sum())
                abs_mass = pos_mass + neg_mass
                net_mass = float(B.sum())

                # Secondary decomposition: what fraction of positive/negative
                # REAL decision mass is carried by the RN differential component?
                pos_rn = 0.0
                neg_rn = 0.0
                for e in scored:
                    br = float(e["real_update_decision_score"])
                    brn = float(e["rn_update_decision_score"])
                    if not math.isfinite(brn):
                        continue
                    if br > 0:
                        pos_rn += brn
                    elif br < 0:
                        # Report same signed-axis contribution magnitude relative
                        # to harmful REAL mass; positive brn here means RN opposes
                        # the harmful real update.
                        neg_rn += -brn

                sample_rows.append(
                    {
                        "sid": sid,
                        "gt": m["gt"],
                        "baseline_prediction": base_pred,
                        "baseline_correct": base_correct,
                        "sequence_prediction": seq_pred,
                        "sequence_margin": seq_margin,
                        "sequence_margin_correct": seq_correct,
                        "sequence_generation_agree": seq_pred == base_pred,
                        "competitor": competitor,
                        "N_updates": len(scored),
                        "positive_fraction": float(np.mean(B > 0)),
                        "negative_fraction": float(np.mean(B < 0)),
                        "positive_mass": pos_mass,
                        "negative_mass": neg_mass,
                        "net_real_update_decision_mass": net_mass,
                        "decision_efficiency": (
                            net_mass / abs_mass if abs_mass > EPS else float("nan")
                        ),
                        "mean_abs_real_update_score": float(np.mean(np.abs(B))),
                        "rn_share_of_positive_mass": (
                            pos_rn / pos_mass if pos_mass > EPS else float("nan")
                        ),
                        "rn_share_of_negative_mass": (
                            neg_rn / neg_mass if neg_mass > EPS else float("nan")
                        ),
                    }
                )

                alignment_rows.append(
                    {
                        "sid": sid,
                        "gt": m["gt"],
                        "real_seq_len": len(rid),
                        "noimage_seq_len": len(nid),
                        "lcs_matches": len(r2n),
                        "selected_causal_states": len(causal_by_sid[sid]),
                        "unique_causal_positions": len(specs),
                        "decision_scored_real_updates": len(scored),
                    }
                )

                # ---------------------------------------------------------
                # Optional finite local sign validation
                # ---------------------------------------------------------
                if a.finite_probe_k > 0:
                    chosen = sorted(
                        scored,
                        key=lambda e: abs(float(e["real_update_decision_score"])),
                        reverse=True,
                    )[: int(a.finite_probe_k)]

                    for e in chosen:
                        pmap = {}
                        add_patch(
                            pmap,
                            e["update_layer"],
                            e["real_position"],
                            float(a.finite_probe_scale)
                            * np.asarray(e["_real_update"], np.float32),
                        )

                        patched_margin = sequence_margin_with_patch(
                            model=model,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            candidate_ids=candidate_ids,
                            reduction=a.sequence_score_reduction,
                            gt=m["gt"],
                            competitor=competitor,
                            patch_map=pmap,
                        )

                        actual_gain = patched_margin - seq_margin
                        predicted_gain = (
                            float(a.finite_probe_scale)
                            * float(e["real_update_decision_score"])
                        )

                        finite_rows.append(
                            {
                                "sid": sid,
                                "gt": m["gt"],
                                "baseline_correct": base_correct,
                                "update_layer": int(e["update_layer"]),
                                "real_position": int(e["real_position"]),
                                "token": str(e["token"]),
                                "real_update_decision_score": float(
                                    e["real_update_decision_score"]
                                ),
                                "probe_scale": float(a.finite_probe_scale),
                                "predicted_margin_gain": predicted_gain,
                                "actual_margin_gain": actual_gain,
                                "sign_agreement": (
                                    np.sign(predicted_gain) == np.sign(actual_gain)
                                ),
                                "gain_ratio": (
                                    actual_gain / predicted_gain
                                    if abs(predicted_gain) > 1e-8
                                    else float("nan")
                                ),
                            }
                        )

                # Save score rows without arrays.
                for e in scored:
                    update_rows_all.append(
                        {
                            k: v
                            for k, v in e.items()
                            if k
                            not in (
                                "_real_update",
                                "_noimage_update",
                                "_rn_update",
                                "_decision_grad",
                            )
                        }
                    )

                # ---------------------------------------------------------
                # Actual generation under REAL-update gating
                # ---------------------------------------------------------
                for cond in conditions:
                    if cond == "direct_rn":
                        pmap = build_direct_rn_patch_map(
                            causal_by_sid[sid],
                            r2n,
                            real_states,
                            no_states,
                            a.direct_beta,
                        )

                        pred, text = generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=pmap,
                            max_new_tokens=a.max_new_tokens,
                        )

                        generation_rows.append(
                            {
                                "sid": sid,
                                "gt": m["gt"],
                                "condition": "direct_rn",
                                "scale": float(a.direct_beta),
                                "prediction": pred,
                                "correct": pred == m["gt"],
                                "n_patched_updates": sum(
                                    len(v) for v in pmap.values()
                                ),
                                "n_positive_updates": np.nan,
                                "n_negative_updates": np.nan,
                                "text": text,
                            }
                        )
                        continue

                    for scale in scales:
                        pmap, counts = build_real_update_patch_map(
                            scored,
                            cond,
                            scale,
                            a.decision_threshold,
                        )

                        pred, text = generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=pmap,
                            max_new_tokens=a.max_new_tokens,
                        )

                        generation_rows.append(
                            {
                                "sid": sid,
                                "gt": m["gt"],
                                "condition": cond,
                                "scale": float(scale),
                                "prediction": pred,
                                "correct": pred == m["gt"],
                                "n_patched_updates": counts["patched"],
                                "n_positive_updates": counts["positive"],
                                "n_negative_updates": counts["negative"],
                                "text": text,
                            }
                        )

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "phase": "sample",
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-60:],
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}"
                )

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =================================================================
        # Save raw tables
        # =================================================================
        align_df = pd.DataFrame(alignment_rows)
        seq_df = pd.DataFrame(sequence_rows)
        upd_df = pd.DataFrame(update_rows_all)
        sample_df = pd.DataFrame(sample_rows)
        finite_df = pd.DataFrame(finite_rows)
        gen_df = pd.DataFrame(generation_rows)

        align_df.to_csv(outdir / "alignment_summary.csv", index=False)
        seq_df.to_csv(outdir / "sequence_score_summary.csv", index=False)
        upd_df.to_csv(outdir / "per_real_update_decision_score.csv", index=False)
        sample_df.to_csv(outdir / "sample_real_update_summary.csv", index=False)
        finite_df.to_csv(outdir / "finite_probe_validation.csv", index=False)
        gen_df.to_csv(outdir / "generation_per_sample.csv", index=False)

        # =================================================================
        # Summaries
        # =================================================================
        by_correct = summarize_samples(sample_df)
        by_correct.to_csv(
            outdir / "real_update_summary_by_generation_correctness.csv",
            index=False,
        )

        layer_summary = summarize_layers(upd_df)
        layer_summary.to_csv(
            outdir / "layer_real_update_summary.csv",
            index=False,
        )

        layer_by_correct = summarize_layers(
            upd_df,
            extra_keys=["baseline_correct"],
        )
        layer_by_correct.to_csv(
            outdir / "layer_real_update_by_generation_correctness.csv",
            index=False,
        )

        gen_summary = summarize_generation(gen_df)
        gen_summary.to_csv(
            outdir / "generation_summary.csv",
            index=False,
        )

        gen_rel = summarize_generation_by_relation(gen_df)
        gen_rel.to_csv(
            outdir / "generation_by_relation.csv",
            index=False,
        )

        # Proxy checks
        if len(seq_df):
            gen_acc = float(seq_df["generation_correct"].astype(bool).mean())
            seq_acc = float(seq_df["sequence_correct"].astype(bool).mean())
            seq_gen_agree = float(
                seq_df["sequence_generation_agree"].astype(bool).mean()
            )
        else:
            gen_acc = seq_acc = seq_gen_agree = float("nan")

        if len(finite_df):
            finite_sign_agree = float(
                finite_df["sign_agreement"].astype(bool).mean()
            )
        else:
            finite_sign_agree = float("nan")

        report = [
            "=" * 202,
            "ACTUAL REAL CAUSAL-TOKEN LAYER UPDATE -> DECISION",
            "=" * 202,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested={len(meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"update layers={update_layers}",
            f"exclude target layer={a.exclude_target_layer}",
            f"scales={scales}",
            f"decision threshold={a.decision_threshold}",
            "",
            "SEQUENCE-SCORE PROXY CHECK",
            "-" * 202,
            f"generation accuracy              : {gen_acc:.4f}",
            f"sequence-score accuracy          : {seq_acc:.4f}",
            f"sequence vs generation agreement : {seq_gen_agree:.4f}",
            f"finite-probe sign agreement      : {finite_sign_agree:.4f}",
            "",
            "ACTUAL REAL UPDATE MASS BY BASELINE GENERATION CORRECTNESS",
            "-" * 202,
            by_correct.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(by_correct)
            else "EMPTY",
            "",
            "LAYERWISE ACTUAL REAL UPDATE SCORE",
            "-" * 202,
            layer_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(layer_summary)
            else "EMPTY",
            "",
            "GENERATION",
            "-" * 202,
            gen_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
            if len(gen_summary)
            else "EMPTY",
            "",
            "PRIMARY definition:",
            "  a_REAL[L,p] = h_REAL[L,p] - h_REAL[L-1,p]",
            "  M_seq       = score(GT) - score(best non-GT competitor)",
            "  B_REAL[L,p] = <a_REAL[L,p], dM_seq/dh_REAL[L,p]>",
            "",
            "B_REAL is about what the REAL decoder block actually did to the causal token.",
            "It is NOT defined with Real-NoImage.",
            "",
            "Secondary decomposition only:",
            "  a_NOIMAGE = h_N[L,q] - h_N[L-1,q]",
            "  a_RN      = a_REAL - a_NOIMAGE",
            "  B_REAL ~= B_NOIMAGE + B_RN",
            "",
            "Evidence for harmful layer computation would be:",
            "  1) wrong samples show larger negative REAL-update mass/fraction;",
            "  2) real_negative_cancel repairs actual generation;",
            "  3) real_signed > real_all_amplify at matched scale.",
            "",
            "Evidence for insufficient positive computation would be:",
            "  wrong samples have weaker positive REAL-update mass, while cancelling",
            "  negative REAL updates gives little benefit.",
            "",
            "NoImage diagnostics can then tell whether helpful/harmful REAL updates are",
            "predominantly image-conditioned or also present in the text-only trajectory.",
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
                "script": "eval_real_causal_token_update_gating_v1.py",
                "model": a.model,
                "repo_id": spec.repo_id,
                "decoder_path": decoder_path,
                "eval_requested_N": len(meta),
                "causal_layers": causal_layers,
                "causal_top_k": a.causal_top_k,
                "update_layers": update_layers,
                "exclude_target_layer": a.exclude_target_layer,
                "scales": scales,
                "decision_threshold": a.decision_threshold,
                "conditions": conditions,
                "direct_beta": a.direct_beta,
                "answer_surface": a.answer_surface,
                "answer_prefix": a.answer_prefix,
                "answer_suffix": a.answer_suffix,
                "candidate_texts": texts,
                "sequence_score_reduction": a.sequence_score_reduction,
                "finite_probe_k": a.finite_probe_k,
                "finite_probe_scale": a.finite_probe_scale,
                "primary_update": "a_REAL[L,p] = h_REAL[L,p] - h_REAL[L-1,p]",
                "decision_score": "dot(a_REAL[L,p], grad_h [S_GT-S_competitor])",
                "secondary_noimage_decomposition": (
                    "a_RN = a_REAL-a_NOIMAGE; not used for primary sign/gating"
                ),
                "oracle_note": (
                    "Causal positions originate from prior oracle ranking; GT defines "
                    "the sequence margin and therefore the update sign."
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
