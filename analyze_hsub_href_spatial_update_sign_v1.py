#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_hsub_href_spatial_update_sign_v1.py

Goal
====
Test whether the FINAL residual-stream object relation state at each decoder
layer,

    r_L = h_L(subject) - h_L(reference),

and, especially, its layer-to-layer update,

    Δr_L = r_L - r_{L-1},

can explain whether the already-measured REAL causal-token updates at that
layer are behaviorally helpful (+) or harmful (-).

NO generation is performed.

The four spatial directions are learned ONLY from Synthetic-400 using the same
source-only codebook rule as the existing Direction-Head selector:

    center_L = mean_i r_{i,L}

    d_{L,k} = normalize(
        mean_{i:y_i=k} r_{i,L} - center_L
    )

    score_L(k | r) =
        cosine(r - center_L, d_{L,k})

with k in {left, right, above, below}.

This script intentionally distinguishes THREE questions:

1) STATE: what spatial relation does the final h_sub-h_ref state at layer L
   look like?

       state_pred_L = argmax_k score_L(k | r_L)

2) UPDATE VECTOR: what relation direction does Δr_L itself resemble?

       delta_score_L(k) = cosine(Δr_L, d_{L,k})
       delta_pred_L = argmax_k delta_score_L(k)

3) EFFECT OF THE UPDATE IN ONE FIXED SPATIAL BASIS:
   did the block transition from r_{L-1} to r_L improve the GT spatial margin,
   when BOTH before and after are evaluated with the SAME layer-L codebook?

       m_L(r) =
           score_L(GT | r) - max_{k != GT} score_L(k | r)

       spatial_margin_shift_L =
           m_L(r_L) - m_L(r_{L-1})

This third quantity is the cleanest "spatial-positive / spatial-negative"
diagnostic in this script because it does not compare margins measured under
two different layer-specific codebooks.

We report both:
    IMG:
        r_L^R = h_sub^R - h_ref^R

and the image-conditioned residual used in the spirit of the old Direction
probe:
    RESIDUAL:
        r_L^RN =
          (h_sub^R - h_ref^R)
          - (h_sub^NoImage - h_ref^NoImage)

NO_IMAGE is also kept as a control.

Important:
==========
The behavioral sign being predicted/evaluated is NOT defined by this script.
It is read from the existing oracle diagnostic:

    <prior-real-update-dir>/per_real_update_decision_score.csv

where:
    B_REAL > 0  = actual causal-token update locally supports GT
    B_REAL < 0  = actual causal-token update locally supports competitor.

Thus this script tests whether residual-stream spatial-state dynamics are
related to the already-established behavioral sign.

A layer-level h_sub-h_ref update gives ONE spatial sign per sample x layer,
whereas the prior causal run can contain MANY causal-token updates at the same
sample x layer.  Therefore the script additionally measures ORACLE LAYER SIGN
PURITY:

    purity(sample,L)
      = max(# positive causal-token updates,
            # negative causal-token updates)
        / (# nonzero causal-token updates)

and an |B|-weighted purity.

If purity is low, then "the layer is positive/negative" is itself not a
well-defined description: different causal-token positions in the same layer
have opposite behavioral signs.  In that case no single h_sub-h_ref layer sign
can perfectly predict every token-level B sign.

Spatial sign rules compared to behavioral sign
===============================================
For each representation (img / residual / no_image):

A) delta_fourway
       sign[
         cosine(Δr,d_GT) - max_{k!=GT} cosine(Δr,d_k)
       ]

B) delta_axis
       sign[
         cosine(Δr,d_GT) - cosine(Δr,d_opposite(GT))
       ]

C) same_basis_margin_shift
       sign[
         m_L(r_L) - m_L(r_{L-1})
       ]

D) same_basis_axis_shift
       sign[
         axis_margin_L(r_L) - axis_margin_L(r_{L-1})
       ]

E) gt_score_change
       sign[
         score_L(GT|r_L) - score_L(GT|r_{L-1})
       ]

We compare each rule against:
  1. every individual causal-token B_REAL sign;
  2. the net sample x layer behavioral sign sign(sum_p B_REAL);
  3. high-purity sample x layer cells only.

This is diagnostic only; GT is used to define "spatial positive/negative".
It is NOT a deployable non-oracle selector.

Recommended N=80 first
======================
CUDA_VISIBLE_DEVICES=0 python -u analyze_hsub_href_spatial_update_sign_v1.py \
  --model qwen-3b \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --target-max-samples 80 \
  --cache-dir output/qwen3b_hsub_href_spatial_cache \
  --output-dir output/qwen3b_hsub_href_spatial_update_sign_n80_v1 \
  --overwrite

Full 440
========
CUDA_VISIBLE_DEVICES=0 python -u analyze_hsub_href_spatial_update_sign_v1.py \
  --model qwen-3b \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --prior-real-update-dir output/qwen3b_real_causal_token_updates_all440_v1 \
  --cache-dir output/qwen3b_hsub_href_spatial_cache \
  --output-dir output/qwen3b_hsub_href_spatial_update_sign_all440_v1 \
  --overwrite

On the second run, Synthetic-400 cache is reused.  The N80 target cache and
all-440 target cache have different filenames.

Main outputs
============
source_oof_layer_accuracy.csv
    Does Synthetic-400 h_sub-h_ref actually carry a transferable four-way
    spatial code at each layer?

target_state_by_layer.csv
    COCO state relation accuracy by layer, all / baseline-wrong /
    baseline-correct, for img / residual / no_image.

per_sample_layer_spatial.csv
    Every sample x layer state and spatial-update metric.

spatial_update_summary_by_layer.csv
    How often each spatial update rule is GT-positive / GT-negative and how
    often Δr directly points toward the GT relation.

oracle_behavior_layer_purity.csv
    Whether one sign per sample x layer is even meaningful.

spatial_vs_behavior_per_update.csv
    Each token-level causal update joined with the sample x layer spatial sign.

spatial_vs_behavior_sign_summary.csv
    Main sign agreement against individual token-level B_REAL.

spatial_vs_behavior_layer_net.csv
    Sign agreement against sign(sum_p B_REAL) at sample x layer level.

spatial_vs_behavior_layer_net_summary.csv
    Summary of the previous table.

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
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    import extract_two_object_relation_states as base
except Exception as exc:
    raise SystemExit(
        "Could not import extract_two_object_relation_states.py.\n"
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )

try:
    import analyze_coco_head_object_residual_direction_probe_v1 as dh
except Exception as exc:
    raise SystemExit(
        "Could not import analyze_coco_head_object_residual_direction_probe_v1.py.\n"
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )


SCRIPT_VERSION = "hsub-href-spatial-update-sign-v1"

REL = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(REL)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
EPS = 1e-12

DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)

SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "top": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "bottom": "below",
}

REPR_NAMES = ("img", "residual", "no_image")

SIGN_RULES = (
    "delta_fourway_margin",
    "delta_axis_margin",
    "same_basis_margin_shift",
    "same_basis_axis_shift",
    "gt_score_change",
)


# =============================================================================
# CLI / basic helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT,
        help=(
            "Keep identical for Synthetic source and COCO target. "
            "Default exactly matches the existing Direction-Head probe."
        ),
    )
    p.add_argument(
        "--pool",
        default="mean",
        choices=["mean", "last"],
        help="Pooling over multi-token subject/reference phrases.",
    )

    p.add_argument(
        "--synthetic-dir",
        default="synthetic_shapes_4dir_400",
    )
    p.add_argument(
        "--synthetic-labels",
        default="",
        help="Default: <synthetic-dir>/labels.jsonl",
    )
    p.add_argument(
        "--source-max-samples",
        type=int,
        default=0,
        help="0 = all Synthetic source samples.",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=17)

    p.add_argument(
        "--target-max-samples",
        type=int,
        default=0,
        help="0 = full COCO target.",
    )

    p.add_argument(
        "--prior-real-update-dir",
        required=True,
        help=(
            "Output from eval_real_causal_token_update_gating_v1.py. "
            "Must contain per_real_update_decision_score.csv and "
            "generation_per_sample.csv."
        ),
    )

    p.add_argument(
        "--analysis-layers",
        default="",
        help=(
            "Optional comma/range list, e.g. 8-26. Empty = all layers that "
            "exist in both hidden-state cache and prior B_REAL table."
        ),
    )

    p.add_argument(
        "--behavior-threshold",
        type=float,
        default=1e-8,
        help="|B_REAL| <= threshold is excluded from sign accuracy.",
    )
    p.add_argument(
        "--spatial-threshold",
        type=float,
        default=0.0,
        help="|spatial sign score| <= threshold abstains.",
    )
    p.add_argument(
        "--purity-threshold",
        type=float,
        default=0.80,
        help="High-purity sample x layer threshold for secondary analysis.",
    )

    p.add_argument(
        "--cache-dir",
        default="output/hsub_href_spatial_cache",
        help="Persistent extraction cache. Not removed by --overwrite.",
    )
    p.add_argument(
        "--overwrite-source-cache",
        action="store_true",
    )
    p.add_argument(
        "--overwrite-target-cache",
        action="store_true",
    )

    p.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Cache hidden relation vectors as fp32 instead of fp16.",
    )
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def canon_rel(x):
    s = str(x).strip().lower().replace("-", "_")
    return SYN_REL_MAP.get(s, s)


def display_rel(x):
    return DISPLAY.get(canon_rel(x), str(x))


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", ""}:
        return False
    return bool(x)


def parse_layers(text):
    if not str(text).strip():
        return None

    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def ensure_outdir(path: Path, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_rows(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def safe_mean(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def weighted_accuracy(correct, weights):
    c = np.asarray(correct, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(c) & np.isfinite(w) & (w >= 0)
    if not np.any(ok):
        return float("nan")
    den = float(w[ok].sum())
    if den <= EPS:
        return float("nan")
    return float(np.sum(c[ok] * w[ok]) / den)


def sign_with_threshold(v, threshold):
    v = float(v)
    if not np.isfinite(v):
        return 0
    if v > threshold:
        return 1
    if v < -threshold:
        return -1
    return 0


# =============================================================================
# Data loading
# =============================================================================

def load_synthetic_rows(args):
    root = Path(args.synthetic_dir)
    labels_path = (
        Path(args.synthetic_labels)
        if args.synthetic_labels
        else root / "labels.jsonl"
    )
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            raw = canon_rel(item["relation"])
            if raw not in REL:
                raise RuntimeError(
                    f"{labels_path}:{line_no}: unsupported relation={item['relation']!r}"
                )

            image_value = Path(str(item["image"]))
            image_path = (
                image_value
                if image_value.is_absolute()
                else root / image_value
            )
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()

            rows.append(
                {
                    "sid": int(item.get("id", len(rows))),
                    "image_path": str(image_path),
                    "subject": subject,
                    "reference": reference,
                    "relation": raw,
                    "question": args.prompt_template.format(
                        subject=subject,
                        reference=reference,
                    ),
                }
            )

    rows.sort(key=lambda x: int(x["sid"]))
    if args.source_max_samples > 0:
        rows = rows[: int(args.source_max_samples)]

    counts = Counter(r["relation"] for r in rows)
    missing = [r for r in REL if counts[r] == 0]
    if missing:
        raise RuntimeError(
            f"Synthetic source missing relations {missing}; counts={dict(counts)}"
        )

    return rows, labels_path


def load_target_rows(args):
    max_samples = (
        int(args.target_max_samples)
        if args.target_max_samples > 0
        else None
    )

    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        max_samples,
    )

    rows = []
    for rec in records:
        rel = canon_rel(rec.relation)
        if rel not in REL:
            continue

        subject = str(rec.subject)
        reference = str(rec.reference)

        rows.append(
            {
                "sid": int(rec.sid),
                "image_path": str(rec.image_path),
                "subject": subject,
                "reference": reference,
                "relation": rel,
                "question": args.prompt_template.format(
                    subject=subject,
                    reference=reference,
                ),
            }
        )

    return rows, audit


# =============================================================================
# Model + final block-output h_sub / h_ref capture
# =============================================================================

def first_tensor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item):
                return item

    raise RuntimeError(
        f"Could not find tensor in layer output type={type(output)}"
    )


class HiddenObjectCapture:
    """
    Capture FINAL block output at subject/reference phrase positions.

    out[L,0,:] = pooled h_sub after decoder block L
    out[L,1,:] = pooled h_ref after decoder block L
    """

    def __init__(self, layers, a_pos, b_pos, pool):
        self.layers = layers
        self.a_pos = list(map(int, a_pos))
        self.b_pos = list(map(int, b_pos))
        self.pool = str(pool)
        self.out = {}
        self.seen = set()
        self.handles = []

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            def make_hook(layer_idx):
                def hook(_module, _inputs, output):
                    x = first_tensor(output)
                    a = dh.pool_positions(
                        x,
                        self.a_pos,
                        self.pool,
                    )
                    b = dh.pool_positions(
                        x,
                        self.b_pos,
                        self.pool,
                    )
                    self.out[layer_idx] = torch.stack(
                        [a, b],
                        dim=0,
                    ).detach().float().cpu()
                    self.seen.add(layer_idx)
                return hook

            self.handles.append(
                layer.register_forward_hook(make_hook(L))
            )

        return self

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles = []

    def __exit__(self, *args):
        self.close()

    def finalize(self):
        missing = [
            L for L in range(len(self.layers))
            if L not in self.seen
        ]
        if missing:
            raise RuntimeError(
                f"Missing block-output captures: {missing[:20]}"
            )

        return torch.stack(
            [self.out[L] for L in range(len(self.layers))],
            dim=0,
        ).numpy()


def capture_hidden_condition(
    *,
    model,
    processor,
    device,
    layers,
    question,
    subject,
    reference,
    image,
    pool,
):
    # Exact same prompt/input/phrase-position machinery as Direction-Head probe.
    rendered = dh.build_chat_prompt(
        processor,
        question,
        image is not None,
    )
    batch = dh.process_inputs(
        processor,
        rendered,
        image,
        device,
    )

    ids = [
        int(x)
        for x in batch["input_ids"][0].detach().cpu().tolist()
    ]

    a_pos = dh.locate_phrase_positions(
        processor.tokenizer,
        ids,
        subject,
    )
    b_pos = dh.locate_phrase_positions(
        processor.tokenizer,
        ids,
        reference,
    )

    cap = HiddenObjectCapture(
        layers,
        a_pos,
        b_pos,
        pool,
    )

    try:
        with cap:
            with torch.inference_mode():
                model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        out = cap.finalize()
    finally:
        cap.close()
        del batch

    return out


def load_model(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.model not in base.SPECS:
        raise KeyError(
            f"Unknown model={args.model!r}; "
            f"available={sorted(base.SPECS.keys())}"
        )

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id}", flush=True)
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)

    layers, decoder_path = dh.resolve_decoder_layers(model)

    print(
        f"[model] decoder={decoder_path} layers={len(layers)}",
        flush=True,
    )

    return model, processor, layers, spec, decoder_path


# =============================================================================
# Hidden relation cache extraction
# =============================================================================

def cache_expected_sids(rows):
    return np.asarray([int(r["sid"]) for r in rows], dtype=int)


def load_hidden_cache(path, expected_sids=None):
    z = np.load(path, allow_pickle=True)

    required = {
        "sample_index",
        "relation",
        "img",
        "no_image",
    }
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(
            f"{path} missing arrays: {sorted(missing)}"
        )

    sid = np.asarray(z["sample_index"]).astype(int)
    y = np.asarray(
        [canon_rel(x) for x in z["relation"]],
        dtype=object,
    )
    Xi = np.asarray(z["img"])
    Xn = np.asarray(z["no_image"])

    if expected_sids is not None:
        expected_sids = np.asarray(expected_sids).astype(int)
        if (
            len(sid) != len(expected_sids)
            or not np.array_equal(sid, expected_sids)
        ):
            raise RuntimeError(
                f"Cache SID mismatch: {path}\n"
                f"cache N={len(sid)}, expected N={len(expected_sids)}\n"
                "Use --overwrite-source-cache / --overwrite-target-cache."
            )

    return sid, y, Xi, Xn


def extract_hidden_cache(
    *,
    args,
    rows,
    model,
    processor,
    layers,
    cache_path,
    desc,
):
    dtype_np = np.float32 if args.keep_fp32 else np.float16
    device = torch.device(args.device)

    sids = []
    labels = []
    img_rel = []
    noimg_rel = []
    errors = []

    for rec in tqdm(rows, desc=desc):
        image = None
        try:
            image = Image.open(rec["image_path"]).convert("RGB")

            hi = capture_hidden_condition(
                model=model,
                processor=processor,
                device=device,
                layers=layers,
                question=rec["question"],
                subject=rec["subject"],
                reference=rec["reference"],
                image=image,
                pool=args.pool,
            )

            hn = capture_hidden_condition(
                model=model,
                processor=processor,
                device=device,
                layers=layers,
                question=rec["question"],
                subject=rec["subject"],
                reference=rec["reference"],
                image=None,
                pool=args.pool,
            )

            # [L,2,D] -> [L,D]
            ri = hi[:, 0, :] - hi[:, 1, :]
            rn = hn[:, 0, :] - hn[:, 1, :]

            sids.append(int(rec["sid"]))
            labels.append(canon_rel(rec["relation"]))
            img_rel.append(ri.astype(dtype_np))
            noimg_rel.append(rn.astype(dtype_np))

            del hi, hn, ri, rn

        except Exception as exc:
            errors.append(
                {
                    "sid": int(rec["sid"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                }
            )
            tqdm.write(
                f"[ERROR] {desc} sid={rec['sid']}: "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                with contextlib.suppress(Exception):
                    image.close()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not img_rel:
        raise RuntimeError(f"{desc}: zero successful samples")

    sid = np.asarray(sids, dtype=int)
    y = np.asarray(labels, dtype=object)
    Xi = np.stack(img_rel)
    Xn = np.stack(noimg_rel)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_index=sid,
        relation=y,
        img=Xi,
        no_image=Xn,
        decoder_block_index=np.arange(Xi.shape[1]),
        vector_definition=np.asarray(
            "block_out[L,subject]-block_out[L,reference]",
            dtype=object,
        ),
        prompt_template=np.asarray(
            args.prompt_template,
            dtype=object,
        ),
        pool=np.asarray(args.pool, dtype=object),
        model=np.asarray(args.model, dtype=object),
    )

    err_path = cache_path.with_suffix(
        cache_path.suffix + ".errors.json"
    )
    err_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[cache] saved {cache_path} | "
        f"N={len(y)} shape={Xi.shape}",
        flush=True,
    )

    return sid, y, Xi, Xn


def representation_array(img, no_image, name):
    if name == "img":
        return np.asarray(img, dtype=np.float32)
    if name == "no_image":
        return np.asarray(no_image, dtype=np.float32)
    if name == "residual":
        return (
            np.asarray(img, dtype=np.float32)
            - np.asarray(no_image, dtype=np.float32)
        )
    raise KeyError(name)


# =============================================================================
# Synthetic-only codebook: EXACT rule used by original Direction selector
# =============================================================================

def fit_layer_codebook(X, y):
    """
    X: [N,L,D]
    Returns:
      center [L,D]
      dirs   [L,R,D]

    Same rule as existing Direction-Head source-only codebook, simply with no
    head dimension.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray([canon_rel(v) for v in y], dtype=object)

    center = X.mean(axis=0)

    dirs = np.zeros(
        (X.shape[1], len(REL), X.shape[2]),
        dtype=np.float32,
    )

    for ri, rel in enumerate(REL):
        mask = y == rel
        if not np.any(mask):
            raise RuntimeError(f"No source examples for {rel}")

        d = X[mask].mean(axis=0) - center
        dirs[:, ri, :] = normalize_rows(d, axis=-1)

    return center.astype(np.float32), dirs


def score_layer_states(X, center, dirs):
    """
    X [N,L,D]
    center [L,D]
    dirs [L,R,D]
    -> scores [N,L,R]
    """
    X = np.asarray(X, dtype=np.float32)
    Xc = X - center[None, :, :]
    Xn = normalize_rows(Xc, axis=-1)
    return np.einsum(
        "nld,lrd->nlr",
        Xn,
        dirs,
        optimize=True,
    )


def stratified_folds(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_folds)]

    for rel in REL:
        idx = np.where(np.asarray(y) == rel)[0].copy()
        rng.shuffle(idx)
        parts = np.array_split(idx, n_folds)
        for fold, part in enumerate(parts):
            buckets[fold].extend(part.tolist())

    return [
        np.asarray(sorted(x), dtype=int)
        for x in buckets
    ]


def source_oof_layer_accuracy(X, y, n_folds, seed, representation):
    y = np.asarray([canon_rel(v) for v in y], dtype=object)
    yi = np.asarray([RID[v] for v in y], dtype=int)

    folds = stratified_folds(y, n_folds, seed)
    all_idx = np.arange(len(y))

    correct = np.zeros(X.shape[1], dtype=np.int64)
    rows = []

    for fold, te in enumerate(folds):
        keep = np.ones(len(y), dtype=bool)
        keep[te] = False
        tr = all_idx[keep]

        center, dirs = fit_layer_codebook(
            X[tr],
            y[tr],
        )
        scores = score_layer_states(
            X[te],
            center,
            dirs,
        )
        pred = np.argmax(scores, axis=-1)
        ok = pred == yi[te, None]

        correct += ok.sum(axis=0)

        for L in range(X.shape[1]):
            rows.append(
                {
                    "representation": representation,
                    "fold": int(fold),
                    "layer": int(L),
                    "N_train": int(len(tr)),
                    "N_val": int(len(te)),
                    "accuracy": float(ok[:, L].mean()),
                }
            )

    acc = correct.astype(np.float64) / float(len(y))

    summary = pd.DataFrame(
        {
            "representation": representation,
            "layer": np.arange(X.shape[1], dtype=int),
            "source_oof_accuracy": acc,
        }
    )

    return summary, pd.DataFrame(rows)


# =============================================================================
# Baseline + oracle behavioral B
# =============================================================================

def load_prior_behavior(prior_dir: Path):
    update_path = prior_dir / "per_real_update_decision_score.csv"
    gen_path = prior_dir / "generation_per_sample.csv"

    if not update_path.exists():
        raise FileNotFoundError(update_path)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    upd = pd.read_csv(update_path)
    gen = pd.read_csv(gen_path)

    req = {
        "sid",
        "gt",
        "update_layer",
        "real_position",
        "real_update_decision_score",
    }
    missing = req - set(upd.columns)
    if missing:
        raise RuntimeError(
            f"{update_path} missing columns: {sorted(missing)}"
        )

    upd["sid"] = pd.to_numeric(
        upd["sid"], errors="raise"
    ).astype(int)
    upd["update_layer"] = pd.to_numeric(
        upd["update_layer"], errors="raise"
    ).astype(int)
    upd["real_position"] = pd.to_numeric(
        upd["real_position"], errors="raise"
    ).astype(int)
    upd["real_update_decision_score"] = pd.to_numeric(
        upd["real_update_decision_score"],
        errors="coerce",
    )
    upd["gt"] = upd["gt"].map(canon_rel)

    baseline = gen.copy()
    if "condition" in baseline.columns:
        baseline = baseline[
            baseline["condition"].astype(str) == "baseline"
        ].copy()

    baseline = baseline.sort_values("sid").drop_duplicates("sid")

    if "correct" in baseline.columns:
        baseline["baseline_correct"] = baseline["correct"].map(boolify)
    elif "baseline_correct" in baseline.columns:
        baseline["baseline_correct"] = baseline[
            "baseline_correct"
        ].map(boolify)
    else:
        raise RuntimeError(
            f"{gen_path}: cannot find baseline correctness"
        )

    if "prediction" in baseline.columns:
        baseline["baseline_prediction"] = baseline[
            "prediction"
        ].map(canon_rel)
    elif "baseline_prediction" in baseline.columns:
        baseline["baseline_prediction"] = baseline[
            "baseline_prediction"
        ].map(canon_rel)
    else:
        baseline["baseline_prediction"] = ""

    cols = [
        "sid",
        "baseline_correct",
        "baseline_prediction",
    ]
    if "gt" in baseline.columns:
        baseline["gt"] = baseline["gt"].map(canon_rel)
        cols.append("gt")

    baseline = baseline[cols].copy()

    return upd, baseline, update_path, gen_path


# =============================================================================
# Spatial state + update analysis
# =============================================================================

def fourway_margin(scores, gt):
    gi = RID[gt]
    other = max(
        float(scores[j])
        for j in range(len(REL))
        if j != gi
    )
    return float(scores[gi] - other)


def axis_margin(scores, gt):
    return float(
        scores[RID[gt]]
        - scores[RID[OPPOSITE[gt]]]
    )


def analyze_target_representation(
    *,
    representation,
    X,
    y,
    sids,
    center,
    dirs,
    baseline_map,
):
    """
    One row per sample x layer.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray([canon_rel(v) for v in y], dtype=object)
    sids = np.asarray(sids, dtype=int)

    state_scores = score_layer_states(
        X,
        center,
        dirs,
    )

    rows = []

    for i in range(len(X)):
        sid = int(sids[i])
        gt = str(y[i])
        bmeta = baseline_map.get(sid, {})

        for L in range(X.shape[1]):
            after = state_scores[i, L]
            state_pred_idx = int(np.argmax(after))
            state_pred = REL[state_pred_idx]

            state_four = fourway_margin(after, gt)
            state_axis = axis_margin(after, gt)

            row = {
                "sid": sid,
                "gt": gt,
                "relation": display_rel(gt),
                "representation": representation,
                "layer": int(L),
                "baseline_correct": bmeta.get(
                    "baseline_correct",
                    np.nan,
                ),
                "baseline_prediction": bmeta.get(
                    "baseline_prediction",
                    "",
                ),
                "state_pred": state_pred,
                "state_correct": state_pred == gt,
                "state_gt_fourway_margin": state_four,
                "state_gt_axis_margin": state_axis,
            }

            for rel in REL:
                row[f"state_score_{rel}"] = float(
                    after[RID[rel]]
                )

            if L == 0:
                row.update(
                    {
                        "delta_norm": np.nan,
                        "delta_pred": "",
                        "delta_score_change_pred": "",
                        "delta_fourway_margin": np.nan,
                        "delta_axis_margin": np.nan,
                        "same_basis_before_fourway_margin": np.nan,
                        "same_basis_before_axis_margin": np.nan,
                        "same_basis_margin_shift": np.nan,
                        "same_basis_axis_shift": np.nan,
                        "gt_score_change": np.nan,
                        "natural_margin_shift_basis_confounded": np.nan,
                    }
                )
                for rel in REL:
                    row[f"delta_direction_score_{rel}"] = np.nan
                    row[f"same_basis_score_change_{rel}"] = np.nan

                rows.append(row)
                continue

            prev = X[i, L - 1]
            cur = X[i, L]
            delta = cur - prev
            delta_norm = float(np.linalg.norm(delta))

            # -------------------------------------------------------------
            # A. Does Δr itself look like one of the source relation dirs?
            # -------------------------------------------------------------
            dnorm = delta / max(delta_norm, EPS)
            delta_scores = np.einsum(
                "rd,d->r",
                dirs[L],
                dnorm,
            )
            delta_pred = REL[int(np.argmax(delta_scores))]

            delta_four = fourway_margin(
                delta_scores,
                gt,
            )
            delta_axis = axis_margin(
                delta_scores,
                gt,
            )

            # -------------------------------------------------------------
            # B. Evaluate BEFORE and AFTER in the SAME layer-L codebook.
            # -------------------------------------------------------------
            prev_centered = prev - center[L]
            prev_n = prev_centered / max(
                float(np.linalg.norm(prev_centered)),
                EPS,
            )
            before_scores = np.einsum(
                "rd,d->r",
                dirs[L],
                prev_n,
            )

            before_four = fourway_margin(
                before_scores,
                gt,
            )
            before_axis = axis_margin(
                before_scores,
                gt,
            )

            same_basis_change = after - before_scores
            score_change_pred = REL[
                int(np.argmax(same_basis_change))
            ]

            margin_shift = state_four - before_four
            axis_shift = state_axis - before_axis
            gt_score_change = float(
                after[RID[gt]]
                - before_scores[RID[gt]]
            )

            # For reference only: this changes codebooks between L-1 and L.
            prev_natural_scores = state_scores[i, L - 1]
            prev_natural_margin = fourway_margin(
                prev_natural_scores,
                gt,
            )
            natural_shift = state_four - prev_natural_margin

            row.update(
                {
                    "delta_norm": delta_norm,
                    "delta_pred": delta_pred,
                    "delta_score_change_pred": score_change_pred,
                    "delta_fourway_margin": float(delta_four),
                    "delta_axis_margin": float(delta_axis),
                    "same_basis_before_fourway_margin": float(before_four),
                    "same_basis_before_axis_margin": float(before_axis),
                    "same_basis_margin_shift": float(margin_shift),
                    "same_basis_axis_shift": float(axis_shift),
                    "gt_score_change": gt_score_change,
                    "natural_margin_shift_basis_confounded": float(natural_shift),
                }
            )

            for rel in REL:
                row[f"delta_direction_score_{rel}"] = float(
                    delta_scores[RID[rel]]
                )
                row[f"same_basis_score_change_{rel}"] = float(
                    same_basis_change[RID[rel]]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def target_state_summary(spatial_df):
    rows = []

    for (representation, L), g in spatial_df.groupby(
        ["representation", "layer"]
    ):
        cohorts = [("all", g)]

        if g["baseline_correct"].notna().any():
            cohorts += [
                (
                    "baseline_wrong",
                    g[g["baseline_correct"] == False],  # noqa: E712
                ),
                (
                    "baseline_correct",
                    g[g["baseline_correct"] == True],  # noqa: E712
                ),
            ]

        for cohort, z in cohorts:
            if not len(z):
                continue

            rows.append(
                {
                    "representation": representation,
                    "layer": int(L),
                    "cohort": cohort,
                    "N": int(len(z)),
                    "state_accuracy": float(
                        z["state_correct"].mean()
                    ),
                    "mean_gt_fourway_margin": safe_mean(
                        z["state_gt_fourway_margin"]
                    ),
                    "mean_gt_axis_margin": safe_mean(
                        z["state_gt_axis_margin"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def spatial_update_summary(spatial_df):
    rows = []

    zall = spatial_df[spatial_df["layer"] > 0].copy()

    for (representation, L), g in zall.groupby(
        ["representation", "layer"]
    ):
        cohorts = [("all", g)]
        if g["baseline_correct"].notna().any():
            cohorts += [
                (
                    "baseline_wrong",
                    g[g["baseline_correct"] == False],  # noqa: E712
                ),
                (
                    "baseline_correct",
                    g[g["baseline_correct"] == True],  # noqa: E712
                ),
            ]

        for cohort, z in cohorts:
            if not len(z):
                continue

            row = {
                "representation": representation,
                "layer": int(L),
                "cohort": cohort,
                "N": int(len(z)),
                "delta_pred_gt_rate": float(
                    (z["delta_pred"] == z["gt"]).mean()
                ),
                "score_change_pred_gt_rate": float(
                    (
                        z["delta_score_change_pred"]
                        == z["gt"]
                    ).mean()
                ),
            }

            for rule in SIGN_RULES:
                vals = pd.to_numeric(
                    z[rule],
                    errors="coerce",
                ).to_numpy(float)
                finite = np.isfinite(vals)
                if np.any(finite):
                    row[f"{rule}_positive_fraction"] = float(
                        np.mean(vals[finite] > 0)
                    )
                    row[f"{rule}_negative_fraction"] = float(
                        np.mean(vals[finite] < 0)
                    )
                    row[f"{rule}_mean"] = float(
                        np.mean(vals[finite])
                    )
                else:
                    row[f"{rule}_positive_fraction"] = np.nan
                    row[f"{rule}_negative_fraction"] = np.nan
                    row[f"{rule}_mean"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Is one behavioral sign per sample x layer even meaningful?
# =============================================================================

def compute_layer_behavior(upd, behavior_threshold):
    rows = []

    z = upd[
        np.isfinite(upd["real_update_decision_score"])
    ].copy()

    for (sid, L), g in z.groupby(
        ["sid", "update_layer"]
    ):
        vals = g["real_update_decision_score"].to_numpy(float)

        pos = vals > behavior_threshold
        neg = vals < -behavior_threshold
        nonzero = pos | neg

        if not np.any(nonzero):
            continue

        vv = vals[nonzero]
        pos = vv > 0
        neg = vv < 0

        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        n = int(len(vv))

        abs_v = np.abs(vv)
        pos_mass = float(abs_v[pos].sum())
        neg_mass = float(abs_v[neg].sum())
        total_mass = pos_mass + neg_mass

        net_B = float(vv.sum())
        net_sign = sign_with_threshold(
            net_B,
            behavior_threshold,
        )

        majority_sign = (
            1 if n_pos > n_neg
            else -1 if n_neg > n_pos
            else 0
        )

        weighted_majority_sign = (
            1 if pos_mass > neg_mass
            else -1 if neg_mass > pos_mass
            else 0
        )

        rows.append(
            {
                "sid": int(sid),
                "layer": int(L),
                "N_behavior_updates": n,
                "N_positive": n_pos,
                "N_negative": n_neg,
                "positive_fraction": n_pos / n,
                "negative_fraction": n_neg / n,
                "sign_purity": max(n_pos, n_neg) / n,
                "positive_abs_mass": pos_mass,
                "negative_abs_mass": neg_mass,
                "weighted_sign_purity": (
                    max(pos_mass, neg_mass)
                    / max(total_mass, EPS)
                ),
                "sum_B": net_B,
                "mean_B": float(vv.mean()),
                "mean_abs_B": float(abs_v.mean()),
                "net_behavior_sign": int(net_sign),
                "majority_behavior_sign": int(majority_sign),
                "weighted_majority_behavior_sign": int(
                    weighted_majority_sign
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_layer_purity(layer_behavior, baseline):
    x = layer_behavior.merge(
        baseline[
            [
                "sid",
                "baseline_correct",
                "baseline_prediction",
            ]
        ],
        on="sid",
        how="left",
    )

    rows = []

    for cohort, g in [
        ("all", x),
        (
            "baseline_wrong",
            x[x["baseline_correct"] == False],  # noqa: E712
        ),
        (
            "baseline_correct",
            x[x["baseline_correct"] == True],  # noqa: E712
        ),
    ]:
        if not len(g):
            continue

        rows.append(
            {
                "cohort": cohort,
                "N_sample_layers": int(len(g)),
                "mean_sign_purity": safe_mean(
                    g["sign_purity"]
                ),
                "median_sign_purity": float(
                    g["sign_purity"].median()
                ),
                "mean_weighted_sign_purity": safe_mean(
                    g["weighted_sign_purity"]
                ),
                "fraction_purity_1": float(
                    np.mean(g["sign_purity"] >= 1.0 - 1e-12)
                ),
                "fraction_purity_ge_0p8": float(
                    np.mean(g["sign_purity"] >= 0.80)
                ),
                "fraction_net_positive": float(
                    np.mean(g["net_behavior_sign"] > 0)
                ),
                "fraction_net_negative": float(
                    np.mean(g["net_behavior_sign"] < 0)
                ),
            }
        )

    return pd.DataFrame(rows), x


# =============================================================================
# Spatial sign versus token-level behavioral sign
# =============================================================================

def join_spatial_to_behavior(
    spatial_df,
    upd,
    baseline,
    analysis_layers,
    behavior_threshold,
    spatial_threshold,
    layer_behavior,
):
    # target spatial rows are one sample x layer x representation.
    s = spatial_df[
        spatial_df["layer"] > 0
    ].copy()

    if analysis_layers is not None:
        s = s[s["layer"].isin(set(analysis_layers))].copy()
        upd = upd[
            upd["update_layer"].isin(set(analysis_layers))
        ].copy()

    s = s.rename(columns={"layer": "update_layer"})

    # Ensure baseline metadata comes from prior run.
    bcols = [
        "sid",
        "baseline_correct",
        "baseline_prediction",
    ]
    s = s.drop(
        columns=[
            c for c in bcols[1:]
            if c in s.columns
        ],
        errors="ignore",
    )
    s = s.merge(
        baseline[bcols],
        on="sid",
        how="left",
        validate="many_to_one",
    )

    keep_upd = [
        "sid",
        "gt",
        "update_layer",
        "real_position",
        "real_update_decision_score",
    ]
    for c in (
        "token",
        "category",
        "broad_category",
        "max_target_layer",
    ):
        if c in upd.columns:
            keep_upd.append(c)

    joined = upd[keep_upd].merge(
        s,
        on=["sid", "update_layer"],
        how="inner",
        suffixes=("_behavior", ""),
        validate="many_to_many",
    )

    # Safety: representation causes 3 rows per behavioral update.
    joined["oracle_behavior_sign"] = joined[
        "real_update_decision_score"
    ].map(
        lambda v: sign_with_threshold(
            v,
            behavior_threshold,
        )
    )
    joined["oracle_abs_B"] = joined[
        "real_update_decision_score"
    ].abs()

    for rule in SIGN_RULES:
        joined[f"pred_sign_{rule}"] = joined[rule].map(
            lambda v: sign_with_threshold(
                v,
                spatial_threshold,
            )
        )

    layer_meta = layer_behavior[
        [
            "sid",
            "layer",
            "sign_purity",
            "weighted_sign_purity",
            "net_behavior_sign",
            "sum_B",
            "mean_abs_B",
            "N_behavior_updates",
        ]
    ].rename(columns={"layer": "update_layer"})

    joined = joined.merge(
        layer_meta,
        on=["sid", "update_layer"],
        how="left",
        validate="many_to_one",
    )

    return joined


def sign_summary_per_update(
    joined,
    purity_threshold,
):
    rows = []

    for representation in REPR_NAMES:
        base = joined[
            joined["representation"] == representation
        ].copy()

        cohort_defs = [
            ("all", base),
            (
                "baseline_wrong",
                base[base["baseline_correct"] == False],  # noqa: E712
            ),
            (
                "baseline_correct",
                base[base["baseline_correct"] == True],  # noqa: E712
            ),
            (
                f"high_purity_ge_{purity_threshold:g}",
                base[
                    base["sign_purity"]
                    >= float(purity_threshold)
                ],
            ),
            (
                f"baseline_wrong_high_purity_ge_{purity_threshold:g}",
                base[
                    (base["baseline_correct"] == False)  # noqa: E712
                    & (
                        base["sign_purity"]
                        >= float(purity_threshold)
                    )
                ],
            ),
        ]

        for cohort, g0 in cohort_defs:
            if not len(g0):
                continue

            for rule in SIGN_RULES:
                pred = g0[f"pred_sign_{rule}"].to_numpy(int)
                oracle = g0["oracle_behavior_sign"].to_numpy(int)

                mask = (pred != 0) & (oracle != 0)
                if not np.any(mask):
                    continue

                correct = pred[mask] == oracle[mask]
                weights = g0.loc[
                    g0.index[mask],
                    "oracle_abs_B",
                ].to_numpy(float)

                rows.append(
                    {
                        "representation": representation,
                        "cohort": cohort,
                        "rule": rule,
                        "N_total_rows": int(len(g0)),
                        "N_evaluated": int(mask.sum()),
                        "coverage": float(mask.mean()),
                        "sign_accuracy": float(
                            np.mean(correct)
                        ),
                        "weighted_sign_accuracy": weighted_accuracy(
                            correct,
                            weights,
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Spatial sign versus NET sample x layer behavioral sign
# =============================================================================

def build_layer_net_table(
    spatial_df,
    layer_behavior,
    baseline,
    analysis_layers,
    spatial_threshold,
):
    s = spatial_df[
        spatial_df["layer"] > 0
    ].copy()

    if analysis_layers is not None:
        s = s[s["layer"].isin(set(analysis_layers))].copy()

    x = s.merge(
        layer_behavior,
        on=["sid", "layer"],
        how="inner",
        validate="many_to_one",
    )

    x = x.drop(
        columns=[
            c for c in (
                "baseline_correct",
                "baseline_prediction",
            )
            if c in x.columns
        ],
        errors="ignore",
    )

    x = x.merge(
        baseline[
            [
                "sid",
                "baseline_correct",
                "baseline_prediction",
            ]
        ],
        on="sid",
        how="left",
        validate="many_to_one",
    )

    for rule in SIGN_RULES:
        x[f"pred_sign_{rule}"] = x[rule].map(
            lambda v: sign_with_threshold(
                v,
                spatial_threshold,
            )
        )

    return x


def summarize_layer_net(layer_net, purity_threshold):
    rows = []

    for representation in REPR_NAMES:
        base = layer_net[
            layer_net["representation"] == representation
        ].copy()

        cohort_defs = [
            ("all", base),
            (
                "baseline_wrong",
                base[base["baseline_correct"] == False],  # noqa: E712
            ),
            (
                "baseline_correct",
                base[base["baseline_correct"] == True],  # noqa: E712
            ),
            (
                f"high_purity_ge_{purity_threshold:g}",
                base[
                    base["sign_purity"]
                    >= float(purity_threshold)
                ],
            ),
            (
                f"baseline_wrong_high_purity_ge_{purity_threshold:g}",
                base[
                    (base["baseline_correct"] == False)  # noqa: E712
                    & (
                        base["sign_purity"]
                        >= float(purity_threshold)
                    )
                ],
            ),
        ]

        for cohort, g0 in cohort_defs:
            if not len(g0):
                continue

            oracle = g0["net_behavior_sign"].to_numpy(int)

            for rule in SIGN_RULES:
                pred = g0[f"pred_sign_{rule}"].to_numpy(int)
                mask = (pred != 0) & (oracle != 0)

                if not np.any(mask):
                    continue

                correct = pred[mask] == oracle[mask]
                weights = np.abs(
                    g0.loc[
                        g0.index[mask],
                        "sum_B",
                    ].to_numpy(float)
                )

                rows.append(
                    {
                        "representation": representation,
                        "cohort": cohort,
                        "rule": rule,
                        "N_sample_layers": int(len(g0)),
                        "N_evaluated": int(mask.sum()),
                        "coverage": float(mask.mean()),
                        "sign_accuracy_vs_net_layer_B": float(
                            np.mean(correct)
                        ),
                        "weighted_sign_accuracy_vs_net_layer_B": (
                            weighted_accuracy(correct, weights)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def sign_by_layer(joined):
    rows = []

    for (representation, L), g0 in joined.groupby(
        ["representation", "update_layer"]
    ):
        for cohort, g in [
            ("all", g0),
            (
                "baseline_wrong",
                g0[g0["baseline_correct"] == False],  # noqa: E712
            ),
            (
                "baseline_correct",
                g0[g0["baseline_correct"] == True],  # noqa: E712
            ),
        ]:
            if not len(g):
                continue

            oracle = g["oracle_behavior_sign"].to_numpy(int)

            for rule in SIGN_RULES:
                pred = g[f"pred_sign_{rule}"].to_numpy(int)
                mask = (pred != 0) & (oracle != 0)
                if not np.any(mask):
                    continue

                correct = pred[mask] == oracle[mask]
                weights = g.loc[
                    g.index[mask],
                    "oracle_abs_B",
                ].to_numpy(float)

                rows.append(
                    {
                        "representation": representation,
                        "update_layer": int(L),
                        "cohort": cohort,
                        "rule": rule,
                        "N": int(mask.sum()),
                        "sign_accuracy": float(
                            np.mean(correct)
                        ),
                        "weighted_sign_accuracy": weighted_accuracy(
                            correct,
                            weights,
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Reporting
# =============================================================================

def top_layer_rows(df, value_col, representation, cohort=None, n=8):
    z = df[df["representation"] == representation].copy()
    if cohort is not None and "cohort" in z.columns:
        z = z[z["cohort"] == cohort].copy()

    if not len(z):
        return "EMPTY"

    z = z.sort_values(value_col, ascending=False).head(n)
    cols = [
        c for c in (
            "representation",
            "layer",
            "cohort",
            "N",
            value_col,
        )
        if c in z.columns
    ]

    return z[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    ensure_outdir(outdir, args.overwrite)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    error_path = outdir / "errors.jsonl"

    # -------------------------------------------------------------------------
    # 0. Inputs: source rows, target rows, old behavioral oracle.
    # -------------------------------------------------------------------------
    source_rows, labels_path = load_synthetic_rows(args)
    target_rows, target_audit = load_target_rows(args)

    prior_dir = Path(args.prior_real_update_dir)
    upd, baseline, update_path, gen_path = load_prior_behavior(
        prior_dir
    )

    baseline_map = {
        int(r.sid): {
            "baseline_correct": bool(r.baseline_correct),
            "baseline_prediction": canon_rel(
                r.baseline_prediction
            ),
        }
        for r in baseline.itertuples()
    }

    target_sid_expected = cache_expected_sids(target_rows)
    source_sid_expected = cache_expected_sids(source_rows)

    target_tag = (
        f"N{int(args.target_max_samples)}"
        if args.target_max_samples > 0
        else "all"
    )
    source_tag = (
        f"N{int(args.source_max_samples)}"
        if args.source_max_samples > 0
        else "all"
    )

    source_cache = cache_dir / (
        f"{args.model}_synthetic_hsub_href_{source_tag}.npz"
    )
    target_cache = cache_dir / (
        f"{args.model}_{args.dataset}_hsub_href_{target_tag}.npz"
    )

    model = processor = layers = spec = None
    decoder_path = ""

    try:
        need_source = (
            args.overwrite_source_cache
            or not source_cache.exists()
        )
        need_target = (
            args.overwrite_target_cache
            or not target_cache.exists()
        )

        if need_source or need_target:
            (
                model,
                processor,
                layers,
                spec,
                decoder_path,
            ) = load_model(args)

        # ---------------------------------------------------------------------
        # 1. Synthetic hidden relation cache.
        # ---------------------------------------------------------------------
        if need_source:
            (
                source_sid,
                source_y,
                source_img,
                source_no,
            ) = extract_hidden_cache(
                args=args,
                rows=source_rows,
                model=model,
                processor=processor,
                layers=layers,
                cache_path=source_cache,
                desc="Synthetic-400 h_sub-h_ref",
            )
        else:
            print(
                f"[source] reusing {source_cache}",
                flush=True,
            )
            (
                source_sid,
                source_y,
                source_img,
                source_no,
            ) = load_hidden_cache(
                source_cache,
                expected_sids=source_sid_expected,
            )

        # ---------------------------------------------------------------------
        # 2. COCO hidden relation cache.
        # ---------------------------------------------------------------------
        if need_target:
            (
                target_sid,
                target_y,
                target_img,
                target_no,
            ) = extract_hidden_cache(
                args=args,
                rows=target_rows,
                model=model,
                processor=processor,
                layers=layers,
                cache_path=target_cache,
                desc=f"{args.dataset} h_sub-h_ref",
            )
        else:
            print(
                f"[target] reusing {target_cache}",
                flush=True,
            )
            (
                target_sid,
                target_y,
                target_img,
                target_no,
            ) = load_hidden_cache(
                target_cache,
                expected_sids=target_sid_expected,
            )

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if source_img.shape[1:] != target_img.shape[1:]:
        raise RuntimeError(
            "Synthetic/COCO hidden geometry mismatch:\n"
            f"source={source_img.shape}\n"
            f"target={target_img.shape}"
        )

    n_layers = int(source_img.shape[1])
    hidden_dim = int(source_img.shape[2])

    analysis_layers = parse_layers(args.analysis_layers)
    behavior_layers = sorted(
        set(
            pd.to_numeric(
                upd["update_layer"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )
    )

    if analysis_layers is None:
        analysis_layers = [
            L for L in behavior_layers
            if 0 <= L < n_layers
        ]
    else:
        analysis_layers = [
            L for L in analysis_layers
            if 0 <= L < n_layers
            and L in set(behavior_layers)
        ]

    if not analysis_layers:
        raise RuntimeError(
            "No analysis layers overlap hidden cache and prior B_REAL table."
        )

    print("=" * 200)
    print("H_SUB - H_REF SPATIAL STATE / UPDATE SIGN")
    print("=" * 200)
    print(
        f"source N={len(source_y)} | target N={len(target_y)} | "
        f"layers={n_layers} | hidden_dim={hidden_dim}"
    )
    print(f"analysis_layers={analysis_layers}")
    print(
        f"prior behavioral rows={len(upd)} | "
        f"baseline N={len(baseline)}"
    )
    print()

    # -------------------------------------------------------------------------
    # 3. Synthetic OOF validation + full frozen codebook.
    # -------------------------------------------------------------------------
    source_oof_parts = []
    source_fold_parts = []
    codebook_payload = {
        "relations": np.asarray(REL, dtype=object),
    }

    source_arrays = {}
    target_arrays = {}

    for representation in REPR_NAMES:
        source_X = representation_array(
            source_img,
            source_no,
            representation,
        )
        target_X = representation_array(
            target_img,
            target_no,
            representation,
        )

        source_arrays[representation] = source_X
        target_arrays[representation] = target_X

        summary, folds = source_oof_layer_accuracy(
            source_X,
            source_y,
            args.cv_folds,
            args.cv_seed,
            representation,
        )
        source_oof_parts.append(summary)
        source_fold_parts.append(folds)

        center, dirs = fit_layer_codebook(
            source_X,
            source_y,
        )

        codebook_payload[f"{representation}_center"] = center
        codebook_payload[f"{representation}_directions"] = dirs

    source_oof = pd.concat(
        source_oof_parts,
        ignore_index=True,
    )
    source_folds = pd.concat(
        source_fold_parts,
        ignore_index=True,
    )

    source_oof.to_csv(
        outdir / "source_oof_layer_accuracy.csv",
        index=False,
    )
    source_folds.to_csv(
        outdir / "source_oof_fold_accuracy.csv",
        index=False,
    )

    np.savez_compressed(
        outdir / "synthetic400_hsub_href_spatial_codebook.npz",
        **codebook_payload,
    )

    # -------------------------------------------------------------------------
    # 4. Apply source-only codebook to COCO state + layer transition.
    # -------------------------------------------------------------------------
    spatial_parts = []

    for representation in REPR_NAMES:
        center = codebook_payload[
            f"{representation}_center"
        ]
        dirs = codebook_payload[
            f"{representation}_directions"
        ]

        part = analyze_target_representation(
            representation=representation,
            X=target_arrays[representation],
            y=target_y,
            sids=target_sid,
            center=center,
            dirs=dirs,
            baseline_map=baseline_map,
        )
        spatial_parts.append(part)

    spatial_df = pd.concat(
        spatial_parts,
        ignore_index=True,
    )

    spatial_df.to_csv(
        outdir / "per_sample_layer_spatial.csv",
        index=False,
    )

    state_summary = target_state_summary(spatial_df)
    state_summary.to_csv(
        outdir / "target_state_by_layer.csv",
        index=False,
    )

    update_summary = spatial_update_summary(spatial_df)
    update_summary.to_csv(
        outdir / "spatial_update_summary_by_layer.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 5. First ask if one sign per sample x layer is even a valid abstraction.
    # -------------------------------------------------------------------------
    layer_behavior = compute_layer_behavior(
        upd,
        float(args.behavior_threshold),
    )
    purity_summary, layer_behavior_with_baseline = (
        summarize_layer_purity(
            layer_behavior,
            baseline,
        )
    )

    layer_behavior_with_baseline.to_csv(
        outdir / "oracle_behavior_layer_purity.csv",
        index=False,
    )
    purity_summary.to_csv(
        outdir / "oracle_behavior_layer_purity_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 6. Compare spatial layer sign to EVERY token-level causal update B sign.
    # -------------------------------------------------------------------------
    joined = join_spatial_to_behavior(
        spatial_df=spatial_df,
        upd=upd,
        baseline=baseline,
        analysis_layers=analysis_layers,
        behavior_threshold=float(args.behavior_threshold),
        spatial_threshold=float(args.spatial_threshold),
        layer_behavior=layer_behavior,
    )

    joined.to_csv(
        outdir / "spatial_vs_behavior_per_update.csv",
        index=False,
    )

    sign_summary = sign_summary_per_update(
        joined,
        float(args.purity_threshold),
    )
    sign_summary.to_csv(
        outdir / "spatial_vs_behavior_sign_summary.csv",
        index=False,
    )

    sign_layer = sign_by_layer(joined)
    sign_layer.to_csv(
        outdir / "spatial_vs_behavior_sign_by_layer.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 7. Fairer comparison: one spatial sign vs one NET behavioral sign per layer.
    # -------------------------------------------------------------------------
    layer_net = build_layer_net_table(
        spatial_df=spatial_df,
        layer_behavior=layer_behavior,
        baseline=baseline,
        analysis_layers=analysis_layers,
        spatial_threshold=float(args.spatial_threshold),
    )
    layer_net.to_csv(
        outdir / "spatial_vs_behavior_layer_net.csv",
        index=False,
    )

    layer_net_summary = summarize_layer_net(
        layer_net,
        float(args.purity_threshold),
    )
    layer_net_summary.to_csv(
        outdir / "spatial_vs_behavior_layer_net_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 8. Console + report.
    # -------------------------------------------------------------------------
    report = []
    report += [
        "=" * 200,
        "H_SUB - H_REF SPATIAL STATE / UPDATE SIGN",
        "=" * 200,
        (
            f"source N={len(source_y)} | target N={len(target_y)} | "
            f"layers={n_layers} | hidden_dim={hidden_dim}"
        ),
        f"analysis_layers={analysis_layers}",
        "",
        "A. SYNTHETIC-400 SOURCE-ONLY OOF: BEST H_SUB-H_REF LAYERS",
        "-" * 200,
    ]

    for representation in REPR_NAMES:
        report.append(f"[{representation}]")
        report.append(
            top_layer_rows(
                source_oof,
                "source_oof_accuracy",
                representation,
                None,
                n=10,
            )
        )
        report.append("")

    report += [
        "B. COCO STATE DECODABILITY: BEST LAYERS",
        "-" * 200,
    ]

    for representation in REPR_NAMES:
        for cohort in (
            "all",
            "baseline_wrong",
            "baseline_correct",
        ):
            report.append(
                f"[{representation} / {cohort}]"
            )
            report.append(
                top_layer_rows(
                    state_summary,
                    "state_accuracy",
                    representation,
                    cohort,
                    n=8,
                )
            )
            report.append("")

    report += [
        "C. IS A SINGLE BEHAVIORAL SIGN PER SAMPLE x LAYER WELL-DEFINED?",
        "-" * 200,
        purity_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        "",
        (
            "If mean/fraction purity is low, token-level causal updates inside "
            "one layer are behaviorally mixed. A single h_sub-h_ref layer sign "
            "cannot perfectly predict all positions by construction."
        ),
        "",
        "D. SPATIAL SIGN -> INDIVIDUAL CAUSAL-TOKEN B_REAL SIGN",
        "-" * 200,
    ]

    focus = sign_summary[
        sign_summary["cohort"].isin(
            [
                "all",
                "baseline_wrong",
                "baseline_correct",
                f"baseline_wrong_high_purity_ge_{args.purity_threshold:g}",
            ]
        )
    ].copy()

    report.append(
        focus.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
        if len(focus)
        else "EMPTY"
    )
    report.append("")

    report += [
        "E. SPATIAL SIGN -> NET SAMPLE x LAYER sign(sum_p B_REAL)",
        "-" * 200,
    ]

    focus_net = layer_net_summary[
        layer_net_summary["cohort"].isin(
            [
                "all",
                "baseline_wrong",
                "baseline_correct",
                f"baseline_wrong_high_purity_ge_{args.purity_threshold:g}",
            ]
        )
    ].copy()

    report.append(
        focus_net.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
        if len(focus_net)
        else "EMPTY"
    )
    report.append("")

    report += [
        "Reading guide:",
        "  1) First inspect Synthetic OOF. If h_sub-h_ref is not spatially",
        "     decodable on Synthetic at a layer, do not interpret its COCO sign.",
        "  2) Then inspect COCO baseline-wrong state_accuracy. High accuracy means",
        "     the GT spatial relation is still present in the final residual-stream",
        "     object relation state even when generation is wrong.",
        "  3) For positive/negative, SAME_BASIS_MARGIN_SHIFT is the main diagnostic:",
        "       >0 : block transition makes h_sub-h_ref more GT-supporting",
        "       <0 : block transition makes h_sub-h_ref less GT-supporting",
        "     under one fixed layer-L Synthetic codebook.",
        "  4) DELTA_FOURWAY / DELTA_AXIS ask the stricter question whether Δr itself",
        "     geometrically resembles the GT relation direction.",
        "  5) Compare spatial sign first against NET layer B, then token-level B.",
        "     A layer-level state cannot explain position-specific mixed signs if",
        "     oracle layer purity is low.",
        "  6) residual = (REAL h_sub-h_ref) - (NoImage h_sub-h_ref) is the closest",
        "     analogue of the existing Direction-Head residual definition.",
        "",
        "Causal caveat:",
        "  This remains a diagnostic relationship. Synthetic directions are frozen",
        "  source-only, but GT is used to label whether a spatial update is positive",
        "  or negative. No generation/intervention is performed here.",
    ]

    report_text = "\n".join(report) + "\n"
    print(report_text)

    (outdir / "analysis_summary.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "prompt_template": args.prompt_template,
        "pool": args.pool,
        "synthetic_labels": str(labels_path),
        "source_cache": str(source_cache),
        "target_cache": str(target_cache),
        "prior_real_update_dir": str(prior_dir),
        "prior_update_csv": str(update_path),
        "prior_generation_csv": str(gen_path),
        "N_source": int(len(source_y)),
        "N_target": int(len(target_y)),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "analysis_layers": analysis_layers,
        "cv_folds": int(args.cv_folds),
        "cv_seed": int(args.cv_seed),
        "behavior_threshold": float(args.behavior_threshold),
        "spatial_threshold": float(args.spatial_threshold),
        "purity_threshold": float(args.purity_threshold),
        "representations": {
            "img": "h_REAL(sub)-h_REAL(ref)",
            "no_image": "h_NoImage(sub)-h_NoImage(ref)",
            "residual": (
                "[h_REAL(sub)-h_REAL(ref)] - "
                "[h_NoImage(sub)-h_NoImage(ref)]"
            ),
        },
        "source_codebook": (
            "center=source mean; relation direction="
            "normalize(source relation mean-center); cosine classification; "
            "same rule as Synthetic-400 Direction-Head selector"
        ),
        "primary_spatial_sign": (
            "same_basis_margin_shift = "
            "GT four-way spatial margin after block - before block, "
            "both evaluated in layer-L frozen Synthetic codebook"
        ),
        "generation_performed": False,
        "intervention_performed": False,
        "uses_target_GT_for_sign_evaluation": True,
    }

    (outdir / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
