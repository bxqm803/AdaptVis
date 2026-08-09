#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt-selected information analysis for RAW A / B / last-token states on COCO-two.

Research question
-----------------
Keep the original spatial question fixed as the target computation:

    Where is the {subject} in relation to the {reference}?
    Answer with left, right, above, or below.

At one fixed decoder layer L*, study THREE naturally occurring states separately:

    raw_A, raw_B, raw_last

without first collapsing A and B by subtraction.

L* is selected ONCE from the original-prompt object-difference direction accuracy
(h_A - h_B), unless --fixed-layer is supplied.  Every subsequent analysis uses
this same L*.

Prompt intervention idea
------------------------
Use separate selector prompts as information readout interventions while keeping
THE SAME IMAGE fixed.  Every selector prompt ends with the same matched probe
suffix, so its probed A/B occurrences are structurally comparable:

    Objects: the {subject} and the {reference}.
    Probe A: the {subject}. Probe B: the {reference}.

Selectors:
    neutral       : generic visual inspection
    loc_a         : focus on A location
    loc_b         : focus on B location
    horizontal    : focus on A-vs-B horizontal relation
    vertical      : focus on A-vs-B vertical relation
    full          : focus on full spatial relation (positive control)
    appearance_a  : focus on A visual appearance
    appearance_b  : focus on B visual appearance

For every prompt P and every slot S in {A, B, last}, extract image-conditioned
state

    Z[P,S] = h_img[P,S] - h_noimg[P,S]

and define prompt-selected increment relative to the neutral selector:

    Delta[P,S] = Z[P,S] - Z[neutral,S]

The RAW target state is

    F[S] = h_img[raw,S] - h_noimg[raw,S]

Main diagnostics
----------------
1) Alignment: does a prompt-selected component already resemble information in
   the raw token?

       cos(F[S], Delta[P,S])

   Also report centered cosine and matched-minus-shuffled cosine to suppress
   generic prompt/image similarity.

2) Reconstruction/composition: can raw token information be reconstructed from
   a neutral visual base plus several selector increments?

       F[S] ?~= alpha0 * Z[neutral,S]
               + alpha1 * Delta[loc_a,S]
               + alpha2 * Delta[loc_b,S]
               + alpha3 * Delta[horizontal,S]
               + alpha4 * Delta[vertical,S]
               + alpha5 * Delta[appearance_a,S]
               + alpha6 * Delta[appearance_b,S]

   Report:
     - zero-training equal-sum cosine
     - gain over best single component
     - held-out global-scalar fit cosine / relative error / R2
     - per-sample oracle span cosine (geometric upper bound; NOT predictive)
     - matched-vs-shuffled reconstruction cosine

Important interpretation
------------------------
High cosine alone does NOT prove a semantic decomposition.  Stronger evidence is:
  * matched > shuffled,
  * combination > best single component,
  * held-out fixed scalar weights generalize,
  * semantically expected selectors are strongest for expected raw slots.

Outputs
-------
  baseline_layer_scan.csv
  same_slot_alignment.csv
  cross_slot_alignment.csv
  reconstruction_by_target_slot.csv
  reconstruction_weights.csv
  summary.json
  states/*.npz

This script imports:
    extract_two_object_relation_states as base
Run it from the AdaptVis repository root (or otherwise make that module importable).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base

EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_ALIASES = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "top": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "bottom": "below",
}
SLOTS = ("A", "B", "last")
SELECTORS = (
    "neutral",
    "loc_a",
    "loc_b",
    "horizontal",
    "vertical",
    "full",
    "appearance_a",
    "appearance_b",
)
# Do NOT use full as a decomposition component; it is too close to the raw task.
RECON_SELECTORS = (
    "loc_a",
    "loc_b",
    "horizontal",
    "vertical",
    "appearance_a",
    "appearance_b",
)


def norm_relation(x: Any) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(key, key)


def raw_prompt(subject: str, reference: str) -> str:
    return (
        f"Where is the {subject} in relation to the {reference}? "
        "Answer with left, right, above, or below."
    )


def selector_prefix(selector: str) -> str:
    if selector == "neutral":
        return "Inspect the image and consider the two named objects."
    if selector == "loc_a":
        return "Focus only on where object A is located in the image."
    if selector == "loc_b":
        return "Focus only on where object B is located in the image."
    if selector == "horizontal":
        return "Focus only on the horizontal relation of object A to object B in the image."
    if selector == "vertical":
        return "Focus only on the vertical relation of object A to object B in the image."
    if selector == "full":
        return "Focus on the full spatial relation of object A to object B in the image."
    if selector == "appearance_a":
        return "Focus only on the visual appearance of object A in the image."
    if selector == "appearance_b":
        return "Focus only on the visual appearance of object B in the image."
    raise ValueError(selector)


def selector_prompt(selector: str, subject: str, reference: str) -> str:
    # The final probe suffix is IDENTICAL across all selector conditions.
    return (
        f"{selector_prefix(selector)} "
        f"Objects: the {subject} and the {reference}. "
        f"Probe A: the {subject}. Probe B: the {reference}."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", required=True, choices=sorted(base.SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=["sdpa", "eager", "flash_attention_2", "none"], default="sdpa")
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--shuffle-repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--fixed-layer",
        type=int,
        default=None,
        help="Use this decoder block. If omitted, select L* once from raw-prompt h_A-h_B accuracy.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument(
        "--scalar-ridge",
        type=float,
        default=1e-6,
        help="Relative ridge strength for global scalar reconstruction weights.",
    )
    return p.parse_args()


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_chat_prompt(processor: Any, text: str, *, with_image: bool) -> str:
    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def process_inputs(
    processor: Any,
    rendered: str,
    image: Optional[Image.Image],
    device: torch.device,
) -> Dict[str, Any]:
    if image is None:
        batch = processor(text=[rendered], return_tensors="pt")
    else:
        batch = processor(text=[rendered], images=[image], return_tensors="pt")
    return move_batch(batch, device)


def atomic_save_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), EPS)


def row_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sum(normalize_rows(A) * normalize_rows(B), axis=-1)


def relative_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=-1) / np.maximum(np.linalg.norm(target, axis=-1), EPS)


def global_r2(pred: np.ndarray, target: np.ndarray) -> float:
    # Conventional test-set R2 around the target test mean.
    mu = target.mean(axis=0, keepdims=True)
    sse = float(np.sum((target - pred) ** 2))
    sst = float(np.sum((target - mu) ** 2))
    return 1.0 - sse / max(sst, EPS)


def make_splits(n: int, ratio: float, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for rep in range(repeats):
        ids = list(range(n))
        random.Random(seed + rep).shuffle(ids)
        n_train = int(n * ratio)
        if n_train <= 0 or n_train >= n:
            raise RuntimeError(f"bad train size={n_train} for n={n}")
        out.append((
            np.asarray(ids[:n_train], dtype=np.int64),
            np.asarray(ids[n_train:], dtype=np.int64),
        ))
    return out


def fit_codebook(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = X.mean(axis=0)
    Xc = X - center
    dirs: List[np.ndarray] = []
    for rel in RELATIONS:
        mask = y == rel
        if int(mask.sum()) == 0:
            raise RuntimeError(f"training split has no samples for relation={rel}")
        dirs.append(normalize(Xc[mask].mean(axis=0)))
    return center, np.stack(dirs, axis=0)


def eval_direction_fixed(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    gt_all = np.asarray([RELATIONS.index(str(v)) for v in y], dtype=np.int64)
    accs: List[float] = []
    for tr, te in splits:
        center, dirs = fit_codebook(X[tr], y[tr])
        scores = normalize_rows(X[te] - center) @ dirs.T
        pred = np.argmax(scores, axis=1)
        accs.append(float(np.mean(pred == gt_all[te])))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))}


def _stack_all_layers(states: Sequence[torch.Tensor], token_index: int, dtype_np: np.dtype) -> np.ndarray:
    return np.stack([
        states[k + 1][0, token_index].detach().float().cpu().numpy()
        for k in range(len(states) - 1)
    ], axis=0).astype(dtype_np)


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------


def extract_raw_all_layers(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[base.Record],
    with_image: bool,
    out_path: Path,
) -> None:
    mode = "correct" if with_image else "no_image"
    if out_path.exists() and not args.overwrite:
        print(f"[reuse] {out_path}")
        return
    if out_path.exists():
        out_path.unlink()

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    sids: List[int] = []
    image_ids: List[str] = []
    subjects: List[str] = []
    references: List[str] = []
    labels: List[str] = []
    apos: List[int] = []
    bpos: List[int] = []
    lastpos: List[int] = []
    Avecs: List[np.ndarray] = []
    Bvecs: List[np.ndarray] = []
    Lvecs: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    blocks_n: Optional[int] = None
    hidden_size: Optional[int] = None

    def save_progress() -> None:
        if not Avecs or blocks_n is None or hidden_size is None:
            return
        atomic_save_npz(out_path, {
            "metadata_json": np.array(json.dumps({
                "model": args.model,
                "prompt_type": "raw",
                "prompt_template": raw_prompt("{subject}", "{reference}"),
                "vision_mode": mode,
                "A_probe": "last_subtoken_of_subject_in_raw_question",
                "B_probe": "last_subtoken_of_reference_in_raw_question",
                "last_probe": "last_nonpadding_generation_boundary_token",
                "decoder_blocks": blocks_n,
                "hidden_size": hidden_size,
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "image_id": np.asarray(image_ids, dtype=object),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "relation": np.asarray(labels, dtype=object),
            "A_position": np.asarray(apos, dtype=np.int64),
            "B_position": np.asarray(bpos, dtype=np.int64),
            "last_position": np.asarray(lastpos, dtype=np.int64),
            "decoder_block_index": np.arange(blocks_n, dtype=np.int32),
            "A_vectors": np.stack(Avecs).astype(dtype_np),
            "B_vectors": np.stack(Bvecs).astype(dtype_np),
            "last_vectors": np.stack(Lvecs).astype(dtype_np),
        })

    desc = f"{args.model}:raw:{mode}:all-layers"
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB") if with_image else None
            rendered = build_chat_prompt(
                processor,
                raw_prompt(rec.subject, rec.reference),
                with_image=with_image,
            )
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()
            aidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.subject)
            bidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.reference)
            if aidx == bidx:
                raise RuntimeError("subject/reference token positions collide")
            if "attention_mask" in batch:
                lidx = int(batch["attention_mask"][0].sum().item()) - 1
            else:
                lidx = len(input_ids) - 1

            with torch.inference_mode():
                outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
                states = base.hidden_tuple(outputs)

            final = states[-1]
            if final.ndim != 3 or final.shape[0] != 1:
                raise RuntimeError(f"unexpected hidden shape {tuple(final.shape)}")
            if int(final.shape[1]) != len(input_ids):
                raise RuntimeError(f"token/hidden length mismatch {len(input_ids)} != {final.shape[1]}")
            cur_blocks = len(states) - 1
            if blocks_n is None:
                blocks_n = cur_blocks
                hidden_size = int(final.shape[-1])
                print(f"[{desc}] decoder_blocks={blocks_n}, hidden={hidden_size}")
            elif cur_blocks != blocks_n:
                raise RuntimeError(f"decoder blocks changed {blocks_n}->{cur_blocks}")

            Avec = _stack_all_layers(states, int(aidx), dtype_np)
            Bvec = _stack_all_layers(states, int(bidx), dtype_np)
            Lvec = _stack_all_layers(states, int(lidx), dtype_np)

            sids.append(int(rec.sid))
            image_ids.append(str(rec.image_id))
            subjects.append(str(rec.subject))
            references.append(str(rec.reference))
            labels.append(norm_relation(rec.relation))
            apos.append(int(aidx))
            bpos.append(int(bidx))
            lastpos.append(int(lidx))
            Avecs.append(Avec)
            Bvecs.append(Bvec)
            Lvecs.append(Lvec)

            del outputs, states, batch
            if len(Avecs) % args.save_every == 0:
                save_progress()
        except Exception as exc:
            errors.append({
                "sid": int(rec.sid),
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "relation": str(rec.relation),
                "mode": mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-10:],
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_progress()
    out_path.with_suffix(".errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {out_path} | n={len(Avecs)}/{len(records)} | errors={len(errors)}")


def extract_selector_fixed_layer(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[base.Record],
    selector: str,
    with_image: bool,
    layer: int,
    out_path: Path,
) -> None:
    mode = "correct" if with_image else "no_image"
    if out_path.exists() and not args.overwrite:
        print(f"[reuse] {out_path}")
        return
    if out_path.exists():
        out_path.unlink()

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    sids: List[int] = []
    labels: List[str] = []
    apos: List[int] = []
    bpos: List[int] = []
    lastpos: List[int] = []
    Avecs: List[np.ndarray] = []
    Bvecs: List[np.ndarray] = []
    Lvecs: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    blocks_n: Optional[int] = None
    hidden_size: Optional[int] = None

    def save_progress() -> None:
        if not Avecs or blocks_n is None or hidden_size is None:
            return
        atomic_save_npz(out_path, {
            "metadata_json": np.array(json.dumps({
                "model": args.model,
                "selector": selector,
                "prompt_template": selector_prompt(selector, "{subject}", "{reference}"),
                "vision_mode": mode,
                "fixed_layer": layer,
                "A_probe": "last_subtoken_of_subject_in_shared_probe_suffix",
                "B_probe": "last_subtoken_of_reference_in_shared_probe_suffix",
                "last_probe": "last_nonpadding_generation_boundary_token",
                "decoder_blocks": blocks_n,
                "hidden_size": hidden_size,
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "relation": np.asarray(labels, dtype=object),
            "A_position": np.asarray(apos, dtype=np.int64),
            "B_position": np.asarray(bpos, dtype=np.int64),
            "last_position": np.asarray(lastpos, dtype=np.int64),
            "A_vectors": np.stack(Avecs).astype(dtype_np),
            "B_vectors": np.stack(Bvecs).astype(dtype_np),
            "last_vectors": np.stack(Lvecs).astype(dtype_np),
        })

    desc = f"{args.model}:{selector}:{mode}:L{layer}"
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB") if with_image else None
            rendered = build_chat_prompt(
                processor,
                selector_prompt(selector, rec.subject, rec.reference),
                with_image=with_image,
            )
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()
            # Because every selector repeats both objects in the identical final suffix,
            # last-match selects the matched probe occurrences.
            aidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.subject)
            bidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.reference)
            if aidx == bidx:
                raise RuntimeError("subject/reference token positions collide")
            if "attention_mask" in batch:
                lidx = int(batch["attention_mask"][0].sum().item()) - 1
            else:
                lidx = len(input_ids) - 1

            with torch.inference_mode():
                outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
                states = base.hidden_tuple(outputs)

            cur_blocks = len(states) - 1
            if layer < 0 or layer >= cur_blocks:
                raise RuntimeError(f"requested L{layer}, model has blocks 0..{cur_blocks-1}")
            final = states[-1]
            if int(final.shape[1]) != len(input_ids):
                raise RuntimeError(f"token/hidden length mismatch {len(input_ids)} != {final.shape[1]}")
            if blocks_n is None:
                blocks_n = cur_blocks
                hidden_size = int(final.shape[-1])
                print(f"[{desc}] decoder_blocks={blocks_n}, hidden={hidden_size}")

            A = states[layer + 1][0, int(aidx)].detach().float().cpu().numpy().astype(dtype_np)
            B = states[layer + 1][0, int(bidx)].detach().float().cpu().numpy().astype(dtype_np)
            L = states[layer + 1][0, int(lidx)].detach().float().cpu().numpy().astype(dtype_np)

            sids.append(int(rec.sid))
            labels.append(norm_relation(rec.relation))
            apos.append(int(aidx))
            bpos.append(int(bidx))
            lastpos.append(int(lidx))
            Avecs.append(A)
            Bvecs.append(B)
            Lvecs.append(L)

            del outputs, states, batch
            if len(Avecs) % args.save_every == 0:
                save_progress()
        except Exception as exc:
            errors.append({
                "sid": int(rec.sid),
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "relation": str(rec.relation),
                "selector": selector,
                "mode": mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-10:],
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_progress()
    out_path.with_suffix(".errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {out_path} | n={len(Avecs)}/{len(records)} | errors={len(errors)}")


# -----------------------------------------------------------------------------
# Alignment / fixed-layer selection
# -----------------------------------------------------------------------------


def align_raw_pair(
    correct: Dict[str, Any],
    noimg: Dict[str, Any],
) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, List[int]]:
    pc = {int(s): i for i, s in enumerate(correct["sample_index"].tolist())}
    pn = {int(s): i for i, s in enumerate(noimg["sample_index"].tolist())}
    common = sorted(set(pc) & set(pn))
    if not common:
        raise RuntimeError("no common raw samples between image/no-image")
    ic = np.asarray([pc[s] for s in common], dtype=np.int64)
    ino = np.asarray([pn[s] for s in common], dtype=np.int64)
    yc = np.asarray([norm_relation(x) for x in correct["relation"][ic]], dtype=object)
    yn = np.asarray([norm_relation(x) for x in noimg["relation"][ino]], dtype=object)
    if not np.array_equal(yc, yn):
        raise RuntimeError("raw image/no-image label mismatch")
    layers = [int(x) for x in correct["decoder_block_index"].tolist()]
    out_c = {
        "A": correct["A_vectors"][ic].astype(np.float32),
        "B": correct["B_vectors"][ic].astype(np.float32),
        "last": correct["last_vectors"][ic].astype(np.float32),
    }
    out_n = {
        "A": noimg["A_vectors"][ino].astype(np.float32),
        "B": noimg["B_vectors"][ino].astype(np.float32),
        "last": noimg["last_vectors"][ino].astype(np.float32),
    }
    return common, out_c, out_n, yc, layers


def select_fixed_layer(
    raw_correct: Dict[str, np.ndarray],
    y: np.ndarray,
    layers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, int, List[Tuple[np.ndarray, np.ndarray]]]:
    Xdiff = raw_correct["A"] - raw_correct["B"]
    splits = make_splits(len(y), args.train_ratio, args.repeats, args.seed)
    rows: List[Dict[str, Any]] = []
    for li, layer in enumerate(layers):
        m = eval_direction_fixed(Xdiff[:, li, :].astype(np.float64), y, splits)
        rows.append({"layer": int(layer), **m})
    df = pd.DataFrame(rows)
    if args.fixed_layer is None:
        best = df.loc[df["acc_mean"].idxmax()]
        fixed = int(best["layer"])
    else:
        fixed = int(args.fixed_layer)
        if fixed not in layers:
            raise RuntimeError(f"--fixed-layer L{fixed} absent; available {layers[0]}..{layers[-1]}")
    return df, fixed, splits


def align_all_fixed_states(
    raw_sids: Sequence[int],
    raw_correct: Dict[str, np.ndarray],
    raw_noimg: Dict[str, np.ndarray],
    raw_y: np.ndarray,
    layers: Sequence[int],
    fixed_layer: int,
    selector_data: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, Dict[str, Dict[str, np.ndarray]]], np.ndarray]:
    raw_pos = {int(s): i for i, s in enumerate(raw_sids)}
    sid_sets = [set(raw_pos)]
    spos: Dict[Tuple[str, str], Dict[int, int]] = {}
    for key, data in selector_data.items():
        p = {int(s): i for i, s in enumerate(data["sample_index"].tolist())}
        spos[key] = p
        sid_sets.append(set(p))
    common = sorted(set.intersection(*sid_sets))
    if not common:
        raise RuntimeError("no common samples across raw + selector conditions")

    li = list(layers).index(int(fixed_layer))
    ridx = np.asarray([raw_pos[s] for s in common], dtype=np.int64)
    raw: Dict[str, np.ndarray] = {}
    for slot in SLOTS:
        raw[f"{slot}_correct"] = raw_correct[slot][ridx, li, :].astype(np.float32)
        raw[f"{slot}_noimg"] = raw_noimg[slot][ridx, li, :].astype(np.float32)
        raw[f"{slot}_residual"] = raw[f"{slot}_correct"] - raw[f"{slot}_noimg"]
    y = raw_y[ridx]

    sel: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for selector in SELECTORS:
        sel[selector] = {}
        for mode in ("correct", "no_image"):
            data = selector_data[(selector, mode)]
            idx = np.asarray([spos[(selector, mode)][s] for s in common], dtype=np.int64)
            yy = np.asarray([norm_relation(x) for x in data["relation"][idx]], dtype=object)
            if not np.array_equal(y, yy):
                raise RuntimeError(f"label mismatch for selector={selector}, mode={mode}")
            sel[selector][mode] = {
                "A": data["A_vectors"][idx].astype(np.float32),
                "B": data["B_vectors"][idx].astype(np.float32),
                "last": data["last_vectors"][idx].astype(np.float32),
            }
        sel[selector]["residual"] = {
            slot: sel[selector]["correct"][slot] - sel[selector]["no_image"][slot]
            for slot in SLOTS
        }

    return common, raw, sel, y


# -----------------------------------------------------------------------------
# Alignment metrics
# -----------------------------------------------------------------------------


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    base_idx = np.arange(n)
    for _ in range(200):
        p = rng.permutation(n)
        if np.all(p != base_idx):
            return p
    # Deterministic guaranteed derangement for n>1.
    if n <= 1:
        return base_idx
    shift = int(rng.integers(1, n))
    return np.roll(base_idx, shift)


def alignment_metrics(
    target: np.ndarray,
    component: np.ndarray,
    *,
    seed: int,
    shuffle_repeats: int,
) -> Dict[str, float]:
    raw_cos = row_cos(target, component)
    tc = target - target.mean(axis=0, keepdims=True)
    cc = component - component.mean(axis=0, keepdims=True)
    centered = row_cos(tc, cc)

    rng = np.random.default_rng(seed)
    shuf_vals: List[float] = []
    for _ in range(shuffle_repeats):
        perm = random_derangement(len(target), rng)
        shuf_vals.append(float(np.mean(row_cos(target, component[perm]))))
    shuf_mean = float(np.mean(shuf_vals))
    return {
        "cos_mean": float(np.mean(raw_cos)),
        "cos_std": float(np.std(raw_cos)),
        "centered_cos_mean": float(np.mean(centered)),
        "centered_cos_std": float(np.std(centered)),
        "shuffled_cos_mean": shuf_mean,
        "matched_minus_shuffled": float(np.mean(raw_cos)) - shuf_mean,
        "component_mean_norm": float(np.mean(np.linalg.norm(component, axis=-1))),
        "target_mean_norm": float(np.mean(np.linalg.norm(target, axis=-1))),
    }


def make_prompt_components(
    sel: Dict[str, Dict[str, Dict[str, np.ndarray]]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return same-slot prompt-selected visual components.

    neutral[S] is a visual base.
    delta_<P>[S] = residual(P,S) - residual(neutral,S).
    """
    out: Dict[str, Dict[str, np.ndarray]] = {slot: {} for slot in SLOTS}
    for slot in SLOTS:
        neutral = sel["neutral"]["residual"][slot]
        out[slot]["neutral"] = neutral
        for selector in SELECTORS:
            if selector == "neutral":
                continue
            out[slot][selector] = sel[selector]["residual"][slot] - neutral
    return out


# -----------------------------------------------------------------------------
# Reconstruction / composition
# -----------------------------------------------------------------------------


def stack_components(component_map: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    # N x K x D
    return np.stack([component_map[n] for n in names], axis=1).astype(np.float64)


def fit_global_scalar_weights(
    C: np.ndarray,
    F: np.ndarray,
    train_idx: np.ndarray,
    ridge_relative: float,
) -> np.ndarray:
    """Fit K global scalar weights using all train samples and dimensions.

    Minimize sum_i || sum_k w_k C[i,k] - F[i] ||^2.
    This is a tiny KxK linear system, not a high-dimensional probe.
    """
    Ct = C[train_idx]
    Ft = F[train_idx]
    # Gram[k,l] = sum_{i,d} C[i,k,d] C[i,l,d]
    gram = np.einsum("nkd,nld->kl", Ct, Ct, optimize=True)
    rhs = np.einsum("nkd,nd->k", Ct, Ft, optimize=True)
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    lam = ridge_relative * max(scale, EPS)
    return np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)


def predict_with_weights(C: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("nkd,k->nd", C, w, optimize=True)


def oracle_span_prediction(C: np.ndarray, F: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Per-sample target-dependent span projection. Geometric upper bound only."""
    n, k, _ = C.shape
    pred = np.empty_like(F, dtype=np.float64)
    eye = np.eye(k)
    for i in range(n):
        Ci = C[i]
        gram = Ci @ Ci.T
        rhs = Ci @ F[i]
        scale = float(np.trace(gram) / max(k, 1))
        w = np.linalg.solve(gram + ridge * max(scale, EPS) * eye, rhs)
        pred[i] = w @ Ci
    return pred


def reconstruction_metrics(
    target: np.ndarray,
    component_map: Dict[str, np.ndarray],
    component_names: Sequence[str],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    ridge_relative: float,
    seed: int,
    shuffle_repeats: int,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    F = target.astype(np.float64)
    C = stack_components(component_map, component_names)

    # Zero-training equal sum.
    equal_pred = C.sum(axis=1)
    equal_cos = row_cos(equal_pred, F)
    equal_rel = relative_error(equal_pred, F)

    # Best single component cosine, sample-wise averaged per component then max.
    single_cos_by_name: Dict[str, float] = {}
    for k, name in enumerate(component_names):
        single_cos_by_name[name] = float(np.mean(row_cos(C[:, k], F)))
    best_single_name = max(single_cos_by_name, key=single_cos_by_name.get)
    best_single_cos = single_cos_by_name[best_single_name]

    # Held-out global scalar weights.
    fit_cos: List[float] = []
    fit_rel: List[float] = []
    fit_r2s: List[float] = []
    weight_rows: List[np.ndarray] = []
    for tr, te in splits:
        w = fit_global_scalar_weights(C, F, tr, ridge_relative)
        pred = predict_with_weights(C[te], w)
        fit_cos.append(float(np.mean(row_cos(pred, F[te]))))
        fit_rel.append(float(np.mean(relative_error(pred, F[te]))))
        fit_r2s.append(float(global_r2(pred, F[te])))
        weight_rows.append(w)
    W = np.stack(weight_rows, axis=0)

    # Oracle geometric upper bound.
    oracle = oracle_span_prediction(C, F)
    oracle_cos = row_cos(oracle, F)
    oracle_rel = relative_error(oracle, F)

    # Matched-vs-shuffled equal-sum reconstruction.
    rng = np.random.default_rng(seed)
    shuf_cos_vals: List[float] = []
    n = len(F)
    for _ in range(shuffle_repeats):
        Cs = np.empty_like(C)
        for k in range(C.shape[1]):
            perm = random_derangement(n, rng)
            Cs[:, k] = C[perm, k]
        shuf_cos_vals.append(float(np.mean(row_cos(Cs.sum(axis=1), F))))
    shuf_mean = float(np.mean(shuf_cos_vals))

    metrics: Dict[str, float] = {
        "equal_sum_cos": float(np.mean(equal_cos)),
        "equal_sum_cos_std": float(np.std(equal_cos)),
        "equal_sum_relative_error": float(np.mean(equal_rel)),
        "best_single_cos": float(best_single_cos),
        "composition_gain_over_best_single": float(np.mean(equal_cos)) - float(best_single_cos),
        "heldout_scalar_cos_mean": float(np.mean(fit_cos)),
        "heldout_scalar_cos_std": float(np.std(fit_cos)),
        "heldout_scalar_relative_error_mean": float(np.mean(fit_rel)),
        "heldout_scalar_relative_error_std": float(np.std(fit_rel)),
        "heldout_scalar_r2_mean": float(np.mean(fit_r2s)),
        "heldout_scalar_r2_std": float(np.std(fit_r2s)),
        "oracle_span_cos": float(np.mean(oracle_cos)),
        "oracle_span_relative_error": float(np.mean(oracle_rel)),
        "equal_sum_shuffled_cos": shuf_mean,
        "equal_sum_matched_minus_shuffled": float(np.mean(equal_cos)) - shuf_mean,
    }
    # Store best-single name outside numeric CSV field via a sentinel key handled by caller.
    metrics["best_single_index"] = float(list(component_names).index(best_single_name))

    weights = {
        name: (float(W[:, k].mean()), float(W[:, k].std()))
        for k, name in enumerate(component_names)
    }
    return metrics, weights


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0,1)")
    if args.shuffle_repeats < 1:
        raise ValueError("--shuffle-repeats must be >=1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    state_dir = out_dir / "states"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    records, audit = base.load_records(args.dataset, Path(args.data_root), args.max_samples)
    records = [r for r in records if norm_relation(r.relation) in RELATIONS]
    if not records:
        raise RuntimeError("no usable records")
    print(f"[{args.dataset}] n={len(records)} counts={dict(Counter(norm_relation(r.relation) for r in records))}")
    (out_dir / "dataset.audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    spec = base.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    started = time.time()
    model = None
    processor = None
    try:
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        # ------------------------------------------------------------------
        # 1) RAW target computation, all layers, image + no-image.
        # ------------------------------------------------------------------
        print("\n" + "=" * 112)
        print("RAW TARGET PROMPT")
        print(raw_prompt("A", "B"))
        print("=" * 112)
        raw_paths = {
            "correct": state_dir / "raw__correct__all_layers.npz",
            "no_image": state_dir / "raw__no_image__all_layers.npz",
        }
        extract_raw_all_layers(
            args=args,
            model=model,
            processor=processor,
            device=device,
            records=records,
            with_image=True,
            out_path=raw_paths["correct"],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        extract_raw_all_layers(
            args=args,
            model=model,
            processor=processor,
            device=device,
            records=records,
            with_image=False,
            out_path=raw_paths["no_image"],
        )
        raw_correct_npz = load_npz(raw_paths["correct"])
        raw_noimg_npz = load_npz(raw_paths["no_image"])
        raw_sids, raw_correct, raw_noimg, raw_y, layers = align_raw_pair(
            raw_correct_npz, raw_noimg_npz
        )

        scan_df, fixed_layer, base_splits = select_fixed_layer(
            raw_correct, raw_y, layers, args
        )
        scan_df.to_csv(out_dir / "baseline_layer_scan.csv", index=False)
        fixed_row = scan_df.loc[scan_df["layer"] == fixed_layer].iloc[0]
        best_row = scan_df.loc[scan_df["acc_mean"].idxmax()]
        print("\n" + "=" * 112)
        print("FIXED LAYER SELECTION")
        print("=" * 112)
        print(
            f"raw h_A-h_B best layer : L{int(best_row['layer'])} "
            f"acc={100*float(best_row['acc_mean']):.2f}%±{100*float(best_row['acc_std']):.2f}%"
        )
        print(
            f"analysis fixed layer   : L{fixed_layer} "
            f"acc={100*float(fixed_row['acc_mean']):.2f}%±{100*float(fixed_row['acc_std']):.2f}%"
        )

        # ------------------------------------------------------------------
        # 2) Selector prompts, fixed L* only.
        # ------------------------------------------------------------------
        selector_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for selector in SELECTORS:
            print("\n" + "-" * 112)
            print(f"SELECTOR: {selector}")
            print(selector_prompt(selector, "A", "B"))
            print("-" * 112)
            for mode, with_image in (("correct", True), ("no_image", False)):
                path = state_dir / f"selector__{selector}__{mode}__L{fixed_layer}.npz"
                extract_selector_fixed_layer(
                    args=args,
                    model=model,
                    processor=processor,
                    device=device,
                    records=records,
                    selector=selector,
                    with_image=with_image,
                    layer=fixed_layer,
                    out_path=path,
                )
                selector_data[(selector, mode)] = load_npz(path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        common_sids, raw_fixed, sel_fixed, y = align_all_fixed_states(
            raw_sids,
            raw_correct,
            raw_noimg,
            raw_y,
            layers,
            fixed_layer,
            selector_data,
        )
        print(f"\n[common] n={len(common_sids)} across raw + all selector image/no-image conditions")
        splits = make_splits(len(common_sids), args.train_ratio, args.repeats, args.seed)
        components = make_prompt_components(sel_fixed)

        # ------------------------------------------------------------------
        # 3) Alignment: raw token vs prompt-selected visual increments.
        # ------------------------------------------------------------------
        cross_rows: List[Dict[str, Any]] = []
        same_rows: List[Dict[str, Any]] = []
        for target_slot in SLOTS:
            target = raw_fixed[f"{target_slot}_residual"]
            for selector in SELECTORS:
                for source_slot in SLOTS:
                    if selector == "neutral":
                        comp = components[source_slot]["neutral"]
                        component_kind = "neutral_visual_base"
                    else:
                        comp = components[source_slot][selector]
                        component_kind = "selector_increment"
                    m = alignment_metrics(
                        target,
                        comp,
                        seed=args.seed + 1000 * SLOTS.index(target_slot)
                        + 100 * SELECTORS.index(selector)
                        + 10 * SLOTS.index(source_slot),
                        shuffle_repeats=args.shuffle_repeats,
                    )
                    row = {
                        "fixed_layer": fixed_layer,
                        "target_raw_slot": target_slot,
                        "selector": selector,
                        "selector_source_slot": source_slot,
                        "component_kind": component_kind,
                        **m,
                    }
                    cross_rows.append(row)
                    if source_slot == target_slot:
                        same_rows.append(row.copy())

        cross_df = pd.DataFrame(cross_rows)
        same_df = pd.DataFrame(same_rows)
        cross_df.to_csv(out_dir / "cross_slot_alignment.csv", index=False)
        same_df.to_csv(out_dir / "same_slot_alignment.csv", index=False)

        print("\n" + "=" * 112)
        print(f"RAW TOKEN vs PROMPT-SELECTED INFORMATION @ FIXED L{fixed_layer}")
        print("same-slot comparison; values are matched cosine / centered cosine / matched-shuffled")
        print("=" * 112)
        for slot in SLOTS:
            print(f"\nRAW {slot} image-conditioned state")
            sub = same_df[same_df["target_raw_slot"] == slot].copy()
            # Print neutral first, then selector increments sorted by matched-shuffled.
            neutral_row = sub[sub["selector"] == "neutral"]
            if len(neutral_row):
                r = neutral_row.iloc[0]
                print(
                    f"  {'neutral(base)':<16s} cos={r['cos_mean']:+.4f} | "
                    f"centered={r['centered_cos_mean']:+.4f} | "
                    f"match-shuf={r['matched_minus_shuffled']:+.4f}"
                )
            rest = sub[sub["selector"] != "neutral"].sort_values(
                "matched_minus_shuffled", ascending=False
            )
            for _, r in rest.iterrows():
                print(
                    f"  {str(r['selector']):<16s} cos={float(r['cos_mean']):+.4f} | "
                    f"centered={float(r['centered_cos_mean']):+.4f} | "
                    f"match-shuf={float(r['matched_minus_shuffled']):+.4f}"
                )

        # ------------------------------------------------------------------
        # 4) Reconstruction / composition of each raw slot.
        # ------------------------------------------------------------------
        recon_component_names = ("neutral",) + RECON_SELECTORS
        recon_rows: List[Dict[str, Any]] = []
        weight_rows: List[Dict[str, Any]] = []
        for slot in SLOTS:
            target = raw_fixed[f"{slot}_residual"]
            cmap = components[slot]
            metrics, weights = reconstruction_metrics(
                target,
                cmap,
                recon_component_names,
                splits,
                ridge_relative=args.scalar_ridge,
                seed=args.seed + 9000 + SLOTS.index(slot),
                shuffle_repeats=args.shuffle_repeats,
            )
            best_idx = int(round(metrics.pop("best_single_index")))
            best_name = recon_component_names[best_idx]
            recon_rows.append({
                "fixed_layer": fixed_layer,
                "target_raw_slot": slot,
                "component_names": "+".join(recon_component_names),
                "best_single_component": best_name,
                **metrics,
            })
            for name, (wm, ws) in weights.items():
                weight_rows.append({
                    "fixed_layer": fixed_layer,
                    "target_raw_slot": slot,
                    "component": name,
                    "weight_mean": wm,
                    "weight_std": ws,
                })

        recon_df = pd.DataFrame(recon_rows)
        weights_df = pd.DataFrame(weight_rows)
        recon_df.to_csv(out_dir / "reconstruction_by_target_slot.csv", index=False)
        weights_df.to_csv(out_dir / "reconstruction_weights.csv", index=False)

        print("\n" + "=" * 112)
        print(f"RAW TOKEN RECONSTRUCTION FROM PROMPT-SELECTED COMPONENTS @ L{fixed_layer}")
        print("components: neutral + loc_a + loc_b + horizontal + vertical + appearance_a + appearance_b")
        print("full selector is excluded from reconstruction and retained only as a positive-control alignment")
        print("=" * 112)
        for _, r in recon_df.iterrows():
            slot = str(r["target_raw_slot"])
            print(f"\nRAW {slot}")
            print(
                f"  equal sum             : cos={float(r['equal_sum_cos']):.4f} | "
                f"relerr={float(r['equal_sum_relative_error']):.4f}"
            )
            print(
                f"  best single           : {r['best_single_component']} "
                f"cos={float(r['best_single_cos']):.4f}"
            )
            print(
                f"  composition gain      : {float(r['composition_gain_over_best_single']):+.4f}"
            )
            print(
                f"  heldout scalar fit    : cos={float(r['heldout_scalar_cos_mean']):.4f}"
                f"±{float(r['heldout_scalar_cos_std']):.4f} | "
                f"relerr={float(r['heldout_scalar_relative_error_mean']):.4f} | "
                f"R2={float(r['heldout_scalar_r2_mean']):.4f}"
            )
            print(
                f"  oracle span upper bd  : cos={float(r['oracle_span_cos']):.4f} | "
                f"relerr={float(r['oracle_span_relative_error']):.4f}"
            )
            print(
                f"  matched - shuffled    : {float(r['equal_sum_matched_minus_shuffled']):+.4f}"
            )
            ws = weights_df[weights_df["target_raw_slot"] == slot]
            print("  heldout-fit weights   : " + ", ".join(
                f"{rr['component']}={float(rr['weight_mean']):+.3f}±{float(rr['weight_std']):.3f}"
                for _, rr in ws.iterrows()
            ))

        # Optional semantic probe: raw A/B/last residual relation ACC at fixed L.
        # This is auxiliary only; reconstruction cosine remains the main test.
        aux_rows: List[Dict[str, Any]] = []
        for slot in SLOTS:
            m = eval_direction_fixed(raw_fixed[f"{slot}_residual"].astype(np.float64), y, splits)
            aux_rows.append({"slot": slot, **m})
        aux_df = pd.DataFrame(aux_rows)
        aux_df.to_csv(out_dir / "aux_raw_slot_relation_accuracy.csv", index=False)

        print("\n" + "=" * 112)
        print("AUXILIARY ONLY: RELATION ACC OF RAW IMAGE-CONDITIONED TOKEN STATES")
        print("=" * 112)
        for _, r in aux_df.iterrows():
            print(
                f"raw {r['slot']:<4s}: {100*float(r['acc_mean']):.2f}%"
                f"±{100*float(r['acc_std']):.2f}%"
            )

        # Summary JSON.
        summary = {
            "config": {
                "model": args.model,
                "repo_id": spec.repo_id,
                "dataset": args.dataset,
                "n_common": len(common_sids),
                "fixed_layer": fixed_layer,
                "layer_selection": "raw h_A-h_B relation direction ACC" if args.fixed_layer is None else "user fixed",
                "raw_prompt": raw_prompt("{subject}", "{reference}"),
                "selector_prompts": {
                    s: selector_prompt(s, "{subject}", "{reference}") for s in SELECTORS
                },
                "raw_target": "h_img(raw,slot)-h_noimg(raw,slot)",
                "selector_residual": "h_img(selector,slot)-h_noimg(selector,slot)",
                "selector_increment": "selector_residual-neutral_residual",
                "reconstruction_components": list(recon_component_names),
                "train_ratio": args.train_ratio,
                "repeats": args.repeats,
                "shuffle_repeats": args.shuffle_repeats,
                "seed": args.seed,
            },
            "baseline_best": {
                "layer": int(best_row["layer"]),
                "acc_mean": float(best_row["acc_mean"]),
                "acc_std": float(best_row["acc_std"]),
            },
            "fixed_layer_baseline": {
                "layer": fixed_layer,
                "acc_mean": float(fixed_row["acc_mean"]),
                "acc_std": float(fixed_row["acc_std"]),
            },
            "reconstruction": recon_df.to_dict(orient="records"),
            "weights": weights_df.to_dict(orient="records"),
            "aux_raw_slot_relation_accuracy": aux_df.to_dict(orient="records"),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\nSaved:")
        for fn in (
            "baseline_layer_scan.csv",
            "same_slot_alignment.csv",
            "cross_slot_alignment.csv",
            "reconstruction_by_target_slot.csv",
            "reconstruction_weights.csv",
            "aux_raw_slot_relation_accuracy.csv",
            "summary.json",
        ):
            print(f"  {out_dir / fn}")
        print(f"Elapsed: {(time.time() - started)/60:.1f} min")

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
