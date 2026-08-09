#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-layer factorial decomposition of COCO-two object-relation representations.

Goal
----
1) Use the ORIGINAL prompt once to find the decoder layer L* with the highest
   held-out 4-way direction-vector accuracy:

     Where is A in relation to B?
     Answer with left, right, above, or below.

   Representation at every layer:
     r_L = h_A,L - h_B,L

2) Freeze L*.  ALL later conditions are evaluated ONLY at this same layer.

3) Run a 2 x 2 x 2 factorial design at L*:

     I = image present?          {-1=no-image, +1=image}
     T = explicit spatial task? {-1=off,      +1=on}
     O = answer options given?  {-1=off,      +1=on}

   The text conditions share the SAME final object-probe suffix, so the probed
   A/B token occurrences are structurally matched across prompt conditions.

   Shared suffix:
     Objects: A and B. Object A: A. Object B: B.

   T=1 adds:
     Determine the spatial relation of object A to object B in the image.

   O=1 adds:
     Allowed relation labels: left, right, above, below.

   I toggles only whether the image is supplied; text stays identical.

4) For each sample, decompose the eight vectors r_{i,t,o} exactly with
   Walsh/Hadamard effect coding:

     r(i,t,o) = mu
              + i*I + t*T + o*O
              + i*t*IT + i*o*IO + t*o*TO
              + i*t*o*ITO

   where i,t,o are -1/+1.

Important: the FULL 8-term decomposition is an algebraic identity, so exact
reconstruction by itself is NOT evidence of disentanglement.  The useful
questions are:

  * Which components carry spatial-relation direction accuracy?
  * How much factor-induced energy is in I/T/O vs interactions?
  * Is a MAIN-EFFECT-ONLY model already a good approximation of the
    factor-centered state, or are pairwise/3-way interactions necessary?
  * Does the image component I align with the original-prompt spatial
    direction?  Do IT/IO carry relation signal, implying prompt-dependent
    modulation of visual information?

Outputs
-------
  baseline_layer_scan.csv
  condition_accuracy_fixed_layer.csv
  image_residual_accuracy_fixed_layer.csv
  component_metrics_fixed_layer.csv
  component_cosine_matrix.csv
  additivity_fixed_layer.csv
  summary.json
  states/*.npz
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

# Effect-coded factor levels.
LEVELS = (-1, +1)
COMPONENTS = ("mu", "I", "T", "O", "IT", "IO", "TO", "ITO")


def norm_relation(x: Any) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(key, key)


def original_prompt(subject: str, reference: str) -> str:
    return (
        f"Where is the {subject} in relation to the {reference}? "
        "Answer with left, right, above, or below."
    )


def factorial_prompt(subject: str, reference: str, task_on: bool, options_on: bool) -> str:
    parts: List[str] = ["Consider object A and object B."]
    if task_on:
        parts.append("Determine the spatial relation of object A to object B in the image.")
    if options_on:
        parts.append("Allowed relation labels: left, right, above, below.")

    # IMPORTANT: this suffix is identical across all T/O conditions.
    # Both object identities appear once before the actual probe mentions,
    # reducing the causal asymmetry of probing the first mention of A/B.
    parts.append(
        f"Objects: the {subject} and the {reference}. "
        f"Object A: the {subject}. Object B: the {reference}."
    )
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", required=True, choices=sorted(base.SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=["sdpa", "eager", "flash_attention_2", "none"], default="sdpa")
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fixed-layer", type=int, default=None,
                   help="If set, use this decoder-block index instead of selecting L* from the original prompt.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-fp32", action="store_true")
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


def process_inputs(processor: Any, rendered: str, image: Optional[Image.Image], device: torch.device) -> Dict[str, Any]:
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


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), EPS)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)


def row_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sum(normalize_rows(A) * normalize_rows(B), axis=-1)


def relative_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=-1) / np.maximum(np.linalg.norm(target, axis=-1), EPS)


# -----------------------------------------------------------------------------
# Direction-vector probe
# -----------------------------------------------------------------------------


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


def eval_direction_fixed(X: np.ndarray, y: np.ndarray, splits: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    gt_all = np.asarray([RELATIONS.index(str(v)) for v in y], dtype=np.int64)
    accs: List[float] = []
    margins: List[float] = []
    for tr, te in splits:
        center, dirs = fit_codebook(X[tr], y[tr])
        Xt = X[te] - center
        Xn = normalize_rows(Xt)
        scores = Xn @ dirs.T
        pred = np.argmax(scores, axis=1)
        gt = gt_all[te]
        accs.append(float(np.mean(pred == gt)))
        true = scores[np.arange(len(gt)), gt]
        tmp = scores.copy()
        tmp[np.arange(len(gt)), gt] = -np.inf
        margins.append(float(np.mean(true - tmp.max(axis=1))))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
    }


def direction_alignment_to_baseline(
    X: np.ndarray,
    Xbase: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    """Mean cosine of same-relation train codebook directions to baseline."""
    vals: List[float] = []
    for tr, _ in splits:
        _, d = fit_codebook(X[tr], y[tr])
        _, db = fit_codebook(Xbase[tr], y[tr])
        vals.append(float(np.mean(np.sum(d * db, axis=1))))
    return float(np.mean(vals)), float(np.std(vals))


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------


def extract_original_all_layers(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[base.Record],
    out_path: Path,
) -> None:
    if out_path.exists() and not args.overwrite:
        print(f"[reuse] {out_path}")
        return
    if out_path.exists():
        out_path.unlink()

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    sids: List[int] = []
    labels: List[str] = []
    vecs: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    blocks_n: Optional[int] = None
    hidden_size: Optional[int] = None

    def save_progress() -> None:
        if not vecs or blocks_n is None or hidden_size is None:
            return
        atomic_save_npz(out_path, {
            "metadata_json": np.array(json.dumps({
                "model": args.model,
                "prompt": original_prompt("{subject}", "{reference}"),
                "vision": "correct",
                "decoder_blocks": blocks_n,
                "hidden_size": hidden_size,
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "relation": np.asarray(labels, dtype=object),
            "decoder_block_index": np.arange(blocks_n, dtype=np.int32),
            "relation_vectors": np.stack(vecs).astype(dtype_np),
        })

    desc = f"{args.model}:baseline-original:correct"
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB")
            text = original_prompt(rec.subject, rec.reference)
            rendered = build_chat_prompt(processor, text, with_image=True)
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()
            sidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.subject)
            ridx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.reference)
            if sidx == ridx:
                raise RuntimeError("subject/reference token positions collide")

            with torch.inference_mode():
                outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
                states = base.hidden_tuple(outputs)

            final = states[-1]
            if int(final.shape[1]) != len(input_ids):
                raise RuntimeError(f"token/hidden length mismatch {len(input_ids)} != {final.shape[1]}")
            cur_blocks = len(states) - 1
            if blocks_n is None:
                blocks_n = cur_blocks
                hidden_size = int(final.shape[-1])
                print(f"[{desc}] decoder_blocks={blocks_n}, hidden={hidden_size}")
            elif cur_blocks != blocks_n:
                raise RuntimeError("decoder block count changed")

            arr = np.stack([
                (states[k + 1][0, sidx] - states[k + 1][0, ridx])
                .detach().float().cpu().numpy()
                for k in range(cur_blocks)
            ], axis=0).astype(dtype_np)
            sids.append(int(rec.sid))
            labels.append(norm_relation(rec.relation))
            vecs.append(arr)

            del outputs, states, batch
            if len(vecs) % args.save_every == 0:
                save_progress()
        except Exception as exc:
            errors.append({
                "sid": int(rec.sid),
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_progress()
    out_path.with_suffix(".errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path} | n={len(vecs)}/{len(records)} | errors={len(errors)}")


def extract_factor_cell_fixed_layer(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[base.Record],
    layer: int,
    image_level: int,
    task_level: int,
    options_level: int,
    out_path: Path,
) -> None:
    if out_path.exists() and not args.overwrite:
        print(f"[reuse] {out_path}")
        return
    if out_path.exists():
        out_path.unlink()

    with_image = image_level == +1
    task_on = task_level == +1
    options_on = options_level == +1
    dtype_np = np.float32 if args.keep_fp32 else np.float16

    sids: List[int] = []
    labels: List[str] = []
    vecs: List[np.ndarray] = []
    subj_positions: List[int] = []
    ref_positions: List[int] = []
    errors: List[Dict[str, Any]] = []
    hidden_size: Optional[int] = None
    blocks_n: Optional[int] = None

    def save_progress() -> None:
        if not vecs or hidden_size is None or blocks_n is None:
            return
        atomic_save_npz(out_path, {
            "metadata_json": np.array(json.dumps({
                "model": args.model,
                "layer": layer,
                "I": image_level,
                "T": task_level,
                "O": options_level,
                "with_image": with_image,
                "prompt_template": factorial_prompt("{subject}", "{reference}", task_on, options_on),
                "probe": "last object mentions in shared suffix; relation=h_A-h_B",
                "decoder_blocks": blocks_n,
                "hidden_size": hidden_size,
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "relation": np.asarray(labels, dtype=object),
            "subject_position": np.asarray(subj_positions, dtype=np.int64),
            "reference_position": np.asarray(ref_positions, dtype=np.int64),
            "relation_vectors": np.stack(vecs).astype(dtype_np),
        })

    tag = f"I{image_level:+d}_T{task_level:+d}_O{options_level:+d}"
    desc = f"{args.model}:{tag}:L{layer}"
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB") if with_image else None
            text = factorial_prompt(rec.subject, rec.reference, task_on, options_on)
            rendered = build_chat_prompt(processor, text, with_image=with_image)
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()

            # Because the shared suffix repeats both objects, find_phrase_last_token
            # selects the matched probe occurrence at the end of the prompt.
            sidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.subject)
            ridx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.reference)
            if sidx == ridx:
                raise RuntimeError("subject/reference token positions collide")

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

            hs = states[layer + 1][0, sidx]
            hr = states[layer + 1][0, ridx]
            rv = (hs - hr).detach().float().cpu().numpy().astype(dtype_np)

            sids.append(int(rec.sid))
            labels.append(norm_relation(rec.relation))
            subj_positions.append(int(sidx))
            ref_positions.append(int(ridx))
            vecs.append(rv)

            del outputs, states, batch
            if len(vecs) % args.save_every == 0:
                save_progress()
        except Exception as exc:
            errors.append({
                "sid": int(rec.sid),
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "relation": str(rec.relation),
                "I": image_level, "T": task_level, "O": options_level,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_progress()
    out_path.with_suffix(".errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path} | n={len(vecs)}/{len(records)} | errors={len(errors)}")


# -----------------------------------------------------------------------------
# Alignment and baseline layer scan
# -----------------------------------------------------------------------------


def baseline_scan(
    data: Dict[str, Any],
    train_ratio: float,
    repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, int, np.ndarray, np.ndarray, List[int], List[Tuple[np.ndarray, np.ndarray]]]:
    X = data["relation_vectors"].astype(np.float64)
    y = np.asarray([norm_relation(x) for x in data["relation"].tolist()], dtype=object)
    layers = [int(x) for x in data["decoder_block_index"].tolist()]
    splits = make_splits(len(y), train_ratio, repeats, seed)

    rows: List[Dict[str, Any]] = []
    for li, layer in enumerate(layers):
        m = eval_direction_fixed(X[:, li, :], y, splits)
        rows.append({"layer": layer, **m})
    df = pd.DataFrame(rows)
    best = df.loc[df["acc_mean"].idxmax()]
    return df, int(best["layer"]), X, y, layers, splits


def align_factor_cells(
    cell_data: Dict[Tuple[int, int, int], Dict[str, Any]]
) -> Tuple[List[int], Dict[Tuple[int, int, int], np.ndarray], np.ndarray]:
    pos: Dict[Tuple[int, int, int], Dict[int, int]] = {}
    sid_sets: List[set[int]] = []
    for key, data in cell_data.items():
        p = {int(s): i for i, s in enumerate(data["sample_index"].tolist())}
        pos[key] = p
        sid_sets.append(set(p))
    common = sorted(set.intersection(*sid_sets))
    if not common:
        raise RuntimeError("no common samples across factorial cells")

    refkey = (+1, +1, +1)
    ref = cell_data[refkey]
    idx0 = np.asarray([pos[refkey][s] for s in common], dtype=np.int64)
    y = np.asarray([norm_relation(x) for x in ref["relation"][idx0]], dtype=object)

    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for key, data in cell_data.items():
        idx = np.asarray([pos[key][s] for s in common], dtype=np.int64)
        yi = np.asarray([norm_relation(x) for x in data["relation"][idx]], dtype=object)
        if not np.array_equal(y, yi):
            raise RuntimeError(f"label mismatch in cell {key}")
        out[key] = data["relation_vectors"][idx].astype(np.float64)
    return common, out, y


def align_baseline_to_sids(data: Dict[str, Any], sids: Sequence[int], layer: int) -> Tuple[np.ndarray, np.ndarray]:
    p = {int(s): i for i, s in enumerate(data["sample_index"].tolist())}
    missing = [s for s in sids if s not in p]
    if missing:
        raise RuntimeError(f"baseline missing {len(missing)} factorial-common samples")
    idx = np.asarray([p[s] for s in sids], dtype=np.int64)
    layers = [int(x) for x in data["decoder_block_index"].tolist()]
    if layer not in layers:
        raise RuntimeError(f"L{layer} absent from baseline states")
    li = layers.index(layer)
    X = data["relation_vectors"][idx, li, :].astype(np.float64)
    y = np.asarray([norm_relation(x) for x in data["relation"][idx]], dtype=object)
    return X, y


# -----------------------------------------------------------------------------
# Exact factorial decomposition + diagnostics
# -----------------------------------------------------------------------------


def factorial_components(cells: Dict[Tuple[int, int, int], np.ndarray]) -> Dict[str, np.ndarray]:
    """Walsh/Hadamard decomposition over I,T,O in {-1,+1}.

    For each sample independently:
      c_S = 1/8 sum_x chi_S(x) r_x
    """
    shape = next(iter(cells.values())).shape
    out = {name: np.zeros(shape, dtype=np.float64) for name in COMPONENTS}
    for (i, t, o), X in cells.items():
        out["mu"] += X
        out["I"] += i * X
        out["T"] += t * X
        out["O"] += o * X
        out["IT"] += i * t * X
        out["IO"] += i * o * X
        out["TO"] += t * o * X
        out["ITO"] += i * t * o * X
    for k in out:
        out[k] /= 8.0
    return out


def reconstruct_from_components(
    comp: Dict[str, np.ndarray], i: int, t: int, o: int, order: int
) -> np.ndarray:
    pred = comp["mu"].copy()
    if order >= 1:
        pred += i*comp["I"] + t*comp["T"] + o*comp["O"]
    if order >= 2:
        pred += i*t*comp["IT"] + i*o*comp["IO"] + t*o*comp["TO"]
    if order >= 3:
        pred += i*t*o*comp["ITO"]
    return pred


def condition_metrics(
    cells: Dict[Tuple[int, int, int], np.ndarray],
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (i, t, o), X in sorted(cells.items()):
        m = eval_direction_fixed(X, y, splits)
        rows.append({
            "I": i, "T": t, "O": o,
            "image": int(i == +1),
            "task": int(t == +1),
            "options": int(o == +1),
            **m,
            "mean_norm": float(np.mean(np.linalg.norm(X, axis=1))),
        })
    return pd.DataFrame(rows)


def image_residual_metrics(
    cells: Dict[Tuple[int, int, int], np.ndarray],
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in LEVELS:
        for o in LEVELS:
            # Exact image/no-image difference for this prompt condition.
            X = cells[(+1, t, o)] - cells[(-1, t, o)]
            m = eval_direction_fixed(X, y, splits)
            rows.append({
                "T": t, "O": o,
                "task": int(t == +1), "options": int(o == +1),
                **m,
                "mean_norm": float(np.mean(np.linalg.norm(X, axis=1))),
            })
    return pd.DataFrame(rows)


def component_metrics(
    comp: Dict[str, np.ndarray],
    baseline_X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    # Parseval-like energy accounting under the orthogonal factorial basis.
    energy = {k: np.mean(np.sum(v*v, axis=1)) for k, v in comp.items()}
    total = max(sum(energy.values()), EPS)
    effect_total = max(sum(energy[k] for k in COMPONENTS if k != "mu"), EPS)

    rows: List[Dict[str, Any]] = []
    for k in COMPONENTS:
        X = comp[k]
        m = eval_direction_fixed(X, y, splits)
        align_mean, align_std = direction_alignment_to_baseline(X, baseline_X, y, splits)
        rows.append({
            "component": k,
            **m,
            "mean_norm": float(np.mean(np.linalg.norm(X, axis=1))),
            "rms_norm": float(np.sqrt(np.mean(np.sum(X*X, axis=1)))),
            "energy_fraction_total": float(energy[k] / total),
            "energy_fraction_effects_only": float(0.0 if k == "mu" else energy[k] / effect_total),
            "direction_alignment_to_original_mean": align_mean,
            "direction_alignment_to_original_std": align_std,
        })
    return pd.DataFrame(rows)


def component_cosine_matrix(comp: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(COMPONENTS)
    mat = np.empty((len(names), len(names)), dtype=np.float64)
    for a, ka in enumerate(names):
        for b, kb in enumerate(names):
            mat[a, b] = float(np.mean(row_cos(comp[ka], comp[kb])))
    return pd.DataFrame(mat, index=names, columns=names)


def additivity_metrics(
    cells: Dict[Tuple[int, int, int], np.ndarray],
    comp: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Measure additivity on FACTOR-CENTERED states, avoiding huge mu cosine.

    For each cell:
      delta_actual = r_cell - mu
      main-only     = iI + tT + oO
      pairwise      = main-only + itIT + ioIO + toTO
      full          = pairwise + itoITO  (must be exact up to FP error)

    Small main-only relative error => factors are approximately additive.
    Large main-only error but much smaller pairwise error => interactions matter.
    """
    rows: List[Dict[str, Any]] = []
    mu = comp["mu"]
    for key, actual in sorted(cells.items()):
        i, t, o = key
        target = actual - mu
        main = i*comp["I"] + t*comp["T"] + o*comp["O"]
        pair = main + i*t*comp["IT"] + i*o*comp["IO"] + t*o*comp["TO"]
        full = pair + i*t*o*comp["ITO"]

        def summarize(pred: np.ndarray, prefix: str) -> Dict[str, float]:
            return {
                f"{prefix}_cos": float(np.mean(row_cos(pred, target))),
                f"{prefix}_relerr": float(np.mean(relative_error(pred, target))),
            }

        row: Dict[str, Any] = {"I": i, "T": t, "O": o}
        row.update(summarize(main, "main"))
        row.update(summarize(pair, "pairwise"))
        row.update(summarize(full, "full"))
        rows.append(row)

    df = pd.DataFrame(rows)
    mean_row: Dict[str, Any] = {"I": 0, "T": 0, "O": 0}
    for col in ("main_cos", "main_relerr", "pairwise_cos", "pairwise_relerr", "full_cos", "full_relerr"):
        mean_row[col] = float(df[col].mean())
    mean_row["summary"] = "mean_over_8_cells"
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0,1)")

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
    (out_dir / "dataset.audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

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
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        # ------------------------------------------------------------------
        # A) ORIGINAL prompt layer scan
        # ------------------------------------------------------------------
        baseline_path = state_dir / "baseline_original__correct_all_layers.npz"
        print("\n" + "="*108)
        print("BASELINE ORIGINAL PROMPT FOR LAYER SELECTION")
        print(original_prompt("A", "B"))
        print("="*108)
        extract_original_all_layers(
            args=args, model=model, processor=processor, device=device,
            records=records, out_path=baseline_path,
        )
        baseline_data = load_npz(baseline_path)
        scan_df, best_layer, baseline_all, baseline_y_all, baseline_layers, _ = baseline_scan(
            baseline_data, args.train_ratio, args.repeats, args.seed
        )
        scan_df.to_csv(out_dir / "baseline_layer_scan.csv", index=False)
        scan_best = scan_df.loc[scan_df["acc_mean"].idxmax()]
        print(
            f"\n[baseline best] L{int(scan_best.layer)} "
            f"acc={100*scan_best.acc_mean:.2f}%±{100*scan_best.acc_std:.2f}%"
        )

        fixed_layer = int(args.fixed_layer) if args.fixed_layer is not None else int(best_layer)
        if fixed_layer not in baseline_layers:
            raise RuntimeError(f"requested fixed L{fixed_layer}, available={baseline_layers}")
        print(f"[FIXED] all factorial analyses will use ONLY L{fixed_layer}")

        # ------------------------------------------------------------------
        # B) Extract 8 factorial cells at the frozen layer
        # ------------------------------------------------------------------
        cell_data: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for i in LEVELS:
            for t in LEVELS:
                for o in LEVELS:
                    tag = f"I{i:+d}__T{t:+d}__O{o:+d}"
                    print("\n" + "="*108)
                    print(f"FACTORIAL CELL {tag}")
                    print(factorial_prompt("A", "B", t == +1, o == +1))
                    print("="*108)
                    path = state_dir / f"factorial__{tag}__L{fixed_layer}.npz"
                    extract_factor_cell_fixed_layer(
                        args=args, model=model, processor=processor, device=device,
                        records=records, layer=fixed_layer,
                        image_level=i, task_level=t, options_level=o,
                        out_path=path,
                    )
                    cell_data[(i, t, o)] = load_npz(path)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        sids, cells, y = align_factor_cells(cell_data)
        baseline_X, baseline_y = align_baseline_to_sids(baseline_data, sids, fixed_layer)
        if not np.array_equal(y, baseline_y):
            raise RuntimeError("baseline/factorial label mismatch")
        print(f"\n[common] n={len(sids)} across all 8 cells + original baseline")
        splits = make_splits(len(sids), args.train_ratio, args.repeats, args.seed)

        # ------------------------------------------------------------------
        # C) Raw condition accuracy at the SAME layer
        # ------------------------------------------------------------------
        base_m = eval_direction_fixed(baseline_X, y, splits)
        cond_df = condition_metrics(cells, y, splits)
        cond_df.to_csv(out_dir / "condition_accuracy_fixed_layer.csv", index=False)

        print("\n" + "="*108)
        print(f"CONDITION DIRECTION ACCURACY @ FIXED L{fixed_layer}")
        print("="*108)
        print(f"original baseline correct-image | acc={100*base_m['acc_mean']:.2f}%±{100*base_m['acc_std']:.2f}%")
        for _, r in cond_df.iterrows():
            print(
                f"I={int(r['I']):+d} T={int(r['T']):+d} O={int(r['O']):+d} | "
                f"acc={100*r.acc_mean:.2f}%±{100*r.acc_std:.2f}% | norm={r.mean_norm:.3f}"
            )

        # ------------------------------------------------------------------
        # D) Image residual under each T/O prompt
        # ------------------------------------------------------------------
        resid_df = image_residual_metrics(cells, y, splits)
        resid_df.to_csv(out_dir / "image_residual_accuracy_fixed_layer.csv", index=False)
        print("\n" + "="*108)
        print(f"IMAGE - NOIMAGE RESIDUAL ACCURACY @ FIXED L{fixed_layer}")
        print("="*108)
        for _, r in resid_df.iterrows():
            print(
                f"T={int(r['T']):+d} O={int(r['O']):+d} | "
                f"acc={100*r.acc_mean:.2f}%±{100*r.acc_std:.2f}% | norm={r.mean_norm:.3f}"
            )

        # ------------------------------------------------------------------
        # E) Exact factorial decomposition
        # ------------------------------------------------------------------
        comp = factorial_components(cells)
        comp_df = component_metrics(comp, baseline_X, y, splits)
        comp_df.to_csv(out_dir / "component_metrics_fixed_layer.csv", index=False)

        cos_df = component_cosine_matrix(comp)
        cos_df.to_csv(out_dir / "component_cosine_matrix.csv")

        add_df = additivity_metrics(cells, comp)
        add_df.to_csv(out_dir / "additivity_fixed_layer.csv", index=False)

        print("\n" + "="*108)
        print(f"FACTORIAL COMPONENTS @ FIXED L{fixed_layer}")
        print("="*108)
        print("component | spatial ACC | effect-energy | align-to-original | mean norm")
        for _, r in comp_df.iterrows():
            eff = "   n/a" if r.component == "mu" else f"{100*r.energy_fraction_effects_only:6.2f}%"
            print(
                f"{str(r.component):>9s} | "
                f"{100*r.acc_mean:6.2f}%±{100*r.acc_std:4.2f}% | "
                f"{eff} | "
                f"{r.direction_alignment_to_original_mean:+.4f}±{r.direction_alignment_to_original_std:.4f} | "
                f"{r.mean_norm:.3f}"
            )

        add_mean = add_df[add_df.get("summary", pd.Series(index=add_df.index, dtype=object)) == "mean_over_8_cells"]
        if len(add_mean) == 1:
            r = add_mean.iloc[0]
            print("\n" + "="*108)
            print("LINEAR ADDITIVITY ON FACTOR-CENTERED STATES (mean over 8 cells)")
            print("="*108)
            print(f"main effects only : cos={r.main_cos:.4f} | relerr={r.main_relerr:.4f}")
            print(f"+ pair interactions: cos={r.pairwise_cos:.4f} | relerr={r.pairwise_relerr:.4f}")
            print(f"+ 3-way interaction: cos={r.full_cos:.4f} | relerr={r.full_relerr:.6f}  (algebraic exactness check)")

        # Helpful interpretation quantities.
        rowmap = {str(r.component): r for _, r in comp_df.iterrows()}
        interaction_energy = sum(
            float(rowmap[k].energy_fraction_effects_only) for k in ("IT", "IO", "TO", "ITO")
        )
        image_related_energy = sum(
            float(rowmap[k].energy_fraction_effects_only) for k in ("I", "IT", "IO", "ITO")
        )

        summary = {
            "model": args.model,
            "repo_id": spec.repo_id,
            "n_common": len(sids),
            "original_prompt": original_prompt("{subject}", "{reference}"),
            "factorial_prompt_examples": {
                f"T{t:+d}_O{o:+d}": factorial_prompt("{subject}", "{reference}", t == +1, o == +1)
                for t in LEVELS for o in LEVELS
            },
            "selected_layer": fixed_layer,
            "layer_selected_automatically": args.fixed_layer is None,
            "baseline_best_layer": int(best_layer),
            "baseline_best_acc_mean": float(scan_best.acc_mean),
            "baseline_best_acc_std": float(scan_best.acc_std),
            "baseline_fixed_layer_acc": base_m,
            "interaction_energy_fraction_of_effects": interaction_energy,
            "image_related_energy_fraction_of_effects": image_related_energy,
            "component_metrics": {
                str(r.component): {
                    "acc_mean": float(r.acc_mean),
                    "acc_std": float(r.acc_std),
                    "effect_energy_fraction": float(r.energy_fraction_effects_only),
                    "alignment_to_original": float(r.direction_alignment_to_original_mean),
                    "mean_norm": float(r.mean_norm),
                }
                for _, r in comp_df.iterrows()
            },
            "notes": [
                "The full 8-term factorial reconstruction is mathematically exact by construction.",
                "Use main-only vs pairwise reconstruction error to assess approximate additivity.",
                "I is half of the average image-minus-noimage effect under effect coding; scaling does not affect cosine-direction ACC.",
                "High IT/IO/ITO spatial ACC means prompt factors modulate image-conditioned relation information rather than merely adding text-only offsets.",
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n" + "="*108)
        print("KEY SUMMARY")
        print("="*108)
        print(f"fixed layer                         : L{fixed_layer}")
        print(f"original direction ACC              : {100*base_m['acc_mean']:.2f}%")
        print(f"I component spatial ACC             : {100*rowmap['I'].acc_mean:.2f}%")
        print(f"IT component spatial ACC            : {100*rowmap['IT'].acc_mean:.2f}%")
        print(f"IO component spatial ACC            : {100*rowmap['IO'].acc_mean:.2f}%")
        print(f"ITO component spatial ACC           : {100*rowmap['ITO'].acc_mean:.2f}%")
        print(f"image-related effect energy         : {100*image_related_energy:.2f}%")
        print(f"all interaction effect energy       : {100*interaction_energy:.2f}%")
        print(f"Saved: {out_dir}")
        print(f"Elapsed: {(time.time()-started)/60:.1f} min")

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
