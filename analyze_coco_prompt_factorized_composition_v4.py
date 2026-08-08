#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt-factorized spatial COMPOSITION / RECONSTRUCTION analysis on COCO-two (v4).

Four short factor prompts + one Full baseline:
    P_A:    Locate the A in the image.
    P_B:    Locate the B in the image.
    P_H:    Where is A relative to B horizontally? Answer left or right.
    P_V:    Where is A relative to B vertically? Answer above or below.
    P_FULL: Where is A relative to B in the image? Answer left, right, above, or below.

This version extracts BOTH:
  1) last prompt token states (after chat template, add_generation_prompt=True)
  2) object-token states

Object-token representations:
  P_A:    h_A from the A mention in P_A
  P_B:    h_B from the B mention in P_B
  P_H:    h_A^H, h_B^H, and d_H = h_A^H - h_B^H
  P_V:    h_A^V, h_B^V, and d_V = h_A^V - h_B^V
  P_FULL: h_A^F, h_B^F, and d_F = h_A^F - h_B^F

For every prompt/representation we run correct-image and no-image conditions.
Residuals are image-conditioned differences:
  last_res(P) = last_img(P) - last_noimg(P)
  obj_res(A)  = h_A,img - h_A,noimg
  diff_res(P) = (h_A,img-h_B,img) - (h_A,noimg-h_B,noimg)

Zero-training composition tests are run separately for LAST and OBJECT spaces:

LAST-token space:
  location_diff = z_A - z_B
  axis_sum      = z_H + z_V
  all_sum       = (z_A-z_B) + z_H + z_V
  target        = z_FULL

OBJECT-token space:
  location_diff = h_A(P_A) - h_B(P_B)
  axis_sum      = d_H + d_V
  all_sum       = [h_A(P_A)-h_B(P_B)] + d_H + d_V
  target        = d_FULL

Matched-vs-shuffled controls are reported for all combinations.
The focus of v4 is reconstruction: can factor prompts be composed to recover the Full representation?
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
PROMPT_TYPES = ("loc_a", "loc_b", "horizontal", "vertical", "full")


def norm_relation(x: Any) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(key, key)


def prompt_text(prompt_type: str, subject: str, reference: str) -> str:
    if prompt_type == "loc_a":
        return f"Locate the {subject} in the image."
    if prompt_type == "loc_b":
        return f"Locate the {reference} in the image."
    if prompt_type == "horizontal":
        return f"Where is the {subject} relative to the {reference} horizontally? Answer left or right."
    if prompt_type == "vertical":
        return f"Where is the {subject} relative to the {reference} vertically? Answer above or below."
    if prompt_type == "full":
        return f"Where is the {subject} relative to the {reference} in the image? Answer left, right, above, or below."
    raise ValueError(prompt_type)


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
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-fp32", action="store_true")
    return p.parse_args()


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_chat_prompt(processor: Any, prompt_type: str, subject: str, reference: str, *, with_image: bool) -> str:
    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text(prompt_type, subject, reference)})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def _stack_layer_vectors(states: Sequence[torch.Tensor], token_index: int, dtype_np: np.dtype) -> np.ndarray:
    return np.stack([
        states[k + 1][0, token_index].detach().float().cpu().numpy()
        for k in range(len(states) - 1)
    ], axis=0).astype(dtype_np)


def extract_condition(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[base.Record],
    prompt_type: str,
    with_image: bool,
    out_path: Path,
) -> None:
    """Extract last-token plus all object-token states needed by a prompt."""
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
    last_positions: List[int] = []
    subject_positions: List[int] = []
    reference_positions: List[int] = []

    last_vectors: List[np.ndarray] = []
    subject_vectors: List[np.ndarray] = []
    reference_vectors: List[np.ndarray] = []
    object_diff_vectors: List[np.ndarray] = []

    errors: List[Dict[str, Any]] = []
    blocks_n: Optional[int] = None
    hidden_size: Optional[int] = None

    has_subject = prompt_type in ("loc_a", "horizontal", "vertical", "full")
    has_reference = prompt_type in ("loc_b", "horizontal", "vertical", "full")
    has_diff = prompt_type in ("horizontal", "vertical", "full")

    def save_progress() -> None:
        if not last_vectors or blocks_n is None or hidden_size is None:
            return
        arrays: Dict[str, Any] = {
            "metadata_json": np.array(json.dumps({
                "model": args.model,
                "repo_id": base.SPECS[args.model].repo_id,
                "prompt_type": prompt_type,
                "prompt_template": prompt_text(prompt_type, "{subject}", "{reference}"),
                "vision_mode": mode,
                "last_probe_position": "last_nonpadding_prompt_token",
                "object_probe_position": "last_subtoken_of_object_phrase",
                "decoder_blocks": blocks_n,
                "hidden_size": hidden_size,
                "n_saved": len(sids),
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "image_id": np.asarray(image_ids, dtype=object),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "relation": np.asarray(labels, dtype=object),
            "last_position": np.asarray(last_positions, dtype=np.int64),
            "decoder_block_index": np.arange(blocks_n, dtype=np.int32),
            "last_vectors": np.stack(last_vectors).astype(dtype_np),
        }
        if has_subject:
            arrays["subject_position"] = np.asarray(subject_positions, dtype=np.int64)
            arrays["subject_vectors"] = np.stack(subject_vectors).astype(dtype_np)
        if has_reference:
            arrays["reference_position"] = np.asarray(reference_positions, dtype=np.int64)
            arrays["reference_vectors"] = np.stack(reference_vectors).astype(dtype_np)
        if has_diff:
            arrays["object_diff_vectors"] = np.stack(object_diff_vectors).astype(dtype_np)
        atomic_save_npz(out_path, arrays)

    desc = f"{args.model}:{prompt_type}:{mode}"
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB") if with_image else None
            rendered = build_chat_prompt(processor, prompt_type, rec.subject, rec.reference, with_image=with_image)
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()

            if "attention_mask" in batch:
                last_idx = int(batch["attention_mask"][0].sum().item()) - 1
            else:
                last_idx = len(input_ids) - 1

            subj_idx: Optional[int] = None
            ref_idx: Optional[int] = None
            if has_subject:
                subj_idx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.subject)
            if has_reference:
                ref_idx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.reference)
            if has_diff and subj_idx == ref_idx:
                raise RuntimeError("Subject/reference token positions collide.")

            with torch.inference_mode():
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                states = base.hidden_tuple(outputs)

            final = states[-1]
            if final.ndim != 3 or final.shape[0] != 1:
                raise RuntimeError(f"Unexpected hidden-state shape: {tuple(final.shape)}")
            if int(final.shape[1]) != len(input_ids):
                raise RuntimeError(
                    "Text token positions do not align with hidden states: "
                    f"input_len={len(input_ids)}, hidden_len={final.shape[1]}"
                )

            current_blocks = len(states) - 1
            if blocks_n is None:
                blocks_n = current_blocks
                hidden_size = int(final.shape[-1])
                print(f"[{desc}] decoder_blocks={blocks_n}, hidden={hidden_size}")
            elif current_blocks != blocks_n:
                raise RuntimeError(f"decoder blocks changed {blocks_n}->{current_blocks}")

            lvec = _stack_layer_vectors(states, last_idx, dtype_np)
            svec = _stack_layer_vectors(states, int(subj_idx), dtype_np) if subj_idx is not None else None
            rvec = _stack_layer_vectors(states, int(ref_idx), dtype_np) if ref_idx is not None else None

            sids.append(int(rec.sid))
            image_ids.append(str(rec.image_id))
            subjects.append(str(rec.subject))
            references.append(str(rec.reference))
            labels.append(norm_relation(rec.relation))
            last_positions.append(int(last_idx))
            last_vectors.append(lvec)

            if has_subject:
                assert subj_idx is not None and svec is not None
                subject_positions.append(int(subj_idx))
                subject_vectors.append(svec)
            if has_reference:
                assert ref_idx is not None and rvec is not None
                reference_positions.append(int(ref_idx))
                reference_vectors.append(rvec)
            if has_diff:
                assert svec is not None and rvec is not None
                object_diff_vectors.append((svec.astype(np.float32) - rvec.astype(np.float32)).astype(dtype_np))

            del outputs, states, batch
            if len(last_vectors) % args.save_every == 0:
                save_progress()
        except Exception as exc:
            errors.append({
                "sid": int(rec.sid),
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "relation": str(rec.relation),
                "prompt_type": prompt_type,
                "vision_mode": mode,
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
    print(f"[saved] {out_path} | n={len(last_vectors)}/{len(records)} | errors={len(errors)}")


def normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), EPS)


def row_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sum(normalize_rows(A) * normalize_rows(B), axis=-1)


def relative_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=-1) / np.maximum(np.linalg.norm(target, axis=-1), EPS)


def make_splits(n: int, ratio: float, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for rep in range(repeats):
        ids = list(range(n))
        random.Random(seed + rep).shuffle(ids)
        n_train = int(n * ratio)
        if n_train <= 0 or n_train >= n:
            raise RuntimeError(f"bad train size={n_train} for n={n}")
        out.append((np.asarray(ids[:n_train], dtype=np.int64), np.asarray(ids[n_train:], dtype=np.int64)))
    return out


def align_state_map(
    state_map: Dict[Tuple[str, str], Dict[str, Any]]
) -> Tuple[List[int], Dict[str, Dict[str, Dict[str, np.ndarray]]], np.ndarray, List[int]]:
    """Align all prompt/vision conditions and expose last/object arrays."""
    positions: Dict[Tuple[str, str], Dict[int, int]] = {}
    sid_sets: List[set[int]] = []
    for key, data in state_map.items():
        pos = {int(s): i for i, s in enumerate(data["sample_index"].tolist())}
        positions[key] = pos
        sid_sets.append(set(pos))
    common = sorted(set.intersection(*sid_sets))
    if not common:
        raise RuntimeError("no common samples across prompt/vision conditions")

    ref = state_map[("full", "correct")]
    ref_pos = positions[("full", "correct")]
    ref_idx = np.asarray([ref_pos[s] for s in common], dtype=np.int64)
    y = np.asarray([norm_relation(x) for x in ref["relation"][ref_idx]], dtype=object)
    layers = [int(x) for x in ref["decoder_block_index"].tolist()]

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for ptype in PROMPT_TYPES:
        out[ptype] = {}
        for mode in ("correct", "no_image"):
            data = state_map[(ptype, mode)]
            idx = np.asarray([positions[(ptype, mode)][s] for s in common], dtype=np.int64)
            yi = np.asarray([norm_relation(x) for x in data["relation"][idx]], dtype=object)
            if not np.array_equal(y, yi):
                raise RuntimeError(f"label mismatch for {ptype}/{mode}")
            if [int(x) for x in data["decoder_block_index"].tolist()] != layers:
                raise RuntimeError(f"layer mismatch for {ptype}/{mode}")

            entry: Dict[str, np.ndarray] = {
                "last": data["last_vectors"][idx].astype(np.float64)
            }
            if "subject_vectors" in data:
                entry["subject"] = data["subject_vectors"][idx].astype(np.float64)
            if "reference_vectors" in data:
                entry["reference"] = data["reference_vectors"][idx].astype(np.float64)
            if "object_diff_vectors" in data:
                entry["diff"] = data["object_diff_vectors"][idx].astype(np.float64)
            out[ptype][mode] = entry
    return common, out, y, layers


def _global_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Global held-out R^2 after target has been train-centered."""
    num = float(np.sum((target - pred) ** 2))
    den = float(np.sum(target ** 2))
    return 1.0 - num / max(den, EPS)


def _fit_three_scalars(
    Gtr: np.ndarray, Htr: np.ndarray, Vtr: np.ndarray, Ftr: np.ndarray, ridge: float = 1e-8
) -> np.ndarray:
    """Fit only 3 global scalars: F ~= alpha*G + beta*H + gamma*V.

    No D x D map is trained. The least-squares problem is accumulated with
    vector dot products, so it is cheap even for large hidden sizes.
    """
    comps = (Gtr, Htr, Vtr)
    gram = np.empty((3, 3), dtype=np.float64)
    rhs = np.empty(3, dtype=np.float64)
    for i, Xi in enumerate(comps):
        rhs[i] = float(np.sum(Xi * Ftr))
        for j, Xj in enumerate(comps):
            gram[i, j] = float(np.sum(Xi * Xj))
    scale = max(float(np.trace(gram)) / 3.0, 1.0)
    gram = gram + ridge * scale * np.eye(3, dtype=np.float64)
    return np.linalg.solve(gram, rhs)


def _oracle_span_reconstruction(G: np.ndarray, H: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-sample oracle projection of F onto span{G,H,V}.

    This uses F to choose 3 coefficients independently for each sample, so it
    is NOT a predictive method. It is an upper-bound geometric diagnostic:
    does the factor-vector span contain the Full vector at all?
    """
    n, d = F.shape
    pred = np.empty_like(F, dtype=np.float64)
    for i in range(n):
        M = np.stack([G[i], H[i], V[i]], axis=1)  # [D,3]
        gram = M.T @ M
        scale = max(float(np.trace(gram)) / 3.0, 1.0)
        coef = np.linalg.solve(gram + 1e-8 * scale * np.eye(3), M.T @ F[i])
        pred[i] = M @ coef
    return pred


def summarize_four_component_composition(
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    layers: Sequence[int],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> pd.DataFrame:
    """Test whether factor-prompt information composes into Full.

    G = A-B is the location-difference factor.
    H and V are horizontal / vertical factors.
    F is the Full representation.

    Reported tests:
      * equal sum:        G + H + V
      * normalized sum:   unit(G)+unit(H)+unit(V)
      * axis-only sum:    H + V
      * matched-shuffled controls
      * 3-scalar held-out reconstruction after train-only centering
      * per-sample oracle span reconstruction (upper bound)

    All metrics are layer-wise and are evaluated separately in LAST and OBJECT
    spaces, for both raw and image-minus-noimage residual representations.
    """
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    perm_g = rng.permutation(len(A))
    perm_h = rng.permutation(len(A))
    perm_v = rng.permutation(len(A))

    for li, layer in enumerate(layers):
        a, b = A[:, li], B[:, li]
        h, v, f = H[:, li], V[:, li], F[:, li]
        g = a - b

        axis_sum = h + v
        equal_sum = g + h + v
        normalized_sum = normalize_rows(g) + normalize_rows(h) + normalize_rows(v)

        # Matched-vs-shuffled: keep Full_i fixed and break factor identity.
        g_shuf = g[perm_g]
        h_shuf = h[perm_h]
        v_shuf = v[perm_v]
        axis_shuf = h_shuf + v_shuf
        all_shuf = g_shuf + h_shuf + v_shuf
        norm_shuf = normalize_rows(g_shuf) + normalize_rows(h_shuf) + normalize_rows(v_shuf)

        c_g = row_cos(g, f)
        c_h = row_cos(h, f)
        c_v = row_cos(v, f)
        c_axis = row_cos(axis_sum, f)
        c_equal = row_cos(equal_sum, f)
        c_norm = row_cos(normalized_sum, f)
        best_factor = np.maximum.reduce([c_g, c_h, c_v])

        # Oracle geometric upper bound: is F in the sample-specific span?
        oracle = _oracle_span_reconstruction(g, h, v, f)
        c_oracle = row_cos(oracle, f)

        # Held-out three-scalar combination. Center EVERY representation using
        # means estimated from training only, to remove prompt-specific offsets.
        fit_cos: List[float] = []
        fit_r2: List[float] = []
        fit_relerr: List[float] = []
        alphas: List[float] = []
        betas: List[float] = []
        gammas: List[float] = []
        equal_centered_cos: List[float] = []
        norm_centered_cos: List[float] = []

        for tr, te in splits:
            mg = g[tr].mean(axis=0, keepdims=True)
            mh = h[tr].mean(axis=0, keepdims=True)
            mv = v[tr].mean(axis=0, keepdims=True)
            mf = f[tr].mean(axis=0, keepdims=True)

            gtr, htr, vtr, ftr = g[tr]-mg, h[tr]-mh, v[tr]-mv, f[tr]-mf
            gte, hte, vte, fte = g[te]-mg, h[te]-mh, v[te]-mv, f[te]-mf

            coef = _fit_three_scalars(gtr, htr, vtr, ftr)
            pred = coef[0]*gte + coef[1]*hte + coef[2]*vte
            fit_cos.append(float(np.mean(row_cos(pred, fte))))
            fit_r2.append(_global_r2(pred, fte))
            fit_relerr.append(float(np.mean(relative_error(pred, fte))))
            alphas.append(float(coef[0])); betas.append(float(coef[1])); gammas.append(float(coef[2]))

            eq = gte + hte + vte
            nm = normalize_rows(gte) + normalize_rows(hte) + normalize_rows(vte)
            equal_centered_cos.append(float(np.mean(row_cos(eq, fte))))
            norm_centered_cos.append(float(np.mean(row_cos(nm, fte))))

        rows.append({
            "layer": int(layer),
            # Single factors / simple combinations
            "cos_location_full": float(np.mean(c_g)),
            "cos_horizontal_full": float(np.mean(c_h)),
            "cos_vertical_full": float(np.mean(c_v)),
            "cos_axis_sum_full": float(np.mean(c_axis)),
            "cos_equal_all_full": float(np.mean(c_equal)),
            "cos_normalized_all_full": float(np.mean(c_norm)),
            "best_single_factor_cos": float(np.mean(best_factor)),
            "equal_gain_over_best_factor": float(np.mean(c_equal-best_factor)),
            "normalized_gain_over_best_factor": float(np.mean(c_norm-best_factor)),
            # Matching controls
            "axis_matched_minus_shuffled": float(np.mean(c_axis) - np.mean(row_cos(axis_shuf, f))),
            "equal_all_matched_minus_shuffled": float(np.mean(c_equal) - np.mean(row_cos(all_shuf, f))),
            "normalized_all_matched_minus_shuffled": float(np.mean(c_norm) - np.mean(row_cos(norm_shuf, f))),
            # Reconstruction error
            "relative_error_equal_all": float(np.mean(relative_error(equal_sum, f))),
            "relative_error_normalized_all": float(np.mean(relative_error(normalized_sum, f))),
            # Train-centered zero-training sums
            "centered_equal_all_cos_mean": float(np.mean(equal_centered_cos)),
            "centered_equal_all_cos_std": float(np.std(equal_centered_cos)),
            "centered_normalized_all_cos_mean": float(np.mean(norm_centered_cos)),
            "centered_normalized_all_cos_std": float(np.std(norm_centered_cos)),
            # Only THREE learned global scalars
            "scalar_fit_cos_mean": float(np.mean(fit_cos)),
            "scalar_fit_cos_std": float(np.std(fit_cos)),
            "scalar_fit_r2_mean": float(np.mean(fit_r2)),
            "scalar_fit_r2_std": float(np.std(fit_r2)),
            "scalar_fit_relative_error_mean": float(np.mean(fit_relerr)),
            "scalar_fit_relative_error_std": float(np.std(fit_relerr)),
            "alpha_location_mean": float(np.mean(alphas)),
            "alpha_location_std": float(np.std(alphas)),
            "beta_horizontal_mean": float(np.mean(betas)),
            "beta_horizontal_std": float(np.std(betas)),
            "gamma_vertical_mean": float(np.mean(gammas)),
            "gamma_vertical_std": float(np.std(gammas)),
            # Oracle span upper bound
            "oracle_span_cos": float(np.mean(c_oracle)),
            "oracle_span_relative_error": float(np.mean(relative_error(oracle, f))),
        })
    return pd.DataFrame(rows)

def axis_from_train(X: np.ndarray, y: np.ndarray, pos_label: str, neg_label: str) -> np.ndarray:
    pos = X[y == pos_label]
    neg = X[y == neg_label]
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError(f"missing labels for axis {neg_label}<->{pos_label}")
    return normalize(pos.mean(axis=0) - neg.mean(axis=0))


def binary_axis_accuracy_with_train_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    axis: np.ndarray,
    pos_label: str,
    neg_label: str,
) -> float:
    tr_mask = np.isin(y_train, [pos_label, neg_label])
    te_mask = np.isin(y_test, [pos_label, neg_label])
    if int(tr_mask.sum()) == 0 or int(te_mask.sum()) == 0:
        return float("nan")
    tr_scores = X_train[tr_mask] @ axis
    tr_y = y_train[tr_mask]
    pos_scores = tr_scores[tr_y == pos_label]
    neg_scores = tr_scores[tr_y == neg_label]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    threshold = 0.5 * (float(pos_scores.mean()) + float(neg_scores.mean()))
    te_scores = X_test[te_mask] @ axis
    gt_pos = y_test[te_mask] == pos_label
    return float(np.mean((te_scores > threshold) == gt_pos))


def summarize_direction_transfer(
    H: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    layers: Sequence[int],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for li, layer in enumerate(layers):
        align_x: List[float] = []
        align_y: List[float] = []
        xy_abs: List[float] = []
        acc_x_full: List[float] = []
        acc_y_full: List[float] = []

        for tr, te in splits:
            htr, vtr, ftr = H[tr, li], V[tr, li], F[tr, li]
            ytr, fte, yte = y[tr], F[te, li], y[te]
            vx_h = axis_from_train(htr, ytr, "right", "left")
            vy_v = axis_from_train(vtr, ytr, "above", "below")
            vx_f = axis_from_train(ftr, ytr, "right", "left")
            vy_f = axis_from_train(ftr, ytr, "above", "below")

            align_x.append(float(np.dot(vx_h, vx_f)))
            align_y.append(float(np.dot(vy_v, vy_f)))
            xy_abs.append(float(abs(np.dot(vx_h, vy_v))))
            acc_x_full.append(binary_axis_accuracy_with_train_threshold(
                ftr, ytr, fte, yte, vx_h, "right", "left"
            ))
            acc_y_full.append(binary_axis_accuracy_with_train_threshold(
                ftr, ytr, fte, yte, vy_v, "above", "below"
            ))

        rows.append({
            "layer": int(layer),
            "cos_vx_horizontal_vs_full_mean": float(np.mean(align_x)),
            "cos_vx_horizontal_vs_full_std": float(np.std(align_x)),
            "cos_vy_vertical_vs_full_mean": float(np.mean(align_y)),
            "cos_vy_vertical_vs_full_std": float(np.std(align_y)),
            "abs_cos_vx_vs_vy_mean": float(np.mean(xy_abs)),
            "abs_cos_vx_vs_vy_std": float(np.std(xy_abs)),
            "full_lr_acc_using_horizontal_axis_mean": float(np.mean(acc_x_full)),
            "full_lr_acc_using_horizontal_axis_std": float(np.std(acc_x_full)),
            "full_ud_acc_using_vertical_axis_mean": float(np.mean(acc_y_full)),
            "full_ud_acc_using_vertical_axis_std": float(np.std(acc_y_full)),
        })
    return pd.DataFrame(rows)


def best_row(df: pd.DataFrame, col: str) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(vals.to_numpy(dtype=float)).any():
        raise RuntimeError(f"no finite values in {col}")
    return df.loc[vals.idxmax()]


def print_composition_block(name: str, rep_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    b_axis = best_row(df, "cos_axis_sum_full")
    b_equal = best_row(df, "cos_equal_all_full")
    b_norm = best_row(df, "cos_normalized_all_full")
    b_center = best_row(df, "centered_normalized_all_cos_mean")
    b_fit = best_row(df, "scalar_fit_cos_mean")
    b_r2 = best_row(df, "scalar_fit_r2_mean")
    b_oracle = best_row(df, "oracle_span_cos")
    b_match = best_row(df, "normalized_all_matched_minus_shuffled")

    print("\n" + "=" * 104)
    print(f"{name} COMPOSITION / RECONSTRUCTION [{rep_name}]")
    print("=" * 104)
    print(f"best H+V -> Full                 : L{int(b_axis.layer):>2d} cos={b_axis.cos_axis_sum_full:.4f}")
    print(f"best G+H+V -> Full               : L{int(b_equal.layer):>2d} cos={b_equal.cos_equal_all_full:.4f} | gain_vs_best_factor={b_equal.equal_gain_over_best_factor:+.4f}")
    print(f"best unit(G)+unit(H)+unit(V)     : L{int(b_norm.layer):>2d} cos={b_norm.cos_normalized_all_full:.4f} | gain_vs_best_factor={b_norm.normalized_gain_over_best_factor:+.4f}")
    print(f"best centered normalized sum     : L{int(b_center.layer):>2d} cos={b_center.centered_normalized_all_cos_mean:.4f}±{b_center.centered_normalized_all_cos_std:.4f}")
    print(f"best 3-scalar heldout fit        : L{int(b_fit.layer):>2d} cos={b_fit.scalar_fit_cos_mean:.4f}±{b_fit.scalar_fit_cos_std:.4f}")
    print(f"best 3-scalar heldout R^2        : L{int(b_r2.layer):>2d} R2={b_r2.scalar_fit_r2_mean:.4f}±{b_r2.scalar_fit_r2_std:.4f}")
    print(f"  weights at best-cos layer      : alpha(G)={b_fit.alpha_location_mean:+.3f}, beta(H)={b_fit.beta_horizontal_mean:+.3f}, gamma(V)={b_fit.gamma_vertical_mean:+.3f}")
    print(f"best oracle span{{G,H,V}} -> Full : L{int(b_oracle.layer):>2d} cos={b_oracle.oracle_span_cos:.4f} | relerr={b_oracle.oracle_span_relative_error:.4f}")
    print(f"best matched - shuffled          : L{int(b_match.layer):>2d} delta={b_match.normalized_all_matched_minus_shuffled:+.4f}")

    return {
        "axis_sum": b_axis.to_dict(),
        "equal_all": b_equal.to_dict(),
        "normalized_all": b_norm.to_dict(),
        "centered_normalized_all": b_center.to_dict(),
        "scalar_fit_cos": b_fit.to_dict(),
        "scalar_fit_r2": b_r2.to_dict(),
        "oracle_span": b_oracle.to_dict(),
        "matched_minus_shuffled": b_match.to_dict(),
    }

def print_direction_block(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    bx = best_row(df, "cos_vx_horizontal_vs_full_mean")
    by = best_row(df, "cos_vy_vertical_vs_full_mean")
    bax = best_row(df, "full_lr_acc_using_horizontal_axis_mean")
    bay = best_row(df, "full_ud_acc_using_vertical_axis_mean")

    print("\n" + "=" * 100)
    print(f"{name} DIRECTION-LEVEL TRANSFER [residual]")
    print("=" * 100)
    print(f"best x-axis alignment H->Full  : L{int(bx.layer):>2d} cos={bx.cos_vx_horizontal_vs_full_mean:.4f}±{bx.cos_vx_horizontal_vs_full_std:.4f} | |cos(x,y)|={bx.abs_cos_vx_vs_vy_mean:.4f}")
    print(f"best y-axis alignment V->Full  : L{int(by.layer):>2d} cos={by.cos_vy_vertical_vs_full_mean:.4f}±{by.cos_vy_vertical_vs_full_std:.4f} | |cos(x,y)|={by.abs_cos_vx_vs_vy_mean:.4f}")
    print(f"best Full LR acc using H-axis  : L{int(bax.layer):>2d} acc={100*bax.full_lr_acc_using_horizontal_axis_mean:.2f}%±{100*bax.full_lr_acc_using_horizontal_axis_std:.2f}%")
    print(f"best Full UD acc using V-axis  : L{int(bay.layer):>2d} acc={100*bay.full_ud_acc_using_vertical_axis_mean:.2f}%±{100*bay.full_ud_acc_using_vertical_axis_std:.2f}%")

    return {
        "x_axis_alignment": bx.to_dict(),
        "y_axis_alignment": by.to_dict(),
        "lr_transfer_accuracy": bax.to_dict(),
        "ud_transfer_accuracy": bay.to_dict(),
    }


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

        state_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for ptype in PROMPT_TYPES:
            print("\n" + "=" * 100)
            print(f"PROMPT TYPE: {ptype}")
            print(prompt_text(ptype, "A", "B"))
            print("=" * 100)
            for mode, with_image in (("correct", True), ("no_image", False)):
                path = state_dir / f"{ptype}__{mode}.npz"
                extract_condition(
                    args=args,
                    model=model,
                    processor=processor,
                    device=device,
                    records=records,
                    prompt_type=ptype,
                    with_image=with_image,
                    out_path=path,
                )
                state_map[(ptype, mode)] = load_npz(path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        sids, aligned, y, layers = align_state_map(state_map)
        print(f"\n[common] n={len(sids)} across all 10 prompt/vision conditions")
        splits = make_splits(len(sids), args.train_ratio, args.repeats, args.seed)

        # ---------------------------- LAST TOKEN ----------------------------
        last_raw = {p: aligned[p]["correct"]["last"] for p in PROMPT_TYPES}
        last_res = {
            p: aligned[p]["correct"]["last"] - aligned[p]["no_image"]["last"]
            for p in PROMPT_TYPES
        }

        # ---------------------------- OBJECT TOKEN --------------------------
        # P_A/P_B use their single object token. H/V/F use object-difference.
        obj_raw = {
            "loc_a": aligned["loc_a"]["correct"]["subject"],
            "loc_b": aligned["loc_b"]["correct"]["reference"],
            "horizontal": aligned["horizontal"]["correct"]["diff"],
            "vertical": aligned["vertical"]["correct"]["diff"],
            "full": aligned["full"]["correct"]["diff"],
        }
        obj_res = {
            "loc_a": aligned["loc_a"]["correct"]["subject"] - aligned["loc_a"]["no_image"]["subject"],
            "loc_b": aligned["loc_b"]["correct"]["reference"] - aligned["loc_b"]["no_image"]["reference"],
            "horizontal": aligned["horizontal"]["correct"]["diff"] - aligned["horizontal"]["no_image"]["diff"],
            "vertical": aligned["vertical"]["correct"]["diff"] - aligned["vertical"]["no_image"]["diff"],
            "full": aligned["full"]["correct"]["diff"] - aligned["full"]["no_image"]["diff"],
        }

        summary: Dict[str, Any] = {
            "config": {
                "model": args.model,
                "repo_id": spec.repo_id,
                "prompts": {p: prompt_text(p, "{subject}", "{reference}") for p in PROMPT_TYPES},
                "last_probe": "last_nonpadding_prompt_token",
                "object_probe": "last_subtoken_of_object_phrase",
                "train_ratio": args.train_ratio,
                "repeats": args.repeats,
                "seed": args.seed,
                "n_common": len(sids),
            },
            "best": {},
        }

        compact_rows: List[Dict[str, Any]] = []
        for space_name, reps in (
            ("last", {"raw": last_raw, "residual": last_res}),
            ("object", {"raw": obj_raw, "residual": obj_res}),
        ):
            summary["best"][space_name] = {}
            for rep_name, rep in reps.items():
                cdf = summarize_four_component_composition(
                    rep["loc_a"], rep["loc_b"], rep["horizontal"], rep["vertical"], rep["full"],
                    layers, splits, args.seed
                )
                cpath = out_dir / f"composition_by_layer__{space_name}__{rep_name}.csv"
                cdf.to_csv(cpath, index=False)
                best = print_composition_block(space_name.upper(), rep_name, cdf)
                summary["best"][space_name][rep_name] = best
                compact_rows.append({
                    "model": args.model,
                    "space": space_name,
                    "representation": rep_name,
                    "best_axis_layer": int(best["axis_sum"]["layer"]),
                    "best_axis_cos": float(best["axis_sum"]["cos_axis_sum_full"]),
                    "best_equal_all_layer": int(best["equal_all"]["layer"]),
                    "best_equal_all_cos": float(best["equal_all"]["cos_equal_all_full"]),
                    "best_normalized_all_layer": int(best["normalized_all"]["layer"]),
                    "best_normalized_all_cos": float(best["normalized_all"]["cos_normalized_all_full"]),
                    "best_scalar_fit_layer": int(best["scalar_fit_cos"]["layer"]),
                    "best_scalar_fit_cos": float(best["scalar_fit_cos"]["scalar_fit_cos_mean"]),
                    "best_scalar_fit_r2": float(best["scalar_fit_r2"]["scalar_fit_r2_mean"]),
                    "best_oracle_span_cos": float(best["oracle_span"]["oracle_span_cos"]),
                    "best_matched_minus_shuffled": float(best["matched_minus_shuffled"]["normalized_all_matched_minus_shuffled"]),
                })

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame(compact_rows).to_csv(out_dir / "best_results.tsv", sep="\t", index=False)

        print(f"\nSaved: {out_dir / 'best_results.tsv'}")
        print(f"Saved: {out_dir / 'composition_by_layer__last__raw.csv'}")
        print(f"Saved: {out_dir / 'composition_by_layer__last__residual.csv'}")
        print(f"Saved: {out_dir / 'composition_by_layer__object__raw.csv'}")
        print(f"Saved: {out_dir / 'composition_by_layer__object__residual.csv'}")
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
