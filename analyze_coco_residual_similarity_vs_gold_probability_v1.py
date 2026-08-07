#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two: test whether Correct-NoImage object-pair residual similarity tracks
LM probability of the GOLD spatial answer at the located answer-generation step.

Core representation at decoder block L:
    r_img(L)  = h_img,L(subject)  - h_img,L(reference)
    r_text(L) = h_text,L(subject) - h_text,L(reference)
    r_vis(L)  = r_img(L) - r_text(L)

For each train/test split and each decoder block L:
  1) fit a 4-way direction codebook on TRAIN residuals only;
  2) for each TEST sample compute cosine to its GOLD direction and the
     gold-vs-best-other cosine margin;
  3) run greedy generation on the SAME image+question;
  4) find the LAST whole-word occurrence of left/right/on/under,
     case-insensitively;
  5) at the generation step where that answer word begins, read the LM
     probability of the GOLD answer category, not merely the realized token.

Internal dataset labels:
    left, right, above, below
Generation surface labels:
    left, right, on, under
with above->on and below->under.

Gold answer probability is reported in several forms:
  * lm_gt_sum_probability:
      sum of full-vocabulary probabilities over all valid one-token variants of
      the gold word, including lowercase/Capitalized/UPPERCASE and common token
      boundary/punctuation forms;
  * lm_gt_max_variant_probability:
      max probability over those variants;
  * lm_gt_conditional_probability:
      gold category mass normalized over the four answer-category masses;
  * lm_gt_logit_margin:
      best gold-variant logit minus best competing-category variant logit.

Main correlation output is held-out, layer-by-layer:
    residual_gold_cosine  vs lm_gt_sum_probability
and also Spearman/Pearson for the conditional probability and margins.

The script saves extracted states so reruns can skip hidden-state extraction.

Example:
    CUDA_VISIBLE_DEVICES=0 python analyze_coco_residual_similarity_vs_gold_probability_v1.py \
      --model qwen-7b \
      --data-root data \
      --device cuda:0 \
      --train-ratio 0.15 \
      --repeats 5 \
      --output-dir output/coco_residual_vs_gold_prob/qwen-7b
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None


EPS = 1e-12
INTERNAL_RELATIONS = ("left", "right", "above", "below")
INTERNAL_TO_INDEX = {r: i for i, r in enumerate(INTERNAL_RELATIONS)}
SURFACE_RELATIONS = ("left", "right", "on", "under")
SURFACE_TO_INDEX = {r: i for i, r in enumerate(SURFACE_RELATIONS)}
INTERNAL_TO_SURFACE = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
SURFACE_TO_INTERNAL = {
    "left": "left",
    "right": "right",
    "on": "above",
    "under": "below",
}
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

# Used only for answer-step candidate scoring. Each candidate must tokenize as
# exactly one token; duplicated token ids are removed per relation.
PREFIXES = ("", " ", "\n", "\n\n", "\t")
SUFFIXES = ("", ".", ",", ":", ";", "!", "?", ")", "]")
CASE_MODES = ("lower", "capitalized", "upper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", required=True, choices=sorted(base.SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
        default="sdpa",
    )
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite-states", action="store_true")
    p.add_argument("--overwrite-generation", action="store_true")
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--quiet-samples", action="store_true")
    return p.parse_args()


def norm_relation(x: Any) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(key, key)


def prompt_text(subject: str, reference: str) -> str:
    # All task instructions precede the final object mentions so the extracted
    # object tokens can causally depend on the entire task definition.
    return (
        "Determine the spatial relation of the target object to the reference "
        "object in the image. "
        "Choose exactly one label from: left, right, on, under. "
        "Use 'on' for vertically above and 'under' for vertically below. "
        "Output exactly one label and nothing else. "
        f"Objects: {subject} and {reference}. "
        f"Target object: {subject}. "
        f"Reference object: {reference}."
    )


def build_chat_prompt(
    processor: Any,
    subject: str,
    reference: str,
    *,
    with_image: bool,
) -> str:
    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text(subject, reference)})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


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


def safe_decode(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool,
) -> str:
    kwargs = {
        "skip_special_tokens": skip_special_tokens,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(map(int, token_ids)), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces", None)
        return str(tokenizer.decode(list(map(int, token_ids)), **kwargs))


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"] if isinstance(enc, Mapping) else enc.input_ids
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def cased_words(word: str) -> List[str]:
    vals = [word.lower(), word.capitalize(), word.upper()]
    return list(dict.fromkeys(vals))


def build_variant_bank(
    tokenizer: Any,
    vocab_size: int,
) -> Tuple[Dict[str, List[int]], List[Dict[str, Any]]]:
    bank: Dict[str, List[int]] = {}
    inventory: List[Dict[str, Any]] = []
    unk = getattr(tokenizer, "unk_token_id", None)

    for relation in SURFACE_RELATIONS:
        ids: List[int] = []
        for cased in cased_words(relation):
            for prefix in PREFIXES:
                for suffix in SUFFIXES:
                    surface = f"{prefix}{cased}{suffix}"
                    encoded = tokenizer_ids(tokenizer, surface)
                    valid = len(encoded) == 1
                    token_id: Optional[int] = None
                    decoded: Optional[str] = None
                    if valid:
                        cand = int(encoded[0])
                        if (
                            0 <= cand < vocab_size
                            and (unk is None or cand != int(unk))
                        ):
                            token_id = cand
                            ids.append(cand)
                            decoded = safe_decode(
                                tokenizer, [cand], skip_special_tokens=False
                            )
                        else:
                            valid = False
                    inventory.append({
                        "relation": relation,
                        "surface_repr": repr(surface),
                        "one_token": bool(valid and token_id is not None),
                        "token_id": token_id,
                        "decoded_token": decoded,
                        "encoded_ids": " ".join(map(str, encoded)),
                    })
        bank[relation] = list(dict.fromkeys(ids))
        if not bank[relation]:
            raise RuntimeError(
                f"No valid one-token variants found for relation={relation}"
            )
    return bank, inventory


def compile_answer_pattern() -> re.Pattern[str]:
    body = "|".join(sorted(SURFACE_RELATIONS, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])({body})(?![A-Za-z])", re.IGNORECASE)


def locate_last_answer_token(
    tokenizer: Any,
    generated_ids: Sequence[int],
    pattern: re.Pattern[str],
) -> Optional[Dict[str, Any]]:
    token_ids = [int(x) for x in generated_ids]
    prefixes = [""]
    for end in range(1, len(token_ids) + 1):
        prefixes.append(
            safe_decode(
                tokenizer,
                token_ids[:end],
                skip_special_tokens=True,
            )
        )
    text = prefixes[-1]
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    char_start, char_end = int(m.start()), int(m.end())

    token_start = None
    for i in range(len(token_ids)):
        if len(prefixes[i + 1]) > char_start:
            token_start = i
            break
    if token_start is None:
        return None

    token_end = None
    for i in range(token_start, len(token_ids)):
        if len(prefixes[i + 1]) >= char_end:
            token_end = i
            break
    if token_end is None:
        return None

    surface = str(m.group(1))
    surface_lower = surface.lower()
    return {
        "generated_text": text,
        "answer_surface": surface,
        "answer_surface_lower": surface_lower,
        "generation_prediction": SURFACE_TO_INTERNAL[surface_lower],
        "answer_token_start_0based": int(token_start),
        "answer_token_end_0based": int(token_end),
        "answer_token_start_1based": int(token_start + 1),
        "answer_token_end_1based": int(token_end + 1),
        "answer_is_single_token": bool(token_start == token_end),
        "answer_token_ids": token_ids[token_start : token_end + 1],
        "answer_token_piece": safe_decode(
            tokenizer,
            token_ids[token_start : token_end + 1],
            skip_special_tokens=False,
        ),
        "generated_token_count": int(len(token_ids)),
    }


def score_relation_categories_at_step(
    logits: torch.Tensor,
    bank: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    logits = logits.float()
    full_probs = torch.softmax(logits, dim=-1)
    relation_max_logits: List[torch.Tensor] = []
    relation_max_probs: List[torch.Tensor] = []
    relation_sum_probs: List[torch.Tensor] = []
    relation_best_ids: List[int] = []

    for relation in SURFACE_RELATIONS:
        ids = torch.as_tensor(
            list(bank[relation]), dtype=torch.long, device=logits.device
        )
        rel_logits = logits.index_select(0, ids)
        rel_probs = full_probs.index_select(0, ids)
        best_i = int(torch.argmax(rel_logits).item())
        relation_max_logits.append(rel_logits[best_i])
        relation_max_probs.append(rel_probs[best_i])
        relation_sum_probs.append(rel_probs.sum())
        relation_best_ids.append(int(ids[best_i].item()))

    max_logits = torch.stack(relation_max_logits)
    max_probs = torch.stack(relation_max_probs)
    sum_probs = torch.stack(relation_sum_probs)
    conditional_sum = sum_probs / sum_probs.sum().clamp_min(1e-30)
    pred_i = int(torch.argmax(max_logits).item())

    return {
        "max_logits": max_logits.detach().cpu().numpy(),
        "max_probs": max_probs.detach().cpu().numpy(),
        "sum_probs": sum_probs.detach().cpu().numpy(),
        "conditional_sum": conditional_sum.detach().cpu().numpy(),
        "best_ids": relation_best_ids,
        "answer_step_prediction": SURFACE_RELATIONS[pred_i],
    }


def realized_token_probability(
    score_steps: Sequence[torch.Tensor],
    generated_ids: Sequence[int],
    token_start: int,
) -> Tuple[float, float, int]:
    scores = score_steps[token_start][0].float()
    token_id = int(generated_ids[token_start])
    log_probs = torch.log_softmax(scores, dim=-1)
    logp = float(log_probs[token_id].item())
    p = float(math.exp(logp))
    rank = int(torch.sum(scores > scores[token_id]).item()) + 1
    return p, logp, rank


def extract_residual_states(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    records: Sequence[base.Record],
    device: torch.device,
    out_path: Path,
) -> Dict[str, Any]:
    if out_path.exists() and not args.overwrite_states:
        print(f"[reuse states] {out_path}")
        with np.load(out_path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    sids: List[int] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    correct_vecs: List[np.ndarray] = []
    noimg_vecs: List[np.ndarray] = []
    residual_vecs: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    decoder_blocks: Optional[int] = None
    hidden_size: Optional[int] = None

    def save_progress() -> None:
        if not residual_vecs or decoder_blocks is None or hidden_size is None:
            return
        arrays = {
            "metadata_json": np.array(json.dumps({
                "dataset": args.dataset,
                "model": args.model,
                "repo_id": base.SPECS[args.model].repo_id,
                "prompt_template": prompt_text("{subject}", "{reference}"),
                "decoder_blocks": decoder_blocks,
                "hidden_size": hidden_size,
                "n_saved": len(sids),
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "relation": np.asarray(relations, dtype=object),
            "decoder_block_index": np.arange(decoder_blocks, dtype=np.int32),
            "correct_vectors": np.stack(correct_vecs).astype(dtype_np),
            "noimage_vectors": np.stack(noimg_vecs).astype(dtype_np),
            "residual_vectors": np.stack(residual_vecs).astype(dtype_np),
        }
        atomic_save_npz(out_path, arrays)

    for record in tqdm(records, desc=f"{args.model}:residual-states", dynamic_ncols=True):
        try:
            image = Image.open(record.image_path).convert("RGB")
            vec_by_mode: Dict[str, np.ndarray] = {}
            for mode in ("correct", "no_image"):
                with_image = mode == "correct"
                rendered = build_chat_prompt(
                    processor,
                    record.subject,
                    record.reference,
                    with_image=with_image,
                )
                batch = process_inputs(
                    processor,
                    rendered,
                    image if with_image else None,
                    device,
                )
                input_ids = batch["input_ids"][0].detach().cpu().tolist()
                subject_idx = base.find_phrase_last_token(
                    processor.tokenizer, input_ids, record.subject
                )
                reference_idx = base.find_phrase_last_token(
                    processor.tokenizer, input_ids, record.reference
                )
                if subject_idx == reference_idx:
                    raise RuntimeError("subject/reference token positions collide")

                with torch.inference_mode():
                    outputs = model(
                        **batch,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    states = base.hidden_tuple(outputs)

                blocks = len(states) - 1
                if decoder_blocks is None:
                    decoder_blocks = blocks
                    hidden_size = int(states[-1].shape[-1])
                    print(
                        f"[{args.model}] decoder_blocks={decoder_blocks}, hidden={hidden_size}"
                    )
                elif blocks != decoder_blocks:
                    raise RuntimeError(
                        f"decoder block count changed: {decoder_blocks}->{blocks}"
                    )

                vec = np.stack([
                    (
                        states[k + 1][0, subject_idx]
                        - states[k + 1][0, reference_idx]
                    ).detach().float().cpu().numpy()
                    for k in range(blocks)
                ], axis=0)
                vec_by_mode[mode] = vec
                del outputs, states, batch

            corr = vec_by_mode["correct"]
            noimg = vec_by_mode["no_image"]
            residual = corr - noimg
            sids.append(int(record.sid))
            subjects.append(str(record.subject))
            references.append(str(record.reference))
            relations.append(norm_relation(record.relation))
            correct_vecs.append(corr.astype(dtype_np))
            noimg_vecs.append(noimg.astype(dtype_np))
            residual_vecs.append(residual.astype(dtype_np))

            if len(sids) % args.save_every == 0:
                save_progress()

        except Exception as exc:
            errors.append({
                "sid": int(record.sid),
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
    with np.load(out_path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def generate_probabilities(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    records_by_sid: Mapping[int, base.Record],
    sids: Sequence[int],
    device: torch.device,
    out_csv: Path,
) -> pd.DataFrame:
    if out_csv.exists() and not args.overwrite_generation:
        print(f"[reuse generation] {out_csv}")
        return pd.read_csv(out_csv)

    emb = model.get_output_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        raise RuntimeError("Model has no usable output embedding / LM head")
    vocab_size = int(emb.weight.shape[0])
    bank, inventory = build_variant_bank(processor.tokenizer, vocab_size)
    pd.DataFrame(inventory).to_csv(
        out_csv.parent / "token_variant_inventory.csv", index=False
    )
    pattern = compile_answer_pattern()

    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for sid in tqdm(sids, desc=f"{args.model}:generation", dynamic_ncols=True):
        record = records_by_sid[int(sid)]
        gt = norm_relation(record.relation)
        gt_surface = INTERNAL_TO_SURFACE[gt]
        try:
            image = Image.open(record.image_path).convert("RGB")
            rendered = build_chat_prompt(
                processor,
                record.subject,
                record.reference,
                with_image=True,
            )
            batch = process_inputs(processor, rendered, image, device)
            input_len = int(batch["input_ids"].shape[1])

            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = [
                int(x)
                for x in generated.sequences[0, input_len:].detach().cpu().tolist()
            ]
            score_steps = list(generated.scores)
            if len(score_steps) != len(generated_ids):
                n = min(len(score_steps), len(generated_ids))
                score_steps = score_steps[:n]
                generated_ids = generated_ids[:n]

            generated_text = safe_decode(
                processor.tokenizer,
                generated_ids,
                skip_special_tokens=True,
            )
            located = locate_last_answer_token(
                processor.tokenizer, generated_ids, pattern
            )
            if located is None:
                row = {
                    "sid": int(sid),
                    "gt": gt,
                    "gt_surface": gt_surface,
                    "answer_found": False,
                    "generated_text": generated_text,
                    "generation_prediction": None,
                    "generation_correct": False,
                }
                rows.append(row)
                missing.append(row)
                if not args.quiet_samples:
                    print(
                        f"[{args.model}] sid={sid} ANSWER NOT FOUND | {generated_text!r}",
                        flush=True,
                    )
                continue

            step = int(located["answer_token_start_0based"])
            logits = score_steps[step][0]
            scored = score_relation_categories_at_step(logits, bank)
            gt_i = SURFACE_TO_INDEX[gt_surface]
            sum_probs = np.asarray(scored["sum_probs"], dtype=np.float64)
            max_probs = np.asarray(scored["max_probs"], dtype=np.float64)
            max_logits = np.asarray(scored["max_logits"], dtype=np.float64)
            cond = np.asarray(scored["conditional_sum"], dtype=np.float64)

            other_logits = np.delete(max_logits, gt_i)
            gt_margin = float(max_logits[gt_i] - np.max(other_logits))
            realized_p, realized_logp, realized_rank = realized_token_probability(
                score_steps, generated_ids, step
            )
            pred = str(located["generation_prediction"])
            correct = bool(pred == gt)

            row: Dict[str, Any] = {
                "sid": int(sid),
                "gt": gt,
                "gt_surface": gt_surface,
                "subject": str(record.subject),
                "reference": str(record.reference),
                "answer_found": True,
                "generated_text": generated_text,
                "generation_prediction": pred,
                "generation_correct": correct,
                "answer_surface": located["answer_surface"],
                "answer_token_start_1based": located["answer_token_start_1based"],
                "answer_is_single_token": located["answer_is_single_token"],
                "realized_answer_token_probability": realized_p,
                "realized_answer_token_logprob": realized_logp,
                "realized_answer_token_rank": realized_rank,
                "lm_gt_sum_probability": float(sum_probs[gt_i]),
                "lm_gt_max_variant_probability": float(max_probs[gt_i]),
                "lm_gt_conditional_probability": float(cond[gt_i]),
                "lm_gt_logit_margin": gt_margin,
                "answer_step_prediction": str(scored["answer_step_prediction"]),
            }
            for i, rel in enumerate(SURFACE_RELATIONS):
                row[f"lm_sum_prob_{rel}"] = float(sum_probs[i])
                row[f"lm_max_prob_{rel}"] = float(max_probs[i])
                row[f"lm_cond_prob_{rel}"] = float(cond[i])
                row[f"lm_max_logit_{rel}"] = float(max_logits[i])
            rows.append(row)

            if not args.quiet_samples:
                print(
                    f"[{args.model}] sid={sid:4d} gt={gt_surface:5s} "
                    f"gen={located['answer_surface']!r:9s} "
                    f"Pgt={sum_probs[gt_i]:.6g} "
                    f"Pgt|4={cond[gt_i]:.6f} "
                    f"correct={correct}",
                    flush=True,
                )

            del generated, batch

        except Exception as exc:
            row = {
                "sid": int(sid),
                "gt": gt,
                "gt_surface": gt_surface,
                "answer_found": False,
                "generation_prediction": None,
                "generation_correct": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            rows.append(row)
            missing.append(row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(rows) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    pd.DataFrame(missing).to_csv(
        out_csv.parent / "missing_answer_samples.csv", index=False
    )
    return frame


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), EPS)


def fit_codebook(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    center = X.mean(axis=0)
    Xc = X - center
    dirs: List[np.ndarray] = []
    for rel in INTERNAL_RELATIONS:
        mask = y == rel
        if int(mask.sum()) == 0:
            raise RuntimeError(f"Training split has no samples for {rel}")
        dirs.append(normalize(Xc[mask].mean(axis=0)))
    return center, np.stack(dirs, axis=0)


def make_splits(
    n: int,
    ratio: float,
    repeats: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for rep in range(repeats):
        rng = random.Random(seed + rep)
        ids = list(range(n))
        rng.shuffle(ids)
        train_n = int(n * ratio)
        if train_n <= 0 or train_n >= n:
            raise RuntimeError(f"Invalid train_n={train_n} for n={n}")
        splits.append((
            np.asarray(ids[:train_n], dtype=np.int64),
            np.asarray(ids[train_n:], dtype=np.int64),
        ))
    return splits


def finite_pair(x: Iterable[Any], y: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def corr_stats(x: Iterable[Any], y: Iterable[Any]) -> Dict[str, float]:
    xa, ya = finite_pair(x, y)
    if len(xa) < 3 or np.std(xa) <= 0 or np.std(ya) <= 0:
        return {
            "n": float(len(xa)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    if pearsonr is not None and spearmanr is not None:
        pr = pearsonr(xa, ya)
        sr = spearmanr(xa, ya)
        return {
            "n": float(len(xa)),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
        }
    return {
        "n": float(len(xa)),
        "pearson_r": float(np.corrcoef(xa, ya)[0, 1]),
        "pearson_p": np.nan,
        "spearman_r": float(pd.Series(xa).corr(pd.Series(ya), method="spearman")),
        "spearman_p": np.nan,
    }


def analyze_heldout(
    *,
    X: np.ndarray,
    y: np.ndarray,
    layers: Sequence[int],
    sids: Sequence[int],
    generation: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gen = generation.copy()
    gen["sid"] = gen["sid"].astype(int)
    gen_by_sid = {int(row.sid): row for row in gen.itertuples(index=False)}
    gt_idx_all = np.asarray([INTERNAL_TO_INDEX[str(v)] for v in y], dtype=np.int64)

    sample_rows: List[Dict[str, Any]] = []
    corr_rows: List[Dict[str, Any]] = []
    acc_rows: List[Dict[str, Any]] = []

    y_targets = [
        "lm_gt_sum_probability",
        "lm_gt_max_variant_probability",
        "lm_gt_conditional_probability",
        "lm_gt_logit_margin",
    ]

    for rep, (train_idx, test_idx) in enumerate(splits):
        for li, layer in enumerate(layers):
            center, dirs = fit_codebook(X[train_idx, li, :], y[train_idx])
            Xt = X[test_idx, li, :]
            Xc = Xt - center
            Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), EPS)
            scores = Xn @ dirs.T
            gt_idx = gt_idx_all[test_idx]
            pred_idx = np.argmax(scores, axis=1)
            gt_score = scores[np.arange(len(test_idx)), gt_idx]
            other = scores.copy()
            other[np.arange(len(test_idx)), gt_idx] = -np.inf
            gt_margin = gt_score - np.max(other, axis=1)

            acc_rows.append({
                "repeat": rep,
                "layer": int(layer),
                "n_test": int(len(test_idx)),
                "residual_accuracy": float(np.mean(pred_idx == gt_idx)),
                "residual_gt_cosine_mean": float(np.mean(gt_score)),
                "residual_gt_margin_mean": float(np.mean(gt_margin)),
            })

            local_rows: List[Dict[str, Any]] = []
            for j, global_i in enumerate(test_idx.tolist()):
                sid = int(sids[global_i])
                g = gen_by_sid.get(sid)
                row = {
                    "repeat": rep,
                    "layer": int(layer),
                    "sid": sid,
                    "gt": str(y[global_i]),
                    "residual_prediction": INTERNAL_RELATIONS[int(pred_idx[j])],
                    "residual_correct": bool(pred_idx[j] == gt_idx[j]),
                    "residual_gt_cosine": float(gt_score[j]),
                    "residual_gt_margin": float(gt_margin[j]),
                }
                if g is None:
                    row["answer_found"] = False
                else:
                    for name in [
                        "answer_found",
                        "generation_prediction",
                        "generation_correct",
                        "answer_surface",
                        "answer_token_start_1based",
                        "realized_answer_token_probability",
                        "lm_gt_sum_probability",
                        "lm_gt_max_variant_probability",
                        "lm_gt_conditional_probability",
                        "lm_gt_logit_margin",
                    ]:
                        row[name] = getattr(g, name, np.nan)
                sample_rows.append(row)
                local_rows.append(row)

            local = pd.DataFrame(local_rows)
            found = local[local["answer_found"] == True].copy()  # noqa: E712
            groups = {
                "all": found,
                "gen_correct": found[found["generation_correct"] == True],  # noqa: E712
                "gen_wrong": found[found["generation_correct"] == False],  # noqa: E712
            }
            for group_name, group_frame in groups.items():
                for x_name in ("residual_gt_cosine", "residual_gt_margin"):
                    for y_name in y_targets:
                        stats = (
                            corr_stats(group_frame[x_name], group_frame[y_name])
                            if len(group_frame)
                            else corr_stats([], [])
                        )
                        corr_rows.append({
                            "repeat": rep,
                            "layer": int(layer),
                            "group": group_name,
                            "x_metric": x_name,
                            "y_metric": y_name,
                            **stats,
                        })

    sample_df = pd.DataFrame(sample_rows)
    corr_df = pd.DataFrame(corr_rows)
    acc_df = pd.DataFrame(acc_rows)

    sample_df.to_csv(out_dir / "heldout_sample_metrics.csv", index=False)
    corr_df.to_csv(out_dir / "correlation_by_repeat_layer.csv", index=False)
    acc_df.to_csv(out_dir / "residual_accuracy_by_repeat_layer.csv", index=False)

    # Aggregate across repeat-level correlations. Fisher-z is not necessary for
    # this diagnostic; report simple mean/std to match the rest of the pipeline.
    agg = (
        corr_df.groupby(["layer", "group", "x_metric", "y_metric"], as_index=False)
        .agg(
            n_mean=("n", "mean"),
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            spearman_r_mean=("spearman_r", "mean"),
            spearman_r_std=("spearman_r", "std"),
        )
    )
    agg.to_csv(out_dir / "correlation_summary.csv", index=False)

    acc_agg = (
        acc_df.groupby("layer", as_index=False)
        .agg(
            residual_accuracy_mean=("residual_accuracy", "mean"),
            residual_accuracy_std=("residual_accuracy", "std"),
            residual_gt_cosine_mean=("residual_gt_cosine_mean", "mean"),
            residual_gt_margin_mean=("residual_gt_margin_mean", "mean"),
        )
    )
    acc_agg.to_csv(out_dir / "residual_accuracy_summary.csv", index=False)
    return sample_df, agg, acc_agg


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if args.repeats < 1:
        raise ValueError("--repeats must be >=1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >=1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states_path = out_dir / "states" / "correct_minus_noimage.npz"
    generation_csv = out_dir / "generation_probabilities.csv"

    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    records = [
        r for r in records
        if norm_relation(r.relation) in INTERNAL_RELATIONS
    ]
    if not records:
        raise RuntimeError("No usable records")
    records_by_sid = {int(r.sid): r for r in records}
    (out_dir / "dataset.audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[{args.dataset}] n={len(records)} counts="
        f"{dict(Counter(norm_relation(r.relation) for r in records))}"
    )

    spec = base.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    model = None
    processor = None
    started = time.time()
    try:
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            for name in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, name):
                    setattr(generation_config, name, None)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        states = extract_residual_states(
            args=args,
            model=model,
            processor=processor,
            records=records,
            device=device,
            out_path=states_path,
        )
        sids = [int(x) for x in np.asarray(states["sample_index"]).tolist()]
        y = np.asarray(
            [norm_relation(x) for x in states["relation"].tolist()],
            dtype=object,
        )
        X = np.asarray(states["residual_vectors"], dtype=np.float64)
        layers = [int(x) for x in states["decoder_block_index"].tolist()]

        generation = generate_probabilities(
            args=args,
            model=model,
            processor=processor,
            records_by_sid=records_by_sid,
            sids=sids,
            device=device,
            out_csv=generation_csv,
        )

        splits = make_splits(
            len(sids), args.train_ratio, args.repeats, args.seed
        )
        sample_df, corr_summary, acc_summary = analyze_heldout(
            X=X,
            y=y,
            layers=layers,
            sids=sids,
            generation=generation,
            splits=splits,
            out_dir=out_dir,
        )

        found = generation[generation["answer_found"] == True].copy()  # noqa: E712
        gen_acc = (
            float(found["generation_correct"].astype(float).mean())
            if len(found) else float("nan")
        )
        print("\n=== GENERATION ===")
        print(
            f"answer_found={len(found)}/{len(generation)} "
            f"generation_acc(found)={100.0*gen_acc:.2f}%"
        )
        if len(found):
            print(
                f"mean P(gold full-vocab mass)="
                f"{found['lm_gt_sum_probability'].mean():.6f}"
            )
            print(
                f"mean P(gold | four relations)="
                f"{found['lm_gt_conditional_probability'].mean():.6f}"
            )

        best_acc = acc_summary.loc[
            acc_summary["residual_accuracy_mean"].idxmax()
        ]
        print("\n=== RESIDUAL DIRECTION ACC ===")
        print(
            f"best=L{int(best_acc['layer'])} "
            f"acc={100.0*best_acc['residual_accuracy_mean']:.2f}%"
        )

        main_corr = corr_summary[
            (corr_summary["group"] == "all")
            & (corr_summary["x_metric"] == "residual_gt_cosine")
            & (corr_summary["y_metric"] == "lm_gt_sum_probability")
        ].copy()
        if len(main_corr):
            best_s = main_corr.loc[main_corr["spearman_r_mean"].abs().idxmax()]
            best_p = main_corr.loc[main_corr["pearson_r_mean"].abs().idxmax()]
            print("\n=== MAIN CORRELATION: residual gold cosine vs P(gold) ===")
            print(
                f"max |Spearman|: L{int(best_s['layer'])} "
                f"rho={best_s['spearman_r_mean']:.4f} "
                f"±{best_s['spearman_r_std']:.4f}"
            )
            print(
                f"max |Pearson| : L{int(best_p['layer'])} "
                f"r={best_p['pearson_r_mean']:.4f} "
                f"±{best_p['pearson_r_std']:.4f}"
            )
            # Also print correlation at the best residual-accuracy layer.
            at_best = main_corr[main_corr["layer"] == int(best_acc["layer"])]
            if len(at_best):
                row = at_best.iloc[0]
                print(
                    f"at best-ACC L{int(best_acc['layer'])}: "
                    f"Pearson={row['pearson_r_mean']:.4f}, "
                    f"Spearman={row['spearman_r_mean']:.4f}"
                )

        print("\nSaved:")
        for name in [
            "generation_probabilities.csv",
            "heldout_sample_metrics.csv",
            "correlation_by_repeat_layer.csv",
            "correlation_summary.csv",
            "residual_accuracy_summary.csv",
            "token_variant_inventory.csv",
        ]:
            print("  ", out_dir / name)
        print(f"Elapsed: {(time.time()-started)/60.0:.1f} min")

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
