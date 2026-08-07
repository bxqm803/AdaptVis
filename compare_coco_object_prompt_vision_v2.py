#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare controlled prompt / vision conditions for two-object spatial relation vectors.

Designed for the bxqm803/AdaptVis `llava16` branch and intended to be placed in
that repository root next to:
    extract_two_object_relation_states.py

Core representation at decoder block L:
    r_L = h_L(subject_token) - h_L(reference_token)

Default prompt conditions all solve the same closed four-way task and use
exactly the same answer vocabulary: left / right / above / below.

    raw
        Direct spatial-relation instruction.

    grounding
        Explicitly locate both objects first, then answer the same relation task.

    ground_relation
        Explicitly locate and compare the two object positions, then answer the
        same relation task.

All task instructions are placed BEFORE the final object mentions used for
hidden-state extraction, so they can condition those object states in a causal
decoder.

Default visual conditions:
    correct   : correct image
    wrong     : a deterministic wrong image with a different image_id
    no_image  : text-only forward pass (no image placeholder / pixel values)

For every condition the script:
  1. extracts all decoder-block object-difference vectors;
  2. evaluates the same 30/70 affine direction codebook probe used by
     spatial_affine_train30_test70.py;
  3. uses identical train/test splits across all conditions;
  4. reports mean±std accuracy per layer and the best layer;
  5. additionally evaluates visual residual vectors:
         correct - no_image
         correct - wrong

The residual tests are useful for asking whether a prompt improves access to
image-conditioned spatial evidence rather than merely strengthening a textual
prior.

Example:
    CUDA_VISIBLE_DEVICES=0 python compare_coco_object_prompt_vision_v1.py \
        --dataset coco_two \
        --data-root data \
        --model qwen-3b \
        --device cuda:0 \
        --output-dir output/coco_prompt_vision/qwen-3b

Quick smoke test:
    CUDA_VISIBLE_DEVICES=0 python compare_coco_object_prompt_vision_v1.py \
        --dataset coco_two \
        --data-root data \
        --model qwen-3b \
        --device cuda:0 \
        --max-samples 40 \
        --prompt-types raw,grounding,ground_relation \
        --vision-modes correct,no_image \
        --output-dir output/coco_prompt_vision/qwen-3b_smoke \
        --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import AutoProcessor

# Reuse the exact model aliases, dataset parser, token-span logic, and backend
# compatibility helpers already used by the repository's object-diff extractor.
import extract_two_object_relation_states as base


EPS = 1e-12

PROMPT_TYPES = (
    "raw",
    "neutral",
    "grounding",
    "ground_relation",
)

VISION_MODES = (
    "correct",
    "wrong",
    "no_image",
    "blank",
)

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


def parse_csv(raw: str) -> List[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def norm_relation(x: str) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(key, key)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", required=True, choices=sorted(base.SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
        default="sdpa",
    )
    p.add_argument(
        "--prompt-types",
        default="raw,grounding,ground_relation",
        help=f"Comma-separated subset of: {','.join(PROMPT_TYPES)}",
    )
    p.add_argument(
        "--vision-modes",
        default="correct,wrong,no_image",
        help=f"Comma-separated subset of: {','.join(VISION_MODES)}",
    )
    p.add_argument(
        "--relations",
        default="left,right,above,below",
        help="Relations evaluated by the held-out affine direction codebook.",
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Store extracted vectors as float32 instead of float16.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[List[str], List[str], List[str]]:
    prompts = parse_csv(args.prompt_types)
    visions = parse_csv(args.vision_modes)
    relations = [norm_relation(x) for x in parse_csv(args.relations)]

    bad_p = sorted(set(prompts) - set(PROMPT_TYPES))
    bad_v = sorted(set(visions) - set(VISION_MODES))
    if bad_p:
        raise ValueError(f"Unknown prompt types: {bad_p}; allowed={PROMPT_TYPES}")
    if bad_v:
        raise ValueError(f"Unknown vision modes: {bad_v}; allowed={VISION_MODES}")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1).")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    return prompts, visions, relations


def prompt_text(prompt_type: str, subject: str, reference: str) -> str:
    """
    Build controlled prompts for object-difference extraction.

    Design rules:
      1. Every condition asks the SAME four-way task.
      2. Every condition uses the SAME closed answer vocabulary:
             left / right / above / below
         and requests exactly one label, with no synonyms or explanation.
      3. Prompt-specific instructions appear BEFORE the final subject/reference
         mentions. This matters for causal decoder-only VLMs: an object token
         cannot be influenced by text that occurs after that token.
      4. Each object is introduced once in an early "Objects" clause and once
         in the final role declaration. The extractor uses the LAST occurrence,
         so both extracted object states have already seen the complete task
         instruction and both object identities.

    The only intended difference among prompt types is how explicitly the model
    is asked to perform visual grounding / structured spatial comparison.
    """

    answer_rule = (
        "Choose exactly one label from: left, right, above, below. "
        "Use only these four labels; do not use synonyms such as beside, next to, "
        "over, under, or underneath. Output exactly one label and nothing else."
    )

    if prompt_type == "raw":
        instruction = (
            "Determine the spatial relation of the target object to the reference "
            "object in the image. "
        )
    elif prompt_type == "neutral":
        instruction = (
            "Inspect the image carefully, then determine the spatial relation of "
            "the target object to the reference object. "
        )
    elif prompt_type == "grounding":
        instruction = (
            "First locate both named objects in the image and identify their visual "
            "positions. Then determine the spatial relation of the target object "
            "to the reference object. "
        )
    elif prompt_type == "ground_relation":
        instruction = (
            "First locate both named objects in the image. Then compare the target "
            "object's position with the reference object's position, and determine "
            "their spatial relation. "
        )
    else:
        raise ValueError(prompt_type)

    # The final subject/reference mentions are intentionally placed after all
    # instructions. base.find_phrase_last_token() will extract these occurrences.
    return (
        instruction
        + answer_rule
        + f" Objects: {subject} and {reference}."
        + f" Target object: {subject}."
        + f" Reference object: {reference}."
    )


def build_chat_prompt(
    processor: Any,
    prompt_type: str,
    subject: str,
    reference: str,
    with_image: bool,
) -> str:
    text = prompt_text(prompt_type, subject, reference)
    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_wrong_image_map(
    records: Sequence[base.Record], seed: int
) -> Tuple[Dict[int, Path], int]:
    """Map every sample to an image with a different image_id when possible."""
    if len(records) < 2:
        raise RuntimeError("Need at least two records for --vision-modes wrong.")

    rng = random.Random(seed + 9137)
    candidates = list(records)
    rng.shuffle(candidates)

    mapping: Dict[int, Path] = {}
    fallback = 0
    for i, record in enumerate(records):
        chosen: Optional[base.Record] = None
        # Search through a deterministic cyclic order so repeated COCO image IDs
        # do not accidentally map a sample to the same underlying image.
        for step in range(1, len(candidates) + 1):
            cand = candidates[(i + step) % len(candidates)]
            if cand.image_id != record.image_id:
                chosen = cand
                break
        if chosen is None:
            fallback += 1
            chosen = candidates[(i + 1) % len(candidates)]
        mapping[record.sid] = chosen.image_path
    return mapping, fallback


def condition_npz_path(out_dir: Path, prompt_type: str, vision_mode: str) -> Path:
    return out_dir / "states" / f"{prompt_type}__{vision_mode}.npz"


def atomic_save_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def load_condition(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
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


def extract_condition(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[base.Record],
    prompt_type: str,
    vision_mode: str,
    wrong_map: Optional[Dict[int, Path]],
    out_path: Path,
) -> None:
    if out_path.exists() and not args.overwrite:
        print(f"[reuse] {out_path}")
        return

    if out_path.exists():
        out_path.unlink()

    sample_indices: List[int] = []
    image_ids: List[str] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    vectors: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []

    decoder_blocks: Optional[int] = None
    hidden_size: Optional[int] = None
    dtype_np = np.float32 if args.keep_fp32 else np.float16
    blank = Image.new("RGB", (512, 512), color=(0, 0, 0)) if vision_mode == "blank" else None

    def save_progress() -> None:
        if not vectors or decoder_blocks is None or hidden_size is None:
            return
        metadata = {
            "dataset": args.dataset,
            "model_alias": args.model,
            "repo_id": base.SPECS[args.model].repo_id,
            "prompt_type": prompt_type,
            "vision_mode": vision_mode,
            "prompt_template": prompt_text(prompt_type, "{subject}", "{reference}"),
            "decoder_blocks": decoder_blocks,
            "hidden_size": hidden_size,
            "n_requested": len(records),
            "n_saved": len(vectors),
            "seed": args.seed,
            "transformers_version": transformers.__version__,
        }
        arrays: Dict[str, Any] = {
            "metadata_json": np.array(json.dumps(metadata), dtype=object),
            "sample_index": np.asarray(sample_indices, dtype=np.int64),
            "image_id": np.asarray(image_ids, dtype=object),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "relation": np.asarray(relations, dtype=object),
            "decoder_block_index": np.arange(decoder_blocks, dtype=np.int32),
            "relation_vectors": np.stack(vectors, axis=0).astype(dtype_np),
        }
        atomic_save_npz(out_path, arrays)

    desc = f"{args.model}:{prompt_type}:{vision_mode}"
    for record in tqdm(records, desc=desc):
        try:
            if vision_mode == "correct":
                image = Image.open(record.image_path).convert("RGB")
                with_image = True
            elif vision_mode == "wrong":
                if wrong_map is None:
                    raise RuntimeError("wrong_map is required for vision_mode=wrong")
                image = Image.open(wrong_map[record.sid]).convert("RGB")
                with_image = True
            elif vision_mode == "blank":
                assert blank is not None
                image = blank.copy()
                with_image = True
            elif vision_mode == "no_image":
                image = None
                with_image = False
            else:
                raise ValueError(vision_mode)

            rendered = build_chat_prompt(
                processor,
                prompt_type,
                record.subject,
                record.reference,
                with_image=with_image,
            )
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()

            subject_index = base.find_phrase_last_token(
                processor.tokenizer, input_ids, record.subject
            )
            reference_index = base.find_phrase_last_token(
                processor.tokenizer, input_ids, record.reference
            )
            if subject_index == reference_index:
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
            if decoder_blocks is None:
                decoder_blocks = current_blocks
                hidden_size = int(final.shape[-1])
                print(
                    f"[{desc}] decoder_blocks={decoder_blocks}, hidden={hidden_size}"
                )
            elif decoder_blocks != current_blocks:
                raise RuntimeError(
                    f"Decoder block count changed: {decoder_blocks} -> {current_blocks}"
                )

            # state[0] = embedding output; state[k+1] = decoder block k output.
            relation_vector = np.stack(
                [
                    (
                        states[k + 1][0, subject_index]
                        - states[k + 1][0, reference_index]
                    )
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for k in range(current_blocks)
                ],
                axis=0,
            ).astype(dtype_np)

            sample_indices.append(record.sid)
            image_ids.append(record.image_id)
            subjects.append(record.subject)
            references.append(record.reference)
            relations.append(norm_relation(record.relation))
            vectors.append(relation_vector)

            del outputs, states, batch
            if len(vectors) % args.save_every == 0:
                save_progress()

        except Exception as exc:
            errors.append(
                {
                    "sid": record.sid,
                    "image_id": record.image_id,
                    "subject": record.subject,
                    "reference": record.reference,
                    "relation": record.relation,
                    "prompt_type": prompt_type,
                    "vision_mode": vision_mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-10:],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_progress()
    err_path = out_path.with_suffix(".errors.json")
    err_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path} | n={len(vectors)}/{len(records)} | errors={len(errors)}")


# -----------------------------------------------------------------------------
# Held-out affine direction probe (same idea as spatial_affine_train30_test70.py)
# -----------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), EPS)


def fit_codebook(
    X: np.ndarray, y: np.ndarray, relations: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    center = X.mean(axis=0)
    Xc = X - center
    dirs = []
    for relation in relations:
        mask = y == relation
        if int(mask.sum()) == 0:
            raise RuntimeError(f"Training split has no samples for relation={relation}")
        dirs.append(normalize(Xc[mask].mean(axis=0)))
    return center, np.stack(dirs, axis=0)


def predict_codebook(X: np.ndarray, center: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    Xc = X - center
    Xc = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), EPS)
    return np.argmax(Xc @ dirs.T, axis=1)


def common_sids(condition_data: Mapping[str, Dict[str, Any]]) -> List[int]:
    sets = [set(map(int, d["sample_index"].tolist())) for d in condition_data.values()]
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def align_condition(
    data: Dict[str, Any], sids: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    sid_arr = np.asarray(data["sample_index"], dtype=np.int64)
    pos = {int(s): i for i, s in enumerate(sid_arr.tolist())}
    missing = [s for s in sids if s not in pos]
    if missing:
        raise RuntimeError(f"Condition missing {len(missing)} requested common sids.")
    idx = np.asarray([pos[s] for s in sids], dtype=np.int64)
    X = data["relation_vectors"][idx].astype(np.float64)
    y = np.asarray([norm_relation(x) for x in data["relation"][idx].tolist()], dtype=object)
    layers = [int(x) for x in data["decoder_block_index"].tolist()]
    return X, y, layers


def make_shared_splits(n: int, ratio: float, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    for rep in range(repeats):
        rng = random.Random(seed + rep)
        ids = list(range(n))
        rng.shuffle(ids)
        train_n = int(n * ratio)
        if train_n <= 0 or train_n >= n:
            raise RuntimeError(f"Invalid train size {train_n} for n={n}, ratio={ratio}")
        splits.append(
            (
                np.asarray(ids[:train_n], dtype=np.int64),
                np.asarray(ids[train_n:], dtype=np.int64),
            )
        )
    return splits


def evaluate_vectors(
    X: np.ndarray,
    y: np.ndarray,
    layers: Sequence[int],
    relations: Sequence[str],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    keep = np.isin(y, relations)
    if not bool(keep.all()):
        # All conditions should share labels, but relation filtering changes the
        # sample index space. Rebuild splits outside if this path is ever used.
        raise RuntimeError(
            "Found labels outside --relations after common-sid alignment. "
            "Filter the dataset before extraction or change --relations."
        )

    gt_all = np.asarray([relations.index(v) for v in y], dtype=np.int64)
    per_layer: Dict[str, Any] = {}

    for li, layer in enumerate(layers):
        vals: List[float] = []
        margins: List[float] = []
        for train_idx, test_idx in splits:
            center, dirs = fit_codebook(X[train_idx, li, :], y[train_idx], relations)
            Xtest = X[test_idx, li, :]
            Xc = Xtest - center
            Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), EPS)
            score = Xn @ dirs.T
            pred = np.argmax(score, axis=1)
            gt = gt_all[test_idx]
            vals.append(float(np.mean(pred == gt)))

            true_score = score[np.arange(len(gt)), gt]
            tmp = score.copy()
            tmp[np.arange(len(gt)), gt] = -np.inf
            margins.append(float(np.mean(true_score - tmp.max(axis=1))))

        per_layer[str(layer)] = {
            "accuracy_mean": float(np.mean(vals)),
            "accuracy_std": float(np.std(vals)),
            "repeat_accuracy": vals,
            "margin_mean": float(np.mean(margins)),
            "margin_std": float(np.std(margins)),
        }

    best_layer = max(
        layers,
        key=lambda l: per_layer[str(l)]["accuracy_mean"],
    )
    return {
        "n": int(len(y)),
        "counts": dict(Counter(y.tolist())),
        "layers": per_layer,
        "best_layer": int(best_layer),
        "best_accuracy_mean": per_layer[str(best_layer)]["accuracy_mean"],
        "best_accuracy_std": per_layer[str(best_layer)]["accuracy_std"],
    }


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), EPS)
    bn = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), EPS)
    return np.sum(an * bn, axis=-1)


def representation_similarity_by_layer(
    X_a: np.ndarray, X_b: np.ndarray, layers: Sequence[int]
) -> Dict[str, Any]:
    if X_a.shape != X_b.shape:
        raise RuntimeError(f"Similarity shape mismatch: {X_a.shape} vs {X_b.shape}")
    out: Dict[str, Any] = {}
    for li, layer in enumerate(layers):
        cos = cosine_rows(X_a[:, li, :], X_b[:, li, :])
        out[str(layer)] = {
            "cosine_mean": float(np.mean(cos)),
            "cosine_std": float(np.std(cos)),
        }
    return out


def print_best(name: str, result: Dict[str, Any]) -> None:
    print(
        f"{name:34s} | "
        f"best=L{result['best_layer']:>2d} "
        f"acc={100.0 * result['best_accuracy_mean']:.2f}%"
        f"±{100.0 * result['best_accuracy_std']:.2f}% "
        f"| n={result['n']}"
    )


def main() -> None:
    args = parse_args()
    prompts, visions, relations = validate_args(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    # Keep only requested relation labels before extraction so every condition
    # uses exactly the same semantic task.
    records = [r for r in records if norm_relation(r.relation) in relations]
    if not records:
        raise RuntimeError("No usable records after relation filtering.")

    (out_dir / "dataset.audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[{args.dataset}] n={len(records)} | "
        f"counts={dict(Counter(norm_relation(r.relation) for r in records))}"
    )

    wrong_map = None
    wrong_fallback = 0
    if "wrong" in visions:
        wrong_map, wrong_fallback = build_wrong_image_map(records, args.seed)
        print(f"[wrong-image] fallback_same_id={wrong_fallback}")

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
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        # Extraction is condition-by-condition so we never keep all conditions'
        # hidden states in GPU/CPU memory simultaneously.
        for prompt_type in prompts:
            for vision_mode in visions:
                path = condition_npz_path(out_dir, prompt_type, vision_mode)
                extract_condition(
                    args=args,
                    model=model,
                    processor=processor,
                    device=device,
                    records=records,
                    prompt_type=prompt_type,
                    vision_mode=vision_mode,
                    wrong_map=wrong_map,
                    out_path=path,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Load extracted arrays only for the analysis stage.
        condition_data: Dict[str, Dict[str, Any]] = {}
        for prompt_type in prompts:
            for vision_mode in visions:
                name = f"{prompt_type}__{vision_mode}"
                condition_data[name] = load_condition(
                    condition_npz_path(out_dir, prompt_type, vision_mode)
                )

        sids = common_sids(condition_data)
        if not sids:
            raise RuntimeError("No samples succeeded in every requested condition.")
        print(f"[common] n={len(sids)} across {len(condition_data)} conditions")

        aligned: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]] = {
            name: align_condition(data, sids)
            for name, data in condition_data.items()
        }

        # Sanity-check labels/layers are exactly identical across conditions.
        first_name = next(iter(aligned))
        _, y0, layers0 = aligned[first_name]
        for name, (_, y, layers) in aligned.items():
            if layers != layers0:
                raise RuntimeError(f"Layer mismatch: {first_name} vs {name}")
            if not np.array_equal(y, y0):
                raise RuntimeError(f"Label mismatch: {first_name} vs {name}")

        splits = make_shared_splits(
            len(sids), args.train_ratio, args.repeats, args.seed
        )

        summary: Dict[str, Any] = {
            "config": {
                "dataset": args.dataset,
                "data_root": args.data_root,
                "model": args.model,
                "repo_id": spec.repo_id,
                "prompt_types": prompts,
                "vision_modes": visions,
                "relations": relations,
                "train_ratio": args.train_ratio,
                "repeats": args.repeats,
                "seed": args.seed,
                "n_common": len(sids),
                "wrong_image_fallback_same_id": wrong_fallback,
            },
            "conditions": {},
            "derived": {},
            "similarity": {},
        }

        print("\n=== DIRECT CONDITIONS ===")
        for name, (X, y, layers) in aligned.items():
            result = evaluate_vectors(X, y, layers, relations, splits)
            summary["conditions"][name] = result
            print_best(name, result)

        print("\n=== VISUAL RESIDUALS ===")
        for prompt_type in prompts:
            correct_name = f"{prompt_type}__correct"
            if correct_name not in aligned:
                continue
            Xc, y, layers = aligned[correct_name]

            for baseline_mode, suffix in [
                ("no_image", "correct_minus_noimage"),
                ("wrong", "correct_minus_wrong"),
                ("blank", "correct_minus_blank"),
            ]:
                base_name = f"{prompt_type}__{baseline_mode}"
                if base_name not in aligned:
                    continue
                Xb, yb, layers_b = aligned[base_name]
                if layers_b != layers or not np.array_equal(yb, y):
                    raise RuntimeError(f"Alignment mismatch for {correct_name} - {base_name}")

                Xres = Xc - Xb
                derived_name = f"{prompt_type}__{suffix}"
                result = evaluate_vectors(Xres, y, layers, relations, splits)
                summary["derived"][derived_name] = result
                print_best(derived_name, result)

                sim_name = f"{correct_name}__vs__{base_name}"
                summary["similarity"][sim_name] = representation_similarity_by_layer(
                    Xc, Xb, layers
                )

        summary_path = out_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Compact TSV makes cross-model aggregation easy later.
        tsv_path = out_dir / "best_results.tsv"
        rows = ["kind\tcondition\tbest_layer\taccuracy_mean\taccuracy_std\tn"]
        for kind in ("conditions", "derived"):
            for name, result in summary[kind].items():
                rows.append(
                    "\t".join(
                        [
                            kind,
                            name,
                            str(result["best_layer"]),
                            f"{result['best_accuracy_mean']:.8f}",
                            f"{result['best_accuracy_std']:.8f}",
                            str(result["n"]),
                        ]
                    )
                )
        tsv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        print(f"\nSaved summary: {summary_path}")
        print(f"Saved best table: {tsv_path}")
        print(f"Elapsed: {(time.time() - started) / 60.0:.1f} min")

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
