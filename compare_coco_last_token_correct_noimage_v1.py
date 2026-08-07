#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two: compare prompt-last hidden states with image, without image, and their residual.

For each decoder block L:
    q_img(L)  = hidden state of the final prompt token with the correct image
    q_txt(L)  = hidden state of the final prompt token with no image
    q_vis(L)  = q_img(L) - q_txt(L)

Each representation is evaluated with the SAME held-out 4-way cosine-direction
codebook:
    left / right / above / below   (surface equivalents: left/right/on/under)

The direction codebook is fit on TRAIN only.  Test prediction is the relation
whose train direction has maximum cosine similarity to the centered test vector.

Example:
    CUDA_VISIBLE_DEVICES=0 python compare_coco_last_token_correct_noimage_v1.py \
      --model qwen-3b \
      --data-root data \
      --device cuda:0 \
      --train-ratio 0.15 \
      --repeats 5 \
      --output-dir output/coco_last_token_correct_noimage/qwen-3b
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
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-fp32", action="store_true")
    return p.parse_args()


def norm_relation(x: Any) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(key, key)


def prompt_text(subject: str, reference: str) -> str:
    # Keep the answer space fully fixed.  The final hidden state is the final
    # prompt token, so all instructions are already in its causal context.
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


def build_chat_prompt(processor: Any, subject: str, reference: str, *, with_image: bool) -> str:
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
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


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


def extract_states(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    records: Sequence[base.Record],
    device: torch.device,
    out_path: Path,
) -> Dict[str, Any]:
    if out_path.exists() and not args.overwrite:
        print(f"[reuse states] {out_path}")
        with np.load(out_path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    sids: List[int] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    correct: List[np.ndarray] = []
    noimage: List[np.ndarray] = []
    residual: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    decoder_blocks: Optional[int] = None
    hidden_size: Optional[int] = None

    def save_progress() -> None:
        if not residual or decoder_blocks is None or hidden_size is None:
            return
        arrays = {
            "metadata_json": np.array(json.dumps({
                "dataset": args.dataset,
                "model": args.model,
                "repo_id": base.SPECS[args.model].repo_id,
                "representation": "last_prompt_token",
                "definition": "h_last(correct_image) - h_last(no_image)",
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
            "correct_vectors": np.stack(correct).astype(dtype_np),
            "noimage_vectors": np.stack(noimage).astype(dtype_np),
            "residual_vectors": np.stack(residual).astype(dtype_np),
        }
        atomic_save_npz(out_path, arrays)

    for record in tqdm(records, desc=f"{args.model}:last-token-states", dynamic_ncols=True):
        try:
            image = Image.open(record.image_path).convert("RGB")
            by_mode: Dict[str, np.ndarray] = {}

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

                # One sample per forward, so -1 is the final non-padding prompt token.
                last_idx = int(batch["input_ids"].shape[1] - 1)

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
                    print(f"[{args.model}] decoder_blocks={decoder_blocks}, hidden={hidden_size}")
                elif blocks != decoder_blocks:
                    raise RuntimeError(
                        f"decoder block count changed: {decoder_blocks}->{blocks}"
                    )

                vec = np.stack([
                    states[k + 1][0, last_idx].detach().float().cpu().numpy()
                    for k in range(blocks)
                ], axis=0)
                by_mode[mode] = vec

                del outputs, states, batch

            q_img = by_mode["correct"]
            q_txt = by_mode["no_image"]
            q_vis = q_img - q_txt

            sids.append(int(record.sid))
            subjects.append(str(record.subject))
            references.append(str(record.reference))
            relations.append(norm_relation(record.relation))
            correct.append(q_img.astype(dtype_np))
            noimage.append(q_txt.astype(dtype_np))
            residual.append(q_vis.astype(dtype_np))

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


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), EPS)


def fit_codebook(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = X.mean(axis=0)
    Xc = X - center
    dirs = []
    for rel in RELATIONS:
        mask = y == rel
        if int(mask.sum()) == 0:
            raise RuntimeError(f"Training split has no samples for relation={rel}")
        dirs.append(normalize(Xc[mask].mean(axis=0)))
    return center, np.stack(dirs, axis=0)


def make_splits(n: int, ratio: float, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
    for rep in range(repeats):
        ids = list(range(n))
        random.Random(seed + rep).shuffle(ids)
        n_train = int(n * ratio)
        if n_train <= 0 or n_train >= n:
            raise RuntimeError(f"Invalid train size={n_train}, n={n}, ratio={ratio}")
        out.append((
            np.asarray(ids[:n_train], dtype=np.int64),
            np.asarray(ids[n_train:], dtype=np.int64),
        ))
    return out


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    layers: Sequence[int],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    gt_all = np.asarray([RELATIONS.index(v) for v in y], dtype=np.int64)
    per_layer: Dict[str, Any] = {}

    for li, layer in enumerate(layers):
        accs: List[float] = []
        margins: List[float] = []
        for train_idx, test_idx in splits:
            center, dirs = fit_codebook(X[train_idx, li, :], y[train_idx])
            Xt = X[test_idx, li, :] - center
            Xt = Xt / np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), EPS)
            score = Xt @ dirs.T
            pred = np.argmax(score, axis=1)
            gt = gt_all[test_idx]
            accs.append(float(np.mean(pred == gt)))

            gold = score[np.arange(len(gt)), gt]
            other = score.copy()
            other[np.arange(len(gt)), gt] = -np.inf
            margins.append(float(np.mean(gold - other.max(axis=1))))

        per_layer[str(layer)] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "repeat_accuracy": accs,
            "margin_mean": float(np.mean(margins)),
            "margin_std": float(np.std(margins)),
        }

    best = max(layers, key=lambda l: per_layer[str(l)]["accuracy_mean"])
    return {
        "n": int(len(y)),
        "counts": dict(Counter(y.tolist())),
        "layers": per_layer,
        "best_layer": int(best),
        "best_accuracy_mean": per_layer[str(best)]["accuracy_mean"],
        "best_accuracy_std": per_layer[str(best)]["accuracy_std"],
    }


def mean_cosine_by_layer(A: np.ndarray, B: np.ndarray, layers: Sequence[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for li, layer in enumerate(layers):
        a = A[:, li, :]
        b = B[:, li, :]
        an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)
        bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), EPS)
        cos = np.sum(an * bn, axis=1)
        nr = np.linalg.norm(A[:, li, :] - B[:, li, :], axis=1) / np.maximum(
            np.linalg.norm(A[:, li, :], axis=1), EPS
        )
        out[str(layer)] = {
            "correct_noimage_cosine_mean": float(np.mean(cos)),
            "correct_noimage_cosine_std": float(np.std(cos)),
            "residual_over_correct_norm_mean": float(np.mean(nr)),
            "residual_over_correct_norm_std": float(np.std(nr)),
        }
    return out


def print_best(name: str, result: Dict[str, Any]) -> None:
    print(
        f"{name:34s} | best=L{result['best_layer']:>2d} "
        f"acc={100*result['best_accuracy_mean']:.2f}%"
        f"±{100*result['best_accuracy_std']:.2f}% | n={result['n']}"
    )


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states_path = out_dir / "states" / "last_token_correct_noimage.npz"

    records, audit = base.load_records(args.dataset, Path(args.data_root), args.max_samples)
    records = [r for r in records if norm_relation(r.relation) in RELATIONS]
    if not records:
        raise RuntimeError("No usable records")

    (out_dir / "dataset.audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[{args.dataset}] n={len(records)} counts={dict(Counter(norm_relation(r.relation) for r in records))}")

    spec = base.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers has no {spec.model_class}")

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
    start = time.time()
    try:
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        data = extract_states(
            args=args,
            model=model,
            processor=processor,
            records=records,
            device=device,
            out_path=states_path,
        )

        y = np.asarray([norm_relation(x) for x in data["relation"].tolist()], dtype=object)
        layers = [int(x) for x in data["decoder_block_index"].tolist()]
        Xc = data["correct_vectors"].astype(np.float64)
        Xn = data["noimage_vectors"].astype(np.float64)
        Xr = data["residual_vectors"].astype(np.float64)

        splits = make_splits(len(y), args.train_ratio, args.repeats, args.seed)

        res_correct = evaluate(Xc, y, layers, splits)
        res_noimage = evaluate(Xn, y, layers, splits)
        res_residual = evaluate(Xr, y, layers, splits)

        print("\n=== LAST-TOKEN REPRESENTATIONS ===")
        print_best("last__correct", res_correct)
        print_best("last__no_image", res_noimage)
        print_best("last__correct_minus_noimage", res_residual)

        sim = mean_cosine_by_layer(Xc, Xn, layers)
        best_res_layer = res_residual["best_layer"]
        best_sim = sim[str(best_res_layer)]
        print(
            f"\nAt residual best layer L{best_res_layer}: "
            f"cos(correct,noimage)={best_sim['correct_noimage_cosine_mean']:.4f}, "
            f"||residual||/||correct||={best_sim['residual_over_correct_norm_mean']:.4f}"
        )

        summary = {
            "config": {
                "dataset": args.dataset,
                "model": args.model,
                "repo_id": spec.repo_id,
                "train_ratio": args.train_ratio,
                "repeats": args.repeats,
                "seed": args.seed,
                "n": len(y),
                "representation": "last_prompt_token",
                "residual": "correct_image - no_image",
            },
            "correct": res_correct,
            "no_image": res_noimage,
            "correct_minus_noimage": res_residual,
            "correct_vs_noimage_similarity": sim,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        rows = ["condition\tbest_layer\taccuracy_mean\taccuracy_std\tn"]
        for name, result in [
            ("last__correct", res_correct),
            ("last__no_image", res_noimage),
            ("last__correct_minus_noimage", res_residual),
        ]:
            rows.append(
                f"{name}\t{result['best_layer']}\t{result['best_accuracy_mean']:.8f}\t"
                f"{result['best_accuracy_std']:.8f}\t{result['n']}"
            )
        (out_dir / "best_results.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

        print(f"\nSaved: {out_dir / 'best_results.tsv'}")
        print(f"Elapsed: {(time.time()-start)/60:.1f} min")

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
