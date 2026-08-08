#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two simple COCO-two spatial prompts on:
  (1) object-token difference classification with correct image,
  (2) object-token difference classification with no image,
  (3) Correct-NoImage object residual classification,
  (4) full greedy generation accuracy with the correct image.

Prompt A (relation-first):
    Determine the spatial relation of the {subject} to the {reference} in the image.
    Answer with left, right, above, or below.

Prompt B (options-first):
    Answer with left, right, above, or below.
    Determine the spatial relation of the {subject} to the {reference} in the image.

The text is identical between correct-image and no-image conditions for a given
prompt. Only the image input is removed in the no-image condition.

Object representation at decoder block L:
    r_img(L)  = h_img,L(subject)  - h_img,L(reference)
    r_txt(L)  = h_txt,L(subject)  - h_txt,L(reference)
    r_res(L)  = r_img(L) - r_txt(L)

Held-out evaluation:
  * train_ratio defaults to 0.15
  * 5 repeated random splits
  * fit four centered mean directions on train only
  * classify test vectors by cosine similarity to left/right/above/below

Generation:
  * correct image only
  * greedy decoding
  * max_new_tokens defaults to 64 (not artificially truncated to one word)
  * parse the LAST whole-word left/right/above/below from the complete generated
    continuation, case-insensitively
  * generation ACC is parsed relation vs dataset gold relation

Designed to run in AdaptVis next to extract_two_object_relation_states.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
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
PROMPT_TYPES = ("relation_first", "options_first")


def norm_relation(x: Any) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(key, key)


def prompt_text(prompt_type: str, subject: str, reference: str) -> str:
    if prompt_type == "relation_first":
        return (
            f"Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        )
    if prompt_type == "options_first":
        return (
            "Answer with left, right, above, or below. "
            f"Determine the spatial relation of the {subject} to the {reference} "
            "in the image."
        )
    raise ValueError(prompt_type)


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
    p.add_argument(
        "--prompt-types",
        default="relation_first,options_first",
        help="Comma-separated subset of relation_first,options_first",
    )
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--quiet-generation", action="store_true")
    return p.parse_args()


def parse_prompt_types(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(vals) - set(PROMPT_TYPES))
    if bad:
        raise ValueError(f"Unknown prompt types: {bad}; allowed={PROMPT_TYPES}")
    if not vals:
        raise ValueError("No prompt types selected")
    return vals


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def build_chat_prompt(
    processor: Any,
    prompt_type: str,
    subject: str,
    reference: str,
    *,
    with_image: bool,
) -> str:
    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({
        "type": "text",
        "text": prompt_text(prompt_type, subject, reference),
    })
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
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


def extract_object_condition(
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
    vectors: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    blocks_n: Optional[int] = None
    hidden_size: Optional[int] = None

    def save_progress() -> None:
        if not vectors or blocks_n is None or hidden_size is None:
            return
        arrays = {
            "metadata_json": np.array(json.dumps({
                "model": args.model,
                "repo_id": base.SPECS[args.model].repo_id,
                "prompt_type": prompt_type,
                "prompt_template": prompt_text(prompt_type, "{subject}", "{reference}"),
                "vision_mode": mode,
                "decoder_blocks": blocks_n,
                "hidden_size": hidden_size,
                "n_saved": len(sids),
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "image_id": np.asarray(image_ids, dtype=object),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "relation": np.asarray(labels, dtype=object),
            "decoder_block_index": np.arange(blocks_n, dtype=np.int32),
            "relation_vectors": np.stack(vectors).astype(dtype_np),
        }
        atomic_save_npz(out_path, arrays)

    desc = f"{args.model}:{prompt_type}:{mode}"
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB") if with_image else None
            rendered = build_chat_prompt(
                processor,
                prompt_type,
                rec.subject,
                rec.reference,
                with_image=with_image,
            )
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()

            sidx = base.find_phrase_last_token(
                processor.tokenizer, input_ids, rec.subject
            )
            ridx = base.find_phrase_last_token(
                processor.tokenizer, input_ids, rec.reference
            )
            if sidx == ridx:
                raise RuntimeError("subject/reference token positions collide")

            with torch.inference_mode():
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                states = base.hidden_tuple(outputs)

            current_blocks = len(states) - 1
            if blocks_n is None:
                blocks_n = current_blocks
                hidden_size = int(states[-1].shape[-1])
                print(f"[{desc}] decoder_blocks={blocks_n}, hidden={hidden_size}")
            elif current_blocks != blocks_n:
                raise RuntimeError(f"decoder blocks changed {blocks_n}->{current_blocks}")

            vec = np.stack([
                (
                    states[k + 1][0, sidx] - states[k + 1][0, ridx]
                ).detach().float().cpu().numpy()
                for k in range(current_blocks)
            ], axis=0).astype(dtype_np)

            sids.append(int(rec.sid))
            image_ids.append(str(rec.image_id))
            subjects.append(str(rec.subject))
            references.append(str(rec.reference))
            labels.append(norm_relation(rec.relation))
            vectors.append(vec)

            del outputs, states, batch
            if len(vectors) % args.save_every == 0:
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
    print(f"[saved] {out_path} | n={len(vectors)}/{len(records)} | errors={len(errors)}")


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), EPS)


def fit_codebook(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    center = X.mean(axis=0)
    Xc = X - center
    dirs = []
    for rel in RELATIONS:
        m = y == rel
        if int(m.sum()) == 0:
            raise RuntimeError(f"train split has no relation={rel}")
        dirs.append(normalize(Xc[m].mean(axis=0)))
    return center, np.stack(dirs)


def make_splits(
    n: int,
    ratio: float,
    repeats: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
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


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    layers: Sequence[int],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    gt_all = np.asarray([RELATIONS.index(str(v)) for v in y], dtype=np.int64)
    per_layer: Dict[str, Any] = {}
    for li, layer in enumerate(layers):
        accs = []
        for tr, te in splits:
            center, dirs = fit_codebook(X[tr, li], y[tr])
            Xt = X[te, li] - center
            Xt = Xt / np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), EPS)
            pred = np.argmax(Xt @ dirs.T, axis=1)
            accs.append(float(np.mean(pred == gt_all[te])))
        per_layer[str(layer)] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "repeat_accuracy": accs,
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


def align_two(
    correct: Dict[str, Any],
    noimg: Dict[str, Any],
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, List[int]]:
    cpos = {int(s): i for i, s in enumerate(correct["sample_index"].tolist())}
    npos = {int(s): i for i, s in enumerate(noimg["sample_index"].tolist())}
    sids = sorted(set(cpos) & set(npos))
    if not sids:
        raise RuntimeError("no common samples between correct and no_image")
    ci = np.asarray([cpos[s] for s in sids], dtype=np.int64)
    ni = np.asarray([npos[s] for s in sids], dtype=np.int64)
    Xc = correct["relation_vectors"][ci].astype(np.float64)
    Xn = noimg["relation_vectors"][ni].astype(np.float64)
    yc = np.asarray([norm_relation(x) for x in correct["relation"][ci]], dtype=object)
    yn = np.asarray([norm_relation(x) for x in noimg["relation"][ni]], dtype=object)
    if not np.array_equal(yc, yn):
        raise RuntimeError("label mismatch between correct and no_image")
    layers = [int(x) for x in correct["decoder_block_index"].tolist()]
    if layers != [int(x) for x in noimg["decoder_block_index"].tolist()]:
        raise RuntimeError("layer mismatch")
    return sids, Xc, Xn, yc, layers


def safe_decode(tokenizer: Any, ids: Sequence[int]) -> str:
    kwargs = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(map(int, ids)), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces", None)
        return str(tokenizer.decode(list(map(int, ids)), **kwargs))


def answer_pattern() -> re.Pattern[str]:
    return re.compile(
        r"(?<![A-Za-z])(left|right|above|below)(?![A-Za-z])",
        re.IGNORECASE,
    )


def parse_generated_relation(text: str, pat: re.Pattern[str]) -> Optional[str]:
    matches = list(pat.finditer(text))
    if not matches:
        return None
    return norm_relation(matches[-1].group(1))


def generate_full_answers(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[base.Record],
    prompt_type: str,
    out_csv: Path,
) -> pd.DataFrame:
    if out_csv.exists() and not args.overwrite:
        print(f"[reuse generation] {out_csv}")
        return pd.read_csv(out_csv)

    pat = answer_pattern()
    rows: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc=f"{args.model}:{prompt_type}:generation", dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB")
            rendered = build_chat_prompt(
                processor,
                prompt_type,
                rec.subject,
                rec.reference,
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
                )

            new_ids = generated.sequences[0, input_len:].detach().cpu().tolist()
            text = safe_decode(processor.tokenizer, new_ids)
            pred = parse_generated_relation(text, pat)
            gt = norm_relation(rec.relation)
            correct = bool(pred == gt)
            row = {
                "sid": int(rec.sid),
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "gt": gt,
                "prediction": pred,
                "correct": correct,
                "answer_found": pred is not None,
                "generated_text": text,
                "generated_token_count": int(len(new_ids)),
            }
            rows.append(row)
            if not args.quiet_generation:
                print(
                    f"[{prompt_type}] sid={rec.sid} gt={gt} pred={pred} "
                    f"ok={int(correct)} | {text!r}",
                    flush=True,
                )
            del generated, batch
        except Exception as exc:
            rows.append({
                "sid": int(rec.sid),
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "gt": norm_relation(rec.relation),
                "prediction": None,
                "correct": False,
                "answer_found": False,
                "generated_text": "",
                "generated_token_count": 0,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def print_probe(name: str, r: Dict[str, Any]) -> None:
    print(
        f"{name:40s} | best=L{r['best_layer']:>2d} "
        f"acc={100*r['best_accuracy_mean']:.2f}%"
        f"±{100*r['best_accuracy_std']:.2f}% | n={r['n']}"
    )


def main() -> None:
    args = parse_args()
    prompt_types = parse_prompt_types(args.prompt_types)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0,1)")
    if args.max_new_tokens < 16:
        print(
            f"[warning] --max-new-tokens={args.max_new_tokens} is low for full-sentence "
            "generation; recommend >=64"
        )

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
    records = [r for r in records if norm_relation(r.relation) in RELATIONS]
    if not records:
        raise RuntimeError("no usable records")
    print(
        f"[{args.dataset}] n={len(records)} counts="
        f"{dict(Counter(norm_relation(r.relation) for r in records))}"
    )
    (out_dir / "dataset.audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
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

    started = time.time()
    model = None
    processor = None
    try:
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        summary: Dict[str, Any] = {
            "config": {
                "model": args.model,
                "repo_id": spec.repo_id,
                "prompt_types": prompt_types,
                "prompt_templates": {
                    p: prompt_text(p, "{subject}", "{reference}")
                    for p in prompt_types
                },
                "train_ratio": args.train_ratio,
                "repeats": args.repeats,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
            },
            "prompt_results": {},
        }

        best_rows: List[Dict[str, Any]] = []

        for ptype in prompt_types:
            print("\n" + "=" * 100)
            print(f"PROMPT TYPE: {ptype}")
            print(prompt_text(ptype, "A", "B"))
            print("=" * 100)

            pdir = out_dir / ptype
            cpath = pdir / "states" / "object__correct.npz"
            npath = pdir / "states" / "object__no_image.npz"

            extract_object_condition(
                args=args,
                model=model,
                processor=processor,
                device=device,
                records=records,
                prompt_type=ptype,
                with_image=True,
                out_path=cpath,
            )
            extract_object_condition(
                args=args,
                model=model,
                processor=processor,
                device=device,
                records=records,
                prompt_type=ptype,
                with_image=False,
                out_path=npath,
            )

            correct = load_npz(cpath)
            noimg = load_npz(npath)
            sids, Xc, Xn, y, layers = align_two(correct, noimg)
            Xr = Xc - Xn
            splits = make_splits(len(sids), args.train_ratio, args.repeats, args.seed)

            rc = evaluate(Xc, y, layers, splits)
            rn = evaluate(Xn, y, layers, splits)
            rr = evaluate(Xr, y, layers, splits)

            print("\n=== OBJECT TOKEN PROBE ===")
            print_probe("object__correct", rc)
            print_probe("object__no_image", rn)
            print_probe("object__correct_minus_noimage", rr)

            gen_csv = pdir / "generation.csv"
            gdf = generate_full_answers(
                args=args,
                model=model,
                processor=processor,
                device=device,
                records=records,
                prompt_type=ptype,
                out_csv=gen_csv,
            )
            gen_acc = float(pd.to_numeric(gdf["correct"], errors="coerce").fillna(False).mean())
            found_rate = float(pd.to_numeric(gdf["answer_found"], errors="coerce").fillna(False).mean())
            mean_tokens = float(pd.to_numeric(gdf["generated_token_count"], errors="coerce").mean())

            print("\n=== FULL GENERATION ===")
            print(
                f"generation__correct_image              | "
                f"acc={100*gen_acc:.2f}% | "
                f"answer_found={100*found_rate:.2f}% | "
                f"mean_new_tokens={mean_tokens:.1f} | "
                f"max_new_tokens={args.max_new_tokens}"
            )

            summary["prompt_results"][ptype] = {
                "object_correct": rc,
                "object_no_image": rn,
                "object_correct_minus_noimage": rr,
                "generation": {
                    "n": int(len(gdf)),
                    "accuracy": gen_acc,
                    "answer_found_rate": found_rate,
                    "mean_generated_tokens": mean_tokens,
                },
            }

            best_rows.extend([
                {
                    "prompt_type": ptype,
                    "metric": "object_correct",
                    "best_layer": rc["best_layer"],
                    "accuracy_mean": rc["best_accuracy_mean"],
                    "accuracy_std": rc["best_accuracy_std"],
                    "n": rc["n"],
                },
                {
                    "prompt_type": ptype,
                    "metric": "object_no_image",
                    "best_layer": rn["best_layer"],
                    "accuracy_mean": rn["best_accuracy_mean"],
                    "accuracy_std": rn["best_accuracy_std"],
                    "n": rn["n"],
                },
                {
                    "prompt_type": ptype,
                    "metric": "object_correct_minus_noimage",
                    "best_layer": rr["best_layer"],
                    "accuracy_mean": rr["best_accuracy_mean"],
                    "accuracy_std": rr["best_accuracy_std"],
                    "n": rr["n"],
                },
                {
                    "prompt_type": ptype,
                    "metric": "generation_correct_image",
                    "best_layer": None,
                    "accuracy_mean": gen_acc,
                    "accuracy_std": None,
                    "n": len(gdf),
                },
            ])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame(best_rows).to_csv(out_dir / "best_results.tsv", sep="\t", index=False)

        print("\n" + "=" * 100)
        print("FINAL COMPARISON")
        print("=" * 100)
        for ptype in prompt_types:
            r = summary["prompt_results"][ptype]
            print(f"\n[{ptype}]")
            print(
                f"  Object correct            : "
                f"{100*r['object_correct']['best_accuracy_mean']:.2f}% "
                f"@ L{r['object_correct']['best_layer']}"
            )
            print(
                f"  Object Correct-NoImage    : "
                f"{100*r['object_correct_minus_noimage']['best_accuracy_mean']:.2f}% "
                f"@ L{r['object_correct_minus_noimage']['best_layer']}"
            )
            print(
                f"  Full generation correct   : "
                f"{100*r['generation']['accuracy']:.2f}%"
            )

        print(f"\nSaved: {out_dir / 'best_results.tsv'}")
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
