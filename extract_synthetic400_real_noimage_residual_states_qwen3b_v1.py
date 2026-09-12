#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_synthetic400_real_noimage_residual_states_qwen3b_v1.py

Extract object-relation hidden states from the existing 400-image synthetic
shapes source dataset and save a REAL-NoImage residual NPZ compatible with the
current COCO spatial-state files.

Source dataset expected by default
----------------------------------
    synthetic_shapes_4dir_400/
        labels.jsonl
        ... images ...

Each labels.jsonl row is expected to contain:
    id, image, subject, reference, relation

Synthetic relation mapping:
    left  -> left
    right -> right
    on    -> above
    above -> above
    under -> below
    below -> below

Representation
--------------
At decoder block L:

    pair_REAL(L)    = h_REAL[L, subject_last_token]
                      - h_REAL[L, reference_last_token]

    pair_NOIMAGE(L) = h_NOIMAGE[L, subject_last_token]
                      - h_NOIMAGE[L, reference_last_token]

    q_L = pair_REAL(L) - pair_NOIMAGE(L)

The last-token object locator and decoder-block indexing intentionally match the
current COCO extractor convention:
    hidden_states[0]   = embedding output
    hidden_states[L+1] = decoder block L output

Outputs
-------
<output-dir>/states/raw__correct.npz
<output-dir>/states/raw__no_image.npz
<output-dir>/states/raw__correct_minus_noimage.npz
<output-dir>/summary.json
<output-dir>/errors.correct.json
<output-dir>/errors.no_image.json

The residual NPZ contains the keys expected by
`eval_nonoracle_synthetic400_to_coco440_spatial_repair_v1.py`:
    sample_index
    image_id
    subject
    reference
    relation
    decoder_block_index
    relation_vectors

Default extraction stores only L20-L26 because those are the current spatial
control layers.  Use `--layers all` to save every decoder block.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u \
  extract_synthetic400_real_noimage_residual_states_qwen3b_v1.py \
  --model qwen-3b \
  --synthetic-dir synthetic_shapes_4dir_400 \
  --layers 20-26 \
  --output-dir output/qwen3b_synthetic400_spatial_real_noimage_v1 \
  --overwrite
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

try:
    import extract_two_object_relation_states as base
except Exception as exc:
    raise SystemExit(
        "Could not import extract_two_object_relation_states.py.\n"
        "Run this script from the AdaptVis llava16 repository root.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
SYN_REL_MAP = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
}

SYNTHETIC_PROMPT = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with left, right, above, or below."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b", choices=sorted(base.SPECS))
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument(
        "--synthetic-labels",
        default=None,
        help="Optional explicit labels.jsonl. Default: <synthetic-dir>/labels.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--layers",
        default="20-26",
        help="Decoder blocks to save, e.g. 20-26, 20,22,24,26, or all.",
    )
    p.add_argument("--max-samples", type=int, default=0, help="0 = all synthetic rows")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Store vectors as float32. Default stores float16, like the COCO extractor.",
    )
    return p.parse_args()


def norm_rel(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in SYN_REL_MAP:
        raise ValueError(f"Unsupported synthetic relation {value!r}")
    return SYN_REL_MAP[key]


def parse_layers(text: str, n_blocks: int) -> List[int]:
    raw = str(text).strip().lower()
    if raw in {"all", "*"}:
        return list(range(n_blocks))
    out = set()
    for part in raw.split(","):
        part = part.strip().replace("l", "")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    layers = sorted(out)
    if not layers:
        raise ValueError(f"No layers parsed from {text!r}")
    bad = [L for L in layers if L < 0 or L >= n_blocks]
    if bad:
        raise ValueError(f"Requested invalid decoder blocks {bad}; model has 0..{n_blocks-1}")
    return layers


def atomic_save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_synthetic_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.synthetic_dir)
    labels = Path(args.synthetic_labels) if args.synthetic_labels else root / "labels.jsonl"
    if not labels.exists():
        raise FileNotFoundError(f"Synthetic labels not found: {labels}")

    rows: List[Dict[str, Any]] = []
    with labels.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            relation = norm_rel(item["relation"])
            subject = str(item["subject"]).strip()
            reference = str(item["reference"]).strip()
            if not subject or not reference:
                raise RuntimeError(f"{labels}:{line_no}: empty subject/reference")

            image_value = Path(str(item["image"]))
            image_path = image_value if image_value.is_absolute() else root / image_value
            if not image_path.exists():
                raise FileNotFoundError(f"{labels}:{line_no}: missing image: {image_path}")

            sid = int(item.get("id", len(rows)))
            rows.append(
                {
                    "sid": sid,
                    "image_id": str(item.get("image", image_path.name)),
                    "image_path": image_path,
                    "subject": subject,
                    "reference": reference,
                    "relation": relation,
                    "source_relation_raw": str(item["relation"]),
                    "question_text": SYNTHETIC_PROMPT.format(
                        subject=subject,
                        reference=reference,
                    ),
                }
            )

    rows.sort(key=lambda r: int(r["sid"]))
    # IDs must be unique because residual alignment uses sample_index.
    ids = [int(r["sid"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate synthetic ids in labels.jsonl")

    if args.max_samples and args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise RuntimeError("Synthetic source dataset is empty")
    return rows


def build_chat_prompt(processor: Any, question: str, with_image: bool) -> str:
    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return question


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def make_batch(
    processor: Any,
    question: str,
    image: Optional[Image.Image],
    device: torch.device,
) -> Dict[str, Any]:
    rendered = build_chat_prompt(processor, question, with_image=image is not None)
    if image is None:
        attempts = [
            lambda: processor(text=[rendered], padding=True, return_tensors="pt"),
            lambda: processor(text=rendered, return_tensors="pt"),
        ]
    else:
        attempts = [
            lambda: processor(text=[rendered], images=[image], padding=True, return_tensors="pt"),
            lambda: processor(text=rendered, images=image, return_tensors="pt"),
        ]

    last_error = None
    for fn in attempts:
        try:
            return move_batch(fn(), device)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Processor failed ({'REAL' if image is not None else 'NOIMAGE'}): "
        f"{type(last_error).__name__}: {last_error}"
    )


def load_saved(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def extract_condition(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    records: Sequence[Mapping[str, Any]],
    save_layers: Sequence[int],
    decoder_blocks: int,
    hidden_size: int,
    vision_mode: str,
    out_path: Path,
) -> None:
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
    relations: List[str] = []
    vectors: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []

    def save_progress() -> None:
        if not vectors:
            return
        metadata = {
            "dataset": "synthetic_shapes_4dir_400",
            "model_alias": args.model,
            "repo_id": base.SPECS[args.model].repo_id,
            "prompt_type": "raw",
            "prompt_template": SYNTHETIC_PROMPT,
            "vision_mode": vision_mode,
            "representation": "object_last_token_pair",
            "decoder_blocks_total": int(decoder_blocks),
            "saved_decoder_blocks": [int(L) for L in save_layers],
            "hidden_size": int(hidden_size),
            "n_requested": int(len(records)),
            "n_saved": int(len(vectors)),
            "seed": int(args.seed),
            "transformers_version": transformers.__version__,
        }
        atomic_save_npz(
            out_path,
            {
                "metadata_json": np.array(json.dumps(metadata), dtype=object),
                "sample_index": np.asarray(sids, dtype=np.int64),
                "image_id": np.asarray(image_ids, dtype=object),
                "subject": np.asarray(subjects, dtype=object),
                "reference": np.asarray(references, dtype=object),
                "relation": np.asarray(relations, dtype=object),
                "decoder_block_index": np.asarray(save_layers, dtype=np.int32),
                "relation_vectors": np.stack(vectors, axis=0).astype(dtype_np),
            },
        )

    desc = f"synthetic400:{args.model}:{vision_mode}"
    for record in tqdm(records, desc=desc):
        image: Optional[Image.Image] = None
        batch: Optional[Dict[str, Any]] = None
        outputs = None
        states = None
        try:
            if vision_mode == "correct":
                image = Image.open(record["image_path"]).convert("RGB")
            elif vision_mode == "no_image":
                image = None
            else:
                raise ValueError(vision_mode)

            batch = make_batch(
                processor,
                str(record["question_text"]),
                image,
                device,
            )
            ids = batch["input_ids"][0].detach().cpu().tolist()

            # Match the COCO residual extractor exactly: last token of the last
            # occurrence of each object phrase.
            subject_index = base.find_phrase_last_token(
                processor.tokenizer,
                ids,
                str(record["subject"]),
            )
            reference_index = base.find_phrase_last_token(
                processor.tokenizer,
                ids,
                str(record["reference"]),
            )
            if int(subject_index) == int(reference_index):
                raise RuntimeError(
                    f"Subject/reference token positions collide at {subject_index}; "
                    f"subject={record['subject']!r}, reference={record['reference']!r}"
                )

            with torch.inference_mode():
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                states = base.hidden_tuple(outputs)

            current_blocks = len(states) - 1
            if current_blocks != decoder_blocks:
                raise RuntimeError(
                    f"Decoder block count changed: expected={decoder_blocks}, got={current_blocks}"
                )
            final = states[-1]
            if final.ndim != 3 or int(final.shape[0]) != 1:
                raise RuntimeError(f"Unexpected hidden state shape: {tuple(final.shape)}")
            if int(final.shape[1]) != len(ids):
                raise RuntimeError(
                    "input_ids / hidden-state length mismatch: "
                    f"ids={len(ids)} hidden={int(final.shape[1])}"
                )
            if int(final.shape[-1]) != hidden_size:
                raise RuntimeError(
                    f"Hidden dim changed: expected={hidden_size}, got={int(final.shape[-1])}"
                )

            pair = np.stack(
                [
                    (
                        states[int(L) + 1][0, int(subject_index)]
                        - states[int(L) + 1][0, int(reference_index)]
                    )
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for L in save_layers
                ],
                axis=0,
            ).astype(dtype_np)

            sids.append(int(record["sid"]))
            image_ids.append(str(record["image_id"]))
            subjects.append(str(record["subject"]))
            references.append(str(record["reference"]))
            relations.append(str(record["relation"]))
            vectors.append(pair)

            if len(vectors) % max(1, int(args.save_every)) == 0:
                save_progress()

        except Exception as exc:
            errors.append(
                {
                    "sid": int(record["sid"]),
                    "image": str(record["image_path"]),
                    "subject": str(record["subject"]),
                    "reference": str(record["reference"]),
                    "relation": str(record["relation"]),
                    "vision_mode": vision_mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc()[-4000:],
                }
            )
            tqdm.write(
                f"[ERROR {vision_mode} sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            if image is not None:
                image.close()
            del outputs, states, batch
            cleanup()

    save_progress()
    err_path = out_path.parent.parent / f"errors.{vision_mode}.json"
    err_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path} | n={len(vectors)}/{len(records)} | errors={len(errors)}")
    if not vectors:
        raise RuntimeError(f"No samples succeeded for vision_mode={vision_mode}")


def make_residual(
    args: argparse.Namespace,
    correct_path: Path,
    noimage_path: Path,
    residual_path: Path,
) -> Dict[str, Any]:
    c = load_saved(correct_path)
    n = load_saved(noimage_path)

    c_sids = np.asarray(c["sample_index"], dtype=np.int64)
    n_sids = np.asarray(n["sample_index"], dtype=np.int64)
    c_map = {int(s): i for i, s in enumerate(c_sids.tolist())}
    n_map = {int(s): i for i, s in enumerate(n_sids.tolist())}
    common = [int(s) for s in c_sids.tolist() if int(s) in n_map]
    if not common:
        raise RuntimeError("No common successful sample IDs between REAL and NoImage")

    ci = np.asarray([c_map[s] for s in common], dtype=np.int64)
    ni = np.asarray([n_map[s] for s in common], dtype=np.int64)

    c_layers = np.asarray(c["decoder_block_index"], dtype=np.int32)
    n_layers = np.asarray(n["decoder_block_index"], dtype=np.int32)
    if not np.array_equal(c_layers, n_layers):
        raise RuntimeError(
            f"Layer mismatch correct={c_layers.tolist()} noimage={n_layers.tolist()}"
        )

    for key in ("subject", "reference", "relation"):
        ca = np.asarray(c[key], dtype=object)[ci]
        na = np.asarray(n[key], dtype=object)[ni]
        if not np.array_equal(ca, na):
            raise RuntimeError(f"Alignment mismatch in key={key}")

    Xc = np.asarray(c["relation_vectors"], dtype=np.float32)[ci]
    Xn = np.asarray(n["relation_vectors"], dtype=np.float32)[ni]
    if Xc.shape != Xn.shape:
        raise RuntimeError(f"Vector shape mismatch correct={Xc.shape} noimage={Xn.shape}")
    Xr = Xc - Xn

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    meta = {
        "dataset": "synthetic_shapes_4dir_400",
        "model_alias": args.model,
        "repo_id": base.SPECS[args.model].repo_id,
        "prompt_type": "raw",
        "prompt_template": SYNTHETIC_PROMPT,
        "representation": "correct_minus_noimage",
        "object_pool": "last_token",
        "source_correct": str(correct_path),
        "source_noimage": str(noimage_path),
        "n_saved": int(len(common)),
        "decoder_block_index": [int(v) for v in c_layers.tolist()],
        "hidden_size": int(Xr.shape[-1]),
        "seed": int(args.seed),
        "transformers_version": transformers.__version__,
    }
    atomic_save_npz(
        residual_path,
        {
            "metadata_json": np.array(json.dumps(meta), dtype=object),
            "sample_index": np.asarray(common, dtype=np.int64),
            "image_id": np.asarray(c["image_id"], dtype=object)[ci],
            "subject": np.asarray(c["subject"], dtype=object)[ci],
            "reference": np.asarray(c["reference"], dtype=object)[ci],
            "relation": np.asarray(c["relation"], dtype=object)[ci],
            "decoder_block_index": c_layers,
            "relation_vectors": Xr.astype(dtype_np),
        },
    )

    labels = np.asarray(c["relation"], dtype=object)[ci]
    counts = {r: int(np.sum(labels == r)) for r in REL}
    print(f"[saved residual] {residual_path}")
    print(f"  shape={Xr.shape} | relation_counts={counts}")
    return {
        "n_correct": int(len(c_sids)),
        "n_noimage": int(len(n_sids)),
        "n_common": int(len(common)),
        "shape": list(map(int, Xr.shape)),
        "relation_counts": counts,
        "layers": [int(v) for v in c_layers.tolist()],
    }


def source_centroid_sanity(residual_path: Path) -> List[Dict[str, Any]]:
    """Leave-in centroid sanity check only; not a held-out performance estimate."""
    d = load_saved(residual_path)
    X = np.asarray(d["relation_vectors"], dtype=np.float32)
    y = np.asarray(d["relation"], dtype=object)
    layers = [int(v) for v in np.asarray(d["decoder_block_index"]).tolist()]
    rows = []
    for li, L in enumerate(layers):
        Xi = X[:, li].astype(np.float64)
        center = Xi.mean(axis=0)
        dirs = {}
        for r in REL:
            m = Xi[y == r].mean(axis=0) - center
            n = np.linalg.norm(m)
            dirs[r] = m / max(float(n), 1e-12)
        Q = Xi - center
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        D = np.stack([dirs[r] for r in REL], axis=1)
        pred = np.argmax(Q @ D, axis=1)
        gt = np.asarray([REL.index(str(v)) for v in y], dtype=np.int64)
        acc = float(np.mean(pred == gt))
        rows.append({"source_layer": int(L), "leave_in_centroid_acc": acc})
    return rows


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    states_dir = outdir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    records = load_synthetic_records(args)
    counts = Counter(str(r["relation"]) for r in records)
    print("=" * 140)
    print("SYNTHETIC-400 REAL / NOIMAGE SPATIAL STATE EXTRACTION")
    print("=" * 140)
    print(f"model={args.model} synthetic_dir={args.synthetic_dir}")
    print(f"N={len(records)} relation_counts={dict(counts)}")
    print(f"prompt={SYNTHETIC_PROMPT}")

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

        # Run one tiny forward first to discover decoder depth/hidden size before
        # parsing the requested output layers.
        probe_img = Image.open(records[0]["image_path"]).convert("RGB")
        probe_batch = make_batch(
            processor,
            str(records[0]["question_text"]),
            probe_img,
            device,
        )
        with torch.inference_mode():
            probe_out = model(
                **probe_batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            probe_states = base.hidden_tuple(probe_out)
        decoder_blocks = len(probe_states) - 1
        hidden_size = int(probe_states[-1].shape[-1])
        save_layers = parse_layers(args.layers, decoder_blocks)
        print(
            f"repo={spec.repo_id} decoder_blocks={decoder_blocks} hidden={hidden_size} "
            f"saved_layers={save_layers}"
        )
        probe_img.close()
        del probe_batch, probe_out, probe_states
        cleanup()

        correct_path = states_dir / "raw__correct.npz"
        noimage_path = states_dir / "raw__no_image.npz"
        residual_path = states_dir / "raw__correct_minus_noimage.npz"

        extract_condition(
            args=args,
            model=model,
            processor=processor,
            device=device,
            records=records,
            save_layers=save_layers,
            decoder_blocks=decoder_blocks,
            hidden_size=hidden_size,
            vision_mode="correct",
            out_path=correct_path,
        )
        extract_condition(
            args=args,
            model=model,
            processor=processor,
            device=device,
            records=records,
            save_layers=save_layers,
            decoder_blocks=decoder_blocks,
            hidden_size=hidden_size,
            vision_mode="no_image",
            out_path=noimage_path,
        )
        residual_summary = make_residual(
            args,
            correct_path,
            noimage_path,
            residual_path,
        )
        sanity = source_centroid_sanity(residual_path)

        print("\nSOURCE CENTROID SANITY (leave-in; diagnostic only)")
        print("-" * 80)
        for row in sanity:
            print(f"L{row['source_layer']:02d}: {row['leave_in_centroid_acc']:.4f}")

        summary = {
            "config": vars(args),
            "model_repo": spec.repo_id,
            "synthetic_prompt": SYNTHETIC_PROMPT,
            "source_relation_counts_requested": dict(counts),
            "residual": residual_summary,
            "source_centroid_sanity_leave_in": sanity,
            "elapsed_sec": float(time.time() - started),
        }
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\nDONE")
        print(f"Use this as source NPZ:")
        print(f"  {residual_path}")

    finally:
        del model, processor
        cleanup()


if __name__ == "__main__":
    main()
