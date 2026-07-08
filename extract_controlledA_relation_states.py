#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract object-token difference states for Controlled_Images_A across VLM backends.

Output matches the COCO/VG two-object extractor format:
  relation_vectors[n, layer, hidden] = h_L(subject) - h_L(reference)

It uses the prompt file:
  prompts/Controlled_Images_A_with_answer_four_options.jsonl
and the dataset_zoo entry:
  Controlled_Images_A

Run from the AdaptVis repository root, next to extract_two_object_relation_states_v3.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None

# Reuse the already-tested backend helpers from the COCO/VG extractor.
from extract_two_object_relation_states_v3 import (  # noqa: E402
    SPECS,
    build_chat_prompt,
    configure_processor,
    find_phrase_last_token,
    hidden_tuple,
    move_batch,
    resolve_dtype,
    select_blocks,
    atomic_save,
)

REL_MAP = {
    "left": "left",
    "right": "right",
    "on": "on",
    "top": "on",
    "above": "on",
    "under": "under",
    "below": "under",
    "bottom": "under",
}
SUPPORTED_RELATIONS = {"left", "right", "on", "under"}

QUESTION_RE = re.compile(
    r"Where\s+(?P<verb>is|are)\s+(?:the\s+)?(?P<subject>.+?)\s+"
    r"in\s+relation\s+to\s+(?:the\s+)?(?P<reference>.+?)\?\s*Answer",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class PromptMeta:
    subject: str
    reference: str
    verb: str


@dataclass(frozen=True)
class Record:
    sid: int
    relation: str
    subject: str
    reference: str
    question: str
    image: Image.Image

    @property
    def group(self) -> str:
        return " || ".join(sorted((self.subject, self.reference)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-path", default="prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    p.add_argument("--model", required=True, choices=sorted(SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=["sdpa", "eager", "flash_attention_2", "none"], default="sdpa")
    p.add_argument("--layer-fracs", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,1.00")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", required=True)
    return p.parse_args()


def parse_fractions(raw: str) -> List[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Layer fraction must lie in (0,1], got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("--layer-fracs must contain at least one value")
    return sorted(set(vals))


def canonical_object(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[\s\.,;:!?]+$", "", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_relation(x: Any) -> str:
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
    key = str(x).strip().lower()
    return REL_MAP.get(key, key)


def parse_prompt(question: str) -> PromptMeta:
    m = QUESTION_RE.search(str(question))
    if m is None:
        raise ValueError(f"Could not parse subject/reference from prompt:\n{question}")
    return PromptMeta(
        subject=canonical_object(m.group("subject")),
        reference=canonical_object(m.group("reference")),
        verb=str(m.group("verb")).lower(),
    )


def load_prompt_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for expected_id, line in enumerate(f):
            row = json.loads(line)
            if int(row.get("id", expected_id)) != expected_id:
                raise ValueError(f"Prompt IDs not contiguous: expected {expected_id}, got {row.get('id')}")
            rows.append(row)
    return rows


def extract_images_from_batch(batch: Mapping[str, Any]) -> Iterable[Any]:
    # Matches existing Controlled_A scripts.  The repository collate usually
    # returns image_options = [[PIL.Image], ...].
    if "image_options" in batch:
        for image_option in batch["image_options"]:
            for image in image_option:
                yield image
    elif "images" in batch:
        for image in batch["images"]:
            yield image
    elif "image" in batch:
        vals = batch["image"]
        if isinstance(vals, (list, tuple)):
            yield from vals
        else:
            yield vals
    else:
        raise KeyError(f"Cannot find images in batch keys={list(batch.keys())}")


def ensure_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 3:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type for Controlled_A: {type(image)} shape={getattr(arr, 'shape', None)}")


def load_records(prompt_path: Path, *, download: bool, max_samples: Optional[int], num_workers: int) -> Tuple[List[Record], List[Dict[str, Any]]]:
    prompt_rows = load_prompt_rows(prompt_path)
    dataset = get_dataset("Controlled_Images_A", image_preprocess=None, download=download)
    total_available = min(len(dataset), len(prompt_rows))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=repository_default_collate,
    )
    records: List[Record] = []
    audit: List[Dict[str, Any]] = []
    sid = 0
    pbar = tqdm(total=total_available, desc="Loading Controlled_A")
    for batch in loader:
        for raw_img in extract_images_from_batch(batch):
            if sid >= total_available:
                break
            row = prompt_rows[sid]
            try:
                rel = normalize_relation(row.get("answer", ""))
                if rel not in SUPPORTED_RELATIONS:
                    audit.append({"sid": sid, "reason": "unsupported_relation", "answer": row.get("answer")})
                    sid += 1
                    pbar.update(1)
                    continue
                meta = parse_prompt(str(row["question"]))
                image = ensure_pil_rgb(raw_img)
                records.append(
                    Record(
                        sid=sid,
                        relation=rel,
                        subject=meta.subject,
                        reference=meta.reference,
                        question=str(row["question"]),
                        image=image,
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    sid += 1
                    pbar.update(1)
                    pbar.close()
                    return records, audit
            except Exception as exc:
                audit.append({
                    "sid": sid,
                    "reason": type(exc).__name__,
                    "error": str(exc),
                    "row": row,
                })
            sid += 1
            pbar.update(1)
        if sid >= total_available:
            break
    pbar.close()
    return records, audit


def load_resume(path: Path, model_alias: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        if meta.get("dataset") != "controlled_A" or meta.get("model_alias") != model_alias:
            raise RuntimeError(f"Existing output metadata does not match controlled_A/{model_alias}: {path}")
        return {k: z[k] for k in z.files}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(args.output)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    fractions = parse_fractions(args.layer_fracs)
    records, audit = load_records(Path(args.prompt_path), download=args.download, max_samples=args.max_samples, num_workers=args.num_workers)
    if not records:
        raise RuntimeError("No usable Controlled_A records")
    out_path.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[controlled_A] usable={len(records)} | relations={dict(sorted({r: sum(x.relation == r for x in records) for r in SUPPORTED_RELATIONS}.items()))} | audit={len(audit)}")

    resume = None if args.overwrite else load_resume(out_path, args.model)
    done_sids: set[int] = set()
    sample_indices: List[int] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    questions: List[str] = []
    groups: List[str] = []
    vectors: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    selected_blocks: Optional[List[int]] = None
    hidden_size: Optional[int] = None
    decoder_blocks: Optional[int] = None

    if resume is not None:
        meta = json.loads(str(resume["metadata_json"].item()))
        sample_indices = [int(v) for v in resume["sample_index"].tolist()]
        subjects = [str(v) for v in resume["subject"].tolist()]
        references = [str(v) for v in resume["reference"].tolist()]
        relations = [str(v) for v in resume["relation"].tolist()]
        questions = [str(v) for v in resume["question"].tolist()]
        groups = [str(v) for v in resume["group"].tolist()] if "group" in resume else [" || ".join(sorted((s, r))) for s, r in zip(subjects, references)]
        vectors = [row for row in resume["relation_vectors"]]
        selected_blocks = [int(v) for v in resume["decoder_block_index"].tolist()]
        hidden_size = int(meta["hidden_size"])
        decoder_blocks = int(meta["decoder_blocks"])
        done_sids = set(sample_indices)
        print(f"Resuming {out_path}: {len(done_sids)} saved samples")

    spec = SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")

    model = None
    processor = None
    started = time.time()
    try:
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": resolve_dtype(spec.dtype_name),
            "low_cpu_mem_usage": True,
            "trust_remote_code": spec.trust_remote_code,
            "device_map": {"": args.device},
        }
        if args.attn_impl != "none":
            load_kwargs["attn_implementation"] = args.attn_impl
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        configure_processor(model, processor)
        device = torch.device(args.device)

        def save_progress() -> None:
            if not vectors or selected_blocks is None or hidden_size is None or decoder_blocks is None:
                return
            metadata = {
                "dataset": "controlled_A",
                "model_alias": args.model,
                "repo_id": spec.repo_id,
                "transformers_version": transformers.__version__,
                "seed": args.seed,
                "layer_fractions": fractions,
                "decoder_blocks": decoder_blocks,
                "hidden_size": hidden_size,
                "n_requested_records": len(records),
                "n_saved_records": len(vectors),
                "relation_counts_saved": {rel: relations.count(rel) for rel in sorted(set(relations))},
                "saved_at_unix": time.time(),
            }
            arrays: Dict[str, Any] = {
                "metadata_json": np.array(json.dumps(metadata), dtype=object),
                "sample_index": np.asarray(sample_indices, dtype=np.int64),
                "sid": np.asarray(sample_indices, dtype=np.int64),
                "subject": np.asarray(subjects, dtype=object),
                "reference": np.asarray(references, dtype=object),
                "relation": np.asarray(relations, dtype=object),
                "question": np.asarray(questions, dtype=object),
                "group": np.asarray(groups, dtype=object),
                "decoder_block_index": np.asarray(selected_blocks, dtype=np.int32),
                "relation_vectors": np.stack(vectors, axis=0).astype(np.float16),
            }
            atomic_save(out_path, arrays)

        for record in tqdm(records, desc=f"controlled_A:{args.model}"):
            if record.sid in done_sids:
                continue
            try:
                rendered = build_chat_prompt(processor, record.subject, record.reference)
                batch = processor(text=[rendered], images=[record.image], return_tensors="pt")
                batch = move_batch(batch, device)
                input_ids = batch["input_ids"][0].detach().cpu().tolist()
                subject_index = find_phrase_last_token(processor.tokenizer, input_ids, record.subject)
                reference_index = find_phrase_last_token(processor.tokenizer, input_ids, record.reference)
                if subject_index == reference_index:
                    raise RuntimeError("Subject/reference token positions collide")
                with torch.inference_mode():
                    outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
                states = hidden_tuple(outputs)
                final = states[-1]
                if final.ndim != 3 or final.shape[0] != 1:
                    raise RuntimeError(f"Unexpected final hidden-state shape: {tuple(final.shape)}")
                if int(final.shape[1]) != len(input_ids):
                    raise RuntimeError(
                        "Text token positions do not align with returned hidden states: "
                        f"input_len={len(input_ids)}, hidden_len={final.shape[1]}"
                    )
                if selected_blocks is None:
                    decoder_blocks = len(states) - 1
                    selected_blocks = select_blocks(decoder_blocks, fractions)
                    hidden_size = int(final.shape[-1])
                    print(f"{args.model}: decoder_blocks={decoder_blocks}, hidden={hidden_size}, selected_blocks={selected_blocks}")
                relation_vector = np.stack(
                    [
                        (states[block + 1][0, subject_index] - states[block + 1][0, reference_index])
                        .detach().float().cpu().numpy()
                        for block in selected_blocks
                    ],
                    axis=0,
                ).astype(np.float16)
                sample_indices.append(record.sid)
                subjects.append(record.subject)
                references.append(record.reference)
                relations.append(record.relation)
                questions.append(record.question)
                groups.append(record.group)
                vectors.append(relation_vector)
                del outputs, states, batch
                if len(vectors) % args.save_every == 0:
                    save_progress()
            except Exception as exc:
                errors.append({
                    "sid": record.sid,
                    "relation": record.relation,
                    "subject": record.subject,
                    "reference": record.reference,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-8:],
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        save_progress()
        out_path.with_suffix(".errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(vectors)}/{len(records)} states to {out_path} | errors={len(errors)} | elapsed={(time.time() - started)/60.0:.1f} min")
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
