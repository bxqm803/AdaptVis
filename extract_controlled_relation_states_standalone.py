#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone extractor for Controlled_Images_A/B object-token relation states.

This file does NOT import extract_two_object_relation_states*.py.

For each Controlled_Images_A/B sample, it parses subject/reference from the
question prompt, extracts the final sub-token hidden state for both object
phrases at selected decoder blocks, and saves

    relation_vectors[n, layer, hidden] = h_L(subject) - h_L(reference)

The output is compatible with plot_score4d_tsne_only.py and the four-direction
relation-codebook analysis scripts.
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
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")

try:
    from dataset_zoo import get_dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import dataset_zoo.get_dataset. Run from AdaptVis repo root. Error: {exc}")

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    model_class: str
    dtype_name: str
    trust_remote_code: bool = False


SPECS: Dict[str, ModelSpec] = {
    "llava-7b": ModelSpec("llava-7b", "llava-hf/llava-1.5-7b-hf", "LlavaForConditionalGeneration", "float16"),
    "llava-13b": ModelSpec("llava-13b", "llava-hf/llava-1.5-13b-hf", "LlavaForConditionalGeneration", "float16"),
    "qwen2-2b": ModelSpec("qwen2-2b", "Qwen/Qwen2-VL-2B-Instruct", "Qwen2VLForConditionalGeneration", "bfloat16"),
    "qwen-3b": ModelSpec("qwen-3b", "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2_5_VLForConditionalGeneration", "bfloat16"),
    "qwen-7b": ModelSpec("qwen-7b", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2_5_VLForConditionalGeneration", "bfloat16"),
    "internvl-1b": ModelSpec("internvl-1b", "OpenGVLab/InternVL3-1B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
    "internvl-2b": ModelSpec("internvl-2b", "OpenGVLab/InternVL3-2B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
    "internvl-8b": ModelSpec("internvl-8b", "OpenGVLab/InternVL3-8B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
    "internvl-14b": ModelSpec("internvl-14b", "OpenGVLab/InternVL3-14B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
}

REL_MAP = {
    "left": "left",
    "right": "right",
    "on": "on",
    "top": "on",
    "above": "on",
    "over": "on",
    "under": "under",
    "below": "under",
    "bottom": "under",
    "underneath": "under",
    "in front": "in_front",
    "in front of": "in_front",
    "front": "in_front",
    "in_front": "in_front",
    "in-front": "in_front",
    "behind": "behind",
    "back": "behind",
}
DEFAULT_RELATIONS = ["left", "right", "on", "under", "in_front", "behind"]

QUESTION_PATTERNS = [
    re.compile(
        r"Where\s+(?P<verb>is|are)\s+(?:the\s+)?(?P<subject>.+?)\s+"
        r"in\s+relation\s+to\s+(?:the\s+)?(?P<reference>.+?)\?\s*Answer",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Where\s+(?P<verb>is|are)\s+(?:the\s+)?(?P<subject>.+?)\s+"
        r"relative\s+to\s+(?:the\s+)?(?P<reference>.+?)\?",
        flags=re.IGNORECASE | re.DOTALL,
    ),
]


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
    p.add_argument("--controlled-set", choices=["A", "B"], default="A", help="Which What'sUp controlled split to load.")
    p.add_argument("--dataset-key", default=None, help="Override dataset_zoo key, e.g. Controlled_Images_B.")
    p.add_argument("--prompt-path", default=None, help="Override prompt jsonl path. Defaults to prompts/Controlled_Images_<A/B>_with_answer_four_options.jsonl.")
    p.add_argument("--relations", default=",".join(DEFAULT_RELATIONS), help="Comma-separated canonical relations to keep after normalization.")
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


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


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
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    return REL_MAP.get(key, key.replace(" ", "_"))


def parse_relation_list(raw: str) -> List[str]:
    vals = []
    for part in raw.split(","):
        rel = normalize_relation(part)
        if rel and rel not in vals:
            vals.append(rel)
    if not vals:
        raise ValueError("--relations must contain at least one relation")
    return vals


def parse_prompt(question: str) -> PromptMeta:
    q = str(question)
    for pat in QUESTION_PATTERNS:
        m = pat.search(q)
        if m is not None:
            return PromptMeta(
                subject=canonical_object(m.group("subject")),
                reference=canonical_object(m.group("reference")),
                verb=str(m.group("verb")).lower(),
            )
    raise ValueError(f"Could not parse subject/reference from prompt:\n{question}")


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
    raise TypeError(f"Unsupported image type for Controlled image: {type(image)} shape={getattr(arr, 'shape', None)}")


def load_records(
    prompt_path: Path,
    *,
    dataset_key: str,
    keep_relations: Sequence[str],
    download: bool,
    max_samples: Optional[int],
    num_workers: int,
) -> Tuple[List[Record], List[Dict[str, Any]]]:
    prompt_rows = load_prompt_rows(prompt_path)
    dataset = get_dataset(dataset_key, image_preprocess=None, download=download)
    total_available = min(len(dataset), len(prompt_rows))
    keep = set(keep_relations)
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
    pbar = tqdm(total=total_available, desc=f"Loading {dataset_key}")
    for batch in loader:
        for raw_img in extract_images_from_batch(batch):
            if sid >= total_available:
                break
            row = prompt_rows[sid]
            try:
                rel = normalize_relation(row.get("answer", ""))
                if rel not in keep:
                    audit.append({"sid": sid, "reason": "unsupported_relation", "answer": row.get("answer"), "normalized": rel})
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


def build_chat_prompt(processor: Any, subject: str, reference: str) -> str:
    prompt = f"Where is the {subject} relative to the {reference}? Answer with one spatial relation."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def move_batch(batch: Any, device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    width = len(needle)
    return [start for start in range(len(haystack) - width + 1) if list(haystack[start : start + width]) == list(needle)]


def find_phrase_last_token(tokenizer: Any, input_ids: Sequence[int], phrase: str) -> int:
    matches: List[Tuple[int, int]] = []
    seen = set()
    variants = [" " + phrase, phrase]
    # Some tokenizers segment articles/punctuation differently; these variants are harmless if unmatched.
    variants += [" the " + phrase, "the " + phrase]
    for variant in variants:
        token_ids = list(tokenizer(variant, add_special_tokens=False).input_ids)
        key = tuple(int(v) for v in token_ids)
        if not key or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, token_ids):
            matches.append((start, start + len(token_ids) - 1))
    if not matches:
        raise ValueError(f"Could not find token span for {phrase!r}.")
    return max(matches, key=lambda item: item[0])[1]


def configure_processor(model: Any, processor: Any) -> None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None and hasattr(processor, "patch_size") and hasattr(vision_config, "patch_size"):
        processor.patch_size = int(vision_config.patch_size)
    strategy = getattr(config, "vision_feature_select_strategy", None)
    if strategy is not None and hasattr(processor, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = str(strategy)
    if getattr(config, "model_type", "") == "llava" and hasattr(processor, "num_additional_image_tokens"):
        processor.num_additional_image_tokens = 1


def hidden_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
        getattr(getattr(outputs, "text_model_output", None), "hidden_states", None),
    ]
    for states in candidates:
        if isinstance(states, (tuple, list)) and states and torch.is_tensor(states[-1]):
            return tuple(states)
    raise RuntimeError("No decoder hidden_states were returned by this backend.")


def select_blocks(n_blocks: int, fractions: Sequence[float]) -> List[int]:
    if n_blocks <= 0:
        raise RuntimeError(f"Invalid decoder block count: {n_blocks}")
    result = sorted({int(round(frac * (n_blocks - 1))) for frac in fractions})
    if min(result) < 0 or max(result) >= n_blocks:
        raise RuntimeError(f"Selected invalid decoder block indices {result} for {n_blocks} blocks.")
    return result


def atomic_save(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def load_resume(path: Path, model_alias: str, dataset_tag: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        if meta.get("dataset") != dataset_tag or meta.get("model_alias") != model_alias:
            raise RuntimeError(f"Existing output metadata does not match {dataset_tag}/{model_alias}: {path}")
        return {k: z[k] for k in z.files}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    if args.model not in SPECS:
        raise ValueError(f"Unknown model alias: {args.model}")

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
    keep_relations = parse_relation_list(args.relations)
    controlled_set = str(args.controlled_set).upper()
    dataset_key = args.dataset_key or f"Controlled_Images_{controlled_set}"
    dataset_tag = f"controlled_{controlled_set}"
    prompt_path = Path(args.prompt_path or f"prompts/Controlled_Images_{controlled_set}_with_answer_four_options.jsonl")

    records, audit = load_records(
        prompt_path,
        dataset_key=dataset_key,
        keep_relations=keep_relations,
        download=args.download,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    if not records:
        raise RuntimeError(f"No usable {dataset_key} records. Check prompt path, dataset key, and --relations.")

    audit_path = out_path.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    relation_counts = {rel: sum(x.relation == rel for x in records) for rel in keep_relations}
    print(f"[{dataset_tag}] dataset_key={dataset_key} prompt={prompt_path} usable={len(records)} | relations={relation_counts} | audit={len(audit)}")

    resume = None if args.overwrite else load_resume(out_path, args.model, dataset_tag)
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
                "dataset": dataset_tag,
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

        for record in tqdm(records, desc=f"{dataset_tag}:{args.model}"):
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
                        f"input_len={len(input_ids)}, hidden_len={final.shape[1]}. "
                        "This backend needs a model-specific merged-token mapper before it can be included."
                    )
                if selected_blocks is None:
                    decoder_blocks = len(states) - 1
                    selected_blocks = select_blocks(decoder_blocks, fractions)
                    hidden_size = int(final.shape[-1])
                    print(f"{args.model}: decoder_blocks={decoder_blocks}, hidden={hidden_size}, selected_blocks={selected_blocks}")

                relation_vector = np.stack(
                    [
                        (states[block + 1][0, subject_index] - states[block + 1][0, reference_index])
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
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
        error_path = out_path.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(vectors)}/{len(records)} states to {out_path} | errors={len(errors)} | elapsed={(time.time() - started)/60.0:.1f} min")
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
