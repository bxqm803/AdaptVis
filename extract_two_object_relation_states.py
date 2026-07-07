#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract frozen two-object relation states from COCO_two or VG_two.

For each sample, this script builds a clean question without answer options,
extracts the final sub-token state for the subject and reference object at a
set of decoder depths, and saves their difference:

    r_L(subject, reference) = h_L(subject) - h_L(reference).

The output is intended for the separate affine opposing-axis analysis script.
No labels are used by the VLM during extraction. Labels are retained only as
metadata for subsequent cross-validated analysis.

Run one model per process. The companion shell runner launches models
sequentially so GPU memory is released between checkpoints.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    model_class: str
    dtype_name: str
    trust_remote_code: bool = False


# Kept identical to the backend smoke test aliases.
SPECS: Dict[str, ModelSpec] = {
    "llava-7b": ModelSpec("llava-7b", "llava-hf/llava-1.5-7b-hf", "LlavaForConditionalGeneration", "float16"),
    "llava-13b": ModelSpec("llava-13b", "llava-hf/llava-1.5-13b-hf", "LlavaForConditionalGeneration", "float16"),
    "qwen2-2b": ModelSpec("qwen2-2b", "Qwen/Qwen2-VL-2B-Instruct", "Qwen2VLForConditionalGeneration", "bfloat16"),
    "qwen-3b": ModelSpec("qwen-3b", "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2_5_VLForConditionalGeneration", "bfloat16"),
    "qwen-7b": ModelSpec("qwen-7b", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2_5_VLForConditionalGeneration", "bfloat16"),
    "llama-11b": ModelSpec("llama-11b", "meta-llama/Llama-3.2-11B-Vision-Instruct", "MllamaForConditionalGeneration", "bfloat16"),
    "internvl-1b": ModelSpec("internvl-1b", "OpenGVLab/InternVL3-1B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
    "internvl-2b": ModelSpec("internvl-2b", "OpenGVLab/InternVL3-2B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
    "internvl-8b": ModelSpec("internvl-8b", "OpenGVLab/InternVL3-8B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
    "internvl-14b": ModelSpec("internvl-14b", "OpenGVLab/InternVL3-14B-hf", "InternVLForConditionalGeneration", "bfloat16", True),
    "gemma-4b": ModelSpec("gemma-4b", "google/gemma-3-4b-it", "Gemma3ForConditionalGeneration", "bfloat16", True),
    "gemma-12b": ModelSpec("gemma-12b", "google/gemma-3-12b-it", "Gemma3ForConditionalGeneration", "bfloat16", True),
}

REL_ALIAS = {
    "left": "left",
    "right": "right",
    "above": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "top": "above",
    "bottom": "below",
    "front": "front",
    "behind": "behind",
}


@dataclass(frozen=True)
class Record:
    sid: int
    image_id: str
    image_path: Path
    caption: str
    opposite_caption: str
    subject: str
    reference: str
    relation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["coco_two", "vg_two"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", required=True, choices=sorted(SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", choices=["sdpa", "eager", "flash_attention_2", "none"], default="sdpa")
    parser.add_argument(
        "--layer-fracs",
        default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,1.00",
        help="Relative decoder depths. 0.50 means the middle decoder block; 1.00 means final block.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap after successful relation parsing.")
    parser.add_argument("--image-mode", choices=["true", "shuffle", "blank"], default="true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", required=True, help="Output .npz path. Parent directories are created.")
    return parser.parse_args()


def parse_fractions(raw: str) -> List[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Layer fraction must lie in (0,1], got {value}.")
        values.append(value)
    if not values:
        raise ValueError("--layer-fracs must contain at least one fraction.")
    return sorted(set(values))


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def canonical_phrase(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s\.,;:!?]+$", "", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_relation_caption(caption: str) -> Optional[Tuple[str, str, str]]:
    """Return canonical ``(subject, reference, relation)`` from a gold caption.

    ARO two-object captions commonly use both sentence-style templates
    (``A is to the left of B``) and noun-phrase templates
    (``A photo of A to the left of B``).  The parser accepts both, while
    retaining only the relation stated in the gold caption.
    """
    text = caption.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,!?:;\n\t")

    # Remove common dataset wrappers so that both of the following reduce to
    # the same relation phrase:
    #   "A photo of a cup to the left of a book"
    #   "A cup is to the left of a book"
    text = re.sub(
        r"^(?:a|an|the)?\s*(?:photo|picture|image)\s+of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # The verbal copula is optional because ARO annotations are often noun
    # phrases rather than full sentences after the wrapper is removed.
    copula = r"(?:(?:is|are)\s+)?"
    patterns: List[Tuple[str, str]] = [
        (
            rf"^(?P<s>.+?)\s+{copula}(?:to\s+the\s+)?(?P<r>left|right)\s+of\s+(?P<o>.+)$",
            "lr",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}(?:in\s+)?front\s+of\s+(?P<o>.+)$",
            "front",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}(?P<r>behind)\s+(?P<o>.+)$",
            "behind",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}(?:on\s+)?top\s+of\s+(?P<o>.+)$",
            "top",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}at\s+the\s+(?P<r>top|bottom)\s+of\s+(?P<o>.+)$",
            "tb",
        ),
        (
            rf"^(?P<s>.+?)\s+{copula}(?P<r>above|below|under|underneath)\s+(?P<o>.+)$",
            "vertical",
        ),
    ]
    for pattern, fixed_relation in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        subject = canonical_phrase(match.group("s"))
        reference = canonical_phrase(match.group("o"))
        if not subject or not reference or subject == reference:
            return None
        raw = match.groupdict().get("r")
        relation = fixed_relation if raw is None else raw.lower()
        if relation == "lr":
            relation = str(raw).lower()
        elif relation == "tb":
            relation = str(raw).lower()
        if relation not in REL_ALIAS:
            return None
        return subject, reference, REL_ALIAS[relation]
    return None

def load_records(dataset: str, data_root: Path, max_samples: Optional[int]) -> Tuple[List[Record], List[Dict[str, Any]]]:
    if dataset == "coco_two":
        annotation_path = data_root / "coco_qa_two_obj.json"
        image_dir = data_root / "val2017"
        image_path = lambda image_id: image_dir / f"{int(image_id):012d}.jpg"
    else:
        annotation_path = data_root / "vg_qa_two_obj.json"
        image_dir = data_root / "vg_images"
        image_path = lambda image_id: image_dir / f"{image_id}.jpg"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotation_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    records: List[Record] = []
    audit: List[Dict[str, Any]] = []
    for sid, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            audit.append({"sid": sid, "reason": "invalid_row", "row": row})
            continue
        image_id, caption, opposite_caption = row[0], str(row[1]), str(row[2])
        parsed = parse_relation_caption(caption)
        if parsed is None:
            audit.append({"sid": sid, "reason": "caption_parse_failed", "caption": caption, "opposite_caption": opposite_caption})
            continue
        subject, reference, relation = parsed
        path = image_path(image_id)
        if not path.exists():
            audit.append({"sid": sid, "reason": "image_missing", "image_id": str(image_id), "image_path": str(path)})
            continue
        records.append(
            Record(
                sid=sid,
                image_id=str(image_id),
                image_path=path,
                caption=caption,
                opposite_caption=opposite_caption,
                subject=subject,
                reference=reference,
                relation=relation,
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    return records, audit


def build_chat_prompt(processor: Any, subject: str, reference: str) -> str:
    prompt = (
        f"Where is the {subject} relative to the {reference}? "
        "Answer with one spatial relation."
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    for variant in (" " + phrase, phrase):
        token_ids = list(tokenizer(variant, add_special_tokens=False).input_ids)
        key = tuple(int(v) for v in token_ids)
        if not key or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, token_ids):
            matches.append((start, start + len(token_ids) - 1))
    if not matches:
        raise ValueError(f"Could not find token span for {phrase!r}.")
    # The user utterance is the last relevant portion of the chat prompt.
    return max(matches, key=lambda item: item[0])[1]


def configure_processor(model: Any, processor: Any) -> None:
    """Fix LLaVA placeholder expansion on recent Transformers; no-op elsewhere."""
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


def load_resume(path: Path, model_alias: str, dataset: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as loaded:
        meta = json.loads(str(loaded["metadata_json"].item()))
        if meta.get("model_alias") != model_alias or meta.get("dataset") != dataset:
            raise RuntimeError(f"Existing output metadata does not match requested {dataset}/{model_alias}: {path}")
        return {key: loaded[key] for key in loaded.files}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    if args.model not in SPECS:
        raise ValueError(f"Unknown model alias: {args.model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(args.output)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    fractions = parse_fractions(args.layer_fracs)
    data_root = Path(args.data_root)
    records, audit = load_records(args.dataset, data_root, args.max_samples)
    if not records:
        raise RuntimeError("No usable records after relation parsing and image existence checks.")

    audit_path = out_path.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    relation_counts = {rel: sum(r.relation == rel for r in records) for rel in sorted({r.relation for r in records})}
    print(f"[{args.dataset}] usable={len(records)} | relations={relation_counts} | audit={len(audit)}")

    resume = None if args.overwrite else load_resume(out_path, args.model, args.dataset)
    done_sids: set[int] = set()
    sample_indices: List[int] = []
    image_ids: List[str] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    captions: List[str] = []
    opposite_captions: List[str] = []
    vectors: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    selected_blocks: Optional[List[int]] = None
    hidden_size: Optional[int] = None
    decoder_blocks: Optional[int] = None

    if resume is not None:
        metadata = json.loads(str(resume["metadata_json"].item()))
        sample_indices = [int(v) for v in resume["sample_index"].tolist()]
        image_ids = [str(v) for v in resume["image_id"].tolist()]
        subjects = [str(v) for v in resume["subject"].tolist()]
        references = [str(v) for v in resume["reference"].tolist()]
        relations = [str(v) for v in resume["relation"].tolist()]
        captions = [str(v) for v in resume["caption"].tolist()]
        opposite_captions = [str(v) for v in resume["opposite_caption"].tolist()]
        vectors = [row for row in resume["relation_vectors"]]
        selected_blocks = [int(v) for v in resume["decoder_block_index"].tolist()]
        hidden_size = int(metadata["hidden_size"])
        decoder_blocks = int(metadata["decoder_blocks"])
        done_sids = set(sample_indices)
        print(f"Resuming {out_path}: {len(done_sids)} saved samples.")

    spec = SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}; "
            "the backend smoke test must be fixed before this run."
        )

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

        if args.image_mode == "shuffle":
            shuffled_paths = [record.image_path for record in records]
            rng = random.Random(args.seed)
            rng.shuffle(shuffled_paths)
            path_by_sid = {record.sid: path for record, path in zip(records, shuffled_paths)}
        else:
            path_by_sid = {record.sid: record.image_path for record in records}

        blank_image = Image.new("RGB", (512, 512), color=(0, 0, 0)) if args.image_mode == "blank" else None

        def save_progress() -> None:
            if not vectors or selected_blocks is None or hidden_size is None or decoder_blocks is None:
                return
            metadata = {
                "dataset": args.dataset,
                "model_alias": args.model,
                "repo_id": spec.repo_id,
                "transformers_version": transformers.__version__,
                "image_mode": args.image_mode,
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
                "image_id": np.asarray(image_ids, dtype=object),
                "subject": np.asarray(subjects, dtype=object),
                "reference": np.asarray(references, dtype=object),
                "relation": np.asarray(relations, dtype=object),
                "caption": np.asarray(captions, dtype=object),
                "opposite_caption": np.asarray(opposite_captions, dtype=object),
                "decoder_block_index": np.asarray(selected_blocks, dtype=np.int32),
                "relation_vectors": np.stack(vectors, axis=0).astype(np.float16),
            }
            atomic_save(out_path, arrays)

        for record in tqdm(records, desc=f"{args.dataset}:{args.model}"):
            if record.sid in done_sids:
                continue
            try:
                image = blank_image.copy() if blank_image is not None else Image.open(path_by_sid[record.sid]).convert("RGB")
                rendered = build_chat_prompt(processor, record.subject, record.reference)
                batch = processor(text=[rendered], images=[image], return_tensors="pt")
                batch = move_batch(batch, device)
                input_ids = batch["input_ids"][0].detach().cpu().tolist()
                subject_index = find_phrase_last_token(processor.tokenizer, input_ids, record.subject)
                reference_index = find_phrase_last_token(processor.tokenizer, input_ids, record.reference)
                if subject_index == reference_index:
                    raise RuntimeError("Subject/reference token positions collide.")

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
                    print(
                        f"{args.model}: decoder_blocks={decoder_blocks}, hidden={hidden_size}, "
                        f"selected_blocks={selected_blocks}"
                    )
                assert selected_blocks is not None
                # state[0] is embedding output; decoder block k is state[k + 1].
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
                image_ids.append(record.image_id)
                subjects.append(record.subject)
                references.append(record.reference)
                relations.append(record.relation)
                captions.append(record.caption)
                opposite_captions.append(record.opposite_caption)
                vectors.append(relation_vector)

                del outputs, states, batch
                if len(vectors) % args.save_every == 0:
                    save_progress()
            except Exception as exc:
                errors.append(
                    {
                        "sid": record.sid,
                        "image_id": record.image_id,
                        "caption": record.caption,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-8:],
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        save_progress()
        error_path = out_path.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"Saved {len(vectors)}/{len(records)} states to {out_path} | "
            f"errors={len(errors)} | elapsed={(time.time() - started) / 60.0:.1f} min"
        )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
