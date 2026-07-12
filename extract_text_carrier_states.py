#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract text-carrier hidden states for spatial-relation carrier ablations.

Supported datasets:
  - controlled_A
  - coco_two
  - vg_two

For every selected decoder block, save:
  subject_last_states
  reference_last_states
  subject_mean_states
  reference_mean_states
  question_last_states
  relation_anchor_states
  question_mean_states
  answer_readout_states
  relation_vectors = subject_last - reference_last

The VLM remains frozen. Labels are only stored as metadata.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


PROMPT_TEMPLATE = (
    "Where is the {subject} relative to the {reference}? "
    "Answer with one spatial relation."
)

CARRIER_KEYS = [
    "subject_last_states",
    "reference_last_states",
    "subject_mean_states",
    "reference_mean_states",
    "question_last_states",
    "relation_anchor_states",
    "question_mean_states",
    "answer_readout_states",
    "relation_vectors",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["controlled_A", "coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-path", default="prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    p.add_argument("--controlled-module", default="", help="Optional explicit controlled extractor module name.")
    p.add_argument("--model", required=True)
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
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Layer fraction must be in (0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("--layer-fracs is empty")
    return sorted(set(values))


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def import_controlled_module(explicit: str = ""):
    names = []
    if explicit:
        names.append(explicit)
    names.extend([
        "extract_controlled_relation_states_standalone",
        "extract_controlledA_relation_states_standalone",
    ])
    errors = []
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ImportError(
        "Could not import a controlled-data extractor module. Tried:\n  "
        + "\n  ".join(errors)
    )


def import_two_object_module():
    return importlib.import_module("extract_two_object_relation_states")


def build_prompt(processor: Any, subject: str, reference: str) -> Tuple[str, str]:
    prompt_text = PROMPT_TEMPLATE.format(subject=subject, reference=reference)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt_text
    return rendered, prompt_text


def move_batch(batch: Any, device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out.input_ids
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    width = len(needle)
    return [
        start
        for start in range(len(haystack) - width + 1)
        if list(haystack[start:start + width]) == list(needle)
    ]


def find_phrase_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    phrase: str,
    *,
    lo: int = 0,
    hi: Optional[int] = None,
    include_article_variants: bool = False,
) -> List[Tuple[int, int]]:
    if hi is None:
        hi = len(input_ids) - 1
    variants = [phrase, " " + phrase]
    if include_article_variants:
        variants.extend(["the " + phrase, " the " + phrase])

    seen = set()
    spans: List[Tuple[int, int]] = []
    for variant in variants:
        ids = tokenizer_ids(tokenizer, variant)
        key = tuple(ids)
        if not ids or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, ids):
            end = start + len(ids) - 1
            if start >= lo and end <= hi:
                spans.append((start, end))
    return sorted(set(spans))


def locate_prompt_span(tokenizer: Any, input_ids: Sequence[int], prompt_text: str) -> Tuple[int, int]:
    spans = find_phrase_spans(tokenizer, input_ids, prompt_text)
    if spans:
        return max(spans, key=lambda x: x[0])

    starts = find_phrase_spans(tokenizer, input_ids, "Where is the")
    if not starts:
        starts = find_phrase_spans(tokenizer, input_ids, "Where is")
    ends = find_phrase_spans(tokenizer, input_ids, "spatial relation")
    if not starts or not ends:
        raise ValueError("Could not locate the full user-question span in tokenized prompt.")

    valid = [
        (s[0], e[1])
        for s in starts
        for e in ends
        if s[0] <= e[1]
    ]
    if not valid:
        raise ValueError("Question start/end anchors were found but could not form a valid span.")
    return max(valid, key=lambda x: x[0])


def locate_object_spans(
    tokenizer: Any,
    input_ids: Sequence[int],
    question_span: Tuple[int, int],
    subject: str,
    reference: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    q_start, q_end = question_span
    subject_spans = find_phrase_spans(
        tokenizer,
        input_ids,
        subject,
        lo=q_start,
        hi=q_end,
        include_article_variants=True,
    )
    reference_spans = find_phrase_spans(
        tokenizer,
        input_ids,
        reference,
        lo=q_start,
        hi=q_end,
        include_article_variants=True,
    )

    valid = [
        (s, r)
        for s in subject_spans
        for r in reference_spans
        if s[1] < r[0]
    ]
    if not valid:
        raise ValueError(
            f"Could not find ordered subject/reference spans: "
            f"subject={subject!r}, reference={reference!r}, "
            f"subject_spans={subject_spans}, reference_spans={reference_spans}"
        )

    return max(valid, key=lambda pair: (pair[1][0], pair[0][0]))


def configure_processor(model: Any, processor: Any) -> None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if (
        vision_config is not None
        and hasattr(processor, "patch_size")
        and hasattr(vision_config, "patch_size")
    ):
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
    raise RuntimeError("No decoder hidden states returned by model backend.")


def select_blocks(n_blocks: int, fractions: Sequence[float]) -> List[int]:
    result = sorted({int(round(frac * (n_blocks - 1))) for frac in fractions})
    if not result or min(result) < 0 or max(result) >= n_blocks:
        raise RuntimeError(f"Invalid selected blocks {result} for n_blocks={n_blocks}")
    return result


def atomic_save(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def record_image(record: Any) -> Image.Image:
    if hasattr(record, "image"):
        return record.image.copy().convert("RGB")
    if hasattr(record, "image_path"):
        return Image.open(record.image_path).convert("RGB")
    raise TypeError("Record has neither image nor image_path.")


def record_group(record: Any) -> str:
    if hasattr(record, "group"):
        value = record.group
        return str(value() if callable(value) else value)
    return " || ".join(sorted((str(record.subject), str(record.reference))))


def load_data(args: argparse.Namespace):
    if args.dataset == "controlled_A":
        module = import_controlled_module(args.controlled_module)
        records, audit = module.load_records(
            Path(args.prompt_path),
            dataset_key="Controlled_Images_A",
            keep_relations=["left", "right", "on", "under"],
            download=args.download,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
        )
        return module, records, audit

    module = import_two_object_module()
    records, audit = module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    return module, records, audit


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fractions = parse_fractions(args.layer_fracs)
    module, records, audit = load_data(args)
    if not records:
        raise RuntimeError("No usable records.")

    if args.model not in module.SPECS:
        raise ValueError(f"Model {args.model!r} not available in {module.__name__}.SPECS")
    spec = module.SPECS[args.model]

    out_path = Path(args.output)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    out_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sample_ids: List[int] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    groups: List[str] = []
    questions: List[str] = []
    token_indices: Dict[str, List[int]] = {
        "subject_start": [],
        "subject_end": [],
        "reference_start": [],
        "reference_end": [],
        "question_start": [],
        "question_end": [],
        "relation_anchor": [],
        "answer_readout": [],
    }
    carrier_lists: Dict[str, List[np.ndarray]] = {key: [] for key in CARRIER_KEYS}
    errors: List[Dict[str, Any]] = []

    selected_blocks: Optional[List[int]] = None
    hidden_size: Optional[int] = None
    decoder_blocks: Optional[int] = None

    if out_path.exists() and not args.overwrite:
        with np.load(out_path, allow_pickle=True) as z:
            metadata = json.loads(str(z["metadata_json"].item()))
            if metadata.get("dataset") != args.dataset or metadata.get("model_alias") != args.model:
                raise RuntimeError("Existing output metadata does not match requested run.")
            sample_ids = [int(x) for x in z["sid"].tolist()]
            subjects = [str(x) for x in z["subject"].tolist()]
            references = [str(x) for x in z["reference"].tolist()]
            relations = [str(x) for x in z["relation"].tolist()]
            groups = [str(x) for x in z["group"].tolist()]
            questions = [str(x) for x in z["question"].tolist()]
            for key in token_indices:
                token_indices[key] = [int(x) for x in z[f"{key}_index"].tolist()]
            for key in CARRIER_KEYS:
                if key not in z.files:
                    raise RuntimeError(f"Resume file missing key {key}; rerun with --overwrite.")
                carrier_lists[key] = [row for row in z[key]]
            selected_blocks = [int(x) for x in z["decoder_block_index"].tolist()]
            hidden_size = int(metadata["hidden_size"])
            decoder_blocks = int(metadata["decoder_blocks"])
        print(f"Resuming {out_path}: {len(sample_ids)} samples.")

    done = set(sample_ids)

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )

    model = None
    processor = None
    started = time.time()

    def save_progress() -> None:
        if not sample_ids or selected_blocks is None or hidden_size is None or decoder_blocks is None:
            return
        metadata = {
            "dataset": args.dataset,
            "model_alias": args.model,
            "repo_id": spec.repo_id,
            "transformers_version": transformers.__version__,
            "prompt_template": PROMPT_TEMPLATE,
            "layer_fractions": fractions,
            "decoder_blocks": decoder_blocks,
            "hidden_size": hidden_size,
            "n_requested": len(records),
            "n_saved": len(sample_ids),
            "relation_counts": {
                rel: relations.count(rel)
                for rel in sorted(set(relations))
            },
            "carrier_keys": CARRIER_KEYS,
            "seed": args.seed,
            "saved_at_unix": time.time(),
        }
        arrays: Dict[str, Any] = {
            "metadata_json": np.array(json.dumps(metadata), dtype=object),
            "sid": np.asarray(sample_ids, dtype=np.int64),
            "sample_index": np.asarray(sample_ids, dtype=np.int64),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "relation": np.asarray(relations, dtype=object),
            "group": np.asarray(groups, dtype=object),
            "question": np.asarray(questions, dtype=object),
            "decoder_block_index": np.asarray(selected_blocks, dtype=np.int32),
        }
        for key, values in token_indices.items():
            arrays[f"{key}_index"] = np.asarray(values, dtype=np.int32)
        for key, values in carrier_lists.items():
            arrays[key] = np.stack(values, axis=0).astype(np.float16)
        atomic_save(out_path, arrays)

    try:
        load_kwargs: Dict[str, Any] = {
            "dtype": resolve_dtype(spec.dtype_name),
            "low_cpu_mem_usage": True,
            "trust_remote_code": spec.trust_remote_code,
            "device_map": {"": args.device},
        }
        if args.attn_impl != "none":
            load_kwargs["attn_implementation"] = args.attn_impl

        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        configure_processor(model, processor)
        device = torch.device(args.device)

        for record in tqdm(records, desc=f"{args.dataset}:{args.model}:carriers"):
            sid = int(record.sid)
            if sid in done:
                continue

            try:
                subject = str(record.subject)
                reference = str(record.reference)
                rendered, prompt_text = build_prompt(processor, subject, reference)
                image = record_image(record)

                batch = processor(
                    text=[rendered],
                    images=[image],
                    return_tensors="pt",
                )
                batch = move_batch(batch, device)
                input_ids = batch["input_ids"][0].detach().cpu().tolist()

                question_span = locate_prompt_span(
                    processor.tokenizer,
                    input_ids,
                    prompt_text,
                )
                subject_span, reference_span = locate_object_spans(
                    processor.tokenizer,
                    input_ids,
                    question_span,
                    subject,
                    reference,
                )

                relation_spans = find_phrase_spans(
                    processor.tokenizer,
                    input_ids,
                    "spatial relation",
                    lo=question_span[0],
                    hi=question_span[1],
                )
                if not relation_spans:
                    raise ValueError("Could not locate generic 'spatial relation' task anchor.")
                relation_anchor = max(relation_spans, key=lambda x: x[0])[1]
                answer_readout = len(input_ids) - 1

                with torch.inference_mode():
                    outputs = model(
                        **batch,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )

                states = hidden_tuple(outputs)
                final = states[-1]
                if final.ndim != 3 or final.shape[0] != 1:
                    raise RuntimeError(f"Unexpected final hidden shape {tuple(final.shape)}")
                if int(final.shape[1]) != len(input_ids):
                    raise RuntimeError(
                        f"Token/hidden length mismatch: input={len(input_ids)} "
                        f"hidden={int(final.shape[1])}"
                    )

                if selected_blocks is None:
                    decoder_blocks = len(states) - 1
                    selected_blocks = select_blocks(decoder_blocks, fractions)
                    hidden_size = int(final.shape[-1])
                    print(
                        f"{args.model}: decoder_blocks={decoder_blocks}, "
                        f"hidden={hidden_size}, selected_blocks={selected_blocks}"
                    )

                per_sample: Dict[str, List[np.ndarray]] = {key: [] for key in CARRIER_KEYS}
                for block in selected_blocks:
                    h = states[block + 1][0]

                    subject_last = h[subject_span[1]]
                    reference_last = h[reference_span[1]]
                    subject_mean = h[subject_span[0]:subject_span[1] + 1].mean(dim=0)
                    reference_mean = h[reference_span[0]:reference_span[1] + 1].mean(dim=0)
                    question_last = h[question_span[1]]
                    relation_state = h[relation_anchor]
                    question_mean = h[question_span[0]:question_span[1] + 1].mean(dim=0)
                    readout_state = h[answer_readout]

                    values = {
                        "subject_last_states": subject_last,
                        "reference_last_states": reference_last,
                        "subject_mean_states": subject_mean,
                        "reference_mean_states": reference_mean,
                        "question_last_states": question_last,
                        "relation_anchor_states": relation_state,
                        "question_mean_states": question_mean,
                        "answer_readout_states": readout_state,
                        "relation_vectors": subject_last - reference_last,
                    }
                    for key, tensor in values.items():
                        per_sample[key].append(
                            tensor.detach().float().cpu().numpy()
                        )

                sample_ids.append(sid)
                subjects.append(subject)
                references.append(reference)
                relations.append(str(record.relation))
                groups.append(record_group(record))
                questions.append(prompt_text)

                index_values = {
                    "subject_start": subject_span[0],
                    "subject_end": subject_span[1],
                    "reference_start": reference_span[0],
                    "reference_end": reference_span[1],
                    "question_start": question_span[0],
                    "question_end": question_span[1],
                    "relation_anchor": relation_anchor,
                    "answer_readout": answer_readout,
                }
                for key, value in index_values.items():
                    token_indices[key].append(int(value))

                for key in CARRIER_KEYS:
                    carrier_lists[key].append(
                        np.stack(per_sample[key], axis=0).astype(np.float16)
                    )

                del outputs, states, batch, image
                if len(sample_ids) % args.save_every == 0:
                    save_progress()

            except Exception as exc:
                errors.append({
                    "sid": sid,
                    "subject": str(getattr(record, "subject", "")),
                    "reference": str(getattr(record, "reference", "")),
                    "relation": str(getattr(record, "relation", "")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-10:],
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_progress()
        out_path.with_suffix(".errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Saved {len(sample_ids)}/{len(records)} samples to {out_path} | "
            f"errors={len(errors)} | elapsed={(time.time() - started) / 60.0:.1f} min"
        )

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
