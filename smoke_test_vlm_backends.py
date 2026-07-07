#!/usr/bin/env python3
"""Sequential visual-forward smoke test for the 12 VLM backends used in
'Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models'.

This is intentionally NOT a COCO/VG experiment. It verifies that each model
can be downloaded/loaded, processed with one image + a clean spatial prompt,
run through a forward pass with hidden states enabled, and expose the subject /
reference word-token positions needed by later spatial-axis probes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

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
    low_cpu_mem_usage: bool = True


# Exact aliases/checkpoints used by the paper's public extraction utility.
SPECS: Dict[str, ModelSpec] = {
    "llava-7b": ModelSpec(
        "llava-7b", "llava-hf/llava-1.5-7b-hf", "LlavaForConditionalGeneration", "float16"
    ),
    "llava-13b": ModelSpec(
        "llava-13b", "llava-hf/llava-1.5-13b-hf", "LlavaForConditionalGeneration", "float16"
    ),
    "qwen2-2b": ModelSpec(
        "qwen2-2b", "Qwen/Qwen2-VL-2B-Instruct", "Qwen2VLForConditionalGeneration", "auto"
    ),
    "qwen-3b": ModelSpec(
        "qwen-3b", "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2_5_VLForConditionalGeneration", "float16"
    ),
    "qwen-7b": ModelSpec(
        "qwen-7b", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2_5_VLForConditionalGeneration", "float16"
    ),
    "llama-11b": ModelSpec(
        "llama-11b", "meta-llama/Llama-3.2-11B-Vision-Instruct", "MllamaForConditionalGeneration", "bfloat16"
    ),
    "internvl-1b": ModelSpec(
        "internvl-1b", "OpenGVLab/InternVL3-1B-hf", "InternVLForConditionalGeneration", "bfloat16", True
    ),
    "internvl-2b": ModelSpec(
        "internvl-2b", "OpenGVLab/InternVL3-2B-hf", "InternVLForConditionalGeneration", "bfloat16", True
    ),
    "internvl-8b": ModelSpec(
        "internvl-8b", "OpenGVLab/InternVL3-8B-hf", "InternVLForConditionalGeneration", "bfloat16", True
    ),
    "internvl-14b": ModelSpec(
        "internvl-14b", "OpenGVLab/InternVL3-14B-hf", "InternVLForConditionalGeneration", "bfloat16", True
    ),
    "gemma-4b": ModelSpec(
        "gemma-4b", "google/gemma-3-4b-it", "Gemma3ForConditionalGeneration", "bfloat16", True
    ),
    "gemma-12b": ModelSpec(
        "gemma-12b", "google/gemma-3-12b-it", "Gemma3ForConditionalGeneration", "bfloat16", True
    ),
}

ALL_ALIASES = list(SPECS.keys())


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype name: {name}")


def build_chat_prompt(processor: Any, prompt: str) -> str:
    """Use the processor's own chat template; all 12 paper backends support this path."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move tensors while retaining integer token IDs. BatchFeature.to(dtype) is avoided."""
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def target_token_positions(processor: Any, input_ids: torch.Tensor, words: List[str]) -> Dict[str, Dict[str, Any]]:
    tokenizer = processor.tokenizer
    ids = input_ids[0].detach().cpu().tolist()
    result: Dict[str, Dict[str, Any]] = {}

    for word in words:
        word_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not word_ids:
            result[word] = {"last_subtoken_id": None, "positions": []}
            continue
        final_id = int(word_ids[-1])
        positions = [idx for idx, token_id in enumerate(ids) if int(token_id) == final_id]
        result[word] = {
            "last_subtoken_id": final_id,
            "token_piece": tokenizer.convert_ids_to_tokens(final_id),
            "positions": positions,
        }
    return result


def first_hidden_shape(outputs: Any) -> Tuple[Optional[int], Optional[List[int]], Optional[int]]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        return None, None, None
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        return None, None, None
    final = hidden_states[-1]
    if not torch.is_tensor(final):
        return len(hidden_states), None, None
    return len(hidden_states), list(final.shape), int(final.shape[-1])


def load_and_forward(spec: ModelSpec, image: Image.Image, prompt: str, target_words: List[str], device: str, attn_impl: str) -> Dict[str, Any]:
    started = time.time()
    result: Dict[str, Any] = {
        "alias": spec.alias,
        "repo_id": spec.repo_id,
        "class": spec.model_class,
        "dtype": spec.dtype_name,
        "status": "failed",
    }

    model = None
    processor = None
    try:
        model_cls = getattr(transformers, spec.model_class, None)
        if model_cls is None:
            raise RuntimeError(
                f"transformers=={transformers.__version__} has no {spec.model_class}. "
                "Upgrade transformers before retrying this backend."
            )

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": resolve_dtype(spec.dtype_name),
            "low_cpu_mem_usage": spec.low_cpu_mem_usage,
            "device_map": device,
            "trust_remote_code": spec.trust_remote_code,
        }
        if attn_impl != "none":
            load_kwargs["attn_implementation"] = attn_impl

        # Some model classes reject a particular attention implementation. Do not silently retry;
        # surfacing the exact error is the point of a backend smoke test.
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)

        rendered = build_chat_prompt(processor, prompt)
        batch = processor(text=[rendered], images=[image], return_tensors="pt")
        batch = move_batch_to_device(batch, torch.device(device))

        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=True, use_cache=False)

        n_hidden, final_shape, hidden_size = first_hidden_shape(outputs)
        token_hits = target_token_positions(processor, batch["input_ids"], target_words)

        # Verify that each requested lexical target occurs in the input exactly enough to be used
        # by a later word-token extractor. A non-empty list is sufficient for this smoke check.
        missing_targets = [word for word, data in token_hits.items() if not data.get("positions")]

        result.update(
            {
                "status": "ok" if not missing_targets else "warning",
                "elapsed_s": round(time.time() - started, 2),
                "input_shape": list(batch["input_ids"].shape),
                "hidden_state_count": n_hidden,
                "final_hidden_shape": final_shape,
                "hidden_size": hidden_size,
                "target_tokens": token_hits,
                "missing_target_words": missing_targets,
                "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)
                if torch.cuda.is_available()
                else None,
            }
        )
    except Exception as exc:
        result.update(
            {
                "elapsed_s": round(time.time() - started, 2),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
                "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)
                if torch.cuda.is_available()
                else None,
            }
        )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return result


def parse_aliases(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return ALL_ALIASES
    aliases = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in aliases if item not in SPECS]
    if unknown:
        raise ValueError(f"Unknown model aliases: {unknown}. Valid: {', '.join(ALL_ALIASES)}")
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Any one existing COCO/VG image path.")
    parser.add_argument("--models", default="all", help="Comma-separated aliases or 'all'.")
    parser.add_argument(
        "--prompt",
        default="Where is the cat relative to the dog? Answer with one spatial relation.",
        help="Clean prompt used only to verify visual forward + target token availability.",
    )
    parser.add_argument("--target-words", default="cat,dog")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager", "none"],
        help="Use sdpa first; only use flash_attention_2 after verifying it is installed.",
    )
    parser.add_argument("--output", required=True, help="JSON summary path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test.")

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image does not exist: {image_path}")
    image = Image.open(image_path).convert("RGB")

    aliases = parse_aliases(args.models)
    words = [item.strip() for item in args.target_words.split(",") if item.strip()]

    print(f"transformers={transformers.__version__}")
    print(f"CUDA device={torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"image={image_path} size={image.size} models={aliases}")

    results: List[Dict[str, Any]] = []
    for index, alias in enumerate(aliases, start=1):
        print("=" * 88)
        print(f"[{index}/{len(aliases)}] {alias}: {SPECS[alias].repo_id}", flush=True)
        record = load_and_forward(
            SPECS[alias],
            image=image,
            prompt=args.prompt,
            target_words=words,
            device=args.device,
            attn_impl=args.attn_impl,
        )
        results.append(record)
        if record["status"] in {"ok", "warning"}:
            print(
                f"[{record['status'].upper()}] layers={record.get('hidden_state_count')} "
                f"shape={record.get('final_hidden_shape')} "
                f"peak={record.get('cuda_peak_allocated_gb')} GB "
                f"elapsed={record.get('elapsed_s')}s"
            )
            print(f"target positions: {record.get('target_tokens')}")
        else:
            print(f"[FAILED] {record.get('error_type')}: {record.get('error')}")

    summary = {
        "transformers_version": transformers.__version__,
        "image": str(image_path),
        "prompt": args.prompt,
        "target_words": words,
        "device": args.device,
        "attn_impl": args.attn_impl,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(item["status"] == "ok" for item in results)
    warn = sum(item["status"] == "warning" for item in results)
    fail = len(results) - ok - warn
    print("=" * 88)
    print(f"Done: ok={ok}, warning={warn}, failed={fail}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
