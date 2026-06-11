#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a pure Hugging Face LLaVA-1.5 baseline while optionally replacing the
checkpoint's text configuration with the parameter values used by the repo's
custom LLaMAConfig().

This script does NOT use:
  - model_zoo/llava/modeling_llava*.py
  - model_zoo/llama/modeling_llama*.py
  - AdaptVis attention scaling
  - the repo's custom greedy-search monkeypatch

It therefore isolates:

    full HF LLaVA merge
  + full HF LLaMA forward
  + checkpoint text config OR custom-equivalent text config

Place this file in the AdaptVis repository root and run it there.

Recommended environment:
    transformers==4.39.1

Examples
--------
1. HF model + custom-equivalent LLaMA configuration:

python3 run_llava15_hf_customconfig_baseline.py \
  --dataset Controlled_Images_A \
  --text-config custom \
  --device cuda \
  --dtype float32 \
  --download

2. Standard HF baseline using the checkpoint's own text configuration:

python3 run_llava15_hf_customconfig_baseline.py \
  --dataset Controlled_Images_A \
  --text-config checkpoint \
  --device cuda \
  --dtype float32 \
  --download

For a controlled comparison, keep every argument identical except
--text-config.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlamaConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
)

from dataset_zoo import get_dataset


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "a272c74"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HF LLaVA-1.5 with either its checkpoint text config or "
            "an HF LlamaConfig equivalent to the repo's custom LLaMAConfig()."
        )
    )
    parser.add_argument(
        "--dataset",
        default="Controlled_Images_A",
        choices=["Controlled_Images_A", "Controlled_Images_B"],
    )
    parser.add_argument(
        "--option",
        default="four",
        choices=["two", "four", "six"],
    )
    parser.add_argument(
        "--text-config",
        default="custom",
        choices=["custom", "checkpoint"],
        help=(
            "'custom': use the values from model_zoo/llama/configuration_llama.py; "
            "'checkpoint': retain the official HF checkpoint text config."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-dir",
        default="data",
        help="Hugging Face model cache directory.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16", "auto"],
        help=(
            "Use float32 for comparison with the current main_aro loading path. "
            "Changing dtype is another experimental variable."
        ),
    )
    parser.add_argument("--max-new-tokens", default=100, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Evaluate only the first N samples. Omit for the full dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow the repository dataset loader to download missing data.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. A descriptive default is used when omitted.",
    )
    parser.add_argument(
        "--ignore-mismatched-sizes",
        action="store_true",
        help=(
            "Allow mismatched checkpoint tensors. Keep this disabled initially "
            "so that accidental random initialization is not hidden."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to Hugging Face loaders.",
    )
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_custom_equivalent_hf_llama_config() -> LlamaConfig:
    """
    Convert the repository's custom LLaMAConfig() defaults into the official
    transformers.LlamaConfig class.

    Directly assigning the repository's custom LLaMAConfig object to an HF
    LlavaConfig is unsafe because AutoModelForCausalLM dispatches by the
    official configuration class type.

    The following values match:
        model_zoo/llama/configuration_llama.py

    Additional HF-only fields are selected to match the behavior of the
    repository's custom LLaMA implementation:
      - num_key_value_heads=32: standard multi-head attention
      - max_position_embeddings=2048: custom RotaryEmbedding default
      - rope_theta=10000.0: custom RotaryEmbedding base
      - attention_bias=False: custom Q/K/V and O projections have no bias
      - attention_dropout=0.0: custom attention has no dropout
      - pretraining_tp=1: no tensor-parallel slicing
    """
    return LlamaConfig(
        vocab_size=32064,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )


CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "pretraining_tp",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "attention_bias",
    "attention_dropout",
)


def config_snapshot(config: LlamaConfig) -> Dict[str, Any]:
    return {name: getattr(config, name, None) for name in CONFIG_FIELDS}


def print_config_comparison(
    checkpoint_config: LlamaConfig,
    active_config: LlamaConfig,
) -> None:
    checkpoint_values = config_snapshot(checkpoint_config)
    active_values = config_snapshot(active_config)

    print("\n" + "=" * 100)
    print("TEXT CONFIGURATION")
    print("=" * 100)
    print(f"{'field':30s} {'checkpoint':30s} {'active':30s} changed")
    for field in CONFIG_FIELDS:
        old = checkpoint_values[field]
        new = active_values[field]
        print(
            f"{field:30s} {str(old):30.30s} {str(new):30.30s} "
            f"{'YES' if old != new else 'no'}"
        )
    print("=" * 100 + "\n")


def load_hf_model(
    args: argparse.Namespace,
) -> Tuple[LlavaForConditionalGeneration, AutoProcessor, Dict[str, Any]]:
    print(f"Loading top-level LLaVA config from {args.model}@{args.revision}")

    llava_config = LlavaConfig.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    checkpoint_text_config = copy.deepcopy(llava_config.text_config)

    if args.text_config == "custom":
        llava_config.text_config = build_custom_equivalent_hf_llama_config()

    # Force the standard eager HF attention path. This is not AdaptVis; it only
    # avoids silently switching between eager, SDPA, and FlashAttention.
    llava_config._attn_implementation = "eager"

    print_config_comparison(
        checkpoint_config=checkpoint_text_config,
        active_config=llava_config.text_config,
    )

    load_kwargs: Dict[str, Any] = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "config": llava_config,
        "ignore_mismatched_sizes": args.ignore_mismatched_sizes,
        "output_loading_info": True,
        "trust_remote_code": args.trust_remote_code,
    }

    dtype = resolve_torch_dtype(args.dtype)
    if args.dtype != "float32":
        load_kwargs["torch_dtype"] = dtype
    # For float32, omit torch_dtype to reproduce the normal from_pretrained
    # construction path used by the current repository.

    loaded = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        **load_kwargs,
    )
    model, loading_info = loaded

    print("\n" + "=" * 100)
    print("CHECKPOINT LOADING INFORMATION")
    print("=" * 100)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading_info.get(key, [])
        print(f"{key}: {len(values)}")
        for item in values[:20]:
            print(f"  {item}")
        if len(values) > 20:
            print(f"  ... {len(values) - 20} more")
    print("=" * 100 + "\n")

    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        print(
            "[WARNING] Some language-model tensors were not loaded exactly. "
            "Do not interpret the accuracy as a clean configuration ablation "
            "until these entries are understood."
        )

    model.eval()
    model.to(args.device)

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    print("Active model classes:")
    print(f"  LLaVA: {type(model).__module__}.{type(model).__name__}")
    print(
        "  language_model: "
        f"{type(model.language_model).__module__}."
        f"{type(model.language_model).__name__}"
    )
    print(f"  language text config: {type(model.language_model.config).__name__}")
    print(f"  attention implementation: {model.config._attn_implementation}")
    print(f"  first parameter dtype: {next(model.parameters()).dtype}")
    print(f"  generation pad token: {model.generation_config.pad_token_id}")
    print(f"  generation bos token: {model.generation_config.bos_token_id}")
    print(f"  generation eos token: {model.generation_config.eos_token_id}")

    return model, processor, loading_info


def norm_gold(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def is_correct(gold: Any, generation: str) -> bool:
    gold_text = norm_gold(gold)
    generation_text = str(generation).strip()

    if not gold_text:
        return False

    correct = (
        gold_text in generation_text
        or gold_text.lower() in generation_text.lower()
    )
    if gold_text.lower() == "on" and "front" in generation_text.lower():
        correct = False
    return bool(correct)


def decode_generated(
    processor: AutoProcessor,
    output: Any,
    prompt_length: int,
) -> str:
    if hasattr(output, "sequences"):
        sequences = output.sequences
    elif isinstance(output, dict):
        sequences = output.get("sequences")
    else:
        sequences = output[0]

    if sequences is None:
        return ""

    return processor.decode(
        sequences[0][int(prompt_length):],
        skip_special_tokens=True,
    ).strip()


def first_step_confidence(output: Any) -> float:
    scores = output.scores if hasattr(output, "scores") else output.get("scores")
    if scores is None or len(scores) == 0:
        return 0.0
    probabilities = torch.softmax(scores[0].detach().float(), dim=-1)
    return float(probabilities[0].max().cpu())


def load_prompts(dataset: str, option: str) -> Tuple[List[str], List[Any]]:
    path = Path(f"prompts/{dataset}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Run this script from the "
            "AdaptVis repository root."
        )

    prompts: List[str] = []
    answers: List[Any] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            prompts.append(item["question"])
            answers.append(item["answer"])
    return prompts, answers


def extract_images_from_batch(batch: Dict[str, Any]) -> Iterable[Any]:
    """
    Match the nesting used by model_zoo/llava15.py:

        for i_option in batch["image_options"]:
            for image in i_option:
                ...

    The repository's Controlled Images experiments normally use batch_size=1.
    """
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
) -> Dict[str, Any]:
    prompts, answers = load_prompts(args.dataset, args.option)

    dataset = get_dataset(
        args.dataset,
        image_preprocess=None,
        download=args.download,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=None,
    )

    total_available = min(len(prompts), len(dataset))
    if args.limit is not None:
        total_target = min(total_available, args.limit)
    else:
        total_target = total_available

    model_dtype = next(model.parameters()).dtype
    records: List[Dict[str, Any]] = []
    correct_count = 0
    sample_index = 0

    progress = tqdm(total=total_target, desc="HF LLaVA baseline")

    for batch in loader:
        for image in extract_images_from_batch(batch):
            if sample_index >= total_target:
                break

            prompt = prompts[sample_index]
            gold = norm_gold(answers[sample_index])

            inputs = processor(
                text=prompt,
                images=image,
                padding="max_length",
                return_tensors="pt",
                max_length=77,
            )
            inputs = inputs.to(args.device)

            # BatchFeature.to(device) does not necessarily align floating image
            # tensors with a half/bfloat16 model.
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    device=args.device,
                    dtype=model_dtype,
                )

            prompt_length = int(inputs["input_ids"].shape[-1])

            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generation = decode_generated(
                processor=processor,
                output=output,
                prompt_length=prompt_length,
            )
            confidence = first_step_confidence(output)
            correct = is_correct(gold, generation)
            correct_count += int(correct)

            record = {
                "sid": sample_index,
                "prompt": prompt,
                "generation": generation,
                "gold": gold,
                "correct": correct,
                "first_step_confidence": confidence,
                "rounded_confidence": float(np.round(confidence, 2)),
            }
            records.append(record)

            print(
                f"\n[SID {sample_index}] "
                f"gold={gold!r} pred={generation!r} "
                f"conf={confidence:.6f} correct={correct}"
            )

            sample_index += 1
            progress.update(1)

        if sample_index >= total_target:
            break

    progress.close()

    accuracy = correct_count / max(sample_index, 1)
    summary = {
        "model": args.model,
        "revision": args.revision,
        "dataset": args.dataset,
        "option": args.option,
        "implementation": "transformers.LlavaForConditionalGeneration",
        "language_model_implementation": (
            "transformers.LlamaForCausalLM"
        ),
        "text_config_mode": args.text_config,
        "dtype_argument": args.dtype,
        "model_parameter_dtype": str(model_dtype),
        "num_samples": sample_index,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "records": records,
    }

    print("\n" + "=" * 100)
    print(
        f"RESULT: {correct_count}/{sample_index} "
        f"accuracy={accuracy:.6f} "
        f"text_config={args.text_config}"
    )
    print("=" * 100)

    return summary


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    model, processor, _ = load_hf_model(args)
    summary = evaluate(args, model, processor)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(
            "output/"
            f"hf_llava15_{args.dataset}_"
            f"{args.text_config}_config_baseline.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
