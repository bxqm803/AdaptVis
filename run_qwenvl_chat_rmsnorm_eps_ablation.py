#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-VL-Chat RMSNorm-epsilon ablation on the AdaptVis Controlled Images datasets.

This is a BASE-MODEL epsilon sweep only: it does not install LLaVA's AdaptVis
pre-softmax image-logit patch. Qwen-VL-Chat uses a different multimodal input
path and a fused QKV projection, so its AdaptVis implementation must be added
and validated separately.

The script changes only every language-backbone RMSNorm module's ``eps`` field.
For the Qwen-VL remote implementation, RMSNorm uses:

    x / sqrt(mean(x^2) + eps)

and stores epsilon as ``module.eps`` (not LLaMA's ``variance_epsilon``).

Run from the AdaptVis repository root, where dataset_zoo.py and prompts/ exist.

Example
-------
python3 run_qwenvl_chat_rmsnorm_eps_ablation.py \
  --dataset Controlled_Images_A \
  --option four \
  --rms-norm-eps 1e-6 \
  --device cuda \
  --dtype float32 \
  --num-workers 0 \
  --seed 1 \
  --download \
  --output output/qwenvl_chat_A_eps1e6.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_zoo import get_dataset

try:
    # Same collate path used by the LLaVA controlled-image runner.
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "Qwen/Qwen-VL-Chat"
DEFAULT_REVISION = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen-VL-Chat while changing only language-backbone "
            "RMSNorm epsilon."
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
        "--rms-norm-eps",
        default=1e-6,
        type=float,
        help="Value assigned to all Qwen language RMSNorm modules.",
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
        choices=["float32", "float16", "bfloat16"],
        help=(
            "Qwen remote-code loading precision. Use the same precision for "
            "every epsilon condition."
        ),
    )
    parser.add_argument("--max-new-tokens", default=100, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Evaluate only the first N samples.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow the repository dataset loader to download missing data.",
    )
    parser.add_argument(
        "--image-cache-dir",
        default="data/qwenvl_chat_eval_images",
        help=(
            "Directory for lossless PNG copies of dataset PIL images. Qwen-VL "
            "remote code consumes local image paths inside <img> tags."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        default="auto",
        choices=["auto", "raw", "llava"],
        help=(
            "How to convert the existing prompt JSON question into the Qwen "
            "chat query. 'auto' removes a surrounding LLaVA USER/ASSISTANT "
            "template only when present."
        ),
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="Fixed Qwen ChatML system prompt used for every condition.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. A descriptive default is used when omitted.",
    )
    parser.add_argument(
        "--print-first",
        default=5,
        type=int,
        help="Print detailed diagnostics for the first N samples.",
    )
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_gold(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def is_correct(gold: Any, generation: str) -> bool:
    """Keep the controlled-image scorer identical to the LLaVA ablation."""
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
    """Match model_zoo/llava15.py and the existing LLaVA runner."""
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def sanitize_prompt_for_qwen(prompt: str, mode: str) -> str:
    """
    Preserve the question/options while removing only image placeholders and,
    in auto/llava modes, a surrounding LLaVA conversation wrapper.
    """
    text = str(prompt).replace("\r\n", "\n").strip()

    # Existing controlled prompts may use one of these LLaVA placeholders.
    text = re.sub(r"<image>", "", text, flags=re.IGNORECASE)
    text = text.replace("<im_start>", "").replace("<im_end>", " ")
    text = text.replace("<im_patch>", " ")

    if mode in {"auto", "llava"}:
        # Typical LLaVA template: USER: <image>\nquestion\nASSISTANT:
        user_match = re.search(r"\bUSER\s*:\s*", text, flags=re.IGNORECASE)
        assistant_matches = list(
            re.finditer(r"\bASSISTANT\s*:\s*", text, flags=re.IGNORECASE)
        )
        if user_match is not None:
            start = user_match.end()
            end = assistant_matches[0].start() if assistant_matches else len(text)
            text = text[start:end]
        elif mode == "llava":
            # Alternate older LLaVA wrapper.
            text = re.sub(r"^\s*###\s*(Human|USER)\s*:\s*", "", text,
                          flags=re.IGNORECASE)
            text = re.split(r"\s*###\s*(Assistant|ASSISTANT)\s*:\s*", text,
                            maxsplit=1, flags=re.IGNORECASE)[0]

    # Do not alter the semantic question/options beyond stripping empty lines.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("Qwen query is empty after prompt conversion.")
    return text


def image_to_local_png(image: Any, sid: int, root: Path) -> str:
    """Save a stable, lossless local image path for Qwen-VL's <img> interface."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sid:05d}.png"
    if not path.exists():
        if not isinstance(image, Image.Image):
            # Dataset images are normally PIL. This retains support for a tensor
            # or numpy-like image only when it can be converted by PIL.
            image = Image.fromarray(np.asarray(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(path, format="PNG")
    return str(path.resolve())


def find_language_rmsnorms(model: Any) -> List[Tuple[str, torch.nn.Module]]:
    """
    Qwen-VL remote code defines RMSNorm with an ``eps`` attribute. Restrict the
    search to model.transformer, so visual-encoder layers cannot be modified by
    accident.
    """
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise RuntimeError(
            "Expected a Qwen-VL remote-code model with a .transformer module."
        )

    norms: List[Tuple[str, torch.nn.Module]] = []
    for name, module in transformer.named_modules():
        if type(module).__name__ == "RMSNorm" and hasattr(module, "eps"):
            norms.append((name, module))

    expected_layers = int(getattr(model.config, "num_hidden_layers", 0))
    expected_count = 2 * expected_layers + 1 if expected_layers else None
    if expected_count is not None and len(norms) != expected_count:
        raise RuntimeError(
            "Unexpected Qwen language RMSNorm count: "
            f"found {len(norms)}, expected {expected_count}. "
            "Refusing to continue because the epsilon override may be partial."
        )
    if not norms:
        raise RuntimeError("No Qwen RMSNorm modules with an .eps attribute found.")
    return norms


def set_qwen_rmsnorm_epsilon(model: Any, epsilon: float) -> Tuple[int, List[float]]:
    norms = find_language_rmsnorms(model)
    before = sorted({float(module.eps) for _, module in norms})
    for _, module in norms:
        module.eps = float(epsilon)

    # Metadata only; computation is controlled by each module's .eps above.
    if hasattr(model.config, "layer_norm_epsilon"):
        model.config.layer_norm_epsilon = float(epsilon)
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "config"):
        if hasattr(transformer.config, "layer_norm_epsilon"):
            transformer.config.layer_norm_epsilon = float(epsilon)

    after = sorted({float(module.eps) for _, module in norms})
    print("Qwen language RMSNorm modules:", len(norms))
    print("Qwen RMSNorm epsilon before override:", before)
    print("Qwen RMSNorm epsilon after override:", after)
    if after != [float(epsilon)]:
        raise RuntimeError(
            f"Failed to set every Qwen RMSNorm eps to {epsilon}; got {after}."
        )
    return len(norms), before


def load_qwen_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    print(f"transformers version: {transformers.__version__}")
    print(f"Loading Qwen-VL-Chat from {args.model}@{args.revision}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    precision_kw = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        **{precision_kw: True},
    ).eval()
    model.to(args.device)

    checkpoint_eps = getattr(model.config, "layer_norm_epsilon", None)
    print("Checkpoint config.layer_norm_epsilon:", checkpoint_eps)
    print("Requested RMSNorm epsilon:", float(args.rms_norm_eps))
    rms_count, rms_before = set_qwen_rmsnorm_epsilon(model, args.rms_norm_eps)

    model._qwen_eps_ablation_count = rms_count
    model._qwen_eps_before = rms_before
    model._qwen_checkpoint_eps = checkpoint_eps

    print("Active model classes:")
    print(f"  Qwen-VL: {type(model).__module__}.{type(model).__name__}")
    print(
        "  transformer: "
        f"{type(model.transformer).__module__}.{type(model.transformer).__name__}"
    )
    print(f"  first parameter dtype: {next(model.parameters()).dtype}")
    print(f"  model config commit: {getattr(model.config, '_commit_hash', None)}")
    return model, tokenizer


@torch.inference_mode()
def generate_qwen_response(
    *,
    model: Any,
    tokenizer: Any,
    query: str,
    system: str,
    max_new_tokens: int,
) -> str:
    response, _ = model.chat(
        tokenizer,
        query=query,
        history=None,
        system=system,
        append_history=False,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    return str(response).strip()


@torch.inference_mode()
def evaluate(args: argparse.Namespace, model: Any, tokenizer: Any) -> Dict[str, Any]:
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
        collate_fn=repository_default_collate,
    )

    total_available = min(len(prompts), len(dataset))
    total_target = (
        min(total_available, args.limit)
        if args.limit is not None
        else total_available
    )
    image_cache_root = Path(args.image_cache_dir) / args.dataset

    records: List[Dict[str, Any]] = []
    correct_count = 0
    sample_index = 0
    progress = tqdm(total=total_target, desc="Qwen-VL-Chat eps ablation")

    for batch in loader:
        for image in extract_images_from_batch(batch):
            if sample_index >= total_target:
                break

            raw_prompt = prompts[sample_index]
            query_text = sanitize_prompt_for_qwen(raw_prompt, args.prompt_mode)
            image_path = image_to_local_png(image, sample_index, image_cache_root)
            query = tokenizer.from_list_format(
                [
                    {"image": image_path},
                    {"text": query_text},
                ]
            )

            gold = norm_gold(answers[sample_index])
            generation = generate_qwen_response(
                model=model,
                tokenizer=tokenizer,
                query=query,
                system=args.system,
                max_new_tokens=args.max_new_tokens,
            )
            correct = is_correct(gold, generation)
            correct_count += int(correct)

            record = {
                "sid": sample_index,
                "prompt": raw_prompt,
                "qwen_query_text": query_text,
                "qwen_query": query,
                "image_path": image_path,
                "gold": gold,
                "generation": generation,
                "correct": bool(correct),
            }
            records.append(record)

            if sample_index < args.print_first:
                print("\n" + "-" * 100)
                print(f"[SID {sample_index}] gold={gold!r}")
                print(f"qwen_query_text={query_text!r}")
                print(f"pred={generation!r} correct={correct}")

            sample_index += 1
            progress.update(1)

        if sample_index >= total_target:
            break

    progress.close()
    accuracy = correct_count / max(sample_index, 1)
    summary = {
        "model": args.model,
        "revision": args.revision,
        "resolved_config_commit": getattr(model.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "dataset": args.dataset,
        "option": args.option,
        "implementation": "Qwen-VL-Chat remote code via AutoModelForCausalLM",
        "epsilon_scope": "language transformer RMSNorm modules only",
        "checkpoint_rms_norm_eps": getattr(model, "_qwen_checkpoint_eps", None),
        "requested_rms_norm_eps": float(args.rms_norm_eps),
        "rmsnorm_module_count": getattr(model, "_qwen_eps_ablation_count", None),
        "rmsnorm_eps_before_override": getattr(model, "_qwen_eps_before", None),
        "active_rms_norm_eps": float(args.rms_norm_eps),
        "system": args.system,
        "prompt_mode": args.prompt_mode,
        "dtype_argument": args.dtype,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": False,
        "num_samples": sample_index,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "records": records,
    }

    print("\n" + "=" * 100)
    print(
        f"RESULT: {correct_count}/{sample_index} "
        f"accuracy={accuracy:.6f} "
        f"rms_norm_eps={args.rms_norm_eps:g} "
        "model=Qwen-VL-Chat"
    )
    print("=" * 100)
    return summary


def default_output_path(args: argparse.Namespace) -> Path:
    eps_suffix = f"{args.rms_norm_eps:.0e}".replace("-", "m")
    return Path(
        "output/"
        f"qwenvl_chat_{args.dataset}_"
        f"rms_eps_{eps_suffix}.json"
    )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    model, tokenizer = load_qwen_model(args)
    summary = evaluate(args, model, tokenizer)

    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
