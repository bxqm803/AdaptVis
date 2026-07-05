#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone LLaVA-1.5 VSR epsilon x ScalingVis runner.

Validated target environment
----------------------------
  torch==2.4.1+cu121
  torchvision==0.19.1+cu121
  transformers==4.49.0

This file deliberately DOES NOT import the repository's old
run_llava15_hf_rmsnorm_eps_ablation_original.py. That script targets the
pre-4.46 LLaVA merge API and older LlamaAttention attributes.

What is changed
---------------
1. Every language-decoder LlamaRMSNorm.variance_epsilon is set to
   --rms-norm-eps.
2. During the initial multimodal prefill only, in layers
   [0, --max-layers), ScalingVis multiplies the pre-softmax attention logits
   from the final prompt query to all <image> token keys by
   --scalvis-weight.

For a fixed ScalingVis=0.5 epsilon sweep:
  --rms-norm-eps 1e-4
  --rms-norm-eps 1e-5
  --rms-norm-eps 1e-6
  --rms-norm-eps 1e-7

The runner writes one JSONL record after every sample and supports --resume.

Examples
--------
# One-sample validation. Must report modified_calls=4.
CUDA_VISIBLE_DEVICES=0 python run_llava15_vsr_scalvis_eps.py \
  --rms-norm-eps 1e-6 --scalvis-weight 0.5 --max-layers 4 --limit 1

# Full VSR, one epsilon.
CUDA_VISIBLE_DEVICES=0 python run_llava15_vsr_scalvis_eps.py \
  --rms-norm-eps 1e-6 --scalvis-weight 0.5 --max-layers 4 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaConfig, LlavaForConditionalGeneration
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "a272c74"
DEFAULT_VSR_ANN = "data/benchmarks/vsr/repo/data/splits/zeroshot/test.jsonl"
DEFAULT_VSR_IMAGE_ROOT = "data/benchmarks/vsr/images"


@dataclass
class GenerationDiagnostics:
    requested_weight: float
    modified_calls: int
    image_token_count: int
    prefill_length: int
    image_start: Optional[int]
    image_end: Optional[int]


class ScalingVisController:
    """Per-generation state consumed by the patched LlamaAttention modules."""

    def __init__(self, max_layers: int) -> None:
        self.max_layers = int(max_layers)
        self.active = False
        self.weight = 1.0
        self.image_mask: Optional[torch.Tensor] = None
        self.prefill_length = 0
        self.modified_calls = 0

    def begin(
        self,
        input_ids: torch.LongTensor,
        image_token_id: int,
        weight: float,
    ) -> None:
        if input_ids.ndim != 2:
            raise RuntimeError(f"Expected input_ids [B,L], got {tuple(input_ids.shape)}.")

        image_mask = input_ids.eq(int(image_token_id))
        if not bool(image_mask.any().item()):
            raise RuntimeError(
                "No LLaVA image placeholder tokens found in input_ids. "
                "The processor did not expand <image> correctly."
            )

        self.active = float(weight) != 1.0
        self.weight = float(weight)
        self.image_mask = image_mask.detach()
        self.prefill_length = int(input_ids.shape[-1])
        self.modified_calls = 0

    def should_apply(self, layer_idx: int, q_len: int, kv_len: int, bsz: int) -> bool:
        mask = self.image_mask
        return bool(
            self.active
            and self.weight != 1.0
            and int(layer_idx) < self.max_layers
            and mask is not None
            and q_len == self.prefill_length
            and kv_len == self.prefill_length
            and q_len > 1
            and mask.ndim == 2
            and mask.shape[0] in (1, bsz)
            and mask.shape[1] == kv_len
        )

    def apply(self, raw_logits: torch.Tensor, layer_idx: int) -> None:
        """Scale only the last prefill query against visual-token keys, in-place."""
        bsz, _, q_len, kv_len = raw_logits.shape
        if not self.should_apply(layer_idx, q_len, kv_len, bsz):
            return

        assert self.image_mask is not None
        image_mask = self.image_mask.to(device=raw_logits.device, dtype=torch.bool)
        if image_mask.shape[0] == 1 and bsz > 1:
            image_mask = image_mask.expand(bsz, -1)

        for batch_idx in range(bsz):
            key_mask = image_mask[batch_idx]
            if not bool(key_mask.any().item()):
                raise RuntimeError(f"Empty visual key mask for batch item {batch_idx}.")
            raw_logits[batch_idx, :, q_len - 1, key_mask] *= self.weight

        self.modified_calls += 1

    def finish(self) -> GenerationDiagnostics:
        image_count = 0
        image_start: Optional[int] = None
        image_end: Optional[int] = None

        if self.image_mask is not None and self.image_mask.numel() > 0:
            indices = torch.nonzero(
                self.image_mask[0].detach().to(dtype=torch.bool).cpu(),
                as_tuple=False,
            ).flatten()
            image_count = int(indices.numel())
            if image_count:
                image_start = int(indices.min())
                image_end = int(indices.max()) + 1

        result = GenerationDiagnostics(
            requested_weight=float(self.weight),
            modified_calls=int(self.modified_calls),
            image_token_count=image_count,
            prefill_length=int(self.prefill_length),
            image_start=image_start,
            image_end=image_end,
        )

        self.active = False
        self.weight = 1.0
        self.image_mask = None
        self.prefill_length = 0
        self.modified_calls = 0
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaVA-1.5 VSR epsilon + fixed ScalingVis evaluator, compatible with transformers 4.49."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument("--rms-norm-eps", required=True, type=float)
    parser.add_argument("--scalvis-weight", default=0.5, type=float)
    parser.add_argument("--max-layers", default=4, type=int)
    parser.add_argument("--max-new-tokens", default=4, type=int)

    parser.add_argument("--vsr-ann", default=DEFAULT_VSR_ANN)
    parser.add_argument("--vsr-image-root", default=DEFAULT_VSR_IMAGE_ROOT)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-first", default=3, type=int)
    parser.add_argument("--output-dir", default="output/llava15_vsr_scalvis_eps")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_vsr_gold(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if int(value) else "no"

    text = normalize_space(value).lower()
    aliases = {
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "correct": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
        "incorrect": "no",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(f"Unsupported VSR label: {value!r}")


def parse_yes_no(generation: str) -> str:
    match = re.search(r"\b(yes|no)\b", normalize_space(generation).lower())
    return match.group(1) if match else ""


def make_vsr_prompt(caption: str) -> str:
    return (
        "USER: <image>\n"
        "Does the statement accurately describe the image? "
        "Reply with exactly one word: yes or no. Do not explain.\n"
        f"Statement: {caption}\n"
        "ASSISTANT:"
    )


def build_image_index(image_root: Path) -> Dict[str, str]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"VSR image root not found: {image_root}")

    index: Dict[str, str] = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for image_path in image_root.rglob(suffix):
            index.setdefault(image_path.name, str(image_path))
    return index


def load_vsr(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ann_path = Path(args.vsr_ann)
    if not ann_path.is_file():
        raise FileNotFoundError(f"VSR annotation JSONL not found: {ann_path}")

    image_root = Path(args.vsr_image_root)
    image_index = build_image_index(image_root)

    samples: List[Dict[str, Any]] = []
    with ann_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue

            row = json.loads(line)
            required = {"caption", "label", "image_link"}
            missing = required - set(row)
            if missing:
                raise KeyError(f"VSR sample {index} missing keys: {sorted(missing)}")

            image_name = Path(str(row["image_link"]).split("?", 1)[0]).name
            image_path = image_index.get(image_name, str(image_root / image_name))
            caption = normalize_space(row["caption"])
            samples.append(
                {
                    "sid": str(index),
                    "caption": caption,
                    "gold": normalize_vsr_gold(row["label"]),
                    "relation": row.get("relation"),
                    "image_link": row["image_link"],
                    "image_path": image_path,
                    "prompt": make_vsr_prompt(caption),
                }
            )

    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    samples = samples[args.start :]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        samples = samples[: args.limit]

    if not samples:
        raise RuntimeError("No VSR samples selected.")

    return samples


def set_rmsnorm_epsilon(model: LlavaForConditionalGeneration, epsilon: float) -> Dict[str, Any]:
    if epsilon <= 0:
        raise ValueError("--rms-norm-eps must be positive.")

    norms = [
        module
        for module in model.language_model.modules()
        if isinstance(module, LlamaRMSNorm)
    ]
    if not norms:
        raise RuntimeError("No language-decoder LlamaRMSNorm modules found.")

    before = sorted({float(module.variance_epsilon) for module in norms})
    for module in norms:
        module.variance_epsilon = float(epsilon)

    model.language_model.config.rms_norm_eps = float(epsilon)
    if hasattr(model.config, "text_config"):
        model.config.text_config.rms_norm_eps = float(epsilon)

    after = sorted({float(module.variance_epsilon) for module in norms})
    if after != [float(epsilon)]:
        raise RuntimeError(f"RMSNorm eps override failed: {after}")

    return {
        "module_count": len(norms),
        "before": before,
        "after": after,
    }


def install_scalvis_attention(
    model: LlavaForConditionalGeneration,
    controller: ScalingVisController,
) -> None:
    """
    transformers==4.49 LlamaAttention patch.

    The model is forced to use eager attention. The non-intervened path always
    calls the checkpoint-native forward. The custom path is entered ONLY for
    the first multimodal prefill and only when --scalvis-weight != 1.0.
    """
    layers = model.language_model.model.layers
    if controller.max_layers <= 0 or controller.max_layers > len(layers):
        raise ValueError(
            f"--max-layers must be in [1, {len(layers)}], got {controller.max_layers}."
        )

    for layer_idx, layer in enumerate(layers):
        attention = layer.self_attn
        original_forward = attention.forward

        def make_forward(attn: Any, original: Any, idx: int):
            def patched_forward(
                self: Any,
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[Any] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs: Any,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                bsz, q_len, _ = hidden_states.shape

                # Initial prefill has kv_len=q_len. All later cached decode steps
                # have q_len=1 and are passed to the original implementation.
                if not controller.should_apply(
                    idx,
                    q_len=q_len,
                    kv_len=q_len,
                    bsz=bsz,
                ):
                    return original(
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_value=past_key_value,
                        cache_position=cache_position,
                        **kwargs,
                    )

                config = attn.config
                num_heads = int(config.num_attention_heads)
                num_kv_heads = int(config.num_key_value_heads)
                num_kv_groups = int(attn.num_key_value_groups)
                head_dim = int(attn.head_dim)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, head_dim)

                query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                if query_states.shape[1] != num_heads:
                    raise RuntimeError("Unexpected Q-head count in patched LlamaAttention.")
                if key_states.shape[1] != num_kv_heads:
                    raise RuntimeError("Unexpected KV-head count in patched LlamaAttention.")

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                )

                # DynamicCache is instantiated by LlamaModel before layer 0.
                if past_key_value is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key_states, value_states = past_key_value.update(
                        key_states,
                        value_states,
                        attn.layer_idx,
                        cache_kwargs,
                    )

                key_states = repeat_kv(key_states, num_kv_groups)
                value_states = repeat_kv(value_states, num_kv_groups)
                kv_len = int(key_states.shape[-2])

                if not controller.should_apply(idx, q_len, kv_len, bsz):
                    raise RuntimeError(
                        "ScalingVis was selected for prefill but KV length does not "
                        f"match the input image mask: q={q_len}, kv={kv_len}, "
                        f"prefill={controller.prefill_length}, layer={idx}."
                    )

                raw_logits = torch.matmul(
                    query_states,
                    key_states.transpose(2, 3),
                ) * float(attn.scaling)

                # The intervention: pre-softmax, final prompt query -> visual keys.
                controller.apply(raw_logits, idx)

                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, :kv_len]
                    raw_logits = raw_logits + causal_mask

                attn_probs = F.softmax(
                    raw_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query_states.dtype)
                attn_probs = F.dropout(
                    attn_probs,
                    p=0.0 if not attn.training else float(attn.attention_dropout),
                    training=attn.training,
                )

                attn_output = torch.matmul(attn_probs, value_states)
                expected = (bsz, num_heads, q_len, head_dim)
                if tuple(attn_output.shape) != expected:
                    raise RuntimeError(
                        f"Unexpected attention output {tuple(attn_output.shape)}; "
                        f"expected {expected}."
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1)
                attn_output = attn.o_proj(attn_output)

                # LlamaAttention in transformers 4.49 returns exactly two values.
                return attn_output, None

            return patched_forward

        attention.forward = types.MethodType(make_forward(attention, original_forward, layer_idx), attention)


def load_model_and_processor(
    args: argparse.Namespace,
) -> Tuple[LlavaForConditionalGeneration, Any, ScalingVisController, Dict[str, Any]]:
    dtype = resolve_dtype(args.dtype)

    config = LlavaConfig.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    checkpoint_eps = float(config.text_config.rms_norm_eps)

    # Explicitly force the eager attention implementation used by the patch.
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"

    load_kwargs: Dict[str, Any] = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "config": config,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if args.dtype != "float32":
        load_kwargs["torch_dtype"] = dtype

    model = LlavaForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    model.eval().to(args.device)

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    # Compatibility for old LLaVA-1.5 processor config under transformers>=4.46.
    # LLaVA default removes CLS from CLIP features, but processor needs to count
    # it before subtracting: 576 patches + 1 CLS - 1 = 576 <image> tokens.
    processor.patch_size = int(model.config.vision_config.patch_size)
    processor.vision_feature_select_strategy = str(
        model.config.vision_feature_select_strategy
    )
    processor.num_additional_image_tokens = 1

    controller = ScalingVisController(max_layers=args.max_layers)
    install_scalvis_attention(model, controller)
    eps_info = set_rmsnorm_epsilon(model, args.rms_norm_eps)

    print("Model:", args.model)
    print("Revision:", args.revision)
    print("Attention backend:", getattr(model.config, "_attn_implementation", None))
    print("Parameter dtype:", next(model.parameters()).dtype)
    print(
        "RMSNorm eps:",
        f"{eps_info['before']} -> {eps_info['after']}",
        f"({eps_info['module_count']} modules)",
    )
    print(f"ScalingVis: weight={args.scalvis_weight}, layers=[0,{args.max_layers})")
    return model, processor, controller, {
        "checkpoint_rms_norm_eps": checkpoint_eps,
        **eps_info,
    }


def prepare_inputs(
    processor: Any,
    image: Image.Image,
    prompt: str,
    device: str,
    model_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    batch = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    result: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            result[key] = value
        elif key == "pixel_values":
            result[key] = value.to(device=device, dtype=model_dtype)
        else:
            result[key] = value.to(device=device)
    return result


@torch.inference_mode()
def generate_one(
    model: LlavaForConditionalGeneration,
    processor: Any,
    controller: ScalingVisController,
    sample: Dict[str, Any],
    weight: float,
    max_new_tokens: int,
    device: str,
) -> Tuple[str, GenerationDiagnostics]:
    image = Image.open(sample["image_path"]).convert("RGB")
    inputs = prepare_inputs(
        processor=processor,
        image=image,
        prompt=sample["prompt"],
        device=device,
        model_dtype=next(model.parameters()).dtype,
    )

    input_ids = inputs["input_ids"]
    image_token_id = int(model.config.image_token_index)
    controller.begin(input_ids, image_token_id, weight)

    output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=int(max_new_tokens),
    )
    diagnostics = controller.finish()

    if float(weight) != 1.0 and diagnostics.modified_calls != controller.max_layers:
        raise RuntimeError(
            "ScalingVis validation failed: expected "
            f"modified_calls={controller.max_layers}, got "
            f"{diagnostics.modified_calls}."
        )

    continuation = output[:, input_ids.shape[1] :]
    generation = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return generation, diagnostics


def eps_tag(value: float) -> str:
    return f"{float(value):.0e}".replace("-", "m")


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    weight_tag = str(args.scalvis_weight).replace(".", "p")
    tag = (
        f"llava15_vsr__eps{eps_tag(args.rms_norm_eps)}"
        f"__w{weight_tag}__l{args.max_layers}"
    )
    return root / f"{tag}.jsonl", root / f"{tag}.summary.json"


def read_existing_records(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return result

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["sid"])] = row
    return result


def per_relation(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("relation") or "unknown")
        groups.setdefault(key, []).append(record)

    result: Dict[str, Dict[str, Any]] = {}
    for relation, rows in sorted(groups.items()):
        num_correct = sum(bool(row.get("correct")) for row in rows)
        result[relation] = {
            "num_samples": len(rows),
            "num_correct": num_correct,
            "accuracy": num_correct / max(len(rows), 1),
        }
    return result


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive.")

    samples = load_vsr(args)
    jsonl_path, summary_path = output_paths(args)
    existing = read_existing_records(jsonl_path) if args.resume else {}

    model, processor, controller, eps_info = load_model_and_processor(args)

    num_correct = sum(bool(row.get("correct")) for row in existing.values())
    num_done = len(existing)
    skipped_missing = 0

    mode = "a" if args.resume else "w"
    with jsonl_path.open(mode, encoding="utf-8") as output_file:
        progress = tqdm(
            samples,
            desc=(
                f"LLaVA-1.5 VSR eps={args.rms_norm_eps:g} "
                f"w={args.scalvis_weight:g}"
            ),
        )

        for display_index, sample in enumerate(progress):
            if sample["sid"] in existing:
                continue

            if not Path(sample["image_path"]).is_file():
                message = f"Missing image for sid={sample['sid']}: {sample['image_path']}"
                if args.skip_missing_images:
                    print("WARNING:", message)
                    skipped_missing += 1
                    continue
                raise FileNotFoundError(message)

            prediction, diagnostics = generate_one(
                model=model,
                processor=processor,
                controller=controller,
                sample=sample,
                weight=args.scalvis_weight,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            prediction_normalized = parse_yes_no(prediction)
            correct = prediction_normalized == sample["gold"]

            record = {
                "sid": sample["sid"],
                "dataset": "vsr",
                "image_path": sample["image_path"],
                "image_link": sample["image_link"],
                "caption": sample["caption"],
                "relation": sample["relation"],
                "prompt": sample["prompt"],
                "gold": sample["gold"],
                "prediction": prediction,
                "prediction_normalized": prediction_normalized,
                "correct": bool(correct),
                "rms_norm_eps": float(args.rms_norm_eps),
                "scalvis_weight": float(args.scalvis_weight),
                "max_layers": int(args.max_layers),
                "generation": {
                    "requested_weight": diagnostics.requested_weight,
                    "modified_calls": diagnostics.modified_calls,
                    "image_token_count": diagnostics.image_token_count,
                    "prefill_length": diagnostics.prefill_length,
                    "image_start": diagnostics.image_start,
                    "image_end": diagnostics.image_end,
                },
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())

            num_correct += int(correct)
            num_done += 1

            if display_index < args.print_first:
                print("-" * 96)
                print(
                    f"[{sample['sid']}] gold={sample['gold']!r} "
                    f"pred={prediction!r} normalized={prediction_normalized!r} "
                    f"correct={correct}"
                )
                print("generation:", record["generation"])

            progress.set_postfix(acc=f"{num_correct / max(num_done, 1):.4f}")

    records = list(read_existing_records(jsonl_path).values())
    total_correct = sum(bool(row.get("correct")) for row in records)
    summary = {
        "model": args.model,
        "revision": args.revision,
        "dataset": "vsr",
        "vsr_ann": str(args.vsr_ann),
        "vsr_image_root": str(args.vsr_image_root),
        "rms_norm_eps": float(args.rms_norm_eps),
        "scalvis_weight": float(args.scalvis_weight),
        "max_layers": int(args.max_layers),
        "max_new_tokens": int(args.max_new_tokens),
        "intervention": (
            "Initial multimodal prefill only; pre-softmax attention logits from "
            "the final prompt query to all LLaVA image placeholder keys are "
            "multiplied in selected LLaMA decoder layers."
        ),
        "checkpoint_rms_norm_eps": eps_info["checkpoint_rms_norm_eps"],
        "rmsnorm_module_count": eps_info["module_count"],
        "rmsnorm_eps_before_override": eps_info["before"],
        "rmsnorm_eps_after_override": eps_info["after"],
        "num_selected": len(samples),
        "num_evaluated": len(records),
        "num_correct": total_correct,
        "accuracy": total_correct / max(len(records), 1),
        "skipped_missing_images_this_run": skipped_missing,
        "per_relation": per_relation(records),
        "records_jsonl": str(jsonl_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 96)
    print(
        f"RESULT: eps={args.rms_norm_eps:g}, w={args.scalvis_weight:g}, "
        f"{summary['num_correct']}/{summary['num_evaluated']} = "
        f"{summary['accuracy']:.6f}"
    )
    print("Records:", jsonl_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
