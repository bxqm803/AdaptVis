#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaliGemma-3B-mix-224 fixed-weight AdaptVis-style image-attention + RMSNorm-epsilon ablation.

Purpose
-------
This runner is a structural control for LLaVA-1.5 AdaptVis. PaliGemma uses:

    SigLIP image encoder -> linear multimodal projector -> Gemma decoder tokens.

For a single image/question prefill, it multiplies the *raw, pre-mask,
pre-softmax* attention logits from the final prompt position to every PaliGemma
image placeholder token. The intervention is applied to each Gemma decoder
layer in [0, --max-layers), only during the initial image+text prefill.
Generation/cache steps are left native.

This file is deliberately pinned to Transformers 4.41.2. Its manual Gemma
attention forward is reproduced exactly for the intervention branch; use
`attn_implementation="eager"` and do not use a newer Transformers release
without re-validating the attention implementation.

Recommended experiment
----------------------
Hold the raw image-attention multiplier fixed (for example, w=0.5) and sweep
Gemma decoder RMSNorm epsilon. This isolates whether the LLaVA epsilon
phenomenon transfers to a non-LLaVA projector-token architecture without
introducing an adaptive gate.

Run from the AdaptVis repository root, where dataset_zoo.py, misc.py, and
prompts/ are available.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm, apply_rotary_pos_emb, repeat_kv

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


PINNED_TRANSFORMERS_VERSION = "4.41.2"
DEFAULT_MODEL = "google/paligemma-3b-mix-224"


@dataclass
class GenerationDiagnostics:
    requested_weight: float
    modified_calls: int
    expected_modified_calls: int
    image_token_count: int
    prompt_sequence_length: int
    image_start: Optional[int]
    image_end: Optional[int]


class PaliGemmaAdaptVisController:
    """Per-generation state read by patched eager Gemma attention modules."""

    def __init__(self, max_layers: int, num_decoder_layers: int) -> None:
        self.max_layers = int(max_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.enabled = False
        self.weight = 1.0
        self.image_mask: Optional[torch.Tensor] = None
        self.prompt_sequence_length = 0
        self.modified_calls = 0

    def begin_generation(self, input_ids: torch.LongTensor, image_token_index: int, weight: float) -> None:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected [B,L] input_ids, got {tuple(input_ids.shape)}")
        image_mask = input_ids.eq(int(image_token_index))
        if not bool(image_mask.any().item()):
            raise RuntimeError(
                "No PaliGemma image placeholder tokens found in input_ids. "
                "The processor input does not contain visual-token positions."
            )
        self.weight = float(weight)
        self.enabled = self.weight != 1.0
        self.image_mask = image_mask
        self.prompt_sequence_length = int(input_ids.shape[-1])
        self.modified_calls = 0

    def finish_generation(self) -> GenerationDiagnostics:
        count = 0
        image_start: Optional[int] = None
        image_end: Optional[int] = None
        if self.image_mask is not None and self.image_mask.numel() > 0:
            positions = torch.nonzero(self.image_mask[0].detach().cpu(), as_tuple=False).flatten()
            count = int(positions.numel())
            if count:
                image_start = int(positions.min())
                image_end = int(positions.max()) + 1

        result = GenerationDiagnostics(
            requested_weight=float(self.weight),
            modified_calls=int(self.modified_calls),
            expected_modified_calls=min(self.max_layers, self.num_decoder_layers) if self.enabled else 0,
            image_token_count=count,
            prompt_sequence_length=int(self.prompt_sequence_length),
            image_start=image_start,
            image_end=image_end,
        )
        self.enabled = False
        self.image_mask = None
        self.prompt_sequence_length = 0
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PaliGemma fixed-weight pre-softmax image-attention and RMSNorm-epsilon ablation."
    )
    parser.add_argument(
        "--dataset",
        default="Controlled_Images_A",
        choices=["Controlled_Images_A", "Controlled_Images_B"],
    )
    parser.add_argument("--option", default="four", choices=["two", "four", "six"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        default="auto",
        help="auto -> bfloat16/float16/main based on --dtype; otherwise a Hugging Face revision.",
    )
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--rms-norm-eps",
        default=None,
        type=float,
        help=(
            "Override every GemmaRMSNorm.eps in the language decoder. "
            "Omit to preserve the checkpoint value."
        ),
    )
    parser.add_argument("--method", default="scaling_vis", choices=["base", "scaling_vis"])
    parser.add_argument(
        "--weight",
        default=0.5,
        type=float,
        help="Raw image-logit multiplier used by scaling_vis. Use 1.0 for exact no-intervention control.",
    )
    parser.add_argument(
        "--max-layers",
        default=32,
        type=int,
        help="Apply to decoder layers [0,max_layers); capped at PaliGemma's number of Gemma layers.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="answer ",
        help="PaliGemma VQA-style task prefix prepended to each stripped Controlled question.",
    )
    parser.add_argument("--max-new-tokens", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--print-first", default=5, type=int)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_revision(args: argparse.Namespace) -> str:
    if args.revision != "auto":
        return str(args.revision)
    return {
        "float32": "main",
        "float16": "float16",
        "bfloat16": "bfloat16",
    }[args.dtype]


def require_supported_transformers() -> None:
    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"This script is pinned to transformers=={PINNED_TRANSFORMERS_VERSION}, "
            f"but found {transformers.__version__}. Install the pinned version in a separate environment "
            "or revalidate the copied GemmaAttention forward before running."
        )


def _unique_eps(modules: Iterable[nn.Module]) -> List[float]:
    return sorted({float(module.eps) for module in modules if hasattr(module, "eps")})


def configure_language_rms_norm_eps(
    model: PaliGemmaForConditionalGeneration,
    requested_eps: Optional[float],
) -> Tuple[int, List[float], List[float]]:
    """Optionally override all Gemma RMSNorm modules in the language decoder only."""
    language_norms = [
        module
        for module in model.language_model.modules()
        if isinstance(module, GemmaRMSNorm)
    ]
    if not language_norms:
        raise RuntimeError("No GemmaRMSNorm modules found in model.language_model.")

    before = _unique_eps(language_norms)
    if requested_eps is not None:
        if requested_eps <= 0.0:
            raise ValueError(f"--rms-norm-eps must be positive, got {requested_eps}.")
        for module in language_norms:
            module.eps = float(requested_eps)
    after = _unique_eps(language_norms)
    return len(language_norms), before, after


def norm_gold(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def normalize_relation_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_correct(gold: Any, generation: str) -> bool:
    """Keep the repository's existing substring-style metric for direct comparison."""
    gold_text = norm_gold(gold)
    prediction = normalize_relation_text(generation)
    if not gold_text:
        return False
    result = gold_text.lower() in prediction
    if gold_text.lower() == "on" and "front" in prediction:
        result = False
    return bool(result)


def extract_relation_mentions(generation: str) -> List[str]:
    text = normalize_relation_text(generation)
    return re.findall(r"\b(left|right|under|on)\b", text)


def load_prompts(dataset: str, option: str) -> Tuple[List[str], List[Any]]:
    path = Path(f"prompts/{dataset}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}. Run from the AdaptVis repository root.")
    prompts: List[str] = []
    answers: List[Any] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            prompts.append(item["question"])
            answers.append(item["answer"])
    return prompts, answers


def sanitize_prompt_for_paligemma(prompt: str, task_prefix: str) -> str:
    """Remove LLaVA conversation markers, retaining only the image question."""
    text = str(prompt).replace("\r\n", "\n").strip()
    text = re.sub(r"<image>", "", text, flags=re.IGNORECASE)
    text = text.replace("<im_start>", "").replace("<im_end>", " ").replace("<im_patch>", " ")

    user_match = re.search(r"\bUSER\s*:\s*", text, flags=re.IGNORECASE)
    assistant_matches = list(re.finditer(r"\bASSISTANT\s*:\s*", text, flags=re.IGNORECASE))
    if user_match is not None:
        start = user_match.end()
        end = assistant_matches[0].start() if assistant_matches else len(text)
        text = text[start:end]
    else:
        text = re.sub(r"^\s*###\s*(Human|USER)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.split(r"\s*###\s*(Assistant|ASSISTANT)\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]

    text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not text:
        raise ValueError("PaliGemma query is empty after prompt conversion.")
    return f"{task_prefix}{text}" if task_prefix else text


def extract_images_from_batch(batch: Dict[str, Any]) -> Iterable[Any]:
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def ensure_rgb(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return image.convert("RGB")


def move_inputs_to_model(inputs: Dict[str, torch.Tensor], device: torch.device, model_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def install_eager_gemma_adaptvis(
    model: PaliGemmaForConditionalGeneration,
    controller: PaliGemmaAdaptVisController,
) -> None:
    """
    Patch only the intervention path of eager GemmaAttention.

    Non-intervention calls delegate to the original module forward, preserving
    native generation/cache behavior. The copied branch follows Transformers
    4.41.2 GemmaAttention.forward exactly, aside from one multiplication:

        attn_weights[:, :, final_prompt_query, image_token_keys] *= weight

    This is pre-mask and pre-softmax, matching the AdaptVis placement.
    """
    decoder = model.language_model.model
    layers = decoder.layers

    for layer_index, layer in enumerate(layers):
        attn = layer.self_attn
        original_forward = attn.forward
        required = ("q_proj", "k_proj", "v_proj", "o_proj", "rotary_emb")
        missing = [name for name in required if not hasattr(attn, name)]
        if missing:
            raise RuntimeError(
                f"Layer {layer_index} is not eager Gemma attention; missing {missing}. "
                "Load the model with attn_implementation='eager'."
            )

        def make_forward(attn_module: nn.Module, original: Any, idx: int):
            def forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs: Any,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
                bsz, q_len, _ = hidden_states.size()
                image_mask = controller.image_mask
                should_intervene = (
                    controller.enabled
                    and controller.weight != 1.0
                    and idx < controller.max_layers
                    and image_mask is not None
                    and image_mask.ndim == 2
                    and image_mask.shape[0] in (1, bsz)
                    and image_mask.shape[-1] == q_len
                    and controller.prompt_sequence_length == q_len
                    and q_len > 1
                )

                if not should_intervene:
                    return original(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )

                # Exact eager GemmaAttention path from Transformers 4.41.2.
                query_states = attn_module.q_proj(hidden_states)
                key_states = attn_module.k_proj(hidden_states)
                value_states = attn_module.v_proj(hidden_states)

                query_states = query_states.view(
                    bsz, q_len, attn_module.num_heads, attn_module.head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim
                ).transpose(1, 2)

                cos, sin = attn_module.rotary_emb(value_states, position_ids, seq_len=None)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states,
                        value_states,
                        attn_module.layer_idx,
                        cache_kwargs,
                    )

                key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
                value_states = repeat_kv(value_states, attn_module.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_module.head_dim)

                if key_states.shape[-2] != q_len:
                    raise RuntimeError(
                        "AdaptVis prefill expected key length to equal the complete prompt length, "
                        f"got keys={key_states.shape[-2]} and prompt={q_len}."
                    )

                local_image_mask = image_mask
                if local_image_mask.shape[0] == 1 and bsz > 1:
                    local_image_mask = local_image_mask.expand(bsz, -1)
                local_image_mask = local_image_mask.to(device=attn_weights.device, dtype=torch.bool)

                # Raw QK logits, final prompt position only; before mask and softmax.
                for batch_index in range(bsz):
                    attn_weights[batch_index, :, q_len - 1, local_image_mask[batch_index]] *= controller.weight
                controller.modified_calls += 1

                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask

                attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_probs = nn.functional.dropout(
                    attn_probs,
                    p=attn_module.attention_dropout,
                    training=attn_module.training,
                )
                attn_output = torch.matmul(attn_probs, value_states)

                expected_shape = (bsz, attn_module.num_heads, q_len, attn_module.head_dim)
                if attn_output.size() != expected_shape:
                    raise ValueError(
                        f"attn_output should be {expected_shape}, got {tuple(attn_output.size())}."
                    )

                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                attn_output = attn_module.o_proj(attn_output)
                returned_weights = attn_probs if output_attentions else None
                return attn_output, returned_weights, past_key_value

            return forward

        attn.forward = make_forward(attn, original_forward, layer_index)


def load_model_and_processor(
    args: argparse.Namespace,
) -> Tuple[PaliGemmaForConditionalGeneration, Any, PaliGemmaAdaptVisController, str]:
    require_supported_transformers()
    dtype = resolve_dtype(args.dtype)
    revision = resolve_revision(args)
    print(f"transformers version: {transformers.__version__}")
    print(f"Loading {args.model}@{revision} with eager Gemma attention")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model,
        revision=revision,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).eval()
    model = model.to(args.device)
    num_rms_norms, rms_before, rms_after = configure_language_rms_norm_eps(
        model,
        args.rms_norm_eps,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=revision,
        cache_dir=args.cache_dir,
    )

    decoder_layers = model.language_model.model.layers
    controller = PaliGemmaAdaptVisController(args.max_layers, len(decoder_layers))
    install_eager_gemma_adaptvis(model, controller)

    observed = {layer.self_attn.__class__.__name__ for layer in decoder_layers}
    print(f"PaliGemma decoder layers: {len(decoder_layers)}")
    print(f"Gemma language RMSNorm modules: {num_rms_norms}")
    print(f"Gemma RMSNorm eps before override: {rms_before}")
    print(f"Gemma RMSNorm eps after override: {rms_after}")
    # Store diagnostics on the model for JSON reporting; this does not affect forward execution.
    model._adaptvis_rms_norm_count = int(num_rms_norms)
    model._adaptvis_rms_norm_eps_before = list(rms_before)
    model._adaptvis_rms_norm_eps_after = list(rms_after)
    print(f"Attention classes: {sorted(observed)}")
    print(f"model dtype: {next(model.parameters()).dtype}")
    print(f"image_token_index: {model.config.image_token_index}")
    return model, processor, controller, revision


@torch.inference_mode()
def generate_once(
    *,
    model: PaliGemmaForConditionalGeneration,
    processor: Any,
    controller: PaliGemmaAdaptVisController,
    image: Image.Image,
    prompt: str,
    weight: float,
    max_new_tokens: int,
) -> Tuple[str, GenerationDiagnostics]:
    prepared = processor(text=prompt, images=image, return_tensors="pt")
    prepared = move_inputs_to_model(prepared, model.device, next(model.parameters()).dtype)
    input_ids = prepared["input_ids"]

    controller.begin_generation(
        input_ids=input_ids,
        image_token_index=int(model.config.image_token_index),
        weight=weight,
    )
    generated_ids = model.generate(
        **prepared,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    diagnostics = controller.finish_generation()

    continuation_ids = generated_ids[:, input_ids.shape[-1] :]
    generation = processor.batch_decode(continuation_ids, skip_special_tokens=True)[0]
    return str(generation).strip(), diagnostics


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: PaliGemmaForConditionalGeneration,
    processor: Any,
    controller: PaliGemmaAdaptVisController,
    revision: str,
) -> Dict[str, Any]:
    prompts, answers = load_prompts(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=repository_default_collate,
    )

    total_available = min(len(prompts), len(dataset))
    total_target = min(total_available, args.limit) if args.limit is not None else total_available
    requested_weight = 1.0 if args.method == "base" else float(args.weight)

    records: List[Dict[str, Any]] = []
    correct_count = 0
    sid = 0
    progress = tqdm(total=total_target, desc=f"PaliGemma {args.method}")

    for batch in loader:
        for raw_image in extract_images_from_batch(batch):
            if sid >= total_target:
                break
            raw_prompt = prompts[sid]
            prompt = sanitize_prompt_for_paligemma(raw_prompt, args.prompt_prefix)
            image = ensure_rgb(raw_image)
            gold = norm_gold(answers[sid])

            generation, diagnostics = generate_once(
                model=model,
                processor=processor,
                controller=controller,
                image=image,
                prompt=prompt,
                weight=requested_weight,
                max_new_tokens=args.max_new_tokens,
            )
            correct = is_correct(gold, generation)
            correct_count += int(correct)

            if requested_weight != 1.0 and diagnostics.modified_calls != diagnostics.expected_modified_calls:
                raise RuntimeError(
                    "AdaptVis did not modify every expected prefill layer: "
                    f"modified={diagnostics.modified_calls}, expected={diagnostics.expected_modified_calls}."
                )
            if diagnostics.image_token_count != 256:
                raise RuntimeError(
                    "Unexpected PaliGemma-224 image-token count: "
                    f"expected 256, got {diagnostics.image_token_count}."
                )

            record = {
                "sid": sid,
                "prompt": raw_prompt,
                "paligemma_prompt": prompt,
                "gold": gold,
                "method": args.method,
                "selected_weight": requested_weight,
                "generation": generation,
                "relation_mentions": extract_relation_mentions(generation),
                "correct": bool(correct),
                "diagnostics": asdict(diagnostics),
            }
            records.append(record)

            if sid < args.print_first:
                print("\n" + "-" * 100)
                print(f"[SID {sid}] gold={gold!r}")
                print(f"paligemma_prompt={prompt!r}")
                print(f"weight={requested_weight} pred={generation!r} correct={correct}")
                print(
                    "image tokens="
                    f"{diagnostics.image_token_count}, range=[{diagnostics.image_start},{diagnostics.image_end}), "
                    f"prompt_len={diagnostics.prompt_sequence_length}, "
                    f"modified_calls={diagnostics.modified_calls}/{diagnostics.expected_modified_calls}"
                )

            sid += 1
            progress.update(1)
        if sid >= total_target:
            break

    progress.close()
    accuracy = correct_count / max(sid, 1)
    summary = {
        "model": args.model,
        "revision": revision,
        "transformers_version": transformers.__version__,
        "implementation": "PaliGemma-3B fixed-weight pre-softmax eager Gemma final-query->image-token-logit scaling with optional language RMSNorm epsilon override",
        "dataset": args.dataset,
        "option": args.option,
        "method": args.method,
        "weight": requested_weight,
        "requested_rms_norm_eps": args.rms_norm_eps,
        "language_rms_norm_count": getattr(model, "_adaptvis_rms_norm_count", None),
        "language_rms_norm_eps_before": getattr(model, "_adaptvis_rms_norm_eps_before", None),
        "active_rms_norm_eps": getattr(model, "_adaptvis_rms_norm_eps_after", None),
        "max_layers": args.max_layers,
        "prompt_prefix": args.prompt_prefix,
        "max_new_tokens": args.max_new_tokens,
        "dtype_argument": args.dtype,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "num_samples": sid,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "records": records,
    }
    print("\n" + "=" * 100)
    print(
        f"RESULT: {correct_count}/{sid} accuracy={accuracy:.6f} "
        f"method={args.method} weight={requested_weight} "
        f"rms_norm_eps={getattr(model, '_adaptvis_rms_norm_eps_after', None)}"
    )
    print("=" * 100)
    return summary


def _float_tag(value: float) -> str:
    return f"{float(value):.0e}".replace("e-", "em").replace("e+", "ep")


def default_output_path(args: argparse.Namespace) -> Path:
    tag = f"paligemma3b_{args.dataset}_{args.method}"
    if args.method == "scaling_vis":
        tag += f"_w{args.weight:g}"
    if args.rms_norm_eps is not None:
        tag += f"_eps{_float_tag(args.rms_norm_eps)}"
    return Path("output") / f"{tag}.json"


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    model, processor, controller, revision = load_model_and_processor(args)
    summary = evaluate(args, model, processor, controller, revision)
    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
