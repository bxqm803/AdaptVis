#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF LLaVA-1.5 baseline / ScalingVis / AdaptVis with selectable text config.

This script uses:
  - transformers.LlavaForConditionalGeneration
  - transformers.LlamaForCausalLM
  - the native Hugging Face LLaVA image/text merge
  - the native Hugging Face generation path

It optionally replaces the checkpoint text config with parameter values
equivalent to this repository's custom LLaMAConfig().

AdaptVis implementation:
  1. Capture the exact image-token mask produced by the native HF LLaVA merge.
  2. Force eager LLaMA attention.
  3. During the initial prompt prefill only, multiply the raw attention logits
     from the final query to all merged image-token keys.
  4. Apply the multiplication before adding the attention mask and before
     softmax.
  5. For adapt_vis, first run weight=1.0, round the first-token confidence to
     two decimals, and select weight1/weight2 using the threshold.

Important:
  - base and weight=1.0 call the original HF attention forward exactly.
  - only a non-unit intervention weight activates the patched attention path.
  - this is an HF-model experiment; it does not use the repository's custom
    LLaVA, custom LLaMA, or custom greedy-search implementation.

Place this file in the AdaptVis repository root.

Recommended:
    transformers==4.39.1

Examples
--------
HF model + custom-equivalent config + AdaptVis:

python3 run_llava15_hf_customconfig_baseline.py \
  --dataset Controlled_Images_A \
  --text-config custom \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.4 \
  --device cuda \
  --dtype float32 \
  --download

HF model + custom-equivalent config + baseline:

python3 run_llava15_hf_customconfig_baseline.py \
  --dataset Controlled_Images_A \
  --text-config custom \
  --method base \
  --device cuda \
  --dtype float32 \
  --download

HF model + checkpoint config + AdaptVis:

python3 run_llava15_hf_customconfig_baseline.py \
  --dataset Controlled_Images_A \
  --text-config checkpoint \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.4 \
  --device cuda \
  --dtype float32 \
  --download
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import (
    AutoProcessor,
    LlamaConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
)
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from dataset_zoo import get_dataset

try:
    # main_aro.py uses the repository collate function when image_preprocess=None.
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "a272c74"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HF LLaVA-1.5 with checkpoint/custom-equivalent text "
            "configuration and optional ScalingVis/AdaptVis."
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
            "'custom': use values equivalent to model_zoo/llama/"
            "configuration_llama.py; 'checkpoint': retain the HF checkpoint "
            "text config."
        ),
    )
    parser.add_argument(
        "--config-patch",
        default="",
        help=(
            "Comma-separated config fields copied from the opposite config. "
            "With --text-config checkpoint, selected fields come from custom; "
            "with --text-config custom, selected fields are restored from the "
            "checkpoint. Aliases: norm, position, tokens, all. Example: "
            "--text-config checkpoint --config-patch rms_norm_eps"
        ),
    )
    parser.add_argument(
        "--method",
        default="base",
        choices=["base", "scaling_vis", "adapt_vis"],
    )
    parser.add_argument(
        "--weight",
        default=0.5,
        type=float,
        help="Fixed image-logit multiplier used by scaling_vis.",
    )
    parser.add_argument("--weight1", default=0.5, type=float)
    parser.add_argument("--weight2", default=1.5, type=float)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument(
        "--max-layers",
        default=32,
        type=int,
        help="Apply the intervention to layers [0, max_layers).",
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
            "Use the same dtype for all compared runs. float32 is the closest "
            "match to the repository's default from_pretrained path."
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
        "--output",
        default=None,
        help="Output JSON path. A descriptive default is used when omitted.",
    )
    parser.add_argument(
        "--ignore-mismatched-sizes",
        action="store_true",
        help=(
            "Allow mismatched checkpoint tensors. Leave disabled for a clean "
            "configuration ablation unless the original experiment requires it."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
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
    Official HF LlamaConfig carrying the values used by the repository's
    custom LLaMAConfig() and custom LLaMA implementation.
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


SAFE_CONFIG_PATCH_FIELDS = (
    "rms_norm_eps",
    "max_position_embeddings",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "use_cache",
    "pretraining_tp",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "attention_bias",
    "attention_dropout",
)

CONFIG_PATCH_ALIASES = {
    "norm": ("rms_norm_eps",),
    "position": ("max_position_embeddings", "rope_theta", "rope_scaling"),
    "tokens": ("pad_token_id", "bos_token_id", "eos_token_id"),
}


def config_snapshot(config: LlamaConfig) -> Dict[str, Any]:
    return {name: getattr(config, name, None) for name in CONFIG_FIELDS}


def resolve_config_patch_fields(
    patch_spec: str,
    checkpoint_config: LlamaConfig,
    custom_config: LlamaConfig,
) -> List[str]:
    spec = str(patch_spec or "").strip()
    if not spec:
        return []

    requested = [item.strip() for item in spec.split(",") if item.strip()]
    fields: List[str] = []
    for item in requested:
        if item == "all":
            for field in SAFE_CONFIG_PATCH_FIELDS:
                if getattr(checkpoint_config, field, None) != getattr(
                    custom_config, field, None
                ):
                    fields.append(field)
            continue
        if item in CONFIG_PATCH_ALIASES:
            fields.extend(CONFIG_PATCH_ALIASES[item])
            continue
        if item not in SAFE_CONFIG_PATCH_FIELDS:
            raise ValueError(
                f"Unsupported --config-patch field {item!r}. Allowed fields: "
                f"{', '.join(SAFE_CONFIG_PATCH_FIELDS)}; aliases: "
                f"{', '.join(CONFIG_PATCH_ALIASES)}, all."
            )
        fields.append(item)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(fields))


def build_active_text_config(
    *,
    base_mode: str,
    patch_spec: str,
    checkpoint_config: LlamaConfig,
) -> Tuple[LlamaConfig, List[str], str]:
    custom_config = build_custom_equivalent_hf_llama_config()
    if base_mode == "custom":
        active = copy.deepcopy(custom_config)
        patch_source = checkpoint_config
        patch_source_name = "checkpoint"
    elif base_mode == "checkpoint":
        active = copy.deepcopy(checkpoint_config)
        patch_source = custom_config
        patch_source_name = "custom"
    else:
        raise ValueError(f"Unsupported text config mode: {base_mode}")

    patch_fields = resolve_config_patch_fields(
        patch_spec, checkpoint_config, custom_config
    )
    for field in patch_fields:
        setattr(active, field, copy.deepcopy(getattr(patch_source, field)))

    return active, patch_fields, patch_source_name


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


@dataclass
class GenerationDiagnostics:
    requested_weight: float
    modified_calls: int
    image_token_count: int
    merged_sequence_length: int
    image_start: Optional[int]
    image_end: Optional[int]


class HFAdaptVisController:
    """
    Runtime state shared by the patched native HF LLaVA merge and HF LLaMA
    attention modules.
    """

    def __init__(self, max_layers: int = 32) -> None:
        self.max_layers = int(max_layers)
        self.enabled = False
        self.weight = 1.0
        self.image_mask: Optional[torch.Tensor] = None
        self.modified_calls = 0
        self.merged_sequence_length = 0

    def begin_generation(self, weight: float) -> None:
        self.weight = float(weight)
        self.enabled = self.weight != 1.0
        self.image_mask = None
        self.modified_calls = 0
        self.merged_sequence_length = 0

    def finish_generation(self) -> GenerationDiagnostics:
        count = 0
        start: Optional[int] = None
        end: Optional[int] = None

        if self.image_mask is not None and self.image_mask.numel() > 0:
            first_mask = self.image_mask[0].detach().bool().cpu()
            indices = torch.nonzero(first_mask, as_tuple=False).flatten()
            count = int(indices.numel())
            if count:
                start = int(indices.min())
                end = int(indices.max()) + 1

        result = GenerationDiagnostics(
            requested_weight=float(self.weight),
            modified_calls=int(self.modified_calls),
            image_token_count=count,
            merged_sequence_length=int(self.merged_sequence_length),
            image_start=start,
            image_end=end,
        )
        self.enabled = False
        return result


def compute_hf_merged_image_mask(
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    image_token_index: int,
    pad_token_id: int,
    num_image_patches: int,
) -> torch.Tensor:
    """
    Reproduce the image_to_overwrite mask constructed by transformers 4.39.1
    LlavaForConditionalGeneration._merge_input_ids_with_image_features.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids [B,L], got {tuple(input_ids.shape)}")

    batch_size, sequence_length = input_ids.shape
    target_device = input_ids.device

    pad_value = torch.tensor(pad_token_id, device=target_device)
    left_padding = not bool(torch.sum(input_ids[:, -1] == pad_value).item())

    special_image_token_mask = input_ids == int(image_token_index)
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)

    if int(num_special_image_tokens.max().item()) == 0:
        return torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=target_device,
        )

    max_embed_dim = int(
        (
            num_special_image_tokens.max() * (int(num_image_patches) - 1)
            + sequence_length
        ).item()
    )

    batch_indices, non_image_indices = torch.where(
        input_ids != int(image_token_index)
    )

    increments = (
        special_image_token_mask.to(torch.long) * (int(num_image_patches) - 1)
        + 1
    )
    new_token_positions = torch.cumsum(increments, dim=-1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]

    if left_padding:
        new_token_positions = new_token_positions + nb_image_pad[:, None]

    text_to_overwrite = new_token_positions[
        batch_indices,
        non_image_indices,
    ]

    image_to_overwrite = torch.ones(
        batch_size,
        max_embed_dim,
        dtype=torch.bool,
        device=target_device,
    )
    image_to_overwrite[batch_indices, text_to_overwrite] = False
    image_to_overwrite &= (
        image_to_overwrite.cumsum(-1) - 1
        >= nb_image_pad[:, None].to(target_device)
    )

    expected = int(num_special_image_tokens.sum().item()) * int(num_image_patches)
    actual = int(image_to_overwrite.sum().item())
    if actual != expected:
        raise RuntimeError(
            "Failed to reconstruct HF merged image mask: "
            f"expected {expected} image positions, found {actual}."
        )

    return image_to_overwrite


def install_merge_capture(
    model: LlavaForConditionalGeneration,
    controller: HFAdaptVisController,
) -> None:
    """
    Wrap the native HF merge without changing its outputs. The wrapper only
    reconstructs and stores the exact merged image-token mask.
    """
    original_merge = model._merge_input_ids_with_image_features

    def wrapped_merge(
        self,
        image_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels,
    ):
        outputs = original_merge(
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask,
            labels,
        )
        final_embedding = outputs[0]

        mask = compute_hf_merged_image_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_index=int(self.config.image_token_index),
            pad_token_id=int(self.pad_token_id),
            num_image_patches=int(image_features.shape[1]),
        )
        controller.image_mask = mask.to(final_embedding.device)
        controller.merged_sequence_length = int(final_embedding.shape[1])
        return outputs

    model._merge_input_ids_with_image_features = types.MethodType(
        wrapped_merge,
        model,
    )


def install_hf_adaptvis_attention(
    model: LlavaForConditionalGeneration,
    controller: HFAdaptVisController,
) -> None:
    """
    Patch native HF eager LLaMA attention modules.

    For base/weight=1.0, the saved original HF forward is called unchanged.
    For a non-unit weight, only initial prefill calls whose q_len equals the
    captured merged sequence length enter the custom path.
    """
    layers = model.language_model.model.layers

    for layer_index, layer in enumerate(layers):
        attn_module = layer.self_attn
        original_forward = attn_module.forward

        def make_forward(attn, original, idx):
            def forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[
                    Tuple[torch.Tensor, torch.Tensor]
                ] = None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                mask = controller.image_mask
                should_intervene = (
                    controller.enabled
                    and controller.weight != 1.0
                    and idx < controller.max_layers
                    and mask is not None
                    and mask.ndim == 2
                    and mask.shape[0] in (1, bsz)
                    and mask.shape[-1] == q_len
                    and controller.merged_sequence_length == q_len
                    and q_len > 1
                )

                if not should_intervene:
                    call_kwargs = dict(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                    # position_embeddings was added in newer Transformers.
                    if position_embeddings is not None:
                        call_kwargs["position_embeddings"] = position_embeddings
                    call_kwargs.update(kwargs)
                    try:
                        return original(**call_kwargs)
                    except TypeError as exc:
                        # Compatibility with 4.39.1, which does not accept
                        # position_embeddings.
                        if "position_embeddings" in call_kwargs:
                            call_kwargs.pop("position_embeddings")
                            return original(**call_kwargs)
                        raise exc

                config = attn.config
                num_heads = int(attn.num_heads)
                head_dim = int(attn.head_dim)
                num_kv_heads = int(attn.num_key_value_heads)
                num_kv_groups = int(attn.num_key_value_groups)

                if getattr(config, "pretraining_tp", 1) > 1:
                    tp = int(config.pretraining_tp)
                    kv_slice = (num_kv_heads * head_dim) // tp
                    q_slices = attn.q_proj.weight.split(
                        (num_heads * head_dim) // tp,
                        dim=0,
                    )
                    k_slices = attn.k_proj.weight.split(kv_slice, dim=0)
                    v_slices = attn.v_proj.weight.split(kv_slice, dim=0)
                    query_states = torch.cat(
                        [F.linear(hidden_states, q_slices[i]) for i in range(tp)],
                        dim=-1,
                    )
                    key_states = torch.cat(
                        [F.linear(hidden_states, k_slices[i]) for i in range(tp)],
                        dim=-1,
                    )
                    value_states = torch.cat(
                        [F.linear(hidden_states, v_slices[i]) for i in range(tp)],
                        dim=-1,
                    )
                else:
                    query_states = attn.q_proj(hidden_states)
                    key_states = attn.k_proj(hidden_states)
                    value_states = attn.v_proj(hidden_states)

                query_states = query_states.view(
                    bsz,
                    q_len,
                    num_heads,
                    head_dim,
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz,
                    q_len,
                    num_kv_heads,
                    head_dim,
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz,
                    q_len,
                    num_kv_heads,
                    head_dim,
                ).transpose(1, 2)

                past_key_value = getattr(
                    attn,
                    "past_key_value",
                    past_key_value,
                )

                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    try:
                        # Transformers 4.39.1 path.
                        cos, sin = attn.rotary_emb(
                            value_states,
                            position_ids,
                        )
                    except TypeError:
                        # Compatibility fallback for older implementations.
                        kv_seq_len = key_states.shape[-2]
                        if past_key_value is not None:
                            try:
                                kv_seq_len += past_key_value.get_usable_length(
                                    kv_seq_len,
                                    getattr(attn, "layer_idx", idx),
                                )
                            except Exception:
                                pass
                        cos, sin = attn.rotary_emb(
                            value_states,
                            seq_len=kv_seq_len,
                        )

                try:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states,
                        key_states,
                        cos,
                        sin,
                    )
                except TypeError:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states,
                        key_states,
                        cos,
                        sin,
                        position_ids,
                    )

                present_key_value = past_key_value
                if past_key_value is not None:
                    if hasattr(past_key_value, "update"):
                        cache_kwargs = {
                            "sin": sin,
                            "cos": cos,
                            "cache_position": cache_position,
                        }
                        key_states, value_states = past_key_value.update(
                            key_states,
                            value_states,
                            getattr(attn, "layer_idx", idx),
                            cache_kwargs,
                        )
                    else:
                        # Legacy tuple fallback.
                        key_states = torch.cat(
                            [past_key_value[0], key_states],
                            dim=2,
                        )
                        value_states = torch.cat(
                            [past_key_value[1], value_states],
                            dim=2,
                        )
                        present_key_value = (
                            (key_states, value_states) if use_cache else None
                        )

                key_states = repeat_kv(key_states, num_kv_groups)
                value_states = repeat_kv(value_states, num_kv_groups)

                attn_weights = torch.matmul(
                    query_states,
                    key_states.transpose(2, 3),
                ) / math.sqrt(head_dim)

                image_mask = mask.to(
                    device=attn_weights.device,
                    dtype=torch.bool,
                )
                if image_mask.shape[0] == 1 and bsz > 1:
                    image_mask = image_mask.expand(bsz, -1)

                kv_len = int(attn_weights.shape[-1])
                if image_mask.shape[-1] != kv_len:
                    raise RuntimeError(
                        "AdaptVis image mask and attention KV length differ: "
                        f"mask={image_mask.shape[-1]}, kv={kv_len}, "
                        f"layer={idx}."
                    )

                weight = float(controller.weight)
                for batch_index in range(bsz):
                    image_indices = torch.nonzero(
                        image_mask[batch_index],
                        as_tuple=False,
                    ).flatten()
                    if image_indices.numel() == 0:
                        continue
                    last_query_logits = attn_weights[
                        batch_index,
                        :,
                        q_len - 1,
                        :,
                    ]
                    selected = last_query_logits.index_select(
                        dim=-1,
                        index=image_indices,
                    )
                    last_query_logits.index_copy_(
                        dim=-1,
                        index=image_indices,
                        source=selected * weight,
                    )

                controller.modified_calls += 1

                if attention_mask is not None:
                    causal_mask = attention_mask[
                        :,
                        :,
                        :,
                        : key_states.shape[-2],
                    ]
                    attn_weights = attn_weights + causal_mask

                attn_weights = F.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query_states.dtype)
                attn_weights = F.dropout(
                    attn_weights,
                    p=float(attn.attention_dropout),
                    training=attn.training,
                )

                attn_output = torch.matmul(attn_weights, value_states)
                expected_shape = (
                    bsz,
                    num_heads,
                    q_len,
                    head_dim,
                )
                if tuple(attn_output.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected attention output shape "
                        f"{tuple(attn_output.shape)}; expected "
                        f"{expected_shape}."
                    )

                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .reshape(bsz, q_len, num_heads * head_dim)
                )

                if getattr(config, "pretraining_tp", 1) > 1:
                    tp = int(config.pretraining_tp)
                    split_outputs = attn_output.split(
                        attn.hidden_size // tp,
                        dim=2,
                    )
                    o_slices = attn.o_proj.weight.split(
                        attn.hidden_size // tp,
                        dim=1,
                    )
                    attn_output = sum(
                        F.linear(split_outputs[i], o_slices[i])
                        for i in range(tp)
                    )
                else:
                    attn_output = attn.o_proj(attn_output)

                returned_weights = (
                    attn_weights if output_attentions else None
                )
                return attn_output, returned_weights, present_key_value

            return forward

        attn_module.forward = make_forward(
            attn_module,
            original_forward,
            layer_index,
        )


def load_hf_model(
    args: argparse.Namespace,
) -> Tuple[
    LlavaForConditionalGeneration,
    AutoProcessor,
    Dict[str, Any],
    HFAdaptVisController,
]:
    print(f"transformers version: {transformers.__version__}")
    print(f"Loading top-level LLaVA config from {args.model}@{args.revision}")

    llava_config = LlavaConfig.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    checkpoint_text_config = copy.deepcopy(llava_config.text_config)
    active_text_config, patch_fields, patch_source_name = build_active_text_config(
        base_mode=args.text_config,
        patch_spec=args.config_patch,
        checkpoint_config=checkpoint_text_config,
    )
    llava_config.text_config = active_text_config

    print(
        f"Config base={args.text_config}; patch fields={patch_fields or 'none'}; "
        f"patch source={patch_source_name if patch_fields else 'none'}"
    )

    # AdaptVis needs access to raw QK logits, so force eager attention.
    llava_config._attn_implementation = "eager"
    try:
        llava_config.text_config._attn_implementation = "eager"
    except Exception:
        pass

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

    model, loading_info = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        **load_kwargs,
    )

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
            "[WARNING] Some tensors were not loaded exactly. Interpret the "
            "result only after checking the entries above."
        )

    model.eval()
    model.to(args.device)

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    controller = HFAdaptVisController(max_layers=args.max_layers)
    install_merge_capture(model, controller)
    install_hf_adaptvis_attention(model, controller)

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
    print(f"  AdaptVis layers: [0, {controller.max_layers})")
    print(f"  generation pad token: {model.generation_config.pad_token_id}")
    print(f"  generation bos token: {model.generation_config.bos_token_id}")
    print(f"  generation eos token: {model.generation_config.eos_token_id}")

    return model, processor, loading_info, controller


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
    sequences = (
        output.sequences
        if hasattr(output, "sequences")
        else output.get("sequences")
    )
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
    Match model_zoo/llava15.py:
        for image_option in batch["image_options"]:
            for image in image_option:
                ...
    """
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


@torch.inference_mode()
def generate_once(
    *,
    model: LlavaForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
    controller: HFAdaptVisController,
    weight: float,
    max_new_tokens: int,
) -> Tuple[Any, GenerationDiagnostics]:
    controller.begin_generation(weight)

    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )

    diagnostics = controller.finish_generation()
    return output, diagnostics


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    controller: HFAdaptVisController,
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
        collate_fn=repository_default_collate,
    )

    total_available = min(len(prompts), len(dataset))
    total_target = (
        min(total_available, args.limit)
        if args.limit is not None
        else total_available
    )

    model_dtype = next(model.parameters()).dtype
    records: List[Dict[str, Any]] = []
    correct_count = 0
    sample_index = 0

    progress = tqdm(
        total=total_target,
        desc=f"HF LLaVA {args.method}",
    )

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

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    device=args.device,
                    dtype=model_dtype,
                )

            prompt_length = int(inputs["input_ids"].shape[-1])

            probe_generation: Optional[str] = None
            probe_confidence: Optional[float] = None
            rounded_confidence: Optional[float] = None
            probe_diag: Optional[GenerationDiagnostics] = None

            if args.method == "base":
                selected_weight = 1.0
            elif args.method == "scaling_vis":
                selected_weight = float(args.weight)
            elif args.method == "adapt_vis":
                probe_output, probe_diag = generate_once(
                    model=model,
                    inputs=inputs,
                    controller=controller,
                    weight=1.0,
                    max_new_tokens=args.max_new_tokens,
                )
                probe_generation = decode_generated(
                    processor,
                    probe_output,
                    prompt_length,
                )
                probe_confidence = first_step_confidence(probe_output)
                rounded_confidence = float(np.round(probe_confidence, 2))
                selected_weight = (
                    float(args.weight1)
                    if rounded_confidence < float(args.threshold)
                    else float(args.weight2)
                )
            else:
                raise ValueError(f"Unsupported method: {args.method}")

            final_output, final_diag = generate_once(
                model=model,
                inputs=inputs,
                controller=controller,
                weight=selected_weight,
                max_new_tokens=args.max_new_tokens,
            )

            generation = decode_generated(
                processor,
                final_output,
                prompt_length,
            )
            final_confidence = first_step_confidence(final_output)
            correct = is_correct(gold, generation)
            correct_count += int(correct)

            if args.method != "adapt_vis":
                probe_confidence = final_confidence
                rounded_confidence = float(np.round(final_confidence, 2))

            record: Dict[str, Any] = {
                "sid": sample_index,
                "prompt": prompt,
                "gold": gold,
                "method": args.method,
                "selected_weight": selected_weight,
                "generation": generation,
                "correct": correct,
                "first_step_confidence": final_confidence,
                "probe_generation": probe_generation,
                "probe_confidence": probe_confidence,
                "rounded_probe_confidence": rounded_confidence,
                "probe_diagnostics": (
                    vars(probe_diag) if probe_diag is not None else None
                ),
                "final_diagnostics": vars(final_diag),
            }
            records.append(record)

            if sample_index < args.print_first:
                print("\n" + "-" * 100)
                print(f"[SID {sample_index}] gold={gold!r}")
                if args.method == "adapt_vis":
                    print(
                        f"probe={probe_generation!r} "
                        f"conf={probe_confidence:.6f} "
                        f"rounded={rounded_confidence:.2f}"
                    )
                print(
                    f"selected_weight={selected_weight} "
                    f"pred={generation!r} correct={correct}"
                )
                print(
                    "image tokens="
                    f"{final_diag.image_token_count}, "
                    f"range=[{final_diag.image_start},"
                    f"{final_diag.image_end}), "
                    f"merged_len={final_diag.merged_sequence_length}, "
                    f"modified_calls={final_diag.modified_calls}"
                )

            sample_index += 1
            progress.update(1)

        if sample_index >= total_target:
            break

    progress.close()

    accuracy = correct_count / max(sample_index, 1)
    low_branch_count = sum(
        int(r["selected_weight"] == float(args.weight1))
        for r in records
    ) if args.method == "adapt_vis" else None
    high_branch_count = sum(
        int(r["selected_weight"] == float(args.weight2))
        for r in records
    ) if args.method == "adapt_vis" else None

    summary = {
        "model": args.model,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "dataset": args.dataset,
        "option": args.option,
        "implementation": "transformers.LlavaForConditionalGeneration",
        "language_model_implementation": "transformers.LlamaForCausalLM",
        "text_config_mode": args.text_config,
        "config_patch": args.config_patch,
        "active_text_config": config_snapshot(model.language_model.config),
        "method": args.method,
        "weight": args.weight,
        "weight1": args.weight1,
        "weight2": args.weight2,
        "threshold": args.threshold,
        "confidence_round_decimals": 2,
        "max_layers": args.max_layers,
        "intervention": (
            "prefill only; final query to merged image-token keys; "
            "raw QK logits multiplied before mask and softmax"
        ),
        "dtype_argument": args.dtype,
        "model_parameter_dtype": str(model_dtype),
        "num_samples": sample_index,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "low_branch_count": low_branch_count,
        "high_branch_count": high_branch_count,
        "records": records,
    }

    print("\n" + "=" * 100)
    print(
        f"RESULT: {correct_count}/{sample_index} "
        f"accuracy={accuracy:.6f} "
        f"text_config={args.text_config} "
        f"method={args.method}"
    )
    if args.method == "adapt_vis":
        print(
            f"branches: weight1({args.weight1})={low_branch_count}, "
            f"weight2({args.weight2})={high_branch_count}"
        )
    print("=" * 100)

    return summary


def default_output_path(args: argparse.Namespace) -> Path:
    method_suffix = args.method
    if args.method == "scaling_vis":
        method_suffix += f"_w{args.weight:g}"
    elif args.method == "adapt_vis":
        method_suffix += (
            f"_w1_{args.weight1:g}_w2_{args.weight2:g}"
            f"_thr_{args.threshold:g}"
        )

    patch_suffix = ""
    if args.config_patch:
        safe_patch = (
            args.config_patch.replace(",", "+")
            .replace(" ", "")
            .replace("/", "-")
        )
        patch_suffix = f"_patch_{safe_patch}"

    return Path(
        "output/"
        f"hf_llava15_{args.dataset}_"
        f"{args.text_config}_config{patch_suffix}_{method_suffix}.json"
    )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    model, processor, _, controller = load_hf_model(args)
    summary = evaluate(args, model, processor, controller)

    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(args)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
