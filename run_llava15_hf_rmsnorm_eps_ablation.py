#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Hugging Face LLaVA-1.5 RMSNorm-epsilon ablation with RMSNorm tracing.

This runner keeps the official HF LlavaConfig / LlamaConfig and only changes
LlamaRMSNorm.variance_epsilon after loading the checkpoint. For AdaptVis, it
patches eager HF LLaMA attention only during the multimodal prefill pass:
the raw QK logits from the final query token to merged image-token keys are
multiplied before the causal mask and softmax.

When --trace-rmsnorm is enabled, every LlamaRMSNorm layer records, for the
last query token:
  m = mean(x^2)
  gain = 1 / sqrt(m + eps)
  RMS before / after RMSNorm

For AdaptVis, the first weight=1.0 probe is stored as the reference. The final
weight=0.5 or 1.5 run stores per-layer input/output differences against that
reference. This lets you identify the first RMSNorm whose input differs after
the attention intervention.

Example: trace a single sample with final multiplier fixed to 1.5

python3 run_llava15_hf_rmsnorm_eps_ablation.py \
  --dataset Controlled_Images_A \
  --rms-norm-eps 1e-6 \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold -1 \
  --max-layers 32 \
  --dtype float32 \
  --device cuda \
  --limit 1 \
  --max-new-tokens 1 \
  --trace-rmsnorm \
  --output output/rms_trace_eps1e6_weight1p5_sid0.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoProcessor, LlavaConfig, LlavaForConditionalGeneration

try:
    from transformers.cache_utils import Cache
except Exception:  # Older transformers releases.
    Cache = Any  # type: ignore[misc,assignment]

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from dataset_zoo import get_dataset

try:
    # This is the collate function used by main_aro.py when image_preprocess=None.
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "a272c74"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate standard HF LLaVA-1.5 while changing only "
            "LlamaRMSNorm epsilon, optionally tracing RMSNorm statistics."
        )
    )
    parser.add_argument(
        "--dataset",
        default="Controlled_Images_A",
        choices=["Controlled_Images_A", "Controlled_Images_B"],
    )
    parser.add_argument("--option", default="four", choices=["two", "four", "six"])
    parser.add_argument(
        "--rms-norm-eps",
        default=1e-6,
        type=float,
        help="Value assigned to every HF LlamaRMSNorm.variance_epsilon.",
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
        help="Use the same dtype for all compared runs.",
    )
    parser.add_argument("--max-new-tokens", default=100, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument(
        "--start-index",
        default=0,
        type=int,
        help="Skip dataset samples before this zero-based index.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Evaluate this many samples after --start-index.",
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
        help="Allow mismatched checkpoint tensors. Leave disabled for a clean ablation.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--print-first",
        default=5,
        type=int,
        help="Print detailed diagnostics for the first N evaluated samples.",
    )
    parser.add_argument(
        "--trace-rmsnorm",
        action="store_true",
        help="Record prefill RMSNorm statistics and scaled-vs-unscaled differences.",
    )
    parser.add_argument(
        "--trace-image-tokens",
        action="store_true",
        help="Also save aggregate RMSNorm-input statistics over merged image tokens.",
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


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def set_llama_rmsnorm_epsilon(
    model: LlavaForConditionalGeneration,
    epsilon: float,
) -> Tuple[int, List[float]]:
    """Change only the epsilon used by every HF LlamaRMSNorm module."""
    norms = [
        module
        for module in model.language_model.modules()
        if isinstance(module, LlamaRMSNorm)
    ]
    before = sorted({float(module.variance_epsilon) for module in norms})

    for module in norms:
        module.variance_epsilon = float(epsilon)

    # These are metadata / logging fields. The module attribute above controls forward.
    model.language_model.config.rms_norm_eps = float(epsilon)
    model.config.text_config.rms_norm_eps = float(epsilon)

    after = sorted({float(module.variance_epsilon) for module in norms})
    print("RMSNorm modules:", len(norms), flush=True)
    print("RMSNorm epsilon before override:", before, flush=True)
    print("RMSNorm epsilon after override:", after, flush=True)

    if not norms:
        raise RuntimeError("No HF LlamaRMSNorm modules were found.")
    if after != [float(epsilon)]:
        raise RuntimeError(
            f"Failed to set all RMSNorm epsilons to {epsilon}; got {after}."
        )
    return len(norms), before


@dataclass
class GenerationDiagnostics:
    requested_weight: float
    modified_calls: int
    image_token_count: int
    merged_sequence_length: int
    image_start: Optional[int]
    image_end: Optional[int]
    rmsnorm_trace: Optional[Dict[str, Any]] = None


class HFAdaptVisController:
    """Runtime state shared by native HF LLaVA merge and LLaMA attention."""

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
    Reproduce the image_to_overwrite mask used by HF LLaVA's merge function.
    Compatible with transformers 4.39.x LLaVA-1.5.
    """
    del attention_mask  # The original merge uses token layout / padding convention.

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
    batch_indices, non_image_indices = torch.where(input_ids != int(image_token_index))
    increments = special_image_token_mask.to(torch.long) * (int(num_image_patches) - 1) + 1
    new_token_positions = torch.cumsum(increments, dim=-1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]

    if left_padding:
        new_token_positions = new_token_positions + nb_image_pad[:, None]

    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
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
    """Wrap native HF merge without changing its outputs."""
    original_merge = model._merge_input_ids_with_image_features

    def wrapped_merge(
        self,
        image_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels=None,
        *args,
        **kwargs,
    ):
        outputs = original_merge(
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask,
            labels,
            *args,
            **kwargs,
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

    model._merge_input_ids_with_image_features = types.MethodType(wrapped_merge, model)


def _delegate_original_attention(
    original,
    *,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_value: Optional[Cache],
    output_attentions: bool,
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    kwargs: Dict[str, Any],
):
    """Call original attention across transformers 4.39+ signatures."""
    call_kwargs: Dict[str, Any] = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_value": past_key_value,
        "output_attentions": output_attentions,
        "use_cache": use_cache,
    }
    if cache_position is not None:
        call_kwargs["cache_position"] = cache_position
    if position_embeddings is not None:
        call_kwargs["position_embeddings"] = position_embeddings
    call_kwargs.update(kwargs)

    try:
        return original(**call_kwargs)
    except TypeError:
        # transformers 4.39.1 does not accept cache_position / position_embeddings.
        call_kwargs.pop("cache_position", None)
        call_kwargs.pop("position_embeddings", None)
        return original(**call_kwargs)


def install_hf_adaptvis_attention(
    model: LlavaForConditionalGeneration,
    controller: HFAdaptVisController,
) -> None:
    """
    Patch native eager HF LLaMA attention only for a non-unit AdaptVis weight.

    The baseline probe (weight=1) remains byte-for-byte on original HF attention
    wherever the installed transformers implementation permits it.
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
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
                    return _delegate_original_attention(
                        original,
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        kwargs=kwargs,
                    )

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
                    bsz, q_len, num_heads, head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, num_kv_heads, head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, num_kv_heads, head_dim
                ).transpose(1, 2)

                effective_past = getattr(attn, "past_key_value", past_key_value)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    try:
                        cos, sin = attn.rotary_emb(value_states, position_ids)
                    except TypeError:
                        kv_seq_len = key_states.shape[-2]
                        if effective_past is not None:
                            try:
                                kv_seq_len += effective_past.get_usable_length(
                                    kv_seq_len,
                                    getattr(attn, "layer_idx", idx),
                                )
                            except Exception:
                                pass
                        cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)

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

                present_key_value = effective_past
                if effective_past is not None:
                    if hasattr(effective_past, "update"):
                        cache_kwargs = {
                            "sin": sin,
                            "cos": cos,
                            "cache_position": cache_position,
                        }
                        key_states, value_states = effective_past.update(
                            key_states,
                            value_states,
                            getattr(attn, "layer_idx", idx),
                            cache_kwargs,
                        )
                    else:
                        key_states = torch.cat([effective_past[0], key_states], dim=2)
                        value_states = torch.cat([effective_past[1], value_states], dim=2)
                        present_key_value = (
                            (key_states, value_states) if use_cache else None
                        )

                key_states = repeat_kv(key_states, num_kv_groups)
                value_states = repeat_kv(value_states, num_kv_groups)

                attn_weights = torch.matmul(
                    query_states,
                    key_states.transpose(2, 3),
                ) / math.sqrt(head_dim)

                image_mask = mask.to(device=attn_weights.device, dtype=torch.bool)
                if image_mask.shape[0] == 1 and bsz > 1:
                    image_mask = image_mask.expand(bsz, -1)
                kv_len = int(attn_weights.shape[-1])
                if image_mask.shape[-1] != kv_len:
                    raise RuntimeError(
                        "AdaptVis image mask and attention KV length differ: "
                        f"mask={image_mask.shape[-1]}, kv={kv_len}, layer={idx}."
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
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
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

                expected_shape = (bsz, num_heads, q_len, head_dim)
                if tuple(attn_output.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected attention output shape {tuple(attn_output.shape)}; "
                        f"expected {expected_shape}."
                    )

                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .reshape(bsz, q_len, num_heads * head_dim)
                )
                if getattr(config, "pretraining_tp", 1) > 1:
                    tp = int(config.pretraining_tp)
                    split_outputs = attn_output.split(attn.hidden_size // tp, dim=2)
                    o_slices = attn.o_proj.weight.split(attn.hidden_size // tp, dim=1)
                    attn_output = sum(
                        F.linear(split_outputs[i], o_slices[i])
                        for i in range(tp)
                    )
                else:
                    attn_output = attn.o_proj(attn_output)

                returned_weights = attn_weights if output_attentions else None
                return attn_output, returned_weights, present_key_value

            return forward

        attn_module.forward = make_forward(attn_module, original_forward, layer_index)


class RMSNormTracer:
    """
    Forward-hook recorder for native HF LlamaRMSNorm modules.

    A trace is recorded only for multimodal prefill (q_len equals the captured
    merged sequence length). Generation steps with q_len=1 are ignored.
    """

    def __init__(
        self,
        language_model: torch.nn.Module,
        controller: HFAdaptVisController,
        trace_image_tokens: bool,
    ) -> None:
        self.controller = controller
        self.trace_image_tokens = bool(trace_image_tokens)
        self.active = False
        self.weight = 1.0
        self.records: List[Dict[str, Any]] = []
        self.reference: Dict[str, Dict[str, torch.Tensor]] = {}
        self.handles = []

        for name, module in language_model.named_modules():
            if isinstance(module, LlamaRMSNorm):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

        if not self.handles:
            raise RuntimeError("RMSNormTracer could not find LlamaRMSNorm modules.")

    @staticmethod
    def _compare(reference: torch.Tensor, current: torch.Tensor) -> Dict[str, float]:
        reference = reference.detach().float().flatten()
        current = current.detach().float().flatten()
        delta = current - reference
        ref_norm = reference.norm().clamp_min(1e-12)
        current_norm = current.norm().clamp_min(1e-12)
        return {
            "relative_l2": scalar(delta.norm() / ref_norm),
            "max_abs_delta": scalar(delta.abs().max()),
            "mean_abs_delta": scalar(delta.abs().mean()),
            "cosine": scalar(torch.dot(reference, current) / (ref_norm * current_norm)),
        }

    @staticmethod
    def _token_summary(states: torch.Tensor, eps: float) -> Optional[Dict[str, float]]:
        if states.numel() == 0:
            return None
        states = states.detach().float()
        mean_x2 = states.pow(2).mean(dim=-1)
        gain = torch.rsqrt(mean_x2 + eps)
        normalized_rms = torch.sqrt(mean_x2 / (mean_x2 + eps))
        eps_over_m = eps / mean_x2.clamp_min(1e-30)
        return {
            "token_count": int(mean_x2.numel()),
            "mean_x2_mean": scalar(mean_x2.mean()),
            "mean_x2_median": scalar(mean_x2.median()),
            "mean_x2_min": scalar(mean_x2.min()),
            "mean_x2_max": scalar(mean_x2.max()),
            "gain_mean": scalar(gain.mean()),
            "gain_median": scalar(gain.median()),
            "normalized_rms_before_gamma_mean": scalar(normalized_rms.mean()),
            "eps_over_mean_x2_median": scalar(eps_over_m.median()),
        }

    def _make_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not self.active or not inputs:
                return
            hidden_states = inputs[0]
            if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
                return

            _, q_len, _ = hidden_states.shape
            if (
                q_len <= 1
                or self.controller.merged_sequence_length <= 1
                or q_len != self.controller.merged_sequence_length
            ):
                return

            if not torch.is_tensor(output) or output.ndim != 3:
                return

            eps = float(getattr(module, "variance_epsilon"))
            input_last = hidden_states[0, -1].detach().float().cpu()
            output_last = output[0, -1].detach().float().cpu()

            input_mean_x2 = input_last.pow(2).mean()
            input_rms = input_mean_x2.sqrt()
            gain = torch.rsqrt(input_mean_x2 + eps)
            norm_rms_before_gamma = torch.sqrt(input_mean_x2 / (input_mean_x2 + eps))
            output_mean_y2 = output_last.pow(2).mean()

            record: Dict[str, Any] = {
                "name": name,
                "epsilon": eps,
                "last_query": {
                    "input_mean_x2": scalar(input_mean_x2),
                    "input_rms": scalar(input_rms),
                    "eps_over_input_mean_x2": scalar(
                        torch.tensor(eps) / input_mean_x2.clamp_min(1e-30)
                    ),
                    "normalization_gain": scalar(gain),
                    "normalized_rms_before_gamma": scalar(norm_rms_before_gamma),
                    "output_mean_y2_after_gamma": scalar(output_mean_y2),
                    "output_rms_after_gamma": scalar(output_mean_y2.sqrt()),
                },
                "all_prefill_tokens": self._token_summary(
                    hidden_states.reshape(-1, hidden_states.shape[-1]),
                    eps,
                ),
            }

            if self.trace_image_tokens:
                image_mask = self.controller.image_mask
                if (
                    image_mask is not None
                    and tuple(image_mask.shape) == tuple(hidden_states.shape[:2])
                ):
                    image_states = hidden_states[
                        image_mask.to(hidden_states.device, dtype=torch.bool)
                    ]
                    record["image_prefill_tokens"] = self._token_summary(image_states, eps)
                else:
                    record["image_prefill_tokens"] = None

            if abs(self.weight - 1.0) < 1e-12:
                self.reference[name] = {
                    "input": input_last.clone(),
                    "output": output_last.clone(),
                }
            else:
                baseline = self.reference.get(name)
                if baseline is not None:
                    record["vs_weight_1"] = {
                        "input": self._compare(baseline["input"], input_last),
                        "output": self._compare(baseline["output"], output_last),
                    }
                else:
                    record["vs_weight_1"] = None

            self.records.append(record)

        return hook

    def begin_generation(self, weight: float) -> None:
        self.active = True
        self.weight = float(weight)
        self.records = []
        if abs(self.weight - 1.0) < 1e-12:
            # Every probe is a baseline reference for its own sample.
            self.reference = {}

    def finish_generation(self) -> Dict[str, Any]:
        result = {
            "weight": float(self.weight),
            "prefill_rmsnorm_records": self.records,
        }
        self.active = False
        return result


def load_hf_model(
    args: argparse.Namespace,
) -> Tuple[
    LlavaForConditionalGeneration,
    AutoProcessor,
    Dict[str, Any],
    HFAdaptVisController,
    Optional[RMSNormTracer],
]:
    print(f"transformers version: {transformers.__version__}", flush=True)
    print(
        f"Loading top-level LLaVA config from {args.model}@{args.revision}",
        flush=True,
    )
    llava_config = LlavaConfig.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    checkpoint_rms_norm_eps = float(llava_config.text_config.rms_norm_eps)
    print("Checkpoint text_config.rms_norm_eps:", checkpoint_rms_norm_eps, flush=True)
    print("Requested RMSNorm epsilon:", float(args.rms_norm_eps), flush=True)

    # Pre-softmax intervention requires eager LLaMA attention.
    if args.method != "base":
        llava_config._attn_implementation = "eager"
        try:
            llava_config.text_config._attn_implementation = "eager"
        except Exception:
            pass

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

    print("\n" + "=" * 100, flush=True)
    print("CHECKPOINT LOADING INFORMATION", flush=True)
    print("=" * 100, flush=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading_info.get(key, [])
        print(f"{key}: {len(values)}", flush=True)
        for item in values[:20]:
            print(f"  {item}", flush=True)
        if len(values) > 20:
            print(f"  ... {len(values) - 20} more", flush=True)
    print("=" * 100 + "\n", flush=True)

    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        print(
            "[WARNING] Some tensors were not loaded exactly. Inspect the report above.",
            flush=True,
        )

    rmsnorm_count, rmsnorm_eps_before = set_llama_rmsnorm_epsilon(
        model,
        args.rms_norm_eps,
    )
    model._rmsnorm_ablation_count = rmsnorm_count
    model._rmsnorm_ablation_before = rmsnorm_eps_before
    model._rmsnorm_checkpoint_eps = checkpoint_rms_norm_eps

    model.eval()
    model.to(args.device)
    model.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    controller = HFAdaptVisController(max_layers=args.max_layers)
    if args.method != "base":
        install_merge_capture(model, controller)
        install_hf_adaptvis_attention(model, controller)

    tracer: Optional[RMSNormTracer] = None
    if args.trace_rmsnorm:
        tracer = RMSNormTracer(
            model.language_model,
            controller,
            trace_image_tokens=args.trace_image_tokens,
        )

    print("Active model classes:", flush=True)
    print(
        f"  LLaVA: {type(model).__module__}.{type(model).__name__}",
        flush=True,
    )
    print(
        "  language_model: "
        f"{type(model.language_model).__module__}."
        f"{type(model.language_model).__name__}",
        flush=True,
    )
    print(
        "  attention implementation: "
        f"{getattr(model.config, '_attn_implementation', 'checkpoint default')}",
        flush=True,
    )
    print("  first parameter dtype:", next(model.parameters()).dtype, flush=True)
    print(f"  AdaptVis layers: [0, {controller.max_layers})", flush=True)
    print("  trace RMSNorm:", bool(tracer is not None), flush=True)

    return model, processor, loading_info, controller, tracer


def norm_gold(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if len(value) else ""
    return str(value).strip() if value else ""


def is_correct(gold: Any, generation: str) -> bool:
    gold_text = norm_gold(gold)
    generation_text = str(generation).strip()
    if not gold_text:
        return False
    correct = gold_text in generation_text or gold_text.lower() in generation_text.lower()
    if gold_text.lower() == "on" and "front" in generation_text.lower():
        correct = False
    return bool(correct)


def decode_generated(
    processor: AutoProcessor,
    output: Any,
    prompt_length: int,
) -> str:
    sequences = output.sequences if hasattr(output, "sequences") else output.get("sequences")
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
            f"Prompt file not found: {path}. Run this script from the AdaptVis root."
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
    """Match model_zoo/llava15.py's image_options iteration."""
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def brief_trace_print(trace: Optional[Dict[str, Any]]) -> None:
    if not trace:
        return
    records = trace.get("prefill_rmsnorm_records", [])
    if not records:
        print("[RMSNorm trace] no prefill records captured", flush=True)
        return

    print("\n" + "=" * 132, flush=True)
    print(f"RMSNorm prefill trace | AdaptVis weight={trace['weight']:g}", flush=True)
    print(
        "module                                                     "
        "m=mean(x^2)   eps/m        gain         out_rms      "
        "vs-w1 input relL2",
        flush=True,
    )
    print("-" * 132, flush=True)
    for record in records:
        q = record["last_query"]
        diff = record.get("vs_weight_1")
        rel_l2 = float("nan")
        if diff is not None:
            rel_l2 = diff["input"]["relative_l2"]
        print(
            f"{record['name']:<58} "
            f"{q['input_mean_x2']:.3e} "
            f"{q['eps_over_input_mean_x2']:.3e} "
            f"{q['normalization_gain']:.3e} "
            f"{q['output_rms_after_gamma']:.3e} "
            f"{rel_l2:.3e}",
            flush=True,
        )
    print("=" * 132 + "\n", flush=True)


@torch.inference_mode()
def generate_once(
    *,
    model: LlavaForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
    controller: HFAdaptVisController,
    tracer: Optional[RMSNormTracer],
    weight: float,
    max_new_tokens: int,
) -> Tuple[Any, GenerationDiagnostics]:
    controller.begin_generation(weight)
    if tracer is not None:
        tracer.begin_generation(weight)

    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )

    diagnostics = controller.finish_generation()
    if tracer is not None:
        diagnostics.rmsnorm_trace = tracer.finish_generation()
    return output, diagnostics


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    controller: HFAdaptVisController,
    tracer: Optional[RMSNormTracer],
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
    start_index = max(0, int(args.start_index))
    if start_index >= total_available:
        raise ValueError(
            f"--start-index={start_index} is outside dataset length {total_available}."
        )
    end_index = total_available
    if args.limit is not None:
        end_index = min(total_available, start_index + max(0, int(args.limit)))

    model_dtype = next(model.parameters()).dtype
    records: List[Dict[str, Any]] = []
    correct_count = 0
    evaluated = 0
    global_index = 0
    progress = tqdm(
        total=end_index - start_index,
        desc=f"HF LLaVA {args.method}",
    )

    stop = False
    for batch in loader:
        for image in extract_images_from_batch(batch):
            if global_index < start_index:
                global_index += 1
                continue
            if global_index >= end_index:
                stop = True
                break

            prompt = prompts[global_index]
            gold = norm_gold(answers[global_index])
            inputs = processor(
                text=prompt,
                images=image,
                padding="max_length",
                return_tensors="pt",
                max_length=77,
            ).to(args.device)
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
                    tracer=tracer,
                    weight=1.0,
                    max_new_tokens=args.max_new_tokens,
                )
                probe_generation = decode_generated(processor, probe_output, prompt_length)
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
                tracer=tracer,
                weight=selected_weight,
                max_new_tokens=args.max_new_tokens,
            )
            generation = decode_generated(processor, final_output, prompt_length)
            final_confidence = first_step_confidence(final_output)
            correct = is_correct(gold, generation)
            correct_count += int(correct)

            if args.method != "adapt_vis":
                probe_confidence = final_confidence
                rounded_confidence = float(np.round(final_confidence, 2))

            record: Dict[str, Any] = {
                "sid": global_index,
                "prompt": prompt,
                "gold": gold,
                "method": args.method,
                "selected_weight": float(selected_weight),
                "generation": generation,
                "correct": bool(correct),
                "first_step_confidence": float(final_confidence),
                "probe_generation": probe_generation,
                "probe_confidence": probe_confidence,
                "rounded_probe_confidence": rounded_confidence,
                "probe_diagnostics": asdict(probe_diag) if probe_diag is not None else None,
                "final_diagnostics": asdict(final_diag),
            }
            records.append(record)

            if evaluated < args.print_first:
                print("\n" + "-" * 100, flush=True)
                print(f"[SID {global_index}] gold={gold!r}", flush=True)
                if args.method == "adapt_vis":
                    print(
                        f"probe={probe_generation!r} "
                        f"conf={probe_confidence:.6f} "
                        f"rounded={rounded_confidence:.2f}",
                        flush=True,
                    )
                print(
                    f"selected_weight={selected_weight} "
                    f"pred={generation!r} correct={correct}",
                    flush=True,
                )
                print(
                    "image tokens="
                    f"{final_diag.image_token_count}, "
                    f"range=[{final_diag.image_start},{final_diag.image_end}), "
                    f"merged_len={final_diag.merged_sequence_length}, "
                    f"modified_calls={final_diag.modified_calls}",
                    flush=True,
                )
                if args.trace_rmsnorm:
                    print("[probe / unscaled]", flush=True)
                    brief_trace_print(
                        probe_diag.rmsnorm_trace if probe_diag is not None else None
                    )
                    print("[final / selected weight]", flush=True)
                    brief_trace_print(final_diag.rmsnorm_trace)

            evaluated += 1
            global_index += 1
            progress.update(1)

        if stop:
            break

    progress.close()
    accuracy = correct_count / max(evaluated, 1)
    low_branch_count = (
        sum(
            int(record["selected_weight"] == float(args.weight1))
            for record in records
        )
        if args.method == "adapt_vis"
        else None
    )
    high_branch_count = (
        sum(
            int(record["selected_weight"] == float(args.weight2))
            for record in records
        )
        if args.method == "adapt_vis"
        else None
    )

    summary = {
        "args": vars(args),
        "model": args.model,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "dataset": args.dataset,
        "option": args.option,
        "implementation": "transformers.LlavaForConditionalGeneration",
        "language_model_implementation": "transformers.LlamaForCausalLM",
        "checkpoint_rms_norm_eps": getattr(model, "_rmsnorm_checkpoint_eps", None),
        "requested_rms_norm_eps": float(args.rms_norm_eps),
        "rmsnorm_module_count": getattr(model, "_rmsnorm_ablation_count", None),
        "rmsnorm_eps_before_override": getattr(model, "_rmsnorm_ablation_before", None),
        "active_rms_norm_eps": float(model.language_model.config.rms_norm_eps),
        "method": args.method,
        "weight": args.weight,
        "weight1": args.weight1,
        "weight2": args.weight2,
        "threshold": args.threshold,
        "confidence_round_decimals": 2,
        "max_layers": args.max_layers,
        "intervention": (
            "prefill only; final query to merged image-token keys; "
            "raw QK logits multiplied before causal mask and softmax"
        ),
        "dtype_argument": args.dtype,
        "model_parameter_dtype": str(model_dtype),
        "start_index": start_index,
        "end_index": end_index,
        "num_samples": evaluated,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "low_branch_count": low_branch_count,
        "high_branch_count": high_branch_count,
        "records": records,
    }

    print("\n" + "=" * 100, flush=True)
    print(
        f"RESULT: {correct_count}/{evaluated} "
        f"accuracy={accuracy:.6f} "
        f"rms_norm_eps={args.rms_norm_eps:g} "
        f"method={args.method}",
        flush=True,
    )
    if args.method == "adapt_vis":
        print(
            f"branches: weight1({args.weight1})={low_branch_count}, "
            f"weight2({args.weight2})={high_branch_count}",
            flush=True,
        )
    print("=" * 100, flush=True)
    return summary


def default_output_path(args: argparse.Namespace) -> Path:
    method_suffix = args.method
    if args.method == "scaling_vis":
        method_suffix += f"_w{args.weight:g}"
    elif args.method == "adapt_vis":
        method_suffix += (
            f"_w1_{args.weight1:g}_w2_{args.weight2:g}_thr_{args.threshold:g}"
        )
    eps_suffix = f"{args.rms_norm_eps:.0e}".replace("-", "m")
    trace_suffix = "_rms_trace" if args.trace_rmsnorm else ""
    return Path(
        "output/"
        f"hf_llava15_{args.dataset}_rms_eps_{eps_suffix}_"
        f"{method_suffix}{trace_suffix}.json"
    )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if args.rms_norm_eps <= 0:
        raise ValueError("--rms-norm-eps must be positive.")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive.")

    model, processor, _, controller, tracer = load_hf_model(args)
    summary = evaluate(args, model, processor, controller, tracer)

    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Saved results to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
