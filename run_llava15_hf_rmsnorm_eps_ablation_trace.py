#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Hugging Face LLaVA-1.5 RMSNorm-epsilon ablation.

The model is loaded with the checkpoint's original LlavaConfig and text_config.
No custom LLaMAConfig is constructed and no text-config fields are replaced.
After from_pretrained finishes, this script changes only:

    LlamaRMSNorm.variance_epsilon

It also mirrors the value into:

    model.language_model.config.rms_norm_eps
    model.config.text_config.rms_norm_eps

Those config assignments are for consistency/logging; the actual computation is
controlled by each LlamaRMSNorm module's variance_epsilon attribute.

Methods:
  - base: native HF model.generate; no attention patch is installed.
  - scaling_vis/adapt_vis: native HF LLaVA merge and generation are retained,
    while the existing pre-softmax AdaptVis intervention is installed in eager
    HF LLaMA attention.

Examples
--------
Checkpoint epsilon (normally 1e-5) + HF AdaptVis:

python3 run_llava15_hf_rmsnorm_eps_ablation_trace.py \
  --dataset Controlled_Images_A \
  --rms-norm-eps 1e-5 \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.4 \
  --device cuda \
  --dtype float32 \
  --download

Only change RMSNorm epsilon to 1e-6:

python3 run_llava15_hf_rmsnorm_eps_ablation_trace.py \
  --dataset Controlled_Images_A \
  --rms-norm-eps 1e-6 \
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
    LlavaConfig,
    LlavaForConditionalGeneration,
)
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
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
            "Evaluate standard HF LLaVA-1.5 while changing only the "
            "LlamaRMSNorm epsilon."
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
        help=(
            "Value assigned to every HF LlamaRMSNorm. Use 1e-5 for the "
            "checkpoint-equivalent control and 1e-6 for the custom value."
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
    parser.add_argument(
        "--trace-rmsnorm",
        action="store_true",
        help=(
            "Observe RMSNorm during the first --print-first samples only. "
            "This installs read-only forward hooks and does not alter the "
            "RMSNorm or generation computation."
        ),
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
    before = sorted(
        {float(module.variance_epsilon) for module in norms}
    )
    for module in norms:
        module.variance_epsilon = float(epsilon)

    # Keep config metadata consistent. These assignments do not rebuild modules.
    model.language_model.config.rms_norm_eps = float(epsilon)
    model.config.text_config.rms_norm_eps = float(epsilon)

    after = sorted(
        {float(module.variance_epsilon) for module in norms}
    )
    print("RMSNorm modules:", len(norms))
    print("RMSNorm epsilon before override:", before)
    print("RMSNorm epsilon after override:", after)

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


class RMSNormTracer:
    """
    Read-only RMSNorm observer.

    The tracer is inactive unless generate_once() explicitly enables it. Its
    forward hooks never return a replacement output and never modify an input,
    weight, cache, controller state, RNG state, or generation argument.

    Each record is for the last token of the initial multimodal prefill pass:
    hidden_states[0, -1]. This is the query token whose image-token attention
    logits are adjusted by AdaptVis.
    """

    EPSILON_1E6 = 1e-6
    EPSILON_1E5 = 1e-5

    def __init__(
        self,
        language_model: torch.nn.Module,
        controller: HFAdaptVisController,
    ) -> None:
        self.controller = controller
        self.active = False
        self.label = ""
        self.weight = 1.0
        self.records: List[Dict[str, Any]] = []
        self._seen_names: set[str] = set()
        self._handles: List[Any] = []

        for name, module in language_model.named_modules():
            if isinstance(module, LlamaRMSNorm):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    @staticmethod
    def _scalar(value: torch.Tensor) -> float:
        return float(value.detach().float().cpu().item())

    def _make_hook(self, name: str):
        def hook(
            module: torch.nn.Module,
            inputs: Tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            # Observation only. Returning None preserves the original output.
            if not self.active or name in self._seen_names:
                return None
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            if not torch.is_tensor(output):
                return None

            hidden_states = inputs[0]
            if hidden_states.ndim != 3 or output.ndim != 3:
                return None

            _, q_len, _ = hidden_states.shape
            # Keep only the initial multimodal prefill; generation steps after
            # the first token have q_len == 1 and are deliberately ignored.
            if q_len <= 1 or q_len != self.controller.merged_sequence_length:
                return None

            x = hidden_states[0, -1].detach().float()
            y = output[0, -1].detach().float()
            m = x.square().mean()
            active_eps = float(module.variance_epsilon)

            # For one fixed RMSNorm input x, these are the exact two possible
            # normalization factors / normalized RMS values for eps=1e-6 and
            # eps=1e-5. They make the epsilon-only effect directly comparable
            # even when later layers follow different trajectories.
            gain_1e6 = torch.rsqrt(m + self.EPSILON_1E6)
            gain_1e5 = torch.rsqrt(m + self.EPSILON_1E5)
            norm_rms_1e6 = torch.sqrt(m / (m + self.EPSILON_1E6))
            norm_rms_1e5 = torch.sqrt(m / (m + self.EPSILON_1E5))
            active_gain = torch.rsqrt(m + active_eps)
            active_norm_rms = torch.sqrt(m / (m + active_eps))

            self.records.append(
                {
                    "module": name,
                    "token": "batch0_last_prefill_query",
                    "active_eps": active_eps,
                    "input_mean_x2": self._scalar(m),
                    "input_rms": self._scalar(torch.sqrt(m)),
                    "active_normalization_gain": self._scalar(active_gain),
                    "active_normalized_rms_before_gamma": self._scalar(
                        active_norm_rms
                    ),
                    "output_rms_after_gamma": self._scalar(
                        y.square().mean().sqrt()
                    ),
                    "epsilon_over_input_mean_x2": self._scalar(
                        torch.as_tensor(active_eps, device=m.device) /
                        m.clamp_min(torch.finfo(m.dtype).tiny)
                    ),
                    "counterfactual_same_input": {
                        "eps_1e-6_gain": self._scalar(gain_1e6),
                        "eps_1e-5_gain": self._scalar(gain_1e5),
                        "gain_ratio_1e-6_over_1e-5": self._scalar(
                            gain_1e6 / gain_1e5
                        ),
                        "eps_1e-6_normalized_rms_before_gamma": self._scalar(
                            norm_rms_1e6
                        ),
                        "eps_1e-5_normalized_rms_before_gamma": self._scalar(
                            norm_rms_1e5
                        ),
                    },
                }
            )
            self._seen_names.add(name)
            return None

        return hook

    def begin(self, label: str, weight: float) -> None:
        self.active = True
        self.label = str(label)
        self.weight = float(weight)
        self.records = []
        self._seen_names = set()

    def finish(self) -> Dict[str, Any]:
        trace = {
            "label": self.label,
            "requested_weight": self.weight,
            "token": "batch0_last_prefill_query",
            "records": self.records,
        }
        self.active = False
        return trace

    @staticmethod
    def print_trace(trace: Dict[str, Any]) -> None:
        records = trace.get("records", [])
        if not records:
            print("[RMSNorm trace] No initial multimodal prefill was captured.")
            return

        print("\\n" + "=" * 156)
        print(
            "RMSNorm trace | "
            f"{trace['label']} | requested_weight={trace['requested_weight']:g}"
        )
        print(
            "Token: batch=0, last token of the initial multimodal prefill. "
            "norm@eps is the RMS after x/sqrt(mean(x^2)+eps), before gamma."
        )
        print(
            f"{'module':<54} {'m=mean(x^2)':>13} {'norm@1e-6':>11} "
            f"{'norm@1e-5':>11} {'g1e-6/g1e-5':>14} {'active_gain':>12} "
            f"{'out_rms':>11}"
        )
        print("-" * 156)
        for record in records:
            cf = record["counterfactual_same_input"]
            print(
                f"{record['module']:<54} "
                f"{record['input_mean_x2']:>13.3e} "
                f"{cf['eps_1e-6_normalized_rms_before_gamma']:>11.6f} "
                f"{cf['eps_1e-5_normalized_rms_before_gamma']:>11.6f} "
                f"{cf['gain_ratio_1e-6_over_1e-5']:>14.6f} "
                f"{record['active_normalization_gain']:>12.3e} "
                f"{record['output_rms_after_gamma']:>11.3e}"
            )
        print("=" * 156)


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
                        f"{tuple(attn_output.shape)}; expected {expected_shape}."
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
    checkpoint_rms_norm_eps = float(llava_config.text_config.rms_norm_eps)
    print("Checkpoint text_config.rms_norm_eps:", checkpoint_rms_norm_eps)
    print("Requested RMSNorm epsilon:", float(args.rms_norm_eps))

    # The HF eager implementation is required only for pre-softmax AdaptVis.
    # Both epsilon groups use the same attention implementation.
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

    rmsnorm_count, rmsnorm_eps_before = set_llama_rmsnorm_epsilon(
        model,
        args.rms_norm_eps,
    )
    model._rmsnorm_ablation_count = rmsnorm_count
    model._rmsnorm_ablation_before = rmsnorm_eps_before
    model._rmsnorm_checkpoint_eps = checkpoint_rms_norm_eps

    model.eval()
    model.to(args.device)

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

    # This is deliberately installed only under --trace-rmsnorm. The hooks are
    # observational: they return None and leave RMSNorm outputs unchanged.
    if args.trace_rmsnorm:
        model._rmsnorm_tracer = RMSNormTracer(model.language_model, controller)

    print("Active model classes:")
    print(f"  LLaVA: {type(model).__module__}.{type(model).__name__}")
    print(
        "  language_model: "
        f"{type(model.language_model).__module__}."
        f"{type(model.language_model).__name__}"
    )
    print(f"  language text config: {type(model.language_model.config).__name__}")
    print(
        "  attention implementation: "
        f"{getattr(model.config, '_attn_implementation', 'checkpoint default')}"
    )
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
    trace_rmsnorm: bool = False,
    trace_label: str = "",
) -> Tuple[Any, GenerationDiagnostics]:
    controller.begin_generation(weight)

    tracer = getattr(model, "_rmsnorm_tracer", None)
    if trace_rmsnorm and tracer is not None:
        tracer.begin(label=trace_label, weight=weight)

    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )
    diagnostics = controller.finish_generation()

    if trace_rmsnorm and tracer is not None:
        diagnostics.rmsnorm_trace = tracer.finish()

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
            trace_this_sample = bool(
                args.trace_rmsnorm and sample_index < args.print_first
            )

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
                    trace_rmsnorm=trace_this_sample,
                    trace_label=f"sid{sample_index}_probe_weight_1.0",
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
                trace_rmsnorm=trace_this_sample,
                trace_label=(
                    f"sid{sample_index}_final_weight_{selected_weight:g}"
                ),
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
                if probe_diag is not None and probe_diag.rmsnorm_trace is not None:
                    RMSNormTracer.print_trace(probe_diag.rmsnorm_trace)
                if final_diag.rmsnorm_trace is not None:
                    RMSNormTracer.print_trace(final_diag.rmsnorm_trace)

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
        "checkpoint_rms_norm_eps": getattr(
            model,
            "_rmsnorm_checkpoint_eps",
            None,
        ),
        "requested_rms_norm_eps": float(args.rms_norm_eps),
        "rmsnorm_module_count": getattr(
            model,
            "_rmsnorm_ablation_count",
            None,
        ),
        "rmsnorm_eps_before_override": getattr(
            model,
            "_rmsnorm_ablation_before",
            None,
        ),
        "active_rms_norm_eps": float(
            model.language_model.config.rms_norm_eps
        ),
        "method": args.method,
        "weight": args.weight,
        "weight1": args.weight1,
        "weight2": args.weight2,
        "threshold": args.threshold,
        "confidence_round_decimals": 2,
        "trace_rmsnorm": bool(args.trace_rmsnorm),
        "trace_rmsnorm_samples": min(
            int(args.print_first),
            int(sample_index),
        ) if args.trace_rmsnorm else 0,
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
        f"rms_norm_eps={args.rms_norm_eps:g} "
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

    eps_suffix = f"{args.rms_norm_eps:.0e}".replace("-", "m")
    return Path(
        "output/"
        f"hf_llava15_{args.dataset}_"
        f"rms_eps_{eps_suffix}_{method_suffix}.json"
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
