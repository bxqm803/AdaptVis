# coding=utf-8
"""
HF-structured LLaMAForCausalLMScal adapter for AdaptVis.

Goal:
  Keep the language model structurally aligned with transformers.LlamaForCausalLM,
  but make it usable by AdaptVis' LlavaForConditionalGenerationScal wrapper.

What this module provides:
  - class LLaMAForCausalLMScal
  - class LLaMAForCausalLM as alias
  - _from_config(config) compatible with the original AdaptVis wrapper call

Important:
  This is not the original AdaptVis custom LLaMA implementation.
  It is a controlled diagnostic adapter:
    original LLaVA wrapper + original generation path + HF-style LLaMA.

Main compatibility additions:
  1. forward accepts AdaptVis arguments:
       keys, weight, pos, adjust_method, caption_length
  2. LLaMA attention applies AdaptVis raw-logit intervention before attention_mask/softmax.
  3. returned legacy past_key_values are sanitized slightly to avoid the original wrapper's
     zero-cache heuristic misreading real HF cached keys as empty slots.

Put this file at:
  model_zoo/llama_hf_scal.py

Then in model_zoo/llava/modeling_llava_scal.py change only:
  from model_zoo import llama
to:
  from model_zoo import llama_hf_scal as llama
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaForCausalLM as HFLlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def _to_hf_llama_config(config):
    """Map AdaptVis local LLaMAConfig or HF LlamaConfig to transformers.LlamaConfig."""
    if isinstance(config, LlamaConfig):
        hf_config = config
    else:
        attrs = {
            "vocab_size": getattr(config, "vocab_size", 32064),
            "hidden_size": getattr(config, "hidden_size", 4096),
            "intermediate_size": getattr(config, "intermediate_size", 11008),
            "num_hidden_layers": getattr(config, "num_hidden_layers", 32),
            "num_attention_heads": getattr(config, "num_attention_heads", 32),
            "hidden_act": getattr(config, "hidden_act", "silu"),
            "max_position_embeddings": getattr(config, "max_position_embeddings", 2048),
            "initializer_range": getattr(config, "initializer_range", 0.02),
            "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
            "use_cache": getattr(config, "use_cache", True),
            "pad_token_id": getattr(config, "pad_token_id", None),
            "bos_token_id": getattr(config, "bos_token_id", 0),
            "eos_token_id": getattr(config, "eos_token_id", 1),
            "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
        }

        # The original LLaMAConfig has no GQA. For LLaVA-1.5 7B, KV heads = attention heads.
        attrs["num_key_value_heads"] = getattr(
            config,
            "num_key_value_heads",
            attrs["num_attention_heads"],
        )

        hf_config = LlamaConfig(**attrs)

    # Avoid SDPA/flash paths so our patched eager attention is actually used.
    try:
        hf_config._attn_implementation = "eager"
    except Exception:
        pass
    return hf_config


def _normalize_keys_to_image_mask(keys, bsz: int, kv_len: int, device):
    """Convert AdaptVis keys/image_id into a boolean image-token mask of shape [B, kv_len]."""
    if keys is None:
        return None

    if isinstance(keys, (list, tuple)):
        # Original generate sometimes passes a list before LLaVA merge.
        keys = torch.stack([k.to(device) for k in keys], dim=0)
    else:
        keys = keys.to(device)

    if keys.dim() == 1:
        keys = keys.unsqueeze(0)

    keys = keys.bool()

    if keys.shape[0] == 1 and bsz > 1:
        keys = keys.expand(bsz, -1)

    # In the full prompt forward, native wrapper passes merged image_id [B, merged_len].
    if keys.shape[-1] == kv_len:
        return keys[:, :kv_len]

    # If not aligned, skip intervention rather than guessing.
    return None


def _get_query_indices(q_len: int, caption_length=None, device=None):
    """
    Approximate original AdaptVis behavior:
    - if caption_length is None: apply to all query rows
    - if caption_length is given: apply to the last caption_length query rows
    """
    if caption_length is None:
        return torch.arange(q_len, device=device)

    if isinstance(caption_length, (list, tuple)):
        try:
            c = int(caption_length[0])
        except Exception:
            c = q_len
    else:
        try:
            c = int(caption_length)
        except Exception:
            c = q_len

    c = max(1, min(q_len, c))
    return torch.arange(q_len - c, q_len, device=device)


def _apply_adaptvis_pre_softmax(
    attn_logits,
    *,
    image_key_mask,
    query_indices,
    weight: float,
    adjust_method: Optional[str] = None,
    pos=None,
):
    """
    Apply AdaptVis-style image-key logit scaling.

    Default method in AdaptVis is effectively multiplying selected image-key logits.
    For this diagnostic adapter we keep the core behavior.
    """
    if image_key_mask is None or weight is None:
        return attn_logits

    try:
        w = float(weight)
    except Exception:
        return attn_logits

    if w == 1.0:
        return attn_logits

    bsz, n_heads, q_len, kv_len = attn_logits.shape
    if query_indices is None or query_indices.numel() == 0:
        return attn_logits

    out = attn_logits.clone()
    for b in range(min(bsz, image_key_mask.shape[0])):
        img_pos = torch.nonzero(image_key_mask[b], as_tuple=False).view(-1)
        img_pos = img_pos[(img_pos >= 0) & (img_pos < kv_len)]
        if img_pos.numel() == 0:
            continue

        # out[b, :, query_indices, img_pos] with advanced indexing needs this shape.
        q_idx = query_indices[(query_indices >= 0) & (query_indices < q_len)]
        if q_idx.numel() == 0:
            continue

        out[b, :, q_idx[:, None], img_pos[None, :]] *= w

    return out


def _sanitize_legacy_past_key_values(past_key_values, eps: float = 1e-6):
    """
    AdaptVis original LLaVA wrapper has a cache branch that uses:
        first_layer_past_key_value = past_key_values[0][0][:, 0, :, 0]
        torch.where(first_layer_past_key_value == 0)
    as a heuristic for non-attended / empty cache positions.

    HF LLaMA real cached keys may contain exact zeros in that inspected channel.
    That causes the wrapper to mask valid tokens or index out of bounds.

    To keep the original wrapper unchanged while using HF-style LLaMA, make the inspected
    channel non-zero wherever it is exactly zero. The perturbation is tiny and only applied
    to the legacy tuple cache returned to generation.
    """
    if past_key_values is None:
        return past_key_values

    # Skip new Cache objects. Native wrapper can handle Cache separately in prepare_inputs,
    # but its forward cache branch directly indexes tuple-like caches.
    if isinstance(past_key_values, Cache):
        return past_key_values

    try:
        pkv = list(past_key_values)
        if len(pkv) == 0:
            return past_key_values

        first = list(pkv[0])
        key = first[0]
        if not torch.is_tensor(key) or key.dim() < 4:
            return past_key_values

        # Clone only the first-layer key tensor.
        key2 = key.clone()
        marker = key2[:, 0, :, 0]
        zero = marker == 0
        if bool(zero.any().detach().cpu()):
            marker = marker.masked_fill(zero, eps)
            key2[:, 0, :, 0] = marker

        first[0] = key2
        pkv[0] = tuple(first)
        return tuple(tuple(layer) for layer in pkv)
    except Exception:
        return past_key_values


class LLaMAForCausalLMScal(HFLlamaForCausalLM):
    """
    HF LlamaForCausalLM with AdaptVis-compatible forward signature and attention patch.
    """

    def __init__(self, config):
        hf_config = _to_hf_llama_config(config)
        super().__init__(hf_config)

        self._adaptvis_keys = None
        self._adaptvis_weight = None
        self._adaptvis_pos = None
        self._adaptvis_caption_length = None
        self._adaptvis_adjust_method = None
        self._adaptvis_max_layers = 32
        self._adaptvis_require_square = True
        self._adaptvis_sanitize_cache = True

        self._patch_llama_attention_modules()

    @classmethod
    def _from_config(cls, config, **kwargs):
        # Match PreTrainedModel._from_config call style used by AdaptVis wrapper.
        return cls(config)

    def _patch_llama_attention_modules(self):
        layers = self.model.layers

        def make_forward(attn_module, layer_idx: int):
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

                query_states = attn_module.q_proj(hidden_states)
                key_states = attn_module.k_proj(hidden_states)
                value_states = attn_module.v_proj(hidden_states)

                num_heads = getattr(attn_module, "num_heads")
                head_dim = getattr(attn_module, "head_dim")
                num_kv_heads = getattr(attn_module, "num_key_value_heads", num_heads)
                num_kv_groups = getattr(
                    attn_module,
                    "num_key_value_groups",
                    num_heads // num_kv_heads,
                )

                query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

                kv_seq_len = key_states.shape[-2]
                if past_key_value is not None:
                    try:
                        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, getattr(attn_module, "layer_idx", layer_idx))
                    except Exception:
                        # Legacy tuple cache path in older transformers.
                        try:
                            kv_seq_len += past_key_value[0].shape[-2]
                        except Exception:
                            pass

                # Robust RoPE handling across transformers versions.
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    try:
                        query_states, key_states = apply_rotary_pos_emb(
                            query_states, key_states, cos, sin, position_ids
                        )
                    except TypeError:
                        query_states, key_states = apply_rotary_pos_emb(
                            query_states, key_states, cos, sin
                        )
                else:
                    try:
                        cos, sin = attn_module.rotary_emb(value_states, seq_len=kv_seq_len)
                        query_states, key_states = apply_rotary_pos_emb(
                            query_states, key_states, cos, sin, position_ids
                        )
                    except TypeError:
                        cos, sin = attn_module.rotary_emb(value_states, position_ids)
                        try:
                            query_states, key_states = apply_rotary_pos_emb(
                                query_states, key_states, cos, sin
                            )
                        except TypeError:
                            query_states, key_states = apply_rotary_pos_emb(
                                query_states, key_states, cos, sin, position_ids
                            )

                if past_key_value is not None:
                    # DynamicCache/new cache object.
                    if hasattr(past_key_value, "update"):
                        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                        key_states, value_states = past_key_value.update(
                            key_states,
                            value_states,
                            getattr(attn_module, "layer_idx", layer_idx),
                            cache_kwargs,
                        )
                    else:
                        # Legacy tuple cache.
                        key_states = torch.cat([past_key_value[0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[1], value_states], dim=2)

                past = (key_states, value_states) if use_cache else None

                key_states = repeat_kv(key_states, num_kv_groups)
                value_states = repeat_kv(value_states, num_kv_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

                # AdaptVis raw-logit intervention, before attention_mask.
                keys = self._adaptvis_keys
                weight = self._adaptvis_weight
                if (
                    layer_idx < self._adaptvis_max_layers
                    and keys is not None
                    and weight is not None
                    and (not self._adaptvis_require_square or q_len == key_states.shape[-2])
                ):
                    image_mask = _normalize_keys_to_image_mask(
                        keys,
                        bsz=bsz,
                        kv_len=key_states.shape[-2],
                        device=attn_weights.device,
                    )
                    q_idx = _get_query_indices(
                        q_len,
                        self._adaptvis_caption_length,
                        device=attn_weights.device,
                    )
                    attn_weights = _apply_adaptvis_pre_softmax(
                        attn_weights,
                        image_key_mask=image_mask,
                        query_indices=q_idx,
                        weight=weight,
                        adjust_method=self._adaptvis_adjust_method,
                        pos=self._adaptvis_pos,
                    )

                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask

                attn_weights = torch.clamp(attn_weights, min=torch.finfo(attn_weights.dtype).min)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, (attn_weights if output_attentions else None), past

            return forward

        for idx, layer in enumerate(layers):
            layer.self_attn.forward = make_forward(layer.self_attn, idx)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        keys: Optional[torch.Tensor] = None,
        weight: Optional[float] = None,
        pos: Optional[torch.Tensor] = None,
        caption_length: Optional[list] = None,
        adjust_method: Optional[str] = None,
        **kwargs,
    ):
        self._adaptvis_keys = keys
        self._adaptvis_weight = weight
        self._adaptvis_pos = pos
        self._adaptvis_caption_length = caption_length
        self._adaptvis_adjust_method = adjust_method

        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        pkv = out.past_key_values
        if self._adaptvis_sanitize_cache:
            pkv = _sanitize_legacy_past_key_values(pkv)

        if return_dict is False:
            # logits first, to match the original AdaptVis wrapper usage outputs[0].
            items = (out.logits,)
            if pkv is not None:
                items += (pkv,)
            if output_hidden_states:
                items += (out.hidden_states,)
            if output_attentions:
                items += (out.attentions,)
            return items

        return CausalLMOutputWithPast(
            loss=out.loss,
            logits=out.logits,
            past_key_values=pkv,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


# Alias for compatibility with comments/imports.
LLaMAForCausalLM = LLaMAForCausalLMScal
