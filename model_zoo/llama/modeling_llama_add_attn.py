# coding=utf-8
"""PyTorch LLaMA model with AdaptVis attention intervention variants."""

import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, logging

from .configuration_llama import LLaMAConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "llama-7b"
_CONFIG_FOR_DOC = "LLaMAConfig"


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).lower() in ["1", "true", "yes", "y", "on"]


# 默认关闭，避免没设置 SAVE_ATTN_PATH 时直接报错。
# 需要保存 attention 时：
#   export SAVE_ATTN=True
#   export SAVE_ATTN_PATH=/path/to/save/
SAVE_ATTN = _env_bool("SAVE_ATTN", False)
SAVE_ORI = _env_bool("SAVE_ORI", True)


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for causal self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
            dim=-1,
        )

    return mask[None, None, :, :].expand(
        bsz,
        1,
        tgt_len,
        tgt_len + past_key_values_length,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len].
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min,
    )


def _adaptvis_variant_name(adjust_method: Optional[str] = None) -> str:
    """
    Supported variants:
      mul_img    : original AdaptVis, pre-softmax s_img = weight * s_img
      add_img    : pre-softmax group suppression, s_img = s_img + weight
      center_img : pre-softmax centered scaling, s_img = mean + weight * (s_img - mean)
      prob_img   : post-softmax p_img = weight * p_img, then renormalize
    """
    supported = {"mul_img", "add_img", "center_img", "prob_img"}

    env_variant = os.environ.get("ADAPTVIS_ATTENTION_VARIANT", "").strip()
    if env_variant:
        if env_variant not in supported:
            raise ValueError(
                f"Unknown ADAPTVIS_ATTENTION_VARIANT={env_variant}. "
                f"Supported: {sorted(supported)}"
            )
        return env_variant

    if adjust_method in supported:
        return adjust_method

    return "mul_img"


def _normalize_keys_to_image_mask(
    keys: Optional[torch.Tensor],
    bsz: int,
    kv_seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Convert keys/image-token mask to bool mask with shape [bsz, kv_seq_len].
    """
    if keys is None or not torch.is_tensor(keys):
        return None

    mask = keys

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    elif mask.dim() > 2:
        mask = mask.view(mask.shape[0], -1)

    if mask.shape[-1] != kv_seq_len:
        return None

    if mask.shape[0] == 1 and bsz > 1:
        mask = mask.expand(bsz, -1)

    if mask.shape[0] != bsz:
        return None

    return mask.to(device=device).bool()


def _get_query_indices(q_len: int, caption_length: Optional[list], device: torch.device) -> torch.Tensor:
    """
    Keep original behavior:
      - no caption_length: edit only last query row
      - with caption_length: edit last len(caption_length[0]) query rows
    """
    if caption_length:
        try:
            first = caption_length[0]
            if isinstance(first, (list, tuple)):
                n_query = len(first)
            else:
                n_query = int(first)
        except Exception:
            n_query = 1

        n_query = max(1, min(int(n_query), q_len))
        return torch.arange(q_len - n_query, q_len, device=device, dtype=torch.long)

    return torch.tensor([q_len - 1], device=device, dtype=torch.long)


def _apply_adaptvis_pre_softmax_variant(
    attn_logits: torch.Tensor,
    image_key_mask: Optional[torch.Tensor],
    query_indices: torch.Tensor,
    weight: Optional[float],
    adjust_method: Optional[str] = None,
) -> torch.Tensor:
    """
    Apply image-token intervention before attention softmax.

    attn_logits: [bsz, num_heads, q_len, kv_len]
    image_key_mask: [bsz, kv_len], True for image tokens

    Variants:
      mul_img:
        s_img_new = alpha * s_img
      add_img:
        s_img_new = s_img + beta
        Use beta < 0 for group suppression, e.g. beta=log(0.5)=-0.69314718056.
      center_img:
        s_img_new = mean(s_img) + alpha * (s_img - mean(s_img))
      prob_img:
        no-op here; handled post-softmax.
    """
    variant = _adaptvis_variant_name(adjust_method)

    if variant == "prob_img":
        return attn_logits

    if image_key_mask is None:
        return attn_logits

    if weight is None:
        weight = 1.0

    weight = float(weight)
    bsz = attn_logits.shape[0]

    for b in range(bsz):
        img_idx = image_key_mask[b].to(attn_logits.device).bool()
        if img_idx.sum().item() == 0:
            continue

        for q in query_indices:
            q_int = int(q.item())
            s_img = attn_logits[b, :, q_int, img_idx]  # [num_heads, num_img_tokens]

            if variant == "mul_img":
                # Method 1: original AdaptVis
                s_img_new = weight * s_img

            elif variant == "add_img":
                # Method 2: group suppression of image logits
                s_img_new = s_img + weight

            elif variant == "center_img":
                # Method 3: reduce image-internal extremeness while preserving image mean
                mu = s_img.mean(dim=-1, keepdim=True)
                s_img_new = mu + weight * (s_img - mu)

            else:
                raise ValueError(
                    f"Unknown AdaptVis variant={variant}. "
                    f"Use mul_img/add_img/center_img/prob_img."
                )

            attn_logits[b, :, q_int, img_idx] = s_img_new

    return attn_logits


def _apply_adaptvis_post_softmax_variant(
    attn_probs: torch.Tensor,
    image_key_mask: Optional[torch.Tensor],
    query_indices: torch.Tensor,
    weight: Optional[float],
    adjust_method: Optional[str] = None,
) -> torch.Tensor:
    """
    Apply post-softmax image probability intervention.

    prob_img:
      p_img_new = gamma * p_img, then normalize the full key row.
    This is equivalent to pre-softmax s_img += log(gamma),
    but NOT equivalent to original s_img *= alpha.
    """
    variant = _adaptvis_variant_name(adjust_method)

    if variant != "prob_img":
        return attn_probs

    if image_key_mask is None:
        return attn_probs

    if weight is None:
        weight = 1.0

    gamma = float(weight)
    bsz = attn_probs.shape[0]

    for b in range(bsz):
        img_idx = image_key_mask[b].to(attn_probs.device).bool()
        if img_idx.sum().item() == 0:
            continue

        for q in query_indices:
            q_int = int(q.item())
            attn_probs[b, :, q_int, img_idx] = gamma * attn_probs[b, :, q_int, img_idx]

            denom = attn_probs[b, :, q_int, :].sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attn_probs[b, :, q_int, :] = attn_probs[b, :, q_int, :] / denom

    return attn_probs


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached,
                device=x.device,
                dtype=self.inv_freq.dtype,
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :].to(dtype=x.dtype)
            self.sin_cached = emb.sin()[None, None, :, :].to(dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LLaMAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaMAAttention(nn.Module):
    """
    Multi-headed causal self-attention.
    """

    def __init__(self, hidden_size: int, num_heads: int, oproj_bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={num_heads})."
            )

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        self.att_out = nn.Identity()
        self.value_out = nn.Identity()
        self.head_out = nn.Identity()

        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=oproj_bias)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(
            bsz,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_head_hidden_states: bool = False,
        keys: Optional[torch.Tensor] = None,
        weight: Optional[float] = None,
        pos: Optional[torch.Tensor] = None,
        idx: Optional[int] = None,
        caption_length: Optional[list] = None,
        adjust_method: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Input shape: [batch, time, channel]
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        offset = 0

        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            offset=offset,
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"but is {attn_weights.size()}."
            )

        unchanged_attn_weights = attn_weights.clone()

        # ------------------------------------------------------------------
        # AdaptVis intervention variants.
        #
        # Scope is kept close to original:
        #   - only layers idx < 32
        #   - only full prompt forward where q_len == kv_seq_len
        #   - normally only last query row
        #
        # Select variant with:
        #   export ADAPTVIS_ATTENTION_VARIANT=mul_img/add_img/center_img/prob_img
        #
        # weight meaning:
        #   mul_img    : alpha, e.g. 0.5
        #   add_img    : beta, e.g. -0.69314718056
        #   center_img : alpha, e.g. 0.5
        #   prob_img   : gamma, e.g. 0.5
        # ------------------------------------------------------------------
        start_idx, end_idx, square_size = -1, -1, -1
        image_key_mask = None
        query_indices = None

        if idx is not None and idx < 32 and keys is not None and weight is not None:
            if attn_weights.size(2) == attn_weights.size(3):
                image_key_mask = _normalize_keys_to_image_mask(
                    keys=keys,
                    bsz=bsz,
                    kv_seq_len=kv_seq_len,
                    device=attn_weights.device,
                )

                if image_key_mask is None:
                    print("[AdaptVis] image_key_mask is None or shape mismatch; skipping.")
                else:
                    true_indices = torch.where(image_key_mask[0])[0]

                    if len(true_indices) == 0:
                        print("[AdaptVis] No True values found in image_key_mask; skipping.")
                    else:
                        start_idx = true_indices[0].item()
                        end_idx = true_indices[-1].item()
                        square_size = end_idx - start_idx + 1

                        query_indices = _get_query_indices(
                            q_len=q_len,
                            caption_length=caption_length,
                            device=attn_weights.device,
                        )

                        attn_weights = _apply_adaptvis_pre_softmax_variant(
                            attn_logits=attn_weights,
                            image_key_mask=image_key_mask,
                            query_indices=query_indices,
                            weight=weight,
                            adjust_method=adjust_method,
                        )
            else:
                start_idx, end_idx, square_size = -1, -1, -1

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}."
                )

            attn_weights = attn_weights + attention_mask

            min_value = torch.tensor(
                torch.finfo(attn_weights.dtype).min,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            attn_weights = torch.max(attn_weights, min_value)

        attn_weights = nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        if (
            idx is not None
            and idx < 32
            and keys is not None
            and weight is not None
            and image_key_mask is not None
            and query_indices is not None
        ):
            attn_weights = _apply_adaptvis_post_softmax_variant(
                attn_probs=attn_weights,
                image_key_mask=image_key_mask,
                query_indices=query_indices,
                weight=weight,
                adjust_method=adjust_method,
            )

        returned_attn_weights = None

        if SAVE_ATTN or output_attentions:
            unchanged = unchanged_attn_weights

            if attention_mask is not None:
                unchanged = unchanged + attention_mask

                min_value = torch.tensor(
                    torch.finfo(unchanged.dtype).min,
                    dtype=unchanged.dtype,
                    device=unchanged.device,
                )
                unchanged = torch.max(unchanged, min_value)

            if SAVE_ATTN:
                save_path = os.getenv("SAVE_ATTN_PATH")
                if not save_path:
                    raise ValueError("SAVE_ATTN_PATH not set.")

                os.makedirs(save_path, exist_ok=True)

                if SAVE_ORI:
                    ori = unchanged[:, :, -1, :]
                    np.save(
                        f"{save_path}diff_{idx}_start{start_idx}_end{end_idx}.npy",
                        ori.cpu().detach().numpy(),
                    )

            returned_attn_weights = nn.functional.softmax(
                unchanged,
                dim=-1,
                dtype=torch.float32,
            ).to(query_states.dtype)

        attn_weights = self.att_out(attn_weights)
        value_states = self.value_out(value_states)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}."
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.head_out(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            returned_attn_weights = None

        return attn_output, returned_attn_weights, past_key_value


class LLaMADecoderLayer(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLaMAAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            oproj_bias=config.oproj_bias,
        )

        self.mlp = LLaMAMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        keys: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        weight: Optional[float] = None,
        idx: Optional[int] = None,
        caption_length: Optional[list] = None,
        adjust_method: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            keys=keys,
            pos=pos,
            weight=weight,
            idx=idx,
            caption_length=caption_length,
            adjust_method=adjust_method,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = [hidden_states]

        if output_attentions:
            outputs += [self_attn_weights]

        if use_cache:
            outputs += [present_key_value]

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`].
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LLaMAPreTrainedModel(PreTrainedModel):
    config_class = LLaMAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaMADecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LLaMADecoderLayer):
            module.gradient_checkpointing = value


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LLaMAModel(LLaMAPreTrainedModel):
    """
    Transformer decoder consisting of config.num_hidden_layers layers.
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [LLaMADecoderLayer(config) for _ in range(config.num_hidden_layers)]
       
