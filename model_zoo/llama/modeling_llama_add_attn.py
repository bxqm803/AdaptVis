# coding=utf-8
"""PyTorch LLaMA model with main-style AdaptVis / ScalingVis intervention."""

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

# Saving attention is disabled by default to avoid requiring SAVE_ATTN_PATH.
# Set SAVE_ATTN=True SAVE_ORI=True SAVE_ATTN_PATH=... to enable.
SAVE_ATTN = os.getenv("SAVE_ATTN", "False").strip().lower() in ["1", "true", "yes", "y", "on"]
SAVE_ORI = os.getenv("SAVE_ORI", "False").strip().lower() in ["1", "true", "yes", "y", "on"]


# ============================================================
# Helpers
# ============================================================

def _parse_layer_set_from_env(name: str):
    """
    Parse comma-separated layer ids from env.

    Examples:
        ADAPTVIS_EXCLUDE_LAYERS=15,16,17,18
        ADAPTVIS_INCLUDE_LAYERS=12,13,14,19,20,21,22,23,24
    """
    s = os.getenv(name, "").strip()
    if not s:
        return set()

    out = set()
    for x in s.split(","):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


def _adaptvis_layer_allowed(layer_idx):
    """
    Decide whether AdaptVis / ScalingVis should apply on this layer.

    Env:
        ADAPTVIS_INCLUDE_LAYERS=12,13,14,19,20
            If non-empty, only apply AdaptVis on these layers.

        ADAPTVIS_EXCLUDE_LAYERS=15,16,17,18
            Skip these layers.

    INCLUDE is applied first, then EXCLUDE.
    """
    if layer_idx is None:
        return True

    include_layers = _parse_layer_set_from_env("ADAPTVIS_INCLUDE_LAYERS")
    exclude_layers = _parse_layer_set_from_env("ADAPTVIS_EXCLUDE_LAYERS")

    layer_idx = int(layer_idx)

    if include_layers and layer_idx not in include_layers:
        return False

    if layer_idx in exclude_layers:
        return False

    return True


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for self-attention.
    """
    bsz, tgt_len = input_ids_shape

    mask = torch.full(
        (tgt_len, tgt_len),
        torch.finfo(dtype).min,
        dtype=dtype,
    )

    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(
        mask_cond < (mask_cond + 1).view(mask.size(-1), 1),
        0,
    )

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype),
                mask,
            ],
            dim=-1,
        )

    return mask[None, None, :, :].expand(
        bsz,
        1,
        tgt_len,
        tgt_len + past_key_values_length,
    )


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
):
    """
    Expands attention_mask from [bsz, seq_len] to
    [bsz, 1, tgt_seq_len, src_seq_len].
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(
        bsz,
        1,
        tgt_len,
        src_len,
    ).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min,
    )


# ============================================================
# Modules
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(
            -1,
            keepdim=True,
        )
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float().to(device) / dim
            )
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
            self.cos_cached[:, :, :seq_len, ...].to(
                dtype=x.dtype,
                device=x.device,
            ),
            self.sin_cached[:, :, :seq_len, ...].to(
                dtype=x.dtype,
                device=x.device,
            ),
        )


def rotate_half(x):
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
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )


class LLaMAAttention(nn.Module):
    """
    Multi-headed attention with main-style AdaptVis / ScalingVis intervention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        oproj_bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={num_heads})."
            )

        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )

        self.att_out = nn.Identity()
        self.value_out = nn.Identity()
        self.head_out = nn.Identity()

        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=oproj_bias,
        )

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
        object_patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Main-style AdaptVis attention.

        Important behavior:
            - AdaptVis / ScalingVis is applied only when q_len == kv_seq_len.
            - This corresponds to the prefill / square-attention stage.
            - Autoregressive decode steps usually have q_len=1, kv_seq_len>1,
              so they are skipped.
            - probe_bias / probe_scale / probe_add / var_sink / object_mask are
              intentionally removed in this clean main-style version.
        """

        if adjust_method is None:
            env_adjust_method = os.getenv("ADJUST_METHOD", "").strip()
            if env_adjust_method:
                adjust_method = env_adjust_method

        if weight is None:
            weight = 1.0

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
            key_states = torch.cat(
                [past_key_value[0], key_states],
                dim=2,
            )
            value_states = torch.cat(
                [past_key_value[1], value_states],
                dim=2,
            )

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

        before_logits_raw = attn_weights.clone()

        start_idx, end_idx = -1, -1
        valid_image_tokens = False

        save_layers_env = os.getenv("SAVE_LAYERS", "")

        if save_layers_env.strip():
            save_layers = {
                int(x.strip())
                for x in save_layers_env.split(",")
                if x.strip()
            }
            should_save_layer = idx in save_layers
        else:
            should_save_layer = True

        # ====================================================
        # Main-style AdaptVis / ScalingVis
        # ====================================================
        #
        # Original-main style:
        #   only apply when q_len == kv_seq_len.
        #
        # This skips autoregressive decode steps:
        #   q_len = 1
        #   kv_seq_len = prompt_len + generated_len
        # ====================================================

        if (
            idx is not None
            and idx < 32
            and keys is not None
            and q_len == kv_seq_len
            and weight is not None
        ):
            if isinstance(keys, (list, tuple)):
                key_mask = keys[0]
            else:
                key_mask = keys

            if key_mask.dim() == 2:
                key_mask = key_mask[0]

            key_mask = key_mask.to(attn_weights.device).bool()
            true_indices = torch.where(key_mask)[0]

            if true_indices.numel() == 0:
                if os.getenv("ADAPTVIS_LAYER_DEBUG", "False") == "True":
                    print(f"[Layer {idx}] No True values found in keys.")
            else:
                start_idx = int(true_indices[0].item())
                end_idx = int(true_indices[-1].item())

                start_idx = max(0, min(start_idx, kv_seq_len - 1))
                end_idx = max(0, min(end_idx, kv_seq_len - 1))

                if end_idx >= start_idx:
                    valid_image_tokens = True

        if valid_image_tokens:
            mask = torch.zeros(
                (q_len, kv_seq_len),
                dtype=torch.bool,
                device=attn_weights.device,
            )

            if adjust_method is None or adjust_method == "last_query":
                mask[-1, start_idx:end_idx + 1] = True

            elif adjust_method == "caption_query" and caption_length:
                n_caption = len(caption_length[0])
                n_caption = min(n_caption, q_len)
                mask[-n_caption:, start_idx:end_idx + 1] = True

            elif adjust_method == "all_query":
                mask[:, start_idx:end_idx + 1] = True

            elif adjust_method == "full":
                mask[:, :] = True

            else:
                # For clean main-style reproduction, unsupported adjust methods
                # fall back to last_query instead of introducing new behavior.
                mask[-1, start_idx:end_idx + 1] = True

            if _adaptvis_layer_allowed(idx):
                attn_weights[:, :, mask] *= weight

                if os.getenv("ADAPTVIS_LAYER_DEBUG", "False") == "True":
                    print(
                        f"[ADAPTVIS_APPLY_MAIN_STYLE] layer={idx}, "
                        f"q_len={q_len}, kv_seq_len={kv_seq_len}, "
                        f"adjust_method={adjust_method}, weight={weight}, "
                        f"start={start_idx}, end={end_idx}, "
                        f"mask_tokens={int(mask.sum().item())}"
                    )
            else:
                if os.getenv("ADAPTVIS_LAYER_DEBUG", "False") == "True":
                    print(
                        f"[ADAPTVIS_SKIP_LAYER] layer={idx}, "
                        f"adjust_method={adjust_method}, weight={weight}, "
                        f"exclude={os.getenv('ADAPTVIS_EXCLUDE_LAYERS', '')}, "
                        f"include={os.getenv('ADAPTVIS_INCLUDE_LAYERS', '')}"
                    )

        else:
            if (
                os.getenv("ADAPTVIS_LAYER_DEBUG", "False") == "True"
                and idx is not None
                and idx == 0
            ):
                print(
                    f"[ADAPTVIS_SKIP_MAIN_STYLE] layer={idx}, "
                    f"q_len={q_len}, kv_seq_len={kv_seq_len}, "
                    f"keys_is_none={keys is None}, weight={weight}"
                )

        after_logits_raw = attn_weights.clone()

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size "
                    f"{(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}."
                )

            before_logits = before_logits_raw + attention_mask
            after_logits = after_logits_raw + attention_mask

            before_logits = torch.max(
                before_logits,
                torch.tensor(
                    torch.finfo(before_logits.dtype).min,
                    device=before_logits.device,
                    dtype=before_logits.dtype,
                ),
            )
            after_logits = torch.max(
                after_logits,
                torch.tensor(
                    torch.finfo(after_logits.dtype).min,
                    device=after_logits.device,
                    dtype=after_logits.dtype,
                ),
            )
        else:
            before_logits = before_logits_raw
            after_logits = after_logits_raw

        before_probs = nn.functional.softmax(
            before_logits,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        after_probs = nn.functional.softmax(
            after_logits,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        if SAVE_ATTN and SAVE_ORI and should_save_layer and valid_image_tokens:
            save_path = os.getenv("SAVE_ATTN_PATH")

            if not save_path:
                raise ValueError("SAVE_ATTN_PATH not set.")

            os.makedirs(save_path, exist_ok=True)

            np.save(
                f"{save_path}before_logits_layer{idx}_start{start_idx}_end{end_idx}.npy",
                before_logits[:, :, -1, :].detach().float().cpu().numpy(),
            )
            np.save(
                f"{save_path}after_logits_layer{idx}_start{start_idx}_end{end_idx}.npy",
                after_logits[:, :, -1, :].detach().float().cpu().numpy(),
            )
            np.save(
                f"{save_path}before_probs_layer{idx}_start{start_idx}_end{end_idx}.npy",
                before_probs[:, :, -1, :].detach().float().cpu().numpy(),
            )
            np.save(
                f"{save_path}after_probs_layer{idx}_start{start_idx}_end{end_idx}.npy",
                after_probs[:, :, -1, :].detach().float().cpu().numpy(),
            )

        attn_weights = after_probs
        attn_weights = self.att_out(attn_weights)
        value_states = self.value_out(value_states)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.num_heads,
            q_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}."
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(
            bsz,
            q_len,
            self.hidden_size,
        )

        attn_output = self.head_out(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
