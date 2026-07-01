"""
HF LLaVA-1.5 text-prefix Q/K/V projection ablation.

This is a mechanism-localization control derived from the validated
run_llava15_hf_token_group_eps_ablation.py path.

All LlamaRMSNorm modules remain at --rms-norm-eps (normally 1e-5).
During only the initial multimodal prefill, this script computes the exact
per-token multiplier that changing text-prefix input RMSNorm epsilon from the
base value to --qkv-target-eps would produce:

    alpha_t = sqrt((mean(x_t^2) + base_eps) / (mean(x_t^2) + target_eps))

where x_t is the input to each decoder layer's input_layernorm. The RMSNorm
output itself is kept unchanged. Instead, alpha_t is injected selectively into
the text-prefix token inputs of q_proj, k_proj, and/or v_proj.

Therefore:
  --qkv-components qkv is an exact local replay of changing text-prefix
  input_layernorm epsilon for the Q/K/V projection inputs, while all actual
  RMSNorm modules still use the base epsilon.
  --qkv-components q, k, or v separates the downstream attention role of the
  same token-wise scaling. These are intentionally artificial controls, not
  native RMSNorm behavior.

Only the prefill path is changed. q_len=1 decode calls retain the native model
and the base RMSNorm epsilon.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
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
        "--target-rms-norm-eps",
        default=None,
        type=float,
        help=(
            "Optional target epsilon used only for --target-token-group during "
            "the initial multimodal prefill. All other tokens and all decode "
            "q_len=1 calls remain at --rms-norm-eps."
        ),
    )
    parser.add_argument(
        "--target-token-group",
        default="last_query",
        choices=[
            "all",
            "image",
            "text",
            "text_prefix",
            "last_query",
            "image_plus_last_query",
            "image_plus_text_prefix",
        ],
        help=(
            "Prefill token group recomputed with --target-rms-norm-eps. "
            "Use all first as a sanity control, then image / text_prefix / "
            "image_plus_last_query to localize the effect."
        ),
    )
    parser.add_argument(
        "--target-layer-range",
        default="0:32",
        help=(
            "Decoder-layer interval [start,end) whose input/post-attention "
            "RMSNorm modules are eligible for the group override. Examples: "
            "0:32, 0:4, 4:8. model.norm is controlled separately by "
            "--target-norm-sites."
        ),
    )
    parser.add_argument(
        "--target-norm-sites",
        default="all",
        help=(
            "Comma-separated RMSNorm sites to target: all, input, post_attn, "
            "final; e.g. input,post_attn or input."
        ),
    )
    parser.add_argument(
        "--qkv-target-eps",
        default=1e-6,
        type=float,
        help=(
            "Target epsilon whose text-prefix input-layer RMSNorm multiplier "
            "is replayed at selected Q/K/V projection inputs. RMSNorm module "
            "epsilons themselves remain at --rms-norm-eps."
        ),
    )
    parser.add_argument(
        "--qkv-components",
        default="qkv",
        help=(
            "Comma-separated subset of q,k,v to receive the text-prefix "
            "token-wise multiplier. Examples: q, k, v, k,v, q,k,v."
        ),
    )
    parser.add_argument(
        "--qkv-layer-range",
        default="0:32",
        help=(
            "Decoder-layer interval [start,end) for the text-prefix input "
            "RMSNorm scale capture and Q/K/V projection replay. Example: 0:32."
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
    target_group_norm_module_calls: int
    target_group_norm_token_positions: int
    qkv_scale_capture_calls: int
    qkv_scale_token_positions: int
    qkv_projection_calls: int
    qkv_projection_token_positions: int
    image_token_count: int
    merged_sequence_length: int
    image_start: Optional[int]
    image_end: Optional[int]


class HFAdaptVisController:
    """
    Runtime state shared by the native HF LLaVA merge, group RMSNorm override,
    and HF LLaMA attention modules.
    """

    def __init__(self, max_layers: int = 32) -> None:
        self.max_layers = int(max_layers)
        self.enabled = False
        self.weight = 1.0
        self.image_mask: Optional[torch.Tensor] = None
        self.valid_mask: Optional[torch.Tensor] = None
        self.modified_calls = 0
        self.target_group_norm_module_calls = 0
        self.target_group_norm_token_positions = 0
        # Per decoder layer: [B,T] alpha multiplier captured from the input
        # RMSNorm input. alpha is 1 for every unselected token.
        self.text_prefix_qkv_scale: Dict[int, torch.Tensor] = {}
        self.qkv_scale_capture_calls = 0
        self.qkv_scale_token_positions = 0
        self.qkv_projection_calls = 0
        self.qkv_projection_token_positions = 0
        self.merged_sequence_length = 0

    def begin_generation(self, weight: float) -> None:
        self.weight = float(weight)
        self.enabled = self.weight != 1.0
        self.image_mask = None
        self.valid_mask = None
        self.modified_calls = 0
        self.target_group_norm_module_calls = 0
        self.target_group_norm_token_positions = 0
        self.text_prefix_qkv_scale = {}
        self.qkv_scale_capture_calls = 0
        self.qkv_scale_token_positions = 0
        self.qkv_projection_calls = 0
        self.qkv_projection_token_positions = 0
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
            target_group_norm_module_calls=int(self.target_group_norm_module_calls),
            target_group_norm_token_positions=int(self.target_group_norm_token_positions),
            qkv_scale_capture_calls=int(self.qkv_scale_capture_calls),
            qkv_scale_token_positions=int(self.qkv_scale_token_positions),
            qkv_projection_calls=int(self.qkv_projection_calls),
            qkv_projection_token_positions=int(self.qkv_projection_token_positions),
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
        # HF merge returns final_attention_mask as the second tuple item.
        # It excludes right/left padding after multimodal expansion.
        final_attention_mask = outputs[1]
        controller.valid_mask = final_attention_mask.to(
            device=final_embedding.device,
            dtype=torch.bool,
        )
        controller.merged_sequence_length = int(final_embedding.shape[1])
        return outputs

    model._merge_input_ids_with_image_features = types.MethodType(
        wrapped_merge,
        model,
    )



def parse_layer_range(spec: str, total_layers: int) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", str(spec))
    if match is None:
        raise ValueError(
            "--target-layer-range must use start:end, e.g. 0:4 or 0:32; "
            f"got {spec!r}."
        )
    start, end = int(match.group(1)), int(match.group(2))
    if not (0 <= start <= end <= total_layers):
        raise ValueError(
            f"Invalid --target-layer-range {spec!r} for {total_layers} layers."
        )
    return start, end


def parse_norm_sites(spec: str) -> set[str]:
    allowed = {"input", "post_attn", "final"}
    requested = {part.strip() for part in str(spec).split(",") if part.strip()}
    if not requested or requested == {"all"}:
        return set(allowed)
    if "all" in requested or not requested.issubset(allowed):
        raise ValueError(
            "--target-norm-sites accepts all or a comma-separated subset of "
            "input,post_attn,final; got "
            f"{spec!r}."
        )
    return requested


def rmsnorm_site_from_name(module_name: str) -> Tuple[Optional[int], str]:
    if module_name == "model.norm":
        return None, "final"
    match = re.fullmatch(
        r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)",
        module_name,
    )
    if match is None:
        return None, "unknown"
    layer_index = int(match.group(1))
    site = "input" if match.group(2) == "input_layernorm" else "post_attn"
    return layer_index, site


def make_last_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """Return [B,T] with one True position: the final valid prefill token."""
    batch_size, q_len = valid_mask.shape
    positions = torch.arange(q_len, device=valid_mask.device).view(1, -1)
    positions = positions.expand(batch_size, -1)
    last_indices = positions.masked_fill(~valid_mask, -1).max(dim=1).values
    if torch.any(last_indices < 0):
        raise RuntimeError("Could not locate final valid text/query token.")
    result = torch.zeros_like(valid_mask, dtype=torch.bool)
    result.scatter_(1, last_indices[:, None], True)
    return result


def select_target_token_mask(
    *,
    hidden_states: torch.Tensor,
    controller: HFAdaptVisController,
    token_group: str,
) -> torch.Tensor:
    """Build a [B,T] target mask for one initial multimodal prefill call."""
    batch_size, q_len, _ = hidden_states.shape
    device = hidden_states.device

    image_mask = controller.image_mask
    if (
        image_mask is None
        or image_mask.ndim != 2
        or image_mask.shape[-1] != q_len
    ):
        raise RuntimeError(
            "Token-group RMSNorm override requires the merged image mask. "
            "The LLaVA merge capture did not match this prefill call."
        )
    image_mask = image_mask.to(device=device, dtype=torch.bool)
    if image_mask.shape[0] == 1 and batch_size > 1:
        image_mask = image_mask.expand(batch_size, -1)
    if image_mask.shape[0] != batch_size:
        raise RuntimeError(
            "Image-mask batch dimension mismatch: "
            f"mask={tuple(image_mask.shape)}, states={tuple(hidden_states.shape)}."
        )

    valid_mask = controller.valid_mask
    if valid_mask is None or tuple(valid_mask.shape) != (batch_size, q_len):
        valid_mask = torch.ones(batch_size, q_len, device=device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    # `all` deliberately includes padded positions, matching the true global
    # epsilon override as closely as possible. All other groups exclude pads.
    if token_group == "all":
        return torch.ones(batch_size, q_len, device=device, dtype=torch.bool)

    text_mask = valid_mask & ~image_mask
    last_query_mask = make_last_valid_mask(text_mask)
    text_prefix_mask = text_mask & ~last_query_mask

    if token_group == "image":
        return image_mask & valid_mask
    if token_group == "text":
        return text_mask
    if token_group == "text_prefix":
        return text_prefix_mask
    if token_group == "last_query":
        return last_query_mask
    if token_group == "image_plus_last_query":
        return (image_mask & valid_mask) | last_query_mask
    if token_group == "image_plus_text_prefix":
        return (image_mask & valid_mask) | text_prefix_mask

    raise ValueError(f"Unknown target token group: {token_group!r}")


def install_token_group_rmsnorm_override(
    model: LlavaForConditionalGeneration,
    controller: HFAdaptVisController,
    target_epsilon: Optional[float],
    token_group: str,
    layer_range: Tuple[int, int],
    norm_sites: set[str],
) -> int:
    """
    Keep every LlamaRMSNorm module at its configured base epsilon. During only
    the initial multimodal prefill, recompute a chosen token group with
    target_epsilon at selected RMSNorm modules. Every unselected token stays on
    the exact original module-forward path; q_len=1 decode calls are untouched.
    """
    if target_epsilon is None:
        return 0
    target_epsilon = float(target_epsilon)
    if target_epsilon <= 0.0:
        raise ValueError(
            "--target-rms-norm-eps must be positive, got "
            f"{target_epsilon}."
        )

    start_layer, end_layer = layer_range
    patched = 0
    for module_name, module in model.language_model.named_modules():
        if not isinstance(module, LlamaRMSNorm):
            continue

        layer_index, site = rmsnorm_site_from_name(module_name)
        if site not in norm_sites:
            continue
        if site != "final" and (
            layer_index is None
            or layer_index < start_layer
            or layer_index >= end_layer
        ):
            continue

        original_forward = module.forward

        def make_forward(original, display_name):
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # Only initial multimodal prefill. The normal decode path with
                # q_len=1 is intentionally unchanged.
                if (
                    hidden_states.ndim != 3
                    or hidden_states.shape[1] <= 1
                    or controller.merged_sequence_length != hidden_states.shape[1]
                ):
                    return original(hidden_states)

                output = original(hidden_states)
                target_mask = select_target_token_mask(
                    hidden_states=hidden_states,
                    controller=controller,
                    token_group=token_group,
                )
                if not bool(target_mask.any().item()):
                    return output

                # This exactly matches HF LlamaRMSNorm arithmetic for the
                # selected positions, but uses target_epsilon only there.
                input_dtype = hidden_states.dtype
                states_fp32 = hidden_states.to(torch.float32)
                variance = states_fp32.pow(2).mean(-1, keepdim=True)
                target_output = states_fp32 * torch.rsqrt(
                    variance + target_epsilon
                )
                target_output = self.weight * target_output.to(input_dtype)

                output = output.clone()
                expanded_mask = target_mask.unsqueeze(-1).expand_as(output)
                output = torch.where(expanded_mask, target_output, output)
                controller.target_group_norm_module_calls += 1
                controller.target_group_norm_token_positions += int(
                    target_mask.sum().item()
                )
                return output

            return forward

        module.forward = types.MethodType(
            make_forward(original_forward, module_name),
            module,
        )
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "No RMSNorm modules matched --target-layer-range and "
            "--target-norm-sites."
        )

    print(
        "Installed token-group RMSNorm override: "
        f"modules={patched}; group={token_group}; target_eps={target_epsilon:g}; "
        f"layers=[{start_layer},{end_layer}); sites={sorted(norm_sites)}."
    )
    print(
        "All unselected tokens and all decode-time q_len=1 calls retain "
        "--rms-norm-eps exactly."
    )
    return patched



def parse_qkv_components(spec: str) -> set[str]:
    allowed = {"q", "k", "v"}
    raw = str(spec).replace(" ", "")
    if not raw:
        raise ValueError("--qkv-components cannot be empty.")

    # Support both q,k,v and compact qkv / kv forms.
    parts = raw.split(",") if "," in raw else list(raw)
    requested = set(parts)
    if not requested.issubset(allowed):
        raise ValueError(
            "--qkv-components accepts a subset of q,k,v; got "
            f"{spec!r}."
        )
    return requested


def install_text_prefix_input_scale_capture(
    model: LlavaForConditionalGeneration,
    controller: HFAdaptVisController,
    target_epsilon: float,
    layer_range: Tuple[int, int],
) -> int:
    """
    Record the exact token-wise multiplier that changing text-prefix
    input_layernorm epsilon would induce, but do not change RMSNorm output.

    For a base RMSNorm epsilon eps_b and a target eps_t:
        RMSNorm_eps_t(x) = alpha(x) * RMSNorm_eps_b(x)
        alpha(x) = sqrt((mean(x^2)+eps_b)/(mean(x^2)+eps_t))
    """
    target_epsilon = float(target_epsilon)
    if target_epsilon <= 0.0:
        raise ValueError(
            "--qkv-target-eps must be positive, got "
            f"{target_epsilon}."
        )

    start_layer, end_layer = layer_range
    patched = 0
    for module_name, module in model.language_model.named_modules():
        if not isinstance(module, LlamaRMSNorm):
            continue

        layer_index, site = rmsnorm_site_from_name(module_name)
        if site != "input" or layer_index is None:
            continue
        if layer_index < start_layer or layer_index >= end_layer:
            continue

        original_forward = module.forward

        def make_forward(original, idx: int):
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # Preserve native RMSNorm output exactly. We only cache alpha.
                output = original(hidden_states)

                # Only the initial multimodal prefill. q_len=1 decoding is
                # intentionally left untouched.
                if (
                    hidden_states.ndim != 3
                    or hidden_states.shape[1] <= 1
                    or controller.merged_sequence_length != hidden_states.shape[1]
                ):
                    return output

                target_mask = select_target_token_mask(
                    hidden_states=hidden_states,
                    controller=controller,
                    token_group="text_prefix",
                )
                if not bool(target_mask.any().item()):
                    return output

                states_fp32 = hidden_states.to(torch.float32)
                mean_x2 = states_fp32.pow(2).mean(dim=-1)
                base_epsilon = float(self.variance_epsilon)
                alpha = torch.sqrt(
                    (mean_x2 + base_epsilon) / (mean_x2 + target_epsilon)
                )
                scale = torch.ones_like(alpha)
                scale = torch.where(target_mask, alpha, scale)

                # Use the same device; cast only when the projection is called.
                controller.text_prefix_qkv_scale[idx] = scale
                controller.qkv_scale_capture_calls += 1
                controller.qkv_scale_token_positions += int(
                    target_mask.sum().item()
                )
                return output

            return forward

        module.forward = types.MethodType(make_forward(original_forward, layer_index), module)
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "No input_layernorm modules matched --qkv-layer-range."
        )

    print(
        "Installed text-prefix input RMSNorm scale capture: "
        f"modules={patched}; target_eps={target_epsilon:g}; "
        f"layers=[{start_layer},{end_layer})."
    )
    return patched


def install_qkv_projection_replay(
    model: LlavaForConditionalGeneration,
    controller: HFAdaptVisController,
    components: set[str],
    layer_range: Tuple[int, int],
) -> int:
    """
    Apply the cached text-prefix alpha multiplier only to selected q_proj,
    k_proj and/or v_proj inputs. This keeps every RMSNorm module at its base
    epsilon and leaves all unselected projections unchanged.
    """
    start_layer, end_layer = layer_range
    patched = 0

    for layer_index, layer in enumerate(model.language_model.model.layers):
        if layer_index < start_layer or layer_index >= end_layer:
            continue

        attn = layer.self_attn
        for component in sorted(components):
            projection = getattr(attn, f"{component}_proj")
            original_forward = projection.forward

            def make_forward(original, idx: int, component_name: str):
                def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                    scale = controller.text_prefix_qkv_scale.get(idx)
                    should_apply = (
                        scale is not None
                        and hidden_states.ndim == 3
                        and hidden_states.shape[1] > 1
                        and hidden_states.shape[1] == controller.merged_sequence_length
                        and tuple(scale.shape) == tuple(hidden_states.shape[:2])
                    )
                    if should_apply:
                        scaled_input = hidden_states * scale.to(
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        ).unsqueeze(-1)
                        controller.qkv_projection_calls += 1
                        controller.qkv_projection_token_positions += int(
                            (scale != 1.0).sum().item()
                        )
                        return original(scaled_input)
                    return original(hidden_states)

                return forward

            projection.forward = types.MethodType(
                make_forward(original_forward, layer_index, component),
                projection,
            )
            patched += 1

    if patched == 0:
        raise RuntimeError("No Q/K/V projection modules were selected.")

    print(
        "Installed Q/K/V projection replay: "
        f"components={sorted(components)}; modules={patched}; "
        f"layers=[{start_layer},{end_layer})."
    )
    return patched


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
    print("Requested base RMSNorm epsilon:", float(args.rms_norm_eps))
    print("Requested target-group RMSNorm epsilon:", args.target_rms_norm_eps)
    print("Requested target token group:", args.target_token_group)
    print("Requested target layer range:", args.target_layer_range)
    print("Requested target norm sites:", args.target_norm_sites)
    print("Requested QKV target epsilon:", args.qkv_target_eps)
    print("Requested QKV components:", args.qkv_components)
    print("Requested QKV layer range:", args.qkv_layer_range)

    if args.target_rms_norm_eps is not None:
        raise ValueError(
            "This script isolates Q/K/V projection replay. Do not pass "
            "--target-rms-norm-eps; all RMSNorm modules must remain at "
            "--rms-norm-eps."
        )

    # The HF eager implementation is required for AdaptVis and for the
    # projection replay path.
    # Both epsilon groups use the same attention implementation.
    if args.method != "base" or args.qkv_target_eps is not None:
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
    qkv_layer_range = parse_layer_range(
        args.qkv_layer_range,
        total_layers=len(model.language_model.model.layers),
    )
    qkv_components = parse_qkv_components(args.qkv_components)

    # The merge capture provides the exact text-prefix mask after image-token
    # expansion. It does not alter native LLaVA merge outputs.
    install_merge_capture(model, controller)
    qkv_capture_module_count = install_text_prefix_input_scale_capture(
        model,
        controller,
        args.qkv_target_eps,
        qkv_layer_range,
    )
    qkv_projection_module_count = install_qkv_projection_replay(
        model,
        controller,
        qkv_components,
        qkv_layer_range,
    )

    # Legacy group-override fields are deliberately retained only as explicit
    # zeros in output metadata; no RMSNorm epsilon is overridden in this file.
    model._target_group_rmsnorm_eps = None
    model._target_group_name = None
    model._target_group_layer_range = None
    model._target_group_norm_sites = []
    model._target_group_rmsnorm_module_count = 0
    model._qkv_target_eps = float(args.qkv_target_eps)
    model._qkv_components = sorted(qkv_components)
    model._qkv_layer_range = qkv_layer_range
    model._qkv_capture_module_count = qkv_capture_module_count
    model._qkv_projection_module_count = qkv_projection_module_count

    if args.method != "base":
        install_hf_adaptvis_attention(model, controller)

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
    print(
        "  QKV replay target epsilon / components / layers: "
        f"{getattr(model, '_qkv_target_eps', None)} / "
        f"{getattr(model, '_qkv_components', None)} / "
        f"{getattr(model, '_qkv_layer_range', None)}"
    )
    print(
        "  QKV replay capture/projection modules: "
        f"{getattr(model, '_qkv_capture_module_count', None)} / "
        f"{getattr(model, '_qkv_projection_module_count', None)}"
    )
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
                    f"modified_calls={final_diag.modified_calls}, "
                    f"qkv_scale_capture_calls="
                    f"{final_diag.qkv_scale_capture_calls}, "
                    f"qkv_scale_token_positions="
                    f"{final_diag.qkv_scale_token_positions}, "
                    f"qkv_projection_calls="
                    f"{final_diag.qkv_projection_calls}, "
                    f"qkv_projection_token_positions="
                    f"{final_diag.qkv_projection_token_positions}"
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
        "target_rms_norm_eps": getattr(
            model,
            "_target_group_rmsnorm_eps",
            None,
        ),
        "target_token_group": getattr(model, "_target_group_name", None),
        "target_layer_range": getattr(model, "_target_group_layer_range", None),
        "target_norm_sites": getattr(model, "_target_group_norm_sites", None),
        "target_rmsnorm_module_count": getattr(
            model,
            "_target_group_rmsnorm_module_count",
            0,
        ),
        "qkv_replay": {
            "base_rms_norm_eps": float(args.rms_norm_eps),
            "target_eps": getattr(model, "_qkv_target_eps", None),
            "token_group": "text_prefix",
            "norm_site": "input_layernorm",
            "components": getattr(model, "_qkv_components", None),
            "layer_range": getattr(model, "_qkv_layer_range", None),
            "capture_module_count": getattr(model, "_qkv_capture_module_count", None),
            "projection_module_count": getattr(model, "_qkv_projection_module_count", None),
            "description": (
                "All LlamaRMSNorm modules retain base epsilon. The exact "
                "text-prefix input-RMSNorm alpha for target epsilon is "
                "replayed only at selected Q/K/V projection inputs during "
                "initial multimodal prefill."
            ),
        },
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
        f"base_rms_norm_eps={args.rms_norm_eps:g} "
        f"qkv_target_eps={args.qkv_target_eps:g} "
        f"qkv_components={args.qkv_components} "
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

    base_eps_suffix = f"{args.rms_norm_eps:.0e}".replace("-", "m")
    target_eps_suffix = f"{args.qkv_target_eps:.0e}".replace("-", "m")
    layer_suffix = args.qkv_layer_range.replace(":", "to")
    components_suffix = args.qkv_components.replace(",", "")
    suffix = (
        f"rms_base_{base_eps_suffix}_textprefix_qkv_{components_suffix}"
        f"_target_{target_eps_suffix}_layers{layer_suffix}"
    )
    return Path(
        "output/"
        f"hf_llava15_{args.dataset}_"
        f"{suffix}_{method_suffix}.json"
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
