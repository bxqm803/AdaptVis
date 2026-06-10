#!/usr/bin/env python3
"""
AdaptVis core-copy simulation for LLaVA-1.5, without importing upstream:
  - model_zoo.llava.LlavaForConditionalGenerationScal
  - model_zoo.llava15.change_greedy_to_add_weight

Copied core behavior:
  A. Compute LLaVA image features and manually merge <image> into visual tokens.
  B. Create the true merged visual-token mask: image_to_overwrite.
  C. Pass that mask to the language model via runtime state.
  D. Patch every LLaMA attention.forward so image-token raw attention logits are
     multiplied by weight BEFORE adding attention_mask.
  E. Replace change_greedy_to_add_weight with a manual greedy loop.

This is intended to test whether the upstream native result can be reproduced
without directly importing the native Scal class / greedy patch.
"""

import argparse
import json
import math
import os
import re
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration
from dataset_zoo.aro_datasets import Controlled_Images
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


# -------------------------
# Basic helpers
# -------------------------
def setup_cache(cache_dir):
    if cache_dir and str(cache_dir).lower() not in {"", "none", "null"}:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
        os.environ.setdefault("TORCH_HOME", "/ddnB/work/mwang32/torch_cache")
        for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]:
            Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def none_if_needed(x):
    if x is None:
        return None
    if str(x).lower() in {"", "none", "null"}:
        return None
    return x


def resolve_image_path(p):
    p = str(p)
    if os.path.exists(p):
        return p
    base = os.path.basename(p)
    for c in [os.path.join("data", "controlled_images", base), os.path.join("data", base)]:
        if os.path.exists(c):
            return c
    hits = list(Path("data").rglob(base))
    if hits:
        return str(hits[0])
    raise FileNotFoundError(p)


def clean_prompt_for_vlm(prompt):
    prompt = str(prompt)
    prompt = prompt.replace("<image>", "")
    prompt = prompt.replace("USER:", "").replace("User:", "").replace("user:", "")
    prompt = prompt.replace("ASSISTANT:", "").replace("Assistant:", "").replace("assistant:", "")
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


def make_prompt(raw_prompt, prompt_mode):
    raw_prompt = str(raw_prompt)
    clean = clean_prompt_for_vlm(raw_prompt)
    if prompt_mode == "raw":
        return raw_prompt
    if prompt_mode == "clean_mainaro":
        return f"<image>\nUSER: {clean}\nASSISTANT:"
    if prompt_mode == "clean_hf":
        return f"USER: <image>\n{clean}\nASSISTANT:"
    raise ValueError(prompt_mode)


def norm_gold(x):
    if isinstance(x, (list, tuple)):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def mainaro_is_correct(gold, gen):
    gold = norm_gold(gold)
    gen = str(gen).strip()
    if not gold:
        return False
    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.lower():
        ok = False
    return bool(ok)


def parse_prep(text):
    t = str(text).lower()
    t = re.sub(r"[^a-z\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if re.search(r"\bleft\b", t):
        return "left"
    if re.search(r"\bright\b", t):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b", t):
        return "under"
    if re.search(r"\bon\b|\btop\b|\babove\b|\bover\b", t):
        return "on"
    return None


def pred_to_option_index(pred_prep, gold_answer, caption_options):
    gold = parse_prep(gold_answer)
    if pred_prep is None:
        for i, cap in enumerate(caption_options):
            if parse_prep(cap) != gold:
                return i, False
        return 0, False

    pred_idx = None
    for i, cap in enumerate(caption_options):
        if parse_prep(cap) == pred_prep:
            pred_idx = i
            break

    if pred_idx is None:
        for i, cap in enumerate(caption_options):
            if parse_prep(cap) != gold:
                return i, False
        return 0, False

    return pred_idx, pred_prep == gold


def get_language_model(model):
    return model.language_model if hasattr(model, "language_model") else model.model.language_model


def get_image_token_id(model, processor):
    if hasattr(model.config, "image_token_index"):
        return int(model.config.image_token_index)
    return int(processor.tokenizer.convert_tokens_to_ids("<image>"))


def get_pad_token_id(model, processor):
    if getattr(model.config, "pad_token_id", None) is not None:
        return int(model.config.pad_token_id)
    if processor.tokenizer.pad_token_id is not None:
        return int(processor.tokenizer.pad_token_id)
    return 0


# -------------------------
# Copied core A: image features + true merged image mask
# -------------------------
@torch.no_grad()
def get_llava_image_features(model, pixel_values):
    """
    Equivalent to HF/LLaVA image feature path:
    vision_tower -> selected hidden state -> optional CLS removal -> projector.
    """
    cfg = model.config

    if hasattr(model, "get_image_features"):
        try:
            return model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=getattr(cfg, "vision_feature_layer", -2),
                vision_feature_select_strategy=getattr(cfg, "vision_feature_select_strategy", "default"),
            )
        except TypeError:
            try:
                return model.get_image_features(pixel_values=pixel_values)
            except TypeError:
                pass

    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
    layer = getattr(cfg, "vision_feature_layer", -2)
    select_strategy = getattr(cfg, "vision_feature_select_strategy", "default")

    selected = image_outputs.hidden_states[layer]
    if select_strategy == "default":
        selected = selected[:, 1:]
    elif select_strategy == "full":
        pass
    else:
        raise ValueError(f"Unknown vision_feature_select_strategy: {select_strategy}")

    return model.multi_modal_projector(selected)


def merge_input_ids_with_image_features_exact_core(
    model,
    image_features,
    inputs_embeds,
    input_ids,
    attention_mask,
    position_ids=None,
    image_token_id=32000,
    pad_token_id=0,
):
    """
    Core copied from LLaVA merge logic. The important output is image_to_overwrite:
    a boolean mask aligned with the merged LLM sequence.
    """
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    special_image_token_mask = input_ids == int(image_token_id)
    num_special_image_tokens = special_image_token_mask.sum(dim=-1)

    max_embed_dim = int((num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length)

    batch_indices, non_image_indices = torch.where(input_ids != int(image_token_id))

    # New text token positions after replacing each <image> with num_image_patches positions.
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), dim=-1) - 1

    # Same left-padding detection as the original merge implementation.
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id, device=input_ids.device))
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions = new_token_positions + nb_image_pad[:, None]

    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    final_embedding = torch.zeros(
        batch_size,
        max_embed_dim,
        embed_dim,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    final_attention_mask = torch.zeros(
        batch_size,
        max_embed_dim,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

    # This is the real merged image mask.
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(image_to_overwrite.device)

    if image_to_overwrite.sum().item() != image_features.numel() // embed_dim:
        raise ValueError(
            f"image mask count mismatch: image_to_overwrite.sum={image_to_overwrite.sum().item()}, "
            f"expected={image_features.numel() // embed_dim}. "
            f"input_ids image tokens={int(special_image_token_mask.sum().item())}, "
            f"num_image_patches={num_image_patches}, max_embed_dim={max_embed_dim}"
        )

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(final_embedding.device)
    final_attention_mask = final_attention_mask | image_to_overwrite.to(final_attention_mask.dtype)

    if position_ids is None:
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill(final_attention_mask == 0, 1)
    else:
        position_ids = position_ids.to(final_attention_mask.device)

    return final_embedding, final_attention_mask, position_ids, image_to_overwrite


# -------------------------
# Copied core B/C/D: pass merged image mask to attention and modify raw logits
# -------------------------
def set_adaptvis_state(lm, enable, weight, image_mask, require_square=True, max_layers=32):
    lm._av_enable = bool(enable)
    lm._av_weight = float(weight)
    lm._av_image_mask = image_mask.bool() if image_mask is not None else None
    lm._av_require_square = bool(require_square)
    lm._av_max_layers = int(max_layers)

    lm._av_modified_calls = 0
    lm._av_modified_pos_count = 0
    lm._av_logit_sum_before = 0.0
    lm._av_logit_sum_after = 0.0


def apply_adaptvis_raw_logit_intervention(attn_weights, lm, layer_idx):
    if not getattr(lm, "_av_enable", False):
        return attn_weights
    if int(layer_idx) >= int(getattr(lm, "_av_max_layers", 32)):
        return attn_weights

    image_mask = getattr(lm, "_av_image_mask", None)
    if image_mask is None:
        return attn_weights

    weight = float(getattr(lm, "_av_weight", 1.0))
    if weight == 1.0:
        return attn_weights

    bsz, _, q_len, kv_len = attn_weights.shape
    if getattr(lm, "_av_require_square", True) and q_len != kv_len:
        return attn_weights

    if image_mask.shape[-1] != kv_len:
        # If this triggers, the mask is not aligned with the actual LLM sequence.
        raise ValueError(f"image_mask length {image_mask.shape[-1]} != kv_len {kv_len}")

    out = attn_weights.clone()
    calls = 0
    pos_count = 0
    before_sum = 0.0
    after_sum = 0.0

    for b in range(bsz):
        pos = torch.nonzero(image_mask[b], as_tuple=False).view(-1).to(out.device)
        if pos.numel() == 0:
            continue

        before = out[b, :, -1:, pos].detach().float().sum().item()
        out[b, :, -1:, pos] *= weight
        after = out[b, :, -1:, pos].detach().float().sum().item()

        calls += 1
        pos_count += int(pos.numel())
        before_sum += before
        after_sum += after

    if calls:
        lm._av_modified_calls = getattr(lm, "_av_modified_calls", 0) + calls
        lm._av_modified_pos_count = getattr(lm, "_av_modified_pos_count", 0) + pos_count
        lm._av_logit_sum_before = getattr(lm, "_av_logit_sum_before", 0.0) + before_sum
        lm._av_logit_sum_after = getattr(lm, "_av_logit_sum_after", 0.0) + after_sum

    return out


def get_adaptvis_trace(lm):
    image_mask = getattr(lm, "_av_image_mask", None)
    return {
        "image_mask_sum": None if image_mask is None else [int(x) for x in image_mask.sum(dim=-1).detach().cpu().tolist()],
        "modified_calls": int(getattr(lm, "_av_modified_calls", 0)),
        "modified_pos_count": int(getattr(lm, "_av_modified_pos_count", 0)),
        "logit_sum_before": float(getattr(lm, "_av_logit_sum_before", 0.0)),
        "logit_sum_after": float(getattr(lm, "_av_logit_sum_after", 0.0)),
    }


def patch_llama_attention_for_adaptvis(language_model, max_layers=32):
    if getattr(language_model, "_av_attention_patched", False):
        return

    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        layers = language_model.model.layers
    elif hasattr(language_model, "layers"):
        layers = language_model.layers
    else:
        raise RuntimeError("Cannot find language_model.model.layers")

    language_model._av_max_layers = int(max_layers)

    def make_forward(attn_module, layer_idx):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            num_heads = getattr(self, "num_heads")
            head_dim = getattr(self, "head_dim")
            num_kv_heads = getattr(self, "num_key_value_heads", num_heads)
            num_kv_groups = getattr(self, "num_key_value_groups", num_heads // num_kv_heads)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                try:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                except Exception:
                    pass

            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                try:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                except TypeError:
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                try:
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                except Exception:
                    pass

            key_states = repeat_kv(key_states, num_kv_groups)
            value_states = repeat_kv(value_states, num_kv_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            # Exact core: raw logits, before attention_mask.
            attn_weights = apply_adaptvis_raw_logit_intervention(attn_weights, language_model, layer_idx)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        return types.MethodType(forward, attn_module)

    patched = 0
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.forward = make_forward(layer.self_attn, idx)
            patched += 1

    language_model._av_attention_patched = True
    print(f"patched LLaMA attention layers: {patched}")


# -------------------------
# Copied core E: manual greedy loop
# -------------------------
@torch.no_grad()
def llava_lm_forward_core(
    model,
    input_ids,
    attention_mask,
    pixel_values,
    image_token_id,
    pad_token_id,
    enable_intervention,
    weight,
    require_square,
    max_layers,
):
    lm = get_language_model(model)

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = get_llava_image_features(model, pixel_values)

    merged_embeds, merged_attention_mask, position_ids, image_mask = merge_input_ids_with_image_features_exact_core(
        model=model,
        image_features=image_features,
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=None,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
    )

    set_adaptvis_state(
        lm=lm,
        enable=enable_intervention,
        weight=weight,
        image_mask=image_mask,
        require_square=require_square,
        max_layers=max_layers,
    )

    outputs = lm(
        inputs_embeds=merged_embeds,
        attention_mask=merged_attention_mask,
        # Match native Scal: modeling_llava_scal.py computes position_ids but does not pass it.
        # position_ids=position_ids,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    return outputs.logits, get_adaptvis_trace(lm), image_mask


@torch.no_grad()
def manual_greedy_core(
    model,
    processor,
    input_ids,
    attention_mask,
    pixel_values,
    image_token_id,
    pad_token_id,
    max_new_tokens,
    enable_intervention,
    weight,
    intervention_scope,
    require_square,
    max_layers,
):
    """
    Native-like generation:
      - Step 0: manually merge image tokens, pass merged image_mask, optionally intervene.
      - Step >0: use KV cache and only pass the newly generated token.
                No image mask / intervention, matching q_len != kv_len native behavior.
    """
    lm = get_language_model(model)

    generated = []
    score_list = []
    trace_list = []
    mask_sums = []

    eos_id = processor.tokenizer.eos_token_id

    # ---------- step 0: full merged prompt ----------
    enable_now = bool(enable_intervention and intervention_scope in {"first_step", "all_steps"})

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = get_llava_image_features(model, pixel_values)

    merged_embeds, merged_attention_mask, _position_ids, image_mask = merge_input_ids_with_image_features_exact_core(
        model=model,
        image_features=image_features,
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=None,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
    )

    set_adaptvis_state(
        lm=lm,
        enable=enable_now,
        weight=weight,
        image_mask=image_mask,
        require_square=require_square,
        max_layers=max_layers,
    )

    out = lm(
        inputs_embeds=merged_embeds,
        attention_mask=merged_attention_mask,
        # Critical: native Scal computes position_ids but comments it out in language_model call.
        # position_ids=_position_ids,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    next_logits = out.logits[:, -1, :]
    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

    score_list.append(next_logits.detach())
    generated.append(next_token)
    trace_list.append(get_adaptvis_trace(lm))
    mask_sums.append([int(x) for x in image_mask.sum(dim=-1).detach().cpu().tolist()])

    past_key_values = out.past_key_values
    cur_attention_mask = torch.cat(
        [merged_attention_mask, torch.ones((merged_attention_mask.shape[0], 1), dtype=merged_attention_mask.dtype, device=merged_attention_mask.device)],
        dim=-1,
    )

    if eos_id is not None and int(next_token[0, 0].item()) == int(eos_id):
        gen_ids = torch.cat(generated, dim=-1)
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        first_conf = float(F.softmax(score_list[0][0].float(), dim=-1).max().item())
        total_trace = {
            "steps": len(trace_list),
            "modified_calls": sum(t["modified_calls"] for t in trace_list),
            "modified_pos_count": sum(t["modified_pos_count"] for t in trace_list),
            "logit_sum_before": sum(t["logit_sum_before"] for t in trace_list),
            "logit_sum_after": sum(t["logit_sum_after"] for t in trace_list),
            "first_step_trace": trace_list[0] if trace_list else None,
            "last_step_trace": trace_list[-1] if trace_list else None,
            "mask_sums_by_step": mask_sums[:3],
        }
        return text, first_conf, total_trace

    # ---------- later steps: cache path, no merge ----------
    # This is much closer to native generate than recomputing the whole merged prefix.
    # Since q_len=1 and kv_len>1, native AdaptVis condition q_len==kv_len would skip intervention.
    cur_input_ids = next_token

    for step in range(1, int(max_new_tokens)):
        set_adaptvis_state(
            lm=lm,
            enable=False,
            weight=weight,
            image_mask=None,
            require_square=require_square,
            max_layers=max_layers,
        )

        out = lm(
            input_ids=cur_input_ids,
            attention_mask=cur_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        next_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        score_list.append(next_logits.detach())
        generated.append(next_token)
        trace_list.append(get_adaptvis_trace(lm))

        past_key_values = out.past_key_values
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones((cur_attention_mask.shape[0], 1), dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)],
            dim=-1,
        )
        cur_input_ids = next_token

        if eos_id is not None and int(next_token[0, 0].item()) == int(eos_id):
            break

    gen_ids = torch.cat(generated, dim=-1) if generated else torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    first_conf = float(F.softmax(score_list[0][0].float(), dim=-1).max().item()) if score_list else 0.0

    total_trace = {
        "steps": len(trace_list),
        "modified_calls": sum(t["modified_calls"] for t in trace_list),
        "modified_pos_count": sum(t["modified_pos_count"] for t in trace_list),
        "logit_sum_before": sum(t["logit_sum_before"] for t in trace_list),
        "logit_sum_after": sum(t["logit_sum_after"] for t in trace_list),
        "first_step_trace": trace_list[0] if trace_list else None,
        "last_step_trace": trace_list[-1] if trace_list else None,
        "mask_sums_by_step": mask_sums[:3],
    }

    return text, first_conf, total_trace


def build_inputs(processor, image, prompt, device, pad_mode, max_length):
    if pad_mode == "none":
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    elif pad_mode == "max77":
        inputs = processor(text=prompt, images=image, padding="max_length", max_length=int(max_length), return_tensors="pt")
    else:
        raise ValueError(pad_mode)
    return inputs.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--revision", default="a272c74")
    ap.add_argument("--cache_dir", default="/ddnB/work/mwang32/hf_cache")
    ap.add_argument("--root_dir", default="data")
    ap.add_argument("--dataset", default="Controlled_Images_A")
    ap.add_argument("--subset", default="A")
    ap.add_argument("--option", default="four")

    ap.add_argument("--prompt_mode", choices=["raw", "clean_mainaro", "clean_hf"], default="raw")
    ap.add_argument("--pad_mode", choices=["none", "max77"], default="max77")
    ap.add_argument("--max_length", type=int, default=77)
    ap.add_argument("--max_new_tokens", type=int, default=16)

    ap.add_argument("--weight1", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--round_confidence", action="store_true")

    ap.add_argument("--intervention_scope", choices=["first_step", "all_steps", "none"], default="first_step")
    ap.add_argument("--no_square_required", action="store_true")
    ap.add_argument("--max_layers", type=int, default=32)

    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--score_mode", choices=["mainaro", "predidx"], default="mainaro")
    args = ap.parse_args()

    setup_cache(args.cache_dir)
    Path("outputs").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = none_if_needed(args.revision)

    print("args:", vars(args))
    print("device:", device)

    dataset = Controlled_Images(image_preprocess=None, root_dir=args.root_dir, download=True, subset=args.subset)
    prompt_file = f"prompts/{args.dataset}_with_answer_{args.option}_options.jsonl"

    raw_prompts, answers = [], []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            raw_prompts.append(r["question"])
            answers.append(r["answer"])

    n_total = len(dataset.dataset)
    if args.limit > 0:
        n_total = min(n_total, args.limit)

    proc_kwargs = dict(cache_dir=none_if_needed(args.cache_dir))
    model_kwargs = dict(
        cache_dir=none_if_needed(args.cache_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    if revision is not None:
        proc_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).eval().to(device)
    model.requires_grad_(False)

    lm = get_language_model(model)
    patch_llama_attention_for_adaptvis(lm, max_layers=args.max_layers)

    image_token_id = get_image_token_id(model, processor)
    pad_token_id = get_pad_token_id(model, processor)
    print("image_token_id:", image_token_id)
    print("pad_token_id:", pad_token_id)
    print("n_total:", n_total)

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_corecopy_mergedmask_{model_tag}_{args.dataset}"
        f"_prompt{args.prompt_mode}_pad{args.pad_mode}_scope{args.intervention_scope}"
        f"_max{args.max_new_tokens}_rev{args.revision}"
        f"_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}"
    )
    out_records = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    scores_arr = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    main_correct = 0
    strict_correct = 0
    unparsed = 0
    require_square = not args.no_square_required

    for i in tqdm(range(n_total), total=n_total):
        d = dataset.dataset[i]
        image_path = resolve_image_path(d["image_path"])
        image = Image.open(image_path).convert("RGB")
        prompt = make_prompt(raw_prompts[i], args.prompt_mode)
        gold = norm_gold(answers[i])
        caption_options = d["caption_options"]

        inputs = build_inputs(processor, image, prompt, device, args.pad_mode, args.max_length)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        pixel_values = inputs["pixel_values"]

        probe_text, probe_conf, probe_trace = manual_greedy_core(
            model=model,
            processor=processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=args.max_new_tokens,
            enable_intervention=False,
            weight=1.0,
            intervention_scope="none",
            require_square=require_square,
            max_layers=args.max_layers,
        )

        conf_used = round(probe_conf, 2) if args.round_confidence else probe_conf
        chosen_weight = args.weight1 if conf_used < args.threshold else args.weight2

        final_text, final_conf, final_trace = manual_greedy_core(
            model=model,
            processor=processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=args.max_new_tokens,
            enable_intervention=True,
            weight=chosen_weight,
            intervention_scope=args.intervention_scope,
            require_square=require_square,
            max_layers=args.max_layers,
        )

        main_ok = mainaro_is_correct(gold, final_text)
        pred = parse_prep(final_text)
        pred_idx, strict_ok = pred_to_option_index(pred, gold, caption_options)

        if pred is None:
            unparsed += 1
        main_correct += int(main_ok)
        strict_correct += int(strict_ok)

        if args.score_mode == "mainaro":
            scores_arr[i, 0, :] = np.array([1, 0, 0, 0], dtype=np.float32) if main_ok else np.array([0, 0, 1, 0], dtype=np.float32)
        else:
            scores_arr[i, 0, pred_idx] = 1.0

        rec = {
            "index": i,
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "probe_generation": probe_text,
            "probe_confidence": probe_conf,
            "probe_confidence_used": conf_used,
            "chosen_weight": chosen_weight,
            "generation": final_text,
            "final_confidence": final_conf,
            "pred_prep": pred,
            "pred_idx": int(pred_idx),
            "mainaro_correct": bool(main_ok),
            "strict_correct": bool(strict_ok),
            "input_len": int(input_ids.shape[-1]),
            "image_token_count_raw": int((input_ids == image_token_id).sum().item()),
            "probe_trace": probe_trace,
            "final_trace": final_trace,
            "caption_options": list(caption_options),
        }
        records.append(rec)

        if args.print_every > 0 and i % args.print_every == 0:
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("input_len:", rec["input_len"], "raw_image_token_count:", rec["image_token_count_raw"])
            print("prompt:", repr(prompt))
            print("gold:", gold)
            print("probe_conf:", probe_conf, "conf_used:", conf_used)
            print("probe_generation:", probe_text)
            print("chosen_weight:", chosen_weight)
            print("generation:", final_text)
            print("pred:", pred, "mainaro_correct:", main_ok, "strict_correct:", strict_ok)
            print("final_trace:", final_trace)
            print("running mainaro acc:", main_correct / (i + 1), "strict acc:", strict_correct / (i + 1), "unparsed:", unparsed)

        if (i + 1) % 25 == 0:
            with open(out_records, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    with open(out_records, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    main_acc = main_correct / max(n_total, 1)
    strict_acc = strict_correct / max(n_total, 1)

    summary = {
        "args": vars(args),
        "n": n_total,
        "mainaro_style_direct_acc": main_acc,
        "strict_parse_direct_acc": strict_acc,
        "unparsed": unparsed,
        "out_records": str(out_records),
        "definition": "Core-copy AdaptVis v2: stock HF LLaVA + manual LLaVA merge mask + raw-logit attention intervention + native-like KV-cache greedy + no explicit position_ids.",
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDirect acc mainaro-style:", main_acc)
    print("Direct acc strict-parse:", strict_acc)
    print("unparsed:", unparsed)
    print("saved records:", out_records)
    print("saved summary:", out_summary)

    if n_total == len(dataset.dataset):
        print("\nRunning Controlled_Images evaluator...")
        dataset.evaluate_scores(
            scores=scores_arr,
            path="outputs",
            dataset=args.dataset,
            model=model_tag + "_corecopy_mergedmask",
            method="adapt_vis",
            weight=1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
