import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor

try:
    from transformers import LlavaProcessor
except Exception:
    LlavaProcessor = None

try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

from dataset_zoo.aro_datasets import Controlled_Images


# ============================================================
# Basic utilities
# ============================================================
def setup_cache():
    os.environ.setdefault("HF_HOME", "/ddnB/work/mwang32/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/ddnB/work/mwang32/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/ddnB/work/mwang32/hf_cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/ddnB/work/mwang32/hf_cache/datasets")
    os.environ.setdefault("TORCH_HOME", "/ddnB/work/mwang32/torch_cache")

    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]:
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def clean_prompt_for_vlm(prompt):
    prompt = str(prompt)
    prompt = prompt.replace("<image>", "")
    prompt = prompt.replace("USER:", "").replace("User:", "").replace("user:", "")
    prompt = prompt.replace("ASSISTANT:", "").replace("Assistant:", "").replace("assistant:", "")
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


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


def resolve_image_path(p):
    p = str(p)
    if os.path.exists(p):
        return p

    base = os.path.basename(p)
    candidates = [
        os.path.join("data", "controlled_images", base),
        os.path.join("data", base),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    hits = list(Path("data").rglob(base))
    if hits:
        return str(hits[0])

    raise FileNotFoundError(p)


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


def get_model_device(model):
    return next(model.parameters()).device


# ============================================================
# LLaVA loading / preprocessing
# This follows the same evaluation style as the InternVL script:
#   image + prompt -> model.generate -> parse generation.
# It does not use the original LLaVA/AdaptVis repo wrapper.
# ============================================================
def load_llava_processor(model_id):
    """
    Prefer AutoProcessor. If the installed tokenizers library cannot read the fast
    tokenizer JSON, fall back to slow tokenizer + image processor.
    """
    try:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        return processor, tokenizer
    except Exception as e:
        print("AutoProcessor failed, falling back to slow tokenizer + image processor.")
        print("AutoProcessor error:", repr(e))

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    if LlavaProcessor is None:
        raise RuntimeError("LlavaProcessor is not available in this transformers install.")

    processor = LlavaProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    return processor, tokenizer


def load_llava_model(model_id, torch_dtype, device_map=None, load_in_8bit=False, attn_implementation="eager"):
    kwargs = dict(
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    if device_map is not None:
        kwargs["device_map"] = device_map

    # Force eager attention if supported. This is important because the softmax patch
    # must see the decoder attention probabilities.
    kwargs["attn_implementation"] = attn_implementation

    if LlavaForConditionalGeneration is not None:
        try:
            return LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        except TypeError as e:
            print("LlavaForConditionalGeneration TypeError, retrying without attn_implementation:", e)
            kwargs.pop("attn_implementation", None)
            return LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()

    if AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(model_id, **kwargs).eval()
        except TypeError as e:
            print("AutoModelForVision2Seq TypeError, retrying without attn_implementation:", e)
            kwargs.pop("attn_implementation", None)
            return AutoModelForVision2Seq.from_pretrained(model_id, **kwargs).eval()

    raise RuntimeError("Neither LlavaForConditionalGeneration nor AutoModelForVision2Seq is available.")


def get_language_model(model):
    # HF LLaVA usually has language_model.
    if hasattr(model, "language_model"):
        return model.language_model

    # Some wrappers place it under model.language_model or model.model.
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model

    # Last fallback: patch the whole model.
    return model


def get_image_token_id(model, processor, tokenizer):
    if hasattr(model, "config") and hasattr(model.config, "image_token_index"):
        return int(model.config.image_token_index)

    for tok in ["<image>", "<im_patch>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            return int(tid)

    if hasattr(processor, "image_token"):
        tid = tokenizer.convert_tokens_to_ids(processor.image_token)
        if tid is not None:
            return int(tid)

    raise RuntimeError("Cannot find LLaVA image token id.")


def get_expected_image_seq_len(model, processor):
    """
    HF LLaVA-1.5 usually has 576 image features for CLIP-L/14-336.
    Prefer config.image_seq_length when available; otherwise infer from vision config.
    """
    if hasattr(model, "config") and hasattr(model.config, "image_seq_length"):
        return int(model.config.image_seq_length)

    cfg = getattr(model, "config", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is not None:
        image_size = getattr(vision_cfg, "image_size", None)
        patch_size = getattr(vision_cfg, "patch_size", None)
        if image_size is not None and patch_size is not None:
            return int((int(image_size) // int(patch_size)) ** 2)

    if hasattr(processor, "num_image_tokens"):
        return int(processor.num_image_tokens)

    return 576


def maybe_expand_single_image_token(inputs, model, processor, tokenizer):
    """
    Some processor versions already expand <image> into 576 repeated image tokens.
    Older processor/tokenizer combinations may leave a single image token. HF LLaVA
    forward expects the number of image-token positions to match image features.

    If there is exactly one image token, expand it to image_seq_length repeated tokens.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    image_token_id = get_image_token_id(model, processor, tokenizer)

    counts = (input_ids == image_token_id).sum(dim=1)
    if int(counts.max().item()) != 1:
        return inputs

    expected = get_expected_image_seq_len(model, processor)

    new_ids = []
    new_masks = []

    for b in range(input_ids.shape[0]):
        ids = input_ids[b]
        pos = torch.nonzero(ids == image_token_id, as_tuple=False).view(-1)
        if pos.numel() != 1:
            new_ids.append(ids)
            if attention_mask is not None:
                new_masks.append(attention_mask[b])
            continue

        p = int(pos.item())
        expanded = torch.full(
            (expected,),
            fill_value=image_token_id,
            dtype=ids.dtype,
            device=ids.device,
        )
        ids2 = torch.cat([ids[:p], expanded, ids[p + 1:]], dim=0)
        new_ids.append(ids2)

        if attention_mask is not None:
            m = attention_mask[b]
            m2 = torch.cat([
                m[:p],
                torch.ones(expected, dtype=m.dtype, device=m.device),
                m[p + 1:],
            ], dim=0)
            new_masks.append(m2)

    max_len = max(x.numel() for x in new_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    padded_ids = []
    padded_masks = []
    for ids in new_ids:
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            ids = torch.cat([
                ids,
                torch.full((pad_len,), pad_id, dtype=ids.dtype, device=ids.device),
            ], dim=0)
        padded_ids.append(ids)

    if attention_mask is not None:
        for m in new_masks:
            pad_len = max_len - m.numel()
            if pad_len > 0:
                m = torch.cat([
                    m,
                    torch.zeros(pad_len, dtype=m.dtype, device=m.device),
                ], dim=0)
            padded_masks.append(m)

    inputs["input_ids"] = torch.stack(padded_ids, dim=0)
    if attention_mask is not None:
        inputs["attention_mask"] = torch.stack(padded_masks, dim=0)

    return inputs


def build_inputs(processor, tokenizer, model, image_path, prompt, device):
    image = Image.open(image_path).convert("RGB")

    # LLaVA-1.5 standard conversation style.
    question = f"<image>\nUSER: {prompt}\nASSISTANT:"

    inputs = processor(
        text=question,
        images=image,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )

    # Do NOT expand the <image> token here.
    # HF LLaVA expects exactly one <image> token per image in input_ids.
    # It expands this single token to image_seq_len visual embeddings internally
    # before calling the language model.
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    return inputs, question


# ============================================================
# Generic AdaptVis via language-model PRE-SOFTMAX attention-logit patch
#
# Equivalence target: main_aro / modeling_llama_add_attn.py behavior.
#   - patch F.softmax for eager attention
#   - patch F.scaled_dot_product_attention for SDPA attention
#   - activate only inside language_model.forward
#   - only modify prefill/full-context attention: q_len == kv_len
#   - only modify the last query's logits to LLaVA image-feature positions
#   - do: attn_logits[..., image_pos] = weight * attn_logits[..., image_pos]
#   - then do normal softmax
# ============================================================
_ORIG_SOFTMAX = None
_ORIG_SDPA = None
_ATTN_PATCHED = False


def _safe_softmax_call(orig_softmax, input_tensor, dim=None, _stacklevel=3, dtype=None):
    if dtype is None:
        try:
            return orig_softmax(input_tensor, dim=dim, _stacklevel=_stacklevel)
        except TypeError:
            return orig_softmax(input_tensor, dim=dim)
    try:
        return orig_softmax(input_tensor, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    except TypeError:
        return orig_softmax(input_tensor, dim=dim, dtype=dtype)


def _call_orig_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    try:
        return _ORIG_SDPA(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    except TypeError:
        try:
            return _ORIG_SDPA(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        except TypeError:
            return _ORIG_SDPA(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )


def _scale_attention_logits_if_needed(attn_logits, lm, weight):
    """
    attn_logits: [B,H,Q,K], pre-softmax attention scores.
    Scale last-query image logits only.
    """
    if attn_logits.dim() != 4:
        return attn_logits

    positions_by_batch = getattr(lm, "_adaptvis_image_positions", None)
    if not positions_by_batch:
        return attn_logits

    bsz, _, q_len, kv_len = attn_logits.shape

    # Match main_aro-style behavior: prefill/full-context only.
    if bool(getattr(lm, "_adaptvis_prefill_only", True)) and q_len != kv_len:
        return attn_logits

    modified = attn_logits.clone()

    local_calls = 0
    local_before = 0.0
    local_after = 0.0
    local_pos_count = 0

    for b in range(min(bsz, len(positions_by_batch))):
        pos = positions_by_batch[b]
        if pos is None:
            continue
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, device=modified.device, dtype=torch.long)
        else:
            pos = pos.to(device=modified.device, dtype=torch.long)

        pos = pos[(pos >= 0) & (pos < kv_len)]
        if pos.numel() == 0:
            continue

        before = modified[b, :, -1:, pos].detach().float().sum().item()
        modified[b, :, -1:, pos] *= weight
        after = modified[b, :, -1:, pos].detach().float().sum().item()

        local_calls += 1
        local_before += before
        local_after += after
        local_pos_count += int(pos.numel())

    if local_calls == 0:
        return attn_logits

    # Explicit fields
    lm._adaptvis_modified_calls = getattr(lm, "_adaptvis_modified_calls", 0) + local_calls
    lm._adaptvis_modified_pos_count = getattr(lm, "_adaptvis_modified_pos_count", 0) + local_pos_count
    lm._adaptvis_logit_sum_before = getattr(lm, "_adaptvis_logit_sum_before", 0.0) + local_before
    lm._adaptvis_logit_sum_after = getattr(lm, "_adaptvis_logit_sum_after", 0.0) + local_after

    # Compatibility fields used by old compare scripts
    lm._adaptvis_scaled_calls = getattr(lm, "_adaptvis_scaled_calls", 0) + local_calls
    lm._adaptvis_scaled_pos_count = getattr(lm, "_adaptvis_scaled_pos_count", 0) + local_pos_count
    lm._adaptvis_scaled_mass_before = getattr(lm, "_adaptvis_scaled_mass_before", 0.0) + local_before
    lm._adaptvis_scaled_mass_after_raw = getattr(lm, "_adaptvis_scaled_mass_after_raw", 0.0) + local_after

    return modified


def _make_causal_mask(q_len, k_len, device):
    return torch.ones((q_len, k_len), dtype=torch.bool, device=device).tril(diagonal=k_len - q_len)


def _manual_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                         is_causal=False, scale=None, enable_gqa=False):
    """
    Replacement for F.scaled_dot_product_attention, so pre-softmax logits are editable.
    Shapes expected: [B,H,Q,D], [B,H,K,D], [B,H,K,D].
    """
    lm = getattr(F.scaled_dot_product_attention, "language_model", None)

    if lm is None or not getattr(lm, "_adaptvis_in_lm_forward", False) or not getattr(lm, "_adaptvis_enable", False):
        return _call_orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    weight = float(getattr(lm, "_adaptvis_weight", 1.0))
    if weight == 1.0:
        return _call_orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return _call_orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    bsz, q_heads, q_len, _ = query.shape
    _, k_heads, k_len, _ = key.shape

    if bool(getattr(lm, "_adaptvis_prefill_only", True)) and q_len != k_len:
        return _call_orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    if k_heads != q_heads:
        if q_heads % k_heads == 0:
            repeat = q_heads // k_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        else:
            return _call_orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    scale_factor = (1.0 / (query.size(-1) ** 0.5)) if scale is None else scale
    attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # Main operation: pre-softmax image-logit scaling.
    attn_logits = _scale_attention_logits_if_needed(attn_logits, lm, weight)

    if is_causal:
        causal_mask = _make_causal_mask(q_len, k_len, query.device)
        attn_logits = attn_logits.masked_fill(
            ~causal_mask.view(1, 1, q_len, k_len),
            torch.finfo(attn_logits.dtype).min,
        )

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_logits = attn_logits.masked_fill(~attn_mask, torch.finfo(attn_logits.dtype).min)
        else:
            attn_logits = attn_logits + attn_mask

    attn_probs = _ORIG_SOFTMAX(attn_logits.float(), dim=-1).to(query.dtype)

    if dropout_p and dropout_p > 0:
        attn_probs = torch.dropout(attn_probs, dropout_p, train=True)

    return torch.matmul(attn_probs, value)


def install_adaptvis_softmax_patch(language_model, verbose=False):
    """
    The function name is retained for compatibility, but this installs
    pre-softmax logit scaling, not post-softmax probability scaling.
    """
    global _ORIG_SOFTMAX, _ORIG_SDPA, _ATTN_PATCHED

    if not getattr(language_model, "_adaptvis_forward_patched", False):
        old_forward = language_model.forward

        def patched_lm_forward(*args, **kwargs):
            language_model._adaptvis_in_lm_forward = True
            try:
                return old_forward(*args, **kwargs)
            finally:
                language_model._adaptvis_in_lm_forward = False

        language_model.forward = patched_lm_forward
        language_model._adaptvis_forward_patched = True
        language_model._adaptvis_in_lm_forward = False

    if not _ATTN_PATCHED:
        _ORIG_SOFTMAX = F.softmax
        _ORIG_SDPA = F.scaled_dot_product_attention

        def patched_softmax(input, dim=None, _stacklevel=3, dtype=None):
            lm = getattr(patched_softmax, "language_model", None)

            if (
                lm is not None
                and getattr(lm, "_adaptvis_in_lm_forward", False)
                and getattr(lm, "_adaptvis_enable", False)
                and dim in (-1, input.dim() - 1)
                and input.dim() == 4
            ):
                weight = float(getattr(lm, "_adaptvis_weight", 1.0))
                if weight != 1.0:
                    input = _scale_attention_logits_if_needed(input, lm, weight)

            return _safe_softmax_call(
                _ORIG_SOFTMAX,
                input,
                dim=dim,
                _stacklevel=_stacklevel,
                dtype=dtype,
            )

        F.softmax = patched_softmax
        F.scaled_dot_product_attention = _manual_scaled_dot_product_attention
        _ATTN_PATCHED = True

    F.softmax.language_model = language_model
    F.scaled_dot_product_attention.language_model = language_model

    if verbose:
        print("Installed PRE-SOFTMAX AdaptVis patch on LLaVA language_model.forward: F.softmax + F.scaled_dot_product_attention.")

def reset_adaptvis_counters(language_model, weight):
    language_model._adaptvis_enable = True
    language_model._adaptvis_weight = float(weight)
    language_model._adaptvis_prefill_only = True

    language_model._adaptvis_scaled_calls = 0
    language_model._adaptvis_scaled_pos_count = 0
    language_model._adaptvis_scaled_mass_before = 0.0
    language_model._adaptvis_scaled_mass_after_raw = 0.0

    language_model._adaptvis_modified_calls = 0
    language_model._adaptvis_modified_pos_count = 0
    language_model._adaptvis_logit_sum_before = 0.0
    language_model._adaptvis_logit_sum_after = 0.0

    language_model._adaptvis_image_positions = None
    language_model._adaptvis_last_image_positions = None


def set_image_positions_from_input_ids(language_model, input_ids, image_token_id, image_seq_len=576):
    """
    HF LLaVA input_ids usually contain exactly one <image> token.
    The model internally expands that single token into image_seq_len visual
    embeddings before calling the language model.

    Therefore:
      - if input has one <image> token at position p, scale merged positions p:p+image_seq_len
      - if input already has many image tokens, use those positions directly
    """
    ids = input_ids.detach()
    positions = []

    for b in range(ids.shape[0]):
        pos = torch.nonzero(ids[b] == int(image_token_id), as_tuple=False).view(-1)

        if pos.numel() == 1:
            start = int(pos.item())
            expanded = torch.arange(start, start + int(image_seq_len), dtype=torch.long)
            positions.append(expanded)
        else:
            positions.append(pos.detach().cpu())

    language_model._adaptvis_image_positions = positions
    language_model._adaptvis_last_image_positions = positions
    return positions


def choose_adapt_weight(confidence, threshold, weight1, weight2, adapt_rule):
    if confidence is None:
        confidence = 0.0

    if adapt_rule == "paper":
        # Same rule as the InternVL script:
        # lower first-token confidence -> weight1; otherwise weight2.
        return weight1 if confidence < threshold else weight2
    if adapt_rule == "high_w1":
        return weight1 if confidence >= threshold else weight2
    if adapt_rule == "high_w2":
        return weight2 if confidence >= threshold else weight1
    raise ValueError(adapt_rule)


@torch.no_grad()
def generate_with_scores(model, processor, tokenizer, inputs, image_token_id, weight, image_seq_len=576, max_new_tokens=16, capture_scores=True):
    language_model = get_language_model(model)
    reset_adaptvis_counters(language_model, weight=weight)
    set_image_positions_from_input_ids(language_model, inputs["input_ids"], image_token_id, image_seq_len=image_seq_len)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    if capture_scores:
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True

    out = model.generate(**gen_kwargs)

    if capture_scores:
        sequences = out.sequences
    else:
        sequences = out

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = sequences[:, prompt_len:]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    score_info = {
        "first_token_max_prob": None,
        "first_token_max_prob_round": None,
        "first_token_argmax_id": None,
        "first_token_argmax_text": None,
        "first_token_top5": None,
    }

    if capture_scores and hasattr(out, "scores") and out.scores is not None and len(out.scores) > 0:
        logits0 = out.scores[0][0].float()
        probs0 = F.softmax(logits0, dim=-1)
        max_prob, max_id = torch.max(probs0, dim=-1)

        top_probs, top_ids = torch.topk(probs0, k=min(5, probs0.numel()))
        top5 = []
        for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
            try:
                tok = tokenizer.decode([int(tid)])
            except Exception:
                tok = str(tid)
            top5.append({
                "token_id": int(tid),
                "token": tok,
                "prob": float(p),
            })

        max_id_int = int(max_id.item())
        try:
            max_text = tokenizer.decode([max_id_int])
        except Exception:
            max_text = str(max_id_int)

        score_info = {
            "first_token_max_prob": float(max_prob.item()),
            "first_token_max_prob_round": round(float(max_prob.item()), 2),
            "first_token_argmax_id": max_id_int,
            "first_token_argmax_text": max_text,
            "first_token_top5": top5,
        }

    image_positions = getattr(language_model, "_adaptvis_last_image_positions", None)
    if image_positions is None:
        image_pos_lens = None
    else:
        image_pos_lens = [int(p.numel()) if torch.is_tensor(p) else len(p) for p in image_positions]

    trace_info = {
        "image_pos_lens": image_pos_lens,

        # Compatibility names:
        "scaled_calls": int(getattr(language_model, "_adaptvis_scaled_calls", 0)),
        "scaled_pos_count": int(getattr(language_model, "_adaptvis_scaled_pos_count", 0)),
        "scaled_mass_before": float(getattr(language_model, "_adaptvis_scaled_mass_before", 0.0)),
        "scaled_mass_after_raw": float(getattr(language_model, "_adaptvis_scaled_mass_after_raw", 0.0)),

        # Explicit pre-softmax-logit names:
        "modified_calls": int(getattr(language_model, "_adaptvis_modified_calls", 0)),
        "modified_pos_count": int(getattr(language_model, "_adaptvis_modified_pos_count", 0)),
        "logit_sum_before": float(getattr(language_model, "_adaptvis_logit_sum_before", 0.0)),
        "logit_sum_after": float(getattr(language_model, "_adaptvis_logit_sum_after", 0.0)),
        "intervention": "pre_softmax_logit_scaling_last_query_prefill_only",
    }

    return response, score_info, trace_info


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--subset", default="A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--method", choices=["base", "scaling_vis", "adapt_vis"], default="base")

    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--weight1", type=float, default=0.5)
    parser.add_argument("--weight2", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--adapt_rule", choices=["paper", "high_w1", "high_w2"], default="paper")

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=16)

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--debug_patch", action="store_true")
    args = parser.parse_args()

    setup_cache()
    Path("outputs").mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = Controlled_Images(
        image_preprocess=None,
        root_dir=args.root_dir,
        download=True,
        subset=args.subset,
    )

    prompt_file = f"prompts/{args.dataset}_with_answer_{args.option}_options.jsonl"
    prompts, answers = [], []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["question"])
            answers.append(r["answer"])

    n_total = len(dataset.dataset)
    if args.limit > 0:
        n_total = min(n_total, args.limit)

    print("dataset total:", len(dataset.dataset))
    print("running n:", n_total)
    print("model_id:", args.model_id)
    print("method:", args.method)
    print("weight:", args.weight)
    print("weight1:", args.weight1)
    print("weight2:", args.weight2)
    print("threshold:", args.threshold)
    print("adapt_rule:", args.adapt_rule)

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print("Loading processor/tokenizer...")
    processor, tokenizer = load_llava_processor(args.model_id)

    try:
        tokenizer.padding_side = "right"
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "right"
    except Exception:
        pass

    print("Loading LLaVA model...")
    model = load_llava_model(
        model_id=args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        load_in_8bit=args.load_in_8bit,
        attn_implementation="eager",
    )

    if torch.cuda.is_available() and not args.load_in_8bit and args.device_map is None:
        model = model.cuda()

    model.requires_grad_(False)

    language_model = get_language_model(model)
    install_adaptvis_softmax_patch(language_model, verbose=True)

    # Print a small attention-class summary for sanity.
    attn_classes = {}
    for name, module in language_model.named_modules():
        cls = module.__class__.__name__
        if "Attention" in cls or "Sdpa" in cls or "Flash" in cls:
            attn_classes[cls] = attn_classes.get(cls, 0) + 1
    print("language_model attention classes:", attn_classes)

    image_token_id = get_image_token_id(model, processor, tokenizer)
    image_seq_len = get_expected_image_seq_len(model, processor)
    print("image_token_id:", image_token_id)
    print("expected_image_seq_len:", image_seq_len)

    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []

    correct = 0
    unparsed = 0

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_internstyle_prelogit_promptfix_pad77_{model_tag}_{args.dataset}_{args.method}"
        f"_w{args.weight}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}_rule{args.adapt_rule}"
    )
    out_json = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    device = get_model_device(model)

    for i in tqdm(range(n_total), total=n_total):
        d = dataset.dataset[i]
        image_path = resolve_image_path(d["image_path"])
        caption_options = d["caption_options"]
        prompt = clean_prompt_for_vlm(prompts[i])
        gold = answers[i]

        inputs, question = build_inputs(
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            image_path=image_path,
            prompt=prompt,
            device=device,
        )

        if i == 0:
            print("first_question_repr:", repr(question))

        if i == 0:
            print("first_input_ids_shape:", tuple(inputs["input_ids"].shape))
            print("first_attention_mask_sum:", int(inputs["attention_mask"].sum().item()) if "attention_mask" in inputs else None)

        img_pos_count = int((inputs["input_ids"] == image_token_id).sum().item())

        probe_generation = None
        probe_score = None
        probe_trace = None

        if args.method == "adapt_vis":
            probe_generation, probe_score, probe_trace = generate_with_scores(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                inputs=inputs,
                image_token_id=image_token_id,
                weight=1.0,
                image_seq_len=image_seq_len,
                max_new_tokens=args.max_new_tokens,
                capture_scores=True,
            )
            confidence = probe_score["first_token_max_prob_round"]
            chosen_weight = choose_adapt_weight(
                confidence=confidence,
                threshold=args.threshold,
                weight1=args.weight1,
                weight2=args.weight2,
                adapt_rule=args.adapt_rule,
            )
        elif args.method == "base":
            chosen_weight = 1.0
        else:
            chosen_weight = args.weight

        response, final_score, final_trace = generate_with_scores(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            inputs=inputs,
            image_token_id=image_token_id,
            weight=chosen_weight,
            image_seq_len=image_seq_len,
            max_new_tokens=args.max_new_tokens,
            capture_scores=True,
        )

        pred_prep = parse_prep(response)
        pred_idx, is_correct_bool = pred_to_option_index(
            pred_prep,
            gold,
            caption_options,
        )

        if pred_prep is None:
            unparsed += 1

        scores[i, 0, pred_idx] = 1.0
        is_correct = int(is_correct_bool)
        correct += is_correct

        rec = {
            "index": i,
            "image_path": image_path,
            "prompt": prompt,
            "question": question,
            "gold": gold,
            "generation": response,
            "pred_prep": pred_prep,
            "pred_idx": int(pred_idx),
            "correct": bool(is_correct),
            "chosen_weight": float(chosen_weight),

            "image_token_count": img_pos_count,

            "probe_generation": probe_generation,
            "probe_first_token_max_prob": None if probe_score is None else probe_score["first_token_max_prob"],
            "probe_first_token_max_prob_round": None if probe_score is None else probe_score["first_token_max_prob_round"],
            "probe_first_token_argmax_id": None if probe_score is None else probe_score["first_token_argmax_id"],
            "probe_first_token_argmax_text": None if probe_score is None else probe_score["first_token_argmax_text"],
            "probe_first_token_top5": None if probe_score is None else probe_score["first_token_top5"],
            "probe_trace": probe_trace,

            "final_first_token_max_prob": final_score["first_token_max_prob"],
            "final_first_token_max_prob_round": final_score["first_token_max_prob_round"],
            "final_first_token_argmax_id": final_score["first_token_argmax_id"],
            "final_first_token_argmax_text": final_score["first_token_argmax_text"],
            "final_first_token_top5": final_score["first_token_top5"],
            "final_trace": final_trace,

            "caption_options": caption_options,
        }
        records.append(rec)

        if args.print_every > 0 and (i % args.print_every == 0):
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("image_token_count:", img_pos_count)
            print("gold:", gold)
            if probe_score is not None:
                print("probe_generation:", probe_generation)
                print("probe_first_token_max_prob:", probe_score["first_token_max_prob"])
                print("probe_first_token_max_prob_round:", probe_score["first_token_max_prob_round"])
                print("probe_trace:", probe_trace)
            print("chosen_weight:", chosen_weight)
            print("generation:", response)
            print("pred:", pred_prep, "pred_idx:", pred_idx, "correct:", bool(is_correct))
            print("final_first_token_max_prob:", final_score["first_token_max_prob"])
            print("final_first_token_argmax_text:", final_score["first_token_argmax_text"])
            print("final_trace:", final_trace)
            print("running acc:", correct / (i + 1), "unparsed:", unparsed)

        if (i + 1) % 25 == 0:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    direct_acc = correct / max(n_total, 1)

    summary = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "method": args.method,
        "weight": args.weight,
        "weight1": args.weight1,
        "weight2": args.weight2,
        "threshold": args.threshold,
        "adapt_rule": args.adapt_rule,
        "n": n_total,
        "direct_acc": direct_acc,
        "unparsed": unparsed,
        "max_new_tokens": args.max_new_tokens,
        "out_json": str(out_json),
        "prob_definition": "first generated token full-vocab max softmax probability",
        "adaptvis_definition": "InternVL-style wrapper but with main_aro-like pre-softmax attention-logit scaling: last query -> image-token positions, prefill only",
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDirect acc:", direct_acc)
    print("unparsed:", unparsed)
    print("saved records:", out_json)
    print("saved summary:", out_summary)

    if n_total == len(dataset.dataset):
        print("\nRunning Controlled_Images evaluator...")
        dataset.evaluate_scores(
            scores=scores,
            path="outputs",
            dataset=args.dataset,
            model=model_tag,
            method=args.method,
            weight=args.weight if args.method == "scaling_vis" else 1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
