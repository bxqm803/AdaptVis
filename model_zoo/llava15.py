import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import random
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor
import pdb
# import probe_llava
from .llava import LlavaForConditionalGeneration, LlavaForConditionalGenerationScal

import torch
import torch.nn.functional as F
from PIL import Image
import requests
import json
import os
from collections import Counter
# from model_zoo.utils import normalize_answer,chat_completion_request,run_conversation

from PIL import Image
import math
MODEL = 'llava-hf/llava-1.5-7b-hf'

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import (
    SampleOutput,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
)
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm


def _add_weight_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    # keys:Optional[torch.Tensor] = None,
    weight: Optional[float] = None,
    adjust_method: Optional[str] = None,
    pos: Optional[torch.Tensor] = None,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    before = () if (return_dict_in_generate) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]

    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if 'Scal' not in str(type(self)):
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            outputs = self(
                **model_inputs,
                weight=weight,
                adjust_method=adjust_method,
                pos=pos,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def change_greedy_to_add_weight():
    transformers.generation.utils.GenerationMixin._greedy_search = _add_weight_greedy_search


def _safe_float_tag(x):
    """Make a float safe for filenames: 1.5 -> 1p5, 1.0 -> 1p0."""
    try:
        s = f"{float(x):g}"
    except Exception:
        s = str(x)
    return s.replace("-", "m").replace(".", "p")


def _raw_generation_correct(gold, gen):
    """Original substring rule used by main ARO generation evaluation."""
    if isinstance(gold, list):
        gold = gold[0] if gold else ""
    gold = str(gold)
    gen = str(gen)
    return bool(
        (gold in gen or gold.lower() in gen.lower())
        and not (gold.lower() == 'on' and 'front' in gen.strip().lower())
    )


def _feature_alpha_name(alpha):
    try:
        a = float(alpha)
    except Exception:
        return "unknown"
    if abs(a - 1.0) < 1e-6:
        return "base"
    if abs(a - 0.5) < 1e-6:
        return "low"
    if abs(a - 1.5) < 1e-6:
        return "high"
    return f"alpha{_safe_float_tag(a)}"


def _parse_relation_objects(prompt):
    """Parse prompts like: Where is/are the bowl in relation to the armchair? ..."""
    import re

    p = str(prompt).replace("\n", " ")
    m = re.search(
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+in relation to\s+(?:the\s+)?(.+?)\?",
        p,
        flags=re.IGNORECASE,
    )
    if not m:
        return "", ""

    obj1 = m.group(1).strip()
    obj2 = m.group(2).strip()

    obj1 = re.sub(r"\s+Answer\s+with.*$", "", obj1, flags=re.IGNORECASE).strip()
    obj2 = re.sub(r"\s+Answer\s+with.*$", "", obj2, flags=re.IGNORECASE).strip()

    return obj1, obj2


def _find_subsequence(seq, sub):
    if not sub:
        return []
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return list(range(i, i + m))
    return []


def _locate_object_raw_positions(input_ids_list, valid_positions, tokenizer, obj):
    """Return raw input_ids positions corresponding to an object phrase, best effort."""
    if not obj:
        return []

    valid_ids = [input_ids_list[p] for p in valid_positions]
    variants = []
    obj = str(obj).strip()

    for v in [obj, " " + obj, "the " + obj, " the " + obj, obj + " "]:
        if v and v not in variants:
            variants.append(v)

    best = []
    for v in variants:
        try:
            ids = tokenizer(v, add_special_tokens=False).input_ids
        except Exception:
            ids = []
        ids = [int(x) for x in ids]
        found = _find_subsequence(valid_ids, ids)
        if found:
            best = [valid_positions[i] for i in found]
            break

    return best


def _extract_first_step_last_hidden(generate_output):
    """Get final-layer hidden state for the first generation step, i.e. prompt-only forward."""
    hs = None

    if hasattr(generate_output, "hidden_states"):
        hs = generate_output.hidden_states
    elif isinstance(generate_output, dict) and "hidden_states" in generate_output:
        hs = generate_output["hidden_states"]

    if hs is None:
        raise RuntimeError("generate_output.hidden_states is None. Did you pass output_hidden_states=True?")

    # Generation output: tuple over generation steps; each step is tuple over layers.
    first_step = hs[0]
    last_layer = first_step[-1]

    # batch size 1
    return last_layer[0]


def _normalize_rows(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _get_exact_adaptvis_masks(model, hidden_len):
    """
    Read the exact merged-sequence masks produced inside LlavaForConditionalGenerationScal.forward().

    Required companion patch in model_zoo/llava/modeling_llava_scal.py:
        self._adaptvis_last_image_id = image_id.detach().cpu()
        self._adaptvis_last_attention_mask = attention_mask.detach().cpu()

    image_id is exactly the mask later passed into language_model(..., keys=image_id),
    so this is the same visual-token span used by AdaptVis attention scaling.
    """
    image_id = getattr(model, "_adaptvis_last_image_id", None)
    merged_attention_mask = getattr(model, "_adaptvis_last_attention_mask", None)

    if image_id is None:
        raise RuntimeError(
            "model._adaptvis_last_image_id not found. "
            "You must patch model_zoo/llava/modeling_llava_scal.py first."
        )

    image_id = image_id.detach().cpu().bool()

    if image_id.dim() == 2:
        image_mask = image_id[0]
    elif image_id.dim() == 1:
        image_mask = image_id
    else:
        raise RuntimeError(f"Unexpected image_id shape: {tuple(image_id.shape)}")

    if image_mask.numel() != hidden_len:
        raise RuntimeError(
            f"image_id length mismatch: image_mask={image_mask.numel()} hidden_len={hidden_len}"
        )

    if merged_attention_mask is not None:
        merged_attention_mask = merged_attention_mask.detach().cpu().bool()

        if merged_attention_mask.dim() == 2:
            valid_mask = merged_attention_mask[0]
        elif merged_attention_mask.dim() == 1:
            valid_mask = merged_attention_mask
        else:
            raise RuntimeError(f"Unexpected merged attention mask shape: {tuple(merged_attention_mask.shape)}")

        if valid_mask.numel() != hidden_len:
            raise RuntimeError(
                f"merged attention mask length mismatch: valid_mask={valid_mask.numel()} hidden_len={hidden_len}"
            )
    else:
        valid_mask = torch.ones_like(image_mask, dtype=torch.bool)

    text_mask = valid_mask & (~image_mask)

    image_positions = torch.where(image_mask)[0].tolist()
    text_positions = torch.where(text_mask)[0].tolist()

    if len(image_positions) == 0:
        raise RuntimeError("Exact image_id mask has zero visual-token positions.")

    return image_mask, text_mask, image_positions, text_positions


def _save_hidden_similarity_from_generate(
    *,
    model,
    tokenizer,
    generate_output,
    single_input,
    prompt,
    gold,
    sample_id,
    alpha,
    selected_weight,
    uncertainty,
    feature_dir,
):
    """
    Save last-layer hidden-state features from the first generation forward.

    Important:
    - image_hidden is sliced by model._adaptvis_last_image_id.
    - That mask is produced inside modeling_llava_scal.py after <image> is expanded
      into visual tokens.
    - It is the same mask passed to language_model(..., keys=image_id), so the
      saved image_hidden span is exactly aligned with AdaptVis scaling.
    """
    import os

    os.makedirs(feature_dir, exist_ok=True)

    H = _extract_first_step_last_hidden(generate_output).detach().float().cpu()
    hidden_len, hidden_dim = H.shape

    image_mask, text_mask, image_positions, text_merged_positions = _get_exact_adaptvis_masks(
        model=model,
        hidden_len=hidden_len,
    )

    H_img = H[image_mask]
    H_text = H[text_mask]

    num_img_tokens = int(H_img.shape[0])
    if num_img_tokens <= 0:
        raise RuntimeError("No image hidden states found from exact image mask.")

    # Original raw input_ids before image-token expansion.
    input_ids = single_input["input_ids"][0].detach().cpu().tolist()
    attention_mask = single_input.get("attention_mask", None)

    if attention_mask is not None:
        attention_mask_list = attention_mask[0].detach().cpu().tolist()
        raw_valid_positions = [i for i, m in enumerate(attention_mask_list) if int(m) == 1]
    else:
        pad_id = tokenizer.pad_token_id
        raw_valid_positions = [
            i for i, t in enumerate(input_ids)
            if pad_id is None or int(t) != int(pad_id)
        ]

    # Use the model config image token index. Fallback to the original repo's 32001.
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32001

    raw_text_positions = []
    raw_image_positions = []

    for p in raw_valid_positions:
        tid = int(input_ids[p])
        if tid == int(image_token_index):
            raw_image_positions.append(p)
        else:
            raw_text_positions.append(p)

    # If tokenizer/processor version hides <image> in raw input_ids, keep all valid raw tokens as text.
    # This does not affect image_hidden, because image_hidden is taken from exact image_id mask.
    if len(raw_text_positions) != len(text_merged_positions):
        if len(raw_valid_positions) == len(text_merged_positions):
            raw_text_positions = list(raw_valid_positions)
            raw_image_positions = []
        else:
            raise RuntimeError(
                "Text token count mismatch between raw input_ids and merged hidden states: "
                f"raw_text={len(raw_text_positions)} merged_text={len(text_merged_positions)} "
                f"raw_valid={len(raw_valid_positions)} image_token_index={image_token_index} "
                f"raw_image_positions={raw_image_positions} hidden_len={hidden_len} "
                f"num_img_tokens={num_img_tokens}"
            )

    raw_to_merged = {
        int(rp): int(mp)
        for rp, mp in zip(raw_text_positions, text_merged_positions)
    }

    obj1, obj2 = _parse_relation_objects(prompt)

    obj1_raw = _locate_object_raw_positions(input_ids, raw_text_positions, tokenizer, obj1)
    obj2_raw = _locate_object_raw_positions(input_ids, raw_text_positions, tokenizer, obj2)

    obj1_merged = [raw_to_merged[p] for p in obj1_raw if p in raw_to_merged]
    obj2_merged = [raw_to_merged[p] for p in obj2_raw if p in raw_to_merged]

    H_obj1 = H[obj1_merged].mean(dim=0) if obj1_merged else torch.zeros((hidden_dim,), dtype=H.dtype)
    H_obj2 = H[obj2_merged].mean(dim=0) if obj2_merged else torch.zeros((hidden_dim,), dtype=H.dtype)
    H_last = H[text_merged_positions[-1]] if text_merged_positions else H[-1]

    # Cosine similarities: object/text token hidden states vs image patch hidden states.
    H_img_n = _normalize_rows(H_img)
    H_text_n = _normalize_rows(H_text) if H_text.numel() else H_text

    obj1_n = H_obj1 / (H_obj1.norm() + 1e-6)
    obj2_n = H_obj2 / (H_obj2.norm() + 1e-6)

    sim_obj1 = torch.matmul(H_img_n, obj1_n).numpy()
    sim_obj2 = torch.matmul(H_img_n, obj2_n).numpy()

    sim_text_to_img = (
        torch.matmul(H_text_n, H_img_n.T).numpy()
        if H_text.numel()
        else np.zeros((0, num_img_tokens), dtype=np.float32)
    )

    # First-step logits for answer distribution.
    if hasattr(generate_output, "scores") and generate_output.scores is not None and len(generate_output.scores) > 0:
        first_logits = generate_output.scores[0][0].detach().float().cpu()
    elif isinstance(generate_output, dict) and generate_output.get("scores", None) is not None and len(generate_output["scores"]) > 0:
        first_logits = generate_output["scores"][0][0].detach().float().cpu()
    else:
        first_logits = torch.empty((0,), dtype=torch.float32)

    answer_words = ["left", "right", "on", "under", "Left", "Right", "On", "Under"]
    answer_token_ids = []
    answer_token_labels = []

    for w in answer_words:
        for txt in [w, " " + w]:
            ids = tokenizer(txt, add_special_tokens=False).input_ids
            if len(ids) == 1:
                answer_token_ids.append(int(ids[0]))
                answer_token_labels.append(txt)

    if first_logits.numel():
        answer_logits = np.array([float(first_logits[i]) for i in answer_token_ids], dtype=np.float32)
    else:
        answer_logits = np.zeros((len(answer_token_ids),), dtype=np.float32)

    try:
        tokens = tokenizer.convert_ids_to_tokens([input_ids[p] for p in raw_text_positions])
    except Exception:
        tokens = [str(input_ids[p]) for p in raw_text_positions]

    alpha_name = _feature_alpha_name(alpha)
    out_path = os.path.join(
        feature_dir,
        f"sid{int(sample_id):04d}_{alpha_name}_alpha{_safe_float_tag(alpha)}.npz",
    )

    np.savez_compressed(
        out_path,
        sample_id=np.array(int(sample_id), dtype=np.int32),
        alpha=np.array(float(alpha), dtype=np.float32),
        alpha_name=np.array(alpha_name),
        selected_weight=np.array(float(selected_weight) if selected_weight is not None else np.nan, dtype=np.float32),
        uncertainty=np.array(float(uncertainty) if uncertainty is not None else np.nan, dtype=np.float32),
        prompt=np.array(str(prompt)),
        gold=np.array(str(gold)),
        obj1=np.array(str(obj1)),
        obj2=np.array(str(obj2)),

        hidden_len=np.array(int(hidden_len), dtype=np.int32),
        hidden_dim=np.array(int(hidden_dim), dtype=np.int32),

        image_positions=np.array(image_positions, dtype=np.int32),
        text_merged_positions=np.array(text_merged_positions, dtype=np.int32),
        num_img_tokens=np.array(int(num_img_tokens), dtype=np.int32),

        raw_valid_positions=np.array(raw_valid_positions, dtype=np.int32),
        raw_text_positions=np.array(raw_text_positions, dtype=np.int32),
        raw_image_positions=np.array(raw_image_positions, dtype=np.int32),
        image_token_index=np.array(int(image_token_index), dtype=np.int32),

        obj1_raw_positions=np.array(obj1_raw, dtype=np.int32),
        obj2_raw_positions=np.array(obj2_raw, dtype=np.int32),
        obj1_merged_positions=np.array(obj1_merged, dtype=np.int32),
        obj2_merged_positions=np.array(obj2_merged, dtype=np.int32),
        tokens=np.array(tokens),

        image_hidden=H_img.half().numpy(),
        text_hidden=H_text.half().numpy(),
        obj1_hidden=H_obj1.half().numpy(),
        obj2_hidden=H_obj2.half().numpy(),
        last_prompt_hidden=H_last.half().numpy(),

        sim_obj1_to_img=sim_obj1.astype(np.float32),
        sim_obj2_to_img=sim_obj2.astype(np.float32),
        sim_text_to_img=sim_text_to_img.astype(np.float32),
        sim_obj1_top10_idx=np.argsort(-sim_obj1)[:10].astype(np.int32),
        sim_obj2_top10_idx=np.argsort(-sim_obj2)[:10].astype(np.int32),

        answer_token_ids=np.array(answer_token_ids, dtype=np.int32),
        answer_token_labels=np.array(answer_token_labels),
        answer_logits=answer_logits,
    )

    return out_path


class LlavaWrapper:
    def __init__(self, root_dir, device, method):

        if method == 'scaling_vis' or method == 'adapt_vis':
            self.model = LlavaForConditionalGenerationScal.from_pretrained(
                MODEL,
                revision='a272c74',
                cache_dir=root_dir,
                ignore_mismatched_sizes=True,
            ).eval().to(device)

        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL,
                revision='a272c74',
                cache_dir=root_dir,
                ignore_mismatched_sizes=True,
            ).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            MODEL,
            revision='a272c74',
            cache_dir=root_dir,
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            MODEL,
            revision='a272c74',
            cache_dir=root_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL,
            revision='a272c74',
            cache_dir=root_dir,
        )

        self.device = device

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=64, normalize=False):
        num_text = len(texts)
        text_embeds = []

        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i: min(num_text, i + text_batch_size)]
            text_input = self.tokenizer(
                text=text,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
            ).to(self.device)

            text_feats = self.model.llava.get_text_features(**text_input).cpu().numpy()[:, 0, :].to(self.device)

            if normalize:
                text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)

            text_embeds.append(text_feats)

        return np.concatenate(text_embeds, axis=0)

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []

        for batch in tqdm(image_loader):
            images = batch["image"]
            inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
            image_feats = self.model.llava.get_image_features(**inputs).cpu().numpy()[:, 0, :]

            if normalize:
                image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)

            image_embeds.append(image_feats)

        return np.concatenate(image_embeds, axis=0)

    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        text_embeds = self.get_text_embeddings(texts, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        return scores

    @torch.no_grad()
    def get_out_scores_wh_batched(self, dataset, joint_loader, method, weight, option, threshold, weight1, weight2):

        scores = []
        index_of_total = 0
        acc = 0
        correct_id = []

        qst_ans_file = f'prompts/{dataset}_with_answer_{option}_options.jsonl'

        with open(qst_ans_file, 'r') as file:
            prompt_list = []
            answer_list = []
            first_prompt_list = []
            second_prompt_list = []

            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        SAMPLE = False
        TEST = os.getenv('TEST_MODE', 'False') == 'True'
        total_data_count = len(prompt_list)

        if SAMPLE:
            idx_file_path = f'./output/sampled_idx_{dataset}.npy'

            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
                sampled_indices.sort()
                np.save(idx_file_path, np.array(sampled_indices))

            if TEST:
                all_indices = set(range(total_data_count))
                unsampled_indices = list(all_indices - set(sampled_indices))
                unsampled_indices.sort()
                sampled_indices = unsampled_indices

            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        save_attn_dir = f"./output/{dataset}_weight{weight:.2f}"
        os.makedirs(save_attn_dir, exist_ok=True)

        run_tag = (
            f"w{_safe_float_tag(weight)}_"
            f"w1{_safe_float_tag(weight1)}_"
            f"w2{_safe_float_tag(weight2)}_"
            f"thr{_safe_float_tag(threshold)}"
        )
        output_file_path = f'./output/results1.5_{dataset}_{method}_{run_tag}_{option}option_{TEST}.json'
        output_score_file = output_file_path.replace(".json", "_scores.json")

        print("[RUN TAG]", run_tag)
        print("[RESULT JSON]", output_file_path)
        print("[SCORE JSON]", output_score_file)

        save_hidden_features = os.getenv("SAVE_HIDDEN_FEATURES", "False") == "True"
        hidden_feature_dir = os.getenv(
            "HIDDEN_FEATURE_DIR",
            f"./output/hidden_features_during_generation/{dataset}_{method}_{run_tag}_{option}option_{TEST}",
        )

        if save_hidden_features:
            os.makedirs(hidden_feature_dir, exist_ok=True)
            print("[SAVE HIDDEN FEATURES]", hidden_feature_dir)

        results = []

        for batch in tqdm(joint_loader):
            batch_scores = []

            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            for i_option in batch["image_options"]:
                im_scores = []

                for _ in i_option:
                    prompt = prompt_list[index_of_total]

                    single_input = self.processor(
                        text=prompt,
                        images=_,
                        padding="max_length",
                        return_tensors="pt",
                        max_length=77,
                    ).to(self.device)

                    keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]

                    selected_weight = None
                    uncertainty = None

                    if method == 'scaling_vis':

                        change_greedy_to_add_weight()
                        selected_weight = weight

                        output = self.model.generate(
                            **single_input,
                            keys=keys,
                            weight=weight,
                            max_new_tokens=100,
                            output_scores=True,
                            output_hidden_states=save_hidden_features,
                            return_dict_in_generate=True,
                        )

                        uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True,
                        )

                    elif method == 'adapt_vis':
                        change_greedy_to_add_weight()

                        base_output = self.model.generate(
                            **single_input,
                            weight=1.0,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                        uncertainty = np.round(float(max(torch.nn.functional.softmax(base_output['scores'][0], dim=-1)[0])), 2)
                        print("[BASE CONFIDENCE]", uncertainty, "threshold=", threshold)

                        if uncertainty < threshold:
                            selected_weight = weight1
                        else:
                            selected_weight = weight2

                        output = self.model.generate(
                            **single_input,
                            keys=keys,
                            weight=selected_weight,
                            max_new_tokens=100,
                            output_scores=True,
                            output_hidden_states=save_hidden_features,
                            return_dict_in_generate=True,
                        )

                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True,
                        )

                    else:
                        output = self.model.generate(
                            **single_input,
                            max_new_tokens=100,
                            output_scores=True,
                            output_hidden_states=save_hidden_features,
                            return_dict_in_generate=True,
                        )

                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True,
                        )
                        uncertainty = np.round(float(max(output['scores'][0][0])), 2)

                    gold = answer_list[index_of_total][0]

                    feature_path = None
                    if save_hidden_features:
                        try:
                            feature_path = _save_hidden_similarity_from_generate(
                                model=self.model,
                                tokenizer=self.tokenizer,
                                generate_output=output,
                                single_input=single_input,
                                prompt=prompt,
                                gold=gold,
                                sample_id=index_of_total,
                                alpha=selected_weight if selected_weight is not None else weight,
                                selected_weight=selected_weight,
                                uncertainty=uncertainty,
                                feature_dir=hidden_feature_dir,
                            )
                            print("[FEATURE SAVED]", feature_path)
                        except Exception as e:
                            print("[FEATURE SAVE ERROR]", repr(e))

                    is_correct = _raw_generation_correct(gold, gen)

                    print(
                        f"[SAMPLE {index_of_total}] selected_weight={selected_weight} "
                        f"correct={is_correct}\n"
                        f"Prompt: {prompt}\nGeneration: {gen}\nGolden: {gold}\n"
                    )

                    result = {
                        "sample_id": index_of_total,
                        "Prompt": prompt,
                        "Generation": gen,
                        "RawGeneration": gen,
                        "Golden": gold,
                        "Correct": is_correct,
                        "RawGenerationCorrect": is_correct,
                        "FeaturePath": feature_path,
                        "method": method,
                        "weight": float(weight) if weight is not None else None,
                        "weight1": float(weight1) if weight1 is not None else None,
                        "weight2": float(weight2) if weight2 is not None else None,
                        "threshold": float(threshold) if threshold is not None else None,
                        "selected_weight": float(selected_weight) if selected_weight is not None else None,
                        "uncertainty": float(uncertainty) if uncertainty is not None else None,
                    }
                    results.append(result)

                    c_option = batch["caption_options"]

                    if len(list(c_option)) == 4:
                        if is_correct:
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0, 0, 0]
                        else:
                            answers = [0, 0, 1, 0]

                    elif len(list(c_option)) == 2:
                        if is_correct:
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0]
                        else:
                            answers = [0, 1]

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    index_of_total += 1

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            print("Saving results to", output_file_path)
            with open(output_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)

            running_acc = acc / index_of_total if index_of_total else 0.0
            print("[RUNNING ACC]", acc, index_of_total, running_acc)

        final_acc = acc / index_of_total if index_of_total else 0.0
        print("[FINAL RAW GENERATION ACC]", final_acc)

        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump(
                {
                    "acc": final_acc,
                    "correct_id": correct_id,
                    "num_correct": acc,
                    "num_total": index_of_total,
                    "run_tag": run_tag,
                    "method": method,
                    "weight": float(weight) if weight is not None else None,
                    "weight1": float(weight1) if weight1 is not None else None,
                    "weight2": float(weight2) if weight2 is not None else None,
                    "threshold": float(threshold) if threshold is not None else None,
                    "result_json": output_file_path,
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        all_scores = np.concatenate(scores, axis=0)

        if dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
            return (all_scores, [])
        else:
            return (acc / index_of_total, correct_id)

    @torch.no_grad()
    def get_judge_scores_vsr_batched(self, dataset, joint_loader, method, weight, threshold, weight1, weight2):

        index = 0
        TP, TN, FP, FN = 0, 0, 0, 0

        save_attn_dir = f"/home/user/shiqi/mmlm_mech/whatsup_vlms/outputs/{dataset}_weight{weight:.2f}"
        if not os.path.exists(save_attn_dir):
            print("Creating directory for saving attention maps:", save_attn_dir)
            os.makedirs(save_attn_dir)

        index_of_total = 0
        results = []

        for batch in tqdm(joint_loader):
            batch_scores = []

            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            for i_option in batch["image_options"]:
                im_scores = []

                for c_option in batch["caption_options"]:
                    prompt = "User: <image>\n Determine whether the description about the spatial relationship is correct or not. Answer with yes or no: "
                    qst = [prompt] * len(list(c_option))
                    end_fix = [" Assistant:"] * len(list(c_option))
                    concatenated_list = [s1 + s2 + s3 for s1, s2, s3 in zip(qst, c_option, end_fix)]

                    for idx, text in enumerate(concatenated_list):
                        single_input = self.processor(
                            text=text,
                            images=list(i_option)[idx],
                            padding="max_length",
                            return_tensors="pt",
                            max_length=77,
                        ).to(self.device)

                        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]

                        if method == 'scaling_vis':
                            change_greedy_to_add_weight()
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                            gen = self.processor.decode(
                                output[0][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )

                        elif method == 'adapt_vis':
                            change_greedy_to_add_weight()

                            output = self.model.generate(
                                **single_input,
                                weight=1.0,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            gen = self.processor.decode(
                                output['sequences'][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )
                            uncertainty = np.round(float(max(output['scores'][0][0])), 2)

                            if uncertainty < threshold:
                                output = self.model.generate(
                                    **single_input,
                                    keys=keys,
                                    weight=weight1,
                                    max_new_tokens=100,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )
                            else:
                                output = self.model.generate(
                                    **single_input,
                                    keys=keys,
                                    weight=weight2,
                                    max_new_tokens=100,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )

                            gen = self.processor.decode(
                                output[0][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )

                        else:
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                            gen = self.processor.decode(
                                output[0][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )

                        label = int(batch['labels'][0][idx])
                        if label == 1:
                            TP += 1 if 'Yes' in gen else 0
                            FN += 1 if 'Yes' not in gen else 0
                        else:
                            TN += 1 if 'No' in gen else 0
                            FP += 1 if 'No' not in gen else 0

                        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

                        gold = 'Yes' if label == 1 else 'No'
                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": gold,
                            "Uncertainty": uncertainty,
                        }
                        results.append(result)
                        index_of_total += 1

                index += 1

        precision = TP / (TP + FN)
        recall = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)

        print(
            f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
            f"Accuracy: {(TN + TP) / (TN + TP + FN + FP)}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1_score}"
        )

        all_scores = (TP, TN, FP, FN)

        output_file_path = f'./outputs/results_{dataset}_{method}_{weight}.json'
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)

        output_score_file = output_file_path.replace(".json", "_scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump(
                {
                    "acc": (TN + TP) / (TN + TP + FN + FP),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score,
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        return all_scores
