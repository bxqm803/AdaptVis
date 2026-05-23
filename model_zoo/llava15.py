import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import random
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor
import pdb
# import probe_llava
from .llava import  LlavaForConditionalGeneration, LlavaForConditionalGenerationScal

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
MODEL='llava-hf/llava-1.5-7b-hf'

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
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput,GenerateEncoderDecoderOutput,GenerateDecoderOnlyOutput,GenerateNonBeamOutput
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
def _add_weight_greedy_search(
    self,
    input_ids: torch. LongTensor,
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
        import pdb
        # 
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
            continue  # don't waste resources running the code we don't need

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

class LlavaWrapper:
    def __init__(self, root_dir, device,method):
        
        if method=='scaling_vis' or method=='adapt_vis':
            self.model = LlavaForConditionalGenerationScal.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(MODEL, revision='a272c74', cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.processor = AutoProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)

        # HF >= 4.46 warns / errors if these two processor attributes are missing.
        # Keep them in the processor so LLaVA image-token expansion is explicit.
        patch_size = getattr(getattr(self.model.config, 'vision_config', None), 'patch_size', 14)
        vision_feature_select_strategy = getattr(
            self.model.config,
            'vision_feature_select_strategy',
            'default',
        )
        self.processor.patch_size = patch_size
        self.processor.vision_feature_select_strategy = vision_feature_select_strategy

        self.device = device
    
    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=64, normalize=False):
        num_text = len(texts)
        text_embeds = []
        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i: min(num_text, i+text_batch_size)]
            text_input = self.tokenizer(text=text, return_tensors="pt", padding="max_length", max_length=77).to(self.device)
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
    
    
    # ============================================================
    # Closed-set helpers for ARO spatial relations
    # ============================================================
    def _get_gold_text(self, answer_entry):
        if isinstance(answer_entry, list):
            return str(answer_entry[0]) if len(answer_entry) > 0 else ""
        return str(answer_entry)

    def _normalize_relation(self, text):
        s = str(text).strip().lower()
        if "under" in s:
            return "under"
        if "left" in s:
            return "left"
        if "right" in s:
            return "right"
        # Avoid treating "in front of" as "on".
        if "front" not in s:
            padded = " " + s.replace(".", " ").replace(",", " ").replace(":", " ") + " "
            if " on " in padded or s == "on":
                return "on"
        return s

    def _first_token_confidence_from_generate(self, output):
        """Return max softmax probability of the first generated-token distribution."""
        try:
            scores = output.scores if hasattr(output, 'scores') else output['scores']
            if scores is None or len(scores) == 0:
                return 0.0
            return float(torch.softmax(scores[0].float(), dim=-1).max().detach().cpu().item())
        except Exception:
            return 0.0

    def _decode_generate_output(self, output, input_len):
        """Decode generated tokens and full sequence without truncating the raw generation."""
        sequences = output.sequences if hasattr(output, 'sequences') else output['sequences']
        seq = sequences[0]
        new_tokens = seq[input_len:]
        raw_new = self.processor.decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # Match the original repo's generation decode exactly for the clean text:
        # self.processor.decode(new_tokens, skip_special_tokens=True)
        clean_new = self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
        )
        full_sequence = self.processor.decode(
            seq,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return raw_new, clean_new, full_sequence

    def _find_image_token_id(self, input_ids):
        tokenizer = self.processor.tokenizer
        candidate_ids = []

        config_id = getattr(self.model.config, 'image_token_index', None)
        if config_id is not None:
            candidate_ids.append(int(config_id))

        try:
            tok_id = tokenizer.convert_tokens_to_ids('<image>')
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                candidate_ids.append(int(tok_id))
        except Exception:
            pass

        # The original repo uses 32001 in llava15.py, while HF LLaVA often uses 32000.
        candidate_ids.extend([32000, 32001])

        seen = set()
        candidate_ids = [x for x in candidate_ids if not (x in seen or seen.add(x))]
        for cid in candidate_ids:
            if bool((input_ids == cid).any().item()):
                return cid
        return candidate_ids[0] if candidate_ids else 32000

    def _infer_num_image_patches(self, inputs, outputs, image_token_id):
        """
        Infer how many embeddings one <image> token expands into.
        LLaVA-1.5 default: 336 / 14 = 24, so 24*24 = 576 patches.
        If logits are not expanded, return 1 so indexing falls back to text positions.
        """
        input_ids = inputs['input_ids']
        out_len = int(outputs.logits.shape[1])
        in_len = int(input_ids.shape[1])
        if out_len <= in_len:
            return 1

        pixel_values = inputs.get('pixel_values', None)
        if pixel_values is not None:
            patch_size = getattr(self.processor, 'patch_size', None)
            if patch_size is None:
                patch_size = getattr(getattr(self.model.config, 'vision_config', None), 'patch_size', 14)
            patch_size = int(patch_size)
            h = int(pixel_values.shape[-2])
            w = int(pixel_values.shape[-1])
            n = (h // patch_size) * (w // patch_size)
            strategy = getattr(self.processor, 'vision_feature_select_strategy', 'default')
            if strategy == 'full':
                n += 1
            return int(n)

        num_image_tokens = int((input_ids == image_token_id).sum(dim=-1).max().item())
        if num_image_tokens <= 0:
            return 1
        return int((out_len - in_len) // num_image_tokens + 1)

    def _compute_expanded_token_positions(self, input_ids, image_token_id, num_image_patches):
        """
        Reproduce LLaVA's text-token position shift after <image> expansion.
        Example with one image token and 576 image patches:
            text positions after image shift by +575.
        """
        special_image_token_mask = (input_ids == image_token_id).long()
        token_increments = special_image_token_mask * (num_image_patches - 1) + 1
        new_token_positions = torch.cumsum(token_increments, dim=-1) - 1

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            return new_token_positions

        # Same left-padding compensation logic used in HF LLaVA merge code.
        left_padding = not torch.sum(input_ids[:, -1] == pad_token_id).item()
        if left_padding:
            max_embed_dim = (
                int(special_image_token_mask.sum(dim=-1).max().item()) * (num_image_patches - 1)
            ) + input_ids.shape[1]
            nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
            new_token_positions = new_token_positions + nb_image_pad[:, None]

        return new_token_positions

    @torch.no_grad()
    def score_closed_set_spatial(self, image, prompt, surfaces, weight=1.0, debug=False, prepared_inputs=None):
        """
        Closed-set scoring aligned with the first step of generation.

        Important:
            We do NOT score logits at the appended candidate token position.
            We run the original prompt-only input, then read outputs.logits[:, -1, :],
            which is exactly the distribution used by greedy generation for the next token.

        Therefore this scores:
            P(" left"  | prompt, image)
            P(" right" | prompt, image)
            P(" under" | prompt, image)
            P(" on"    | prompt, image)

        This makes the AdaptVis weight affect the same query/state that generation uses.
        """
        tokenizer = self.processor.tokenizer

        # Use the same preprocessed input as the generation branch whenever possible.
        # This keeps padding=max_length, max_length=77, and image preprocessing identical
        # to the raw generation code path.
        if prepared_inputs is None:
            inputs = self.processor(
                text=prompt,
                images=image,
                padding="max_length",
                return_tensors="pt",
                max_length=77,
            ).to(self.device)
        else:
            inputs = prepared_inputs

        model_kwargs = dict(inputs)
        if 'Scal' in str(type(self.model)):
            outputs = self.model(**model_kwargs, weight=weight, return_dict=True)
        else:
            outputs = self.model(**model_kwargs, return_dict=True)

        # This is intentionally the same position used in _add_weight_greedy_search:
        #     next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = outputs.logits[:, -1, :]
        next_token_log_probs = F.log_softmax(next_token_logits.float(), dim=-1)[0]
        next_token_vocab_probs = torch.softmax(next_token_logits.float(), dim=-1)[0]

        raw = {}
        normalized_scores = {}
        normalized_details = {}

        for surface in surfaces:
            candidate_text = " " + surface
            candidate_token_ids = tokenizer(
                candidate_text,
                add_special_tokens=False,
            ).input_ids

            if len(candidate_token_ids) == 0:
                first_token_id = None
                first_token_text = ""
                avg_lp = float('-inf')
                sum_lp = float('-inf')
                vocab_prob = 0.0
                token_lps = []
            else:
                # For LLaMA tokenizer, " left", " right", " under", " on" are normally
                # one token. If any surface becomes multiple tokens, we deliberately score
                # the first next-token only, because this closed set is meant to match the
                # first generation decision.
                first_token_id = int(candidate_token_ids[0])
                first_token_text = tokenizer.decode(
                    [first_token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                first_lp = float(next_token_log_probs[first_token_id].detach().cpu().item())
                vocab_prob = float(next_token_vocab_probs[first_token_id].detach().cpu().item())
                avg_lp = first_lp
                sum_lp = first_lp
                token_lps = [first_lp]

            norm = self._normalize_relation(surface)
            detail = {
                'surface': surface,
                'normalized': norm,
                'candidate_text': candidate_text,
                'avg_logprob': avg_lp,
                'sum_logprob': sum_lp,
                'prob': None,  # closed-set normalized prob, filled below
                'vocab_prob': vocab_prob,
                'num_tokens_scored': len(token_lps),
                'candidate_token_ids': [int(x) for x in candidate_token_ids],
                'scored_token_id': first_token_id,
                'scored_token_text': first_token_text,
                'token_logprobs': token_lps,
                'scoring_mode': 'prompt_only_next_token_logits_minus_1_not_appended_candidate_state',
                'logits_position': int(outputs.logits.shape[1] - 1),
            }
            raw[surface] = detail

            # If two surfaces normalize to the same relation, keep the higher log-prob.
            if norm not in normalized_scores or avg_lp > normalized_scores[norm]:
                normalized_scores[norm] = avg_lp
                normalized_details[norm] = detail

            if debug:
                print(
                    f"[CLOSED-SET NEXT-TOKEN DEBUG] surface={surface!r} norm={norm!r} "
                    f"candidate_ids={candidate_token_ids} scored_id={first_token_id} "
                    f"scored_text={first_token_text!r} logp={avg_lp:.6f} "
                    f"vocab_prob={vocab_prob:.8f} weight={weight}"
                )

        relation_order = [self._normalize_relation(s) for s in surfaces]
        relation_order = list(dict.fromkeys(relation_order))
        score_vec = torch.tensor(
            [normalized_scores.get(r, float('-inf')) for r in relation_order],
            dtype=torch.float32,
        )
        probs_vec = torch.softmax(score_vec, dim=0)
        probs = {r: float(probs_vec[i].item()) for i, r in enumerate(relation_order)}
        for r, p in probs.items():
            if r in normalized_details:
                normalized_details[r]['prob'] = p

        pred = max(normalized_scores.items(), key=lambda kv: kv[1])[0]
        confidence = probs.get(pred, 0.0)

        image_token_id = self._find_image_token_id(inputs['input_ids'])
        num_image_patches = self._infer_num_image_patches(inputs, outputs, image_token_id)

        return {
            'surfaces': surfaces,
            'relation_order': relation_order,
            'scores': normalized_scores,
            'probs': probs,
            'pred': pred,
            'confidence': confidence,
            'raw': raw,
            'normalized_details': normalized_details,
            'image_token_id': int(image_token_id),
            'num_image_patches': int(num_image_patches),
            'logits_seq_len': int(outputs.logits.shape[1]),
            'input_seq_len': int(inputs['input_ids'].shape[1]),
            'scoring_mode': 'prompt_only_next_token',
            'logits_position': int(outputs.logits.shape[1] - 1),
        }


    @torch.no_grad()
    def score_closed_set_from_generate_scores(self, output, surfaces, debug=False):
        """
        Closed-set scoring using the exact first-step scores returned by generate().

        This is the most apples-to-apples comparison with raw generation:
            raw generation first token = argmax over the whole vocabulary of output.scores[0]
            closed-set prediction     = argmax over only left/right/under/on token ids of output.scores[0]

        Therefore closed-set may still differ from raw generation if the global top token is
        something like "The" or "It", because closed-set restricts the vocabulary.
        """
        tokenizer = self.processor.tokenizer
        scores_tuple = output.scores if hasattr(output, 'scores') else output['scores']
        sequences = output.sequences if hasattr(output, 'sequences') else output['sequences']

        if scores_tuple is None or len(scores_tuple) == 0:
            raise ValueError('generate() output has no scores. Set output_scores=True and return_dict_in_generate=True.')

        first_step_scores = scores_tuple[0].float()[0]  # [vocab]
        first_step_log_probs = F.log_softmax(first_step_scores, dim=-1)
        first_step_vocab_probs = torch.softmax(first_step_scores, dim=-1)

        first_generated_id = int(sequences[0, -len(scores_tuple)].detach().cpu().item()) if sequences.ndim == 2 else None
        # More robust: the first generated token is the first token generated after the prompt.
        # Since output.scores length is new-token count, this indexing works for generate output.
        try:
            first_generated_id = int(sequences[0, sequences.shape[1] - len(scores_tuple)].detach().cpu().item())
        except Exception:
            pass
        if first_generated_id is not None:
            first_generated_text = tokenizer.decode(
                [first_generated_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            first_generated_logprob = float(first_step_log_probs[first_generated_id].detach().cpu().item())
            first_generated_vocab_prob = float(first_step_vocab_probs[first_generated_id].detach().cpu().item())
        else:
            first_generated_text = ''
            first_generated_logprob = float('nan')
            first_generated_vocab_prob = float('nan')

        raw = {}
        normalized_scores = {}
        normalized_details = {}

        for surface in surfaces:
            candidate_text = ' ' + surface
            candidate_token_ids = tokenizer(candidate_text, add_special_tokens=False).input_ids

            if len(candidate_token_ids) == 0:
                first_token_id = None
                first_token_text = ''
                lp = float('-inf')
                vocab_prob = 0.0
            else:
                # Closed-set first-step comparison: only the first answer token is comparable
                # to generation's first step.
                first_token_id = int(candidate_token_ids[0])
                first_token_text = tokenizer.decode(
                    [first_token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                lp = float(first_step_log_probs[first_token_id].detach().cpu().item())
                vocab_prob = float(first_step_vocab_probs[first_token_id].detach().cpu().item())

            norm = self._normalize_relation(surface)
            detail = {
                'surface': surface,
                'normalized': norm,
                'candidate_text': candidate_text,
                'avg_logprob': lp,
                'sum_logprob': lp,
                'prob': None,
                'vocab_prob': vocab_prob,
                'num_tokens_scored': 1 if first_token_id is not None else 0,
                'candidate_token_ids': [int(x) for x in candidate_token_ids],
                'scored_token_id': first_token_id,
                'scored_token_text': first_token_text,
                'token_logprobs': [lp] if first_token_id is not None else [],
                'scoring_mode': 'generate_scores_first_step_restricted_vocab',
            }
            raw[surface] = detail

            if norm not in normalized_scores or lp > normalized_scores[norm]:
                normalized_scores[norm] = lp
                normalized_details[norm] = detail

            if debug:
                print(
                    f"[CLOSED-SET FROM GENERATE DEBUG] surface={surface!r} norm={norm!r} "
                    f"candidate_ids={candidate_token_ids} scored_id={first_token_id} "
                    f"scored_text={first_token_text!r} logp={lp:.6f} vocab_prob={vocab_prob:.8f}"
                )

        relation_order = [self._normalize_relation(s) for s in surfaces]
        relation_order = list(dict.fromkeys(relation_order))
        score_vec = torch.tensor(
            [normalized_scores.get(r, float('-inf')) for r in relation_order],
            dtype=torch.float32,
        )
        probs_vec = torch.softmax(score_vec, dim=0)
        probs = {r: float(probs_vec[i].item()) for i, r in enumerate(relation_order)}
        for r, prob in probs.items():
            if r in normalized_details:
                normalized_details[r]['prob'] = prob

        pred = max(normalized_scores.items(), key=lambda kv: kv[1])[0]
        confidence = probs.get(pred, 0.0)

        return {
            'surfaces': surfaces,
            'relation_order': relation_order,
            'scores': normalized_scores,
            'probs': probs,
            'pred': pred,
            'confidence': confidence,
            'raw': raw,
            'normalized_details': normalized_details,
            'first_generated_token_id': first_generated_id,
            'first_generated_token_text': first_generated_text,
            'first_generated_token_logprob': first_generated_logprob,
            'first_generated_token_vocab_prob': first_generated_vocab_prob,
            'scoring_mode': 'generate_scores_first_step_restricted_vocab',
        }

    def _format_closed_set_line(self, tag, closed_result):
        pieces = []
        for rel in closed_result['relation_order']:
            lp = closed_result['scores'].get(rel, float('-inf'))
            p = closed_result['probs'].get(rel, 0.0)
            pieces.append(f"{rel}: logp={lp:.4f}, prob={p:.4f}")
        return (
            f"{tag}: pred={closed_result['pred']}, "
            f"conf={closed_result['confidence']:.4f}, "
            + "; ".join(pieces)
        )

    def _classify_alpha_scaling_case(self, base_correct, low_correct, high_correct):
        """
        Mutually exclusive case label for alpha-scaling analysis.

        base_correct: closed-set correctness at weight=1.0
        low_correct:  closed-set correctness at weight1, normally < 1
        high_correct: closed-set correctness at weight2, normally > 1

        The user's six main categories are:
            low_wrong_to_correct
            high_wrong_to_correct
            always_correct
            low_correct_to_wrong
            high_correct_to_wrong
            always_wrong

        Two extra buckets are kept so every sample is counted exactly once when both
        low and high scaling flip the base result in the same direction.
        """
        base_correct = bool(base_correct)
        low_correct = bool(low_correct)
        high_correct = bool(high_correct)

        if base_correct and low_correct and high_correct:
            return "always_correct"
        if (not base_correct) and (not low_correct) and (not high_correct):
            return "always_wrong"

        if (not base_correct) and low_correct and (not high_correct):
            return "low_wrong_to_correct"
        if (not base_correct) and (not low_correct) and high_correct:
            return "high_wrong_to_correct"

        if base_correct and (not low_correct) and high_correct:
            return "low_correct_to_wrong"
        if base_correct and low_correct and (not high_correct):
            return "high_correct_to_wrong"

        if (not base_correct) and low_correct and high_correct:
            return "both_scalings_wrong_to_correct"
        if base_correct and (not low_correct) and (not high_correct):
            return "both_scalings_correct_to_wrong"

        return "unclassified"

    def _new_alpha_effect_stats(self):
        names = [
            "low_wrong_to_correct",
            "high_wrong_to_correct",
            "always_correct",
            "low_correct_to_wrong",
            "high_correct_to_wrong",
            "always_wrong",
            "both_scalings_wrong_to_correct",
            "both_scalings_correct_to_wrong",
            "unclassified",
        ]
        return {name: {"count": 0, "sample_ids": []} for name in names}

    def _record_alpha_effect_case(self, stats, case_name, sample_id):
        if case_name not in stats:
            stats[case_name] = {"count": 0, "sample_ids": []}
        stats[case_name]["count"] += 1
        stats[case_name]["sample_ids"].append(int(sample_id))

    def _short_closed_pred(self, closed_result):
        if closed_result is None:
            return "None"
        return f"{closed_result['pred']}@{closed_result['confidence']:.4f}"

    @torch.no_grad()
    def get_out_scores_wh_batched(self, dataset, joint_loader, method, weight, option, threshold, weight1, weight2):

        scores = []  # To store scores for each batch
        index_of_total = 0  # Track total number of prompts processed

        # For adapt_vis below, acc is based on lowercase closed-set prediction.
        # Raw generation accuracy is still tracked and printed for comparison.
        acc = 0
        raw_acc = 0
        upper_acc = 0
        correct_id = []
        raw_correct_id = []
        upper_correct_id = []

        # Determine the correct question-answer file based on the dataset
        qst_ans_file = f'prompts/{dataset}_with_answer_{option}_options.jsonl'

        # Load prompts and answers from the question-answer file
        with open(qst_ans_file, 'r') as file:
            prompt_list = []
            answer_list = []
            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        # Sampling configuration
        SAMPLE = False
        TEST = os.getenv('TEST_MODE', 'False') == 'True'
        total_data_count = len(prompt_list)

        # Perform sampling if enabled
        if SAMPLE:
            idx_file_path = f'./output/sampled_idx_{dataset}.npy'

            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
                sampled_indices.sort()
                np.save(idx_file_path, np.array(sampled_indices))

            # For testing mode, use unsampled indices
            if TEST:
                all_indices = set(range(total_data_count))
                unsampled_indices = list(all_indices - set(sampled_indices))
                unsampled_indices.sort()
                sampled_indices = unsampled_indices

            # Subset prompts and answers based on sampled indices
            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        # Create directory for saving attention maps
        save_attn_dir = f"./output/{dataset}_weight{weight:.2f}"
        os.makedirs(save_attn_dir, exist_ok=True)

        lower_surfaces = ["left", "right", "under", "on"]
        upper_surfaces = ["Left", "Right", "Under", "On"]

        # Exclusive 6(+2) case statistics for alpha-scaling analysis.
        # Base is weight=1.0. Low is weight1, normally <1. High is weight2, normally >1.
        alpha_effect_stats = self._new_alpha_effect_stats()

        results = []  # Store results for each generated sequence
        for batch in tqdm(joint_loader):
            batch_scores = []

            # Set environment variable for attention map save path
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over each image option in the batch
            for i_option in batch["image_options"]:
                im_scores = []

                for image in i_option:
                    prompt = prompt_list[index_of_total]
                    gold_text = self._get_gold_text(answer_list[index_of_total])
                    gold_norm = self._normalize_relation(gold_text)

                    # Original generation input is kept unchanged.
                    single_input = self.processor(
                        text=prompt,
                        images=image,
                        padding="max_length",
                        return_tensors="pt",
                        max_length=77,
                    ).to(self.device)

                    # Keep the original key mask construction for the generation branch.
                    keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]

                    raw_generation = ""
                    raw_generation_clean = ""
                    raw_generation_full_sequence = ""
                    base_raw_generation = ""
                    base_uncertainty = 0.0
                    final_weight = weight
                    lower_closed = None
                    upper_closed = None
                    base_lower_closed = None
                    base_upper_closed = None
                    low_lower_closed = None
                    low_upper_closed = None
                    high_lower_closed = None
                    high_upper_closed = None
                    alpha_effect_case = None
                    base_lower_correct = False
                    low_lower_correct = False
                    high_lower_correct = False
                    gen_for_original_match = ""
                    eval_pred = ""

                    # Generate predictions based on specified method
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
                        base_uncertainty = self._first_token_confidence_from_generate(output)
                        raw_generation, raw_generation_clean, raw_generation_full_sequence = self._decode_generate_output(
                            output,
                            input_len=single_input['input_ids'].shape[-1],
                        )
                        gen_for_original_match = raw_generation_clean
                        eval_pred = self._normalize_relation(raw_generation_clean)

                    elif method == 'adapt_vis':
                        change_greedy_to_add_weight()

                        # 1) Original AdaptVis uncertainty probe, unchanged.
                        base_output = self.model.generate(
                            **single_input,
                            weight=1.0,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                        base_uncertainty_raw = self._first_token_confidence_from_generate(base_output)
                        # Match the original repo exactly: it rounds to 2 decimals before comparing with threshold.
                        base_uncertainty = float(np.round(float(base_uncertainty_raw), 2))
                        base_raw_generation, _, _ = self._decode_generate_output(
                            base_output,
                            input_len=single_input['input_ids'].shape[-1],
                        )

                        # 2) Original AdaptVis weight selection, unchanged.
                        if base_uncertainty < threshold:
                            final_weight = weight1
                        else:
                            final_weight = weight2

                        # 3) Traditional generation branch, unchanged except that raw text is saved fully.
                        output = self.model.generate(
                            **single_input,
                            keys=keys,
                            weight=final_weight,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                        raw_generation, raw_generation_clean, raw_generation_full_sequence = self._decode_generate_output(
                            output,
                            input_len=single_input['input_ids'].shape[-1],
                        )
                        gen_for_original_match = raw_generation_clean

                        # 4) Closed-set at alpha=1.0, weight1, and weight2 for case statistics.
                        #    These use generate().scores[0], i.e. the exact first-step distribution.
                        base_lower_closed = self.score_closed_set_from_generate_scores(
                            output=base_output,
                            surfaces=lower_surfaces,
                            debug=False,
                        )
                        base_upper_closed = self.score_closed_set_from_generate_scores(
                            output=base_output,
                            surfaces=upper_surfaces,
                            debug=False,
                        )

                        # Reuse the full 100-token generation as the selected branch; only run a
                        # one-token generation for the unselected branch to get first-step scores.
                        if float(final_weight) == float(weight1):
                            low_output_for_closed = output
                            high_output_for_closed = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight2,
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                        elif float(final_weight) == float(weight2):
                            high_output_for_closed = output
                            low_output_for_closed = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight1,
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                        else:
                            low_output_for_closed = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight1,
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            high_output_for_closed = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight2,
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                        low_lower_closed = self.score_closed_set_from_generate_scores(
                            output=low_output_for_closed,
                            surfaces=lower_surfaces,
                            debug=False,
                        )
                        low_upper_closed = self.score_closed_set_from_generate_scores(
                            output=low_output_for_closed,
                            surfaces=upper_surfaces,
                            debug=False,
                        )
                        high_lower_closed = self.score_closed_set_from_generate_scores(
                            output=high_output_for_closed,
                            surfaces=lower_surfaces,
                            debug=False,
                        )
                        high_upper_closed = self.score_closed_set_from_generate_scores(
                            output=high_output_for_closed,
                            surfaces=upper_surfaces,
                            debug=False,
                        )

                        # Keep the previous behavior: the main closed-set prediction follows the
                        # branch selected by AdaptVis thresholding.
                        if float(final_weight) == float(weight1):
                            lower_closed = low_lower_closed
                            upper_closed = low_upper_closed
                        else:
                            lower_closed = high_lower_closed
                            upper_closed = high_upper_closed

                        eval_pred = lower_closed['pred']

                    else:
                        # Default generation method, unchanged.
                        output = self.model.generate(
                            **single_input,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                        base_uncertainty = self._first_token_confidence_from_generate(output)
                        raw_generation, raw_generation_clean, raw_generation_full_sequence = self._decode_generate_output(
                            output,
                            input_len=single_input['input_ids'].shape[-1],
                        )
                        gen_for_original_match = raw_generation_clean
                        eval_pred = self._normalize_relation(raw_generation_clean)

                    # Correctness for raw generation, using the original substring-style rule.
                    raw_correct = (
                        (gold_text in gen_for_original_match or gold_text.lower() in gen_for_original_match.lower())
                        and not (gold_text.lower() == 'on' and 'front' in gen_for_original_match.strip().lower())
                    )
                    if raw_correct:
                        raw_acc += 1
                        raw_correct_id.append(index_of_total)

                    # Correctness for closed-set branches.
                    lower_correct = bool(eval_pred == gold_norm)
                    upper_correct = False
                    if upper_closed is not None:
                        upper_correct = bool(upper_closed['pred'] == gold_norm)

                    if method == 'adapt_vis':
                        base_lower_correct = bool(base_lower_closed is not None and base_lower_closed['pred'] == gold_norm)
                        low_lower_correct = bool(low_lower_closed is not None and low_lower_closed['pred'] == gold_norm)
                        high_lower_correct = bool(high_lower_closed is not None and high_lower_closed['pred'] == gold_norm)
                        alpha_effect_case = self._classify_alpha_scaling_case(
                            base_correct=base_lower_correct,
                            low_correct=low_lower_correct,
                            high_correct=high_lower_correct,
                        )
                        self._record_alpha_effect_case(alpha_effect_stats, alpha_effect_case, index_of_total)

                    if method == 'adapt_vis':
                        if lower_correct:
                            acc += 1
                            correct_id.append(index_of_total)
                        if upper_correct:
                            upper_acc += 1
                            upper_correct_id.append(index_of_total)
                    else:
                        if raw_correct:
                            acc += 1
                            correct_id.append(index_of_total)

                    print("\n" + "=" * 100)
                    print(f"Sample ID: {index_of_total}")
                    print(f"Prompt: {prompt}")
                    print(f"Golden: {gold_text}  | normalized={gold_norm}")
                    print(f"Method: {method}")
                    print(f"AdaptVis base_confidence={base_uncertainty:.4f}, threshold={threshold}, final_weight={final_weight}")
                    print(f"[RAW GENERATION | new tokens | skip_special_tokens=False]\n{raw_generation}")
                    print(f"[RAW GENERATION CLEAN | used by original substring match]\n{raw_generation_clean}")
                    if method == 'adapt_vis':
                        print(f"[BASE RAW GENERATION before AdaptVis weight selection]\n{base_raw_generation}")
                        print(
                            f"[RAW GENERATION FIRST TOKEN] "
                            f"id={lower_closed.get('first_generated_token_id')}, "
                            f"text={lower_closed.get('first_generated_token_text')!r}, "
                            f"vocab_prob={lower_closed.get('first_generated_token_vocab_prob'):.6f}"
                        )
                        print(self._format_closed_set_line("[CLOSED-SET lowercase selected by threshold]", lower_closed))
                        print(self._format_closed_set_line("[CLOSED-SET Capitalized selected by threshold]", upper_closed))
                        print(self._format_closed_set_line(f"[ALPHA=1.0 lowercase baseline]", base_lower_closed))
                        print(self._format_closed_set_line(f"[ALPHA=weight1={weight1} lowercase]", low_lower_closed))
                        print(self._format_closed_set_line(f"[ALPHA=weight2={weight2} lowercase]", high_lower_closed))
                        print(
                            f"[ALPHA EFFECT CASE | lowercase] {alpha_effect_case} | "
                            f"base@1.0={self._short_closed_pred(base_lower_closed)} correct={base_lower_correct}; "
                            f"low@{weight1}={self._short_closed_pred(low_lower_closed)} correct={low_lower_correct}; "
                            f"high@{weight2}={self._short_closed_pred(high_lower_closed)} correct={high_lower_correct}"
                        )
                        print(
                            f"Correctness: lower_closed_selected={lower_correct}, "
                            f"upper_closed_selected={upper_correct}, raw_generation={raw_correct}"
                        )
                    else:
                        print(f"Correctness: raw_generation={raw_correct}")
                    print("=" * 100)

                    result = {
                        "SampleID": index_of_total,
                        "Prompt": prompt,
                        "Golden": gold_text,
                        "GoldenNormalized": gold_norm,
                        "Method": method,
                        "Generation": raw_generation_clean,
                        "RawGeneration": raw_generation,
                        "RawGenerationFullSequence": raw_generation_full_sequence,
                        "RawGenerationCorrect": raw_correct,
                        "BaseRawGeneration": base_raw_generation,
                        "BaseConfidence": base_uncertainty,
                        "Threshold": threshold,
                        "FinalWeight": final_weight,
                        "AlphaEffectCaseLower": alpha_effect_case,
                        "AlphaBaseWeight": 1.0,
                        "AlphaLowWeight": weight1,
                        "AlphaHighWeight": weight2,
                        "AlphaBaseLowerPred": base_lower_closed['pred'] if base_lower_closed is not None else None,
                        "AlphaBaseLowerConfidence": base_lower_closed['confidence'] if base_lower_closed is not None else None,
                        "AlphaBaseLowerCorrect": base_lower_correct,
                        "AlphaLowLowerPred": low_lower_closed['pred'] if low_lower_closed is not None else None,
                        "AlphaLowLowerConfidence": low_lower_closed['confidence'] if low_lower_closed is not None else None,
                        "AlphaLowLowerCorrect": low_lower_correct,
                        "AlphaHighLowerPred": high_lower_closed['pred'] if high_lower_closed is not None else None,
                        "AlphaHighLowerConfidence": high_lower_closed['confidence'] if high_lower_closed is not None else None,
                        "AlphaHighLowerCorrect": high_lower_correct,
                    }
                    if lower_closed is not None:
                        result.update({
                            "ClosedSetLowerSurfaces": lower_surfaces,
                            "ClosedSetLowerPred": lower_closed['pred'],
                            "ClosedSetLowerConfidence": lower_closed['confidence'],
                            "ClosedSetLowerScores": lower_closed['scores'],
                            "ClosedSetLowerProbs": lower_closed['probs'],
                            "ClosedSetLowerRaw": lower_closed['raw'],
                            "ClosedSetLowerCorrect": lower_correct,
                            "ClosedSetScoringMode": lower_closed.get('scoring_mode'),
                            "ClosedSetFirstGeneratedTokenID": lower_closed.get('first_generated_token_id'),
                            "ClosedSetFirstGeneratedTokenText": lower_closed.get('first_generated_token_text'),
                            "ClosedSetFirstGeneratedTokenVocabProb": lower_closed.get('first_generated_token_vocab_prob'),
                        })
                    if upper_closed is not None:
                        result.update({
                            "ClosedSetUpperSurfaces": upper_surfaces,
                            "ClosedSetUpperPred": upper_closed['pred'],
                            "ClosedSetUpperConfidence": upper_closed['confidence'],
                            "ClosedSetUpperScores": upper_closed['scores'],
                            "ClosedSetUpperProbs": upper_closed['probs'],
                            "ClosedSetUpperRaw": upper_closed['raw'],
                            "ClosedSetUpperCorrect": upper_correct,
                        })
                    if method == 'adapt_vis':
                        if base_lower_closed is not None:
                            result.update({
                                "AlphaBaseLowerScores": base_lower_closed['scores'],
                                "AlphaBaseLowerProbs": base_lower_closed['probs'],
                                "AlphaBaseUpperPred": base_upper_closed['pred'] if base_upper_closed is not None else None,
                                "AlphaBaseUpperScores": base_upper_closed['scores'] if base_upper_closed is not None else None,
                                "AlphaBaseUpperProbs": base_upper_closed['probs'] if base_upper_closed is not None else None,
                            })
                        if low_lower_closed is not None:
                            result.update({
                                "AlphaLowLowerScores": low_lower_closed['scores'],
                                "AlphaLowLowerProbs": low_lower_closed['probs'],
                                "AlphaLowUpperPred": low_upper_closed['pred'] if low_upper_closed is not None else None,
                                "AlphaLowUpperScores": low_upper_closed['scores'] if low_upper_closed is not None else None,
                                "AlphaLowUpperProbs": low_upper_closed['probs'] if low_upper_closed is not None else None,
                            })
                        if high_lower_closed is not None:
                            result.update({
                                "AlphaHighLowerScores": high_lower_closed['scores'],
                                "AlphaHighLowerProbs": high_lower_closed['probs'],
                                "AlphaHighUpperPred": high_upper_closed['pred'] if high_upper_closed is not None else None,
                                "AlphaHighUpperScores": high_upper_closed['scores'] if high_upper_closed is not None else None,
                                "AlphaHighUpperProbs": high_upper_closed['probs'] if high_upper_closed is not None else None,
                            })

                    results.append(result)

                    # Build the score array returned to the ARO evaluator.
                    c_option = batch["caption_options"]
                    if len(list(c_option)) == 4:
                        if method == 'adapt_vis':
                            is_correct_for_return = lower_correct
                        else:
                            is_correct_for_return = raw_correct

                        if is_correct_for_return:
                            answers = [1, 0, 0, 0]
                        else:
                            answers = [0, 0, 1, 0]

                    elif len(list(c_option)) == 2:
                        if method == 'adapt_vis':
                            is_correct_for_return = lower_correct
                        else:
                            is_correct_for_return = raw_correct

                        if is_correct_for_return:
                            answers = [1, 0]
                        else:
                            answers = [0, 1]
                    else:
                        raise ValueError(f"Unexpected number of caption options: {len(list(c_option))}")

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    index_of_total += 1

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            # Save results to file
            output_file_path = f'./output/results1.5_{dataset}_{method}_{weight}_{option}option_{TEST}.json'
            print("Saving results to", output_file_path)
            with open(output_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)

            denom = max(index_of_total, 1)
            if method == 'adapt_vis':
                print(
                    f"running acc lower_closed={acc}/{denom}={acc / denom:.6f}, "
                    f"upper_closed={upper_acc}/{denom}={upper_acc / denom:.6f}, "
                    f"raw_generation={raw_acc}/{denom}={raw_acc / denom:.6f}"
                )
                print("running alpha effect cases:", {k: v["count"] for k, v in alpha_effect_stats.items() if v["count"] > 0})
            else:
                print(f"running acc raw_generation={acc}/{denom}={acc / denom:.6f}")

        # Save accuracy and correct IDs to file
        denom = max(index_of_total, 1)
        print("\n[DONE]")
        if method == 'adapt_vis':
            print(f"lower closed-set acc: {acc}/{denom}={acc / denom:.6f}")
            print(f"upper closed-set acc: {upper_acc}/{denom}={upper_acc / denom:.6f}")
            print(f"raw generation acc: {raw_acc}/{denom}={raw_acc / denom:.6f}")
            print("alpha effect case counts:")
            for case_name, item in alpha_effect_stats.items():
                print(f"  {case_name}: {item['count']} | sample_ids={item['sample_ids']}")
        else:
            print(f"raw generation acc: {acc}/{denom}={acc / denom:.6f}")

        output_score_file = output_file_path.replace(".json", "scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({
                "acc": acc / denom,
                "correct_id": correct_id,
                "lower_closedset_acc": acc / denom,
                "lower_closedset_correct_id": correct_id,
                "upper_closedset_acc": upper_acc / denom,
                "upper_closedset_correct_id": upper_correct_id,
                "raw_generation_acc": raw_acc / denom,
                "raw_generation_correct_id": raw_correct_id,
                "alpha_effect_stats_lower": alpha_effect_stats,
                "alpha_effect_weights": {
                    "base": 1.0,
                    "low": weight1,
                    "high": weight2,
                    "threshold": threshold,
                },
            }, fout, ensure_ascii=False, indent=4)

        if method == 'adapt_vis':
            alpha_stats_file = output_file_path.replace(".json", "_alpha_effect_stats.json")
            with open(alpha_stats_file, 'w', encoding='utf-8') as fout:
                json.dump({
                    "dataset": dataset,
                    "option": option,
                    "method": method,
                    "weights": {"base": 1.0, "low": weight1, "high": weight2, "threshold": threshold},
                    "case_definition": {
                        "base": "lowercase closed-set prediction at weight=1.0",
                        "low": "lowercase closed-set prediction at weight1, normally <1",
                        "high": "lowercase closed-set prediction at weight2, normally >1",
                        "always_correct": "base, low, and high are all correct",
                        "always_wrong": "base, low, and high are all wrong",
                        "low_wrong_to_correct": "base is wrong, low becomes correct, high remains wrong",
                        "high_wrong_to_correct": "base is wrong, high becomes correct, low remains wrong",
                        "low_correct_to_wrong": "base is correct, low becomes wrong, high remains correct",
                        "high_correct_to_wrong": "base is correct, high becomes wrong, low remains correct",
                        "both_scalings_wrong_to_correct": "base is wrong, both low and high become correct",
                        "both_scalings_correct_to_wrong": "base is correct, both low and high become wrong"
                    },
                    "stats": alpha_effect_stats,
                }, fout, ensure_ascii=False, indent=4)
            print("alpha effect stats saved to", alpha_stats_file)

        # Concatenate all scores and return based on dataset type
        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        if dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
            return (all_scores, [])
        else:
            return (acc / denom, correct_id)

    
    
    @torch.no_grad()
    def get_judge_scores_vsr_batched(self, dataset, joint_loader, method, weight, threshold, weight1, weight2):
        
        
        index = 0
        TP, TN, FP, FN = 0, 0, 0, 0

        # Set the directory to save attention maps
        save_attn_dir = f"/home/user/shiqi/mmlm_mech/whatsup_vlms/outputs/{dataset}_weight{weight:.2f}"
        if not os.path.exists(save_attn_dir):
            print("Creating directory for saving attention maps:", save_attn_dir)
            os.makedirs(save_attn_dir)
        
        index_of_total = 0
        results = []

        # Process each batch in the joint loader
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Create directory for saving attention maps for each batch
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over image options in the batch
            for i_option in batch["image_options"]:
                im_scores = []

                # Iterate over caption options
                for c_option in batch["caption_options"]:
                    prompt = "User: <image>\n Determine whether the description about the spatial relationship is correct or not. Answer with yes or no: "
                    qst = [prompt] * len(list(c_option))
                    end_fix = [" Assistant:"] * len(list(c_option))
                    concatenated_list = [s1 + s2 + s3 for s1, s2, s3 in zip(qst, c_option, end_fix)]
                    
                    # Generate responses for each concatenated input
                    for idx, text in enumerate(concatenated_list):
                        # Prepare input data for the model
                        single_input = self.processor(text=text, images=list(i_option)[idx], padding="max_length", return_tensors="pt", max_length=77).to(self.device)
                        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
                        
                        # Apply different attention adjustment methods based on the 'method' argument
                        if method == 'scaling_vis':
                            change_greedy_to_add_weight()
                            output = self.model.generate(**single_input, keys=keys, weight=weight, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                        
                        elif method == 'adapt_vis':
                            change_greedy_to_add_weight()
                            # Basic generation step
                            output = self.model.generate(**single_input, weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                            uncertainty = np.round(float(max(output['scores'][0][0])), 2)
                            
                            # Apply weighted generation based on uncertainty
                            if uncertainty < threshold:
                                output = self.model.generate(**single_input, keys=keys, weight=weight1, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            else:
                                output = self.model.generate(**single_input, keys=keys, weight=weight2, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)

                        else:
                            output = self.model.generate(**single_input, keys=keys, weight=weight, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                        
                        # Check correctness of the generated response
                        label = int(batch['labels'][0][idx])
                        if label == 1:
                            TP += 1 if 'Yes' in gen else 0
                            FN += 1 if 'Yes' not in gen else 0
                        else:
                            TN += 1 if 'No' in gen else 0
                            FP += 1 if 'No' not in gen else 0
                        
                        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                        
                        # Create result entry for the current sample
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
        # Calculate metrics
        precision = TP / (TP + FN)
        recall = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)

        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
            f"Accuracy: {(TN + TP) / (TN + TP + FN + FP)}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1_score}")
        
        all_scores = (TP, TN, FP, FN)
        
        # Save results to JSON file
        output_file_path = f'./outputs/results_{dataset}_{method}_{weight}.json'
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
        
        # Save evaluation metrics
        output_score_file = output_file_path.replace(".json", "_scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({"acc": (TN + TP) / (TN + TP + FN + FP), "precision": precision, "recall": recall, "f1": f1_score}, fout, ensure_ascii=False, indent=4)
        return all_scores
    
