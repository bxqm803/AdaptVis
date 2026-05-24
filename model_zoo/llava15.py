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



def _compat_greedy_generate(
    self_model,
    input_ids=None,
    attention_mask=None,
    pixel_values=None,
    max_new_tokens=100,
    output_scores=False,
    return_dict_in_generate=False,
    weight=None,
    keys=None,
    **kwargs,
):
    """
    Minimal greedy generation for this AdaptVis custom LLaVA model under newer Transformers.

    Newer Transformers may not expose .generate() on the custom
    LlavaForConditionalGenerationScal class.  This local generator keeps the
    old evaluation behavior used by this repo: greedy decoding and optional
    first-step scores, while explicitly forwarding AdaptVis `weight`.
    """
    if input_ids is None:
        raise ValueError("input_ids is required")

    cur_input_ids = input_ids
    device = cur_input_ids.device

    if attention_mask is None:
        cur_attention_mask = torch.ones_like(cur_input_ids, device=device)
    else:
        cur_attention_mask = attention_mask

    eos_token_id = kwargs.get("eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(getattr(self_model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(getattr(self_model, "config", None), "eos_token_id", None)

    if isinstance(eos_token_id, int):
        eos_ids = torch.tensor([eos_token_id], device=device)
    elif isinstance(eos_token_id, (list, tuple)) and len(eos_token_id) > 0:
        eos_ids = torch.tensor(list(eos_token_id), device=device)
    else:
        eos_ids = None

    scores = []

    for _step in range(int(max_new_tokens)):
        forward_kwargs = {
            "input_ids": cur_input_ids,
            "attention_mask": cur_attention_mask,
            "return_dict": True,
            "use_cache": False,
        }

        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values

        if "Scal" in str(type(self_model)):
            forward_kwargs["weight"] = weight

        outputs = self_model(**forward_kwargs)
        next_token_logits = outputs.logits[:, -1, :]

        if output_scores:
            scores.append(next_token_logits)

        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        cur_input_ids = torch.cat([cur_input_ids, next_tokens], dim=-1)
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones_like(next_tokens, device=device)],
            dim=-1,
        )

        if eos_ids is not None:
            hit_eos = (next_tokens == eos_ids.view(1, -1)).any(dim=-1)
            if bool(hit_eos.all().item()):
                break

    if return_dict_in_generate:
        return GenerateDecoderOnlyOutput(
            sequences=cur_input_ids,
            scores=tuple(scores),
            logits=None,
            attentions=None,
            hidden_states=None,
            past_key_values=None,
        )

    return cur_input_ids


class LlavaWrapper:
    def __init__(self, root_dir, device,method):
        
        if method=='scaling_vis' or method=='adapt_vis':
            self.model = LlavaForConditionalGenerationScal.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(MODEL, revision='a272c74', cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.processor = AutoProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)

        self.device = device

        # Newer Transformers may not provide .generate() for this custom model class.
        # Bind a local greedy generator so AdaptVis can still pass `weight`.
        import types
        self.model.generate = types.MethodType(_compat_greedy_generate, self.model)
        # Newer Transformers' LlavaProcessor.__call__ expands <image> tokens.
        # AdaptVis custom LLaVA expects the legacy behavior: keep one <image>
        # token in input_ids and let modeling_llava_scal.py merge/expand image
        # features internally.  Therefore all generation code should build
        # inputs through _build_llava_legacy_inputs() instead of calling
        # self.processor(text=..., images=...).
        try:
            vision_config = getattr(self.model.config, "vision_config", None)
            self.processor.patch_size = getattr(vision_config, "patch_size", 14)
            self.processor.vision_feature_select_strategy = getattr(
                self.model.config,
                "vision_feature_select_strategy",
                "default",
            )
        except Exception:
            pass

    def _build_llava_legacy_inputs(self, text, image, max_length=77):
        """
        Build LLaVA inputs in the legacy format expected by this AdaptVis repo.

        This intentionally avoids AutoProcessor.__call__(text=..., images=...),
        because newer Transformers expands the <image> token at processor level.
        The custom AdaptVis model expects one <image> token and performs
        image-token expansion inside _merge_input_ids_with_image_features().
        """
        text_inputs = self.processor.tokenizer(
            text=text,
            padding="max_length",
            return_tensors="pt",
            max_length=max_length,
            truncation=False,
        )

        image_inputs = self.processor.image_processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {}
        inputs.update(dict(text_inputs))
        inputs.update(dict(image_inputs))

        return {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    
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
    
    
    @torch.no_grad()
    def get_out_scores_wh_batched(self, dataset, joint_loader, method, weight, option, threshold, weight1, weight2):

        
        scores = []  # To store scores for each batch
        index_of_total = 0  # Track total number of prompts processed
        acc = 0  # Track the number of correct predictions
        correct_id = []  # Track indices of correct predictions

        # Determine the correct question-answer file based on the dataset
        qst_ans_file = f'prompts/{dataset}_with_answer_{option}_options.jsonl'
        
        # Load prompts and answers from the question-answer file
        with open(qst_ans_file, 'r') as file:
            prompt_list = []
            answer_list = []
            first_prompt_list = []
            second_prompt_list = []
            for line in file:
                data = json.loads(line)
                # Select prompt based on mode
                
                prompt_list.append(data["question"])
                
                # Store additional prompts if adjustment method is 'sub'
                
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

        # Unique output path for this run. The original code did not include weight1/weight2,
        # so three alpha runs would overwrite each other.
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

        results = []  # Store results for each generated sequence
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Set environment variable for attention map save path
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over each image option in the batch
            for i_option in batch["image_options"]:
                im_scores = []
                
                for _ in i_option:
                    prompt = prompt_list[index_of_total]
                    
                    # Preprocess input for the model
                    single_input = self._build_llava_legacy_inputs(
                        text=prompt,
                        image=_,
                        max_length=77,
                    )
                    
                    # Create key mask for special token
                    keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]

                    # Generate predictions based on specified method
                    selected_weight = None
                    uncertainty = None
                    if method == 'scaling_vis':
                        
                        change_greedy_to_add_weight()
                        selected_weight = weight
                        output = self.model.generate(
                            **single_input, keys=keys, weight=weight,
                            max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                    
                    elif method == 'adapt_vis':
                        change_greedy_to_add_weight()

                        base_output = self.model.generate(
                            **single_input, weight=1.0, max_new_tokens=100,
                            output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty = np.round(float(max(torch.nn.functional.softmax(base_output['scores'][0], dim=-1)[0])), 2)
                        print("[BASE CONFIDENCE]", uncertainty, "threshold=", threshold)

                        # Adjust attention based on uncertainty. For your three-run study,
                        # use w1=w2=1.0 / 0.5 / 1.5, so the threshold branch no longer changes alpha.
                        if uncertainty < threshold:
                            selected_weight = weight1
                        else:
                            selected_weight = weight2

                        output = self.model.generate(
                            **single_input, keys=keys, weight=selected_weight,
                            max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True
                        )

                    else:
                        # Default generation method
                        output = self.model.generate(
                            **single_input, max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                        uncertainty = np.round(float(max(output['scores'][0][0])), 2)

                    gold = answer_list[index_of_total][0]
                    is_correct = _raw_generation_correct(gold, gen)

                    # Print prompt, generated text, and expected answer/correctness.
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
                        "method": method,
                        "weight": float(weight) if weight is not None else None,
                        "weight1": float(weight1) if weight1 is not None else None,
                        "weight2": float(weight2) if weight2 is not None else None,
                        "threshold": float(threshold) if threshold is not None else None,
                        "selected_weight": float(selected_weight) if selected_weight is not None else None,
                        "uncertainty": float(uncertainty) if uncertainty is not None else None,
                    }
                    results.append(result)

                    # Check if the generation matches the expected answer.
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

            # Save results incrementally.
            print("Saving results to", output_file_path)
            with open(output_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)
            running_acc = acc / index_of_total if index_of_total else 0.0
            print("[RUNNING ACC]", acc, index_of_total, running_acc)

        # Save accuracy and correct IDs to file.
        final_acc = acc / index_of_total if index_of_total else 0.0
        print("[FINAL RAW GENERATION ACC]", final_acc)
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({
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
            }, fout, ensure_ascii=False, indent=4)

        # Concatenate all scores and return based on dataset type
        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        if dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
            return (all_scores, [])
        else:
            return (acc / index_of_total, correct_id)

    
    
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
                        single_input = self._build_llava_legacy_inputs(
                            text=text,
                            image=list(i_option)[idx],
                            max_length=77,
                        )
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
    
