import os
import re
import json
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

import transformers
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
)

from .llava import LlavaForConditionalGeneration, LlavaForConditionalGenerationScal


MODEL = "llava-hf/llava-1.5-7b-hf"
IMAGE_TOKEN_ID = 32001


# ============================================================
# Small helpers
# ============================================================

def _env_flag(name: str, default: str = "False") -> bool:
    return str(os.getenv(name, default)).strip().lower() in [
        "1", "true", "yes", "y", "on",
    ]


def load_probe_sample_id_set():
    """
    Optional sample filter.

    Env:
        PROBE_SAMPLE_IDS_FILE=/path/to/ids.txt

    The ids are original dataset indices.
    """
    sample_id_file = os.getenv("PROBE_SAMPLE_IDS_FILE", "").strip()
    if not sample_id_file:
        return None

    if not os.path.exists(sample_id_file):
        raise FileNotFoundError(f"PROBE_SAMPLE_IDS_FILE not found: {sample_id_file}")

    sample_id_set = set()
    with open(sample_id_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample_id_set.add(int(line))

    print(f"[FILTER] loaded {len(sample_id_set)} sample ids from {sample_id_file}")
    return sample_id_set


def make_tagged_output_path(dataset, method, weight, option, test_flag):
    base = f"./output/results1.5_{dataset}_{method}_{weight}_{option}option_{test_flag}"
    tag = os.getenv("PROBE_RUN_TAG", "").strip()

    if tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", tag)
        return f"{base}_{safe_tag}.json"

    return f"{base}.json"


def is_generation_correct(golden: str, gen: str) -> bool:
    golden = str(golden)
    gen = str(gen)

    golden_l = golden.strip().lower()
    gen_l = gen.strip().lower()

    golden_norm = golden_l.replace("in front", "in-front").replace("in_front", "in-front")
    gen_norm = gen_l.replace("in front", "in-front").replace("in_front", "in-front")

    ok = (golden in gen) or (golden_l in gen_l) or (golden_norm in gen_norm)

    if golden_norm in ["in-front", "front"]:
        ok = (
            "front" in gen_norm
            or "in-front" in gen_norm
            or "in front" in gen_l
        )

    # Avoid counting "front" as "on".
    if golden_norm == "on" and "front" in gen_l:
        ok = False

    return bool(ok)


def build_legacy_image_keys(input_ids_batch: torch.Tensor) -> List[torch.Tensor]:
    """
    Original AdaptVis/main-style image key mask.

    It marks the single <image> placeholder in input_ids rather than expanding it
    to 576 patch tokens. This intentionally matches the original llava15.py style.
    """
    image_token_id = int(os.getenv("IMAGE_TOKEN_ID", str(IMAGE_TOKEN_ID)))
    return [torch.where(input_id == image_token_id, 1, 0) for input_id in input_ids_batch]


def first_step_confidence_softmax(output) -> float:
    """
    Controlled_Images AdaptVis confidence:
    max softmax probability of the first generated-token distribution.
    """
    if output is None or output.get("scores", None) is None or len(output["scores"]) == 0:
        return 0.0

    probs = torch.softmax(output["scores"][0].detach().float(), dim=-1)
    return float(np.round(float(probs[0].max().item()), 2))


def first_step_confidence_raw(output) -> float:
    """
    Raw max score variant, kept for VSR compatibility.
    """
    if output is None or output.get("scores", None) is None or len(output["scores"]) == 0:
        return 0.0

    return float(
        np.round(
            float(torch.max(output["scores"][0][0].detach().float()).item()),
            2,
        )
    )


def decode_generated(processor, output, input_len: int) -> str:
    return processor.decode(
        output["sequences"][0][input_len:],
        skip_special_tokens=True,
    )


# ============================================================
# Custom greedy search
# ============================================================

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
    weight: Optional[float] = None,
    adjust_method: Optional[str] = None,
    pos: Optional[torch.Tensor] = None,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """
    Greedy search with AdaptVis arguments.

    Important:
        Some model.prepare_inputs_for_generation implementations preserve
        keys/weight/adjust_method/pos. Some do not.

        To avoid both failure modes:
            - if model_inputs already has a custom arg, keep it.
            - if it does not, inject the value from model_kwargs / function args.
            - call self(**model_inputs) once, without duplicate keyword args.

    This makes w1/w2 effective when modeling_llava_scal.py and
    modeling_llama_add_attn.py are restored to keep decode-step keys.
    """
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function; use stopping_criteria instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]

    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size,
        dtype=torch.long,
        device=input_ids.device,
    )

    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Safely preserve custom AdaptVis args.
        # Do not pass duplicate keyword arguments to self().
        custom_args = {
            "keys": model_kwargs.get("keys", None),
            "weight": weight if weight is not None else model_kwargs.get("weight", None),
            "adjust_method": (
                adjust_method
                if adjust_method is not None
                else model_kwargs.get("adjust_method", None)
            ),
            "pos": pos if pos is not None else model_kwargs.get("pos", None),
            "caption_length": model_kwargs.get("caption_length", None),
            "object_patch_mask": model_kwargs.get("object_patch_mask", None),
        }

        if "Scal" in str(type(self)):
            for k, v in custom_args.items():
                if k not in model_inputs and v is not None:
                    model_inputs[k] = v

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, `pad_token_id` must be defined.")

            next_tokens = (
                next_tokens * unfinished_sequences
                + pad_token_id * (1 - unfinished_sequences)
            )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        # Preserve custom AdaptVis args across generation updates.
        for k, v in custom_args.items():
            if v is not None:
                model_kwargs[k] = v

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
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

        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )

    return input_ids


def change_greedy_to_add_weight():
    transformers.generation.utils.GenerationMixin._greedy_search = _add_weight_greedy_search


# ============================================================
# LLaVA wrapper
# ============================================================

class LlavaWrapper:
    def __init__(self, root_dir, device, method):
        if method in ["scaling_vis", "adapt_vis"]:
            self.model = LlavaForConditionalGenerationScal.from_pretrained(
                MODEL,
                revision="a272c74",
                cache_dir=root_dir,
                ignore_mismatched_sizes=True,
            ).eval().to(device)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL,
                revision="a272c74",
                cache_dir=root_dir,
                ignore_mismatched_sizes=True,
            ).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            MODEL,
            revision="a272c74",
            cache_dir=root_dir,
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            MODEL,
            revision="a272c74",
            cache_dir=root_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL,
            revision="a272c74",
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

            text_feats = self.model.llava.get_text_features(**text_input).cpu().numpy()[:, 0, :]
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
    def get_out_scores_wh_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        option,
        threshold,
        weight1,
        weight2,
    ):
        scores = []
        index_of_total = 0
        processed_count = 0
        skipped_count = 0
        acc = 0
        correct_id = []
        processed_sample_ids = []

        sample_id_set = load_probe_sample_id_set()

        qst_ans_file = f"prompts/{dataset}_with_answer_{option}_options.jsonl"
        prompt_list = []
        answer_list = []

        with open(qst_ans_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        SAMPLE = _env_flag("SAMPLE", "False")
        TEST = _env_flag("TEST_MODE", "False")
        total_data_count = len(prompt_list)

        if SAMPLE:
            idx_file_path = f"./output/sampled_idx_{dataset}.npy"

            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
                sampled_indices.sort()
                np.save(idx_file_path, np.array(sampled_indices))

            if TEST:
                all_indices = set(range(total_data_count))
                sampled_indices = sorted(list(all_indices - set(sampled_indices)))

            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        if method == "scaling_vis":
            save_attn_dir = f"./output/{dataset}_scaling_w{weight:.2f}"
        elif method == "adapt_vis":
            save_attn_dir = f"./output/{dataset}_adapt_th{threshold:.2f}_w1{weight1:.2f}_w2{weight2:.2f}"
        else:
            save_attn_dir = f"./output/{dataset}_{method}"

        os.makedirs(save_attn_dir, exist_ok=True)

        output_file_path = make_tagged_output_path(
            dataset=dataset,
            method=method,
            weight=weight,
            option=option,
            test_flag=TEST,
        )
        print(f"[OUTPUT] result path = {output_file_path}")

        results = []

        for batch in tqdm(joint_loader):
            if sample_id_set is not None and index_of_total not in sample_id_set:
                skipped_count += 1
                index_of_total += 1
                continue

            batch_scores = []
            os.environ["SAVE_ATTN_PATH"] = f"{save_attn_dir}/{index_of_total}/"
            os.makedirs(os.environ["SAVE_ATTN_PATH"], exist_ok=True)

            for i_option in batch["image_options"]:
                im_scores = []

                for image in i_option:
                    sample_id = int(index_of_total)
                    prompt = prompt_list[index_of_total]

                    single_input = self.processor(
                        text=prompt,
                        images=image,
                        padding="max_length",
                        return_tensors="pt",
                        max_length=77,
                    ).to(self.device)

                    input_len = len(single_input["input_ids"][-1])
                    keys = build_legacy_image_keys(single_input["input_ids"])

                    selected_weight = None
                    uncertainty = None

                    if method == "scaling_vis":
                        change_greedy_to_add_weight()
                        selected_weight = float(weight)

                        output = self.model.generate(
                            **single_input,
                            keys=keys,
                            weight=selected_weight,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                        uncertainty = first_step_confidence_softmax(output)

                    elif method == "adapt_vis":
                        change_greedy_to_add_weight()

                        # First pass: original/main-style confidence.
                        # Do not pass keys here.
                        first_output = self.model.generate(
                            **single_input,
                            weight=1.0,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                        uncertainty = first_step_confidence_softmax(first_output)
                        print(uncertainty, threshold)

                        selected_weight = float(weight1) if uncertainty < threshold else float(weight2)

                        # Second pass: actual AdaptVis intervention.
                        output = self.model.generate(
                            **single_input,
                            keys=keys,
                            weight=selected_weight,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                    else:
                        output = self.model.generate(
                            **single_input,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                        uncertainty = first_step_confidence_raw(output)

                    gen = decode_generated(self.processor, output, input_len=input_len)
                    golden = answer_list[index_of_total][0]
                    is_correct = is_generation_correct(golden, gen)

                    print(
                        f"Prompt: {prompt}\n"
                        f"Generation: {gen}\n"
                        f"Golden: {golden}\n"
                        f"Correct: {is_correct}"
                    )

                    if is_correct:
                        acc += 1
                        correct_id.append(index_of_total)

                    c_option = batch["caption_options"]
                    if len(list(c_option)) == 2:
                        answers = [1, 0] if is_correct else [0, 1]
                    else:
                        answers = [1, 0, 0, 0] if is_correct else [0, 0, 1, 0]

                    result = {
                        "sample_id": sample_id,
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": golden,
                        "Correct": bool(is_correct),
                        "Uncertainty": float(uncertainty) if uncertainty is not None else None,
                        "selected_weight": (
                            float(selected_weight)
                            if selected_weight is not None
                            else None
                        ),
                        "method": method,
                        "weight": float(weight),
                        "threshold": float(threshold),
                        "weight1": float(weight1),
                        "weight2": float(weight2),
                        "probe_run_tag": os.getenv("PROBE_RUN_TAG", ""),
                    }
                    results.append(result)

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    processed_count += 1
                    processed_sample_ids.append(sample_id)
                    index_of_total += 1

                if len(im_scores) > 0:
                    batch_scores.append(np.concatenate(im_scores, axis=-1))

            if len(batch_scores) > 0:
                scores.append(batch_scores)

            with open(output_file_path, "w", encoding="utf-8") as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)

            denom = processed_count if processed_count > 0 else 1
            print(
                f"[RUNNING] acc={acc}/{processed_count}={acc / denom:.6f}, "
                f"scanned={index_of_total}, skipped={skipped_count}"
            )

        denom = processed_count if processed_count > 0 else 1
        final_acc = acc / denom

        print(
            f"[FINAL] acc={acc}/{processed_count}={final_acc:.6f}, "
            f"scanned={index_of_total}, skipped={skipped_count}"
        )

        output_score_file = output_file_path.replace(".json", "scores.json")
        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump(
                {
                    "acc": final_acc,
                    "correct_id": correct_id,
                    "processed_count": processed_count,
                    "skipped_count": skipped_count,
                    "processed_sample_ids": processed_sample_ids,
                    "sample_filter_file": os.getenv("PROBE_SAMPLE_IDS_FILE", ""),
                    "probe_run_tag": os.getenv("PROBE_RUN_TAG", ""),
                    "method": method,
                    "weight": float(weight),
                    "threshold": float(threshold),
                    "weight1": float(weight1),
                    "weight2": float(weight2),
                    "sample_enabled": SAMPLE,
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        if len(scores) > 0:
            all_scores = np.concatenate(scores, axis=0)
        else:
            if option == "two":
                all_scores = np.zeros((0, 2, 1))
            else:
                all_scores = np.zeros((0, 4, 1))

        if dataset in ["Controlled_Images_B", "Controlled_Images_A"]:
            return all_scores, []

        return final_acc, correct_id

    @torch.no_grad()
    def get_judge_scores_vsr_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        threshold,
        weight1,
        weight2,
    ):
        index_of_total = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        results = []

        save_attn_dir = f"./outputs/{dataset}_weight{weight:.2f}"
        os.makedirs(save_attn_dir, exist_ok=True)

        for batch in tqdm(joint_loader):
            os.environ["SAVE_ATTN_PATH"] = f"{save_attn_dir}/{index_of_total}/"
            os.makedirs(os.environ["SAVE_ATTN_PATH"], exist_ok=True)

            for i_option in batch["image_options"]:
                for c_option in batch["caption_options"]:
                    prompt = (
                        "User: \n Determine whether the description about the spatial relationship "
                        "is correct or not. Answer with yes or no: "
                    )
                    qst = [prompt] * len(list(c_option))
                    end_fix = [" Assistant:"] * len(list(c_option))
                    concatenated_list = [
                        s1 + s2 + s3
                        for s1, s2, s3 in zip(qst, c_option, end_fix)
                    ]

                    for idx, text in enumerate(concatenated_list):
                        image = list(i_option)[idx]
                        single_input = self.processor(
                            text=text,
                            images=image,
                            padding="max_length",
                            return_tensors="pt",
                            max_length=77,
                        ).to(self.device)

                        input_len = len(single_input["input_ids"][-1])
                        keys = build_legacy_image_keys(single_input["input_ids"])

                        selected_weight = None

                        if method == "scaling_vis":
                            change_greedy_to_add_weight()
                            selected_weight = float(weight)

                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=selected_weight,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            uncertainty = first_step_confidence_softmax(output)

                        elif method == "adapt_vis":
                            change_greedy_to_add_weight()

                            first_output = self.model.generate(
                                **single_input,
                                weight=1.0,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            uncertainty = first_step_confidence_raw(first_output)

                            selected_weight = float(weight1) if uncertainty < threshold else float(weight2)

                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=selected_weight,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                        else:
                            output = self.model.generate(
                                **single_input,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            uncertainty = first_step_confidence_raw(output)

                        gen = decode_generated(self.processor, output, input_len=input_len)
                        label = int(batch["labels"][0][idx])

                        if label == 1:
                            TP += 1 if "Yes" in gen else 0
                            FN += 1 if "Yes" not in gen else 0
                        else:
                            TN += 1 if "No" in gen else 0
                            FP += 1 if "No" not in gen else 0

                        gold = "Yes" if label == 1 else "No"
                        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

                        results.append(
                            {
                                "Prompt": text,
                                "Generation": gen,
                                "Golden": gold,
                                "Uncertainty": float(uncertainty),
                                "selected_weight": (
                                    float(selected_weight)
                                    if selected_weight is not None
                                    else None
                                ),
                            }
                        )

                        index_of_total += 1

        precision = TP / max(TP + FN, 1)
        recall = TN / max(TN + FP, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-12)
        acc = (TN + TP) / max(TN + TP + FN + FP, 1)

        print(
            f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
            f"Accuracy: {acc}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1_score}"
        )

        output_file_path = f"./outputs/results_{dataset}_{method}_{weight}.json"
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)

        output_score_file = output_file_path.replace(".json", "_scores.json")
        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump(
                {
                    "acc": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score,
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        return TP, TN, FP, FN
