import os
import re
import json
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)

MODEL = os.getenv("LLAVA16_MODEL", "llava-hf/llava-v1.6-vicuna-7b-hf")


def _norm_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def _is_correct(gold, gen):
    gold = _norm_gold(gold)
    gen = str(gen)
    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False
    return bool(ok)


def _generation_scores(output):
    if hasattr(output, "scores"):
        return output.scores
    if isinstance(output, dict):
        return output.get("scores", None)
    return output["scores"]


def _generation_sequences(output):
    if hasattr(output, "sequences"):
        return output.sequences
    if isinstance(output, dict):
        return output.get("sequences", None)
    return output["sequences"]


def _first_step_confidence(output, use_softmax=True, round_digits=2):
    scores = _generation_scores(output)
    if scores is None or len(scores) == 0:
        return 0.0
    x = scores[0]
    if use_softmax:
        x = torch.nn.functional.softmax(x, dim=-1)
    val = float(torch.max(x[0]).detach().float().cpu())
    if round_digits is not None:
        val = round(val, int(round_digits))
    return val


def _decode_generated(processor, output, prompt_len):
    seq = _generation_sequences(output)
    return processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True).strip()


def _strip_prompt(raw_prompt):
    q = str(raw_prompt)
    q = q.replace("<image>", "").strip()
    q = re.sub(r"^USER:\s*", "", q, flags=re.I).strip()
    q = re.sub(r"^User:\s*", "", q, flags=re.I).strip()
    q = re.sub(r"ASSISTANT:\s*$", "", q, flags=re.I).strip()
    q = re.sub(r"Assistant:\s*$", "", q, flags=re.I).strip()
    return q


def _build_chat_prompt(processor, raw_prompt):
    question = _strip_prompt(raw_prompt)
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        return "<image>\nUSER: " + question + "\nASSISTANT:"


def _make_prompt(processor, raw_prompt):
    # Default is llava15.py-compatible: use prompt exactly as prompts/*.jsonl stores it.
    # Set LLAVA16_USE_CHAT_TEMPLATE=1 only for debugging HF chat-template behavior.
    if os.getenv("LLAVA16_USE_CHAT_TEMPLATE", "0") == "1":
        return _build_chat_prompt(processor, raw_prompt)
    return str(raw_prompt)


class _AdaptVisContext:
    def __init__(self, layers=None, num_layers=32):
        self.active = False
        self.in_lm = False
        self.weight = 1.0
        if layers is None:
            layers = list(range(num_layers))
        self.layers = set(int(x) for x in layers)
        self.num_layers = int(num_layers)
        self.call_idx = 0
        self.modified_calls = 0
        self.input_len = None
        self.image_positions = []
        self.image_token_index = None

    def reset_for_sample(self, input_ids, image_token_index, weight=1.0, active=False):
        self.call_idx = 0
        self.modified_calls = 0
        self.weight = float(weight)
        self.active = bool(active)
        self.input_len = int(input_ids.shape[-1])
        self.image_token_index = int(image_token_index)
        ids = input_ids[0]
        pos = torch.where(ids == int(image_token_index))[0]
        self.image_positions = [int(x.detach().cpu()) for x in pos]

    def get_image_span(self, kv_len):
        if not self.image_positions:
            return None
        first = min(self.image_positions)
        last = max(self.image_positions)
        n_placeholder = len(self.image_positions)
        if int(kv_len) == int(self.input_len):
            return first, last + 1
        image_len = int(kv_len) - (int(self.input_len) - int(n_placeholder))
        if image_len <= 0:
            return None
        start = first
        end = start + image_len
        if start < 0 or end > int(kv_len) or end <= start:
            return None
        return start, end


def _apply_adaptvis_to_image_logits(attn_logits, ctx):
    # Same multiplication rule as llava15 paper-code path, but applied to the
    # detected LLaVA-1.6 whole image-token span rather than fixed 24x24 patch ids.
    if not ctx.active:
        return attn_logits
    if not ctx.in_lm:
        return attn_logits
    if not torch.is_tensor(attn_logits) or attn_logits.dim() != 4:
        return attn_logits

    q_len = attn_logits.shape[-2]
    kv_len = attn_logits.shape[-1]
    if q_len <= 1:
        return attn_logits

    layer = ctx.call_idx % ctx.num_layers
    ctx.call_idx += 1
    if layer not in ctx.layers:
        return attn_logits

    span = ctx.get_image_span(kv_len)
    if span is None:
        return attn_logits
    image_start, image_end = span
    image_start = max(0, int(image_start))
    image_end = min(int(kv_len), int(image_end))
    if image_end <= image_start:
        return attn_logits

    x = attn_logits.clone()
    region = x[..., :, image_start:image_end]
    neg_mask = region < 0
    region_new = torch.where(neg_mask, region * float(ctx.weight), region)
    x[..., :, image_start:image_end] = region_new
    ctx.modified_calls += 1
    return x


@contextmanager
def _patch_language_model_and_softmax(model, ctx):
    old_f_softmax = F.softmax
    old_torch_softmax = torch.softmax
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise AttributeError("LLaVA-NeXT model has no language_model")
    old_lm_forward = language_model.forward

    def wrapped_lm_forward(*args, **kwargs):
        old_flag = ctx.in_lm
        ctx.in_lm = True
        try:
            return old_lm_forward(*args, **kwargs)
        finally:
            ctx.in_lm = old_flag

    def wrapped_f_softmax(input, *args, **kwargs):
        dim = kwargs.get("dim", None)
        if dim is None and len(args) > 0:
            dim = args[0]
        if torch.is_tensor(input) and (dim == -1 or dim == input.dim() - 1):
            input = _apply_adaptvis_to_image_logits(input, ctx)
        return old_f_softmax(input, *args, **kwargs)

    def wrapped_torch_softmax(input, *args, **kwargs):
        dim = kwargs.get("dim", None)
        if dim is None and len(args) > 0:
            dim = args[0]
        if torch.is_tensor(input) and (dim == -1 or dim == input.dim() - 1):
            input = _apply_adaptvis_to_image_logits(input, ctx)
        return old_torch_softmax(input, *args, **kwargs)

    language_model.forward = wrapped_lm_forward
    F.softmax = wrapped_f_softmax
    torch.softmax = wrapped_torch_softmax
    try:
        yield
    finally:
        language_model.forward = old_lm_forward
        F.softmax = old_f_softmax
        torch.softmax = old_torch_softmax


def _parse_layers_from_env(num_layers=32):
    text = os.getenv("LLAVA16_ADAPTVIS_LAYERS", "").strip()
    if not text:
        text = os.getenv("ADAPTVIS_LAYERS", "").strip()
    if not text or text.lower() in ["all", "none"]:
        return list(range(num_layers))
    return [int(x) for x in text.split(",") if x.strip()]


class LlavaWrapper:
    def __init__(self, root_dir, device, method):
        self.device = device
        self.method = method

        image_processor = LlavaNextImageProcessor.from_pretrained(MODEL, cache_dir=root_dir)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=root_dir, use_fast=False)
        self.processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.feature_extractor = image_processor

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        try:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                MODEL,
                cache_dir=root_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        except TypeError:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                MODEL,
                cache_dir=root_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        self.model = self.model.eval().to(device)

        self.num_layers = int(os.getenv("LLAVA16_NUM_LAYERS", "32"))
        self.layers = _parse_layers_from_env(self.num_layers)
        self.ctx = _AdaptVisContext(layers=self.layers, num_layers=self.num_layers)

        # llava15.py-compatible defaults.
        self.max_length = int(os.getenv("LLAVA16_MAX_LENGTH", "77"))
        self.max_new_tokens = int(os.getenv("LLAVA16_MAX_NEW_TOKENS", "100"))
        self.conf_round_digits = int(os.getenv("LLAVA16_CONF_ROUND_DIGITS", "2"))
        self.conf_use_softmax = os.getenv("LLAVA16_CONF_USE_SOFTMAX", "1") != "0"

        print("[LLaVA-1.6 wrapper: llava15-consistent control flow]")
        print("MODEL:", MODEL)
        print("method:", method)
        print("layers:", self.layers)
        print("max_length:", self.max_length)
        print("max_new_tokens:", self.max_new_tokens)
        print("use_chat_template:", os.getenv("LLAVA16_USE_CHAT_TEMPLATE", "0"))

    @torch.no_grad()
    def _generate_one(self, image, prompt, weight=1.0, active=False):
        text = _make_prompt(self.processor, prompt)
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]

        image_token_index = getattr(getattr(self.model, "config", None), "image_token_index", None)
        if image_token_index is None:
            image_token_index = 32000

        self.ctx.reset_for_sample(
            inputs["input_ids"],
            image_token_index=image_token_index,
            weight=weight,
            active=active,
        )

        with _patch_language_model_and_softmax(self.model, self.ctx):
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen = _decode_generated(self.processor, output, prompt_len)
        conf = _first_step_confidence(
            output,
            use_softmax=self.conf_use_softmax,
            round_digits=self.conf_round_digits,
        )
        return gen, conf, int(self.ctx.modified_calls)

    def _run_generation_by_method(self, image, prompt, method, weight, threshold, weight1, weight2):
        method = str(method)

        if method == "scaling_vis":
            gen, conf, modified = self._generate_one(image, prompt, weight=weight, active=True)
            return gen, conf, modified, float(weight), "scaling_vis"

        if method == "adapt_vis":
            # llava15.py: base generation -> confidence -> weighted generation.
            _, base_conf, _ = self._generate_one(image, prompt, weight=1.0, active=False)
            if float(base_conf) < float(threshold):
                selected_weight = float(weight1)
                branch = "low_conf_lt_threshold"
            else:
                selected_weight = float(weight2)
                branch = "high_conf_ge_threshold"
            gen, _, modified = self._generate_one(image, prompt, weight=selected_weight, active=True)
            return gen, base_conf, modified, selected_weight, branch

        gen, conf, modified = self._generate_one(image, prompt, weight=1.0, active=False)
        return gen, conf, modified, 1.0, "base"

    @torch.no_grad()
    def get_out_scores_wh_batched(self, dataset, joint_loader, method, weight, option, threshold=1.0, weight1=1.0, weight2=1.0):
        scores = []
        index_of_total = 0
        acc = 0
        correct_id = []

        qst_ans_file = f"prompts/{dataset}_with_answer_{option}_options.jsonl"
        prompt_list = []
        answer_list = []
        with open(qst_ans_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        SAMPLE = True
        TEST = os.getenv("TEST_MODE", "False") == "True"
        total_data_count = len(prompt_list)
        if SAMPLE:
            idx_file_path = f"./output/sampled_idx_{dataset}.npy"
            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
                sampled_indices.sort()
                os.makedirs("./output", exist_ok=True)
                np.save(idx_file_path, np.array(sampled_indices))
            if TEST:
                all_indices = set(range(total_data_count))
                sampled_indices = sorted(list(all_indices - set(sampled_indices)))
            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        results = []
        for batch in tqdm(joint_loader, desc=f"llava1.6 {dataset} {method}"):
            batch_scores = []
            for i_option in batch["image_options"]:
                images = list(i_option) if isinstance(i_option, (list, tuple)) else [i_option]
                im_scores = []
                for image in images:
                    if index_of_total >= len(prompt_list):
                        break
                    prompt = prompt_list[index_of_total]
                    gold = answer_list[index_of_total][0]

                    gen, conf, modified, selected_weight, branch = self._run_generation_by_method(
                        image=image,
                        prompt=prompt,
                        method=method,
                        weight=weight,
                        threshold=threshold,
                        weight1=weight1,
                        weight2=weight2,
                    )

                    corr = _is_correct(gold, gen)
                    if corr:
                        acc += 1
                        correct_id.append(index_of_total)

                    c_option = batch["caption_options"]
                    n_options = len(list(c_option))
                    if n_options == 4:
                        answers = [1, 0, 0, 0] if corr else [0, 0, 1, 0]
                    elif n_options == 2:
                        answers = [1, 0] if corr else [0, 1]
                    else:
                        answers = [0] * n_options
                        answers[0 if corr else min(1, n_options - 1)] = 1
                    im_scores.append(np.expand_dims(np.array(answers), -1))

                    print(f"Prompt: {prompt}\nGeneration: {gen}\nGolden: {gold}")
                    print(
                        f"[llava1.6] sid={index_of_total} correct={corr} "
                        f"conf={conf:.4f} selected_weight={selected_weight} "
                        f"branch={branch} modified_calls={modified}"
                    )

                    results.append({
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": gold,
                        "Correct": bool(corr),
                        "Uncertainty": float(conf),
                        "Selected_weight": float(selected_weight),
                        "Branch": branch,
                        "Modified_calls": int(modified),
                    })
                    index_of_total += 1
                if im_scores:
                    batch_scores.append(np.concatenate(im_scores, axis=-1))
            if batch_scores:
                scores.append(batch_scores)

        os.makedirs("./output", exist_ok=True)
        output_file_path = f"./output/results1.6_{dataset}_{method}_{weight}_{option}option_{TEST}.json"
        print("Saving results to", output_file_path)
        with open(output_file_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)

        final_acc = acc / max(index_of_total, 1)
        print(acc, index_of_total, final_acc)
        print(final_acc)
        output_score_file = output_file_path.replace(".json", "scores.json")
        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump({"acc": final_acc, "correct_id": correct_id}, fout, ensure_ascii=False, indent=4)

        all_scores = np.concatenate(scores, axis=0) if scores else np.array([])
        if dataset in ["Controlled_Images_B", "Controlled_Images_A"]:
            return (all_scores, [])
        return (final_acc, correct_id)

    @torch.no_grad()
    def get_judge_scores_vsr_batched(self, dataset, joint_loader, method, weight, threshold, weight1, weight2):
        TP, TN, FP, FN = 0, 0, 0, 0
        results = []
        for batch in tqdm(joint_loader, desc=f"llava1.6 {dataset} {method}"):
            for i_option in batch["image_options"]:
                images = list(i_option) if isinstance(i_option, (list, tuple)) else [i_option]
                captions = list(batch["caption_options"])
                labels = list(batch["labels"][0]) if isinstance(batch["labels"], (list, tuple)) else list(batch["labels"])
                for idx, image in enumerate(images):
                    caption = captions[idx] if idx < len(captions) else captions[0]
                    label = int(labels[idx]) if idx < len(labels) else int(labels[0])
                    prompt = (
                        "User: \n Determine whether the description about the spatial relationship is correct or not.\n"
                        "Answer with yes or no: " + str(caption) + " Assistant:"
                    )
                    gen, conf, modified, selected_weight, branch = self._run_generation_by_method(
                        image=image,
                        prompt=prompt,
                        method=method,
                        weight=weight,
                        threshold=threshold,
                        weight1=weight1,
                        weight2=weight2,
                    )
                    if label == 1:
                        TP += 1 if "yes" in gen.lower() else 0
                        FN += 1 if "yes" not in gen.lower() else 0
                        gold = "Yes"
                    else:
                        TN += 1 if "no" in gen.lower() else 0
                        FP += 1 if "no" not in gen.lower() else 0
                        gold = "No"
                    results.append({
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": gold,
                        "Uncertainty": float(conf),
                        "Selected_weight": float(selected_weight),
                        "Branch": branch,
                        "Modified_calls": int(modified),
                    })

        precision = TP / max((TP + FN), 1)
        recall = TN / max((TN + FP), 1)
        f1_score = 2 * precision * recall / max((precision + recall), 1e-12)
        acc = (TN + TP) / max((TN + TP + FN + FP), 1)
        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")

        os.makedirs("./outputs", exist_ok=True)
        output_file_path = f"./outputs/results_{dataset}_{method}_{weight}.json"
        with open(output_file_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
        output_score_file = output_file_path.replace(".json", "_scores.json")
        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump({"acc": acc, "precision": precision, "recall": recall, "f1": f1_score}, fout, ensure_ascii=False, indent=4)
        return (TP, TN, FP, FN)
