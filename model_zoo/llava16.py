"""
LLaVA-1.6 / LLaVA-NeXT wrapper for AdaptVis.

Drop this file into:
    model_zoo/llava16.py

This version is designed for the HuggingFace LLaVA-NeXT implementation
(e.g. llava-hf/llava-v1.6-vicuna-7b-hf). It keeps the public methods used
by AdaptVis: get_out_scores_wh_batched() and get_judge_scores_vsr_batched().
"""

import os
import re
import json
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

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


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def _norm_gold(x):
    if isinstance(x, (list, tuple)):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def _is_correct(gold, gen):
    gold = _norm_gold(gold)
    gen = str(gen).strip()
    if not gold:
        return False
    ok = (gold in gen) or (gold.lower() in gen.lower())
    # Original AdaptVis special case.
    if gold.lower() == "on" and "front" in gen.lower():
        ok = False
    return bool(ok)


def _strip_prompt(raw_prompt):
    """Remove old LLaVA-1.5 style wrappers if the prompt file already has them."""
    q = str(raw_prompt)
    q = q.replace("<image>", "").strip()
    q = re.sub(r"^\s*USER:\s*", "", q, flags=re.I).strip()
    q = re.sub(r"^\s*User:\s*", "", q, flags=re.I).strip()
    q = re.sub(r"\s*ASSISTANT:\s*$", "", q, flags=re.I).strip()
    q = re.sub(r"\s*Assistant:\s*$", "", q, flags=re.I).strip()
    q = re.sub(r"^\s*\[INST\]\s*", "", q).strip()
    q = re.sub(r"\s*\[/INST\]\s*$", "", q).strip()
    return q


def _format_controlled_relation_prompt(raw_prompt, choices=None):
    """Force ARO Controlled Images to produce a single relation label.

    The Controlled_Images_A/B prompts ask about spatial relations, but an
    open-ended LLaVA prompt often generates full captions like
    "The lemon is on the table."  For this benchmark wrapper we need the
    model to output one label so the downstream string match is meaningful.
    """
    question = _strip_prompt(raw_prompt)
    if choices is None:
        choices = ["Left", "Right", "On", "Under"]
    choice_text = ", ".join(choices)
    return (
        "Answer the spatial relationship question based on the image.\n"
        f"Choose exactly one option from: {choice_text}.\n"
        "Output only the option word. Do not output a sentence or explanation.\n"
        f"Question: {question}"
    )


def _build_prompt(processor, raw_prompt, force_relation_options=False, choices=None):
    """
    Build a stable LLaVA/Vicuna-style prompt.

    We intentionally do not call processor.apply_chat_template() here. Some
    HF LLaVA-NeXT chat templates insert [INST] ... [/INST]. Under attention-logit
    interventions, the model can copy the template terminator as the answer.
    The original LLaVA-style USER/ASSISTANT template is more stable for this
    AdaptVis evaluation.
    """
    if force_relation_options:
        question = _format_controlled_relation_prompt(raw_prompt, choices=choices)
    else:
        question = _strip_prompt(raw_prompt)

    return f"USER: <image>\n{question}\nASSISTANT:"

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


def _first_step_confidence(output):
    scores = _generation_scores(output)
    if scores is None or len(scores) == 0:
        return 0.0
    probs = torch.softmax(scores[0].detach().float(), dim=-1)
    return float(probs[0].max().cpu())


def _decode_generated(processor, output, prompt_len=None, input_ids=None, debug=False):
    """Decode only newly generated tokens.

    For multimodal generation, prompt_len can be unreliable because image
    placeholders may be expanded before the language model. The most stable
    signal is len(output.scores): one score tensor per generated token.
    """
    seq = _generation_sequences(output)
    if seq is None:
        return ""

    full_ids = seq[0].detach().cpu()
    prompt_len = int(prompt_len or 0)

    scores = _generation_scores(output)
    gen_len = len(scores) if scores is not None else 0

    if gen_len > 0 and full_ids.numel() >= gen_len:
        gen_ids = full_ids[-gen_len:]
        mode = "last_scores_len_tokens"
    elif prompt_len > 0 and full_ids.numel() > prompt_len:
        gen_ids = full_ids[prompt_len:]
        mode = "full_sequence_slice"
    else:
        gen_ids = full_ids
        mode = "generated_only_fallback"

    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    # Defensive cleanup for broken chat-template fallbacks or legacy outputs.
    text = re.sub(r"^\s*\[/INST\]\s*", "", text).strip()
    text = re.sub(r"^\s*ASSISTANT:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^\s*Assistant:\s*", "", text).strip()

    if debug:
        raw = processor.decode(gen_ids, skip_special_tokens=False)
        print(
            f"[llava16 decode] mode={mode} prompt_len={prompt_len} "
            f"seq_len={full_ids.numel()} gen_len={gen_ids.numel()} "
            f"text={text!r} raw={raw!r} ids={gen_ids[:20].tolist()}"
        )

    return text


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _num_caption_options(caption_options):
    c = caption_options
    if isinstance(c, torch.Tensor):
        return int(c.numel())
    if isinstance(c, (list, tuple)):
        if len(c) == 1 and isinstance(c[0], (list, tuple)):
            return len(c[0])
        return len(c)
    return 1


def _parse_layers_from_env(num_layers=32):
    text = os.getenv("LLAVA16_ADAPTVIS_LAYERS", "").strip()
    if not text:
        text = os.getenv("ADAPTVIS_LAYERS", "").strip()
    if not text or text.lower() in {"all", "none"}:
        return list(range(int(num_layers)))

    layers = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


# -----------------------------------------------------------------------------
# AdaptVis attention-logit patching
# -----------------------------------------------------------------------------

class _AdaptVisContext:
    """
    Tracks the image-token span in the language-model sequence.

    HF LLaVA-NeXT has two possible behaviours depending on transformers version:
      1. input_ids already contains many repeated image tokens; then kv_len == input_len.
      2. input_ids contains a single image placeholder; the model expands it before
         the language model; then kv_len > input_len.

    This class supports both cases.
    """

    def __init__(self, layers=None, num_layers=32, debug=False):
        self.active = False
        self.in_lm = False
        self.weight = 1.0
        self.layers = set(int(x) for x in (layers if layers is not None else range(num_layers)))
        self.num_layers = int(num_layers)
        self.debug = bool(debug)

        self.call_idx = 0
        self.modified_calls = 0
        self.input_len = 0
        self.image_positions: List[int] = []
        self.image_token_index: Optional[int] = None
        self._printed_span = False

    def reset_for_sample(self, input_ids, image_token_index, weight=1.0, active=False):
        self.call_idx = 0
        self.modified_calls = 0
        self.weight = float(weight)
        self.active = bool(active)
        self.input_len = int(input_ids.shape[-1])
        self.image_token_index = int(image_token_index)
        self._printed_span = False

        ids = input_ids[0].detach()
        pos = torch.where(ids == int(image_token_index))[0]
        self.image_positions = [int(x.cpu()) for x in pos]

    def get_image_span(self, kv_len) -> Optional[Tuple[int, int]]:
        kv_len = int(kv_len)
        if not self.image_positions:
            return None

        first = min(self.image_positions)
        last = max(self.image_positions)
        n_placeholder = len(self.image_positions)

        # Newer HF processor: image tokens are already expanded in input_ids.
        if kv_len == int(self.input_len):
            span = (first, last + 1)
        else:
            # Older behaviour: one placeholder is expanded to image features before LM.
            expanded_image_len = kv_len - (int(self.input_len) - int(n_placeholder))
            if expanded_image_len <= 0:
                return None
            span = (first, first + expanded_image_len)

        image_start, image_end = span
        image_start = max(0, int(image_start))
        image_end = min(kv_len, int(image_end))
        if image_end <= image_start:
            return None

        if self.debug and not self._printed_span:
            print(
                f"[llava16 span] input_len={self.input_len} kv_len={kv_len} "
                f"n_image_placeholders={n_placeholder} span=({image_start}, {image_end})"
            )
            self._printed_span = True
        return image_start, image_end


def _apply_adaptvis_to_image_logits(attn_logits, ctx: _AdaptVisContext):
    """
    AdaptVis intervention: scale negative attention logits whose KEY positions
    are visual tokens. This matches the original idea without assuming LLaVA-1.5's
    fixed 24x24 patch ordering.
    """
    if not ctx.active or not ctx.in_lm:
        return attn_logits
    if not torch.is_tensor(attn_logits) or attn_logits.dim() != 4:
        return attn_logits

    q_len = int(attn_logits.shape[-2])
    kv_len = int(attn_logits.shape[-1])

    # Only modify the prefill pass. During decoding q_len == 1 and cached keys
    # already contain the modified prefill state.
    if q_len <= 1:
        return attn_logits

    # One self-attention softmax per decoder layer in eager attention.
    layer = ctx.call_idx % ctx.num_layers
    ctx.call_idx += 1
    if layer not in ctx.layers:
        return attn_logits

    span = ctx.get_image_span(kv_len)
    if span is None:
        return attn_logits
    image_start, image_end = span

    x = attn_logits.clone()

    # AdaptVis / ScalingVis should intervene only on the last prefill query.
    # Do not modify every query row. Modifying all rows corrupts the prompt
    # representation and can make the model generate template tokens such as
    # [/INST].
    q_pos = q_len - 1
    region = x[..., q_pos:q_pos + 1, image_start:image_end]

    # Original AdaptVis scaling only acts on negative logits.
    region_new = torch.where(region < 0, region * float(ctx.weight), region)
    x[..., q_pos:q_pos + 1, image_start:image_end] = region_new

    ctx.modified_calls += 1
    return x


@contextmanager
def _patch_language_model_and_softmax(model, ctx: _AdaptVisContext):
    """
    Patches softmax only while we are inside model.language_model.forward().
    Requires eager attention; the wrapper sets attn_implementation='eager'.
    """
    old_f_softmax = F.softmax
    old_torch_softmax = torch.softmax

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise AttributeError("Expected HuggingFace LlavaNextForConditionalGeneration.language_model")

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


# -----------------------------------------------------------------------------
# Public wrapper used by AdaptVis
# -----------------------------------------------------------------------------

class LlavaWrapper:
    def __init__(self, root_dir, device, method):
        self.device = device
        self.method = method

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        # Load processor directly from the checkpoint. This preserves the correct
        # LLaVA-NeXT chat template and image-processing metadata.
        try:
            self.processor = LlavaNextProcessor.from_pretrained(MODEL, cache_dir=root_dir)
            self.tokenizer = self.processor.tokenizer
            self.feature_extractor = self.processor.image_processor
        except Exception:
            image_processor = LlavaNextImageProcessor.from_pretrained(MODEL, cache_dir=root_dir)
            tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=root_dir, use_fast=False)
            self.processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)
            self.tokenizer = tokenizer
            self.feature_extractor = image_processor

        self.tokenizer.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        # Force eager attention where possible; otherwise monkey-patching softmax
        # will not catch FlashAttention/SDPA kernels.
        for cfg in [getattr(self.model, "config", None), getattr(getattr(self.model, "language_model", None), "config", None)]:
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "eager"

        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.model.generation_config, "eos_token_id", None) is None:
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # Newer transformers expects these processor attributes for correct
        # LLaVA-NeXT image-token expansion. Fill them if missing.
        vision_cfg = getattr(getattr(self.model, "config", None), "vision_config", None)
        if getattr(self.processor, "patch_size", None) is None and vision_cfg is not None:
            self.processor.patch_size = getattr(vision_cfg, "patch_size", None)
        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            self.processor.vision_feature_select_strategy = getattr(self.model.config, "vision_feature_select_strategy", "default")
        if getattr(self.processor, "num_additional_image_tokens", None) is None:
            self.processor.num_additional_image_tokens = 1

        lm_cfg = getattr(getattr(self.model, "language_model", None), "config", None)
        default_layers = getattr(lm_cfg, "num_hidden_layers", 32)
        self.num_layers = int(os.getenv("LLAVA16_NUM_LAYERS", str(default_layers)))
        self.layers = _parse_layers_from_env(self.num_layers)
        self.ctx = _AdaptVisContext(
            layers=self.layers,
            num_layers=self.num_layers,
            debug=os.getenv("LLAVA16_DEBUG_SPAN", "0") == "1",
        )

        self.max_new_tokens = int(os.getenv("LLAVA16_MAX_NEW_TOKENS", "50"))

        print("[LLaVA-1.6 / LLaVA-NeXT wrapper]")
        print("MODEL:", MODEL)
        print("method:", method)
        print("num_layers:", self.num_layers)
        print("layers:", self.layers)
        print("tokenizer.padding_side:", self.processor.tokenizer.padding_side)
        print("max_new_tokens:", self.max_new_tokens)

    @torch.no_grad()
    def _generate_one(
        self,
        image,
        prompt,
        weight=1.0,
        active=False,
        max_new_tokens=None,
        force_relation_options=False,
        choices=None,
    ):
        hf_prompt = _build_prompt(
            self.processor,
            prompt,
            force_relation_options=force_relation_options,
            choices=choices,
        )
        if os.getenv("LLAVA16_DEBUG_PROMPT", "0") == "1":
            print("[llava16 prompt]", repr(hf_prompt[:1000]))
        inputs = self.processor(text=hf_prompt, images=image, return_tensors="pt").to(self.device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        image_token_index = getattr(getattr(self.model, "config", None), "image_token_index", None)
        if image_token_index is None:
            image_token_index = getattr(self.tokenizer, "convert_tokens_to_ids", lambda x: None)("<image>")
        if image_token_index is None or image_token_index < 0:
            image_token_index = 32000

        self.ctx.reset_for_sample(
            inputs["input_ids"],
            image_token_index=int(image_token_index),
            weight=float(weight),
            active=bool(active),
        )

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        with _patch_language_model_and_softmax(self.model, self.ctx):
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                min_new_tokens=int(os.getenv("LLAVA16_MIN_NEW_TOKENS", "1")),
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        debug_decode = os.getenv("LLAVA16_DEBUG_DECODE", "0") == "1"
        gen = _decode_generated(
            self.processor,
            output,
            prompt_len=prompt_len,
            input_ids=inputs.get("input_ids", None),
            debug=debug_decode,
        )
        conf = _first_step_confidence(output)
        return gen, conf, int(self.ctx.modified_calls)

    def _run_generation_by_method(
        self,
        image,
        prompt,
        method,
        weight,
        threshold,
        weight1,
        weight2,
        force_relation_options=False,
        choices=None,
    ):
        method = str(method)

        if method == "scaling_vis":
            gen, conf, modified = self._generate_one(
                image,
                prompt,
                weight=weight,
                active=True,
                force_relation_options=force_relation_options,
                choices=choices,
            )
            return gen, conf, modified, float(weight), "scaling_vis"

        if method == "adapt_vis":
            _, base_conf, _ = self._generate_one(
                image,
                prompt,
                weight=1.0,
                active=False,
                force_relation_options=force_relation_options,
                choices=choices,
            )
            selected_weight = float(weight1) if float(base_conf) < float(threshold) else float(weight2)
            branch = "low_conf_lt_threshold" if float(base_conf) < float(threshold) else "high_conf_ge_threshold"
            gen, _, modified = self._generate_one(
                image,
                prompt,
                weight=selected_weight,
                active=True,
                force_relation_options=force_relation_options,
                choices=choices,
            )
            return gen, base_conf, modified, selected_weight, branch

        gen, conf, modified = self._generate_one(
            image,
            prompt,
            weight=1.0,
            active=False,
            force_relation_options=force_relation_options,
            choices=choices,
        )
        return gen, conf, modified, 1.0, "base"

    @torch.no_grad()
    def get_out_scores_wh_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        option,
        threshold=1.0,
        weight1=1.0,
        weight2=1.0,
    ):
        qst_ans_file = f"prompts/{dataset}_with_answer_{option}_options.jsonl"

        prompt_list, answer_list = [], []
        with open(qst_ans_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        TEST = os.getenv("TEST_MODE", "False") == "True"

        # Use the original jsonl prompt directly.
        # Controlled_Images_A/B already contain prompts like:
        #   "Where is the X in relation to the Y? Answer with left, right, on or under."
        # Do not wrap the prompt again with extra option instructions; doing so
        # changes the final-query attention distribution and first-token confidence.
        force_relation_options = False
        relation_choices = None

        results = []
        scores = []
        correct_id = []
        acc = 0
        index_of_total = 0

        for batch in tqdm(joint_loader, desc=f"llava1.6 {dataset} {method}"):
            batch_scores = []

            for i_option in batch["image_options"]:
                images = _as_list(i_option)
                im_scores = []

                for image in images:
                    if index_of_total >= len(prompt_list):
                        break

                    prompt = prompt_list[index_of_total]
                    gold = _norm_gold(answer_list[index_of_total])

                    gen, conf, modified, selected_weight, branch = self._run_generation_by_method(
                        image=image,
                        prompt=prompt,
                        method=method,
                        weight=weight,
                        threshold=threshold,
                        weight1=weight1,
                        weight2=weight2,
                        force_relation_options=force_relation_options,
                        choices=relation_choices,
                    )

                    corr = _is_correct(gold, gen)
                    if corr:
                        acc += 1
                        correct_id.append(index_of_total)

                    n_options = _num_caption_options(batch.get("caption_options", []))
                    if n_options == 4:
                        answers = [1, 0, 0, 0] if corr else [0, 0, 1, 0]
                    elif n_options == 2:
                        answers = [1, 0] if corr else [0, 1]
                    else:
                        answers = [0] * max(n_options, 2)
                        answers[0 if corr else 1] = 1

                    im_scores.append(np.expand_dims(np.asarray(answers), -1))

                    print(
                        f"[{dataset}] sid={index_of_total} "
                        f"gold={gold} gen={gen} correct={corr} "
                        f"conf={conf:.4f} selected_weight={selected_weight} "
                        f"branch={branch} modified_calls={modified}"
                    )

                    results.append({
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": gold,
                        "Correct": bool(corr),
                        "Confidence": float(conf),
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
        if method == "adapt_vis":
            output_file_path = (
                f"./output/results1.6_{dataset}_{method}_"
                f"w1_{weight1}_w2_{weight2}_thr_{threshold}_{option}option_{TEST}.json"
            )
        else:
            output_file_path = f"./output/results1.6_{dataset}_{method}_{weight}_{option}option_{TEST}.json"

        print("Saving results to", output_file_path)
        with open(output_file_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)

        final_acc = acc / max(index_of_total, 1)
        print(acc, index_of_total, final_acc)

        output_score_file = output_file_path.replace(".json", "scores.json")
        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump({"acc": final_acc, "correct_id": correct_id}, fout, ensure_ascii=False, indent=4)

        all_scores = np.concatenate(scores, axis=0) if scores else np.empty((0,))
        if dataset in ["Controlled_Images_B", "Controlled_Images_A"]:
            return (all_scores, [])
        return (final_acc, correct_id)

    @torch.no_grad()
    def get_judge_scores_vsr_batched(self, dataset, joint_loader, method, weight, threshold, weight1, weight2):
        TP, TN, FP, FN = 0, 0, 0, 0
        force_relation_options = False
        relation_choices = None
        results = []

        for batch in tqdm(joint_loader, desc=f"llava1.6 {dataset} {method}"):
            image_options = _as_list(batch["image_options"])
            caption_options = _as_list(batch["caption_options"])
            labels_obj = batch.get("labels", [])

            for option_idx, i_option in enumerate(image_options):
                images = _as_list(i_option)
                captions = _as_list(caption_options[option_idx]) if option_idx < len(caption_options) else _as_list(caption_options[0])

                # labels are usually stored as batch['labels'][0][idx] in AdaptVis.
                if isinstance(labels_obj, torch.Tensor):
                    labels = labels_obj.detach().cpu().view(-1).tolist()
                elif isinstance(labels_obj, (list, tuple)) and labels_obj and isinstance(labels_obj[0], (list, tuple, torch.Tensor)):
                    first = labels_obj[0]
                    labels = first.detach().cpu().view(-1).tolist() if isinstance(first, torch.Tensor) else list(first)
                else:
                    labels = list(labels_obj) if isinstance(labels_obj, (list, tuple)) else []

                n = min(len(images), len(captions))
                for idx in range(n):
                    image = images[idx]
                    caption = captions[idx]
                    label = int(labels[idx]) if idx < len(labels) else 0

                    prompt = (
                        "Determine whether the description about the spatial relationship is correct or not. "
                        "Answer with yes or no: " + str(caption)
                    )

                    gen, conf, modified, selected_weight, branch = self._run_generation_by_method(
                        image=image,
                        prompt=prompt,
                        method=method,
                        weight=weight,
                        threshold=threshold,
                        weight1=weight1,
                        weight2=weight2,
                        force_relation_options=force_relation_options,
                        choices=relation_choices,
                    )

                    gen_l = gen.lower()
                    is_yes = ("yes" in gen_l) and ("no" not in gen_l[:10])
                    if label == 1:
                        TP += int(is_yes)
                        FN += int(not is_yes)
                        gold = "Yes"
                    else:
                        TN += int(not is_yes)
                        FP += int(is_yes)
                        gold = "No"

                    results.append({
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": gold,
                        "Confidence": float(conf),
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
