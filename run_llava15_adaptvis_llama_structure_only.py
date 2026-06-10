#!/usr/bin/env python3
"""
Use AdaptVis repo's LLaMA STRUCTURE only, without using:
  - LlavaForConditionalGenerationScal
  - change_greedy_to_add_weight

What this script does:
  1) Load stock HF LlavaForConditionalGeneration for vision_tower/projector + checkpoint weights.
  2) Instantiate AdaptVis repo's LLaMAForCausalLMScal structure.
  3) Load stock HF language_model weights into that AdaptVis LLaMA structure.
  4) Replace model.language_model with the AdaptVis LLaMA structure.
  5) Manually perform LLaVA image-token merge to get true image_to_overwrite mask.
  6) Call the custom LLaMA directly with keys=image_to_overwrite and weight=...
  7) Use a manual KV-cache greedy loop instead of change_greedy_to_add_weight.

This tests: "Is the missing 0.86 behavior due to the AdaptVis LLaMA structure itself?"
"""

import argparse
import json
import os
import re
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration
from dataset_zoo.aro_datasets import Controlled_Images

# Use only the AdaptVis LLaMA structure, not the AdaptVis LLaVA wrapper/generation patch.
from model_zoo.llama.modeling_llama_add_attn import LLaMAConfig, LLaMAForCausalLMScal


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


def make_adaptvis_llama_config_from_hf(hf_text_cfg):
    """
    Build AdaptVis LLaMAConfig with the same architecture dimensions as the HF LLaMA config.
    This uses structure from AdaptVis, weights from the HF checkpoint.
    """
    names = [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "tie_word_embeddings",
    ]
    kwargs = {}
    for n in names:
        if hasattr(hf_text_cfg, n):
            kwargs[n] = getattr(hf_text_cfg, n)

    # AdaptVis LLaMAConfig usually accepts these old-LLaMA kwargs.
    cfg = LLaMAConfig(**kwargs)

    # Make sure critical attrs exist even if constructor ignored something.
    for n in names:
        if hasattr(hf_text_cfg, n):
            try:
                setattr(cfg, n, getattr(hf_text_cfg, n))
            except Exception:
                pass
    return cfg


def load_adaptvis_llama_structure_with_hf_weights(model, dtype=torch.float16):
    """
    Replace model.language_model:
      stock HF LlamaForCausalLM weights -> AdaptVis LLaMAForCausalLMScal structure.
    """
    old_lm = model.language_model
    hf_cfg = old_lm.config
    av_cfg = make_adaptvis_llama_config_from_hf(hf_cfg)

    print("Building AdaptVis LLaMAForCausalLMScal structure...")
    try:
        new_lm = LLaMAForCausalLMScal._from_config(av_cfg)
    except Exception:
        new_lm = LLaMAForCausalLMScal(av_cfg)

    new_lm = new_lm.to(dtype=dtype)

    print("Loading HF language_model weights into AdaptVis LLaMA structure...")
    sd = old_lm.state_dict()
    missing, unexpected = new_lm.load_state_dict(sd, strict=False)

    print("load_state_dict missing keys:", len(missing))
    print("load_state_dict unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("first missing:", missing[:20])
    if len(unexpected) > 0:
        print("first unexpected:", unexpected[:20])

    model.language_model = new_lm

    del old_lm, sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, missing, unexpected


@torch.no_grad()
def get_llava_image_features(model, pixel_values):
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
    strategy = getattr(cfg, "vision_feature_select_strategy", "default")

    selected = image_outputs.hidden_states[layer]
    if strategy == "default":
        selected = selected[:, 1:]
    elif strategy == "full":
        pass
    else:
        raise ValueError(f"Unknown vision_feature_select_strategy={strategy}")

    return model.multi_modal_projector(selected)


def merge_input_ids_with_image_features_core(
    image_features,
    inputs_embeds,
    input_ids,
    attention_mask,
    image_token_id,
    pad_token_id,
):
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    special_image_token_mask = input_ids == int(image_token_id)
    num_special_image_tokens = special_image_token_mask.sum(dim=-1)
    max_embed_dim = int((num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length)

    batch_indices, non_image_indices = torch.where(input_ids != int(image_token_id))

    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), dim=-1) - 1

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

    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(image_to_overwrite.device)

    expected = image_features.numel() // embed_dim
    got = int(image_to_overwrite.sum().item())
    if got != expected:
        raise ValueError(f"image_to_overwrite mismatch: got={got}, expected={expected}")

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(final_embedding.device)
    final_attention_mask = final_attention_mask | image_to_overwrite.to(final_attention_mask.dtype)

    # Native Scal computes position_ids but does not pass it to language_model.
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill(final_attention_mask == 0, 1)

    return final_embedding, final_attention_mask, position_ids, image_to_overwrite


@torch.no_grad()
def llava_first_step_forward(
    model,
    input_ids,
    attention_mask,
    pixel_values,
    image_token_id,
    pad_token_id,
    enable_intervention,
    weight,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = get_llava_image_features(model, pixel_values)

    merged_embeds, merged_attention_mask, _position_ids, image_mask = merge_input_ids_with_image_features_core(
        image_features=image_features,
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
    )

    lm_kwargs = dict(
        inputs_embeds=merged_embeds,
        attention_mask=merged_attention_mask,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    if enable_intervention:
        lm_kwargs["keys"] = image_mask
        lm_kwargs["weight"] = float(weight)
    else:
        # Probe path in main_aro has weight=1.0 but no keys; either way no intervention.
        lm_kwargs["weight"] = float(weight)

    outputs = model.language_model(**lm_kwargs)
    return outputs, merged_attention_mask, image_mask


@torch.no_grad()
def manual_greedy_with_custom_llama(
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
):
    generated = []
    score_list = []
    eos_id = processor.tokenizer.eos_token_id

    # Step 0: full prompt + image merge + optional AdaptVis.
    outputs, merged_attention_mask, image_mask = llava_first_step_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
        enable_intervention=enable_intervention,
        weight=weight,
    )

    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    score_list.append(logits.detach())
    generated.append(next_token)

    past_key_values = outputs.past_key_values
    cur_attention_mask = torch.cat(
        [
            merged_attention_mask,
            torch.ones((merged_attention_mask.shape[0], 1), dtype=merged_attention_mask.dtype, device=merged_attention_mask.device),
        ],
        dim=-1,
    )

    if eos_id is not None and int(next_token[0, 0].item()) == int(eos_id):
        gen_ids = torch.cat(generated, dim=-1)
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        conf = float(F.softmax(score_list[0][0].float(), dim=-1).max().item())
        return text, conf, {"image_mask_sum": [int(x) for x in image_mask.sum(dim=-1).detach().cpu().tolist()], "steps": 1}

    cur_input_ids = next_token

    # Later steps: cache path. Do not pass keys/weight, native condition would skip because q_len != kv_len anyway.
    for step in range(1, int(max_new_tokens)):
        outputs = model.language_model(
            input_ids=cur_input_ids,
            attention_mask=cur_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        score_list.append(logits.detach())
        generated.append(next_token)

        past_key_values = outputs.past_key_values
        cur_attention_mask = torch.cat(
            [
                cur_attention_mask,
                torch.ones((cur_attention_mask.shape[0], 1), dtype=cur_attention_mask.dtype, device=cur_attention_mask.device),
            ],
            dim=-1,
        )
        cur_input_ids = next_token

        if eos_id is not None and int(next_token[0, 0].item()) == int(eos_id):
            break

    gen_ids = torch.cat(generated, dim=-1) if generated else torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    first_conf = float(F.softmax(score_list[0][0].float(), dim=-1).max().item()) if score_list else 0.0
    trace = {
        "image_mask_sum": [int(x) for x in image_mask.sum(dim=-1).detach().cpu().tolist()],
        "steps": len(generated),
    }
    return text, first_conf, trace


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
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--weight1", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--round_confidence", action="store_true")
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
    model_kwargs = dict(cache_dir=none_if_needed(args.cache_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if revision is not None:
        proc_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)

    print("Loading stock HF LlavaForConditionalGeneration on CPU...")
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).eval()

    model, missing, unexpected = load_adaptvis_llama_structure_with_hf_weights(model, dtype=torch.float16)

    print("Moving full model to device...")
    model = model.to(device)
    model.requires_grad_(False)

    image_token_id = get_image_token_id(model, processor)
    pad_token_id = get_pad_token_id(model, processor)
    print("image_token_id:", image_token_id)
    print("pad_token_id:", pad_token_id)
    print("n_total:", n_total)

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_adaptvis_llama_structure_only_{model_tag}_{args.dataset}"
        f"_prompt{args.prompt_mode}_pad{args.pad_mode}_max{args.max_new_tokens}"
        f"_rev{args.revision}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}"
    )
    out_records = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    scores_arr = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    main_correct = 0
    strict_correct = 0
    unparsed = 0

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

        # Probe: no keys, no intervention. Use custom LLaMA structure.
        probe_text, probe_conf, probe_trace = manual_greedy_with_custom_llama(
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
        )

        conf_used = round(probe_conf, 2) if args.round_confidence else probe_conf
        chosen_weight = args.weight1 if conf_used < args.threshold else args.weight2

        final_text, final_conf, final_trace = manual_greedy_with_custom_llama(
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
            "raw_image_token_count": int((input_ids == image_token_id).sum().item()),
            "probe_trace": probe_trace,
            "final_trace": final_trace,
            "caption_options": list(caption_options),
        }
        records.append(rec)

        if args.print_every > 0 and i % args.print_every == 0:
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("input_len:", rec["input_len"], "raw_image_token_count:", rec["raw_image_token_count"])
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
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_first20": missing[:20],
        "unexpected_keys_first20": unexpected[:20],
        "out_records": str(out_records),
        "definition": "Stock HF LLaVA vision/projector + AdaptVis LLaMAForCausalLMScal structure loaded with HF language_model weights + manual merged image mask + manual greedy.",
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDirect acc mainaro-style:", main_acc)
    print("Direct acc strict-parse:", strict_acc)
    print("unparsed:", unparsed)
    print("missing keys:", len(missing), "unexpected keys:", len(unexpected))
    print("saved records:", out_records)
    print("saved summary:", out_summary)

    if n_total == len(dataset.dataset):
        print("\nRunning Controlled_Images evaluator...")
        dataset.evaluate_scores(
            scores=scores_arr,
            path="outputs",
            dataset=args.dataset,
            model=model_tag + "_adaptvis_llama_structure_only",
            method="adapt_vis",
            weight=1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
