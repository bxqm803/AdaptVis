#!/usr/bin/env python3
"""
Simulate the two native AdaptVis modules without importing:
  - model_zoo.llava.LlavaForConditionalGenerationScal
  - model_zoo.llava15.change_greedy_to_add_weight

What is simulated:
  1) change_greedy_to_add_weight()
     -> replaced by an explicit manual greedy loop, where each model forward is called
        directly and the AdaptVis state is set before the forward.

  2) LlavaForConditionalGenerationScal
     -> approximated on stock HF LlavaForConditionalGeneration by monkey-patching
        the language model attention softmax and scaling pre-softmax image logits.

This is a diagnostic script. If it does NOT reach native main_aro accuracy, that means
the remaining gap is inside the native Scal implementation, especially exact merged
image-mask propagation / custom LLaMA attention behavior.
"""

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

from transformers import AutoProcessor, LlavaForConditionalGeneration
from dataset_zoo.aro_datasets import Controlled_Images


# ============================================================
# Utilities
# ============================================================
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
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    return model


def get_image_token_id(model, processor):
    if hasattr(model.config, "image_token_index"):
        return int(model.config.image_token_index)
    return int(processor.tokenizer.convert_tokens_to_ids("<image>"))


def get_image_seq_len(model):
    if hasattr(model.config, "image_seq_length"):
        return int(model.config.image_seq_length)
    vc = getattr(model.config, "vision_config", None)
    if vc is not None and hasattr(vc, "image_size") and hasattr(vc, "patch_size"):
        return int((int(vc.image_size) // int(vc.patch_size)) ** 2)
    return 576


# ============================================================
# Simulated Scal attention intervention on stock HF LLaVA
# ============================================================
_ORIG_SOFTMAX = None
_PATCH_INSTALLED = False


def _scale_attention_logits(attn_logits, lm):
    if attn_logits.dim() != 4:
        return attn_logits
    if not getattr(lm, "_sim_enable", False):
        return attn_logits

    positions_by_batch = getattr(lm, "_sim_image_positions", None)
    if not positions_by_batch:
        return attn_logits

    weight = float(getattr(lm, "_sim_weight", 1.0))
    if weight == 1.0:
        return attn_logits

    bsz, _, q_len, kv_len = attn_logits.shape
    if getattr(lm, "_sim_require_square", True) and q_len != kv_len:
        return attn_logits

    out = attn_logits.clone()

    local_calls = 0
    local_pos = 0
    local_before = 0.0
    local_after = 0.0

    for b in range(min(bsz, len(positions_by_batch))):
        pos = positions_by_batch[b]
        if pos is None:
            continue
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, device=out.device, dtype=torch.long)
        else:
            pos = pos.to(device=out.device, dtype=torch.long)
        pos = pos[(pos >= 0) & (pos < kv_len)]
        if pos.numel() == 0:
            continue

        before = out[b, :, -1:, pos].detach().float().sum().item()
        out[b, :, -1:, pos] *= weight
        after = out[b, :, -1:, pos].detach().float().sum().item()

        local_calls += 1
        local_pos += int(pos.numel())
        local_before += before
        local_after += after

    if local_calls:
        lm._sim_modified_calls = getattr(lm, "_sim_modified_calls", 0) + local_calls
        lm._sim_modified_pos_count = getattr(lm, "_sim_modified_pos_count", 0) + local_pos
        lm._sim_logit_sum_before = getattr(lm, "_sim_logit_sum_before", 0.0) + local_before
        lm._sim_logit_sum_after = getattr(lm, "_sim_logit_sum_after", 0.0) + local_after

    return out


def install_softmax_patch(language_model):
    global _ORIG_SOFTMAX, _PATCH_INSTALLED

    if not getattr(language_model, "_sim_forward_patched", False):
        old_forward = language_model.forward

        def patched_forward(*args, **kwargs):
            language_model._sim_in_lm_forward = True
            try:
                return old_forward(*args, **kwargs)
            finally:
                language_model._sim_in_lm_forward = False

        language_model.forward = patched_forward
        language_model._sim_forward_patched = True
        language_model._sim_in_lm_forward = False

    if not _PATCH_INSTALLED:
        _ORIG_SOFTMAX = F.softmax

        def patched_softmax(input, dim=None, _stacklevel=3, dtype=None):
            lm = getattr(patched_softmax, "language_model", None)
            if (
                lm is not None
                and getattr(lm, "_sim_in_lm_forward", False)
                and getattr(lm, "_sim_enable", False)
                and input.dim() == 4
                and dim in (-1, input.dim() - 1)
            ):
                input = _scale_attention_logits(input, lm)

            if dtype is None:
                try:
                    return _ORIG_SOFTMAX(input, dim=dim, _stacklevel=_stacklevel)
                except TypeError:
                    return _ORIG_SOFTMAX(input, dim=dim)
            try:
                return _ORIG_SOFTMAX(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
            except TypeError:
                return _ORIG_SOFTMAX(input, dim=dim, dtype=dtype)

        F.softmax = patched_softmax
        _PATCH_INSTALLED = True

    F.softmax.language_model = language_model


def set_sim_state(language_model, enable, weight, input_ids, image_token_id, image_seq_len, image_pos_shift, require_square=True):
    language_model._sim_enable = bool(enable)
    language_model._sim_weight = float(weight)
    language_model._sim_require_square = bool(require_square)

    language_model._sim_modified_calls = 0
    language_model._sim_modified_pos_count = 0
    language_model._sim_logit_sum_before = 0.0
    language_model._sim_logit_sum_after = 0.0

    positions = []
    for b in range(input_ids.shape[0]):
        pos = torch.nonzero(input_ids[b] == int(image_token_id), as_tuple=False).view(-1)
        if pos.numel() == 1:
            start = int(pos.item()) + int(image_pos_shift)
            positions.append(torch.arange(start, start + int(image_seq_len), dtype=torch.long))
        else:
            positions.append(pos.detach().cpu() + int(image_pos_shift))
    language_model._sim_image_positions = positions
    language_model._sim_last_positions = positions


def get_trace(language_model):
    pos = getattr(language_model, "_sim_last_positions", None)
    if pos is None:
        lens = None
    else:
        lens = [int(p.numel()) for p in pos]
    return {
        "image_pos_lens": lens,
        "modified_calls": int(getattr(language_model, "_sim_modified_calls", 0)),
        "modified_pos_count": int(getattr(language_model, "_sim_modified_pos_count", 0)),
        "logit_sum_before": float(getattr(language_model, "_sim_logit_sum_before", 0.0)),
        "logit_sum_after": float(getattr(language_model, "_sim_logit_sum_after", 0.0)),
    }


# ============================================================
# Manual greedy: simulates change_greedy_to_add_weight()
# ============================================================
@torch.no_grad()
def manual_greedy(
    model,
    processor,
    inputs,
    image_token_id,
    image_seq_len,
    max_new_tokens,
    weight,
    enable_intervention,
    intervention_scope,
    image_pos_shift,
    require_square,
):
    lm = get_language_model(model)

    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).clone()

    model_kwargs = {}
    for k, v in inputs.items():
        if k not in {"input_ids", "attention_mask"}:
            model_kwargs[k] = v

    generated_ids = []
    scores = []
    traces = []

    eos_id = processor.tokenizer.eos_token_id

    for step in range(int(max_new_tokens)):
        if intervention_scope == "all_steps":
            enable_now = enable_intervention
        elif intervention_scope == "first_step":
            enable_now = enable_intervention and step == 0
        elif intervention_scope == "none":
            enable_now = False
        else:
            raise ValueError(intervention_scope)

        set_sim_state(
            lm,
            enable=enable_now,
            weight=weight,
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_seq_len=image_seq_len,
            image_pos_shift=image_pos_shift,
            require_square=require_square,
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **model_kwargs,
        )

        logits = out.logits[:, -1, :]
        scores.append(logits.detach())
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated_ids.append(next_token)
        traces.append(get_trace(lm))

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        if eos_id is not None and int(next_token[0, 0].item()) == int(eos_id):
            break

    if generated_ids:
        gen_ids = torch.cat(generated_ids, dim=1)
    else:
        gen_ids = torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)

    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    first_conf = float(F.softmax(scores[0][0].float(), dim=-1).max().item()) if scores else 0.0

    total_trace = {
        "steps": len(traces),
        "modified_calls": sum(t["modified_calls"] for t in traces),
        "modified_pos_count": sum(t["modified_pos_count"] for t in traces),
        "logit_sum_before": sum(t["logit_sum_before"] for t in traces),
        "logit_sum_after": sum(t["logit_sum_after"] for t in traces),
        "first_step_trace": traces[0] if traces else None,
        "last_step_trace": traces[-1] if traces else None,
    }

    return text, first_conf, scores, total_trace


def build_inputs(processor, image, prompt, device, pad_mode, max_length):
    if pad_mode == "none":
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    elif pad_mode == "max77":
        inputs = processor(
            text=prompt,
            images=image,
            padding="max_length",
            max_length=int(max_length),
            return_tensors="pt",
        )
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
    ap.add_argument("--pad_mode", choices=["none", "max77"], default="none")
    ap.add_argument("--max_length", type=int, default=77)
    ap.add_argument("--max_new_tokens", type=int, default=16)

    ap.add_argument("--weight1", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--round_confidence", action="store_true")

    ap.add_argument("--intervention_scope", choices=["first_step", "all_steps", "none"], default="first_step")
    ap.add_argument("--image_pos_shift", type=int, default=0)
    ap.add_argument("--no_square_required", action="store_true")

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

    dataset = Controlled_Images(
        image_preprocess=None,
        root_dir=args.root_dir,
        download=True,
        subset=args.subset,
    )

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
    install_softmax_patch(lm)

    image_token_id = get_image_token_id(model, processor)
    image_seq_len = get_image_seq_len(model)
    print("image_token_id:", image_token_id)
    print("image_seq_len:", image_seq_len)
    print("n_total:", n_total)

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_sim_scal_greedy_{model_tag}_{args.dataset}"
        f"_prompt{args.prompt_mode}_pad{args.pad_mode}_scope{args.intervention_scope}"
        f"_max{args.max_new_tokens}_shift{args.image_pos_shift}_rev{args.revision}"
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

        inputs = build_inputs(
            processor=processor,
            image=image,
            prompt=prompt,
            device=device,
            pad_mode=args.pad_mode,
            max_length=args.max_length,
        )

        # Probe: no intervention, get first-token confidence.
        probe_text, probe_conf, _, probe_trace = manual_greedy(
            model=model,
            processor=processor,
            inputs=inputs,
            image_token_id=image_token_id,
            image_seq_len=image_seq_len,
            max_new_tokens=args.max_new_tokens,
            weight=1.0,
            enable_intervention=False,
            intervention_scope="none",
            image_pos_shift=args.image_pos_shift,
            require_square=require_square,
        )
        conf_used = round(probe_conf, 2) if args.round_confidence else probe_conf
        chosen_weight = args.weight1 if conf_used < args.threshold else args.weight2

        final_text, final_conf, _, final_trace = manual_greedy(
            model=model,
            processor=processor,
            inputs=inputs,
            image_token_id=image_token_id,
            image_seq_len=image_seq_len,
            max_new_tokens=args.max_new_tokens,
            weight=chosen_weight,
            enable_intervention=True,
            intervention_scope=args.intervention_scope,
            image_pos_shift=args.image_pos_shift,
            require_square=require_square,
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
            "input_len": int(inputs["input_ids"].shape[-1]),
            "image_token_count": int((inputs["input_ids"] == image_token_id).sum().item()),
            "probe_trace": probe_trace,
            "final_trace": final_trace,
            "caption_options": list(caption_options),
        }
        records.append(rec)

        if args.print_every > 0 and i % args.print_every == 0:
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("input_len:", rec["input_len"], "image_token_count:", rec["image_token_count"])
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
        "definition": "No native AdaptVis imports. Simulates change_greedy via manual greedy loop and simulates Scal via stock HF LLaVA pre-softmax attention-logit patch.",
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
            model=model_tag + "_sim_scal_greedy",
            method="adapt_vis",
            weight=1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
