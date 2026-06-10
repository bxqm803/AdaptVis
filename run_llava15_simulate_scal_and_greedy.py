#!/usr/bin/env python3
"""
Closer simulation of AdaptVis WITHOUT importing:
  - model_zoo.llava.LlavaForConditionalGenerationScal
  - model_zoo.llava15.change_greedy_to_add_weight

It simulates:
  1) change_greedy_to_add_weight -> explicit manual greedy loop.
  2) Scal attention -> patch each LLaMA attention forward and modify raw
     attention logits BEFORE adding attention_mask.
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


def set_sim_state(lm, enable, weight, input_ids, image_token_id, image_seq_len, image_pos_shift, require_square=True):
    lm._sim_enable = bool(enable)
    lm._sim_weight = float(weight)
    lm._sim_require_square = bool(require_square)
    lm._sim_modified_calls = 0
    lm._sim_modified_pos_count = 0
    lm._sim_logit_sum_before = 0.0
    lm._sim_logit_sum_after = 0.0

    positions = []
    for b in range(input_ids.shape[0]):
        raw_pos = torch.nonzero(input_ids[b] == int(image_token_id), as_tuple=False).view(-1)
        if raw_pos.numel() == 1:
            start = int(raw_pos.item()) + int(image_pos_shift)
            positions.append(torch.arange(start, start + int(image_seq_len), dtype=torch.long))
        else:
            positions.append(raw_pos.detach().cpu() + int(image_pos_shift))
    lm._sim_image_positions = positions
    lm._sim_last_positions = positions


def apply_sim_intervention(attn_weights, lm, layer_idx):
    if not getattr(lm, "_sim_enable", False):
        return attn_weights
    if layer_idx is not None and int(layer_idx) >= int(getattr(lm, "_sim_max_layers", 32)):
        return attn_weights
    positions_by_batch = getattr(lm, "_sim_image_positions", None)
    if not positions_by_batch:
        return attn_weights
    weight = float(getattr(lm, "_sim_weight", 1.0))
    if weight == 1.0:
        return attn_weights
    bsz, _, q_len, kv_len = attn_weights.shape
    if getattr(lm, "_sim_require_square", True) and q_len != kv_len:
        return attn_weights

    out = attn_weights.clone()
    calls = 0
    pos_count = 0
    before_sum = 0.0
    after_sum = 0.0
    for b in range(min(bsz, len(positions_by_batch))):
        pos = positions_by_batch[b]
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, dtype=torch.long, device=out.device)
        else:
            pos = pos.to(device=out.device, dtype=torch.long)
        pos = pos[(pos >= 0) & (pos < kv_len)]
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
        lm._sim_modified_calls = getattr(lm, "_sim_modified_calls", 0) + calls
        lm._sim_modified_pos_count = getattr(lm, "_sim_modified_pos_count", 0) + pos_count
        lm._sim_logit_sum_before = getattr(lm, "_sim_logit_sum_before", 0.0) + before_sum
        lm._sim_logit_sum_after = getattr(lm, "_sim_logit_sum_after", 0.0) + after_sum
    return out


def get_trace(lm):
    pos = getattr(lm, "_sim_last_positions", None)
    lens = None if pos is None else [int(p.numel()) for p in pos]
    return {
        "image_pos_lens": lens,
        "modified_calls": int(getattr(lm, "_sim_modified_calls", 0)),
        "modified_pos_count": int(getattr(lm, "_sim_modified_pos_count", 0)),
        "logit_sum_before": float(getattr(lm, "_sim_logit_sum_before", 0.0)),
        "logit_sum_after": float(getattr(lm, "_sim_logit_sum_after", 0.0)),
    }


def patch_llama_attention_raw_logits(language_model, max_layers=32):
    language_model._sim_max_layers = int(max_layers)
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        layers = language_model.model.layers
    elif hasattr(language_model, "layers"):
        layers = language_model.layers
    else:
        raise RuntimeError("Cannot find LLaMA layers under language_model.")

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

            # Critical placement: raw logits are modified BEFORE attention_mask.
            attn_weights = apply_sim_intervention(attn_weights, language_model, layer_idx)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output, (attn_weights if output_attentions else None), past_key_value

        return types.MethodType(forward, attn_module)

    patched = 0
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.forward = make_forward(layer.self_attn, idx)
            patched += 1
    print(f"patched raw-logit attention layers: {patched}")


@torch.no_grad()
def manual_greedy(model, processor, inputs, image_token_id, image_seq_len, max_new_tokens, weight,
                  enable_intervention, intervention_scope, image_pos_shift, require_square):
    lm = get_language_model(model)
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).clone()
    model_kwargs = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}

    generated = []
    score_list = []
    trace_list = []
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

        set_sim_state(lm, enable_now, weight, input_ids, image_token_id, image_seq_len, image_pos_shift, require_square)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True, **model_kwargs)
        logits = out.logits[:, -1, :]
        score_list.append(logits.detach())
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token)
        trace_list.append(get_trace(lm))

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        if eos_id is not None and int(next_token[0, 0].item()) == int(eos_id):
            break

    gen_ids = torch.cat(generated, dim=1) if generated else torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
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
    ap.add_argument("--pad_mode", choices=["none", "max77"], default="none")
    ap.add_argument("--max_length", type=int, default=77)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--weight1", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--round_confidence", action="store_true")
    ap.add_argument("--intervention_scope", choices=["first_step", "all_steps", "none"], default="all_steps")
    ap.add_argument("--image_pos_shift", type=int, default=0)
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
    model_kwargs = dict(cache_dir=none_if_needed(args.cache_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager")
    if revision is not None:
        proc_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).eval().to(device)
    model.requires_grad_(False)

    lm = get_language_model(model)
    patch_llama_attention_raw_logits(lm, max_layers=args.max_layers)

    image_token_id = get_image_token_id(model, processor)
    image_seq_len = get_image_seq_len(model)
    print("image_token_id:", image_token_id)
    print("image_seq_len:", image_seq_len)
    print("n_total:", n_total)

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_sim_rawattn_greedy_{model_tag}_{args.dataset}"
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
        inputs = build_inputs(processor, image, prompt, device, args.pad_mode, args.max_length)

        probe_text, probe_conf, probe_trace = manual_greedy(
            model, processor, inputs, image_token_id, image_seq_len, args.max_new_tokens,
            1.0, False, "none", args.image_pos_shift, require_square
        )
        conf_used = round(probe_conf, 2) if args.round_confidence else probe_conf
        chosen_weight = args.weight1 if conf_used < args.threshold else args.weight2

        final_text, final_conf, final_trace = manual_greedy(
            model, processor, inputs, image_token_id, image_seq_len, args.max_new_tokens,
            chosen_weight, True, args.intervention_scope, args.image_pos_shift, require_square
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
        "definition": "No native AdaptVis imports. Manual greedy + raw-attention-logit patch before attention_mask.",
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
            model=model_tag + "_sim_rawattn_greedy",
            method="adapt_vis",
            weight=1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
