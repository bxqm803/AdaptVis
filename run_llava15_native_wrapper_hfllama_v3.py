#!/usr/bin/env python3
"""
Diagnostic: use AdaptVis native LLaVA wrapper + native generation patch,
but replace its language_model with HF transformers LlamaForCausalLM.

Native:
  - LlavaForConditionalGenerationScal wrapper
  - its _merge_input_ids_with_image_features / prepare_inputs_for_generation
  - change_greedy_to_add_weight generation path
HF:
  - transformers.LlamaForCausalLM as language_model
  - patched to accept keys/weight and apply raw-logit AdaptVis
"""

import argparse, gc, json, math, os, re, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from dataset_zoo.aro_datasets import Controlled_Images

from model_zoo.llava15 import change_greedy_to_add_weight
from model_zoo.llava import LlavaForConditionalGenerationScal


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
    return None if x is None or str(x).lower() in {"", "none", "null"} else x


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


def decode_generated(processor, output, prompt_len):
    seq = output.sequences if hasattr(output, "sequences") else output["sequences"]
    return processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True).strip()


def first_step_confidence(output):
    scores = output.scores if hasattr(output, "scores") else output.get("scores", None)
    if scores is None or len(scores) == 0:
        return 0.0
    return float(torch.softmax(scores[0].detach().float(), dim=-1)[0].max().cpu())


def first_step_top5(output, processor):
    scores = output.scores if hasattr(output, "scores") else output.get("scores", None)
    if scores is None or len(scores) == 0:
        return []
    probs = F.softmax(scores[0][0].detach().float(), dim=-1)
    vals, ids = torch.topk(probs, k=5)
    out = []
    for v, tid in zip(vals.tolist(), ids.tolist()):
        try:
            tok = processor.tokenizer.decode([int(tid)])
        except Exception:
            tok = str(tid)
        out.append({"token_id": int(tid), "token": tok, "prob": float(v)})
    return out


# -----------------------------
# Patch HF LLaMA to accept AdaptVis keys/weight
# -----------------------------
def _apply_adaptvis_to_raw_logits(attn_weights, lm, layer_idx):
    if not getattr(lm, "_av_enable", False):
        return attn_weights
    if int(layer_idx) >= int(getattr(lm, "_av_max_layers", 32)):
        return attn_weights
    keys = getattr(lm, "_av_keys", None)
    weight = getattr(lm, "_av_weight", None)
    if keys is None or weight is None:
        return attn_weights
    weight = float(weight)
    if weight == 1.0:
        return attn_weights
    bsz, _, q_len, kv_len = attn_weights.shape
    if getattr(lm, "_av_require_square", True) and q_len != kv_len:
        return attn_weights
    if isinstance(keys, list):
        keys = torch.stack([k.to(attn_weights.device) for k in keys], dim=0)
    if keys.dim() == 1:
        keys = keys.unsqueeze(0)
    keys = keys.to(device=attn_weights.device)
    if keys.shape[-1] != kv_len:
        return attn_weights
    image_mask = keys.bool()
    out = attn_weights.clone()
    calls = pos_count = 0
    before_sum = after_sum = 0.0
    for b in range(min(bsz, image_mask.shape[0])):
        pos = torch.nonzero(image_mask[b], as_tuple=False).view(-1)
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
        lm._av_modified_calls = getattr(lm, "_av_modified_calls", 0) + calls
        lm._av_modified_pos_count = getattr(lm, "_av_modified_pos_count", 0) + pos_count
        lm._av_logit_sum_before = getattr(lm, "_av_logit_sum_before", 0.0) + before_sum
        lm._av_logit_sum_after = getattr(lm, "_av_logit_sum_after", 0.0) + after_sum
    return out


def patch_hf_llama_attention(lm, max_layers=32, require_square=True):
    lm._av_max_layers = int(max_layers)
    lm._av_require_square = bool(require_square)
    original_forward = lm.forward

    def lm_forward_with_adaptvis(
        input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, keys=None, weight=None, pos=None,
        caption_length=None, adjust_method=None, **kwargs,
    ):
        lm._av_enable = keys is not None and weight is not None
        lm._av_keys = keys
        lm._av_weight = weight
        lm._av_modified_calls = 0
        lm._av_modified_pos_count = 0
        lm._av_logit_sum_before = 0.0
        lm._av_logit_sum_after = 0.0
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
    lm.forward = lm_forward_with_adaptvis

    def make_attn_forward(attn_module, layer_idx):
        def forward(
            self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs,
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
            # Robust RoPE handling across transformers versions.
            # Important: do NOT pass position_ids as the positional seq_len argument
            # to old-style LlamaRotaryEmbedding. That can create cos/sin with the
            # wrong length and later trigger CUDA index out-of-bounds.
            if position_embeddings is not None:
                cos, sin = position_embeddings
                try:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, position_ids
                    )
                except TypeError:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin
                    )
            else:
                try:
                    # Old-style transformers: rotary_emb(x, seq_len=kv_seq_len)
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, position_ids
                    )
                except TypeError:
                    # New-style transformers: rotary_emb(x, position_ids)
                    cos, sin = self.rotary_emb(value_states, position_ids)
                    try:
                        query_states, key_states = apply_rotary_pos_emb(
                            query_states, key_states, cos, sin
                        )
                    except TypeError:
                        query_states, key_states = apply_rotary_pos_emb(
                            query_states, key_states, cos, sin, position_ids
                        )
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                try:
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                except Exception:
                    pass
            key_states = repeat_kv(key_states, num_kv_groups)
            value_states = repeat_kv(value_states, num_kv_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = _apply_adaptvis_to_raw_logits(attn_weights, lm, layer_idx)
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            # Same logical clamp as AdaptVis, but avoid constructing a CUDA scalar tensor.
            # The previous CUDA stacktrace often points here even when the real device assert
            # happened earlier asynchronously.
            attn_weights = torch.clamp(attn_weights, min=torch.finfo(attn_weights.dtype).min)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output, (attn_weights if output_attentions else None), past_key_value
        return types.MethodType(forward, attn_module)

    patched = 0
    for idx, layer in enumerate(lm.model.layers):
        layer.self_attn.forward = make_attn_forward(layer.self_attn, idx)
        patched += 1
    print("patched HF LLaMA attention layers:", patched)


def build_hf_lm_from_native_lm(native_lm, text_config, dtype=torch.float16):
    print("Building HF LlamaForCausalLM from text_config...")
    try:
        setattr(text_config, "_attn_implementation", "eager")
    except Exception:
        pass
    hf_lm = LlamaForCausalLM(text_config).to(dtype=dtype)
    sd = native_lm.state_dict()
    missing, unexpected = hf_lm.load_state_dict(sd, strict=False)
    print("HF LLaMA load_state_dict missing:", len(missing), "unexpected:", len(unexpected))
    if missing:
        print("first missing:", missing[:20])
    if unexpected:
        print("first unexpected:", unexpected[:20])
    return hf_lm, missing, unexpected


@torch.no_grad()
def run_generate(model, single_input, method, weight, max_new_tokens, keys=None):
    kwargs = dict(
        **single_input,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if method in {"scaling_vis", "adapt_vis"} and keys is not None:
        kwargs["keys"] = keys
        kwargs["weight"] = weight
    elif method in {"scaling_vis", "adapt_vis"} and weight is not None:
        kwargs["weight"] = weight
    return model.generate(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--revision", default="a272c74")
    ap.add_argument("--cache_dir", default="/ddnB/work/mwang32/hf_cache")
    ap.add_argument("--root_dir", default="data")
    ap.add_argument("--dataset", default="Controlled_Images_A")
    ap.add_argument("--subset", default="A")
    ap.add_argument("--option", default="four")
    ap.add_argument("--method", choices=["scaling_vis", "adapt_vis"], default="adapt_vis")
    ap.add_argument("--weight", type=float, default=0.5)
    ap.add_argument("--weight1", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--max_length", type=int, default=77)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--score_mode", choices=["mainaro", "predidx"], default="mainaro")
    ap.add_argument("--max_layers", type=int, default=32)
    args = ap.parse_args()

    setup_cache(args.cache_dir)
    Path("outputs").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = none_if_needed(args.revision)
    print("args:", vars(args))
    print("device:", device)

    dataset = Controlled_Images(image_preprocess=None, root_dir=args.root_dir, download=True, subset=args.subset)
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

    proc_kwargs = dict(cache_dir=none_if_needed(args.cache_dir))
    model_kwargs = dict(cache_dir=none_if_needed(args.cache_dir), ignore_mismatched_sizes=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if revision is not None:
        proc_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    print("Loading native AdaptVis LLaVA wrapper on CPU...")
    model = LlavaForConditionalGenerationScal.from_pretrained(args.model_id, **model_kwargs).eval()

    print("Replacing native AdaptVis LLaMA with HF LlamaForCausalLM...")
    text_config = model.config.text_config if hasattr(model.config, "text_config") else model.language_model.config
    hf_lm, missing, unexpected = build_hf_lm_from_native_lm(model.language_model, text_config, dtype=torch.float16)
    del model.language_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    patch_hf_llama_attention(hf_lm, max_layers=args.max_layers, require_square=True)
    model.language_model = hf_lm
    print("Moving model to device...")
    model = model.to(device)
    model.requires_grad_(False)

    print("Installing native generation patch...")
    change_greedy_to_add_weight()

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = f"llava15_nativewrapper_hfllama_{model_tag}_{args.dataset}_{args.method}_maxlen{args.max_length}_maxnew{args.max_new_tokens}_rev{args.revision}_w{args.weight}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}"
    out_records = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    correct = strict_correct = unparsed = 0

    for i in tqdm(range(n_total), total=n_total):
        # Useful with CUDA_LAUNCH_BLOCKING=1 if a CUDA assert occurs.
        if args.print_every > 0 and i % args.print_every == 0:
            print(f"[sample_start] i={i}", flush=True)
        d = dataset.dataset[i]
        image_path = resolve_image_path(d["image_path"])
        image = Image.open(image_path).convert("RGB")
        caption_options = d["caption_options"]
        prompt = prompts[i]
        gold = norm_gold(answers[i])
        single_input = processor(text=prompt, images=image, padding="max_length", return_tensors="pt", max_length=args.max_length).to(device)
        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input["input_ids"]]

        probe_gen = probe_unc = probe_top5 = None
        if args.method == "adapt_vis":
            probe_out = run_generate(model, single_input, args.method, 1.0, args.max_new_tokens, keys=None)
            probe_unc = round(first_step_confidence(probe_out), 2)
            probe_gen = decode_generated(processor, probe_out, len(single_input["input_ids"][-1]))
            probe_top5 = first_step_top5(probe_out, processor)
            chosen_weight = args.weight1 if probe_unc < args.threshold else args.weight2
            out = run_generate(model, single_input, args.method, chosen_weight, args.max_new_tokens, keys=keys)
        else:
            chosen_weight = args.weight
            out = run_generate(model, single_input, args.method, chosen_weight, args.max_new_tokens, keys=keys)

        gen = decode_generated(processor, out, len(single_input["input_ids"][-1]))
        main_ok = mainaro_is_correct(gold, gen)
        pred = parse_prep(gen)
        pred_idx, strict_ok = pred_to_option_index(pred, gold, caption_options)
        if pred is None:
            unparsed += 1
        correct += int(main_ok)
        strict_correct += int(strict_ok)
        if args.score_mode == "mainaro":
            scores[i, 0, :] = np.array([1, 0, 0, 0], dtype=np.float32) if main_ok else np.array([0, 0, 1, 0], dtype=np.float32)
        else:
            scores[i, 0, pred_idx] = 1.0
        av_trace = {
            "modified_calls": int(getattr(model.language_model, "_av_modified_calls", 0)),
            "modified_pos_count": int(getattr(model.language_model, "_av_modified_pos_count", 0)),
            "logit_sum_before": float(getattr(model.language_model, "_av_logit_sum_before", 0.0)),
            "logit_sum_after": float(getattr(model.language_model, "_av_logit_sum_after", 0.0)),
        }
        rec = {
            "index": i,
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "generation": gen,
            "mainaro_correct": bool(main_ok),
            "strict_correct": bool(strict_ok),
            "pred_prep": pred,
            "pred_idx": int(pred_idx),
            "chosen_weight": float(chosen_weight),
            "probe_generation": probe_gen,
            "probe_uncertainty_round": probe_unc,
            "probe_top5": probe_top5,
            "input_len": int(single_input["input_ids"].shape[-1]),
            "image_token_count": int((single_input["input_ids"] == 32001).sum().item()),
            "keys_sum": int(sum(k.sum().item() for k in keys)),
            "av_trace_last_forward": av_trace,
            "caption_options": list(caption_options),
        }
        records.append(rec)
        if args.print_every > 0 and i % args.print_every == 0:
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("input_len:", rec["input_len"], "image_token_count:", rec["image_token_count"], "keys_sum:", rec["keys_sum"])
            print("prompt:", prompt)
            print("gold:", gold)
            if args.method == "adapt_vis":
                print("probe_uncertainty_round:", probe_unc)
                print("probe_generation:", probe_gen)
            print("chosen_weight:", chosen_weight)
            print("generation:", gen)
            print("pred:", pred, "mainaro_correct:", main_ok, "strict_correct:", strict_ok)
            print("av_trace_last_forward:", av_trace)
            print("running mainaro acc:", correct / (i + 1), "strict acc:", strict_correct / (i + 1), "unparsed:", unparsed)
        if (i + 1) % 25 == 0:
            with open(out_records, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    with open(out_records, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    main_acc = correct / max(n_total, 1)
    strict_acc = strict_correct / max(n_total, 1)
    summary = {
        "args": vars(args),
        "n": n_total,
        "mainaro_style_direct_acc": main_acc,
        "strict_parse_direct_acc": strict_acc,
        "unparsed": unparsed,
        "hf_lm_missing_keys_count": len(missing),
        "hf_lm_unexpected_keys_count": len(unexpected),
        "hf_lm_missing_keys_first20": missing[:20],
        "hf_lm_unexpected_keys_first20": unexpected[:20],
        "out_records": str(out_records),
        "definition": "Native AdaptVis LLaVA wrapper + native generation patch + HF LlamaForCausalLM patched to accept keys/weight.",
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nDirect acc mainaro-style:", main_acc)
    print("Direct acc strict-parse:", strict_acc)
    print("unparsed:", unparsed)
    print("HF LLaMA missing keys:", len(missing), "unexpected keys:", len(unexpected))
    print("saved records:", out_records)
    print("saved summary:", out_summary)
    if n_total == len(dataset.dataset):
        print("\nRunning Controlled_Images evaluator...")
        dataset.evaluate_scores(scores=scores, path="outputs", dataset=args.dataset, model=model_tag + "_nativewrapper_hfllama", method=args.method, weight=args.weight if args.method == "scaling_vis" else 1.0, sampled_indices=[], option=args.option)
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
