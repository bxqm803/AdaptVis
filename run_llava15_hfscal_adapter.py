#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean runner for testing:

  original AdaptVis LlavaForConditionalGenerationScal wrapper
  + model_zoo/llama_hf_scal.py::LLaMAForCausalLMScal
  + original change_greedy_to_add_weight()

This script does NOT manually replace model.language_model.

Before running:
  cp /mnt/data/llama_hf_scal.py model_zoo/llama_hf_scal.py
  cp /mnt/data/modeling_llava_scal_use_hfscal.py model_zoo/llava/modeling_llava_scal.py

Expected startup log:
  language_model module: model_zoo.llama_hf_scal
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

from transformers import AutoProcessor
from dataset_zoo.aro_datasets import Controlled_Images
from model_zoo.llava15 import change_greedy_to_add_weight
from model_zoo.llava import LlavaForConditionalGenerationScal


def none_if_needed(x):
    if x is None:
        return None
    x = str(x)
    return None if x.lower() in {"", "none", "null"} else x


def setup_cache(cache_dir):
    cache_dir = none_if_needed(cache_dir)
    if cache_dir is None:
        return
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
    os.environ.setdefault("TORCH_HOME", "/ddnB/work/mwang32/torch_cache")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]:
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


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
        return str(x[0]).strip() if len(x) else ""
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

    for i, cap in enumerate(caption_options):
        if parse_prep(cap) == pred_prep:
            return i, pred_prep == gold

    for i, cap in enumerate(caption_options):
        if parse_prep(cap) != gold:
            return i, False
    return 0, False


def decode_generated(processor, output, prompt_len):
    seq = output.sequences if hasattr(output, "sequences") else output["sequences"]
    return processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True).strip()


def first_step_confidence(output):
    scores = output.scores if hasattr(output, "scores") else output.get("scores", None)
    if scores is None or len(scores) == 0:
        return 0.0
    return float(torch.softmax(scores[0].detach().float(), dim=-1)[0].max().cpu())


def first_step_topk(output, processor, k=5):
    scores = output.scores if hasattr(output, "scores") else output.get("scores", None)
    if scores is None or len(scores) == 0:
        return []
    probs = F.softmax(scores[0][0].detach().float(), dim=-1)
    vals, ids = torch.topk(probs, k=k)
    out = []
    for v, tid in zip(vals.tolist(), ids.tolist()):
        try:
            tok = processor.tokenizer.decode([int(tid)])
        except Exception:
            tok = str(tid)
        out.append({"token_id": int(tid), "token": tok, "prob": float(v)})
    return out


@torch.no_grad()
def run_generate(model, single_input, method, weight, max_new_tokens, keys=None):
    kwargs = dict(
        **single_input,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if method in {"scaling_vis", "adapt_vis"}:
        kwargs["weight"] = weight
        if keys is not None:
            kwargs["keys"] = keys
    return model.generate(**kwargs)


def load_prompts(dataset_name, option):
    prompt_file = Path(f"prompts/{dataset_name}_with_answer_{option}_options.jsonl")
    if not prompt_file.exists():
        raise FileNotFoundError(prompt_file)
    prompts, answers = [], []
    with prompt_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["question"])
            answers.append(row["answer"])
    return prompts, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--revision", default="a272c74")
    parser.add_argument("--cache_dir", default="/ddnB/work/mwang32/hf_cache")
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--subset", default="A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--method", choices=["scaling_vis", "adapt_vis"], default="adapt_vis")
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--weight1", type=float, default=0.5)
    parser.add_argument("--weight2", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--score_mode", choices=["mainaro", "predidx"], default="mainaro")
    args = parser.parse_args()

    setup_cache(args.cache_dir)
    Path("outputs").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = none_if_needed(args.revision)

    print("args:", vars(args), flush=True)
    print("device:", device, flush=True)

    print("Loading dataset...", flush=True)
    dataset = Controlled_Images(image_preprocess=None, root_dir=args.root_dir, download=True, subset=args.subset)
    prompts, answers = load_prompts(args.dataset, args.option)

    total_dataset_len = len(dataset.dataset)
    start = max(0, int(args.start_index))
    end = total_dataset_len if args.limit <= 0 else min(total_dataset_len, start + int(args.limit))
    run_indices = list(range(start, end))
    print("dataset_len:", total_dataset_len, "run_start:", start, "run_end:", end, "n_run:", len(run_indices), flush=True)

    proc_kwargs = {"cache_dir": none_if_needed(args.cache_dir)}
    model_kwargs = {
        "cache_dir": none_if_needed(args.cache_dir),
        "ignore_mismatched_sizes": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if revision is not None:
        proc_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)

    print("Loading LlavaForConditionalGenerationScal...", flush=True)
    model = LlavaForConditionalGenerationScal.from_pretrained(args.model_id, **model_kwargs).eval()

    lm_cls = type(model.language_model)
    print("language_model class:", lm_cls, flush=True)
    print("language_model module:", lm_cls.__module__, flush=True)
    if "llama_hf_scal" not in lm_cls.__module__:
        print("[WARN] language_model is not model_zoo.llama_hf_scal. Check that modeling_llava_scal.py imports llama_hf_scal.", flush=True)

    print("Moving model to device...", flush=True)
    model = model.to(device)
    model.requires_grad_(False)

    print("Installing AdaptVis greedy-search patch...", flush=True)
    change_greedy_to_add_weight()

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_hfscaladapter_{model_tag}_{args.dataset}_{args.method}"
        f"_start{start}_end{end}"
        f"_maxlen{args.max_length}_maxnew{args.max_new_tokens}"
        f"_rev{args.revision}_w{args.weight}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}"
    )
    out_records = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    scores = np.zeros((total_dataset_len, 1, 4), dtype=np.float32)
    records = []
    correct = 0
    strict_correct = 0
    unparsed = 0

    for local_step, i in enumerate(tqdm(run_indices, total=len(run_indices))):
        if args.print_every > 0 and local_step % args.print_every == 0:
            print(f"[sample_start] local_step={local_step} i={i}", flush=True)

        row = dataset.dataset[i]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        caption_options = row["caption_options"]
        prompt = prompts[i]
        gold = norm_gold(answers[i])

        single_input = processor(
            text=prompt,
            images=image,
            padding="max_length",
            return_tensors="pt",
            max_length=args.max_length,
        ).to(device)

        input_len = int(single_input["input_ids"].shape[-1])
        image_token_count = int((single_input["input_ids"] == 32001).sum().item())
        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input["input_ids"]]

        probe_gen = None
        probe_unc = None
        probe_top5 = None

        if args.method == "adapt_vis":
            probe_out = run_generate(model, single_input, args.method, 1.0, args.max_new_tokens, keys=None)
            probe_unc = round(first_step_confidence(probe_out), 2)
            probe_gen = decode_generated(processor, probe_out, input_len)
            probe_top5 = first_step_topk(probe_out, processor, k=5)
            chosen_weight = args.weight1 if probe_unc < args.threshold else args.weight2
            out = run_generate(model, single_input, args.method, chosen_weight, args.max_new_tokens, keys=keys)
        else:
            chosen_weight = args.weight
            out = run_generate(model, single_input, args.method, chosen_weight, args.max_new_tokens, keys=keys)

        gen = decode_generated(processor, out, input_len)
        pred = parse_prep(gen)
        main_ok = mainaro_is_correct(gold, gen)
        pred_idx, strict_ok = pred_to_option_index(pred, gold, caption_options)

        if pred is None:
            unparsed += 1
        correct += int(main_ok)
        strict_correct += int(strict_ok)

        if args.score_mode == "mainaro":
            scores[i, 0, :] = (
                np.array([1, 0, 0, 0], dtype=np.float32)
                if main_ok
                else np.array([0, 0, 1, 0], dtype=np.float32)
            )
        else:
            scores[i, 0, pred_idx] = 1.0

        rec = {
            "index": int(i),
            "local_step": int(local_step),
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
            "input_len": input_len,
            "image_token_count": image_token_count,
            "keys_sum": int(sum(k.sum().item() for k in keys)),
            "caption_options": list(caption_options),
        }
        records.append(rec)

        if args.print_every > 0 and local_step % args.print_every == 0:
            denom = local_step + 1
            print("\n" + "=" * 80, flush=True)
            print("idx:", i, flush=True)
            print("image:", image_path, flush=True)
            print("input_len:", input_len, "image_token_count:", image_token_count, "keys_sum:", rec["keys_sum"], flush=True)
            print("prompt:", prompt, flush=True)
            print("gold:", gold, flush=True)
            if args.method == "adapt_vis":
                print("probe_uncertainty_round:", probe_unc, flush=True)
                print("probe_generation:", repr(probe_gen), flush=True)
            print("chosen_weight:", chosen_weight, flush=True)
            print("generation:", repr(gen), flush=True)
            print("pred:", pred, "mainaro_correct:", main_ok, "strict_correct:", strict_ok, flush=True)
            print("running mainaro acc:", correct / denom, "strict acc:", strict_correct / denom, "unparsed:", unparsed, flush=True)

        if (local_step + 1) % 25 == 0:
            with out_records.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    with out_records.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_run = max(1, len(run_indices))
    main_acc = correct / n_run
    strict_acc = strict_correct / n_run

    summary = {
        "args": vars(args),
        "dataset_len": total_dataset_len,
        "start": start,
        "end": end,
        "n_run": len(run_indices),
        "mainaro_style_direct_acc": main_acc,
        "strict_parse_direct_acc": strict_acc,
        "unparsed": unparsed,
        "out_records": str(out_records),
        "language_model_class": str(lm_cls),
        "language_model_module": str(lm_cls.__module__),
        "definition": "AdaptVis LLaVA wrapper + model_zoo.llama_hf_scal.LLaMAForCausalLMScal adapter + native generation patch.",
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDirect acc mainaro-style:", main_acc, flush=True)
    print("Direct acc strict-parse:", strict_acc, flush=True)
    print("unparsed:", unparsed, flush=True)
    print("saved records:", out_records, flush=True)
    print("saved summary:", out_summary, flush=True)

    if start == 0 and end == total_dataset_len:
        print("\nRunning Controlled_Images evaluator...", flush=True)
        dataset.evaluate_scores(
            scores=scores,
            path="outputs",
            dataset=args.dataset,
            model=model_tag + "_hfscaladapter",
            method=args.method,
            weight=args.weight if args.method == "scaling_vis" else 1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --start_index/--limit was used.", flush=True)


if __name__ == "__main__":
    main()
