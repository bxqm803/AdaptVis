#!/usr/bin/env python3
"""
Clean runner for:
  original AdaptVis LlavaForConditionalGenerationScal wrapper
  + model_zoo/llama_hf_scal.py LLaMAForCausalLMScal adapter
  + original change_greedy_to_add_weight()

Important:
  This runner does NOT replace model.language_model after loading.
  Therefore it will only use llama_hf_scal if you copied:
      cp /mnt/data/llama_hf_scal.py model_zoo/llama_hf_scal.py
      cp /mnt/data/modeling_llava_scal_use_hfscal.py model_zoo/llava/modeling_llava_scal.py
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
    model_kwargs = dict(
        cache_dir=none_if_needed(args.cache_dir),
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if revision is not None:
        proc_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)

    print("Loading LlavaForConditionalGenerationScal...")
    model = LlavaForConditionalGenerationScal.from_pretrained(args.model_id, **model_kwargs).eval()
    print("language_model class:", type(model.language_model))
    print("language_model module:", type(model.language_model).__module__)

    print("Moving model to device...")
    model = model.to(device)
    model.requires_grad_(False)

    print("Installing native generation patch...")
    change_greedy_to_add_weight()

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_hfscaladapter_{model_tag}_{args.dataset}_{args.method}"
        f"_maxlen{args.max_length}_maxnew{args.max_new_tokens}_rev{args.revision}"
        f"_w{args.weight}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}"
    )
    out_records = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    correct = 0
    strict_correct = 0
    unparsed = 0

    for i in tqdm(range(n_total), total=n_total):
        if args.print_every > 0 and i % args.print_every == 0:
            print(f"[sample_start] i={i}", flush=True)

        d = dataset.dataset[i]
        image_path = resolve_image_path(d["image_path"])
        image = Image.open(image_path).convert("RGB")
        caption_options = d["caption_options"]
        prompt = prompts[i]
        gold = norm_gold(answers[i])

        single_input = processor(
            text=prompt,
            images=image,
            padding="max_length",
            return_tensors="pt",
            max_length=args.max_length,
        ).to(device)

        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input["input_ids"]]

        probe_gen = None
        probe_unc = None
        probe_top5 = None

        if args.method == "adapt_vis":
            probe_out = run_generate(
                model=model,
                single_input=single_input,
                method=args.method,
                weight=1.0,
                max_new_tokens=args.max_new_tokens,
                keys=None,
            )
            probe_unc = round(first_step_confidence(probe_out), 2)
            probe_gen = decode_generated(processor, probe_out, len(single_input["input_ids"][-1]))
            probe_top5 = first_step_top5(probe_out, processor)
            chosen_weight = args.weight1 if probe_unc < args.threshold else args.weight2

            out = run_generate(
                model=model,
                single_input=single_input,
                method=args.method,
                weight=chosen_weight,
                max_new_tokens=args.max_new_tokens,
                keys=keys,
            )
        else:
            chosen_weight = args.weight
            out = run_generate(
                model=model,
                single_input=single_input,
                method=args.method,
                weight=chosen_weight,
                max_new_tokens=args.max_new_tokens,
                keys=keys,
            )

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
        "out_records": str(out_records),
        "language_model_class": str(type(model.language_model)),
        "language_model_module": str(type(model.language_model).__module__),
        "definition": "Original AdaptVis LLaVA wrapper + llama_hf_scal.LLaMAForCausalLMScal adapter + native generation patch.",
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
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
