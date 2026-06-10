#!/usr/bin/env python3
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

# Native AdaptVis / main_aro path
from model_zoo.llava15 import change_greedy_to_add_weight
from model_zoo.llava import LlavaForConditionalGeneration, LlavaForConditionalGenerationScal


def setup_cache(cache_dir):
    if cache_dir:
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
    if hasattr(output, "sequences"):
        seq = output.sequences
    elif isinstance(output, dict):
        seq = output.get("sequences", None)
    else:
        seq = output[0]
    return processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True).strip()


def first_step_confidence(output):
    scores = None
    if hasattr(output, "scores"):
        scores = output.scores
    elif isinstance(output, dict):
        scores = output.get("scores", None)
    if scores is None or len(scores) == 0:
        return 0.0
    return float(torch.softmax(scores[0].detach().float(), dim=-1)[0].max().cpu())


def first_step_top5(output, processor):
    scores = None
    if hasattr(output, "scores"):
        scores = output.scores
    elif isinstance(output, dict):
        scores = output.get("scores", None)
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


def load_model(args, device):
    model_cls = LlavaForConditionalGenerationScal if args.method in {"scaling_vis", "adapt_vis"} or args.force_scal else LlavaForConditionalGeneration
    model = model_cls.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=True,
    ).eval().to(device)
    return model


@torch.no_grad()
def run_generate(model, processor, single_input, method, weight, max_new_tokens, keys=None):
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
        # main_aro probe path for adapt_vis passes weight=1.0 without keys.
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
    ap.add_argument("--method", choices=["base", "scaling_vis", "adapt_vis"], default="adapt_vis")

    ap.add_argument("--weight", type=float, default=0.5)
    ap.add_argument("--weight1", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.4)

    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--max_length", type=int, default=77)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--force_scal", action="store_true")
    ap.add_argument("--score_mode", choices=["mainaro", "predidx"], default="mainaro")
    args = ap.parse_args()

    setup_cache(args.cache_dir)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    dataset = Controlled_Images(
        image_preprocess=None,
        root_dir=args.root_dir,
        download=True,
        subset=args.subset,
    )

    prompt_file = f"prompts/{args.dataset}_with_answer_{args.option}_options.jsonl"
    prompts, answers = [], []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # Important: do NOT clean prompt. main_aro uses this string directly.
            prompts.append(r["question"])
            answers.append(r["answer"])

    n_total = len(dataset.dataset)
    if args.limit > 0:
        n_total = min(n_total, args.limit)

    print("dataset total:", len(dataset.dataset))
    print("running n:", n_total)
    print("device:", device)
    print("model_id:", args.model_id)
    print("revision:", args.revision)
    print("method:", args.method)
    print("max_length:", args.max_length)
    print("max_new_tokens:", args.max_new_tokens)
    print("threshold:", args.threshold)
    print("weight1:", args.weight1, "weight2:", args.weight2)
    print("score_mode:", args.score_mode)

    print("Loading native processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    print("Loading native model...")
    model = load_model(args, device=device)
    model.requires_grad_(False)

    if args.method in {"scaling_vis", "adapt_vis"}:
        change_greedy_to_add_weight()

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = (
        f"llava15_native_scal_bridge_{model_tag}_{args.dataset}_{args.method}"
        f"_maxlen{args.max_length}_maxnew{args.max_new_tokens}_rev{args.revision}"
        f"_w{args.weight}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}_score{args.score_mode}"
    )
    out_records = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    correct = 0
    strict_correct = 0
    unparsed = 0

    for i in tqdm(range(n_total), total=n_total):
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

        # Native main_aro keys: raw <image> token id == 32001.
        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input["input_ids"]]

        probe_gen = None
        probe_unc = None
        probe_top5 = None

        if args.method == "adapt_vis":
            probe_out = run_generate(
                model=model,
                processor=processor,
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
                processor=processor,
                single_input=single_input,
                method=args.method,
                weight=chosen_weight,
                max_new_tokens=args.max_new_tokens,
                keys=keys,
            )
        elif args.method == "scaling_vis":
            chosen_weight = args.weight
            out = run_generate(
                model=model,
                processor=processor,
                single_input=single_input,
                method=args.method,
                weight=chosen_weight,
                max_new_tokens=args.max_new_tokens,
                keys=keys,
            )
        else:
            chosen_weight = 1.0
            out = model.generate(
                **single_input,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
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
            if len(list(caption_options)) == 4:
                scores[i, 0, :] = np.array([1, 0, 0, 0], dtype=np.float32) if main_ok else np.array([0, 0, 1, 0], dtype=np.float32)
            else:
                # Kept for compatibility; Controlled A uses four options.
                scores[i, 0, :2] = np.array([1, 0], dtype=np.float32) if main_ok else np.array([0, 1], dtype=np.float32)
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
            "final_first_token_confidence": first_step_confidence(out),
            "final_top5": first_step_top5(out, processor),
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
        "model_id": args.model_id,
        "revision": args.revision,
        "dataset": args.dataset,
        "method": args.method,
        "n": n_total,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "weight": args.weight,
        "weight1": args.weight1,
        "weight2": args.weight2,
        "threshold": args.threshold,
        "score_mode": args.score_mode,
        "mainaro_style_direct_acc": main_acc,
        "strict_parse_direct_acc": strict_acc,
        "unparsed": unparsed,
        "out_records": str(out_records),
        "definition": "Native AdaptVis bridge: repo LlavaForConditionalGenerationScal + change_greedy_to_add_weight + processor padding=max_length,max_length=77 + keys=(input_ids==32001) + max_new_tokens=100.",
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
            model=model_tag + "_native_scal_bridge",
            method=args.method,
            weight=args.weight if args.method == "scaling_vis" else 1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
