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

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from dataset_zoo.aro_datasets import Controlled_Images

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def setup_cache():
    os.environ.setdefault("HF_HOME", "/ddnB/work/mwang32/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/ddnB/work/mwang32/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/ddnB/work/mwang32/hf_cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/ddnB/work/mwang32/hf_cache/datasets")
    os.environ.setdefault("TORCH_HOME", "/ddnB/work/mwang32/torch_cache")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]:
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def clean_prompt_for_vlm(prompt):
    prompt = str(prompt)
    prompt = prompt.replace("<image>", "")
    prompt = prompt.replace("USER:", "").replace("User:", "").replace("user:", "")
    prompt = prompt.replace("ASSISTANT:", "").replace("Assistant:", "").replace("assistant:", "")
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


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


def resolve_image_path(p):
    p = str(p)
    if os.path.exists(p):
        return p
    base = os.path.basename(p)
    candidates = [os.path.join("data", "controlled_images", base), os.path.join("data", base)]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = list(Path("data").rglob(base))
    if hits:
        return str(hits[0])
    raise FileNotFoundError(p)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12, use_thumbnail=True):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


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


@torch.no_grad()
def chat_with_scores(model, tokenizer, pixel_values, question, generation_config, capture_scores=True):
    captured = {}
    old_generate = model.generate
    if capture_scores:
        def wrapped_generate(*args, **kwargs):
            kwargs.pop("output_scores", None)
            kwargs.pop("return_dict_in_generate", None)
            out = old_generate(*args, output_scores=True, return_dict_in_generate=True, **kwargs)
            captured["output"] = out
            return out.sequences
        model.generate = wrapped_generate
    try:
        response = model.chat(tokenizer, pixel_values, question, generation_config)
    finally:
        if capture_scores:
            model.generate = old_generate
    score_info = {
        "first_token_max_prob": None,
        "first_token_max_prob_round": None,
        "first_token_argmax_id": None,
        "first_token_argmax_text": None,
        "first_token_top5": None,
    }
    if capture_scores and "output" in captured:
        out = captured["output"]
        if hasattr(out, "scores") and out.scores is not None and len(out.scores) > 0:
            logits0 = out.scores[0][0].float()
            probs0 = F.softmax(logits0, dim=-1)
            max_prob, max_id = torch.max(probs0, dim=-1)
            top_probs, top_ids = torch.topk(probs0, k=min(5, probs0.numel()))
            top5 = []
            for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
                try:
                    tok = tokenizer.decode([int(tid)])
                except Exception:
                    tok = str(tid)
                top5.append({"token_id": int(tid), "token": tok, "prob": float(p)})
            max_id_int = int(max_id.item())
            try:
                max_text = tokenizer.decode([max_id_int])
            except Exception:
                max_text = str(max_id_int)
            score_info = {
                "first_token_max_prob": float(max_prob.item()),
                "first_token_max_prob_round": round(float(max_prob.item()), 2),
                "first_token_argmax_id": max_id_int,
                "first_token_argmax_text": max_text,
                "first_token_top5": top5,
            }
    return str(response).strip(), score_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="OpenGVLab/InternVL2_5-2B")
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--subset", default="A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--max_num", type=int, default=1)
    parser.add_argument("--use_thumbnail", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()

    setup_cache()
    Path("outputs").mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
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

    print("dataset total:", len(dataset.dataset))
    print("running n:", n_total)
    print("model_id:", args.model_id)
    print("input_size:", args.input_size, "max_num:", args.max_num, "use_thumbnail:", args.use_thumbnail)

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)

    print("Loading model...")
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
    )
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    model = AutoModel.from_pretrained(args.model_id, **model_kwargs).eval()
    if torch.cuda.is_available() and not args.load_in_8bit:
        model = model.cuda()
    model.requires_grad_(False)

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    correct = 0
    unparsed = 0

    model_tag = args.model_id.replace("/", "_").replace("-", "_")
    out_tag = f"internvl25_{model_tag}_{args.dataset}_base_maxnum{args.max_num}_thumb{int(args.use_thumbnail)}"
    out_json = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    for i in tqdm(range(n_total), total=n_total):
        d = dataset.dataset[i]
        image_path = resolve_image_path(d["image_path"])
        caption_options = d["caption_options"]
        prompt = clean_prompt_for_vlm(prompts[i])
        gold = answers[i]
        question = "<image>\n" + prompt

        pixel_values = load_image(image_path, input_size=args.input_size, max_num=args.max_num, use_thumbnail=args.use_thumbnail).to(torch_dtype)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        response, score_info = chat_with_scores(model, tokenizer, pixel_values, question, generation_config, capture_scores=True)
        pred_prep = parse_prep(response)
        pred_idx, is_correct_bool = pred_to_option_index(pred_prep, gold, caption_options)
        if pred_prep is None:
            unparsed += 1
        scores[i, 0, pred_idx] = 1.0
        is_correct = int(is_correct_bool)
        correct += is_correct

        rec = {
            "index": i,
            "image_path": image_path,
            "prompt": prompt,
            "question": question,
            "gold": gold,
            "generation": response,
            "pred_prep": pred_prep,
            "pred_idx": int(pred_idx),
            "correct": bool(is_correct),
            "first_token_max_prob": score_info["first_token_max_prob"],
            "first_token_max_prob_round": score_info["first_token_max_prob_round"],
            "first_token_argmax_id": score_info["first_token_argmax_id"],
            "first_token_argmax_text": score_info["first_token_argmax_text"],
            "first_token_top5": score_info["first_token_top5"],
            "num_image_tiles": int(pixel_values.shape[0]),
            "caption_options": caption_options,
        }
        records.append(rec)

        if args.print_every > 0 and (i % args.print_every == 0):
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("tiles:", int(pixel_values.shape[0]))
            print("gold:", gold)
            print("generation:", response)
            print("pred:", pred_prep, "pred_idx:", pred_idx, "correct:", bool(is_correct))
            print("first_token_max_prob:", score_info["first_token_max_prob"])
            print("first_token_argmax_text:", score_info["first_token_argmax_text"])
            print("running acc:", correct / (i + 1), "unparsed:", unparsed)

        if (i + 1) % 25 == 0:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    direct_acc = correct / max(n_total, 1)
    summary = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "n": n_total,
        "direct_acc": direct_acc,
        "unparsed": unparsed,
        "input_size": args.input_size,
        "max_num": args.max_num,
        "use_thumbnail": args.use_thumbnail,
        "max_new_tokens": args.max_new_tokens,
        "out_json": str(out_json),
        "prob_definition": "first generated token full-vocab max softmax probability from model.chat/model.generate output_scores",
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDirect acc:", direct_acc)
    print("unparsed:", unparsed)
    print("saved records:", out_json)
    print("saved summary:", out_summary)

    if n_total == len(dataset.dataset):
        print("\nRunning Controlled_Images evaluator...")
        dataset.evaluate_scores(
            scores=scores,
            path="outputs",
            dataset=args.dataset,
            model=model_tag,
            method="base",
            weight=1.0,
            sampled_indices=[],
            option=args.option,
        )
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
