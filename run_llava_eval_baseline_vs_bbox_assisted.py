import os
import re
import argparse
from typing import Any

import torch
from PIL import Image

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int, help="-1 means run all samples.")
    p.add_argument("--max-new-tokens", default=24, type=int)
    return p.parse_args()


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", "" if x is None else str(x)).strip()


def clean_question(question: str) -> str:
    q = clean_text(question).replace("<image>", " ")

    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()

    q = re.sub(r"\s*(?:Please\s+)?(?:answer|respond).*$", "", q, flags=re.I)
    q = re.sub(r"\s*Your answer should be.*$", "", q, flags=re.I)
    q = re.sub(r"\s*Answer with.*$", "", q, flags=re.I)
    q = re.sub(r"\s*Choose from.*$", "", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip()

    if q and q[-1] not in "?.!":
        q += "?"

    return q


def normalize_relation_answer(text: str) -> str:
    t = clean_text(text).lower()

    if re.search(r"\bleft\b", t):
        return "left"
    if re.search(r"\bright\b", t):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b", t):
        return "under"
    if re.search(r"\bon\b|\bon top of\b|\batop\b", t):
        return "on"

    return t


def normalize_gold_relation(x: Any) -> str:
    t = clean_text(x).lower()
    for label in ["left", "right", "under", "on"]:
        if label in t:
            return label
    return t


def build_prompt(question: str) -> str:
    q = clean_question(question)
    return (
        "USER: <image>\n"
        f"{q}\n\n"
        "Choose the best spatial relation between the first object and the second object.\n"
        "Allowed labels: left, right, on, under.\n"
        "Return only the final label.\n"
        "ASSISTANT:"
    )


@torch.no_grad()
def run_llava_once(wrapper, image: Image.Image, prompt: str, max_new_tokens: int = 24) -> str:
    processor = wrapper.processor
    model = wrapper.model

    inputs = processor(
        images=image,
        text=prompt,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: (v.to(wrapper.device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
        if v is not None
    }

    n_img_tokens = int(
        (inputs["input_ids"] == model.config.image_token_index).sum().item()
    )

    if n_img_tokens != 1:
        decoded = processor.decode(
            inputs["input_ids"][0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        raise ValueError(
            f"Expected 1 image token, got {n_img_tokens}\n"
            f"Prompt:\n{prompt}\n\nDecoded:\n{decoded}"
        )

    prompt_len = int(inputs["input_ids"].shape[1])

    output = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        do_sample=False,
        use_cache=True,
    )

    gen_ids = output.sequences[0][prompt_len:]

    return processor.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def main():
    args = parse_args()
    seed_all(args.seed)

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(
        args.dataset,
        args.option,
    )

    if sampled_indices is not None:
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    total = 0
    correct = 0

    print(f"dataset = {args.dataset}")
    print(f"option = {args.option}")
    print(f"model = {args.model_name}")
    print(f"running samples [{start}, {end})")
    print("=" * 100)

    for local_idx in range(start, end):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image = item["image_options"][0]

        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image)}")

        raw_question = rec["question"]
        gold_relation = rec.get("answer", "")

        prompt = build_prompt(raw_question)
        raw_output = run_llava_once(
            wrapper=wrapper,
            image=image,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )

        pred_norm = normalize_relation_answer(raw_output)
        gold_norm = normalize_gold_relation(gold_relation)

        is_correct = pred_norm == gold_norm
        total += 1
        correct += int(is_correct)

        print("=" * 100)
        print(f"[{local_idx}] {image_name}")
        print(f"image_path: {image_path}")
        print(f"question: {clean_question(raw_question)}")
        print(f"gold_relation: {gold_relation}")
        print(f"gold_norm: {gold_norm}")
        print("[PROMPT]")
        print(prompt)
        print("[RAW OUTPUT]")
        print(raw_output)
        print(f"[NORMALIZED] {pred_norm}")
        print(f"[CORRECT] {is_correct}")

    acc = correct / total if total > 0 else 0.0
    print("=" * 100)
    print(f"TOTAL = {total}")
    print(f"CORRECT = {correct}")
    print(f"ACC = {acc:.4f}")


if __name__ == "__main__":
    main()
