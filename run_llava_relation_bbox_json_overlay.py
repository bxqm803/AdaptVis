import os
import re
import csv
import json
import argparse
from typing import Any, Optional, Tuple, List, Dict

import torch
from PIL import Image

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="output_llava_relation_bbox_raw", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=10, type=int)
    p.add_argument("--max-new-tokens", default=192, type=int)
    p.add_argument("--print-prompts", action="store_true")
    return p.parse_args()


def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()


def clean_question_text(question: str) -> str:
    q = clean_text(question)
    q = q.replace("<image>", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    return re.sub(r"\s+", " ", q).strip()


def strip_answer_order_clause(q: str) -> str:
    q = clean_question_text(q)
    q = re.sub(r"\s*(?:Please\s+)?(?:answer|respond).*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Your answer should be.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Answer with.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Choose from.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    if q and q[-1] not in "?.!":
        q += "?"
    return q


def normalize_object_name(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"^(a|an|the)\s+", "", name)
    name = re.sub(r"[?.!,;:]+$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def extract_objects_from_question(question: str) -> Tuple[Optional[str], Optional[str], str]:
    q = strip_answer_order_clause(question)
    q_lower = q.lower()

    patterns = [
        r"where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+in\s+relation\s+to\s+(?:the\s+)?(.+?)\?",
        r"where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+relative\s+to\s+(?:the\s+)?(.+?)\?",
        r"where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+with\s+respect\s+to\s+(?:the\s+)?(.+?)\?",
        r"what\s+is\s+the\s+position\s+of\s+(?:the\s+)?(.+?)\s+relative\s+to\s+(?:the\s+)?(.+?)\?",
    ]

    for pat in patterns:
        m = re.search(pat, q_lower, flags=re.IGNORECASE)
        if m:
            obj1 = normalize_object_name(m.group(1))
            obj2 = normalize_object_name(m.group(2))
            return obj1 or None, obj2 or None, q

    return None, None, q


def build_prompt(obj1: str, obj2: str, original_question: str) -> str:
    original_q = strip_answer_order_clause(original_question)
    relation_q = f"Where is the {obj1} in relation to the {obj2}?"

    prompt = (
        "USER: <image>\n"
        f"Question: {relation_q}\n"
        f"Original dataset question: {original_q}\n"
        "Please locate both queried objects in the image.\n"
        "Answer with bounding boxes for both objects.\n"
        "Use normalized coordinates from 0 to 1000 for the full image.\n"
        "You may answer in any clear format.\n"
        "ASSISTANT:"
    )
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()


@torch.no_grad()
def run_llava_once(wrapper, image: Image.Image, prompt: str, max_new_tokens: int = 192) -> str:
    processor = wrapper.processor
    model = wrapper.model

    single_input = processor(
        images=image,
        text=prompt,
        padding=True,
        return_tensors="pt",
    )

    single_input = {
        k: (v.to(wrapper.device) if torch.is_tensor(v) else v)
        for k, v in single_input.items()
        if v is not None
    }

    image_id = single_input["input_ids"] == model.config.image_token_index
    num_image_tokens = int(image_id.sum().item())

    if num_image_tokens != 1:
        decoded_prompt = processor.decode(
            single_input["input_ids"][0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        raise ValueError(
            f"Prompt must contain exactly 1 image token, got {num_image_tokens}.\n"
            f"Prompt:\n{prompt}\n\nDecoded:\n{decoded_prompt}"
        )

    prompt_len = int(single_input["input_ids"].shape[1])

    output = model.generate(
        input_ids=single_input["input_ids"],
        pixel_values=single_input["pixel_values"],
        attention_mask=single_input.get("attention_mask", None),
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

    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)

    if sampled_indices is not None:
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    out_root = os.path.join(args.out_dir, args.dataset, f"{args.model_name}_raw_bbox_outputs")
    os.makedirs(out_root, exist_ok=True)

    summary_csv = os.path.join(out_root, "summary_raw_outputs.csv")
    summary_json = os.path.join(out_root, "summary_raw_outputs.json")

    rows: List[Dict[str, Any]] = []

    for local_idx in range(start, end):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image = item["image_options"][0]

        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image)}")

        raw_question = rec["question"]
        gold_relation = clean_text(rec.get("answer", ""))

        obj1, obj2, stripped_q = extract_objects_from_question(raw_question)

        sample_dir = os.path.join(out_root, os.path.splitext(image_name)[0])
        os.makedirs(sample_dir, exist_ok=True)

        prompt_path = os.path.join(sample_dir, "prompt.txt")
        raw_output_path = os.path.join(sample_dir, "raw_output.txt")
        result_json_path = os.path.join(sample_dir, "result.json")

        if obj1 is None or obj2 is None:
            prompt = ""
            raw_output = ""
            error = "Could not extract objects from question."
        else:
            prompt = build_prompt(obj1, obj2, raw_question)

            if args.print_prompts:
                print("=" * 100)
                print("[PROMPT]")
                print(prompt)
                print("num_<image> =", prompt.count("<image>"))

            raw_output = run_llava_once(
                wrapper=wrapper,
                image=image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
            error = ""

        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        with open(raw_output_path, "w", encoding="utf-8") as f:
            f.write(raw_output)

        result = {
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            "original_question": raw_question,
            "clean_question": stripped_q,
            "gold_relation": gold_relation,
            "object_1": obj1,
            "object_2": obj2,
            "prompt": prompt,
            "raw_output": raw_output,
            "error": error,
            "prompt_path": prompt_path,
            "raw_output_path": raw_output_path,
        }

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        rows.append(result)

        print("=" * 100)
        print(f"[{local_idx}] {image_name}")
        print(f"question: {stripped_q}")
        print(f"objects: {obj1} | {obj2}")
        if error:
            print(f"[ERROR] {error}")
        print("[RAW OUTPUT]")
        print(raw_output)
        print(f"saved raw_output: {raw_output_path}")

    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "local_index",
            "image_name",
            "image_path",
            "original_question",
            "clean_question",
            "gold_relation",
            "object_1",
            "object_2",
            "prompt",
            "raw_output",
            "error",
            "prompt_path",
            "raw_output_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"Saved CSV: {summary_csv}")
    print(f"Saved JSON: {summary_json}")


if __name__ == "__main__":
    main()
