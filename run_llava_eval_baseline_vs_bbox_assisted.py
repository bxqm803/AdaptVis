import os
import re
import json
import argparse
from glob import glob
from typing import Any, Dict, List

import torch
from PIL import Image

from misc import seed_all
from model_zoo import get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default="output_llava_two_object_bbox_overlay/Controlled_Images_A/llava1.5_two_object_overlay",
    )
    p.add_argument("--model-name", default="llava1.5")
    p.add_argument("--method", default="base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--limit", default=10, type=int)
    p.add_argument("--max-new-tokens", default=16, type=int)
    p.add_argument("--image-mode", default="original", choices=["original", "overlay"])
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


def build_baseline_prompt(question: str) -> str:
    q = clean_question(question)

    return (
        "USER: <image>\n"
        f"{q}\n"
        "Answer with left, right, on or under.\n"
        "ASSISTANT:"
    )


def build_bbox_assisted_prompt(result: Dict[str, Any]) -> str:
    q = clean_question(result.get("question") or result.get("original_question") or "")
    obj1 = result.get("object_1")
    obj2 = result.get("object_2")

    return (
        "USER: <image>\n"
        f"Question: {q}\n"
        "Use the following bounding boxes to answer the spatial relation.\n"
        f"{obj1} bbox normalized [x1, y1, x2, y2]: {result.get('bbox1_norm')}\n"
        f"{obj2} bbox normalized [x1, y1, x2, y2]: {result.get('bbox2_norm')}\n"
        "Answer with left, right, on or under.\n"
        "ASSISTANT:"
    )


@torch.no_grad()
def run_llava_once(wrapper, image: Image.Image, prompt: str, max_new_tokens: int = 16) -> str:
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


def load_result_files(input_dir: str, limit: int) -> List[str]:
    paths = sorted(glob(os.path.join(input_dir, "*", "result.json")))
    return paths[:limit] if limit and limit > 0 else paths


def resolve_image_path(result: Dict[str, Any], sample_dir: str, image_mode: str) -> str:
    overlay_path = result.get("overlay_path") or os.path.join(sample_dir, "overlay_two_objects.png")

    if image_mode == "overlay":
        return overlay_path

    image_path = result.get("image_path")
    if image_path and os.path.exists(image_path):
        return image_path

    return overlay_path


def main():
    args = parse_args()
    seed_all(args.seed)

    wrapper, _ = get_model(args.model_name, args.device, args.method)
    result_paths = load_result_files(args.input_dir, args.limit)

    print(f"Loaded {len(result_paths)} result files from: {args.input_dir}")
    print(f"image_mode = {args.image_mode}")
    print("=" * 100)

    for idx, result_path in enumerate(result_paths):
        sample_dir = os.path.dirname(result_path)

        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        image_path = resolve_image_path(result, sample_dir, args.image_mode)

        if not os.path.exists(image_path):
            print("=" * 100)
            print(f"[{idx}] SKIP")
            print(f"missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        question = result.get("question") or result.get("original_question") or ""
        gold_relation = result.get("gold_relation", "")

        baseline_prompt = build_baseline_prompt(question)
        assisted_prompt = build_bbox_assisted_prompt(result)

        baseline_output = run_llava_once(
            wrapper=wrapper,
            image=image,
            prompt=baseline_prompt,
            max_new_tokens=args.max_new_tokens,
        )

        assisted_output = run_llava_once(
            wrapper=wrapper,
            image=image,
            prompt=assisted_prompt,
            max_new_tokens=args.max_new_tokens,
        )

        print("=" * 100)
        print(f"[{idx}] {result.get('image_name', os.path.basename(sample_dir))}")
        print(f"image_path: {image_path}")
        print(f"question: {clean_question(question)}")
        print(f"gold_relation: {gold_relation}")
        print(f"object_1: {result.get('object_1')}")
        print(f"bbox1_norm: {result.get('bbox1_norm')}")
        print(f"object_2: {result.get('object_2')}")
        print(f"bbox2_norm: {result.get('bbox2_norm')}")
        print("[BASELINE PROMPT]")
        print(baseline_prompt)
        print("[BASELINE OUTPUT]")
        print(baseline_output)
        print("[BBOX-ASSISTED PROMPT]")
        print(assisted_prompt)
        print("[BBOX-ASSISTED OUTPUT]")
        print(assisted_output)


if __name__ == "__main__":
    main()
