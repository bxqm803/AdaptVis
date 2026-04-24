import os
import re
import csv
import json
import argparse
from glob import glob
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from misc import seed_all
from model_zoo import get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default="output_llava_two_object_bbox_overlay/Controlled_Images_A/llava1.5_two_object_overlay",
        type=str,
    )
    p.add_argument(
        "--out-dir",
        default="output_llava_bbox_assisted_eval",
        type=str,
    )
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--limit", default=10, type=int)
    p.add_argument("--max-new-tokens", default=64, type=int)
    p.add_argument(
        "--image-mode",
        default="original",
        choices=["original", "overlay"],
        help="original = use original image_path from json; overlay = use overlay_two_objects.png",
    )
    p.add_argument("--print-raw", action="store_true")
    return p.parse_args()


def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()


def clean_question(question: str) -> str:
    q = clean_text(question)
    q = q.replace("<image>", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()

    q = re.sub(r"\s*(?:Please\s+)?(?:answer|respond).*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Your answer should be.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Answer with.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Choose from.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()

    if q and q[-1] not in "?.!":
        q += "?"
    return q


def build_baseline_prompt(question: str) -> str:
    q = clean_question(question)
    return (
        "USER: <image>\n"
        f"{q}\n"
        "Answer the spatial relation concisely.\n"
        "ASSISTANT:"
    )


def build_bbox_assisted_prompt(result: Dict[str, Any]) -> str:
    question = clean_question(result.get("question") or result.get("original_question") or "")
    obj1 = result.get("object_1")
    obj2 = result.get("object_2")

    bbox1_norm = result.get("bbox1_norm")
    bbox2_norm = result.get("bbox2_norm")
    bbox1_px = result.get("bbox1_px")
    bbox2_px = result.get("bbox2_px")

    return (
        "USER: <image>\n"
        "You are given an image and predicted bounding boxes for two objects.\n"
        "Use the bounding box information to answer the spatial relation question.\n\n"
        f"Question: {question}\n\n"
        f"Object 1: {obj1}\n"
        f"Object 1 bbox normalized [x1, y1, x2, y2]: {bbox1_norm}\n"
        f"Object 1 bbox pixels [x1, y1, x2, y2]: {bbox1_px}\n\n"
        f"Object 2: {obj2}\n"
        f"Object 2 bbox normalized [x1, y1, x2, y2]: {bbox2_norm}\n"
        f"Object 2 bbox pixels [x1, y1, x2, y2]: {bbox2_px}\n\n"
        "Answer the spatial relation concisely.\n"
        "ASSISTANT:"
    )


@torch.no_grad()
def run_llava_once(wrapper, image: Image.Image, prompt: str, max_new_tokens: int = 64) -> str:
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

    num_image_tokens = int(
        (single_input["input_ids"] == model.config.image_token_index).sum().item()
    )

    if num_image_tokens != 1:
        decoded = processor.decode(
            single_input["input_ids"][0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        raise ValueError(
            f"Prompt must contain exactly 1 image token, got {num_image_tokens}.\n"
            f"Prompt:\n{prompt}\n\nDecoded:\n{decoded}"
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


def load_result_files(input_dir: str, limit: int) -> List[str]:
    paths = sorted(glob(os.path.join(input_dir, "*", "result.json")))
    if limit is not None and limit > 0:
        paths = paths[:limit]
    return paths


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

    os.makedirs(args.out_dir, exist_ok=True)

    wrapper, _ = get_model(args.model_name, args.device, args.method)

    result_paths = load_result_files(args.input_dir, args.limit)

    rows: List[Dict[str, Any]] = []

    for idx, result_path in enumerate(result_paths):
        sample_dir = os.path.dirname(result_path)

        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        image_path = resolve_image_path(result, sample_dir, args.image_mode)

        if not os.path.exists(image_path):
            print(f"[SKIP] missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        image_name = result.get("image_name", os.path.basename(sample_dir))
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

        out_sample_dir = os.path.join(args.out_dir, os.path.basename(sample_dir))
        os.makedirs(out_sample_dir, exist_ok=True)

        with open(os.path.join(out_sample_dir, "baseline_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_prompt)

        with open(os.path.join(out_sample_dir, "bbox_assisted_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(assisted_prompt)

        with open(os.path.join(out_sample_dir, "baseline_output.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_output)

        with open(os.path.join(out_sample_dir, "bbox_assisted_output.txt"), "w", encoding="utf-8") as f:
            f.write(assisted_output)

        output_json = {
            "index": idx,
            "image_name": image_name,
            "image_path_used": image_path,
            "image_mode": args.image_mode,
            "question": clean_question(question),
            "gold_relation": gold_relation,
            "object_1": result.get("object_1"),
            "object_2": result.get("object_2"),
            "bbox1_norm": result.get("bbox1_norm"),
            "bbox2_norm": result.get("bbox2_norm"),
            "bbox1_px": result.get("bbox1_px"),
            "bbox2_px": result.get("bbox2_px"),
            "baseline_prompt": baseline_prompt,
            "bbox_assisted_prompt": assisted_prompt,
            "baseline_output": baseline_output,
            "bbox_assisted_output": assisted_output,
        }

        with open(os.path.join(out_sample_dir, "eval_result.json"), "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        rows.append(output_json)

        print("=" * 100)
        print(f"[{idx}] {image_name}")
        print(f"question: {clean_question(question)}")
        print(f"gold_relation: {gold_relation}")
        print(f"object_1: {result.get('object_1')} bbox1_norm={result.get('bbox1_norm')}")
        print(f"object_2: {result.get('object_2')} bbox2_norm={result.get('bbox2_norm')}")
        print(f"[BASELINE] {baseline_output}")
        print(f"[BBOX-ASSISTED] {assisted_output}")

        if args.print_raw:
            print("[BASELINE PROMPT]")
            print(baseline_prompt)
            print("[BBOX ASSISTED PROMPT]")
            print(assisted_prompt)

    summary_csv = os.path.join(args.out_dir, "summary_baseline_vs_bbox_assisted.csv")
    summary_json = os.path.join(args.out_dir, "summary_baseline_vs_bbox_assisted.json")

    fieldnames = [
        "index",
        "image_name",
        "image_path_used",
        "image_mode",
        "question",
        "gold_relation",
        "object_1",
        "object_2",
        "bbox1_norm",
        "bbox2_norm",
        "bbox1_px",
        "bbox2_px",
        "baseline_output",
        "bbox_assisted_output",
    ]

    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            rr = {k: r.get(k, "") for k in fieldnames}
            for k in ["bbox1_norm", "bbox2_norm", "bbox1_px", "bbox2_px"]:
                rr[k] = json.dumps(rr[k], ensure_ascii=False)
            writer.writerow(rr)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"Saved CSV: {summary_csv}")
    print(f"Saved JSON: {summary_json}")


if __name__ == "__main__":
    main()
