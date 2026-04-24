import os
import re
import csv
import json
import argparse
from typing import Any, Optional, Tuple, List, Dict

import torch
from PIL import Image, ImageDraw

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
    p.add_argument("--out-dir", default="output_llava_two_object_bbox_overlay", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=10, type=int)
    p.add_argument("--max-new-tokens", default=64, type=int)
    p.add_argument("--print-raw", action="store_true")
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


def build_single_object_prompt(obj: str) -> str:
    return (
        "USER: <image>\n"
        f"Locate the {obj} in the image.\n"
        "Answer with only one bounding box in this exact format:\n"
        "[x1, y1, x2, y2]\n"
        "Use normalized coordinates between 0 and 1 for the full image.\n"
        "Do not explain.\n"
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

    num_image_tokens = int((single_input["input_ids"] == model.config.image_token_index).sum().item())
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


def parse_bbox_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    text = text.replace("\\_", "_")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 4:
        return None

    vals = [float(x) for x in nums[:4]]
    x1, y1, x2, y2 = vals

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_to_pixels(
    bbox: Optional[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))

    if max_abs <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    elif max_abs <= 1001:
        x1 = x1 / 1000.0 * width
        x2 = x2 / 1000.0 * width
        y1 = y1 / 1000.0 * height
        y2 = y2 / 1000.0 * height

    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def draw_box(draw: ImageDraw.ImageDraw, box, label: str, color):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    try:
        tb = draw.textbbox((x1, y1), label)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        bg_y1 = max(0, y1 - th - 8)
        draw.rectangle([x1, bg_y1, x1 + tw + 8, bg_y1 + th + 6], fill=color)
        draw.text((x1 + 4, bg_y1 + 3), label, fill=(255, 255, 255))
    except Exception:
        draw.text((x1, max(0, y1 - 14)), label, fill=color)


def save_overlay(
    image: Image.Image,
    obj1: str,
    box1_px: Optional[Tuple[int, int, int, int]],
    obj2: str,
    box2_px: Optional[Tuple[int, int, int, int]],
    out_path: str,
):
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)

    if box1_px is not None:
        draw_box(draw, box1_px, obj1, (255, 0, 0))

    if box2_px is not None:
        draw_box(draw, box2_px, obj2, (0, 128, 255))

    vis.save(out_path)


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

    out_root = os.path.join(args.out_dir, args.dataset, f"{args.model_name}_two_object_overlay")
    os.makedirs(out_root, exist_ok=True)

    summary_csv = os.path.join(out_root, "summary_two_object_bbox_overlay.csv")
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
        clean_question = strip_answer_order_clause(raw_question)
        gold_relation = clean_text(rec.get("answer", ""))

        obj1, obj2, _ = extract_objects_from_question(raw_question)

        sample_name = os.path.splitext(image_name)[0]
        sample_dir = os.path.join(out_root, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        overlay_path = os.path.join(sample_dir, "overlay_two_objects.png")
        result_json_path = os.path.join(sample_dir, "result.json")
        raw1_path = os.path.join(sample_dir, "raw_output_obj1.txt")
        raw2_path = os.path.join(sample_dir, "raw_output_obj2.txt")

        raw1 = ""
        raw2 = ""
        bbox1_norm = None
        bbox2_norm = None
        bbox1_px = None
        bbox2_px = None
        error = ""

        if obj1 is None or obj2 is None:
            error = "Could not extract objects from question."
        else:
            prompt1 = build_single_object_prompt(obj1)
            prompt2 = build_single_object_prompt(obj2)

            raw1 = run_llava_once(wrapper, image, prompt1, args.max_new_tokens)
            raw2 = run_llava_once(wrapper, image, prompt2, args.max_new_tokens)

            bbox1_norm = parse_bbox_from_text(raw1)
            bbox2_norm = parse_bbox_from_text(raw2)

            w, h = image.size
            bbox1_px = bbox_to_pixels(bbox1_norm, w, h)
            bbox2_px = bbox_to_pixels(bbox2_norm, w, h)

            save_overlay(image, obj1, bbox1_px, obj2, bbox2_px, overlay_path)

            with open(raw1_path, "w", encoding="utf-8") as f:
                f.write(raw1)

            with open(raw2_path, "w", encoding="utf-8") as f:
                f.write(raw2)

        result = {
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            "question": clean_question,
            "gold_relation": gold_relation,
            "object_1": obj1,
            "object_2": obj2,
            "raw_output_obj1": raw1,
            "raw_output_obj2": raw2,
            "bbox1_norm": list(bbox1_norm) if bbox1_norm is not None else None,
            "bbox2_norm": list(bbox2_norm) if bbox2_norm is not None else None,
            "bbox1_px": list(bbox1_px) if bbox1_px is not None else None,
            "bbox2_px": list(bbox2_px) if bbox2_px is not None else None,
            "overlay_path": overlay_path if os.path.exists(overlay_path) else "",
            "error": error,
        }

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        rows.append(result)

        print("=" * 100)
        print(f"[{local_idx}] {image_name}")
        print(f"question: {clean_question}")
        print(f"objects: {obj1} | {obj2}")

        if args.print_raw:
            print(f"[RAW {obj1}] {raw1}")
            print(f"[RAW {obj2}] {raw2}")

        print(f"bbox1_norm={bbox1_norm} bbox1_px={bbox1_px}")
        print(f"bbox2_norm={bbox2_norm} bbox2_px={bbox2_px}")
        print(f"overlay saved: {overlay_path if os.path.exists(overlay_path) else 'N/A'}")

    fieldnames = [
        "local_index",
        "image_name",
        "image_path",
        "question",
        "gold_relation",
        "object_1",
        "object_2",
        "raw_output_obj1",
        "raw_output_obj2",
        "bbox1_norm",
        "bbox2_norm",
        "bbox1_px",
        "bbox2_px",
        "overlay_path",
        "error",
    ]

    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            rr = dict(r)
            for k in ["bbox1_norm", "bbox2_norm", "bbox1_px", "bbox2_px"]:
                rr[k] = json.dumps(rr[k], ensure_ascii=False)
            writer.writerow(rr)

    print("=" * 100)
    print(f"Saved summary: {summary_csv}")
    print(f"Saved overlays under: {out_root}")


if __name__ == "__main__":
    main()
