import os
import re
import csv
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from dataset_zoo import get_dataset
from misc import seed_all


SUPPORTED_VLM_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.5-9B",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-9B",
        type=str,
        help="Examples: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3.5-9B",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--option", default="four", type=str)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--cache-dir", default=None, type=str)
    parser.add_argument("--out-dir", default="output_qwen_bbox", type=str)
    parser.add_argument(
        "--ask-mode",
        default="single",
        choices=["single", "joint"],
        help="single: ask one object per generation; joint: ask all objects together once.",
    )
    parser.add_argument(
        "--auto-rescale-1000",
        action="store_true",
        help="If bbox looks like 0~1000 normalized coordinates, rescale to image pixels.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a sample if bbox.json already exists.",
    )
    return parser.parse_args()


def load_prompt_records(dataset_name: str, option: str):
    prompt_path = Path("prompts") / f"{dataset_name}_with_answer_{option}_options.jsonl"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    records = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def clean_text(x):
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def strip_legacy_prompt(prompt: str) -> str:
    prompt = clean_text(prompt)
    prompt = prompt.replace("<image>", "").strip()
    prompt = re.sub(r"^\s*USER:\s*", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s*ASSISTANT:\s*$", "", prompt, flags=re.IGNORECASE)
    return clean_text(prompt)


def strip_answer_order_clause(question_text: str):
    q = clean_text(question_text)
    q = re.sub(
        r"Answer with\s+left,\s*right,\s*on\s+or\s+under(?:\s+only)?\.\s*$",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q


def normalize_object_name(name: str) -> str:
    name = clean_text(name).lower()
    name = re.sub(r"^(a|an|the)\s+", "", name)
    name = re.sub(r"[?.!,;:]+$", "", name)
    name = clean_text(name)
    return name


def extract_two_objects_from_question(question_text: str):
    """
    Tries to parse object1/object2 from the original spatial-relation question.
    """
    q = strip_legacy_prompt(question_text)
    q = strip_answer_order_clause(q)
    q_low = q.lower()

    patterns = [
        r"where is the (.+?) with respect to the (.+?)\??$",
        r"is the (.+?) to the left of the (.+?)\??$",
        r"is the (.+?) to the right of the (.+?)\??$",
        r"is the (.+?) on the (.+?)\??$",
        r"is the (.+?) under the (.+?)\??$",
        r"is the (.+?) below the (.+?)\??$",
        r"is the (.+?) beneath the (.+?)\??$",
        r"is the (.+?) above the (.+?)\??$",
        r"is the (.+?) on top of the (.+?)\??$",
    ]

    for pat in patterns:
        m = re.search(pat, q_low, flags=re.IGNORECASE)
        if m:
            obj1 = normalize_object_name(m.group(1))
            obj2 = normalize_object_name(m.group(2))
            if obj1 and obj2:
                return [obj1, obj2]

    # Fallback: try "X with respect to Y"
    m = re.search(r"(.+?) with respect to (.+?)\??$", q_low, flags=re.IGNORECASE)
    if m:
        obj1 = normalize_object_name(m.group(1))
        obj2 = normalize_object_name(m.group(2))
        if obj1 and obj2:
            return [obj1, obj2]

    return []


def make_user_messages(image, question_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ],
        }
    ]


def build_inputs(processor, messages, add_generation_prompt=True):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )


def build_single_bbox_prompt(obj_name, image_w=None, image_h=None):
    size_hint = ""
    if image_w is not None and image_h is not None:
        size_hint = f"The image size is width={image_w}, height={image_h}. "

    return (
        f'{size_hint}'
        f'Find the object "{obj_name}" in the image. '
        f'Return JSON only, with no extra words and no markdown fences. '
        f'Use exactly this schema: '
        f'{{"label":"{obj_name}","bbox_2d":[x1,y1,x2,y2]}}. '
        f'If the object is not visible, return '
        f'{{"label":"{obj_name}","bbox_2d":null}}. '
        f'Use integer coordinates. Prefer a tight box.'
    )


def build_joint_bbox_prompt(object_names, image_w=None, image_h=None):
    size_hint = ""
    if image_w is not None and image_h is not None:
        size_hint = f"The image size is width={image_w}, height={image_h}. "

    obj_list = ", ".join([f'"{x}"' for x in object_names])

    # Build template objects
    obj_template = ", ".join(
        [f'{{"label":"{x}","bbox_2d":[x1,y1,x2,y2]}}' for x in object_names]
    )

    return (
        f"{size_hint}"
        f"Locate these objects in the image: {obj_list}. "
        f"Return JSON only, with no extra words and no markdown fences. "
        f'Use exactly this schema: {{"objects":[{obj_template}]}}. '
        f"If an object is not visible, use null for its bbox_2d. "
        f"Use integer coordinates. Prefer tight boxes."
    )


@torch.no_grad()
def generate_free(model, processor, image, question_text, max_new_tokens=256, temperature=0.0):
    messages = make_user_messages(image, question_text)
    inputs = build_inputs(processor, messages, add_generation_prompt=True)

    model_device = next(model.parameters()).device
    inputs = {
        k: (v.to(model_device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": False,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, prompt_len:]

    pred_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return {"pred_text": pred_text}


def extract_post_think_text(text: str):
    """
    If the model emits thinking text, keep only the suffix after </think>.
    If not present, return original text.
    """
    if text is None:
        return "", False

    text = str(text).strip()
    if not text:
        return "", False

    low = text.lower()
    tag = "</think>"
    idx = low.rfind(tag)

    if idx == -1:
        return text, False

    suffix = text[idx + len(tag):].strip()
    if suffix:
        return suffix, True

    return text, True


def extract_json_block(text: str):
    if text is None:
        return None

    text = str(text).strip()

    # remove markdown fences if any
    text = re.sub(r"^\s*```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*```\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # try dict block
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except Exception:
            continue

    # try greedy dict
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # try list
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def clip_bbox_to_image(bbox, image_w, image_h):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), image_w - 1))
    y1 = max(0, min(int(y1), image_h - 1))
    x2 = max(0, min(int(x2), image_w - 1))
    y2 = max(0, min(int(y2), image_h - 1))
    return [x1, y1, x2, y2]


def maybe_rescale_from_1000(bbox, image_w, image_h, auto_rescale_1000=False):
    if bbox is None:
        return None

    if not auto_rescale_1000:
        return bbox

    # If all coordinates lie in [0,1000], treat as 1000-normalized coords.
    if all(0 <= float(v) <= 1000 for v in bbox):
        x1, y1, x2, y2 = bbox
        x1 = int(round(float(x1) * image_w / 1000.0))
        y1 = int(round(float(y1) * image_h / 1000.0))
        x2 = int(round(float(x2) * image_w / 1000.0))
        y2 = int(round(float(y2) * image_h / 1000.0))
        return [x1, y1, x2, y2]

    return bbox


def normalize_bbox_item(item, image_w, image_h, auto_rescale_1000=False):
    if not isinstance(item, dict):
        return None

    label = clean_text(item.get("label", ""))
    bbox = item.get("bbox_2d", None)

    if bbox is None:
        return {
            "label": label,
            "bbox_2d": None,
        }

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        bbox = [float(v) for v in bbox]
    except Exception:
        return None

    bbox = maybe_rescale_from_1000(
        bbox,
        image_w=image_w,
        image_h=image_h,
        auto_rescale_1000=auto_rescale_1000,
    )

    try:
        bbox = [int(round(v)) for v in bbox]
    except Exception:
        return None

    bbox = clip_bbox_to_image(bbox, image_w, image_h)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    # allow slightly malformed output but try to fix ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return {
        "label": label,
        "bbox_2d": [x1, y1, x2, y2],
    }


def parse_bbox_output(text, image_w, image_h, auto_rescale_1000=False):
    payload = extract_json_block(text)
    if payload is None:
        return None

    # case 1: single object dict
    if isinstance(payload, dict) and "bbox_2d" in payload:
        item = normalize_bbox_item(payload, image_w, image_h, auto_rescale_1000)
        if item is None:
            return None
        return [item]

    # case 2: {"objects": [...]}
    if isinstance(payload, dict) and "objects" in payload and isinstance(payload["objects"], list):
        out = []
        for x in payload["objects"]:
            item = normalize_bbox_item(x, image_w, image_h, auto_rescale_1000)
            if item is not None:
                out.append(item)
        return out if out else None

    # case 3: list of dicts
    if isinstance(payload, list):
        out = []
        for x in payload:
            item = normalize_bbox_item(x, image_w, image_h, auto_rescale_1000)
            if item is not None:
                out.append(item)
        return out if out else None

    return None


def bbox_to_string(bbox):
    if bbox is None:
        return ""
    return json.dumps(bbox, ensure_ascii=False)


def find_bbox_by_label(parsed_items, target_label):
    if parsed_items is None:
        return None

    target_label = normalize_object_name(target_label)

    # exact normalized match first
    for item in parsed_items:
        label = normalize_object_name(item.get("label", ""))
        if label == target_label:
            return item.get("bbox_2d", None)

    # loose contains match
    for item in parsed_items:
        label = normalize_object_name(item.get("label", ""))
        if target_label in label or label in target_label:
            return item.get("bbox_2d", None)

    return None


def load_model_and_processor(args, cache_dir):
    print(f"Loading model: {args.model_id}")
    print(f"Using cache_dir: {cache_dir}")

    if args.model_id not in SUPPORTED_VLM_MODELS:
        print(f"[Warning] {args.model_id} not in tested list: {SUPPORTED_VLM_MODELS}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        device_map="auto" if args.device.startswith("cuda") else None,
        torch_dtype="auto",
    ).eval()

    if not args.device.startswith("cuda"):
        model.to(args.device)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
    )

    return model, processor


def main():
    args = parse_args()
    seed_all(args.seed)

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ['USER']}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    prompt_records = load_prompt_records(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)

    if len(prompt_records) != len(dataset):
        raise ValueError(
            f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)})."
        )

    model, processor = load_model_and_processor(args, cache_dir)

    if args.limit < 0:
        end_index = len(prompt_records)
    else:
        end_index = min(args.start_index + args.limit, len(prompt_records))

    model_name = args.model_id.split("/")[-1]
    out_root = os.path.join(args.out_dir, args.dataset, model_name)
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []
    summary_csv = os.path.join(out_root, "summary_bbox.csv")

    for local_idx in tqdm(range(args.start_index, end_index), desc=f"{args.dataset}:{model_name}"):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image = item["image_options"][0]
        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")
        image_stem = os.path.splitext(image_name)[0]

        sample_dir = os.path.join(out_root, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        bbox_json_path = os.path.join(sample_dir, "bbox.json")
        if args.skip_existing and os.path.exists(bbox_json_path):
            continue

        base_question = strip_legacy_prompt(rec["question"])
        base_question = strip_answer_order_clause(base_question)

        object_names = extract_two_objects_from_question(rec["question"])

        image_w, image_h = image.size

        sample_record = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "image_width": image_w,
            "image_height": image_h,
            "base_question": base_question,
            "object_names": object_names,
            "ask_mode": args.ask_mode,
            "model_id": args.model_id,
            "results": [],
        }

        obj1 = object_names[0] if len(object_names) >= 1 else ""
        obj2 = object_names[1] if len(object_names) >= 2 else ""

        obj1_bbox = None
        obj2_bbox = None
        obj1_raw = ""
        obj2_raw = ""
        joint_raw = ""
        joint_parsed = None

        if len(object_names) == 0:
            sample_record["error"] = "Failed to parse object names from question."
        else:
            if args.ask_mode == "single":
                for obj_name in object_names:
                    prompt_text = build_single_bbox_prompt(
                        obj_name=obj_name,
                        image_w=image_w,
                        image_h=image_h,
                    )

                    gen_out = generate_free(
                        model=model,
                        processor=processor,
                        image=image,
                        question_text=prompt_text,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )

                    pred_text = gen_out["pred_text"]
                    postthink_text, has_think_close = extract_post_think_text(pred_text)
                    parsed_items = parse_bbox_output(
                        postthink_text,
                        image_w=image_w,
                        image_h=image_h,
                        auto_rescale_1000=args.auto_rescale_1000,
                    )
                    bbox = find_bbox_by_label(parsed_items, obj_name)

                    sample_record["results"].append({
                        "label": obj_name,
                        "prompt_text": prompt_text,
                        "raw_output": pred_text,
                        "has_think_close": has_think_close,
                        "postthink_text": postthink_text,
                        "parsed_items": parsed_items,
                        "bbox_2d": bbox,
                    })

                    if normalize_object_name(obj_name) == normalize_object_name(obj1):
                        obj1_bbox = bbox
                        obj1_raw = pred_text
                    if normalize_object_name(obj_name) == normalize_object_name(obj2):
                        obj2_bbox = bbox
                        obj2_raw = pred_text

            else:  # joint
                prompt_text = build_joint_bbox_prompt(
                    object_names=object_names,
                    image_w=image_w,
                    image_h=image_h,
                )

                gen_out = generate_free(
                    model=model,
                    processor=processor,
                    image=image,
                    question_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

                pred_text = gen_out["pred_text"]
                postthink_text, has_think_close = extract_post_think_text(pred_text)
                parsed_items = parse_bbox_output(
                    postthink_text,
                    image_w=image_w,
                    image_h=image_h,
                    auto_rescale_1000=args.auto_rescale_1000,
                )

                sample_record["results"].append({
                    "label": "__joint__",
                    "prompt_text": prompt_text,
                    "raw_output": pred_text,
                    "has_think_close": has_think_close,
                    "postthink_text": postthink_text,
                    "parsed_items": parsed_items,
                })

                joint_raw = pred_text
                joint_parsed = parsed_items
                if obj1:
                    obj1_bbox = find_bbox_by_label(parsed_items, obj1)
                if obj2:
                    obj2_bbox = find_bbox_by_label(parsed_items, obj2)

        with open(bbox_json_path, "w", encoding="utf-8") as f:
            json.dump(sample_record, f, indent=2, ensure_ascii=False)

        row = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "image_width": image_w,
            "image_height": image_h,
            "base_question": base_question,
            "object_names_json": json.dumps(object_names, ensure_ascii=False),
            "object_1": obj1,
            "object_1_found": obj1_bbox is not None,
            "object_1_bbox": bbox_to_string(obj1_bbox),
            "object_2": obj2,
            "object_2_found": obj2_bbox is not None,
            "object_2_bbox": bbox_to_string(obj2_bbox),
            "ask_mode": args.ask_mode,
            "bbox_json": os.path.join(image_stem, "bbox.json"),
            "joint_raw_output": joint_raw,
            "joint_parsed_json": json.dumps(joint_parsed, ensure_ascii=False) if joint_parsed is not None else "",
            "object_1_raw_output": obj1_raw,
            "object_2_raw_output": obj2_raw,
        }
        summary_rows.append(row)

        fieldnames = [
            "image_name",
            "image_path",
            "local_index",
            "image_width",
            "image_height",
            "base_question",
            "object_names_json",
            "object_1",
            "object_1_found",
            "object_1_bbox",
            "object_2",
            "object_2_found",
            "object_2_bbox",
            "ask_mode",
            "bbox_json",
            "joint_raw_output",
            "joint_parsed_json",
            "object_1_raw_output",
            "object_2_raw_output",
        ]

        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
