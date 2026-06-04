# run_bbox_prompt_attention.py

import os
import re
import csv
import json
import math
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from transformers import AutoProcessor


def load_json_or_jsonl(path: str):
    path = str(path)
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_first_existing_key(item: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in item and item[k] is not None:
            return item[k]
    return default


def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def normalize_bbox_1000(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
        int(round(x2 / width * 1000)),
        int(round(y2 / height * 1000)),
    ]


def bbox_to_region(bbox_norm: List[int]) -> str:
    x1, y1, x2, y2 = bbox_norm
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < 333:
        horizontal = "left"
    elif cx < 666:
        horizontal = "center"
    else:
        horizontal = "right"

    if cy < 333:
        vertical = "upper"
    elif cy < 666:
        vertical = "middle"
    else:
        vertical = "lower"

    return f"{horizontal}-{vertical}"


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def load_groundingdino_detections(path: str):
    """
    支持两种格式：

    1. dict:
    {
      "xxx.jpg": [
        {"label": "cat", "bbox": [x1,y1,x2,y2], "score": 0.9},
        ...
      ]
    }

    2. list:
    [
      {
        "image": "xxx.jpg",
        "detections": [
          {"label": "cat", "bbox": [x1,y1,x2,y2], "score": 0.9}
        ]
      }
    ]

    或者：
    [
      {"image": "xxx.jpg", "label": "cat", "bbox": [...], "score": 0.9},
      ...
    ]
    """
    raw = load_json_or_jsonl(path)

    det_map = {}

    if isinstance(raw, dict):
        for image_key, dets in raw.items():
            det_map[str(image_key)] = dets
        return det_map

    if isinstance(raw, list):
        for item in raw:
            image_key = get_first_existing_key(
                item,
                ["image", "image_path", "img_path", "file_name", "filename", "image_id", "id"],
            )

            if image_key is None:
                continue

            image_key = str(image_key)

            if "detections" in item:
                det_map.setdefault(image_key, []).extend(item["detections"])
            else:
                det = {
                    "label": get_first_existing_key(
                        item,
                        ["label", "class", "class_name", "phrase", "object", "name"],
                    ),
                    "bbox": get_first_existing_key(
                        item,
                        ["bbox", "box", "xyxy"],
                    ),
                    "score": get_first_existing_key(
                        item,
                        ["score", "confidence", "logit"],
                        1.0,
                    ),
                }
                det_map.setdefault(image_key, []).append(det)

        return det_map

    raise ValueError(f"Unsupported detection json format: {type(raw)}")


def match_detections_for_image(det_map, image_path: str, image_id: Optional[str] = None):
    candidates = []

    keys = []
    if image_id is not None:
        keys.append(str(image_id))

    image_path = str(image_path)
    keys.append(image_path)
    keys.append(os.path.basename(image_path))
    keys.append(Path(image_path).stem)

    for k in keys:
        if k in det_map:
            candidates.extend(det_map[k])

    return candidates


def find_best_detection(dets: List[Dict[str, Any]], obj_name: str):
    """
    优先 exact match；然后 substring match；同名多个框取 score 最高。
    """
    target = normalize_name(obj_name)

    exact = []
    partial = []

    for det in dets:
        label = get_first_existing_key(
            det,
            ["label", "class", "class_name", "phrase", "object", "name"],
        )
        bbox = get_first_existing_key(det, ["bbox", "box", "xyxy"])
        if label is None or bbox is None:
            continue

        label_norm = normalize_name(label)
        score = float(get_first_existing_key(det, ["score", "confidence", "logit"], 1.0))

        wrapped = {
            "label": label,
            "bbox": bbox,
            "score": score,
        }

        if label_norm == target:
            exact.append(wrapped)
        elif target in label_norm or label_norm in target:
            partial.append(wrapped)

    pool = exact if exact else partial

    if not pool:
        return None

    pool = sorted(pool, key=lambda x: x["score"], reverse=True)
    return pool[0]


def parse_two_objects_from_question(question: str):
    """
    只是 fallback。
    最稳还是你的 data json 里直接有 object_1/object_2 字段。
    """
    q = question.strip()

    patterns = [
        r"Is the (.*?) to the .*? of the (.*?)\?",
        r"is the (.*?) to the .*? of the (.*?)\?",
        r"whether the (.*?) is .*? the (.*?)[\?\.]",
        r"between the (.*?) and the (.*?)[\?\.]",
    ]

    for p in patterns:
        m = re.search(p, q)
        if m:
            obj1 = m.group(1).strip()
            obj2 = m.group(2).strip()
            obj1 = re.sub(r"^(a|an|the)\s+", "", obj1, flags=re.I)
            obj2 = re.sub(r"^(a|an|the)\s+", "", obj2, flags=re.I)
            return obj1, obj2

    return None, None


def get_objects(item: Dict[str, Any], question: str):
    obj1 = get_first_existing_key(
        item,
        [
            "object_1",
            "obj1",
            "subject",
            "target_object",
            "objectA",
            "object_a",
            "left_object",
        ],
    )
    obj2 = get_first_existing_key(
        item,
        [
            "object_2",
            "obj2",
            "reference",
            "reference_object",
            "objectB",
            "object_b",
            "right_object",
        ],
    )

    if obj1 is None or obj2 is None:
        p1, p2 = parse_two_objects_from_question(question)
        obj1 = obj1 if obj1 is not None else p1
        obj2 = obj2 if obj2 is not None else p2

    return obj1, obj2


def build_bbox_prompt(
    original_question: str,
    obj1_name: str,
    obj1_bbox_norm: List[int],
    obj1_region: str,
    obj2_name: str,
    obj2_bbox_norm: List[int],
    obj2_region: str,
    question_mode: str = "keep",
):
    bbox_context = (
        "given two object bbox\n"
        f"object_1: {obj1_name}, location={obj1_region}, bbox={obj1_bbox_norm}\n"
        f"object_2: {obj2_name}, location={obj2_region}, bbox={obj2_bbox_norm}\n\n"
    )

    if question_mode == "simple_qa":
        return bbox_context + f"Question: {original_question}\nAnswer:"

    return bbox_context + original_question


def load_model_and_processor(model_id: str, dtype: str = "bf16"):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForVision2Seq,
    )

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = None
    last_err = None

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except Exception as e:
        last_err = e

    if model is None:
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception as e:
            last_err = e

    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model. Last error: {last_err}\n{e}")

    model.eval()
    model.config.output_attentions = True

    return model, processor


def build_model_inputs(model_id: str, processor, image: Image.Image, prompt: str, device):
    model_name = model_id.lower()

    if "qwen" in model_name and hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
    elif hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
        except Exception:
            text = "<image>\n" + prompt
            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
            )
    else:
        text = "<image>\n" + prompt
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
        )

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    return inputs


def get_image_token_ids(model, processor):
    """
    尽量自动找 image token id。
    Qwen2-VL 常见是 <|image_pad|>。
    LLaVA 常见是 config.image_token_index。
    """
    ids = set()

    cfg = getattr(model, "config", None)
    for attr in [
        "image_token_id",
        "image_token_index",
        "vision_token_id",
        "vision_start_token_id",
    ]:
        if cfg is not None and hasattr(cfg, attr):
            val = getattr(cfg, attr)
            if isinstance(val, int):
                ids.add(val)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        for tok in [
            "<image>",
            "<|image_pad|>",
            "<im_patch>",
        ]:
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != tokenizer.unk_token_id:
                    ids.add(int(tid))
            except Exception:
                pass

    return sorted(ids)


def infer_visual_and_text_positions(
    input_ids: torch.Tensor,
    attn_len: int,
    image_token_ids: List[int],
):
    """
    batch_size=1。
    返回 visual_positions, text_positions。

    情况 A：attention seq len == input_ids len
        visual token 通常就是 input_ids 里的 image_pad token。
    情况 B：attention seq len > input_ids len
        LLaVA 这类可能把一个 <image> placeholder 展开成很多 visual token。
    """
    ids = input_ids[0].detach().cpu().tolist()
    input_len = len(ids)

    placeholder_positions = [
        i for i, tid in enumerate(ids)
        if int(tid) in set(image_token_ids)
    ]

    if len(placeholder_positions) == 0:
        raise RuntimeError(
            "No image token found in input_ids. "
            "Please manually set image token logic for this model."
        )

    if attn_len == input_len:
        visual_positions = placeholder_positions
        text_positions = [
            i for i in range(attn_len)
            if i not in set(visual_positions)
        ]
        return visual_positions, text_positions

    # one or several image placeholders expanded into many visual tokens
    start = placeholder_positions[0]
    n_visual = attn_len - input_len + len(placeholder_positions)

    visual_positions = list(range(start, start + n_visual))

    text_positions = list(range(0, start))
    after_start_in_input = placeholder_positions[-1] + 1
    n_after = input_len - after_start_in_input
    text_positions += list(range(start + n_visual, start + n_visual + n_after))

    text_positions = [p for p in text_positions if 0 <= p < attn_len]
    visual_positions = [p for p in visual_positions if 0 <= p < attn_len]

    return visual_positions, text_positions


def closest_factor_grid(n: int):
    root = int(math.sqrt(n))
    best_h, best_w = 1, n
    best_gap = n

    for h in range(1, root + 1):
        if n % h == 0:
            w = n // h
            gap = abs(w - h)
            if gap < best_gap:
                best_h, best_w = h, w
                best_gap = gap

    return best_h, best_w


def get_visual_grid_shape(inputs, n_visual: int):
    """
    Qwen2-VL / Qwen2.5-VL 一般有 image_grid_thw。
    否则 fallback 到最接近正方形的因子分解。
    """
    if "image_grid_thw" in inputs:
        grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
        if len(grid) == 3:
            t, h, w = grid
            h = int(h)
            w = int(w)
            if h * w == n_visual:
                return h, w
            if t * h * w == n_visual:
                return int(t * h), int(w)

    h, w = closest_factor_grid(n_visual)
    return h, w


def save_attention_heatmap(
    image: Image.Image,
    values_1d: np.ndarray,
    grid_shape: Tuple[int, int],
    save_path: str,
    title: str = "",
):
    h, w = grid_shape

    if len(values_1d) != h * w:
        h, w = closest_factor_grid(len(values_1d))

    heat = values_1d.reshape(h, w)
    heat = heat.astype(np.float32)

    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)

    image_np = np.array(image.convert("RGB"))
    image_h, image_w = image_np.shape[:2]

    heat_img = Image.fromarray(np.uint8(heat * 255)).resize(
        (image_w, image_h),
        resample=Image.BILINEAR,
    )
    heat_np = np.array(heat_img)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(heat_np, alpha=0.45, cmap="jet")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--dino_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./bbox_prompt_attn_out")

    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument(
        "--question_mode",
        type=str,
        default="keep",
        choices=["keep", "simple_qa"],
        help=(
            "keep: bbox context + original question/prompt, 不改原始格式；"
            "simple_qa: bbox context + Question: ... Answer:"
        ),
    )

    parser.add_argument(
        "--print_first_n",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num_heatmap_images",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    heatmap_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    data = load_json_or_jsonl(args.data_json)
    if isinstance(data, dict):
        # 如果你的 json 是 {"data": [...]} 这种
        if "data" in data:
            data = data["data"]
        elif "examples" in data:
            data = data["examples"]
        else:
            raise ValueError("data_json is dict but no `data` or `examples` key found.")

    if args.max_samples > 0:
        data = data[:args.max_samples]

    det_map = load_groundingdino_detections(args.dino_json)

    print(f"Loaded {len(data)} samples")
    print(f"Loaded detection entries: {len(det_map)}")

    model, processor = load_model_and_processor(args.model_id, args.dtype)
    device = next(model.parameters()).device

    image_token_ids = get_image_token_ids(model, processor)
    print(f"Detected image token ids: {image_token_ids}")

    heatmap_indices = set(
        random.sample(
            range(len(data)),
            k=min(args.num_heatmap_images, len(data)),
        )
    )

    mass_csv_path = out_dir / "attention_mass.csv"
    prompt_preview_path = out_dir / "prompt_preview_first5.txt"
    failed_path = out_dir / "failed_bbox_samples.jsonl"

    mass_fields = [
        "sample_idx",
        "sample_id",
        "image_path",
        "object_1",
        "object_2",
        "layer",
        "text_attention_mass",
        "visual_attention_mass",
        "other_attention_mass",
        "num_text_tokens",
        "num_visual_tokens",
        "question",
        "prompt",
    ]

    failed_rows = []

    with open(mass_csv_path, "w", newline="", encoding="utf-8") as f_csv, \
         open(prompt_preview_path, "w", encoding="utf-8") as f_preview:

        writer = csv.DictWriter(f_csv, fieldnames=mass_fields)
        writer.writeheader()

        for idx, item in enumerate(data):
            sample_id = get_first_existing_key(
                item,
                ["id", "sample_id", "question_id", "image_id"],
                default=str(idx),
            )

            image_rel = get_first_existing_key(
                item,
                ["image", "image_path", "img_path", "file_name", "filename"],
            )

            question = get_first_existing_key(
                item,
                ["question", "prompt", "text", "query"],
            )

            if image_rel is None or question is None:
                failed_rows.append({
                    "sample_idx": idx,
                    "reason": "missing image or question",
                    "item": item,
                })
                continue

            image_path = image_rel
            if args.image_root and not os.path.isabs(str(image_path)):
                image_path = os.path.join(args.image_root, str(image_path))

            if not os.path.exists(image_path):
                failed_rows.append({
                    "sample_idx": idx,
                    "reason": "image not found",
                    "image_path": image_path,
                    "item": item,
                })
                continue

            obj1, obj2 = get_objects(item, str(question))

            if obj1 is None or obj2 is None:
                failed_rows.append({
                    "sample_idx": idx,
                    "reason": "cannot infer object_1/object_2",
                    "question": question,
                    "item": item,
                })
                continue

            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            dets = match_detections_for_image(
                det_map,
                image_path=image_rel,
                image_id=sample_id,
            )

            det1 = find_best_detection(dets, obj1)
            det2 = find_best_detection(dets, obj2)

            if det1 is None or det2 is None:
                failed_rows.append({
                    "sample_idx": idx,
                    "reason": "bbox not found",
                    "image_path": image_path,
                    "object_1": obj1,
                    "object_2": obj2,
                    "num_detections": len(dets),
                    "question": question,
                })
                continue

            bbox1_norm = normalize_bbox_1000(det1["bbox"], width, height)
            bbox2_norm = normalize_bbox_1000(det2["bbox"], width, height)

            region1 = bbox_to_region(bbox1_norm)
            region2 = bbox_to_region(bbox2_norm)

            prompt = build_bbox_prompt(
                original_question=str(question),
                obj1_name=str(obj1),
                obj1_bbox_norm=bbox1_norm,
                obj1_region=region1,
                obj2_name=str(obj2),
                obj2_bbox_norm=bbox2_norm,
                obj2_region=region2,
                question_mode=args.question_mode,
            )

            if idx < args.print_first_n:
                block = (
                    f"\n================ SAMPLE {idx} ================\n"
                    f"sample_id: {sample_id}\n"
                    f"image: {image_path}\n"
                    f"original question:\n{question}\n\n"
                    f"bbox prompt:\n{prompt}\n"
                )
                print(block)
                f_preview.write(block + "\n")

            try:
                inputs = build_model_inputs(
                    model_id=args.model_id,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                )

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_attentions=True,
                        use_cache=False,
                    )

                attentions = outputs.attentions
                if attentions is None:
                    raise RuntimeError(
                        "outputs.attentions is None. "
                        "Make sure attn_implementation='eager' is supported by this model."
                    )

                input_ids = inputs["input_ids"]
                first_attn = attentions[0]
                attn_len = first_attn.shape[-1]

                visual_positions, text_positions = infer_visual_and_text_positions(
                    input_ids=input_ids,
                    attn_len=attn_len,
                    image_token_ids=image_token_ids,
                )

                visual_set = set(visual_positions)
                text_set = set(text_positions)
                all_set = set(range(attn_len))
                other_positions = sorted(list(all_set - visual_set - text_set))

                n_visual = len(visual_positions)
                n_text = len(text_positions)

                grid_shape = get_visual_grid_shape(inputs, n_visual)

                for layer_idx, attn in enumerate(attentions):
                    # attn: [batch, heads, query_len, key_len]
                    attn_last_query = attn[0, :, -1, :].float().detach().cpu()
                    attn_mean = attn_last_query.mean(dim=0).numpy()

                    visual_mass = float(attn_mean[visual_positions].sum())
                    text_mass = float(attn_mean[text_positions].sum())
                    other_mass = float(attn_mean[other_positions].sum()) if other_positions else 0.0

                    writer.writerow({
                        "sample_idx": idx,
                        "sample_id": sample_id,
                        "image_path": image_path,
                        "object_1": obj1,
                        "object_2": obj2,
                        "layer": layer_idx,
                        "text_attention_mass": text_mass,
                        "visual_attention_mass": visual_mass,
                        "other_attention_mass": other_mass,
                        "num_text_tokens": n_text,
                        "num_visual_tokens": n_visual,
                        "question": str(question).replace("\n", "\\n"),
                        "prompt": prompt.replace("\n", "\\n"),
                    })

                    # 随机 5 张图：只保存前两层的 visual-token heatmap
                    if idx in heatmap_indices and layer_idx in [0, 1]:
                        visual_values = attn_mean[visual_positions]

                        save_name = (
                            f"sample_{idx:06d}_id_{str(sample_id).replace('/', '_')}"
                            f"_layer_{layer_idx}_last_query_visual_attn.png"
                        )
                        save_path = heatmap_dir / save_name

                        save_attention_heatmap(
                            image=image,
                            values_1d=visual_values,
                            grid_shape=grid_shape,
                            save_path=str(save_path),
                            title=f"sample {idx}, layer {layer_idx}",
                        )

            except Exception as e:
                failed_rows.append({
                    "sample_idx": idx,
                    "reason": f"forward/attention error: {repr(e)}",
                    "image_path": image_path,
                    "object_1": obj1,
                    "object_2": obj2,
                    "question": question,
                    "prompt": prompt,
                })
                continue

    save_jsonl(failed_rows, str(failed_path))

    print("\nDone.")
    print(f"Saved attention mass: {mass_csv_path}")
    print(f"Saved prompt preview: {prompt_preview_path}")
    print(f"Saved heatmaps: {heatmap_dir}")
    print(f"Saved failed samples: {failed_path}")


if __name__ == "__main__":
    main()
