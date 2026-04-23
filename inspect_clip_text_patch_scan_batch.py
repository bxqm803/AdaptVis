import os
import re
import csv
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from transformers import CLIPModel, CLIPProcessor

from misc import seed_all
from dataset_zoo import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manual-bbox-csv", required=True, type=str)
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14-336", type=str)
    p.add_argument("--out-dir", default="output_clip_text_patch_scan", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=10, type=int)   # 默认前10张
    return p.parse_args()


def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_object_name(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"^(a|an|the)\s+", "", name)
    name = re.sub(r"[?.!,;:]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def infer_names_from_filename(image_name: str) -> Tuple[str, str]:
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    for marker in ["_left_of_", "_right_of_", "_on_", "_under_"]:
        if marker in stem:
            a, b = stem.split(marker, 1)
            return normalize_object_name(a), normalize_object_name(b)
    return "object_1", "object_2"


def parse_objects_from_question(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = clean_text(question)
    patterns = [
        r"Where is the (.+?) in relation to the (.+?)\?",
        r"Where are the (.+?) in relation to the (.+?)\?",
        r"Where is (.+?) in relation to (.+?)\?",
        r"Where are (.+?) in relation to (.+?)\?",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            obj1 = normalize_object_name(m.group(1))
            obj2 = normalize_object_name(m.group(2))
            return obj1, obj2
    return None, None


def load_prompt_records(dataset_name: str, option: str) -> List[Dict[str, Any]]:
    prompt_path = os.path.join("prompts", f"{dataset_name}_with_answer_{option}_options.jsonl")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    records = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_manual_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = clean_text(row.get("image_name", ""))
            if not image_name:
                continue

            obj1_name = clean_text(row.get("object_1_name", ""))
            obj2_name = clean_text(row.get("object_2_name", ""))

            if not obj1_name or not obj2_name:
                fallback1, fallback2 = infer_names_from_filename(image_name)
                obj1_name = obj1_name or fallback1
                obj2_name = obj2_name or fallback2

            rows.append({
                "image_name": image_name,
                "image_path": clean_text(row.get("image_path", "")),
                "object_1_name": obj1_name,
                "object_2_name": obj2_name,
            })
    return rows


def build_dataset_index(dataset) -> Dict[str, int]:
    mapping = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        image_name = clean_text(item.get("image_name", f"sample_{idx:04d}"))
        mapping[image_name] = idx
    return mapping


def infer_patch_grid(num_patches: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(num_patches)))
    if side * side == num_patches:
        return side, side
    for h in range(int(math.sqrt(num_patches)), 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    return 1, num_patches


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    if mx > 1e-8:
        x = x / mx
    return x


def pixel_values_to_pil(pixel_values: torch.Tensor, processor: CLIPProcessor) -> Image.Image:
    """
    把 CLIP 预处理后的像素张量还原成 336x336 的可视化图
    """
    x = pixel_values[0].detach().cpu().permute(1, 2, 0).numpy()

    image_mean = getattr(processor.image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    image_std = getattr(processor.image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)

    x = x * std + mean
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)

    return Image.fromarray(x)


@torch.no_grad()
def compute_patch_text_heatmap(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image_pil: Image.Image,
    text_phrase: str,
    device: str,
):
    # image preprocess
    image_inputs = clip_processor(images=image_pil, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    # text preprocess
    text_inputs = clip_processor(text=[text_phrase], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # text embedding: pooled projected text feature
    text_features = clip_model.get_text_features(**text_inputs)  # [1, D]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # vision outputs
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
    last_hidden = vision_outputs.last_hidden_state  # [1, 1+N, H]

    # apply post_layernorm to all vision tokens if available
    if hasattr(clip_model.vision_model, "vision_model") and hasattr(clip_model.vision_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.vision_model.post_layernorm(last_hidden)
    elif hasattr(clip_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.post_layernorm(last_hidden)

    # drop CLS
    patch_tokens = last_hidden[:, 1:, :]  # [1, N, H]

    # project patches into CLIP shared embedding space
    patch_proj = clip_model.visual_projection(patch_tokens)  # [1, N, D]
    patch_proj = patch_proj / patch_proj.norm(dim=-1, keepdim=True)

    # cosine similarity with text feature
    sims = torch.matmul(patch_proj[0], text_features[0])  # [N]
    sims = sims.detach().float().cpu().numpy()

    gh, gw = infer_patch_grid(len(sims))
    heatmap = sims.reshape(gh, gw)
    heatmap = normalize_map(heatmap)

    clip_view_pil = pixel_values_to_pil(pixel_values, clip_processor)

    return {
        "heatmap": heatmap,
        "similarity_vector": sims,
        "grid_h": gh,
        "grid_w": gw,
        "clip_view_pil": clip_view_pil,
    }


def save_overlay_figure(
    save_path: str,
    original_image: Image.Image,
    clip_view_image: Image.Image,
    obj1: str,
    obj2: str,
    obj1_heatmap: np.ndarray,
    obj2_heatmap: np.ndarray,
):
    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(original_image)
    ax1.set_title("Original image", fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(clip_view_image)
    ax2.set_title("CLIP processed view", fontsize=12)
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(clip_view_image)
    ax3.imshow(obj1_heatmap, cmap="jet", alpha=0.45, interpolation="bilinear",
               extent=(0, clip_view_image.size[0], clip_view_image.size[1], 0))
    ax3.set_title(f"Overlay: {obj1}", fontsize=12)
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(obj1_heatmap, cmap="jet")
    ax4.set_title(f"Patch heatmap: {obj1}", fontsize=12)
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(clip_view_image)
    ax5.imshow(obj2_heatmap, cmap="jet", alpha=0.45, interpolation="bilinear",
               extent=(0, clip_view_image.size[0], clip_view_image.size[1], 0))
    ax5.set_title(f"Overlay: {obj2}", fontsize=12)
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(obj2_heatmap, cmap="jet")
    ax6.set_title(f"Patch heatmap: {obj2}", fontsize=12)
    ax6.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    seed_all(args.seed)

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    # same selected images as your manual CSV
    manual_rows = load_manual_rows(args.manual_bbox_csv)
    if len(manual_rows) == 0:
        raise ValueError(f"No usable rows found in: {args.manual_bbox_csv}")

    manual_rows = manual_rows[args.sample_index: args.sample_index + args.limit]

    prompt_records = load_prompt_records(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)

    if len(prompt_records) != len(dataset):
        raise ValueError(f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).")

    dataset_index = build_dataset_index(dataset)

    print(f"Loading CLIP model: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model, cache_dir=cache_dir).to(args.device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model, cache_dir=cache_dir)

    out_root = os.path.join(
        args.out_dir,
        args.dataset,
        os.path.basename(args.clip_model).replace("/", "_"),
    )
    os.makedirs(out_root, exist_ok=True)

    summary_csv = os.path.join(out_root, "summary_clip_text_patch_scan.csv")
    summary_rows = []

    used = 0
    for row in manual_rows:
        image_name = row["image_name"]
        if image_name not in dataset_index:
            print(f"[Skip] image not found in dataset: {image_name}")
            continue

        idx = dataset_index[image_name]
        item = dataset[idx]
        rec = prompt_records[idx]

        image = item["image_options"][0].convert("RGB")
        image_path = clean_text(item.get("image_path", ""))
        question = clean_text(rec["question"])

        obj1 = normalize_object_name(row["object_1_name"])
        obj2 = normalize_object_name(row["object_2_name"])

        if not obj1 or not obj2:
            auto1, auto2 = parse_objects_from_question(question)
            obj1 = obj1 or auto1
            obj2 = obj2 or auto2

        if not obj1 or not obj2:
            print(f"[Skip] failed to get obj names for: {image_name}")
            continue

        print("=" * 100)
        print(f"image_name: {image_name}")
        print(f"question: {question}")
        print(f"obj1: {obj1}")
        print(f"obj2: {obj2}")

        try:
            obj1_out = compute_patch_text_heatmap(
                clip_model=clip_model,
                clip_processor=clip_processor,
                image_pil=image,
                text_phrase=obj1,
                device=args.device,
            )
            obj2_out = compute_patch_text_heatmap(
                clip_model=clip_model,
                clip_processor=clip_processor,
                image_pil=image,
                text_phrase=obj2,
                device=args.device,
            )
        except Exception as e:
            print(f"[Error] {image_name}: {e}")
            summary_rows.append({
                "image_name": image_name,
                "image_path": image_path,
                "local_index": idx,
                "question": question,
                "obj1": obj1,
                "obj2": obj2,
                "status": "error",
                "error": str(e),
            })
            continue

        stem = os.path.splitext(os.path.basename(image_name))[0]
        save_dir = os.path.join(out_root, stem)
        os.makedirs(save_dir, exist_ok=True)

        overlay_path = os.path.join(save_dir, "clip_text_patch_scan.png")
        meta_json_path = os.path.join(save_dir, "meta.json")

        save_overlay_figure(
            save_path=overlay_path,
            original_image=image,
            clip_view_image=obj1_out["clip_view_pil"],  # obj1/obj2 预处理应一致
            obj1=obj1,
            obj2=obj2,
            obj1_heatmap=obj1_out["heatmap"],
            obj2_heatmap=obj2_out["heatmap"],
        )

        meta_payload = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": idx,
            "question": question,
            "obj1": obj1,
            "obj2": obj2,
            "clip_model": args.clip_model,
            "num_patches": int(len(obj1_out["similarity_vector"])),
            "grid_h": int(obj1_out["grid_h"]),
            "grid_w": int(obj1_out["grid_w"]),
            "overlay_path": os.path.relpath(overlay_path, out_root),
        }
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        summary_rows.append({
            "image_name": image_name,
            "image_path": image_path,
            "local_index": idx,
            "question": question,
            "obj1": obj1,
            "obj2": obj2,
            "clip_model": args.clip_model,
            "num_patches": int(len(obj1_out["similarity_vector"])),
            "grid_h": int(obj1_out["grid_h"]),
            "grid_w": int(obj1_out["grid_w"]),
            "overlay_path": os.path.relpath(overlay_path, out_root),
            "meta_json": os.path.relpath(meta_json_path, out_root),
            "status": "ok",
            "error": "",
        })

        fieldnames = [
            "image_name",
            "image_path",
            "local_index",
            "question",
            "obj1",
            "obj2",
            "clip_model",
            "num_patches",
            "grid_h",
            "grid_w",
            "overlay_path",
            "meta_json",
            "status",
            "error",
        ]
        with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        used += 1

    print("=" * 100)
    print(f"Finished. used={used}")
    print(f"Saved to: {out_root}")
    print(f"Summary:  {summary_csv}")


if __name__ == "__main__":
    main()
