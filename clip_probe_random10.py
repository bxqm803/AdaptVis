import os
import re
import json
import math
import random
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor


QUESTION_PATTERNS = [
    re.compile(
        r"Where is the (.+?) in relation to (?:the )?(.+?)\?\s*Answer with left,\s*right,\s*on or under\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Where is (.+?) in relation to (?:the )?(.+?)\?\s*Answer with left,\s*right,\s*on or under\.?",
        re.IGNORECASE,
    ),
]


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s/]+", "_", s)
    return s[:80]


def build_text_prompts(obj_name: str) -> List[str]:
    obj_name = obj_name.strip()
    return [
        obj_name,
        f"a photo of a {obj_name}",
        f"an image of a {obj_name}",
        f"the {obj_name}",
    ]


def extract_question(entry: dict) -> Optional[str]:
    captions = entry.get("caption_options", None)
    if captions is None:
        return None

    if isinstance(captions, str):
        candidates = [captions]
    elif isinstance(captions, list):
        candidates = [c for c in captions if isinstance(c, str)]
    else:
        return None

    for q in candidates:
        for pat in QUESTION_PATTERNS:
            if pat.search(q):
                return q

    # 如果没有精确匹配，就返回第一个字符串，后面再尝试 parse
    return candidates[0] if len(candidates) > 0 else None


def extract_objects(question: str) -> Optional[Tuple[str, str]]:
    if question is None:
        return None

    for pat in QUESTION_PATTERNS:
        m = pat.search(question)
        if m:
            obj1 = m.group(1).strip()
            obj2 = m.group(2).strip()
            return obj1, obj2
    return None


@torch.no_grad()
def get_text_embedding(model: CLIPModel, processor: CLIPProcessor, obj_name: str, device: str) -> torch.Tensor:
    prompts = build_text_prompts(obj_name)
    text_inputs = processor.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_outputs = model.text_model(**text_inputs)
    pooled = text_outputs.pooler_output                      # [num_prompts, hidden_dim]
    text_embeds = model.text_projection(pooled)             # [num_prompts, proj_dim]
    text_embeds = normalize(text_embeds, dim=-1)

    text_embed = text_embeds.mean(dim=0, keepdim=True)      # [1, proj_dim]
    text_embed = normalize(text_embed, dim=-1)
    return text_embed


@torch.no_grad()
def get_patch_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str
) -> Tuple[torch.Tensor, int]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    vision_outputs = model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=False,
        return_dict=True,
    )

    # [1, 1+N, hidden_dim], 去掉 CLS
    last_hidden = vision_outputs.last_hidden_state[:, 1:, :]
    patch_embeds = model.visual_projection(last_hidden)     # [1, N, proj_dim]
    patch_embeds = normalize(patch_embeds, dim=-1)

    n_patches = patch_embeds.shape[1]
    grid_size = int(math.sqrt(n_patches))
    if grid_size * grid_size != n_patches:
        raise ValueError(f"Patch number {n_patches} is not a square.")

    return patch_embeds, grid_size


@torch.no_grad()
def compute_similarity_map(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    obj_name: str,
    device: str
) -> np.ndarray:
    patch_embeds, grid_size = get_patch_embeddings(model, processor, image, device)
    text_embed = get_text_embedding(model, processor, obj_name, device)  # [1, D]

    sim = torch.matmul(patch_embeds, text_embed.T).squeeze(-1)            # [1, N]
    sim = sim[0]                                                          # [N]
    sim_map = sim.view(grid_size, grid_size).detach().cpu().numpy()
    return sim_map


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    image_np = np.array(image).astype(np.float32) / 255.0
    h, w = image_np.shape[:2]

    heatmap_t = torch.tensor(heatmap, dtype=torch.float32)[None, None, :, :]
    heatmap_up = F.interpolate(
        heatmap_t,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()

    # normalize to [0,1]
    heatmap_up = heatmap_up - heatmap_up.min()
    denom = heatmap_up.max() + 1e-8
    heatmap_up = heatmap_up / denom

    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap_up)[..., :3]

    overlay = (1 - alpha) * image_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay, heatmap_up


def save_triptych(
    image: Image.Image,
    question: str,
    obj1: str,
    obj2: str,
    heatmap1: np.ndarray,
    heatmap2: np.ndarray,
    out_path: str
):
    overlay1, _ = overlay_heatmap_on_image(image, heatmap1, alpha=0.45)
    overlay2, _ = overlay_heatmap_on_image(image, heatmap2, alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay1)
    axes[1].set_title(f"CLIP patch heatmap: {obj1}")
    axes[1].axis("off")

    axes[2].imshow(overlay2)
    axes[2].set_title(f"CLIP patch heatmap: {obj2}")
    axes[2].axis("off")

    fig.suptitle(question, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        type=str,
        default="data/controlled_images_dataset.json",
        help="Path to Controlled_Images_A annotation json"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="Standalone CLIP model; recommended to match LLaVA family"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/clip_random10_attention_maps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Loading annotations: {args.annotation}")
    with open(args.annotation, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 过滤出能成功解析 question / obj1 / obj2 的样本
    valid_entries = []
    for entry in data:
        q = extract_question(entry)
        if q is None:
            continue
        objs = extract_objects(q)
        if objs is None:
            continue
        img_path = entry.get("image_path", None)
        if img_path is None or (not os.path.exists(img_path)):
            continue
        valid_entries.append((entry, q, objs[0], objs[1]))

    if len(valid_entries) == 0:
        raise RuntimeError("No valid samples found. Check annotation path or question format.")

    sample_count = min(args.num_samples, len(valid_entries))
    selected = random.sample(valid_entries, sample_count)

    print(f"[INFO] Found {len(valid_entries)} valid samples. Randomly selected {sample_count}.")
    print(f"[INFO] Loading CLIP model: {args.clip_model}")
    model = CLIPModel.from_pretrained(args.clip_model).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    metadata_path = os.path.join(args.out_dir, "metadata.jsonl")
    meta_f = open(metadata_path, "w", encoding="utf-8")

    for idx, (entry, question, obj1, obj2) in enumerate(selected):
        image_path = entry["image_path"]
        print(f"\n[{idx+1}/{sample_count}]")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"obj1 = {obj1}")
        print(f"obj2 = {obj2}")

        try:
            image = Image.open(image_path).convert("RGB")
            sim_map1 = compute_similarity_map(model, processor, image, obj1, args.device)
            sim_map2 = compute_similarity_map(model, processor, image, obj2, args.device)

            base_name = f"{idx:02d}_{sanitize_filename(obj1)}_vs_{sanitize_filename(obj2)}"
            out_path = os.path.join(args.out_dir, f"{base_name}.png")

            save_triptych(
                image=image,
                question=question,
                obj1=obj1,
                obj2=obj2,
                heatmap1=sim_map1,
                heatmap2=sim_map2,
                out_path=out_path,
            )

            # 同时保存 raw 数值，方便后面做 top-k patch / mask
            np.save(os.path.join(args.out_dir, f"{base_name}_{sanitize_filename(obj1)}.npy"), sim_map1)
            np.save(os.path.join(args.out_dir, f"{base_name}_{sanitize_filename(obj2)}.npy"), sim_map2)

            record = {
                "index": idx,
                "image_path": image_path,
                "question": question,
                "obj1": obj1,
                "obj2": obj2,
                "figure_path": out_path,
                "obj1_npy": os.path.join(args.out_dir, f"{base_name}_{sanitize_filename(obj1)}.npy"),
                "obj2_npy": os.path.join(args.out_dir, f"{base_name}_{sanitize_filename(obj2)}.npy"),
            }
            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[WARN] Failed on sample {idx}: {e}")

    meta_f.close()
    print(f"\n[INFO] Done. Saved all outputs to: {args.out_dir}")
    print(f"[INFO] Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
