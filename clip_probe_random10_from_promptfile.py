import os
import re
import json
import math
import random
import argparse
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor


QUESTION_RE = re.compile(
    r"Where\s+is\s+(?:the\s+)?(.+?)\s+in\s+relation\s+to\s+(?:the\s+)?(.+?)\?\s*"
    r"Answer\s+with\s+left,\s*right,\s*on\s+or\s+under\.?",
    re.IGNORECASE,
)


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def clean_obj_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_objects_from_question(question: str) -> Optional[Tuple[str, str]]:
    m = QUESTION_RE.search(question)
    if m is None:
        return None

    obj1 = clean_obj_name(m.group(1))
    obj2 = clean_obj_name(m.group(2))
    return obj1, obj2


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_image_path(image_path: str, annotation_path: str) -> Optional[str]:
    if image_path is None:
        return None

    candidates = [
        image_path,
        os.path.join(os.getcwd(), image_path),
        os.path.join(os.path.dirname(annotation_path), image_path),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


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


@torch.no_grad()
def get_text_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    obj_name: str,
    device: str,
) -> torch.Tensor:
    prompts = build_text_prompts(obj_name)

    text_inputs = processor.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_outputs = model.text_model(**text_inputs)
    pooled = text_outputs.pooler_output
    text_embeds = model.text_projection(pooled)
    text_embeds = normalize(text_embeds, dim=-1)

    text_embed = text_embeds.mean(dim=0, keepdim=True)
    text_embed = normalize(text_embed, dim=-1)
    return text_embed


@torch.no_grad()
def get_patch_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str,
) -> Tuple[torch.Tensor, int]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    vision_outputs = model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=False,
        return_dict=True,
    )

    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
    patch_embeds = model.visual_projection(patch_tokens)
    patch_embeds = normalize(patch_embeds, dim=-1)

    n_patches = patch_embeds.shape[1]
    grid_size = int(math.sqrt(n_patches))

    if grid_size * grid_size != n_patches:
        raise ValueError(f"Patch number {n_patches} is not square.")

    return patch_embeds, grid_size


@torch.no_grad()
def compute_similarity_map(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    obj_name: str,
    device: str,
) -> np.ndarray:
    patch_embeds, grid_size = get_patch_embeddings(model, processor, image, device)
    text_embed = get_text_embedding(model, processor, obj_name, device)

    sim = torch.matmul(patch_embeds, text_embed.T).squeeze(-1)
    sim = sim[0]

    sim_map = sim.view(grid_size, grid_size).detach().cpu().numpy()
    return sim_map


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
):
    image_np = np.array(image).astype(np.float32) / 255.0
    h, w = image_np.shape[:2]

    heatmap_t = torch.tensor(heatmap, dtype=torch.float32)[None, None, :, :]
    heatmap_up = F.interpolate(
        heatmap_t,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    heatmap_up = heatmap_up - heatmap_up.min()
    heatmap_up = heatmap_up / (heatmap_up.max() + 1e-8)

    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap_up)[..., :3]

    overlay = (1.0 - alpha) * image_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    return overlay, heatmap_up


def save_triptych(
    image: Image.Image,
    question: str,
    obj1: str,
    obj2: str,
    heatmap1: np.ndarray,
    heatmap2: np.ndarray,
    out_path: str,
):
    overlay1, _ = overlay_heatmap_on_image(image, heatmap1)
    overlay2, _ = overlay_heatmap_on_image(image, heatmap2)

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
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/clip_random10_attention_maps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Loading annotation: {args.annotation}")
    with open(args.annotation, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"[INFO] Loading prompt file: {args.prompt_file}")
    prompt_rows = load_jsonl(args.prompt_file)

    print(f"[INFO] Number of images: {len(annotations)}")
    print(f"[INFO] Number of prompts: {len(prompt_rows)}")

    if len(annotations) != len(prompt_rows):
        print("[WARN] annotation and prompt file have different lengths.")
        print("[WARN] Will only use min length.")

    n = min(len(annotations), len(prompt_rows))

    valid_indices = []
    debug_bad = []

    for i in range(n):
        ann = annotations[i]
        prompt_row = prompt_rows[i]

        image_path = resolve_image_path(ann.get("image_path"), args.annotation)
        question = prompt_row.get("question", None)
        answer = prompt_row.get("answer", None)

        objs = extract_objects_from_question(question) if question is not None else None

        if image_path is None or question is None or objs is None:
            debug_bad.append({
                "index": i,
                "image_path": ann.get("image_path"),
                "resolved_image_path": image_path,
                "question": question,
                "answer": answer,
            })
            continue

        valid_indices.append(i)

    print(f"[INFO] Valid samples: {len(valid_indices)}")

    if len(valid_indices) == 0:
        print("\n[DEBUG] First 5 failed samples:")
        for item in debug_bad[:5]:
            print(json.dumps(item, indent=2, ensure_ascii=False))
        raise RuntimeError("No valid prompt/image pairs found.")

    sample_count = min(args.num_samples, len(valid_indices))
    selected_indices = random.sample(valid_indices, sample_count)

    print(f"[INFO] Random selected indices: {selected_indices}")
    print(f"[INFO] Loading CLIP model: {args.clip_model}")

    model = CLIPModel.from_pretrained(args.clip_model).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    metadata_path = os.path.join(args.out_dir, "metadata.jsonl")

    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        for out_i, data_i in enumerate(selected_indices):
            ann = annotations[data_i]
            prompt_row = prompt_rows[data_i]

            image_path = resolve_image_path(ann.get("image_path"), args.annotation)
            question = prompt_row["question"]
            answer = prompt_row.get("answer", None)

            obj1, obj2 = extract_objects_from_question(question)

            print(f"\n[{out_i + 1}/{sample_count}] dataset index = {data_i}")
            print(f"Image: {image_path}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"obj1 = {obj1}")
            print(f"obj2 = {obj2}")

            image = Image.open(image_path).convert("RGB")

            sim_map1 = compute_similarity_map(
                model=model,
                processor=processor,
                image=image,
                obj_name=obj1,
                device=args.device,
            )

            sim_map2 = compute_similarity_map(
                model=model,
                processor=processor,
                image=image,
                obj_name=obj2,
                device=args.device,
            )

            base_name = (
                f"{out_i:02d}_idx{data_i}_"
                f"{sanitize_filename(obj1)}_vs_{sanitize_filename(obj2)}"
            )

            fig_path = os.path.join(args.out_dir, f"{base_name}.png")
            obj1_npy = os.path.join(args.out_dir, f"{base_name}_{sanitize_filename(obj1)}.npy")
            obj2_npy = os.path.join(args.out_dir, f"{base_name}_{sanitize_filename(obj2)}.npy")

            save_triptych(
                image=image,
                question=question,
                obj1=obj1,
                obj2=obj2,
                heatmap1=sim_map1,
                heatmap2=sim_map2,
                out_path=fig_path,
            )

            np.save(obj1_npy, sim_map1)
            np.save(obj2_npy, sim_map2)

            record = {
                "sample_id": out_i,
                "dataset_index": data_i,
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "obj1": obj1,
                "obj2": obj2,
                "figure_path": fig_path,
                "obj1_sim_map": obj1_npy,
                "obj2_sim_map": obj2_npy,
            }

            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Done.")
    print(f"[INFO] Saved figures to: {args.out_dir}")
    print(f"[INFO] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
