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


def unnormalize_clip_image(pixel_values: torch.Tensor, processor: CLIPProcessor) -> np.ndarray:
    """
    pixel_values: [3, H, W], after CLIPProcessor normalization.
    return: [H, W, 3], range [0, 1].
    """
    image_processor = processor.image_processor

    mean = torch.tensor(
        image_processor.image_mean,
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        image_processor.image_std,
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    ).view(3, 1, 1)

    img = pixel_values * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    return img


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
def get_patch_embeddings_and_image(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str,
    use_post_layernorm: bool = True,
):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    vision_outputs = model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=False,
        return_dict=True,
    )

    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]

    if use_post_layernorm and hasattr(model.vision_model, "post_layernorm"):
        patch_tokens = model.vision_model.post_layernorm(patch_tokens)

    patch_embeds = model.visual_projection(patch_tokens)
    patch_embeds = normalize(patch_embeds, dim=-1)

    n_patches = patch_embeds.shape[1]
    grid_size = int(math.sqrt(n_patches))

    if grid_size * grid_size != n_patches:
        raise ValueError(f"Patch number {n_patches} is not square.")

    proc_image_np = unnormalize_clip_image(pixel_values[0], processor)

    return patch_embeds, grid_size, proc_image_np


@torch.no_grad()
def compute_sim_map(
    model: CLIPModel,
    processor: CLIPProcessor,
    patch_embeds: torch.Tensor,
    grid_size: int,
    obj_name: str,
    device: str,
) -> np.ndarray:
    text_embed = get_text_embedding(model, processor, obj_name, device)
    sim = torch.matmul(patch_embeds, text_embed.T).squeeze(-1)[0]
    sim_map = sim.view(grid_size, grid_size).detach().cpu().numpy()
    return sim_map


def make_score_map(sim_map: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    If invert=True, low CLIP similarity becomes high score.
    """
    if invert:
        return -sim_map
    return sim_map


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def overlay_heatmap_on_np_image(
    image_np: np.ndarray,
    score_map: np.ndarray,
    alpha: float = 0.45,
):
    h, w = image_np.shape[:2]

    heatmap_t = torch.tensor(score_map, dtype=torch.float32)[None, None, :, :]
    heatmap_up = F.interpolate(
        heatmap_t,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    heatmap_up = normalize_for_display(heatmap_up)

    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap_up)[..., :3]

    overlay = (1.0 - alpha) * image_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    return overlay, heatmap_up


def get_topk_patches(score_map: np.ndarray, k: int = 10):
    flat = score_map.reshape(-1)
    idxs = np.argsort(flat)[::-1][:k]

    h, w = score_map.shape
    results = []

    for rank, idx in enumerate(idxs, start=1):
        row = int(idx // w)
        col = int(idx % w)
        score = float(flat[idx])
        results.append((rank, row, col, int(idx), score))

    return results


def draw_topk_boxes(
    ax,
    score_map: np.ndarray,
    image_shape,
    k: int = 10,
):
    h, w = image_shape[:2]
    grid_h, grid_w = score_map.shape

    patch_h = h / grid_h
    patch_w = w / grid_w

    topk = get_topk_patches(score_map, k=k)

    for rank, row, col, patch_idx, score in topk:
        x = col * patch_w
        y = row * patch_h

        rect = plt.Rectangle(
            (x, y),
            patch_w,
            patch_h,
            fill=False,
            linewidth=1.8,
            edgecolor="black",
        )
        ax.add_patch(rect)

        ax.text(
            x,
            y,
            str(rank),
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    return topk


def save_one_figure(
    proc_image_np: np.ndarray,
    question: str,
    answer: str,
    obj1: str,
    obj2: str,
    score_map1: np.ndarray,
    score_map2: np.ndarray,
    out_path: str,
    topk: int = 10,
    invert: bool = True,
):
    overlay1, _ = overlay_heatmap_on_np_image(proc_image_np, score_map1)
    overlay2, _ = overlay_heatmap_on_np_image(proc_image_np, score_map2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(proc_image_np)
    axes[0].set_title("CLIP preprocessed image")
    axes[0].axis("off")

    axes[1].imshow(overlay1)
    title1 = f"{obj1} | {'inverted' if invert else 'normal'} score"
    axes[1].set_title(title1)
    axes[1].axis("off")
    topk1 = draw_topk_boxes(axes[1], score_map1, proc_image_np.shape, k=topk)

    axes[2].imshow(overlay2)
    title2 = f"{obj2} | {'inverted' if invert else 'normal'} score"
    axes[2].set_title(title2)
    axes[2].axis("off")
    topk2 = draw_topk_boxes(axes[2], score_map2, proc_image_np.shape, k=topk)

    fig.suptitle(f"{question}\nanswer: {answer}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return topk1, topk2


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
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument(
        "--invert",
        action="store_true",
        help="Use negative CLIP similarity as score. Recommended for this test.",
    )
    parser.add_argument(
        "--no_post_layernorm",
        action="store_true",
        help="Disable post_layernorm on patch tokens.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/clip_inverse_random10",
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

    n = min(len(annotations), len(prompt_rows))
    print(f"[INFO] Number of usable pairs: {n}")

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

    selected_indices = random.sample(valid_indices, min(args.num_samples, len(valid_indices)))
    print(f"[INFO] Selected indices: {selected_indices}")

    print(f"[INFO] Loading CLIP model: {args.clip_model}")
    model = CLIPModel.from_pretrained(args.clip_model).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    metadata = []

    for out_i, data_i in enumerate(selected_indices):
        ann = annotations[data_i]
        prompt_row = prompt_rows[data_i]

        image_path = resolve_image_path(ann.get("image_path"), args.annotation)
        question = prompt_row["question"]
        answer = prompt_row.get("answer", None)
        obj1, obj2 = extract_objects_from_question(question)

        print(f"\n[{out_i + 1}/{len(selected_indices)}] dataset index = {data_i}")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"obj1 = {obj1}")
        print(f"obj2 = {obj2}")

        image = Image.open(image_path).convert("RGB")

        patch_embeds, grid_size, proc_image_np = get_patch_embeddings_and_image(
            model=model,
            processor=processor,
            image=image,
            device=args.device,
            use_post_layernorm=not args.no_post_layernorm,
        )

        sim_map1 = compute_sim_map(
            model=model,
            processor=processor,
            patch_embeds=patch_embeds,
            grid_size=grid_size,
            obj_name=obj1,
            device=args.device,
        )

        sim_map2 = compute_sim_map(
            model=model,
            processor=processor,
            patch_embeds=patch_embeds,
            grid_size=grid_size,
            obj_name=obj2,
            device=args.device,
        )

        score_map1 = make_score_map(sim_map1, invert=args.invert)
        score_map2 = make_score_map(sim_map2, invert=args.invert)

        base_name = (
            f"{out_i:02d}_idx{data_i}_"
            f"{sanitize_filename(obj1)}_vs_{sanitize_filename(obj2)}"
        )

        fig_path = os.path.join(args.out_dir, f"{base_name}.png")

        topk1, topk2 = save_one_figure(
            proc_image_np=proc_image_np,
            question=question,
            answer=answer,
            obj1=obj1,
            obj2=obj2,
            score_map1=score_map1,
            score_map2=score_map2,
            out_path=fig_path,
            topk=args.topk,
            invert=args.invert,
        )

        metadata.append({
            "sample_id": out_i,
            "dataset_index": data_i,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "obj1": obj1,
            "obj2": obj2,
            "grid_size": grid_size,
            "invert": args.invert,
            "figure_path": fig_path,
            "obj1_topk": topk1,
            "obj2_topk": topk2,
        })

    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Done.")
    print(f"[INFO] Saved only {len(selected_indices)} figures to: {args.out_dir}")
    print(f"[INFO] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
