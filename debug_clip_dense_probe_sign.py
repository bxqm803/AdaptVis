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
from PIL import Image, ImageDraw

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
    p.add_argument("--out-dir", default="output_clip_dense_probe_debug", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=10, type=int)
    p.add_argument("--topk", default=10, type=int)
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


def parse_bbox_string(x: Any) -> Optional[List[int]]:
    x = clean_text(x)
    if not x:
        return None
    try:
        val = json.loads(x)
    except Exception:
        return None
    if not isinstance(val, list) or len(val) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in val]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


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
                "object_1_bbox": parse_bbox_string(row.get("object_1_bbox", "")),
                "object_2_bbox": parse_bbox_string(row.get("object_2_bbox", "")),
            })
    return rows


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
    x = pixel_values[0].detach().cpu().permute(1, 2, 0).numpy()

    image_mean = getattr(processor.image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    image_std = getattr(processor.image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)

    x = x * std + mean
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)

    return Image.fromarray(x)


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def map_bbox_to_clip_view(bbox, orig_w, orig_h, clip_processor):
    """
    尽量按 CLIPImageProcessor 的常见逻辑：
    resize shortest edge -> center crop
    """
    if bbox is None:
        return None

    img_proc = clip_processor.image_processor

    size = getattr(img_proc, "size", {})
    crop_size = getattr(img_proc, "crop_size", {})
    do_resize = bool(getattr(img_proc, "do_resize", True))
    do_center_crop = bool(getattr(img_proc, "do_center_crop", True))

    if isinstance(size, dict):
        shortest_edge = int(size.get("shortest_edge", size.get("height", 336)))
    else:
        shortest_edge = int(size)

    if isinstance(crop_size, dict):
        crop_h = int(crop_size.get("height", 336))
        crop_w = int(crop_size.get("width", 336))
    else:
        crop_h = int(crop_size)
        crop_w = int(crop_size)

    x1, y1, x2, y2 = bbox
    rw, rh = orig_w, orig_h

    if do_resize:
        scale = float(shortest_edge) / float(min(orig_w, orig_h))
        x1 = int(round(x1 * scale))
        y1 = int(round(y1 * scale))
        x2 = int(round(x2 * scale))
        y2 = int(round(y2 * scale))
        rw = int(round(orig_w * scale))
        rh = int(round(orig_h * scale))

    if do_center_crop:
        left = max(0, int(round((rw - crop_w) / 2.0)))
        top = max(0, int(round((rh - crop_h) / 2.0)))
        x1 -= left
        x2 -= left
        y1 -= top
        y2 -= top
        return clip_box([x1, y1, x2, y2], crop_w, crop_h)

    return clip_box([x1, y1, x2, y2], rw, rh)


@torch.no_grad()
def compute_patch_text_scores(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image_pil: Image.Image,
    text_phrase: str,
    device: str,
):
    image_inputs = clip_processor(images=image_pil, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    text_inputs = clip_processor(text=[text_phrase], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    text_features = clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 与你原脚本一致
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
    last_hidden = vision_outputs.last_hidden_state  # [1, 1+N, H]

    if hasattr(clip_model.vision_model, "vision_model") and hasattr(clip_model.vision_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.vision_model.post_layernorm(last_hidden)
    elif hasattr(clip_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.post_layernorm(last_hidden)

    patch_tokens = last_hidden[:, 1:, :]
    patch_proj = clip_model.visual_projection(patch_tokens)
    patch_proj = patch_proj / patch_proj.norm(dim=-1, keepdim=True)

    sims = torch.matmul(patch_proj[0], text_features[0])
    sims = sims.detach().float().cpu().numpy()

    gh, gw = infer_patch_grid(len(sims))
    heatmap = sims.reshape(gh, gw)
    heatmap_norm = normalize_map(heatmap)

    clip_view_pil = pixel_values_to_pil(pixel_values, clip_processor)

    return {
        "similarity_vector": sims,
        "heatmap_raw": heatmap,
        "heatmap_norm": heatmap_norm,
        "grid_h": gh,
        "grid_w": gw,
        "clip_view_pil": clip_view_pil,
    }


def make_patch_boxes(img_w: int, img_h: int, gh: int, gw: int) -> List[List[int]]:
    boxes = []
    for r in range(gh):
        for c in range(gw):
            x1 = int(round(c * img_w / gw))
            y1 = int(round(r * img_h / gh))
            x2 = int(round((c + 1) * img_w / gw))
            y2 = int(round((r + 1) * img_h / gh))
            boxes.append([x1, y1, x2, y2])
    return boxes


def box_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def mask_patches_inside_bbox(patch_boxes, bbox):
    if bbox is None:
        return np.zeros(len(patch_boxes), dtype=bool)

    mask = []
    for pb in patch_boxes:
        inter = box_intersection_area(pb, bbox)
        area = (pb[2] - pb[0]) * (pb[3] - pb[1])
        mask.append(inter > 0 and (inter / max(area, 1)) > 0.15)
    return np.array(mask, dtype=bool)


def draw_patch_boxes_on_image(image_pil, patch_boxes, selected_indices, color="red", width=3):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for idx in selected_indices:
        x1, y1, x2, y2 = patch_boxes[idx]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def save_debug_figure(
    save_path: str,
    clip_view_image: Image.Image,
    bbox_proc,
    top_img: Image.Image,
    bottom_img: Image.Image,
    heatmap_norm: np.ndarray,
    obj_name: str,
):
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(clip_view_image)
    if bbox_proc is not None:
        x1, y1, x2, y2 = bbox_proc
        ax1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
    ax1.set_title(f"CLIP view + GT bbox: {obj_name}")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(top_img)
    ax2.set_title("Top-k highest-score patches")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(bottom_img)
    ax3.set_title("Bottom-k lowest-score patches")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(heatmap_norm, cmap="jet")
    ax4.set_title("Normalized heatmap (red=high)")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(1.0 - heatmap_norm, cmap="jet")
    ax5.set_title("Inverted heatmap (red=low)")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(clip_view_image)
    ax6.imshow(heatmap_norm, cmap="jet", alpha=0.45, interpolation="bilinear",
               extent=(0, clip_view_image.size[0], clip_view_image.size[1], 0))
    ax6.set_title("Overlay (red=high)")
    ax6.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_one_object(
    image_pil: Image.Image,
    obj_name: str,
    bbox_orig,
    clip_model,
    clip_processor,
    device: str,
    topk: int,
):
    out = compute_patch_text_scores(
        clip_model=clip_model,
        clip_processor=clip_processor,
        image_pil=image_pil,
        text_phrase=obj_name,
        device=device,
    )

    clip_view = out["clip_view_pil"]
    gh, gw = out["grid_h"], out["grid_w"]
    sims = out["similarity_vector"]
    heatmap_norm = out["heatmap_norm"]

    bbox_proc = map_bbox_to_clip_view(bbox_orig, image_pil.size[0], image_pil.size[1], clip_processor)
    patch_boxes = make_patch_boxes(clip_view.size[0], clip_view.size[1], gh, gw)
    in_mask = mask_patches_inside_bbox(patch_boxes, bbox_proc)

    if in_mask.sum() == 0:
        mean_in = None
        mean_out = None
        percentile_mean_in = None
    else:
        mean_in = float(np.mean(sims[in_mask]))
        mean_out = float(np.mean(sims[~in_mask])) if (~in_mask).sum() > 0 else None

        order = np.argsort(np.argsort(sims))  # rank
        percentiles = order / max(len(sims) - 1, 1)
        percentile_mean_in = float(np.mean(percentiles[in_mask]))

    top_idx = np.argsort(-sims)[:topk]
    bottom_idx = np.argsort(sims)[:topk]

    top_overlap = float(np.mean(in_mask[top_idx])) if len(top_idx) > 0 else 0.0
    bottom_overlap = float(np.mean(in_mask[bottom_idx])) if len(bottom_idx) > 0 else 0.0

    top_img = draw_patch_boxes_on_image(clip_view, patch_boxes, top_idx, color="red", width=3)
    bottom_img = draw_patch_boxes_on_image(clip_view, patch_boxes, bottom_idx, color="blue", width=3)

    return {
        "obj_name": obj_name,
        "bbox_orig": bbox_orig,
        "bbox_proc": bbox_proc,
        "grid_h": gh,
        "grid_w": gw,
        "mean_in_bbox": mean_in,
        "mean_out_bbox": mean_out,
        "mean_percentile_in_bbox": percentile_mean_in,
        "topk_overlap_ratio": top_overlap,
        "bottomk_overlap_ratio": bottom_overlap,
        "clip_view": clip_view,
        "heatmap_norm": heatmap_norm,
        "top_img": top_img,
        "bottom_img": bottom_img,
        "sims_min": float(np.min(sims)),
        "sims_max": float(np.max(sims)),
        "sims_mean": float(np.mean(sims)),
    }


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

    summary_csv = os.path.join(out_root, "summary_clip_dense_probe_debug.csv")
    summary_rows = []

    for row in manual_rows:
        image_name = row["image_name"]
        if image_name not in dataset_index:
            print(f"[Skip] image not found in dataset: {image_name}")
            continue

        idx = dataset_index[image_name]
        item = dataset[idx]
        image = item["image_options"][0].convert("RGB")

        obj1 = normalize_object_name(row["object_1_name"])
        obj2 = normalize_object_name(row["object_2_name"])
        bbox1 = row["object_1_bbox"]
        bbox2 = row["object_2_bbox"]

        stem = os.path.splitext(os.path.basename(image_name))[0]
        save_dir = os.path.join(out_root, stem)
        os.makedirs(save_dir, exist_ok=True)

        print("=" * 100)
        print("image_name:", image_name)
        print("obj1:", obj1, "bbox1:", bbox1)
        print("obj2:", obj2, "bbox2:", bbox2)

        try:
            ana1 = analyze_one_object(
                image_pil=image,
                obj_name=obj1,
                bbox_orig=bbox1,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=args.device,
                topk=args.topk,
            )
            ana2 = analyze_one_object(
                image_pil=image,
                obj_name=obj2,
                bbox_orig=bbox2,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=args.device,
                topk=args.topk,
            )
        except Exception as e:
            print(f"[Error] {image_name}: {e}")
            summary_rows.append({
                "image_name": image_name,
                "local_index": idx,
                "status": "error",
                "error": str(e),
            })
            continue

        save_debug_figure(
            save_path=os.path.join(save_dir, "obj1_debug.png"),
            clip_view_image=ana1["clip_view"],
            bbox_proc=ana1["bbox_proc"],
            top_img=ana1["top_img"],
            bottom_img=ana1["bottom_img"],
            heatmap_norm=ana1["heatmap_norm"],
            obj_name=obj1,
        )
        save_debug_figure(
            save_path=os.path.join(save_dir, "obj2_debug.png"),
            clip_view_image=ana2["clip_view"],
            bbox_proc=ana2["bbox_proc"],
            top_img=ana2["top_img"],
            bottom_img=ana2["bottom_img"],
            heatmap_norm=ana2["heatmap_norm"],
            obj_name=obj2,
        )

        print(f"[{obj1}] mean_in={ana1['mean_in_bbox']}, mean_out={ana1['mean_out_bbox']}, "
              f"top_overlap={ana1['topk_overlap_ratio']:.3f}, bottom_overlap={ana1['bottomk_overlap_ratio']:.3f}")
        print(f"[{obj2}] mean_in={ana2['mean_in_bbox']}, mean_out={ana2['mean_out_bbox']}, "
              f"top_overlap={ana2['topk_overlap_ratio']:.3f}, bottom_overlap={ana2['bottomk_overlap_ratio']:.3f}")

        summary_rows.append({
            "image_name": image_name,
            "local_index": idx,
            "obj1": obj1,
            "obj1_mean_in_bbox": ana1["mean_in_bbox"],
            "obj1_mean_out_bbox": ana1["mean_out_bbox"],
            "obj1_mean_percentile_in_bbox": ana1["mean_percentile_in_bbox"],
            "obj1_topk_overlap_ratio": ana1["topk_overlap_ratio"],
            "obj1_bottomk_overlap_ratio": ana1["bottomk_overlap_ratio"],
            "obj2": obj2,
            "obj2_mean_in_bbox": ana2["mean_in_bbox"],
            "obj2_mean_out_bbox": ana2["mean_out_bbox"],
            "obj2_mean_percentile_in_bbox": ana2["mean_percentile_in_bbox"],
            "obj2_topk_overlap_ratio": ana2["topk_overlap_ratio"],
            "obj2_bottomk_overlap_ratio": ana2["bottomk_overlap_ratio"],
            "obj1_debug": os.path.relpath(os.path.join(save_dir, "obj1_debug.png"), out_root),
            "obj2_debug": os.path.relpath(os.path.join(save_dir, "obj2_debug.png"), out_root),
            "status": "ok",
            "error": "",
        })

        fieldnames = [
            "image_name",
            "local_index",
            "obj1",
            "obj1_mean_in_bbox",
            "obj1_mean_out_bbox",
            "obj1_mean_percentile_in_bbox",
            "obj1_topk_overlap_ratio",
            "obj1_bottomk_overlap_ratio",
            "obj2",
            "obj2_mean_in_bbox",
            "obj2_mean_out_bbox",
            "obj2_mean_percentile_in_bbox",
            "obj2_topk_overlap_ratio",
            "obj2_bottomk_overlap_ratio",
            "obj1_debug",
            "obj2_debug",
            "status",
            "error",
        ]
        with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("=" * 100)
    print(f"Saved to: {out_root}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
