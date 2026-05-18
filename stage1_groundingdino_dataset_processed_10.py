import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor

try:
    from transformers import AutoModelForZeroShotObjectDetection as GroundingDINOModel
except ImportError:
    from transformers import GroundingDinoForObjectDetection as GroundingDINOModel

from dataset_zoo import get_dataset


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "a272c74"


# ============================================================
# Prompt / dataset helpers
# ============================================================

def load_prompt_rows(dataset_name: str, option: str) -> List[dict]:
    path = Path(f"prompts/{dataset_name}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_question(q: str) -> str:
    q = str(q)
    q = q.replace("<image>", " ")
    q = q.replace("USER:", " ")
    q = q.replace("ASSISTANT:", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def strip_article(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[.?!,:;]+$", "", x)
    return x.strip()


def parse_two_objects_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Expected examples:
        Where is the beer bottle in relation to the armchair?
        Where is beer bottle in relation to armchair?
    """
    q = clean_question(prompt)

    # Remove answer instruction.
    q = re.sub(r"Answer\s+with\s+.*$", "", q, flags=re.IGNORECASE).strip()

    patterns = [
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]

    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            obj1 = strip_article(m.group(1))
            obj2 = strip_article(m.group(2))
            return obj1, obj2

    return None, None


def get_raw_pil_from_dataset(dataset, idx: int) -> Image.Image:
    """
    AdaptVis / LLaVA path usually returns:
        item["image_options"][0] as PIL image
    when image_preprocess=None.
    """
    item = dataset[idx]

    if isinstance(item, dict):
        if "image_options" in item:
            image = item["image_options"][0]
        elif "image" in item:
            image = item["image"]
        else:
            raise KeyError(f"Cannot find image in dataset item keys: {list(item.keys())}")
    else:
        raise TypeError(f"Expected dataset item to be dict, got {type(item)}")

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)}")

    return image.convert("RGB")


def get_gold_from_prompt_row(row: dict) -> str:
    ans = row.get("answer", "")
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return str(ans)


# ============================================================
# LLaVA preprocess geometry
# ============================================================

def get_size_value(size_obj, key: str, default: int) -> int:
    if isinstance(size_obj, dict):
        return int(size_obj.get(key, default))
    if isinstance(size_obj, int):
        return int(size_obj)
    return int(default)


def get_resample_from_processor(image_processor):
    resample = getattr(image_processor, "resample", None)
    if resample is not None:
        return resample
    return Image.BICUBIC


def infer_llava_geometry(image_processor, fallback_size: int = 336) -> Dict:
    """
    Infer geometric preprocessing from HF LLaVA image processor.

    For llava-hf/llava-1.5-7b-hf, this is usually:
        resize shortest edge to 336
        center crop 336 x 336
    """
    do_resize = bool(getattr(image_processor, "do_resize", True))
    do_center_crop = bool(getattr(image_processor, "do_center_crop", True))
    do_pad = bool(getattr(image_processor, "do_pad", False))

    size = getattr(image_processor, "size", None) or {}
    crop_size = getattr(image_processor, "crop_size", None) or {}

    shortest_edge = None
    resize_h = None
    resize_w = None

    if isinstance(size, dict):
        if "shortest_edge" in size:
            shortest_edge = int(size["shortest_edge"])
        if "height" in size and "width" in size:
            resize_h = int(size["height"])
            resize_w = int(size["width"])
    elif isinstance(size, int):
        shortest_edge = int(size)

    crop_h = get_size_value(crop_size, "height", fallback_size)
    crop_w = get_size_value(crop_size, "width", fallback_size)

    if shortest_edge is None and resize_h is None:
        shortest_edge = fallback_size

    return {
        "do_resize": do_resize,
        "do_center_crop": do_center_crop,
        "do_pad": do_pad,
        "shortest_edge": shortest_edge,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "resample": get_resample_from_processor(image_processor),
    }


def expand2square(img: Image.Image, background_color: Tuple[int, int, int]):
    w, h = img.size
    if w == h:
        return img, 0, 0, w

    size = max(w, h)
    out = Image.new("RGB", (size, size), background_color)

    pad_x = (size - w) // 2
    pad_y = (size - h) // 2

    out.paste(img, (pad_x, pad_y))
    return out, pad_x, pad_y, size


def make_processed_pil_like_llava(
    raw: Image.Image,
    image_processor,
    force_mode: str = "auto",
) -> Tuple[Image.Image, Dict]:
    """
    Create the final LLaVA-geometry image before normalization.

    force_mode:
        auto : infer from HF processor
        crop : resize shortest edge + center crop
        pad  : square pad + resize
    """
    raw = raw.convert("RGB")
    geom = infer_llava_geometry(image_processor)

    if force_mode not in ["auto", "crop", "pad"]:
        raise ValueError(f"Unknown force_mode={force_mode}")

    mode = force_mode
    if mode == "auto":
        mode = "pad" if geom["do_pad"] else "crop"

    if mode == "pad":
        mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        bg = tuple(int(float(x) * 255) for x in mean)

        square, pad_x, pad_y, square_size = expand2square(raw, bg)

        target_h = geom["crop_h"]
        target_w = geom["crop_w"]
        processed = square.resize((target_w, target_h), geom["resample"])

        meta = {
            "mode": "pad",
            "raw_size": raw.size,
            "square_size": square_size,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "processed_size": processed.size,
            "grid": target_h // 14,
        }
        return processed, meta

    # crop mode
    w, h = raw.size

    if geom["resize_h"] is not None and geom["resize_w"] is not None:
        resized = raw.resize((geom["resize_w"], geom["resize_h"]), geom["resample"])
    else:
        shortest = geom["shortest_edge"]
        scale = shortest / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = raw.resize((new_w, new_h), geom["resample"])

    rw, rh = resized.size
    crop_w = geom["crop_w"]
    crop_h = geom["crop_h"]

    left = max(0, int(round((rw - crop_w) / 2)))
    top = max(0, int(round((rh - crop_h) / 2)))

    processed = resized.crop((left, top, left + crop_w, top + crop_h))

    meta = {
        "mode": "crop",
        "raw_size": raw.size,
        "resized_size": resized.size,
        "crop_left": left,
        "crop_top": top,
        "processed_size": processed.size,
        "grid": crop_h // 14,
    }
    return processed, meta


# ============================================================
# GroundingDINO
# ============================================================

def detect_one(
    image: Image.Image,
    phrase: str,
    processor,
    model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[Optional[List[float]], Optional[float]]:
    text = phrase.strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    try:
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        # Some transformers versions use a slightly different signature.
        result = processor.post_process_grounded_object_detection(
            outputs,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    if len(boxes) == 0:
        return None, None

    best = int(torch.argmax(scores).item())
    box = boxes[best].detach().cpu().tolist()
    score = float(scores[best].detach().cpu().item())

    return box, score


# ============================================================
# Visualization
# ============================================================

def box_to_patch_range(
    box: Optional[List[float]],
    image_size: int = 336,
    patch_size: int = 14,
) -> Tuple[List[int], Optional[Tuple[int, int, int, int]]]:
    if box is None:
        return [], None

    x1, y1, x2, y2 = box
    grid = image_size // patch_size

    c1 = max(0, min(grid - 1, int(math.floor(x1 / patch_size))))
    r1 = max(0, min(grid - 1, int(math.floor(y1 / patch_size))))
    c2 = max(0, min(grid - 1, int(math.ceil(x2 / patch_size)) - 1))
    r2 = max(0, min(grid - 1, int(math.ceil(y2 / patch_size)) - 1))

    ids = []
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            ids.append(r * grid + c)

    return ids, (r1, c1, r2, c2)


def draw_grid(draw: ImageDraw.ImageDraw, size: int, grid: int):
    step = size / grid
    for i in range(1, grid):
        x = int(round(i * step))
        y = int(round(i * step))
        draw.line([(x, 0), (x, size)], fill=(200, 200, 200), width=1)
        draw.line([(0, y), (size, y)], fill=(200, 200, 200), width=1)


def safe_text(s: str, max_len: int = 90) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def draw_box_and_label(
    draw: ImageDraw.ImageDraw,
    box: Optional[List[float]],
    label: str,
    score: Optional[float],
    color: Tuple[int, int, int],
    font,
):
    if box is None:
        return

    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    text = f"{safe_text(label, 28)} {score:.2f}" if score is not None else f"{safe_text(label, 28)} NOT FOUND"

    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]

    x_text = max(0, int(x1))
    y_text = max(0, int(y1) - th - 8)

    draw.rectangle(
        [x_text, y_text, x_text + tw + 8, y_text + th + 6],
        fill=(255, 255, 255),
    )
    draw.text((x_text + 4, y_text + 3), text, fill=color, font=font)


def draw_info_panel(
    img: Image.Image,
    sample_id: int,
    question: str,
    gold: str,
    obj1: str,
    obj2: str,
    patch_info1: str,
    patch_info2: str,
) -> Image.Image:
    w, h = img.size
    panel_h = 118
    out = Image.new("RGB", (w, h + panel_h), (255, 255, 255))
    out.paste(img, (0, 0))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    lines = [
        f"sample_id={sample_id} | gold={gold}",
        f"obj1={obj1} | {patch_info1}",
        f"obj2={obj2} | {patch_info2}",
        safe_text(clean_question(question), 100),
    ]

    y = h + 6
    for line in lines:
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 26

    return out


def make_vis_image(
    processed: Image.Image,
    sample_id: int,
    question: str,
    gold: str,
    obj1: str,
    obj2: str,
    box1: Optional[List[float]],
    score1: Optional[float],
    box2: Optional[List[float]],
    score2: Optional[float],
    patch_size: int = 14,
) -> Image.Image:
    img = processed.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    size = img.size[0]
    grid = size // patch_size

    draw_grid(draw, size=size, grid=grid)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    ids1, grid_box1 = box_to_patch_range(box1, image_size=size, patch_size=patch_size)
    ids2, grid_box2 = box_to_patch_range(box2, image_size=size, patch_size=patch_size)

    draw_box_and_label(draw, box1, obj1, score1, color=(255, 0, 0), font=font)
    draw_box_and_label(draw, box2, obj2, score2, color=(0, 80, 255), font=font)

    patch_info1 = f"grid_box={grid_box1}, num_patches={len(ids1)}"
    patch_info2 = f"grid_box={grid_box2}, num_patches={len(ids2)}"

    return draw_info_panel(
        img,
        sample_id=sample_id,
        question=question,
        gold=gold,
        obj1=obj1,
        obj2=obj2,
        patch_info1=patch_info1,
        patch_info2=patch_info2,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--option", default="four")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llava-model-id", default=LLAVA_MODEL_ID)
    parser.add_argument("--llava-revision", default=LLAVA_REVISION)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--patch-size", type=int, default=14)

    # auto usually becomes center-crop for llava-hf/llava-1.5-7b-hf.
    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "crop", "pad"])

    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"output/stage1_groundingdino_{args.dataset}_processed_10")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep only this run's saved images.
    for p in out_dir.glob("*.png"):
        p.unlink()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] device={device}")
    print(f"[INFO] output dir={out_dir}")

    print("[INFO] loading LLaVA AutoProcessor only")
    llava_processor = AutoProcessor.from_pretrained(
        args.llava_model_id,
        revision=args.llava_revision,
        cache_dir=args.root_dir,
    )
    llava_image_processor = llava_processor.image_processor

    geom = infer_llava_geometry(llava_image_processor)
    print("[INFO] inferred LLaVA image geometry:")
    for k, v in geom.items():
        if k != "resample":
            print(f"  {k}: {v}")
    print("[INFO] preprocess mode:", args.preprocess_mode)

    print("[INFO] loading dataset via dataset_zoo.get_dataset")
    dataset = get_dataset(
        args.dataset,
        image_preprocess=None,
        download=args.download,
    )

    prompt_rows = load_prompt_rows(args.dataset, args.option)

    if len(prompt_rows) != len(dataset):
        print(
            f"[WARN] prompt rows ({len(prompt_rows)}) != dataset length ({len(dataset)}). "
            f"Will use min length."
        )

    n_total = min(len(dataset), len(prompt_rows))

    valid = []
    for idx in range(n_total):
        prompt = prompt_rows[idx].get("question", "")
        obj1, obj2 = parse_two_objects_from_prompt(prompt)
        if obj1 and obj2:
            valid.append(idx)

    if len(valid) < args.num_samples:
        raise RuntimeError(
            f"Only found {len(valid)} samples with two parsed objects, "
            f"but num_samples={args.num_samples}."
        )

    random.seed(args.seed)
    sampled_ids = random.sample(valid, args.num_samples)

    print(f"[INFO] parsed-valid samples={len(valid)}")
    print(f"[INFO] sampled ids={sampled_ids}")

    print(f"[INFO] loading GroundingDINO: {args.grounding_model_id}")
    gdino_processor = AutoProcessor.from_pretrained(args.grounding_model_id)
    gdino_model = GroundingDINOModel.from_pretrained(args.grounding_model_id).to(device).eval()

    saved = 0

    for out_idx, sample_id in enumerate(sampled_ids):
        prompt = prompt_rows[sample_id].get("question", "")
        gold = get_gold_from_prompt_row(prompt_rows[sample_id])
        obj1, obj2 = parse_two_objects_from_prompt(prompt)

        raw = get_raw_pil_from_dataset(dataset, sample_id)

        processed, meta = make_processed_pil_like_llava(
            raw=raw,
            image_processor=llava_image_processor,
            force_mode=args.preprocess_mode,
        )

        # GroundingDINO runs directly on the final LLaVA-processed image.
        # Therefore boxes are already aligned with the LLaVA feature map.
        box1, score1 = detect_one(
            image=processed,
            phrase=obj1,
            processor=gdino_processor,
            model=gdino_model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        box2, score2 = detect_one(
            image=processed,
            phrase=obj2,
            processor=gdino_processor,
            model=gdino_model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        vis = make_vis_image(
            processed=processed,
            sample_id=sample_id,
            question=prompt,
            gold=gold,
            obj1=obj1,
            obj2=obj2,
            box1=box1,
            score1=score1,
            box2=box2,
            score2=score2,
            patch_size=args.patch_size,
        )

        safe_obj1 = re.sub(r"[^a-zA-Z0-9_-]+", "_", obj1)[:35]
        safe_obj2 = re.sub(r"[^a-zA-Z0-9_-]+", "_", obj2)[:35]
        out_path = out_dir / f"{out_idx:02d}_sid{sample_id}_{safe_obj1}__{safe_obj2}.png"

        vis.save(out_path)
        saved += 1

        ids1, grid_box1 = box_to_patch_range(
            box1,
            image_size=processed.size[0],
            patch_size=args.patch_size,
        )
        ids2, grid_box2 = box_to_patch_range(
            box2,
            image_size=processed.size[0],
            patch_size=args.patch_size,
        )

        print("\n" + "=" * 80)
        print(f"[{out_idx + 1}/{args.num_samples}] sample_id={sample_id}")
        print("gold:", gold)
        print("question:", clean_question(prompt))
        print("processed meta:", meta)
        print(
            "obj1:",
            obj1,
            "score:",
            score1,
            "box:",
            box1,
            "grid_box:",
            grid_box1,
            "patch_count:",
            len(ids1),
        )
        print(
            "obj2:",
            obj2,
            "score:",
            score2,
            "box:",
            box2,
            "grid_box:",
            grid_box2,
            "patch_count:",
            len(ids2),
        )
        print("[SAVE]", out_path)

    print("\n[DONE]")
    print(f"saved {saved} processed images to {out_dir}")
    print("Each image is the final LLaVA-preprocessed image with two GroundingDINO boxes and a 24x24 grid.")


if __name__ == "__main__":
    main()
