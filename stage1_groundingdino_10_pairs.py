import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, GroundingDinoForObjectDetection


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_question(row):
    for k in ["question", "Question", "prompt", "Prompt"]:
        if k in row and row[k]:
            return str(row[k])

    # fallback: concatenate text-like fields
    vals = []
    for k, v in row.items():
        if isinstance(v, str) and ("Where is" in v or "relation to" in v):
            vals.append(v)
    return "\n".join(vals)


def clean_question(q):
    q = q.replace("<image>", " ")
    q = q.replace("USER:", " ")
    q = q.replace("ASSISTANT:", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def parse_two_objects(question):
    """
    Expected examples:
    Where is the beer bottle in relation to the armchair?
    Where is beer bottle in relation to armchair?
    """
    q = clean_question(question)

    # remove answer instruction
    q = re.sub(r"Answer\s+with\s+.*?$", "", q, flags=re.IGNORECASE).strip()

    patterns = [
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]

    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            obj1 = m.group(1).strip()
            obj2 = m.group(2).strip()
            obj1 = strip_article(obj1)
            obj2 = strip_article(obj2)
            return obj1, obj2

    return None, None


def strip_article(x):
    x = x.strip()
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[.?!,:;]+$", "", x)
    return x.strip()


def find_image_path(row, image_root):
    keys = [
        "image_path",
        "image",
        "img",
        "image_file",
        "image_filename",
        "file_name",
        "filename",
        "path",
    ]

    candidates = []
    for k in keys:
        if k in row and row[k]:
            candidates.append(str(row[k]))

    for c in candidates:
        p = Path(c)
        if p.is_absolute() and p.exists():
            return p

        if p.exists():
            return p

        p2 = Path(image_root) / c
        if p2.exists():
            return p2

        # common fallback: if row stores only basename
        p3 = Path(image_root) / Path(c).name
        if p3.exists():
            return p3

    return None


def detect_one_object(image, obj_name, processor, model, device, box_threshold, text_threshold):
    """
    Run GroundingDINO for one phrase and return top box.
    Returns:
        box: [x1, y1, x2, y2] or None
        score: float or None
    """
    text = obj_name.strip()
    if not text.endswith("."):
        text = text + "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])

    if len(boxes) == 0:
        return None, None

    best_idx = int(torch.argmax(scores).item())
    box = boxes[best_idx].detach().cpu().tolist()
    score = float(scores[best_idx].detach().cpu().item())

    return box, score


def crop_box(image, box, pad_ratio=0.08):
    w, h = image.size

    if box is None:
        return Image.new("RGB", (224, 224), color=(245, 245, 245))

    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad = max(bw, bh) * pad_ratio

    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad))
    y2 = min(h, int(y2 + pad))

    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (224, 224), color=(245, 245, 245))

    return image.crop((x1, y1, x2, y2)).convert("RGB")


def resize_keep_aspect(img, size=256):
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_label(img, label, score=None):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    if score is None:
        text = f"{label}: NOT FOUND"
    else:
        text = f"{label}: {score:.3f}"

    # default PIL font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(255, 255, 255))
    draw.text((pad, pad), text, fill=(0, 0, 0), font=font)

    return img


def make_pair_image(crop1, crop2, label1, label2, score1, score2, size=256):
    c1 = resize_keep_aspect(crop1, size=size)
    c2 = resize_keep_aspect(crop2, size=size)

    c1 = draw_label(c1, label1, score1)
    c2 = draw_label(c2, label2, score2)

    gap = 16
    out = Image.new("RGB", (size * 2 + gap, size), color=(240, 240, 240))
    out.paste(c1, (0, 0))
    out.paste(c2, (size + gap, 0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-file",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--image-root",
        default=".",
        help="Root directory used when image paths in jsonl are relative.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/stage1_groundingdino_10_pairs",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-base",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--crop-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear old png/jpg outputs so this folder contains only this run's 10 images.
    for p in list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.jpeg")):
        p.unlink()

    rows = load_jsonl(args.prompt_file)

    valid = []
    for idx, row in enumerate(rows):
        q = extract_question(row)
        obj1, obj2 = parse_two_objects(q)
        img_path = find_image_path(row, args.image_root)

        if obj1 and obj2 and img_path is not None:
            valid.append((idx, row, img_path, obj1, obj2, q))

    if len(valid) < args.num_samples:
        raise RuntimeError(
            f"Only found {len(valid)} valid samples, but num_samples={args.num_samples}. "
            f"Check --prompt-file and --image-root."
        )

    random.seed(args.seed)
    sampled = random.sample(valid, args.num_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] loading GroundingDINO: {args.model_id}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model_id).to(device)
    model.eval()

    saved = 0

    for out_idx, (sample_id, row, img_path, obj1, obj2, q) in enumerate(sampled):
        print("\n" + "=" * 80)
        print(f"[{out_idx+1}/{args.num_samples}] sample_id={sample_id}")
        print("image:", img_path)
        print("obj1:", obj1)
        print("obj2:", obj2)

        image = Image.open(img_path).convert("RGB")

        box1, score1 = detect_one_object(
            image=image,
            obj_name=obj1,
            processor=processor,
            model=model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        box2, score2 = detect_one_object(
            image=image,
            obj_name=obj2,
            processor=processor,
            model=model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        print("box1:", box1, "score1:", score1)
        print("box2:", box2, "score2:", score2)

        crop1 = crop_box(image, box1)
        crop2 = crop_box(image, box2)

        pair_img = make_pair_image(
            crop1=crop1,
            crop2=crop2,
            label1=obj1,
            label2=obj2,
            score1=score1,
            score2=score2,
            size=args.crop_size,
        )

        safe_obj1 = re.sub(r"[^a-zA-Z0-9_-]+", "_", obj1)[:40]
        safe_obj2 = re.sub(r"[^a-zA-Z0-9_-]+", "_", obj2)[:40]
        out_path = out_dir / f"{out_idx:02d}_sid{sample_id}_{safe_obj1}__{safe_obj2}.png"

        pair_img.save(out_path)
        print("[SAVE]", out_path)
        saved += 1

    print("\n[DONE]")
    print(f"saved {saved} pair images to: {out_dir}")
    print("Each output image contains two crops: left = first object, right = second object.")


if __name__ == "__main__":
    main()
