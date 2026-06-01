import os
import re
import csv
import json
import argparse
from PIL import Image
from tqdm import tqdm


def clean_obj(x):
    x = x.strip().strip(".")
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.I)
    return x.strip()


def parse_caption(caption):
    """
    Parse captions like:
      A photo of a bird to the right of a giraffe
      A photo of a bird below a giraffe
    Return:
      obj1, obj2, answer
    """
    s = caption.strip()
    s = re.sub(r"^A photo of\s+", "", s, flags=re.I).strip()

    patterns = [
        (r"(.+?)\s+to the left of\s+(.+)", "Left"),
        (r"(.+?)\s+to the right of\s+(.+)", "Right"),
        (r"(.+?)\s+above\s+(.+)", "Above"),
        (r"(.+?)\s+below\s+(.+)", "Below"),
    ]

    for pat, ans in patterns:
        m = re.match(pat, s, flags=re.I)
        if m:
            obj1 = clean_obj(m.group(1))
            obj2 = clean_obj(m.group(2))
            return obj1, obj2, ans

    return "", "", ""


def make_prompt(obj1, obj2):
    if obj1 and obj2:
        return (
            "<image>\n"
            f"USER: Where is the {obj1} in relation to the {obj2}? "
            "Answer with left, right, above, or below.\n"
            "ASSISTANT:"
        )
    return ""


def preprocess_like_llava_clip(image, image_size=336, mode="pad", pad_color=(0, 0, 0)):
    """
    Save the visible RGB processed image, without normalization.

    mode="pad":
      preserve the full COCO image, pad to square, then resize to 336x336.
      This is recommended for COCO_QA_two_obj because objects are often near
      image borders and center-crop can remove them.

    mode="crop":
      resize shortest edge to image_size, then center crop image_size x image_size.
      This matches CLIP default geometry, but may cut off COCO objects.

    mode="resize":
      directly resize to image_size x image_size. This does not preserve aspect ratio.
    """
    image = image.convert("RGB")
    w, h = image.size

    if mode == "pad":
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), pad_color)
        left = (side - w) // 2
        top = (side - h) // 2
        canvas.paste(image, (left, top))
        return canvas.resize((image_size, image_size), Image.BICUBIC)

    if mode == "crop":
        scale = float(image_size) / float(min(w, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        left = max(0, (new_w - image_size) // 2)
        top = max(0, (new_h - image_size) // 2)
        return resized.crop((left, top, left + image_size, top + image_size))

    if mode == "resize":
        return image.resize((image_size, image_size), Image.BICUBIC)

    raise ValueError(f"Unknown preprocess mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", default="data/coco_qa_two_obj.json")
    parser.add_argument("--image-dir", default="data/val2017")
    parser.add_argument("--out-dir", default="output/coco_qa_two_obj_processed_images_pad")
    parser.add_argument("--out-json", default="output/coco_qa_two_obj_processed_manifest_pad.json")
    parser.add_argument("--out-csv", default="output/coco_qa_two_obj_processed_manifest_pad.csv")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--preprocess-mode", choices=["pad", "crop", "resize"], default="pad")
    parser.add_argument("--fresh-limit", type=int, default=-1)
    parser.add_argument("--idxs", default="", help="Optional comma-separated sample idxs, e.g. 0,5,8")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    data = json.load(open(args.qa_json, "r", encoding="utf-8"))

    if args.idxs.strip():
        idxs = [int(x) for x in args.idxs.split(",") if x.strip()]
    else:
        idxs = list(range(len(data)))
        if args.fresh_limit > 0:
            idxs = idxs[:args.fresh_limit]

    manifest = {}
    csv_rows = []

    for idx in tqdm(idxs, desc="dump COCO-QA processed images"):
        rec = data[idx]

        image_id = int(rec[0])
        positive_caption = rec[1]
        negative_caption = rec[2]

        obj1, obj2, gold = parse_caption(positive_caption)
        prompt = make_prompt(obj1, obj2)

        raw_image_path = os.path.join(args.image_dir, f"{image_id:012d}.jpg")
        if not os.path.exists(raw_image_path):
            print("[WARN] missing image:", raw_image_path)
            continue

        raw_image = Image.open(raw_image_path).convert("RGB")
        processed_image = preprocess_like_llava_clip(raw_image, image_size=args.image_size, mode=args.preprocess_mode)

        out_name = f"idx{idx:04d}_img{image_id:012d}.png"
        processed_image_path = os.path.join(args.out_dir, out_name)
        processed_image.save(processed_image_path)

        item = {
            "sample_idx": int(idx),
            "image_id": int(image_id),
            "raw_image_path": raw_image_path,
            "processed_image_path": processed_image_path,
            "positive_caption": positive_caption,
            "negative_caption": negative_caption,
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,
            "prompt": prompt,
            "processed_width": int(processed_image.size[0]),
            "processed_height": int(processed_image.size[1]),
            "coordinate_system": f"processed_image_{args.preprocess_mode}_{args.image_size}x{args.image_size}",
            "preprocess_mode": args.preprocess_mode,
        }

        manifest[str(idx)] = item

        csv_rows.append({
            "sample_idx": idx,
            "image_id": image_id,
            "raw_image_path": raw_image_path,
            "processed_image_path": processed_image_path,
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,
            "prompt": prompt,
            "positive_caption": positive_caption,
            "negative_caption": negative_caption,
            "preprocess_mode": args.preprocess_mode,
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_idx",
            "image_id",
            "raw_image_path",
            "processed_image_path",
            "obj1",
            "obj2",
            "gold",
            "prompt",
            "positive_caption",
            "negative_caption",
            "preprocess_mode",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print("[DONE]")
    print("[OUT DIR]", args.out_dir)
    print("[JSON]", args.out_json)
    print("[CSV]", args.out_csv)
    print("num saved:", len(csv_rows))


if __name__ == "__main__":
    main()
