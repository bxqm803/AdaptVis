import os
import json
import argparse
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def box_to_patch_ids(box, image_size=336, patch_side=24):
    if box is None:
        return []

    x1, y1, x2, y2 = [float(v) for v in box]
    patch = image_size / patch_side

    ids = []
    for r in range(patch_side):
        for c in range(patch_side):
            px1 = c * patch
            py1 = r * patch
            px2 = (c + 1) * patch
            py2 = (r + 1) * patch

            if px2 <= x1 or px1 >= x2 or py2 <= y1 or py1 >= y2:
                continue
            ids.append(r * patch_side + c)

    return ids


def norm_box(box, image_size=336):
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size]


def valid_box(box):
    if box is None:
        return False
    x1, y1, x2, y2 = [float(v) for v in box]
    return x2 > x1 and y2 > y1


def run_one_object(processor, model, image, text, device, box_threshold, text_threshold):
    text_prompt = text.strip()
    if not text_prompt.endswith("."):
        text_prompt += "."

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

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
    labels = results.get("labels", [])

    dets = []
    for i in range(len(boxes)):
        box = boxes[i].detach().float().cpu().tolist()
        if not valid_box(box):
            continue

        score = float(scores[i].detach().float().cpu()) if hasattr(scores[i], "detach") else float(scores[i])
        label = labels[i] if i < len(labels) else text

        dets.append({
            "target": text,
            "label": str(label),
            "score": score,
            "box_xyxy_processed": box,
            "box_norm": norm_box(box, image_size=image.size[0]),
            "patch_ids": box_to_patch_ids(box, image_size=image.size[0], patch_side=24),
        })

    dets = sorted(dets, key=lambda x: x["score"], reverse=True)
    return dets


def draw_examples(image, rec, out_path):
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)

    colors = {
        "obj1": "red",
        "obj2": "blue",
    }

    for key in ["obj1", "obj2"]:
        box = rec.get(f"{key}_box_xyxy_processed")
        if box is None:
            continue

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=colors[key], width=4)
        draw.text((x1 + 3, y1 + 3), f"{key}: {rec.get(key)}", fill=colors[key])

    im.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", default="output/controlledA_model_processed_manifest_crop.json")
    parser.add_argument("--out-json", default="output/controlledA_groundingdino_boxes_crop.json")
    parser.add_argument("--example-dir", default="output/controlledA_groundingdino_boxes_crop_examples")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-boxes-per-object", type=int, default=1)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--fresh-limit", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.example_dir, exist_ok=True)

    manifest = json.load(open(args.manifest_json, "r", encoding="utf-8"))

    print("[LOAD GROUNDINGDINO]", args.model)
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, cache_dir=args.cache_dir).to(args.device)
    model.eval()

    out = {}

    keys = sorted(manifest.keys(), key=lambda x: int(x))
    if args.fresh_limit > 0:
        keys = keys[:args.fresh_limit]

    for k in tqdm(keys, desc="GroundingDINO on crop processed images"):
        m = manifest[k]
        sid = int(m["sample_id"])
        img_path = m["processed_image_path"]
        image = Image.open(img_path).convert("RGB")

        obj1 = m.get("obj1", "")
        obj2 = m.get("obj2", "")

        obj1_dets = run_one_object(
            processor, model, image, obj1, args.device,
            args.box_threshold, args.text_threshold,
        )[:args.max_boxes_per_object]

        obj2_dets = run_one_object(
            processor, model, image, obj2, args.device,
            args.box_threshold, args.text_threshold,
        )[:args.max_boxes_per_object]

        obj1_top = obj1_dets[0] if obj1_dets else None
        obj2_top = obj2_dets[0] if obj2_dets else None

        patch_ids = []
        for d in obj1_dets + obj2_dets:
            patch_ids.extend(d.get("patch_ids", []))
        patch_ids = sorted(set(int(x) for x in patch_ids))

        rec = {
            "sample_id": sid,
            "processed_image_path": img_path,
            "obj1": obj1,
            "obj2": obj2,
            "gold": m.get("gold", ""),
            "prompt": m.get("prompt", ""),
            "obj1_detections": obj1_dets,
            "obj2_detections": obj2_dets,
            "obj1_box_xyxy_processed": obj1_top["box_xyxy_processed"] if obj1_top else None,
            "obj2_box_xyxy_processed": obj2_top["box_xyxy_processed"] if obj2_top else None,
            "obj1_box_norm": obj1_top["box_norm"] if obj1_top else None,
            "obj2_box_norm": obj2_top["box_norm"] if obj2_top else None,
            "obj1_patch_ids": obj1_top["patch_ids"] if obj1_top else [],
            "obj2_patch_ids": obj2_top["patch_ids"] if obj2_top else [],
            "patch_ids": patch_ids,
            "coordinate_system": m.get("coordinate_system", "model_processed_image_crop"),
        }

        out[str(sid)] = rec

        if sid < args.num_examples:
            ex_path = os.path.join(args.example_dir, f"idx{sid:04d}_bbox.png")
            draw_examples(image, rec, ex_path)

            if sid < 5:
                print("\n[CHECK]", sid)
                print("obj1:", obj1, "box:", rec["obj1_box_xyxy_processed"], "score:", obj1_top["score"] if obj1_top else None)
                print("obj2:", obj2, "box:", rec["obj2_box_xyxy_processed"], "score:", obj2_top["score"] if obj2_top else None)
                print("example:", ex_path)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    both = sum(
        1 for r in out.values()
        if r["obj1_box_xyxy_processed"] is not None and r["obj2_box_xyxy_processed"] is not None
    )

    print("[DONE]")
    print("[OUT JSON]", args.out_json)
    print("[EXAMPLE DIR]", args.example_dir)
    print("num:", len(out))
    print("both boxes found:", both, "/", len(out))


if __name__ == "__main__":
    main()
