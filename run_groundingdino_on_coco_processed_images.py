import os
import json
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection


def detect_one(processor, model, image, text, device, box_threshold, text_threshold, max_boxes):
    text = text.strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(
        images=image,
        text=text.lower(),
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    labels = results.get("labels", [text] * len(boxes))

    if len(boxes) == 0:
        return []

    scores_tensor = scores if torch.is_tensor(scores) else torch.tensor(scores)
    order = torch.argsort(scores_tensor, descending=True)

    if max_boxes > 0:
        order = order[:max_boxes]

    dets = []
    w, h = image.size

    for idx in order.detach().cpu().tolist():
        box = boxes[idx].detach().cpu().tolist()
        score = scores[idx]
        label = labels[idx]

        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(w, x1))
        x2 = max(0.0, min(w, x2))
        y1 = max(0.0, min(h, y1))
        y2 = max(0.0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        dets.append({
            "target": text.strip("."),
            "label": str(label),
            "score": float(score.detach().cpu() if torch.is_tensor(score) else score),
            "box_xyxy_processed": [x1, y1, x2, y2],
        })

    return dets


def draw_overlay(image, detections, out_path):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 0, 0),
        (0, 120, 255),
        (255, 180, 0),
        (0, 180, 80),
    ]

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["box_xyxy_processed"]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        label = f'{det["target"]} {det["score"]:.2f}'
        tx, ty = x1, max(0, y1 - 16)

        draw.rectangle(
            [tx, ty, tx + max(80, 8 * len(label)), ty + 16],
            fill=color,
        )
        draw.text((tx + 2, ty + 1), label, fill=(255, 255, 255))

    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", default="output/coco_qa_two_obj_processed_manifest.json")
    parser.add_argument("--out-json", default="output/coco_qa_two_obj_processed_groundingdino_boxes.json")
    parser.add_argument("--example-dir", default="output/coco_qa_two_obj_processed_groundingdino_examples")
    parser.add_argument("--example-idxs", default="0,5,8")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-boxes-per-object", type=int, default=1)
    parser.add_argument("--fresh-limit", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.example_dir, exist_ok=True)

    manifest = json.load(open(args.manifest_json, "r", encoding="utf-8"))

    example_idxs = set()
    if args.example_idxs.strip():
        example_idxs = {int(x) for x in args.example_idxs.split(",") if x.strip()}

    keys = sorted(manifest.keys(), key=lambda x: int(x))
    if args.fresh_limit > 0:
        keys = keys[:args.fresh_limit]

    print("[LOAD GroundingDINO]", args.model)
    processor = GroundingDinoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model, cache_dir=args.cache_dir).to(args.device)
    model.eval()

    results = {}

    for k in tqdm(keys, desc="GroundingDINO processed images"):
        item = manifest[k]
        idx = int(item["sample_idx"])

        img_path = item["processed_image_path"]
        obj1 = item.get("obj1", "")
        obj2 = item.get("obj2", "")

        if not os.path.exists(img_path):
            print("[WARN] missing processed image:", img_path)
            continue

        image = Image.open(img_path).convert("RGB")

        detections = []
        for obj in [obj1, obj2]:
            if not obj:
                continue

            dets = detect_one(
                processor=processor,
                model=model,
                image=image,
                text=obj,
                device=args.device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                max_boxes=args.max_boxes_per_object,
            )
            detections.extend(dets)

        rec = dict(item)
        rec["detections"] = detections
        rec["coordinate_system"] = "processed_image_336x336"
        results[str(idx)] = rec

        if idx in example_idxs:
            out_name = f"idx{idx:04d}_img{int(item['image_id']):012d}_bbox_overlay.png"
            out_path = os.path.join(args.example_dir, out_name)
            draw_overlay(image, detections, out_path)
            results[str(idx)]["example_overlay_path"] = out_path

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("[DONE]")
    print("[BOX JSON]", args.out_json)
    print("[EXAMPLE DIR]", args.example_dir)
    print("num records:", len(results))


if __name__ == "__main__":
    main()
