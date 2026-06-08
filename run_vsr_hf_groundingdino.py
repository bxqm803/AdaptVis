import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def norm_text(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return x


def match_phrase(label, target):
    label = norm_text(label)
    target = norm_text(target)

    if not label or not target:
        return False

    if label == target:
        return True
    if target in label:
        return True
    if label in target:
        return True

    lt = set(label.split())
    tt = set(target.split())
    if tt and len(lt & tt) / len(tt) >= 0.5:
        return True

    return False


def get_labels(result):
    # transformers versions may use "labels" or "text_labels"
    if "text_labels" in result:
        return result["text_labels"]
    if "labels" in result:
        return result["labels"]
    return [""] * len(result.get("scores", []))


def select_best(result, target):
    labels = get_labels(result)
    boxes = result["boxes"]
    scores = result["scores"]

    best = None
    for label, box, score in zip(labels, boxes, scores):
        label_str = str(label)
        score_f = float(score.detach().cpu()) if torch.is_tensor(score) else float(score)
        box_list = box.detach().cpu().tolist() if torch.is_tensor(box) else list(box)

        if match_phrase(label_str, target):
            cand = {
                "phrase": label_str,
                "score": score_f,
                "box_xyxy_pixel": [float(x) for x in box_list],
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

    return best


@torch.no_grad()
def run_one(model, processor, image_path, prompt, device, box_threshold, text_threshold):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # GroundingDINO prompts work better as period-separated lowercase phrases.
    # Example: "cat. suitcase."
    prompt = str(prompt).lower().strip()
    if not prompt.endswith("."):
        prompt += "."

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    labels = get_labels(result)
    detections = []
    for label, box, score in zip(labels, result["boxes"], result["scores"]):
        box_list = box.detach().cpu().tolist() if torch.is_tensor(box) else list(box)
        score_f = float(score.detach().cpu()) if torch.is_tensor(score) else float(score)

        detections.append({
            "phrase": str(label),
            "score": score_f,
            "box_xyxy_pixel": [float(x) for x in box_list],
        })

    return result, detections, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--unique_csv", default="data/vsr_grounding_queries_unique.csv")
    ap.add_argument("--all_csv", default="data/vsr_grounding_queries_all.csv")
    ap.add_argument("--image_col", required=True, choices=["pad336_path", "resize336_path"])
    ap.add_argument("--out_unique", required=True)
    ap.add_argument("--out_all", required=True)
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("model_id:", args.model_id)
    print("device:", device)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)
    model.eval()

    unique_df = pd.read_csv(args.unique_csv)
    all_df = pd.read_csv(args.all_csv)

    out_unique = Path(args.out_unique)
    out_all = Path(args.out_all)
    out_unique.parent.mkdir(parents=True, exist_ok=True)
    out_all.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc=f"HF GroundingDINO {args.image_col}"):
        filename = str(row["filename"])
        image_path = str(row[args.image_col])
        prompt = str(row["grounding_prompt"])
        subject = str(row["subject"])
        obj = str(row["object"])

        if not Path(image_path).exists():
            rec = {
                "filename": filename,
                "image_path": image_path,
                "grounding_prompt": prompt,
                "subject": subject,
                "object": obj,
                "status": "missing_image",
                "width": None,
                "height": None,
                "subject_box": None,
                "subject_score": None,
                "subject_phrase": None,
                "object_box": None,
                "object_score": None,
                "object_phrase": None,
                "detections": [],
            }
            records.append(rec)
            continue

        try:
            result, detections, w, h = run_one(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt=prompt,
                device=device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )

            subj_det = select_best(result, subject)
            obj_det = select_best(result, obj)

            rec = {
                "filename": filename,
                "image_path": image_path,
                "grounding_prompt": prompt,
                "subject": subject,
                "object": obj,
                "status": "ok",
                "width": w,
                "height": h,
                "subject_box": subj_det["box_xyxy_pixel"] if subj_det else None,
                "subject_score": subj_det["score"] if subj_det else None,
                "subject_phrase": subj_det["phrase"] if subj_det else None,
                "object_box": obj_det["box_xyxy_pixel"] if obj_det else None,
                "object_score": obj_det["score"] if obj_det else None,
                "object_phrase": obj_det["phrase"] if obj_det else None,
                "detections": detections,
            }

        except Exception as e:
            rec = {
                "filename": filename,
                "image_path": image_path,
                "grounding_prompt": prompt,
                "subject": subject,
                "object": obj,
                "status": f"error:{repr(e)}",
                "width": None,
                "height": None,
                "subject_box": None,
                "subject_score": None,
                "subject_phrase": None,
                "object_box": None,
                "object_score": None,
                "object_phrase": None,
                "detections": [],
            }

        records.append(rec)

    with open(out_unique, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lookup = {
        (r["filename"], r["grounding_prompt"]): r
        for r in records
    }

    with open(out_all, "w", encoding="utf-8") as f:
        for _, row in all_df.iterrows():
            key = (str(row["filename"]), str(row["grounding_prompt"]))
            det = lookup.get(key, {})

            sample = {
                "sid": int(row["sid"]),
                "filename": str(row["filename"]),
                "image": str(row["image"]),
                "caption": str(row["caption"]),
                "label": int(row["label"]),
                "subject": str(row["subject"]),
                "relation": str(row["relation"]),
                "object": str(row["object"]),
                "grounding_prompt": str(row["grounding_prompt"]),
                "image_path": str(row[args.image_col]),
                "status": det.get("status", "missing_detection"),
                "width": det.get("width"),
                "height": det.get("height"),
                "subject_box": det.get("subject_box"),
                "subject_score": det.get("subject_score"),
                "subject_phrase": det.get("subject_phrase"),
                "object_box": det.get("object_box"),
                "object_score": det.get("object_score"),
                "object_phrase": det.get("object_phrase"),
                "detections": det.get("detections", []),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total = len(records)
    ok = sum(r["status"] == "ok" for r in records)
    subj_ok = sum(r["subject_box"] is not None for r in records)
    obj_ok = sum(r["object_box"] is not None for r in records)
    both_ok = sum((r["subject_box"] is not None and r["object_box"] is not None) for r in records)

    print("\nSaved unique:", out_unique)
    print("Saved all:", out_all)
    print("total unique:", total)
    print("ok:", ok)
    print("subject found:", subj_ok)
    print("object found:", obj_ok)
    print("both found:", both_ok)


if __name__ == "__main__":
    main()
