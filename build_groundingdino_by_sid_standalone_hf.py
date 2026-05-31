#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone HF GroundingDINO -> by_sid patch json generator.

No AdaptVis imports:
  - no model_zoo
  - no dataset_zoo
  - no LLaVA wrapper

Inputs:
  --image-root: folder containing images
  --qa-file: optional CSV/JSON/JSONL with columns/keys:
      image_path / path / image / filename / file
      prompt / question / query
      gold / answer / label / target

Output format is compatible with run_objectbox_negpatch_once.py:
{
  "0": {"sample_id": 0, "image_path": ..., "patch_ids": [...], ...},
  "1": ...
}

Patch mapping:
  GroundingDINO bbox on original image
  -> binary mask on original image
  -> LLaVA-like preprocessing geometry
  -> 24x24 patch ids

Use --preprocess-mode pad for LLaVA-1.5 if image_aspect_ratio=pad.
Use --preprocess-mode crop for CLIP center-crop style.
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def norm_text(x: Any) -> str:
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def first_existing(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return default


def find_images(image_root: str) -> List[str]:
    root = Path(image_root)
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_records_from_csv(path: str, image_root: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            img = first_existing(r, ["image_path", "path", "image", "filename", "file"])
            if img and not os.path.isabs(img):
                cand = os.path.join(image_root, img)
                img = cand if os.path.exists(cand) else img
            out.append({
                "sample_id": i,
                "source_sample_id": first_existing(r, ["sample_id", "sid"], str(i)),
                "image_path": img,
                "prompt": first_existing(r, ["prompt", "question", "query"]),
                "gold": first_existing(r, ["gold", "answer", "label", "target"]),
            })
    return out


def load_records_from_json(path: str, image_root: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            items = data["records"]
        else:
            items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        raise TypeError(f"Unsupported json type: {type(data)}")

    out = []
    for i, r in enumerate(items):
        if not isinstance(r, dict):
            continue
        img = first_existing(r, ["image_path", "path", "image", "filename", "file"])
        if img and not os.path.isabs(img):
            cand = os.path.join(image_root, img)
            img = cand if os.path.exists(cand) else img
        out.append({
            "sample_id": i,
            "source_sample_id": r.get("sample_id", r.get("sid", i)),
            "image_path": img,
            "prompt": first_existing(r, ["prompt", "question", "query"]),
            "gold": first_existing(r, ["gold", "answer", "label", "target"]),
        })
    return out


def load_records_from_jsonl(path: str, image_root: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            img = first_existing(r, ["image_path", "path", "image", "filename", "file"])
            if img and not os.path.isabs(img):
                cand = os.path.join(image_root, img)
                img = cand if os.path.exists(cand) else img
            out.append({
                "sample_id": i,
                "source_sample_id": r.get("sample_id", r.get("sid", i)),
                "image_path": img,
                "prompt": first_existing(r, ["prompt", "question", "query"]),
                "gold": first_existing(r, ["gold", "answer", "label", "target"]),
            })
    return out


def load_records(image_root: str, qa_file: str = "") -> List[Dict[str, Any]]:
    if qa_file:
        suffix = Path(qa_file).suffix.lower()
        if suffix == ".csv":
            records = load_records_from_csv(qa_file, image_root)
        elif suffix == ".json":
            records = load_records_from_json(qa_file, image_root)
        elif suffix == ".jsonl":
            records = load_records_from_jsonl(qa_file, image_root)
        else:
            raise ValueError(f"Unsupported qa-file suffix: {suffix}")
    else:
        imgs = find_images(image_root)
        records = []
        for i, p in enumerate(imgs):
            stem = Path(p).stem.replace("_", " ")
            records.append({
                "sample_id": i,
                "source_sample_id": i,
                "image_path": p,
                "prompt": stem,
                "gold": stem,
            })

    good = []
    for r in records:
        if not r.get("image_path") or not os.path.exists(r["image_path"]):
            print("[WARN] missing image:", r.get("image_path"))
            continue
        rr = dict(r)
        rr["sample_id"] = len(good)  # ensure contiguous sid matching downstream iteration
        good.append(rr)
    return good


def parse_candidate_objects(prompt: str) -> Tuple[str, str]:
    p = str(prompt).strip()

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", p)
    if len(quoted) >= 2:
        return quoted[0].strip(), quoted[1].strip()

    patterns = [
        r"between\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+and\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:[?.]|$)",
        r"(?:a\s+|an\s+|the\s+)?([^?.;:,]+?)\s+or\s+(?:a\s+|an\s+|the\s+)?([^?.;:,]+?)(?:[?.]|$)",
        r"(?:a\s+|an\s+|the\s+)?([^?.;:,]+?)\s+and\s+(?:a\s+|an\s+|the\s+)?([^?.;:,]+?)(?:[?.]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, p, flags=re.I)
        if not m:
            continue
        left, right = m.group(1).strip(), m.group(2).strip()
        left = re.split(r"[,;:]", left)[-1].strip()
        left = re.sub(
            r"^(which|what)\s+(object|one|item)?\s*(is|are)?\s*(larger|smaller|bigger|closer|farther)?\s*",
            "",
            left,
            flags=re.I,
        ).strip()
        right = re.sub(r"\s*(larger|smaller|bigger|closer|farther)\s*$", "", right, flags=re.I).strip()
        if left and right:
            return left, right

    return "", ""


def build_targets(prompt: str, gold: str, image_path: str, target_mode: str) -> List[str]:
    obj1, obj2 = parse_candidate_objects(prompt)
    if target_mode == "gold":
        targets = [gold]
    elif target_mode == "candidates":
        targets = [x for x in [obj1, obj2] if x]
        if not targets:
            targets = [gold or Path(image_path).stem.replace("_", " ")]
    elif target_mode == "gold_and_candidates":
        targets = [gold] + [x for x in [obj1, obj2] if x]
    elif target_mode == "prompt":
        targets = [prompt]
    elif target_mode == "filename":
        targets = [Path(image_path).stem.replace("_", " ")]
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    out, seen = [], set()
    for t in targets:
        t = str(t).strip()
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def add_period(text: str) -> str:
    text = str(text).strip()
    return text if text.endswith(".") else text + "."


def load_groundingdino(model_name: str, cache_dir: str, device: str):
    """
    Requires transformers version with GroundingDINO support.
    This script intentionally does not import AutoProcessor from AdaptVis/LLaVA.
    """
    try:
        from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
        processor = GroundingDinoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model = GroundingDinoForObjectDetection.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e:
        print("[WARN] explicit GroundingDINO import failed:", repr(e))
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, cache_dir=cache_dir)

    model = model.to(device)
    model.eval()
    return processor, model


def detect_hf(
    processor: Any,
    model: Any,
    image: Image.Image,
    targets: List[str],
    box_threshold: float,
    text_threshold: float,
    device: str,
    max_boxes_per_target: int,
) -> List[Dict[str, Any]]:
    detections = []
    w, h = image.size

    for target in targets:
        text = add_period(target).lower()
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
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
        labels = results.get("labels", [target] * len(boxes))

        if len(boxes) == 0:
            continue

        scores_tensor = scores if torch.is_tensor(scores) else torch.tensor(scores)
        order = torch.argsort(scores_tensor, descending=True)
        if max_boxes_per_target > 0:
            order = order[:max_boxes_per_target]

        for idx in order.detach().cpu().tolist():
            box = boxes[idx]
            score = scores[idx]
            label = labels[idx]
            x1, y1, x2, y2 = [float(x) for x in box.detach().cpu().tolist()]
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "label": str(label),
                "target": str(target),
                "score": float(score.detach().cpu() if torch.is_tensor(score) else score),
                "box_xyxy": [x1, y1, x2, y2],
                "backend": "hf_transformers",
            })

    return detections


# ============================================================
# LLaVA-like geometry preprocessing
# ============================================================
def expand_to_square(img: Image.Image, background_color) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    out = Image.new(img.mode, (size, size), background_color)
    if w > h:
        out.paste(img, (0, (w - h) // 2))
    else:
        out.paste(img, ((h - w) // 2, 0))
    return out


def preprocess_geometry(img: Image.Image, mode: str, image_size: int, is_mask: bool) -> Image.Image:
    if mode not in {"pad", "crop", "resize"}:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    interp = Image.NEAREST if is_mask else Image.BICUBIC
    bg = 0 if is_mask else (0, 0, 0)

    if mode == "pad":
        return expand_to_square(img, bg).resize((image_size, image_size), interp)

    if mode == "resize":
        return img.resize((image_size, image_size), interp)

    # crop: resize shortest side to image_size, then center crop
    w, h = img.size
    scale = float(image_size) / float(min(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), interp)
    left = max(0, (nw - image_size) // 2)
    top = max(0, (nh - image_size) // 2)
    return resized.crop((left, top, left + image_size, top + image_size))


def mask_from_boxes(image_size: Tuple[int, int], boxes: List[List[float]]) -> Image.Image:
    w, h = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def mask_to_patch_ids(mask: Image.Image, patch_side: int, threshold: float, min_area_frac: float) -> List[int]:
    arr = np.asarray(mask).astype(np.float32) / 255.0
    h, w = arr.shape
    cell_h, cell_w = h / patch_side, w / patch_side

    ids = []
    for r in range(patch_side):
        for c in range(patch_side):
            y0, y1 = int(round(r * cell_h)), int(round((r + 1) * cell_h))
            x0, x1 = int(round(c * cell_w)), int(round((c + 1) * cell_w))
            crop = arr[y0:y1, x0:x1]
            if crop.size and float((crop >= threshold).mean()) >= min_area_frac:
                ids.append(r * patch_side + c)
    return ids


def draw_patch_overlay(img: Image.Image, patch_ids: List[int], patch_side: int) -> Image.Image:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    cw, ch = w / patch_side, h / patch_side

    for pid in patch_ids:
        r, c = int(pid) // patch_side, int(pid) % patch_side
        x0, y0 = c * cw, r * ch
        x1, y1 = (c + 1) * cw, (r + 1) * ch
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 220), fill=(255, 0, 0, 80), width=2)

    return Image.alpha_composite(base, overlay).convert("RGB")


def save_debug_vis(
    vis_dir: str,
    sid: int,
    image: Image.Image,
    detections: List[Dict[str, Any]],
    proc_mask: Image.Image,
    patch_ids: List[int],
    patch_side: int,
    preprocess_mode: str,
    image_size: int,
) -> None:
    os.makedirs(vis_dir, exist_ok=True)

    raw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(raw)
    for det in detections:
        box = det["box_xyxy"]
        draw.rectangle(box, outline=(255, 0, 0), width=4)
        draw.text((box[0], max(0, box[1] - 12)), f"{det.get('target','')}:{det.get('score',0):.2f}", fill=(255, 0, 0))
    raw.save(os.path.join(vis_dir, f"sid{sid:04d}_raw_bbox.png"))

    proc_img = preprocess_geometry(image, preprocess_mode, image_size, is_mask=False)
    proc_img.save(os.path.join(vis_dir, f"sid{sid:04d}_processed_image.png"))
    draw_patch_overlay(proc_img, patch_ids, patch_side).save(
        os.path.join(vis_dir, f"sid{sid:04d}_processed_image_patch_ids.png")
    )

    mask_rgb = Image.merge("RGB", [proc_mask, proc_mask, proc_mask])
    draw_patch_overlay(mask_rgb, patch_ids, patch_side).save(
        os.path.join(vis_dir, f"sid{sid:04d}_processed_mask_patch_ids.png")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="Folder containing images.")
    parser.add_argument("--qa-file", default="", help="Optional CSV/JSON/JSONL with image_path,prompt,gold.")
    parser.add_argument("--fresh-limit", type=int, default=-1)

    parser.add_argument("--hf-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-boxes-per-target", type=int, default=1)

    parser.add_argument("--target-mode", choices=["gold", "candidates", "gold_and_candidates", "prompt", "filename"], default="candidates")

    parser.add_argument("--preprocess-mode", choices=["pad", "crop", "resize"], default="pad")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--patch-side", type=int, default=24)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-patch-area-frac", type=float, default=0.01)

    parser.add_argument("--out-json", default="output/groundingdino_object_patch_masks_B_by_sid.json")
    parser.add_argument("--missing-json", default="output/groundingdino_object_patch_masks_B_missing.json")
    parser.add_argument("--manifest-json", default="output/groundingdino_dataset_manifest_B.json")
    parser.add_argument("--vis-dir", default="")
    parser.add_argument("--vis-first", type=int, default=0)
    args = parser.parse_args()

    ensure_dir_for_file(args.out_json)
    ensure_dir_for_file(args.missing_json)
    ensure_dir_for_file(args.manifest_json)

    records = load_records(args.image_root, args.qa_file)
    if args.fresh_limit > 0:
        records = records[:args.fresh_limit]

    print("[RECORDS]", len(records))
    print("[LOAD HF GroundingDINO]", args.hf_dino_model)
    processor, model = load_groundingdino(args.hf_dino_model, args.cache_dir, args.device)

    by_sid: Dict[str, Dict[str, Any]] = {}
    missing: Dict[str, Dict[str, Any]] = {}
    manifest: List[Dict[str, Any]] = []

    for rec_in in tqdm(records, desc="build masks"):
        sid = int(rec_in["sample_id"])
        image_path = rec_in["image_path"]
        prompt = rec_in.get("prompt", "")
        gold = norm_text(rec_in.get("gold", ""))

        image = Image.open(image_path).convert("RGB")
        image.filename = image_path

        obj1, obj2 = parse_candidate_objects(prompt)
        targets = build_targets(prompt, gold, image_path, args.target_mode)

        detections = detect_hf(
            processor=processor,
            model=model,
            image=image,
            targets=targets,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
            max_boxes_per_target=args.max_boxes_per_target,
        )

        raw_mask = mask_from_boxes(image.size, [d["box_xyxy"] for d in detections])
        proc_mask = preprocess_geometry(raw_mask, args.preprocess_mode, args.image_size, is_mask=True)
        patch_ids = mask_to_patch_ids(proc_mask, args.patch_side, args.mask_threshold, args.min_patch_area_frac)

        rec = {
            "sample_id": sid,
            "source_sample_id": rec_in.get("source_sample_id", sid),
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "obj1": obj1,
            "obj2": obj2,
            "targets": targets,
            "target_mode": args.target_mode,
            "image_width": int(image.size[0]),
            "image_height": int(image.size[1]),
            "patch_side": int(args.patch_side),
            "patch_ids": [int(x) for x in patch_ids],
            "detections": detections,
            "backend": "hf_transformers_standalone",
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "mask_threshold": float(args.mask_threshold),
            "min_patch_area_frac": float(args.min_patch_area_frac),
            "preprocess_mode": args.preprocess_mode,
            "image_size": int(args.image_size),
            "processor_aware_patch_mapping": "standalone_geometry",
        }
        by_sid[str(sid)] = rec

        manifest.append({
            "sample_id": sid,
            "source_sample_id": rec_in.get("source_sample_id", sid),
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "targets": targets,
            "num_detections": len(detections),
            "num_patch_ids": len(patch_ids),
        })

        if len(detections) == 0 or len(patch_ids) == 0:
            missing[str(sid)] = rec

        if args.vis_dir and sid < args.vis_first:
            save_debug_vis(
                args.vis_dir,
                sid,
                image,
                detections,
                proc_mask,
                patch_ids,
                args.patch_side,
                args.preprocess_mode,
                args.image_size,
            )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(by_sid, f, ensure_ascii=False, indent=2)
    with open(args.missing_json, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)
    with open(args.manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print("[OUT]", args.out_json, "num_records=", len(by_sid))
    print("[MISSING]", args.missing_json, "num_missing=", len(missing))
    print("[MANIFEST]", args.manifest_json)
    if args.vis_dir:
        print("[VIS]", args.vis_dir)


if __name__ == "__main__":
    main()
