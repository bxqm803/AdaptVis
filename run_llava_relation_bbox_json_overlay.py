import os
import re
import csv
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ask LLaVA for object bboxes derived from the dataset relation question, save JSON, and overlay predicted boxes on images."
    )
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="output_llava_relation_bbox_json_overlay", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=10, type=int, help="Default 10, set to -1 for all.")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--print-prompts", action="store_true")
    p.add_argument("--print-raw", action="store_true")
    return p.parse_args()


# -------------------------
# Text helpers
# -------------------------
def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def strip_answer_order_clause(q: str) -> str:
    q = clean_text(q)
    # Remove typical trailing answer-choice instructions.
    q = re.sub(r"\s*(?:Please\s+)?(?:answer|respond).*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Your answer should be.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Answer with.*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*Choose from.*$", "", q, flags=re.IGNORECASE)
    q = q.strip()
    if q and q[-1] not in "?.!":
        q += "?"
    return q


def normalize_object_name(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"^(a|an|the)\s+", "", name)
    name = re.sub(r"[?.!,;:]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_objects_from_question(question: str) -> Tuple[Optional[str], Optional[str], str]:
    q = strip_answer_order_clause(question)

    patterns = [
        r"where\s+is\s+(?:the\s+)?(.+?)\s+in\s+relation\s+to\s+(?:the\s+)?(.+?)\?",
        r"where\s+is\s+(?:the\s+)?(.+?)\s+relative\s+to\s+(?:the\s+)?(.+?)\?",
        r"where\s+is\s+(?:the\s+)?(.+?)\s+with\s+respect\s+to\s+(?:the\s+)?(.+?)\?",
        r"what\s+is\s+the\s+position\s+of\s+(?:the\s+)?(.+?)\s+relative\s+to\s+(?:the\s+)?(.+?)\?",
    ]

    q_lower = q.lower()
    for pat in patterns:
        m = re.search(pat, q_lower, flags=re.IGNORECASE)
        if m:
            obj1 = normalize_object_name(m.group(1))
            obj2 = normalize_object_name(m.group(2))
            return obj1 or None, obj2 or None, q

    # Conservative fallback: try relation words.
    rel_words = ["left of", "right of", "in front of", "behind", "above", "below", "under", "on"]
    for rel in rel_words:
        pat = rf"(?:the\s+)?(.+?)\s+{re.escape(rel)}\s+(?:the\s+)?(.+?)$"
        m = re.search(pat, q_lower.rstrip("?"), flags=re.IGNORECASE)
        if m:
            obj1 = normalize_object_name(m.group(1))
            obj2 = normalize_object_name(m.group(2))
            return obj1 or None, obj2 or None, q

    return None, None, q


# -------------------------
# JSON parsing helpers
# -------------------------
def _strip_code_fences(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def try_parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw = _strip_code_fences(text)
    decoder = json.JSONDecoder()

    # Try direct parse first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    # Scan for first JSON object.
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(raw[i:])
            if isinstance(obj, dict):
                return obj, None
        except Exception:
            continue

    return None, "Could not parse a JSON object from model output."


# -------------------------
# Box conversion helpers
# -------------------------
def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _convert_box_dict(box: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    # x1,y1,x2,y2
    if all(k in box for k in ["x1", "y1", "x2", "y2"]):
        vals = [_to_float(box.get(k)) for k in ["x1", "y1", "x2", "y2"]]
        if all(v is not None for v in vals):
            return vals[0], vals[1], vals[2], vals[3]

    # xmin,ymin,xmax,ymax
    if all(k in box for k in ["xmin", "ymin", "xmax", "ymax"]):
        vals = [_to_float(box.get(k)) for k in ["xmin", "ymin", "xmax", "ymax"]]
        if all(v is not None for v in vals):
            return vals[0], vals[1], vals[2], vals[3]

    # x,y,w,h
    if all(k in box for k in ["x", "y", "w", "h"]):
        x, y, w, h = [_to_float(box.get(k)) for k in ["x", "y", "w", "h"]]
        if None not in (x, y, w, h):
            return x, y, x + w, y + h

    # left,top,right,bottom
    if all(k in box for k in ["left", "top", "right", "bottom"]):
        vals = [_to_float(box.get(k)) for k in ["left", "top", "right", "bottom"]]
        if all(v is not None for v in vals):
            return vals[0], vals[1], vals[2], vals[3]

    # cx,cy,w,h
    if all(k in box for k in ["cx", "cy", "w", "h"]):
        cx, cy, w, h = [_to_float(box.get(k)) for k in ["cx", "cy", "w", "h"]]
        if None not in (cx, cy, w, h):
            return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    return None


def coerce_box(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None

    if isinstance(value, dict):
        return _convert_box_dict(value)

    if isinstance(value, (list, tuple)) and len(value) == 4:
        vals = [_to_float(x) for x in value]
        if all(v is not None for v in vals):
            return vals[0], vals[1], vals[2], vals[3]

    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) >= 4:
            vals = [float(x) for x in nums[:4]]
            return vals[0], vals[1], vals[2], vals[3]

    return None


BOX_KEY_CANDIDATES = {
    "target": [
        "target_bbox", "object_1_bbox", "obj1_bbox", "bbox", "box",
        "target_box", "subject_bbox", "subject_box"
    ],
    "reference": [
        "reference_bbox", "object_2_bbox", "obj2_bbox", "ref_bbox",
        "reference_box", "context_bbox", "other_bbox", "other_box"
    ],
}


NAME_KEY_CANDIDATES = {
    "target": ["target_object", "object_1_name", "obj1_name", "subject", "object"],
    "reference": ["reference_object", "object_2_name", "obj2_name", "reference", "other_object"],
}


def extract_named_box(payload: Dict[str, Any], role: str) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]]]:
    name = None
    for k in NAME_KEY_CANDIDATES[role]:
        if k in payload and clean_text(payload[k]):
            name = normalize_object_name(payload[k])
            break

    box = None
    for k in BOX_KEY_CANDIDATES[role]:
        if k in payload:
            box = coerce_box(payload[k])
            if box is not None:
                break

    # nested object forms
    if box is None:
        nested_keys = []
        if role == "target":
            nested_keys = ["target", "object_1", "obj1", "subject"]
        else:
            nested_keys = ["reference", "object_2", "obj2", "other"]
        for nk in nested_keys:
            if isinstance(payload.get(nk), dict):
                box = coerce_box(payload[nk])
                if name is None:
                    for kk in ["name", "label", "object"]:
                        if kk in payload[nk] and clean_text(payload[nk][kk]):
                            name = normalize_object_name(payload[nk][kk])
                            break
                if box is not None:
                    break

    return name, box


def normalize_box_to_pixels(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    vals = [x1, y1, x2, y2]
    max_abs = max(abs(v) for v in vals)

    # 0..1 normalized
    if max_abs <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    # 0..1000 normalized
    elif max_abs <= 1001:
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    # else assume already in pixels

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x1 = int(round(_clamp(x1, 0, width - 1)))
    y1 = int(round(_clamp(y1, 0, height - 1)))
    x2 = int(round(_clamp(x2, 0, width - 1)))
    y2 = int(round(_clamp(y2, 0, height - 1)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# -------------------------
# Prompt + generation
# -------------------------
def build_bbox_json_prompt(obj1: str, obj2: str, original_question: str) -> str:
    relation_q = f"Where is the {obj1} in relation to the {obj2}?"
    return (
        "<image>\n"
        "USER: Identify the two objects from the image and answer with bounding boxes only.\n"
        f"Original dataset question: {strip_answer_order_clause(original_question)}\n"
        f"Use this object query: {relation_q}\n"
        "Return ONLY valid JSON, with no extra text and no markdown fences.\n"
        "Use this exact schema and integer coordinates in a 0-1000 scale relative to the full image:\n"
        '{"target_object":"%s","reference_object":"%s","target_bbox":{"x1":0,"y1":0,"x2":0,"y2":0},"reference_bbox":{"x1":0,"y1":0,"x2":0,"y2":0}}\n'
        "Rules: x1 < x2, y1 < y2. If an object is not visible, set its bbox to null.\n"
        "ASSISTANT:" % (obj1, obj2)
    )


def run_llava_once(wrapper, image: Image.Image, prompt: str) -> str:
    out = wrapper.run_single_prompt(
        image=image,
        prompt=prompt,
        method="base",
        weight=None,
    )
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for k in ["response", "text", "pred_text", "output", "answer"]:
            if k in out:
                return str(out[k])
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return str(out[0])
    return str(out)


# -------------------------
# Overlay helpers
# -------------------------
def draw_bbox(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    text = label[:80]
    # simple text background
    try:
        bbox = draw.textbbox((x1, y1), text)
        tx1, ty1, tx2, ty2 = bbox
        pad = 3
        bg = [tx1 - pad, max(0, ty1 - pad), tx2 + pad, ty2 + pad]
        draw.rectangle(bg, fill=(255, 0, 0))
        draw.text((x1, max(0, y1 - (ty2 - ty1) - 6)), text, fill=(255, 255, 255))
    except Exception:
        draw.text((x1, max(0, y1 - 12)), text, fill=(255, 0, 0))


def save_overlay(
    image: Image.Image,
    target_box: Optional[Tuple[int, int, int, int]],
    target_name: str,
    ref_box: Optional[Tuple[int, int, int, int]],
    ref_name: str,
    out_path: str,
) -> None:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    if target_box is not None:
        draw_bbox(draw, target_box, f"target: {target_name}")
    if ref_box is not None:
        # blue-ish outline via repeated rectangles, since PIL ImageDraw uses tuple color.
        x1, y1, x2, y2 = ref_box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 128, 255), width=4)
        try:
            bbox = draw.textbbox((x1, y1), f"ref: {ref_name}")
            tx1, ty1, tx2, ty2 = bbox
            pad = 3
            bg = [tx1 - pad, max(0, ty1 - pad), tx2 + pad, ty2 + pad]
            draw.rectangle(bg, fill=(0, 128, 255))
            draw.text((x1, max(0, y1 - (ty2 - ty1) - 6)), f"ref: {ref_name}", fill=(255, 255, 255))
        except Exception:
            draw.text((x1, max(0, y1 - 12)), f"ref: {ref_name}", fill=(0, 128, 255))
    vis.save(out_path)


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        import torch
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    model_tag = f"{args.model_name}_relation_bbox_json"
    out_root = os.path.join(args.out_dir, args.dataset, model_tag)
    os.makedirs(out_root, exist_ok=True)
    summary_csv = os.path.join(out_root, "summary_relation_bbox_json.csv")
    report_json = os.path.join(out_root, "report.json")

    rows: List[Dict[str, Any]] = []

    for local_idx in range(start, end):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image = item["image_options"][0]
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL image from dataset[local_idx]['image_options'][0], got {type(image)}. "
                "This script assumes the repo's llava path where image_preprocess is None."
            )

        sample_dir = os.path.join(out_root, os.path.splitext(image_name)[0])
        os.makedirs(sample_dir, exist_ok=True)
        sample_json = os.path.join(sample_dir, "result.json")
        overlay_png = os.path.join(sample_dir, "overlay_pred.png")

        if args.skip_existing and os.path.exists(sample_json) and os.path.exists(overlay_png):
            continue

        raw_question = rec["question"]
        gold_relation = clean_text(rec.get("answer", ""))
        obj1, obj2, stripped_q = extract_objects_from_question(raw_question)

        parse_error = None
        if not obj1 or not obj2:
            parse_error = "Could not reliably extract the two objects from the original question."
            prompt = None
            raw_output = ""
            parsed_payload = None
            target_name = None
            ref_name = None
            target_box_px = None
            ref_box_px = None
            overlay_saved = False
        else:
            prompt = build_bbox_json_prompt(obj1, obj2, raw_question)
            raw_output = run_llava_once(wrapper, image, prompt)
            parsed_payload, json_error = try_parse_json_object(raw_output)
            parse_error = json_error

            if parsed_payload is None:
                target_name = obj1
                ref_name = obj2
                target_box_px = None
                ref_box_px = None
                overlay_saved = False
            else:
                target_name_pred, target_box = extract_named_box(parsed_payload, "target")
                ref_name_pred, ref_box = extract_named_box(parsed_payload, "reference")

                target_name = target_name_pred or obj1
                ref_name = ref_name_pred or obj2

                w, h = image.size
                target_box_px = normalize_box_to_pixels(target_box, w, h) if target_box is not None else None
                ref_box_px = normalize_box_to_pixels(ref_box, w, h) if ref_box is not None else None

                if target_box_px is not None or ref_box_px is not None:
                    save_overlay(image, target_box_px, target_name, ref_box_px, ref_name, overlay_png)
                    overlay_saved = True
                else:
                    overlay_saved = False

        payload = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "original_question": raw_question,
            "parsed_relation_question": stripped_q,
            "gold_relation": gold_relation,
            "object_1_name": obj1,
            "object_2_name": obj2,
            "prompt": prompt,
            "raw_model_output": raw_output,
            "parsed_json": parsed_payload,
            "target_name_final": target_name,
            "reference_name_final": ref_name,
            "target_bbox_px": list(target_box_px) if target_box_px is not None else None,
            "reference_bbox_px": list(ref_box_px) if ref_box_px is not None else None,
            "parse_error": parse_error,
            "overlay_png": os.path.relpath(overlay_png, out_root) if overlay_saved else None,
        }
        with open(sample_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        row = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "gold_relation": gold_relation,
            "original_question": raw_question,
            "object_1_name": obj1,
            "object_2_name": obj2,
            "prompt": prompt or "",
            "raw_model_output": raw_output,
            "target_name_final": target_name or "",
            "reference_name_final": ref_name or "",
            "target_bbox_px": json.dumps(list(target_box_px) if target_box_px is not None else None, ensure_ascii=False),
            "reference_bbox_px": json.dumps(list(ref_box_px) if ref_box_px is not None else None, ensure_ascii=False),
            "parse_error": parse_error or "",
            "overlay_png": os.path.relpath(overlay_png, out_root) if overlay_saved else "",
            "sample_json": os.path.relpath(sample_json, out_root),
        }
        rows.append(row)

        print("=" * 100)
        print(f"[{local_idx}] {image_name}")
        print(f"question: {strip_answer_order_clause(raw_question)}")
        print(f"objects: {obj1} | {obj2}")
        if args.print_prompts and prompt:
            print("[PROMPT]")
            print(prompt)
        if args.print_raw:
            print("[RAW OUTPUT]")
            print(raw_output)
        print(f"target_bbox_px={target_box_px} | reference_bbox_px={ref_box_px} | parse_error={parse_error}")

    fieldnames = [
        "image_name", "image_path", "local_index", "gold_relation", "original_question",
        "object_1_name", "object_2_name", "prompt", "raw_model_output",
        "target_name_final", "reference_name_final", "target_bbox_px", "reference_bbox_px",
        "parse_error", "overlay_png", "sample_json",
    ]
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "num_rows": len(rows),
        "num_json_parsed": sum(1 for r in rows if not r["parse_error"]),
        "num_target_bbox": sum(1 for r in rows if r["target_bbox_px"] not in ("null", "", "None")),
        "num_reference_bbox": sum(1 for r in rows if r["reference_bbox_px"] not in ("null", "", "None")),
        "summary_csv": summary_csv,
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved report to: {report_json}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
