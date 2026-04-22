import os
import re
import csv
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


CHOICES = ["left", "right", "on", "under"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manual-bbox-csv", required=True, type=str)
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="output_llava_manual_bbox_compare_repo", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--print-prompts", action="store_true")
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


def bbox_to_json(bbox: Optional[List[int]]) -> str:
    return "" if bbox is None else json.dumps(bbox, ensure_ascii=False)


def normalize_rel(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            return "UNK"
        answer = answer[0]
    if answer is None:
        return "UNK"
    rel = str(answer).strip().lower()
    mapping = {
        "left": "left",
        "right": "right",
        "on": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
        "top": "on",
        "above": "on",
        "to the left of": "left",
        "to the right of": "right",
        "on top of": "on",
    }
    return mapping.get(rel, rel)


def parse_prediction(text: str) -> str:
    t = clean_text(text).lower()
    m = re.search(r"\b(left|right|on|under)\b", t)
    return m.group(1) if m else "UNK"


def clean_question_text(question: str) -> str:
    q = question.strip().replace("\n", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def strip_answer_order_clause(question_text: str) -> str:
    q = clean_question_text(question_text)
    q = re.sub(
        r"Answer with\s+left,\s*right,\s*on\s+or\s+under(?:\s+only)?\.\s*$",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q


def load_manual_bbox_rows(csv_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
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
            out[image_name] = {
                "image_name": image_name,
                "image_path": clean_text(row.get("image_path", "")),
                "object_1_name": obj1_name,
                "object_2_name": obj2_name,
                "object_1_bbox": parse_bbox_string(row.get("object_1_bbox", "")),
                "object_2_bbox": parse_bbox_string(row.get("object_2_bbox", "")),
            }
    return out


def _get_shortest_edge(feature_extractor) -> Optional[int]:
    size = getattr(feature_extractor, "size", None)
    if isinstance(size, dict):
        if "shortest_edge" in size:
            return int(size["shortest_edge"])
        if "height" in size and "width" in size and int(size["height"]) == int(size["width"]):
            return int(size["height"])
    elif isinstance(size, int):
        return int(size)
    return None


def _get_crop_hw(feature_extractor) -> Tuple[Optional[int], Optional[int]]:
    crop = getattr(feature_extractor, "crop_size", None)
    if isinstance(crop, dict):
        h = crop.get("height", crop.get("shortest_edge", None))
        w = crop.get("width", crop.get("shortest_edge", None))
        return (int(h) if h is not None else None, int(w) if w is not None else None)
    if isinstance(crop, int):
        return int(crop), int(crop)
    return None, None


def clip_bbox(bbox: Optional[List[int]], width: int, height: int) -> Optional[List[int]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def map_bbox_to_repo_processor_coords(bbox: Optional[List[int]], image_w: int, image_h: int, feature_extractor) -> Optional[List[int]]:
    if bbox is None:
        return None

    do_resize = bool(getattr(feature_extractor, "do_resize", True))
    do_center_crop = bool(getattr(feature_extractor, "do_center_crop", False))
    shortest_edge = _get_shortest_edge(feature_extractor)
    crop_h, crop_w = _get_crop_hw(feature_extractor)

    x1, y1, x2, y2 = bbox
    rw, rh = image_w, image_h

    if do_resize and shortest_edge is not None:
        scale = float(shortest_edge) / float(min(image_w, image_h))
        x1 = int(round(x1 * scale))
        y1 = int(round(y1 * scale))
        x2 = int(round(x2 * scale))
        y2 = int(round(y2 * scale))
        rw = int(round(image_w * scale))
        rh = int(round(image_h * scale))

    if do_center_crop and crop_h is not None and crop_w is not None:
        left = max(0, int(round((rw - crop_w) / 2.0)))
        top = max(0, int(round((rh - crop_h) / 2.0)))
        x1 -= left
        x2 -= left
        y1 -= top
        y2 -= top
        return clip_bbox([x1, y1, x2, y2], crop_w, crop_h)

    return clip_bbox([x1, y1, x2, y2], rw, rh)


def infer_bbox_relation(obj1_bbox: Optional[List[int]], obj2_bbox: Optional[List[int]]) -> str:
    if obj1_bbox is None or obj2_bbox is None:
        return "UNK"
    x1a, y1a, x2a, y2a = obj1_bbox
    x1b, y1b, x2b, y2b = obj2_bbox
    cxa = (x1a + x2a) / 2.0
    cya = (y1a + y2a) / 2.0
    cxb = (x1b + x2b) / 2.0
    cyb = (y1b + y2b) / 2.0
    dx = cxa - cxb
    dy = cya - cyb
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "under" if dy > 0 else "on"


def build_bbox_prompt(raw_question: str,
                      obj1_name: str,
                      obj2_name: str,
                      obj1_bbox_proc: Optional[List[int]],
                      obj2_bbox_proc: Optional[List[int]],
                      feature_extractor) -> str:
    stem = strip_answer_order_clause(raw_question)
    crop_h, crop_w = _get_crop_hw(feature_extractor)
    shortest_edge = _get_shortest_edge(feature_extractor)
    relation_hint = infer_bbox_relation(obj1_bbox_proc, obj2_bbox_proc)

    proc_desc_parts = []
    if shortest_edge is not None:
        proc_desc_parts.append(f"resize shortest edge to {shortest_edge}")
    if getattr(feature_extractor, "do_center_crop", False) and crop_h is not None and crop_w is not None:
        proc_desc_parts.append(f"center crop to {crop_w}x{crop_h}")
    proc_desc = ", then ".join(proc_desc_parts) if proc_desc_parts else "the repo image processor"

    bbox_info = (
        f"Bounding boxes in {proc_desc} image space: "
        f"{obj1_name}={bbox_to_json(obj1_bbox_proc)}, "
        f"{obj2_name}={bbox_to_json(obj2_bbox_proc)}."
    )

    return (
        f"<image>\n"
        f"USER: {bbox_info}\n"
        f"Question: {stem} Answer with left, right, on or under only.\n"
        f"ASSISTANT:"
    )


def run_repo_llava_once(wrapper, image, prompt: str) -> str:
    return wrapper.run_single_prompt(
        image=image,
        prompt=prompt,
        method="base",
        weight=None,
    )


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    base_correct = sum(1 for r in rows if bool(r["baseline_correct"]))
    bbox_correct = sum(1 for r in rows if bool(r["bbox_correct"]))
    improved = sum(1 for r in rows if (not bool(r["baseline_correct"])) and bool(r["bbox_correct"]))
    worsened = sum(1 for r in rows if bool(r["baseline_correct"]) and (not bool(r["bbox_correct"])))
    same = total - improved - worsened
    return {
        "num_images": total,
        "baseline_accuracy": 0.0 if total == 0 else base_correct / total,
        "bbox_accuracy": 0.0 if total == 0 else bbox_correct / total,
        "baseline_correct": base_correct,
        "bbox_correct": bbox_correct,
        "improved": improved,
        "worsened": worsened,
        "same": same,
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

    manual_bbox = load_manual_bbox_rows(args.manual_bbox_csv)
    if not manual_bbox:
        raise ValueError(f"No usable rows found in manual bbox csv: {args.manual_bbox_csv}")

    wrapper, image_preprocess = get_model("llava1.5", args.device, method="base", root_dir=cache_dir)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    # Use the repo's own prompt-record loading path for consistency.
    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        dataset = __import__('torch').utils.data.Subset(dataset, sampled_indices)

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    model_name = "llava1.5_repo"
    out_root = os.path.join(args.out_dir, args.dataset, model_name)
    os.makedirs(out_root, exist_ok=True)

    summary_csv = os.path.join(out_root, "summary_manual_bbox_compare.csv")
    report_json = os.path.join(out_root, "report.json")

    rows: List[Dict[str, Any]] = []

    for local_idx in range(start, end):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        if image_name not in manual_bbox:
            continue

        image = item["image_options"][0]
        image_path = clean_text(item.get("image_path", ""))
        raw_question = rec["question"]
        gold = normalize_rel(rec.get("answer", "UNK"))

        bbox_row = manual_bbox[image_name]
        obj1_name = bbox_row["object_1_name"]
        obj2_name = bbox_row["object_2_name"]
        obj1_bbox_orig = bbox_row["object_1_bbox"]
        obj2_bbox_orig = bbox_row["object_2_bbox"]

        image_w, image_h = image.size
        obj1_bbox_proc = map_bbox_to_repo_processor_coords(obj1_bbox_orig, image_w, image_h, wrapper.feature_extractor)
        obj2_bbox_proc = map_bbox_to_repo_processor_coords(obj2_bbox_orig, image_w, image_h, wrapper.feature_extractor)

        baseline_prompt = raw_question
        bbox_prompt = build_bbox_prompt(
            raw_question=raw_question,
            obj1_name=obj1_name,
            obj2_name=obj2_name,
            obj1_bbox_proc=obj1_bbox_proc,
            obj2_bbox_proc=obj2_bbox_proc,
            feature_extractor=wrapper.feature_extractor,
        )

        baseline_gen = run_repo_llava_once(wrapper, image, baseline_prompt)
        bbox_gen = run_repo_llava_once(wrapper, image, bbox_prompt)

        baseline_pred = parse_prediction(baseline_gen)
        bbox_pred = parse_prediction(bbox_gen)
        baseline_correct = baseline_pred == gold
        bbox_correct = bbox_pred == gold

        print("=" * 100)
        print(f"image_name: {image_name}")
        print("[BASELINE PROMPT]")
        print(baseline_prompt)
        print("[BBOX PROMPT]")
        print(bbox_prompt)
        print(f"gold={gold} | baseline_pred={baseline_pred} | bbox_pred={bbox_pred}")

        if args.print_prompts:
            print(f"[BASELINE GEN] {baseline_gen}")
            print(f"[BBOX GEN] {bbox_gen}")

        sample_dir = os.path.join(out_root, os.path.splitext(image_name)[0])
        os.makedirs(sample_dir, exist_ok=True)
        sample_json = os.path.join(sample_dir, "compare.json")

        sample_payload = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "gold": gold,
            "object_1_name": obj1_name,
            "object_2_name": obj2_name,
            "object_1_bbox_orig": obj1_bbox_orig,
            "object_2_bbox_orig": obj2_bbox_orig,
            "object_1_bbox_repo_proc": obj1_bbox_proc,
            "object_2_bbox_repo_proc": obj2_bbox_proc,
            "baseline_prompt": baseline_prompt,
            "bbox_prompt": bbox_prompt,
            "baseline_generation": baseline_gen,
            "bbox_generation": bbox_gen,
            "baseline_pred": baseline_pred,
            "bbox_pred": bbox_pred,
            "baseline_correct": baseline_correct,
            "bbox_correct": bbox_correct,
        }
        with open(sample_json, "w", encoding="utf-8") as f:
            json.dump(sample_payload, f, ensure_ascii=False, indent=2)

        rows.append({
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "gold": gold,
            "object_1_name": obj1_name,
            "object_2_name": obj2_name,
            "object_1_bbox_orig": bbox_to_json(obj1_bbox_orig),
            "object_2_bbox_orig": bbox_to_json(obj2_bbox_orig),
            "object_1_bbox_repo_proc": bbox_to_json(obj1_bbox_proc),
            "object_2_bbox_repo_proc": bbox_to_json(obj2_bbox_proc),
            "baseline_prompt": baseline_prompt,
            "bbox_prompt": bbox_prompt,
            "baseline_generation": baseline_gen,
            "bbox_generation": bbox_gen,
            "baseline_pred": baseline_pred,
            "bbox_pred": bbox_pred,
            "baseline_correct": baseline_correct,
            "bbox_correct": bbox_correct,
            "delta": int(bbox_correct) - int(baseline_correct),
            "sample_json": os.path.relpath(sample_json, out_root),
        })

        fieldnames = [
            "image_name",
            "image_path",
            "local_index",
            "gold",
            "object_1_name",
            "object_2_name",
            "object_1_bbox_orig",
            "object_2_bbox_orig",
            "object_1_bbox_repo_proc",
            "object_2_bbox_repo_proc",
            "baseline_prompt",
            "bbox_prompt",
            "baseline_generation",
            "bbox_generation",
            "baseline_pred",
            "bbox_pred",
            "baseline_correct",
            "bbox_correct",
            "delta",
            "sample_json",
        ]
        with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    report = build_report(rows)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved report to: {report_json}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
