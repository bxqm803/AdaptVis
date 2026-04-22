import os
import re
import csv
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manual-bbox-csv", required=True, type=str)
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="output_llava_manual_bbox_image_compare_repo", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--line-width", default=6, type=int)
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


def run_repo_llava_once(wrapper, image, prompt: str) -> str:
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


def draw_bbox_on_image(image, bbox1: Optional[List[int]], bbox2: Optional[List[int]], line_width: int = 6):
    from PIL import ImageDraw

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    if bbox1 is not None:
        draw.rectangle(bbox1, outline="red", width=line_width)
    if bbox2 is not None:
        draw.rectangle(bbox2, outline="blue", width=line_width)

    return img


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    baseline_correct = sum(1 for r in rows if bool(r["baseline_correct"]))
    bbox_image_correct = sum(1 for r in rows if bool(r["bbox_image_correct"]))
    improved = sum(1 for r in rows if (not bool(r["baseline_correct"])) and bool(r["bbox_image_correct"]))
    worsened = sum(1 for r in rows if bool(r["baseline_correct"]) and (not bool(r["bbox_image_correct"])))
    same = total - improved - worsened

    return {
        "num_images": total,
        "baseline_accuracy": 0.0 if total == 0 else baseline_correct / total,
        "bbox_image_accuracy": 0.0 if total == 0 else bbox_image_correct / total,
        "baseline_correct": baseline_correct,
        "bbox_image_correct": bbox_image_correct,
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

    # 严格走仓库里的 llava1.5
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    # 严格走仓库自己的 prompt-record 路径
    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)

    if sampled_indices is not None:
        import torch
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    model_name = f"{args.model_name}_repo_bbox_image"
    out_root = os.path.join(args.out_dir, args.dataset, model_name)
    os.makedirs(out_root, exist_ok=True)

    summary_csv = os.path.join(out_root, "summary_manual_bbox_image_compare.csv")
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
        obj1_bbox = bbox_row["object_1_bbox"]
        obj2_bbox = bbox_row["object_2_bbox"]

        if obj1_bbox is None or obj2_bbox is None:
            continue

        baseline_prompt = raw_question
        bbox_image_prompt = raw_question

        image_with_boxes = draw_bbox_on_image(
            image=image,
            bbox1=obj1_bbox,
            bbox2=obj2_bbox,
            line_width=args.line_width,
        )

        baseline_gen = run_repo_llava_once(wrapper, image, baseline_prompt)
        bbox_image_gen = run_repo_llava_once(wrapper, image_with_boxes, bbox_image_prompt)

        baseline_pred = parse_prediction(baseline_gen)
        bbox_image_pred = parse_prediction(bbox_image_gen)
        baseline_correct = baseline_pred == gold
        bbox_image_correct = bbox_image_pred == gold

        print("=" * 100)
        print(f"image_name: {image_name}")
        print("[BASELINE PROMPT]")
        print(baseline_prompt)
        print("[BBOX-IMAGE PROMPT]")
        print(bbox_image_prompt)
        print(f"gold={gold} | baseline_pred={baseline_pred} | bbox_image_pred={bbox_image_pred}")

        if args.print_prompts:
            print(f"[BASELINE GEN] {baseline_gen}")
            print(f"[BBOX-IMAGE GEN] {bbox_image_gen}")

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
            "object_1_bbox": obj1_bbox,
            "object_2_bbox": obj2_bbox,
            "baseline_prompt": baseline_prompt,
            "bbox_image_prompt": bbox_image_prompt,
            "baseline_generation": baseline_gen,
            "bbox_image_generation": bbox_image_gen,
            "baseline_pred": baseline_pred,
            "bbox_image_pred": bbox_image_pred,
            "baseline_correct": baseline_correct,
            "bbox_image_correct": bbox_image_correct,
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
            "object_1_bbox": bbox_to_json(obj1_bbox),
            "object_2_bbox": bbox_to_json(obj2_bbox),
            "baseline_prompt": baseline_prompt,
            "bbox_image_prompt": bbox_image_prompt,
            "baseline_generation": baseline_gen,
            "bbox_image_generation": bbox_image_gen,
            "baseline_pred": baseline_pred,
            "bbox_image_pred": bbox_image_pred,
            "baseline_correct": baseline_correct,
            "bbox_image_correct": bbox_image_correct,
            "delta": int(bbox_image_correct) - int(baseline_correct),
            "sample_json": os.path.relpath(sample_json, out_root),
        })

        fieldnames = [
            "image_name",
            "image_path",
            "local_index",
            "gold",
            "object_1_name",
            "object_2_name",
            "object_1_bbox",
            "object_2_bbox",
            "baseline_prompt",
            "bbox_image_prompt",
            "baseline_generation",
            "bbox_image_generation",
            "baseline_pred",
            "bbox_image_pred",
            "baseline_correct",
            "bbox_image_correct",
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
