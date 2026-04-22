import os
import re
import csv
import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-bbox-csv", required=True, type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--option", default="four", type=str)
    parser.add_argument("--model-name", default="llava1.5", type=str)
    parser.add_argument("--method", default="baseline", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out-dir", default="output_llava_manual_bbox_image_compare_repo", type=str)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--line-width", default=5, type=int)
    return parser.parse_args()


def clean_text(x):
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def parse_bbox(x):
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
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]
    except Exception:
        return None


def bbox_to_json(bbox):
    if bbox is None:
        return ""
    return json.dumps(bbox, ensure_ascii=False)


def infer_gold_from_image_name(image_name):
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    if "_left_of_" in stem:
        return "left"
    if "_right_of_" in stem:
        return "right"
    if "_on_" in stem:
        return "on"
    if "_under_" in stem:
        return "under"
    return "UNK"


def parse_prediction(text):
    t = clean_text(text).lower()
    m = re.search(r"\b(left|right|on|under)\b", t)
    if m:
        return m.group(1)
    return "UNK"


def strip_legacy_prompt(prompt: str) -> str:
    prompt = clean_text(prompt)
    prompt = prompt.replace("<image>", "").strip()
    prompt = re.sub(r"^\s*USER:\s*", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s*ASSISTANT:\s*$", "", prompt, flags=re.IGNORECASE)
    return clean_text(prompt)


def load_prompt_records(dataset_name: str, option: str):
    prompt_path = Path("prompts") / f"{dataset_name}_with_answer_{option}_options.jsonl"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    records = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_manual_bbox_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = clean_text(row.get("image_name", ""))
            if not image_name:
                continue
            rows.append(row)
    return rows


def draw_bboxes_on_image(image, bbox1=None, bbox2=None, line_width=5):
    """
    在图上只画框，不画文字标签。
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    if bbox1 is not None:
        draw.rectangle(bbox1, outline="red", width=line_width)

    if bbox2 is not None:
        draw.rectangle(bbox2, outline="blue", width=line_width)

    return img


def build_repo_prompt_from_stem(stem):
    """
    和仓库里原始风格保持一致：
    <image>
    USER: ...
    ASSISTANT:
    """
    stem = clean_text(stem)
    if re.search(r"answer with .* only\.?$", stem, flags=re.IGNORECASE):
        question = stem
    else:
        if stem.endswith("?"):
            question = stem + " Answer with left, right, on or under."
        else:
            question = stem + " Answer with left, right, on or under."
    return f"<image>\nUSER: {question}\nASSISTANT:"


def main():
    args = parse_args()
    seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    prompt_records = load_prompt_records(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)

    if len(prompt_records) != len(dataset):
        raise ValueError(
            f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)})."
        )

    # repo strict: 用仓库自己的 llava1.5 wrapper
    model = get_model(
        args.model_name,
        method=args.method,
    )

    manual_rows = load_manual_bbox_rows(args.manual_bbox_csv)

    # image_name -> manual bbox row
    manual_map = {}
    for row in manual_rows:
        image_name = clean_text(row.get("image_name", ""))
        if image_name:
            manual_map[image_name] = row

    # 只挑 manual_bboxes.csv 里有的图片
    selected_indices = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        image_name = clean_text(item.get("image_name", f"sample_{idx:04d}"))
        if image_name in manual_map:
            selected_indices.append(idx)

    if args.limit >= 0:
        selected_indices = selected_indices[args.start_index: args.start_index + args.limit]
    else:
        selected_indices = selected_indices[args.start_index:]

    out_root = os.path.join(args.out_dir, args.dataset, args.model_name)
    os.makedirs(out_root, exist_ok=True)

    summary_csv = os.path.join(out_root, "summary_manual_bbox_image_compare.csv")
    report_json = os.path.join(out_root, "report.json")

    summary_rows = []

    for run_i, idx in enumerate(selected_indices):
        item = dataset[idx]
        rec = prompt_records[idx]

        image = item["image_options"][0]
        image_name = clean_text(item.get("image_name", f"sample_{idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image_stem = os.path.splitext(image_name)[0]

        sample_dir = os.path.join(out_root, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        compare_json_path = os.path.join(sample_dir, "compare.json")

        if args.skip_existing and os.path.exists(compare_json_path):
            continue

        manual_row = manual_map[image_name]
        bbox1 = parse_bbox(manual_row.get("object_1_bbox", ""))
        bbox2 = parse_bbox(manual_row.get("object_2_bbox", ""))

        obj1_name = clean_text(manual_row.get("object_1_name", "object_1"))
        obj2_name = clean_text(manual_row.get("object_2_name", "object_2"))

        stem = strip_legacy_prompt(rec["question"])
        baseline_prompt = build_repo_prompt_from_stem(stem)

        # bbox-image 版本：prompt 完全一样，只是图片上画了框
        bbox_image_prompt = baseline_prompt

        annotated_image = draw_bboxes_on_image(
            image=image,
            bbox1=bbox1,
            bbox2=bbox2,
            line_width=args.line_width,
        )

        print("=" * 100)
        print(f"image_name: {image_name}")
        print("[BASELINE PROMPT]")
        print(baseline_prompt)
        print("[BBOX-IMAGE PROMPT]")
        print(bbox_image_prompt)

        # baseline
        baseline_text = model.run_single_prompt(
            image,
            baseline_prompt,
        )
        baseline_pred = parse_prediction(baseline_text)

        # bbox-image
        bbox_image_text = model.run_single_prompt(
            annotated_image,
            bbox_image_prompt,
        )
        bbox_image_pred = parse_prediction(bbox_image_text)

        gold = infer_gold_from_image_name(image_name)

        baseline_correct = (baseline_pred == gold)
        bbox_image_correct = (bbox_image_pred == gold)

        delta = int(bbox_image_correct) - int(baseline_correct)

        print(f"gold={gold} | baseline_pred={baseline_pred} | bbox_image_pred={bbox_image_pred}")

        compare_record = {
            "image_name": image_name,
            "image_path": image_path,
            "dataset_index": idx,
            "object_1_name": obj1_name,
            "object_2_name": obj2_name,
            "object_1_bbox": bbox1,
            "object_2_bbox": bbox2,
            "gold": gold,
            "baseline_prompt": baseline_prompt,
            "bbox_image_prompt": bbox_image_prompt,
            "baseline_text": baseline_text,
            "bbox_image_text": bbox_image_text,
            "baseline_pred": baseline_pred,
            "bbox_image_pred": bbox_image_pred,
            "baseline_correct": baseline_correct,
            "bbox_image_correct": bbox_image_correct,
            "delta": delta,
        }

        with open(compare_json_path, "w", encoding="utf-8") as f:
            json.dump(compare_record, f, indent=2, ensure_ascii=False)

        summary_rows.append({
            "image_name": image_name,
            "image_path": image_path,
            "dataset_index": idx,
            "object_1_name": obj1_name,
            "object_2_name": obj2_name,
            "object_1_bbox": bbox_to_json(bbox1),
            "object_2_bbox": bbox_to_json(bbox2),
            "gold": gold,
            "baseline_pred": baseline_pred,
            "bbox_image_pred": bbox_image_pred,
            "baseline_correct": baseline_correct,
            "bbox_image_correct": bbox_image_correct,
            "delta": delta,
            "baseline_prompt": baseline_prompt,
            "bbox_image_prompt": bbox_image_prompt,
            "baseline_text": baseline_text,
            "bbox_image_text": bbox_image_text,
            "compare_json": os.path.relpath(compare_json_path, out_root),
        })

        fieldnames = [
            "image_name",
            "image_path",
            "dataset_index",
            "object_1_name",
            "object_2_name",
            "object_1_bbox",
            "object_2_bbox",
            "gold",
            "baseline_pred",
            "bbox_image_pred",
            "baseline_correct",
            "bbox_image_correct",
            "delta",
            "baseline_prompt",
            "bbox_image_prompt",
            "baseline_text",
            "bbox_image_text",
            "compare_json",
        ]

        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    # 汇总
    n = len(summary_rows)
    baseline_acc = sum(int(r["baseline_correct"]) for r in summary_rows) / n if n > 0 else 0.0
    bbox_image_acc = sum(int(r["bbox_image_correct"]) for r in summary_rows) / n if n > 0 else 0.0

    report = {
        "num_examples": n,
        "baseline_acc": baseline_acc,
        "bbox_image_acc": bbox_image_acc,
        "acc_delta": bbox_image_acc - baseline_acc,
        "num_improved": sum(1 for r in summary_rows if int(r["delta"]) == 1),
        "num_worsened": sum(1 for r in summary_rows if int(r["delta"]) == -1),
        "num_unchanged": sum(1 for r in summary_rows if int(r["delta"]) == 0),
        "summary_csv": summary_csv,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved report to: {report_json}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
