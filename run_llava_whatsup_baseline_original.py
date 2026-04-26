import os
import re
import csv
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


# =========================================================
# args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="./output_llava_whatsup_baseline_original", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--print-first-n", default=10, type=int)
    return p.parse_args()


# =========================================================
# utils
# =========================================================
def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def extract_raw_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for k in ["response", "text", "pred_text", "output", "answer"]:
            if k in output:
                return str(output[k])
        return str(output)
    if isinstance(output, (list, tuple)) and len(output) > 0:
        return str(output[0])
    return str(output)


def normalize_rel(answer: Any) -> Optional[str]:
    """
    Map dataset answers / model outputs into:
    left / right / under / on
    """
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            return None
        answer = answer[0]

    if answer is None:
        return None

    rel = clean_text(str(answer)).lower()

    mapping = {
        "left": "left",
        "left of": "left",
        "to the left of": "left",

        "right": "right",
        "right of": "right",
        "to the right of": "right",

        "under": "under",
        "below": "under",
        "beneath": "under",

        "on": "on",
        "on top of": "on",
        "top": "on",
        "above": "on",
    }

    return mapping.get(rel, None)


def relation_from_image_name(image_name: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    if "_left_of_" in stem:
        return "left"
    if "_right_of_" in stem:
        return "right"
    if "_on_" in stem:
        return "on"
    if "_under_" in stem:
        return "under"
    return None


def parse_prediction(text: str) -> str:
    """
    Parse the model's raw output into:
    left / right / under / on / UNK

    We take the last valid relation mention.
    """
    t = clean_text(text).lower()

    phrase_patterns = [
        (r"\bto the left of\b", "left"),
        (r"\bleft of\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bright of\b", "right"),
        (r"\bon top of\b", "on"),
        (r"\bunder\b", "under"),
        (r"\bbelow\b", "under"),
        (r"\bbeneath\b", "under"),
        (r"\bon the\b", "on"),
    ]

    found = []
    for pat, lab in phrase_patterns:
        for m in re.finditer(pat, t):
            found.append((m.start(), m.end(), lab))

    if found:
        found.sort(key=lambda x: (x[0], x[1]))
        return found[-1][2]

    single_patterns = [
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bon\b", "on"),
    ]
    found = []
    for pat, lab in single_patterns:
        for m in re.finditer(pat, t):
            found.append((m.start(), m.end(), lab))

    if found:
        found.sort(key=lambda x: (x[0], x[1]))
        return found[-1][2]

    return "UNK"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_summary_fieldnames() -> List[str]:
    return [
        "image_name",
        "image_path",
        "local_index",
        "gold_raw",
        "gold_norm",
        "pred",
        "correct",
        "raw_question",
        "raw_output",
    ]


def print_stats(name: str, total: int, correct: int,
                per_gold_total: Dict[str, int],
                per_gold_correct: Dict[str, int]):
    print("=" * 120)
    print(f"[{name}]")
    print(f"total = {total}")
    print(f"correct = {correct}")
    print(f"overall_acc = {correct / total:.4f}" if total > 0 else "overall_acc = N/A")
    for rel in ["left", "right", "under", "on"]:
        n = per_gold_total[rel]
        c = per_gold_correct[rel]
        acc = 0.0 if n == 0 else c / n
        print(f"{rel}_acc = {acc:.4f} ({c}/{n})")


def write_summary_txt(
    out_path: str,
    first_rows: List[Dict[str, Any]],
    report: Dict[str, Any],
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("LLaVA What’sUp baseline summary (original question, no prompt rewrite)\n")
        f.write("=" * 100 + "\n\n")

        f.write("First printed examples\n")
        f.write("-" * 100 + "\n")
        for row in first_rows:
            f.write(f"local_index: {row['local_index']}\n")
            f.write(f"image_name: {row['image_name']}\n")
            f.write(f"gold_raw: {row['gold_raw']}\n")
            f.write(f"gold_norm: {row['gold_norm']}\n")
            f.write(f"pred: {row['pred']} | correct: {row['correct']}\n")
            f.write(f"raw_question: {row['raw_question']}\n")
            f.write(f"raw_output: {row['raw_output']}\n")
            f.write("-" * 100 + "\n")

        f.write("\nOverall metrics\n")
        f.write("-" * 100 + "\n")
        f.write(f"total = {report['total']}\n")
        f.write(f"correct = {report['correct']}\n")
        f.write(f"overall_acc = {report['overall_acc']:.4f}\n")
        for rel in ["left", "right", "under", "on"]:
            f.write(
                f"{rel}_acc = {report[f'{rel}_acc']:.4f} "
                f"({report[f'{rel}_correct']}/{report[f'{rel}_total']})\n"
            )

def run_repo_llava_once(wrapper, image, prompt: str) -> str:
    return extract_raw_text(
        wrapper.run_single_prompt(
            image=image,
            prompt=prompt,
            method="base",
            weight=None,
        )
    ).strip()


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    seed_all(args.seed)

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    ensure_dir(cache_dir)
    ensure_dir(args.out_dir)

    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    print("Loading LLaVA with repo get_model(...)")
    wrapper, image_preprocess = get_model(
        args.model_name,
        args.device,
        method=args.method,
        root_dir=cache_dir,
    )

    print(f"Loading dataset: {args.dataset} (download={args.download})")
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        import torch
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    if len(prompt_records) != len(dataset):
        raise ValueError(f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).")

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    out_root = os.path.join(out_dir := args.out_dir, args.dataset, "llava1.5_repo_baseline_original_question")
    ensure_dir(out_root)

    summary_csv = os.path.join(out_root, "summary_baseline_original_question.csv")
    summary_txt = os.path.join(out_root, "summary_baseline_original_question.txt")
    report_json = os.path.join(out_root, "report_baseline_original_question.json")

    rows: List[Dict[str, Any]] = []
    first_rows: List[Dict[str, Any]] = []

    total = 0
    correct = 0
    per_gold_total = defaultdict(int)
    per_gold_correct = defaultdict(int)

    shown = 0

    for local_idx in tqdm(range(start, end), desc="examples"):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image = item["image_options"][0].convert("RGB")

        raw_question = rec["question"]
        gold_raw = rec.get("answer", None)
        gold = normalize_rel(gold_raw)
        if gold is None:
            gold = relation_from_image_name(image_name)
        if gold is None:
            continue

        raw_output = run_repo_llava_once(
            wrapper=wrapper,
            image=image,
            prompt=raw_question,   # 原题目原样跑
        )
        pred = parse_prediction(raw_output)
        is_correct = (pred == gold)

        row = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "gold_raw": clean_text(gold_raw),
            "gold_norm": gold,
            "pred": pred,
            "correct": is_correct,
            "raw_question": clean_text(raw_question),
            "raw_output": raw_output,
        }
        rows.append(row)

        total += 1
        per_gold_total[gold] += 1
        if is_correct:
            correct += 1
            per_gold_correct[gold] += 1

        if shown < args.print_first_n:
            print("=" * 120)
            print(f"idx={local_idx}")
            print(f"image_name: {image_name}")
            print(f"gold_raw={clean_text(gold_raw)}")
            print(f"gold_norm={gold}")
            print(f"[RAW QUESTION] {clean_text(raw_question)}")
            print(f"[RAW OUTPUT] {raw_output}")
            print(f"[PRED] {pred}")
            first_rows.append(row)
            shown += 1

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=build_summary_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "dataset": args.dataset,
        "total": total,
        "correct": correct,
        "overall_acc": 0.0 if total == 0 else correct / total,
    }
    for rel in ["left", "right", "under", "on"]:
        n = per_gold_total[rel]
        c = per_gold_correct[rel]
        report[f"{rel}_total"] = n
        report[f"{rel}_correct"] = c
        report[f"{rel}_acc"] = 0.0 if n == 0 else c / n

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_summary_txt(
        out_path=summary_txt,
        first_rows=first_rows,
        report=report,
    )

    print(f"Saved summary csv to: {summary_csv}")
    print(f"Saved summary txt to: {summary_txt}")
    print(f"Saved report json to: {report_json}")

    print_stats(
        f"BASELINE_ORIGINAL_QUESTION_{args.dataset}",
        total,
        correct,
        per_gold_total,
        per_gold_correct,
    )


if __name__ == "__main__":
    main()
