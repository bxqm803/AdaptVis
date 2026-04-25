import os
import re
import csv
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


CHOICES = ["left", "right", "under", "on"]


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
    p.add_argument("--out-dir", default="./output_llava_mc5_baseline_only", type=str)
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


def clean_question_text(question: str) -> str:
    q = str(question).strip().replace("\n", " ")
    if "<image>" in q:
        q = q.replace("<image>", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def strip_existing_answer_clause(question_text: str) -> str:
    q = clean_question_text(question_text)

    patterns = [
        r"Answer with\s+left,\s*right,\s*on\s+or\s+under(?:\s+only)?\.?\s*$",
        r"Answer with\s+only\s+one\s+word:.*$",
        r"Output exactly one label from:.*$",
        r"Choose one option only\..*$",
        r"Which of the following best describes.*$",
        r"Select the correct relation.*$",
        r"Identify the spatial relation.*$",
    ]
    for pat in patterns:
        q = re.sub(pat, "", q, flags=re.IGNORECASE)

    q = re.sub(r"\s+", " ", q).strip()
    return q


def normalize_rel(answer: Any) -> Optional[str]:
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            return None
        answer = answer[0]

    if answer is None:
        return None

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


def parse_prediction_generic(text: str) -> str:
    t = clean_text(text).lower()

    phrase_patterns = [
        (r"\bto the left of\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bon top of\b", "on"),
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


def parse_prediction_mc(raw_output: str, prompt_id: int) -> str:
    t = clean_text(raw_output)

    # Prompt 3: A/B/C/D format
    if prompt_id == 3:
        m = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
        if m:
            ch = m.group(1).upper()
            return {
                "A": "left",
                "B": "right",
                "C": "under",
                "D": "on",
            }[ch]

    # fallback: parse natural-language output or direct label word
    return parse_prediction_generic(t)


def build_mc_prompts(base_question: str) -> Dict[int, str]:
    """
    Uses only the 5 multiple-choice / closed-answer variants.
    """
    stem = strip_existing_answer_clause(base_question)
    if not stem.endswith("?"):
        stem = stem.rstrip(".") + "?"

    prompts = {
        1: f"<image> USER: {stem} Answer with only one word: left, right, under, or on. ASSISTANT:",
        2: f"<image> USER: {stem} Output exactly one label from: left, right, under, on. Do not output anything else. ASSISTANT:",
        3: f"<image> USER: {stem} Choose one option only. A. left B. right C. under D. on. Answer with only A, B, C, or D. ASSISTANT:",
        4: f"<image> USER: Which of the following best describes the relation asked in this question? {stem} Options: left, right, under, on. Answer with only one word. ASSISTANT:",
        5: f"<image> USER: Select the correct relation for this question: {stem} Choices: left, right, under, on. Respond with only the chosen word. ASSISTANT:",
    }
    return prompts


def run_repo_llava_once(wrapper, image, prompt: str) -> str:
    return extract_raw_text(
        wrapper.run_single_prompt(
            image=image,
            prompt=prompt,
            method="base",
            weight=None,
        )
    ).strip()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_summary_fieldnames() -> List[str]:
    return [
        "template_id",
        "template_text",
        "image_name",
        "image_path",
        "local_index",
        "gold",
        "pred",
        "correct",
        "raw_question",
        "prompt_text",
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
    prompt_templates: Dict[int, str],
    first10_rows: List[Dict[str, Any]],
    reports: Dict[int, Dict[str, Any]],
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("LLaVA multiple-choice baseline summary\n")
        f.write("=" * 100 + "\n\n")

        f.write("Prompt templates\n")
        f.write("-" * 100 + "\n")
        for pid in sorted(prompt_templates):
            f.write(f"[Template {pid}]\n{prompt_templates[pid]}\n\n")

        f.write("\nFirst 10 examples x 5 prompts\n")
        f.write("-" * 100 + "\n")
        for row in first10_rows:
            f.write(f"local_index: {row['local_index']}\n")
            f.write(f"template_id: {row['template_id']}\n")
            f.write(f"image_name: {row['image_name']}\n")
            f.write(f"gold: {row['gold']} | pred: {row['pred']} | correct: {row['correct']}\n")
            f.write(f"raw_question: {row['raw_question']}\n")
            f.write(f"prompt_text: {row['prompt_text']}\n")
            f.write(f"raw_output: {row['raw_output']}\n")
            f.write("-" * 100 + "\n")

        f.write("\nPer-template metrics\n")
        f.write("-" * 100 + "\n")
        for pid in sorted(reports):
            r = reports[pid]
            f.write(f"[Template {pid}]\n")
            f.write(f"total = {r['total']}\n")
            f.write(f"correct = {r['correct']}\n")
            f.write(f"overall_acc = {r['overall_acc']:.4f}\n")
            for rel in ["left", "right", "under", "on"]:
                f.write(
                    f"{rel}_acc = {r[f'{rel}_acc']:.4f} "
                    f"({r[f'{rel}_correct']}/{r[f'{rel}_total']})\n"
                )
            f.write("\n")


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

    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    # Strictly use repo's prompt-record loading path
    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        import torch
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    if len(prompt_records) != len(dataset):
        raise ValueError(f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).")

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    out_root = os.path.join(args.out_dir, args.dataset, "llava1.5_repo_mc5_baseline")
    ensure_dir(out_root)

    summary_csv = os.path.join(out_root, "summary_mc5_baseline.csv")
    summary_txt = os.path.join(out_root, "summary_mc5_baseline.txt")
    report_json = os.path.join(out_root, "report_mc5_baseline.json")

    rows: List[Dict[str, Any]] = []
    first10_rows: List[Dict[str, Any]] = []

    # stats per template
    total_by_template = defaultdict(int)
    correct_by_template = defaultdict(int)
    per_gold_total = {pid: defaultdict(int) for pid in range(1, 6)}
    per_gold_correct = {pid: defaultdict(int) for pid in range(1, 6)}

    shown = 0
    prompt_templates_cache = None

    for local_idx in tqdm(range(start, end), desc="examples"):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image = item["image_options"][0].convert("RGB")

        raw_question = rec["question"]
        gold = normalize_rel(rec.get("answer", None))
        if gold is None:
            gold = relation_from_image_name(image_name)
        if gold is None:
            continue

        prompt_templates = build_mc_prompts(raw_question)
        prompt_templates_cache = prompt_templates

        for template_id, prompt_text in prompt_templates.items():
            raw_output = run_repo_llava_once(wrapper, image, prompt_text)
            pred = parse_prediction_mc(raw_output, template_id)
            correct = (pred == gold)

            row = {
                "template_id": template_id,
                "template_text": prompt_text,
                "image_name": image_name,
                "image_path": image_path,
                "local_index": local_idx,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "raw_question": clean_question_text(raw_question),
                "prompt_text": prompt_text,
                "raw_output": raw_output,
            }
            rows.append(row)

            total_by_template[template_id] += 1
            per_gold_total[template_id][gold] += 1
            if correct:
                correct_by_template[template_id] += 1
                per_gold_correct[template_id][gold] += 1

            if local_idx < start + args.print_first_n:
                first10_rows.append(row)

        if shown < args.print_first_n:
            print("=" * 120)
            print(f"idx={local_idx}")
            print(f"image_name: {image_name}")
            print(f"gold={gold}")
            for template_id in range(1, 6):
                last_row = rows[-(6 - template_id)]
                print(f"[TEMPLATE {template_id} PRED] {last_row['pred']}")
                print(f"[TEMPLATE {template_id} RAW OUTPUT] {last_row['raw_output']}")
            shown += 1

    # save csv
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=build_summary_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    # build reports
    reports: Dict[int, Dict[str, Any]] = {}
    for pid in range(1, 6):
        total = total_by_template[pid]
        correct = correct_by_template[pid]
        report = {
            "template_id": pid,
            "template_text": prompt_templates_cache[pid] if prompt_templates_cache is not None else "",
            "total": total,
            "correct": correct,
            "overall_acc": 0.0 if total == 0 else correct / total,
        }
        for rel in ["left", "right", "under", "on"]:
            n = per_gold_total[pid][rel]
            c = per_gold_correct[pid][rel]
            report[f"{rel}_total"] = n
            report[f"{rel}_correct"] = c
            report[f"{rel}_acc"] = 0.0 if n == 0 else c / n
        reports[pid] = report

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    write_summary_txt(
        out_path=summary_txt,
        prompt_templates=prompt_templates_cache if prompt_templates_cache is not None else {},
        first10_rows=first10_rows,
        reports=reports,
    )

    print(f"Saved summary csv to: {summary_csv}")
    print(f"Saved summary txt to: {summary_txt}")
    print(f"Saved report json to: {report_json}")

    # print 5 prompts x 4 label acc
    for pid in range(1, 6):
        print_stats(
            f"TEMPLATE_{pid}",
            reports[pid]["total"],
            reports[pid]["correct"],
            {k: reports[pid][f"{k}_total"] for k in ["left", "right", "under", "on"]},
            {k: reports[pid][f"{k}_correct"] for k in ["left", "right", "under", "on"]},
        )


if __name__ == "__main__":
    main()
