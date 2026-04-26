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


# =========================================================
# Relation config
# =========================================================
RELATIONS_A = ["left", "right", "under", "on"]
RELATIONS_B = ["left", "right", "in-front", "behind"]


def get_relation_labels(dataset_name: str) -> List[str]:
    name = dataset_name.lower()
    if "controlled_images_b" in name:
        return RELATIONS_B
    # COCO_QA_two_obj and Controlled_Images_A usually use left/right/on/under.
    return RELATIONS_A


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
    p.add_argument("--out-dir", default="./output_llava_open5_allrels", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--print-per-class", default=3, type=int,
                   help="Print this many examples for each gold relation label.")
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
        r"Answer with\s+left,\s*right,\s*in[- ]front\s+or\s+behind(?:\s+only)?\.?\s*$",
        r"Answer with\s+only\s+one\s+word:.*$",
        r"Output exactly one label from:.*$",
        r"Choose one option only\..*$",
        r"Which of the following best describes.*$",
        r"Select the correct relation.*$",
        r"Identify the spatial relation.*$",
        r"Respond with only the chosen word\.?\s*$",
        r"Do not output anything else\.?\s*$",
    ]
    for pat in patterns:
        q = re.sub(pat, "", q, flags=re.IGNORECASE)

    q = re.sub(r"\s+", " ", q).strip()
    return q


def normalize_object_name(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"^(a|an|the)\s+", "", name)
    name = re.sub(r"[?.!,;:]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_objects_from_question(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = strip_existing_answer_clause(question)

    patterns = [
        r"Where is the (.+?) in relation to the (.+?)\?",
        r"Where are the (.+?) in relation to the (.+?)\?",
        r"Where is (.+?) in relation to (.+?)\?",
        r"Where are (.+?) in relation to (.+?)\?",
        r"What is the spatial relation between the (.+?) and the (.+?)\?",
        r"What is the relation between the (.+?) and the (.+?)\?",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            obj1 = normalize_object_name(m.group(1))
            obj2 = normalize_object_name(m.group(2))
            return obj1, obj2
    return None, None


def normalize_rel(answer: Any, valid_labels: Optional[List[str]] = None) -> Optional[str]:
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            return None
        answer = answer[0]

    if answer is None:
        return None

    rel = str(answer).strip().lower()
    rel = rel.replace("_", "-")
    rel = re.sub(r"\s+", " ", rel)

    mapping = {
        "left": "left",
        "left-of": "left",
        "left of": "left",
        "to the left of": "left",
        "right": "right",
        "right-of": "right",
        "right of": "right",
        "to the right of": "right",
        "on": "on",
        "top": "on",
        "above": "on",
        "on top of": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
        "underneath": "under",
        "in-front": "in-front",
        "in front": "in-front",
        "in front of": "in-front",
        "front": "in-front",
        "in-front-of": "in-front",
        "behind": "behind",
        "back": "behind",
        "in back of": "behind",
    }

    if rel in mapping:
        out = mapping[rel]
    elif "left" in rel:
        out = "left"
    elif "right" in rel:
        out = "right"
    elif "in front" in rel or "in-front" in rel:
        out = "in-front"
    elif "behind" in rel:
        out = "behind"
    elif "under" in rel or "below" in rel or "beneath" in rel:
        out = "under"
    elif "on top" in rel or re.search(r"\bon\b", rel) or "above" in rel:
        out = "on"
    else:
        out = None

    if valid_labels is not None and out not in valid_labels:
        return None
    return out


def relation_from_image_name(image_name: str, valid_labels: List[str]) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    stem = stem.replace("-", "_")

    candidates = []
    if "_left_of_" in stem:
        candidates.append("left")
    if "_right_of_" in stem:
        candidates.append("right")
    if "_on_" in stem:
        candidates.append("on")
    if "_under_" in stem:
        candidates.append("under")
    if "_in_front_of_" in stem or "_in_front_" in stem:
        candidates.append("in-front")
    if "_behind_" in stem:
        candidates.append("behind")

    for c in candidates:
        if c in valid_labels:
            return c
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


def parse_prediction_generic(text: str, valid_labels: List[str]) -> str:
    """Parse an open-ended output into one of valid_labels or UNK.

    The parser intentionally keeps the last relation mention, because many VLMs
    answer after restating candidate relations or correcting themselves.
    """
    t = clean_text(text).lower()
    t = t.replace("_", "-")

    phrase_patterns = [
        (r"\bto the left of\b", "left"),
        (r"\bleft of\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bright of\b", "right"),
        (r"\bin[- ]front of\b", "in-front"),
        (r"\bin[- ]front\b", "in-front"),
        (r"\bbehind\b", "behind"),
        (r"\bin back of\b", "behind"),
        (r"\bunderneath\b", "under"),
        (r"\bbeneath\b", "under"),
        (r"\bbelow\b", "under"),
        (r"\bunder\b", "under"),
        (r"\bon top of\b", "on"),
        (r"\bon the top of\b", "on"),
        (r"\babove\b", "on"),
        (r"\bon\b", "on"),
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
    ]

    found = []
    for pat, lab in phrase_patterns:
        if lab not in valid_labels:
            continue
        for m in re.finditer(pat, t):
            found.append((m.start(), m.end(), lab))

    if found:
        found.sort(key=lambda x: (x[0], x[1]))
        return found[-1][2]
    return "UNK"


def build_open5_prompts(base_question: str) -> Dict[int, str]:
    stem = strip_existing_answer_clause(base_question)
    if not stem.endswith("?"):
        stem = stem.rstrip(".") + "?"

    obj1, obj2 = parse_objects_from_question(base_question)
    if obj1 is None or obj2 is None:
        return {
            1: f"<image> USER: {stem} ASSISTANT:",
            2: "<image> USER: Describe the spatial relation in this image in one short sentence. ASSISTANT:",
            3: "<image> USER: What is the relation in this image? Answer naturally. ASSISTANT:",
            4: "<image> USER: State the relation in a short phrase. ASSISTANT:",
            5: "<image> USER: Describe how the two queried objects are positioned relative to each other. ASSISTANT:",
        }

    return {
        1: f"<image> USER: Where is the {obj1} in relation to the {obj2}? ASSISTANT:",
        2: f"<image> USER: Describe the spatial relation between the {obj1} and the {obj2} in one short sentence. ASSISTANT:",
        3: f"<image> USER: What is the relation between the {obj1} and the {obj2}? Answer naturally. ASSISTANT:",
        4: f"<image> USER: State where the {obj1} is relative to the {obj2} in a short phrase. ASSISTANT:",
        5: f"<image> USER: Describe how the {obj1} is positioned relative to the {obj2}. ASSISTANT:",
    }


def run_repo_llava_once(wrapper, image, prompt: str, method: str) -> str:
    return extract_raw_text(
        wrapper.run_single_prompt(
            image=image,
            prompt=prompt,
            method=method,
            weight=None,
        )
    ).strip()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_summary_fieldnames() -> List[str]:
    return [
        "template_id",
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


def print_stats(name: str, total: int, correct: int, pred_counter: Dict[str, int], labels: List[str]):
    print("=" * 120)
    print(f"[{name}]")
    print(f"total = {total}")
    print(f"correct = {correct}")
    print(f"overall_acc = {correct / total:.4f}" if total > 0 else "overall_acc = N/A")
    print("prediction_count:")
    for lab in labels + ["UNK"]:
        print(f"  {lab}: {pred_counter[lab]}")


def write_summary_txt(
    out_path: str,
    labels: List[str],
    printed_examples: List[Dict[str, Any]],
    reports: Dict[int, Dict[str, Any]],
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("LLaVA open-ended 5-template baseline summary, all relations\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"relation_labels: {labels}\n\n")

        f.write("Printed examples\n")
        f.write("-" * 100 + "\n")
        for row in printed_examples:
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
            f.write("prediction_count:\n")
            for lab in labels + ["UNK"]:
                f.write(f"  {lab}: {r['pred_counter'][lab]}\n")
            f.write("per_gold_acc:\n")
            for lab in labels:
                g_total = r["per_gold_total"][lab]
                g_correct = r["per_gold_correct"][lab]
                acc = None if g_total == 0 else g_correct / g_total
                f.write(f"  {lab}: {g_correct}/{g_total} acc={acc if acc is not None else 'N/A'}\n")
            f.write("\n")


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    seed_all(args.seed)

    labels = get_relation_labels(args.dataset)
    print(f"Using relation labels for {args.dataset}: {labels}")

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

    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        import torch
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    if len(prompt_records) != len(dataset):
        raise ValueError(f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).")

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    run_name = f"{args.model_name}_repo_open5_allrels_{args.method}"
    out_root = os.path.join(args.out_dir, args.dataset, run_name)
    ensure_dir(out_root)

    summary_csv = os.path.join(out_root, "summary_open5_allrels.csv")
    summary_txt = os.path.join(out_root, "summary_open5_allrels.txt")
    report_json = os.path.join(out_root, "report_open5_allrels.json")

    rows: List[Dict[str, Any]] = []
    printed_examples: List[Dict[str, Any]] = []

    total_by_template = defaultdict(int)
    correct_by_template = defaultdict(int)
    pred_counter_by_template = {pid: defaultdict(int) for pid in range(1, 6)}
    per_gold_total = {pid: defaultdict(int) for pid in range(1, 6)}
    per_gold_correct = {pid: defaultdict(int) for pid in range(1, 6)}

    # Print 3 examples for each gold relation by default.
    printed_per_gold = defaultdict(int)

    skipped_no_gold = 0

    for local_idx in tqdm(range(start, end), desc="examples"):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image = item["image_options"][0].convert("RGB")

        raw_question = rec.get("question", "")
        gold = normalize_rel(rec.get("answer", None), labels)
        if gold is None:
            gold = relation_from_image_name(image_name, labels)

        if gold is None:
            skipped_no_gold += 1
            continue

        prompt_templates = build_open5_prompts(raw_question)
        example_rows = []

        for template_id, prompt_text in prompt_templates.items():
            raw_output = run_repo_llava_once(wrapper, image, prompt_text, args.method)
            pred = parse_prediction_generic(raw_output, labels)
            correct = (pred == gold)

            row = {
                "template_id": template_id,
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
            example_rows.append(row)

            total_by_template[template_id] += 1
            pred_counter_by_template[template_id][pred] += 1
            per_gold_total[template_id][gold] += 1
            if correct:
                correct_by_template[template_id] += 1
                per_gold_correct[template_id][gold] += 1

        if printed_per_gold[gold] < args.print_per_class:
            print("=" * 120)
            print(f"[PRINT EXAMPLE gold={gold} #{printed_per_gold[gold] + 1}/{args.print_per_class}]")
            print(f"idx={local_idx}")
            print(f"image_name: {image_name}")
            print(f"gold={gold}")
            print(f"[RAW QUESTION] {clean_question_text(raw_question)}")
            for row in example_rows:
                print("-" * 80)
                print(f"[TEMPLATE {row['template_id']}]")
                print(f"[PROMPT] {row['prompt_text']}")
                print(f"[RAW OUTPUT] {row['raw_output']}")
                print(f"[PRED] {row['pred']}")
                printed_examples.append(row)
            printed_per_gold[gold] += 1

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=build_summary_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    reports: Dict[int, Dict[str, Any]] = {}
    for pid in range(1, 6):
        total = total_by_template[pid]
        correct = correct_by_template[pid]
        pred_counter = {lab: pred_counter_by_template[pid][lab] for lab in labels + ["UNK"]}
        report = {
            "template_id": pid,
            "total": total,
            "correct": correct,
            "overall_acc": 0.0 if total == 0 else correct / total,
            "pred_counter": pred_counter,
            "per_gold_total": {lab: per_gold_total[pid][lab] for lab in labels},
            "per_gold_correct": {lab: per_gold_correct[pid][lab] for lab in labels},
        }
        reports[pid] = report

    full_report = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "method": args.method,
        "relation_labels": labels,
        "num_rows": len(rows),
        "skipped_no_gold": skipped_no_gold,
        "printed_per_gold": {lab: printed_per_gold[lab] for lab in labels},
        "templates": reports,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    write_summary_txt(
        out_path=summary_txt,
        labels=labels,
        printed_examples=printed_examples,
        reports=reports,
    )

    print(f"Saved summary csv to: {summary_csv}")
    print(f"Saved summary txt to: {summary_txt}")
    print(f"Saved report json to: {report_json}")
    print(f"skipped_no_gold = {skipped_no_gold}")

    for pid in range(1, 6):
        print_stats(
            f"OPEN_TEMPLATE_{pid}_ALL_RELATIONS",
            reports[pid]["total"],
            reports[pid]["correct"],
            reports[pid]["pred_counter"],
            labels,
        )
        print("per_gold_acc:")
        for lab in labels:
            g_total = reports[pid]["per_gold_total"][lab]
            g_correct = reports[pid]["per_gold_correct"][lab]
            acc = "N/A" if g_total == 0 else f"{g_correct / g_total:.4f}"
            print(f"  {lab}: {g_correct}/{g_total} acc={acc}")


if __name__ == "__main__":
    main()
