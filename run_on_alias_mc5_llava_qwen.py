# run_on_alias_mc5_llava_qwen.py
import os
import re
import csv
import json
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from PIL import Image

from misc import seed_all
from dataset_zoo import get_dataset


# =========================================================
# args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--dataset", default="Controlled_Images_A", type=str,
                   choices=["Controlled_Images_A", "COCO_QA_two_obj"])
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--download", action="store_true")

    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="./output_on_alias_mc5", type=str)

    # llava1.5 / qwen-vl-chat
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--model-id", default=None, type=str)
    p.add_argument("--method", default="base", type=str)

    # alias to replace the original "on" option
    p.add_argument(
        "--on-alias",
        default="above",
        choices=["above", "at_the_top_of"],
        help="Replace the MC option 'on' with this phrase."
    )

    p.add_argument("--max-new-tokens", default=32, type=int)
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--print-first-n", default=10, type=int)

    return p.parse_args()


# =========================================================
# text utils
# =========================================================
def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()


def clean_question_text(question: str) -> str:
    q = str(question).strip().replace("\n", " ")
    q = q.replace("<image>", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    return re.sub(r"\s+", " ", q).strip()


def strip_existing_answer_clause(question_text: str) -> str:
    q = clean_question_text(question_text)

    patterns = [
        r"Answer with\s+left,\s*right,\s*on\s+or\s+under(?:\s+only)?\.?\s*$",
        r"Answer with\s+left,\s*right,\s*above\s+or\s+below(?:\s+only)?\.?\s*$",
        r"Answer with\s+left,\s*right,\s*under\s+or\s+on(?:\s+only)?\.?\s*$",
        r"Answer with\s+only\s+one\s+word:.*$",
        r"Output exactly one label from:.*$",
        r"Choose one option only\..*$",
        r"Options?:.*$",
        r"Choices?:.*$",
        r"Respond with only.*$",
        r"Do not output anything else\.?\s*$",
    ]

    for pat in patterns:
        q = re.sub(pat, "", q, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", q).strip()


def normalize_rel(answer: Any) -> Optional[str]:
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if len(answer) > 0 else None

    if answer is None:
        return None

    s = clean_text(str(answer)).lower().replace("_", " ")

    if re.search(r"\bto the left of\b|\bleft of\b|\bleft\b", s):
        return "left"
    if re.search(r"\bto the right of\b|\bright of\b|\bright\b", s):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b|\bunderneath\b", s):
        return "under"
    if re.search(r"\bon top of\b|\bat the top of\b|\babove\b|\bover\b|\batop\b|\bon\b", s):
        return "on"

    return None


def relation_from_image_name(image_name: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()

    if "_left_of_" in stem:
        return "left"
    if "_right_of_" in stem:
        return "right"
    if "_under_" in stem:
        return "under"
    if "_on_" in stem:
        return "on"

    return None


def alias_text(args) -> str:
    if args.on_alias == "above":
        return "above"
    if args.on_alias == "at_the_top_of":
        return "at the top of"
    raise ValueError(args.on_alias)


def valid_labels_with_alias(args) -> List[str]:
    # Keep original gold labels internally:
    # left / right / under / on
    # But display the fourth option as alias.
    return ["left", "right", "under", alias_text(args)]


def canonicalize_pred_label(label: str) -> str:
    s = clean_text(label).lower().replace("_", " ")

    if s in {"left"}:
        return "left"
    if s in {"right"}:
        return "right"
    if s in {"under", "below", "beneath", "underneath"}:
        return "under"
    if s in {"on", "above", "over", "atop", "on top of", "at the top of"}:
        return "on"

    if re.search(r"\bleft\b", s):
        return "left"
    if re.search(r"\bright\b", s):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b|\bunderneath\b", s):
        return "under"
    if re.search(r"\bon top of\b|\bat the top of\b|\babove\b|\bover\b|\batop\b|\bon\b", s):
        return "on"

    return "UNK"


def parse_prediction_generic(text: str) -> str:
    t = clean_text(text).lower().replace("_", " ")

    # longer phrases first
    patterns = [
        (r"\bat the top of\b", "on"),
        (r"\bon top of\b", "on"),
        (r"\bto the left of\b", "left"),
        (r"\bleft of\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bright of\b", "right"),
        (r"\bunderneath\b", "under"),
        (r"\bbeneath\b", "under"),
        (r"\bbelow\b", "under"),
        (r"\bunder\b", "under"),
        (r"\babove\b", "on"),
        (r"\bover\b", "on"),
        (r"\batop\b", "on"),
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bon\b", "on"),
    ]

    found = []
    for pat, lab in patterns:
        for m in re.finditer(pat, t):
            found.append((m.start(), m.end(), lab))

    if found:
        found.sort(key=lambda x: (x[0], x[1]))
        return found[-1][2]

    return "UNK"


def parse_prediction_mc(raw_output: str, template_id: int, args) -> str:
    t = clean_text(raw_output)
    display_labels = valid_labels_with_alias(args)

    # Template 3 uses A/B/C/D.
    # D corresponds to alias, but canonical label is still "on".
    if template_id == 3:
        m = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < len(display_labels):
                return canonicalize_pred_label(display_labels[idx])

    return parse_prediction_generic(t)


# =========================================================
# prompts
# =========================================================
def build_mc_prompts_on_alias(base_question: str, args) -> Dict[int, str]:
    stem = strip_existing_answer_clause(base_question)
    if not stem.endswith("?"):
        stem = stem.rstrip(".") + "?"

    a, b, c, d = valid_labels_with_alias(args)
    label_list = ", ".join([a, b, c, d])

    return {
        1: f"{stem} Answer with only one word: {a}, {b}, {c}, or {d}.",
        2: f"{stem} Output exactly one label from: {label_list}. Do not output anything else.",
        3: f"{stem} Choose one option only. A. {a} B. {b} C. {c} D. {d}. Answer with only A, B, C, or D.",
        4: f"Which of the following best describes the relation asked in this question? {stem} Options: {label_list}. Answer with only one word.",
        5: f"Select the correct relation for this question: {stem} Choices: {label_list}. Respond with only the chosen word.",
    }


def format_prompt_for_llava(prompt_text: str) -> str:
    return f"<image> USER: {prompt_text} ASSISTANT:"


# =========================================================
# prompt records
# =========================================================
def load_prompt_records(dataset_name: str, option: str) -> List[Dict[str, Any]]:
    prompt_path = Path("prompts") / f"{dataset_name}_with_answer_{option}_options.jsonl"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    records = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


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


# =========================================================
# model runners
# =========================================================
class BaseRunner:
    def __init__(self, args):
        self.args = args
        self.image_preprocess = None

    def generate(self, image: Image.Image, image_path: str, prompt_text: str) -> str:
        raise NotImplementedError


class LlavaRepoRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        from model_zoo import get_model

        print("Loading LLaVA with repo get_model(...)")
        self.wrapper, self.image_preprocess = get_model(
            args.model_name,
            args.device,
            method=args.method,
            root_dir=args.cache_dir,
        )

    def load_prompt_records_with_sampling(self, dataset_name: str, option: str):
        return self.wrapper.load_prompt_records_with_sampling(dataset_name, option)

    def generate(self, image: Image.Image, image_path: str, prompt_text: str) -> str:
        out = self.wrapper.run_single_prompt(
            image=image,
            prompt=format_prompt_for_llava(prompt_text),
            method=self.args.method,
            weight=None,
        )
        return extract_raw_text(out).strip()


class QwenVLChatRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = args.model_id or "Qwen/Qwen-VL-Chat"
        print(f"Loading Qwen-VL-Chat: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=args.cache_dir,
            device_map="auto" if args.device.startswith("cuda") else None,
            trust_remote_code=True,
            fp16=args.device.startswith("cuda"),
        ).eval()

        if not args.device.startswith("cuda"):
            self.model.to(args.device)

        self._tmpdir = tempfile.TemporaryDirectory(prefix="qwen_vl_images_")
        self._counter = 0

    def _ensure_image_path(self, image: Image.Image, image_path: str) -> str:
        if image_path and os.path.exists(image_path):
            return image_path

        self._counter += 1
        path = os.path.join(self._tmpdir.name, f"qwen_img_{self._counter}.jpg")
        image.save(path)
        return path

    @torch.no_grad()
    def generate(self, image: Image.Image, image_path: str, prompt_text: str) -> str:
        img_path = self._ensure_image_path(image, image_path)

        query = self.tokenizer.from_list_format([
            {"image": img_path},
            {"text": prompt_text},
        ])

        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return clean_text(response)


def build_runner(args) -> BaseRunner:
    name = args.model_name.lower()

    if name in {"llava", "llava1.5", "llava-1.5", "llava15"}:
        return LlavaRepoRunner(args)

    if name in {"qwen-vl-chat", "qwenvl", "qwen-vl"}:
        return QwenVLChatRunner(args)

    raise ValueError(
        f"Unsupported --model-name {args.model_name}. "
        "Use llava1.5 or qwen-vl-chat."
    )


# =========================================================
# stats / output
# =========================================================
def safe_model_tag(args) -> str:
    mid = args.model_id or args.model_name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", mid)


def safe_alias_tag(args) -> str:
    return args.on_alias


def print_stats(name: str, total: int, correct: int, pred_counter: Dict[str, int]):
    print("=" * 120)
    print(f"[{name}]")
    print(f"total_on_examples = {total}")
    print(f"correct_as_on = {correct}")
    print(f"on_acc = {correct / total:.4f}" if total > 0 else "on_acc = N/A")
    print("prediction_count:")
    for lab in ["left", "right", "under", "on", "UNK"]:
        print(f"  {lab}: {pred_counter[lab]}")


def main():
    args = parse_args()
    seed_all(args.seed)

    args.cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    os.makedirs(args.cache_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", args.cache_dir)

    runner = build_runner(args)

    print(f"Loading dataset: {args.dataset} (download={args.download})")
    dataset = get_dataset(
        args.dataset,
        image_preprocess=runner.image_preprocess,
        download=args.download,
    )

    if isinstance(runner, LlavaRepoRunner):
        prompt_records, sampled_indices = runner.load_prompt_records_with_sampling(
            args.dataset,
            args.option,
        )
        if sampled_indices is not None:
            dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        prompt_records = load_prompt_records(args.dataset, args.option)

    if len(prompt_records) != len(dataset):
        raise ValueError(
            f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)})."
        )

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    total_by_template = defaultdict(int)
    correct_by_template = defaultdict(int)
    pred_counter_by_template = {pid: defaultdict(int) for pid in range(1, 6)}

    printed_examples = 0
    max_print_examples = 5
    total_gold_on_samples = 0

    print("=" * 120)
    print(
        f"RUNNING ON-ONLY MC5 | dataset={args.dataset} | "
        f"model={args.model_name} | on_alias={alias_text(args)}"
    )
    print("=" * 120)

    for local_idx in tqdm(range(start, end), desc="on-only mc5 examples"):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
        image_path = clean_text(item.get("image_path", ""))
        image = item["image_options"][0].convert("RGB")

        raw_question = rec["question"]

        gold = normalize_rel(rec.get("answer", None))
        if gold is None:
            gold = relation_from_image_name(image_name)

        if gold != "on":
            continue

        total_gold_on_samples += 1
        prompt_templates = build_mc_prompts_on_alias(raw_question, args)

        example_rows = []

        for template_id, prompt_text in prompt_templates.items():
            raw_output = runner.generate(image, image_path, prompt_text)
            pred = parse_prediction_mc(raw_output, template_id, args)
            correct = pred == "on"

            total_by_template[template_id] += 1
            pred_counter_by_template[template_id][pred] += 1

            if correct:
                correct_by_template[template_id] += 1

            example_rows.append({
                "template_id": template_id,
                "prompt_text": prompt_text,
                "raw_output": raw_output,
                "pred": pred,
                "correct": correct,
            })

        if printed_examples < max_print_examples:
            printed_examples += 1

            print("\n" + "=" * 120)
            print(f"[ON EXAMPLE {printed_examples}/{max_print_examples}]")
            print(f"idx={local_idx}")
            print(f"image_name={image_name}")
            print("gold=on")
            print(f"on_alias={alias_text(args)}")
            print(f"[RAW QUESTION] {clean_question_text(raw_question)}")

            for row in example_rows:
                print("-" * 80)
                print(f"[TEMPLATE {row['template_id']}]")
                print(f"[RAW PROMPT] {row['prompt_text']}")
                print(f"[RAW OUTPUT] {row['raw_output']}")
                print(f"[PRED] {row['pred']}")
                print(f"[CORRECT] {row['correct']}")

    print("\n" + "#" * 120)
    print(
        f"FINAL RESULTS | MC5 ON-ONLY | dataset={args.dataset} | "
        f"model={args.model_name} | on_alias={alias_text(args)}"
    )
    print("#" * 120)

    print(f"total_gold_on_samples = {total_gold_on_samples}")

    for pid in range(1, 6):
        total = total_by_template[pid]
        correct = correct_by_template[pid]
        acc = 0.0 if total == 0 else correct / total

        print("=" * 120)
        print(f"[TEMPLATE {pid}]")
        print(f"total = {total}")
        print(f"correct = {correct}")
        print(f"on_acc = {acc:.4f}" if total > 0 else "on_acc = N/A")

        print("prediction_count:")
        for lab in ["left", "right", "under", "on", "UNK"]:
            print(f"  {lab}: {pred_counter_by_template[pid][lab]}")


if __name__ == "__main__":
    main()
