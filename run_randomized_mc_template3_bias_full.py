# run_randomized_mc_template3_first_option_bias.py
import os
import re
import json
import argparse
import tempfile
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from PIL import Image

from misc import seed_all
from dataset_zoo import get_dataset


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--device", default="cuda", type=str)
    p.add_argument(
        "--dataset",
        default="Controlled_Images_A",
        type=str,
        choices=["Controlled_Images_A", "COCO_QA_two_obj"],
    )
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--download", action="store_true")

    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--print-first-n", default=5, type=int)

    p.add_argument("--cache-dir", default=None, type=str)

    # llava1.5 / qwen-vl-chat
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--model-id", default=None, type=str)
    p.add_argument("--method", default="base", type=str)

    p.add_argument(
        "--on-alias",
        default="on",
        choices=["on", "above", "at_the_top_of"],
        help="How to write the on option in choices.",
    )

    p.add_argument("--max-new-tokens", default=32, type=int)
    p.add_argument("--temperature", default=0.0, type=float)

    # all: evaluate all labels
    # only_on: only evaluate gold=on
    p.add_argument("--eval-scope", default="all", choices=["all", "only_on"], type=str)

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


def alias_text(args) -> str:
    if args.on_alias == "on":
        return "on"
    if args.on_alias == "above":
        return "above"
    if args.on_alias == "at_the_top_of":
        return "at the top of"
    raise ValueError(args.on_alias)


def labels_with_alias(args) -> List[str]:
    return ["left", "right", "under", alias_text(args)]


def canonicalize_label(label: str) -> str:
    s = clean_text(label).lower().replace("_", " ")

    if re.search(r"\bto the left of\b|\bleft of\b|\bleft\b", s):
        return "left"
    if re.search(r"\bto the right of\b|\bright of\b|\bright\b", s):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b|\bunderneath\b", s):
        return "under"
    if re.search(r"\bat the top of\b|\bon top of\b|\babove\b|\bover\b|\batop\b|\bon\b", s):
        return "on"

    return "UNK"


def normalize_rel(answer: Any) -> Optional[str]:
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if len(answer) > 0 else None

    if answer is None:
        return None

    label = canonicalize_label(str(answer))
    return None if label == "UNK" else label


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


def parse_prediction_generic(text: str) -> str:
    t = clean_text(text).lower().replace("_", " ")

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


# =========================================================
# randomized template 3
# =========================================================
def build_random_template3_prompt(
    base_question: str,
    args,
    rng: random.Random,
) -> Tuple[str, Dict[str, str], List[str]]:
    stem = strip_existing_answer_clause(base_question)
    if not stem.endswith("?"):
        stem = stem.rstrip(".") + "?"

    labels = labels_with_alias(args)

    # 1) Randomize label order.
    shuffled_labels = labels[:]
    rng.shuffle(shuffled_labels)

    # 2) Randomize displayed option-letter order.
    # Example: D. on A. right C. left B. under
    option_letters = ["A", "B", "C", "D"]
    rng.shuffle(option_letters)

    letter_to_label = {
        letter: label
        for letter, label in zip(option_letters, shuffled_labels)
    }

    option_text = " ".join(
        [f"{letter}. {letter_to_label[letter]}" for letter in option_letters]
    )

    # Important:
    # No "Answer with only A, B, C, or D."
    # This isolates whether the model tends to pick the first displayed option.
    prompt = (
        f"{stem} Choose one option only. "
        f"{option_text}."
    )

    return prompt, letter_to_label, option_letters


def parse_random_template3_output(
    raw_output: str,
    letter_to_label: Dict[str, str],
) -> Tuple[str, str]:
    t = clean_text(raw_output)

    # Prefer explicit A/B/C/D answer if present.
    m = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
    if m:
        pred_letter = m.group(1).upper()
        pred_label = canonicalize_label(letter_to_label[pred_letter])
        return pred_letter, pred_label

    # Otherwise parse generated label text.
    pred_label = parse_prediction_generic(t)
    pred_letter = "UNK"

    for letter, label in letter_to_label.items():
        if canonicalize_label(label) == pred_label:
            pred_letter = letter
            break

    return pred_letter, pred_label


def get_gold_letter(gold: str, letter_to_label: Dict[str, str]) -> str:
    for letter, label in letter_to_label.items():
        if canonicalize_label(label) == gold:
            return letter
    return "UNK"


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
        llava_prompt = f"<image> USER: {prompt_text} ASSISTANT:"
        out = self.wrapper.run_single_prompt(
            image=image,
            prompt=llava_prompt,
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

    if name in {"qwen-vl-chat", "qwen-vl", "qwenvl"}:
        return QwenVLChatRunner(args)

    raise ValueError(
        f"Unsupported --model-name {args.model_name}. "
        "Use llava1.5 or qwen-vl-chat."
    )


# =========================================================
# stats
# =========================================================
def print_counter(title: str, counter: Counter, keys: List[str]):
    print(title)
    for k in keys:
        print(f"  {k}: {counter[k]}")


def safe_rate(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


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

    rng = random.Random(args.seed)

    total = 0
    correct = 0

    total_by_gold = Counter()
    correct_by_gold = Counter()

    pred_label_counter = Counter()
    pred_letter_counter = Counter()

    gold_label_counter = Counter()
    gold_letter_counter = Counter()

    first_option_chosen = 0
    first_option_letter_counter = Counter()
    pred_is_first_option_by_gold = Counter()
    total_by_gold_for_first = Counter()

    confusion = defaultdict(Counter)
    printed = 0

    print("=" * 120)
    print(
        f"RUNNING FIRST-DISPLAYED-OPTION BIAS TEST | "
        f"dataset={args.dataset} | model={args.model_name} | "
        f"on_alias={alias_text(args)} | eval_scope={args.eval_scope}"
    )
    print("=" * 120)

    for local_idx in tqdm(range(start, end), desc="first-option-bias"):
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

        if args.eval_scope == "only_on" and gold != "on":
            continue

        prompt_text, letter_to_label, option_letters = build_random_template3_prompt(
            raw_question, args, rng
        )

        first_option_letter = option_letters[0]
        gold_letter = get_gold_letter(gold, letter_to_label)

        raw_output = runner.generate(image, image_path, prompt_text)
        pred_letter, pred = parse_random_template3_output(raw_output, letter_to_label)

        is_correct = pred == gold
        pred_is_first_option = pred_letter == first_option_letter

        total += 1
        correct += int(is_correct)

        total_by_gold[gold] += 1
        correct_by_gold[gold] += int(is_correct)

        gold_label_counter[gold] += 1
        gold_letter_counter[gold_letter] += 1

        pred_label_counter[pred] += 1
        pred_letter_counter[pred_letter] += 1

        first_option_letter_counter[first_option_letter] += 1
        first_option_chosen += int(pred_is_first_option)

        total_by_gold_for_first[gold] += 1
        pred_is_first_option_by_gold[gold] += int(pred_is_first_option)

        confusion[gold][pred] += 1

        if printed < args.print_first_n:
            printed += 1
            print("\n" + "=" * 120)
            print(f"[RANDOMIZED EXAMPLE {printed}/{args.print_first_n}]")
            print(f"idx={local_idx}")
            print(f"image_name={image_name}")
            print(f"gold={gold}")
            print(f"gold_letter={gold_letter}")
            print(f"letter_to_label={letter_to_label}")
            print(f"option_letters={option_letters}")
            print(f"first_option_letter={first_option_letter}")
            print(f"[RAW QUESTION] {clean_question_text(raw_question)}")
            print(f"[RAW PROMPT] {prompt_text}")
            print(f"[RAW OUTPUT] {raw_output}")
            print(f"[PRED LETTER] {pred_letter}")
            print(f"[PRED] {pred}")
            print(f"[CORRECT] {is_correct}")
            print(f"[PRED_IS_FIRST_DISPLAYED_OPTION] {pred_is_first_option}")

    print("\n" + "#" * 120)
    print(
        f"FINAL RESULTS | FIRST-DISPLAYED-OPTION BIAS TEST | "
        f"dataset={args.dataset} | model={args.model_name} | "
        f"on_alias={alias_text(args)} | eval_scope={args.eval_scope}"
    )
    print("#" * 120)

    print(f"total = {total}")
    print(f"correct = {correct}")
    print(f"overall_acc = {safe_rate(correct, total):.4f}" if total > 0 else "overall_acc = N/A")

    print("\nfirst-position bias:")
    print(
        f"  chose_first_displayed_option = {first_option_chosen}/{total} "
        f"({safe_rate(first_option_chosen, total):.4f})"
    )

    print("\nfirst displayed option chosen by gold:")
    for lab in ["left", "right", "under", "on"]:
        n = total_by_gold_for_first[lab]
        c = pred_is_first_option_by_gold[lab]
        print(f"  {lab}: {c}/{n} ({safe_rate(c, n):.4f})")

    print("\nper-gold accuracy:")
    for lab in ["left", "right", "under", "on"]:
        n = total_by_gold[lab]
        c = correct_by_gold[lab]
        print(f"  {lab}_acc = {safe_rate(c, n):.4f} ({c}/{n})")

    print()
    print_counter("gold label distribution:", gold_label_counter, ["left", "right", "under", "on"])
    print()
    print_counter("gold letter distribution after randomization:", gold_letter_counter, ["A", "B", "C", "D", "UNK"])
    print()
    print_counter("first displayed option letter distribution:", first_option_letter_counter, ["A", "B", "C", "D"])
    print()
    print_counter("pred label distribution:", pred_label_counter, ["left", "right", "under", "on", "UNK"])
    print()
    print_counter("pred letter distribution:", pred_letter_counter, ["A", "B", "C", "D", "UNK"])

    print("\nconfusion matrix: gold -> pred")
    for gold_lab in ["left", "right", "under", "on"]:
        row = confusion[gold_lab]
        print(
            f"  {gold_lab}: "
            f"left={row['left']} "
            f"right={row['right']} "
            f"under={row['under']} "
            f"on={row['on']} "
            f"UNK={row['UNK']}"
        )


if __name__ == "__main__":
    main()
