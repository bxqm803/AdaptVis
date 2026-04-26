import os
import re
import csv
import json
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="./output_vlm_mc_open", type=str)

    # llava1.5 / instructblip / qwen-vl-chat
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--model-id", default=None, type=str,
                   help="HF model id. Defaults: Salesforce/instructblip-vicuna-7b for instructblip; Qwen/Qwen-VL-Chat for qwen-vl-chat.")
    p.add_argument("--method", default="base", type=str, help="Only used by repo LLaVA wrapper.")
    p.add_argument("--task", default="mc", choices=["mc", "open", "both"], type=str)
    p.add_argument("--max-new-tokens", default=64, type=int)
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--print-per-class", default=3, type=int)
    return p.parse_args()


# =========================================================
# text / label utils
# =========================================================
def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_question_text(question: str) -> str:
    q = str(question).strip().replace("\n", " ")
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
        r"Answer with\s+left,\s*right,\s*in-front\s+or\s+behind(?:\s+only)?\.?\s*$",
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
    q = re.sub(r"\s+", " ", q).strip()
    return q


def labels_for_dataset(dataset_name: str) -> List[str]:
    d = dataset_name.lower()
    if "controlled_images_b" in d:
        return ["left", "right", "in-front", "behind"]
    return ["left", "right", "under", "on"]


def normalize_rel(answer: Any) -> Optional[str]:
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            return None
        answer = answer[0]
    if answer is None:
        return None
    s = clean_text(str(answer)).lower().replace("_", "-")

    # Check longer / more specific labels first.
    if re.search(r"\bin[\s-]?front\b", s):
        return "in-front"
    if re.search(r"\bbehind\b", s):
        return "behind"
    if re.search(r"\bto the left of\b|\bleft of\b|\bleft\b", s):
        return "left"
    if re.search(r"\bto the right of\b|\bright of\b|\bright\b", s):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b", s):
        return "under"
    if re.search(r"\bon top of\b|\babove\b|\bon\b", s):
        return "on"
    return None


def relation_from_image_name(image_name: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_name))[0].lower().replace("_in_front_of_", "_in-front_of_")
    if "_left_of_" in stem:
        return "left"
    if "_right_of_" in stem:
        return "right"
    if "_in-front_of_" in stem or "_in_front_of_" in stem:
        return "in-front"
    if "_behind_" in stem:
        return "behind"
    if "_on_" in stem:
        return "on"
    if "_under_" in stem:
        return "under"
    return None


def parse_prediction_generic(text: str, valid_labels: List[str]) -> str:
    t = clean_text(text).lower().replace("_", "-")
    patterns = [
        (r"\bin[\s-]?front\b", "in-front"),
        (r"\bbehind\b", "behind"),
        (r"\bto the left of\b", "left"),
        (r"\bleft of\b", "left"),
        (r"\bleft\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bright of\b", "right"),
        (r"\bright\b", "right"),
        (r"\bon top of\b", "on"),
        (r"\babove\b", "on"),
        (r"\bon\b", "on"),
        (r"\bunder\b", "under"),
        (r"\bbelow\b", "under"),
        (r"\bbeneath\b", "under"),
    ]
    found = []
    valid = set(valid_labels)
    for pat, lab in patterns:
        if lab not in valid:
            continue
        for m in re.finditer(pat, t):
            found.append((m.start(), m.end(), lab))
    if found:
        found.sort(key=lambda x: (x[0], x[1]))
        return found[-1][2]
    return "UNK"


def parse_prediction_mc(raw_output: str, template_id: int, valid_labels: List[str]) -> str:
    t = clean_text(raw_output)
    if template_id == 3:
        m = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < len(valid_labels):
                return valid_labels[idx]
    return parse_prediction_generic(t, valid_labels)


# =========================================================
# prompt utils
# =========================================================
def build_mc_prompts(base_question: str, valid_labels: List[str]) -> Dict[int, str]:
    stem = strip_existing_answer_clause(base_question)
    if not stem.endswith("?"):
        stem = stem.rstrip(".") + "?"
    a, b, c, d = valid_labels
    label_list = ", ".join(valid_labels)
    return {
        1: f"{stem} Answer with only one word: {a}, {b}, {c}, or {d}.",
        2: f"{stem} Output exactly one label from: {label_list}. Do not output anything else.",
        3: f"{stem} Choose one option only. A. {a} B. {b} C. {c} D. {d}. Answer with only A, B, C, or D.",
        4: f"Which of the following best describes the relation asked in this question? {stem} Options: {label_list}. Answer with only one word.",
        5: f"Select the correct relation for this question: {stem} Choices: {label_list}. Respond with only the chosen word.",
    }


def build_open_prompts(base_question: str) -> Dict[int, str]:
    stem = strip_existing_answer_clause(base_question)
    if not stem.endswith("?"):
        stem = stem.rstrip(".") + "?"
    return {
        1: stem,
        2: f"Describe the spatial relation in this image in one short sentence. Question: {stem}",
        3: f"What is the relation between the two queried objects? Answer naturally. Question: {stem}",
        4: f"State the relation in a short phrase. Question: {stem}",
        5: f"Describe how the first queried object is positioned relative to the second queried object. Question: {stem}",
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


class InstructBLIPRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        model_id = args.model_id or "Salesforce/instructblip-vicuna-7b"
        print(f"Loading InstructBLIP: {model_id}")
        self.processor = InstructBlipProcessor.from_pretrained(model_id, cache_dir=args.cache_dir)
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
            device_map="auto" if args.device.startswith("cuda") else None,
        ).eval()
        if not args.device.startswith("cuda"):
            self.model.to(args.device)

    @torch.no_grad()
    def generate(self, image: Image.Image, image_path: str, prompt_text: str) -> str:
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": self.args.max_new_tokens}
        if self.args.temperature and self.args.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.args.temperature})
        else:
            gen_kwargs.update({"do_sample": False})
        out = self.model.generate(**inputs, **gen_kwargs)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


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

    def _ensure_image_path(self, image: Image.Image, image_path: str, idx: Optional[int] = None) -> str:
        if image_path and os.path.exists(image_path):
            return image_path
        path = os.path.join(self._tmpdir.name, f"qwen_img_{idx or 0}.jpg")
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


class Qwen25VLRunner(BaseRunner):
    """Optional support for Qwen2.5-VL / Qwen3-VL style HF models."""
    def __init__(self, args):
        super().__init__(args)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model_id = args.model_id or "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading Qwen ImageTextToText model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=args.cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            cache_dir=args.cache_dir,
            torch_dtype="auto",
            device_map="auto" if args.device.startswith("cuda") else None,
        ).eval()
        if not args.device.startswith("cuda"):
            self.model.to(args.device)

    @torch.no_grad()
    def generate(self, image: Image.Image, image_path: str, prompt_text: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        dev = next(self.model.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": self.args.max_new_tokens}
        if self.args.temperature and self.args.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.args.temperature})
        else:
            gen_kwargs.update({"do_sample": False})
        out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, prompt_len:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def build_runner(args) -> BaseRunner:
    name = args.model_name.lower()
    if name in {"llava", "llava1.5", "llava-1.5", "llava15"}:
        return LlavaRepoRunner(args)
    if name in {"instructblip", "instruct-blip", "instructblip-vicuna-7b", "instructblip-vicuna-13b"}:
        return InstructBLIPRunner(args)
    if name in {"qwen-vl-chat", "qwenvl", "qwen-vl"}:
        return QwenVLChatRunner(args)
    if name in {"qwen2.5-vl", "qwen25-vl", "qwen3-vl"}:
        return Qwen25VLRunner(args)
    raise ValueError(f"Unsupported --model-name {args.model_name}. Use llava1.5, instructblip, qwen-vl-chat, or qwen2.5-vl.")


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
# stats / I/O
# =========================================================
def safe_model_tag(args) -> str:
    mid = args.model_id or args.model_name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", mid)


def print_stats(name: str, labels: List[str], total: int, correct: int,
                per_gold_total: Dict[str, int], per_gold_correct: Dict[str, int]):
    print("=" * 120)
    print(f"[{name}]")
    print(f"total = {total}")
    print(f"correct = {correct}")
    print(f"overall_acc = {correct / total:.4f}" if total > 0 else "overall_acc = N/A")
    for rel in labels:
        n = per_gold_total[rel]
        c = per_gold_correct[rel]
        acc = 0.0 if n == 0 else c / n
        print(f"{rel}_acc = {acc:.4f} ({c}/{n})")


def run_task(args, runner: BaseRunner, dataset, prompt_records, start: int, end: int, task: str):
    labels = labels_for_dataset(args.dataset)
    out_root = Path(args.out_dir) / args.dataset / safe_model_tag(args) / task
    out_root.mkdir(parents=True, exist_ok=True)
    result_csv = out_root / f"results_{task}.csv"
    result_jsonl = out_root / f"results_{task}.jsonl"

    total_by_template = defaultdict(int)
    correct_by_template = defaultdict(int)
    per_gold_total = {pid: defaultdict(int) for pid in range(1, 6)}
    per_gold_correct = {pid: defaultdict(int) for pid in range(1, 6)}
    printed_by_gold = defaultdict(int)
    rows = []

    with open(result_jsonl, "w", encoding="utf-8") as fj:
        for local_idx in tqdm(range(start, end), desc=f"{task} examples"):
            rec = prompt_records[local_idx]
            item = dataset[local_idx]
            image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}"))
            image_path = clean_text(item.get("image_path", ""))
            image = item["image_options"][0].convert("RGB")
            raw_question = rec["question"]
            gold = normalize_rel(rec.get("answer", None)) or relation_from_image_name(image_name)
            if gold is None or gold not in labels:
                continue

            prompt_templates = build_mc_prompts(raw_question, labels) if task == "mc" else build_open_prompts(raw_question)
            example_rows = []
            for template_id, prompt_text in prompt_templates.items():
                raw_output = runner.generate(image, image_path, prompt_text)
                pred = parse_prediction_mc(raw_output, template_id, labels) if task == "mc" else parse_prediction_generic(raw_output, labels)
                correct = pred == gold

                row = {
                    "task": task,
                    "model_name": args.model_name,
                    "model_id": args.model_id or "",
                    "dataset": args.dataset,
                    "template_id": template_id,
                    "local_index": local_idx,
                    "image_name": image_name,
                    "image_path": image_path,
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "raw_question": clean_question_text(raw_question),
                    "prompt_text": prompt_text,
                    "raw_output": raw_output,
                }
                rows.append(row)
                fj.write(json.dumps(row, ensure_ascii=False) + "\n")
                example_rows.append(row)

                total_by_template[template_id] += 1
                per_gold_total[template_id][gold] += 1
                if correct:
                    correct_by_template[template_id] += 1
                    per_gold_correct[template_id][gold] += 1

            if printed_by_gold[gold] < args.print_per_class:
                printed_by_gold[gold] += 1
                print("=" * 120)
                print(f"[{task.upper()} {gold.upper()} EXAMPLE {printed_by_gold[gold]}/{args.print_per_class}]")
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

    if rows:
        with open(result_csv, "w", newline="", encoding="utf-8") as fc:
            writer = csv.DictWriter(fc, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    reports = {}
    for pid in range(1, 6):
        total = total_by_template[pid]
        correct = correct_by_template[pid]
        reports[str(pid)] = {
            "template_id": pid,
            "total": total,
            "correct": correct,
            "overall_acc": 0.0 if total == 0 else correct / total,
            "per_gold_total": {lab: per_gold_total[pid][lab] for lab in labels},
            "per_gold_correct": {lab: per_gold_correct[pid][lab] for lab in labels},
        }
        print_stats(
            f"{task.upper()}_TEMPLATE_{pid}_{args.dataset}_{args.model_name}",
            labels,
            total,
            correct,
            per_gold_total[pid],
            per_gold_correct[pid],
        )

    summary_path = out_root / f"summary_{task}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "model_name": args.model_name,
            "model_id": args.model_id,
            "dataset": args.dataset,
            "labels": labels,
            "start": start,
            "end": end,
            "result_csv": str(result_csv),
            "result_jsonl": str(result_jsonl),
            "reports": reports,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved {task} csv to: {result_csv}")
    print(f"Saved {task} jsonl to: {result_jsonl}")
    print(f"Saved {task} summary to: {summary_path}")


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    seed_all(args.seed)

    args.cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", args.cache_dir)

    runner = build_runner(args)

    print(f"Loading dataset: {args.dataset} (download={args.download})")
    dataset = get_dataset(args.dataset, image_preprocess=runner.image_preprocess, download=args.download)

    if isinstance(runner, LlavaRepoRunner):
        prompt_records, sampled_indices = runner.load_prompt_records_with_sampling(args.dataset, args.option)
        if sampled_indices is not None:
            dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        prompt_records = load_prompt_records(args.dataset, args.option)

    if len(prompt_records) != len(dataset):
        raise ValueError(f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).")

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    tasks = ["mc", "open"] if args.task == "both" else [args.task]
    for task in tasks:
        run_task(args, runner, dataset, prompt_records, start, end, task)


if __name__ == "__main__":
    main()
