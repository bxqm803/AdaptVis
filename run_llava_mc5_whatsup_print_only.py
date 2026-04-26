# run_vlm_mc_open_whatsup.py

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
    p.add_argument("--model-id", default=None, type=str)
    p.add_argument("--method", default="base", type=str)

    # mc / open / both
    p.add_argument("--task", default="both", choices=["mc", "open", "both"], type=str)

    p.add_argument("--max-new-tokens", default=64, type=int)
    p.add_argument("--temperature", default=0.0, type=float)

    # 4 classes * 3 examples = 12 examples total
    p.add_argument("--print-per-class", default=3, type=int)
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
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    stem = stem.replace("_in_front_of_", "_in-front_of_")

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
    t = clean_text(text).lower().replace("_", "-")
    valid = set(valid_labels)

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
# prompt builders
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

        self.processor = InstructBlipProcessor.from_pretrained(
            model_id,
            cache_dir=args.cache_dir,
        )

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
    def generate(self, image:
