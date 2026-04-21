import os
import re
import csv
import json
import argparse
import itertools
from collections import OrderedDict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

CHOICES = ["left", "right", "on", "under"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", required=True, type=str)
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct", type=str)
    p.add_argument("--out-dir", default="./output_llama_text_from_summary", type=str)
    p.add_argument("--max-new-tokens", default=16, type=int)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--do-sample", action="store_true")
    return p.parse_args()


def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_rel(answer):
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


def parse_prediction(text: str):
    t = clean_text(text).lower()
    m = re.search(r"\b(left|right|on|under)\b", t)
    return m.group(1) if m else "UNK"


def strip_answer_order_clause(question_text: str):
    q = clean_text(question_text)
    q = re.sub(
        r"Answer with\s+left,\s*right,\s*on\s+or\s+under(?:\s+only)?\.\s*$",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q


def build_q0_permuted_prompts(base_question, caption_text):
    stem = strip_answer_order_clause(base_question)
    cap = clean_text(caption_text)

    out = []
    for perm in itertools.permutations(CHOICES):
        order_text = ", ".join(perm[:-1]) + f" or {perm[-1]}"
        prompt_text = (
            f"Caption: {cap}\n"
            f"Question: {stem} Answer with {order_text} only."
        )
        perm_id = "_".join(perm)
        out.append({
            "qid": "q0",
            "perm_id": perm_id,
            "order": list(perm),
            "caption_text": cap,
            "prompt_text": prompt_text,
        })
    return out


def safe_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_unique_examples_from_summary(summary_csv):
    rows = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    unique = OrderedDict()
    for row in rows:
        image_name = clean_text(row.get("image_name", ""))
        local_index = clean_text(row.get("local_index", ""))
        base_question = clean_text(row.get("base_question", row.get("question", "")))
        caption_text = clean_text(row.get("caption_text", ""))
        gold = normalize_rel(row.get("gold", "UNK"))
        image_path = clean_text(row.get("image_path", ""))

        if not base_question or not caption_text:
            continue

        key = (image_name, local_index, base_question, caption_text, gold)
        if key not in unique:
            unique[key] = {
                "image_name": image_name,
                "image_path": image_path,
                "local_index": local_index,
                "base_question": base_question,
                "caption_text": caption_text,
                "gold": gold,
            }

    return list(unique.values())


def load_model_and_tokenizer(model_id, requested_device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if requested_device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
        ).eval()
        model.to("cpu")

    return model, tokenizer


def run_one_generation(model, tokenizer, prompt_text, max_new_tokens=16, do_sample=False, temperature=0.0):
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return gen_text


def build_summary_fieldnames():
    return [
        "model_id",
        "summary_csv",
        "image_name",
        "image_path",
        "local_index",
        "qid",
        "perm_id",
        "order_text",
        "base_question",
        "caption_text",
        "prompt_text",
        "gold",
        "pred",
        "correct",
        "generated_text",
    ]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    examples = load_unique_examples_from_summary(args.summary_csv)
    if not examples:
        raise ValueError(f"No usable rows found in summary csv: {args.summary_csv}")

    start = args.sample_index
    end = len(examples) if args.limit < 0 else min(len(examples), start + args.limit)
    examples = examples[start:end]

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)

    summary_rows = []
    fieldnames = build_summary_fieldnames()

    summary_name = os.path.splitext(os.path.basename(args.summary_csv))[0]
    out_csv = os.path.join(args.out_dir, f"llama_text_only_from_{summary_name}.csv")

    for ex in tqdm(examples, desc="Examples"):
        permuted_prompts = build_q0_permuted_prompts(
            base_question=ex["base_question"],
            caption_text=ex["caption_text"],
        )

        for q in permuted_prompts:
            gen_text = run_one_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=q["prompt_text"],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )

            pred = parse_prediction(gen_text)
            correct = (pred == ex["gold"])

            summary_rows.append({
                "model_id": args.model_id,
                "summary_csv": args.summary_csv,
                "image_name": ex["image_name"],
                "image_path": ex["image_path"],
                "local_index": ex["local_index"],
                "qid": "q0",
                "perm_id": q["perm_id"],
                "order_text": " | ".join(q["order"]),
                "base_question": ex["base_question"],
                "caption_text": ex["caption_text"],
                "prompt_text": q["prompt_text"],
                "gold": ex["gold"],
                "pred": pred,
                "correct": correct,
                "generated_text": gen_text,
            })

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    total = len(summary_rows)
    correct = sum(int(safe_bool(r["correct"])) for r in summary_rows)
    acc = 0.0 if total == 0 else correct / total

    print(f"Saved summary to: {out_csv}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
