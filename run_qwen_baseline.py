# run_qwen_baseline.py
import os
import re
import csv
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from dataset_zoo import get_dataset
from misc import seed_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--option", default="four", type=str)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--max-new-tokens", default=20, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--cache-dir", default=None, type=str)
    parser.add_argument("--out-dir", default="output_qwen_baseline", type=str)
    return parser.parse_args()


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


def strip_legacy_prompt(prompt: str) -> str:
    prompt = prompt.replace("<image>", "").strip()
    prompt = re.sub(r"^\s*USER:\s*", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\n?\s*ASSISTANT:\s*$", "", prompt, flags=re.IGNORECASE)
    return prompt.strip()


def normalize_label(text: str):
    if text is None:
        return None

    s = str(text).strip().lower()

    if re.search(r"\btrue\b", s) or re.search(r"\byes\b", s):
        return "true"
    if re.search(r"\bfalse\b", s) or re.search(r"\bno\b", s):
        return "false"

    if re.search(r"\bin[\s-]?front\b", s):
        return "in-front"
    if re.search(r"\bbehind\b", s):
        return "behind"
    if re.search(r"\bleft\b", s):
        return "left"
    if re.search(r"\bright\b", s):
        return "right"
    if re.search(r"\bunder\b", s) or re.search(r"\bbelow\b", s) or re.search(r"\bbeneath\b", s):
        return "under"
    if re.search(r"\bon\b", s) or re.search(r"\bon top of\b", s):
        return "on"

    s = re.sub(r"[^a-zA-Z\- ]+", " ", s).strip()
    if not s:
        return None
    return s.split()[0]


def get_gold_label(answer_field):
    if isinstance(answer_field, list):
        if len(answer_field) == 0:
            return None
        answer_field = answer_field[0]
    return normalize_label(answer_field)


def build_inputs(processor, image, question_text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


@torch.no_grad()
def generate_one(model, processor, image, question_text, max_new_tokens=20, temperature=0.0):
    inputs = build_inputs(processor, image, question_text)

    model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {
        k: (v.to(model_device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }

    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    generated_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_len:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return output_text


def main():
    args = parse_args()
    seed_all(args.seed)

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ['USER']}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    prompt_records = load_prompt_records(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)

    if len(prompt_records) != len(dataset):
        raise ValueError(
            f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)})."
        )

    print(f"Loading model: {args.model_id}")
    print(f"Using cache_dir: {cache_dir}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        device_map="auto" if args.device.startswith("cuda") else None,
        torch_dtype="auto",
    ).eval()

    if not args.device.startswith("cuda"):
        model.to(args.device)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
    )

    if args.limit < 0:
        end_index = len(dataset)
    else:
        end_index = min(args.start_index + args.limit, len(dataset))

    model_name = args.model_id.split("/")[-1]
    result_jsonl = os.path.join(args.out_dir, f"{args.dataset}_{model_name}_results.jsonl")
    result_csv = os.path.join(args.out_dir, f"{args.dataset}_{model_name}_results.csv")
    summary_json = os.path.join(args.out_dir, f"{args.dataset}_{model_name}_summary.json")
    summary_csv = os.path.join(args.out_dir, f"{args.dataset}_{model_name}_summary.csv")

    all_rows = []
    num_total = 0
    num_correct = 0

    with open(result_jsonl, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(args.start_index, end_index), desc="Running"):
            rec = prompt_records[idx]
            item = dataset[idx]

            image = item["image_options"][0]
            image_path = item.get("image_path", "")
            image_name = item.get("image_name", f"sample_{idx:05d}")

            raw_prompt = rec["question"]
            question_text = strip_legacy_prompt(raw_prompt)

            gold = get_gold_label(rec["answer"])
            pred_text = generate_one(
                model=model,
                processor=processor,
                image=image,
                question_text=question_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            pred = normalize_label(pred_text)
            correct = (pred == gold)

            num_total += 1
            num_correct += int(correct)

            row = {
                "index": idx,
                "image_name": image_name,
                "image_path": image_path,
                "question_raw": raw_prompt,
                "question_qwen": question_text,
                "gold_raw": rec["answer"],
                "gold": gold,
                "pred_text": pred_text,
                "pred": pred,
                "correct": correct,
            }
            all_rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if all_rows:
        with open(result_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summary = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "option": args.option,
        "start_index": args.start_index,
        "end_index": end_index,
        "num_total": num_total,
        "num_correct": num_correct,
        "accuracy": (num_correct / num_total) if num_total > 0 else 0.0,
        "result_jsonl": result_jsonl,
        "result_csv": result_csv,
        "cache_dir": cache_dir,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
