import os
import re
import csv
import json
import math
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from dataset_zoo import get_dataset
from misc import seed_all


SUPPORTED_VLM_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.5-9B",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-9B",
        type=str,
        help="Examples: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3.5-9B",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--option", default="four", type=str)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--cache-dir", default=None, type=str)
    parser.add_argument("--out-dir", default="output_qwen_baseline", type=str)
    parser.add_argument("--outfile", default=None, type=str)
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

    return None


def extract_answer_from_text(text: str, choices=None):
    if text is None:
        return None

    s = str(text).strip().lower()
    if len(s) == 0:
        return None

    if not choices:
        return normalize_label(s)

    patterns = {
        "true": r"\btrue\b|\byes\b",
        "false": r"\bfalse\b|\bno\b",
        "left": r"\bleft\b",
        "right": r"\bright\b",
        "on": r"\bon\b|\bon top of\b",
        "under": r"\bunder\b|\bbelow\b|\bbeneath\b",
        "behind": r"\bbehind\b",
        "in-front": r"\bin[\s-]?front\b",
    }

    matches = []
    for choice in choices:
        patt = patterns.get(choice)
        if patt is None:
            continue
        for m in re.finditer(patt, s):
            matches.append((m.start(), choice))

    if len(matches) == 0:
        return None

    # Prefer the last explicit answer mention in the free-form text.
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def get_gold_label(answer_field):
    if isinstance(answer_field, list):
        if len(answer_field) == 0:
            return None
        answer_field = answer_field[0]
    return normalize_label(answer_field)


def extract_choices(question_text: str):
    """
    Parse candidates from prompts like:
    'Answer with left, right, on or under.'
    'Answer with true or false.'
    """
    m = re.search(r"answer with\s+(.+?)(?:[\.!?]|$)", question_text, flags=re.IGNORECASE)
    if not m:
        return []

    raw = m.group(1).strip().lower()
    raw = raw.replace(" or ", ",")
    parts = [p.strip(" ,.") for p in raw.split(",") if p.strip(" ,.")]

    choices = []
    for p in parts:
        n = normalize_label(p)
        if n is not None and n not in choices:
            choices.append(n)
    return choices


def make_user_messages(image, question_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ],
        }
    ]


def make_scored_messages(image, question_text, answer_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(answer_text)},
            ],
        },
    ]


def build_inputs(processor, messages, add_generation_prompt):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


def longest_common_prefix_len_list(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return i


def find_subsequence(seq, pattern, start=0):
    if len(pattern) == 0:
        return None
    for i in range(start, len(seq) - len(pattern) + 1):
        if seq[i:i + len(pattern)] == pattern:
            return i
    return None


def get_answer_id_variants(tokenizer, answer_text):
    raw = str(answer_text)
    candidates = [raw, " " + raw, "\n" + raw]
    seen = set()
    variants = []
    for cand in candidates:
        ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
        key = tuple(ids)
        if len(ids) > 0 and key not in seen:
            seen.add(key)
            variants.append(ids)
    return variants


def make_empty_score(answer_text=None):
    return {
        "answer": answer_text,
        "token_ids": [],
        "tokens": [],
        "token_probs": [],
        "token_logits": [],
        "seq_logprob": float("-inf"),
    }


@torch.no_grad()
def generate_one(model, processor, image, question_text, max_new_tokens=256, temperature=0.0):
    messages = make_user_messages(image, question_text)
    inputs = build_inputs(processor, messages, add_generation_prompt=True)

    model_device = next(model.parameters()).device
    inputs = {
        k: (v.to(model_device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, prompt_len:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    free_token_ids = generated_ids[0].tolist()
    free_tokens = processor.tokenizer.convert_ids_to_tokens(free_token_ids)

    free_token_probs = []
    free_token_logits = []

    for step, token_id in enumerate(free_token_ids):
        step_logits = outputs.scores[step][0]
        step_probs = torch.softmax(step_logits, dim=-1)
        free_token_probs.append(float(step_probs[token_id].item()))
        free_token_logits.append(float(step_logits[token_id].item()))

    return {
        "pred_text": output_text,
        "free_token_ids": free_token_ids,
        "free_tokens": free_tokens,
        "free_token_probs": free_token_probs,
        "free_token_logits": free_token_logits,
    }


@torch.no_grad()
def score_candidate_answer(model, processor, image, question_text, answer_text):
    if answer_text is None or len(str(answer_text).strip()) == 0:
        return make_empty_score(answer_text)

    model_device = next(model.parameters()).device

    prefix_messages = make_user_messages(image, question_text)
    prefix_inputs = build_inputs(processor, prefix_messages, add_generation_prompt=True)
    prefix_inputs = {
        k: (v.to(model_device) if torch.is_tensor(v) else v)
        for k, v in prefix_inputs.items()
    }

    full_messages = make_scored_messages(image, question_text, answer_text)
    full_inputs = build_inputs(processor, full_messages, add_generation_prompt=False)
    full_inputs = {
        k: (v.to(model_device) if torch.is_tensor(v) else v)
        for k, v in full_inputs.items()
    }

    outputs = model(**full_inputs)
    logits = outputs.logits  # [1, seq_len, vocab]

    prefix_ids = prefix_inputs["input_ids"][0].tolist()
    full_ids = full_inputs["input_ids"][0].tolist()

    common_len = longest_common_prefix_len_list(prefix_ids, full_ids)

    variants = get_answer_id_variants(processor.tokenizer, answer_text)
    ans_start = None
    matched_ids = None

    for ids in variants:
        pos = find_subsequence(full_ids, ids, start=max(0, common_len - 32))
        if pos is not None:
            ans_start = pos
            matched_ids = ids
            break

    if ans_start is None or matched_ids is None:
        return make_empty_score(answer_text)

    tokens = processor.tokenizer.convert_ids_to_tokens(matched_ids)
    token_probs = []
    token_logits = []

    for i, token_id in enumerate(matched_ids):
        pos = ans_start + i - 1
        if pos < 0:
            continue

        step_logits = logits[0, pos, :]
        step_probs = torch.softmax(step_logits, dim=-1)

        p = float(step_probs[token_id].item())
        l = float(step_logits[token_id].item())

        token_probs.append(p)
        token_logits.append(l)

    if len(token_probs) == 0:
        seq_logprob = float("-inf")
    else:
        seq_logprob = float(sum(math.log(max(p, 1e-45)) for p in token_probs))

    return {
        "answer": answer_text,
        "token_ids": matched_ids,
        "tokens": tokens,
        "token_probs": token_probs,
        "token_logits": token_logits,
        "seq_logprob": seq_logprob,
    }


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
    if args.model_id not in SUPPORTED_VLM_MODELS:
        print(f"[Warning] {args.model_id} not in tested list: {SUPPORTED_VLM_MODELS}")

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

    if args.outfile is not None:
        result_csv = args.outfile
        out_parent = os.path.dirname(os.path.abspath(result_csv))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
    else:
        result_csv = os.path.join(args.out_dir, f"{args.dataset}_{model_name}_results.csv")

    summary_json = os.path.join(args.out_dir, f"{args.dataset}_{model_name}_summary.json")
    summary_csv = os.path.join(args.out_dir, f"{args.dataset}_{model_name}_summary.csv")

    all_rows = []
    num_total = 0
    num_correct_from_text = 0
    num_correct_by_score = 0

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

            # 1) Real free-form output from the model
            gen_out = generate_one(
                model=model,
                processor=processor,
                image=image,
                question_text=question_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            pred_text_free = gen_out["pred_text"]

            # 2) Parse answer from the real free-form output
            choices = extract_choices(question_text)
            if gold is not None and gold not in choices:
                choices.append(gold)

            pred_from_text = extract_answer_from_text(pred_text_free, choices=choices)
            correct_from_text = (pred_from_text == gold)

            # 3) Candidate scoring as reference
            choice_scores = [
                score_candidate_answer(
                    model=model,
                    processor=processor,
                    image=image,
                    question_text=question_text,
                    answer_text=choice,
                )
                for choice in choices
            ]

            if len(choice_scores) > 0:
                pred_out = max(choice_scores, key=lambda x: x["seq_logprob"])
                pred_by_score = pred_out["answer"]
            else:
                pred_out = make_empty_score(None)
                pred_by_score = None

            correct_by_score = (pred_by_score == gold)

            gold_out = None
            for item_score in choice_scores:
                if item_score["answer"] == gold:
                    gold_out = item_score
                    break
            if gold_out is None:
                gold_out = score_candidate_answer(
                    model=model,
                    processor=processor,
                    image=image,
                    question_text=question_text,
                    answer_text=gold,
                )

            num_total += 1
            num_correct_from_text += int(correct_from_text)
            num_correct_by_score += int(correct_by_score)

            row = {
                "index": idx,
                "image_name": image_name,
                "image_path": image_path,
                "question_raw": raw_prompt,
                "question_qwen": question_text,

                "gold_raw": json.dumps(rec["answer"], ensure_ascii=False),
                "gold": gold,

                "pred_text_free": pred_text_free,
                "pred_from_text": pred_from_text,
                "correct_from_text": correct_from_text,

                "pred_by_score": pred_by_score,
                "correct_by_score": correct_by_score,

                "choices": json.dumps(choices, ensure_ascii=False),
                "choice_seq_logprobs": json.dumps(
                    {x["answer"]: x["seq_logprob"] for x in choice_scores},
                    ensure_ascii=False,
                ),

                # Raw free-form output token info
                "free_token_ids": json.dumps(gen_out["free_token_ids"], ensure_ascii=False),
                "free_tokens": json.dumps(gen_out["free_tokens"], ensure_ascii=False),
                "free_token_probs": json.dumps(gen_out["free_token_probs"], ensure_ascii=False),
                "free_token_logits": json.dumps(gen_out["free_token_logits"], ensure_ascii=False),

                # Predicted answer token info (score-based reference)
                "pred_token_ids": json.dumps(pred_out["token_ids"], ensure_ascii=False),
                "pred_tokens": json.dumps(pred_out["tokens"], ensure_ascii=False),
                "pred_token_probs": json.dumps(pred_out["token_probs"], ensure_ascii=False),
                "pred_token_logits": json.dumps(pred_out["token_logits"], ensure_ascii=False),

                # Gold answer token info
                "gold_token_ids": json.dumps(gold_out["token_ids"], ensure_ascii=False),
                "gold_tokens": json.dumps(gold_out["tokens"], ensure_ascii=False),
                "gold_token_probs": json.dumps(gold_out["token_probs"], ensure_ascii=False),
                "gold_token_logits": json.dumps(gold_out["token_logits"], ensure_ascii=False),
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
        "num_correct_from_text": num_correct_from_text,
        "accuracy_from_text": (num_correct_from_text / num_total) if num_total > 0 else 0.0,
        "num_correct_by_score": num_correct_by_score,
        "accuracy_by_score": (num_correct_by_score / num_total) if num_total > 0 else 0.0,
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
