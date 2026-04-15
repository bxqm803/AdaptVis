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
from multiq_utils import (
    build_object_pool,
    build_questions,
    parse_prediction,
)

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
    parser.add_argument("--out-dir", default="output_qwen_multiq", type=str)
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


def normalize_tf_label(x):
    if x is None:
        return "UNK"
    x = str(x).strip().lower()
    if x in {"t", "true"}:
        return "True"
    if x in {"f", "false"}:
        return "False"
    return "UNK"


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


def build_inputs(processor, messages, add_generation_prompt=True):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )


@torch.no_grad()
def generate_free(model, processor, image, question_text, max_new_tokens=256, temperature=0.0):
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

    pred_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    token_ids = generated_ids[0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

    token_probs = []
    token_logits = []

    for step, token_id in enumerate(token_ids):
        step_logits = outputs.scores[step][0]
        step_probs = torch.softmax(step_logits, dim=-1)
        token_probs.append(float(step_probs[token_id].item()))
        token_logits.append(float(step_logits[token_id].item()))

    return {
        "pred_text": pred_text,
        "token_ids": token_ids,
        "tokens": tokens,
        "token_probs": token_probs,
        "token_logits": token_logits,
        "final_prob": token_probs[-1] if len(token_probs) > 0 else None,
        "final_logit": token_logits[-1] if len(token_logits) > 0 else None,
    }


def get_candidate_choices(mode: str):
    if mode == "orig":
        return ["left", "right", "on", "under"]
    return ["True", "False"]


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
    object_pool = build_object_pool(prompt_records)

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
        end_index = len(prompt_records)
    else:
        end_index = min(args.start_index + args.limit, len(prompt_records))

    model_name = args.model_id.split("/")[-1]
    out_root = os.path.join(args.out_dir, args.dataset, model_name)
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []
    summary_csv = os.path.join(out_root, "summary.csv")

    for local_idx in tqdm(range(args.start_index, end_index), desc=f"{args.dataset}:{model_name}"):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image = item["image_options"][0]
        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")
        image_stem = os.path.splitext(image_name)[0]

        base_answer = rec["answer"][0] if isinstance(rec["answer"], list) else rec["answer"]
        questions, meta = build_questions(
            base_prompt=rec["question"],
            base_answer=base_answer,
            sample_idx=local_idx,
            object_pool=object_pool,
        )

        sample_dir = os.path.join(out_root, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        meta_out = {
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            **meta,
            "questions": [],
        }

        q_correct_map = {}
        q_prob_map = {}
        q_json_map = {}
        q_pred_map = {}
        q_gold_map = {}
        q_raw_text_map = {}

        q_pred_parse_map = {}
        q_correct_parse_map = {}
        q_pred_score_map = {}
        q_correct_score_map = {}
        q_choice_scores_map = {}

        for q in questions:
            qid = q["qid"]
            question_text = strip_legacy_prompt(q["prompt"])

            gen_out = generate_free(
                model=model,
                processor=processor,
                image=image,
                question_text=question_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            pred_text = gen_out["pred_text"]

            # parse-based result
            pred_parse = parse_prediction(pred_text, q["mode"])
            if q["mode"] == "tf":
                correct_parse = (normalize_tf_label(pred_parse) == normalize_tf_label(q["gold"]))
            else:
                correct_parse = (pred_parse == q["gold"])

            # score-based result
            choices = get_candidate_choices(q["mode"])
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
                pred_score_out = max(choice_scores, key=lambda x: x["seq_logprob"])
                pred_score = pred_score_out["answer"]
            else:
                pred_score_out = make_empty_score(None)
                pred_score = "UNK"

            if q["mode"] == "tf":
                correct_score = (normalize_tf_label(pred_score) == normalize_tf_label(q["gold"]))
            else:
                correct_score = (pred_score == q["gold"])

            # summary 主结果默认用 score 版
            q_correct_map[qid] = correct_score
            q_prob_map[qid] = gen_out["final_prob"]
            q_json_map[qid] = f"{qid}.json"
            q_pred_map[qid] = pred_score
            q_gold_map[qid] = q["gold"]
            q_raw_text_map[qid] = pred_text

            q_pred_parse_map[qid] = pred_parse
            q_correct_parse_map[qid] = correct_parse
            q_pred_score_map[qid] = pred_score
            q_correct_score_map[qid] = correct_score
            q_choice_scores_map[qid] = {
                x["answer"]: x["seq_logprob"] for x in choice_scores
            }

            q_record = {
                "qid": qid,
                "base_qid": q.get("base_qid"),
                "order": q.get("order"),
                "mode": q["mode"],
                "prompt_raw": q["prompt"],
                "question_text": question_text,
                "gold": q["gold"],

                "pred_text": pred_text,

                "pred_parse": pred_parse,
                "correct_parse": correct_parse,

                "pred_score": pred_score,
                "correct_score": correct_score,
                "choice_scores": q_choice_scores_map[qid],

                "free_token_ids": gen_out["token_ids"],
                "free_tokens": gen_out["tokens"],
                "free_token_probs": gen_out["token_probs"],
                "free_token_logits": gen_out["token_logits"],
                "final_prob": gen_out["final_prob"],
                "final_logit": gen_out["final_logit"],

                "target_texts": q.get("target_texts", {}),
            }

            q_json_path = os.path.join(sample_dir, f"{qid}.json")
            with open(q_json_path, "w", encoding="utf-8") as f:
                json.dump(q_record, f, indent=2, ensure_ascii=False)

            meta_out["questions"].append(q_record)

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        qids = [f"q{i}" for i in range(1, 10)]
        pattern_q1_q9 = "_".join("C" if q_correct_map.get(qid, False) else "W" for qid in qids)

        row = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "q0": "C" if q_correct_map.get("q0", False) else "W",
            "pattern_q1_q9": pattern_q1_q9,
            "num_correct_q1_q9": sum(q_correct_map.get(qid, False) for qid in qids),
        }

        for i in range(1, 10):
            row[f"q{i}"] = "C" if q_correct_map.get(f"q{i}", False) else "W"

        for qid in sorted(q_json_map.keys()):
            row[f"{qid}_json"] = os.path.join(image_stem, q_json_map[qid])

            row[f"{qid}_pred_parse"] = q_pred_parse_map.get(qid, "UNK")
            row[f"{qid}_correct_parse"] = q_correct_parse_map.get(qid, False)

            row[f"{qid}_pred_score"] = q_pred_score_map.get(qid, "UNK")
            row[f"{qid}_correct_score"] = q_correct_score_map.get(qid, False)

            row[f"{qid}_gold"] = q_gold_map.get(qid, "")
            row[f"{qid}_raw_output"] = q_raw_text_map.get(qid, "")
            row[f"{qid}_choice_scores"] = json.dumps(q_choice_scores_map.get(qid, {}), ensure_ascii=False)
            row[f"{qid}_final_prob"] = q_prob_map.get(qid, None)

        summary_rows.append(row)

        fieldnames = [
            "image_name",
            "image_path",
            "local_index",
            "q0",
            "pattern_q1_q9",
            "num_correct_q1_q9",
        ]

        for i in range(1, 10):
            fieldnames.append(f"q{i}")

        for qid in sorted(q_json_map.keys()):
            fieldnames.extend([
                f"{qid}_json",
                f"{qid}_pred_parse",
                f"{qid}_correct_parse",
                f"{qid}_pred_score",
                f"{qid}_correct_score",
                f"{qid}_gold",
                f"{qid}_raw_output",
                f"{qid}_choice_scores",
                f"{qid}_final_prob",
            ])

        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
