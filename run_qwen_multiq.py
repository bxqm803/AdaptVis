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
    if x in {"t", "true", "yes"}:
        return "True"
    if x in {"f", "false", "no"}:
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

    out_root = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []
    summary_csv = os.path.join(out_root, "summary.csv")

    for local_idx in tqdm(range(args.start_index, end_index), desc=f"{args.dataset}:{args.model_id}"):
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
            pred = parse_prediction(pred_text, q["mode"])

            if q["mode"] == "tf":
                correct = (normalize_tf_label(pred) == normalize_tf_label(q["gold"]))
            else:
                correct = (pred == q["gold"])

            q_correct_map[qid] = correct
            q_prob_map[qid] = gen_out["token_probs"][-1] if len(gen_out["token_probs"]) > 0 else None

            q_json_name = f"{qid}.json"
            q_json_path = os.path.join(sample_dir, q_json_name)
            q_json_map[qid] = q_json_name

            q_record = {
                "qid": qid,
                "mode": q["mode"],
                "prompt_raw": q["prompt"],
                "question_text": question_text,
                "gold": q["gold"],
                "pred_text": pred_text,
                "pred": pred,
                "correct": correct,
                "free_token_ids": gen_out["token_ids"],
                "free_tokens": gen_out["tokens"],
                "free_token_probs": gen_out["token_probs"],
                "free_token_logits": gen_out["token_logits"],
                "final_prob": q_prob_map[qid],
                "target_texts": q.get("target_texts", {}),
            }

            with open(q_json_path, "w", encoding="utf-8") as f:
                json.dump(q_record, f, indent=2, ensure_ascii=False)

            meta_out["questions"].append(q_record)

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        pattern_q1_q9 = "_".join(
            "C" if q_correct_map.get(f"q{i}", False) else "W"
            for i in range(1, 10)
        )

        row = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "q1": "C" if q_correct_map.get("q1", False) else "W",
            "q2": "C" if q_correct_map.get("q2", False) else "W",
            "q3": "C" if q_correct_map.get("q3", False) else "W",
            "q4": "C" if q_correct_map.get("q4", False) else "W",
            "q5": "C" if q_correct_map.get("q5", False) else "W",
            "q6": "C" if q_correct_map.get("q6", False) else "W",
            "q7": "C" if q_correct_map.get("q7", False) else "W",
            "q8": "C" if q_correct_map.get("q8", False) else "W",
            "q9": "C" if q_correct_map.get("q9", False) else "W",
            "pattern_q1_q9": pattern_q1_q9,
            "num_correct_q1_q9": sum(q_correct_map.get(f"q{i}", False) for i in range(1, 10)),
        }

        for i in range(10):
            qid = f"q{i}"
            row[f"{qid}_json"] = os.path.join(image_stem, q_json_map.get(qid, ""))
            row[f"{qid}_final_prob"] = q_prob_map.get(qid, None)

        summary_rows.append(row)

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image_name",
                "image_path",
                "local_index",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "q9",
                "pattern_q1_q9",
                "num_correct_q1_q9",
            ]
            for i in range(10):
                fieldnames.extend([f"q{i}_json", f"q{i}_final_prob"])

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
