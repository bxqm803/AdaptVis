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
    parser.add_argument("--only-q0", action="store_true", help="Only run the original question q0.")
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


def extract_post_think_text(text: str):
    """
    兼容两种情况：
    1) 有 </think>：取后半段
    2) 没有 </think>：直接返回原始输出，避免 Qwen2.5 全部变成 UNK
    """
    if text is None:
        return "", False

    text = str(text).strip()
    if not text:
        return "", False

    low = text.lower()
    tag = "</think>"
    idx = low.rfind(tag)

    if idx == -1:
        return text, False

    suffix = text[idx + len(tag):].strip()
    if suffix:
        return suffix, True

    return text, True


def parse_answer_from_text(text: str, mode: str):
    if text is None:
        return "UNK"

    t = text.strip().lower()
    if not t:
        return "UNK"

    if mode == "orig":
        matches = list(re.finditer(r"\b(left|right|on|under)\b", t))
        return matches[-1].group(1) if matches else "UNK"

    matches = list(re.finditer(r"\b(true|false|t|f)\b", t))
    if not matches:
        return "UNK"
    tok = matches[-1].group(1)
    return "True" if tok in {"t", "true"} else "False"


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

        if args.only_q0:
            questions = [q for q in questions if q["qid"] == "q0"]

        sample_dir = os.path.join(out_root, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        meta_out = {
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            **meta,
            "questions": [],
        }

        q_pred_postthink_map = {}
        q_correct_postthink_map = {}
        q_gold_map = {}
        q_raw_text_map = {}
        q_postthink_text_map = {}
        q_has_think_close_map = {}
        q_json_map = {}
        q_final_prob_map = {}

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

            # 关键修复：没有 </think> 时，直接用原始输出
            postthink_text, has_think_close = extract_post_think_text(pred_text)
            pred_postthink = parse_answer_from_text(postthink_text, q["mode"])

            if q["mode"] == "tf":
                correct_postthink = (normalize_tf_label(pred_postthink) == normalize_tf_label(q["gold"]))
            else:
                correct_postthink = (pred_postthink == q["gold"])

            q_pred_postthink_map[qid] = pred_postthink
            q_correct_postthink_map[qid] = correct_postthink
            q_gold_map[qid] = q["gold"]
            q_raw_text_map[qid] = pred_text
            q_postthink_text_map[qid] = postthink_text
            q_has_think_close_map[qid] = has_think_close
            q_json_map[qid] = f"{qid}.json"
            q_final_prob_map[qid] = gen_out["final_prob"]

            q_record = {
                "qid": qid,
                "base_qid": q.get("base_qid"),
                "order": q.get("order"),
                "mode": q["mode"],
                "prompt_raw": q["prompt"],
                "question_text": question_text,
                "gold": q["gold"],
                "pred_text": pred_text,
                "has_think_close": has_think_close,
                "postthink_text": postthink_text,
                "pred_postthink": pred_postthink,
                "correct_postthink": correct_postthink,
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

        row = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "q0_postthink": "C" if q_correct_postthink_map.get("q0", False) else "W",
        }

        if not args.only_q0:
            qids = [f"q{i}" for i in range(1, 10)]
            pattern_q1_q9_postthink = "_".join(
                "C" if q_correct_postthink_map.get(qid, False) else "W"
                for qid in qids
            )
            row["pattern_q1_q9_postthink"] = pattern_q1_q9_postthink
            row["num_correct_q1_q9_postthink"] = sum(
                q_correct_postthink_map.get(qid, False) for qid in qids
            )
            for i in range(1, 10):
                row[f"q{i}_postthink"] = "C" if q_correct_postthink_map.get(f"q{i}", False) else "W"

        for qid in sorted(q_json_map.keys()):
            row[f"{qid}_json"] = os.path.join(image_stem, q_json_map[qid])
            row[f"{qid}_gold"] = q_gold_map.get(qid, "")
            row[f"{qid}_raw_output"] = q_raw_text_map.get(qid, "")
            row[f"{qid}_has_think_close"] = q_has_think_close_map.get(qid, False)
            row[f"{qid}_postthink_text"] = q_postthink_text_map.get(qid, "")
            row[f"{qid}_pred_postthink"] = q_pred_postthink_map.get(qid, "UNK")
            row[f"{qid}_correct_postthink"] = q_correct_postthink_map.get(qid, False)
            row[f"{qid}_final_prob"] = q_final_prob_map.get(qid, None)

        summary_rows.append(row)

        fieldnames = [
            "image_name",
            "image_path",
            "local_index",
            "q0_postthink",
        ]

        if not args.only_q0:
            fieldnames.extend([
                "pattern_q1_q9_postthink",
                "num_correct_q1_q9_postthink",
            ])
            for i in range(1, 10):
                fieldnames.append(f"q{i}_postthink")

        for qid in sorted(q_json_map.keys()):
            fieldnames.extend([
                f"{qid}_json",
                f"{qid}_gold",
                f"{qid}_raw_output",
                f"{qid}_has_think_close",
                f"{qid}_postthink_text",
                f"{qid}_pred_postthink",
                f"{qid}_correct_postthink",
                f"{qid}_final_prob",
            ])

        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
