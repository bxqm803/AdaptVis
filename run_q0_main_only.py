import os
import re
import csv
import json
import argparse

import torch
from tqdm import tqdm

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset
from run_interrupt import (
    install_attention_perturbation,
    build_generation_trace,
    run_one_generation,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument(
        "--method",
        default="scaling_vis",
        type=str,
        choices=["scaling_vis"],
    )
    p.add_argument("--max-new-tokens", default=20, type=int)
    p.add_argument("--trace-topk", default=10, type=int)
    p.add_argument(
        "--save-first-topk",
        default=10,
        type=int,
        help="How many top candidates to save for the first generated token in summary.",
    )
    p.add_argument(
        "--out-dir",
        default="./output_q0_main_only",
        type=str,
    )
    return p.parse_args()


def clean_question_text(question: str):
    q = str(question).strip().replace("\n", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def strip_answer_clause(question: str):
    """
    Remove trailing answer constraint from the original q0 question, e.g.
    'Answer with left, right, on or under.'
    """
    q = str(question).strip()

    patterns = [
        r"\s*Answer with\s+left\s*,\s*right\s*,\s*on\s*or\s*under\.?\s*$",
        r"\s*Answer with\s+left\s*,\s*right\s*,\s*on\s*,\s*or\s*under\.?\s*$",
        r"\s*Answer with\s+under\s*,\s*right\s*,\s*left\s*or\s*on\.?\s*$",
        r"\s*Answer with\s+.*?\s+only\.?\s*$",
    ]

    for pat in patterns:
        q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()

    return q


def normalize_rel(answer):
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            raise ValueError("Empty answer list.")
        answer = answer[0]
    if answer is None:
        raise ValueError("Answer is None.")

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
    if rel not in mapping:
        raise ValueError(f"Unsupported relation answer: {answer}")
    return mapping[rel]


def parse_prediction(text: str):
    t = str(text).strip().lower()
    m = re.search(r"\b(left|right|on|under)\b", t)
    return m.group(1) if m else "UNK"


def normalize_token_text(token_text: str):
    if token_text is None:
        return ""
    x = str(token_text)
    x = x.replace("Ġ", "")
    x = x.replace("▁", "")
    x = x.strip().lower()
    return x


def extract_first_topk_candidates(trace, topn=10):
    """
    Extract top-k candidate tokens from the first generation step.

    Supports multiple possible trace schemas, including:
      - first["top10"]
      - first["topk"]
      - first["top_k"]
      - first["top_tokens"]
      - first["top_candidates"]
      - first["candidates"]
    """
    if not trace or not isinstance(trace, list):
        return []

    first = trace[0]
    if not isinstance(first, dict):
        return []

    candidate_keys = [
        "top10",
        "topk",
        "top_k",
        "top_tokens",
        "top_candidates",
        "candidates",
    ]

    candidates = None
    for key in candidate_keys:
        value = first.get(key, None)
        if isinstance(value, list):
            candidates = value
            break

    if not candidates:
        return []

    out = []
    for cand in candidates[:topn]:
        if not isinstance(cand, dict):
            continue

        token_text = (
            cand.get("token_text")
            if cand.get("token_text") is not None
            else cand.get("token")
            if cand.get("token") is not None
            else cand.get("text")
            if cand.get("text") is not None
            else cand.get("decoded")
            if cand.get("decoded") is not None
            else ""
        )

        raw_logit = (
            cand.get("raw_logit")
            if cand.get("raw_logit") is not None
            else cand.get("logit")
            if cand.get("logit") is not None
            else cand.get("raw_score")
            if cand.get("raw_score") is not None
            else None
        )

        probability = (
            cand.get("probability")
            if cand.get("probability") is not None
            else cand.get("prob")
            if cand.get("prob") is not None
            else None
        )

        out.append(
            {
                "token_text": token_text,
                "token_norm": normalize_token_text(token_text),
                "raw_logit": float(raw_logit) if raw_logit is not None else None,
                "probability": float(probability) if probability is not None else None,
            }
        )

    return out


def build_first_topk_summary_fields(trace, topn=10):
    candidates = extract_first_topk_candidates(trace, topn=topn)
    row = {
        f"first_top{topn}_json": json.dumps(candidates, ensure_ascii=False)
    }

    for i in range(1, topn + 1):
        if i <= len(candidates):
            c = candidates[i - 1]
            row[f"first_top{i}_token"] = c.get("token_text", "")
            row[f"first_top{i}_token_norm"] = c.get("token_norm", "")
            row[f"first_top{i}_raw_logit"] = c.get("raw_logit", None)
            row[f"first_top{i}_probability"] = c.get("probability", None)
        else:
            row[f"first_top{i}_token"] = ""
            row[f"first_top{i}_token_norm"] = ""
            row[f"first_top{i}_raw_logit"] = None
            row[f"first_top{i}_probability"] = None

    return row


def build_summary_fieldnames(save_first_topk=10):
    fieldnames = [
        "image_mode",
        "image_name",
        "image_path",
        "local_index",
        "qid",
        "raw_question",
        "base_question",
        "prompt_text",
        "gold",
        "pred",
        "correct",
        "perturb_mode",
        "target_layers",
        "generated_text",
        "num_generated_tokens",
        "first_token",
        "first_raw_logit",
        "first_score",
        "first_probability",
        f"first_top{save_first_topk}_json",
    ]

    for i in range(1, save_first_topk + 1):
        fieldnames.extend(
            [
                f"first_top{i}_token",
                f"first_top{i}_token_norm",
                f"first_top{i}_raw_logit",
                f"first_top{i}_probability",
            ]
        )

    fieldnames.extend(
        [
            "final_token",
            "final_raw_logit",
            "final_score",
            "final_probability",
            "trace_json",
        ]
    )
    return fieldnames


def main():
    args = parse_args()
    seed_all(args.seed)

    install_attention_perturbation(target_layers=None)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=args.download,
    )
    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(
        args.dataset,
        args.option,
    )

    if sampled_indices is not None:
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        sub_dataset = dataset

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    base_dir = os.path.join(args.out_dir, args.dataset, "original_q0_main_only")
    os.makedirs(base_dir, exist_ok=True)

    summary_csv = os.path.join(base_dir, "summary_q0_main_only.csv")
    summary_rows = []
    summary_fieldnames = build_summary_fieldnames(save_first_topk=args.save_first_topk)
    trace_topk_to_use = max(args.trace_topk, args.save_first_topk)

    for local_idx in tqdm(range(start, end), desc="Samples"):
        rec = prompt_records[local_idx]
        item = sub_dataset[local_idx]

        image = item["image_options"][0]
        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")
        image_stem = os.path.splitext(image_name)[0]

        sample_dir = os.path.join(base_dir, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        gold = normalize_rel(rec["answer"])

        raw_question = clean_question_text(rec["question"])
        base_question = strip_answer_clause(raw_question)

        prompt_text = base_question
        prompt = f"<image>\nUSER: {prompt_text}\nASSISTANT:"

        output, gen_text, prompt_len = run_one_generation(
            wrapper=wrapper,
            model_name=args.model_name,
            image=image,
            prompt=prompt,
            perturb_mode="none",
            max_new_tokens=args.max_new_tokens,
        )

        trace = build_generation_trace(
            wrapper.processor,
            output,
            prompt_len,
            topk=trace_topk_to_use,
        )

        pred = parse_prediction(gen_text)
        correct = pred == gold

        trace_rel_path = os.path.join(
            image_stem,
            "q0_main_original_none_trace.json",
        )
        trace_path = os.path.join(base_dir, trace_rel_path)

        first_topk_fields = build_first_topk_summary_fields(
            trace,
            topn=args.save_first_topk,
        )

        write_json(
            trace_path,
            {
                "image_mode": "original",
                "image_name": image_name,
                "image_path": image_path,
                "local_index": local_idx,
                "qid": "q0",
                "raw_question": raw_question,
                "base_question": base_question,
                "prompt_text": prompt_text,
                "prompt": prompt,
                "gold": gold,
                "generated_text": gen_text,
                "pred": pred,
                "correct": correct,
                "perturb_mode": "none",
                "target_layers": "all",
                "first_topk_summary": json.loads(
                    first_topk_fields[f"first_top{args.save_first_topk}_json"]
                ),
                "token_trace": trace,
            },
        )

        meta_path = os.path.join(sample_dir, "meta_q0_main_only.json")
        if not os.path.exists(meta_path):
            write_json(
                meta_path,
                {
                    "local_index": local_idx,
                    "image_name": image_name,
                    "image_path": image_path,
                    "image_mode": "original",
                    "qid": "q0",
                    "raw_question": raw_question,
                    "base_question": base_question,
                    "gold": gold,
                    "prompt_text": prompt_text,
                    "perturb_mode": "none",
                    "target_layers": "all",
                },
            )

        first = trace[0] if trace else {}
        last = trace[-1] if trace else {}

        row = {
            "image_mode": "original",
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "qid": "q0",
            "raw_question": raw_question,
            "base_question": base_question,
            "prompt_text": prompt_text,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "perturb_mode": "none",
            "target_layers": "all",
            "generated_text": gen_text,
            "num_generated_tokens": len(trace),
            "first_token": first.get("token_text", ""),
            "first_raw_logit": first.get("raw_logit", None),
            "first_score": first.get("score", None),
            "first_probability": first.get("probability", None),
            "final_token": last.get("token_text", ""),
            "final_raw_logit": last.get("raw_logit", None),
            "final_score": last.get("score", None),
            "final_probability": last.get("probability", None),
            "trace_json": trace_rel_path,
        }
        row.update(first_topk_fields)
        summary_rows.append(row)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
