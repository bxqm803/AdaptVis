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
    p.add_argument("--open-max-new-tokens", default=40, type=int)
    p.add_argument("--mcq-max-new-tokens", default=20, type=int)
    p.add_argument("--trace-topk", default=10, type=int)
    p.add_argument(
        "--save-first-topk",
        default=10,
        type=int,
        help="How many top candidates to save for the first generated token in summary.",
    )
    p.add_argument(
        "--out-dir",
        default="./output_q0_open_then_mcq",
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


def build_first_topk_summary_fields(trace, prefix, topn=10):
    candidates = extract_first_topk_candidates(trace, topn=topn)
    row = {
        f"{prefix}_first_top{topn}_json": json.dumps(candidates, ensure_ascii=False)
    }

    for i in range(1, topn + 1):
        if i <= len(candidates):
            c = candidates[i - 1]
            row[f"{prefix}_first_top{i}_token"] = c.get("token_text", "")
            row[f"{prefix}_first_top{i}_token_norm"] = c.get("token_norm", "")
            row[f"{prefix}_first_top{i}_raw_logit"] = c.get("raw_logit", None)
            row[f"{prefix}_first_top{i}_probability"] = c.get("probability", None)
        else:
            row[f"{prefix}_first_top{i}_token"] = ""
            row[f"{prefix}_first_top{i}_token_norm"] = ""
            row[f"{prefix}_first_top{i}_raw_logit"] = None
            row[f"{prefix}_first_top{i}_probability"] = None

    return row


def add_stage_fieldnames(fieldnames, prefix, save_first_topk=10):
    stage_fields = [
        f"{prefix}_prompt_text",
        f"{prefix}_generated_text",
        f"{prefix}_num_generated_tokens",
        f"{prefix}_first_token",
        f"{prefix}_first_raw_logit",
        f"{prefix}_first_score",
        f"{prefix}_first_probability",
        f"{prefix}_first_top{save_first_topk}_json",
    ]

    for i in range(1, save_first_topk + 1):
        stage_fields.extend(
            [
                f"{prefix}_first_top{i}_token",
                f"{prefix}_first_top{i}_token_norm",
                f"{prefix}_first_top{i}_raw_logit",
                f"{prefix}_first_top{i}_probability",
            ]
        )

    stage_fields.extend(
        [
            f"{prefix}_final_token",
            f"{prefix}_final_raw_logit",
            f"{prefix}_final_score",
            f"{prefix}_final_probability",
            f"{prefix}_trace_json",
        ]
    )
    return fieldnames + stage_fields


def build_summary_fieldnames(save_first_topk=10):
    fieldnames = [
        "image_mode",
        "image_name",
        "image_path",
        "local_index",
        "qid",
        "raw_question",
        "base_question",
        "gold",
        "pred",
        "correct",
        "perturb_mode",
        "target_layers",
        "open_answer_as_caption",
    ]

    fieldnames = add_stage_fieldnames(fieldnames, "open", save_first_topk)
    fieldnames = add_stage_fieldnames(fieldnames, "mcq", save_first_topk)

    return fieldnames


def trace_basic_fields(trace, prefix):
    first = trace[0] if trace else {}
    last = trace[-1] if trace else {}
    return {
        f"{prefix}_num_generated_tokens": len(trace),
        f"{prefix}_first_token": first.get("token_text", ""),
        f"{prefix}_first_raw_logit": first.get("raw_logit", None),
        f"{prefix}_first_score": first.get("score", None),
        f"{prefix}_first_probability": first.get("probability", None),
        f"{prefix}_final_token": last.get("token_text", ""),
        f"{prefix}_final_raw_logit": last.get("raw_logit", None),
        f"{prefix}_final_score": last.get("score", None),
        f"{prefix}_final_probability": last.get("probability", None),
    }


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

    base_dir = os.path.join(args.out_dir, args.dataset, "original_q0_open_then_mcq")
    os.makedirs(base_dir, exist_ok=True)

    summary_csv = os.path.join(base_dir, "summary_q0_open_then_mcq.csv")
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

        # ---------------------------
        # Stage 1: open-ended question
        # ---------------------------
        open_prompt_text = base_question
        open_prompt = f"<image>\nUSER: {open_prompt_text}\nASSISTANT:"

        open_output, open_gen_text, open_prompt_len = run_one_generation(
            wrapper=wrapper,
            model_name=args.model_name,
            image=image,
            prompt=open_prompt,
            perturb_mode="none",
            max_new_tokens=args.open_max_new_tokens,
        )

        open_trace = build_generation_trace(
            wrapper.processor,
            open_output,
            open_prompt_len,
            topk=trace_topk_to_use,
        )

        open_trace_rel_path = os.path.join(
            image_stem,
            "q0_open_original_none_trace.json",
        )
        open_trace_path = os.path.join(base_dir, open_trace_rel_path)

        open_first_topk_fields = build_first_topk_summary_fields(
            open_trace,
            prefix="open",
            topn=args.save_first_topk,
        )

        write_json(
            open_trace_path,
            {
                "stage": "open",
                "image_mode": "original",
                "image_name": image_name,
                "image_path": image_path,
                "local_index": local_idx,
                "qid": "q0",
                "raw_question": raw_question,
                "base_question": base_question,
                "prompt_text": open_prompt_text,
                "prompt": open_prompt,
                "gold": gold,
                "generated_text": open_gen_text,
                "perturb_mode": "none",
                "target_layers": "all",
                "first_topk_summary": json.loads(
                    open_first_topk_fields[f"open_first_top{args.save_first_topk}_json"]
                ),
                "token_trace": open_trace,
            },
        )

        # -----------------------------------------
        # Stage 2: use open answer as caption, then
        # ask original constrained question
        # -----------------------------------------
        mcq_prompt_text = f"Caption: {open_gen_text.strip()}\nQuestion: {raw_question}"
        mcq_prompt = f"<image>\nUSER: {mcq_prompt_text}\nASSISTANT:"

        mcq_output, mcq_gen_text, mcq_prompt_len = run_one_generation(
            wrapper=wrapper,
            model_name=args.model_name,
            image=image,
            prompt=mcq_prompt,
            perturb_mode="none",
            max_new_tokens=args.mcq_max_new_tokens,
        )

        mcq_trace = build_generation_trace(
            wrapper.processor,
            mcq_output,
            mcq_prompt_len,
            topk=trace_topk_to_use,
        )

        pred = parse_prediction(mcq_gen_text)
        correct = pred == gold

        mcq_trace_rel_path = os.path.join(
            image_stem,
            "q0_mcq_from_open_caption_original_none_trace.json",
        )
        mcq_trace_path = os.path.join(base_dir, mcq_trace_rel_path)

        mcq_first_topk_fields = build_first_topk_summary_fields(
            mcq_trace,
            prefix="mcq",
            topn=args.save_first_topk,
        )

        write_json(
            mcq_trace_path,
            {
                "stage": "mcq_from_open_caption",
                "image_mode": "original",
                "image_name": image_name,
                "image_path": image_path,
                "local_index": local_idx,
                "qid": "q0",
                "raw_question": raw_question,
                "base_question": base_question,
                "open_answer_as_caption": open_gen_text,
                "prompt_text": mcq_prompt_text,
                "prompt": mcq_prompt,
                "gold": gold,
                "generated_text": mcq_gen_text,
                "pred": pred,
                "correct": correct,
                "perturb_mode": "none",
                "target_layers": "all",
                "first_topk_summary": json.loads(
                    mcq_first_topk_fields[f"mcq_first_top{args.save_first_topk}_json"]
                ),
                "token_trace": mcq_trace,
            },
        )

        meta_path = os.path.join(sample_dir, "meta_q0_open_then_mcq.json")
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
                    "perturb_mode": "none",
                    "target_layers": "all",
                },
            )

        row = {
            "image_mode": "original",
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "qid": "q0",
            "raw_question": raw_question,
            "base_question": base_question,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "perturb_mode": "none",
            "target_layers": "all",
            "open_answer_as_caption": open_gen_text,
            "open_prompt_text": open_prompt_text,
            "open_generated_text": open_gen_text,
            "open_trace_json": open_trace_rel_path,
            "mcq_prompt_text": mcq_prompt_text,
            "mcq_generated_text": mcq_gen_text,
            "mcq_trace_json": mcq_trace_rel_path,
        }

        row.update(trace_basic_fields(open_trace, "open"))
        row.update(trace_basic_fields(mcq_trace, "mcq"))
        row.update(open_first_topk_fields)
        row.update(mcq_first_topk_fields)

        summary_rows.append(row)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
