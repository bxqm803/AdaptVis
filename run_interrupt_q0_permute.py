import os
import re
import csv
import json
import argparse
import itertools

import torch
from tqdm import tqdm

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset

# 复用原 run_interrupt.py 里的工具函数
from run_interrupt import (
    parse_mode_list,
    parse_target_layers,
    install_attention_perturbation,
    make_black_image_like,
    build_generation_trace,
    run_one_generation,
    write_json,
)


CHOICES = ["left", "right", "on", "under"]


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
        help="Use the Scal LLaVA path so custom kwargs reach decoder attention.",
    )
    p.add_argument(
        "--perturb-modes",
        default="none,uniform,random,reverse",
        type=str,
        help="Comma-separated list from {none,uniform,random,reverse}.",
    )
    p.add_argument(
        "--target-layers",
        default="all",
        type=str,
        help='"all" or comma-separated decoder layer indices, e.g. "16" or "8,16,24".',
    )
    p.add_argument(
        "--image-mode",
        default="original",
        type=str,
        choices=["original", "black"],
        help='Input image mode: "original" or "black".',
    )
    p.add_argument("--max-new-tokens", default=20, type=int)
    p.add_argument("--trace-topk", default=10, type=int)
    p.add_argument("--out-dir", default="./output_interrupt_q0_permute", type=str)
    return p.parse_args()


def clean_question_text(question: str):
    q = question.strip().replace("\n", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    q = re.sub(r"\s+", " ", q).strip()
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
    t = text.strip().lower()
    m = re.search(r"\b(left|right|on|under)\b", t)
    return m.group(1) if m else "UNK"


def strip_answer_order_clause(question_text: str):
    q = clean_question_text(question_text)
    q = re.sub(
        r"Answer with\s+left,\s*right,\s*on\s+or\s+under\.\s*$",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q


def build_q0_permuted_prompts(base_prompt):
    stem = strip_answer_order_clause(base_prompt)
    out = []
    for perm in itertools.permutations(CHOICES):
        order_text = ", ".join(perm[:-1]) + f" or {perm[-1]}"
        prompt_text = f"{stem} Answer with {order_text} only."
        prompt = f"<image>\nUSER: {prompt_text}\nASSISTANT:"
        perm_id = "_".join(perm)
        out.append({
            "qid": "q0",
            "perm_id": perm_id,
            "order": list(perm),
            "prompt_text": prompt_text,
            "prompt": prompt,
        })
    return out


def main():
    args = parse_args()
    seed_all(args.seed)

    perturb_modes = parse_mode_list(args.perturb_modes)
    target_layers = parse_target_layers(args.target_layers)
    install_attention_perturbation(target_layers)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)

    if sampled_indices is not None:
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        sub_dataset = dataset

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    base_dir = os.path.join(args.out_dir, args.dataset, args.image_mode)
    os.makedirs(base_dir, exist_ok=True)
    summary_csv = os.path.join(base_dir, f"summary_{args.image_mode}.csv")
    summary_rows = []

    for local_idx in tqdm(range(start, end), desc="Samples"):
        rec = prompt_records[local_idx]
        item = sub_dataset[local_idx]

        image = item["image_options"][0]
        if args.image_mode == "black":
            image = make_black_image_like(image)

        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")
        image_stem = os.path.splitext(image_name)[0]
        sample_dir = os.path.join(base_dir, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        gold = normalize_rel(rec["answer"])
        base_question = clean_question_text(rec["question"])
        permuted_prompts = build_q0_permuted_prompts(rec["question"])

        meta_path = os.path.join(sample_dir, "meta_q0_permutations.json")
        if not os.path.exists(meta_path):
            write_json(meta_path, {
                "local_index": local_idx,
                "image_name": image_name,
                "image_path": image_path,
                "image_mode": args.image_mode,
                "base_question": base_question,
                "gold": gold,
                "num_permutations": len(permuted_prompts),
                "permutations": [
                    {"perm_id": x["perm_id"], "order": x["order"], "prompt_text": x["prompt_text"]}
                    for x in permuted_prompts
                ],
            })

        for q in permuted_prompts:
            for perturb_mode in perturb_modes:
                output, gen_text, prompt_len = run_one_generation(
                    wrapper=wrapper,
                    model_name=args.model_name,
                    image=image,
                    prompt=q["prompt"],
                    perturb_mode=perturb_mode,
                    max_new_tokens=args.max_new_tokens,
                )

                trace = build_generation_trace(
                    wrapper.processor,
                    output,
                    prompt_len,
                    topk=args.trace_topk,
                )
                pred = parse_prediction(gen_text)
                correct = (pred == gold)

                trace_rel_path = os.path.join(
                    image_stem,
                    f"q0_{q['perm_id']}_{args.image_mode}_{perturb_mode}_trace.json",
                )
                trace_path = os.path.join(base_dir, trace_rel_path)

                write_json(trace_path, {
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "qid": "q0",
                    "perm_id": q["perm_id"],
                    "order": q["order"],
                    "base_question": base_question,
                    "prompt_text": q["prompt_text"],
                    "gold": gold,
                    "prompt": q["prompt"],
                    "image_mode": args.image_mode,
                    "perturb_mode": perturb_mode,
                    "target_layers": "all" if target_layers is None else sorted(target_layers),
                    "generated_text": gen_text,
                    "pred": pred,
                    "correct": correct,
                    "token_trace": trace,
                })

                first = trace[0] if trace else {}
                last = trace[-1] if trace else {}

                summary_rows.append({
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "qid": "q0",
                    "perm_id": q["perm_id"],
                    "order_text": " | ".join(q["order"]),
                    "base_question": base_question,
                    "prompt_text": q["prompt_text"],
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "image_mode": args.image_mode,
                    "perturb_mode": perturb_mode,
                    "target_layers": "all" if target_layers is None else ",".join(str(x) for x in sorted(target_layers)),
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
                })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "image_path",
                "local_index",
                "qid",
                "perm_id",
                "order_text",
                "base_question",
                "prompt_text",
                "gold",
                "pred",
                "correct",
                "image_mode",
                "perturb_mode",
                "target_layers",
                "generated_text",
                "num_generated_tokens",
                "first_token",
                "first_raw_logit",
                "first_score",
                "first_probability",
                "final_token",
                "final_raw_logit",
                "final_score",
                "final_probability",
                "trace_json",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
