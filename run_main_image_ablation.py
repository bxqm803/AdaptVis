import os
import re
import csv
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset

# 复用现有工具
from run_interrupt import (
    make_black_image_like,
    build_generation_trace,
    write_json,
)

LABELS = ["left", "right", "on", "under"]
VALID_IMAGE_MODES = {"original", "black", "no_image"}


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
        help="Use the same model loading path as the repo.",
    )
    p.add_argument(
        "--image-modes",
        default="original,black,no_image",
        type=str,
        help='Comma-separated list from {original, black, no_image}.',
    )
    p.add_argument("--max-new-tokens", default=20, type=int)
    p.add_argument("--trace-topk", default=10, type=int)
    p.add_argument("--out-dir", default="./output_image_ablation", type=str)
    return p.parse_args()


def parse_mode_list(text: str):
    modes = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not modes:
        raise ValueError("No image modes provided.")
    bad = [m for m in modes if m not in VALID_IMAGE_MODES]
    if bad:
        raise ValueError(f"Unsupported image modes: {bad}")
    return modes


def clean_question_text(question: str):
    q = question.strip().replace("\n", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def remove_image_token_from_prompt(prompt: str):
    # 原 prompt 常见格式：<image>\nUSER: ...\nASSISTANT:
    text = prompt.replace("<image>\n", "")
    text = text.replace("<image>", "")
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


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


@torch.no_grad()
def run_one_generation_ablation(
    wrapper,
    image,
    prompt: str,
    image_mode: str,
    max_new_tokens: int,
):
    processor = wrapper.processor
    model = wrapper.model
    device = wrapper.device

    if image_mode == "no_image":
        prompt_for_model = remove_image_token_from_prompt(prompt)
        single_input = processor(text=prompt_for_model, padding=True, return_tensors="pt")
    else:
        img = image if image_mode == "original" else make_black_image_like(image)
        prompt_for_model = prompt
        single_input = processor(images=img, text=prompt_for_model, padding=True, return_tensors="pt")

    single_input = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in single_input.items()
        if v is not None
    }

    prompt_len = int(single_input["input_ids"].shape[1])

    gen_kwargs = dict(
        input_ids=single_input["input_ids"],
        attention_mask=single_input.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
        output_attentions=False,
    )

    # 只有有图时才传 pixel_values
    if "pixel_values" in single_input:
        gen_kwargs["pixel_values"] = single_input["pixel_values"]

    output = model.generate(**gen_kwargs)

    gen_ids = output.sequences[0][prompt_len:]
    gen_text = processor.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return output, gen_text, prompt_len, prompt_for_model


def summarize_mode(df_mode: pd.DataFrame):
    total = len(df_mode)
    ok = int(df_mode["correct"].fillna(False).sum())
    acc = ok / total if total > 0 else float("nan")

    gold_counts = Counter(df_mode["gold"])
    pred_counts = Counter(df_mode["pred"])

    first_logit_mean = pd.to_numeric(df_mode["first_raw_logit"], errors="coerce").mean()
    final_logit_mean = pd.to_numeric(df_mode["final_raw_logit"], errors="coerce").mean()
    first_prob_mean = pd.to_numeric(df_mode["first_probability"], errors="coerce").mean()
    final_prob_mean = pd.to_numeric(df_mode["final_probability"], errors="coerce").mean()

    per_gold = []
    for lbl in LABELS:
        sub = df_mode[df_mode["gold"] == lbl]
        n = len(sub)
        acc_lbl = sub["correct"].fillna(False).mean() if n > 0 else float("nan")
        per_gold.append((lbl, n, acc_lbl))

    cm = pd.crosstab(df_mode["gold"], df_mode["pred"], dropna=False)
    for lbl in LABELS:
        if lbl not in cm.index:
            cm.loc[lbl] = 0
        if lbl not in cm.columns:
            cm[lbl] = 0
    cm = cm.reindex(index=LABELS, columns=LABELS, fill_value=0)

    return {
        "total": total,
        "correct": ok,
        "acc": acc,
        "gold_counts": gold_counts,
        "pred_counts": pred_counts,
        "first_logit_mean": first_logit_mean,
        "final_logit_mean": final_logit_mean,
        "first_prob_mean": first_prob_mean,
        "final_prob_mean": final_prob_mean,
        "per_gold": per_gold,
        "cm": cm,
    }


def write_report(path: str, df: pd.DataFrame, image_modes):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"rows_total: {len(df)}\n")
        f.write(f"image_modes: {image_modes}\n\n")

        for mode in image_modes:
            f.write("=" * 120 + "\n")
            f.write(f"IMAGE MODE: {mode}\n")
            f.write("=" * 120 + "\n")

            sub = df[df["image_mode"] == mode].copy()

            if len(sub) == 0:
                f.write("No rows.\n\n")
                continue

            errs = sub["status"].fillna("ok").value_counts().to_dict()
            f.write(f"status_counts: {errs}\n\n")

            sub_ok = sub[sub["status"] == "ok"].copy()
            if len(sub_ok) == 0:
                f.write("No successful generations.\n\n")
                continue

            stats = summarize_mode(sub_ok)
            f.write(f"total_ok: {stats['total']}\n")
            f.write(f"correct: {stats['correct']}\n")
            f.write(f"accuracy: {stats['acc']:.6f}\n")
            f.write(f"mean_first_raw_logit: {stats['first_logit_mean']:.6f}\n")
            f.write(f"mean_final_raw_logit: {stats['final_logit_mean']:.6f}\n")
            f.write(f"mean_first_probability: {stats['first_prob_mean']:.6f}\n")
            f.write(f"mean_final_probability: {stats['final_prob_mean']:.6f}\n\n")

            f.write("gold_counts:\n")
            for lbl in LABELS:
                f.write(f"  {lbl}: {stats['gold_counts'].get(lbl, 0)}\n")
            f.write("\n")

            f.write("pred_counts:\n")
            for lbl in LABELS + ["UNK"]:
                if stats["pred_counts"].get(lbl, 0) > 0:
                    f.write(f"  {lbl}: {stats['pred_counts'].get(lbl, 0)}\n")
            f.write("\n")

            f.write("per-gold accuracy:\n")
            for lbl, n, acc_lbl in stats["per_gold"]:
                f.write(f"  {lbl}: n={n}, acc={acc_lbl:.6f}\n")
            f.write("\n")

            f.write("confusion_matrix (gold x pred):\n")
            f.write(stats["cm"].to_string())
            f.write("\n\n")

            # 错例中最常见的预测
            wrong = sub_ok[sub_ok["correct"] == False]
            f.write(f"wrong_examples: {len(wrong)}\n")
            if len(wrong) > 0:
                wrong_pred_counts = wrong["pred"].value_counts().to_dict()
                f.write(f"wrong_pred_distribution: {wrong_pred_counts}\n")
            f.write("\n")


def main():
    args = parse_args()
    seed_all(args.seed)
    image_modes = parse_mode_list(args.image_modes)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    # 复用 repo 现有 prompt 抽样逻辑
    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        sub_dataset = dataset

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    base_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(base_dir, exist_ok=True)

    summary_csv = os.path.join(base_dir, "summary_image_ablation.csv")
    report_txt = os.path.join(base_dir, "report_image_ablation.txt")
    summary_rows = []

    for local_idx in tqdm(range(start, end), desc="Samples"):
        rec = prompt_records[local_idx]
        item = sub_dataset[local_idx]

        image = item["image_options"][0]
        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")

        prompt = rec["question"]
        question_clean = clean_question_text(prompt)
        gold = normalize_rel(rec["answer"][0] if isinstance(rec["answer"], list) else rec["answer"])

        for image_mode in image_modes:
            row = {
                "image_name": image_name,
                "image_path": image_path,
                "local_index": local_idx,
                "image_mode": image_mode,
                "question": question_clean,
                "gold": gold,
                "status": "ok",
                "error": "",
            }

            try:
                output, gen_text, prompt_len, prompt_used = run_one_generation_ablation(
                    wrapper=wrapper,
                    image=image,
                    prompt=prompt,
                    image_mode=image_mode,
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

                first = trace[0] if trace else {}
                last = trace[-1] if trace else {}

                trace_rel_path = os.path.join(
                    image_mode,
                    os.path.splitext(image_name)[0],
                    "q0_trace.json",
                )
                trace_path = os.path.join(base_dir, trace_rel_path)

                write_json(trace_path, {
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "image_mode": image_mode,
                    "question": question_clean,
                    "prompt_used": prompt_used,
                    "gold": gold,
                    "generated_text": gen_text,
                    "pred": pred,
                    "correct": correct,
                    "token_trace": trace,
                })

                row.update({
                    "prompt_used": prompt_used,
                    "generated_text": gen_text,
                    "pred": pred,
                    "correct": correct,
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
            except Exception as e:
                row.update({
                    "status": "error",
                    "error": repr(e),
                    "prompt_used": remove_image_token_from_prompt(prompt) if image_mode == "no_image" else prompt,
                    "generated_text": "",
                    "pred": "UNK",
                    "correct": False,
                    "num_generated_tokens": 0,
                    "first_token": "",
                    "first_raw_logit": None,
                    "first_score": None,
                    "first_probability": None,
                    "final_token": "",
                    "final_raw_logit": None,
                    "final_score": None,
                    "final_probability": None,
                    "trace_json": "",
                })

            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_report(report_txt, df, image_modes)

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved report to: {report_txt}")


if __name__ == "__main__":
    main()
