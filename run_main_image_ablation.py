import os
import re
import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset

from run_interrupt import (
    make_black_image_like,
    build_generation_trace,
    write_json,
    run_one_generation,
)

VALID_IMAGE_MODES = {"original", "black"}
LABELS = ["left", "right", "on", "under"]


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
    p.add_argument(
        "--image-modes",
        default="original,black",
        type=str,
        help="Comma-separated list from {original, black}.",
    )
    p.add_argument("--max-new-tokens", default=20, type=int)
    p.add_argument("--trace-topk", default=10, type=int)
    p.add_argument("--out-dir", default="./output_image_ablation_llava15", type=str)
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


def summarize_mode(df_mode: pd.DataFrame):
    total = len(df_mode)
    ok = int(df_mode["correct"].fillna(False).sum())
    acc = ok / total if total > 0 else float("nan")

    gold_counts = df_mode["gold"].value_counts().to_dict()
    pred_counts = df_mode["pred"].value_counts().to_dict()

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

            if "error" in errs:
                f.write("top_errors:\n")
                top_errors = sub[sub["status"] == "error"]["error"].value_counts().head(10)
                for msg, cnt in top_errors.items():
                    f.write(f"  [{cnt}] {msg}\n")
                f.write("\n")

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


def main():
    args = parse_args()
    seed_all(args.seed)
    image_modes = parse_mode_list(args.image_modes)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        sub_dataset = pd.Series(range(len(sampled_indices)))
        dataset_view = [dataset[i] for i in sampled_indices]
    else:
        dataset_view = dataset

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    base_dir = Path(args.out_dir) / args.dataset
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = base_dir / "summary_image_ablation.csv"
    report_txt = base_dir / "report_image_ablation.txt"
    summary_rows = []

    for local_idx in tqdm(range(start, end), desc="Samples"):
        rec = prompt_records[local_idx]
        item = dataset_view[local_idx]

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
                run_dir = base_dir / image_mode / Path(image_name).stem
                run_dir.mkdir(parents=True, exist_ok=True)

                # 关键修正：给 patched attention 一个可写路径
                os.environ["SAVE_ATTN_PATH"] = str(run_dir / "saved_attn.pt")

                use_image = image if image_mode == "original" else make_black_image_like(image)

                output, gen_text, prompt_len = run_one_generation(
                    wrapper=wrapper,
                    model_name=args.model_name,
                    image=use_image,
                    prompt=prompt,
                    perturb_mode="none",
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
                    image_mode,
                    os.path.splitext(image_name)[0],
                    "q0_trace.json",
                )
                trace_path = base_dir / trace_rel_path

                write_json(trace_path, {
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "image_mode": image_mode,
                    "question": question_clean,
                    "gold": gold,
                    "generated_text": gen_text,
                    "pred": pred,
                    "correct": correct,
                    "token_trace": trace,
                })

                first = trace[0] if trace else {}
                last = trace[-1] if trace else {}

                row.update({
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
