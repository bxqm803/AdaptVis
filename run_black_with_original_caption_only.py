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

TARGET_VARIANT = {
    "variant": "black_with_original_caption",
    "image_mode": "black",
    "use_caption": True,
    "caption_source_mode": "original",
}


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
        "--perturb-modes",
        default="none,uniform,random,reverse",
        type=str,
        help="Comma-separated list from {none,uniform,random,reverse}.",
    )
    p.add_argument(
        "--target-layers",
        default="all",
        type=str,
        help='"all" or comma-separated decoder layer indices.',
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
        default="./output_interrupt_q0_black_with_original_caption_only",
        type=str,
    )

    p.add_argument(
        "--caption-prompt",
        default="Describe the image in one concise sentence.",
        type=str,
    )
    p.add_argument(
        "--caption-max-new-tokens",
        default=40,
        type=int,
    )
    p.add_argument(
        "--caption-perturb-mode",
        default="none",
        type=str,
        choices=["none", "uniform", "random", "reverse"],
    )
    p.add_argument(
        "--reuse-caption-from-dir",
        default="",
        type=str,
        help="Optional previous output root dir. If the original-image caption trace exists there, reuse it.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a sample if all 24 x len(perturb_modes) trace files already exist.",
    )
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
        r"Answer with\s+left,\s*right,\s*on\s+or\s+under(?:\s+only)?\.\s*$",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q


def clean_caption_text(text: str):
    x = str(text).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_token_text(token_text: str):
    if token_text is None:
        return ""
    x = str(token_text)
    x = x.replace("Ġ", "")
    x = x.replace("▁", "")
    x = x.strip().lower()
    return x


def extract_first_topk_candidates(trace, topn=10):
    if not trace:
        return []

    first = trace[0]
    candidate_keys = ["topk", "top_k", "top_tokens", "top_candidates", "candidates"]
    candidates = None

    for k in candidate_keys:
        if k in first and isinstance(first[k], list):
            candidates = first[k]
            break

    if candidates is None:
        return []

    out = []
    for cand in candidates[:topn]:
        if not isinstance(cand, dict):
            continue

        token_text = (
            cand.get("token_text")
            if cand.get("token_text") is not None
            else cand.get("token", cand.get("text", cand.get("decoded", "")))
        )

        raw_logit = cand.get("raw_logit", cand.get("logit", None))
        probability = cand.get("probability", None)

        out.append({
            "token_text": token_text,
            "token_norm": normalize_token_text(token_text),
            "raw_logit": raw_logit,
            "probability": probability,
        })

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


def build_q0_permuted_prompts(base_prompt, caption_text):
    stem = strip_answer_order_clause(base_prompt)
    cap = clean_caption_text(caption_text)

    out = []
    for perm in itertools.permutations(CHOICES):
        order_text = ", ".join(perm[:-1]) + f" or {perm[-1]}"
        prompt_text = (
            f"Caption: {cap}\n"
            f"Question: {stem} Answer with {order_text} only."
        )
        prompt = f"<image>\nUSER: {prompt_text}\nASSISTANT:"
        perm_id = "_".join(perm)

        out.append({
            "qid": "q0",
            "perm_id": perm_id,
            "order": list(perm),
            "caption_text": cap,
            "prompt_text": prompt_text,
            "prompt": prompt,
        })
    return out


def build_summary_fieldnames(save_first_topk=10):
    fieldnames = [
        "variant",
        "use_caption",
        "caption_source_mode",
        "image_mode",
        "image_name",
        "image_path",
        "local_index",
        "qid",
        "perm_id",
        "order_text",
        "base_question",
        "caption_text",
        "caption_prompt",
        "caption_perturb_mode",
        "caption_trace_json",
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
        fieldnames.extend([
            f"first_top{i}_token",
            f"first_top{i}_token_norm",
            f"first_top{i}_raw_logit",
            f"first_top{i}_probability",
        ])

    fieldnames.extend([
        "final_token",
        "final_raw_logit",
        "final_score",
        "final_probability",
        "trace_json",
    ])
    return fieldnames


def generate_caption_for_image(
    wrapper,
    processor,
    model_name,
    image,
    image_mode_name,
    args,
    image_name,
    image_path,
    image_stem,
    local_idx,
    base_dir,
    trace_topk_to_use,
):
    caption_output, caption_text_raw, caption_prompt_len = run_one_generation(
        wrapper=wrapper,
        model_name=model_name,
        image=image,
        prompt=f"<image>\nUSER: {args.caption_prompt}\nASSISTANT:",
        perturb_mode=args.caption_perturb_mode,
        max_new_tokens=args.caption_max_new_tokens,
    )

    caption_text = clean_caption_text(caption_text_raw)
    caption_trace = build_generation_trace(
        processor,
        caption_output,
        caption_prompt_len,
        topk=trace_topk_to_use,
    )

    caption_trace_rel_path = os.path.join(
        image_stem,
        f"caption_source_{image_mode_name}_{args.caption_perturb_mode}_trace.json",
    )
    caption_trace_path = os.path.join(base_dir, caption_trace_rel_path)

    write_json(caption_trace_path, {
        "image_name": image_name,
        "image_path": image_path,
        "local_index": local_idx,
        "caption_source_mode": image_mode_name,
        "caption_prompt": args.caption_prompt,
        "caption_perturb_mode": args.caption_perturb_mode,
        "caption_text": caption_text,
        "token_trace": caption_trace,
    })

    return {
        "caption_text": caption_text,
        "caption_trace_rel_path": caption_trace_rel_path,
    }


def maybe_reuse_original_caption(base_dir, reuse_base_dir, dataset_name, image_stem, caption_perturb_mode):
    if not reuse_base_dir:
        return None

    rel_path = os.path.join(
        dataset_name,
        image_stem,
        f"caption_source_original_{caption_perturb_mode}_trace.json",
    )
    old_path = os.path.join(reuse_base_dir, rel_path)
    if not os.path.exists(old_path):
        return None

    with open(old_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    caption_text = clean_caption_text(payload.get("caption_text", ""))
    if not caption_text:
        return None

    new_rel_path = os.path.join(
        image_stem,
        f"caption_source_original_{caption_perturb_mode}_trace.json",
    )
    new_path = os.path.join(base_dir, new_rel_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    if not os.path.exists(new_path):
        write_json(new_path, payload)

    return {
        "caption_text": caption_text,
        "caption_trace_rel_path": new_rel_path,
    }


def expected_trace_rel_paths(image_stem, perturb_modes):
    rel_paths = []
    for perm in itertools.permutations(CHOICES):
        perm_id = "_".join(perm)
        for perturb_mode in perturb_modes:
            rel_paths.append(
                os.path.join(
                    image_stem,
                    f"black_with_original_caption_q0_{perm_id}_{perturb_mode}_trace.json",
                )
            )
    return rel_paths


def load_existing_trace_rows(summary_csv):
    if not os.path.exists(summary_csv):
        return set(), []

    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    seen = set()
    for row in rows:
        seen.add((
            row.get("image_name", ""),
            row.get("perm_id", ""),
            row.get("perturb_mode", ""),
        ))
    return seen, rows


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

    base_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(base_dir, exist_ok=True)

    summary_csv = os.path.join(base_dir, "summary_black_with_original_caption.csv")
    summary_fieldnames = build_summary_fieldnames(save_first_topk=args.save_first_topk)
    seen_rows, summary_rows = load_existing_trace_rows(summary_csv)

    trace_topk_to_use = max(args.trace_topk, args.save_first_topk)

    for local_idx in tqdm(range(start, end), desc="Samples"):
        rec = prompt_records[local_idx]
        item = sub_dataset[local_idx]

        original_image = item["image_options"][0]
        black_image = make_black_image_like(original_image)

        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")
        image_stem = os.path.splitext(image_name)[0]

        sample_dir = os.path.join(base_dir, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        if args.skip_existing:
            rel_paths = expected_trace_rel_paths(image_stem, perturb_modes)
            if all(os.path.exists(os.path.join(base_dir, p)) for p in rel_paths):
                continue

        gold = normalize_rel(rec["answer"])
        base_question = clean_question_text(rec["question"])

        original_caption_info = maybe_reuse_original_caption(
            base_dir=base_dir,
            reuse_base_dir=args.reuse_caption_from_dir,
            dataset_name=args.dataset,
            image_stem=image_stem,
            caption_perturb_mode=args.caption_perturb_mode,
        )

        if original_caption_info is None:
            original_caption_info = generate_caption_for_image(
                wrapper=wrapper,
                processor=wrapper.processor,
                model_name=args.model_name,
                image=original_image,
                image_mode_name="original",
                args=args,
                image_name=image_name,
                image_path=image_path,
                image_stem=image_stem,
                local_idx=local_idx,
                base_dir=base_dir,
                trace_topk_to_use=trace_topk_to_use,
            )

        caption_text = original_caption_info["caption_text"]
        caption_trace_rel_path = original_caption_info["caption_trace_rel_path"]
        permuted_prompts = build_q0_permuted_prompts(rec["question"], caption_text=caption_text)

        variant_meta = {
            "variant": TARGET_VARIANT["variant"],
            "image_mode": TARGET_VARIANT["image_mode"],
            "use_caption": TARGET_VARIANT["use_caption"],
            "caption_source_mode": TARGET_VARIANT["caption_source_mode"],
            "caption_text": caption_text,
            "caption_trace_json": caption_trace_rel_path,
            "num_permutations": len(permuted_prompts),
        }

        for q in permuted_prompts:
            for perturb_mode in perturb_modes:
                row_key = (image_name, q["perm_id"], perturb_mode)
                trace_rel_path = os.path.join(
                    image_stem,
                    f"black_with_original_caption_q0_{q['perm_id']}_{perturb_mode}_trace.json",
                )
                trace_path = os.path.join(base_dir, trace_rel_path)

                if row_key in seen_rows and os.path.exists(trace_path):
                    continue

                output, gen_text, prompt_len = run_one_generation(
                    wrapper=wrapper,
                    model_name=args.model_name,
                    image=black_image,
                    prompt=q["prompt"],
                    perturb_mode=perturb_mode,
                    max_new_tokens=args.max_new_tokens,
                )

                trace = build_generation_trace(
                    wrapper.processor,
                    output,
                    prompt_len,
                    topk=trace_topk_to_use,
                )

                pred = parse_prediction(gen_text)
                correct = (pred == gold)

                first_topk_fields = build_first_topk_summary_fields(
                    trace,
                    topn=args.save_first_topk,
                )

                write_json(trace_path, {
                    "variant": TARGET_VARIANT["variant"],
                    "image_mode": TARGET_VARIANT["image_mode"],
                    "use_caption": TARGET_VARIANT["use_caption"],
                    "caption_source_mode": TARGET_VARIANT["caption_source_mode"],
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "qid": "q0",
                    "perm_id": q["perm_id"],
                    "order": q["order"],
                    "base_question": base_question,
                    "caption_text": caption_text,
                    "prompt_text": q["prompt_text"],
                    "gold": gold,
                    "prompt": q["prompt"],
                    "caption_prompt": args.caption_prompt,
                    "caption_perturb_mode": args.caption_perturb_mode,
                    "perturb_mode": perturb_mode,
                    "target_layers": "all" if target_layers is None else sorted(target_layers),
                    "generated_text": gen_text,
                    "pred": pred,
                    "correct": correct,
                    "first_topk_summary": json.loads(first_topk_fields[f"first_top{args.save_first_topk}_json"]),
                    "token_trace": trace,
                })

                first = trace[0] if trace else {}
                last = trace[-1] if trace else {}

                row = {
                    "variant": TARGET_VARIANT["variant"],
                    "use_caption": True,
                    "caption_source_mode": "original",
                    "image_mode": "black",
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "qid": "q0",
                    "perm_id": q["perm_id"],
                    "order_text": " | ".join(q["order"]),
                    "base_question": base_question,
                    "caption_text": caption_text,
                    "caption_prompt": args.caption_prompt,
                    "caption_perturb_mode": args.caption_perturb_mode,
                    "caption_trace_json": caption_trace_rel_path,
                    "prompt_text": q["prompt_text"],
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
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
                }
                row.update(first_topk_fields)
                summary_rows.append(row)
                seen_rows.add(row_key)

        meta_path = os.path.join(sample_dir, "meta_black_with_original_caption.json")
        if not os.path.exists(meta_path):
            write_json(meta_path, {
                "local_index": local_idx,
                "image_name": image_name,
                "image_path": image_path,
                "base_question": base_question,
                "gold": gold,
                "caption_prompt": args.caption_prompt,
                "caption_perturb_mode": args.caption_perturb_mode,
                "variant": variant_meta,
            })

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
