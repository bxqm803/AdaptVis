import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset

from model_zoo.llava15 import LlavaWrapper


def build_prompt(caption: str) -> str:
    return (
        "USER: <image>\n"
        "Determine whether the following caption correctly describes the image. "
        "Answer only True or False.\n\n"
        f"Caption: {caption}\n"
        "ASSISTANT:"
    )


def normalize_pred(text: str) -> int:
    t = str(text).strip().lower()

    if "true" in t and "false" not in t:
        return 1
    if "false" in t and "true" not in t:
        return 0

    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0

    return -1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="random", choices=["random", "zeroshot"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])

    parser.add_argument("--cache-dir", default="/ddnB/work/mwang32/hf_cache")
    parser.add_argument("--method", default="default", choices=["default", "scaling_vis", "adapt_vis"])

    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--weight1", type=float, default=1.0)
    parser.add_argument("--weight2", type=float, default=1.0)

    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--out", default="vsr_llava_results.csv")

    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Loading VSR dataset: config={args.config}, split={args.split}")
    ds = load_dataset(
        "juletxara/visual-spatial-reasoning",
        args.config,
        split=args.split,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    if args.max_samples > 0:
        n = min(args.max_samples, len(ds))
        ds = ds.select(range(n))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LLaVA on {device}")

    model = LlavaWrapper(
        root_dir=args.cache_dir,
        device=device,
        method=args.method,
    )

    rows = []
    correct = 0
    valid_pred = 0

    for idx, ex in enumerate(tqdm(ds)):
        image = ex["image"].convert("RGB")
        caption = ex["caption"]
        label = int(ex["label"])

        prompt = build_prompt(caption)

        pred_text = model.run_single_prompt(
            image=image,
            prompt=prompt,
            method=args.method,
            weight=args.weight,
            threshold=args.threshold,
            weight1=args.weight1,
            weight2=args.weight2,
        )

        pred = normalize_pred(pred_text)

        if pred != -1:
            valid_pred += 1

        is_correct = int(pred == label)
        correct += is_correct

        rows.append({
            "idx": idx,
            "caption": caption,
            "relation": ex.get("relation", ""),
            "label": label,
            "label_text": "True" if label == 1 else "False",
            "pred": pred,
            "pred_text": pred_text,
            "correct": is_correct,
            "image_link": ex.get("image_link", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    total = len(df)
    acc = correct / total if total > 0 else 0.0
    valid_rate = valid_pred / total if total > 0 else 0.0

    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Valid prediction rate: {valid_rate:.4f}")
    print(f"Saved to: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
