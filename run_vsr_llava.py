import argparse
import os
import json
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from model_zoo.llava15 import LlavaWrapper


SPLIT_URL = "https://github.com/cambridgeltl/visual-spatial-reasoning/raw/master/data/splits"


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


def split_filename(split: str) -> str:
    if split == "validation":
        return "dev.jsonl"
    return f"{split}.jsonl"


def download_jsonl(config: str, split: str, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)

    fname = split_filename(split)
    local_path = os.path.join(cache_dir, f"vsr_{config}_{fname}")

    if os.path.exists(local_path):
        return local_path

    url = f"{SPLIT_URL}/{config}/{fname}"
    print(f"Downloading split file: {url}")

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(r.content)

    return local_path


def load_vsr_jsonl(config: str, split: str, cache_dir: str):
    path = download_jsonl(config, split, cache_dir)

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def load_image_from_url(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


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

    print(f"Loading VSR jsonl: config={args.config}, split={args.split}")
    data = load_vsr_jsonl(args.config, args.split, args.cache_dir)

    if args.max_samples > 0:
        data = data[:args.max_samples]

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

    for idx, ex in enumerate(tqdm(data)):
        try:
            image = load_image_from_url(ex["image_link"])
        except Exception as e:
            print(f"[WARN] Failed to load image idx={idx}: {e}")
            continue

        caption = ex["caption"]
        label = int(ex["label"])
        relation = ex.get("relation", "")

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
            "relation": relation,
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
    print(f"Total evaluated samples: {total}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Valid prediction rate: {valid_rate:.4f}")
    print(f"Saved to: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
