import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from model_zoo.llava15 import LLaVA15Model


def normalize_pred(text):
    t = text.lower()
    if "true" in t and "false" not in t:
        return 1
    if "false" in t and "true" not in t:
        return 0
    if t.strip().startswith("yes"):
        return 1
    if t.strip().startswith("no"):
        return 0
    return -1


def build_prompt(caption):
    return (
        "Determine whether the following caption correctly describes the image. "
        "Answer only True or False.\n\n"
        f"Caption: {caption}\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="random", choices=["random", "zeroshot"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--out", default="vsr_llava_results.csv")
    args = parser.parse_args()

    ds = load_dataset(
        "juletxara/visual-spatial-reasoning",
        args.config,
        split=args.split,
        trust_remote_code=True,
    )

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    model = LLaVA15Model(args.model_path)

    rows = []
    correct = 0

    for i, ex in enumerate(tqdm(ds)):
        image = ex["image"].convert("RGB")
        caption = ex["caption"]
        label = int(ex["label"])

        prompt = build_prompt(caption)

        pred_text = model.run_single_prompt(
            image=image,
            prompt=prompt,
            method="default",
            weight=1.0,
            threshold=0.0,
            weight1=1.0,
            weight2=1.0,
        )

        pred = normalize_pred(pred_text)
        is_correct = int(pred == label)
        correct += is_correct

        rows.append({
            "idx": i,
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

    acc = correct / len(ds)
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
