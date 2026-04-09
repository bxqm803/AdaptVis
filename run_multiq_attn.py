import os
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all
from multiq_utils import build_object_pool, build_questions, parse_prediction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--model-name", default="llava1.5", type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--method", default="scaling_vis", type=str)
    parser.add_argument("--weight", default=1.0, type=float)
    parser.add_argument("--threshold", default=1.0, type=float)
    parser.add_argument("--weight1", default=1.0, type=float)
    parser.add_argument("--weight2", default=1.0, type=float)
    parser.add_argument("--option", default="four", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--sample-index", default=0, type=int)
    parser.add_argument("--limit", default=1, type=int)
    parser.add_argument("--attn-layer", default=17, type=int)
    parser.add_argument("--out-dir", default="./output_multiq", type=str)
    return parser.parse_args()

def save_attn_png(npy_path, png_path):
    arr = np.load(npy_path)
    n = len(arr)
    side = int(round(math.sqrt(n)))
    if side * side == n:
        grid = arr.reshape(side, side)
        plt.figure(figsize=(5, 5))
        plt.imshow(grid)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(10, 3))
        plt.plot(arr)
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()

def main():
    args = parse_args()
    seed_all(args.seed)

    model, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    prompt_records, sampled_indices = model.load_prompt_records_with_sampling(args.dataset, args.option)
    object_pool = build_object_pool(prompt_records)

    TEST = os.getenv('TEST_MODE', 'False') == 'True'

    if sampled_indices is not None:
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        sub_dataset = dataset

    end_idx = min(args.sample_index + args.limit, len(prompt_records))

    os.makedirs(args.out_dir, exist_ok=True)

    for local_idx in range(args.sample_index, end_idx):
        rec = prompt_records[local_idx]
        item = sub_dataset[local_idx]
        image = item["image_options"][0]

        questions, meta = build_questions(
            base_question=rec["question"],
            base_answer=rec["answer"][0] if isinstance(rec["answer"], list) else rec["answer"],
            sample_idx=local_idx,
            object_pool=object_pool,
        )

        sample_dir = os.path.join(args.out_dir, args.dataset, f"sample_{local_idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        meta_out = {
            "local_index": local_idx,
            "test_mode": TEST,
            **meta,
            "questions": []
        }

        for q in questions:
            qid = q["qid"]
            qdir = os.path.join(sample_dir, qid)
            os.makedirs(qdir, exist_ok=True)

            os.environ["SAVE_ATTN_PATH"] = qdir + "/"
            os.environ["SAVE_ATTN_LAYER"] = str(args.attn_layer)

            pred_text = model.run_single_prompt(
                image=image,
                prompt=q["prompt"],
                method=args.method,
                weight=args.weight,
                threshold=args.threshold,
                weight1=args.weight1,
                weight2=args.weight2,
            )

            pred = parse_prediction(pred_text, q["mode"])
            correct = (pred == q["gold"])

            attn_npy = os.path.join(qdir, f"attn_map_layer{args.attn_layer}.npy")
            attn_png = os.path.join(sample_dir, f"{qid}_attn.png")

            if os.path.exists(attn_npy):
                save_attn_png(attn_npy, attn_png)
                os.remove(attn_npy)

            meta_out["questions"].append({
                "qid": qid,
                "mode": q["mode"],
                "prompt": q["prompt"],
                "gold": q["gold"],
                "pred_text": pred_text,
                "pred": pred,
                "correct": correct,
                "attn_png": f"{qid}_attn.png",
            })

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()