import os
import json
import math
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import shutil
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
    parser.add_argument(
        "--limit",
        default=-1,
        type=int,
        help="-1 means run all remaining samples"
    )
    parser.add_argument("--attn-layer", default=17, type=int)
    parser.add_argument("--out-dir", default="./output_multiq", type=str)
    return parser.parse_args()

from PIL import Image

def get_shape_only_vis_image(model, raw_image_path):
    img = Image.open(raw_image_path).convert("RGB")

    image_processor = getattr(model.processor, "image_processor", None)
    if image_processor is None:
        image_processor = getattr(model, "feature_extractor", None)
    if image_processor is None:
        return np.array(img).astype(np.float32) / 255.0

    # 1) resize like processor
    if getattr(image_processor, "do_resize", False):
        size = image_processor.size
        if isinstance(size, dict):
            short_edge = size.get("shortest_edge", None)
            height = size.get("height", None)
            width = size.get("width", None)

            if short_edge is not None:
                w, h = img.size
                if w < h:
                    new_w = short_edge
                    new_h = int(round(h * short_edge / w))
                else:
                    new_h = short_edge
                    new_w = int(round(w * short_edge / h))
                img = img.resize((new_w, new_h), Image.BICUBIC)
            elif height is not None and width is not None:
                img = img.resize((width, height), Image.BICUBIC)

    # 2) center crop like processor
    if getattr(image_processor, "do_center_crop", False):
        crop_size = image_processor.crop_size
        if isinstance(crop_size, dict):
            crop_h = crop_size.get("height", None)
            crop_w = crop_size.get("width", None)
            if crop_h is not None and crop_w is not None:
                w, h = img.size
                left = max((w - crop_w) // 2, 0)
                top = max((h - crop_h) // 2, 0)
                img = img.crop((left, top, left + crop_w, top + crop_h))

    return np.array(img).astype(np.float32) / 255.0


def save_attn_overlay_shapeonly(attn_npy_path, out_png_path, base_img_np, alpha=0.45):
    arr = np.load(attn_npy_path).astype(np.float32).reshape(-1)

    side = int(round(np.sqrt(len(arr))))
    if side * side != len(arr):
        raise ValueError(f"Attention length {len(arr)} is not a square number")

    heat = arr.reshape(side, side)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    h, w = base_img_np.shape[:2]
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
    heat_np = np.array(heat_img).astype(np.float32) / 255.0

    plt.figure(figsize=(6, 6))
    plt.imshow(base_img_np)
    plt.imshow(heat_np, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0, dpi=180)
    plt.close()
    
def normalize_tf_label(x):
    if x is None:
        return "UNK"
    x = str(x).strip().lower()
    if x in {"t", "true", "yes"}:
        return "True"
    if x in {"f", "false", "no"}:
        return "False"
    return "UNK"
    
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
    print("DEBUG first raw prompt:")
    print(repr(prompt_records[0]["question"]))
    print("DEBUG first raw answer:")
    print(repr(prompt_records[0]["answer"]))
    object_pool = build_object_pool(prompt_records)

    TEST = os.getenv('TEST_MODE', 'False') == 'True'

    if sampled_indices is not None:
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        sub_dataset = dataset

    if args.limit < 0:
        end_idx = len(prompt_records)
    else:
        end_idx = min(args.sample_index + args.limit, len(prompt_records))

    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows = []

    for local_idx in tqdm(range(args.sample_index, end_idx), desc="Samples"):
        rec = prompt_records[local_idx]
        item = sub_dataset[local_idx]
        image = item["image_options"][0]

        questions, meta = build_questions(
            base_prompt=rec["question"],
            base_answer=rec["answer"][0] if isinstance(rec["answer"], list) else rec["answer"],
            sample_idx=local_idx,
            object_pool=object_pool,
        )

        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")
        image_stem = os.path.splitext(image_name)[0]   
        sample_dir = os.path.join(args.out_dir, args.dataset, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        if image_path and os.path.exists(image_path):
            raw_img_name = os.path.basename(image_path)
            dst_img_path = os.path.join(sample_dir, raw_img_name)
            if not os.path.exists(dst_img_path):
                shutil.copy2(image_path, dst_img_path)
                
        meta_out = {
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            "test_mode": TEST,
            **meta,
            "questions": []
        }

        q_correct_map = {}

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

            if q["mode"] == "tf":
                correct = (normalize_tf_label(pred) == normalize_tf_label(q["gold"]))
            else:
                correct = (pred == q["gold"])
            q_correct_map[qid] = correct

            attn_npy = os.path.join(qdir, f"attn_map_layer{args.attn_layer}.npy")
            attn_png = os.path.join(sample_dir, f"{qid}_attn.png")

            if os.path.exists(attn_npy):
                base_img_np = get_shape_only_vis_image(model, image_path)
                save_attn_overlay_shapeonly(attn_npy, attn_png, base_img_np)
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

        pattern_q1_q9 = "_".join(
            "C" if q_correct_map.get(f"q{i}", False) else "W"
            for i in range(1, 10)
        )

        summary_rows.append({
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "q1": "C" if q_correct_map.get("q1", False) else "W",
            "q2": "C" if q_correct_map.get("q2", False) else "W",
            "q3": "C" if q_correct_map.get("q3", False) else "W",
            "q4": "C" if q_correct_map.get("q4", False) else "W",
            "q5": "C" if q_correct_map.get("q5", False) else "W",
            "q6": "C" if q_correct_map.get("q6", False) else "W",
            "q7": "C" if q_correct_map.get("q7", False) else "W",
            "q8": "C" if q_correct_map.get("q8", False) else "W",
            "q9": "C" if q_correct_map.get("q9", False) else "W",
            "pattern_q1_q9": pattern_q1_q9,
            "num_correct_q1_q9": sum(q_correct_map.get(f"q{i}", False) for i in range(1, 10)),
        })

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

    summary_csv = os.path.join(args.out_dir, args.dataset, "summary.csv")
    os.makedirs(os.path.join(args.out_dir, args.dataset), exist_ok=True)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "image_path",
                "local_index",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "q9",
                "pattern_q1_q9",
                "num_correct_q1_q9",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")

if __name__ == "__main__":
    main()
