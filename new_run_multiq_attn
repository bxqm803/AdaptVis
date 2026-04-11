import os
import csv
import json
import math
import shutil
import argparse
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all
from multiq_utils import build_object_pool, build_questions, parse_prediction


REL_CORE = {
    "to the left of": "left",
    "on the left of": "left",
    "left": "left",
    "to the right of": "right",
    "on the right of": "right",
    "right": "right",
    "on top of": "top",
    "on": "on",
    "under": "under",
    "beneath": "beneath",
    None: None,
}


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
    parser.add_argument("--limit", default=-1, type=int, help="-1 means run all remaining samples")
    parser.add_argument("--attn-layer", default=17, type=int, help="self-attention layer saved by model internals")
    parser.add_argument(
        "--prompt-token-layer",
        default=16,
        type=int,
        help="single transformer layer used for obj1/obj2/rel prompt-token attention overlays",
    )
    parser.add_argument("--out-dir", default="./output_multiq", type=str)
    parser.add_argument("--targets", default="obj1,obj2,rel", type=str)
    parser.add_argument("--alpha", default=0.45, type=float)
    return parser.parse_args()


TARGET_NAMES = ("obj1", "obj2", "rel")


def normalize_tf_label(x):
    if x is None:
        return "UNK"
    x = str(x).strip().lower()
    if x in {"t", "true", "yes"}:
        return "True"
    if x in {"f", "false", "no"}:
        return "False"
    return "UNK"



def get_shape_only_vis_image(model, raw_image_path: str):
    img = Image.open(raw_image_path).convert("RGB")

    image_processor = getattr(model.processor, "image_processor", None)
    if image_processor is None:
        image_processor = getattr(model, "feature_extractor", None)
    if image_processor is None:
        return np.array(img).astype(np.float32) / 255.0

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



def save_attn_overlay_shapeonly(attn_npy_path: str, out_png_path: str, base_img_np: np.ndarray, alpha: float = 0.45):
    arr = np.load(attn_npy_path).astype(np.float32).reshape(-1)
    save_vec_overlay(arr, out_png_path, base_img_np, alpha=alpha)



def save_vec_overlay(vec: np.ndarray, out_png_path: str, base_img_np: np.ndarray, alpha: float = 0.45, title: Optional[str] = None):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    side = int(round(np.sqrt(len(vec))))
    if side * side != len(vec):
        raise ValueError(f"Attention length {len(vec)} is not a square number")

    heat = vec.reshape(side, side)
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
    if title:
        plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0, dpi=180)
    plt.close()



def normalize_prompt_token_targets(target_texts: Optional[Dict[str, Optional[str]]]) -> Optional[Dict[str, Optional[str]]]:
    if target_texts is None:
        return None

    rel_text = target_texts.get("rel")
    rel_core = REL_CORE.get(rel_text, rel_text)
    return {
        "obj1": target_texts.get("obj1"),
        "obj2": target_texts.get("obj2"),
        "rel": rel_core,
    }



def save_prompt_token_attn_maps(
    prompt_token_attn: Dict,
    base_img_np: np.ndarray,
    sample_dir: str,
    qid: str,
    alpha: float,
    keep_targets: List[str],
) -> Dict[str, str]:
    saved = {}
    layer_list = prompt_token_attn.get("meta", {}).get("layers", [])
    maps = prompt_token_attn.get("maps", {})

    for target_name in keep_targets:
        target_maps = maps.get(target_name, {})
        for layer_idx in layer_list:
            key = str(layer_idx)
            if key not in target_maps:
                continue
            vec = np.array(target_maps[key], dtype=np.float32)
            out_name = f"{qid}_{target_name}_L{layer_idx}.png"
            out_path = os.path.join(sample_dir, out_name)
            save_vec_overlay(vec, out_path, base_img_np, alpha=alpha, title=f"{qid} | {target_name} | L{layer_idx}")
            saved[f"{target_name}_L{layer_idx}"] = out_name
    return saved



def make_summary_fieldnames(qids: List[str], layer_idx: int, targets: List[str]) -> List[str]:
    fields = [
        "image_name",
        "image_path",
        "local_index",
        "pattern_q1_q9",
        "num_correct_q1_q9",
    ]
    fields.extend(qids)
    fields.extend([f"{qid}_trace_json" for qid in qids])
    fields.extend([f"{qid}_prompt_token_attn_json" for qid in qids])
    fields.extend([f"{qid}_final_prob" for qid in qids])
    for qid in qids:
        for target in targets:
            fields.append(f"{qid}_{target}_L{layer_idx}_png")
    return fields



def main():
    args = parse_args()
    seed_all(args.seed)

    keep_targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    keep_targets = [x for x in keep_targets if x in TARGET_NAMES]
    if not keep_targets:
        raise ValueError("--targets must contain at least one of: obj1,obj2,rel")

    model, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    prompt_records, sampled_indices = model.load_prompt_records_with_sampling(args.dataset, args.option)
    object_pool = build_object_pool(prompt_records)

    if sampled_indices is not None:
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        sub_dataset = dataset

    start_idx = args.sample_index
    if args.limit < 0:
        end_idx = len(prompt_records)
    else:
        end_idx = min(start_idx + args.limit, len(prompt_records))

    dataset_out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)
    summary_csv = os.path.join(dataset_out_dir, "summary.csv")

    qids = [f"q{i}" for i in range(10)]
    fieldnames = make_summary_fieldnames(qids, args.prompt_token_layer, keep_targets)
    summary_rows = []

    def write_summary_csv():
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    iterator = tqdm(range(start_idx, end_idx), desc=f"{args.dataset}:{args.model_name}")
    for local_idx in iterator:
        sample = sub_dataset[local_idx]
        if isinstance(sample, dict):
            image = sample.get("image", sample.get("images", None))
            image_path = sample.get("image_path", sample.get("img_path", None))
            image_name = sample.get("image_name", os.path.basename(image_path) if image_path else f"sample_{local_idx}")
        else:
            image = sample[0]
            image_path = None
            image_name = f"sample_{local_idx}"

        record = prompt_records[local_idx]
        questions, meta = build_questions(record["question"], record["answer"], local_idx, object_pool)

        image_stem = os.path.splitext(os.path.basename(image_name))[0]
        sample_dir = os.path.join(dataset_out_dir, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        if image_path and os.path.exists(image_path):
            raw_img_name = os.path.basename(image_path)
            dst_img_path = os.path.join(sample_dir, raw_img_name)
            if not os.path.exists(dst_img_path):
                shutil.copy2(image_path, dst_img_path)

        base_img_np = None
        if image_path and os.path.exists(image_path):
            base_img_np = get_shape_only_vis_image(model, image_path)

        meta_out = {
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            **meta,
            "prompt_token_layer": args.prompt_token_layer,
            "targets": keep_targets,
            "questions": [],
        }

        q_correct_map = {}
        q_trace_summary = {}
        q_png_summary = {}

        for q in questions:
            qid = q["qid"]
            qdir = os.path.join(sample_dir, qid)
            os.makedirs(qdir, exist_ok=True)

            os.environ["SAVE_ATTN_PATH"] = qdir + "/"
            os.environ["SAVE_ATTN_LAYER"] = str(args.attn_layer)

            prompt_token_targets = normalize_prompt_token_targets(q.get("target_texts", None))

            pred_text, token_trace, prompt_token_attn = model.run_single_prompt(
                image=image,
                prompt=q["prompt"],
                method=args.method,
                weight=args.weight,
                threshold=args.threshold,
                weight1=args.weight1,
                weight2=args.weight2,
                return_trace=True,
                trace_topk=10,
                return_prompt_token_attn=True,
                prompt_token_targets=prompt_token_targets,
                prompt_token_layers=(args.prompt_token_layer,),
            )

            pred = parse_prediction(pred_text, q["mode"])
            if q["mode"] == "tf":
                correct = normalize_tf_label(pred) == normalize_tf_label(q["gold"])
            else:
                correct = pred == q["gold"]
            q_correct_map[qid] = bool(correct)

            trace_json_name = f"{qid}_token_trace.json"
            trace_json_path = os.path.join(sample_dir, trace_json_name)
            with open(trace_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "qid": qid,
                        "mode": q["mode"],
                        "prompt": q["prompt"],
                        "gold": q["gold"],
                        "pred_text": pred_text,
                        "pred": pred,
                        "correct": correct,
                        "token_trace": token_trace,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            prompt_token_attn_json_name = None
            prompt_token_pngs = {}
            if prompt_token_attn is not None:
                prompt_token_attn_json_name = f"{qid}_prompt_token_attn.json"
                prompt_token_attn_json_path = os.path.join(sample_dir, prompt_token_attn_json_name)
                with open(prompt_token_attn_json_path, "w", encoding="utf-8") as f:
                    json.dump(prompt_token_attn, f, indent=2, ensure_ascii=False)

                if base_img_np is not None:
                    prompt_token_pngs = save_prompt_token_attn_maps(
                        prompt_token_attn=prompt_token_attn,
                        base_img_np=base_img_np,
                        sample_dir=sample_dir,
                        qid=qid,
                        alpha=args.alpha,
                        keep_targets=keep_targets,
                    )

            attn_png_name = f"{qid}_attn.png"
            attn_png_path = os.path.join(sample_dir, attn_png_name)
            attn_npy = os.path.join(qdir, f"attn_map_layer{args.attn_layer}.npy")
            if os.path.exists(attn_npy) and base_img_np is not None:
                save_attn_overlay_shapeonly(attn_npy, attn_png_path, base_img_np, alpha=args.alpha)
                os.remove(attn_npy)
            else:
                attn_png_name = None

            q_trace_summary[qid] = {
                "trace_json": trace_json_name,
                "prompt_token_attn_json": prompt_token_attn_json_name,
                "num_tokens": len(token_trace),
                "first_token": token_trace[0]["token_text"] if token_trace else "",
                "first_prob": token_trace[0]["chosen_prob"] if token_trace else None,
                "final_token": token_trace[-1]["token_text"] if token_trace else "",
                "final_prob": token_trace[-1]["chosen_prob"] if token_trace else None,
            }
            q_png_summary[qid] = prompt_token_pngs

            meta_out["questions"].append(
                {
                    "qid": qid,
                    "mode": q["mode"],
                    "prompt": q["prompt"],
                    "gold": q["gold"],
                    "pred_text": pred_text,
                    "pred": pred,
                    "correct": correct,
                    "attn_png": attn_png_name,
                    "token_trace_json": trace_json_name,
                    "prompt_token_targets": prompt_token_targets,
                    "prompt_token_attn_json": prompt_token_attn_json_name,
                    "prompt_token_attn_pngs": prompt_token_pngs,
                    "num_generated_tokens": len(token_trace),
                    "first_token": token_trace[0]["token_text"] if token_trace else "",
                    "first_token_prob": token_trace[0]["chosen_prob"] if token_trace else None,
                    "final_token": token_trace[-1]["token_text"] if token_trace else "",
                    "final_token_prob": token_trace[-1]["chosen_prob"] if token_trace else None,
                }
            )

            with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta_out, f, indent=2, ensure_ascii=False)

        pattern_q1_q9 = "_".join("C" if q_correct_map.get(f"q{i}", False) else "W" for i in range(1, 10))
        row = {
            "image_name": image_name,
            "image_path": image_path,
            "local_index": local_idx,
            "pattern_q1_q9": pattern_q1_q9,
            "num_correct_q1_q9": sum(q_correct_map.get(f"q{i}", False) for i in range(1, 10)),
        }

        for qid in qids:
            row[qid] = "C" if q_correct_map.get(qid, False) else "W"
            row[f"{qid}_trace_json"] = os.path.join(image_stem, f"{qid}_token_trace.json")
            row[f"{qid}_prompt_token_attn_json"] = os.path.join(image_stem, f"{qid}_prompt_token_attn.json")
            row[f"{qid}_final_prob"] = q_trace_summary.get(qid, {}).get("final_prob", None)
            for target in keep_targets:
                row[f"{qid}_{target}_L{args.prompt_token_layer}_png"] = (
                    os.path.join(image_stem, q_png_summary.get(qid, {}).get(f"{target}_L{args.prompt_token_layer}", ""))
                    if q_png_summary.get(qid, {}).get(f"{target}_L{args.prompt_token_layer}")
                    else ""
                )

        summary_rows.append(row)
        write_summary_csv()

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
