import os
import re
import math
import csv
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--method", default="base", type=str)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--out-dir", default="output_llava_object_attention_scan", type=str)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=10, type=int)   # 默认前10张
    p.add_argument("--last-k-layers", default=8, type=int)
    p.add_argument("--show-token-debug", action="store_true")
    return p.parse_args()


def clean_text(x: Any) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_obj_name(x: str) -> str:
    x = clean_text(x).lower()
    x = x.replace("-", " ").replace("_", " ")
    x = re.sub(r"^(a|an|the)\s+", "", x)
    x = re.sub(r"[?.!,;:]+$", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def parse_objects_from_question(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = clean_text(question)

    patterns = [
        r"Where is the (.+?) in relation to the (.+?)\?",
        r"Where are the (.+?) in relation to the (.+?)\?",
        r"Where is (.+?) in relation to (.+?)\?",
        r"Where are (.+?) in relation to (.+?)\?",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            obj1 = normalize_obj_name(m.group(1))
            obj2 = normalize_obj_name(m.group(2))
            return obj1, obj2

    return None, None


def build_prompt(question: str) -> str:
    q = clean_text(question)
    if q.startswith("<image>"):
        return q
    return f"<image>\nUSER: {q}\nASSISTANT:"


def try_set_eager_attention(model):
    # 某些 transformers 版本下，为了 output_attentions 稳一点
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass

    for attr_name in ["language_model", "model"]:
        if hasattr(model, attr_name):
            sub = getattr(model, attr_name)
            if hasattr(sub, "config"):
                try:
                    sub.config._attn_implementation = "eager"
                except Exception:
                    pass


def find_subsequence_positions(seq: List[int], pattern: List[int]) -> Optional[List[int]]:
    n = len(seq)
    m = len(pattern)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if seq[i:i + m] == pattern:
            return list(range(i, i + m))
    return None


def find_object_token_positions(tokenizer, input_ids: torch.Tensor, obj_name: str):
    """
    在完整 input_ids 中找 obj_name 的 token span
    优先找 ' obj'，再找 'obj'
    """
    obj_name = normalize_obj_name(obj_name)
    candidates = [
        tokenizer.encode(" " + obj_name, add_special_tokens=False),
        tokenizer.encode(obj_name, add_special_tokens=False),
    ]

    seq = input_ids.tolist()
    for cand in candidates:
        pos = find_subsequence_positions(seq, cand)
        if pos is not None:
            return pos, cand

    # 再尝试把多词形式压成原始字符串
    raw_no_space = obj_name.replace(" ", "")
    more_candidates = [
        tokenizer.encode(" " + raw_no_space, add_special_tokens=False),
        tokenizer.encode(raw_no_space, add_special_tokens=False),
    ]
    for cand in more_candidates:
        pos = find_subsequence_positions(seq, cand)
        if pos is not None:
            return pos, cand

    return None, None


def infer_patch_grid(num_image_tokens: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(num_image_tokens)))
    if side * side == num_image_tokens:
        return side, side

    # fallback
    for h in range(int(math.sqrt(num_image_tokens)), 0, -1):
        if num_image_tokens % h == 0:
            w = num_image_tokens // h
            return h, w
    return 1, num_image_tokens


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    if mx > 1e-8:
        x = x / mx
    return x


def save_attention_figure(
    save_path: str,
    image_pil: Image.Image,
    obj1: str,
    obj2: str,
    obj1_heatmap: np.ndarray,
    obj2_heatmap: np.ndarray,
):
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image_pil)
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(image_pil)
    ax2.imshow(
        obj1_heatmap,
        cmap="jet",
        alpha=0.45,
        interpolation="bilinear",
        extent=(0, image_pil.size[0], image_pil.size[1], 0),
    )
    ax2.set_title(f"Attention from '{obj1}'", fontsize=12)
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(image_pil)
    ax3.imshow(
        obj2_heatmap,
        cmap="jet",
        alpha=0.45,
        interpolation="bilinear",
        extent=(0, image_pil.size[0], image_pil.size[1], 0),
    )
    ax3.set_title(f"Attention from '{obj2}'", fontsize=12)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_patch_only_figure(
    save_path: str,
    obj1: str,
    obj2: str,
    obj1_heatmap: np.ndarray,
    obj2_heatmap: np.ndarray,
):
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(obj1_heatmap, cmap="jet")
    ax1.set_title(f"Patch heatmap: {obj1}")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(obj2_heatmap, cmap="jet")
    ax2.set_title(f"Patch heatmap: {obj2}")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one_attention_scan(
    wrapper,
    image: Image.Image,
    prompt: str,
    obj1: str,
    obj2: str,
    last_k_layers: int = 8,
    show_token_debug: bool = False,
):
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    processor = wrapper.processor

    try_set_eager_attention(model)
    model.eval()

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(wrapper.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"][0]

    image_token_index = model.config.image_token_index
    image_positions = (input_ids == image_token_index).nonzero(as_tuple=False).squeeze(-1).tolist()
    if len(image_positions) == 0:
        raise ValueError("No image tokens found in input_ids.")

    obj1_positions, obj1_token_ids = find_object_token_positions(tokenizer, input_ids.detach().cpu(), obj1)
    obj2_positions, obj2_token_ids = find_object_token_positions(tokenizer, input_ids.detach().cpu(), obj2)

    if obj1_positions is None:
        raise ValueError(f"Cannot find token span for obj1='{obj1}' in prompt.")
    if obj2_positions is None:
        raise ValueError(f"Cannot find token span for obj2='{obj2}' in prompt.")

    if show_token_debug:
        tokens = tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().tolist())
        print("\n[TOKEN DEBUG]")
        for i, tok in enumerate(tokens):
            mark = ""
            if i in obj1_positions:
                mark += "[OBJ1]"
            if i in obj2_positions:
                mark += "[OBJ2]"
            if i in image_positions[:3]:
                mark += "[IMG]"
            print(f"{i:4d}: {tok:20s} {mark}")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    attentions = outputs.attentions
    if attentions is None:
        raise ValueError("outputs.attentions is None.")

    num_layers = len(attentions)
    last_k = min(last_k_layers, num_layers)
    selected_layers = list(range(num_layers - last_k, num_layers))
    image_positions_t = torch.tensor(image_positions, device=wrapper.device)

    def collect_object_to_image_attention(obj_positions: List[int]):
        per_layer_maps = []

        for layer_idx in selected_layers:
            # [batch, heads, seq, seq]
            attn = attentions[layer_idx][0]  # [heads, seq, seq]

            # 取 object tokens 作为 query，对 image tokens 的 attention
            # shape -> [heads, len(obj_tokens), len(image_tokens)]
            sub = attn[:, obj_positions, :][:, :, image_positions_t]

            # 头平均 + token平均
            vec = sub.mean(dim=0).mean(dim=0)  # [len(image_tokens)]
            per_layer_maps.append(vec.detach().float().cpu().numpy())

        avg_vec = np.mean(np.stack(per_layer_maps, axis=0), axis=0)
        gh, gw = infer_patch_grid(len(image_positions))
        heatmap = avg_vec.reshape(gh, gw)
        heatmap = normalize_map(heatmap)
        return heatmap, avg_vec, (gh, gw)

    obj1_heatmap, obj1_vec, (gh, gw) = collect_object_to_image_attention(obj1_positions)
    obj2_heatmap, obj2_vec, _ = collect_object_to_image_attention(obj2_positions)

    return {
        "prompt": prompt,
        "obj1": obj1,
        "obj2": obj2,
        "obj1_positions": obj1_positions,
        "obj2_positions": obj2_positions,
        "obj1_token_ids": obj1_token_ids,
        "obj2_token_ids": obj2_token_ids,
        "num_image_tokens": len(image_positions),
        "grid_h": gh,
        "grid_w": gw,
        "selected_layers": selected_layers,
        "obj1_heatmap": obj1_heatmap,
        "obj2_heatmap": obj2_heatmap,
        "obj1_vec": obj1_vec,
        "obj2_vec": obj2_vec,
    }


def main():
    args = parse_args()
    seed_all(args.seed)

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    if sampled_indices is not None:
        import torch.utils.data
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    out_root = os.path.join(
        args.out_dir,
        args.dataset,
        f"{args.model_name}_object_attention_scan"
    )
    os.makedirs(out_root, exist_ok=True)

    summary_csv = os.path.join(out_root, "summary_attention_scan.csv")
    summary_rows = []

    start = args.sample_index
    end = min(len(prompt_records), start + args.limit)

    used = 0
    for local_idx in range(start, end):
        rec = prompt_records[local_idx]
        item = dataset[local_idx]

        image = item["image_options"][0].convert("RGB")
        image_name = clean_text(item.get("image_name", f"sample_{local_idx:04d}.png"))
        image_path = clean_text(item.get("image_path", ""))
        raw_question = clean_text(rec["question"])

        obj1, obj2 = parse_objects_from_question(raw_question)
        if obj1 is None or obj2 is None:
            print(f"[Skip] failed to parse obj1/obj2 from question: {raw_question}")
            continue

        prompt = build_prompt(raw_question)

        print("=" * 100)
        print(f"local_idx: {local_idx}")
        print(f"image_name: {image_name}")
        print(f"obj1: {obj1}")
        print(f"obj2: {obj2}")
        print("[PROMPT]")
        print(prompt)

        try:
            result = run_one_attention_scan(
                wrapper=wrapper,
                image=image,
                prompt=prompt,
                obj1=obj1,
                obj2=obj2,
                last_k_layers=args.last_k_layers,
                show_token_debug=args.show_token_debug,
            )
        except Exception as e:
            print(f"[Error] {image_name}: {e}")
            summary_rows.append({
                "local_index": local_idx,
                "image_name": image_name,
                "image_path": image_path,
                "question": raw_question,
                "obj1": obj1,
                "obj2": obj2,
                "status": "error",
                "error": str(e),
            })
            continue

        stem = os.path.splitext(os.path.basename(image_name))[0]
        save_dir = os.path.join(out_root, stem)
        os.makedirs(save_dir, exist_ok=True)

        overlay_path = os.path.join(save_dir, "attention_overlay.png")
        patch_only_path = os.path.join(save_dir, "patch_heatmaps.png")
        meta_json_path = os.path.join(save_dir, "meta.json")

        save_attention_figure(
            save_path=overlay_path,
            image_pil=image,
            obj1=obj1,
            obj2=obj2,
            obj1_heatmap=result["obj1_heatmap"],
            obj2_heatmap=result["obj2_heatmap"],
        )

        save_patch_only_figure(
            save_path=patch_only_path,
            obj1=obj1,
            obj2=obj2,
            obj1_heatmap=result["obj1_heatmap"],
            obj2_heatmap=result["obj2_heatmap"],
        )

        meta_payload = {
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            "question": raw_question,
            "prompt": result["prompt"],
            "obj1": obj1,
            "obj2": obj2,
            "obj1_token_positions": result["obj1_positions"],
            "obj2_token_positions": result["obj2_positions"],
            "obj1_token_ids": result["obj1_token_ids"],
            "obj2_token_ids": result["obj2_token_ids"],
            "num_image_tokens": result["num_image_tokens"],
            "grid_h": result["grid_h"],
            "grid_w": result["grid_w"],
            "selected_layers": result["selected_layers"],
            "overlay_path": os.path.relpath(overlay_path, out_root),
            "patch_only_path": os.path.relpath(patch_only_path, out_root),
        }
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        summary_rows.append({
            "local_index": local_idx,
            "image_name": image_name,
            "image_path": image_path,
            "question": raw_question,
            "obj1": obj1,
            "obj2": obj2,
            "obj1_token_positions": json.dumps(result["obj1_positions"], ensure_ascii=False),
            "obj2_token_positions": json.dumps(result["obj2_positions"], ensure_ascii=False),
            "num_image_tokens": result["num_image_tokens"],
            "grid_h": result["grid_h"],
            "grid_w": result["grid_w"],
            "selected_layers": json.dumps(result["selected_layers"], ensure_ascii=False),
            "overlay_path": os.path.relpath(overlay_path, out_root),
            "patch_only_path": os.path.relpath(patch_only_path, out_root),
            "meta_json": os.path.relpath(meta_json_path, out_root),
            "status": "ok",
            "error": "",
        })

        used += 1

        fieldnames = [
            "local_index",
            "image_name",
            "image_path",
            "question",
            "obj1",
            "obj2",
            "obj1_token_positions",
            "obj2_token_positions",
            "num_image_tokens",
            "grid_h",
            "grid_w",
            "selected_layers",
            "overlay_path",
            "patch_only_path",
            "meta_json",
            "status",
            "error",
        ]
        with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("=" * 100)
    print(f"Finished. used={used}")
    print(f"Saved to: {out_root}")
    print(f"Summary:  {summary_csv}")


if __name__ == "__main__":
    main()
