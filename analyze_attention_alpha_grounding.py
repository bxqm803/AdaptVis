#!/usr/bin/env python3
"""
Analyze how image-token attention changes across alpha settings for LLaVA/AdaptVis.

It reads three raw-generation result JSON files:
  - base: w1=w2=1.0
  - low : w1=w2=0.5
  - high: w1=w2=1.5

Then it classifies samples by correctness transition and, for selected samples,
extracts final-query attention to image tokens for alpha=1.0/0.5/1.5.
Optionally it runs official GroundingDINO to get boxes for the two objects parsed
from prompts like:
  Where is/are the bowl in relation to the armchair? Answer with left, right, on or under only.

Run from the AdaptVis repo root.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

from dataset_zoo import get_dataset
from model_zoo.llava15 import LlavaWrapper


# -------------------------
# Result-file helpers
# -------------------------

def _find_one(patterns: List[str]) -> str:
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))
    hits = [
        h for h in hits
        if not h.endswith("_scores.json")
        and "alpha_effect_stats" not in h
        and "summary" not in h
        and "per_sample" not in h
    ]
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError("No file matched patterns:\n" + "\n".join(patterns))
    return hits[0]


def auto_find_files(dataset: str, option: str, threshold_tag: str = "thr0p4", test_tag: str = "True"):
    base = _find_one([
        f"output/*{dataset}*adapt_vis*w1_w11_w21*{threshold_tag}*{option}option_{test_tag}.json",
        f"output/*{dataset}*adapt_vis*w1_w11_w21*{threshold_tag}*{option}option_{test_tag.lower()}.json",
    ])
    low = _find_one([
        f"output/*{dataset}*adapt_vis*w1_w10p5_w20p5*{threshold_tag}*{option}option_{test_tag}.json",
        f"output/*{dataset}*adapt_vis*w1_w10p5_w20p5*{threshold_tag}*{option}option_{test_tag.lower()}.json",
    ])
    high = _find_one([
        f"output/*{dataset}*adapt_vis*w1_w11p5_w21p5*{threshold_tag}*{option}option_{test_tag}.json",
        f"output/*{dataset}*adapt_vis*w1_w11p5_w21p5*{threshold_tag}*{option}option_{test_tag.lower()}.json",
    ])
    return base, low, high


def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON: {path}")
    return data


def normalize_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def is_correct_item(item: dict) -> bool:
    for k in ["RawGenerationCorrect", "Correct", "correct"]:
        if k in item:
            return bool(item[k])
    gold = normalize_gold(item.get("Golden", item.get("gold", "")))
    gen = str(item.get("RawGeneration", item.get("Generation", item.get("generation", ""))))
    ok = bool(gold) and ((gold in gen) or (gold.lower() in gen.lower()))
    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False
    return ok


def sample_id_of(i: int, item: dict) -> int:
    return int(item.get("sample_id", item.get("SampleID", item.get("id", i))))


def classify_samples(base_data, low_data, high_data):
    n = min(len(base_data), len(low_data), len(high_data))
    rows = []
    for i in range(n):
        sid = sample_id_of(i, base_data[i])
        b = is_correct_item(base_data[i])
        l = is_correct_item(low_data[i])
        h = is_correct_item(high_data[i])
        rows.append({"sample_id": sid, "base": b, "low": l, "high": h, "idx": i})

    # Non-exclusive six categories. One sample may appear in both low/high transition categories.
    six = OrderedDict([
        ("low_wrong_to_correct", []),
        ("high_wrong_to_correct", []),
        ("always_correct", []),
        ("low_correct_to_wrong", []),
        ("high_correct_to_wrong", []),
        ("always_wrong", []),
    ])

    # Exclusive categories are cleaner for selecting visually distinct examples.
    exclusive = OrderedDict([
        ("always_correct", []),
        ("always_wrong", []),
        ("low_only_wrong_to_correct", []),
        ("high_only_wrong_to_correct", []),
        ("both_scalings_wrong_to_correct", []),
        ("low_only_correct_to_wrong", []),
        ("high_only_correct_to_wrong", []),
        ("both_scalings_correct_to_wrong", []),
    ])

    for r in rows:
        sid, b, l, h = r["sample_id"], r["base"], r["low"], r["high"]
        if (not b) and l:
            six["low_wrong_to_correct"].append(sid)
        if (not b) and h:
            six["high_wrong_to_correct"].append(sid)
        if b and l and h:
            six["always_correct"].append(sid)
        if b and (not l):
            six["low_correct_to_wrong"].append(sid)
        if b and (not h):
            six["high_correct_to_wrong"].append(sid)
        if (not b) and (not l) and (not h):
            six["always_wrong"].append(sid)

        if b and l and h:
            exclusive["always_correct"].append(sid)
        elif (not b) and (not l) and (not h):
            exclusive["always_wrong"].append(sid)
        elif (not b) and l and (not h):
            exclusive["low_only_wrong_to_correct"].append(sid)
        elif (not b) and (not l) and h:
            exclusive["high_only_wrong_to_correct"].append(sid)
        elif (not b) and l and h:
            exclusive["both_scalings_wrong_to_correct"].append(sid)
        elif b and (not l) and h:
            exclusive["low_only_correct_to_wrong"].append(sid)
        elif b and l and (not h):
            exclusive["high_only_correct_to_wrong"].append(sid)
        elif b and (not l) and (not h):
            exclusive["both_scalings_correct_to_wrong"].append(sid)

    status_by_id = {r["sample_id"]: r for r in rows}
    return rows, six, exclusive, status_by_id


# -------------------------
# Prompt/object parsing
# -------------------------

def clean_object_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .?\n\t")
    # Keep object phrase mostly unchanged; GroundingDINO usually likes noun phrases.
    return s


def parse_objects_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse prompts with both `Where is ...` and `Where are ...`."""
    p = re.sub(r"\s+", " ", str(prompt)).strip()

    # Main pattern: Where is/are the OBJ1 in relation to the OBJ2? Answer with ...
    pat = re.compile(
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+in\s+relation\s+to\s+(?:the\s+)?(.+?)\?\s*Answer\s+with",
        flags=re.IGNORECASE,
    )
    m = pat.search(p)
    if m:
        return clean_object_name(m.group(1)), clean_object_name(m.group(2))

    # Fallback: split by relation phrase and question mark.
    m = re.search(r"Where\s+(?:is|are)\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?", p, flags=re.IGNORECASE)
    if m:
        obj1 = re.sub(r"^the\s+", "", m.group(1), flags=re.IGNORECASE)
        obj2 = re.sub(r"^the\s+", "", m.group(2), flags=re.IGNORECASE)
        return clean_object_name(obj1), clean_object_name(obj2)

    return None, None


def load_prompt_rows(dataset_name: str, option: str) -> List[dict]:
    path = Path(f"prompts/{dataset_name}_with_answer_{option}_options.jsonl")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset_any_signature(dataset_name: str, root_dir: str, download: bool):
    try:
        return get_dataset(dataset_name, root_dir=root_dir, image_preprocess=None, download=download)
    except TypeError:
        return get_dataset(dataset_name, image_preprocess=None, download=download)


def get_raw_pil_from_dataset(dataset, idx: int) -> Image.Image:
    item = dataset[idx]
    if "image_options" in item:
        image = item["image_options"][0]
    elif "image" in item:
        image = item["image"]
    else:
        raise KeyError(f"Cannot find image in dataset item keys: {list(item.keys())}")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)}")
    return image.convert("RGB")


# -------------------------
# LLaVA attention extraction
# -------------------------

def _to_device(inputs, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def infer_num_image_patches(model) -> Tuple[int, int]:
    vc = getattr(model.config, "vision_config", None)
    image_size = int(getattr(vc, "image_size", 336))
    patch_size = int(getattr(vc, "patch_size", 14))
    grid = image_size // patch_size
    n = grid * grid
    return n, grid


def get_image_token_id(model, processor, input_ids: torch.Tensor) -> int:
    # Try tokenizer first.
    try:
        tid = processor.tokenizer.convert_tokens_to_ids("<image>")
        if tid is not None and tid != processor.tokenizer.unk_token_id and (input_ids == tid).any():
            return int(tid)
    except Exception:
        pass
    # Try config.
    for tid in [getattr(model.config, "image_token_index", None), 32000, 32001]:
        if tid is not None and (input_ids == int(tid)).any():
            return int(tid)
    raise ValueError("Could not find <image> token id in input_ids")


def expanded_positions(input_ids_1d: torch.Tensor, attn_mask_1d: torch.Tensor, image_token_id: int, num_patches: int):
    """Return image expanded positions and final non-pad query position."""
    device = input_ids_1d.device
    special = (input_ids_1d == image_token_id).long()
    # HF-style mapping: each <image> token consumes num_patches slots instead of 1.
    new_pos = torch.cumsum(special * (num_patches - 1) + 1, dim=-1) - 1

    img_idxs = torch.nonzero(special, as_tuple=False).squeeze(-1)
    if img_idxs.numel() != 1:
        raise ValueError(f"Expected exactly one image token, found {img_idxs.numel()}")
    img_idx = int(img_idxs[0].item())
    img_end = int(new_pos[img_idx].item())
    img_start = img_end - num_patches + 1
    img_positions = torch.arange(img_start, img_end + 1, device=device, dtype=torch.long)

    nonpad = torch.nonzero(attn_mask_1d, as_tuple=False).squeeze(-1)
    final_input_idx = int(nonpad[-1].item())
    final_query_pos = int(new_pos[final_input_idx].item())
    return img_positions, final_query_pos, new_pos.detach().cpu().tolist()


@torch.no_grad()
def extract_final_query_image_attention(wrapper: LlavaWrapper, image: Image.Image, prompt: str, alpha: float, max_length: int = 77):
    processor = wrapper.processor
    model = wrapper.model
    device = wrapper.device

    inputs = processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    )
    inputs = _to_device(inputs, device)

    outputs = model(
        **inputs,
        weight=float(alpha),
        output_attentions=True,
        return_dict=True,
    )

    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Try disabling SDPA/flash attention or use the repo's custom attention path."
        )

    input_ids = inputs["input_ids"][0]
    attn_mask = inputs["attention_mask"][0]
    num_patches, grid = infer_num_image_patches(model)
    image_token_id = get_image_token_id(model, processor, input_ids)
    img_positions, final_query_pos, new_pos = expanded_positions(input_ids, attn_mask, image_token_id, num_patches)

    per_layer = []
    per_layer_per_head = []
    for layer_attn in outputs.attentions:
        # Expected shape: [batch, heads, seq, seq]
        a = layer_attn[0, :, final_query_pos, img_positions].float().detach().cpu()  # [heads, num_patches]
        per_layer_per_head.append(a)
        per_layer.append(a.mean(dim=0))

    per_layer = torch.stack(per_layer, dim=0)  # [layers, num_patches]
    last = per_layer[-1]
    all_avg = per_layer.mean(dim=0)

    def normalize_map(x):
        x = x.clone().float()
        s = x.sum().item()
        if s > 0:
            x = x / s
        return x

    last_n = normalize_map(last)
    avg_n = normalize_map(all_avg)

    def metrics(x):
        x = normalize_map(x)
        eps = 1e-12
        ent = float(-(x * (x + eps).log()).sum().item())
        ent_norm = ent / math.log(len(x))
        top1 = float(x.max().item())
        top5 = float(torch.topk(x, k=min(5, len(x))).values.sum().item())
        top10 = float(torch.topk(x, k=min(10, len(x))).values.sum().item())
        return {"entropy": ent, "entropy_norm": ent_norm, "top1_mass": top1, "top5_mass": top5, "top10_mass": top10}

    return {
        "last_layer_map": last_n.reshape(grid, grid).numpy(),
        "all_layers_avg_map": avg_n.reshape(grid, grid).numpy(),
        "last_layer_metrics": metrics(last),
        "all_layers_avg_metrics": metrics(all_avg),
        "grid": grid,
        "num_patches": num_patches,
        "image_token_id": image_token_id,
        "final_query_pos": final_query_pos,
        "image_pos_start": int(img_positions[0].item()),
        "image_pos_end": int(img_positions[-1].item()),
        "expanded_positions": new_pos,
    }


# -------------------------
# GroundingDINO official backend
# -------------------------

class GroundingDINOOfficial:
    def __init__(self, config_path: str, checkpoint_path: str, device: str, box_threshold: float, text_threshold: float):
        from groundingdino.util.inference import load_model, load_image, predict
        self.load_image = load_image
        self.predict = predict
        self.model = load_model(config_path, checkpoint_path, device=device)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, pil_img: Image.Image, obj1: str, obj2: str):
        # GroundingDINO likes captions ending with periods and object phrases separated by periods.
        caption = f"{obj1}. {obj2}."
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
            pil_img.save(path)
        try:
            image_source, image_tensor = self.load_image(path)
            boxes, logits, phrases = self.predict(
                model=self.model,
                image=image_tensor,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        w, h = pil_img.size
        dets = []
        # boxes: normalized cxcywh in official GroundingDINO inference.
        for box, logit, phrase in zip(boxes, logits, phrases):
            b = box.detach().cpu().numpy().tolist()
            cx, cy, bw, bh = b
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            dets.append({
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(logit.detach().cpu().item() if torch.is_tensor(logit) else logit),
                "phrase": str(phrase),
            })
        return dets


# -------------------------
# Visualization / metrics
# -------------------------

def bbox_to_patch_mask(box_xyxy, image_size, grid: int):
    w, h = image_size
    x1, y1, x2, y2 = box_xyxy
    # Convert bbox to patch indices on grid.
    gx1 = max(0, min(grid - 1, int(math.floor(x1 / w * grid))))
    gy1 = max(0, min(grid - 1, int(math.floor(y1 / h * grid))))
    gx2 = max(0, min(grid - 1, int(math.ceil(x2 / w * grid)) - 1))
    gy2 = max(0, min(grid - 1, int(math.ceil(y2 / h * grid)) - 1))
    mask = np.zeros((grid, grid), dtype=bool)
    mask[gy1:gy2 + 1, gx1:gx2 + 1] = True
    return mask


def attention_box_mass(attn_map, detections, image_size, grid):
    if not detections:
        return {}
    result = {}
    union = np.zeros((grid, grid), dtype=bool)
    for j, det in enumerate(detections):
        mask = bbox_to_patch_mask(det["box_xyxy"], image_size, grid)
        union |= mask
        result[f"det{j}_mass"] = float(attn_map[mask].sum())
        result[f"det{j}_phrase"] = det.get("phrase", "")
        result[f"det{j}_score"] = float(det.get("score", 0.0))
        result[f"det{j}_box_xyxy"] = det["box_xyxy"]
    result["union_box_mass"] = float(attn_map[union].sum()) if union.any() else 0.0
    return result


def save_overlay(pil_img: Image.Image, attn_map: np.ndarray, detections: List[dict], title: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        # Minimal fallback: draw boxes only.
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)
        for d in detections:
            box = d["box_xyxy"]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], max(0, box[1] - 12)), d.get("phrase", "obj"), fill="red")
        img.save(path)
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(pil_img)
    # Upsample attention map to image size for display.
    hm = Image.fromarray((attn_map / (attn_map.max() + 1e-12) * 255).astype(np.uint8)).resize(pil_img.size, Image.BICUBIC)
    ax.imshow(hm, alpha=0.45, cmap="jet")
    for d in detections:
        x1, y1, x2, y2 = d["box_xyxy"]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 4), f"{d.get('phrase','')} {d.get('score',0):.2f}", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def select_ids(categories: Dict[str, List[int]], per_category: int, seed: int, requested_categories: Optional[List[str]] = None):
    rng = random.Random(seed)
    selected = OrderedDict()
    cats = requested_categories or list(categories.keys())
    for c in cats:
        ids = list(categories.get(c, []))
        if per_category > 0 and len(ids) > per_category:
            ids = rng.sample(ids, per_category)
        selected[c] = sorted(ids)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--max-length", type=int, default=77)

    parser.add_argument("--base-json", default="")
    parser.add_argument("--low-json", default="")
    parser.add_argument("--high-json", default="")
    parser.add_argument("--threshold-tag", default="thr0p4")
    parser.add_argument("--test-tag", default="True")

    parser.add_argument("--out-dir", default="output/attention_alpha_grounding")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", default="", help="Comma-separated category names. Empty = all exclusive categories.")

    parser.add_argument("--grounding", choices=["none", "official"], default="none")
    parser.add_argument("--gdino-config", default="")
    parser.add_argument("--gdino-checkpoint", default="")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.base_json and args.low_json and args.high_json:
        base_file, low_file, high_file = args.base_json, args.low_json, args.high_json
    else:
        base_file, low_file, high_file = auto_find_files(args.dataset, args.option, args.threshold_tag, args.test_tag)

    print("[FILES]")
    print("  base:", base_file)
    print("  low :", low_file)
    print("  high:", high_file)

    base_data = load_json_list(base_file)
    low_data = load_json_list(low_file)
    high_data = load_json_list(high_file)
    rows, six, exclusive, status_by_id = classify_samples(base_data, low_data, high_data)

    print("\n[ACCURACY]")
    print("  base:", sum(r["base"] for r in rows), "/", len(rows), "=", sum(r["base"] for r in rows) / len(rows))
    print("  low :", sum(r["low"] for r in rows), "/", len(rows), "=", sum(r["low"] for r in rows) / len(rows))
    print("  high:", sum(r["high"] for r in rows), "/", len(rows), "=", sum(r["high"] for r in rows) / len(rows))

    print("\n[EXCLUSIVE CATEGORIES]")
    for k, v in exclusive.items():
        print(f"  {k}: {len(v)}")

    requested_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or None
    selected = select_ids(exclusive, args.per_category, args.seed, requested_categories)
    print("\n[SELECTED]")
    for k, ids in selected.items():
        print(f"  {k}: {ids}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[LOAD DATASET]")
    dataset = load_dataset_any_signature(args.dataset, args.root_dir, args.download)
    prompt_rows = load_prompt_rows(args.dataset, args.option)

    print("\n[LOAD LLAVA]")
    wrapper = LlavaWrapper(args.root_dir, device, args.method)

    gdino = None
    if args.grounding == "official":
        if not args.gdino_config or not args.gdino_checkpoint:
            raise ValueError("--grounding official requires --gdino-config and --gdino-checkpoint")
        print("\n[LOAD GROUNDINGDINO]")
        gdino = GroundingDINOOfficial(
            args.gdino_config,
            args.gdino_checkpoint,
            device,
            args.box_threshold,
            args.text_threshold,
        )

    summary_rows = []
    alpha_values = [("base", 1.0), ("low", 0.5), ("high", 1.5)]

    for category, ids in selected.items():
        for sid in ids:
            prompt = prompt_rows[sid]["question"]
            obj1, obj2 = parse_objects_from_prompt(prompt)
            image = get_raw_pil_from_dataset(dataset, sid)
            status = status_by_id.get(sid, {})
            gold = normalize_gold(base_data[status.get("idx", sid)].get("Golden", prompt_rows[sid].get("answer", ""))) if status else ""

            print(f"\n[SAMPLE] category={category} sid={sid} status={status}")
            print("  prompt:", prompt)
            print("  parsed objects:", obj1, "|", obj2)

            detections = []
            if gdino is not None and obj1 and obj2:
                detections = gdino.detect(image, obj1, obj2)
                print("  grounding detections:", detections)

            for name, alpha in alpha_values:
                att = extract_final_query_image_attention(wrapper, image, prompt, alpha, max_length=args.max_length)
                last_map = att["last_layer_map"]
                avg_map = att["all_layers_avg_map"]
                box_mass_last = attention_box_mass(last_map, detections, image.size, att["grid"])
                box_mass_avg = attention_box_mass(avg_map, detections, image.size, att["grid"])

                sample_dir = out_dir / category / f"sample_{sid}"
                save_overlay(
                    image,
                    last_map,
                    detections,
                    f"sid={sid} {category} alpha={alpha} last-query last-layer",
                    sample_dir / f"alpha_{name}_{alpha}_last_layer.png",
                )
                save_overlay(
                    image,
                    avg_map,
                    detections,
                    f"sid={sid} {category} alpha={alpha} last-query all-layer-avg",
                    sample_dir / f"alpha_{name}_{alpha}_all_layers_avg.png",
                )

                row = {
                    "sample_id": sid,
                    "category": category,
                    "alpha_name": name,
                    "alpha": alpha,
                    "gold": gold,
                    "prompt": prompt,
                    "obj1": obj1,
                    "obj2": obj2,
                    "base_correct": status.get("base"),
                    "low_correct": status.get("low"),
                    "high_correct": status.get("high"),
                    "grid": att["grid"],
                    "final_query_pos": att["final_query_pos"],
                    "image_pos_start": att["image_pos_start"],
                    "image_pos_end": att["image_pos_end"],
                    **{f"last_{k}": v for k, v in att["last_layer_metrics"].items()},
                    **{f"avg_{k}": v for k, v in att["all_layers_avg_metrics"].items()},
                    **{f"last_box_{k}": v for k, v in box_mass_last.items()},
                    **{f"avg_box_{k}": v for k, v in box_mass_avg.items()},
                }
                summary_rows.append(row)
                print(
                    f"  alpha={alpha}: last_entropy_norm={row['last_entropy_norm']:.4f}, "
                    f"last_top5={row['last_top5_mass']:.4f}, avg_entropy_norm={row['avg_entropy_norm']:.4f}"
                )

    csv_path = out_dir / "attention_alpha_summary.csv"
    if summary_rows:
        keys = sorted({k for r in summary_rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print("\n[DONE] csv:", csv_path)
        print("[DONE] overlays under:", out_dir)
    else:
        print("\n[DONE] no selected samples")


if __name__ == "__main__":
    main()
