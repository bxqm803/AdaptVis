import os
import re
import csv
import json
import math
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from transformers import CLIPModel, CLIPProcessor

from model_zoo import get_model


def load_manifest(path):
    data = json.load(open(path, "r", encoding="utf-8"))

    if isinstance(data, dict):
        try:
            return [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
        except Exception:
            return list(data.values())

    if isinstance(data, list):
        return data

    raise TypeError(f"Unsupported manifest type: {type(data)}")


def clean_obj(x):
    x = str(x).strip().strip(".").strip("?")
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.I)
    return x.strip()


def parse_objects_from_prompt(prompt):
    s = str(prompt)
    s = s.replace("<image>", " ")
    s = re.sub(r"USER:\s*", " ", s, flags=re.I)
    s = re.sub(r"ASSISTANT:\s*", " ", s, flags=re.I)
    s = " ".join(s.split())

    patterns = [
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+in relation to\s+(?:the\s+)?(.+?)\?",
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+relative to\s+(?:the\s+)?(.+?)\?",
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+with respect to\s+(?:the\s+)?(.+?)\?",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return clean_obj(m.group(1)), clean_obj(m.group(2))

    return "", ""


def get_record_value(rec, keys, default=""):
    for k in keys:
        if isinstance(rec, dict) and k in rec and rec[k] not in [None, ""]:
            return rec[k]
    return default


def get_sample_id(rec, fallback):
    return get_record_value(rec, ["sample_id", "sid", "idx", "index"], fallback)


def get_image_path(rec):
    return get_record_value(
        rec,
        ["processed_image_path", "image_path", "img_path", "path", "file_path", "processed_path"],
        "",
    )


def get_prompt(rec):
    return get_record_value(rec, ["prompt", "question", "text", "query"], "")


def get_gold(rec):
    return get_record_value(rec, ["gold", "answer", "label"], "")


def get_objects(rec):
    obj1 = get_record_value(rec, ["obj1", "object1", "subject"], "")
    obj2 = get_record_value(rec, ["obj2", "object2", "reference"], "")

    if obj1 and obj2:
        return clean_obj(obj1), clean_obj(obj2)

    return parse_objects_from_prompt(get_prompt(rec))


def get_vision_tower(model):
    if hasattr(model, "get_vision_tower"):
        vt = model.get_vision_tower()
    else:
        vt = getattr(model, "vision_tower", None)

    if isinstance(vt, (list, tuple)):
        vt = vt[0]

    if vt is None:
        raise AttributeError("Could not find LLaVA vision_tower")

    return vt


def get_vision_feature_layer(model, default=-2):
    cfg = getattr(model, "config", None)
    return getattr(cfg, "vision_feature_layer", default)


def get_vision_feature_select_strategy(model, default="default"):
    cfg = getattr(model, "config", None)
    return getattr(cfg, "vision_feature_select_strategy", default)


@torch.no_grad()
def get_llava_clip_patch_features(wrapper, clip_model, image_pil, device):
    """
    Use LLaVA's own vision tower to extract CLIP patch hidden states.
    Then use matched CLIP visual_projection to map patches into CLIP image-text shared space.
    Do NOT use LLaVA mm_projector.
    """
    image_processor = wrapper.processor.image_processor
    image_inputs = image_processor(images=image_pil, return_tensors="pt")

    vision_tower = get_vision_tower(wrapper.model)
    vision_tower.eval()

    vt_dtype = next(vision_tower.parameters()).dtype
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=vt_dtype)

    vt_out = vision_tower(pixel_values, output_hidden_states=True)

    layer_idx = get_vision_feature_layer(wrapper.model, default=-2)
    strategy = get_vision_feature_select_strategy(wrapper.model, default="default")

    hs = vt_out.hidden_states[layer_idx]

    # LLaVA default drops CLS token.
    if strategy == "default":
        hs = hs[:, 1:]
    elif strategy == "full":
        pass
    else:
        # Most LLaVA-1.5 models use "default"; keep fallback conservative.
        hs = hs[:, 1:]

    proj_dtype = clip_model.visual_projection.weight.dtype
    hs = hs.to(dtype=proj_dtype)

    patch_feats = clip_model.visual_projection(hs)  # [1, N, D]
    patch_feats = F.normalize(patch_feats, dim=-1)

    return patch_feats[0]  # [N, D]


@torch.no_grad()
def get_clip_text_features(clip_model, clip_processor, texts, device):
    text_inputs = clip_processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_feats = clip_model.get_text_features(**text_inputs)
    text_feats = F.normalize(text_feats, dim=-1)
    return text_feats


def compute_maps(wrapper, clip_model, clip_processor, image_pil, obj1, obj2, device):
    patch_feats = get_llava_clip_patch_features(wrapper, clip_model, image_pil, device)
    text_feats = get_clip_text_features(clip_model, clip_processor, [obj1, obj2], device)

    sims = torch.matmul(patch_feats, text_feats.T).detach().float().cpu().numpy()

    n_patch = sims.shape[0]
    side = int(math.sqrt(n_patch))
    if side * side != n_patch:
        raise RuntimeError(f"patch number {n_patch} is not square; cannot reshape to heatmap")

    maps = [
        sims[:, 0].reshape(side, side),
        sims[:, 1].reshape(side, side),
    ]

    stats = []
    for arr in maps:
        stats.append({
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        })

    return maps, stats


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", x)
    return x.strip("_")[:80] or "obj"


def save_raw_heatmap(arr, title, out_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(arr, cmap="viridis")
    plt.colorbar()
    plt.title(title, fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_overlay(image_pil, arr, title, out_path, alpha=0.45):
    base = image_pil.convert("RGB").resize((336, 336))

    a = np.asarray(arr, dtype=np.float32)
    a_norm = a - np.nanmin(a)
    if np.nanmax(a_norm) > 0:
        a_norm = a_norm / np.nanmax(a_norm)

    heat_small = Image.fromarray((a_norm * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    heat = np.asarray(heat_small).astype(np.float32) / 255.0
    color = (cm.jet(heat)[:, :, :3] * 255).astype(np.uint8)

    base_arr = np.asarray(base).astype(np.float32)
    out = (1.0 - alpha) * base_arr + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)

    plt.figure(figsize=(5, 5))
    plt.imshow(out)
    plt.title(title, fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def stat_title(obj, st):
    return (
        f"{obj} | raw CLIP sim "
        f"min={st['min']:+.4f}, max={st['max']:+.4f}, mean={st['mean']:+.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    print("[LOAD LLaVA WRAPPER]", args.model_name, args.method)
    wrapper, _ = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD MATCHED CLIP TEXT/PROJECTION]", args.clip_model)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model, cache_dir=args.root_dir)
    clip_model = CLIPModel.from_pretrained(
        args.clip_model,
        cache_dir=args.root_dir,
        torch_dtype=dtype,
    ).eval().to(args.device)

    # Check projection dimension compatibility.
    vt = get_vision_tower(wrapper.model)
    vt_hidden = getattr(getattr(vt, "config", None), "hidden_size", None)
    proj_in = clip_model.visual_projection.weight.shape[1]
    print("[CHECK] LLaVA vision hidden_size:", vt_hidden)
    print("[CHECK] CLIP visual_projection in_features:", proj_in)
    if vt_hidden is not None and int(vt_hidden) != int(proj_in):
        print("[WARN] hidden size mismatch. This CLIP text/projection may not match the LLaVA vision tower.")

    records = load_manifest(args.manifest_json)
    print("[MANIFEST]", args.manifest_json)
    print("[NUM RECORDS]", len(records))

    chosen = records[args.start_index: args.start_index + args.num_samples]

    summary_rows = []

    for local_i, rec in enumerate(chosen):
        sid = get_sample_id(rec, args.start_index + local_i)
        img_path = get_image_path(rec)
        prompt = get_prompt(rec)
        gold = get_gold(rec)
        obj1, obj2 = get_objects(rec)

        if not img_path or not os.path.exists(img_path):
            print(f"[SKIP] sid={sid} missing image path: {img_path}")
            continue

        if not obj1 or not obj2:
            print(f"[SKIP] sid={sid} failed to parse obj1/obj2")
            print("prompt:", prompt)
            continue

        image_pil = Image.open(img_path).convert("RGB")

        print("=" * 120)
        print(f"[RUN] sid={sid} obj1={obj1} obj2={obj2} gold={gold}")
        print("image:", img_path)

        try:
            maps, stats = compute_maps(
                wrapper=wrapper,
                clip_model=clip_model,
                clip_processor=clip_processor,
                image_pil=image_pil,
                obj1=obj1,
                obj2=obj2,
                device=args.device,
            )
        except Exception as e:
            print(f"[FAILED] sid={sid}: {e}")
            continue

        sample_dir = os.path.join(args.out_dir, f"idx{int(sid):04d}")
        os.makedirs(sample_dir, exist_ok=True)

        image_pil.save(os.path.join(sample_dir, "processed_image.png"))

        obj_names = [obj1, obj2]

        for j, obj in enumerate(obj_names):
            st = stats[j]
            arr = maps[j]
            name = safe_name(obj)

            title = stat_title(obj, st)

            save_raw_heatmap(
                arr,
                title,
                os.path.join(sample_dir, f"{j+1}_{name}_raw_heatmap.png"),
            )

            if args.save_overlay:
                save_overlay(
                    image_pil,
                    arr,
                    title,
                    os.path.join(sample_dir, f"{j+1}_{name}_overlay.png"),
                )

            if args.save_npy:
                np.save(os.path.join(sample_dir, f"{j+1}_{name}_raw_sim.npy"), arr)

            print(
                f"  {obj}: "
                f"min={st['min']:+.6f} max={st['max']:+.6f} "
                f"mean={st['mean']:+.6f} std={st['std']:+.6f} "
                f"p05={st['p05']:+.6f} p50={st['p50']:+.6f} p95={st['p95']:+.6f}"
            )

            summary_rows.append({
                "sample_id": sid,
                "image_path": img_path,
                "obj_label": f"obj{j+1}",
                "obj": obj,
                "gold": gold,
                "min": st["min"],
                "max": st["max"],
                "mean": st["mean"],
                "std": st["std"],
                "p05": st["p05"],
                "p50": st["p50"],
                "p95": st["p95"],
                "prompt": prompt,
            })

        with open(os.path.join(sample_dir, "info.txt"), "w", encoding="utf-8") as f:
            f.write(f"sample_id: {sid}\n")
            f.write(f"image_path: {img_path}\n")
            f.write(f"obj1: {obj1}\n")
            f.write(f"obj2: {obj2}\n")
            f.write(f"gold: {gold}\n")
            f.write(f"clip_model: {args.clip_model}\n")
            f.write("visual_side: LLaVA vision_tower hidden_states + CLIP visual_projection\n")
            f.write("text_side: matched CLIP text tower\n")
            f.write("mm_projector_used: False\n\n")
            f.write("PROMPT:\n")
            f.write(str(prompt))
            f.write("\n\nSTATS:\n")
            for j, obj in enumerate(obj_names):
                st = stats[j]
                f.write(
                    f"{obj}: min={st['min']:+.6f}, max={st['max']:+.6f}, "
                    f"mean={st['mean']:+.6f}, std={st['std']:+.6f}, "
                    f"p05={st['p05']:+.6f}, p50={st['p50']:+.6f}, p95={st['p95']:+.6f}\n"
                )

        print("[SAVED]", sample_dir)

    summary_csv = os.path.join(args.out_dir, "summary_stats.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "image_path",
            "obj_label",
            "obj",
            "gold",
            "min",
            "max",
            "mean",
            "std",
            "p05",
            "p50",
            "p95",
            "prompt",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print("[DONE]")
    print("[OUT DIR]", args.out_dir)
    print("[SUMMARY CSV]", summary_csv)


if __name__ == "__main__":
    main()
