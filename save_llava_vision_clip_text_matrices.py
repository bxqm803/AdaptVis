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
from tqdm import tqdm
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

    if strategy == "default":
        hs = hs[:, 1:]
    elif strategy == "full":
        pass
    else:
        hs = hs[:, 1:]

    proj_dtype = clip_model.visual_projection.weight.dtype
    hs = hs.to(dtype=proj_dtype)

    patch_feats = clip_model.visual_projection(hs)
    patch_feats = F.normalize(patch_feats, dim=-1)

    return patch_feats[0]


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
        raise RuntimeError(f"patch number {n_patch} is not square")

    sim_obj1 = sims[:, 0].reshape(side, side)
    sim_obj2 = sims[:, 1].reshape(side, side)

    return sim_obj1, sim_obj2


def stats(arr):
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14-336")

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    print("[LOAD LLaVA]", args.model_name, args.method)
    wrapper, _ = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD CLIP TEXT/PROJECTION]", args.clip_model)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model, cache_dir=args.root_dir)
    clip_model = CLIPModel.from_pretrained(
        args.clip_model,
        cache_dir=args.root_dir,
        torch_dtype=dtype,
    ).eval().to(args.device)

    vt = get_vision_tower(wrapper.model)
    vt_hidden = getattr(getattr(vt, "config", None), "hidden_size", None)
    proj_in = clip_model.visual_projection.weight.shape[1]
    print("[CHECK] LLaVA vision hidden_size:", vt_hidden)
    print("[CHECK] CLIP visual_projection in_features:", proj_in)
    if vt_hidden is not None and int(vt_hidden) != int(proj_in):
        print("[WARN] hidden size mismatch: CLIP checkpoint may not match LLaVA vision tower")

    records = load_manifest(args.manifest_json)
    if args.num_samples is None or args.num_samples < 0:
        chosen = records[args.start_index:]
    else:
        chosen = records[args.start_index: args.start_index + args.num_samples]

    print("[MANIFEST]", args.manifest_json)
    print("[NUM RECORDS]", len(records))
    print("[RUN RECORDS]", len(chosen))

    sample_ids = []
    image_paths = []
    obj1_names = []
    obj2_names = []
    golds = []
    prompts = []
    sim_obj1_list = []
    sim_obj2_list = []
    summary_rows = []

    for local_i, rec in enumerate(tqdm(chosen, desc="save clip sim matrices")):
        fallback_sid = args.start_index + local_i
        sid = get_sample_id(rec, fallback_sid)
        img_path = get_image_path(rec)
        prompt = get_prompt(rec)
        gold = get_gold(rec)
        obj1, obj2 = get_objects(rec)

        if not img_path or not os.path.exists(img_path):
            print(f"[SKIP] sid={sid} missing image: {img_path}")
            continue

        if not obj1 or not obj2:
            print(f"[SKIP] sid={sid} parse obj failed")
            print("prompt:", prompt)
            continue

        try:
            image_pil = Image.open(img_path).convert("RGB")
            sim1, sim2 = compute_maps(
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

        sample_ids.append(int(sid))
        image_paths.append(str(img_path))
        obj1_names.append(str(obj1))
        obj2_names.append(str(obj2))
        golds.append(str(gold))
        prompts.append(str(prompt))
        sim_obj1_list.append(sim1.astype(np.float32))
        sim_obj2_list.append(sim2.astype(np.float32))

        s1 = stats(sim1)
        s2 = stats(sim2)

        summary_rows.append({
            "sample_id": sid,
            "image_path": img_path,
            "obj_label": "obj1",
            "obj": obj1,
            "gold": gold,
            **s1,
            "prompt": prompt,
        })
        summary_rows.append({
            "sample_id": sid,
            "image_path": img_path,
            "obj_label": "obj2",
            "obj": obj2,
            "gold": gold,
            **s2,
            "prompt": prompt,
        })

    if not sim_obj1_list:
        raise RuntimeError("No valid samples processed")

    sim_obj1_arr = np.stack(sim_obj1_list, axis=0)
    sim_obj2_arr = np.stack(sim_obj2_list, axis=0)

    out_npz = os.path.join(args.out_dir, "clip_obj_sim_matrices.npz")
    np.savez_compressed(
        out_npz,
        sample_ids=np.array(sample_ids, dtype=np.int64),
        image_paths=np.array(image_paths, dtype=object),
        obj1_names=np.array(obj1_names, dtype=object),
        obj2_names=np.array(obj2_names, dtype=object),
        gold=np.array(golds, dtype=object),
        prompts=np.array(prompts, dtype=object),
        sim_obj1=sim_obj1_arr,
        sim_obj2=sim_obj2_arr,
    )

    out_csv = os.path.join(args.out_dir, "clip_obj_sim_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "image_path", "obj_label", "obj", "gold",
            "min", "max", "mean", "std", "p05", "p50", "p95", "prompt",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print("[DONE]")
    print("[OUT NPZ]", out_npz)
    print("[OUT CSV]", out_csv)
    print("num_valid:", len(sample_ids))
    print("sim_obj1 shape:", sim_obj1_arr.shape)
    print("sim_obj2 shape:", sim_obj2_arr.shape)


if __name__ == "__main__":
    main()
