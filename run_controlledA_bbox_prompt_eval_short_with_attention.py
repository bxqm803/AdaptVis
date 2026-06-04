import os
import re
import csv
import json
import random
import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from model_zoo import get_model
from dataset_zoo import get_dataset
import save_llava_hidden_similarity_features as sf

try:
    from misc import _default_collate
except Exception:
    _default_collate = None


def norm_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def raw_generation_correct(gold, gen):
    gold = norm_gold(gold)
    gen = str(gen)
    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False
    return bool(ok)


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


def clean_obj(x):
    x = str(x).strip().strip(".").strip("?")
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.I)
    return x.strip()


def parse_objects_from_prompt(prompt):
    s = str(prompt)
    s = s.replace("<image>", " ")
    s = re.sub(r"USER:\s*", " ", s, flags=re.I)
    s = re.sub(r"ASSISTANT:\s*", " ", s, flags=re.I)

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


def load_bbox_json(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, list):
        new_data = {}
        for x in data:
            sid = x.get("sample_id", x.get("sid", x.get("idx", None)))
            if sid is not None:
                new_data[str(int(sid))] = x
        data = new_data
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported bbox json type: {type(data)}")
    return data


def norm_name(x):
    x = str(x).lower().strip()
    x = re.sub(r"^(a|an|the)\s+", "", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return x.strip()


def label_match(label, obj):
    a = norm_name(label)
    b = norm_name(obj)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def patch_ids_to_norm_box(patch_ids, patch_side=24):
    patch_ids = [int(x) for x in patch_ids if 0 <= int(x) < patch_side * patch_side]
    if not patch_ids:
        return None

    rows = [p // patch_side for p in patch_ids]
    cols = [p % patch_side for p in patch_ids]

    r1, r2 = min(rows), max(rows) + 1
    c1, c2 = min(cols), max(cols) + 1

    return [c1 / patch_side, r1 / patch_side, c2 / patch_side, r2 / patch_side]


def box_to_norm(box, image_size=336):
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        out = [
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        ]
    else:
        out = [
            max(0.0, min(1.0, x1 / image_size)),
            max(0.0, min(1.0, y1 / image_size)),
            max(0.0, min(1.0, x2 / image_size)),
            max(0.0, min(1.0, y2 / image_size)),
        ]

    if out[2] <= out[0] or out[3] <= out[1]:
        return None

    return out


def get_box_from_det(det):
    for k in ["box_xyxy_processed", "box_xyxy", "bbox_xyxy", "bbox", "box"]:
        if k in det:
            return det[k]
    return None


def get_label_from_det(det):
    for k in ["target", "label", "name", "object", "class_name"]:
        if k in det:
            return str(det[k])
    return ""


def extract_obj_box(rec, obj, obj_label=None, image_size=336):
    if rec is None:
        return None, "missing_record"

    if obj_label:
        for k in [
            f"{obj_label}_box_norm",
            f"{obj_label}_box_xyxy_processed",
            f"{obj_label}_box_xyxy",
            f"{obj_label}_box",
            f"{obj_label}_bbox",
        ]:
            if k in rec:
                b = box_to_norm(rec[k], image_size=image_size)
                if b is not None:
                    return b, k

        for k in [f"{obj_label}_patch_ids", f"{obj_label}_patches"]:
            if k in rec:
                b = patch_ids_to_norm_box(rec[k])
                if b is not None:
                    return b, k

    for dict_key in ["boxes_by_object", "bbox_by_object", "object_boxes"]:
        d = rec.get(dict_key, None)
        if isinstance(d, dict):
            for name, val in d.items():
                if label_match(name, obj):
                    return box_to_norm(val, image_size=image_size), dict_key

    for dict_key in ["patch_ids_by_object", "patch_ids_by_obj", "object_patch_ids"]:
        d = rec.get(dict_key, None)
        if isinstance(d, dict):
            for name, val in d.items():
                if label_match(name, obj):
                    return patch_ids_to_norm_box(val), dict_key

    dets = rec.get("detections", rec.get("objects", rec.get("boxes", [])))
    if isinstance(dets, dict):
        dets = list(dets.values())

    best = None
    best_score = -1e9

    if isinstance(dets, list):
        for det in dets:
            if not isinstance(det, dict):
                continue
            label = get_label_from_det(det)
            if not label_match(label, obj):
                continue

            box = get_box_from_det(det)
            normed_box = box_to_norm(box, image_size=image_size) if box is not None else None

            if normed_box is None and "patch_ids" in det:
                normed_box = patch_ids_to_norm_box(det["patch_ids"])
                if normed_box is not None:
                    return normed_box, "detections.patch_ids"

            if normed_box is None:
                continue

            score = float(det.get("score", det.get("confidence", 0.0)))
            if score > best_score:
                best_score = score
                best = normed_box

    if best is not None:
        return best, "detections.box"

    return None, "not_found"


def extract_union_patch_box(rec):
    if rec is None:
        return None
    for k in ["patch_ids", "object_patch_ids", "all_patch_ids"]:
        if k in rec:
            return patch_ids_to_norm_box(rec[k])
    return None


def fmt_box_short(box):
    x1, y1, x2, y2 = box
    return f"[{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]"


def build_short_bbox_hint(obj1, obj2, box1, box2, union_box=None):
    if box1 is not None and box2 is not None:
        return (
            "Given two object bounding boxes in normalized image coordinates [x1,y1,x2,y2], "
            "where x increases left-to-right and y increases top-to-bottom:\n"
            f"obj1 = {obj1}, bbox1 = {fmt_box_short(box1)}\n"
            f"obj2 = {obj2}, bbox2 = {fmt_box_short(box2)}"
        )

    if union_box is not None:
        return (
            "Given the relevant object region in normalized image coordinates [x1,y1,x2,y2], "
            "where x increases left-to-right and y increases top-to-bottom:\n"
            f"region = {fmt_box_short(union_box)}"
        )

    return ""


def inject_hint(prompt, hint):
    if not hint:
        return prompt

    q = str(prompt)
    q = q.replace("<image>", "").strip()
    q = re.sub(r"^USER:\s*", "", q, flags=re.I).strip()
    q = re.sub(r"ASSISTANT:\s*$", "", q, flags=re.I).strip()

    return (
        "<image>\n"
        "USER: "
        + hint
        + "\nQuestion: "
        + q
        + "\nASSISTANT:"
    )


def generation_scores(output):
    if hasattr(output, "scores"):
        return output.scores
    if isinstance(output, dict):
        return output.get("scores", None)
    return output["scores"]


def generation_sequences(output):
    if hasattr(output, "sequences"):
        return output.sequences
    if isinstance(output, dict):
        return output.get("sequences", None)
    return output["sequences"]


def first_step_confidence(output):
    scores = generation_scores(output)
    if scores is None or len(scores) == 0:
        return 0.0
    prob = torch.nn.functional.softmax(scores[0], dim=-1)
    return float(torch.max(prob[0]).detach().float().cpu())


def decode_generated(processor, output, prompt_len):
    seq = generation_sequences(output)
    return processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True)


def get_image_token_index(model):
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32001
    return int(image_token_index)


def get_image_and_text_indices(single_input, model, num_image_tokens=576):
    image_token_index = get_image_token_index(model)
    ids = single_input["input_ids"][0]

    attn_mask = single_input.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask[0]
    else:
        attn_mask = torch.ones_like(ids)

    pos = torch.where(ids == int(image_token_index))[0]
    if len(pos) == 0:
        return None, None, []

    image_pos = int(pos[0].detach().cpu())
    image_start = image_pos
    image_end = image_start + int(num_image_tokens)

    text_indices = []
    shift = int(num_image_tokens) - 1

    for i in range(ids.shape[0]):
        if int(attn_mask[i].detach().cpu()) == 0:
            continue
        if i == image_pos:
            continue

        if i < image_pos:
            expanded_i = i
        else:
            expanded_i = i + shift

        text_indices.append(int(expanded_i))

    return image_start, image_end, text_indices


def pixel_values_to_pil(single_input, processor):
    pv = single_input.get("pixel_values", None)
    if pv is None:
        return None

    x = pv[0].detach().float().cpu()

    image_processor = getattr(processor, "image_processor", None)
    mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_heatmap(arr24, out_path, title=""):
    plt.figure(figsize=(4, 4))
    plt.imshow(arr24, cmap="inferno")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=8)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(base_img, arr24, out_path, alpha=0.45):
    base = base_img.convert("RGB").resize((336, 336))
    a = np.asarray(arr24, dtype=np.float32)
    a = a - np.nanmin(a)
    if np.nanmax(a) > 0:
        a = a / np.nanmax(a)

    heat_small = Image.fromarray((a * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    heat = np.asarray(heat_small).astype(np.float32) / 255.0
    color = (cm.inferno(heat)[:, :, :3] * 255).astype(np.uint8)

    base_arr = np.asarray(base).astype(np.float32)
    out = (1 - alpha) * base_arr + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


class AttentionCollector:
    def __init__(self, num_layers, image_start, image_end, text_indices, heatmap_layers):
        self.num_layers = int(num_layers)
        self.image_start = int(image_start)
        self.image_end = int(image_end)
        self.text_indices = [int(x) for x in text_indices]
        self.heatmap_layers = set(int(x) for x in heatmap_layers)
        self.call_idx = 0
        self.mass = {}
        self.heatmaps = {}

    def update(self, probs):
        if not torch.is_tensor(probs):
            return
        if probs.dim() != 4:
            return
        if probs.shape[-1] < self.image_end:
            return

        # Only collect the prefill forward, not decoding q_len=1 steps.
        if probs.shape[-2] <= 1:
            return

        layer = self.call_idx % self.num_layers
        self.call_idx += 1

        row = probs[:, :, -1, :].detach().float()
        row = row[torch.isfinite(row).all(dim=-1)]
        if row.numel() == 0:
            return

        kv_len = row.shape[-1]
        image_abs = list(range(self.image_start, min(self.image_end, kv_len)))
        text_abs = [i for i in self.text_indices if 0 <= i < kv_len]

        if not image_abs:
            return

        image_idx = torch.tensor(image_abs, device=row.device, dtype=torch.long)
        image_probs = row.index_select(dim=-1, index=image_idx)
        image_mass = image_probs.sum(dim=-1)

        if text_abs:
            text_idx = torch.tensor(text_abs, device=row.device, dtype=torch.long)
            text_probs = row.index_select(dim=-1, index=text_idx)
            text_mass = text_probs.sum(dim=-1)
        else:
            text_mass = torch.zeros_like(image_mass)

        row_mass = row.sum(dim=-1)
        other_mass = row_mass - image_mass - text_mass

        self.mass[int(layer)] = {
            "image_mass": float(image_mass.mean().detach().cpu()),
            "text_mass": float(text_mass.mean().detach().cpu()),
            "other_mass": float(other_mass.mean().detach().cpu()),
            "row_mass": float(row_mass.mean().detach().cpu()),
            "image_frac_over_image_text": float(
                (image_mass / torch.clamp(image_mass + text_mass, min=1e-12)).mean().detach().cpu()
            ),
            "text_frac_over_image_text": float(
                (text_mass / torch.clamp(image_mass + text_mass, min=1e-12)).mean().detach().cpu()
            ),
        }

        if int(layer) in self.heatmap_layers:
            arr = image_probs.mean(dim=0).detach().float().cpu().numpy().reshape(24, 24)
            self.heatmaps[int(layer)] = arr


@contextmanager
def patch_softmax_collect_attention(collector):
    old_f_softmax = F.softmax
    old_torch_softmax = torch.softmax

    def wrapped_f_softmax(input, *args, **kwargs):
        out = old_f_softmax(input, *args, **kwargs)
        collector.update(out)
        return out

    def wrapped_torch_softmax(input, *args, **kwargs):
        out = old_torch_softmax(input, *args, **kwargs)
        collector.update(out)
        return out

    F.softmax = wrapped_f_softmax
    torch.softmax = wrapped_torch_softmax
    try:
        yield
    finally:
        F.softmax = old_f_softmax
        torch.softmax = old_torch_softmax


def run_one_with_attention(wrapper, image, prompt, device, max_length, max_new_tokens, num_layers, num_image_tokens, heatmap_layers):
    single_input = wrapper.processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    ).to(device)

    prompt_len = len(single_input["input_ids"][-1])
    image_start, image_end, text_indices = get_image_and_text_indices(
        single_input,
        wrapper.model,
        num_image_tokens=num_image_tokens,
    )

    collector = None
    if image_start is not None:
        collector = AttentionCollector(
            num_layers=num_layers,
            image_start=image_start,
            image_end=image_end,
            text_indices=text_indices,
            heatmap_layers=heatmap_layers,
        )

    proc_img = pixel_values_to_pil(single_input, wrapper.processor)

    with torch.no_grad():
        if collector is None:
            output = wrapper.model.generate(
                **single_input,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            with patch_softmax_collect_attention(collector):
                output = wrapper.model.generate(
                    **single_input,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

    gen = decode_generated(wrapper.processor, output, prompt_len)
    conf = first_step_confidence(output)

    return {
        "generation": gen,
        "confidence": conf,
        "mass": collector.mass if collector is not None else {},
        "heatmaps": collector.heatmaps if collector is not None else {},
        "proc_img": proc_img,
    }


def save_random_examples(records, example_ids, out_dir, prefix="bbox_prompt"):
    os.makedirs(out_dir, exist_ok=True)
    for r in records:
        sid = int(r["sample_id"])
        if sid not in example_ids:
            continue

        sample_dir = os.path.join(out_dir, f"idx{sid:04d}_{prefix}")
        os.makedirs(sample_dir, exist_ok=True)

        if r.get("_proc_img") is not None:
            r["_proc_img"].save(os.path.join(sample_dir, "model_input_image.png"))

        for layer, arr in r.get("_heatmaps", {}).items():
            heat_path = os.path.join(sample_dir, f"layer{int(layer):02d}_last_query_visual_heat.png")
            overlay_path = os.path.join(sample_dir, f"layer{int(layer):02d}_last_query_visual_overlay.png")
            save_heatmap(arr, heat_path, title=f"idx{sid} layer {layer} last-query -> visual")
            if r.get("_proc_img") is not None:
                save_overlay(r["_proc_img"], arr, overlay_path)

        with open(os.path.join(sample_dir, "prompt_and_answer.txt"), "w", encoding="utf-8") as f:
            f.write("GOLD: " + str(r["gold"]) + "\n")
            f.write("BASE: " + str(r["base_generation"]) + "\n")
            f.write("BBOX_PROMPT: " + str(r["bbox_prompt_generation"]) + "\n\n")
            f.write(str(r["bbox_prompt"]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--bbox-json", default="output/groundingdino_object_patch_masks_by_sid.json")
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-image-tokens", type=int, default=576)
    parser.add_argument("--heatmap-layers", default="0,1")
    parser.add_argument("--fresh-limit", type=int, default=-1)
    parser.add_argument("--require-two-boxes", action="store_true")
    parser.add_argument("--random-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-csv", default="output/ControlledA_bbox_prompt_eval_short_with_attention.csv")
    parser.add_argument("--mass-csv", default="output/ControlledA_bbox_prompt_attention_mass.csv")
    parser.add_argument("--example-dir", default="output/ControlledA_bbox_prompt_random5_attention_examples")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.mass_csv), exist_ok=True)
    os.makedirs(args.example_dir, exist_ok=True)

    heatmap_layers = [int(x) for x in str(args.heatmap_layers).split(",") if x.strip()]

    print("[LOAD BBOX]", args.bbox_json)
    bbox_data = load_bbox_json(args.bbox_json)

    print("[LOAD MODEL]", args.model_name, args.method)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
    collate_fn = _default_collate if image_preprocess is None else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    prompts, answers = sf.load_prompts(args.dataset, args.option)

    rows = []
    mass_rows = []

    base_correct = 0
    bbox_correct = 0
    used = 0
    two_box_count = 0
    w2c = 0
    c2w = 0
    c2c = 0
    w2w = 0

    pbar = tqdm(iter_samples(loader), desc="bbox prompt eval + attention")

    for sid, image in pbar:
        if sid >= len(prompts):
            break
        if args.fresh_limit > 0 and used >= args.fresh_limit:
            break

        prompt = prompts[sid]
        gold = norm_gold(answers[sid])
        obj1, obj2 = parse_objects_from_prompt(prompt)

        rec = bbox_data.get(str(sid), None)

        box1, src1 = extract_obj_box(rec, obj1, obj_label="obj1")
        box2, src2 = extract_obj_box(rec, obj2, obj_label="obj2")
        union_box = extract_union_patch_box(rec)

        has_two_boxes = box1 is not None and box2 is not None
        if has_two_boxes:
            two_box_count += 1

        if args.require_two_boxes and not has_two_boxes:
            continue

        hint = build_short_bbox_hint(obj1, obj2, box1, box2, union_box=union_box)
        bbox_prompt = inject_hint(prompt, hint)

        if used < 5:
            print("\n" + "=" * 80)
            print(f"[PROMPT CHECK sid={sid}]")
            print("[ORIGINAL]")
            print(prompt)
            print("[BBOX PROMPT]")
            print(bbox_prompt)
            print("[GOLD]", gold)
            print("[BOX1]", obj1, box1, src1)
            print("[BOX2]", obj2, box2, src2)
            print("=" * 80)

        base_res = run_one_with_attention(
            wrapper=wrapper,
            image=image,
            prompt=prompt,
            device=args.device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            num_layers=args.num_layers,
            num_image_tokens=args.num_image_tokens,
            heatmap_layers=[],
        )

        bbox_res = run_one_with_attention(
            wrapper=wrapper,
            image=image,
            prompt=bbox_prompt,
            device=args.device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            num_layers=args.num_layers,
            num_image_tokens=args.num_image_tokens,
            heatmap_layers=heatmap_layers,
        )

        base_gen = base_res["generation"]
        bbox_gen = bbox_res["generation"]
        base_conf = base_res["confidence"]
        bbox_conf = bbox_res["confidence"]

        base_corr = raw_generation_correct(gold, base_gen)
        bbox_corr = raw_generation_correct(gold, bbox_gen)

        base_correct += int(base_corr)
        bbox_correct += int(bbox_corr)
        used += 1

        if (not base_corr) and bbox_corr:
            w2c += 1
            transition = "wrong_to_correct"
        elif base_corr and (not bbox_corr):
            c2w += 1
            transition = "correct_to_wrong"
        elif base_corr and bbox_corr:
            c2c += 1
            transition = "correct_to_correct"
        else:
            w2w += 1
            transition = "wrong_to_wrong"

        row = {
            "sample_id": sid,
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,
            "has_two_boxes": has_two_boxes,
            "box1_source": src1,
            "box2_source": src2,
            "box1": box1,
            "box2": box2,
            "union_box": union_box,
            "base_generation": base_gen,
            "base_correct": base_corr,
            "base_confidence": base_conf,
            "bbox_prompt_generation": bbox_gen,
            "bbox_prompt_correct": bbox_corr,
            "bbox_prompt_confidence": bbox_conf,
            "transition": transition,
            "bbox_prompt": bbox_prompt,
            "_heatmaps": bbox_res["heatmaps"],
            "_proc_img": bbox_res["proc_img"],
        }
        rows.append(row)

        for layer in range(args.num_layers):
            m = bbox_res["mass"].get(layer, None)
            if m is None:
                continue
            mass_rows.append({
                "sample_id": sid,
                "mode": "bbox_prompt",
                "layer": layer,
                "image_mass": m["image_mass"],
                "text_mass": m["text_mass"],
                "other_mass": m["other_mass"],
                "row_mass": m["row_mass"],
                "image_frac_over_image_text": m["image_frac_over_image_text"],
                "text_frac_over_image_text": m["text_frac_over_image_text"],
            })

        pbar.set_postfix({
            "sid": sid,
            "base_acc": f"{base_correct / max(used, 1):.3f}",
            "bbox_acc": f"{bbox_correct / max(used, 1):.3f}",
            "two_box": two_box_count,
        })

    random.seed(args.seed)
    ids = [int(r["sample_id"]) for r in rows]
    example_ids = set(random.sample(ids, min(args.random_examples, len(ids))))
    save_random_examples(rows, example_ids, args.example_dir, prefix="bbox_prompt")

    fieldnames = [
        "sample_id",
        "obj1",
        "obj2",
        "gold",
        "has_two_boxes",
        "box1_source",
        "box2_source",
        "box1",
        "box2",
        "union_box",
        "base_generation",
        "base_correct",
        "base_confidence",
        "bbox_prompt_generation",
        "bbox_prompt_correct",
        "bbox_prompt_confidence",
        "transition",
        "bbox_prompt",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    mass_fieldnames = [
        "sample_id",
        "mode",
        "layer",
        "image_mass",
        "text_mass",
        "other_mass",
        "row_mass",
        "image_frac_over_image_text",
        "text_frac_over_image_text",
    ]

    with open(args.mass_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mass_fieldnames)
        w.writeheader()
        w.writerows(mass_rows)

    print("\n[DONE]")
    print("[CSV]", args.out_csv)
    print("[MASS CSV]", args.mass_csv)
    print("[EXAMPLE DIR]", args.example_dir)
    print("num_used:", used)
    print("num_two_boxes:", two_box_count)
    print("base_acc:", base_correct / max(used, 1))
    print("bbox_prompt_acc:", bbox_correct / max(used, 1))
    print("wrong_to_correct:", w2c)
    print("correct_to_wrong:", c2w)
    print("correct_to_correct:", c2c)
    print("wrong_to_wrong:", w2w)
    print("random_example_ids:", sorted(example_ids))


if __name__ == "__main__":
    main()
