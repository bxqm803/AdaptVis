import os
import re
import json
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
        r"Where is\s+(?:the\s+)?(.+?)\s+in relation to\s+(?:the\s+)?(.+?)\?",
        r"Where is\s+(?:the\s+)?(.+?)\s+relative to\s+(?:the\s+)?(.+?)\?",
        r"Where is\s+(?:the\s+)?(.+?)\s+with respect to\s+(?:the\s+)?(.+?)\?",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return clean_obj(m.group(1)), clean_obj(m.group(2))

    return "", ""


def find_subsequence(haystack, needle, banned_positions=None):
    if banned_positions is None:
        banned_positions = set()
    n = len(needle)
    if n == 0:
        return []

    hits = []
    for i in range(0, len(haystack) - n + 1):
        if any((i + j) in banned_positions for j in range(n)):
            continue
        if haystack[i:i+n] == needle:
            hits.append(list(range(i, i+n)))
    return hits


def encode_phrase_variants(tokenizer, phrase):
    phrase = str(phrase).strip()
    variants = []
    for v in [
        phrase,
        " " + phrase,
        phrase.lower(),
        " " + phrase.lower(),
        phrase.title(),
        " " + phrase.title(),
        phrase.upper(),
        " " + phrase.upper(),
    ]:
        if v not in variants:
            variants.append(v)

    encoded = []
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids and ids not in encoded:
            encoded.append(ids)
    return encoded


def find_object_token_span(tokenizer, input_ids, obj, image_token_index):
    ids = [int(x) for x in input_ids]
    banned = {i for i, x in enumerate(ids) if x == int(image_token_index)}

    candidates = encode_phrase_variants(tokenizer, obj)

    for cand in candidates:
        hits = find_subsequence(ids, cand, banned_positions=banned)
        if hits:
            return hits[0], cand

    return [], []


def original_to_expanded_pos(orig_pos, image_pos, num_image_tokens=576):
    shift = int(num_image_tokens) - 1
    if orig_pos < image_pos:
        return int(orig_pos)
    if orig_pos > image_pos:
        return int(orig_pos + shift)
    raise ValueError("object token cannot be the image token")


def get_image_token_index(model):
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32001
    return int(image_token_index)


def get_image_span(single_input, model, num_image_tokens=576):
    image_token_index = get_image_token_index(model)
    ids = single_input["input_ids"][0]
    pos = torch.where(ids == image_token_index)[0]
    if len(pos) == 0:
        return None, None, None
    image_pos = int(pos[0].detach().cpu())
    image_start = image_pos
    image_end = image_start + int(num_image_tokens)
    return image_pos, image_start, image_end


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


class ObjectTokenAttentionCollector:
    def __init__(self, layers, num_layers=32, image_start=None, image_end=None, targets=None):
        self.layers = set(int(x) for x in layers)
        self.num_layers = int(num_layers)
        self.image_start = image_start
        self.image_end = image_end
        self.targets = targets or []
        self.call_idx = 0
        self.maps = {}

    def update(self, probs):
        if not torch.is_tensor(probs):
            return
        if probs.dim() != 4:
            return
        if self.image_start is None or self.image_end is None:
            return
        if probs.shape[-1] < self.image_end:
            return

        max_q = max([t["expanded_pos"] for t in self.targets], default=-1)
        if probs.shape[-2] <= max_q:
            return

        layer = self.call_idx % self.num_layers
        self.call_idx += 1

        if layer not in self.layers:
            return

        for t in self.targets:
            qpos = int(t["expanded_pos"])
            vals = probs[0, :, qpos, self.image_start:self.image_end].detach().float().cpu()
            # mean over heads -> 576
            arr = vals.mean(dim=0).numpy().reshape(24, 24)
            key = (layer, t["obj_label"], t["token_order"])
            self.maps[key] = arr


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


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_\-]+", "_", x)
    return x.strip("_")[:80]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-image-tokens", type=int, default=576)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--fresh-limit", type=int, default=20)
    parser.add_argument("--idxs", default="")
    parser.add_argument("--out-dir", default="output/controlledA_object_token_visual_attention")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    target_idx_set = None
    if args.idxs.strip():
        target_idx_set = {int(x) for x in args.idxs.split(",") if x.strip()}

    print("[LOAD MODEL]", args.model_name, args.method)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    tokenizer = wrapper.processor.tokenizer
    image_token_index = get_image_token_index(wrapper.model)

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
    collate_fn = _default_collate if image_preprocess is None else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    prompts, answers = sf.load_prompts(args.dataset, args.option)

    meta = {}
    num_done = 0

    for sid, image in tqdm(iter_samples(loader), desc="object-token visual attention"):
        if sid >= len(prompts):
            break
        if target_idx_set is not None and sid not in target_idx_set:
            continue
        if target_idx_set is None and args.fresh_limit > 0 and num_done >= args.fresh_limit:
            break

        prompt = prompts[sid]
        gold = answers[sid]
        obj1, obj2 = parse_objects_from_prompt(prompt)

        if not obj1 or not obj2:
            print("[WARN] parse obj failed sid", sid, prompt)
            continue

        single_input = wrapper.processor(
            text=prompt,
            images=image,
            padding="max_length",
            return_tensors="pt",
            max_length=args.max_length,
        ).to(args.device)

        input_ids = [int(x) for x in single_input["input_ids"][0].detach().cpu().tolist()]
        image_pos, image_start, image_end = get_image_span(
            single_input,
            wrapper.model,
            num_image_tokens=args.num_image_tokens,
        )

        if image_pos is None:
            print("[WARN] no image token sid", sid)
            continue

        targets = []
        obj_infos = []

        for obj_label, obj in [("obj1", obj1), ("obj2", obj2)]:
            span, matched_ids = find_object_token_span(tokenizer, input_ids, obj, image_token_index)
            if not span:
                print("[WARN] cannot find token span", sid, obj_label, obj)
                continue

            token_texts = [tokenizer.decode([input_ids[p]], skip_special_tokens=False) for p in span]
            expanded_positions = [
                original_to_expanded_pos(p, image_pos, args.num_image_tokens)
                for p in span
            ]

            obj_infos.append({
                "obj_label": obj_label,
                "obj": obj,
                "orig_span": span,
                "expanded_positions": expanded_positions,
                "token_texts": token_texts,
                "matched_ids": matched_ids,
            })

            for j, (orig_p, exp_p, tok_txt) in enumerate(zip(span, expanded_positions, token_texts)):
                targets.append({
                    "obj_label": obj_label,
                    "obj": obj,
                    "token_order": j,
                    "orig_pos": int(orig_p),
                    "expanded_pos": int(exp_p),
                    "token_text": tok_txt,
                })

        if not targets:
            continue

        proc_img = pixel_values_to_pil(single_input, wrapper.processor)
        if proc_img is None:
            proc_img = image if isinstance(image, Image.Image) else Image.new("RGB", (336, 336), (0, 0, 0))

        collector = ObjectTokenAttentionCollector(
            layers=layers,
            num_layers=args.num_layers,
            image_start=image_start,
            image_end=image_end,
            targets=targets,
        )

        with torch.no_grad():
            with patch_softmax_collect_attention(collector):
                try:
                    wrapper.model(**single_input, use_cache=False)
                except TypeError:
                    wrapper.model(**single_input)

        sample_dir = os.path.join(args.out_dir, f"idx{sid:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        proc_img.save(os.path.join(sample_dir, "model_input_image.png"))

        sample_meta = {
            "sample_idx": sid,
            "prompt": prompt,
            "gold": gold,
            "obj1": obj1,
            "obj2": obj2,
            "image_pos": image_pos,
            "image_start": image_start,
            "image_end": image_end,
            "objects": obj_infos,
            "saved": [],
        }

        for layer in layers:
            for info in obj_infos:
                obj_label = info["obj_label"]
                obj = info["obj"]

                token_maps = []

                for j, tok_txt in enumerate(info["token_texts"]):
                    key = (layer, obj_label, j)
                    if key not in collector.maps:
                        continue

                    arr = collector.maps[key]
                    token_maps.append(arr)

                    base = f"layer{layer:02d}_{obj_label}_{safe_name(obj)}_tok{j}_{safe_name(tok_txt)}"
                    heat_path = os.path.join(sample_dir, base + "_heat.png")
                    overlay_path = os.path.join(sample_dir, base + "_overlay.png")

                    save_heatmap(arr, heat_path, title=base)
                    save_overlay(proc_img, arr, overlay_path)

                    sample_meta["saved"].append({
                        "layer": layer,
                        "obj_label": obj_label,
                        "obj": obj,
                        "token_order": j,
                        "token_text": tok_txt,
                        "heat_path": heat_path,
                        "overlay_path": overlay_path,
                    })

                if token_maps:
                    phrase_arr = np.mean(np.stack(token_maps, axis=0), axis=0)
                    base = f"layer{layer:02d}_{obj_label}_{safe_name(obj)}_phrase_mean"
                    heat_path = os.path.join(sample_dir, base + "_heat.png")
                    overlay_path = os.path.join(sample_dir, base + "_overlay.png")

                    save_heatmap(phrase_arr, heat_path, title=base)
                    save_overlay(proc_img, phrase_arr, overlay_path)

                    sample_meta["saved"].append({
                        "layer": layer,
                        "obj_label": obj_label,
                        "obj": obj,
                        "token_order": "phrase_mean",
                        "token_text": "phrase_mean",
                        "heat_path": heat_path,
                        "overlay_path": overlay_path,
                    })

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2, ensure_ascii=False)

        meta[str(sid)] = sample_meta
        num_done += 1

        if sid < 3:
            print("[DEBUG]", sid, obj1, obj2)
            for info in obj_infos:
                print(" ", info)

    with open(os.path.join(args.out_dir, "all_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[DONE]")
    print("[OUT DIR]", args.out_dir)
    print("num samples:", num_done)


if __name__ == "__main__":
    main()
