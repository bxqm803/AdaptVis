import os
import re
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

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

    pats = [
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+in relation to\s+(?:the\s+)?(.+?)\?",
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+relative to\s+(?:the\s+)?(.+?)\?",
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+with respect to\s+(?:the\s+)?(.+?)\?",
    ]

    for p in pats:
        m = re.search(p, s, flags=re.I)
        if m:
            return clean_obj(m.group(1)), clean_obj(m.group(2))

    return "", ""


def pixel_values_to_pil(pixel_values, image_processor):
    x = pixel_values.detach().float().cpu()

    mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)

    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="output/controlledA_model_processed_images_crop")
    parser.add_argument("--out-json", default="output/controlledA_model_processed_manifest_crop.json")
    parser.add_argument("--fresh-limit", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    print("[LOAD MODEL]", args.model_name, args.method)
    wrapper, _ = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    image_processor = wrapper.processor.image_processor

    print("[LOAD DATASET RAW]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_default_collate)

    prompts, answers = sf.load_prompts(args.dataset, args.option)

    manifest = {}

    for sid, image in tqdm(iter_samples(loader), desc="dump model processed crop images"):
        if sid >= len(prompts):
            break
        if args.fresh_limit > 0 and sid >= args.fresh_limit:
            break

        prompt = prompts[sid]
        gold = answers[sid]
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        gold = str(gold).strip()

        obj1, obj2 = parse_objects_from_prompt(prompt)

        image = image.convert("RGB")
        proc = image_processor(images=image, return_tensors="pt")
        pixel_values = proc["pixel_values"][0]
        proc_img = pixel_values_to_pil(pixel_values, image_processor)

        out_path = os.path.join(args.out_dir, f"idx{sid:04d}_crop.png")
        proc_img.save(out_path)

        manifest[str(sid)] = {
            "sample_id": int(sid),
            "processed_image_path": out_path,
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,
            "prompt": prompt,
            "preprocess_mode": "model_processor_crop",
            "coordinate_system": f"model_processed_image_{proc_img.size[0]}x{proc_img.size[1]}",
            "width": proc_img.size[0],
            "height": proc_img.size[1],
        }

        if sid < 5:
            print("[CHECK]", sid, obj1, obj2, gold, out_path, proc_img.size)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[DONE]")
    print("[OUT DIR]", args.out_dir)
    print("[MANIFEST]", args.out_json)
    print("num:", len(manifest))


if __name__ == "__main__":
    main()
