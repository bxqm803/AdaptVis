import os
import re
import csv
import json
import argparse
from PIL import Image
from tqdm import tqdm

import save_llava_hidden_similarity_features as sf


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


def get_image_id_from_record(rec):
    if isinstance(rec, list):
        return int(rec[0])

    if isinstance(rec, dict):
        for k in ["image_id", "img_id", "id", "vg_image_id"]:
            if k in rec:
                return int(rec[k])

    raise ValueError(f"Cannot parse image id from record: {rec}")


def find_image_path(image_dir, image_id):
    candidates = [
        os.path.join(image_dir, f"{image_id}.jpg"),
        os.path.join(image_dir, f"{image_id}.png"),
        os.path.join(image_dir, f"{image_id:012d}.jpg"),
        os.path.join(image_dir, f"{image_id:012d}.png"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0]


def preprocess_pad_resize(image, image_size=336, pad_color=(0, 0, 0)):
    image = image.convert("RGB")
    w, h = image.size

    side = max(w, h)
    canvas = Image.new("RGB", (side, side), pad_color)

    left = (side - w) // 2
    top = (side - h) // 2

    canvas.paste(image, (left, top))
    return canvas.resize((image_size, image_size), Image.BICUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", default="data/vg_qa_two_obj.json")
    parser.add_argument("--image-dir", default="data/vg_images")
    parser.add_argument("--dataset", default="VG_QA_two_obj")
    parser.add_argument("--option", default="six")

    parser.add_argument("--out-dir", default="output/vg_qa_two_obj_processed_images_pad")
    parser.add_argument("--out-json", default="output/vg_qa_two_obj_processed_manifest_pad.json")
    parser.add_argument("--out-csv", default="output/vg_qa_two_obj_processed_manifest_pad.csv")

    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--fresh-limit", type=int, default=-1)
    parser.add_argument("--idxs", default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    data = json.load(open(args.qa_json, "r", encoding="utf-8"))
    prompts, answers = sf.load_prompts(args.dataset, args.option)

    if args.idxs.strip():
        idxs = [int(x) for x in args.idxs.split(",") if x.strip()]
    else:
        idxs = list(range(len(data)))
        if args.fresh_limit > 0:
            idxs = idxs[:args.fresh_limit]

    manifest = {}
    csv_rows = []

    for idx in tqdm(idxs, desc="dump VG-QA processed images pad"):
        rec = data[idx]
        image_id = get_image_id_from_record(rec)

        prompt = prompts[idx] if idx < len(prompts) else ""
        gold = answers[idx] if idx < len(answers) else ""
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        gold = str(gold).strip()

        obj1, obj2 = parse_objects_from_prompt(prompt)

        raw_image_path = find_image_path(args.image_dir, image_id)
        if not os.path.exists(raw_image_path):
            print("[WARN] missing image:", raw_image_path)
            continue

        raw_image = Image.open(raw_image_path).convert("RGB")
        processed_image = preprocess_pad_resize(raw_image, image_size=args.image_size)

        out_name = f"idx{idx:04d}_img{image_id}_pad.png"
        processed_image_path = os.path.join(args.out_dir, out_name)
        processed_image.save(processed_image_path)

        item = {
            "sample_idx": int(idx),
            "image_id": int(image_id),
            "raw_image_path": raw_image_path,
            "processed_image_path": processed_image_path,
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,
            "prompt": prompt,
            "raw_record": rec,
            "processed_width": int(processed_image.size[0]),
            "processed_height": int(processed_image.size[1]),
            "preprocess_mode": "pad",
            "coordinate_system": f"processed_image_pad_{args.image_size}x{args.image_size}",
        }

        manifest[str(idx)] = item

        csv_rows.append({
            "sample_idx": idx,
            "image_id": image_id,
            "raw_image_path": raw_image_path,
            "processed_image_path": processed_image_path,
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,
            "prompt": prompt,
            "preprocess_mode": "pad",
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_idx",
            "image_id",
            "raw_image_path",
            "processed_image_path",
            "obj1",
            "obj2",
            "gold",
            "prompt",
            "preprocess_mode",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print("[DONE]")
    print("[OUT DIR]", args.out_dir)
    print("[JSON]", args.out_json)
    print("[CSV]", args.out_csv)
    print("num saved:", len(csv_rows))

    if csv_rows:
        print("[FIRST]", csv_rows[0])


if __name__ == "__main__":
    main()
