import os
import re
import csv
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm

from model_zoo import get_model


def norm_text(x):
    return str(x).strip()


def contains_obj(text, obj):
    text = str(text).lower()
    obj = str(obj).lower().strip()
    if not obj:
        return False
    return obj in text


def parse_relation(text):
    t = str(text).lower()

    # avoid "not left" being counted too aggressively is hard here;
    # this is only a rough diagnostic parser.
    hits = []
    for rel in ["left", "right", "above", "below"]:
        if rel in t:
            hits.append(rel.capitalize())

    # extra phrases
    if "on top of" in t or "over" in t:
        hits.append("Above")
    if "under" in t or "beneath" in t:
        hits.append("Below")

    # keep first unique relation
    seen = []
    for h in hits:
        if h not in seen:
            seen.append(h)

    return seen[0] if seen else ""


def build_object_prompt():
    return (
        "<image>\n"
        "USER: What objects are present in the image? Describe the visible objects briefly.\n"
        "ASSISTANT:"
    )


def build_relation_prompt(obj1, obj2):
    # No "Answer with left/right/above/below" here.
    return (
        "<image>\n"
        f"USER: Where is the {obj1} in relation to the {obj2}?\n"
        "ASSISTANT:"
    )


def generate_answer(wrapper, image, prompt, device, max_length, max_new_tokens):
    single_input = wrapper.processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    ).to(device)

    prompt_len = len(single_input["input_ids"][-1])

    with torch.no_grad():
        output = wrapper.model.generate(
            **single_input,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    seq = output.sequences if hasattr(output, "sequences") else output["sequences"]
    gen = wrapper.processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True)

    # first-step max probability
    scores = output.scores if hasattr(output, "scores") else output.get("scores", None)
    if scores is not None and len(scores) > 0:
        prob = torch.nn.functional.softmax(scores[0], dim=-1)
        conf = float(torch.max(prob[0]).detach().float().cpu())
    else:
        conf = 0.0

    return gen, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", default="output/coco_qa_two_obj_processed_manifest_pad.json")
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--fresh-limit", type=int, default=-1)
    parser.add_argument("--out-csv", default="output/coco_processed_probe_object_relation.csv")
    parser.add_argument("--out-json", default="output/coco_processed_probe_object_relation.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    data = json.load(open(args.manifest_json, "r", encoding="utf-8"))
    records = list(data.values())
    records = sorted(records, key=lambda r: int(r.get("sample_idx", r.get("sample_id", 0))))

    if args.fresh_limit > 0:
        records = records[:args.fresh_limit]

    print("[LOAD MODEL]", args.model_name, args.method)
    wrapper, _ = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    rows = []
    json_out = {}

    for r in tqdm(records, desc="probe processed images"):
        idx = int(r.get("sample_idx", r.get("sample_id", len(rows))))
        image_id = int(r.get("image_id", -1))

        img_path = r.get("processed_image_path", "")
        obj1 = norm_text(r.get("obj1", ""))
        obj2 = norm_text(r.get("obj2", ""))
        gold = norm_text(r.get("gold", ""))

        if not img_path or not os.path.exists(img_path):
            print("[WARN] missing image:", img_path)
            continue

        image = Image.open(img_path).convert("RGB")
        image.filename = img_path

        object_prompt = build_object_prompt()
        relation_prompt = build_relation_prompt(obj1, obj2)

        object_gen, object_conf = generate_answer(
            wrapper, image, object_prompt, args.device, args.max_length, args.max_new_tokens
        )

        relation_gen, relation_conf = generate_answer(
            wrapper, image, relation_prompt, args.device, args.max_length, args.max_new_tokens
        )

        pred_relation = parse_relation(relation_gen)
        relation_correct = pred_relation.lower() == gold.lower()

        obj1_in_desc = contains_obj(object_gen, obj1)
        obj2_in_desc = contains_obj(object_gen, obj2)
        both_in_desc = obj1_in_desc and obj2_in_desc

        obj1_in_relation = contains_obj(relation_gen, obj1)
        obj2_in_relation = contains_obj(relation_gen, obj2)

        row = {
            "sample_idx": idx,
            "image_id": image_id,
            "processed_image_path": img_path,
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,

            "object_prompt": object_prompt,
            "object_generation": object_gen,
            "object_conf": object_conf,
            "obj1_in_object_desc": obj1_in_desc,
            "obj2_in_object_desc": obj2_in_desc,
            "both_objects_in_desc": both_in_desc,

            "relation_prompt": relation_prompt,
            "relation_generation": relation_gen,
            "relation_conf": relation_conf,
            "pred_relation": pred_relation,
            "relation_correct": relation_correct,
            "obj1_in_relation_answer": obj1_in_relation,
            "obj2_in_relation_answer": obj2_in_relation,
        }

        rows.append(row)
        json_out[str(idx)] = row

    fieldnames = [
        "sample_idx",
        "image_id",
        "processed_image_path",
        "obj1",
        "obj2",
        "gold",
        "object_prompt",
        "object_generation",
        "object_conf",
        "obj1_in_object_desc",
        "obj2_in_object_desc",
        "both_objects_in_desc",
        "relation_prompt",
        "relation_generation",
        "relation_conf",
        "pred_relation",
        "relation_correct",
        "obj1_in_relation_answer",
        "obj2_in_relation_answer",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    n = len(rows)
    both_seen = sum(int(r["both_objects_in_desc"]) for r in rows)
    rel_acc = sum(int(r["relation_correct"]) for r in rows) / max(n, 1)

    print("[DONE]")
    print("[CSV]", args.out_csv)
    print("[JSON]", args.out_json)
    print("num:", n)
    print("both objects mentioned in object description:", both_seen, "/", n)
    print("free-form relation acc:", rel_acc)


if __name__ == "__main__":
    main()
