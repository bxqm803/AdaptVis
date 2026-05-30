import json
import csv
import os

records_path = "output/objectbox_negmean_all_l0_4_records.json"
patch_path = "output/groundingdino_object_patch_masks_by_sid.json"
out_csv = "output/objectbox_negmean_all_l0_4_effect_by_image.csv"

with open(records_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(patch_path, "r", encoding="utf-8") as f:
    patch_meta = json.load(f)

base = data["base"]

# 自动找 object pass 的 key
obj_keys = [k for k in data.keys() if k != "base"]
assert len(obj_keys) == 1, obj_keys
obj_key = obj_keys[0]
obj = data[obj_key]

base_by_sid = {int(r["sample_id"]): r for r in base}
obj_by_sid = {int(r["sample_id"]): r for r in obj}

rows = []
stats = {
    "wrong_to_correct": 0,
    "wrong_to_wrong": 0,
    "correct_to_wrong": 0,
    "correct_to_correct": 0,
}

for sid in sorted(base_by_sid.keys()):
    b = base_by_sid[sid]
    o = obj_by_sid[sid]

    b_corr = bool(b["correct"])
    o_corr = bool(o["correct"])

    if (not b_corr) and o_corr:
        effect = "wrong_to_correct"
    elif (not b_corr) and (not o_corr):
        effect = "wrong_to_wrong"
    elif b_corr and (not o_corr):
        effect = "correct_to_wrong"
    else:
        effect = "correct_to_correct"

    stats[effect] += 1

    meta = patch_meta.get(str(sid), {})

    rows.append({
        "sample_id": sid,
        "image_path": meta.get("image_path", ""),
        "prompt": meta.get("prompt", ""),
        "gold": b.get("gold", ""),
        "obj1": meta.get("obj1", ""),
        "obj2": meta.get("obj2", ""),
        "base_correct": b_corr,
        "negmean_correct": o_corr,
        "effect": effect,
        "base_confidence": b.get("confidence", ""),
        "selected_weight": o.get("selected_weight", ""),
        "num_object_patch_ids": o.get("num_object_patch_ids", ""),
        "base_generation": b.get("generation", ""),
        "negmean_generation": o.get("generation", ""),
    })

stats["net_gain"] = stats["wrong_to_correct"] - stats["correct_to_wrong"]
stats["base_acc"] = sum(bool(r["correct"]) for r in base) / len(base)
stats["negmean_acc"] = sum(bool(r["correct"]) for r in obj) / len(obj)
stats["total"] = len(rows)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)

fieldnames = [
    "sample_id",
    "image_path",
    "prompt",
    "gold",
    "obj1",
    "obj2",
    "base_correct",
    "negmean_correct",
    "effect",
    "base_confidence",
    "selected_weight",
    "num_object_patch_ids",
    "base_generation",
    "negmean_generation",
]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("[OBJECT KEY]", obj_key)
print("[CSV SAVED]", out_csv)
print(json.dumps(stats, indent=2, ensure_ascii=False))
