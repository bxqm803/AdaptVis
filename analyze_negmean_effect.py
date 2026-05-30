import argparse
import json
import csv
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_object_key(data):
    keys = [k for k in data.keys() if k != "base"]
    if len(keys) == 1:
        return keys[0]

    # Prefer negonly_mean_img if present.
    for k in keys:
        if "negonly_mean" in k:
            return k

    # Otherwise prefer object_box key.
    for k in keys:
        if k.startswith("object_box"):
            return k

    raise ValueError(f"Cannot infer object/intervention key from keys={list(data.keys())}")


def load_patch_meta(path):
    if not path:
        return {}

    data = load_json(path)
    if isinstance(data, list):
        data = {str(int(x["sample_id"])): x for x in data}

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--patch-json", default="")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--active-weight", type=float, default=0.5)
    parser.add_argument("--out-csv", default="output/negonly_mean_effect_by_sample.csv")
    parser.add_argument("--out-summary", default="output/negonly_mean_effect_summary.json")
    args = parser.parse_args()

    data = load_json(args.records_json)
    obj_key = get_object_key(data)

    base_records = data["base"]
    obj_records = data[obj_key]

    patch_meta = load_patch_meta(args.patch_json)

    base_by_sid = {int(r["sample_id"]): r for r in base_records}
    obj_by_sid = {int(r["sample_id"]): r for r in obj_records}

    rows = []

    summary = {
        "records_json": args.records_json,
        "object_key": obj_key,
        "threshold": args.threshold,
        "active_weight": args.active_weight,

        "total": 0,

        # selected by AdaptVis confidence rule
        "selected_active_total": 0,
        "selected_inactive_total": 0,

        # effect groups among active samples
        "active_wrong_to_correct": 0,
        "active_wrong_to_wrong": 0,
        "active_correct_to_correct": 0,
        "active_correct_to_wrong": 0,
        "active_net_gain": 0,

        # all samples
        "all_wrong_to_correct": 0,
        "all_wrong_to_wrong": 0,
        "all_correct_to_correct": 0,
        "all_correct_to_wrong": 0,
        "all_net_gain": 0,
    }

    for sid in sorted(base_by_sid.keys()):
        if sid not in obj_by_sid:
            continue

        b = base_by_sid[sid]
        o = obj_by_sid[sid]

        base_correct = bool(b["correct"])
        obj_correct = bool(o["correct"])
        base_conf = float(b.get("confidence", 0.0))
        selected_weight = float(o.get("selected_weight", 1.0))

        # Whether this sample belongs to the low-confidence branch.
        # This is the branch that usually applies weight1=0.5.
        selected_by_prob = base_conf < args.threshold

        # Whether the saved selected_weight equals active_weight.
        # Useful if you used --weight1 0.5 --weight2 1.0 or --weight2 1.5.
        selected_active_weight = abs(selected_weight - args.active_weight) <= 1e-6

        if (not base_correct) and obj_correct:
            effect = "wrong_to_correct"
            effective = True
        elif base_correct and (not obj_correct):
            effect = "correct_to_wrong"
            effective = False
        elif base_correct and obj_correct:
            effect = "correct_to_correct"
            effective = False
        else:
            effect = "wrong_to_wrong"
            effective = False

        summary["total"] += 1

        if selected_by_prob:
            summary["selected_active_total"] += 1
        else:
            summary["selected_inactive_total"] += 1

        if effect == "wrong_to_correct":
            summary["all_wrong_to_correct"] += 1
        elif effect == "correct_to_wrong":
            summary["all_correct_to_wrong"] += 1
        elif effect == "correct_to_correct":
            summary["all_correct_to_correct"] += 1
        elif effect == "wrong_to_wrong":
            summary["all_wrong_to_wrong"] += 1

        # Main statistics: only count samples selected by probability rule.
        if selected_by_prob:
            if effect == "wrong_to_correct":
                summary["active_wrong_to_correct"] += 1
            elif effect == "correct_to_wrong":
                summary["active_correct_to_wrong"] += 1
            elif effect == "correct_to_correct":
                summary["active_correct_to_correct"] += 1
            elif effect == "wrong_to_wrong":
                summary["active_wrong_to_wrong"] += 1

        meta = patch_meta.get(str(sid), {})

        rows.append({
            "sample_id": sid,
            "image_path": meta.get("image_path", ""),
            "prompt": meta.get("prompt", ""),
            "gold": b.get("gold", ""),
            "obj1": meta.get("obj1", ""),
            "obj2": meta.get("obj2", ""),

            "base_correct": base_correct,
            "negmean_correct": obj_correct,
            "effect": effect,
            "effective_wrong_to_correct": effective,

            "base_confidence": base_conf,
            "threshold": args.threshold,
            "selected_by_probability": selected_by_prob,
            "selected_weight": selected_weight,
            "selected_active_weight": selected_active_weight,

            "num_object_patch_ids": o.get("num_object_patch_ids", 0),
            "base_generation": b.get("generation", ""),
            "negmean_generation": o.get("generation", ""),
        })

    summary["all_net_gain"] = summary["all_wrong_to_correct"] - summary["all_correct_to_wrong"]
    summary["active_net_gain"] = (
        summary["active_wrong_to_correct"] - summary["active_correct_to_wrong"]
    )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
        "effective_wrong_to_correct",
        "base_confidence",
        "threshold",
        "selected_by_probability",
        "selected_weight",
        "selected_active_weight",
        "num_object_patch_ids",
        "base_generation",
        "negmean_generation",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_summary_dir = os.path.dirname(args.out_summary)
    if out_summary_dir:
        os.makedirs(out_summary_dir, exist_ok=True)

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OBJECT KEY]", obj_key)
    print("[CSV SAVED]", args.out_csv)
    print("[SUMMARY SAVED]", args.out_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
