import os
import re
import csv
import glob
import json
import argparse
import numpy as np


def parse_file(path):
    """
    Expected filename:
      sid0007_layer00_img_logits.npz
    """
    name = os.path.basename(path)
    m = re.search(r"sid(\d+)_layer(\d+)_img_logits\.npz", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="output/negmean_patch_visuals",
        help="Root visualization folder containing img_logits/*.npz files.",
    )
    parser.add_argument(
        "--patch-json",
        default="output/groundingdino_object_patch_masks_by_sid.json",
        help="Optional patch json. Used to separately count object-box vs non-object patches.",
    )
    parser.add_argument(
        "--out-csv",
        default="output/negative_logit_stats_by_layer.csv",
    )
    parser.add_argument(
        "--out-summary",
        default="output/negative_logit_stats_summary.json",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "img_logits", "*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No img_logits npz files found under {args.root}. "
            f"Rerun visualization with --keep-raw."
        )

    patch_meta = {}
    if args.patch_json and os.path.exists(args.patch_json):
        with open(args.patch_json, "r", encoding="utf-8") as f:
            patch_meta = json.load(f)

    rows = []

    global_ori_neg = 0
    global_ori_total = 0
    global_edit_neg = 0
    global_edit_total = 0

    for path in files:
        sid, layer = parse_file(path)
        if sid is None:
            continue

        d = np.load(path)
        ori = d["ori_img_logits"][0]       # [num_heads, num_patches], usually [32, 576]
        edit = d["edited_img_logits"][0]   # [num_heads, num_patches]

        num_heads, num_patches = ori.shape
        total = ori.size

        ori_neg_count = int((ori < 0).sum())
        edit_neg_count = int((edit < 0).sum())

        ori_neg_ratio = ori_neg_count / total
        edit_neg_ratio = edit_neg_count / total

        # patch-level statistics
        ori_neg_frac_per_patch = (ori < 0).mean(axis=0)      # [576]
        edit_neg_frac_per_patch = (edit < 0).mean(axis=0)    # [576]

        # how many patches are negative in most heads
        ori_patch_neg_50 = int((ori_neg_frac_per_patch >= 0.5).sum())
        ori_patch_neg_80 = int((ori_neg_frac_per_patch >= 0.8).sum())
        ori_patch_neg_100 = int((ori_neg_frac_per_patch >= 1.0).sum())

        edit_patch_neg_50 = int((edit_neg_frac_per_patch >= 0.5).sum())
        edit_patch_neg_80 = int((edit_neg_frac_per_patch >= 0.8).sum())
        edit_patch_neg_100 = int((edit_neg_frac_per_patch >= 1.0).sum())

        # object-box vs background, if patch json exists
        obj_patch_ids = []
        if str(sid) in patch_meta:
            obj_patch_ids = [int(x) for x in patch_meta[str(sid)].get("patch_ids", [])]

        if obj_patch_ids:
            obj_patch_ids = [p for p in obj_patch_ids if 0 <= p < num_patches]
            bg_patch_ids = [p for p in range(num_patches) if p not in set(obj_patch_ids)]

            ori_obj = ori[:, obj_patch_ids]
            ori_bg = ori[:, bg_patch_ids]

            edit_obj = edit[:, obj_patch_ids]
            edit_bg = edit[:, bg_patch_ids]

            ori_obj_neg_ratio = float((ori_obj < 0).mean()) if ori_obj.size else None
            ori_bg_neg_ratio = float((ori_bg < 0).mean()) if ori_bg.size else None
            edit_obj_neg_ratio = float((edit_obj < 0).mean()) if edit_obj.size else None
            edit_bg_neg_ratio = float((edit_bg < 0).mean()) if edit_bg.size else None
        else:
            ori_obj_neg_ratio = None
            ori_bg_neg_ratio = None
            edit_obj_neg_ratio = None
            edit_bg_neg_ratio = None

        rows.append({
            "sample_id": sid,
            "layer": layer,
            "file": path,

            "num_heads": num_heads,
            "num_patches": num_patches,
            "total_head_patch_logits": total,

            "ori_neg_count": ori_neg_count,
            "ori_neg_ratio": ori_neg_ratio,
            "edit_neg_count": edit_neg_count,
            "edit_neg_ratio": edit_neg_ratio,

            "ori_patch_neg_frac_ge_0p5_count": ori_patch_neg_50,
            "ori_patch_neg_frac_ge_0p8_count": ori_patch_neg_80,
            "ori_patch_neg_frac_eq_1p0_count": ori_patch_neg_100,

            "edit_patch_neg_frac_ge_0p5_count": edit_patch_neg_50,
            "edit_patch_neg_frac_ge_0p8_count": edit_patch_neg_80,
            "edit_patch_neg_frac_eq_1p0_count": edit_patch_neg_100,

            "num_object_patches": len(obj_patch_ids),
            "ori_object_neg_ratio": ori_obj_neg_ratio,
            "ori_background_neg_ratio": ori_bg_neg_ratio,
            "edit_object_neg_ratio": edit_obj_neg_ratio,
            "edit_background_neg_ratio": edit_bg_neg_ratio,
        })

        global_ori_neg += ori_neg_count
        global_ori_total += total
        global_edit_neg += edit_neg_count
        global_edit_total += total

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary by layer
    by_layer = {}
    for r in rows:
        layer = int(r["layer"])
        if layer not in by_layer:
            by_layer[layer] = {
                "num_files": 0,
                "ori_neg_count": 0,
                "ori_total": 0,
                "edit_neg_count": 0,
                "edit_total": 0,
            }
        by_layer[layer]["num_files"] += 1
        by_layer[layer]["ori_neg_count"] += int(r["ori_neg_count"])
        by_layer[layer]["ori_total"] += int(r["total_head_patch_logits"])
        by_layer[layer]["edit_neg_count"] += int(r["edit_neg_count"])
        by_layer[layer]["edit_total"] += int(r["total_head_patch_logits"])

    for layer, s in by_layer.items():
        s["ori_neg_ratio"] = s["ori_neg_count"] / max(s["ori_total"], 1)
        s["edit_neg_ratio"] = s["edit_neg_count"] / max(s["edit_total"], 1)

    summary = {
        "num_npz_files": len(rows),
        "global_ori_neg_count": global_ori_neg,
        "global_ori_total": global_ori_total,
        "global_ori_neg_ratio": global_ori_neg / max(global_ori_total, 1),
        "global_edit_neg_count": global_edit_neg,
        "global_edit_total": global_edit_total,
        "global_edit_neg_ratio": global_edit_neg / max(global_edit_total, 1),
        "by_layer": by_layer,
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[CSV SAVED]", args.out_csv)
    print("[SUMMARY SAVED]", args.out_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
