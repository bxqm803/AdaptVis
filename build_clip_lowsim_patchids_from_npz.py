import os
import json
import argparse
import numpy as np


def matrix_to_patch_ids(mat, thr):
    ids = []
    h, w = mat.shape
    for r in range(h):
        for c in range(w):
            if float(mat[r, c]) < float(thr):
                ids.append(r * w + c)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-patches", type=int, default=1)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)

    sample_ids = data["sample_ids"]
    obj1_names = data["obj1_names"]
    obj2_names = data["obj2_names"]
    sim_obj1 = data["sim_obj1"]
    sim_obj2 = data["sim_obj2"]

    out = {}

    for i in range(len(sample_ids)):
        sid = int(sample_ids[i])
        m1 = sim_obj1[i]
        m2 = sim_obj2[i]

        obj1_patch_ids = matrix_to_patch_ids(m1, args.threshold)
        obj2_patch_ids = matrix_to_patch_ids(m2, args.threshold)

        patch_ids = sorted(set(obj1_patch_ids + obj2_patch_ids))

        if len(patch_ids) < args.min_patches:
            patch_ids = []

        out[str(sid)] = {
            "sample_id": sid,
            "obj1": str(obj1_names[i]),
            "obj2": str(obj2_names[i]),
            "clip_low_sim_threshold": float(args.threshold),
            "patch_side": int(m1.shape[0]),
            "obj1_patch_ids": [int(x) for x in obj1_patch_ids],
            "obj2_patch_ids": [int(x) for x in obj2_patch_ids],
            "patch_ids": [int(x) for x in patch_ids],
            "num_obj1_patch_ids": len(obj1_patch_ids),
            "num_obj2_patch_ids": len(obj2_patch_ids),
            "num_patch_ids": len(patch_ids),
            "obj1_min": float(m1.min()),
            "obj1_max": float(m1.max()),
            "obj1_mean": float(m1.mean()),
            "obj2_min": float(m2.min()),
            "obj2_max": float(m2.max()),
            "obj2_mean": float(m2.mean()),
        }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    nums = [v["num_patch_ids"] for v in out.values()]
    print("[DONE]", args.out_json)
    print("num samples:", len(out))
    print("threshold:", args.threshold)
    print("avg patches:", sum(nums) / max(len(nums), 1))
    print("min patches:", min(nums) if nums else 0)
    print("max patches:", max(nums) if nums else 0)


if __name__ == "__main__":
    main()
