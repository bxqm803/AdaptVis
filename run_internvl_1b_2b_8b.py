#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_internvl_1b_2b_8b.py

One launcher for:
  OpenGVLab/InternVL2_5-1B
  OpenGVLab/InternVL2_5-2B
  OpenGVLab/InternVL2_5-8B

It does not modify eval_internvl_last_causal_auto_v1.py.

For each model:
1) recursively searches --search-root for a matching directory containing BOTH
   vectors.npz and sample_split_and_generation.csv
2) verifies vectors.npz contains --direction-key
3) launches eval_internvl_last_causal_auto_v1.py with identical experiment args
4) writes results to:
   <output-root>/internvl25_1b_last_causal_auto_v1
   <output-root>/internvl25_2b_last_causal_auto_v1
   <output-root>/internvl25_8b_last_causal_auto_v1

Default is sequential execution for stable GPU memory use.

Example:
CUDA_VISIBLE_DEVICES=0 python run_internvl_1b_2b_8b.py \
  --search-root output \
  --device cuda:0 \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --annotation-json data/coco_qa_two_obj.json \
  --data-root data \
  --scan-layers all \
  --val-frac 0.25 \
  --top-k-single 5 \
  --window-lengths 2,3,4 \
  --scale 1.0 \
  --max-num-tiles 12 \
  --output-root output \
  --overwrite

Only inspect which vectors.npz would be used:
python run_internvl_1b_2b_8b.py --search-root output --find-only

If automatic matching is ambiguous:
  --dir-1b output/...
  --dir-2b output/...
  --dir-8b output/...

Run only a subset:
  --sizes 1b,8b
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


MODELS = {
    "1b": "OpenGVLab/InternVL2_5-1B",
    "2b": "OpenGVLab/InternVL2_5-2B",
    "8b": "OpenGVLab/InternVL2_5-8B",
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--experiment-script",
        default="eval_internvl_last_causal_auto_v1.py",
    )
    p.add_argument("--sizes", default="1b,2b,8b")
    p.add_argument("--search-root", default="output")
    p.add_argument("--dir-1b", default="")
    p.add_argument("--dir-2b", default="")
    p.add_argument("--dir-8b", default="")
    p.add_argument("--find-only", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    p.add_argument("--direction-key", default="residual")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--annotation-json", default="data/coco_qa_two_obj.json")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--use-flash-attn", action="store_true")
    p.add_argument("--scan-layers", default="all")
    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--min-val-per-relation", type=int, default=4)
    p.add_argument("--top-k-single", type=int, default=5)
    p.add_argument("--window-lengths", default="2,3,4")
    p.add_argument("--guide-window-lengths", default="1,2,3,4,5,6,7")
    p.add_argument(
        "--guide-train-controls",
        default="correct",
        choices=["correct", "all"],
    )
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--max-num-tiles", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-fit", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--output-root", default="output")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def normalize(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def inspect_npz(path, direction_key):
    try:
        with np.load(path, allow_pickle=True) as z:
            if direction_key not in z.files:
                return None

            arr = np.asarray(z[direction_key])
            n = (
                len(np.asarray(z["sample_index"]))
                if "sample_index" in z.files
                else None
            )

            return {
                "shape": tuple(arr.shape),
                "n": n,
            }
    except Exception:
        return None


def score_candidate(directory, size):
    s = normalize(directory)
    score = 0

    score += 50 if "internvl" in s else -100
    if "internvl25" in s:
        score += 20

    if size in s:
        score += 60

    for alias in [
        f"internvl{size}",
        f"internvl25{size}",
        f"internvl2_5{size}",
        f"internvl2.5{size}",
    ]:
        if normalize(alias) in s:
            score += 35

    for other in ["1b", "2b", "4b", "8b", "26b", "38b", "78b"]:
        if other != size and other in s:
            score -= 35

    if "qwen" in s:
        score -= 100
    if "llava" in s:
        score -= 100
    if "direction" in s:
        score += 5
    if "scan" in s:
        score += 2

    return score


def find_candidates(search_root, size, direction_key):
    search_root = Path(search_root)

    if not search_root.exists():
        raise FileNotFoundError(search_root)

    out = []

    for vectors_path in search_root.rglob("vectors.npz"):
        directory = vectors_path.parent
        split_path = directory / "sample_split_and_generation.csv"

        if not split_path.exists():
            continue

        info = inspect_npz(vectors_path, direction_key)

        if info is None:
            continue

        out.append({
            "dir": directory.resolve(),
            "score": score_candidate(directory, size),
            "shape": info["shape"],
            "n": info["n"],
        })

    out.sort(
        key=lambda x: (x["score"], str(x["dir"])),
        reverse=True,
    )
    return out


def validate_dir(path, direction_key):
    d = Path(path)
    vectors = d / "vectors.npz"
    split = d / "sample_split_and_generation.csv"

    if not vectors.exists():
        raise FileNotFoundError(vectors)
    if not split.exists():
        raise FileNotFoundError(split)

    info = inspect_npz(vectors, direction_key)

    if info is None:
        raise RuntimeError(
            f"{vectors} does not contain usable key={direction_key!r}"
        )

    return {
        "dir": d.resolve(),
        "score": 999999,
        "shape": info["shape"],
        "n": info["n"],
    }


def choose(size, args):
    explicit = getattr(args, f"dir_{size}")

    if explicit:
        return validate_dir(explicit, args.direction_key), []

    candidates = find_candidates(
        args.search_root,
        size,
        args.direction_key,
    )

    if not candidates:
        return None, []

    best_score = candidates[0]["score"]
    tied = [x for x in candidates if x["score"] == best_score]

    if len(tied) != 1:
        return None, tied

    return tied[0], candidates


def build_command(size, selected, args):
    outdir = (
        Path(args.output_root)
        / f"internvl25_{size}_last_causal_auto_v1"
    )

    cmd = [
        sys.executable,
        args.experiment_script,
        "--model-id", MODELS[size],
        "--direction-dir", str(selected["dir"]),
        "--direction-key", args.direction_key,
        "--prompt-jsonl", args.prompt_jsonl,
        "--annotation-json", args.annotation_json,
        "--data-root", args.data_root,
        "--device", args.device,
        "--dtype", args.dtype,
        "--scan-layers", args.scan_layers,
        "--template-filter", args.template_filter,
        "--val-frac", str(args.val_frac),
        "--min-val-per-relation", str(args.min_val_per_relation),
        "--top-k-single", str(args.top_k_single),
        "--window-lengths", args.window_lengths,
        "--guide-window-lengths", args.guide_window_lengths,
        "--guide-train-controls", args.guide_train_controls,
        "--scale", str(args.scale),
        "--gray-value", str(args.gray_value),
        "--input-size", str(args.input_size),
        "--max-num-tiles", str(args.max_num_tiles),
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
        "--output-dir", str(outdir),
    ]

    if args.use_flash_attn:
        cmd.append("--use-flash-attn")
    if args.max_fit is not None:
        cmd += ["--max-fit", str(args.max_fit)]
    if args.max_val is not None:
        cmd += ["--max-val", str(args.max_val)]
    if args.max_test is not None:
        cmd += ["--max-test", str(args.max_test)]
    if args.overwrite:
        cmd.append("--overwrite")

    return cmd, outdir


def main():
    args = parse_args()

    sizes = [
        x.strip().lower()
        for x in args.sizes.split(",")
        if x.strip()
    ]

    bad = [x for x in sizes if x not in MODELS]

    if bad:
        raise ValueError(
            f"unsupported sizes={bad}; valid={list(MODELS)}"
        )

    if not args.find_only and not Path(args.experiment_script).exists():
        raise FileNotFoundError(args.experiment_script)

    selected = {}
    failed_search = False

    # Search every model first, before launching anything.
    for size in sizes:
        best, candidates = choose(size, args)

        print("\n" + "=" * 130)
        print(f"{MODELS[size]} — direction bundle")
        print("=" * 130)

        if best is not None:
            selected[size] = best
            print(
                f"SELECTED score={best['score']} "
                f"N={best['n']} shape={best['shape']}"
            )
            print(best["dir"])
        elif candidates:
            failed_search = True
            print("AMBIGUOUS BEST MATCHES:")
            for c in candidates:
                print(
                    f"score={c['score']} N={c['n']} "
                    f"shape={c['shape']} {c['dir']}"
                )
            print(f"Specify --dir-{size} explicitly.")
        else:
            failed_search = True
            print("NO MATCH FOUND")
            print(f"Specify --dir-{size} explicitly.")

    if failed_search:
        print(
            "\nNo experiments launched because at least one requested model "
            "does not have a unique direction bundle."
        )
        raise SystemExit(2)

    if args.find_only:
        return

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selection_json = {
        size: {
            "model_id": MODELS[size],
            "direction_dir": str(selected[size]["dir"]),
            "N": selected[size]["n"],
            "shape": selected[size]["shape"],
        }
        for size in sizes
    }

    (
        output_root
        / "internvl_1b_2b_8b_launcher_selection.json"
    ).write_text(
        json.dumps(selection_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results = []

    for i, size in enumerate(sizes, 1):
        cmd, outdir = build_command(size, selected[size], args)

        print("\n" + "#" * 130)
        print(f"[{i}/{len(sizes)}] RUNNING {MODELS[size]}")
        print(f"direction_dir={selected[size]['dir']}")
        print(f"output_dir={outdir}")
        print("#" * 130)
        print(" ".join(shlex.quote(x) for x in cmd))
        print()

        completed = subprocess.run(cmd, check=False)

        results.append({
            "size": size,
            "model_id": MODELS[size],
            "returncode": completed.returncode,
            "direction_dir": str(selected[size]["dir"]),
            "output_dir": str(outdir),
        })

        if completed.returncode != 0 and not args.continue_on_error:
            break

    (
        output_root
        / "internvl_1b_2b_8b_launcher_results.json"
    ).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)

    for r in results:
        status = "OK" if r["returncode"] == 0 else f"FAILED({r['returncode']})"
        print(
            f"{r['size']:>2s} | {status:12s} | {r['output_dir']}"
        )

    if any(r["returncode"] != 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
