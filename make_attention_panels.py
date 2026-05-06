#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import glob
import argparse
import random
import textwrap

import numpy as np
import matplotlib.pyplot as plt


def is_correct(gen, gold):
    """
    Match the repository's correctness logic as closely as possible.
    """
    gen = "" if gen is None else str(gen)
    gold = "" if gold is None else str(gold)

    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False
    return ok


def load_results_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for i, item in enumerate(data):
        prompt = item.get("Prompt", "")
        gen = item.get("Generation", "")
        gold = item.get("Golden", "")
        correct = is_correct(gen, gold)
        results.append({
            "sample_id": i,
            "prompt": prompt,
            "generation": gen,
            "golden": gold,
            "correct": correct,
        })
    return results


def find_first_matching_file(patterns):
    for p in patterns:
        files = sorted(glob.glob(p))
        if files:
            return files[0]
    return None


def parse_start_end_from_filename(path):
    """
    Parse ...start{N}_end{M}... from filename.
    """
    m = re.search(r"start(\d+)_end(\d+)", os.path.basename(path))
    if m is None:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_attention_vector(path):
    """
    Load saved attention/prob/logit numpy file and reduce to 1D vector over keys.

    Expected common shapes:
    - [batch, heads, key_len]
    - [heads, key_len]
    - [key_len]
    """
    arr = np.load(path)
    arr = np.asarray(arr)

    if arr.ndim == 3:
        # average over batch and heads
        vec = arr.mean(axis=(0, 1))
    elif arr.ndim == 2:
        vec = arr.mean(axis=0)
    elif arr.ndim == 1:
        vec = arr
    else:
        raise ValueError(f"Unsupported attention shape {arr.shape} for file {path}")

    return vec.astype(np.float32)


def extract_image_token_region(vec, file_path):
    """
    Extract image-token slice from full key vector using filename start/end.
    """
    start, end = parse_start_end_from_filename(file_path)
    if start is None or end is None:
        # fallback: use all
        img_vec = vec
    else:
        # inclusive end
        end = min(end, len(vec) - 1)
        start = max(0, start)
        img_vec = vec[start:end + 1]

    return img_vec


def reshape_to_grid(vec):
    """
    Reshape 1D image-token vector into a square-ish 2D grid.

    Common cases:
    - 576 -> 24x24
    - 577 -> trim 1 -> 24x24
    """
    n = len(vec)

    # exact square
    s = int(math.isqrt(n))
    if s * s == n:
        return vec.reshape(s, s)

    # trim one token if that makes square
    if n > 1:
        s = int(math.isqrt(n - 1))
        if s * s == (n - 1):
            return vec[:-1].reshape(s, s)

    # nearest next square with padding
    s = int(math.ceil(math.sqrt(n)))
    pad = s * s - n
    if pad > 0:
        vec = np.pad(vec, (0, pad), mode="constant", constant_values=np.nan)
    return vec.reshape(s, s)


def load_attention_grid(path):
    vec = load_attention_vector(path)
    img_vec = extract_image_token_region(vec, path)
    grid = reshape_to_grid(img_vec)
    return grid


def get_required_paths(sample_dir, layer, mode):
    """
    mode:
      baseline -> expects after_probs or after_logits, with fallbacks
      adapt_before -> expects before_probs or before_logits, with fallbacks
      adapt_after -> expects after_probs or after_logits, with fallbacks
    """
    if mode == "baseline":
        patterns = [
            os.path.join(sample_dir, f"after_probs_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"after_logits_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"before_probs_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"before_logits_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"diff_{layer}_start*_end*.npy"),
        ]
    elif mode == "adapt_before":
        patterns = [
            os.path.join(sample_dir, f"before_probs_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"before_logits_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"diff_{layer}_start*_end*.npy"),
        ]
    elif mode == "adapt_after":
        patterns = [
            os.path.join(sample_dir, f"after_probs_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"after_logits_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"before_probs_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"before_logits_layer{layer}_start*_end*.npy"),
            os.path.join(sample_dir, f"diff_{layer}_start*_end*.npy"),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return find_first_matching_file(patterns)


def make_panel(
    sample_id,
    sample_type,
    baseline_info,
    adapt_info,
    baseline_attn_root,
    adapt_attn_root,
    layer_a,
    layer_b,
    output_path,
):
    """
    Layout:
                 Baseline        Adapt-before      Adapt-after
    Layer A         1                2                 3
    Layer B         4                5                 6
    """
    baseline_sample_dir = os.path.join(baseline_attn_root, str(sample_id))
    adapt_sample_dir = os.path.join(adapt_attn_root, str(sample_id))

    file_map = {
        (0, 0): get_required_paths(baseline_sample_dir, layer_a, "baseline"),
        (0, 1): get_required_paths(adapt_sample_dir, layer_a, "adapt_before"),
        (0, 2): get_required_paths(adapt_sample_dir, layer_a, "adapt_after"),
        (1, 0): get_required_paths(baseline_sample_dir, layer_b, "baseline"),
        (1, 1): get_required_paths(adapt_sample_dir, layer_b, "adapt_before"),
        (1, 2): get_required_paths(adapt_sample_dir, layer_b, "adapt_after"),
    }

    # Load all six grids first
    grids = {}
    finite_values = []

    for key, path in file_map.items():
        if path is None:
            grids[key] = None
            continue
        grid = load_attention_grid(path)
        grids[key] = grid
        vals = grid[np.isfinite(grid)]
        if vals.size > 0:
            finite_values.append(vals)

    if finite_values:
        all_vals = np.concatenate(finite_values)
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    col_titles = ["Baseline (scal α=1.0)", "Adapt before", "Adapt after"]
    row_titles = [f"Layer {layer_a}", f"Layer {layer_b}"]

    for c in range(3):
        axes[0, c].set_title(col_titles[c], fontsize=11)

    for r in range(2):
        axes[r, 0].set_ylabel(row_titles[r], fontsize=11)

    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            grid = grids[(r, c)]
            if grid is None:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            path = file_map[(r, c)]
            ax.text(
                0.5,
                -0.08,
                os.path.basename(path),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                rotation=0,
            )

    # One shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Header text
    prompt_wrapped = textwrap.fill(str(baseline_info["prompt"]), width=100)
    base_pred = baseline_info["generation"]
    adapt_pred = adapt_info["generation"]
    gold = baseline_info["golden"]

    title = (
        f"sample_id={sample_id} | {sample_type}\n"
        f"Gold: {gold}\n"
        f"Baseline pred: {base_pred}\n"
        f"Adapt pred: {adapt_pred}\n"
        f"Prompt: {prompt_wrapped}"
    )

    fig.suptitle(title, fontsize=10, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=str, required=True,
                        help="Path to baseline-like results json, e.g. scaling_vis weight=1.0 result json")
    parser.add_argument("--adapt-results", type=str, required=True,
                        help="Path to adapt_vis results json")
    parser.add_argument("--baseline-attn-dir", type=str, required=True,
                        help="Attention root folder for baseline-like run, e.g. output/Controlled_A_scal_w1")
    parser.add_argument("--adapt-attn-dir", type=str, required=True,
                        help="Attention root folder for adapt run, e.g. output/Controlled_A_adapt_w05_w15_thr04")
    parser.add_argument("--out-dir", type=str, default="attention_panels_20",
                        help="Output directory for 20 combined images")
    parser.add_argument("--layer-a", type=int, default=17, help="First layer index, default 17")
    parser.add_argument("--layer-b", type=int, default=31, help="Second layer index, default 31")
    parser.add_argument("--num-each", type=int, default=10, help="How many from each category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    baseline = load_results_json(args.baseline_results)
    adapt = load_results_json(args.adapt_results)

    if len(baseline) != len(adapt):
        raise ValueError(
            f"Result length mismatch: baseline={len(baseline)} vs adapt={len(adapt)}"
        )

    right_to_wrong = []
    wrong_to_right = []

    for b, a in zip(baseline, adapt):
        sid = b["sample_id"]
        if sid != a["sample_id"]:
            raise ValueError("Sample ids do not align.")

        if b["correct"] and (not a["correct"]):
            right_to_wrong.append(sid)
        elif (not b["correct"]) and a["correct"]:
            wrong_to_right.append(sid)

    print(f"right->wrong candidates: {len(right_to_wrong)}")
    print(f"wrong->right candidates: {len(wrong_to_right)}")

    if len(right_to_wrong) == 0 and len(wrong_to_right) == 0:
        print("No changed samples found.")
        return

    selected_r2w = random.sample(right_to_wrong, min(args.num_each, len(right_to_wrong)))
    selected_w2r = random.sample(wrong_to_right, min(args.num_each, len(wrong_to_right)))

    selected = [("right_to_wrong", sid) for sid in selected_r2w] + \
               [("wrong_to_right", sid) for sid in selected_w2r]

    # save selection log
    with open(os.path.join(args.out_dir, "selected_samples.json"), "w", encoding="utf-8") as f:
        json.dump({
            "right_to_wrong": selected_r2w,
            "wrong_to_right": selected_w2r,
        }, f, ensure_ascii=False, indent=2)

    baseline_map = {x["sample_id"]: x for x in baseline}
    adapt_map = {x["sample_id"]: x for x in adapt}

    for sample_type, sid in selected:
        out_name = f"{sample_type}_sample{sid}.png"
        out_path = os.path.join(args.out_dir, out_name)

        make_panel(
            sample_id=sid,
            sample_type=sample_type,
            baseline_info=baseline_map[sid],
            adapt_info=adapt_map[sid],
            baseline_attn_root=args.baseline_attn_dir,
            adapt_attn_root=args.adapt_attn_dir,
            layer_a=args.layer_a,
            layer_b=args.layer_b,
            output_path=out_path,
        )
        print(f"saved: {out_path}")

    print(f"\nDone. Panels saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
