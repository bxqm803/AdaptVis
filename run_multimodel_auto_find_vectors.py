#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_multimodel_auto_find_vectors.py

Wrapper for:
    eval_multimodel_last_causal_direction_auto_layers_v1.py

What it does
------------
1. Recursively search for vectors.npz under --search-root (default: output).
2. Keep only candidates whose same directory also contains:
       sample_split_and_generation.csv
3. Score candidates against --model-id, e.g.
       Qwen/Qwen2.5-VL-3B-Instruct  -> qwen3b / qwen25-3b / qwen2.5-3b ...
       llava-hf/llava-1.5-7b-hf    -> llava7b / llava-7b / llava1.5 ...
4. Check that vectors.npz contains --direction-key (default: residual).
5. Automatically pass the selected directory as:
       --direction-dir <FOUND_DIR>
   to the original experiment script.
6. Then run the original script unchanged.

If multiple candidates have the same best score, the script prints them and exits
instead of guessing. You can then specify one with:
    --vectors-dir /path/to/the/correct/output_dir

Examples
--------

Qwen3B:
CUDA_VISIBLE_DEVICES=0 python run_multimodel_auto_find_vectors.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --search-root output \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --annotation-json data/coco_qa_two_obj.json \
  --data-root data \
  --device cuda:0 \
  --scan-layers all \
  --val-frac 0.25 \
  --top-k-single 5 \
  --window-lengths 2,3,4 \
  --scale 1.0 \
  --output-dir output/qwen3b_last_causal_auto_v1 \
  --overwrite

LLaVA7B:
CUDA_VISIBLE_DEVICES=0 python run_multimodel_auto_find_vectors.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --search-root output \
  --dtype float16 \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --annotation-json data/coco_qa_two_obj.json \
  --data-root data \
  --device cuda:0 \
  --scan-layers all \
  --val-frac 0.25 \
  --top-k-single 5 \
  --window-lengths 2,3,4 \
  --scale 1.0 \
  --output-dir output/llava7b_last_causal_auto_v1 \
  --overwrite

Only search / inspect candidates without launching experiment:
    python run_multimodel_auto_find_vectors.py \
      --model-id Qwen/Qwen2.5-VL-3B-Instruct \
      --search-root output \
      --find-only
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


ORIGINAL_SCRIPT = "eval_multimodel_last_causal_direction_auto_layers_v1.py"


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def model_aliases(model_id: str):
    low = model_id.lower()
    aliases = set()

    # Generic normalized forms.
    aliases.add(normalize(model_id))
    aliases.add(normalize(model_id.split("/")[-1]))

    # Qwen2.5-VL aliases.
    if "qwen" in low:
        aliases.update({
            "qwen",
            "qwen25",
            "qwen25vl",
            "qwen2.5",
            "qwen2.5vl",
        })

        m = re.search(r"(\d+(?:\.\d+)?)b", low)
        if m:
            size = m.group(1).replace(".", "")
            raw_size = m.group(1)

            aliases.update({
                f"qwen{raw_size}b",
                f"qwen{size}b",
                f"qwen25{raw_size}b",
                f"qwen25{size}b",
                f"qwen2.5{raw_size}b",
                f"qwen2.5vl{raw_size}b",
                f"qwen25vl{raw_size}b",
            })

    # LLaVA aliases.
    if "llava" in low:
        aliases.update({
            "llava",
            "llava15",
            "llava1.5",
            "llavanext",
            "llava16",
            "llava1.6",
        })

        m = re.search(r"(\d+(?:\.\d+)?)b", low)
        if m:
            raw_size = m.group(1)
            aliases.update({
                f"llava{raw_size}b",
                f"llava-{raw_size}b",
                f"llava15{raw_size}b",
                f"llava1.5{raw_size}b",
                f"llava16{raw_size}b",
                f"llava1.6{raw_size}b",
            })

    # Normalize everything once.
    aliases = {
        normalize(a)
        for a in aliases
        if a
    }

    # Avoid overly generic alias dominating scoring.
    aliases.discard("qwen")
    aliases.discard("llava")

    return sorted(
        aliases,
        key=len,
        reverse=True,
    )


def candidate_score(path: Path, model_id: str):
    """
    Score only from path/name heuristics.
    Higher is better.
    """
    s = normalize(str(path))
    aliases = model_aliases(model_id)

    score = 0
    matched = []

    for alias in aliases:
        if len(alias) < 4:
            continue

        if alias in s:
            # Prefer more specific aliases.
            value = min(30, 4 + len(alias))
            score += value
            matched.append(alias)

    low = model_id.lower()

    # Strong size match / mismatch checks.
    size_match = re.search(r"(\d+(?:\.\d+)?)b", low)

    if size_match:
        target_size = normalize(
            size_match.group(1)
            + "b"
        )

        if target_size in s:
            score += 25
            matched.append(
                f"size:{target_size}"
            )

        # Penalize obvious other common model sizes in path.
        for other in [
            "1b",
            "2b",
            "3b",
            "7b",
            "8b",
            "13b",
        ]:
            if (
                other != target_size
                and other in s
            ):
                score -= 20

    # Family.
    if "qwen" in low:
        if "qwen" in s:
            score += 20
        if "llava" in s:
            score -= 60

    if "llava" in low:
        if "llava" in s:
            score += 20
        if "qwen" in s:
            score -= 60

    return score, matched


def inspect_npz(path: Path, direction_key: str):
    try:
        with np.load(
            path,
            allow_pickle=True,
        ) as z:

            keys = list(
                z.files
            )

            has_key = (
                direction_key in keys
            )

            shape = None

            if has_key:
                shape = tuple(
                    np.asarray(
                        z[direction_key]
                    ).shape
                )

            n = None

            if "sample_index" in keys:
                n = len(
                    np.asarray(
                        z["sample_index"]
                    )
                )

            return {
                "ok": True,
                "keys": keys,
                "has_key": has_key,
                "shape": shape,
                "n": n,
                "error": "",
            }

    except Exception as exc:
        return {
            "ok": False,
            "keys": [],
            "has_key": False,
            "shape": None,
            "n": None,
            "error":
                f"{type(exc).__name__}: {exc}",
        }


def find_candidates(
    search_root: Path,
    model_id: str,
    direction_key: str,
):
    if not search_root.exists():
        raise FileNotFoundError(
            search_root
        )

    candidates = []

    for vectors_path in search_root.rglob(
        "vectors.npz"
    ):
        directory = vectors_path.parent

        split_path = (
            directory
            / "sample_split_and_generation.csv"
        )

        if not split_path.exists():
            continue

        info = inspect_npz(
            vectors_path,
            direction_key,
        )

        if (
            not info["ok"]
            or not info["has_key"]
        ):
            continue

        score, matched = candidate_score(
            directory,
            model_id,
        )

        candidates.append({
            "directory":
                directory.resolve(),
            "vectors":
                vectors_path.resolve(),
            "split":
                split_path.resolve(),
            "score":
                score,
            "matched":
                matched,
            "shape":
                info["shape"],
            "n":
                info["n"],
            "keys":
                info["keys"],
        })

    candidates.sort(
        key=lambda x: (
            x["score"],
            str(x["directory"]),
        ),
        reverse=True,
    )

    return candidates


def print_candidates(
    candidates,
    model_id,
):
    print(
        "\n"
        + "=" * 130
    )

    print(
        f"vectors.npz candidates for model: {model_id}"
    )

    print(
        "=" * 130
    )

    if not candidates:
        print(
            "No valid candidates found."
        )
        return

    for i, c in enumerate(
        candidates,
        1,
    ):
        print(
            f"[{i:02d}] score={c['score']:4d} | "
            f"N={c['n']} | shape={c['shape']}"
        )

        print(
            f"     dir: {c['directory']}"
        )

        print(
            f"     matched: "
            f"{','.join(c['matched']) if c['matched'] else '-'}"
        )


def choose_candidate(
    candidates,
):
    if not candidates:
        return None

    best_score = candidates[
        0
    ]["score"]

    best = [
        c
        for c in candidates
        if c[
            "score"
        ] == best_score
    ]

    if len(best) != 1:
        return None

    return best[0]


def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Find the correct vectors.npz directory, then invoke "
            f"{ORIGINAL_SCRIPT}."
        ),
    )

    # Wrapper-specific args.
    p.add_argument(
        "--search-root",
        default="output",
    )

    p.add_argument(
        "--vectors-dir",
        default="",
        help=(
            "Explicit directory containing vectors.npz. "
            "If supplied, automatic search is skipped."
        ),
    )

    p.add_argument(
        "--experiment-script",
        default=ORIGINAL_SCRIPT,
    )

    p.add_argument(
        "--find-only",
        action="store_true",
    )

    # Must know model-id and direction-key for candidate scoring/check.
    p.add_argument(
        "--model-id",
        required=True,
    )

    p.add_argument(
        "--direction-key",
        default="residual",
    )

    # Everything else is forwarded to the original experiment.
    args, remaining = p.parse_known_args()

    return args, remaining


def validate_explicit_dir(
    directory: Path,
    direction_key: str,
):
    vectors = directory / "vectors.npz"
    split = directory / "sample_split_and_generation.csv"

    if not vectors.exists():
        raise FileNotFoundError(
            vectors
        )

    if not split.exists():
        raise FileNotFoundError(
            split
        )

    info = inspect_npz(
        vectors,
        direction_key,
    )

    if not info["ok"]:
        raise RuntimeError(
            info["error"]
        )

    if not info["has_key"]:
        raise KeyError(
            f"{direction_key!r} not found in {vectors}; "
            f"keys={info['keys']}"
        )

    return {
        "directory":
            directory.resolve(),
        "vectors":
            vectors.resolve(),
        "split":
            split.resolve(),
        "score":
            9999,
        "matched":
            ["explicit"],
        "shape":
            info["shape"],
        "n":
            info["n"],
        "keys":
            info["keys"],
    }


def main():
    args, remaining = build_parser()

    if args.vectors_dir:
        selected = validate_explicit_dir(
            Path(args.vectors_dir),
            args.direction_key,
        )

        candidates = [
            selected
        ]

    else:
        candidates = find_candidates(
            Path(args.search_root),
            args.model_id,
            args.direction_key,
        )

        print_candidates(
            candidates,
            args.model_id,
        )

        selected = choose_candidate(
            candidates
        )

    if selected is None:
        print(
            "\nCould not choose a unique best vectors.npz automatically."
        )

        if candidates:
            print(
                "Use --vectors-dir with the correct candidate directory."
            )
        else:
            print(
                "Search returned no directory containing BOTH vectors.npz "
                "and sample_split_and_generation.csv with the requested key."
            )

        raise SystemExit(
            2
        )

    print(
        "\n"
        + "=" * 130
    )

    print(
        "SELECTED DIRECTION DIRECTORY"
    )

    print(
        "=" * 130
    )

    print(
        selected[
            "directory"
        ]
    )

    print(
        f"vectors shape={selected['shape']} | "
        f"N={selected['n']} | "
        f"direction_key={args.direction_key}"
    )

    if args.find_only:
        return

    experiment_script = Path(
        args.experiment_script
    )

    if not experiment_script.exists():
        raise FileNotFoundError(
            experiment_script
        )

    # Do not forward wrapper-only flags.
    cmd = [
        sys.executable,
        str(
            experiment_script
        ),
        "--model-id",
        args.model_id,
        "--direction-dir",
        str(
            selected[
                "directory"
            ]
        ),
        "--direction-key",
        args.direction_key,
    ]

    cmd.extend(
        remaining
    )

    print(
        "\n"
        + "=" * 130
    )

    print(
        "LAUNCHING ORIGINAL EXPERIMENT"
    )

    print(
        "=" * 130
    )

    print(
        " ".join(
            shlex.quote(
                x
            )
            for x in cmd
        )
    )

    print()

    completed = subprocess.run(
        cmd,
        check=False,
    )

    raise SystemExit(
        completed.returncode
    )


if __name__ == "__main__":
    main()
