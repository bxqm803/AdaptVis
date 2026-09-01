#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_crossdataset_late_causal_qwen25_v1.py

Replicate the already-discovered Qwen2.5-VL late last-token causal steering
mechanism on two new spatial datasets without repeating the COCO all-layer search.

Reused model-specific sites
---------------------------
Qwen2.5-VL-3B:
    actuator layers = L32,L33,L34,L35
    middle residual-Direction guide = L19

Qwen2.5-VL-7B:
    actuator layers = L25,L26,L27
    middle residual-Direction guide = L16

Datasets
--------
controlled_a:
    data/controlled_images_dataset.json
    labels: left, right, on, under
    images are read from each record's image_path.
    Split is by unordered object pair, so the four layouts of one pair do not
    cross TRAIN/TEST.

vg2:
    Uses the repo-native four-option VG2 files already used by
    analyze_vg2_centroid_generation_step1_v1.py:

      data/vg_qa_two_obj_four_options.json
      prompts/VG_QA_two_obj_with_answer_four_options.jsonl
      data/vg/images/

    The data JSON supplies image_id; the prompt JSONL is authoritative for
    subject/reference and the four-way GT relation.

Core method
-----------
TRAIN, at each known actuator layer l:

    delta_i,l = h_last(real)_i,l - h_last(gray)_i,l

    mu_r,l = mean(delta_i,l | relation=r)
    mu_global,l = balanced mean_r(mu_r,l)
    s_r,l = mu_r,l - mu_global,l

TEST oracle:
    h_last,l <- h_last,l + s_GT,l

TEST non-oracle guide:
    at a known middle layer, compute the object-pair residual

      q_i = [(h_sub-h_ref)_real] - [(h_sub-h_ref)_gray]

    classify q_i by TRAIN relation prototypes, then use that prediction to
    choose which late causal direction to add.

This script reports actual model.generate() results:
    baseline
    oracle_all_add
    guide_all_add
    guide_conflict_add
    guide_all_contrast
    guide_conflict_contrast

The goal here is replication on new datasets, not another layer search.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


EPS = 1e-12


# =============================================================================
# CLI / model presets
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--model-id",
        required=True,
        choices=[
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
    )

    p.add_argument(
        "--dataset",
        required=True,
        choices=["controlled_a", "vg2"],
    )

    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--controlled-json",
        default="data/controlled_images_dataset.json",
    )

    p.add_argument(
        "--vg-json",
        default="data/vg_qa_two_obj_four_options.json",
        help=(
            "Repo-native filtered VG2 data JSON. This is aligned one-to-one "
            "with --vg-prompt-jsonl and each row starts with image_id."
        ),
    )

    p.add_argument(
        "--vg-prompt-jsonl",
        default="prompts/VG_QA_two_obj_with_answer_four_options.jsonl",
        help=(
            "Four-option VG2 prompt JSONL. If this file is missing, the script "
            "reconstructs it in memory from the existing six-option VG prompt "
            "and the filtered four-option data JSON."
        ),
    )

    p.add_argument(
        "--vg-six-prompt-jsonl",
        default="prompts/VG_QA_two_obj_with_answer_six_options.jsonl",
        help="Existing six-option VG2 prompt JSONL used for reconstruction.",
    )

    p.add_argument(
        "--vg-original-json",
        default="data/vg_qa_two_obj.json",
        help=(
            "Original unfiltered VG2 data JSON. Used to recover the original "
            "prompt id for each row in the filtered four-option data."
        ),
    )

    p.add_argument(
        "--vg-index-map",
        default="data/vg_qa_two_obj_four_options_index_map.json",
        help=(
            "Optional filtered->original index map. If its schema is not "
            "recognized, exact row matching against --vg-original-json is used."
        ),
    )

    p.add_argument(
        "--vg-image-root",
        default="auto",
        help=(
            "Visual Genome image root. 'auto' searches common local layouts "
            "under --data-root and verifies them against actual VG image ids."
        ),
    )

    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )

    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.30,
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def model_preset(model_id):
    if "3B" in model_id:
        return {
            "actuator_layers": [32, 33, 34, 35],
            "guide_layers": [19],
        }

    if "7B" in model_id:
        return {
            "actuator_layers": [25, 26, 27],
            "guide_layers": [16],
        }

    raise ValueError(model_id)


def dataset_spec(dataset):
    if dataset == "controlled_a":
        return {
            "labels": ("left", "right", "on", "under"),
            "answer_words": ("left", "right", "on", "under"),
        }

    if dataset == "vg2":
        return {
            "labels": ("left", "right"),
            "answer_words": ("left", "right"),
        }

    raise ValueError(dataset)


# =============================================================================
# generic utilities
# =============================================================================

def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields, seen = [], set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    vals = []

    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue

        if math.isfinite(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else float("nan")


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def auto_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def normalize_spaces(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def strip_terminal(s):
    s = normalize_spaces(s)
    s = re.sub(r"[\s\.,;:!?]+$", "", s)
    return s.strip()


def remove_article(s):
    return re.sub(
        r"^(?:the|a|an)\s+",
        "",
        strip_terminal(s),
        flags=re.I,
    ).strip()


# =============================================================================
# relation normalization
# =============================================================================

CONTROLLED_REL_ALIASES = {
    "left": "left",
    "left_of": "left",
    "left of": "left",
    "right": "right",
    "right_of": "right",
    "right of": "right",
    "on": "on",
    "on top": "on",
    "on top of": "on",
    "above": "on",
    "under": "under",
    "below": "under",
    "beneath": "under",
}

VG_REL_ALIASES = {
    "left": "left",
    "left_of": "left",
    "left of": "left",
    "right": "right",
    "right_of": "right",
    "right of": "right",
    "above": "above",
    "on": "above",
    "on top": "above",
    "on top of": "above",
    "top": "above",
    "top of": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "bottom": "below",
    "bottom of": "below",
}


def normalize_relation(value, dataset):
    x = str(value).strip().lower()
    x = x.replace("-", " ")
    x = normalize_spaces(x)

    aliases = (
        CONTROLLED_REL_ALIASES
        if dataset == "controlled_a"
        else VG_REL_ALIASES
    )

    if x in aliases:
        return aliases[x]

    # phrase fallback
    if re.search(r"\bleft(?:\s+of)?\b", x):
        return "left"

    if re.search(r"\bright(?:\s+of)?\b", x):
        return "right"

    if dataset == "controlled_a":
        if re.search(r"\b(?:on|above|top|over)\b", x):
            return "on"

        if re.search(r"\b(?:under|below|beneath|bottom)\b", x):
            return "under"

    else:
        if re.search(r"\b(?:above|on top of|top of|over)\b", x):
            return "above"

        if re.search(r"\b(?:below|under|beneath|bottom of)\b", x):
            return "below"

    return None


def parse_pred(text, dataset):
    s = str(text).lower()

    hits = []

    candidates = (
        [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
            ("on", r"\bon\b"),
            ("on", r"\babove\b"),
            ("under", r"\bunder(?:neath)?\b"),
            ("under", r"\bbelow\b"),
        ]
        if dataset == "controlled_a"
        else [
            ("left", r"\bleft\b"),
            ("right", r"\bright\b"),
        ]
    )

    for relation, pattern in candidates:
        m = re.search(pattern, s)

        if m:
            hits.append((m.start(), relation))

    if not hits:
        return None

    return sorted(hits)[0][1]


# =============================================================================
# Controlled A loader
# =============================================================================

CONTROLLED_FILENAME_PATTERNS = [
    ("left", "_left_of_"),
    ("right", "_right_of_"),
    ("on", "_on_"),
    ("under", "_under_"),
]


def parse_controlled_filename(image_path):
    name = Path(image_path).stem

    for relation, marker in CONTROLLED_FILENAME_PATTERNS:
        if marker in name:
            subject, reference = name.split(marker, 1)

            return (
                remove_article(subject.replace("_", " ")),
                remove_article(reference.replace("_", " ")),
                relation,
            )

    return None, None, None


def parse_caption_relation(caption, dataset):
    text = strip_terminal(caption)

    patterns = (
        [
            ("left", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(.+)$"),
            ("right", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(.+)$"),
            ("on", r"^(.+?)\s+(?:is\s+)?on(?:\s+top\s+of)?\s+(.+)$"),
            ("under", r"^(.+?)\s+(?:is\s+)?(?:under|below|beneath)\s+(.+)$"),
        ]
        if dataset == "controlled_a"
        else [
            ("left", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(.+)$"),
            ("right", r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(.+)$"),
            ("above", r"^(.+?)\s+(?:is\s+)?(?:above|over|on\s+top\s+of|at\s+the\s+top\s+of)\s+(.+)$"),
            ("below", r"^(.+?)\s+(?:is\s+)?(?:below|under|beneath|at\s+the\s+bottom\s+of)\s+(.+)$"),
        ]
    )

    for relation, pattern in patterns:
        m = re.match(pattern, text, flags=re.I)

        if m:
            return (
                remove_article(m.group(1)),
                remove_article(m.group(2)),
                relation,
            )

    return None, None, None


def load_controlled_a(args):
    path = Path(args.controlled_json)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for i, row in enumerate(data):
        image_path = str(row.get("image_path", ""))

        if not image_path:
            continue

        subject, reference, relation = parse_controlled_filename(
            image_path
        )

        if subject is None:
            captions = row.get("caption_options", [])

            if captions:
                subject, reference, relation = parse_caption_relation(
                    captions[0],
                    "controlled_a",
                )

        if (
            subject is None
            or reference is None
            or relation is None
        ):
            continue

        candidate = Path(image_path)

        if not candidate.exists():
            candidate2 = Path(args.data_root) / image_path

            if candidate2.exists():
                candidate = candidate2

        records.append({
            "sid": i,
            "dataset": "controlled_a",
            "image_path": str(candidate),
            "subject": subject,
            "reference": reference,
            "relation": relation,
            "pair_group": " || ".join(
                sorted(
                    [
                        subject.lower(),
                        reference.lower(),
                    ]
                )
            ),
        })

    print(
        f"[Controlled_A] loaded={len(records)} | "
        f"relations={dict(Counter(r['relation'] for r in records))}"
    )

    return records


# =============================================================================
# VG2 loader -- exact repo-native four-option schema
# =============================================================================

VG_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def extract_standard_user_text(raw_question):
    """
    Match the existing repo's VG2 loader:
    remove a leading <image>, then extract the USER text if the stored prompt
    contains USER:/ASSISTANT: wrappers.
    """
    text = str(raw_question).strip()

    text = re.sub(
        r"^\s*<image>\s*",
        "",
        text,
        flags=re.I,
    )

    match = re.search(
        r"\bUSER\s*:\s*(.*?)(?:\s*\bASSISTANT\s*:|\Z)",
        text,
        flags=re.I | re.S,
    )

    if match:
        text = match.group(1)

    return text.strip()


VG_STANDARD_OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?"
    r"\s*Answer\s+with",
    flags=re.I | re.S,
)


def parse_vg_standard_objects(question_text):
    compact = normalize_spaces(question_text)

    match = VG_STANDARD_OBJECT_RE.search(
        compact
    )

    if not match:
        raise ValueError(
            "Could not parse subject/reference from standard VG2 question: "
            f"{compact!r}"
        )

    subject = match.group(1).strip()
    reference = match.group(2).strip()

    if not subject or not reference:
        raise ValueError(
            f"Empty subject/reference in VG2 question: {compact!r}"
        )

    return subject, reference


def standard_answer_value(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None

    return value


def load_jsonl_by_id(path):
    path = Path(path)

    rows = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_no, line in enumerate(
            handle,
            1,
        ):
            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            if "id" not in row:
                raise ValueError(
                    f"{path}:{line_no} has no id field"
                )

            sid = int(
                row["id"]
            )

            if sid in rows:
                raise ValueError(
                    f"Duplicate id={sid} in {path}"
                )

            rows[
                sid
            ] = row

    return rows


def canonical_json_row(row):
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def infer_old_id_from_map_entry(entry):
    """
    Accept several likely local index-map schemas. We intentionally avoid
    guessing from arbitrary integer fields: only explicit source/original names
    are considered.
    """
    if not isinstance(
        entry,
        dict,
    ):
        return None

    for key in (
        "old_id",
        "original_id",
        "orig_id",
        "source_id",
        "old_index",
        "original_index",
        "orig_index",
        "source_index",
    ):
        if key in entry:
            try:
                return int(
                    entry[key]
                )
            except Exception:
                return None

    return None


def reconstruct_vg_four_option_prompt_rows(
    filtered_data,
    args,
):
    """
    Reconstruct the missing
      prompts/VG_QA_two_obj_with_answer_four_options.jsonl

    from files that are actually present in this checkout:

      data/vg_qa_two_obj_four_options.json
      data/vg_qa_two_obj.json
      data/vg_qa_two_obj_four_options_index_map.json   (optional)
      prompts/VG_QA_two_obj_with_answer_six_options.jsonl

    The filtered four-option data is a subset of the original VG2 data.
    We recover each filtered row's original index, then reuse the corresponding
    six-option question/answer and remove front/behind from the answer choices.
    """
    six_prompt_path = Path(
        args.vg_six_prompt_jsonl
    )

    original_data_path = Path(
        args.vg_original_json
    )

    index_map_path = Path(
        args.vg_index_map
    )

    if not six_prompt_path.exists():
        raise FileNotFoundError(
            f"Missing reconstruction source: {six_prompt_path}"
        )

    if not original_data_path.exists():
        raise FileNotFoundError(
            f"Missing reconstruction source: {original_data_path}"
        )

    six_prompts = load_jsonl_by_id(
        six_prompt_path
    )

    with original_data_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        original_data = json.load(
            handle
        )

    if not isinstance(
        original_data,
        list,
    ):
        raise TypeError(
            f"{original_data_path} must contain a top-level list"
        )

    # --------------------------------------------------------------
    # Preferred: use explicit local index map when its schema is recognized.
    # --------------------------------------------------------------
    old_ids = [
        None
        for _ in filtered_data
    ]

    used_map = False

    if index_map_path.exists():
        try:
            with index_map_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                index_map = json.load(
                    handle
                )

            if (
                isinstance(
                    index_map,
                    list,
                )
                and len(
                    index_map
                )
                == len(
                    filtered_data
                )
            ):
                candidate_old_ids = []

                for new_id, entry in enumerate(
                    index_map
                ):
                    old_id = infer_old_id_from_map_entry(
                        entry
                    )

                    if old_id is None:
                        candidate_old_ids = []
                        break

                    # If new_id exists, verify it rather than assuming order.
                    if (
                        isinstance(
                            entry,
                            dict,
                        )
                        and "new_id" in entry
                        and int(
                            entry[
                                "new_id"
                            ]
                        )
                        != new_id
                    ):
                        candidate_old_ids = []
                        break

                    candidate_old_ids.append(
                        old_id
                    )

                if len(
                    candidate_old_ids
                ) == len(
                    filtered_data
                ):
                    old_ids = candidate_old_ids
                    used_map = True

                    print(
                        f"[VG2 reconstruction] using index map: "
                        f"{index_map_path}"
                    )

        except Exception as exc:
            print(
                "[VG2 reconstruction] index map unusable; "
                f"fall back to exact row matching: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------------
    # Robust fallback: exact filtered-row -> original-row matching.
    # Both JSON files contain the same [image_id, positive, negative] tuples.
    # --------------------------------------------------------------
    if not used_map:
        positions = defaultdict(
            list
        )

        for old_id, row in enumerate(
            original_data
        ):
            positions[
                canonical_json_row(
                    row
                )
            ].append(
                old_id
            )

        cursors = defaultdict(
            int
        )

        old_ids = []

        for new_id, row in enumerate(
            filtered_data
        ):
            key = canonical_json_row(
                row
            )

            matches = positions.get(
                key,
                [],
            )

            cursor = cursors[
                key
            ]

            if cursor >= len(
                matches
            ):
                raise RuntimeError(
                    f"Filtered VG row new_id={new_id} could not be "
                    "matched exactly to data/vg_qa_two_obj.json"
                )

            old_ids.append(
                matches[
                    cursor
                ]
            )

            cursors[
                key
            ] += 1

        print(
            "[VG2 reconstruction] recovered original ids by exact "
            "filtered-row matching"
        )

    # --------------------------------------------------------------
    # Construct four-way prompt rows.
    # --------------------------------------------------------------
    result = []

    relation_counts = Counter()

    for new_id, old_id in enumerate(
        old_ids
    ):
        if old_id not in six_prompts:
            raise RuntimeError(
                f"No six-option prompt for original id={old_id}"
            )

        source = six_prompts[
            old_id
        ]

        question = str(
            source[
                "question"
            ]
        )

        # Preserve wording/object names. Only remove the two excluded options.
        question4, replacements = re.subn(
            (
                r"left\s*,\s*right\s*,\s*front\s*,\s*behind\s*,\s*"
                r"above\s*(?:,?\s*or\s*)below"
            ),
            "left, right, above, or below",
            question,
            flags=re.I,
        )

        if replacements == 0:
            # Slightly more permissive fallback for punctuation variants.
            question4 = re.sub(
                r"\bfront\s*,?\s*behind\s*,?\s*",
                "",
                question,
                flags=re.I,
            )

            question4 = re.sub(
                r"left\s*,\s*right\s*,\s*above\s+or\s+below",
                "left, right, above, or below",
                question4,
                flags=re.I,
            )

        raw_answer = standard_answer_value(
            source.get(
                "answer"
            )
        )

        relation = normalize_relation(
            raw_answer,
            "vg2",
        )

        if relation not in {
            "left",
            "right",
            "above",
            "below",
        }:
            raise RuntimeError(
                f"Filtered row new_id={new_id} maps to original "
                f"id={old_id}, but six-option answer={raw_answer!r} "
                "is not one of the retained four relations."
            )

        relation_counts[
            relation
        ] += 1

        # Keep answer as a list to match the existing prompt files.
        pretty_answer = {
            "left": "Left",
            "right": "Right",
            "above": "Above",
            "below": "Below",
        }[
            relation
        ]

        result.append({
            "id": new_id,
            "question": question4,
            "answer": [
                pretty_answer
            ],
            "_original_id": old_id,
        })

    print(
        f"[VG2 reconstruction] built {len(result)} four-option prompts | "
        f"relations={dict(relation_counts)}"
    )

    return result


def save_reconstructed_vg_prompt(rows, output_dir):
    path = (
        Path(
            output_dir
        )
        / "reconstructed_VG_QA_two_obj_with_answer_four_options.jsonl"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            clean = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
            }

            handle.write(
                json.dumps(
                    clean,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[VG2 reconstruction] audit copy saved to {path}"
    )

    return path


def parse_vg_prompt_rows(rows, source_label):
    parsed = {}

    for row_no, row in enumerate(
        rows,
        1,
    ):
        required = {
            "id",
            "question",
            "answer",
        }

        if not required.issubset(
            row
        ):
            raise ValueError(
                f"{source_label}: row {row_no} must contain "
                f"id/question/answer; keys={sorted(row.keys())}"
            )

        sid = int(
            row[
                "id"
            ]
        )

        if sid in parsed:
            raise ValueError(
                f"Duplicate VG2 prompt id={sid} in {source_label}"
            )

        raw_question = str(
            row[
                "question"
            ]
        )

        question_text = extract_standard_user_text(
            raw_question
        )

        subject, reference = parse_vg_standard_objects(
            question_text
        )

        relation = normalize_relation(
            standard_answer_value(
                row[
                    "answer"
                ]
            ),
            "vg2",
        )

        parsed[
            sid
        ] = {
            "sid": sid,
            "raw_question": raw_question,
            "question_text": question_text,
            "subject": subject,
            "reference": reference,
            "relation": relation,
        }

    if not parsed:
        raise RuntimeError(
            f"No VG2 prompts loaded from {source_label}"
        )

    return parsed


def load_vg_standard_prompts(
    path,
    filtered_data=None,
    args=None,
):
    path = Path(
        path
    )

    if path.exists():
        rows = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if line.strip():
                    rows.append(
                        json.loads(
                            line
                        )
                    )

        return parse_vg_prompt_rows(
            rows,
            str(
                path
            ),
        )

    if filtered_data is None or args is None:
        raise FileNotFoundError(
            f"Missing VG2 prompt JSONL: {path}"
        )

    print(
        f"[VG2] four-option prompt file is missing: {path}"
    )

    print(
        "[VG2] reconstructing it from the existing six-option prompt + "
        "filtered/original VG data"
    )

    rows = reconstruct_vg_four_option_prompt_rows(
        filtered_data,
        args,
    )

    save_reconstructed_vg_prompt(
        rows,
        args.output_dir,
    )

    return parse_vg_prompt_rows(
        rows,
        "reconstructed four-option VG2 prompts",
    )



def find_vg_image(image_root, image_id):
    """
    Mirrors the existing repo VG2 image resolver. It supports the flat
    data/vg/images layout as well as common Visual Genome subdirectories.
    """
    image_root = Path(
        image_root
    )

    stem = str(
        image_id
    ).strip()

    for suffix in VG_IMAGE_SUFFIXES:
        candidate = (
            image_root
            / f"{stem}{suffix}"
        )

        if candidate.exists():
            return candidate

    for subdir in (
        "VG_100K",
        "VG_100K_2",
        "images",
    ):
        base = (
            image_root
            / subdir
        )

        for suffix in VG_IMAGE_SUFFIXES:
            candidate = (
                base
                / f"{stem}{suffix}"
            )

            if candidate.exists():
                return candidate

    return None


def resolve_vg_image_root(args, data):
    if args.vg_image_root != "auto":
        root = Path(args.vg_image_root)
        if not root.exists():
            raise FileNotFoundError(
                f"Explicit --vg-image-root does not exist: {root}"
            )
        return root

    data_root = Path(args.data_root)

    candidates = [
        data_root / "vg" / "images",
        data_root / "vg",
        data_root / "visual_genome" / "images",
        data_root / "visual_genome",
        data_root / "VisualGenome" / "images",
        data_root / "VisualGenome",
        data_root / "VG",
        data_root,
    ]

    probe_ids = []
    for item in data[:50]:
        if isinstance(item, (list, tuple)) and item:
            probe_ids.append(item[0])
        if len(probe_ids) >= 8:
            break

    for root in candidates:
        if not root.exists():
            continue

        hits = sum(
            find_vg_image(root, image_id) is not None
            for image_id in probe_ids
        )

        if hits:
            print(
                f"[VG2] auto image root={root} "
                f"(resolved {hits}/{len(probe_ids)} probe ids)"
            )
            return root

    if probe_ids:
        stem = str(probe_ids[0]).strip()

        for suffix in VG_IMAGE_SUFFIXES:
            matches = list(data_root.rglob(f"{stem}{suffix}"))
            if matches:
                root = matches[0].parent
                print(
                    f"[VG2] recursive image discovery found {matches[0]}; "
                    f"using root={root}"
                )
                return root

    raise FileNotFoundError(
        "Could not locate Visual Genome images automatically. "
        "Run: find data -type f -name '<image_id>.jpg' | head "
        "and pass that directory with --vg-image-root."
    )


def load_vg2(args):
    """
    Binary VG2 left/right replication.

    Uses files that already exist in this checkout:
      data/vg_qa_two_obj_four_options.json
      data/vg_qa_two_obj.json
      data/vg_qa_two_obj_four_options_index_map.json (optional)
      prompts/VG_QA_two_obj_with_answer_six_options.jsonl

    We recover each filtered row's original prompt id, then KEEP ONLY samples
    whose original six-option GT is left or right.

    Prompt text is rewritten only at the answer-choice suffix:
        left, right, front, behind, above or below
    ->
        left or right

    Object wording and question body remain exactly from the repo prompt.
    """
    data_path = Path(args.vg_json)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing VG2 filtered data JSON: {data_path}"
        )

    with data_path.open("r", encoding="utf-8") as handle:
        filtered_data = json.load(handle)

    if not isinstance(filtered_data, list):
        raise TypeError(
            f"{data_path} must contain a top-level list"
        )

    image_root = resolve_vg_image_root(
        args,
        filtered_data,
    )

    six_prompt_path = Path(args.vg_six_prompt_jsonl)
    original_data_path = Path(args.vg_original_json)
    index_map_path = Path(args.vg_index_map)

    if not six_prompt_path.exists():
        raise FileNotFoundError(six_prompt_path)

    if not original_data_path.exists():
        raise FileNotFoundError(original_data_path)

    six_prompts = load_jsonl_by_id(
        six_prompt_path
    )

    with original_data_path.open("r", encoding="utf-8") as handle:
        original_data = json.load(handle)

    # --------------------------------------------------------------
    # Recover filtered row -> original row id.
    # --------------------------------------------------------------
    old_ids = []
    used_map = False

    if index_map_path.exists():
        try:
            with index_map_path.open("r", encoding="utf-8") as handle:
                index_map = json.load(handle)

            if (
                isinstance(index_map, list)
                and len(index_map) == len(filtered_data)
            ):
                candidate = []

                for new_id, entry in enumerate(index_map):
                    old_id = infer_old_id_from_map_entry(entry)

                    if old_id is None:
                        candidate = []
                        break

                    if (
                        isinstance(entry, dict)
                        and "new_id" in entry
                        and int(entry["new_id"]) != new_id
                    ):
                        candidate = []
                        break

                    candidate.append(old_id)

                if len(candidate) == len(filtered_data):
                    old_ids = candidate
                    used_map = True
                    print(
                        f"[VG2-LR] using index map: {index_map_path}"
                    )

        except Exception as exc:
            print(
                f"[VG2-LR] index map unusable: "
                f"{type(exc).__name__}: {exc}"
            )

    if not used_map:
        positions = defaultdict(list)

        for old_id, row in enumerate(original_data):
            positions[
                canonical_json_row(row)
            ].append(old_id)

        cursors = defaultdict(int)

        for new_id, row in enumerate(filtered_data):
            key = canonical_json_row(row)
            matches = positions.get(key, [])
            cursor = cursors[key]

            if cursor >= len(matches):
                raise RuntimeError(
                    f"Could not match filtered VG row {new_id} "
                    "to data/vg_qa_two_obj.json"
                )

            old_ids.append(matches[cursor])
            cursors[key] += 1

        print(
            "[VG2-LR] recovered original ids by exact row matching"
        )

    # --------------------------------------------------------------
    # Build binary left/right records.
    # --------------------------------------------------------------
    records = []
    skipped = Counter()
    missing_images = []

    for new_id, old_id in enumerate(old_ids):
        if old_id not in six_prompts:
            skipped["missing_prompt"] += 1
            continue

        source = six_prompts[old_id]

        raw_answer = standard_answer_value(
            source.get("answer")
        )

        relation = normalize_relation(
            raw_answer,
            "vg2",
        )

        if relation not in {"left", "right"}:
            skipped[f"drop_{relation or 'unknown'}"] += 1
            continue

        raw_question = str(
            source["question"]
        )

        # Keep repo wording; only reduce the option list to the binary task.
        question2 = re.sub(
            (
                r"Answer\\s+with\\s+left\\s*,\\s*right\\s*,\\s*front\\s*,\\s*"
                r"behind\\s*,\\s*above\\s*(?:,?\\s*or\\s*)below\\s*\\."
            ),
            "Answer with left or right.",
            raw_question,
            flags=re.I,
        )

        if question2 == raw_question:
            question2 = re.sub(
                r"Answer\\s+with\\s+left\\s*,\\s*right\\s*,\\s*front\\s*,\\s*"
                r"behind\\s*,\\s*above\\s+or\\s+below",
                "Answer with left or right",
                raw_question,
                flags=re.I,
            )

        question_text = extract_standard_user_text(
            question2
        )

        subject, reference = parse_vg_standard_objects(
            question_text
        )

        item = filtered_data[new_id]

        if (
            not isinstance(item, (list, tuple))
            or len(item) < 1
        ):
            skipped["malformed_data_row"] += 1
            continue

        image_id = item[0]

        image_path = find_vg_image(
            image_root,
            image_id,
        )

        if image_path is None:
            skipped["missing_image"] += 1

            if len(missing_images) < 20:
                missing_images.append(str(image_id))

            continue

        records.append({
            "sid": new_id,
            "dataset": "vg2",
            "image_id": str(image_id),
            "image_path": str(image_path),
            "subject": subject,
            "reference": reference,
            "relation": relation,
            "question_text": question_text,
            "raw_question": question2,
            "original_id": old_id,
            "pair_group": " || ".join(
                sorted(
                    [
                        subject.lower(),
                        reference.lower(),
                    ]
                )
            ),
        })

    counts = Counter(
        r["relation"]
        for r in records
    )

    print(
        f"[VG2-LR] filtered_rows={len(filtered_data)} | "
        f"loaded={len(records)} | "
        f"relations={dict(counts)} | "
        f"skipped={dict(skipped)}"
    )

    if missing_images:
        print(
            "[VG2-LR] missing image examples="
            + ", ".join(missing_images)
        )

    for relation in ("left", "right"):
        if counts[relation] == 0:
            raise RuntimeError(
                f"VG2-LR loader has zero {relation} samples."
            )

    return records


# =============================================================================
# split
# =============================================================================

def split_controlled_by_pair(records, train_frac, seed):
    groups = defaultdict(list)

    for record in records:
        groups[record["pair_group"]].append(
            record
        )

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_train = int(
        round(
            len(keys)
            * train_frac
        )
    )

    n_train = max(
        1,
        min(
            n_train,
            len(keys) - 1,
        ),
    )

    train_keys = set(
        keys[:n_train]
    )

    train = [
        record
        for record in records
        if record["pair_group"] in train_keys
    ]

    test = [
        record
        for record in records
        if record["pair_group"] not in train_keys
    ]

    return train, test


def split_stratified(records, labels, train_frac, seed):
    rng = random.Random(seed)

    train = []
    test = []

    for relation in labels:
        rows = [
            record
            for record in records
            if record["relation"] == relation
        ]

        rows = sorted(
            rows,
            key=lambda r: r["sid"],
        )

        rng.shuffle(rows)

        n_train = int(
            round(
                len(rows)
                * train_frac
            )
        )

        n_train = max(
            1,
            min(
                n_train,
                len(rows) - 1,
            ),
        )

        train.extend(
            rows[:n_train]
        )

        test.extend(
            rows[n_train:]
        )

    return (
        sorted(
            train,
            key=lambda r: r["sid"],
        ),
        sorted(
            test,
            key=lambda r: r["sid"],
        ),
    )


def make_split(records, args, labels):
    if args.dataset == "controlled_a":
        train, test = split_controlled_by_pair(
            records,
            args.train_frac,
            args.seed,
        )

    else:
        train, test = split_stratified(
            records,
            labels,
            args.train_frac,
            args.seed,
        )

    print(
        f"[split] TRAIN={len(train)} "
        f"{dict(Counter(r['relation'] for r in train))}"
    )

    print(
        f"[split] TEST ={len(test)} "
        f"{dict(Counter(r['relation'] for r in test))}"
    )

    return train, test


# =============================================================================
# Qwen loading / prompt
# =============================================================================

def load_model(args):
    if not hasattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    ):
        raise RuntimeError(
            "Transformers has no Qwen2_5_VLForConditionalGeneration."
        )

    dtype = (
        torch.bfloat16
        if args.dtype == "auto"
        else auto_dtype(args.dtype)
    )

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {
            "": args.device
        },
    }

    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl

    cls = (
        transformers
        .Qwen2_5_VLForConditionalGeneration
    )

    try:
        model = cls.from_pretrained(
            args.model_id,
            dtype=dtype,
            **kwargs,
        )

    except TypeError:
        model = cls.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            **kwargs,
        )

    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            use_fast=False,
        )

    except TypeError:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )

    return model, processor


def resolve_layers(model):
    for path in [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ]:
        current = model

        try:
            for part in path.split("."):
                current = getattr(
                    current,
                    part,
                )

            if len(current):
                return current, path

        except Exception:
            pass

    raise RuntimeError(
        "Could not locate decoder blocks."
    )


def prompt_text(record, answer_words):
    # For repo-native VG2, preserve the exact standard prompt used by the
    # existing VG2 experiments. This keeps prompting aligned with prior runs.
    if (
        record.get("dataset") == "vg2"
        and record.get("question_text")
    ):
        return record["question_text"]

    answer_text = ", ".join(
        answer_words[:-1]
    ) + f", or {answer_words[-1]}"

    return (
        f"Determine the spatial relation of the {record['subject']} "
        f"to the {record['reference']} in the image. "
        f"Answer with {answer_text}."
    )


def build_prompt(processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception:
        return question


def build_batch(processor, image, record, args, answer_words):
    question = prompt_text(
        record,
        answer_words,
    )

    prompt = build_prompt(
        processor,
        image,
        question,
    )

    errors = []

    for fn in [
        lambda: processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ),
    ]:
        try:
            batch = fn()
            break

        except Exception as exc:
            errors.append(exc)

    else:
        raise RuntimeError(
            f"Processor failed: {errors[-1]}"
        )

    return {
        key: (
            value.to(args.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def generate(model, processor, batch, args):
    input_len = int(
        batch["input_ids"].shape[1]
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    if output_ids.shape[1] > input_len:
        generated = output_ids[
            0,
            input_len:
        ]

    else:
        generated = output_ids[0]

    text = processor.tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    del output_ids

    return text


def make_gray(real, value):
    v = int(
        max(
            0,
            min(
                255,
                value,
            ),
        )
    )

    return Image.new(
        "RGB",
        real.size,
        (
            v,
            v,
            v,
        ),
    )


# =============================================================================
# token span finding
# =============================================================================

def find_all_subsequences(sequence, pattern):
    if not pattern:
        return []

    starts = []
    n = len(pattern)

    for i in range(
        len(sequence) - n + 1
    ):
        if (
            list(
                sequence[
                    i:
                    i + n
                ]
            )
            == list(pattern)
        ):
            starts.append(i)

    return starts


def phrase_spans(tokenizer, full_ids, phrase):
    variants = [
        phrase,
        " " + phrase,
        phrase.strip(),
        " " + phrase.strip(),
    ]

    spans = []
    seen = set()

    for variant in variants:
        ids = tokenizer.encode(
            variant,
            add_special_tokens=False,
        )

        for start in find_all_subsequences(
            full_ids,
            ids,
        ):
            span = tuple(
                range(
                    start,
                    start + len(ids),
                )
            )

            if span not in seen:
                seen.add(span)
                spans.append(
                    list(span)
                )

    return spans


def locate_pair_spans(
    tokenizer,
    input_ids,
    subject,
    reference,
):
    subject_spans = phrase_spans(
        tokenizer,
        input_ids,
        subject,
    )

    reference_spans = phrase_spans(
        tokenizer,
        input_ids,
        reference,
    )

    if not subject_spans or not reference_spans:
        return None, None

    best = None

    for s_span in subject_spans:
        for r_span in reference_spans:
            if set(s_span) & set(r_span):
                continue

            # Prompt has subject before reference. Prefer that ordering, then
            # minimal distance, then latest occurrence.
            order_penalty = int(
                s_span[0] >= r_span[0]
            )

            distance = abs(
                float(np.mean(s_span))
                - float(np.mean(r_span))
            )

            score = (
                order_penalty,
                distance,
                -s_span[0],
            )

            if best is None or score < best[0]:
                best = (
                    score,
                    s_span,
                    r_span,
                )

    if best is None:
        return None, None

    return best[1], best[2]


# =============================================================================
# hooks: last state + object pair state
# =============================================================================

def extract_hidden(output):
    if torch.is_tensor(output):
        return output, ("tensor", 0)

    if isinstance(output, tuple):
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                return item, ("tuple", i)

    if isinstance(output, list):
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                return item, ("list", i)

    raise RuntimeError(
        f"Could not extract hidden from {type(output)}"
    )


def replace_hidden(output, descriptor, hidden):
    kind, index = descriptor

    if kind == "tensor":
        return hidden

    values = list(output)
    values[index] = hidden

    return (
        tuple(values)
        if kind == "tuple"
        else values
    )


class CaptureStates:
    def __init__(
        self,
        layers,
        actuator_layers,
        guide_layers,
        subject_span,
        reference_span,
    ):
        self.handles = []
        self.last_states = {}
        self.pair_states = {}

        selected = sorted(
            set(
                actuator_layers
                + guide_layers
            )
        )

        self.done = {
            layer: False
            for layer in selected
        }

        self.actuator_set = set(
            actuator_layers
        )

        self.guide_set = set(
            guide_layers
        )

        self.subject_span = subject_span
        self.reference_span = reference_span

        for layer in selected:
            self.handles.append(
                layers[layer].register_forward_hook(
                    self._hook(layer)
                )
            )

    def _hook(self, layer):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.done[layer]:
                return output

            hidden, _ = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            if layer in self.actuator_set:
                self.last_states[layer] = (
                    hidden[
                        0,
                        -1,
                        :
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            if layer in self.guide_set:
                max_index = max(
                    self.subject_span
                    + self.reference_span
                )

                if max_index >= hidden.shape[1]:
                    raise RuntimeError(
                        f"L{layer}: object token span exceeds seq_len={hidden.shape[1]}"
                    )

                hs = hidden[
                    0,
                    self.subject_span,
                    :
                ].mean(dim=0)

                hr = hidden[
                    0,
                    self.reference_span,
                    :
                ].mean(dim=0)

                self.pair_states[layer] = (
                    (hs - hr)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            self.done[layer] = True

            return output

        return hook

    def close(self):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SteerLast:
    def __init__(
        self,
        layers,
        templates,
        selected_layers,
        target_relation,
        scale,
        edit_mode,
        source_relation,
    ):
        self.handles = []

        self.done = {
            layer: False
            for layer in selected_layers
        }

        self.templates = templates
        self.target = target_relation
        self.source = source_relation
        self.scale = float(scale)
        self.edit_mode = edit_mode

        for layer in selected_layers:
            self.handles.append(
                layers[layer].register_forward_hook(
                    self._hook(layer)
                )
            )

    def vector(self, layer):
        target = np.asarray(
            self.templates[layer]["shared"][
                self.target
            ],
            dtype=np.float32,
        )

        if (
            self.edit_mode == "add"
            or self.source not in self.templates[layer]["shared"]
        ):
            return self.scale * target

        source = np.asarray(
            self.templates[layer]["shared"][
                self.source
            ],
            dtype=np.float32,
        )

        return (
            self.scale
            * (
                target
                - source
            )
        )

    def _hook(self, layer):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if self.done[layer]:
                return output

            hidden, descriptor = extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            vec = torch.as_tensor(
                self.vector(layer),
                device=hidden.device,
                dtype=hidden.dtype,
            )

            edited = hidden.clone()

            edited[
                :,
                -1,
                :
            ] += vec[
                None,
                :
            ]

            self.done[layer] = True

            return replace_hidden(
                output,
                descriptor,
                edited,
            )

        return hook

    def close(self):
        for handle in reversed(
            self.handles
        ):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# run real/gray with state capture
# =============================================================================

def prepare_batch_and_spans(
    processor,
    image,
    record,
    args,
    answer_words,
):
    batch = build_batch(
        processor,
        image,
        record,
        args,
        answer_words,
    )

    ids = (
        batch["input_ids"][0]
        .detach()
        .cpu()
        .tolist()
    )

    subject_span, reference_span = locate_pair_spans(
        processor.tokenizer,
        ids,
        record["subject"],
        record["reference"],
    )

    if (
        subject_span is None
        or reference_span is None
    ):
        raise RuntimeError(
            f"Could not locate text tokens: "
            f"subject={record['subject']!r}, "
            f"reference={record['reference']!r}"
        )

    return (
        batch,
        subject_span,
        reference_span,
    )


def generate_capture(
    model,
    processor,
    layers,
    image,
    record,
    args,
    answer_words,
    actuator_layers,
    guide_layers,
):
    (
        batch,
        subject_span,
        reference_span,
    ) = prepare_batch_and_spans(
        processor,
        image,
        record,
        args,
        answer_words,
    )

    with CaptureStates(
        layers,
        actuator_layers,
        guide_layers,
        subject_span,
        reference_span,
    ) as capture:

        text = generate(
            model,
            processor,
            batch,
            args,
        )

        last_states = dict(
            capture.last_states
        )

        pair_states = dict(
            capture.pair_states
        )

    del batch

    pred = parse_pred(
        text,
        args.dataset,
    )

    return (
        text,
        pred,
        last_states,
        pair_states,
    )


def forward_capture_pair_only(
    model,
    processor,
    layers,
    image,
    record,
    args,
    answer_words,
    guide_layers,
):
    (
        batch,
        subject_span,
        reference_span,
    ) = prepare_batch_and_spans(
        processor,
        image,
        record,
        args,
        answer_words,
    )

    with CaptureStates(
        layers,
        [],
        guide_layers,
        subject_span,
        reference_span,
    ) as capture:

        with torch.inference_mode():
            outputs = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

        pair_states = dict(
            capture.pair_states
        )

    del batch
    del outputs

    return pair_states


# =============================================================================
# training collection
# =============================================================================

def filter_sequence(requested):
    if requested == "real_correct_gray_wrong":
        return [
            "real_correct_gray_wrong",
            "real_correct",
            "all",
        ]

    if requested == "real_correct":
        return [
            "real_correct",
            "all",
        ]

    return ["all"]


def train_sample_allowed(row, mode):
    if mode == "all":
        return True

    if mode == "real_correct":
        return bool(
            row["real_correct"]
        )

    if mode == "real_correct_gray_wrong":
        return (
            bool(
                row["real_correct"]
            )
            and not bool(
                row["gray_correct"]
            )
        )

    raise ValueError(mode)


def collect_train(
    model,
    processor,
    layers,
    train_records,
    args,
    answer_words,
    actuator_layers,
    guide_layers,
):
    rows = []
    cache = {}

    for record in tqdm(
        train_records,
        desc="TRAIN Real/Gray states",
    ):
        real = None
        gray = None

        try:
            real = Image.open(
                record["image_path"]
            ).convert("RGB")

            gray = make_gray(
                real,
                args.gray_value,
            )

            (
                real_text,
                real_pred,
                real_last,
                real_pair,
            ) = generate_capture(
                model,
                processor,
                layers,
                real,
                record,
                args,
                answer_words,
                actuator_layers,
                guide_layers,
            )

            (
                gray_text,
                gray_pred,
                gray_last,
                gray_pair,
            ) = generate_capture(
                model,
                processor,
                layers,
                gray,
                record,
                args,
                answer_words,
                actuator_layers,
                guide_layers,
            )

            gt = record["relation"]

            row = {
                "sid": record["sid"],
                "relation": gt,
                "subject": record["subject"],
                "reference": record["reference"],
                "real_pred": real_pred or "",
                "gray_pred": gray_pred or "",
                "real_correct": int(
                    real_pred == gt
                ),
                "gray_correct": int(
                    gray_pred == gt
                ),
                "real_text": real_text,
                "gray_text": gray_text,
            }

            rows.append(row)

            cache[
                record["sid"]
            ] = {
                "last_delta": {
                    layer: (
                        real_last[layer]
                        - gray_last[layer]
                    ).astype(np.float32)
                    for layer in actuator_layers
                },
                "pair_delta": {
                    layer: (
                        real_pair[layer]
                        - gray_pair[layer]
                    ).astype(np.float32)
                    for layer in guide_layers
                },
            }

        except Exception as exc:
            tqdm.write(
                f"[TRAIN ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    return rows, cache


def fit_actuator_templates(
    train_records,
    train_rows,
    cache,
    actuator_layers,
    labels,
    requested_filter,
):
    row_map = {
        row["sid"]: row
        for row in train_rows
    }

    record_map = {
        record["sid"]: record
        for record in train_records
    }

    for mode in filter_sequence(
        requested_filter
    ):
        bags = {
            layer: {
                relation: []
                for relation in labels
            }
            for layer in actuator_layers
        }

        used = []

        for sid, item in cache.items():
            if sid not in row_map:
                continue

            if not train_sample_allowed(
                row_map[sid],
                mode,
            ):
                continue

            relation = record_map[sid][
                "relation"
            ]

            used.append(sid)

            for layer in actuator_layers:
                bags[layer][relation].append(
                    item["last_delta"][
                        layer
                    ]
                )

        missing = [
            (layer, relation)
            for layer in actuator_layers
            for relation in labels
            if not bags[layer][relation]
        ]

        if missing:
            print(
                f"[actuator template] filter={mode} "
                f"missing={missing[:8]} -> relax"
            )

            continue

        templates = {}

        for layer in actuator_layers:
            relation_mean = {
                relation: np.stack(
                    bags[layer][relation],
                    axis=0,
                ).mean(axis=0).astype(
                    np.float32
                )
                for relation in labels
            }

            global_mean = np.stack(
                [
                    relation_mean[relation]
                    for relation in labels
                ],
                axis=0,
            ).mean(axis=0).astype(
                np.float32
            )

            shared = {
                relation: (
                    relation_mean[relation]
                    - global_mean
                ).astype(
                    np.float32
                )
                for relation in labels
            }

            templates[layer] = {
                "global": global_mean,
                "shared": shared,
            }

        print(
            f"[actuator template] requested={requested_filter} "
            f"used={mode} N={len(used)} "
            f"counts={dict(Counter(record_map[sid]['relation'] for sid in used))}"
        )

        return templates, mode

    raise RuntimeError(
        "Could not fit four actuator relation templates."
    )


def fit_guide_codebook(
    train_records,
    train_rows,
    cache,
    guide_layers,
    labels,
):
    record_map = {
        record["sid"]: record
        for record in train_records
    }

    # Guide uses all TRAIN samples; no need for generation correctness.
    codebook = {}

    for layer in guide_layers:
        vectors = []
        y = []

        for sid, item in cache.items():
            vectors.append(
                item["pair_delta"][
                    layer
                ]
            )

            y.append(
                record_map[sid][
                    "relation"
                ]
            )

        X = np.stack(
            vectors,
            axis=0,
        ).astype(np.float64)

        y = np.asarray(
            y,
            dtype=object,
        )

        center = X.mean(
            axis=0
        )

        prototypes = {}

        for relation in labels:
            mask = (
                y == relation
            )

            if not np.any(mask):
                raise RuntimeError(
                    f"Guide L{layer} missing relation={relation}"
                )

            mu = (
                X[mask]
                - center[
                    None,
                    :
                ]
            ).mean(
                axis=0
            )

            norm = np.linalg.norm(
                mu
            )

            if norm <= EPS:
                raise RuntimeError(
                    f"Guide L{layer} degenerate relation={relation}"
                )

            prototypes[
                relation
            ] = (
                mu
                / norm
            ).astype(
                np.float32
            )

        codebook[layer] = {
            "center": center.astype(
                np.float32
            ),
            "prototypes": prototypes,
        }

    return codebook


def guide_predict(
    pair_delta,
    guide_codebook,
    guide_layers,
    labels,
):
    score_sum = {
        relation: 0.0
        for relation in labels
    }

    for layer in guide_layers:
        q = pair_delta[
            layer
        ].astype(
            np.float64
        )

        q = (
            q
            - guide_codebook[
                layer
            ][
                "center"
            ].astype(
                np.float64
            )
        )

        qnorm = np.linalg.norm(
            q
        )

        if qnorm <= EPS:
            continue

        q = q / qnorm

        for relation in labels:
            score_sum[
                relation
            ] += float(
                np.dot(
                    q,
                    guide_codebook[
                        layer
                    ][
                        "prototypes"
                    ][
                        relation
                    ].astype(
                        np.float64
                    ),
                )
            )

    return max(
        labels,
        key=lambda relation:
            score_sum[
                relation
            ],
    )


# =============================================================================
# TEST baseline + guide states
# =============================================================================

def collect_test_baseline_and_guide(
    model,
    processor,
    layers,
    test_records,
    args,
    answer_words,
    guide_layers,
):
    rows = []
    guide_delta_cache = {}

    for record in tqdm(
        test_records,
        desc="TEST baseline + guide",
    ):
        real = None
        gray = None

        try:
            real = Image.open(
                record["image_path"]
            ).convert("RGB")

            gray = make_gray(
                real,
                args.gray_value,
            )

            (
                real_text,
                real_pred,
                _real_last,
                real_pair,
            ) = generate_capture(
                model,
                processor,
                layers,
                real,
                record,
                args,
                answer_words,
                [],
                guide_layers,
            )

            gray_pair = forward_capture_pair_only(
                model,
                processor,
                layers,
                gray,
                record,
                args,
                answer_words,
                guide_layers,
            )

            gt = record["relation"]

            rows.append({
                "sid": record["sid"],
                "relation": gt,
                "baseline_pred": real_pred or "",
                "baseline_correct": int(
                    real_pred == gt
                ),
                "baseline_text": real_text,
            })

            guide_delta_cache[
                record["sid"]
            ] = {
                layer: (
                    real_pair[layer]
                    - gray_pair[layer]
                ).astype(
                    np.float32
                )
                for layer in guide_layers
            }

        except Exception as exc:
            tqdm.write(
                f"[TEST BASE ERROR sid={record['sid']}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    return rows, guide_delta_cache


# =============================================================================
# steering eval
# =============================================================================

def run_steering(
    model,
    processor,
    layers,
    templates,
    actuator_layers,
    test_record_map,
    baseline_rows,
    targets,
    args,
    answer_words,
    edit_mode,
    apply_mode,
    condition,
):
    rows = []

    for base in tqdm(
        baseline_rows,
        desc=condition,
    ):
        sid = int(
            base["sid"]
        )

        record = test_record_map[
            sid
        ]

        baseline_pred = (
            base["baseline_pred"]
            if base["baseline_pred"]
            else None
        )

        target = targets[
            sid
        ]

        if apply_mode == "all":
            apply_edit = True

        elif apply_mode == "conflict_only":
            apply_edit = (
                baseline_pred is not None
                and target
                != baseline_pred
            )

        else:
            raise ValueError(
                apply_mode
            )

        if not apply_edit:
            rows.append({
                "condition": condition,
                "sid": sid,
                "relation": record["relation"],
                "target": target,
                "baseline_pred": baseline_pred or "",
                "edit_pred": baseline_pred or "",
                "baseline_correct": int(
                    base["baseline_correct"]
                ),
                "edit_correct": int(
                    base["baseline_correct"]
                ),
                "applied": 0,
                "W2C": 0,
                "C2W": 0,
                "changed": 0,
            })

            continue

        image = None

        try:
            image = Image.open(
                record["image_path"]
            ).convert("RGB")

            batch = build_batch(
                processor,
                image,
                record,
                args,
                answer_words,
            )

            with SteerLast(
                layers,
                templates,
                actuator_layers,
                target,
                args.scale,
                edit_mode,
                baseline_pred,
            ):
                text = generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            pred = parse_pred(
                text,
                args.dataset,
            )

            base_correct = int(
                base["baseline_correct"]
            )

            edit_correct = int(
                pred
                == record["relation"]
            )

            rows.append({
                "condition": condition,
                "sid": sid,
                "relation": record["relation"],
                "target": target,
                "baseline_pred": baseline_pred or "",
                "edit_pred": pred or "",
                "baseline_correct": base_correct,
                "edit_correct": edit_correct,
                "applied": 1,
                "W2C": int(
                    base_correct == 0
                    and edit_correct == 1
                ),
                "C2W": int(
                    base_correct == 1
                    and edit_correct == 0
                ),
                "changed": int(
                    (pred or "")
                    != (baseline_pred or "")
                ),
                "text": text,
            })

            del batch

        except Exception as exc:
            tqdm.write(
                f"[STEER ERROR sid={sid}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                image.close()

            cleanup()

    return rows


def summarize(rows, condition):
    n = len(rows)

    base_acc = safe_mean(
        row["baseline_correct"]
        for row in rows
    )

    edit_acc = safe_mean(
        row["edit_correct"]
        for row in rows
    )

    wrong_n = sum(
        1 - int(
            row["baseline_correct"]
        )
        for row in rows
    )

    correct_n = n - wrong_n

    w2c = sum(
        int(row["W2C"])
        for row in rows
    )

    c2w = sum(
        int(row["C2W"])
        for row in rows
    )

    applied = sum(
        int(row["applied"])
        for row in rows
    )

    return {
        "condition": condition,
        "N": n,
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "applied": applied,
        "applied_rate": (
            applied / n
            if n
            else float("nan")
        ),
        "W2C": w2c,
        "W2C_rate_wrong": (
            w2c / wrong_n
            if wrong_n
            else float("nan")
        ),
        "C2W": c2w,
        "C2W_rate_correct": (
            c2w / correct_n
            if correct_n
            else float("nan")
        ),
        "net": w2c - c2w,
        "changed_rate": safe_mean(
            row["changed"]
            for row in rows
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(
        args.output_dir
    )

    if (
        args.overwrite
        and outdir.exists()
    ):
        shutil.rmtree(
            outdir
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preset = model_preset(
        args.model_id
    )

    spec = dataset_spec(
        args.dataset
    )

    labels = spec[
        "labels"
    ]

    answer_words = spec[
        "answer_words"
    ]

    if args.dataset == "controlled_a":
        records = load_controlled_a(
            args
        )

    else:
        records = load_vg2(
            args
        )

    if args.max_samples is not None:
        records = records[
            :args.max_samples
        ]

    train_records, test_records = make_split(
        records,
        args,
        labels,
    )

    model, processor = load_model(
        args
    )

    layers, layer_path = resolve_layers(
        model
    )

    actuator_layers = preset[
        "actuator_layers"
    ]

    guide_layers = preset[
        "guide_layers"
    ]

    if max(
        actuator_layers
        + guide_layers
    ) >= len(layers):
        raise RuntimeError(
            f"Model has {len(layers)} blocks, "
            f"but requested layers "
            f"{actuator_layers + guide_layers}"
        )

    print(
        "\n"
        + "=" * 155
    )

    print(
        "VG2 LEFT/RIGHT LATE CAUSAL REPLICATION"
    )

    print(
        "=" * 155
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"dataset={args.dataset}"
    )

    print(
        f"decoder={layer_path}"
    )

    print(
        f"decoder_blocks={len(layers)}"
    )

    print(
        f"actuator_layers={actuator_layers} [REUSED FROM COCO]"
    )

    print(
        f"guide_layers={guide_layers} [REUSED FROM COCO]"
    )

    print(
        f"labels={labels}"
    )

    print(
        f"TRAIN={len(train_records)} TEST={len(test_records)}"
    )

    print(
        "=" * 155
    )

    # TRAIN
    train_rows, train_cache = collect_train(
        model,
        processor,
        layers,
        train_records,
        args,
        answer_words,
        actuator_layers,
        guide_layers,
    )

    write_csv(
        outdir
        / "train_real_gray_generation.csv",
        train_rows,
    )

    actuator_templates, filter_used = fit_actuator_templates(
        train_records,
        train_rows,
        train_cache,
        actuator_layers,
        labels,
        args.template_filter,
    )

    guide_codebook = fit_guide_codebook(
        train_records,
        train_rows,
        train_cache,
        guide_layers,
        labels,
    )

    # TEST baseline + guide
    baseline_rows, guide_delta_cache = (
        collect_test_baseline_and_guide(
            model,
            processor,
            layers,
            test_records,
            args,
            answer_words,
            guide_layers,
        )
    )

    write_csv(
        outdir
        / "test_baseline.csv",
        baseline_rows,
    )

    test_record_map = {
        record["sid"]: record
        for record in test_records
    }

    oracle_targets = {
        row["sid"]:
            test_record_map[
                row["sid"]
            ][
                "relation"
            ]
        for row in baseline_rows
    }

    guide_targets = {
        row["sid"]:
            guide_predict(
                guide_delta_cache[
                    row["sid"]
                ],
                guide_codebook,
                guide_layers,
                labels,
            )
        for row in baseline_rows
    }

    guide_rows = []

    for row in baseline_rows:
        sid = row["sid"]

        guide_rows.append({
            "sid": sid,
            "relation": test_record_map[sid]["relation"],
            "guide_pred": guide_targets[sid],
            "guide_correct": int(
                guide_targets[sid]
                == test_record_map[sid]["relation"]
            ),
            "baseline_pred": row["baseline_pred"],
            "baseline_correct": row["baseline_correct"],
        })

    write_csv(
        outdir
        / "test_guide_predictions.csv",
        guide_rows,
    )

    guide_acc = safe_mean(
        row["guide_correct"]
        for row in guide_rows
    )

    baseline_acc = safe_mean(
        row["baseline_correct"]
        for row in baseline_rows
    )

    conditions = [
        (
            "oracle_all_add",
            oracle_targets,
            "add",
            "all",
        ),
        (
            "guide_all_add",
            guide_targets,
            "add",
            "all",
        ),
        (
            "guide_conflict_add",
            guide_targets,
            "add",
            "conflict_only",
        ),
        (
            "guide_all_contrast",
            guide_targets,
            "contrast",
            "all",
        ),
        (
            "guide_conflict_contrast",
            guide_targets,
            "contrast",
            "conflict_only",
        ),
    ]

    summaries = []
    details = []

    for (
        condition,
        targets,
        edit_mode,
        apply_mode,
    ) in conditions:

        rows = run_steering(
            model,
            processor,
            layers,
            actuator_templates,
            actuator_layers,
            test_record_map,
            baseline_rows,
            targets,
            args,
            answer_words,
            edit_mode,
            apply_mode,
            condition,
        )

        details.extend(
            rows
        )

        summaries.append(
            summarize(
                rows,
                condition,
            )
        )

        write_csv(
            outdir
            / "test_steering_details.csv",
            details,
        )

        write_csv(
            outdir
            / "test_summary.csv",
            summaries,
        )

    print(
        "\n"
        + "=" * 170
    )

    print(
        "FINAL TEST — VG2 LEFT/RIGHT LATE CAUSAL REPLICATION"
    )

    print(
        "=" * 170
    )

    print(
        f"model={args.model_id}"
    )

    print(
        f"dataset={args.dataset}"
    )

    print(
        f"decoder_blocks={len(layers)}"
    )

    print(
        f"causal_layers={actuator_layers}"
    )

    print(
        f"guide_layers={guide_layers}"
    )

    print(
        f"template_filter={filter_used}"
    )

    print(
        f"baseline={baseline_acc:.4f}"
    )

    print(
        f"guide_acc={guide_acc:.4f}"
    )

    print()

    print(
        "condition                 | "
        "acc base->edit gain | applied | "
        "W2C/wrong | C2W/correct | net | changed"
    )

    for row in summaries:
        print(
            f"{row['condition']:25s} | "
            f"{row['base_acc']:.4f}->"
            f"{row['edit_acc']:.4f} "
            f"{row['gain']:+.4f} | "
            f"{row['applied']}/{row['applied_rate']:.3f} | "
            f"{row['W2C']}/{row['W2C_rate_wrong']:.3f} | "
            f"{row['C2W']}/{row['C2W_rate_correct']:.3f} | "
            f"{row['net']:+d} | "
            f"{row['changed_rate']:.3f}"
        )

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "dataset": args.dataset,
                "decoder_blocks": len(layers),
                "actuator_layers": actuator_layers,
                "guide_layers": guide_layers,
                "labels": list(labels),
                "train_n": len(train_records),
                "test_n": len(test_records),
                "template_filter": filter_used,
                "baseline_acc": baseline_acc,
                "guide_acc": guide_acc,
                "train_counts": dict(
                    Counter(
                        r["relation"]
                        for r in train_records
                    )
                ),
                "test_counts": dict(
                    Counter(
                        r["relation"]
                        for r in test_records
                    )
                ),
                "important_note": (
                    "Actuator and guide layer locations are reused from the "
                    "previous COCO experiments. Dataset-specific Real-Gray "
                    "relation templates are learned only on this dataset's TRAIN split."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
