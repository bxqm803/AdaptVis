#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InternVL2.5 late last-token causal-direction auto-layer search.

Requires a directory containing:
  vectors.npz
  sample_split_and_generation.csv

The vectors are only used as the final non-oracle guide.
The new causal directions are re-learned from InternVL Real-vs-Gray last-token deltas.

Recommended:
CUDA_VISIBLE_DEVICES=0 python eval_internvl_last_causal_auto_v1.py \
  --model-id OpenGVLab/InternVL2_5-2B \
  --direction-dir output/<internvl_dir_with_vectors.npz> \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --annotation-json data/coco_qa_two_obj.json \
  --data-root data \
  --device cuda:0 \
  --scan-layers all \
  --val-frac 0.25 \
  --top-k-single 5 \
  --window-lengths 2,3,4 \
  --scale 1.0 \
  --max-num-tiles 12 \
  --output-dir output/internvl25_2b_last_causal_auto_v1 \
  --overwrite

Notes:
- InternVL official-style dynamic tiling is used.
- hidden[:, -1, :] on the first LLM prefill pass is the intervention site.
- FIT/VAL come only from original TRAIN; TEST is untouched during layer selection.
- Template filter automatically falls back:
    real_correct_gray_wrong -> real_correct -> all
  if one of the four relations is missing.
"""

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.generation import GenerationMixin

RELS = ("left", "right", "above", "below")
RELSET = set(RELS)
EPS = 1e-12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model-id", default="OpenGVLab/InternVL2_5-2B")
    p.add_argument(
        "--direction-dir",
        default="",
        help=(
            "Optional existing directory containing vectors.npz and "
            "sample_split_and_generation.csv. If omitted or missing, this "
            "script builds both automatically, then continues the causal experiment."
        ),
    )
    p.add_argument(
        "--direction-output-dir",
        default="",
        help=(
            "Where to create vectors.npz when --direction-dir is absent. "
            "Default: <output-dir>/direction_bundle"
        ),
    )
    p.add_argument(
        "--split-source-dir",
        default="",
        help=(
            "Optional directory containing an existing "
            "sample_split_and_generation.csv whose train/test split should be reused."
        ),
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.30,
        help="Train fraction used only when a split must be created from scratch.",
    )
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
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
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
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def norm_rel(x):
    s = str(x).strip().lower()
    if re.search(r"\bleft\b", s):
        return "left"
    if re.search(r"\bright\b", s):
        return "right"
    if re.search(r"\babove\b", s) or re.search(r"\bover\b", s) or re.search(r"\bon top of\b", s):
        return "above"
    if re.search(r"\bbelow\b", s) or re.search(r"\bunder(?:neath)?\b", s) or re.search(r"\bbeneath\b", s):
        return "below"
    return s


def int_list(spec):
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def layer_spec(spec, n):
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(n))
    if spec == "late_half":
        return list(range(n // 2, n))
    out = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            out += list(range(a, b + step, step))
        else:
            out.append(int(piece))
    out = sorted(set(out))
    bad = [x for x in out if x < 0 or x >= n]
    if bad:
        raise ValueError(f"bad layers={bad}; valid 0..{n-1}")
    return out


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# ---------------------------------------------------------------------
# automatic residual Direction bundle builder
# ---------------------------------------------------------------------

def load_all_records_without_split(args):
    """Load all valid COCO-two records before any train/test split exists."""
    with open(args.annotation_json, "r", encoding="utf-8") as f:
        anns = json.load(f)

    prompts = []
    with open(args.prompt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    records = {}

    for row_no, row in enumerate(prompts):
        sid = int(row.get("id", row_no))

        if not (0 <= sid < len(anns)):
            continue

        sub, ref = parse_sr(row.get("question", ""))

        if sub is None or ref is None:
            continue

        gt = norm_rel(row.get("answer", ""))

        if gt not in RELSET:
            continue

        image_id = int(anns[sid][0])

        path = (
            Path(args.data_root)
            / "val2017"
            / f"{image_id:012d}.jpg"
        )

        records[sid] = {
            "sid": sid,
            "gt": gt,
            "subject": sub,
            "reference": ref,
            "image_path": str(path),
        }

    if not records:
        raise RuntimeError("No valid COCO-two records found.")

    return records


def make_stratified_train_test_split(records, train_frac, seed):
    rng = random.Random(seed)

    split = {}

    for relation in RELS:
        xs = sorted(
            sid
            for sid, rec in records.items()
            if rec["gt"] == relation
        )

        rng.shuffle(xs)

        n_train = int(round(len(xs) * float(train_frac)))
        n_train = max(1, min(n_train, len(xs) - 1))

        train_set = set(xs[:n_train])

        for sid in xs:
            split[sid] = "train" if sid in train_set else "test"

    return split


def load_split_from_csv(path):
    rows = read_csv(path)

    return {
        int(row["sample_index"]): str(row.get("split", "")).strip().lower()
        for row in rows
    }


def find_subsequence(sequence, pattern, start=0):
    if not pattern:
        return None

    n = len(pattern)

    for i in range(start, len(sequence) - n + 1):
        if sequence[i:i+n] == pattern:
            return list(range(i, i+n))

    return None


def token_span_for_phrase(tokenizer, input_ids, phrase, start=0):
    """
    Locate an object phrase in the exact InternVL chat input token sequence.
    We try variants because BPE/SentencePiece may absorb preceding whitespace.
    """
    candidates = []

    for variant in [
        " " + phrase,
        phrase,
        "\n" + phrase,
    ]:
        ids = tokenizer(
            variant,
            add_special_tokens=False,
        ).input_ids

        if ids:
            candidates.append(ids)

    for ids in candidates:
        span = find_subsequence(
            input_ids,
            ids,
            start=start,
        )

        if span is not None:
            return span

    # Last fallback: search phrase without its first token, then include one token
    # before it only if necessary. This helps tokenizers whose first word token
    # merges whitespace differently in the full prompt.
    raw = tokenizer(
        phrase,
        add_special_tokens=False,
    ).input_ids

    if len(raw) >= 2:
        span = find_subsequence(
            input_ids,
            raw[1:],
            start=start,
        )

        if span is not None:
            first = max(
                start,
                span[0] - 1,
            )

            return list(
                range(
                    first,
                    span[-1] + 1,
                )
            )

    return None


def build_exact_internvl_query_and_spans(
    model,
    tokenizer,
    rec,
    args,
    num_patches,
):
    """
    Reproduce InternVL2.5 chat() prompt construction closely enough to recover
    subject/reference text token positions in the LLM sequence.

    InternVL chat constructs a conversation prompt, replaces <image> with
    <img> + repeated <IMG_CONTEXT> + </img>, then tokenizes the full query.
    """
    question = (
        "<image>\n"
        + args.prompt_template.format(
            subject=rec["subject"],
            reference=rec["reference"],
        )
    )

    get_conv_template = model.chat.__globals__.get(
        "get_conv_template",
        None,
    )

    if get_conv_template is None:
        raise RuntimeError(
            "Could not access InternVL get_conv_template from model.chat."
        )

    template = get_conv_template(
        model.template
    )

    if hasattr(
        model,
        "system_message",
    ):
        template.system_message = model.system_message

    template.append_message(
        template.roles[0],
        question,
    )

    template.append_message(
        template.roles[1],
        None,
    )

    query = template.get_prompt()

    img_start = "<img>"
    img_end = "</img>"
    img_context = "<IMG_CONTEXT>"

    image_tokens = (
        img_start
        + img_context
        * int(model.num_image_token)
        * int(num_patches)
        + img_end
    )

    query = query.replace(
        "<image>",
        image_tokens,
        1,
    )

    tokenized = tokenizer(
        query,
        return_tensors="pt",
    )

    input_ids = tokenized[
        "input_ids"
    ][0].tolist()

    context_token_id = tokenizer.convert_tokens_to_ids(
        img_context
    )

    context_positions = [
        i
        for i, token_id in enumerate(input_ids)
        if token_id == context_token_id
    ]

    search_start = (
        context_positions[-1] + 1
        if context_positions
        else 0
    )

    subject_span = token_span_for_phrase(
        tokenizer,
        input_ids,
        rec["subject"],
        start=search_start,
    )

    if subject_span is None:
        raise RuntimeError(
            f"Could not locate subject tokens: {rec['subject']!r}"
        )

    reference_span = token_span_for_phrase(
        tokenizer,
        input_ids,
        rec["reference"],
        start=subject_span[-1] + 1,
    )

    if reference_span is None:
        # In unusual prompts reference text may tokenize before a repeated subject
        # occurrence; retry from after the image region.
        reference_span = token_span_for_phrase(
            tokenizer,
            input_ids,
            rec["reference"],
            start=search_start,
        )

    if reference_span is None:
        raise RuntimeError(
            f"Could not locate reference tokens: {rec['reference']!r}"
        )

    return (
        input_ids,
        subject_span,
        reference_span,
    )


class CapturePairDirection:
    """
    Capture q_l = mean(h_subject) - mean(h_reference)
    at every selected decoder block on the first prefill call.
    """

    def __init__(
        self,
        layers,
        selected,
        subject_span,
        reference_span,
    ):
        self.handles = []
        self.states = {}

        self.subject_span = list(subject_span)
        self.reference_span = list(reference_span)

        self.done = {
            l: False
            for l in selected
        }

        for l in selected:
            self.handles.append(
                layers[l].register_forward_hook(
                    self._hook(l)
                )
            )

    def _hook(self, l):
        def hook(_m, _inp, out):
            if self.done[l]:
                return out

            h, _ = get_hidden(out)

            if h.ndim != 3:
                return out

            seq_len = int(h.shape[1])

            if (
                max(self.subject_span) >= seq_len
                or max(self.reference_span) >= seq_len
            ):
                raise RuntimeError(
                    f"L{l}: text span exceeds decoder seq_len={seq_len}. "
                    f"subject={self.subject_span} reference={self.reference_span}"
                )

            hs = h[
                0,
                self.subject_span,
                :
            ].mean(dim=0)

            hr = h[
                0,
                self.reference_span,
                :
            ].mean(dim=0)

            self.states[l] = (
                (hs - hr)
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            self.done[l] = True

            return out

        return hook

    def close(self):
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def collect_pair_direction_one_image(
    model,
    tokenizer,
    layers,
    image,
    rec,
    selected,
    args,
    dtype,
):
    pixels = pixels_from_pil(
        image,
        args,
    )

    (
        _input_ids,
        subject_span,
        reference_span,
    ) = build_exact_internvl_query_and_spans(
        model,
        tokenizer,
        rec,
        args,
        num_patches=int(pixels.shape[0]),
    )

    with CapturePairDirection(
        layers,
        selected,
        subject_span,
        reference_span,
    ) as capture:

        text, pred = generate(
            model,
            tokenizer,
            pixels,
            rec,
            args,
            dtype,
        )

        states = dict(
            capture.states
        )

    del pixels

    return (
        text,
        pred,
        states,
    )


def build_direction_bundle(
    model,
    tokenizer,
    layers,
    args,
    dtype,
    direction_dir,
):
    """
    Build the old-style object-pair residual Direction vectors internally:

      r_l^res =
        [(h_sub-h_ref)_real] -
        [(h_sub-h_ref)_gray]

    for every sample and every decoder layer.

    This creates exactly the two files expected by the causal experiment:
      vectors.npz
      sample_split_and_generation.csv
    """
    direction_dir = Path(
        direction_dir
    )

    direction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectors_path = (
        direction_dir
        / "vectors.npz"
    )

    split_path = (
        direction_dir
        / "sample_split_and_generation.csv"
    )

    all_records = load_all_records_without_split(
        args
    )

    # Reuse an existing split if requested.
    split = None

    if args.split_source_dir:
        source_split = (
            Path(args.split_source_dir)
            / "sample_split_and_generation.csv"
        )

        if not source_split.exists():
            raise FileNotFoundError(
                source_split
            )

        split = load_split_from_csv(
            source_split
        )

        missing = [
            sid
            for sid in all_records
            if sid not in split
        ]

        if missing:
            raise RuntimeError(
                f"split source is missing {len(missing)} dataset samples"
            )

        print(
            f"[direction build] reusing split from {source_split}"
        )

    else:
        split = make_stratified_train_test_split(
            all_records,
            args.train_frac,
            args.seed,
        )

        print(
            f"[direction build] created stratified split "
            f"train_frac={args.train_frac:.3f}"
        )

    selected = list(
        range(
            len(layers)
        )
    )

    sample_ids = sorted(
        all_records
    )

    residual_vectors = []
    labels = []
    metadata_rows = []

    failed = []

    for sid in tqdm(
        sample_ids,
        desc="BUILD residual Direction vectors",
    ):
        rec = all_records[
            sid
        ]

        real = None
        gray = None

        try:
            real = Image.open(
                rec["image_path"]
            ).convert(
                "RGB"
            )

            gray = gray_image(
                real,
                args.gray_value,
            )

            (
                real_text,
                real_pred,
                real_states,
            ) = collect_pair_direction_one_image(
                model,
                tokenizer,
                layers,
                real,
                rec,
                selected,
                args,
                dtype,
            )

            (
                gray_text,
                gray_pred,
                gray_states,
            ) = collect_pair_direction_one_image(
                model,
                tokenizer,
                layers,
                gray,
                rec,
                selected,
                args,
                dtype,
            )

            if (
                len(real_states) != len(selected)
                or len(gray_states) != len(selected)
            ):
                raise RuntimeError(
                    f"incomplete layer capture "
                    f"real={len(real_states)} gray={len(gray_states)} "
                    f"expected={len(selected)}"
                )

            residual = np.stack(
                [
                    (
                        real_states[l]
                        - gray_states[l]
                    ).astype(np.float32)
                    for l in selected
                ],
                axis=0,
            )

            residual_vectors.append(
                residual
            )

            labels.append(
                rec["gt"]
            )

            metadata_rows.append({
                "sample_index":
                    sid,
                "split":
                    split[sid],
                "generation_group":
                    (
                        "correct"
                        if real_pred == rec["gt"]
                        else "wrong"
                    ),
                "relation":
                    rec["gt"],
                "real_pred":
                    real_pred
                    or "",
                "gray_pred":
                    gray_pred
                    or "",
                "real_correct":
                    int(
                        real_pred
                        == rec["gt"]
                    ),
                "gray_correct":
                    int(
                        gray_pred
                        == rec["gt"]
                    ),
                "real_text":
                    real_text,
                "gray_text":
                    gray_text,
            })

        except Exception as exc:
            failed.append(
                (
                    sid,
                    f"{type(exc).__name__}: {exc}",
                )
            )

            tqdm.write(
                f"[DIRECTION BUILD ERROR sid={sid}] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if real is not None:
                real.close()

            if gray is not None:
                gray.close()

            cleanup()

    if failed:
        fail_path = (
            direction_dir
            / "direction_build_failures.csv"
        )

        write_csv(
            fail_path,
            [
                {
                    "sample_index":
                        sid,
                    "error":
                        error,
                }
                for sid, error in failed
            ],
        )

        raise RuntimeError(
            f"Direction bundle build failed for {len(failed)} samples. "
            f"See {fail_path}. Fix span/prompt issues before causal evaluation."
        )

    residual_vectors = np.stack(
        residual_vectors,
        axis=0,
    ).astype(
        np.float32
    )

    saved_sids = np.asarray(
        [
            int(
                row[
                    "sample_index"
                ]
            )
            for row in metadata_rows
        ],
        dtype=np.int64,
    )

    saved_labels = np.asarray(
        labels,
        dtype=object,
    )

    np.savez_compressed(
        vectors_path,
        sample_index=
            saved_sids,
        relation=
            saved_labels,
        residual=
            residual_vectors,
    )

    write_csv(
        split_path,
        metadata_rows,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "BUILT DIRECTION BUNDLE"
    )

    print(
        "=" * 120
    )

    print(
        f"dir={direction_dir}"
    )

    print(
        f"vectors={vectors_path}"
    )

    print(
        f"shape={residual_vectors.shape}"
    )

    print(
        f"split_csv={split_path}"
    )

    print(
        f"TRAIN={sum(row['split']=='train' for row in metadata_rows)} "
        f"TEST={sum(row['split']=='test' for row in metadata_rows)}"
    )

    return direction_dir


def resolve_or_build_direction_dir(
    model,
    tokenizer,
    layers,
    args,
    dtype,
):
    # Explicit existing directory.
    if args.direction_dir:
        d = Path(
            args.direction_dir
        )

        vectors_path = (
            d
            / "vectors.npz"
        )

        split_path = (
            d
            / "sample_split_and_generation.csv"
        )

        if (
            vectors_path.exists()
            and split_path.exists()
        ):
            print(
                f"[direction] reusing existing bundle: {d}"
            )

            return d

        print(
            f"[direction] requested dir is incomplete; building it: {d}"
        )

        return build_direction_bundle(
            model,
            tokenizer,
            layers,
            args,
            dtype,
            d,
        )

    # Default location under current experiment output.
    if args.direction_output_dir:
        d = Path(
            args.direction_output_dir
        )

    else:
        d = (
            Path(args.output_dir)
            / "direction_bundle"
        )

    vectors_path = (
        d
        / "vectors.npz"
    )

    split_path = (
        d
        / "sample_split_and_generation.csv"
    )

    if (
        vectors_path.exists()
        and split_path.exists()
        and not args.overwrite
    ):
        print(
            f"[direction] reusing existing bundle: {d}"
        )

        return d

    return build_direction_bundle(
        model,
        tokenizer,
        layers,
        args,
        dtype,
        d,
    )


# ---------------------------------------------------------------------
# data / old Direction guide
# ---------------------------------------------------------------------

def load_bundle(direction_dir, key):
    root = Path(direction_dir)
    vectors_path = root / "vectors.npz"
    split_path = root / "sample_split_and_generation.csv"

    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    with np.load(vectors_path, allow_pickle=True) as z:
        if key not in z.files:
            raise KeyError(f"{key!r} not found; keys={z.files}")
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray([norm_rel(x) for x in z["relation"]], dtype=object)
        arr = np.asarray(z[key], dtype=np.float32)

    n = len(sids)
    if arr.shape[0] == n:
        vectors = arr
    elif arr.shape[1] == n:
        vectors = np.transpose(arr, (1, 0, 2))
    else:
        raise RuntimeError(f"cannot align vectors {arr.shape} with N={n}")

    split, group = {}, {}
    for row in read_csv(split_path):
        sid = int(row["sample_index"])
        split[sid] = str(row.get("split", "")).strip().lower()
        group[sid] = str(row.get("generation_group", "")).strip().lower()

    return {
        "sids": sids,
        "labels": labels,
        "vectors": vectors,
        "split": split,
        "group": group,
        "sid2i": {int(s): i for i, s in enumerate(sids.tolist())},
    }


def parse_sr(question):
    q = str(question)
    for pat in [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]:
        m = re.search(pat, q, flags=re.I | re.S)
        if m:
            return (
                re.sub(r"\s+", " ", m.group(1)).strip(),
                re.sub(r"\s+", " ", m.group(2)).strip(),
            )
    return None, None


def load_records(args, bundle):
    with open(args.annotation_json, "r", encoding="utf-8") as f:
        anns = json.load(f)

    prompts = []
    with open(args.prompt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    records = {}
    for row_no, row in enumerate(prompts):
        sid = int(row.get("id", row_no))
        if sid not in bundle["split"] or not (0 <= sid < len(anns)):
            continue
        sub, ref = parse_sr(row.get("question", ""))
        if sub is None:
            continue
        gt = norm_rel(row.get("answer", ""))
        if gt not in RELSET:
            continue
        image_id = int(anns[sid][0])
        path = Path(args.data_root) / "val2017" / f"{image_id:012d}.jpg"
        records[sid] = {
            "sid": sid,
            "gt": gt,
            "subject": sub,
            "reference": ref,
            "split": bundle["split"][sid],
            "image_path": str(path),
        }

    print(f"[records] N={len(records)}")
    return records


def fit_val_split(records, args):
    rng = random.Random(args.seed)
    fit, val = [], []
    for relation in RELS:
        xs = sorted(
            sid for sid, rec in records.items()
            if rec["split"] == "train" and rec["gt"] == relation
        )
        rng.shuffle(xs)
        if len(xs) < 2:
            raise RuntimeError(f"too few TRAIN samples for {relation}")
        nv = max(args.min_val_per_relation, int(round(len(xs) * args.val_frac)))
        nv = min(nv, len(xs) - 1)
        val += xs[:nv]
        fit += xs[nv:]
    return sorted(fit), sorted(val)


# ---------------------------------------------------------------------
# official-style InternVL preprocessing
# ---------------------------------------------------------------------

def build_transform(size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def closest_ratio(ar, ratios, width, height, size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(ar - tar)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff:
            if area > 0.5 * size * size * ratio[0] * ratio[1]:
                best = ratio
    return best


def dynamic_preprocess(image, size=448, max_num=12):
    w, h = image.size
    ar = w / h
    ratios = set(
        (i, j)
        for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 1 <= i * j <= max_num
    )
    ratios = sorted(ratios, key=lambda x: x[0] * x[1])
    ratio = closest_ratio(ar, ratios, w, h, size)
    tw, th = size * ratio[0], size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    cols = tw // size
    images = []
    for i in range(blocks):
        box = (
            (i % cols) * size,
            (i // cols) * size,
            ((i % cols) + 1) * size,
            ((i // cols) + 1) * size,
        )
        images.append(resized.crop(box))
    if len(images) != 1:
        images.append(image.resize((size, size)))
    return images


def pixels_from_pil(image, args):
    transform = build_transform(args.input_size)
    tiles = dynamic_preprocess(
        image,
        size=args.input_size,
        max_num=args.max_num_tiles,
    )
    return torch.stack([transform(tile) for tile in tiles])


def gray_image(real, value):
    v = int(max(0, min(255, value)))
    return Image.new("RGB", real.size, (v, v, v))


# ---------------------------------------------------------------------
# InternVL model + hook
# ---------------------------------------------------------------------

def get_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_layers(model):
    for path in [
        "language_model.model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "model.language_model.layers",
    ]:
        try:
            layers = get_path(model, path)
            if len(layers):
                return layers, path
        except Exception:
            pass
    raise RuntimeError("cannot resolve InternVL LLM decoder layers")


def ensure_language_model_generation_mixin(model):
    """
    Compatibility for InternVL2.5 models whose language model is InternLM2.

    In Transformers >= 4.50, PreTrainedModel no longer automatically inherits
    GenerationMixin. Older InternVL remote code still calls
    self.language_model.generate(...), so InternLM2ForCausalLM can lose
    generate() even though its generation hooks are implemented.

    This patches only the runtime class; checkpoint weights/files are unchanged.
    """
    lm = getattr(model, "language_model", None)

    if lm is None:
        raise RuntimeError("InternVL model has no language_model attribute.")

    if hasattr(lm, "generate"):
        print(
            f"[compat] language_model={lm.__class__.__name__} "
            "already has generate()"
        )
        return model

    original_cls = lm.__class__

    patched_cls = type(
        original_cls.__name__ + "WithGenerationMixin",
        (original_cls, GenerationMixin),
        {},
    )

    try:
        lm.__class__ = patched_cls
    except TypeError as exc:
        raise RuntimeError(
            "Could not attach GenerationMixin at runtime. "
            "Fallback: install transformers<=4.49.0."
        ) from exc

    if not hasattr(lm, "generate"):
        raise RuntimeError("GenerationMixin patch failed to expose generate().")

    print(
        f"[compat] patched {original_cls.__name__} -> "
        f"{patched_cls.__name__}; generate() restored"
    )

    return model


def load_model(args):
    dtype = dtype_from_name(args.dtype)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=bool(args.use_flash_attn),
        trust_remote_code=True,
    ).eval().to(args.device)

    model = ensure_language_model_generation_mixin(model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    return model, tokenizer, dtype


def parse_pred(text):
    s = str(text).lower()
    hits = []
    for relation, pat in [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("above", r"\bon top of\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
        ("below", r"\bbeneath\b"),
    ]:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), relation))
    return sorted(hits)[0][1] if hits else None


def generate(model, tokenizer, pixels, rec, args, dtype):
    question = (
        "<image>\n"
        + args.prompt_template.format(
            subject=rec["subject"],
            reference=rec["reference"],
        )
    )
    pixels = pixels.to(args.device, dtype=dtype)
    config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    with torch.inference_mode():
        response = model.chat(
            tokenizer,
            pixels,
            question,
            config,
        )
    text = str(response).strip()
    return text, parse_pred(text)


def get_hidden(output):
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
    raise RuntimeError(f"cannot extract hidden from {type(output)}")


def put_hidden(output, desc, hidden):
    kind, idx = desc
    if kind == "tensor":
        return hidden
    xs = list(output)
    xs[idx] = hidden
    return tuple(xs) if kind == "tuple" else xs


class CaptureLast:
    def __init__(self, layers, selected):
        self.handles = []
        self.states = {}
        self.done = {l: False for l in selected}
        for l in selected:
            self.handles.append(
                layers[l].register_forward_hook(self._hook(l))
            )

    def _hook(self, l):
        def hook(_m, _inp, out):
            if self.done[l]:
                return out
            h, _ = get_hidden(out)
            if h.ndim != 3:
                return out
            self.states[l] = (
                h[0, -1, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            self.done[l] = True
            return out
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SteerLast:
    def __init__(self, layers, templates, selected, target, scale,
                 mode="add", source=None):
        self.handles = []
        self.done = {l: False for l in selected}
        self.templates = templates
        self.target = target
        self.scale = scale
        self.mode = mode
        self.source = source
        for l in selected:
            self.handles.append(
                layers[l].register_forward_hook(self._hook(l))
            )

    def vector(self, l):
        t = np.asarray(
            self.templates[l]["shared"][self.target],
            dtype=np.float32,
        )
        if self.mode == "add" or self.source not in RELSET:
            return self.scale * t
        s = np.asarray(
            self.templates[l]["shared"][self.source],
            dtype=np.float32,
        )
        return self.scale * (t - s)

    def _hook(self, l):
        def hook(_m, _inp, out):
            if self.done[l]:
                return out
            h, desc = get_hidden(out)
            if h.ndim != 3:
                return out
            v = torch.as_tensor(
                self.vector(l),
                device=h.device,
                dtype=h.dtype,
            )
            y = h.clone()
            y[:, -1, :] += v[None, :]
            self.done[l] = True
            return put_hidden(out, desc, y)
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------
# delta collection + template fit
# ---------------------------------------------------------------------

def collect_deltas(model, tokenizer, layers, records, sids, selected,
                   args, dtype, desc):
    rows = []
    deltas = {}

    for sid in tqdm(sids, desc=desc):
        rec = records[sid]
        real = gray = None
        try:
            real = Image.open(rec["image_path"]).convert("RGB")
            gray = gray_image(real, args.gray_value)

            def one(image):
                pixels = pixels_from_pil(image, args)
                with CaptureLast(layers, selected) as cap:
                    text, pred = generate(
                        model, tokenizer, pixels, rec, args, dtype
                    )
                    states = dict(cap.states)
                del pixels
                return text, pred, states

            rt, rp, rs = one(real)
            gt, gp, gs = one(gray)

            rc = int(rp == rec["gt"])
            gray_ok = int(gp == rec["gt"])

            deltas[sid] = {
                l: (rs[l] - gs[l]).astype(np.float32)
                for l in selected
                if l in rs and l in gs
            }

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "real_pred": rp or "",
                "gray_pred": gp or "",
                "real_correct": rc,
                "gray_correct": gray_ok,
                "real_text": rt,
                "gray_text": gt,
            })

        except Exception as e:
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "error": f"{type(e).__name__}: {e}",
            })
            tqdm.write(f"[delta sid={sid}] {type(e).__name__}: {e}")

        finally:
            if real is not None:
                real.close()
            if gray is not None:
                gray.close()
            cleanup()

    return deltas, rows


def filter_ok(row, mode):
    rc = int(row.get("real_correct", 0))
    gray_ok = int(row.get("gray_correct", 0))
    if mode == "all":
        return True
    if mode == "real_correct":
        return bool(rc)
    return bool(rc) and not bool(gray_ok)


def filter_sequence(requested):
    if requested == "real_correct_gray_wrong":
        return ["real_correct_gray_wrong", "real_correct", "all"]
    if requested == "real_correct":
        return ["real_correct", "all"]
    return ["all"]


def fit_templates(deltas, rows, records, sids, selected, requested):
    row_map = {
        int(r["sid"]): r
        for r in rows
        if "sid" in r
    }

    for mode in filter_sequence(requested):
        bags = {
            l: {r: [] for r in RELS}
            for l in selected
        }

        used = []

        for sid in sids:
            if sid not in deltas or sid not in row_map:
                continue
            if not filter_ok(row_map[sid], mode):
                continue
            if any(l not in deltas[sid] for l in selected):
                continue

            relation = records[sid]["gt"]
            used.append(sid)
            for l in selected:
                bags[l][relation].append(deltas[sid][l])

        missing = [
            (l, r)
            for l in selected
            for r in RELS
            if not bags[l][r]
        ]

        if missing:
            print(
                f"[template] filter={mode}: "
                f"missing {len(missing)} layer/relation cells -> relax"
            )
            continue

        templates = {}
        summary = []

        for l in selected:
            mus = {
                r: np.stack(bags[l][r]).mean(0).astype(np.float32)
                for r in RELS
            }
            global_mu = np.stack([mus[r] for r in RELS]).mean(0).astype(np.float32)
            shared = {
                r: (mus[r] - global_mu).astype(np.float32)
                for r in RELS
            }
            templates[l] = {
                "relation_mean": mus,
                "global_mean": global_mu,
                "shared": shared,
            }
            summary.append({
                "filter": mode,
                "layer": l,
                **{f"n_{r}": len(bags[l][r]) for r in RELS},
                "global_norm": float(np.linalg.norm(global_mu)),
                **{
                    f"{r}_norm": float(np.linalg.norm(shared[r]))
                    for r in RELS
                },
            })

        print(
            f"[template] using filter={mode} N={len(used)} | "
            + " ".join(
                f"{r}={sum(records[s]['gt']==r for s in used)}"
                for r in RELS
            )
        )
        return templates, summary, mode

    raise RuntimeError("could not fit templates even with filter=all")


# ---------------------------------------------------------------------
# baseline / steering
# ---------------------------------------------------------------------

def baseline(model, tokenizer, records, sids, args, dtype, desc):
    rows = []

    for sid in tqdm(sids, desc=desc):
        rec = records[sid]
        image = None
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
            pixels = pixels_from_pil(image, args)
            text, pred = generate(
                model, tokenizer, pixels, rec, args, dtype
            )
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "pred": pred or "",
                "correct": int(pred == rec["gt"]),
                "text": text,
            })
            del pixels
        except Exception as e:
            tqdm.write(f"[baseline sid={sid}] {type(e).__name__}: {e}")
        finally:
            if image is not None:
                image.close()
            cleanup()

    return rows


def steer(model, tokenizer, layers, templates, records, base_rows,
          selected, targets, args, dtype, condition,
          mode="add", apply_mode="all"):
    rows = []

    for base in tqdm(base_rows, desc=condition):
        sid = int(base["sid"])
        rec = records[sid]
        bp = norm_rel(base["pred"])
        target = targets[sid]

        do_edit = True
        if apply_mode == "conflict_only":
            do_edit = (
                bp in RELSET and target in RELSET and bp != target
            )

        if not do_edit:
            rows.append({
                "condition": condition,
                "sid": sid,
                "gt": rec["gt"],
                "target": target,
                "baseline_pred": bp or "",
                "edit_pred": bp or "",
                "baseline_correct": int(base["correct"]),
                "edit_correct": int(base["correct"]),
                "applied": 0,
                "W2C": 0,
                "C2W": 0,
                "changed": 0,
            })
            continue

        image = None
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
            pixels = pixels_from_pil(image, args)

            with SteerLast(
                layers, templates, selected, target,
                args.scale, mode=mode, source=bp,
            ):
                text, pred = generate(
                    model, tokenizer, pixels, rec, args, dtype
                )

            bc = int(base["correct"])
            ec = int(pred == rec["gt"])

            rows.append({
                "condition": condition,
                "sid": sid,
                "gt": rec["gt"],
                "target": target,
                "baseline_pred": bp or "",
                "edit_pred": pred or "",
                "baseline_correct": bc,
                "edit_correct": ec,
                "applied": 1,
                "W2C": int(not bc and ec),
                "C2W": int(bc and not ec),
                "changed": int((pred or "") != (bp or "")),
                "text": text,
            })
            del pixels

        except Exception as e:
            tqdm.write(f"[steer sid={sid} {condition}] {type(e).__name__}: {e}")

        finally:
            if image is not None:
                image.close()
            cleanup()

    return rows


def summarize(rows, condition, layers_text=""):
    n = len(rows)
    ba = mean(r["baseline_correct"] for r in rows)
    ea = mean(r["edit_correct"] for r in rows)
    nw = sum(1 - int(r["baseline_correct"]) for r in rows)
    nc = n - nw
    w2c = sum(int(r["W2C"]) for r in rows)
    c2w = sum(int(r["C2W"]) for r in rows)

    return {
        "condition": condition,
        "layers": layers_text,
        "N": n,
        "base_acc": ba,
        "edit_acc": ea,
        "gain": ea - ba,
        "W2C": w2c,
        "W2C_rate": w2c / nw if nw else float("nan"),
        "C2W": c2w,
        "C2W_rate": c2w / nc if nc else float("nan"),
        "net": w2c - c2w,
        "changed_rate": mean(r["changed"] for r in rows),
    }


def candidate_windows(scan, anchors, lengths):
    allowed = set(scan)
    out = set()
    for anchor in anchors:
        for length in lengths:
            for start in range(anchor - length + 1, anchor + 1):
                w = tuple(range(start, start + length))
                if anchor in w and all(l in allowed for l in w):
                    out.add(w)
    return sorted(out, key=lambda w: (len(w), w[0]))


# ---------------------------------------------------------------------
# old Direction guide
# ---------------------------------------------------------------------

def fit_guide(bundle, train_sids, layers, controls):
    train_set = set(train_sids)

    def indices_for(mode):
        idx = []
        for i, sid in enumerate(bundle["sids"].tolist()):
            sid = int(sid)
            if sid not in train_set:
                continue
            if mode == "correct" and bundle["group"].get(sid) != "correct":
                continue
            idx.append(i)
        return idx

    modes = [controls] if controls == "all" else ["correct", "all"]

    for mode in modes:
        idx = indices_for(mode)
        if not idx:
            continue

        cb = {}
        good = True

        for l in layers:
            X = bundle["vectors"][idx, l].astype(np.float64)
            Y = bundle["labels"][idx]
            center = X.mean(0)
            protos = {}

            for relation in RELS:
                mask = Y == relation
                if not np.any(mask):
                    good = False
                    break
                mu = (X[mask] - center).mean(0)
                n = np.linalg.norm(mu)
                if n <= EPS:
                    good = False
                    break
                protos[relation] = (mu / n).astype(np.float32)

            if not good:
                break

            cb[l] = {
                "center": center.astype(np.float32),
                "protos": protos,
            }

        if good:
            print(f"[guide] using train_controls={mode}")
            return cb, mode

    raise RuntimeError("could not fit Direction guide")


def guide_pred(bundle, cb, sid, layers):
    if sid not in bundle["sid2i"]:
        return None
    i = bundle["sid2i"][sid]
    votes = {r: 0 for r in RELS}
    score_sum = {r: 0.0 for r in RELS}

    for l in layers:
        q = bundle["vectors"][i, l].astype(np.float64)
        q -= cb[l]["center"].astype(np.float64)
        q /= max(np.linalg.norm(q), EPS)

        scores = {
            r: float(np.dot(q, cb[l]["protos"][r]))
            for r in RELS
        }
        pred = max(scores, key=scores.get)
        votes[pred] += 1
        for r in RELS:
            score_sum[r] += scores[r]

    mv = max(votes.values())
    tied = [r for r in RELS if votes[r] == mv]
    return tied[0] if len(tied) == 1 else max(
        tied, key=lambda r: score_sum[r]
    )


def choose_guide(bundle, fit_sids, val_sids, lengths, controls):
    n_layers = bundle["vectors"].shape[1]
    all_layers = list(range(n_layers))
    cb, used_mode = fit_guide(
        bundle, fit_sids, all_layers, controls
    )

    rows = []
    for length in lengths:
        if not (1 <= length <= n_layers):
            continue
        for start in range(n_layers - length + 1):
            window = list(range(start, start + length))
            correct = total = 0

            for sid in val_sids:
                if sid not in bundle["sid2i"]:
                    continue
                p = guide_pred(bundle, cb, sid, window)
                gt = str(bundle["labels"][bundle["sid2i"][sid]])
                correct += int(p == gt)
                total += 1

            rows.append({
                "layers": ",".join(map(str, window)),
                "length": length,
                "start": start,
                "end": start + length - 1,
                "N": total,
                "acc": correct / total if total else float("nan"),
            })

    best = max(
        rows,
        key=lambda r: (float(r["acc"]), -int(r["length"])),
    )
    selected = [int(x) for x in best["layers"].split(",")]
    return selected, rows, best, used_mode


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model first because vectors.npz is now built automatically when absent.
    model, tokenizer, dtype = load_model(args)
    layers, layer_path = resolve_layers(model)
    n_layers = len(layers)
    scan = layer_spec(args.scan_layers, n_layers)

    direction_dir = resolve_or_build_direction_dir(
        model,
        tokenizer,
        layers,
        args,
        dtype,
    )

    bundle = load_bundle(
        direction_dir,
        args.direction_key,
    )

    records = load_records(
        args,
        bundle,
    )

    fit_sids, val_sids = fit_val_split(
        records,
        args,
    )

    test_sids = sorted(
        sid for sid, rec in records.items()
        if rec["split"] == "test"
    )

    if args.max_fit is not None:
        fit_sids = fit_sids[:args.max_fit]
    if args.max_val is not None:
        val_sids = val_sids[:args.max_val]
    if args.max_test is not None:
        test_sids = test_sids[:args.max_test]

    print("\n" + "=" * 125)
    print(f"MODEL={args.model_id}")
    print(f"decoder={layer_path}")
    print(f"decoder_blocks={n_layers}")
    print(f"scan={scan}")
    print(f"FIT={len(fit_sids)} VAL={len(val_sids)} TEST={len(test_sids)}")
    print("=" * 125)

    # FIT causal templates at all candidate layers.
    fit_deltas, fit_rows = collect_deltas(
        model, tokenizer, layers, records, fit_sids,
        scan, args, dtype, "FIT Real-Gray deltas"
    )
    write_csv(outdir / "fit_real_gray_generation.csv", fit_rows)

    fit_template_bank, fit_summary, fit_filter = fit_templates(
        fit_deltas, fit_rows, records, fit_sids,
        scan, args.template_filter,
    )
    write_csv(outdir / "fit_template_summary.csv", fit_summary)

    # VAL baseline.
    val_base = baseline(
        model, tokenizer, records, val_sids,
        args, dtype, "VAL baseline"
    )
    write_csv(outdir / "val_baseline.csv", val_base)

    val_targets = {
        int(r["sid"]): records[int(r["sid"])]["gt"]
        for r in val_base
    }

    # Single layer scan.
    single_rows = []
    for l in scan:
        name = f"val_oracle_L{l:02d}"
        rows = steer(
            model, tokenizer, layers, fit_template_bank, records,
            val_base, [l], val_targets, args, dtype,
            name, mode="add", apply_mode="all",
        )
        s = summarize(rows, name, str(l))
        s["window_length"] = 1
        single_rows.append(s)
        write_csv(outdir / "val_single_layer_scan.csv", single_rows)

    ranked = sorted(
        single_rows,
        key=lambda r: (
            float(r["edit_acc"]),
            int(r["net"]),
            -int(r["C2W"]),
        ),
        reverse=True,
    )
    top_layers = [
        int(r["layers"])
        for r in ranked[:args.top_k_single]
    ]

    # Contiguous windows.
    windows = candidate_windows(
        scan,
        top_layers,
        int_list(args.window_lengths),
    )
    window_rows = []

    for w in windows:
        txt = ",".join(map(str, w))
        name = "val_oracle_" + "_".join(f"L{x:02d}" for x in w)
        rows = steer(
            model, tokenizer, layers, fit_template_bank, records,
            val_base, list(w), val_targets, args, dtype,
            name, mode="add", apply_mode="all",
        )
        s = summarize(rows, name, txt)
        s["window_length"] = len(w)
        window_rows.append(s)
        write_csv(outdir / "val_window_scan.csv", window_rows)

    best = max(
        single_rows + window_rows,
        key=lambda r: (
            float(r["edit_acc"]),
            int(r["net"]),
            -int(r["window_length"]),
        ),
    )
    causal_layers = [
        int(x) for x in str(best["layers"]).split(",")
    ]

    print("\nTop single layers:")
    for r in ranked[:10]:
        print(
            f"  L{int(r['layers']):02d} "
            f"{r['base_acc']:.4f}->{r['edit_acc']:.4f} "
            f"{r['gain']:+.4f} "
            f"W2C={r['W2C']} C2W={r['C2W']}"
        )

    print(
        f"\nSELECTED CAUSAL WINDOW = {causal_layers} | "
        f"VAL {best['base_acc']:.4f}->{best['edit_acc']:.4f} "
        f"{best['gain']:+.4f}"
    )

    # Select non-oracle Direction guide on FIT/VAL.
    guide_layers, guide_scan, guide_best, guide_mode = choose_guide(
        bundle,
        fit_sids,
        val_sids,
        int_list(args.guide_window_lengths),
        args.guide_train_controls,
    )
    write_csv(outdir / "val_guide_window_scan.csv", guide_scan)

    print(
        f"SELECTED GUIDE WINDOW = {guide_layers} | "
        f"VAL guide acc={guide_best['acc']:.4f}"
    )

    # Refit new causal directions on full TRAIN.
    full_train = sorted(set(fit_sids + val_sids))
    full_deltas, full_rows = collect_deltas(
        model, tokenizer, layers, records, full_train,
        causal_layers, args, dtype, "FULL TRAIN refit"
    )
    write_csv(outdir / "full_train_real_gray_generation.csv", full_rows)

    final_templates, final_summary, final_filter = fit_templates(
        full_deltas, full_rows, records, full_train,
        causal_layers, args.template_filter,
    )
    write_csv(outdir / "final_template_summary.csv", final_summary)

    # Refit guide on full TRAIN.
    final_guide, final_guide_mode = fit_guide(
        bundle,
        full_train,
        guide_layers,
        args.guide_train_controls,
    )

    # Untouched TEST.
    test_base = baseline(
        model, tokenizer, records, test_sids,
        args, dtype, "TEST baseline"
    )
    write_csv(outdir / "test_baseline.csv", test_base)

    oracle_targets = {
        int(r["sid"]): records[int(r["sid"])]["gt"]
        for r in test_base
    }

    guide_targets = {}
    guide_rows = []

    for r in test_base:
        sid = int(r["sid"])
        p = guide_pred(bundle, final_guide, sid, guide_layers)
        guide_targets[sid] = p
        guide_rows.append({
            "sid": sid,
            "gt": records[sid]["gt"],
            "baseline_pred": r["pred"],
            "baseline_correct": int(r["correct"]),
            "guide_pred": p or "",
            "guide_correct": int(p == records[sid]["gt"]),
            "conflict": int(
                p in RELSET
                and norm_rel(r["pred"]) in RELSET
                and p != norm_rel(r["pred"])
            ),
        })

    write_csv(outdir / "test_guide_predictions.csv", guide_rows)

    conditions = [
        ("oracle_all_add", oracle_targets, "add", "all"),
        ("guide_all_add", guide_targets, "add", "all"),
        ("guide_conflict_add", guide_targets, "add", "conflict_only"),
        ("guide_all_contrast", guide_targets, "contrast", "all"),
        ("guide_conflict_contrast", guide_targets, "contrast", "conflict_only"),
    ]

    details, summaries = [], []

    for name, targets, mode, apply_mode in conditions:
        rows = steer(
            model, tokenizer, layers, final_templates, records,
            test_base, causal_layers, targets, args, dtype,
            name, mode=mode, apply_mode=apply_mode,
        )
        details += rows
        summaries.append(
            summarize(
                rows,
                name,
                ",".join(map(str, causal_layers)),
            )
        )
        write_csv(outdir / "test_steering_details.csv", details)
        write_csv(outdir / "test_summary.csv", summaries)

    base_acc = mean(r["correct"] for r in test_base)
    guide_acc = mean(r["guide_correct"] for r in guide_rows)

    print("\n" + "=" * 155)
    print("FINAL TEST — INTERNVL AUTO-SELECTED LAST-TOKEN CAUSAL WINDOW")
    print("=" * 155)
    print(f"model={args.model_id}")
    print(f"decoder_blocks={n_layers}")
    print(f"causal_layers={causal_layers}")
    print(f"template_filter_fit={fit_filter}")
    print(f"template_filter_final={final_filter}")
    print(f"guide_layers={guide_layers}")
    print(f"baseline={base_acc:.4f}")
    print(f"guide_acc={guide_acc:.4f}")
    print()
    print(
        "condition                 | acc base->edit gain | "
        "W2C/wrong | C2W/correct | net | changed"
    )

    for r in summaries:
        print(
            f"{r['condition']:25s} | "
            f"{r['base_acc']:.4f}->{r['edit_acc']:.4f} "
            f"{r['gain']:+.4f} | "
            f"{r['W2C']}/{r['W2C_rate']:.3f} | "
            f"{r['C2W']}/{r['C2W_rate']:.3f} | "
            f"{r['net']:+d} | "
            f"{r['changed_rate']:.3f}"
        )

    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "direction_dir": str(direction_dir),
                "decoder_blocks": n_layers,
                "fit_n": len(fit_sids),
                "val_n": len(val_sids),
                "test_n": len(test_sids),
                "causal_layers": causal_layers,
                "guide_layers": guide_layers,
                "fit_template_filter": fit_filter,
                "final_template_filter": final_filter,
                "guide_fit_mode": final_guide_mode,
                "test_baseline_acc": base_acc,
                "test_guide_acc": guide_acc,
                "scale": args.scale,
                "max_num_tiles": args.max_num_tiles,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
