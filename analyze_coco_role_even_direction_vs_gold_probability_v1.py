#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two: role-even object-pair relation directions vs. gold relation-word probability.

Goal
----
For the original question

    Q_AB: Where is A in relation to B?

and the swapped question

    Q_BA: Where is B in relation to A?

extract object-token hidden states at every selected decoder layer:

    h_A,S^AB, h_B,R^AB, h_B,S^BA, h_A,R^BA

Align the swapped question back to the physical A-B order and construct:

    r_original = h_A,S^AB - h_B,R^AB

    r_swap_aligned = h_A,R^BA - h_B,S^BA

    r_role_even = 0.5 * (r_original + r_swap_aligned)
                = 0.5 * (h_A,S^AB + h_A,R^BA)
                  - 0.5 * (h_B,R^AB + h_B,S^BA)

The role-even vector averages each physical object over subject/reference roles.
It is not assumed to be a pure semantic or pure spatial vector; this script tests
whether it gives a cleaner held-out relation representation.

For each layer and requested representation:

1. Split samples into 20% TRAIN and 80% TEST (default), stratified by unordered
   object-pair groups.
2. Fit one independent hidden-space direction per relation on TRAIN only:

       center = mean(r_train)
       d_k = normalize(mean(r_train[y=k] - center))

   No opposite-axis or orthogonality constraint is imposed.
3. On TEST, compute cosine similarity to all learned relation directions:

       score_k = cos(r_test - center, d_k)

4. Report held-out four-class accuracy and, within each fixed gold relation,
   correlate:

       gold direction cosine / gold-vs-best-other cosine margin

   with the previously computed teacher-forced probability of the same gold
   relation word.

The probability CSV should normally be produced by:

    analyze_coco_gold_direction_cosine_vs_gold_word_probability_v4.py

Main outputs
------------
<output-dir>/
    config.json
    split_assignments.csv
    extraction_errors.jsonl
    state_cache/<model>.npz
    sample_test_metrics.csv
    correlation_by_same_gold.csv
    layer_summary.csv
    best_layers.csv
    learned_directions/<model>_<representation>_repeat<R>.npz

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_coco_role_even_direction_vs_gold_probability_v1.py \
  --models qwen2-2b,qwen-3b,qwen-7b,llava-7b,llava-13b \
  --probability-csv \
    output/coco_gold_direction_cosine_vs_gold_word_probability_v4/sample_gold_probability.csv \
  --train-ratio 0.2 \
  --representations role_even \
  --layers all \
  --device cuda:0 \
  --output-dir output/coco_role_even_direction_vs_gold_probability_v1
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scipy is required: {exc}")


SCRIPT_VERSION = "coco-role-even-direction-vs-gold-probability-v1"
EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
DISPLAY_RELATION = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
VALID_REPRESENTATIONS = (
    "original",
    "swap_aligned",
    "role_even",
)
PROBABILITY_COLUMNS = (
    "gold_candidate_word_probability",
    "gold_candidate_word_geometric_mean_probability",
    "gold_candidate_first_token_probability",
    "gold_candidate_word_logprob",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
        help="Existing helper script in the AdaptVis repository root.",
    )
    p.add_argument(
        "--models",
        default="qwen2-2b,qwen-3b,qwen-7b,llava-7b,llava-13b",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    p.add_argument(
        "--layers",
        default="all",
        help="all or comma-separated zero-based decoder block indices.",
    )
    p.add_argument(
        "--object-state",
        default="last",
        choices=("last", "mean"),
        help="Use the final sub-token or mean over the object phrase.",
    )
    p.add_argument(
        "--representations",
        default="original,role_even",
        help="Comma-separated subset of original,swap_aligned,role_even.",
    )
    p.add_argument(
        "--probability-csv",
        default=(
            "output/coco_gold_direction_cosine_vs_gold_word_probability_v4/"
            "sample_gold_probability.csv"
        ),
    )
    p.add_argument(
        "--probability-column",
        default="gold_candidate_word_probability",
        choices=PROBABILITY_COLUMNS,
    )
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument(
        "--split-unit",
        default="pair",
        choices=("pair", "sample"),
        help=(
            "pair keeps the same unordered object-name pair in one split; "
            "sample performs relation-stratified sample splitting."
        ),
    )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save layer curves and best-layer same-gold scatter plots.",
    )
    p.add_argument(
        "--overwrite-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def import_module(path: Path, name: str) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(name, None)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def parse_csv_list(value: str) -> List[str]:
    return list(dict.fromkeys(x.strip() for x in str(value).split(",") if x.strip()))


def parse_layers(value: str, n_layers: int) -> List[int]:
    text = str(value).strip().lower()
    if text == "all":
        return list(range(n_layers))
    layers = sorted({int(x.strip().lstrip("lL")) for x in text.split(",") if x.strip()})
    invalid = [x for x in layers if x < 0 or x >= n_layers]
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; decoder has {n_layers} blocks")
    if not layers:
        raise ValueError("No layers selected")
    return layers


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, EPS)


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def model_load(step1: Any, spec: Any, device: str, attn_impl: str) -> Tuple[Any, Any]:
    model_cls = getattr(step1.transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={step1.transformers.__version__} has no {spec.model_class}"
        )

    kwargs: Dict[str, Any] = {
        "dtype": step1.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": device},
        "attn_implementation": attn_impl,
    }
    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()

    processor = step1.AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    step1.configure_processor(model, processor)
    return model, processor


def span_positions(span: Sequence[int], mode: str) -> List[int]:
    start, end = int(span[0]), int(span[1])
    if mode == "last":
        return [end]
    if mode == "mean":
        return list(range(start, end + 1))
    raise ValueError(mode)


def extract_prompt_object_states(
    *,
    step1: Any,
    model: Any,
    processor: Any,
    batch: Mapping[str, Any],
    subject: str,
    reference: str,
    selected_layers: Sequence[int],
    object_state: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return subject/reference states with shape [selected_layers, hidden]."""
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = step1.locate_object_spans(
        processor.tokenizer,
        input_ids,
        subject,
        reference,
    )
    subject_positions = span_positions(subject_span, object_state)
    reference_positions = span_positions(reference_span, object_state)

    with torch.inference_mode():
        outputs = model(
            **dict(batch),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = step1.hidden_tuple(outputs)
    if len(hidden_states) < max(selected_layers) + 2:
        raise RuntimeError(
            f"Hidden tuple has {len(hidden_states)} entries but requested "
            f"decoder layer {max(selected_layers)}"
        )

    subject_rows: List[np.ndarray] = []
    reference_rows: List[np.ndarray] = []
    for layer in selected_layers:
        # hidden_states[0] is embedding output; decoder block L is [L + 1].
        hidden = hidden_states[int(layer) + 1]
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
            raise RuntimeError(f"Unexpected hidden shape at L{layer}: {tuple(hidden.shape)}")
        if int(hidden.shape[1]) != len(input_ids):
            raise RuntimeError(
                f"Token/hidden length mismatch at L{layer}: "
                f"input={len(input_ids)} hidden={hidden.shape[1]}"
            )
        subject_tensor = hidden[0, subject_positions].float().mean(dim=0)
        reference_tensor = hidden[0, reference_positions].float().mean(dim=0)
        subject_rows.append(subject_tensor.detach().cpu().numpy())
        reference_rows.append(reference_tensor.detach().cpu().numpy())

    del outputs, hidden_states
    return (
        np.stack(subject_rows, axis=0).astype(np.float32),
        np.stack(reference_rows, axis=0).astype(np.float32),
    )


def cache_metadata_matches(
    metadata: Mapping[str, Any],
    *,
    model_name: str,
    selected_layers: Sequence[int],
    object_state: str,
) -> bool:
    return (
        str(metadata.get("model")) == str(model_name)
        and [int(x) for x in metadata.get("selected_layers", [])]
        == [int(x) for x in selected_layers]
        and str(metadata.get("object_state")) == str(object_state)
    )


def load_cache(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        result = {key: z[key] for key in z.files}
    metadata_text = str(result["metadata_json"].item())
    result["metadata"] = json.loads(metadata_text)
    return result


def extract_model_cache(
    *,
    args: argparse.Namespace,
    step1: Any,
    model_name: str,
    spec: Any,
    records: Sequence[Any],
    prompt_rows: Mapping[int, Mapping[str, Any]],
    cache_path: Path,
    errors_path: Path,
) -> Dict[str, Any]:
    model, processor = model_load(step1, spec, args.device, args.attn_impl)
    decoder_layers, decoder_path = step1.resolve_decoder_layers(model)
    selected_layers = parse_layers(args.layers, len(decoder_layers))

    if cache_path.exists() and not args.overwrite_cache:
        cache = load_cache(cache_path)
        if not cache_metadata_matches(
            cache["metadata"],
            model_name=model_name,
            selected_layers=selected_layers,
            object_state=args.object_state,
        ):
            raise RuntimeError(
                f"Cache configuration mismatch: {cache_path}. "
                "Use --overwrite-cache."
            )
        print(f"Loaded cache: {cache_path}")
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cache

    print(
        f"Extracting {model_name}: decoder={decoder_path}, "
        f"layers={selected_layers}, object_state={args.object_state}",
        flush=True,
    )

    sids: List[int] = []
    image_ids: List[str] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    original_vectors: List[np.ndarray] = []
    swap_aligned_vectors: List[np.ndarray] = []
    role_even_vectors: List[np.ndarray] = []

    device = torch.device(args.device)
    started = time.time()

    for record_index, record in enumerate(
        tqdm(records, desc=f"extract:{model_name}", dynamic_ncols=True)
    ):
        sid = int(record.sid)
        image = None
        original_batch = None
        swapped_batch = None
        try:
            prompt = prompt_rows[sid]
            subject = str(prompt["subject"])
            reference = str(prompt["reference"])
            question_text = str(prompt["question_text"])
            gt = step1.normalize_relation(prompt["answer_raw"])
            if gt not in REL_TO_ID:
                raise ValueError(f"Unsupported GT {gt!r}")

            image = step1.record_image(record)
            original_batch = step1.make_question_batch(
                processor=processor,
                image=image,
                question_text=question_text,
                device=device,
            )
            swapped_question = step1.build_swapped_question(subject, reference)
            swapped_batch = step1.make_question_batch(
                processor=processor,
                image=image,
                question_text=swapped_question,
                device=device,
            )

            # Q_AB: subject=A, reference=B.
            orig_a_subject, orig_b_reference = extract_prompt_object_states(
                step1=step1,
                model=model,
                processor=processor,
                batch=original_batch,
                subject=subject,
                reference=reference,
                selected_layers=selected_layers,
                object_state=args.object_state,
            )

            # Q_BA: subject=B, reference=A.
            swap_b_subject, swap_a_reference = extract_prompt_object_states(
                step1=step1,
                model=model,
                processor=processor,
                batch=swapped_batch,
                subject=reference,
                reference=subject,
                selected_layers=selected_layers,
                object_state=args.object_state,
            )

            original = orig_a_subject - orig_b_reference
            swap_aligned = swap_a_reference - swap_b_subject
            role_even = 0.5 * (original + swap_aligned)

            sids.append(sid)
            image_ids.append(str(getattr(record, "image_id", sid)))
            subjects.append(subject)
            references.append(reference)
            relations.append(gt)
            original_vectors.append(original)
            swap_aligned_vectors.append(swap_aligned)
            role_even_vectors.append(role_even)

            if args.print_every > 0 and len(sids) % args.print_every == 0:
                elapsed = time.time() - started
                print(
                    f"[{model_name}] completed={len(sids)} "
                    f"last_sid={sid} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        except Exception as exc:
            append_jsonl(
                errors_path,
                {
                    "model": model_name,
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"ERROR [{model_name}] sid={sid}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            del image, original_batch, swapped_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not sids:
        raise RuntimeError(f"No usable states extracted for {model_name}")

    array_dtype = np.float16 if args.cache_dtype == "float16" else np.float32
    metadata = {
        "script_version": SCRIPT_VERSION,
        "model": model_name,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "selected_layers": selected_layers,
        "object_state": args.object_state,
        "n_samples": len(sids),
        "definition": {
            "original": "h_A_subject_original - h_B_reference_original",
            "swap_aligned": "h_A_reference_swap - h_B_subject_swap",
            "role_even": "0.5 * (original + swap_aligned)",
        },
    }
    arrays: Dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata), dtype=object),
        "sid": np.asarray(sids, dtype=np.int64),
        "image_id": np.asarray(image_ids, dtype=object),
        "subject": np.asarray(subjects, dtype=object),
        "reference": np.asarray(references, dtype=object),
        "relation": np.asarray(relations, dtype=object),
        "decoder_block_index": np.asarray(selected_layers, dtype=np.int32),
        "original_vectors": np.stack(original_vectors).astype(array_dtype),
        "swap_aligned_vectors": np.stack(swap_aligned_vectors).astype(array_dtype),
        "role_even_vectors": np.stack(role_even_vectors).astype(array_dtype),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(cache_path)
    print(f"Saved cache: {cache_path}", flush=True)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    arrays["metadata"] = metadata
    return arrays


def group_key(subject: str, reference: str, sid: int, split_unit: str) -> str:
    if split_unit == "sample":
        return f"sid:{sid}"
    return " || ".join(sorted((str(subject), str(reference))))


def group_signature(indices: Sequence[int], labels: np.ndarray) -> Tuple[Tuple[str, int], ...]:
    counts = Counter(str(labels[index]) for index in indices)
    return tuple(sorted(counts.items()))


def make_split(
    *,
    sids: np.ndarray,
    subjects: np.ndarray,
    references: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    split_unit: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index in range(len(sids)):
        key = group_key(
            str(subjects[index]),
            str(references[index]),
            int(sids[index]),
            split_unit,
        )
        group_to_indices[key].append(index)

    strata: Dict[Tuple[Tuple[str, int], ...], List[str]] = defaultdict(list)
    for group, indices in group_to_indices.items():
        strata[group_signature(indices, labels)].append(group)

    rng = np.random.default_rng(seed)
    train_groups: set[str] = set()
    test_groups: set[str] = set()
    for signature, groups in sorted(strata.items(), key=lambda item: str(item[0])):
        groups = list(groups)
        rng.shuffle(groups)
        if len(groups) == 1:
            # A singleton stratum cannot be split. Put it in the larger test set.
            test_groups.add(groups[0])
            continue
        n_train = int(round(len(groups) * train_ratio))
        n_train = max(1, min(n_train, len(groups) - 1))
        train_groups.update(groups[:n_train])
        test_groups.update(groups[n_train:])

    train_idx: List[int] = []
    test_idx: List[int] = []
    split_by_sid: Dict[int, str] = {}
    for group, indices in group_to_indices.items():
        if group in train_groups:
            target = train_idx
            split_name = "train"
        elif group in test_groups:
            target = test_idx
            split_name = "test"
        else:
            raise RuntimeError(f"Group assigned to neither split: {group}")
        target.extend(indices)
        for index in indices:
            split_by_sid[int(sids[index])] = split_name

    train = np.asarray(sorted(train_idx), dtype=np.int64)
    test = np.asarray(sorted(test_idx), dtype=np.int64)
    if len(np.intersect1d(train, test)):
        raise RuntimeError("Train/test overlap")
    if len(train) + len(test) != len(sids):
        raise RuntimeError("Train/test split lost samples")

    missing_train = [relation for relation in RELATIONS if not np.any(labels[train] == relation)]
    missing_test = [relation for relation in RELATIONS if not np.any(labels[test] == relation)]
    if missing_train or missing_test:
        raise RuntimeError(
            f"Split missing relations: train={missing_train}, test={missing_test}. "
            "Try --split-unit sample or another seed."
        )
    return train, test, split_by_sid


def fit_relation_directions(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(x_train, dtype=np.float64).mean(axis=0)
    centered = np.asarray(x_train, dtype=np.float64) - center
    directions: List[np.ndarray] = []
    for relation in RELATIONS:
        mask = y_train == relation
        if not np.any(mask):
            raise RuntimeError(f"Training split has no {relation} samples")
        direction = centered[mask].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= EPS:
            raise RuntimeError(f"Near-zero learned direction for {relation}")
        directions.append(direction / norm)
    return center.astype(np.float32), np.stack(directions).astype(np.float32)


def score_relation_directions(
    x_test: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(x_test, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    normalized = normalize_rows(centered)
    return normalized @ np.asarray(directions, dtype=np.float64).T


def safe_correlation(x: Iterable[Any], y: Iterable[Any]) -> Dict[str, Any]:
    frame = pd.DataFrame({"x": list(x), "y": list(y)})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(frame))
    if n < 3:
        return {
            "N": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    if frame["x"].std(ddof=0) <= EPS or frame["y"].std(ddof=0) <= EPS:
        return {
            "N": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    pr = pearsonr(frame["x"], frame["y"])
    sr = spearmanr(frame["x"], frame["y"])
    return {
        "N": n,
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def macro_accuracy(frame: pd.DataFrame) -> float:
    values = []
    for relation in RELATIONS:
        part = frame[frame["gt_internal"] == relation]
        if len(part):
            values.append(float(part["direction_correct"].mean()))
    return float(np.mean(values)) if values else float("nan")


def probability_lookup(probability_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Any]]:
    required = {"model", "sid", "gt_internal"}
    missing = sorted(required.difference(probability_df.columns))
    if missing:
        raise KeyError(f"Probability CSV missing columns: {missing}")

    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in probability_df.to_dict(orient="records"):
        key = (str(row["model"]), int(row["sid"]))
        if key in lookup:
            raise RuntimeError(f"Duplicate probability row: {key}")
        lookup[key] = row
    return lookup


def test_model_representation(
    *,
    model_name: str,
    representation: str,
    cache: Mapping[str, Any],
    probability_rows: Mapping[Tuple[str, int], Mapping[str, Any]],
    probability_column: str,
    repeat: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_directions_dir: Path,
) -> List[Dict[str, Any]]:
    sids = np.asarray(cache["sid"], dtype=np.int64)
    image_ids = np.asarray(cache["image_id"], dtype=object)
    subjects = np.asarray(cache["subject"], dtype=object)
    references = np.asarray(cache["reference"], dtype=object)
    labels = np.asarray([str(x) for x in cache["relation"].tolist()], dtype=object)
    layers = np.asarray(cache["decoder_block_index"], dtype=np.int64)
    vectors = np.asarray(cache[f"{representation}_vectors"], dtype=np.float32)

    if vectors.ndim != 3 or vectors.shape[:2] != (len(sids), len(layers)):
        raise RuntimeError(
            f"Unexpected {representation} vector shape for {model_name}: {vectors.shape}"
        )

    rows: List[Dict[str, Any]] = []
    centers: List[np.ndarray] = []
    directions_by_layer: List[np.ndarray] = []

    for layer_position, layer in enumerate(layers.tolist()):
        x = vectors[:, layer_position, :].astype(np.float64)
        center, directions = fit_relation_directions(x[train_idx], labels[train_idx])
        scores = score_relation_directions(x[test_idx], center, directions)
        prediction_ids = np.argmax(scores, axis=1)

        centers.append(center)
        directions_by_layer.append(directions)

        for test_position, sample_index in enumerate(test_idx.tolist()):
            sid = int(sids[sample_index])
            gt = str(labels[sample_index])
            gt_id = REL_TO_ID[gt]
            sample_scores = scores[test_position]
            other_scores = np.delete(sample_scores, gt_id)
            prediction = RELATIONS[int(prediction_ids[test_position])]
            probability_row = probability_rows.get((model_name, sid), {})
            probability_gt = str(probability_row.get("gt_internal", gt))
            if probability_row and probability_gt != gt:
                raise RuntimeError(
                    f"GT mismatch model={model_name} sid={sid}: "
                    f"cache={gt} probability={probability_gt}"
                )

            row: Dict[str, Any] = {
                "model": model_name,
                "representation": representation,
                "repeat": int(repeat),
                "layer": int(layer),
                "sid": sid,
                "image_id": str(image_ids[sample_index]),
                "subject": str(subjects[sample_index]),
                "reference": str(references[sample_index]),
                "gt_internal": gt,
                "gold_relation": DISPLAY_RELATION[gt],
                "direction_prediction_internal": prediction,
                "direction_prediction": DISPLAY_RELATION[prediction],
                "direction_correct": bool(prediction == gt),
                "gold_cosine": float(sample_scores[gt_id]),
                "strongest_wrong_cosine": float(np.max(other_scores)),
                "gold_cosine_margin": float(
                    sample_scores[gt_id] - np.max(other_scores)
                ),
                "top1_cosine": float(np.max(sample_scores)),
                "top2_cosine": float(np.partition(sample_scores, -2)[-2]),
                "top1_margin": float(
                    np.max(sample_scores) - np.partition(sample_scores, -2)[-2]
                ),
                "cosine_left": float(sample_scores[REL_TO_ID["left"]]),
                "cosine_right": float(sample_scores[REL_TO_ID["right"]]),
                "cosine_on": float(sample_scores[REL_TO_ID["above"]]),
                "cosine_under": float(sample_scores[REL_TO_ID["below"]]),
                "probability_available": bool(probability_row),
                "probability_column": probability_column,
                "gold_relation_probability": safe_float(
                    probability_row.get(probability_column, np.nan)
                ),
                "gold_candidate_word_probability": safe_float(
                    probability_row.get("gold_candidate_word_probability", np.nan)
                ),
                "gold_candidate_word_geometric_mean_probability": safe_float(
                    probability_row.get(
                        "gold_candidate_word_geometric_mean_probability", np.nan
                    )
                ),
                "gold_candidate_first_token_probability": safe_float(
                    probability_row.get("gold_candidate_first_token_probability", np.nan)
                ),
                "gold_candidate_word_logprob": safe_float(
                    probability_row.get("gold_candidate_word_logprob", np.nan)
                ),
                "generation_correct": probability_row.get("generation_correct", np.nan),
                "generation_prediction": probability_row.get(
                    "generation_prediction", ""
                ),
            }
            rows.append(row)

    output_directions_dir.mkdir(parents=True, exist_ok=True)
    direction_path = output_directions_dir / (
        f"{model_name}_{representation}_repeat{repeat}.npz"
    )
    np.savez_compressed(
        direction_path,
        relation_order=np.asarray(RELATIONS, dtype=object),
        decoder_block_index=layers.astype(np.int32),
        train_sid=sids[train_idx],
        test_sid=sids[test_idx],
        centers=np.stack(centers).astype(np.float32),
        directions=np.stack(directions_by_layer).astype(np.float32),
    )
    return rows


def build_correlation_table(
    samples: pd.DataFrame,
    probability_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_names = ("gold_cosine", "gold_cosine_margin")

    group_columns = ["model", "representation", "repeat", "layer"]
    for keys, layer_frame in samples.groupby(group_columns, sort=False):
        model_name, representation, repeat, layer = keys
        subsets = {
            "all": layer_frame,
            "generation_correct": layer_frame[
                layer_frame["generation_correct"].astype(str).str.lower() == "true"
            ],
            "generation_wrong": layer_frame[
                layer_frame["generation_correct"].astype(str).str.lower() == "false"
            ],
        }

        for subset_name, subset in subsets.items():
            for gold in RELATIONS:
                part = subset[subset["gt_internal"] == gold]
                for metric in metric_names:
                    stats = safe_correlation(
                        part[metric],
                        part["gold_relation_probability"],
                    )
                    rows.append({
                        "model": model_name,
                        "representation": representation,
                        "repeat": int(repeat),
                        "layer": int(layer),
                        "subset": subset_name,
                        "gt_internal": gold,
                        "gold_relation": DISPLAY_RELATION[gold],
                        "similarity_metric": metric,
                        "probability_column": probability_column,
                        "mean_similarity": safe_float(part[metric].mean()),
                        "mean_probability": safe_float(
                            part["gold_relation_probability"].mean()
                        ),
                        **stats,
                    })
    return pd.DataFrame(rows)


def build_layer_summary(
    samples: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_columns = ["model", "representation", "repeat", "layer"]
    for keys, frame in samples.groupby(group_columns, sort=False):
        model_name, representation, repeat, layer = keys
        corr_part = correlations[
            (correlations["model"] == model_name)
            & (correlations["representation"] == representation)
            & (correlations["repeat"] == repeat)
            & (correlations["layer"] == layer)
            & (correlations["subset"] == "all")
        ]

        def macro_corr(metric: str, column: str) -> float:
            values = pd.to_numeric(
                corr_part.loc[
                    corr_part["similarity_metric"] == metric,
                    column,
                ],
                errors="coerce",
            ).dropna()
            return float(values.mean()) if len(values) else float("nan")

        cosine_corr = corr_part[corr_part["similarity_metric"] == "gold_cosine"]
        margin_corr = corr_part[
            corr_part["similarity_metric"] == "gold_cosine_margin"
        ]

        rows.append({
            "model": model_name,
            "representation": representation,
            "repeat": int(repeat),
            "layer": int(layer),
            "N_test": int(len(frame)),
            "N_probability": int(frame["gold_relation_probability"].notna().sum()),
            "accuracy": float(frame["direction_correct"].mean()),
            "macro_accuracy": macro_accuracy(frame),
            "mean_gold_cosine": safe_float(frame["gold_cosine"].mean()),
            "mean_gold_cosine_margin": safe_float(
                frame["gold_cosine_margin"].mean()
            ),
            "macro_pearson_gold_cosine": macro_corr(
                "gold_cosine", "pearson_r"
            ),
            "macro_spearman_gold_cosine": macro_corr(
                "gold_cosine", "spearman_r"
            ),
            "macro_pearson_gold_margin": macro_corr(
                "gold_cosine_margin", "pearson_r"
            ),
            "macro_spearman_gold_margin": macro_corr(
                "gold_cosine_margin", "spearman_r"
            ),
            "positive_gold_cosine_pearson_relations": int(
                (pd.to_numeric(cosine_corr["pearson_r"], errors="coerce") > 0).sum()
            ),
            "significant_positive_gold_cosine_relations": int(
                (
                    (pd.to_numeric(cosine_corr["pearson_r"], errors="coerce") > 0)
                    & (pd.to_numeric(cosine_corr["pearson_p"], errors="coerce") < 0.05)
                ).sum()
            ),
            "positive_gold_margin_pearson_relations": int(
                (pd.to_numeric(margin_corr["pearson_r"], errors="coerce") > 0).sum()
            ),
            "significant_positive_gold_margin_relations": int(
                (
                    (pd.to_numeric(margin_corr["pearson_r"], errors="coerce") > 0)
                    & (pd.to_numeric(margin_corr["pearson_p"], errors="coerce") < 0.05)
                ).sum()
            ),
        })
    return pd.DataFrame(rows)


def build_best_layers(layer_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_columns = ["model", "representation", "repeat"]
    for keys, frame in layer_summary.groupby(group_columns, sort=False):
        model_name, representation, repeat = keys
        frame = frame.copy()

        def best_row(metric: str) -> Optional[pd.Series]:
            valid = frame[pd.to_numeric(frame[metric], errors="coerce").notna()]
            if valid.empty:
                return None
            return valid.loc[pd.to_numeric(valid[metric], errors="coerce").idxmax()]

        accuracy_best = best_row("accuracy")
        cosine_best = best_row("macro_pearson_gold_cosine")
        margin_best = best_row("macro_pearson_gold_margin")
        rows.append({
            "model": model_name,
            "representation": representation,
            "repeat": int(repeat),
            "best_accuracy_layer": (
                int(accuracy_best["layer"]) if accuracy_best is not None else np.nan
            ),
            "best_accuracy": (
                float(accuracy_best["accuracy"]) if accuracy_best is not None else np.nan
            ),
            "best_accuracy_macro_pearson_gold_cosine": (
                float(accuracy_best["macro_pearson_gold_cosine"])
                if accuracy_best is not None
                else np.nan
            ),
            "best_accuracy_macro_pearson_gold_margin": (
                float(accuracy_best["macro_pearson_gold_margin"])
                if accuracy_best is not None
                else np.nan
            ),
            "best_cosine_correlation_layer": (
                int(cosine_best["layer"]) if cosine_best is not None else np.nan
            ),
            "best_macro_pearson_gold_cosine": (
                float(cosine_best["macro_pearson_gold_cosine"])
                if cosine_best is not None
                else np.nan
            ),
            "best_margin_correlation_layer": (
                int(margin_best["layer"]) if margin_best is not None else np.nan
            ),
            "best_macro_pearson_gold_margin": (
                float(margin_best["macro_pearson_gold_margin"])
                if margin_best is not None
                else np.nan
            ),
        })
    return pd.DataFrame(rows)



def make_plots(
    *,
    samples: pd.DataFrame,
    layer_summary: pd.DataFrame,
    best_layers: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: {exc}", flush=True)
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # One plot per metric; no multi-panel figures.
    for (model_name, representation, repeat), frame in layer_summary.groupby(
        ["model", "representation", "repeat"], sort=False
    ):
        frame = frame.sort_values("layer")
        stem = f"{model_name}_{representation}_repeat{int(repeat)}"

        for metric, ylabel in (
            ("accuracy", "Held-out accuracy"),
            ("macro_pearson_gold_cosine", "Macro Pearson r"),
            ("macro_spearman_gold_cosine", "Macro Spearman rho"),
            ("macro_pearson_gold_margin", "Macro Pearson r"),
        ):
            plt.figure(figsize=(8, 5))
            plt.plot(frame["layer"], frame[metric], marker="o")
            plt.xlabel("Decoder layer")
            plt.ylabel(ylabel)
            plt.title(f"{model_name} | {representation} | {metric}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{stem}_{metric}.png", dpi=180)
            plt.close()

    for best in best_layers.itertuples(index=False):
        if not math.isfinite(float(best.best_accuracy_layer)):
            continue
        layer = int(best.best_accuracy_layer)
        frame = samples[
            (samples["model"] == best.model)
            & (samples["representation"] == best.representation)
            & (samples["repeat"] == best.repeat)
            & (samples["layer"] == layer)
        ]
        for gold in RELATIONS:
            part = frame[frame["gt_internal"] == gold].copy()
            part = part.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["gold_relation_probability"]
            )
            if part.empty:
                continue
            for metric in ("gold_cosine", "gold_cosine_margin"):
                stats = safe_correlation(
                    part[metric], part["gold_relation_probability"]
                )
                plt.figure(figsize=(7, 5))
                plt.scatter(
                    part[metric],
                    part["gold_relation_probability"],
                    alpha=0.7,
                )
                if len(part) >= 2 and part[metric].std(ddof=0) > EPS:
                    slope, intercept = np.polyfit(
                        part[metric].to_numpy(dtype=float),
                        part["gold_relation_probability"].to_numpy(dtype=float),
                        deg=1,
                    )
                    xs = np.linspace(part[metric].min(), part[metric].max(), 100)
                    plt.plot(xs, slope * xs + intercept)
                plt.xlabel(metric)
                plt.ylabel("Teacher-forced gold relation-word probability")
                plt.title(
                    f"{best.model} | {best.representation} | L{layer} | "
                    f"gold={DISPLAY_RELATION[gold]} | N={stats['N']} | "
                    f"r={stats['pearson_r']:.3f} | rho={stats['spearman_r']:.3f}"
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                filename = (
                    f"{best.model}_{best.representation}_repeat{int(best.repeat)}_"
                    f"L{layer}_{DISPLAY_RELATION[gold]}_{metric}.png"
                )
                plt.savefig(plots_dir / filename, dpi=180)
                plt.close()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0,1)")

    models = parse_csv_list(args.models)
    representations = parse_csv_list(args.representations)
    invalid_representations = sorted(set(representations).difference(VALID_REPRESENTATIONS))
    if invalid_representations:
        raise ValueError(
            f"Invalid representations {invalid_representations}; "
            f"available={VALID_REPRESENTATIONS}"
        )

    output_dir = Path(args.output_dir)
    if args.overwrite_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "state_cache"
    direction_dir = output_dir / "learned_directions"
    errors_path = output_dir / "extraction_errors.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_path = Path(args.base_script)
    step1 = import_module(base_path, "coco_centroid_step1_v4_role_even_helpers")
    two_object = step1.import_two_object_module()
    records, audit = two_object.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    if not records:
        raise RuntimeError("No usable records")
    prompt_rows = step1.load_standard_prompts(Path(args.prompt_jsonl))
    records = [record for record in records if int(record.sid) in prompt_rows]
    if not records:
        raise RuntimeError("No records have matching prompt rows")

    specs = step1.merged_model_specs(two_object)
    missing_models = [model for model in models if model not in specs]
    if missing_models:
        raise ValueError(
            f"Unknown models {missing_models}; available={sorted(specs)}"
        )

    probability_path = Path(args.probability_csv)
    if not probability_path.exists():
        raise FileNotFoundError(probability_path)
    probability_df = pd.read_csv(probability_path)
    if args.probability_column not in probability_df.columns:
        raise KeyError(
            f"Probability column {args.probability_column!r} missing. "
            f"Available={list(probability_df.columns)}"
        )
    probability_rows = probability_lookup(probability_df)

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "prompt_jsonl": args.prompt_jsonl,
        "base_script": str(base_path),
        "models": models,
        "representations": representations,
        "object_state": args.object_state,
        "layers": args.layers,
        "train_ratio": args.train_ratio,
        "test_ratio": 1.0 - args.train_ratio,
        "split_unit": args.split_unit,
        "repeats": args.repeats,
        "seed": args.seed,
        "probability_csv": str(probability_path),
        "probability_column": args.probability_column,
        "attn_impl": args.attn_impl,
        "make_plots": args.make_plots,
        "audit": audit,
        "relation_order": list(RELATIONS),
        "relation_vectors_are_independent": True,
        "opposite_or_orthogonal_constraints": False,
    }
    write_json(output_dir / "config.json", config)

    all_sample_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for model_name in models:
        cache_path = cache_dir / f"{model_name}.npz"
        cache = extract_model_cache(
            args=args,
            step1=step1,
            model_name=model_name,
            spec=specs[model_name],
            records=records,
            prompt_rows=prompt_rows,
            cache_path=cache_path,
            errors_path=errors_path,
        )

        sids = np.asarray(cache["sid"], dtype=np.int64)
        subjects = np.asarray(cache["subject"], dtype=object)
        references = np.asarray(cache["reference"], dtype=object)
        labels = np.asarray([str(x) for x in cache["relation"].tolist()], dtype=object)

        print("\n" + "=" * 150)
        print(
            f"MODEL={model_name} N={len(sids)} "
            f"relations={dict(Counter(labels.tolist()))}"
        )
        print("=" * 150)

        for repeat in range(args.repeats):
            split_seed = int(args.seed + repeat)
            train_idx, test_idx, split_by_sid = make_split(
                sids=sids,
                subjects=subjects,
                references=references,
                labels=labels,
                train_ratio=args.train_ratio,
                split_unit=args.split_unit,
                seed=split_seed,
            )

            for index, sid in enumerate(sids.tolist()):
                split_rows.append({
                    "model": model_name,
                    "repeat": repeat,
                    "seed": split_seed,
                    "sid": int(sid),
                    "subject": str(subjects[index]),
                    "reference": str(references[index]),
                    "gt_internal": str(labels[index]),
                    "gold_relation": DISPLAY_RELATION[str(labels[index])],
                    "split": split_by_sid[int(sid)],
                })

            print(
                f"repeat={repeat} seed={split_seed} "
                f"train={len(train_idx)} ({len(train_idx)/len(sids):.3f}) "
                f"test={len(test_idx)} ({len(test_idx)/len(sids):.3f}) "
                f"train_counts={dict(Counter(labels[train_idx].tolist()))}",
                flush=True,
            )

            for representation in representations:
                rows = test_model_representation(
                    model_name=model_name,
                    representation=representation,
                    cache=cache,
                    probability_rows=probability_rows,
                    probability_column=args.probability_column,
                    repeat=repeat,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    output_directions_dir=direction_dir,
                )
                all_sample_rows.extend(rows)

        # Drop one model cache from RAM before loading the next checkpoint.
        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    samples = pd.DataFrame(all_sample_rows)
    if samples.empty:
        raise RuntimeError("No test rows produced")
    samples_path = output_dir / "sample_test_metrics.csv"
    samples.to_csv(samples_path, index=False)

    split_df = pd.DataFrame(split_rows)
    split_path = output_dir / "split_assignments.csv"
    split_df.to_csv(split_path, index=False)

    correlations = build_correlation_table(samples, args.probability_column)
    correlation_path = output_dir / "correlation_by_same_gold.csv"
    correlations.to_csv(correlation_path, index=False)

    layer_summary = build_layer_summary(samples, correlations)
    layer_summary_path = output_dir / "layer_summary.csv"
    layer_summary.to_csv(layer_summary_path, index=False)

    best_layers = build_best_layers(layer_summary)
    best_layers_path = output_dir / "best_layers.csv"
    best_layers.to_csv(best_layers_path, index=False)

    if args.make_plots:
        make_plots(
            samples=samples,
            layer_summary=layer_summary,
            best_layers=best_layers,
            output_dir=output_dir,
        )

    print("\n" + "=" * 150)
    print("BEST LAYERS")
    print("=" * 150)
    print(
        best_layers.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    role_even_best = best_layers[best_layers["representation"] == "role_even"]
    if not role_even_best.empty:
        print("\nROLE-EVEN BEST-ACCURACY LAYER: SAME-GOLD CORRELATIONS")
        selected_parts: List[pd.DataFrame] = []
        for row in role_even_best.itertuples(index=False):
            selected_parts.append(
                correlations[
                    (correlations["model"] == row.model)
                    & (correlations["representation"] == "role_even")
                    & (correlations["repeat"] == row.repeat)
                    & (correlations["layer"] == row.best_accuracy_layer)
                    & (correlations["subset"] == "all")
                ]
            )
        if selected_parts:
            selected = pd.concat(selected_parts, ignore_index=True)
            print(
                selected[
                    [
                        "model",
                        "repeat",
                        "layer",
                        "gold_relation",
                        "similarity_metric",
                        "N",
                        "pearson_r",
                        "pearson_p",
                        "spearman_r",
                        "spearman_p",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda value: f"{value:.6f}",
                )
            )

    print("\nSaved:")
    for path in (
        samples_path,
        correlation_path,
        layer_summary_path,
        best_layers_path,
        split_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
