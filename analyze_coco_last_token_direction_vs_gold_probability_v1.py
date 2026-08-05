#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two: prompt-last-token relation directions vs. generation relation-word probability.

This script uses the hidden state of the last non-padding prompt token, i.e. the
state that predicts the first generated answer token.

For every decoder layer l and sample i:

    q_i^(l) = hidden state at the final prompt token

It supports two relation-direction methods.

1. last_own
   Learn one independent relation direction from prompt-last states using only
   the 20% training split:

       center_last = mean(q_train)
       d_last,k = normalize(mean(q_train[y=k] - center_last))

2. object_to_last
   Reuse the independent object-pair relation directions learned by
   analyze_coco_role_even_direction_vs_gold_probability_v1.py, then directly
   score centered prompt-last states against those object-derived directions:

       score_k = cos(q_test - center_last, d_object,k)

   This tests whether the object-pair relation direction is preserved in the
   same residual-stream basis at the prompt-last token. Low object_to_last
   accuracy does not imply that relation information is absent; it may have
   been rotated or re-encoded.

For each held-out sample, the script computes:

    gold_cosine = score_gold
    gold_cosine_margin = score_gold - max(score_non_gold)

and correlates them, within each fixed gold relation, with the previously saved
teacher-forced probability of that same gold relation word.

Default split: 20% train / 80% test, grouped by unordered object-name pair.
No opposite-axis or orthogonality constraint is imposed: left, right, on and
under are learned/scored as four independent directions.

Expected object-direction files (optional, for object_to_last):

    <object-direction-dir>/<model>_<object-representation>_repeat<R>.npz

Typically:

    output/coco_role_even_direction_vs_gold_probability_v1/learned_directions/

Main outputs
------------
<output-dir>/
    config.json
    extraction_errors.jsonl
    split_assignments.csv
    state_cache/<model>.npz
    learned_directions/<model>_last_own_repeat<R>.npz
    sample_test_metrics.csv
    correlation_by_same_gold.csv
    layer_summary.csv
    best_layers.csv
    direction_alignment.csv
    plots/*.png

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_coco_last_token_direction_vs_gold_probability_v1.py \
  --models qwen2-2b,qwen-3b,qwen-7b,llava-7b,llava-13b \
  --probability-csv \
    output/coco_gold_direction_cosine_vs_gold_word_probability_v4/sample_gold_probability.csv \
  --train-ratio 0.2 \
  --split-unit pair \
  --methods last_own,object_to_last \
  --object-direction-dir \
    output/coco_role_even_direction_vs_gold_probability_v1/learned_directions \
  --object-representation role_even \
  --layers all \
  --device cuda:0 \
  --output-dir output/coco_last_token_direction_vs_gold_probability_v1
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
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


SCRIPT_VERSION = "coco-last-token-direction-vs-gold-probability-v1"
EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
DISPLAY_RELATION = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
VALID_METHODS = ("last_own", "object_to_last")
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
        "--methods",
        default="last_own,object_to_last",
        help="Comma-separated subset of last_own,object_to_last.",
    )
    p.add_argument(
        "--object-direction-dir",
        default=(
            "output/coco_role_even_direction_vs_gold_probability_v1/"
            "learned_directions"
        ),
        help="Directory containing object-pair learned-direction NPZ files.",
    )
    p.add_argument(
        "--object-representation",
        default="role_even",
        choices=("original", "swap_aligned", "role_even"),
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
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_layers(value: str, n_layers: int) -> List[int]:
    if str(value).strip().lower() == "all":
        return list(range(n_layers))
    layers = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    invalid = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; model has {n_layers} decoder blocks")
    if not layers:
        raise ValueError("No layers selected")
    return layers


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def normalize_direction_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, EPS)


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


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


def final_prompt_position(batch: Mapping[str, Any]) -> int:
    input_ids = batch["input_ids"]
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError(f"Expected input_ids shape [1,T], got {tuple(input_ids.shape)}")

    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return int(input_ids.shape[1]) - 1

    mask = attention_mask[0].detach().to("cpu")
    active = torch.nonzero(mask > 0, as_tuple=False).flatten()
    if active.numel() == 0:
        raise RuntimeError("attention_mask contains no active prompt token")
    return int(active[-1].item())


def extract_prompt_last_states(
    *,
    step1: Any,
    model: Any,
    batch: Mapping[str, Any],
    selected_layers: Sequence[int],
) -> Tuple[np.ndarray, int, int]:
    """Return [selected_layers, hidden], last position, prompt token count."""
    last_position = final_prompt_position(batch)
    prompt_token_count = int(batch["attention_mask"][0].sum().item()) \
        if "attention_mask" in batch else int(batch["input_ids"].shape[1])

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

    rows: List[np.ndarray] = []
    input_length = int(batch["input_ids"].shape[1])
    for layer in selected_layers:
        # hidden_states[0] is embedding output; decoder block L is hidden_states[L+1].
        hidden = hidden_states[int(layer) + 1]
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
            raise RuntimeError(f"Unexpected hidden shape at L{layer}: {tuple(hidden.shape)}")
        if int(hidden.shape[1]) != input_length:
            raise RuntimeError(
                f"Token/hidden length mismatch at L{layer}: "
                f"input={input_length} hidden={hidden.shape[1]}"
            )
        rows.append(hidden[0, last_position].float().detach().cpu().numpy())

    del outputs, hidden_states
    return np.stack(rows, axis=0).astype(np.float32), last_position, prompt_token_count


def cache_metadata_matches(
    metadata: Mapping[str, Any],
    *,
    model_name: str,
    selected_layers: Sequence[int],
) -> bool:
    return (
        str(metadata.get("model")) == str(model_name)
        and [int(x) for x in metadata.get("selected_layers", [])]
        == [int(x) for x in selected_layers]
        and str(metadata.get("state_definition")) == "last_non_padding_prompt_token"
    )


def load_cache(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        result = {key: z[key] for key in z.files}
    result["metadata"] = json.loads(str(result["metadata_json"].item()))
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
        ):
            raise RuntimeError(
                f"Cache configuration mismatch: {cache_path}. Use --overwrite-cache."
            )
        print(f"Loaded cache: {cache_path}")
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cache

    print(
        f"Extracting {model_name}: decoder={decoder_path}, layers={selected_layers}, "
        "state=last_non_padding_prompt_token",
        flush=True,
    )

    sids: List[int] = []
    image_ids: List[str] = []
    subjects: List[str] = []
    references: List[str] = []
    relations: List[str] = []
    states: List[np.ndarray] = []
    last_positions: List[int] = []
    prompt_lengths: List[int] = []

    device = torch.device(args.device)
    started = time.time()

    for record in tqdm(records, desc=f"extract-last:{model_name}", dynamic_ncols=True):
        sid = int(record.sid)
        image = None
        batch = None
        try:
            prompt = prompt_rows[sid]
            subject = str(prompt["subject"])
            reference = str(prompt["reference"])
            question_text = str(prompt["question_text"])
            gt = step1.normalize_relation(prompt["answer_raw"])
            if gt not in REL_TO_ID:
                raise ValueError(f"Unsupported GT {gt!r}")

            image = step1.record_image(record)
            batch = step1.make_question_batch(
                processor=processor,
                image=image,
                question_text=question_text,
                device=device,
            )
            sample_states, last_position, prompt_length = extract_prompt_last_states(
                step1=step1,
                model=model,
                batch=batch,
                selected_layers=selected_layers,
            )

            sids.append(sid)
            image_ids.append(str(getattr(record, "image_id", sid)))
            subjects.append(subject)
            references.append(reference)
            relations.append(gt)
            states.append(sample_states)
            last_positions.append(last_position)
            prompt_lengths.append(prompt_length)

            if args.print_every > 0 and len(sids) % args.print_every == 0:
                elapsed = time.time() - started
                print(
                    f"[{model_name}] completed={len(sids)} last_sid={sid} "
                    f"elapsed={elapsed:.1f}s",
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
            del image, batch
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
        "state_definition": "last_non_padding_prompt_token",
        "n_samples": len(sids),
    }
    arrays: Dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object),
        "sid": np.asarray(sids, dtype=np.int64),
        "image_id": np.asarray(image_ids, dtype=object),
        "subject": np.asarray(subjects, dtype=object),
        "reference": np.asarray(references, dtype=object),
        "relation": np.asarray(relations, dtype=object),
        "decoder_block_index": np.asarray(selected_layers, dtype=np.int32),
        "last_prompt_position": np.asarray(last_positions, dtype=np.int32),
        "prompt_token_count": np.asarray(prompt_lengths, dtype=np.int32),
        "last_states": np.stack(states).astype(array_dtype),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
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
    normalized_directions = normalize_direction_rows(directions)
    return normalized @ normalized_directions.T


def load_object_direction_file(
    *,
    directory: Path,
    model_name: str,
    representation: str,
    repeat: int,
) -> Dict[str, Any]:
    path = directory / f"{model_name}_{representation}_repeat{repeat}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing object-direction file: {path}\n"
            "Run analyze_coco_role_even_direction_vs_gold_probability_v1.py first, "
            "or omit object_to_last from --methods."
        )
    with np.load(path, allow_pickle=True) as z:
        result = {key: z[key] for key in z.files}
    result["path"] = str(path)
    relation_order = tuple(str(x) for x in result["relation_order"].tolist())
    if relation_order != RELATIONS:
        raise RuntimeError(
            f"Object direction relation order mismatch: {relation_order} != {RELATIONS}"
        )
    return result


def validate_object_direction_split(
    object_file: Mapping[str, Any],
    train_sid: np.ndarray,
    test_sid: np.ndarray,
) -> None:
    object_train = set(int(x) for x in np.asarray(object_file["train_sid"]).tolist())
    object_test = set(int(x) for x in np.asarray(object_file["test_sid"]).tolist())
    current_train = set(int(x) for x in np.asarray(train_sid).tolist())
    current_test = set(int(x) for x in np.asarray(test_sid).tolist())
    if object_train != current_train or object_test != current_test:
        raise RuntimeError(
            "Object-direction split does not match the current last-token split. "
            "Use the same --train-ratio, --split-unit, --seed, --repeats and sample set.\n"
            f"train only in object={len(object_train-current_train)}, "
            f"train only current={len(current_train-object_train)}, "
            f"test only in object={len(object_test-current_test)}, "
            f"test only current={len(current_test-object_test)}"
        )


def safe_correlation(x: Iterable[Any], y: Iterable[Any]) -> Dict[str, Any]:
    frame = pd.DataFrame({"x": list(x), "y": list(y)})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(frame))
    if n < 3 or frame["x"].std(ddof=0) <= EPS or frame["y"].std(ddof=0) <= EPS:
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


def macro_accuracy(frame: pd.DataFrame) -> float:
    values: List[float] = []
    for relation in RELATIONS:
        part = frame[frame["gt_internal"] == relation]
        if len(part):
            values.append(float(part["direction_correct"].mean()))
    return float(np.mean(values)) if values else float("nan")


def score_method(
    *,
    model_name: str,
    method: str,
    repeat: int,
    cache: Mapping[str, Any],
    probability_rows: Mapping[Tuple[str, int], Mapping[str, Any]],
    probability_column: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    last_own_centers: np.ndarray,
    last_own_directions: np.ndarray,
    object_file: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sids = np.asarray(cache["sid"], dtype=np.int64)
    image_ids = np.asarray(cache["image_id"], dtype=object)
    subjects = np.asarray(cache["subject"], dtype=object)
    references = np.asarray(cache["reference"], dtype=object)
    labels = np.asarray([str(x) for x in cache["relation"].tolist()], dtype=object)
    layers = np.asarray(cache["decoder_block_index"], dtype=np.int64)
    states = np.asarray(cache["last_states"], dtype=np.float32)

    object_layer_to_pos: Dict[int, int] = {}
    if object_file is not None:
        object_layers = np.asarray(object_file["decoder_block_index"], dtype=np.int64)
        object_layer_to_pos = {int(layer): pos for pos, layer in enumerate(object_layers.tolist())}

    rows: List[Dict[str, Any]] = []
    for layer_pos, layer in enumerate(layers.tolist()):
        center = last_own_centers[layer_pos]
        if method == "last_own":
            directions = last_own_directions[layer_pos]
        elif method == "object_to_last":
            if object_file is None:
                raise RuntimeError("object_to_last requested without object direction file")
            if int(layer) not in object_layer_to_pos:
                raise RuntimeError(f"Object direction file has no decoder layer {layer}")
            directions = np.asarray(
                object_file["directions"][object_layer_to_pos[int(layer)]],
                dtype=np.float32,
            )
        else:
            raise ValueError(method)

        scores = score_relation_directions(states[test_idx, layer_pos, :], center, directions)
        pred_ids = np.argmax(scores, axis=1)

        for test_pos, sample_index in enumerate(test_idx.tolist()):
            sid = int(sids[sample_index])
            gt = str(labels[sample_index])
            gt_id = REL_TO_ID[gt]
            sample_scores = scores[test_pos]
            other_scores = np.delete(sample_scores, gt_id)
            prediction = RELATIONS[int(pred_ids[test_pos])]
            probability_row = probability_rows.get((model_name, sid), {})
            probability_gt = str(probability_row.get("gt_internal", gt))
            if probability_row and probability_gt != gt:
                raise RuntimeError(
                    f"GT mismatch model={model_name} sid={sid}: "
                    f"cache={gt} probability={probability_gt}"
                )

            rows.append({
                "model": model_name,
                "method": method,
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
                "gold_cosine_margin": float(sample_scores[gt_id] - np.max(other_scores)),
                "top1_cosine": float(np.max(sample_scores)),
                "top2_cosine": float(np.partition(sample_scores, -2)[-2]),
                "top1_margin": float(np.max(sample_scores) - np.partition(sample_scores, -2)[-2]),
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
                "generation_prediction": probability_row.get("generation_prediction", ""),
            })
    return rows


def build_correlation_table(samples: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = ("gold_cosine", "gold_cosine_margin")
    group_columns = ["model", "method", "repeat", "layer"]

    for keys, layer_frame in samples.groupby(group_columns, sort=False):
        model_name, method, repeat, layer = keys
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
            for relation in RELATIONS:
                part = subset[subset["gt_internal"] == relation]
                for metric in metrics:
                    stats = safe_correlation(
                        part[metric],
                        part["gold_relation_probability"],
                    )
                    rows.append({
                        "model": model_name,
                        "method": method,
                        "repeat": int(repeat),
                        "layer": int(layer),
                        "subset": subset_name,
                        "gt_internal": relation,
                        "gold_relation": DISPLAY_RELATION[relation],
                        "similarity_metric": metric,
                        **stats,
                    })
    return pd.DataFrame(rows)


def build_layer_summary(samples: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_columns = ["model", "method", "repeat", "layer"]

    corr_all = correlations[correlations["subset"] == "all"]
    for keys, frame in samples.groupby(group_columns, sort=False):
        model_name, method, repeat, layer = keys
        row: Dict[str, Any] = {
            "model": model_name,
            "method": method,
            "repeat": int(repeat),
            "layer": int(layer),
            "N_test": int(len(frame)),
            "N_probability": int(frame["gold_relation_probability"].notna().sum()),
            "accuracy": float(frame["direction_correct"].mean()),
            "macro_accuracy": macro_accuracy(frame),
            "mean_gold_cosine": float(frame["gold_cosine"].mean()),
            "mean_gold_cosine_margin": float(frame["gold_cosine_margin"].mean()),
        }
        for relation in RELATIONS:
            part = frame[frame["gt_internal"] == relation]
            row[f"accuracy_{DISPLAY_RELATION[relation]}"] = (
                float(part["direction_correct"].mean()) if len(part) else np.nan
            )

        corr_part = corr_all[
            (corr_all["model"] == model_name)
            & (corr_all["method"] == method)
            & (corr_all["repeat"] == repeat)
            & (corr_all["layer"] == layer)
        ]
        for metric in ("gold_cosine", "gold_cosine_margin"):
            metric_part = corr_part[corr_part["similarity_metric"] == metric]
            valid_p = metric_part["pearson_r"].dropna()
            valid_s = metric_part["spearman_r"].dropna()
            row[f"macro_pearson_{metric}"] = (
                float(valid_p.mean()) if len(valid_p) else np.nan
            )
            row[f"macro_spearman_{metric}"] = (
                float(valid_s.mean()) if len(valid_s) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_best_layers(layer_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, frame in layer_summary.groupby(["model", "method", "repeat"], sort=False):
        model_name, method, repeat = keys
        frame = frame.sort_values("layer")

        def best_row(column: str) -> pd.Series:
            valid = frame.dropna(subset=[column])
            if valid.empty:
                return frame.iloc[0]
            max_value = valid[column].max()
            return valid[valid[column] == max_value].sort_values("layer").iloc[0]

        acc = best_row("accuracy")
        cosine = best_row("macro_pearson_gold_cosine")
        margin = best_row("macro_pearson_gold_cosine_margin")
        rows.append({
            "model": model_name,
            "method": method,
            "repeat": int(repeat),
            "best_accuracy_layer": int(acc["layer"]),
            "best_accuracy": float(acc["accuracy"]),
            "best_accuracy_macro_pearson_gold_cosine": float(
                acc["macro_pearson_gold_cosine"]
            ),
            "best_accuracy_macro_pearson_gold_margin": float(
                acc["macro_pearson_gold_cosine_margin"]
            ),
            "best_cosine_correlation_layer": int(cosine["layer"]),
            "best_macro_pearson_gold_cosine": float(
                cosine["macro_pearson_gold_cosine"]
            ),
            "best_margin_correlation_layer": int(margin["layer"]),
            "best_macro_pearson_gold_margin": float(
                margin["macro_pearson_gold_cosine_margin"]
            ),
        })
    return pd.DataFrame(rows)


def build_direction_alignment(
    *,
    model_name: str,
    repeat: int,
    layers: np.ndarray,
    last_directions: np.ndarray,
    object_file: Mapping[str, Any],
) -> pd.DataFrame:
    object_layers = np.asarray(object_file["decoder_block_index"], dtype=np.int64)
    object_map = {int(layer): pos for pos, layer in enumerate(object_layers.tolist())}
    rows: List[Dict[str, Any]] = []

    for last_pos, layer in enumerate(layers.tolist()):
        if int(layer) not in object_map:
            continue
        obj = normalize_direction_rows(
            np.asarray(object_file["directions"][object_map[int(layer)]], dtype=np.float64)
        )
        last = normalize_direction_rows(
            np.asarray(last_directions[last_pos], dtype=np.float64)
        )
        same_relation = np.sum(last * obj, axis=1)
        row: Dict[str, Any] = {
            "model": model_name,
            "repeat": int(repeat),
            "layer": int(layer),
            "macro_same_relation_alignment": float(np.mean(same_relation)),
        }
        for relation, value in zip(RELATIONS, same_relation.tolist()):
            row[f"alignment_{DISPLAY_RELATION[relation]}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def make_plots(
    *,
    samples: pd.DataFrame,
    layer_summary: pd.DataFrame,
    best_layers: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    curve_metrics = (
        ("accuracy", "Held-out accuracy"),
        ("macro_pearson_gold_cosine", "Macro Pearson r: gold cosine vs probability"),
        (
            "macro_pearson_gold_cosine_margin",
            "Macro Pearson r: gold cosine margin vs probability",
        ),
    )
    for keys, frame in layer_summary.groupby(["model", "method", "repeat"], sort=False):
        model_name, method, repeat = keys
        frame = frame.sort_values("layer")
        for metric, ylabel in curve_metrics:
            plt.figure(figsize=(9, 5.5))
            plt.plot(frame["layer"], frame[metric], marker="o")
            plt.xlabel("Decoder layer")
            plt.ylabel(ylabel)
            plt.title(f"{model_name} | {method} | repeat={repeat} | {metric}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                plots_dir / f"{model_name}_{method}_repeat{repeat}_{metric}.png",
                dpi=180,
            )
            plt.close()

    for best in best_layers.itertuples(index=False):
        model_name = str(best.model)
        method = str(best.method)
        repeat = int(best.repeat)
        layer = int(best.best_accuracy_layer)
        frame = samples[
            (samples["model"] == model_name)
            & (samples["method"] == method)
            & (samples["repeat"] == repeat)
            & (samples["layer"] == layer)
        ]
        for relation in RELATIONS:
            relation_frame = frame[frame["gt_internal"] == relation]
            for metric in ("gold_cosine", "gold_cosine_margin"):
                part = relation_frame[[metric, "gold_relation_probability"]].dropna()
                if len(part) < 3:
                    continue
                stats = safe_correlation(part[metric], part["gold_relation_probability"])
                plt.figure(figsize=(7.5, 5.8))
                plt.scatter(
                    part[metric],
                    part["gold_relation_probability"],
                    alpha=0.58,
                )
                if part[metric].std(ddof=0) > EPS:
                    slope, intercept = np.polyfit(
                        part[metric].to_numpy(dtype=float),
                        part["gold_relation_probability"].to_numpy(dtype=float),
                        deg=1,
                    )
                    x_line = np.linspace(part[metric].min(), part[metric].max(), 200)
                    plt.plot(x_line, slope * x_line + intercept)
                plt.xlabel(metric)
                plt.ylabel("Teacher-forced gold relation-word probability")
                plt.title(
                    f"{model_name} | {method} | L{layer} | "
                    f"gold={DISPLAY_RELATION[relation]} | N={stats['N']} | "
                    f"r={stats['pearson_r']:.3f} | rho={stats['spearman_r']:.3f}"
                )
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    plots_dir /
                    f"{model_name}_{method}_repeat{repeat}_L{layer}_"
                    f"{DISPLAY_RELATION[relation]}_{metric}.png",
                    dpi=180,
                )
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
    methods = parse_csv_list(args.methods)
    invalid_methods = sorted(set(methods).difference(VALID_METHODS))
    if invalid_methods:
        raise ValueError(f"Invalid methods {invalid_methods}; available={VALID_METHODS}")

    output_dir = Path(args.output_dir)
    if args.overwrite_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "state_cache"
    learned_dir = output_dir / "learned_directions"
    errors_path = output_dir / "extraction_errors.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_path = Path(args.base_script)
    step1 = import_module(base_path, "coco_centroid_step1_v4_last_token_helpers")
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
        raise ValueError(f"Unknown models {missing_models}; available={sorted(specs)}")

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

    object_direction_dir = Path(args.object_direction_dir)
    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "prompt_jsonl": args.prompt_jsonl,
        "base_script": str(base_path),
        "models": models,
        "methods": methods,
        "layers": args.layers,
        "train_ratio": args.train_ratio,
        "test_ratio": 1.0 - args.train_ratio,
        "split_unit": args.split_unit,
        "repeats": args.repeats,
        "seed": args.seed,
        "probability_csv": str(probability_path),
        "probability_column": args.probability_column,
        "object_direction_dir": str(object_direction_dir),
        "object_representation": args.object_representation,
        "attn_impl": args.attn_impl,
        "state_definition": "last_non_padding_prompt_token",
        "relation_order": list(RELATIONS),
        "relation_vectors_are_independent": True,
        "opposite_or_orthogonal_constraints": False,
        "audit": audit,
    }
    write_json(output_dir / "config.json", config)

    all_sample_rows: List[Dict[str, Any]] = []
    all_split_rows: List[Dict[str, Any]] = []
    alignment_frames: List[pd.DataFrame] = []

    for model_name in models:
        cache = extract_model_cache(
            args=args,
            step1=step1,
            model_name=model_name,
            spec=specs[model_name],
            records=records,
            prompt_rows=prompt_rows,
            cache_path=cache_dir / f"{model_name}.npz",
            errors_path=errors_path,
        )

        sids = np.asarray(cache["sid"], dtype=np.int64)
        subjects = np.asarray(cache["subject"], dtype=object)
        references = np.asarray(cache["reference"], dtype=object)
        labels = np.asarray([str(x) for x in cache["relation"].tolist()], dtype=object)
        layers = np.asarray(cache["decoder_block_index"], dtype=np.int64)
        states = np.asarray(cache["last_states"], dtype=np.float32)

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
            print(
                f"repeat={repeat} seed={split_seed} "
                f"train={len(train_idx)} ({len(train_idx)/len(sids):.3f}) "
                f"test={len(test_idx)} ({len(test_idx)/len(sids):.3f}) "
                f"train_counts={dict(Counter(labels[train_idx].tolist()))}",
                flush=True,
            )

            for index, sid in enumerate(sids.tolist()):
                all_split_rows.append({
                    "model": model_name,
                    "repeat": int(repeat),
                    "seed": split_seed,
                    "sid": int(sid),
                    "subject": str(subjects[index]),
                    "reference": str(references[index]),
                    "gt_internal": str(labels[index]),
                    "gold_relation": DISPLAY_RELATION[str(labels[index])],
                    "split": split_by_sid[int(sid)],
                })

            centers: List[np.ndarray] = []
            last_directions: List[np.ndarray] = []
            for layer_pos in range(len(layers)):
                center, directions = fit_relation_directions(
                    states[train_idx, layer_pos, :],
                    labels[train_idx],
                )
                centers.append(center)
                last_directions.append(directions)
            centers_array = np.stack(centers).astype(np.float32)
            last_directions_array = np.stack(last_directions).astype(np.float32)

            learned_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                learned_dir / f"{model_name}_last_own_repeat{repeat}.npz",
                relation_order=np.asarray(RELATIONS, dtype=object),
                decoder_block_index=layers.astype(np.int32),
                train_sid=sids[train_idx],
                test_sid=sids[test_idx],
                centers=centers_array,
                directions=last_directions_array,
            )

            object_file: Optional[Dict[str, Any]] = None
            if "object_to_last" in methods:
                object_file = load_object_direction_file(
                    directory=object_direction_dir,
                    model_name=model_name,
                    representation=args.object_representation,
                    repeat=repeat,
                )
                validate_object_direction_split(
                    object_file,
                    train_sid=sids[train_idx],
                    test_sid=sids[test_idx],
                )
                alignment_frames.append(
                    build_direction_alignment(
                        model_name=model_name,
                        repeat=repeat,
                        layers=layers,
                        last_directions=last_directions_array,
                        object_file=object_file,
                    )
                )

            for method in methods:
                rows = score_method(
                    model_name=model_name,
                    method=method,
                    repeat=repeat,
                    cache=cache,
                    probability_rows=probability_rows,
                    probability_column=args.probability_column,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    last_own_centers=centers_array,
                    last_own_directions=last_directions_array,
                    object_file=object_file,
                )
                all_sample_rows.extend(rows)

        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    samples = pd.DataFrame(all_sample_rows)
    if samples.empty:
        raise RuntimeError("No test rows produced")
    samples.to_csv(output_dir / "sample_test_metrics.csv", index=False)
    pd.DataFrame(all_split_rows).to_csv(
        output_dir / "split_assignments.csv", index=False
    )

    correlations = build_correlation_table(samples)
    correlations.to_csv(output_dir / "correlation_by_same_gold.csv", index=False)

    layer_summary = build_layer_summary(samples, correlations)
    layer_summary.to_csv(output_dir / "layer_summary.csv", index=False)

    best_layers = build_best_layers(layer_summary)
    best_layers.to_csv(output_dir / "best_layers.csv", index=False)

    if alignment_frames:
        alignment_df = pd.concat(alignment_frames, ignore_index=True)
    else:
        alignment_df = pd.DataFrame()
    alignment_df.to_csv(output_dir / "direction_alignment.csv", index=False)

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
    print(best_layers.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nLAST-OWN BEST-ACCURACY LAYER: SAME-GOLD CORRELATIONS")
    selected_parts: List[pd.DataFrame] = []
    for row in best_layers[best_layers["method"] == "last_own"].itertuples(index=False):
        part = correlations[
            (correlations["model"] == row.model)
            & (correlations["method"] == "last_own")
            & (correlations["repeat"] == row.repeat)
            & (correlations["layer"] == row.best_accuracy_layer)
            & (correlations["subset"] == "all")
        ]
        selected_parts.append(part)
    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
        print(
            selected[
                [
                    "model", "repeat", "layer", "gold_relation",
                    "similarity_metric", "N", "pearson_r", "pearson_p",
                    "spearman_r", "spearman_p",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )

    print("\nSaved:")
    for filename in (
        "sample_test_metrics.csv",
        "correlation_by_same_gold.csv",
        "layer_summary.csv",
        "best_layers.csv",
        "direction_alignment.csv",
    ):
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
