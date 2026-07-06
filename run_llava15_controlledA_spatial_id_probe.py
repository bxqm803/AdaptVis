#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe object-token position components in LLaVA-1.5 on Controlled_Images_A.

This script evaluates a fixed-layer, horizontal (left/right) analogue of the
"spatial ID" test. It does NOT claim to recover a full 2-D spatial grid: the
Controlled_Images_A labels provide only relative left/right supervision.

For each horizontal sample, it:
  1. removes answer-label words from the prompt by default;
  2. extracts the final sub-token state of the subject and reference phrases
     after selected decoder blocks (default: 14 and 17, zero-based);
  3. forms object-centred residuals using train-only means;
  4. learns a train-only left-to-right axis;
  5. evaluates subject, reference, and pairwise relation decoding on a held-out
     split grouped by unordered object pair.

Run from the AdaptVis repository root.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "a272c74"
HORIZONTAL = {"left", "right"}

QUESTION_RE = re.compile(
    r"Where\s+(?P<verb>is|are)\s+(?:the\s+)?(?P<subject>.+?)\s+"
    r"in\s+relation\s+to\s+(?:the\s+)?(?P<reference>.+?)\?\s*Answer",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class PromptMeta:
    subject: str
    reference: str
    verb: str


@dataclass
class SampleRecord:
    sid: int
    relation: str
    subject: str
    reference: str
    prompt: str
    group: str
    states: Dict[int, Dict[str, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-layer spatial-position probe for HF LLaVA-1.5 on Controlled_Images_A."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--layers", default="14,17", help="Zero-based decoder block indices, e.g. 14,17.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument(
        "--split-unit",
        choices=["pair", "sample"],
        default="pair",
        help="'pair' keeps an unordered subject/reference pair in one split.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["clean", "original"],
        default="clean",
        help="'clean' removes 'Answer with left, right, on or under' to avoid answer-word leakage.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after horizontal filtering.")
    parser.add_argument("--output", default="output/llava15_controlledA_spatial_id_probe")
    parser.add_argument("--save-states", action="store_true", help="Also save token states and learned axes in .npz.")
    parser.add_argument("--print-first", type=int, default=5)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layers(text: str) -> List[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or min(values) < 0:
        raise ValueError(f"Invalid --layers: {text!r}")
    return values


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_relation(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().lower()


def canonical_object(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_prompt(prompt: str) -> PromptMeta:
    match = QUESTION_RE.search(prompt)
    if match is None:
        raise ValueError(f"Could not parse subject/reference from prompt:\n{prompt}")
    return PromptMeta(
        subject=canonical_object(match.group("subject")),
        reference=canonical_object(match.group("reference")),
        verb=match.group("verb").lower(),
    )


def make_clean_prompt(meta: PromptMeta) -> str:
    # Preserve the original singular/plural wording while removing explicit answer labels.
    return (
        f"<image>\nUSER: Where {meta.verb} the {meta.subject} "
        f"in relation to the {meta.reference}?\nASSISTANT:"
    )


def load_prompt_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            row = json.loads(line)
            if int(row.get("id", expected_id)) != expected_id:
                raise ValueError(
                    f"Prompt IDs must be contiguous and aligned with dataset order. "
                    f"Expected {expected_id}, found {row.get('id')}"
                )
            rows.append(row)
    return rows


def extract_images_from_batch(batch: Mapping[str, Any]) -> Iterable[Any]:
    # Matches the existing run_llava15_hf_eps_logit_lens.py iteration convention.
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    result: List[int] = []
    width = len(needle)
    for start in range(0, len(haystack) - width + 1):
        if list(haystack[start : start + width]) == list(needle):
            result.append(start)
    return result


def find_phrase_last_token(input_ids: Sequence[int], tokenizer, phrase: str) -> int:
    """Locate the final sub-token of an object phrase in the prompt token sequence.

    The leading-space variant is needed for LLaMA-style tokenizers, where a word
    after 'the ' normally uses a whitespace-prefixed token realization.
    """
    candidates: List[Tuple[int, int, List[int]]] = []
    seen: set[Tuple[int, ...]] = set()
    for variant in (" " + phrase, phrase):
        token_ids = tokenizer(variant, add_special_tokens=False).input_ids
        key = tuple(int(x) for x in token_ids)
        if not key or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, token_ids):
            candidates.append((start, start + len(token_ids) - 1, list(token_ids)))

    if not candidates:
        raise ValueError(
            f"Could not find phrase {phrase!r} in tokenized prompt. "
            "This normally means the prompt template changed."
        )

    # Prefer the longest token match, then require that it is unambiguous.
    max_len = max(len(ids) for _, _, ids in candidates)
    best = [(start, end) for start, end, ids in candidates if len(ids) == max_len]
    unique = sorted(set(best))
    if len(unique) != 1:
        raise ValueError(
            f"Ambiguous token match for phrase {phrase!r}: {unique}. "
            "Use a more explicit prompt template before running this probe."
        )
    return unique[0][1]


def map_text_index_to_merged_index(text_index: int, image_index: int, image_seq_length: int) -> int:
    if text_index == image_index:
        raise ValueError("The requested object token cannot be the <image> placeholder.")
    if text_index < image_index:
        return int(text_index)
    return int(text_index + image_seq_length - 1)


class SelectedTokenCapture:
    """Capture subject/reference residuals after selected decoder blocks only."""

    def __init__(self, language_model, layers: Sequence[int]) -> None:
        self.layers = list(language_model.model.layers)
        if max(layers) >= len(self.layers):
            raise ValueError(
                f"Requested layer {max(layers)} but model has only {len(self.layers)} decoder blocks."
            )
        self.selected = set(int(layer) for layer in layers)
        self.subject_index = -1
        self.reference_index = -1
        self.states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.handles = [
            self.layers[layer].register_forward_hook(self._make_hook(layer))
            for layer in sorted(self.selected)
        ]

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"Expected [1, L, H] decoder output, got {tuple(hidden.shape)}")
            if not (0 <= self.subject_index < hidden.shape[1]):
                raise RuntimeError(
                    f"Subject merged index {self.subject_index} is outside sequence length {hidden.shape[1]}"
                )
            if not (0 <= self.reference_index < hidden.shape[1]):
                raise RuntimeError(
                    f"Reference merged index {self.reference_index} is outside sequence length {hidden.shape[1]}"
                )
            self.states[layer_index] = {
                "subject": hidden[0, self.subject_index].detach().float().cpu().clone(),
                "reference": hidden[0, self.reference_index].detach().float().cpu().clone(),
            }

        return hook

    def begin(self, subject_index: int, reference_index: int) -> None:
        self.subject_index = int(subject_index)
        self.reference_index = int(reference_index)
        self.states = {}

    def collect(self) -> Dict[int, Dict[str, torch.Tensor]]:
        missing = sorted(self.selected - set(self.states))
        if missing:
            raise RuntimeError(f"No residuals captured for layers: {missing}")
        return self.states

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def choose_group_split(
    records: Sequence[SampleRecord],
    *,
    test_fraction: float,
    seed: int,
    split_unit: str,
) -> Tuple[set[int], set[int]]:
    """Choose a deterministic split with both labels represented.

    For pair splitting, several candidate group assignments are sampled and the
    one closest to the requested size/class balance is kept.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must lie in (0, 1).")

    if split_unit == "sample":
        group_of = {record.sid: str(record.sid) for record in records}
    else:
        group_of = {record.sid: record.group for record in records}

    group_to_records: Dict[str, List[SampleRecord]] = defaultdict(list)
    for record in records:
        group_to_records[group_of[record.sid]].append(record)
    groups = sorted(group_to_records)
    if len(groups) < 2:
        raise RuntimeError("Need at least two split groups.")

    total = len(records)
    total_by_label = Counter(record.relation for record in records)
    target_test = total * test_fraction
    target_by_label = {label: count * test_fraction for label, count in total_by_label.items()}

    best: Optional[Tuple[float, set[str]]] = None
    for trial in range(512):
        rng = random.Random(seed + 1009 * trial)
        test_groups = {group for group in groups if rng.random() < test_fraction}
        if not test_groups or len(test_groups) == len(groups):
            continue
        test_records = [record for group in test_groups for record in group_to_records[group]]
        test_by_label = Counter(record.relation for record in test_records)
        if any(test_by_label[label] == 0 for label in HORIZONTAL):
            continue
        score = abs(len(test_records) - target_test)
        score += 3.0 * sum(abs(test_by_label[label] - target_by_label[label]) for label in HORIZONTAL)
        if best is None or score < best[0]:
            best = (score, test_groups)

    if best is None:
        raise RuntimeError("Could not construct a split containing both left and right examples.")

    test_sids = {record.sid for group in best[1] for record in group_to_records[group]}
    train_sids = {record.sid for record in records} - test_sids
    return train_sids, test_sids


def position_from_occurrence(relation: str, role: str) -> str:
    if role == "subject":
        return relation
    if role == "reference":
        return "right" if relation == "left" else "left"
    raise ValueError(f"Unknown role: {role}")


def make_occurrences(records: Sequence[SampleRecord], layer: int) -> List[Dict[str, Any]]:
    occurrences: List[Dict[str, Any]] = []
    for record in records:
        for role, name in (("subject", record.subject), ("reference", record.reference)):
            occurrences.append(
                {
                    "sid": record.sid,
                    "role": role,
                    "object": name,
                    "position": position_from_occurrence(record.relation, role),
                    "vector": record.states[layer][role],
                }
            )
    return occurrences


def train_object_means(train_occurrences: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    by_object: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in train_occurrences:
        by_object[item["object"]].append(item)

    means: Dict[str, np.ndarray] = {}
    dropped: Dict[str, int] = {}
    for name, items in by_object.items():
        positions = {item["position"] for item in items}
        # This is necessary for centre-subtraction to remove object semantics
        # rather than merely reproduce one class's mean.
        if positions != HORIZONTAL:
            dropped[name] = len(items)
            continue
        means[name] = np.mean(np.stack([item["vector"] for item in items], axis=0), axis=0)
    return means, dropped


def score_layer(
    records: Sequence[SampleRecord],
    *,
    layer: int,
    train_sids: set[int],
    test_sids: set[int],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    train_records = [record for record in records if record.sid in train_sids]
    test_records = [record for record in records if record.sid in test_sids]
    train_occurrences = make_occurrences(train_records, layer)
    test_occurrences = make_occurrences(test_records, layer)

    object_means, dropped_objects = train_object_means(train_occurrences)

    def centered(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for item in items:
            mean = object_means.get(item["object"])
            if mean is None:
                continue
            copied = dict(item)
            copied["residual"] = item["vector"] - mean
            result.append(copied)
        return result

    centered_train = centered(train_occurrences)
    centered_test = centered(test_occurrences)
    if not centered_train or not centered_test:
        raise RuntimeError(
            f"Layer {layer}: no usable centred train/test occurrences. "
            "Try --split-unit sample or inspect object coverage."
        )

    by_pos_train: Dict[str, List[np.ndarray]] = defaultdict(list)
    for item in centered_train:
        by_pos_train[item["position"]].append(item["residual"])
    if any(not by_pos_train[position] for position in HORIZONTAL):
        raise RuntimeError(f"Layer {layer}: missing a horizontal class after centering.")

    mu_left = np.mean(np.stack(by_pos_train["left"], axis=0), axis=0)
    mu_right = np.mean(np.stack(by_pos_train["right"], axis=0), axis=0)
    axis = mu_right - mu_left
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm < 1e-8:
        raise RuntimeError(f"Layer {layer}: learnt horizontal axis has near-zero norm.")
    axis = axis / axis_norm

    train_scores = {position: [] for position in HORIZONTAL}
    for item in centered_train:
        train_scores[item["position"]].append(float(np.dot(item["residual"], axis)))
    threshold = 0.5 * (float(np.mean(train_scores["left"])) + float(np.mean(train_scores["right"])))

    evaluated: List[Dict[str, Any]] = []
    for item in centered_test:
        score = float(np.dot(item["residual"], axis))
        prediction = "right" if score > threshold else "left"
        enriched = dict(item)
        enriched["score"] = score
        enriched["prediction"] = prediction
        enriched["correct"] = prediction == item["position"]
        evaluated.append(enriched)

    def accuracy(items: Sequence[Dict[str, Any]]) -> Optional[float]:
        if not items:
            return None
        return float(np.mean([item["correct"] for item in items]))

    subject_eval = [item for item in evaluated if item["role"] == "subject"]
    reference_eval = [item for item in evaluated if item["role"] == "reference"]

    score_by_sid_role = {(item["sid"], item["role"]): item["score"] for item in evaluated}
    pair_predictions: List[bool] = []
    pair_total = 0
    for record in test_records:
        key_subject = (record.sid, "subject")
        key_reference = (record.sid, "reference")
        if key_subject not in score_by_sid_role or key_reference not in score_by_sid_role:
            continue
        pair_total += 1
        predicted_relation = "right" if score_by_sid_role[key_subject] > score_by_sid_role[key_reference] else "left"
        pair_predictions.append(predicted_relation == record.relation)

    summary = {
        "layer": int(layer),
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "train_occurrences": len(train_occurrences),
        "test_occurrences": len(test_occurrences),
        "objects_with_balanced_train_positions": len(object_means),
        "objects_dropped_unbalanced_train_positions": len(dropped_objects),
        "usable_test_occurrences": len(evaluated),
        "subject_test_n": len(subject_eval),
        "reference_test_n": len(reference_eval),
        "pair_test_n": pair_total,
        "occurrence_accuracy": accuracy(evaluated),
        "subject_position_accuracy": accuracy(subject_eval),
        "reference_position_accuracy": accuracy(reference_eval),
        "pairwise_relation_accuracy": (float(np.mean(pair_predictions)) if pair_predictions else None),
        "train_axis_margin": float(np.mean(train_scores["right"]) - np.mean(train_scores["left"])),
        "threshold": float(threshold),
        "axis_norm_before_normalization": axis_norm,
        "test_score_means": {
            position: float(np.mean([item["score"] for item in evaluated if item["position"] == position]))
            if any(item["position"] == position for item in evaluated)
            else None
            for position in ("left", "right")
        },
    }
    artifacts = {
        "axis": axis.astype(np.float32),
        "object_names": np.array(sorted(object_means), dtype=object),
        "object_means": np.stack([object_means[name] for name in sorted(object_means)], axis=0).astype(np.float32),
        "threshold": np.array([threshold], dtype=np.float32),
    }
    return summary, artifacts


def load_model_and_processor(args: argparse.Namespace):
    dtype = resolve_dtype(args.dtype)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    model.eval().to(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    # The old LLaVA-1.5 processor checkpoint does not serialize these fields,
    # but recent Transformers versions require them to expand <image> into the
    # correct number of visual placeholder tokens.
    vision_config = getattr(model.config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", None)
    if patch_size is None:
        raise RuntimeError(
            "Could not recover the vision patch size from model.config.vision_config."
        )
    processor.patch_size = int(patch_size)

    feature_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    processor.vision_feature_select_strategy = str(feature_strategy)
    processor.num_additional_image_tokens = 1 if feature_strategy == "full" else 0

    return model, processor


@torch.inference_mode()
def extract_states_for_sample(
    *,
    model: LlavaForConditionalGeneration,
    processor,
    capture: SelectedTokenCapture,
    image: Any,
    prompt: str,
    meta: PromptMeta,
    layers: Sequence[int],
    device: str,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Any]]:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to(device)
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=model_dtype)

    input_ids = [int(token) for token in inputs["input_ids"][0].detach().cpu().tolist()]
    image_token_id = int(model.config.image_token_index)
    image_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]
    image_seq_length = int(getattr(model.config, "image_seq_length", 0))
    if image_seq_length <= 0:
        raise RuntimeError("model.config.image_seq_length is required to map text tokens after <image>.")

    subject_text_index = find_phrase_last_token(input_ids, processor.tokenizer, meta.subject)
    reference_text_index = find_phrase_last_token(input_ids, processor.tokenizer, meta.reference)

    # HF LLaVA has used both conventions across transformers versions:
    # (a) one <image> placeholder expanded inside model.forward;
    # (b) image_seq_length repeated <image> placeholders expanded by the processor.
    # Text-token indices are shifted only in convention (a).
    if len(image_positions) == 1:
        image_layout = "single_placeholder_expanded_in_model"
        image_index = image_positions[0]
        subject_merged_index = map_text_index_to_merged_index(subject_text_index, image_index, image_seq_length)
        reference_merged_index = map_text_index_to_merged_index(reference_text_index, image_index, image_seq_length)
        expected_merged_length = len(input_ids) + image_seq_length - 1
    elif len(image_positions) >= image_seq_length:
        image_layout = "processor_expanded_image_placeholders"
        image_index = image_positions[0]
        subject_merged_index = subject_text_index
        reference_merged_index = reference_text_index
        expected_merged_length = len(input_ids)
    else:
        raise RuntimeError(
            f"Unsupported number of <image> placeholder tokens: {len(image_positions)}. "
            f"Expected 1 or at least image_seq_length={image_seq_length}."
        )
    capture.begin(subject_merged_index, reference_merged_index)

    outputs = model(
        **inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    actual_merged_length = int(outputs.logits.shape[1])
    if actual_merged_length != expected_merged_length:
        raise RuntimeError(
            "Unexpected merged sequence length: "
            f"expected {expected_merged_length}, got {actual_merged_length}. "
            "The image-token mapping may need updating for this transformers version."
        )

    captured = capture.collect()
    states = {
        int(layer): {
            "subject": captured[layer]["subject"].numpy().astype(np.float32),
            "reference": captured[layer]["reference"].numpy().astype(np.float32),
        }
        for layer in layers
    }
    debug = {
        "input_length": len(input_ids),
        "merged_length": actual_merged_length,
        "image_text_index": image_index,
        "image_placeholder_count": len(image_positions),
        "image_token_layout": image_layout,
        "subject_text_index": subject_text_index,
        "reference_text_index": reference_text_index,
        "subject_merged_index": subject_merged_index,
        "reference_merged_index": reference_merged_index,
        "subject_final_token": processor.tokenizer.decode(
            [input_ids[subject_text_index]], skip_special_tokens=False
        ),
        "reference_final_token": processor.tokenizer.decode(
            [input_ids[reference_text_index]], skip_special_tokens=False
        ),
    }
    return states, debug


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    layers = parse_layers(args.layers)
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = Path("prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    prompt_rows = load_prompt_rows(prompt_path)
    model, processor = load_model_and_processor(args)
    print(f"Loaded {args.model}@{args.revision}; decoder blocks={len(model.language_model.model.layers)}")
    print(f"Fixed probe layers (zero-based block indices): {layers}")
    print(f"Prompt mode: {args.prompt_mode}")

    dataset = get_dataset("Controlled_Images_A", image_preprocess=None, download=args.download)
    total_available = min(len(dataset), len(prompt_rows))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=repository_default_collate,
    )
    capture = SelectedTokenCapture(model.language_model, layers)
    records: List[SampleRecord] = []
    skipped: Counter[str] = Counter()
    debug_first: List[Dict[str, Any]] = []

    try:
        sid = 0
        progress = tqdm(total=total_available, desc="Extracting Controlled_A token states")
        for batch in loader:
            for image in extract_images_from_batch(batch):
                if sid >= total_available:
                    break
                row = prompt_rows[sid]
                relation = normalize_relation(row["answer"])
                if relation not in HORIZONTAL:
                    skipped[relation] += 1
                elif args.limit is not None and len(records) >= args.limit:
                    skipped["limit"] += 1
                else:
                    meta = parse_prompt(str(row["question"]))
                    prompt = make_clean_prompt(meta) if args.prompt_mode == "clean" else str(row["question"])
                    states, debug = extract_states_for_sample(
                        model=model,
                        processor=processor,
                        capture=capture,
                        image=image,
                        prompt=prompt,
                        meta=meta,
                        layers=layers,
                        device=args.device,
                    )
                    group = " || ".join(sorted((meta.subject, meta.reference)))
                    records.append(
                        SampleRecord(
                            sid=sid,
                            relation=relation,
                            subject=meta.subject,
                            reference=meta.reference,
                            prompt=prompt,
                            group=group,
                            states=states,
                        )
                    )
                    if len(debug_first) < args.print_first:
                        debug_first.append(
                            {
                                "sid": sid,
                                "relation": relation,
                                "subject": meta.subject,
                                "reference": meta.reference,
                                "prompt": prompt,
                                **debug,
                            }
                        )
                sid += 1
                progress.update(1)
            if sid >= total_available:
                break
        progress.close()
    finally:
        capture.close()

    if len(records) < 20:
        raise RuntimeError(f"Only {len(records)} horizontal samples were collected; expected many more.")

    train_sids, test_sids = choose_group_split(
        records,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    per_layer: Dict[str, Dict[str, Any]] = {}
    artifacts: Dict[str, np.ndarray] = {}
    for layer in layers:
        summary, layer_artifacts = score_layer(
            records,
            layer=layer,
            train_sids=train_sids,
            test_sids=test_sids,
        )
        per_layer[str(layer)] = summary
        for name, value in layer_artifacts.items():
            artifacts[f"layer_{layer}_{name}"] = value

    payload = {
        "metadata": {
            "model": args.model,
            "revision": args.revision,
            "dataset": "Controlled_Images_A",
            "prompt_file": str(prompt_path),
            "prompt_mode": args.prompt_mode,
            "relations": sorted(HORIZONTAL),
            "layers_zero_based": layers,
            "dtype": args.dtype,
            "seed": args.seed,
            "split_unit": args.split_unit,
            "test_fraction": args.test_fraction,
            "interpretation": (
                "A positive direction is defined from train-only object-centred residuals: "
                "right minus left. Pairwise relation is decoded by comparing subject and "
                "reference scores along that same axis."
            ),
        },
        "collection": {
            "total_prompt_rows": len(prompt_rows),
            "total_dataset_items_seen": total_available,
            "horizontal_samples_used": len(records),
            "skipped": dict(skipped),
            "relation_counts": dict(Counter(record.relation for record in records)),
            "unique_unordered_pairs": len({record.group for record in records}),
            "train_sids": sorted(train_sids),
            "test_sids": sorted(test_sids),
        },
        "debug_first_samples": debug_first,
        "results": per_layer,
    }

    json_path = output_base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResults")
    for layer in layers:
        result = per_layer[str(layer)]
        def fmt(value: Optional[float]) -> str:
            return "NA" if value is None else f"{value:.3f}"
        print(
            f"  L{layer}: occurrence={fmt(result['occurrence_accuracy'])} | "
            f"subject={fmt(result['subject_position_accuracy'])} | "
            f"reference={fmt(result['reference_position_accuracy'])} | "
            f"pairwise={fmt(result['pairwise_relation_accuracy'])} "
            f"(pair n={result['pair_test_n']})"
        )
    print(f"Saved summary: {json_path}")

    if args.save_states:
        ordered = sorted(records, key=lambda item: item.sid)
        arrays: Dict[str, np.ndarray] = {
            "sid": np.array([record.sid for record in ordered], dtype=np.int64),
            "relation": np.array([record.relation for record in ordered], dtype=object),
            "subject_name": np.array([record.subject for record in ordered], dtype=object),
            "reference_name": np.array([record.reference for record in ordered], dtype=object),
            "is_train": np.array([record.sid in train_sids for record in ordered], dtype=np.bool_),
        }
        for layer in layers:
            arrays[f"layer_{layer}_subject"] = np.stack(
                [record.states[layer]["subject"] for record in ordered], axis=0
            ).astype(np.float32)
            arrays[f"layer_{layer}_reference"] = np.stack(
                [record.states[layer]["reference"] for record in ordered], axis=0
            ).astype(np.float32)
        arrays.update(artifacts)
        npz_path = output_base.with_suffix(".npz")
        np.savez_compressed(npz_path, **arrays)
        print(f"Saved states/axes: {npz_path}")


if __name__ == "__main__":
    main()
