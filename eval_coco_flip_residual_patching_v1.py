#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COCO-two image-flip counterfactual residual patching.

The prompt stays unchanged.  The image is flipped horizontally for left/right
and vertically for above/below, so the expected relation becomes its opposite.

For each selected decoder block, copy the donor block-output residual state into
the counterfactual recipient at selected text-token groups:
subject, reference, both objects, prompt_last, or all_text.

Recovery is measured on the donor-vs-recipient relation margin:

    R = (m_patched - m_recipient) / (m_donor - m_recipient)

No CLIP, detector, box, centroid, trained probe, relation direction, or weight
update is used.  Ground truth is used only to define the known flip relation and
to select clean original/flip pairs for causal localization.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import shutil
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

VERSION = "coco-flip-residual-patching-v1"
RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
TOKEN_GROUPS = ("subject", "reference", "both", "prompt_last", "all_text")
DIRECTIONS = ("orig_to_flip", "flip_to_orig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-script", default="analyze_coco_centroid_generation_step1_v4.py")
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--relations", default="left,right,above,below")
    p.add_argument(
        "--layers",
        default="auto:4",
        help="'all', 'auto:N', or explicit zero-based block indices such as 3,7,11,15.",
    )
    p.add_argument("--token-groups", default="subject,reference,both,prompt_last,all_text")
    p.add_argument("--directions", default="orig_to_flip")
    p.add_argument(
        "--pair-mode",
        default="both_correct",
        choices=["both_correct", "both_opposite", "all"],
    )
    p.add_argument(
        "--control",
        default="random_text",
        choices=["none", "random_text"],
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-clean-pairs", type=int, default=None)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def import_file(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_subset(value: str, allowed: Sequence[str], label: str) -> List[str]:
    allowed = set(allowed)
    out: List[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported {label}: {item}; allowed={sorted(allowed)}")
        if item not in out:
            out.append(item)
    if not out:
        raise ValueError(f"{label} is empty")
    return out


def parse_layers(value: str, n_layers: int) -> List[int]:
    text = value.strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text.startswith("auto:"):
        stride = int(text.split(":", 1)[1])
        if stride <= 0:
            raise ValueError("auto stride must be positive")
        layers = list(range(stride - 1, n_layers, stride))
        if not layers or layers[-1] != n_layers - 1:
            layers.append(n_layers - 1)
        return sorted(set(layers))
    out: List[int] = []
    for raw in text.split(","):
        if not raw.strip():
            continue
        layer = int(raw)
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} outside 0..{n_layers - 1}")
        if layer not in out:
            out.append(layer)
    if not out:
        raise ValueError("No layers selected")
    return out


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported layer output type: {type(output).__name__}")


def replace_first(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    raise TypeError(type(output).__name__)


class CaptureBlockOutputs:
    def __init__(self, layers: Sequence[Any], indices: Sequence[int]) -> None:
        self.layers = layers
        self.indices = list(indices)
        self.handles: List[Any] = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> "CaptureBlockOutputs":
        for index in self.indices:
            def make_hook(layer_index: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    self.outputs[layer_index] = first_tensor(output).detach()
                return hook
            self.handles.append(self.layers[index].register_forward_hook(make_hook(index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


class PatchBlockOutput:
    """Patch one block output for multiple token groups in a single batched pass."""

    def __init__(
        self,
        layer: Any,
        donor_hidden: torch.Tensor,
        position_sets: Sequence[Sequence[int]],
        sequence_length: int,
    ) -> None:
        self.layer = layer
        self.donor_hidden = donor_hidden
        self.position_sets = [sorted(set(map(int, positions))) for positions in position_sets]
        self.sequence_length = int(sequence_length)
        self.handle: Optional[Any] = None
        self.events = 0

    def __enter__(self) -> "PatchBlockOutput":
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = first_tensor(output)
            if hidden.ndim != 3:
                raise RuntimeError(f"Expected [B,S,H], got {tuple(hidden.shape)}")
            batch, seq_len, hidden_size = hidden.shape
            if seq_len != self.sequence_length:
                raise RuntimeError(f"Sequence length changed: {seq_len} != {self.sequence_length}")
            if batch != len(self.position_sets):
                raise RuntimeError(f"Batch/condition mismatch: {batch} != {len(self.position_sets)}")
            donor = self.donor_hidden
            if tuple(donor.shape[1:]) != (seq_len, hidden_size):
                raise RuntimeError(f"Donor shape {tuple(donor.shape)} incompatible with {tuple(hidden.shape)}")
            donor = donor.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden.clone()
            for b, positions in enumerate(self.position_sets):
                valid = [p for p in positions if 0 <= p < seq_len]
                if not valid:
                    raise RuntimeError(f"Empty patch position set for row {b}")
                idx = torch.tensor(valid, device=hidden.device, dtype=torch.long)
                patched[b].index_copy_(0, idx, donor[0].index_select(0, idx))
            self.events += 1
            return replace_first(output, patched)

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ]
    for value in candidates:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("No language-model logits found")


def score_relations(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for relation in RELATIONS:
        ids = [int(x) for x in token_map[relation] if 0 <= int(x) < logits.numel()]
        if not ids:
            raise RuntimeError(f"No token variants for {relation}")
        idx = torch.tensor(ids, device=logits.device, dtype=torch.long)
        scores[relation] = float(logits.index_select(0, idx).max().detach().cpu())
    return scores


def run_forward(
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    capture_layers: Sequence[int] = (),
) -> Tuple[List[Dict[str, Any]], Dict[int, torch.Tensor]]:
    with torch.inference_mode():
        if capture_layers:
            with CaptureBlockOutputs(decoder_layers, capture_layers) as capture:
                outputs = model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            captured = dict(capture.outputs)
        else:
            outputs = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            captured = {}
    logits = extract_logits(outputs)[:, -1, :]
    results: List[Dict[str, Any]] = []
    for i in range(logits.shape[0]):
        scores = score_relations(logits[i], token_map)
        prediction = max(RELATIONS, key=lambda r: scores[r])
        results.append({"scores": scores, "prediction": prediction})
    del outputs, logits
    return results, captured


def flip_image(image: Image.Image, relation: str) -> Image.Image:
    if relation in ("left", "right"):
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if relation in ("above", "below"):
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    raise ValueError(relation)


def span_positions(span: Tuple[int, int]) -> List[int]:
    return list(range(int(span[0]), int(span[1]) + 1))


def build_conditions(
    token_groups: Sequence[str],
    control: str,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    prompt_last: int,
    text_positions: Sequence[int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    subject = sorted(set(map(int, subject_positions)))
    reference = sorted(set(map(int, reference_positions)))
    both = sorted(set(subject + reference))
    text = sorted(set(map(int, text_positions)))
    actual = {
        "subject": subject,
        "reference": reference,
        "both": both,
        "prompt_last": [int(prompt_last)],
        "all_text": text,
    }
    excluded = set(both)
    excluded.add(int(prompt_last))
    random_candidates = [p for p in text if p not in excluded]

    conditions: List[Dict[str, Any]] = []
    positions: List[List[int]] = []
    for group_index, group in enumerate(token_groups):
        pos = list(actual[group])
        conditions.append({
            "condition": group,
            "token_group": group,
            "control": False,
            "n_positions": len(pos),
        })
        positions.append(pos)
        if control == "random_text" and group != "all_text":
            if len(random_candidates) < len(pos):
                raise RuntimeError(f"Not enough random text tokens for {group}")
            rng = random.Random(seed + 1009 * (group_index + 1))
            rand_pos = sorted(rng.sample(random_candidates, len(pos)))
            conditions.append({
                "condition": f"{group}_random_text",
                "token_group": group,
                "control": True,
                "n_positions": len(rand_pos),
            })
            positions.append(rand_pos)
    return conditions, positions


def repeated_batch(
    processor: Any,
    rendered: str,
    image: Image.Image,
    repeats: int,
    device: torch.device,
    base: Any,
) -> Dict[str, Any]:
    batch = processor(
        text=[rendered] * repeats,
        images=[image] * repeats,
        return_tensors="pt",
        padding=True,
    )
    return base.move_batch(batch, device)


def eligible_pair(
    mode: str,
    original_prediction: str,
    flipped_prediction: str,
    original_relation: str,
    flipped_relation: str,
) -> bool:
    if mode == "both_correct":
        return original_prediction == original_relation and flipped_prediction == flipped_relation
    if mode == "both_opposite":
        return OPPOSITE.get(original_prediction) == flipped_prediction
    return True


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["axis"],
            row["direction"],
            int(row["layer"]),
            row["condition"],
            row["token_group"],
            bool(row["control"]),
        )
        grouped[key].append(row)

    result: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        axis, direction, layer, condition, token_group, control = key
        recovery = finite_values(item.get("recovery") for item in items)
        shifts = finite_values(item.get("margin_shift") for item in items)
        result.append({
            "axis": axis,
            "direction": direction,
            "layer": layer,
            "condition": condition,
            "token_group": token_group,
            "control": control,
            "n": len(items),
            "mean_recovery": float(np.mean(recovery)) if recovery else None,
            "median_recovery": float(np.median(recovery)) if recovery else None,
            "fraction_recovery_gt_0_25": float(np.mean([x > 0.25 for x in recovery])) if recovery else None,
            "fraction_recovery_gt_0_50": float(np.mean([x > 0.50 for x in recovery])) if recovery else None,
            "mean_margin_shift": float(np.mean(shifts)) if shifts else None,
            "donor_relation_rate": float(np.mean([
                item["patched_prediction"] == item["donor_relation"] for item in items
            ])),
            "recipient_relation_rate": float(np.mean([
                item["patched_prediction"] == item["recipient_relation"] for item in items
            ])),
        })

    lookup = {
        (row["axis"], row["direction"], row["layer"], row["token_group"], row["control"]): row
        for row in result
    }
    for row in result:
        if row["control"]:
            continue
        control = lookup.get((row["axis"], row["direction"], row["layer"], row["token_group"], True))
        if control and row["mean_recovery"] is not None and control["mean_recovery"] is not None:
            row["excess_recovery_vs_random"] = row["mean_recovery"] - control["mean_recovery"]
            row["excess_donor_rate_vs_random"] = row["donor_relation_rate"] - control["donor_relation_rate"]
        else:
            row["excess_recovery_vs_random"] = None
            row["excess_donor_rate_vs_random"] = None
    return sorted(result, key=lambda x: (x["axis"], x["direction"], x["layer"], x["control"], x["condition"]))


def report_text(model: str, seen: int, clean: int, counts: Mapping[str, int], rows: Sequence[Mapping[str, Any]]) -> str:
    actual = [r for r in rows if r["axis"] == "all" and not r["control"]]
    actual.sort(key=lambda r: (
        -(r["excess_recovery_vs_random"] if r["excess_recovery_vs_random"] is not None else -999.0),
        -(r["mean_recovery"] if r["mean_recovery"] is not None else -999.0),
    ))
    header = (
        f"{'Direction':<14}{'Layer':>7}{'Token group':>16}{'N':>7}"
        f"{'Mean R':>10}{'Median R':>11}{'R>0.5':>9}{'Donor%':>9}{'ExR':>9}"
    )
    lines = [
        "=" * len(header),
        "COCO IMAGE-FLIP COUNTERFACTUAL RESIDUAL PATCHING",
        f"model={model} | seen={seen} | clean_pairs={clean}",
        "baseline: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    def f4(v: Any) -> str:
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "-"
    for row in actual[:40]:
        lines.append(
            f"{row['direction']:<14}{row['layer']:>7}{row['token_group']:>16}{row['n']:>7}"
            f"{f4(row['mean_recovery']):>10}{f4(row['median_recovery']):>11}"
            f"{f4(row['fraction_recovery_gt_0_50']):>9}{f4(row['donor_relation_rate']):>9}"
            f"{f4(row['excess_recovery_vs_random']):>9}"
        )
    lines += [
        "",
        "Interpretation:",
        "- R≈0: this block-output/token group does not transfer the donor spatial state.",
        "- R≈1: patching it nearly transfers the donor relation into the counterfactual run.",
        "- ExR should be positive; otherwise the effect is not object-group specific.",
        "- Horizontal and vertical flips are summarized separately in patch_summary.csv.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    relations = parse_subset(args.relations, RELATIONS, "relation")
    token_groups = parse_subset(args.token_groups, TOKEN_GROUPS, "token group")
    directions = parse_subset(args.directions, DIRECTIONS, "direction")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = import_file(Path(args.base_script), "_coco_flip_base")
    data_module = base.import_two_object_module()
    records, audit = data_module.load_records(args.dataset, Path(args.data_root), args.max_samples)
    prompt_rows = base.load_standard_prompts(Path(args.prompt_jsonl))

    specs = base.merged_model_specs(data_module)
    if args.model not in specs:
        raise ValueError(f"Unknown model {args.model}; available={sorted(specs)}")
    spec = specs[args.model]

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output directory not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_dir / "baseline_pairs.jsonl"
    patch_path = out_dir / "patch_results.jsonl"
    error_path = out_dir / "errors.jsonl"

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers lacks {spec.model_class}")
    kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl

    print(f"Version: {VERSION}")
    print(f"Loading {args.model}: {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = base.resolve_decoder_layers(model)
    layers = parse_layers(args.layers, len(decoder_layers))
    token_map = base.relation_token_variants(processor.tokenizer)

    config = {
        "version": VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "n_decoder_layers": len(decoder_layers),
        "layers": layers,
        "token_groups": token_groups,
        "directions": directions,
        "relations": relations,
        "pair_mode": args.pair_mode,
        "control": args.control,
        "max_samples": args.max_samples,
        "max_clean_pairs": args.max_clean_pairs,
        "audit": audit,
        "patch_location": "decoder_block_output",
        "uses_external_model": False,
        "uses_visual_coordinates": False,
        "uses_centroid_prediction": False,
        "updates_model_weights": False,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Decoder={decoder_path}, layers={len(decoder_layers)}, scan={layers}")
    print(f"Groups={token_groups}, directions={directions}, pair_mode={args.pair_mode}")

    seen = 0
    clean = 0
    counts: Counter = Counter()
    start = time.time()

    try:
        for record in tqdm(records, desc=f"flip-patching:{args.model}"):
            if args.max_clean_pairs is not None and clean >= args.max_clean_pairs:
                break
            sid = int(record.sid)
            seen += 1
            try:
                row = prompt_rows[sid]
                subject = str(row["subject"])
                reference = str(row["reference"])
                question = str(row["question_text"])
                original_relation = base.normalize_relation(row["answer_raw"])
                if original_relation not in relations:
                    continue
                flipped_relation = OPPOSITE[original_relation]
                axis = "horizontal" if original_relation in ("left", "right") else "vertical"

                original_image = base.record_image(record).convert("RGB")
                flipped_image = flip_image(original_image, original_relation)
                rendered = base.build_prompt(processor, question)

                original_batch = base.move_batch(
                    processor(text=[rendered], images=[original_image], return_tensors="pt"),
                    device,
                )
                flipped_batch = base.move_batch(
                    processor(text=[rendered], images=[flipped_image], return_tensors="pt"),
                    device,
                )
                original_ids = original_batch["input_ids"][0].detach().cpu().tolist()
                flipped_ids = flipped_batch["input_ids"][0].detach().cpu().tolist()
                if original_ids != flipped_ids:
                    raise RuntimeError("Original/flip tokenization differs")

                subject_span, reference_span = base.locate_object_spans(
                    processor.tokenizer, original_ids, subject, reference
                )
                subject_positions = span_positions(subject_span)
                reference_positions = span_positions(reference_span)
                prompt_last = len(original_ids) - 1
                visual_indices = base.resolve_visual_indices(
                    model, processor, original_batch, original_ids
                )
                visual_set = set(map(int, visual_indices))
                text_positions = [i for i in range(len(original_ids)) if i not in visual_set]

                original_result, original_capture = run_forward(
                    model, original_batch, token_map, decoder_layers, layers
                )
                flipped_result, flipped_capture = run_forward(
                    model, flipped_batch, token_map, decoder_layers, layers
                )
                original_result = original_result[0]
                flipped_result = flipped_result[0]
                op = original_result["prediction"]
                fp = flipped_result["prediction"]
                oc = op == original_relation
                fc = fp == flipped_relation

                counts["eligible_relation_seen"] += 1
                counts["original_correct"] += int(oc)
                counts["flip_correct"] += int(fc)
                counts["both_correct"] += int(oc and fc)
                counts["predictions_opposite"] += int(OPPOSITE.get(op) == fp)

                is_eligible = eligible_pair(
                    args.pair_mode, op, fp, original_relation, flipped_relation
                )
                append_jsonl(pair_path, {
                    "sid": sid,
                    "subject": subject,
                    "reference": reference,
                    "axis": axis,
                    "original_relation": original_relation,
                    "flipped_relation": flipped_relation,
                    "original_prediction": op,
                    "flipped_prediction": fp,
                    "original_correct": oc,
                    "flipped_correct": fc,
                    "eligible": is_eligible,
                    "original_scores": original_result["scores"],
                    "flipped_scores": flipped_result["scores"],
                    "subject_span": list(subject_span),
                    "reference_span": list(reference_span),
                    "prompt_last": prompt_last,
                })
                if not is_eligible:
                    continue

                clean += 1
                conditions, position_sets = build_conditions(
                    token_groups,
                    args.control,
                    subject_positions,
                    reference_positions,
                    prompt_last,
                    text_positions,
                    args.seed * 1000003 + sid * 101,
                )
                repeats = len(conditions)

                repeated: Dict[str, Dict[str, Any]] = {}
                if "orig_to_flip" in directions:
                    repeated["orig_to_flip"] = repeated_batch(
                        processor, rendered, flipped_image, repeats, device, base
                    )
                if "flip_to_orig" in directions:
                    repeated["flip_to_orig"] = repeated_batch(
                        processor, rendered, original_image, repeats, device, base
                    )

                for direction in directions:
                    if direction == "orig_to_flip":
                        donor_relation = original_relation
                        recipient_relation = flipped_relation
                        donor_result = original_result
                        recipient_result = flipped_result
                        donor_capture = original_capture
                    else:
                        donor_relation = flipped_relation
                        recipient_relation = original_relation
                        donor_result = flipped_result
                        recipient_result = original_result
                        donor_capture = flipped_capture

                    donor_margin = (
                        donor_result["scores"][donor_relation]
                        - donor_result["scores"][recipient_relation]
                    )
                    recipient_margin = (
                        recipient_result["scores"][donor_relation]
                        - recipient_result["scores"][recipient_relation]
                    )
                    denominator = donor_margin - recipient_margin
                    batch = repeated[direction]
                    if batch["input_ids"].shape[1] != len(original_ids):
                        raise RuntimeError("Repeated batch sequence length differs")

                    for layer in layers:
                        with PatchBlockOutput(
                            decoder_layers[layer],
                            donor_capture[layer],
                            position_sets,
                            len(original_ids),
                        ) as patcher:
                            patched, _ = run_forward(
                                model, batch, token_map, decoder_layers, ()
                            )
                        if patcher.events != 1:
                            raise RuntimeError(f"Expected one patch event, got {patcher.events}")

                        for condition, result in zip(conditions, patched):
                            patched_margin = (
                                result["scores"][donor_relation]
                                - result["scores"][recipient_relation]
                            )
                            shift = patched_margin - recipient_margin
                            recovery = shift / denominator if abs(denominator) > 1e-8 else None
                            append_jsonl(patch_path, {
                                "sid": sid,
                                "axis": axis,
                                "direction": direction,
                                "original_relation": original_relation,
                                "flipped_relation": flipped_relation,
                                "donor_relation": donor_relation,
                                "recipient_relation": recipient_relation,
                                "layer": layer,
                                **condition,
                                "donor_prediction": donor_result["prediction"],
                                "recipient_prediction": recipient_result["prediction"],
                                "patched_prediction": result["prediction"],
                                "donor_margin": donor_margin,
                                "recipient_margin": recipient_margin,
                                "patched_margin": patched_margin,
                                "margin_shift": shift,
                                "recovery": recovery,
                            })

                if args.print_every > 0 and clean % args.print_every == 0:
                    tqdm.write(
                        f"\n[clean {clean}] sid={sid} "
                        f"{original_relation}->{flipped_relation}, "
                        f"pred={op}->{fp}, layers={len(layers)}, "
                        f"conditions={len(conditions)}"
                    )

                del repeated, original_capture, flipped_capture
                if clean % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as exc:
                append_jsonl(error_path, {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-24:],
                })
                tqdm.write(f"\n[ERROR] sid={sid}: {type(exc).__name__}: {exc}")

        patch_rows = read_jsonl(patch_path)
        if not patch_rows:
            raise RuntimeError(
                "No patch rows. Inspect baseline_pairs.jsonl/errors.jsonl or relax --pair-mode."
            )

        summary_rows = summarize(patch_rows)
        summary_rows += summarize([{**row, "axis": "all"} for row in patch_rows])
        summary_rows = sorted(
            summary_rows,
            key=lambda r: (r["axis"], r["direction"], r["layer"], r["control"], r["condition"]),
        )
        write_csv(out_dir / "patch_summary.csv", summary_rows)

        report = report_text(args.model, seen, clean, counts, summary_rows)
        print("\n" + report)
        (out_dir / "report.txt").write_text(report, encoding="utf-8")

        top = [
            row for row in summary_rows if row["axis"] == "all" and not row["control"]
        ]
        top.sort(key=lambda r: (
            -(r["excess_recovery_vs_random"] if r["excess_recovery_vs_random"] is not None else -999.0),
            -(r["mean_recovery"] if r["mean_recovery"] is not None else -999.0),
        ))
        summary_json = {
            "config": config,
            "seen": seen,
            "clean_pairs": clean,
            "baseline_counts": dict(counts),
            "n_patch_rows": len(patch_rows),
            "elapsed_minutes": (time.time() - start) / 60.0,
            "top_aggregate_conditions": top[:50],
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("Saved:")
        for name in (
            "report.txt",
            "patch_summary.csv",
            "summary.json",
            "baseline_pairs.jsonl",
            "patch_results.jsonl",
        ):
            print(" ", out_dir / name)
        if error_path.exists():
            print(" ", error_path)

    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
