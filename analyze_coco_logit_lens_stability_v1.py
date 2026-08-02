#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen prompt-last Logit-Lens stability analysis for COCO two-object relations.

Main question
-------------
Do normally generated correct answers become LM-head-readable and remain stable
through most middle/late decoder layers, while wrong answers show late flips,
competition, or stable wrong trajectories?

For decoder block l, the script reads the prompt-last state from
``hidden_states[l + 1]`` and applies the model's frozen final normalization and
frozen LM head:

    scores_l(r) = max_{token variant of r} LMHead(FinalNorm(h_l))[token]

for r in {left, right, above, below}. No probe is trained and no model parameter
is changed.

Correct/wrong groups are defined by normal greedy free generation, not by the
closed-set final-layer Logit-Lens prediction.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u analyze_coco_logit_lens_stability_v1.py \
  --model qwen-3b \
  --dataset coco_two \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --layers all \
  --mid-start-layer 18 \
  --max-new-tokens 64 \
  --device cuda:0 \
  --attn-impl sdpa \
  --output-dir output/coco_logit_lens_stability/qwen-3b \
  --overwrite

Outputs
-------
  logit_lens_cache.npz
      Dense per-sample layer logits, predictions, margins and metadata.
  sample_summary.csv
      One row per sample with free-generation correctness and trajectory metrics.
  sample_layer_trajectory.csv
      One row per sample x selected layer.
  layerwise_group_summary.csv
      Layer curves for all / generation-correct / generation-wrong samples.
  group_stability_summary.csv
      Correct-vs-wrong aggregate stability statistics.
  trajectory_class_summary.csv
      Descriptive trajectory-pattern counts. These are not causal labels.
  report.txt, config.json, errors.json
  layerwise_*.png, suffix_generation_agreement_hist.png

Important scope
---------------
* Logit Lens measures frozen-LM-head readability, not causal use.
* Stability onset is computed over the selected layers. Use ``--layers all`` for
  an exact all-block suffix statement.
* Relation scores use all one-token surface variants discovered by the repository
  helper, matching prior COCO analyses.
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
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-logit-lens-stability-v1"
RELATIONS: Tuple[str, ...] = ("left", "right", "above", "below")
RELATION_TO_INDEX = {name: i for i, name in enumerate(RELATIONS)}
INDEX_TO_RELATION = np.asarray(RELATIONS, dtype="<U8")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
        help="Repository helper used for dataset/model/prompt utilities.",
    )
    parser.add_argument("--model", default="qwen-3b")
    parser.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated zero-based decoder blocks or 'all'.",
    )
    parser.add_argument(
        "--mid-start-layer",
        type=int,
        default=18,
        help="Start of the middle/late suffix used for stability statistics.",
    )
    parser.add_argument(
        "--stable-early-tolerance",
        type=int,
        default=2,
        help=(
            "A trajectory is called stable-from-mid when its exact suffix onset is "
            "no later than mid_start_layer + this value."
        ),
    )
    parser.add_argument(
        "--min-gt-run",
        type=int,
        default=3,
        help="Minimum consecutive middle/late GT predictions for transient-GT flags.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Upper bound for ordinary greedy free generation; EOS may stop earlier.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2", "none"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_attr_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_final_norm(model: Any, decoder_path: str) -> Tuple[Optional[torch.nn.Module], str]:
    parent = decoder_path.rsplit(".", 1)[0] if "." in decoder_path else ""
    candidates: List[str] = []
    if parent:
        candidates.extend(
            [
                f"{parent}.norm",
                f"{parent}.final_layernorm",
                f"{parent}.ln_f",
            ]
        )
    candidates.extend(
        [
            "model.language_model.norm",
            "model.model.language_model.norm",
            "language_model.model.norm",
            "language_model.norm",
            "model.model.norm",
            "model.norm",
            "model.text_model.norm",
            "text_model.norm",
        ]
    )
    for path in dict.fromkeys(candidates):
        value = get_attr_path(model, path)
        if isinstance(value, torch.nn.Module):
            return value, path
    return None, "identity"


def resolve_output_head(model: Any) -> Tuple[torch.nn.Module, str]:
    try:
        head = model.get_output_embeddings()
    except Exception:
        head = None
    if isinstance(head, torch.nn.Module) and hasattr(head, "weight"):
        return head, "get_output_embeddings()"

    for path in (
        "lm_head",
        "language_model.lm_head",
        "model.lm_head",
        "model.language_model.lm_head",
        "model.model.language_model.lm_head",
    ):
        value = get_attr_path(model, path)
        if isinstance(value, torch.nn.Module) and hasattr(value, "weight"):
            return value, path
    raise RuntimeError("Could not resolve frozen LM output head")


def apply_final_norm(hidden: torch.Tensor, norm: Optional[torch.nn.Module]) -> torch.Tensor:
    if norm is None:
        return hidden
    source_dtype = hidden.dtype
    try:
        out = norm(hidden.unsqueeze(0)).squeeze(0)
    except Exception:
        parameter = next(norm.parameters(), None)
        if parameter is None:
            raise
        out = norm(hidden.to(parameter.dtype).unsqueeze(0)).squeeze(0)
    return out.to(source_dtype)


def selected_relation_logits(
    hidden: torch.Tensor,
    output_head: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    """Project one hidden vector only onto relation-token rows of the frozen LM head."""
    if hidden.ndim != 1:
        raise ValueError(f"Expected [D] hidden state, got {tuple(hidden.shape)}")
    weight = getattr(output_head, "weight")
    bias = getattr(output_head, "bias", None)
    scores: List[torch.Tensor] = []
    for relation in RELATIONS:
        ids = torch.as_tensor(
            relation_token_map[relation],
            device=weight.device,
            dtype=torch.long,
        )
        rows = weight.index_select(0, ids)
        local = torch.mv(rows.float(), hidden.to(weight.device).float())
        if bias is not None:
            local = local + bias.index_select(0, ids).float()
        # Use the strongest one-token surface variant for this relation.
        scores.append(local.max())
    return torch.stack(scores)


def infer_final_state_projection(
    *,
    outputs: Any,
    final_hidden: torch.Tensor,
    prompt_last: int,
    final_norm: Optional[torch.nn.Module],
    output_head: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
) -> Tuple[str, Dict[str, float]]:
    """Detect whether hidden_states[-1] is already final-normalized."""
    output_logits = getattr(outputs, "logits", None)
    if not torch.is_tensor(output_logits):
        return "raw", {"raw_mae": float("nan"), "norm_mae": float("nan")}

    targets: List[torch.Tensor] = []
    for relation in RELATIONS:
        ids = torch.as_tensor(
            relation_token_map[relation],
            device=output_logits.device,
            dtype=torch.long,
        )
        targets.append(output_logits[0, prompt_last].index_select(0, ids).float().max())
    target = torch.stack(targets)

    raw = selected_relation_logits(final_hidden, output_head, relation_token_map).to(target.device)
    raw_mae = float(torch.mean(torch.abs(raw - target)).item())
    if final_norm is None:
        return "raw", {"raw_mae": raw_mae, "norm_mae": float("nan")}

    normed_hidden = apply_final_norm(final_hidden, final_norm)
    normed = selected_relation_logits(normed_hidden, output_head, relation_token_map).to(target.device)
    norm_mae = float(torch.mean(torch.abs(normed - target)).item())
    mode = "raw" if raw_mae <= norm_mae else "norm"
    return mode, {"raw_mae": raw_mae, "norm_mae": norm_mae}


def lens_logits_for_state(
    *,
    hidden: torch.Tensor,
    layer: int,
    n_layers: int,
    final_state_mode: str,
    final_norm: Optional[torch.nn.Module],
    output_head: torch.nn.Module,
    relation_token_map: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    if layer == n_layers - 1 and final_state_mode == "raw":
        projected = hidden
    else:
        projected = apply_final_norm(hidden, final_norm)
    return selected_relation_logits(projected, output_head, relation_token_map)


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), EPS)


def entropy_np(probability: np.ndarray) -> np.ndarray:
    return -np.sum(probability * np.log(np.maximum(probability, EPS)), axis=-1)


def class_margin(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    """logits=[N,L,R], target=[N] or scalar -> target minus best other."""
    scores = np.asarray(logits, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.int64)
    if target_array.ndim == 0:
        target_array = np.full(scores.shape[0], int(target_array), dtype=np.int64)
    sample = np.arange(scores.shape[0])[:, None]
    layer = np.arange(scores.shape[1])[None, :]
    target_score = scores[sample, layer, target_array[:, None]]
    masked = scores.copy()
    masked[sample, layer, target_array[:, None]] = -np.inf
    return target_score - np.max(masked, axis=-1)


def top_margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(logits, dtype=np.float64), axis=-1)
    return ordered[..., -1] - ordered[..., -2]


def first_exact_suffix_onset(
    predictions: np.ndarray,
    layers: Sequence[int],
    target: int,
    start_position: int = 0,
) -> int:
    """Earliest selected layer from which every later prediction equals target."""
    for position in range(max(0, int(start_position)), len(predictions)):
        if np.all(predictions[position:] == int(target)):
            return int(layers[position])
    return -1


def first_occurrence(predictions: np.ndarray, layers: Sequence[int], target: int) -> int:
    hits = np.flatnonzero(predictions == int(target))
    return int(layers[int(hits[0])]) if hits.size else -1


def longest_run(predictions: np.ndarray, target: int) -> int:
    best = 0
    current = 0
    for value in predictions.tolist():
        if int(value) == int(target):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def longest_run_before_layer(
    predictions: np.ndarray,
    layers: Sequence[int],
    target: int,
    before_layer: int,
) -> int:
    selected = [predictions[i] for i, layer in enumerate(layers) if int(layer) < int(before_layer)]
    if not selected:
        return 0
    return longest_run(np.asarray(selected, dtype=np.int64), target)


def nanmean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def nanmedian_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trajectory_class(
    *,
    generation_correct: bool,
    generation_index: int,
    gt_index: int,
    final_lens_index: int,
    suffix_generation_agreement: float,
    suffix_flip_count: int,
    stable_onset_generation_mid: int,
    mid_start_layer: int,
    stable_early_tolerance: int,
    longest_gt_run_mid: int,
    longest_gt_run_before_generation_onset: int,
    min_gt_run: int,
) -> str:
    """Descriptive, non-causal trajectory class."""
    early_limit = int(mid_start_layer) + int(stable_early_tolerance)
    stable_from_mid = (
        stable_onset_generation_mid >= 0
        and stable_onset_generation_mid <= early_limit
    )

    if generation_correct:
        if final_lens_index != generation_index:
            return "final_lens_generation_mismatch_correct"
        if stable_from_mid:
            return "stable_correct_from_mid"
        if suffix_generation_agreement >= 0.80:
            return "mostly_stable_correct"
        return "unstable_or_late_correct"

    if final_lens_index != generation_index:
        return "final_lens_generation_mismatch_wrong"
    if stable_from_mid:
        return "stable_wrong_from_mid"
    if (
        stable_onset_generation_mid >= 0
        and longest_gt_run_before_generation_onset >= int(min_gt_run)
    ):
        return "late_flip_to_wrong"
    if longest_gt_run_mid >= int(min_gt_run):
        return "transient_gt_support_wrong"
    if suffix_generation_agreement < 0.60 and suffix_flip_count >= 2:
        return "unstable_wrong"
    return "mixed_wrong"


def create_plots(
    output_dir: Path,
    layer_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; plots skipped: {exc}", flush=True)
        return

    groups = ("all", "generation_correct", "generation_wrong")
    metrics = (
        ("gt_accuracy", "Layer-wise GT accuracy", "Accuracy", "layerwise_gt_accuracy.png"),
        (
            "generation_agreement",
            "Layer-wise agreement with free generation",
            "Agreement",
            "layerwise_generation_agreement.png",
        ),
        (
            "mean_gt_margin",
            "Layer-wise mean GT margin",
            "GT logit margin",
            "layerwise_mean_gt_margin.png",
        ),
    )
    for metric, title, ylabel, filename in metrics:
        plt.figure(figsize=(8.5, 5.2))
        for group in groups:
            rows = [row for row in layer_rows if row["group"] == group]
            rows.sort(key=lambda row: int(row["layer"]))
            if not rows:
                continue
            plt.plot(
                [int(row["layer"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker="o",
                markersize=3,
                label=group,
            )
        plt.xlabel("Decoder block")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    correct = [
        float(row["suffix_generation_agreement"])
        for row in sample_rows
        if int(row["generation_correct"]) == 1
    ]
    wrong = [
        float(row["suffix_generation_agreement"])
        for row in sample_rows
        if int(row["generation_correct"]) == 0
    ]
    plt.figure(figsize=(8.5, 5.2))
    bins = np.linspace(0.0, 1.0, 21)
    if correct:
        plt.hist(correct, bins=bins, alpha=0.55, label="generation_correct")
    if wrong:
        plt.hist(wrong, bins=bins, alpha=0.55, label="generation_wrong")
    plt.xlabel("Middle/late agreement with free generation")
    plt.ylabel("Samples")
    plt.title("Prompt-last Logit-Lens suffix stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "suffix_generation_agreement_hist.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.min_gt_run < 1:
        raise ValueError("--min-gt-run must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = import_file(Path(args.base_script), "_logit_lens_stability_base")
    data_module = base.import_two_object_module()
    specs = base.merged_model_specs(data_module)
    if args.model not in specs:
        raise ValueError(f"Unknown model {args.model}; available={sorted(specs)}")
    spec = specs[args.model]

    prompt_path = Path(args.prompt_jsonl)
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    prompt_rows = base.load_standard_prompts(prompt_path)
    records, audit = data_module.load_records(
        args.dataset,
        Path(args.data_root),
        args.max_samples,
    )
    records = [record for record in records if int(record.sid) in prompt_rows]
    if not records:
        raise RuntimeError("No records remain after matching the standard prompt file")

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} lacks {spec.model_class}"
        )

    load_kwargs: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(getattr(spec, "trust_remote_code", False)),
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    model = None
    processor = None
    started = time.time()
    errors: List[Dict[str, Any]] = []

    try:
        print(f"Loading {args.model}: {spec.repo_id}", flush=True)
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=bool(getattr(spec, "trust_remote_code", False)),
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)
        selected_layers = base.parse_layers(args.layers, n_layers)
        if (n_layers - 1) not in selected_layers:
            selected_layers = sorted(set(selected_layers + [n_layers - 1]))
            print(
                f"[WARN] Added final decoder block L{n_layers - 1} because final-lens "
                "agreement requires the actual final block.",
                flush=True,
            )
        if selected_layers != list(range(n_layers)):
            print(
                "[WARN] --layers is not all decoder blocks; exact suffix onset is over "
                "selected layers only.",
                flush=True,
            )
        mid_positions = [
            i for i, layer in enumerate(selected_layers)
            if int(layer) >= int(args.mid_start_layer)
        ]
        if not mid_positions:
            raise ValueError(
                f"No selected layer is >= --mid-start-layer {args.mid_start_layer}; "
                f"selected={selected_layers}"
            )
        mid_start_position = int(mid_positions[0])
        effective_mid_start_layer = int(selected_layers[mid_start_position])

        final_norm, final_norm_path = resolve_final_norm(model, decoder_path)
        output_head, output_head_path = resolve_output_head(model)
        relation_token_map = base.relation_token_variants(processor.tokenizer)

        print(
            f"Decoder={decoder_path}; blocks={n_layers}; selected={selected_layers}",
            flush=True,
        )
        print(
            f"FinalNorm={final_norm_path}; LMHead={output_head_path}; "
            f"mid_start=L{effective_mid_start_layer}",
            flush=True,
        )
        print(
            "Relation token variants:\n"
            + json.dumps(
                {
                    relation: [
                        {
                            "id": int(token_id),
                            "decoded": processor.tokenizer.decode([int(token_id)]),
                        }
                        for token_id in ids
                    ]
                    for relation, ids in relation_token_map.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

        sids: List[int] = []
        image_ids: List[str] = []
        subjects: List[str] = []
        references: List[str] = []
        gt_indices: List[int] = []
        generated_indices: List[int] = []
        generated_texts: List[str] = []
        layer_logits_list: List[np.ndarray] = []
        final_state_mode: Optional[str] = None
        final_projection_diagnostics: Optional[Dict[str, float]] = None

        progress = tqdm(records, desc=f"logit-lens-stability:{args.model}")
        completed = 0
        for record in progress:
            sid = int(record.sid)
            batch = None
            outputs = None
            generated_ids = None
            try:
                prompt_row = prompt_rows[sid]
                gt = base.normalize_relation(prompt_row["answer_raw"])
                if gt not in RELATION_TO_INDEX:
                    raise ValueError(f"Unsupported GT relation: {gt!r}")
                subject = str(prompt_row["subject"])
                reference = str(prompt_row["reference"])
                question_text = str(prompt_row["question_text"])
                image = base.record_image(record)

                batch = base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question_text,
                    device=device,
                )
                input_length = int(batch["input_ids"].shape[1])
                prompt_last = input_length - 1

                # Ordinary free generation defines the correct/wrong groups.
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **batch,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                        use_cache=True,
                    )
                generated_text = base.decode_new_tokens(
                    processor,
                    generated_ids,
                    input_length,
                )
                generated_relation = base.normalize_relation(generated_text)
                generated_index = (
                    RELATION_TO_INDEX[generated_relation]
                    if generated_relation in RELATION_TO_INDEX
                    else -1
                )

                # Separate prefill forward pass avoids storing hidden states for every
                # autoregressive decoding step.
                with torch.inference_mode():
                    outputs = model(
                        **batch,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                states = base.hidden_tuple(outputs)
                if len(states) != n_layers + 1:
                    raise RuntimeError(
                        f"Expected {n_layers + 1} hidden-state tensors, got {len(states)}"
                    )
                if int(states[-1].shape[1]) != input_length:
                    raise RuntimeError(
                        f"Token/hidden mismatch: input={input_length}, hidden={states[-1].shape[1]}"
                    )

                if final_state_mode is None:
                    final_state_mode, final_projection_diagnostics = infer_final_state_projection(
                        outputs=outputs,
                        final_hidden=states[-1][0, prompt_last],
                        prompt_last=prompt_last,
                        final_norm=final_norm,
                        output_head=output_head,
                        relation_token_map=relation_token_map,
                    )
                    print(
                        f"Final hidden projection mode={final_state_mode}; "
                        f"diagnostics={final_projection_diagnostics}",
                        flush=True,
                    )

                sample_logits: List[np.ndarray] = []
                for layer in selected_layers:
                    scores = lens_logits_for_state(
                        hidden=states[layer + 1][0, prompt_last],
                        layer=int(layer),
                        n_layers=n_layers,
                        final_state_mode=str(final_state_mode),
                        final_norm=final_norm,
                        output_head=output_head,
                        relation_token_map=relation_token_map,
                    )
                    sample_logits.append(scores.detach().float().cpu().numpy())

                sids.append(sid)
                image_ids.append(str(record.image_id))
                subjects.append(subject)
                references.append(reference)
                gt_indices.append(RELATION_TO_INDEX[gt])
                generated_indices.append(generated_index)
                generated_texts.append(generated_text)
                layer_logits_list.append(np.stack(sample_logits, axis=0).astype(np.float32))

                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    final_lens = RELATIONS[int(np.argmax(sample_logits[-1]))]
                    parsed = generated_relation if generated_relation is not None else "UNPARSED"
                    running_valid = np.asarray(generated_indices, dtype=np.int64) >= 0
                    running_gt = np.asarray(gt_indices, dtype=np.int64)
                    running_gen = np.asarray(generated_indices, dtype=np.int64)
                    running_acc = float(
                        np.mean(running_gen[running_valid] == running_gt[running_valid])
                    ) if np.any(running_valid) else float("nan")
                    tqdm.write(
                        f"done={completed} sid={sid} gt={gt} generation={parsed} "
                        f"final_lens={final_lens} running_generation_acc={running_acc:.4f}"
                    )

            except Exception as exc:
                errors.append(
                    {
                        "sid": sid,
                        "image_id": str(getattr(record, "image_id", "")),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-14:],
                    }
                )
                tqdm.write(f"[ERROR] sid={sid}: {type(exc).__name__}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                del batch, outputs, generated_ids

        if not layer_logits_list:
            raise RuntimeError("No samples completed successfully")
        assert final_state_mode is not None

        logits = np.stack(layer_logits_list, axis=0).astype(np.float32)  # [N,L,4]
        predictions = np.argmax(logits, axis=-1).astype(np.int64)
        probability = softmax_np(logits)
        entropy = entropy_np(probability)
        top_margins = top_margin(logits)
        gt_array = np.asarray(gt_indices, dtype=np.int64)
        generation_array = np.asarray(generated_indices, dtype=np.int64)
        parse_ok = generation_array >= 0
        final_lens_array = predictions[:, -1]
        gt_margins = class_margin(logits, gt_array)

        generation_margins = np.full((len(gt_array), len(selected_layers)), np.nan, dtype=np.float64)
        valid_indices = np.flatnonzero(parse_ok)
        if valid_indices.size:
            generation_margins[valid_indices] = class_margin(
                logits[valid_indices],
                generation_array[valid_indices],
            )

        sample_rows: List[Dict[str, Any]] = []
        trajectory_rows: List[Dict[str, Any]] = []
        for i in range(len(gt_array)):
            gt_index = int(gt_array[i])
            generation_index = int(generation_array[i])
            final_lens_index = int(final_lens_array[i])
            pred = predictions[i]
            suffix_pred = pred[mid_start_position:]
            suffix_gt_margin = gt_margins[i, mid_start_position:]
            suffix_generation_margin = generation_margins[i, mid_start_position:]

            generation_correct = bool(parse_ok[i] and generation_index == gt_index)
            suffix_flip_count = int(np.sum(suffix_pred[1:] != suffix_pred[:-1]))
            all_flip_count = int(np.sum(pred[1:] != pred[:-1]))
            longest_gt_run_mid = longest_run(suffix_pred, gt_index)

            if parse_ok[i]:
                suffix_generation_agreement = float(np.mean(suffix_pred == generation_index))
                stable_onset_generation_all = first_exact_suffix_onset(
                    pred, selected_layers, generation_index, 0
                )
                stable_onset_generation_mid = first_exact_suffix_onset(
                    pred, selected_layers, generation_index, mid_start_position
                )
                longest_generation_run_mid = longest_run(suffix_pred, generation_index)
                longest_gt_before_generation_onset = (
                    longest_run_before_layer(
                        pred,
                        selected_layers,
                        gt_index,
                        stable_onset_generation_mid,
                    )
                    if stable_onset_generation_mid >= 0
                    else longest_gt_run_mid
                )
                sample_class = trajectory_class(
                    generation_correct=generation_correct,
                    generation_index=generation_index,
                    gt_index=gt_index,
                    final_lens_index=final_lens_index,
                    suffix_generation_agreement=suffix_generation_agreement,
                    suffix_flip_count=suffix_flip_count,
                    stable_onset_generation_mid=stable_onset_generation_mid,
                    mid_start_layer=effective_mid_start_layer,
                    stable_early_tolerance=int(args.stable_early_tolerance),
                    longest_gt_run_mid=longest_gt_run_mid,
                    longest_gt_run_before_generation_onset=longest_gt_before_generation_onset,
                    min_gt_run=int(args.min_gt_run),
                )
            else:
                suffix_generation_agreement = float("nan")
                stable_onset_generation_all = -1
                stable_onset_generation_mid = -1
                longest_generation_run_mid = 0
                longest_gt_before_generation_onset = longest_gt_run_mid
                sample_class = "unparsed_generation"

            stable_onset_gt_mid = first_exact_suffix_onset(
                pred, selected_layers, gt_index, mid_start_position
            )
            stable_onset_final_lens_mid = first_exact_suffix_onset(
                pred, selected_layers, final_lens_index, mid_start_position
            )
            suffix_gt_agreement = float(np.mean(suffix_pred == gt_index))
            suffix_final_lens_agreement = float(np.mean(suffix_pred == final_lens_index))

            sample_rows.append(
                {
                    "sid": int(sids[i]),
                    "image_id": image_ids[i],
                    "subject": subjects[i],
                    "reference": references[i],
                    "gt": RELATIONS[gt_index],
                    "generated_relation": (
                        RELATIONS[generation_index] if generation_index >= 0 else ""
                    ),
                    "generated_text": generated_texts[i],
                    "parse_ok": int(parse_ok[i]),
                    "generation_correct": int(generation_correct),
                    "final_lens_relation": RELATIONS[final_lens_index],
                    "final_lens_correct": int(final_lens_index == gt_index),
                    "final_lens_agrees_generation": int(
                        parse_ok[i] and final_lens_index == generation_index
                    ),
                    "trajectory_class": sample_class,
                    "mid_start_layer": effective_mid_start_layer,
                    "suffix_layer_count": len(suffix_pred),
                    "suffix_generation_agreement": suffix_generation_agreement,
                    "suffix_gt_agreement": suffix_gt_agreement,
                    "suffix_final_lens_agreement": suffix_final_lens_agreement,
                    "all_flip_count": all_flip_count,
                    "suffix_flip_count": suffix_flip_count,
                    "first_gt_layer": first_occurrence(pred, selected_layers, gt_index),
                    "first_generation_layer": (
                        first_occurrence(pred, selected_layers, generation_index)
                        if parse_ok[i]
                        else -1
                    ),
                    "stable_onset_generation_all": stable_onset_generation_all,
                    "stable_onset_generation_mid": stable_onset_generation_mid,
                    "stable_onset_gt_mid": stable_onset_gt_mid,
                    "stable_onset_final_lens_mid": stable_onset_final_lens_mid,
                    "longest_gt_run_mid": longest_gt_run_mid,
                    "longest_generation_run_mid": longest_generation_run_mid,
                    "longest_gt_run_before_generation_onset": longest_gt_before_generation_onset,
                    "suffix_mean_gt_margin": nanmean_or_nan(suffix_gt_margin),
                    "suffix_min_gt_margin": (
                        float(np.min(suffix_gt_margin)) if suffix_gt_margin.size else float("nan")
                    ),
                    "suffix_positive_gt_margin_rate": float(np.mean(suffix_gt_margin > 0)),
                    "suffix_mean_generation_margin": nanmean_or_nan(suffix_generation_margin),
                    "suffix_positive_generation_margin_rate": (
                        float(np.mean(suffix_generation_margin > 0))
                        if parse_ok[i]
                        else float("nan")
                    ),
                    "suffix_mean_top_margin": float(
                        np.mean(top_margins[i, mid_start_position:])
                    ),
                    "suffix_mean_entropy": float(
                        np.mean(entropy[i, mid_start_position:])
                    ),
                }
            )

            for local_layer_index, layer in enumerate(selected_layers):
                layer_prediction = int(pred[local_layer_index])
                trajectory_rows.append(
                    {
                        "sid": int(sids[i]),
                        "image_id": image_ids[i],
                        "layer": int(layer),
                        "gt": RELATIONS[gt_index],
                        "generated_relation": (
                            RELATIONS[generation_index] if generation_index >= 0 else ""
                        ),
                        "generation_correct": int(generation_correct),
                        "lens_prediction": RELATIONS[layer_prediction],
                        "lens_correct_gt": int(layer_prediction == gt_index),
                        "lens_agrees_generation": int(
                            parse_ok[i] and layer_prediction == generation_index
                        ),
                        "lens_agrees_final_lens": int(layer_prediction == final_lens_index),
                        "gt_margin": float(gt_margins[i, local_layer_index]),
                        "generation_margin": float(
                            generation_margins[i, local_layer_index]
                        ),
                        "top_margin": float(top_margins[i, local_layer_index]),
                        "entropy": float(entropy[i, local_layer_index]),
                        "logit_left": float(logits[i, local_layer_index, 0]),
                        "logit_right": float(logits[i, local_layer_index, 1]),
                        "logit_above": float(logits[i, local_layer_index, 2]),
                        "logit_below": float(logits[i, local_layer_index, 3]),
                    }
                )

        layer_rows: List[Dict[str, Any]] = []
        group_masks = {
            "all": parse_ok,
            "generation_correct": parse_ok & (generation_array == gt_array),
            "generation_wrong": parse_ok & (generation_array != gt_array),
        }
        for group_name, mask in group_masks.items():
            indices = np.flatnonzero(mask)
            for layer_position, layer in enumerate(selected_layers):
                if indices.size == 0:
                    continue
                layer_prediction = predictions[indices, layer_position]
                previous_flip_rate = (
                    float(
                        np.mean(
                            predictions[indices, layer_position]
                            != predictions[indices, layer_position - 1]
                        )
                    )
                    if layer_position > 0
                    else float("nan")
                )
                layer_rows.append(
                    {
                        "group": group_name,
                        "layer": int(layer),
                        "N": int(indices.size),
                        "gt_accuracy": float(
                            np.mean(layer_prediction == gt_array[indices])
                        ),
                        "generation_agreement": float(
                            np.mean(layer_prediction == generation_array[indices])
                        ),
                        "final_lens_agreement": float(
                            np.mean(layer_prediction == final_lens_array[indices])
                        ),
                        "positive_gt_margin_rate": float(
                            np.mean(gt_margins[indices, layer_position] > 0)
                        ),
                        "positive_generation_margin_rate": float(
                            np.mean(generation_margins[indices, layer_position] > 0)
                        ),
                        "mean_gt_margin": float(
                            np.mean(gt_margins[indices, layer_position])
                        ),
                        "mean_generation_margin": float(
                            np.mean(generation_margins[indices, layer_position])
                        ),
                        "mean_top_margin": float(
                            np.mean(top_margins[indices, layer_position])
                        ),
                        "mean_entropy": float(
                            np.mean(entropy[indices, layer_position])
                        ),
                        "prediction_flip_rate_from_previous": previous_flip_rate,
                    }
                )

        group_rows: List[Dict[str, Any]] = []
        for group_name, desired_correctness in (
            ("generation_correct", 1),
            ("generation_wrong", 0),
        ):
            rows = [
                row
                for row in sample_rows
                if int(row["parse_ok"]) == 1
                and int(row["generation_correct"]) == desired_correctness
            ]
            if not rows:
                continue
            onset_values = np.asarray(
                [
                    float(row["stable_onset_generation_mid"])
                    if int(row["stable_onset_generation_mid"]) >= 0
                    else np.nan
                    for row in rows
                ],
                dtype=np.float64,
            )
            early_limit = effective_mid_start_layer + int(args.stable_early_tolerance)
            group_rows.append(
                {
                    "group": group_name,
                    "N": len(rows),
                    "mean_suffix_generation_agreement": float(
                        np.mean([float(row["suffix_generation_agreement"]) for row in rows])
                    ),
                    "median_suffix_generation_agreement": float(
                        np.median([float(row["suffix_generation_agreement"]) for row in rows])
                    ),
                    "mean_suffix_gt_agreement": float(
                        np.mean([float(row["suffix_gt_agreement"]) for row in rows])
                    ),
                    "mean_suffix_flip_count": float(
                        np.mean([float(row["suffix_flip_count"]) for row in rows])
                    ),
                    "median_stable_onset_generation_mid": nanmedian_or_nan(onset_values),
                    "stable_onset_exists_rate": float(np.mean(np.isfinite(onset_values))),
                    "stable_from_mid_rate": float(
                        np.mean(
                            [
                                int(row["stable_onset_generation_mid"]) >= 0
                                and int(row["stable_onset_generation_mid"]) <= early_limit
                                for row in rows
                            ]
                        )
                    ),
                    "final_lens_agrees_generation_rate": float(
                        np.mean([int(row["final_lens_agrees_generation"]) for row in rows])
                    ),
                    "mean_suffix_gt_margin": float(
                        np.mean([float(row["suffix_mean_gt_margin"]) for row in rows])
                    ),
                    "mean_suffix_generation_margin": float(
                        np.mean(
                            [float(row["suffix_mean_generation_margin"]) for row in rows]
                        )
                    ),
                    "mean_suffix_top_margin": float(
                        np.mean([float(row["suffix_mean_top_margin"]) for row in rows])
                    ),
                    "mean_suffix_entropy": float(
                        np.mean([float(row["suffix_mean_entropy"]) for row in rows])
                    ),
                }
            )

        class_counts = Counter(str(row["trajectory_class"]) for row in sample_rows)
        parsed_denominator = max(int(np.sum(parse_ok)), 1)
        class_rows = []
        for name, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0])):
            class_rows.append(
                {
                    "trajectory_class": name,
                    "N": int(count),
                    "fraction_all_samples": float(count / max(len(sample_rows), 1)),
                    "fraction_parsed": (
                        float(count / parsed_denominator)
                        if name != "unparsed_generation"
                        else float("nan")
                    ),
                }
            )

        # Save dense cache before text summaries.
        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "prompt_jsonl": str(prompt_path),
            "decoder_path": decoder_path,
            "decoder_blocks": n_layers,
            "selected_layers": selected_layers,
            "mid_start_layer_requested": int(args.mid_start_layer),
            "mid_start_layer_effective": effective_mid_start_layer,
            "final_norm_path": final_norm_path,
            "output_head_path": output_head_path,
            "final_state_mode": final_state_mode,
            "final_projection_diagnostics": final_projection_diagnostics,
            "relation_token_map": relation_token_map,
            "transformers_version": transformers.__version__,
            "max_new_tokens": int(args.max_new_tokens),
            "n_records_requested": len(records),
            "n_completed": len(gt_array),
            "n_errors": len(errors),
            "audit": audit,
            "generation_defines_correct_wrong": True,
            "test_uses_gt_for_lens_prediction": False,
            "model_modified": False,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        np.savez_compressed(
            output_dir / "logit_lens_cache.npz",
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object),
            sid=np.asarray(sids, dtype=np.int64),
            image_id=np.asarray(image_ids, dtype=object),
            subject=np.asarray(subjects, dtype=object),
            reference=np.asarray(references, dtype=object),
            gt_index=gt_array,
            generated_index=generation_array,
            generated_text=np.asarray(generated_texts, dtype=object),
            selected_layers=np.asarray(selected_layers, dtype=np.int32),
            logits=logits,
            probability=probability.astype(np.float32),
            prediction=predictions,
            gt_margin=gt_margins.astype(np.float32),
            generation_margin=generation_margins.astype(np.float32),
            top_margin=top_margins.astype(np.float32),
            entropy=entropy.astype(np.float32),
        )

        write_csv(output_dir / "sample_summary.csv", sample_rows)
        write_csv(output_dir / "sample_layer_trajectory.csv", trajectory_rows)
        write_csv(output_dir / "layerwise_group_summary.csv", layer_rows)
        write_csv(output_dir / "group_stability_summary.csv", group_rows)
        write_csv(output_dir / "trajectory_class_summary.csv", class_rows)
        (output_dir / "errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        create_plots(output_dir, layer_rows, sample_rows)

        parsed_count = int(np.sum(parse_ok))
        correct_count = int(np.sum(parse_ok & (generation_array == gt_array)))
        wrong_count = int(np.sum(parse_ok & (generation_array != gt_array)))
        generation_accuracy = correct_count / max(parsed_count, 1)
        final_lens_accuracy = float(np.mean(final_lens_array == gt_array))
        final_lens_generation_agreement = float(
            np.mean(final_lens_array[parse_ok] == generation_array[parse_ok])
        ) if parsed_count else float("nan")

        group_by_name = {str(row["group"]): row for row in group_rows}
        key_layers = [
            layer
            for layer in (18, 19, 21, 23, 24, 26, 28, 30, 32, n_layers - 1)
            if layer in selected_layers
        ]
        key_layer_lines: List[str] = []
        for layer in list(dict.fromkeys(key_layers)):
            correct_row = next(
                (
                    row
                    for row in layer_rows
                    if row["group"] == "generation_correct" and int(row["layer"]) == layer
                ),
                None,
            )
            wrong_row = next(
                (
                    row
                    for row in layer_rows
                    if row["group"] == "generation_wrong" and int(row["layer"]) == layer
                ),
                None,
            )
            if correct_row is None or wrong_row is None:
                continue
            key_layer_lines.append(
                f"L{layer:<2d} | correct: GT/final={float(correct_row['gt_accuracy']):.4f} "
                f"flip={float(correct_row['prediction_flip_rate_from_previous']):.4f} | "
                f"wrong: GT={float(wrong_row['gt_accuracy']):.4f} "
                f"generated={float(wrong_row['generation_agreement']):.4f} "
                f"flip={float(wrong_row['prediction_flip_rate_from_previous']):.4f}"
            )

        report_lines = [
            "=" * 118,
            "PROMPT-LAST FROZEN LOGIT-LENS STABILITY",
            f"model={args.model} | samples={len(gt_array)} | parsed={parsed_count} | "
            f"correct={correct_count} | wrong={wrong_count}",
            f"free-generation parse rate={parsed_count / len(gt_array):.4f} | "
            f"accuracy among parsed={generation_accuracy:.4f}",
            f"final-layer closed-set lens accuracy={final_lens_accuracy:.4f} | "
            f"final-lens/free-generation agreement={final_lens_generation_agreement:.4f}",
            f"selected layers={selected_layers[0]}..{selected_layers[-1]} ({len(selected_layers)}) | "
            f"middle/late suffix starts at L{effective_mid_start_layer}",
            "=" * 118,
            "",
            "CORRECT VS WRONG SUFFIX STABILITY",
            f"{'Group':<24}{'N':>7}{'GenAgree':>12}{'GTAgree':>12}{'Flips':>10}"
            f"{'StableOnset':>15}{'StableMid':>12}{'Lens=Gen':>11}",
            "-" * 118,
        ]
        for name in ("generation_correct", "generation_wrong"):
            row = group_by_name.get(name)
            if row is None:
                continue
            report_lines.append(
                f"{name:<24}{int(row['N']):>7}"
                f"{float(row['mean_suffix_generation_agreement']):>12.4f}"
                f"{float(row['mean_suffix_gt_agreement']):>12.4f}"
                f"{float(row['mean_suffix_flip_count']):>10.4f}"
                f"{float(row['median_stable_onset_generation_mid']):>15.2f}"
                f"{float(row['stable_from_mid_rate']):>12.4f}"
                f"{float(row['final_lens_agrees_generation_rate']):>11.4f}"
            )

        report_lines.extend(
            [
                "",
                "KEY LAYER SNAPSHOT",
                *key_layer_lines,
                "",
                "DESCRIPTIVE TRAJECTORY CLASSES",
            ]
        )
        for row in class_rows:
            report_lines.append(
                f"{str(row['trajectory_class']):<46} N={int(row['N']):>4} "
                f"fraction_all={float(row['fraction_all_samples']):.4f} "
                f"fraction_parsed={float(row['fraction_parsed']):.4f}"
            )
        report_lines.extend(
            [
                "",
                "Interpretation limits:",
                "- A stable lens trajectory means the frozen final LM head reads the same relation from those states.",
                "- It does not prove that every later block causally uses that relation direction.",
                "- late_flip_to_wrong and related classes are descriptive candidates; causal patching is required for attribution.",
                "",
                "Saved:",
                f"  {output_dir / 'logit_lens_cache.npz'}",
                f"  {output_dir / 'sample_summary.csv'}",
                f"  {output_dir / 'sample_layer_trajectory.csv'}",
                f"  {output_dir / 'layerwise_group_summary.csv'}",
                f"  {output_dir / 'group_stability_summary.csv'}",
                f"  {output_dir / 'trajectory_class_summary.csv'}",
                f"  {output_dir / 'report.txt'}",
            ]
        )
        report = "\n".join(report_lines) + "\n"
        (output_dir / "report.txt").write_text(report, encoding="utf-8")
        print("\n" + report, flush=True)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
