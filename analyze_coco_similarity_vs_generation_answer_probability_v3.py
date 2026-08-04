#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two: compare best-layer hidden-similarity strength with the probability of
THE ACTUAL ANSWER WORD in a complete greedy generation.

This script deliberately does NOT perform a second left/right/on/under
classification at the answer step.

For each sample it:
  1. Re-runs the same complete greedy generation used by the centroid pipeline.
  2. Finds the LAST whole-word occurrence of
         left / right / on / under / above / below
     case-insensitively in the complete generated continuation.
  3. Maps that text occurrence back to its generated-token span.
  4. Reads the probability of every ACTUALLY GENERATED token in that span from
     generated.scores[step].
  5. Reports the answer token position (1-based), token pieces, per-token
     probabilities, joint word probability, and mean log probability.
  6. Computes the same best-Sim@L geometric signal and correlates its strength
     with the actual generated answer probability.

Generation accuracy is defined only from the located final answer word versus
GT. There is no separate "token-step accuracy" and no candidate-word rescoring.

Main outputs:
  sample_metrics.csv
  model_summary.csv
  correlation_summary.csv
  missing_answer_samples.csv
  plots/*.png
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None


INTERNAL_RELATIONS = ("left", "right", "above", "below")
INTERNAL_TO_INDEX = {name: i for i, name in enumerate(INTERNAL_RELATIONS)}

SURFACE_TO_INTERNAL = {
    "left": "left",
    "right": "right",
    "on": "above",
    "above": "above",
    "under": "below",
    "below": "below",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default="output/coco_centroid_generation_multimodel_v4_rerun",
        help="Root of the reproduced centroid-generation run.",
    )
    p.add_argument(
        "--models",
        default="qwen2-2b,qwen-3b,qwen-7b,llava-7b,llava-13b",
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--step1-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--similarity-temperature", type=float, default=0.07)
    p.add_argument(
        "--search-words",
        default="left,right,on,under,above,below",
        help=(
            "Whole words searched in the complete generation. The LAST match "
            "is treated as the generated answer."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="output/coco_similarity_vs_generation_answer_probability_v3",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--flush-every", type=int, default=25)
    p.add_argument(
        "--quiet-samples",
        action="store_true",
        help="Suppress one-line per-sample answer-token printing.",
    )
    return p.parse_args()


def import_module_from_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_decode(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool,
) -> str:
    kwargs = {
        "skip_special_tokens": skip_special_tokens,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(map(int, token_ids)), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces", None)
        return str(tokenizer.decode(list(map(int, token_ids)), **kwargs))


def compile_answer_pattern(search_words: Sequence[str]) -> re.Pattern[str]:
    clean: List[str] = []
    for raw in search_words:
        word = str(raw).strip().lower()
        if word not in SURFACE_TO_INTERNAL:
            raise ValueError(
                f"Unsupported answer word {raw!r}; supported="
                f"{sorted(SURFACE_TO_INTERNAL)}"
            )
        clean.append(word)
    clean = list(dict.fromkeys(clean))
    if not clean:
        raise ValueError("--search-words resolved to an empty list")

    # Long words first; alphabetic boundaries prevent matching 'on' in words
    # such as 'condition'.
    body = "|".join(re.escape(x) for x in sorted(clean, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])({body})(?![A-Za-z])", re.IGNORECASE)


def locate_last_answer_token(
    tokenizer: Any,
    generated_ids: Sequence[int],
    pattern: re.Pattern[str],
) -> Optional[Dict[str, Any]]:
    """Locate the final answer-word occurrence and map it to token indices."""
    token_ids = [int(x) for x in generated_ids]

    # Prefix decoding is used instead of assuming that one text word equals one
    # tokenizer token. It also handles leading-space tokens and word pieces.
    decoded_prefixes = [""]
    for end in range(1, len(token_ids) + 1):
        decoded_prefixes.append(
            safe_decode(
                tokenizer,
                token_ids[:end],
                skip_special_tokens=True,
            )
        )

    final_text = decoded_prefixes[-1]
    matches = list(pattern.finditer(final_text))
    if not matches:
        return None

    match = matches[-1]
    char_start = int(match.start())
    char_end = int(match.end())

    token_start: Optional[int] = None
    for index in range(len(token_ids)):
        if len(decoded_prefixes[index + 1]) > char_start:
            token_start = index
            break
    if token_start is None:
        return None

    token_end: Optional[int] = None
    for index in range(token_start, len(token_ids)):
        if len(decoded_prefixes[index + 1]) >= char_end:
            token_end = index
            break
    if token_end is None:
        return None

    surface = str(match.group(1))
    surface_lower = surface.lower()
    internal_relation = SURFACE_TO_INTERNAL[surface_lower]

    pieces = [
        safe_decode(tokenizer, [token_id], skip_special_tokens=False)
        for token_id in token_ids[token_start : token_end + 1]
    ]

    return {
        "generated_text": final_text,
        "answer_surface": surface,
        "answer_surface_lower": surface_lower,
        "generation_prediction": internal_relation,
        "answer_char_start": char_start,
        "answer_char_end": char_end,
        "answer_token_start_0based": int(token_start),
        "answer_token_end_0based": int(token_end),
        "answer_token_start_1based": int(token_start + 1),
        "answer_token_end_1based": int(token_end + 1),
        "answer_token_span_length": int(token_end - token_start + 1),
        "answer_is_single_token": bool(token_start == token_end),
        "answer_token_ids": token_ids[token_start : token_end + 1],
        "answer_token_pieces": pieces,
        "answer_token_span_text": safe_decode(
            tokenizer,
            token_ids[token_start : token_end + 1],
            skip_special_tokens=False,
        ),
        "generated_token_count": int(len(token_ids)),
        "prefix_text_before_answer": final_text[:char_start],
    }


def actual_answer_span_probability(
    *,
    score_steps: Sequence[torch.Tensor],
    generated_ids: Sequence[int],
    token_start: int,
    token_end: int,
    tokenizer: Any,
) -> Dict[str, Any]:
    """Read probabilities of the actual generated token(s) in the answer span."""
    token_rows: List[Dict[str, Any]] = []
    logprobs: List[float] = []
    probs: List[float] = []

    for step in range(int(token_start), int(token_end) + 1):
        if not (0 <= step < len(score_steps)):
            raise IndexError(
                f"Answer token step {step} outside generated.scores={len(score_steps)}"
            )
        token_id = int(generated_ids[step])
        scores = score_steps[step][0].float()
        log_probs = torch.log_softmax(scores, dim=-1)
        logprob = float(log_probs[token_id].item())
        probability = float(math.exp(logprob))

        # Greedy output should normally be rank 1 in the processed score vector.
        # Save rank for validation without redefining any accuracy metric.
        token_score = scores[token_id]
        rank = int(torch.sum(scores > token_score).item()) + 1
        argmax_id = int(torch.argmax(scores).item())

        piece = safe_decode(
            tokenizer,
            [token_id],
            skip_special_tokens=False,
        )
        token_rows.append({
            "position_0based": int(step),
            "position_1based": int(step + 1),
            "token_id": token_id,
            "token_piece": piece,
            "probability": probability,
            "logprob": logprob,
            "rank": rank,
            "is_step_argmax": bool(token_id == argmax_id),
            "argmax_token_id": argmax_id,
            "argmax_token_piece": safe_decode(
                tokenizer,
                [argmax_id],
                skip_special_tokens=False,
            ),
        })
        probs.append(probability)
        logprobs.append(logprob)

    word_logprob = float(np.sum(logprobs))
    word_mean_logprob = float(np.mean(logprobs))
    # exp(-745) is near the smallest positive float64 value.
    word_probability = float(math.exp(word_logprob)) if word_logprob > -745 else 0.0
    word_geometric_mean_probability = float(math.exp(word_mean_logprob))

    return {
        "answer_token_details": token_rows,
        "answer_token_probabilities": probs,
        "answer_token_logprobs": logprobs,
        "answer_first_token_probability": float(probs[0]),
        "answer_last_token_probability": float(probs[-1]),
        "answer_min_token_probability": float(np.min(probs)),
        "answer_mean_token_probability": float(np.mean(probs)),
        "answer_word_logprob": word_logprob,
        "answer_word_mean_logprob": word_mean_logprob,
        "answer_word_probability": word_probability,
        "answer_word_geometric_mean_probability": word_geometric_mean_probability,
        "answer_all_tokens_are_step_argmax": bool(
            all(row["is_step_argmax"] for row in token_rows)
        ),
        "answer_max_token_rank": int(max(row["rank"] for row in token_rows)),
    }


def relation_scores_from_centroids(centroids: np.ndarray) -> Dict[str, Any]:
    delta = np.asarray(centroids[0] - centroids[1], dtype=np.float64)
    dx, dy = float(delta[0]), float(delta[1])
    scores = np.asarray([-dx, dx, -dy, dy], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    return {
        "dx": dx,
        "dy": dy,
        "scores": scores,
        "prediction": INTERNAL_RELATIONS[int(order[0])],
        "top1_margin": float(scores[order[0]] - scores[order[1]]),
        "distance": float(np.linalg.norm(delta)),
        "axis_confidence": float(
            abs(abs(dx) - abs(dy)) / (abs(dx) + abs(dy) + 1e-12)
        ),
    }


def finite_pair(x: Iterable[Any], y: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def correlation_stats(x: Iterable[Any], y: Iterable[Any]) -> Dict[str, Any]:
    xa, ya = finite_pair(x, y)
    if len(xa) < 3 or np.std(xa) <= 0 or np.std(ya) <= 0:
        return {
            "n": int(len(xa)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    if pearsonr is not None and spearmanr is not None:
        pr = pearsonr(xa, ya)
        sr = spearmanr(xa, ya)
        return {
            "n": int(len(xa)),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
        }
    return {
        "n": int(len(xa)),
        "pearson_r": float(np.corrcoef(xa, ya)[0, 1]),
        "pearson_p": np.nan,
        "spearman_r": float(pd.Series(xa).corr(pd.Series(ya), method="spearman")),
        "spearman_p": np.nan,
    }


def load_model_and_processor(step1: Any, spec: Any, device: str) -> Tuple[Any, Any]:
    model_cls = getattr(step1.transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers has no model class {spec.model_class}")

    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=step1.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)

    processor = step1.AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    step1.configure_processor(model, processor)
    return model, processor


def extract_raw_similarity_metrics(
    *,
    step1: Any,
    generated: Any,
    batch: Mapping[str, Any],
    processor: Any,
    model: Any,
    prompt: Mapping[str, Any],
    best_layer: int,
    temperature: float,
) -> Dict[str, float]:
    """Raw object-to-visual cosine statistics at the selected Sim@L layer."""
    input_ids_list = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = step1.locate_object_spans(
        processor.tokenizer,
        input_ids_list,
        str(prompt["subject"]),
        str(prompt["reference"]),
    )
    visual_indices = step1.resolve_visual_indices(
        model,
        processor,
        batch,
        input_ids_list,
    )

    hidden_steps = step1.generation_steps(generated, "hidden_states")
    if not hidden_steps:
        raise RuntimeError("generate() returned no hidden-state steps")
    prompt_hidden_layers = step1.step_layers(hidden_steps[0])
    hidden = prompt_hidden_layers[best_layer + 1][0].float()

    visual_index_tensor = torch.as_tensor(
        visual_indices,
        dtype=torch.long,
        device=hidden.device,
    )
    visual_hidden = hidden.index_select(0, visual_index_tensor)
    object_hidden = torch.stack([
        hidden[int(subject_span[1])],
        hidden[int(reference_span[1])],
    ])

    raw_cosine = torch.matmul(
        torch.nn.functional.normalize(object_hidden, dim=-1),
        torch.nn.functional.normalize(visual_hidden, dim=-1).T,
    )
    raw_peak = raw_cosine.max(dim=-1).values
    raw_mean = raw_cosine.mean(dim=-1)
    raw_top2 = torch.topk(
        raw_cosine,
        k=min(2, int(raw_cosine.shape[-1])),
        dim=-1,
    ).values
    if raw_top2.shape[-1] >= 2:
        raw_peak_gap = raw_top2[:, 0] - raw_top2[:, 1]
    else:
        raw_peak_gap = torch.zeros_like(raw_top2[:, 0])

    weights = torch.softmax(raw_cosine / float(temperature), dim=-1)
    entropy_conf = step1.entropy_confidence(weights)
    map_separation = float(
        (0.5 * torch.sum(torch.abs(weights[0] - weights[1]))).item()
    )

    return {
        "raw_sim_peak_subject": float(raw_peak[0].item()),
        "raw_sim_peak_reference": float(raw_peak[1].item()),
        "raw_sim_peak_mean": float(raw_peak.mean().item()),
        "raw_sim_peak_min": float(raw_peak.min().item()),
        "raw_sim_mean_subject": float(raw_mean[0].item()),
        "raw_sim_mean_reference": float(raw_mean[1].item()),
        "raw_sim_peak_gap_mean": float(raw_peak_gap.mean().item()),
        "raw_sim_entropy_confidence_mean": float(entropy_conf.mean().item()),
        "raw_sim_object_map_separation": map_separation,
    }


def json_dump_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    step1_path = Path(args.step1_script)
    if not step1_path.exists():
        raise FileNotFoundError(step1_path)
    step1 = import_module_from_path(step1_path, "coco_centroid_step1_v4")

    report_path = root / "reports" / "model_comparison.csv"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = pd.read_csv(report_path)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    report = report[report["model"].isin(models)].copy()
    missing_models = sorted(set(models) - set(report["model"]))
    if missing_models:
        raise RuntimeError(f"Missing models in report: {missing_models}")

    search_words = [x.strip() for x in args.search_words.split(",") if x.strip()]
    answer_pattern = compile_answer_pattern(search_words)

    two_object = step1.import_two_object_module()
    records, _audit = two_object.load_records(
        "coco_two",
        Path(args.data_root),
        None,
    )
    record_by_sid = {int(record.sid): record for record in records}
    prompt_rows = step1.load_standard_prompts(Path(args.prompt_jsonl))
    specs = step1.merged_model_specs(two_object)

    sample_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for model_name in models:
        if model_name not in specs:
            raise RuntimeError(
                f"Unknown model {model_name}; available={sorted(specs)}"
            )

        model_report = report.loc[report["model"] == model_name].iloc[-1]
        best_layer = int(model_report["similarity_centroid_best_layer"])

        model_dir = root / "step1" / model_name
        samples_path = model_dir / "samples.jsonl"
        arrays_dir = model_dir / "sample_arrays"
        if not samples_path.exists():
            raise FileNotFoundError(samples_path)

        saved_samples = read_jsonl(samples_path)
        if args.max_samples is not None:
            saved_samples = saved_samples[: int(args.max_samples)]

        print("\n" + "=" * 170)
        print(
            f"MODEL={model_name} | best Sim layer=L{best_layer} | "
            f"N={len(saved_samples)} | max_new_tokens={args.max_new_tokens}"
        )
        print("=" * 170)

        model, processor = load_model_and_processor(
            step1,
            specs[model_name],
            args.device,
        )

        for sample_index, saved in enumerate(
            tqdm(saved_samples, desc=model_name, dynamic_ncols=True)
        ):
            sid = int(saved["sid"])
            gt = step1.normalize_relation(saved.get("gt"))
            if gt not in INTERNAL_TO_INDEX:
                raise RuntimeError(f"Invalid GT sid={sid}: {saved.get('gt')}")
            if sid not in record_by_sid or sid not in prompt_rows:
                raise RuntimeError(f"Missing record/prompt for sid={sid}")

            array_path = arrays_dir / f"{sid}.npz"
            if not array_path.exists():
                raise FileNotFoundError(array_path)

            with np.load(array_path, allow_pickle=False) as data:
                layer_indices = np.asarray(data["layer_indices"], dtype=np.int64)
                positions = np.where(layer_indices == best_layer)[0]
                if len(positions) != 1:
                    raise RuntimeError(
                        f"sid={sid}: L{best_layer} absent/duplicated in {array_path}"
                    )
                layer_pos = int(positions[0])
                original_centroids = np.asarray(
                    data["original_similarity_centroids"][layer_pos],
                    dtype=np.float64,
                )
                swapped_role = np.asarray(
                    data["swapped_similarity_centroids_role_order"][layer_pos],
                    dtype=np.float64,
                )
                swapped_aligned = swapped_role[[1, 0], :]
                average_centroids = 0.5 * (original_centroids + swapped_aligned)
                original_sep = float(
                    data["original_similarity_separation"][layer_pos]
                )
                swapped_sep = float(
                    data["swapped_similarity_separation"][layer_pos]
                )

            sim = relation_scores_from_centroids(average_centroids)
            sim_scores = sim["scores"]
            gt_index = INTERNAL_TO_INDEX[gt]
            sim_gt_margin = float(
                sim_scores[gt_index] - np.max(np.delete(sim_scores, gt_index))
            )

            prompt = prompt_rows[sid]
            image = step1.record_image(record_by_sid[sid])
            batch = step1.make_question_batch(
                processor=processor,
                image=image,
                question_text=str(prompt["question_text"]),
                device=torch.device(args.device),
            )
            input_length = int(batch["input_ids"].shape[1])

            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )

            generated_ids = [
                int(x)
                for x in generated.sequences[0, input_length:].detach().cpu().tolist()
            ]
            score_steps = list(generated.scores)
            if len(score_steps) != len(generated_ids):
                common = min(len(score_steps), len(generated_ids))
                generated_ids = generated_ids[:common]
                score_steps = score_steps[:common]

            generated_text = safe_decode(
                processor.tokenizer,
                generated_ids,
                skip_special_tokens=True,
            )
            located = locate_last_answer_token(
                processor.tokenizer,
                generated_ids,
                answer_pattern,
            )

            raw_similarity = extract_raw_similarity_metrics(
                step1=step1,
                generated=generated,
                batch=batch,
                processor=processor,
                model=model,
                prompt=prompt,
                best_layer=best_layer,
                temperature=float(args.similarity_temperature),
            )

            saved_prediction = step1.normalize_relation(saved.get("original_prediction"))
            base_row: Dict[str, Any] = {
                "model": model_name,
                "sid": sid,
                "gt": gt,
                "question": str(saved.get("question", prompt.get("question_text", ""))),
                "saved_generation_text": str(saved.get("original_generated_text", "")),
                "rerun_generation_text": generated_text,
                "rerun_matches_saved_generation": bool(
                    generated_text.strip()
                    == str(saved.get("original_generated_text", "")).strip()
                ),
                "saved_generation_prediction": saved_prediction,
                "saved_generation_correct": bool(saved.get("original_correct", False)),
                "generated_token_count": int(len(generated_ids)),
                "best_similarity_layer": best_layer,
                "sim_prediction": sim["prediction"],
                "sim_correct": bool(sim["prediction"] == gt),
                "sim_dx": sim["dx"],
                "sim_dy": sim["dy"],
                "sim_distance": sim["distance"],
                "sim_axis_confidence": sim["axis_confidence"],
                "sim_top1_margin": sim["top1_margin"],
                "sim_gt_margin": sim_gt_margin,
                "sim_gt_score": float(sim_scores[gt_index]),
                "sim_predicted_score": float(
                    sim_scores[INTERNAL_TO_INDEX[sim["prediction"]]]
                ),
                "sim_map_separation": float(0.5 * (original_sep + swapped_sep)),
                **raw_similarity,
            }
            for rel_index, relation in enumerate(INTERNAL_RELATIONS):
                base_row[f"sim_score_{relation}"] = float(sim_scores[rel_index])

            if located is None:
                row = {
                    **base_row,
                    "answer_found": False,
                    "generation_prediction": None,
                    "generation_correct": False,
                }
                sample_rows.append(row)
                missing_rows.append({
                    **row,
                    "reason": "no_configured_answer_word_in_complete_generation",
                    "search_words": ",".join(search_words),
                })
                if not args.quiet_samples:
                    print(
                        f"[{model_name}] sid={sid:4d} | ANSWER NOT FOUND | "
                        f"tokens={len(generated_ids)} | text={generated_text!r}",
                        flush=True,
                    )
            else:
                answer_probability = actual_answer_span_probability(
                    score_steps=score_steps,
                    generated_ids=generated_ids,
                    token_start=int(located["answer_token_start_0based"]),
                    token_end=int(located["answer_token_end_0based"]),
                    tokenizer=processor.tokenizer,
                )
                generation_prediction = str(located["generation_prediction"])
                generation_correct = bool(generation_prediction == gt)

                token_details = answer_probability["answer_token_details"]
                row = {
                    **base_row,
                    **located,
                    "answer_found": True,
                    "generation_prediction": generation_prediction,
                    "generation_correct": generation_correct,
                    "saved_vs_located_prediction_disagree": bool(
                        saved_prediction != generation_prediction
                    ),
                    "answer_token_ids_json": json_dump_compact(
                        located["answer_token_ids"]
                    ),
                    "answer_token_pieces_json": json_dump_compact(
                        located["answer_token_pieces"]
                    ),
                    "answer_token_probabilities_json": json_dump_compact(
                        answer_probability["answer_token_probabilities"]
                    ),
                    "answer_token_logprobs_json": json_dump_compact(
                        answer_probability["answer_token_logprobs"]
                    ),
                    "answer_token_details_json": json_dump_compact(token_details),
                    **{
                        key: value
                        for key, value in answer_probability.items()
                        if key not in {
                            "answer_token_details",
                            "answer_token_probabilities",
                            "answer_token_logprobs",
                        }
                    },
                }
                sample_rows.append(row)

                if not args.quiet_samples:
                    if located["answer_is_single_token"]:
                        token_span = str(located["answer_token_start_1based"])
                    else:
                        token_span = (
                            f"{located['answer_token_start_1based']}-"
                            f"{located['answer_token_end_1based']}"
                        )
                    token_prob_text = ", ".join(
                        f"#{item['position_1based']} {item['token_piece']!r} "
                        f"p={item['probability']:.6g}"
                        for item in token_details
                    )
                    print(
                        f"[{model_name}] sid={sid:4d} | "
                        f"answer={located['answer_surface']!r} "
                        f"token={token_span}/{located['generated_token_count']} | "
                        f"{token_prob_text} | "
                        f"word_p={answer_probability['answer_word_probability']:.6g} "
                        f"geom_p={answer_probability['answer_word_geometric_mean_probability']:.6g} | "
                        f"pred={generation_prediction} gt={gt} "
                        f"correct={generation_correct}",
                        flush=True,
                    )

            del generated, batch, image
            if (sample_index + 1) % max(1, int(args.flush_every)) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_df = pd.DataFrame(sample_rows)
    missing_df = pd.DataFrame(missing_rows)
    sample_df.to_csv(output_dir / "sample_metrics.csv", index=False)
    missing_df.to_csv(output_dir / "missing_answer_samples.csv", index=False)

    found_df = sample_df[sample_df["answer_found"] == True].copy()  # noqa: E712

    correlation_pairs = [
        ("raw_sim_peak_mean", "answer_first_token_probability"),
        ("raw_sim_peak_mean", "answer_word_probability"),
        ("raw_sim_peak_mean", "answer_word_geometric_mean_probability"),
        ("raw_sim_peak_mean", "answer_word_mean_logprob"),
        ("raw_sim_peak_min", "answer_word_mean_logprob"),
        ("raw_sim_peak_gap_mean", "answer_word_mean_logprob"),
        ("raw_sim_entropy_confidence_mean", "answer_word_mean_logprob"),
        ("raw_sim_object_map_separation", "answer_word_mean_logprob"),
        ("sim_top1_margin", "answer_word_mean_logprob"),
        ("sim_gt_margin", "answer_word_mean_logprob"),
        ("sim_distance", "answer_word_mean_logprob"),
        ("sim_axis_confidence", "answer_word_mean_logprob"),
        ("sim_map_separation", "answer_word_mean_logprob"),
    ]

    correlation_rows: List[Dict[str, Any]] = []
    for model_name, frame in found_df.groupby("model"):
        subsets = {
            "all_found": frame,
            "generation_correct": frame[frame["generation_correct"] == True],  # noqa: E712
            "generation_wrong": frame[frame["generation_correct"] == False],  # noqa: E712
            "single_token": frame[frame["answer_is_single_token"] == True],  # noqa: E712
        }
        for subset_name, subset in subsets.items():
            for x_name, y_name in correlation_pairs:
                stats = correlation_stats(subset[x_name], subset[y_name])
                correlation_rows.append({
                    "model": model_name,
                    "subset": subset_name,
                    "x": x_name,
                    "y": y_name,
                    **stats,
                })

    correlation_df = pd.DataFrame(correlation_rows)
    correlation_df.to_csv(output_dir / "correlation_summary.csv", index=False)

    model_summary_rows: List[Dict[str, Any]] = []
    for model_name in models:
        all_frame = sample_df[sample_df["model"] == model_name].copy()
        frame = found_df[found_df["model"] == model_name].copy()
        correct_frame = frame[frame["generation_correct"] == True]  # noqa: E712
        wrong_frame = frame[frame["generation_correct"] == False]  # noqa: E712

        model_summary_rows.append({
            "model": model_name,
            "N_total": int(len(all_frame)),
            "N_answer_found": int(len(frame)),
            "answer_found_rate": float(len(frame) / max(1, len(all_frame))),
            "best_similarity_layer": int(all_frame["best_similarity_layer"].iloc[0]),
            "rerun_matches_saved_generation_rate": float(
                all_frame["rerun_matches_saved_generation"].mean()
            ),
            "saved_generation_accuracy": float(
                all_frame["saved_generation_correct"].mean()
            ),
            # Missing answers are counted as wrong here.
            "located_generation_accuracy_all": float(
                all_frame["generation_correct"].fillna(False).astype(bool).mean()
            ),
            "located_generation_accuracy_found": (
                float(frame["generation_correct"].mean()) if len(frame) else np.nan
            ),
            "saved_vs_located_prediction_disagreement_rate_found": (
                float(frame["saved_vs_located_prediction_disagree"].mean())
                if len(frame)
                else np.nan
            ),
            "similarity_accuracy": float(all_frame["sim_correct"].mean()),
            "answer_single_token_rate": (
                float(frame["answer_is_single_token"].mean()) if len(frame) else np.nan
            ),
            "mean_answer_token_start_position": (
                float(frame["answer_token_start_1based"].mean())
                if len(frame)
                else np.nan
            ),
            "mean_answer_token_span_length": (
                float(frame["answer_token_span_length"].mean())
                if len(frame)
                else np.nan
            ),
            "all_answer_tokens_step_argmax_rate": (
                float(frame["answer_all_tokens_are_step_argmax"].mean())
                if len(frame)
                else np.nan
            ),
            "mean_answer_first_token_probability": (
                float(frame["answer_first_token_probability"].mean())
                if len(frame)
                else np.nan
            ),
            "mean_answer_word_probability": (
                float(frame["answer_word_probability"].mean())
                if len(frame)
                else np.nan
            ),
            "mean_answer_word_geometric_mean_probability": (
                float(frame["answer_word_geometric_mean_probability"].mean())
                if len(frame)
                else np.nan
            ),
            "mean_answer_word_mean_logprob": (
                float(frame["answer_word_mean_logprob"].mean())
                if len(frame)
                else np.nan
            ),
            "mean_word_probability_generation_correct": (
                float(correct_frame["answer_word_probability"].mean())
                if len(correct_frame)
                else np.nan
            ),
            "mean_word_probability_generation_wrong": (
                float(wrong_frame["answer_word_probability"].mean())
                if len(wrong_frame)
                else np.nan
            ),
            "mean_word_geom_probability_generation_correct": (
                float(
                    correct_frame["answer_word_geometric_mean_probability"].mean()
                )
                if len(correct_frame)
                else np.nan
            ),
            "mean_word_geom_probability_generation_wrong": (
                float(
                    wrong_frame["answer_word_geometric_mean_probability"].mean()
                )
                if len(wrong_frame)
                else np.nan
            ),
        })

    model_summary = pd.DataFrame(model_summary_rows)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        plot_pairs = [
            (
                "raw_sim_peak_mean",
                "answer_word_geometric_mean_probability",
                "Best-layer raw similarity peak mean",
                "Actual answer-word geometric mean probability",
                "raw_similarity_vs_actual_answer_probability",
            ),
            (
                "sim_gt_margin",
                "answer_word_mean_logprob",
                "Similarity GT geometric margin",
                "Actual answer-word mean log probability",
                "sim_gt_margin_vs_actual_answer_logprob",
            ),
            (
                "sim_top1_margin",
                "answer_word_mean_logprob",
                "Similarity top-1 margin",
                "Actual answer-word mean log probability",
                "sim_top1_margin_vs_actual_answer_logprob",
            ),
        ]
        for model_name, frame in found_df.groupby("model"):
            for x_name, y_name, x_label, y_label, suffix in plot_pairs:
                fig, ax = plt.subplots(figsize=(6.5, 5.0))
                ax.scatter(frame[x_name], frame[y_name], alpha=0.55, s=18)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(model_name)
                fig.tight_layout()
                fig.savefig(plots_dir / f"{model_name}_{suffix}.png", dpi=180)
                plt.close(fig)
    except Exception as exc:
        print(f"[warning] plotting skipped: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 180)
    print("MODEL SUMMARY")
    print("=" * 180)
    print(
        model_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 180)
    print("KEY CORRELATIONS")
    print("=" * 180)
    key = correlation_df[
        (
            (correlation_df["x"] == "raw_sim_peak_mean")
            & (
                correlation_df["y"]
                == "answer_word_geometric_mean_probability"
            )
        )
        |
        (
            (correlation_df["x"] == "sim_gt_margin")
            & (correlation_df["y"] == "answer_word_mean_logprob")
        )
    ]
    print(
        key.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
