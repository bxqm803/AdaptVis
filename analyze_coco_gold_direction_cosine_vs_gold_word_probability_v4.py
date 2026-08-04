#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two exact same-GT test.

For each sample with gold relation g:
    x = cos(delta_c, u_g)
    y = P(gold relation word | prefix at the located answer position)

Crucially, y is always the probability of the GOLD relation word, even when
normal greedy generation outputs another relation. This differs from v3, which
stored the probability of the actually generated answer word.

The script re-runs the original greedy generation only to locate the answer
position and recover the exact output formatting (leading separator and case).
It then teacher-forces the gold relation word at that same prefix and computes
its token-level and whole-word probability.

Internal relations above/below are displayed as on/under in the output, while
the canonical scored words default to above/below because those are the COCO
four-direction prompt surfaces.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
DISPLAY_RELATION = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
DEFAULT_GOLD_SURFACE = {
    "left": "left",
    "right": "right",
    "above": "above",
    "below": "below",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default="output/coco_centroid_generation_multimodel_v4_rerun",
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
    p.add_argument(
        "--v3-script",
        default="analyze_coco_similarity_vs_generation_answer_probability_v3.py",
        help="Used only for shared model/loading/token-location helpers.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--search-words",
        default="left,right,on,under,above,below",
    )
    p.add_argument(
        "--vertical-gold-surfaces",
        choices=("above-below", "on-under"),
        default="above-below",
        help=(
            "Words teacher-forced for internal above/below gold labels. "
            "Use above-below for the original four-direction prompts."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="output/coco_gold_direction_cosine_vs_gold_word_probability_v4",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--flush-every", type=int, default=25)
    p.add_argument("--quiet-samples", action="store_true")
    return p.parse_args()


def import_module(path: Path, name: str) -> Any:
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


def apply_case_style(word: str, observed_surface: str) -> str:
    observed = str(observed_surface)
    if observed.isupper():
        return word.upper()
    if observed[:1].isupper() and observed[1:].islower():
        return word.capitalize()
    return word.lower()


def answer_leading_separator(
    span_text: str,
    observed_surface: str,
) -> str:
    """Return text inside the answer token span before the regex word."""
    haystack = str(span_text)
    needle = str(observed_surface)
    index = haystack.lower().find(needle.lower())
    if index < 0:
        return ""
    return haystack[:index]


def encode_without_specials(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def extend_sequence_tensor(
    value: torch.Tensor,
    append_length: int,
    *,
    fill_value: int,
) -> torch.Tensor:
    if append_length <= 0:
        return value
    if value.ndim != 2 or value.shape[0] != 1:
        return value
    extension = torch.full(
        (1, append_length),
        fill_value=fill_value,
        dtype=value.dtype,
        device=value.device,
    )
    return torch.cat([value, extension], dim=1)


def teacher_forced_candidate_probability(
    *,
    model: Any,
    batch: Mapping[str, Any],
    generated_prefix_ids: Sequence[int],
    candidate_ids: Sequence[int],
) -> Dict[str, Any]:
    """Score one candidate continuation at the exact answer-start prefix."""
    candidate = [int(x) for x in candidate_ids]
    if not candidate:
        raise ValueError("Candidate tokenization is empty")

    prompt_ids = batch["input_ids"]
    prefix_tensor = torch.as_tensor(
        list(map(int, generated_prefix_ids)),
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    ).unsqueeze(0)
    candidate_tensor = torch.as_tensor(
        candidate,
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    ).unsqueeze(0)

    full_input_ids = torch.cat(
        [prompt_ids, prefix_tensor, candidate_tensor],
        dim=1,
    )
    prefix_length = int(prompt_ids.shape[1] + prefix_tensor.shape[1])
    append_length = int(prefix_tensor.shape[1] + candidate_tensor.shape[1])

    forward_batch: Dict[str, Any] = dict(batch)
    forward_batch["input_ids"] = full_input_ids

    if "attention_mask" in forward_batch:
        forward_batch["attention_mask"] = extend_sequence_tensor(
            forward_batch["attention_mask"],
            append_length,
            fill_value=1,
        )

    # Let each model recompute sequence-derived values for the extended input.
    for key in (
        "position_ids",
        "cache_position",
        "labels",
        "past_key_values",
    ):
        forward_batch.pop(key, None)

    with torch.inference_mode():
        outputs = model(
            **forward_batch,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits[0].float()
    token_logprobs: List[float] = []
    token_probs: List[float] = []

    for offset, token_id in enumerate(candidate):
        # Causal LM logits at position k predict token k+1.
        prediction_position = prefix_length + offset - 1
        if prediction_position < 0 or prediction_position >= logits.shape[0]:
            raise IndexError(
                f"Invalid prediction position={prediction_position}, "
                f"logits_length={logits.shape[0]}"
            )
        log_probs = torch.log_softmax(logits[prediction_position], dim=-1)
        logprob = float(log_probs[token_id].item())
        token_logprobs.append(logprob)
        token_probs.append(float(math.exp(logprob)))

    word_logprob = float(np.sum(token_logprobs))
    mean_logprob = float(np.mean(token_logprobs))
    return {
        "gold_candidate_token_ids": candidate,
        "gold_candidate_token_probabilities": token_probs,
        "gold_candidate_token_logprobs": token_logprobs,
        "gold_candidate_first_token_probability": float(token_probs[0]),
        "gold_candidate_word_logprob": word_logprob,
        "gold_candidate_word_mean_logprob": mean_logprob,
        "gold_candidate_word_probability": (
            float(math.exp(word_logprob)) if word_logprob > -745 else 0.0
        ),
        "gold_candidate_word_geometric_mean_probability": float(
            math.exp(mean_logprob)
        ),
        "gold_candidate_token_count": int(len(candidate)),
    }


def correlation_stats(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(frame)
    if n < 3 or frame["x"].std(ddof=0) <= 1e-12 or frame["y"].std(ddof=0) <= 1e-12:
        return {
            "N": int(n),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    if pearsonr is None or spearmanr is None:
        return {
            "N": int(n),
            "pearson_r": float(frame["x"].corr(frame["y"], method="pearson")),
            "pearson_p": np.nan,
            "spearman_r": float(frame["x"].corr(frame["y"], method="spearman")),
            "spearman_p": np.nan,
        }
    pr = pearsonr(frame["x"].to_numpy(), frame["y"].to_numpy())
    sr = spearmanr(frame["x"].to_numpy(), frame["y"].to_numpy())
    return {
        "N": int(n),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step1_path = Path(args.step1_script)
    v3_path = Path(args.v3_script)
    if not step1_path.exists():
        raise FileNotFoundError(step1_path)
    if not v3_path.exists():
        raise FileNotFoundError(v3_path)

    step1 = import_module(step1_path, "coco_centroid_step1_v4_for_gold_prob")
    v3 = import_module(v3_path, "coco_generation_probability_v3_helpers")

    report = pd.read_csv(root / "reports" / "model_comparison.csv")
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    report = report[report["model"].isin(models)].copy()

    search_words = [x.strip() for x in args.search_words.split(",") if x.strip()]
    answer_pattern = v3.compile_answer_pattern(search_words)

    gold_surface = dict(DEFAULT_GOLD_SURFACE)
    if args.vertical_gold_surfaces == "on-under":
        gold_surface["above"] = "on"
        gold_surface["below"] = "under"

    two_object = step1.import_two_object_module()
    records, _audit = two_object.load_records("coco_two", Path(args.data_root), None)
    record_by_sid = {int(record.sid): record for record in records}
    prompt_rows = step1.load_standard_prompts(Path(args.prompt_jsonl))
    specs = step1.merged_model_specs(two_object)

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for model_name in models:
        model_report = report.loc[report["model"] == model_name]
        if model_report.empty:
            raise RuntimeError(f"Missing report row for {model_name}")
        best_layer = int(model_report.iloc[-1]["similarity_centroid_best_layer"])

        model_dir = root / "step1" / model_name
        saved_samples = read_jsonl(model_dir / "samples.jsonl")
        if args.max_samples is not None:
            saved_samples = saved_samples[: int(args.max_samples)]

        model, processor = v3.load_model_and_processor(
            step1,
            specs[model_name],
            args.device,
        )

        print("\n" + "=" * 150)
        print(f"MODEL={model_name} | L{best_layer} | N={len(saved_samples)}")
        print("=" * 150)

        for sample_index, saved in enumerate(
            tqdm(saved_samples, desc=model_name, dynamic_ncols=True)
        ):
            sid = int(saved["sid"])
            gt = step1.normalize_relation(saved.get("gt"))
            if gt not in INTERNAL_TO_INDEX:
                raise RuntimeError(f"Invalid GT sid={sid}: {saved.get('gt')}")

            array_path = model_dir / "sample_arrays" / f"{sid}.npz"
            with np.load(array_path, allow_pickle=False) as data:
                layer_indices = np.asarray(data["layer_indices"], dtype=np.int64)
                positions = np.where(layer_indices == best_layer)[0]
                if len(positions) != 1:
                    raise RuntimeError(f"sid={sid}: L{best_layer} not unique")
                pos = int(positions[0])
                original = np.asarray(
                    data["original_similarity_centroids"][pos],
                    dtype=np.float64,
                )
                swapped_role = np.asarray(
                    data["swapped_similarity_centroids_role_order"][pos],
                    dtype=np.float64,
                )
                centroids = 0.5 * (original + swapped_role[[1, 0], :])

            sim = v3.relation_scores_from_centroids(centroids)
            gt_index = INTERNAL_TO_INDEX[gt]
            gt_projection = float(sim["scores"][gt_index])
            gt_cosine = (
                gt_projection / float(sim["distance"])
                if float(sim["distance"]) > 1e-12
                else np.nan
            )
            gt_cosine = float(np.clip(gt_cosine, -1.0, 1.0))

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
                )

            generated_ids = [
                int(x)
                for x in generated.sequences[0, input_length:].detach().cpu().tolist()
            ]
            generated_text = v3.safe_decode(
                processor.tokenizer,
                generated_ids,
                skip_special_tokens=True,
            )
            located = v3.locate_last_answer_token(
                processor.tokenizer,
                generated_ids,
                answer_pattern,
            )

            if located is None:
                skipped.append({
                    "model": model_name,
                    "sid": sid,
                    "gt": gt,
                    "reason": "answer_position_not_found",
                    "generation_text": generated_text,
                })
                del generated, batch, image
                continue

            token_start = int(located["answer_token_start_0based"])
            generated_prefix_ids = generated_ids[:token_start]
            separator = answer_leading_separator(
                str(located["answer_token_span_text"]),
                str(located["answer_surface"]),
            )
            styled_surface = apply_case_style(
                gold_surface[gt],
                str(located["answer_surface"]),
            )
            candidate_text = separator + styled_surface
            candidate_ids = encode_without_specials(
                processor.tokenizer,
                candidate_text,
            )

            scored = teacher_forced_candidate_probability(
                model=model,
                batch=batch,
                generated_prefix_ids=generated_prefix_ids,
                candidate_ids=candidate_ids,
            )

            generation_prediction = str(located["generation_prediction"])
            generation_correct = bool(generation_prediction == gt)

            # Validation: when the generated relation equals GT and the chosen
            # canonical spelling/case matches the actual answer span, teacher-
            # forced candidate IDs should equal the generated answer IDs.
            actual_answer_ids = [int(x) for x in located["answer_token_ids"]]
            candidate_matches_actual_ids = bool(candidate_ids == actual_answer_ids)

            row: Dict[str, Any] = {
                "model": model_name,
                "sid": sid,
                "gt_internal": gt,
                "gold_relation": DISPLAY_RELATION[gt],
                "best_similarity_layer": best_layer,
                "gold_direction_cosine": gt_cosine,
                "sim_distance": float(sim["distance"]),
                "sim_prediction": str(sim["prediction"]),
                "generation_text": generated_text,
                "generation_prediction_internal": generation_prediction,
                "generation_prediction": DISPLAY_RELATION[generation_prediction],
                "generation_correct": generation_correct,
                "answer_token_start_1based": int(located["answer_token_start_1based"]),
                "answer_surface": str(located["answer_surface"]),
                "answer_separator": separator,
                "gold_candidate_surface": styled_surface,
                "gold_candidate_text": candidate_text,
                "candidate_matches_actual_answer_ids": candidate_matches_actual_ids,
                "gold_candidate_token_ids_json": json_compact(
                    scored["gold_candidate_token_ids"]
                ),
                "gold_candidate_token_probabilities_json": json_compact(
                    scored["gold_candidate_token_probabilities"]
                ),
                "gold_candidate_token_logprobs_json": json_compact(
                    scored["gold_candidate_token_logprobs"]
                ),
                **{
                    key: value
                    for key, value in scored.items()
                    if key not in {
                        "gold_candidate_token_ids",
                        "gold_candidate_token_probabilities",
                        "gold_candidate_token_logprobs",
                    }
                },
            }
            rows.append(row)

            if not args.quiet_samples:
                print(
                    f"[{model_name}] sid={sid:4d} gold={DISPLAY_RELATION[gt]:<5} "
                    f"cos={gt_cosine:+.4f} "
                    f"P_gold={scored['gold_candidate_word_probability']:.6g} "
                    f"geom={scored['gold_candidate_word_geometric_mean_probability']:.6g} "
                    f"generated={DISPLAY_RELATION[generation_prediction]:<5} "
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

    sample_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    sample_df.to_csv(output_dir / "sample_gold_probability.csv", index=False)
    skipped_df.to_csv(output_dir / "skipped_samples.csv", index=False)

    correlation_rows: List[Dict[str, Any]] = []
    for (model_name, gold), frame in sample_df.groupby(
        ["model", "gold_relation"],
        sort=False,
    ):
        subsets = {
            "all_same_gold": frame,
            "generation_correct": frame[frame["generation_correct"] == True],  # noqa: E712
            "generation_wrong": frame[frame["generation_correct"] == False],  # noqa: E712
        }
        for subset_name, subset in subsets.items():
            for y_name in (
                "gold_candidate_word_probability",
                "gold_candidate_word_geometric_mean_probability",
                "gold_candidate_word_logprob",
                "gold_candidate_word_mean_logprob",
            ):
                stats = correlation_stats(
                    subset["gold_direction_cosine"],
                    subset[y_name],
                )
                correlation_rows.append({
                    "model": model_name,
                    "gold_relation": gold,
                    "subset": subset_name,
                    "x": "gold_direction_cosine",
                    "y": y_name,
                    "mean_gold_direction_cosine": float(
                        subset["gold_direction_cosine"].mean()
                    ) if len(subset) else np.nan,
                    "mean_gold_probability": float(
                        subset["gold_candidate_word_probability"].mean()
                    ) if len(subset) else np.nan,
                    **stats,
                })

    correlation_df = pd.DataFrame(correlation_rows)
    correlation_df.to_csv(output_dir / "correlation_by_same_gold.csv", index=False)

    # Within each fixed gold, inspect monotonicity by cosine quantile bins.
    bin_rows: List[Dict[str, Any]] = []
    for (model_name, gold), frame in sample_df.groupby(
        ["model", "gold_relation"],
        sort=False,
    ):
        frame = frame.copy()
        if len(frame) < 10:
            continue
        try:
            frame["cosine_bin"] = pd.qcut(
                frame["gold_direction_cosine"],
                q=5,
                duplicates="drop",
            )
        except ValueError:
            continue
        for cosine_bin, part in frame.groupby(
            "cosine_bin",
            observed=True,
            sort=True,
        ):
            bin_rows.append({
                "model": model_name,
                "gold_relation": gold,
                "cosine_bin": str(cosine_bin),
                "N": int(len(part)),
                "mean_gold_direction_cosine": float(
                    part["gold_direction_cosine"].mean()
                ),
                "mean_gold_word_probability": float(
                    part["gold_candidate_word_probability"].mean()
                ),
                "mean_gold_word_geometric_probability": float(
                    part["gold_candidate_word_geometric_mean_probability"].mean()
                ),
                "generation_accuracy": float(part["generation_correct"].mean()),
            })

    bins_df = pd.DataFrame(bin_rows)
    bins_df.to_csv(output_dir / "same_gold_cosine_probability_bins.csv", index=False)

    print("\n" + "=" * 150)
    print("SAME-GOLD CORRELATION")
    print("X = cosine(delta_c, gold direction)")
    print("Y = teacher-forced probability of that same gold relation word")
    print("=" * 150)

    main_view = correlation_df[
        (correlation_df["subset"] == "all_same_gold")
        & (
            correlation_df["y"]
            == "gold_candidate_word_geometric_mean_probability"
        )
    ].copy()
    print(
        main_view[
            [
                "model",
                "gold_relation",
                "N",
                "mean_gold_direction_cosine",
                "mean_gold_probability",
                "pearson_r",
                "pearson_p",
                "spearman_r",
                "spearman_p",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSaved:")
    print(output_dir / "sample_gold_probability.csv")
    print(output_dir / "correlation_by_same_gold.csv")
    print(output_dir / "same_gold_cosine_probability_bins.csv")
    print(output_dir / "skipped_samples.csv")


if __name__ == "__main__":
    main()
