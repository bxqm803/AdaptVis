#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO-two: relate the best-layer hidden-similarity signal to the probability of
THE ANSWER TOKEN INSIDE A COMPLETE GREEDY GENERATION.

For each sample:
  1. Run the normal complete generation (default max_new_tokens=8).
  2. Decode the whole generated continuation.
  3. Find the LAST whole-word occurrence of one of the configured answer words
     (default: left, right, on, under), case-insensitively.
  4. Map that character occurrence back to its generated-token position by
     decoding every generated-token prefix.
  5. Use generated.scores[token_position] to obtain the vocabulary logits that
     produced that answer token.
  6. At exactly that generation step, compare left/right/on/under probabilities,
     including lowercase, Capitalized and UPPERCASE one-token variants with
     common leading whitespace/newline and punctuation forms.
  7. Correlate those answer-step probabilities/margins with the best-Sim@L
     similarity magnitude and geometric relation margins.

Important:
  * generated.scores[t] is the distribution used to choose generated token t+1.
  * The main answer parser follows the user's requested LAST-occurrence rule.
  * COCO canonical above/below labels are mapped to on/under for answer-token
    scoring. Optional search aliases above/below can be enabled with
    --search-words left,right,on,under,above,below.
  * Best Sim@L layers are read from the already reproduced model_comparison.csv;
    those layer choices remain diagnostic full-set oracle choices.

Main outputs:
  sample_metrics.csv
  model_summary.csv
  token_case_accuracy.csv
  correlation_summary.csv
  token_variant_inventory.csv
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


# Internal geometric labels used by the original COCO centroid pipeline.
INTERNAL_RELATIONS = ("left", "right", "above", "below")
INTERNAL_TO_INDEX = {name: i for i, name in enumerate(INTERNAL_RELATIONS)}

# Surface answer categories requested by the user.
ANSWER_RELATIONS = ("left", "right", "on", "under")
ANSWER_TO_INDEX = {name: i for i, name in enumerate(ANSWER_RELATIONS)}

INTERNAL_TO_ANSWER = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
ANSWER_TO_INTERNAL = {
    "left": "left",
    "right": "right",
    "on": "above",
    "under": "below",
    "above": "above",
    "below": "below",
}

CASE_MODES = ("lower", "capitalized", "upper", "all")
# Common token-boundary forms. Suffixes handle tokenizers that merge punctuation.
PREFIXES = ("", " ", "\n", "\n\n", "\t")
SUFFIXES = ("", ".", ",", ":", ";", "!", "?", ")", "]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default="output/coco_centroid_generation_multimodel_v4_rerun",
        help="Root of the reproduced centroid rerun.",
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
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument(
        "--search-words",
        default="left,right,on,under",
        help=(
            "Whole words searched in the full generation. The LAST occurrence "
            "is the answer. Use left,right,on,under,above,below to include aliases."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="output/coco_similarity_vs_answer_token_probability_v2",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--flush-every", type=int, default=25)
    p.add_argument(
        "--quiet-samples",
        action="store_true",
        help="Do not print one answer-token line per sample.",
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


def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def cased_words(word: str, mode: str) -> List[str]:
    if mode == "lower":
        values = [word.lower()]
    elif mode == "capitalized":
        values = [word.capitalize()]
    elif mode == "upper":
        values = [word.upper()]
    elif mode == "all":
        values = [word.lower(), word.capitalize(), word.upper()]
    else:
        raise ValueError(mode)
    return list(dict.fromkeys(values))


def build_variant_bank(
    tokenizer: Any,
    vocab_size: int,
) -> Tuple[Dict[str, Dict[str, List[int]]], List[Dict[str, Any]]]:
    """Build one-token left/right/on/under banks for each case mode."""
    bank: Dict[str, Dict[str, List[int]]] = {}
    inventory: List[Dict[str, Any]] = []
    unk = getattr(tokenizer, "unk_token_id", None)

    for case_mode in CASE_MODES:
        relation_ids: Dict[str, List[int]] = {}
        for relation in ANSWER_RELATIONS:
            ids: List[int] = []
            for cased in cased_words(relation, case_mode):
                for prefix in PREFIXES:
                    for suffix in SUFFIXES:
                        surface = f"{prefix}{cased}{suffix}"
                        encoded = tokenizer_ids(tokenizer, surface)
                        valid = len(encoded) == 1
                        token_id: Optional[int] = None
                        decoded: Optional[str] = None
                        if valid:
                            candidate = int(encoded[0])
                            if (
                                0 <= candidate < vocab_size
                                and (unk is None or candidate != int(unk))
                            ):
                                token_id = candidate
                                ids.append(candidate)
                                decoded = safe_decode(
                                    tokenizer,
                                    [candidate],
                                    skip_special_tokens=False,
                                )
                            else:
                                valid = False
                        inventory.append({
                            "case_mode": case_mode,
                            "relation": relation,
                            "surface_repr": repr(surface),
                            "one_token": bool(valid and token_id is not None),
                            "token_id": token_id,
                            "decoded_token": decoded,
                            "encoded_ids": " ".join(map(str, encoded)),
                        })
            relation_ids[relation] = list(dict.fromkeys(ids))
            if case_mode == "all" and not relation_ids[relation]:
                raise RuntimeError(
                    f"No valid one-token variants for all-case/{relation}"
                )
        bank[case_mode] = relation_ids
    return bank, inventory


def score_answer_relations(
    logits: torch.Tensor,
    ids_by_relation: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    """Score answer categories at one exact autoregressive generation step."""
    missing_relations = [
        relation for relation in ANSWER_RELATIONS
        if not list(ids_by_relation.get(relation, ()))
    ]
    if missing_relations:
        return {
            "valid": False,
            "missing_relations": missing_relations,
        }

    logits = logits.float()
    full_probs = torch.softmax(logits, dim=-1)

    max_logits: List[torch.Tensor] = []
    max_probs: List[torch.Tensor] = []
    sum_probs: List[torch.Tensor] = []
    max_ids: List[int] = []

    for relation in ANSWER_RELATIONS:
        ids = torch.as_tensor(
            list(ids_by_relation[relation]),
            dtype=torch.long,
            device=logits.device,
        )
        rel_logits = logits.index_select(0, ids)
        rel_probs = full_probs.index_select(0, ids)
        best = int(torch.argmax(rel_logits).item())
        max_logits.append(rel_logits[best])
        max_probs.append(rel_probs[best])
        sum_probs.append(rel_probs.sum())
        max_ids.append(int(ids[best].item()))

    max_logits_t = torch.stack(max_logits)
    max_probs_t = torch.stack(max_probs)
    sum_probs_t = torch.stack(sum_probs)
    conditional_max_t = torch.softmax(max_logits_t, dim=0)
    conditional_sum_t = sum_probs_t / sum_probs_t.sum().clamp_min(1e-30)

    order = torch.argsort(max_logits_t, descending=True)
    pred_max_i = int(order[0].item())
    pred_sum_i = int(torch.argmax(sum_probs_t).item())

    return {
        "valid": True,
        "missing_relations": [],
        "max_logits": max_logits_t.detach().cpu().numpy(),
        "max_probs": max_probs_t.detach().cpu().numpy(),
        "sum_probs": sum_probs_t.detach().cpu().numpy(),
        "conditional_max": conditional_max_t.detach().cpu().numpy(),
        "conditional_sum": conditional_sum_t.detach().cpu().numpy(),
        "max_ids": max_ids,
        "prediction_max": ANSWER_RELATIONS[pred_max_i],
        "prediction_sum": ANSWER_RELATIONS[pred_sum_i],
        "top1_logit_margin": float(
            (max_logits_t[order[0]] - max_logits_t[order[1]]).item()
        ),
        "total_answer_word_mass": float(sum_probs_t.sum().item()),
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


def compile_answer_pattern(search_words: Sequence[str]) -> re.Pattern[str]:
    clean = []
    for raw in search_words:
        word = str(raw).strip().lower()
        if word not in ANSWER_TO_INTERNAL:
            raise ValueError(
                f"Unsupported --search-words entry {raw!r}; supported="
                f"{sorted(ANSWER_TO_INTERNAL)}"
            )
        clean.append(word)
    clean = list(dict.fromkeys(clean))
    if not clean:
        raise ValueError("--search-words resolved to an empty list")
    # Alphabetic boundaries prevent matching 'on' inside 'condition'.
    body = "|".join(re.escape(word) for word in sorted(clean, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])({body})(?![A-Za-z])", flags=re.IGNORECASE)


def locate_last_answer_token(
    tokenizer: Any,
    generated_ids: Sequence[int],
    pattern: re.Pattern[str],
) -> Optional[Dict[str, Any]]:
    """Map the final text's LAST answer-word occurrence to generated token indices."""
    token_ids = [int(x) for x in generated_ids]
    prefixes = [""]
    for end in range(1, len(token_ids) + 1):
        prefixes.append(
            safe_decode(
                tokenizer,
                token_ids[:end],
                skip_special_tokens=True,
            )
        )
    final_text = prefixes[-1]
    matches = list(pattern.finditer(final_text))
    if not matches:
        return None
    match = matches[-1]
    char_start, char_end = int(match.start()), int(match.end())

    token_start: Optional[int] = None
    for index in range(len(token_ids)):
        if len(prefixes[index + 1]) > char_start:
            token_start = index
            break
    if token_start is None:
        return None

    token_end: Optional[int] = None
    for index in range(token_start, len(token_ids)):
        if len(prefixes[index + 1]) >= char_end:
            token_end = index
            break
    if token_end is None:
        return None

    surface = str(match.group(1))
    answer_internal = ANSWER_TO_INTERNAL[surface.lower()]
    answer_relation = INTERNAL_TO_ANSWER[answer_internal]

    return {
        "generated_text": final_text,
        "answer_surface": surface,
        "answer_surface_lower": surface.lower(),
        "answer_relation": answer_relation,
        "answer_internal_relation": answer_internal,
        "answer_char_start": char_start,
        "answer_char_end": char_end,
        "answer_token_start_0based": int(token_start),
        "answer_token_end_0based": int(token_end),
        "answer_token_start_1based": int(token_start + 1),
        "answer_token_end_1based": int(token_end + 1),
        "answer_token_span_length": int(token_end - token_start + 1),
        "answer_is_single_token": bool(token_start == token_end),
        "answer_token_ids": token_ids[token_start : token_end + 1],
        "answer_token_piece": safe_decode(
            tokenizer,
            token_ids[token_start : token_end + 1],
            skip_special_tokens=False,
        ),
        "generated_token_count": int(len(token_ids)),
        "prefix_text_before_answer": final_text[:char_start],
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

    sample_rows_out: List[Dict[str, Any]] = []
    case_rows_out: List[Dict[str, Any]] = []
    inventory_rows_out: List[Dict[str, Any]] = []
    missing_rows_out: List[Dict[str, Any]] = []

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

        print("\n" + "=" * 150)
        print(
            f"MODEL={model_name} | best Sim layer=L{best_layer} | "
            f"N={len(saved_samples)} | full_generation_max_new_tokens={args.max_new_tokens}"
        )
        print("=" * 150)

        model, processor = load_model_and_processor(
            step1,
            specs[model_name],
            args.device,
        )
        vocab_size = int(model.get_output_embeddings().weight.shape[0])
        variant_bank, inventory = build_variant_bank(
            processor.tokenizer,
            vocab_size,
        )
        for row in inventory:
            row["model"] = model_name
        inventory_rows_out.extend(inventory)

        for sample_index, saved in enumerate(
            tqdm(saved_samples, desc=model_name, dynamic_ncols=True)
        ):
            sid = int(saved["sid"])
            gt_internal = step1.normalize_relation(saved.get("gt"))
            if gt_internal not in INTERNAL_TO_INDEX:
                raise RuntimeError(f"Invalid GT sid={sid}: {saved.get('gt')}")
            gt_answer = INTERNAL_TO_ANSWER[gt_internal]

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
                pos = int(positions[0])
                original_centroids = np.asarray(
                    data["original_similarity_centroids"][pos],
                    dtype=np.float64,
                )
                swapped_role = np.asarray(
                    data["swapped_similarity_centroids_role_order"][pos],
                    dtype=np.float64,
                )
                swapped_aligned = swapped_role[[1, 0], :]
                average_centroids = 0.5 * (original_centroids + swapped_aligned)
                original_sep = float(data["original_similarity_separation"][pos])
                swapped_sep = float(data["swapped_similarity_separation"][pos])

            sim = relation_scores_from_centroids(average_centroids)
            sim_scores = sim["scores"]
            gt_internal_index = INTERNAL_TO_INDEX[gt_internal]
            sim_gt_margin = float(
                sim_scores[gt_internal_index]
                - np.max(np.delete(sim_scores, gt_internal_index))
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
                # This normally should be equal in greedy decoding. Keep the common
                # prefix rather than silently indexing a wrong step.
                common = min(len(score_steps), len(generated_ids))
                generated_ids = generated_ids[:common]
                score_steps = score_steps[:common]

            located = locate_last_answer_token(
                processor.tokenizer,
                generated_ids,
                answer_pattern,
            )
            generated_text = safe_decode(
                processor.tokenizer,
                generated_ids,
                skip_special_tokens=True,
            )

            raw_similarity = extract_raw_similarity_metrics(
                step1=step1,
                generated=generated,
                batch=batch,
                processor=processor,
                model=model,
                prompt=prompt,
                best_layer=best_layer,
                temperature=float(args.temperature),
            )

            base_row: Dict[str, Any] = {
                "model": model_name,
                "sid": sid,
                "gt_internal": gt_internal,
                "gt_answer": gt_answer,
                "question": str(saved.get("question", prompt.get("question_text", ""))),
                "saved_generation_text": str(saved.get("original_generated_text", "")),
                "rerun_generation_text": generated_text,
                "rerun_matches_saved_generation": (
                    generated_text.strip()
                    == str(saved.get("original_generated_text", "")).strip()
                ),
                "saved_generation_prediction": step1.normalize_relation(
                    saved.get("original_prediction")
                ),
                "saved_generation_correct": bool(saved.get("original_correct", False)),
                "generated_token_count": len(generated_ids),
                "best_similarity_layer": best_layer,
                "sim_prediction": sim["prediction"],
                "sim_correct": sim["prediction"] == gt_internal,
                "sim_dx": sim["dx"],
                "sim_dy": sim["dy"],
                "sim_distance": sim["distance"],
                "sim_axis_confidence": sim["axis_confidence"],
                "sim_top1_margin": sim["top1_margin"],
                "sim_gt_margin": sim_gt_margin,
                "sim_map_separation": 0.5 * (original_sep + swapped_sep),
                **raw_similarity,
            }
            for rel_i, rel in enumerate(INTERNAL_RELATIONS):
                base_row[f"sim_score_{rel}"] = float(sim_scores[rel_i])

            if located is None:
                missing_row = {
                    **base_row,
                    "reason": "no_configured_answer_word_in_full_generation",
                    "search_words": ",".join(search_words),
                }
                missing_rows_out.append(missing_row)
                sample_rows_out.append({
                    **base_row,
                    "answer_found": False,
                })
                if not args.quiet_samples:
                    print(
                        f"[{model_name}] sid={sid:4d} | ANSWER NOT FOUND | "
                        f"tokens={len(generated_ids)} | text={generated_text!r}",
                        flush=True,
                    )
            else:
                answer_step = int(located["answer_token_start_0based"])
                if not (0 <= answer_step < len(score_steps)):
                    raise RuntimeError(
                        f"sid={sid}: answer step {answer_step} outside scores={len(score_steps)}"
                    )
                logits = score_steps[answer_step][0].float()
                full_probs = torch.softmax(logits, dim=-1)
                realized_token_id = int(generated_ids[answer_step])
                realized_token_probability = float(full_probs[realized_token_id].item())
                realized_token_logprob = float(
                    torch.log(full_probs[realized_token_id].clamp_min(1e-45)).item()
                )

                primary = score_answer_relations(logits, variant_bank["all"])
                if not primary.get("valid", False):
                    raise RuntimeError(
                        f"{model_name}: all-case bank invalid: "
                        f"{primary.get('missing_relations')}"
                    )
                gt_answer_index = ANSWER_TO_INDEX[gt_answer]
                answer_relation_index = ANSWER_TO_INDEX[located["answer_relation"]]
                max_logits = primary["max_logits"]
                lm_gt_logit_margin = float(
                    max_logits[gt_answer_index]
                    - np.max(np.delete(max_logits, gt_answer_index))
                )

                last_answer_internal = str(located["answer_internal_relation"])
                last_answer_correct = bool(last_answer_internal == gt_internal)
                saved_vs_last_disagree = bool(
                    step1.normalize_relation(saved.get("original_prediction"))
                    != last_answer_internal
                )

                row = {
                    **base_row,
                    **located,
                    "answer_found": True,
                    "last_answer_correct": last_answer_correct,
                    "saved_vs_last_answer_disagree": saved_vs_last_disagree,
                    "answer_realized_token_id": realized_token_id,
                    "answer_realized_token_piece": safe_decode(
                        processor.tokenizer,
                        [realized_token_id],
                        skip_special_tokens=False,
                    ),
                    "answer_realized_token_probability": realized_token_probability,
                    "answer_realized_token_logprob": realized_token_logprob,
                    "lm_prediction_max_case_variant": primary["prediction_max"],
                    "lm_prediction_sum_case_mass": primary["prediction_sum"],
                    "lm_max_case_accuracy": primary["prediction_max"] == gt_answer,
                    "lm_sum_case_accuracy": primary["prediction_sum"] == gt_answer,
                    "lm_generated_answer_max_variant_probability": float(
                        primary["max_probs"][answer_relation_index]
                    ),
                    "lm_generated_answer_sum_variant_probability": float(
                        primary["sum_probs"][answer_relation_index]
                    ),
                    "lm_generated_answer_conditional_probability": float(
                        primary["conditional_max"][answer_relation_index]
                    ),
                    "lm_gt_max_variant_probability": float(
                        primary["max_probs"][gt_answer_index]
                    ),
                    "lm_gt_sum_variant_probability": float(
                        primary["sum_probs"][gt_answer_index]
                    ),
                    "lm_gt_conditional_probability": float(
                        primary["conditional_max"][gt_answer_index]
                    ),
                    "lm_gt_logit_margin": lm_gt_logit_margin,
                    "lm_top1_logit_margin": primary["top1_logit_margin"],
                    "lm_total_answer_word_mass": primary["total_answer_word_mass"],
                    "sim_lm_prediction_agree": (
                        INTERNAL_TO_ANSWER[sim["prediction"]]
                        == primary["prediction_max"]
                    ),
                }

                for rel_i, rel in enumerate(ANSWER_RELATIONS):
                    row[f"lm_max_variant_prob_{rel}"] = float(
                        primary["max_probs"][rel_i]
                    )
                    row[f"lm_sum_variant_prob_{rel}"] = float(
                        primary["sum_probs"][rel_i]
                    )
                    row[f"lm_conditional_prob_{rel}"] = float(
                        primary["conditional_max"][rel_i]
                    )
                    row[f"lm_max_variant_logit_{rel}"] = float(
                        primary["max_logits"][rel_i]
                    )
                    best_id = int(primary["max_ids"][rel_i])
                    row[f"lm_best_variant_id_{rel}"] = best_id
                    row[f"lm_best_variant_text_{rel}"] = safe_decode(
                        processor.tokenizer,
                        [best_id],
                        skip_special_tokens=False,
                    )

                sample_rows_out.append(row)

                for case_mode, ids_by_relation in variant_bank.items():
                    scored = score_answer_relations(logits, ids_by_relation)
                    valid_case = bool(scored.get("valid", False))
                    case_rows_out.append({
                        "model": model_name,
                        "sid": sid,
                        "case_mode": case_mode,
                        "valid_case_bank": valid_case,
                        "missing_relations": ",".join(
                            scored.get("missing_relations", [])
                        ),
                        "gt_answer": gt_answer,
                        "last_answer_relation": located["answer_relation"],
                        "answer_token_position_1based": located[
                            "answer_token_start_1based"
                        ],
                        "answer_is_single_token": located["answer_is_single_token"],
                        "prediction_max_variant": (
                            scored["prediction_max"] if valid_case else None
                        ),
                        "correct_max_variant": (
                            scored["prediction_max"] == gt_answer
                            if valid_case else np.nan
                        ),
                        "prediction_sum_mass": (
                            scored["prediction_sum"] if valid_case else None
                        ),
                        "correct_sum_mass": (
                            scored["prediction_sum"] == gt_answer
                            if valid_case else np.nan
                        ),
                        "total_answer_word_mass": (
                            scored["total_answer_word_mass"] if valid_case else np.nan
                        ),
                    })

                if not args.quiet_samples:
                    probs = " ".join(
                        f"{rel}={primary['max_probs'][rel_i]:.6g}"
                        for rel_i, rel in enumerate(ANSWER_RELATIONS)
                    )
                    span = (
                        f"{located['answer_token_start_1based']}"
                        if located["answer_is_single_token"]
                        else (
                            f"{located['answer_token_start_1based']}-"
                            f"{located['answer_token_end_1based']}"
                        )
                    )
                    print(
                        f"[{model_name}] sid={sid:4d} | "
                        f"answer={located['answer_surface']!r} "
                        f"token={span}/{located['generated_token_count']} "
                        f"piece={located['answer_token_piece']!r} "
                        f"p(realized)={realized_token_probability:.6g} | "
                        f"{probs} | pred={primary['prediction_max']} "
                        f"gt={gt_answer} correct={last_answer_correct}",
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

    sample_df = pd.DataFrame(sample_rows_out)
    case_df = pd.DataFrame(case_rows_out)
    inventory_df = pd.DataFrame(inventory_rows_out)
    missing_df = pd.DataFrame(missing_rows_out)

    sample_df.to_csv(output_dir / "sample_metrics.csv", index=False)
    case_df.to_csv(output_dir / "token_case_per_sample.csv", index=False)
    inventory_df.to_csv(output_dir / "token_variant_inventory.csv", index=False)
    missing_df.to_csv(output_dir / "missing_answer_samples.csv", index=False)

    found_df = sample_df[sample_df["answer_found"] == True].copy()  # noqa: E712

    if not case_df.empty:
        case_summary = (
            case_df.groupby(["model", "case_mode"], as_index=False)
            .agg(
                N=("sid", "size"),
                valid_case_bank_rate=("valid_case_bank", "mean"),
                single_token_rate=("answer_is_single_token", "mean"),
                accuracy_max_variant=("correct_max_variant", "mean"),
                accuracy_sum_mass=("correct_sum_mass", "mean"),
                mean_total_answer_word_mass=("total_answer_word_mass", "mean"),
                mean_answer_token_position=("answer_token_position_1based", "mean"),
            )
        )
    else:
        case_summary = pd.DataFrame()
    case_summary.to_csv(output_dir / "token_case_accuracy.csv", index=False)

    correlation_pairs = [
        ("raw_sim_peak_mean", "answer_realized_token_probability"),
        ("raw_sim_peak_min", "answer_realized_token_probability"),
        ("raw_sim_peak_gap_mean", "answer_realized_token_probability"),
        ("raw_sim_entropy_confidence_mean", "answer_realized_token_probability"),
        ("raw_sim_object_map_separation", "answer_realized_token_probability"),
        ("raw_sim_peak_mean", "lm_generated_answer_max_variant_probability"),
        ("raw_sim_peak_mean", "lm_gt_max_variant_probability"),
        ("raw_sim_peak_mean", "lm_gt_conditional_probability"),
        ("sim_gt_margin", "lm_gt_logit_margin"),
        ("sim_top1_margin", "lm_top1_logit_margin"),
        ("sim_distance", "answer_realized_token_probability"),
        ("sim_axis_confidence", "answer_realized_token_probability"),
        ("sim_map_separation", "answer_realized_token_probability"),
    ]

    correlation_rows: List[Dict[str, Any]] = []
    for model_name, frame in found_df.groupby("model"):
        subsets = {
            "all_found": frame,
            "last_answer_correct": frame[frame["last_answer_correct"] == True],  # noqa: E712
            "last_answer_wrong": frame[frame["last_answer_correct"] == False],  # noqa: E712
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
        frame_all = sample_df[sample_df["model"] == model_name]
        frame = found_df[found_df["model"] == model_name]
        model_summary_rows.append({
            "model": model_name,
            "N_total": int(len(frame_all)),
            "N_answer_found": int(len(frame)),
            "answer_found_rate": float(len(frame) / max(1, len(frame_all))),
            "best_similarity_layer": int(frame_all["best_similarity_layer"].iloc[0]),
            "rerun_matches_saved_generation_rate": float(
                frame_all["rerun_matches_saved_generation"].mean()
            ),
            "saved_generation_accuracy": float(
                frame_all["saved_generation_correct"].mean()
            ),
            "last_answer_accuracy_found": (
                float(frame["last_answer_correct"].mean()) if len(frame) else np.nan
            ),
            "saved_vs_last_answer_disagreement_rate": (
                float(frame["saved_vs_last_answer_disagree"].mean())
                if len(frame)
                else np.nan
            ),
            "similarity_accuracy": float(frame_all["sim_correct"].mean()),
            "answer_single_token_rate": (
                float(frame["answer_is_single_token"].mean()) if len(frame) else np.nan
            ),
            "mean_answer_token_position": (
                float(frame["answer_token_start_1based"].mean())
                if len(frame)
                else np.nan
            ),
            "answer_step_max_case_accuracy": (
                float(frame["lm_max_case_accuracy"].mean()) if len(frame) else np.nan
            ),
            "answer_step_sum_case_accuracy": (
                float(frame["lm_sum_case_accuracy"].mean()) if len(frame) else np.nan
            ),
            "mean_realized_answer_token_probability": (
                float(frame["answer_realized_token_probability"].mean())
                if len(frame)
                else np.nan
            ),
            "mean_gt_relation_probability_at_answer_step": (
                float(frame["lm_gt_max_variant_probability"].mean())
                if len(frame)
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
                "answer_realized_token_probability",
                "Best-layer raw similarity peak mean",
                "Probability of realized answer token",
                "raw_similarity_vs_realized_answer_probability",
            ),
            (
                "sim_gt_margin",
                "lm_gt_logit_margin",
                "Similarity GT geometric margin",
                "GT answer-token logit margin at answer step",
                "sim_gt_margin_vs_answer_step_gt_margin",
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
                fig.savefig(
                    plots_dir / f"{model_name}_{suffix}.png",
                    dpi=180,
                )
                plt.close(fig)
    except Exception as exc:
        print(f"[warning] plotting skipped: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 160)
    print("MODEL SUMMARY")
    print("=" * 160)
    print(
        model_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 160)
    print("TOKEN CASE ACCURACY AT THE LOCATED ANSWER TOKEN")
    print("=" * 160)
    if case_summary.empty:
        print("(no located answers)")
    else:
        print(
            case_summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
