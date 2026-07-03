#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layerwise logit-lens analysis for frozen HF LLaVA-1.5 under RMSNorm-epsilon
changes, without AdaptVis.

What it measures
----------------
For the initial multimodal prefill pass, this script captures the residual
stream at the final prompt position:
    s_0          : input to decoder layer 0
    s_{l+1}      : output of decoder layer l, l=0,...,31

At every captured state it applies the language model's final RMSNorm and
lm_head, then records:
  * four-way left/right/on/under logit-lens scores;
  * the four-way probability and gold margin;
  * full-vocabulary top-k tokens and candidate probability mass;
  * an optional fixed-reference final-norm probe, so different model epsilons
    can be compared through the same output normalization map.

No attention module, visual token, prompt, or generation rule is altered.
Only LlamaRMSNorm.variance_epsilon changes between runs.

Recommended first run
---------------------
Use a small set of known flip SIDs, then inspect the produced JSON / plot:

python3 run_llava15_hf_eps_logit_lens.py \\
  --dataset Controlled_Images_A --option four \\
  --eps-list 1e-5,1e-6,1e-7,1e-4 \\
  --sids 42,54,80,165,189 \\
  --reference-eps 1e-5 \\
  --top-k 10 --dtype float32 --download \\
  --output output/llava15_A_eps_logitlens_flips.json

Then draw four-way gold margins / probabilities:

python3 plot_llava15_hf_eps_logit_lens.py \\
  --input output/llava15_A_eps_logitlens_flips.json \\
  --output-dir output/llava15_A_eps_logitlens_plots
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from transformers import AutoProcessor, LlavaConfig, LlavaForConditionalGeneration
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "a272c74"
RELATIONS = ("left", "right", "on", "under")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-AdaptVis LLaVA-1.5 RMSNorm-epsilon logit-lens analysis."
        )
    )
    parser.add_argument(
        "--dataset",
        default="Controlled_Images_A",
        choices=["Controlled_Images_A", "Controlled_Images_B"],
    )
    parser.add_argument("--option", default="four", choices=["two", "four", "six"])
    parser.add_argument(
        "--eps-list",
        default="1e-5,1e-6,1e-7,1e-4",
        help="Comma-separated RMSNorm epsilon values, e.g. 1e-5,1e-6,1e-7.",
    )
    parser.add_argument(
        "--reference-eps",
        type=float,
        default=1e-5,
        help=(
            "A shared epsilon used only for the fixed-reference logit lens. "
            "Set <=0 to disable the fixed-reference lens."
        ),
    )
    parser.add_argument(
        "--sids",
        default=None,
        help=(
            "Comma-separated sample IDs to inspect. Omit to scan all samples, "
            "or combine with --limit for the first N samples."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16", "auto"],
    )
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--print-first", default=5, type=int)
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path. A descriptive default is used when omitted.",
    )
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def parse_eps_list(text: str) -> List[float]:
    values: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0:
            raise ValueError(f"RMSNorm epsilon must be positive; got {value}.")
        values.append(value)
    if not values:
        raise ValueError("--eps-list did not contain any positive value.")
    return values


def parse_sids(text: Optional[str]) -> Optional[List[int]]:
    if text is None or not text.strip():
        return None
    values: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        return None
    return sorted(set(values))


def eps_key(eps: float) -> str:
    return f"{eps:.0e}".replace("e-0", "e-")


def norm_gold(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def load_prompts(dataset: str, option: str) -> Tuple[List[str], List[Any]]:
    path = Path(f"prompts/{dataset}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Run this script from the AdaptVis repository root."
        )
    prompts: List[str] = []
    answers: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            prompts.append(item["question"])
            answers.append(item["answer"])
    return prompts, answers


def extract_images_from_batch(batch: Mapping[str, Any]) -> Iterable[Any]:
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def set_rmsnorm_eps(model: LlavaForConditionalGeneration, eps: float) -> Tuple[int, List[float]]:
    norms = [
        module
        for module in model.language_model.modules()
        if isinstance(module, LlamaRMSNorm)
    ]
    if not norms:
        raise RuntimeError("No LlamaRMSNorm modules found in model.language_model.")
    before = sorted({float(module.variance_epsilon) for module in norms})
    for module in norms:
        module.variance_epsilon = float(eps)
    model.language_model.config.rms_norm_eps = float(eps)
    model.config.text_config.rms_norm_eps = float(eps)
    after = sorted({float(module.variance_epsilon) for module in norms})
    if after != [float(eps)]:
        raise RuntimeError(f"Failed to set RMSNorm eps to {eps}; got {after}.")
    return len(norms), before


def discover_relation_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Find every one-token lowercase spelling whose decoded text is a relation.

    We aggregate variants such as ``'left'`` and ``' left'`` with logsumexp.
    This avoids assuming a specific LLaMA whitespace token at the first answer
    position.
    """
    discovered: Dict[str, List[int]] = defaultdict(list)
    word_re = re.compile(r"^[a-z]+$")
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))

    for token_id in range(vocab_size):
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        normalized = text.strip().lower()
        if not word_re.fullmatch(normalized):
            continue
        if normalized in RELATIONS:
            discovered[normalized].append(token_id)

    result: Dict[str, List[int]] = {}
    for relation in RELATIONS:
        ids = sorted(set(discovered.get(relation, [])))
        if not ids:
            # Fallback: standard leading-space realization must be one token.
            candidate_ids = tokenizer(
                " " + relation,
                add_special_tokens=False,
            ).input_ids
            if len(candidate_ids) == 1:
                ids = [int(candidate_ids[0])]
        if not ids:
            raise RuntimeError(
                "Could not find a one-token realization for relation "
                f"{relation!r}. This script expects the four labels to be "
                "single tokens under the LLaMA tokenizer."
            )
        result[relation] = ids
    return result


def safe_token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).replace("\n", "\\n")


def rmsnorm_project(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the final LLaMA RMSNorm then lm_head to a single residual vector."""
    x = hidden.float()
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
    normed = x * denom
    normed = normed * weight.float()
    # F.linear does not accept mixed float32 / fp16 weights.
    return F.linear(normed.to(lm_head_weight.dtype), lm_head_weight).float()


@dataclass
class CapturedResiduals:
    query_index: int = -1
    input_state: Optional[torch.Tensor] = None
    layer_states: Optional[List[Optional[torch.Tensor]]] = None

    def __post_init__(self) -> None:
        if self.layer_states is None:
            self.layer_states = []

    def clear(self) -> None:
        self.input_state = None
        self.layer_states = [None for _ in self.layer_states]


class ResidualCapture:
    """Read-only hooks that keep only the final-query residual vector."""

    def __init__(self, language_model) -> None:
        self.language_model = language_model
        self.layers = list(language_model.model.layers)
        self.state = CapturedResiduals(
            query_index=-1,
            layer_states=[None for _ in self.layers],
        )
        self.handles: List[Any] = []

        self.handles.append(self.layers[0].register_forward_pre_hook(self._input_hook))
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._make_layer_hook(index)))

    def _take_last(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise RuntimeError(f"Expected [1,L,H] hidden states, got {tuple(hidden.shape)}")
        index = self.state.query_index
        if index < 0:
            index = hidden.shape[1] + index
        if not 0 <= index < hidden.shape[1]:
            raise RuntimeError(
                f"Invalid final query index {self.state.query_index} for sequence length {hidden.shape[1]}."
            )
        # clone avoids retaining the full layer activation graph/storage.
        return hidden[0, index, :].detach().clone()

    def _input_hook(self, _module, inputs) -> None:
        self.state.input_state = self._take_last(inputs[0])

    def _make_layer_hook(self, index: int):
        def hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.state.layer_states[index] = self._take_last(hidden)
        return hook

    def begin(self, query_index: int = -1) -> None:
        self.state.query_index = int(query_index)
        self.state.input_state = None
        self.state.layer_states = [None for _ in self.layers]

    def collect(self) -> List[Tuple[str, int, torch.Tensor]]:
        if self.state.input_state is None:
            raise RuntimeError("Residual input hook did not fire.")
        missing = [
            idx
            for idx, value in enumerate(self.state.layer_states or [])
            if value is None
        ]
        if missing:
            raise RuntimeError(f"Residual hooks missing layers: {missing}")
        values: List[Tuple[str, int, torch.Tensor]] = [("input_to_layer_0", -1, self.state.input_state)]
        values.extend(
            (f"after_layer_{idx}", idx, value)
            for idx, value in enumerate(self.state.layer_states or [])
            if value is not None
        )
        return values

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def build_lens_entry(
    *,
    logits: torch.Tensor,
    tokenizer,
    relation_ids: Mapping[str, Sequence[int]],
    gold: str,
    stage: str,
    layer_index: int,
    top_k: int,
) -> Dict[str, Any]:
    relation_scores: Dict[str, torch.Tensor] = {}
    for relation, ids in relation_ids.items():
        index = torch.tensor(ids, device=logits.device, dtype=torch.long)
        relation_scores[relation] = torch.logsumexp(logits.index_select(0, index), dim=0)

    stacked = torch.stack([relation_scores[relation] for relation in RELATIONS])
    fourway_log_probs = torch.log_softmax(stacked, dim=0)
    fourway_probs = torch.exp(fourway_log_probs)
    pred_index = int(torch.argmax(stacked).item())
    pred_relation = RELATIONS[pred_index]

    vocab_lse = torch.logsumexp(logits, dim=0)
    group_vocab_probs = {
        relation: float(torch.exp(score - vocab_lse).detach().cpu())
        for relation, score in relation_scores.items()
    }

    gold_key = gold.strip().lower()
    if gold_key in relation_scores:
        wrong_scores = torch.stack(
            [relation_scores[r] for r in RELATIONS if r != gold_key]
        )
        gold_margin = float(
            (relation_scores[gold_key] - torch.max(wrong_scores)).detach().cpu()
        )
        gold_prob_4way = float(fourway_probs[RELATIONS.index(gold_key)].detach().cpu())
        gold_prob_vocab = group_vocab_probs[gold_key]
    else:
        gold_margin = float("nan")
        gold_prob_4way = float("nan")
        gold_prob_vocab = float("nan")

    values, indices = torch.topk(logits, k=min(int(top_k), int(logits.numel())))
    top_vocab: List[Dict[str, Any]] = []
    for value, token_id in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        top_vocab.append(
            {
                "id": int(token_id),
                "text": safe_token_text(tokenizer, int(token_id)),
                "logit": float(value),
                "prob": float(math.exp(float(value) - float(vocab_lse.detach().cpu()))),
            }
        )

    return {
        "stage": stage,
        "layer_index": int(layer_index),
        "fourway_logits": {
            relation: float(relation_scores[relation].detach().cpu())
            for relation in RELATIONS
        },
        "fourway_probs": {
            relation: float(fourway_probs[idx].detach().cpu())
            for idx, relation in enumerate(RELATIONS)
        },
        "fourway_prediction": pred_relation,
        "gold_margin_vs_best_other": gold_margin,
        "gold_prob_4way": gold_prob_4way,
        "candidate_vocab_probs": group_vocab_probs,
        "gold_candidate_vocab_prob": gold_prob_vocab,
        "top_vocab": top_vocab,
    }


def make_output_path(args: argparse.Namespace, eps_values: Sequence[float]) -> Path:
    if args.output:
        return Path(args.output)
    eps_name = "_".join(eps_key(value).replace("-", "m") for value in eps_values)
    sid_name = "all" if args.sids is None else f"{len(parse_sids(args.sids) or [])}sids"
    return Path(f"output/llava15_{args.dataset}_logitlens_{eps_name}_{sid_name}.json")


def load_model_and_processor(args: argparse.Namespace):
    print(f"transformers version: {transformers.__version__}")
    llava_config = LlavaConfig.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    checkpoint_eps = float(llava_config.text_config.rms_norm_eps)
    print(f"Loading LLaVA from {args.model}@{args.revision}")
    print(f"Checkpoint text_config.rms_norm_eps: {checkpoint_eps}")

    kwargs: Dict[str, Any] = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "config": llava_config,
    }
    if args.dtype != "float32":
        kwargs["torch_dtype"] = resolve_dtype(args.dtype)

    model = LlavaForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model.eval()
    model.to(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Decoder layers: {len(model.language_model.model.layers)}")
    return model, processor, checkpoint_eps


@torch.inference_mode()
def analyze_one_pass(
    *,
    model: LlavaForConditionalGeneration,
    processor,
    residual_capture: ResidualCapture,
    image: Any,
    prompt: str,
    gold: str,
    relation_ids: Mapping[str, Sequence[int]],
    active_eps: float,
    reference_eps: Optional[float],
    top_k: int,
    device: str,
) -> Dict[str, Any]:
    model_dtype = next(model.parameters()).dtype
    inputs = processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=77,
    )
    inputs = inputs.to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=model_dtype)

    residual_capture.begin(query_index=-1)
    outputs = model(
        **inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    residuals = residual_capture.collect()
    final_norm = model.language_model.model.norm
    lm_head_weight = model.language_model.lm_head.weight

    active_entries: List[Dict[str, Any]] = []
    reference_entries: List[Dict[str, Any]] = []
    for stage, layer_index, residual in residuals:
        active_logits = rmsnorm_project(
            residual,
            final_norm.weight,
            active_eps,
            lm_head_weight,
        )
        active_entries.append(
            build_lens_entry(
                logits=active_logits,
                tokenizer=processor.tokenizer,
                relation_ids=relation_ids,
                gold=gold,
                stage=stage,
                layer_index=layer_index,
                top_k=top_k,
            )
        )
        if reference_eps is not None:
            reference_logits = rmsnorm_project(
                residual,
                final_norm.weight,
                reference_eps,
                lm_head_weight,
            )
            reference_entries.append(
                build_lens_entry(
                    logits=reference_logits,
                    tokenizer=processor.tokenizer,
                    relation_ids=relation_ids,
                    gold=gold,
                    stage=stage,
                    layer_index=layer_index,
                    top_k=top_k,
                )
            )

    final_logits = outputs.logits[0, -1, :].detach().float()
    final_lens_logits = rmsnorm_project(
        residuals[-1][2],
        final_norm.weight,
        active_eps,
        lm_head_weight,
    )
    lens_max_abs_error = float(torch.max(torch.abs(final_logits - final_lens_logits)).cpu())
    lens_mean_abs_error = float(torch.mean(torch.abs(final_logits - final_lens_logits)).cpu())
    final_direct = build_lens_entry(
        logits=final_logits,
        tokenizer=processor.tokenizer,
        relation_ids=relation_ids,
        gold=gold,
        stage="model_output_logits",
        layer_index=len(residuals) - 2,
        top_k=top_k,
    )

    return {
        "input_token_length": int(inputs["input_ids"].shape[-1]),
        "merged_sequence_length": int(outputs.logits.shape[1]),
        "final_output": final_direct,
        "active_lens": active_entries,
        "fixed_reference_lens": reference_entries if reference_eps is not None else None,
        "final_lens_match": {
            "max_abs_error": lens_max_abs_error,
            "mean_abs_error": lens_mean_abs_error,
        },
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    eps_values = parse_eps_list(args.eps_list)
    selected_sids = parse_sids(args.sids)
    selected_set = set(selected_sids) if selected_sids is not None else None
    reference_eps: Optional[float] = (
        float(args.reference_eps) if float(args.reference_eps) > 0.0 else None
    )

    output_path = make_output_path(args, eps_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor, checkpoint_eps = load_model_and_processor(args)
    relation_ids = discover_relation_token_ids(processor.tokenizer)
    print("Relation candidate token IDs:")
    for relation in RELATIONS:
        rendered = [safe_token_text(processor.tokenizer, token_id) for token_id in relation_ids[relation]]
        print(f"  {relation}: {relation_ids[relation]} -> {rendered}")

    prompts, answers = load_prompts(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)
    total_available = min(len(prompts), len(dataset))
    if selected_sids is not None:
        bad = [sid for sid in selected_sids if sid < 0 or sid >= total_available]
        if bad:
            raise ValueError(f"Requested SIDs outside [0,{total_available}): {bad}")
        target_count = len(selected_sids)
    else:
        target_count = min(total_available, args.limit) if args.limit is not None else total_available

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=repository_default_collate,
    )

    residual_capture = ResidualCapture(model.language_model)
    all_eps_records: Dict[str, List[Dict[str, Any]]] = {}

    try:
        for eps in eps_values:
            count_norms, before = set_rmsnorm_eps(model, eps)
            print("\n" + "=" * 100)
            print(f"Active RMSNorm epsilon: {eps} ({eps_key(eps)})")
            print(f"RMSNorm modules: {count_norms}; pre-run values: {before}")
            print("No AdaptVis patch is installed.")

            records: List[Dict[str, Any]] = []
            progress = tqdm(total=target_count, desc=f"LLaVA logit lens eps={eps_key(eps)}")
            sid = 0
            for batch in loader:
                for image in extract_images_from_batch(batch):
                    if sid >= total_available:
                        break
                    include = (
                        sid in selected_set
                        if selected_set is not None
                        else (args.limit is None or sid < args.limit)
                    )
                    if include:
                        prompt = prompts[sid]
                        gold = norm_gold(answers[sid])
                        analysis = analyze_one_pass(
                            model=model,
                            processor=processor,
                            residual_capture=residual_capture,
                            image=image,
                            prompt=prompt,
                            gold=gold,
                            relation_ids=relation_ids,
                            active_eps=eps,
                            reference_eps=reference_eps,
                            top_k=args.top_k,
                            device=args.device,
                        )
                        record = {
                            "sid": sid,
                            "gold": gold,
                            "prompt": prompt,
                            "eps": float(eps),
                            **analysis,
                        }
                        records.append(record)

                        if len(records) <= args.print_first:
                            final = analysis["final_output"]
                            match = analysis["final_lens_match"]
                            print("\n" + "-" * 100)
                            print(f"[SID {sid}] gold={gold!r}")
                            print(f"final four-way pred={final['fourway_prediction']!r}")
                            print(f"four-way probs={final['fourway_probs']}")
                            print(f"gold margin={final['gold_margin_vs_best_other']:.5f}")
                            print(
                                "final lens vs model logits: "
                                f"max_abs={match['max_abs_error']:.3e}, "
                                f"mean_abs={match['mean_abs_error']:.3e}"
                            )
                            last_lens = analysis["active_lens"][-1]
                            print(f"last active-lens top vocab={last_lens['top_vocab'][:5]}")

                        progress.update(1)
                    sid += 1
                if sid >= total_available:
                    break
            progress.close()
            all_eps_records[eps_key(eps)] = records
    finally:
        residual_capture.close()

    payload = {
        "metadata": {
            "model": args.model,
            "revision": args.revision,
            "dataset": args.dataset,
            "option": args.option,
            "seed": args.seed,
            "dtype": args.dtype,
            "checkpoint_rms_norm_eps": checkpoint_eps,
            "eps_list": eps_values,
            "reference_eps": reference_eps,
            "selected_sids": selected_sids,
            "candidate_relation_token_ids": relation_ids,
            "lens_definition": {
                "position": "final prefill query / last merged sequence position",
                "states": "input to layer 0 plus output after every decoder layer",
                "active_lens": "active final RMSNorm epsilon + lm_head",
                "fixed_reference_lens": (
                    "reference final RMSNorm epsilon + lm_head"
                    if reference_eps is not None
                    else None
                ),
                "adaptvis": "not installed / not used",
            },
        },
        "eps_runs": all_eps_records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print(f"Saved logit-lens JSON to: {output_path}")


if __name__ == "__main__":
    main()
