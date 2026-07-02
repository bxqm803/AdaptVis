#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-VL-Chat RMSNorm-epsilon ablation with native Qwen AdaptVis.

This runner keeps Qwen-VL-Chat's original visual-token insertion and generation
path. During the initial multimodal prefill only, it multiplies the raw
pre-softmax attention logits from the final query position to visual-token keys,
in layers [0, --max-layers), exactly analogous to the LLaVA AdaptVis protocol.

For --method adapt_vis:
  1) Generate a no-intervention probe (weight=1.0).
  2) Take the maximum first-step probability, round it to two decimals.
  3) Select weight1 if rounded confidence < threshold, otherwise weight2.
  4) Regenerate with the selected image-logit multiplier.

The Qwen-VL remote implementation uses a fused c_attn projection and its own
visual-token insertion. Therefore this is not a copy of the LLaVA attention
patch; it patches QWenAttention._attn while preserving Qwen's normal forward
and cache behavior outside the single prefill intervention.

Run from the AdaptVis repository root, where dataset_zoo.py and prompts/ exist.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import random
import re
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "Qwen/Qwen-VL-Chat"
DEFAULT_REVISION = "main"


@dataclass
class GenerationDiagnostics:
    requested_weight: float
    modified_calls: int
    image_token_count: int
    prompt_sequence_length: int
    image_start: Optional[int]
    image_end: Optional[int]


class QwenAdaptVisController:
    """Runtime state shared by the Qwen transformer and attention patches."""

    def __init__(self, max_layers: int) -> None:
        self.max_layers = int(max_layers)
        self.enabled = False
        self.weight = 1.0
        self.image_mask: Optional[torch.Tensor] = None
        self.prompt_sequence_length = 0
        self.modified_calls = 0

    def begin_generation(self, weight: float) -> None:
        self.weight = float(weight)
        self.enabled = self.weight != 1.0
        self.image_mask = None
        self.prompt_sequence_length = 0
        self.modified_calls = 0

    def capture_image_mask(self, input_ids: torch.LongTensor, image_start_id: int) -> None:
        """
        Qwen replaces hidden_states[a+1:b] by visual embeddings, where a is
        image_start_id and b is image_start_id + 1. Capture those exact slots.
        """
        if input_ids.ndim != 2:
            raise RuntimeError(f"Expected Qwen input_ids [B,L], got {tuple(input_ids.shape)}")

        end_id = int(image_start_id) + 1
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for batch_idx in range(input_ids.shape[0]):
            starts = torch.nonzero(input_ids[batch_idx] == int(image_start_id), as_tuple=False).flatten()
            ends = torch.nonzero(input_ids[batch_idx] == end_id, as_tuple=False).flatten()
            if starts.numel() != ends.numel() or starts.numel() == 0:
                raise RuntimeError(
                    "Could not recover Qwen visual-token span from input_ids: "
                    f"starts={starts.tolist()}, ends={ends.tolist()}"
                )
            for start, end in zip(starts.tolist(), ends.tolist()):
                if end <= start + 1:
                    raise RuntimeError(
                        "Invalid Qwen image span: "
                        f"start={start}, end={end}."
                    )
                mask[batch_idx, start + 1 : end] = True

        self.image_mask = mask
        self.prompt_sequence_length = int(input_ids.shape[-1])

    def finish_generation(self) -> GenerationDiagnostics:
        count = 0
        start: Optional[int] = None
        end: Optional[int] = None
        if self.image_mask is not None and self.image_mask.numel() > 0:
            indices = torch.nonzero(self.image_mask[0].detach().bool().cpu(), as_tuple=False).flatten()
            count = int(indices.numel())
            if count:
                start = int(indices.min())
                end = int(indices.max()) + 1

        diagnostics = GenerationDiagnostics(
            requested_weight=float(self.weight),
            modified_calls=int(self.modified_calls),
            image_token_count=count,
            prompt_sequence_length=int(self.prompt_sequence_length),
            image_start=start,
            image_end=end,
        )
        self.enabled = False
        return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen-VL-Chat AdaptVis + RMSNorm-epsilon ablation."
    )
    parser.add_argument(
        "--dataset",
        default="Controlled_Images_A",
        choices=["Controlled_Images_A", "Controlled_Images_B"],
    )
    parser.add_argument("--option", default="four", choices=["two", "four", "six"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rms-norm-eps", default=1e-6, type=float)
    parser.add_argument("--method", default="adapt_vis", choices=["base", "scaling_vis", "adapt_vis"])
    parser.add_argument("--weight", default=1.0, type=float, help="Fixed multiplier for --method scaling_vis.")
    parser.add_argument("--weight1", default=0.5, type=float)
    parser.add_argument("--weight2", default=1.5, type=float)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument("--max-layers", default=32, type=int)
    parser.add_argument("--max-new-tokens", default=100, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-cache-dir", default="data/qwenvl_chat_eval_images")
    parser.add_argument("--prompt-mode", default="auto", choices=["auto", "raw", "llava"])
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--print-first", default=5, type=int)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_gold(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def is_correct(gold: Any, generation: str) -> bool:
    gold_text = norm_gold(gold)
    generation_text = str(generation).strip()
    if not gold_text:
        return False
    correct = gold_text in generation_text or gold_text.lower() in generation_text.lower()
    if gold_text.lower() == "on" and "front" in generation_text.lower():
        correct = False
    return bool(correct)


def first_step_confidence(output: Any) -> float:
    scores = output.scores if hasattr(output, "scores") else output.get("scores")
    if scores is None or len(scores) == 0:
        return 0.0
    return float(torch.softmax(scores[0].detach().float(), dim=-1)[0].max().cpu())


def load_prompts(dataset: str, option: str) -> Tuple[List[str], List[Any]]:
    path = Path(f"prompts/{dataset}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}. Run from the AdaptVis repository root.")
    prompts: List[str] = []
    answers: List[Any] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            prompts.append(item["question"])
            answers.append(item["answer"])
    return prompts, answers


def extract_images_from_batch(batch: Dict[str, Any]) -> Iterable[Any]:
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def sanitize_prompt_for_qwen(prompt: str, mode: str) -> str:
    text = str(prompt).replace("\r\n", "\n").strip()
    text = re.sub(r"<image>", "", text, flags=re.IGNORECASE)
    text = text.replace("<im_start>", "").replace("<im_end>", " ").replace("<im_patch>", " ")
    if mode in {"auto", "llava"}:
        user_match = re.search(r"\bUSER\s*:\s*", text, flags=re.IGNORECASE)
        assistant_matches = list(re.finditer(r"\bASSISTANT\s*:\s*", text, flags=re.IGNORECASE))
        if user_match is not None:
            start = user_match.end()
            end = assistant_matches[0].start() if assistant_matches else len(text)
            text = text[start:end]
        elif mode == "llava":
            text = re.sub(r"^\s*###\s*(Human|USER)\s*:\s*", "", text, flags=re.IGNORECASE)
            text = re.split(r"\s*###\s*(Assistant|ASSISTANT)\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip()).strip()
    if not text:
        raise ValueError("Qwen query is empty after prompt conversion.")
    return text


def image_to_local_png(image: Any, sid: int, root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sid:05d}.png"
    if not path.exists():
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(path, format="PNG")
    return str(path.resolve())


def find_language_rmsnorms(model: Any) -> List[Tuple[str, nn.Module]]:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise RuntimeError("Expected Qwen-VL remote-code model with .transformer.")
    norms = [(name, module) for name, module in transformer.named_modules() if type(module).__name__ == "RMSNorm" and hasattr(module, "eps")]
    expected_layers = int(getattr(model.config, "num_hidden_layers", 0))
    expected_count = 2 * expected_layers + 1 if expected_layers else None
    if expected_count is not None and len(norms) != expected_count:
        raise RuntimeError(f"Unexpected Qwen RMSNorm count: found {len(norms)}, expected {expected_count}.")
    if not norms:
        raise RuntimeError("No Qwen RMSNorm modules with .eps attribute found.")
    return norms


def set_qwen_rmsnorm_epsilon(model: Any, epsilon: float) -> Tuple[int, List[float]]:
    norms = find_language_rmsnorms(model)
    before = sorted({float(module.eps) for _, module in norms})
    for _, module in norms:
        module.eps = float(epsilon)
    if hasattr(model.config, "layer_norm_epsilon"):
        model.config.layer_norm_epsilon = float(epsilon)
    if hasattr(model.transformer, "config") and hasattr(model.transformer.config, "layer_norm_epsilon"):
        model.transformer.config.layer_norm_epsilon = float(epsilon)
    after = sorted({float(module.eps) for _, module in norms})
    print("Qwen language RMSNorm modules:", len(norms))
    print("Qwen RMSNorm epsilon before override:", before)
    print("Qwen RMSNorm epsilon after override:", after)
    if after != [float(epsilon)]:
        raise RuntimeError(f"Failed to set Qwen RMSNorm eps to {epsilon}; got {after}.")
    return len(norms), before


def install_qwen_image_span_capture(model: Any, controller: QwenAdaptVisController) -> None:
    """Capture Qwen's actual visual-token positions without changing outputs."""
    transformer = model.transformer
    original_forward = transformer.forward
    image_start_id = int(model.config.visual["image_start_id"])

    def wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        past_key_values = kwargs.get("past_key_values", args[1] if len(args) > 1 else None)
        if input_ids is not None and past_key_values is None:
            if torch.any(input_ids == image_start_id):
                controller.capture_image_mask(input_ids, image_start_id)
        return original_forward(*args, **kwargs)

    transformer.forward = types.MethodType(wrapped_forward, transformer)


def install_qwen_adaptvis_attention(model: Any, controller: QwenAdaptVisController) -> None:
    """Patch native QWenAttention._attn only for initial multimodal prefill."""
    layers = getattr(model.transformer, "h", None)
    if layers is None:
        raise RuntimeError("Expected Qwen language layers at model.transformer.h.")

    for layer_index, layer in enumerate(layers):
        attn = getattr(layer, "attn", None)
        if attn is None or not hasattr(attn, "_attn"):
            raise RuntimeError(f"Layer {layer_index} has no native Qwen _attn method.")
        original_attn = attn._attn

        def make_attn(original: Any, idx: int):
            def patched_attn(
                self: Any,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                registered_causal_mask: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                q_len = int(query.shape[-2])
                k_len = int(key.shape[-2])
                mask = controller.image_mask
                should_intervene = (
                    controller.enabled
                    and controller.weight != 1.0
                    and idx < controller.max_layers
                    and mask is not None
                    and mask.ndim == 2
                    and mask.shape[0] in (1, query.shape[0])
                    and controller.prompt_sequence_length == q_len
                    and k_len == q_len
                    and mask.shape[-1] == k_len
                    and q_len > 1
                )
                if not should_intervene:
                    return original(query, key, value, registered_causal_mask, attention_mask, head_mask)

                attn_weights = torch.matmul(query, key.transpose(-1, -2))
                if getattr(self, "scale_attn_weights", True):
                    attn_weights = attn_weights / math.sqrt(float(value.size(-1)))

                image_mask = mask.to(device=attn_weights.device, dtype=torch.bool)
                if image_mask.shape[0] == 1 and query.shape[0] > 1:
                    image_mask = image_mask.expand(query.shape[0], -1)
                if image_mask.shape[-1] != k_len:
                    raise RuntimeError(
                        "Qwen AdaptVis image mask and KV length differ: "
                        f"mask={image_mask.shape[-1]}, kv={k_len}, layer={idx}."
                    )

                for batch_idx in range(query.shape[0]):
                    image_indices = torch.nonzero(image_mask[batch_idx], as_tuple=False).flatten()
                    if image_indices.numel() == 0:
                        continue
                    logits = attn_weights[batch_idx, :, q_len - 1, :]
                    selected = logits.index_select(-1, image_indices)
                    logits.index_copy_(-1, image_indices, selected * float(controller.weight))

                controller.modified_calls += 1
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                attn_weights = attn_weights.type(value.dtype)
                attn_weights = self.attn_dropout(attn_weights)
                if head_mask is not None:
                    attn_weights = attn_weights * head_mask
                attn_output = torch.matmul(attn_weights, value).transpose(1, 2)
                return attn_output, attn_weights

            return types.MethodType(patched_attn, attn)

        attn._attn = make_attn(original_attn, layer_index)


def load_qwen_model(args: argparse.Namespace) -> Tuple[Any, Any, QwenAdaptVisController]:
    print(f"transformers version: {transformers.__version__}")
    print(f"Loading Qwen-VL-Chat from {args.model}@{args.revision}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, cache_dir=args.cache_dir, trust_remote_code=True)
    precision_kw = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        **{precision_kw: True},
    ).eval()
    model.to(args.device)
    checkpoint_eps = getattr(model.config, "layer_norm_epsilon", None)
    print("Checkpoint config.layer_norm_epsilon:", checkpoint_eps)
    print("Requested RMSNorm epsilon:", float(args.rms_norm_eps))
    rms_count, rms_before = set_qwen_rmsnorm_epsilon(model, args.rms_norm_eps)
    controller = QwenAdaptVisController(max_layers=args.max_layers)
    install_qwen_image_span_capture(model, controller)
    install_qwen_adaptvis_attention(model, controller)
    model._qwen_eps_ablation_count = rms_count
    model._qwen_eps_before = rms_before
    model._qwen_checkpoint_eps = checkpoint_eps
    print("Active model classes:")
    print(f"  Qwen-VL: {type(model).__module__}.{type(model).__name__}")
    print(f"  transformer: {type(model.transformer).__module__}.{type(model.transformer).__name__}")
    print(f"  first parameter dtype: {next(model.parameters()).dtype}")
    return model, tokenizer, controller


def qwen_remote_helpers(model: Any) -> Tuple[Any, Any, Any]:
    module = importlib.import_module(type(model).__module__)
    missing = [name for name in ("make_context", "get_stop_words_ids", "decode_tokens") if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Qwen remote module is missing required chat helpers: {missing}")
    return module.make_context, module.get_stop_words_ids, module.decode_tokens


@torch.inference_mode()
def generate_once(
    *,
    model: Any,
    tokenizer: Any,
    query: str,
    system: str,
    controller: QwenAdaptVisController,
    weight: float,
    max_new_tokens: int,
) -> Tuple[str, float, GenerationDiagnostics]:
    make_context, get_stop_words_ids, decode_tokens = qwen_remote_helpers(model)
    generation_config = copy.deepcopy(model.generation_config)
    chat_format = generation_config.chat_format
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=[],
        system=system,
        max_window_size=generation_config.max_window_size,
        chat_format=chat_format,
    )
    input_ids = torch.tensor([context_tokens], device=model.device)
    stop_words_ids = copy.deepcopy(get_stop_words_ids(chat_format, tokenizer))

    controller.begin_generation(weight)
    output = model.generate(
        input_ids,
        stop_words_ids=stop_words_ids,
        return_dict_in_generate=True,
        output_scores=True,
        generation_config=generation_config,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    diagnostics = controller.finish_generation()
    sequences = output.sequences if hasattr(output, "sequences") else output["sequences"]
    generation = decode_tokens(
        sequences[0],
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=len(context_tokens),
        chat_format=chat_format,
        verbose=False,
        errors="replace",
    )
    return str(generation).strip(), first_step_confidence(output), diagnostics


@torch.inference_mode()
def evaluate(args: argparse.Namespace, model: Any, tokenizer: Any, controller: QwenAdaptVisController) -> Dict[str, Any]:
    prompts, answers = load_prompts(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=repository_default_collate)
    total_available = min(len(prompts), len(dataset))
    total_target = min(total_available, args.limit) if args.limit is not None else total_available
    image_cache_root = Path(args.image_cache_dir) / args.dataset

    records: List[Dict[str, Any]] = []
    correct_count = 0
    sample_index = 0
    progress = tqdm(total=total_target, desc=f"Qwen-VL-Chat {args.method}")

    for batch in loader:
        for image in extract_images_from_batch(batch):
            if sample_index >= total_target:
                break
            raw_prompt = prompts[sample_index]
            query_text = sanitize_prompt_for_qwen(raw_prompt, args.prompt_mode)
            image_path = image_to_local_png(image, sample_index, image_cache_root)
            query = tokenizer.from_list_format([{"image": image_path}, {"text": query_text}])
            gold = norm_gold(answers[sample_index])

            probe_generation: Optional[str] = None
            probe_confidence: Optional[float] = None
            rounded_confidence: Optional[float] = None
            probe_diag: Optional[GenerationDiagnostics] = None

            if args.method == "base":
                selected_weight = 1.0
            elif args.method == "scaling_vis":
                selected_weight = float(args.weight)
            elif args.method == "adapt_vis":
                probe_generation, probe_confidence, probe_diag = generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    system=args.system,
                    controller=controller,
                    weight=1.0,
                    max_new_tokens=args.max_new_tokens,
                )
                rounded_confidence = float(np.round(probe_confidence, 2))
                selected_weight = float(args.weight1) if rounded_confidence < float(args.threshold) else float(args.weight2)
            else:
                raise ValueError(f"Unsupported method: {args.method}")

            generation, final_confidence, final_diag = generate_once(
                model=model,
                tokenizer=tokenizer,
                query=query,
                system=args.system,
                controller=controller,
                weight=selected_weight,
                max_new_tokens=args.max_new_tokens,
            )
            correct = is_correct(gold, generation)
            correct_count += int(correct)
            if args.method != "adapt_vis":
                probe_confidence = final_confidence
                rounded_confidence = float(np.round(final_confidence, 2))

            record = {
                "sid": sample_index,
                "prompt": raw_prompt,
                "qwen_query_text": query_text,
                "qwen_query": query,
                "image_path": image_path,
                "gold": gold,
                "method": args.method,
                "selected_weight": selected_weight,
                "generation": generation,
                "correct": bool(correct),
                "first_step_confidence": final_confidence,
                "probe_generation": probe_generation,
                "probe_confidence": probe_confidence,
                "rounded_probe_confidence": rounded_confidence,
                "probe_diagnostics": asdict(probe_diag) if probe_diag is not None else None,
                "final_diagnostics": asdict(final_diag),
            }
            records.append(record)

            if sample_index < args.print_first:
                print("\n" + "-" * 100)
                print(f"[SID {sample_index}] gold={gold!r}")
                print(f"qwen_query_text={query_text!r}")
                if args.method == "adapt_vis":
                    print(f"probe={probe_generation!r} conf={probe_confidence:.6f} rounded={rounded_confidence:.2f}")
                print(f"selected_weight={selected_weight} pred={generation!r} correct={correct}")
                print(
                    "image tokens="
                    f"{final_diag.image_token_count}, range=[{final_diag.image_start},{final_diag.image_end}), "
                    f"prompt_len={final_diag.prompt_sequence_length}, modified_calls={final_diag.modified_calls}"
                )
                if selected_weight != 1.0 and final_diag.modified_calls != min(int(args.max_layers), int(getattr(model.config, "num_hidden_layers", 0))):
                    print("WARNING: AdaptVis did not modify the expected number of attention layers.")

            sample_index += 1
            progress.update(1)
        if sample_index >= total_target:
            break

    progress.close()
    accuracy = correct_count / max(sample_index, 1)
    low_branch_count = sum(int(record["selected_weight"] == float(args.weight1)) for record in records) if args.method == "adapt_vis" else None
    high_branch_count = sum(int(record["selected_weight"] == float(args.weight2)) for record in records) if args.method == "adapt_vis" else None
    summary = {
        "model": args.model,
        "revision": args.revision,
        "resolved_config_commit": getattr(model.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "dataset": args.dataset,
        "option": args.option,
        "implementation": "Qwen-VL-Chat native QWenAttention pre-softmax AdaptVis patch",
        "epsilon_scope": "language transformer RMSNorm modules only",
        "checkpoint_rms_norm_eps": getattr(model, "_qwen_checkpoint_eps", None),
        "requested_rms_norm_eps": float(args.rms_norm_eps),
        "rmsnorm_module_count": getattr(model, "_qwen_eps_ablation_count", None),
        "rmsnorm_eps_before_override": getattr(model, "_qwen_eps_before", None),
        "active_rms_norm_eps": float(args.rms_norm_eps),
        "method": args.method,
        "weight": float(args.weight),
        "weight1": float(args.weight1),
        "weight2": float(args.weight2),
        "threshold": float(args.threshold),
        "max_layers": int(args.max_layers),
        "system": args.system,
        "prompt_mode": args.prompt_mode,
        "dtype_argument": args.dtype,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": False,
        "num_samples": sample_index,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "branches": {"weight1": low_branch_count, "weight2": high_branch_count},
        "records": records,
    }
    print("\n" + "=" * 100)
    print(
        f"RESULT: {correct_count}/{sample_index} accuracy={accuracy:.6f} "
        f"rms_norm_eps={args.rms_norm_eps:g} method={args.method}"
    )
    if args.method == "adapt_vis":
        print(f"branches: weight1({args.weight1})={low_branch_count}, weight2({args.weight2})={high_branch_count}")
    print("=" * 100)
    return summary


def default_output_path(args: argparse.Namespace) -> Path:
    eps = f"{args.rms_norm_eps:.0e}".replace("-", "m")
    base = f"qwenvl_chat_{args.dataset}_{args.method}_eps_{eps}"
    if args.method == "adapt_vis":
        base += f"_w1_{args.weight1:g}_w2_{args.weight2:g}_thr_{args.threshold:g}"
    elif args.method == "scaling_vis":
        base += f"_w_{args.weight:g}"
    return Path("output") / f"{base}.json"


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    model, tokenizer, controller = load_qwen_model(args)
    summary = evaluate(args, model, tokenizer, controller)
    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
