#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified epsilon x ScalingVis evaluation for:
  qwen2vl    : Qwen/Qwen2-VL-7B-Instruct
  qwen25vl   : Qwen/Qwen2.5-VL-7B-Instruct
  internvl25 : OpenGVLab/InternVL2_5-8B

Benchmarks:
  vsr : VSR yes/no test JSONL
  gqa : GQA testdev-balanced short-answer JSON

This runner is intentionally pinned to the attention APIs in:
  torch==2.4.1, transformers==4.49.0

Intervention definition
-----------------------
During the INITIAL multimodal prefill only, in every selected decoder layer,
multiply pre-softmax attention logits from the final prompt query to ALL visual
token keys by `--scalvis-weight`. No vision-tower normalisation is altered.

This is a one-combination-per-process runner. Use the supplied launcher to
schedule:
  2 datasets x 3 models x 4 eps values x 2 ScalingVis weights = 48 jobs.

Examples
--------
python run_eps_scalvis_vsr_gqa.py \
  --backend qwen2vl --dataset vsr \
  --rms-norm-eps 1e-6 --scalvis-weight 0.5 \
  --output-dir output/eps_scalvis_grid

python run_eps_scalvis_vsr_gqa.py \
  --backend internvl25 --dataset gqa \
  --rms-norm-eps 1e-5 --scalvis-weight 1.0 \
  --output-dir output/eps_scalvis_grid
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


DEFAULT_MODELS: Dict[str, str] = {
    "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl25": "OpenGVLab/InternVL2_5-8B",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_VSR_ANN = "data/benchmarks/vsr/repo/data/splits/zeroshot/test.jsonl"
DEFAULT_VSR_IMAGES = "data/benchmarks/vsr/images"
DEFAULT_GQA_QUESTIONS = "data/benchmarks/gqa/questions1.2/testdev_balanced_questions.json"
DEFAULT_GQA_IMAGES = "data/benchmarks/gqa/images"


@dataclass
class Sample:
    sid: str
    image_path: str
    prompt: str
    gold: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-model epsilon x ScalingVis evaluation on VSR or GQA."
    )
    p.add_argument("--backend", choices=sorted(DEFAULT_MODELS), required=True)
    p.add_argument("--dataset", choices=["vsr", "gqa"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    p.add_argument("--rms-norm-eps", required=True, type=float)
    p.add_argument(
        "--scalvis-weight",
        required=True,
        type=float,
        choices=[0.5, 1.0],
    )
    p.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Default: all language-decoder layers.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Default: 4 for VSR; 8 for GQA.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--print-first", type=int, default=3)
    p.add_argument("--skip-missing-images", action="store_true")

    p.add_argument("--vsr-ann", default=DEFAULT_VSR_ANN)
    p.add_argument("--vsr-image-root", default=DEFAULT_VSR_IMAGES)
    p.add_argument("--gqa-questions", default=DEFAULT_GQA_QUESTIONS)
    p.add_argument("--gqa-image-root", default=DEFAULT_GQA_IMAGES)

    p.add_argument(
        "--internvl-max-num",
        type=int,
        default=12,
        help="Maximum InternVL dynamic 448x448 tiles.",
    )
    p.add_argument(
        "--internvl-no-thumbnail",
        action="store_true",
        help="Disable InternVL thumbnail tile.",
    )
    p.add_argument(
        "--output-dir",
        default="output/eps_scalvis_grid",
        help="Directory containing one JSONL and one summary JSON per job.",
    )
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_vsr(value: Any) -> str:
    value = norm_space(value).lower()
    match = re.search(r"\b(yes|no)\b", value)
    if match:
        return match.group(1)
    aliases = {
        "true": "yes",
        "1": "yes",
        "correct": "yes",
        "false": "no",
        "0": "no",
        "incorrect": "no",
    }
    return aliases.get(value, "")


_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_gqa(value: Any) -> str:
    text = norm_space(value).lower().replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    words = [_NUMBER_WORDS.get(word, word) for word in words]
    return " ".join(words)


def resolve_vsr_gold(label: Any) -> str:
    if isinstance(label, bool):
        return "yes" if label else "no"
    if isinstance(label, (int, float)):
        return "yes" if int(label) else "no"
    return normalize_vsr(label)


def find_gqa_image(image_root: Path, image_id: str, index: Dict[str, str]) -> Path:
    candidates = [
        image_root / f"{image_id}.jpg",
        image_root / f"{image_id}.png",
        image_root / image_id,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    indexed = index.get(str(image_id))
    if indexed:
        return Path(indexed)
    return image_root / f"{image_id}.jpg"


def load_vsr(args: argparse.Namespace) -> List[Sample]:
    ann = Path(args.vsr_ann)
    root = Path(args.vsr_image_root)
    if not ann.is_file():
        raise FileNotFoundError(f"VSR annotation JSONL not found: {ann}")
    samples: List[Sample] = []
    with ann.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            caption = norm_space(item["caption"])
            image_name = Path(str(item["image_link"]).split("?", 1)[0]).name
            gold = resolve_vsr_gold(item["label"])
            if gold not in {"yes", "no"}:
                raise ValueError(f"Unsupported VSR label at {idx}: {item['label']!r}")
            prompt = (
                "Does the statement accurately describe the image? "
                "Reply with exactly one word: yes or no. Do not explain.\n"
                f"Statement: {caption}\n"
                "Answer:"
            )
            samples.append(
                Sample(
                    sid=str(idx),
                    image_path=str(root / image_name),
                    prompt=prompt,
                    gold=gold,
                    metadata={
                        "caption": caption,
                        "relation": item.get("relation"),
                        "image_link": item["image_link"],
                    },
                )
            )
    return samples


def iter_gqa_records(data: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(data, dict):
        for sid, record in data.items():
            if not isinstance(record, dict):
                raise TypeError(f"GQA record {sid!r} is not a JSON object.")
            yield str(sid), record
        return
    if isinstance(data, list):
        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                raise TypeError(f"GQA record #{idx} is not a JSON object.")
            sid = record.get("questionId", record.get("question_id", idx))
            yield str(sid), record
        return
    raise TypeError(f"Unsupported GQA JSON top-level type: {type(data).__name__}")


def load_gqa(args: argparse.Namespace) -> List[Sample]:
    questions_path = Path(args.gqa_questions)
    image_root = Path(args.gqa_image_root)
    if not questions_path.is_file():
        raise FileNotFoundError(f"GQA questions JSON not found: {questions_path}")
    with questions_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_index: Dict[str, str] = {}
    if image_root.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in image_root.rglob(ext):
                image_index.setdefault(path.stem, str(path))

    samples: List[Sample] = []
    for sid, item in iter_gqa_records(data):
        image_id = item.get("imageId", item.get("image_id"))
        question = item.get("question")
        answer = item.get("answer")
        if image_id is None or question is None or answer is None:
            raise KeyError(
                f"GQA {sid}: expected imageId/question/answer; got keys={sorted(item.keys())}"
            )
        prompt = (
            "Answer the question about the image with the shortest answer only. "
            "Do not explain.\n"
            f"Question: {norm_space(question)}\n"
            "Answer:"
        )
        samples.append(
            Sample(
                sid=sid,
                image_path=str(find_gqa_image(image_root, str(image_id), image_index)),
                prompt=prompt,
                gold=norm_space(answer),
                metadata={
                    "question": norm_space(question),
                    "image_id": str(image_id),
                    "types": item.get("types"),
                    "semantic": item.get("semantic"),
                },
            )
        )
    return samples


def load_samples(args: argparse.Namespace) -> List[Sample]:
    samples = load_vsr(args) if args.dataset == "vsr" else load_gqa(args)
    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    samples = samples[args.start :]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("No samples selected.")
    return samples


class ScalingVisController:
    """Per-generation state shared by patched attention implementations."""

    def __init__(self, num_layers: int, max_layers: Optional[int]) -> None:
        self.num_layers = int(num_layers)
        self.max_layers = int(max_layers) if max_layers is not None else int(num_layers)
        if self.max_layers <= 0 or self.max_layers > self.num_layers:
            raise ValueError(
                f"max_layers must be in [1, {self.num_layers}], got {self.max_layers}."
            )
        self.active = False
        self.weight = 1.0
        self.image_mask: Optional[torch.Tensor] = None
        self.prompt_length = 0
        self.modified_calls = 0

    def begin(self, input_ids: torch.Tensor, image_token_id: int, weight: float) -> None:
        if input_ids.ndim != 2:
            raise RuntimeError(f"Expected input_ids [B,L], got {tuple(input_ids.shape)}")
        image_mask = input_ids.eq(int(image_token_id))
        if not bool(image_mask.any().item()):
            raise RuntimeError(
                "No image tokens were found in initial input_ids. "
                "The processor/template is not producing a visual prompt."
            )
        self.active = True
        self.weight = float(weight)
        self.image_mask = image_mask.detach()
        self.prompt_length = int(input_ids.shape[-1])
        self.modified_calls = 0

    def end(self) -> int:
        calls = int(self.modified_calls)
        self.active = False
        self.image_mask = None
        self.prompt_length = 0
        return calls

    def should_apply(self, layer_index: int, q_len: int, kv_len: int, bsz: int) -> bool:
        mask = self.image_mask
        return bool(
            self.active
            and self.weight != 1.0
            and layer_index < self.max_layers
            and mask is not None
            and q_len == self.prompt_length
            and kv_len == self.prompt_length
            and mask.ndim == 2
            and mask.shape[0] in {1, bsz}
            and mask.shape[1] == kv_len
        )

    def apply(self, attn_logits: torch.Tensor, layer_index: int) -> None:
        """In-place raw logit scaling at the final prefill query only."""
        bsz, _, q_len, kv_len = attn_logits.shape
        if not self.should_apply(layer_index, q_len, kv_len, bsz):
            return
        assert self.image_mask is not None
        mask = self.image_mask
        if mask.shape[0] == 1 and bsz > 1:
            mask = mask.expand(bsz, -1)
        mask = mask.to(device=attn_logits.device, dtype=torch.bool)
        for batch_idx in range(bsz):
            key_mask = mask[batch_idx]
            if not bool(key_mask.any().item()):
                raise RuntimeError("Visual key mask is empty.")
            attn_logits[batch_idx, :, q_len - 1, key_mask] *= self.weight
        self.modified_calls += 1


def locate_language_model(model: Any, backend: str) -> Any:
    paths = (
        ("model", "language_model"),
        ("language_model",),
        ("model",),
    )
    for path in paths:
        node = model
        ok = True
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                ok = False
                break
        if ok and hasattr(node, "layers"):
            return node
    raise RuntimeError(f"Could not locate language decoder for backend={backend}.")


def set_language_rmsnorm_eps(model: Any, backend: str, epsilon: float) -> Dict[str, Any]:
    if epsilon <= 0:
        raise ValueError("--rms-norm-eps must be positive.")
    language_model = locate_language_model(model, backend)
    norms = [
        (name, module)
        for name, module in language_model.named_modules()
        if "RMSNorm" in type(module).__name__ and hasattr(module, "variance_epsilon")
    ]
    if not norms:
        raise RuntimeError("No language-decoder RMSNorm modules with variance_epsilon found.")
    before = sorted({float(module.variance_epsilon) for _, module in norms})
    for _, module in norms:
        module.variance_epsilon = float(epsilon)
    after = sorted({float(module.variance_epsilon) for _, module in norms})
    if after != [float(epsilon)]:
        raise RuntimeError(f"Failed RMSNorm epsilon override: {after}")

    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(language_model, "config", None),
    ):
        if config is not None and hasattr(config, "rms_norm_eps"):
            config.rms_norm_eps = float(epsilon)

    return {
        "count": len(norms),
        "classes": sorted({type(module).__name__ for _, module in norms}),
        "before": before,
        "after": after,
    }


def load_qwen(args: argparse.Namespace, dtype: torch.dtype) -> Tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        if args.backend == "qwen25vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen2-VL / Qwen2.5-VL requires transformers==4.49.0 and qwen-vl-utils."
        ) from exc

    model_id = args.model or DEFAULT_MODELS[args.backend]
    model_cls = (
        Qwen2VLForConditionalGeneration
        if args.backend == "qwen2vl"
        else Qwen2_5_VLForConditionalGeneration
    )
    kwargs: Dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    pkwargs: Dict[str, Any] = {"cache_dir": args.cache_dir}
    if args.revision:
        kwargs["revision"] = args.revision
        pkwargs["revision"] = args.revision

    processor = AutoProcessor.from_pretrained(model_id, **pkwargs)
    model = model_cls.from_pretrained(model_id, **kwargs).eval().to(args.device)
    language_model = locate_language_model(model, args.backend)
    if len(getattr(language_model, "layers", [])) == 0:
        raise RuntimeError("Qwen language decoder has no layers.")
    return model, processor


def install_qwen2_scalvis(model: Any, controller: ScalingVisController) -> None:
    """
    Patch Qwen2-VL's own eager decoder attention.

    Qwen2-VL does NOT route its language decoder through
    transformers.models.qwen2.modeling_qwen2.eager_attention_forward. Its
    Qwen2VLAttention class is implemented in modeling_qwen2_vl.py and has a
    separate multimodal-RoPE forward path. Therefore patch each self_attn
    module directly at the raw, pre-softmax attention logits.
    """
    from transformers.models.qwen2_vl import modeling_qwen2_vl as qwen2vl_mod

    language_model = locate_language_model(model, "qwen2vl")
    for layer_idx, layer in enumerate(language_model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise RuntimeError(f"Qwen2-VL layer {layer_idx} has no self_attn.")
        if type(attn).__name__ != "Qwen2VLAttention":
            raise RuntimeError(
                f"Expected eager Qwen2VLAttention, got {type(attn).__name__}. "
                "Load with attn_implementation='eager'."
            )

        original_forward = attn.forward

        def make_forward(attn_module: Any, native_forward: Any, idx: int):
            def patched_forward(
                self: Any,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Any] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
                bsz, q_len, _ = hidden_states.size()

                # Intercept only the initial multimodal prefill. Cached decode
                # steps remain checkpoint-native.
                if not controller.should_apply(idx, q_len, q_len, bsz):
                    return native_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                if position_embeddings is None:
                    raise RuntimeError(
                        "Qwen2-VL ScalingVis prefill patch requires position_embeddings."
                    )

                query_states = attn_module.q_proj(hidden_states)
                key_states = attn_module.k_proj(hidden_states)
                value_states = attn_module.v_proj(hidden_states)

                query_states = query_states.view(
                    bsz, q_len, -1, attn_module.head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, -1, attn_module.head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, -1, attn_module.head_dim
                ).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = qwen2vl_mod.apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    attn_module.rope_scaling["mrope_section"],
                )

                # In generation, HF creates DynamicCache before initial prefill.
                if past_key_value is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key_states, value_states = past_key_value.update(
                        key_states,
                        value_states,
                        attn_module.layer_idx,
                        cache_kwargs,
                    )

                key_states = qwen2vl_mod.repeat_kv(
                    key_states,
                    attn_module.num_key_value_groups,
                )
                value_states = qwen2vl_mod.repeat_kv(
                    value_states,
                    attn_module.num_key_value_groups,
                )

                attn_logits = torch.matmul(
                    query_states,
                    key_states.transpose(2, 3),
                ) / math.sqrt(attn_module.head_dim)

                # Scale only [all heads, final prompt query, image-token keys],
                # before causal mask addition and softmax.
                controller.apply(attn_logits, idx)

                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_logits = attn_logits + causal_mask

                if query_states.dtype == torch.float16:
                    attn_logits = torch.where(
                        torch.isinf(attn_logits),
                        torch.zeros_like(attn_logits),
                        attn_logits,
                    )

                attn_weights = nn.functional.softmax(
                    attn_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query_states.dtype)
                attn_weights = nn.functional.dropout(
                    attn_weights,
                    p=attn_module.attention_dropout,
                    training=attn_module.training,
                )
                attn_output = torch.matmul(attn_weights, value_states)

                expected = (
                    bsz,
                    attn_module.num_heads,
                    q_len,
                    attn_module.head_dim,
                )
                if tuple(attn_output.size()) != expected:
                    raise RuntimeError(
                        f"Unexpected Qwen2-VL attention output: "
                        f"{tuple(attn_output.size())}; expected {expected}."
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_module.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None
                return attn_output, attn_weights, past_key_value

            return patched_forward

        attn.forward = types.MethodType(
            make_forward(attn, original_forward, layer_idx),
            attn,
        )


def install_qwen25_scalvis(model: Any, controller: ScalingVisController) -> None:
    """Patch eager Qwen2.5-VL attention modules exactly at raw logits."""
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen25_mod

    language_model = locate_language_model(model, "qwen25vl")
    for layer_idx, layer in enumerate(language_model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise RuntimeError(f"Layer {layer_idx} has no self_attn.")
        if type(attn).__name__ != "Qwen2_5_VLAttention":
            raise RuntimeError(
                f"Expected eager Qwen2_5_VLAttention, got {type(attn).__name__}. "
                "Load with attn_implementation='eager'."
            )
        original = attn.forward

        def make_forward(attn_module: Any, original_forward: Any, idx: int):
            def patched_forward(
                self: Any,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Any] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
                bsz, q_len, _ = hidden_states.size()
                if not controller.should_apply(idx, q_len, q_len, bsz):
                    return original_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                if position_embeddings is None:
                    raise RuntimeError("Qwen2.5-VL prefill patch requires position_embeddings.")

                query_states = attn_module.q_proj(hidden_states)
                key_states = attn_module.k_proj(hidden_states)
                value_states = attn_module.v_proj(hidden_states)
                query_states = query_states.view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = qwen25_mod.apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    attn_module.rope_scaling["mrope_section"],
                )
                if past_key_value is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key_states, value_states = past_key_value.update(
                        key_states,
                        value_states,
                        attn_module.layer_idx,
                        cache_kwargs,
                    )

                key_states = qwen25_mod.repeat_kv(
                    key_states,
                    attn_module.num_key_value_groups,
                )
                value_states = qwen25_mod.repeat_kv(
                    value_states,
                    attn_module.num_key_value_groups,
                )
                attn_logits = torch.matmul(query_states, key_states.transpose(2, 3))
                attn_logits = attn_logits / math.sqrt(attn_module.head_dim)
                controller.apply(attn_logits, idx)

                if attention_mask is not None:
                    attn_logits = attn_logits + attention_mask[:, :, :, : key_states.shape[-2]]
                if query_states.dtype == torch.float16:
                    attn_logits = torch.where(
                        torch.isinf(attn_logits),
                        torch.zeros_like(attn_logits),
                        attn_logits,
                    )

                attn_weights = nn.functional.softmax(
                    attn_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query_states.dtype)
                attn_weights = nn.functional.dropout(
                    attn_weights,
                    p=attn_module.attention_dropout,
                    training=attn_module.training,
                )
                attn_output = torch.matmul(attn_weights, value_states)
                expected = (bsz, attn_module.num_heads, q_len, attn_module.head_dim)
                if tuple(attn_output.size()) != expected:
                    raise RuntimeError(
                        f"Unexpected Qwen2.5 attention output: {tuple(attn_output.size())}; "
                        f"expected {expected}."
                    )
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_module.o_proj(attn_output)
                if not output_attentions:
                    attn_weights = None
                return attn_output, attn_weights, past_key_value

            return patched_forward

        attn.forward = types.MethodType(make_forward(attn, original, layer_idx), attn)


def qwen_inputs(
    processor: Any,
    image_path: Path,
    prompt: str,
    device: str,
) -> Any:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError(
            "Install qwen-vl-utils==0.0.8 for Qwen2-VL/Qwen2.5-VL input processing."
        ) from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


@torch.inference_mode()
def generate_qwen(
    model: Any,
    processor: Any,
    controller: ScalingVisController,
    sample: Sample,
    weight: float,
    max_new_tokens: int,
    backend: str,
    device: str,
) -> Tuple[str, Dict[str, Any]]:
    inputs = qwen_inputs(processor, Path(sample.image_path), sample.prompt, device)
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError("Qwen config has no image_token_id.")
    controller.begin(inputs.input_ids, int(image_token_id), weight)
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=int(max_new_tokens),
    )
    modified_calls = controller.end()
    if weight != 1.0 and modified_calls <= 0:
        raise RuntimeError(
            "ScalingVis weight != 1, but no Qwen attention call was modified. "
            "Refusing to save an invalid experiment."
        )

    prompt_len = int(inputs.input_ids.shape[1])
    answer = processor.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer, {
        "prompt_token_count": prompt_len,
        "image_token_count": int((inputs.input_ids == int(image_token_id)).sum().item()),
        "modified_calls": modified_calls,
        "max_layers": controller.max_layers,
    }


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def internvl_dynamic_preprocess(
    image: Image.Image,
    image_size: int,
    max_num: int,
    use_thumbnail: bool,
) -> List[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda pair: pair[0] * pair[1],
    )
    ratio_w, ratio_h = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width,
        height,
        image_size,
    )
    target_w, target_h = image_size * ratio_w, image_size * ratio_h
    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    tiles = [
        resized.crop(
            (
                (idx % ratio_w) * image_size,
                (idx // ratio_w) * image_size,
                ((idx % ratio_w) + 1) * image_size,
                ((idx // ratio_w) + 1) * image_size,
            )
        )
        for idx in range(ratio_w * ratio_h)
    ]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


def internvl_image_tensor(image: Image.Image, size: int = 448) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def load_internvl(args: argparse.Namespace, dtype: torch.dtype) -> Tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("InternVL requires transformers==4.49.0.") from exc

    model_id = args.model or DEFAULT_MODELS["internvl25"]
    kwargs: Dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_flash_attn": False,
    }
    tkwargs: Dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
        "use_fast": False,
    }
    if args.revision:
        kwargs["revision"] = args.revision
        tkwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tkwargs)
    model = AutoModel.from_pretrained(model_id, **kwargs).eval().to(args.device)
    return model, tokenizer


def internvl_layers(model: Any) -> Sequence[Any]:
    try:
        layers = model.language_model.model.layers
    except AttributeError as exc:
        raise RuntimeError(
            "Unexpected InternVL decoder layout; expected model.language_model.model.layers."
        ) from exc
    if not layers:
        raise RuntimeError("InternVL decoder layers not found.")
    return layers


def install_internvl_generate_capture(model: Any, controller: ScalingVisController) -> None:
    original_generate = model.generate

    def wrapped_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        if controller.active:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                raise RuntimeError("InternVL generate lacks initial input_ids.")
            image_id = getattr(model, "img_context_token_id", None)
            if image_id is None:
                raise RuntimeError("InternVL img_context_token_id was not initialized.")
            controller.begin(input_ids, int(image_id), controller.weight)
        return original_generate(*args, **kwargs)

    model.generate = types.MethodType(wrapped_generate, model)


def _internvl_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _internvl_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (q * cos) + (_internvl_rotate_half(q) * sin), (
        k * cos
    ) + (_internvl_rotate_half(k) * sin)


def _internvl_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, heads, n_rep, seq_len, head_dim)
        .reshape(batch, heads * n_rep, seq_len, head_dim)
    )


def install_internvl_scalvis(model: Any, controller: ScalingVisController) -> None:
    """Copy eager InternLM2 attention only for the exact initial prefill intervention."""
    for layer_idx, layer in enumerate(internvl_layers(model)):
        attn = getattr(layer, "attention", None)
        if attn is None:
            raise RuntimeError(f"InternVL layer {layer_idx} has no .attention.")
        original = attn.forward
        needed = (
            "wqkv",
            "wo",
            "rotary_emb",
            "num_heads",
            "num_key_value_heads",
            "num_key_value_groups",
            "head_dim",
            "hidden_size",
        )
        missing = [key for key in needed if not hasattr(attn, key)]
        if missing:
            raise RuntimeError(
                f"InternVL attention is not eager-compatible; missing {missing}. "
                "Use use_flash_attn=False."
            )

        def make_forward(attn_module: Any, original_forward: Any, idx: int):
            def patched_forward(
                self: Any,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs: Any,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
                bsz, q_len, _ = hidden_states.size()
                # The native InternVL prefill has no tuple cache yet.
                if (
                    not controller.should_apply(idx, q_len, q_len, bsz)
                    or past_key_value is not None
                ):
                    return original_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )
                if position_ids is None:
                    raise RuntimeError("InternVL ScalingVis patch requires position_ids.")

                qkv = attn_module.wqkv(hidden_states)
                group_size = 2 + int(attn_module.num_key_value_groups)
                qkv = qkv.view(
                    bsz,
                    q_len,
                    int(attn_module.num_key_value_heads),
                    group_size,
                    int(attn_module.head_dim),
                )
                q = qkv[..., : int(attn_module.num_key_value_groups), :]
                q = q.reshape(
                    bsz,
                    q_len,
                    int(attn_module.num_heads),
                    int(attn_module.head_dim),
                ).transpose(1, 2)
                k = qkv[..., -2, :].transpose(1, 2)
                v = qkv[..., -1, :].transpose(1, 2)

                cos, sin = attn_module.rotary_emb(v, seq_len=k.shape[-2])
                q, k = _internvl_apply_rope(q, k, cos, sin, position_ids)
                next_cache = (k, v) if use_cache else None
                k = _internvl_repeat_kv(k, int(attn_module.num_key_value_groups))
                v = _internvl_repeat_kv(v, int(attn_module.num_key_value_groups))

                logits = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(int(attn_module.head_dim))
                controller.apply(logits, idx)
                if attention_mask is not None:
                    logits = logits + attention_mask
                probs = nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
                output = torch.matmul(probs, v)
                output = output.transpose(1, 2).contiguous().reshape(
                    bsz,
                    q_len,
                    int(attn_module.hidden_size),
                )
                output = attn_module.wo(output)
                if not output_attentions:
                    probs = None
                return output, probs, next_cache

            return patched_forward

        attn.forward = types.MethodType(make_forward(attn, original, layer_idx), attn)


@torch.inference_mode()
def generate_internvl(
    model: Any,
    tokenizer: Any,
    controller: ScalingVisController,
    sample: Sample,
    weight: float,
    max_new_tokens: int,
    dtype: torch.dtype,
    device: str,
    max_num: int,
    use_thumbnail: bool,
) -> Tuple[str, Dict[str, Any]]:
    image = Image.open(sample.image_path).convert("RGB")
    tiles = internvl_dynamic_preprocess(
        image,
        image_size=448,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    pixels = torch.stack([internvl_image_tensor(tile) for tile in tiles]).to(
        device=device,
        dtype=dtype,
    )
    controller.active = weight != 1.0
    controller.weight = float(weight)
    controller.modified_calls = 0
    controller.image_mask = None
    controller.prompt_length = 0

    answer = model.chat(
        tokenizer,
        pixels,
        "<image>\n" + sample.prompt,
        generation_config={
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        },
        history=None,
        return_history=False,
    )
    if isinstance(answer, tuple):
        answer = answer[0]
    modified_calls = controller.end()
    if weight != 1.0 and modified_calls <= 0:
        raise RuntimeError(
            "ScalingVis weight != 1, but no InternVL attention call was modified."
        )
    return str(answer).strip(), {
        "num_image_tiles": len(tiles),
        "pixel_values_shape": list(pixels.shape),
        "modified_calls": modified_calls,
        "max_layers": controller.max_layers,
    }


def evaluate_prediction(dataset: str, prediction: str, gold: str) -> Tuple[bool, str, str]:
    if dataset == "vsr":
        pred = normalize_vsr(prediction)
        ref = normalize_vsr(gold)
    else:
        pred = normalize_gqa(prediction)
        ref = normalize_gqa(gold)
    return bool(ref) and pred == ref, pred, ref


def slug_float(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def run_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    model_label = args.model.split("/")[-1] if args.model else DEFAULT_MODELS[args.backend].split("/")[-1]
    tag = (
        f"{args.backend}__{model_label}__{args.dataset}"
        f"__eps{slug_float(args.rms_norm_eps)}"
        f"__w{str(args.scalvis_weight).replace('.', 'p')}"
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{tag}.jsonl", root / f"{tag}.summary.json"


def read_resume_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.is_file():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["sid"]))
    return ids


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if args.rms_norm_eps <= 0:
        raise ValueError("--rms-norm-eps must be positive.")

    dtype = dtype_from_name(args.dtype)
    samples = load_samples(args)
    max_tokens = args.max_new_tokens or (4 if args.dataset == "vsr" else 8)
    records_path, summary_path = run_paths(args)
    completed = read_resume_ids(records_path) if args.resume else set()

    model_id = args.model or DEFAULT_MODELS[args.backend]
    print(f"Loading {args.backend}: {model_id}")
    if args.backend in {"qwen2vl", "qwen25vl"}:
        model, processor = load_qwen(args, dtype)
        language_model = locate_language_model(model, args.backend)
        controller = ScalingVisController(
            num_layers=len(language_model.layers),
            max_layers=args.max_layers,
        )
        if args.backend == "qwen2vl":
            install_qwen2_scalvis(model, controller)
        else:
            install_qwen25_scalvis(model, controller)
        runner = "qwen"
    else:
        model, tokenizer = load_internvl(args, dtype)
        controller = ScalingVisController(
            num_layers=len(internvl_layers(model)),
            max_layers=args.max_layers,
        )
        install_internvl_generate_capture(model, controller)
        install_internvl_scalvis(model, controller)
        runner = "internvl"

    eps_info = set_language_rmsnorm_eps(model, args.backend, args.rms_norm_eps)
    print(
        f"Language RMSNorm eps: {eps_info['before']} -> {eps_info['after']} "
        f"({eps_info['count']} modules)"
    )
    print(
        f"ScalingVis: weight={args.scalvis_weight}, "
        f"layers=[0,{controller.max_layers})/{controller.num_layers}"
    )

    existing_correct = 0
    evaluated = 0
    if args.resume and records_path.is_file():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    evaluated += 1
                    existing_correct += int(bool(json.loads(line).get("correct")))
    num_correct = existing_correct
    num_new = 0
    missing = 0

    mode = "a" if args.resume else "w"
    with records_path.open(mode, encoding="utf-8") as output:
        bar = tqdm(samples, desc=f"{args.backend}/{args.dataset}/eps={args.rms_norm_eps:g}/w={args.scalvis_weight:g}")
        for index, sample in enumerate(bar):
            if sample.sid in completed:
                continue
            if not Path(sample.image_path).is_file():
                message = f"Missing image for sid={sample.sid}: {sample.image_path}"
                if args.skip_missing_images:
                    print("WARNING:", message)
                    missing += 1
                    continue
                raise FileNotFoundError(message)

            if runner == "qwen":
                prediction, generation = generate_qwen(
                    model=model,
                    processor=processor,
                    controller=controller,
                    sample=sample,
                    weight=args.scalvis_weight,
                    max_new_tokens=max_tokens,
                    backend=args.backend,
                    device=args.device,
                )
            else:
                prediction, generation = generate_internvl(
                    model=model,
                    tokenizer=tokenizer,
                    controller=controller,
                    sample=sample,
                    weight=args.scalvis_weight,
                    max_new_tokens=max_tokens,
                    dtype=dtype,
                    device=args.device,
                    max_num=args.internvl_max_num,
                    use_thumbnail=not args.internvl_no_thumbnail,
                )

            correct, pred_norm, gold_norm = evaluate_prediction(
                args.dataset,
                prediction,
                sample.gold,
            )
            record = {
                "sid": sample.sid,
                "dataset": args.dataset,
                "image_path": sample.image_path,
                "prompt": sample.prompt,
                "gold": sample.gold,
                "gold_normalized": gold_norm,
                "prediction": prediction,
                "prediction_normalized": pred_norm,
                "correct": bool(correct),
                "metadata": sample.metadata,
                "generation": generation,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

            evaluated += 1
            num_new += 1
            num_correct += int(correct)
            if index < args.print_first:
                print("-" * 96)
                print(f"[{sample.sid}] gold={sample.gold!r} pred={prediction!r} correct={correct}")
                print("generation:", generation)
            bar.set_postfix(acc=f"{num_correct / max(evaluated, 1):.4f}")

    summary = {
        "backend": args.backend,
        "model": model_id,
        "dataset": args.dataset,
        "eps": float(args.rms_norm_eps),
        "scalvis_weight": float(args.scalvis_weight),
        "intervention": {
            "definition": (
                "At initial multimodal prefill only, multiply pre-softmax attention "
                "logits from the final prompt query to all visual-token keys."
            ),
            "max_layers": controller.max_layers,
            "decoder_layers": controller.num_layers,
        },
        "eps_info": eps_info,
        "dtype": args.dtype,
        "device": args.device,
        "max_new_tokens": int(max_tokens),
        "requested_samples": len(samples),
        "evaluated_samples": evaluated,
        "new_samples_this_run": num_new,
        "skipped_missing_images": missing,
        "num_correct": num_correct,
        "accuracy": num_correct / max(evaluated, 1),
        "records_jsonl": str(records_path),
        "data_paths": {
            "vsr_ann": args.vsr_ann if args.dataset == "vsr" else None,
            "vsr_image_root": args.vsr_image_root if args.dataset == "vsr" else None,
            "gqa_questions": args.gqa_questions if args.dataset == "gqa" else None,
            "gqa_image_root": args.gqa_image_root if args.dataset == "gqa" else None,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 96)
    print(
        f"RESULT {args.backend} {args.dataset} eps={args.rms_norm_eps:g} "
        f"w={args.scalvis_weight:g}: {num_correct}/{evaluated} = "
        f"{summary['accuracy']:.6f}"
    )
    print(f"Records: {records_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
