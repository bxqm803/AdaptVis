#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternVL2.5-2B fixed-weight AdaptVis-style epsilon ablation with last-relation-token restricted probabilities.

This runner evaluates OpenGVLab/InternVL2_5-2B on the repository's Controlled
Images A/B tasks. It keeps the model's native visual pipeline:

    InternViT -> pixel shuffle -> MLP projector -> <IMG_CONTEXT> token slots
    -> InternLM2 decoder

During the initial multimodal prefill only, it multiplies the raw pre-mask,
pre-softmax attention logits from the final prompt query to all visual-token
keys by a fixed scalar --weight (default 0.5), in decoder layers
[0, --max-layers). It also optionally overrides every InternLM2RMSNorm
variance_epsilon inside the language decoder.

There is deliberately no adaptive gate in this runner. It is for the direct
comparison requested here: fixed weight=0.5 across an epsilon sweep.

The runner records the logits at the generation step that emits the final
relation token in the natural-language answer (the last generated occurrence
of ``left``, ``right``, ``under``, or ``on``). It then applies a restricted
four-way softmax over those single-token candidates. The four probabilities
sum to one. This avoids treating the sentence opener (often ``The``) as the
relation-decision position.

Run from the AdaptVis repository root, where dataset_zoo.py and prompts/ exist.
"""

from __future__ import annotations

import argparse
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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


DEFAULT_MODEL = "OpenGVLab/InternVL2_5-2B"
DEFAULT_REVISION = "main"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class GenerationDiagnostics:
    requested_weight: float
    modified_calls: int
    expected_modified_calls: int
    image_token_count: int
    prompt_sequence_length: int
    image_start: Optional[int]
    image_end: Optional[int]
    num_image_tiles: int



@dataclass
class LastRelationProbabilityDiagnostics:
    """Four-way relation probabilities at the final relation-token generation step."""

    relation_token_ids: Dict[str, int]
    relation_token_texts: Dict[str, str]
    generated_token_ids: List[int]
    generated_token_texts: List[str]
    generated_text_from_token_ids: str
    relation_generated_positions: List[int]
    relation_generated_labels: List[str]
    selected_relation_position: Optional[int]
    selected_relation_label: Optional[str]
    selected_relation_token_id: Optional[int]
    selected_relation_token_text: Optional[str]
    selected_step_top_token_id: Optional[int]
    selected_step_top_token_text: Optional[str]
    selected_step_top_token_probability: Optional[float]
    selected_step_top5: List[Dict[str, Any]]
    relation_logits: Optional[Dict[str, float]]
    relation_probabilities: Optional[Dict[str, float]]
    relation_prediction: Optional[str]
    relation_confidence: Optional[float]


class RelationStepLogitCapture:
    """
    Read-only capture of every decoder output distribution during natural generation.

    Index i in ``step_logits`` is the distribution that generated token i in
    ``generated_token_ids``: prefill supplies i=0, and each cached decoding step
    supplies the next index. The selected relation position is the last generated
    token whose id matches a single-token relation candidate.
    """

    RELATION_CANDIDATES: Tuple[str, ...] = ("left", "right", "under", "on")

    def __init__(self, language_model: Any, tokenizer: Any) -> None:
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.active = False
        self.step_logits: List[torch.Tensor] = []
        self.generated_token_ids: Optional[List[int]] = None
        self.relation_token_ids = self._resolve_relation_token_ids()
        self.inverse_relation_token_ids = {token_id: name for name, token_id in self.relation_token_ids.items()}
        self.output_head_name, output_head = self._resolve_output_head()
        self._handle = output_head.register_forward_hook(self._forward_hook)

    def _resolve_relation_token_ids(self) -> Dict[str, int]:
        """Require each restricted relation candidate to be exactly one tokenizer token."""
        token_ids: Dict[str, int] = {}
        for candidate in self.RELATION_CANDIDATES:
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) != 1:
                tokens = self.tokenizer.convert_ids_to_tokens(ids)
                raise RuntimeError(
                    f"Restricted candidate {candidate!r} is not a single token: "
                    f"ids={ids}, tokens={tokens}. This runner intentionally requires single-token candidates."
                )
            token_ids[candidate] = int(ids[0])
        if len(set(token_ids.values())) != len(token_ids):
            raise RuntimeError(f"Restricted relation candidates share token ids: {token_ids}")
        return token_ids

    def _resolve_output_head(self) -> Tuple[str, nn.Module]:
        output_head = None
        getter = getattr(self.language_model, "get_output_embeddings", None)
        if callable(getter):
            output_head = getter()

        if isinstance(output_head, nn.Module):
            for name, module in self.language_model.named_modules():
                if module is output_head:
                    return name or "<root>", output_head
            return "<unregistered-output-head>", output_head

        vocab_size = int(getattr(getattr(self.language_model, "config", None), "vocab_size", 0))
        candidates = [
            (name, module)
            for name, module in self.language_model.named_modules()
            if isinstance(module, nn.Linear) and vocab_size > 0 and int(module.out_features) == vocab_size
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise RuntimeError(
                "Could not identify the language-model output head for relation-step probability capture."
            )
        preferred = [
            pair for pair in candidates
            if pair[0].endswith(("lm_head", "output")) or pair[0] in {"lm_head", "output"}
        ]
        if len(preferred) == 1:
            return preferred[0]
        names = [name for name, _ in candidates]
        raise RuntimeError(
            "Ambiguous language-model output-head candidates for relation-step probability capture: "
            f"{names}"
        )

    def _forward_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if not self.active:
            return
        logits = output[0] if isinstance(output, tuple) else output
        if not isinstance(logits, torch.Tensor):
            return
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
            return
        # Prefill is [1, prompt_len, vocab]; cached decode is [1, 1, vocab].
        # In both cases the final row is the next-token distribution.
        self.step_logits.append(logits[0, -1, :].detach().to(dtype=torch.float32, device="cpu"))

    def begin_generation(self) -> None:
        self.active = True
        self.step_logits = []
        self.generated_token_ids = None

    def capture_generate_output(self, input_ids: torch.Tensor, output_ids: Any) -> None:
        if not self.active:
            return
        sequences = getattr(output_ids, "sequences", output_ids)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
            raise RuntimeError(
                "InternVL generate did not return a [1, sequence_length] token tensor; "
                f"got type={type(sequences)}, shape={getattr(sequences, 'shape', None)}."
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(f"Unexpected InternVL generate input_ids shape: {tuple(input_ids.shape)}")
        prompt_len = int(input_ids.shape[1])
        if int(sequences.shape[1]) < prompt_len:
            raise RuntimeError(
                "Generated sequence is shorter than the input prompt: "
                f"sequence_len={sequences.shape[1]}, prompt_len={prompt_len}."
            )
        self.generated_token_ids = [int(x) for x in sequences[0, prompt_len:].detach().cpu().tolist()]

    def _token_text(self, token_id: int) -> str:
        try:
            return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=False))
        except Exception:
            return str(self.tokenizer.convert_ids_to_tokens(int(token_id)))

    def finish_generation(self) -> LastRelationProbabilityDiagnostics:
        self.active = False
        if self.generated_token_ids is None:
            raise RuntimeError(
                "Failed to capture generated token ids from InternVL generate. "
                "The model.chat / generate path may have changed."
            )
        if not self.step_logits:
            raise RuntimeError("No decoder output logits were captured during InternVL generation.")

        generated_ids = list(self.generated_token_ids)
        # Standard greedy generation has one score distribution per generated token.
        # Keep a hard failure rather than silently misaligning a relation position.
        if len(self.step_logits) != len(generated_ids):
            raise RuntimeError(
                "Generation-logit alignment mismatch: "
                f"captured_step_logits={len(self.step_logits)}, generated_token_ids={len(generated_ids)}."
            )

        generated_token_texts = [self._token_text(token_id) for token_id in generated_ids]
        generated_text = str(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
        relation_positions = [
            index for index, token_id in enumerate(generated_ids)
            if token_id in self.inverse_relation_token_ids
        ]
        relation_labels = [self.inverse_relation_token_ids[generated_ids[index]] for index in relation_positions]
        relation_token_texts = {
            name: self._token_text(token_id) for name, token_id in self.relation_token_ids.items()
        }

        if not relation_positions:
            return LastRelationProbabilityDiagnostics(
                relation_token_ids=dict(self.relation_token_ids),
                relation_token_texts=relation_token_texts,
                generated_token_ids=generated_ids,
                generated_token_texts=generated_token_texts,
                generated_text_from_token_ids=generated_text,
                relation_generated_positions=[],
                relation_generated_labels=[],
                selected_relation_position=None,
                selected_relation_label=None,
                selected_relation_token_id=None,
                selected_relation_token_text=None,
                selected_step_top_token_id=None,
                selected_step_top_token_text=None,
                selected_step_top_token_probability=None,
                selected_step_top5=[],
                relation_logits=None,
                relation_probabilities=None,
                relation_prediction=None,
                relation_confidence=None,
            )

        selected_position = int(relation_positions[-1])
        selected_token_id = int(generated_ids[selected_position])
        selected_label = self.inverse_relation_token_ids[selected_token_id]
        logits = self.step_logits[selected_position]
        full_probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(full_probs, k=min(5, int(full_probs.numel())))
        selected_step_top5 = [
            {
                "token_id": int(token_id),
                "token": self._token_text(int(token_id)),
                "probability": float(prob),
            }
            for prob, token_id in zip(top_probs.tolist(), top_ids.tolist())
        ]

        ordered_candidates = list(self.RELATION_CANDIDATES)
        relation_ids = torch.tensor(
            [self.relation_token_ids[name] for name in ordered_candidates],
            dtype=torch.long,
        )
        relation_logits_tensor = logits.index_select(0, relation_ids)
        relation_probs_tensor = torch.softmax(relation_logits_tensor, dim=0)
        relation_logits = {
            name: float(value)
            for name, value in zip(ordered_candidates, relation_logits_tensor.tolist())
        }
        relation_probabilities = {
            name: float(value)
            for name, value in zip(ordered_candidates, relation_probs_tensor.tolist())
        }
        best_index = int(torch.argmax(relation_probs_tensor).item())
        relation_prediction = ordered_candidates[best_index]
        relation_confidence = float(relation_probs_tensor[best_index].item())

        return LastRelationProbabilityDiagnostics(
            relation_token_ids=dict(self.relation_token_ids),
            relation_token_texts=relation_token_texts,
            generated_token_ids=generated_ids,
            generated_token_texts=generated_token_texts,
            generated_text_from_token_ids=generated_text,
            relation_generated_positions=[int(x) for x in relation_positions],
            relation_generated_labels=relation_labels,
            selected_relation_position=selected_position,
            selected_relation_label=selected_label,
            selected_relation_token_id=selected_token_id,
            selected_relation_token_text=self._token_text(selected_token_id),
            selected_step_top_token_id=int(top_ids[0].item()),
            selected_step_top_token_text=self._token_text(int(top_ids[0].item())),
            selected_step_top_token_probability=float(top_probs[0].item()),
            selected_step_top5=selected_step_top5,
            relation_logits=relation_logits,
            relation_probabilities=relation_probabilities,
            relation_prediction=relation_prediction,
            relation_confidence=relation_confidence,
        )

class InternVLAdaptVisController:
    """Per-generation state used by the patched InternLM2 attention layers."""

    def __init__(self, max_layers: int, num_decoder_layers: int) -> None:
        self.max_layers = int(max_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.active = False
        self.enabled = False
        self.weight = 1.0
        self.image_mask: Optional[torch.Tensor] = None
        self.prompt_sequence_length = 0
        self.modified_calls = 0
        self.num_image_tiles = 0

    def begin_generation(self, weight: float, num_image_tiles: int) -> None:
        self.active = True
        self.weight = float(weight)
        self.enabled = self.weight != 1.0
        self.image_mask = None
        self.prompt_sequence_length = 0
        self.modified_calls = 0
        self.num_image_tiles = int(num_image_tiles)

    def capture_image_mask(self, input_ids: torch.LongTensor, image_context_token_id: int) -> None:
        if input_ids.ndim != 2:
            raise RuntimeError(f"Expected input_ids [B,L], got {tuple(input_ids.shape)}")
        image_mask = input_ids.eq(int(image_context_token_id))
        if not bool(image_mask.any().item()):
            raise RuntimeError(
                "Could not locate any <IMG_CONTEXT> tokens in InternVL input_ids. "
                "The model's native prompt construction may have changed."
            )
        self.image_mask = image_mask
        self.prompt_sequence_length = int(input_ids.shape[-1])

    def finish_generation(self) -> GenerationDiagnostics:
        count = 0
        start: Optional[int] = None
        end: Optional[int] = None
        if self.image_mask is not None and self.image_mask.numel() > 0:
            positions = torch.nonzero(self.image_mask[0].detach().cpu(), as_tuple=False).flatten()
            count = int(positions.numel())
            if count:
                start = int(positions.min())
                end = int(positions.max()) + 1

        result = GenerationDiagnostics(
            requested_weight=float(self.weight),
            modified_calls=int(self.modified_calls),
            expected_modified_calls=(
                min(int(self.max_layers), int(self.num_decoder_layers))
                if self.enabled
                else 0
            ),
            image_token_count=count,
            prompt_sequence_length=int(self.prompt_sequence_length),
            image_start=start,
            image_end=end,
            num_image_tiles=int(self.num_image_tiles),
        )
        self.active = False
        self.enabled = False
        self.image_mask = None
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InternVL2.5-2B fixed-weight pre-softmax image-attention + RMSNorm-epsilon ablation with restricted four-way probabilities at the final generated relation token."
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
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--rms-norm-eps",
        default=None,
        type=float,
        help="Override every InternLM2RMSNorm.variance_epsilon in the language decoder only.",
    )
    parser.add_argument("--method", default="scaling_vis", choices=["base", "scaling_vis"])
    parser.add_argument(
        "--weight",
        default=0.5,
        type=float,
        help="Fixed raw image-logit multiplier. Use 1.0 for the no-intervention control.",
    )
    parser.add_argument(
        "--max-layers",
        default=24,
        type=int,
        help="Apply to decoder layers [0,max_layers); capped at InternVL2.5-2B's 24 language layers.",
    )
    parser.add_argument(
        "--max-num",
        default=12,
        type=int,
        help="Maximum dynamic 448x448 image tiles, following the official single-image loader default.",
    )
    parser.add_argument(
        "--use-thumbnail",
        action="store_true",
        default=True,
        help="Append a thumbnail tile when dynamic preprocessing uses more than one tile (official default).",
    )
    parser.add_argument(
        "--no-thumbnail",
        dest="use_thumbnail",
        action="store_false",
        help="Disable the extra thumbnail tile.",
    )
    parser.add_argument("--max-new-tokens", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--print-first", default=5, type=int)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def require_supported_transformers() -> None:
    from packaging import version

    if version.parse(transformers.__version__) < version.parse("4.37.2"):
        raise RuntimeError(
            "InternVL2.5 requires transformers>=4.37.2; "
            f"found {transformers.__version__}."
        )


def norm_gold(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def normalize_relation_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_correct(gold: Any, generation: str) -> bool:
    """Preserve the repository's current substring-style Controlled metric."""
    gold_text = norm_gold(gold)
    prediction = normalize_relation_text(generation)
    if not gold_text:
        return False
    result = gold_text.lower() in prediction
    if gold_text.lower() == "on" and "front" in prediction:
        result = False
    return bool(result)


def extract_relation_mentions(generation: str) -> List[str]:
    return re.findall(r"\b(left|right|under|on)\b", normalize_relation_text(generation))


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


def sanitize_prompt_for_internvl(prompt: str) -> str:
    """Drop LLaVA conversation scaffolding and retain the visual relation question."""
    text = str(prompt).replace("\r\n", "\n").strip()
    text = re.sub(r"<image>", "", text, flags=re.IGNORECASE)
    text = text.replace("<im_start>", "").replace("<im_end>", " ").replace("<im_patch>", " ")

    user_match = re.search(r"\bUSER\s*:\s*", text, flags=re.IGNORECASE)
    assistant_matches = list(re.finditer(r"\bASSISTANT\s*:\s*", text, flags=re.IGNORECASE))
    if user_match is not None:
        start = user_match.end()
        end = assistant_matches[0].start() if assistant_matches else len(text)
        text = text[start:end]
    else:
        text = re.sub(r"^\s*###\s*(Human|USER)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.split(r"\s*###\s*(Assistant|ASSISTANT)\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]

    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        raise ValueError("InternVL question is empty after prompt conversion.")
    return text


def extract_images_from_batch(batch: Dict[str, Any]) -> Iterable[Any]:
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def ensure_rgb(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return image.convert("RGB")


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """Official InternVL dynamic tile selection, reproduced without torchvision."""
    if max_num < min_num or min_num < 1:
        raise ValueError(f"Invalid tile range: min_num={min_num}, max_num={max_num}")

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    ratio_w, ratio_h = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * ratio_w
    target_height = image_size * ratio_h
    blocks = ratio_w * ratio_h
    resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

    processed_images: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % ratio_w) * image_size,
            (i // ratio_w) * image_size,
            ((i % ratio_w) + 1) * image_size,
            ((i // ratio_w) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return processed_images


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_image(
    image: Image.Image,
    *,
    input_size: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    tiles = dynamic_preprocess(
        image,
        image_size=int(input_size),
        max_num=int(max_num),
        use_thumbnail=bool(use_thumbnail),
    )
    return torch.stack([image_to_tensor(tile, input_size) for tile in tiles])


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        .reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    )


def configure_language_rmsnorm_eps(model: Any, requested_eps: Optional[float]) -> Tuple[int, List[float], List[float]]:
    """Override language-model InternLM2 RMSNorm only; never vision/projector norms."""
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise RuntimeError("InternVL model does not expose .language_model.")

    norms = [
        module
        for module in language_model.modules()
        if type(module).__name__ == "InternLM2RMSNorm" and hasattr(module, "variance_epsilon")
    ]
    if not norms:
        raise RuntimeError("No InternLM2RMSNorm modules with .variance_epsilon found in model.language_model.")

    before = sorted({float(module.variance_epsilon) for module in norms})
    if requested_eps is not None:
        if requested_eps <= 0:
            raise ValueError(f"--rms-norm-eps must be positive, got {requested_eps}")
        for module in norms:
            module.variance_epsilon = float(requested_eps)
    after = sorted({float(module.variance_epsilon) for module in norms})
    if requested_eps is not None and after != [float(requested_eps)]:
        raise RuntimeError(f"Failed to set all InternLM2 RMSNorm eps values: {after}")
    return len(norms), before, after


def get_decoder_layers(model: Any) -> Sequence[nn.Module]:
    try:
        layers = model.language_model.model.layers
    except AttributeError as exc:
        raise RuntimeError(
            "Unexpected InternVL language decoder layout; expected model.language_model.model.layers."
        ) from exc
    if not layers:
        raise RuntimeError("No language decoder layers found.")
    return layers


def install_generate_input_capture(
    model: Any,
    controller: InternVLAdaptVisController,
    relation_step_capture: RelationStepLogitCapture,
) -> None:
    """Capture exact visual-token positions and generated token ids from native InternVL generate."""
    original_generate = model.generate

    def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            raise RuntimeError("InternVL generate was called without keyword input_ids; cannot capture image token span.")
        if controller.active:
            context_id = getattr(model, "img_context_token_id", None)
            if context_id is None:
                raise RuntimeError("model.img_context_token_id was not initialized before generation.")
            controller.capture_image_mask(input_ids, int(context_id))
        output_ids = original_generate(*args, **kwargs)
        if controller.active:
            relation_step_capture.capture_generate_output(input_ids, output_ids)
        return output_ids

    model.generate = wrapped_generate

def install_eager_internlm2_adaptvis(model: Any, controller: InternVLAdaptVisController) -> None:
    """
    Patch eager InternLM2Attention only on the selected initial prefill calls.

    The replacement copies the remote model's InternLM2Attention.forward logic,
    with exactly one insertion before mask/softmax:

      attn_logits[:, :, final_prompt_query, image_token_keys] *= weight

    All calls outside that condition delegate to the original native forward.
    """
    for layer_index, layer in enumerate(get_decoder_layers(model)):
        attn = getattr(layer, "attention", None)
        if attn is None:
            raise RuntimeError(f"Layer {layer_index} lacks .attention; expected eager InternLM2Attention.")
        original_forward = attn.forward
        required = (
            "wqkv",
            "wo",
            "rotary_emb",
            "num_heads",
            "num_key_value_heads",
            "num_key_value_groups",
            "head_dim",
            "hidden_size",
        )
        missing = [name for name in required if not hasattr(attn, name)]
        if missing:
            raise RuntimeError(
                f"Layer {layer_index} does not expose eager InternLM2 attention fields: {missing}. "
                "Load with use_flash_attn=False."
            )

        def make_forward(attn_module: nn.Module, original: Any, idx: int):
            def patched_forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs: Any,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                bsz, q_len, _ = hidden_states.size()
                image_mask = controller.image_mask
                should_intervene = (
                    controller.enabled
                    and controller.weight != 1.0
                    and idx < controller.max_layers
                    and image_mask is not None
                    and image_mask.ndim == 2
                    and image_mask.shape[0] in (1, bsz)
                    and image_mask.shape[-1] == q_len
                    and controller.prompt_sequence_length == q_len
                    and past_key_value is None
                )
                if not should_intervene:
                    return original(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )

                if position_ids is None:
                    raise RuntimeError("InternLM2 eager attention patch requires position_ids during prefill.")

                # Native InternLM2Attention.forward, reproduced from the remote model code.
                qkv_states = attn_module.wqkv(hidden_states)
                group_size = 2 + int(attn_module.num_key_value_groups)
                qkv_states = qkv_states.view(
                    bsz,
                    q_len,
                    int(attn_module.num_key_value_heads),
                    group_size,
                    int(attn_module.head_dim),
                )
                query_states = qkv_states[..., : int(attn_module.num_key_value_groups), :]
                query_states = query_states.reshape(
                    bsz,
                    q_len,
                    int(attn_module.num_heads),
                    int(attn_module.head_dim),
                ).transpose(1, 2)
                key_states = qkv_states[..., -2, :].transpose(1, 2)
                value_states = qkv_states[..., -1, :].transpose(1, 2)

                kv_seq_len = key_states.shape[-2]
                cos, sin = attn_module.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )

                # No cache should be present in the initial prefill branch, but preserve native behavior.
                if past_key_value is not None:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                next_past_key_value = (key_states, value_states) if use_cache else None

                key_states = repeat_kv(key_states, int(attn_module.num_key_value_groups))
                value_states = repeat_kv(value_states, int(attn_module.num_key_value_groups))
                attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                    int(attn_module.head_dim)
                )
                expected_shape = (bsz, int(attn_module.num_heads), q_len, kv_seq_len)
                if tuple(attn_logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected InternLM2 attention-logit shape: expected {expected_shape}, "
                        f"got {tuple(attn_logits.shape)}."
                    )

                local_image_mask = image_mask
                if local_image_mask.shape[0] == 1 and bsz > 1:
                    local_image_mask = local_image_mask.expand(bsz, -1)
                local_image_mask = local_image_mask.to(device=attn_logits.device, dtype=torch.bool)
                for batch_index in range(bsz):
                    key_mask = local_image_mask[batch_index]
                    if key_mask.numel() != kv_seq_len:
                        raise RuntimeError(
                            "InternVL image mask / KV length mismatch during initial prefill: "
                            f"mask={key_mask.numel()} kv={kv_seq_len}."
                        )
                    if not bool(key_mask.any().item()):
                        raise RuntimeError("No visual keys selected for the InternVL AdaptVis intervention.")
                    attn_logits[batch_index, :, q_len - 1, key_mask] *= float(controller.weight)
                controller.modified_calls += 1

                if attention_mask is not None:
                    expected_mask = (bsz, 1, q_len, kv_seq_len)
                    if tuple(attention_mask.shape) != expected_mask:
                        raise RuntimeError(
                            f"Unexpected InternLM2 attention-mask shape: expected {expected_mask}, "
                            f"got {tuple(attention_mask.shape)}."
                        )
                    attn_logits = attn_logits + attention_mask

                attn_probs = nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_probs, value_states)
                expected_output = (bsz, int(attn_module.num_heads), q_len, int(attn_module.head_dim))
                if tuple(attn_output.shape) != expected_output:
                    raise RuntimeError(
                        f"Unexpected InternLM2 attention output shape: expected {expected_output}, "
                        f"got {tuple(attn_output.shape)}."
                    )
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(
                    bsz, q_len, int(attn_module.hidden_size)
                )
                attn_output = attn_module.wo(attn_output)
                return attn_output, (attn_probs if output_attentions else None), next_past_key_value

            return patched_forward

        attn.forward = make_forward(attn, original_forward, layer_index)


def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> Tuple[Any, Any, InternVLAdaptVisController, RelationStepLogitCapture]:
    require_supported_transformers()
    dtype = resolve_dtype(args.dtype)
    print(f"transformers version: {transformers.__version__}")
    print(f"Loading {args.model}@{args.revision} with eager InternLM2 attention")

    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval()
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        use_fast=False,
    )

    num_norms, rms_before, rms_after = configure_language_rmsnorm_eps(model, args.rms_norm_eps)
    decoder_layers = get_decoder_layers(model)
    controller = InternVLAdaptVisController(args.max_layers, len(decoder_layers))
    relation_step_capture = RelationStepLogitCapture(model.language_model, tokenizer)
    install_generate_input_capture(model, controller, relation_step_capture)
    install_eager_internlm2_adaptvis(model, controller)

    image_size = int(getattr(model, "config").force_image_size or getattr(model, "config").vision_config.image_size)
    num_image_token = int(getattr(model, "num_image_token"))
    attention_classes = sorted({layer.attention.__class__.__name__ for layer in decoder_layers})
    print(f"InternVL language decoder layers: {len(decoder_layers)}")
    print(f"InternVL language RMSNorm modules: {num_norms}")
    print(f"InternVL RMSNorm eps before override: {rms_before}")
    print(f"InternVL RMSNorm eps after override: {rms_after}")
    print(f"Attention classes: {attention_classes}")
    print(f"vision input size: {image_size}; visual tokens per tile: {num_image_token}")
    print(f"model parameter dtype: {next(model.parameters()).dtype}")
    print(f"relation-step probability output head: {relation_step_capture.output_head_name}")
    print(
        "restricted four-way token ids: "
        f"{relation_step_capture.relation_token_ids}"
    )

    model._adaptvis_language_rmsnorm_count = int(num_norms)
    model._adaptvis_rmsnorm_before = list(rms_before)
    model._adaptvis_rmsnorm_after = list(rms_after)
    model._adaptvis_image_size = int(image_size)
    model._adaptvis_num_image_token = int(num_image_token)
    return model, tokenizer, controller, relation_step_capture


@torch.inference_mode()
def generate_once(
    *,
    model: Any,
    tokenizer: Any,
    controller: InternVLAdaptVisController,
    relation_step_capture: RelationStepLogitCapture,
    image: Image.Image,
    question: str,
    weight: float,
    max_num: int,
    use_thumbnail: bool,
    max_new_tokens: int,
) -> Tuple[str, GenerationDiagnostics, LastRelationProbabilityDiagnostics]:
    image_size = int(model._adaptvis_image_size)
    pixel_values = preprocess_image(
        image,
        input_size=image_size,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    ).to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

    controller.begin_generation(weight=weight, num_image_tiles=int(pixel_values.shape[0]))
    relation_step_capture.begin_generation()
    generation_config = {
        "num_beams": 1,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    try:
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=False,
            num_patches_list=[int(pixel_values.shape[0])],
            verbose=False,
        )
    finally:
        diagnostics = controller.finish_generation()
        relation_step_prob = relation_step_capture.finish_generation()
    return str(response).strip(), diagnostics, relation_step_prob


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    controller: InternVLAdaptVisController,
    relation_step_capture: RelationStepLogitCapture,
) -> Dict[str, Any]:
    prompts, answers = load_prompts(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=repository_default_collate,
    )

    total_available = min(len(prompts), len(dataset))
    total_target = min(total_available, args.limit) if args.limit is not None else total_available
    requested_weight = 1.0 if args.method == "base" else float(args.weight)

    records: List[Dict[str, Any]] = []
    correct_count = 0
    last_relation_correct_count = 0
    last_relation_covered_count = 0
    sid = 0
    progress = tqdm(total=total_target, desc=f"InternVL2.5 {args.method}")

    for batch in loader:
        for raw_image in extract_images_from_batch(batch):
            if sid >= total_target:
                break
            raw_prompt = prompts[sid]
            question = sanitize_prompt_for_internvl(raw_prompt)
            image = ensure_rgb(raw_image)
            gold = norm_gold(answers[sid])

            generation, diagnostics, relation_step_prob = generate_once(
                model=model,
                tokenizer=tokenizer,
                controller=controller,
                relation_step_capture=relation_step_capture,
                image=image,
                question=question,
                weight=requested_weight,
                max_num=args.max_num,
                use_thumbnail=args.use_thumbnail,
                max_new_tokens=args.max_new_tokens,
            )
            correct = is_correct(gold, generation)
            correct_count += int(correct)

            selected_relation_label = relation_step_prob.selected_relation_label
            last_relation_correct: Optional[bool]
            if selected_relation_label is None:
                last_relation_correct = None
            else:
                last_relation_covered_count += 1
                last_relation_correct = selected_relation_label == gold.lower()
                last_relation_correct_count += int(last_relation_correct)

            if requested_weight != 1.0 and diagnostics.modified_calls != diagnostics.expected_modified_calls:
                raise RuntimeError(
                    "AdaptVis failed to modify every expected initial-prefill layer: "
                    f"modified={diagnostics.modified_calls}, expected={diagnostics.expected_modified_calls}."
                )
            expected_image_tokens = diagnostics.num_image_tiles * int(model._adaptvis_num_image_token)
            if diagnostics.image_token_count != expected_image_tokens:
                raise RuntimeError(
                    "Unexpected InternVL visual-token count: "
                    f"observed={diagnostics.image_token_count}, expected={expected_image_tokens} "
                    f"({diagnostics.num_image_tiles} tiles x {model._adaptvis_num_image_token})."
                )

            records.append(
                {
                    "sid": sid,
                    "prompt": raw_prompt,
                    "internvl_question": question,
                    "gold": gold,
                    "method": args.method,
                    "selected_weight": requested_weight,
                    "generation": generation,
                    "relation_mentions": extract_relation_mentions(generation),
                    "correct": bool(correct),
                    "last_relation_correct": last_relation_correct,
                    "relation_step_token_ids": relation_step_prob.relation_token_ids,
                    "relation_step_token_texts": relation_step_prob.relation_token_texts,
                    "generated_token_ids": relation_step_prob.generated_token_ids,
                    "generated_token_texts": relation_step_prob.generated_token_texts,
                    "generated_text_from_token_ids": relation_step_prob.generated_text_from_token_ids,
                    "relation_generated_positions": relation_step_prob.relation_generated_positions,
                    "relation_generated_labels": relation_step_prob.relation_generated_labels,
                    "last_relation_position": relation_step_prob.selected_relation_position,
                    "last_relation_label": relation_step_prob.selected_relation_label,
                    "last_relation_token_id": relation_step_prob.selected_relation_token_id,
                    "last_relation_token_text": relation_step_prob.selected_relation_token_text,
                    "last_relation_step_top_token_id": relation_step_prob.selected_step_top_token_id,
                    "last_relation_step_top_token": relation_step_prob.selected_step_top_token_text,
                    "last_relation_step_top_probability": relation_step_prob.selected_step_top_token_probability,
                    "last_relation_step_top5": relation_step_prob.selected_step_top5,
                    "last_relation_four_way_logits": relation_step_prob.relation_logits,
                    "last_relation_four_way_probs": relation_step_prob.relation_probabilities,
                    "last_relation_four_way_pred": relation_step_prob.relation_prediction,
                    "last_relation_four_way_confidence": relation_step_prob.relation_confidence,
                    "diagnostics": asdict(diagnostics),
                }
            )

            if sid < args.print_first:
                print("\n" + "-" * 104)
                print(f"[SID {sid}] gold={gold!r}")
                print(f"internvl_question={question!r}")
                print(f"weight={requested_weight} pred={generation!r} correct={correct}")
                print(
                    "last relation token "
                    f"pos={relation_step_prob.selected_relation_position} "
                    f"label={relation_step_prob.selected_relation_label!r} "
                    f"last_relation_correct={last_relation_correct} "
                    f"all_relation_labels={relation_step_prob.relation_generated_labels}"
                )
                if relation_step_prob.selected_relation_label is None:
                    print("last-relation four-way probabilities: unavailable (no generated relation token)")
                else:
                    print(
                        f"last-relation-step top token={relation_step_prob.selected_step_top_token_text!r} "
                        f"prob={relation_step_prob.selected_step_top_token_probability:.6f}"
                    )
                    print(
                        "last-relation four-way probs="
                        f"{relation_step_prob.relation_probabilities} "
                        f"pred={relation_step_prob.relation_prediction!r} "
                        f"conf={relation_step_prob.relation_confidence:.6f}"
                    )
                print(
                    "image tiles/tokens="
                    f"{diagnostics.num_image_tiles}/{diagnostics.image_token_count}, "
                    f"range=[{diagnostics.image_start},{diagnostics.image_end}), "
                    f"prompt_len={diagnostics.prompt_sequence_length}, "
                    f"modified_calls={diagnostics.modified_calls}/{diagnostics.expected_modified_calls}"
                )

            sid += 1
            progress.update(1)
        if sid >= total_target:
            break

    progress.close()
    accuracy = correct_count / max(sid, 1)
    last_relation_accuracy = (
        last_relation_correct_count / max(last_relation_covered_count, 1)
        if last_relation_covered_count
        else None
    )
    summary = {
        "model": args.model,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "implementation": (
            "InternVL2.5 fixed-weight pre-softmax eager InternLM2 final-query-to-all-visual-token "
            "raw-logit scaling with optional language RMSNorm epsilon override; restricted four-way "
            "probabilities captured at the final naturally generated relation token"
        ),
        "dataset": args.dataset,
        "option": args.option,
        "method": args.method,
        "weight": requested_weight,
        "requested_rms_norm_eps": args.rms_norm_eps,
        "language_rms_norm_count": getattr(model, "_adaptvis_language_rmsnorm_count", None),
        "language_rms_norm_eps_before": getattr(model, "_adaptvis_rmsnorm_before", None),
        "active_rms_norm_eps": getattr(model, "_adaptvis_rmsnorm_after", None),
        "max_layers": args.max_layers,
        "max_num": args.max_num,
        "use_thumbnail": bool(args.use_thumbnail),
        "vision_input_size": getattr(model, "_adaptvis_image_size", None),
        "visual_tokens_per_tile": getattr(model, "_adaptvis_num_image_token", None),
        "max_new_tokens": args.max_new_tokens,
        "dtype_argument": args.dtype,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "restricted_four_way_candidates": list(relation_step_capture.RELATION_CANDIDATES),
        "restricted_four_way_token_ids": dict(relation_step_capture.relation_token_ids),
        "num_samples": sid,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "num_last_relation_covered": last_relation_covered_count,
        "num_last_relation_correct": last_relation_correct_count,
        "last_relation_accuracy": last_relation_accuracy,
        "records": records,
    }
    print("\n" + "=" * 104)
    print(
        f"RESULT: {correct_count}/{sid} accuracy={accuracy:.6f} "
        f"last_relation={last_relation_correct_count}/{last_relation_covered_count} "
        f"last_relation_accuracy={last_relation_accuracy} "
        f"method={args.method} weight={requested_weight} "
        f"rms_norm_eps={getattr(model, '_adaptvis_rmsnorm_after', None)}"
    )
    print("=" * 104)
    return summary


def _float_tag(value: float) -> str:
    return f"{float(value):.0e}".replace("e-", "em").replace("e+", "ep")


def default_output_path(args: argparse.Namespace) -> Path:
    tag = f"internvl25_2b_{args.dataset}_{args.method}"
    if args.method == "scaling_vis":
        tag += f"_w{args.weight:g}"
    if args.rms_norm_eps is not None:
        tag += f"_eps{_float_tag(args.rms_norm_eps)}"
    return Path("output") / f"{tag}_lastrelationprob.json"


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive.")
    if args.max_num <= 0:
        raise ValueError("--max-num must be positive.")

    model, tokenizer, controller, relation_step_capture = load_model_and_tokenizer(args)
    summary = evaluate(args, model, tokenizer, controller, relation_step_capture)
    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
