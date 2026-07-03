#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route B: natural-answer probe, then replay AdaptVis only at the decoder step
that predicts the probe's final relation word. Supports:

  --backend internvl25   OpenGVLab/InternVL2_5-2B
  --backend qwenvlchat   Qwen/Qwen-VL-Chat

Pass 1 uses weight=1.0 and records the final naturally generated relation token
(left/right/under/on). Pass 2 starts from the same prompt and applies the raw,
pre-softmax image-logit multiplier only at the forward pass predicting that
specific generated token. If the relation word is the first generated token,
the intervention occurs in the initial prefill; otherwise it occurs at the
corresponding cached decode step.

The InternVL branch delegates to the previously validated InternVL Route-B
runner. The Qwen branch implements the analogous targeted replay while
preserving Qwen-VL-Chat's native visual-token insertion and QWenAttention path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import zlib
import copy
import importlib
import json
import math
import random
import re
import subprocess
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset_zoo import get_dataset

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None


BACKEND_DEFAULTS = {
    "internvl25": ("OpenGVLab/InternVL2_5-2B", "main"),
    "qwenvlchat": ("Qwen/Qwen-VL-Chat", "main"),
}
RELATIONS = ("left", "right", "under", "on")


@dataclass
class TargetedQwenDiagnostics:
    requested_weight: float
    target_generation_index: Optional[int]
    current_generation_indices: List[int]
    modified_calls: int
    expected_modified_calls: int
    image_token_count: int
    image_start: Optional[int]
    image_end: Optional[int]
    prompt_sequence_length: int


class QwenRelationTokenResolver:
    """Locate relation words from decoded output, including multi-token Qwen BPE forms.

    Qwen's tiktoken-style vocabulary does not necessarily expose a literal
    single-token ``" right"`` or ``"on"`` entry through ``get_vocab()``.
    Instead of assuming one-token variants, decode cumulative generated-id
    prefixes and map each complete relation word back to the *first* generated
    token that overlaps the word.  That index is the query step that predicts
    the relation word's first subtoken.
    """

    _PATTERN = re.compile(r"\b(left|right|under|on)\b", flags=re.IGNORECASE)

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def _decode(self, ids: Sequence[int]) -> str:
        values = [int(x) for x in ids]
        try:
            return str(
                self.tokenizer.decode(
                    values,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    errors="replace",
                )
            )
        except TypeError:
            try:
                return str(
                    self.tokenizer.decode(
                        values,
                        skip_special_tokens=True,
                        errors="replace",
                    )
                )
            except TypeError:
                return str(self.tokenizer.decode(values, skip_special_tokens=True))

    @staticmethod
    def _first_overlapping_token(
        prefix_texts: Sequence[str],
        char_start: int,
        char_end: int,
    ) -> Optional[int]:
        # prefix_texts[i] is the decoded text after i generated tokens.
        # Return the first token whose decoded contribution overlaps the word.
        for token_index in range(len(prefix_texts) - 1):
            before = len(prefix_texts[token_index])
            after = len(prefix_texts[token_index + 1])
            if after > char_start and before < char_end:
                return token_index

        # Conservative fallback for an unusual non-monotonic tokenizer decode.
        for token_index in range(len(prefix_texts) - 1):
            if len(prefix_texts[token_index + 1]) >= char_end:
                return token_index
        return None

    def find_mentions(self, generated_ids: Sequence[int]) -> Tuple[List[int], List[str]]:
        ids = [int(x) for x in generated_ids]
        if not ids:
            return [], []

        full_text = self._decode(ids)
        matches = list(self._PATTERN.finditer(full_text))
        if not matches:
            return [], []

        prefix_texts = [self._decode(ids[:i]) for i in range(len(ids) + 1)]
        positions: List[int] = []
        labels: List[str] = []
        for match in matches:
            token_index = self._first_overlapping_token(
                prefix_texts,
                match.start(),
                match.end(),
            )
            if token_index is not None:
                positions.append(int(token_index))
                labels.append(match.group(1).lower())
        return positions, labels

class TargetedQwenController:
    """State for one replay that intervenes at exactly one generation index."""

    def __init__(self, max_layers: int, num_layers: int) -> None:
        self.max_layers = int(max_layers)
        self.num_layers = int(num_layers)
        self.active = False
        self.enabled = False
        self.weight = 1.0
        self.target_generation_index: Optional[int] = None
        self.image_mask: Optional[torch.Tensor] = None
        self.prompt_sequence_length = 0
        self.current_generation_index: Optional[int] = None
        self.decode_generation_index = 0
        self.modified_calls = 0
        self.current_generation_indices: List[int] = []

    def begin_generation(self, *, weight: float, target_generation_index: Optional[int]) -> None:
        self.active = True
        self.weight = float(weight)
        self.target_generation_index = target_generation_index
        self.enabled = bool(self.weight != 1.0 and target_generation_index is not None)
        self.image_mask = None
        self.prompt_sequence_length = 0
        self.current_generation_index = None
        self.decode_generation_index = 0
        self.modified_calls = 0
        self.current_generation_indices = []

    def capture_image_mask(self, input_ids: torch.Tensor, image_start_id: int) -> None:
        end_id = int(image_start_id) + 1
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(input_ids.shape[0]):
            starts = torch.nonzero(input_ids[b] == int(image_start_id), as_tuple=False).flatten()
            ends = torch.nonzero(input_ids[b] == end_id, as_tuple=False).flatten()
            if starts.numel() != ends.numel() or starts.numel() == 0:
                raise RuntimeError(
                    "Could not identify Qwen visual-token span from image_start/image_end ids: "
                    f"starts={starts.tolist()} ends={ends.tolist()}"
                )
            for start, end in zip(starts.tolist(), ends.tolist()):
                if end <= start + 1:
                    raise RuntimeError(f"Invalid Qwen visual span: start={start}, end={end}")
                mask[b, start + 1 : end] = True
        self.image_mask = mask
        self.prompt_sequence_length = int(input_ids.shape[-1])

    def observe_layer0_call(self, q_len: int, k_len: int) -> None:
        if not self.active:
            return
        if q_len == self.prompt_sequence_length and k_len == q_len:
            self.current_generation_index = 0
        elif q_len == 1 and k_len >= self.prompt_sequence_length:
            self.decode_generation_index += 1
            self.current_generation_index = self.decode_generation_index
        else:
            self.current_generation_index = None
        if self.current_generation_index is not None:
            self.current_generation_indices.append(int(self.current_generation_index))

    def should_intervene(self, layer_index: int) -> bool:
        return bool(
            self.enabled
            and layer_index < self.max_layers
            and self.current_generation_index == self.target_generation_index
            and self.image_mask is not None
        )

    def image_mask_for_kv(self, *, batch_size: int, kv_len: int, device: torch.device) -> torch.Tensor:
        if self.image_mask is None:
            raise RuntimeError("Image mask is not available.")
        if kv_len < self.prompt_sequence_length:
            raise RuntimeError(f"KV length {kv_len} is shorter than prompt {self.prompt_sequence_length}.")
        source = self.image_mask
        if source.shape[0] == 1 and batch_size > 1:
            source = source.expand(batch_size, -1)
        if source.shape[0] != batch_size:
            raise RuntimeError("Qwen visual-mask batch size mismatch.")
        result = torch.zeros((batch_size, kv_len), dtype=torch.bool, device=device)
        result[:, : self.prompt_sequence_length] = source.to(device=device, dtype=torch.bool)
        return result

    def finish_generation(self) -> TargetedQwenDiagnostics:
        count = 0
        start = end = None
        if self.image_mask is not None:
            positions = torch.nonzero(self.image_mask[0].detach().cpu(), as_tuple=False).flatten()
            count = int(positions.numel())
            if count:
                start = int(positions.min())
                end = int(positions.max()) + 1
        out = TargetedQwenDiagnostics(
            requested_weight=float(self.weight),
            target_generation_index=self.target_generation_index,
            current_generation_indices=list(self.current_generation_indices),
            modified_calls=int(self.modified_calls),
            expected_modified_calls=(min(self.max_layers, self.num_layers) if self.enabled else 0),
            image_token_count=count,
            image_start=start,
            image_end=end,
            prompt_sequence_length=int(self.prompt_sequence_length),
        )
        self.active = False
        self.enabled = False
        self.current_generation_index = None
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-contained Route B targeted relation-step AdaptVis for InternVL2.5 or Qwen-VL-Chat.")
    p.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS), required=True)
    p.add_argument("--dataset", default="Controlled_Images_A", choices=["Controlled_Images_A", "Controlled_Images_B"])
    p.add_argument("--option", default="four", choices=["two", "four", "six"])
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--rms-norm-eps", default=1e-5, type=float)
    p.add_argument("--weight", default=0.5, type=float)
    p.add_argument("--max-layers", default=None, type=int)
    p.add_argument("--max-num", default=12, type=int)
    p.add_argument("--use-thumbnail", dest="use_thumbnail", action="store_true", default=True)
    p.add_argument("--no-thumbnail", dest="use_thumbnail", action="store_false")
    p.add_argument("--max-new-tokens", default=32, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--limit", default=None, type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--image-cache-dir", default="data/qwenvl_chat_eval_images")
    p.add_argument("--prompt-mode", default="auto", choices=["auto", "raw", "llava"])
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--output", default=None)
    p.add_argument("--print-first", default=5, type=int)
    return p.parse_args()


def resolve_model_args(args: argparse.Namespace) -> Tuple[str, str, int]:
    default_model, default_revision = BACKEND_DEFAULTS[args.backend]
    model = args.model or default_model
    revision = args.revision or default_revision
    max_layers = args.max_layers
    if max_layers is None:
        max_layers = 24 if args.backend == "internvl25" else 32
    return model, revision, int(max_layers)



# Embedded InternVL Route-B runner. It is materialized beside this script on
# first use, so this file is self-contained for both backends.
_INTERNVL_ROUTE_B_PAYLOAD = "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"
_INTERNVL_ROUTE_B_SHA256 = "513b0f683aa9e08647e5900557ab5d8d6d7f2d415da500067f71d8de5987074e"


def ensure_internvl_runner() -> Path:
    runner = HERE / ".internvl_routeB_embedded_runner.py"
    expected = _INTERNVL_ROUTE_B_SHA256
    needs_write = True
    if runner.exists():
        try:
            needs_write = hashlib.sha256(runner.read_bytes()).hexdigest() != expected
        except OSError:
            needs_write = True
    if needs_write:
        source = zlib.decompress(base64.b64decode(_INTERNVL_ROUTE_B_PAYLOAD)).decode("utf-8")
        runner.write_text(source, encoding="utf-8")
    return runner


def internvl_command(args: argparse.Namespace) -> List[str]:
    model, revision, max_layers = resolve_model_args(args)
    runner = ensure_internvl_runner()
    cmd = [
        sys.executable, str(runner),
        "--dataset", args.dataset,
        "--option", args.option,
        "--model", model,
        "--revision", revision,
        "--cache-dir", args.cache_dir,
        "--device", args.device,
        "--dtype", args.dtype,
        "--rms-norm-eps", str(args.rms_norm_eps),
        "--method", "scaling_vis",
        "--weight", str(args.weight),
        "--max-layers", str(max_layers),
        "--max-num", str(args.max_num),
        "--max-new-tokens", str(args.max_new_tokens),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--print-first", str(args.print_first),
    ]
    cmd.append("--use-thumbnail" if args.use_thumbnail else "--no-thumbnail")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.download:
        cmd.append("--download")
    if args.output:
        cmd += ["--output", args.output]
    return cmd


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_gold(x: Any) -> str:
    if isinstance(x, (list, tuple)):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def load_prompts(dataset: str, option: str) -> Tuple[List[str], List[Any]]:
    path = Path(f"prompts/{dataset}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts, answers = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        prompts.append(item["question"])
        answers.append(item["answer"])
    return prompts, answers


def extract_images_from_batch(batch: Dict[str, Any]) -> Iterable[Any]:
    for image_option in batch["image_options"]:
        for image in image_option:
            yield image


def install_qwen_image_span_capture(model: Any, controller: TargetedQwenController) -> None:
    transformer = model.transformer
    original_forward = transformer.forward
    image_start_id = int(model.config.visual["image_start_id"])

    def wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        past = kwargs.get("past_key_values", args[1] if len(args) > 1 else None)
        if input_ids is not None and past is None and torch.any(input_ids == image_start_id):
            controller.capture_image_mask(input_ids, image_start_id)
        return original_forward(*args, **kwargs)

    transformer.forward = types.MethodType(wrapped_forward, transformer)


def install_targeted_qwen_attention(model: Any, controller: TargetedQwenController) -> None:
    layers = getattr(model.transformer, "h", None)
    if layers is None:
        raise RuntimeError("Expected Qwen language layers at model.transformer.h")

    for layer_index, layer in enumerate(layers):
        attn = getattr(layer, "attn", None)
        if attn is None or not hasattr(attn, "_attn"):
            raise RuntimeError(f"Layer {layer_index} lacks native Qwen _attn")
        original_attn = attn._attn

        def make_patch(original: Any, idx: int, owner: Any):
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
                if idx == 0:
                    controller.observe_layer0_call(q_len, k_len)
                if not controller.should_intervene(idx):
                    return original(query, key, value, registered_causal_mask, attention_mask, head_mask)

                image_mask = controller.image_mask_for_kv(
                    batch_size=int(query.shape[0]), kv_len=k_len, device=query.device
                )
                attn_weights = torch.matmul(query, key.transpose(-1, -2))
                if getattr(self, "scale_attn_weights", True):
                    attn_weights = attn_weights / math.sqrt(float(value.size(-1)))
                for b in range(query.shape[0]):
                    image_indices = torch.nonzero(image_mask[b], as_tuple=False).flatten()
                    if image_indices.numel() == 0:
                        continue
                    row = attn_weights[b, :, q_len - 1, :]
                    row.index_copy_(-1, image_indices, row.index_select(-1, image_indices) * controller.weight)
                controller.modified_calls += 1
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_probs = nn.functional.softmax(attn_weights, dim=-1).type(value.dtype)
                attn_probs = self.attn_dropout(attn_probs)
                if head_mask is not None:
                    attn_probs = attn_probs * head_mask
                return torch.matmul(attn_probs, value).transpose(1, 2), attn_probs
            return types.MethodType(patched_attn, owner)

        attn._attn = make_patch(original_attn, layer_index, attn)


def load_qwen_targeted(args: argparse.Namespace) -> Tuple[Any, Any, TargetedQwenController, Any]:
    import run_qwenvl_chat_adaptvis_rmsnorm_eps_ablation as qwen
    model_name, revision, max_layers = resolve_model_args(args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, cache_dir=args.cache_dir, trust_remote_code=True
    )
    precision_kw = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        **{precision_kw: True},
    ).eval().to(args.device)
    count, before = qwen.set_qwen_rmsnorm_epsilon(model, float(args.rms_norm_eps))
    controller = TargetedQwenController(
        max_layers=max_layers,
        num_layers=int(getattr(model.config, "num_hidden_layers", 0)),
    )
    install_qwen_image_span_capture(model, controller)
    install_targeted_qwen_attention(model, controller)
    model._targeted_qwen_rms_count = count
    model._targeted_qwen_rms_before = before
    return model, tokenizer, controller, qwen


def qwen_helpers(model: Any) -> Tuple[Any, Any, Any]:
    module = importlib.import_module(type(model).__module__)
    return module.make_context, module.get_stop_words_ids, module.decode_tokens


@torch.inference_mode()
def qwen_generate(
    *,
    model: Any,
    tokenizer: Any,
    query: str,
    system: str,
    controller: TargetedQwenController,
    weight: float,
    target_index: Optional[int],
    max_new_tokens: int,
) -> Tuple[str, List[int], TargetedQwenDiagnostics]:
    make_context, get_stop_words_ids, decode_tokens = qwen_helpers(model)
    cfg = copy.deepcopy(model.generation_config)
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=[],
        system=system,
        max_window_size=cfg.max_window_size,
        chat_format=cfg.chat_format,
    )
    input_ids = torch.tensor([context_tokens], device=model.device)
    stops = copy.deepcopy(get_stop_words_ids(cfg.chat_format, tokenizer))
    controller.begin_generation(weight=weight, target_generation_index=target_index)
    output = model.generate(
        input_ids,
        stop_words_ids=stops,
        return_dict_in_generate=True,
        output_scores=True,
        generation_config=cfg,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    diagnostics = controller.finish_generation()
    seq = output.sequences if hasattr(output, "sequences") else output["sequences"]
    if seq.ndim != 2 or seq.shape[0] != 1:
        raise RuntimeError(f"Unexpected Qwen generated sequence shape: {tuple(seq.shape)}")
    context_len = len(context_tokens)
    generated = seq[0, context_len:] if seq.shape[1] >= context_len else seq[0]
    generation = decode_tokens(
        seq[0], tokenizer, raw_text_len=len(raw_text), context_length=context_len,
        chat_format=cfg.chat_format, verbose=False, errors="replace"
    )
    return str(generation).strip(), [int(x) for x in generated.detach().cpu().tolist()], diagnostics


@torch.inference_mode()
def run_qwen(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    seed_all(args.seed)
    model, tokenizer, controller, qwen = load_qwen_targeted(args)
    resolver = QwenRelationTokenResolver(tokenizer)
    prompts, answers = load_prompts(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=repository_default_collate)
    total = min(len(prompts), len(dataset))
    if args.limit is not None:
        total = min(total, int(args.limit))
    cache_root = Path(args.image_cache_dir) / args.dataset / "routeB_relationstep"

    records: List[Dict[str, Any]] = []
    correct_count = 0
    sid = 0
    bar = tqdm(total=total, desc="Qwen-VL-Chat Route-B relation-step")
    for batch in loader:
        for image in extract_images_from_batch(batch):
            if sid >= total:
                break
            raw_prompt = prompts[sid]
            question = qwen.sanitize_prompt_for_qwen(raw_prompt, args.prompt_mode)
            image_path = qwen.image_to_local_png(image, sid, cache_root)
            query = tokenizer.from_list_format([{"image": image_path}, {"text": question}])
            probe_text, probe_ids, probe_diag = qwen_generate(
                model=model, tokenizer=tokenizer, query=query, system=args.system,
                controller=controller, weight=1.0, target_index=None,
                max_new_tokens=args.max_new_tokens,
            )
            positions, labels = resolver.find_mentions(probe_ids)
            target_idx = positions[-1] if positions else None
            target_label = labels[-1] if labels else None
            final_text, final_ids, final_diag = qwen_generate(
                model=model, tokenizer=tokenizer, query=query, system=args.system,
                controller=controller, weight=args.weight, target_index=target_idx,
                max_new_tokens=args.max_new_tokens,
            )
            final_positions, final_labels = resolver.find_mentions(final_ids)
            final_label = final_labels[-1] if final_labels else None
            gold = norm_gold(answers[sid]).lower()
            correct = final_label == gold
            correct_count += int(correct)
            prefix_match = (target_idx is None) or final_ids[:target_idx] == probe_ids[:target_idx]
            rec = {
                "sid": sid,
                "prompt": raw_prompt,
                "question": question,
                "image_path": image_path,
                "gold": gold,
                "route": "B_probe_relation_step",
                "weight": float(args.weight),
                "probe_generation": probe_text,
                "probe_generated_ids": probe_ids,
                "probe_relation_positions": positions,
                "probe_relation_labels": labels,
                "target_generation_index": target_idx,
                "target_relation_label": target_label,
                "final_generation": final_text,
                "final_generated_ids": final_ids,
                "final_relation_positions": final_positions,
                "final_relation_labels": final_labels,
                "last_relation_label": final_label,
                "last_relation_correct": bool(correct),
                "prefix_before_target_matches_probe": bool(prefix_match),
                "probe_diagnostics": asdict(probe_diag),
                "final_diagnostics": asdict(final_diag),
            }
            records.append(rec)
            if sid < args.print_first:
                print("\n" + "-" * 110)
                print(f"[SID {sid}] gold={gold!r}")
                print(f"probe={probe_text!r}")
                print(f"target_generation_index={target_idx}, target_relation={target_label!r}")
                print(f"final={final_text!r} final_relation={final_label!r} correct={correct}")
                print(
                    f"replay diagnostics: target={final_diag.target_generation_index}, "
                    f"seen={final_diag.current_generation_indices}, modified_calls="
                    f"{final_diag.modified_calls}/{final_diag.expected_modified_calls}, "
                    f"prefix_match={prefix_match}"
                )
            sid += 1
            bar.update(1)
        if sid >= total:
            break
    bar.close()

    model_name, revision, max_layers = resolve_model_args(args)
    summary = {
        "backend": "qwenvlchat",
        "model": model_name,
        "revision": revision,
        "dataset": args.dataset,
        "option": args.option,
        "route": "B_probe_relation_step",
        "rms_norm_eps": float(args.rms_norm_eps),
        "weight": float(args.weight),
        "max_layers": max_layers,
        "num_samples": sid,
        "num_correct": correct_count,
        "accuracy": correct_count / max(sid, 1),
        "records": records,
    }
    output = Path(args.output) if args.output else Path("output") / f"qwenvlchat_routeB_{args.dataset}_eps{args.rms_norm_eps:.0e}_w{args.weight:g}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print(f"RESULT: {correct_count}/{sid} last-relation accuracy={summary['accuracy']:.6f}")
    print(f"Saved results to: {output}")


def main() -> None:
    args = parse_args()
    if args.backend == "internvl25":
        cmd = internvl_command(args)
        print("Dispatching InternVL Route-B runner:\n", " ".join(cmd))
        raise SystemExit(subprocess.run(cmd, check=False).returncode)
    run_qwen(args)


if __name__ == "__main__":
    main()
