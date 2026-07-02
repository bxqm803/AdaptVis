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
    """Resolve word-boundary single-token variants in Qwen's actual vocabulary."""

    def __init__(self, tokenizer: Any) -> None:
        vocab = tokenizer.get_vocab()
        self.id_to_label: Dict[int, str] = {}
        for token, token_id_raw in vocab.items():
            raw = str(token)
            surface = raw
            if raw.startswith("▁") or raw.startswith("Ġ"):
                surface = raw[1:]
            elif raw.startswith(" "):
                surface = raw[1:]
            label = surface.lower()
            if label in RELATIONS:
                self.id_to_label.setdefault(int(token_id_raw), label)
        if not self.id_to_label:
            raise RuntimeError("No exact one-token left/right/under/on variants were found in the Qwen tokenizer.")

    def find_mentions(self, generated_ids: Sequence[int]) -> Tuple[List[int], List[str]]:
        positions: List[int] = []
        labels: List[str] = []
        for i, token_id in enumerate(generated_ids):
            label = self.id_to_label.get(int(token_id))
            if label is not None:
                positions.append(i)
                labels.append(label)
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
    p = argparse.ArgumentParser(description="Route B targeted relation-step AdaptVis for InternVL2.5 or Qwen-VL-Chat.")
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


def internvl_command(args: argparse.Namespace) -> List[str]:
    model, revision, max_layers = resolve_model_args(args)
    runner = HERE / "run_internvl25_2b_probe_relation_step_adaptvis.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing InternVL Route-B runner beside this script: {runner}")
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
