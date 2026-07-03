#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route A: force a one-word spatial-relation answer, then apply AdaptVis during
initial multimodal prefill. Supports:

  --backend internvl25   OpenGVLab/InternVL2_5-2B
  --backend qwenvlchat   Qwen/Qwen-VL-Chat

For both backends, the question is rewritten to end with:
  "Respond with exactly one lowercase word: left, right, on, or under.
   Do not explain. Answer:"

Therefore the initial prefill's final query is the query that predicts the
relation token. The intervention is the original AdaptVis form: multiply the
raw, pre-softmax attention logits from that final query to all visual-token
keys by --weight, in decoder layers [0, --max-layers).

The InternVL runner is embedded and materialized automatically on first use; no companion file is required. The Qwen branch uses the existing Qwen-VL-Chat loader
and native prefill AdaptVis patch, but replaces the question with the strict
one-word instruction above.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import zlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Route A one-word prefill AdaptVis for InternVL2.5 or Qwen-VL-Chat."
    )
    p.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS), required=True)
    p.add_argument("--dataset", default="Controlled_Images_A", choices=["Controlled_Images_A", "Controlled_Images_B"])
    p.add_argument("--option", default="four", choices=["two", "four", "six"])
    p.add_argument("--model", default=None, help="Override the backend default checkpoint.")
    p.add_argument("--revision", default=None, help="Override the backend default revision.")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--rms-norm-eps", default=1e-5, type=float)
    p.add_argument("--weight", default=0.5, type=float)
    p.add_argument("--max-layers", default=None, type=int, help="Default: 24 for InternVL2.5, 32 for Qwen-VL-Chat.")
    p.add_argument("--max-num", default=12, type=int, help="InternVL dynamic-image tile limit.")
    p.add_argument("--use-thumbnail", dest="use_thumbnail", action="store_true", default=True)
    p.add_argument("--no-thumbnail", dest="use_thumbnail", action="store_false")
    p.add_argument("--max-new-tokens", default=4, type=int)
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



# Embedded InternVL Route-A runner. It is materialized beside this script on
# first use, so this file is self-contained for both backends.
_INTERNVL_ROUTE_A_PAYLOAD = "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"
_INTERNVL_ROUTE_A_SHA256 = "cc46fadeb45cc32eb0ad01758cf8a0130ed16aeaf70c7d5d83c69739562628d5"


def ensure_internvl_runner() -> Path:
    runner = HERE / ".internvl_routeA_embedded_runner.py"
    expected = _INTERNVL_ROUTE_A_SHA256
    needs_write = True
    if runner.exists():
        try:
            needs_write = hashlib.sha256(runner.read_bytes()).hexdigest() != expected
        except OSError:
            needs_write = True
    if needs_write:
        source = zlib.decompress(base64.b64decode(_INTERNVL_ROUTE_A_PAYLOAD)).decode("utf-8")
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


def strict_oneword_question(question: str) -> str:
    import re
    text = " ".join(str(question).split())
    # The dataset prompt already ends with a weaker answer instruction. Replace
    # it rather than appending a competing second instruction.
    text = re.sub(r"\s*Answer\s+with\s+.*$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        raise ValueError("Question became empty after removing its original answer instruction.")
    return (
        text
        + " Respond with exactly one lowercase word: left, right, on, or under."
          " Do not explain. Answer:"
    )


def relation_from_text(text: str) -> str | None:
    import re
    hits = re.findall(r"\b(left|right|under|on)\b", str(text).lower())
    return hits[-1] if hits else None


def run_qwen(args: argparse.Namespace) -> None:
    # Reuse the verified native loader / prefill patch from the prior Qwen runner.
    import run_qwenvl_chat_adaptvis_rmsnorm_eps_ablation as qwen

    model_name, revision, max_layers = resolve_model_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    class QwenArgs:
        pass

    qa = QwenArgs()
    qa.model = model_name
    qa.revision = revision
    qa.cache_dir = args.cache_dir
    qa.device = args.device
    qa.dtype = args.dtype
    qa.rms_norm_eps = float(args.rms_norm_eps)
    qa.max_layers = int(max_layers)

    qwen.seed_all(args.seed)
    model, tokenizer, controller = qwen.load_qwen_model(qa)
    prompts, answers = qwen.load_prompts(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=repository_default_collate,
    )

    total = min(len(prompts), len(dataset))
    if args.limit is not None:
        total = min(total, int(args.limit))
    cache_root = Path(args.image_cache_dir) / args.dataset / "routeA_oneword"

    records: List[Dict[str, Any]] = []
    correct_count = 0
    sid = 0
    bar = tqdm(total=total, desc="Qwen-VL-Chat Route-A one-word")
    for batch in loader:
        for image in qwen.extract_images_from_batch(batch):
            if sid >= total:
                break
            raw_prompt = prompts[sid]
            base_question = qwen.sanitize_prompt_for_qwen(raw_prompt, args.prompt_mode)
            question = strict_oneword_question(base_question)
            image_path = qwen.image_to_local_png(image, sid, cache_root)
            query = tokenizer.from_list_format([{"image": image_path}, {"text": question}])
            generation, first_prob, diag = qwen.generate_once(
                model=model,
                tokenizer=tokenizer,
                query=query,
                system=args.system,
                controller=controller,
                weight=float(args.weight),
                max_new_tokens=int(args.max_new_tokens),
            )
            gold = qwen.norm_gold(answers[sid])
            relation = relation_from_text(generation)
            strict_correct = relation == gold.lower()
            substring_correct = qwen.is_correct(gold, generation)
            correct_count += int(strict_correct)
            rec = {
                "sid": sid,
                "prompt": raw_prompt,
                "question": question,
                "qwen_query": query,
                "image_path": image_path,
                "gold": gold,
                "generation": generation,
                "last_relation_label": relation,
                "last_relation_correct": bool(strict_correct),
                "correct": bool(substring_correct),
                "first_step_top_probability": float(first_prob),
                "diagnostics": asdict(diag),
            }
            records.append(rec)
            if sid < args.print_first:
                print("\n" + "-" * 100)
                print(f"[SID {sid}] gold={gold!r}")
                print(f"qwen_question={question!r}")
                print(f"weight={args.weight} pred={generation!r} relation={relation!r} correct={strict_correct}")
                print(
                    f"image tokens={diag.image_token_count}, range=[{diag.image_start},{diag.image_end}), "
                    f"prompt_len={diag.prompt_sequence_length}, modified_calls={diag.modified_calls}"
                )
            sid += 1
            bar.update(1)
        if sid >= total:
            break
    bar.close()

    summary = {
        "backend": "qwenvlchat",
        "model": model_name,
        "revision": revision,
        "dataset": args.dataset,
        "option": args.option,
        "route": "A_oneword_prefill",
        "prompt_suffix": "Respond with exactly one lowercase word: left, right, on, or under. Do not explain. Answer:",
        "rms_norm_eps": float(args.rms_norm_eps),
        "weight": float(args.weight),
        "max_layers": int(max_layers),
        "num_samples": sid,
        "num_correct": correct_count,
        "accuracy": correct_count / max(sid, 1),
        "records": records,
    }
    output = Path(args.output) if args.output else Path("output") / f"qwenvlchat_routeA_{args.dataset}_eps{args.rms_norm_eps:.0e}_w{args.weight:g}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print(f"RESULT: {correct_count}/{sid} strict-last-relation accuracy={summary['accuracy']:.6f}")
    print(f"Saved results to: {output}")


def main() -> None:
    args = parse_args()
    if args.backend == "internvl25":
        cmd = internvl_command(args)
        print("Dispatching InternVL Route-A runner:\n", " ".join(cmd))
        raise SystemExit(subprocess.run(cmd, check=False).returncode)
    run_qwen(args)


if __name__ == "__main__":
    main()
