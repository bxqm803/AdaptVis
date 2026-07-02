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

The InternVL branch delegates to the already validated InternVL Route-A runner
in the same directory. The Qwen branch uses the existing Qwen-VL-Chat loader
and native prefill AdaptVis patch, but replaces the question with the strict
one-word instruction above.
"""

from __future__ import annotations

import argparse
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


def internvl_command(args: argparse.Namespace) -> List[str]:
    model, revision, max_layers = resolve_model_args(args)
    runner = HERE / "run_internvl25_2b_oneword_prefill_adaptvis.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing InternVL Route-A runner beside this script: {runner}")
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
    return (
        question.rstrip()
        + "\nRespond with exactly one lowercase word: left, right, on, or under."
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
