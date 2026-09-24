#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor

import extract_two_object_relation_states as backend


COLORS = (
    "black",
    "white",
    "red",
    "blue",
    "green",
    "brown",
)

DEFAULT_MODELS = (
    "qwen-3b",
    "qwen-7b",
    "llava-7b",
    "llava-13b",
    "qwen2-2b",
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        required=True,
        choices=DEFAULT_MODELS,
    )

    p.add_argument(
        "--subset-jsonl",
        default="data/gqa/subsets/color600/val.jsonl",
    )

    p.add_argument(
        "--subset-root",
        default="data/gqa/subsets/color600",
    )

    p.add_argument(
        "--raw-image-root",
        default="data/gqa/raw/images",
    )

    p.add_argument(
        "--modes",
        default="real,gray",
        help="Comma-separated subset of real,gray",
    )

    p.add_argument(
        "--device",
        default="cuda:0",
    )

    p.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="For a quick test, e.g. 60 or 120.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--print-every",
        type=int,
        default=20,
    )

    p.add_argument(
        "--output-dir",
        default="output/gqa_color_viability",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def configure_processor(model, processor):
    # Same compatibility handling used by the repository backend.
    if hasattr(backend, "configure_processor"):
        backend.configure_processor(model, processor)


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def resolve_image_path(row, subset_root, raw_image_root):
    # First try the subset-local image path.
    image_rel = row.get("image", "")
    if image_rel:
        p = subset_root / image_rel
        if p.exists():
            return p.resolve()

    # Fall back to the full GQA image directory.
    image_id = str(row["image_id"])
    p = raw_image_root / f"{image_id}.jpg"
    if p.exists():
        return p.resolve()

    raise FileNotFoundError(
        f"Cannot find image for image_id={image_id}; "
        f"tried {subset_root / image_rel} and {p}"
    )


def make_image(image, mode):
    if mode == "real":
        return image

    if mode == "gray":
        return ImageOps.grayscale(image).convert("RGB")

    raise ValueError(mode)


def build_prompt(processor, question):
    # Preserve the original GQA question, but standardize the answer space.
    user_text = (
        question.strip()
        + "\nAnswer with exactly one color from: "
        + ", ".join(COLORS)
        + "."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return rendered, user_text


def parse_color(text):
    """
    Extract exactly one of the six target colors from model output.
    Prefer the earliest explicit color word.
    """
    text = text.strip().lower()

    hits = []

    for color in COLORS:
        m = re.search(rf"\b{re.escape(color)}\b", text)
        if m is not None:
            hits.append((m.start(), color))

    if not hits:
        return None

    hits.sort()
    return hits[0][1]


def load_model(model_alias, device, attn_impl):
    if model_alias not in backend.SPECS:
        raise KeyError(
            f"{model_alias} not present in backend.SPECS. "
            f"Available={sorted(backend.SPECS)}"
        )

    spec = backend.SPECS[model_alias]

    model_cls = getattr(transformers, spec.model_class, None)

    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"does not provide {spec.model_class}"
        )

    kwargs = dict(
        torch_dtype=resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": device},
    )

    if attn_impl != "none":
        kwargs["attn_implementation"] = attn_impl

    print("=" * 80)
    print(f"[LOAD] alias      = {model_alias}")
    print(f"[LOAD] repo       = {spec.repo_id}")
    print(f"[LOAD] class      = {spec.model_class}")
    print(f"[LOAD] dtype      = {spec.dtype_name}")
    print(f"[LOAD] transformers = {transformers.__version__}")
    print("=" * 80, flush=True)

    try:
        model = model_cls.from_pretrained(
            spec.repo_id,
            **kwargs,
        )
    except TypeError:
        # Some backends may reject attn_implementation.
        kwargs.pop("attn_implementation", None)
        model = model_cls.from_pretrained(
            spec.repo_id,
            **kwargs,
        )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )

    configure_processor(model, processor)

    # Ensure deterministic greedy generation.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)

    return model, processor, spec


@torch.inference_mode()
def generate_one(
    model,
    processor,
    image,
    question,
    device,
    max_new_tokens,
):
    rendered, user_text = build_prompt(
        processor,
        question,
    )

    batch = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
    )

    batch = move_batch(
        batch,
        torch.device(device),
    )

    input_len = int(batch["input_ids"].shape[1])

    generated = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    new_tokens = generated[:, input_len:]

    generation = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    pred = parse_color(generation)

    return {
        "prompt": user_text,
        "generation": generation,
        "pred": pred,
    }


def compute_summary(rows, model_alias, repo_id, mode):
    valid = [
        r for r in rows
        if r["mode"] == mode
    ]

    n = len(valid)
    correct = sum(int(r["correct"]) for r in valid)
    parseable = sum(r["pred"] is not None for r in valid)

    per_color = {}

    for color in COLORS:
        subset = [
            r for r in valid
            if r["gt"] == color
        ]

        c = sum(int(r["correct"]) for r in subset)

        per_color[color] = {
            "n": len(subset),
            "correct": c,
            "accuracy": (
                c / len(subset)
                if subset
                else None
            ),
        }

    pred_counts = Counter(
        r["pred"] if r["pred"] is not None else "<unparsed>"
        for r in valid
    )

    return {
        "model": model_alias,
        "repo_id": repo_id,
        "mode": mode,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "wrong": n - correct,
        "parseable": parseable,
        "parse_rate": parseable / n if n else None,
        "per_color": per_color,
        "prediction_counts": dict(pred_counts),
    }


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable.")

    modes = [
        x.strip()
        for x in args.modes.split(",")
        if x.strip()
    ]

    for mode in modes:
        if mode not in {"real", "gray"}:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

    subset_path = Path(args.subset_jsonl)
    subset_root = Path(args.subset_root)
    raw_image_root = Path(args.raw_image_root)

    rows = load_jsonl(subset_path)

    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    print(
        f"[DATA] {subset_path} | "
        f"N={len(rows)} | "
        f"modes={modes}"
    )

    print(
        "[GT COUNTS]",
        dict(Counter(r["answer"] for r in rows)),
    )

    # Verify all images before loading a 13B model.
    resolved_paths = {}

    for row in rows:
        image_id = str(row["image_id"])
        resolved_paths[image_id] = resolve_image_path(
            row,
            subset_root,
            raw_image_root,
        )

    print(
        f"[IMAGES] resolved {len(resolved_paths)}/{len(rows)}"
    )

    output_root = (
        Path(args.output_dir)
        / args.model
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_path = output_root / "samples.jsonl"
    summary_path = output_root / "summary.json"

    if samples_path.exists():
        if args.overwrite:
            samples_path.unlink()
        else:
            raise FileExistsError(
                f"{samples_path} exists. "
                f"Use --overwrite."
            )

    model = None
    processor = None

    started = time.time()

    results = []

    try:
        model, processor, spec = load_model(
            args.model,
            args.device,
            args.attn_impl,
        )

        for sample_idx, row in enumerate(
            tqdm(rows, desc=args.model),
            start=1,
        ):
            image_path = resolved_paths[
                str(row["image_id"])
            ]

            real_image = Image.open(
                image_path
            ).convert("RGB")

            gt = str(
                row["answer"]
            ).strip().lower()

            for mode in modes:
                image = make_image(
                    real_image,
                    mode,
                )

                try:
                    generated = generate_one(
                        model=model,
                        processor=processor,
                        image=image,
                        question=row["question"],
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                    )

                    pred = generated["pred"]

                    out = {
                        "sample_index": sample_idx - 1,
                        "question_id": row["question_id"],
                        "image_id": row["image_id"],
                        "image_path": str(image_path),
                        "question": row["question"],
                        "gt": gt,
                        "mode": mode,
                        "pred": pred,
                        "correct": pred == gt,
                        "generation": generated["generation"],
                        "prompt": generated["prompt"],
                        "semanticStr": row.get(
                            "semanticStr",
                            "",
                        ),
                    }

                except Exception as exc:
                    out = {
                        "sample_index": sample_idx - 1,
                        "question_id": row["question_id"],
                        "image_id": row["image_id"],
                        "image_path": str(image_path),
                        "question": row["question"],
                        "gt": gt,
                        "mode": mode,
                        "pred": None,
                        "correct": False,
                        "generation": "",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

                results.append(out)

                with samples_path.open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        json.dumps(
                            out,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if (
                args.print_every > 0
                and sample_idx % args.print_every == 0
            ):
                print()

                for mode in modes:
                    subset = [
                        r for r in results
                        if r["mode"] == mode
                    ]

                    if subset:
                        acc = sum(
                            int(r["correct"])
                            for r in subset
                        ) / len(subset)

                        print(
                            f"[{args.model}] "
                            f"{sample_idx}/{len(rows)} "
                            f"{mode}: "
                            f"acc={acc:.4f}"
                        )

                print(flush=True)

            del real_image

            if (
                torch.cuda.is_available()
                and sample_idx % 25 == 0
            ):
                torch.cuda.empty_cache()

        summaries = {}

        for mode in modes:
            summaries[mode] = compute_summary(
                results,
                args.model,
                spec.repo_id,
                mode,
            )

        real_acc = (
            summaries["real"]["accuracy"]
            if "real" in summaries
            else None
        )

        gray_acc = (
            summaries["gray"]["accuracy"]
            if "gray" in summaries
            else None
        )

        if (
            real_acc is not None
            and gray_acc is not None
        ):
            vision_gain = real_acc - gray_acc
        else:
            vision_gain = None

        final_summary = {
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": "GQA-Color",
            "subset_jsonl": str(subset_path),
            "classes": list(COLORS),
            "n_questions": len(rows),
            "modes": modes,
            "summary_by_mode": summaries,
            "real_minus_gray": vision_gain,
            "elapsed_minutes": (
                time.time() - started
            ) / 60.0,
        }

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                final_summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n" + "=" * 80)
        print("GQA COLOR VIABILITY")
        print("=" * 80)

        for mode in modes:
            s = summaries[mode]

            print(
                f"{args.model:12s} | "
                f"{mode:4s} | "
                f"N={s['n']:3d} | "
                f"acc={s['accuracy']:.4f} | "
                f"wrong={s['wrong']:3d} | "
                f"parse={s['parse_rate']:.4f}"
            )

            print("  per-color:")
            for color in COLORS:
                x = s["per_color"][color]
                if x["n"]:
                    print(
                        f"    {color:6s}: "
                        f"{x['accuracy']:.4f} "
                        f"({x['correct']}/{x['n']})"
                    )

        if vision_gain is not None:
            print(
                f"\nREAL - GRAY = "
                f"{vision_gain:+.4f}"
            )

        print(
            f"\nSaved: {samples_path}"
        )
        print(
            f"Saved: {summary_path}"
        )

    finally:
        if model is not None:
            del model

        if processor is not None:
            del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
