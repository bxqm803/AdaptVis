import os
import re
import csv
import json
import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)


def load_manifest(path, fresh_limit=-1):
    data = json.load(open(path, "r", encoding="utf-8"))

    if isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        raise TypeError(f"Unsupported manifest type: {type(data)}")

    def get_idx(r):
        return int(r.get("sample_idx", r.get("sample_id", r.get("sid", 0))))

    records = sorted(records, key=get_idx)

    if fresh_limit > 0:
        records = records[:fresh_limit]

    return records


def get_value(rec, keys, default=""):
    for k in keys:
        if k in rec and rec[k] not in [None, ""]:
            return rec[k]
    return default


def norm_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def raw_generation_correct(gold, gen):
    gold = norm_gold(gold)
    gen = str(gen)

    ok = (gold in gen) or (gold.lower() in gen.lower())

    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False

    return bool(ok)


def strip_prompt(raw_prompt):
    q = str(raw_prompt)
    q = q.replace("<image>", "").strip()
    q = re.sub(r"^USER:\s*", "", q, flags=re.I).strip()
    q = re.sub(r"ASSISTANT:\s*$", "", q, flags=re.I).strip()
    return q


def build_prompt(processor, raw_prompt):
    question = strip_prompt(raw_prompt)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return "<image>\nUSER: " + question + "\nASSISTANT:"


def generation_scores(output):
    if hasattr(output, "scores"):
        return output.scores
    if isinstance(output, dict):
        return output.get("scores", None)
    return output["scores"]


def generation_sequences(output):
    if hasattr(output, "sequences"):
        return output.sequences
    if isinstance(output, dict):
        return output.get("sequences", None)
    return output["sequences"]


def first_step_confidence(output):
    scores = generation_scores(output)
    if scores is None or len(scores) == 0:
        return 0.0

    prob = torch.nn.functional.softmax(scores[0], dim=-1)
    return float(torch.max(prob[0]).detach().float().cpu())


def decode_generated(processor, output, prompt_len):
    seq = generation_sequences(output)
    return processor.decode(
        seq[0][int(prompt_len):],
        skip_special_tokens=True,
    ).strip()


class AdaptVisContext:
    def __init__(self, layers, num_layers, intervention, mean_scale, mul_factor):
        self.active = False
        self.in_lm = False

        self.layers = set(int(x) for x in layers)
        self.num_layers = int(num_layers)
        self.intervention = str(intervention).strip().lower()
        self.mean_scale = float(mean_scale)
        self.mul_factor = float(mul_factor)

        if self.intervention not in ["mean", "mul"]:
            raise ValueError(f"Unknown intervention={self.intervention}; use mean or mul")

        self.call_idx = 0
        self.modified_calls = 0

        self.input_len = None
        self.image_positions = []
        self.image_token_index = None

    def reset_for_sample(self, input_ids, image_token_index):
        self.call_idx = 0
        self.modified_calls = 0

        self.input_len = int(input_ids.shape[-1])
        self.image_token_index = int(image_token_index)

        ids = input_ids[0]
        pos = torch.where(ids == int(image_token_index))[0]
        self.image_positions = [int(x.detach().cpu()) for x in pos]

    def get_image_span(self, kv_len):
        if not self.image_positions:
            return None

        first = min(self.image_positions)
        last = max(self.image_positions)
        n_placeholder = len(self.image_positions)

        # Case 1: processor expanded image tokens in input_ids already.
        if int(kv_len) == int(self.input_len):
            start = first
            end = last + 1
            return start, end

        # Case 2: model expands one or more <image> placeholders internally.
        image_len = int(kv_len) - (int(self.input_len) - int(n_placeholder))
        if image_len <= 0:
            return None

        start = first
        end = start + image_len

        if start < 0 or end > int(kv_len) or end <= start:
            return None

        return start, end


def apply_adaptvis_to_image_logits(attn_logits, ctx):
    """
    attn_logits: [bsz, heads, q_len, kv_len]

    Supported interventions on selected image-key logits in selected LM layers:

    1) mean:
       negative logits += mean_scale * abs(mean(selected image logits)).
       This matches the LLaVA-1.5 negonly_mean_img formula, except this
       LLaVA-1.6 script currently applies it to the whole image-token span
       rather than a 24x24 patch-id subset.

    2) mul:
       negative logits *= mul_factor.
       For mul_factor < 1, negative logits move toward zero, e.g. -8 * 0.5 = -4.
    """
    if not ctx.active:
        return attn_logits

    if not ctx.in_lm:
        return attn_logits

    if not torch.is_tensor(attn_logits):
        return attn_logits

    if attn_logits.dim() != 4:
        return attn_logits

    q_len = attn_logits.shape[-2]
    kv_len = attn_logits.shape[-1]

    # Only modify prefill. Decode q_len=1 is skipped.
    if q_len <= 1:
        return attn_logits

    layer = ctx.call_idx % ctx.num_layers
    ctx.call_idx += 1

    if layer not in ctx.layers:
        return attn_logits

    span = ctx.get_image_span(kv_len)
    if span is None:
        return attn_logits

    image_start, image_end = span
    image_start = max(0, int(image_start))
    image_end = min(int(kv_len), int(image_end))

    if image_end <= image_start:
        return attn_logits

    x = attn_logits.clone()
    region = x[..., :, image_start:image_end]
    neg_mask = region < 0

    if ctx.intervention == "mean":
        mean_abs = region.detach().float().mean(dim=-1, keepdim=True).abs()
        mean_abs = mean_abs.to(dtype=region.dtype, device=region.device)
        region_new = torch.where(
            neg_mask,
            region + float(ctx.mean_scale) * mean_abs,
            region,
        )
    elif ctx.intervention == "mul":
        region_new = torch.where(
            neg_mask,
            region * float(ctx.mul_factor),
            region,
        )
    else:
        raise ValueError(f"Unknown intervention={ctx.intervention}")

    x[..., :, image_start:image_end] = region_new
    ctx.modified_calls += 1

    return x


@contextmanager
def patch_language_model_and_softmax(model, ctx):
    old_f_softmax = F.softmax
    old_torch_softmax = torch.softmax

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise AttributeError("model has no language_model")

    old_lm_forward = language_model.forward

    def wrapped_lm_forward(*args, **kwargs):
        old_flag = ctx.in_lm
        ctx.in_lm = True
        try:
            return old_lm_forward(*args, **kwargs)
        finally:
            ctx.in_lm = old_flag

    def wrapped_f_softmax(input, *args, **kwargs):
        dim = kwargs.get("dim", None)
        if dim is None and len(args) > 0:
            dim = args[0]

        if dim == -1 or dim == input.dim() - 1:
            input = apply_adaptvis_to_image_logits(input, ctx)

        return old_f_softmax(input, *args, **kwargs)

    def wrapped_torch_softmax(input, *args, **kwargs):
        dim = kwargs.get("dim", None)
        if dim is None and len(args) > 0:
            dim = args[0]

        if dim == -1 or dim == input.dim() - 1:
            input = apply_adaptvis_to_image_logits(input, ctx)

        return old_torch_softmax(input, *args, **kwargs)

    language_model.forward = wrapped_lm_forward
    F.softmax = wrapped_f_softmax
    torch.softmax = wrapped_torch_softmax

    try:
        yield
    finally:
        language_model.forward = old_lm_forward
        F.softmax = old_f_softmax
        torch.softmax = old_torch_softmax


def load_llava16(model_name, cache_dir, device):
    image_processor = LlavaNextImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False,
    )

    processor = LlavaNextProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    model = model.eval().to(device)
    return processor, model


def run_one(processor, model, image, raw_prompt, device, max_new_tokens, ctx=None, active=False):
    prompt = build_prompt(processor, raw_prompt)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(device)

    prompt_len = inputs["input_ids"].shape[-1]

    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32000

    if ctx is not None:
        ctx.reset_for_sample(inputs["input_ids"], image_token_index)
        ctx.active = bool(active)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    gen = decode_generated(processor, output, prompt_len)
    conf = first_step_confidence(output)

    return gen, conf, prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default="")

    parser.add_argument("--model", default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--layers", default="0,1,2,3,4")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument(
        "--intervention",
        default="mean",
        choices=["mean", "mul"],
        help="mean: negative image logits += mean_scale * abs(mean); mul: negative image logits *= mul_factor.",
    )
    parser.add_argument("--mean-scale", type=float, default=1.0)
    parser.add_argument("--mul-factor", type=float, default=0.5)

    parser.add_argument("--fresh-limit", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--print-each-sample", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    layers = [int(x) for x in str(args.layers).split(",") if x.strip()]

    print("[LOAD MANIFEST]", args.manifest_json)
    records = load_manifest(args.manifest_json, fresh_limit=args.fresh_limit)
    print("[NUM RECORDS]", len(records))

    print("[LOAD LLaVA-1.6]", args.model)
    processor, model = load_llava16(args.model, args.cache_dir, args.device)

    ctx = AdaptVisContext(
        layers=layers,
        num_layers=args.num_layers,
        intervention=args.intervention,
        mean_scale=args.mean_scale,
        mul_factor=args.mul_factor,
    )

    base_records = []
    mixed_records = []

    base_correct = 0
    mixed_correct = 0
    adapt_count = 0
    modified_call_total = 0

    print("\n" + "=" * 80)
    print("[PASS 1] BASE")
    print("=" * 80)

    for i, rec in enumerate(tqdm(records, desc="llava1.6 base")):
        sid = int(get_value(rec, ["sample_id", "sid", "idx", "index"], i))
        img_path = get_value(rec, ["processed_image_path", "image_path", "img_path", "path"])
        raw_prompt = get_value(rec, ["prompt", "question", "text", "query"])
        gold = norm_gold(get_value(rec, ["gold", "answer", "label"]))

        if not img_path or not os.path.exists(img_path):
            print("[SKIP missing image]", sid, img_path)
            continue

        image = Image.open(img_path).convert("RGB")

        with patch_language_model_and_softmax(model, ctx):
            gen, conf, hf_prompt = run_one(
                processor=processor,
                model=model,
                image=image,
                raw_prompt=raw_prompt,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                ctx=ctx,
                active=False,
            )

        corr = raw_generation_correct(gold, gen)
        base_correct += int(corr)

        row = {
            "sample_id": sid,
            "image_path": img_path,
            "gold": gold,
            "base_generation": gen,
            "base_correct": bool(corr),
            "base_confidence": float(conf),
            "prompt": raw_prompt,
            "hf_prompt": hf_prompt,
        }
        base_records.append(row)

        if args.print_each_sample or i < 5:
            print(
                f"[BASE] sid={sid} gold={gold} gen={gen} "
                f"correct={corr} conf={conf:.4f} "
                f"running_acc={base_correct / max(len(base_records), 1):.4f}"
            )

    print("\n" + "=" * 80)
    print("[PASS 2] MIXED BASE + ADAPTVIS")
    print("[GATE] base_confidence < ", args.threshold)
    print("[LAYERS]", layers)
    print("[INTERVENTION]", args.intervention)
    print("[MEAN_SCALE]", args.mean_scale)
    print("[MUL_FACTOR]", args.mul_factor)
    print("=" * 80)

    base_by_sid = {int(r["sample_id"]): r for r in base_records}

    for i, rec in enumerate(tqdm(records, desc="llava1.6 mixed")):
        sid = int(get_value(rec, ["sample_id", "sid", "idx", "index"], i))
        img_path = get_value(rec, ["processed_image_path", "image_path", "img_path", "path"])
        raw_prompt = get_value(rec, ["prompt", "question", "text", "query"])
        gold = norm_gold(get_value(rec, ["gold", "answer", "label"]))

        if sid not in base_by_sid:
            continue

        base_r = base_by_sid[sid]
        base_conf = float(base_r["base_confidence"])
        do_adapt = base_conf < float(args.threshold)

        if not do_adapt:
            gen = base_r["base_generation"]
            conf = base_r["base_confidence"]
            corr = base_r["base_correct"]
            modified_calls = 0
        else:
            image = Image.open(img_path).convert("RGB")
            adapt_count += 1

            with patch_language_model_and_softmax(model, ctx):
                gen, conf, _ = run_one(
                    processor=processor,
                    model=model,
                    image=image,
                    raw_prompt=raw_prompt,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    ctx=ctx,
                    active=True,
                )
                modified_calls = int(ctx.modified_calls)

            corr = raw_generation_correct(gold, gen)
            modified_call_total += modified_calls

        mixed_correct += int(corr)
        running_acc = mixed_correct / max(len(mixed_records) + 1, 1)

        transition = "same"
        if bool(base_r["base_correct"]) and not bool(corr):
            transition = "correct_to_wrong"
        elif (not bool(base_r["base_correct"])) and bool(corr):
            transition = "wrong_to_correct"
        elif bool(base_r["base_correct"]) and bool(corr):
            transition = "correct_to_correct"
        else:
            transition = "wrong_to_wrong"

        row = {
            "sample_id": sid,
            "image_path": img_path,
            "gold": gold,

            "base_generation": base_r["base_generation"],
            "base_correct": base_r["base_correct"],
            "base_confidence": base_conf,

            "mixed_generation": gen,
            "mixed_correct": bool(corr),
            "mixed_confidence": float(conf),

            "did_adaptvis": bool(do_adapt),
            "modified_softmax_calls": int(modified_calls),
            "threshold": float(args.threshold),
            "layers": args.layers,
            "intervention": args.intervention,
            "mean_scale": float(args.mean_scale),
            "mul_factor": float(args.mul_factor),
            "transition": transition,
            "prompt": raw_prompt,
        }
        mixed_records.append(row)

        if args.print_each_sample or i < 5:
            print(
                f"[MIXED] sid={sid} gate={do_adapt} "
                f"base_conf={base_conf:.4f} modified_calls={modified_calls} "
                f"gold={gold} gen={gen} correct={bool(corr)} "
                f"transition={transition} running_acc={running_acc:.4f}"
            )

    base_acc = sum(int(r["base_correct"]) for r in base_records) / max(len(base_records), 1)
    mixed_acc = sum(int(r["mixed_correct"]) for r in mixed_records) / max(len(mixed_records), 1)

    w2c = sum(1 for r in mixed_records if r["transition"] == "wrong_to_correct")
    c2w = sum(1 for r in mixed_records if r["transition"] == "correct_to_wrong")
    c2c = sum(1 for r in mixed_records if r["transition"] == "correct_to_correct")
    w2w = sum(1 for r in mixed_records if r["transition"] == "wrong_to_wrong")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "image_path",
            "gold",
            "base_generation",
            "base_correct",
            "base_confidence",
            "mixed_generation",
            "mixed_correct",
            "mixed_confidence",
            "did_adaptvis",
            "modified_softmax_calls",
            "threshold",
            "layers",
            "intervention",
            "mean_scale",
            "mul_factor",
            "transition",
            "prompt",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in mixed_records:
            w.writerow(r)

    payload = {
        "model": args.model,
        "manifest_json": args.manifest_json,
        "threshold": args.threshold,
        "layers": args.layers,
        "intervention": args.intervention,
        "mean_scale": args.mean_scale,
        "mul_factor": args.mul_factor,
        "num_total": len(mixed_records),
        "base_acc": base_acc,
        "mixed_acc": mixed_acc,
        "adaptvis_count": adapt_count,
        "modified_call_total": modified_call_total,
        "wrong_to_correct": w2c,
        "correct_to_wrong": c2w,
        "correct_to_correct": c2c,
        "wrong_to_wrong": w2w,
        "records": mixed_records,
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n[DONE]")
    print("[OUT CSV]", args.out_csv)
    if args.out_json:
        print("[OUT JSON]", args.out_json)
    print("num_total:", len(mixed_records))
    print("base_acc:", base_acc)
    print("mixed_acc:", mixed_acc)
    print("adaptvis_count:", adapt_count)
    print("wrong_to_correct:", w2c)
    print("correct_to_wrong:", c2w)
    print("correct_to_correct:", c2c)
    print("wrong_to_wrong:", w2w)
    print("modified_call_total:", modified_call_total)


if __name__ == "__main__":
    main()
