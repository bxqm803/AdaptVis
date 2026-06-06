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


OPTIONS = ["left", "right", "on", "under"]


def option_token_ids(tokenizer, opt):
    """
    Return possible first-token ids for an option.
    We sum variants like "left" and " left" because tokenizer behavior can differ.
    """
    ids = set()
    variants = [
        opt,
        " " + opt,
        opt.capitalize(),
        " " + opt.capitalize(),
        opt.upper(),
        " " + opt.upper(),
    ]

    for v in variants:
        toks = tokenizer(v, add_special_tokens=False).input_ids
        if toks:
            ids.add(int(toks[0]))

    return sorted(ids)


_OPTION_TOKEN_CACHE = {}


def get_option_token_ids_cached(tokenizer, opt):
    key = (id(tokenizer), str(opt))
    if key not in _OPTION_TOKEN_CACHE:
        _OPTION_TOKEN_CACHE[key] = option_token_ids(tokenizer, opt)
    return _OPTION_TOKEN_CACHE[key]


def option_probs_from_output(processor, output):
    """
    Use generation scores[0], i.e. the next-token distribution at the first generated step.
    Returns:
      probs: dict for left/right/on/under
      pred: option with max probability
    """
    scores = generation_scores(output)
    if scores is None or len(scores) == 0:
        probs = {opt: 0.0 for opt in OPTIONS}
        return probs, ""

    first_scores = scores[0][0].float()
    prob = torch.nn.functional.softmax(first_scores, dim=-1)

    out = {}
    for opt in OPTIONS:
        tids = get_option_token_ids_cached(processor.tokenizer, opt)
        out[opt] = float(sum(prob[t].detach().cpu() for t in tids)) if tids else 0.0

    pred = max(out, key=out.get)
    return out, pred


def norm_option_gold(x):
    return norm_gold(x).lower()


def option_correct(gold, pred):
    return norm_option_gold(gold) == str(pred).strip().lower()


def option_prob_fields(prefix, probs):
    return {
        f"{prefix}_prob_{opt}": float(probs.get(opt, 0.0))
        for opt in OPTIONS
    }


def add_delta_prob_fields(row):
    for opt in OPTIONS:
        row[f"delta_prob_{opt}"] = (
            float(row.get(f"mixed_prob_{opt}", 0.0))
            - float(row.get(f"base_prob_{opt}", 0.0))
        )
    return row


def option_summary(records):
    total = len(records)
    if total == 0:
        summary = {
            "n": 0,
            "base_option_acc": 0.0,
            "mixed_option_acc": 0.0,
            "delta_option_acc": 0.0,
        }
        for opt in OPTIONS:
            summary[f"base_avg_prob_{opt}"] = 0.0
            summary[f"mixed_avg_prob_{opt}"] = 0.0
            summary[f"delta_avg_prob_{opt}"] = 0.0
        return summary

    base_correct = sum(int(bool(r.get("base_option_correct", False))) for r in records)
    mixed_correct = sum(int(bool(r.get("mixed_option_correct", False))) for r in records)

    summary = {
        "n": total,
        "base_option_acc": base_correct / total,
        "mixed_option_acc": mixed_correct / total,
        "delta_option_acc": mixed_correct / total - base_correct / total,
    }

    for opt in OPTIONS:
        b = sum(float(r.get(f"base_prob_{opt}", 0.0)) for r in records) / total
        m = sum(float(r.get(f"mixed_prob_{opt}", 0.0)) for r in records) / total
        summary[f"base_avg_prob_{opt}"] = b
        summary[f"mixed_avg_prob_{opt}"] = m
        summary[f"delta_avg_prob_{opt}"] = m - b

    return summary


def option_summary_by_gold(records):
    out = {}
    for gold in OPTIONS:
        sub = [r for r in records if norm_option_gold(r.get("gold", "")) == gold]
        out[gold] = option_summary(sub)
    return out


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
    opt_probs, opt_pred = option_probs_from_output(processor, output)

    return gen, conf, prompt, opt_probs, opt_pred


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
    parser.add_argument(
        "--mul-factor-high",
        type=float,
        default=None,
        help="For --intervention mul: factor used when base_confidence >= threshold. If unset, uses --mul-factor.",
    )
    parser.add_argument(
        "--mul-factor-low",
        type=float,
        default=None,
        help="For --intervention mul: factor used when base_confidence < threshold. If unset, uses --mul-factor.",
    )

    parser.add_argument("--fresh-limit", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--print-each-sample", action="store_true")
    args = parser.parse_args()

    if args.mul_factor_high is None:
        args.mul_factor_high = float(args.mul_factor)
    if args.mul_factor_low is None:
        args.mul_factor_low = float(args.mul_factor)

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
            gen, conf, hf_prompt, base_opt_probs, base_opt_pred = run_one(
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
        base_opt_corr = option_correct(gold, base_opt_pred)
        base_correct += int(corr)

        row = {
            "sample_id": sid,
            "image_path": img_path,
            "gold": gold,
            "base_generation": gen,
            "base_correct": bool(corr),
            "base_confidence": float(conf),
            "base_option_pred": base_opt_pred,
            "base_option_correct": bool(base_opt_corr),
            **option_prob_fields("base", base_opt_probs),
            "prompt": raw_prompt,
            "hf_prompt": hf_prompt,
        }
        base_records.append(row)

        if args.print_each_sample or i < 5:
            print(
                f"[BASE] sid={sid} gold={gold} gen={gen} "
                f"correct={corr} conf={conf:.4f} "
                f"option_pred={base_opt_pred} option_correct={base_opt_corr} "
                f"running_acc={base_correct / max(len(base_records), 1):.4f}"
            )

    print("\n" + "=" * 80)
    print("[PASS 2] MIXED BASE + ADAPTVIS")
    print("[GATE] base_confidence < ", args.threshold)
    print("[LAYERS]", layers)
    print("[INTERVENTION]", args.intervention)
    print("[MEAN_SCALE]", args.mean_scale)
    print("[MUL_FACTOR]", args.mul_factor)
    print("[MUL_FACTOR_HIGH]", args.mul_factor_high)
    print("[MUL_FACTOR_LOW]", args.mul_factor_low)
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

        selected_mul_factor = ""
        gate_branch = "none"

        if args.intervention == "mul":
            # For multiplication, always run AdaptVis, but choose factor by confidence:
            # base_confidence >= threshold -> w1 / high factor
            # base_confidence <  threshold -> w2 / low factor
            do_adapt = True
            if base_conf >= float(args.threshold):
                selected_mul_factor = float(args.mul_factor_high)
                gate_branch = "high_conf_ge_threshold"
            else:
                selected_mul_factor = float(args.mul_factor_low)
                gate_branch = "low_conf_lt_threshold"
            ctx.mul_factor = float(selected_mul_factor)
        else:
            # For mean, keep the old gate: only low-confidence samples are re-run.
            do_adapt = base_conf < float(args.threshold)
            selected_mul_factor = float(ctx.mul_factor)
            gate_branch = "low_conf_lt_threshold" if do_adapt else "high_conf_ge_threshold"

        if not do_adapt:
            gen = base_r["base_generation"]
            conf = base_r["base_confidence"]
            corr = base_r["base_correct"]
            mixed_opt_pred = base_r.get("base_option_pred", "")
            mixed_opt_corr = base_r.get("base_option_correct", False)
            mixed_opt_probs = {
                opt: float(base_r.get(f"base_prob_{opt}", 0.0))
                for opt in OPTIONS
            }
            modified_calls = 0
        else:
            image = Image.open(img_path).convert("RGB")
            adapt_count += 1

            with patch_language_model_and_softmax(model, ctx):
                gen, conf, _, mixed_opt_probs, mixed_opt_pred = run_one(
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
            mixed_opt_corr = option_correct(gold, mixed_opt_pred)
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
            "base_option_pred": base_r.get("base_option_pred", ""),
            "base_option_correct": bool(base_r.get("base_option_correct", False)),
            **option_prob_fields(
                "base",
                {opt: float(base_r.get(f"base_prob_{opt}", 0.0)) for opt in OPTIONS},
            ),

            "mixed_generation": gen,
            "mixed_correct": bool(corr),
            "mixed_confidence": float(conf),
            "mixed_option_pred": mixed_opt_pred,
            "mixed_option_correct": bool(mixed_opt_corr),
            **option_prob_fields("mixed", mixed_opt_probs),

            "did_adaptvis": bool(do_adapt),
            "modified_softmax_calls": int(modified_calls),
            "threshold": float(args.threshold),
            "layers": args.layers,
            "intervention": args.intervention,
            "mean_scale": float(args.mean_scale),
            "mul_factor": float(args.mul_factor),
            "mul_factor_high": float(args.mul_factor_high),
            "mul_factor_low": float(args.mul_factor_low),
            "selected_mul_factor": selected_mul_factor,
            "gate_branch": gate_branch,
            "transition": transition,
            "prompt": raw_prompt,
        }
        mixed_records.append(row)

        if args.print_each_sample or i < 5:
            print(
                f"[MIXED] sid={sid} gate={do_adapt} branch={gate_branch} "
                f"base_conf={base_conf:.4f} selected_mul={selected_mul_factor} "
                f"modified_calls={modified_calls} "
                f"gold={gold} gen={gen} correct={bool(corr)} "
                f"option_pred={mixed_opt_pred} option_correct={mixed_opt_corr} "
                f"transition={transition} running_acc={running_acc:.4f}"
            )

    for r in mixed_records:
        add_delta_prob_fields(r)

    base_acc = sum(int(r["base_correct"]) for r in base_records) / max(len(base_records), 1)
    mixed_acc = sum(int(r["mixed_correct"]) for r in mixed_records) / max(len(mixed_records), 1)

    base_option_acc = (
        sum(int(bool(r.get("base_option_correct", False))) for r in mixed_records)
        / max(len(mixed_records), 1)
    )
    mixed_option_acc = (
        sum(int(bool(r.get("mixed_option_correct", False))) for r in mixed_records)
        / max(len(mixed_records), 1)
    )

    opt_summary = option_summary(mixed_records)
    opt_summary_by_gold = option_summary_by_gold(mixed_records)

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
            "base_option_pred",
            "base_option_correct",
            "base_prob_left",
            "base_prob_right",
            "base_prob_on",
            "base_prob_under",
            "mixed_generation",
            "mixed_correct",
            "mixed_confidence",
            "mixed_option_pred",
            "mixed_option_correct",
            "mixed_prob_left",
            "mixed_prob_right",
            "mixed_prob_on",
            "mixed_prob_under",
            "delta_prob_left",
            "delta_prob_right",
            "delta_prob_on",
            "delta_prob_under",
            "did_adaptvis",
            "modified_softmax_calls",
            "threshold",
            "layers",
            "intervention",
            "mean_scale",
            "mul_factor",
            "mul_factor_high",
            "mul_factor_low",
            "selected_mul_factor",
            "gate_branch",
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
        "mul_factor_high": args.mul_factor_high,
        "mul_factor_low": args.mul_factor_low,
        "num_total": len(mixed_records),
        "base_acc": base_acc,
        "mixed_acc": mixed_acc,
        "base_option_acc": base_option_acc,
        "mixed_option_acc": mixed_option_acc,
        "option_summary": opt_summary,
        "option_summary_by_gold": opt_summary_by_gold,
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
    print("base_option_acc:", base_option_acc)
    print("mixed_option_acc:", mixed_option_acc)
    print("[OPTION SUMMARY]", opt_summary)
    print("[OPTION SUMMARY BY GOLD]", opt_summary_by_gold)
    print("adaptvis_count:", adapt_count)
    print("wrong_to_correct:", w2c)
    print("correct_to_wrong:", c2w)
    print("correct_to_correct:", c2c)
    print("wrong_to_wrong:", w2w)
    print("modified_call_total:", modified_call_total)


if __name__ == "__main__":
    main()
