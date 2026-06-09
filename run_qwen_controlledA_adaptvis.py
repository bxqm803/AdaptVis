import argparse
import json
import os
import re
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation import GenerationConfig
except Exception:
    GenerationConfig = None

from dataset_zoo.aro_datasets import Controlled_Images


def setup_cache():
    os.environ.setdefault("HF_HOME", "/ddnB/work/mwang32/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/ddnB/work/mwang32/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/ddnB/work/mwang32/hf_cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/ddnB/work/mwang32/hf_cache/datasets")
    os.environ.setdefault("TORCH_HOME", "/ddnB/work/mwang32/torch_cache")

    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]:
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def clean_prompt_for_qwen(prompt):
    prompt = str(prompt)
    prompt = prompt.replace("<image>", "")
    prompt = prompt.replace("USER:", "").replace("User:", "").replace("user:", "")
    prompt = prompt.replace("ASSISTANT:", "").replace("Assistant:", "").replace("assistant:", "")
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


def parse_prep(text):
    t = str(text).lower()
    t = re.sub(r"[^a-z\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if re.search(r"\bleft\b", t):
        return "left"
    if re.search(r"\bright\b", t):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b", t):
        return "under"
    if re.search(r"\bon\b|\btop\b|\babove\b|\bover\b", t):
        return "on"
    return None


def resolve_image_path(p):
    p = str(p)
    if os.path.exists(p):
        return p

    base = os.path.basename(p)
    candidates = [
        os.path.join("data", "controlled_images", base),
        os.path.join("data", base),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    hits = list(Path("data").rglob(base))
    if hits:
        return str(hits[0])

    raise FileNotFoundError(p)


def get_device(model):
    return next(model.parameters()).device


def get_img_token_ids(tokenizer):
    img_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_id = tokenizer.convert_tokens_to_ids("</img>")
    if img_id is None or img_end_id is None:
        raise RuntimeError("Cannot find <img> or </img> token ids.")
    return int(img_id), int(img_end_id)


def find_image_span(input_ids, tokenizer):
    img_id, img_end_id = get_img_token_ids(tokenizer)
    ids = input_ids[0].tolist()

    starts = [i for i, x in enumerate(ids) if x == img_id]
    ends = [i for i, x in enumerate(ids) if x == img_end_id]

    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError(f"Bad image tokens: starts={starts}, ends={ends}")

    st = starts[0] + 1
    ed = ends[0]

    if ed <= st:
        raise RuntimeError(f"Bad image span: [{st}, {ed})")

    return st, ed


def patch_qwen_attention(model, debug=False):
    """
    Post-softmax image-key scaling:
        attn_probs[..., image_start:image_end] *= weight
        renormalize over key dimension
        attn_output = attn_probs @ value

    This does not train anything and does not edit model files.
    """

    patched = 0

    for name, module in model.named_modules():
        if module.__class__.__name__ != "QWenAttention":
            continue
        if not hasattr(module, "_attn"):
            continue

        old_attn = module._attn

        def make_patched(old_attn_func, module_name):
            def patched_attn(self, *args, **kwargs):
                out = old_attn_func(*args, **kwargs)

                if not getattr(model, "_adaptvis_enable", False):
                    return out

                weight = float(getattr(model, "_adaptvis_weight", 1.0))
                spans = getattr(model, "_adaptvis_image_spans", None)

                if weight == 1.0 or not spans:
                    return out

                if not isinstance(out, tuple) or len(out) < 2:
                    return out

                value = args[2] if len(args) >= 3 else kwargs.get("value", None)
                if value is None:
                    return out

                attn_output, attn_probs = out[0], out[1]

                if attn_probs is None:
                    return out

                # attn_probs: [bsz, heads, q_len, kv_len]
                if attn_probs.dim() != 4:
                    return out

                bsz, n_heads, q_len, kv_len = attn_probs.shape

                scaled = attn_probs.clone()

                for b in range(min(bsz, len(spans))):
                    st, ed = spans[b]
                    st = int(st)
                    ed = int(ed)

                    if st < 0:
                        st = 0
                    if ed > kv_len:
                        ed = kv_len

                    if ed <= st:
                        continue

                    scaled[b, :, :, st:ed] = scaled[b, :, :, st:ed] * weight

                denom = scaled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                scaled = scaled / denom

                new_attn_output = torch.matmul(scaled, value)

                if len(out) == 2:
                    return new_attn_output, scaled
                else:
                    return (new_attn_output, scaled) + tuple(out[2:])

            return patched_attn

        module._attn = types.MethodType(make_patched(old_attn, name), module)
        patched += 1

        if debug:
            print("patched:", name)

    if patched == 0:
        raise RuntimeError("No QWenAttention._attn modules patched.")

    print(f"Patched QWenAttention modules: {patched}")


def build_query(tokenizer, image_path, prompt):
    return tokenizer.from_list_format([
        {"image": image_path},
        {"text": prompt},
    ])


def encode_query(tokenizer, query, model):
    enc = tokenizer(query, return_tensors="pt")
    device = get_device(model)
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def candidate_token_ids(tokenizer):
    cands = {
        "left": " left",
        "right": " right",
        "on": " on",
        "under": " under",
    }

    out = {}
    for k, s in cands.items():
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) < 1:
            raise RuntimeError(f"Cannot tokenize candidate {s}")
        out[k] = int(ids[0])

    return out


@torch.no_grad()
def get_candidate_confidence(model, tokenizer, enc, span, cand_ids):
    model._adaptvis_enable = True
    model._adaptvis_weight = 1.0
    model._adaptvis_image_spans = [span]

    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask", None),
        use_cache=False,
        return_dict=True,
    )

    logits = out.logits[0, -1, :]
    labels = ["left", "right", "on", "under"]
    ids = torch.tensor([cand_ids[x] for x in labels], device=logits.device)

    cand_logits = logits[ids]
    probs = F.softmax(cand_logits.float(), dim=-1)

    best_idx = int(torch.argmax(probs).item())
    conf = float(probs[best_idx].item())

    return {
        "confidence": conf,
        "best_candidate": labels[best_idx],
        "candidate_probs": {labels[i]: float(probs[i].item()) for i in range(len(labels))},
    }


@torch.no_grad()
def generate_response(model, tokenizer, enc, span, weight, max_new_tokens):
    model._adaptvis_enable = True
    model._adaptvis_weight = float(weight)
    model._adaptvis_image_spans = [span]

    input_len = enc["input_ids"].shape[-1]

    gen = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask", None),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=getattr(tokenizer, "eod_id", None),
    )

    new_ids = gen[0, input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    text = str(text).strip()
    return text


def pred_index_from_generation(gen, gold_answer, caption_options):
    pred = parse_prep(gen)
    gold = parse_prep(gold_answer)

    if pred is not None and gold is not None and pred == gold:
        return 0, pred

    if pred is not None:
        for i, cap in enumerate(caption_options):
            if parse_prep(cap) == pred:
                return i, pred

    return 2 if len(caption_options) >= 3 else 1, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--subset", default="A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--method", choices=["base", "scaling_vis", "adapt_vis"], default="adapt_vis")

    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--weight1", type=float, default=0.5)
    parser.add_argument("--weight2", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.4)

    # high_w1: confidence >= threshold -> weight1, else weight2
    # high_w2: confidence >= threshold -> weight2, else weight1
    parser.add_argument("--adapt_rule", choices=["high_w1", "high_w2"], default="high_w1")

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--debug_patch", action="store_true")
    parser.add_argument("--print_every", type=int, default=1)
    args = parser.parse_args()

    setup_cache()
    Path("outputs").mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = Controlled_Images(
        image_preprocess=None,
        root_dir=args.root_dir,
        download=True,
        subset=args.subset,
    )

    prompt_file = f"prompts/{args.dataset}_with_answer_{args.option}_options.jsonl"

    prompts = []
    answers = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["question"])
            answers.append(r["answer"])

    n_total = len(dataset.dataset)
    if args.limit > 0:
        n_total = min(n_total, args.limit)

    print("dataset total:", len(dataset.dataset))
    print("running n:", n_total)
    print("method:", args.method)
    print("weight:", args.weight)
    print("weight1:", args.weight1)
    print("weight2:", args.weight2)
    print("threshold:", args.threshold)
    print("adapt_rule:", args.adapt_rule)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    print("Loading Qwen model...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            use_flash_attn=False,
            **load_kwargs,
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **load_kwargs,
        ).eval()

    if GenerationConfig is not None:
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                args.model_id,
                trust_remote_code=True,
            )
        except Exception as e:
            print("GenerationConfig skipped:", repr(e))

    model.requires_grad_(False)

    patch_qwen_attention(model, debug=args.debug_patch)

    cand_ids = candidate_token_ids(tokenizer)
    print("candidate token ids:", cand_ids)

    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    correct = 0

    out_tag = (
        f"qwen_vl_chat_{args.dataset}_{args.method}"
        f"_w{args.weight}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}"
    )
    out_json = Path("outputs") / f"{out_tag}_generations.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    for i in tqdm(range(n_total), total=n_total):
        d = dataset.dataset[i]
        image_path = resolve_image_path(d["image_path"])
        caption_options = d["caption_options"]

        prompt = clean_prompt_for_qwen(prompts[i])
        gold = answers[i]

        query = build_query(tokenizer, image_path, prompt)
        enc = encode_query(tokenizer, query, model)
        span = find_image_span(enc["input_ids"], tokenizer)

        conf_info = get_candidate_confidence(
            model=model,
            tokenizer=tokenizer,
            enc=enc,
            span=span,
            cand_ids=cand_ids,
        )

        if args.method == "base":
            chosen_weight = 1.0
        elif args.method == "scaling_vis":
            chosen_weight = args.weight
        else:
            conf = conf_info["confidence"]
            if args.adapt_rule == "high_w1":
                chosen_weight = args.weight1 if conf >= args.threshold else args.weight2
            else:
                chosen_weight = args.weight2 if conf >= args.threshold else args.weight1

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            enc=enc,
            span=span,
            weight=chosen_weight,
            max_new_tokens=args.max_new_tokens,
        )

        pred_idx, pred_prep = pred_index_from_generation(
            response,
            gold,
            caption_options,
        )

        scores[i, 0, pred_idx] = 1.0

        is_correct = int(pred_idx == 0)
        correct += is_correct

        rec = {
            "index": i,
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "generation": response,
            "pred_prep": pred_prep,
            "pred_idx": int(pred_idx),
            "correct": bool(is_correct),
            "confidence": conf_info["confidence"],
            "base_best_candidate": conf_info["best_candidate"],
            "candidate_probs": conf_info["candidate_probs"],
            "chosen_weight": float(chosen_weight),
            "image_span": [int(span[0]), int(span[1])],
            "caption_options": caption_options,
        }
        records.append(rec)

        if args.print_every > 0 and (i % args.print_every == 0):
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("gold:", gold)
            print("conf:", conf_info["confidence"], "base_best:", conf_info["best_candidate"])
            print("chosen_weight:", chosen_weight, "span:", span)
            print("generation:", response)
            print("pred:", pred_prep, "pred_idx:", pred_idx, "correct:", bool(is_correct))
            print("running acc:", correct / (i + 1))

        if (i + 1) % 25 == 0:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    direct_acc = correct / max(n_total, 1)

    summary = {
        "dataset": args.dataset,
        "model": "qwen_vl_chat",
        "method": args.method,
        "weight": args.weight,
        "weight1": args.weight1,
        "weight2": args.weight2,
        "threshold": args.threshold,
        "adapt_rule": args.adapt_rule,
        "n": n_total,
        "direct_acc": direct_acc,
        "out_json": str(out_json),
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDirect acc:", direct_acc)
    print("saved generations:", out_json)
    print("saved summary:", out_summary)

    print("\nRunning Controlled_Images evaluator...")
    dataset.evaluate_scores(
        scores=scores,
        path="outputs",
        dataset=args.dataset,
        model="qwen_vl_chat",
        method=args.method,
        weight=args.weight if args.method == "scaling_vis" else 1.0,
        sampled_indices=[],
        option=args.option,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
