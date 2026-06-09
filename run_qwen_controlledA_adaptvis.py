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


def find_image_spans_in_input_ids(input_ids, tokenizer):
    img_id, img_end_id = get_img_token_ids(tokenizer)

    spans = []
    input_ids_cpu = input_ids.detach().cpu()

    for b in range(input_ids_cpu.shape[0]):
        ids = input_ids_cpu[b].tolist()

        starts = [i for i, x in enumerate(ids) if x == img_id]
        ends = [i for i, x in enumerate(ids) if x == img_end_id]

        if len(starts) == 1 and len(ends) == 1 and ends[0] > starts[0]:
            spans.append((starts[0] + 1, ends[0]))
        else:
            spans.append(None)

    return spans


def patch_qwen_forward_for_span(model, tokenizer, debug=False):
    old_forward = model.forward

    def patched_forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        if input_ids is not None and torch.is_tensor(input_ids):
            spans = find_image_spans_in_input_ids(input_ids, tokenizer)

            if any(s is not None for s in spans):
                fixed = []
                for s in spans:
                    if s is None:
                        fixed.append((-1, -1))
                    else:
                        fixed.append(s)
                model._adaptvis_image_spans = fixed
                model._adaptvis_last_image_spans = fixed

                if debug:
                    print("forward found image spans:", fixed)

            elif getattr(model, "_adaptvis_last_image_spans", None) is not None:
                # generation 后续 step input_ids 只有新 token，没有 <img>...</img>
                # 继续使用第一次 full context 的 image span
                model._adaptvis_image_spans = model._adaptvis_last_image_spans

        return old_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)
    print("Patched model.forward for dynamic image-span detection.")


def patch_qwen_attention(model, debug=False):
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
                if value is None or not torch.is_tensor(value):
                    return out

                attn_output, attn_probs = out[0], out[1]

                if attn_probs is None or attn_probs.dim() != 4:
                    return out

                # attn_probs: [B, H, Q, K]
                bsz, n_heads, q_len, kv_len = attn_probs.shape

                if value.dim() != 4:
                    return out

                # Qwen _attn receives value usually as [B, K, H, D].
                # Convert to [B, H, K, D] for matmul.
                if value.shape[1] == kv_len and value.shape[2] == n_heads:
                    value_for_mm = value.permute(0, 2, 1, 3).contiguous()
                elif value.shape[1] == n_heads and value.shape[2] == kv_len:
                    value_for_mm = value
                else:
                    if debug:
                        print("Unexpected value shape:", tuple(value.shape), "attn_probs:", tuple(attn_probs.shape))
                    return out

                scaled = attn_probs.clone()

                for b in range(min(bsz, len(spans))):
                    st, ed = spans[b]
                    st = max(0, int(st))
                    ed = min(kv_len, int(ed))

                    if ed > st:
                        scaled[b, :, :, st:ed] *= weight

                scaled = scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                # [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
                new_attn_output = torch.matmul(scaled, value_for_mm)

                # Qwen expects [B, Q, H, D] before _merge_heads
                expected_bqhd = (bsz, q_len, n_heads, value_for_mm.shape[-1])
                expected_bhqd = (bsz, n_heads, q_len, value_for_mm.shape[-1])

                if tuple(attn_output.shape) == expected_bqhd:
                    new_attn_output = new_attn_output.permute(0, 2, 1, 3).contiguous()
                elif tuple(attn_output.shape) == expected_bhqd:
                    new_attn_output = new_attn_output.contiguous()
                else:
                    # 默认按 Qwen 原版格式 [B,Q,H,D]
                    new_attn_output = new_attn_output.permute(0, 2, 1, 3).contiguous()

                    if tuple(new_attn_output.shape) != tuple(attn_output.shape):
                        if debug:
                            print(
                                "Output shape mismatch. old:",
                                tuple(attn_output.shape),
                                "new:",
                                tuple(new_attn_output.shape),
                            )
                        return out

                if len(out) == 2:
                    return new_attn_output, scaled
                return (new_attn_output, scaled) + tuple(out[2:])

            return patched_attn

        module._attn = types.MethodType(make_patched(old_attn, name), module)
        patched += 1

        if debug:
            print("patched attention:", name)

    if patched == 0:
        raise RuntimeError("No QWenAttention._attn modules patched.")

    print("Patched QWenAttention modules:", patched)


def build_query(tokenizer, image_path, prompt):
    return tokenizer.from_list_format([
        {"image": image_path},
        {"text": prompt},
    ])


def encode_query(tokenizer, query, model):
    enc = tokenizer(query, return_tensors="pt")
    device = get_device(model)
    return {k: v.to(device) for k, v in enc.items()}


def candidate_token_ids(tokenizer):
    labels = ["left", "right", "on", "under"]
    out = {}

    for lab in labels:
        ids1 = tokenizer(" " + lab, add_special_tokens=False)["input_ids"]
        ids2 = tokenizer(lab, add_special_tokens=False)["input_ids"]

        if isinstance(ids1[0], list):
            ids1 = ids1[0]
        if isinstance(ids2[0], list):
            ids2 = ids2[0]

        out[lab] = {
            "space": int(ids1[0]),
            "plain": int(ids2[0]),
        }

    return out


@torch.no_grad()
def score_candidates_for_conf(model, tokenizer, query, cand_ids):
    # 只用于 AdaptVis threshold，不用于最终答案
    enc = encode_query(tokenizer, query, model)

    model._adaptvis_enable = True
    model._adaptvis_weight = 1.0

    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask", None),
        use_cache=False,
        return_dict=True,
    )

    logits = out.logits[0, -1, :].float()
    labels = ["left", "right", "on", "under"]
    cand_logits = []

    for lab in labels:
        tid_space = cand_ids[lab]["space"]
        tid_plain = cand_ids[lab]["plain"]
        cand_logits.append(torch.maximum(logits[tid_space], logits[tid_plain]))

    cand_logits = torch.stack(cand_logits)
    probs = F.softmax(cand_logits, dim=-1)

    best_idx = int(torch.argmax(probs).item())

    return {
        "best": labels[best_idx],
        "confidence": float(probs[best_idx].item()),
        "probs": {labels[i]: float(probs[i].item()) for i in range(len(labels))},
    }


@torch.no_grad()
def chat_generate(model, tokenizer, query, weight):
    model._adaptvis_enable = True
    model._adaptvis_weight = float(weight)

    response, history = model.chat(
        tokenizer,
        query=query,
        history=None,
    )

    return str(response).strip()


def pred_index_from_generation(response, gold_answer, caption_options):
    pred = parse_prep(response)
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
    parser.add_argument("--method", choices=["base", "scaling_vis", "adapt_vis"], default="base")

    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--weight1", type=float, default=0.5)
    parser.add_argument("--weight2", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--adapt_rule", choices=["high_w1", "high_w2"], default="high_w1")

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--debug_patch", action="store_true")
    parser.add_argument("--print_every", type=int, default=20)
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

    prompts, answers = [], []
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

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Loading Qwen model...")
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
        except Exception:
            pass

    try:
        model.generation_config.max_new_tokens = 16
        model.generation_config.do_sample = False
    except Exception:
        pass

    model.requires_grad_(False)

    patch_qwen_forward_for_span(model, tokenizer, debug=args.debug_patch)
    patch_qwen_attention(model, debug=args.debug_patch)

    cand_ids = candidate_token_ids(tokenizer)
    print("candidate token ids:", cand_ids)

    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []
    correct = 0
    unparsed = 0

    out_tag = (
        f"qwen_vl_chat_{args.dataset}_{args.method}"
        f"_generation_w{args.weight}_w1{args.weight1}_w2{args.weight2}_thr{args.threshold}"
    )

    out_json = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    for i in tqdm(range(n_total), total=n_total):
        d = dataset.dataset[i]
        image_path = resolve_image_path(d["image_path"])
        caption_options = d["caption_options"]

        prompt = clean_prompt_for_qwen(prompts[i])
        gold = answers[i]

        query = build_query(tokenizer, image_path, prompt)

        conf_info = score_candidates_for_conf(
            model=model,
            tokenizer=tokenizer,
            query=query,
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

        response = chat_generate(
            model=model,
            tokenizer=tokenizer,
            query=query,
            weight=chosen_weight,
        )

        pred_idx, pred_prep = pred_index_from_generation(
            response,
            gold,
            caption_options,
        )

        if pred_prep is None:
            unparsed += 1

        scores[i, 0, pred_idx] = 1.0

        is_correct = int(pred_idx == 0)
        correct += is_correct

        image_span = getattr(model, "_adaptvis_last_image_spans", None)

        rec = {
            "index": i,
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "generation": response,
            "pred_prep": pred_prep,
            "pred_idx": int(pred_idx),
            "correct": bool(is_correct),
            "conf_best": conf_info["best"],
            "confidence": conf_info["confidence"],
            "conf_probs": conf_info["probs"],
            "chosen_weight": float(chosen_weight),
            "image_span": image_span,
            "caption_options": caption_options,
        }

        records.append(rec)

        if args.print_every > 0 and (i % args.print_every == 0):
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("gold:", gold)
            print("conf:", conf_info)
            print("chosen_weight:", chosen_weight)
            print("generation:", response)
            print("pred:", pred_prep, "pred_idx:", pred_idx, "correct:", bool(is_correct))
            print("running acc:", correct / (i + 1), "unparsed:", unparsed)
            print("last image_span:", image_span)

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
        "unparsed": unparsed,
        "out_json": str(out_json),
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDirect acc:", direct_acc)
    print("unparsed:", unparsed)
    print("saved records:", out_json)
    print("saved summary:", out_summary)

    if n_total == len(dataset.dataset):
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
    else:
        print("\nSkip evaluator because --limit was used.")


if __name__ == "__main__":
    main()
