import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation import GenerationConfig
except Exception:
    GenerationConfig = None

from dataset_zoo.aro_datasets import Controlled_Images


def clean_prompt_for_qwen(prompt):
    prompt = str(prompt)
    prompt = prompt.replace("<image>", "")
    prompt = prompt.replace("USER:", "").replace("User:", "").replace("user:", "")
    prompt = prompt.replace("ASSISTANT:", "").replace("Assistant:", "").replace("assistant:", "")
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


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


def get_nested_attr(obj, name):
    cur = obj
    for part in name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def print_config_image_fields(config):
    print("\n" + "=" * 100)
    print("CONFIG FIELDS CONTAINING image/img/vision/visual")
    print("=" * 100)

    d = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    for k, v in d.items():
        lk = str(k).lower()
        if any(x in lk for x in ["image", "img", "vision", "visual"]):
            print(k, "=", v)

    for name in [
        "visual",
        "vision_config",
        "visual_config",
        "image_start_id",
        "image_end_id",
        "image_pad_id",
        "img_start_id",
        "img_end_id",
        "img_pad_id",
    ]:
        v = getattr(config, name, None)
        if v is not None:
            print(f"config.{name} =", v)


def collect_special_ids(tokenizer, model):
    candidates = [
        "<img>",
        "</img>",
        "<image>",
        "</image>",
        "<|image|>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<ref>",
        "</ref>",
        "<box>",
        "</box>",
    ]

    out = {}

    print("\n" + "=" * 100)
    print("TOKENIZER SPECIAL TOKEN MAP")
    print("=" * 100)
    print(getattr(tokenizer, "special_tokens_map", None))

    print("\n" + "=" * 100)
    print("CANDIDATE SPECIAL TOKEN IDS")
    print("=" * 100)

    for tok in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None

        if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
            out[tok] = tid
            print(f"{tok:20s} -> {tid}")

    # common config paths
    config = model.config
    attr_names = [
        "image_start_id",
        "image_end_id",
        "image_pad_id",
        "img_start_id",
        "img_end_id",
        "img_pad_id",
        "visual.image_start_id",
        "visual.image_end_id",
        "visual.image_pad_id",
        "visual.img_start_id",
        "visual.img_end_id",
        "visual.img_pad_id",
    ]

    print("\n" + "=" * 100)
    print("CANDIDATE IDS FROM MODEL CONFIG")
    print("=" * 100)

    for name in attr_names:
        val = get_nested_attr(config, name)
        if val is not None:
            out[f"config.{name}"] = int(val)
            print(f"config.{name:25s} -> {val}")

    return out


def inspect_tokens(tokenizer, input_ids, special_ids):
    ids = input_ids[0].tolist()

    print("\n" + "=" * 100)
    print("INPUT IDS BASIC")
    print("=" * 100)
    print("input_len:", len(ids))
    print("first 80 ids:", ids[:80])

    print("\nDecoded prompt preview:")
    try:
        print(tokenizer.decode(ids[:300]))
    except Exception as e:
        print("decode failed:", repr(e))

    print("\n" + "=" * 100)
    print("SPECIAL ID POSITIONS")
    print("=" * 100)

    id_to_names = {}
    for name, tid in special_ids.items():
        id_to_names.setdefault(int(tid), []).append(name)

    for tid, names in sorted(id_to_names.items()):
        pos = [i for i, x in enumerate(ids) if x == tid]
        if pos:
            print(f"id={tid} names={names} positions={pos[:20]} count={len(pos)}")

    print("\n" + "=" * 100)
    print("TOKEN TABLE AROUND SPECIAL IDS")
    print("=" * 100)

    interesting_pos = set()
    for tid in id_to_names:
        for i, x in enumerate(ids):
            if x == tid:
                for j in range(max(0, i - 10), min(len(ids), i + 25)):
                    interesting_pos.add(j)

    if not interesting_pos:
        interesting_pos = set(range(min(len(ids), 120)))

    for i in sorted(interesting_pos):
        tid = ids[i]
        names = id_to_names.get(tid, [])
        try:
            piece = tokenizer.decode([tid])
        except Exception:
            piece = "<?>"
        marker = " ".join(names)
        print(f"{i:5d}  id={tid:8d}  {repr(piece):30s}  {marker}")


def find_span_candidates(input_ids, special_ids):
    ids = input_ids[0].tolist()
    candidates = []

    pair_names = [
        ("<img>", "</img>"),
        ("<image>", "</image>"),
        ("<|vision_start|>", "<|vision_end|>"),
    ]

    # direct token string pairs
    for s_name, e_name in pair_names:
        if s_name not in special_ids or e_name not in special_ids:
            continue

        sid = int(special_ids[s_name])
        eid = int(special_ids[e_name])

        starts = [i for i, x in enumerate(ids) if x == sid]
        ends = [i for i, x in enumerate(ids) if x == eid]

        for st in starts:
            eds = [e for e in ends if e > st]
            if not eds:
                continue
            ed = eds[0]
            candidates.append({
                "name": f"{s_name}..{e_name} exclusive",
                "start": st + 1,
                "end": ed,
                "len": ed - st - 1,
            })
            candidates.append({
                "name": f"{s_name}..{e_name} inclusive",
                "start": st,
                "end": ed + 1,
                "len": ed - st + 1,
            })

    # config ids
    config_pairs = [
        ("config.image_start_id", "config.image_end_id"),
        ("config.img_start_id", "config.img_end_id"),
        ("config.visual.image_start_id", "config.visual.image_end_id"),
        ("config.visual.img_start_id", "config.visual.img_end_id"),
    ]

    for s_name, e_name in config_pairs:
        if s_name not in special_ids or e_name not in special_ids:
            continue

        sid = int(special_ids[s_name])
        eid = int(special_ids[e_name])

        starts = [i for i, x in enumerate(ids) if x == sid]
        ends = [i for i, x in enumerate(ids) if x == eid]

        for st in starts:
            eds = [e for e in ends if e > st]
            if not eds:
                continue
            ed = eds[0]
            candidates.append({
                "name": f"{s_name}..{e_name} exclusive",
                "start": st + 1,
                "end": ed,
                "len": ed - st - 1,
            })
            candidates.append({
                "name": f"{s_name}..{e_name} inclusive",
                "start": st,
                "end": ed + 1,
                "len": ed - st + 1,
            })

    # image pad repeated ids
    for key in [
        "<|image_pad|>",
        "<|vision_pad|>",
        "config.image_pad_id",
        "config.img_pad_id",
        "config.visual.image_pad_id",
        "config.visual.img_pad_id",
    ]:
        if key not in special_ids:
            continue

        tid = int(special_ids[key])
        pos = [i for i, x in enumerate(ids) if x == tid]
        if pos:
            candidates.append({
                "name": f"repeated pad {key}",
                "start": min(pos),
                "end": max(pos) + 1,
                "len": max(pos) - min(pos) + 1,
                "count": len(pos),
            })

    return candidates


def print_span_candidates(candidates):
    print("\n" + "=" * 100)
    print("SPAN CANDIDATES FROM INPUT IDS")
    print("=" * 100)

    if not candidates:
        print("No explicit image span candidate found from token ids.")
        return

    for c in candidates:
        L = int(c["len"])
        side = int(round(L ** 0.5)) if L > 0 else 0
        square = side * side == L
        print(
            f"{c['name']:45s} "
            f"start={c['start']:5d} end={c['end']:5d} len={L:5d} "
            f"square={square} side={side if square else '-'} "
            f"extra={ {k:v for k,v in c.items() if k not in ['name','start','end','len']} }"
        )


def inspect_attention_modules(model, max_print=80):
    print("\n" + "=" * 100)
    print("ATTENTION-LIKE MODULES")
    print("=" * 100)

    n = 0
    for name, module in model.named_modules():
        lname = name.lower()
        cname = module.__class__.__name__
        lcname = cname.lower()
        if "attn" in lname or "attention" in lname or "attn" in lcname or "attention" in lcname:
            print(f"{name:80s}  {cname}")
            n += 1
            if n >= max_print:
                print(f"... stopped after {max_print} modules")
                break

    if n == 0:
        print("No attention-like modules found by name.")


def run_forward_attention(model, input_ids, attention_mask):
    print("\n" + "=" * 100)
    print("FORWARD WITH output_attentions=True")
    print("=" * 100)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.config.output_attentions = True

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    attentions = getattr(out, "attentions", None)

    if attentions is None:
        print("out.attentions is None.")
        print("Output type:", type(out))
        if hasattr(out, "keys"):
            print("Output keys:", out.keys())
        return None

    print("num layers attentions:", len(attentions))

    for li, a in enumerate(attentions[:6]):
        if a is None:
            print(f"layer {li}: None")
            continue

        shape = tuple(a.shape)
        print(f"layer {li}: attention shape = {shape}")

    # first non-none layer
    for li, a in enumerate(attentions):
        if a is not None:
            q_len = a.shape[-2]
            kv_len = a.shape[-1]
            print("\nFirst non-none attention layer:", li)
            print("q_len:", q_len)
            print("kv_len:", kv_len)
            return {
                "layer": li,
                "q_len": q_len,
                "kv_len": kv_len,
                "attn_shape": tuple(a.shape),
            }

    print("All attentions are None.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--subset", default="A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--skip_forward", action="store_true")
    parser.add_argument("--max_attn_modules", type=int, default=80)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/ddnB/work/mwang32/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/ddnB/work/mwang32/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/ddnB/work/mwang32/hf_cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/ddnB/work/mwang32/hf_cache/datasets")
    os.environ.setdefault("TORCH_HOME", "/ddnB/work/mwang32/torch_cache")

    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]:
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

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

    idx = args.index
    d = dataset.dataset[idx]
    image_path = resolve_image_path(d["image_path"])
    prompt = clean_prompt_for_qwen(prompts[idx])

    print("\n" + "=" * 100)
    print("SAMPLE")
    print("=" * 100)
    print("idx:", idx)
    print("image_path:", image_path)
    print("gold answer:", answers[idx])
    print("caption_options:", d.get("caption_options"))
    print("prompt:", prompt)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    print("\nLoading model...")
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

    print_config_image_fields(model.config)
    special_ids = collect_special_ids(tokenizer, model)
    inspect_attention_modules(model, max_print=args.max_attn_modules)

    query = tokenizer.from_list_format([
        {"image": image_path},
        {"text": prompt},
    ])

    print("\n" + "=" * 100)
    print("QWEN QUERY STRING")
    print("=" * 100)
    print(repr(query[:1000]))

    enc = tokenizer(query, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    inspect_tokens(tokenizer, input_ids, special_ids)
    candidates = find_span_candidates(input_ids, special_ids)
    print_span_candidates(candidates)

    if args.skip_forward:
        print("\n--skip_forward set. Stop before model forward.")
        return

    try:
        attn_info = run_forward_attention(model, input_ids, attention_mask)
    except Exception as e:
        print("\nForward failed:")
        print(repr(e))
        print("\nThis may happen if Qwen remote-code forward does not expose attentions or needs chat/generate context.")
        return

    if attn_info is not None:
        print("\n" + "=" * 100)
        print("ATTENTION LENGTH VS INPUT LENGTH")
        print("=" * 100)
        input_len = input_ids.shape[-1]
        kv_len = attn_info["kv_len"]
        print("input_len:", input_len)
        print("kv_len:", kv_len)
        print("kv_len - input_len:", kv_len - input_len)

        print("\nSpan candidates relative to kv_len:")
        for c in candidates:
            st, ed = c["start"], c["end"]
            ok = 0 <= st < ed <= kv_len
            print(
                f"{c['name']:45s} "
                f"[{st}, {ed}) len={c['len']} valid_in_kv={ok}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
