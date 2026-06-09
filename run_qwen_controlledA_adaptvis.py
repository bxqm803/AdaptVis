import argparse
import csv
import json
import math
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


# ============================================================
# Basic utilities
# ============================================================
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


# ============================================================
# Processed image + bbox utilities
# ============================================================
def load_processed_manifest(path):
    """
    Manifest produced by the processed-image preprocessing step.
    Expected columns:
      src,dst,orig_w,orig_h,proc_w,proc_h

    Returns:
      basename(src) -> row dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src = r["src"]
            base = os.path.basename(src)
            out[base] = {
                "src": src,
                "dst": r["dst"],
                "orig_w": int(float(r.get("orig_w", 0) or 0)),
                "orig_h": int(float(r.get("orig_h", 0) or 0)),
                "proc_w": int(float(r.get("proc_w", 0) or 0)),
                "proc_h": int(float(r.get("proc_h", 0) or 0)),
            }
    return out


def load_bbox_json(path):
    """
    Expected GroundingDINO json format:
      [
        {
          "sid": 0,
          "subject_best": {"box_xyxy": [x1,y1,x2,y2], ...},
          "object_best": {"box_xyxy": [x1,y1,x2,y2], ...}
        }, ...
      ]

    Coordinates should be on the processed image if --use_processed_image is used.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.load(open(path, "r", encoding="utf-8"))
    out = {}
    for j, r in enumerate(data):
        sid = int(r.get("sid", r.get("index", j)))
        out[sid] = r
    return out


def make_bbox_region(bbox_record, target="both"):
    """
    Convert one GroundingDINO record into a region payload.
    target: subject | object | both
    """
    if bbox_record is None:
        return None

    boxes = []
    sources = []

    if target in ["subject", "both"]:
        b = bbox_record.get("subject_best", None)
        if b is not None and b.get("box_xyxy", None) is not None:
            boxes.append([float(x) for x in b["box_xyxy"]])
            sources.append("subject")

    if target in ["object", "both"]:
        b = bbox_record.get("object_best", None)
        if b is not None and b.get("box_xyxy", None) is not None:
            boxes.append([float(x) for x in b["box_xyxy"]])
            sources.append("object")

    if not boxes:
        return None

    return {
        "boxes": boxes,
        "source": "+".join(sources),
    }


def boxes_to_patch_offsets(boxes, image_w, image_h, n_img_tokens):
    """
    Convert xyxy bbox coordinates to row-major patch offsets inside the image-token span.

    For Qwen-VL-Chat observed image span:
      n_img_tokens = 256 -> 16x16 grid
      offset = row * 16 + col

    The bbox is mapped by patch intersection.
    """
    if not boxes or image_w <= 0 or image_h <= 0 or n_img_tokens <= 0:
        return []

    grid = int(round(math.sqrt(n_img_tokens)))
    if grid * grid != n_img_tokens:
        raise RuntimeError(f"Cannot map bbox to square grid: n_img_tokens={n_img_tokens}")

    offsets = set()

    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]

        x1 = max(0.0, min(float(image_w), x1))
        x2 = max(0.0, min(float(image_w), x2))
        y1 = max(0.0, min(float(image_h), y1))
        y2 = max(0.0, min(float(image_h), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        c0 = int(math.floor(x1 / image_w * grid))
        c1 = int(math.ceil(x2 / image_w * grid)) - 1
        r0 = int(math.floor(y1 / image_h * grid))
        r1 = int(math.ceil(y2 / image_h * grid)) - 1

        c0 = max(0, min(grid - 1, c0))
        c1 = max(0, min(grid - 1, c1))
        r0 = max(0, min(grid - 1, r0))
        r1 = max(0, min(grid - 1, r1))

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                offsets.add(rr * grid + cc)

    return sorted(offsets)


# ============================================================
# Qwen image-span detection
# ============================================================
def get_img_token_ids(tokenizer):
    img_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_id = tokenizer.convert_tokens_to_ids("</img>")

    if img_id is None or img_end_id is None:
        raise RuntimeError("Cannot find <img> / </img> token ids.")

    return int(img_id), int(img_end_id)


def find_image_spans_in_input_ids(input_ids, tokenizer):
    img_id, img_end_id = get_img_token_ids(tokenizer)
    input_ids_cpu = input_ids.detach().cpu()

    spans = []
    for b in range(input_ids_cpu.shape[0]):
        ids = input_ids_cpu[b].tolist()
        starts = [i for i, x in enumerate(ids) if x == img_id]
        ends = [i for i, x in enumerate(ids) if x == img_end_id]

        if len(starts) == 1 and len(ends) == 1 and ends[0] > starts[0]:
            # exclusive image-token span: (<img>, </img>)
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
                fixed = [s if s is not None else (-1, -1) for s in spans]
                model._adaptvis_image_spans = fixed
                model._adaptvis_last_image_spans = fixed

                if debug:
                    print("forward image spans:", fixed)
            else:
                # generation later steps only contain generated tokens;
                # reuse the full-context image span.
                last = getattr(model, "_adaptvis_last_image_spans", None)
                if last is not None:
                    model._adaptvis_image_spans = last

        return old_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)
    print("Patched model.forward for dynamic image-span detection.")


# ============================================================
# Qwen attention patch: whole-image or bbox-patch scaling
# ============================================================
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
                args = list(args)

                # Qwen _attn signature usually:
                # _attn(query, key, value, registered_causal_mask, attention_mask=None, head_mask=None)
                # In this remote code, query/key entering _attn are [B,H,Q/K,D].
                if len(args) >= 5 and args[4] is None:
                    query = args[0]
                    key = args[1]
                    bsz = query.shape[0]
                    q_len = query.shape[-2]
                    k_len = key.shape[-2]
                    args[4] = torch.zeros(
                        (bsz, 1, q_len, k_len),
                        device=query.device,
                        dtype=query.dtype,
                    )

                if "attention_mask" in kwargs and kwargs["attention_mask"] is None:
                    query = args[0] if len(args) > 0 else kwargs.get("query", None)
                    key = args[1] if len(args) > 1 else kwargs.get("key", None)
                    if query is not None and key is not None:
                        bsz = query.shape[0]
                        q_len = query.shape[-2]
                        k_len = key.shape[-2]
                        kwargs["attention_mask"] = torch.zeros(
                            (bsz, 1, q_len, k_len),
                            device=query.device,
                            dtype=query.dtype,
                        )

                out = old_attn_func(*args, **kwargs)

                if not getattr(model, "_adaptvis_enable", False):
                    return out

                weight = float(getattr(model, "_adaptvis_weight", 1.0))
                spans = getattr(model, "_adaptvis_image_spans", None)

                if weight == 1.0 or not spans:
                    return out

                if not isinstance(out, tuple) or len(out) < 2:
                    raise RuntimeError(f"{module_name}: _attn output is not tuple with attention weights.")

                value = args[2] if len(args) >= 3 else kwargs.get("value", None)
                if value is None or not torch.is_tensor(value):
                    raise RuntimeError(f"{module_name}: cannot find value tensor.")

                attn_output, attn_probs = out[0], out[1]

                if attn_probs is None or attn_probs.dim() != 4:
                    raise RuntimeError(
                        f"{module_name}: bad attn_probs shape: "
                        f"{None if attn_probs is None else tuple(attn_probs.shape)}"
                    )

                # attn_probs: [B,H,Q,K]
                bsz, n_heads, q_len, kv_len = attn_probs.shape

                if value.dim() != 4:
                    raise RuntimeError(f"{module_name}: bad value shape: {tuple(value.shape)}")

                # Convert value to [B,H,K,D].
                if value.shape[1] == n_heads and value.shape[2] == kv_len:
                    value_for_mm = value.contiguous()
                elif value.shape[1] == kv_len and value.shape[2] == n_heads:
                    value_for_mm = value.permute(0, 2, 1, 3).contiguous()
                else:
                    raise RuntimeError(
                        f"{module_name}: unexpected value shape {tuple(value.shape)} "
                        f"for attn_probs {tuple(attn_probs.shape)}"
                    )

                scaled = attn_probs.clone()

                local_calls = 0
                local_before = 0.0
                local_after_raw = 0.0

                region_mode = getattr(model, "_adaptvis_region_mode", "all_image")
                bbox_regions = getattr(model, "_adaptvis_bbox_regions", None)
                bbox_fallback = getattr(model, "_adaptvis_bbox_fallback", "skip")

                for b in range(min(bsz, len(spans))):
                    st, ed = spans[b]
                    st = max(0, int(st))
                    ed = min(kv_len, int(ed))

                    if ed <= st:
                        continue

                    token_indices = None
                    bbox_offsets = None

                    if region_mode == "bbox":
                        region = None
                        if bbox_regions is not None and b < len(bbox_regions):
                            region = bbox_regions[b]

                        if region is not None:
                            n_img_tokens = ed - st
                            bbox_offsets = boxes_to_patch_offsets(
                                boxes=region.get("boxes", []),
                                image_w=int(region.get("image_w", 0)),
                                image_h=int(region.get("image_h", 0)),
                                n_img_tokens=n_img_tokens,
                            )

                        if bbox_offsets:
                            idx = [st + int(o) for o in bbox_offsets if 0 <= int(o) < (ed - st)]
                            if idx:
                                token_indices = torch.tensor(idx, device=scaled.device, dtype=torch.long)
                        elif bbox_fallback == "all_image":
                            token_indices = None
                        else:
                            # No valid bbox: do not scale this sample.
                            continue

                    if token_indices is None:
                        # Original AdaptVis behavior: scale whole image-token span.
                        before = scaled[b, :, -1:, st:ed].detach().float().sum().item()
                        scaled[b, :, -1:, st:ed] *= weight
                        after_raw = scaled[b, :, -1:, st:ed].detach().float().sum().item()
                        scaled_patch_count = ed - st
                    else:
                        # Bbox behavior: scale only visual tokens whose patches intersect bbox.
                        before = scaled[b, :, -1:, token_indices].detach().float().sum().item()
                        scaled[b, :, -1:, token_indices] *= weight
                        after_raw = scaled[b, :, -1:, token_indices].detach().float().sum().item()
                        scaled_patch_count = int(token_indices.numel())

                    local_calls += 1
                    local_before += before
                    local_after_raw += after_raw

                    model._adaptvis_scaled_patch_count = (
                        getattr(model, "_adaptvis_scaled_patch_count", 0) + int(scaled_patch_count)
                    )

                    if bbox_offsets is not None and getattr(model, "_adaptvis_last_bbox_offsets", None) is None:
                        model._adaptvis_last_bbox_offsets = [[int(x) for x in bbox_offsets]]

                if local_calls == 0:
                    return out

                scaled = scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                # [B,H,Q,K] @ [B,H,K,D] -> [B,H,Q,D]
                new_attn_output_bhqd = torch.matmul(scaled, value_for_mm).contiguous()

                # Match original _attn output layout exactly.
                if tuple(new_attn_output_bhqd.shape) == tuple(attn_output.shape):
                    final_attn_output = new_attn_output_bhqd
                    layout = "BHQD"
                elif tuple(new_attn_output_bhqd.permute(0, 2, 1, 3).shape) == tuple(attn_output.shape):
                    final_attn_output = new_attn_output_bhqd.permute(0, 2, 1, 3).contiguous()
                    layout = "BQHD"
                else:
                    raise RuntimeError(
                        f"{module_name}: cannot match attn_output shape. "
                        f"old={tuple(attn_output.shape)}, "
                        f"new_bhqd={tuple(new_attn_output_bhqd.shape)}, "
                        f"new_bqhd={tuple(new_attn_output_bhqd.permute(0, 2, 1, 3).shape)}"
                    )

                model._adaptvis_scaled_calls = getattr(model, "_adaptvis_scaled_calls", 0) + local_calls
                model._adaptvis_scaled_mass_before = getattr(model, "_adaptvis_scaled_mass_before", 0.0) + local_before
                model._adaptvis_scaled_mass_after_raw = getattr(model, "_adaptvis_scaled_mass_after_raw", 0.0) + local_after_raw
                model._adaptvis_last_layout = layout

                if len(out) == 2:
                    return final_attn_output, scaled
                return (final_attn_output, scaled) + tuple(out[2:])

            return patched_attn

        module._attn = types.MethodType(make_patched(old_attn, name), module)
        patched += 1

        if debug:
            print("patched attention:", name)

    if patched == 0:
        raise RuntimeError("No QWenAttention._attn modules patched.")

    print("Patched QWenAttention modules:", patched)


# ============================================================
# Qwen generation and probability capture
# ============================================================
def build_query(tokenizer, image_path, prompt):
    return tokenizer.from_list_format([
        {"image": image_path},
        {"text": prompt},
    ])


def chat_generate_with_scores(
    model,
    tokenizer,
    query,
    weight,
    capture_scores=True,
    bbox_regions=None,
    region_mode="all_image",
    bbox_fallback="skip",
):
    model._adaptvis_enable = True
    model._adaptvis_weight = float(weight)
    model._adaptvis_last_image_spans = None
    model._adaptvis_image_spans = None

    model._adaptvis_scaled_calls = 0
    model._adaptvis_scaled_patch_count = 0
    model._adaptvis_scaled_mass_before = 0.0
    model._adaptvis_scaled_mass_after_raw = 0.0
    model._adaptvis_last_layout = None
    model._adaptvis_last_bbox_offsets = None

    model._adaptvis_region_mode = region_mode
    model._adaptvis_bbox_regions = bbox_regions
    model._adaptvis_bbox_fallback = bbox_fallback

    captured = {}
    old_generate = model.generate

    if capture_scores:
        def wrapped_generate(*args, **kwargs):
            kwargs.pop("output_scores", None)
            kwargs.pop("return_dict_in_generate", None)

            out = old_generate(
                *args,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            captured["output"] = out
            return out.sequences

        model.generate = wrapped_generate

    try:
        response, history = model.chat(
            tokenizer,
            query=query,
            history=None,
        )
    finally:
        if capture_scores:
            model.generate = old_generate

    score_info = {
        "first_token_max_prob": None,
        "first_token_max_prob_round": None,
        "first_token_argmax_id": None,
        "first_token_argmax_text": None,
        "first_token_top5": None,
    }

    if capture_scores and "output" in captured:
        out = captured["output"]
        if hasattr(out, "scores") and out.scores is not None and len(out.scores) > 0:
            logits0 = out.scores[0][0].float()
            probs0 = F.softmax(logits0, dim=-1)
            max_prob, max_id = torch.max(probs0, dim=-1)

            top_probs, top_ids = torch.topk(probs0, k=min(5, probs0.numel()))
            top5 = []
            for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
                try:
                    tok = tokenizer.decode([int(tid)])
                except Exception:
                    tok = str(tid)
                top5.append({
                    "token_id": int(tid),
                    "token": tok,
                    "prob": float(p),
                })

            max_id_int = int(max_id.item())
            try:
                max_text = tokenizer.decode([max_id_int])
            except Exception:
                max_text = str(max_id_int)

            score_info = {
                "first_token_max_prob": float(max_prob.item()),
                "first_token_max_prob_round": round(float(max_prob.item()), 2),
                "first_token_argmax_id": max_id_int,
                "first_token_argmax_text": max_text,
                "first_token_top5": top5,
            }

    return str(response).strip(), score_info


# ============================================================
# Evaluation helpers
# ============================================================
def pred_to_option_index(pred_prep, gold_answer, caption_options):
    gold = parse_prep(gold_answer)

    if pred_prep is None:
        for i, cap in enumerate(caption_options):
            if parse_prep(cap) != gold:
                return i, False
        return 0, False

    pred_idx = None
    for i, cap in enumerate(caption_options):
        if parse_prep(cap) == pred_prep:
            pred_idx = i
            break

    if pred_idx is None:
        for i, cap in enumerate(caption_options):
            if parse_prep(cap) != gold:
                return i, False
        return 0, False

    return pred_idx, pred_prep == gold


def choose_adapt_weight(confidence, threshold, weight1, weight2, adapt_rule):
    if confidence is None:
        confidence = 0.0

    if adapt_rule == "paper":
        # Original AdaptVis/LLaVA logic:
        # if first-token full-vocab max prob < threshold -> weight1, else weight2.
        return weight1 if confidence < threshold else weight2

    if adapt_rule == "high_w1":
        return weight1 if confidence >= threshold else weight2

    if adapt_rule == "high_w2":
        return weight2 if confidence >= threshold else weight1

    raise ValueError(adapt_rule)


# ============================================================
# Main
# ============================================================
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
    parser.add_argument("--adapt_rule", choices=["paper", "high_w1", "high_w2"], default="paper")

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--debug_patch", action="store_true")
    parser.add_argument("--print_every", type=int, default=20)

    # Processed image input.
    parser.add_argument("--use_processed_image", action="store_true")
    parser.add_argument("--processed_manifest", default="data/controlledA_llava15_processed_manifest.csv")

    # Region scaling.
    # all_image: original AdaptVis behavior.
    # bbox: only scale visual tokens whose patches intersect detected bbox.
    parser.add_argument("--scale_region", choices=["all_image", "bbox"], default="all_image")
    parser.add_argument("--bbox_json", default="data/controlledA_groundingdino_bbox_on_processed.json")
    parser.add_argument("--bbox_target", choices=["subject", "object", "both"], default="both")
    parser.add_argument("--bbox_fallback", choices=["skip", "all_image"], default="skip")

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

    processed_manifest = {}
    if args.use_processed_image:
        print("Loading processed image manifest:", args.processed_manifest)
        processed_manifest = load_processed_manifest(args.processed_manifest)

    bbox_by_sid = {}
    if args.scale_region == "bbox":
        print("Loading bbox json:", args.bbox_json)
        bbox_by_sid = load_bbox_json(args.bbox_json)

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
    print("use_processed_image:", args.use_processed_image)
    print("scale_region:", args.scale_region)
    print("bbox_target:", args.bbox_target)
    print("bbox_fallback:", args.bbox_fallback)

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
        model.generation_config.do_sample = False
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50
        model.generation_config.max_new_tokens = 16
    except Exception:
        pass

    model.requires_grad_(False)

    patch_qwen_forward_for_span(model, tokenizer, debug=args.debug_patch)
    patch_qwen_attention(model, debug=args.debug_patch)

    scores = np.zeros((len(dataset.dataset), 1, 4), dtype=np.float32)
    records = []

    correct = 0
    unparsed = 0

    extra_tag = (
        f"_proc{int(args.use_processed_image)}"
        f"_region{args.scale_region}"
        f"_target{args.bbox_target}"
        f"_fallback{args.bbox_fallback}"
    )

    out_tag = (
        f"qwen_vl_chat_{args.dataset}_{args.method}"
        f"_llava_aligned_generation_w{args.weight}_w1{args.weight1}_w2{args.weight2}"
        f"_thr{args.threshold}_rule{args.adapt_rule}{extra_tag}"
    )

    out_json = Path("outputs") / f"{out_tag}_records.json"
    out_summary = Path("outputs") / f"{out_tag}_summary.json"

    for i in tqdm(range(n_total), total=n_total):
        d = dataset.dataset[i]

        image_path = resolve_image_path(d["image_path"])
        caption_options = d["caption_options"]

        qwen_image_path = image_path
        proc_info = None

        if args.use_processed_image:
            base = os.path.basename(image_path)
            if base not in processed_manifest:
                raise KeyError(f"processed image not found for basename: {base}")
            proc_info = processed_manifest[base]
            qwen_image_path = proc_info["dst"]

        prompt = clean_prompt_for_qwen(prompts[i])
        gold = answers[i]

        bbox_region = None
        if args.scale_region == "bbox":
            bbox_rec = bbox_by_sid.get(i, None)
            bbox_region = make_bbox_region(bbox_rec, target=args.bbox_target)

            if bbox_region is not None:
                if proc_info is not None:
                    bbox_region["image_w"] = int(proc_info["proc_w"])
                    bbox_region["image_h"] = int(proc_info["proc_h"])
                else:
                    # Assume bbox coordinates are on the image actually passed to Qwen.
                    from PIL import Image
                    with Image.open(qwen_image_path) as im:
                        bbox_region["image_w"], bbox_region["image_h"] = im.size

        bbox_regions_for_batch = [bbox_region] if bbox_region is not None else None
        query = build_query(tokenizer, qwen_image_path, prompt)

        probe_generation = None
        probe_score = None

        if args.method == "adapt_vis":
            # Original AdaptVis style:
            # 1) generate with weight=1.0
            # 2) take first generated token full-vocab max softmax prob
            # 3) choose weight by threshold
            probe_generation, probe_score = chat_generate_with_scores(
                model=model,
                tokenizer=tokenizer,
                query=query,
                weight=1.0,
                capture_scores=True,
                bbox_regions=bbox_regions_for_batch,
                region_mode=args.scale_region,
                bbox_fallback=args.bbox_fallback,
            )

            confidence = probe_score["first_token_max_prob_round"]
            chosen_weight = choose_adapt_weight(
                confidence=confidence,
                threshold=args.threshold,
                weight1=args.weight1,
                weight2=args.weight2,
                adapt_rule=args.adapt_rule,
            )

        elif args.method == "base":
            chosen_weight = 1.0

        else:
            chosen_weight = args.weight

        response, final_score = chat_generate_with_scores(
            model=model,
            tokenizer=tokenizer,
            query=query,
            weight=chosen_weight,
            capture_scores=True,
            bbox_regions=bbox_regions_for_batch,
            region_mode=args.scale_region,
            bbox_fallback=args.bbox_fallback,
        )

        pred_prep = parse_prep(response)
        pred_idx, is_correct_bool = pred_to_option_index(
            pred_prep,
            gold,
            caption_options,
        )

        if pred_prep is None:
            unparsed += 1

        scores[i, 0, pred_idx] = 1.0

        is_correct = int(is_correct_bool)
        correct += is_correct

        image_span = getattr(model, "_adaptvis_last_image_spans", None)
        bbox_offsets = getattr(model, "_adaptvis_last_bbox_offsets", None)

        rec = {
            "index": i,
            "image_path": image_path,
            "qwen_image_path": qwen_image_path,
            "using_processed_image": bool(args.use_processed_image),
            "processed_info": proc_info,
            "prompt": prompt,
            "gold": gold,
            "generation": response,
            "pred_prep": pred_prep,
            "pred_idx": int(pred_idx),
            "correct": bool(is_correct),

            "chosen_weight": float(chosen_weight),

            "probe_generation": probe_generation,
            "probe_first_token_max_prob": None if probe_score is None else probe_score["first_token_max_prob"],
            "probe_first_token_max_prob_round": None if probe_score is None else probe_score["first_token_max_prob_round"],
            "probe_first_token_argmax_id": None if probe_score is None else probe_score["first_token_argmax_id"],
            "probe_first_token_argmax_text": None if probe_score is None else probe_score["first_token_argmax_text"],
            "probe_first_token_top5": None if probe_score is None else probe_score["first_token_top5"],

            "final_first_token_max_prob": final_score["first_token_max_prob"],
            "final_first_token_max_prob_round": final_score["first_token_max_prob_round"],
            "final_first_token_argmax_id": final_score["first_token_argmax_id"],
            "final_first_token_argmax_text": final_score["first_token_argmax_text"],
            "final_first_token_top5": final_score["first_token_top5"],

            "image_span": image_span,
            "scale_region": args.scale_region,
            "bbox_target": args.bbox_target,
            "bbox_fallback": args.bbox_fallback,
            "bbox_region": bbox_region,
            "bbox_offsets": bbox_offsets,
            "bbox_patch_count": 0 if not bbox_offsets else len(bbox_offsets[0]),

            "scaled_calls": int(getattr(model, "_adaptvis_scaled_calls", 0)),
            "scaled_patch_count": int(getattr(model, "_adaptvis_scaled_patch_count", 0)),
            "scaled_mass_before": float(getattr(model, "_adaptvis_scaled_mass_before", 0.0)),
            "scaled_mass_after_raw": float(getattr(model, "_adaptvis_scaled_mass_after_raw", 0.0)),
            "attn_output_layout": getattr(model, "_adaptvis_last_layout", None),

            "caption_options": caption_options,
        }

        records.append(rec)

        if args.print_every > 0 and (i % args.print_every == 0):
            print("\n" + "=" * 80)
            print("idx:", i)
            print("image:", image_path)
            print("qwen_image:", qwen_image_path)
            print("gold:", gold)
            print("bbox_region:", bbox_region)
            print("bbox_offsets:", bbox_offsets)
            print("bbox_patch_count:", rec["bbox_patch_count"])

            if probe_score is not None:
                print("probe_generation:", probe_generation)
                print("probe_first_token_max_prob:", probe_score["first_token_max_prob"])
                print("probe_first_token_max_prob_round:", probe_score["first_token_max_prob_round"])
                print("probe_first_token_argmax_text:", probe_score["first_token_argmax_text"])

            print("chosen_weight:", chosen_weight)
            print("generation:", response)
            print("final_first_token_max_prob:", final_score["first_token_max_prob"])
            print("final_first_token_argmax_text:", final_score["first_token_argmax_text"])
            print("pred:", pred_prep, "pred_idx:", pred_idx, "correct:", bool(is_correct))
            print("running acc:", correct / (i + 1), "unparsed:", unparsed)
            print("last image_span:", image_span)
            print("scaled_calls:", rec["scaled_calls"])
            print("scaled_patch_count:", rec["scaled_patch_count"])
            print("scaled_mass_before:", rec["scaled_mass_before"])
            print("scaled_mass_after_raw:", rec["scaled_mass_after_raw"])
            print("attn_output_layout:", rec["attn_output_layout"])

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
        "use_processed_image": args.use_processed_image,
        "processed_manifest": args.processed_manifest,
        "scale_region": args.scale_region,
        "bbox_json": args.bbox_json,
        "bbox_target": args.bbox_target,
        "bbox_fallback": args.bbox_fallback,
        "n": n_total,
        "direct_acc": direct_acc,
        "unparsed": unparsed,
        "out_json": str(out_json),
        "prob_definition": "first generated token full-vocab max softmax probability from model.chat/model.generate output_scores",
        "scaling_definition": "last-query attention probability scaling over either whole image tokens or bbox-intersecting image patch tokens",
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
