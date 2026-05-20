import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import LlavaForConditionalGeneration as LlavaModel
except ImportError:
    from transformers import AutoModelForVision2Seq as LlavaModel

from dataset_zoo import get_dataset


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "a272c74"

RELATIONS = ["left", "right", "on", "under"]

SURFACE_FORMS = [
    "lower_space",
    "cap_space",
    "lower_nospace",
    "cap_nospace",
]


# ============================================================
# Prompt / dataset helpers
# ============================================================

def load_prompt_rows(dataset_name: str, option: str) -> List[dict]:
    path = Path(f"prompts/{dataset_name}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def normalize_relation(x: str) -> str:
    s = str(x).strip().lower()

    if "under" in s:
        return "under"
    if re.search(r"\bon\b", s) and "front" not in s:
        return "on"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"

    return "unknown"


def get_gold_from_prompt_row(row: dict) -> str:
    ans = row.get("answer", "")
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return normalize_relation(ans)


def clean_question(q: str) -> str:
    q = str(q)
    q = q.replace("<image>", " ")
    q = q.replace("USER:", " ")
    q = q.replace("ASSISTANT:", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def get_raw_pil_from_dataset(dataset, idx: int) -> Image.Image:
    item = dataset[idx]

    if not isinstance(item, dict):
        raise TypeError(f"Expected dataset item to be dict, got {type(item)}")

    if "image_options" in item:
        image = item["image_options"][0]
    elif "image" in item:
        image = item["image"]
    else:
        raise KeyError(f"Cannot find image in dataset item keys: {list(item.keys())}")

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)}")

    return image.convert("RGB")


def load_dataset_any_signature(dataset_name: str, root_dir: str, download: bool):
    try:
        return get_dataset(
            dataset_name,
            root_dir=root_dir,
            image_preprocess=None,
            download=download,
        )
    except TypeError:
        return get_dataset(
            dataset_name,
            image_preprocess=None,
            download=download,
        )


# ============================================================
# LLaVA-like manual image preprocessing
# ============================================================

def get_size_value(size_obj, key: str, default: int) -> int:
    if isinstance(size_obj, dict):
        return int(size_obj.get(key, default))
    if isinstance(size_obj, int):
        return int(size_obj)
    return int(default)


def get_resample_from_processor(image_processor):
    resample = getattr(image_processor, "resample", None)
    if resample is not None:
        return resample
    return Image.BICUBIC


def infer_llava_geometry(image_processor, fallback_size: int = 336) -> Dict:
    do_resize = bool(getattr(image_processor, "do_resize", True))
    do_center_crop = bool(getattr(image_processor, "do_center_crop", True))
    do_pad = bool(getattr(image_processor, "do_pad", False))

    size = getattr(image_processor, "size", None) or {}
    crop_size = getattr(image_processor, "crop_size", None) or {}

    shortest_edge = None
    resize_h = None
    resize_w = None

    if isinstance(size, dict):
        if "shortest_edge" in size:
            shortest_edge = int(size["shortest_edge"])
        if "height" in size and "width" in size:
            resize_h = int(size["height"])
            resize_w = int(size["width"])
    elif isinstance(size, int):
        shortest_edge = int(size)

    crop_h = get_size_value(crop_size, "height", fallback_size)
    crop_w = get_size_value(crop_size, "width", fallback_size)

    if shortest_edge is None and resize_h is None:
        shortest_edge = fallback_size

    return {
        "do_resize": do_resize,
        "do_center_crop": do_center_crop,
        "do_pad": do_pad,
        "shortest_edge": shortest_edge,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "resample": get_resample_from_processor(image_processor),
    }


def expand2square(img: Image.Image, background_color: Tuple[int, int, int]):
    w, h = img.size
    if w == h:
        return img, 0, 0, w

    size = max(w, h)
    out = Image.new("RGB", (size, size), background_color)

    pad_x = (size - w) // 2
    pad_y = (size - h) // 2

    out.paste(img, (pad_x, pad_y))
    return out, pad_x, pad_y, size


def make_processed_pil_like_llava(
    raw: Image.Image,
    image_processor,
    force_mode: str = "auto",
) -> Tuple[Image.Image, Dict]:
    raw = raw.convert("RGB")
    geom = infer_llava_geometry(image_processor)

    if force_mode not in ["auto", "crop", "pad"]:
        raise ValueError(f"Unknown force_mode={force_mode}")

    mode = force_mode
    if mode == "auto":
        mode = "pad" if geom["do_pad"] else "crop"

    if mode == "pad":
        mean = getattr(
            image_processor,
            "image_mean",
            [0.48145466, 0.4578275, 0.40821073],
        )
        bg = tuple(int(float(x) * 255) for x in mean)

        square, pad_x, pad_y, square_size = expand2square(raw, bg)

        target_h = geom["crop_h"]
        target_w = geom["crop_w"]
        processed = square.resize((target_w, target_h), geom["resample"])

        meta = {
            "mode": "pad",
            "raw_size": raw.size,
            "square_size": square_size,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "processed_size": processed.size,
            "grid": target_h // 14,
        }
        return processed, meta

    w, h = raw.size

    if geom["resize_h"] is not None and geom["resize_w"] is not None:
        resized = raw.resize((geom["resize_w"], geom["resize_h"]), geom["resample"])
    else:
        shortest = geom["shortest_edge"]
        scale = shortest / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = raw.resize((new_w, new_h), geom["resample"])

    rw, rh = resized.size
    crop_w = geom["crop_w"]
    crop_h = geom["crop_h"]

    left = max(0, int(round((rw - crop_w) / 2)))
    top = max(0, int(round((rh - crop_h) / 2)))

    processed = resized.crop((left, top, left + crop_w, top + crop_h))

    meta = {
        "mode": "crop",
        "raw_size": raw.size,
        "resized_size": resized.size,
        "crop_left": left,
        "crop_top": top,
        "processed_size": processed.size,
        "grid": crop_h // 14,
    }
    return processed, meta


# ============================================================
# HF LLaVA loading
# ============================================================

def load_llava_hf(
    model_id: str,
    revision: str,
    cache_dir: str,
    device: str,
    dtype: str,
):
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
    )

    model = LlavaModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    patch_size = getattr(getattr(model.config, "vision_config", None), "patch_size", 14)
    vision_feature_select_strategy = getattr(
        model.config,
        "vision_feature_select_strategy",
        "default",
    )

    processor.patch_size = patch_size
    processor.vision_feature_select_strategy = vision_feature_select_strategy

    model = model.to(device).eval()

    print("[INFO] loaded official HF LLaVA")
    print(f"  model_id={model_id}")
    print(f"  revision={revision}")
    print(f"  model={type(model)}")
    print(f"  processor={type(processor)}")
    print(f"  patch_size={processor.patch_size}")
    print(f"  vision_feature_select_strategy={processor.vision_feature_select_strategy}")

    return processor, model


def nested_getattr(obj, path: str):
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def get_lm_head_and_final_norm(model):
    lm_head_candidates = [
        "language_model.lm_head",
        "lm_head",
        "model.lm_head",
    ]

    norm_candidates = [
        "language_model.model.norm",
        "language_model.norm",
        "model.norm",
        "norm",
    ]

    lm_head = None
    for p in lm_head_candidates:
        lm_head = nested_getattr(model, p)
        if lm_head is not None:
            break

    final_norm = None
    for p in norm_candidates:
        final_norm = nested_getattr(model, p)
        if final_norm is not None:
            break

    if lm_head is None:
        raise RuntimeError("Cannot find lm_head in model.")

    if final_norm is None:
        print("[WARN] Cannot find final norm. Intermediate layers use raw hidden states.")

    return lm_head, final_norm


# ============================================================
# Surface forms
# ============================================================

def candidate_text_for_form(rel: str, form: str) -> str:
    rel = str(rel).strip().lower()
    cap = rel[:1].upper() + rel[1:]

    if form == "lower_space":
        return " " + rel

    if form == "cap_space":
        return " " + cap

    if form == "lower_nospace":
        return rel

    if form == "cap_nospace":
        return cap

    raise ValueError(f"Unknown surface form: {form}")


# ============================================================
# Layer-wise closed-set scoring
# ============================================================

def layer_name_from_idx(idx: int, num_states: int) -> str:
    if idx == 0:
        return "emb"
    if idx == num_states - 1:
        return "final"
    return f"layer_{idx}"


def pack_scores(scores: Dict[str, float], relations: List[str], gold: str) -> Dict:
    """
    scores = average log-probability per relation.
    probs = softmax over the four average log-prob scores.
    """
    vals = torch.tensor([float(scores[r]) for r in relations], dtype=torch.float32)
    probs = torch.softmax(vals, dim=-1).detach().cpu().tolist()
    vals_list = vals.detach().cpu().tolist()

    pred_idx = int(torch.argmax(vals).item())
    pred = relations[pred_idx]

    score_dict = {r: float(vals_list[i]) for i, r in enumerate(relations)}
    prob_dict = {r: float(probs[i]) for i, r in enumerate(relations)}

    if gold in relations:
        gold_idx = relations.index(gold)
        gold_score = float(vals_list[gold_idx])
        gold_prob = float(probs[gold_idx])
        best_non_gold_idx = max(
            [i for i in range(len(relations)) if i != gold_idx],
            key=lambda i: vals_list[i],
        )
        best_non_gold = relations[best_non_gold_idx]
        best_non_gold_score = float(vals_list[best_non_gold_idx])
        gold_margin = gold_score - best_non_gold_score
    else:
        gold_score = None
        gold_prob = None
        best_non_gold = None
        best_non_gold_score = None
        gold_margin = None

    return {
        "pred": pred,
        "correct": bool(pred == gold),
        "scores": score_dict,
        "probs": prob_dict,
        "gold_score": gold_score,
        "gold_prob": gold_prob,
        "best_non_gold": best_non_gold,
        "best_non_gold_score": best_non_gold_score,
        "gold_margin": gold_margin,
    }


@torch.no_grad()
def score_one_surface_form_layerwise(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    candidates: List[str],
    gold: str,
    form: str,
    device: str,
    apply_final_norm: bool = True,
    debug: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    One surface form only.

    Saved values:
      scores[rel] = average log-prob of candidate tokens.
      logits[rel] = average raw logit of candidate tokens.
      probs[rel]  = softmax over scores[left/right/on/under].
    """
    tokenizer = processor.tokenizer

    rows = []
    full_texts = []
    images = []

    for cand in candidates:
        cand_text = candidate_text_for_form(cand, form)
        rows.append(
            {
                "form": form,
                "relation": cand,
                "candidate_text": cand_text,
            }
        )
        full_texts.append(prompt + cand_text)
        images.append(image)

    inputs = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

    if outputs.hidden_states is None:
        raise RuntimeError("outputs.hidden_states is None.")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    lm_head, final_norm = get_lm_head_and_final_norm(model)

    hidden_states = outputs.hidden_states
    num_states = len(hidden_states)

    target_positions_by_row = []
    answer_token_ids_by_row = []
    answer_token_texts_by_row = []

    for b, row in enumerate(rows):
        cand_text = row["candidate_text"]
        cand_ids = tokenizer(
            cand_text,
            add_special_tokens=False,
        ).input_ids

        n_tok = len(cand_ids)
        if n_tok == 0:
            raise RuntimeError(f"Empty candidate tokenization: {cand_text!r}")

        nonpad_positions = torch.nonzero(
            attention_mask[b],
            as_tuple=False,
        ).squeeze(-1)

        cand_positions_in_input = nonpad_positions[-n_tok:]
        cand_positions_in_target = cand_positions_in_input - 1
        cand_positions_in_target = cand_positions_in_target[
            cand_positions_in_target >= 0
        ]

        target_positions_by_row.append(cand_positions_in_target)

        answer_ids = input_ids[b, cand_positions_in_input].detach().cpu().tolist()
        answer_ids = [int(x) for x in answer_ids]
        answer_token_ids_by_row.append(answer_ids)

        answer_text = tokenizer.decode(
            answer_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        answer_token_texts_by_row.append(answer_text)

    debug_info = {
        "form": form,
        "input_ids_shape": tuple(input_ids.shape),
        "outputs_logits_shape": tuple(outputs.logits.shape),
        "attention_sum": attention_mask.sum(dim=1).detach().cpu().tolist(),
        "rows": rows,
        "answer_token_ids_by_row": answer_token_ids_by_row,
        "answer_token_texts_by_row": answer_token_texts_by_row,
    }

    if debug:
        image_token_id = getattr(model.config, "image_token_index", None)
        if image_token_id is None:
            image_token_id = tokenizer.convert_tokens_to_ids("<image>")

        print(f"[DEBUG FORM={form}]")
        print("  input_ids.shape:", tuple(input_ids.shape))
        print("  outputs.logits.shape:", tuple(outputs.logits.shape))
        print(
            "  hidden_states:",
            [tuple(h.shape) for h in hidden_states[:3]],
            "...",
            tuple(hidden_states[-1].shape),
        )
        print("  attention_sum:", debug_info["attention_sum"])
        print("  image_token_id:", image_token_id)

        if image_token_id is not None:
            print(
                "  num_image_tokens:",
                (input_ids == image_token_id).sum(dim=1).detach().cpu().tolist(),
            )

        for b, row in enumerate(rows):
            print(
                f"  form={row['form']:>13s}",
                f"cand={row['relation']:>5s}",
                "candidate_text=",
                repr(row["candidate_text"]),
                "answer_ids=",
                answer_token_ids_by_row[b],
                "answer_text=",
                repr(answer_token_texts_by_row[b]),
                "target_positions=",
                target_positions_by_row[b].detach().cpu().tolist(),
            )

    layer_records = []

    for layer_idx, h in enumerate(hidden_states):
        is_final = layer_idx == num_states - 1

        if is_final:
            layer_logits = outputs.logits[:, :-1, :]
        else:
            h_use = h
            if apply_final_norm and final_norm is not None:
                h_use = final_norm(h_use)
            layer_logits = lm_head(h_use[:, :-1, :])

        raw_logits = layer_logits.float()
        log_probs = F.log_softmax(raw_logits, dim=-1)

        scores = {}
        logits_score = {}
        nested = {}

        for b, row in enumerate(rows):
            rel = row["relation"]
            pos = target_positions_by_row[b]

            if pos.numel() == 0:
                sum_lp = float("-inf")
                avg_lp = float("-inf")
                sum_logit = float("-inf")
                avg_logit = float("-inf")
                n_used = 0
                token_lps = []
                token_logits = []
            else:
                target_ids = input_ids[b, pos + 1]

                vals = log_probs[b, pos, :].gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)

                vals_logit = raw_logits[b, pos, :].gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)

                sum_lp = float(vals.sum().detach().cpu().item())
                avg_lp = float(vals.mean().detach().cpu().item())

                sum_logit = float(vals_logit.sum().detach().cpu().item())
                avg_logit = float(vals_logit.mean().detach().cpu().item())

                n_used = int(vals.numel())
                token_lps = [float(x) for x in
