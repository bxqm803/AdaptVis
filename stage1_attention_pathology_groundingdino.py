import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import LlavaForConditionalGeneration as LlavaModel
except ImportError:
    from transformers import AutoModelForVision2Seq as LlavaModel

try:
    from transformers import AutoModelForZeroShotObjectDetection as GroundingDINOModel
except ImportError:
    from transformers import GroundingDinoForObjectDetection as GroundingDINOModel

from dataset_zoo import get_dataset


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "a272c74"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

RELATIONS = ["left", "right", "on", "under"]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}


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


def clean_question(q: str) -> str:
    q = str(q)
    q = q.replace("<image>", " ")
    q = q.replace("USER:", " ")
    q = q.replace("ASSISTANT:", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def strip_article(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[.?!,:;]+$", "", x)
    return x.strip()


def parse_two_objects_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    q = clean_question(prompt)
    q = re.sub(r"Answer\s+with\s+.*$", "", q, flags=re.IGNORECASE).strip()

    patterns = [
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]

    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            return strip_article(m.group(1)), strip_article(m.group(2))

    return None, None


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
# LLaVA-like preprocessing
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
        "do_pad": bool(getattr(image_processor, "do_pad", False)),
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
    }
    return processed, meta


# ============================================================
# Model loading
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

    try:
        model = LlavaModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except TypeError:
        print("[WARN] attn_implementation='eager' not supported by this transformers version.")
        print("[WARN] If output_attentions returns None, upgrade transformers or load model with eager attention.")
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
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.use_cache = False

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
        raise RuntimeError("Cannot find lm_head.")
    if final_norm is None:
        raise RuntimeError("Cannot find final_norm.")

    return lm_head, final_norm


# ============================================================
# GroundingDINO
# ============================================================

@torch.no_grad()
def detect_one(
    image: Image.Image,
    phrase: str,
    processor,
    model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[Optional[List[float]], Optional[float]]:
    text = phrase.strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    try:
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        result = processor.post_process_grounded_object_detection(
            outputs,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    if len(boxes) == 0:
        return None, None

    best = int(torch.argmax(scores).item())
    return boxes[best].detach().cpu().tolist(), float(scores[best].detach().cpu().item())


# ============================================================
# Token / grid helpers
# ============================================================

def candidate_text_for_rel(rel: str, form: str = "lower_nospace") -> str:
    if form == "lower_nospace":
        return rel
    if form == "lower_space":
        return " " + rel
    if form == "cap_nospace":
        return rel.capitalize()
    if form == "cap_space":
        return " " + rel.capitalize()
    raise ValueError(form)


def build_relation_token_ids(tokenizer, form: str = "lower_nospace") -> torch.Tensor:
    ids = []
    print("\n[INFO] relation token ids")
    for rel in RELATIONS:
        text = candidate_text_for_rel(rel, form)
        tid = tokenizer(text, add_special_tokens=False).input_ids
        print(f"  {rel:>5s} text={text!r} ids={tid}")
        if len(tid) != 1:
            print(f"  [WARN] {text!r} has multiple tokens; using first token {tid[0]}")
        ids.append(int(tid[0]))
    return torch.tensor(ids, dtype=torch.long)


def get_image_token_positions(inputs, model, processor) -> torch.Tensor:
    input_ids = inputs["input_ids"][0]

    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            image_token_id = None

    if image_token_id is None:
        raise RuntimeError("Cannot infer image token id.")

    pos = torch.nonzero(input_ids == image_token_id, as_tuple=False).squeeze(-1)

    if pos.numel() == 0:
        raise RuntimeError(
            "No image token found in input_ids. "
            "Check prompt format and processor.patch_size / vision_feature_select_strategy."
        )

    return pos


def box_to_patch_ids(
    box: Optional[List[float]],
    image_size: int = 336,
    patch_size: int = 14,
) -> Tuple[List[int], Optional[Tuple[int, int, int, int]]]:
    if box is None:
        return [], None

    x1, y1, x2, y2 = box
    grid = image_size // patch_size

    c1 = max(0, min(grid - 1, int(math.floor(x1 / patch_size))))
    r1 = max(0, min(grid - 1, int(math.floor(y1 / patch_size))))
    c2 = max(0, min(grid - 1, int(math.ceil(x2 / patch_size)) - 1))
    r2 = max(0, min(grid - 1, int(math.ceil(y2 / patch_size)) - 1))

    ids = []
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            ids.append(r * grid + c)

    return ids, (r1, c1, r2, c2)


def union_ids(a: List[int], b: List[int]) -> List[int]:
    return sorted(set(a) | set(b))


# ============================================================
# Relation logits and attention metrics
# ============================================================

@torch.no_grad()
def relation_probs_from_hidden_states(
    hidden_states,
    last_pos: int,
    rel_token_ids: torch.Tensor,
    final_norm,
    lm_head,
    device: str,
) -> Dict:
    rel_token_ids = rel_token_ids.to(device)
    model_dtype = next(lm_head.parameters()).dtype

    all_logits = []
    all_probs = []

    for h in hidden_states:
        h_last = h[0, last_pos, :].to(model_dtype)
        logits = lm_head(final_norm(h_last)).float()
        rel_logits = logits.index_select(dim=-1, index=rel_token_ids)
        rel_probs = torch.softmax(rel_logits, dim=-1)

        all_logits.append(rel_logits.detach().cpu())
        all_probs.append(rel_probs.detach().cpu())

    layer_logits = torch.stack(all_logits, dim=0)  # [num_hidden_layers, 4]
    layer_probs = torch.stack(all_probs, dim=0)

    return {
        "layer_logits": layer_logits,
        "layer_probs": layer_probs,
    }


def top_margin(prob: torch.Tensor) -> Tuple[int, float]:
    top2 = torch.topk(prob, k=2)
    pred = int(top2.indices[0].item())
    margin = float((top2.values[0] - top2.values[1]).item())
    return pred, margin


def entropy_metrics(vec: torch.Tensor, eps: float = 1e-12) -> Dict:
    vec = vec.float()
    total = float(vec.sum().item())

    if total <= eps:
        n = int(vec.numel())
        return {
            "image_mass": 0.0,
            "entropy": 0.0,
            "entropy_norm": 0.0,
            "effective_patches": 0.0,
            "top1_in_image": 0.0,
            "top5_in_image": 0.0,
        }

    p = vec / vec.sum()
    ent = float((-(p + eps).log() * p).sum().item())
    n = int(p.numel())
    ent_norm = ent / math.log(max(n, 2))
    eff = math.exp(ent)

    topk = torch.topk(p, k=min(5, n)).values

    return {
        "image_mass": total,
        "entropy": ent,
        "entropy_norm": ent_norm,
        "effective_patches": eff,
        "top1_in_image": float(p.max().item()),
        "top5_in_image": float(topk.sum().item()),
    }


def attention_box_metrics(
    att_img_raw: torch.Tensor,
    obj1_ids: List[int],
    obj2_ids: List[int],
    eps: float = 1e-12,
) -> Dict:
    base = entropy_metrics(att_img_raw, eps=eps)
    image_mass = base["image_mass"]

    n_img = int(att_img_raw.numel())
    obj1_ids = [i for i in obj1_ids if 0 <= i < n_img]
    obj2_ids = [i for i in obj2_ids if 0 <= i < n_img]
    pair_ids = union_ids(obj1_ids, obj2_ids)

    def sum_ids(ids):
        if not ids:
            return 0.0
        idx = torch.tensor(ids, dtype=torch.long, device=att_img_raw.device)
        return float(att_img_raw.index_select(0, idx).sum().item())

    obj1_mass = sum_ids(obj1_ids)
    obj2_mass = sum_ids(obj2_ids)
    pair_mass = sum_ids(pair_ids)

    if image_mass <= eps:
        obj1_ratio = obj2_ratio = pair_ratio = background_ratio = balance = 0.0
    else:
        obj1_ratio = obj1_mass / image_mass
        obj2_ratio = obj2_mass / image_mass
        pair_ratio = pair_mass / image_mass
        background_ratio = max(0.0, 1.0 - pair_ratio)

        denom = max(obj1_ratio, obj2_ratio, eps)
        balance = min(obj1_ratio, obj2_ratio) / denom

    base.update(
        {
            "obj1_mass_raw": obj1_mass,
            "obj2_mass_raw": obj2_mass,
            "pair_mass_raw": pair_mass,
            "obj1_ratio_in_image": obj1_ratio,
            "obj2_ratio_in_image": obj2_ratio,
            "pair_ratio_in_image": pair_ratio,
            "background_ratio_in_image": background_ratio,
            "pair_balance": balance,
            "obj1_patch_count": len(obj1_ids),
            "obj2_patch_count": len(obj2_ids),
            "pair_patch_count": len(pair_ids),
        }
    )

    return base


@torch.no_grad()
def compute_sample_diagnostics(
    model,
    processor,
    final_norm,
    lm_head,
    gdino_processor,
    gdino_model,
    dataset,
    prompt_rows,
    sample_id: int,
    device: str,
    rel_token_ids: torch.Tensor,
    image_source: str,
    preprocess_mode: str,
    box_threshold: float,
    text_threshold: float,
    patch_size: int,
    attn_layers: List[int],
    mid_layers: List[int],
    save_heatmap_tensor: bool = False,
):
    prompt = prompt_rows[sample_id].get("question", "")
    gold = get_gold_from_prompt_row(prompt_rows[sample_id])
    obj1, obj2 = parse_two_objects_from_prompt(prompt)

    if obj1 is None or obj2 is None:
        return None

    raw = get_raw_pil_from_dataset(dataset, sample_id)
    processed, meta = make_processed_pil_like_llava(
        raw=raw,
        image_processor=processor.image_processor,
        force_mode=preprocess_mode,
    )
    image = processed if image_source == "processed" else raw

    box1, score1 = detect_one(
        image=processed,
        phrase=obj1,
        processor=gdino_processor,
        model=gdino_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    box2, score2 = detect_one(
        image=processed,
        phrase=obj2,
        processor=gdino_processor,
        model=gdino_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    image_size = processed.size[0]
    grid = image_size // patch_size
    obj1_ids, grid_box1 = box_to_patch_ids(box1, image_size=image_size, patch_size=patch_size)
    obj2_ids, grid_box2 = box_to_patch_ids(box2, image_size=image_size, patch_size=patch_size)

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )

    if outputs.attentions is None:
        raise RuntimeError(
            "outputs.attentions is None. "
            "Load LLaVA with attn_implementation='eager' or upgrade transformers."
        )

    attention_mask = inputs["attention_mask"]
    nonpad_positions = torch.nonzero(attention_mask[0], as_tuple=False).squeeze(-1)
    last_pos = int(nonpad_positions[-1].item())

    image_positions = get_image_token_positions(inputs, model, processor).to(device)

    rel_obj = relation_probs_from_hidden_states(
        hidden_states=outputs.hidden_states,
        last_pos=last_pos,
        rel_token_ids=rel_token_ids,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
    )

    layer_probs = rel_obj["layer_probs"]
    layer_logits = rel_obj["layer_logits"]

    final_probs = layer_probs[-1]
    final_logits = layer_logits[-1]

    mid_valid_layers = [l for l in mid_layers if 0 <= l < layer_probs.shape[0]]
    mid_probs = layer_probs[mid_valid_layers].mean(dim=0)
    mid_logits = layer_logits[mid_valid_layers].mean(dim=0)

    final_pred_id, final_margin = top_margin(final_probs)
    mid_pred_id, mid_margin = top_margin(mid_probs)

    attn_layer_rows = []
    heatmap_sum = None
    heatmap_count = 0

    for layer_id in attn_layers:
        # hidden_state layer_i corresponds to attention tuple index i - 1
        if layer_id <= 0:
            continue

        attn_idx = layer_id - 1
        if attn_idx < 0 or attn_idx >= len(outputs.attentions):
            continue

        A = outputs.attentions[attn_idx]  # [1, heads, tgt, src]
        A_img = A[0, :, last_pos, :].index_select(dim=-1, index=image_positions)  # [heads, n_img]

        # average over heads first
        A_img_mean = A_img.mean(dim=0).detach().float().cpu()

        metrics = attention_box_metrics(
            att_img_raw=A_img_mean,
            obj1_ids=obj1_ids,
            obj2_ids=obj2_ids,
        )
        metrics["attn_layer"] = layer_id
        attn_layer_rows.append(metrics)

        if save_heatmap_tensor:
            if heatmap_sum is None:
                heatmap_sum = A_img_mean.clone()
            else:
                heatmap_sum += A_img_mean
            heatmap_count += 1

    if len(attn_layer_rows) == 0:
        return None

    attn_df = pd.DataFrame(attn_layer_rows)
    avg_metrics = attn_df.drop(columns=["attn_layer"]).mean(numeric_only=True).to_dict()

    row = {
        "sample_id": sample_id,
        "gold": gold,
        "question": clean_question(prompt),
        "obj1": obj1,
        "obj2": obj2,
        "gdino_score1": score1,
        "gdino_score2": score2,
        "box1": json.dumps(box1),
        "box2": json.dumps(box2),
        "grid_box1": json.dumps(grid_box1),
        "grid_box2": json.dumps(grid_box2),
        "final_pred": RELATIONS[final_pred_id],
        "mid_pred": RELATIONS[mid_pred_id],
        "final_margin": final_margin,
        "mid_margin": mid_margin,
        "middle_final_disagree": bool(final_pred_id != mid_pred_id),
        "final_correct": bool(gold in REL2ID and final_pred_id == REL2ID[gold]),
        "mid_correct": bool(gold in REL2ID and mid_pred_id == REL2ID[gold]),
        "processed_width": processed.size[0],
        "processed_height": processed.size[1],
        "grid": grid,
    }

    for i, rel in enumerate(RELATIONS):
        row[f"final_prob_{rel}"] = float(final_probs[i].item())
        row[f"mid_prob_{rel}"] = float(mid_probs[i].item())
        row[f"final_logit_{rel}"] = float(final_logits[i].item())
        row[f"mid_logit_{rel}"] = float(mid_logits[i].item())

    row.update({f"attn_{k}": v for k, v in avg_metrics.items()})

    heatmap = None
    if save_heatmap_tensor and heatmap_sum is not None and heatmap_count > 0:
        heatmap = heatmap_sum / heatmap_count

    return {
        "row": row,
        "processed": processed,
        "heatmap": heatmap,
        "box1": box1,
        "box2": box2,
        "obj1": obj1,
        "obj2": obj2,
    }


# ============================================================
# Classification after collecting metrics
# ============================================================

def classify_rows_with_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eff = df["attn_effective_patches"]
    top5 = df["attn_top5_in_image"]
    pair = df["attn_pair_ratio_in_image"]
    bal = df["attn_pair_balance"]

    high_eff = eff.quantile(0.75)
    low_eff = eff.quantile(0.25)
    high_top5 = top5.quantile(0.75)
    low_pair = pair.quantile(0.33)
    low_balance = bal.quantile(0.33)

    final_low_margin = df["final_margin"].quantile(0.35)
    mid_high_margin = df["mid_margin"].quantile(0.65)

    categories = []

    for _, r in df.iterrows():
        is_stable = (
            r["final_margin"] > final_low_margin
            and r["middle_final_disagree"] is False
        )

        is_diffuse = (
            r["attn_effective_patches"] >= high_eff
            and r["attn_pair_ratio_in_image"] <= low_pair
        )

        is_wrong_region_focus = (
            r["attn_effective_patches"] <= low_eff
            and r["attn_top5_in_image"] >= high_top5
            and r["attn_pair_ratio_in_image"] <= low_pair
        )

        is_too_sharp_one_object = (
            r["attn_effective_patches"] <= low_eff
            and r["attn_top5_in_image"] >= high_top5
            and r["attn_pair_balance"] <= low_balance
            and r["attn_pair_ratio_in_image"] > low_pair
        )

        is_middle_final_mismatch = (
            r["final_margin"] <= final_low_margin
            and r["mid_margin"] >= mid_high_margin
            and bool(r["middle_final_disagree"])
        )

        if is_wrong_region_focus:
            cat = "wrong_region_focus"
        elif is_too_sharp_one_object:
            cat = "too_sharp_one_object"
        elif is_diffuse:
            cat = "too_diffuse"
        elif is_middle_final_mismatch:
            cat = "middle_final_mismatch"
        elif is_stable:
            cat = "stable_no_intervention"
        else:
            cat = "ambiguous"

        categories.append(cat)

    df["pathology"] = categories

    print("\n[INFO] dynamic thresholds")
    print(f"  high_eff(q75)={high_eff:.4f}")
    print(f"  low_eff(q25)={low_eff:.4f}")
    print(f"  high_top5(q75)={high_top5:.4f}")
    print(f"  low_pair(q33)={low_pair:.4f}")
    print(f"  low_balance(q33)={low_balance:.4f}")
    print(f"  final_low_margin(q35)={final_low_margin:.4f}")
    print(f"  mid_high_margin(q65)={mid_high_margin:.4f}")

    return df


# ============================================================
# Visualization
# ============================================================

def draw_grid(draw: ImageDraw.ImageDraw, size: int, grid: int):
    step = size / grid
    for i in range(1, grid):
        x = int(round(i * step))
        y = int(round(i * step))
        draw.line([(x, 0), (x, size)], fill=(210, 210, 210), width=1)
        draw.line([(0, y), (size, y)], fill=(210, 210, 210), width=1)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: Optional[List[float]],
    label: str,
    color: Tuple[int, int, int],
    font,
):
    if box is None:
        return

    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    y_text = max(0, int(y1) - th - 8)
    x_text = max(0, int(x1))
    draw.rectangle([x_text, y_text, x_text + tw + 8, y_text + th + 6], fill=(255, 255, 255))
    draw.text((x_text + 4, y_text + 3), label, fill=color, font=font)


def make_heatmap_overlay(
    img: Image.Image,
    heatmap: Optional[torch.Tensor],
    alpha: float = 0.45,
) -> Image.Image:
    img = img.convert("RGB")

    if heatmap is None:
        return img

    n = int(heatmap.numel())
    grid = int(math.sqrt(n))
    if grid * grid != n:
        return img

    h = heatmap.float()
    h = h / (h.max() + 1e-12)
    h_img = h.view(grid, grid).numpy()

    heat = Image.fromarray(np.uint8(h_img * 255)).resize(img.size, Image.BICUBIC).convert("L")

    red = Image.new("RGB", img.size, (255, 0, 0))
    out = Image.composite(red, img, heat.point(lambda x: int(x * alpha)))
    return out


def make_vis(
    processed: Image.Image,
    heatmap: Optional[torch.Tensor],
    box1,
    box2,
    obj1,
    obj2,
    row: Dict,
) -> Image.Image:
    base = make_heatmap_overlay(processed, heatmap)
    draw = ImageDraw.Draw(base)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    grid = int(row["grid"])
    draw_grid(draw, size=base.size[0], grid=grid)

    draw_box(draw, box1, f"obj1: {obj1}", (255, 0, 0), font)
    draw_box(draw, box2, f"obj2: {obj2}", (0, 80, 255), font)

    panel_h = 150
    out = Image.new("RGB", (base.size[0], base.size[1] + panel_h), (255, 255, 255))
    out.paste(base, (0, 0))
    draw = ImageDraw.Draw(out)

    lines = [
        f"sid={row['sample_id']} | pathology={row.get('pathology', 'NA')} | gold={row['gold']}",
        f"final={row['final_pred']} margin={row['final_margin']:.4f} | mid={row['mid_pred']} margin={row['mid_margin']:.4f}",
        f"eff={row['attn_effective_patches']:.2f} | top5={row['attn_top5_in_image']:.3f} | pair={row['attn_pair_ratio_in_image']:.3f} | balance={row['attn_pair_balance']:.3f}",
        f"obj1={obj1} | obj2={obj2}",
        str(row["question"])[:110],
    ]

    y = base.size[1] + 6
    for line in lines:
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 27

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--option", default="four")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out-dir", default=None)

    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--llava-model-id", default=LLAVA_MODEL_ID)
    parser.add_argument("--llava-revision", default=LLAVA_REVISION)
    parser.add_argument("--grounding-model-id", default=GDINO_MODEL_ID)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "crop", "pad"])
    parser.add_argument("--image-source", default="processed", choices=["processed", "raw"])

    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--patch-size", type=int, default=14)

    parser.add_argument("--attn-layers", default="16,17,18,19,20")
    parser.add_argument("--mid-layers", default="16,17,18,19,20")
    parser.add_argument("--relation-form", default="lower_nospace")

    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-per-category", type=int, default=20)

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        args.dtype = "float32"

    out_dir = Path(
        args.out_dir
        or f"output/stage1_attention_pathology_{args.dataset}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = out_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    attn_layers = [int(x) for x in args.attn_layers.split(",") if x.strip()]
    mid_layers = [int(x) for x in args.mid_layers.split(",") if x.strip()]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[INFO] dataset:", args.dataset)
    print("[INFO] out_dir:", out_dir)
    print("[INFO] device:", device)
    print("[INFO] attn_layers:", attn_layers)
    print("[INFO] mid_layers:", mid_layers)

    print("[INFO] loading LLaVA")
    processor, model = load_llava_hf(
        model_id=args.llava_model_id,
        revision=args.llava_revision,
        cache_dir=args.root_dir,
        device=device,
        dtype=args.dtype,
    )
    final_norm, lm_head = None, None
    lm_head, final_norm = get_lm_head_and_final_norm(model)

    rel_token_ids = build_relation_token_ids(
        tokenizer=processor.tokenizer,
        form=args.relation_form,
    )

    print("[INFO] loading GroundingDINO")
    gdino_processor = AutoProcessor.from_pretrained(args.grounding_model_id)
    gdino_model = GroundingDINOModel.from_pretrained(args.grounding_model_id).to(device).eval()

    print("[INFO] loading dataset")
    dataset = load_dataset_any_signature(
        dataset_name=args.dataset,
        root_dir=args.root_dir,
        download=args.download,
    )
    prompt_rows = load_prompt_rows(args.dataset, args.option)

    n_total = min(len(dataset), len(prompt_rows))

    if args.sample_ids.strip():
        indices = []
        for x in args.sample_ids.split(","):
            x = x.strip()
            if not x:
                continue
            sid = int(x)
            if 0 <= sid < n_total:
                indices.append(sid)
            else:
                print("[WARN] sample_id out of range:", sid)
    else:
        indices = list(range(n_total))
        if args.max_samples > 0:
            random.shuffle(indices)
            indices = indices[: args.max_samples]

    print("[INFO] samples:", len(indices))

    rows = []
    vis_cache = []

    for sid in tqdm(indices, desc="diagnosing samples"):
        try:
            obj = compute_sample_diagnostics(
                model=model,
                processor=processor,
                final_norm=final_norm,
                lm_head=lm_head,
                gdino_processor=gdino_processor,
                gdino_model=gdino_model,
                dataset=dataset,
                prompt_rows=prompt_rows,
                sample_id=sid,
                device=device,
                rel_token_ids=rel_token_ids,
                image_source=args.image_source,
                preprocess_mode=args.preprocess_mode,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                patch_size=args.patch_size,
                attn_layers=attn_layers,
                mid_layers=mid_layers,
                save_heatmap_tensor=args.save_vis,
            )

            if obj is None:
                continue

            rows.append(obj["row"])

            if args.save_vis:
                vis_cache.append(obj)

        except RuntimeError as e:
            print(f"\n[ERROR] sid={sid}: {e}")
            if "out of memory" in str(e).lower() and device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\n[WARN] sid={sid} skipped due to: {repr(e)}")
            continue

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(rows) == 0:
        raise RuntimeError("No valid rows collected.")

    df = pd.DataFrame(rows)
    df = classify_rows_with_quantiles(df)

    csv_path = out_dir / "attention_pathology.csv"
    df.to_csv(csv_path, index=False)
    print("\n[SAVED]", csv_path)

    print("\n================ SUMMARY ================")
    print("n:", len(df))
    print("\n[pathology counts]")
    print(df["pathology"].value_counts().to_string())

    print("\n[final pred ratio]")
    print(df["final_pred"].value_counts(normalize=True).reindex(RELATIONS).fillna(0).to_string())

    print("\n[mid pred ratio]")
    print(df["mid_pred"].value_counts(normalize=True).reindex(RELATIONS).fillna(0).to_string())

    if "final_correct" in df.columns:
        print("\n[accuracy by pathology]")
        print(
            df.groupby("pathology")[["final_correct", "mid_correct"]]
            .mean()
            .sort_values("final_correct")
            .to_string()
        )

    print("\n[mean metrics by pathology]")
    metric_cols = [
        "attn_image_mass",
        "attn_effective_patches",
        "attn_top5_in_image",
        "attn_pair_ratio_in_image",
        "attn_background_ratio_in_image",
        "attn_pair_balance",
        "final_margin",
        "mid_margin",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    print(df.groupby("pathology")[metric_cols].mean().to_string())

    if args.save_vis:
        print("\n[INFO] saving visualizations")
        saved_per_cat = {}

        row_by_sid = {
            int(r["sample_id"]): r
            for _, r in df.iterrows()
        }

        for obj in vis_cache:
            sid = int(obj["row"]["sample_id"])
            if sid not in row_by_sid:
                continue

            row = dict(row_by_sid[sid])
            cat = row["pathology"]
            saved_per_cat.setdefault(cat, 0)

            if saved_per_cat[cat] >= args.vis_per_category:
                continue

            cat_dir = vis_dir / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

            vis = make_vis(
                processed=obj["processed"],
                heatmap=obj["heatmap"],
                box1=obj["box1"],
                box2=obj["box2"],
                obj1=obj["obj1"],
                obj2=obj["obj2"],
                row=row,
            )

            out_path = cat_dir / f"sid{sid}_{row['gold']}_final-{row['final_pred']}_mid-{row['mid_pred']}.png"
            vis.save(out_path)

            saved_per_cat[cat] += 1

        print("[SAVED VIS]", vis_dir)
        print("saved_per_cat:", saved_per_cat)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
