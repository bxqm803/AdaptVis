import argparse
import contextlib
import json
import math
import random
import re
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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

try:
    from transformers import AutoModelForZeroShotObjectDetection as GroundingDINOModel
except ImportError:
    from transformers import GroundingDinoForObjectDetection as GroundingDINOModel

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
except Exception as e:
    apply_rotary_pos_emb = None
    repeat_kv = None
    LLAMA_IMPORT_ERROR = e
else:
    LLAMA_IMPORT_ERROR = None

from dataset_zoo import get_dataset


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "a272c74"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

RELATIONS = ["left", "right", "on", "under"]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
ID2REL = {i: r for r, i in REL2ID.items()}


# ============================================================
# Dataset / prompt helpers
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
    x = re.sub(r"\s+", " ", x).strip()
    return x


def remove_answer_suffix(q: str) -> str:
    q = clean_question(q)
    q = re.sub(r"Answer\s+with\s+.*$", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"Choose\s+from\s+.*$", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"Options?\s*:.*$", "", q, flags=re.IGNORECASE).strip()
    return q.strip()


def parse_two_objects_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Supports:
      Where is/are X in relation to Y?
      What is/are X in relation to Y?
      Where is/are X relative to Y?
      What is/are the spatial relation between X and Y?
    """
    q = remove_answer_suffix(prompt)

    patterns = [
        r"(?:where|what)\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+in\s+relation\s+to\s+(?:the\s+)?(.+?)\?",
        r"(?:where|what)\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+relative\s+to\s+(?:the\s+)?(.+?)\?",
        r"(?:what)\s+(?:is|are)\s+(?:the\s+)?spatial\s+relation\s+(?:of|between)\s+(?:the\s+)?(.+?)\s+(?:to|and)\s+(?:the\s+)?(.+?)\?",
        r"(?:what)\s+(?:is|are)\s+(?:the\s+)?(?:relationship|relation)\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\?",
    ]

    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            obj1 = strip_article(m.group(1))
            obj2 = strip_article(m.group(2))
            return obj1, obj2

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
# LLaVA image preprocessing
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
        print("[WARN] attn_implementation='eager' not supported.")
        print("[WARN] output_attentions may be None unless transformers supports eager attention.")
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

    print("[INFO] loaded LLaVA")
    print("  model:", type(model))
    print("  processor:", type(processor))
    print("  patch_size:", processor.patch_size)
    print("  vision_feature_select_strategy:", processor.vision_feature_select_strategy)

    return processor, model


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
    text = str(phrase).strip()
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
# Tokens / boxes / metrics
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
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    pos = torch.nonzero(input_ids == image_token_id, as_tuple=False).squeeze(-1)

    if pos.numel() == 0:
        raise RuntimeError("No image token found in input_ids.")

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


def entropy_metrics(vec: torch.Tensor, eps: float = 1e-12) -> Dict:
    vec = vec.float()
    total = float(vec.sum().item())

    if total <= eps:
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


def rel_from_logits(vocab_logits: torch.Tensor, rel_token_ids: torch.Tensor) -> Dict:
    rel_token_ids = rel_token_ids.to(vocab_logits.device)
    rel_logits = vocab_logits.index_select(dim=-1, index=rel_token_ids).float()
    rel_probs = torch.softmax(rel_logits, dim=-1)
    pred_id = int(rel_probs.argmax().item())

    top2 = torch.topk(rel_probs, k=2)
    margin = float((top2.values[0] - top2.values[1]).item())

    out = {
        "pred_id": pred_id,
        "pred": RELATIONS[pred_id],
        "margin": margin,
        "rel_logits": rel_logits.detach().cpu(),
        "rel_probs": rel_probs.detach().cpu(),
    }
    return out


def collect_attention_metrics(
    attentions,
    image_positions: torch.Tensor,
    last_pos: int,
    attn_layers: List[int],
    obj1_ids: List[int],
    obj2_ids: List[int],
) -> Dict:
    rows = []

    for layer_id in attn_layers:
        if layer_id <= 0:
            continue

        attn_idx = layer_id - 1
        if attn_idx < 0 or attn_idx >= len(attentions):
            continue

        A = attentions[attn_idx]  # [1, heads, tgt, src]
        A_img = A[0, :, last_pos, :].index_select(dim=-1, index=image_positions)
        A_img_mean = A_img.mean(dim=0).detach().float().cpu()

        m = attention_box_metrics(A_img_mean, obj1_ids, obj2_ids)
        m["attn_layer"] = layer_id
        rows.append(m)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    avg = df.drop(columns=["attn_layer"]).mean(numeric_only=True).to_dict()
    return avg


# ============================================================
# Attention intervention
# ============================================================

class AttentionScalingController:
    """
    Post-softmax renormalized attention scaling.

    For selected decoder layers and selected query position:
      A[..., target_key_positions] *= alpha
      A = A / sum(A)

    This directly tests whether boosting/suppressing specific key groups
    changes final relation logits.
    """
    def __init__(
        self,
        layers_hidden_idx: List[int],
        target_positions: torch.Tensor,
        query_position: int,
        alpha: float,
        heads: Optional[List[int]] = None,
    ):
        self.layers_hidden_idx = set(int(x) for x in layers_hidden_idx)
        self.target_positions = target_positions
        self.query_position = int(query_position)
        self.alpha = float(alpha)
        self.heads = heads

    def should_apply(self, decoder_layer_idx: int) -> bool:
        # decoder layer 0 corresponds to hidden state layer 1
        hidden_layer_idx = decoder_layer_idx + 1
        return hidden_layer_idx in self.layers_hidden_idx

    def modify(self, decoder_layer_idx: int, attn_probs: torch.Tensor) -> torch.Tensor:
        if not self.should_apply(decoder_layer_idx):
            return attn_probs

        if self.target_positions is None or self.target_positions.numel() == 0:
            return attn_probs

        q = self.query_position
        target = self.target_positions.to(attn_probs.device)

        attn_probs = attn_probs.clone()

        if self.heads is None:
            slice_q = attn_probs[:, :, q, :]
            slice_q[:, :, target] = slice_q[:, :, target] * self.alpha
            slice_q = slice_q / (slice_q.sum(dim=-1, keepdim=True) + 1e-12)
            attn_probs[:, :, q, :] = slice_q
        else:
            heads = torch.tensor(self.heads, dtype=torch.long, device=attn_probs.device)
            slice_q = attn_probs[:, heads, q, :]
            slice_q[:, :, target] = slice_q[:, :, target] * self.alpha
            slice_q = slice_q / (slice_q.sum(dim=-1, keepdim=True) + 1e-12)
            attn_probs[:, heads, q, :] = slice_q

        return attn_probs


def get_decoder_layers(model):
    candidates = [
        "language_model.model.layers",
        "model.layers",
        "language_model.layers",
    ]

    cur = None
    for path in candidates:
        obj = model
        ok = True
        for p in path.split("."):
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok:
            cur = obj
            break

    if cur is None:
        raise RuntimeError("Cannot find decoder layers in model.")

    return cur


def make_patched_llama_attention_forward(controller: AttentionScalingController):
    if apply_rotary_pos_emb is None or repeat_kv is None:
        raise RuntimeError(f"Cannot import LLaMA attention utilities: {LLAMA_IMPORT_ERROR}")

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        num_heads = getattr(self, "num_heads")
        num_kv_heads = getattr(self, "num_key_value_heads")
        num_kv_groups = getattr(self, "num_key_value_groups")
        head_dim = getattr(self, "head_dim")

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        if position_embeddings is None:
            # older/newer transformers compatibility
            try:
                cos, sin = self.rotary_emb(value_states, position_ids)
            except TypeError:
                kv_seq_len = key_states.shape[-2]
                if past_key_value is not None:
                    try:
                        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                    except Exception:
                        pass
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, num_kv_groups)
        value_states = repeat_kv(value_states, num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        layer_idx = int(getattr(self, "layer_idx", -1))
        attn_weights = controller.modify(layer_idx, attn_weights)

        dropout_p = float(getattr(self, "attention_dropout", 0.0))
        attn_weights_drop = F.dropout(attn_weights, p=dropout_p, training=self.training)

        attn_output = torch.matmul(attn_weights_drop, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights_to_return = None
        else:
            attn_weights_to_return = attn_weights

        return attn_output, attn_weights_to_return, past_key_value

    return patched_forward


@contextlib.contextmanager
def patch_llama_attentions(model, controller: AttentionScalingController):
    layers = get_decoder_layers(model)
    patched_forward = make_patched_llama_attention_forward(controller)

    originals = []
    try:
        for layer in layers:
            attn = layer.self_attn
            originals.append((attn, attn.forward))
            attn.forward = types.MethodType(patched_forward, attn)
        yield
    finally:
        for attn, orig_forward in originals:
            attn.forward = orig_forward


def local_patch_ids_for_target(
    mode: str,
    n_img: int,
    obj1_ids: List[int],
    obj2_ids: List[int],
    rng: random.Random,
) -> List[int]:
    obj1_ids = [i for i in obj1_ids if 0 <= i < n_img]
    obj2_ids = [i for i in obj2_ids if 0 <= i < n_img]
    pair_ids = union_ids(obj1_ids, obj2_ids)
    all_ids = list(range(n_img))
    bg_ids = sorted(set(all_ids) - set(pair_ids))

    if mode == "global":
        return all_ids

    if mode == "object1":
        return obj1_ids

    if mode == "object2":
        return obj2_ids

    if mode == "object_pair":
        return pair_ids

    if mode == "background":
        return bg_ids

    if mode == "random":
        k = max(1, len(pair_ids))
        k = min(k, n_img)
        return sorted(rng.sample(all_ids, k))

    raise ValueError(f"Unknown target mode: {mode}")


# ============================================================
# Forward + collection
# ============================================================

@torch.no_grad()
def run_forward_collect(
    model,
    inputs,
    rel_token_ids: torch.Tensor,
    image_positions: torch.Tensor,
    last_pos: int,
    attn_layers: List[int],
    obj1_ids: List[int],
    obj2_ids: List[int],
) -> Dict:
    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    if outputs.attentions is None:
        raise RuntimeError(
            "outputs.attentions is None. "
            "Use attn_implementation='eager' or compatible transformers."
        )

    # Pure final output: exactly the model output logits.
    final_vocab_logits = outputs.logits[0, last_pos, :].detach().float()
    rel = rel_from_logits(final_vocab_logits, rel_token_ids.to(final_vocab_logits.device))

    attn_metrics = collect_attention_metrics(
        attentions=outputs.attentions,
        image_positions=image_positions,
        last_pos=last_pos,
        attn_layers=attn_layers,
        obj1_ids=obj1_ids,
        obj2_ids=obj2_ids,
    )

    return {
        "rel": rel,
        "attn_metrics": attn_metrics,
    }


def add_rel_values(row: Dict, prefix: str, rel_obj: Dict):
    row[f"{prefix}_pred"] = rel_obj["pred"]
    row[f"{prefix}_pred_id"] = rel_obj["pred_id"]
    row[f"{prefix}_margin"] = rel_obj["margin"]

    logits = rel_obj["rel_logits"]
    probs = rel_obj["rel_probs"]

    for i, r in enumerate(RELATIONS):
        row[f"{prefix}_logit_{r}"] = float(logits[i].item())
        row[f"{prefix}_prob_{r}"] = float(probs[i].item())


def add_metric_values(row: Dict, prefix: str, metrics: Dict):
    for k, v in metrics.items():
        row[f"{prefix}_{k}"] = float(v)

、


def safe_add_delta_values(row: Dict, before_prefix: str, after_prefix: str):
    for r in RELATIONS:
        row[f"delta_logit_{r}"] = row[f"{after_prefix}_logit_{r}"] - row[f"{before_prefix}_logit_{r}"]
        row[f"delta_prob_{r}"] = row[f"{after_prefix}_prob_{r}"] - row[f"{before_prefix}_prob_{r}"]

    metric_names = [
        "image_mass",
        "entropy_norm",
        "effective_patches",
        "top1_in_image",
        "top5_in_image",
        "obj1_ratio_in_image",
        "obj2_ratio_in_image",
        "pair_ratio_in_image",
        "background_ratio_in_image",
        "pair_balance",
    ]

    for m in metric_names:
        b = row.get(f"{before_prefix}_{m}", np.nan)
        a = row.get(f"{after_prefix}_{m}", np.nan)
        try:
            row[f"delta_{m}"] = float(a) - float(b)
        except Exception:
            row[f"delta_{m}"] = np.nan


def parse_experiments(spec: str):
    """
    spec example:
      global:1.5,object_pair:1.5,background:1.5,random:1.5,global:0.7,object_pair:0.7
    """
    exps = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        mode, alpha = item.split(":")
        exps.append((mode.strip(), float(alpha)))
    return exps


# ============================================================
# Sample processing
# ============================================================

@torch.no_grad()
def process_one_sample(
    sid: int,
    model,
    processor,
    gdino_processor,
    gdino_model,
    dataset,
    prompt_rows,
    rel_token_ids,
    args,
    attn_layers: List[int],
    intervention_layers: List[int],
    experiments: List[Tuple[str, float]],
    rng: random.Random,
):
    prompt = prompt_rows[sid].get("question", "")
    gold = get_gold_from_prompt_row(prompt_rows[sid])
    obj1, obj2 = parse_two_objects_from_prompt(prompt)

    if obj1 is None or obj2 is None:
        return [], {"parse_fail": 1, "parse_prompt": clean_question(prompt)}

    raw = get_raw_pil_from_dataset(dataset, sid)
    processed, _ = make_processed_pil_like_llava(
        raw=raw,
        image_processor=processor.image_processor,
        force_mode=args.preprocess_mode,
    )
    image = processed if args.image_source == "processed" else raw

    box1, score1 = detect_one(
        image=processed,
        phrase=obj1,
        processor=gdino_processor,
        model=gdino_model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    box2, score2 = detect_one(
        image=processed,
        phrase=obj2,
        processor=gdino_processor,
        model=gdino_model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    image_size = processed.size[0]
    obj1_ids, grid_box1 = box_to_patch_ids(
        box1,
        image_size=image_size,
        patch_size=args.patch_size,
    )
    obj2_ids, grid_box2 = box_to_patch_ids(
        box2,
        image_size=image_size,
        patch_size=args.patch_size,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(args.device)

    attention_mask = inputs["attention_mask"]
    nonpad_positions = torch.nonzero(attention_mask[0], as_tuple=False).squeeze(-1)
    last_pos = int(nonpad_positions[-1].item())

    image_positions = get_image_token_positions(inputs, model, processor).to(args.device)
    n_img = int(image_positions.numel())

    base = run_forward_collect(
        model=model,
        inputs=inputs,
        rel_token_ids=rel_token_ids,
        image_positions=image_positions,
        last_pos=last_pos,
        attn_layers=attn_layers,
        obj1_ids=obj1_ids,
        obj2_ids=obj2_ids,
    )

    base_pred = base["rel"]["pred"]
    base_correct = bool(gold in REL2ID and base_pred == gold)

    rows = []

    for target_mode, alpha in experiments:
        local_target_ids = local_patch_ids_for_target(
            mode=target_mode,
            n_img=n_img,
            obj1_ids=obj1_ids,
            obj2_ids=obj2_ids,
            rng=rng,
        )

        if len(local_target_ids) == 0:
            continue

        target_local_t = torch.tensor(local_target_ids, dtype=torch.long, device=args.device)
        target_positions = image_positions.index_select(dim=0, index=target_local_t)

        controller = AttentionScalingController(
            layers_hidden_idx=intervention_layers,
            target_positions=target_positions,
            query_position=last_pos,
            alpha=alpha,
            heads=None,
        )

        with patch_llama_attentions(model, controller):
            after = run_forward_collect(
                model=model,
                inputs=inputs,
                rel_token_ids=rel_token_ids,
                image_positions=image_positions,
                last_pos=last_pos,
                attn_layers=attn_layers,
                obj1_ids=obj1_ids,
                obj2_ids=obj2_ids,
            )

        after_pred = after["rel"]["pred"]
        after_correct = bool(gold in REL2ID and after_pred == gold)

        row = {
            "sample_id": sid,
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
            "target_mode": target_mode,
            "alpha": alpha,
            "intervention_layers": ",".join(map(str, intervention_layers)),
            "attn_layers": ",".join(map(str, attn_layers)),
            "target_patch_count": len(local_target_ids),
            "n_image_tokens": n_img,
            "base_correct": base_correct,
            "after_correct": after_correct,
            "corrected": bool((not base_correct) and after_correct),
            "damaged": bool(base_correct and (not after_correct)),
            "pred_changed": bool(base_pred != after_pred),
        }

        add_rel_values(row, "base", base["rel"])
        add_rel_values(row, "after", after["rel"])

        add_metric_values(row, "base_attn", base["attn_metrics"])
        add_metric_values(row, "after_attn", after["attn_metrics"])

        safe_add_delta_values(row, "base", "after")
        safe_add_delta_values(row, "base_attn", "after_attn")

        if gold in REL2ID:
            gid = REL2ID[gold]
            row["base_gold_logit"] = row[f"base_logit_{gold}"]
            row["after_gold_logit"] = row[f"after_logit_{gold}"]
            row["delta_gold_logit"] = row[f"delta_logit_{gold}"]

            row["base_gold_prob"] = row[f"base_prob_{gold}"]
            row["after_gold_prob"] = row[f"after_prob_{gold}"]
            row["delta_gold_prob"] = row[f"delta_prob_{gold}"]
        else:
            row["base_gold_logit"] = np.nan
            row["after_gold_logit"] = np.nan
            row["delta_gold_logit"] = np.nan
            row["base_gold_prob"] = np.nan
            row["after_gold_prob"] = np.nan
            row["delta_gold_prob"] = np.nan

        rows.append(row)

    return rows, {"parse_fail": 0}


# ============================================================
# Summary
# ============================================================

def print_summary(df: pd.DataFrame):
    print("\n================ SUMMARY ================")
    print("rows:", len(df))
    print("unique samples:", df["sample_id"].nunique())

    print("\n[base pred ratio]")
    print(df.drop_duplicates("sample_id")["base_pred"].value_counts(normalize=True).reindex(RELATIONS).fillna(0).to_string())

    print("\n[by experiment]")
    cols = [
        "base_correct",
        "after_correct",
        "corrected",
        "damaged",
        "pred_changed",
        "delta_gold_logit",
        "delta_gold_prob",
        "delta_pair_ratio_in_image",
        "delta_background_ratio_in_image",
        "delta_pair_balance",
        "delta_effective_patches",
    ]
    cols = [c for c in cols if c in df.columns]

    g = (
        df.groupby(["target_mode", "alpha"])[cols]
        .mean()
        .reset_index()
        .sort_values(["after_correct", "corrected"], ascending=False)
    )
    print(g.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n[pred ratio after by experiment]")
    for (mode, alpha), sub in df.groupby(["target_mode", "alpha"]):
        print(f"\n--- target={mode}, alpha={alpha} ---")
        print(sub["after_pred"].value_counts(normalize=True).reindex(RELATIONS).fillna(0).to_string())

    print("\n[per gold by experiment]")
    rows = []
    for (mode, alpha, gold), sub in df.groupby(["target_mode", "alpha", "gold"]):
        rows.append({
            "target_mode": mode,
            "alpha": alpha,
            "gold": gold,
            "n": len(sub),
            "base_acc": sub["base_correct"].mean(),
            "after_acc": sub["after_correct"].mean(),
            "corrected": sub["corrected"].mean(),
            "damaged": sub["damaged"].mean(),
            "delta_gold_logit": sub["delta_gold_logit"].mean(),
            "delta_gold_prob": sub["delta_gold_prob"].mean(),
            "delta_pair_ratio": sub.get("delta_pair_ratio_in_image", pd.Series([np.nan])).mean(),
            "delta_balance": sub.get("delta_pair_balance", pd.Series([np.nan])).mean(),
        })

    per_gold = pd.DataFrame(rows)
    print(
        per_gold.sort_values(["target_mode", "alpha", "gold"])
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\n[mechanism hint]")
    print("Grounding-like effect: corrected samples should show delta_pair_ratio > 0, delta_pair_balance > 0, delta_background < 0.")
    print("Prior-shift effect: one relation logit/prob rises broadly even without pair_ratio improvement; check after pred ratio and delta_logit_*.")


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
    parser.add_argument("--intervention-layers", default="16,17,18,19,20")
    parser.add_argument("--relation-form", default="lower_nospace")

    parser.add_argument(
        "--experiments",
        default="global:1.5,object_pair:1.5,background:1.5,random:1.5,global:0.7,object_pair:0.7",
        help="Comma list like global:1.5,object_pair:1.5,background:1.5,random:1.5",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        args.dtype = "float32"
    args.device = device

    out_dir = Path(
        args.out_dir
        or f"output/stage1_attention_scaling_mechanism_{args.dataset}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_layers = [int(x) for x in args.attn_layers.split(",") if x.strip()]
    intervention_layers = [int(x) for x in args.intervention_layers.split(",") if x.strip()]
    experiments = parse_experiments(args.experiments)

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[INFO] dataset:", args.dataset)
    print("[INFO] out_dir:", out_dir)
    print("[INFO] device:", device)
    print("[INFO] attn_layers:", attn_layers)
    print("[INFO] intervention_layers:", intervention_layers)
    print("[INFO] experiments:", experiments)

    print("[INFO] loading LLaVA")
    processor, model = load_llava_hf(
        model_id=args.llava_model_id,
        revision=args.llava_revision,
        cache_dir=args.root_dir,
        device=device,
        dtype=args.dtype,
    )

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
            rng.shuffle(indices)
            indices = indices[: args.max_samples]

    print("[INFO] samples requested:", len(indices))

    all_rows = []
    parse_fail_examples = []
    parse_fail_n = 0

    for sid in tqdm(indices, desc="mechanism probe"):
        try:
            rows, meta = process_one_sample(
                sid=sid,
                model=model,
                processor=processor,
                gdino_processor=gdino_processor,
                gdino_model=gdino_model,
                dataset=dataset,
                prompt_rows=prompt_rows,
                rel_token_ids=rel_token_ids,
                args=args,
                attn_layers=attn_layers,
                intervention_layers=intervention_layers,
                experiments=experiments,
                rng=rng,
            )

            if meta.get("parse_fail", 0):
                parse_fail_n += 1
                if len(parse_fail_examples) < 30:
                    parse_fail_examples.append((sid, meta.get("parse_prompt", "")))
                continue

            all_rows.extend(rows)

        except RuntimeError as e:
            print(f"\n[ERROR] sid={sid}: {e}")
            if "out of memory" in str(e).lower() and device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\n[WARN] sid={sid} skipped: {repr(e)}")
            continue

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n[PARSE]")
    print("  parse_fail_n:", parse_fail_n)
    if parse_fail_examples:
        print("  failed examples:")
        for sid, q in parse_fail_examples:
            print(f"    sid={sid}: {q}")

    if not all_rows:
        raise RuntimeError("No rows collected.")

    df = pd.DataFrame(all_rows)

    csv_path = out_dir / "attention_scaling_mechanism_probe.csv"
    df.to_csv(csv_path, index=False)
    print("\n[SAVED]", csv_path)

    print_summary(df)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
