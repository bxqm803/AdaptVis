import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
ID2REL = {i: r for r, i in REL2ID.items()}

DEFAULT_FORMS = ["lower_nospace", "cap_nospace"]


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
# LLaVA-like image preprocessing
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
# LLaVA loading
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
        raise RuntimeError("Cannot find final_norm in model.")

    for p in lm_head.parameters():
        p.requires_grad_(False)
    for p in final_norm.parameters():
        p.requires_grad_(False)

    return lm_head, final_norm


# ============================================================
# Surface forms
# ============================================================

def candidate_text_for_form(rel: str, form: str) -> str:
    rel = str(rel).strip().lower()
    cap = rel[:1].upper() + rel[1:]

    if form == "lower_nospace":
        return rel
    if form == "cap_nospace":
        return cap
    if form == "lower_space":
        return " " + rel
    if form == "cap_space":
        return " " + cap

    raise ValueError(f"Unknown surface form: {form}")


def build_candidate_token_ids(tokenizer, forms: List[str]):
    token_ids_by_form = {}

    print("\n[INFO] candidate token ids")
    for form in forms:
        ids = []
        for rel in RELATIONS:
            text = candidate_text_for_form(rel, form)
            tid = tokenizer(text, add_special_tokens=False).input_ids
            print(f"  form={form:>13s} rel={rel:>5s} text={text!r} ids={tid}")

            if len(tid) != 1:
                print(
                    f"  [WARN] {text!r} has {len(tid)} tokens. "
                    f"Prompt-only tuned lens will use the FIRST token only: {tid[0]}"
                )
            ids.append(int(tid[0]))

        token_ids_by_form[form] = ids

    return token_ids_by_form


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_prompt_only_features_and_teacher(
    model,
    processor,
    dataset,
    prompt_rows: List[dict],
    indices: List[int],
    image_source: str,
    preprocess_mode: str,
    device: str,
    out_dir: Path,
    cache_name: str = "prompt_only_vocab_tuned_lens_cache.pt",
):
    cache_path = out_dir / cache_name

    if cache_path.exists():
        print(f"[INFO] loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    image_processor = processor.image_processor

    all_features = []
    all_teacher_logits = []
    all_labels = []
    all_sample_ids = []
    all_prompts = []
    all_clean_prompts = []

    print("[INFO] extracting prompt-only hidden states and final teacher logits")
    print("[INFO] feature position: last non-padding prompt token")
    print("[INFO] teacher target: outputs.logits at last prompt token")

    for sample_id in tqdm(indices):
        prompt = prompt_rows[sample_id].get("question", "")
        gold = get_gold_from_prompt_row(prompt_rows[sample_id])

        if gold not in RELATIONS:
            print(f"[SKIP] sample_id={sample_id}, invalid gold={gold}")
            continue

        raw = get_raw_pil_from_dataset(dataset, sample_id)
        processed, _ = make_processed_pil_like_llava(
            raw=raw,
            image_processor=image_processor,
            force_mode=preprocess_mode,
        )
        image = processed if image_source == "processed" else raw

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(device)

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("outputs.hidden_states is None.")

        attention_mask = inputs["attention_mask"]
        nonpad_positions = torch.nonzero(
            attention_mask[0],
            as_tuple=False,
        ).squeeze(-1)
        last_pos = int(nonpad_positions[-1].item())

        feat_layers = []
        for h in hidden_states:
            feat_layers.append(h[0, last_pos, :].detach().cpu().to(torch.float16))
        feat_layers = torch.stack(feat_layers, dim=0)

        teacher_logits = outputs.logits[0, last_pos, :].detach().cpu().to(torch.float16)

        all_features.append(feat_layers)
        all_teacher_logits.append(teacher_logits)
        all_labels.append(REL2ID[gold])
        all_sample_ids.append(sample_id)
        all_prompts.append(prompt)
        all_clean_prompts.append(clean_question(prompt))

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    features = torch.stack(all_features, dim=0)          # [N, L, H]
    teacher_logits = torch.stack(all_teacher_logits, 0)  # [N, V]
    labels = torch.tensor(all_labels, dtype=torch.long)

    obj = {
        "features": features,
        "teacher_logits": teacher_logits,
        "labels": labels,
        "sample_ids": all_sample_ids,
        "prompts": all_prompts,
        "clean_prompts": all_clean_prompts,
        "relations": RELATIONS,
        "image_source": image_source,
        "preprocess_mode": preprocess_mode,
        "feature_position": "last_prompt_token",
        "teacher_target": "final_logits_at_last_prompt_token",
    }

    torch.save(obj, cache_path)

    print("[INFO] saved cache:", cache_path)
    print("[INFO] features shape:", tuple(features.shape))
    print("[INFO] teacher_logits shape:", tuple(teacher_logits.shape))
    print("[INFO] labels shape:", tuple(labels.shape))

    return obj


# ============================================================
# Split
# ============================================================

def stratified_split(labels: torch.Tensor, val_ratio: float, seed: int):
    rng = random.Random(seed)

    labels_np = labels.cpu().numpy().tolist()
    train_idx = []
    val_idx = []

    for c in range(len(RELATIONS)):
        idxs = [i for i, y in enumerate(labels_np) if y == c]
        rng.shuffle(idxs)

        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


# ============================================================
# Translators
# ============================================================

class LowRankResidualTranslator(nn.Module):
    """
    h' = h + Up(Down(LN(h))) + bias
    """
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        h = h.float()
        return h + self.up(self.down(self.ln(h))) + self.bias


class DiagonalLinearTranslator(nn.Module):
    """
    h' = scale * h + bias
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h):
        h = h.float()
        return h * self.scale + self.bias


class FullLinearTranslator(nn.Module):
    """
    h' = W h + b
    """
    def __init__(self, hidden_dim: int, init_identity: bool = True):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

        if init_identity:
            nn.init.eye_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, h):
        return self.linear(h.float())


def build_translator(hidden_dim: int, translator_type: str, rank: int):
    if translator_type == "low_rank":
        return LowRankResidualTranslator(hidden_dim=hidden_dim, rank=rank)

    if translator_type == "diagonal_linear":
        return DiagonalLinearTranslator(hidden_dim=hidden_dim)

    if translator_type == "full_linear":
        return FullLinearTranslator(hidden_dim=hidden_dim, init_identity=True)

    raise ValueError(
        f"Unknown translator_type={translator_type}. "
        f"Choose from: low_rank, diagonal_linear, full_linear"
    )


def student_logits_from_translator(
    translator,
    x,
    final_norm,
    lm_head,
):
    h = translator(x.float())

    model_dtype = next(lm_head.parameters()).dtype
    h_model = h.to(model_dtype)

    h_model = final_norm(h_model)
    logits = lm_head(h_model).float()

    return logits


def distill_kl_loss(student_logits, teacher_logits, temperature: float):
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature ** 2)
    return loss


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def relation_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_ids_by_form: Dict[str, List[int]],
):
    out = {}

    for form, ids in token_ids_by_form.items():
        ids_t = torch.tensor(ids, dtype=torch.long, device=logits.device)
        cand_logits = logits.index_select(dim=-1, index=ids_t)  # [N, 4]
        cand_probs = torch.softmax(cand_logits.float(), dim=-1)

        pred = cand_logits.argmax(dim=-1)
        correct = pred.eq(labels.to(logits.device))

        d = {
            "acc": float(correct.float().mean().item()),
            "pred": pred.detach().cpu(),
            "probs": cand_probs.detach().cpu(),
            "logits": cand_logits.detach().cpu(),
        }

        for i, rel in enumerate(RELATIONS):
            d[f"mean_prob_{rel}"] = float(cand_probs[:, i].mean().item())
            d[f"mean_logit_{rel}"] = float(cand_logits[:, i].float().mean().item())

            mask = labels.to(logits.device).eq(i)
            if int(mask.sum().item()) > 0:
                d[f"acc_{rel}"] = float(correct[mask].float().mean().item())
                d[f"n_{rel}"] = int(mask.sum().item())
            else:
                d[f"acc_{rel}"] = None
                d[f"n_{rel}"] = 0

        out[form] = d

    return out


@torch.no_grad()
def evaluate_translator(
    translator,
    X: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    final_norm,
    lm_head,
    token_ids_by_form: Dict[str, List[int]],
    batch_size: int,
    device: str,
    temperature: float,
):
    translator.eval()

    total_kl = 0.0
    total_n = 0
    all_student_logits = []
    all_teacher_logits = []
    all_labels = []

    idx_list = idx.tolist()

    for start in range(0, len(idx_list), batch_size):
        batch_ids = idx_list[start:start + batch_size]

        xb = X[batch_ids].to(device).float()
        tb = teacher_logits[batch_ids].to(device).float()
        yb = labels[batch_ids].to(device)

        sb = student_logits_from_translator(
            translator=translator,
            x=xb,
            final_norm=final_norm,
            lm_head=lm_head,
        )

        kl = distill_kl_loss(sb, tb, temperature=temperature)

        total_kl += float(kl.item()) * len(batch_ids)
        total_n += len(batch_ids)

        all_student_logits.append(sb.detach().cpu())
        all_teacher_logits.append(tb.detach().cpu())
        all_labels.append(yb.detach().cpu())

    student_logits = torch.cat(all_student_logits, dim=0)
    teacher_logits_eval = torch.cat(all_teacher_logits, dim=0)
    labels_eval = torch.cat(all_labels, dim=0)

    student_rel = relation_metrics_from_logits(
        logits=student_logits,
        labels=labels_eval,
        token_ids_by_form=token_ids_by_form,
    )
    teacher_rel = relation_metrics_from_logits(
        logits=teacher_logits_eval,
        labels=labels_eval,
        token_ids_by_form=token_ids_by_form,
    )

    result = {
        "kl": total_kl / max(total_n, 1),
        "student_rel": student_rel,
        "teacher_rel": teacher_rel,
    }

    for form in token_ids_by_form:
        student_pred = student_rel[form]["pred"]
        teacher_pred = teacher_rel[form]["pred"]
        result[f"match_teacher_{form}"] = float(
            student_pred.eq(teacher_pred).float().mean().item()
        )

    return result


def train_one_layer_tuned_lens(
    X_layer: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    layer_idx: int,
    hidden_dim: int,
    translator_type: str,
    rank: int,
    final_norm,
    lm_head,
    token_ids_by_form: Dict[str, List[int]],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    patience: int,
    seed: int,
):
    torch.manual_seed(seed + layer_idx)

    translator = build_translator(
        hidden_dim=hidden_dim,
        translator_type=translator_type,
        rank=rank,
    ).to(device)

    opt = torch.optim.AdamW(
        translator.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_kl = float("inf")
    best_epoch = -1
    wait = 0

    train_ids = train_idx.tolist()

    for ep in range(epochs):
        translator.train()

        rng = random.Random(seed + layer_idx * 100000 + ep)
        rng.shuffle(train_ids)

        for start in range(0, len(train_ids), batch_size):
            batch_ids = train_ids[start:start + batch_size]

            xb = X_layer[batch_ids].to(device).float()
            tb = teacher_logits[batch_ids].to(device).float()

            student = student_logits_from_translator(
                translator=translator,
                x=xb,
                final_norm=final_norm,
                lm_head=lm_head,
            )

            loss = distill_kl_loss(
                student_logits=student,
                teacher_logits=tb,
                temperature=temperature,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_eval = evaluate_translator(
            translator=translator,
            X=X_layer,
            teacher_logits=teacher_logits,
            labels=labels,
            idx=val_idx,
            final_norm=final_norm,
            lm_head=lm_head,
            token_ids_by_form=token_ids_by_form,
            batch_size=batch_size,
            device=device,
            temperature=temperature,
        )
        val_kl = val_eval["kl"]

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            best_epoch = ep
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in translator.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if patience > 0 and wait >= patience:
            break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in translator.state_dict().items()
        }

    translator.load_state_dict(best_state)
    translator.eval()

    train_eval = evaluate_translator(
        translator=translator,
        X=X_layer,
        teacher_logits=teacher_logits,
        labels=labels,
        idx=train_idx,
        final_norm=final_norm,
        lm_head=lm_head,
        token_ids_by_form=token_ids_by_form,
        batch_size=batch_size,
        device=device,
        temperature=temperature,
    )

    val_eval = evaluate_translator(
        translator=translator,
        X=X_layer,
        teacher_logits=teacher_logits,
        labels=labels,
        idx=val_idx,
        final_norm=final_norm,
        lm_head=lm_head,
        token_ids_by_form=token_ids_by_form,
        batch_size=batch_size,
        device=device,
        temperature=temperature,
    )

    return {
        "layer_idx": layer_idx,
        "best_epoch": best_epoch,
        "best_val_kl": best_val_kl,
        "train_eval": train_eval,
        "val_eval": val_eval,
        "state_dict": best_state,
    }


# ============================================================
# Main experiment
# ============================================================

def layer_name_from_idx(layer_idx: int, num_layers: int):
    if layer_idx == 0:
        return "emb"
    if layer_idx == num_layers - 1:
        return "final"
    return f"layer_{layer_idx}"


def run_tuned_lens(
    features: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: List[int],
    final_norm,
    lm_head,
    token_ids_by_form: Dict[str, List[int]],
    out_dir: Path,
    device: str,
    val_ratio: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    translator_type: str,
    rank: int,
    patience: int,
    save_translators: bool,
):
    train_idx, val_idx = stratified_split(labels, val_ratio=val_ratio, seed=seed)

    N, num_layers, hidden_dim = features.shape

    print("\n[INFO] tuned lens training")
    print(f"  N={N}, num_layers={num_layers}, hidden_dim={hidden_dim}")
    print(f"  train={len(train_idx)}, val={len(val_idx)}")
    print(f"  translator_type={translator_type}, rank={rank}")
    print(f"  epochs={epochs}, batch_size={batch_size}")
    print(f"  lr={lr}, weight_decay={weight_decay}, temperature={temperature}, patience={patience}")

    summary_rows = []
    pred_rows = []

    translators_dir = out_dir / "translators"
    if save_translators:
        translators_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in tqdm(range(num_layers), desc="training vocab tuned lens"):
        layer_name = layer_name_from_idx(layer_idx, num_layers)

        result = train_one_layer_tuned_lens(
            X_layer=features[:, layer_idx, :],
            teacher_logits=teacher_logits,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            layer_idx=layer_idx,
            hidden_dim=hidden_dim,
            translator_type=translator_type,
            rank=rank,
            final_norm=final_norm,
            lm_head=lm_head,
            token_ids_by_form=token_ids_by_form,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            temperature=temperature,
            patience=patience,
            seed=seed,
        )

        if save_translators:
            torch.save(
                result["state_dict"],
                translators_dir / f"translator_{translator_type}_layer_{layer_idx:02d}.pt",
            )

        for form in token_ids_by_form:
            train_rel = result["train_eval"]["student_rel"][form]
            val_rel = result["val_eval"]["student_rel"][form]
            val_teacher_rel = result["val_eval"]["teacher_rel"][form]

            row = {
                "translator_type": translator_type,
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "form": form,
                "best_epoch": result["best_epoch"],
                "train_kl": result["train_eval"]["kl"],
                "val_kl": result["val_eval"]["kl"],
                "student_train_acc_gold": train_rel["acc"],
                "student_val_acc_gold": val_rel["acc"],
                "teacher_val_acc_gold": val_teacher_rel["acc"],
                "student_match_teacher_val": result["val_eval"][f"match_teacher_{form}"],
            }

            for rel in RELATIONS:
                row[f"student_val_mean_prob_{rel}"] = val_rel[f"mean_prob_{rel}"]
                row[f"student_val_mean_logit_{rel}"] = val_rel[f"mean_logit_{rel}"]
                row[f"teacher_val_mean_prob_{rel}"] = val_teacher_rel[f"mean_prob_{rel}"]
                row[f"teacher_val_mean_logit_{rel}"] = val_teacher_rel[f"mean_logit_{rel}"]
                row[f"student_val_acc_{rel}"] = val_rel[f"acc_{rel}"]
                row[f"teacher_val_acc_{rel}"] = val_teacher_rel[f"acc_{rel}"]
                row[f"val_n_{rel}"] = val_rel[f"n_{rel}"]

            summary_rows.append(row)

            val_pred = val_rel["pred"].tolist()
            teacher_pred = val_teacher_rel["pred"].tolist()

            for local_i, global_i in enumerate(val_idx.tolist()):
                gold_id = int(labels[global_i].item())
                pred_id = int(val_pred[local_i])
                teacher_id = int(teacher_pred[local_i])

                pred_rows.append(
                    {
                        "translator_type": translator_type,
                        "layer_idx": layer_idx,
                        "layer_name": layer_name,
                        "form": form,
                        "sample_id": sample_ids[global_i],
                        "gold_id": gold_id,
                        "gold": ID2REL[gold_id],
                        "student_pred_id": pred_id,
                        "student_pred": ID2REL[pred_id],
                        "teacher_pred_id": teacher_id,
                        "teacher_pred": ID2REL[teacher_id],
                        "student_correct": bool(pred_id == gold_id),
                        "teacher_correct": bool(teacher_id == gold_id),
                        "match_teacher": bool(pred_id == teacher_id),
                    }
                )

        msg = [f"layer {layer_idx:02d} ({layer_name})"]
        for form in token_ids_by_form:
            val_rel = result["val_eval"]["student_rel"][form]
            val_teacher = result["val_eval"]["teacher_rel"][form]
            msg.append(
                f"{form}: student_acc={val_rel['acc']:.4f}, "
                f"teacher_acc={val_teacher['acc']:.4f}, "
                f"match_teacher={result['val_eval'][f'match_teacher_{form}']:.4f}"
            )
        msg.append(f"val_kl={result['val_eval']['kl']:.4f}")
        print(" | ".join(msg))

        del result
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    pred_df = pd.DataFrame(pred_rows)

    summary_path = out_dir / "vocab_tuned_lens_summary.csv"
    pred_path = out_dir / "vocab_tuned_lens_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    print("\n[SAVED]", summary_path)
    print("[SAVED]", pred_path)

    return summary_df, pred_df


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "crop", "pad"])
    parser.add_argument("--image-source", default="processed", choices=["processed", "raw"])

    parser.add_argument(
        "--surface-forms",
        default="lower_nospace,cap_nospace",
        help="Recommended for prompt-only: lower_nospace,cap_nospace",
    )

    parser.add_argument("--val-ratio", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument(
        "--translator-type",
        default="low_rank",
        choices=["low_rank", "diagonal_linear", "full_linear"],
        help=(
            "low_rank = h + Up(Down(LN(h))) + bias; "
            "diagonal_linear = scale*h + bias; "
            "full_linear = W*h + b."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Only used when --translator-type low_rank.",
    )

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-translators", action="store_true")

    args = parser.parse_args()

    surface_forms = [
        x.strip()
        for x in str(args.surface_forms).split(",")
        if x.strip()
    ]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        args.dtype = "float32"

    out_dir = Path(args.out_dir or f"output/stage1_prompt_only_vocab_tuned_lens_{args.dataset}_{args.translator_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] device={device}, dtype={args.dtype}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] image_source={args.image_source}")
    print(f"[INFO] surface_forms={surface_forms}")
    print(f"[INFO] translator_type={args.translator_type}")

    processor, model = load_llava_hf(
        model_id=args.llava_model_id,
        revision=args.llava_revision,
        cache_dir=args.root_dir,
        device=device,
        dtype=args.dtype,
    )

    lm_head, final_norm = get_lm_head_and_final_norm(model)

    token_ids_by_form = build_candidate_token_ids(
        tokenizer=processor.tokenizer,
        forms=surface_forms,
    )

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
                print(f"[WARN] sample_id out of range and skipped: {sid}")
    else:
        indices = list(range(n_total))
        if args.max_samples > 0:
            random.seed(args.seed)
            indices = random.sample(indices, min(args.max_samples, len(indices)))

    print(f"[INFO] total samples to extract: {len(indices)}")

    obj = extract_prompt_only_features_and_teacher(
        model=model,
        processor=processor,
        dataset=dataset,
        prompt_rows=prompt_rows,
        indices=indices,
        image_source=args.image_source,
        preprocess_mode=args.preprocess_mode,
        device=device,
        out_dir=out_dir,
    )

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_ids": obj["sample_ids"],
                "labels": obj["labels"].tolist(),
                "relations": RELATIONS,
                "image_source": args.image_source,
                "preprocess_mode": args.preprocess_mode,
                "feature_position": obj["feature_position"],
                "teacher_target": obj["teacher_target"],
                "surface_forms": surface_forms,
                "token_ids_by_form": token_ids_by_form,
                "translator_type": args.translator_type,
                "rank": args.rank,
                "temperature": args.temperature,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[SAVED]", metadata_path)

    summary_df, pred_df = run_tuned_lens(
        features=obj["features"],
        teacher_logits=obj["teacher_logits"],
        labels=obj["labels"],
        sample_ids=obj["sample_ids"],
        final_norm=final_norm,
        lm_head=lm_head,
        token_ids_by_form=token_ids_by_form,
        out_dir=out_dir,
        device=device,
        val_ratio=args.val_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        translator_type=args.translator_type,
        rank=args.rank,
        patience=args.patience,
        save_translators=args.save_translators,
    )

    print("\n================ BEST STUDENT LAYERS BY FORM ================")
    for form in surface_forms:
        sub = summary_df[summary_df["form"] == form].copy()
        sub = sub.sort_values("student_val_acc_gold", ascending=False).head(10)
        print(f"\n[FORM={form}]")
        print(
            sub[
                [
                    "translator_type",
                    "layer_idx",
                    "layer_name",
                    "student_val_acc_gold",
                    "teacher_val_acc_gold",
                    "student_match_teacher_val",
                    "val_kl",
                    "best_epoch",
                ]
            ].to_string(index=False)
        )

    print("\n================ ALL LAYERS ================")
    for form in surface_forms:
        sub = summary_df[summary_df["form"] == form].copy()
        print(f"\n[FORM={form}]")
        print(
            sub[
                [
                    "translator_type",
                    "layer_idx",
                    "layer_name",
                    "student_val_acc_gold",
                    "teacher_val_acc_gold",
                    "student_match_teacher_val",
                    "val_kl",
                ]
            ].to_string(index=False)
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
