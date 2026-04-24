import os
import re
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor

from dataset_zoo import get_dataset
from model_zoo import get_model


# =========================================================
# config
# =========================================================
DATASET = "Controlled_Images_A"
OPTION = "four"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"

DEVICE = "cuda"
CACHE_DIR = f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"

SEED = 1
PRINT_FIRST_N = 10
MAX_EXAMPLES = None   # None = 全部；想先少跑点就改成 20 / 50

# soft background suppression
BG_RATIO = 0.20       # 背景保底比例，gate=0 时仍保留 20% token 强度
BOOST = 1.00          # 如需强调前景，可试 1.2 / 1.5
USE_DILATE = True
DILATE_KERNEL = 3

# CLIP 模型 dtype
CLIP_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32


# =========================================================
# utils
# =========================================================
def seed_all(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(x):
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_object_name(name):
    name = clean_text(name).lower()
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"^(a|an|the)\s+", "", name)
    name = re.sub(r"[?.!,;:]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def infer_names_from_filename(image_name):
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    for marker in ["_left_of_", "_right_of_", "_on_", "_under_"]:
        if marker in stem:
            a, b = stem.split(marker, 1)
            return normalize_object_name(a), normalize_object_name(b)
    return "object 1", "object 2"


def parse_objects_from_question(question):
    q = clean_text(question)

    # 去掉可能的包装
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()

    patterns = [
        r"Where is the (.+?) in relation to the (.+?)\?",
        r"Where are the (.+?) in relation to the (.+?)\?",
        r"Where is (.+?) in relation to (.+?)\?",
        r"Where are (.+?) in relation to (.+?)\?",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            obj1 = normalize_object_name(m.group(1))
            obj2 = normalize_object_name(m.group(2))
            return obj1, obj2
    return None, None


def normalize_rel(answer):
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            return None
        answer = answer[0]
    if answer is None:
        return None

    rel = str(answer).strip().lower()
    mapping = {
        "left": "left",
        "right": "right",
        "on": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
        "top": "on",
        "above": "on",
        "to the left of": "left",
        "to the right of": "right",
        "on top of": "on",
    }
    return mapping.get(rel, rel)


def relation_from_image_name(image_name):
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    if "_left_of_" in stem:
        return "left"
    if "_right_of_" in stem:
        return "right"
    if "_on_" in stem:
        return "on"
    if "_under_" in stem:
        return "under"
    return None


def parse_prediction(text):
    text = clean_text(text).lower()
    for cand in ["left", "right", "on", "under"]:
        if re.search(rf"\b{cand}\b", text):
            return cand
    return None


def extract_raw_text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for k in ["response", "text", "pred_text", "output", "answer"]:
            if k in output:
                return str(output[k])
        return str(output)
    if isinstance(output, (list, tuple)) and len(output) > 0:
        return str(output[0])
    return str(output)


def infer_patch_grid(num_patches):
    side = int(round(math.sqrt(num_patches)))
    if side * side == num_patches:
        return side, side
    for h in range(int(math.sqrt(num_patches)), 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    return 1, num_patches


def normalize_1d_torch(x, eps=1e-6):
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


def dilate_patch_mask(mask, gh, gw, kernel_size=3):
    # mask: [B, N]
    b, n = mask.shape
    x = mask.view(b, 1, gh, gw)
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return x.view(b, n)


def resize_gate_to_num_patches(gate, target_num_patches):
    """
    gate: [B, N]
    """
    src_h, src_w = infer_patch_grid(gate.shape[1])
    tgt_h, tgt_w = infer_patch_grid(target_num_patches)

    if src_h == tgt_h and src_w == tgt_w:
        return gate

    x = gate.view(gate.shape[0], 1, src_h, src_w)
    x = F.interpolate(x, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    return x.view(gate.shape[0], tgt_h * tgt_w)


# =========================================================
# CLIP text tower -> inverse soft mask
# =========================================================
@torch.no_grad()
def get_clip_patch_features(clip_model, clip_processor, image_pil, device):
    image_inputs = clip_processor(images=image_pil, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=CLIP_DTYPE)

    vision_outputs = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
    last_hidden = vision_outputs.last_hidden_state  # [1, 1+N, H]

    # 和你之前那版一致：给 vision token 过 post_layernorm
    if hasattr(clip_model.vision_model, "vision_model") and hasattr(clip_model.vision_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.vision_model.post_layernorm(last_hidden)
    elif hasattr(clip_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.post_layernorm(last_hidden)

    patch_tokens = last_hidden[:, 1:, :]  # [1, N, H]
    patch_proj = clip_model.visual_projection(patch_tokens)  # [1, N, D]
    patch_proj = patch_proj / (patch_proj.norm(dim=-1, keepdim=True) + 1e-6)

    return patch_proj


@torch.no_grad()
def get_clip_text_feature(clip_model, clip_processor, text_phrase, device):
    text_inputs = clip_processor(text=[text_phrase], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = clip_model.get_text_features(**text_inputs)  # [1, D]
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    return text_features


@torch.no_grad()
def build_union_gate_from_text_tower(
    clip_model,
    clip_processor,
    image_pil,
    obj1_name,
    obj2_name,
    device,
    use_dilate=True,
    dilate_kernel=3,
):
    """
    你前面已经验证过：
    raw similarity 的 low-score valley 对应目标物体
    所以这里 objectness = normalize(-raw_sims)
    """
    patch_proj = get_clip_patch_features(clip_model, clip_processor, image_pil, device)   # [1, N, D]
    text1 = get_clip_text_feature(clip_model, clip_processor, obj1_name, device)          # [1, D]
    text2 = get_clip_text_feature(clip_model, clip_processor, obj2_name, device)          # [1, D]

    sims1 = torch.matmul(patch_proj[0], text1[0]).unsqueeze(0)   # [1, N]
    sims2 = torch.matmul(patch_proj[0], text2[0]).unsqueeze(0)   # [1, N]

    gate1 = normalize_1d_torch(-sims1)   # low raw score -> high objectness
    gate2 = normalize_1d_torch(-sims2)

    union_gate = torch.maximum(gate1, gate2)   # [1, N]

    gh, gw = infer_patch_grid(union_gate.shape[1])
    if use_dilate:
        union_gate = dilate_patch_mask(union_gate, gh, gw, kernel_size=dilate_kernel)

    union_gate = normalize_1d_torch(union_gate)
    return union_gate


# =========================================================
# patch llava model: only change image features internally
# =========================================================
def install_soft_gate_patch(llava_wrapper, bg_ratio=0.2, boost=1.0):
    """
    直接 patch repo 自定义 LLaVA 的 multi_modal_projector.forward
    位置：vision_tower -> selected_image_feature -> multi_modal_projector -> image_features
    """
    llava_model = llava_wrapper.model

    if not hasattr(llava_model, "multi_modal_projector"):
        raise RuntimeError(
            f"repo-loaded model has no .multi_modal_projector, type={type(llava_model)}"
        )

    original_projector_forward = llava_model.multi_modal_projector.forward

    def patched_projector_forward(image_features):
        # image_features: [B, N, Dv]  (vision tower 输出选中的 patch features)
        feats = original_projector_forward(image_features)  # [B, N, Dt]

        gate = getattr(llava_model, "_current_soft_gate", None)
        if gate is None:
            return feats

        if feats is None or not torch.is_tensor(feats):
            return feats

        gate_local = gate
        if gate_local.ndim == 1:
            gate_local = gate_local.unsqueeze(0)

        if gate_local.shape[0] != feats.shape[0]:
            if gate_local.shape[0] == 1 and feats.shape[0] > 1:
                gate_local = gate_local.expand(feats.shape[0], -1)
            else:
                raise ValueError(
                    f"gate batch {gate_local.shape[0]} != feats batch {feats.shape[0]}"
                )

        if gate_local.shape[1] != feats.shape[1]:
            gate_local = resize_gate_to_num_patches(gate_local, feats.shape[1])

        gate_local = gate_local.to(device=feats.device, dtype=feats.dtype).clamp(0, 1)

        # soft background suppression
        scale = bg_ratio + (1.0 - bg_ratio) * gate_local  # [B, N]
        if boost != 1.0:
            scale = scale * (1.0 + (boost - 1.0) * gate_local)

        feats = feats * scale.unsqueeze(-1)
        return feats

    llava_model.multi_modal_projector.forward = patched_projector_forward
    return llava_model


# =========================================================
# main
# =========================================================
def main():
    seed_all(SEED)

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_DIR, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

    print("Loading LLaVA with repo get_model(...)")
    llava_wrapper, image_preprocess = get_model(
        "llava1.5",
        DEVICE,
        method="base",
        root_dir=CACHE_DIR,
    )

    llava_model = llava_wrapper.model
    install_soft_gate_patch(llava_wrapper, bg_ratio=BG_RATIO, boost=BOOST)

    dataset = get_dataset(DATASET, image_preprocess=image_preprocess, download=False)

    prompt_records, sampled_indices = llava_wrapper.load_prompt_records_with_sampling(DATASET, OPTION)
    if sampled_indices is not None:
        import torch.utils.data
        dataset = torch.utils.data.Subset(dataset, sampled_indices)

    if len(prompt_records) != len(dataset):
        raise ValueError(f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).")

    print(f"Loading CLIP text tower: {CLIP_MODEL_ID}")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=CACHE_DIR)
    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        torch_dtype=CLIP_DTYPE,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    ).to(DEVICE).eval()

    total = 0
    correct = 0
    shown = 0

    num_examples = len(dataset) if MAX_EXAMPLES is None else min(MAX_EXAMPLES, len(dataset))

    for idx in tqdm(range(num_examples), desc="examples"):
        item = dataset[idx]
        rec = prompt_records[idx]

        image = item["image_options"][0].convert("RGB")
        image_name = clean_text(item.get("image_name", f"sample_{idx:04d}"))
        raw_question = clean_text(rec.get("question", ""))

        gold = normalize_rel(rec.get("answer", None))
        if gold is None:
            gold = relation_from_image_name(image_name)
        if gold is None:
            continue

        obj1, obj2 = parse_objects_from_question(raw_question)
        if not obj1 or not obj2:
            obj1, obj2 = infer_names_from_filename(image_name)

        # 1) text tower -> inverse soft mask
        union_gate = build_union_gate_from_text_tower(
            clip_model=clip_model,
            clip_processor=clip_processor,
            image_pil=image,
            obj1_name=obj1,
            obj2_name=obj2,
            device=DEVICE,
            use_dilate=USE_DILATE,
            dilate_kernel=DILATE_KERNEL,
        )

        # 2) 把 gate 塞给 llava，内部改视觉 token
        llava_model._current_soft_gate = union_gate

        # 3) 跑原始问题（保持仓库原生方式）
        try:
            out = llava_wrapper.run_single_prompt(
                image=image,
                prompt=raw_question,
                method="base",
                weight=None,
            )
            raw_output = extract_raw_text(out).strip()
        finally:
            llava_model._current_soft_gate = None

        pred = parse_prediction(raw_output)

        total += 1
        if pred == gold:
            correct += 1

        if shown < PRINT_FIRST_N:
            print("=" * 120)
            print(f"idx={idx}")
            print(f"image_name: {image_name}")
            print(f"obj1={obj1} | obj2={obj2}")
            print(f"gold={gold} | pred={pred}")
            print("[PROMPT]")
            print(raw_question)
            print("[RAW OUTPUT]")
            print(raw_output)
            shown += 1

    print("=" * 120)
    print(f"total = {total}")
    print(f"correct = {correct}")
    print(f"acc = {correct / total:.4f}" if total > 0 else "acc = N/A")


if __name__ == "__main__":
    main()
