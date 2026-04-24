import os
import re
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor
from transformers import LlavaForConditionalGeneration, AutoProcessor

from dataset_zoo import get_dataset


# =========================
# config
# =========================
DATASET = "Controlled_Images_A"
OPTION = "four"

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"

DEVICE = "cuda"
DTYPE = torch.float16

CACHE_DIR = f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

SEED = 1

# soft background suppression
BG_RATIO = 0.20      # gate=0 的背景保留比例
BOOST = 1.00         # 如需再强调前景，可改成 1.2 / 1.5
USE_DILATE = True
DILATE_KERNEL = 3

MAX_EXAMPLES = None  # None = 全部
PRINT_FIRST_N = 10

MAX_NEW_TOKENS = 16
DO_SAMPLE = False


# =========================
# utils
# =========================
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
    # 优先取第一个合法答案
    for cand in ["left", "right", "on", "under"]:
        if re.search(rf"\b{cand}\b", text):
            return cand
    return None


def load_prompt_records(dataset_name, option):
    prompt_path = os.path.join("prompts", f"{dataset_name}_with_answer_{option}_options.jsonl")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    records = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def infer_patch_grid(num_patches):
    side = int(round(math.sqrt(num_patches)))
    if side * side == num_patches:
        return side, side
    for h in range(int(math.sqrt(num_patches)), 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    return 1, num_patches


def normalize_1d_torch(x, eps=1e-6):
    # x: [B, N]
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
    # gate: [1, N]
    src_h, src_w = infer_patch_grid(gate.shape[1])
    tgt_h, tgt_w = infer_patch_grid(target_num_patches)

    if src_h == tgt_h and src_w == tgt_w:
        return gate

    x = gate.view(1, 1, src_h, src_w)
    x = F.interpolate(x, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    return x.view(1, tgt_h * tgt_w)


def build_prompt(question):
    question = clean_text(question)
    return f"<image>\nUSER: {question} Answer with left, right, on or under only.\nASSISTANT:"


# =========================
# clip text tower -> inverse soft mask
# =========================
@torch.no_grad()
def get_clip_patch_features(clip_model, clip_processor, image_pil, device):
    image_inputs = clip_processor(images=image_pil, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    vision_outputs = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
    last_hidden = vision_outputs.last_hidden_state  # [1, 1+N, H]

    # 和你之前脚本一致：post_layernorm
    if hasattr(clip_model.vision_model, "vision_model") and hasattr(clip_model.vision_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.vision_model.post_layernorm(last_hidden)
    elif hasattr(clip_model.vision_model, "post_layernorm"):
        last_hidden = clip_model.vision_model.post_layernorm(last_hidden)

    patch_tokens = last_hidden[:, 1:, :]  # [1, N, H]
    patch_proj = clip_model.visual_projection(patch_tokens)  # [1, N, D]
    patch_proj = patch_proj / patch_proj.norm(dim=-1, keepdim=True)

    return patch_proj  # [1, N, D]


@torch.no_grad()
def get_text_feature(clip_model, clip_processor, text_phrase, device):
    text_inputs = clip_processor(text=[text_phrase], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = clip_model.get_text_features(**text_inputs)  # [1, D]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  # [1, D]


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
    关键点：
    1) 先用 text tower 扫 patch 得到 raw similarity
    2) 你前面验证过目标物体对应 low raw score
    3) 所以这里 objectness = normalize(-raw_sims)
    """
    patch_proj = get_clip_patch_features(clip_model, clip_processor, image_pil, device)  # [1, N, D]
    text1 = get_text_feature(clip_model, clip_processor, obj1_name, device)               # [1, D]
    text2 = get_text_feature(clip_model, clip_processor, obj2_name, device)               # [1, D]

    sims1 = torch.matmul(patch_proj[0], text1[0]).unsqueeze(0)  # [1, N]
    sims2 = torch.matmul(patch_proj[0], text2[0]).unsqueeze(0)  # [1, N]

    # 目标 = 低值谷底，所以取反
    obj1_gate = normalize_1d_torch(-sims1)  # [1, N]
    obj2_gate = normalize_1d_torch(-sims2)  # [1, N]

    union_gate = torch.maximum(obj1_gate, obj2_gate)  # [1, N]

    gh, gw = infer_patch_grid(union_gate.shape[1])
    if use_dilate:
        union_gate = dilate_patch_mask(union_gate, gh, gw, kernel_size=dilate_kernel)

    union_gate = normalize_1d_torch(union_gate)
    return union_gate, sims1, sims2


# =========================
# monkey patch llava image features
# =========================
def install_soft_gate_patch(model, bg_ratio=0.2, boost=1.0):
    original_get_image_features = model.get_image_features

    def patched_get_image_features(pixel_values, vision_feature_layer=None, vision_feature_select_strategy=None):
        feats = original_get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )  # [B, N, D]

        gate = getattr(model, "_current_soft_gate", None)
        if gate is None:
            return feats

        if gate.ndim == 1:
            gate_local = gate.unsqueeze(0)
        else:
            gate_local = gate

        if gate_local.shape[0] != feats.shape[0]:
            if gate_local.shape[0] == 1 and feats.shape[0] > 1:
                gate_local = gate_local.expand(feats.shape[0], -1)
            else:
                raise ValueError(f"gate batch {gate_local.shape[0]} != feats batch {feats.shape[0]}")

        if gate_local.shape[1] != feats.shape[1]:
            gate_local = resize_gate_to_num_patches(gate_local, feats.shape[1])

        gate_local = gate_local.to(device=feats.device, dtype=feats.dtype).clamp(0, 1)

        scale = bg_ratio + (1.0 - bg_ratio) * gate_local  # [B, N]
        if boost != 1.0:
            scale = scale * (1.0 + (boost - 1.0) * gate_local)

        feats = feats * scale.unsqueeze(-1)
        return feats

    model.get_image_features = patched_get_image_features
    return model


# =========================
# main
# =========================
def main():
    seed_all(SEED)

    prompt_records = load_prompt_records(DATASET, OPTION)
    dataset = get_dataset(DATASET, image_preprocess=None, download=False)

    if len(prompt_records) != len(dataset):
        raise ValueError(f"Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).")

    print(f"Loading LLaVA: {LLAVA_MODEL_ID}")
    llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID, cache_dir=CACHE_DIR)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    ).to(DEVICE).eval()

    print(f"Loading CLIP text tower: {CLIP_MODEL_ID}")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=CACHE_DIR)
    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    ).to(DEVICE).eval()

    # 给 llava 打 patch：内部改视觉 token，不改 prompt
    install_soft_gate_patch(llava_model, bg_ratio=BG_RATIO, boost=BOOST)

    total = 0
    correct = 0
    shown = 0

    iterator = range(len(dataset))
    if MAX_EXAMPLES is not None:
        iterator = range(min(MAX_EXAMPLES, len(dataset)))

    for idx in tqdm(iterator, desc="examples"):
        item = dataset[idx]
        rec = prompt_records[idx]

        image_name = clean_text(item.get("image_name", f"sample_{idx:04d}"))
        image = item["image_options"][0].convert("RGB")
        question = clean_text(rec["question"])

        gold = relation_from_image_name(image_name)
        if gold is None:
            # 没法算 acc 就跳过
            continue

        obj1, obj2 = parse_objects_from_question(question)
        if not obj1 or not obj2:
            obj1, obj2 = infer_names_from_filename(image_name)

        prompt = build_prompt(question)

        # 1) 用 CLIP text tower 做 inverse soft mask
        union_gate, sims1, sims2 = build_union_gate_from_text_tower(
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
        llava_model._current_soft_gate = union_gate  # [1, N]

        # 3) 跑原始问题
        inputs = llava_processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = llava_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                use_cache=True,
            )

        # 只解码新生成部分
        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_output = llava_processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
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
            print(prompt)
            print("[RAW OUTPUT]")
            print(raw_output)
            shown += 1

        # 清掉当前 gate
        llava_model._current_soft_gate = None

    print("=" * 120)
    print(f"total = {total}")
    print(f"correct = {correct}")
    print(f"acc = {correct / total:.4f}" if total > 0 else "acc = N/A")


if __name__ == "__main__":
    main()
