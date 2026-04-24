import os
import re
import math
import random
from collections import defaultdict

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

DEVICE = "cuda"
CACHE_DIR = f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CLIP_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

SEED = 1
MAX_EXAMPLES = None
PRINT_FIRST_N = 10
MAX_NEW_TOKENS = 24

# local branch: soft background suppression
BG_RATIO = 0.10
BOOST = 1.00
USE_DILATE = False
DILATE_KERNEL = 3

LABELS = ["left", "right", "under", "on"]

# 概率级融合：
# fused_prob[c] = alpha[c] * global_prob[c] + (1 - alpha[c]) * local_prob[c]
ALPHA_BY_CLASS = {
    "left": 0.90,
    "right": 0.90,
    "under": 0.50,
    "on": 0.10,
}


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
    q = q.replace("<image>", " ").strip()

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


def prepare_repo_prompt(question):
    q = clean_text(question)
    if "<image>" in q and "USER:" in q and "ASSISTANT:" in q:
        return q
    return f"<image> USER: {q} Answer with left, right, on or under. ASSISTANT:"


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
    b, n = mask.shape
    x = mask.view(b, 1, gh, gw)
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return x.view(b, n)


def resize_gate_to_num_patches(gate, target_num_patches):
    src_h, src_w = infer_patch_grid(gate.shape[1])
    tgt_h, tgt_w = infer_patch_grid(target_num_patches)

    if src_h == tgt_h and src_w == tgt_w:
        return gate

    x = gate.view(gate.shape[0], 1, src_h, src_w)
    x = F.interpolate(x, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    return x.view(gate.shape[0], tgt_h * tgt_w)


def parse_prediction(text):
    text = clean_text(text).lower()

    phrase_patterns = [
        (r"\bto the left of\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bon top of\b", "on"),
        (r"\bon the\b", "on"),
    ]

    found = []
    for pat, label in phrase_patterns:
        for m in re.finditer(pat, text):
            found.append((m.start(), m.end(), label, m.group(0)))

    if found:
        found.sort(key=lambda x: x[0])
        return found[-1][2]

    single_patterns = [
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bon\b", "on"),
    ]
    found = []
    for pat, label in single_patterns:
        for m in re.finditer(pat, text):
            found.append((m.start(), m.end(), label, m.group(0)))

    if found:
        found.sort(key=lambda x: x[0])
        return found[-1][2]

    return None


def find_selected_relation_span(text):
    """
    和 parse_prediction 同口径：
    取最后一个有效关系表达
    返回:
      (start, end, label, matched_text) or None
    """
    raw = str(text)
    lower = raw.lower()

    phrase_patterns = [
        (r"\bto the left of\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bon top of\b", "on"),
        (r"\bon the\b", "on"),
    ]

    found = []
    for pat, label in phrase_patterns:
        for m in re.finditer(pat, lower):
            found.append((m.start(), m.end(), label, raw[m.start():m.end()]))

    if found:
        found.sort(key=lambda x: x[0])
        return found[-1]

    single_patterns = [
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bon\b", "on"),
    ]
    found = []
    for pat, label in single_patterns:
        for m in re.finditer(pat, lower):
            found.append((m.start(), m.end(), label, raw[m.start():m.end()]))

    if found:
        found.sort(key=lambda x: x[0])
        return found[-1]

    return None


def concat_text(a, b):
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    if a[-1].isspace() or b[0].isspace():
        return a + b
    return a + " " + b


def continuation_phrases():
    return {
        "left": "to the left of",
        "right": "to the right of",
        "under": "under",
        "on": "on",
    }


def softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def probs_dict_to_pred(prob_dict):
    return max(prob_dict.items(), key=lambda x: x[1])[0]


def print_stats(name, total, correct, per_gold_total, per_gold_correct):
    print("=" * 120)
    print(f"[{name}]")
    print(f"total = {total}")
    print(f"correct = {correct}")
    print(f"overall_acc = {correct / total:.4f}" if total > 0 else "overall_acc = N/A")
    for rel in ["left", "right", "under", "on"]:
        n = per_gold_total[rel]
        c = per_gold_correct[rel]
        acc = (c / n) if n > 0 else 0.0
        print(f"{rel}_acc = {acc:.4f} ({c}/{n})")


# =========================================================
# CLIP text tower -> inverse soft mask
# =========================================================
@torch.no_grad()
def get_clip_patch_features(clip_model, clip_processor, image_pil, device):
    image_inputs = clip_processor(images=image_pil, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=CLIP_DTYPE)

    vision_outputs = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
    last_hidden = vision_outputs.last_hidden_state  # [1, 1+N, H]

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
    patch_proj = get_clip_patch_features(clip_model, clip_processor, image_pil, device)
    text1 = get_clip_text_feature(clip_model, clip_processor, obj1_name, device)
    text2 = get_clip_text_feature(clip_model, clip_processor, obj2_name, device)

    sims1 = torch.matmul(patch_proj[0], text1[0]).unsqueeze(0)   # [1, N]
    sims2 = torch.matmul(patch_proj[0], text2[0]).unsqueeze(0)   # [1, N]

    gate1 = normalize_1d_torch(-sims1)
    gate2 = normalize_1d_torch(-sims2)

    union_gate = torch.maximum(gate1, gate2)

    gh, gw = infer_patch_grid(union_gate.shape[1])
    if use_dilate:
        union_gate = dilate_patch_mask(union_gate, gh, gw, kernel_size=dilate_kernel)

    union_gate = normalize_1d_torch(union_gate)
    return union_gate


# =========================================================
# patch repo llava projector
# =========================================================
def install_projector_patch(llava_wrapper, bg_ratio=0.2, boost=1.0):
    llava_model = llava_wrapper.model

    if not hasattr(llava_model, "multi_modal_projector"):
        raise RuntimeError(
            f"repo-loaded model has no .multi_modal_projector, type={type(llava_model)}"
        )

    original_projector_forward = llava_model.multi_modal_projector.forward

    def patched_projector_forward(image_features):
        feats = original_projector_forward(image_features)  # [B, N, D]

        gate = getattr(llava_model, "_current_soft_gate", None)
        if gate is not None:
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

            scale = bg_ratio + (1.0 - bg_ratio) * gate_local
            if boost != 1.0:
                scale = scale * (1.0 + (boost - 1.0) * gate_local)

            feats = feats * scale.unsqueeze(-1)

        return feats

    llava_model.multi_modal_projector.forward = patched_projector_forward
    return llava_model


# =========================================================
# generation + continuation scoring from selected relation prefix
# =========================================================
@torch.no_grad()
def run_free_generation(llava_wrapper, image, prompt, gate=None, max_new_tokens=24):
    llava_model = llava_wrapper.model
    processor = llava_wrapper.processor
    tokenizer = processor.tokenizer

    inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt")
    inputs = {
        k: (v.to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
        if v is not None
    }

    llava_model._current_soft_gate = gate
    try:
        generated_ids = llava_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )[0]
    finally:
        llava_model._current_soft_gate = None

    prompt_len = inputs["input_ids"].shape[1]
    gen_only = generated_ids[prompt_len:]
    raw_output = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    return raw_output


@torch.no_grad()
def score_text_continuation(llava_wrapper, image, prefix_text, continuation_text, gate=None):
    """
    对 continuation_text 做 teacher-forcing continuation 打分
    返回平均 log-prob（避免长度偏置）
    """
    llava_model = llava_wrapper.model
    processor = llava_wrapper.processor

    full_text = concat_text(prefix_text, continuation_text)

    prefix_inputs = processor(images=image, text=prefix_text, padding=True, return_tensors="pt")
    prefix_inputs = {
        k: (v.to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in prefix_inputs.items()
        if v is not None
    }
    prefix_len = prefix_inputs["input_ids"].shape[1]

    full_inputs = processor(images=image, text=full_text, padding=True, return_tensors="pt")
    full_inputs = {
        k: (v.to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in full_inputs.items()
        if v is not None
    }

    llava_model._current_soft_gate = gate
    try:
        outputs = llava_model(**full_inputs)
    finally:
        llava_model._current_soft_gate = None

    logits = outputs.logits                # [1, T, V]
    input_ids = full_inputs["input_ids"]   # [1, T]

    target_ids = input_ids[:, prefix_len:]            # continuation token ids
    pred_logits = logits[:, prefix_len - 1:-1, :]     # aligned next-token logits

    log_probs = F.log_softmax(pred_logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    # 用平均 log-prob，减少长度偏置
    return float(gathered.mean().item())


@torch.no_grad()
def run_generate_and_get_relation_probs(llava_wrapper, image, prompt, gate=None, max_new_tokens=24):
    """
    先自由生成整句，再找到最终选中的关系短语位置，
    从“关系短语前缀”出发，对四个关系短语做 continuation 打分。
    返回:
        raw_output
        relation_probs(dict) 或 None
        matched_label
        matched_text
    """
    raw_output = run_free_generation(
        llava_wrapper=llava_wrapper,
        image=image,
        prompt=prompt,
        gate=gate,
        max_new_tokens=max_new_tokens,
    )

    span = find_selected_relation_span(raw_output)
    if span is None:
        return raw_output, None, None, None

    start, end, matched_label, matched_text = span
    prefix_before_relation = raw_output[:start]

    prefix_text = prompt + prefix_before_relation

    rel_scores = {}
    cand_map = continuation_phrases()
    for lab in LABELS:
        rel_scores[lab] = score_text_continuation(
            llava_wrapper=llava_wrapper,
            image=image,
            prefix_text=prefix_text,
            continuation_text=cand_map[lab],
            gate=gate,
        )

    score_vec = np.array([rel_scores[lab] for lab in LABELS], dtype=np.float32)
    prob_vec = softmax_np(score_vec)

    relation_probs = {lab: float(prob_vec[i]) for i, lab in enumerate(LABELS)}
    return raw_output, relation_probs, matched_label, matched_text


def fuse_relation_probs(global_probs, local_probs, alpha_by_class):
    fused = {}
    for lab in LABELS:
        a = alpha_by_class[lab]
        fused[lab] = a * global_probs[lab] + (1.0 - a) * local_probs[lab]

    total = sum(fused.values()) + 1e-12
    for lab in fused:
        fused[lab] /= total
    return fused


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
    install_projector_patch(llava_wrapper, bg_ratio=BG_RATIO, boost=BOOST)

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

    total_g = 0
    correct_g = 0
    per_gold_total_g = defaultdict(int)
    per_gold_correct_g = defaultdict(int)

    total_l = 0
    correct_l = 0
    per_gold_total_l = defaultdict(int)
    per_gold_correct_l = defaultdict(int)

    total_f = 0
    correct_f = 0
    per_gold_total_f = defaultdict(int)
    per_gold_correct_f = defaultdict(int)

    shown = 0
    num_examples = len(dataset) if MAX_EXAMPLES is None else min(MAX_EXAMPLES, len(dataset))

    for idx in tqdm(range(num_examples), desc="examples"):
        item = dataset[idx]
        rec = prompt_records[idx]

        image = item["image_options"][0].convert("RGB")
        image_name = clean_text(item.get("image_name", f"sample_{idx:04d}"))
        raw_question = clean_text(rec.get("question", ""))
        prompt = prepare_repo_prompt(raw_question)

        gold = normalize_rel(rec.get("answer", None))
        if gold is None:
            gold = relation_from_image_name(image_name)
        if gold is None:
            continue

        obj1, obj2 = parse_objects_from_question(prompt)
        if not obj1 or not obj2:
            obj1, obj2 = infer_names_from_filename(image_name)

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

        global_raw, global_probs, global_match, global_text = run_generate_and_get_relation_probs(
            llava_wrapper=llava_wrapper,
            image=image,
            prompt=prompt,
            gate=None,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        local_raw, local_probs, local_match, local_text = run_generate_and_get_relation_probs(
            llava_wrapper=llava_wrapper,
            image=image,
            prompt=prompt,
            gate=union_gate,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        pred_g = parse_prediction(global_raw)
        pred_l = parse_prediction(local_raw)
        pred_f = pred_g
        fused_probs = None

        if global_probs is not None:
            pred_g = probs_dict_to_pred(global_probs)

        if local_probs is not None:
            pred_l = probs_dict_to_pred(local_probs)

        if global_probs is not None and local_probs is not None:
            fused_probs = fuse_relation_probs(global_probs, local_probs, ALPHA_BY_CLASS)
            pred_f = probs_dict_to_pred(fused_probs)
        else:
            if pred_g is not None:
                pred_f = pred_g
            elif pred_l is not None:
                pred_f = pred_l

        total_g += 1
        per_gold_total_g[gold] += 1
        if pred_g == gold:
            correct_g += 1
            per_gold_correct_g[gold] += 1

        total_l += 1
        per_gold_total_l[gold] += 1
        if pred_l == gold:
            correct_l += 1
            per_gold_correct_l[gold] += 1

        total_f += 1
        per_gold_total_f[gold] += 1
        if pred_f == gold:
            correct_f += 1
            per_gold_correct_f[gold] += 1

        if shown < PRINT_FIRST_N:
            print("=" * 120)
            print(f"idx={idx}")
            print(f"image_name: {image_name}")
            print(f"obj1={obj1} | obj2={obj2}")
            print(f"gold={gold}")

            print("[PROMPT]")
            print(prompt)

            print("[GLOBAL RAW OUTPUT]")
            print(global_raw)
            print(f"[GLOBAL MATCHED LABEL] {global_match} | [MATCHED TEXT] {global_text}")
            if global_probs is not None:
                print("[GLOBAL RELATION PROBS]")
                for lab in LABELS:
                    print(f"  {lab:>5}: {global_probs[lab]:.4f}")
            print(f"[GLOBAL PRED] {pred_g}")

            print("[LOCAL RAW OUTPUT]")
            print(local_raw)
            print(f"[LOCAL MATCHED LABEL] {local_match} | [MATCHED TEXT] {local_text}")
            if local_probs is not None:
                print("[LOCAL RELATION PROBS]")
                for lab in LABELS:
                    print(f"  {lab:>5}: {local_probs[lab]:.4f}")
            print(f"[LOCAL PRED] {pred_l}")

            if fused_probs is not None:
                print("[FUSED RELATION PROBS]")
                for lab in LABELS:
                    print(f"  {lab:>5}: {fused_probs[lab]:.4f}")
            print(f"[FUSED PRED] {pred_f}")

            shown += 1

    print_stats("GLOBAL_ONLY", total_g, correct_g, per_gold_total_g, per_gold_correct_g)
    print_stats("LOCAL_ONLY", total_l, correct_l, per_gold_total_l, per_gold_correct_l)
    print_stats("FUSED_PROB_STEP", total_f, correct_f, per_gold_total_f, per_gold_correct_f)


if __name__ == "__main__":
    main()
