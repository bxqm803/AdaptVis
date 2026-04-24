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
PRINT_FIRST_N = 10
MAX_EXAMPLES = None   # None = 全部；想先少跑一点就改成 20 / 50

# 只对前 N 个样本做 token-prob 分析，避免太慢
ANALYZE_FIRST_N = 10
MAX_NEW_TOKENS_ANALYSIS = 32
TOPK = 5

# soft-mask suppression params
BG_RATIO = 0.10
BOOST = 1.00
USE_DILATE = False
DILATE_KERNEL = 3

LABELS = ["left", "right", "under", "on"]


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


def parse_prediction(text):
    """
    修正版：
    - 先找更明确的关系短语
    - 如果有多个，取最后一个关系表达
      因为模型常写成：
      "on the floor, under the chair"
      真正关系通常在后面
    """
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
        found.sort(key=lambda x: (x[0], x[1]))
        return found[-1][1]

    single_patterns = [
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bunder\b", "under"),
        (r"\bon\b", "on"),
    ]
    found = []
    for pat, label in single_patterns:
        for m in re.finditer(pat, text):
            found.append((m.start(), m.end(), label))

    if found:
        found.sort(key=lambda x: (x[0], x[1]))
        return found[-1][1]

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
    """
    你前面已经验证过：
    raw similarity 的 low-score valley 对应目标物体
    所以这里用 normalize(-raw_sims)
    """
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
# patch repo llava: suppress background inside visual stream
# =========================================================
def install_soft_gate_patch(llava_wrapper, bg_ratio=0.2, boost=1.0):
    llava_model = llava_wrapper.model

    if not hasattr(llava_model, "multi_modal_projector"):
        raise RuntimeError(
            f"repo-loaded model has no .multi_modal_projector, type={type(llava_model)}"
        )

    original_projector_forward = llava_model.multi_modal_projector.forward

    def patched_projector_forward(image_features):
        feats = original_projector_forward(image_features)  # [B, N, D]

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

        scale = bg_ratio + (1.0 - bg_ratio) * gate_local
        if boost != 1.0:
            scale = scale * (1.0 + (boost - 1.0) * gate_local)

        feats = feats * scale.unsqueeze(-1)
        return feats

    llava_model.multi_modal_projector.forward = patched_projector_forward
    return llava_model


# =========================================================
# original generation path (keep predictions identical to original compare)
# =========================================================
def run_one_mode(llava_wrapper, llava_model, image, prompt, gate=None):
    llava_model._current_soft_gate = gate
    try:
        out = llava_wrapper.run_single_prompt(
            image=image,
            prompt=prompt,
            method="base",
            weight=None,
        )
        raw_output = extract_raw_text(out).strip()
    finally:
        llava_model._current_soft_gate = None

    pred = parse_prediction(raw_output)
    return raw_output, pred


# =========================================================
# token-step analysis only
# =========================================================
def get_relation_token_id_sets(tokenizer):
    """
    为每个关系词收集若干可能的单-token形式。
    如果某个形式 encode 后长度为1，就加入集合。
    这样可以同时兼容:
      left / Left / " left" / " Left"
    """
    out = {}
    for lab in LABELS:
        forms = [lab, lab.capitalize(), " " + lab, " " + lab.capitalize()]
        ids = set()
        for form in forms:
            pieces = tokenizer.encode(form, add_special_tokens=False)
            if len(pieces) == 1:
                ids.add(int(pieces[0]))

        # 如果一个都没收集到，退化到裸词的最后一个 token
        if len(ids) == 0:
            pieces = tokenizer.encode(lab, add_special_tokens=False)
            if len(pieces) == 0:
                raise ValueError(f"Cannot encode label: {lab}")
            ids.add(int(pieces[-1]))

        out[lab] = ids
    return out


def find_last_relation_token_step(generated_ids, relation_token_id_sets, target_label=None):
    """
    在真实生成 token 序列里，找最后一个 relation token 的 step。
    如果给了 target_label，就只在那个 label 的 token 集合里找。
    """
    best = None

    if target_label is not None:
        target_ids = relation_token_id_sets[target_label]
        for step_idx, tid in enumerate(generated_ids):
            tid = int(tid)
            if tid in target_ids:
                best = (step_idx, target_label, tid)
        return best

    id2label = {}
    for lab, idset in relation_token_id_sets.items():
        for tid in idset:
            id2label[int(tid)] = lab

    for step_idx, tid in enumerate(generated_ids):
        tid = int(tid)
        if tid in id2label:
            best = (step_idx, id2label[tid], tid)

    return best


@torch.no_grad()
def analyze_last_relation_step_topk(
    llava_wrapper,
    image,
    prompt,
    parsed_label,
    gate=None,
    max_new_tokens=32,
    topk=5,
):
    """
    单纯做分析，不改变原始预测定义。
    - 先重新 generate 一次（deterministic）
    - 在真实生成 token 里，找到最后一个 relation token step
    - 取该 step 的完整词表概率
    - 打印 top-k token
    """
    if parsed_label is None:
        return None

    llava_model = llava_wrapper.model
    processor = llava_wrapper.processor
    tokenizer = processor.tokenizer

    relation_token_id_sets = get_relation_token_id_sets(tokenizer)

    inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt")
    inputs = {
        k: (v.to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
        if v is not None
    }

    llava_model._current_soft_gate = gate
    try:
        gen_out = llava_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
    finally:
        llava_model._current_soft_gate = None

    full_seq = gen_out.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = full_seq[prompt_len:]
    analysis_raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    found = find_last_relation_token_step(
        generated_ids=generated_ids.tolist(),
        relation_token_id_sets=relation_token_id_sets,
        target_label=parsed_label,
    )

    if found is None:
        return {
            "analysis_raw_output": analysis_raw_output,
            "matched_step": None,
            "matched_label": None,
            "matched_token_id": None,
            "matched_token_text": None,
            "matched_prob": None,
            "topk": [],
        }

    step_idx, matched_label, matched_token_id = found

    if step_idx >= len(gen_out.scores):
        return {
            "analysis_raw_output": analysis_raw_output,
            "matched_step": step_idx,
            "matched_label": matched_label,
            "matched_token_id": matched_token_id,
            "matched_token_text": tokenizer.decode([matched_token_id], skip_special_tokens=False),
            "matched_prob": None,
            "topk": [],
        }

    step_logits = gen_out.scores[step_idx][0].float()
    step_probs = F.softmax(step_logits, dim=-1)

    matched_prob = float(step_probs[int(matched_token_id)].item())

    top_vals, top_ids = torch.topk(step_probs, k=min(topk, step_probs.shape[0]))
    topk_list = []
    for p, tid in zip(top_vals.tolist(), top_ids.tolist()):
        token_piece = tokenizer.convert_ids_to_tokens(int(tid))
        token_text = tokenizer.decode([int(tid)], skip_special_tokens=False)
        topk_list.append({
            "token_id": int(tid),
            "token_piece": str(token_piece),
            "token_text": str(token_text),
            "prob": float(p),
        })

    return {
        "analysis_raw_output": analysis_raw_output,
        "matched_step": int(step_idx),
        "matched_label": str(matched_label),
        "matched_token_id": int(matched_token_id),
        "matched_token_text": tokenizer.decode([int(matched_token_id)], skip_special_tokens=False),
        "matched_prob": matched_prob,
        "topk": topk_list,
    }


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

    # baseline stats
    total_base = 0
    correct_base = 0
    per_gold_total_base = defaultdict(int)
    per_gold_correct_base = defaultdict(int)

    # gated stats
    total_gate = 0
    correct_gate = 0
    per_gold_total_gate = defaultdict(int)
    per_gold_correct_gate = defaultdict(int)

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

        # -------------------------------------------------
        # baseline
        # -------------------------------------------------
        raw_output_base, pred_base = run_one_mode(
            llava_wrapper=llava_wrapper,
            llava_model=llava_model,
            image=image,
            prompt=raw_question,
            gate=None,
        )

        total_base += 1
        per_gold_total_base[gold] += 1
        if pred_base == gold:
            correct_base += 1
            per_gold_correct_base[gold] += 1

        # -------------------------------------------------
        # soft-mask suppression
        # -------------------------------------------------
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

        raw_output_gate, pred_gate = run_one_mode(
            llava_wrapper=llava_wrapper,
            llava_model=llava_model,
            image=image,
            prompt=raw_question,
            gate=union_gate,
        )

        total_gate += 1
        per_gold_total_gate[gold] += 1
        if pred_gate == gold:
            correct_gate += 1
            per_gold_correct_gate[gold] += 1

        if shown < PRINT_FIRST_N:
            print("=" * 120)
            print(f"idx={idx}")
            print(f"image_name: {image_name}")
            print(f"obj1={obj1} | obj2={obj2}")
            print(f"gold={gold}")

            print("[PROMPT]")
            print(raw_question)

            print("[BASELINE RAW OUTPUT]")
            print(raw_output_base)
            print(f"[BASELINE PRED] {pred_base}")

            print("[SOFT-MASK RAW OUTPUT]")
            print(raw_output_gate)
            print(f"[SOFT-MASK PRED] {pred_gate}")

            shown += 1

        # -------------------------------------------------
        # token-prob analysis only for first few samples
        # -------------------------------------------------
        if idx < ANALYZE_FIRST_N:
            base_analysis = analyze_last_relation_step_topk(
                llava_wrapper=llava_wrapper,
                image=image,
                prompt=raw_question,
                parsed_label=pred_base,
                gate=None,
                max_new_tokens=MAX_NEW_TOKENS_ANALYSIS,
                topk=TOPK,
            )

            gate_analysis = analyze_last_relation_step_topk(
                llava_wrapper=llava_wrapper,
                image=image,
                prompt=raw_question,
                parsed_label=pred_gate,
                gate=union_gate,
                max_new_tokens=MAX_NEW_TOKENS_ANALYSIS,
                topk=TOPK,
            )

            print("-" * 120)
            print(f"[ANALYSIS idx={idx}] image_name={image_name}")

            print("[BASELINE ANALYSIS RAW OUTPUT]")
            print(base_analysis["analysis_raw_output"])
            print(f"[BASELINE MATCHED LABEL] {base_analysis['matched_label']}")
            print(f"[BASELINE MATCHED STEP] {base_analysis['matched_step']}")
            print(f"[BASELINE MATCHED TOKEN ID] {base_analysis['matched_token_id']}")
            print(f"[BASELINE MATCHED TOKEN TEXT] {repr(base_analysis['matched_token_text'])}")
            print(f"[BASELINE MATCHED TOKEN PROB] {base_analysis['matched_prob']}")
            print("[BASELINE TOP-5 TOKENS]")
            for j, item_top in enumerate(base_analysis["topk"], 1):
                print(
                    f"  {j}. id={item_top['token_id']:<6d} "
                    f"piece={repr(item_top['token_piece'])} "
                    f"text={repr(item_top['token_text'])} "
                    f"prob={item_top['prob']:.6f}"
                )

            print("[SOFT-MASK ANALYSIS RAW OUTPUT]")
            print(gate_analysis["analysis_raw_output"])
            print(f"[SOFT-MASK MATCHED LABEL] {gate_analysis['matched_label']}")
            print(f"[SOFT-MASK MATCHED STEP] {gate_analysis['matched_step']}")
            print(f"[SOFT-MASK MATCHED TOKEN ID] {gate_analysis['matched_token_id']}")
            print(f"[SOFT-MASK MATCHED TOKEN TEXT] {repr(gate_analysis['matched_token_text'])}")
            print(f"[SOFT-MASK MATCHED TOKEN PROB] {gate_analysis['matched_prob']}")
            print("[SOFT-MASK TOP-5 TOKENS]")
            for j, item_top in enumerate(gate_analysis["topk"], 1):
                print(
                    f"  {j}. id={item_top['token_id']:<6d} "
                    f"piece={repr(item_top['token_piece'])} "
                    f"text={repr(item_top['token_text'])} "
                    f"prob={item_top['prob']:.6f}"
                )

    print_stats(
        "BASELINE",
        total_base,
        correct_base,
        per_gold_total_base,
        per_gold_correct_base,
    )

    print_stats(
        "SOFT_MASK_SUPPRESSION",
        total_gate,
        correct_gate,
        per_gold_total_gate,
        per_gold_correct_gate,
    )


if __name__ == "__main__":
    main()
