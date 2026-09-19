#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find an intro figure pair using ONLY hidden-state -> final_norm -> lm_head.

Goal:
- GT relation fixed to `left` by default
- one sample final-correct
- one sample final-wrong->right
- mid-layer decision evidence similar
- late-layer decision evidence diverges

It produces:
1) prob-gap plot:
     p(mapped_GT) - p(mapped_opp)
2) mapped-GT raw logit plot:
     logit(mapped_GT)

Author: standalone script for AdaptVis
"""

import os
import re
import json
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# =========================================================
# Model registry
# =========================================================

MODEL_REGISTRY = {
    "qwen-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}


# =========================================================
# Basic utils
# =========================================================

def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_relation(x: str) -> str:
    x = x.strip().lower()
    mapping = {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
        "on": "above",
        "under": "below",
        "top": "above",
        "bottom": "below",
    }
    if x not in mapping:
        raise ValueError(f"Unknown relation: {x}")
    return mapping[x]


def relation_display_name(x: str) -> str:
    x = normalize_relation(x)
    if x == "above":
        return "on"
    if x == "below":
        return "under"
    return x


def opposite_relation(rel: str) -> str:
    rel = normalize_relation(rel)
    if rel == "left":
        return "right"
    if rel == "right":
        return "left"
    if rel == "above":
        return "below"
    if rel == "below":
        return "above"
    raise ValueError(rel)


def nested_getattr(obj, path: str):
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def get_final_norm_and_lm_head(model):
    norm_candidates = [
        "model.norm",
        "language_model.model.norm",
        "language_model.norm",
    ]
    lm_head_candidates = [
        "lm_head",
        "language_model.lm_head",
    ]

    final_norm = None
    lm_head = None

    for c in norm_candidates:
        try:
            final_norm = nested_getattr(model, c)
            break
        except Exception:
            pass
    for c in lm_head_candidates:
        try:
            lm_head = nested_getattr(model, c)
            break
        except Exception:
            pass

    if final_norm is None:
        raise RuntimeError("Could not find final norm module.")
    if lm_head is None:
        raise RuntimeError("Could not find lm_head module.")
    return final_norm, lm_head


def get_option_token_id(tokenizer, letter: str) -> int:
    candidates = [letter, " " + letter]
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) >= 1:
        return ids[-1]
    raise RuntimeError(f"Failed to get token id for option letter {letter}")


# =========================================================
# Dataset parsing
# =========================================================

def try_get(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def resolve_image_path(sample: dict, data_root: str) -> str:
    cand = try_get(sample, [
        "image_path", "image", "img_path", "file_name", "filename"
    ])
    if cand is None:
        raise ValueError(f"No image path field found in sample keys={list(sample.keys())}")

    if os.path.isabs(cand) and os.path.exists(cand):
        return cand

    p = os.path.join(data_root, cand)
    if os.path.exists(p):
        return p

    # common fallback
    p2 = os.path.join(data_root, "images", cand)
    if os.path.exists(p2):
        return p2

    raise FileNotFoundError(f"Cannot resolve image path: {cand}")


def extract_sid(sample: dict, fallback_idx: int) -> int:
    sid = try_get(sample, ["sid", "sample_id", "id"], fallback_idx)
    try:
        return int(sid)
    except Exception:
        return fallback_idx


def extract_question(sample: dict) -> str:
    q = try_get(sample, ["question", "prompt", "text"])
    if q is None:
        raise ValueError("No question/prompt/text found.")
    return q.strip()


def extract_answer_letter(sample: dict) -> Optional[str]:
    ans = try_get(sample, ["answer", "gt_answer", "label", "correct_option"])
    if ans is None:
        return None
    ans = str(ans).strip().upper()
    if ans in ["A", "B", "C", "D"]:
        return ans
    return None


def extract_option_relation_map(sample: dict) -> Dict[str, str]:
    """
    Expected return:
        {"A":"left","B":"right","C":"above","D":"below"}
    """
    # Case 1: already stored directly
    direct = try_get(sample, ["option_to_relation", "relation_map", "choice_to_relation"])
    if isinstance(direct, dict):
        out = {}
        for k, v in direct.items():
            kk = str(k).strip().upper()
            if kk in ["A", "B", "C", "D"]:
                out[kk] = normalize_relation(str(v))
        if len(out) == 4:
            return out

    # Case 2: fields like A/B/C/D
    out = {}
    for letter in ["A", "B", "C", "D"]:
        v = sample.get(letter, None)
        if v is not None:
            out[letter] = normalize_relation(str(v))
    if len(out) == 4:
        return out

    # Case 3: parse from question text like "A=left B=right C=above D=below"
    q = extract_question(sample)
    patt = r"([ABCD])\s*=\s*(left|right|above|below|on|under)"
    found = re.findall(patt, q, flags=re.IGNORECASE)
    if len(found) >= 4:
        out = {}
        for a, b in found:
            out[a.upper()] = normalize_relation(b)
        if len(out) == 4:
            return out

    raise ValueError(f"Cannot extract A/B/C/D relation map for sid={sample.get('sid', 'NA')}")


def invert_relation_map(opt2rel: Dict[str, str]) -> Dict[str, str]:
    rel2opt = {}
    for k, v in opt2rel.items():
        rel2opt[normalize_relation(v)] = k
    return rel2opt


def extract_gt_relation(sample: dict, opt2rel: Dict[str, str]) -> str:
    # first try explicit relation
    gt_rel = try_get(sample, ["gt_relation", "relation", "label_relation"])
    if gt_rel is not None:
        return normalize_relation(gt_rel)

    # else via answer letter
    ans = extract_answer_letter(sample)
    if ans is not None:
        if ans not in opt2rel:
            raise ValueError(f"Answer letter {ans} not in option map.")
        return normalize_relation(opt2rel[ans])

    raise ValueError("Cannot determine GT relation.")


def build_prompt(question: str) -> str:
    return (
        f"{question}\n"
        f"Answer with only one letter: A, B, C, or D."
    )


# =========================================================
# Model forward + layerwise scores
# =========================================================

@torch.no_grad()
def compute_layerwise_decision_scores(
    model,
    processor,
    image_path: str,
    question: str,
    rel2opt: Dict[str, str],
    gt_rel: str,
    opp_rel: str,
    device: str = "cuda",
):
    gt_rel = normalize_relation(gt_rel)
    opp_rel = normalize_relation(opp_rel)

    gt_opt = rel2opt[gt_rel]
    opp_opt = rel2opt[opp_rel]

    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(question)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    # hidden_states[0] is embedding output, hidden_states[1:] are layer outputs
    n_layers = len(hidden_states) - 1

    final_norm, lm_head = get_final_norm_and_lm_head(model)
    tokenizer = processor.tokenizer

    option_ids = {
        "A": get_option_token_id(tokenizer, "A"),
        "B": get_option_token_id(tokenizer, "B"),
        "C": get_option_token_id(tokenizer, "C"),
        "D": get_option_token_id(tokenizer, "D"),
    }

    last_idx = int(inputs["input_ids"].shape[1] - 1)

    prob_gap = []
    gt_prob = []
    opp_prob = []
    gt_logit = []
    opp_logit = []
    pred_letters = []

    for l in range(1, n_layers + 1):
        hs = hidden_states[l][0, last_idx, :]   # [hidden]
        hs = hs.unsqueeze(0).unsqueeze(0)       # [1,1,H]
        normed = final_norm(hs)
        logits = lm_head(normed)[0, 0]          # [vocab]

        abcd_logits = torch.tensor(
            [logits[option_ids["A"]],
             logits[option_ids["B"]],
             logits[option_ids["C"]],
             logits[option_ids["D"]]],
            device=logits.device,
            dtype=logits.dtype,
        )
        abcd_probs = torch.softmax(abcd_logits, dim=0)

        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        gt_idx = letter2idx[gt_opt]
        opp_idx = letter2idx[opp_opt]

        cur_gt_prob = abcd_probs[gt_idx].item()
        cur_opp_prob = abcd_probs[opp_idx].item()
        cur_gt_logit = abcd_logits[gt_idx].item()
        cur_opp_logit = abcd_logits[opp_idx].item()

        prob_gap.append(cur_gt_prob - cur_opp_prob)
        gt_prob.append(cur_gt_prob)
        opp_prob.append(cur_opp_prob)
        gt_logit.append(cur_gt_logit)
        opp_logit.append(cur_opp_logit)

        pred_idx = int(torch.argmax(abcd_logits).item())
        pred_letter = ["A", "B", "C", "D"][pred_idx]
        pred_letters.append(pred_letter)

    final_pred_letter = pred_letters[-1]

    return {
        "prob_gap": np.array(prob_gap, dtype=np.float32),   # p(gt)-p(opp)
        "gt_prob": np.array(gt_prob, dtype=np.float32),
        "opp_prob": np.array(opp_prob, dtype=np.float32),
        "gt_logit": np.array(gt_logit, dtype=np.float32),   # raw mapped-GT logit
        "opp_logit": np.array(opp_logit, dtype=np.float32),
        "pred_letters": pred_letters,
        "final_pred_letter": final_pred_letter,
        "gt_opt": gt_opt,
        "opp_opt": opp_opt,
        "layers": np.arange(1, n_layers + 1),
    }


# =========================================================
# Pair selection
# =========================================================

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mean_abs(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a)))


def select_best_pair(
    correct_items: List[dict],
    wrong_items: List[dict],
    mid_layers: List[int],
    late_layers: List[int],
    layer_numbers: np.ndarray,
    weight_mid: float = 2.0,
    weight_late: float = 1.0,
):
    layer_to_idx = {int(x): i for i, x in enumerate(layer_numbers)}
    mid_idx = [layer_to_idx[x] for x in mid_layers if x in layer_to_idx]
    late_idx = [layer_to_idx[x] for x in late_layers if x in layer_to_idx]

    assert len(mid_idx) > 0
    assert len(late_idx) > 0

    best = None
    best_score = -1e9

    for c in correct_items:
        for w in wrong_items:
            c_mid = c["scores"]["prob_gap"][mid_idx]
            w_mid = w["scores"]["prob_gap"][mid_idx]
            c_late = c["scores"]["prob_gap"][late_idx]
            w_late = w["scores"]["prob_gap"][late_idx]

            mid_rmse = rmse(c_mid, w_mid)
            late_sep = float(np.mean(c_late - w_late))

            # auxiliary: also encourage raw GT logit to diverge late
            c_late_logit = c["scores"]["gt_logit"][late_idx]
            w_late_logit = w["scores"]["gt_logit"][late_idx]
            late_logit_sep = float(np.mean(c_late_logit - w_late_logit))

            # stronger if both middle stay positive (same direction)
            same_mid_sign_bonus = 0.0
            if np.mean(c_mid) > 0 and np.mean(w_mid) > 0:
                same_mid_sign_bonus = 0.2

            score = (
                weight_late * late_sep
                + 0.15 * late_logit_sep
                + same_mid_sign_bonus
                - weight_mid * mid_rmse
            )

            item = {
                "correct": c,
                "wrong": w,
                "mid_rmse": mid_rmse,
                "late_sep": late_sep,
                "late_logit_sep": late_logit_sep,
                "score": score,
            }

            if score > best_score:
                best_score = score
                best = item

    return best


# =========================================================
# Plotting
# =========================================================

def plot_prob_gap(pair, save_path: str, stage_split: int):
    c = pair["correct"]
    w = pair["wrong"]

    x = c["scores"]["layers"]
    yc = c["scores"]["prob_gap"]
    yw = w["scores"]["prob_gap"]

    plt.figure(figsize=(11, 6))
    plt.plot(
        x, yc, marker="o", linewidth=3,
        label=f"Final correct (sid={c['sid']})"
    )
    plt.plot(
        x, yw, marker="o", linewidth=3, linestyle="--",
        label=f"Final wrong→{relation_display_name(w['final_pred_relation'])} (sid={w['sid']})"
    )
    plt.axhline(0.0, linestyle="--", linewidth=2, color="gray")
    plt.axvline(stage_split + 0.5, linestyle=":", linewidth=2.5, color="gray")

    y_top = max(float(np.max(yc)), float(np.max(yw)))
    plt.text(x[1], y_top * 0.92, "spatial/early decision evidence", fontsize=20)
    plt.text(stage_split + 1.8, y_top * 0.92, "late decision", fontsize=20)

    plt.title("Stage-specific GT preference", fontsize=28, pad=14)
    plt.xlabel("Decoder layer", fontsize=24)
    plt.ylabel("GT-vs-opposite preference", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_gt_logit(pair, save_path: str, stage_split: int, target_relation: str):
    c = pair["correct"]
    w = pair["wrong"]

    x = c["scores"]["layers"]
    yc = c["scores"]["gt_logit"]
    yw = w["scores"]["gt_logit"]

    plt.figure(figsize=(11, 6))
    plt.plot(
        x, yc, marker="o", linewidth=3,
        label=f"Final correct (sid={c['sid']})"
    )
    plt.plot(
        x, yw, marker="o", linewidth=3, linestyle="--",
        label=f"Final wrong→{relation_display_name(w['final_pred_relation'])} (sid={w['sid']})"
    )
    plt.axhline(0.0, linestyle="--", linewidth=2, color="gray")
    plt.axvline(stage_split + 0.5, linestyle=":", linewidth=2.5, color="gray")

    y_top = max(float(np.max(yc)), float(np.max(yw)))
    plt.text(x[1], y_top * 0.90, "early", fontsize=20)
    plt.text(stage_split + 1.8, y_top * 0.90, "late", fontsize=20)

    disp = relation_display_name(target_relation)
    plt.title(f"Mapped-{disp} logit across layers", fontsize=28, pad=14)
    plt.xlabel("Decoder layer", fontsize=24)
    plt.ylabel(f"Logit for mapped '{disp}' option", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--prompt-jsonl", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen-3b", choices=["qwen-3b"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 means all")
    parser.add_argument("--target-relation", type=str, default="left",
                        choices=["left", "right", "above", "below"])
    parser.add_argument("--wrong-target", type=str, default=None,
                        help="default is semantic opposite of target-relation")
    parser.add_argument("--mid-layers", type=str, default="18,19,20,21,22,23,24,25",
                        help="comma-separated layers")
    parser.add_argument("--late-layers", type=str, default="27,28,29,30,31,32",
                        help="comma-separated layers")
    parser.add_argument("--stage-split", type=int, default=26)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    target_relation = normalize_relation(args.target_relation)
    wrong_target = normalize_relation(args.wrong_target) if args.wrong_target else opposite_relation(target_relation)

    mid_layers = [int(x) for x in args.mid_layers.split(",") if x.strip()]
    late_layers = [int(x) for x in args.late_layers.split(",") if x.strip()]

    model_name = MODEL_REGISTRY[args.model]
    print(f"[LOAD] {args.model} -> {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    samples = load_jsonl(args.prompt_jsonl)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"[DATA] loaded {len(samples)} samples")

    kept = []
    for idx, sample in enumerate(samples):
        try:
            sid = extract_sid(sample, idx)
            image_path = resolve_image_path(sample, args.data_root)
            question = extract_question(sample)
            opt2rel = extract_option_relation_map(sample)
            rel2opt = invert_relation_map(opt2rel)
            gt_rel = extract_gt_relation(sample, opt2rel)

            if gt_rel != target_relation:
                continue
            if wrong_target not in rel2opt:
                continue

            scores = compute_layerwise_decision_scores(
                model=model,
                processor=processor,
                image_path=image_path,
                question=question,
                rel2opt=rel2opt,
                gt_rel=target_relation,
                opp_rel=wrong_target,
                device=args.device,
            )

            final_pred_letter = scores["final_pred_letter"]
            final_pred_relation = normalize_relation(opt2rel[final_pred_letter])

            item = {
                "sid": sid,
                "image_path": image_path,
                "question": question,
                "opt2rel": opt2rel,
                "rel2opt": rel2opt,
                "gt_rel": gt_rel,
                "scores": scores,
                "final_pred_letter": final_pred_letter,
                "final_pred_relation": final_pred_relation,
            }
            kept.append(item)

            if len(kept) % 20 == 0:
                print(f"[PROGRESS] kept={len(kept)}")

        except Exception as e:
            print(f"[WARN] skip idx={idx}: {e}")

    print(f"[FILTER] GT={target_relation}: {len(kept)} usable samples")

    correct_items = [x for x in kept if x["final_pred_relation"] == target_relation]
    wrong_items = [x for x in kept if x["final_pred_relation"] == wrong_target]

    print(f"[GROUP] final-correct={len(correct_items)} | final-wrong->{wrong_target}={len(wrong_items)}")

    if len(correct_items) == 0 or len(wrong_items) == 0:
        raise RuntimeError("Need at least one final-correct and one final-wrong->opposite sample.")

    pair = select_best_pair(
        correct_items=correct_items,
        wrong_items=wrong_items,
        mid_layers=mid_layers,
        late_layers=late_layers,
        layer_numbers=correct_items[0]["scores"]["layers"],
        weight_mid=2.0,
        weight_late=1.0,
    )

    c = pair["correct"]
    w = pair["wrong"]

    print("\n" + "=" * 80)
    print("BEST PAIR")
    print("=" * 80)
    print(f"target GT relation : {target_relation}")
    print(f"wrong target       : {wrong_target}")
    print(f"correct sid        : {c['sid']}")
    print(f"wrong sid          : {w['sid']}")
    print(f"mid RMSE           : {pair['mid_rmse']:.4f}")
    print(f"late separation    : {pair['late_sep']:.4f}")
    print(f"late logit sep     : {pair['late_logit_sep']:.4f}")
    print(f"score              : {pair['score']:.4f}")
    print(f"correct final pred : {c['final_pred_relation']} ({c['final_pred_letter']})")
    print(f"wrong final pred   : {w['final_pred_relation']} ({w['final_pred_letter']})")
    print("=" * 80)

    # Save metadata
    meta = {
        "target_relation": target_relation,
        "wrong_target": wrong_target,
        "correct_sid": c["sid"],
        "wrong_sid": w["sid"],
        "mid_rmse": pair["mid_rmse"],
        "late_sep": pair["late_sep"],
        "late_logit_sep": pair["late_logit_sep"],
        "score": pair["score"],
        "correct_question": c["question"],
        "wrong_question": w["question"],
        "correct_image_path": c["image_path"],
        "wrong_image_path": w["image_path"],
    }
    with open(os.path.join(args.output_dir, "best_pair_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Plot 1: probability-gap figure
    plot_prob_gap(
        pair,
        save_path=os.path.join(args.output_dir, "intro_pair_prob_gap.png"),
        stage_split=args.stage_split,
    )

    # Plot 2: mapped-GT raw-logit figure
    plot_gt_logit(
        pair,
        save_path=os.path.join(args.output_dir, "intro_pair_gt_logit.png"),
        stage_split=args.stage_split,
        target_relation=target_relation,
    )

    print(f"[DONE] saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
