#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import copy
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


REL_SET = ["left", "right", "above", "below"]


# =========================================================
# utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_layers(spec: str):
    """
    "10-20" -> [10,11,...,20]
    "14,16,19" -> [14,16,19]
    "10-12,15,18-20" -> [...]
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    return sorted(list(dict.fromkeys(out)))

def normalize_relation(x: str):
    if x is None:
        return None
    s = str(x).strip().lower()
    s = s.replace("under", "below")
    s = s.replace("beneath", "below")
    s = s.replace("on top of", "above")
    s = s.replace("over", "above")
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    if "above" in s:
        return "above"
    if "below" in s:
        return "below"
    return None

def parse_answer_from_generation(text: str):
    if text is None:
        return None
    s = text.strip().lower()
    # 优先找四个标准词
    for w in ["left", "right", "above", "below", "under", "beneath", "over"]:
        if re.search(rf"\b{re.escape(w)}\b", s):
            return normalize_relation(w)
    return None

def safe_mean(xs):
    xs = list(xs)
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(xs))

def ensure_rgb(path):
    img = Image.open(path).convert("RGB")
    return img

def find_subsequence(haystack, needle):
    """return all start positions where needle occurs in haystack"""
    out = []
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return out
    for i in range(n - m + 1):
        if haystack[i:i+m] == needle:
            out.append(i)
    return out

def pick_best_occurrence(full_ids, target_ids, prefer_after=None):
    """
    简单启发式：
    - 如果只有一个匹配，直接返回
    - 如果有多个匹配，优先选最靠近 prefer_after 且在其后的位置
    - 否则选最后一个
    """
    hits = find_subsequence(full_ids, target_ids)
    if not hits:
        return None
    if len(hits) == 1:
        s = hits[0]
        return list(range(s, s + len(target_ids)))
    if prefer_after is not None:
        cand = [h for h in hits if h >= prefer_after]
        if len(cand) > 0:
            s = cand[0]
            return list(range(s, s + len(target_ids)))
    s = hits[-1]
    return list(range(s, s + len(target_ids)))

def format_question(record):
    if "question" in record and isinstance(record["question"], str) and record["question"].strip():
        return record["question"].strip()
    subj = record["subject"]
    ref = record["reference"]
    return (
        f"What is the spatial relation of the {subj} to the {ref}? "
        f"Answer with exactly one word from: left, right, above, below."
    )


# =========================================================
# data
# =========================================================

def load_records(records_csv):
    df = pd.read_csv(records_csv)
    needed = ["sample_index", "image_path", "subject", "reference", "answer"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"records csv must contain column: {c}")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "sample_index": int(r["sample_index"]),
            "image_path": str(r["image_path"]),
            "subject": str(r["subject"]),
            "reference": str(r["reference"]),
            "answer": normalize_relation(r["answer"]),
            "question": str(r["question"]) if "question" in df.columns and pd.notna(r["question"]) else "",
        })
    rows = [r for r in rows if r["answer"] in REL_SET]
    return rows

def split_train_test(records, train_frac=0.3, seed=0):
    idxs = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_train = int(round(len(idxs) * train_frac))
    train_ids = set(idxs[:n_train])
    train_records = [records[i] for i in idxs[:n_train]]
    test_records = [records[i] for i in idxs[n_train:]]
    return train_records, test_records


# =========================================================
# model / processor
# =========================================================

def get_decoder_layers(model):
    candidates = [
        "model.layers",
        "language_model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
    ]
    for path in candidates:
        cur = model
        ok = True
        for name in path.split("."):
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok:
            return cur, path
    raise RuntimeError("Could not find decoder layers. Please inspect the model structure.")

def build_qwen_inputs(processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    return text, inputs

def locate_subject_reference_spans(tokenizer, full_input_ids, subject, reference):
    # 编码 subject / reference
    subj_ids = tokenizer.encode(subject, add_special_tokens=False)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    if len(subj_ids) == 0 or len(ref_ids) == 0:
        return None, None

    subj_span = pick_best_occurrence(full_input_ids, subj_ids, prefer_after=None)
    if subj_span is None:
        return None, None

    ref_span = pick_best_occurrence(full_input_ids, ref_ids, prefer_after=subj_span[-1] + 1)
    if ref_span is None:
        # 如果 reference 在 subject 前面也允许
        ref_span = pick_best_occurrence(full_input_ids, ref_ids, prefer_after=None)
    return subj_span, ref_span


# =========================================================
# capture states
# =========================================================

class LayerStateCatcher:
    def __init__(self, layers):
        self.layers = layers
        self.cache = {}

    def hook_fn(self, layer_idx):
        def _hook(module, inputs, outputs):
            hs = outputs[0] if isinstance(outputs, tuple) else outputs
            self.cache[layer_idx] = hs.detach().cpu()
            return outputs
        return _hook


# =========================================================
# direction basis learning
# =========================================================

def orthonormalize(vecs, eps=1e-8):
    basis = []
    for v in vecs:
        v = v.clone()
        for b in basis:
            v = v - torch.dot(v, b) * b
        n = torch.norm(v)
        if n > eps:
            basis.append(v / n)
    if len(basis) == 0:
        return None
    return torch.stack(basis, dim=1)  # [d, k]

def compute_pair_diff_from_hidden(hidden_states, subj_span, ref_span):
    """
    hidden_states: [1, T, D]
    使用 span mean
    """
    s = hidden_states[0, subj_span, :].mean(dim=0)
    r = hidden_states[0, ref_span, :].mean(dim=0)
    return s - r

def learn_direction_bases(model, processor, train_records, layers, device="cuda"):
    decoder_layers, layer_path = get_decoder_layers(model)
    print(f"[decoder] {layer_path}, selected={layers}")

    catcher = LayerStateCatcher(layers)
    handles = []
    for l in layers:
        handles.append(decoder_layers[l].register_forward_hook(catcher.hook_fn(l)))

    per_layer_by_rel = {l: defaultdict(list) for l in layers}

    for rec in tqdm(train_records, desc="learn direction basis"):
        image = ensure_rgb(rec["image_path"])
        question = format_question(rec)
        _, inputs = build_qwen_inputs(processor, image, question)
        full_ids = inputs["input_ids"][0].tolist()
        subj_span, ref_span = locate_subject_reference_spans(
            processor.tokenizer, full_ids, rec["subject"], rec["reference"]
        )
        if subj_span is None or ref_span is None:
            continue

        inputs = {k: v.to(device) for k, v in inputs.items()}

        catcher.cache = {}
        with torch.no_grad():
            _ = model(**inputs)

        label = rec["answer"]
        for l in layers:
            if l not in catcher.cache:
                continue
            diff = compute_pair_diff_from_hidden(catcher.cache[l], subj_span, ref_span)
            per_layer_by_rel[l][label].append(diff)

    for h in handles:
        h.remove()

    bases = {}
    for l in layers:
        rel_means = {}
        ok = True
        for rel in REL_SET:
            if len(per_layer_by_rel[l][rel]) == 0:
                ok = False
                break
            rel_means[rel] = torch.stack(per_layer_by_rel[l][rel], dim=0).mean(dim=0)

        if not ok:
            print(f"[warn] layer {l}: insufficient train samples for all 4 relations.")
            continue

        # d_RL / d_AB
        d_rl = rel_means["right"] - rel_means["left"]
        d_ab = rel_means["above"] - rel_means["below"]
        B = orthonormalize([d_rl, d_ab])
        if B is None or B.shape[1] == 0:
            print(f"[warn] layer {l}: degenerate basis.")
            continue

        bases[l] = {
            "basis": B,               # [D, k]
            "d_rl_norm": float(torch.norm(d_rl)),
            "d_ab_norm": float(torch.norm(d_ab)),
            "rel_means": rel_means,
        }
        print(f"[direction basis] L{l:02d}: rank={B.shape[1]}, ||R-L||={torch.norm(d_rl):.3f}, ||A-B||={torch.norm(d_ab):.3f}")

    return bases


# =========================================================
# intervention hook
# =========================================================

class TrueDirectionRemovalEditor:
    """
    在指定层对 subject/reference span 做真正的 direction removal:
        r = mean(subj) - mean(ref)
        proj = P_D(r)
        subj -= 0.5 * proj
        ref  += 0.5 * proj
    """

    def __init__(self, bases, layers_to_edit, subj_span, ref_span):
        self.bases = bases
        self.layers_to_edit = set(layers_to_edit)
        self.subj_span = subj_span
        self.ref_span = ref_span

    def make_hook(self, layer_idx):
        def _hook(module, inputs, outputs):
            if layer_idx not in self.layers_to_edit:
                return outputs
            if layer_idx not in self.bases:
                return outputs

            hs = outputs[0] if isinstance(outputs, tuple) else outputs
            # hs: [B, T, D],这里只处理 batch=1
            Bmat = self.bases[layer_idx]["basis"].to(hs.device, dtype=hs.dtype)  # [D, k]

            s_vec = hs[:, self.subj_span, :].mean(dim=1)   # [1, D]
            r_vec = hs[:, self.ref_span, :].mean(dim=1)    # [1, D]
            pair_diff = s_vec - r_vec                      # [1, D]

            coeff = pair_diff @ Bmat                       # [1, k]
            proj = coeff @ Bmat.T                          # [1, D]
            delta = 0.5 * proj

            hs = hs.clone()
            hs[:, self.subj_span, :] = hs[:, self.subj_span, :] - delta[:, None, :]
            hs[:, self.ref_span, :] = hs[:, self.ref_span, :] + delta[:, None, :]

            if isinstance(outputs, tuple):
                outputs = (hs,) + outputs[1:]
                return outputs
            return hs
        return _hook


# =========================================================
# generation
# =========================================================

def generate_one(model, processor, image_path, question, device="cuda", max_new_tokens=8):
    image = ensure_rgb(image_path)
    _, inputs = build_qwen_inputs(processor, image, question)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )

    gen_ids = out[0, input_len:]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred = parse_answer_from_generation(gen_text)
    return pred, gen_text, inputs["input_ids"][0].detach().cpu().tolist()

def generate_one_with_edit(
    model, processor, image_path, question, subject, reference,
    layers_to_edit, bases, device="cuda", max_new_tokens=8
):
    image = ensure_rgb(image_path)
    _, inputs = build_qwen_inputs(processor, image, question)
    full_ids = inputs["input_ids"][0].tolist()
    subj_span, ref_span = locate_subject_reference_spans(
        processor.tokenizer, full_ids, subject, reference
    )
    if subj_span is None or ref_span is None:
        return None, "", False

    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    decoder_layers, _ = get_decoder_layers(model)
    editor = TrueDirectionRemovalEditor(bases, layers_to_edit, subj_span, ref_span)
    handles = []
    for l in layers_to_edit:
        if l in bases:
            handles.append(decoder_layers[l].register_forward_hook(editor.make_hook(l)))

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )
    finally:
        for h in handles:
            h.remove()

    gen_ids = out[0, input_len:]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred = parse_answer_from_generation(gen_text)
    return pred, gen_text, True


# =========================================================
# evaluation
# =========================================================

def evaluate_baseline(model, processor, records, device="cuda", max_new_tokens=8):
    rows = []
    for rec in tqdm(records, desc="baseline generate"):
        q = format_question(rec)
        pred, text, _ = generate_one(
            model, processor, rec["image_path"], q,
            device=device, max_new_tokens=max_new_tokens
        )
        rows.append({
            "sample_index": rec["sample_index"],
            "gt": rec["answer"],
            "pred": pred,
            "correct": int(pred == rec["answer"]) if pred is not None else 0,
            "gen_text": text,
            "subject": rec["subject"],
            "reference": rec["reference"],
            "image_path": rec["image_path"],
            "question": q,
        })
    return rows

def evaluate_edit(model, processor, baseline_rows, layers_to_edit, bases, device="cuda", max_new_tokens=8):
    rows = []
    for rec in tqdm(baseline_rows, desc=f"edit layers={layers_to_edit}"):
        pred, text, ok = generate_one_with_edit(
            model=model,
            processor=processor,
            image_path=rec["image_path"],
            question=rec["question"],
            subject=rec["subject"],
            reference=rec["reference"],
            layers_to_edit=layers_to_edit,
            bases=bases,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        rows.append({
            "sample_index": rec["sample_index"],
            "gt": rec["gt"],
            "base_pred": rec["pred"],
            "edit_pred": pred,
            "base_correct": rec["correct"],
            "edit_correct": int(pred == rec["gt"]) if pred is not None else 0,
            "hook_ok": int(ok),
            "edit_text": text,
        })
    return rows

def summarize_edit(edit_rows):
    n = len(edit_rows)
    base_acc = safe_mean(r["base_correct"] for r in edit_rows)
    edit_acc = safe_mean(r["edit_correct"] for r in edit_rows)

    w2c = 0
    c2w = 0
    base_wrong = 0
    base_correct = 0

    for r in edit_rows:
        if r["base_correct"] == 0:
            base_wrong += 1
            if r["edit_correct"] == 1:
                w2c += 1
        else:
            base_correct += 1
            if r["edit_correct"] == 0:
                c2w += 1

    return {
        "N": n,
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "W2C": w2c,
        "W2C_rate_wrong": (w2c / base_wrong) if base_wrong > 0 else float("nan"),
        "C2W": c2w,
        "C2W_rate_correct": (c2w / base_correct) if base_correct > 0 else float("nan"),
        "net": w2c - c2w,
    }


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--records-csv", type=str, required=True,
                        help="must contain: sample_index,image_path,subject,reference,answer[,question]")
    parser.add_argument("--layers", type=str, default="10-20",
                        help="e.g. 10-20 or 14,16,19")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--max-test", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--scan-single", action="store_true",
                        help="evaluate each chosen layer independently")
    parser.add_argument("--run-multi", action="store_true",
                        help="evaluate one joint intervention across all chosen layers")
    parser.add_argument("--save-dir", type=str, default="output/true_direction_removal_v1")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    layers = parse_layers(args.layers)
    print(f"[layers] {layers}")

    # data
    records = load_records(args.records_csv)
    train_records, test_records = split_train_test(records, train_frac=args.train_frac, seed=args.seed)

    if args.max_train > 0:
        train_records = train_records[:args.max_train]
    if args.max_test > 0:
        test_records = test_records[:args.max_test]

    print(f"[data] total={len(records)} train={len(train_records)} test={len(test_records)}")

    # model
    print(f"[model] loading {args.model_path} on {args.device}")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=None,
    )
    model.to(args.device)
    model.eval()

    # learn direction bases
    bases = learn_direction_bases(
        model=model,
        processor=processor,
        train_records=train_records,
        layers=layers,
        device=args.device
    )
    usable_layers = [l for l in layers if l in bases]
    print(f"[usable layers] {usable_layers}")
    if len(usable_layers) == 0:
        raise RuntimeError("No usable direction bases were learned.")

    # baseline
    baseline_rows = evaluate_baseline(
        model=model,
        processor=processor,
        records=test_records,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    base_df = pd.DataFrame(baseline_rows)
    base_df.to_csv(os.path.join(args.save_dir, "baseline_rows.csv"), index=False)

    base_acc = safe_mean(base_df["correct"].tolist())
    print("\n" + "=" * 100)
    print("BASELINE")
    print("=" * 100)
    print(f"N={len(base_df)}  baseline_acc={base_acc:.4f}")

    summary_rows = []

    # single-layer scan
    if args.scan_single:
        print("\n" + "=" * 100)
        print("TRUE DIRECTION REMOVAL — SINGLE LAYER")
        print("=" * 100)

        print("layer | acc(base->edit gain) | W2C/wrong C2W/correct net")
        for l in usable_layers:
            edit_rows = evaluate_edit(
                model=model,
                processor=processor,
                baseline_rows=baseline_rows,
                layers_to_edit=[l],
                bases=bases,
                device=args.device,
                max_new_tokens=args.max_new_tokens
            )
            out_csv = os.path.join(args.save_dir, f"edit_rows_L{l:02d}.csv")
            pd.DataFrame(edit_rows).to_csv(out_csv, index=False)

            s = summarize_edit(edit_rows)
            summary_rows.append({
                "mode": "single",
                "layer_spec": f"L{l:02d}",
                **s
            })

            print(
                f"L{l:02d} | "
                f"{s['base_acc']:.4f}->{s['edit_acc']:.4f} {s['gain']:+.4f} | "
                f"{s['W2C']}/{s['W2C_rate_wrong']:.3f} "
                f"{s['C2W']}/{s['C2W_rate_correct']:.3f} "
                f"{s['net']:+d}"
            )

    # multi-layer
    if args.run_multi:
        print("\n" + "=" * 100)
        print("TRUE DIRECTION REMOVAL — MULTI LAYER")
        print("=" * 100)

        edit_rows = evaluate_edit(
            model=model,
            processor=processor,
            baseline_rows=baseline_rows,
            layers_to_edit=usable_layers,
            bases=bases,
            device=args.device,
            max_new_tokens=args.max_new_tokens
        )
        out_csv = os.path.join(args.save_dir, f"edit_rows_multi_{'_'.join(map(str, usable_layers))}.csv")
        pd.DataFrame(edit_rows).to_csv(out_csv, index=False)

        s = summarize_edit(edit_rows)
        summary_rows.append({
            "mode": "multi",
            "layer_spec": ",".join(map(str, usable_layers)),
            **s
        })

        print(
            f"multi[{usable_layers}] | "
            f"{s['base_acc']:.4f}->{s['edit_acc']:.4f} {s['gain']:+.4f} | "
            f"{s['W2C']}/{s['W2C_rate_wrong']:.3f} "
            f"{s['C2W']}/{s['C2W_rate_correct']:.3f} "
            f"{s['net']:+d}"
        )

    if len(summary_rows) > 0:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(os.path.join(args.save_dir, "summary.csv"), index=False)
        print("\nSaved:")
        print(f"  {os.path.join(args.save_dir, 'baseline_rows.csv')}")
        print(f"  {os.path.join(args.save_dir, 'summary.csv')}")
    else:
        print("\n[warn] neither --scan-single nor --run-multi was set.")


if __name__ == "__main__":
    main()
