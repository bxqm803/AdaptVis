#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick standalone test of direct relation-Direction removal.

No prior experiment script is imported.

TRAIN:
  residual[sample, layer, hidden] from vectors.npz
  center each layer, then learn four prototypes:
    mu_left, mu_right, mu_above, mu_below

TEST, GT relation g:
  r_saved = residual_i,l - train_center_l

Modes:
  gt_fixed:
      removed = mu_g
  gt_project:
      alpha = <r_saved, mu_g> / ||mu_g||^2
      removed = alpha * mu_g

Apply the removed vector to REAL-image subject/reference text states at the
selected decoder block OUTPUT while preserving pair mean:

  h_sub' = h_sub - 0.5 * removed
  h_ref' = h_ref + 0.5 * removed

Thus the actual object-pair difference loses exactly `removed`.

Primary metric: fresh actual model.generate().
Default evaluation cohort: fresh baseline-correct TEST samples.

Example:
CUDA_VISIBLE_DEVICES=0 python eval_gt_direction_prototype_removal_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --direction-key residual \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --annotation-json data/coco_qa_two_obj.json \
  --data-root data \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --layers 10-20 \
  --modes gt_project \
  --output-dir output/qwen7b_gt_project_remove_L10_20_v1 \
  --overwrite

To compare literal prototype subtraction:
  --modes gt_project,gt_fixed
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

RELS = ("left", "right", "above", "below")
RELSET = set(RELS)
EPS = 1e-10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--direction-key", default="residual")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--annotation-json", default="data/coco_qa_two_obj.json")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument("--layers", default="10-20")
    p.add_argument("--modes", default="gt_project", help="gt_project,gt_fixed")
    p.add_argument("--train-controls", default="correct", choices=["correct", "all"])
    p.add_argument("--eval-cohort", default="correct", choices=["correct", "all"])
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def norm_rel(x):
    s = str(x).strip().lower()
    if re.search(r"\bleft\b", s): return "left"
    if re.search(r"\bright\b", s): return "right"
    if re.search(r"\babove\b", s) or re.search(r"\bover\b", s): return "above"
    if re.search(r"\bbelow\b", s) or re.search(r"\bunder\b", s) or re.search(r"\bbeneath\b", s): return "below"
    return s


def parse_layers(spec, n_layers):
    out = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(piece))
    out = sorted(set(out))
    bad = [x for x in out if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"invalid layers={bad}; valid 0..{n_layers-1}")
    return out


def parse_modes(spec):
    modes = [x.strip() for x in spec.split(",") if x.strip()]
    allowed = {"gt_project", "gt_fixed"}
    bad = [x for x in modes if x not in allowed]
    if bad:
        raise ValueError(f"invalid modes={bad}; allowed={sorted(allowed)}")
    return modes


def torch_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# -----------------------------------------------------------------------------
# saved Direction bundle / four prototypes
# -----------------------------------------------------------------------------

def load_direction_bundle(direction_dir, key):
    root = Path(direction_dir)
    vp = root / "vectors.npz"
    gp = root / "sample_split_and_generation.csv"

    with np.load(vp, allow_pickle=True) as z:
        if key not in z.files:
            raise KeyError(f"{key!r} not found; keys={z.files}")
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray([norm_rel(x) for x in z["relation"]], dtype=object)
        arr = np.asarray(z[key], dtype=np.float32)

    n = len(sids)
    if arr.ndim != 3:
        raise ValueError(f"{key} must be 3D; got {arr.shape}")
    if arr.shape[0] == n:
        vectors = arr
    elif arr.shape[1] == n:
        vectors = np.transpose(arr, (1, 0, 2))
    else:
        raise ValueError(f"cannot align N={n} with {arr.shape}")

    split, group = {}, {}
    for row in read_csv(gp):
        sid = int(row["sample_index"])
        split[sid] = str(row.get("split", "")).strip().lower()
        group[sid] = str(row.get("generation_group", "")).strip().lower()

    return {
        "sids": sids,
        "labels": labels,
        "vectors": vectors,
        "gt": {int(s): str(labels[i]) for i, s in enumerate(sids.tolist())},
        "split": split,
        "group": group,
        "sid_to_index": {int(s): i for i, s in enumerate(sids.tolist())},
    }


def fit_prototypes(bundle, layers, controls):
    idx = []
    for i, sid in enumerate(bundle["sids"].tolist()):
        sid = int(sid)
        if bundle["split"].get(sid) != "train":
            continue
        if bundle["labels"][i] not in RELSET:
            continue
        if controls == "correct" and bundle["group"].get(sid) != "correct":
            continue
        idx.append(i)

    out, rows = {}, []
    for l in layers:
        X = bundle["vectors"][idx, l].astype(np.float64)
        Y = bundle["labels"][idx]
        center = X.mean(0)
        Xc = X - center[None, :]
        mus = {}
        for rel in RELS:
            mask = Y == rel
            if not np.any(mask):
                raise RuntimeError(f"L{l}: no TRAIN {rel}")
            mus[rel] = Xc[mask].mean(0).astype(np.float32)
        out[l] = {"center": center.astype(np.float32), "mu": mus}
        rows.append({
            "layer": l,
            "n_train": len(idx),
            "norm_left": float(np.linalg.norm(mus["left"])),
            "norm_right": float(np.linalg.norm(mus["right"])),
            "norm_above": float(np.linalg.norm(mus["above"])),
            "norm_below": float(np.linalg.norm(mus["below"])),
        })
    return out, rows


# -----------------------------------------------------------------------------
# COCO-two records
# -----------------------------------------------------------------------------

def parse_subject_reference(question):
    q = str(question)
    for pat in [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]:
        m = re.search(pat, q, re.I | re.S)
        if m:
            return (
                re.sub(r"\s+", " ", m.group(1)).strip(),
                re.sub(r"\s+", " ", m.group(2)).strip(),
            )
    return None, None


def load_records(prompt_jsonl, annotation_json, data_root, bundle):
    with open(annotation_json, "r", encoding="utf-8") as f:
        anns = json.load(f)
    prompts = []
    with open(prompt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    records = {}
    for row_no, row in enumerate(prompts):
        sid = int(row.get("id", row_no))
        if sid < 0 or sid >= len(anns):
            continue
        image_id = int(anns[sid][0])
        subject, reference = parse_subject_reference(row.get("question", ""))
        if subject is None:
            continue
        records[sid] = {
            "sid": sid,
            "subject": subject,
            "reference": reference,
            "image_path": str(Path(data_root) / "val2017" / f"{image_id:012d}.jpg"),
            "gt": bundle["gt"].get(sid, ""),
            "split": bundle["split"].get(sid, ""),
        }

    overlap = len(set(records) & set(map(int, bundle["sids"].tolist())))
    existing = sum(Path(r["image_path"]).exists() for r in records.values())
    print(f"[records] records={len(records)} overlap={overlap} existing_images={existing}/{len(records)}")
    return records


# -----------------------------------------------------------------------------
# model / prompt / generation
# -----------------------------------------------------------------------------

def attr_path(obj, path):
    cur = obj
    for x in path.split("."):
        cur = getattr(cur, x)
    return cur


def resolve_decoder_layers(model):
    for path in [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
    ]:
        try:
            ls = attr_path(model, path)
            if len(ls):
                return ls, path
        except Exception:
            pass
    raise RuntimeError("cannot find decoder layers")


def load_model(model_id, dtype, device, attn_impl):
    names = [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    cls = next((getattr(transformers, n) for n in names if hasattr(transformers, n)), None)
    if cls is None:
        raise RuntimeError("no multimodal model class")

    kw = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }
    if attn_impl != "none":
        kw["attn_implementation"] = attn_impl

    print(f"[model] {cls.__name__} | {model_id} | {device}")
    try:
        model = cls.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:
        model = cls.from_pretrained(model_id, torch_dtype=dtype, **kw)
    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def build_batch(processor, image, question, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    err = None
    for fn in [
        lambda: processor(text=[prompt], images=[image], padding=True, return_tensors="pt"),
        lambda: processor(text=prompt, images=image, return_tensors="pt"),
    ]:
        try:
            batch = fn()
            break
        except Exception as e:
            err = e
    else:
        raise RuntimeError(f"processor failed: {err}")

    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def parse_pred(text):
    s = str(text).lower()
    hits = []
    for rel, pat in [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
    ]:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))
    return sorted(hits)[0][1] if hits else None


def generate(model, processor, batch, max_new_tokens):
    n = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    text = processor.tokenizer.decode(out[0, n:], skip_special_tokens=True).strip()
    pred = parse_pred(text)
    del out
    return text, pred


# -----------------------------------------------------------------------------
# object token spans
# -----------------------------------------------------------------------------

def subseq(seq: Sequence[int], pat: Sequence[int]):
    hits = []
    m = len(pat)
    if not m:
        return hits
    for i in range(len(seq) - m + 1):
        if list(seq[i:i+m]) == list(pat):
            hits.append(i)
    return hits


def phrase_spans(tokenizer, full_ids, phrase):
    spans, seen = [], set()
    for text in [phrase, " " + phrase, phrase.strip(), " " + phrase.strip()]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for start in subseq(full_ids, ids):
            span = tuple(range(start, start + len(ids)))
            if span and span not in seen:
                seen.add(span)
                spans.append(list(span))
    return spans


def locate_spans(tokenizer, full_ids, subject, reference):
    ss = phrase_spans(tokenizer, full_ids, subject)
    rr = phrase_spans(tokenizer, full_ids, reference)
    if not ss or not rr:
        return None, None
    best = None
    for s in ss:
        for r in rr:
            if set(s) & set(r):
                continue
            score = (abs(float(np.mean(s)) - float(np.mean(r))), -min(s[0], r[0]))
            if best is None or score < best[0]:
                best = (score, s, r)
    return (best[1], best[2]) if best else (None, None)


# -----------------------------------------------------------------------------
# hook
# -----------------------------------------------------------------------------

def extract_hidden(output):
    if torch.is_tensor(output):
        return output, ("tensor", 0)
    if isinstance(output, tuple):
        for i, x in enumerate(output):
            if torch.is_tensor(x):
                return x, ("tuple", i)
    if isinstance(output, list):
        for i, x in enumerate(output):
            if torch.is_tensor(x):
                return x, ("list", i)
    raise RuntimeError(f"cannot extract hidden from {type(output)}")


def replace_hidden(output, desc, hidden):
    kind, idx = desc
    if kind == "tensor":
        return hidden
    vals = list(output)
    vals[idx] = hidden
    return tuple(vals) if kind == "tuple" else vals


class SubtractPairVector:
    def __init__(self, block, subject_span, reference_span, removed_vector):
        self.ss = list(subject_span)
        self.rr = list(reference_span)
        self.removed = np.asarray(removed_vector, dtype=np.float32)
        self.applied = False
        self.handle = block.register_forward_hook(self.hook)

    def hook(self, _module, _inputs, output):
        if self.applied:
            return output
        h, desc = extract_hidden(output)
        if h.ndim != 3 or max(self.ss + self.rr) >= h.shape[1]:
            return output

        d = torch.as_tensor(self.removed, device=h.device, dtype=h.dtype)
        if d.shape[-1] != h.shape[-1]:
            raise RuntimeError(f"vector dim={d.shape[-1]} != hidden dim={h.shape[-1]}")

        y = h.clone()
        half = 0.5 * d
        y[:, self.ss, :] = y[:, self.ss, :] - half[None, None, :]
        y[:, self.rr, :] = y[:, self.rr, :] + half[None, None, :]
        self.applied = True
        return replace_hidden(output, desc, y)

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# -----------------------------------------------------------------------------
# intervention vector
# -----------------------------------------------------------------------------

def removal_vector(bundle, protos, sid, layer, gt, mode):
    i = bundle["sid_to_index"][sid]
    residual = bundle["vectors"][i, layer].astype(np.float64)
    center = protos[layer]["center"].astype(np.float64)
    mu = protos[layer]["mu"][gt].astype(np.float64)
    r = residual - center

    if mode == "gt_fixed":
        alpha = 1.0
        removed = mu
    elif mode == "gt_project":
        denom = float(np.dot(mu, mu))
        alpha = float(np.dot(r, mu) / denom) if denom > EPS else 0.0
        removed = alpha * mu
    else:
        raise ValueError(mode)

    rn = float(np.linalg.norm(r))
    dn = float(np.linalg.norm(removed))

    return {
        "vector": removed.astype(np.float32),
        "alpha": alpha,
        "removed_norm": dn,
        "saved_direction_norm": rn,
        "removed_fraction": dn / max(rn, EPS),
        "prototype_norm": float(np.linalg.norm(mu)),
    }


# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------

def prepare_sample(processor, rec, prompt_template, device):
    image = Image.open(rec["image_path"]).convert("RGB")
    question = prompt_template.format(
        subject=rec["subject"], reference=rec["reference"]
    )
    batch = build_batch(processor, image, question, device)
    ids = batch["input_ids"][0].detach().cpu().tolist()
    ss, rr = locate_spans(
        processor.tokenizer, ids, rec["subject"], rec["reference"]
    )
    return image, batch, ss, rr


def baseline_eval(model, processor, records, test_sids, args, device, outdir):
    rows = []
    for sid in tqdm(test_sids, desc="fresh baseline"):
        image = None
        rec = records[sid]
        try:
            image, batch, ss, rr = prepare_sample(
                processor, rec, args.prompt_template, device
            )
            if ss is None or rr is None:
                rows.append({
                    "sid": sid, "gt": rec["gt"], "pred": "",
                    "correct": 0, "span_ok": 0,
                    "error": "object token span not found",
                })
                continue
            text, pred = generate(model, processor, batch, args.max_new_tokens)
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "pred": pred or "",
                "correct": int(pred == rec["gt"]),
                "span_ok": 1,
                "generation_text": text,
                "error": "",
            })
            del batch
        except Exception as e:
            rows.append({
                "sid": sid, "gt": rec["gt"], "pred": "",
                "correct": 0, "span_ok": 0,
                "error": f"{type(e).__name__}: {e}",
            })
            tqdm.write(f"[baseline ERROR sid={sid}] {type(e).__name__}: {e}")
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(outdir / "baseline.csv", rows)
    return rows


def run_layer_mode(
    model, processor, decoder, bundle, protos, records,
    baseline, layer, mode, args, device
):
    bmap = {int(r["sid"]): r for r in baseline}

    if args.eval_cohort == "correct":
        sids = [
            sid for sid, r in bmap.items()
            if int(r.get("span_ok", 0)) == 1 and int(r.get("correct", 0)) == 1
        ]
    else:
        sids = [sid for sid, r in bmap.items() if int(r.get("span_ok", 0)) == 1]

    rows = []
    for sid in tqdm(sorted(sids), desc=f"L{layer:02d} {mode}"):
        rec = records[sid]
        base = bmap[sid]
        image = None
        try:
            rv = removal_vector(bundle, protos, sid, layer, rec["gt"], mode)
            image, batch, ss, rr = prepare_sample(
                processor, rec, args.prompt_template, device
            )
            if ss is None or rr is None:
                continue

            with SubtractPairVector(decoder[layer], ss, rr, rv["vector"]):
                text, pred = generate(model, processor, batch, args.max_new_tokens)

            rows.append({
                "sid": sid,
                "layer": layer,
                "mode": mode,
                "gt": rec["gt"],
                "base_pred": base["pred"],
                "edit_pred": pred or "",
                "base_correct": int(base["correct"]),
                "edit_correct": int(pred == rec["gt"]),
                "C2W": int(int(base["correct"]) == 1 and pred != rec["gt"]),
                "W2C": int(int(base["correct"]) == 0 and pred == rec["gt"]),
                "changed": int((pred or "") != str(base["pred"])),
                "alpha": rv["alpha"],
                "removed_norm": rv["removed_norm"],
                "saved_direction_norm": rv["saved_direction_norm"],
                "removed_fraction_saved_direction": rv["removed_fraction"],
                "prototype_norm": rv["prototype_norm"],
                "generation_text": text,
            })
            del batch
        except Exception as e:
            tqdm.write(
                f"[edit ERROR sid={sid} L{layer} {mode}] {type(e).__name__}: {e}"
            )
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


def summarize(rows):
    if not rows:
        return {
            "N": 0, "edit_acc": float("nan"), "gain": float("nan"),
            "C2W": 0, "C2W_rate": float("nan"), "changed_rate": float("nan"),
            "mean_alpha": float("nan"), "mean_abs_alpha": float("nan"),
            "mean_removed_fraction": float("nan"),
        }

    base_acc = safe_mean(r["base_correct"] for r in rows)
    edit_acc = safe_mean(r["edit_correct"] for r in rows)
    n_base_correct = sum(int(r["base_correct"]) for r in rows)
    c2w = sum(int(r["C2W"]) for r in rows)

    return {
        "N": len(rows),
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "C2W": c2w,
        "C2W_rate": c2w / n_base_correct if n_base_correct else float("nan"),
        "changed_rate": safe_mean(r["changed"] for r in rows),
        "mean_alpha": safe_mean(r["alpha"] for r in rows),
        "mean_abs_alpha": safe_mean(abs(float(r["alpha"])) for r in rows),
        "mean_removed_fraction": safe_mean(
            r["removed_fraction_saved_direction"] for r in rows
        ),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    modes = parse_modes(args.modes)
    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_direction_bundle(args.direction_dir, args.direction_key)
    print(f"[direction] key={args.direction_key!r}, shape={bundle['vectors'].shape}")

    records = load_records(
        args.prompt_jsonl, args.annotation_json, args.data_root, bundle
    )

    model, processor = load_model(
        args.model_id, torch_dtype(args.dtype), args.device, args.attn_impl
    )
    decoder, decoder_path = resolve_decoder_layers(model)
    layers = parse_layers(args.layers, len(decoder))
    print(f"[decoder] {decoder_path}, layers={layers}")

    protos, proto_rows = fit_prototypes(bundle, layers, args.train_controls)
    write_csv(outdir / "prototype_summary.csv", proto_rows)

    print("[four centered relation prototypes]")
    for r in proto_rows:
        print(
            f"  L{r['layer']:02d}: Ntrain={r['n_train']} | "
            f"L={r['norm_left']:.3f} R={r['norm_right']:.3f} "
            f"A={r['norm_above']:.3f} B={r['norm_below']:.3f}"
        )

    test_sids = sorted([
        int(sid) for sid in bundle["sids"].tolist()
        if bundle["split"].get(int(sid)) == "test" and int(sid) in records
    ])

    if args.max_test_samples is not None and len(test_sids) > args.max_test_samples:
        rng = random.Random(args.seed)
        rng.shuffle(test_sids)
        test_sids = sorted(test_sids[:args.max_test_samples])

    device = torch.device(args.device)
    baseline = baseline_eval(
        model, processor, records, test_sids, args, device, outdir
    )

    valid = [r for r in baseline if int(r.get("span_ok", 0)) == 1]
    base_acc = safe_mean(r["correct"] for r in valid)
    n_correct = sum(int(r["correct"]) for r in valid)

    print("\n" + "=" * 120)
    print(
        f"FRESH BASELINE | valid N={len(valid)} | acc={base_acc:.4f} | "
        f"correct={n_correct} | wrong={len(valid)-n_correct}"
    )
    print("=" * 120)

    details, summaries = [], []

    for layer in layers:
        for mode in modes:
            rows = run_layer_mode(
                model, processor, decoder, bundle, protos, records,
                baseline, layer, mode, args, device
            )
            details.extend(rows)
            s = summarize(rows)
            summaries.append({"layer": layer, "mode": mode, **s})
            write_csv(outdir / "intervention_details.csv", details)
            write_csv(outdir / "summary.csv", summaries)

    print("\n" + "=" * 150)
    print("DIRECT GT DIRECTION REMOVAL — ACTUAL model.generate()")
    print("=" * 150)
    print(
        "layer mode       | editAcc gain C2W/rate changed | "
        "alpha mean|abs | removedFrac(savedDirection)"
    )

    for r in summaries:
        print(
            f"L{r['layer']:02d} {r['mode']:10s} | "
            f"{r['edit_acc']:.4f} {r['gain']:+.4f} "
            f"{r['C2W']}/{r['C2W_rate']:.3f} "
            f"{r['changed_rate']:.3f} | "
            f"{r['mean_alpha']:+.3f}|{r['mean_abs_alpha']:.3f} | "
            f"{r['mean_removed_fraction']:.3f}"
        )

    (outdir / "summary.json").write_text(
        json.dumps({
            "experiment": "direct GT Direction prototype removal",
            "direction_key": args.direction_key,
            "layers": layers,
            "modes": modes,
            "prototype_definition": "mean centered TRAIN residual vector per relation",
            "gt_project": (
                "subtract projection of current sample's saved centered residual "
                "Direction vector onto the GT relation prototype"
            ),
            "gt_fixed": "subtract one full GT relation prototype",
            "actual_edit_site": "decoder block OUTPUT subject/reference text tokens",
            "pair_mean_preserved": True,
            "actual_generate": True,
            "eval_cohort": args.eval_cohort,
            "baseline_valid_n": len(valid),
            "baseline_acc": base_acc,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for p in [
        outdir / "prototype_summary.csv",
        outdir / "baseline.csv",
        outdir / "intervention_details.csv",
        outdir / "summary.csv",
        outdir / "summary.json",
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
