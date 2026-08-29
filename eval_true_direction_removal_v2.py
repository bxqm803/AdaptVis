#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_true_direction_removal_v2.py

Standalone test of whether the saved spatial Direction representation is
causally necessary at its native subject/reference text-token site.

No previous project script is imported.

For layer l:
    pair = mean(h_subject) - mean(h_reference)
    pair_D = P_D(pair)

Direction removal preserves pair mean:
    subject'   = subject   - 0.5 * pair_D
    reference' = reference + 0.5 * pair_D

Random control:
    remove a same-norm component in a random same-rank subspace orthogonal
    to Direction.

Primary metric: fresh actual model.generate().

The script directly reads the AdaptVis COCO-two files, so there is NO
YOUR_RECORDS.csv argument.
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
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


RELS = ("left", "right", "above", "below")
RELSET = set(RELS)
EPS = 1e-10


# ---------------------------------------------------------------------
# basic utils
# ---------------------------------------------------------------------

def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--direction-key", default="residual")

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--annotation-json",
        default="data/coco_qa_two_obj.json",
    )
    p.add_argument("--data-root", default="data")

    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument(
        "--layers",
        default="10-20",
        help="10-20, or 14,16,19, or 10-12,15,18-20",
    )

    p.add_argument(
        "--train-controls",
        default="correct",
        choices=["correct", "all"],
    )

    p.add_argument(
        "--eval-cohort",
        default="correct",
        choices=["correct", "all"],
        help="correct is recommended for a necessity/removal experiment.",
    )

    p.add_argument("--scan-single", action="store_true")
    p.add_argument("--run-multi", action="store_true")
    p.add_argument("--random-seeds", type=int, default=1)

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
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def std(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0


def norm_rel(x):
    s = str(x).strip().lower()
    if re.search(r"\bleft\b", s):
        return "left"
    if re.search(r"\bright\b", s):
        return "right"
    if re.search(r"\babove\b", s) or re.search(r"\bover\b", s):
        return "above"
    if (
        re.search(r"\bbelow\b", s)
        or re.search(r"\bunder\b", s)
        or re.search(r"\bbeneath\b", s)
    ):
        return "below"
    return s


def parse_layers(spec, n_layers):
    out = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        if "-" in x:
            a, b = map(int, x.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(x))
    out = sorted(set(out))
    bad = [l for l in out if l < 0 or l >= n_layers]
    if bad:
        raise ValueError(f"bad layers={bad}; valid 0..{n_layers-1}")
    return out


def torch_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


# ---------------------------------------------------------------------
# Direction bundle
# ---------------------------------------------------------------------

def load_direction_bundle(direction_dir, key):
    d = Path(direction_dir)
    vp = d / "vectors.npz"
    gp = d / "sample_split_and_generation.csv"

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
    for r in read_csv(gp):
        sid = int(r["sample_index"])
        split[sid] = str(r.get("split", "")).strip().lower()
        group[sid] = str(r.get("generation_group", "")).strip().lower()

    return {
        "sids": sids,
        "labels": labels,
        "vectors": vectors,
        "gt": {int(s): str(labels[i]) for i, s in enumerate(sids.tolist())},
        "split": split,
        "group": group,
    }


def orth_span(cols):
    M = np.stack([np.asarray(v, dtype=np.float64) for v in cols], axis=1)
    Q, R = np.linalg.qr(M, mode="reduced")
    keep = np.abs(np.diag(R)) > 1e-8
    Q = Q[:, keep]
    if Q.shape[1] == 0:
        raise RuntimeError("degenerate Direction basis")
    return Q.astype(np.float32)


def fit_bases(bundle, layers, controls):
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

    bases, rows = {}, []

    for l in layers:
        X = bundle["vectors"][idx, l].astype(np.float64)
        Y = bundle["labels"][idx]
        mus = {}
        for rel in RELS:
            mask = Y == rel
            if not np.any(mask):
                raise RuntimeError(f"L{l}: no TRAIN samples for {rel}")
            mus[rel] = X[mask].mean(0)

        d_lr = mus["right"] - mus["left"]
        d_ab = mus["above"] - mus["below"]
        B = orth_span([d_lr, d_ab])

        bases[l] = B
        rows.append({
            "layer": l,
            "rank": B.shape[1],
            "n_train": len(idx),
            "norm_R_minus_L": float(np.linalg.norm(d_lr)),
            "norm_A_minus_B": float(np.linalg.norm(d_ab)),
        })

    return bases, rows


# ---------------------------------------------------------------------
# COCO-two loader
# ---------------------------------------------------------------------

def parse_subject_reference(question):
    q = str(question)
    pats = [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]
    for pat in pats:
        m = re.search(pat, q, re.I | re.S)
        if m:
            s = re.sub(r"\s+", " ", m.group(1)).strip()
            r = re.sub(r"\s+", " ", m.group(2)).strip()
            return s, r
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

        ann = anns[sid]
        image_id = int(ann[0])

        subject, reference = parse_subject_reference(row.get("question", ""))

        if subject is None:
            # fallback from caption
            cap = str(ann[1]) if len(ann) > 1 else ""
            pats = [
                r"(.+?)\s+is\s+to\s+the\s+left\s+of\s+(.+)",
                r"(.+?)\s+is\s+to\s+the\s+right\s+of\s+(.+)",
                r"(.+?)\s+is\s+above\s+(.+)",
                r"(.+?)\s+is\s+below\s+(.+)",
            ]
            for pat in pats:
                m = re.search(pat, cap, re.I)
                if m:
                    subject = m.group(1).strip()
                    reference = m.group(2).strip().rstrip(".")
                    break

        if subject is None:
            continue

        image_path = Path(data_root) / "val2017" / f"{image_id:012d}.jpg"

        records[sid] = {
            "sid": sid,
            "subject": subject,
            "reference": reference,
            "image_path": str(image_path),
            "gt": bundle["gt"].get(sid, ""),
            "split": bundle["split"].get(sid, ""),
        }

    overlap = len(set(records) & set(map(int, bundle["sids"].tolist())))
    existing = sum(Path(r["image_path"]).exists() for r in records.values())

    print(
        f"[records] prompts={len(prompts)} annotations={len(anns)} "
        f"records={len(records)} overlap={overlap} "
        f"existing_images={existing}/{len(records)}"
    )

    if overlap == 0:
        raise RuntimeError("zero overlap with Direction sample ids")

    return records


# ---------------------------------------------------------------------
# model / processor
# ---------------------------------------------------------------------

def attr_path(obj, path):
    cur = obj
    for x in path.split("."):
        cur = getattr(cur, x)
    return cur


def decoder_layers(model):
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
        raise RuntimeError("no multimodal generation class found")

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
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    err = None
    for fn in [
        lambda: processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ),
    ]:
        try:
            batch = fn()
            break
        except Exception as e:
            err = e
    else:
        raise RuntimeError(f"processor failed: {err}")

    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    return prompt, batch


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
    n = batch["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    text = processor.tokenizer.decode(
        out[0, n:],
        skip_special_tokens=True,
    ).strip()
    pred = parse_pred(text)
    del out
    return text, pred


# ---------------------------------------------------------------------
# object-token span matching
# ---------------------------------------------------------------------

def subseq(seq, pat):
    hits = []
    m = len(pat)
    if not m:
        return hits
    for i in range(len(seq) - m + 1):
        if seq[i:i+m] == pat:
            hits.append(i)
    return hits


def phrase_spans(tokenizer, full_ids, phrase):
    spans, seen = [], set()

    for text in [phrase, " " + phrase, phrase.strip(), " " + phrase.strip()]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        for start in subseq(full_ids, ids):
            span = tuple(range(start, start + len(ids)))
            if span not in seen:
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
            dist = abs(float(np.mean(s)) - float(np.mean(r)))
            score = (dist, -min(s[0], r[0]))
            if best is None or score < best[0]:
                best = (score, s, r)

    return (best[1], best[2]) if best else (None, None)


# ---------------------------------------------------------------------
# true removal hook
# ---------------------------------------------------------------------

def output_hidden(output):
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
    raise RuntimeError(f"cannot find hidden in output type={type(output)}")


def replace_output_hidden(output, descriptor, hidden):
    kind, idx = descriptor
    if kind == "tensor":
        return hidden
    vals = list(output)
    vals[idx] = hidden
    return tuple(vals) if kind == "tuple" else vals


def random_orth_basis(dim, rank, direction_basis, seed):
    rng = np.random.default_rng(seed)
    D = np.asarray(direction_basis, dtype=np.float64)
    cols = []

    for _ in range(rank):
        for _trial in range(500):
            v = rng.standard_normal(dim)
            v = v - D @ (D.T @ v)
            if cols:
                C = np.stack(cols, axis=1)
                v = v - C @ (C.T @ v)
            n = np.linalg.norm(v)
            if n > 1e-8:
                cols.append(v / n)
                break
        else:
            raise RuntimeError("random basis construction failed")

    return np.stack(cols, axis=1).astype(np.float32)


class RemovalHooks:
    def __init__(
        self,
        layers,
        bases,
        selected_layers,
        subject_span,
        reference_span,
        mode,
        seed,
    ):
        self.handles = []
        self.stats = {}
        self.ss = subject_span
        self.rr = reference_span
        self.mode = mode
        self.random_bases = {}

        for l in selected_layers:
            B = bases[l]

            if mode == "random_matched":
                self.random_bases[l] = random_orth_basis(
                    B.shape[0],
                    B.shape[1],
                    B,
                    seed + l * 1000003,
                )

            self.handles.append(
                layers[l].register_forward_hook(self.make_hook(l, B))
            )

    def make_hook(self, l, basis_np):
        def hook(_module, _inputs, output):
            h, desc = output_hidden(output)

            if h.ndim != 3:
                return output

            # Prefill only. Cached decoding normally has seq_len=1.
            if max(self.ss + self.rr) >= h.shape[1]:
                return output

            B = torch.as_tensor(
                basis_np,
                device=h.device,
                dtype=h.dtype,
            )

            if B.shape[0] != h.shape[-1]:
                raise RuntimeError(
                    f"L{l}: Direction dim={B.shape[0]} != hidden dim={h.shape[-1]}"
                )

            s = h[:, self.ss, :].mean(1)
            r = h[:, self.rr, :].mean(1)
            pair = s - r

            dir_comp = (pair @ B) @ B.T
            dir_norm = torch.linalg.vector_norm(
                dir_comp.float(), dim=-1, keepdim=True
            )

            if self.mode == "direction":
                removed = dir_comp

            elif self.mode == "random_matched":
                RB = torch.as_tensor(
                    self.random_bases[l],
                    device=h.device,
                    dtype=h.dtype,
                )
                rnd = (pair @ RB) @ RB.T
                rnd_norm = torch.linalg.vector_norm(
                    rnd.float(), dim=-1, keepdim=True
                )
                scale = (
                    dir_norm / torch.clamp(rnd_norm, min=1e-8)
                ).to(h.dtype)
                removed = rnd * scale

            else:
                raise ValueError(self.mode)

            y = h.clone()
            half = 0.5 * removed

            y[:, self.ss, :] = y[:, self.ss, :] - half[:, None, :]
            y[:, self.rr, :] = y[:, self.rr, :] + half[:, None, :]

            pair_norm = float(
                torch.linalg.vector_norm(pair.float(), dim=-1).mean().detach().cpu()
            )
            rem_norm = float(
                torch.linalg.vector_norm(removed.float(), dim=-1).mean().detach().cpu()
            )
            dir_norm_scalar = float(dir_norm.mean().detach().cpu())

            self.stats[l] = {
                "pair_norm": pair_norm,
                "direction_norm": dir_norm_scalar,
                "removed_norm": rem_norm,
                "removed_fraction_pair": rem_norm / max(pair_norm, EPS),
            }

            return replace_output_hidden(output, desc, y)

        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------

def prepare_sample(processor, rec, prompt_template, device):
    image = Image.open(rec["image_path"]).convert("RGB")
    question = prompt_template.format(
        subject=rec["subject"],
        reference=rec["reference"],
    )
    _prompt, batch = build_batch(processor, image, question, device)
    ids = batch["input_ids"][0].detach().cpu().tolist()
    ss, rr = locate_spans(
        processor.tokenizer,
        ids,
        rec["subject"],
        rec["reference"],
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
                    "sid": sid,
                    "gt": rec["gt"],
                    "pred": "",
                    "correct": 0,
                    "span_ok": 0,
                    "error": "subject/reference token span not found",
                })
                continue

            text, pred = generate(
                model, processor, batch, args.max_new_tokens
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "pred": pred or "",
                "correct": int(pred == rec["gt"]),
                "span_ok": 1,
                "subject_span": " ".join(map(str, ss)),
                "reference_span": " ".join(map(str, rr)),
                "generation_text": text,
                "error": "",
            })

            del batch

        except Exception as e:
            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "pred": "",
                "correct": 0,
                "span_ok": 0,
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


def run_edit(
    model,
    processor,
    layers,
    bases,
    records,
    baseline_rows,
    selected_layers,
    mode,
    seed,
    args,
    device,
    desc,
):
    bmap = {int(r["sid"]): r for r in baseline_rows}

    if args.eval_cohort == "correct":
        sids = [
            sid for sid, r in bmap.items()
            if int(r.get("span_ok", 0)) == 1
            and int(r.get("correct", 0)) == 1
        ]
    else:
        sids = [
            sid for sid, r in bmap.items()
            if int(r.get("span_ok", 0)) == 1
        ]

    rows = []

    for sid in tqdm(sorted(sids), desc=desc):
        image = None
        rec = records[sid]
        base = bmap[sid]

        try:
            image, batch, ss, rr = prepare_sample(
                processor, rec, args.prompt_template, device
            )

            if ss is None or rr is None:
                continue

            with RemovalHooks(
                layers=layers,
                bases=bases,
                selected_layers=selected_layers,
                subject_span=ss,
                reference_span=rr,
                mode=mode,
                seed=seed,
            ) as hooks:

                text, pred = generate(
                    model, processor, batch, args.max_new_tokens
                )

                stats = dict(hooks.stats)

            fracs = [
                stats[l]["removed_fraction_pair"]
                for l in selected_layers
                if l in stats
            ]
            norms = [
                stats[l]["removed_norm"]
                for l in selected_layers
                if l in stats
            ]

            rows.append({
                "sid": sid,
                "mode": mode,
                "layers": ",".join(map(str, selected_layers)),
                "gt": rec["gt"],
                "base_pred": base["pred"],
                "edit_pred": pred or "",
                "base_correct": int(base["correct"]),
                "edit_correct": int(pred == rec["gt"]),
                "C2W": int(
                    int(base["correct"]) == 1 and pred != rec["gt"]
                ),
                "W2C": int(
                    int(base["correct"]) == 0 and pred == rec["gt"]
                ),
                "changed": int((pred or "") != str(base["pred"])),
                "mean_removed_fraction_pair": mean(fracs),
                "mean_removed_norm": mean(norms),
                "generation_text": text,
            })

            del batch

        except Exception as e:
            tqdm.write(
                f"[edit ERROR sid={sid} layers={selected_layers} mode={mode}] "
                f"{type(e).__name__}: {e}"
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
            "N": 0,
            "base_acc": float("nan"),
            "edit_acc": float("nan"),
            "gain": float("nan"),
            "C2W": 0,
            "C2W_rate": float("nan"),
            "W2C": 0,
            "W2C_rate": float("nan"),
            "changed_rate": float("nan"),
            "removed_fraction_pair": float("nan"),
        }

    base_correct = sum(int(r["base_correct"]) for r in rows)
    base_wrong = len(rows) - base_correct
    c2w = sum(int(r["C2W"]) for r in rows)
    w2c = sum(int(r["W2C"]) for r in rows)

    ba = mean(r["base_correct"] for r in rows)
    ea = mean(r["edit_correct"] for r in rows)

    return {
        "N": len(rows),
        "base_acc": ba,
        "edit_acc": ea,
        "gain": ea - ba,
        "C2W": c2w,
        "C2W_rate": c2w / base_correct if base_correct else float("nan"),
        "W2C": w2c,
        "W2C_rate": w2c / base_wrong if base_wrong else float("nan"),
        "changed_rate": mean(r["changed"] for r in rows),
        "removed_fraction_pair": mean(
            r["mean_removed_fraction_pair"] for r in rows
        ),
        "removed_norm": mean(
            r["mean_removed_norm"] for r in rows
        ),
    }


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    args = args_parser()

    if not args.scan_single and not args.run_multi:
        args.scan_single = True
        print("[note] defaulting to --scan-single")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_direction_bundle(
        args.direction_dir,
        args.direction_key,
    )

    print(
        f"[direction] key={args.direction_key!r} "
        f"shape={bundle['vectors'].shape}"
    )

    records = load_records(
        args.prompt_jsonl,
        args.annotation_json,
        args.data_root,
        bundle,
    )

    model, processor = load_model(
        args.model_id,
        torch_dtype(args.dtype),
        args.device,
        args.attn_impl,
    )

    layers, layer_path = decoder_layers(model)
    selected = parse_layers(args.layers, len(layers))

    print(f"[decoder] {layer_path}, selected={selected}")

    bases, basis_rows = fit_bases(
        bundle,
        selected,
        args.train_controls,
    )
    write_csv(outdir / "direction_basis_summary.csv", basis_rows)

    print("[direction bases]")
    for r in basis_rows:
        print(
            f"  L{r['layer']:02d}: rank={r['rank']} Ntrain={r['n_train']} "
            f"||R-L||={r['norm_R_minus_L']:.3f} "
            f"||A-B||={r['norm_A_minus_B']:.3f}"
        )

    test_sids = sorted([
        int(sid)
        for sid in bundle["sids"].tolist()
        if bundle["split"].get(int(sid)) == "test"
        and int(sid) in records
    ])

    if (
        args.max_test_samples is not None
        and len(test_sids) > args.max_test_samples
    ):
        rng = random.Random(args.seed)
        rng.shuffle(test_sids)
        test_sids = sorted(test_sids[:args.max_test_samples])

    print(f"[TEST] N={len(test_sids)}")

    device = torch.device(args.device)

    baseline = baseline_eval(
        model,
        processor,
        records,
        test_sids,
        args,
        device,
        outdir,
    )

    valid = [r for r in baseline if int(r.get("span_ok", 0)) == 1]
    base_acc = mean(r["correct"] for r in valid)
    n_correct = sum(int(r["correct"]) for r in valid)

    print(
        "\n" + "=" * 120
        + f"\nFRESH BASELINE | valid N={len(valid)} | "
          f"acc={base_acc:.4f} | correct={n_correct} | "
          f"wrong={len(valid)-n_correct}"
        + "\n" + "=" * 120
    )

    detail = []
    single_summary = []

    if args.scan_single:
        for l in selected:
            drows = run_edit(
                model, processor, layers, bases, records, baseline,
                [l], "direction", args.seed,
                args, device, f"L{l:02d} Direction removal"
            )
            for r in drows:
                r["experiment"] = "single"
                r["layer"] = l
            detail.extend(drows)
            ds = summarize(drows)

            random_summaries = []

            for rs in range(args.random_seeds):
                rrows = run_edit(
                    model, processor, layers, bases, records, baseline,
                    [l], "random_matched",
                    args.seed + 100003 * rs,
                    args, device, f"L{l:02d} random matched seed={rs}"
                )
                for r in rrows:
                    r["experiment"] = "single"
                    r["layer"] = l
                    r["random_seed"] = rs
                detail.extend(rrows)
                random_summaries.append(summarize(rrows))

            single_summary.append({
                "layer": l,
                "N_eval": ds["N"],
                "direction_edit_acc": ds["edit_acc"],
                "direction_gain": ds["gain"],
                "direction_C2W": ds["C2W"],
                "direction_C2W_rate": ds["C2W_rate"],
                "direction_changed_rate": ds["changed_rate"],
                "direction_removed_fraction_pair": ds["removed_fraction_pair"],
                "random_edit_acc_mean": mean(
                    x["edit_acc"] for x in random_summaries
                ),
                "random_edit_acc_std": std(
                    x["edit_acc"] for x in random_summaries
                ),
                "random_C2W_rate_mean": mean(
                    x["C2W_rate"] for x in random_summaries
                ),
                "random_C2W_rate_std": std(
                    x["C2W_rate"] for x in random_summaries
                ),
                "random_changed_rate_mean": mean(
                    x["changed_rate"] for x in random_summaries
                ),
            })

            write_csv(outdir / "single_layer_summary.csv", single_summary)
            write_csv(outdir / "intervention_details.csv", detail)

        print("\n" + "=" * 155)
        print("TRUE OBJECT-TOKEN DIRECTION REMOVAL — SINGLE LAYER")
        print("=" * 155)
        print(
            "layer | Direction: editAcc gain C2W/rate changed removeFrac | "
            "RandomMatched: editAcc C2Wrate changed"
        )
        for r in single_summary:
            print(
                f"L{r['layer']:02d} | "
                f"{r['direction_edit_acc']:.4f} "
                f"{r['direction_gain']:+.4f} "
                f"{r['direction_C2W']}/{r['direction_C2W_rate']:.3f} "
                f"{r['direction_changed_rate']:.3f} "
                f"{r['direction_removed_fraction_pair']:.3f} | "
                f"{r['random_edit_acc_mean']:.4f} "
                f"{r['random_C2W_rate_mean']:.3f} "
                f"{r['random_changed_rate_mean']:.3f}"
            )

    multi_summary = []

    if args.run_multi:
        drows = run_edit(
            model, processor, layers, bases, records, baseline,
            selected, "direction", args.seed,
            args, device, f"MULTI Direction removal {selected}"
        )
        for r in drows:
            r["experiment"] = "multi"
        detail.extend(drows)
        ds = summarize(drows)

        random_summaries = []
        for rs in range(args.random_seeds):
            rrows = run_edit(
                model, processor, layers, bases, records, baseline,
                selected, "random_matched",
                args.seed + 500009 + 100003 * rs,
                args, device, f"MULTI random matched seed={rs}"
            )
            for r in rrows:
                r["experiment"] = "multi"
                r["random_seed"] = rs
            detail.extend(rrows)
            random_summaries.append(summarize(rrows))

        multi_summary.append({
            "layers": ",".join(map(str, selected)),
            "N_eval": ds["N"],
            "direction_edit_acc": ds["edit_acc"],
            "direction_gain": ds["gain"],
            "direction_C2W": ds["C2W"],
            "direction_C2W_rate": ds["C2W_rate"],
            "direction_changed_rate": ds["changed_rate"],
            "direction_removed_fraction_pair": ds["removed_fraction_pair"],
            "random_edit_acc_mean": mean(
                x["edit_acc"] for x in random_summaries
            ),
            "random_C2W_rate_mean": mean(
                x["C2W_rate"] for x in random_summaries
            ),
            "random_changed_rate_mean": mean(
                x["changed_rate"] for x in random_summaries
            ),
        })

        write_csv(outdir / "multi_layer_summary.csv", multi_summary)
        write_csv(outdir / "intervention_details.csv", detail)

        r = multi_summary[0]
        print("\n" + "=" * 155)
        print("TRUE OBJECT-TOKEN DIRECTION REMOVAL — MULTI")
        print("=" * 155)
        print(
            f"layers={r['layers']} | "
            f"Direction editAcc={r['direction_edit_acc']:.4f} "
            f"gain={r['direction_gain']:+.4f} "
            f"C2W={r['direction_C2W']}/{r['direction_C2W_rate']:.3f} "
            f"changed={r['direction_changed_rate']:.3f} "
            f"removeFrac={r['direction_removed_fraction_pair']:.3f} | "
            f"Random editAcc={r['random_edit_acc_mean']:.4f} "
            f"C2Wrate={r['random_C2W_rate_mean']:.3f}"
        )

    write_csv(outdir / "intervention_details.csv", detail)

    (outdir / "summary.json").write_text(
        json.dumps({
            "experiment": "true object-token Direction removal",
            "direction_key": args.direction_key,
            "layers": selected,
            "site": "decoder block OUTPUT residual stream",
            "pair_mean_preserved": True,
            "actual_generate": True,
            "eval_cohort": args.eval_cohort,
            "baseline_valid_n": len(valid),
            "baseline_acc": base_acc,
            "baseline_correct": n_correct,
            "baseline_wrong": len(valid) - n_correct,
            "random_control": (
                "same-rank random subspace orthogonal to Direction; "
                "per-sample removal norm matched to Direction"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for p in [
        outdir / "direction_basis_summary.csv",
        outdir / "baseline.csv",
        outdir / "single_layer_summary.csv",
        outdir / "multi_layer_summary.csv",
        outdir / "intervention_details.csv",
        outdir / "summary.json",
    ]:
        if p.exists():
            print(" ", p)


if __name__ == "__main__":
    main()
