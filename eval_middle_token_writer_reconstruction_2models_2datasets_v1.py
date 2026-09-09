#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-model / cross-dataset validation of the oracle-selected middle-token
amplification mechanism.

Supported combinations
----------------------
Models:
    qwen-3b  = Qwen2.5-VL-3B-Instruct
    qwen-7b  = Qwen2.5-VL-7B-Instruct

Datasets:
    coco          = COCO_two standard four-option spatial QA
    controlled_A  = Controlled_Images_A standard four-option spatial QA

The experiment is the SAME mechanism used in the strong Qwen3B/COCO result:

1) CALIBRATE late relation writers on a relation-stratified TRAIN split

       q_i^T = h_real^T(last) - h_gray^T(last)

       s_r^T = E[q_i^T | relation=r] - common_mean

2) On each held-out sample, use the correct relation writer ONLY AS AN ORACLE
   CAUSAL TARGET and compute, for each non-last middle token:

       J = sum_T <h_real^T(last), normalize(s_r^T)>

       M_{S,p} =
           (h_real^{S,p} - h_gray^{S,p})^T
           dJ/dh_real^{S,p}

3) GLOBAL_UNIQUE selection:
   across all requested source layers, keep only the strongest layer for each
   token position, then select the global top-K positive M tokens.

4) Causal edit:
   each selected token gets ONLY ITS OWN naturally occurring Real-Gray delta

       h_edit^{S,p}
           = h_real^{S,p}
           + alpha * (h_real^{S,p} - h_gray^{S,p})

   No late writer is injected into middle tokens.

5) Compare actual greedy generation:
       baseline
       middle-selected
       direct late-writer reference/ceiling

IMPORTANT
---------
This remains an ORACLE MECHANISM experiment because GT relation chooses which
late relation writer s_r defines the selection objective. It is intended here
to test whether the same upstream reconstruction phenomenon generalizes across
model size and dataset.

Auto layer profiles
-------------------
qwen-3b:
    source bundles:
        22+24+26
        20+22+24
        18+20+22+24+26
    target writer layers:
        32,34,35

qwen-7b:
    source bundles:
        20+22+23
        18+20+22
        16+18+20+22+23
        14+16+18+20+22+23
    target writer layers:
        24,25,26,27

Recommended runs
----------------
# The two new combinations you asked for first:

CUDA_VISIBLE_DEVICES=0 python -u eval_middle_token_writer_reconstruction_2models_2datasets_v1.py \
  --model qwen-7b \
  --dataset coco \
  --eval-max-samples 80 \
  --ks 8,16,24,32 \
  --alphas 0.5,0.75,1.0 \
  --output-dir output/qwen7b_coco_middle_writer_reconstruction \
  --overwrite

CUDA_VISIBLE_DEVICES=0 python -u eval_middle_token_writer_reconstruction_2models_2datasets_v1.py \
  --model qwen-3b \
  --dataset controlled_A \
  --eval-max-samples 80 \
  --ks 8,16,24,32 \
  --alphas 0.5,0.75,1.0 \
  --output-dir output/qwen3b_controlledA_middle_writer_reconstruction \
  --overwrite

# Existing reference combination:
CUDA_VISIBLE_DEVICES=0 python -u eval_middle_token_writer_reconstruction_2models_2datasets_v1.py \
  --model qwen-3b --dataset coco \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_coco_middle_writer_reconstruction \
  --overwrite

# Fourth combination:
CUDA_VISIBLE_DEVICES=0 python -u eval_middle_token_writer_reconstruction_2models_2datasets_v1.py \
  --model qwen-7b --dataset controlled_A \
  --eval-max-samples 80 \
  --output-dir output/qwen7b_controlledA_middle_writer_reconstruction \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as coco
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj

try:
    import analyze_controlledA_centroid_step1_v2 as ctrla
except Exception:
    ctrla = None


CANON_REL = ("left", "right", "on", "under")

MODEL_PROFILES = {
    "qwen-3b": {
        "source_bundles": "22+24+26;20+22+24;18+20+22+24+26",
        "target_layers": "32,34,35",
    },
    "qwen-7b": {
        "source_bundles": "20+22+23;18+20+22;16+18+20+22+23;14+16+18+20+22+23",
        "target_layers": "24,25,26,27",
    },
}


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model", required=True, choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--dataset", required=True, choices=["coco", "controlled_A"])

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--coco-prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--controlled-prompt-jsonl",
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    p.add_argument("--controlled-module", default="")
    p.add_argument("--controlled-dataset-key", default="Controlled_Images_A")
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument(
        "--source-bundles",
        default="auto",
        help='Semicolon-separated bundles such as "22+24+26;20+22+24", or auto.',
    )
    p.add_argument(
        "--target-layers",
        default="auto",
        help='Comma-separated late writer layers, or auto.',
    )
    p.add_argument("--ks", default="8,16,24,32")
    p.add_argument("--alphas", default="0.5,0.75,1.0")
    p.add_argument("--direct-writer-alphas", default="1.0")
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])

    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional stratified cap BEFORE train/test split. 0 = use all.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=80,
        help="Relation-stratified cap on held-out evaluation set. 0 = all held-out.",
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return sorted({
        int(x.strip().lower().replace("l", ""))
        for x in str(text).split(",") if x.strip()
    })


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_bundles(text: str) -> List[Tuple[int, ...]]:
    bundles = []
    seen = set()
    for raw in str(text).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        b = tuple(sorted({
            int(x.strip().lower().replace("l", ""))
            for x in raw.split("+") if x.strip()
        }))
        if b and b not in seen:
            seen.add(b)
            bundles.append(b)
    if not bundles:
        raise ValueError("No source bundles")
    return bundles


def bundle_name(bundle: Sequence[int]) -> str:
    return "+".join(f"L{x}" for x in bundle)


# =====================================================================
# Generic utilities
# =====================================================================

def canon_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    x = str(value).strip().lower().replace("_", " ").replace("-", " ")
    x = " ".join(x.split())

    if x in {"left", "left of", "to the left", "to the left of"}:
        return "left"
    if x in {"right", "right of", "to the right", "to the right of"}:
        return "right"
    if x in {"on", "above", "over", "on top of", "top"}:
        return "on"
    if x in {"under", "below", "beneath", "bottom"}:
        return "under"

    # Allow the repo normalizers to have returned longer generation text.
    import re
    patterns = [
        (r"\b(left|leftward)\b", "left"),
        (r"\b(right|rightward)\b", "right"),
        (r"\b(under|below|beneath|bottom)\b", "under"),
        (r"\b(on top|above|over|top)\b", "on"),
        (r"\bon\b", "on"),
    ]
    hits = []
    for pat, rel in patterns:
        m = re.search(pat, x)
        if m:
            hits.append((m.start(), rel))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def make_gray_image(real_image: Image.Image, value: int) -> Image.Image:
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < 1e-12 else (v / n).astype(np.float32)


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def safe_mean(xs):
    vals = np.asarray([float(x) for x in xs], np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def stratified_cap(rows: Sequence[dict], max_n: int, seed: int) -> List[dict]:
    rows = list(rows)
    if not max_n or max_n <= 0 or max_n >= len(rows):
        return rows

    by_rel = defaultdict(list)
    for r in rows:
        by_rel[r["gt"]].append(r)

    rng = random.Random(seed)
    for rel in by_rel:
        rng.shuffle(by_rel[rel])

    # Round-robin keeps class balance as close as possible.
    out = []
    idx = {rel: 0 for rel in CANON_REL}
    while len(out) < max_n:
        progressed = False
        for rel in CANON_REL:
            bucket = by_rel.get(rel, [])
            j = idx[rel]
            if j < len(bucket) and len(out) < max_n:
                out.append(bucket[j])
                idx[rel] += 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(out)
    return out


def stratified_split(rows: Sequence[dict], train_ratio: float, seed: int):
    by_rel = defaultdict(list)
    for r in rows:
        by_rel[r["gt"]].append(r)

    train, test = [], []
    for rel in CANON_REL:
        bucket = list(by_rel.get(rel, []))
        rng = random.Random(seed * 1009 + CANON_REL.index(rel) * 97)
        rng.shuffle(bucket)
        if len(bucket) <= 1:
            n_train = len(bucket)
        else:
            n_train = int(round(len(bucket) * train_ratio))
            n_train = max(1, min(len(bucket) - 1, n_train))
        train.extend(bucket[:n_train])
        test.extend(bucket[n_train:])

    random.Random(seed + 111).shuffle(train)
    random.Random(seed + 222).shuffle(test)
    return train, test


# =====================================================================
# Dataset adapters
# =====================================================================

class DatasetAdapter:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.backend = None
        self.extractor = None
        self.prompts = None
        self.records = None
        self.rec_by_sid = None
        self.specs = None

    def load(self):
        if self.dataset == "coco":
            self._load_coco()
        elif self.dataset == "controlled_A":
            self._load_controlled()
        else:
            raise ValueError(self.dataset)

    def _load_coco(self):
        two = coco.import_two_object_module()
        prompt_path = Path(self.args.coco_prompt_jsonl)

        prompts = coco.load_standard_prompts(prompt_path)
        records, audit = two.load_records(
            "coco_two",
            Path(self.args.data_root),
            None,
        )

        self.backend = coco
        self.extractor = two
        self.prompts = prompts
        self.records = records
        self.rec_by_sid = {int(r.sid): r for r in records}
        self.specs = coco.merged_model_specs(two)
        self.audit = audit
        self.prompt_path = prompt_path

    def _load_controlled(self):
        if ctrla is None:
            raise ImportError(
                "Could not import analyze_controlledA_centroid_step1_v2.py. "
                "Run this script from the AdaptVis repo root."
            )

        module = ctrla.import_controlled_module(self.args.controlled_module)
        prompt_path = Path(self.args.controlled_prompt_jsonl)

        records, audit = module.load_records(
            prompt_path,
            dataset_key=self.args.controlled_dataset_key,
            keep_relations=["left", "right", "on", "under"],
            download=self.args.download,
            max_samples=None,
            num_workers=self.args.num_workers,
        )
        prompts = ctrla.load_standard_prompts(prompt_path)

        self.backend = ctrla
        self.extractor = module
        self.prompts = prompts
        self.records = records
        self.rec_by_sid = {int(r.sid): r for r in records}
        self.specs = ctrla.merged_model_specs(module)
        self.audit = audit
        self.prompt_path = prompt_path

    def meta_rows(self):
        out = []
        for rec in self.records:
            sid = int(rec.sid)
            if sid not in self.prompts:
                continue
            p = self.prompts[sid]

            if self.dataset == "coco":
                raw_gt = traj.normalize_relation(coco, p["answer_raw"])
            else:
                raw_gt = ctrla.normalize_relation(p["answer_raw"])

            gt = canon_relation(raw_gt)
            if gt not in CANON_REL:
                continue

            out.append({
                "sid": sid,
                "gt": gt,
                "subject": str(p["subject"]),
                "reference": str(p["reference"]),
                "question_text": str(p["question_text"]),
            })
        return out

    def record_image(self, sid):
        return self.backend.record_image(self.rec_by_sid[int(sid)])

    def make_batch(self, processor, image, question_text, device):
        return self.backend.make_question_batch(
            processor=processor,
            image=image,
            question_text=question_text,
            device=device,
        )

    def configure_processor(self, model, processor):
        return self.backend.configure_processor(model, processor)

    def resolve_decoder_layers(self, model):
        return self.backend.resolve_decoder_layers(model)

    def locate_object_spans(self, tokenizer, ids, subject, reference):
        return self.backend.locate_object_spans(
            tokenizer, ids, subject, reference
        )

    def generate_text(self, model, processor, batch, max_new_tokens):
        return self.backend.generate_text(
            model, processor, batch, max_new_tokens=max_new_tokens
        )

    def normalize_generation(self, text):
        if self.dataset == "coco":
            return canon_relation(traj.normalize_relation(coco, text))
        return canon_relation(ctrla.normalize_relation(text))


# =====================================================================
# Hidden-state capture and hooks
# =====================================================================

class Capture:
    def __init__(self, decoder_layers, layers, cut_layer=None, cpu=False):
        self.states = {}
        self.handles = []
        self.cut_layer = cut_layer
        self.cpu = cpu

        for L in layers:
            if cut_layer is not None and L == cut_layer:
                h = decoder_layers[L].register_forward_hook(self._cut(L))
            else:
                h = decoder_layers[L].register_forward_hook(self._keep(L))
            self.handles.append(h)

    def _cut(self, L):
        def hook(_m, _inp, out):
            x = traj.first_tensor(out)
            y = x.detach().clone().requires_grad_(True)
            self.states[L] = y
            return traj.replace_first_tensor(out, y)
        return hook

    def _keep(self, L):
        def hook(_m, _inp, out):
            x = traj.first_tensor(out)
            self.states[L] = (
                x.detach().float().cpu() if self.cpu else x
            )
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_cpu(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers, cpu=True)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing captured layers: {missing}")
        return {
            L: cap.states[L].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


def forward_graph(model, decoder_layers, batch, layers, cut):
    cap = Capture(
        decoder_layers,
        layers,
        cut_layer=cut,
        cpu=False,
    )
    kw = dict(batch)
    kw["use_cache"] = False
    _ = model(**kw)
    missing = [L for L in layers if L not in cap.states]
    if missing:
        cap.close()
        raise RuntimeError(f"Missing graph states: {missing}")
    return cap


class TokenDeltaEditor:
    def __init__(self, decoder_layers, specs, alpha, prompt_len):
        self.handles = []
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)
        self.applied = defaultdict(int)

        for L, entries in specs.items():
            if entries:
                self.handles.append(
                    decoder_layers[L].register_forward_hook(
                        self._hook(L, entries)
                    )
                )

    def _hook(self, L, entries):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)

            # Edit only the prefill pass; cached decode steps have length 1.
            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()
            for pos, delta_np in entries:
                pos = int(pos)
                if 0 <= pos < y.shape[1]:
                    d = torch.as_tensor(
                        delta_np,
                        device=y.device,
                        dtype=torch.float32,
                    )
                    y[:, pos, :] = y[:, pos, :] + self.alpha * d
                    self.applied[L] += 1

            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


class DirectWriterEditor:
    def __init__(
        self,
        decoder_layers,
        targets,
        writers_for_relation,
        alpha,
        prompt_len,
    ):
        self.handles = []
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)
        self.applied = defaultdict(int)

        for T in targets:
            vec = np.asarray(writers_for_relation[T], np.float32)
            self.handles.append(
                decoder_layers[T].register_forward_hook(
                    self._hook(T, vec)
                )
            )

    def _hook(self, T, vec_np):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()
            v = torch.as_tensor(
                vec_np,
                device=y.device,
                dtype=torch.float32,
            )
            y[:, -1, :] = y[:, -1, :] + self.alpha * v
            self.applied[T] += 1
            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


class TargetRecorder:
    def __init__(self, decoder_layers, targets, prompt_len):
        self.states = {}
        self.handles = []
        self.prompt_len = int(prompt_len)

        for T in targets:
            self.handles.append(
                decoder_layers[T].register_forward_hook(self._hook(T))
            )

    def _hook(self, T):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) == self.prompt_len and T not in self.states:
                self.states[T] = (
                    h[0, -1, :]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            return out
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def run_generation(
    adapter,
    model,
    processor,
    decoder_layers,
    batch,
    targets,
    writers_r,
    max_new_tokens,
    token_specs=None,
    token_alpha=None,
    direct_alpha=None,
):
    prompt_len = int(batch["input_ids"].shape[1])
    editor = direct = recorder = None

    try:
        # Registration order matters: recorder must see edited states.
        if token_specs is not None:
            editor = TokenDeltaEditor(
                decoder_layers,
                token_specs,
                token_alpha,
                prompt_len,
            )
        if direct_alpha is not None:
            direct = DirectWriterEditor(
                decoder_layers,
                targets,
                writers_r,
                direct_alpha,
                prompt_len,
            )

        recorder = TargetRecorder(
            decoder_layers,
            targets,
            prompt_len,
        )

        text = adapter.generate_text(
            model,
            processor,
            batch,
            max_new_tokens,
        )
        pred = adapter.normalize_generation(text)

        projections = {}
        for T in targets:
            if T not in recorder.states:
                projections[T] = float("nan")
            else:
                projections[T] = float(
                    np.dot(
                        recorder.states[T],
                        normalize_np(writers_r[T]),
                    )
                )

        return {
            "text": text,
            "prediction": pred,
            "mean_projection": safe_mean(projections.values()),
            "projection_by_target": projections,
            "applied": (
                dict(editor.applied)
                if editor is not None
                else dict(direct.applied)
                if direct is not None
                else {}
            ),
        }
    finally:
        if recorder is not None:
            recorder.close()
        if direct is not None:
            direct.close()
        if editor is not None:
            editor.close()


# =====================================================================
# Writer calibration / token ranking
# =====================================================================

def learn_writers(train, q_by_sid, targets, mode):
    writers = {T: {} for T in targets}
    geometry = []

    for T in targets:
        means = {}
        for rel in CANON_REL:
            xs = [
                q_by_sid[int(r["sid"])][T]
                for r in train if r["gt"] == rel
            ]
            if not xs:
                raise RuntimeError(
                    f"No calibration samples for relation={rel}, L{T}"
                )
            means[rel] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        common = np.mean(
            np.stack([means[r] for r in CANON_REL]),
            axis=0,
        ).astype(np.float32)

        for rel in CANON_REL:
            writers[T][rel] = (
                means[rel] - common
                if mode == "centered"
                else means[rel]
            ).astype(np.float32)

        for rel in CANON_REL:
            row = {
                "target_layer": T,
                "relation": rel,
                "writer_norm": float(np.linalg.norm(writers[T][rel])),
            }
            for other in CANON_REL:
                row[f"cos_{other}"] = cosine_np(
                    writers[T][rel],
                    writers[T][other],
                )
            geometry.append(row)

    return writers, geometry


def build_token_metadata(
    adapter,
    processor,
    ids,
    subject,
    reference,
):
    tokenizer = processor.tokenizer
    toks = [
        str(x)
        for x in tokenizer.convert_ids_to_tokens(ids)
    ]

    try:
        sspan, rspan = adapter.locate_object_spans(
            tokenizer, ids, subject, reference
        )
        spos = set(range(int(sspan[0]), int(sspan[1]) + 1))
        rpos = set(range(int(rspan[0]), int(rspan[1]) + 1))
    except Exception:
        spos, rpos = set(), set()

    visual = {
        i for i, tok in enumerate(toks)
        if "image_pad" in tok or "video_pad" in tok
    }

    categories = []
    for p, tok in enumerate(toks):
        if p == len(toks) - 1:
            cat = "last"
        elif p in spos:
            cat = "subject"
        elif p in rpos:
            cat = "reference"
        elif p in visual:
            cat = "visual"
        else:
            low = tok.lower().replace("ġ", "").replace("▁", "")
            if any(x in low for x in ("left", "right", "above", "below", "under")):
                cat = "relation_word"
            else:
                cat = "other_text"
        categories.append(cat)

    return toks, categories


def global_unique_selection(rows_by_layer, bundle, k):
    """
    Positive-only, global_unique:
    1) keep M>0
    2) for each absolute token position p, keep only the source layer with
       strongest M
    3) select global top-K positions
    """
    best_by_pos = {}

    for L in bundle:
        for r in rows_by_layer.get(L, []):
            if float(r["mediation"]) <= 0:
                continue
            pos = int(r["position"])
            cur = best_by_pos.get(pos)
            if cur is None or float(r["mediation"]) > float(cur["mediation"]):
                rr = dict(r)
                rr["source_layer"] = int(L)
                best_by_pos[pos] = rr

    selected = sorted(
        best_by_pos.values(),
        key=lambda x: float(x["mediation"]),
        reverse=True,
    )[:int(k)]

    specs = defaultdict(list)
    for r in selected:
        specs[int(r["source_layer"])].append(
            (int(r["position"]), r["delta_h"])
        )

    return dict(specs), selected


# =====================================================================
# Summary
# =====================================================================

def summarize(eval_rows, condition_rows):
    base = {int(r["sid"]): r for r in eval_rows}
    base_acc = safe_mean(r["baseline_correct"] for r in eval_rows)

    groups = defaultdict(list)
    for r in condition_rows:
        groups[
            (
                r["condition"],
                r["source_bundle"],
                int(r["k"]),
                float(r["alpha"]),
            )
        ].append(r)

    out = []
    for (condition, bundle, k, alpha), rows in groups.items():
        w2c = c2w = changed = 0
        corr = []
        proj_gain = []

        for r in rows:
            sid = int(r["sid"])
            b = base[sid]
            c = bool(r["correct"])
            corr.append(c)

            w2c += int((not bool(b["baseline_correct"])) and c)
            c2w += int(bool(b["baseline_correct"]) and (not c))
            changed += int(r["prediction"] != b["baseline_prediction"])

            proj_gain.append(
                float(r["mean_projection"])
                - float(b["baseline_mean_projection"])
            )

        acc = safe_mean(corr)
        out.append({
            "condition": condition,
            "source_bundle": bundle,
            "k": k,
            "alpha": alpha,
            "N": len(rows),
            "baseline_acc": base_acc,
            "edited_acc": acc,
            "gain": acc - base_acc,
            "W2C": w2c,
            "C2W": c2w,
            "net": w2c - c2w,
            "changed": changed,
            "mean_selected_tokens": safe_mean(
                r["n_selected_total"] for r in rows
            ),
            "mean_writer_projection_gain": safe_mean(proj_gain),
        })

    return sorted(
        out,
        key=lambda x: (
            x["edited_acc"],
            x["net"],
            x["mean_writer_projection_gain"],
        ),
        reverse=True,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    profile = MODEL_PROFILES[a.model]

    source_text = (
        profile["source_bundles"]
        if str(a.source_bundles).lower() == "auto"
        else a.source_bundles
    )
    target_text = (
        profile["target_layers"]
        if str(a.target_layers).lower() == "auto"
        else a.target_layers
    )

    bundles = parse_bundles(source_text)
    all_sources = sorted({L for b in bundles for L in b})
    targets = parse_ints(target_text)
    ks = parse_ints(a.ks)
    alphas = parse_floats(a.alphas)
    direct_alphas = parse_floats(a.direct_writer_alphas)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    adapter = DatasetAdapter(a)
    adapter.load()

    meta = adapter.meta_rows()
    meta = stratified_cap(meta, a.max_samples, a.seed)
    train, test = stratified_split(meta, a.train_ratio, a.seed)
    test = stratified_cap(test, a.eval_max_samples, a.seed + 1)

    if a.model not in adapter.specs:
        raise ValueError(
            f"{a.model!r} unavailable for dataset={a.dataset}; "
            f"available={sorted(adapter.specs)}"
        )
    spec = adapter.specs[a.model]

    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no {spec.model_class}"
        )

    load_kw = dict(
        dtype=(
            coco.resolve_dtype(spec.dtype_name)
            if hasattr(coco, "resolve_dtype")
            else {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[spec.dtype_name]
        ),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    print("=" * 150)
    print("CROSS-MODEL / CROSS-DATASET MIDDLE-TOKEN -> LATE-WRITER RECONSTRUCTION")
    print("=" * 150)
    print(f"model={a.model} | repo={spec.repo_id}")
    print(f"dataset={a.dataset}")
    print(f"prompt={adapter.prompt_path}")
    print(f"usable N={len(meta)} | calibration/train={len(train)} | eval={len(test)}")
    print(f"source bundles={[bundle_name(b) for b in bundles]}")
    print(f"target writer layers={targets}")
    print(f"K={ks} | alpha={alphas}")
    print("selection=positive global_unique")
    print("source last token EXCLUDED")
    print("ORACLE: GT relation chooses the late writer used as the causal target")
    print()

    model = processor = None

    try:
        model = model_cls.from_pretrained(
            spec.repo_id,
            **load_kw,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        adapter.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = adapter.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        for L in all_sources + targets:
            if not (0 <= L < n_layers):
                raise ValueError(
                    f"L{L} invalid for {a.model}; decoder has "
                    f"L0...L{n_layers - 1}"
                )

        for b in bundles:
            if max(b) >= max(targets):
                raise ValueError(
                    f"Source bundle {b} reaches/passes target region {targets}"
                )

        device = torch.device(a.device)

        # -------------------------------------------------------------
        # 1) Calibrate same-dataset late Real-Gray relation writers.
        # -------------------------------------------------------------
        q_by_sid = {}

        for m in tqdm(train, desc="CALIBRATE writers"):
            sid = int(m["sid"])
            real = gray = rb = gb = None

            try:
                real = adapter.record_image(sid)
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = adapter.make_batch(
                    processor,
                    real,
                    m["question_text"],
                    device,
                )
                gb = adapter.make_batch(
                    processor,
                    gray,
                    m["question_text"],
                    device,
                )

                hr = capture_cpu(
                    model,
                    decoder_layers,
                    rb,
                    targets,
                )
                hg = capture_cpu(
                    model,
                    decoder_layers,
                    gb,
                    targets,
                )

                q_by_sid[sid] = {
                    T: (
                        hr[T][0, -1, :]
                        - hg[T][0, -1, :]
                    ).astype(np.float32)
                    for T in targets
                }

            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        writers, writer_geometry = learn_writers(
            train,
            q_by_sid,
            targets,
            a.writer_mode,
        )
        write_csv(
            outdir / "writer_geometry.csv",
            writer_geometry,
        )

        np.savez_compressed(
            outdir / "learned_writers.npz",
            **{
                f"L{T}_{rel}": writers[T][rel]
                for T in targets
                for rel in CANON_REL
            },
        )

        # -------------------------------------------------------------
        # 2) Held-out token ranking + causal generation.
        # -------------------------------------------------------------
        eval_rows = []
        condition_rows = []
        selected_rows = []
        top_token_rows = []

        graph_layers = sorted(set(all_sources + targets))
        cut = min(all_sources)

        for m in tqdm(test, desc="EVAL middle reconstruction"):
            sid = int(m["sid"])
            gt = m["gt"]
            writers_r = {
                T: writers[T][gt]
                for T in targets
            }

            real = gray = rb = gb = cap = None

            try:
                real = adapter.record_image(sid)
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = adapter.make_batch(
                    processor,
                    real,
                    m["question_text"],
                    device,
                )
                gb = adapter.make_batch(
                    processor,
                    gray,
                    m["question_text"],
                    device,
                )

                ids = (
                    rb["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )
                toks, categories = build_token_metadata(
                    adapter,
                    processor,
                    ids,
                    m["subject"],
                    m["reference"],
                )

                # Clean baseline + target writer projection.
                clean = run_generation(
                    adapter,
                    model,
                    processor,
                    decoder_layers,
                    rb,
                    targets,
                    writers_r,
                    a.max_new_tokens,
                )
                base_pred = clean["prediction"]

                eval_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "baseline_prediction": base_pred,
                    "baseline_correct": base_pred == gt,
                    "baseline_text": clean["text"],
                    "baseline_mean_projection": clean["mean_projection"],
                    **{
                        f"baseline_proj_L{T}":
                        clean["projection_by_target"][T]
                        for T in targets
                    },
                })

                # Gray source states.
                hgray = capture_cpu(
                    model,
                    decoder_layers,
                    gb,
                    all_sources,
                )

                # One clean graph gives gradients for all source layers.
                with torch.enable_grad():
                    cap = forward_graph(
                        model,
                        decoder_layers,
                        rb,
                        graph_layers,
                        cut,
                    )

                    terms = []
                    for T in targets:
                        s_hat = torch.as_tensor(
                            normalize_np(writers_r[T]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        terms.append(
                            torch.dot(
                                cap.states[T][0, -1].float(),
                                s_hat,
                            )
                        )
                    objective = torch.stack(terms).sum()

                    grads = torch.autograd.grad(
                        objective,
                        [cap.states[S] for S in all_sources],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                    rows_by_layer = {}

                    for S, grad in zip(all_sources, grads):
                        Hreal = (
                            cap.states[S][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        Hgray = hgray[S][0].astype(np.float32)
                        G = (
                            grad[0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                        npos = min(
                            len(ids),
                            len(toks),
                            len(categories),
                            Hreal.shape[0],
                            Hgray.shape[0],
                            G.shape[0],
                        )

                        rowsS = []
                        # Exclude source last token.
                        for pos in range(max(0, npos - 1)):
                            delta = (
                                Hreal[pos] - Hgray[pos]
                            ).astype(np.float32)
                            g = G[pos]
                            med = float(np.dot(delta, g))

                            rowsS.append({
                                "source_layer": int(S),
                                "position": int(pos),
                                "token": toks[pos].replace("\n", "\\n"),
                                "category": categories[pos],
                                "mediation": med,
                                "delta_h_norm": float(np.linalg.norm(delta)),
                                "grad_norm": float(np.linalg.norm(g)),
                                "delta_h": delta,
                            })

                        rows_by_layer[S] = rowsS

                        # Save the strongest concrete positions for inspection.
                        positive = sorted(
                            [r for r in rowsS if r["mediation"] > 0],
                            key=lambda x: x["mediation"],
                            reverse=True,
                        )[:20]
                        for rank, r in enumerate(positive, 1):
                            top_token_rows.append({
                                "sid": sid,
                                "gt": gt,
                                "source_layer": S,
                                "rank": rank,
                                "position": r["position"],
                                "token": r["token"],
                                "category": r["category"],
                                "mediation": r["mediation"],
                                "delta_h_norm": r["delta_h_norm"],
                                "grad_norm": r["grad_norm"],
                            })

                cap.close()
                cap = None

                # Direct writer reference.
                for alpha_direct in direct_alphas:
                    direct = run_generation(
                        adapter,
                        model,
                        processor,
                        decoder_layers,
                        rb,
                        targets,
                        writers_r,
                        a.max_new_tokens,
                        direct_alpha=alpha_direct,
                    )

                    condition_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "condition": "direct_late_writer",
                        "source_bundle": (
                            "late:"
                            + "+".join(f"L{T}" for T in targets)
                        ),
                        "k": 0,
                        "alpha": alpha_direct,
                        "prediction": direct["prediction"],
                        "correct": direct["prediction"] == gt,
                        "text": direct["text"],
                        "mean_projection": direct["mean_projection"],
                        "n_selected_total": 0,
                    })

                # Positive global_unique middle-token conditions.
                for bundle in bundles:
                    bname = bundle_name(bundle)

                    for k in ks:
                        token_specs, selected = global_unique_selection(
                            rows_by_layer,
                            bundle,
                            k,
                        )
                        if not selected:
                            continue

                        for rank, r in enumerate(selected, 1):
                            selected_rows.append({
                                "sid": sid,
                                "gt": gt,
                                "source_bundle": bname,
                                "k": k,
                                "rank": rank,
                                "source_layer": r["source_layer"],
                                "position": r["position"],
                                "token": r["token"],
                                "category": r["category"],
                                "mediation": r["mediation"],
                                "delta_h_norm": r["delta_h_norm"],
                                "grad_norm": r["grad_norm"],
                            })

                        for alpha in alphas:
                            edited = run_generation(
                                adapter,
                                model,
                                processor,
                                decoder_layers,
                                rb,
                                targets,
                                writers_r,
                                a.max_new_tokens,
                                token_specs=token_specs,
                                token_alpha=alpha,
                            )

                            condition_rows.append({
                                "sid": sid,
                                "gt": gt,
                                "condition": "middle_positive_global_unique",
                                "source_bundle": bname,
                                "k": k,
                                "alpha": alpha,
                                "prediction": edited["prediction"],
                                "correct": edited["prediction"] == gt,
                                "text": edited["text"],
                                "mean_projection": edited["mean_projection"],
                                "n_selected_total": len(selected),
                            })

            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        summary = summarize(
            eval_rows,
            condition_rows,
        )

        write_csv(outdir / "baseline.csv", eval_rows)
        write_csv(outdir / "generation_conditions.csv", condition_rows)
        write_csv(outdir / "selected_tokens.csv", selected_rows)
        write_csv(outdir / "top_positive_tokens.csv", top_token_rows)
        write_csv(outdir / "summary.csv", summary)

        # Per-relation summary for the top-level diagnosis.
        per_rel = []
        base_lookup = {int(r["sid"]): r for r in eval_rows}

        for srow in summary:
            cond = srow["condition"]
            bundle = srow["source_bundle"]
            k = int(srow["k"])
            alpha = float(srow["alpha"])

            rows = [
                r for r in condition_rows
                if r["condition"] == cond
                and r["source_bundle"] == bundle
                and int(r["k"]) == k
                and float(r["alpha"]) == alpha
            ]

            for rel in CANON_REL:
                rr = [r for r in rows if r["gt"] == rel]
                if not rr:
                    continue

                bacc = safe_mean(
                    base_lookup[int(r["sid"])]["baseline_correct"]
                    for r in rr
                )
                eacc = safe_mean(r["correct"] for r in rr)

                w2c = sum(
                    (not base_lookup[int(r["sid"])]["baseline_correct"])
                    and bool(r["correct"])
                    for r in rr
                )
                c2w = sum(
                    base_lookup[int(r["sid"])]["baseline_correct"]
                    and (not bool(r["correct"]))
                    for r in rr
                )

                per_rel.append({
                    "condition": cond,
                    "source_bundle": bundle,
                    "k": k,
                    "alpha": alpha,
                    "relation": rel,
                    "N": len(rr),
                    "baseline_acc": bacc,
                    "edited_acc": eacc,
                    "gain": eacc - bacc,
                    "W2C": w2c,
                    "C2W": c2w,
                    "net": w2c - c2w,
                })

        write_csv(
            outdir / "per_relation_summary.csv",
            per_rel,
        )

        # -------------------------------------------------------------
        # Console report
        # -------------------------------------------------------------
        print("\n" + "=" * 150)
        print("RESULTS")
        print("=" * 150)

        base_acc = safe_mean(
            r["baseline_correct"]
            for r in eval_rows
        )
        print(
            f"{a.model} / {a.dataset} | "
            f"Baseline N={len(eval_rows)} acc={base_acc:.4f}"
        )

        direct_rows = [
            r for r in summary
            if r["condition"] == "direct_late_writer"
        ]
        middle_rows = [
            r for r in summary
            if r["condition"] == "middle_positive_global_unique"
        ]

        print("\nDirect late-writer reference:")
        for r in direct_rows:
            print(
                f"  alpha={float(r['alpha']):.3f} | "
                f"acc={float(r['edited_acc']):.4f} "
                f"gain={float(r['gain']):+.4f} | "
                f"W2C/C2W={int(r['W2C'])}/{int(r['C2W'])} | "
                f"dWriterProj={float(r['mean_writer_projection_gain']):+.4f}"
            )

        print("\nTop middle-token configurations:")
        print(
            f"{'bundle':<32s} {'K':>4s} {'alpha':>7s} "
            f"{'acc':>8s} {'gain':>8s} "
            f"{'W2C':>5s} {'C2W':>5s} {'net':>5s} "
            f"{'dWriterProj':>12s}"
        )
        print("-" * 150)

        for r in middle_rows[:30]:
            print(
                f"{r['source_bundle']:<32s} "
                f"{int(r['k']):>4d} "
                f"{float(r['alpha']):>7.3f} "
                f"{float(r['edited_acc']):>8.4f} "
                f"{float(r['gain']):>+8.4f} "
                f"{int(r['W2C']):>5d} "
                f"{int(r['C2W']):>5d} "
                f"{int(r['net']):>5d} "
                f"{float(r['mean_writer_projection_gain']):>+12.4f}"
            )

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "dataset": a.dataset,
            "prompt_path": str(adapter.prompt_path),
            "decoder_path": decoder_path,
            "n_decoder_layers": n_layers,
            "source_bundles": [list(b) for b in bundles],
            "target_layers": targets,
            "ks": ks,
            "alphas": alphas,
            "direct_writer_alphas": direct_alphas,
            "writer_mode": a.writer_mode,
            "train_ratio": a.train_ratio,
            "seed": a.seed,
            "usable_N": len(meta),
            "calibration_N": len(train),
            "eval_N": len(test),
            "relations": list(CANON_REL),
            "selection": "oracle relation -> joint late writer objective -> positive global_unique top-K",
            "mediation": "(h_real-h_gray)^T grad_h sum_T <h_T,last, normalize(s_r^T)>",
            "edit": "h_real[S,p] += alpha * (h_real[S,p] - h_gray[S,p])",
            "source_last_excluded": True,
            "gt_used_for_writer_target_selection": True,
            "model_parameter_training": False,
            "dataset_audit": adapter.audit,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

        print("\nSaved:", outdir)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
