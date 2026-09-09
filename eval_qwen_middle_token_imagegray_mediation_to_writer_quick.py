#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL: middle-token Real-vs-Gray mediation into the OLD learned late writer.

Question
--------
Which NON-LAST token in an earlier/middle layer actually carries image-dependent
activation that pushes a later last-token state toward the learned writer s_r^T?

Old late writer
---------------
On TRAIN:

    q_i^T = h_real^T(last) - h_gray^T(last)

For relation r in {left, right, on, under}:

    mu_r^T = E[q_i^T | relation=r]
    s_r^T  = mu_r^T - mean_k mu_k^T        # default centered writer

This is the same Image-Gray relation writer family used in the old oracle
late-steering experiments.

Middle-token mediation score
-----------------------------
For a held-out sample and source layer S < target layer T:

    J_T = < h_real^T(last), normalize(s_r^T) >

For every source token p:

    g_{S,p->T} = d J_T / d h_real^S(p)

and the natural image-dependent activation is

    delta h_{S,p} = h_real^S(p) - h_gray^S(p)

Define the first-order mediation / contribution score

    M_{S,p->T} = < delta h_{S,p}, g_{S,p->T} >

Interpretation:
    M > 0 : the token's natural Real-vs-Gray change pushes the late last state
            TOWARD the learned writer.
    M < 0 : it pushes away.
    |M|   : first-order magnitude.

This approximates the loss of writer projection if that one source token were
patched from REAL back to GRAY:

    J_clean - J_patch(real->gray at S,p) ≈ M

Crucially:
- source LAST token is EXCLUDED from ranking/aggregation by default;
- no A/B/C/D randomized mapping;
- target is the fixed learned Image-Gray writer s_r^T;
- this is about middle tokens -> late last-token writer.

Outputs
-------
per_token_mediation.csv
    Signed M, |M|, grad norm, delta-h norm for every non-last source token.
top_tokens.csv
    Top-K positive middle tokens per sample/source/target.
category_per_sample.csv
    Per-sample category sums for visual / subject / reference / relation_words /
    other_text.
category_summary.csv
    Aggregate by source layer, target writer layer, token category.
relation_summary.csv
    Same split by left/right/on/under.
writer_geometry.csv
    Learned s_r^T geometry.

Recommended Qwen3B quick run
----------------------------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_middle_token_imagegray_mediation_to_writer_quick.py \
  --model qwen-3b \
  --target-layers 28,30,32,34,35 \
  --source-layers 8,12,16,18,20,22,24,26 \
  --eval-max-samples 32 \
  --topk 12 \
  --output-dir output/qwen3b_middle_token_mediation_to_writer \
  --overwrite

Stronger:
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_middle_token_imagegray_mediation_to_writer_quick.py \
  --model qwen-3b \
  --target-layers 28,30,32,34,35 \
  --source-layers 8,10,12,14,16,18,20,22,24,26 \
  --eval-max-samples 80 \
  --topk 10 \
  --output-dir output/qwen3b_middle_token_mediation_to_writer_N80 \
  --overwrite

Qwen7B:
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_middle_token_imagegray_mediation_to_writer_quick.py \
  --model qwen-7b \
  --target-layers 22,24,26,27 \
  --source-layers 6,8,10,12,14,16,18,20 \
  --eval-max-samples 32 \
  --topk 12 \
  --output-dir output/qwen7b_middle_token_mediation_to_writer \
  --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj

REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl",
                   default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--target-layers", default="auto")
    p.add_argument("--source-layers", default="auto")
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all COCO_two before split")
    p.add_argument("--eval-max-samples", type=int, default=32, help="0 = all held-out")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted({int(x.strip().upper().replace("L", "")) for x in s.split(",") if x.strip()})


def write_csv(path: Path, rows):
    if not rows:
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


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < 1e-12 else (v / n).astype(np.float32)


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def all_subsequence_positions(ids, pat):
    if not pat or len(pat) > len(ids):
        return []
    out = []
    n = len(pat)
    for i in range(len(ids) - n + 1):
        if ids[i:i+n] == pat:
            out.extend(range(i, i+n))
    return sorted(set(out))


def find_text_positions(tokenizer, ids, text):
    out = set()
    for variant in (text, " " + text):
        pat = tokenizer.encode(variant, add_special_tokens=False)
        out.update(all_subsequence_positions(ids, pat))
    return sorted(out)


def build_categories(model, tokenizer, ids, subject, reference):
    toks = [str(x) for x in tokenizer.convert_ids_to_tokens(ids)]
    sub_pos = set(find_text_positions(tokenizer, ids, subject))
    ref_pos = set(find_text_positions(tokenizer, ids, reference))

    rel_pos = {}
    for w in ("left", "right", "above", "below", "on", "under"):
        rel_pos[w] = set(find_text_positions(tokenizer, ids, w))

    image_ids = set()
    for attr in ("image_token_id", "video_token_id"):
        x = getattr(model.config, attr, None)
        if isinstance(x, int):
            image_ids.add(int(x))
    for sp in ("<|image_pad|>", "<|video_pad|>"):
        with contextlib.suppress(Exception):
            x = tokenizer.convert_tokens_to_ids(sp)
            if isinstance(x, int) and x >= 0:
                image_ids.add(int(x))

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    cats = []
    for p, (tid, tok) in enumerate(zip(ids, toks)):
        if p == len(ids) - 1:
            cat = "last"
        elif p in sub_pos:
            cat = "subject"
        elif p in ref_pos:
            cat = "reference"
        else:
            hit = None
            for w, ps in rel_pos.items():
                if p in ps:
                    hit = w
                    break
            if hit is not None:
                cat = f"relation_word:{hit}"
            elif tid in image_ids or "image_pad" in tok or "video_pad" in tok:
                cat = "visual"
            elif "vision_start" in tok or "vision_end" in tok:
                cat = "vision_boundary"
            elif tid in special_ids or (tok.startswith("<|") and tok.endswith("|>")):
                cat = "special"
            else:
                cat = "other_text"
        cats.append(cat)
    return cats, toks


def broad_category(cat):
    if cat.startswith("relation_word:"):
        return "relation_words"
    if cat in ("visual", "subject", "reference"):
        return cat
    if cat == "last":
        return "last"
    return "other_text"


class Capture:
    def __init__(self, decoder_layers, req, cut_layer=None, detach_cpu=False):
        self.states = {}
        self.handles = []
        self.cut_layer = cut_layer
        self.detach_cpu = detach_cpu
        for L in req:
            if cut_layer is not None and L == cut_layer:
                self.handles.append(decoder_layers[L].register_forward_hook(self._cut(L)))
            else:
                self.handles.append(decoder_layers[L].register_forward_hook(self._keep(L)))

    def _cut(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            y = h.detach().clone().requires_grad_(True)
            self.states[L] = y
            return traj.replace_first_tensor(out, y)
        return hook

    def _keep(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if self.detach_cpu:
                self.states[L] = h.detach().float().cpu()
            else:
                self.states[L] = h
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_states_cpu(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers, detach_cpu=True)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing states at {missing}")
        return {L: cap.states[L].numpy().astype(np.float32) for L in layers}
    finally:
        cap.close()


def forward_graph(model, decoder_layers, batch, layers, cut):
    cap = Capture(decoder_layers, layers, cut_layer=cut, detach_cpu=False)
    kw = dict(batch)
    kw["use_cache"] = False
    _ = model(**kw)
    missing = [L for L in layers if L not in cap.states]
    if missing:
        cap.close()
        raise RuntimeError(f"Missing graph states at {missing}")
    return cap


def learn_writers(train, q_by_sid, target_layers, mode):
    means = {T: {} for T in target_layers}
    writers = {T: {} for T in target_layers}
    rows = []

    for T in target_layers:
        for r in REL:
            xs = [q_by_sid[int(m["sid"])][T] for m in train if m["gt"] == r]
            if not xs:
                raise RuntimeError(f"No train examples for relation={r}, target=L{T}")
            means[T][r] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        common = np.mean(np.stack([means[T][r] for r in REL]), axis=0).astype(np.float32)

        for r in REL:
            writers[T][r] = (
                means[T][r] - common if mode == "centered" else means[T][r]
            ).astype(np.float32)

        for r in REL:
            rows.append(dict(
                target_layer=T,
                relation=DISPLAY[r],
                writer_mode=mode,
                writer_norm=float(np.linalg.norm(writers[T][r])),
                cos_left=cosine_np(writers[T][r], writers[T]["left"]),
                cos_right=cosine_np(writers[T][r], writers[T]["right"]),
                cos_on=cosine_np(writers[T][r], writers[T]["above"]),
                cos_under=cosine_np(writers[T][r], writers[T]["below"]),
            ))
    return writers, rows


def mean_or_nan(xs):
    xs = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def sem_or_nan(xs):
    xs = [float(x) for x in xs if np.isfinite(float(x))]
    if len(xs) < 2:
        return float("nan")
    return float(np.std(xs, ddof=1) / np.sqrt(len(xs)))


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.target_layers == "auto":
        target_layers = [28,30,32,34,35] if a.model == "qwen-3b" else [22,24,26,27]
    else:
        target_layers = parse_ints(a.target_layers)

    if a.source_layers == "auto":
        source_layers = [8,12,16,18,20,22,24,26] if a.model == "qwen-3b" else [6,8,10,12,14,16,18,20]
    else:
        source_layers = parse_ints(a.source_layers)

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _ = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(x.sid): x for x in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append(dict(
            sid=sid,
            gt=gt,
            subject=str(p["subject"]),
            reference=str(p["reference"]),
            question_text=str(p["question_text"]),
        ))

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, test = traj.stratified_split(meta, a.train_ratio, a.seed)
    test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    model = processor = None
    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        for L in target_layers + source_layers:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid; model has L0...L{n_layers-1}")

        source_layers = sorted({S for S in source_layers if S < max(target_layers)})
        if not source_layers:
            raise ValueError("No valid source layer earlier than target layers")
        cut = min(source_layers)
        capture_layers = sorted(set(source_layers + target_layers))
        device = torch.device(a.device)

        print("=" * 126)
        print("MIDDLE TOKEN (REAL-GRAY) MEDIATION -> LEARNED LATE WRITER")
        print("=" * 126)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} | n_layers={n_layers}")
        print(f"train/test={len(train)}/{len(test)}")
        print(f"source middle layers={source_layers}")
        print(f"target writer layers={target_layers}")
        print("source last token is excluded from mediation ranking")
        print("M = <h_real_source-h_gray_source, d<h_real_target,last,s_r>/dh_real_source>")
        print()

        # ------------------------------------------------------------
        # TRAIN: learn old Image-Gray late writers.
        # ------------------------------------------------------------
        q_by_sid = {}
        for m in tqdm(train, desc="TRAIN learned writers"):
            sid = int(m["sid"])
            real = gray = rb = gb = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor, image=real,
                    question_text=m["question_text"], device=device
                )
                gb = base.make_question_batch(
                    processor=processor, image=gray,
                    question_text=m["question_text"], device=device
                )

                hr = capture_states_cpu(model, decoder_layers, rb, target_layers)
                hg = capture_states_cpu(model, decoder_layers, gb, target_layers)

                q_by_sid[sid] = {
                    T: (hr[T][0, -1, :] - hg[T][0, -1, :]).astype(np.float32)
                    for T in target_layers
                }
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                del rb, gb
                gc.collect()

        writers, writer_geom = learn_writers(train, q_by_sid, target_layers, a.writer_mode)
        write_csv(out / "writer_geometry.csv", writer_geom)
        np.savez_compressed(
            out / "learned_imagegray_writers.npz",
            **{f"L{T}_{DISPLAY[r]}": writers[T][r]
               for T in target_layers for r in REL}
        )

        # ------------------------------------------------------------
        # TEST: gray source activations + real graph and gradient.
        # ------------------------------------------------------------
        per_rows = []
        top_rows = []
        cat_rows = []

        for m in tqdm(test, desc="TEST middle-token mediation"):
            sid = int(m["sid"])
            gt = m["gt"]
            real = gray = rb = gb = cap = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor, image=real,
                    question_text=m["question_text"], device=device
                )
                gb = base.make_question_batch(
                    processor=processor, image=gray,
                    question_text=m["question_text"], device=device
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = build_categories(
                    model, processor.tokenizer, ids,
                    m["subject"], m["reference"]
                )

                # Gray source activations are only needed at middle source layers.
                hgray = capture_states_cpu(model, decoder_layers, gb, source_layers)

                with torch.enable_grad():
                    cap = forward_graph(model, decoder_layers, rb, capture_layers, cut)

                    for ti, T in enumerate(target_layers):
                        srcs = [S for S in source_layers if S < T]
                        if not srcs:
                            continue

                        sr = writers[T][gt]
                        sr_hat = torch.as_tensor(
                            normalize_np(sr),
                            device=cap.states[T].device,
                            dtype=torch.float32
                        )
                        hT = cap.states[T][0, -1, :].float()
                        score = torch.dot(hT, sr_hat)

                        grads = torch.autograd.grad(
                            score,
                            [cap.states[S] for S in srcs],
                            retain_graph=(ti < len(target_layers)-1),
                            create_graph=False,
                            allow_unused=False,
                        )

                        for S, g in zip(srcs, grads):
                            Hreal = cap.states[S][0].detach().float().cpu().numpy().astype(np.float32)
                            Hgray = hgray[S][0].astype(np.float32)
                            G = g[0].detach().float().cpu().numpy().astype(np.float32)

                            npos = min(len(ids), len(cats), len(toks),
                                       Hreal.shape[0], Hgray.shape[0], G.shape[0])
                            # Explicitly EXCLUDE the source last token.
                            valid_positions = list(range(max(0, npos - 1)))

                            layer_rows = []
                            for p in valid_positions:
                                delta = Hreal[p] - Hgray[p]
                                grad = G[p]
                                mediation = float(np.dot(delta, grad))
                                delta_norm = float(np.linalg.norm(delta))
                                grad_norm = float(np.linalg.norm(grad))
                                denom = max(delta_norm * grad_norm, 1e-20)
                                align = float(mediation / denom)

                                row = dict(
                                    sid=sid,
                                    relation=DISPLAY[gt],
                                    target_layer=T,
                                    source_layer=S,
                                    layer_distance=T-S,
                                    position=p,
                                    token_id=int(ids[p]),
                                    token=str(toks[p]).replace("\n", "\\n"),
                                    category=cats[p],
                                    broad_category=broad_category(cats[p]),
                                    mediation=mediation,
                                    abs_mediation=abs(mediation),
                                    positive=(mediation > 0),
                                    delta_h_norm=delta_norm,
                                    grad_norm=grad_norm,
                                    cos_delta_grad=align,
                                    target_projection=float(score.detach().item()),
                                )
                                per_rows.append(row)
                                layer_rows.append(row)

                            # Normalize absolute contribution within the non-last positions
                            # of this source layer, for category composition only.
                            total_abs = sum(r["abs_mediation"] for r in layer_rows)
                            by_cat = defaultdict(list)
                            for r0 in layer_rows:
                                r0["abs_share_nonlast"] = (
                                    r0["abs_mediation"] / max(total_abs, 1e-20)
                                )
                                by_cat[r0["broad_category"]].append(r0)

                            for cat, rs in by_cat.items():
                                cat_rows.append(dict(
                                    sid=sid,
                                    relation=DISPLAY[gt],
                                    target_layer=T,
                                    source_layer=S,
                                    layer_distance=T-S,
                                    category=cat,
                                    token_count=len(rs),
                                    signed_sum=float(sum(x["mediation"] for x in rs)),
                                    positive_sum=float(sum(max(x["mediation"], 0.0) for x in rs)),
                                    negative_sum=float(sum(min(x["mediation"], 0.0) for x in rs)),
                                    abs_sum=float(sum(x["abs_mediation"] for x in rs)),
                                    abs_share_nonlast=float(sum(x["abs_share_nonlast"] for x in rs)),
                                    positive_token_rate=float(np.mean([x["positive"] for x in rs])),
                                    mean_delta_h_norm=float(np.mean([x["delta_h_norm"] for x in rs])),
                                    mean_grad_norm=float(np.mean([x["grad_norm"] for x in rs])),
                                ))

                            # Top positive contributors, not top |M|.
                            for rank, r0 in enumerate(
                                sorted(layer_rows, key=lambda x: x["mediation"], reverse=True)[:a.topk],
                                1
                            ):
                                top_rows.append(dict(rank=rank, **r0))

                cap.close()
                cap = None

            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        write_csv(out / "per_token_mediation.csv", per_rows)
        write_csv(out / "top_tokens.csv", top_rows)
        write_csv(out / "category_per_sample.csv", cat_rows)

        # ------------------------------------------------------------
        # Aggregate by T, S, category.
        # ------------------------------------------------------------
        groups = defaultdict(list)
        for r in cat_rows:
            groups[(r["target_layer"], r["source_layer"], r["category"])].append(r)

        summary = []
        for (T, S, cat), rs in sorted(groups.items()):
            summary.append(dict(
                target_layer=T,
                source_layer=S,
                layer_distance=T-S,
                category=cat,
                N=len(rs),
                mean_signed_sum=mean_or_nan([r["signed_sum"] for r in rs]),
                sem_signed_sum=sem_or_nan([r["signed_sum"] for r in rs]),
                mean_positive_sum=mean_or_nan([r["positive_sum"] for r in rs]),
                mean_negative_sum=mean_or_nan([r["negative_sum"] for r in rs]),
                mean_abs_sum=mean_or_nan([r["abs_sum"] for r in rs]),
                mean_abs_share_nonlast=mean_or_nan([r["abs_share_nonlast"] for r in rs]),
                mean_positive_token_rate=mean_or_nan([r["positive_token_rate"] for r in rs]),
                mean_delta_h_norm=mean_or_nan([r["mean_delta_h_norm"] for r in rs]),
                mean_grad_norm=mean_or_nan([r["mean_grad_norm"] for r in rs]),
            ))
        write_csv(out / "category_summary.csv", summary)

        # Relation-stratified.
        rgroups = defaultdict(list)
        for r in cat_rows:
            rgroups[(r["relation"], r["target_layer"], r["source_layer"], r["category"])].append(r)

        rsummary = []
        for (rel, T, S, cat), rs in sorted(rgroups.items()):
            rsummary.append(dict(
                relation=rel,
                target_layer=T,
                source_layer=S,
                layer_distance=T-S,
                category=cat,
                N=len(rs),
                mean_signed_sum=mean_or_nan([r["signed_sum"] for r in rs]),
                sem_signed_sum=sem_or_nan([r["signed_sum"] for r in rs]),
                mean_positive_sum=mean_or_nan([r["positive_sum"] for r in rs]),
                mean_abs_sum=mean_or_nan([r["abs_sum"] for r in rs]),
                mean_abs_share_nonlast=mean_or_nan([r["abs_share_nonlast"] for r in rs]),
            ))
        write_csv(out / "relation_summary.csv", rsummary)

        # ------------------------------------------------------------
        # Console summary.
        # ------------------------------------------------------------
        order = ("visual", "subject", "reference", "relation_words", "other_text")
        print("\n" + "=" * 126)
        print("NON-LAST MIDDLE-TOKEN REAL-GRAY MEDIATION INTO LEARNED WRITER")
        print("M = <delta h_source(real-gray), gradient of target writer projection>")
        print("=" * 126)

        for T in target_layers:
            print(f"\nTARGET learned writer at L{T}")
            srcs = sorted({r["source_layer"] for r in summary if r["target_layer"] == T})
            for S in srcs:
                rs = [r for r in summary if r["target_layer"] == T and r["source_layer"] == S]
                bycat = {r["category"]: r for r in rs}

                # Total signed mediation over all non-last categories.
                total_signed = sum(bycat.get(c, {}).get("mean_signed_sum", 0.0) for c in order)
                total_positive = sum(bycat.get(c, {}).get("mean_positive_sum", 0.0) for c in order)

                parts = []
                for c in order:
                    r = bycat.get(c)
                    if r is None:
                        parts.append(f"{c}=NA")
                    else:
                        parts.append(
                            f"{c}={r['mean_signed_sum']:+.3f}"
                            f"(absShare={r['mean_abs_share_nonlast']:.2f})"
                        )
                print(
                    f"  L{S:02d}->L{T:02d} d={T-S:2d} | "
                    f"signed_total={total_signed:+.3f} positive_total={total_positive:+.3f} | "
                    + " ".join(parts)
                )

        print("\nInterpretation:")
        print("  positive signed_sum => that token category's natural Real-Gray activation pushes the late state toward s_r")
        print("  negative signed_sum => pushes away")
        print("  absShare => where the first-order non-last mediation magnitude is concentrated")
        print("  source last token is excluded")
        print("\nSaved:", out)

        (out / "metadata.json").write_text(json.dumps({
            "model_alias": a.model,
            "repo_id": spec.repo_id,
            "writer_definition": "TRAIN mean(Real-Gray late last | relation), common-mean centered by default",
            "relations": ["left", "right", "on", "under"],
            "source_layers": source_layers,
            "target_layers": target_layers,
            "mediation_definition": "<h_real_source-h_gray_source, d<h_real_target_last,s_r_hat>/dh_real_source>",
            "source_last_token_excluded": True,
            "train_N": len(train),
            "test_N": len(test),
            "seed": a.seed,
        }, indent=2), encoding="utf-8")

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
