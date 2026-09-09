#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which EARLIER TOKENS can most efficiently drive a late last-token state toward
the OLD learned Image-Gray relation writer s_r^L?

This script matches the intended question exactly.

1) Learn the old writer on a TRAIN split:
       q_i^L = h_real^L(last) - h_gray^L(last)
       mu_r^L = E[q_i^L | relation=r]
       s_r^L = mu_r^L - mean_k mu_k^L      (default: centered)

   r is the ORIGINAL fixed relation:
       left / right / on / under
   Internally COCO helpers may call on=above and under=below.

2) On a HELD-OUT REAL image, DO NOT use Real-Gray as the target.
   The target is the already learned fixed writer s_r^L itself.

   For target late layer L:
       J_{i,L} = < h_real^L(last), normalize(s_{r_i}^L) >

3) Backpropagate this objective to every earlier token state:
       g_{l,p -> L} = d J_{i,L} / d h_i^l(p)

   ||g|| answers:
       "If I am allowed a small unit-norm change to token p at earlier layer l,
        how effectively can I make the late last-token state move toward the
        learned writer s_r^L through the remaining network?"

Equivalently, g = J_{l,p->L}^T s_r^L is a Jacobian-transpose pullback of the
learned late writer.

IMPORTANT
---------
- The writer is learned from Image-Gray data.
- The evaluation target is NOT a sample's Real-Gray residual.
- The evaluation target is NOT the natural generation last state.
- The evaluation target is the fixed learned s_r^L direction.
- No random A/B/C/D mapping is used.
- This is local sensitivity/control geometry, not proof that the token naturally
  contributes that amount. Use activation patching later for causal validation.

Outputs
-------
writer_geometry.csv
    Norm/cosines of learned s_r^L.

per_token.csv
    Every source token's ||dJ/dh|| for each source->target layer pair.

top_tokens.csv
    Top-K most influential token positions per sample / source / target.

category_per_sample.csv
category_summary.csv
    Aggregate influence for visual / subject / reference / relation words /
    last token / other text.

relation_summary.csv
    Same summaries split by left/right/on/under.

target_alignment.csv
    Current held-out real last-state projection/cosine to learned s_r^L.

Recommended Qwen3B run (L28 onward):
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_learned_writer_token_pullback_quick.py \
  --model qwen-3b \
  --target-layers 28,30,32,34,35 \
  --source-layers 8,12,16,20,22,24,26,27,28,29,30,31,32,33,34 \
  --eval-max-samples 32 \
  --output-dir output/qwen3b_learned_writer_token_pullback \
  --overwrite

Qwen7B:
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_learned_writer_token_pullback_quick.py \
  --model qwen-7b \
  --target-layers 22,24,26,27 \
  --source-layers 6,10,14,18,20,21,22,23,24,25,26 \
  --eval-max-samples 32 \
  --output-dir output/qwen7b_learned_writer_token_pullback \
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
    p.add_argument("--max-samples", type=int, default=0, help="0 = all COCO_two")
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
                seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cosine_np(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
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
    if cat in ("visual", "subject", "reference", "last"):
        return cat
    return "other"


class Capture:
    def __init__(self, decoder_layers, req, cut_layer=None):
        self.states = {}
        self.handles = []
        self.cut_layer = cut_layer
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
            self.states[L] = traj.first_tensor(out)
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_last(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers)
    try:
        kw = dict(batch); kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing states at {missing}")
        return {
            L: cap.states[L][0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


def forward_graph(model, decoder_layers, batch, layers, cut):
    cap = Capture(decoder_layers, layers, cut_layer=cut)
    kw = dict(batch); kw["use_cache"] = False
    _ = model(**kw)
    missing = [L for L in layers if L not in cap.states]
    if missing:
        cap.close()
        raise RuntimeError(f"Missing graph states at {missing}")
    return cap


def learn_writers(train, q_by_sid, target_layers, mode):
    means = {L: {} for L in target_layers}
    writers = {L: {} for L in target_layers}
    rows = []

    for L in target_layers:
        for r in REL:
            xs = [q_by_sid[int(m["sid"])][L] for m in train if m["gt"] == r]
            if not xs:
                raise RuntimeError(f"No train examples for {r}")
            means[L][r] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        common = np.mean(np.stack([means[L][r] for r in REL]), axis=0).astype(np.float32)
        for r in REL:
            writers[L][r] = (
                means[L][r] - common if mode == "centered" else means[L][r]
            ).astype(np.float32)

        for r in REL:
            rows.append(dict(
                target_layer=L,
                relation=DISPLAY[r],
                writer_mode=mode,
                norm=float(np.linalg.norm(writers[L][r])),
                cos_left=cosine_np(writers[L][r], writers[L]["left"]),
                cos_right=cosine_np(writers[L][r], writers[L]["right"]),
                cos_on=cosine_np(writers[L][r], writers[L]["above"]),
                cos_under=cosine_np(writers[L][r], writers[L]["below"]),
            ))
    return writers, rows


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    if a.target_layers == "auto":
        target_layers = [28,30,32,34,35] if a.model == "qwen-3b" else [22,24,26,27]
    else:
        target_layers = parse_ints(a.target_layers)

    if a.source_layers == "auto":
        source_layers = (
            [8,12,16,20,22,24,26,27,28,29,30,31,32,33,34]
            if a.model == "qwen-3b"
            else [6,10,14,18,20,21,22,23,24,25,26]
        )
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
            sid=sid, gt=gt,
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
            if not 0 <= L < n_layers:
                raise ValueError(f"L{L} invalid: model has L0...L{n_layers-1}")

        source_layers = sorted({S for S in source_layers if S < max(target_layers)})
        cut = min(source_layers)
        graph_layers = sorted(set(source_layers + target_layers))
        device = torch.device(a.device)

        print("=" * 124)
        print("LEARNED IMAGE-GRAY WRITER s_r^L <- EARLIER TOKEN PULLBACK")
        print("=" * 124)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(f"train/test={len(train)}/{len(test)}")
        print(f"targets={target_layers}")
        print(f"sources={source_layers}")
        print(f"writer mode={a.writer_mode}")
        print("EVAL objective: < REAL h_target,last , normalized learned s_r^L >")
        print("No sample-specific Real-Gray residual is used as the evaluation target.")
        print()

        # ------------------------------------------------------------
        # Learn s_r^L from TRAIN Image-Gray residuals.
        # ------------------------------------------------------------
        q_by_sid = {}
        for m in tqdm(train, desc="TRAIN Image-Gray writers"):
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
                hr = capture_last(model, decoder_layers, rb, target_layers)
                hg = capture_last(model, decoder_layers, gb, target_layers)
                q_by_sid[sid] = {
                    L: (hr[L] - hg[L]).astype(np.float32) for L in target_layers
                }
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                del rb, gb
                gc.collect()

        writers, writer_geom = learn_writers(
            train, q_by_sid, target_layers, a.writer_mode
        )
        write_csv(out / "writer_geometry.csv", writer_geom)
        np.savez_compressed(
            out / "learned_imagegray_writers.npz",
            **{f"L{L}_{DISPLAY[r]}": writers[L][r]
               for L in target_layers for r in REL}
        )

        # ------------------------------------------------------------
        # Held-out REAL examples: pull fixed learned s_r^L backward.
        # ------------------------------------------------------------
        per_token = []
        top_rows = []
        cat_rows = []
        align_rows = []

        for m in tqdm(test, desc="TEST writer pullback"):
            sid = int(m["sid"])
            gt = m["gt"]
            image = batch = cap = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = base.make_question_batch(
                    processor=processor, image=image,
                    question_text=m["question_text"], device=device
                )

                ids = batch["input_ids"][0].detach().cpu().tolist()
                cats, toks = build_categories(
                    model, processor.tokenizer, ids,
                    m["subject"], m["reference"]
                )

                with torch.enable_grad():
                    cap = forward_graph(model, decoder_layers, batch, graph_layers, cut)

                    for ti, T in enumerate(target_layers):
                        sr = writers[T][gt]
                        sr_hat = torch.as_tensor(
                            normalize_np(sr),
                            device=cap.states[T].device,
                            dtype=torch.float32
                        )
                        hT = cap.states[T][0, -1, :].float()

                        # EXACT intended target:
                        # move the REAL target last state toward the FIXED
                        # learned Image-Gray writer s_r^L.
                        score = torch.dot(hT, sr_hat)

                        h_np = hT.detach().cpu().numpy().astype(np.float32)
                        align_rows.append(dict(
                            sid=sid,
                            relation=DISPLAY[gt],
                            target_layer=T,
                            projection_to_writer=float(score.detach().item()),
                            cosine_to_writer=cosine_np(h_np, sr),
                            h_norm=float(np.linalg.norm(h_np)),
                            writer_norm=float(np.linalg.norm(sr)),
                        ))

                        srcs = [S for S in source_layers if S < T]
                        grads = torch.autograd.grad(
                            score,
                            [cap.states[S] for S in srcs],
                            retain_graph=(ti < len(target_layers)-1),
                            create_graph=False,
                            allow_unused=False,
                        )

                        for S, g in zip(srcs, grads):
                            G = g[0].detach().float().cpu().numpy().astype(np.float32)
                            npos = min(len(ids), len(cats), len(toks), G.shape[0])
                            norms = np.linalg.norm(G[:npos], axis=1)
                            sq = norms * norms
                            total = float(np.sum(sq))
                            distance = T - S
                            path_type = "direct_parent_block" if distance == 1 else "mediated"

                            layer_rows = []
                            by_cat = defaultdict(list)
                            for p in range(npos):
                                row = dict(
                                    sid=sid,
                                    relation=DISPLAY[gt],
                                    target_layer=T,
                                    source_layer=S,
                                    layer_distance=distance,
                                    path_type=path_type,
                                    position=p,
                                    token_id=int(ids[p]),
                                    token=str(toks[p]).replace("\n", "\\n"),
                                    category=cats[p],
                                    broad_category=broad_category(cats[p]),
                                    grad_norm=float(norms[p]),
                                    grad_norm_sq=float(sq[p]),
                                    grad_energy_share=float(sq[p] / max(total, 1e-20)),
                                    target_projection=float(score.detach().item()),
                                )
                                per_token.append(row)
                                layer_rows.append(row)
                                by_cat[row["broad_category"]].append(row)

                            for cat, rs in by_cat.items():
                                cat_rows.append(dict(
                                    sid=sid,
                                    relation=DISPLAY[gt],
                                    target_layer=T,
                                    source_layer=S,
                                    layer_distance=distance,
                                    path_type=path_type,
                                    category=cat,
                                    token_count=len(rs),
                                    energy_share=float(
                                        sum(x["grad_norm_sq"] for x in rs) / max(total, 1e-20)
                                    ),
                                    mean_grad_norm=float(np.mean([x["grad_norm"] for x in rs])),
                                    max_grad_norm=float(np.max([x["grad_norm"] for x in rs])),
                                ))

                            for rank, row in enumerate(
                                sorted(layer_rows, key=lambda x: x["grad_norm"], reverse=True)[:a.topk],
                                1
                            ):
                                top_rows.append(dict(rank=rank, **row))

                cap.close(); cap = None

            finally:
                if cap is not None:
                    cap.close()
                if image is not None:
                    with contextlib.suppress(Exception): image.close()
                del batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        write_csv(out / "per_token.csv", per_token)
        write_csv(out / "top_tokens.csv", top_rows)
        write_csv(out / "category_per_sample.csv", cat_rows)
        write_csv(out / "target_alignment.csv", align_rows)

        groups = defaultdict(list)
        for r in cat_rows:
            groups[(r["target_layer"], r["source_layer"], r["category"])].append(r)
        summary = []
        for (T, S, cat), rs in sorted(groups.items()):
            summary.append(dict(
                target_layer=T,
                source_layer=S,
                layer_distance=T-S,
                path_type="direct_parent_block" if T-S == 1 else "mediated",
                category=cat,
                N=len(rs),
                mean_energy_share=float(np.mean([x["energy_share"] for x in rs])),
                mean_grad_norm=float(np.mean([x["mean_grad_norm"] for x in rs])),
                mean_max_grad_norm=float(np.mean([x["max_grad_norm"] for x in rs])),
            ))
        write_csv(out / "category_summary.csv", summary)

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
                mean_energy_share=float(np.mean([x["energy_share"] for x in rs])),
                mean_grad_norm=float(np.mean([x["mean_grad_norm"] for x in rs])),
                mean_max_grad_norm=float(np.mean([x["max_grad_norm"] for x in rs])),
            ))
        write_csv(out / "relation_summary.csv", rsummary)

        # Compact console summary.
        order = ("visual", "subject", "reference", "relation_words", "last", "other")
        print("\n" + "=" * 124)
        print("WHICH EARLIER TOKEN CAN DRIVE THE LATE LAST STATE TOWARD LEARNED s_r^L?")
        print("energy share = fraction of squared ||d<h_T,last,s_r>/dh_source_token||")
        print("=" * 124)

        for T in target_layers:
            print(f"\nTARGET learned writer at L{T}")
            srcs = sorted({r["source_layer"] for r in cat_rows if r["target_layer"] == T})
            for S in srcs:
                rs = [r for r in cat_rows if r["target_layer"] == T and r["source_layer"] == S]
                by_sid = defaultdict(lambda: defaultdict(float))
                for r in rs:
                    by_sid[int(r["sid"])][r["category"]] += float(r["energy_share"])
                val = {}
                for cat in order:
                    xs = [by_sid[s].get(cat, 0.0) for s in by_sid]
                    val[cat] = float(np.mean(xs)) if xs else 0.0
                typ = "DIRECT" if T-S == 1 else "mediated"
                print(
                    f"  L{S:02d}->L{T:02d} {typ:8s} d={T-S:2d} | "
                    f"vis={val['visual']:.3f} "
                    f"sub={val['subject']:.3f} "
                    f"ref={val['reference']:.3f} "
                    f"relWords={val['relation_words']:.3f} "
                    f"last={val['last']:.3f} "
                    f"other={val['other']:.3f}"
                )

        print("\nSaved:", out)

        (out / "metadata.json").write_text(json.dumps({
            "model_alias": a.model,
            "repo_id": spec.repo_id,
            "writer_definition": "s_r^L = class mean of TRAIN (Real-Gray) late last states, common-mean centered by default",
            "eval_objective": "<held-out REAL h_target,last, normalized fixed learned s_r^L>",
            "interpretation": "gradient norm is local optimal sensitivity of earlier token state for increasing late writer projection",
            "relations": ["left", "right", "on", "under"],
            "target_layers": target_layers,
            "source_layers": source_layers,
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
