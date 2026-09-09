#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL: find WHICH BLOCKS NATURALLY WRITE / ENHANCE the old learned
Image-Gray late relation writer s_r^T.

This is a forward-only "write increment" diagnostic.

Old learned writer
------------------
For every target late layer T and relation r:

    q_i^T = h_real^T(last) - h_gray^T(last)
    mu_r^T = E_train[q_i^T | relation=r]

Default:
    s_r^T = mu_r^T - mean_k mu_k^T

Relations are the ORIGINAL task:
    left / right / on / under
Internally the existing COCO helpers may normalize on->above, under->below.

Natural block write
-------------------
For a held-out sample i, define at every block l:

    q_i^l = h_real^l(last) - h_gray^l(last)

and the block's newly added Image-Gray residual:

    Delta q_i^l = q_i^l - q_i^(l-1)

For a FIXED learned late writer s_r^T, measure:

    carry_{l->T} = < q_i^(l-1), normalize(s_r^T) >
    after_{l->T} = < q_i^l,     normalize(s_r^T) >
    write_{l->T} = after - carry
                 = < Delta q_i^l, normalize(s_r^T) >

Therefore:
- write > 0 : block l NATURALLY moves the Image-Gray last-token residual
              toward the learned late writer.
- write < 0 : block l moves it away.
- positive_rate: fraction of held-out samples where the block writes toward it.

We also decompose:
    real_write = <h_real^l - h_real^(l-1), s_hat>
    gray_write = <h_gray^l - h_gray^(l-1), s_hat>
    image_specific_write = real_write - gray_write = write

This avoids the residual-identity-path problem from gradient sensitivity:
we measure NEWLY ADDED forward-pass increment, not total controllability.

Outputs
-------
block_write_summary.csv
    Main table: target writer T x source block l.
block_write_by_relation.csv
    Same, split by left/right/on/under.
per_sample_block_write.csv
    Per-sample values.
writer_geometry.csv
    Learned writer norms/cosines.
writer_alignment_trajectory.csv
    Mean <q_l, s_r^T> trajectory before/after each block.

Recommended Qwen3B quick run
----------------------------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_imagegray_writer_block_write_increment_quick.py \
  --model qwen-3b \
  --target-layers 28,30,32,34,35 \
  --block-layers 18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_imagegray_writer_block_write_increment \
  --overwrite

Qwen7B
------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_imagegray_writer_block_write_increment_quick.py \
  --model qwen-7b \
  --target-layers 22,24,26,27 \
  --block-layers 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --eval-max-samples 80 \
  --output-dir output/qwen7b_imagegray_writer_block_write_increment \
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
from typing import Dict, List

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
    p.add_argument("--target-layers", default="auto",
                   help="Late layers whose learned Image-Gray writer s_r^T is the target.")
    p.add_argument("--block-layers", default="auto",
                   help="Blocks l whose natural increment q_l-q_(l-1) is measured.")
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all COCO_two before split")
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all held-out")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted({int(x.strip().upper().replace("L", "")) for x in s.split(",") if x.strip()})


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
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
    if n < 1e-12:
        return v.copy()
    return (v / n).astype(np.float32)


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


class LastStateCapture:
    def __init__(self, decoder_layers, layers_req):
        self.states = {}
        self.handles = [decoder_layers[L].register_forward_hook(self._mk(L)) for L in layers_req]

    def _mk(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if h.ndim != 3:
                raise RuntimeError(f"Unexpected hidden shape at L{L}: {tuple(h.shape)}")
            self.states[L] = h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_last_states(model, decoder_layers, batch, layers_req):
    cap = LastStateCapture(decoder_layers, layers_req)
    try:
        kwargs = dict(batch)
        kwargs["use_cache"] = False
        _ = model(**kwargs)
        missing = [L for L in layers_req if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing last states at {missing}")
        return cap.states
    finally:
        cap.close()


def learn_writers(train, q_by_sid, target_layers, mode):
    means = {T: {} for T in target_layers}
    writers = {T: {} for T in target_layers}
    geom_rows = []

    for T in target_layers:
        for r in REL:
            xs = [q_by_sid[int(m["sid"])][T] for m in train if m["gt"] == r]
            if not xs:
                raise RuntimeError(f"No training examples for relation={r}, target L{T}")
            means[T][r] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        common = np.mean(np.stack([means[T][r] for r in REL]), axis=0).astype(np.float32)

        for r in REL:
            if mode == "centered":
                writers[T][r] = (means[T][r] - common).astype(np.float32)
            else:
                writers[T][r] = means[T][r].astype(np.float32)

        for r in REL:
            geom_rows.append(dict(
                target_layer=T,
                relation=DISPLAY[r],
                writer_mode=mode,
                writer_norm=float(np.linalg.norm(writers[T][r])),
                cos_left=cosine_np(writers[T][r], writers[T]["left"]),
                cos_right=cosine_np(writers[T][r], writers[T]["right"]),
                cos_on=cosine_np(writers[T][r], writers[T]["above"]),
                cos_under=cosine_np(writers[T][r], writers[T]["below"]),
            ))

    return writers, geom_rows


def mean_or_nan(xs):
    xs = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def sem_or_nan(xs):
    xs = [float(x) for x in xs if np.isfinite(float(x))]
    if len(xs) <= 1:
        return float("nan")
    return float(np.std(xs, ddof=1) / np.sqrt(len(xs)))


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.target_layers == "auto":
        target_layers = [28, 30, 32, 34, 35] if a.model == "qwen-3b" else [22, 24, 26, 27]
    else:
        target_layers = parse_ints(a.target_layers)

    if a.block_layers == "auto":
        block_layers = (
            list(range(18, 36)) if a.model == "qwen-3b"
            else list(range(10, 28))
        )
    else:
        block_layers = parse_ints(a.block_layers)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

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

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        for L in target_layers + block_layers:
            if not (0 <= L < n_layers):
                raise ValueError(
                    f"L{L} invalid for {a.model}; model has decoder layers L0...L{n_layers-1}"
                )

        # To compute Delta at block l we need output of l-1.
        valid_blocks = [L for L in block_layers if L >= 1]
        capture_layers = sorted(set(
            target_layers
            + valid_blocks
            + [L - 1 for L in valid_blocks]
        ))

        device = torch.device(a.device)

        print("=" * 128)
        print("NATURAL BLOCK WRITE INTO LEARNED IMAGE-GRAY LATE WRITER")
        print("=" * 128)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} | n_layers={n_layers}")
        print(f"train/test={len(train)}/{len(test)}")
        print(f"target writer layers={target_layers}")
        print(f"measured blocks={valid_blocks}")
        print(f"writer mode={a.writer_mode}")
        print("q_l = h_real_l(last) - h_gray_l(last)")
        print("Delta q_l = q_l - q_(l-1)")
        print("write(l->T) = <Delta q_l, normalized s_r^T>")
        print()

        # ------------------------------------------------------------------
        # TRAIN: learn the old Image-Gray writers at target late layers.
        # ------------------------------------------------------------------
        train_q = {}

        for m in tqdm(train, desc="TRAIN learned writers"):
            sid = int(m["sid"])
            real = gray = rb = gb = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                hr = capture_last_states(model, decoder_layers, rb, target_layers)
                hg = capture_last_states(model, decoder_layers, gb, target_layers)

                train_q[sid] = {
                    T: (hr[T] - hg[T]).astype(np.float32)
                    for T in target_layers
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

        writers, writer_geom = learn_writers(
            train, train_q, target_layers, a.writer_mode
        )
        write_csv(outdir / "writer_geometry.csv", writer_geom)
        np.savez_compressed(
            outdir / "learned_imagegray_writers.npz",
            **{
                f"L{T}_{DISPLAY[r]}": writers[T][r]
                for T in target_layers for r in REL
            }
        )

        # ------------------------------------------------------------------
        # TEST: measure natural forward increment at each block.
        # ------------------------------------------------------------------
        per_rows = []
        traj_rows = []

        for m in tqdm(test, desc="TEST natural block write"):
            sid = int(m["sid"])
            gt = m["gt"]
            real = gray = rb = gb = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                hr = capture_last_states(model, decoder_layers, rb, capture_layers)
                hg = capture_last_states(model, decoder_layers, gb, capture_layers)

                q = {
                    L: (hr[L] - hg[L]).astype(np.float32)
                    for L in capture_layers
                }

                for T in target_layers:
                    s = writers[T][gt]
                    s_hat = normalize_np(s)

                    # Full q trajectory alignment to this fixed target writer.
                    for L in sorted(x for x in capture_layers if x <= T):
                        traj_rows.append(dict(
                            sid=sid,
                            relation=DISPLAY[gt],
                            target_layer=T,
                            state_layer=L,
                            projection=float(np.dot(q[L], s_hat)),
                            cosine=cosine_np(q[L], s),
                            q_norm=float(np.linalg.norm(q[L])),
                        ))

                    for L in valid_blocks:
                        if L > T:
                            continue
                        prev = L - 1
                        if prev not in q or L not in q:
                            continue

                        dq = (q[L] - q[prev]).astype(np.float32)
                        dreal = (hr[L] - hr[prev]).astype(np.float32)
                        dgray = (hg[L] - hg[prev]).astype(np.float32)

                        carry = float(np.dot(q[prev], s_hat))
                        after = float(np.dot(q[L], s_hat))
                        write = float(np.dot(dq, s_hat))
                        real_write = float(np.dot(dreal, s_hat))
                        gray_write = float(np.dot(dgray, s_hat))

                        # Numerical identity check:
                        residual_check = float(write - (real_write - gray_write))

                        per_rows.append(dict(
                            sid=sid,
                            relation=DISPLAY[gt],
                            target_layer=T,
                            block_layer=L,
                            distance_to_target=T-L,
                            carry_projection=carry,
                            after_projection=after,
                            write_gain=write,
                            real_write=real_write,
                            gray_write=gray_write,
                            real_minus_gray_write=real_write-gray_write,
                            residual_check=residual_check,
                            delta_q_norm=float(np.linalg.norm(dq)),
                            cos_deltaq_writer=cosine_np(dq, s),
                            writer_norm=float(np.linalg.norm(s)),
                            positive_write=(write > 0),
                        ))
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

        write_csv(outdir / "per_sample_block_write.csv", per_rows)
        write_csv(outdir / "writer_alignment_trajectory.csv", traj_rows)

        # ------------------------------------------------------------------
        # Aggregate overall.
        # ------------------------------------------------------------------
        groups = defaultdict(list)
        for r in per_rows:
            groups[(r["target_layer"], r["block_layer"])].append(r)

        summary = []
        for (T, L), rs in sorted(groups.items()):
            writes = [r["write_gain"] for r in rs]
            real_w = [r["real_write"] for r in rs]
            gray_w = [r["gray_write"] for r in rs]
            coss = [r["cos_deltaq_writer"] for r in rs]

            summary.append(dict(
                target_layer=T,
                block_layer=L,
                distance_to_target=T-L,
                N=len(rs),
                mean_write_gain=mean_or_nan(writes),
                sem_write_gain=sem_or_nan(writes),
                median_write_gain=float(np.median(writes)),
                positive_write_rate=float(np.mean([r["positive_write"] for r in rs])),
                mean_real_write=mean_or_nan(real_w),
                mean_gray_write=mean_or_nan(gray_w),
                mean_cos_deltaq_writer=mean_or_nan(coss),
                mean_delta_q_norm=mean_or_nan([r["delta_q_norm"] for r in rs]),
                mean_carry_projection=mean_or_nan([r["carry_projection"] for r in rs]),
                mean_after_projection=mean_or_nan([r["after_projection"] for r in rs]),
            ))
        write_csv(outdir / "block_write_summary.csv", summary)

        # Relation-stratified.
        rgroups = defaultdict(list)
        for r in per_rows:
            rgroups[(r["relation"], r["target_layer"], r["block_layer"])].append(r)

        rel_summary = []
        for (rel, T, L), rs in sorted(rgroups.items()):
            writes = [r["write_gain"] for r in rs]
            rel_summary.append(dict(
                relation=rel,
                target_layer=T,
                block_layer=L,
                distance_to_target=T-L,
                N=len(rs),
                mean_write_gain=mean_or_nan(writes),
                sem_write_gain=sem_or_nan(writes),
                median_write_gain=float(np.median(writes)),
                positive_write_rate=float(np.mean([r["positive_write"] for r in rs])),
                mean_cos_deltaq_writer=mean_or_nan([r["cos_deltaq_writer"] for r in rs]),
                mean_delta_q_norm=mean_or_nan([r["delta_q_norm"] for r in rs]),
            ))
        write_csv(outdir / "block_write_by_relation.csv", rel_summary)

        # ------------------------------------------------------------------
        # Console: ranked natural writers for each late target.
        # ------------------------------------------------------------------
        print("\n" + "=" * 128)
        print("BLOCKS THAT NATURALLY WRITE TOWARD THE LEARNED LATE WRITER")
        print("write = < (Real-Gray)_L - (Real-Gray)_(L-1), normalized s_r^T >")
        print("=" * 128)

        for T in target_layers:
            rowsT = [r for r in summary if r["target_layer"] == T]
            rowsT_rank = sorted(rowsT, key=lambda x: x["mean_write_gain"], reverse=True)

            print(f"\nTARGET writer s_r^L{T}")
            print("  ranked by mean positive write gain:")
            for rank, r in enumerate(rowsT_rank, 1):
                print(
                    f"  #{rank:02d} block L{r['block_layer']:02d} "
                    f"(d={r['distance_to_target']:2d}) | "
                    f"write={r['mean_write_gain']:+.4f} ± {r['sem_write_gain']:.4f} | "
                    f"positive={r['positive_write_rate']:.3f} | "
                    f"cos(Δq,s)={r['mean_cos_deltaq_writer']:+.4f} | "
                    f"real={r['mean_real_write']:+.4f} gray={r['mean_gray_write']:+.4f}"
                )

        print("\nInterpretation:")
        print("  large +write and high positive rate => block naturally ADDS the learned writer")
        print("  ~0 write => mostly carries / transforms without adding this writer direction")
        print("  negative write => block naturally moves away from the learned writer")
        print("\nSaved:", outdir)

        (outdir / "metadata.json").write_text(json.dumps({
            "model_alias": a.model,
            "repo_id": spec.repo_id,
            "writer_mode": a.writer_mode,
            "writer_definition": "TRAIN class mean of Real-Gray last residual, common-mean centered by default",
            "relations": ["left", "right", "on", "under"],
            "target_layers": target_layers,
            "block_layers": valid_blocks,
            "write_definition": "<[(h_real_L-h_gray_L)-(h_real_L-1-h_gray_L-1)], normalized s_r^T>",
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
