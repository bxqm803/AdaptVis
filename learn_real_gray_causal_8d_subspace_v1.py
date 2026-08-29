#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learn an 8D paired Real-vs-Gray causal subspace at the last prompt token.

The rank is 8 from the start.  Spatial labels do NOT define the subspace.

At layer L:
    delta_i = h_last(real)_i - h_last(gray)_i

TRAIN SVD is used only to define a broad candidate pool B_pool (default 64D).
Inside that pool we learn an 8D orthonormal subspace W:

    W = B_pool A,  A^T A = I
    delta_i^8 = W W^T delta_i

Bidirectional training objective with the model frozen:
    Gray + delta_i^8 -> Real behavior
    Real - delta_i^8 -> Gray behavior

Training uses differentiable first-step four-relation logits.
Final evaluation uses fresh actual model.generate().

Default TRAIN/TEST cohort = full_rescuable:
    Real generation correct,
    Gray generation wrong,
    Gray + FULL Real-Gray last-token delta recovers the Real/GT answer.

Controls:
    learned8       learned causal 8D
    top8           top-8 SVD variance subspace
    random8        random 8D inside the SAME 64D pool, norm-matched per sample
    complement     full_delta - learned8_delta
    reverse test   Real - learned8_delta -> Gray behavior

Recommended:
CUDA_VISIBLE_DEVICES=0 python learn_real_gray_causal_8d_subspace_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --train-cache output/qwen7b_real_gray_causal_spatial_subspace_v1/train_real_gray_last_delta.npz \
  --dataset coco_two --data-root data --model qwen-7b --device cuda:0 \
  --layer 27 --rank 8 --pool-rank 64 --epochs 4 --lr 0.003 \
  --train-cohort full_rescuable --eval-cohort full_rescuable \
  --random-seeds 10 \
  --output-dir output/qwen7b_real_gray_causal_8d_l27_v1 --overwrite
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as direction

RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-10


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--prompt-template", default=(
        "Determine the spatial relation of the {subject} to the {reference} "
        "in the image. Answer with left, right, above, or below."
    ))
    p.add_argument("--layer", type=int, default=27)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--pool-rank", type=int, default=64)
    p.add_argument("--train-cohort", default="full_rescuable",
                   choices=["full_rescuable", "recoverable", "real_gray_different"])
    p.add_argument("--eval-cohort", default="full_rescuable",
                   choices=["full_rescuable", "recoverable", "real_gray_different"])
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--eval-max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--reverse-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--init", default="top", choices=["top", "random"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--random-seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    rows = list(rows)
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
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0


# -----------------------------------------------------------------------------
# Metadata / model helpers
# -----------------------------------------------------------------------------

def norm_relation(x: Any) -> str:
    return direction.norm_relation(x)


def load_metadata(direction_dir: Path):
    vp = direction_dir / "vectors.npz"
    gp = direction_dir / "sample_split_and_generation.csv"
    with np.load(vp, allow_pickle=True) as z:
        sids = z["sample_index"].astype(np.int64)
        labels = [norm_relation(x) for x in z["relation"]]
    gt = {int(s): str(labels[i]) for i, s in enumerate(sids.tolist())}
    split = {}
    for row in read_csv(gp):
        split[int(row["sample_index"])] = str(row.get("split", "")).strip()
    return {"sids": [int(x) for x in sids.tolist()], "gt": gt, "split": split}


def get_attr_path(obj: Any, path: str):
    cur = obj
    for piece in path.split("."):
        cur = getattr(cur, piece)
    return cur


def resolve_decoder_layers(model):
    candidates = [
        "model.language_model.layers", "language_model.layers",
        "model.model.layers", "model.layers", "language_model.model.layers",
    ]
    for path in candidates:
        try:
            layers = get_attr_path(model, path)
            if len(layers) and hasattr(layers[0], "self_attn"):
                return layers, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers")


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            if torch.is_tensor(y):
                return y
    raise RuntimeError(f"No tensor in {type(x)}")


def infer_last_position(batch):
    if "attention_mask" in batch:
        nz = torch.nonzero(batch["attention_mask"][0], as_tuple=False).flatten()
        if len(nz):
            return int(nz[-1].item())
    return int(batch["input_ids"].shape[1] - 1)


def build_batch(processor, rec, question, image, device):
    rendered = direction.build_chat_prompt(processor, question, image is not None)
    batch = direction.process_inputs(processor, rendered, image, device)
    return batch, infer_last_position(batch)


def make_gray_image(image: Image.Image, gray_value: int):
    v = int(np.clip(gray_value, 0, 255))
    return Image.new("RGB", image.size, color=(v, v, v))


# -----------------------------------------------------------------------------
# Generation and differentiable four-relation scores
# -----------------------------------------------------------------------------

def parse_generated_relation(text: str) -> Optional[str]:
    s = str(text).strip().lower()
    pats = [
        ("left", r"\bleft\b"), ("right", r"\bright\b"),
        ("above", r"\babove\b"), ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"), ("below", r"\bunder(?:neath)?\b"),
    ]
    hits = []
    for rel, pat in pats:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def generate_answer(model, processor, batch, max_new_tokens):
    input_len = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(
            **batch, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True
        )
    text = processor.tokenizer.decode(
        out[0, input_len:], skip_special_tokens=True
    ).strip()
    pred = parse_generated_relation(text)
    del out
    return text, pred


def build_relation_token_sets(tokenizer):
    out = {}
    for rel in RELATIONS:
        ids = []
        for text in [rel, f" {rel}", rel.capitalize(), f" {rel.capitalize()}"]:
            toks = tokenizer.encode(text, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(int(toks[0]))
        ids = sorted(set(ids))
        if not ids:
            raise RuntimeError(f"No single-token variant for relation={rel}")
        out[rel] = ids
    return out


def relation_scores(logits_at_last: torch.Tensor, token_sets):
    vals = []
    for rel in RELATIONS:
        ids = torch.as_tensor(token_sets[rel], device=logits_at_last.device)
        vals.append(torch.logsumexp(logits_at_last.index_select(-1, ids).float(), dim=-1))
    return torch.stack(vals, dim=-1)


# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------

class FixedDeltaHook:
    def __init__(self, block, last_pos, delta, sign=1.0):
        self.last_pos = int(last_pos)
        self.delta = delta
        self.sign = float(sign)
        self.applied = False
        self.handle = block.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if self.applied or not args:
            return None
        vals = list(args)
        idx, x = None, None
        for i, item in enumerate(vals):
            if torch.is_tensor(item):
                idx, x = i, item
                break
        if x is None or x.ndim != 3 or self.last_pos >= int(x.shape[1]):
            return None
        d = torch.as_tensor(self.delta, device=x.device, dtype=x.dtype)
        y = x.clone()
        y[0, self.last_pos, :] = y[0, self.last_pos, :] + self.sign * d
        self.applied = True
        vals[idx] = y
        return tuple(vals)

    def close(self):
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class LastStateCapture:
    def __init__(self, block, last_pos):
        self.last_pos = int(last_pos)
        self.state = None
        self.handle = block.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if self.state is not None or not args:
            return None
        x = first_tensor(args)
        if x.ndim != 3 or self.last_pos >= int(x.shape[1]):
            return None
        self.state = x[0, self.last_pos].detach().float().cpu().numpy().astype(np.float32)
        return None

    def close(self):
        if self.handle is not None:
            with contextlib.suppress(Exception): self.handle.remove()
        self.handle = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def capture_last_state(model, block, processor, rec, image, device, prompt_template):
    q = prompt_template.format(subject=rec.subject, reference=rec.reference)
    batch, last_pos = build_batch(processor, rec, q, image, device)
    with LastStateCapture(block, last_pos) as cap:
        with torch.inference_mode():
            _ = model(**batch, use_cache=False, return_dict=True)
    if cap.state is None:
        raise RuntimeError("Failed to capture last state")
    del batch
    return cap.state


# -----------------------------------------------------------------------------
# Cache and SVD pool
# -----------------------------------------------------------------------------

def load_train_cache(path: Path, layer: int):
    key = f"L{layer}_delta_last"
    with np.load(path, allow_pickle=True) as z:
        if key not in z.files:
            raise RuntimeError(f"Cache missing {key}")
        sids = np.asarray(z["sid"], dtype=np.int64)
        deltas = np.asarray(z[key], dtype=np.float32)
        gray = int(np.asarray(z["gray_value"]).reshape(-1)[0]) if "gray_value" in z.files else None
    return {"sids": sids, "deltas": deltas, "gray_value": gray}


def fit_pool_basis(deltas: np.ndarray, pool_rank: int):
    _, S, Vh = np.linalg.svd(deltas.astype(np.float64), full_matrices=False)
    tol = 1e-8 * max(float(S.max()) if len(S) else 0.0, 1.0)
    eff = int(np.sum(S > tol))
    k = min(int(pool_rank), eff, int(Vh.shape[0]))
    if k <= 0:
        raise RuntimeError("Degenerate delta matrix")
    B = Vh[:k].T.astype(np.float32)
    energy = float(np.sum(S[:k] ** 2) / max(np.sum(S ** 2), EPS))
    return B, eff, energy


# -----------------------------------------------------------------------------
# Trainable 8D subspace
# -----------------------------------------------------------------------------

class CausalSubspace(nn.Module):
    def __init__(self, pool_basis, rank, init, seed, device):
        super().__init__()
        B = torch.as_tensor(pool_basis, dtype=torch.float32, device=device)
        self.register_buffer("B", B)
        p = int(B.shape[1])
        if rank > p:
            raise ValueError(f"rank {rank} > pool rank {p}")
        if init == "top":
            A0 = torch.zeros(p, rank, device=device)
            A0[:rank, :] = torch.eye(rank, device=device)
        else:
            gen = torch.Generator(device=device); gen.manual_seed(seed)
            R = torch.randn(p, rank, generator=gen, device=device)
            A0, _ = torch.linalg.qr(R, mode="reduced")
        self.A = nn.Parameter(A0.float())

    def basis(self):
        return self.B @ self.A

    def project(self, delta):
        W = self.basis()
        return W @ (W.T @ delta.float())

    @torch.no_grad()
    def reorthonormalize_(self):
        Q, _ = torch.linalg.qr(self.A.data, mode="reduced")
        self.A.data.copy_(Q)

    @torch.no_grad()
    def export_basis(self):
        return self.basis().detach().float().cpu().numpy().astype(np.float32)


class LearnedHook:
    def __init__(self, block, last_pos, delta_np, subspace, sign):
        self.last_pos = int(last_pos)
        self.delta_np = np.asarray(delta_np, dtype=np.float32)
        self.subspace = subspace
        self.sign = float(sign)
        self.applied = False
        self.handle = block.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if self.applied or not args:
            return None
        vals = list(args)
        idx, x = None, None
        for i, item in enumerate(vals):
            if torch.is_tensor(item): idx, x = i, item; break
        if x is None or x.ndim != 3 or self.last_pos >= int(x.shape[1]):
            return None
        delta = torch.as_tensor(self.delta_np, device=x.device, dtype=torch.float32)
        d = self.subspace.project(delta)
        y = x.clone()
        y[0, self.last_pos, :] = y[0, self.last_pos, :] + self.sign * d.to(y.dtype)
        self.applied = True
        vals[idx] = y
        return tuple(vals)

    def close(self):
        if self.handle is not None:
            with contextlib.suppress(Exception): self.handle.remove()
        self.handle = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# -----------------------------------------------------------------------------
# Fresh cohort discovery
# -----------------------------------------------------------------------------

def cohort_match(info, cohort):
    if cohort == "full_rescuable": return bool(info["full_rescuable"])
    if cohort == "recoverable": return bool(info["recoverable"])
    if cohort == "real_gray_different": return bool(info["real_gray_different"])
    raise ValueError(cohort)


def discover_train_cohort(model, processor, block, records, metadata, cache, args, device, out_dir):
    sid_to_idx = {int(s): i for i, s in enumerate(cache["sids"].tolist())}
    candidates = [
        int(s) for s in cache["sids"].tolist()
        if metadata["split"].get(int(s), "") == "train" and int(s) in records
    ]
    rows, selected, behavior = [], [], {}

    for sid in tqdm(candidates, desc="discover fresh TRAIN full-rescuable cohort"):
        rec, gt = records[sid], metadata["gt"][sid]
        delta = cache["deltas"][sid_to_idx[sid]]
        real_img = gray_img = None
        try:
            real_img = Image.open(rec.image_path).convert("RGB")
            gray_img = make_gray_image(real_img, args.gray_value)
            q = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
            rb, rl = build_batch(processor, rec, q, real_img, device)
            gb, gl = build_batch(processor, rec, q, gray_img, device)
            if rl != gl: raise RuntimeError("last_pos mismatch")
            _, rp = generate_answer(model, processor, rb, args.max_new_tokens)
            _, gp = generate_answer(model, processor, gb, args.max_new_tokens)
            with FixedDeltaHook(block, gl, delta, +1.0):
                _, fp = generate_answer(model, processor, gb, args.max_new_tokens)
            info = {
                "real_pred": rp, "gray_pred": gp, "full_pred": fp,
                "real_gray_different": int(rp in REL2ID and gp in REL2ID and rp != gp),
                "recoverable": int(rp == gt and gp in REL2ID and gp != gt),
                "full_rescuable": int(rp == gt and gp in REL2ID and gp != gt and fp == gt),
            }
            keep = cohort_match(info, args.train_cohort)
            behavior[sid] = info
            rows.append({"sid": sid, "gt": gt, **info, "selected": int(keep)})
            if keep: selected.append(sid)
            del rb, gb
        except Exception as e:
            rows.append({"sid": sid, "gt": gt, "selected": 0,
                         "error": f"{type(e).__name__}: {e}"})
        finally:
            if real_img is not None: real_img.close()
            if gray_img is not None: gray_img.close()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if args.train_max_samples is not None and len(selected) > args.train_max_samples:
        rng = random.Random(args.seed); rng.shuffle(selected)
        selected = selected[:args.train_max_samples]
        keep_set = set(selected)
        for row in rows:
            if int(row.get("selected", 0)):
                row["selected"] = int(int(row["sid"]) in keep_set)

    write_csv(out_dir / "train_behavior_cohort.csv", rows)
    if not selected:
        raise RuntimeError(f"No TRAIN samples in cohort={args.train_cohort}")
    return sorted(selected), behavior


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_subspace(model, processor, block, subspace, records, cache, train_sids,
                   behavior, token_sets, args, device, out_dir):
    sid_to_idx = {int(s): i for i, s in enumerate(cache["sids"].tolist())}
    for p in model.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam([subspace.A], lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    logs = []

    for epoch in range(1, args.epochs + 1):
        order = list(train_sids); rng.shuffle(order)
        losses = []; f_acc = []; r_acc = []

        for sid in tqdm(order, desc=f"train 8D epoch {epoch}/{args.epochs}"):
            rec = records[sid]
            delta = cache["deltas"][sid_to_idx[sid]]
            real_target = behavior[sid]["real_pred"]
            gray_target = behavior[sid]["gray_pred"]
            if real_target not in REL2ID or gray_target not in REL2ID:
                continue

            real_img = gray_img = None
            try:
                real_img = Image.open(rec.image_path).convert("RGB")
                gray_img = make_gray_image(real_img, args.gray_value)
                q = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
                rb, rl = build_batch(processor, rec, q, real_img, device)
                gb, gl = build_batch(processor, rec, q, gray_img, device)
                if rl != gl: raise RuntimeError("last_pos mismatch")

                opt.zero_grad(set_to_none=True)

                with LearnedHook(block, gl, delta, subspace, +1.0):
                    fo = model(**gb, use_cache=False, return_dict=True)
                fs = relation_scores(fo.logits[0, gl, :], token_sets).unsqueeze(0)
                ft = torch.tensor([REL2ID[real_target]], device=fs.device)
                lf = F.cross_entropy(fs, ft)

                with LearnedHook(block, rl, delta, subspace, -1.0):
                    ro = model(**rb, use_cache=False, return_dict=True)
                rs = relation_scores(ro.logits[0, rl, :], token_sets).unsqueeze(0)
                rt = torch.tensor([REL2ID[gray_target]], device=rs.device)
                lr = F.cross_entropy(rs, rt)

                loss = lf + args.reverse_weight * lr
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([subspace.A], args.grad_clip)
                opt.step(); subspace.reorthonormalize_()

                losses.append(float(loss.detach().cpu()))
                f_acc.append(int(torch.argmax(fs, -1).item() == REL2ID[real_target]))
                r_acc.append(int(torch.argmax(rs, -1).item() == REL2ID[gray_target]))
                del rb, gb, fo, ro, fs, rs, lf, lr, loss
            finally:
                if real_img is not None: real_img.close()
                if gray_img is not None: gray_img.close()
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        row = {
            "epoch": epoch, "n": len(losses), "mean_loss": safe_mean(losses),
            "forward_relation_acc": safe_mean(f_acc),
            "reverse_relation_acc": safe_mean(r_acc),
        }
        logs.append(row)
        print(f"[epoch {epoch}] loss={row['mean_loss']:.4f} "
              f"fwd={row['forward_relation_acc']:.3f} rev={row['reverse_relation_acc']:.3f}")
        np.save(out_dir / f"learned_basis_epoch{epoch}.npy", subspace.export_basis())
        write_csv(out_dir / "train_log.csv", logs)
    return logs


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------

def project_np(delta, basis):
    x = np.asarray(delta, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    return (B @ (B.T @ x)).astype(np.float32)


def random_basis_in_pool(pool_basis, rank, seed):
    B = np.asarray(pool_basis, dtype=np.float64)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((B.shape[1], rank))
    Q, _ = np.linalg.qr(R, mode="reduced")
    return (B @ Q[:, :rank]).astype(np.float32)


def norm_match(delta, target_norm, basis, seed):
    x = np.asarray(delta, dtype=np.float32)
    if target_norm <= EPS: return np.zeros_like(x)
    n = float(np.linalg.norm(x))
    if n > EPS: return (x * (target_norm / n)).astype(np.float32)
    rng = np.random.default_rng(seed)
    v = basis @ rng.standard_normal(basis.shape[1])
    vn = float(np.linalg.norm(v))
    return (v / vn * target_norm).astype(np.float32) if vn > EPS else np.zeros_like(x)


def gen_with_delta(model, processor, block, batch, last_pos, delta, sign, max_new_tokens):
    with FixedDeltaHook(block, last_pos, delta, sign):
        return generate_answer(model, processor, batch, max_new_tokens)


# -----------------------------------------------------------------------------
# Actual model.generate() TEST evaluation
# -----------------------------------------------------------------------------

def evaluate_test(model, processor, block, records, metadata, learned_basis, top8_basis,
                  pool_basis, args, device, out_dir):
    sids = [
        sid for sid in metadata["sids"]
        if metadata["split"].get(sid, "") == "test" and sid in records
        and metadata["gt"].get(sid, "") in REL2ID
    ]
    if args.eval_max_samples is not None and len(sids) > args.eval_max_samples:
        rng = random.Random(args.seed); rng.shuffle(sids); sids = sids[:args.eval_max_samples]

    random_bases = [
        random_basis_in_pool(pool_basis, args.rank, args.seed + 1000003 * k + 17)
        for k in range(args.random_seeds)
    ]
    baselines, rows, errors = [], [], []

    for sid in tqdm(sorted(sids), desc="TEST actual generate learned8"):
        rec, gt = records[sid], metadata["gt"][sid]
        real_img = gray_img = None
        try:
            real_img = Image.open(rec.image_path).convert("RGB")
            gray_img = make_gray_image(real_img, args.gray_value)
            q = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
            rb, rl = build_batch(processor, rec, q, real_img, device)
            gb, gl = build_batch(processor, rec, q, gray_img, device)
            if rl != gl: raise RuntimeError("last_pos mismatch")

            _, rp = generate_answer(model, processor, rb, args.max_new_tokens)
            _, gp = generate_answer(model, processor, gb, args.max_new_tokens)

            hr = capture_last_state(model, block, processor, rec, real_img, device, args.prompt_template)
            hg = capture_last_state(model, block, processor, rec, gray_img, device, args.prompt_template)
            full_delta = (hr - hg).astype(np.float32)
            _, fp = gen_with_delta(model, processor, block, gb, gl, full_delta, +1.0, args.max_new_tokens)

            info = {
                "real_gray_different": int(rp in REL2ID and gp in REL2ID and rp != gp),
                "recoverable": int(rp == gt and gp in REL2ID and gp != gt),
                "full_rescuable": int(rp == gt and gp in REL2ID and gp != gt and fp == gt),
            }
            keep = cohort_match(info, args.eval_cohort)
            baselines.append({
                "sid": sid, "gt": gt, "real_pred": rp or "", "gray_pred": gp or "",
                "full_pred": fp or "", **info, "in_eval_cohort": int(keep),
            })
            if not keep:
                del rb, gb, hr, hg, full_delta
                continue

            ld = project_np(full_delta, learned_basis)
            td = project_np(full_delta, top8_basis)
            cd = (full_delta - ld).astype(np.float32)

            _, lfp = gen_with_delta(model, processor, block, gb, gl, ld, +1.0, args.max_new_tokens)
            _, lrp = gen_with_delta(model, processor, block, rb, rl, ld, -1.0, args.max_new_tokens)
            _, tfp = gen_with_delta(model, processor, block, gb, gl, td, +1.0, args.max_new_tokens)
            _, trp = gen_with_delta(model, processor, block, rb, rl, td, -1.0, args.max_new_tokens)
            _, cfp = gen_with_delta(model, processor, block, gb, gl, cd, +1.0, args.max_new_tokens)

            def add(mode, seed, fpred, rpred, edit):
                rows.append({
                    "sid": sid, "mode": mode, "random_seed": seed,
                    "gt": gt, "real_pred": rp or "", "gray_pred": gp or "",
                    "forward_pred": fpred or "", "reverse_pred": rpred or "",
                    "forward_to_real": int(fpred == rp),
                    "forward_to_gt": int(fpred == gt),
                    "reverse_to_gray": int(rpred == gp) if rpred is not None else "",
                    "bidirectional": int(fpred == rp and rpred == gp) if rpred is not None else "",
                    "edit_norm": float(np.linalg.norm(edit)),
                    "full_delta_norm": float(np.linalg.norm(full_delta)),
                })

            add("learned8", "", lfp, lrp, ld)
            add("top8", "", tfp, trp, td)
            add("complement", "", cfp, None, cd)

            lnorm = float(np.linalg.norm(ld))
            for rseed, rbasis in enumerate(random_bases):
                raw = project_np(full_delta, rbasis)
                rd = norm_match(raw, lnorm, rbasis, args.seed + sid * 100019 + rseed)
                _, rfp = gen_with_delta(model, processor, block, gb, gl, rd, +1.0, args.max_new_tokens)
                _, rrp = gen_with_delta(model, processor, block, rb, rl, rd, -1.0, args.max_new_tokens)
                add("random8", rseed, rfp, rrp, rd)

            del rb, gb, hr, hg, full_delta, ld, td, cd
        except Exception as e:
            errors.append({"sid": sid, "error_type": type(e).__name__, "error": str(e)})
            tqdm.write(f"[TEST ERROR sid={sid}] {type(e).__name__}: {e}")
        finally:
            if real_img is not None: real_img.close()
            if gray_img is not None: gray_img.close()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    write_csv(out_dir / "test_baseline_and_cohort.csv", baselines)
    write_csv(out_dir / "test_interventions.csv", rows)
    write_csv(out_dir / "test_errors.csv", errors)
    return baselines, rows, errors


def summarize_test(baselines, rows):
    summary = []
    for mode in ["learned8", "top8", "complement"]:
        rr = [r for r in rows if r["mode"] == mode]
        if not rr: continue
        summary.append({
            "mode": mode, "n": len(rr),
            "forward_to_real_rate": safe_mean(r["forward_to_real"] for r in rr),
            "forward_to_gt_rate": safe_mean(r["forward_to_gt"] for r in rr),
            "reverse_to_gray_rate": safe_mean(r["reverse_to_gray"] for r in rr if r["reverse_to_gray"] != ""),
            "bidirectional_rate": safe_mean(r["bidirectional"] for r in rr if r["bidirectional"] != ""),
            "mean_edit_norm": safe_mean(r["edit_norm"] for r in rr),
        })

    seed_rows = []
    seeds = sorted(set(int(r["random_seed"]) for r in rows if r["mode"] == "random8"))
    for seed in seeds:
        rr = [r for r in rows if r["mode"] == "random8" and int(r["random_seed"]) == seed]
        seed_rows.append({
            "random_seed": seed, "n": len(rr),
            "forward_to_real_rate": safe_mean(r["forward_to_real"] for r in rr),
            "reverse_to_gray_rate": safe_mean(r["reverse_to_gray"] for r in rr),
            "bidirectional_rate": safe_mean(r["bidirectional"] for r in rr),
        })
    if seed_rows:
        summary.append({
            "mode": "random8_mean", "n": int(safe_mean(r["n"] for r in seed_rows)),
            "forward_to_real_rate": safe_mean(r["forward_to_real_rate"] for r in seed_rows),
            "forward_to_real_std": safe_std(r["forward_to_real_rate"] for r in seed_rows),
            "reverse_to_gray_rate": safe_mean(r["reverse_to_gray_rate"] for r in seed_rows),
            "reverse_to_gray_std": safe_std(r["reverse_to_gray_rate"] for r in seed_rows),
            "bidirectional_rate": safe_mean(r["bidirectional_rate"] for r in seed_rows),
            "bidirectional_std": safe_std(r["bidirectional_rate"] for r in seed_rows),
            "mean_edit_norm": "",
        })

    cohort = [{
        "n_test_candidates": len(baselines),
        "n_fresh_recoverable": sum(int(r["recoverable"]) for r in baselines),
        "n_fresh_full_rescuable": sum(int(r["full_rescuable"]) for r in baselines),
        "n_eval_cohort": sum(int(r["in_eval_cohort"]) for r in baselines),
    }]
    return cohort, summary, seed_rows


def print_summary(cohort, summary):
    print("\n" + "=" * 130)
    print("LEARNED 8D REAL<->GRAY CAUSAL SUBSPACE — ACTUAL model.generate()")
    print("=" * 130)
    if cohort:
        c = cohort[0]
        print(f"test={c['n_test_candidates']} | recoverable={c['n_fresh_recoverable']} | "
              f"full-rescuable={c['n_fresh_full_rescuable']} | eval N={c['n_eval_cohort']}")
    print("mode | Gray->Real | Real->Gray | bidirectional | mean edit norm")
    for r in summary:
        print(f"{r['mode']:14s} | {float(r['forward_to_real_rate']):.3f} | "
              f"{float(r['reverse_to_gray_rate']):.3f} | {float(r['bidirectional_rate']):.3f} | "
              f"{r.get('mean_edit_norm','')}")
        if r["mode"] == "random8_mean":
            print(f"  random std: fwd={r['forward_to_real_std']:.3f} "
                  f"rev={r['reverse_to_gray_std']:.3f} bi={r['bidirectional_std']:.3f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.rank <= 0 or args.pool_rank < args.rank:
        raise ValueError("Require 0 < rank <= pool-rank")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(Path(args.direction_dir))
    records_list, _ = base.load_records(args.dataset, Path(args.data_root), None)
    records = {int(r.sid): r for r in records_list}

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    dtype = base.resolve_dtype(spec.dtype_name)
    kw = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none": kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id} on {args.device}")
    try:
        model = cls.from_pretrained(spec.repo_id, dtype=dtype, **kw)
    except TypeError:
        model = cls.from_pretrained(spec.repo_id, torch_dtype=dtype, **kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    base.configure_processor(model, processor)

    layers, layer_path = resolve_decoder_layers(model)
    if not (0 <= args.layer < len(layers)):
        raise ValueError(f"Invalid layer {args.layer}; model has {len(layers)}")
    block = layers[args.layer]
    device = torch.device(args.device)
    print(f"[site] {layer_path}[{args.layer}] last prompt token")

    cache = load_train_cache(Path(args.train_cache), args.layer)
    if cache["gray_value"] is not None and cache["gray_value"] != args.gray_value:
        print(f"[warning] cache gray={cache['gray_value']} vs requested gray={args.gray_value}")

    pool_basis, eff_rank, pool_energy = fit_pool_basis(cache["deltas"], args.pool_rank)
    print(f"[pool] Ntrain={len(cache['sids'])}, hidden={cache['deltas'].shape[1]}, "
          f"effective_rank={eff_rank}, pool={pool_basis.shape[1]}, energy={pool_energy:.4f}")
    if args.rank > pool_basis.shape[1]:
        raise RuntimeError("Requested rank exceeds actual pool rank")

    token_sets = build_relation_token_sets(processor.tokenizer)
    print("[relation tokens] " + ", ".join(f"{r}:{token_sets[r]}" for r in RELATIONS))

    train_sids, behavior = discover_train_cohort(
        model, processor, block, records, metadata, cache, args, device, out_dir
    )
    print(f"[TRAIN cohort] {args.train_cohort}: N={len(train_sids)}")

    subspace = CausalSubspace(
        pool_basis, args.rank, args.init, args.seed, device
    )
    np.save(out_dir / "initial_basis.npy", subspace.export_basis())

    train_subspace(
        model, processor, block, subspace, records, cache, train_sids,
        behavior, token_sets, args, device, out_dir
    )

    learned_basis = subspace.export_basis()
    top8_basis = pool_basis[:, :args.rank].astype(np.float32)
    np.save(out_dir / "learned_basis.npy", learned_basis)
    np.save(out_dir / "top8_variance_basis.npy", top8_basis)

    baselines, results, errors = evaluate_test(
        model, processor, block, records, metadata, learned_basis,
        top8_basis, pool_basis, args, device, out_dir
    )
    cohort_summary, result_summary, seed_rows = summarize_test(baselines, results)
    write_csv(out_dir / "test_cohort_summary.csv", cohort_summary)
    write_csv(out_dir / "test_summary.csv", result_summary)
    write_csv(out_dir / "random_seed_summary.csv", seed_rows)
    print_summary(cohort_summary, result_summary)

    (out_dir / "summary.json").write_text(json.dumps({
        "experiment": "learned paired Real-vs-Gray causal subspace",
        "layer": args.layer,
        "rank": args.rank,
        "pool_rank": int(pool_basis.shape[1]),
        "pool_train_energy": pool_energy,
        "train_cohort": args.train_cohort,
        "eval_cohort": args.eval_cohort,
        "n_train": len(train_sids),
        "epochs": args.epochs,
        "lr": args.lr,
        "model_frozen": True,
        "gradient_used_for_subspace_search": True,
        "spatial_labels_used_to_define_subspace": False,
        "final_metric": "fresh actual model.generate() Real<->Gray behavior",
        "n_test_errors": len(errors),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved:")
    for name in [
        "train_behavior_cohort.csv", "train_log.csv", "initial_basis.npy",
        "learned_basis.npy", "top8_variance_basis.npy",
        "test_baseline_and_cohort.csv", "test_interventions.csv",
        "test_cohort_summary.csv", "test_summary.csv", "random_seed_summary.csv",
        "test_errors.csv", "summary.json",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
