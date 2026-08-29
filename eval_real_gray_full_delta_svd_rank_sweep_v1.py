#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-vs-Gray Full-Delta SVD Rank Sweep
======================================

Question
--------
How many shared linear dimensions of the SAME-SAMPLE Real-Gray last-token
activation difference are needed to preserve its generation-level causal effect?

At decoder layer l:
    delta_i,l = h_last,l(real) - h_last,l(gray)

On TRAIN samples, stack raw deltas and use an UNCENTERED SVD:
    D_l = U Sigma V^T
The top-k right singular vectors define a shared linear subspace S_l,k.

For each TEST sample:
    full_delta = h_real - h_gray
    pca_delta  = P_k full_delta
    rest_delta = full_delta - pca_delta

Interventions (fresh model.generate()):
  full_restore:       gray + full_delta
  pca_restore:        gray + pca_delta
  complement_restore: gray + rest_delta
  random_restore:     same-rank random subspace, edit norm matched to pca_delta
  necessity:          real - pca_delta

This script can reuse train_real_gray_last_delta.npz produced by
`eval_real_gray_causal_spatial_subspace_v1.py`.

Recommended dimensionality scan:

CUDA_VISIBLE_DEVICES=0 python eval_real_gray_full_delta_svd_rank_sweep_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --train-cache output/qwen7b_real_gray_causal_spatial_subspace_v1/train_real_gray_last_delta.npz \
  --dataset coco_two --data-root data --model qwen-7b --device cuda:0 \
  --layers 25,26,27 --ranks 1,2,4,8,16,32,64,96,128 \
  --cohort recoverable --modes full,pca \
  --output-dir output/qwen7b_real_gray_full_delta_svd_rank_sweep_v1 --overwrite

Then validate promising ranks with random/complement controls, e.g.:

CUDA_VISIBLE_DEVICES=0 python eval_real_gray_full_delta_svd_rank_sweep_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --train-cache output/qwen7b_real_gray_causal_spatial_subspace_v1/train_real_gray_last_delta.npz \
  --dataset coco_two --data-root data --model qwen-7b --device cuda:0 \
  --layers 27 --ranks 16,32,64 --cohort recoverable \
  --modes full,pca,complement,random --random-seeds 20 \
  --output-dir output/qwen7b_real_gray_full_delta_svd_validate_v1 --overwrite
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as direction

RELATIONS = ("left", "right", "above", "below")
EPS = 1e-10


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--train-cache", default="")
    p.add_argument("--rebuild-train-cache", action="store_true")
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
    p.add_argument("--layers", default="25,26,27")
    p.add_argument("--ranks", default="1,2,4,8,16,32,64,96,128")
    p.add_argument("--modes", default="full,pca",
                   help="subset of full,pca,complement,random,necessity")
    p.add_argument("--cohort", default="recoverable",
                   choices=["recoverable", "real_correct", "gray_wrong", "all"])
    p.add_argument("--train-controls", default="all", choices=["all", "correct"])
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--random-seeds", type=int, default=10)
    p.add_argument("--random-orthogonal-to-top", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def parse_words(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_ints(s: str) -> List[int]:
    return sorted(set(int(x) for x in parse_words(s)))


def parse_layers(s: str, n_layers: int) -> List[int]:
    if s.strip().lower() == "all":
        return list(range(n_layers))
    out = []
    for x in parse_words(s):
        if "-" in x:
            a, b = map(int, x.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(x))
    out = sorted(set(out))
    bad = [x for x in out if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Invalid layers {bad}; valid 0..{n_layers-1}")
    return out


def norm_relation(x: Any) -> str:
    return direction.norm_relation(x)


def load_metadata(direction_dir: Path):
    vp = direction_dir / "vectors.npz"
    gp = direction_dir / "sample_split_and_generation.csv"
    if not vp.exists():
        raise FileNotFoundError(vp)
    if not gp.exists():
        raise FileNotFoundError(gp)
    with np.load(vp, allow_pickle=True) as z:
        sids = z["sample_index"].astype(np.int64)
        labels = [norm_relation(x) for x in z["relation"]]
    gt = {int(sid): labels[i] for i, sid in enumerate(sids.tolist())}
    split, generation = {}, {}
    for r in read_csv(gp):
        sid = int(r["sample_index"])
        split[sid] = str(r.get("split", "")).strip()
        pred = norm_relation(r.get("generation_pred", ""))
        group = str(r.get("generation_group", "")).strip().lower()
        if group not in ("correct", "wrong"):
            g = gt.get(sid, "")
            if g in RELATIONS and pred in RELATIONS:
                group = "correct" if pred == g else "wrong"
        generation[sid] = {
            "generation_group": group,
            "generation_pred": pred,
            "generation_text": str(r.get("generation_text", "")),
        }
    return {
        "sids": [int(x) for x in sids.tolist()],
        "gt": gt,
        "split": split,
        "generation": generation,
    }


def get_attr_path(obj: Any, path: str):
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def resolve_decoder_layers(model):
    candidates = [
        "model.language_model.layers", "language_model.layers",
        "model.model.layers", "model.layers", "language_model.model.layers",
    ]
    for path in candidates:
        try:
            layers = get_attr_path(model, path)
            if len(layers) > 0 and hasattr(layers[0], "self_attn"):
                return layers, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers")


def first_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            if torch.is_tensor(y):
                return y
    raise RuntimeError(f"No tensor in {type(x)}")


def infer_last_position(batch) -> int:
    if "attention_mask" in batch:
        nz = torch.nonzero(batch["attention_mask"][0], as_tuple=False).flatten()
        if len(nz):
            return int(nz[-1].item())
    return int(batch["input_ids"].shape[1] - 1)


def build_batch(processor, rec, question, image, device):
    rendered = direction.build_chat_prompt(processor, question, image is not None)
    batch = direction.process_inputs(processor, rendered, image, device)
    return batch, infer_last_position(batch)


def make_gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(np.clip(value, 0, 255))
    return Image.new("RGB", image.size, color=(v, v, v))


def parse_generated_relation(text: str) -> Optional[str]:
    s = str(text).strip().lower()
    patterns = [
        ("left", r"\bleft\b"), ("right", r"\bright\b"),
        ("above", r"\babove\b"), ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"), ("below", r"\bunder(?:neath)?\b"),
    ]
    hits = []
    for rel, pat in patterns:
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
        g = model.generate(
            **batch, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True
        )
    text = processor.tokenizer.decode(
        g[0, input_len:], skip_special_tokens=True
    ).strip()
    pred = parse_generated_relation(text)
    del g
    return text, pred


class LastStateCapture:
    def __init__(self, decoder_layers, selected_layers, last_pos):
        self.last_pos = int(last_pos)
        self.states = {}
        self.handles = []
        for li in selected_layers:
            self.handles.append(
                decoder_layers[li].register_forward_pre_hook(self._make_hook(li))
            )

    def _make_hook(self, li):
        def hook(_module, args):
            if not args:
                return None
            x = first_tensor(args)
            if x.ndim != 3 or self.last_pos >= int(x.shape[1]):
                return None
            self.states[li] = (
                x[0, self.last_pos].detach().float().cpu().numpy().astype(np.float32)
            )
            return None
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


def capture_last_states(
    *, model, processor, decoder_layers, selected_layers, rec, image, device,
    prompt_template
):
    question = prompt_template.format(subject=rec.subject, reference=rec.reference)
    batch, last_pos = build_batch(processor, rec, question, image, device)
    with LastStateCapture(decoder_layers, selected_layers, last_pos) as cap:
        with torch.inference_mode():
            _ = model(
                **batch, output_attentions=False, output_hidden_states=False,
                use_cache=False, return_dict=True
            )
    missing = [li for li in selected_layers if li not in cap.states]
    if missing:
        raise RuntimeError(f"Missing last-token states at {missing}")
    del batch
    return {"states": dict(cap.states), "last_pos": last_pos}


def select_train_sids(metadata, records, controls, max_samples, seed):
    sids = []
    for sid in metadata["sids"]:
        if metadata["split"].get(sid, "") != "train" or sid not in records:
            continue
        if metadata["gt"].get(sid, "") not in RELATIONS:
            continue
        if controls == "correct":
            if metadata["generation"].get(sid, {}).get("generation_group", "") != "correct":
                continue
        sids.append(sid)
    if max_samples is not None and len(sids) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(sids)
        sids = sids[:max_samples]
    return sorted(sids)


def collect_train_cache(
    *, cache_path, model, processor, decoder_layers, selected_layers, records,
    metadata, device, prompt_template, gray_value, train_controls, max_samples,
    seed
):
    sids = select_train_sids(
        metadata, records, train_controls, max_samples, seed
    )
    if not sids:
        raise RuntimeError("No TRAIN samples selected")
    vals = {li: [] for li in selected_layers}
    kept_sid, kept_rel, errors = [], [], []
    for sid in tqdm(sids, desc="collect TRAIN Real-Gray deltas"):
        rec = records[sid]
        real_img = gray_img = None
        try:
            real_img = Image.open(rec.image_path).convert("RGB")
            gray_img = make_gray_image(real_img, gray_value)
            real = capture_last_states(
                model=model, processor=processor, decoder_layers=decoder_layers,
                selected_layers=selected_layers, rec=rec, image=real_img,
                device=device, prompt_template=prompt_template
            )
            gray = capture_last_states(
                model=model, processor=processor, decoder_layers=decoder_layers,
                selected_layers=selected_layers, rec=rec, image=gray_img,
                device=device, prompt_template=prompt_template
            )
            for li in selected_layers:
                vals[li].append(
                    (real["states"][li] - gray["states"][li]).astype(np.float32)
                )
            kept_sid.append(sid)
            kept_rel.append(metadata["gt"][sid])
        except Exception as e:
            errors.append({"sid": sid, "error_type": type(e).__name__, "error": str(e)})
            tqdm.write(f"[TRAIN ERROR sid={sid}] {type(e).__name__}: {e}")
        finally:
            if real_img is not None:
                real_img.close()
            if gray_img is not None:
                gray_img.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not kept_sid:
        raise RuntimeError("TRAIN cache collection produced zero samples")
    payload = {
        "sid": np.asarray(kept_sid, dtype=np.int64),
        "relation": np.asarray(kept_rel, dtype=object),
        "selected_layers": np.asarray(selected_layers, dtype=np.int64),
        "gray_value": np.asarray([gray_value], dtype=np.int64),
        "train_controls": np.asarray([train_controls], dtype=object),
    }
    for li in selected_layers:
        payload[f"L{li}_delta_last"] = np.stack(vals[li], axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    write_csv(cache_path.with_suffix(".errors.csv"), errors)
    print(f"[train-cache] saved {cache_path}; N={len(kept_sid)}")


def validate_train_cache(cache_path: Path, selected_layers):
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=True) as z:
        missing = [li for li in selected_layers if f"L{li}_delta_last" not in z.files]
    if missing:
        raise RuntimeError(
            f"TRAIN cache missing layers {missing}; rebuild cache with those layers"
        )


def fit_svd(cache_path: Path, selected_layers, requested_ranks):
    models, rows = {}, []
    with np.load(cache_path, allow_pickle=True) as z:
        for li in selected_layers:
            X = np.asarray(z[f"L{li}_delta_last"], dtype=np.float64)
            if X.ndim != 2:
                raise RuntimeError(f"L{li}: expected 2D matrix, got {X.shape}")
            # Uncentered SVD: we want a linear subspace of the actual delta.
            _, S, Vh = np.linalg.svd(X, full_matrices=False)
            tol = 1e-8 * max(float(S.max()) if len(S) else 0.0, 1.0)
            eff = int(np.sum(S > tol))
            energy = float(np.sum(S ** 2))
            models[li] = {
                "S": S.astype(np.float32),
                "Vh": Vh.astype(np.float32),
                "effective_rank": eff,
                "n_train": int(X.shape[0]),
                "hidden_dim": int(X.shape[1]),
            }
            for rk in requested_ranks:
                actual = min(rk, eff, Vh.shape[0])
                captured = float(np.sum(S[:actual] ** 2))
                rows.append({
                    "layer": li,
                    "requested_rank": rk,
                    "actual_rank": actual,
                    "n_train": int(X.shape[0]),
                    "hidden_dim": int(X.shape[1]),
                    "effective_rank": eff,
                    "captured_train_energy": captured / energy if energy > EPS else 0.0,
                    "sv_first": float(S[0]) if len(S) else 0.0,
                    "sv_at_rank": float(S[actual-1]) if actual > 0 else 0.0,
                })
    return models, rows


def get_basis(info, requested_rank):
    k = min(requested_rank, info["effective_rank"], info["Vh"].shape[0])
    if k <= 0:
        return np.zeros((info["hidden_dim"], 0), dtype=np.float32)
    return info["Vh"][:k].T.astype(np.float32)


def project(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.ndim != 2 or B.shape[1] == 0:
        return np.zeros_like(v, dtype=np.float32)
    x = np.asarray(v, dtype=np.float64)
    Q = np.asarray(B, dtype=np.float64)
    return (Q @ (Q.T @ x)).astype(np.float32)


def make_random_basis(dim, rank, seed, orthogonal_to=None):
    if rank <= 0:
        return np.zeros((dim, 0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, rank))
    if orthogonal_to is not None and orthogonal_to.shape[1] > 0:
        Q0 = np.asarray(orthogonal_to, dtype=np.float64)
        A = A - Q0 @ (Q0.T @ A)
    Q, R = np.linalg.qr(A, mode="reduced")
    if Q.shape[1] < rank or np.any(np.abs(np.diag(R)[:rank]) < 1e-10):
        raise RuntimeError("Degenerate random basis")
    return Q[:, :rank].astype(np.float32)


def norm_match(v, target_norm, basis, seed):
    x = np.asarray(v, dtype=np.float32)
    if target_norm <= EPS:
        return np.zeros_like(x)
    n = float(np.linalg.norm(x))
    if n > EPS:
        return (x * (target_norm / n)).astype(np.float32)
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(basis.shape[1])
    y = np.asarray(basis, dtype=np.float64) @ c
    yn = float(np.linalg.norm(y))
    if yn <= EPS:
        return np.zeros_like(x)
    return (y / yn * target_norm).astype(np.float32)


class LastTokenAddHook:
    def __init__(self, block, last_pos, delta):
        self.block = block
        self.last_pos = int(last_pos)
        self.delta = np.asarray(delta, dtype=np.float32)
        self.applied = False
        self.handle = None

    def _hook(self, _module, args):
        if self.applied or not args:
            return None
        vals = list(args)
        idx = None
        x = None
        for i, item in enumerate(vals):
            if torch.is_tensor(item):
                idx, x = i, item
                break
        if x is None or x.ndim != 3 or self.last_pos >= int(x.shape[1]):
            return None
        y = x.clone()
        if float(np.linalg.norm(self.delta)) > EPS:
            d = torch.from_numpy(self.delta).to(device=y.device, dtype=y.dtype)
            y[0, self.last_pos, :] = y[0, self.last_pos, :] + d
        self.applied = True
        vals[idx] = y
        return tuple(vals)

    def __enter__(self):
        self.handle = self.block.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


def select_test_sids(metadata, records, max_samples, seed):
    sids = [
        sid for sid in metadata["sids"]
        if metadata["split"].get(sid, "") == "test"
        and sid in records
        and metadata["gt"].get(sid, "") in RELATIONS
    ]
    if max_samples is not None and len(sids) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(sids)
        sids = sids[:max_samples]
    return sorted(sids)


def cohort_match(cohort, real_correct, gray_correct):
    if cohort == "recoverable":
        return bool(real_correct and not gray_correct)
    if cohort == "real_correct":
        return bool(real_correct)
    if cohort == "gray_wrong":
        return bool(not gray_correct)
    if cohort == "all":
        return True
    raise ValueError(cohort)


def run_eval(
    *, model, processor, decoder_layers, selected_layers, requested_ranks,
    svd_models, records, metadata, eval_sids, args, out_dir
):
    baselines, patches, errors = [], [], []
    random_cache = {}
    max_rank = max(requested_ranks)
    top_for_random = {
        li: get_basis(svd_models[li], max_rank) for li in selected_layers
    } if args.random_orthogonal_to_top else {}

    for sample_i, sid in enumerate(tqdm(eval_sids, desc="SVD rank sweep"), 1):
        rec = records[sid]
        real_img = gray_img = None
        try:
            gt = metadata["gt"][sid]
            real_img = Image.open(rec.image_path).convert("RGB")
            gray_img = make_gray_image(real_img, args.gray_value)
            question = args.prompt_template.format(
                subject=rec.subject, reference=rec.reference
            )
            real_batch, real_last = build_batch(
                processor, rec, question, real_img, torch.device(args.device)
            )
            gray_batch, gray_last = build_batch(
                processor, rec, question, gray_img, torch.device(args.device)
            )
            if real_last != gray_last:
                raise RuntimeError(f"real/gray last position mismatch {real_last}/{gray_last}")
            last_pos = real_last

            real_text, real_pred = generate_answer(
                model, processor, real_batch, args.max_new_tokens
            )
            gray_text, gray_pred = generate_answer(
                model, processor, gray_batch, args.max_new_tokens
            )
            rc = int(real_pred == gt)
            gcorr = int(gray_pred == gt)
            in_cohort = cohort_match(args.cohort, bool(rc), bool(gcorr))
            baselines.append({
                "sid": sid, "gt": gt,
                "real_text": real_text, "real_pred": real_pred or "", "real_correct": rc,
                "gray_text": gray_text, "gray_pred": gray_pred or "", "gray_correct": gcorr,
                "recoverable": int(rc == 1 and gcorr == 0),
                "in_requested_cohort": int(in_cohort),
            })
            if not in_cohort:
                del real_batch, gray_batch
                continue

            real_states = capture_last_states(
                model=model, processor=processor, decoder_layers=decoder_layers,
                selected_layers=selected_layers, rec=rec, image=real_img,
                device=torch.device(args.device), prompt_template=args.prompt_template
            )["states"]
            gray_states = capture_last_states(
                model=model, processor=processor, decoder_layers=decoder_layers,
                selected_layers=selected_layers, rec=rec, image=gray_img,
                device=torch.device(args.device), prompt_template=args.prompt_template
            )["states"]

            for li in selected_layers:
                full_delta = (real_states[li] - gray_states[li]).astype(np.float32)
                full_norm = float(np.linalg.norm(full_delta))

                if "full" in args.modes:
                    with LastTokenAddHook(decoder_layers[li], last_pos, full_delta):
                        txt, pred = generate_answer(
                            model, processor, gray_batch, args.max_new_tokens
                        )
                    ec = int(pred == gt)
                    patches.append({
                        "sid": sid, "layer": li, "requested_rank": 0, "actual_rank": 0,
                        "mode": "full_restore", "random_seed": "", "gt": gt,
                        "real_pred": real_pred or "", "gray_pred": gray_pred or "",
                        "edited_text": txt, "edited_pred": pred or "", "edited_correct": ec,
                        "recovered_from_gray": int(gcorr == 0 and ec == 1),
                        "damaged_from_real": 0,
                        "full_delta_norm": full_norm, "edit_norm": full_norm,
                        "component_norm_fraction": 1.0,
                    })

                for rk in requested_ranks:
                    B = get_basis(svd_models[li], rk)
                    actual = int(B.shape[1])
                    if actual <= 0:
                        continue
                    pca_delta = project(full_delta, B)
                    pca_norm = float(np.linalg.norm(pca_delta))
                    rest = (full_delta - pca_delta).astype(np.float32)
                    frac = pca_norm / full_norm if full_norm > EPS else 0.0

                    if "pca" in args.modes:
                        with LastTokenAddHook(decoder_layers[li], last_pos, pca_delta):
                            txt, pred = generate_answer(
                                model, processor, gray_batch, args.max_new_tokens
                            )
                        ec = int(pred == gt)
                        patches.append({
                            "sid": sid, "layer": li, "requested_rank": rk,
                            "actual_rank": actual, "mode": "pca_restore", "random_seed": "",
                            "gt": gt, "real_pred": real_pred or "", "gray_pred": gray_pred or "",
                            "edited_text": txt, "edited_pred": pred or "", "edited_correct": ec,
                            "recovered_from_gray": int(gcorr == 0 and ec == 1),
                            "damaged_from_real": 0, "full_delta_norm": full_norm,
                            "edit_norm": pca_norm, "component_norm_fraction": frac,
                        })

                    if "complement" in args.modes:
                        with LastTokenAddHook(decoder_layers[li], last_pos, rest):
                            txt, pred = generate_answer(
                                model, processor, gray_batch, args.max_new_tokens
                            )
                        ec = int(pred == gt)
                        patches.append({
                            "sid": sid, "layer": li, "requested_rank": rk,
                            "actual_rank": actual, "mode": "complement_restore",
                            "random_seed": "", "gt": gt,
                            "real_pred": real_pred or "", "gray_pred": gray_pred or "",
                            "edited_text": txt, "edited_pred": pred or "", "edited_correct": ec,
                            "recovered_from_gray": int(gcorr == 0 and ec == 1),
                            "damaged_from_real": 0, "full_delta_norm": full_norm,
                            "edit_norm": float(np.linalg.norm(rest)),
                            "component_norm_fraction": frac,
                        })

                    if "necessity" in args.modes:
                        with LastTokenAddHook(decoder_layers[li], last_pos, -pca_delta):
                            txt, pred = generate_answer(
                                model, processor, real_batch, args.max_new_tokens
                            )
                        ec = int(pred == gt)
                        patches.append({
                            "sid": sid, "layer": li, "requested_rank": rk,
                            "actual_rank": actual, "mode": "real_minus_pca",
                            "random_seed": "", "gt": gt,
                            "real_pred": real_pred or "", "gray_pred": gray_pred or "",
                            "edited_text": txt, "edited_pred": pred or "", "edited_correct": ec,
                            "recovered_from_gray": 0,
                            "damaged_from_real": int(rc == 1 and ec == 0),
                            "full_delta_norm": full_norm, "edit_norm": pca_norm,
                            "component_norm_fraction": frac,
                        })

                    if "random" in args.modes and args.random_seeds > 0:
                        for rseed in range(args.random_seeds):
                            key = (li, rk, rseed)
                            if key not in random_cache:
                                random_cache[key] = make_random_basis(
                                    dim=len(full_delta), rank=actual,
                                    seed=args.seed + li*1009 + rk*100003 + rseed*1000003,
                                    orthogonal_to=top_for_random.get(li),
                                )
                            RB = random_cache[key]
                            raw = project(full_delta, RB)
                            rd = norm_match(
                                raw, pca_norm, RB,
                                args.seed + sid*10000019 + li*1009 + rk*100003 + rseed,
                            )
                            with LastTokenAddHook(decoder_layers[li], last_pos, rd):
                                txt, pred = generate_answer(
                                    model, processor, gray_batch, args.max_new_tokens
                                )
                            ec = int(pred == gt)
                            patches.append({
                                "sid": sid, "layer": li, "requested_rank": rk,
                                "actual_rank": actual, "mode": "random_restore",
                                "random_seed": rseed, "gt": gt,
                                "real_pred": real_pred or "", "gray_pred": gray_pred or "",
                                "edited_text": txt, "edited_pred": pred or "",
                                "edited_correct": ec,
                                "recovered_from_gray": int(gcorr == 0 and ec == 1),
                                "damaged_from_real": 0, "full_delta_norm": full_norm,
                                "edit_norm": float(np.linalg.norm(rd)),
                                "component_norm_fraction": frac,
                            })

            del real_batch, gray_batch, real_states, gray_states
            if args.save_every > 0 and sample_i % args.save_every == 0:
                write_csv(out_dir / "baseline_per_sample.csv", baselines)
                write_csv(out_dir / "patch_per_sample.csv", patches)

        except Exception as e:
            errors.append({"sid": sid, "error_type": type(e).__name__, "error": str(e)})
            tqdm.write(f"[ERROR sid={sid}] {type(e).__name__}: {e}")
        finally:
            if real_img is not None:
                real_img.close()
            if gray_img is not None:
                gray_img.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(out_dir / "baseline_per_sample.csv", baselines)
    write_csv(out_dir / "patch_per_sample.csv", patches)
    write_csv(out_dir / "errors.csv", errors)
    return baselines, patches, errors


def summarize_baselines(rows):
    if not rows:
        return []
    return [{
        "n": len(rows),
        "real_accuracy": safe_mean(r["real_correct"] for r in rows),
        "gray_accuracy": safe_mean(r["gray_correct"] for r in rows),
        "n_recoverable": sum(int(r["recoverable"]) for r in rows),
        "n_in_requested_cohort": sum(int(r["in_requested_cohort"]) for r in rows),
    }]


def summarize_ranks(patches, selected_layers, requested_ranks):
    full, pca, comp, nec = {}, {}, {}, {}
    rand = defaultdict(dict)
    actual_rank = {}
    fractions = defaultdict(list)

    for r in patches:
        sid, li, rk = int(r["sid"]), int(r["layer"]), int(r["requested_rank"])
        mode = str(r["mode"])
        rec = int(r["recovered_from_gray"])
        if mode == "full_restore":
            full[(sid, li)] = rec
        elif mode == "pca_restore":
            pca[(sid, li, rk)] = rec
            actual_rank[(li, rk)] = int(r["actual_rank"])
            fractions[(li, rk)].append(float(r["component_norm_fraction"]))
        elif mode == "complement_restore":
            comp[(sid, li, rk)] = rec
        elif mode == "real_minus_pca":
            nec[(sid, li, rk)] = int(r["damaged_from_real"])
        elif mode == "random_restore":
            rand[(li, rk, int(r["random_seed"]))][sid] = rec

    out = []
    for li in selected_layers:
        sids = sorted(
            {sid for sid, lli in full if lli == li} |
            {sid for sid, lli, _ in pca if lli == li}
        )
        for rk in requested_ranks:
            fv = [full[(sid, li)] for sid in sids if (sid, li) in full]
            pv = [pca[(sid, li, rk)] for sid in sids if (sid, li, rk) in pca]
            cv = [comp[(sid, li, rk)] for sid in sids if (sid, li, rk) in comp]
            nv = [nec[(sid, li, rk)] for sid in sids if (sid, li, rk) in nec]

            full_s = {sid for sid in sids if full.get((sid, li), 0) == 1}
            pca_s = {sid for sid in sids if pca.get((sid, li, rk), 0) == 1}
            both = full_s & pca_s
            union = full_s | pca_s

            seed_rates = []
            seeds = sorted({seed for lli, rr, seed in rand if lli == li and rr == rk})
            for seed in seeds:
                seed_rates.append(safe_mean(rand[(li, rk, seed)].values()))

            fr = safe_mean(fv)
            pr = safe_mean(pv)
            rr = safe_mean(seed_rates)
            out.append({
                "layer": li,
                "requested_rank": rk,
                "actual_rank": actual_rank.get((li, rk), ""),
                "n": len(pv) if pv else len(fv),
                "full_recovery_rate": fr,
                "pca_recovery_rate": pr,
                "rate_ratio_pca_over_full": (
                    pr / fr if math.isfinite(pr) and math.isfinite(fr) and fr > EPS
                    else float("nan")
                ),
                "n_full_recovered": len(full_s),
                "n_pca_recovered": len(pca_s),
                "n_both_recovered": len(both),
                "pca_given_full_recovered": len(both)/len(full_s) if full_s else float("nan"),
                "recovery_jaccard": len(both)/len(union) if union else float("nan"),
                "pca_only_recovered": len(pca_s - full_s),
                "full_only_recovered": len(full_s - pca_s),
                "complement_recovery_rate": safe_mean(cv),
                "random_recovery_mean": rr,
                "random_recovery_std": safe_std(seed_rates),
                "pca_minus_random": (
                    pr - rr if math.isfinite(pr) and math.isfinite(rr) else float("nan")
                ),
                "necessity_damage_rate": safe_mean(nv),
                "mean_pca_component_norm_fraction": safe_mean(fractions[(li, rk)]),
            })
    return out


def print_summary(rows):
    print("\n" + "="*176)
    print("REAL-GRAY FULL-DELTA SVD RANK SWEEP — LAST TOKEN")
    print("="*176)
    print(
        "layer rank(actual) | fullRec pcaRec ratio | P(PCA|Full) both/full | "
        "random(mean±sd) complement necessity | mean||PCA||/||full||"
    )
    for r in rows:
        ar = str(r["actual_rank"])
        print(
            f"L{int(r['layer']):02d} k={int(r['requested_rank']):3d}({ar:>3s}) | "
            f"{float(r['full_recovery_rate']):.3f} "
            f"{float(r['pca_recovery_rate']):.3f} "
            f"{float(r['rate_ratio_pca_over_full']):.3f} | "
            f"{float(r['pca_given_full_recovered']):.3f} "
            f"{int(r['n_both_recovered'])}/{int(r['n_full_recovered'])} | "
            f"{float(r['random_recovery_mean']):.3f}±{float(r['random_recovery_std']):.3f} "
            f"{float(r['complement_recovery_rate']):.3f} "
            f"{float(r['necessity_damage_rate']):.3f} | "
            f"{float(r['mean_pca_component_norm_fraction']):.3f}"
        )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(Path(args.direction_dir))
    records_list, _ = base.load_records(args.dataset, Path(args.data_root), None)
    records = {int(r.sid): r for r in records_list}

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    dtype = base.resolve_dtype(spec.dtype_name)
    kw: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

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

    decoder_layers, layer_path = resolve_decoder_layers(model)
    selected_layers = parse_layers(args.layers, len(decoder_layers))
    ranks = parse_ints(args.ranks)
    if not ranks or any(k <= 0 for k in ranks):
        raise ValueError("Ranks must be positive")
    modes = parse_words(args.modes)
    valid = {"full", "pca", "complement", "random", "necessity"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"Unknown modes {bad}")
    args.modes = modes

    print(f"[decoder] {layer_path}; selected={selected_layers}; ranks={ranks}")

    train_cache = (
        Path(args.train_cache) if args.train_cache
        else out_dir / "train_real_gray_last_delta.npz"
    )
    if args.rebuild_train_cache or not train_cache.exists():
        collect_train_cache(
            cache_path=train_cache, model=model, processor=processor,
            decoder_layers=decoder_layers, selected_layers=selected_layers,
            records=records, metadata=metadata, device=torch.device(args.device),
            prompt_template=args.prompt_template, gray_value=args.gray_value,
            train_controls=args.train_controls, max_samples=args.train_max_samples,
            seed=args.seed,
        )
    validate_train_cache(train_cache, selected_layers)

    svd_models, spectrum = fit_svd(train_cache, selected_layers, ranks)
    write_csv(out_dir / "svd_spectrum.csv", spectrum)
    print("\nTRAIN SVD")
    for li in selected_layers:
        info = svd_models[li]
        print(
            f"  L{li:02d}: Ntrain={info['n_train']} hidden={info['hidden_dim']} "
            f"effective_rank={info['effective_rank']}"
        )

    eval_sids = select_test_sids(metadata, records, args.max_samples, args.seed)
    print(f"\n[eval] candidate test N={len(eval_sids)}, cohort={args.cohort}, modes={modes}")

    baselines, patches, errors = run_eval(
        model=model, processor=processor, decoder_layers=decoder_layers,
        selected_layers=selected_layers, requested_ranks=ranks,
        svd_models=svd_models, records=records, metadata=metadata,
        eval_sids=eval_sids, args=args, out_dir=out_dir,
    )

    bsum = summarize_baselines(baselines)
    rsum = summarize_ranks(patches, selected_layers, ranks)
    write_csv(out_dir / "baseline_summary.csv", bsum)
    write_csv(out_dir / "rank_summary.csv", rsum)

    if bsum:
        b = bsum[0]
        print(
            f"\nBASELINE N={int(b['n'])} real_acc={b['real_accuracy']:.4f} "
            f"gray_acc={b['gray_accuracy']:.4f} recoverable={int(b['n_recoverable'])} "
            f"in_cohort={int(b['n_in_requested_cohort'])}"
        )
    print_summary(rsum)

    (out_dir / "summary.json").write_text(
        json.dumps({
            "experiment": "Real-vs-Gray full-delta uncentered SVD rank sweep",
            "site": "last prompt token at decoder block input",
            "gradient_used": False,
            "spatial_labels_used_to_fit_subspace": False,
            "svd_centered": False,
            "layers": selected_layers,
            "requested_ranks": ranks,
            "modes": modes,
            "cohort": args.cohort,
            "train_cache": str(train_cache),
            "random_seeds": args.random_seeds,
            "n_eval_candidates": len(eval_sids),
            "n_errors": len(errors),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for p in [
        train_cache, out_dir / "svd_spectrum.csv",
        out_dir / "baseline_per_sample.csv", out_dir / "baseline_summary.csv",
        out_dir / "patch_per_sample.csv", out_dir / "rank_summary.csv",
        out_dir / "errors.csv", out_dir / "summary.json",
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
