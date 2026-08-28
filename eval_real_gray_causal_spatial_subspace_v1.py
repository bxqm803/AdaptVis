#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-vs-Gray causal spatial-subspace scan (last prompt token, no gradients).

Idea
----
For each layer l and sample i:
    delta_i,l = h_last,l(real) - h_last,l(gray)

Using TRAIN samples, group delta by GT relation, subtract the global mean,
compute the four relation means, and SVD those means. This gives a low-rank
between-relation subspace (rank <= 3).

On TEST samples, compare actual model.generate() after:
  full_restore:        gray + full Real-Gray delta
  spatial_restore:     gray + P_spatial delta
  complement_restore:  gray + (I-P_spatial) delta
  random_restore:      gray + matched-rank, norm-matched random component
  necessity:           real - P_spatial delta

The clean sufficiency cohort is fresh real-correct / gray-wrong samples.
Primary metric is actual generation recovery to GT.

Depends only on project-native:
  extract_two_object_relation_states.py
  analyze_layerwise_direction_failure_scan_v1.py
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
    p.add_argument("--layers", default="14-27")
    p.add_argument("--ranks", default="1,2,3")
    p.add_argument("--modes", default="full,spatial,complement,random")
    p.add_argument("--cohort", default="recoverable",
                   choices=["recoverable", "real_correct", "gray_wrong", "all"])
    p.add_argument("--train-controls", default="correct", choices=["correct", "all"])
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--random-seeds", type=int, default=5)
    p.add_argument("--random-orthogonal-to-spatial",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def safe_mean(xs: Iterable[float]):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(xs: Iterable[float]):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0


def parse_words(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_layers(text: str, n_layers: int):
    if text.strip().lower() == "all":
        return list(range(n_layers))
    vals = []
    for x in parse_words(text):
        if "-" in x:
            a, b = map(int, x.split("-", 1))
            step = 1 if b >= a else -1
            vals.extend(range(a, b + step, step))
        else:
            vals.append(int(x))
    vals = sorted(set(vals))
    bad = [x for x in vals if not (0 <= x < n_layers)]
    if bad:
        raise ValueError(f"Invalid layers {bad}; valid 0..{n_layers-1}")
    return vals


def load_metadata(direction_dir: Path):
    vec_path = direction_dir / "vectors.npz"
    gen_path = direction_dir / "sample_split_and_generation.csv"
    if not vec_path.exists(): raise FileNotFoundError(vec_path)
    if not gen_path.exists(): raise FileNotFoundError(gen_path)

    with np.load(vec_path, allow_pickle=True) as z:
        sids = z["sample_index"].astype(np.int64)
        labels = [direction.norm_relation(x) for x in z["relation"]]
    gt = {int(sid): labels[i] for i, sid in enumerate(sids.tolist())}

    split, generation = {}, {}
    for r in read_csv(gen_path):
        sid = int(r["sample_index"])
        split[sid] = str(r.get("split", "")).strip()
        pred = direction.norm_relation(r.get("generation_pred", ""))
        group = str(r.get("generation_group", "")).strip().lower()
        g = gt.get(sid, "")
        if group not in ("correct", "wrong") and g in RELATIONS and pred in RELATIONS:
            group = "correct" if pred == g else "wrong"
        generation[sid] = {"generation_group": group, "generation_pred": pred}
    return {"sids": [int(x) for x in sids.tolist()], "gt": gt,
            "split": split, "generation": generation}


def get_attr_path(obj: Any, path: str):
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def resolve_decoder_layers(model):
    for path in [
        "model.language_model.layers", "language_model.layers",
        "model.model.layers", "model.layers", "language_model.model.layers",
    ]:
        try:
            layers = get_attr_path(model, path)
            if len(layers) and hasattr(layers[0], "self_attn"):
                return layers, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers")


def first_tensor(x):
    if torch.is_tensor(x): return x
    if isinstance(x, (tuple, list)):
        for y in x:
            if torch.is_tensor(y): return y
    raise RuntimeError(f"No tensor in {type(x)}")


def infer_last_position(batch):
    if "attention_mask" in batch:
        nz = torch.nonzero(batch["attention_mask"][0], as_tuple=False).flatten()
        if len(nz): return int(nz[-1].item())
    return int(batch["input_ids"].shape[1] - 1)


def build_batch(processor, rec, question, image, device):
    rendered = direction.build_chat_prompt(processor, question, image is not None)
    batch = direction.process_inputs(processor, rendered, image, device)
    return batch, infer_last_position(batch)


def make_gray_image(image: Image.Image, value: int):
    v = int(np.clip(value, 0, 255))
    return Image.new("RGB", image.size, (v, v, v))


def parse_generated_relation(text: str) -> Optional[str]:
    s = str(text).strip().lower()
    hits = []
    for rel, pat in [
        ("left", r"\bleft\b"), ("right", r"\bright\b"),
        ("above", r"\babove\b"), ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"), ("below", r"\bunder(?:neath)?\b"),
    ]:
        m = re.search(pat, s)
        if m: hits.append((m.start(), rel))
    if not hits: return None
    hits.sort(); return hits[0][1]


def generate_answer(model, processor, batch, max_new_tokens):
    input_len = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(**batch, do_sample=False,
                             max_new_tokens=max_new_tokens, use_cache=True)
    text = processor.tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
    pred = parse_generated_relation(text)
    del out
    return text, pred


class LastStateCapture:
    def __init__(self, layers, selected_layers, last_pos):
        self.last_pos = int(last_pos)
        self.states = {}
        self.handles = []
        for li in selected_layers:
            self.handles.append(layers[li].register_forward_pre_hook(self._make(li)))

    def _make(self, li):
        def hook(_m, args):
            if not args: return None
            x = first_tensor(args)
            if x.ndim != 3 or self.last_pos >= int(x.shape[1]): return None
            self.states[li] = x[0, self.last_pos].detach().float().cpu().numpy().astype(np.float32)
            return None
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception): h.remove()
        self.handles = []

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def capture_last_states(model, processor, layers, selected_layers, rec, image,
                        device, prompt_template):
    q = prompt_template.format(subject=rec.subject, reference=rec.reference)
    batch, last_pos = build_batch(processor, rec, q, image, device)
    with LastStateCapture(layers, selected_layers, last_pos) as cap:
        with torch.inference_mode():
            _ = model(**batch, output_attentions=False, output_hidden_states=False,
                      use_cache=False, return_dict=True)
    missing = [li for li in selected_layers if li not in cap.states]
    if missing: raise RuntimeError(f"Missing last states {missing}")
    del batch
    return dict(cap.states)


def select_train_sids(meta, records, controls, max_samples, seed):
    sids = []
    for sid in meta["sids"]:
        if meta["split"].get(sid) != "train" or sid not in records: continue
        if meta["gt"].get(sid) not in RELATIONS: continue
        if controls == "correct":
            if meta["generation"].get(sid, {}).get("generation_group") != "correct":
                continue
        sids.append(sid)
    if max_samples is not None and len(sids) > max_samples:
        rng = random.Random(seed); rng.shuffle(sids); sids = sids[:max_samples]
    return sorted(sids)


def collect_train_cache(cache_path, model, processor, layers, selected_layers,
                        records, meta, device, prompt_template, gray_value,
                        controls, max_samples, seed):
    sids = select_train_sids(meta, records, controls, max_samples, seed)
    if not sids: raise RuntimeError("No TRAIN samples selected")
    print(f"[train-cache] N={len(sids)} controls={controls} layers={selected_layers}")

    vals = {li: [] for li in selected_layers}
    kept_sid, kept_rel, errors = [], [], []
    for sid in tqdm(sids, desc="collect TRAIN Real-Gray last deltas"):
        rec = records[sid]; real = gray = None
        try:
            real = Image.open(rec.image_path).convert("RGB")
            gray = make_gray_image(real, gray_value)
            rs = capture_last_states(model, processor, layers, selected_layers,
                                     rec, real, device, prompt_template)
            gs = capture_last_states(model, processor, layers, selected_layers,
                                     rec, gray, device, prompt_template)
            for li in selected_layers:
                vals[li].append((rs[li] - gs[li]).astype(np.float32))
            kept_sid.append(sid); kept_rel.append(meta["gt"][sid])
        except Exception as e:
            errors.append({"sid": sid, "error_type": type(e).__name__, "error": str(e)})
            tqdm.write(f"[TRAIN ERROR sid={sid}] {type(e).__name__}: {e}")
        finally:
            if real is not None: real.close()
            if gray is not None: gray.close()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    payload = {
        "sid": np.asarray(kept_sid, dtype=np.int64),
        "relation": np.asarray(kept_rel, dtype=object),
        "selected_layers": np.asarray(selected_layers, dtype=np.int64),
        "gray_value": np.asarray([gray_value], dtype=np.int64),
        "train_controls": np.asarray([controls], dtype=object),
    }
    for li in selected_layers:
        payload[f"L{li}_delta_last"] = np.stack(vals[li], axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    write_csv(cache_path.with_suffix(".errors.csv"), errors)
    print(f"[train-cache] saved {cache_path}; N={len(kept_sid)}")


def validate_cache(path: Path, selected_layers):
    if not path.exists(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        avail = set(int(x) for x in z["selected_layers"].tolist())
        missing = [li for li in selected_layers
                   if li not in avail or f"L{li}_delta_last" not in z.files]
    if missing: raise RuntimeError(f"TRAIN cache missing layers {missing}")


def fit_subspaces(cache_path: Path, selected_layers):
    subspaces, rows = {}, []
    with np.load(cache_path, allow_pickle=True) as z:
        labels = np.asarray([direction.norm_relation(x) for x in z["relation"]], dtype=object)
        for li in selected_layers:
            X = np.asarray(z[f"L{li}_delta_last"], dtype=np.float32)
            gmean = X.mean(axis=0)
            rel_means = []
            for rel in RELATIONS:
                mask = labels == rel
                if not np.any(mask): raise RuntimeError(f"No TRAIN {rel} at L{li}")
                rel_means.append(X[mask].mean(axis=0) - gmean)
            M = np.stack(rel_means, axis=0).astype(np.float64)
            M -= M.mean(axis=0, keepdims=True)
            _, S, Vh = np.linalg.svd(M, full_matrices=False)
            tol = 1e-8 * max(float(S.max()) if len(S) else 0.0, 1.0)
            eff = int(np.sum(S > tol))
            total = float(np.sum(S ** 2))
            subspaces[li] = {"Vh": Vh.astype(np.float32), "S": S.astype(np.float32),
                             "effective_rank": eff}
            row = {"layer": li, "effective_rank": eff,
                   "sv1": float(S[0]) if len(S)>0 else 0.0,
                   "sv2": float(S[1]) if len(S)>1 else 0.0,
                   "sv3": float(S[2]) if len(S)>2 else 0.0}
            for k in (1,2,3):
                row[f"explained_var_rank{k}"] = (
                    float(np.sum(S[:k]**2))/total if total > EPS else 0.0
                )
            rows.append(row)
    return subspaces, rows


def basis_for(sub, rank):
    k = min(int(rank), int(sub["effective_rank"]), int(sub["Vh"].shape[0]))
    if k <= 0:
        return np.zeros((sub["Vh"].shape[1], 0), dtype=np.float32)
    return sub["Vh"][:k].T.astype(np.float32)


def project(v, B):
    x = np.asarray(v, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[1] == 0:
        return np.zeros_like(x, dtype=np.float32)
    return (B @ (B.T @ x)).astype(np.float32)


def random_basis(dim, rank, spatial_B, seed, orthogonal=True):
    if rank <= 0: return np.zeros((dim,0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    S = np.asarray(spatial_B, dtype=np.float64)
    cols = []
    for _ in range(rank):
        found = None
        for _try in range(200):
            v = rng.standard_normal(dim)
            if orthogonal and S.ndim == 2 and S.shape[1] > 0:
                v -= S @ (S.T @ v)
            if cols:
                C = np.stack(cols, axis=1)
                v -= C @ (C.T @ v)
            n = float(np.linalg.norm(v))
            if n > 1e-8:
                found = v/n; break
        if found is None: raise RuntimeError("Failed random basis")
        cols.append(found)
    return np.stack(cols, axis=1).astype(np.float32)


def norm_match(v, target_norm, B, seed):
    x = np.asarray(v, dtype=np.float32)
    if target_norm <= EPS: return np.zeros_like(x)
    n = float(np.linalg.norm(x))
    if n > EPS: return (x * (target_norm/n)).astype(np.float32)
    if B.shape[1] > 0:
        rng = np.random.default_rng(seed)
        y = B @ rng.standard_normal(B.shape[1])
        yn = float(np.linalg.norm(y))
        if yn > EPS: return (y/yn*target_norm).astype(np.float32)
    return np.zeros_like(x)


class AddLastHook:
    def __init__(self, block, last_pos, delta):
        self.block = block; self.last_pos = int(last_pos)
        self.delta = np.asarray(delta, dtype=np.float32)
        self.applied = False; self.handle = None

    def _hook(self, _m, args):
        if self.applied or not args: return None
        vals = list(args); idx = None; x = None
        for i, item in enumerate(vals):
            if torch.is_tensor(item): idx, x = i, item; break
        if x is None or x.ndim != 3 or self.last_pos >= int(x.shape[1]): return None
        y = x.clone()
        if float(np.linalg.norm(self.delta)) > EPS:
            d = torch.from_numpy(self.delta).to(device=y.device, dtype=y.dtype)
            y[0, self.last_pos, :] = y[0, self.last_pos, :] + d
        self.applied = True; vals[idx] = y
        return tuple(vals)

    def __enter__(self):
        self.handle = self.block.register_forward_pre_hook(self._hook); return self

    def __exit__(self, *_):
        if self.handle is not None:
            with contextlib.suppress(Exception): self.handle.remove()
        self.handle = None


def select_test_sids(meta, records, max_samples, seed):
    sids = [sid for sid in meta["sids"]
            if meta["split"].get(sid) == "test" and sid in records
            and meta["gt"].get(sid) in RELATIONS]
    if max_samples is not None and len(sids) > max_samples:
        rng = random.Random(seed); rng.shuffle(sids); sids = sids[:max_samples]
    return sorted(sids)


def in_cohort(name, rc, gc):
    if name == "recoverable": return bool(rc and not gc)
    if name == "real_correct": return bool(rc)
    if name == "gray_wrong": return bool(not gc)
    return True


def run_eval(model, processor, layers, selected_layers, ranks, subspaces,
             records, meta, eval_sids, args, out_dir):
    baselines, patches, errors = [], [], []
    rb_cache = {}

    for n, sid in enumerate(tqdm(eval_sids, desc="Real-Gray causal subspace scan"), 1):
        rec = records[sid]; real = gray = None
        try:
            gt = meta["gt"][sid]
            real = Image.open(rec.image_path).convert("RGB")
            gray = make_gray_image(real, args.gray_value)
            q = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
            real_batch, rlast = build_batch(processor, rec, q, real, torch.device(args.device))
            gray_batch, glast = build_batch(processor, rec, q, gray, torch.device(args.device))
            if rlast != glast: raise RuntimeError(f"last_pos mismatch {rlast} vs {glast}")

            rtext, rpred = generate_answer(model, processor, real_batch, args.max_new_tokens)
            gtext, gpred = generate_answer(model, processor, gray_batch, args.max_new_tokens)
            rc, gcorr = int(rpred == gt), int(gpred == gt)
            cohort = in_cohort(args.cohort, rc, gcorr)
            baselines.append({"sid": sid, "gt": gt, "real_pred": rpred or "",
                              "gray_pred": gpred or "", "real_correct": rc,
                              "gray_correct": gcorr, "recoverable": int(rc and not gcorr),
                              "in_requested_cohort": int(cohort)})
            if not cohort:
                del real_batch, gray_batch
                continue

            rs = capture_last_states(model, processor, layers, selected_layers,
                                     rec, real, torch.device(args.device), args.prompt_template)
            gs = capture_last_states(model, processor, layers, selected_layers,
                                     rec, gray, torch.device(args.device), args.prompt_template)

            for li in selected_layers:
                full = (rs[li] - gs[li]).astype(np.float32)
                full_norm = float(np.linalg.norm(full))

                if "full" in args.modes:
                    with AddLastHook(layers[li], rlast, full):
                        et, ep = generate_answer(model, processor, gray_batch, args.max_new_tokens)
                    ec = int(ep == gt)
                    patches.append({"sid": sid, "layer": li, "rank": 0, "actual_rank": 0,
                                    "mode": "full_restore", "random_seed": "", "gt": gt,
                                    "edited_pred": ep or "", "edited_correct": ec,
                                    "recovered_from_gray": int((not gcorr) and ec),
                                    "damaged_from_real": 0, "full_delta_norm": full_norm,
                                    "edit_norm": full_norm, "spatial_component_norm": "",
                                    "component_fraction_of_full": 1.0})

                for rank in ranks:
                    B = basis_for(subspaces[li], rank)
                    actual_rank = int(B.shape[1])
                    spat = project(full, B)
                    snorm = float(np.linalg.norm(spat))
                    comp = (full - spat).astype(np.float32)
                    frac = snorm/full_norm if full_norm > EPS else 0.0

                    if "spatial" in args.modes:
                        with AddLastHook(layers[li], rlast, spat):
                            et, ep = generate_answer(model, processor, gray_batch, args.max_new_tokens)
                        ec = int(ep == gt)
                        patches.append({"sid": sid, "layer": li, "rank": rank,
                                        "actual_rank": actual_rank, "mode": "spatial_restore",
                                        "random_seed": "", "gt": gt, "edited_pred": ep or "",
                                        "edited_correct": ec,
                                        "recovered_from_gray": int((not gcorr) and ec),
                                        "damaged_from_real": 0, "full_delta_norm": full_norm,
                                        "edit_norm": snorm, "spatial_component_norm": snorm,
                                        "component_fraction_of_full": frac})

                    if "complement" in args.modes:
                        with AddLastHook(layers[li], rlast, comp):
                            et, ep = generate_answer(model, processor, gray_batch, args.max_new_tokens)
                        ec = int(ep == gt)
                        patches.append({"sid": sid, "layer": li, "rank": rank,
                                        "actual_rank": actual_rank, "mode": "complement_restore",
                                        "random_seed": "", "gt": gt, "edited_pred": ep or "",
                                        "edited_correct": ec,
                                        "recovered_from_gray": int((not gcorr) and ec),
                                        "damaged_from_real": 0, "full_delta_norm": full_norm,
                                        "edit_norm": float(np.linalg.norm(comp)),
                                        "spatial_component_norm": snorm,
                                        "component_fraction_of_full": frac})

                    if "necessity" in args.modes:
                        with AddLastHook(layers[li], rlast, -spat):
                            et, ep = generate_answer(model, processor, real_batch, args.max_new_tokens)
                        ec = int(ep == gt)
                        patches.append({"sid": sid, "layer": li, "rank": rank,
                                        "actual_rank": actual_rank, "mode": "real_minus_spatial",
                                        "random_seed": "", "gt": gt, "edited_pred": ep or "",
                                        "edited_correct": ec, "recovered_from_gray": 0,
                                        "damaged_from_real": int(rc and not ec),
                                        "full_delta_norm": full_norm, "edit_norm": snorm,
                                        "spatial_component_norm": snorm,
                                        "component_fraction_of_full": frac})

                    if "random" in args.modes and actual_rank > 0 and snorm > EPS:
                        for rseed in range(args.random_seeds):
                            key = (li, rank, rseed)
                            if key not in rb_cache:
                                rb_cache[key] = random_basis(
                                    len(full), actual_rank, B,
                                    args.seed + li*1009 + rank*100003 + rseed*1000003,
                                    args.random_orthogonal_to_spatial,
                                )
                            RB = rb_cache[key]
                            raw = project(full, RB)
                            rnd = norm_match(raw, snorm, RB,
                                             args.seed + sid*10000019 + li*1009 + rank*100003 + rseed)
                            with AddLastHook(layers[li], rlast, rnd):
                                et, ep = generate_answer(model, processor, gray_batch, args.max_new_tokens)
                            ec = int(ep == gt)
                            patches.append({"sid": sid, "layer": li, "rank": rank,
                                            "actual_rank": actual_rank, "mode": "random_restore",
                                            "random_seed": rseed, "gt": gt, "edited_pred": ep or "",
                                            "edited_correct": ec,
                                            "recovered_from_gray": int((not gcorr) and ec),
                                            "damaged_from_real": 0, "full_delta_norm": full_norm,
                                            "edit_norm": float(np.linalg.norm(rnd)),
                                            "spatial_component_norm": snorm,
                                            "component_fraction_of_full": frac})

            del real_batch, gray_batch, rs, gs
            if args.save_every > 0 and n % args.save_every == 0:
                write_csv(out_dir / "baseline_per_sample.csv", baselines)
                write_csv(out_dir / "patch_per_sample.csv", patches)

        except Exception as e:
            errors.append({"sid": sid, "error_type": type(e).__name__, "error": str(e)})
            tqdm.write(f"[ERROR sid={sid}] {type(e).__name__}: {e}")
        finally:
            if real is not None: real.close()
            if gray is not None: gray.close()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    write_csv(out_dir / "baseline_per_sample.csv", baselines)
    write_csv(out_dir / "patch_per_sample.csv", patches)
    write_csv(out_dir / "errors.csv", errors)
    return baselines, patches, errors


def summarize_baselines(rows):
    return [{
        "n": len(rows),
        "real_accuracy": safe_mean(r["real_correct"] for r in rows),
        "gray_accuracy": safe_mean(r["gray_correct"] for r in rows),
        "n_recoverable": sum(int(r["recoverable"]) for r in rows),
        "n_in_requested_cohort": sum(int(r["in_requested_cohort"]) for r in rows),
    }]


def causal_summary(patches, layers, ranks):
    values = defaultdict(list)
    random_values = defaultdict(list)
    necessity_values = defaultdict(list)

    for r in patches:
        mode = r["mode"]; li = int(r["layer"]); rank = int(r["rank"])
        if mode == "random_restore":
            random_values[(li, rank, int(r["random_seed"]))].append(int(r["recovered_from_gray"]))
        elif mode == "real_minus_spatial":
            necessity_values[(li, rank)].append(int(r["damaged_from_real"]))
        else:
            values[(mode, li, rank)].append(int(r["recovered_from_gray"]))

    out = []
    for li in layers:
        full = safe_mean(values.get(("full_restore", li, 0), []))
        for rank in ranks:
            spatial = safe_mean(values.get(("spatial_restore", li, rank), []))
            comp = safe_mean(values.get(("complement_restore", li, rank), []))
            nec = safe_mean(necessity_values.get((li, rank), []))
            seed_rates = []
            seeds = sorted({s for (l,k,s) in random_values if l == li and k == rank})
            for s in seeds:
                seed_rates.append(safe_mean(random_values[(li, rank, s)]))
            rand_mean, rand_std = safe_mean(seed_rates), safe_std(seed_rates)
            crr = spatial/full if (math.isfinite(spatial) and math.isfinite(full) and full > EPS) else float("nan")
            out.append({
                "layer": li, "rank": rank,
                "full_recovery_rate": full,
                "spatial_recovery_rate": spatial,
                "causal_recovery_ratio": crr,
                "complement_recovery_rate": comp,
                "random_recovery_mean": rand_mean,
                "random_recovery_std": rand_std,
                "necessity_damage_rate": nec,
                "spatial_minus_random": spatial-rand_mean if math.isfinite(spatial) and math.isfinite(rand_mean) else float("nan"),
                "spatial_minus_complement": spatial-comp if math.isfinite(spatial) and math.isfinite(comp) else float("nan"),
            })
    return out


def print_summary(rows):
    print("\n" + "="*158)
    print("REAL-GRAY LOW-RANK CAUSAL SPATIAL SUBSPACE — LAST TOKEN")
    print("="*158)
    print("layer rank | fullRec spatialRec CRR | random(mean±sd) complement | necessity | spatial-random spatial-complement")
    for r in rows:
        print(
            f"L{int(r['layer']):02d} k={int(r['rank'])} | "
            f"{float(r['full_recovery_rate']):.3f} "
            f"{float(r['spatial_recovery_rate']):.3f} "
            f"{float(r['causal_recovery_ratio']):.3f} | "
            f"{float(r['random_recovery_mean']):.3f}±{float(r['random_recovery_std']):.3f} "
            f"{float(r['complement_recovery_rate']):.3f} | "
            f"{float(r['necessity_damage_rate']):.3f} | "
            f"{float(r['spatial_minus_random']):+.3f} "
            f"{float(r['spatial_minus_complement']):+.3f}"
        )


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(Path(args.direction_dir))
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
    if args.attn_impl != "none": kw["attn_implementation"] = args.attn_impl
    print(f"[model] loading {spec.repo_id} on {args.device}")
    try:
        model = cls.from_pretrained(spec.repo_id, dtype=dtype, **kw)
    except TypeError:
        model = cls.from_pretrained(spec.repo_id, torch_dtype=dtype, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)

    layers, layer_path = resolve_decoder_layers(model)
    selected_layers = parse_layers(args.layers, len(layers))
    ranks = sorted(set(int(x) for x in parse_words(args.ranks)))
    args.modes = parse_words(args.modes)
    valid = {"full", "spatial", "complement", "random", "necessity"}
    bad = [x for x in args.modes if x not in valid]
    if bad: raise ValueError(f"Unknown modes {bad}")

    print(f"[decoder] {layer_path}; selected={selected_layers}; ranks={ranks}")

    train_cache = Path(args.train_cache) if args.train_cache else out_dir / "train_real_gray_last_delta.npz"
    if args.rebuild_train_cache or not train_cache.exists():
        collect_train_cache(
            train_cache, model, processor, layers, selected_layers, records, meta,
            torch.device(args.device), args.prompt_template, args.gray_value,
            args.train_controls, args.train_max_samples, args.seed,
        )
    validate_cache(train_cache, selected_layers)
    subs, spectrum = fit_subspaces(train_cache, selected_layers)
    write_csv(out_dir / "spatial_subspace_spectrum.csv", spectrum)

    eval_sids = select_test_sids(meta, records, args.max_samples, args.seed)
    print(f"[eval] candidate test N={len(eval_sids)} cohort={args.cohort} modes={args.modes}")
    baselines, patches, errors = run_eval(
        model, processor, layers, selected_layers, ranks, subs,
        records, meta, eval_sids, args, out_dir,
    )

    bsum = summarize_baselines(baselines)
    csum = causal_summary(patches, selected_layers, ranks)
    write_csv(out_dir / "baseline_summary.csv", bsum)
    write_csv(out_dir / "causal_recovery_summary.csv", csum)

    if bsum:
        b = bsum[0]
        print("\nBASELINES")
        print(f"N={int(b['n'])} real_acc={float(b['real_accuracy']):.4f} "
              f"gray_acc={float(b['gray_accuracy']):.4f} "
              f"recoverable={int(b['n_recoverable'])} in_cohort={int(b['n_in_requested_cohort'])}")
    print_summary(csum)

    (out_dir / "summary.json").write_text(json.dumps({
        "experiment": "Real-vs-Gray low-rank causal spatial subspace",
        "site": "last prompt token at decoder block input",
        "gradient_used": False,
        "image_flip_used": False,
        "gray_value": args.gray_value,
        "train_controls": args.train_controls,
        "cohort": args.cohort,
        "layers": selected_layers,
        "ranks": ranks,
        "modes": args.modes,
        "random_seeds": args.random_seeds,
        "train_cache": str(train_cache),
        "n_errors": len(errors),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved:")
    for p in [
        train_cache,
        out_dir / "spatial_subspace_spectrum.csv",
        out_dir / "baseline_per_sample.csv",
        out_dir / "baseline_summary.csv",
        out_dir / "patch_per_sample.csv",
        out_dir / "causal_recovery_summary.csv",
        out_dir / "errors.csv",
        out_dir / "summary.json",
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
