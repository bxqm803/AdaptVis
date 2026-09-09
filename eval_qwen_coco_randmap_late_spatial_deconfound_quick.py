#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL 3B/7B + COCO random mapping:
decompose late last-token Real-Gray writers into

  (1) answer-controlled spatial relation components
  (2) answer/logit components

and test whether the spatial component remains causal after explicitly
projecting out the A/B/C/D LM-head decision subspace.

Why this script
---------------
In the original fixed-answer task, relation and output identity are confounded:
    LEFT <-> "left"
so a strong late writer can simultaneously be spatial and answer/logit aligned.

Here every sample uses a randomized relation -> A/B/C/D mapping. On TRAIN,
for each late layer L we capture

    q_i,L = h_real[L,last] - h_gray[L,last]

and form 16 cells:
    mu[r,a] = E[q | GT relation=r, correct answer letter=a].

We then estimate spatial axes while HOLDING ANSWER IDENTITY FIXED:

    d_LR = mean_a [ mu[right,a] - mu[left,a] ]
    d_AB = mean_a [ mu[above,a] - mu[below,a] ]

Each bracket compares two samples that would output the SAME letter, so the
fixed answer-symbol component is controlled. We then additionally remove the
A/B/C/D LM-head decision subspace:

    d_null = (I - P_logit) d_raw

and norm-match d_null back to d_raw by default.

Tests
-----
A) Geometry:
   - raw/null norm
   - fraction of raw spatial direction lying in answer-logit subspace
   - cosine to the four answer decision directions
B) Held-out readout:
   - 4-way relation decode from q_last
   - 4-way answer-letter decode from q_last
C) Causal full-generation steering under NEW randomized mappings:
   - spatial_raw_correct / spatial_raw_wrong
   - spatial_null_correct / spatial_null_wrong
   - letter_correct positive control

For a spatial_wrong test, the target letter is sample-specific:
if we inject RIGHT and this sample says RIGHT -> C, targetFollow means C;
another sample may have RIGHT -> A. Therefore successful target following
cannot be explained by a fixed answer token.

Examples
--------
Qwen3B:
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_coco_randmap_late_spatial_deconfound_quick.py \
  --model qwen-3b --layers 26,28,30,32,34,35 \
  --max-samples 320 --eval-max-samples 80 --alphas 1 \
  --output-dir output/qwen3b_randmap_late_spatial_deconfound --overwrite

Qwen7B:
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_coco_randmap_late_spatial_deconfound_quick.py \
  --model qwen-7b --layers 20,22,24,25,26,27 \
  --max-samples 320 --eval-max-samples 80 --alphas 1 \
  --output-dir output/qwen7b_randmap_late_spatial_deconfound --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import random
import re
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj

REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
OPP = {"left": "right", "right": "left", "above": "below", "below": "above"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--layers", default="auto",
                   help="Late last-token layers. auto=3B:26,28,30,32,34,35; 7B:20,22,24,25,26,27")
    p.add_argument("--alphas", default="1")
    p.add_argument("--conditions",
                   default="spatial_raw_correct,spatial_null_correct,spatial_raw_wrong,spatial_null_wrong,letter_correct")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=320, help="0 = all COCO_two before split")
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all held-out")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--null-norm-mode", default="match_raw", choices=["match_raw", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted({int(x.strip().upper().replace("L", "")) for x in s.split(",") if x.strip()})


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def cosine_np(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Dict[str, str]) -> str:
    letter_to_rel = {a: r for r, a in rel_to_letter.items()}
    lines = [
        f"Where is the {subject} relative to the {reference}?",
        "Use ONLY the mapping below and answer with exactly one letter: A, B, C, or D.",
    ]
    for a in LETTERS:
        lines.append(f"{a} = {letter_to_rel[a]}")
    lines.append("Answer:")
    return "\n".join(lines)


def assign_relation_balanced_mappings(items: List[dict], seed: int) -> Dict[int, Dict[str, str]]:
    """Balance GT relation -> correct answer letter inside each GT-relation subgroup."""
    rng = random.Random(seed)
    out = {}
    by_rel = {r: [] for r in REL}
    for m in items:
        by_rel[m["gt"]].append(m)

    for gt in REL:
        group = list(by_rel[gt])
        rng.shuffle(group)
        cycle = list(LETTERS)
        rng.shuffle(cycle)
        for i, m in enumerate(group):
            correct = cycle[i % 4]
            other_r = [r for r in REL if r != gt]
            other_a = [a for a in LETTERS if a != correct]
            rng.shuffle(other_r); rng.shuffle(other_a)
            mp = {gt: correct}
            mp.update({r: a for r, a in zip(other_r, other_a)})
            out[int(m["sid"])] = mp
    return out


def mapping_string(mp):
    return "|".join(f"{r}:{mp[r]}" for r in REL)


def parse_answer_letter(text: str, rel_to_letter: Dict[str, str]) -> Tuple[str | None, str]:
    t = str(text).strip()
    up = t.upper()
    for pat in (
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([ABCD])\b",
        r"^\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ):
        m = re.search(pat, up)
        if m:
            return m.group(1), "letter"
    low = t.lower()
    hits = [r for r in REL if re.search(rf"\b{re.escape(r)}\b", low)]
    if len(hits) == 1:
        return rel_to_letter[hits[0]], "relation_fallback"
    return None, "unparsed"


def make_real_image(record):
    im = base.record_image(record)
    if hasattr(im, "convert"):
        im = im.convert("RGB")
    return im


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def make_batch(processor, device, image, prompt):
    return base.make_question_batch(
        processor=processor, image=image, question_text=prompt, device=device
    )


class LastStateCapture:
    def __init__(self, decoder_layers, layers_req):
        self.states = {}
        self.handles = [decoder_layers[L].register_forward_hook(self._mk(L)) for L in layers_req]

    def _mk(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if h.ndim == 3 and h.shape[1] >= 1:
                self.states[L] = h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_prefix_last(model, decoder_layers, batch, layers_req):
    cap = LastStateCapture(decoder_layers, layers_req)
    try:
        kwargs = dict(batch)
        kwargs["use_cache"] = False
        _ = model(**kwargs)
        miss = [L for L in layers_req if L not in cap.states]
        if miss:
            raise RuntimeError(f"Missing last states at {miss}")
        return cap.states
    finally:
        cap.close()


class SingleLastPatch:
    """Patch prefix-last only; generation steps with seq_len==1 are untouched."""
    def __init__(self, layer, direction, alpha):
        self.d = np.asarray(direction, np.float32)
        self.alpha = float(alpha)
        self.applied = 0
        self.delta_norm = float("nan")
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        if h.ndim != 3 or int(h.shape[1]) <= 1:
            return out
        y = h.float().clone()
        d = torch.as_tensor(self.d, device=y.device, dtype=torch.float32)
        delta = self.alpha * d
        y[:, -1, :] += delta
        self.applied += 1
        self.delta_norm = float(delta.norm().item())
        return traj.replace_first_tensor(out, y.to(h.dtype))

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


@torch.inference_mode()
def generate_one(model, processor, decoder_layers, batch, mapping, max_new_tokens,
                 L=None, direction=None, alpha=0.0):
    patch = None
    try:
        if direction is not None:
            patch = SingleLastPatch(decoder_layers[L], direction, alpha)
        text = base.generate_text(model, processor, batch, max_new_tokens=max_new_tokens)
        if patch is not None and patch.applied < 1:
            raise RuntimeError(f"L{L} last patch did not fire")
        pred, mode = parse_answer_letter(text, mapping)
        dn = patch.delta_norm if patch is not None else 0.0
        return pred, text, mode, dn
    finally:
        if patch is not None:
            patch.close()


def lm_head_decision_dirs_and_basis(model, tokenizer):
    W = model.lm_head.weight.detach().float().cpu().numpy().astype(np.float32)
    u = {}
    ids_out = {}
    for a in LETTERS:
        ids = tokenizer.encode(a, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Answer symbol {a!r} tokenizes to {ids}; expected one token")
        ids_out[a] = int(ids[0])
        u[a] = W[ids[0]]
    d = {}
    for a in LETTERS:
        d[a] = (u[a] - np.mean(np.stack([u[b] for b in LETTERS if b != a]), axis=0)).astype(np.float32)

    # The four centered decision vectors have rank <= 3.
    M = np.stack([d[a] for a in LETTERS], axis=1).astype(np.float64)  # [D,4]
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    tol = max(M.shape) * np.finfo(np.float64).eps * (S[0] if len(S) else 1.0)
    rank = int(np.sum(S > tol))
    Q = U[:, :rank].astype(np.float32)  # orthonormal answer-logit basis
    return d, Q, ids_out


def project_logit(v, Q):
    v = np.asarray(v, np.float32)
    if Q.size == 0:
        return np.zeros_like(v), v.copy()
    p = Q @ (Q.T @ v)
    return p.astype(np.float32), (v - p).astype(np.float32)


def fit_late_decomposition(train, train_map, q_by_sid, layers):
    """
    Returns:
      cells[L][r][a]
      rel_mu[L][r]
      ans_mu[L][a]
      rel_effect[L][r] = E[q|r] - grand
      ans_effect[L][a] = E[q|a] - grand
      axes_raw[L]["LR"/"AB"] = matched-answer relation contrast
    """
    cells = {L: {r: {} for r in REL} for L in layers}
    rel_mu = {L: {} for L in layers}
    ans_mu = {L: {} for L in layers}
    rel_effect = {L: {} for L in layers}
    ans_effect = {L: {} for L in layers}
    axes_raw = {L: {} for L in layers}
    counts = []

    for L in layers:
        all_x = [q_by_sid[int(m["sid"])][L] for m in train]
        grand = np.mean(np.stack(all_x), axis=0).astype(np.float32)

        for r in REL:
            xs = [q_by_sid[int(m["sid"])][L] for m in train if m["gt"] == r]
            rel_mu[L][r] = np.mean(np.stack(xs), axis=0).astype(np.float32)
            rel_effect[L][r] = (rel_mu[L][r] - grand).astype(np.float32)

        for a in LETTERS:
            xs = [q_by_sid[int(m["sid"])][L] for m in train
                  if train_map[int(m["sid"])][m["gt"]] == a]
            ans_mu[L][a] = np.mean(np.stack(xs), axis=0).astype(np.float32)
            ans_effect[L][a] = (ans_mu[L][a] - grand).astype(np.float32)

        for r in REL:
            for a in LETTERS:
                xs = [q_by_sid[int(m["sid"])][L] for m in train
                      if m["gt"] == r and train_map[int(m["sid"])][m["gt"]] == a]
                counts.append(dict(layer=L, relation=r, answer=a, n=len(xs)))
                if not xs:
                    raise RuntimeError(f"Empty train cell L{L} relation={r} answer={a}; increase --max-samples")
                cells[L][r][a] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        # Matched-answer contrasts: each term uses the SAME final answer letter.
        axes_raw[L]["LR"] = np.mean(np.stack([
            cells[L]["right"][a] - cells[L]["left"][a] for a in LETTERS
        ]), axis=0).astype(np.float32)
        axes_raw[L]["AB"] = np.mean(np.stack([
            cells[L]["above"][a] - cells[L]["below"][a] for a in LETTERS
        ]), axis=0).astype(np.float32)

    return cells, rel_mu, ans_mu, rel_effect, ans_effect, axes_raw, counts


def spatial_target_direction(axes_L, target_relation):
    if target_relation == "right":
        return axes_L["LR"]
    if target_relation == "left":
        return -axes_L["LR"]
    if target_relation == "above":
        return axes_L["AB"]
    if target_relation == "below":
        return -axes_L["AB"]
    raise ValueError(target_relation)


def nearest_cosine(x, centroids):
    labels = list(centroids)
    vals = [cosine_np(x, centroids[k]) for k in labels]
    vals2 = [-1e30 if not np.isfinite(v) else v for v in vals]
    return labels[int(np.argmax(vals2))]


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    if a.layers == "auto":
        layers = [26, 28, 30, 32, 34, 35] if a.model == "qwen-3b" else [20, 22, 24, 25, 26, 27]
    else:
        layers = parse_ints(a.layers)
    alphas = parse_floats(a.alphas)
    conditions = [x.strip() for x in a.conditions.split(",") if x.strip()]
    valid_conditions = {
        "spatial_raw_correct", "spatial_null_correct",
        "spatial_raw_wrong", "spatial_null_wrong",
        "letter_correct",
    }
    bad = [x for x in conditions if x not in valid_conditions]
    if bad:
        raise ValueError(f"Unknown conditions: {bad}")

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    err_path = out / "errors.jsonl"

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for r in records:
        sid = int(r.sid)
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
        ))

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, test = traj.stratified_split(meta, a.train_ratio, a.seed)
    test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)
    train_map = assign_relation_balanced_mappings(train, a.seed + 1001)
    test_map = assign_relation_balanced_mappings(test, a.seed + 2003)

    # Mapping balance audit.
    audit_rows = []
    for split_name, items, maps in (("train", train, train_map), ("test", test, test_map)):
        for r in REL:
            for aa in LETTERS:
                n = sum(1 for m in items if m["gt"] == r and maps[int(m["sid"])][r] == aa)
                audit_rows.append(dict(split=split_name, relation=r, answer=aa, n=n))
    traj.write_csv(out / "mapping_balance.csv", audit_rows)

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
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor)
        device = torch.device(a.device)
        decoder_layers, path = base.resolve_decoder_layers(model)

        for L in layers:
            if not (0 <= L < len(decoder_layers)):
                raise ValueError(f"L{L} invalid for model with {len(decoder_layers)} decoder layers")

        logit_dirs, logit_Q, answer_token_ids = lm_head_decision_dirs_and_basis(model, processor.tokenizer)

        print("=" * 118)
        print("COCO RANDOM-MAP — LATE SPATIAL/ANSWER DECONFOUND")
        print("=" * 118)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={path} | n_layers={len(decoder_layers)} | eval layers={layers}")
        print(f"train/test={len(train)}/{len(test)} | alphas={alphas}")
        print("q = Real-Gray prefix-last residual")
        print("spatial axis = matched-answer relation contrast")
        print("spatial_null = spatial axis projected orthogonal to A/B/C/D LM-head decision subspace")
        print(f"answer token ids={answer_token_ids}")
        print()

        # ------------------------------------------------------------------
        # Capture TRAIN + TEST q = Real-Gray last states.
        # ------------------------------------------------------------------
        q_by_sid = {}
        for split_name, items, maps in (("TRAIN", train, train_map), ("TEST", test, test_map)):
            for m in tqdm(items, desc=f"{split_name} Real-Gray last states"):
                sid = int(m["sid"]); mp = maps[sid]
                prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
                real = gray = rb = gb = None
                try:
                    real = make_real_image(rec_by_sid[sid])
                    gray = make_gray_image(real, a.gray_value)
                    rb = make_batch(processor, device, real, prompt)
                    gb = make_batch(processor, device, gray, prompt)
                    hr = capture_prefix_last(model, decoder_layers, rb, layers)
                    hg = capture_prefix_last(model, decoder_layers, gb, layers)
                    q_by_sid[sid] = {L: (hr[L] - hg[L]).astype(np.float32) for L in layers}
                except Exception as e:
                    traj.append_jsonl(err_path, dict(
                        phase=f"{split_name.lower()}_capture", sid=sid,
                        error=str(e), traceback=traceback.format_exc()
                    ))
                    raise
                finally:
                    for im in (real, gray):
                        if im is not None:
                            with contextlib.suppress(Exception):
                                im.close()
                    del rb, gb
                    gc.collect()

        cells, rel_mu, ans_mu, rel_effect, ans_effect, axes_raw, cell_counts = fit_late_decomposition(
            train, train_map, q_by_sid, layers
        )
        traj.write_csv(out / "train_cell_counts.csv", cell_counts)

        # ------------------------------------------------------------------
        # Geometry + logit-null directions.
        # ------------------------------------------------------------------
        axes_null = {L: {} for L in layers}
        geom_rows = []
        save_raw = {}
        save_null = {}
        save_answer = {}

        for L in layers:
            for axis in ("LR", "AB"):
                raw = axes_raw[L][axis]
                p, null = project_logit(raw, logit_Q)
                raw_norm = float(np.linalg.norm(raw))
                p_norm = float(np.linalg.norm(p))
                null_native_norm = float(np.linalg.norm(null))
                proj_frac = (p_norm * p_norm) / max(raw_norm * raw_norm, 1e-12)
                retained_frac = (null_native_norm * null_native_norm) / max(raw_norm * raw_norm, 1e-12)

                if a.null_norm_mode == "match_raw" and null_native_norm > 1e-12:
                    null_use = null * (raw_norm / null_native_norm)
                else:
                    null_use = null
                axes_null[L][axis] = null_use.astype(np.float32)

                row = dict(
                    layer=L, axis=axis,
                    raw_norm=raw_norm,
                    logit_projection_norm=p_norm,
                    logit_projection_energy_frac=proj_frac,
                    null_native_norm=null_native_norm,
                    null_retained_energy_frac=retained_frac,
                    null_used_norm=float(np.linalg.norm(null_use)),
                )
                for aa in LETTERS:
                    row[f"cos_raw_logit_{aa}"] = cosine_np(raw, logit_dirs[aa])
                    row[f"cos_null_logit_{aa}"] = cosine_np(null_use, logit_dirs[aa])
                geom_rows.append(row)
                save_raw[f"L{L}_{axis}"] = raw
                save_null[f"L{L}_{axis}"] = null_use

            for aa in LETTERS:
                save_answer[f"L{L}_{aa}"] = ans_effect[L][aa]

        traj.write_csv(out / "geometry.csv", geom_rows)
        np.savez_compressed(out / "matched_answer_spatial_raw.npz", **save_raw)
        np.savez_compressed(out / "matched_answer_spatial_logit_null.npz", **save_null)
        np.savez_compressed(out / "late_answer_main_effects.npz", **save_answer)

        print("\n" + "=" * 118)
        print("GEOMETRY")
        print("=" * 118)
        for row in geom_rows:
            print(
                f"L{row['layer']:02d} {row['axis']} | "
                f"logitProjEnergy={row['logit_projection_energy_frac']:.3f} | "
                f"nullRetained={row['null_retained_energy_frac']:.3f} | "
                f"norm raw/nullUse={row['raw_norm']:.3g}/{row['null_used_norm']:.3g}"
            )

        # ------------------------------------------------------------------
        # Held-out decode: relation vs answer identity from q_last.
        # ------------------------------------------------------------------
        decode_rows = []
        print("\n" + "=" * 118)
        print("HELD-OUT READOUT")
        print("=" * 118)
        for L in layers:
            rel_ok = ans_ok = 0
            n = 0
            per = []
            for m in test:
                sid = int(m["sid"])
                q = q_by_sid[sid][L]
                gt = m["gt"]
                aa = test_map[sid][gt]
                pr = nearest_cosine(q, rel_mu[L])
                pa = nearest_cosine(q, ans_mu[L])
                rel_ok += int(pr == gt)
                ans_ok += int(pa == aa)
                n += 1
                per.append(dict(
                    sid=sid, layer=L, gt_relation=gt, gt_answer=aa,
                    pred_relation=pr, pred_answer=pa,
                    relation_correct=(pr == gt), answer_correct=(pa == aa)
                ))
            rel_acc = rel_ok / max(n, 1)
            ans_acc = ans_ok / max(n, 1)
            decode_rows.append(dict(layer=L, N=n, relation_decode_acc=rel_acc, answer_decode_acc=ans_acc))
            print(f"L{L:02d} relationDecode={rel_acc:.4f} | answerDecode={ans_acc:.4f}")
        traj.write_csv(out / "decode_summary.csv", decode_rows)

        # ------------------------------------------------------------------
        # Baseline full generation.
        # ------------------------------------------------------------------
        baseline = []
        for m in tqdm(test, desc="BASELINE generation"):
            sid = int(m["sid"]); mp = test_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
            real = batch = None
            try:
                real = make_real_image(rec_by_sid[sid])
                batch = make_batch(processor, device, real, prompt)
                pred, text, mode, _ = generate_one(
                    model, processor, decoder_layers, batch, mp, a.max_new_tokens
                )
                gt = m["gt"]; opp = OPP[gt]
                correct_letter = mp[gt]; opp_letter = mp[opp]
                baseline.append(dict(
                    sid=sid, gt=gt, opposite=opp, mapping=mapping_string(mp),
                    correct_letter=correct_letter, opposite_letter=opp_letter,
                    baseline_pred=pred, baseline_text=text, parse_mode=mode,
                    baseline_correct=(pred == correct_letter),
                    baseline_opp_follow=(pred == opp_letter),
                ))
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                del batch
                gc.collect()

        traj.write_csv(out / "baseline.csv", baseline)
        N = len(baseline)
        Nc = sum(int(x["baseline_correct"]) for x in baseline)
        Nw = N - Nc
        base_acc = Nc / max(N, 1)
        print(f"\nBASELINE: N={N} acc={base_acc:.4f} correct={Nc} wrong={Nw}")

        b_by_sid = {int(x["sid"]): x for x in baseline}
        m_by_sid = {int(x["sid"]): x for x in test}

        # ------------------------------------------------------------------
        # Causal steering.
        # ------------------------------------------------------------------
        per_rows = []
        summary_rows = []

        for L in layers:
            for alpha in alphas:
                for cond in conditions:
                    rows = []
                    for b in tqdm(baseline, desc=f"L{L} a{alpha:g} {cond}", leave=False):
                        sid = int(b["sid"])
                        m = m_by_sid[sid]
                        mp = test_map[sid]
                        gt = m["gt"]; opp = OPP[gt]

                        if cond == "spatial_raw_correct":
                            target_rel = gt
                            target_letter = mp[target_rel]
                            d = spatial_target_direction(axes_raw[L], target_rel)
                        elif cond == "spatial_null_correct":
                            target_rel = gt
                            target_letter = mp[target_rel]
                            d = spatial_target_direction(axes_null[L], target_rel)
                        elif cond == "spatial_raw_wrong":
                            target_rel = opp
                            target_letter = mp[target_rel]
                            d = spatial_target_direction(axes_raw[L], target_rel)
                        elif cond == "spatial_null_wrong":
                            target_rel = opp
                            target_letter = mp[target_rel]
                            d = spatial_target_direction(axes_null[L], target_rel)
                        elif cond == "letter_correct":
                            target_rel = gt
                            target_letter = mp[gt]
                            d = ans_effect[L][target_letter]
                        else:
                            raise ValueError(cond)

                        prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
                        real = batch = None
                        try:
                            real = make_real_image(rec_by_sid[sid])
                            batch = make_batch(processor, device, real, prompt)
                            pred, text, mode, dn = generate_one(
                                model, processor, decoder_layers, batch, mp,
                                a.max_new_tokens, L=L, direction=d, alpha=alpha
                            )
                            correct_letter = mp[gt]
                            row = dict(
                                sid=sid, layer=L, alpha=alpha, condition=cond,
                                gt=gt, target_relation=target_rel,
                                correct_letter=correct_letter, target_letter=target_letter,
                                mapping=mapping_string(mp),
                                baseline_pred=b["baseline_pred"],
                                baseline_correct=bool(b["baseline_correct"]),
                                baseline_target_follow=(b["baseline_pred"] == target_letter),
                                patched_pred=pred, patched_text=text, parse_mode=mode,
                                patched_correct=(pred == correct_letter),
                                target_follow=(pred == target_letter),
                                changed=(pred != b["baseline_pred"]),
                                W2C=(not b["baseline_correct"] and pred == correct_letter),
                                C2W=(b["baseline_correct"] and pred != correct_letter),
                                delta_norm=dn,
                            )
                            rows.append(row); per_rows.append(row)
                        except Exception as e:
                            traj.append_jsonl(err_path, dict(
                                phase=cond, sid=sid, layer=L, alpha=alpha,
                                error=str(e), traceback=traceback.format_exc()
                            ))
                            raise
                        finally:
                            if real is not None:
                                with contextlib.suppress(Exception): real.close()
                            del batch
                            gc.collect()

                    n = len(rows)
                    acc = sum(int(r["patched_correct"]) for r in rows) / max(n, 1)
                    w2c = sum(int(r["W2C"]) for r in rows)
                    c2w = sum(int(r["C2W"]) for r in rows)
                    tf = sum(int(r["target_follow"]) for r in rows) / max(n, 1)
                    btf = sum(int(r["baseline_target_follow"]) for r in rows) / max(n, 1)
                    changed = sum(int(r["changed"]) for r in rows) / max(n, 1)
                    parsed = sum(int(r["patched_pred"] is not None) for r in rows) / max(n, 1)
                    mdn = float(np.mean([r["delta_norm"] for r in rows])) if rows else 0.0

                    s = dict(
                        layer=L, alpha=alpha, condition=cond, N=n,
                        baseline_acc=base_acc, patched_acc=acc, delta_acc=acc-base_acc,
                        N_baseline_correct=Nc, N_baseline_wrong=Nw,
                        W2C=w2c, W2C_rate=w2c/max(Nw,1),
                        C2W=c2w, C2W_rate=c2w/max(Nc,1),
                        preserve=1.0-c2w/max(Nc,1),
                        baseline_target_follow=btf, target_follow=tf,
                        delta_target_follow=tf-btf,
                        changed_rate=changed, parsed_rate=parsed,
                        mean_delta_norm=mdn,
                    )
                    summary_rows.append(s)
                    print(
                        f"L{L:02d} a={alpha:g} {cond:20s} | "
                        f"acc={acc:.4f} ({acc-base_acc:+.4f}) | "
                        f"W2C={w2c}/{Nw}={s['W2C_rate']:.3f} | "
                        f"C2W={c2w}/{Nc}={s['C2W_rate']:.3f} | "
                        f"targetFollow={tf:.3f} (base={btf:.3f}, delta={tf-btf:+.3f}) | "
                        f"changed={changed:.3f}"
                    )

                    traj.write_csv(out / "per_sample.csv", per_rows)
                    traj.write_csv(out / "steering_summary.csv", summary_rows)

        (out / "metadata.json").write_text(json.dumps({
            "model_alias": a.model,
            "repo_id": spec.repo_id,
            "layers": layers,
            "train_ratio": a.train_ratio,
            "seed": a.seed,
            "max_samples": a.max_samples,
            "eval_max_samples": a.eval_max_samples,
            "gray_value": a.gray_value,
            "null_norm_mode": a.null_norm_mode,
            "q_definition": "h_real_last - h_gray_last",
            "spatial_LR": "mean_answer [mu(right,answer)-mu(left,answer)]",
            "spatial_AB": "mean_answer [mu(above,answer)-mu(below,answer)]",
            "logit_null": "(I-P_span(centered A/B/C/D lm_head directions)) spatial",
            "patch": "single late layer, prefix-last only",
            "conditions": conditions,
        }, indent=2), encoding="utf-8")

        print("\nSaved:", out)
        print("Main readouts:")
        print("  geometry.csv: how much of matched-answer spatial direction lies in answer-logit subspace")
        print("  decode_summary.csv: relation vs answer information in q_last")
        print("  steering_summary.csv: does logit-null spatial direction causally follow the CURRENT random mapping?")

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
