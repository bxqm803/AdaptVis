
# -*- coding: utf-8 -*-
"""Qwen2.5-VL-7B + COCO random-map: late answer-writer onset + causal pullback.

Purpose
-------
We already know that late prefix-last writers are strongly answer/logit aligned.
This script treats that late writer as the *target effect* and asks:

    What change to an earlier hidden state would most efficiently make the
    model's own downstream computation move the late last-token state along
    that writer direction?

For a chosen late writer layer L* and earlier source layer l:

    score_y = < h[L*, last],  unit(s_y[L*]) >

and the first-order upstream control direction is the pullback

    g_l = d score_y / d h_l = J_{l->L*}^T unit(s_y[L*]).

For object-pair steering, we collapse token gradients to the same symmetric
subject/reference edit used in prior experiments:

    g_pair = 0.5 * (mean grad_subject - mean grad_reference).

Because answer mappings are randomized per sample, we can test whether pullbacks
for DIFFERENT output letters that encode the SAME spatial relation converge onto
one common relation direction upstream.

The script does four things:
1) Fit late answer-letter writers s_A/s_B/s_C/s_D from Real-Gray prefix-last
   states under randomized relation->A/B/C/D mappings.
2) Sweep single-layer last-token steering to find the earliest late layer where
   the answer writer has a strong behavioral effect (writer onset).
3) Pull that writer objective backward to earlier object-pair states using
   autograd, and compare the resulting g_pair against:
      - the mapping-invariant semantic relation direction at that source layer;
      - the source-layer answer-letter direction;
      - the direct LM-head answer-margin direction.
4) Optionally validate the pullback by actually editing the earlier object pair
   along g_pair and checking whether the chosen late writer score increases.

Key readouts
------------
- onset_summary.csv:
    earliest layer where a single late answer writer strongly changes behavior.
- pullback_summary.csv:
    mean cosine(pullback, semantic relation direction) vs answer-letter/logit
    directions at each earlier layer.
- mapping_invariance.csv:
    for one target relation (e.g. RIGHT), compare pullbacks when RIGHT maps to
    A vs B vs C vs D. High cross-letter cosine upstream is the key signal.

Quick example
-------------
CUDA_VISIBLE_DEVICES=0 python -u qwen7b_coco_randmap_late_writer_pullback_relation_control.py \
  --writer-candidates 20,22,24,25,26,27 \
  --source-layers 10,14,18,22 \
  --max-samples 160 --onset-eval-max 32 --pullback-eval-max 24 \
  --output-dir output/qwen7b_coco_randmap_writer_pullback_quick --overwrite

If you already know the writer onset and want to skip the generation sweep:

CUDA_VISIBLE_DEVICES=0 python -u qwen7b_coco_randmap_late_writer_pullback_relation_control.py \
  --writer-layer 32 --source-layers 10,14,18,22 \
  --max-samples 160 --pullback-eval-max 32 \
  --output-dir output/qwen7b_coco_randmap_writer_pullback_L24 --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import math
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
    p.add_argument("--model", default="qwen-7b", choices=["qwen-7b"])
    p.add_argument("--writer-candidates", default="20,22,24,25,26,27",
                   help="Late layers for SINGLE-LAYER writer-onset sweep.")
    p.add_argument("--writer-layer", type=int, default=-1,
                   help="If >=0, skip onset generation sweep and use this late target layer.")
    p.add_argument("--source-layers", default="10,14,18,22",
                   help="Earlier layers to pull the selected late writer back to.")
    p.add_argument("--writer-alpha", type=float, default=1.0,
                   help="Single-layer writer strength used in onset behavioral sweep.")
    p.add_argument("--onset-delta", type=float, default=0.10,
                   help="Earliest layer with >= this accuracy gain is selected; otherwise best layer.")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--max-samples", type=int, default=160, help="0 = all before split")
    p.add_argument("--onset-eval-max", type=int, default=32,
                   help="Held-out examples for onset generation sweep; 0 = all test.")
    p.add_argument("--pullback-eval-max", type=int, default=24,
                   help="Held-out examples used for autograd pullback; 0 = all test.")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--pullback-targets", default="correct,opposite",
                   help="Comma-separated: correct,opposite")
    p.add_argument("--validate-pullback", action="store_true",
                   help="Actually edit source pair along pullback and measure late writer-score gain.")
    p.add_argument("--validate-scale", type=float, default=1.0,
                   help="Validation edit norm = this * norm(source semantic target-vs-opposite direction).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted({int(x.strip().upper().replace("L", "")) for x in s.split(",") if x.strip()})


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def unit_np(x):
    x = np.asarray(x, np.float32)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-12)


def make_real_image(record):
    im = base.record_image(record)
    if hasattr(im, "convert"):
        im = im.convert("RGB")
    return im


def make_gray_image(real_image, value: int):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def make_batch(processor, device, image, question_text):
    return base.make_question_batch(
        processor=processor, image=image, question_text=question_text, device=device
    )


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


def assign_balanced_random_mappings(items: List[dict], seed: int) -> Dict[int, Dict[str, str]]:
    perms = list(itertools.permutations(LETTERS))
    rng = random.Random(seed)
    rng.shuffle(perms)
    order = list(range(len(items)))
    rng.shuffle(order)
    out = {}
    for rank, idx in enumerate(order):
        perm = perms[rank % len(perms)]
        out[int(items[idx]["sid"])] = {r: a for r, a in zip(REL, perm)}
    return out


def parse_answer_letter(text: str, rel_to_letter: Dict[str, str]) -> Tuple[str | None, str]:
    t = str(text).strip(); up = t.upper()
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


class LastStateCapture:
    def __init__(self, decoder_layers, layers_req):
        self.states = {}
        self.handles = [decoder_layers[L].register_forward_hook(self._make_hook(L)) for L in layers_req]

    def _make_hook(self, L):
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
        kwargs = dict(batch); kwargs["use_cache"] = False
        _ = model(**kwargs)
        missing = [L for L in layers_req if L not in cap.states]
        if missing:
            raise RuntimeError(f"Did not capture prefix-last at layers {missing}")
        return cap.states
    finally:
        cap.close()


class SingleLastPatch:
    def __init__(self, layer, direction, alpha):
        self.d = np.asarray(direction, np.float32)
        self.alpha = float(alpha)
        self.applied = 0
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        if h.ndim != 3 or int(h.shape[1]) <= 1:  # prefill only
            return out
        y = h.float().clone()
        d = torch.as_tensor(self.d, device=y.device, dtype=torch.float32)
        y[:, -1, :] += self.alpha * d
        self.applied += 1
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
            raise RuntimeError(f"L{L} last-token patch did not fire")
        pred, mode = parse_answer_letter(text, mapping)
        return pred, text, mode
    finally:
        if patch is not None:
            patch.close()


def fit_late_letter_writers(train_meta, q_by_sid, mappings, writer_layers):
    out = {L: {} for L in writer_layers}
    for L in writer_layers:
        mu = {}
        for a in LETTERS:
            xs = [q_by_sid[int(m["sid"])][L] for m in train_meta
                  if mappings[int(m["sid"])][m["gt"]] == a]
            if not xs:
                raise RuntimeError(f"No train states for answer {a} at L{L}")
            mu[a] = np.mean(np.stack(xs), axis=0).astype(np.float32)
        common = np.mean(np.stack([mu[a] for a in LETTERS]), axis=0)
        for a in LETTERS:
            out[L][a] = (mu[a] - common).astype(np.float32)
    return out


def fit_middle_main_effects(train_meta, pair_by_sid, mappings, source_layers):
    rel_mu = {L: {} for L in source_layers}
    letter_mu = {L: {} for L in source_layers}
    for L in source_layers:
        for r in REL:
            xs = [pair_by_sid[int(m["sid"])][L] for m in train_meta if m["gt"] == r]
            rel_mu[L][r] = np.mean(np.stack(xs), axis=0).astype(np.float32)
        for a in LETTERS:
            xs = [pair_by_sid[int(m["sid"])][L] for m in train_meta
                  if mappings[int(m["sid"])][m["gt"]] == a]
            letter_mu[L][a] = np.mean(np.stack(xs), axis=0).astype(np.float32)
    return rel_mu, letter_mu


def centered_letter_direction(letter_mu_L, a):
    others = [letter_mu_L[b] for b in LETTERS if b != a]
    return (letter_mu_L[a] - np.mean(np.stack(others), axis=0)).astype(np.float32)


def lm_head_decision_dirs(model, tokenizer):
    W = model.lm_head.weight.detach().float().cpu()
    u = {}
    for a in LETTERS:
        ids = tokenizer.encode(a, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Answer symbol {a!r} tokenizes to {ids}; expected one token")
        u[a] = W[ids[0]].numpy().astype(np.float32)
    d = {}
    for a in LETTERS:
        others = [u[b] for b in LETTERS if b != a]
        d[a] = (u[a] - np.mean(np.stack(others), axis=0)).astype(np.float32)
    return d


class GradSourceHook:
    """Cut graph at one source layer output and make that state the leaf variable."""
    def __init__(self, layer):
        self.tensor = None
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        y = h.detach().clone().requires_grad_(True)
        self.tensor = y
        return traj.replace_first_tensor(out, y)

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


class GradTargetHook:
    def __init__(self, layer):
        self.tensor = None
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        self.tensor = h
        return out

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


def pullback_pair_grad(model, decoder_layers, batch, source_layer, writer_layer,
                       sub_pos, ref_pos, writer_vecs: Dict[str, np.ndarray]):
    if source_layer >= writer_layer:
        raise ValueError(f"source L{source_layer} must be before writer L{writer_layer}")
    src = GradSourceHook(decoder_layers[source_layer])
    tgt = GradTargetHook(decoder_layers[writer_layer])
    try:
        kwargs = dict(batch); kwargs["use_cache"] = False
        _ = model(**kwargs)
        if src.tensor is None or tgt.tensor is None:
            raise RuntimeError("Gradient hooks did not capture source/target tensors")
        hlast = tgt.tensor[0, -1, :].float()
        results = {}
        keys = list(writer_vecs)
        for j, key in enumerate(keys):
            w_np = unit_np(writer_vecs[key])
            w = torch.as_tensor(w_np, device=hlast.device, dtype=torch.float32)
            score = torch.dot(hlast, w)
            grad = torch.autograd.grad(
                score, src.tensor,
                retain_graph=(j < len(keys) - 1),
                create_graph=False, allow_unused=False,
            )[0][0].detach().float().cpu().numpy()
            gs = np.mean(grad[list(sub_pos)], axis=0)
            gr = np.mean(grad[list(ref_pos)], axis=0)
            gpair = (0.5 * (gs - gr)).astype(np.float32)
            results[key] = dict(
                score=float(score.detach().item()),
                g_pair=gpair,
                source_grad_norm=float(np.linalg.norm(gpair)),
                target_state=hlast.detach().cpu().numpy().astype(np.float32),
            )
        return results
    finally:
        src.close(); tgt.close()


class PairPatchAndTargetCapture:
    def __init__(self, source_layer, target_layer, sub_pos, ref_pos, delta):
        self.sub_pos = tuple(map(int, sub_pos)); self.ref_pos = tuple(map(int, ref_pos))
        self.delta = np.asarray(delta, np.float32)
        self.target = None
        self.applied = 0
        self.hs = source_layer.register_forward_hook(self.source_hook)
        self.ht = target_layer.register_forward_hook(self.target_hook)

    def source_hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        y = h.float().clone()
        d = torch.as_tensor(self.delta, device=y.device, dtype=torch.float32)
        for p in self.sub_pos:
            y[:, p, :] += 0.5 * d
        for p in self.ref_pos:
            y[:, p, :] -= 0.5 * d
        self.applied += 1
        return traj.replace_first_tensor(out, y.to(h.dtype))

    def target_hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        self.target = h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
        return out

    def close(self):
        with contextlib.suppress(Exception): self.hs.remove()
        with contextlib.suppress(Exception): self.ht.remove()


@torch.inference_mode()
def validate_pullback_edit(model, decoder_layers, batch, source_layer, writer_layer,
                           sub_pos, ref_pos, delta, writer_vec, baseline_target_state):
    cap = PairPatchAndTargetCapture(
        decoder_layers[source_layer], decoder_layers[writer_layer],
        sub_pos, ref_pos, delta,
    )
    try:
        kwargs = dict(batch); kwargs["use_cache"] = False
        _ = model(**kwargs)
        if cap.applied < 1 or cap.target is None:
            raise RuntimeError("Pullback validation patch/capture failed")
        w = unit_np(writer_vec)
        b = np.asarray(baseline_target_state, np.float32)
        p = cap.target
        db = float(np.dot(b, w)); dp = float(np.dot(p, w))
        shift = (p - b).astype(np.float32)
        return dict(
            baseline_writer_score=db,
            patched_writer_score=dp,
            delta_writer_score=dp-db,
            late_shift_norm=float(np.linalg.norm(shift)),
            late_shift_cos_writer=cosine_np(shift, w),
        )
    finally:
        cap.close()


def aggregate_pullback(rows):
    out = []
    keys = sorted({(r["source_layer"], r["target_condition"]) for r in rows})
    for L, cond in keys:
        rs = [r for r in rows if r["source_layer"] == L and r["target_condition"] == cond]
        def m(k):
            vals = [float(r[k]) for r in rs if np.isfinite(float(r[k]))]
            return float(np.mean(vals)) if vals else float("nan")
        def sd(k):
            vals = [float(r[k]) for r in rs if np.isfinite(float(r[k]))]
            return float(np.std(vals)) if vals else float("nan")
        out.append(dict(
            source_layer=L, target_condition=cond, N=len(rs),
            mean_cos_semantic=m("cos_semantic"), std_cos_semantic=sd("cos_semantic"),
            mean_cos_middle_letter=m("cos_middle_letter"),
            mean_cos_output_logit=m("cos_output_logit"),
            mean_grad_norm=m("pullback_norm"),
            frac_cos_semantic_pos=float(np.mean([r["cos_semantic"] > 0 for r in rs])) if rs else float("nan"),
            mean_delta_writer_score=m("validate_delta_writer_score") if any("validate_delta_writer_score" in r for r in rs) else float("nan"),
            mean_late_shift_cos_writer=m("validate_late_shift_cos_writer") if any("validate_late_shift_cos_writer" in r for r in rs) else float("nan"),
        ))
    return out


def mapping_invariance_rows(rows, vectors):
    """Compare pullback centroids for same relation but different answer letters."""
    out = []
    for L in sorted({r["source_layer"] for r in rows}):
        for cond in sorted({r["target_condition"] for r in rows}):
            for rel in REL:
                means = {}
                counts = {}
                for a in LETTERS:
                    vs = [vectors[i] for i, r in enumerate(rows)
                          if r["source_layer"] == L and r["target_condition"] == cond
                          and r["target_relation"] == rel and r["target_letter"] == a]
                    if vs:
                        # normalize per sample before averaging so large gradients do not dominate
                        means[a] = np.mean(np.stack([unit_np(v) for v in vs]), axis=0).astype(np.float32)
                        counts[a] = len(vs)
                letters = sorted(means)
                pair_cos = []
                for i in range(len(letters)):
                    for j in range(i+1, len(letters)):
                        pair_cos.append(cosine_np(means[letters[i]], means[letters[j]]))
                if len(letters) >= 2:
                    out.append(dict(
                        source_layer=L, target_condition=cond, target_relation=rel,
                        letters_present="".join(letters),
                        counts="|".join(f"{a}:{counts[a]}" for a in letters),
                        mean_cross_letter_cos=float(np.nanmean(pair_cos)),
                        min_cross_letter_cos=float(np.nanmin(pair_cos)),
                        max_cross_letter_cos=float(np.nanmax(pair_cos)),
                        n_pairs=len(pair_cos),
                    ))
    return out



def relation_specificity_control(rows, vectors):
    """Balanced relation centroids and 4x4 between-relation cosine matrices.

    For each relation, first build a centroid separately for each output letter from
    unit-normalized per-sample pullbacks, then average the available letter centroids
    equally. This avoids a relation centroid being driven by one answer letter.
    """
    results = []
    for L in sorted({r["source_layer"] for r in rows}):
        for cond in sorted({r["target_condition"] for r in rows}):
            rel_centroids = {}
            rel_letter_counts = {}
            for rel in REL:
                letter_centroids = []
                counts = {}
                for a in LETTERS:
                    vs = [vectors[i] for i, r in enumerate(rows)
                          if r["source_layer"] == L and r["target_condition"] == cond
                          and r["target_relation"] == rel and r["target_letter"] == a]
                    if vs:
                        c = np.mean(np.stack([unit_np(v) for v in vs]), axis=0).astype(np.float32)
                        letter_centroids.append(unit_np(c))
                        counts[a] = len(vs)
                if letter_centroids:
                    rc = np.mean(np.stack(letter_centroids), axis=0).astype(np.float32)
                    rel_centroids[rel] = unit_np(rc)
                    rel_letter_counts[rel] = counts

            # 4x4 matrix rows
            for r1 in REL:
                if r1 not in rel_centroids:
                    continue
                row = {
                    "source_layer": L,
                    "target_condition": cond,
                    "relation": r1,
                    "counts": "|".join(f"{a}:{rel_letter_counts[r1].get(a,0)}" for a in LETTERS),
                }
                for r2 in REL:
                    row[f"cos_{r2}"] = (
                        cosine_np(rel_centroids[r1], rel_centroids[r2])
                        if r2 in rel_centroids else float("nan")
                    )
                results.append(row)
    return results


def main():
    a = parse_args()
    writer_candidates = parse_ints(a.writer_candidates)
    source_layers = parse_ints(a.source_layers)
    targets = [x.strip().lower() for x in a.pullback_targets.split(",") if x.strip()]
    for x in targets:
        if x not in ("correct", "opposite"):
            raise ValueError(f"Unknown pullback target {x}")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    err_path = outdir / "errors.jsonl"

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
        meta.append(dict(sid=sid, gt=gt, subject=str(p["subject"]), reference=str(p["reference"])))
    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, test_all = traj.stratified_split(meta, a.train_ratio, a.seed)
    onset_test = traj.stratified_cap(test_all, a.onset_eval_max, a.seed + 1)
    pullback_test = traj.stratified_cap(test_all, a.pullback_eval_max, a.seed + 2)
    train_map = assign_balanced_random_mappings(train, a.seed + 1001)
    test_map = assign_balanced_random_mappings(test_all, a.seed + 2003)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)
    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name), low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code, device_map={"": a.device},
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
        token_map = base.relation_token_variants(processor.tokenizer)

        all_writer_layers = sorted(set(writer_candidates + ([a.writer_layer] if a.writer_layer >= 0 else [])))
        for L in sorted(set(all_writer_layers + source_layers)):
            if not (0 <= L < len(decoder_layers)):
                raise ValueError(f"L{L} invalid for {len(decoder_layers)} decoder layers")

        print("=" * 116)
        print("QWEN7B COCO RANDOM-MAP — LATE ANSWER-WRITER ONSET + CAUSAL PULLBACK")
        print("=" * 116)
        print(f"decoder={path}")
        print(f"train/onset/pullback={len(train)}/{len(onset_test)}/{len(pullback_test)}")
        print(f"writer candidates={writer_candidates} | source layers={source_layers}")
        print("late target = randomized answer-letter writer from Real-Gray prefix-last")
        print("pullback = J^T * unit(late writer), collapsed to symmetric subject-reference edit")
        print()

        # ------------------------------------------------------------------
        # 1) TRAIN: late answer writers + source-layer semantic/letter geometry
        # ------------------------------------------------------------------
        q_by_sid = {}
        pair_by_sid = {}
        for m in tqdm(train, desc="TRAIN writers + middle pair states"):
            sid = int(m["sid"]); mp = train_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
            real = gray = rb = gb = None
            try:
                real = make_real_image(rec_by_sid[sid]); gray = make_gray_image(real, a.gray_value)
                rb = make_batch(processor, device, real, prompt)
                gb = make_batch(processor, device, gray, prompt)
                clean = traj.clean_forward(
                    base, model, processor, decoder_layers, rb,
                    m["subject"], m["reference"], source_layers,
                    a.object_state, token_map,
                )
                pair_by_sid[sid] = clean["states"]
                hr = capture_prefix_last(model, decoder_layers, rb, all_writer_layers)
                hg = capture_prefix_last(model, decoder_layers, gb, all_writer_layers)
                q_by_sid[sid] = {L: (hr[L] - hg[L]).astype(np.float32) for L in all_writer_layers}
            except Exception as e:
                traj.append_jsonl(err_path, {"phase": "train", "sid": sid, "error": str(e),
                                             "traceback": traceback.format_exc()})
                raise
            finally:
                for im in (real, gray):
                    if im is not None:
                        with contextlib.suppress(Exception): im.close()
                del rb, gb
                gc.collect()

        letter_writers = fit_late_letter_writers(train, q_by_sid, train_map, all_writer_layers)
        rel_mu, middle_letter_mu = fit_middle_main_effects(train, pair_by_sid, train_map, source_layers)
        logit_dirs = lm_head_decision_dirs(model, processor.tokenizer)

        np.savez_compressed(
            outdir / "late_answer_letter_writers.npz",
            **{f"L{L}_{a0}": letter_writers[L][a0] for L in all_writer_layers for a0 in LETTERS}
        )
        np.savez_compressed(
            outdir / "middle_relation_centroids.npz",
            **{f"L{L}_{r}": rel_mu[L][r] for L in source_layers for r in REL}
        )

        # ------------------------------------------------------------------
        # 2) SINGLE-LAYER writer onset sweep (or fixed writer layer)
        # ------------------------------------------------------------------
        onset_rows = []
        if a.writer_layer < 0:
            baseline_cache = {}
            for m in tqdm(onset_test, desc="ONSET baseline"):
                sid = int(m["sid"]); mp = test_map[sid]
                real = batch = None
                try:
                    prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
                    real = make_real_image(rec_by_sid[sid])
                    batch = make_batch(processor, device, real, prompt)
                    pred, text, mode = generate_one(
                        model, processor, decoder_layers, batch, mp, a.max_new_tokens
                    )
                    baseline_cache[sid] = dict(pred=pred, correct=(pred == mp[m["gt"]]), text=text, mode=mode)
                finally:
                    if real is not None:
                        with contextlib.suppress(Exception): real.close()
                    del batch
                    gc.collect()

            base_acc = np.mean([v["correct"] for v in baseline_cache.values()]) if baseline_cache else float("nan")
            print(f"ONSET baseline: N={len(baseline_cache)} acc={base_acc:.4f}")

            for L in writer_candidates:
                rows_L = []
                for m in tqdm(onset_test, desc=f"ONSET L{L}", leave=False):
                    sid = int(m["sid"]); mp = test_map[sid]; target_letter = mp[m["gt"]]
                    real = batch = None
                    try:
                        prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
                        real = make_real_image(rec_by_sid[sid])
                        batch = make_batch(processor, device, real, prompt)
                        pred, text, mode = generate_one(
                            model, processor, decoder_layers, batch, mp, a.max_new_tokens,
                            L=L, direction=letter_writers[L][target_letter], alpha=a.writer_alpha,
                        )
                        b = baseline_cache[sid]
                        rows_L.append(dict(
                            sid=sid, layer=L, target_letter=target_letter,
                            baseline_pred=b["pred"], baseline_correct=b["correct"],
                            pred=pred, correct=(pred == target_letter),
                            changed=(pred != b["pred"]),
                            W2C=((not b["correct"]) and pred == target_letter),
                            C2W=(b["correct"] and pred != target_letter),
                        ))
                    finally:
                        if real is not None:
                            with contextlib.suppress(Exception): real.close()
                        del batch
                        gc.collect()
                n = len(rows_L); Nc = sum(int(r["baseline_correct"]) for r in rows_L); Nw = n - Nc
                acc = sum(int(r["correct"]) for r in rows_L) / max(n, 1)
                w2c = sum(int(r["W2C"]) for r in rows_L); c2w = sum(int(r["C2W"]) for r in rows_L)
                s = dict(layer=L, N=n, baseline_acc=base_acc, patched_acc=acc,
                         delta_acc=acc-base_acc, W2C=w2c, W2C_rate=w2c/max(Nw,1),
                         C2W=c2w, C2W_rate=c2w/max(Nc,1),
                         changed_rate=np.mean([r["changed"] for r in rows_L]) if rows_L else 0.0)
                onset_rows.append(s)
                print(f"ONSET L{L:02d} | acc={acc:.4f} ({acc-base_acc:+.4f}) | "
                      f"W2C={w2c}/{Nw}={s['W2C_rate']:.3f} | C2W={c2w}/{Nc}={s['C2W_rate']:.3f}")

            eligible = [s for s in onset_rows if s["delta_acc"] >= a.onset_delta]
            if eligible:
                writer_layer = min(s["layer"] for s in eligible)
                why = f"earliest delta_acc >= {a.onset_delta:g}"
            else:
                best = max(onset_rows, key=lambda s: (s["delta_acc"], -s["layer"]))
                writer_layer = int(best["layer"])
                why = "no layer crossed threshold; selected best delta_acc"
            traj.write_csv(outdir / "onset_summary.csv", onset_rows)
            print(f"SELECTED writer onset L{writer_layer}: {why}")
        else:
            writer_layer = int(a.writer_layer)
            print(f"Using fixed writer layer L{writer_layer}; onset sweep skipped")

        valid_sources = [L for L in source_layers if L < writer_layer]
        if not valid_sources:
            raise RuntimeError(f"No source layer is earlier than selected writer layer L{writer_layer}")

        # Freeze parameters. Gradients are created only from the detached source activation onward.
        for p in model.parameters():
            p.requires_grad_(False)

        # ------------------------------------------------------------------
        # 3) PULLBACK: late writer objective -> earlier object-pair direction
        # ------------------------------------------------------------------
        pull_rows = []
        pull_vecs = []  # same order as pull_rows, only for mapping-invariance aggregation

        for m in tqdm(pullback_test, desc=f"PULLBACK to L<{writer_layer}"):
            sid = int(m["sid"]); mp = test_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
            real = batch = None
            try:
                real = make_real_image(rec_by_sid[sid])
                batch = make_batch(processor, device, real, prompt)
                # Positions only; same prompt/image as pullback forward.
                with torch.inference_mode():
                    clean = traj.clean_forward(
                        base, model, processor, decoder_layers, batch,
                        m["subject"], m["reference"], valid_sources,
                        a.object_state, token_map,
                    )
                sp = tuple(clean["subject_positions"]); rp = tuple(clean["reference_positions"])

                target_specs = {}
                if "correct" in targets:
                    target_specs["correct"] = m["gt"]
                if "opposite" in targets:
                    target_specs["opposite"] = OPP[m["gt"]]
                writer_vecs = {cond: letter_writers[writer_layer][mp[rel]]
                               for cond, rel in target_specs.items()}

                for L in valid_sources:
                    # Need grad mode despite model.eval().
                    with torch.enable_grad():
                        pb = pullback_pair_grad(
                            model, decoder_layers, batch, L, writer_layer,
                            sp, rp, writer_vecs,
                        )

                    for cond, target_rel in target_specs.items():
                        target_letter = mp[target_rel]
                        g = pb[cond]["g_pair"]
                        # Target-vs-opposite semantic relation direction at source layer.
                        d_sem = (rel_mu[L][target_rel] - rel_mu[L][OPP[target_rel]]).astype(np.float32)
                        d_mid_letter = centered_letter_direction(middle_letter_mu[L], target_letter)
                        d_logit = logit_dirs[target_letter]

                        row = dict(
                            sid=sid, source_layer=L, writer_layer=writer_layer,
                            target_condition=cond, gt=m["gt"], target_relation=target_rel,
                            target_letter=target_letter,
                            mapping="|".join(f"{r}:{mp[r]}" for r in REL),
                            writer_score=pb[cond]["score"], pullback_norm=float(np.linalg.norm(g)),
                            semantic_dir_norm=float(np.linalg.norm(d_sem)),
                            cos_semantic=cosine_np(g, d_sem),
                            cos_middle_letter=cosine_np(g, d_mid_letter),
                            cos_output_logit=cosine_np(g, d_logit),
                        )

                        if a.validate_pullback:
                            edit_norm = a.validate_scale * float(np.linalg.norm(d_sem))
                            delta = unit_np(g) * edit_norm
                            val = validate_pullback_edit(
                                model, decoder_layers, batch, L, writer_layer,
                                sp, rp, delta, writer_vecs[cond], pb[cond]["target_state"],
                            )
                            row.update({f"validate_{k}": v for k, v in val.items()})

                        pull_rows.append(row)
                        pull_vecs.append(g.astype(np.float32))
            except Exception as e:
                traj.append_jsonl(err_path, {"phase": "pullback", "sid": sid, "error": str(e),
                                             "traceback": traceback.format_exc()})
                raise
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                del batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        traj.write_csv(outdir / "pullback_per_sample.csv", pull_rows)
        summary = aggregate_pullback(pull_rows)
        traj.write_csv(outdir / "pullback_summary.csv", summary)
        inv = mapping_invariance_rows(pull_rows, pull_vecs)
        traj.write_csv(outdir / "mapping_invariance.csv", inv)
        relctl = relation_specificity_control(pull_rows, pull_vecs)
        traj.write_csv(outdir / "relation_specificity_control.csv", relctl)

        print("\n" + "=" * 116)
        print(f"PULLBACK SUMMARY | late writer target L{writer_layer}")
        print("=" * 116)
        for s in summary:
            msg = (
                f"L{s['source_layer']:02d} {s['target_condition']:8s} | "
                f"cos(spatial)={s['mean_cos_semantic']:+.4f} | "
                f"cos(mid-letter)={s['mean_cos_middle_letter']:+.4f} | "
                f"cos(output-logit)={s['mean_cos_output_logit']:+.4f} | "
                f"|g|={s['mean_grad_norm']:.4g}"
            )
            if a.validate_pullback:
                msg += (f" | induced writer Δ={s['mean_delta_writer_score']:+.4f} "
                        f"lateShiftCos={s['mean_late_shift_cos_writer']:+.4f}")
            print(msg)

        if inv:
            print("\nRELATION-SPECIFICITY CONTROL: between-relation cosine of balanced pullback centroids")
        for L in valid_sources:
            for cond in targets:
                sub = [r for r in relctl if r["source_layer"] == L and r["target_condition"] == cond]
                if not sub:
                    continue
                print(f"L{L} {cond}")
                print("          " + "  ".join(f"{r:>7}" for r in REL))
                for rr in REL:
                    row = next((x for x in sub if x["relation"] == rr), None)
                    if row is None:
                        continue
                    vals = "  ".join(f"{row[f'cos_{c}']:+.4f}" for c in REL)
                    print(f"{rr:>7}  {vals}")
                # summary contrast: within-relation cross-letter vs between-relation centroids
                same = [x["mean_cross_letter_cos"] for x in inv
                        if x["source_layer"] == L and x["target_condition"] == cond]
                cent = {x["relation"]: x for x in sub}
                between=[]
                for i,r1 in enumerate(REL):
                    for r2 in REL[i+1:]:
                        if r1 in cent and r2 in cent:
                            between.append(cent[r1][f"cos_{r2}"])
                if same and between:
                    print(f"  mean same-relation cross-letter={np.mean(same):+.4f} | "
                          f"mean between-relation={np.mean(between):+.4f} | "
                          f"gap={np.mean(same)-np.mean(between):+.4f}")

        print("\nMAPPING-INVARIANCE: same relation, different final letters")
        for r in inv:
            if r["target_condition"] == "opposite":
                print(f"L{r['source_layer']:02d} {r['target_relation']:5s} "
                          f"letters={r['letters_present']} cross-letter-cos={r['mean_cross_letter_cos']:+.4f}")

        meta_out = dict(
            model=spec.repo_id,
            writer_layer=writer_layer,
            writer_candidates=writer_candidates,
            source_layers=valid_sources,
            writer_definition="Real-Gray prefix-last answer-letter centered mean under randomized mappings",
            pullback_definition="g_pair = .5*(grad_sub-grad_ref), grad = d <h_writer,last, unit(s_letter)> / d h_source",
            interpretation=(
                "If pullbacks for different output letters that encode the same relation converge upstream "
                "and align with mapping-invariant relation directions, this links the late answer writer "
                "to an upstream spatial variable rather than merely observing two unrelated probes."
            ),
            validate_pullback=a.validate_pullback,
            validate_scale=a.validate_scale,
        )
        (outdir / "metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
        print("\nSaved:", outdir)
        print("Main files: onset_summary.csv, pullback_summary.csv, mapping_invariance.csv")

    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
