#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick diagnostic: is a strong late-last spatial writer an abstract relation
writer, or mostly an answer-token/answer-symbol bias?

Model/data
----------
- Qwen2.5-VL-3B (repo alias: qwen-3b)
- COCO two-object spatial task
- train/test split is relation-stratified

Core idea
---------
Each example gets its OWN random permutation between semantic relations and
answer symbols A/B/C/D, e.g.

    A = right
    B = below
    C = left
    D = above

For another sample, LEFT may map to A/B/D instead. Therefore the semantic
relation LEFT is deliberately decorrelated from any fixed output symbol.

On TRAIN samples, at selected late decoder layers, collect the final prompt
(prefix-last) hidden state on REAL and GRAY images:

    q_{i,L} = h_real[L,last] - h_gray[L,last]

Then fit two kinds of centered writers from THE SAME q states:

1) semantic relation writer
       s_rel[r,L] = E[q | GT relation=r] - common relation mean
   Because relation->letter is randomized, a fixed A/B/C/D output component
   should average out inside s_rel.

2) answer-symbol writer (positive control)
       s_letter[a,L] = E[q | correct answer symbol=a] - common letter mean
   This intentionally captures answer-symbol-aligned information.

On TEST samples (with independently assigned random mappings), run full greedy
model.generate() and patch ONE late layer at the prefix-last token:

    h[L,last] += alpha * s

Conditions:
- semantic_correct: use s_rel[GT]. If this helps despite LEFT mapping to a
  different A/B/C/D on every sample, the writer cannot be only a fixed answer
  symbol bias.
- semantic_wrong (optional, default ON): use s_rel[opposite(GT)] and ask whether
  generation follows the *current sample's* letter for that opposite relation.
  This is a strong targeted semantic-steering test.
- letter_correct: use s_letter[current correct symbol]. Positive control for
  direct answer-side/symbol steering.

Interpretation
--------------
A) semantic_correct / semantic_wrong remain strong under randomized mappings:
   evidence that late writer contains relation-level content, not only a fixed
   answer token direction.
B) letter_correct is strong but semantic_* collapses:
   late effect is much more consistent with answer-symbol / lexical steering.
C) both are strong:
   late state contains both an abstract relation component and an answer-stage
   component.

This script uses the llava16 repo helper modules already used by the user's
other COCO scripts:
    analyze_coco_centroid_generation_step1_v4.py
    eval_coco_multilayer_relation_trajectory_repair_v1.py

Example quick run
-----------------
CUDA_VISIBLE_DEVICES=0 python -u qwen3b_coco_random_answer_mapping_late_writer_quick.py \
  --layers 32,34,35 --alphas 1 \
  --max-samples 160 --eval-max-samples 80 \
  --output-dir output/qwen3b_coco_randmap_late_quick --overwrite

For a fairer relation-vs-letter magnitude comparison, rerun with:
    --norm-mode equalize
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
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
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--layers", default="32,34,35",
                   help="Late decoder block indices to test.")
    p.add_argument("--alphas", default="1",
                   help="Comma-separated steering strengths.")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=160,
                   help="0 = use all available samples. Quick default keeps runtime modest.")
    p.add_argument("--eval-max-samples", type=int, default=80,
                   help="0 = evaluate all held-out samples.")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--norm-mode", default="native", choices=["native", "equalize"],
                   help=("native: use fitted writer norms. equalize: at each layer, rescale every "
                         "relation/letter writer to the same mean relation-writer norm."))
    p.add_argument("--no-semantic-wrong", action="store_true",
                   help="Skip opposite-relation semantic steering to save generation time.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted({int(x.strip().upper().replace("L", "")) for x in s.split(",") if x.strip()})


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def make_real_image(record):
    im = base.record_image(record)
    if hasattr(im, "convert"):
        im = im.convert("RGB")
    return im


def make_gray_image(real_image, value: int):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Dict[str, str]) -> str:
    """Fresh prompt with no fixed relation->answer-symbol association."""
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
    """Assign all 24 permutations cyclically; each full cycle is perfectly balanced."""
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


def make_batch(processor, device, image, question_text):
    return base.make_question_batch(
        processor=processor,
        image=image,
        question_text=question_text,
        device=device,
    )


class LastStateCapture:
    def __init__(self, decoder_layers, layers_req):
        self.states: Dict[int, np.ndarray] = {}
        self.handles = []
        for L in layers_req:
            self.handles.append(decoder_layers[L].register_forward_hook(self._make_hook(L)))

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
def capture_prefix_last(model, decoder_layers, batch, layers_req) -> Dict[int, np.ndarray]:
    cap = LastStateCapture(decoder_layers, layers_req)
    try:
        kwargs = dict(batch)
        kwargs["use_cache"] = False
        _ = model(**kwargs)
        missing = [L for L in layers_req if L not in cap.states]
        if missing:
            raise RuntimeError(f"Did not capture layers: {missing}")
        return cap.states
    finally:
        cap.close()


class LastTokenPatch:
    """Patch only the prefill prefix-last state; cached decoding seq_len=1 is left untouched."""
    def __init__(self, layer, direction: np.ndarray, alpha: float):
        self.direction = np.asarray(direction, np.float32)
        self.alpha = float(alpha)
        self.applied = 0
        self.delta_norm = float("nan")
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        # During Qwen generation, prefill has seq_len > 1; cached decode is usually 1.
        if h.ndim != 3 or int(h.shape[1]) <= 1:
            return out
        y = h.float().clone()
        delta = self.alpha * torch.as_tensor(self.direction, device=y.device, dtype=torch.float32)
        y[:, -1, :] += delta
        self.applied += 1
        self.delta_norm = float(delta.norm().item())
        return traj.replace_first_tensor(out, y.to(h.dtype))

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


def parse_answer_letter(text: str, rel_to_letter: Dict[str, str]) -> Tuple[str | None, str]:
    """Prefer an explicit A/B/C/D. Relation-word fallback is logged separately."""
    t = str(text).strip()
    up = t.upper()

    # Explicit standalone answer symbol. Prefer early answer-like occurrences.
    pats = [
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([ABCD])\b",
        r"^\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    for pat in pats:
        m = re.search(pat, up)
        if m:
            return m.group(1), "letter"

    low = t.lower()
    hits = []
    for r in REL:
        if re.search(rf"\b{re.escape(r)}\b", low):
            hits.append(r)
    if len(hits) == 1:
        return rel_to_letter[hits[0]], "relation_fallback"
    return None, "unparsed"


@torch.inference_mode()
def generate_one(model, processor, decoder_layers, batch, mapping,
                 L=None, direction=None, alpha=0.0, max_new_tokens=4):
    patch = None
    try:
        if direction is not None:
            patch = LastTokenPatch(decoder_layers[L], direction, alpha)
        text = base.generate_text(model, processor, batch, max_new_tokens=max_new_tokens)
        if patch is not None and patch.applied < 1:
            raise RuntimeError(f"L{L} prefix-last patch did not fire")
        pred, parse_mode = parse_answer_letter(text, mapping)
        return pred, text, parse_mode, (patch.delta_norm if patch is not None else 0.0)
    finally:
        if patch is not None:
            patch.close()


def fit_centered_writers(train_meta, q_by_sid, mappings, layers_req):
    rel_writers = {L: {} for L in layers_req}
    letter_writers = {L: {} for L in layers_req}
    diagnostics = []

    for L in layers_req:
        rel_mu = {}
        for r in REL:
            xs = [q_by_sid[int(m["sid"])][L] for m in train_meta if m["gt"] == r]
            if not xs:
                raise RuntimeError(f"No train samples for relation {r} at L{L}")
            rel_mu[r] = np.mean(np.stack(xs), axis=0).astype(np.float32)
        rel_common = np.mean(np.stack([rel_mu[r] for r in REL]), axis=0)
        for r in REL:
            rel_writers[L][r] = (rel_mu[r] - rel_common).astype(np.float32)

        letter_mu = {}
        for a in LETTERS:
            xs = []
            for m in train_meta:
                sid = int(m["sid"])
                if mappings[sid][m["gt"]] == a:
                    xs.append(q_by_sid[sid][L])
            if not xs:
                raise RuntimeError(f"No train samples for answer letter {a} at L{L}")
            letter_mu[a] = np.mean(np.stack(xs), axis=0).astype(np.float32)
        letter_common = np.mean(np.stack([letter_mu[a] for a in LETTERS]), axis=0)
        for a in LETTERS:
            letter_writers[L][a] = (letter_mu[a] - letter_common).astype(np.float32)

        rel_norms = [float(np.linalg.norm(rel_writers[L][r])) for r in REL]
        letter_norms = [float(np.linalg.norm(letter_writers[L][a])) for a in LETTERS]
        for r, n in zip(REL, rel_norms):
            diagnostics.append(dict(layer=L, kind="semantic_relation", key=r, raw_norm=n))
        for a, n in zip(LETTERS, letter_norms):
            diagnostics.append(dict(layer=L, kind="answer_letter", key=a, raw_norm=n))

    return rel_writers, letter_writers, diagnostics


def equalize_writer_norms(rel_writers, letter_writers, layers_req):
    """Rescale every writer at a layer to the mean semantic relation-writer norm."""
    for L in layers_req:
        target = float(np.mean([np.linalg.norm(rel_writers[L][r]) for r in REL]))
        target = max(target, 1e-12)
        for r in REL:
            v = rel_writers[L][r]
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                rel_writers[L][r] = (v * (target / n)).astype(np.float32)
        for a in LETTERS:
            v = letter_writers[L][a]
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                letter_writers[L][a] = (v * (target / n)).astype(np.float32)


def summarize(rows, baseline_by_sid):
    if not rows:
        return {}
    N = len(rows)
    correct = sum(int(r["correct"]) for r in rows)
    W2C = C2W = 0
    base_wrong = base_correct = 0
    target_follow = 0
    parsed = 0
    for r in rows:
        b = baseline_by_sid[int(r["sid"])]
        bc = bool(b["baseline_correct"])
        if bc:
            base_correct += 1
            if not r["correct"]:
                C2W += 1
        else:
            base_wrong += 1
            if r["correct"]:
                W2C += 1
        if r["pred_letter"] is not None:
            parsed += 1
        if r["pred_letter"] == r["steer_target_letter"]:
            target_follow += 1

    return dict(
        N=N,
        acc=correct / max(N, 1),
        W2C=W2C,
        W2C_rate=W2C / max(base_wrong, 1),
        C2W=C2W,
        C2W_rate=C2W / max(base_correct, 1),
        preserve=1.0 - C2W / max(base_correct, 1),
        target_follow=target_follow / max(N, 1),
        parsed_rate=parsed / max(N, 1),
    )


def main():
    a = parse_args()
    layers_req = parse_ints(a.layers)
    alphas = parse_floats(a.alphas)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

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
            sid=sid,
            gt=gt,
            subject=str(p["subject"]),
            reference=str(p["reference"]),
        ))

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, test = traj.stratified_split(meta, a.train_ratio, a.seed)
    test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)

    # Independent mapping assignments for train/test. Every 24-sample cycle is balanced.
    train_map = assign_balanced_random_mappings(train, a.seed + 1001)
    test_map = assign_balanced_random_mappings(test, a.seed + 2003)

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
        for L in layers_req:
            if not (0 <= L < len(decoder_layers)):
                raise ValueError(f"L{L} invalid for {len(decoder_layers)} decoder layers")

        print("=" * 110)
        print("QWEN3B COCO RANDOMIZED ANSWER MAPPING — LATE PREFIX-LAST WRITER DIAGNOSTIC")
        print("=" * 110)
        print(f"decoder={path} | layers={layers_req} | alphas={alphas}")
        print(f"train/test={len(train)}/{len(test)} | norm_mode={a.norm_mode}")
        print("train writer state: q = h_real(prefix-last) - h_gray(prefix-last)")
        print("semantic writer grouped by GT relation; letter writer grouped by current correct A/B/C/D")
        print()

        # ---------------------------
        # 1) TRAIN writer extraction
        # ---------------------------
        q_by_sid: Dict[int, Dict[int, np.ndarray]] = {}
        for m in tqdm(train, desc="TRAIN random-map Real-Gray prefix-last"):
            sid = int(m["sid"])
            real = gray = real_batch = gray_batch = None
            try:
                mapping = train_map[sid]
                prompt = build_randmap_prompt(m["subject"], m["reference"], mapping)
                real = make_real_image(rec_by_sid[sid])
                gray = make_gray_image(real, a.gray_value)
                real_batch = make_batch(processor, device, real, prompt)
                gray_batch = make_batch(processor, device, gray, prompt)
                hr = capture_prefix_last(model, decoder_layers, real_batch, layers_req)
                hg = capture_prefix_last(model, decoder_layers, gray_batch, layers_req)
                q_by_sid[sid] = {L: (hr[L] - hg[L]).astype(np.float32) for L in layers_req}
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "train_capture", "sid": sid, "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                for im in (real, gray):
                    if im is not None:
                        with contextlib.suppress(Exception):
                            im.close()
                del real_batch, gray_batch
                gc.collect()

        rel_writers, letter_writers, geom = fit_centered_writers(
            train, q_by_sid, train_map, layers_req
        )
        traj.write_csv(out / "writer_raw_norms.csv", geom)

        np.savez_compressed(
            out / "semantic_relation_writers_raw.npz",
            **{f"L{L}_{r}": rel_writers[L][r] for L in layers_req for r in REL}
        )
        np.savez_compressed(
            out / "answer_letter_writers_raw.npz",
            **{f"L{L}_{a0}": letter_writers[L][a0] for L in layers_req for a0 in LETTERS}
        )

        if a.norm_mode == "equalize":
            equalize_writer_norms(rel_writers, letter_writers, layers_req)

        # ---------------------------
        # 2) TEST baseline generation
        # ---------------------------
        baseline = []
        prompt_cache = {}
        for m in tqdm(test, desc="TEST random-map baseline"):
            sid = int(m["sid"])
            real = batch = None
            try:
                mapping = test_map[sid]
                prompt = build_randmap_prompt(m["subject"], m["reference"], mapping)
                prompt_cache[sid] = prompt
                correct_letter = mapping[m["gt"]]
                real = make_real_image(rec_by_sid[sid])
                batch = make_batch(processor, device, real, prompt)
                pred, text, parse_mode, _ = generate_one(
                    model, processor, decoder_layers, batch, mapping,
                    max_new_tokens=a.max_new_tokens
                )
                baseline.append({
                    **m,
                    "mapping": "|".join(f"{r}:{mapping[r]}" for r in REL),
                    "correct_letter": correct_letter,
                    "baseline_pred_letter": pred,
                    "baseline_text": text,
                    "baseline_parse_mode": parse_mode,
                    "baseline_correct": pred == correct_letter,
                })
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "test_baseline", "sid": sid, "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                del batch
                gc.collect()

        traj.write_csv(out / "baseline.csv", baseline)
        baseline_by_sid = {int(x["sid"]): x for x in baseline}
        N = len(baseline)
        Nc = sum(int(x["baseline_correct"]) for x in baseline)
        Nw = N - Nc
        print(f"BASELINE random-map: N={N} acc={Nc/max(N,1):.4f} correct={Nc} wrong={Nw}")

        # ---------------------------
        # 3) Steering conditions
        # ---------------------------
        conditions = ["semantic_correct", "letter_correct"]
        if not a.no_semantic_wrong:
            conditions.insert(1, "semantic_wrong")

        all_rows = []
        summary_rows = []

        test_meta_by_sid = {int(m["sid"]): m for m in test}
        for L in layers_req:
            for alpha in alphas:
                for condition in conditions:
                    rows = []
                    for b in tqdm(baseline, desc=f"L{L} a{alpha:g} {condition}", leave=False):
                        sid = int(b["sid"])
                        m = test_meta_by_sid[sid]
                        mapping = test_map[sid]
                        gt = m["gt"]
                        correct_letter = mapping[gt]

                        if condition == "semantic_correct":
                            steer_relation = gt
                            steer_target_letter = mapping[steer_relation]
                            d = rel_writers[L][steer_relation]
                        elif condition == "semantic_wrong":
                            steer_relation = OPP[gt]
                            steer_target_letter = mapping[steer_relation]
                            d = rel_writers[L][steer_relation]
                        elif condition == "letter_correct":
                            steer_relation = ""
                            steer_target_letter = correct_letter
                            d = letter_writers[L][steer_target_letter]
                        else:
                            raise AssertionError(condition)

                        real = batch = None
                        try:
                            real = make_real_image(rec_by_sid[sid])
                            batch = make_batch(processor, device, real, prompt_cache[sid])
                            pred, text, parse_mode, delta_norm = generate_one(
                                model, processor, decoder_layers, batch, mapping,
                                L=L, direction=d, alpha=alpha,
                                max_new_tokens=a.max_new_tokens,
                            )
                            row = dict(
                                sid=sid,
                                gt=gt,
                                layer=L,
                                alpha=alpha,
                                condition=condition,
                                correct_letter=correct_letter,
                                steer_relation=steer_relation,
                                steer_target_letter=steer_target_letter,
                                baseline_pred_letter=b["baseline_pred_letter"],
                                baseline_correct=b["baseline_correct"],
                                pred_letter=pred,
                                text=text,
                                parse_mode=parse_mode,
                                correct=(pred == correct_letter),
                                target_follow=(pred == steer_target_letter),
                                delta_norm=delta_norm,
                                mapping=b["mapping"],
                            )
                            rows.append(row)
                            all_rows.append(row)
                        except Exception as e:
                            traj.append_jsonl(err_path, {
                                "phase": "steer", "sid": sid, "layer": L,
                                "alpha": alpha, "condition": condition,
                                "error": str(e), "traceback": traceback.format_exc(),
                            })
                            raise
                        finally:
                            if real is not None:
                                with contextlib.suppress(Exception):
                                    real.close()
                            del batch

                    s = summarize(rows, baseline_by_sid)
                    s.update(layer=L, alpha=alpha, condition=condition,
                             baseline_acc=Nc/max(N,1), norm_mode=a.norm_mode)
                    summary_rows.append(s)
                    print(
                        f"L{L:02d} a={alpha:g} {condition:16s} | "
                        f"acc={s['acc']:.4f} ({s['acc']-Nc/max(N,1):+.4f}) | "
                        f"W2C={s['W2C']}/{Nw}={s['W2C_rate']:.3f} | "
                        f"C2W={s['C2W']}/{Nc}={s['C2W_rate']:.3f} | "
                        f"targetFollow={s['target_follow']:.3f} | parsed={s['parsed_rate']:.3f}"
                    )
                    gc.collect()

        traj.write_csv(out / "per_sample.csv", all_rows)
        traj.write_csv(out / "summary.csv", summary_rows)

        # Small human-readable interpretation aid.
        with open(out / "README_RESULTS.txt", "w", encoding="utf-8") as f:
            f.write(
                "Interpretation guide\n"
                "====================\n"
                "semantic_correct: relation writer grouped by semantic GT under randomized mappings.\n"
                "  Strong W2C/accuracy gain means the late writer retains relation-level content even\n"
                "  though LEFT/RIGHT/ABOVE/BELOW never has a fixed A/B/C/D answer symbol.\n\n"
                "semantic_wrong: writer for OPP(GT).\n"
                "  targetFollow asks whether generation follows the CURRENT SAMPLE'S mapped letter\n"
                "  for that wrong semantic relation. Strong targetFollow is particularly diagnostic\n"
                "  of abstract relation steering.\n\n"
                "letter_correct: writer grouped by the current correct A/B/C/D symbol.\n"
                "  This is a positive control for answer-side/symbol steering.\n\n"
                "If letter_correct is strong but semantic_* collapses, the late effect is largely\n"
                "answer-symbol/lexical. If semantic_* remains strong, the late state contains a\n"
                "relation-level causal component beyond a fixed output symbol.\n"
            )

        print(f"\nSaved to: {out}")
        print("Key file: summary.csv")

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
