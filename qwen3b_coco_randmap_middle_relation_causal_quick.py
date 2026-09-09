
# -*- coding: utf-8 -*-
"""Quick causal test: does the MID-layer object relation code remain semantic
under randomized relation->answer mappings?

Model/data
----------
- Qwen2.5-VL-3B (repo alias qwen-3b)
- COCO two-object spatial task
- Per-example randomized A/B/C/D mapping, balanced so each GT relation maps
  roughly equally to A/B/C/D.

Core test
---------
At each requested middle layer L, extract the raw object-pair state

    r_{i,L} = h_L(subject) - h_L(reference)

from TRAIN examples presented with randomized answer mappings. Fit relation
centroids mu_left/right/above/below. Because the answer letter is randomized,
these centroids cannot rely on a fixed LEFT->A/B/C/D association.

For every TEST example with GT relation g and opposite relation o:

    d_correct = mu_g - mu_o
    d_wrong   = mu_o - mu_g = -d_correct

Patch the subject/reference token states symmetrically:

    h_sub += 0.5 * alpha * d
    h_ref -= 0.5 * alpha * d

Then run full greedy generation. Crucially, for semantic_wrong, the target is
NOT a fixed letter: it is the CURRENT SAMPLE'S letter assigned to relation o.

If the same d_wrong makes the model follow mapping[o] across different random
mappings, that is strong evidence that the middle direction controls relation
semantics rather than a fixed answer token.

Key metric
----------
semantic_wrong targetFollow vs baselineOppFollow.
Chance is ~0.25 in a balanced four-way task, but baselineOppFollow is the more
appropriate empirical reference.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u qwen3b_coco_randmap_middle_relation_causal_quick.py \
  --layers 18,22 --alphas 1,5 \
  --max-samples 160 --eval-max-samples 80 \
  --output-dir output/qwen3b_coco_randmap_middle_quick --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import gc
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
    p.add_argument("--layers", default="18,22", help="Middle layers, evaluated separately.")
    p.add_argument("--alphas", default="1,5")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--max-samples", type=int, default=160, help="0 = all")
    p.add_argument("--eval-max-samples", type=int, default=80, help="0 = all held-out")
    p.add_argument("--max-new-tokens", type=int, default=4)
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
    """Random permutation per sample, while balancing GT relation -> correct letter.

    For each GT-relation subgroup, correct letters cycle A/B/C/D after shuffling.
    The other three relations receive a random permutation of the remaining letters.
    This directly decorrelates semantic GT relation from answer-symbol identity.
    """
    rng = random.Random(seed)
    out: Dict[int, Dict[str, str]] = {}
    by_rel = {r: [] for r in REL}
    for m in items:
        by_rel[m["gt"]].append(m)

    for gt in REL:
        group = list(by_rel[gt])
        rng.shuffle(group)
        letter_cycle = list(LETTERS)
        rng.shuffle(letter_cycle)
        for i, m in enumerate(group):
            correct_letter = letter_cycle[i % len(LETTERS)]
            other_rel = [r for r in REL if r != gt]
            other_letters = [a for a in LETTERS if a != correct_letter]
            rng.shuffle(other_rel)
            rng.shuffle(other_letters)
            mapping = {gt: correct_letter}
            mapping.update({r: a for r, a in zip(other_rel, other_letters)})
            out[int(m["sid"])] = mapping
    return out


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

    # fallback if model ignores instruction and writes a relation word
    low = t.lower()
    hits = [r for r in REL if re.search(rf"\b{re.escape(r)}\b", low)]
    if len(hits) == 1:
        return rel_to_letter[hits[0]], "relation_fallback"
    return None, "unparsed"


class PairDeltaPatch:
    def __init__(self, layer, sub_pos, ref_pos, d, alpha):
        self.sub_pos = tuple(map(int, sub_pos))
        self.ref_pos = tuple(map(int, ref_pos))
        self.d = np.asarray(d, np.float32)
        self.alpha = float(alpha)
        self.applied = 0
        self.delta_norm = float("nan")
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        if h.ndim != 3 or int(h.shape[1]) <= max(self.sub_pos + self.ref_pos):
            return out
        y = h.float().clone()
        delta = self.alpha * torch.as_tensor(self.d, device=y.device, dtype=torch.float32)
        for pos in self.sub_pos:
            y[:, pos, :] += 0.5 * delta
        for pos in self.ref_pos:
            y[:, pos, :] -= 0.5 * delta
        self.applied += 1
        self.delta_norm = float(delta.norm().item())
        return traj.replace_first_tensor(out, y.to(h.dtype))

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


@torch.inference_mode()
def patched_generate(model, processor, decoder_layers, batch, mapping,
                     L, sub_pos, ref_pos, d, alpha, max_new_tokens):
    patch = PairDeltaPatch(decoder_layers[L], sub_pos, ref_pos, d, alpha)
    try:
        text = base.generate_text(model, processor, batch, max_new_tokens=max_new_tokens)
        if patch.applied < 1:
            raise RuntimeError(f"L{L} middle pair patch did not fire")
        pred, parse_mode = parse_answer_letter(text, mapping)
        return pred, text, parse_mode, patch.delta_norm
    finally:
        patch.close()


def make_batch(processor, device, record, prompt):
    image = base.record_image(record)
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    batch = base.make_question_batch(
        processor=processor,
        image=image,
        question_text=prompt,
        device=device,
    )
    return image, batch


def mapping_string(mp):
    return "|".join(f"{r}:{mp[r]}" for r in REL)


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

    train_map = assign_relation_balanced_mappings(train, a.seed + 1001)
    test_map = assign_relation_balanced_mappings(test, a.seed + 2003)

    # mapping audit: relation x correct letter
    audit_rows = []
    for split_name, items, maps in (("train", train, train_map), ("test", test, test_map)):
        for gt in REL:
            for letter in LETTERS:
                n = sum(1 for m in items if m["gt"] == gt and maps[int(m["sid"])][gt] == letter)
                audit_rows.append(dict(split=split_name, gt=gt, correct_letter=letter, n=n))
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
    state_by_sid: Dict[int, Dict[int, np.ndarray]] = {}
    pos_by_sid = {}
    baseline = []

    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor)
        device = torch.device(a.device)
        decoder_layers, path = base.resolve_decoder_layers(model)
        token_map = base.relation_token_variants(processor.tokenizer)

        for L in layers_req:
            if not (0 <= L < len(decoder_layers)):
                raise ValueError(f"L{L} invalid for {len(decoder_layers)} decoder layers")

        print("=" * 112)
        print("QWEN3B COCO RANDOM-MAPPING — MIDDLE RELATION CAUSAL TEST")
        print("=" * 112)
        print(f"decoder={path} | layers={layers_req} (separate) | alphas={alphas}")
        print(f"train/test={len(train)}/{len(test)} | object_state={a.object_state}")
        print("train direction = raw h_subject-h_reference under randomized answer mappings")
        print("semantic_wrong target = CURRENT sample's letter for opposite relation")
        print()

        test_ids = {int(m["sid"]) for m in test}

        # ------------------------------------------------------------
        # 1) Capture randomized-prompt middle pair states + test baseline
        # ------------------------------------------------------------
        for m in tqdm(train + test, desc="rand-map clean middle states"):
            sid = int(m["sid"])
            mp = train_map[sid] if sid in train_map else test_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
            image = batch = None
            try:
                image, batch = make_batch(processor, device, rec_by_sid[sid], prompt)
                clean = traj.clean_forward(
                    base, model, processor, decoder_layers, batch,
                    m["subject"], m["reference"], layers_req,
                    a.object_state, token_map,
                )
                state_by_sid[sid] = clean["states"]
                pos_by_sid[sid] = (clean["subject_positions"], clean["reference_positions"])

                if sid in test_ids:
                    text = base.generate_text(model, processor, batch, max_new_tokens=a.max_new_tokens)
                    pred, parse_mode = parse_answer_letter(text, mp)
                    correct_letter = mp[m["gt"]]
                    opp = OPP[m["gt"]]
                    opp_letter = mp[opp]
                    baseline.append(dict(
                        **m,
                        mapping=mapping_string(mp),
                        correct_letter=correct_letter,
                        opposite=opp,
                        opposite_letter=opp_letter,
                        baseline_pred_letter=pred,
                        baseline_text=text,
                        baseline_parse_mode=parse_mode,
                        baseline_correct=(pred == correct_letter),
                        baseline_opp_follow=(pred == opp_letter),
                    ))
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "clean", "sid": sid, "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

        valid = set(state_by_sid)
        train = [m for m in train if int(m["sid"]) in valid]
        baseline = [b for b in baseline if int(b["sid"]) in valid]

        # relation centroids only: answer letter was randomized/balanced
        cent = traj.fit_centroids(train, state_by_sid, layers_req)
        np.savez_compressed(
            out / "train_relation_centroids_randmap_raw_subref.npz",
            **{f"L{L}_{r}": cent[L][r] for L in layers_req for r in REL},
        )
        traj.write_csv(out / "baseline.csv", baseline)

        N = len(baseline)
        Nc = sum(int(b["baseline_correct"]) for b in baseline)
        Nw = N - Nc
        base_acc = Nc / max(N, 1)
        base_opp_follow = sum(int(b["baseline_opp_follow"]) for b in baseline) / max(N, 1)
        parsed = sum(int(b["baseline_pred_letter"] is not None) for b in baseline) / max(N, 1)
        print(
            f"BASELINE random-map: N={N} acc={base_acc:.4f} correct={Nc} wrong={Nw} | "
            f"oppFollow={base_opp_follow:.3f} | parsed={parsed:.3f}"
        )

        b_by_sid = {int(b["sid"]): b for b in baseline}
        m_by_sid = {int(m["sid"]): m for m in test}
        per_rows = []
        summary_rows = []

        # ------------------------------------------------------------
        # 2) Causal semantic relation steering at middle object states
        # ------------------------------------------------------------
        for L in layers_req:
            for alpha in alphas:
                for condition in ("semantic_correct", "semantic_wrong"):
                    rows_this = []
                    for b in tqdm(baseline, desc=f"L{L} a{alpha:g} {condition}", leave=False):
                        sid = int(b["sid"])
                        m = m_by_sid[sid]
                        mp = test_map[sid]
                        gt = m["gt"]
                        opp = OPP[gt]
                        d_correct = cent[L][gt] - cent[L][opp]
                        if condition == "semantic_correct":
                            d = d_correct
                            target_rel = gt
                        else:
                            d = -d_correct
                            target_rel = opp
                        target_letter = mp[target_rel]

                        image = batch = None
                        try:
                            prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
                            image, batch = make_batch(processor, device, rec_by_sid[sid], prompt)
                            sp, rp = pos_by_sid[sid]
                            pred, text, parse_mode, delta_norm = patched_generate(
                                model, processor, decoder_layers, batch, mp,
                                L, sp, rp, d, alpha, a.max_new_tokens,
                            )
                            correct_letter = mp[gt]
                            row = dict(
                                sid=sid,
                                layer=L,
                                alpha=alpha,
                                condition=condition,
                                gt=gt,
                                opposite=opp,
                                mapping=mapping_string(mp),
                                correct_letter=correct_letter,
                                steer_target_relation=target_rel,
                                steer_target_letter=target_letter,
                                baseline_pred_letter=b["baseline_pred_letter"],
                                baseline_correct=bool(b["baseline_correct"]),
                                baseline_target_follow=(b["baseline_pred_letter"] == target_letter),
                                patched_pred_letter=pred,
                                patched_text=text,
                                parse_mode=parse_mode,
                                patched_correct=(pred == correct_letter),
                                target_follow=(pred == target_letter),
                                changed=(pred != b["baseline_pred_letter"]),
                                W2C=(not b["baseline_correct"] and pred == correct_letter),
                                C2W=(b["baseline_correct"] and pred != correct_letter),
                                delta_norm=delta_norm,
                            )
                            rows_this.append(row)
                            per_rows.append(row)
                        except Exception as e:
                            traj.append_jsonl(err_path, {
                                "phase": condition, "sid": sid, "layer": L, "alpha": alpha,
                                "error": str(e), "traceback": traceback.format_exc(),
                            })
                            raise
                        finally:
                            if image is not None:
                                with contextlib.suppress(Exception):
                                    image.close()
                            del batch
                            gc.collect()

                    n = len(rows_this)
                    acc = sum(int(r["patched_correct"]) for r in rows_this) / max(n, 1)
                    w2c = sum(int(r["W2C"]) for r in rows_this)
                    c2w = sum(int(r["C2W"]) for r in rows_this)
                    target_follow = sum(int(r["target_follow"]) for r in rows_this) / max(n, 1)
                    baseline_target_follow = sum(int(r["baseline_target_follow"]) for r in rows_this) / max(n, 1)
                    changed = sum(int(r["changed"]) for r in rows_this) / max(n, 1)
                    parsed_rate = sum(int(r["patched_pred_letter"] is not None) for r in rows_this) / max(n, 1)
                    mean_dn = float(np.mean([r["delta_norm"] for r in rows_this])) if rows_this else 0.0

                    s = dict(
                        layer=L,
                        alpha=alpha,
                        condition=condition,
                        N=n,
                        baseline_acc=base_acc,
                        patched_acc=acc,
                        delta_acc=acc - base_acc,
                        N_baseline_correct=Nc,
                        N_baseline_wrong=Nw,
                        W2C=w2c,
                        W2C_rate=w2c / max(Nw, 1),
                        C2W=c2w,
                        C2W_rate=c2w / max(Nc, 1),
                        preserve=1.0 - c2w / max(Nc, 1),
                        baseline_target_follow=baseline_target_follow,
                        target_follow=target_follow,
                        delta_target_follow=target_follow - baseline_target_follow,
                        changed_rate=changed,
                        parsed_rate=parsed_rate,
                        mean_delta_norm=mean_dn,
                    )
                    summary_rows.append(s)
                    print(
                        f"L{L:02d} a={alpha:g} {condition:16s} | "
                        f"acc={acc:.4f} ({acc-base_acc:+.4f}) | "
                        f"W2C={w2c}/{Nw}={s['W2C_rate']:.3f} | "
                        f"C2W={c2w}/{Nc}={s['C2W_rate']:.3f} | "
                        f"targetFollow={target_follow:.3f} "
                        f"(base={baseline_target_follow:.3f}, delta={s['delta_target_follow']:+.3f}) | "
                        f"changed={changed:.3f} | parsed={parsed_rate:.3f}"
                    )

                traj.write_csv(out / "per_sample.csv", per_rows)
                traj.write_csv(out / "summary.csv", summary_rows)

        (out / "metadata.json").write_text(json.dumps({
            "model": spec.repo_id,
            "layers": layers_req,
            "alphas": alphas,
            "train_ratio": a.train_ratio,
            "seed": a.seed,
            "object_state": a.object_state,
            "prompt": "per-example randomized relation->A/B/C/D mapping",
            "mapping_balance": "GT relation -> correct letter balanced within each relation subgroup",
            "pair_state": "raw h_subject-h_reference under randomized mapping prompt",
            "correct_direction": "mu_GT - mu_OPPOSITE",
            "wrong_direction": "mu_OPPOSITE - mu_GT",
            "patch": "subject += .5*alpha*d; reference -= .5*alpha*d; one middle layer at a time",
            "main_readout": "semantic_wrong targetFollow compared against baseline targetFollow for current sample mapping[opposite]",
        }, indent=2), encoding="utf-8")

        print("Saved:", out)
        print("Main thing to inspect: semantic_wrong targetFollow and delta_target_follow at L18/L22.")

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
