
# -*- coding: utf-8 -*-
"""Qwen2.5-VL-3B COCO: ALL-CORRECT vs ALL-WRONG middle spatial steering.

NO Gray images. Train directions come only from raw object-pair states:
    r_L = h_L(subject) - h_L(reference)

For every test sample with GT relation g and opposite relation o:
    d_correct = mu_g - mu_o
    d_wrong   = mu_o - mu_g = -d_correct

At one decoder layer, patch the subject/reference token hidden states symmetrically:
    h_sub += 0.5 * alpha * d
    h_ref -= 0.5 * alpha * d
so the pair state moves by alpha*d.

Both conditions run on ALL test samples, including baseline-correct and baseline-wrong.
Reports overall accuracy plus W2C/C2W/preserve/target-follow.
"""
from __future__ import annotations

import argparse, contextlib, gc, json, random, shutil, traceback
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj

REL = ("left", "right", "above", "below")
OPP = {"left":"right", "right":"left", "above":"below", "below":"above"}


def args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--layers", default="14,18,22,26,30,32")
    p.add_argument("--alphas", default="1,5")
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last","mean"])
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager","sdpa","flash_attention_2","none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s):
    return sorted({int(x.strip().upper().replace("L","")) for x in s.split(",") if x.strip()})

def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


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
        # only prefill; cached decoding usually has seq len 1
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
def patched_generate(model, processor, decoder_layers, batch, L, sub_pos, ref_pos, d, alpha, max_new):
    patch = PairDeltaPatch(decoder_layers[L], sub_pos, ref_pos, d, alpha)
    try:
        text = base.generate_text(model, processor, batch, max_new_tokens=max_new)
        if patch.applied < 1:
            raise RuntimeError(f"L{L} hook did not fire")
        pred = traj.normalize_relation(base, text)
        return pred, text, patch.delta_norm
    finally:
        patch.close()


def make_batch(processor, device, record, meta):
    image = base.record_image(record)
    batch = base.make_question_batch(
        processor=processor,
        image=image,
        question_text=meta["question_text"],
        device=device,
    )
    return image, batch


def main():
    a = args_parser()
    layers_req = parse_ints(a.layers)
    alphas = parse_floats(a.alphas)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
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
    records, audit = two.load_records("coco_two", Path(a.data_root), None)
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
            if not 0 <= L < len(decoder_layers):
                raise ValueError(f"L{L} invalid for {len(decoder_layers)} layers")

        test_ids = {int(x["sid"]) for x in test}
        print(f"decoder={path} | layers={layers_req} | train/test={len(train)}/{len(test)} | alphas={alphas}")
        print("direction: raw pair state h_sub-h_ref; NO Gray")
        print("ALL-CORRECT d = mu_GT - mu_OPP(GT); ALL-WRONG = -d")

        # 1) clean pair states + clean generation
        for m in tqdm(train + test, desc="clean"):
            sid = int(m["sid"])
            image = batch = None
            try:
                image, batch = make_batch(processor, device, rec_by_sid[sid], m)
                clean = traj.clean_forward(
                    base, model, processor, decoder_layers, batch,
                    m["subject"], m["reference"], layers_req,
                    a.object_state, token_map,
                )
                state_by_sid[sid] = clean["states"]
                pos_by_sid[sid] = (clean["subject_positions"], clean["reference_positions"])
                if sid in test_ids:
                    text = base.generate_text(model, processor, batch, max_new_tokens=a.max_new_tokens)
                    pred = traj.normalize_relation(base, text)
                    baseline.append({
                        **m,
                        "baseline_pred": pred,
                        "baseline_text": text,
                        "baseline_correct": pred == m["gt"],
                    })
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase":"clean", "sid":sid, "error":str(e),
                    "traceback":traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception): image.close()
                del batch
                gc.collect()

        valid = set(state_by_sid)
        train = [x for x in train if int(x["sid"]) in valid]
        baseline = [x for x in baseline if int(x["sid"]) in valid]
        cent = traj.fit_centroids(train, state_by_sid, layers_req)
        np.savez_compressed(
            out / "train_centroids_raw_subref.npz",
            **{f"L{L}_{r}": cent[L][r] for L in layers_req for r in REL},
        )
        traj.write_csv(out / "baseline.csv", baseline)

        N = len(baseline)
        Nc = sum(bool(x["baseline_correct"]) for x in baseline)
        Nw = N - Nc
        base_acc = Nc / max(N, 1)
        print(f"BASELINE N={N} acc={base_acc:.4f} correct={Nc} wrong={Nw}")
        m_by_sid = {int(x["sid"]): x for x in test}

        per = []
        summary = []
        for L in layers_req:
            for alpha in alphas:
                for condition in ("all_correct", "all_wrong"):
                    for b in tqdm(baseline, desc=f"L{L} a{alpha:g} {condition}", leave=False):
                        sid = int(b["sid"])
                        m = m_by_sid[sid]
                        gt = m["gt"]
                        opp = OPP[gt]
                        # exact same axis and norm, opposite sign
                        d_correct = cent[L][gt] - cent[L][opp]
                        d = d_correct if condition == "all_correct" else -d_correct
                        target = gt if condition == "all_correct" else opp
                        image = batch = None
                        try:
                            image, batch = make_batch(processor, device, rec_by_sid[sid], m)
                            sp, rp = pos_by_sid[sid]
                            pred, text, dn = patched_generate(
                                model, processor, decoder_layers, batch,
                                L, sp, rp, d, alpha, a.max_new_tokens,
                            )
                            per.append(dict(
                                condition=condition,
                                sid=sid,
                                layer=L,
                                alpha=alpha,
                                gt=gt,
                                opposite=opp,
                                target=target,
                                baseline_pred=b["baseline_pred"],
                                baseline_correct=bool(b["baseline_correct"]),
                                patched_pred=pred,
                                patched_correct=(pred == gt),
                                target_follow=(pred == target),
                                changed=(pred != b["baseline_pred"]),
                                W2C=(not b["baseline_correct"] and pred == gt),
                                C2W=(b["baseline_correct"] and pred != gt),
                                C2C=(b["baseline_correct"] and pred == gt),
                                W2W=(not b["baseline_correct"] and pred != gt),
                                delta_norm=dn,
                                text=text,
                            ))
                        finally:
                            if image is not None:
                                with contextlib.suppress(Exception): image.close()
                            del batch
                            gc.collect()

                # summarize both full-dataset conditions
                for condition in ("all_correct", "all_wrong"):
                    rows = [x for x in per if x["condition"] == condition and x["layer"] == L and x["alpha"] == alpha]
                    n = len(rows)
                    nacc = sum(bool(x["patched_correct"]) for x in rows)
                    w2c = sum(bool(x["W2C"]) for x in rows)
                    c2w = sum(bool(x["C2W"]) for x in rows)
                    c2c = sum(bool(x["C2C"]) for x in rows)
                    changed = sum(bool(x["changed"]) for x in rows)
                    follow = sum(bool(x["target_follow"]) for x in rows)
                    row = dict(
                        layer=L,
                        alpha=alpha,
                        condition=condition,
                        N=n,
                        baseline_acc=base_acc,
                        patched_acc=nacc/max(n,1),
                        delta_acc=nacc/max(n,1)-base_acc,
                        N_baseline_correct=Nc,
                        N_baseline_wrong=Nw,
                        W2C=w2c,
                        W2C_rate=w2c/max(Nw,1),
                        C2W=c2w,
                        C2W_rate=c2w/max(Nc,1),
                        C2C=c2c,
                        preserve_rate=c2c/max(Nc,1),
                        changed=changed,
                        changed_rate=changed/max(n,1),
                        target_follow=follow,
                        target_follow_rate=follow/max(n,1),
                    )
                    summary.append(row)
                    print(
                        f"L{L:02d} a={alpha:g} {condition:11s} | "
                        f"acc={row['patched_acc']:.4f} ({row['delta_acc']:+.4f}) | "
                        f"W2C={w2c}/{Nw}={row['W2C_rate']:.3f} | "
                        f"C2W={c2w}/{Nc}={row['C2W_rate']:.3f} | "
                        f"preserve={row['preserve_rate']:.3f} | "
                        f"follow={row['target_follow_rate']:.3f} | changed={row['changed_rate']:.3f}"
                    )

                traj.write_csv(out / "per_sample.csv", per)
                traj.write_csv(out / "summary.csv", summary)

        (out / "metadata.json").write_text(json.dumps(dict(
            model=spec.repo_id,
            layers=layers_req,
            alphas=alphas,
            train_ratio=a.train_ratio,
            seed=a.seed,
            object_state=a.object_state,
            pair_state="raw h_subject-h_reference (NO Gray)",
            correct_direction="mu_GT(train)-mu_OPPOSITE_GT(train)",
            wrong_direction="mu_OPPOSITE_GT(train)-mu_GT(train) = -correct_direction",
            patch="subject += .5*alpha*d; reference -= .5*alpha*d; one layer only",
            evaluation="both ALL-CORRECT and ALL-WRONG run on every test sample",
        ), indent=2), encoding="utf-8")
        print("Saved:", out)
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
