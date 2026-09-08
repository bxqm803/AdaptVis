#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick Qwen2.5-VL-3B COCO middle-state HARM vs REPAIR test using Real-Gray directions.

Uses the llava16 repo's existing helpers.

Train (default 30%): for each layer L and each sample, extract
    r_real,L = h_L(subject; real) - h_L(reference; real)
    r_gray,L = h_L(subject; gray) - h_L(reference; gray)
    q_L      = r_real,L - r_gray,L
then fit four relation centroids mu_r,L = E[q_L | relation=r].
The gray image is a constant RGB image with the SAME size as the real image.

Test (default 70%): actual model.generate() on REAL images only.
  REPAIR, baseline-wrong only:
    d = mu_GT - mu_baselinePred
  HARM, baseline-correct only:
    d = mu_opposite(GT) - mu_GT

At one decoder layer, patch the object-pair hidden state symmetrically:
    h_sub += 0.5 * alpha * d
    h_ref -= 0.5 * alpha * d
Thus the pair residual moves by alpha*d.

Example:
CUDA_VISIBLE_DEVICES=0 python -u eval_coco_qwen3b_middle_direction_harm_repair_quick.py \
  --layers 14,18,22,26,30,32 --alphas 1,5 --gray-value 128 \
  --output-dir output/qwen3b_mid_grayresid_harm_repair --overwrite
"""
from __future__ import annotations

import argparse, contextlib, gc, json, random, shutil, traceback
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from PIL import Image
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
    p.add_argument("--gray-value", type=int, default=128, help="Constant RGB value for Gray control image.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager","sdpa","flash_attention_2","none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s): return sorted({int(x.strip().upper().replace("L","")) for x in s.split(",") if x.strip()})
def parse_floats(s): return [float(x) for x in s.split(",") if x.strip()]


class PairDeltaPatch:
    def __init__(self, layer, sub_pos, ref_pos, d, alpha):
        self.sub_pos = tuple(map(int, sub_pos)); self.ref_pos = tuple(map(int, ref_pos))
        self.d = np.asarray(d, np.float32); self.alpha = float(alpha)
        self.applied = 0; self.delta_norm = float("nan")
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        h = traj.first_tensor(out)
        # prefill only; cached decoding has sequence length 1
        if h.ndim != 3 or int(h.shape[1]) <= max(self.sub_pos + self.ref_pos): return out
        y = h.float().clone()
        delta = self.alpha * torch.as_tensor(self.d, device=y.device, dtype=torch.float32)
        for p in self.sub_pos: y[:,p,:] += 0.5 * delta
        for p in self.ref_pos: y[:,p,:] -= 0.5 * delta
        self.applied += 1; self.delta_norm = float(delta.norm().item())
        return traj.replace_first_tensor(out, y.to(h.dtype))

    def close(self):
        with contextlib.suppress(Exception): self.handle.remove()


@torch.inference_mode()
def patched_generate(model, processor, layers, batch, L, sub_pos, ref_pos, d, alpha, max_new):
    patch = PairDeltaPatch(layers[L], sub_pos, ref_pos, d, alpha)
    try:
        text = base.generate_text(model, processor, batch, max_new_tokens=max_new)
        if patch.applied < 1: raise RuntimeError(f"L{L} hook did not fire")
        pred = traj.normalize_relation(base, text)
        return pred, text, patch.delta_norm
    finally:
        patch.close()


def make_batch_from_image(processor, device, image, meta):
    return base.make_question_batch(
        processor=processor, image=image,
        question_text=meta["question_text"], device=device
    )


def make_real_image(record):
    image = base.record_image(record)
    # Repo helpers normally return PIL.Image; convert defensively to RGB.
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return image


def make_gray_image(real_image, value):
    v = int(max(0, min(255, int(value))))
    return Image.new("RGB", real_image.size, (v, v, v))


def main():
    a = args_parser(); layers_req = parse_ints(a.layers); alphas = parse_floats(a.alphas)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")

    out = Path(a.output_dir)
    if a.overwrite and out.exists(): shutil.rmtree(out)
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"Non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True); err_path = out/"errors.jsonl"

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid):r for r in records}

    meta = []
    for r in records:
        sid = int(r.sid)
        if sid not in prompts: continue
        p = prompts[sid]; gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL: continue
        meta.append(dict(sid=sid, gt=gt, subject=str(p["subject"]),
                         reference=str(p["reference"]), question_text=str(p["question_text"])))
    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, test = traj.stratified_split(meta, a.train_ratio, a.seed)
    test = traj.stratified_cap(test, a.eval_max_samples, a.seed+1)

    specs = base.merged_model_specs(two); spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)
    kw = dict(dtype=base.resolve_dtype(spec.dtype_name), low_cpu_mem_usage=True,
              trust_remote_code=spec.trust_remote_code, device_map={"":a.device})
    if a.attn_impl != "none": kw["attn_implementation"] = a.attn_impl

    model = processor = None
    state_by_sid: Dict[int,Dict[int,np.ndarray]] = {}; pos_by_sid = {}; baseline = []
    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **kw); model.eval()
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor); device = torch.device(a.device)
        decoder_layers, path = base.resolve_decoder_layers(model)
        token_map = base.relation_token_variants(processor.tokenizer)
        for L in layers_req:
            if not 0 <= L < len(decoder_layers): raise ValueError(f"L{L} invalid for {len(decoder_layers)} layers")

        train_ids = {int(x["sid"]) for x in train}; test_ids = {int(x["sid"]) for x in test}
        print(f"decoder={path} | layers={layers_req} | train/test={len(train)}/{len(test)} | alphas={alphas}")

        # 1) TRAIN: extract Real-Gray object-pair residuals.
        #    q_L = [(h_sub-h_ref)_real] - [(h_sub-h_ref)_gray]
        # Only TRAIN uses Gray. TEST remains a normal real-image generation task.
        raw_real_train = {}
        raw_gray_train = {}
        for m in tqdm(train, desc="TRAIN Real-Gray states"):
            sid = int(m["sid"]); real=gray=real_batch=gray_batch=None
            try:
                real = make_real_image(rec_by_sid[sid])
                gray = make_gray_image(real, a.gray_value)
                real_batch = make_batch_from_image(processor, device, real, m)
                gray_batch = make_batch_from_image(processor, device, gray, m)

                real_clean = traj.clean_forward(
                    base, model, processor, decoder_layers, real_batch,
                    m["subject"], m["reference"], layers_req,
                    a.object_state, token_map
                )
                gray_clean = traj.clean_forward(
                    base, model, processor, decoder_layers, gray_batch,
                    m["subject"], m["reference"], layers_req,
                    a.object_state, token_map
                )

                raw_real_train[sid] = real_clean["states"]
                raw_gray_train[sid] = gray_clean["states"]
                state_by_sid[sid] = {
                    L: (real_clean["states"][L] - gray_clean["states"][L]).astype(np.float32)
                    for L in layers_req
                }
            except Exception as e:
                traj.append_jsonl(err_path,{"phase":"train_real_gray","sid":sid,"error":str(e),"traceback":traceback.format_exc()})
                raise
            finally:
                for im in (real, gray):
                    if im is not None:
                        with contextlib.suppress(Exception): im.close()
                del real_batch, gray_batch
                gc.collect()

        valid_train=set(state_by_sid)
        train=[x for x in train if int(x["sid"]) in valid_train]
        cent = traj.fit_centroids(train,state_by_sid,layers_req)

        # Save Real-Gray relation centroids. Global centering is unnecessary for
        # src->tgt differences because the common component cancels.
        np.savez_compressed(
            out/"train_grayresid_centroids.npz",
            **{f"L{L}_{r}":cent[L][r] for L in layers_req for r in REL}
        )

        # Useful geometry diagnostics: how much Gray subtraction changes the
        # relation centroids relative to raw Real pair states.
        raw_real_cent = traj.fit_centroids(train, raw_real_train, layers_req)
        geom=[]
        for L in layers_req:
            for r in REL:
                a1=np.asarray(raw_real_cent[L][r],np.float64)
                a2=np.asarray(cent[L][r],np.float64)
                cos=float(np.dot(a1,a2)/max(np.linalg.norm(a1)*np.linalg.norm(a2),1e-12))
                geom.append(dict(layer=L,relation=r,raw_real_norm=float(np.linalg.norm(a1)),
                                 real_minus_gray_norm=float(np.linalg.norm(a2)),
                                 raw_vs_grayresid_cosine=cos))
        traj.write_csv(out/"train_grayresid_geometry.csv",geom)

        # 2) TEST: normal REAL-image clean forward + greedy baseline generation.
        test_valid=set()
        for m in tqdm(test, desc="TEST real baseline"):
            sid=int(m["sid"]); real=batch=None
            try:
                real=make_real_image(rec_by_sid[sid])
                batch=make_batch_from_image(processor,device,real,m)
                clean=traj.clean_forward(
                    base,model,processor,decoder_layers,batch,
                    m["subject"],m["reference"],layers_req,
                    a.object_state,token_map
                )
                pos_by_sid[sid]=(clean["subject_positions"],clean["reference_positions"])
                text=base.generate_text(model,processor,batch,max_new_tokens=a.max_new_tokens)
                pred=traj.normalize_relation(base,text)
                baseline.append({**m,"baseline_pred":pred,"baseline_text":text,
                                 "baseline_correct":pred==m["gt"]})
                test_valid.add(sid)
            except Exception as e:
                traj.append_jsonl(err_path,{"phase":"test_baseline","sid":sid,"error":str(e),"traceback":traceback.format_exc()})
                raise
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                del batch; gc.collect()

        baseline=[x for x in baseline if int(x["sid"]) in test_valid]
        traj.write_csv(out/"baseline.csv",baseline)

        N=len(baseline); Nc=sum(bool(x["baseline_correct"]) for x in baseline); Nw=N-Nc; base_acc=Nc/max(N,1)
        print(f"BASELINE N={N} acc={base_acc:.4f} correct={Nc} wrong={Nw}")
        b_by_sid={int(x["sid"]):x for x in baseline}; m_by_sid={int(x["sid"]):x for x in test}

        per=[]; summary=[]
        for L in layers_req:
            for alpha in alphas:
                # REPAIR: baseline Pred -> GT, only natural errors
                repair_set=[x for x in baseline if not x["baseline_correct"] and x["baseline_pred"] in REL and x["baseline_pred"]!=x["gt"]]
                for b in tqdm(repair_set,desc=f"L{L} a{alpha:g} repair",leave=False):
                    sid=int(b["sid"]); m=m_by_sid[sid]; image=batch=None
                    src=b["baseline_pred"]; tgt=m["gt"]; d=cent[L][tgt]-cent[L][src]
                    try:
                        image=make_real_image(rec_by_sid[sid]); batch=make_batch_from_image(processor,device,image,m); sp,rp=pos_by_sid[sid]
                        pred,text,dn=patched_generate(model,processor,decoder_layers,batch,L,sp,rp,d,alpha,a.max_new_tokens)
                        per.append(dict(condition="repair",sid=sid,layer=L,alpha=alpha,gt=tgt,source=src,target=tgt,
                                        baseline_pred=b["baseline_pred"],patched_pred=pred,patched_correct=pred==tgt,
                                        target_follow=pred==tgt,changed=pred!=b["baseline_pred"],delta_norm=dn,text=text))
                    finally:
                        if image is not None:
                            with contextlib.suppress(Exception): image.close()
                        del batch; gc.collect()

                # HARM: GT -> exact opposite, only baseline-correct samples
                harm_set=[x for x in baseline if x["baseline_correct"]]
                for b in tqdm(harm_set,desc=f"L{L} a{alpha:g} harm",leave=False):
                    sid=int(b["sid"]); m=m_by_sid[sid]; image=batch=None
                    src=m["gt"]; tgt=OPP[src]; d=cent[L][tgt]-cent[L][src]
                    try:
                        image=make_real_image(rec_by_sid[sid]); batch=make_batch_from_image(processor,device,image,m); sp,rp=pos_by_sid[sid]
                        pred,text,dn=patched_generate(model,processor,decoder_layers,batch,L,sp,rp,d,alpha,a.max_new_tokens)
                        per.append(dict(condition="harm",sid=sid,layer=L,alpha=alpha,gt=src,source=src,target=tgt,
                                        baseline_pred=b["baseline_pred"],patched_pred=pred,patched_correct=pred==src,
                                        target_follow=pred==tgt,changed=pred!=b["baseline_pred"],delta_norm=dn,text=text))
                    finally:
                        if image is not None:
                            with contextlib.suppress(Exception): image.close()
                        del batch; gc.collect()

                rr=[x for x in per if x["condition"]=="repair" and x["layer"]==L and x["alpha"]==alpha]
                hh=[x for x in per if x["condition"]=="harm" and x["layer"]==L and x["alpha"]==alpha]
                w2c=sum(bool(x["patched_correct"]) for x in rr); c2w=sum(not bool(x["patched_correct"]) for x in hh)
                row=dict(layer=L,alpha=alpha,N=N,baseline_acc=base_acc,N_correct=Nc,N_wrong=Nw,
                         repair_eligible=len(rr),W2C=w2c,repair_rate=w2c/max(len(rr),1),
                         repair_only_acc=(Nc+w2c)/max(N,1),repair_follow=traj.safe_mean(float(x["target_follow"]) for x in rr),
                         C2W=c2w,harm_rate=c2w/max(len(hh),1),harm_only_acc=(Nc-c2w)/max(N,1),
                         harm_follow=traj.safe_mean(float(x["target_follow"]) for x in hh))
                summary.append(row); traj.write_csv(out/"per_sample.csv",per); traj.write_csv(out/"summary.csv",summary)
                print(f"L{L:02d} a={alpha:g} | base={base_acc:.4f} | REPAIR W2C={w2c}/{len(rr)}={row['repair_rate']:.3f} "
                      f"repair-only-acc={row['repair_only_acc']:.4f} | HARM C2W={c2w}/{len(hh)}={row['harm_rate']:.3f} "
                      f"harm-only-acc={row['harm_only_acc']:.4f} | follow repair/harm={row['repair_follow']:.3f}/{row['harm_follow']:.3f}")

        (out/"metadata.json").write_text(json.dumps(dict(model=spec.repo_id,layers=layers_req,alphas=alphas,
            train_ratio=a.train_ratio,seed=a.seed,object_state=a.object_state,
            pair_state="[(h_subject-h_reference)_real]-[(h_subject-h_reference)_gray]",
            gray_value=a.gray_value,
            direction="mu_target_realminusgray(train)-mu_source_realminusgray(train)",
            patch="REAL test image only: subject += .5*alpha*d; reference -= .5*alpha*d; one layer only",
            repair="baseline-wrong: baseline_pred -> GT",harm="baseline-correct: GT -> opposite(GT)"),indent=2),encoding="utf-8")
        print("Saved:",out)
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__": main()
