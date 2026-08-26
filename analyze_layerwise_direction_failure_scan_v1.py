#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer-wise spatial direction trajectory scan for AdaptVis.

For each decoder layer:
    r_img      = h_img(subject) - h_img(reference)
    r_noimg    = h_noimg(subject) - h_noimg(reference)
    r_residual = r_img - r_noimg

Fit four relation prototypes (left/right/above/below) on TRAIN samples,
classify TEST samples by cosine similarity, and compare the trajectories of
actual generation-correct vs generation-wrong samples.

This is intentionally the cheap first-stage diagnosis. It does NOT perform
causal subspace removal yet.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base


SCRIPT_VERSION = "layerwise-direction-failure-scan-v1"
RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two", "vg_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b", choices=sorted(base.SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--prompt-template", default=(
        "Determine the spatial relation of the {subject} to the {reference} "
        "in the image. Answer with left, right, above, or below."
    ))
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stable-k", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def norm_relation(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_")
    aliases = {
        "left": "left", "right": "right",
        "above": "above", "on": "above", "top": "above", "over": "above",
        "below": "below", "under": "below", "underneath": "below", "bottom": "below",
    }
    return aliases.get(s, s)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def build_chat_prompt(processor, question: str, with_image: bool) -> str:
    content = ([{"type": "image"}] if with_image else []) + [
        {"type": "text", "text": question}
    ]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def process_inputs(processor, rendered: str, image: Optional[Image.Image], device):
    if image is None:
        batch = processor(text=[rendered], padding=True, return_tensors="pt")
    else:
        batch = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    return batch.to(device)


def find_subsequence_last(haystack: Sequence[int], needle: Sequence[int]):
    best = None
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if list(haystack[i:i+n]) == list(needle):
            best = (i, i+n)
    return best


def locate_phrase_positions(tokenizer, input_ids: Sequence[int], phrase: str) -> List[int]:
    hits = []
    for text in (str(phrase), " " + str(phrase)):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            ids = []
        if ids:
            hit = find_subsequence_last(input_ids, ids)
            if hit is not None:
                hits.append(hit)
    if hits:
        s, e = max(hits, key=lambda x: x[0])
        return list(range(s, e))
    idx = int(base.find_phrase_last_token(tokenizer, list(input_ids), str(phrase)))
    return [idx]


def pool_positions(tensor: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    valid = [int(p) for p in positions if 0 <= int(p) < int(tensor.shape[1])]
    if not valid:
        raise RuntimeError("No valid object-token positions")
    if mode == "last":
        return tensor[0, valid[-1]]
    idx = torch.as_tensor(valid, device=tensor.device, dtype=torch.long)
    return tensor[0].index_select(0, idx).mean(dim=0)


def hidden_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
        getattr(getattr(outputs, "text_model_output", None), "hidden_states", None),
    ]
    for states in candidates:
        if isinstance(states, (tuple, list)) and states and torch.is_tensor(states[-1]):
            return tuple(states)
    raise RuntimeError("No decoder hidden_states returned")


def parse_generated_relation(text: str) -> Optional[str]:
    s = text.strip().lower()
    pats = [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"),
        ("below", r"\bunder(?:neath)?\b"),
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


def capture_layer_vectors(model, processor, device, question: str,
                          subject: str, reference: str,
                          image: Optional[Image.Image], pool: str) -> Tuple[np.ndarray, int]:
    rendered = build_chat_prompt(processor, question, image is not None)
    batch = process_inputs(processor, rendered, image, device)
    input_ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    subj_pos = locate_phrase_positions(processor.tokenizer, input_ids, subject)
    ref_pos = locate_phrase_positions(processor.tokenizer, input_ids, reference)

    with torch.inference_mode():
        outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)

    states = hidden_tuple(outputs)
    n_blocks = len(states) - 1
    if n_blocks <= 0:
        raise RuntimeError("No decoder blocks found")

    final = states[-1]
    if final.ndim != 3 or final.shape[0] != 1:
        raise RuntimeError(f"Unexpected hidden-state shape {tuple(final.shape)}")
    if int(final.shape[1]) != len(input_ids):
        raise RuntimeError(
            "Text positions do not align with hidden states: "
            f"input_len={len(input_ids)} hidden_len={int(final.shape[1])}"
        )

    vecs = []
    for block in range(n_blocks):
        h = states[block + 1]  # state[0] = embedding output
        hs = pool_positions(h, subj_pos, pool)
        hr = pool_positions(h, ref_pos, pool)
        vecs.append((hs - hr).detach().float().cpu().numpy())

    arr = np.stack(vecs, axis=0)
    del outputs, states, batch
    return arr, n_blocks


def generate_real_answer(model, processor, device, question: str,
                         image: Image.Image, max_new_tokens: int) -> Tuple[str, Optional[str]]:
    rendered = build_chat_prompt(processor, question, True)
    batch = process_inputs(processor, rendered, image, device)
    input_len = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    suffix_ids = generated[0, input_len:]
    text = processor.tokenizer.decode(suffix_ids, skip_special_tokens=True).strip()
    rel = parse_generated_relation(text)
    del batch, generated
    return text, rel


def stratified_split(labels: np.ndarray, train_ratio: float, seed: int):
    rng = random.Random(seed)
    train, test = [], []
    for rel in RELATIONS:
        ids = np.flatnonzero(labels == rel).tolist()
        rng.shuffle(ids)
        if len(ids) < 2:
            raise RuntimeError(f"Need >=2 samples for {rel}, got {len(ids)}")
        ntr = int(round(len(ids) * train_ratio))
        ntr = max(1, min(len(ids) - 1, ntr))
        train.extend(ids[:ntr])
        test.extend(ids[ntr:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def fit_codebook(X_train: np.ndarray, y_train: np.ndarray):
    # Same idea as current Direction-head analysis:
    # center globally, then average same-relation samples to suppress object/sample noise.
    center = X_train.mean(axis=0)
    Xc = X_train - center
    dirs = []
    for rel in RELATIONS:
        m = y_train == rel
        if not np.any(m):
            raise RuntimeError(f"No training samples for {rel}")
        d = Xc[m].mean(axis=0)
        d = d / max(float(np.linalg.norm(d)), EPS)
        dirs.append(d)
    return center, np.stack(dirs, axis=0)


def score_codebook(X: np.ndarray, center: np.ndarray, dirs: np.ndarray):
    Xn = normalize_rows(X - center)
    scores = Xn @ dirs.T
    pred = np.argmax(scores, axis=1)
    return pred, scores


def first_true(mask: Sequence[bool]) -> Optional[int]:
    for i, v in enumerate(mask):
        if v:
            return i
    return None


def stable_onset(mask: Sequence[bool], k: int) -> Optional[int]:
    if k <= 1:
        return first_true(mask)
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= k:
            return i - k + 1
    return None


def longest_true_run(mask: Sequence[bool]) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def failure_type(gen_group: str, mask: Sequence[bool], stable_k: int) -> str:
    if gen_group == "correct":
        return "generation_correct"
    if gen_group == "unparsed":
        return "generation_unparsed"
    if not any(mask):
        return "never_acquired"
    onset = stable_onset(mask, stable_k)
    final_stable = all(mask[-stable_k:]) if len(mask) >= stable_k else False
    if onset is None:
        return "transient_or_unstable"
    if final_stable:
        return "internal_correct_output_wrong"
    return "corruption_after_acquisition"


def extract_or_load(args, out: Path) -> Dict[str, np.ndarray]:
    cache = out / "vectors.npz"
    if cache.exists() and not args.overwrite:
        print(f"[cache] loading {cache}")
        with np.load(cache, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    records, audit = base.load_records(args.dataset, Path(args.data_root), args.max_samples)
    records = [r for r in records if norm_relation(r.relation) in RELATIONS]
    if not records:
        raise RuntimeError("No usable four-direction records")
    print(f"[{args.dataset}] n={len(records)} counts={dict(Counter(norm_relation(r.relation) for r in records))}")

    spec = base.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")

    kw = dict(
        torch_dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] {spec.repo_id} -> {args.device}")
    model = model_cls.from_pretrained(spec.repo_id, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    sids, labels, subjects, references, image_ids = [], [], [], [], []
    img_vecs, noimg_vecs, residual_vecs = [], [], []
    gen_texts, gen_preds, gen_groups = [], [], []
    errors = []
    n_blocks_ref = None

    def save_partial():
        if not img_vecs:
            return
        atomic_npz(
            cache,
            sample_index=np.asarray(sids, dtype=np.int64),
            relation=np.asarray(labels),
            subject=np.asarray(subjects),
            reference=np.asarray(references),
            image_id=np.asarray(image_ids),
            img=np.stack(img_vecs).astype(dtype_np),
            no_image=np.stack(noimg_vecs).astype(dtype_np),
            residual=np.stack(residual_vecs).astype(dtype_np),
            generation_text=np.asarray(gen_texts),
            generation_pred=np.asarray(gen_preds),
            generation_group=np.asarray(gen_groups),
            decoder_block_index=np.arange(n_blocks_ref, dtype=np.int64),
        )

    for rec in tqdm(records, desc=f"extract:{args.model}"):
        image = None
        try:
            gt = norm_relation(rec.relation)
            q = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
            image = Image.open(rec.image_path).convert("RGB")

            vi, nbi = capture_layer_vectors(
                model, processor, device, q, str(rec.subject), str(rec.reference), image, args.pool
            )
            vn, nbn = capture_layer_vectors(
                model, processor, device, q, str(rec.subject), str(rec.reference), None, args.pool
            )
            if nbi != nbn:
                raise RuntimeError(f"img/noimg block mismatch {nbi} vs {nbn}")
            if n_blocks_ref is None:
                n_blocks_ref = nbi
                print(f"[shape] decoder_blocks={nbi} hidden={vi.shape[-1]} pool={args.pool}")
            elif nbi != n_blocks_ref:
                raise RuntimeError(f"block count changed {nbi} vs {n_blocks_ref}")

            vr = vi - vn
            gen_text, gen_pred = generate_real_answer(
                model, processor, device, q, image, args.max_new_tokens
            )
            if gen_pred is None:
                gg = "unparsed"
            elif gen_pred == gt:
                gg = "correct"
            else:
                gg = "wrong"

            sids.append(int(rec.sid))
            labels.append(gt)
            subjects.append(str(rec.subject))
            references.append(str(rec.reference))
            image_ids.append(str(rec.image_id))
            img_vecs.append(vi.astype(dtype_np))
            noimg_vecs.append(vn.astype(dtype_np))
            residual_vecs.append(vr.astype(dtype_np))
            gen_texts.append(gen_text)
            gen_preds.append("" if gen_pred is None else gen_pred)
            gen_groups.append(gg)

            if len(img_vecs) % args.save_every == 0:
                save_partial()
            del vi, vn, vr

        except Exception as e:
            errors.append({
                "sid": int(rec.sid),
                "image_id": str(rec.image_id),
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback_tail": traceback.format_exc().splitlines()[-10:],
            })
            tqdm.write(f"[ERROR] sid={rec.sid}: {type(e).__name__}: {str(e)[:180]}")
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_partial()
    (out / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    (out / "extraction_meta.json").write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "repo_id": spec.repo_id,
        "dataset": args.dataset,
        "n_saved": len(sids),
        "n_errors": len(errors),
        "pool": args.pool,
        "prompt_template": args.prompt_template,
        "vector_definition": {
            "img": "h_img(subject)-h_img(reference)",
            "no_image": "h_noimg(subject)-h_noimg(reference)",
            "residual": "r_img-r_noimg",
        },
    }, indent=2), encoding="utf-8")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with np.load(cache, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def analyze(args, out: Path, data: Dict[str, np.ndarray]) -> None:
    y = np.asarray([norm_relation(v) for v in data["relation"]])
    sids = data["sample_index"].astype(np.int64)
    gen_group = np.asarray(data["generation_group"]).astype(str)
    gen_pred = np.asarray(data["generation_pred"]).astype(str)
    gen_text = np.asarray(data["generation_text"]).astype(str)

    N = len(y)
    train_idx, test_idx = stratified_split(y, args.train_ratio, args.seed)
    is_train = np.zeros(N, dtype=bool)
    is_train[train_idx] = True

    save_csv(out / "sample_split_and_generation.csv", [
        {
            "sample_index": int(sids[i]),
            "split": "train" if is_train[i] else "test",
            "relation": y[i],
            "generation_group": gen_group[i],
            "generation_pred": gen_pred[i],
            "generation_text": gen_text[i],
        }
        for i in range(N)
    ])

    print(f"[split] train={len(train_idx)} test={len(test_idx)}")
    print(f"[generation:test] {dict(Counter(gen_group[test_idx].tolist()))}")

    layer_rows = []
    sample_layer_rows = []
    residual_correct = defaultdict(list)
    residual_margin = defaultdict(list)
    residual_pred = defaultdict(list)
    gt_all = np.asarray([REL2ID[v] for v in y], dtype=np.int64)

    for metric in ("img", "no_image", "residual"):
        Xall = np.asarray(data[metric], dtype=np.float32)
        if Xall.ndim != 3:
            raise RuntimeError(f"{metric}: expected [N,L,D], got {Xall.shape}")
        _, L, _ = Xall.shape

        for l in tqdm(range(L), desc=f"probe:{metric}", leave=False):
            X = Xall[:, l, :]
            center, dirs = fit_codebook(X[train_idx], y[train_idx])
            pred_all, scores_all = score_codebook(X, center, dirs)

            te = test_idx
            gt = gt_all[te]
            pred = pred_all[te]
            scores = scores_all[te]
            correct = pred == gt
            gt_score = scores[np.arange(len(te)), gt]
            tmp = scores.copy()
            tmp[np.arange(len(te)), gt] = -np.inf
            best_wrong = np.max(tmp, axis=1)
            margin = gt_score - best_wrong

            row = {
                "metric": metric,
                "layer": l,
                "n_test": len(te),
                "accuracy": safe_mean(correct.astype(np.float32)),
                "mean_gt_cosine": safe_mean(gt_score),
                "mean_margin": safe_mean(margin),
            }
            for rel in RELATIONS:
                m = y[te] == rel
                row[f"acc_{rel}"] = safe_mean(correct[m].astype(np.float32))
            for group in ("correct", "wrong", "unparsed"):
                m = gen_group[te] == group
                row[f"n_gen_{group}"] = int(m.sum())
                row[f"acc_gen_{group}"] = safe_mean(correct[m].astype(np.float32))
                row[f"margin_gen_{group}"] = safe_mean(margin[m])
                row[f"gtcos_gen_{group}"] = safe_mean(gt_score[m])
            layer_rows.append(row)

            for j, idx in enumerate(te):
                sid = int(sids[idx])
                pred_rel = RELATIONS[int(pred[j])]
                sample_layer_rows.append({
                    "sample_index": sid,
                    "metric": metric,
                    "layer": l,
                    "relation": y[idx],
                    "generation_group": gen_group[idx],
                    "generation_pred": gen_pred[idx],
                    "probe_pred": pred_rel,
                    "probe_correct": int(bool(correct[j])),
                    "gt_cosine": float(gt_score[j]),
                    "best_wrong_cosine": float(best_wrong[j]),
                    "margin": float(margin[j]),
                })
                if metric == "residual":
                    residual_correct[sid].append(bool(correct[j]))
                    residual_margin[sid].append(float(margin[j]))
                    residual_pred[sid].append(pred_rel)

    save_csv(out / "per_layer_summary.csv", layer_rows)
    save_csv(out / "per_sample_layer_predictions.csv", sample_layer_rows)

    trajectory_rows = []
    for idx in test_idx:
        sid = int(sids[idx])
        mask = residual_correct[sid]
        margins = residual_margin[sid]
        preds = residual_pred[sid]
        if not mask:
            continue
        fc = first_true(mask)
        so = stable_onset(mask, args.stable_k)
        final_stable = all(mask[-args.stable_k:]) if len(mask) >= args.stable_k else False
        trajectory_rows.append({
            "sample_index": sid,
            "relation": y[idx],
            "generation_group": gen_group[idx],
            "generation_pred": gen_pred[idx],
            "generation_text": gen_text[idx],
            "first_correct_layer": -1 if fc is None else fc,
            "stable_onset_layer": -1 if so is None else so,
            "final_probe_correct": int(mask[-1]),
            "final_stable_correct": int(final_stable),
            "ever_probe_correct": int(any(mask)),
            "correct_layer_fraction": float(np.mean(mask)),
            "longest_correct_run": int(longest_true_run(mask)),
            "failure_type": failure_type(gen_group[idx], mask, args.stable_k),
            "probe_correct_trajectory": "".join("1" if x else "0" for x in mask),
            "probe_pred_trajectory": "|".join(preds),
            "margin_trajectory": "|".join(f"{m:.4f}" for m in margins),
        })
    save_csv(out / "residual_trajectories.csv", trajectory_rows)

    wrong_rows = [r for r in trajectory_rows if r["generation_group"] == "wrong"]
    wrong_tax = Counter(r["failure_type"] for r in wrong_rows)
    save_csv(out / "failure_taxonomy.csv", [
        {
            "failure_type": k,
            "count": v,
            "fraction_of_generation_wrong": v / max(1, len(wrong_rows)),
        }
        for k, v in wrong_tax.most_common()
    ])

    residual_rows = [r for r in layer_rows if r["metric"] == "residual"]
    gap_rows = []
    for r in residual_rows:
        ac, aw = r["acc_gen_correct"], r["acc_gen_wrong"]
        mc, mw = r["margin_gen_correct"], r["margin_gen_wrong"]
        gap_rows.append({
            "layer": r["layer"],
            "acc_gen_correct": ac,
            "acc_gen_wrong": aw,
            "accuracy_gap_correct_minus_wrong": ac-aw if np.isfinite(ac) and np.isfinite(aw) else float("nan"),
            "margin_gen_correct": mc,
            "margin_gen_wrong": mw,
            "margin_gap_correct_minus_wrong": mc-mw if np.isfinite(mc) and np.isfinite(mw) else float("nan"),
        })
    save_csv(out / "residual_correct_vs_wrong_gap.csv", gap_rows)

    parsed_test = [r for r in trajectory_rows if r["generation_group"] != "unparsed"]
    correct_test = [r for r in trajectory_rows if r["generation_group"] == "correct"]
    summary = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "N": N,
        "train_N": int(len(train_idx)),
        "test_N": int(len(test_idx)),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "stable_k": args.stable_k,
        "generation_test_counts": dict(Counter(gen_group[test_idx].tolist())),
        "generation_accuracy_on_parsed_test": len(correct_test) / max(1, len(parsed_test)),
        "generation_wrong_failure_taxonomy": dict(wrong_tax),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*100)
    print("LAYER-WISE RESIDUAL DIRECTION SCAN")
    print("="*100)
    print(f"model={args.model} dataset={args.dataset} N={N} train={len(train_idx)} test={len(test_idx)}")
    print(f"generation parsed-test acc={summary['generation_accuracy_on_parsed_test']:.4f}")
    print("\nlayer  all_acc  C_acc   W_acc   C-W     all_margin  C_margin  W_margin")
    for r in residual_rows:
        ac, aw = r["acc_gen_correct"], r["acc_gen_wrong"]
        gap = ac-aw if np.isfinite(ac) and np.isfinite(aw) else float("nan")
        print(
            f"L{int(r['layer']):02d}   {r['accuracy']:.4f}   {ac:.4f}  {aw:.4f}  {gap:+.4f}   "
            f"{r['mean_margin']:+.4f}     {r['margin_gen_correct']:+.4f}    {r['margin_gen_wrong']:+.4f}"
        )

    print("\nGeneration-wrong taxonomy:")
    for k, v in wrong_tax.most_common():
        print(f"  {k:<32s} {v:4d} ({v/max(1,len(wrong_rows)):.3f})")

    print(f"\nSaved to {out}")
    print("Key files:")
    print("  per_layer_summary.csv")
    print("  residual_correct_vs_wrong_gap.csv")
    print("  residual_trajectories.csv")
    print("  failure_taxonomy.csv")
    print("  per_sample_layer_predictions.csv")
    print("  vectors.npz")


def main():
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if args.stable_k < 1:
        raise ValueError("--stable-k must be >=1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >=1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = extract_or_load(args, out)
    analyze(args, out, data)


if __name__ == "__main__":
    main()
