#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_small_causal_probe_qwen25_v1.py

Training-free routing by SMALL CAUSAL PROBES for the already-discovered
Real-Gray late last-token actuator.

Core idea
---------
We already have TRAIN-derived relation-specific late directions:

    Delta_i,l   = h_last(real)_i,l - h_last(gray)_i,l
    mu_r,l      = mean(Delta_i,l | y_i = r)
    mu_global,l = balanced mean_r(mu_r,l)
    s_r,l       = mu_r,l - mu_global,l

For each TEST sample, do NOT classify hidden states and do NOT use cosine.

Instead, for each candidate relation r in {L,R,A,B}, apply a very small
intervention epsilon * s_r on the existing late actuator trajectory and inspect
the model's own first-answer-token relation logits.

Main selector score
-------------------
Let m_r(x) be the target-vs-best-other relation logit margin for relation r.

    gain_real(r)
      = m_r( real + eps*s_r ) - m_r(real)

    gain_gray(r)
      = m_r( gray + eps*s_r ) - m_r(gray)

    interaction_margin(r)
      = gain_real(r) - gain_gray(r)

Prediction:

    r_hat = argmax_r interaction_margin(r)

Why subtract Gray?
------------------
A direction s_r is intrinsically designed to push relation r, so comparing only
its raw effect on Real can trivially favor whichever actuator is strongest.
Subtracting the same probe's generic effect on Gray asks instead:

    Which causal direction interacts most strongly with the ACTUAL visual state?

No selector classifier is trained.
No TEST labels are used for routing.
GT is used only for evaluation metrics.

The script also reports several diagnostic scores from the SAME probe forwards:
  - interaction_margin   [main]
  - interaction_selflogit
  - real_margin_gain
  - postprobe_rg_margin

Then, for ONE chosen score (default interaction_margin), it actually runs
full-scale model.generate() steering and reports:
  baseline -> edited accuracy
  W2C / C2W / net
for both:
  - all
  - conflict_only

This reuses utilities from:
    eval_cosine_confidence_selector_qwen25_v1.py

Example
-------
CUDA_VISIBLE_DEVICES=0 python eval_small_causal_probe_qwen25_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --prior-output-dir output/qwen3b_last_causal_auto_v1 \
  --train-delta-cache output/qwen3b_real_gray_increment_similarity_v1/real_gray_last_deltas.npz \
  --probe-scale 0.05 \
  --full-scale 1.0 \
  --selector-score interaction_margin \
  --output-dir output/qwen3b_small_causal_probe_v1 \
  --overwrite

Notes
-----
1) --train-delta-cache may contain a SUPERSET of actuator layers (e.g. L18-35).
   This script will subset it to the known actuator window.
2) Probe forwards use model(...) only; they do NOT generate text.
3) Full steering uses the existing real model.generate() path.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import eval_cosine_confidence_selector_qwen25_v1 as base


RELS = base.RELS
RELSET = set(RELS)
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--model-id", required=True)
    p.add_argument("--prior-output-dir", required=True)

    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--annotation-json",
        default="data/coco_qa_two_obj.json",
    )
    p.add_argument("--data-root", default="data")

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
        help="TRAIN filter used to construct the existing Real-Gray actuator.",
    )
    p.add_argument("--gray-value", type=int, default=128)

    p.add_argument(
        "--probe-scale",
        type=float,
        default=0.05,
        help="epsilon for the small causal probe.",
    )
    p.add_argument(
        "--probe-layers",
        default="auto",
        help=(
            "Layers receiving the small probe. "
            "auto = known actuator trajectory (3B L32-35, 7B L25-27). "
            "You may also pass e.g. '32' or '32-35'."
        ),
    )

    p.add_argument(
        "--selector-score",
        default="interaction_margin",
        choices=[
            "interaction_margin",
            "interaction_selflogit",
            "real_margin_gain",
            "postprobe_rg_margin",
        ],
        help="Which training-free probe score routes the final full intervention.",
    )

    p.add_argument(
        "--full-scale",
        type=float,
        default=1.0,
        help="Scale used by the actual final steering generate().",
    )
    p.add_argument(
        "--full-edit-mode",
        default="add",
        choices=["add", "contrast"],
        help=(
            "Actual final steering mode. 'add' adds selected s_r. "
            "'contrast' adds s_selected - s_baseline."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument(
        "--train-delta-cache",
        default="",
        help=(
            "Optional existing Real-Gray last-token npz cache. "
            "It may contain a superset of actuator layers. "
            "If omitted, a TRAIN-only actuator cache is built in output-dir."
        ),
    )

    p.add_argument(
        "--skip-final-generate",
        action="store_true",
        help="Only evaluate probe routing; do not run full steering generation.",
    )

    p.add_argument(
        "--limit-test",
        type=int,
        default=0,
        help="Debug only: if >0, evaluate only first N TEST samples.",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_layer_spec(spec):
    return base.parse_layer_spec(spec)


def resolve_probe_layers(spec, preset, n_layers):
    if str(spec).strip().lower() == "auto":
        layers = list(preset["actuator_layers"])
    else:
        layers = parse_layer_spec(spec)

    if not layers:
        raise ValueError("No probe layers selected.")

    bad = [x for x in layers if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(
            f"Invalid probe layers {bad}; decoder has blocks 0..{n_layers-1}"
        )

    return sorted(set(layers))


def load_cached_subset(cache_path, wanted_layers, wanted_sids):
    """
    Load an existing npz delta cache and subset a superset of layers/sids.
    Returns None if cache_path is empty or absent.
    """
    if not cache_path:
        return None

    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        layers = np.asarray(z["layers"], dtype=np.int64).tolist()
        deltas = np.asarray(z["deltas"], dtype=np.float32)

    layer_to_idx = {int(l): i for i, l in enumerate(layers)}
    missing_layers = [l for l in wanted_layers if l not in layer_to_idx]
    if missing_layers:
        raise RuntimeError(
            f"Cache {path} lacks requested layers {missing_layers}; "
            f"available={layers}"
        )

    sid_to_idx = {int(s): i for i, s in enumerate(sids.tolist())}
    missing_sids = [int(s) for s in wanted_sids if int(s) not in sid_to_idx]
    if missing_sids:
        raise RuntimeError(
            f"Cache {path} lacks {len(missing_sids)} requested TRAIN samples. "
            f"First missing={missing_sids[:10]}"
        )

    cols = [layer_to_idx[l] for l in wanted_layers]
    result = {
        int(sid): deltas[sid_to_idx[int(sid)]][cols].astype(np.float32)
        for sid in wanted_sids
    }

    print(
        f"[reuse] delta cache subset: {path} | "
        f"cached_layers={layers} -> wanted={wanted_layers} | N={len(result)}"
    )
    return result


def build_train_delta_cache(
    model,
    processor,
    decoder_layers,
    records,
    train_sids,
    actuator_layers,
    args,
    outdir,
):
    cached = load_cached_subset(
        args.train_delta_cache,
        actuator_layers,
        train_sids,
    )
    if cached is not None:
        return cached

    cache_path = outdir / "train_actuator_real_gray_deltas.npz"

    # Build only TRAIN examples, because TEST deltas are not needed by this selector.
    train_records = {sid: records[sid] for sid in train_sids}

    return base.build_or_load_delta_cache(
        model=model,
        processor=processor,
        layers=decoder_layers,
        records=train_records,
        selected_layers=actuator_layers,
        args=args,
        cache_path=cache_path,
    )


def relation_token_ids(tokenizer):
    """
    Resolve one canonical single token for each relation.
    Prefer the plain lowercase relation because generation starts after
    the assistant prompt/newline in Qwen chat templates.
    Fall back to a leading-space token if necessary.
    """
    result = {}

    for relation in RELS:
        candidates = [
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ]

        chosen = None
        tried = []

        for text in candidates:
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            tried.append((text, ids))
            if len(ids) == 1:
                chosen = int(ids[0])
                break

        if chosen is None:
            raise RuntimeError(
                f"Could not find a single-token form for relation={relation}. "
                f"Tried={tried}. Use a sequence-logprob implementation for this tokenizer."
            )

        result[relation] = chosen

    print("\nRelation answer token IDs:")
    for relation in RELS:
        tid = result[relation]
        decoded = tokenizer.decode([tid])
        print(f"  {relation:>5s}: id={tid} decoded={decoded!r}")

    if len(set(result.values())) != len(RELS):
        raise RuntimeError(f"Relation token IDs are not unique: {result}")

    return result


def extract_relation_scores(logits_last, token_ids):
    """
    logits_last: [vocab]
    returns raw next-token logits for four relation answer tokens.
    """
    return {
        relation: float(logits_last[token_ids[relation]].item())
        for relation in RELS
    }


def margins(scores):
    """
    target-vs-best-other margin for every relation.
    """
    return {
        relation: float(
            scores[relation]
            - max(scores[other] for other in RELS if other != relation)
        )
        for relation in RELS
    }


def forward_relation_scores(
    model,
    batch,
    token_ids,
):
    with torch.inference_mode():
        outputs = model(
            **batch,
            use_cache=False,
            return_dict=True,
        )

    logits_last = outputs.logits[0, -1].float()
    scores = extract_relation_scores(logits_last, token_ids)

    del logits_last
    del outputs

    return scores


def probed_relation_scores(
    model,
    decoder_layers,
    templates,
    probe_layers,
    target,
    scale,
    batch,
    token_ids,
):
    with base.SteerLast(
        layers=decoder_layers,
        templates=templates,
        selected=probe_layers,
        target=target,
        scale=scale,
        mode="add",
        source=None,
    ):
        return forward_relation_scores(
            model,
            batch,
            token_ids,
        )


def winner_and_margin(score_dict):
    ordered = sorted(
        score_dict.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return (
        ordered[0][0],
        float(ordered[0][1] - ordered[1][1]),
    )


def compute_probe_for_sample(
    model,
    processor,
    decoder_layers,
    templates,
    probe_layers,
    rec,
    real_image,
    gray_image,
    args,
    token_ids,
):
    real_batch = base.build_batch(
        processor,
        real_image,
        rec,
        args,
    )
    gray_batch = base.build_batch(
        processor,
        gray_image,
        rec,
        args,
    )

    # Unperturbed model evidence.
    real_base = forward_relation_scores(
        model,
        real_batch,
        token_ids,
    )
    gray_base = forward_relation_scores(
        model,
        gray_batch,
        token_ids,
    )

    real_base_margin = margins(real_base)
    gray_base_margin = margins(gray_base)

    probe_real = {}
    probe_gray = {}

    # Four candidate causal probes.
    for relation in RELS:
        probe_real[relation] = probed_relation_scores(
            model=model,
            decoder_layers=decoder_layers,
            templates=templates,
            probe_layers=probe_layers,
            target=relation,
            scale=args.probe_scale,
            batch=real_batch,
            token_ids=token_ids,
        )

        probe_gray[relation] = probed_relation_scores(
            model=model,
            decoder_layers=decoder_layers,
            templates=templates,
            probe_layers=probe_layers,
            target=relation,
            scale=args.probe_scale,
            batch=gray_batch,
            token_ids=token_ids,
        )

    score_modes = {
        "interaction_margin": {},
        "interaction_selflogit": {},
        "real_margin_gain": {},
        "postprobe_rg_margin": {},
    }

    raw = {}

    for relation in RELS:
        pr_margin = margins(probe_real[relation])
        pg_margin = margins(probe_gray[relation])

        gain_real_margin = (
            pr_margin[relation] - real_base_margin[relation]
        )
        gain_gray_margin = (
            pg_margin[relation] - gray_base_margin[relation]
        )

        gain_real_self = (
            probe_real[relation][relation] - real_base[relation]
        )
        gain_gray_self = (
            probe_gray[relation][relation] - gray_base[relation]
        )

        interaction_margin = (
            gain_real_margin - gain_gray_margin
        )
        interaction_selflogit = (
            gain_real_self - gain_gray_self
        )

        # This mixes the original Real-Gray evidence with the small probe response.
        postprobe_rg_margin = (
            pr_margin[relation] - pg_margin[relation]
        )

        score_modes["interaction_margin"][relation] = float(
            interaction_margin
        )
        score_modes["interaction_selflogit"][relation] = float(
            interaction_selflogit
        )
        score_modes["real_margin_gain"][relation] = float(
            gain_real_margin
        )
        score_modes["postprobe_rg_margin"][relation] = float(
            postprobe_rg_margin
        )

        raw[relation] = {
            "base_real_logit": real_base[relation],
            "base_gray_logit": gray_base[relation],
            "base_real_margin": real_base_margin[relation],
            "base_gray_margin": gray_base_margin[relation],
            "probe_real_target_logit": probe_real[relation][relation],
            "probe_gray_target_logit": probe_gray[relation][relation],
            "probe_real_target_margin": pr_margin[relation],
            "probe_gray_target_margin": pg_margin[relation],
            "gain_real_margin": gain_real_margin,
            "gain_gray_margin": gain_gray_margin,
            "gain_real_selflogit": gain_real_self,
            "gain_gray_selflogit": gain_gray_self,
            "interaction_margin": interaction_margin,
            "interaction_selflogit": interaction_selflogit,
            "postprobe_rg_margin": postprobe_rg_margin,
        }

    predictions = {}
    for mode, scores in score_modes.items():
        pred, margin = winner_and_margin(scores)
        predictions[mode] = {
            "pred": pred,
            "confidence": margin,
            "scores": scores,
        }

    del real_batch
    del gray_batch

    return predictions, raw


def baseline_correct(existing_test, records, sid):
    pred = existing_test[sid].get("pred")
    if pred not in RELSET:
        return pred, 0
    return pred, int(pred == records[sid]["gt"])


def summarize_selector(rows, mode):
    n = len(rows)
    wrong = [r for r in rows if r["baseline_correct"] == 0]
    correct = [r for r in rows if r["baseline_correct"] == 1]

    pred_key = f"{mode}_pred"
    conf_key = f"{mode}_confidence"

    selector_acc = (
        sum(r[pred_key] == r["gt"] for r in rows) / n
        if n else float("nan")
    )
    wrong_acc = (
        sum(r[pred_key] == r["gt"] for r in wrong) / len(wrong)
        if wrong else float("nan")
    )
    correct_acc = (
        sum(r[pred_key] == r["gt"] for r in correct) / len(correct)
        if correct else float("nan")
    )

    # If one naively replaced baseline answer by the selector label:
    w2c = sum(
        r["baseline_correct"] == 0 and r[pred_key] == r["gt"]
        for r in rows
    )
    c2w = sum(
        r["baseline_correct"] == 1 and r[pred_key] != r["gt"]
        for r in rows
    )

    conflict = [
        r for r in rows
        if r["baseline_pred"] in RELSET
        and r[pred_key] in RELSET
        and r["baseline_pred"] != r[pred_key]
    ]
    conflict_wrong = [
        r for r in conflict
        if r["baseline_correct"] == 0
    ]

    return {
        "mode": mode,
        "N": n,
        "selector_acc": selector_acc,
        "baseline_wrong_N": len(wrong),
        "selector_acc_on_baseline_wrong": wrong_acc,
        "baseline_correct_N": len(correct),
        "selector_acc_on_baseline_correct": correct_acc,
        "potential_W2C": w2c,
        "potential_C2W": c2w,
        "potential_net": w2c - c2w,
        "conflict_N": len(conflict),
        "conflict_rate": len(conflict) / n if n else float("nan"),
        "conflict_baseline_wrong_N": len(conflict_wrong),
        "selector_acc_on_conflict_baseline_wrong": (
            sum(r[pred_key] == r["gt"] for r in conflict_wrong)
            / len(conflict_wrong)
            if conflict_wrong else float("nan")
        ),
        "mean_confidence": (
            float(np.mean([r[conf_key] for r in rows]))
            if rows else float("nan")
        ),
    }


def run_full_steering(
    model,
    processor,
    decoder_layers,
    templates,
    actuator_layers,
    records,
    rows,
    existing_test,
    selector_mode,
    args,
):
    """
    Generate once per TEST sample with the selected direction at full scale.
    From that one edited generation we can report both:
      - apply all
      - conflict_only (use baseline when no selector/baseline conflict)
    """
    by_sid = {int(r["sid"]): r for r in rows}
    edited = {}

    for sid in tqdm(
        sorted(by_sid),
        desc=f"Full steering [{selector_mode}]",
    ):
        rec = records[sid]
        selector_pred = by_sid[sid][f"{selector_mode}_pred"]
        baseline_pred = existing_test[sid].get("pred")

        real = None
        try:
            real = Image.open(
                rec["image_path"]
            ).convert("RGB")

            batch = base.build_batch(
                processor,
                real,
                rec,
                args,
            )

            with base.SteerLast(
                layers=decoder_layers,
                templates=templates,
                selected=actuator_layers,
                target=selector_pred,
                scale=args.full_scale,
                mode=args.full_edit_mode,
                source=baseline_pred,
            ):
                text, pred = base.generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            edited[sid] = {
                "edit_pred": pred,
                "edit_text": text,
                "edit_correct": int(pred == rec["gt"]),
            }

            del batch

        finally:
            if real is not None:
                real.close()
            cleanup()

    baseline_acc = np.mean(
        [
            baseline_correct(existing_test, records, sid)[1]
            for sid in sorted(by_sid)
        ]
    )

    summaries = []
    details = []

    for apply_mode in ["all", "conflict_only"]:
        final_corrects = []
        w2c = 0
        c2w = 0
        applied = 0

        for sid in sorted(by_sid):
            gt = records[sid]["gt"]
            base_pred, base_ok = baseline_correct(
                existing_test,
                records,
                sid,
            )
            selector_pred = by_sid[sid][f"{selector_mode}_pred"]

            do_apply = (
                apply_mode == "all"
                or (
                    base_pred in RELSET
                    and selector_pred in RELSET
                    and base_pred != selector_pred
                )
            )

            if do_apply:
                final_pred = edited[sid]["edit_pred"]
                final_ok = edited[sid]["edit_correct"]
                applied += 1
            else:
                final_pred = base_pred
                final_ok = base_ok

            w2c += int(base_ok == 0 and final_ok == 1)
            c2w += int(base_ok == 1 and final_ok == 0)
            final_corrects.append(final_ok)

            details.append({
                "apply_mode": apply_mode,
                "sid": sid,
                "gt": gt,
                "baseline_pred": base_pred or "",
                "selector_pred": selector_pred,
                "edit_pred": edited[sid]["edit_pred"] or "",
                "final_pred": final_pred or "",
                "baseline_correct": base_ok,
                "edit_correct": edited[sid]["edit_correct"],
                "final_correct": final_ok,
                "applied": int(do_apply),
                "W2C": int(base_ok == 0 and final_ok == 1),
                "C2W": int(base_ok == 1 and final_ok == 0),
            })

        final_acc = float(np.mean(final_corrects))
        summaries.append({
            "selector_mode": selector_mode,
            "probe_scale": args.probe_scale,
            "full_scale": args.full_scale,
            "full_edit_mode": args.full_edit_mode,
            "apply_mode": apply_mode,
            "N": len(by_sid),
            "baseline_acc": float(baseline_acc),
            "final_acc": final_acc,
            "gain": final_acc - float(baseline_acc),
            "applied": applied,
            "W2C": w2c,
            "C2W": c2w,
            "net": w2c - c2w,
        })

    return summaries, details


def main():
    args = parse_args()

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    existing_test, baseline_path = base.load_existing_test_baseline(
        args.prior_output_dir
    )
    if existing_test is None:
        raise RuntimeError(
            "Could not recover existing TEST baseline from --prior-output-dir."
        )

    train_generation, train_gen_path = base.load_existing_train_generation(
        args.prior_output_dir
    )

    records = base.load_all_records(args)
    train_sids, test_sids = base.derive_train_test_ids(
        records,
        existing_test,
    )

    if args.limit_test > 0:
        test_sids = test_sids[: args.limit_test]

    preset = base.model_preset(args.model_id)
    actuator_layers = list(preset["actuator_layers"])

    print("\n" + "=" * 132)
    print("SMALL CAUSAL PROBE — TRAINING-FREE DIRECTION ROUTER")
    print("=" * 132)
    print(f"model={args.model_id}")
    print(f"TRAIN={len(train_sids)} TEST={len(test_sids)}")
    print(f"baseline_source={baseline_path}")
    print(f"train_generation_source={train_gen_path}")
    print(f"actuator_layers={actuator_layers}")
    print(f"probe_scale={args.probe_scale}")
    print(f"selector_score={args.selector_score}")
    print(f"full_scale={args.full_scale} full_edit_mode={args.full_edit_mode}")
    print("=" * 132)

    model, processor = base.load_model(args)
    decoder_layers, decoder_path = base.resolve_layers(model)

    probe_layers = resolve_probe_layers(
        args.probe_layers,
        preset,
        len(decoder_layers),
    )

    # Probe layers must have direction templates.
    template_layers = sorted(
        set(actuator_layers) | set(probe_layers)
    )

    print(f"decoder={decoder_path} | blocks={len(decoder_layers)}")
    print(f"probe_layers={probe_layers}")
    print(f"template_layers={template_layers}")

    # Get TRAIN Real-Gray deltas only for needed late layers.
    train_delta_cache = build_train_delta_cache(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        records=records,
        train_sids=train_sids,
        actuator_layers=template_layers,
        args=args,
        outdir=outdir,
    )

    layer_index = {
        layer: i
        for i, layer in enumerate(template_layers)
    }

    templates, filter_used = base.fit_templates(
        delta_cache=train_delta_cache,
        layer_index=layer_index,
        records=records,
        sids=train_sids,
        selected_layers=template_layers,
        train_generation=train_generation,
        requested_filter=args.template_filter,
    )

    print(f"template_filter_used={filter_used}")

    token_ids = relation_token_ids(
        processor.tokenizer
    )

    probe_rows = []

    for sid in tqdm(
        test_sids,
        desc="Small causal probes: Real + Gray x 4 directions",
    ):
        rec = records[sid]
        real = None
        gray = None

        try:
            real = Image.open(
                rec["image_path"]
            ).convert("RGB")
            gray = base.make_gray_image(
                real,
                args.gray_value,
            )

            predictions, raw = compute_probe_for_sample(
                model=model,
                processor=processor,
                decoder_layers=decoder_layers,
                templates=templates,
                probe_layers=probe_layers,
                rec=rec,
                real_image=real,
                gray_image=gray,
                args=args,
                token_ids=token_ids,
            )

            base_pred, base_ok = baseline_correct(
                existing_test,
                records,
                sid,
            )

            row = {
                "sid": sid,
                "gt": rec["gt"],
                "baseline_pred": base_pred or "",
                "baseline_correct": base_ok,
            }

            for mode, info in predictions.items():
                row[f"{mode}_pred"] = info["pred"]
                row[f"{mode}_confidence"] = info["confidence"]
                row[f"{mode}_correct"] = int(
                    info["pred"] == rec["gt"]
                )
                for relation in RELS:
                    row[f"{mode}_score_{relation}"] = (
                        info["scores"][relation]
                    )

            # Save raw causal-response decomposition.
            for relation in RELS:
                for key, value in raw[relation].items():
                    row[f"{relation}_{key}"] = value

            probe_rows.append(row)

        finally:
            if real is not None:
                real.close()
            if gray is not None:
                gray.close()
            cleanup()

    write_csv(
        outdir / "test_small_probe_predictions.csv",
        probe_rows,
    )

    selector_modes = [
        "interaction_margin",
        "interaction_selflogit",
        "real_margin_gain",
        "postprobe_rg_margin",
    ]

    selector_summaries = [
        summarize_selector(
            probe_rows,
            mode,
        )
        for mode in selector_modes
    ]

    write_csv(
        outdir / "test_small_probe_selector_summary.csv",
        selector_summaries,
    )

    baseline_acc = float(
        np.mean([r["baseline_correct"] for r in probe_rows])
    )

    print("\nPROBE SELECTOR RESULTS")
    print("-" * 132)
    print(f"baseline generation acc = {baseline_acc:.4f}")
    for s in selector_summaries:
        print(
            f"{s['mode']:24s} | "
            f"sel_acc={s['selector_acc']:.4f} "
            f"({s['selector_acc'] - baseline_acc:+.4f} vs baseline label acc) | "
            f"wrong_acc={s['selector_acc_on_baseline_wrong']:.4f} | "
            f"correct_acc={s['selector_acc_on_baseline_correct']:.4f} | "
            f"potential W2C/C2W/net="
            f"{s['potential_W2C']}/{s['potential_C2W']}/{s['potential_net']:+d} | "
            f"conflict={s['conflict_N']} | "
            f"conflict_wrong_acc={s['selector_acc_on_conflict_baseline_wrong']:.4f}"
        )

    if args.skip_final_generate:
        print("\n[skip] final model.generate() steering disabled.")
        print(f"[saved] {outdir / 'test_small_probe_predictions.csv'}")
        print(f"[saved] {outdir / 'test_small_probe_selector_summary.csv'}")
        return

    # Actual final steering using ONE selected routing score.
    final_summaries, final_details = run_full_steering(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        templates=templates,
        actuator_layers=actuator_layers,
        records=records,
        rows=probe_rows,
        existing_test=existing_test,
        selector_mode=args.selector_score,
        args=args,
    )

    write_csv(
        outdir / "test_small_probe_final_steering_summary.csv",
        final_summaries,
    )
    write_csv(
        outdir / "test_small_probe_final_steering_details.csv",
        final_details,
    )

    print("\nACTUAL model.generate() — SMALL-PROBE ROUTED STEERING")
    print("-" * 132)
    for s in final_summaries:
        print(
            f"{s['apply_mode']:13s} | "
            f"acc {s['baseline_acc']:.4f}->{s['final_acc']:.4f} "
            f"{s['gain']:+.4f} | "
            f"applied={s['applied']:3d} | "
            f"W2C={s['W2C']:3d} "
            f"C2W={s['C2W']:3d} "
            f"net={s['net']:+d}"
        )

    print("-" * 132)
    print(f"[saved] {outdir / 'test_small_probe_predictions.csv'}")
    print(f"[saved] {outdir / 'test_small_probe_selector_summary.csv'}")
    print(f"[saved] {outdir / 'test_small_probe_final_steering_summary.csv'}")
    print(f"[saved] {outdir / 'test_small_probe_final_steering_details.csv'}")


if __name__ == "__main__":
    main()
