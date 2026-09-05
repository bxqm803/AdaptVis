#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_middle_gradient_selector_qwen25_v1.py

Training-free gradient-attribution selector for the existing late causal actuator.

Core:
    q_l = (h_sub-h_ref)_real - (h_sub-h_ref)_control

For each candidate relation r:
    g_pair(r,l) = 0.5 * (d z_r / d h_sub,l - d z_r / d h_ref,l)
    A_r,l       = <q_l, g_pair(r,l)>
    A_r         = mean_l A_r,l

Then:
    r_hat = argmax_r A_r

Finally add the already-established late relation-specific actuator s_{r_hat}
to the late last-token window.

No external classifier is trained.
No TEST GT is used for routing.
Exactly one actuator direction is selected per sample.

Default middle control:
    noimage

Optional:
    gray

Default guide layers:
    Qwen2.5-VL-3B: L19
    Qwen2.5-VL-7B: L14-L20

Late actuator:
    Qwen2.5-VL-3B: L32-L35
    Qwen2.5-VL-7B: L25-L27

Example:
CUDA_VISIBLE_DEVICES=0 python eval_middle_gradient_selector_qwen25_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --prior-output-dir output/qwen3b_last_causal_auto_v1 \
  --guide-layers 19 \
  --control noimage \
  --selector-score grad_dot_raw \
  --actuator-cache output/qwen3b_real_gray_increment_similarity_v1/real_gray_last_deltas.npz \
  --scale 1.0 \
  --output-dir output/qwen3b_middle_gradient_selector_v1 \
  --overwrite

Quick selector-only test:
    add --skip-steering
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import math
import shutil
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import eval_cosine_confidence_selector_qwen25_v1 as causal


RELS = ("left", "right", "above", "below")
RELSET = set(RELS)
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
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

    p.add_argument("--guide-layers", default="auto")
    p.add_argument(
        "--control",
        default="noimage",
        choices=["noimage", "gray"],
    )
    p.add_argument(
        "--pool",
        default="mean",
        choices=["mean", "last"],
    )

    p.add_argument(
        "--selector-score",
        default="grad_dot_raw",
        choices=[
            "grad_dot_raw",
            "grad_cos",
            "opposite_margin",
        ],
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )

    p.add_argument(
        "--actuator-cache",
        default="",
        help="Existing Real-Gray last-token cache; superset of late layers is okay.",
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument(
        "--edit-mode",
        default="add",
        choices=["add", "contrast"],
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--skip-steering", action="store_true")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


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
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def resolve_guide_layers(model_id, spec):
    if str(spec).strip().lower() != "auto":
        return causal.parse_layer_spec(spec)

    low = model_id.lower()
    if "3b" in low:
        return [19]
    if "7b" in low:
        return list(range(14, 21))

    raise ValueError(
        "--guide-layers auto supports Qwen2.5-VL-3B/7B only."
    )


# -----------------------------------------------------------------------------
# Subject/reference token spans
# -----------------------------------------------------------------------------

def find_all_subsequences(sequence, pattern):
    if not pattern:
        return []
    out = []
    n = len(pattern)
    for i in range(len(sequence) - n + 1):
        if list(sequence[i:i+n]) == list(pattern):
            out.append(i)
    return out


def phrase_spans(tokenizer, full_ids, phrase):
    variants = [
        phrase,
        " " + phrase,
        phrase.strip(),
        " " + phrase.strip(),
    ]

    spans = []
    seen = set()

    for variant in variants:
        ids = tokenizer.encode(
            variant,
            add_special_tokens=False,
        )
        for start in find_all_subsequences(full_ids, ids):
            span = tuple(range(start, start + len(ids)))
            if span not in seen:
                seen.add(span)
                spans.append(list(span))

    return spans


def locate_pair_spans(
    tokenizer,
    input_ids,
    subject,
    reference,
):
    subject_spans = phrase_spans(
        tokenizer,
        input_ids,
        subject,
    )
    reference_spans = phrase_spans(
        tokenizer,
        input_ids,
        reference,
    )

    if not subject_spans or not reference_spans:
        return None, None

    best = None

    for s_span in subject_spans:
        for r_span in reference_spans:
            if set(s_span) & set(r_span):
                continue

            order_penalty = int(s_span[0] >= r_span[0])
            distance = abs(
                float(np.mean(s_span))
                - float(np.mean(r_span))
            )

            score = (
                order_penalty,
                distance,
                -s_span[0],
            )

            if best is None or score < best[0]:
                best = (score, s_span, r_span)

    if best is None:
        return None, None

    return best[1], best[2]


def pool_hidden(
    hidden_2d,
    positions,
    mode,
):
    pos = [
        int(x)
        for x in positions
        if 0 <= int(x) < int(hidden_2d.shape[0])
    ]

    if not pos:
        raise RuntimeError("No valid object-token positions.")

    if mode == "last":
        return hidden_2d[pos[-1]]

    idx = torch.as_tensor(
        pos,
        device=hidden_2d.device,
        dtype=torch.long,
    )
    return hidden_2d.index_select(
        0,
        idx,
    ).mean(dim=0)


# -----------------------------------------------------------------------------
# Relation token logits
# -----------------------------------------------------------------------------

def relation_token_variants(tokenizer):
    out = {}
    unk = getattr(tokenizer, "unk_token_id", None)

    for relation in RELS:
        ids = []

        for text in [
            relation,
            " " + relation,
            "\n" + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ]:
            xx = tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if len(xx) != 1:
                continue

            tid = int(xx[0])

            if unk is not None and tid == int(unk):
                continue

            ids.append(tid)

        ids = list(dict.fromkeys(ids))

        if not ids:
            raise RuntimeError(
                f"No one-token variants for {relation}"
            )

        out[relation] = ids

    return out


def relation_scores(score_vector, token_map):
    vals = []

    for relation in RELS:
        ids = torch.as_tensor(
            token_map[relation],
            device=score_vector.device,
            dtype=torch.long,
        )
        vals.append(
            score_vector.index_select(
                0,
                ids,
            ).max()
        )

    return torch.stack(vals)


# -----------------------------------------------------------------------------
# Batch construction
# -----------------------------------------------------------------------------

def build_question(rec, args):
    return args.prompt_template.format(
        subject=rec["subject"],
        reference=rec["reference"],
    )


def move_batch(batch, device):
    return {
        k: (
            v.to(device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in batch.items()
    }


def build_noimage_batch(
    processor,
    rec,
    args,
):
    question = build_question(rec, args)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                }
            ],
        }
    ]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = question

    last_error = None

    for fn in [
        lambda: processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            return_tensors="pt",
        ),
    ]:
        try:
            batch = fn()
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"No-image processor failed: {last_error}"
        )

    return move_batch(
        batch,
        args.device,
    )


def batch_spans(
    processor,
    batch,
    rec,
):
    ids = (
        batch["input_ids"][0]
        .detach()
        .cpu()
        .tolist()
    )

    subject_span, reference_span = (
        locate_pair_spans(
            processor.tokenizer,
            ids,
            rec["subject"],
            rec["reference"],
        )
    )

    if (
        subject_span is None
        or reference_span is None
    ):
        raise RuntimeError(
            "Could not locate subject/reference tokens: "
            f"{rec['subject']!r} / {rec['reference']!r}"
        )

    return subject_span, reference_span


# -----------------------------------------------------------------------------
# Middle-state capture
# -----------------------------------------------------------------------------

class CaptureMiddle:
    def __init__(
        self,
        layers,
        selected_layers,
        detach,
    ):
        self.handles = []
        self.states = {}
        self.detach = bool(detach)

        for layer in selected_layers:
            self.handles.append(
                layers[layer].register_forward_hook(
                    self._make_hook(layer)
                )
            )

    def _make_hook(self, layer):
        def hook(_module, _inputs, output):
            hidden, _ = causal.extract_hidden(output)

            if hidden.ndim != 3:
                return output

            self.states[layer] = (
                hidden.detach()
                if self.detach
                else hidden
            )

            return output

        return hook

    def validate(self, selected_layers):
        missing = [
            l for l in selected_layers
            if l not in self.states
        ]
        if missing:
            raise RuntimeError(
                f"Missing captures: {missing}"
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def pair_state(
    hidden,
    subject_span,
    reference_span,
    pool,
):
    h = hidden[0]

    hs = pool_hidden(
        h,
        subject_span,
        pool,
    )
    hr = pool_hidden(
        h,
        reference_span,
        pool,
    )

    return hs - hr


def pair_gradient(
    grad_hidden,
    subject_span,
    reference_span,
    pool,
):
    g = grad_hidden[0]

    gs = pool_hidden(
        g,
        subject_span,
        pool,
    )
    gr = pool_hidden(
        g,
        reference_span,
        pool,
    )

    # Symmetric pair-coordinate perturbation:
    # h_sub += eps/2 q; h_ref -= eps/2 q
    return 0.5 * (gs - gr)


# -----------------------------------------------------------------------------
# Control branch
# -----------------------------------------------------------------------------

def capture_control_pairs(
    model,
    processor,
    layers,
    guide_layers,
    rec,
    real_image,
    args,
):
    control_image = None

    try:
        if args.control == "gray":
            control_image = causal.make_gray_image(
                real_image,
                args.gray_value,
            )
            batch = causal.build_batch(
                processor,
                control_image,
                rec,
                args,
            )

        elif args.control == "noimage":
            batch = build_noimage_batch(
                processor,
                rec,
                args,
            )

        else:
            raise ValueError(args.control)

        subject_span, reference_span = (
            batch_spans(
                processor,
                batch,
                rec,
            )
        )

        with CaptureMiddle(
            layers,
            guide_layers,
            detach=True,
        ) as cap:
            with torch.inference_mode():
                outputs = model(
                    **batch,
                    use_cache=False,
                    return_dict=True,
                )

            cap.validate(guide_layers)

        result = {}

        for layer in guide_layers:
            result[layer] = (
                pair_state(
                    cap.states[layer],
                    subject_span,
                    reference_span,
                    args.pool,
                )
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        del outputs
        del batch

        return result

    finally:
        if control_image is not None:
            control_image.close()


# -----------------------------------------------------------------------------
# Gradient selector
# -----------------------------------------------------------------------------

def gradient_selector_one(
    model,
    processor,
    layers,
    guide_layers,
    rec,
    image,
    control_pairs,
    token_map,
    args,
):
    batch = causal.build_batch(
        processor,
        image,
        rec,
        args,
    )

    subject_span, reference_span = (
        batch_spans(
            processor,
            batch,
            rec,
        )
    )

    with torch.enable_grad():
        with CaptureMiddle(
            layers,
            guide_layers,
            detach=False,
        ) as cap:
            outputs = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

            cap.validate(guide_layers)

            logits = outputs.logits[
                0,
                -1,
                :
            ]

            rel_scores = relation_scores(
                logits,
                token_map,
            )

            real_pairs = {
                layer: pair_state(
                    cap.states[layer],
                    subject_span,
                    reference_span,
                    args.pool,
                )
                for layer in guide_layers
            }

            q_by_layer = {
                layer: (
                    real_pairs[layer]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    - control_pairs[layer]
                )
                for layer in guide_layers
            }

            captured_tensors = [
                cap.states[layer]
                for layer in guide_layers
            ]

            raw_effects = {
                relation: {}
                for relation in RELS
            }
            cos_effects = {
                relation: {}
                for relation in RELS
            }
            grad_norms = {
                relation: {}
                for relation in RELS
            }

            for ri, relation in enumerate(RELS):
                grads = torch.autograd.grad(
                    rel_scores[ri],
                    captured_tensors,
                    retain_graph=(
                        ri < len(RELS) - 1
                    ),
                    create_graph=False,
                    allow_unused=True,
                )

                for layer, grad_hidden in zip(
                    guide_layers,
                    grads,
                ):
                    if grad_hidden is None:
                        raw_effects[relation][layer] = float("nan")
                        cos_effects[relation][layer] = float("nan")
                        grad_norms[relation][layer] = float("nan")
                        continue

                    g_pair = pair_gradient(
                        grad_hidden,
                        subject_span,
                        reference_span,
                        args.pool,
                    ).float()

                    q = torch.from_numpy(
                        q_by_layer[layer]
                    ).to(
                        device=g_pair.device,
                        dtype=torch.float32,
                    )

                    dot = torch.dot(
                        q,
                        g_pair,
                    )

                    qn = torch.linalg.vector_norm(q)
                    gn = torch.linalg.vector_norm(g_pair)

                    raw_effects[
                        relation
                    ][layer] = float(
                        dot.detach().cpu().item()
                    )

                    cos_effects[
                        relation
                    ][layer] = float(
                        (
                            dot
                            / (qn * gn + EPS)
                        )
                        .detach()
                        .cpu()
                        .item()
                    )

                    grad_norms[
                        relation
                    ][layer] = float(
                        gn.detach().cpu().item()
                    )

                del grads

            relation_logits = {
                relation: float(
                    rel_scores[i]
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                for i, relation in enumerate(RELS)
            }

    grad_dot_raw = {
        relation: safe_mean(
            raw_effects[relation][layer]
            for layer in guide_layers
        )
        for relation in RELS
    }

    grad_cos = {
        relation: safe_mean(
            cos_effects[relation][layer]
            for layer in guide_layers
        )
        for relation in RELS
    }

    opposite_margin = {
        relation: (
            grad_dot_raw[relation]
            - grad_dot_raw[
                OPPOSITE[relation]
            ]
        )
        for relation in RELS
    }

    score_modes = {
        "grad_dot_raw": grad_dot_raw,
        "grad_cos": grad_cos,
        "opposite_margin": opposite_margin,
    }

    predictions = {}

    for mode, scores in score_modes.items():
        ordered = sorted(
            scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        predictions[mode] = {
            "pred": ordered[0][0],
            "margin": float(
                ordered[0][1]
                - ordered[1][1]
            ),
            "scores": scores,
        }

    diagnostics = {
        "relation_logits": relation_logits,
        "raw_effects": raw_effects,
        "cos_effects": cos_effects,
        "grad_norms": grad_norms,
        "q_norms": {
            layer: float(
                np.linalg.norm(
                    q_by_layer[layer]
                )
            )
            for layer in guide_layers
        },
    }

    del outputs
    del batch

    return predictions, diagnostics


# -----------------------------------------------------------------------------
# Actuator cache / template
# -----------------------------------------------------------------------------

def load_actuator_cache_subset(
    path,
    actuator_layers,
    train_sids,
):
    path = Path(path)

    with np.load(
        path,
        allow_pickle=True,
    ) as z:
        sids = np.asarray(
            z["sample_index"],
            dtype=np.int64,
        )
        layers = np.asarray(
            z["layers"],
            dtype=np.int64,
        ).tolist()
        deltas = np.asarray(
            z["deltas"],
            dtype=np.float32,
        )

    layer_to_col = {
        int(layer): i
        for i, layer in enumerate(layers)
    }

    missing_layers = [
        layer
        for layer in actuator_layers
        if layer not in layer_to_col
    ]
    if missing_layers:
        raise RuntimeError(
            f"Cache lacks actuator layers={missing_layers}; "
            f"available={layers}"
        )

    sid_to_row = {
        int(sid): i
        for i, sid in enumerate(sids.tolist())
    }

    missing_sids = [
        int(sid)
        for sid in train_sids
        if int(sid) not in sid_to_row
    ]
    if missing_sids:
        raise RuntimeError(
            f"Cache lacks {len(missing_sids)} TRAIN samples. "
            f"First missing={missing_sids[:10]}"
        )

    cols = [
        layer_to_col[layer]
        for layer in actuator_layers
    ]

    result = {
        int(sid): deltas[
            sid_to_row[int(sid)]
        ][cols].astype(np.float32)
        for sid in train_sids
    }

    print(
        f"[reuse actuator cache] {path} | "
        f"cached_layers={layers} -> actuator_layers={actuator_layers}"
    )

    return result


def get_actuator_templates(
    model,
    processor,
    layers,
    records,
    train_sids,
    train_generation,
    actuator_layers,
    args,
    outdir,
):
    if args.actuator_cache:
        delta_cache = load_actuator_cache_subset(
            args.actuator_cache,
            actuator_layers,
            train_sids,
        )
    else:
        train_records = {
            sid: records[sid]
            for sid in train_sids
        }

        delta_cache = causal.build_or_load_delta_cache(
            model=model,
            processor=processor,
            layers=layers,
            records=train_records,
            selected_layers=actuator_layers,
            args=args,
            cache_path=(
                outdir
                / "train_real_gray_actuator_deltas.npz"
            ),
        )

    layer_index = {
        layer: i
        for i, layer in enumerate(actuator_layers)
    }

    templates, filter_used = causal.fit_templates(
        delta_cache=delta_cache,
        layer_index=layer_index,
        records=records,
        sids=train_sids,
        selected_layers=actuator_layers,
        train_generation=train_generation,
        requested_filter=args.template_filter,
    )

    return templates, filter_used


# -----------------------------------------------------------------------------
# Metrics / steering
# -----------------------------------------------------------------------------

def baseline_info(
    existing_test,
    records,
    sid,
):
    pred = existing_test[sid].get("pred")

    correct = int(
        pred in RELSET
        and pred == records[sid]["gt"]
    )

    return pred, correct


def selector_summary(rows, mode):
    pred_key = f"{mode}_pred"

    wrong = [
        row
        for row in rows
        if int(row["baseline_correct"]) == 0
    ]
    correct = [
        row
        for row in rows
        if int(row["baseline_correct"]) == 1
    ]

    n = len(rows)

    return {
        "selector_mode": mode,
        "N": n,
        "selector_acc": (
            sum(
                row[pred_key] == row["gt"]
                for row in rows
            ) / n
            if n else float("nan")
        ),
        "baseline_wrong_N": len(wrong),
        "selector_acc_on_baseline_wrong": (
            sum(
                row[pred_key] == row["gt"]
                for row in wrong
            ) / len(wrong)
            if wrong else float("nan")
        ),
        "baseline_correct_N": len(correct),
        "selector_acc_on_baseline_correct": (
            sum(
                row[pred_key] == row["gt"]
                for row in correct
            ) / len(correct)
            if correct else float("nan")
        ),
    }


def run_late_steering(
    model,
    processor,
    layers,
    actuator_templates,
    actuator_layers,
    records,
    test_sids,
    selector_rows,
    existing_test,
    args,
):
    by_sid = {
        int(row["sid"]): row
        for row in selector_rows
    }

    rows = []

    for sid in tqdm(
        test_sids,
        desc="Gradient-routed late steering",
    ):
        rec = records[sid]
        selector_pred = by_sid[
            sid
        ][
            f"{args.selector_score}_pred"
        ]

        baseline_pred, baseline_correct = (
            baseline_info(
                existing_test,
                records,
                sid,
            )
        )

        image = None

        try:
            image = Image.open(
                rec["image_path"]
            ).convert("RGB")

            batch = causal.build_batch(
                processor,
                image,
                rec,
                args,
            )

            with causal.SteerLast(
                layers=layers,
                templates=actuator_templates,
                selected=actuator_layers,
                target=selector_pred,
                scale=args.scale,
                mode=args.edit_mode,
                source=baseline_pred,
            ):
                text, pred = causal.generate(
                    model,
                    processor,
                    batch,
                    args,
                )

            edit_correct = int(
                pred == rec["gt"]
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "selector_pred": selector_pred,
                "baseline_pred": baseline_pred or "",
                "edit_pred": pred or "",
                "baseline_correct": baseline_correct,
                "edit_correct": edit_correct,
                "W2C": int(
                    baseline_correct == 0
                    and edit_correct == 1
                ),
                "C2W": int(
                    baseline_correct == 1
                    and edit_correct == 0
                ),
                "changed": int(
                    (baseline_pred or "")
                    != (pred or "")
                ),
                "text": text,
            })

            del batch

        finally:
            if image is not None:
                image.close()

            cleanup()

    n = len(rows)
    base_acc = safe_mean(
        row["baseline_correct"]
        for row in rows
    )
    edit_acc = safe_mean(
        row["edit_correct"]
        for row in rows
    )
    w2c = sum(
        int(row["W2C"])
        for row in rows
    )
    c2w = sum(
        int(row["C2W"])
        for row in rows
    )

    return {
        "N": n,
        "selector_score": args.selector_score,
        "control": args.control,
        "baseline_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "W2C": w2c,
        "C2W": c2w,
        "net": w2c - c2w,
        "changed": sum(
            int(row["changed"])
            for row in rows
        ),
    }, rows


def main():
    args = parse_args()

    outdir = Path(args.output_dir)

    if (
        args.overwrite
        and outdir.exists()
    ):
        shutil.rmtree(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_test, baseline_path = (
        causal.load_existing_test_baseline(
            args.prior_output_dir
        )
    )
    if existing_test is None:
        raise RuntimeError(
            "Could not reuse TEST baseline."
        )

    train_generation, train_generation_path = (
        causal.load_existing_train_generation(
            args.prior_output_dir
        )
    )

    records = causal.load_all_records(args)

    train_sids, test_sids = (
        causal.derive_train_test_ids(
            records,
            existing_test,
        )
    )

    if args.max_test_samples > 0:
        test_sids = test_sids[
            : args.max_test_samples
        ]

    guide_layers = resolve_guide_layers(
        args.model_id,
        args.guide_layers,
    )

    actuator_layers = list(
        causal.model_preset(
            args.model_id
        )["actuator_layers"]
    )

    model, processor = causal.load_model(args)
    layers, layer_path = causal.resolve_layers(model)

    bad = [
        l
        for l in (
            guide_layers
            + actuator_layers
        )
        if l < 0 or l >= len(layers)
    ]
    if bad:
        raise RuntimeError(
            f"Invalid layers={bad}; decoder has {len(layers)} blocks."
        )

    token_map = relation_token_variants(
        processor.tokenizer
    )

    print("\n" + "=" * 150)
    print(
        "MIDDLE GRADIENT SELECTOR -> LATE CAUSAL ACTUATOR"
    )
    print("=" * 150)
    print(f"model={args.model_id}")
    print(
        f"decoder={layer_path} | blocks={len(layers)}"
    )
    print(
        f"TRAIN={len(train_sids)} TEST={len(test_sids)}"
    )
    print(f"guide_layers={guide_layers}")
    print(f"control={args.control}")
    print(f"selector_score={args.selector_score}")
    print(f"actuator_layers={actuator_layers}")
    print(f"baseline_source={baseline_path}")
    print(
        f"train_generation_source={train_generation_path}"
    )
    print("=" * 150)

    selector_rows = []

    for sid in tqdm(
        test_sids,
        desc=f"Gradient selector [{args.control}]",
    ):
        rec = records[sid]
        image = None

        try:
            image = Image.open(
                rec["image_path"]
            ).convert("RGB")

            control_pairs = capture_control_pairs(
                model=model,
                processor=processor,
                layers=layers,
                guide_layers=guide_layers,
                rec=rec,
                real_image=image,
                args=args,
            )

            predictions, diagnostics = (
                gradient_selector_one(
                    model=model,
                    processor=processor,
                    layers=layers,
                    guide_layers=guide_layers,
                    rec=rec,
                    image=image,
                    control_pairs=control_pairs,
                    token_map=token_map,
                    args=args,
                )
            )

            baseline_pred, baseline_correct = (
                baseline_info(
                    existing_test,
                    records,
                    sid,
                )
            )

            row = {
                "sid": sid,
                "gt": rec["gt"],
                "baseline_pred": baseline_pred or "",
                "baseline_correct": baseline_correct,
            }

            for mode, info in predictions.items():
                row[f"{mode}_pred"] = info["pred"]
                row[f"{mode}_margin"] = info["margin"]
                row[f"{mode}_correct"] = int(
                    info["pred"] == rec["gt"]
                )

                for relation in RELS:
                    row[
                        f"{mode}_score_{relation}"
                    ] = info["scores"][relation]

            for relation in RELS:
                row[
                    f"real_logit_{relation}"
                ] = diagnostics[
                    "relation_logits"
                ][relation]

                for layer in guide_layers:
                    row[
                        f"L{layer}_{relation}_dot"
                    ] = diagnostics[
                        "raw_effects"
                    ][relation][layer]

                    row[
                        f"L{layer}_{relation}_gradcos"
                    ] = diagnostics[
                        "cos_effects"
                    ][relation][layer]

                    row[
                        f"L{layer}_{relation}_gradnorm"
                    ] = diagnostics[
                        "grad_norms"
                    ][relation][layer]

            for layer in guide_layers:
                row[
                    f"L{layer}_qnorm"
                ] = diagnostics[
                    "q_norms"
                ][layer]

            selector_rows.append(row)

        finally:
            if image is not None:
                image.close()

            cleanup()

    write_csv(
        outdir
        / "test_gradient_selector_predictions.csv",
        selector_rows,
    )

    modes = [
        "grad_dot_raw",
        "grad_cos",
        "opposite_margin",
    ]

    summaries = [
        selector_summary(
            selector_rows,
            mode,
        )
        for mode in modes
    ]

    write_csv(
        outdir
        / "test_gradient_selector_summary.csv",
        summaries,
    )

    baseline_acc = safe_mean(
        row["baseline_correct"]
        for row in selector_rows
    )

    print("\nGRADIENT SELECTOR RESULTS")
    print("-" * 150)
    print(
        f"baseline generation acc = {baseline_acc:.4f}"
    )

    for summary in summaries:
        print(
            f"{summary['selector_mode']:20s} | "
            f"selector_acc={summary['selector_acc']:.4f} | "
            f"wrong_acc="
            f"{summary['selector_acc_on_baseline_wrong']:.4f} | "
            f"correct_acc="
            f"{summary['selector_acc_on_baseline_correct']:.4f}"
        )

    if args.skip_steering:
        print(
            "\n[skip] late steering disabled."
        )
        return

    actuator_templates, filter_used = (
        get_actuator_templates(
            model=model,
            processor=processor,
            layers=layers,
            records=records,
            train_sids=train_sids,
            train_generation=train_generation,
            actuator_layers=actuator_layers,
            args=args,
            outdir=outdir,
        )
    )

    steering_summary, steering_rows = (
        run_late_steering(
            model=model,
            processor=processor,
            layers=layers,
            actuator_templates=actuator_templates,
            actuator_layers=actuator_layers,
            records=records,
            test_sids=test_sids,
            selector_rows=selector_rows,
            existing_test=existing_test,
            args=args,
        )
    )

    steering_summary[
        "actuator_template_filter"
    ] = filter_used
    steering_summary[
        "guide_layers"
    ] = ",".join(map(str, guide_layers))
    steering_summary[
        "actuator_layers"
    ] = ",".join(map(str, actuator_layers))

    write_csv(
        outdir
        / "test_gradient_routed_steering_summary.csv",
        [steering_summary],
    )

    write_csv(
        outdir
        / "test_gradient_routed_steering_details.csv",
        steering_rows,
    )

    print("\n" + "=" * 150)
    print(
        "ACTUAL model.generate() — GRADIENT-ROUTED LATE STEERING"
    )
    print("=" * 150)
    print(
        f"selector={args.selector_score} | "
        f"control={args.control} | "
        f"guide={guide_layers} | "
        f"actuator={actuator_layers}"
    )
    print(
        f"acc "
        f"{steering_summary['baseline_acc']:.4f}"
        f"->{steering_summary['edit_acc']:.4f} "
        f"{steering_summary['gain']:+.4f} | "
        f"W2C={steering_summary['W2C']} "
        f"C2W={steering_summary['C2W']} "
        f"net={steering_summary['net']:+d} | "
        f"changed={steering_summary['changed']}"
    )
    print("=" * 150)


if __name__ == "__main__":
    main()
