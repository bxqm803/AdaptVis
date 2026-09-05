#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_middle_patch_decoder_selector_qwen25_v1.py

Model-native middle-state patch decoder -> ONE late causal actuator.

Idea
----
Keep the current read/write framework:

    middle spatial evidence -> choose ONE relation -> late causal vector s_r

but replace the hand-designed cosine readout with the model's OWN downstream
decoder.

For each TEST sample:

(1) REAL branch
    Run the normal image prompt and capture subject/reference hidden states at
    the chosen middle guide layer(s).

(2) NO-IMAGE branch
    Run the same textual question without image input.

(3) PATCH
    At the same middle guide layer(s), patch the REAL subject/reference states
    into the NO-IMAGE branch.

    Default --patch-mode replace:
        h_noimg_sub,l <- h_real_sub,l
        h_noimg_ref,l <- h_real_ref,l

    Optional --patch-mode pair_delta:
        preserve the no-image midpoint but replace only the pair-difference:

        pair_real = mean(h_real_sub) - mean(h_real_ref)
        pair_no   = mean(h_no_sub)   - mean(h_no_ref)
        q         = pair_real - pair_no

        h_no_sub <- h_no_sub + 0.5 q
        h_no_ref <- h_no_ref - 0.5 q

(4) MODEL-NATIVE READOUT
    Let the frozen model continue through all remaining layers.
    Read the four final next-token logits:
        left / right / above / below

Two selector scores are reported:

    patched_logits:
        score_r = z_r(patched no-image)

    patch_delta [DEFAULT]:
        score_r = z_r(patched no-image) - z_r(no-image)

The second one isolates what the patched middle state itself contributes,
instead of inheriting the no-image language prior.

Prediction:
    r_hat = argmax_r score_r

No classifier is trained.
No cosine prototype is used.
No TEST GT is used for routing.
Exactly ONE late actuator is selected per sample.

Then actual generation uses the existing TRAIN-derived Real-Gray late causal
directions, e.g. Qwen2.5-VL-3B L32-L35.

Example
-------
CUDA_VISIBLE_DEVICES=0 python eval_middle_patch_decoder_selector_qwen25_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --prior-output-dir output/qwen3b_last_causal_auto_v1 \
  --guide-layers 19 \
  --patch-mode replace \
  --selector-score patch_delta \
  --actuator-cache output/qwen3b_real_gray_increment_similarity_v1/real_gray_last_deltas.npz \
  --scale 1.0 \
  --output-dir output/qwen3b_middle_patch_decoder_v1 \
  --overwrite

Selector-only first:
    add --skip-steering

Useful second diagnostic:
    --patch-mode pair_delta
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import math
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import eval_cosine_confidence_selector_qwen25_v1 as base


RELS = ("left", "right", "above", "below")
RELSET = set(RELS)


# =============================================================================
# CLI / basic utilities
# =============================================================================

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
        "--guide-layers",
        default="auto",
        help="auto: Qwen3B=L19, Qwen7B=L14-20.",
    )

    p.add_argument(
        "--patch-mode",
        default="replace",
        choices=["replace", "pair_delta"],
        help=(
            "replace: patch exact REAL subject/reference states into NoImage. "
            "pair_delta: patch only the REAL-vs-NoImage pair difference."
        ),
    )

    p.add_argument(
        "--selector-score",
        default="patch_delta",
        choices=["patch_delta", "patched_logits", "patch_margin_gain"],
    )

    p.add_argument(
        "--template-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
        help="Used only for the existing late Real-Gray actuator.",
    )

    p.add_argument(
        "--actuator-cache",
        default="",
        help="Existing Real-Gray last-token cache; layer superset is accepted.",
    )

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument(
        "--edit-mode",
        default="add",
        choices=["add", "contrast"],
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Debug only; 0 means full TEST.",
    )

    p.add_argument(
        "--skip-steering",
        action="store_true",
        help="Only evaluate the selector, without final late steering.",
    )

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
        return base.parse_layer_spec(spec)

    low = model_id.lower()
    if "3b" in low:
        return [19]
    if "7b" in low:
        return list(range(14, 21))

    raise ValueError(
        "--guide-layers auto supports Qwen2.5-VL-3B/7B only; "
        "otherwise pass explicit layers."
    )


# =============================================================================
# Subject/reference token span utilities
# =============================================================================

def find_all_subsequences(sequence, pattern):
    if not pattern:
        return []

    starts = []
    n = len(pattern)

    for i in range(len(sequence) - n + 1):
        if list(sequence[i:i+n]) == list(pattern):
            starts.append(i)

    return starts


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

            # Prefer the expected prompt ordering:
            # subject appears before reference.
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
                best = (
                    score,
                    s_span,
                    r_span,
                )

    if best is None:
        return None, None

    return best[1], best[2]


def batch_pair_spans(
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

    s_span, r_span = locate_pair_spans(
        processor.tokenizer,
        ids,
        rec["subject"],
        rec["reference"],
    )

    if s_span is None or r_span is None:
        raise RuntimeError(
            "Could not locate subject/reference token spans: "
            f"subject={rec['subject']!r}, "
            f"reference={rec['reference']!r}"
        )

    return s_span, r_span


def valid_positions(positions: Sequence[int], seq_len: int):
    return [
        int(x)
        for x in positions
        if 0 <= int(x) < seq_len
    ]


def mean_span(hidden_2d, positions):
    pos = valid_positions(
        positions,
        int(hidden_2d.shape[0]),
    )

    if not pos:
        raise RuntimeError("No valid token positions.")

    idx = torch.as_tensor(
        pos,
        device=hidden_2d.device,
        dtype=torch.long,
    )

    return hidden_2d.index_select(
        0,
        idx,
    ).mean(dim=0)


# =============================================================================
# Relation logits
# =============================================================================

def relation_token_variants(tokenizer):
    """
    Use all one-token spellings and take max logit per relation.
    """
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
            toks = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            if len(toks) != 1:
                continue

            tid = int(toks[0])

            if unk is not None and tid == int(unk):
                continue

            ids.append(tid)

        ids = list(dict.fromkeys(ids))

        if not ids:
            raise RuntimeError(
                f"No single-token spelling found for relation={relation}."
            )

        out[relation] = ids

    return out


def four_relation_logits(logits_last, token_map):
    scores = {}

    for relation in RELS:
        ids = torch.as_tensor(
            token_map[relation],
            device=logits_last.device,
            dtype=torch.long,
        )

        scores[relation] = float(
            logits_last.index_select(
                0,
                ids,
            ).max().float().cpu().item()
        )

    return scores


def relation_margins(scores):
    return {
        relation: (
            float(scores[relation])
            - max(
                float(scores[other])
                for other in RELS
                if other != relation
            )
        )
        for relation in RELS
    }


def winner(scores):
    ordered = sorted(
        scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    return (
        ordered[0][0],
        float(ordered[0][1] - ordered[1][1]),
    )


# =============================================================================
# NoImage batch
# =============================================================================

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
    question = build_question(
        rec,
        args,
    )

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
            f"NoImage processor failed: {last_error}"
        )

    return move_batch(
        batch,
        args.device,
    )


# =============================================================================
# REAL state capture
# =============================================================================

class CaptureObjectStates:
    """
    Capture subject/reference token states at selected decoder-layer OUTPUTS.
    """

    def __init__(
        self,
        layers,
        selected_layers,
        subject_span,
        reference_span,
    ):
        self.handles = []
        self.states = {}

        self.subject_span = subject_span
        self.reference_span = reference_span

        for layer in selected_layers:
            self.handles.append(
                layers[layer].register_forward_hook(
                    self._hook(layer)
                )
            )

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            hidden, _descriptor = base.extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            h2 = hidden[0]

            s_pos = valid_positions(
                self.subject_span,
                int(h2.shape[0]),
            )
            r_pos = valid_positions(
                self.reference_span,
                int(h2.shape[0]),
            )

            if not s_pos or not r_pos:
                return output

            s_idx = torch.as_tensor(
                s_pos,
                device=h2.device,
                dtype=torch.long,
            )
            r_idx = torch.as_tensor(
                r_pos,
                device=h2.device,
                dtype=torch.long,
            )

            self.states[layer] = {
                "subject_tokens": (
                    h2.index_select(0, s_idx)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ),
                "reference_tokens": (
                    h2.index_select(0, r_idx)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ),
            }

            return output

        return hook

    def validate(self, selected_layers):
        missing = [
            layer
            for layer in selected_layers
            if layer not in self.states
        ]

        if missing:
            raise RuntimeError(
                f"Missing REAL middle states at layers={missing}"
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def capture_real_branch(
    model,
    processor,
    layers,
    guide_layers,
    rec,
    image,
    token_map,
    args,
):
    batch = base.build_batch(
        processor,
        image,
        rec,
        args,
    )

    s_span, r_span = batch_pair_spans(
        processor,
        batch,
        rec,
    )

    with CaptureObjectStates(
        layers,
        guide_layers,
        s_span,
        r_span,
    ) as capture:
        with torch.inference_mode():
            outputs = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

        capture.validate(
            guide_layers
        )

    real_logits = four_relation_logits(
        outputs.logits[0, -1, :],
        token_map,
    )

    states = dict(
        capture.states
    )

    del outputs
    del batch

    return states, real_logits


# =============================================================================
# NoImage patch hook
# =============================================================================

class PatchObjectStates:
    """
    Patch REAL middle subject/reference information into a NoImage branch.

    replace:
        exact token-state replacement when span lengths match;
        if they do not, use the REAL pooled mean for every target token.

    pair_delta:
        only replace the subject-reference pair difference while preserving
        the current NoImage pair midpoint.
    """

    def __init__(
        self,
        layers,
        selected_layers,
        source_states,
        subject_span,
        reference_span,
        patch_mode,
    ):
        self.handles = []
        self.source_states = source_states
        self.subject_span = subject_span
        self.reference_span = reference_span
        self.patch_mode = patch_mode

        self.done = {
            layer: False
            for layer in selected_layers
        }

        for layer in selected_layers:
            self.handles.append(
                layers[layer].register_forward_hook(
                    self._hook(layer)
                )
            )

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            if self.done[layer]:
                return output

            hidden, descriptor = base.extract_hidden(
                output
            )

            if hidden.ndim != 3:
                return output

            h2 = hidden[0]

            s_pos = valid_positions(
                self.subject_span,
                int(h2.shape[0]),
            )
            r_pos = valid_positions(
                self.reference_span,
                int(h2.shape[0]),
            )

            if not s_pos or not r_pos:
                return output

            edited = hidden.clone()
            e2 = edited[0]

            src_s_np = self.source_states[
                layer
            ]["subject_tokens"]

            src_r_np = self.source_states[
                layer
            ]["reference_tokens"]

            src_s = torch.as_tensor(
                src_s_np,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            src_r = torch.as_tensor(
                src_r_np,
                device=hidden.device,
                dtype=hidden.dtype,
            )

            s_idx = torch.as_tensor(
                s_pos,
                device=hidden.device,
                dtype=torch.long,
            )
            r_idx = torch.as_tensor(
                r_pos,
                device=hidden.device,
                dtype=torch.long,
            )

            if self.patch_mode == "replace":
                if src_s.shape[0] == len(s_pos):
                    e2.index_copy_(
                        0,
                        s_idx,
                        src_s,
                    )
                else:
                    pooled_s = src_s.mean(
                        dim=0,
                        keepdim=True,
                    )
                    e2.index_copy_(
                        0,
                        s_idx,
                        pooled_s.expand(
                            len(s_pos),
                            -1,
                        ),
                    )

                if src_r.shape[0] == len(r_pos):
                    e2.index_copy_(
                        0,
                        r_idx,
                        src_r,
                    )
                else:
                    pooled_r = src_r.mean(
                        dim=0,
                        keepdim=True,
                    )
                    e2.index_copy_(
                        0,
                        r_idx,
                        pooled_r.expand(
                            len(r_pos),
                            -1,
                        ),
                    )

            elif self.patch_mode == "pair_delta":
                cur_s = e2.index_select(
                    0,
                    s_idx,
                ).mean(dim=0)

                cur_r = e2.index_select(
                    0,
                    r_idx,
                ).mean(dim=0)

                real_pair = (
                    src_s.mean(dim=0)
                    - src_r.mean(dim=0)
                )

                no_pair = (
                    cur_s
                    - cur_r
                )

                q = real_pair - no_pair

                new_s = e2.index_select(
                    0,
                    s_idx,
                ) + 0.5 * q[None, :]

                new_r = e2.index_select(
                    0,
                    r_idx,
                ) - 0.5 * q[None, :]

                e2.index_copy_(
                    0,
                    s_idx,
                    new_s,
                )

                e2.index_copy_(
                    0,
                    r_idx,
                    new_r,
                )

            else:
                raise ValueError(
                    self.patch_mode
                )

            self.done[layer] = True

            return base.replace_hidden(
                output,
                descriptor,
                edited,
            )

        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_noimage_logits(
    model,
    batch,
    token_map,
):
    with torch.inference_mode():
        outputs = model(
            **batch,
            use_cache=False,
            return_dict=True,
        )

    result = four_relation_logits(
        outputs.logits[0, -1, :],
        token_map,
    )

    del outputs

    return result


def model_native_selector_one(
    model,
    processor,
    layers,
    guide_layers,
    rec,
    image,
    token_map,
    args,
):
    # ------------------------------------------------------------
    # 1) REAL branch: source middle states.
    # ------------------------------------------------------------
    real_states, real_logits = capture_real_branch(
        model=model,
        processor=processor,
        layers=layers,
        guide_layers=guide_layers,
        rec=rec,
        image=image,
        token_map=token_map,
        args=args,
    )

    # ------------------------------------------------------------
    # 2) NoImage baseline branch.
    # ------------------------------------------------------------
    no_batch = build_noimage_batch(
        processor,
        rec,
        args,
    )

    no_s_span, no_r_span = batch_pair_spans(
        processor,
        no_batch,
        rec,
    )

    no_logits = run_noimage_logits(
        model,
        no_batch,
        token_map,
    )

    # ------------------------------------------------------------
    # 3) Patched NoImage branch.
    # ------------------------------------------------------------
    with PatchObjectStates(
        layers=layers,
        selected_layers=guide_layers,
        source_states=real_states,
        subject_span=no_s_span,
        reference_span=no_r_span,
        patch_mode=args.patch_mode,
    ):
        patched_logits = run_noimage_logits(
            model,
            no_batch,
            token_map,
        )

    # ------------------------------------------------------------
    # 4) Readout scores.
    # ------------------------------------------------------------
    patch_delta = {
        relation: (
            patched_logits[relation]
            - no_logits[relation]
        )
        for relation in RELS
    }

    no_margin = relation_margins(
        no_logits
    )
    patched_margin = relation_margins(
        patched_logits
    )

    patch_margin_gain = {
        relation: (
            patched_margin[relation]
            - no_margin[relation]
        )
        for relation in RELS
    }

    score_modes = {
        "patch_delta": patch_delta,
        "patched_logits": patched_logits,
        "patch_margin_gain": patch_margin_gain,
        "real_logits": real_logits,
    }

    predictions = {}

    for mode, scores in score_modes.items():
        pred, margin = winner(scores)

        predictions[mode] = {
            "pred": pred,
            "margin": margin,
            "scores": scores,
        }

    diagnostics = {
        "real_logits": real_logits,
        "noimage_logits": no_logits,
        "patched_logits": patched_logits,
        "patch_delta": patch_delta,
        "patch_margin_gain": patch_margin_gain,
    }

    del no_batch

    return predictions, diagnostics


# =============================================================================
# Existing late actuator
# =============================================================================

def load_actuator_cache_subset(
    path,
    actuator_layers,
    train_sids,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

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
            f"Cache lacks {len(missing_sids)} TRAIN samples; "
            f"first={missing_sids[:10]}"
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

        delta_cache = base.build_or_load_delta_cache(
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

    templates, filter_used = base.fit_templates(
        delta_cache=delta_cache,
        layer_index=layer_index,
        records=records,
        sids=train_sids,
        selected_layers=actuator_layers,
        train_generation=train_generation,
        requested_filter=args.template_filter,
    )

    return templates, filter_used


# =============================================================================
# Metrics
# =============================================================================

def baseline_info(
    existing_test,
    records,
    sid,
):
    pred = existing_test[sid].get(
        "pred"
    )

    correct = int(
        pred in RELSET
        and pred == records[sid]["gt"]
    )

    return pred, correct


def selector_summary(
    rows,
    mode,
):
    pred_key = f"{mode}_pred"

    n = len(rows)

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

    return {
        "selector_mode": mode,
        "N": n,
        "selector_acc": (
            sum(
                row[pred_key] == row["gt"]
                for row in rows
            ) / n
            if n
            else float("nan")
        ),
        "baseline_wrong_N": len(wrong),
        "selector_acc_on_baseline_wrong": (
            sum(
                row[pred_key] == row["gt"]
                for row in wrong
            ) / len(wrong)
            if wrong
            else float("nan")
        ),
        "baseline_correct_N": len(correct),
        "selector_acc_on_baseline_correct": (
            sum(
                row[pred_key] == row["gt"]
                for row in correct
            ) / len(correct)
            if correct
            else float("nan")
        ),
        "potential_W2C": sum(
            int(row["baseline_correct"]) == 0
            and row[pred_key] == row["gt"]
            for row in rows
        ),
        "potential_C2W": sum(
            int(row["baseline_correct"]) == 1
            and row[pred_key] != row["gt"]
            for row in rows
        ),
    }


# =============================================================================
# Actual late steering
# =============================================================================

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
        desc=(
            f"Patch-decoder routed steering "
            f"[{args.selector_score}]"
        ),
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

            batch = base.build_batch(
                processor,
                image,
                rec,
                args,
            )

            with base.SteerLast(
                layers=layers,
                templates=actuator_templates,
                selected=actuator_layers,
                target=selector_pred,
                scale=args.scale,
                mode=args.edit_mode,
                source=baseline_pred,
            ):
                text, pred = base.generate(
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

    summary = {
        "selector_score": args.selector_score,
        "patch_mode": args.patch_mode,
        "N": n,
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
    }

    return summary, rows


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    outdir = Path(
        args.output_dir
    )

    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_test, baseline_path = (
        base.load_existing_test_baseline(
            args.prior_output_dir
        )
    )

    if existing_test is None:
        raise RuntimeError(
            "Could not reuse TEST baseline from --prior-output-dir."
        )

    train_generation, train_generation_path = (
        base.load_existing_train_generation(
            args.prior_output_dir
        )
    )

    records = base.load_all_records(
        args
    )

    train_sids, test_sids = (
        base.derive_train_test_ids(
            records,
            existing_test,
        )
    )

    if args.max_test_samples > 0:
        test_sids = test_sids[
            :args.max_test_samples
        ]

    guide_layers = resolve_guide_layers(
        args.model_id,
        args.guide_layers,
    )

    actuator_layers = list(
        base.model_preset(
            args.model_id
        )["actuator_layers"]
    )

    model, processor = base.load_model(
        args
    )

    layers, layer_path = base.resolve_layers(
        model
    )

    bad = [
        layer
        for layer in (
            guide_layers
            + actuator_layers
        )
        if layer < 0
        or layer >= len(layers)
    ]

    if bad:
        raise RuntimeError(
            f"Invalid layers={bad}; decoder has "
            f"blocks 0..{len(layers)-1}"
        )

    token_map = relation_token_variants(
        processor.tokenizer
    )

    print("\n" + "=" * 152)
    print(
        "MODEL-NATIVE MIDDLE PATCH DECODER -> "
        "ONE LATE RELATION-SPECIFIC CAUSAL ACTUATOR"
    )
    print("=" * 152)

    print(f"model={args.model_id}")
    print(
        f"decoder={layer_path} | blocks={len(layers)}"
    )
    print(
        f"TRAIN={len(train_sids)} TEST={len(test_sids)}"
    )
    print(f"guide_layers={guide_layers}")
    print(f"patch_mode={args.patch_mode}")
    print(f"selector_score={args.selector_score}")
    print(f"actuator_layers={actuator_layers}")
    print(f"baseline_source={baseline_path}")
    print(
        f"train_generation_source={train_generation_path}"
    )
    print("=" * 152)

    print("\nRelation token variants:")
    for relation in RELS:
        vals = [
            (
                tid,
                processor.tokenizer.decode([tid]),
            )
            for tid in token_map[relation]
        ]
        print(
            f"  {relation:>5s}: {vals}"
        )

    # ---------------------------------------------------------------------
    # TEST selector
    # ---------------------------------------------------------------------
    selector_rows = []

    for sid in tqdm(
        test_sids,
        desc=(
            f"Model-native patch selector "
            f"[{args.patch_mode}]"
        ),
    ):
        rec = records[sid]
        image = None

        try:
            image = Image.open(
                rec["image_path"]
            ).convert("RGB")

            predictions, diagnostics = (
                model_native_selector_one(
                    model=model,
                    processor=processor,
                    layers=layers,
                    guide_layers=guide_layers,
                    rec=rec,
                    image=image,
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
                row[
                    f"{mode}_pred"
                ] = info["pred"]

                row[
                    f"{mode}_margin"
                ] = info["margin"]

                row[
                    f"{mode}_correct"
                ] = int(
                    info["pred"]
                    == rec["gt"]
                )

                for relation in RELS:
                    row[
                        f"{mode}_score_{relation}"
                    ] = info[
                        "scores"
                    ][relation]

            for relation in RELS:
                row[
                    f"real_logit_{relation}"
                ] = diagnostics[
                    "real_logits"
                ][relation]

                row[
                    f"noimage_logit_{relation}"
                ] = diagnostics[
                    "noimage_logits"
                ][relation]

                row[
                    f"patched_logit_{relation}"
                ] = diagnostics[
                    "patched_logits"
                ][relation]

                row[
                    f"patch_delta_{relation}"
                ] = diagnostics[
                    "patch_delta"
                ][relation]

                row[
                    f"patch_margin_gain_{relation}"
                ] = diagnostics[
                    "patch_margin_gain"
                ][relation]

            selector_rows.append(
                row
            )

        finally:
            if image is not None:
                image.close()

            cleanup()

    write_csv(
        outdir
        / "test_patch_decoder_predictions.csv",
        selector_rows,
    )

    selector_modes = [
        "patch_delta",
        "patched_logits",
        "patch_margin_gain",
        "real_logits",
    ]

    summaries = [
        selector_summary(
            selector_rows,
            mode,
        )
        for mode in selector_modes
    ]

    write_csv(
        outdir
        / "test_patch_decoder_selector_summary.csv",
        summaries,
    )

    baseline_acc = safe_mean(
        row["baseline_correct"]
        for row in selector_rows
    )

    print(
        "\nMODEL-NATIVE PATCH SELECTOR RESULTS"
    )
    print("-" * 152)

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
            f"{summary['selector_acc_on_baseline_correct']:.4f} | "
            f"potential W2C/C2W="
            f"{summary['potential_W2C']}/"
            f"{summary['potential_C2W']}"
        )

    if args.skip_steering:
        print(
            "\n[skip] actual late steering disabled."
        )
        print(
            f"[saved] "
            f"{outdir / 'test_patch_decoder_predictions.csv'}"
        )
        print(
            f"[saved] "
            f"{outdir / 'test_patch_decoder_selector_summary.csv'}"
        )
        return

    # ---------------------------------------------------------------------
    # Existing late actuator: unchanged.
    # ---------------------------------------------------------------------
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
        "guide_layers"
    ] = ",".join(
        map(str, guide_layers)
    )

    steering_summary[
        "actuator_layers"
    ] = ",".join(
        map(str, actuator_layers)
    )

    steering_summary[
        "actuator_template_filter"
    ] = filter_used

    write_csv(
        outdir
        / "test_patch_decoder_routed_steering_summary.csv",
        [steering_summary],
    )

    write_csv(
        outdir
        / "test_patch_decoder_routed_steering_details.csv",
        steering_rows,
    )

    print("\n" + "=" * 152)
    print(
        "ACTUAL model.generate() — "
        "PATCH-DECODER ROUTED LATE STEERING"
    )
    print("=" * 152)

    print(
        f"selector={args.selector_score} | "
        f"patch_mode={args.patch_mode} | "
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

    print("=" * 152)

    print(
        f"[saved] "
        f"{outdir / 'test_patch_decoder_predictions.csv'}"
    )
    print(
        f"[saved] "
        f"{outdir / 'test_patch_decoder_selector_summary.csv'}"
    )
    print(
        f"[saved] "
        f"{outdir / 'test_patch_decoder_routed_steering_summary.csv'}"
    )


if __name__ == "__main__":
    main()
