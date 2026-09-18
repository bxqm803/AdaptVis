#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure-A style single-sample spatial steering across decoder layers.

Goal
====
Adapt Fig. 6A of "Linear Mechanisms for Spatiotemporal Reasoning in Vision
Language Models" to AdaptVis' relation -> option setting.

For one held-out COCO-two sample with a randomized relation->A/B/C/D mapping:

  1. Fit per-layer spatial relation centroids on TRAIN from the image-dependent
     object-pair residual

         q_L = (h_real_sub - h_real_ref) - (h_gray_sub - h_gray_ref).

  2. Let g be the sample's GT relation and o its semantic opposite.
     At every decoder layer L independently construct

         delta_L = mu_{g,L} - mu_{o,L}.

  3. Run three conditions at each intervention layer:

         orig        : no intervention
         toward_gt   : h_sub += .5 * alpha * delta_L
                       h_ref -= .5 * alpha * delta_L
         toward_opp  : same edit with -delta_L

     Only subject/reference residual states at ONE layer are edited.
     Prompt-last and logits are never patched directly.

  4. Measure first-step log probability of the CURRENT SAMPLE'S option letter
     mapped to g and to o.  Because relation->letter mapping is randomized per
     sample, this tests whether a semantic spatial edit controls the option that
     currently denotes that relation rather than a fixed answer symbol.

Main output
===========
  figureA_single_sample_layer_steering.png
      Left: image + query/mapping.
      Right: log P(mapped GT option) and log P(mapped opposite option)
             across intervention layers for orig / toward_gt / toward_opp.

  figureA_values.csv
  train_relation_centroids_realgray.npz
  mapping_balance.csv
  metadata.json
  errors.jsonl

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u figureA_qwen3b_single_sample_spatial_steering_layers_v1.py \
  --model qwen-3b \
  --layers all \
  --alpha 1.0 \
  --train-ratio 0.30 \
  --sample-relations left,right \
  --sample-mode opposite \
  --output-dir output/figureA_qwen3b_single_sample_spatial_steering_v1 \
  --overwrite

To reproduce a fixed illustrative sample after inspecting the first run:

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u figureA_qwen3b_single_sample_spatial_steering_layers_v1.py \
  --sid <SID_FROM_METADATA> \
  --layers all --alpha 1.0 \
  --output-dir output/figureA_qwen3b_sid<SID> \
  --overwrite

Notes
=====
- --sample-mode opposite chooses the first held-out sample whose ORIGINAL
  first-step option is the semantic opposite of GT.  If none exists it falls
  back to any wrong sample, then any sample.  For the final paper figure,
  prefer passing an explicit --sid and report that it is illustrative.
- This is a single-sample visualization.  The dataset-level causal result should
  be a separate panel/experiment.
- Run from the AdaptVis llava16 repository root next to:
    analyze_coco_centroid_generation_step1_v4.py
    eval_coco_multilayer_relation_trajectory_repair_v1.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
OPP = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-12
SCRIPT_VERSION = "figureA-qwen3b-single-sample-spatial-steering-layers-v1"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument(
        "--layers",
        default="all",
        help="all, 20-26, or comma-separated decoder block indices.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all COCO-two samples before train/test split.",
    )
    p.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="Optional cap on TRAIN examples used to fit spatial centroids. 0 = all train.",
    )
    p.add_argument(
        "--sample-relations",
        default="left,right,above,below",
        help="Relations eligible for automatic illustrative-sample selection.",
    )
    p.add_argument(
        "--sample-mode",
        default="opposite",
        choices=["opposite", "wrong", "any"],
        help=(
            "Automatic sample selection when --sid is omitted. "
            "opposite: original first-step predicts mapped semantic opposite; "
            "wrong: any wrong first-step option; any: first eligible held-out sample."
        ),
    )
    p.add_argument(
        "--sid",
        type=int,
        default=None,
        help="Explicit held-out sample id. Preferred for a fixed final figure.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_layers(spec: str, n_layers: int) -> List[int]:
    s = str(spec).strip().lower()
    if s == "all":
        return list(range(n_layers))
    vals: List[int] = []
    for part in s.split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            vals.extend(range(int(lo), int(hi) + 1))
        else:
            vals.append(int(part))
    vals = sorted(set(vals))
    for L in vals:
        if not 0 <= L < n_layers:
            raise ValueError(f"Invalid layer L{L}; decoder has {n_layers} blocks")
    if not vals:
        raise ValueError("No layers selected")
    return vals


def parse_relations(s: str) -> List[str]:
    out = []
    for x in str(s).split(","):
        r = x.strip().lower()
        if r:
            if r not in REL:
                raise ValueError(f"Unknown relation: {r}")
            if r not in out:
                out.append(r)
    if not out:
        raise ValueError("No sample relations selected")
    return out


# =============================================================================
# Mapping / prompt
# =============================================================================

def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Mapping[str, str]) -> str:
    letter_to_rel = {a: r for r, a in rel_to_letter.items()}
    lines = [
        f"Where is the {subject} relative to the {reference}?",
        "Use ONLY the mapping below and answer with exactly one letter: A, B, C, or D.",
    ]
    for a in LETTERS:
        lines.append(f"{a} = {letter_to_rel[a]}")
    lines.append("Answer:")
    return "\n".join(lines)


def assign_relation_balanced_mappings(items: Sequence[Mapping[str, Any]], seed: int) -> Dict[int, Dict[str, str]]:
    """Random mapping per sample; balance GT relation -> correct letter within each GT subgroup."""
    rng = random.Random(seed)
    out: Dict[int, Dict[str, str]] = {}
    by_rel: Dict[str, List[Mapping[str, Any]]] = {r: [] for r in REL}
    for m in items:
        by_rel[str(m["gt"])].append(m)

    for gt in REL:
        group = list(by_rel[gt])
        rng.shuffle(group)
        cycle = list(LETTERS)
        rng.shuffle(cycle)
        for i, m in enumerate(group):
            correct_letter = cycle[i % len(LETTERS)]
            other_rel = [r for r in REL if r != gt]
            other_letters = [a for a in LETTERS if a != correct_letter]
            rng.shuffle(other_rel)
            rng.shuffle(other_letters)
            mapping = {gt: correct_letter}
            mapping.update({r: a for r, a in zip(other_rel, other_letters)})
            out[int(m["sid"])] = mapping
    return out


def mapping_string(mp: Mapping[str, str]) -> str:
    return ", ".join(f"{r}->{mp[r]}" for r in REL)


# =============================================================================
# A/B/C/D first-step scoring
# =============================================================================

def build_option_token_map(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids: List[int] = []
        for text in (letter, " " + letter, "\n" + letter, "(" + letter + ")"):
            enc = tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == 1 and int(enc[0]) not in ids:
                ids.append(int(enc[0]))
        if not ids:
            enc = tokenizer.encode(letter, add_special_tokens=False)
            if not enc:
                raise RuntimeError(f"Tokenizer cannot encode option {letter}")
            ids = [int(enc[0])]
        out[letter] = ids
    return out


def extract_logits(outputs: Any) -> torch.Tensor:
    for value in (
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ):
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("Could not find LM logits in model output")


def option_logprobs_from_vector(logits: torch.Tensor, token_map: Mapping[str, Sequence[int]]) -> Dict[str, float]:
    logp = F.log_softmax(logits.float(), dim=-1)
    out: Dict[str, float] = {}
    for letter in LETTERS:
        ids = sorted({int(i) for i in token_map[letter] if 0 <= int(i) < int(logp.numel())})
        if not ids:
            raise RuntimeError(f"No valid token ids for option {letter}")
        idx = torch.as_tensor(ids, device=logp.device, dtype=torch.long)
        # Sum probability mass over single-token variants of the same answer letter.
        out[letter] = float(torch.logsumexp(logp.index_select(0, idx), dim=0).detach().cpu())
    return out


@torch.inference_mode()
def first_step_scores(model: Any, batch: Mapping[str, Any], token_map: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    outputs = model(**batch, use_cache=False, return_dict=True)
    logits = extract_logits(outputs)[0, -1]
    lp = option_logprobs_from_vector(logits, token_map)
    pred = max(LETTERS, key=lambda a: lp[a])
    del outputs
    return {"logprob": lp, "prediction": pred}


# =============================================================================
# Images / state extraction
# =============================================================================

def make_gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, color=(v, v, v))


def make_batch(processor: Any, device: torch.device, image: Image.Image, prompt: str) -> Dict[str, Any]:
    return base.make_question_batch(
        processor=processor,
        image=image,
        question_text=prompt,
        device=device,
    )


def collect_realgray_pair_states(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    rec: Any,
    prompt: str,
    subject: str,
    reference: str,
    layers: Sequence[int],
    object_state: str,
    relation_token_map: Mapping[str, Sequence[int]],
    gray_value: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Tuple[int, ...], Tuple[int, ...]]:
    real = gray = None
    rb = gb = None
    try:
        real = base.record_image(rec)
        if hasattr(real, "convert"):
            real = real.convert("RGB")
        gray = make_gray_image(real, gray_value)
        rb = make_batch(processor, device, real, prompt)
        gb = make_batch(processor, device, gray, prompt)

        cr = traj.clean_forward(
            base, model, processor, decoder_layers, rb,
            subject, reference, layers, object_state, relation_token_map,
        )
        cg = traj.clean_forward(
            base, model, processor, decoder_layers, gb,
            subject, reference, layers, object_state, relation_token_map,
        )

        q = {
            int(L): (np.asarray(cr["states"][L], np.float32) - np.asarray(cg["states"][L], np.float32)).astype(np.float32)
            for L in layers
        }
        return q, tuple(cr["subject_positions"]), tuple(cr["reference_positions"])
    finally:
        for im in (real, gray):
            if im is not None:
                with contextlib.suppress(Exception):
                    im.close()
        del rb, gb


# =============================================================================
# Spatial patch
# =============================================================================
class PairDeltaPatch:
    def __init__(
        self,
        layer_module: Any,
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
        delta_pair: np.ndarray,
    ):
        self.subject_positions = tuple(map(int, subject_positions))
        self.reference_positions = tuple(map(int, reference_positions))
        self.delta_pair = np.asarray(delta_pair, np.float32)
        self.applied = 0
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        h = traj.first_tensor(output)
        max_pos = max(self.subject_positions + self.reference_positions)
        if h.ndim != 3 or int(h.shape[1]) <= max_pos:
            return output
        y = h.float().clone()
        d = torch.as_tensor(self.delta_pair, device=y.device, dtype=torch.float32)
        for p in self.subject_positions:
            y[:, p, :] += 0.5 * d
        for p in self.reference_positions:
            y[:, p, :] -= 0.5 * d
        self.applied += 1
        return traj.replace_first_tensor(output, y.to(h.dtype))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()


@torch.inference_mode()
def patched_first_step_scores(
    *,
    model: Any,
    decoder_layers: Sequence[Any],
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    layer: int,
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
    delta_pair: np.ndarray,
) -> Dict[str, Any]:
    patch = PairDeltaPatch(
        decoder_layers[layer],
        subject_positions,
        reference_positions,
        delta_pair,
    )
    try:
        outputs = model(**batch, use_cache=False, return_dict=True)
        if patch.applied < 1:
            raise RuntimeError(f"L{layer} pair-state patch did not fire")
        logits = extract_logits(outputs)[0, -1]
        lp = option_logprobs_from_vector(logits, token_map)
        pred = max(LETTERS, key=lambda a: lp[a])
        del outputs
        return {"logprob": lp, "prediction": pred}
    finally:
        patch.close()


# =============================================================================
# IO / plotting
# =============================================================================

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    traj.write_csv(path, rows)


def plot_figure(
    *,
    output_path: Path,
    image: Image.Image,
    subject: str,
    reference: str,
    gt: str,
    opp: str,
    mapping: Mapping[str, str],
    sid: int,
    rows: Sequence[Mapping[str, Any]],
    alpha: float,
) -> None:
    layers = sorted({int(r["layer"]) for r in rows})
    by = {(str(r["condition"]), int(r["layer"])): r for r in rows}
    gt_letter = mapping[gt]
    opp_letter = mapping[opp]

    fig = plt.figure(figsize=(11.2, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.8], wspace=0.24)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(f"COCO sid={sid}", fontsize=10)
    query = f"Q: Where is the {subject} relative to the {reference}?"
    map_txt = "   ".join(f"{mapping[r]}={r}" for r in REL)
    ax0.text(
        0.0, -0.10,
        query + "\n" + map_txt + f"\nGT: {gt} -> {gt_letter}",
        transform=ax0.transAxes,
        va="top",
        fontsize=9,
        wrap=True,
    )

    ax = fig.add_subplot(gs[0, 1])
    condition_labels = {
        "orig": "orig",
        "toward_gt": f"steer -> {gt}",
        "toward_opp": f"steer -> {opp}",
    }

    condition_handles = []
    for cond in ("orig", "toward_gt", "toward_opp"):
        y_gt = [float(by[(cond, L)][f"logp_{gt_letter}"]) for L in layers]
        y_opp = [float(by[(cond, L)][f"logp_{opp_letter}"]) for L in layers]
        line_gt, = ax.plot(layers, y_gt, linestyle="-", linewidth=1.6, marker=".", label=condition_labels[cond])
        ax.plot(layers, y_opp, linestyle="--", linewidth=1.4, marker=".", color=line_gt.get_color())
        condition_handles.append(line_gt)

    ax.set_xlabel("Intervention Layer")
    ax.set_ylabel("Log Probability")
    ax.set_title(f"Spatial steering across layers (alpha={alpha:g})", fontsize=11)
    ax.grid(True, alpha=0.25)

    leg1 = ax.legend(handles=condition_handles, loc="best", fontsize=8, title="Condition")
    ax.add_artist(leg1)
    style_handles = [
        Line2D([0], [0], linestyle="-", linewidth=1.5, label=f"P({gt_letter}) = {gt}"),
        Line2D([0], [0], linestyle="--", linewidth=1.5, label=f"P({opp_letter}) = {opp}"),
    ]
    ax.legend(handles=style_handles, loc="lower left", fontsize=8, title="Answer belief")

    fig.suptitle("Belief Steering on Single Sample Across Layers", fontsize=12, y=0.98)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    a = parse_args()
    if a.alpha < 0:
        raise ValueError("--alpha must be >= 0")
    if not 0 <= a.gray_value <= 255:
        raise ValueError("--gray-value must be in [0,255]")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    err_path = out / "errors.jsonl"

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta: List[Dict[str, Any]] = []
    for r in records:
        sid = int(r.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
        })

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, test = traj.stratified_split(meta, a.train_ratio, a.seed)
    if a.train_max_samples > 0:
        train = traj.stratified_cap(train, a.train_max_samples, a.seed + 31)

    train_map = assign_relation_balanced_mappings(train, a.seed + 1001)
    test_map = assign_relation_balanced_mappings(test, a.seed + 2003)

    audit_rows: List[Dict[str, Any]] = []
    for split_name, items, maps in (("train", train, train_map), ("test", test, test_map)):
        for gt in REL:
            for letter in LETTERS:
                audit_rows.append({
                    "split": split_name,
                    "gt": gt,
                    "correct_letter": letter,
                    "n": sum(1 for m in items if m["gt"] == gt and maps[int(m["sid"])][gt] == letter),
                })
    write_csv(out / "mapping_balance.csv", audit_rows)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)
    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    model = processor = None
    try:
        print(f"[LOAD] {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        layers = parse_layers(a.layers, len(decoder_layers))
        relation_token_map = base.relation_token_variants(processor.tokenizer)
        option_token_map = build_option_token_map(processor.tokenizer)

        print("=" * 100)
        print("FIGURE A — SINGLE-SAMPLE SPATIAL STEERING ACROSS LAYERS")
        print("=" * 100)
        print(f"decoder={decoder_path} | layers={layers[0]}..{layers[-1]} ({len(layers)})")
        print(f"TRAIN={len(train)} TEST={len(test)} | alpha={a.alpha:g} | state=Real-Gray object pair")
        print()

        # ------------------------------------------------------------------
        # 1) Fit per-layer relation centroids from TRAIN Real-Gray object state.
        # ------------------------------------------------------------------
        train_q: Dict[int, Dict[int, np.ndarray]] = {}
        for m in tqdm(train, desc="TRAIN Real-Gray spatial states"):
            sid = int(m["sid"])
            mapping = train_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mapping)
            try:
                q, _sp, _rp = collect_realgray_pair_states(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    rec=rec_by_sid[sid],
                    prompt=prompt,
                    subject=m["subject"],
                    reference=m["reference"],
                    layers=layers,
                    object_state=a.object_state,
                    relation_token_map=relation_token_map,
                    gray_value=a.gray_value,
                    device=device,
                )
                train_q[sid] = q
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "train_realgray",
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                gc.collect()

        train_valid = [m for m in train if int(m["sid"]) in train_q]
        cent = traj.fit_centroids(train_valid, train_q, layers)
        np.savez_compressed(
            out / "train_relation_centroids_realgray.npz",
            **{f"L{L}_{r}": cent[L][r] for L in layers for r in REL},
        )

        # ------------------------------------------------------------------
        # 2) Choose one held-out illustrative sample by ORIGINAL first-step belief.
        # ------------------------------------------------------------------
        eligible_rel = set(parse_relations(a.sample_relations))
        test_by_sid = {int(m["sid"]): m for m in test}
        if a.sid is not None:
            if int(a.sid) not in test_by_sid:
                raise ValueError(
                    f"--sid {a.sid} is not in the held-out TEST split under seed={a.seed}. "
                    "Use metadata from an automatic run or change the split seed."
                )
            candidates = [test_by_sid[int(a.sid)]]
        else:
            candidates = [m for m in test if m["gt"] in eligible_rel]

        scored_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for m in tqdm(candidates, desc="select illustrative TEST sample"):
            sid = int(m["sid"])
            mapping = test_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mapping)
            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = make_batch(processor, device, image, prompt)
                sc = first_step_scores(model, batch, option_token_map)
                scored_candidates.append((m, sc))
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

        if not scored_candidates:
            raise RuntimeError("No eligible TEST sample could be scored")

        if a.sid is not None:
            chosen_m, chosen_sc = scored_candidates[0]
            selection_reason = "explicit_sid"
        else:
            def qualifies(pair, mode: str) -> bool:
                m, sc = pair
                mapping = test_map[int(m["sid"])]
                gt_letter = mapping[m["gt"]]
                opp_letter = mapping[OPP[m["gt"]]]
                if mode == "opposite":
                    return sc["prediction"] == opp_letter
                if mode == "wrong":
                    return sc["prediction"] != gt_letter
                return True

            chosen = None
            for mode in ([a.sample_mode] if a.sample_mode == "any" else [a.sample_mode, "wrong", "any"]):
                chosen = next((p for p in scored_candidates if qualifies(p, mode)), None)
                if chosen is not None:
                    selection_reason = mode
                    break
            assert chosen is not None
            chosen_m, chosen_sc = chosen

        sid = int(chosen_m["sid"])
        gt = str(chosen_m["gt"])
        opp = OPP[gt]
        mapping = test_map[sid]
        gt_letter = mapping[gt]
        opp_letter = mapping[opp]
        prompt = build_randmap_prompt(chosen_m["subject"], chosen_m["reference"], mapping)

        print(
            f"[SAMPLE] sid={sid} gt={gt}->{gt_letter} opp={opp}->{opp_letter} "
            f"orig_pred={chosen_sc['prediction']} selection={selection_reason}"
        )
        print(f"         mapping: {mapping_string(mapping)}")

        # Obtain object token positions on the selected real prompt.
        selected_image = selected_batch = None
        try:
            selected_image = base.record_image(rec_by_sid[sid])
            if hasattr(selected_image, "convert"):
                selected_image = selected_image.convert("RGB")
            selected_batch = make_batch(processor, device, selected_image, prompt)
            clean = traj.clean_forward(
                base, model, processor, decoder_layers, selected_batch,
                chosen_m["subject"], chosen_m["reference"], layers,
                a.object_state, relation_token_map,
            )
            subject_positions = tuple(clean["subject_positions"])
            reference_positions = tuple(clean["reference_positions"])

            orig_lp = chosen_sc["logprob"]
            rows: List[Dict[str, Any]] = []

            # ------------------------------------------------------------------
            # 3) Scan intervention layer. Each layer uses its own spatial axis.
            # ------------------------------------------------------------------
            for L in tqdm(layers, desc="single-sample layer steering"):
                raw_axis = (
                    np.asarray(cent[L][gt], np.float32)
                    - np.asarray(cent[L][opp], np.float32)
                )
                axis_norm = float(np.linalg.norm(raw_axis))
                if axis_norm < EPS:
                    raise RuntimeError(f"Near-zero {gt}<->{opp} centroid gap at L{L}")

                deltas = {
                    "orig": np.zeros_like(raw_axis, dtype=np.float32),
                    "toward_gt": (a.alpha * raw_axis).astype(np.float32),
                    "toward_opp": (-a.alpha * raw_axis).astype(np.float32),
                }

                for condition in ("orig", "toward_gt", "toward_opp"):
                    if condition == "orig":
                        sc = chosen_sc
                    else:
                        sc = patched_first_step_scores(
                            model=model,
                            decoder_layers=decoder_layers,
                            batch=selected_batch,
                            token_map=option_token_map,
                            layer=L,
                            subject_positions=subject_positions,
                            reference_positions=reference_positions,
                            delta_pair=deltas[condition],
                        )

                    row: Dict[str, Any] = {
                        "sid": sid,
                        "layer": int(L),
                        "condition": condition,
                        "alpha": float(a.alpha),
                        "gt_relation": gt,
                        "opposite_relation": opp,
                        "gt_letter": gt_letter,
                        "opposite_letter": opp_letter,
                        "prediction": sc["prediction"],
                        "centroid_gap_norm": axis_norm,
                        "edit_norm": float(np.linalg.norm(deltas[condition])),
                    }
                    for letter in LETTERS:
                        row[f"logp_{letter}"] = float(sc["logprob"][letter])
                    row["gt_vs_opp_margin"] = float(
                        sc["logprob"][gt_letter] - sc["logprob"][opp_letter]
                    )
                    rows.append(row)

            write_csv(out / "figureA_values.csv", rows)

            # Save a detached image copy before closing it.
            figure_image = selected_image.copy()
            plot_figure(
                output_path=out / "figureA_single_sample_layer_steering.png",
                image=figure_image,
                subject=chosen_m["subject"],
                reference=chosen_m["reference"],
                gt=gt,
                opp=opp,
                mapping=mapping,
                sid=sid,
                rows=rows,
                alpha=a.alpha,
            )
            figure_image.close()

            # Print the layers with strongest correct-direction margin gain.
            orig_margin = float(orig_lp[gt_letter] - orig_lp[opp_letter])
            gain_rows = []
            by = {(r["condition"], int(r["layer"])): r for r in rows}
            for L in layers:
                rg = by[("toward_gt", L)]
                ro = by[("toward_opp", L)]
                gain_rows.append((
                    float(rg["gt_vs_opp_margin"] - orig_margin),
                    int(L),
                    float(rg["gt_vs_opp_margin"]),
                    float(ro["gt_vs_opp_margin"]),
                ))
            gain_rows.sort(reverse=True)
            print("\nTop layers by GT-vs-opposite margin gain under spatial steering:")
            for gain, L, gt_m, opp_m in gain_rows[:10]:
                print(
                    f"  L{L:02d} gain={gain:+.4f} | toward_gt margin={gt_m:+.4f} "
                    f"| toward_opp margin={opp_m:+.4f}"
                )

            metadata = {
                "script_version": SCRIPT_VERSION,
                "model": spec.repo_id,
                "decoder_path": decoder_path,
                "layers": layers,
                "alpha": a.alpha,
                "train_ratio": a.train_ratio,
                "train_n": len(train_valid),
                "test_n": len(test),
                "seed": a.seed,
                "object_state": a.object_state,
                "gray_value": a.gray_value,
                "selected_sid": sid,
                "selection_reason": selection_reason,
                "subject": chosen_m["subject"],
                "reference": chosen_m["reference"],
                "gt_relation": gt,
                "opposite_relation": opp,
                "mapping": mapping,
                "gt_letter": gt_letter,
                "opposite_letter": opp_letter,
                "original_prediction": chosen_sc["prediction"],
                "original_logprobs": chosen_sc["logprob"],
                "spatial_state_fit": "q=(real_sub-real_ref)-(gray_sub-gray_ref), TRAIN only",
                "spatial_edit": "delta_L=alpha*(mu_GT,L-mu_OPPOSITE,L); subject += delta/2; reference -= delta/2",
                "readout": "first-step log probability mass over A/B/C/D single-token variants",
            }
            traj.write_json(out / "metadata.json", metadata)

            print(f"\n[SAVED] {out / 'figureA_single_sample_layer_steering.png'}")
            print(f"[SAVED] {out / 'figureA_values.csv'}")

        finally:
            if selected_image is not None:
                with contextlib.suppress(Exception):
                    selected_image.close()
            del selected_batch

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
