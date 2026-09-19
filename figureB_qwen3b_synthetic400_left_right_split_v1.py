#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure B: Synthetic-400 spatial direction -> COCO left/right belief steering.

This version matches the intended experiment more closely:

  Source for the steering direction:
      ALL 400 synthetic-shape samples (100 per spatial relation)

  Target/evaluation set:
      ALL COCO two-object samples whose GT relation is left or right
      (no COCO train/test split)

  Per selected decoder layer L:
      q_{i,L} = (h_real_sub - h_real_ref) - (h_gray_sub - h_gray_ref)

      mu_left,L  = mean q over synthetic-left samples
      mu_right,L = mean q over synthetic-right samples

      d_H,L = mu_right,L - mu_left,L

  Intervention on each COCO sample at the SAME layer L:
      h_sub += 0.5 * alpha * d_H,L
      h_ref -= 0.5 * alpha * d_H,L

  Readout:
      Each COCO sample keeps its own randomized relation -> A/B/C/D mapping.
      We measure the mapped option for LEFT and RIGHT, but the plot is aligned
      by semantics and does not display the letters.

  Split exactly like the Linear-style plot:
      panel 1: Original GT = left
      panel 2: Original GT = right

  Curves:
      Delta log P(mapped-right option)
      Delta log P(mapped-left option)

  Shaded regions:
      mean +/- SEM across all samples in that panel.

Important:
  - Synthetic-400 is used ONLY to construct the spatial direction.
  - COCO does not contribute to direction fitting.
  - There is no COCO train/test split in this script.
  - If --layer is changed, the synthetic direction is re-estimated at that
    exact layer, i.e. L22 uses d_H,22, L23 uses d_H,23, etc.

Expected synthetic layout (default):
    synthetic_shapes_4dir_400/
        labels.jsonl
        images/ ...

Each labels.jsonl row should contain at least:
    image, subject, reference, relation
and may optionally contain id.
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
DISPLAY_REL = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
SCRIPT_VERSION = "figureB-qwen3b-synthetic400-left-right-split-v1"


def disp_rel(r: str) -> str:
    return DISPLAY_REL.get(str(r), str(r))


def canonical_relation(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    table = {
        "left": "left",
        "left_of": "left",
        "right": "right",
        "right_of": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "top": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
        "bottom": "below",
    }
    if s not in table:
        raise ValueError(f"Unknown relation label: {x!r}")
    return table[s]


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
    p.add_argument(
        "--synthetic-root",
        default="synthetic_shapes_4dir_400",
        help="Root directory of the original 400 synthetic shapes dataset.",
    )
    p.add_argument(
        "--synthetic-labels",
        default="labels.jsonl",
        help="labels.jsonl path relative to --synthetic-root (or absolute path).",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--layer", type=int, default=22)
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        help="Space-separated steering strengths. Negative=leftward, positive=rightward.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument(
        "--synthetic-max-samples",
        type=int,
        default=0,
        help="0 = all synthetic samples. Intended paper run should use all 400.",
    )
    p.add_argument(
        "--coco-max-samples",
        type=int,
        default=0,
        help="0 = all COCO-two samples before keeping only left/right.",
    )
    p.add_argument(
        "--test-filter",
        default="all",
        choices=["all", "correct", "wrong"],
        help="Optional filter based on ORIGINAL mapped-option prediction. Paper default: all.",
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


# =============================================================================
# Random option mapping / prompt / option scoring
# =============================================================================

def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Mapping[str, str]) -> str:
    letter_to_rel = {a: r for r, a in rel_to_letter.items()}
    return "\n".join([
        f"Determine the spatial relation of the {subject} to the {reference} in the image.",
        *(f"{a}. {disp_rel(letter_to_rel[a])}" for a in LETTERS),
        "Answer with only A, B, C, or D.",
    ])


def assign_relation_balanced_mappings(
    items: Sequence[Mapping[str, Any]],
    seed: int,
) -> Dict[int, Dict[str, str]]:
    import itertools

    by_rel: Dict[str, List[Mapping[str, Any]]] = {r: [] for r in REL}
    for m in items:
        by_rel[str(m["gt"])].append(m)

    out: Dict[int, Dict[str, str]] = {}
    rng = random.Random(seed)
    perms = list(itertools.permutations(LETTERS, 4))

    for r in REL:
        arr = list(by_rel[r])
        rng.shuffle(arr)
        for i, m in enumerate(arr):
            perm = perms[i % len(perms)]
            rels = list(REL)
            rels.remove(r)
            rng_local = random.Random(seed * 1000003 + int(m["sid"]) * 9176 + i)
            rng_local.shuffle(rels)
            order = [r] + rels
            out[int(m["sid"])] = {rel: letter for rel, letter in zip(order, perm)}
    return out


def mapping_string(mp: Mapping[str, str]) -> str:
    return ",".join(f"{r}->{mp[r]}" for r in REL)


def build_option_token_map(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids: List[int] = []
        for text in [letter, " " + letter, "\n" + letter, "(" + letter + ")"]:
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
    for value in [
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ]:
        if torch.is_tensor(value) and value.ndim == 3:
            return value
    raise RuntimeError("Could not locate logits tensor")


def option_scores_from_vector(
    logits: torch.Tensor,
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, float]]:
    """Return both raw option logits and full-vocab log probabilities."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    raw_out: Dict[str, float] = {}
    lp_out: Dict[str, float] = {}
    for letter in LETTERS:
        idx = torch.tensor(
            [int(x) for x in token_map[letter]],
            device=logits.device,
            dtype=torch.long,
        )
        raw_out[letter] = float(logits.index_select(0, idx).max().detach().cpu())
        lp_out[letter] = float(logp.index_select(0, idx).max().detach().cpu())
    return {"letter_logits": raw_out, "letter_logprobs": lp_out}


@torch.inference_mode()
def first_step_scores(
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    outputs = model(**batch, use_cache=False, return_dict=True)
    logits = extract_logits(outputs)[0, -1]
    scores = option_scores_from_vector(logits, token_map)
    pred = max(LETTERS, key=lambda a: scores["letter_logits"][a])
    del outputs
    return {**scores, "prediction": pred}


# =============================================================================
# Synthetic-400 loading
# =============================================================================

def resolve_synthetic_labels(root: Path, spec: str) -> Path:
    p = Path(spec)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"Synthetic labels not found: {p}")
    return p


def resolve_synthetic_image(root: Path, image_field: str) -> Path:
    p = Path(str(image_field))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            root / p,
            root / "images" / p,
            root / "images" / p.name,
        ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not resolve synthetic image {image_field!r}. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def load_synthetic400(root: Path, labels_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            relation_raw = obj.get("relation", obj.get("answer", obj.get("label")))
            if relation_raw is None:
                raise KeyError(f"No relation/answer/label in {labels_path}:{line_no}")
            gt = canonical_relation(relation_raw)
            subject = obj.get("subject")
            reference = obj.get("reference")
            image_field = obj.get("image", obj.get("image_path", obj.get("file")))
            if subject is None or reference is None or image_field is None:
                raise KeyError(
                    f"Synthetic row {line_no} needs image, subject, reference; got keys={sorted(obj.keys())}"
                )
            sid_raw = obj.get("id", obj.get("sid", line_no - 1))
            try:
                sid = int(sid_raw)
            except Exception:
                sid = line_no - 1
            image_path = resolve_synthetic_image(root, str(image_field))
            rows.append({
                "sid": sid,
                "gt": gt,
                "subject": str(subject),
                "reference": str(reference),
                "image_path": str(image_path),
            })

    # Ensure unique integer ids for mapping assignment even if the JSON ids collide.
    seen = set()
    for i, r in enumerate(rows):
        if r["sid"] in seen:
            r["sid"] = 1_000_000 + i
        seen.add(r["sid"])
    return rows


def stratified_cap(
    items: Sequence[Mapping[str, Any]],
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_samples <= 0 or len(items) <= max_samples:
        return [dict(x) for x in items]
    return traj.stratified_cap([dict(x) for x in items], max_samples, seed)


# =============================================================================
# Image batches / state extraction
# =============================================================================

def make_gray_image(image: Image.Image, value: int) -> Image.Image:
    v = int(max(0, min(255, value)))
    return Image.new("RGB", image.size, color=(v, v, v))


def make_batch(
    processor: Any,
    device: torch.device,
    image: Image.Image,
    prompt: str,
) -> Dict[str, Any]:
    return base.make_question_batch(
        processor=processor,
        image=image,
        question_text=prompt,
        device=device,
    )


def collect_realgray_pair_states_from_image(
    *,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    image: Image.Image,
    prompt: str,
    subject: str,
    reference: str,
    layers: Sequence[int],
    object_state: str,
    relation_token_map: Mapping[str, Sequence[int]],
    gray_value: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Tuple[int, ...], Tuple[int, ...]]:
    real = image.convert("RGB") if hasattr(image, "convert") else image
    gray = make_gray_image(real, gray_value)
    rb = gb = None
    try:
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
            int(L): (
                np.asarray(cr["states"][L], np.float32)
                - np.asarray(cg["states"][L], np.float32)
            ).astype(np.float32)
            for L in layers
        }
        return q, tuple(cr["subject_positions"]), tuple(cr["reference_positions"])
    finally:
        with contextlib.suppress(Exception):
            gray.close()
        del rb, gb


def collect_coco_realgray_pair_states(
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
    image = None
    try:
        image = base.record_image(rec)
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return collect_realgray_pair_states_from_image(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            image=image,
            prompt=prompt,
            subject=subject,
            reference=reference,
            layers=layers,
            object_state=object_state,
            relation_token_map=relation_token_map,
            gray_value=gray_value,
            device=device,
        )
    finally:
        if image is not None:
            with contextlib.suppress(Exception):
                image.close()


# =============================================================================
# Intervention
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
        scores = option_scores_from_vector(logits, token_map)
        pred = max(LETTERS, key=lambda a: scores["letter_logits"][a])
        del outputs
        return {**scores, "prediction": pred}
    finally:
        patch.close()


# =============================================================================
# Statistics / plotting
# =============================================================================

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sem(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    original_relation: str,
) -> List[Dict[str, Any]]:
    by_alpha: Dict[float, List[Mapping[str, Any]]] = {}
    for r in rows:
        if str(r["original_relation"]) != original_relation:
            continue
        by_alpha.setdefault(float(r["alpha"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for alpha in sorted(by_alpha):
        chunk = by_alpha[alpha]
        left_lp = np.asarray([float(x["delta_left_logprob"]) for x in chunk], np.float64)
        right_lp = np.asarray([float(x["delta_right_logprob"]) for x in chunk], np.float64)
        left_logit = np.asarray([float(x["delta_left_logit"]) for x in chunk], np.float64)
        right_logit = np.asarray([float(x["delta_right_logit"]) for x in chunk], np.float64)
        out.append({
            "original_relation": original_relation,
            "alpha": alpha,
            "n": len(chunk),
            "left_logprob_mean": float(left_lp.mean()),
            "left_logprob_sem": sem(left_lp),
            "right_logprob_mean": float(right_lp.mean()),
            "right_logprob_sem": sem(right_lp),
            "left_logit_mean": float(left_logit.mean()),
            "left_logit_sem": sem(left_logit),
            "right_logit_mean": float(right_logit.mean()),
            "right_logit_sem": sem(right_logit),
        })
    return out


def plot_panel(
    ax: Any,
    summary_rows: Sequence[Mapping[str, Any]],
    title: str,
) -> None:
    x = np.asarray([float(r["alpha"]) for r in summary_rows], np.float64)
    y_right = np.asarray([float(r["right_logprob_mean"]) for r in summary_rows], np.float64)
    e_right = np.asarray([float(r["right_logprob_sem"]) for r in summary_rows], np.float64)
    y_left = np.asarray([float(r["left_logprob_mean"]) for r in summary_rows], np.float64)
    e_left = np.asarray([float(r["left_logprob_sem"]) for r in summary_rows], np.float64)

    ax.axhline(0.0, linestyle="--", linewidth=1.3, color="0.35", zorder=0)

    line_r, = ax.plot(
        x, y_right,
        marker="o", markersize=5.5, linewidth=2.4,
        label="Right",
    )
    ax.fill_between(
        x, y_right - e_right, y_right + e_right,
        color=line_r.get_color(), alpha=0.18, linewidth=0,
    )

    line_l, = ax.plot(
        x, y_left,
        marker="o", markersize=5.5, linewidth=2.4,
        label="Left",
    )
    ax.fill_between(
        x, y_left - e_left, y_left + e_left,
        color=line_l.get_color(), alpha=0.18, linewidth=0,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r"Spatial edit strength $\alpha$", fontsize=14)
    ax.set_ylabel("Change in log probability", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.20)
    ax.legend(frameon=False, fontsize=11, loc="best")


def save_two_panel(
    path: Path,
    left_summary: Sequence[Mapping[str, Any]],
    right_summary: Sequence[Mapping[str, Any]],
    layer: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), dpi=240)
    plot_panel(axes[0], left_summary, f"Layer {layer} — Original: 'left'")
    plot_panel(axes[1], right_summary, f"Layer {layer} — Original: 'right'")
    fig.tight_layout(pad=0.7, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    a = parse_args()
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

    alphas = sorted(float(x) for x in a.alphas)
    if not alphas:
        raise ValueError("No --alphas supplied")

    # ------------------------------------------------------------------
    # Synthetic-400 source set
    # ------------------------------------------------------------------
    synth_root = Path(a.synthetic_root)
    synth_labels = resolve_synthetic_labels(synth_root, a.synthetic_labels)
    synthetic = load_synthetic400(synth_root, synth_labels)
    synthetic = stratified_cap(synthetic, a.synthetic_max_samples, a.seed + 101)

    synth_counts = {r: sum(1 for x in synthetic if x["gt"] == r) for r in REL}
    print(f"[Synthetic] n={len(synthetic)} counts={synth_counts}")

    # Balanced randomized relation->letter mappings prevent the source spatial
    # centroid from being tied to any fixed answer letter.
    synth_map = assign_relation_balanced_mappings(synthetic, a.seed + 1001)

    # ------------------------------------------------------------------
    # COCO target set: ALL samples, no train/test split
    # ------------------------------------------------------------------
    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    coco_all: List[Dict[str, Any]] = []
    for r in records:
        sid = int(r.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        coco_all.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
        })

    coco_all = stratified_cap(coco_all, a.coco_max_samples, a.seed + 211)
    coco_lr = [x for x in coco_all if x["gt"] in ("left", "right")]
    coco_counts = {
        "left": sum(1 for x in coco_lr if x["gt"] == "left"),
        "right": sum(1 for x in coco_lr if x["gt"] == "right"),
    }
    print(f"[COCO left/right] n={len(coco_lr)} counts={coco_counts}")

    coco_map = assign_relation_balanced_mappings(coco_all, a.seed + 2003)

    # Mapping audit.
    audit_rows: List[Dict[str, Any]] = []
    for split_name, items, maps in (
        ("synthetic_source", synthetic, synth_map),
        ("coco_all", coco_all, coco_map),
    ):
        for gt in REL:
            for letter in LETTERS:
                audit_rows.append({
                    "set": split_name,
                    "gt": gt,
                    "correct_letter": letter,
                    "n": sum(
                        1 for m in items
                        if m["gt"] == gt and maps[int(m["sid"])][gt] == letter
                    ),
                })
    write_csv(out / "mapping_balance.csv", audit_rows)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
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
        layer = int(a.layer)
        if not 0 <= layer < len(decoder_layers):
            raise ValueError(f"Invalid --layer {layer}; decoder has {len(decoder_layers)} blocks")
        one_layer = [layer]

        relation_token_map = base.relation_token_variants(processor.tokenizer)
        option_token_map = build_option_token_map(processor.tokenizer)

        print("=" * 108)
        print("FIGURE B — SYNTHETIC-400 HORIZONTAL DIRECTION -> COCO LEFT/RIGHT")
        print("=" * 108)
        print(f"decoder={decoder_path} | layer=L{layer} | alphas={alphas}")
        print(f"synthetic source n={len(synthetic)} | COCO left/right n={len(coco_lr)}")
        print(f"COCO filter={a.test_filter} | no COCO train/test split")
        print()

        # --------------------------------------------------------------
        # 1) Fit relation centroids from ALL synthetic-400 at this layer.
        # --------------------------------------------------------------
        synth_q: Dict[int, Dict[int, np.ndarray]] = {}
        synth_meta_valid: List[Dict[str, Any]] = []

        for m in tqdm(synthetic, desc=f"Synthetic-400 Real-Gray states @ L{layer}"):
            sid = int(m["sid"])
            prompt = build_randmap_prompt(
                m["subject"], m["reference"], synth_map[sid]
            )
            image = None
            try:
                image = Image.open(m["image_path"]).convert("RGB")
                q, _sp, _rp = collect_realgray_pair_states_from_image(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    image=image,
                    prompt=prompt,
                    subject=m["subject"],
                    reference=m["reference"],
                    layers=one_layer,
                    object_state=a.object_state,
                    relation_token_map=relation_token_map,
                    gray_value=a.gray_value,
                    device=device,
                )
                synth_q[sid] = q
                synth_meta_valid.append(m)
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "synthetic_direction_fit",
                    "sid": sid,
                    "image_path": m["image_path"],
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                gc.collect()

        cent = traj.fit_centroids(synth_meta_valid, synth_q, one_layer)
        mu_left = np.asarray(cent[layer]["left"], np.float32)
        mu_right = np.asarray(cent[layer]["right"], np.float32)
        d_h = (mu_right - mu_left).astype(np.float32)
        d_h_norm = float(np.linalg.norm(d_h))

        np.savez_compressed(
            out / f"synthetic400_relation_centroids_L{layer}.npz",
            **{f"L{layer}_{r}": np.asarray(cent[layer][r], np.float32) for r in REL},
            horizontal_direction=d_h,
        )

        print(f"[DIRECTION] ||mu_right - mu_left|| = {d_h_norm:.6f}")

        # --------------------------------------------------------------
        # 2) Evaluate on ALL COCO left/right samples.
        # --------------------------------------------------------------
        rows: List[Dict[str, Any]] = []
        kept_left = 0
        kept_right = 0

        for m in tqdm(coco_lr, desc=f"COCO left/right steering @ L{layer}"):
            sid = int(m["sid"])
            mapping = coco_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mapping)
            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = make_batch(processor, device, image, prompt)
                base_sc = first_step_scores(model, batch, option_token_map)

                gt_letter = mapping[m["gt"]]
                base_correct = (base_sc["prediction"] == gt_letter)
                if a.test_filter == "correct" and not base_correct:
                    continue
                if a.test_filter == "wrong" and base_correct:
                    continue

                # Get object token positions using the same Real/Gray protocol.
                _q, subject_positions, reference_positions = collect_coco_realgray_pair_states(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    rec=rec_by_sid[sid],
                    prompt=prompt,
                    subject=m["subject"],
                    reference=m["reference"],
                    layers=one_layer,
                    object_state=a.object_state,
                    relation_token_map=relation_token_map,
                    gray_value=a.gray_value,
                    device=device,
                )

                if m["gt"] == "left":
                    kept_left += 1
                else:
                    kept_right += 1

                left_letter = mapping["left"]
                right_letter = mapping["right"]

                base_left_logit = float(base_sc["letter_logits"][left_letter])
                base_right_logit = float(base_sc["letter_logits"][right_letter])
                base_left_logp = float(base_sc["letter_logprobs"][left_letter])
                base_right_logp = float(base_sc["letter_logprobs"][right_letter])

                for alpha in alphas:
                    patched = patched_first_step_scores(
                        model=model,
                        decoder_layers=decoder_layers,
                        batch=batch,
                        token_map=option_token_map,
                        layer=layer,
                        subject_positions=subject_positions,
                        reference_positions=reference_positions,
                        delta_pair=(float(alpha) * d_h).astype(np.float32),
                    )

                    left_logit = float(patched["letter_logits"][left_letter])
                    right_logit = float(patched["letter_logits"][right_letter])
                    left_logp = float(patched["letter_logprobs"][left_letter])
                    right_logp = float(patched["letter_logprobs"][right_letter])

                    rows.append({
                        "sid": sid,
                        "original_relation": m["gt"],
                        "mapping": mapping_string(mapping),
                        "left_letter": left_letter,
                        "right_letter": right_letter,
                        "base_prediction": base_sc["prediction"],
                        "base_correct": int(base_correct),
                        "layer": layer,
                        "alpha": float(alpha),
                        "delta_left_logprob": left_logp - base_left_logp,
                        "delta_right_logprob": right_logp - base_right_logp,
                        "delta_left_logit": left_logit - base_left_logit,
                        "delta_right_logit": right_logit - base_right_logit,
                        "patched_prediction": patched["prediction"],
                    })

            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "coco_steering",
                    "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

        if not rows:
            raise RuntimeError("No COCO evaluation rows were produced")

        write_csv(out / "figureB_synth400_left_right_rows.csv", rows)

        left_summary = summarize_rows(rows, "left")
        right_summary = summarize_rows(rows, "right")
        write_csv(
            out / "figureB_synth400_left_right_summary.csv",
            left_summary + right_summary,
        )

        save_two_panel(
            out / "figureB_synth400_left_right_two_panel.png",
            left_summary,
            right_summary,
            layer,
        )

        metadata = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "layer": layer,
            "alphas": alphas,
            "seed": int(a.seed),
            "object_state": a.object_state,
            "gray_value": int(a.gray_value),
            "synthetic_root": str(synth_root),
            "synthetic_labels": str(synth_labels),
            "synthetic_n": len(synthetic),
            "synthetic_counts": synth_counts,
            "synthetic_valid_n": len(synth_meta_valid),
            "horizontal_direction_norm": d_h_norm,
            "coco_all_n": len(coco_all),
            "coco_left_right_n": len(coco_lr),
            "coco_left_right_counts": coco_counts,
            "test_filter": a.test_filter,
            "kept_left": kept_left,
            "kept_right": kept_right,
            "n_rows": len(rows),
            "direction_definition": "mu_right,L - mu_left,L from Synthetic-400 Real-Gray object-pair residual",
            "evaluation_split": "all COCO left/right samples; no COCO train/test split",
            "plot_metric": "change in full-vocabulary log probability of each relation's mapped option",
            "error_band": "mean +/- SEM across samples",
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + "=" * 108)
        print("FIGURE B COMPLETE")
        print("=" * 108)
        print(f"Synthetic source: n={len(synthetic)} counts={synth_counts}")
        print(f"COCO kept: left={kept_left}, right={kept_right}")
        print(f"Direction norm: {d_h_norm:.6f}")
        print(f"[SAVED] {out / 'figureB_synth400_left_right_two_panel.png'}")
        print(f"[SAVED] {out / 'figureB_synth400_left_right_rows.csv'}")
        print(f"[SAVED] {out / 'figureB_synth400_left_right_summary.csv'}")

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
