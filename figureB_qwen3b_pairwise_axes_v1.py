#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairwise Figure-B style dataset-average spatial steering along semantic axes.

This script is a cleaner Linear-Fig6B-style variant of Figure B.
Instead of plotting one target relation against the mean of the other three,
it directly contrasts the two opposite relations on each spatial axis:

  Horizontal axis: left  vs right
  Vertical axis:   under vs on

For a fixed intervention layer L, we fit TRAIN relation centroids from the
Real-Gray object-pair state, then steer TEST samples along one semantic axis:

  d_H = mu_right - mu_left
  d_V = mu_above - mu_below

Positive alpha means:
  horizontal: toward right
  vertical:   toward on
Negative alpha means:
  horizontal: toward left
  vertical:   toward under

For each sample we read the logits of the *mapped option letters* currently
assigned to the corresponding spatial relations under the sample's randomized
A/B/C/D mapping.

Outputs:
  figureB_horizontal_left_vs_right.png
  figureB_vertical_under_vs_on.png
  figureB_pairwise_two_panel.png
  figureB_pairwise_rows.csv
  figureB_pairwise_summary.csv
  train_relation_centroids_realgray.npz
  mapping_balance.csv
  metadata.json
  errors.jsonl
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
SCRIPT_VERSION = "figureB-qwen3b-pairwise-axes-v1"

DISPLAY_REL = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}

AXES = {
    "horizontal": {
        "neg_rel": "left",
        "pos_rel": "right",
        "direction": ("left", "right"),  # mu_pos - mu_neg
        "filename": "horizontal_left_vs_right",
    },
    "vertical": {
        "neg_rel": "below",
        "pos_rel": "above",
        "direction": ("below", "above"),
        "filename": "vertical_under_vs_on",
    },
}


def disp_rel(r: str) -> str:
    return DISPLAY_REL.get(str(r), str(r))


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
    p.add_argument("--layer", type=int, default=22)
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        help="Space-separated strengths, e.g. --alphas -1.5 -1.0 -0.5 0 0.5 1.0 1.5",
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--object-state", default="last", choices=["last", "mean"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--train-max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=0)
    p.add_argument(
        "--test-filter",
        default="all",
        choices=["all", "correct", "wrong"],
        help="Filter by ORIGINAL unedited first-step prediction vs GT-mapped option.",
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


def parse_alphas(spec: Sequence[float]) -> List[float]:
    vals = sorted(float(x) for x in spec)
    if not vals:
        raise ValueError("No alpha values provided")
    return vals


def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Mapping[str, str]) -> str:
    letter_to_rel = {a: r for r, a in rel_to_letter.items()}
    lines = [
        f"Determine the spatial relation of the {subject} to the {reference} in the image.",
        *(f"{a}. {disp_rel(letter_to_rel[a])}" for a in LETTERS),
        "Answer with only A, B, C, or D.",
    ]
    return "\n".join(lines)


def assign_relation_balanced_mappings(items: Sequence[Mapping[str, Any]], seed: int) -> Dict[int, Dict[str, str]]:
    by_rel: Dict[str, List[Mapping[str, Any]]] = {r: [] for r in REL}
    for m in items:
        by_rel[m["gt"]].append(m)
    out: Dict[int, Dict[str, str]] = {}
    rng = random.Random(seed)
    perms = list(__import__("itertools").permutations(LETTERS, 4))
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
            mp = {rel: letter for rel, letter in zip(order, perm)}
            out[int(m["sid"])] = mp
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


def option_logits_from_vector(logits: torch.Tensor, token_map: Mapping[str, Sequence[int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for letter in LETTERS:
        idx = torch.tensor([int(x) for x in token_map[letter]], device=logits.device, dtype=torch.long)
        out[letter] = float(logits.index_select(0, idx).max().detach().cpu())
    return out


@torch.inference_mode()
def first_step_scores(model: Any, batch: Mapping[str, Any], token_map: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    outputs = model(**batch, use_cache=False, return_dict=True)
    logits = extract_logits(outputs)[0, -1]
    letter_logits = option_logits_from_vector(logits, token_map)
    pred = max(LETTERS, key=lambda a: letter_logits[a])
    del outputs
    return {"letter_logits": letter_logits, "prediction": pred}


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


class PairDeltaPatch:
    def __init__(self, layer_module: Any, subject_positions: Sequence[int], reference_positions: Sequence[int], delta_pair: np.ndarray):
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
    patch = PairDeltaPatch(decoder_layers[layer], subject_positions, reference_positions, delta_pair)
    try:
        outputs = model(**batch, use_cache=False, return_dict=True)
        if patch.applied < 1:
            raise RuntimeError(f"L{layer} pair-state patch did not fire")
        logits = extract_logits(outputs)[0, -1]
        letter_logits = option_logits_from_vector(logits, token_map)
        pred = max(LETTERS, key=lambda a: letter_logits[a])
        del outputs
        return {"letter_logits": letter_logits, "prediction": pred}
    finally:
        patch.close()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_axis_rows(rows: Sequence[Mapping[str, Any]], axis_name: str) -> List[Dict[str, Any]]:
    by_alpha: Dict[float, List[Mapping[str, Any]]] = {}
    for r in rows:
        if r["axis"] != axis_name:
            continue
        by_alpha.setdefault(float(r["alpha"]), []).append(r)
    out: List[Dict[str, Any]] = []
    for alpha in sorted(by_alpha):
        chunk = by_alpha[alpha]
        neg = np.asarray([float(x["delta_neg_logit"]) for x in chunk], np.float64)
        pos = np.asarray([float(x["delta_pos_logit"]) for x in chunk], np.float64)
        out.append({
            "axis": axis_name,
            "alpha": alpha,
            "n": int(len(chunk)),
            "neg_mean": float(neg.mean()) if len(neg) else float("nan"),
            "neg_sem": float(neg.std(ddof=1) / math.sqrt(len(neg))) if len(neg) > 1 else 0.0,
            "pos_mean": float(pos.mean()) if len(pos) else float("nan"),
            "pos_sem": float(pos.std(ddof=1) / math.sqrt(len(pos))) if len(pos) > 1 else 0.0,
        })
    return out


def plot_axis(ax: Any, summary_rows: Sequence[Mapping[str, Any]], axis_name: str) -> None:
    cfg = AXES[axis_name]
    neg_name = disp_rel(cfg["neg_rel"])
    pos_name = disp_rel(cfg["pos_rel"])
    x = np.asarray([float(r["alpha"]) for r in summary_rows], np.float64)
    y_neg = np.asarray([float(r["neg_mean"]) for r in summary_rows], np.float64)
    e_neg = np.asarray([float(r["neg_sem"]) for r in summary_rows], np.float64)
    y_pos = np.asarray([float(r["pos_mean"]) for r in summary_rows], np.float64)
    e_pos = np.asarray([float(r["pos_sem"]) for r in summary_rows], np.float64)

    ax.axhline(0.0, linestyle="--", linewidth=1.4, color="0.35")
    ax.plot(x, y_pos, marker="o", markersize=5.8, linewidth=2.6, label=f"$\\Delta$P({pos_name})")
    ax.fill_between(x, y_pos - e_pos, y_pos + e_pos, alpha=0.18)
    ax.plot(x, y_neg, marker="o", markersize=5.8, linewidth=2.6, label=f"$\\Delta$P({neg_name})")
    ax.fill_between(x, y_neg - e_neg, y_neg + e_neg, alpha=0.18)
    ax.set_xlabel("Spatial edit strength $\\alpha$", fontsize=15)
    ax.set_ylabel("Mean $\\Delta$ answer logit", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=12, loc="best")
    if axis_name == "horizontal":
        ax.set_title("Horizontal axis: left vs right", fontsize=15)
    else:
        ax.set_title("Vertical axis: under vs on", fontsize=15)


def save_single_plot(path: Path, summary_rows: Sequence[Mapping[str, Any]], axis_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.25, 4.25), dpi=220)
    plot_axis(ax, summary_rows, axis_name)
    fig.tight_layout(pad=0.7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_two_panel(path: Path, horiz: Sequence[Mapping[str, Any]], vert: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.2), dpi=220)
    plot_axis(axes[0], horiz, "horizontal")
    plot_axis(axes[1], vert, "vertical")
    fig.tight_layout(pad=0.8, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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

    alphas = parse_alphas(a.alphas)

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
    if a.eval_max_samples > 0:
        test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 53)

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
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor)
        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        if not 0 <= int(a.layer) < len(decoder_layers):
            raise ValueError(f"Invalid --layer {a.layer}; decoder has {len(decoder_layers)} blocks")
        relation_token_map = base.relation_token_variants(processor.tokenizer)
        option_token_map = build_option_token_map(processor.tokenizer)

        print("=" * 100)
        print("PAIRWISE FIGURE B — DATASET-AVERAGE SPATIAL STEERING ALONG SEMANTIC AXES")
        print("=" * 100)
        print(f"decoder={decoder_path} | layer=L{a.layer} | alphas={alphas}")
        print(f"TRAIN={len(train)} TEST={len(test)} | filter={a.test_filter} | state=Real-Gray object pair")
        print()

        train_q: Dict[int, Dict[int, np.ndarray]] = {}
        one_layer = [int(a.layer)]
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
                    layers=one_layer,
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
        cent = traj.fit_centroids(train_valid, train_q, one_layer)
        np.savez_compressed(
            out / "train_relation_centroids_realgray.npz",
            **{f"L{a.layer}_{r}": cent[a.layer][r] for r in REL},
        )

        rows: List[Dict[str, Any]] = []
        kept = 0
        for m in tqdm(test, desc="TEST pairwise steering rows"):
            sid = int(m["sid"])
            mapping = test_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mapping)
            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = make_batch(processor, device, image, prompt)
                base_sc = first_step_scores(model, batch, option_token_map)
                gt_letter = mapping[m["gt"]]
                is_correct = (base_sc["prediction"] == gt_letter)
                if a.test_filter == "correct" and not is_correct:
                    continue
                if a.test_filter == "wrong" and is_correct:
                    continue

                _q, subject_positions, reference_positions = collect_realgray_pair_states(
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
                kept += 1

                for axis_name, cfg in AXES.items():
                    neg_rel = cfg["neg_rel"]
                    pos_rel = cfg["pos_rel"]
                    direction = np.asarray(cent[a.layer][pos_rel] - cent[a.layer][neg_rel], np.float32)

                    neg_letter = mapping[neg_rel]
                    pos_letter = mapping[pos_rel]
                    base_neg = float(base_sc["letter_logits"][neg_letter])
                    base_pos = float(base_sc["letter_logits"][pos_letter])

                    for alpha in alphas:
                        patched = patched_first_step_scores(
                            model=model,
                            decoder_layers=decoder_layers,
                            batch=batch,
                            token_map=option_token_map,
                            layer=int(a.layer),
                            subject_positions=subject_positions,
                            reference_positions=reference_positions,
                            delta_pair=(float(alpha) * direction).astype(np.float32),
                        )
                        neg_val = float(patched["letter_logits"][neg_letter])
                        pos_val = float(patched["letter_logits"][pos_letter])
                        rows.append({
                            "sid": sid,
                            "gt_relation": m["gt"],
                            "base_correct": int(is_correct),
                            "base_prediction": base_sc["prediction"],
                            "mapping": mapping_string(mapping),
                            "axis": axis_name,
                            "neg_relation": neg_rel,
                            "neg_letter": neg_letter,
                            "pos_relation": pos_rel,
                            "pos_letter": pos_letter,
                            "alpha": float(alpha),
                            "layer": int(a.layer),
                            "delta_neg_logit": neg_val - base_neg,
                            "delta_pos_logit": pos_val - base_pos,
                            "patched_prediction": patched["prediction"],
                        })
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "test_pairwise_steering",
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
            raise RuntimeError("No evaluation rows were produced")

        write_csv(out / "figureB_pairwise_rows.csv", rows)
        horiz = summarize_axis_rows(rows, "horizontal")
        vert = summarize_axis_rows(rows, "vertical")
        write_csv(out / "figureB_pairwise_summary.csv", horiz + vert)

        save_single_plot(out / "figureB_horizontal_left_vs_right.png", horiz, "horizontal")
        save_single_plot(out / "figureB_vertical_under_vs_on.png", vert, "vertical")
        save_two_panel(out / "figureB_pairwise_two_panel.png", horiz, vert)

        meta_out = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "layer": int(a.layer),
            "alphas": alphas,
            "seed": int(a.seed),
            "train_ratio": float(a.train_ratio),
            "object_state": a.object_state,
            "gray_value": int(a.gray_value),
            "max_samples": int(a.max_samples),
            "train_max_samples": int(a.train_max_samples),
            "eval_max_samples": int(a.eval_max_samples),
            "test_filter": a.test_filter,
            "n_train_total": len(train),
            "n_test_total": len(test),
            "n_test_kept": kept,
            "n_rows": len(rows),
        }
        (out / "metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

        print("=" * 100)
        print("PAIRWISE FIGURE B COMPLETE")
        print("=" * 100)
        print(f"model={a.model} | layer=L{a.layer} | TEST kept={kept} | rows={len(rows)} | output={out}", flush=True)
        print("Generated figures:")
        print(f"  - {out / 'figureB_horizontal_left_vs_right.png'}")
        print(f"  - {out / 'figureB_vertical_under_vs_on.png'}")
        print(f"  - {out / 'figureB_pairwise_two_panel.png'}")
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
