#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-layer object-relation trajectory repair for COCO two-object spatial reasoning.

Core idea
=========
At decoder block-output layer L:

    r_L = h_A^L - h_B^L

TRAIN examples define four relation centroids per layer.  On held-out eval
examples, a CLEAN multi-layer detector votes on the internal relation.  Repair
is triggered only when that internal relation confidently disagrees with the
model's clean greedy-generation answer (default: only exact opposite pairs).

For a gated sample, suppose clean generation says RIGHT and the clean internal
detector says LEFT.  At every patched layer L we use that layer's own axis:

    u_L = normalize(mu_L[LEFT] - mu_L[RIGHT])

During the patched forward we recompute the CURRENT already-patched state:

    r_current = h_A - h_B
    s_current = <r_current, u_L>
    s_target  = <mu_L[LEFT], u_L>
    delta_r   = alpha * (s_target - s_current) * u_L

and patch symmetrically:

    h_A <- h_A + delta_r/2
    h_B <- h_B - delta_r/2

Thus the object-pair mean is preserved.  Different layers use different axes;
the target relation is frozen from the CLEAN pass; receiver heads, prompt-last,
and output logits are never patched directly.

Default bundles:
    L22
    L21-L22
    L21-L23
    L21-L24
    L22-L24

Primary question:
    Does full held-out greedy-generation ACC improve?

Outputs:
    clean_rows.csv
    baseline_eval.csv
    train_centroids.npz
    detector_layer_metrics.csv
    gate_candidates.csv
    patch_results.csv / patch_results.jsonl
    summary.csv
    report.txt
    config.json
    errors.jsonl

Smoke test:
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_coco_multilayer_relation_trajectory_repair_v1.py \
  --model qwen-3b \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --diagnostic-layers 21-24 \
  --bundles "22;21-22;21-23;21-24;22-24" \
  --alphas 0.05,0.10,0.20 \
  --train-ratio 0.15 \
  --max-samples 80 \
  --device cuda:0 \
  --output-dir output/qwen3b_relation_trajectory_repair_smoke \
  --overwrite

Full run:
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_coco_multilayer_relation_trajectory_repair_v1.py \
  --model qwen-3b \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --diagnostic-layers 21-24 \
  --bundles "22;21-22;21-23;21-24;22-24" \
  --alphas 0.05,0.10,0.20 \
  --train-ratio 0.15 \
  --device cuda:0 \
  --output-dir output/qwen3b_relation_trajectory_repair_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

SCRIPT_VERSION = "coco-multilayer-relation-trajectory-repair-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager", "sdpa", "flash_attention_2", "none"),
    )
    p.add_argument(
        "--core-module",
        default="analyze_coco_centroid_generation_step1_v4",
        help="Existing AdaptVis helper module.",
    )
    p.add_argument(
        "--diagnostic-layers",
        default="21-24",
        help="CLEAN layers that vote on the pseudo-target.",
    )
    p.add_argument(
        "--bundles",
        default="22;21-22;21-23;21-24;22-24",
        help="Semicolon-separated intervention layer bundles.",
    )
    p.add_argument(
        "--alphas",
        default="0.05,0.10,0.20",
        help="Per-layer pull fraction toward target centroid coordinate.",
    )
    p.add_argument(
        "--object-state",
        default="mean",
        choices=("last", "mean"),
    )
    p.add_argument(
        "--centroid-metric",
        default="cosine",
        choices=("cosine", "euclidean"),
    )
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="0 = all; otherwise stratified cap before split.",
    )
    p.add_argument(
        "--eval-max-samples", type=int, default=0,
        help="0 = all eval rows; otherwise stratified eval cap.",
    )
    p.add_argument("--min-layer-agreement", type=float, default=0.75)
    p.add_argument("--min-mean-margin", type=float, default=0.0)
    p.add_argument(
        "--gate",
        default="disagree_opposite",
        choices=("disagree_opposite", "disagree_any"),
    )
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast", action=argparse.BooleanOptionalAction, default=True
    )
    return p.parse_args()


def parse_layer_spec(text: str) -> List[int]:
    out, seen = [], set()
    for raw in str(text).split(","):
        item = raw.strip().upper().replace("L", "")
        if not item:
            continue
        if "-" in item:
            a, b = map(int, item.split("-", 1))
            vals = range(min(a, b), max(a, b) + 1)
        else:
            vals = (int(item),)
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
    if not out:
        raise ValueError(f"Empty layer spec: {text!r}")
    return sorted(out)


def parse_bundles(text: str) -> List[Tuple[int, ...]]:
    out, seen = [], set()
    for raw in str(text).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        b = tuple(parse_layer_spec(raw))
        if b not in seen:
            seen.add(b)
            out.append(b)
    if not out:
        raise ValueError("No intervention bundles")
    return out


def bundle_name(bundle: Sequence[int]) -> str:
    return "+".join(f"L{x}" for x in bundle)


def parse_float_list(text: str) -> List[float]:
    out = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        x = float(raw)
        if x < 0:
            raise ValueError("alpha must be >= 0")
        if x not in out:
            out.append(x)
    if not out:
        raise ValueError("No alphas")
    return out


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def normalize_relation(base: Any, value: Any) -> Optional[str]:
    if value is None:
        return None
    out = base.normalize_relation(value)
    return str(out) if out in RELATIONS else None


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unable to find first tensor in {type(output).__name__}")


def replace_first_tensor(output: Any, new_first: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return new_first
    if isinstance(output, tuple):
        return (new_first, *output[1:])
    if isinstance(output, list):
        return [new_first, *output[1:]]
    raise TypeError(f"Unsupported output type: {type(output).__name__}")


def span_positions(span: Tuple[int, int], mode: str) -> Tuple[int, ...]:
    start, end = int(span[0]), int(span[1])
    return (end,) if mode == "last" else tuple(range(start, end + 1))


def span_state(hidden: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor(list(map(int, positions)), device=hidden.device, dtype=torch.long)
    return hidden[0].index_select(0, idx).mean(dim=0)


def output_hidden_layers(outputs: Any, n_layers: int) -> Tuple[torch.Tensor, ...]:
    hs = tuple(getattr(outputs, "hidden_states", ()) or ())
    if len(hs) == n_layers + 1:
        return hs[1:]
    if len(hs) == n_layers:
        return hs
    raise RuntimeError(f"Unexpected hidden_states length={len(hs)} n_layers={n_layers}")


def stratified_cap(rows, limit: int, seed: int):
    rows = [dict(r) for r in rows]
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[str(r["gt"])].append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    ptr = {k: 0 for k in keys}
    out = []
    while len(out) < limit:
        moved = False
        for k in keys:
            if len(out) >= limit:
                break
            i = ptr[k]
            if i < len(groups[k]):
                out.append(groups[k][i])
                ptr[k] += 1
                moved = True
        if not moved:
            break
    rng.shuffle(out)
    return out


def stratified_split(rows, train_ratio: float, seed: int):
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be in (0,1)")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[str(r["gt"])].append(dict(r))
    train, eval_rows = [], []
    for rel in RELATIONS:
        vals = groups[rel]
        rng.shuffle(vals)
        if len(vals) < 2:
            raise RuntimeError(f"Need >=2 samples for {rel}")
        n = max(1, int(round(len(vals) * train_ratio)))
        n = min(n, len(vals) - 1)
        train += vals[:n]
        eval_rows += vals[n:]
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    return train, eval_rows


def fit_centroids(train_rows, state_by_sid, layers):
    out = {}
    for L in layers:
        out[L] = {}
        for rel in RELATIONS:
            xs = [state_by_sid[int(r["sid"])][L] for r in train_rows if r["gt"] == rel]
            if not xs:
                raise RuntimeError(f"No TRAIN states for L{L} {rel}")
            out[L][rel] = np.mean(np.stack(xs), axis=0).astype(np.float32)
    return out


def centroid_scores(r, centroids, metric):
    r = np.asarray(r, np.float32)
    scores = {}
    if metric == "cosine":
        rn = float(np.linalg.norm(r))
        for rel in RELATIONS:
            c = np.asarray(centroids[rel], np.float32)
            scores[rel] = float(np.dot(r, c) / max(rn * float(np.linalg.norm(c)), EPS))
    else:
        for rel in RELATIONS:
            c = np.asarray(centroids[rel], np.float32)
            scores[rel] = -float(np.linalg.norm(r - c))
    return scores


def detector_one(r, centroids, metric):
    scores = centroid_scores(r, centroids, metric)
    ranked = sorted(RELATIONS, key=lambda x: scores[x], reverse=True)
    return ranked[0], float(scores[ranked[0]] - scores[ranked[1]]), scores


def detector_consensus(sid, state_by_sid, centroid_by_layer, layers, metric):
    votes = Counter()
    margin_sum = defaultdict(float)
    per_layer = {}
    for L in layers:
        pred, margin, scores = detector_one(state_by_sid[sid][L], centroid_by_layer[L], metric)
        per_layer[L] = {"prediction": pred, "margin": margin, "scores": scores}
        votes[pred] += 1
        margin_sum[pred] += margin
    ranked = sorted(RELATIONS, key=lambda r: (votes[r], margin_sum[r]), reverse=True)
    target = ranked[0]
    target_layers = [L for L in layers if per_layer[L]["prediction"] == target]
    agreement = len(target_layers) / len(layers)
    mean_margin = safe_mean(per_layer[L]["margin"] for L in target_layers)
    return target, agreement, mean_margin, per_layer, dict(votes)


class DynamicLayerRepair:
    def __init__(
        self,
        layer_module,
        layer: int,
        subject_positions,
        reference_positions,
        source_relation: str,
        target_relation: str,
        centroids,
        alpha: float,
    ):
        self.layer_module = layer_module
        self.layer = int(layer)
        self.subject_positions = tuple(map(int, subject_positions))
        self.reference_positions = tuple(map(int, reference_positions))
        self.alpha = float(alpha)
        self.applied = 0
        self.last_meta = {}
        mu_src = np.asarray(centroids[source_relation], np.float32)
        mu_tgt = np.asarray(centroids[target_relation], np.float32)
        axis = (mu_tgt - mu_src).astype(np.float64)
        n = float(np.linalg.norm(axis))
        if n < EPS:
            raise RuntimeError(f"L{layer}: zero source-target centroid axis")
        self.axis = (axis / n).astype(np.float32)
        self.target_coord = float(np.dot(mu_tgt.astype(np.float64), self.axis.astype(np.float64)))
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        hidden = first_tensor(output)
        max_pos = max(self.subject_positions + self.reference_positions)
        # Full prompt / prefill only. Cached decoding later has sequence length 1.
        if int(hidden.shape[1]) <= max_pos:
            return output
        y = hidden.float().clone()
        a = span_state(y, self.subject_positions)
        b = span_state(y, self.reference_positions)
        r = a - b
        u = torch.as_tensor(self.axis, device=y.device, dtype=torch.float32)
        current_coord = float(torch.dot(r.float(), u).item())
        scalar = self.alpha * (self.target_coord - current_coord)
        delta = float(scalar) * u
        for pos in self.subject_positions:
            y[:, pos, :] = y[:, pos, :] + 0.5 * delta
        for pos in self.reference_positions:
            y[:, pos, :] = y[:, pos, :] - 0.5 * delta
        self.applied += 1
        self.last_meta = {
            "current_coord": current_coord,
            "target_coord": self.target_coord,
            "scalar": float(scalar),
            "delta_norm": float(delta.norm().item()),
        }
        return replace_first_tensor(output, y.to(hidden.dtype))

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()


class MultiLayerRepair:
    def __init__(self, decoder_layers, bundle, subject_positions, reference_positions,
                 source_relation, target_relation, centroid_by_layer, alpha):
        self.hooks = [
            DynamicLayerRepair(
                decoder_layers[L], L, subject_positions, reference_positions,
                source_relation, target_relation, centroid_by_layer[L], alpha
            )
            for L in sorted(bundle)
        ]

    def validate(self):
        missing = [h.layer for h in self.hooks if h.applied < 1]
        if missing:
            raise RuntimeError(f"Repair hooks did not fire: {missing}")

    def metadata(self):
        return {h.layer: dict(h.last_meta) for h in self.hooks}

    def close(self):
        for h in reversed(self.hooks):
            h.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


@torch.inference_mode()
def clean_forward(base, model, processor, decoder_layers, batch, subject, reference,
                  layers, object_state, relation_token_map):
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    subject_span, reference_span = base.locate_object_spans(
        processor.tokenizer, input_ids, subject, reference
    )
    apos = span_positions(subject_span, object_state)
    bpos = span_positions(reference_span, object_state)
    outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_layers = output_hidden_layers(outputs, len(decoder_layers))
    states = {}
    for L in layers:
        a = span_state(hidden_layers[L], apos)
        b = span_state(hidden_layers[L], bpos)
        states[L] = (a - b).detach().float().cpu().numpy().astype(np.float32)
    rel = base.relation_scores(outputs.logits[0, -1], relation_token_map, gt=None)
    result = {
        "states": states,
        "subject_positions": apos,
        "reference_positions": bpos,
        "first_step_prediction": rel["prediction"],
        "first_step_logits": np.asarray(rel["logits"], np.float32),
    }
    del outputs, hidden_layers
    return result


@torch.inference_mode()
def patched_first_step(base, model, decoder_layers, batch, apos, bpos, bundle,
                       source_relation, target_relation, centroid_by_layer, alpha,
                       relation_token_map):
    with MultiLayerRepair(
        decoder_layers, bundle, apos, bpos, source_relation, target_relation,
        centroid_by_layer, alpha
    ) as repair:
        outputs = model(**batch, use_cache=False, return_dict=True)
        repair.validate()
        rel = base.relation_scores(outputs.logits[0, -1], relation_token_map, gt=None)
        meta = repair.metadata()
    out = {"prediction": rel["prediction"], "logits": np.asarray(rel["logits"], np.float32)}
    del outputs
    return out, meta


@torch.inference_mode()
def patched_generation(base, model, processor, decoder_layers, batch, apos, bpos,
                       bundle, source_relation, target_relation, centroid_by_layer,
                       alpha, max_new_tokens):
    with MultiLayerRepair(
        decoder_layers, bundle, apos, bpos, source_relation, target_relation,
        centroid_by_layer, alpha
    ) as repair:
        text = base.generate_text(model, processor, batch, max_new_tokens=max_new_tokens)
        repair.validate()
        meta = repair.metadata()
    return normalize_relation(base, text), text, meta


def build_summary(eval_rows, patch_rows):
    baseline_correct = {int(r["sid"]): bool(r["baseline_generation_correct"]) for r in eval_rows}
    baseline_pred = {int(r["sid"]): r["baseline_generation_prediction"] for r in eval_rows}
    baseline_acc = safe_mean(float(v) for v in baseline_correct.values())
    groups = defaultdict(list)
    for r in patch_rows:
        groups[(r["bundle"], float(r["alpha"]))].append(r)
    out = []
    for (bundle, alpha), rows in sorted(groups.items()):
        patched_correct = dict(baseline_correct)
        patched_pred = dict(baseline_pred)
        for r in rows:
            sid = int(r["sid"])
            patched_correct[sid] = bool(r["patched_generation_correct"])
            patched_pred[sid] = r["patched_generation_prediction"]
        acc = safe_mean(float(v) for v in patched_correct.values())
        w2c = sum((not baseline_correct[s]) and patched_correct[s] for s in baseline_correct)
        c2w = sum(baseline_correct[s] and (not patched_correct[s]) for s in baseline_correct)
        out.append({
            "bundle": bundle,
            "alpha": alpha,
            "N_eval": len(eval_rows),
            "N_gated_patched": len(rows),
            "gate_rate": len(rows) / max(len(eval_rows), 1),
            "baseline_generation_acc": baseline_acc,
            "patched_generation_acc": acc,
            "delta_acc": acc - baseline_acc,
            "wrong_to_correct": int(w2c),
            "correct_to_wrong": int(c2w),
            "net_repairs": int(w2c - c2w),
            "generation_changed": int(sum(patched_pred[s] != baseline_pred[s] for s in baseline_pred)),
            "gated_follow_internal": safe_mean(float(r["patched_generation_prediction"] == r["internal_target"]) for r in rows),
            "first_step_follow_internal": safe_mean(float(r["patched_first_step_prediction"] == r["internal_target"]) for r in rows),
            "mean_total_delta_norm": safe_mean(float(r["total_delta_norm"]) for r in rows),
        })
    return out


def print_summary(rows):
    print("\n" + "=" * 132)
    print("MULTI-LAYER RELATION TRAJECTORY REPAIR")
    print("=" * 132)
    print(f"{'bundle':<22s} {'alpha':>6s} {'N':>5s} {'gated':>6s} {'baseACC':>9s} {'patchACC':>9s} {'delta':>8s} {'W->C':>5s} {'C->W':>5s} {'net':>5s} {'follow':>9s}")
    print("-" * 132)
    for r in rows:
        print(
            f"{r['bundle']:<22s} {float(r['alpha']):>6.2f} {int(r['N_eval']):>5d} {int(r['N_gated_patched']):>6d} "
            f"{100*float(r['baseline_generation_acc']):>8.2f}% {100*float(r['patched_generation_acc']):>8.2f}% "
            f"{100*float(r['delta_acc']):>+7.2f} {int(r['wrong_to_correct']):>5d} {int(r['correct_to_wrong']):>5d} "
            f"{int(r['net_repairs']):>5d} {100*float(r['gated_follow_internal']):>8.2f}%"
        )
    print("=" * 132)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    diagnostic_layers = parse_layer_spec(args.diagnostic_layers)
    bundles = parse_bundles(args.bundles)
    alphas = parse_float_list(args.alphas)
    all_layers = sorted(set(diagnostic_layers).union(*[set(b) for b in bundles]))

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Output dir not empty: {outdir}; use --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)
    errors_path = outdir / "errors.jsonl"
    patch_jsonl = outdir / "patch_results.jsonl"

    base = importlib.import_module(args.core_module)
    two_object = base.import_two_object_module()
    prompt_path = Path(args.prompt_jsonl)
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    prompt_rows = base.load_standard_prompts(prompt_path)
    records, audit = two_object.load_records(args.dataset, Path(args.data_root), None)
    records_by_sid = {int(r.sid): r for r in records}

    metadata = []
    for record in records:
        sid = int(record.sid)
        if sid not in prompt_rows:
            continue
        p = prompt_rows[sid]
        gt = normalize_relation(base, p["answer_raw"])
        if gt not in RELATIONS:
            continue
        metadata.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })
    metadata = stratified_cap(metadata, args.max_samples, args.seed)
    train_rows, eval_meta = stratified_split(metadata, args.train_ratio, args.seed)
    eval_meta = stratified_cap(eval_meta, args.eval_max_samples, args.seed + 1)
    meta_by_sid = {int(r["sid"]): r for r in eval_meta}

    specs = base.merged_model_specs(two_object)
    if args.model not in specs:
        raise KeyError(f"Unknown model {args.model}; available={sorted(specs)}")
    spec = specs[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers has no {spec.model_class}")
    load_kwargs = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    model = processor = None
    state_by_sid = {}
    clean_rows = []
    try:
        print(f"Loading {args.model}: {spec.repo_id}", flush=True)
        model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor)
        device = torch.device(args.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        relation_token_map = base.relation_token_variants(processor.tokenizer)
        for L in all_layers:
            if not 0 <= L < len(decoder_layers):
                raise ValueError(f"L{L} outside 0..{len(decoder_layers)-1}")

        train_sid = {int(r["sid"]) for r in train_rows}
        eval_sid = {int(r["sid"]) for r in eval_meta}
        baseline_eval = []

        print("\n" + "=" * 112)
        print("PHASE 1: CLEAN STATE EXTRACTION")
        print("=" * 112)
        print("train/eval       :", len(train_rows), "/", len(eval_meta))
        print("diagnostic layers:", diagnostic_layers)
        print("bundles          :", [bundle_name(b) for b in bundles])
        print("alphas           :", alphas)
        print("object state     :", args.object_state)
        print("=" * 112, flush=True)

        for idx, meta in enumerate(tqdm(train_rows + eval_meta, desc="clean-extract"), 1):
            sid = int(meta["sid"])
            image = batch = None
            try:
                image = base.record_image(records_by_sid[sid])
                batch = base.make_question_batch(
                    processor=processor, image=image, question_text=meta["question_text"], device=device
                )
                clean = clean_forward(
                    base, model, processor, decoder_layers, batch,
                    meta["subject"], meta["reference"], all_layers,
                    args.object_state, relation_token_map
                )
                state_by_sid[sid] = clean["states"]
                row = {
                    **meta,
                    "split": "train" if sid in train_sid else "eval",
                    "clean_first_step_prediction": clean["first_step_prediction"],
                    "clean_first_step_logits": clean["first_step_logits"].tolist(),
                }
                if sid in eval_sid:
                    text = base.generate_text(model, processor, batch, max_new_tokens=args.max_new_tokens)
                    pred = normalize_relation(base, text)
                    row.update({
                        "baseline_generation_text": text,
                        "baseline_generation_prediction": pred,
                        "baseline_generation_correct": pred == meta["gt"],
                    })
                    baseline_eval.append(row)
                clean_rows.append(row)
            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "clean", "sid": sid,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception): image.close()
                del batch
                gc.collect()
                if torch.cuda.is_available() and args.empty_cache_every > 0 and idx % args.empty_cache_every == 0:
                    torch.cuda.empty_cache()

        valid = set(state_by_sid)
        train_rows = [r for r in train_rows if int(r["sid"]) in valid]
        baseline_eval = [r for r in baseline_eval if int(r["sid"]) in valid]
        write_csv(outdir / "clean_rows.csv", clean_rows)
        write_csv(outdir / "baseline_eval.csv", baseline_eval)

        centroid_by_layer = fit_centroids(train_rows, state_by_sid, all_layers)
        np.savez_compressed(
            outdir / "train_centroids.npz",
            **{f"L{L}_{rel}": centroid_by_layer[L][rel] for L in all_layers for rel in RELATIONS}
        )

        layer_metrics = []
        for L in diagnostic_layers:
            correct = []
            for r in baseline_eval:
                pred, _, _ = detector_one(state_by_sid[int(r["sid"])][L], centroid_by_layer[L], args.centroid_metric)
                correct.append(pred == r["gt"])
            layer_metrics.append({"layer": L, "N_eval": len(correct), "centroid_accuracy": safe_mean(float(x) for x in correct)})
        write_csv(outdir / "detector_layer_metrics.csv", layer_metrics)

        gate_rows, gate_by_sid = [], {}
        for r in baseline_eval:
            sid = int(r["sid"])
            target, agreement, mean_margin, per_layer, votes = detector_consensus(
                sid, state_by_sid, centroid_by_layer, diagnostic_layers, args.centroid_metric
            )
            gen = r["baseline_generation_prediction"]
            reason = ""
            gate = True
            if gen not in RELATIONS:
                gate, reason = False, "generation_unparsed"
            elif agreement < args.min_layer_agreement:
                gate, reason = False, "low_agreement"
            elif not math.isfinite(mean_margin) or mean_margin < args.min_mean_margin:
                gate, reason = False, "low_margin"
            elif target == gen:
                gate, reason = False, "internal_equals_generation"
            elif args.gate == "disagree_opposite" and target != OPPOSITE[gen]:
                gate, reason = False, "not_opposite"
            else:
                reason = args.gate
            gr = {
                "sid": sid,
                "gt": r["gt"],
                "baseline_generation_prediction": gen,
                "baseline_generation_correct": r["baseline_generation_correct"],
                "internal_target": target,
                "internal_target_correct": target == r["gt"],
                "agreement": agreement,
                "mean_margin": mean_margin,
                "votes": json.dumps(votes, ensure_ascii=False),
                "gate": gate,
                "gate_reason": reason,
            }
            for L in diagnostic_layers:
                gr[f"L{L}_pred"] = per_layer[L]["prediction"]
                gr[f"L{L}_margin"] = per_layer[L]["margin"]
            gate_rows.append(gr)
            gate_by_sid[sid] = gr
        write_csv(outdir / "gate_candidates.csv", gate_rows)

        gated = [r for r in gate_rows if r["gate"]]
        baseline_acc = safe_mean(float(r["baseline_generation_correct"]) for r in baseline_eval)
        gate_precision = safe_mean(float(r["internal_target_correct"]) for r in gated)
        print("\n" + "=" * 112)
        print("DETECTOR / GATE")
        print("=" * 112)
        print(f"baseline generation ACC : {100*baseline_acc:.2f}%")
        print(f"gated                  : {len(gated)}/{len(baseline_eval)} ({100*len(gated)/max(len(baseline_eval),1):.2f}%)")
        print(f"gated target precision : {100*gate_precision:.2f}%  [GT diagnostic only]")
        for m in layer_metrics:
            print(f"L{m['layer']} centroid ACC       : {100*m['centroid_accuracy']:.2f}%")
        print("=" * 112, flush=True)
        if not gated:
            raise RuntimeError("Gate selected zero samples")

        baseline_by_sid = {int(r["sid"]): r for r in baseline_eval}
        patch_rows = []
        gated_sids = [int(r["sid"]) for r in gated]

        print("\n" + "=" * 112)
        print("PHASE 2: MULTI-LAYER DYNAMIC REPAIR")
        print("=" * 112, flush=True)

        for ci, (bundle, alpha) in enumerate([(b,a) for b in bundles for a in alphas], 1):
            print(f"\n[{ci}/{len(bundles)*len(alphas)}] {bundle_name(bundle)} alpha={alpha:g}", flush=True)
            for j, sid in enumerate(tqdm(gated_sids, desc=f"{bundle_name(bundle)}:{alpha:g}"), 1):
                image = batch = None
                try:
                    base_row = baseline_by_sid[sid]
                    gate_row = gate_by_sid[sid]
                    meta = meta_by_sid[sid]
                    source = str(base_row["baseline_generation_prediction"])
                    target = str(gate_row["internal_target"])
                    image = base.record_image(records_by_sid[sid])
                    batch = base.make_question_batch(
                        processor=processor, image=image, question_text=meta["question_text"], device=device
                    )
                    ids = batch["input_ids"][0].detach().cpu().tolist()
                    sspan, rspan = base.locate_object_spans(processor.tokenizer, ids, meta["subject"], meta["reference"])
                    apos = span_positions(sspan, args.object_state)
                    bpos = span_positions(rspan, args.object_state)

                    first, first_meta = patched_first_step(
                        base, model, decoder_layers, batch, apos, bpos, bundle,
                        source, target, centroid_by_layer, alpha, relation_token_map
                    )
                    gpred, gtext, _ = patched_generation(
                        base, model, processor, decoder_layers, batch, apos, bpos, bundle,
                        source, target, centroid_by_layer, alpha, args.max_new_tokens
                    )
                    total_norm = sum(float(x.get("delta_norm", 0.0)) for x in first_meta.values())
                    out = {
                        "sid": sid,
                        "gt": base_row["gt"],
                        "bundle": bundle_name(bundle),
                        "layers": ",".join(map(str, bundle)),
                        "alpha": float(alpha),
                        "baseline_generation_prediction": source,
                        "baseline_generation_correct": bool(base_row["baseline_generation_correct"]),
                        "internal_target": target,
                        "internal_target_correct": target == base_row["gt"],
                        "agreement": gate_row["agreement"],
                        "mean_margin": gate_row["mean_margin"],
                        "baseline_first_step_prediction": base_row["clean_first_step_prediction"],
                        "patched_first_step_prediction": first["prediction"],
                        "patched_generation_text": gtext,
                        "patched_generation_prediction": gpred,
                        "patched_generation_correct": gpred == base_row["gt"],
                        "generation_changed": gpred != source,
                        "generation_followed_internal": gpred == target,
                        "wrong_to_correct": (not base_row["baseline_generation_correct"]) and gpred == base_row["gt"],
                        "correct_to_wrong": base_row["baseline_generation_correct"] and gpred != base_row["gt"],
                        "total_delta_norm": float(total_norm),
                        "layer_patch_meta": json.dumps({str(k):v for k,v in first_meta.items()}, ensure_ascii=False),
                    }
                    patch_rows.append(out)
                    append_jsonl(patch_jsonl, out)
                except Exception as exc:
                    append_jsonl(errors_path, {
                        "phase": "patch", "sid": sid, "bundle": bundle_name(bundle), "alpha": alpha,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    if args.fail_fast:
                        raise
                finally:
                    if image is not None:
                        with contextlib.suppress(Exception): image.close()
                    del batch
                    gc.collect()
                    if torch.cuda.is_available() and args.empty_cache_every > 0 and j % args.empty_cache_every == 0:
                        torch.cuda.empty_cache()

        write_csv(outdir / "patch_results.csv", patch_rows)
        summary = build_summary(baseline_eval, patch_rows)
        write_csv(outdir / "summary.csv", summary)
        print_summary(summary)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "transformers_version": transformers.__version__,
            "decoder_path": decoder_path,
            "diagnostic_layers": diagnostic_layers,
            "bundles": [list(b) for b in bundles],
            "alphas": alphas,
            "object_state": args.object_state,
            "centroid_metric": args.centroid_metric,
            "train_ratio": args.train_ratio,
            "gate": args.gate,
            "min_layer_agreement": args.min_layer_agreement,
            "min_mean_margin": args.min_mean_margin,
            "N_train": len(train_rows),
            "N_eval": len(baseline_eval),
            "N_gated": len(gated),
            "baseline_generation_acc": baseline_acc,
            "gated_internal_precision_diagnostic_only": gate_precision,
            "uses_gt_to_choose_eval_repair": False,
            "patch_formula": "u=normalize(mu_target-mu_source); delta=alpha*(<mu_target,u>-<hA-hB,u>)*u; hA+=delta/2; hB-=delta/2",
            "audit": audit,
        }
        write_json(outdir / "config.json", config)

        best = sorted(summary, key=lambda r: (-r["delta_acc"], -r["net_repairs"]))
        lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"model: {args.model}",
            f"train/eval: {len(train_rows)}/{len(baseline_eval)}",
            f"baseline generation ACC: {100*baseline_acc:.2f}%",
            f"gated: {len(gated)}/{len(baseline_eval)}",
            f"gated target precision (diagnostic only): {100*gate_precision:.2f}%",
            "",
            "BEST SETTINGS",
        ]
        for r in best[:10]:
            lines.append(
                f"{r['bundle']} alpha={r['alpha']:.3f}: "
                f"{100*r['baseline_generation_acc']:.2f}% -> {100*r['patched_generation_acc']:.2f}% "
                f"(delta={100*r['delta_acc']:+.2f} pp, W->C={r['wrong_to_correct']}, C->W={r['correct_to_wrong']}, net={r['net_repairs']})"
            )
        (outdir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        print("\nSaved:")
        for name in [
            "config.json", "clean_rows.csv", "baseline_eval.csv", "train_centroids.npz",
            "detector_layer_metrics.csv", "gate_candidates.csv", "patch_results.csv",
            "patch_results.jsonl", "summary.csv", "report.txt"
        ]:
            print(" ", outdir / name)

    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
