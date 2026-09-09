#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Middle-token amplification multi-layer search with true all-data evaluation.
# Compile-checked clean copy; leading long docstring removed to avoid quote corruption.

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument(
        "--source-bundles",
        default="22+24;20+22+24;22+24+26;18+20+22+24+26",
        help='Semicolon-separated source-layer bundles, e.g. "22;24;22+24".',
    )
    p.add_argument("--target-layers", default="32,34,35")
    p.add_argument("--ks", default="4,8,12,16,24,32")
    p.add_argument("--alphas", default="0.125,0.25,0.5,0.75,1.0")
    p.add_argument(
        "--conditions",
        default="positive",
        help="Subset of positive,random,negative. Positive-only is recommended for the expanded search.",
    )
    p.add_argument(
        "--selection-strategies",
        default="global,global_unique",
        help=(
            "Comma-separated: per_layer, global, global_unique. "
            "per_layer means K per source layer (old behavior); "
            "global means K total (layer,position) edits across the bundle; "
            "global_unique means K total and edits each token position at most once, "
            "choosing the strongest source layer for that position."
        ),
    )
    p.add_argument("--direct-writer-alphas", default="1.0")
    p.add_argument("--writer-mode", default="centered", choices=["centered", "raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=32)
    p.add_argument(
        "--eval-scope",
        default="test",
        choices=["test", "all_data"],
        help=(
            "test: evaluate only the held-out 70%% split (original behavior). "
            "all_data: evaluate every sample in meta (normally all 440 COCO_two "
            "samples). Note that all_data overlaps the 30%% writer-calibration "
            "split and is therefore a full-dataset mechanistic diagnostic."
        ),
    )
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return sorted(
        {int(x.strip().upper().replace("L", "")) for x in str(text).split(",") if x.strip()}
    )


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_bundles(text: str) -> List[Tuple[int, ...]]:
    out = []
    seen = set()
    for raw in str(text).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        vals = tuple(sorted({int(x.strip().upper().replace("L", ""))
                             for x in raw.split("+") if x.strip()}))
        if vals and vals not in seen:
            seen.add(vals)
            out.append(vals)
    if not out:
        raise ValueError("No source bundles")
    return out


def bundle_name(bundle: Sequence[int]) -> str:
    return "+".join(f"L{x}" for x in bundle)


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    vals = np.asarray([float(x) for x in xs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def safe_sem(xs):
    vals = np.asarray([float(x) for x in xs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    return float(vals.std(ddof=1) / np.sqrt(vals.size))


def cosine_np(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v / n).astype(np.float32)


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def span_positions(span):
    return set(range(int(span[0]), int(span[1]) + 1))


def all_subsequence_positions(ids, pat):
    if not pat or len(pat) > len(ids):
        return []
    ans = []
    n = len(pat)
    for i in range(len(ids) - n + 1):
        if ids[i:i+n] == pat:
            ans.extend(range(i, i+n))
    return sorted(set(ans))


def find_text_positions(tokenizer, ids, text):
    ans = set()
    for variant in (text, " " + text):
        pat = tokenizer.encode(variant, add_special_tokens=False)
        ans.update(all_subsequence_positions(ids, pat))
    return sorted(ans)


def build_categories(model, processor, batch, ids, subject, reference):
    tokenizer = processor.tokenizer
    toks = [str(x) for x in tokenizer.convert_ids_to_tokens(ids)]

    try:
        sspan, rspan = base.locate_object_spans(tokenizer, ids, subject, reference)
        sub = span_positions(sspan)
        ref = span_positions(rspan)
    except Exception:
        sub = set(find_text_positions(tokenizer, ids, subject))
        ref = set(find_text_positions(tokenizer, ids, reference))

    try:
        visual = set(map(int, base.resolve_visual_indices(model, processor, batch, ids)))
    except Exception:
        visual = set()
        for p, tok in enumerate(toks):
            if "image_pad" in tok or "video_pad" in tok:
                visual.add(p)

    rel_pos = {}
    for word in ("left", "right", "above", "below", "on", "under"):
        rel_pos[word] = set(find_text_positions(tokenizer, ids, word))

    cats = []
    for p in range(len(ids)):
        if p == len(ids) - 1:
            cat = "last"
        elif p in sub:
            cat = "subject"
        elif p in ref:
            cat = "reference"
        else:
            hit = next((w for w, ps in rel_pos.items() if p in ps), None)
            if hit is not None:
                cat = f"relation_word:{hit}"
            elif p in visual:
                cat = "visual"
            else:
                cat = "other_text"
        cats.append(cat)
    return cats, toks


def broad_category(cat):
    if cat.startswith("relation_word:"):
        return "relation_words"
    if cat in ("visual", "subject", "reference"):
        return cat
    if cat == "last":
        return "last"
    return "other_text"


class Capture:
    def __init__(self, decoder_layers, layers, cut_layer=None, cpu=False):
        self.states = {}
        self.handles = []
        self.cut_layer = cut_layer
        self.cpu = cpu
        for L in layers:
            if cut_layer is not None and L == cut_layer:
                h = decoder_layers[L].register_forward_hook(self._cut(L))
            else:
                h = decoder_layers[L].register_forward_hook(self._keep(L))
            self.handles.append(h)

    def _cut(self, L):
        def hook(_m, _inp, out):
            x = traj.first_tensor(out)
            y = x.detach().clone().requires_grad_(True)
            self.states[L] = y
            return traj.replace_first_tensor(out, y)
        return hook

    def _keep(self, L):
        def hook(_m, _inp, out):
            x = traj.first_tensor(out)
            self.states[L] = x.detach().float().cpu() if self.cpu else x
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_cpu(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers, cpu=True)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing captured states: {missing}")
        return {L: cap.states[L].numpy().astype(np.float32) for L in layers}
    finally:
        cap.close()


def forward_graph(model, decoder_layers, batch, layers, cut):
    cap = Capture(decoder_layers, layers, cut_layer=cut, cpu=False)
    kw = dict(batch)
    kw["use_cache"] = False
    _ = model(**kw)
    missing = [L for L in layers if L not in cap.states]
    if missing:
        cap.close()
        raise RuntimeError(f"Missing graph states: {missing}")
    return cap


def learn_writers(train, q_by_sid, targets, mode):
    writers = {T: {} for T in targets}
    geometry = []
    for T in targets:
        means = {}
        for r in REL:
            xs = [q_by_sid[int(row["sid"])][T] for row in train if row["gt"] == r]
            if not xs:
                raise RuntimeError(f"No TRAIN examples for relation={r}, L{T}")
            means[r] = np.mean(np.stack(xs), axis=0).astype(np.float32)

        common = np.mean(np.stack([means[r] for r in REL]), axis=0).astype(np.float32)
        for r in REL:
            writers[T][r] = (
                means[r] - common if mode == "centered" else means[r]
            ).astype(np.float32)

        for r in REL:
            geometry.append({
                "target_layer": T,
                "relation": DISPLAY[r],
                "writer_norm": float(np.linalg.norm(writers[T][r])),
                "cos_left": cosine_np(writers[T][r], writers[T]["left"]),
                "cos_right": cosine_np(writers[T][r], writers[T]["right"]),
                "cos_on": cosine_np(writers[T][r], writers[T]["above"]),
                "cos_under": cosine_np(writers[T][r], writers[T]["below"]),
            })
    return writers, geometry


class TokenDeltaEditor:
    """Amplify precomputed clean Real-Gray deltas at selected source positions."""
    def __init__(self, decoder_layers, specs, alpha, prompt_len):
        self.handles = []
        self.applied = defaultdict(int)
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)

        for L, entries in specs.items():
            if not entries:
                continue
            self.handles.append(
                decoder_layers[L].register_forward_hook(self._hook(L, entries))
            )

    def _hook(self, L, entries):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            # Prefill only. Cached generation steps are length 1.
            if int(h.shape[1]) != self.prompt_len:
                return out
            y = h.float().clone()
            for pos, delta_np in entries:
                pos = int(pos)
                if 0 <= pos < y.shape[1]:
                    delta = torch.as_tensor(delta_np, device=y.device, dtype=torch.float32)
                    y[:, pos, :] = y[:, pos, :] + self.alpha * delta
                    self.applied[L] += 1
            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


class DirectWriterEditor:
    """Reference ceiling: directly add the learned writer at late last token."""
    def __init__(self, decoder_layers, target_layers, writers_for_relation, alpha, prompt_len):
        self.handles = []
        self.applied = defaultdict(int)
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)
        for T in target_layers:
            vec = np.asarray(writers_for_relation[T], np.float32)
            self.handles.append(
                decoder_layers[T].register_forward_hook(self._hook(T, vec))
            )

    def _hook(self, T, vec_np):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) != self.prompt_len:
                return out
            y = h.float().clone()
            v = torch.as_tensor(vec_np, device=y.device, dtype=torch.float32)
            y[:, -1, :] = y[:, -1, :] + self.alpha * v
            self.applied[T] += 1
            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


class TargetRecorder:
    """Record post-hook target-layer last states during the prefill generation pass."""
    def __init__(self, decoder_layers, target_layers, prompt_len):
        self.states = {}
        self.handles = []
        self.prompt_len = int(prompt_len)
        for T in target_layers:
            self.handles.append(
                decoder_layers[T].register_forward_hook(self._hook(T))
            )

    def _hook(self, T):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) == self.prompt_len and T not in self.states:
                self.states[T] = (
                    h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
                )
            return out
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def run_generation_with_hooks(
    model,
    processor,
    decoder_layers,
    batch,
    target_layers,
    writers_for_relation,
    max_new_tokens,
    token_specs=None,
    token_alpha=None,
    direct_alpha=None,
):
    prompt_len = int(batch["input_ids"].shape[1])
    editor = None
    direct = None
    recorder = None
    try:
        # Editors must be registered before recorder so recorder sees edited states.
        if token_specs is not None:
            editor = TokenDeltaEditor(
                decoder_layers, token_specs, float(token_alpha), prompt_len
            )
        if direct_alpha is not None:
            direct = DirectWriterEditor(
                decoder_layers, target_layers, writers_for_relation,
                float(direct_alpha), prompt_len
            )
        recorder = TargetRecorder(decoder_layers, target_layers, prompt_len)

        text = base.generate_text(
            model, processor, batch, max_new_tokens=max_new_tokens
        )
        pred = traj.normalize_relation(base, text)

        projections = {}
        cosines = {}
        for T in target_layers:
            if T not in recorder.states:
                projections[T] = float("nan")
                cosines[T] = float("nan")
                continue
            h = recorder.states[T]
            s = writers_for_relation[T]
            projections[T] = float(np.dot(h, normalize_np(s)))
            cosines[T] = cosine_np(h, s)

        return {
            "text": text,
            "prediction": pred,
            "projection_by_target": projections,
            "cosine_by_target": cosines,
            "mean_projection": safe_mean(projections.values()),
            "mean_cosine": safe_mean(cosines.values()),
            "edit_applied": (
                dict(editor.applied) if editor is not None
                else dict(direct.applied) if direct is not None
                else {}
            ),
        }
    finally:
        if recorder is not None:
            recorder.close()
        if direct is not None:
            direct.close()
        if editor is not None:
            editor.close()


def choose_positive(rows, k):
    vals = [r for r in rows if r["mediation"] > 0]
    vals.sort(key=lambda x: x["mediation"], reverse=True)
    return vals[:k]


def choose_negative(rows, k):
    vals = [r for r in rows if r["mediation"] < 0]
    vals.sort(key=lambda x: x["mediation"])  # most negative first
    return vals[:k]


def choose_random_matched(rows, positive_rows, rng):
    """Match broad token category distribution of positive selection."""
    selected_pos = {int(r["position"]) for r in positive_rows}
    by_cat = defaultdict(list)
    for r in rows:
        if int(r["position"]) not in selected_pos:
            by_cat[r["broad_category"]].append(r)

    chosen = []
    used = set()
    for ref in positive_rows:
        cat = ref["broad_category"]
        pool = [x for x in by_cat[cat] if int(x["position"]) not in used]
        if not pool:
            pool = [x for x in rows
                    if int(x["position"]) not in selected_pos
                    and int(x["position"]) not in used]
        if not pool:
            break
        x = rng.choice(pool)
        chosen.append(x)
        used.add(int(x["position"]))
    return chosen


def _global_random_matched(candidates, positive_rows, rng, k):
    """Category-match a global positive selection as closely as possible."""
    selected_keys = {(int(r["source_layer"]), int(r["position"])) for r in positive_rows}
    by_cat = defaultdict(list)
    for r in candidates:
        key = (int(r["source_layer"]), int(r["position"]))
        if key not in selected_keys:
            by_cat[r["broad_category"]].append(r)

    chosen = []
    used = set()
    for ref in positive_rows[:k]:
        cat = ref["broad_category"]
        pool = [
            x for x in by_cat[cat]
            if (int(x["source_layer"]), int(x["position"])) not in used
        ]
        if not pool:
            pool = [
                x for x in candidates
                if (int(x["source_layer"]), int(x["position"])) not in selected_keys
                and (int(x["source_layer"]), int(x["position"])) not in used
            ]
        if not pool:
            break
        x = rng.choice(pool)
        chosen.append(x)
        used.add((int(x["source_layer"]), int(x["position"])))
    return chosen


def make_specs(bundle, rows_by_layer, mode, k, sid, seed, strategy="per_layer"):
    """
    strategy:
      per_layer     : old behavior, K edits per source layer.
      global        : K total (layer,position) edits across the whole bundle.
      global_unique : K total edits, each token position used at most once;
                      for each position keep the strongest eligible source layer.
    """
    if strategy not in {"per_layer", "global", "global_unique"}:
        raise ValueError(strategy)

    selected = []

    if strategy == "per_layer":
        for L in bundle:
            rows = rows_by_layer.get(L, [])
            pos = choose_positive(rows, k)
            if mode == "positive":
                use = pos
            elif mode == "negative":
                use = choose_negative(rows, k)
            elif mode == "random":
                rng = random.Random(seed * 1000003 + int(sid) * 1009 + L * 97 + k)
                use = choose_random_matched(rows, pos, rng)
            else:
                raise ValueError(mode)

            for r in use:
                rr = dict(r)
                rr["source_layer"] = L
                selected.append(rr)

    else:
        candidates = []
        for L in bundle:
            for r in rows_by_layer.get(L, []):
                rr = dict(r)
                rr["source_layer"] = L
                candidates.append(rr)

        if mode == "positive":
            eligible = [r for r in candidates if r["mediation"] > 0]
            reverse = True
        elif mode == "negative":
            eligible = [r for r in candidates if r["mediation"] < 0]
            reverse = False
        elif mode == "random":
            eligible = candidates
            reverse = True
        else:
            raise ValueError(mode)

        if strategy == "global_unique":
            # Keep only the strongest eligible layer for each token position.
            best_by_pos = {}
            for r in eligible:
                pos = int(r["position"])
                if pos not in best_by_pos:
                    best_by_pos[pos] = r
                    continue
                cur = best_by_pos[pos]
                if mode == "negative":
                    if r["mediation"] < cur["mediation"]:
                        best_by_pos[pos] = r
                else:
                    if r["mediation"] > cur["mediation"]:
                        best_by_pos[pos] = r
            eligible = list(best_by_pos.values())

        if mode == "positive":
            eligible.sort(key=lambda x: x["mediation"], reverse=True)
            use = eligible[:k]
        elif mode == "negative":
            eligible.sort(key=lambda x: x["mediation"])
            use = eligible[:k]
        else:
            positive_all = [r for r in candidates if r["mediation"] > 0]
            if strategy == "global_unique":
                tmp = {}
                for r in positive_all:
                    pos = int(r["position"])
                    if pos not in tmp or r["mediation"] > tmp[pos]["mediation"]:
                        tmp[pos] = r
                positive_all = list(tmp.values())
            positive_all.sort(key=lambda x: x["mediation"], reverse=True)
            pos_ref = positive_all[:k]
            rng = random.Random(seed * 1000003 + int(sid) * 1009 + k * 97 + len(bundle))
            use = _global_random_matched(eligible, pos_ref, rng, k)

        selected.extend(use)

    # Build hook specs.
    specs = defaultdict(list)
    selected_export = []
    # Re-rank globally for interpretable output.
    if mode == "negative":
        selected = sorted(selected, key=lambda x: x["mediation"])
    else:
        selected = sorted(selected, key=lambda x: x["mediation"], reverse=True)

    for rank, r in enumerate(selected, 1):
        L = int(r["source_layer"])
        specs[L].append((int(r["position"]), r["delta_h"]))
        selected_export.append({
            "source_layer": L,
            "rank": rank,
            "position": int(r["position"]),
            "token": r["token"],
            "category": r["category"],
            "broad_category": r["broad_category"],
            "mediation": float(r["mediation"]),
            "delta_h_norm": float(r["delta_h_norm"]),
            "grad_norm": float(r["grad_norm"]),
        })

    return dict(specs), selected_export


def summarize(eval_rows, condition_rows):
    baseline = {int(r["sid"]): bool(r["baseline_correct"]) for r in eval_rows}
    baseline_pred = {int(r["sid"]): r["baseline_prediction"] for r in eval_rows}
    baseline_proj = {int(r["sid"]): float(r["baseline_mean_projection"]) for r in eval_rows}
    base_acc = safe_mean(float(x) for x in baseline.values())

    groups = defaultdict(list)
    for r in condition_rows:
        key = (
            r["condition"],
            r["source_bundle"],
            int(r["k"]),
            float(r["alpha"]),
        )
        groups[key].append(r)

    out = []
    for (condition, bundle, k, alpha), rows in sorted(groups.items()):
        by_sid = {int(r["sid"]): r for r in rows}
        correct = {
            sid: bool(by_sid[sid]["correct"]) if sid in by_sid else baseline[sid]
            for sid in baseline
        }
        pred = {
            sid: by_sid[sid]["prediction"] if sid in by_sid else baseline_pred[sid]
            for sid in baseline
        }
        acc = safe_mean(float(x) for x in correct.values())
        w2c = sum((not baseline[s]) and correct[s] for s in baseline)
        c2w = sum(baseline[s] and (not correct[s]) for s in baseline)

        proj_gains = []
        for sid, row in by_sid.items():
            proj_gains.append(
                float(row["mean_projection"]) - baseline_proj[sid]
            )

        out.append({
            "condition": condition,
            "source_bundle": bundle,
            "k": k,
            "alpha": alpha,
            "N": len(baseline),
            "baseline_acc": base_acc,
            "edited_acc": acc,
            "delta_acc": acc - base_acc,
            "W2C": int(w2c),
            "C2W": int(c2w),
            "net": int(w2c - c2w),
            "changed": int(sum(pred[s] != baseline_pred[s] for s in baseline)),
            "mean_writer_projection_gain": safe_mean(proj_gains),
            "mean_selected_tokens": safe_mean(
                float(r["n_selected_total"]) for r in rows
            ),
        })
    return out


def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    bundles = parse_bundles(a.source_bundles)
    all_sources = sorted(set(x for b in bundles for x in b))
    targets = parse_ints(a.target_layers)
    ks = parse_ints(a.ks)
    alphas = parse_floats(a.alphas)
    direct_alphas = parse_floats(a.direct_writer_alphas)
    conditions = [x.strip().lower() for x in a.conditions.split(",") if x.strip()]
    bad = [x for x in conditions if x not in {"positive", "random", "negative"}]
    if bad:
        raise ValueError(f"Unknown conditions: {bad}")
    selection_strategies = [
        x.strip().lower() for x in a.selection_strategies.split(",") if x.strip()
    ]
    bad_strategy = [
        x for x in selection_strategies
        if x not in {"per_layer", "global", "global_unique"}
    ]
    if bad_strategy:
        raise ValueError(f"Unknown selection strategies: {bad_strategy}")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
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
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    train, heldout_test = traj.stratified_split(meta, a.train_ratio, a.seed)

    if a.eval_scope == "all_data":
        test = list(meta)
    else:
        test = list(heldout_test)

    test = traj.stratified_cap(test, a.eval_max_samples, a.seed + 1)

    train_sids = {int(x["sid"]) for x in train}
    eval_sids = {int(x["sid"]) for x in test}
    calibration_eval_overlap = len(train_sids & eval_sids)

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
    try:
        print(f"Loading {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id, trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model, processor)

        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        n_layers = len(decoder_layers)

        for L in all_sources + targets:
            if not (0 <= L < n_layers):
                raise ValueError(f"L{L} invalid; model has L0...L{n_layers-1}")
        for S in all_sources:
            if S >= max(targets):
                raise ValueError(f"Source L{S} is not earlier than all useful targets")

        cut = min(all_sources)
        graph_layers = sorted(set(all_sources + targets))
        device = torch.device(a.device)

        print("=" * 132)
        print("ORACLE MIDDLE-TOKEN AMPLIFICATION -> LATE LEARNED WRITER -> ACTUAL GENERATION")
        print("=" * 132)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path} n_layers={n_layers}")
        print(
            f"writer calibration N={len(train)} | eval scope={a.eval_scope} "
            f"| eval N={len(test)} | calibration/eval overlap={calibration_eval_overlap}"
        )
        if a.eval_scope == "all_data":
            print(
                "NOTE: all_data is a full-dataset mechanistic diagnostic; "
                "writer calibration samples are included in evaluation."
            )
        print(f"source bundles={[bundle_name(b) for b in bundles]}")
        print(f"target writer layers={targets}")
        print(f"K={ks} | alpha={alphas} | conditions={conditions} | selection={selection_strategies}")
        print("source last token EXCLUDED")
        print("selection is ORACLE: correct relation chooses learned s_r")
        print()

        # ------------------------------------------------------------
        # 1. Learn old late Image-Gray writers.
        # ------------------------------------------------------------
        q_by_sid = {}
        for m in tqdm(train, desc="TRAIN late writers"):
            sid = int(m["sid"])
            real = gray = rb = gb = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor, image=real,
                    question_text=m["question_text"], device=device
                )
                gb = base.make_question_batch(
                    processor=processor, image=gray,
                    question_text=m["question_text"], device=device
                )
                hr = capture_cpu(model, decoder_layers, rb, targets)
                hg = capture_cpu(model, decoder_layers, gb, targets)
                q_by_sid[sid] = {
                    T: (hr[T][0, -1] - hg[T][0, -1]).astype(np.float32)
                    for T in targets
                }
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()

        writers, writer_geom = learn_writers(
            train, q_by_sid, targets, a.writer_mode
        )
        write_csv(outdir / "writer_geometry.csv", writer_geom)
        np.savez_compressed(
            outdir / "learned_writers.npz",
            **{
                f"L{T}_{DISPLAY[r]}": writers[T][r]
                for T in targets for r in REL
            }
        )

        # ------------------------------------------------------------
        # 2. Held-out: rank concrete middle tokens, then causally amplify.
        # ------------------------------------------------------------
        eval_rows = []
        med_rows_all = []
        selection_rows = []
        condition_rows = []

        for m in tqdm(test, desc="TEST select + generate"):
            sid = int(m["sid"])
            gt = m["gt"]
            writers_r = {T: writers[T][gt] for T in targets}

            real = gray = rb = gb = cap = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor, image=real,
                    question_text=m["question_text"], device=device
                )
                gb = base.make_question_batch(
                    processor=processor, image=gray,
                    question_text=m["question_text"], device=device
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats, toks = build_categories(
                    model, processor, rb, ids, m["subject"], m["reference"]
                )

                # Clean baseline generation AND clean target writer projections.
                clean = run_generation_with_hooks(
                    model, processor, decoder_layers, rb,
                    targets, writers_r, a.max_new_tokens
                )
                clean_pred = clean["prediction"]
                clean_correct = (clean_pred == gt)

                eval_rows.append({
                    "sid": sid,
                    "gt": DISPLAY[gt],
                    "baseline_prediction": DISPLAY.get(clean_pred, clean_pred),
                    "baseline_correct": clean_correct,
                    "baseline_text": clean["text"],
                    "baseline_mean_projection": clean["mean_projection"],
                    **{
                        f"baseline_proj_L{T}": clean["projection_by_target"][T]
                        for T in targets
                    },
                })

                # Gray source activations.
                hgray = capture_cpu(model, decoder_layers, gb, all_sources)

                # Clean real graph for the joint target objective.
                with torch.enable_grad():
                    cap = forward_graph(
                        model, decoder_layers, rb, graph_layers, cut
                    )

                    objective_terms = []
                    for T in targets:
                        s_hat = torch.as_tensor(
                            normalize_np(writers_r[T]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        objective_terms.append(
                            torch.dot(cap.states[T][0, -1].float(), s_hat)
                        )
                    objective = torch.stack(objective_terms).sum()

                    grads = torch.autograd.grad(
                        objective,
                        [cap.states[S] for S in all_sources],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                    rows_by_layer = {}
                    for S, g in zip(all_sources, grads):
                        Hreal = (
                            cap.states[S][0].detach().float().cpu().numpy().astype(np.float32)
                        )
                        Hgray = hgray[S][0].astype(np.float32)
                        G = g[0].detach().float().cpu().numpy().astype(np.float32)
                        npos = min(
                            len(ids), len(cats), len(toks),
                            Hreal.shape[0], Hgray.shape[0], G.shape[0]
                        )

                        rowsS = []
                        # EXCLUDE source last token.
                        for pos in range(max(0, npos - 1)):
                            delta = (Hreal[pos] - Hgray[pos]).astype(np.float32)
                            grad = G[pos]
                            med = float(np.dot(delta, grad))
                            row = {
                                "sid": sid,
                                "relation": DISPLAY[gt],
                                "source_layer": S,
                                "position": pos,
                                "token_id": int(ids[pos]),
                                "token": str(toks[pos]).replace("\n", "\\n"),
                                "category": cats[pos],
                                "broad_category": broad_category(cats[pos]),
                                "mediation": med,
                                "abs_mediation": abs(med),
                                "delta_h_norm": float(np.linalg.norm(delta)),
                                "grad_norm": float(np.linalg.norm(grad)),
                                # Keep in memory only; removed from CSV later.
                                "delta_h": delta,
                            }
                            rowsS.append(row)

                        rowsS.sort(key=lambda x: x["mediation"], reverse=True)
                        for rank, r0 in enumerate(rowsS, 1):
                            export = {k: v for k, v in r0.items() if k != "delta_h"}
                            export["positive_rank"] = rank
                            med_rows_all.append(export)
                        rows_by_layer[S] = rowsS

                cap.close()
                cap = None

                # Direct late writer reference/ceiling.
                for da in direct_alphas:
                    direct = run_generation_with_hooks(
                        model, processor, decoder_layers, rb,
                        targets, writers_r, a.max_new_tokens,
                        direct_alpha=da,
                    )
                    condition_rows.append({
                        "sid": sid,
                        "gt": DISPLAY[gt],
                        "condition": "direct_late_writer",
                        "selection_strategy": "direct",
                        "source_bundle": "late:" + "+".join(f"L{T}" for T in targets),
                        "k": 0,
                        "alpha": da,
                        "prediction": DISPLAY.get(direct["prediction"], direct["prediction"]),
                        "correct": direct["prediction"] == gt,
                        "text": direct["text"],
                        "mean_projection": direct["mean_projection"],
                        "n_selected_total": 0,
                        **{
                            f"proj_L{T}": direct["projection_by_target"][T]
                            for T in targets
                        },
                    })

                # Expanded multi-layer middle-token amplification search.
                for bundle in bundles:
                    for strategy in selection_strategies:
                        labeled_bundle = f"{bundle_name(bundle)}[{strategy}]"
                        for k in ks:
                            # For global/global_unique, K is TOTAL edits in the bundle.
                            # For per_layer, K retains the old meaning: K per source layer.
                            for mode in conditions:
                                token_specs, selected = make_specs(
                                    bundle, rows_by_layer, mode, k, sid, a.seed,
                                    strategy=strategy,
                                )
                                if not selected:
                                    continue

                                for sel in selected:
                                    selection_rows.append({
                                        "sid": sid,
                                        "relation": DISPLAY[gt],
                                        "condition": mode,
                                        "selection_strategy": strategy,
                                        "source_bundle": labeled_bundle,
                                        "k": k,
                                        **sel,
                                    })

                                for alpha in alphas:
                                    edited = run_generation_with_hooks(
                                        model, processor, decoder_layers, rb,
                                        targets, writers_r, a.max_new_tokens,
                                        token_specs=token_specs,
                                        token_alpha=alpha,
                                    )

                                    condition_rows.append({
                                        "sid": sid,
                                        "gt": DISPLAY[gt],
                                        "condition": f"middle_{mode}",
                                        "selection_strategy": strategy,
                                        "source_bundle": labeled_bundle,
                                        "k": k,
                                        "alpha": alpha,
                                        "prediction": DISPLAY.get(
                                            edited["prediction"], edited["prediction"]
                                        ),
                                        "correct": edited["prediction"] == gt,
                                        "text": edited["text"],
                                        "mean_projection": edited["mean_projection"],
                                        "n_selected_total": len(selected),
                                        **{
                                            f"proj_L{T}": edited["projection_by_target"][T]
                                            for T in targets
                                        },
                                    })

            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                del rb, gb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        write_csv(outdir / "baseline.csv", eval_rows)
        write_csv(outdir / "mediation_tokens.csv", med_rows_all)
        write_csv(outdir / "selected_tokens.csv", selection_rows)
        write_csv(outdir / "generation_conditions.csv", condition_rows)

        # Direct writer rows need a compatible summary key.  They already have
        # source_bundle/k/alpha, so the same summary function works.
        summary = summarize(eval_rows, condition_rows)
        write_csv(outdir / "summary.csv", summary)

        (outdir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "model": a.model,
                    "eval_scope": a.eval_scope,
                    "full_meta_N": len(meta),
                    "writer_calibration_N": len(train),
                    "eval_N": len(test),
                    "calibration_eval_overlap": calibration_eval_overlap,
                    "source_bundles": [bundle_name(b) for b in bundles],
                    "target_layers": targets,
                    "ks": ks,
                    "alphas": alphas,
                    "conditions": conditions,
                    "selection_strategies": selection_strategies,
                    "writer_mode": a.writer_mode,
                    "oracle_relation_used_for_writer_target": True,
                    "sample_specific_token_selection": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------------
        # Console report.
        # ------------------------------------------------------------
        print("\n" + "=" * 146)
        print("EXPANDED MULTI-LAYER MIDDLE TOKEN AMPLIFICATION -> ACTUAL GENERATION")
        print("=" * 146)
        baseline_acc = safe_mean(float(r["baseline_correct"]) for r in eval_rows)
        print(f"Baseline: N={len(eval_rows)} acc={baseline_acc:.4f}")

        direct_rows = [r for r in summary if r["condition"] == "direct_late_writer"]
        middle_rows = [r for r in summary if r["condition"] != "direct_late_writer"]
        middle_rows = sorted(
            middle_rows,
            key=lambda r: (
                float(r["edited_acc"]),
                int(r["net"]),
                float(r["mean_writer_projection_gain"]),
            ),
            reverse=True,
        )

        print("\nDirect late-writer reference:")
        for r in direct_rows:
            print(
                f"  acc={float(r['edited_acc']):.4f} "
                f"gain={float(r['delta_acc']):+.4f} "
                f"W2C/C2W={int(r['W2C'])}/{int(r['C2W'])} "
                f"dWriterProj={float(r['mean_writer_projection_gain']):+.4f}"
            )

        print("\nTop 30 middle-token configurations:")
        print(
            f"{'bundle[strategy]':<48s} {'K':>4s} {'alpha':>7s} "
            f"{'acc':>8s} {'gain':>8s} {'W2C':>5s} {'C2W':>5s} {'net':>5s} "
            f"{'nEdit':>7s} {'dWriterProj':>12s}"
        )
        print("-" * 146)
        for r in middle_rows[:30]:
            print(
                f"{r['source_bundle']:<48s} "
                f"{int(r['k']):>4d} "
                f"{float(r['alpha']):>7.3f} "
                f"{float(r['edited_acc']):>8.4f} "
                f"{float(r['delta_acc']):>+8.4f} "
                f"{int(r['W2C']):>5d} "
                f"{int(r['C2W']):>5d} "
                f"{int(r['net']):>5d} "
                f"{float(r['mean_selected_tokens']):>7.1f} "
                f"{float(r['mean_writer_projection_gain']):>+12.4f}"
            )

        best_path = outdir / "best_configs.csv"
        write_csv(best_path, middle_rows)
        print(f"\nFull ranked middle configurations: {best_path}")
        # Also print globally strongest concrete non-last tokens.
        print("\nTop concrete positive mediation tokens (across held-out samples):")
        strongest = sorted(
            med_rows_all, key=lambda x: x["mediation"], reverse=True
        )[:30]
        for r in strongest:
            print(
                f"sid={r['sid']:>4} L{r['source_layer']:02d} "
                f"pos={r['position']:>4} "
                f"M={r['mediation']:+.4f} "
                f"{r['broad_category']:<14s} token={r['token']!r}"
            )

        print("\nPrimary evidence to look for:")
        print("  1) middle_positive raises mean_writer_projection")
        print("  2) generation W2C rises with K/alpha")
        print("  3) positive >> random")
        print("  4) negative is neutral or harmful")
        print("  5) middle_positive approaches direct_late_writer accuracy")
        print("\nSaved:", outdir)

        (outdir / "metadata.json").write_text(
            json.dumps({
                "model_alias": a.model,
                "repo_id": spec.repo_id,
                "writer_mode": a.writer_mode,
                "writer_definition": (
                    "TRAIN mean(Real-Gray late last | relation), "
                    "common-mean centered by default"
                ),
                "selection_objective": (
                    "sum_T <h_real_T,last, normalized learned s_r^T>"
                ),
                "mediation": (
                    "<h_real_source_token-h_gray_source_token, "
                    "gradient of selection objective wrt source token>"
                ),
                "edit": (
                    "h_real_source_token += alpha * "
                    "(h_real_source_token-h_gray_source_token)"
                ),
                "oracle": True,
                "source_last_excluded": True,
                "source_bundles": [list(b) for b in bundles],
                "target_layers": targets,
                "ks": ks,
                "alphas": alphas,
                "direct_writer_alphas": direct_alphas,
                "conditions": conditions,
                "selection_strategies": selection_strategies,
                "train_N": len(train),
                "test_N": len(test),
                "seed": a.seed,
            }, indent=2),
            encoding="utf-8",
        )

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
