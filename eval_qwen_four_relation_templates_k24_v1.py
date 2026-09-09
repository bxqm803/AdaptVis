#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four fixed relation-template counterfactual intervention.

Question
--------
After discovering relation-specific K24 middle-token patterns offline, do those
patterns themselves behave like directional causal templates?

For the SAME unseen sample, run four counterfactual edits:

    T_left, T_right, T_on, T_under

and ask whether applying T_r makes the model answer r.

This is NOT an inference/repair method and uses NO selector at evaluation time.
The four templates are all fixed before seeing the evaluation sample.

Offline template construction
-----------------------------
From an existing oracle writer-guided selected_tokens.csv:

  - filter one bundle, e.g. L22+L24+L26[global_unique]
  - filter discovery K=24 positive selections
  - canonicalize positions across prompts
  - separately for left/right/on/under, rank canonical (layer,slot) candidates
    by sample-level selection frequency, breaking ties by mean mediation
  - take top --template-k unique candidates for each relation

Evaluation
----------
For each unseen sample:
  1) get Real and Gray hidden states at all template source layers
  2) resolve each fixed canonical template candidate to a current token position
  3) for EACH of the four templates independently:
         h'[L,p] = h_real[L,p] + alpha * (h_real[L,p]-h_gray[L,p])
  4) generate normally

No GT relation, late writer, gradient, centroid, probe, relation selector, or
answer logits are used to CHOOSE the intervention during evaluation.

GT is used only after generation for analysis.

Key outputs
-----------
template_follow_summary.csv
    P(output == template direction | apply template direction)

template_output_matrix.csv
    4x4 template -> generated-relation distribution

matched_mismatched_summary.csv
    matched template (template==GT) repair vs mismatched-template induction

same_sample_counterfactuals.csv
    all four edited outputs for each sample, side by side

template_overlap.csv
    pairwise overlap/Jaccard of the four K24 templates

A strong directional-template result would look like:
    apply T_left  -> output tends to left
    apply T_right -> output tends to right
    apply T_on    -> output tends to on
    apply T_under -> output tends to under

even on the exact same underlying image/question.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -u eval_qwen_four_relation_templates_k24_v1.py \
  --model qwen-3b \
  --selected-csv output/qwen3b_coco_minimal_token_search/selected_tokens.csv \
  --discovery-bundle "L22+L24+L26[global_unique]" \
  --discovery-k 24 \
  --template-k 24 \
  --alphas 0.5,0.75,1.0 \
  --eval-scope unseen \
  --eval-max-samples 80 \
  --output-dir output/qwen3b_coco_four_templates_k24_unseen80 \
  --overwrite
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL_INTERNAL = ("left", "right", "above", "below")
REL = ("left", "right", "on", "under")
DISPLAY = {"left": "left", "right": "right", "above": "on", "below": "under"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])

    p.add_argument("--selected-csv", required=True)
    p.add_argument("--discovery-bundle", default="L22+L24+L26[global_unique]")
    p.add_argument("--discovery-k", type=int, default=24)
    p.add_argument("--discovery-condition", default="positive")
    p.add_argument("--template-k", type=int, default=24)
    p.add_argument("--candidate-min-rate", type=float, default=0.0)
    p.add_argument("--visual-bins", type=int, default=16)

    p.add_argument(
        "--resolve-policy",
        default="center",
        choices=["center", "all_normalized"],
        help=(
            "center: each canonical template unit edits exactly one fixed center "
            "token position; all_normalized: edit all positions in the unit with "
            "1/sqrt(n) weight."
        ),
    )

    p.add_argument("--alphas", default="0.5,0.75,1.0")
    p.add_argument(
        "--eval-scope",
        default="unseen",
        choices=["unseen", "selected", "all_test"],
    )
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=80)

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


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def canon_rel(x):
    x = str(x).strip().lower()
    if x == "above":
        return "on"
    if x == "below":
        return "under"
    return x


def safe_mean(xs):
    vals = np.asarray([float(x) for x in xs], np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_gray_image(real_image, value):
    v = max(0, min(255, int(value)))
    return Image.new("RGB", real_image.size, (v, v, v))


def stratified_take(rows, n, seed):
    if not n or n <= 0 or n >= len(rows):
        return list(rows)
    return traj.stratified_cap(rows, n, seed)


# ----------------------------------------------------------------------
# Canonical mapping
# ----------------------------------------------------------------------

def visual_positions_from_tokens(tokens):
    return {
        i for i, tok in enumerate(tokens)
        if "image_pad" in str(tok) or "video_pad" in str(tok)
    }


def object_positions(tokenizer, ids, subject, reference):
    sspan, rspan = base.locate_object_spans(tokenizer, ids, subject, reference)
    spos = set(range(int(sspan[0]), int(sspan[1]) + 1))
    rpos = set(range(int(rspan[0]), int(rspan[1]) + 1))
    return spos, rpos


def build_canonical_mapping(tokens, spos, rpos, vpos, visual_bins):
    mapping = {}
    n = len(tokens)
    c = 0

    visual_sorted = sorted(vpos)
    visual_ord = {p: j for j, p in enumerate(visual_sorted)}
    nvis = len(visual_sorted)

    i = 0
    while i < n:
        if i in vpos:
            block = []
            while i < n and i in vpos:
                block.append(i)
                i += 1
            for p in block:
                j = visual_ord[p]
                if nvis <= 1:
                    b = 0
                else:
                    frac = j / (nvis - 1)
                    b = min(visual_bins - 1, int(frac * visual_bins))
                mapping[p] = {
                    "kind": "visual",
                    "key": f"VISBIN_{b:02d}",
                    "canonical_slot": f"C{c:03d}:<VISUAL_BLOCK>",
                }
            c += 1
            continue

        if i in spos:
            block = []
            while i < n and i in spos:
                block.append(i)
                i += 1
            label = f"C{c:03d}:<SUBJ>"
            for p in block:
                mapping[p] = {"kind": "text", "key": label, "canonical_slot": label}
            c += 1
            continue

        if i in rpos:
            block = []
            while i < n and i in rpos:
                block.append(i)
                i += 1
            label = f"C{c:03d}:<REF>"
            for p in block:
                mapping[p] = {"kind": "text", "key": label, "canonical_slot": label}
            c += 1
            continue

        tok = str(tokens[i]).replace("\n", "\\n")
        label = f"C{c:03d}:{tok}"
        mapping[i] = {"kind": "text", "key": label, "canonical_slot": label}
        c += 1
        i += 1

    return mapping


def candidate_id(layer, kind, key):
    if kind == "visual":
        return f"L{int(layer)}::<VISUAL>::{key}"
    return f"L{int(layer)}::{key}"


# ----------------------------------------------------------------------
# Offline relation-template construction
# ----------------------------------------------------------------------

def canonicalize_discovery(sel, prompts, rec_by_sid, processor, visual_bins):
    tokenizer = processor.tokenizer
    maps = {}

    for sid in tqdm(sorted(set(sel["sid"].astype(int))), desc="Canonicalize discovery"):
        p = prompts[sid]
        real = batch = None
        try:
            real = base.record_image(rec_by_sid[sid])
            if hasattr(real, "convert"):
                real = real.convert("RGB")
            batch = base.make_question_batch(
                processor=processor,
                image=real,
                question_text=str(p["question_text"]),
                device=torch.device("cpu"),
            )
            ids = batch["input_ids"][0].detach().cpu().tolist()
            toks = [str(x) for x in tokenizer.convert_ids_to_tokens(ids)]
            spos, rpos = object_positions(
                tokenizer, ids, str(p["subject"]), str(p["reference"])
            )
            vpos = visual_positions_from_tokens(toks)
            maps[sid] = build_canonical_mapping(
                toks, spos, rpos, vpos, visual_bins
            )
        finally:
            if real is not None:
                with contextlib.suppress(Exception):
                    real.close()
            del batch

    rows = []
    for r in sel.to_dict("records"):
        sid = int(r["sid"])
        pos = int(r["position"])
        info = maps.get(sid, {}).get(pos)
        if info is None:
            continue
        rows.append({
            **r,
            "candidate_kind": info["kind"],
            "candidate_key": info["key"],
            "candidate": candidate_id(
                int(r["source_layer"]), info["kind"], info["key"]
            ),
        })
    return pd.DataFrame(rows)


def build_templates(ann, template_k, min_rate):
    all_rankings = []
    templates = {}

    for rel in REL:
        rr = ann[ann["relation"] == rel].copy()
        denom = int(rr["sid"].nunique())
        if denom == 0:
            raise RuntimeError(f"No discovery samples for relation={rel}")

        rr = rr.sort_values("mediation", ascending=False).drop_duplicates(
            ["sid", "candidate"], keep="first"
        )

        agg = (
            rr.groupby(
                ["candidate", "source_layer", "candidate_kind", "candidate_key"],
                as_index=False,
            )
            .agg(
                n_samples_selected=("sid", "nunique"),
                mean_mediation=("mediation", "mean"),
                median_mediation=("mediation", "median"),
                max_mediation=("mediation", "max"),
            )
        )
        agg["relation"] = rel
        agg["discovery_N"] = denom
        agg["selection_rate"] = agg["n_samples_selected"] / denom
        agg = agg[agg["selection_rate"] >= float(min_rate)]
        agg = agg.sort_values(
            ["selection_rate", "mean_mediation", "max_mediation"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        agg["rank"] = np.arange(1, len(agg) + 1)
        agg["in_template"] = agg["rank"] <= int(template_k)

        all_rankings.append(agg)
        templates[rel] = agg.head(int(template_k)).copy()

    return templates, pd.concat(all_rankings, ignore_index=True)


def template_overlap_rows(templates):
    rows = []
    for i, a in enumerate(REL):
        A = set(templates[a]["candidate"].astype(str))
        for b in REL[i + 1:]:
            B = set(templates[b]["candidate"].astype(str))
            inter = A & B
            union = A | B
            rows.append({
                "template_a": a,
                "template_b": b,
                "n_a": len(A),
                "n_b": len(B),
                "intersection": len(inter),
                "union": len(union),
                "jaccard": len(inter) / len(union) if union else float("nan"),
                "shared_candidates": " | ".join(sorted(inter)),
            })
    return rows


# ----------------------------------------------------------------------
# Hidden state capture / fixed-template editing
# ----------------------------------------------------------------------

class Capture:
    def __init__(self, decoder_layers, layers):
        self.states = {}
        self.handles = []
        for L in layers:
            self.handles.append(
                decoder_layers[L].register_forward_hook(self._hook(L))
            )

    def _hook(self, L):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            self.states[L] = h.detach().float().cpu()
            return out
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()


@torch.inference_mode()
def capture_states(model, decoder_layers, batch, layers):
    cap = Capture(decoder_layers, layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        _ = model(**kw)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Missing source layers: {missing}")
        return {
            L: cap.states[L][0].numpy().astype(np.float32)
            for L in layers
        }
    finally:
        cap.close()


def resolve_template_specs(
    template_df,
    mapping,
    hreal,
    hgray,
    resolve_policy,
):
    specs = defaultdict(list)
    resolved_rows = []

    for row in template_df.to_dict("records"):
        L = int(row["source_layer"])
        kind = str(row["candidate_kind"])
        key = str(row["candidate_key"])

        positions = []
        for p, info in mapping.items():
            if info["kind"] == kind and str(info["key"]) == key:
                positions.append(int(p))

        positions = sorted(positions)
        if not positions:
            continue

        if resolve_policy == "center":
            chosen = [positions[len(positions) // 2]]
            per_pos_weight = 1.0
        else:
            chosen = positions
            per_pos_weight = 1.0 / math.sqrt(len(chosen))

        for p in chosen:
            if p >= hreal[L].shape[0] or p >= hgray[L].shape[0]:
                continue
            delta = (hreal[L][p] - hgray[L][p]).astype(np.float32)
            specs[L].append((p, delta, per_pos_weight))

        resolved_rows.append({
            "candidate": row["candidate"],
            "source_layer": L,
            "candidate_kind": kind,
            "candidate_key": key,
            "n_positions_available": len(positions),
            "positions_used": ",".join(map(str, chosen)),
            "per_position_weight": per_pos_weight,
        })

    return dict(specs), resolved_rows


class FixedTemplateEditor:
    def __init__(self, decoder_layers, specs, alpha, prompt_len):
        self.handles = []
        self.applied = defaultdict(int)
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)

        for L, entries in specs.items():
            if entries:
                self.handles.append(
                    decoder_layers[L].register_forward_hook(
                        self._hook(L, entries)
                    )
                )

    def _hook(self, L, entries):
        def hook(_m, _inp, out):
            h = traj.first_tensor(out)
            if int(h.shape[1]) != self.prompt_len:
                return out

            y = h.float().clone()
            for p, delta_np, weight in entries:
                if 0 <= p < y.shape[1]:
                    d = torch.as_tensor(
                        delta_np, device=y.device, dtype=torch.float32
                    )
                    y[:, p, :] = y[:, p, :] + self.alpha * float(weight) * d
                    self.applied[L] += 1
            return traj.replace_first_tensor(out, y.to(h.dtype))
        return hook

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def generate_edit(
    model,
    processor,
    decoder_layers,
    batch,
    specs,
    alpha,
    max_new_tokens,
):
    editor = FixedTemplateEditor(
        decoder_layers, specs, alpha, int(batch["input_ids"].shape[1])
    )
    try:
        text = base.generate_text(
            model, processor, batch, max_new_tokens=max_new_tokens
        )
        pred = traj.normalize_relation(base, text)
        return text, pred, dict(editor.applied)
    finally:
        editor.close()


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def make_summaries(baseline_rows, condition_rows):
    base = {int(r["sid"]): r for r in baseline_rows}
    summary = []

    groups = defaultdict(list)
    for r in condition_rows:
        groups[(r["template"], float(r["alpha"]))].append(r)

    for (template, alpha), rows in groups.items():
        N = len(rows)
        follow = sum(r["prediction"] == template for r in rows)
        changed_to_template = sum(
            base[int(r["sid"])]["baseline_prediction"] != template
            and r["prediction"] == template
            for r in rows
        )
        baseline_not_template = sum(
            base[int(r["sid"])]["baseline_prediction"] != template
            for r in rows
        )

        matched = [r for r in rows if r["gt"] == template]
        mismatched = [r for r in rows if r["gt"] != template]

        matched_acc = safe_mean(r["correct"] for r in matched) if matched else float("nan")
        matched_base = safe_mean(
            base[int(r["sid"])]["baseline_correct"] for r in matched
        ) if matched else float("nan")

        matched_w2c = sum(
            (not base[int(r["sid"])]["baseline_correct"]) and r["correct"]
            for r in matched
        )
        matched_c2w = sum(
            base[int(r["sid"])]["baseline_correct"] and (not r["correct"])
            for r in matched
        )

        mismatch_induce = sum(
            r["prediction"] == template for r in mismatched
        )
        mismatch_correct_to_template = sum(
            base[int(r["sid"])]["baseline_correct"]
            and r["prediction"] == template
            for r in mismatched
        )

        summary.append({
            "template": template,
            "alpha": alpha,
            "N": N,
            "target_follow_rate": follow / N if N else float("nan"),
            "baseline_not_target_N": baseline_not_template,
            "new_target_induction_rate": (
                changed_to_template / baseline_not_template
                if baseline_not_template else float("nan")
            ),
            "matched_N": len(matched),
            "matched_baseline_acc": matched_base,
            "matched_edited_acc": matched_acc,
            "matched_gain": matched_acc - matched_base,
            "matched_W2C": matched_w2c,
            "matched_C2W": matched_c2w,
            "mismatched_N": len(mismatched),
            "mismatched_target_follow_rate": (
                mismatch_induce / len(mismatched)
                if mismatched else float("nan")
            ),
            "mismatched_correct_to_template_N": mismatch_correct_to_template,
        })

    return sorted(
        summary,
        key=lambda x: (x["alpha"], REL.index(x["template"]))
    )


def output_matrix(condition_rows):
    rows = []
    for (template, alpha), g in pd.DataFrame(condition_rows).groupby(
        ["template", "alpha"]
    ):
        N = len(g)
        for out_rel in REL + ("other",):
            n = int((g["prediction"] == out_rel).sum())
            rows.append({
                "template": template,
                "alpha": float(alpha),
                "output_relation": out_rel,
                "N": N,
                "count": n,
                "rate": n / N if N else float("nan"),
            })
    return rows


def same_sample_rows(baseline_rows, condition_rows):
    base = {int(r["sid"]): r for r in baseline_rows}
    grouped = defaultdict(dict)

    for r in condition_rows:
        grouped[(int(r["sid"]), float(r["alpha"]))][r["template"]] = r

    rows = []
    for (sid, alpha), d in grouped.items():
        if not all(rel in d for rel in REL):
            continue

        outs = {rel: d[rel]["prediction"] for rel in REL}
        exact4 = all(outs[rel] == rel for rel in REL)
        n_distinct = len(set(outs.values()))

        rows.append({
            "sid": sid,
            "gt": base[sid]["gt"],
            "alpha": alpha,
            "baseline_prediction": base[sid]["baseline_prediction"],
            "left_template_output": outs["left"],
            "right_template_output": outs["right"],
            "on_template_output": outs["on"],
            "under_template_output": outs["under"],
            "n_distinct_template_outputs": n_distinct,
            "exact_four_way_controllable": exact4,
            "n_templates_followed": sum(outs[r] == r for r in REL),
        })

    return rows


def main():

    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    alphas = parse_floats(a.alphas)

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
        if gt not in REL_INTERNAL:
            continue
        meta.append({
            "sid": sid,
            "gt": gt,
            "subject": str(p["subject"]),
            "reference": str(p["reference"]),
            "question_text": str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta, a.max_samples, a.seed)
    _train, test = traj.stratified_split(meta, a.train_ratio, a.seed)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]

    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )

    # Offline template discovery.
    sel = pd.read_csv(a.selected_csv)
    sel["relation"] = sel["relation"].map(canon_rel)
    sel["k"] = pd.to_numeric(sel["k"], errors="raise").astype(int)
    sel["source_layer"] = pd.to_numeric(
        sel["source_layer"], errors="raise"
    ).astype(int)
    sel["position"] = pd.to_numeric(sel["position"], errors="raise").astype(int)
    sel["mediation"] = pd.to_numeric(sel["mediation"], errors="coerce")

    sel = sel[
        (sel["condition"].astype(str) == str(a.discovery_condition))
        & (sel["source_bundle"].astype(str) == str(a.discovery_bundle))
        & (sel["k"] == int(a.discovery_k))
        & sel["relation"].isin(REL)
    ].copy()

    if not len(sel):
        raise RuntimeError("No discovery rows after selected_tokens.csv filtering")

    sel = sel.sort_values("mediation", ascending=False).drop_duplicates(
        ["sid", "relation", "source_layer", "position"], keep="first"
    )
    discovery_sids = set(sel["sid"].astype(int))

    ann = canonicalize_discovery(
        sel, prompts, rec_by_sid, processor, a.visual_bins
    )
    ann.to_csv(outdir / "discovery_canonical_rows.csv", index=False)

    templates, rankings = build_templates(
        ann, a.template_k, a.candidate_min_rate
    )
    rankings.to_csv(outdir / "relation_template_rankings.csv", index=False)

    template_rows = []
    for rel in REL:
        for r in templates[rel].to_dict("records"):
            template_rows.append({
                "template": rel,
                **r,
            })
    pd.DataFrame(template_rows).to_csv(
        outdir / "relation_templates.csv", index=False
    )

    overlaps = template_overlap_rows(templates)
    write_csv(outdir / "template_overlap.csv", overlaps)

    # Eval split.
    if a.eval_scope == "unseen":
        pool = [r for r in test if int(r["sid"]) not in discovery_sids]
    elif a.eval_scope == "selected":
        pool = [r for r in test if int(r["sid"]) in discovery_sids]
    else:
        pool = list(test)

    eval_set = stratified_take(pool, a.eval_max_samples, a.seed + 101)
    if not eval_set:
        raise RuntimeError("Evaluation set is empty")

    source_layers = sorted({
        int(r["source_layer"])
        for rel in REL
        for r in templates[rel].to_dict("records")
    })

    model_cls = getattr(transformers, spec.model_class)
    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    print("=" * 150)
    print("FOUR FIXED K24 RELATION TEMPLATES — SAME-SAMPLE COUNTERFACTUAL TEST")
    print("=" * 150)
    print(f"model={a.model}")
    print(f"discovery bundle={a.discovery_bundle} K={a.discovery_k}")
    print(f"template K={a.template_k} canonical candidates per relation")
    print(f"resolve_policy={a.resolve_policy}")
    print(f"source layers={source_layers}")
    print(f"eval_scope={a.eval_scope} N={len(eval_set)}")
    print(f"discovery/eval overlap={len(discovery_sids & set(int(x['sid']) for x in eval_set))}")
    print(f"alphas={alphas}")
    print()
    print("Template overlap:")
    for r in overlaps:
        print(
            f"  {r['template_a']:>5s} vs {r['template_b']:<5s} "
            f"| shared={r['intersection']:2d} "
            f"| Jaccard={r['jaccard']:.3f}"
        )
    print()

    for rel in REL:
        print(f"{rel.upper()} template:")
        for r in templates[rel].head(a.template_k).to_dict("records"):
            print(
                f"  #{int(r['rank']):02d} {r['candidate']:<48s} "
                f"rate={float(r['selection_rate']):.3f} "
                f"meanM={float(r['mean_mediation']):+.3f}"
            )
        print()

    model = None
    try:
        model = model_cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        base.configure_processor(model, processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        device = torch.device(a.device)

        baseline_rows = []
        condition_rows = []
        resolved_rows = []

        for m in tqdm(eval_set, desc="Four-template counterfactuals"):
            sid = int(m["sid"])
            gt = m["gt"]
            real = gray = rb = gb = None

            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real, "convert"):
                    real = real.convert("RGB")
                gray = make_gray_image(real, a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,
                    image=real,
                    question_text=m["question_text"],
                    device=device,
                )
                gb = base.make_question_batch(
                    processor=processor,
                    image=gray,
                    question_text=m["question_text"],
                    device=device,
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                toks = [
                    str(x)
                    for x in processor.tokenizer.convert_ids_to_tokens(ids)
                ]
                spos, rpos = object_positions(
                    processor.tokenizer,
                    ids,
                    m["subject"],
                    m["reference"],
                )
                vpos = visual_positions_from_tokens(toks)
                mapping = build_canonical_mapping(
                    toks, spos, rpos, vpos, a.visual_bins
                )

                hreal = capture_states(
                    model, decoder_layers, rb, source_layers
                )
                hgray = capture_states(
                    model, decoder_layers, gb, source_layers
                )

                baseline_text = base.generate_text(
                    model, processor, rb, max_new_tokens=a.max_new_tokens
                )
                baseline_pred_raw = traj.normalize_relation(base, baseline_text)
                baseline_pred = DISPLAY.get(
                    baseline_pred_raw, baseline_pred_raw
                )
                gt_disp = DISPLAY[gt]

                baseline_rows.append({
                    "sid": sid,
                    "gt": gt_disp,
                    "baseline_prediction": baseline_pred,
                    "baseline_correct": baseline_pred_raw == gt,
                    "baseline_text": baseline_text,
                })

                specs_by_template = {}
                for template in REL:
                    specs_t, resolved_t = resolve_template_specs(
                        templates[template],
                        mapping,
                        hreal,
                        hgray,
                        a.resolve_policy,
                    )
                    specs_by_template[template] = specs_t

                    for rr in resolved_t:
                        resolved_rows.append({
                            "sid": sid,
                            "gt": gt_disp,
                            "template": template,
                            **rr,
                        })

                for template in REL:
                    specs_t = specs_by_template[template]
                    for alpha in alphas:
                        text, pred_raw, applied = generate_edit(
                            model,
                            processor,
                            decoder_layers,
                            rb,
                            specs_t,
                            alpha,
                            a.max_new_tokens,
                        )
                        pred = DISPLAY.get(pred_raw, pred_raw)
                        condition_rows.append({
                            "sid": sid,
                            "gt": gt_disp,
                            "template": template,
                            "alpha": alpha,
                            "prediction": pred,
                            "correct": pred_raw == gt,
                            "template_follow": pred == template,
                            "text": text,
                            "n_edited": sum(applied.values()),
                        })

            finally:
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

        summaries = make_summaries(baseline_rows, condition_rows)
        matrix_rows = output_matrix(condition_rows)
        cf_rows = same_sample_rows(baseline_rows, condition_rows)

        write_csv(outdir / "baseline.csv", baseline_rows)
        write_csv(outdir / "generation_conditions.csv", condition_rows)
        write_csv(outdir / "resolved_template_positions.csv", resolved_rows)
        write_csv(outdir / "template_follow_summary.csv", summaries)
        write_csv(outdir / "template_output_matrix.csv", matrix_rows)
        write_csv(outdir / "same_sample_counterfactuals.csv", cf_rows)

        # Alpha-level same-sample control summary.
        control_rows = []
        cf_df = pd.DataFrame(cf_rows)
        for alpha, g in cf_df.groupby("alpha"):
            control_rows.append({
                "alpha": float(alpha),
                "N": len(g),
                "mean_n_distinct_outputs": float(
                    g["n_distinct_template_outputs"].mean()
                ),
                "mean_n_templates_followed": float(
                    g["n_templates_followed"].mean()
                ),
                "exact_four_way_controllable_rate": float(
                    g["exact_four_way_controllable"].mean()
                ),
                "any_template_changes_output_rate": float(
                    (
                        (g["left_template_output"] != g["baseline_prediction"])
                        | (g["right_template_output"] != g["baseline_prediction"])
                        | (g["on_template_output"] != g["baseline_prediction"])
                        | (g["under_template_output"] != g["baseline_prediction"])
                    ).mean()
                ),
            })
        write_csv(outdir / "same_sample_control_summary.csv", control_rows)

        print("\n" + "=" * 150)
        print("RESULTS")
        print("=" * 150)
        base_acc = safe_mean(r["baseline_correct"] for r in baseline_rows)
        print(f"Baseline: N={len(baseline_rows)} acc={base_acc:.4f}")
        print()
        print(
            f"{'template':<10s} {'alpha':>7s} {'follow':>9s} "
            f"{'new->target':>12s} {'match_base':>11s} {'match_edit':>11s} "
            f"{'match_gain':>11s} {'W2C':>5s} {'C2W':>5s} "
            f"{'mismatch->target':>16s}"
        )
        print("-" * 150)
        for r in summaries:
            print(
                f"{r['template']:<10s} "
                f"{float(r['alpha']):>7.3f} "
                f"{float(r['target_follow_rate']):>9.3f} "
                f"{float(r['new_target_induction_rate']):>12.3f} "
                f"{float(r['matched_baseline_acc']):>11.3f} "
                f"{float(r['matched_edited_acc']):>11.3f} "
                f"{float(r['matched_gain']):>+11.3f} "
                f"{int(r['matched_W2C']):>5d} "
                f"{int(r['matched_C2W']):>5d} "
                f"{float(r['mismatched_target_follow_rate']):>16.3f}"
            )

        print("\nSame-sample counterfactual controllability:")
        for r in control_rows:
            print(
                f"  alpha={r['alpha']:.3f} | "
                f"distinct outputs={r['mean_n_distinct_outputs']:.3f}/4 | "
                f"templates followed={r['mean_n_templates_followed']:.3f}/4 | "
                f"exact 4-way={r['exact_four_way_controllable_rate']:.3f} | "
                f"any change={r['any_template_changes_output_rate']:.3f}"
            )

        metadata = {
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "selected_csv": a.selected_csv,
            "discovery_bundle": a.discovery_bundle,
            "discovery_k": a.discovery_k,
            "template_k": a.template_k,
            "resolve_policy": a.resolve_policy,
            "visual_bins": a.visual_bins,
            "eval_scope": a.eval_scope,
            "eval_N": len(eval_set),
            "discovery_eval_overlap": len(
                discovery_sids & set(int(x["sid"]) for x in eval_set)
            ),
            "alphas": alphas,
            "gt_used_for_intervention": False,
            "relation_selector_used": False,
            "late_writer_used_at_evaluation": False,
            "gradient_used_at_evaluation": False,
            "four_templates_applied_to_every_sample": True,
            "edit": "h_real[L,p] += alpha * (h_real[L,p]-h_gray[L,p])",
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        print("\nSaved:", outdir)

    finally:
        if model is not None:
            del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
