#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact canonical-position analysis for selected middle tokens.

Hypothesis
----------
All questions use the same template. The only sample-specific text spans are
SUBJECT and REFERENCE. Therefore, instead of grouping selected tokens only by
broad roles, align every sample to one canonical prompt template:

    concrete subject subtokens   -> <SUBJ>
    concrete reference subtokens -> <REF>
    every other fixed text token -> its exact canonical template slot
                                    (e.g. C037:'where')
    visual tokens                -> analyzed separately by normalized visual
                                    position bins, because image-token counts /
                                    spatial locations are not equivalent to text
                                    template slots.

This lets us test:
  "Do LEFT / RIGHT / ON / UNDER select the SAME canonical token positions?"

Input
-----
selected_tokens.csv from:
  eval_qwen_middle_token_amplify_multilayer_search_v2.py

The script re-tokenizes the original questions with the SAME processor, but it
does NOT run the model.

Example
-------
python analyze_selected_token_canonical_positions.py \
  --selected-csv output/qwen3b_middle_token_multilayer_search_v2/selected_tokens.csv \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --model qwen-3b \
  --bundle "L22+L24+L26[global_unique]" \
  --k 24 \
  --condition positive \
  --visual-bins 16 \
  --output-dir output/qwen3b_canonical_position_commonality_K24 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL_ORDER = ["left", "right", "on", "under"]
REL_CANON = {
    "left": "left",
    "right": "right",
    "above": "on",
    "on": "on",
    "below": "under",
    "under": "under",
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--selected-csv", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--bundle", default="L22+L24+L26[global_unique]")
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--condition", default="positive")
    p.add_argument("--visual-bins", type=int, default=16)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def canon_rel(x):
    return REL_CANON.get(str(x).strip().lower(), str(x).strip().lower())


def cosine(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def js_similarity(a, b):
    a = np.maximum(np.asarray(a, np.float64), 0)
    b = np.maximum(np.asarray(b, np.float64), 0)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    a, b = a / a.sum(), b / b.sum()
    m = 0.5 * (a + b)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-300))))

    return float(1.0 - (0.5 * kl(a, m) + 0.5 * kl(b, m)) / math.log(2.0))


def safe_cv(vals):
    vals = np.asarray(vals, np.float64)
    m = vals.mean()
    return float(vals.std() / m) if abs(m) > 1e-12 else float("nan")


def make_gray_image_like(real_image, value=128):
    from PIL import Image
    return Image.new("RGB", real_image.size, (value, value, value))


def visual_positions_from_tokens(tokens):
    return {
        i for i, tok in enumerate(tokens)
        if "image_pad" in str(tok) or "video_pad" in str(tok)
    }


def get_object_spans(tokenizer, ids, subject, reference):
    """
    Use the repo's exact locator. Return sets of token positions.
    """
    sspan, rspan = base.locate_object_spans(tokenizer, ids, subject, reference)
    spos = set(range(int(sspan[0]), int(sspan[1]) + 1))
    rpos = set(range(int(rspan[0]), int(rspan[1]) + 1))
    return spos, rpos


def build_canonical_mapping(tokens, subject_pos, reference_pos, visual_pos, visual_bins):
    """
    Map every original token position to:
      - exact canonical text slot, after collapsing subject/ref/visual spans
      - special semantic slot (<SUBJ>/<REF>)
      - normalized visual bin for visual tokens

    All subtokens inside subject map to one <SUBJ> slot.
    All subtokens inside reference map to one <REF> slot.
    All visual tokens are excluded from exact text-slot comparison and get
    VISBIN_XX labels instead.
    """
    n = len(tokens)
    mapping = {}
    canonical_seq = []
    canonical_index = 0

    # Visual ordinal for normalized binning.
    visual_sorted = sorted(visual_pos)
    visual_ord = {p: j for j, p in enumerate(visual_sorted)}
    nvis = len(visual_sorted)

    i = 0
    while i < n:
        if i in visual_pos:
            # Consume one contiguous visual block as one template placeholder.
            block = []
            while i < n and i in visual_pos:
                block.append(i)
                i += 1

            canonical_label = f"C{canonical_index:03d}:<VISUAL_BLOCK>"
            canonical_seq.append("<VISUAL_BLOCK>")
            canonical_index += 1

            for p in block:
                j = visual_ord[p]
                if nvis <= 1:
                    b = 0
                else:
                    frac = j / (nvis - 1)
                    b = min(visual_bins - 1, int(frac * visual_bins))
                mapping[p] = {
                    "canonical_kind": "visual",
                    "canonical_slot": None,
                    "semantic_slot": "<VISUAL>",
                    "visual_ordinal": j,
                    "visual_count": nvis,
                    "visual_frac": (j / (nvis - 1)) if nvis > 1 else 0.0,
                    "visual_bin": f"VISBIN_{b:02d}",
                }
            continue

        if i in subject_pos:
            block = []
            while i < n and i in subject_pos:
                block.append(i)
                i += 1

            label = f"C{canonical_index:03d}:<SUBJ>"
            canonical_seq.append("<SUBJ>")
            canonical_index += 1
            for p in block:
                mapping[p] = {
                    "canonical_kind": "subject",
                    "canonical_slot": label,
                    "semantic_slot": "<SUBJ>",
                    "visual_ordinal": None,
                    "visual_count": nvis,
                    "visual_frac": None,
                    "visual_bin": None,
                }
            continue

        if i in reference_pos:
            block = []
            while i < n and i in reference_pos:
                block.append(i)
                i += 1

            label = f"C{canonical_index:03d}:<REF>"
            canonical_seq.append("<REF>")
            canonical_index += 1
            for p in block:
                mapping[p] = {
                    "canonical_kind": "reference",
                    "canonical_slot": label,
                    "semantic_slot": "<REF>",
                    "visual_ordinal": None,
                    "visual_count": nvis,
                    "visual_frac": None,
                    "visual_bin": None,
                }
            continue

        tok = str(tokens[i]).replace("\n", "\\n")
        label = f"C{canonical_index:03d}:{tok}"
        canonical_seq.append(tok)
        mapping[i] = {
            "canonical_kind": "fixed_text",
            "canonical_slot": label,
            "semantic_slot": tok,
            "visual_ordinal": None,
            "visual_count": nvis,
            "visual_frac": None,
            "visual_bin": None,
        }
        canonical_index += 1
        i += 1

    signature = "\t".join(canonical_seq)
    signature_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return mapping, canonical_seq, signature_hash


def build_rate_table(df, slot_col, relations, all_samples):
    """
    Selection rate = fraction of samples in that relation where the slot was
    selected at least once. This avoids subject/ref subtoken-count confounds.
    """
    valid = df[df[slot_col].notna()].copy()
    valid = valid.drop_duplicates(["relation", "sid", slot_col])

    slots = sorted(valid[slot_col].astype(str).unique())
    rows = []
    for slot in slots:
        row = {"slot": slot}
        vals = []
        counts = []
        for rel in relations:
            denom = len(all_samples.get(rel, []))
            n = int(
                valid[
                    (valid["relation"] == rel)
                    & (valid[slot_col].astype(str) == slot)
                ]["sid"].nunique()
            )
            rate = n / denom if denom else float("nan")
            row[f"n_{rel}"] = n
            row[f"rate_{rel}"] = rate
            if np.isfinite(rate):
                vals.append(rate)
                counts.append(n)

        if vals:
            row["mean_rate"] = float(np.mean(vals))
            row["min_rate"] = float(np.min(vals))
            row["max_rate"] = float(np.max(vals))
            row["std_rate"] = float(np.std(vals))
            row["cv_rate"] = safe_cv(vals)
            row["range_rate"] = float(np.max(vals) - np.min(vals))
            row["relations_present"] = int(sum(v > 0 for v in vals))
        else:
            row.update({
                "mean_rate": np.nan, "min_rate": np.nan, "max_rate": np.nan,
                "std_rate": np.nan, "cv_rate": np.nan, "range_rate": np.nan,
                "relations_present": 0,
            })
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["min_rate", "mean_rate", "cv_rate"],
            ascending=[False, False, True],
        )
    return out


def pairwise_slot_similarity(rate_table, relations):
    slots = rate_table["slot"].tolist()
    vecs = {}
    for rel in relations:
        vecs[rel] = rate_table[f"rate_{rel}"].fillna(0).to_numpy(np.float64)

    rows = []
    for i, a in enumerate(relations):
        for b in relations[i + 1:]:
            rows.append({
                "relation_a": a,
                "relation_b": b,
                "cosine_similarity": cosine(vecs[a], vecs[b]),
                "js_similarity": js_similarity(vecs[a], vecs[b]),
            })
    return pd.DataFrame(rows)


def main():
    a = parse_args()

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(a.selected_csv)
    sel["relation"] = sel["relation"].map(canon_rel)
    sel["k"] = pd.to_numeric(sel["k"], errors="raise").astype(int)
    sel["source_layer"] = pd.to_numeric(sel["source_layer"], errors="raise").astype(int)
    sel["position"] = pd.to_numeric(sel["position"], errors="raise").astype(int)

    sel = sel[
        (sel["condition"].astype(str) == str(a.condition))
        & (sel["source_bundle"].astype(str) == str(a.bundle))
        & (sel["k"] == int(a.k))
        & (sel["relation"].isin(REL_ORDER))
    ].copy()

    if not len(sel):
        raise RuntimeError(
            f"No selected-token rows for condition={a.condition}, "
            f"bundle={a.bundle}, K={a.k}"
        )

    # Defensive dedupe.
    sel = sel.sort_values("mediation", ascending=False).drop_duplicates(
        ["sid", "relation", "source_layer", "position"], keep="first"
    )

    wanted_sids = sorted(set(sel["sid"].astype(int)))

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    tokenizer = processor.tokenizer

    # Rebuild exact tokenization and canonical maps sample by sample.
    canonical_maps = {}
    template_rows = []

    for sid in tqdm(wanted_sids, desc="Canonicalize prompts"):
        if sid not in prompts or sid not in rec_by_sid:
            raise RuntimeError(f"sid={sid} missing from prompts or records")

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
            tokens = [str(x) for x in tokenizer.convert_ids_to_tokens(ids)]

            subject = str(p["subject"])
            reference = str(p["reference"])
            spos, rpos = get_object_spans(tokenizer, ids, subject, reference)
            vpos = visual_positions_from_tokens(tokens)

            mapping, canonical_seq, sig_hash = build_canonical_mapping(
                tokens, spos, rpos, vpos, a.visual_bins
            )
            canonical_maps[sid] = mapping

            template_rows.append({
                "sid": sid,
                "subject": subject,
                "reference": reference,
                "n_input_tokens": len(tokens),
                "n_visual_tokens": len(vpos),
                "subject_n_tokens": len(spos),
                "reference_n_tokens": len(rpos),
                "canonical_n_slots": len(canonical_seq),
                "template_signature": sig_hash,
                "canonical_sequence": " | ".join(canonical_seq),
            })
        finally:
            if real is not None:
                with contextlib.suppress(Exception):
                    real.close()
            del batch

    templates = pd.DataFrame(template_rows)
    templates.to_csv(outdir / "template_signatures.csv", index=False)

    # Attach canonical position info to every selected row.
    annotated = []
    misses = []
    for row in sel.to_dict("records"):
        sid = int(row["sid"])
        pos = int(row["position"])
        info = canonical_maps.get(sid, {}).get(pos)
        if info is None:
            misses.append((sid, pos))
            continue
        rr = dict(row)
        rr.update(info)
        annotated.append(rr)

    ann = pd.DataFrame(annotated)
    ann.to_csv(outdir / "canonical_selected_rows.csv", index=False)

    if misses:
        pd.DataFrame(misses, columns=["sid", "position"]).to_csv(
            outdir / "unmapped_positions.csv", index=False
        )

    all_samples = {
        rel: sorted(set(ann.loc[ann["relation"] == rel, "sid"].astype(int)))
        for rel in REL_ORDER
    }

    # ------------------------------------------------------------------
    # Exact canonical TEXT slots.
    # ------------------------------------------------------------------
    text_ann = ann[ann["canonical_kind"] != "visual"].copy()

    exact = build_rate_table(
        text_ann,
        slot_col="canonical_slot",
        relations=REL_ORDER,
        all_samples=all_samples,
    )
    exact.to_csv(outdir / "canonical_text_slot_commonality.csv", index=False)

    exact_sim = pairwise_slot_similarity(exact, REL_ORDER)
    exact_sim.to_csv(
        outdir / "pairwise_exact_text_position_similarity.csv", index=False
    )

    # Also collapse canonical layer + slot. This asks:
    # "Is 'where at L22' consistently selected?"
    text_ann["layer_slot"] = (
        "L" + text_ann["source_layer"].astype(str)
        + "::" + text_ann["canonical_slot"].astype(str)
    )
    layer_exact = build_rate_table(
        text_ann,
        slot_col="layer_slot",
        relations=REL_ORDER,
        all_samples=all_samples,
    )
    layer_exact.to_csv(
        outdir / "canonical_layer_text_slot_commonality.csv", index=False
    )

    layer_exact_sim = pairwise_slot_similarity(layer_exact, REL_ORDER)
    layer_exact_sim.to_csv(
        outdir / "pairwise_exact_layer_text_position_similarity.csv", index=False
    )

    # Semantic SUBJ/REF separately by layer.
    sr = text_ann[text_ann["semantic_slot"].isin(["<SUBJ>", "<REF>"])].copy()
    if len(sr):
        sr["layer_semantic_slot"] = (
            "L" + sr["source_layer"].astype(str)
            + "::" + sr["semantic_slot"].astype(str)
        )
        sr_common = build_rate_table(
            sr,
            slot_col="layer_semantic_slot",
            relations=REL_ORDER,
            all_samples=all_samples,
        )
        sr_common.to_csv(outdir / "subject_reference_commonality.csv", index=False)
    else:
        sr_common = pd.DataFrame()

    # Fixed literal tokens, but disambiguated by canonical position.
    fixed = text_ann[text_ann["canonical_kind"] == "fixed_text"].copy()
    if len(fixed):
        fixed_common = build_rate_table(
            fixed,
            slot_col="canonical_slot",
            relations=REL_ORDER,
            all_samples=all_samples,
        )
        fixed_common.to_csv(
            outdir / "fixed_text_exact_position_commonality.csv", index=False
        )
    else:
        fixed_common = pd.DataFrame()

    # ------------------------------------------------------------------
    # VISUAL positions: normalized bins, separate from text.
    # ------------------------------------------------------------------
    vis = ann[ann["canonical_kind"] == "visual"].copy()
    if len(vis):
        vis["layer_visual_bin"] = (
            "L" + vis["source_layer"].astype(str)
            + "::" + vis["visual_bin"].astype(str)
        )
        vis_common = build_rate_table(
            vis,
            slot_col="layer_visual_bin",
            relations=REL_ORDER,
            all_samples=all_samples,
        )
        vis_common.to_csv(outdir / "visual_normalized_bin_commonality.csv", index=False)
        vis_sim = pairwise_slot_similarity(vis_common, REL_ORDER)
        vis_sim.to_csv(
            outdir / "pairwise_visual_bin_similarity.csv", index=False
        )
    else:
        vis_common = pd.DataFrame()
        vis_sim = pd.DataFrame()

    # ------------------------------------------------------------------
    # Per-sample canonical set overlap statistics.
    # ------------------------------------------------------------------
    sample_sets = {}
    for (rel, sid), g in ann.groupby(["relation", "sid"]):
        # Exact text positions + normalized visual bins; layer is included.
        s = set()
        for r in g.itertuples(index=False):
            if r.canonical_kind == "visual":
                s.add(f"L{int(r.source_layer)}::<VISUAL>::{r.visual_bin}")
            else:
                s.add(f"L{int(r.source_layer)}::{r.canonical_slot}")
        sample_sets[(rel, int(sid))] = s

    overlap_rows = []
    keys = sorted(sample_sets)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            sa, sb = sample_sets[ka], sample_sets[kb]
            union = len(sa | sb)
            jac = len(sa & sb) / union if union else float("nan")
            overlap_rows.append({
                "relation_a": ka[0],
                "sid_a": ka[1],
                "relation_b": kb[0],
                "sid_b": kb[1],
                "same_relation": ka[0] == kb[0],
                "jaccard": jac,
                "intersection": len(sa & sb),
                "union": union,
            })
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(outdir / "pairwise_sample_canonical_jaccard.csv", index=False)

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    sig_counts = templates["template_signature"].value_counts()
    n_sigs = int(len(sig_counts))

    lines = []
    lines.append("=" * 120)
    lines.append("EXACT CANONICAL TOKEN-POSITION COMMONALITY")
    lines.append("=" * 120)
    lines.append(f"model      : {a.model} ({spec.repo_id})")
    lines.append(f"bundle     : {a.bundle}")
    lines.append(f"K          : {a.k}")
    lines.append(
        "samples    : " +
        ", ".join(f"{r}={len(all_samples[r])}" for r in REL_ORDER)
    )
    lines.append(f"selected rows mapped: {len(ann)} / {len(sel)}")
    lines.append(f"canonical template signatures: {n_sigs}")
    if n_sigs == 1:
        lines.append(
            "TEMPLATE CHECK: PASS — after collapsing <SUBJ>, <REF>, and the visual block, "
            "all samples have the same canonical token sequence."
        )
    else:
        lines.append(
            "TEMPLATE CHECK: MULTIPLE SIGNATURES — inspect template_signatures.csv before "
            "treating absolute canonical slots as identical."
        )

    lines.append("")
    lines.append("TOP EXACT LAYER + TEXT POSITIONS SHARED ACROSS ALL FOUR RELATIONS")
    lines.append("-" * 120)
    if len(layer_exact):
        for r in layer_exact.head(30).itertuples(index=False):
            lines.append(
                f"{r.slot:<52s} | "
                f"L={r.rate_left:.3f} R={r.rate_right:.3f} "
                f"ON={r.rate_on:.3f} U={r.rate_under:.3f} | "
                f"mean={r.mean_rate:.3f} min={r.min_rate:.3f} "
                f"CV={r.cv_rate:.3f}"
            )

    lines.append("")
    lines.append("SUBJECT / REFERENCE EXACT ROLE POSITIONS")
    lines.append("-" * 120)
    if len(sr_common):
        for r in sr_common.itertuples(index=False):
            lines.append(
                f"{r.slot:<28s} | "
                f"L={r.rate_left:.3f} R={r.rate_right:.3f} "
                f"ON={r.rate_on:.3f} U={r.rate_under:.3f} | "
                f"mean={r.mean_rate:.3f} min={r.min_rate:.3f}"
            )

    lines.append("")
    lines.append("PAIRWISE EXACT LAYER+TEXT POSITION SIMILARITY")
    lines.append("-" * 120)
    if len(layer_exact_sim):
        for r in layer_exact_sim.itertuples(index=False):
            lines.append(
                f"{r.relation_a:>5s} vs {r.relation_b:<5s} | "
                f"cos={r.cosine_similarity:.4f} | JS-sim={r.js_similarity:.4f}"
            )
        lines.append(
            f"mean cosine={layer_exact_sim['cosine_similarity'].mean():.4f}, "
            f"min cosine={layer_exact_sim['cosine_similarity'].min():.4f}"
        )

    if len(vis_common):
        lines.append("")
        lines.append("TOP NORMALIZED VISUAL BINS (SEPARATE ANALYSIS)")
        lines.append("-" * 120)
        for r in vis_common.head(20).itertuples(index=False):
            lines.append(
                f"{r.slot:<28s} | "
                f"L={r.rate_left:.3f} R={r.rate_right:.3f} "
                f"ON={r.rate_on:.3f} U={r.rate_under:.3f} | "
                f"mean={r.mean_rate:.3f} min={r.min_rate:.3f}"
            )

    if len(overlap):
        same = overlap[overlap["same_relation"]]["jaccard"]
        cross = overlap[~overlap["same_relation"]]["jaccard"]
        lines.append("")
        lines.append("PER-SAMPLE CANONICAL SET OVERLAP")
        lines.append("-" * 120)
        lines.append(
            f"same-relation mean Jaccard={same.mean():.4f} "
            f"(Npairs={len(same)})"
        )
        lines.append(
            f"cross-relation mean Jaccard={cross.mean():.4f} "
            f"(Npairs={len(cross)})"
        )

    # Easy-to-read strongest universal candidates.
    universal = layer_exact[
        (layer_exact["relations_present"] == 4)
        & (layer_exact["min_rate"] >= 0.50)
    ] if len(layer_exact) else pd.DataFrame()

    lines.append("")
    lines.append("CANDIDATE FIXED WRITER-FREE TEXT MASK")
    lines.append("-" * 120)
    if len(universal):
        lines.append(
            "Exact layer+text slots selected in >=50% of samples in EVERY relation:"
        )
        for r in universal.itertuples(index=False):
            lines.append(
                f"  {r.slot} | min={r.min_rate:.3f}, mean={r.mean_rate:.3f}"
            )
    else:
        lines.append(
            "No exact layer+text slot reached >=50% selection rate in every relation "
            "at this K. Use the ranked table rather than forcing a universal mask."
        )

    report = "\n".join(lines)
    print(report)
    (outdir / "summary.txt").write_text(report, encoding="utf-8")

    meta = {
        "selected_csv": str(a.selected_csv),
        "model": a.model,
        "repo_id": spec.repo_id,
        "bundle": a.bundle,
        "k": a.k,
        "condition": a.condition,
        "visual_bins": a.visual_bins,
        "n_samples": len(wanted_sids),
        "n_template_signatures": n_sigs,
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\nSaved:")
    for p in sorted(outdir.iterdir()):
        print(" ", p)


if __name__ == "__main__":
    main()
