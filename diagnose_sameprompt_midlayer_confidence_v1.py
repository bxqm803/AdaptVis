#!/usr/bin/env python3
"""Same-question generation -> fixed-layer conflict confidence diagnostic.

Independent entry point; does NOT import diagnose_midlayer_vs_final_v1.py.
Uses existing repository extraction/generation utilities for GPU inference.
Default: fixed residual L25 (zero-based block output), plus REAL/NoImage controls.
Reuses legacy hsub/href caches and generates clean answers with exactly the
readout question. Synthetic labels construct the centered cosine codebook.
No COCO labels select layers, confidence thresholds, or policies.

Step 1: same-question free generation and middle/final paired outcomes.
Step 2: conflict confidence distributions, AUC, fixed bins, and source-OOF
        quantile thresholds for answer replacement. Every policy is reported;
        no target-optimal threshold is selected.
Step 3 prerequisite: per-sample four-way cosine scores are exported for a
        later soft-evidence intervention experiment. No update is patched here.
        A positive diagnostic is NOT evidence of successful HOW prediction.

--baseline-csv can reuse a previously generated same-question CSV, provided
its sibling metadata.json verifies the CSV hash, model, question and settings.
Otherwise clean generation is mandatory. Default full COCO; --target-max-samples
uses the old first-N cache convention, not a stratified cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REL = ("left", "right", "above", "below")
DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)
ALIASES = {"on": "above", "over": "above", "top": "above",
           "under": "below", "underneath": "below", "beneath": "below",
           "bottom": "below"}


def canon(x):
    s = str(x).strip().lower()
    return ALIASES.get(s, s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default="")
    p.add_argument("--source-max-samples", type=int, default=0)
    p.add_argument("--target-max-samples", type=int, default=0,
                   help="0=all; positive=first N target rows, matching old cache convention")
    p.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--cache-dir", default="output/qwen3b_hsub_href_spatial_cache")
    p.add_argument("--source-cache", default="", help="Explicit existing trusted NPZ path")
    p.add_argument("--target-cache", default="", help="Explicit existing trusted NPZ path")
    p.add_argument("--baseline-csv", default="", help="Reuse a verified SAME-question baseline; sibling metadata.json required")
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--analysis-layers", default="25")
    p.add_argument("--primary-layer", default="25",
                   help="source_oof or a predeclared integer; never selected on target labels")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if min(a.source_max_samples, a.target_max_samples) < 0 or a.cv_folds < 2:
        p.error("sample limits must be nonnegative and cv-folds >= 2")
    if a.max_new_tokens < 1:
        p.error("max-new-tokens must be positive")
    a.generate_baseline = not bool(a.baseline_csv)
    return a


def layer_list(text, n):
    if text.strip().lower() == "all":
        return list(range(n))
    result = set()
    for part in text.split(","):
        bits = part.strip().split("-")
        if len(bits) == 1:
            result.add(int(bits[0]))
        elif len(bits) == 2:
            lo, hi = map(int, bits)
            if lo > hi:
                raise ValueError(f"Reversed layer range: {part}")
            result.update(range(lo, hi + 1))
        else:
            raise ValueError(f"Bad layer range: {part}")
    if not result or min(result) < 0 or max(result) >= n:
        raise ValueError(f"Layers {sorted(result)} outside 0..{n-1}")
    return sorted(result)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_cache(path, a):
    # The legacy NPZ stores labels and metadata as object arrays. Trusted files only.
    with np.load(path, allow_pickle=True) as z:
        for key, expected in [("model", a.model), ("pool", a.pool),
                              ("prompt_template", a.prompt_template)]:
            if key not in z or str(z[key].item()) != expected:
                raise ValueError(f"{path}: missing/mismatched {key}; use a separate cache-dir")
        sid = np.asarray(z["sample_index"])
        if not np.issubdtype(sid.dtype, np.integer):
            raise ValueError(f"{path}: sample_index must be integer")
        y = np.asarray([canon(v) for v in z["relation"]])
        img, no = np.asarray(z["img"]), np.asarray(z["no_image"])
        idx = np.asarray(z["decoder_block_index"])
    if img.ndim != 3 or img.shape != no.shape or len(sid) != len(img) or len(y) != len(sid):
        raise ValueError(f"{path}: inconsistent [N,L,D] shapes")
    if not len(sid) or len(np.unique(sid)) != len(sid):
        raise ValueError(f"{path}: empty or duplicate SIDs")
    if not np.array_equal(idx, np.arange(img.shape[1])):
        raise ValueError(f"{path}: nonstandard block indexing")
    if not set(y) <= set(REL) or not np.isfinite(img).all() or not np.isfinite(no).all():
        raise ValueError(f"{path}: invalid labels or nonfinite hidden states")
    return sid.astype(int), y, img, no


def obtain_inputs(a, out):
    cache = Path(a.cache_dir)
    stag = f"N{a.source_max_samples}" if a.source_max_samples else "all"
    ttag = f"N{a.target_max_samples}" if a.target_max_samples else "all"
    sp = Path(a.source_cache) if a.source_cache else cache / f"{a.model}_synthetic_hsub_href_{stag}.npz"
    tp = Path(a.target_cache) if a.target_cache else cache / f"{a.model}_{a.dataset}_hsub_href_{ttag}.npz"
    bp = Path(a.baseline_csv) if a.baseline_csv else out / "baseline_generated.csv"
    if a.baseline_csv:
        verify_same_question_baseline(bp, a)
    if not a.generate_baseline and not bp.exists():
        raise FileNotFoundError(f"{bp}; supply --baseline-csv or use --generate-baseline")
    for explicit, path in [(a.source_cache, sp), (a.target_cache, tp)]:
        if explicit and not path.exists():
            raise FileNotFoundError(path)
    if sp.resolve() == tp.resolve():
        raise ValueError("Source and target caches must differ")
    if not sp.exists() or not tp.exists() or a.generate_baseline:
        import torch
        import analyze_hsub_href_spatial_update_sign_v1 as old
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        source_rows = target_rows = None
        if not sp.exists():
            source_rows, _ = old.load_synthetic_rows(a)
        if not tp.exists() or a.generate_baseline:
            target_rows, _ = old.load_target_rows(a)
        model, processor, layers, _, _ = old.load_model(a)
        try:
            for path, rows, name in [(sp, source_rows, "Synthetic"), (tp, target_rows, "Target")]:
                if not path.exists():
                    got = old.extract_hidden_cache(args=a, rows=rows, model=model,
                        processor=processor, layers=layers, cache_path=path, desc=name)
                    expected = np.asarray([r["sid"] for r in rows])
                    if not np.array_equal(got[0], expected):
                        raise RuntimeError(f"{path}: extraction skipped samples; inspect {path}.errors.json before continuing")
            if a.generate_baseline:
                import analyze_coco_centroid_generation_step1_v4 as gen
                from PIL import Image
                from tqdm import tqdm
                bp = out / "baseline_generated.csv"
                results = []
                for rec in tqdm(target_rows, desc="SAME-QUESTION baseline generation"):
                    with Image.open(rec["image_path"]) as raw:
                        image = raw.convert("RGB")
                        try:
                            rendered = old.dh.build_chat_prompt(processor, rec["question"], True)
                            batch = old.dh.process_inputs(processor, rendered, image, torch.device(a.device))
                            text = gen.generate_text(model, processor, batch, a.max_new_tokens)
                        finally:
                            image.close()
                    pred = gen.normalize_relation(text)
                    pred = canon(pred) if pred is not None else "invalid"
                    results.append(dict(sid=rec["sid"], gt=rec["relation"], condition="baseline",
                        prediction=pred, correct=pred == rec["relation"], text=text,
                        question=rec["question"]))
                    del batch
                pd.DataFrame(results).to_csv(bp, index=False)
        finally:
            del model, processor, layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    source, target = read_cache(sp, a), read_cache(tp, a)
    if a.source_max_samples and len(source[0]) != a.source_max_samples:
        raise ValueError("source cache N differs from source-max-samples")
    if a.target_max_samples and len(target[0]) != a.target_max_samples:
        raise ValueError("target cache N differs from target-max-samples")
    if source[2].shape[1:] != target[2].shape[1:]:
        raise ValueError("Source/target hidden geometries differ")
    for path in [sp, tp]:
        ep = path.with_suffix(path.suffix + ".errors.json")
        if ep.exists() and json.loads(ep.read_text()):
            raise ValueError(f"{ep}: previous extraction errors; repair and regenerate the cache")
    return source, target, bp, sp, tp


def boolean(x):
    s = str(x).strip().lower()
    if s in ("true", "1", "1.0"):
        return True
    if s in ("false", "0", "0.0"):
        return False
    raise ValueError(f"Invalid correctness value: {x!r}")


def load_baseline(path, sids, labels):
    df = pd.read_csv(path, keep_default_na=False)
    if "condition" in df:
        df = df[df.condition == "baseline"].copy()
    pred_col = "prediction" if "prediction" in df else "baseline_prediction"
    if not {"sid", "gt", pred_col} <= set(df.columns):
        raise ValueError(f"{path}: requires sid, gt, prediction (or baseline_prediction)")
    nums = pd.to_numeric(df.sid, errors="raise")
    if not np.isfinite(nums).all() or (nums % 1 != 0).any():
        raise ValueError("Baseline SIDs must be integers")
    df["sid"] = nums.astype(int)
    if df.sid.duplicated().any():
        raise ValueError("Duplicate baseline SIDs; do not mix seeds/runs")
    missing = sorted(set(sids) - set(df.sid))
    if missing:
        raise ValueError(f"Missing baseline rows for {len(missing)} target SIDs: {missing[:15]}")
    aligned = df.set_index("sid").loc[sids].copy()
    gt = np.asarray([canon(x) for x in aligned["gt"]])
    if not np.array_equal(gt, labels):
        raise ValueError(f"GT mismatch after SID join: {sids[gt != labels][:15].tolist()}")
    pred = np.asarray([canon(x) if canon(x) in REL else "invalid" for x in aligned[pred_col]])
    correct = pred == labels
    for col in ("correct", "baseline_correct"):
        if col in aligned and not np.array_equal(aligned[col].map(boolean).to_numpy(), correct):
            raise ValueError(f"Baseline {col} disagrees with normalized prediction vs GT")
    return pred, correct, dict(baseline_rows=len(df), target_rows=len(sids),
                              extra_baseline_rows=len(df) - len(sids))


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fit(X, y):
    center = X.mean(axis=0)
    directions = np.stack([X[y == r].mean(axis=0) - center for r in REL], axis=1)
    return center, normalize(directions)


def score(X, center, directions):
    return np.einsum("nld,lrd->nlr", normalize(X - center[None]), directions, optimize=True)


def source_oof(X, y, folds, seed, return_scores=False):
    counts = {r: int(np.sum(y == r)) for r in REL}
    if min(counts.values()) < folds:
        raise ValueError(f"Need >= cv-folds examples per source class: {counts}")
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    for r in REL:
        indices = np.flatnonzero(y == r)
        rng.shuffle(indices)
        for f, part in enumerate(np.array_split(indices, folds)):
            buckets[f].extend(part.tolist())
    predictions = np.empty((len(y), X.shape[1]), dtype=int)
    oof_scores = np.empty((len(y), X.shape[1], 4), dtype=np.float32)
    for bucket in buckets:
        te = np.asarray(sorted(bucket))
        train = np.ones(len(y), dtype=bool)
        train[te] = False
        oof_scores[te] = score(X[te], *fit(X[train], y[train]))
        predictions[te] = oof_scores[te].argmax(axis=-1)
    yi = np.asarray([REL.index(r) for r in y])
    accuracy = (predictions == yi[:, None]).mean(axis=0)
    return (accuracy, oof_scores) if return_scores else accuracy


def ratio(k, n):
    return float(k / n) if n else float("nan")


def wilson(k, n):
    if not n:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p, den = k / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def summarize(df):
    bc, mc = df.baseline_correct.to_numpy(bool), df.mid_correct.to_numpy(bool)
    conflict = df.conflict.to_numpy(bool)
    n, w2c, c2w = len(df), int(np.sum(~bc & mc)), int(np.sum(bc & ~mc))
    nc, nw = int(conflict.sum()), int((~bc).sum())
    informative = w2c + c2w
    # Exact paired McNemar/binomial test. Exploratory for scanned layers.
    p = min(1.0, 2 * sum(math.exp(math.lgamma(informative + 1) - math.lgamma(k + 1)
            - math.lgamma(informative - k + 1) - informative * math.log(2))
            for k in range(min(w2c, c2w) + 1))) if informative else 1.0
    wlo, whi = wilson(w2c, nw)
    clo, chi = wilson(w2c, nc)
    return dict(N=n, baseline_accuracy=ratio(bc.sum(), n), mid_accuracy=ratio(mc.sum(), n),
        baseline_wrong_N=nw, mid_accuracy_on_baseline_wrong=ratio(w2c, nw),
        wrong_acc_ci95_low=wlo, wrong_acc_ci95_high=whi,
        baseline_correct_N=int(bc.sum()), mid_accuracy_on_baseline_correct=ratio((bc & mc).sum(), bc.sum()),
        conflict_N=nc, conflict_rate=ratio(nc, n),
        mid_wins=w2c, baseline_wins=c2w,
        conflict_both_wrong=int(np.sum(conflict & ~bc & ~mc)),
        mid_accuracy_on_conflict=ratio(w2c, nc), baseline_accuracy_on_conflict=ratio(c2w, nc),
        conflict_mid_acc_ci95_low=clo, conflict_mid_acc_ci95_high=chi,
        mid_win_share_when_one_correct=ratio(w2c, informative),
        agree_both_correct=int(np.sum(~conflict & bc & mc)),
        agree_both_wrong=int(np.sum(~conflict & ~bc & ~mc)),
        W2C=w2c, C2W=c2w, net=w2c-c2w, accuracy_gain=ratio(w2c-c2w, n),
        oracle_union_accuracy=ratio(np.sum(bc | mc), n),
        invalid_baseline_N=int(np.sum(df.baseline_prediction == "invalid")),
        paired_exact_p=p)


def verify_same_question_baseline(path, a):
    """Reject legacy/external baselines with unknown prompt provenance."""
    meta_path = path.parent / "metadata.json"
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Need {path} and its sibling metadata.json")
    meta = json.loads(meta_path.read_text())
    if meta.get("baseline_protocol") != "same_readout_question_free_generation":
        raise ValueError("Baseline is not verified same-question generation. Omit --baseline-csv to regenerate.")
    settings = meta.get("args", {})
    for key in ("model", "dataset", "prompt_template", "pool", "max_new_tokens", "attn_impl"):
        if settings.get(key) != getattr(a, key):
            raise ValueError(f"Baseline metadata mismatch: {key}")
    recorded = [h for p, h in meta.get("inputs", {}).items() if Path(p).name == path.name]
    if len(recorded) != 1 or recorded[0] != sha256(path):
        raise ValueError("Baseline CSV hash is missing, ambiguous, or changed")
    header = pd.read_csv(path, nrows=1)
    if "question" not in header:
        raise ValueError("Verified baseline must retain each generation question")


def binary_auc(labels, values):
    """Mann-Whitney AUC, higher confidence predicts label=True; ties count half."""
    labels, values = np.asarray(labels, dtype=bool), np.asarray(values, dtype=float)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        return float("nan")
    ranks = pd.Series(values).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def confidence_analysis(df, source_scores):
    """All source-derived thresholds frozen before target outcome evaluation."""
    ident = dict(representation=str(df.representation.iloc[0]), layer=int(df.layer.iloc[0]))
    conflicts = df[df.conflict].copy()
    outcomes = np.where(conflicts.mid_correct, "mid_wins",
                        np.where(conflicts.baseline_correct, "baseline_wins", "both_wrong"))
    conflicts["outcome"] = outcomes
    gap = conflicts.mid_top2_cosine_gap.to_numpy(float)
    decisive = outcomes != "both_wrong"
    auc_rows = []
    for name, mask in [("mid_wins_vs_baseline_wins", decisive),
                       ("mid_correct_vs_all_other_conflicts", np.ones(len(gap), dtype=bool))]:
        positive = outcomes[mask] == "mid_wins"
        auc_rows.append(dict(**ident, comparison=name, N=int(mask.sum()),
            positive_N=int(positive.sum()), negative_N=int((~positive).sum()),
            auc=binary_auc(positive, gap[mask])))
    distributions = []
    for name in ("mid_wins", "baseline_wins", "both_wrong"):
        v = gap[outcomes == name]
        row = dict(**ident, outcome=name, N=len(v), mean=float(v.mean()) if len(v) else float("nan"))
        for q in (0, .1, .25, .5, .75, .9, 1):
            row[f"q{int(q*100):02d}"] = float(np.quantile(v, q)) if len(v) else float("nan")
        distributions.append(row)
    # Fixed intervals do not depend on COCO scores or labels. Last bin includes 2.
    edges = [0, .05, .1, .2, .3, .5, 1, float("inf")]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (gap >= lo) & (gap < hi)
        subset = conflicts.loc[mask]
        k, n = int(subset.mid_correct.sum()), len(subset)
        ci = wilson(k, n)
        bins.append(dict(**ident, gap_lo=lo, gap_hi=hi, N=n,
            mid_wins=k, baseline_wins=int(subset.baseline_correct.sum()),
            both_wrong=int(np.sum((~subset.mid_correct) & (~subset.baseline_correct))),
            mid_accuracy=ratio(k, n), ci95_low=ci[0], ci95_high=ci[1]))
    sorted_source = np.sort(source_scores, axis=-1)
    source_gap = sorted_source[:, -1] - sorted_source[:, -2]
    # Zero is an all-conflicts reference. Repeated thresholds are intentionally kept.
    thresholds = [("all_conflicts_reference", -1.0, 0.0)] + [
        (f"source_oof_q{int(q*100):02d}", q, float(np.quantile(source_gap, q)))
        for q in (0, .25, .5, .75, .9, .95)]
    policies = []
    base_ok = df.baseline_correct.to_numpy(bool)
    mid_ok = df.mid_correct.to_numpy(bool)
    is_conflict = df.conflict.to_numpy(bool)
    for name, q, threshold in thresholds:
        trigger = is_conflict & (df.mid_top2_cosine_gap.to_numpy(float) >= threshold)
        final_ok = np.where(trigger, mid_ok, base_ok)
        w2c = int(np.sum(trigger & ~base_ok & mid_ok))
        c2w = int(np.sum(trigger & base_ok & ~mid_ok))
        n = int(trigger.sum())
        ci = wilson(w2c, n)
        policies.append(dict(**ident, policy=name, source_quantile=q, threshold=threshold,
            N=len(df), conflict_N=int(is_conflict.sum()), triggered_N=n,
            target_coverage=ratio(n, len(df)), conflict_coverage=ratio(n, is_conflict.sum()),
            baseline_accuracy=float(base_ok.mean()), final_accuracy=float(final_ok.mean()),
            gain=float(final_ok.mean() - base_ok.mean()), W2C=w2c, C2W=c2w, net=w2c-c2w,
            triggered_both_wrong=int(np.sum(trigger & ~base_ok & ~mid_ok)),
            triggered_mid_accuracy=ratio(w2c, n), triggered_mid_ci95_low=ci[0],
            triggered_mid_ci95_high=ci[1],
            untriggered_repairable_N=int(np.sum(~trigger & ~base_ok & mid_ok))))
    return dict(auc=auc_rows, distributions=distributions, bins=bins, policies=policies)


def save_confidence(parts, samples, out, primary):
    names = {"auc": "confidence_auc.csv", "distributions": "confidence_distributions.csv",
             "bins": "confidence_bins.csv", "policies": "source_threshold_policies.csv"}
    frames = {}
    for key, filename in names.items():
        frames[key] = pd.DataFrame([row for part in parts for row in part[key]])
        frames[key].to_csv(out / filename, index=False)
    conflicts = samples[samples.conflict].copy()
    conflicts["outcome"] = np.where(conflicts.mid_correct, "mid_wins",
                                  np.where(conflicts.baseline_correct, "baseline_wins", "both_wrong"))
    conflicts.to_csv(out / "conflicts.csv", index=False)
    lines = ["STEP 2: CONFLICT CONFIDENCE", "",
        "Confidence = top1 minus top2 cosine; NOT a calibrated probability.",
        "AUC > .5 means higher gap ranks middle-wins above the comparison group.",
        "The win-vs-win AUC excludes both-wrong; the second AUC includes them.",
        "Thresholds come from Synthetic OOF gaps only; every threshold is reported.",
        "These policies replace answers, NOT actual updates. Do not report them as HOW steering.", ""]
    for row in primary:
        r, L = row["representation"], row["layer"]
        lines.append(f"{r} PRIMARY L{L}")
        for key, cols in [
            ("auc", ["comparison", "N", "positive_N", "negative_N", "auc"]),
            ("distributions", ["outcome", "N", "mean", "q25", "q50", "q75"]),
            ("policies", ["policy", "threshold", "triggered_N", "W2C", "C2W", "net", "final_accuracy"])]:
            table = frames[key]
            selected = table[(table.representation == r) & (table.layer == L)]
            lines.append(selected[cols].to_string(index=False))
        lines.append("")
    lines += ["STEP 3: FOLLOW-UP DECISION (no automatic target-tuned gate)",
        "First check that the same-question middle readout still repairs baseline errors.",
        "Then inspect win/loss confidence distributions, both AUCs, and all source-threshold policies.",
        "A favorable result motivates a separate soft-evidence-to-update experiment; it does not prove polarity.",
        "per_sample_layer.csv retains four continuous cosine scores for that follow-up.",
        "No threshold, layer, softmax temperature or update policy is optimized on these target labels."]
    report = "\n".join(lines) + "\n"
    (out / "confidence_summary.txt").write_text(report, encoding="utf-8")
    return report


def main():
    a = parse_args()
    out = Path(a.output_dir)
    output_names = ["baseline_generated.csv", "baseline_aligned.csv", "per_sample_layer.csv",
        "summary_by_layer.csv", "summary_by_relation.csv", "source_oof_layer_accuracy.csv",
        "primary_summary.csv", "analysis_summary.txt", "metadata.json",
        "confidence_auc.csv", "confidence_distributions.csv", "confidence_bins.csv",
        "source_threshold_policies.csv", "conflicts.csv", "confidence_summary.txt"]
    if not a.overwrite and any((out / n).exists() for n in output_names):
        raise FileExistsError(f"Existing diagnostic outputs in {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    source, target, bp, sp, tp = obtain_inputs(a, out)
    ss, sy, si, sn = source
    ts, ty, ti, tn = target
    layers = layer_list(a.analysis_layers, si.shape[1])
    bp_pred, bp_ok, join_audit = load_baseline(bp, ts, ty)
    pd.DataFrame(dict(sid=ts, gt=ty, baseline_prediction=bp_pred,
                     baseline_correct=bp_ok)).to_csv(out / "baseline_aligned.csv", index=False)
    summary, by_rel, oof_rows, sample_parts, primary = [], [], [], [], []
    confidence_parts = []
    for representation in ("residual", "img", "no_image"):
        if representation == "residual":
            X = si[:, layers].astype(np.float32) - sn[:, layers].astype(np.float32)
            Y = ti[:, layers].astype(np.float32) - tn[:, layers].astype(np.float32)
        else:
            X = (si if representation == "img" else sn)[:, layers].astype(np.float32)
            Y = (ti if representation == "img" else tn)[:, layers].astype(np.float32)
        acc, oof_scores = source_oof(X, sy, a.cv_folds, a.seed, return_scores=True)
        chosen = layers[int(np.argmax(acc))] if a.primary_layer == "source_oof" else int(a.primary_layer)
        if chosen not in layers:
            raise ValueError("primary-layer must be in analysis-layers")
        scores = score(Y, *fit(X, sy))
        for j, L in enumerate(layers):
            preds = np.asarray(REL)[scores[:, j].argmax(axis=-1)]
            sorted_scores = np.sort(scores[:, j], axis=-1)
            df = pd.DataFrame(dict(sid=ts, gt=ty, representation=representation, layer=L,
                baseline_prediction=bp_pred, baseline_correct=bp_ok, mid_prediction=preds,
                mid_correct=preds == ty, conflict=preds != bp_pred,
                mid_top2_cosine_gap=sorted_scores[:, -1] - sorted_scores[:, -2]))
            for k, r in enumerate(REL):
                df[f"mid_score_{r}"] = scores[:, j, k]
            confidence_parts.append(confidence_analysis(df, oof_scores[:, j]))
            row = dict(representation=representation, layer=L,
                       source_oof_accuracy=float(acc[j]), **summarize(df))
            summary.append(row)
            if L == chosen:
                primary.append(dict(selection_rule=a.primary_layer, **row))
            for r in REL:
                by_rel.append(dict(representation=representation, layer=L, relation=r,
                                   **summarize(df[df["gt"] == r])))
            oof_rows.append(dict(representation=representation, layer=L,
                                 source_oof_accuracy=float(acc[j]), selected_primary=L == chosen))
            sample_parts.append(df)
    all_samples = pd.concat(sample_parts, ignore_index=True)
    all_samples.to_csv(out / "per_sample_layer.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary_by_layer.csv", index=False)
    pd.DataFrame(by_rel).to_csv(out / "summary_by_relation.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "source_oof_layer_accuracy.csv", index=False)
    pd.DataFrame(primary).to_csv(out / "primary_summary.csv", index=False)
    protocol = "same_readout_question_free_generation"
    confidence_report = save_confidence(confidence_parts, all_samples, out, primary)
    lines = ["MIDDLE OBJECT STATE vs FINAL FREE GENERATION", "",
        f"Source N={len(ss)}; target N={len(ts)}; zero-based block outputs={layers}",
        f"Baseline protocol: {protocol}",
        "Source-only labeled centered-cosine codebook; target labels are evaluation-only.",
        "Primary layer: source OOF or predeclared. Target layer scan is exploratory.",
        "No intervention: W2C/C2W describe replacing the answer with the readout.",
        "oracle_union_accuracy is an unavailable GT-dependent ceiling, not a policy.",
        "Conflicts include BOTH-WRONG cases; invalid generations count as baseline wrong.", ""]
    for r in primary:
        lines += [f"{r['representation']} PRIMARY L{r['layer']} (source OOF={r['source_oof_accuracy']:.4f})",
            f"  baseline={r['baseline_accuracy']:.4f}; middle={r['mid_accuracy']:.4f}",
            f"  baseline-wrong N={r['baseline_wrong_N']}; middle accuracy={r['mid_accuracy_on_baseline_wrong']:.4f} "
            f"95% CI=[{r['wrong_acc_ci95_low']:.4f},{r['wrong_acc_ci95_high']:.4f}]",
            f"  conflicts={r['conflict_N']}: mid wins={r['mid_wins']}, baseline wins={r['baseline_wins']}, "
            f"both wrong={r['conflict_both_wrong']}",
            f"  W2C={r['W2C']} C2W={r['C2W']} net={r['net']:+d}; gain={r['accuracy_gain']:+.4f}", ""]
    lines += ["ALL LAYERS (do not select the target peak as a validated method)",
              pd.DataFrame(summary)[["representation", "layer", "mid_accuracy", "mid_accuracy_on_baseline_wrong",
                  "conflict_N", "mid_wins", "baseline_wins", "conflict_both_wrong", "net"]].to_string(index=False)]
    report = "\n".join(lines) + "\n\n" + confidence_report
    (out / "analysis_summary.txt").write_text(report, encoding="utf-8")
    metadata = dict(args=vars(a), join_audit=join_audit, baseline_protocol=protocol,
        layer_indexing="zero-based decoder block output", source_N=len(ss), target_N=len(ts),
        source_class_counts={r: int(np.sum(sy == r)) for r in REL},
        target_class_counts={r: int(np.sum(ty == r)) for r in REL},
        inputs={str(p.resolve()): sha256(p) for p in [sp, tp, bp]},
        notes=["No WHERE/HOW oracle used; same-question baseline only.",
               "No target label used for codebook fitting or automatic primary layer choice.",
               "Source OOF used for layer selection is not an unbiased selected-layer source estimate.",
               "Paired p-values for layer scans are unadjusted and exploratory.",
               "Confidence policies use only source OOF gap quantiles, never target GT.",
               "Report all policies; selecting the best on these targets needs new held-out validation.",
               "Cosine gap is a ranking signal, not calibrated probability."])
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report, flush=True)
    print(f"Saved: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
