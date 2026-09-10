#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_qwen_oracle_rank_marginal_v1.py

Question
--------
Does the magnitude/rank of the oracle writer contribution score M actually
predict behavioral usefulness?

For each sample:

  1. Rebuild the SAME positive global_unique K36 ranking:
         M(L,p) = (h_real - h_gray)^T grad_h J_writer

  2. Sort:
         s1, s2, ..., s36     with M1 >= M2 >= ...

  3. Run actual model.generate() for every nested prefix:
         K1, K2, ..., K_R

  4. Attribute the SEQUENTIAL marginal effect of rank r as:
         behavior(K_r) - behavior(K_{r-1})

     So if adding rank 6 changes wrong -> correct, rank 6 receives a +1
     behavioral marginal event for that sample. If correct -> wrong, -1.

This is designed to test whether the long K36 tail is behaviorally redundant /
low-value and whether larger M states tend to have larger actual marginal effect.

Why sequential marginal instead of single-token-only?
------------------------------------------------------
The current question is specifically about shrinking K36. Sequential marginal
asks whether each lower-ranked state adds anything AFTER all stronger states are
already present. That is the most direct redundancy test.

Important:
- This is still an ORACLE diagnostic because GT relation selects the writer.
- Accuracy transitions are discrete and non-additive.
- Therefore the script reports both:
    (a) actual generation W2C/C2W marginal events, and
    (b) continuous late-writer projection marginal gains.
- Pooled raw M correlations can be distorted by sample-to-sample scale.
  `M/M1` and rank-based summaries are the primary quantities.

Default:
  ranking_max=36
  test_rank_max=10

This requires 10 edited generations per eval sample. Increase test_rank_max to
20 or 36 if you want to inspect the tail more deeply.

Example
-------
python eval_qwen_oracle_rank_marginal_v1.py \
  --model qwen-3b \
  --source-layers 20,21,22,23,24,25,26 \
  --target-layers 32,34,35 \
  --ranking-max 36 \
  --test-rank-max 10 \
  --alpha 1.0 \
  --eval-scope all_data \
  --eval-max-samples 0 \
  --output-dir outputs/oracle_rank_marginal_v1_r10 \
  --overwrite
"""

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
from typing import List

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj
import eval_qwen_dynamic_k24_all440_v1 as dyn


REL = ("left", "right", "above", "below")
DISPLAY = {"left":"left", "right":"right", "above":"on", "below":"under"}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b","qwen-7b"])
    p.add_argument("--source-layers", default="20,21,22,23,24,25,26")
    p.add_argument("--target-layers", default="32,34,35")

    p.add_argument(
        "--ranking-max", type=int, default=36,
        help="Build the full ranked causal set to this size."
    )
    p.add_argument(
        "--test-rank-max", type=int, default=10,
        help="Run actual nested-prefix generation K1..K_this_value."
    )
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--writer-mode", default="centered", choices=["centered","raw"])
    p.add_argument("--train-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-scope", default="all_data", choices=["test","all_data"])
    p.add_argument(
        "--eval-max-samples", type=int, default=0,
        help="0 = all samples in eval scope."
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl", default="eager",
        choices=["eager","sdpa","flash_attention_2","none"]
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_ints(s: str) -> List[int]:
    return sorted({
        int(x.strip().upper().replace("L",""))
        for x in str(s).split(",") if x.strip()
    })


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
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_median(xs):
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def normalize_np(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v.copy() if n < EPS else (v/n).astype(np.float32)


def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def safe_corr(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def safe_spearman(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    return safe_corr(rankdata(x), rankdata(y))


def build_ranked_global_unique(rows_by_layer, max_rank):
    best_by_pos = {}
    for L, rows in rows_by_layer.items():
        for row in rows:
            m = float(row["mediation"])
            if not np.isfinite(m) or m <= 0:
                continue
            rr = dict(row)
            rr["source_layer"] = int(L)
            pos = int(rr["position"])
            if pos not in best_by_pos or m > float(best_by_pos[pos]["mediation"]):
                best_by_pos[pos] = rr

    ranked = sorted(
        best_by_pos.values(),
        key=lambda x: float(x["mediation"]),
        reverse=True
    )
    return ranked[:int(max_rank)]


def prefix_specs(ranked, k):
    use = ranked[:min(int(k),len(ranked))]
    specs = defaultdict(list)
    for r in use:
        specs[int(r["source_layer"])].append(
            (int(r["position"]), np.asarray(r["delta_h"],np.float32))
        )
    return dict(specs), use


def aggregate_rank_summary(effect_rows, baseline_rows, test_rank_max):
    base_by_sid = {int(r["sid"]):r for r in baseline_rows}
    base_acc = safe_mean(float(r["baseline_correct"]) for r in baseline_rows)
    wrong_N = sum(not bool(r["baseline_correct"]) for r in baseline_rows)
    correct_N = sum(bool(r["baseline_correct"]) for r in baseline_rows)

    out = []
    for rank in range(1,int(test_rank_max)+1):
        rr = [r for r in effect_rows if int(r["rank"]) == rank]
        if not rr:
            continue

        # Incremental transition from K_(r-1) -> K_r.
        inc_w2c = sum(bool(r["incremental_W2C"]) for r in rr)
        inc_c2w = sum(bool(r["incremental_C2W"]) for r in rr)

        # Cumulative status relative to baseline.
        cum_w2c = sum(
            (not bool(base_by_sid[int(r["sid"])]["baseline_correct"]))
            and bool(r["current_correct"])
            for r in rr
        )
        cum_c2w = sum(
            bool(base_by_sid[int(r["sid"])]["baseline_correct"])
            and (not bool(r["current_correct"]))
            for r in rr
        )
        cum_acc = safe_mean(float(r["current_correct"]) for r in rr)

        eligible_prev_wrong = sum(not bool(r["previous_correct"]) for r in rr)
        eligible_prev_correct = sum(bool(r["previous_correct"]) for r in rr)

        out.append({
            "rank":rank,
            "N":len(rr),
            "mean_M":safe_mean(r["mediation"] for r in rr),
            "median_M":safe_median(r["mediation"] for r in rr),
            "mean_M_over_M1":safe_mean(r["M_over_M1"] for r in rr),
            "mean_individual_K36_mass_share":safe_mean(r["K36_mass_share"] for r in rr),
            "mean_cumulative_K36_mass_fraction":safe_mean(
                r["cumulative_K36_mass_fraction"] for r in rr
            ),
            "incremental_W2C":inc_w2c,
            "incremental_C2W":inc_c2w,
            "incremental_net":inc_w2c-inc_c2w,
            "prev_wrong_N":eligible_prev_wrong,
            "conditional_incremental_repair_rate":(
                inc_w2c/eligible_prev_wrong if eligible_prev_wrong else float("nan")
            ),
            "prev_correct_N":eligible_prev_correct,
            "conditional_incremental_break_rate":(
                inc_c2w/eligible_prev_correct if eligible_prev_correct else float("nan")
            ),
            "mean_incremental_writer_projection":safe_mean(
                r["marginal_writer_projection"] for r in rr
            ),
            "median_incremental_writer_projection":safe_median(
                r["marginal_writer_projection"] for r in rr
            ),
            "baseline_acc":base_acc,
            "cumulative_acc":cum_acc,
            "cumulative_gain":cum_acc-base_acc,
            "cumulative_W2C_vs_baseline":cum_w2c,
            "cumulative_C2W_vs_baseline":cum_c2w,
            "cumulative_net_vs_baseline":cum_w2c-cum_c2w,
            "baseline_wrong_N":wrong_N,
            "baseline_correct_N":correct_N,
        })
    return out


def correlation_summary(effect_rows):
    rows = [
        r for r in effect_rows
        if np.isfinite(float(r["mediation"]))
        and np.isfinite(float(r["M_over_M1"]))
        and np.isfinite(float(r["marginal_writer_projection"]))
    ]

    rawM = [r["mediation"] for r in rows]
    relM = [r["M_over_M1"] for r in rows]
    ranks = [r["rank"] for r in rows]
    dwrite = [r["marginal_writer_projection"] for r in rows]
    dcorrect = [r["marginal_correctness"] for r in rows]

    # Only rows where previous prefix is currently wrong: can this newly added
    # state repair the sample?
    prev_wrong = [r for r in rows if not bool(r["previous_correct"])]
    pw_relM = [r["M_over_M1"] for r in prev_wrong]
    pw_rawM = [r["mediation"] for r in prev_wrong]
    pw_rank = [r["rank"] for r in prev_wrong]
    pw_repair = [float(bool(r["incremental_W2C"])) for r in prev_wrong]

    return [
        {
            "scope":"all_prefix_steps",
            "N":len(rows),
            "pearson_rawM_vs_marginal_writer":safe_corr(rawM,dwrite),
            "spearman_rawM_vs_marginal_writer":safe_spearman(rawM,dwrite),
            "pearson_relM_vs_marginal_writer":safe_corr(relM,dwrite),
            "spearman_relM_vs_marginal_writer":safe_spearman(relM,dwrite),
            "spearman_rank_vs_marginal_writer":safe_spearman(ranks,dwrite),
            "pearson_relM_vs_marginal_correctness":safe_corr(relM,dcorrect),
            "spearman_relM_vs_marginal_correctness":safe_spearman(relM,dcorrect),
        },
        {
            "scope":"previous_prefix_wrong_only",
            "N":len(prev_wrong),
            "pearson_rawM_vs_incremental_repair":safe_corr(pw_rawM,pw_repair),
            "spearman_rawM_vs_incremental_repair":safe_spearman(pw_rawM,pw_repair),
            "pearson_relM_vs_incremental_repair":safe_corr(pw_relM,pw_repair),
            "spearman_relM_vs_incremental_repair":safe_spearman(pw_relM,pw_repair),
            # Negative is expected if earlier ranks are more useful.
            "spearman_rank_vs_incremental_repair":safe_spearman(pw_rank,pw_repair),
        },
    ]


def magnitude_bins(effect_rows, n_bins=5):
    """
    Pool normalized M/M1 and bin into quantiles. For behavior, only count
    prefix steps whose previous state is wrong, because those steps are eligible
    to repair.
    """
    rr = [
        r for r in effect_rows
        if not bool(r["previous_correct"])
        and np.isfinite(float(r["M_over_M1"]))
    ]
    if len(rr) < n_bins:
        return []

    vals = np.asarray([float(r["M_over_M1"]) for r in rr],dtype=np.float64)
    edges = np.quantile(vals,np.linspace(0,1,n_bins+1))

    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b+1]
        if b == n_bins-1:
            sel = [r for r in rr if lo <= float(r["M_over_M1"]) <= hi]
        else:
            sel = [r for r in rr if lo <= float(r["M_over_M1"]) < hi]
        if not sel:
            continue
        out.append({
            "bin":b+1,
            "M_over_M1_low":float(lo),
            "M_over_M1_high":float(hi),
            "N":len(sel),
            "mean_M_over_M1":safe_mean(r["M_over_M1"] for r in sel),
            "incremental_repairs":sum(bool(r["incremental_W2C"]) for r in sel),
            "incremental_repair_rate":safe_mean(
                float(bool(r["incremental_W2C"])) for r in sel
            ),
            "mean_marginal_writer_projection":safe_mean(
                r["marginal_writer_projection"] for r in sel
            ),
        })
    return out


def main():
    a = parse_args()

    if a.test_rank_max > a.ranking_max:
        raise ValueError("--test-rank-max cannot exceed --ranking-max")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    source_layers = parse_ints(a.source_layers)
    targets = parse_ints(a.target_layers)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records,_audit = two.load_records("coco_two",Path(a.data_root),None)
    rec_by_sid = {int(r.sid):r for r in records}

    meta = []
    for rec in records:
        sid = int(rec.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base,p["answer_raw"])
        if gt not in REL:
            continue
        meta.append({
            "sid":sid,
            "gt":gt,
            "subject":str(p["subject"]),
            "reference":str(p["reference"]),
            "question_text":str(p["question_text"]),
        })

    meta = traj.stratified_cap(meta,a.max_samples,a.seed)
    train,heldout = traj.stratified_split(meta,a.train_ratio,a.seed)
    test = list(meta) if a.eval_scope=="all_data" else list(heldout)
    if int(a.eval_max_samples) > 0:
        test = traj.stratified_cap(test,a.eval_max_samples,a.seed+1)

    train_sids = {int(x["sid"]) for x in train}
    test_sids = {int(x["sid"]) for x in test}
    overlap = len(train_sids & test_sids)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers,spec.model_class)

    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"":a.device},
    )
    if a.attn_impl != "none":
        kw["attn_implementation"] = a.attn_impl

    model = processor = None

    try:
        print(f"Loading {spec.repo_id}",flush=True)
        model = cls.from_pretrained(spec.repo_id,**kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,trust_remote_code=spec.trust_remote_code
        )
        base.configure_processor(model,processor)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder_layers,decoder_path = base.resolve_decoder_layers(model)
        graph_layers = sorted(set(source_layers+targets))
        cut = min(source_layers)
        device = torch.device(a.device)

        print("="*138)
        print("ORACLE RANK MAGNITUDE -> SEQUENTIAL MARGINAL BEHAVIOR")
        print("="*138)
        print(f"model={a.model} repo={spec.repo_id}")
        print(f"decoder={decoder_path}")
        print(
            f"writer calibration N={len(train)} | eval={a.eval_scope} N={len(test)} "
            f"| calibration/eval overlap={overlap}"
        )
        if overlap:
            print("NOTE: ORACLE mechanistic diagnostic; calibration/eval overlap exists.")
        print(f"source={source_layers} writer targets={targets}")
        print(
            f"ranking_max={a.ranking_max} | actual prefixes K1..K{a.test_rank_max} "
            f"| alpha={a.alpha}"
        )
        print()

        # ---------------------------------------------------------
        # 1) Calibrate late writer directions.
        # ---------------------------------------------------------
        q_by_sid = {}
        for m in tqdm(train,desc="CALIBRATE writers"):
            sid = int(m["sid"])
            real = gray = rb = gb = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real,"convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real,a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,image=real,
                    question_text=m["question_text"],device=device
                )
                gb = base.make_question_batch(
                    processor=processor,image=gray,
                    question_text=m["question_text"],device=device
                )

                hr = dyn.capture_cpu(model,decoder_layers,rb,targets)
                hg = dyn.capture_cpu(model,decoder_layers,gb,targets)
                q_by_sid[sid] = {
                    T:(hr[T][0,-1]-hg[T][0,-1]).astype(np.float32)
                    for T in targets
                }
            finally:
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()

        writers,writer_geom = dyn.learn_writers(
            train,q_by_sid,targets,a.writer_mode
        )
        write_csv(outdir/"writer_geometry.csv",writer_geom)

        # ---------------------------------------------------------
        # 2) Rank K36 and run every nested prefix.
        # ---------------------------------------------------------
        baseline_rows = []
        ranking_rows = []
        effect_rows = []

        for m in tqdm(test,desc=f"EVAL K1..K{a.test_rank_max}"):
            sid = int(m["sid"])
            gt = m["gt"]
            writers_r = {T:writers[T][gt] for T in targets}

            real = gray = rb = gb = cap = None
            try:
                real = base.record_image(rec_by_sid[sid])
                if hasattr(real,"convert"):
                    real = real.convert("RGB")
                gray = dyn.make_gray_image(real,a.gray_value)

                rb = base.make_question_batch(
                    processor=processor,image=real,
                    question_text=m["question_text"],device=device
                )
                gb = base.make_question_batch(
                    processor=processor,image=gray,
                    question_text=m["question_text"],device=device
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                cats,toks = dyn.build_categories(
                    model,processor,rb,ids,m["subject"],m["reference"]
                )

                clean = dyn.run_generation_with_hooks(
                    model,processor,decoder_layers,rb,targets,writers_r,
                    a.max_new_tokens
                )
                base_pred = clean["prediction"]
                base_correct = base_pred == gt

                baseline_rows.append({
                    "sid":sid,
                    "gt":DISPLAY[gt],
                    "baseline_prediction":DISPLAY.get(base_pred,base_pred),
                    "baseline_correct":base_correct,
                    "baseline_text":clean["text"],
                    "baseline_mean_projection":clean["mean_projection"],
                })

                hgray = dyn.capture_cpu(
                    model,decoder_layers,gb,source_layers
                )

                with torch.enable_grad():
                    cap = dyn.forward_graph(
                        model,decoder_layers,rb,graph_layers,cut
                    )
                    terms = []
                    for T in targets:
                        shat = torch.as_tensor(
                            normalize_np(writers_r[T]),
                            device=cap.states[T].device,
                            dtype=torch.float32,
                        )
                        terms.append(
                            torch.dot(cap.states[T][0,-1].float(),shat)
                        )
                    J = torch.stack(terms).sum()

                    grads = torch.autograd.grad(
                        J,[cap.states[S] for S in source_layers],
                        retain_graph=False,create_graph=False,allow_unused=False
                    )

                    rows_by_layer = {}
                    for S,g in zip(source_layers,grads):
                        Hr = (
                            cap.states[S][0].detach().float().cpu().numpy()
                            .astype(np.float32)
                        )
                        Hg = hgray[S][0].astype(np.float32)
                        G = (
                            g[0].detach().float().cpu().numpy()
                            .astype(np.float32)
                        )
                        npos = min(
                            len(ids),len(cats),len(toks),
                            Hr.shape[0],Hg.shape[0],G.shape[0]
                        )
                        rr = []
                        for pos in range(max(0,npos-1)):
                            delta = (Hr[pos]-Hg[pos]).astype(np.float32)
                            med = float(np.dot(delta,G[pos]))
                            rr.append({
                                "source_layer":S,
                                "position":pos,
                                "token_id":int(ids[pos]),
                                "token":str(toks[pos]).replace("\n","\\n"),
                                "category":cats[pos],
                                "broad_category":dyn.broad_category(cats[pos]),
                                "mediation":med,
                                "delta_h":delta,
                                "delta_h_norm":float(np.linalg.norm(delta)),
                                "grad_norm":float(np.linalg.norm(G[pos])),
                            })
                        rows_by_layer[S] = rr

                cap.close()
                cap = None

                ranked = build_ranked_global_unique(
                    rows_by_layer,a.ranking_max
                )
                if not ranked:
                    continue

                M1 = float(ranked[0]["mediation"])
                totalM = sum(float(r["mediation"]) for r in ranked)
                cumM = 0.0

                for rank,r in enumerate(ranked,1):
                    cumM += float(r["mediation"])
                    ranking_rows.append({
                        "sid":sid,
                        "gt":DISPLAY[gt],
                        "rank":rank,
                        "source_layer":int(r["source_layer"]),
                        "position":int(r["position"]),
                        "token_id":int(r["token_id"]),
                        "token":r["token"],
                        "category":r["category"],
                        "broad_category":r["broad_category"],
                        "mediation":float(r["mediation"]),
                        "M_over_M1":(
                            float(r["mediation"])/M1 if M1 > EPS else float("nan")
                        ),
                        "K36_mass_share":(
                            float(r["mediation"])/totalM
                            if totalM > EPS else float("nan")
                        ),
                        "cumulative_K36_mass_fraction":(
                            cumM/totalM if totalM > EPS else float("nan")
                        ),
                        "delta_h_norm":float(r["delta_h_norm"]),
                        "grad_norm":float(r["grad_norm"]),
                    })

                previous_pred = base_pred
                previous_correct = base_correct
                previous_proj = float(clean["mean_projection"])

                cumM = 0.0
                for rank in range(1,min(a.test_rank_max,len(ranked))+1):
                    r = ranked[rank-1]
                    cumM += float(r["mediation"])
                    specs_prefix,selected = prefix_specs(ranked,rank)

                    edited = dyn.run_generation_with_hooks(
                        model,processor,decoder_layers,rb,targets,writers_r,
                        a.max_new_tokens,
                        token_specs=specs_prefix,
                        token_alpha=a.alpha,
                    )
                    pred = edited["prediction"]
                    correct = pred == gt
                    proj = float(edited["mean_projection"])

                    effect_rows.append({
                        "sid":sid,
                        "gt":DISPLAY[gt],
                        "rank":rank,
                        "source_layer":int(r["source_layer"]),
                        "position":int(r["position"]),
                        "token":r["token"],
                        "category":r["category"],
                        "broad_category":r["broad_category"],
                        "mediation":float(r["mediation"]),
                        "M_over_M1":(
                            float(r["mediation"])/M1 if M1 > EPS else float("nan")
                        ),
                        "K36_mass_share":(
                            float(r["mediation"])/totalM
                            if totalM > EPS else float("nan")
                        ),
                        "cumulative_K36_mass_fraction":(
                            cumM/totalM if totalM > EPS else float("nan")
                        ),
                        "baseline_correct":base_correct,
                        "previous_prediction":DISPLAY.get(previous_pred,previous_pred),
                        "previous_correct":previous_correct,
                        "current_prediction":DISPLAY.get(pred,pred),
                        "current_correct":correct,
                        "marginal_correctness":int(correct)-int(previous_correct),
                        "incremental_W2C":(
                            (not previous_correct) and correct
                        ),
                        "incremental_C2W":(
                            previous_correct and (not correct)
                        ),
                        "previous_writer_projection":previous_proj,
                        "current_writer_projection":proj,
                        "marginal_writer_projection":proj-previous_proj,
                        "prefix_size":len(selected),
                    })

                    previous_pred = pred
                    previous_correct = correct
                    previous_proj = proj

            finally:
                if cap is not None:
                    cap.close()
                if real is not None:
                    with contextlib.suppress(Exception): real.close()
                if gray is not None:
                    with contextlib.suppress(Exception): gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        rank_summary = aggregate_rank_summary(
            effect_rows,baseline_rows,a.test_rank_max
        )
        corr_summary = correlation_summary(effect_rows)
        bins = magnitude_bins(effect_rows,n_bins=5)

        # First-repair-rank distribution for originally wrong samples.
        effects_by_sid = defaultdict(list)
        for r in effect_rows:
            effects_by_sid[int(r["sid"])].append(r)

        first_repairs = []
        for b in baseline_rows:
            sid = int(b["sid"])
            if bool(b["baseline_correct"]):
                continue
            rr = sorted(effects_by_sid.get(sid,[]),key=lambda x:int(x["rank"]))
            first = None
            for r in rr:
                if bool(r["current_correct"]):
                    first = int(r["rank"])
                    break
            first_repairs.append({
                "sid":sid,
                "first_repair_rank":(
                    first if first is not None else f">{a.test_rank_max}"
                ),
            })

        write_csv(outdir/"baseline.csv",baseline_rows)
        write_csv(outdir/"ranked_k36_tokens.csv",ranking_rows)
        write_csv(outdir/"prefix_marginal_effects.csv",effect_rows)
        write_csv(outdir/"rank_summary.csv",rank_summary)
        write_csv(outdir/"correlation_summary.csv",corr_summary)
        write_csv(outdir/"magnitude_bins.csv",bins)
        write_csv(outdir/"first_repair_rank.csv",first_repairs)

        metadata = {
            "model":a.model,
            "repo_id":spec.repo_id,
            "source_layers":source_layers,
            "target_layers":targets,
            "ranking_max":a.ranking_max,
            "test_rank_max":a.test_rank_max,
            "alpha":a.alpha,
            "writer_mode":a.writer_mode,
            "writer_calibration_N":len(train),
            "eval_scope":a.eval_scope,
            "eval_N":len(test),
            "calibration_eval_overlap":overlap,
            "oracle_relation_used":True,
            "selection":(
                "positive global_unique ranked by "
                "M=(h_real-h_gray)^T grad J_writer"
            ),
            "behavioral_effect":(
                "sequential nested-prefix marginal K_r - K_(r-1)"
            ),
        }
        (outdir/"run_metadata.json").write_text(
            json.dumps(metadata,indent=2),encoding="utf-8"
        )

        print("\n"+"="*138)
        print("RANK -> ACTUAL SEQUENTIAL MARGINAL EFFECT")
        print("="*138)
        print(
            f"{'r':>3s} | {'M/M1':>7s} | {'Mshare':>7s} | {'cumMass':>7s} | "
            f"{'incW2C':>6s} | {'incC2W':>6s} | {'net':>5s} | "
            f"{'condRepair':>10s} | {'Δwriter':>9s} | {'cumAcc':>7s}"
        )
        print("-"*116)

        for r in rank_summary:
            print(
                f"{int(r['rank']):3d} | "
                f"{float(r['mean_M_over_M1']):7.3f} | "
                f"{float(r['mean_individual_K36_mass_share']):7.3f} | "
                f"{float(r['mean_cumulative_K36_mass_fraction']):7.3f} | "
                f"{int(r['incremental_W2C']):6d} | "
                f"{int(r['incremental_C2W']):6d} | "
                f"{int(r['incremental_net']):5d} | "
                f"{float(r['conditional_incremental_repair_rate']):10.3f} | "
                f"{float(r['mean_incremental_writer_projection']):+9.3f} | "
                f"{float(r['cumulative_acc']):7.4f}"
            )

        print("\n"+"="*138)
        print("CORRELATION: M MAGNITUDE vs ACTUAL MARGINAL EFFECT")
        print("="*138)
        for r in corr_summary:
            print(r["scope"])
            for k,v in r.items():
                if k in ("scope","N"):
                    continue
                print(f"  {k:<46s} {float(v):+.4f}")
            print(f"  {'N':<46s} {int(r['N'])}")

        print("\n"+"="*138)
        print("M/M1 QUINTILES — ONLY STEPS WHOSE PREVIOUS PREFIX IS WRONG")
        print("="*138)
        for r in bins:
            print(
                f"bin{int(r['bin'])}: M/M1 [{r['M_over_M1_low']:.3f},"
                f"{r['M_over_M1_high']:.3f}] N={int(r['N'])} "
                f"repair={r['incremental_repair_rate']:.4f} "
                f"Δwriter={r['mean_marginal_writer_projection']:+.3f}"
            )

        print("\nHow to judge:")
        print(
            "  1) If M/M1 falls with rank AND incremental W2C / Δwriter also falls, "
            "larger M is tracking actual usefulness."
        )
        print(
            "  2) If early ranks produce most incremental W2C while later ranks are "
            "mostly zero, the K36 tail is behaviorally redundant/low-value."
        )
        print(
            "  3) `spearman_relM_vs_incremental_repair > 0` and "
            "`spearman_rank_vs_incremental_repair < 0` support magnitude ordering."
        )
        print(
            "  4) Continuous `relM vs marginal_writer` should usually be easier to "
            "interpret than correlation with discrete correctness flips."
        )
        print("\nSaved:",outdir)

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
