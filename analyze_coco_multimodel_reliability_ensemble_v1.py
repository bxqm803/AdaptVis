#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-model correctness overlap and detector-guided ensemble analysis."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

RELATIONS: Tuple[str, ...] = ("left", "right", "above", "below")
REL_TO_ID = {name: i for i, name in enumerate(RELATIONS)}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--detector-root",
        default="output/coco_all_relation_head_prototype_detector",
    )
    p.add_argument(
        "--models",
        default="auto",
        help="Comma list, or auto to discover completed detector directories.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--scope", choices=("intersection", "available", "both"), default="both")
    p.add_argument("--run-stacking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stacking-folds", type=int, default=5)
    p.add_argument("--stacking-repeats", type=int, default=5)
    p.add_argument("--stacking-c", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=97)
    p.add_argument(
        "--output-dir",
        default="output/coco_multimodel_reliability_ensemble",
    )
    return p.parse_args()


def normalize_relation(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    return {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
        "on": "above",
        "under": "below",
    }.get(text)


def first_existing(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def discover_models(root: Path, spec: str) -> List[str]:
    if spec.strip().lower() != "auto":
        return [x.strip() for x in spec.split(",") if x.strip()]
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "oof_predictions.csv").exists()
    )


def read_model(root: Path, model: str, threshold: float) -> pd.DataFrame:
    path = root / model / "oof_predictions.csv"
    frame = pd.read_csv(path)

    sid_col = first_existing(frame, ("sid", "sample_index"))
    gt_col = first_existing(frame, ("gt", "relation", "ground_truth"))
    pred_col = first_existing(frame, ("model_prediction", "baseline_prediction", "prediction"))
    error_col = first_existing(frame, ("error_label",))
    correct_col = first_existing(frame, ("baseline_correct", "generation_correct"))
    p_col = "error_probability__prototype_head"

    missing = []
    if sid_col is None: missing.append("sid")
    if gt_col is None: missing.append("gt")
    if pred_col is None: missing.append("prediction")
    if error_col is None and correct_col is None: missing.append("error_label/baseline_correct")
    if p_col not in frame.columns: missing.append(p_col)
    if missing:
        raise KeyError(f"{path}: missing {missing}; columns={frame.columns.tolist()}")

    out = pd.DataFrame({
        "sid": frame[sid_col].astype(int),
        "gt": frame[gt_col].map(normalize_relation),
        "prediction": frame[pred_col].map(normalize_relation),
        "p_error": pd.to_numeric(frame[p_col], errors="coerce"),
    })
    if error_col is not None:
        out["error"] = frame[error_col].astype(int)
    else:
        out["error"] = (~frame[correct_col].astype(bool)).astype(int)

    out = out.dropna(subset=["gt", "prediction", "p_error"]).copy()
    out["correct"] = out["prediction"] == out["gt"]
    out["predicted_unreliable"] = out["p_error"] >= threshold
    if not np.all(out["error"].to_numpy() == (~out["correct"].to_numpy()).astype(int)):
        raise RuntimeError(f"{model}: error_label disagrees with prediction==GT")
    if out["sid"].duplicated().any():
        raise RuntimeError(f"{model}: duplicate SIDs")
    return out.sort_values("sid").reset_index(drop=True)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pairwise_table(models: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    names = list(models)
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            a = models[a_name].rename(columns={
                "gt": "gt_a", "prediction": "pred_a", "correct": "correct_a",
                "p_error": "risk_a", "predicted_unreliable": "unrel_a",
            })
            b = models[b_name].rename(columns={
                "gt": "gt_b", "prediction": "pred_b", "correct": "correct_b",
                "p_error": "risk_b", "predicted_unreliable": "unrel_b",
            })
            m = a[["sid", "gt_a", "pred_a", "correct_a", "risk_a", "unrel_a"]].merge(
                b[["sid", "gt_b", "pred_b", "correct_b", "risk_b", "unrel_b"]],
                on="sid", how="inner", validate="one_to_one",
            )
            if m.empty:
                continue
            if not np.all(m["gt_a"] == m["gt_b"]):
                raise RuntimeError(f"GT mismatch: {a_name} vs {b_name}")

            ca = m["correct_a"].astype(bool).to_numpy()
            cb = m["correct_b"].astype(bool).to_numpy()
            ea, eb = ~ca, ~cb
            ua = m["unrel_a"].astype(bool).to_numpy()
            ub = m["unrel_b"].astype(bool).to_numpy()
            disagree = m["pred_a"].to_numpy() != m["pred_b"].to_numpy()

            rows.append({
                "model_a": a_name,
                "model_b": b_name,
                "N_common": len(m),
                "accuracy_a": ca.mean(),
                "accuracy_b": cb.mean(),
                "both_correct": np.logical_and(ca, cb).sum(),
                "a_only_correct": np.logical_and(ca, eb).sum(),
                "b_only_correct": np.logical_and(ea, cb).sum(),
                "both_wrong": np.logical_and(ea, eb).sum(),
                "oracle_pair_accuracy": np.logical_or(ca, cb).mean(),
                "answer_agreement": (~disagree).mean(),
                "correctness_agreement": (ca == cb).mean(),
                "actual_error_jaccard": jaccard(ea, eb),
                "actual_error_phi": corr(ea.astype(int), eb.astype(int)),
                "predicted_unreliable_jaccard": jaccard(ua, ub),
                "risk_score_spearman": corr(
                    m["risk_a"].rank().to_numpy(),
                    m["risk_b"].rank().to_numpy(),
                ),
                "both_predicted_unreliable": np.logical_and(ua, ub).sum(),
                "a_only_predicted_unreliable": np.logical_and(ua, ~ub).sum(),
                "b_only_predicted_unreliable": np.logical_and(~ua, ub).sum(),
                "both_predicted_reliable": np.logical_and(~ua, ~ub).sum(),
                "disagreement_N": disagree.sum(),
                "a_correct_when_disagree": np.logical_and(disagree, ca).sum(),
                "b_correct_when_disagree": np.logical_and(disagree, cb).sum(),
            })
    return pd.DataFrame(rows)


def build_wide(models: Dict[str, pd.DataFrame], scope: str) -> pd.DataFrame:
    sid_sets = [set(f["sid"].tolist()) for f in models.values()]
    sids = set.intersection(*sid_sets) if scope == "intersection" else set.union(*sid_sets)
    wide = pd.DataFrame({"sid": sorted(sids)})

    gt_map = {}
    for frame in models.values():
        for sid, gt in zip(frame["sid"], frame["gt"]):
            if sid in gt_map and gt_map[sid] != gt:
                raise RuntimeError(f"SID {sid}: inconsistent GT")
            gt_map[sid] = gt
    wide["gt"] = wide["sid"].map(gt_map)

    for model, frame in models.items():
        x = frame[["sid", "prediction", "correct", "p_error", "predicted_unreliable"]].rename(columns={
            "prediction": f"{model}__prediction",
            "correct": f"{model}__correct",
            "p_error": f"{model}__p_error",
            "predicted_unreliable": f"{model}__predicted_unreliable",
        })
        wide = wide.merge(x, on="sid", how="left", validate="one_to_one")
    return wide


def vote(row: pd.Series, models: Sequence[str], method: str, threshold: float):
    available = []
    for model in models:
        pred = row.get(f"{model}__prediction")
        risk = row.get(f"{model}__p_error")
        if pd.isna(pred) or pd.isna(risk):
            continue
        available.append((model, str(pred), float(risk)))
    if not available:
        raise RuntimeError(f"SID {row['sid']}: no predictions")

    if method == "min_risk_model":
        model, pred, risk = min(available, key=lambda x: (x[2], x[0]))
        return pred, model

    candidates = available
    fallback = False
    if method == "reliable_only_vote":
        reliable = [x for x in available if x[2] < threshold]
        if reliable:
            candidates = reliable
        else:
            fallback = True

    score = {r: 0.0 for r in RELATIONS}
    count = {r: 0 for r in RELATIONS}
    risk_sum = {r: 0.0 for r in RELATIONS}
    voters = {r: [] for r in RELATIONS}

    for model, pred, risk in candidates:
        count[pred] += 1
        risk_sum[pred] += risk
        voters[pred].append(model)
        if method == "majority_vote":
            score[pred] += 1.0
        else:
            score[pred] += max(1.0 - risk, EPS)

    max_score = max(score.values())
    tied = [r for r in RELATIONS if abs(score[r] - max_score) <= 1e-12]
    chosen = min(
        tied,
        key=lambda r: (risk_sum[r] / max(count[r], 1), REL_TO_ID[r]),
    )
    source = ",".join(voters[chosen])
    if fallback:
        source = "fallback_all:" + source
    return chosen, source


def metric_row(scope: str, method: str, gt: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "scope": scope,
        "method": method,
        "N": len(gt),
        "accuracy": accuracy_score(gt, pred),
        "balanced_accuracy": balanced_accuracy_score(gt, pred),
        "macro_f1": f1_score(gt, pred, labels=list(RELATIONS), average="macro", zero_division=0),
    }


def fixed_ensembles(wide: pd.DataFrame, models: Sequence[str], threshold: float, scope: str):
    summary, predictions = [], []
    gt = wide["gt"].astype(str).to_numpy()
    for method in ("majority_vote", "risk_weighted_vote", "min_risk_model", "reliable_only_vote"):
        pred, source = [], []
        for _, row in wide.iterrows():
            p, s = vote(row, models, method, threshold)
            pred.append(p)
            source.append(s)
        pred = np.asarray(pred, object)
        summary.append(metric_row(scope, method, gt, pred))
        for i, sid in enumerate(wide["sid"]):
            predictions.append({
                "scope": scope, "sid": int(sid), "gt": gt[i], "method": method,
                "prediction": pred[i], "correct": int(pred[i] == gt[i]),
                "source_models": source[i],
            })

    for model in models:
        mask = wide[f"{model}__prediction"].notna().to_numpy()
        pred = wide.loc[mask, f"{model}__prediction"].astype(str).to_numpy()
        local_gt = wide.loc[mask, "gt"].astype(str).to_numpy()
        summary.append(metric_row(scope, f"single::{model}", local_gt, pred))

    correct_matrix = np.stack([
        wide[f"{m}__correct"].fillna(False).astype(bool).to_numpy()
        for m in models
    ], axis=1)
    any_correct = correct_matrix.any(axis=1)
    summary.append({
        "scope": scope,
        "method": "oracle_any_model_correct",
        "N": len(wide),
        "accuracy": any_correct.mean(),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "all_models_wrong_N": int((~any_correct).sum()),
    })
    return pd.DataFrame(summary), pd.DataFrame(predictions)


def sample_patterns(wide: pd.DataFrame, models: Sequence[str], scope: str):
    rows, actual, predicted = [], Counter(), Counter()
    for _, row in wide.iterrows():
        a_parts, p_parts, answers = [], [], []
        n_correct = n_unreliable = n_available = 0
        out = {"scope": scope, "sid": int(row["sid"]), "gt": row["gt"]}
        for model in models:
            pred = row.get(f"{model}__prediction")
            if pd.isna(pred):
                a_parts.append(f"{model}=M")
                p_parts.append(f"{model}=M")
                continue
            correct = bool(row[f"{model}__correct"])
            unrel = bool(row[f"{model}__predicted_unreliable"])
            risk = float(row[f"{model}__p_error"])
            n_available += 1
            n_correct += int(correct)
            n_unreliable += int(unrel)
            answers.append(str(pred))
            a_parts.append(f"{model}={'C' if correct else 'W'}")
            p_parts.append(f"{model}={'U' if unrel else 'R'}")
            out[f"{model}__prediction"] = pred
            out[f"{model}__correct"] = int(correct)
            out[f"{model}__p_error"] = risk
            out[f"{model}__predicted_unreliable"] = int(unrel)

        a_pattern = "|".join(a_parts)
        p_pattern = "|".join(p_parts)
        actual[a_pattern] += 1
        predicted[p_pattern] += 1
        out.update({
            "available_models": n_available,
            "correct_models": n_correct,
            "predicted_unreliable_models": n_unreliable,
            "all_models_wrong": int(n_correct == 0),
            "any_model_correct": int(n_correct > 0),
            "answer_disagreement": int(len(set(answers)) > 1),
            "actual_pattern": a_pattern,
            "risk_pattern": p_pattern,
        })
        rows.append(out)

    actual_rows = [{"scope": scope, "pattern": p, "count": n, "fraction": n / len(wide)} for p, n in actual.most_common()]
    risk_rows = [{"scope": scope, "pattern": p, "count": n, "fraction": n / len(wide)} for p, n in predicted.most_common()]
    return pd.DataFrame(rows), pd.DataFrame(actual_rows), pd.DataFrame(risk_rows)


def stacking_features(wide: pd.DataFrame, models: Sequence[str]) -> np.ndarray:
    cols = []
    for model in models:
        available = wide[f"{model}__prediction"].notna().astype(float).to_numpy()
        risk = wide[f"{model}__p_error"].fillna(1.0).astype(float).to_numpy()
        cols.extend([available[:, None], risk[:, None]])
        for relation in RELATIONS:
            indicator = (wide[f"{model}__prediction"].fillna("") == relation).astype(float).to_numpy()
            cols.extend([indicator[:, None], (indicator * np.maximum(1.0 - risk, 0.0))[:, None]])
    return np.concatenate(cols, axis=1)


def run_stacking(wide: pd.DataFrame, models: Sequence[str], args: argparse.Namespace, scope: str):
    x = stacking_features(wide, models)
    y = wide["gt"].map(REL_TO_ID).astype(int).to_numpy()
    min_class = min(Counter(y.tolist()).values())
    folds = min(args.stacking_folds, min_class)
    if folds < 2:
        raise RuntimeError("Not enough samples per relation for stacking")

    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=args.stacking_repeats, random_state=args.seed)
    prob_sum = np.zeros((len(y), len(RELATIONS)), float)
    count = np.zeros(len(y), int)
    for split_i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x[tr])
        xte = scaler.transform(x[te])
        clf = LogisticRegression(C=args.stacking_c, max_iter=5000, solver="lbfgs", random_state=args.seed + split_i)
        clf.fit(xtr, y[tr])
        local = clf.predict_proba(xte)
        aligned = np.zeros((len(te), len(RELATIONS)), float)
        for j, cls in enumerate(clf.classes_):
            aligned[:, int(cls)] = local[:, j]
        prob_sum[te] += aligned
        count[te] += 1
    if not np.all(count == args.stacking_repeats):
        raise RuntimeError(f"Unexpected stacking OOF counts: {np.unique(count, return_counts=True)}")

    prob = prob_sum / count[:, None]
    pred = np.asarray(RELATIONS, object)[np.argmax(prob, axis=1)]
    gt = wide["gt"].astype(str).to_numpy()
    summary = metric_row(scope, "stacked_logreg_exploratory", gt, pred)
    summary.update({"stacking_folds": folds, "stacking_repeats": args.stacking_repeats, "stacking_C": args.stacking_c})
    rows = []
    for i, sid in enumerate(wide["sid"]):
        row = {"scope": scope, "sid": int(sid), "gt": gt[i], "prediction": pred[i], "correct": int(pred[i] == gt[i])}
        for j, relation in enumerate(RELATIONS):
            row[f"probability_{relation}"] = prob[i, j]
        rows.append(row)
    return summary, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.detector_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = discover_models(root, args.models)
    models: Dict[str, pd.DataFrame] = {}
    failures = {}
    for model in requested:
        try:
            models[model] = read_model(root, model, args.threshold)
            print(f"[loaded] {model}: N={len(models[model])}, acc={models[model]['correct'].mean():.4f}")
        except Exception as exc:
            failures[model] = f"{type(exc).__name__}: {exc}"
            print(f"[skip] {model}: {failures[model]}")
    if len(models) < 2:
        raise RuntimeError(f"Need at least two usable models; failures={failures}")

    names = list(models)
    model_rows = []
    for model, frame in models.items():
        model_rows.append({
            "model": model,
            "N": len(frame),
            "generation_accuracy": frame["correct"].mean(),
            "error_prevalence": frame["error"].mean(),
            "detector_accuracy": ((frame["p_error"] >= args.threshold).astype(int) == frame["error"]).mean(),
            "mean_p_error_correct": frame.loc[frame["correct"], "p_error"].mean(),
            "mean_p_error_wrong": frame.loc[~frame["correct"], "p_error"].mean(),
        })
    model_summary = pd.DataFrame(model_rows)
    pairwise = pairwise_table(models)
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    pairwise.to_csv(out_dir / "pairwise_overlap.csv", index=False)

    scopes = ["intersection", "available"] if args.scope == "both" else [args.scope]
    sample_all, actual_all, risk_all = [], [], []
    ensemble_all, prediction_all, stacking_all = [], [], []

    for scope in scopes:
        wide = build_wide(models, scope)
        print(f"[scope={scope}] N={len(wide)}")
        samples, actual_patterns, risk_patterns = sample_patterns(wide, names, scope)
        ensemble, predictions = fixed_ensembles(wide, names, args.threshold, scope)
        sample_all.append(samples)
        actual_all.append(actual_patterns)
        risk_all.append(risk_patterns)
        ensemble_all.append(ensemble)
        prediction_all.append(predictions)
        if args.run_stacking:
            stack_summary, stack_predictions = run_stacking(wide, names, args, scope)
            ensemble_all.append(pd.DataFrame([stack_summary]))
            stacking_all.append(stack_predictions)

    sample_df = pd.concat(sample_all, ignore_index=True)
    actual_df = pd.concat(actual_all, ignore_index=True)
    risk_df = pd.concat(risk_all, ignore_index=True)
    ensemble_df = pd.concat(ensemble_all, ignore_index=True)
    prediction_df = pd.concat(prediction_all, ignore_index=True)

    sample_df.to_csv(out_dir / "sample_overlap.csv", index=False)
    actual_df.to_csv(out_dir / "actual_error_pattern_counts.csv", index=False)
    risk_df.to_csv(out_dir / "predicted_risk_pattern_counts.csv", index=False)
    ensemble_df.to_csv(out_dir / "ensemble_summary.csv", index=False)
    prediction_df.to_csv(out_dir / "ensemble_predictions.csv", index=False)
    if stacking_all:
        pd.concat(stacking_all, ignore_index=True).to_csv(out_dir / "stacking_oof_predictions.csv", index=False)

    config = {
        "requested_models": requested,
        "loaded_models": names,
        "failures": failures,
        "threshold": args.threshold,
        "scope": args.scope,
        "run_stacking": args.run_stacking,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "=" * 120,
        "CROSS-MODEL RELIABILITY OVERLAP AND ENSEMBLE",
        "=" * 120,
        f"models={','.join(names)}",
        f"threshold={args.threshold:.3f}",
        "",
        "ENSEMBLE SUMMARY",
        ensemble_df.sort_values(["scope", "accuracy"], ascending=[True, False]).to_string(index=False),
        "",
        "MOST COMPLEMENTARY PAIRS",
        pairwise.sort_values(["oracle_pair_accuracy", "actual_error_jaccard"], ascending=[False, True]).head(15).to_string(index=False),
        "",
        "Notes:",
        "- intersection is the fair main scope when model coverage differs.",
        "- oracle_any_model_correct is an upper bound, not a deployable ensemble.",
        "- fixed vote methods use only model answers and OOF detector risks.",
        "- stacked_logreg_exploratory is not a strict fully nested estimate.",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
