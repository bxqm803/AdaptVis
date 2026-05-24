import os
import glob
import json
import csv
import math
import argparse
import random
from collections import defaultdict

import numpy as np


EPS = 1e-12
ANSWER_CLASSES = ["left", "right", "on", "under"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", default="Controlled_Images_A")
    p.add_argument("--option", default="four")
    p.add_argument("--test-tag", default="True")

    p.add_argument("--base-json", default=None)
    p.add_argument("--low-json", default=None)

    p.add_argument("--base-feature-dir", default="output/hidden_features_base_w1")
    p.add_argument("--out-dir", default="base_router_alpha05_quick_validate")

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-ratio", type=float, default=0.5)
    p.add_argument("--num-thresholds", type=int, default=200)

    return p.parse_args()


def find_one(name, patterns, exclude=()):
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))

    hits = [
        h for h in sorted(set(hits))
        if h.endswith(".json")
        and not h.endswith("_scores.json")
        and "summary" not in h
        and all(x not in h for x in exclude)
    ]

    if not hits:
        raise FileNotFoundError(f"No file found for {name}:\n" + "\n".join(patterns))

    print(f"[USE {name}] {hits[0]}")
    return hits[0]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sid_of(i, item):
    return int(item.get("sample_id", item.get("SampleID", item.get("id", i))))


def correct(item):
    for k in ["RawGenerationCorrect", "Correct", "correct"]:
        if k in item:
            return bool(item[k])

    gold = str(item.get("Golden", item.get("gold", ""))).strip()
    gen = str(item.get("RawGeneration", item.get("Generation", ""))).strip()

    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.lower():
        ok = False
    return bool(ok)


def generation(item):
    return str(item.get("RawGeneration", item.get("Generation", ""))).strip()


def gold_of(item):
    g = item.get("Golden", item.get("gold", ""))
    if isinstance(g, list):
        return str(g[0]).strip() if g else ""
    return str(g).strip()


def get_uncertainty(item):
    for k in ["uncertainty", "Uncertainty", "confidence", "Confidence"]:
        if k in item and item[k] not in [None, ""]:
            try:
                return float(item[k])
            except Exception:
                pass
    return None


def feature_file(feature_dir, sid):
    pats = [
        os.path.join(feature_dir, f"sid{sid:04d}_base*.npz"),
        os.path.join(feature_dir, f"sid{sid:04d}_*alpha1*.npz"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    hits = sorted(set(hits))
    return hits[0] if hits else None


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + EPS)


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(p + EPS)))


def norm_label(s):
    s = str(s).strip().lower()
    s = s.replace("▁", "").replace("Ġ", "")
    return s


def load_base_answer_metrics(feature_dir, sid):
    f = feature_file(feature_dir, sid)
    if f is None:
        return {}

    try:
        x = np.load(f, allow_pickle=True)
    except Exception:
        return {}

    if "answer_logits" not in x.files or "answer_token_labels" not in x.files:
        return {}

    logits = np.asarray(x["answer_logits"], dtype=np.float32)
    labels = [str(t) for t in x["answer_token_labels"].tolist()]

    cls_logits = {c: [] for c in ANSWER_CLASSES}

    for lab, logit in zip(labels, logits):
        nl = norm_label(lab)
        for c in ANSWER_CLASSES:
            if nl == c:
                cls_logits[c].append(float(logit))

    # Aggregate duplicate forms, e.g. "left" and " left", by max logit.
    cls_vec = []
    for c in ANSWER_CLASSES:
        if cls_logits[c]:
            cls_vec.append(max(cls_logits[c]))
        else:
            cls_vec.append(float("-inf"))

    cls_vec = np.asarray(cls_vec, dtype=np.float64)

    if not np.all(np.isfinite(cls_vec)):
        return {}

    probs = softmax(cls_vec)
    order = np.argsort(-probs)

    top1 = int(order[0])
    top2 = int(order[1])

    return {
        "base_answer_pred": ANSWER_CLASSES[top1],
        "base_answer_conf": float(probs[top1]),
        "base_answer_gap": float(probs[top1] - probs[top2]),
        "base_answer_entropy": entropy(probs),
        "base_answer_logits_left": float(cls_vec[0]),
        "base_answer_logits_right": float(cls_vec[1]),
        "base_answer_logits_on": float(cls_vec[2]),
        "base_answer_logits_under": float(cls_vec[3]),
    }


def build_rows(base, low, base_feature_dir):
    rows = []
    n = min(len(base), len(low))

    for i in range(n):
        sid = sid_of(i, base[i])

        b_corr = correct(base[i])
        l_corr = correct(low[i])

        row = {
            "idx": i,
            "sid": sid,
            "gold": gold_of(base[i]),
            "base_correct": b_corr,
            "low_correct": l_corr,
            "base_generation": generation(base[i]),
            "low_generation": generation(low[i]),
            "base_uncertainty": get_uncertainty(base[i]),
        }

        row.update(load_base_answer_metrics(base_feature_dir, sid))

        # What would happen if this sample used alpha=0.5?
        if (not b_corr) and l_corr:
            row["low_effect"] = "wrong_to_correct"
        elif b_corr and (not l_corr):
            row["low_effect"] = "correct_to_wrong"
        elif b_corr and l_corr:
            row["low_effect"] = "correct_to_correct"
        else:
            row["low_effect"] = "wrong_to_wrong"

        rows.append(row)

    return rows


def eval_rule(rows, metric, direction, thr, indices=None):
    if indices is None:
        use_rows = rows
    else:
        use_rows = [rows[i] for i in indices]

    selected = []
    final_correct = 0

    counts = defaultdict(int)

    for r in use_rows:
        v = r.get(metric, None)

        if v is None or not np.isfinite(v):
            use_low = False
        else:
            if direction == "lt":
                use_low = v < thr
            elif direction == "gt":
                use_low = v > thr
            else:
                raise ValueError(direction)

        if use_low:
            selected.append(r)
            final = bool(r["low_correct"])
            counts[r["low_effect"]] += 1
        else:
            final = bool(r["base_correct"])

        final_correct += int(final)

    n = len(use_rows)
    acc = final_correct / n if n else 0.0

    return {
        "metric": metric,
        "direction": direction,
        "threshold": float(thr),
        "n": n,
        "acc": acc,
        "num_correct": final_correct,
        "selected_low": len(selected),
        "selected_low_ratio": len(selected) / n if n else 0.0,
        "selected_wrong_to_correct": counts["wrong_to_correct"],
        "selected_correct_to_wrong": counts["correct_to_wrong"],
        "selected_correct_to_correct": counts["correct_to_correct"],
        "selected_wrong_to_wrong": counts["wrong_to_wrong"],
        "net_gain": counts["wrong_to_correct"] - counts["correct_to_wrong"],
    }


def candidate_thresholds(rows, metric, num=200):
    vals = []
    for r in rows:
        v = r.get(metric, None)
        if v is not None and np.isfinite(v):
            vals.append(float(v))

    vals = sorted(set(vals))
    if len(vals) <= 1:
        return []

    lo, hi = min(vals), max(vals)

    if len(vals) <= num:
        mids = []
        for a, b in zip(vals[:-1], vals[1:]):
            mids.append(0.5 * (a + b))
        return [lo - 1e-9] + mids + [hi + 1e-9]

    return np.linspace(lo, hi, num).tolist()


def search_best_threshold(rows, train_idx, metric, direction, num_thresholds):
    thrs = candidate_thresholds([rows[i] for i in train_idx], metric, num_thresholds)

    best = None
    all_results = []

    for thr in thrs:
        res = eval_rule(rows, metric, direction, thr, train_idx)
        all_results.append(res)

        key = (
            res["acc"],
            res["net_gain"],
            -res["selected_correct_to_wrong"],
            res["selected_wrong_to_correct"],
        )

        if best is None or key > best[0]:
            best = (key, res)

    return best[1] if best else None, all_results


def write_csv(path, rows):
    if not rows:
        return

    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_json = args.base_json
    low_json = args.low_json

    if base_json is None:
        base_json = find_one(
            "BASE fixed w1=w2=1.0",
            [
                f"output/*{args.dataset}*adapt_vis*w1_w11_w21_thr0p4_{args.option}option_{args.test_tag}.json",
                f"output/*{args.dataset}*adapt_vis*w1_w11p0_w21p0_thr0p4_{args.option}option_{args.test_tag}.json",
            ],
        )

    if low_json is None:
        low_json = find_one(
            "LOW fixed w1=w2=0.5",
            [
                f"output/*{args.dataset}*adapt_vis*w1_w10p5_w20p5_thr0p4_{args.option}option_{args.test_tag}.json",
                f"output/*{args.dataset}*adapt_vis*w1_w10.5_w20.5_thr0p4_{args.option}option_{args.test_tag}.json",
            ],
            exclude=("w1_w10p5_w21p5",),
        )

    base = load_json(base_json)
    low = load_json(low_json)

    rows = build_rows(base, low, args.base_feature_dir)
    n = len(rows)

    base_acc = sum(r["base_correct"] for r in rows) / n
    low_acc = sum(r["low_correct"] for r in rows) / n
    oracle_acc = sum(r["base_correct"] or r["low_correct"] for r in rows) / n

    effects = defaultdict(int)
    for r in rows:
        effects[r["low_effect"]] += 1

    print("\n[DATA]")
    print("base json:", base_json)
    print("low json :", low_json)
    print("n:", n)

    print("\n[BASE vs FIXED LOW]")
    print(f"base acc     : {sum(r['base_correct'] for r in rows)}/{n} = {base_acc:.6f}")
    print(f"fixed 0.5 acc: {sum(r['low_correct'] for r in rows)}/{n} = {low_acc:.6f}")
    print(f"oracle acc   : {sum(r['base_correct'] or r['low_correct'] for r in rows)}/{n} = {oracle_acc:.6f}")

    print("\n[FIXED 0.5 EFFECT RELATIVE TO BASE]")
    for k in ["wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"]:
        print(f"{k}: {effects[k]}")

    # Metrics that can be used without gold.
    metric_specs = []

    if any(r.get("base_uncertainty", None) is not None for r in rows):
        metric_specs.append(("base_uncertainty", "lt"))

    if any(r.get("base_answer_conf", None) is not None for r in rows):
        metric_specs.extend([
            ("base_answer_conf", "lt"),
            ("base_answer_gap", "lt"),
            ("base_answer_entropy", "gt"),
        ])

    if not metric_specs:
        raise RuntimeError("No base-only metric found. Need base uncertainty in JSON or answer_logits in base feature npz.")

    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)

    split = int(round(n * args.train_ratio))
    train_idx = sorted(indices[:split])
    test_idx = sorted(indices[split:])

    print("\n[SPLIT]")
    print("train:", len(train_idx))
    print("test :", len(test_idx))

    final_results = []
    full_grid_results = []

    print("\n[BASE-ONLY ROUTER SEARCH]")
    for metric, direction in metric_specs:
        best_train, grid = search_best_threshold(
            rows,
            train_idx,
            metric,
            direction,
            args.num_thresholds,
        )
        full_grid_results.extend(grid)

        if best_train is None:
            continue

        test_res = eval_rule(rows, metric, direction, best_train["threshold"], test_idx)
        all_res = eval_rule(rows, metric, direction, best_train["threshold"], None)

        result = {
            "metric": metric,
            "direction": direction,
            "threshold_from_train": best_train["threshold"],

            "train_acc": best_train["acc"],
            "train_selected_low": best_train["selected_low"],
            "train_w2c": best_train["selected_wrong_to_correct"],
            "train_c2w": best_train["selected_correct_to_wrong"],
            "train_net_gain": best_train["net_gain"],

            "test_acc": test_res["acc"],
            "test_selected_low": test_res["selected_low"],
            "test_w2c": test_res["selected_wrong_to_correct"],
            "test_c2w": test_res["selected_correct_to_wrong"],
            "test_net_gain": test_res["net_gain"],

            "all_acc": all_res["acc"],
            "all_selected_low": all_res["selected_low"],
            "all_w2c": all_res["selected_wrong_to_correct"],
            "all_c2w": all_res["selected_correct_to_wrong"],
            "all_net_gain": all_res["net_gain"],
        }

        final_results.append(result)

        op = "<" if direction == "lt" else ">"
        print(
            f"\nmetric={metric} rule: use 0.5 if {metric} {op} {best_train['threshold']:.6f}"
        )
        print(
            f"TRAIN acc={best_train['acc']:.6f}, selected={best_train['selected_low']}, "
            f"w2c={best_train['selected_wrong_to_correct']}, "
            f"c2w={best_train['selected_correct_to_wrong']}, "
            f"net={best_train['net_gain']}"
        )
        print(
            f"TEST  acc={test_res['acc']:.6f}, selected={test_res['selected_low']}, "
            f"w2c={test_res['selected_wrong_to_correct']}, "
            f"c2w={test_res['selected_correct_to_wrong']}, "
            f"net={test_res['net_gain']}"
        )
        print(
            f"ALL   acc={all_res['acc']:.6f}, selected={all_res['selected_low']}, "
            f"w2c={all_res['selected_wrong_to_correct']}, "
            f"c2w={all_res['selected_correct_to_wrong']}, "
            f"net={all_res['net_gain']}"
        )

    # Reference: original threshold 0.4 if uncertainty exists.
    ref = None
    if any(r.get("base_uncertainty", None) is not None for r in rows):
        ref = eval_rule(rows, "base_uncertainty", "lt", 0.4, None)
        print("\n[REFERENCE original-like rule]")
        print("use 0.5 if base_uncertainty < 0.4")
        print(
            f"acc={ref['acc']:.6f}, selected={ref['selected_low']}, "
            f"w2c={ref['selected_wrong_to_correct']}, "
            f"c2w={ref['selected_correct_to_wrong']}, "
            f"net={ref['net_gain']}"
        )

    summary = {
        "base_json": base_json,
        "low_json": low_json,
        "base_feature_dir": args.base_feature_dir,
        "n": n,
        "base_acc": base_acc,
        "fixed_low_acc": low_acc,
        "oracle_acc": oracle_acc,
        "fixed_low_effect_counts": dict(effects),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "router_results": final_results,
        "reference_uncertainty_0p4": ref,
    }

    rows_csv = os.path.join(args.out_dir, "base_router_rows.csv")
    result_csv = os.path.join(args.out_dir, "base_router_results.csv")
    grid_csv = os.path.join(args.out_dir, "base_router_full_grid_train.csv")
    summary_json = os.path.join(args.out_dir, "base_router_summary.json")

    write_csv(rows_csv, rows)
    write_csv(result_csv, final_results)
    write_csv(grid_csv, full_grid_results)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(rows_csv)
    print(result_csv)
    print(grid_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
