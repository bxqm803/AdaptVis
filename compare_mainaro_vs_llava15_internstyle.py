#!/usr/bin/env python3
import argparse, csv, json, os, re
from pathlib import Path
from collections import defaultdict

LABELS = ["left", "right", "on", "under"]

def parse_label(x):
    t = str(x).lower()
    t = re.sub(r"[^a-z\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if re.search(r"\bleft\b", t): return "left"
    if re.search(r"\bright\b", t): return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b", t): return "under"
    if re.search(r"\bon\b|\btop\b|\babove\b|\bover\b", t): return "on"
    return None

def as_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "correct"}

def acc_from_log(path):
    if not path or not Path(path).exists(): return None
    text = Path(path).read_text(errors="ignore")
    pats = [
        r"Direct acc:\s*([0-9]*\.?[0-9]+)",
        r"Individual accuracy:\s*([0-9]*\.?[0-9]+)",
        r"Accuracy:\s*([0-9]*\.?[0-9]+)",
        r"accuracy:\s*([0-9]*\.?[0-9]+)",
        r"acc:\s*([0-9]*\.?[0-9]+)",
    ]
    vals = []
    for p in pats:
        vals += [float(m.group(1)) for m in re.finditer(p, text)]
    if not vals: return None
    v = vals[-1]
    return v / 100.0 if v > 1 else v

def load_records(path):
    if not path or not Path(path).exists(): return None
    p = Path(path)
    if p.suffix == ".json":
        obj = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(obj, dict):
            obj = obj.get("records") or obj.get("results") or obj.get("examples") or obj.get("data")
        recs = obj
    elif p.suffix == ".jsonl":
        recs = [json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip()]
    elif p.suffix == ".csv":
        recs = list(csv.DictReader(open(p, "r", encoding="utf-8")))
    else:
        raise ValueError(f"unsupported file type: {p}")

    if not isinstance(recs, list): return None

    out = {}
    for j, r in enumerate(recs):
        idx = int(r.get("index", r.get("idx", j)))
        gold = parse_label(r.get("gold", r.get("answer", "")))
        pred = parse_label(r.get("pred_prep", r.get("pred", r.get("prediction", ""))))
        if pred is None:
            pred = parse_label(r.get("generation", r.get("response", "")))
        if "correct" in r:
            correct = as_bool(r["correct"])
        elif "is_correct" in r:
            correct = as_bool(r["is_correct"])
        elif gold is not None and pred is not None:
            correct = (gold == pred)
        else:
            correct = None
        prob = None
        for k in ["final_first_token_max_prob", "first_token_max_prob", "probe_first_token_max_prob"]:
            if r.get(k) is not None:
                try:
                    prob = float(r[k]); break
                except Exception:
                    pass
        out[idx] = {
            "idx": idx,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "generation": r.get("generation", r.get("response", "")),
            "prob": prob,
            "trace": r.get("final_trace", r.get("trace", None)),
        }
    return out

def find_one(patterns):
    for pat in patterns:
        hits = sorted(Path(".").glob(pat))
        if hits: return str(hits[-1])
    return None

def acc_from_records(recs):
    if not recs: return None
    vals = [r["correct"] for r in recs.values() if r["correct"] is not None]
    return None if not vals else sum(vals) / len(vals)

def print_run(name, recs, log):
    print(f"{name}")
    print(f"  records: {len(recs) if recs else 'NA'}")
    print(f"  acc(records): {acc_from_records(recs) if recs else 'NA'}")
    print(f"  acc(log): {acc_from_log(log) if log else 'NA'}")
    print(f"  log: {log}")

def compare(name_a, A, name_b, B, out_prefix=None):
    if A is None or B is None:
        print(f"\n[{name_a} vs {name_b}] skipped: missing per-sample records")
        return
    common = sorted(set(A) & set(B))
    print(f"\n[{name_a} vs {name_b}] n={len(common)}")
    stats = defaultdict(int)
    by_gold = defaultdict(lambda: defaultdict(int))
    flips = []
    for i in common:
        a, b = A[i], B[i]
        ac, bc = a["correct"], b["correct"]
        gold = a["gold"] or b["gold"]
        if ac and bc: cat = "both_correct"
        elif (ac is False) and (bc is False): cat = "both_wrong"
        elif ac and (bc is False): cat = "A_correct_B_wrong"
        elif (ac is False) and bc: cat = "A_wrong_B_correct"
        else: cat = "unknown"
        same_pred = a["pred"] == b["pred"]
        stats[cat] += 1
        stats["same_pred"] += int(same_pred)
        stats["changed_pred"] += int(not same_pred)
        if gold:
            by_gold[gold]["n"] += 1
            by_gold[gold][cat] += 1
            by_gold[gold]["same_pred"] += int(same_pred)
            by_gold[gold]["changed_pred"] += int(not same_pred)
        if cat in {"A_correct_B_wrong", "A_wrong_B_correct"}:
            flips.append({
                "index": i, "gold": gold, "category": cat,
                f"{name_a}_pred": a["pred"], f"{name_b}_pred": b["pred"],
                f"{name_a}_correct": ac, f"{name_b}_correct": bc,
                f"{name_a}_prob": a["prob"], f"{name_b}_prob": b["prob"],
                f"{name_a}_generation": a["generation"],
                f"{name_b}_generation": b["generation"],
            })

    print("both_correct:", stats["both_correct"])
    print("both_wrong:", stats["both_wrong"])
    print(f"{name_a}_correct_to_{name_b}_wrong:", stats["A_correct_B_wrong"])
    print(f"{name_a}_wrong_to_{name_b}_correct:", stats["A_wrong_B_correct"])
    print("net_gain_B_minus_A:", stats["A_wrong_B_correct"] - stats["A_correct_B_wrong"])
    print("same_pred:", stats["same_pred"])
    print("changed_pred:", stats["changed_pred"])

    print("per_gold")
    header = ["gold","n","both_correct","both_wrong",
              f"{name_a}_correct_to_{name_b}_wrong",
              f"{name_a}_wrong_to_{name_b}_correct",
              "net_gain_B_minus_A","changed_pred","same_pred"]
    rows = []
    print(",".join(header))
    for g in LABELS:
        d = by_gold[g]
        row = {
            "gold": g, "n": d["n"],
            "both_correct": d["both_correct"],
            "both_wrong": d["both_wrong"],
            f"{name_a}_correct_to_{name_b}_wrong": d["A_correct_B_wrong"],
            f"{name_a}_wrong_to_{name_b}_correct": d["A_wrong_B_correct"],
            "net_gain_B_minus_A": d["A_wrong_B_correct"] - d["A_correct_B_wrong"],
            "changed_pred": d["changed_pred"],
            "same_pred": d["same_pred"],
        }
        rows.append(row)
        print(",".join(str(row[h]) for h in header))

    if out_prefix:
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out_prefix)+".flips.json", "w", encoding="utf-8") as f:
            json.dump(flips, f, ensure_ascii=False, indent=2)
        with open(str(out_prefix)+".per_gold.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader(); w.writerows(rows)
        print("saved:", str(out_prefix)+".flips.json")
        print("saved:", str(out_prefix)+".per_gold.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--main_base_records", default=None)
    ap.add_argument("--main_scale_records", default=None)
    ap.add_argument("--intern_base_records", default=None)
    ap.add_argument("--intern_scale_records", default=None)
    ap.add_argument("--main_base_log", default=None)
    ap.add_argument("--main_scale_log", default=None)
    ap.add_argument("--intern_base_log", default=None)
    ap.add_argument("--intern_scale_log", default=None)
    args = ap.parse_args()
    out = Path(args.outputs)

    args.intern_base_records = args.intern_base_records or str(out / "llava15_internstyle_llava_hf_llava_1.5_7b_hf_Controlled_Images_A_base_w1.0_w10.5_w21.5_thr0.4_rulepaper_records.json")
    args.intern_scale_records = args.intern_scale_records or str(out / "llava15_internstyle_llava_hf_llava_1.5_7b_hf_Controlled_Images_A_scaling_vis_w0.5_w10.5_w21.5_thr0.4_rulepaper_records.json")

    args.main_base_log = args.main_base_log or str(out / "log_main_aro_llava15_ControlledA_base.txt")
    args.main_scale_log = args.main_scale_log or str(out / "log_main_aro_llava15_ControlledA_scaling_w0.5.txt")
    args.intern_base_log = args.intern_base_log or str(out / "log_llava15_internstyle_ControlledA_base.txt")
    args.intern_scale_log = args.intern_scale_log or str(out / "log_llava15_internstyle_ControlledA_scaling_w0.5.txt")

    # main_aro normally may not save per-sample records. Try common names.
    args.main_base_records = args.main_base_records or find_one([
        "outputs/*main*aro*llava*Controlled*base*records.json",
        "outputs/*llava1.5*Controlled*base*records.json",
        "outputs/*llava15*Controlled*base*records.json",
    ])
    args.main_scale_records = args.main_scale_records or find_one([
        "outputs/*main*aro*llava*Controlled*scaling*0.5*records.json",
        "outputs/*llava1.5*Controlled*scaling*0.5*records.json",
        "outputs/*llava15*Controlled*scaling*0.5*records.json",
    ])

    print("resolved paths")
    for k,v in vars(args).items():
        print(f"  {k}: {v}")

    MB = load_records(args.main_base_records)
    MS = load_records(args.main_scale_records)
    IB = load_records(args.intern_base_records)
    IS = load_records(args.intern_scale_records)

    print("\n" + "="*80)
    print("summary")
    print_run("main_aro_base", MB, args.main_base_log)
    print_run("main_aro_scale0.5", MS, args.main_scale_log)
    print_run("internstyle_base", IB, args.intern_base_log)
    print_run("internstyle_scale0.5", IS, args.intern_scale_log)

    print("\n" + "="*80)
    print("base -> scale")
    compare("main_base", MB, "main_scale0.5", MS, out / "compare_main_base_vs_scale0.5")
    compare("intern_base", IB, "intern_scale0.5", IS, out / "compare_internstyle_base_vs_scale0.5")

    print("\n" + "="*80)
    print("main_aro vs internstyle")
    compare("main_base", MB, "intern_base", IB, out / "compare_main_vs_internstyle_base")
    compare("main_scale0.5", MS, "intern_scale0.5", IS, out / "compare_main_vs_internstyle_scale0.5")

    if MB is None or MS is None:
        print("\nNOTE: 没找到 main_aro 的逐样本 records，所以 main_aro 只能从 log 对比整体 acc。")
        print("如果你有 main_aro 的逐样本 json/csv，重新运行时加：")
        print("  --main_base_records PATH --main_scale_records PATH")

if __name__ == "__main__":
    main()
