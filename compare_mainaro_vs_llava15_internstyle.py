#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

LABELS = ["left", "right", "on", "under"]


def parse_label(x):
    t = str(x).lower()
    t = re.sub(r"[^a-z\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if re.search(r"\bleft\b", t):
        return "left"
    if re.search(r"\bright\b", t):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b", t):
        return "under"
    if re.search(r"\bon\b|\btop\b|\babove\b|\bover\b", t):
        return "on"
    return None


def as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "correct"}


def find_latest(patterns):
    hits = []
    for pat in patterns:
        hits.extend(Path(".").glob(pat))
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return str(hits[-1]) if hits else None


def acc_from_log(path):
    if not path or not Path(path).exists():
        return None
    text = Path(path).read_text(errors="ignore")
    vals = []
    patterns = [
        r"Direct acc:\s*([0-9]*\.?[0-9]+)",
        r"Individual accuracy:\s*([0-9]*\.?[0-9]+)",
        r"\n\s*([0-9]+)\s+([0-9]+)\s+([0-9]*\.?[0-9]+)\s*\n",
        r"accuracy:\s*([0-9]*\.?[0-9]+)",
        r"acc:\s*([0-9]*\.?[0-9]+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                if len(m.groups()) == 3:
                    vals.append(float(m.group(3)))
                else:
                    vals.append(float(m.group(1)))
            except Exception:
                pass
    if not vals:
        return None
    v = vals[-1]
    return v / 100.0 if v > 1 else v


def load_json_obj(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_list_from_obj(obj):
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for k in ["records", "results", "predictions", "examples", "data", "details", "outputs"]:
            if isinstance(obj.get(k), list):
                return obj[k]

        # dict indexed by sample id
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals

    return None


def normalize_records(records):
    out = {}
    for j, r in enumerate(records):
        if not isinstance(r, dict):
            continue

        idx = r.get("index", r.get("idx", r.get("sample_id", r.get("id", j))))
        try:
            idx = int(idx)
        except Exception:
            idx = j

        gold = parse_label(r.get("gold", r.get("answer", r.get("label", r.get("target", "")))))

        pred = parse_label(r.get("pred_prep", r.get("pred", r.get("prediction", r.get("pred_label", "")))))
        if pred is None:
            pred = parse_label(r.get("generation", r.get("response", r.get("output", r.get("text", "")))))

        # main_aro sometimes stores "Generation:" / "Golden:" style strings in records
        if pred is None:
            for k, v in r.items():
                if "generation" in str(k).lower() or "pred" in str(k).lower():
                    pred = parse_label(v)
                    if pred is not None:
                        break

        if gold is None:
            for k, v in r.items():
                if "gold" in str(k).lower() or "answer" in str(k).lower() or "label" in str(k).lower():
                    gold = parse_label(v)
                    if gold is not None:
                        break

        if "correct" in r:
            correct = as_bool(r["correct"])
        elif "is_correct" in r:
            correct = as_bool(r["is_correct"])
        elif gold is not None and pred is not None:
            correct = (gold == pred)
        else:
            correct = None

        prob = None
        for k in ["final_first_token_max_prob", "first_token_max_prob", "probe_first_token_max_prob", "prob", "confidence"]:
            if r.get(k) is not None:
                try:
                    prob = float(r[k])
                    break
                except Exception:
                    pass

        out[idx] = {
            "index": idx,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "generation": r.get("generation", r.get("response", r.get("output", ""))),
            "prob": prob,
            "trace": r.get("final_trace", r.get("trace", None)),
        }
    return out if out else None


def load_records(path):
    if not path or not Path(path).exists():
        return None

    p = Path(path)
    if p.suffix == ".json":
        obj = load_json_obj(p)
        records = extract_list_from_obj(obj)
        if records is None:
            print(f"[WARN] {p} is JSON but not a per-sample record list.")
            if isinstance(obj, dict):
                print("[WARN] keys:", list(obj.keys())[:30])
            return None
        return normalize_records(records)

    if p.suffix == ".jsonl":
        records = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return normalize_records(records)

    if p.suffix == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            return normalize_records(list(csv.DictReader(f)))

    print(f"[WARN] unsupported records file type: {p}")
    return None


def acc_from_records(records):
    if not records:
        return None
    vals = [r["correct"] for r in records.values() if r["correct"] is not None]
    return None if not vals else sum(vals) / len(vals)


def print_run(name, records, log_path, record_path):
    print(name)
    print("  records_path:", record_path)
    print("  records_n:", len(records) if records else "NA")
    print("  acc(records):", acc_from_records(records) if records else "NA")
    print("  log_path:", log_path)
    print("  acc(log):", acc_from_log(log_path) if log_path else "NA")


def compare(name_a, A, name_b, B, out_prefix):
    if A is None or B is None:
        print(f"\n[{name_a} vs {name_b}] skipped: missing per-sample records.")
        return

    common = sorted(set(A) & set(B))
    print(f"\n[{name_a} vs {name_b}] n={len(common)}")

    stats = defaultdict(int)
    by_gold = defaultdict(lambda: defaultdict(int))
    flips = []

    for idx in common:
        a, b = A[idx], B[idx]
        ac, bc = a["correct"], b["correct"]
        gold = a["gold"] or b["gold"]

        if ac is True and bc is True:
            cat = "both_correct"
        elif ac is False and bc is False:
            cat = "both_wrong"
        elif ac is True and bc is False:
            cat = "A_correct_B_wrong"
        elif ac is False and bc is True:
            cat = "A_wrong_B_correct"
        else:
            cat = "unknown"

        same_pred = (a["pred"] == b["pred"])

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
                "index": idx,
                "gold": gold,
                "category": cat,
                f"{name_a}_pred": a["pred"],
                f"{name_b}_pred": b["pred"],
                f"{name_a}_correct": ac,
                f"{name_b}_correct": bc,
                f"{name_a}_generation": a["generation"],
                f"{name_b}_generation": b["generation"],
                f"{name_a}_prob": a["prob"],
                f"{name_b}_prob": b["prob"],
            })

    print("both_correct:", stats["both_correct"])
    print("both_wrong:", stats["both_wrong"])
    print(f"{name_a}_correct_to_{name_b}_wrong:", stats["A_correct_B_wrong"])
    print(f"{name_a}_wrong_to_{name_b}_correct:", stats["A_wrong_B_correct"])
    print("net_gain_B_minus_A:", stats["A_wrong_B_correct"] - stats["A_correct_B_wrong"])
    print("same_pred:", stats["same_pred"])
    print("changed_pred:", stats["changed_pred"])

    header = [
        "gold", "n", "both_correct", "both_wrong",
        f"{name_a}_correct_to_{name_b}_wrong",
        f"{name_a}_wrong_to_{name_b}_correct",
        "net_gain_B_minus_A", "changed_pred", "same_pred",
    ]
    rows = []
    print("per_gold")
    print(",".join(header))
    for gold in LABELS:
        d = by_gold[gold]
        row = {
            "gold": gold,
            "n": d["n"],
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

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(str(out_prefix) + ".flips.json", "w", encoding="utf-8") as f:
        json.dump(flips, f, ensure_ascii=False, indent=2)

    with open(str(out_prefix) + ".per_gold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print("saved:", str(out_prefix) + ".flips.json")
    print("saved:", str(out_prefix) + ".per_gold.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_base_records", default=None)
    ap.add_argument("--main_scale_records", default=None)
    ap.add_argument("--intern_base_records", default=None)
    ap.add_argument("--intern_scale_records", default=None)

    ap.add_argument("--main_base_log", default="outputs/log_main_aro_llava15_ControlledA_base.txt")
    ap.add_argument("--main_scale_log", default="outputs/log_main_aro_llava15_ControlledA_scaling_w0.5.txt")
    ap.add_argument("--intern_base_log", default="outputs/log_llava15_internstyle_ControlledA_base.txt")
    ap.add_argument("--intern_scale_log", default="outputs/log_llava15_internstyle_ControlledA_scaling_w0.5.txt")

    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    # Important: main_aro saves under ./output, not ./outputs.
    # Do NOT auto-match outputs/*llava15_internstyle* as main_aro.
    if args.main_base_records is None:
        args.main_base_records = find_latest([
            "output/results*Controlled_Images_A_base_1.0_fouroption_False.json",
            "output/results*Controlled_Images_A_base*.json",
        ])
    if args.main_scale_records is None:
        args.main_scale_records = find_latest([
            "output/results*Controlled_Images_A_scaling_vis_0.5_fouroption_False.json",
            "output/results*Controlled_Images_A_scaling_vis*0.5*.json",
        ])

    if args.intern_base_records is None:
        args.intern_base_records = "outputs/llava15_internstyle_llava_hf_llava_1.5_7b_hf_Controlled_Images_A_base_w1.0_w10.5_w21.5_thr0.4_rulepaper_records.json"
    if args.intern_scale_records is None:
        args.intern_scale_records = "outputs/llava15_internstyle_llava_hf_llava_1.5_7b_hf_Controlled_Images_A_scaling_vis_w0.5_w10.5_w21.5_thr0.4_rulepaper_records.json"

    print("resolved paths")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    MB = load_records(args.main_base_records)
    MS = load_records(args.main_scale_records)
    IB = load_records(args.intern_base_records)
    IS = load_records(args.intern_scale_records)

    print("\n" + "=" * 80)
    print("summary")
    print_run("main_aro_base", MB, args.main_base_log, args.main_base_records)
    print_run("main_aro_scale0.5", MS, args.main_scale_log, args.main_scale_records)
    print_run("internstyle_base", IB, args.intern_base_log, args.intern_base_records)
    print_run("internstyle_scale0.5", IS, args.intern_scale_log, args.intern_scale_records)

    out = Path(args.out_dir)

    print("\n" + "=" * 80)
    print("base -> scale")
    compare("main_base", MB, "main_scale0.5", MS, out / "v2_compare_main_base_vs_scale0.5")
    compare("intern_base", IB, "intern_scale0.5", IS, out / "v2_compare_internstyle_base_vs_scale0.5")

    print("\n" + "=" * 80)
    print("main_aro vs internstyle")
    compare("main_base", MB, "intern_base", IB, out / "v2_compare_main_vs_internstyle_base")
    compare("main_scale0.5", MS, "intern_scale0.5", IS, out / "v2_compare_main_vs_internstyle_scale0.5")

    print("\nNOTE:")
    print("If main_aro records_n is NA, its result JSON is probably only aggregate metrics.")
    print("Then only acc(log) is valid for main_aro; per-sample flips need a real per-sample main_aro record file.")


if __name__ == "__main__":
    main()
