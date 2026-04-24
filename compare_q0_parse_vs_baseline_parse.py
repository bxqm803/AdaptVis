import argparse
import json
import re
from pathlib import Path
from collections import Counter


RELATIONS = {"left", "right", "on", "under"}


def parse_first_relation(text: str):
    """
    模拟 multiq q0 / orig 的解析逻辑：
    遇到第一个 left/right/on/under 就返回。
    """
    text = text.lower()

    patterns = [
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bon\b", "on"),
        (r"\bunder\b", "under"),
    ]

    found = []
    for pat, rel in patterns:
        m = re.search(pat, text)
        if m:
            found.append((m.start(), rel))

    if not found:
        return "UNK"

    found.sort(key=lambda x: x[0])
    return found[0][1]


def parse_last_relation(text: str):
    """
    模拟 baseline 的解析逻辑：
    找到所有关系词，返回最后一个。
    """
    text = text.lower()

    patterns = [
        (r"\bto the left of\b", "left"),
        (r"\bto the right of\b", "right"),
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\bon top of\b", "on"),
        (r"\bon the\b", "on"),
        (r"\bon\b", "on"),
        (r"\bunder\b", "under"),
    ]

    found = []
    for pat, rel in patterns:
        for m in re.finditer(pat, text):
            found.append((m.start(), rel))

    if not found:
        return "UNK"

    found.sort(key=lambda x: x[0])
    return found[-1][1]


def load_json_or_jsonl(path):
    path = Path(path)

    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # 兼容 {"results": [...]} / {"data": [...]} 这种格式
        for key in ["results", "data", "records", "samples"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(f"Unsupported file format: {path}")


def get_field(row, candidates):
    for k in candidates:
        if k in row and row[k] is not None:
            return row[k]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="json/jsonl result file")
    parser.add_argument("--out", default="parse_compare_diff.jsonl")
    parser.add_argument("--text-key", default=None, help="output text field name, optional")
    parser.add_argument("--gold-key", default=None, help="gold answer field name, optional")
    args = parser.parse_args()

    rows = load_json_or_jsonl(args.input)

    text_keys = (
        [args.text_key]
        if args.text_key
        else [
            "raw_output",
            "pred_text",
            "prediction_text",
            "output",
            "response",
            "answer_text",
            "text",
        ]
    )

    gold_keys = (
        [args.gold_key]
        if args.gold_key
        else [
            "gold",
            "gold_rel",
            "answer",
            "base_answer",
            "label",
            "gt",
        ]
    )

    total = 0
    diff_parse = 0
    first_correct = 0
    last_correct = 0
    first_only_correct = 0
    last_only_correct = 0
    both_wrong = 0
    both_correct = 0

    diff_rows = []
    counter = Counter()

    for i, row in enumerate(rows):
        text = get_field(row, text_keys)
        gold = get_field(row, gold_keys)

        if text is None:
            continue

        text = str(text)
        gold = str(gold).lower().strip() if gold is not None else None

        # 只保留标准关系词
        if gold not in RELATIONS:
            gold = None

        pred_first = parse_first_relation(text)
        pred_last = parse_last_relation(text)

        total += 1

        if pred_first != pred_last:
            diff_parse += 1

        first_ok = gold is not None and pred_first == gold
        last_ok = gold is not None and pred_last == gold

        if first_ok:
            first_correct += 1
        if last_ok:
            last_correct += 1

        if first_ok and last_ok:
            both_correct += 1
        elif first_ok and not last_ok:
            first_only_correct += 1
        elif last_ok and not first_ok:
            last_only_correct += 1
        elif gold is not None:
            both_wrong += 1

        counter[(pred_first, pred_last)] += 1

        if pred_first != pred_last or first_ok != last_ok:
            diff_rows.append(
                {
                    "index": i,
                    "gold": gold,
                    "pred_first_multiq_style": pred_first,
                    "pred_last_baseline_style": pred_last,
                    "first_correct": first_ok,
                    "last_correct": last_ok,
                    "text": text,
                    "row": row,
                }
            )

    with open(args.out, "w", encoding="utf-8") as f:
        for r in diff_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("Parse comparison")
    print("=" * 80)
    print(f"Total parsed rows: {total}")
    print(f"Different predictions: {diff_parse}")
    print(f"Diff rate: {diff_parse / total:.4f}" if total else "Diff rate: N/A")

    print()
    print("Accuracy, only if gold exists:")
    print(f"First-relation / multiq-style correct: {first_correct}")
    print(f"Last-relation  / baseline-style correct: {last_correct}")

    valid_gold_count = both_correct + first_only_correct + last_only_correct + both_wrong
    if valid_gold_count:
        print(f"First-relation acc: {first_correct / valid_gold_count:.4f}")
        print(f"Last-relation acc:  {last_correct / valid_gold_count:.4f}")

    print()
    print("Correctness split:")
    print(f"Both correct:        {both_correct}")
    print(f"First only correct:  {first_only_correct}")
    print(f"Last only correct:   {last_only_correct}")
    print(f"Both wrong:          {both_wrong}")

    print()
    print("Prediction pair counts: first -> last")
    for (a, b), c in counter.most_common():
        print(f"{a:>5} -> {b:<5}: {c}")

    print()
    print(f"Saved differing cases to: {args.out}")


if __name__ == "__main__":
    main()
