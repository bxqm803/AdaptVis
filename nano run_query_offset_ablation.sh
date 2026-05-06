#!/usr/bin/env bash
set -euo pipefail

DATASET="Controlled_Images_A"
OPTION="four"
MODEL_NAME="llava1.5"

W1="0.5"
W2="1.5"
THR="0.4"

RESULT_DIR="output/query_offset_ablation"
mkdir -p "${RESULT_DIR}"

# 每次 main_aro.py 会覆盖这个默认结果文件，所以每跑完一次必须立刻复制出来
DEFAULT_RESULT_JSON="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False.json"
DEFAULT_SCORE_JSON="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_Falsescores.json"

for k in $(seq 0 15); do
    echo "=============================="
    echo "Running text_offset query ${k}"
    echo "=============================="

    # 这里用临时 attention tag，跑完立刻删掉，避免保存一堆 attention
    ATTN_TAG="tmp_text_offset_${k}"

    ATTN_RUN_TAG="${ATTN_TAG}" \
    ADJUST_METHOD="text_offset" \
    QUERY_POS="${k}" \
    python3 main_aro.py \
      --dataset="${DATASET}" \
      --model-name="${MODEL_NAME}" \
      --download \
      --method=adapt_vis \
      --weight1="${W1}" \
      --weight2="${W2}" \
      --threshold="${THR}" \
      --option="${OPTION}"

    cp "${DEFAULT_RESULT_JSON}" "${RESULT_DIR}/results_text_offset_${k}.json"
    cp "${DEFAULT_SCORE_JSON}" "${RESULT_DIR}/scores_text_offset_${k}.json"

    # 不保存 attention，只保留结果
    rm -rf "output/${ATTN_TAG}"
done

python3 - <<'PY'
import os
import json
import csv

result_dir = "output/query_offset_ablation"

def is_correct(gen, gold):
    gen = "" if gen is None else str(gen)
    gold = "" if gold is None else str(gold)
    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False
    return ok

summary_rows = []
per_sample = {}

for k in range(16):
    result_path = os.path.join(result_dir, f"results_text_offset_{k}.json")
    score_path = os.path.join(result_dir, f"scores_text_offset_{k}.json")

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    correct_count = 0
    total = len(results)

    for sid, item in enumerate(results):
        prompt = item.get("Prompt", "")
        gen = item.get("Generation", "")
        gold = item.get("Golden", "")
        correct = is_correct(gen, gold)
        correct_count += int(correct)

        if sid not in per_sample:
            per_sample[sid] = {
                "sample_id": sid,
                "prompt": prompt,
                "gold": gold,
            }

        per_sample[sid][f"q{k}_generation"] = gen
        per_sample[sid][f"q{k}_correct"] = int(correct)

    acc = correct_count / total if total > 0 else 0.0

    # 如果 scores json 存在，也读一下里面的 acc
    score_acc = ""
    if os.path.exists(score_path):
        with open(score_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        score_acc = s.get("acc", "")

    summary_rows.append({
        "query_offset": k,
        "num_samples": total,
        "correct_count": correct_count,
        "accuracy_by_generation": acc,
        "accuracy_from_scores_json": score_acc,
    })

summary_csv = os.path.join(result_dir, "query_offset_summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "query_offset",
            "num_samples",
            "correct_count",
            "accuracy_by_generation",
            "accuracy_from_scores_json",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

# per-sample matrix
fieldnames = ["sample_id", "gold", "prompt"]
for k in range(16):
    fieldnames.append(f"q{k}_generation")
    fieldnames.append(f"q{k}_correct")

per_sample_csv = os.path.join(result_dir, "query_offset_per_sample.csv")
with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for sid in sorted(per_sample):
        writer.writerow(per_sample[sid])

print("Saved:", summary_csv)
print("Saved:", per_sample_csv)
PY
