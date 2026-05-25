#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Controlled_Images_A}"
MODEL="${MODEL:-llava1.5}"
OPTION="${OPTION:-four}"
GPU="${GPU:-0}"

BASE_JSON="${BASE_JSON:-output/results1.5_Controlled_Images_A_adapt_vis_w1_w11_w21_thr0p4_fouroption_True.json}"
THRESHOLD="${THRESHOLD:-0.4}"

LOWPROB_IDS="${LOWPROB_IDS:-lowprob_lt0p4_ids.json}"
RESULT_DIR="${RESULT_DIR:-attention_variant_lowprob_zero_shrink_results}"
mkdir -p "${RESULT_DIR}"

echo "[1] make low-probability ids from base json"
python3 - <<PY
import json

base_json = "${BASE_JSON}"
threshold = float("${THRESHOLD}")
out = "${LOWPROB_IDS}"

with open(base_json, "r", encoding="utf-8") as f:
    data = json.load(f)

prob_keys = [
    "base_probability",
    "probability",
    "base_confidence",
    "confidence",
    "Confidence",
    "uncertainty",
    "Uncertainty",
]

ids = []
used_key = None
missing = 0

for i, item in enumerate(data):
    p = None
    k_used = None

    for k in prob_keys:
        if k in item and item[k] not in [None, ""]:
            try:
                p = float(item[k])
                k_used = k
                break
            except Exception:
                pass

    if p is None:
        missing += 1
        continue

    used_key = used_key or k_used

    if p < threshold:
        ids.append(i)

print("[BASE_JSON]", base_json)
print("[PROB_KEY]", used_key)
print("[THRESHOLD]", threshold)
print("[TOTAL]", len(data))
print("[SELECTED]", len(ids))
print("[MISSING_PROB]", missing)
print("[FIRST 50 IDS]", ids[:50])

if used_key is None:
    print("[ERROR] Cannot find probability/confidence key. First item keys:")
    print(list(data[0].keys()))
    raise SystemExit(1)

with open(out, "w", encoding="utf-8") as f:
    json.dump(ids, f, indent=2)

print("[SAVED]", out)
PY

echo
echo "[2] compile check"
python3 -m py_compile main_aro.py
python3 -m py_compile model_zoo/llama/modeling_llama_add_attn.py

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export SAVE_ATTN=False
export TEST_MODE=False

# 关键：只跑 probability < threshold 的样本
export RUN_SAMPLE_IDS_FILE="${LOWPROB_IDS}"

run_one () {
  local variant="$1"
  local weight1="$2"
  local out_name="$3"

  echo
  echo "================================================"
  echo "[RUN LOWPROB] variant=${variant}, weight1=${weight1}, out=${out_name}"
  echo "================================================"

  export ADAPTVIS_ATTENTION_VARIANT="${variant}"

  CUDA_VISIBLE_DEVICES="${GPU}" python3 main_aro.py \
    --dataset "${DATASET}" \
    --model-name "${MODEL}" \
    --download \
    --method adapt_vis \
    --weight1 "${weight1}" \
    --weight2 1.5 \
    --threshold 0.4 \
    --option "${OPTION}" \
    --batch-size 1 \
    --num_workers 0

  latest=$(find output -maxdepth 1 -type f -name "*.json" ! -name "*_scores.json" -printf "%T@ %p\n" \
    | sort -nr | head -n 1 | cut -d' ' -f2-)

  if [[ -z "${latest}" ]]; then
    echo "[ERROR] cannot find latest output json"
    exit 1
  fi

  cp -f "${latest}" "${RESULT_DIR}/${out_name}.json"
  echo "[COPIED] ${latest} -> ${RESULT_DIR}/${out_name}.json"

  python3 - <<PY
import json
path = "${RESULT_DIR}/${out_name}.json"
ids_path = "${LOWPROB_IDS}"

data = json.load(open(path, "r", encoding="utf-8"))
ids = json.load(open(ids_path, "r", encoding="utf-8"))

print("[CHECK]", path, "num_records=", len(data), "expected=", len(ids))
if len(data) != len(ids):
    print("[WARNING] result length != lowprob ids length")

for i, r in enumerate(data[:5]):
    sid = ids[i] if i < len(ids) else None
    gen = r.get("RawGeneration", r.get("Generation", ""))
    gold = r.get("Golden", r.get("gold", ""))
    corr = r.get("RawGenerationCorrect", r.get("Correct", r.get("correct", None)))
    print(f"  local={i}, sid={sid}: correct={corr}, gold={gold}, gen={str(gen)[:80]}")
PY
}

# baseline：原始往 0 线性收缩
run_one "mul_img" "0.5" "mul_img_0p5"

# hard shrink：直接把 visual logits 截断到 [-a, a]
run_one "clip_img" "2.0" "clip_img_a2p0"
run_one "clip_img" "1.0" "clip_img_a1p0"

# smooth shrink：a * tanh(s/a)
run_one "tanh_img" "2.0" "tanh_img_a2p0"
run_one "tanh_img" "1.0" "tanh_img_a1p0"

# softsign shrink：s / (1 + lambda * |s|)
run_one "softsign_img" "0.5" "softsign_img_lam0p5"
run_one "softsign_img" "1.0" "softsign_img_lam1p0"

echo
echo "[3] summarize variants against base"
python3 - <<PY
import os
import json
import glob
import csv

base_json = "${BASE_JSON}"
ids_json = "${LOWPROB_IDS}"
variant_dir = "${RESULT_DIR}"
out_csv = os.path.join(variant_dir, "summary.csv")

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

with open(base_json, "r", encoding="utf-8") as f:
    base = json.load(f)

with open(ids_json, "r", encoding="utf-8") as f:
    ids = [int(x) for x in json.load(f)]

base_correct = sum(correct(base[sid]) for sid in ids)

print()
print("[BASE SUBSET]")
print(f"base on selected ids: {base_correct}/{len(ids)} = {base_correct / len(ids):.6f}")

rows = []

for path in sorted(glob.glob(os.path.join(variant_dir, "*.json"))):
    name = os.path.splitext(os.path.basename(path))[0]

    with open(path, "r", encoding="utf-8") as f:
        var = json.load(f)

    if len(var) != len(ids):
        print(f"[SKIP LENGTH MISMATCH] {name}: {len(var)} vs {len(ids)}")
        continue

    w2c, c2w, c2c, w2w = [], [], [], []

    for j, sid in enumerate(ids):
        b_corr = correct(base[sid])
        v_corr = correct(var[j])

        if (not b_corr) and v_corr:
            w2c.append(sid)
        elif b_corr and (not v_corr):
            c2w.append(sid)
        elif b_corr and v_corr:
            c2c.append(sid)
        else:
            w2w.append(sid)

    final_correct = len(w2c) + len(c2c)

    row = {
        "variant": name,
        "n": len(ids),
        "final_correct": final_correct,
        "acc": final_correct / len(ids),
        "wrong_to_correct": len(w2c),
        "correct_to_wrong": len(c2w),
        "correct_to_correct": len(c2c),
        "wrong_to_wrong": len(w2w),
        "net_gain": len(w2c) - len(c2w),
        "result_json": path,
    }
    rows.append(row)

rows = sorted(rows, key=lambda r: (r["acc"], r["net_gain"]), reverse=True)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "variant",
        "n",
        "final_correct",
        "acc",
        "wrong_to_correct",
        "correct_to_wrong",
        "correct_to_correct",
        "wrong_to_wrong",
        "net_gain",
        "result_json",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print()
print("[RANKED]")
for r in rows:
    print(
        f"{r['variant']:25s} "
        f"acc={r['acc']:.6f} "
        f"w2c={r['wrong_to_correct']:3d} "
        f"c2w={r['correct_to_wrong']:3d} "
        f"net={r['net_gain']:3d}"
    )

print()
print("[SAVED]", out_csv)
PY

echo
echo "[DONE]"
echo "Lowprob ids: ${LOWPROB_IDS}"
echo "Results copied to: ${RESULT_DIR}/"
echo "Summary: ${RESULT_DIR}/summary.csv"
ls -lh "${RESULT_DIR}"
