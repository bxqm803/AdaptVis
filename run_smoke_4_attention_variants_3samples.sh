#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Controlled_Images_A}"
MODEL="${MODEL:-llava1.5}"
OPTION="${OPTION:-four}"
GPU="${GPU:-0}"

BASE_JSON="${BASE_JSON:-output/results1.5_Controlled_Images_A_adapt_vis_w1_w11_w21_thr0p4_fouroption_True.json}"
THRESHOLD="${THRESHOLD:-0.4}"

LOWPROB_IDS="${LOWPROB_IDS:-lowprob_lt0p4_ids.json}"
RESULT_DIR="${RESULT_DIR:-attention_variant_lowprob_lt0p4_results}"
mkdir -p "${RESULT_DIR}"

echo "[1] make low-probability ids from base json"
python3 - <<PY
import json, os

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

  echo
  echo "================================================"
  echo "[RUN LOWPROB] variant=${variant}, weight1=${weight1}"
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

  cp -f "${latest}" "${RESULT_DIR}/${variant}.json"
  echo "[COPIED] ${latest} -> ${RESULT_DIR}/${variant}.json"

  python3 - <<PY
import json
path = "${RESULT_DIR}/${variant}.json"
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

# 方法1：原始乘法，应该和原来 selected 0.5 子集一致
run_one "mul_img" "0.5"

# 方法2：pre-softmax 整体压低 image logits
# beta = log(0.5) = -0.69314718056
run_one "add_img" "-0.69314718056"

# 方法3：只压 image 内部极端程度，不改变 image logits 均值
run_one "center_img" "0.5"

# 方法4：post-softmax 压 image probability mass
run_one "prob_img" "0.5"

echo
echo "[DONE]"
echo "Lowprob ids: ${LOWPROB_IDS}"
echo "Results copied to: ${RESULT_DIR}/"
ls -lh "${RESULT_DIR}"
