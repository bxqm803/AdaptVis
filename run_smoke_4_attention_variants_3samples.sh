#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Controlled_Images_A}"
MODEL="${MODEL:-llava1.5}"
OPTION="${OPTION:-four}"
GPU="${GPU:-0}"

# 如果你已经有 lowprob_lt0p4_ids.json，就默认取里面前 3 个。
# 如果没有，就用 [0,1,2]。
LOWPROB_IDS="${LOWPROB_IDS:-lowprob_lt0p4_ids.json}"
SMOKE_IDS="${SMOKE_IDS:-smoke_3_ids.json}"
SMOKE_N="${SMOKE_N:-3}"

SMOKE_DIR="${SMOKE_DIR:-attention_variant_smoke_3samples}"
mkdir -p "${SMOKE_DIR}"

echo "[1] make smoke ids"
python3 - <<PY
import json, os

lowprob = "${LOWPROB_IDS}"
out = "${SMOKE_IDS}"
n = int("${SMOKE_N}")

if os.path.exists(lowprob):
    ids = json.load(open(lowprob, "r", encoding="utf-8"))
    ids = [int(x) for x in ids[:n]]
    print(f"[USE LOWPROB IDS] {lowprob} -> first {n}: {ids}")
else:
    ids = list(range(n))
    print(f"[USE DEFAULT IDS] {ids}")

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
export RUN_SAMPLE_IDS_FILE="${SMOKE_IDS}"
export SAVE_ATTN=False
export TEST_MODE=False

run_one () {
  local variant="$1"
  local weight1="$2"

  echo
  echo "================================================"
  echo "[RUN] variant=${variant}, weight1=${weight1}"
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

  cp -f "${latest}" "${SMOKE_DIR}/${variant}.json"
  echo "[COPIED] ${latest} -> ${SMOKE_DIR}/${variant}.json"

  python3 - <<PY
import json
path = "${SMOKE_DIR}/${variant}.json"
data = json.load(open(path, "r", encoding="utf-8"))
print("[CHECK]", path, "num_records=", len(data))
for i, r in enumerate(data[:3]):
    gen = r.get("RawGeneration", r.get("Generation", ""))
    gold = r.get("Golden", r.get("gold", ""))
    corr = r.get("RawGenerationCorrect", r.get("Correct", r.get("correct", None)))
    print(f"  {i}: correct={corr}, gold={gold}, gen={str(gen)[:80]}")
PY
}

# 方法1：原始乘法，应该和原来 alpha=0.5 的结果一致
run_one "mul_img" "0.5"

# 方法2：整体压低 image logits
# beta = log(0.5) = -0.69314718056
run_one "add_img" "-0.69314718056"

# 方法3：只压 image 内部极端程度，不改变 image logits 均值
run_one "center_img" "0.5"

# 方法4：softmax 后压 image probability mass
run_one "prob_img" "0.5"

echo
echo "[DONE]"
echo "Smoke ids: ${SMOKE_IDS}"
echo "Results copied to: ${SMOKE_DIR}/"
ls -lh "${SMOKE_DIR}"
