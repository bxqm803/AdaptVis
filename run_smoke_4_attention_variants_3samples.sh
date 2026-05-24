#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Controlled_Images_A}"
MODEL="${MODEL:-llava1.5}"
OPTION="${OPTION:-four}"
GPU="${GPU:-0}"

RESULT_DIR="${RESULT_DIR:-attention_variant_full_results}"
mkdir -p "${RESULT_DIR}"

echo "[1] compile check"
python3 -m py_compile main_aro.py
python3 -m py_compile model_zoo/llama/modeling_llama_add_attn.py

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export SAVE_ATTN=False
export TEST_MODE=False

# 关键：跑全量，不使用 RUN_SAMPLE_IDS_FILE
unset RUN_SAMPLE_IDS_FILE

run_one () {
  local variant="$1"
  local weight1="$2"

  echo
  echo "================================================"
  echo "[RUN FULL] variant=${variant}, weight1=${weight1}"
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
data = json.load(open(path, "r", encoding="utf-8"))
print("[CHECK]", path, "num_records=", len(data))
for i, r in enumerate(data[:5]):
    gen = r.get("RawGeneration", r.get("Generation", ""))
    gold = r.get("Golden", r.get("gold", ""))
    corr = r.get("RawGenerationCorrect", r.get("Correct", r.get("correct", None)))
    print(f"  {i}: correct={corr}, gold={gold}, gen={str(gen)[:80]}")
PY
}

# 方法1：原始乘法，应该和原来 fixed alpha=0.5 一致
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
echo "Results copied to: ${RESULT_DIR}/"
ls -lh "${RESULT_DIR}"
