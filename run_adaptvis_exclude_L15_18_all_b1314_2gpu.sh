#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/adaptvis_w05_w15_exclude_each_and_all_L15_18_original_all_b1314_allrels"

mkdir -p "${OUT_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

# 单 GPU 顺序跑。
GPU="${GPU:-0}"

# 默认不跑 full baseline。
# 如果需要原始 full_adaptvis 对照：
# RUN_FULL_BASELINE=1 bash run_adaptvis_w05_w15_exclude_each_and_all_L15_18_1gpu.sh
RUN_FULL_BASELINE="${RUN_FULL_BASELINE:-0}"

# AdaptVis 双 weight。
WEIGHT1="${WEIGHT1:-0.5}"
WEIGHT2="${WEIGHT2:-1.5}"

# 注意：threshold 决定用 weight1 还是 weight2。
# 如果 threshold=1.0，通常几乎全走 weight1=0.5。
# 可以运行时改：
# THRESHOLD=0.7 bash xxx.sh
THRESHOLD="${THRESHOLD:-0.5}"

# 只跑原图，不跑 black / shuffle。
IMAGE_CONTROL_VALUE="none"

# 不跳过 left/right/on/under，跑全 412。
unset PROBE_SAMPLE_IDS_FILE || true

# 两个区域。
GROUP_NAMES=("all" "b13_14")

echo "[CHECK] compile python files"
python -m py_compile main_aro.py
python -m py_compile model_zoo/llava15.py
python -m py_compile model_zoo/llama/modeling_llama_add_attn.py

grep -q "ADAPTVIS_EXCLUDE_LAYERS" model_zoo/llama/modeling_llama_add_attn.py || {
  echo "[ERROR] ADAPTVIS_EXCLUDE_LAYERS not found in modeling_llama_add_attn.py"
  exit 1
}

grep -q "PROBE_SINGLE_PASS" model_zoo/llava15.py || {
  echo "[ERROR] PROBE_SINGLE_PASS not found in llava15.py"
  echo "llava15.py may still make PROBE_RUN_TAG trigger single-pass."
  exit 1
}

float_tag () {
  local X="$1"
  if [[ "${X}" == "0.5" ]]; then
    echo "05"
  elif [[ "${X}" == "1.5" ]]; then
    echo "15"
  elif [[ "${X}" == "1.0" ]]; then
    echo "10"
  elif [[ "${X}" == "0.7" ]]; then
    echo "07"
  elif [[ "${X}" == "0.6" ]]; then
    echo "06"
  else
    echo "$(echo "${X}" | sed 's/\.//g')"
  fi
}

W1TAG="$(float_tag "${WEIGHT1}")"
W2TAG="$(float_tag "${WEIGHT2}")"
TTAG="$(float_tag "${THRESHOLD}")"

run_one () {
  local VARIANT="$1"
  local EXCLUDE_LAYERS_VALUE="$2"
  local GROUP="$3"

  local ADJUST_METHOD_VALUE
  local PATCH_MODE_VALUE=""
  local PATCH_BLOCK_IDS_VALUE=""

  if [[ "${GROUP}" == "all" ]]; then
    ADJUST_METHOD_VALUE="last_query"
  elif [[ "${GROUP}" == "b13_14" ]]; then
    ADJUST_METHOD_VALUE="object_mask"
    PATCH_MODE_VALUE="blocks"
    PATCH_BLOCK_IDS_VALUE="13,14"
  else
    echo "[ERROR] unknown group: ${GROUP}"
    exit 1
  fi

  local TAG="adaptvis_${VARIANT}_${GROUP}_original_allrels_w${W1TAG}_${W2TAG}_t${TTAG}"

  local RESULT_SRC="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${TAG}.json"
  local SCORE_SRC="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${TAG}scores.json"

  local RESULT_DST="${OUT_DIR}/results_${TAG}.json"
  local SCORE_DST="${OUT_DIR}/scores_${TAG}.json"

  if [[ -f "${RESULT_DST}" ]]; then
    echo "[SKIP] result exists: ${RESULT_DST}"
    return
  fi

  echo ""
  echo "============================================================"
  echo "GPU=${GPU}"
  echo "VARIANT=${VARIANT}"
  echo "GROUP=${GROUP}"
  echo "ADJUST_METHOD=${ADJUST_METHOD_VALUE}"
  echo "WEIGHT1=${WEIGHT1}"
  echo "WEIGHT2=${WEIGHT2}"
  echo "THRESHOLD=${THRESHOLD}"
  echo "ADAPTVIS_EXCLUDE_LAYERS=${EXCLUDE_LAYERS_VALUE}"
  echo "RUN ALL RELATIONS: left/right/on/under"
  echo "TAG=${TAG}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  ADAPTVIS_EXCLUDE_LAYERS="${EXCLUDE_LAYERS_VALUE}" \
  ADAPTVIS_INCLUDE_LAYERS="" \
  ADAPTVIS_LAYER_DEBUG=False \
  IMAGE_CONTROL="${IMAGE_CONTROL_VALUE}" \
  IMAGE_CONTROL_SEED=1 \
  IMAGE_CONTROL_SIZE=336 \
  IMAGE_CONTROL_GRID=24 \
  CLIP_OBJ_MASK=False \
  ADJUST_METHOD="${ADJUST_METHOD_VALUE}" \
  PATCH_MASK_MODE="${PATCH_MODE_VALUE}" \
  PATCH_GRID_SIZE=24 \
  PATCH_BLOCK_GRID=4 \
  PATCH_BLOCK_IDS="${PATCH_BLOCK_IDS_VALUE}" \
  PATCH_MASK_DEBUG=False \
  PROBE_SAMPLE_IDS_FILE="" \
  PROBE_RELATION_PROBS=True \
  PROBE_RELATION_TOPK=10 \
  PROBE_RUN_TAG="${TAG}" \
  ATTN_RUN_TAG="${TAG}" \
  PROBE_SINGLE_PASS=False \
  SAVE_LAYERS=-1 \
  python3 main_aro.py \
    --dataset="${DATASET}" \
    --model-name="${MODEL_NAME}" \
    --download \
    --method=adapt_vis \
    --weight1="${WEIGHT1}" \
    --weight2="${WEIGHT2}" \
    --threshold="${THRESHOLD}" \
    --option="${OPTION}"

  if [[ -f "${RESULT_SRC}" ]]; then
    cp "${RESULT_SRC}" "${RESULT_DST}"
  else
    echo "[WARN] missing result file: ${RESULT_SRC}"
  fi

  if [[ -f "${SCORE_SRC}" ]]; then
    cp "${SCORE_SRC}" "${SCORE_DST}"
  else
    echo "[WARN] missing score file: ${SCORE_SRC}"
  fi
}

JOBS_FILE="${OUT_DIR}/jobs.txt"
: > "${JOBS_FILE}"

# Optional full AdaptVis baseline.
if [[ "${RUN_FULL_BASELINE}" == "1" ]]; then
  for GROUP in "${GROUP_NAMES[@]}"; do
    echo "full_adaptvis||${GROUP}" >> "${JOBS_FILE}"
  done
fi

for GROUP in "${GROUP_NAMES[@]}"; do
  echo "exclude_L15|15|${GROUP}" >> "${JOBS_FILE}"
  echo "exclude_L16|16|${GROUP}" >> "${JOBS_FILE}"
  echo "exclude_L17|17|${GROUP}" >> "${JOBS_FILE}"
  echo "exclude_L18|18|${GROUP}" >> "${JOBS_FILE}"

  # 同时去除 L15/L16/L17/L18。
  echo "exclude_L15_18|15,16,17,18|${GROUP}" >> "${JOBS_FILE}"
done

echo ""
echo "[INFO] GPU: ${GPU}"
echo "[INFO] WEIGHT1=${WEIGHT1}, WEIGHT2=${WEIGHT2}, THRESHOLD=${THRESHOLD}"
echo "[INFO] total jobs: $(wc -l < "${JOBS_FILE}")"
echo "[INFO] expected rows per result file: 412"
echo "[INFO] expected skipped count: 0"
echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] jobs:"
cat "${JOBS_FILE}"

while IFS='|' read -r VARIANT EXCLUDE_LAYERS_VALUE GROUP; do
  run_one "${VARIANT}" "${EXCLUDE_LAYERS_VALUE}" "${GROUP}"
done < "${JOBS_FILE}"

echo ""
echo "All AdaptVis w1/w2 exclude jobs finished."
echo "Results saved to ${OUT_DIR}"

echo "[CHECK] result files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | wc -l

echo "[CHECK] score files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "scores_*.json" | wc -l

echo "[CHECK] files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | sort
