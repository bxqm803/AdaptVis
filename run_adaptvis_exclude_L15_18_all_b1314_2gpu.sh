#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/adaptvis_exclude_each_L15_18_original_all_b1314_allrels"

mkdir -p "${OUT_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

# 单 GPU 顺序跑。
GPU="${GPU:-0}"

# 默认不跑 full baseline。
# 如果你缺原始 full_adaptvis 对照，可以这样运行：
# RUN_FULL_BASELINE=1 bash run_adaptvis_exclude_each_L15_18_original_all_b1314_allrels_1gpu.sh
RUN_FULL_BASELINE="${RUN_FULL_BASELINE:-0}"

# 只跑原图，不跑 black / shuffle。
IMAGE_CONTROL_VALUE="none"

# 不跳过 left/right/on/under，跑全 412。
unset PROBE_SAMPLE_IDS_FILE || true

# 分别排除这四层。
EXCLUDE_LIST=("15" "16" "17" "18")

# 两个区域。
GROUP_NAMES=("all" "b13_14")

# AdaptVis 乘法强度。
WEIGHTS=("1.5")

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

weight_tag () {
  local W="$1"
  if [[ "${W}" == "1.5" ]]; then
    echo "w15"
  elif [[ "${W}" == "0.5" ]]; then
    echo "w05"
  elif [[ "${W}" == "2.0" ]]; then
    echo "w20"
  else
    echo "w$(echo "${W}" | sed 's/\.//g')"
  fi
}

run_one () {
  local EXCLUDE_LAYER="$1"
  local GROUP="$2"
  local WEIGHT="$3"

  local WTAG
  WTAG=$(weight_tag "${WEIGHT}")

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

  local VARIANT
  local EXCLUDE_LAYERS_VALUE

  if [[ "${EXCLUDE_LAYER}" == "full" ]]; then
    VARIANT="full_adaptvis"
    EXCLUDE_LAYERS_VALUE=""
  else
    VARIANT="exclude_L${EXCLUDE_LAYER}"
    EXCLUDE_LAYERS_VALUE="${EXCLUDE_LAYER}"
  fi

  local TAG="adaptvis_${VARIANT}_${GROUP}_original_allrels_${WTAG}"

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
  echo "WEIGHT=${WEIGHT}"
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
    --weight1="${WEIGHT}" \
    --weight2="${WEIGHT}" \
    --threshold=1.0 \
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

if [[ "${RUN_FULL_BASELINE}" == "1" ]]; then
  for GROUP in "${GROUP_NAMES[@]}"; do
    for WEIGHT in "${WEIGHTS[@]}"; do
      echo "full|${GROUP}|${WEIGHT}" >> "${JOBS_FILE}"
    done
  done
fi

for L in "${EXCLUDE_LIST[@]}"; do
  for GROUP in "${GROUP_NAMES[@]}"; do
    for WEIGHT in "${WEIGHTS[@]}"; do
      echo "${L}|${GROUP}|${WEIGHT}" >> "${JOBS_FILE}"
    done
  done
done

echo ""
echo "[INFO] GPU: ${GPU}"
echo "[INFO] total jobs: $(wc -l < "${JOBS_FILE}")"
echo "[INFO] expected rows per result file: 412"
echo "[INFO] expected skipped count: 0"
echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] jobs:"
cat "${JOBS_FILE}"

while IFS='|' read -r EXCLUDE_LAYER GROUP WEIGHT; do
  run_one "${EXCLUDE_LAYER}" "${GROUP}" "${WEIGHT}"
done < "${JOBS_FILE}"

echo ""
echo "All single-layer exclude AdaptVis jobs finished."
echo "Results saved to ${OUT_DIR}"

echo "[CHECK] result files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | wc -l

echo "[CHECK] score files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "scores_*.json" | wc -l

echo "[CHECK] files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | sort
