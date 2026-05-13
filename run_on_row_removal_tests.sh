#!/usr/bin/env bash
set -euo pipefail

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

W1="0.5"
W2="1.5"
CONF_THR="0.4"

GRID_SIZE="24"
BLOCK_GRID="4"

OUT_DIR="output/on_row_removal_tests_grid${BLOCK_GRID}"
mkdir -p "${OUT_DIR}"

export SAVE_LAYERS=-1

SRC_RESULT="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False.json"
SRC_SCORE="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_Falsescores.json"

run_group () {
  local NAME="$1"
  local IDS="$2"

  echo "========================================"
  echo "Running ${NAME}: ${IDS}"
  echo "========================================"

  TAG="onrow_${NAME}_blocks${IDS//,/plus}_grid${BLOCK_GRID}_conf${CONF_THR}_w1${W1}_w2${W2}"

  CLIP_OBJ_MASK=False \
  ADJUST_METHOD=object_mask \
  PATCH_MASK_MODE=blocks \
  PATCH_BLOCK_IDS="${IDS}" \
  PATCH_GRID_SIZE="${GRID_SIZE}" \
  PATCH_BLOCK_GRID="${BLOCK_GRID}" \
  PATCH_MASK_DEBUG=True \
  ATTN_RUN_TAG="${TAG}" \
  python3 main_aro.py \
    --dataset="${DATASET}" \
    --model-name="${MODEL_NAME}" \
    --download \
    --method=adapt_vis \
    --weight1="${W1}" \
    --weight2="${W2}" \
    --threshold="${CONF_THR}" \
    --option="${OPTION}"

  cp "${SRC_RESULT}" "${OUT_DIR}/results_${TAG}.json"
  cp "${SRC_SCORE}" "${OUT_DIR}/scores_${TAG}.json"
}

# B: 去掉最上排 0,1,2,3
run_group "except_top_row" "4,5,6,7,8,9,10,11,12,13,14,15"

# C: 去掉第二排 4,5,6,7
run_group "except_second_row" "0,1,2,3,8,9,10,11,12,13,14,15"

# D: 去掉最底排 12,13,14,15
run_group "except_bottom_row" "0,1,2,3,4,5,6,7,8,9,10,11"

echo "Done. Results saved to ${OUT_DIR}"
