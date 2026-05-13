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

OUT_DIR="output/on_multi_block_groups_grid${BLOCK_GRID}"
mkdir -p "${OUT_DIR}"

export SAVE_LAYERS=-1

SRC_RESULT="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False.json"
SRC_SCORE="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_Falsescores.json"

declare -A GROUPS

GROUPS["row3_bottom"]="12,13,14,15"
GROUPS["row2_lower_middle"]="8,9,10,11"
GROUPS["lower_half"]="8,9,10,11,12,13,14,15"

GROUPS["center"]="5,6,9,10"
GROUPS["lower_center"]="9,10,13,14"

GROUPS["bottom_mid"]="13,14"
GROUPS["bottom_mid_right"]="13,14,15"
GROUPS["bottom_left_mid"]="12,13,14"

GROUPS["quad_bottom_left"]="8,9,12,13"
GROUPS["quad_bottom_right"]="10,11,14,15"

GROUPS["contact_band"]="9,10,11,13,14,15"

for NAME in "${!GROUPS[@]}"; do
  IDS="${GROUPS[$NAME]}"

  echo "========================================"
  echo "Running group: ${NAME} = ${IDS}"
  echo "========================================"

  TAG="onmulti_${NAME}_blocks${IDS//,/plus}_grid${BLOCK_GRID}_conf${CONF_THR}_w1${W1}_w2${W2}"

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
done

echo "Done. Results saved to ${OUT_DIR}"
