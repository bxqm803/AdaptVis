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

OUT_DIR="output/only_block_ablation_grid${BLOCK_GRID}"
mkdir -p "${OUT_DIR}"

export SAVE_LAYERS=-1

SRC_RESULT="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False.json"
SRC_SCORE="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_Falsescores.json"

for BLOCK_ID in $(seq 0 15); do
  echo "========================================"
  echo "Running only block ${BLOCK_ID}"
  echo "========================================"

  TAG="only_block${BLOCK_ID}_grid${BLOCK_GRID}_conf${CONF_THR}_w1${W1}_w2${W2}"

  CLIP_OBJ_MASK=False \
  ADJUST_METHOD=object_mask \
  PATCH_MASK_MODE=block \
  PATCH_GRID_SIZE="${GRID_SIZE}" \
  PATCH_BLOCK_GRID="${BLOCK_GRID}" \
  PATCH_BLOCK_ID="${BLOCK_ID}" \
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
