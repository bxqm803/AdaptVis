#!/usr/bin/env bash
set -euo pipefail

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

W1="0.5"
W2="1.5"
CONF_THR="0.4"

OUT_DIR="output/object_mask_threshold_ablation"
mkdir -p "${OUT_DIR}"

# 保存中间层 attention；15 是中间层，31 是最后一层
export SAVE_LAYERS=17

for OBJ_THR in 0.0 0.2 0.4 0.6 0.7; do
  echo "========================================"
  echo "Running object-mask threshold = ${OBJ_THR}"
  echo "========================================"

  TAG="objmask_objthr${OBJ_THR}_layer${SAVE_LAYERS}_conf${CONF_THR}_w1${W1}_w2${W2}"

  CLIP_OBJ_MASK=True \
  CLIP_OBJ_MODEL=openai/clip-vit-large-patch14-336 \
  CLIP_OBJ_THRESHOLD="${OBJ_THR}" \
  CLIP_OBJ_DILATE=1 \
  CLIP_OBJ_INVERT=True \
  CLIP_OBJ_DEBUG=True \
  ADJUST_METHOD=object_mask \
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

  SRC_RESULT="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False.json"
  SRC_SCORE="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_Falsescores.json"

  cp "${SRC_RESULT}" "${OUT_DIR}/results_${TAG}.json"
  cp "${SRC_SCORE}" "${OUT_DIR}/scores_${TAG}.json"

  echo "Saved:"
  echo "  ${OUT_DIR}/results_${TAG}.json"
  echo "  ${OUT_DIR}/scores_${TAG}.json"
done
