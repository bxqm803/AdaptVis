#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/relation_contribution_probe_mid12_24"
mkdir -p "${OUT_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

ID_FILE="${ROOT_DIR}/output/relation_contribution_probe/ids_gold_on_under.txt"

LAYERS=(12 16 20 24)

GROUP_NAMES=(
  "b13"
  "b14"
  "b13_14"
  "b12_13"
  "b14_15"
  "b9_13"
  "b10_14"
  "third_row"
  "bottom_row"
  "bottom_half"
  "center"
  "all"
)

GROUP_BLOCKS=(
  "13"
  "14"
  "13,14"
  "12,13"
  "14,15"
  "9,13"
  "10,14"
  "8,9,10,11"
  "12,13,14,15"
  "8,9,10,11,12,13,14,15"
  "5,6,9,10"
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
)

GPUS=(0 1)
NUM_GPUS=2

run_one () {
  local GPU="$1"
  local LAYER="$2"
  local GNAME="$3"
  local BLOCKS="$4"

  local TAG="probe_L${LAYER}_${GNAME}_hall_beta005"

  echo ""
  echo "============================================================"
  echo "GPU=${GPU} TAG=${TAG}"
  echo "layer=${LAYER}, group=${GNAME}, blocks=${BLOCKS}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  CLIP_OBJ_MASK=False \
  ADJUST_METHOD=probe_bias \
  PATCH_MASK_MODE=all \
  PATCH_GRID_SIZE=24 \
  PATCH_BLOCK_GRID=4 \
  PATCH_MASK_DEBUG=False \
  PROBE_SAMPLE_IDS_FILE="${ID_FILE}" \
  PROBE_LAYER="${LAYER}" \
  PROBE_HEAD="-1" \
  PROBE_BLOCK_IDS="${BLOCKS}" \
  PROBE_BETA="0.05" \
  PROBE_DEBUG=False \
  PROBE_RELATION_PROBS=True \
  PROBE_RELATION_TOPK=10 \
  PROBE_RUN_TAG="${TAG}" \
  ATTN_RUN_TAG="${TAG}" \
  SAVE_LAYERS=-1 \
  python3 main_aro.py \
    --dataset="${DATASET}" \
    --model-name="${MODEL_NAME}" \
    --download \
    --method=adapt_vis \
    --weight1=1.0 \
    --weight2=1.0 \
    --threshold=1.0 \
    --option="${OPTION}"

  local RESULT_SRC="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${TAG}.json"
  local SCORE_SRC="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${TAG}scores.json"

  if [[ -f "${RESULT_SRC}" ]]; then
    cp "${RESULT_SRC}" "${OUT_DIR}/results_${TAG}.json"
  else
    echo "[WARN] missing result file: ${RESULT_SRC}"
  fi

  if [[ -f "${SCORE_SRC}" ]]; then
    cp "${SCORE_SRC}" "${OUT_DIR}/scores_${TAG}.json"
  else
    echo "[WARN] missing score file: ${SCORE_SRC}"
  fi
}

worker () {
  local GPU="$1"
  local OFFSET="$2"
  local JOB_ID=0

  for LAYER in "${LAYERS[@]}"; do
    for i in "${!GROUP_NAMES[@]}"; do
      if (( JOB_ID % NUM_GPUS == OFFSET )); then
        run_one "${GPU}" "${LAYER}" "${GROUP_NAMES[$i]}" "${GROUP_BLOCKS[$i]}"
      fi
      JOB_ID=$((JOB_ID + 1))
    done
  done
}

worker 0 0 &
worker 1 1 &

wait

echo ""
echo "All probe jobs finished."
echo "Results saved to ${OUT_DIR}"
