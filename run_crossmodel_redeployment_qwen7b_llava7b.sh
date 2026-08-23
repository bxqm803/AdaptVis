#!/usr/bin/env bash
set -euo pipefail

# Run from the AdaptVis llava16 repository root.
# GPU0: Qwen2.5-VL-7B
# GPU1: LLaVA-1.5-7B
#
# Each model runs three sequential stages:
#   1) extract Image / NoImage / residual Direction-head vectors
#   2) build held-out multi-head consensus + baseline generation
#   3) test relation-specific last-token repair over model-relative layer windows

DATA_ROOT="${DATA_ROOT:-data}"
SEED="${SEED:-17}"

run_one () {
  local MODEL="$1"
  local GPU="$2"
  local TAG="$3"

  local DIR_OUT="output/${TAG}_coco_head_direction_residual"
  local FEAS_OUT="output/${TAG}_coco_grounded_consensus_v1"
  local REPAIR_OUT="output/${TAG}_coco_relation_delta_crossmodel_v1"

  echo "================================================================================"
  echo "MODEL=${MODEL} GPU=${GPU} TAG=${TAG}"
  echo "================================================================================"

  if [[ ! -f "${DIR_OUT}/relation_vectors.npz" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" python analyze_coco_head_object_residual_direction_probe_v1.py \
      --dataset coco_two \
      --data-root "${DATA_ROOT}" \
      --model "${MODEL}" \
      --device cuda:0 \
      --attn-impl eager \
      --train-ratio 0.15 \
      --repeats 5 \
      --output-dir "${DIR_OUT}" \
      --overwrite
  else
    echo "[reuse] ${DIR_OUT}/relation_vectors.npz"
  fi

  if [[ ! -f "${FEAS_OUT}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" python validate_grounded_spatial_consensus_v1.py \
      --direction-dir "${DIR_OUT}" \
      --dataset coco_two \
      --data-root "${DATA_ROOT}" \
      --model "${MODEL}" \
      --device cuda:0 \
      --attn-impl eager \
      --seed "${SEED}" \
      --output-dir "${FEAS_OUT}" \
      --overwrite
  else
    echo "[reuse] ${FEAS_OUT}/summary.json"
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" python eval_relation_delta_crossmodel_v1.py \
    --feasibility-dir "${FEAS_OUT}" \
    --dataset coco_two \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --device cuda:0 \
    --attn-impl eager \
    --single-layers all \
    --alpha 1.0 \
    --max-delta-ratio 0.15 \
    --policy conflict_only \
    --seed "${SEED}" \
    --output-dir "${REPAIR_OUT}" \
    --overwrite
}

run_one qwen-7b 0 qwen7b &
PID0=$!
run_one llava-7b 1 llava7b &
PID1=$!

wait "${PID0}"
wait "${PID1}"

echo "Done."
echo "  output/qwen7b_coco_relation_delta_crossmodel_v1/summary.json"
echo "  output/llava7b_coco_relation_delta_crossmodel_v1/summary.json"
