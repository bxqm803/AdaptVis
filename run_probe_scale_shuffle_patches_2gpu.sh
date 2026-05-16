#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/relation_contribution_probe_mid_bias_controls"
ID_DIR="${ROOT_DIR}/output/relation_contribution_probe"
ID_FILE="${ID_DIR}/ids_gold_on_under.txt"

mkdir -p "${OUT_DIR}" "${ID_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

# 同时跑原图、黑图、shuffle 图
CONTROLS=("original" "blank_black" "shuffle_patches")

python -m py_compile main_aro.py
python -m py_compile model_zoo/llava15.py
python -m py_compile model_zoo/llama/modeling_llama_add_attn.py
python -m py_compile model_zoo/llava/modeling_llava_scal.py

grep -q "IMAGE_CONTROL" model_zoo/llava15.py || { echo "[ERROR] IMAGE_CONTROL not found in llava15.py"; exit 1; }
grep -q "blank_black" model_zoo/llava15.py || { echo "[ERROR] blank_black not found in llava15.py"; exit 1; }
grep -q "shuffle_patches" model_zoo/llava15.py || { echo "[ERROR] shuffle_patches not found in llava15.py"; exit 1; }
grep -q "apply_image_control_from_env" model_zoo/llava15.py || { echo "[ERROR] apply_image_control_from_env not found in llava15.py"; exit 1; }
grep -q "probe_scale" model_zoo/llama/modeling_llama_add_attn.py || { echo "[ERROR] probe_scale not found in modeling_llama_add_attn.py"; exit 1; }

if [[ ! -f "${ID_FILE}" ]]; then
python - <<'PY'
import json
import re
from pathlib import Path

prompt_file = Path("prompts/Controlled_Images_A_with_answer_four_options.jsonl")
out_file = Path("output/relation_contribution_probe/ids_gold_on_under.txt")
out_file.parent.mkdir(parents=True, exist_ok=True)

def norm(x):
    x = str(x).strip().lower()
    if "under" in x:
        return "under"
    if re.search(r"\bon\b", x) and "front" not in x:
        return "on"
    return "other"

ids = []
with open(prompt_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        ans = d.get("answer", "")
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        if norm(ans) in ["on", "under"]:
            ids.append(i)

with open(out_file, "w", encoding="utf-8") as f:
    for i in ids:
        f.write(str(i) + "\n")

print("wrote", len(ids), "ids to", out_file)
PY
fi

echo "[CHECK] number of ids:"
wc -l "${ID_FILE}"

# 只测试 15 / 17 / 18 层，重点看中层附近是否存在 bias-like steering。
# groups 选 all / bottom_row / b13_14，因为之前 L16 的 bias-like 信号主要在 all 和 bottom_row。
# format: layer|group_name|block_ids|scale
JOBS=(
  "15|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0.5"
  "15|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|1.5"
  "15|bottom_row|12,13,14,15|0.5"
  "15|bottom_row|12,13,14,15|1.5"
  "15|b13_14|13,14|0.5"
  "15|b13_14|13,14|1.5"

  "17|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0.5"
  "17|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|1.5"
  "17|bottom_row|12,13,14,15|0.5"
  "17|bottom_row|12,13,14,15|1.5"
  "17|b13_14|13,14|0.5"
  "17|b13_14|13,14|1.5"

  "18|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0.5"
  "18|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|1.5"
  "18|bottom_row|12,13,14,15|0.5"
  "18|bottom_row|12,13,14,15|1.5"
  "18|b13_14|13,14|0.5"
  "18|b13_14|13,14|1.5"
)

NUM_GPUS=2

scale_tag () {
  local S="$1"
  if [[ "${S}" == "0.5" ]]; then
    echo "x05"
  elif [[ "${S}" == "1.5" ]]; then
    echo "x15"
  elif [[ "${S}" == "2.0" ]]; then
    echo "x20"
  else
    echo "x$(echo "${S}" | sed 's/\.//g')"
  fi
}

control_env_value () {
  local CONTROL="$1"
  if [[ "${CONTROL}" == "original" ]]; then
    echo "none"
  else
    echo "${CONTROL}"
  fi
}

run_baseline () {
  local GPU="$1"
  local CONTROL="$2"
  local IMAGE_CONTROL_VALUE
  IMAGE_CONTROL_VALUE=$(control_env_value "${CONTROL}")

  local TAG="baseline_on_under_relprob_${CONTROL}"

  echo ""
  echo "============================================================"
  echo "GPU=${GPU} BASELINE CONTROL=${CONTROL} IMAGE_CONTROL=${IMAGE_CONTROL_VALUE} TAG=${TAG}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  IMAGE_CONTROL="${IMAGE_CONTROL_VALUE}" \
  IMAGE_CONTROL_SEED=1 \
  IMAGE_CONTROL_SIZE=336 \
  IMAGE_CONTROL_GRID=24 \
  CLIP_OBJ_MASK=False \
  ADJUST_METHOD=last_query \
  PATCH_MASK_MODE=all \
  PATCH_GRID_SIZE=24 \
  PATCH_BLOCK_GRID=4 \
  PATCH_MASK_DEBUG=False \
  PROBE_SAMPLE_IDS_FILE="${ID_FILE}" \
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
    echo "[WARN] missing baseline result file: ${RESULT_SRC}"
  fi

  if [[ -f "${SCORE_SRC}" ]]; then
    cp "${SCORE_SRC}" "${OUT_DIR}/scores_${TAG}.json"
  else
    echo "[WARN] missing baseline score file: ${SCORE_SRC}"
  fi
}

run_one () {
  local GPU="$1"
  local CONTROL="$2"
  local LAYER="$3"
  local GNAME="$4"
  local BLOCKS="$5"
  local SCALE="$6"

  local IMAGE_CONTROL_VALUE
  IMAGE_CONTROL_VALUE=$(control_env_value "${CONTROL}")

  local STAG
  STAG=$(scale_tag "${SCALE}")

  local TAG="ctrl_${CONTROL}_scale_L${LAYER}_${GNAME}_hall_${STAG}"

  echo ""
  echo "============================================================"
  echo "GPU=${GPU} CONTROL=${CONTROL} IMAGE_CONTROL=${IMAGE_CONTROL_VALUE} TAG=${TAG}"
  echo "layer=${LAYER}, group=${GNAME}, blocks=${BLOCKS}, scale=${SCALE}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  IMAGE_CONTROL="${IMAGE_CONTROL_VALUE}" \
  IMAGE_CONTROL_SEED=1 \
  IMAGE_CONTROL_SIZE=336 \
  IMAGE_CONTROL_GRID=24 \
  CLIP_OBJ_MASK=False \
  ADJUST_METHOD=probe_scale \
  PATCH_MASK_MODE=all \
  PATCH_GRID_SIZE=24 \
  PATCH_BLOCK_GRID=4 \
  PATCH_MASK_DEBUG=False \
  PROBE_SAMPLE_IDS_FILE="${ID_FILE}" \
  PROBE_LAYER="${LAYER}" \
  PROBE_HEAD="-1" \
  PROBE_BLOCK_IDS="${BLOCKS}" \
  PROBE_SCALE="${SCALE}" \
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

# 先跑三个 baseline：original / black / shuffle。
# baseline 可以并行跑两个，再跑第三个。
run_baseline 0 "original" &
run_baseline 1 "blank_black" &
wait
run_baseline 0 "shuffle_patches"

worker () {
  local GPU="$1"
  local OFFSET="$2"
  local JOB_ID=0

  for CONTROL in "${CONTROLS[@]}"; do
    for JOB in "${JOBS[@]}"; do
      IFS='|' read -r LAYER GNAME BLOCKS SCALE <<< "${JOB}"

      if (( JOB_ID % NUM_GPUS == OFFSET )); then
        run_one "${GPU}" "${CONTROL}" "${LAYER}" "${GNAME}" "${BLOCKS}" "${SCALE}"
      fi

      JOB_ID=$((JOB_ID + 1))
    done
  done
}

worker 0 0 &
worker 1 1 &
wait

echo ""
echo "All mid-layer bias-control probe jobs finished."
echo "Results saved to ${OUT_DIR}"

echo "[CHECK] result files:"
ls "${OUT_DIR}"/results_*.json | wc -l

echo "[CHECK] score files:"
ls "${OUT_DIR}"/scores_*.json | wc -l
