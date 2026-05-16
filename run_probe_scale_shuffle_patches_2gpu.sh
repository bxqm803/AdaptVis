#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/relation_contribution_probe_shuffle_patches"
ID_DIR="${ROOT_DIR}/output/relation_contribution_probe"
ID_FILE="${ID_DIR}/ids_gold_on_under.txt"

mkdir -p "${OUT_DIR}" "${ID_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

CONTROL="shuffle_patches"

python -m py_compile main_aro.py
python -m py_compile model_zoo/llava15.py
python -m py_compile model_zoo/llama/modeling_llama_add_attn.py
python -m py_compile model_zoo/llava/modeling_llava_scal.py

grep -q "IMAGE_CONTROL" model_zoo/llava15.py || { echo "[ERROR] IMAGE_CONTROL not found in llava15.py"; exit 1; }
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

# 只跑前面 signal 强的组合，和 blank_black 对齐，方便比较。
# format: layer|group_name|block_ids|scale
JOBS=(
  "12|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0.5"
  "12|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|1.5"
  "12|b13_14|13,14|0.5"
  "12|bottom_row|12,13,14,15|0.5"

  "16|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0.5"
  "16|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|1.5"
  "16|bottom_row|12,13,14,15|0.5"
  "16|bottom_row|12,13,14,15|1.5"
  "16|b13_14|13,14|1.5"

  "20|center|5,6,9,10|1.5"
  "20|all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|1.5"
  "20|b13_14|13,14|0.5"
  "20|bottom_row|12,13,14,15|0.5"
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

run_baseline () {
  local GPU="$1"
  local TAG="baseline_on_under_relprob_${CONTROL}"

  echo ""
  echo "============================================================"
  echo "GPU=${GPU} BASELINE CONTROL=${CONTROL} TAG=${TAG}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  IMAGE_CONTROL="${CONTROL}" \
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

  [[ -f "${RESULT_SRC}" ]] && cp "${RESULT_SRC}" "${OUT_DIR}/results_${TAG}.json"
  [[ -f "${SCORE_SRC}" ]] && cp "${SCORE_SRC}" "${OUT_DIR}/scores_${TAG}.json"
}

run_one () {
  local GPU="$1"
  local LAYER="$2"
  local GNAME="$3"
  local BLOCKS="$4"
  local SCALE="$5"

  local STAG
  STAG=$(scale_tag "${SCALE}")

  local TAG="ctrl_${CONTROL}_scale_L${LAYER}_${GNAME}_hall_${STAG}"

  echo ""
  echo "============================================================"
  echo "GPU=${GPU} CONTROL=${CONTROL} TAG=${TAG}"
  echo "layer=${LAYER}, group=${GNAME}, blocks=${BLOCKS}, scale=${SCALE}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  IMAGE_CONTROL="${CONTROL}" \
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

# 先跑 shuffle baseline。
run_baseline 0

worker () {
  local GPU="$1"
  local OFFSET="$2"
  local JOB_ID=0

  for JOB in "${JOBS[@]}"; do
    IFS='|' read -r LAYER GNAME BLOCKS SCALE <<< "${JOB}"

    if (( JOB_ID % NUM_GPUS == OFFSET )); then
      run_one "${GPU}" "${LAYER}" "${GNAME}" "${BLOCKS}" "${SCALE}"
    fi

    JOB_ID=$((JOB_ID + 1))
  done
}

worker 0 0 &
worker 1 1 &
wait

echo ""
echo "All shuffle-patches probe jobs finished."
echo "Results saved to ${OUT_DIR}"

echo "[CHECK] result files:"
ls "${OUT_DIR}"/results_*.json | wc -l

echo "[CHECK] score files:"
ls "${OUT_DIR}"/scores_*.json | wc -l
