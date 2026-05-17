#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/relation_contribution_probe_logitadd_L15_18_blocks"
ID_DIR="${ROOT_DIR}/output/relation_contribution_probe"
ID_FILE="${ID_DIR}/ids_gold_on_under.txt"

mkdir -p "${OUT_DIR}" "${ID_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

NUM_GPUS=2

# 跑 original / black / shuffle，方便判断 add 是视觉依赖还是 bias-like。
CONTROLS=("original" "blank_black" "shuffle_patches")

# 只跑 15-18 层。
LAYERS=(15 16 17 18)

# logit-add 强度。
# mode=std 时：实际 beta = alpha × 当前 image-token attention-logit std。
# +0.5 比较温和，+1.0 更明显。
ALPHAS=("0.5" "1.0")

# block 组合。
# 4x4 block id:
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
GROUP_NAMES=("all" "bottom_row" "b13_14" "center" "b13" "b14")
GROUP_BLOCKS=(
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  "12,13,14,15"
  "13,14"
  "5,6,9,10"
  "13"
  "14"
)

echo "[CHECK] compile python files"
python -m py_compile main_aro.py
python -m py_compile model_zoo/llava15.py
python -m py_compile model_zoo/llama/modeling_llama_add_attn.py

grep -q "probe_add" main_aro.py || { echo "[ERROR] probe_add not found in main_aro.py"; exit 1; }
grep -q "probe_add" model_zoo/llava15.py || { echo "[ERROR] probe_add not found in llava15.py"; exit 1; }
grep -q "PROBE_ADD_LOGIT" model_zoo/llama/modeling_llama_add_attn.py || { echo "[ERROR] PROBE_ADD_LOGIT not found in modeling_llama_add_attn.py"; exit 1; }
grep -q "PROBE_ADD_BETA_MODE" model_zoo/llama/modeling_llama_add_attn.py || { echo "[ERROR] PROBE_ADD_BETA_MODE not found in modeling_llama_add_attn.py"; exit 1; }

# ------------------------------------------------------------
# Build gold=on/under id file
# ------------------------------------------------------------

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

control_env_value () {
  local CONTROL="$1"
  if [[ "${CONTROL}" == "original" ]]; then
    echo "none"
  else
    echo "${CONTROL}"
  fi
}

alpha_tag () {
  local A="$1"
  if [[ "${A}" == "0.5" ]]; then
    echo "p05"
  elif [[ "${A}" == "1.0" ]]; then
    echo "p10"
  elif [[ "${A}" == "-0.5" ]]; then
    echo "m05"
  elif [[ "${A}" == "-1.0" ]]; then
    echo "m10"
  else
    echo "$(echo "${A}" | sed 's/-/m/g; s/+//g; s/\.//g')"
  fi
}

run_baseline () {
  local GPU="$1"
  local CONTROL="$2"

  local IMAGE_CONTROL_VALUE
  IMAGE_CONTROL_VALUE=$(control_env_value "${CONTROL}")

  local TAG="baseline_on_under_relprob_${CONTROL}"

  local RESULT_SRC="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${TAG}.json"
  local SCORE_SRC="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${TAG}scores.json"

  local RESULT_DST="${OUT_DIR}/results_${TAG}.json"
  local SCORE_DST="${OUT_DIR}/scores_${TAG}.json"

  if [[ -f "${RESULT_DST}" ]]; then
    echo "[SKIP] baseline exists: ${RESULT_DST}"
    return
  fi

  echo ""
  echo "============================================================"
  echo "GPU=${GPU} BASELINE CONTROL=${CONTROL} IMAGE_CONTROL=${IMAGE_CONTROL_VALUE}"
  echo "TAG=${TAG}"
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

  if [[ -f "${RESULT_SRC}" ]]; then
    cp "${RESULT_SRC}" "${RESULT_DST}"
  else
    echo "[WARN] missing baseline result file: ${RESULT_SRC}"
  fi

  if [[ -f "${SCORE_SRC}" ]]; then
    cp "${SCORE_SRC}" "${SCORE_DST}"
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
  local ALPHA="$6"

  local IMAGE_CONTROL_VALUE
  IMAGE_CONTROL_VALUE=$(control_env_value "${CONTROL}")

  local ATAG
  ATAG=$(alpha_tag "${ALPHA}")

  local TAG="logitadd_${CONTROL}_L${LAYER}_${GNAME}_stdalpha_${ATAG}"

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
  echo "GPU=${GPU} CONTROL=${CONTROL} IMAGE_CONTROL=${IMAGE_CONTROL_VALUE}"
  echo "TAG=${TAG}"
  echo "layer=${LAYER}, group=${GNAME}, blocks=${BLOCKS}, alpha=${ALPHA}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  IMAGE_CONTROL="${IMAGE_CONTROL_VALUE}" \
  IMAGE_CONTROL_SEED=1 \
  IMAGE_CONTROL_SIZE=336 \
  IMAGE_CONTROL_GRID=24 \
  CLIP_OBJ_MASK=False \
  ADJUST_METHOD=probe_add \
  PATCH_MASK_MODE=all \
  PATCH_GRID_SIZE=24 \
  PATCH_BLOCK_GRID=4 \
  PATCH_MASK_DEBUG=False \
  PROBE_SAMPLE_IDS_FILE="${ID_FILE}" \
  PROBE_LAYER="${LAYER}" \
  PROBE_HEAD="-1" \
  PROBE_BLOCK_IDS="${BLOCKS}" \
  PROBE_ADD_BETA_MODE=std \
  PROBE_ADD_ALPHA="${ALPHA}" \
  PROBE_ADD_BETA_CLAMP=2.0 \
  PROBE_ADD_STD_EPS=1e-6 \
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

# ------------------------------------------------------------
# Run baselines
# ------------------------------------------------------------

run_baseline 0 "original" &
run_baseline 1 "blank_black" &
wait
run_baseline 0 "shuffle_patches"

# ------------------------------------------------------------
# Build jobs
# ------------------------------------------------------------

JOBS_FILE="${OUT_DIR}/jobs.txt"
: > "${JOBS_FILE}"

for CONTROL in "${CONTROLS[@]}"; do
  for LAYER in "${LAYERS[@]}"; do
    for gi in "${!GROUP_NAMES[@]}"; do
      GNAME="${GROUP_NAMES[$gi]}"
      BLOCKS="${GROUP_BLOCKS[$gi]}"

      for ALPHA in "${ALPHAS[@]}"; do
        echo "${CONTROL}|${LAYER}|${GNAME}|${BLOCKS}|${ALPHA}" >> "${JOBS_FILE}"
      done
    done
  done
done

TOTAL_JOBS=$(wc -l < "${JOBS_FILE}")

echo ""
echo "[INFO] Total probe jobs: ${TOTAL_JOBS}"
echo "[INFO] Expected result files: $((TOTAL_JOBS + 3)) including baselines"
echo "[INFO] Output dir: ${OUT_DIR}"

# ------------------------------------------------------------
# 2-GPU worker
# ------------------------------------------------------------

worker () {
  local GPU="$1"
  local OFFSET="$2"
  local JOB_ID=0

  while IFS='|' read -r CONTROL LAYER GNAME BLOCKS ALPHA; do
    if (( JOB_ID % NUM_GPUS == OFFSET )); then
      run_one "${GPU}" "${CONTROL}" "${LAYER}" "${GNAME}" "${BLOCKS}" "${ALPHA}"
    fi
    JOB_ID=$((JOB_ID + 1))
  done < "${JOBS_FILE}"
}

worker 0 0 &
worker 1 1 &
wait

echo ""
echo "All logit-add jobs finished."
echo "Results saved to ${OUT_DIR}"

echo "[CHECK] result files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | wc -l

echo "[CHECK] score files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "scores_*.json" | wc -l

echo "[CHECK] first few result files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | sort | head -n 10
