#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/adaptvis_exclude_L15_18_original_all_b1314"
ID_DIR="${ROOT_DIR}/output/relation_contribution_probe"
ID_FILE="${ID_DIR}/ids_gold_on_under.txt"

mkdir -p "${OUT_DIR}" "${ID_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"
NUM_GPUS=2

# 只跑原图，不跑 blank / shuffle。
CONTROL="original"
IMAGE_CONTROL_VALUE="none"

# 只跑 exclude L15-L18，不跑 full_adaptvis baseline。
EXCLUDE_LAYERS="15,16,17,18"

# 两个区域：
# all: 全部 image tokens
# b13_14: bottom-center 两个 block
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
  echo "Your llava15.py may still make PROBE_RUN_TAG trigger single-pass."
  exit 1
}

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

echo "[CHECK] number of on/under ids:"
wc -l "${ID_FILE}"

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
  local GPU="$1"
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

  local TAG="adaptvis_exclude_L15_18_${GROUP}_original_${WTAG}"

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
  echo "CONTROL=${CONTROL} IMAGE_CONTROL=${IMAGE_CONTROL_VALUE}"
  echo "GROUP=${GROUP} ADJUST_METHOD=${ADJUST_METHOD_VALUE}"
  echo "WEIGHT=${WEIGHT}"
  echo "EXCLUDE_LAYERS=${EXCLUDE_LAYERS}"
  echo "TAG=${TAG}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  ADAPTVIS_EXCLUDE_LAYERS="${EXCLUDE_LAYERS}" \
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
  PROBE_SAMPLE_IDS_FILE="${ID_FILE}" \
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

# ------------------------------------------------------------
# Build jobs
# ------------------------------------------------------------

JOBS_FILE="${OUT_DIR}/jobs.txt"
: > "${JOBS_FILE}"

for GROUP in "${GROUP_NAMES[@]}"; do
  for WEIGHT in "${WEIGHTS[@]}"; do
    echo "${GROUP}|${WEIGHT}" >> "${JOBS_FILE}"
  done
done

TOTAL_JOBS=$(wc -l < "${JOBS_FILE}")

echo ""
echo "[INFO] total jobs: ${TOTAL_JOBS}"
echo "[INFO] expected result files: ${TOTAL_JOBS}"
echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] jobs:"
cat "${JOBS_FILE}"

# ------------------------------------------------------------
# 2-GPU worker
# ------------------------------------------------------------

worker () {
  local GPU="$1"
  local OFFSET="$2"
  local JOB_ID=0

  while IFS='|' read -r GROUP WEIGHT; do
    if (( JOB_ID % NUM_GPUS == OFFSET )); then
      run_one "${GPU}" "${GROUP}" "${WEIGHT}"
    fi
    JOB_ID=$((JOB_ID + 1))
  done < "${JOBS_FILE}"
}

worker 0 0 &
worker 1 1 &
wait

echo ""
echo "All exclude-L15-L18 AdaptVis jobs finished."
echo "Results saved to ${OUT_DIR}"

echo "[CHECK] result files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | wc -l

echo "[CHECK] score files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "scores_*.json" | wc -l

echo "[CHECK] files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "results_*.json" | sort
