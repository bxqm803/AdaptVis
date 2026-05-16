#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="${ROOT_DIR}/output/relation_contribution_probe_scale_mid"
ID_DIR="${ROOT_DIR}/output/relation_contribution_probe"
ID_FILE="${ID_DIR}/ids_gold_on_under.txt"

mkdir -p "${OUT_DIR}" "${ID_DIR}"

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

# ------------------------------------------------------------
# 0. Basic checks
# ------------------------------------------------------------

python -m py_compile main_aro.py
python -m py_compile model_zoo/llava15.py
python -m py_compile model_zoo/llama/modeling_llama_add_attn.py
python -m py_compile model_zoo/llava/modeling_llava_scal.py

grep -q "probe_scale" model_zoo/llama/modeling_llama_add_attn.py
grep -q "PROBE_SCALE" model_zoo/llama/modeling_llama_add_attn.py
grep -q "probe_scale" main_aro.py
grep -q "PROBE_SCALE" main_aro.py
grep -q "probe_scale" model_zoo/llava15.py
grep -q "PROBE_SCALE" model_zoo/llava15.py

# ------------------------------------------------------------
# 1. Build gold=on/under sample ids from prompt file
# ------------------------------------------------------------

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
    if "left" in x:
        return "left"
    if "right" in x:
        return "right"
    return "unknown"

ids = []

with open(prompt_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        ans = d.get("answer", "")

        if isinstance(ans, list):
            gold = ans[0] if ans else ""
        else:
            gold = ans

        if norm(gold) in ["on", "under"]:
            ids.append(i)

with open(out_file, "w", encoding="utf-8") as f:
    for i in ids:
        f.write(str(i) + "\n")

print(f"[ID FILE] wrote {len(ids)} gold on/under ids to {out_file}")
PY

# Expected: 206
echo "[CHECK] number of ids:"
wc -l "${ID_FILE}"

# ------------------------------------------------------------
# 2. Probe config
# ------------------------------------------------------------

LAYERS=(12 16 20 24)
SCALES=("0.5" "1.5")

GROUP_NAMES=(
  "b13"
  "b14"
  "b13_14"
  "bottom_row"
  "center"
  "all"
)

GROUP_BLOCKS=(
  "13"
  "14"
  "13,14"
  "12,13,14,15"
  "5,6,9,10"
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
)

GPUS=(0 1)
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

run_one () {
  local GPU="$1"
  local LAYER="$2"
  local GNAME="$3"
  local BLOCKS="$4"
  local SCALE="$5"

  local STAG
  STAG=$(scale_tag "${SCALE}")

  local TAG="scale_L${LAYER}_${GNAME}_hall_${STAG}"

  echo ""
  echo "============================================================"
  echo "GPU=${GPU} TAG=${TAG}"
  echo "layer=${LAYER}, group=${GNAME}, blocks=${BLOCKS}, scale=${SCALE}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
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

worker () {
  local GPU="$1"
  local OFFSET="$2"
  local JOB_ID=0

  for SCALE in "${SCALES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
      for i in "${!GROUP_NAMES[@]}"; do
        if (( JOB_ID % NUM_GPUS == OFFSET )); then
          run_one "${GPU}" "${LAYER}" "${GROUP_NAMES[$i]}" "${GROUP_BLOCKS[$i]}" "${SCALE}"
        fi
        JOB_ID=$((JOB_ID + 1))
      done
    done
  done
}

worker 0 0 &
worker 1 1 &

wait

echo ""
echo "All scale probe jobs finished."
echo "Results saved to ${OUT_DIR}"

echo ""
echo "[CHECK] result files:"
ls "${OUT_DIR}"/results_*.json | wc -l

echo "[CHECK] score files:"
ls "${OUT_DIR}"/scores_*.json | wc -l
