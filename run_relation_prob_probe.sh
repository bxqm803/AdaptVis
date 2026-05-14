#!/usr/bin/env bash
set -euo pipefail

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"

CONF_THR="0.4"

GRID_SIZE="24"
BLOCK_GRID="4"

OUT_DIR="output/relation_prob_probe_grid${BLOCK_GRID}"
mkdir -p "${OUT_DIR}"

export SAVE_LAYERS=-1
export PROBE_RELATION_PROBS=True
export PROBE_RELATION_TOPK=10

SRC_RESULT="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False.json"
SRC_SCORE="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_Falsescores.json"

print_probe_preview () {
  local RESULT_PATH="$1"
  local NAME="$2"

  echo ""
  echo "############################################################"
  echo "Preview relation top-10 probabilities: ${NAME}"
  echo "File: ${RESULT_PATH}"
  echo "############################################################"

  python - "${RESULT_PATH}" "${NAME}" <<'PY'
import json
import sys

path = sys.argv[1]
name = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)

shown = 0

print(f"\n===== {name} =====")

for r in rows:
    probe = r.get("relation_probe", None)

    if not probe:
        continue

    if not probe.get("found", False):
        continue

    print("\n----------------------------------------")
    print(
        f"sample_id={r.get('sample_id')} | "
        f"gold={r.get('Golden')} | "
        f"correct={r.get('Correct')} | "
        f"selected_weight={r.get('selected_weight')}"
    )
    print(f"generation: {r.get('Generation')!r}")
    print(
        f"relation_token={probe.get('relation')} | "
        f"step={probe.get('step')} | "
        f"generated_token={probe.get('generated_token')!r} | "
        f"generated_token_prob={probe.get('generated_token_prob'):.6f}"
    )
    print(f"text_until_relation: {probe.get('generated_text_until_relation')!r}")
    print("top10:")

    for t in probe.get("top_tokens", []):
        print(
            f"  {t['rank']:2d}. "
            f"token={t['token']!r:>12s} | "
            f"clean={t['token_clean']!r:>8s} | "
            f"prob={t['prob']:.6f}"
        )

    shown += 1
    if shown >= 5:
        break

if shown == 0:
    print("No relation_probe with found=True in this result file.")
PY
}

run_cfg () {
  local NAME="$1"
  local MODE="$2"
  local IDS="$3"
  local W1="$4"
  local W2="$5"

  echo ""
  echo "============================================================"
  echo "Running ${NAME}"
  echo "mode=${MODE}, ids=${IDS}, w1=${W1}, w2=${W2}"
  echo "============================================================"

  TAG="relprob_${NAME}_grid${BLOCK_GRID}_conf${CONF_THR}_w1${W1}_w2${W2}"

  if [[ "${MODE}" == "blocks" ]]; then
    CLIP_OBJ_MASK=False \
    ADJUST_METHOD=object_mask \
    PATCH_MASK_MODE=blocks \
    PATCH_BLOCK_IDS="${IDS}" \
    PATCH_GRID_SIZE="${GRID_SIZE}" \
    PATCH_BLOCK_GRID="${BLOCK_GRID}" \
    PATCH_MASK_DEBUG=True \
    ATTN_RUN_TAG="${TAG}" \
    PROBE_RELATION_PROBS=True \
    PROBE_RELATION_TOPK=10 \
    python3 main_aro.py \
      --dataset="${DATASET}" \
      --model-name="${MODEL_NAME}" \
      --download \
      --method=adapt_vis \
      --weight1="${W1}" \
      --weight2="${W2}" \
      --threshold="${CONF_THR}" \
      --option="${OPTION}"

  elif [[ "${MODE}" == "all" ]]; then
    CLIP_OBJ_MASK=False \
    ADJUST_METHOD=object_mask \
    PATCH_MASK_MODE=all \
    PATCH_GRID_SIZE="${GRID_SIZE}" \
    PATCH_BLOCK_GRID="${BLOCK_GRID}" \
    PATCH_MASK_DEBUG=True \
    ATTN_RUN_TAG="${TAG}" \
    PROBE_RELATION_PROBS=True \
    PROBE_RELATION_TOPK=10 \
    python3 main_aro.py \
      --dataset="${DATASET}" \
      --model-name="${MODEL_NAME}" \
      --download \
      --method=adapt_vis \
      --weight1="${W1}" \
      --weight2="${W2}" \
      --threshold="${CONF_THR}" \
      --option="${OPTION}"

  else
    echo "Unknown MODE=${MODE}"
    exit 1
  fi

  cp "${SRC_RESULT}" "${OUT_DIR}/results_${NAME}.json"
  cp "${SRC_SCORE}" "${OUT_DIR}/scores_${NAME}.json"

  print_probe_preview "${OUT_DIR}/results_${NAME}.json" "${NAME}"
}

# 1. 不乘系数：第一轮/第二轮都是 weight=1.0
run_cfg "noscale_w1" "all" "" "1.0" "1.0"

# 2. 只在 block 13,14 上按照 AdaptVis confidence 规则乘系数
run_cfg "bottom13_14_adaptvis" "blocks" "13,14" "0.5" "1.5"

# 3. 全局 image tokens 按照 AdaptVis confidence 规则乘系数
run_cfg "global_all_adaptvis" "all" "" "0.5" "1.5"

echo ""
echo "Done. Results saved to ${OUT_DIR}"
