#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-llava1.5}"

CONF_THR="${CONF_THR:-0.4}"
W1_ADAPT="${W1_ADAPT:-0.5}"
W2_ADAPT="${W2_ADAPT:-1.5}"

GRID_SIZE="${GRID_SIZE:-24}"
BLOCK_GRID="${BLOCK_GRID:-4}"

export SAVE_LAYERS="${SAVE_LAYERS:--1}"
export PROBE_RELATION_PROBS=True
export PROBE_RELATION_TOPK="${PROBE_RELATION_TOPK:-10}"

choose_option () {
  local DATASET="$1"

  if [[ -f "prompts/${DATASET}_with_answer_four_options.jsonl" ]]; then
    echo "four"
  elif [[ -f "prompts/${DATASET}_with_answer_two_options.jsonl" ]]; then
    echo "two"
  else
    echo "four"
  fi
}

relation_set_for_dataset () {
  local DATASET="$1"
  if [[ "${DATASET}" == "Controlled_Images_B" ]]; then
    echo "controlled_b"
  elif [[ "${DATASET}" == "COCO_QA_two_obj" ]]; then
    echo "coco_two_obj"
  else
    echo "controlled_a"
  fi
}

run_one_cfg () {
  local DATASET="$1"
  local OPTION="$2"
  local REL_SET="$3"
  local NAME="$4"
  local MODE="$5"
  local IDS="$6"
  local W1="$7"
  local W2="$8"

  local OUT_DIR="output/relation_prob_probe_${DATASET}_grid${BLOCK_GRID}"
  mkdir -p "${OUT_DIR}"

  local SRC_RESULT="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False.json"
  local SRC_SCORE="output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_Falsescores.json"

  echo ""
  echo "============================================================"
  echo "Dataset=${DATASET} option=${OPTION} relation_set=${REL_SET}"
  echo "Running ${NAME}: mode=${MODE}, ids=${IDS}, w1=${W1}, w2=${W2}"
  echo "============================================================"

  local TAG="relprob_${DATASET}_${NAME}_grid${BLOCK_GRID}_conf${CONF_THR}_w1${W1}_w2${W2}"

  if [[ "${MODE}" == "blocks" ]]; then
    CLIP_OBJ_MASK=False \
    ADJUST_METHOD=object_mask \
    PATCH_MASK_MODE=blocks \
    PATCH_BLOCK_IDS="${IDS}" \
    PATCH_GRID_SIZE="${GRID_SIZE}" \
    PATCH_BLOCK_GRID="${BLOCK_GRID}" \
    PATCH_MASK_DEBUG=True \
    ATTN_RUN_TAG="${TAG}" \
    PROBE_RELATION_SET="${REL_SET}" \
    PROBE_RELATION_PROBS=True \
    PROBE_RELATION_TOPK="${PROBE_RELATION_TOPK}" \
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
    PROBE_RELATION_SET="${REL_SET}" \
    PROBE_RELATION_PROBS=True \
    PROBE_RELATION_TOPK="${PROBE_RELATION_TOPK}" \
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

  python - "${OUT_DIR}/results_${NAME}.json" "${DATASET}" "${NAME}" <<'PY'
import json
import sys

path, dataset, name = sys.argv[1:4]

with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)

print("")
print("############################################################")
print(f"Preview: dataset={dataset}, cfg={name}")
print(f"File: {path}")
print("############################################################")

shown = 0

for r in rows:
    probe = r.get("relation_probe")
    if not probe or not probe.get("found", False):
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
        f"relation={probe.get('relation')} | "
        f"step={probe.get('step')} | "
        f"token={probe.get('generated_token')!r} | "
        f"prob={probe.get('generated_token_prob')}"
    )
    print(f"relation_set={probe.get('relation_set')}")
    print(f"hits={probe.get('relation_hits')}")
    print("top10:")
    for t in probe.get("top_tokens", []):
        print(
            f"  {t['rank']:2d}. token={t['token']!r:>12s} "
            f"clean={t['token_clean']!r:>10s} prob={t['prob']:.6f}"
        )

    cand = probe.get("relation_candidate_probs", {})
    if cand:
        print("candidate relation probs:")
        for rel, info in cand.items():
            singles = info.get("single_token", {})
            parts = []
            for alias, val in singles.items():
                p = val.get("prob")
                if p is not None:
                    parts.append(f"{alias}={p:.6f}")
            print(f"  {rel}: " + ", ".join(parts))

            if rel == "in-front":
                front_parts = info.get("in_front_parts", {})
                part_s = []
                for k, val in front_parts.items():
                    p = val.get("prob")
                    if p is not None:
                        part_s.append(f"{k}={p:.6f}")
                if part_s:
                    print("    in_front_parts: " + ", ".join(part_s))

    shown += 1
    if shown >= 5:
        break

if shown == 0:
    print("No found relation_probe rows in preview.")
PY
}

run_dataset () {
  local DATASET="$1"
  local OPTION
  OPTION="$(choose_option "${DATASET}")"

  local REL_SET
  REL_SET="$(relation_set_for_dataset "${DATASET}")"

  run_one_cfg "${DATASET}" "${OPTION}" "${REL_SET}" "noscale_w1" "all" "" "1.0" "1.0"
  run_one_cfg "${DATASET}" "${OPTION}" "${REL_SET}" "bottom13_14_adaptvis" "blocks" "13,14" "${W1_ADAPT}" "${W2_ADAPT}"
  run_one_cfg "${DATASET}" "${OPTION}" "${REL_SET}" "global_all_adaptvis" "all" "" "${W1_ADAPT}" "${W2_ADAPT}"

  echo ""
  echo "Done dataset=${DATASET}"
  echo "Results saved to output/relation_prob_probe_${DATASET}_grid${BLOCK_GRID}/"
}

run_dataset "Controlled_Images_B"
run_dataset "COCO_QA_two_obj"

echo ""
echo "All done."
