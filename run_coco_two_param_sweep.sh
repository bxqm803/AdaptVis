#!/usr/bin/env bash
set -euo pipefail

DATASET="COCO_QA_two_obj"
MODEL_NAME="${MODEL_NAME:-llava1.5}"
OPTION="${OPTION:-four}"

OUT_DIR="output/coco_two_param_sweep"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

export SAVE_LAYERS="${SAVE_LAYERS:--1}"
export PROBE_RELATION_PROBS=True
export PROBE_RELATION_TOPK="${PROBE_RELATION_TOPK:-10}"
export PROBE_RELATION_SET=coco_two_obj

GRID_SIZE="${GRID_SIZE:-24}"
BLOCK_GRID="${BLOCK_GRID:-4}"

# 如果已经有结果，默认跳过；想重跑就 FORCE=True bash run_coco_two_param_sweep.sh
FORCE="${FORCE:-False}"

# ScalingVis 候选系数
SCALING_WEIGHTS=(0.3 0.5 0.8 1.0 1.2 1.5 2.0 2.5 3.0)

# AdaptVis 候选参数
ADAPT_W1S=(0.3 0.5 0.8 1.0)
ADAPT_W2S=(1.2 1.5 2.0 2.5 3.0)
ADAPT_THRS=(0.2 0.3 0.4 0.5 0.6)

latest_file () {
  local pattern="$1"
  compgen -G "${pattern}" | xargs -r ls -t | head -n 1 || true
}

copy_latest_result () {
  local METHOD="$1"
  local TAG="$2"

  local RES_PATTERN="output/results*_${DATASET}_${METHOD}_*_${OPTION}option_False.json"
  local SCORE_PATTERN="output/results*_${DATASET}_${METHOD}_*_${OPTION}option_Falsescores.json"

  local RES_FILE
  RES_FILE="$(latest_file "${RES_PATTERN}")"

  if [[ -z "${RES_FILE}" ]]; then
    echo "[ERROR] Cannot find result file with pattern:"
    echo "  ${RES_PATTERN}"
    exit 1
  fi

  cp "${RES_FILE}" "${OUT_DIR}/results_${TAG}.json"

  local SCORE_FILE
  SCORE_FILE="$(latest_file "${SCORE_PATTERN}")"

  if [[ -n "${SCORE_FILE}" ]]; then
    cp "${SCORE_FILE}" "${OUT_DIR}/scores_${TAG}.json"
  fi

  echo "[COPIED] ${RES_FILE}"
  echo "      -> ${OUT_DIR}/results_${TAG}.json"
}

run_one () {
  local METHOD="$1"
  local TAG="$2"
  local WEIGHT="$3"
  local W1="$4"
  local W2="$5"
  local THR="$6"

  local DST="${OUT_DIR}/results_${TAG}.json"

  if [[ "${FORCE}" != "True" && -f "${DST}" ]]; then
    echo ""
    echo "[SKIP] ${TAG} already exists."
    return
  fi

  echo ""
  echo "============================================================"
  echo "Running ${TAG}"
  echo "method=${METHOD}, weight=${WEIGHT}, w1=${W1}, w2=${W2}, threshold=${THR}"
  echo "============================================================"

  local LOG_FILE="${LOG_DIR}/${TAG}.log"

  CLIP_OBJ_MASK=False \
  ADJUST_METHOD=object_mask \
  PATCH_MASK_MODE=all \
  PATCH_GRID_SIZE="${GRID_SIZE}" \
  PATCH_BLOCK_GRID="${BLOCK_GRID}" \
  PATCH_MASK_DEBUG=False \
  ATTN_RUN_TAG="${TAG}" \
  PROBE_RELATION_SET=coco_two_obj \
  PROBE_RELATION_PROBS=True \
  PROBE_RELATION_TOPK="${PROBE_RELATION_TOPK}" \
  python3 main_aro.py \
    --dataset="${DATASET}" \
    --model-name="${MODEL_NAME}" \
    --download \
    --method="${METHOD}" \
    --weight="${WEIGHT}" \
    --weight1="${W1}" \
    --weight2="${W2}" \
    --threshold="${THR}" \
    --option="${OPTION}" 2>&1 | tee "${LOG_FILE}"

  copy_latest_result "${METHOD}" "${TAG}"
}

echo "============================================================"
echo "COCO_QA_two_obj parameter sweep"
echo "OUT_DIR=${OUT_DIR}"
echo "OPTION=${OPTION}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "============================================================"

# 0. baseline / no scale
run_one "adapt_vis" "noscale_w1" "1.0" "1.0" "1.0" "1.0"

# 1. ScalingVis sweep
for W in "${SCALING_WEIGHTS[@]}"; do
  TAG="scaling_w${W}"
  run_one "scaling_vis" "${TAG}" "${W}" "1.0" "1.0" "1.0"
done

# 2. AdaptVis sweep
for W1 in "${ADAPT_W1S[@]}"; do
  for W2 in "${ADAPT_W2S[@]}"; do
    for THR in "${ADAPT_THRS[@]}"; do
      TAG="adapt_w1${W1}_w2${W2}_thr${THR}"
      run_one "adapt_vis" "${TAG}" "1.0" "${W1}" "${W2}" "${THR}"
    done
  done
done

echo ""
echo "============================================================"
echo "Evaluating all saved results"
echo "============================================================"

python - <<'PY'
import os
import json
import glob
from collections import defaultdict

OUT_DIR = "output/coco_two_param_sweep"
RELS = ["left", "right", "above", "below"]

def norm_rel(x):
    x = str(x).strip().lower()
    for r in RELS:
        if r in x:
            return r
    return "unknown"

def pred_rel(row):
    probe = row.get("relation_probe")
    if probe and probe.get("found", False):
        rel = norm_rel(probe.get("relation", ""))
        if rel in RELS:
            return rel

    gen = str(row.get("Generation", "")).lower()
    return norm_rel(gen)

def gold_rel(row):
    return norm_rel(row.get("Golden", ""))

def is_correct(row):
    if "Correct" in row:
        return bool(row["Correct"])

    g = gold_rel(row)
    p = pred_rel(row)
    return g == p

records = []

for path in sorted(glob.glob(os.path.join(OUT_DIR, "results_*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    correct = 0
    total = 0
    by_gold = defaultdict(lambda: [0, 0])

    for row in rows:
        g = gold_rel(row)
        if g not in RELS:
            continue

        ok = is_correct(row)
        correct += int(ok)
        total += 1
        by_gold[g][0] += int(ok)
        by_gold[g][1] += 1

    acc = correct / total if total else 0.0

    records.append({
        "name": os.path.basename(path).replace("results_", "").replace(".json", ""),
        "path": path,
        "acc": acc,
        "correct": correct,
        "total": total,
        "by_gold": by_gold,
    })

records.sort(key=lambda x: (x["acc"], x["correct"]), reverse=True)

print("\nTop results:")
print("rank | acc    | correct | name")
print("-" * 90)

for i, r in enumerate(records[:40], start=1):
    print(
        f"{i:4d} | "
        f"{r['acc']:.4f} | "
        f"{r['correct']:3d}/{r['total']:3d} | "
        f"{r['name']}"
    )

print("\nBest detail:")
if records:
    best = records[0]
    print(f"name = {best['name']}")
    print(f"path = {best['path']}")
    print(f"acc  = {best['acc']:.4f} ({best['correct']}/{best['total']})")
    for rel in RELS:
        c, t = best["by_gold"][rel]
        print(f"  {rel:6s}: {c}/{t} = {c/t if t else 0:.4f}")
PY

echo ""
echo "Done. Results saved to ${OUT_DIR}"
