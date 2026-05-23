#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Controlled_Images_A}"
OPTION="${OPTION:-four}"
GPU="${GPU:-0}"
THRESHOLD="${THRESHOLD:-0.4}"
MODEL_NAME="${MODEL_NAME:-llava1.5}"
METHOD="${METHOD:-adapt_vis}"
DOWNLOAD_FLAG="${DOWNLOAD_FLAG:---download}"
TEST_MODE="${TEST_MODE:-True}"

export TEST_MODE
mkdir -p output

float_tag() {
  python3 - "$1" <<'PY'
import sys
x = float(sys.argv[1])
s = f"{x:g}"
print(s.replace('-', 'm').replace('.', 'p'))
PY
}

thr_tag="$(float_tag "$THRESHOLD")"

run_alpha() {
  local alpha="$1"
  local name="$2"
  local tag
  tag="$(float_tag "$alpha")"
  echo ""
  echo "===== Running ${name}: w1=w2=${alpha} ====="
  CUDA_VISIBLE_DEVICES="$GPU" python3 main_aro.py \
    --dataset "$DATASET" \
    --model-name "$MODEL_NAME" \
    $DOWNLOAD_FLAG \
    --method "$METHOD" \
    --weight1 "$alpha" \
    --weight2 "$alpha" \
    --threshold "$THRESHOLD" \
    --option "$OPTION" \
    2>&1 | tee "output/log_${DATASET}_${METHOD}_${name}_w1w2_${tag}.txt"

  local pattern="output/results1.5_${DATASET}_${METHOD}_w*_w1${tag}_w2${tag}_thr${thr_tag}_${OPTION}option_${TEST_MODE}.json"
  local file
  file="$(ls $pattern 2>/dev/null | tail -n 1 || true)"
  if [[ -z "$file" ]]; then
    echo "Cannot find result json with pattern: $pattern" >&2
    exit 1
  fi
  echo "$file" > "output/${DATASET}_${METHOD}_${name}_result_path.txt"
  echo "Saved ${name} path: $file"
}

run_alpha "1.0" "base"
run_alpha "1.5" "high"
run_alpha "0.5" "low"

BASE_JSON="$(cat "output/${DATASET}_${METHOD}_base_result_path.txt")"
HIGH_JSON="$(cat "output/${DATASET}_${METHOD}_high_result_path.txt")"
LOW_JSON="$(cat "output/${DATASET}_${METHOD}_low_result_path.txt")"

python3 scripts/compare_three_generation_runs.py \
  --base "$BASE_JSON" \
  --low "$LOW_JSON" \
  --high "$HIGH_JSON" \
  --out "output/${DATASET}_${METHOD}_alpha_generation_compare_summary.json" \
  --csv "output/${DATASET}_${METHOD}_alpha_generation_compare_per_sample.csv"
