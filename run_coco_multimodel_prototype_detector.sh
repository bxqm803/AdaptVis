#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU pipeline for the prototype_head baseline-error detector.
#
# Default model registry:
#   qwen2-2b, qwen-7b, llava-7b, llava-13b,
#   internvl-1b, internvl-2b, internvl-8b
#
# The repository registry contains InternVL3-8B, not an internvl-7b alias.

DATA_ROOT="${DATA_ROOT:-data}"
PROMPT_JSONL="${PROMPT_JSONL:-prompts/COCO_QA_two_obj_with_answer_four_options.jsonl}"

SOURCE_ROOT="${SOURCE_ROOT:-output/spatial_storage_transport_utilization/coco}"
SWAP_ROOT="${SWAP_ROOT:-output/coco_head_swap_error_detector}"
DETECTOR_ROOT="${DETECTOR_ROOT:-output/coco_all_relation_head_prototype_detector}"
SUMMARY_DIR="${SUMMARY_DIR:-${DETECTOR_ROOT}/multimodel_summary}"
LOG_ROOT="${LOG_ROOT:-output/logs/coco_multimodel_prototype_detector}"

TRACE_LAYER_CHUNK="${TRACE_LAYER_CHUNK:-4}"
SOURCE_MAX_NEW_TOKENS="${SOURCE_MAX_NEW_TOKENS:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

OUTER_REPEATS="${OUTER_REPEATS:-5}"
TOP_K_FEATURES="${TOP_K_FEATURES:-512}"
LOGREG_C="${LOGREG_C:-0.10}"
L1_RATIO="${L1_RATIO:-0.50}"
MAX_ITER="${MAX_ITER:-5000}"

FORCE_SOURCE="${FORCE_SOURCE:-0}"
FORCE_SWAP="${FORCE_SWAP:-0}"
FORCE_DETECTOR="${FORCE_DETECTOR:-0}"

GPU0_MODELS_DEFAULT="qwen2-2b,qwen-7b,internvl-1b,internvl-8b"
GPU1_MODELS_DEFAULT="llava-7b,llava-13b,internvl-2b"
GPU0_MODELS="${GPU0_MODELS:-$GPU0_MODELS_DEFAULT}"
GPU1_MODELS="${GPU1_MODELS:-$GPU1_MODELS_DEFAULT}"

mkdir -p "$SOURCE_ROOT" "$SWAP_ROOT" "$DETECTOR_ROOT" "$SUMMARY_DIR" "$LOG_ROOT"

required=(
  analyze_spatial_storage_transport_utilization_v3.py
  analyze_coco_head_swap_error_detector_v1_20260801.py
  analyze_coco_all_relation_head_prototype_detector_v1.py
  prepare_coco_baseline_from_v3.py
  summarize_coco_multimodel_prototype_detector.py
)
for file in "${required[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[FATAL] Missing repository file: $file" >&2
    exit 1
  fi
done

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "[RUN] $*" > "$log_file"
  if ! "$@" >> "$log_file" 2>&1; then
    echo "[FAILED] See $log_file" >&2
    tail -n 120 "$log_file" >&2 || true
    return 1
  fi
}

choose_cv() {
  local cells_jsonl="$1"
  python - "$cells_jsonl" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
rows = []
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            rows.append(json.loads(line))

if not rows:
    raise SystemExit("No parsed baseline rows")

error = [0 if bool(row["baseline_correct"]) else 1 for row in rows]
error_counts = Counter(error)
max_folds = min(error_counts.get(0, 0), error_counts.get(1, 0), 5)
if max_folds < 2:
    print("not_evaluable 0")
    raise SystemExit(0)

relation_error = Counter(
    f"{row['gt']}__{0 if bool(row['baseline_correct']) else 1}"
    for row in rows
)
min_relation_error = min(relation_error.values()) if relation_error else 0

if min_relation_error >= max_folds:
    print(f"relation_error {max_folds}")
else:
    print(f"error {max_folds}")
PY
}

run_model() {
  local physical_gpu="$1"
  local model="$2"

  local source_dir="${SOURCE_ROOT}/${model}"
  local baseline_jsonl="${source_dir}/baseline_generation.jsonl"
  local swap_dir="${SWAP_ROOT}/${model}"
  local detector_dir="${DETECTOR_ROOT}/${model}"
  local model_log_dir="${LOG_ROOT}/${model}"
  mkdir -p "$model_log_dir"

  echo "[$(date '+%F %T')] GPU=${physical_gpu} MODEL=${model}: start"

  # 1. Reuse or create the common v3 source extraction.
  if [[ "$FORCE_SOURCE" == "1" || ! -s "${source_dir}/extraction.jsonl" ]]; then
    rm -rf "$source_dir"
    local source_max_args=()
    if [[ "$MAX_SAMPLES" -gt 0 ]]; then
      source_max_args=(--max-samples "$MAX_SAMPLES")
    fi
    run_logged "${model_log_dir}/01_source_v3.log" \
      env CUDA_VISIBLE_DEVICES="$physical_gpu" \
      python -u analyze_spatial_storage_transport_utilization_v3.py \
        --dataset coco_two \
        --data-root "$DATA_ROOT" \
        --prompt-jsonl "$PROMPT_JSONL" \
        --model "$model" \
        --device cuda:0 \
        --attn-impl eager \
        --layers all \
        --trace-layer-chunk "$TRACE_LAYER_CHUNK" \
        --max-new-tokens "$SOURCE_MAX_NEW_TOKENS" \
        "${source_max_args[@]}" \
        --skip-head-ablation \
        --skip-factorial \
        --output-dir "$source_dir" \
        --overwrite
  else
    echo "[$model] Reusing source extraction: $source_dir"
  fi

  # 2. Convert v3 free-generation metadata into the baseline schema required
  #    by the head-swap extractor. Unparsed generations are excluded explicitly.
  run_logged "${model_log_dir}/02_prepare_baseline.log" \
    python -u prepare_coco_baseline_from_v3.py \
      --input "${source_dir}/extraction.jsonl" \
      --output "$baseline_jsonl"

  # 3. Capture all decoder heads at identity-aligned object tokens under
  #    original and subject/reference-swapped questions.
  local swap_overwrite=()
  if [[ "$FORCE_SWAP" == "1" ]]; then
    swap_overwrite=(--overwrite)
  elif [[ ! -s "${swap_dir}/swap_cells.jsonl" ]]; then
    swap_overwrite=(--overwrite)
  fi

  run_logged "${model_log_dir}/03_head_swap_extract.log" \
    env CUDA_VISIBLE_DEVICES="$physical_gpu" \
    python -u analyze_coco_head_swap_error_detector_v1_20260801.py \
      --phase extract \
      --model "$model" \
      --source-output-dir "$source_dir" \
      --baseline-generation-jsonl "$baseline_jsonl" \
      --dataset coco_two \
      --data-root "$DATA_ROOT" \
      --prompt-jsonl "$PROMPT_JSONL" \
      --device cuda:0 \
      --attn-impl eager \
      --object-state last \
      --capture-pool mean \
      --scan-layers all \
      --save-dtype float16 \
      --sample-max-samples "$MAX_SAMPLES" \
      --output-dir "$swap_dir" \
      --resume \
      "${swap_overwrite[@]}"

  # 4. Pick the strongest feasible repeated-CV stratification.
  local cv
  cv="$(choose_cv "${swap_dir}/swap_cells.jsonl")"
  local stratify folds
  read -r stratify folds <<< "$cv"
  if [[ "$stratify" == "not_evaluable" || "$folds" -lt 2 ]]; then
    echo "[$model] Detector not evaluable: fewer than two correct or wrong samples." \
      | tee "${model_log_dir}/04_detector_not_evaluable.log"
    return 0
  fi
  echo "[$model] detector CV: stratify=$stratify folds=$folds repeats=$OUTER_REPEATS"

  if [[ "$FORCE_DETECTOR" == "1" ]]; then
    rm -rf "$detector_dir"
  fi

  run_logged "${model_log_dir}/04_prototype_detector.log" \
    python -u analyze_coco_all_relation_head_prototype_detector_v1.py \
      --input-dir "$swap_dir" \
      --output-dir "$detector_dir" \
      --outer-folds "$folds" \
      --outer-repeats "$OUTER_REPEATS" \
      --stratify "$stratify" \
      --top-k-features "$TOP_K_FEATURES" \
      --logreg-c "$LOGREG_C" \
      --l1-ratio "$L1_RATIO" \
      --max-iter "$MAX_ITER" \
      --model-groups confidence,prototype_global,prototype_head,confidence_plus_prototype_global,confidence_plus_prototype_head \
      --model-prediction-source baseline \
      --prototype-training correct_only \
      --overwrite

  echo "[$(date '+%F %T')] GPU=${physical_gpu} MODEL=${model}: done"
}

run_queue() {
  local gpu="$1"
  local csv="$2"
  IFS=',' read -r -a models <<< "$csv"
  local model
  for model in "${models[@]}"; do
    model="${model//[[:space:]]/}"
    [[ -z "$model" ]] && continue
    run_model "$gpu" "$model"
  done
}

echo "GPU0 queue: $GPU0_MODELS"
echo "GPU1 queue: $GPU1_MODELS"

run_queue 0 "$GPU0_MODELS" > "${LOG_ROOT}/gpu0_queue.log" 2>&1 &
pid0=$!
run_queue 1 "$GPU1_MODELS" > "${LOG_ROOT}/gpu1_queue.log" 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

if [[ "$status" != "0" ]]; then
  echo "[FAILED] At least one GPU queue failed." >&2
  echo "GPU0 tail:" >&2
  tail -n 100 "${LOG_ROOT}/gpu0_queue.log" >&2 || true
  echo "GPU1 tail:" >&2
  tail -n 100 "${LOG_ROOT}/gpu1_queue.log" >&2 || true
  exit 1
fi

python -u summarize_coco_multimodel_prototype_detector.py \
  --models "qwen2-2b,qwen-7b,llava-7b,llava-13b,internvl-1b,internvl-2b,internvl-8b" \
  --detector-root "$DETECTOR_ROOT" \
  --threshold 0.5 \
  --output-dir "$SUMMARY_DIR"

echo "Done. Summary: ${SUMMARY_DIR}/report.txt"
