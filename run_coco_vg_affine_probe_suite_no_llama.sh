#!/usr/bin/env bash
# Sequential full extraction + affine-axis analysis for COCO_two or VG_two.
# Run from the AdaptVis repository root.
set -euo pipefail

DATASET=""
DATA_ROOT="data"
OUT_ROOT="output/two_object_affine_axes"
MODELS="all"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
ATTN_IMPL="sdpa"
LAYER_FRACS="0.20,0.30,0.40,0.50,0.60,0.70,0.80,1.00"
CV_FOLDS=5
SHUFFLE_REPEATS=20
MAX_SAMPLES=""
OVERWRITE=0
IMAGE_MODE="true"

usage() {
  cat <<EOF
Usage:
  bash run_coco_vg_affine_probe_suite.sh --dataset {coco_two|vg_two} [options]

Options:
  --data-root PATH          Dataset root (default: data)
  --out-root PATH           Output root (default: output/two_object_affine_axes)
  --models CSV|all          Model aliases (default: all)
  --gpu ID                  CUDA visible GPU id (default: CUDA_VISIBLE_DEVICES or 0)
  --attn-impl NAME          sdpa|eager|flash_attention_2|none (default: sdpa)
  --layer-fracs CSV         Relative decoder depths
  --cv-folds N              CV folds (default: 5)
  --shuffle-repeats N       Label-shuffle control repeats (default: 20)
  --max-samples N           Optional extraction cap for debugging
  --image-mode MODE         true|shuffle|blank (default: true)
  --overwrite               Ignore/resave existing .npz output
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --attn-impl) ATTN_IMPL="$2"; shift 2 ;;
    --layer-fracs) LAYER_FRACS="$2"; shift 2 ;;
    --cv-folds) CV_FOLDS="$2"; shift 2 ;;
    --shuffle-repeats) SHUFFLE_REPEATS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --image-mode) IMAGE_MODE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$DATASET" != "coco_two" && "$DATASET" != "vg_two" ]]; then
  echo "--dataset must be coco_two or vg_two" >&2
  exit 2
fi

# Llama-3.2-11B-Vision is intentionally excluded: its Hugging Face repo is gated
# and this benchmark suite should run without requiring a separate access grant.
ALL_MODELS=(
  qwen2-2b qwen-3b qwen-7b
  llava-7b llava-13b
  internvl-1b internvl-2b internvl-8b internvl-14b
  gemma-4b gemma-12b
)

if [[ "$MODELS" == "all" ]]; then
  RUN_MODELS=("${ALL_MODELS[@]}")
else
  IFS=',' read -r -a RUN_MODELS <<< "$MODELS"
fi

for MODEL in "${RUN_MODELS[@]}"; do
  MODEL="${MODEL//[[:space:]]/}"
  [[ -z "$MODEL" ]] && continue

  RUN_DIR="$OUT_ROOT/$DATASET/$MODEL/$IMAGE_MODE"
  STATES="$RUN_DIR/states.npz"
  SUMMARY="$RUN_DIR/affine_axes_cv.json"
  mkdir -p "$RUN_DIR"

  echo
  echo "================================================================"
  echo "Dataset=$DATASET | Model=$MODEL | image_mode=$IMAGE_MODE"
  echo "================================================================"

  EXTRACT_CMD=(
    python3 extract_two_object_relation_states.py
    --dataset "$DATASET"
    --data-root "$DATA_ROOT"
    --model "$MODEL"
    --device cuda:0
    --attn-impl "$ATTN_IMPL"
    --layer-fracs "$LAYER_FRACS"
    --image-mode "$IMAGE_MODE"
    --output "$STATES"
  )
  if [[ -n "$MAX_SAMPLES" ]]; then
    EXTRACT_CMD+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ "$OVERWRITE" -eq 1 ]]; then
    EXTRACT_CMD+=(--overwrite)
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "${EXTRACT_CMD[@]}"

  python3 analyze_two_object_affine_axes.py \
    --input-npz "$STATES" \
    --cv-folds "$CV_FOLDS" \
    --label-shuffle-repeats "$SHUFFLE_REPEATS" \
    --output "$SUMMARY"

done
