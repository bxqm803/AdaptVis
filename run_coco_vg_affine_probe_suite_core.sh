#!/usr/bin/env bash
# Sequential extraction + affine-axis analysis for COCO_two or VG_two.
# Runs Qwen, LLaVA, and InternVL only. Llama and Gemma are intentionally excluded.
# A failed model is logged and skipped; remaining models continue.
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
  cat <<EOF_USAGE
Usage:
  bash run_coco_vg_affine_probe_suite_core.sh --dataset {coco_two|vg_two} [options]

Models included by --models all:
  qwen2-2b,qwen-3b,qwen-7b,
  llava-7b,llava-13b,
  internvl-1b,internvl-2b,internvl-8b,internvl-14b

Intentionally excluded:
  llama-11b, gemma-4b, gemma-12b

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
EOF_USAGE
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

# Excluded intentionally:
# - llama-11b: gated Hugging Face repository in the current environment.
# - gemma-4b / gemma-12b: unsupported by the current transformers installation.
ALL_MODELS=(
  qwen2-2b qwen-3b qwen-7b
  llava-7b llava-13b
  internvl-1b internvl-2b internvl-8b internvl-14b
)

if [[ "$MODELS" == "all" ]]; then
  RUN_MODELS=("${ALL_MODELS[@]}")
else
  IFS=',' read -r -a RUN_MODELS <<< "$MODELS"
fi

SUITE_DIR="$OUT_ROOT/$DATASET"
mkdir -p "$SUITE_DIR"
FAILED_LOG="$SUITE_DIR/failed_models_${IMAGE_MODE}.log"
STATUS_LOG="$SUITE_DIR/suite_status_${IMAGE_MODE}.tsv"
printf "# dataset=%s\timage_mode=%s\n" "$DATASET" "$IMAGE_MODE" > "$FAILED_LOG"
printf "model\tstage\tstatus\tdetail\n" > "$STATUS_LOG"

ok_count=0
failed_count=0
skipped_count=0

for MODEL in "${RUN_MODELS[@]}"; do
  MODEL="${MODEL//[[:space:]]/}"
  [[ -z "$MODEL" ]] && continue

  RUN_DIR="$OUT_ROOT/$DATASET/$MODEL/$IMAGE_MODE"
  STATES="$RUN_DIR/states.npz"
  SUMMARY="$RUN_DIR/affine_axes_cv.json"
  EXTRACT_LOG="$RUN_DIR/extract.log"
  ANALYZE_LOG="$RUN_DIR/analyze.log"
  mkdir -p "$RUN_DIR"

  case "$MODEL" in
    llama-11b|gemma-4b|gemma-12b)
      echo "[SKIP] $MODEL is intentionally excluded from this suite."
      printf "%s\tprecheck\tskipped\tintentionally excluded\n" "$MODEL" >> "$STATUS_LOG"
      ((skipped_count+=1))
      continue
      ;;
  esac

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

  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "${EXTRACT_CMD[@]}" 2>&1 | tee "$EXTRACT_LOG"
  extract_status=${PIPESTATUS[0]}
  set -e

  if [[ "$extract_status" -ne 0 ]]; then
    echo "[FAILED] extraction: $MODEL (exit=$extract_status); continuing."
    printf "%s\textract\tfailed\texit=%s; log=%s\n" "$MODEL" "$extract_status" "$EXTRACT_LOG" >> "$STATUS_LOG"
    printf "%s\textract\texit=%s\t%s\n" "$MODEL" "$extract_status" "$EXTRACT_LOG" >> "$FAILED_LOG"
    ((failed_count+=1))
    continue
  fi

  set +e
  python3 analyze_two_object_affine_axes.py \
    --input-npz "$STATES" \
    --cv-folds "$CV_FOLDS" \
    --label-shuffle-repeats "$SHUFFLE_REPEATS" \
    --output "$SUMMARY" 2>&1 | tee "$ANALYZE_LOG"
  analyze_status=${PIPESTATUS[0]}
  set -e

  if [[ "$analyze_status" -ne 0 ]]; then
    echo "[FAILED] analysis: $MODEL (exit=$analyze_status); continuing."
    printf "%s\tanalyze\tfailed\texit=%s; log=%s\n" "$MODEL" "$analyze_status" "$ANALYZE_LOG" >> "$STATUS_LOG"
    printf "%s\tanalyze\texit=%s\t%s\n" "$MODEL" "$analyze_status" "$ANALYZE_LOG" >> "$FAILED_LOG"
    ((failed_count+=1))
    continue
  fi

  printf "%s\tboth\tok\t%s\n" "$MODEL" "$SUMMARY" >> "$STATUS_LOG"
  ((ok_count+=1))
done

echo
echo "================================================================"
echo "Suite complete: ok=$ok_count failed=$failed_count skipped=$skipped_count"
echo "Status: $STATUS_LOG"
echo "Failures: $FAILED_LOG"
echo "================================================================"
