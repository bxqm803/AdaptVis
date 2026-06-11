#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRIPT=${SCRIPT:-run_llava15_hf_customconfig_baseline.py}

COMMON=(
  --dataset Controlled_Images_A
  --method adapt_vis
  --weight1 0.5
  --weight2 1.5
  --threshold 0.4
  --max-layers 32
  --option four
  --device cuda
  --dtype float32
  --num-workers 0
  --download
)

run_case() {
  local name="$1"
  shift
  echo "================================================================================"
  echo "RUN: ${name}"
  echo "================================================================================"
  "$PYTHON" "$SCRIPT" "${COMMON[@]}" "$@" \
    --output "output/config_ablation_${name}.json"
}

# Two endpoints.
run_case checkpoint_full --text-config checkpoint
run_case custom_full     --text-config custom

# RMSNorm epsilon: sufficiency and necessity.
run_case checkpoint_plus_custom_rms \
  --text-config checkpoint --config-patch rms_norm_eps
run_case custom_minus_custom_rms \
  --text-config custom --config-patch rms_norm_eps

# Position/RoPE config group: sufficiency and necessity.
run_case checkpoint_plus_custom_position \
  --text-config checkpoint --config-patch position
run_case custom_minus_custom_position \
  --text-config custom --config-patch position

# BOS/EOS/PAD IDs: sufficiency and necessity.
run_case checkpoint_plus_custom_tokens \
  --text-config checkpoint --config-patch tokens
run_case custom_minus_custom_tokens \
  --text-config custom --config-patch tokens
