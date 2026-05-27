#!/usr/bin/env bash
set -e

VARIANT="negonly_mean_img"
LAYERS_END=4
GRID=4
STRENGTH=1.0

run_combo () {
  MODE="$1"
  BLOCKS="$2"
  NAME="$3"

  echo "[RUN] mode=${MODE} blocks=${BLOCKS} name=${NAME}"

  ADAPTVIS_ATTENTION_VARIANT="$VARIANT" \
  ADAPTVIS_POSNEG_STRENGTH="$STRENGTH" \
  ADAPTVIS_PATCH_GRID="$GRID" \
  ADAPTVIS_PATCH_BLOCK_MODE="$MODE" \
  ADAPTVIS_PATCH_BLOCKS="$BLOCKS" \
  python3 run_layer_ablation_once.py \
    --start-end-layer "$LAYERS_END" \
    --stop-end-layer "$LAYERS_END" \
    --out-csv "output/${NAME}.csv"
}

run_combo except "13" "combo_except_13_layers_0_to_4"
run_combo except "13,14,15" "combo_except_13_14_15_layers_0_to_4"
run_combo except "12,13,14,15" "combo_except_12_13_14_15_layers_0_to_4"
run_combo except "9,10,12,13,14,15" "combo_except_9_10_12_13_14_15_layers_0_to_4"

run_combo only "0,1,2,3,6,7" "combo_only_0_1_2_3_6_7_layers_0_to_4"
run_combo only "0,1,2,3,7" "combo_only_0_1_2_3_7_layers_0_to_4"

echo "[DONE]"
