cat > run_closedset_adaptvis_first_try.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

DATASET="Controlled_Images_A"
OPTION="four"
MODEL="llava1.5"
DEVICE="cuda"

# closed-set mode
export DECISION_MODE="closed_set"
export CLOSED_SET_SCORING="True"
export CLOSED_SET_CONFIDENCE_MODE="prob"
export ADJUST_METHOD="last_query"

# after fixing llava15.py, this should expand <image> placeholder to 576 image tokens
export NUM_IMAGE_TOKENS="576"

# Avoid PROBE_RUN_TAG making main_aro treat this as a probe-only run.
# It is okay even if llava15.py already computes its own acc; this keeps behavior explicit.
export FORCE_DATASET_EVAL="True"

mkdir -p output/closedset_adaptvis_first_try

run_one () {
  METHOD="$1"
  WEIGHT="$2"
  W1="$3"
  W2="$4"
  TH="$5"
  TAG="$6"

  export PROBE_RUN_TAG="$TAG"

  echo
  echo "============================================================"
  echo "RUN: method=$METHOD weight=$WEIGHT weight1=$W1 weight2=$W2 threshold=$TH tag=$TAG"
  echo "============================================================"

  python main_aro.py \
    --device "$DEVICE" \
    --batch-size 1 \
    --num_workers 0 \
    --model-name "$MODEL" \
    --dataset "$DATASET" \
    --download \
    --method "$METHOD" \
    --weight "$WEIGHT" \
    --weight1 "$W1" \
    --weight2 "$W2" \
    --threshold "$TH" \
    --option "$OPTION" \
    --decision-mode closed_set \
    --closed-set-scoring \
    --output-dir output/closedset_adaptvis_first_try
}

# 1. closed-set baseline
run_one base 1.0 1.0 1.0 1.0 base_closedset

# 2. closed-set + ScalingVis
run_one scaling_vis 1.2 1.0 1.0 1.0 scaling_w1p2
run_one scaling_vis 1.5 1.0 1.0 1.0 scaling_w1p5
run_one scaling_vis 2.0 1.0 1.0 1.0 scaling_w2p0

# 3. closed-set + AdaptVis
run_one adapt_vis 1.0 0.5 1.2 0.3 adapt_w1_0p5_w2_1p2_th0p3
run_one adapt_vis 1.0 0.5 1.5 0.3 adapt_w1_0p5_w2_1p5_th0p3
run_one adapt_vis 1.0 0.8 1.5 0.3 adapt_w1_0p8_w2_1p5_th0p3

python - <<'PY'
from pathlib import Path
import json
import re
import pandas as pd

expected_tags = [
    "base_closedset",
    "scaling_w1p2",
    "scaling_w1p5",
    "scaling_w2p0",
    "adapt_w1_0p5_w2_1p2_th0p3",
    "adapt_w1_0p5_w2_1p5_th0p3",
    "adapt_w1_0p8_w2_1p5_th0p3",
]

rows = []

# llava15.py writes files like:
# output/results1.5_<dataset>_<method>_<weight>_<option>option_<TEST>_closedset_<tag>.json
paths = sorted(Path("output").glob("*closedset*.json"))

for p in paths:
    try:
        obj = json.loads(p.read_text())
    except Exception:
        continue

    tag = obj.get("probe_run_tag", "")

    if not tag:
        m = re.search(r"_closedset_(.*?)\.json$", p.name)
        tag = m.group(1) if m else p.stem

    if tag not in expected_tags:
        continue

    rows.append({
        "tag": tag,
        "acc": obj.get("acc"),
        "processed_count": obj.get("processed_count"),
        "skipped_count": obj.get("skipped_count"),
        "decision_mode": obj.get("decision_mode"),
        "closed_set_scoring": obj.get("closed_set_scoring"),
        "closed_set_confidence_mode": obj.get("closed_set_confidence_mode"),
        "sample_filter_file": obj.get("sample_filter_file"),
        "path": str(p),
    })

df = pd.DataFrame(rows)

if len(df) == 0:
    print("[WARN] no closed-set json found.")
    print("[DEBUG] existing closedset json files:")
    for p in paths[-20:]:
        print(" ", p)
else:
    # Preserve expected order as much as possible
    order = {tag: i for i, tag in enumerate(expected_tags)}
    df["order"] = df["tag"].map(order)
    df = df.sort_values(["acc", "order"], ascending=[False, True]).drop(columns=["order"])

    print("\n================ CLOSED-SET ADAPTVIS SUMMARY ================")
    print(df.to_string(index=False))

    out = Path("output/closedset_adaptvis_first_try/summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n[SAVED]", out)
PY
BASH

chmod +x run_closedset_adaptvis_first_try.sh
