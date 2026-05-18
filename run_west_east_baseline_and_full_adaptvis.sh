#!/usr/bin/env bash
set -euo pipefail

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"
GPU="${GPU:-0}"

WEIGHT1="${WEIGHT1:-0.5}"
WEIGHT2="${WEIGHT2:-1.5}"
THRESHOLD="${THRESHOLD:-0.5}"

LEFT_SYN="west"
RIGHT_SYN="east"

PROMPT_FILE="prompts/${DATASET}_with_answer_${OPTION}_options.jsonl"
BACKUP_FILE="${PROMPT_FILE}.bak.west_east.$(date +%Y%m%d_%H%M%S)"

cp "${PROMPT_FILE}" "${BACKUP_FILE}"

restore_prompt () {
  if [[ -f "${BACKUP_FILE}" ]]; then
    cp "${BACKUP_FILE}" "${PROMPT_FILE}"
    echo "[RESTORE] prompt restored from ${BACKUP_FILE}"
  fi
}

trap restore_prompt EXIT

float_tag () {
  local X="$1"
  if [[ "${X}" == "0.5" ]]; then
    echo "05"
  elif [[ "${X}" == "1.5" ]]; then
    echo "15"
  elif [[ "${X}" == "0.4" ]]; then
    echo "04"
  elif [[ "${X}" == "0.6" ]]; then
    echo "06"
  elif [[ "${X}" == "0.7" ]]; then
    echo "07"
  elif [[ "${X}" == "1.0" ]]; then
    echo "10"
  else
    echo "$(echo "${X}" | sed 's/\.//g')"
  fi
}

W1TAG="$(float_tag "${WEIGHT1}")"
W2TAG="$(float_tag "${WEIGHT2}")"
TTAG="$(float_tag "${THRESHOLD}")"

BASE_TAG="baseline_west_east_original_allrels"
ADAPT_TAG="adaptvis_full_all_west_east_w${W1TAG}_${W2TAG}_t${TTAG}"

BASE_RESULT="./output/results1.5_${DATASET}_base_1.0_${OPTION}option_False_${BASE_TAG}.json"
ADAPT_RESULT="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${ADAPT_TAG}.json"

echo "[CHECK] compile python files"
python -m py_compile main_aro.py
python -m py_compile model_zoo/llava15.py
python -m py_compile model_zoo/llama/modeling_llama_add_attn.py

echo "[PATCH] replace left/right with west/east in prompt and answer"
python - <<PY
import json
import re
from pathlib import Path

path = Path("${PROMPT_FILE}")
left_syn = "${LEFT_SYN}"
right_syn = "${RIGHT_SYN}"

def map_answer(x):
    s = str(x).strip().lower()
    if s == "left":
        return left_syn
    if s == "right":
        return right_syn
    return x

rows = []

with path.open("r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)

        q = d.get("question", "")

        q2 = re.sub(
            r"Answer\s+with\s+left,\s*right,\s*on\s+or\s+under\.?",
            f"Answer with {left_syn}, {right_syn}, on or under.",
            q,
            flags=re.IGNORECASE,
        )

        q2 = q2.replace(
            "Answer with left, right, on or under.",
            f"Answer with {left_syn}, {right_syn}, on or under."
        )
        q2 = q2.replace(
            "Answer with left, right, on or under",
            f"Answer with {left_syn}, {right_syn}, on or under"
        )

        d["question"] = q2

        ans = d.get("answer", "")
        if isinstance(ans, list):
            d["answer"] = [map_answer(a) for a in ans]
        else:
            d["answer"] = map_answer(ans)

        rows.append(d)

with path.open("w", encoding="utf-8") as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("patched prompt:", path)
print("num rows:", len(rows))
print("example question:", rows[0]["question"])
print("example answer:", rows[0]["answer"])
PY

echo ""
echo "============================================================"
echo "RUN 1: baseline, no AdaptVis"
echo "TAG=${BASE_TAG}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" \
IMAGE_CONTROL=none \
CLIP_OBJ_MASK=False \
ADJUST_METHOD=last_query \
PROBE_SAMPLE_IDS_FILE="" \
PROBE_RELATION_PROBS=False \
PROBE_RUN_TAG="${BASE_TAG}" \
ATTN_RUN_TAG="${BASE_TAG}" \
PROBE_SINGLE_PASS=False \
SAVE_LAYERS=-1 \
python3 main_aro.py \
  --dataset="${DATASET}" \
  --model-name="${MODEL_NAME}" \
  --download \
  --method=base \
  --weight=1.0 \
  --option="${OPTION}"

echo ""
echo "============================================================"
echo "RUN 2: full global all-layer AdaptVis"
echo "TAG=${ADAPT_TAG}"
echo "weight1=${WEIGHT1}, weight2=${WEIGHT2}, threshold=${THRESHOLD}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" \
ADAPTVIS_EXCLUDE_LAYERS="" \
ADAPTVIS_INCLUDE_LAYERS="" \
ADAPTVIS_LAYER_DEBUG=False \
IMAGE_CONTROL=none \
IMAGE_CONTROL_SEED=1 \
IMAGE_CONTROL_SIZE=336 \
IMAGE_CONTROL_GRID=24 \
CLIP_OBJ_MASK=False \
ADJUST_METHOD=last_query \
PATCH_MASK_MODE="" \
PATCH_GRID_SIZE=24 \
PATCH_BLOCK_GRID=4 \
PATCH_BLOCK_IDS="" \
PATCH_MASK_DEBUG=False \
PROBE_SAMPLE_IDS_FILE="" \
PROBE_RELATION_PROBS=False \
PROBE_RUN_TAG="${ADAPT_TAG}" \
ATTN_RUN_TAG="${ADAPT_TAG}" \
PROBE_SINGLE_PASS=False \
SAVE_LAYERS=-1 \
python3 main_aro.py \
  --dataset="${DATASET}" \
  --model-name="${MODEL_NAME}" \
  --download \
  --method=adapt_vis \
  --weight1="${WEIGHT1}" \
  --weight2="${WEIGHT2}" \
  --threshold="${THRESHOLD}" \
  --option="${OPTION}"

echo ""
echo "============================================================"
echo "STATS"
echo "============================================================"

python - <<PY
import json
import re
from collections import Counter, defaultdict

paths = [
    ("baseline_west_east", "${BASE_RESULT}"),
    ("adaptvis_full_all_west_east", "${ADAPT_RESULT}"),
]

def norm_rel(x):
    s = str(x).strip().lower()

    if "under" in s:
        return "under"
    if re.search(r"\\bon\\b", s) and "front" not in s:
        return "on"

    if "west" in s:
        return "left"
    if "east" in s:
        return "right"

    if "left" in s:
        return "left"
    if "right" in s:
        return "right"

    return "unknown"

for name, path in paths:
    print("\\n" + "=" * 90)
    print(name)
    print("file:", path)

    rows = json.load(open(path, "r", encoding="utf-8"))

    strict_correct = sum(bool(r.get("Correct", False)) for r in rows)
    mapped_correct = sum(
        norm_rel(r.get("Golden", "")) == norm_rel(r.get("Generation", ""))
        for r in rows
    )

    print(f"strict repo acc:  {strict_correct}/{len(rows)} = {strict_correct / len(rows):.4f}")
    print(f"mapped semantic acc: {mapped_correct}/{len(rows)} = {mapped_correct / len(rows):.4f}")
    print("selected_weight:", Counter(str(r.get("selected_weight")) for r in rows))

    print("\\nPer-gold strict accuracy:")
    for g in ["left", "right", "on", "under"]:
        sub = [r for r in rows if norm_rel(r.get("Golden", "")) == g]
        c = sum(bool(r.get("Correct", False)) for r in sub)
        pred = Counter(norm_rel(r.get("Generation", "")) for r in sub)
        print(f"{g:6s}: {c:3d}/{len(sub):3d} = {c/len(sub):.4f} | mapped_pred={dict(pred)}")

    print("\\nMapped confusion matrix:")
    conf = defaultdict(Counter)
    for r in rows:
        conf[norm_rel(r.get("Golden", ""))][norm_rel(r.get("Generation", ""))] += 1

    print("gold\\\\pred     left  right     on  under unknown")
    for g in ["left", "right", "on", "under", "unknown"]:
        row = conf[g]
        print(
            f"{g:10s}"
            f"{row.get('left', 0):7d}"
            f"{row.get('right', 0):7d}"
            f"{row.get('on', 0):7d}"
            f"{row.get('under', 0):7d}"
            f"{row.get('unknown', 0):8d}"
        )
PY

echo ""
echo "[DONE]"
echo "baseline result: ${BASE_RESULT}"
echo "adaptvis result: ${ADAPT_RESULT}"
