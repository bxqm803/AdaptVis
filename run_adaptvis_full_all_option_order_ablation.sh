#!/usr/bin/env bash
set -euo pipefail

DATASET="Controlled_Images_A"
MODEL_NAME="llava1.5"
OPTION="four"
GPU="${GPU:-0}"

# 标准 AdaptVis 双 weight。
WEIGHT1="${WEIGHT1:-0.5}"
WEIGHT2="${WEIGHT2:-1.5}"
THRESHOLD="${THRESHOLD:-0.5}"

PROMPT_FILE="prompts/${DATASET}_with_answer_${OPTION}_options.jsonl"

# 当前文件备份，防止出错。
RAW_BACKUP_FILE="${PROMPT_FILE}.bak.raw.option_order_only.$(date +%Y%m%d_%H%M%S)"

# clean backup：统一恢复成 canonical instruction。
CLEAN_BACKUP_FILE="${PROMPT_FILE}.bak.clean.option_order_only.$(date +%Y%m%d_%H%M%S)"

cp "${PROMPT_FILE}" "${RAW_BACKUP_FILE}"

restore_prompt () {
  if [[ -f "${CLEAN_BACKUP_FILE}" ]]; then
    cp "${CLEAN_BACKUP_FILE}" "${PROMPT_FILE}"
    echo "[RESTORE] prompt restored to clean canonical version from ${CLEAN_BACKUP_FILE}"
  elif [[ -f "${RAW_BACKUP_FILE}" ]]; then
    cp "${RAW_BACKUP_FILE}" "${PROMPT_FILE}"
    echo "[RESTORE] prompt restored from raw backup ${RAW_BACKUP_FILE}"
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

echo "[CHECK] compile python files"
python -m py_compile main_aro.py
python -m py_compile model_zoo/llava15.py
python -m py_compile model_zoo/llama/modeling_llama_add_attn.py

grep -q "PROBE_SINGLE_PASS" model_zoo/llava15.py || {
  echo "[ERROR] PROBE_SINGLE_PASS not found in llava15.py"
  echo "llava15.py may still make PROBE_RUN_TAG trigger single-pass."
  exit 1
}

echo "[BUILD CLEAN PROMPT] normalize current prompt to canonical order with only"
python - <<PY
import json
import re
from pathlib import Path
from collections import Counter

src = Path("${RAW_BACKUP_FILE}")
dst = Path("${CLEAN_BACKUP_FILE}")

canonical = "left, right, on or under"

def replace_instruction(q, order_text):
    # Robustly replace anything after "Answer with" until ASSISTANT or line end.
    q2, n = re.subn(
        r"Answer\s+with\s+.*?(?=\s*ASSISTANT:|$)",
        f"Answer with {order_text} only.",
        q,
        flags=re.IGNORECASE | re.S,
    )
    if n != 1:
        raise ValueError(f"Failed to replace answer instruction:\\n{q}")
    return q2

def extract_instruction(q):
    m = re.search(
        r"Answer\s+with\s+(.+?)(?=\s*ASSISTANT:|$)",
        q,
        flags=re.IGNORECASE | re.S,
    )
    return m.group(1).strip() if m else "NO_MATCH"

rows = []
instr_before = Counter()
instr_after = Counter()

with src.open("r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        q = d.get("question", "")
        instr_before[extract_instruction(q)] += 1
        d["question"] = replace_instruction(q, canonical)
        instr_after[extract_instruction(d["question"])] += 1
        rows.append(d)

with dst.open("w", encoding="utf-8") as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\\n")

# Also put clean canonical version into active prompt before starting.
with Path("${PROMPT_FILE}").open("w", encoding="utf-8") as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\\n")

print("raw backup:", src)
print("clean backup:", dst)
print("before:", instr_before)
print("after:", instr_after)
print("first clean question:")
print(rows[0]["question"])
print("first answer:", rows[0]["answer"])
PY

run_one_order () {
  local ORDER_TAG="$1"
  local ORDER_TEXT="$2"

  local TAG="adaptvis_full_all_order_${ORDER_TAG}_only_w${W1TAG}_${W2TAG}_t${TTAG}"
  local RESULT_FILE="./output/results1.5_${DATASET}_adapt_vis_1.0_${OPTION}option_False_${TAG}.json"
  local SCORE_FILE="${RESULT_FILE%.json}scores.json"

  echo ""
  echo "============================================================"
  echo "ORDER_TAG=${ORDER_TAG}"
  echo "ORDER_TEXT=${ORDER_TEXT}"
  echo "TAG=${TAG}"
  echo "WEIGHT1=${WEIGHT1}, WEIGHT2=${WEIGHT2}, THRESHOLD=${THRESHOLD}"
  echo "============================================================"

  # 每次都从 clean canonical prompt 重新生成。
  cp "${CLEAN_BACKUP_FILE}" "${PROMPT_FILE}"

  python - <<PY
import json
import re
from pathlib import Path
from collections import Counter

path = Path("${PROMPT_FILE}")
order_text = "${ORDER_TEXT}"

def replace_instruction(q, order_text):
    q2, n = re.subn(
        r"Answer\s+with\s+.*?(?=\s*ASSISTANT:|$)",
        f"Answer with {order_text} only.",
        q,
        flags=re.IGNORECASE | re.S,
    )
    if n != 1:
        raise ValueError(f"Failed to replace answer instruction:\\n{q}")
    return q2

def extract_instruction(q):
    m = re.search(
        r"Answer\s+with\s+(.+?)(?=\s*ASSISTANT:|$)",
        q,
        flags=re.IGNORECASE | re.S,
    )
    return m.group(1).strip() if m else "NO_MATCH"

rows = []
instr = Counter()

with path.open("r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        q = d.get("question", "")
        d["question"] = replace_instruction(q, order_text)
        instr[extract_instruction(d["question"])] += 1

        # Do not change answer.
        # Golden remains left/right/on/under.
        rows.append(d)

with path.open("w", encoding="utf-8") as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\\n")

print("patched prompt:", path)
print("instruction distribution:", instr)
print("num rows:", len(rows))
print("example question:", rows[0]["question"])
print("example answer:", rows[0]["answer"])
PY

  # 删除旧结果，避免读到上次错误文件。
  rm -f "${RESULT_FILE}" "${SCORE_FILE}"

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
  PROBE_RUN_TAG="${TAG}" \
  ATTN_RUN_TAG="${TAG}" \
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

  echo "[DONE ORDER] result: ${RESULT_FILE}"
}

run_one_order "right_left_under_on" "right, left, under or on"
run_one_order "under_on_left_right" "under, on, left or right"
run_one_order "on_under_right_left" "on, under, right or left"
run_one_order "under_left_right_on" "under, left, right or on"

# Restore clean canonical prompt before stats.
cp "${CLEAN_BACKUP_FILE}" "${PROMPT_FILE}"

echo ""
echo "============================================================"
echo "VERIFY PROMPTS"
echo "============================================================"

python - <<PY
import json
import re
from collections import Counter

DATASET = "${DATASET}"
OPTION = "${OPTION}"

tags = [
    "adaptvis_full_all_order_right_left_under_on_only_w${W1TAG}_${W2TAG}_t${TTAG}",
    "adaptvis_full_all_order_under_on_left_right_only_w${W1TAG}_${W2TAG}_t${TTAG}",
    "adaptvis_full_all_order_on_under_right_left_only_w${W1TAG}_${W2TAG}_t${TTAG}",
    "adaptvis_full_all_order_under_left_right_on_only_w${W1TAG}_${W2TAG}_t${TTAG}",
]

def extract_instruction(prompt):
    m = re.search(
        r"Answer\s+with\s+(.+?)(?=\s*ASSISTANT:|$)",
        prompt,
        flags=re.IGNORECASE | re.S,
    )
    return m.group(1).strip() if m else "NO_MATCH"

for tag in tags:
    path = f"./output/results1.5_{DATASET}_adapt_vis_1.0_{OPTION}option_False_{tag}.json"
    rows = json.load(open(path, "r", encoding="utf-8"))
    print("\\n" + tag)
    print(Counter(extract_instruction(r["Prompt"]) for r in rows))
    print("first prompt:", rows[0]["Prompt"])
PY

echo ""
echo "============================================================"
echo "STATS"
echo "============================================================"

python - <<PY
import json
import re
from collections import Counter, defaultdict

DATASET = "${DATASET}"
OPTION = "${OPTION}"

tags = [
    "adaptvis_full_all_order_right_left_under_on_only_w${W1TAG}_${W2TAG}_t${TTAG}",
    "adaptvis_full_all_order_under_on_left_right_only_w${W1TAG}_${W2TAG}_t${TTAG}",
    "adaptvis_full_all_order_on_under_right_left_only_w${W1TAG}_${W2TAG}_t${TTAG}",
    "adaptvis_full_all_order_under_left_right_on_only_w${W1TAG}_${W2TAG}_t${TTAG}",
]

def norm_rel(x):
    s = str(x).strip().lower()

    if "under" in s:
        return "under"
    if re.search(r"\\bon\\b", s) and "front" not in s:
        return "on"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"

    return "unknown"

def summarize(tag):
    path = f"./output/results1.5_{DATASET}_adapt_vis_1.0_{OPTION}option_False_{tag}.json"
    rows = json.load(open(path, "r", encoding="utf-8"))

    correct = sum(bool(r.get("Correct", False)) for r in rows)
    mapped_correct = sum(
        norm_rel(r.get("Golden", "")) == norm_rel(r.get("Generation", ""))
        for r in rows
    )

    print("\\n" + "=" * 90)
    print(tag)
    print("file:", path)
    print(f"repo acc:   {correct}/{len(rows)} = {correct / len(rows):.4f}")
    print(f"mapped acc: {mapped_correct}/{len(rows)} = {mapped_correct / len(rows):.4f}")
    print("selected_weight:", Counter(str(r.get("selected_weight")) for r in rows))

    print("\\nPer-gold accuracy:")
    for g in ["left", "right", "on", "under"]:
        sub = [r for r in rows if norm_rel(r.get("Golden", "")) == g]
        c = sum(bool(r.get("Correct", False)) for r in sub)
        mc = sum(norm_rel(r.get("Golden", "")) == norm_rel(r.get("Generation", "")) for r in sub)
        pred = Counter(norm_rel(r.get("Generation", "")) for r in sub)
        print(
            f"{g:6s}: repo={c:3d}/{len(sub):3d}={c/len(sub):.4f} "
            f"mapped={mc:3d}/{len(sub):3d}={mc/len(sub):.4f} "
            f"pred={dict(pred)}"
        )

    print("\\nConfusion matrix:")
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
            f"{row.get('under', 0):8d}"
            f"{row.get('unknown', 0):8d}"
        )

for tag in tags:
    summarize(tag)
PY

echo ""
echo "[DONE ALL]"
echo "Prompt restored to clean canonical version:"
echo "Answer with left, right, on or under only."
