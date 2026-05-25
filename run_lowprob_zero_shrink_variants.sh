#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Controlled_Images_A}"
MODEL="${MODEL:-llava1.5}"
OPTION="${OPTION:-four}"
GPU="${GPU:-0}"

BASE_JSON="${BASE_JSON:-output/results1.5_Controlled_Images_A_adapt_vis_w1_w11_w21_thr0p4_fouroption_True.json}"
THRESHOLD="${THRESHOLD:-0.4}"

LOWPROB_IDS_ALL="${LOWPROB_IDS_ALL:-lowprob_lt0p4_ids_all.json}"
LOWPROB_IDS="${LOWPROB_IDS:-lowprob_lt0p4_ids_3samples.json}"
SMOKE_N="${SMOKE_N:-3}"

RESULT_DIR="${RESULT_DIR:-attention_variant_lowprob_zero_shrink_3samples}"
mkdir -p "${RESULT_DIR}"

echo "[1] make first ${SMOKE_N} low-probability ids from base json"
python3 - <<PY
import json

base_json = "${BASE_JSON}"
threshold = float("${THRESHOLD}")
out_all = "${LOWPROB_IDS_ALL}"
out = "${LOWPROB_IDS}"
n = int("${SMOKE_N}")

with open(base_json, "r", encoding="utf-8") as f:
    data = json.load(f)

prob_keys = [
    "base_probability",
    "probability",
    "base_confidence",
    "confidence",
    "Confidence",
    "uncertainty",
    "Uncertainty",
]

ids_all = []
used_key = None
missing = 0

for i, item in enumerate(data):
    p = None
    k_used = None

    for k in prob_keys:
        if k in item and item[k] not in [None, ""]:
            try:
                p = float(item[k])
                k_used = k
                break
            except Exception:
                pass

    if p is None:
        missing += 1
        continue

    used_key = used_key or k_used

    if p < threshold:
        ids_all.append(i)

if used_key is None:
    print("[ERROR] Cannot find probability/confidence key. First item keys:")
    print(list(data[0].keys()))
    raise SystemExit(1)

ids = ids_all[:n]

print("[BASE_JSON]", base_json)
print("[PROB_KEY]", used_key)
print("[THRESHOLD]", threshold)
print("[TOTAL]", len(data))
print("[LOWPROB_TOTAL]", len(ids_all))
print("[SMOKE_SELECTED]", len(ids))
print("[IDS]", ids)
print("[MISSING_PROB]", missing)

with open(out_all, "w", encoding="utf-8") as f:
    json.dump(ids_all, f, indent=2)

with open(out, "w", encoding="utf-8") as f:
    json.dump(ids, f, indent=2)

print("[SAVED ALL]", out_all)
print("[SAVED SMOKE]", out)
PY

echo
echo "[2] compile check"
python3 -m py_compile main_aro.py
python3 -m py_compile model_zoo/llama/modeling_llama_add_attn.py

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export SAVE_ATTN=False
export TEST_MODE=False

# 关键：只跑前 3 个 probability < threshold 的样本
export RUN_SAMPLE_IDS_FILE="${LOWPROB_IDS}"

run_one () {
  local variant="$1"
  local weight1="$2"
  local out_name="$3"

  echo
  echo "================================================"
  echo "[RUN 3 LOWPROB] variant=${variant}, weight1=${weight1}, out=${out_name}"
  echo "================================================"

  export ADAPTVIS_ATTENTION_VARIANT="${variant}"

  CUDA_VISIBLE_DEVICES="${GPU}" python3 main_aro.py \
    --dataset "${DATASET}" \
    --model-name "${MODEL}" \
    --download \
    --method adapt_vis \
    --weight1 "${weight1}" \
    --weight2 1.5 \
    --threshold 0.4 \
    --option "${OPTION}" \
    --batch-size 1 \
    --num_workers 0

  latest=$(find output -maxdepth 1 -type f -name "*.json" ! -name "*_scores.json" -printf "%T@ %p\n" \
    | sort -nr | head -n 1 | cut -d' ' -f2-)

  if [[ -z "${latest}" ]]; then
    echo "[ERROR] cannot find latest output json"
    exit 1
  fi

  cp -f "${latest}" "${RESULT_DIR}/${out_name}.json"
  echo "[COPIED] ${latest} -> ${RESULT_DIR}/${out_name}.json"

  python3 - <<PY
import json
path = "${RESULT_DIR}/${out_name}.json"
ids_path = "${LOWPROB_IDS}"

data = json.load(open(path, "r", encoding="utf-8"))
ids = json.load(open(ids_path, "r", encoding="utf-8"))

print("[CHECK]", path, "num_records=", len(data), "expected=", len(ids))
if len(data) != len(ids):
    print("[WARNING] result length != lowprob ids length")

for i, r in enumerate(data):
    sid = ids[i] if i < len(ids) else None
    gen = r.get("RawGeneration", r.get("Generation", ""))
    gold = r.get("Golden", r.get("gold", ""))
    corr = r.get("RawGenerationCorrect", r.get("Correct", r.get("correct", None)))
    print(f"  local={i}, sid={sid}: correct={corr}, gold={gold}, gen={str(gen)[:100]}")
PY
}

# baseline：原始往 0 线性收缩
run_one "mul_img" "0.5" "mul_img_0p5"

# hard shrink：直接把 visual logits 截断到 [-a, a]
run_one "clip_img" "2.0" "clip_img_a2p0"
run_one "clip_img" "1.0" "clip_img_a1p0"

# smooth shrink：a * tanh(s/a)
run_one "tanh_img" "2.0" "tanh_img_a2p0"
run_one "tanh_img" "1.0" "tanh_img_a1p0"

# softsign shrink：s / (1 + lambda * |s|)
run_one "softsign_img" "0.5" "softsign_img_lam0p5"
run_one "softsign_img" "1.0" "softsign_img_lam1p0"

echo
echo "[DONE]"
echo "Smoke lowprob ids: ${LOWPROB_IDS}"
echo "Results copied to: ${RESULT_DIR}/"
ls -lh "${RESULT_DIR}"
