#!/usr/bin/env bash
set -euo pipefail

# Launch the complete 2 datasets x 3 models x 4 eps x 2 weights grid.
# Two jobs at once: one per A100. Every job resumes safely from its JSONL.

ROOT="/ddnB/work/mwang32/llava16/AdaptVis"
PYTHON="${PYTHON:-python}"
RUNNER="${ROOT}/run_eps_scalvis_vsr_gqa.py"
OUT="${ROOT}/output/eps_scalvis_grid"
LOGS="${OUT}/logs"

cd "${ROOT}"
mkdir -p "${OUT}" "${LOGS}"

# The runner defaults are the paths created by the earlier download commands.
VSR_ANN="${VSR_ANN:-${ROOT}/data/benchmarks/vsr/repo/data/splits/zeroshot/test.jsonl}"
VSR_IMAGES="${VSR_IMAGES:-${ROOT}/data/benchmarks/vsr/images}"

# Resolve the official GQA extracted directory robustly.
GQA_QUESTIONS="${GQA_QUESTIONS:-$(find "${ROOT}/data/benchmarks/gqa" -type f -name testdev_balanced_questions.json -print -quit)}"
GQA_IMAGES="${GQA_IMAGES:-$(find "${ROOT}/data/benchmarks/gqa" -type d -name images -print -quit)}"

[[ -f "${RUNNER}" ]] || { echo "Runner not found: ${RUNNER}" >&2; exit 1; }
[[ -f "${VSR_ANN}" ]] || { echo "VSR annotation not found: ${VSR_ANN}" >&2; exit 1; }
[[ -d "${VSR_IMAGES}" ]] || { echo "VSR images not found: ${VSR_IMAGES}" >&2; exit 1; }
[[ -n "${GQA_QUESTIONS}" && -f "${GQA_QUESTIONS}" ]] || { echo "GQA questions not found." >&2; exit 1; }
[[ -n "${GQA_IMAGES}" && -d "${GQA_IMAGES}" ]] || { echo "GQA images not found." >&2; exit 1; }

# Check the environment before committing 48 jobs.
"${PYTHON}" - <<'PY'
import torch, torchvision, transformers
assert torch.__version__.startswith("2.4.1"), torch.__version__
assert torchvision.__version__.startswith("0.19.1"), torchvision.__version__
assert transformers.__version__ == "4.49.0", transformers.__version__
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from torchvision import ops
print("environment OK:", torch.__version__, torchvision.__version__, transformers.__version__)
PY

BACKENDS=(qwen2vl qwen25vl internvl25)
DATASETS=(vsr gqa)
EPSES=(1e-4 1e-5 1e-6 1e-7)
WEIGHTS=(1.0 0.5)
GPUS=(0 1)

run_one() {
  local gpu="$1"
  local backend="$2"
  local dataset="$3"
  local eps="$4"
  local weight="$5"

  local tag="${backend}__${dataset}__eps${eps}__w${weight}"
  tag="${tag//./p}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PYTHON}" "${RUNNER}" \
    --backend "${backend}" \
    --dataset "${dataset}" \
    --rms-norm-eps "${eps}" \
    --scalvis-weight "${weight}" \
    --dtype bfloat16 \
    --vsr-ann "${VSR_ANN}" \
    --vsr-image-root "${VSR_IMAGES}" \
    --gqa-questions "${GQA_QUESTIONS}" \
    --gqa-image-root "${GQA_IMAGES}" \
    --output-dir "${OUT}" \
    --resume \
    > "${LOGS}/${tag}.log" 2>&1
}

pids=()
job_index=0

for dataset in "${DATASETS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    for eps in "${EPSES[@]}"; do
      for weight in "${WEIGHTS[@]}"; do
        gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"

        # At most two simultaneous jobs, one on each GPU.
        if (( ${#pids[@]} >= ${#GPUS[@]} )); then
          for pid in "${pids[@]}"; do
            wait "${pid}"
          done
          pids=()
        fi

        echo "[launch] gpu=${gpu} backend=${backend} dataset=${dataset} eps=${eps} weight=${weight}"
        run_one "${gpu}" "${backend}" "${dataset}" "${eps}" "${weight}" &
        pids+=("$!")
        ((job_index+=1))
      done
    done
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All 48 jobs completed."
