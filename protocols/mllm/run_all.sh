#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export VIDTOUCH_DATASET_ROOT="${VIDTOUCH_DATASET_ROOT:-/root/VBTSINT_DATASET}"
export VIDTOUCH_BENCHMARK_OUTPUT="${VIDTOUCH_BENCHMARK_OUTPUT:-${VIDTOUCH_DATASET_ROOT}/benchmark/protocol_v2/outputs/formal_v1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${VIDTOUCH_BENCHMARK_OUTPUT}/logs"
python "${SCRIPT_DIR}/preflight.py" | tee "${VIDTOUCH_BENCHMARK_OUTPUT}/logs/preflight.json"

MODELS="${MODELS:-llava15 qwen2vl glm4v llama32 gemma3 internvl25}"

run_model() {
  local name="$1"
  local runner="$2"
  local log="${VIDTOUCH_BENCHMARK_OUTPUT}/logs/${name}.log"
  echo "[$(date --iso-8601=seconds)] START ${name}" | tee -a "${log}"
  python -u "${SCRIPT_DIR}/${runner}" 2>&1 | tee -a "${log}"
  echo "[$(date --iso-8601=seconds)] DONE ${name}" | tee -a "${log}"
}

for model in ${MODELS}; do
  case "${model}" in
    llava15)
      run_model "llava15" "run_llava15.py"
      ;;
    qwen2vl)
      run_model "qwen2vl" "run_qwen2vl.py"
      ;;
    glm4v)
      run_model "glm4v" "run_glm4v.py"
      ;;
    llama32)
      if [[ -z "${LLAMA32_ADAPTER_PATH:-}" && -z "${LLAMA32_MODEL_PATH:-}" ]]; then
        echo "WARNING: no MLLM-Fabric checkpoint configured; evaluating base Llama-3.2-11B."
      fi
      run_model "llama32" "run_llama32.py"
      ;;
    gemma3)
      run_model "gemma3" "run_gemma3.py"
      ;;
    internvl25)
      if [[ ! -d "${INTERNVL25_MODEL_PATH:-/nonexistent}" ]]; then
        if [[ "${AUTO_DOWNLOAD_INTERNVL:-1}" == "1" ]]; then
          echo "InternVL2.5 is not persistent on this instance; downloading it now."
          INTERNVL25_MODEL_PATH="$(
            python "${SCRIPT_DIR}/prepare_internvl.py" | tail -n 1
          )"
          export INTERNVL25_MODEL_PATH
        else
          echo "INTERNVL25_MODEL_PATH is missing and AUTO_DOWNLOAD_INTERNVL=0." >&2
          exit 1
        fi
      fi
      run_model "internvl25" "run_internvl25.py"
      ;;
    *)
      echo "Unknown model key: ${model}" >&2
      exit 1
      ;;
  esac
done

python "${SCRIPT_DIR}/summarize_results.py" \
  2>&1 | tee "${VIDTOUCH_BENCHMARK_OUTPUT}/logs/summary.log"
