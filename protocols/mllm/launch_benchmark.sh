#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${SESSION_NAME:-vidtouch_mllm_v2}"
MASTER_LOG="${VIDTOUCH_BENCHMARK_OUTPUT:-/root/VBTSINT_DATASET/benchmark/protocol_v2/outputs/formal_v1}/master.log"

if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "Screen session ${SESSION_NAME} is already running."
  exit 1
fi

mkdir -p "$(dirname "${MASTER_LOG}")"
screen -dmS "${SESSION_NAME}" bash -lc \
  "bash '${SCRIPT_DIR}/run_all.sh' > '${MASTER_LOG}' 2>&1"

echo "Started screen session: ${SESSION_NAME}"
echo "Master log: ${MASTER_LOG}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Follow: tail -f '${MASTER_LOG}'"
