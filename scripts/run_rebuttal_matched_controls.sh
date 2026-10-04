#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT [DEVICE]" >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_ROOT=$2
DEVICE=${3:-cuda}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$OUTPUT_ROOT"

for variant in simple_matched simple_matched_fpr mesa_fpr_unique; do
  for seed in 7 42 123; do
    run_dir="$OUTPUT_ROOT/${variant}_seed${seed}"
    python -m vidtouch.train \
      --config "$ROOT/configs/rebuttal/${variant}.yaml" \
      --data-root "$DATA_ROOT" \
      --output-dir "$run_dir" \
      --device "$DEVICE" \
      --seed "$seed"
  done
done
